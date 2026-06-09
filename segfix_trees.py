#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segfix_trees.py
===============

Streaming, overlap-driven assignment for target-tree point clouds against mask point clouds.

Workflow:
- Build an expanded bounding box for every mask (expanded by `--distance`).
- For each target tree, gather candidate masks using only bbox overlap.
- For each target point, assign it to the closest candidate mask.
- Mark point confidence by distance to that closest mask:
    - `match_flag=1` if distance <= `--distance` (matched)
    - `match_flag=0` if distance > `--distance` (uncertain)

Outputs:
- One matched and one uncertain PLY per mask:
    - `{mask}_matched.ply`
    - `{mask}_uncertain.ply`
- Per-vertex fields: `(x, y, z, current_id, match_flag, src)`
    - `current_id`: integer id of source target file
    - `match_flag`: matched vs uncertain by radius
    - `src`: always `1` for target points

Target files with no overlapping masks are copied through unchanged so points are not lost.

The writer is a single-phase appender (no memmap stitching, no temporary `.npy` chunks),
so I/O is minimal and RAM stays flat. LAS/LAZ reading is chunked via laspy; PLY reading
uses array slicing in chunks.

Requirements:
  pip install numpy scipy plyfile laspy tqdm

Usage:
    python segfix_trees.py \
                /path/to/target_trees \
                -m /path/to/masks \
                -o /path/to/output \
                --distance 1.0 \
                --chunk-size 500000 \
                --workers 1

Notes:
- This script writes **PLY** outputs to support extra per-vertex fields cleanly.
- Set `--workers` to 1 when running many files sequentially; if you avoid multiprocessing,
  you can raise it (SciPy parallelizes inside KD queries). Avoid oversubscribing CPUs.
"""

from __future__ import annotations
import argparse
import csv
import sys
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
import laspy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

SUPPORTED_EXTS = {'.ply', '.las', '.laz'}
RESERVED_OUTPUT_FIELDS = {'x', 'y', 'z', 'current_id', 'match_flag', 'src'}
PROGRESS_KW = {
    'leave': False,
    'mininterval': 0.5,
    'dynamic_ncols': True,
    'disable': not sys.stderr.isatty(),
}


def stage(msg: str) -> None:
    print(f"[segfix] {msg}")


def normalize_cli_path(path: Path, must_exist: bool) -> Path:
    p = path.expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p)
    return p.resolve(strict=must_exist)


def _normalize_passthrough_dtype(src_dtype: np.dtype) -> np.dtype:
    """Map source numeric dtypes to writer-supported output dtypes."""
    dt = np.dtype(src_dtype)
    if dt.kind in ('f',):
        return np.dtype('<f4')
    if dt.kind in ('i', 'u', 'b'):
        return np.dtype('<i4')
    raise TypeError(f'Unsupported passthrough dtype: {dt}')


def infer_passthrough_fields(a_files: List[Path]) -> List[Tuple[str, np.dtype]]:
    """Infer passthrough fields by scanning A files and taking schema union."""
    field_map: Dict[str, np.dtype] = {}

    for path in a_files:
        ext = path.suffix.lower()
        try:
            if ext == '.ply':
                with open(path, 'rb') as f:
                    ply = PlyData.read(f)
                v = ply['vertex']
                names = list(v.data.dtype.names or [])
                for name in names:
                    if name in RESERVED_OUTPUT_FIELDS or name in ('x', 'y', 'z'):
                        continue
                    src_dt = np.dtype(v.data.dtype.fields[name][0])
                    try:
                        out_dt = _normalize_passthrough_dtype(src_dt)
                    except TypeError:
                        continue
                    prev = field_map.get(name)
                    if prev is None or (prev == np.dtype('<i4') and out_dt == np.dtype('<f4')):
                        field_map[name] = out_dt
                continue

            if ext in ('.las', '.laz'):
                with laspy.open(path) as src:
                    dim_names = list(src.point_format.dimension_names)
                    it = src.chunk_iterator(1)
                    first_chunk = next(it, None)
                if first_chunk is None:
                    continue
                for name in dim_names:
                    lname = name.lower()
                    if lname in RESERVED_OUTPUT_FIELDS or lname in ('x', 'y', 'z'):
                        continue
                    if not hasattr(first_chunk, name):
                        continue
                    vals = np.asarray(getattr(first_chunk, name))
                    if vals.size == 0:
                        continue
                    try:
                        out_dt = _normalize_passthrough_dtype(vals.dtype)
                    except TypeError:
                        continue
                    prev = field_map.get(name)
                    if prev is None or (prev == np.dtype('<i4') and out_dt == np.dtype('<f4')):
                        field_map[name] = out_dt
        except Exception as e:
            print(f"Warning: failed to infer passthrough fields from {path}: {e}")
            continue

    return list(field_map.items())

# ---------------- PLY structured appender ----------------
class PlyStructAppender:
    """Append structured vertex records to a temp body; finalize with correct header."""
    def __init__(self, out_path: Path, dtype: np.dtype):
        self.out_path = out_path
        self.tmp_path = out_path.with_suffix(out_path.suffix + '.bin')
        # fresh start
        if self.out_path.exists():
            self.out_path.unlink()
        if self.tmp_path.exists():
            self.tmp_path.unlink()
        self.count = 0
        self.dtype = dtype
        self._f = open(self.tmp_path, 'ab')

    def append(self, recs: np.ndarray):
        if recs is None or recs.size == 0:
            return
        if recs.dtype != self.dtype:
            recs = recs.astype(self.dtype, copy=False)
        # ensure C-contiguous
        recs = np.ascontiguousarray(recs)
        self._f.write(recs.tobytes())
        self.count += recs.shape[0]

    def close(self):
        if not self._f.closed:
            self._f.close()

    def _ply_header(self) -> bytes:
        # Map numpy dtype to PLY property lines
        lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {self.count}",
        ]
        for name, dt in self.dtype.fields.items():
            t = dt[0]
            if t == np.dtype('<f4'):
                lines.append(f"property float {name}")
            elif t == np.dtype('<i4'):
                lines.append(f"property int {name}")
            elif t == np.dtype('<u1'):
                lines.append(f"property uchar {name}")
            else:
                raise TypeError(f"Unsupported dtype for PLY: {name}:{t}")
        lines.append("end_header")
        return ("\n".join(lines) + "\n").encode('ascii')

    def finalize(self):
        self.close()
        header = self._ply_header()
        with open(self.out_path, 'wb') as fout:
            fout.write(header)
        with open(self.out_path, 'ab') as fout, open(self.tmp_path, 'rb') as ftmp:
            for chunk in iter(lambda: ftmp.read(1<<20), b''):
                fout.write(chunk)
        try:
            os.remove(self.tmp_path)
        except OSError:
            pass

    def discard(self):
        """Drop temporary state without producing an output PLY."""
        self.close()
        try:
            os.remove(self.tmp_path)
        except OSError:
            pass
        try:
            os.remove(self.out_path)
        except OSError:
            pass

# ---------------- IO helpers ----------------

def read_xyz(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    try:
        if ext == '.ply':
            with open(path, 'rb') as f:
                ply = PlyData.read(f)
            v = ply['vertex']
            return np.vstack((v['x'], v['y'], v['z'])).T.astype(np.float32, copy=False)
        elif ext in ('.las', '.laz'):
            with laspy.open(path) as src:
                las = src.read()
            return np.vstack((las.x, las.y, las.z)).T.astype(np.float32, copy=False)
        else:
            raise ValueError('Unsupported extension')
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")
        return np.empty((0,3), dtype=np.float32)


def list_point_files(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in SUPPORTED_EXTS and p.is_file()]


def read_ply_vertex_records(path: Path) -> np.ndarray:
    """Read binary/ascii PLY vertex records as a structured numpy array."""
    with open(path, 'rb') as f:
        ply = PlyData.read(f)
    return np.asarray(ply['vertex'].data)

# ---------------- Geometry & trees ----------------
@dataclass
class TreeInfo:
    name: str
    path: Path
    xyz: np.ndarray
    base_xy: np.ndarray
    z0: float
    hag: np.ndarray
    total_height: float = np.nan
    dbh: float = np.nan
    dbh_point_count: int = 0
    align_xyz: Optional[np.ndarray] = None

@dataclass
class MaskInfo(TreeInfo):
    kd: Optional[cKDTree] = None


def estimate_hag(xyz: np.ndarray, percentile: float) -> Tuple[np.ndarray, float]:
    if xyz.size == 0:
        return np.empty((0,), dtype=np.float32), 0.0
    z = xyz[:,2]
    z0 = float(np.quantile(z, percentile))
    return (z - z0).astype(np.float32, copy=False), z0


def compute_total_height(xyz: np.ndarray) -> float:
    if xyz.size == 0:
        return np.nan
    return float(np.max(xyz[:,2]) - np.min(xyz[:,2]))


def compute_dbh(xyz: np.ndarray, hag: np.ndarray,
                breast_height: float, breast_band: float,
                min_pts: int = 12) -> Tuple[float, int, Optional[np.ndarray]]:
    if xyz.size == 0 or hag.size == 0:
        return np.nan, 0, None

    sel = np.abs(hag - breast_height) <= breast_band
    pts = xyz[sel]
    if pts.shape[0] < min_pts:
        return np.nan, int(pts.shape[0]), None

    centroid = np.mean(pts, axis=0).astype(np.float32, copy=False)
    radial = np.linalg.norm(pts[:,:2] - centroid[:2], axis=1)
    if radial.size == 0:
        return np.nan, int(pts.shape[0]), centroid

    dbh = float(2.0 * np.quantile(radial, 0.9))
    return dbh, int(pts.shape[0]), centroid


def compute_alignment_centroid(xyz: np.ndarray, hag: np.ndarray,
                               z0: float,
                               base_low: float, base_high: float,
                               breast_height: float, breast_band: float,
                               fallback_percent: float,
                               min_breast_pts: int = 12,
                               min_base_pts: int = 30) -> np.ndarray:
    dbh, dbh_count, breast_centroid = compute_dbh(
        xyz, hag, breast_height=breast_height, breast_band=breast_band, min_pts=min_breast_pts
    )
    if dbh_count >= min_breast_pts and breast_centroid is not None and np.isfinite(dbh):
        return breast_centroid.astype(np.float32, copy=False)

    base_sel = (hag >= base_low) & (hag <= base_high)
    base_pts = xyz[base_sel]
    if base_pts.shape[0] >= min_base_pts:
        return np.mean(base_pts, axis=0).astype(np.float32, copy=False)

    if xyz.shape[0] == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    z = xyz[:,2]
    thr = np.quantile(z, fallback_percent)
    low_pts = xyz[z <= thr]
    if low_pts.shape[0] == 0:
        low_pts = xyz
    return np.mean(low_pts, axis=0).astype(np.float32, copy=False)


def build_tree_info(name: str,
                    path: Path,
                    xyz: np.ndarray,
                    hag_percentile: float,
                    base_low: float,
                    base_high: float,
                    fallback_percent: float,
                    dbh_height: float,
                    dbh_band_width: float,
                    dbh_min_points: int) -> TreeInfo:
    hag, z0 = estimate_hag(xyz, hag_percentile)
    base_xy = compute_base_xy(xyz, hag, base_low, base_high, fallback_percent=fallback_percent)
    total_height = compute_total_height(xyz)
    dbh, dbh_point_count, _ = compute_dbh(xyz, hag, dbh_height, dbh_band_width, dbh_min_points)
    align_xyz = compute_alignment_centroid(
        xyz, hag, z0, base_low, base_high, dbh_height, dbh_band_width, fallback_percent, dbh_min_points
    )
    return TreeInfo(
        name=name,
        path=path,
        xyz=xyz,
        base_xy=base_xy,
        z0=z0,
        hag=hag,
        total_height=total_height,
        dbh=dbh,
        dbh_point_count=dbh_point_count,
        align_xyz=align_xyz,
    )


def compute_base_xy(xyz: np.ndarray, hag: np.ndarray,
                    base_low: float, base_high: float,
                    fallback_percent: float = 0.05,
                    min_pts: int = 30) -> np.ndarray:
    sel = (hag >= base_low) & (hag <= base_high)
    pts = xyz[sel]
    if pts.shape[0] < min_pts:
        if xyz.shape[0] == 0:
            return np.array([np.nan, np.nan], dtype=np.float32)
        z = xyz[:,2]
        thr = np.quantile(z, fallback_percent)
        pts = xyz[z <= thr]
        if pts.shape[0] == 0:
            pts = xyz
    return np.mean(pts[:,:2], axis=0).astype(np.float32, copy=False)


def build_mask_index(mask_paths: List[Path], hag_percentile: float,
                     base_low: float, base_high: float,
                     fallback_percent: float,
                     dbh_height: float,
                     dbh_band_width: float,
                     dbh_min_points: int,
                     distance: float,
                     verbose: bool = True) -> Tuple[List[MaskInfo], List[Tuple[np.ndarray,np.ndarray]]]:
    mask_list: List[MaskInfo] = []
    bases: List[np.ndarray] = []
    bboxes: List[Tuple[np.ndarray,np.ndarray]] = []
    it = mask_paths
    if verbose:
        it = tqdm(it, desc='Load masks', unit='mask', **PROGRESS_KW)
    for p in it:
        xyz = read_xyz(p)
        tree = build_tree_info(
            name=p.stem,
            path=p,
            xyz=xyz,
            hag_percentile=hag_percentile,
            base_low=base_low,
            base_high=base_high,
            fallback_percent=fallback_percent,
            dbh_height=dbh_height,
            dbh_band_width=dbh_band_width,
            dbh_min_points=dbh_min_points,
        )
        mask = MaskInfo(
            name=tree.name,
            path=tree.path,
            xyz=tree.xyz,
            base_xy=tree.base_xy,
            z0=tree.z0,
            hag=tree.hag,
            total_height=tree.total_height,
            dbh=tree.dbh,
            dbh_point_count=tree.dbh_point_count,
            align_xyz=tree.align_xyz,
        )
        mask_list.append(mask)
        bases.append(mask.base_xy)
        if xyz.size == 0:
            mins = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
            maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
        else:
            mins = xyz.min(axis=0) - distance
            maxs = xyz.max(axis=0) + distance
        bboxes.append((mins.astype(np.float32), maxs.astype(np.float32)))
    if len(mask_list) == 0:
        raise SystemExit('No usable masks found.')
    return mask_list, bboxes


def ensure_mask_kd(mask: MaskInfo) -> cKDTree:
    if mask.kd is None:
        pts = mask.xyz if mask.xyz.size > 0 else np.zeros((1,3), dtype=np.float32)
        mask.kd = cKDTree(pts, leafsize=64, compact_nodes=True, balanced_tree=True)
    return mask.kd


def assign_chunk_to_closest_mask(a_xyz: np.ndarray,
                                 candidates: List[int],
                                 mask_list: List[MaskInfo],
                                 distance: float,
                                 workers: int) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each target point to its closest candidate mask.
    Points within `distance` are matched (1), otherwise uncertain (0)."""
    n = a_xyz.shape[0]
    assigned_mask = np.full(n, -1, dtype=np.int64)
    match_flag = np.zeros(n, dtype=np.int32)
    if n == 0 or not candidates:
        return assigned_mask, match_flag

    best_dist = np.full(n, np.inf, dtype=np.float32)
    best_mask = np.full(n, -1, dtype=np.int64)

    for midx in candidates:
        d, _ = ensure_mask_kd(mask_list[midx]).query(a_xyz, k=1, workers=workers)
        better = d < best_dist
        if np.any(better):
            best_dist[better] = d[better]
            best_mask[better] = midx

    assigned_mask[:] = best_mask
    match_flag[best_dist <= distance] = 1
    match_flag[best_dist > distance] = 0
    return assigned_mask, match_flag


def find_overlapping_bboxes(a_mins: np.ndarray, a_maxs: np.ndarray,
                            bboxes: List[Tuple[np.ndarray, np.ndarray]]) -> List[int]:
    """Return B indices whose bbox intersects the A bbox."""
    overlaps: List[int] = []
    for bidx, (b_mins, b_maxs) in enumerate(bboxes):
        if np.all(a_maxs >= b_mins) and np.all(a_mins <= b_maxs):
            overlaps.append(bidx)
    return overlaps

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description='Streaming target-tree to mask assignment using overlap-only mask candidates.')
    ap.add_argument('input_folder', type=Path, help='Folder of target tree files (.ply/.las/.laz)')
    ap.add_argument('-m', '--mask_dir', type=Path, required=True, help='Folder of mask files (.ply/.las/.laz)')
    ap.add_argument('-o', '--output', type=Path, required=True, help='Output folder for per-mask PLY files')
    ap.add_argument('-stf', '--save-target-filename', dest='save_target_filename', action='store_true',
                    help='Rename saved outputs to majority contributing target filename')

    ap.add_argument('--hag-percentile', type=float, default=0.02, help='Percentile of z as ground (default 0.02)')
    ap.add_argument('--base-slice', type=float, nargs=2, default=[0.10, 0.40], metavar=('LOW', 'HIGH'), help='HAG slice used for summary metrics')
    ap.add_argument('--fallback-percent', type=float, default=0.05, help='Fallback lowest proportion when base slice is sparse')
    ap.add_argument('--dbh-height', type=float, default=1.3, help='Height above ground for DBH summary metric (m)')
    ap.add_argument('--dbh-band-width', type=float, default=0.2, help='Half-width of DBH sampling band (m)')
    ap.add_argument('--dbh-min-points', type=int, default=12, help='Minimum points required in DBH band')

    ap.add_argument('-d', '--distance', type=float, default=0.5, help='Match radius (m): points inside are matched, outside are uncertain')
    ap.add_argument('--chunk-size', type=int, default=500000, help='Points per chunk for target files')
    ap.add_argument('--workers', type=int, default=None, help='Threads for processing target files in parallel (default: physical CPUs)')
    ap.add_argument('--threads', type=int, default=None, help='Threads for KD queries (default: logical CPUs / workers)')

    ap.add_argument('--report', type=Path, default=None, help='Optional CSV report path (default: output/merge_report.csv)')

    args = ap.parse_args()

    args.input_folder = normalize_cli_path(args.input_folder, must_exist=True)
    args.mask_dir = normalize_cli_path(args.mask_dir, must_exist=True)
    args.output = normalize_cli_path(args.output, must_exist=False)
    if args.report is not None:
        args.report = normalize_cli_path(args.report, must_exist=False)

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.workers is None:
        import psutil
        args.workers = psutil.cpu_count(logical=False)
    if args.threads is None:
        import psutil
        args.threads = max(1, psutil.cpu_count(logical=True) // max(1, args.workers))
    stage(f"Using {args.workers} worker threads for target files, {args.threads} threads for KD queries")

    target_files = list_point_files(args.input_folder)
    mask_files = list_point_files(args.mask_dir)
    if not target_files:
        raise SystemExit('No target tree files found')
    if not mask_files:
        raise SystemExit('No mask files found')

    base_low, base_high = args.base_slice

    mask_list, bboxes = build_mask_index(
        mask_files,
        args.hag_percentile,
        base_low,
        base_high,
        args.fallback_percent,
        args.dbh_height,
        args.dbh_band_width,
        args.dbh_min_points,
        args.distance,
        verbose=True,
    )

    passthrough_fields = infer_passthrough_fields(target_files)
    if passthrough_fields:
        stage('Passthrough fields: ' + ', '.join(name for name, _ in passthrough_fields))
    else:
        stage('Passthrough fields: none')

    out_dtype_fields: List[Tuple[str, str]] = [
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('current_id', '<i4'), ('match_flag', '<i4'), ('src', '<i4')
    ]
    out_dtype_fields.extend((name, str(dt)) for name, dt in passthrough_fields)
    out_dtype = np.dtype(out_dtype_fields)

    appenders_matched: List[PlyStructAppender] = []
    appenders_uncertain: List[PlyStructAppender] = []
    counts_matched = np.zeros((len(mask_list),), dtype=np.int64)
    counts_uncertain = np.zeros((len(mask_list),), dtype=np.int64)
    for midx, mask in enumerate(mask_list):
        app_m = PlyStructAppender(out_dir / f"{mask.name}_matched.ply", out_dtype)
        app_u = PlyStructAppender(out_dir / f"{mask.name}_uncertain.ply", out_dtype)
        appenders_matched.append(app_m)
        appenders_uncertain.append(app_u)

    target_id_map: Dict[int, str] = {}
    for idx, target_path in enumerate(target_files, start=1):
        target_id_map[idx] = target_path.name

    stage(f'Processing {len(target_files)} target trees')
    passthrough_copied = 0
    link_rows: List[List[object]] = []

    def process_target_file(target_path: Path):
        ext = target_path.suffix.lower()
        target_id = next(k for k, v in target_id_map.items() if v == target_path.name)
        passthrough_names = [name for name, _ in passthrough_fields]

        def convert_passthrough_chunk(chunk_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            out: Dict[str, np.ndarray] = {}
            for name, out_dt in passthrough_fields:
                vals = chunk_data.get(name)
                if vals is None:
                    continue
                out[name] = np.asarray(vals).astype(out_dt, copy=False)
            return out

        if ext in ('.las', '.laz'):
            with laspy.open(target_path) as src:
                n = min(int(src.header.point_count), 1000000)
                it = src.chunk_iterator(n)
                first = next(it)
                target_xyz_sample = np.vstack((first.x, first.y, first.z)).T.astype(np.float32, copy=False)
                target_bbox_mins = np.asarray(src.header.mins, dtype=np.float32)
                target_bbox_maxs = np.asarray(src.header.maxs, dtype=np.float32)
        elif ext == '.ply':
            with open(target_path, 'rb') as f:
                ply = PlyData.read(f)
            v = ply['vertex']
            target_xyz_sample = np.vstack((v['x'], v['y'], v['z'])).T.astype(np.float32, copy=False)
            if target_xyz_sample.shape[0] > 0:
                target_bbox_mins = np.min(target_xyz_sample, axis=0)
                target_bbox_maxs = np.max(target_xyz_sample, axis=0)
            else:
                target_bbox_mins = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
                target_bbox_maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
        else:
            return {'mode': 'skip', 'target_id': target_id, 'path': target_path}

        target_tree = build_tree_info(
            name=target_path.stem,
            path=target_path,
            xyz=target_xyz_sample,
            hag_percentile=args.hag_percentile,
            base_low=base_low,
            base_high=base_high,
            fallback_percent=args.fallback_percent,
            dbh_height=args.dbh_height,
            dbh_band_width=args.dbh_band_width,
            dbh_min_points=args.dbh_min_points,
        )

        candidate_masks = find_overlapping_bboxes(target_bbox_mins, target_bbox_maxs, bboxes)
        if not candidate_masks:
            return {
                'mode': 'copy_passthrough',
                'target_id': target_id,
                'path': target_path,
                'link_row': [
                    target_path.name,
                    '',
                    0,
                    0,
                    0,
                    '',
                    float(target_tree.dbh) if np.isfinite(target_tree.dbh) else '',
                    float(target_tree.total_height) if np.isfinite(target_tree.total_height) else '',
                    'no_overlapping_masks',
                ],
            }

        results = [[] for _ in range(len(mask_list))]
        matched_counts: Dict[int, int] = {midx: 0 for midx in candidate_masks}
        assigned_counts: Dict[int, int] = {midx: 0 for midx in candidate_masks}
        total_matched = 0
        total_uncertain = 0

        if ext in ('.las', '.laz'):
            with laspy.open(target_path) as src:
                for chunk in src.chunk_iterator(args.chunk_size):
                    chunk_xyz = np.vstack((chunk.x, chunk.y, chunk.z)).T.astype(np.float32, copy=False)
                    passthrough_chunk = convert_passthrough_chunk({
                        name: np.asarray(getattr(chunk, name)) for name in passthrough_names if hasattr(chunk, name)
                    })
                    assigned_mask, match_flag = assign_chunk_to_closest_mask(
                        chunk_xyz, candidate_masks, mask_list, args.distance, args.threads
                    )
                    total_matched += int((match_flag == 1).sum())
                    total_uncertain += int((match_flag == 0).sum())
                    for midx in np.unique(assigned_mask[assigned_mask >= 0]):
                        sel = (assigned_mask == midx)
                        assigned_counts[int(midx)] = assigned_counts.get(int(midx), 0) + int(sel.sum())
                        msel = sel & (match_flag == 1)
                        if np.any(msel):
                            matched_counts[int(midx)] = matched_counts.get(int(midx), 0) + int(msel.sum())
                        results[midx].append((
                            chunk_xyz[sel],
                            match_flag[sel],
                            {name: vals[sel] for name, vals in passthrough_chunk.items()}
                        ))
        else:
            with open(target_path, 'rb') as f:
                v = PlyData.read(f)['vertex']
            pts = np.vstack((v['x'], v['y'], v['z'])).T.astype(np.float32, copy=False)
            for s in range(0, pts.shape[0], args.chunk_size):
                e = min(s + args.chunk_size, pts.shape[0])
                chunk_xyz = pts[s:e]
                passthrough_chunk = convert_passthrough_chunk({
                    name: np.asarray(v[name][s:e]) for name in passthrough_names if name in v.data.dtype.names
                })
                assigned_mask, match_flag = assign_chunk_to_closest_mask(
                    chunk_xyz, candidate_masks, mask_list, args.distance, args.threads
                )
                total_matched += int((match_flag == 1).sum())
                total_uncertain += int((match_flag == 0).sum())
                for midx in np.unique(assigned_mask[assigned_mask >= 0]):
                    sel = (assigned_mask == midx)
                    assigned_counts[int(midx)] = assigned_counts.get(int(midx), 0) + int(sel.sum())
                    msel = sel & (match_flag == 1)
                    if np.any(msel):
                        matched_counts[int(midx)] = matched_counts.get(int(midx), 0) + int(msel.sum())
                    results[midx].append((
                        chunk_xyz[sel],
                        match_flag[sel],
                        {name: vals[sel] for name, vals in passthrough_chunk.items()}
                    ))

        if total_matched == 0 and total_uncertain == 0:
            return {
                'mode': 'copy_passthrough',
                'target_id': target_id,
                'path': target_path,
                'link_row': [
                    target_path.name,
                    '',
                    len(candidate_masks),
                    0,
                    0,
                    '',
                    float(target_tree.dbh) if np.isfinite(target_tree.dbh) else '',
                    float(target_tree.total_height) if np.isfinite(target_tree.total_height) else '',
                    'empty_target',
                ],
            }

        selected_mask_idx = max(candidate_masks, key=lambda midx: (assigned_counts.get(midx, 0), matched_counts.get(midx, 0), -midx))
        matched_summary = ';'.join(
            f"{mask_list[midx].name}:{matched_counts.get(midx, 0)}"
            for midx in sorted(candidate_masks, key=lambda i: (-matched_counts.get(i, 0), i))
        )
        link_row = [
            target_path.name,
            mask_list[selected_mask_idx].name,
            len(candidate_masks),
            total_matched,
            total_uncertain,
            matched_summary,
            float(target_tree.dbh) if np.isfinite(target_tree.dbh) else '',
            float(target_tree.total_height) if np.isfinite(target_tree.total_height) else '',
            'assigned',
        ]
        return {'mode': 'assigned', 'target_id': target_id, 'results': results, 'link_row': link_row}

    current_ids_per_mask: List[Dict[int, int]] = [{} for _ in range(len(mask_list))]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_target_file, p): p for p in target_files}
        for future in tqdm(
            as_completed(futures),
            total=len(target_files),
            desc='Assign trees',
            unit='tree',
            **PROGRESS_KW,
        ):
            result = future.result()
            if result is None:
                continue

            mode = result.get('mode')
            if mode == 'copy_passthrough':
                src_path = result['path']
                dst_path = out_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                passthrough_copied += 1
                link_row = result.get('link_row')
                if link_row is not None:
                    link_rows.append(link_row)
                continue

            if mode != 'assigned':
                continue

            target_id = result.get('target_id')
            results = result.get('results')
            link_row = result.get('link_row')
            if target_id is None or results is None:
                continue
            if link_row is not None:
                link_rows.append(link_row)
            for midx, chunks in enumerate(results):
                if len(chunks) > 0 and target_id not in current_ids_per_mask[midx]:
                    current_ids_per_mask[midx][target_id] = 0
                for pts, mflags, passthrough_chunk in chunks:
                    if pts.shape[0] == 0:
                        continue
                    recs = np.zeros(pts.shape[0], dtype=out_dtype)
                    recs['x'], recs['y'], recs['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
                    recs['current_id'][:] = target_id
                    recs['match_flag'][:] = mflags
                    recs['src'][:] = 1
                    for name, vals in passthrough_chunk.items():
                        recs[name] = vals

                    matched_sel = (mflags == 1)
                    uncertain_sel = (mflags == 0)
                    if np.any(matched_sel):
                        appenders_matched[midx].append(recs[matched_sel])
                        counts_matched[midx] += int(matched_sel.sum())
                    if np.any(uncertain_sel):
                        appenders_uncertain[midx].append(recs[uncertain_sel])
                        counts_uncertain[midx] += int(uncertain_sel.sum())
                    current_ids_per_mask[midx][target_id] += pts.shape[0]

    for app in appenders_matched:
        if app.count > 0:
            app.finalize()
        else:
            app.discard()
    for app in appenders_uncertain:
        if app.count > 0:
            app.finalize()
        else:
            app.discard()

    with open(out_dir / 'current_id_map.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['target_id', 'target_filename'])
        for target_id, name in target_id_map.items():
            w.writerow([target_id, name])

    with open(out_dir / 'link_report.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'target_filename', 'selected_mask_name', 'overlapping_mask_count',
            'matched_points_total', 'uncertain_points_total', 'matched_points_per_mask',
            'target_dbh', 'target_height', 'status'
        ])
        w.writerows(link_rows)

    report_path = args.report if args.report is not None else (out_dir / 'merge_report.csv')
    with open(report_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mask_name', 'points_matched', 'points_uncertain', 'target_ids'])
        for midx, mask in enumerate(mask_list):
            sorted_ids = sorted(current_ids_per_mask[midx].items(), key=lambda x: x[1], reverse=True)
            id_list = ','.join(f"{target_id}({count})" for target_id, count in sorted_ids)
            w.writerow([mask.name, int(counts_matched[midx]), int(counts_uncertain[midx]), id_list])

    if args.save_target_filename:
        majority_groups: Dict[str, List[int]] = {}
        for midx, _mask in enumerate(mask_list):
            if len(current_ids_per_mask[midx]) == 0:
                continue
            majority_target_id = max(current_ids_per_mask[midx].items(), key=lambda x: x[1])[0]
            majority_name = target_id_map.get(majority_target_id, f"target_{majority_target_id}")
            majority_name = majority_name.replace('.ply', '').replace('.las', '').replace('.laz', '')
            majority_groups.setdefault(majority_name, []).append(midx)

        for majority_name, midx_group in majority_groups.items():
            for suffix in ['_matched', '_uncertain']:
                src_paths = [
                    out_dir / f"{mask_list[midx].name}{suffix}.ply"
                    for midx in midx_group
                    if (out_dir / f"{mask_list[midx].name}{suffix}.ply").exists()
                ]
                if len(src_paths) == 0:
                    continue

                dst_path = out_dir / f"{majority_name}{suffix}.ply"
                if len(src_paths) == 1:
                    src = src_paths[0]
                    if src != dst_path:
                        src.rename(dst_path)
                    continue

                merged_tmp_path = out_dir / f"{majority_name}{suffix}.merge_tmp.ply"
                merger = PlyStructAppender(merged_tmp_path, out_dtype)
                for src in src_paths:
                    recs = read_ply_vertex_records(src)
                    if recs.size > 0:
                        merger.append(recs)
                if merger.count > 0:
                    merger.finalize()
                    if dst_path.exists() and dst_path != merged_tmp_path:
                        dst_path.unlink()
                    merged_tmp_path.rename(dst_path)
                else:
                    merger.discard()

                for src in src_paths:
                    if src.exists() and src != dst_path:
                        src.unlink()

        stage('Renamed/merged output files based on majority contributing target_id')

    stage(f"Done. Wrote per-mask outputs to {out_dir}")
    if passthrough_copied > 0:
        stage(f"Copied {passthrough_copied} target tree file(s) through to output")
    stage(f"Target-ID map: {out_dir / 'current_id_map.csv'}")
    stage(f"Link report: {out_dir / 'link_report.csv'}")
    stage(f"Report: {report_path}")

if __name__ == '__main__':
    main()
