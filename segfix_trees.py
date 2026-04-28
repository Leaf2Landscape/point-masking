#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seg_mask_from_existing_stream.py
================================

Streaming, bbox-filtered, single-phase writer that merges **existing (B)** tree masks
with **current (A)** points using the *fast* processing logic you provided:

- **Match A→B by base proximity** (HAG slice). For each A tree select a **primary B**
  (nearest base). Also consider a small set of nearby B candidates (`--k-alt`) for
  possible reassignment.
- For each A point chunk:
  1) **Primary-first**: try NN to the primary B cloud using `distance_upper_bound`.
  2) For points that failed primary, try **alternative B candidates** and assign to the
     closest that satisfies the threshold; otherwise, fall back to the primary.
  3) Use **expanded B bboxes** to cull NN queries (fast reject), just like your script.
- **Outputs (one per B, split by confidence)**: we **add B's own points** *and* the assigned A points into
   separate PLY streams per confidence tier:
   - `{tree}_matched.ply` — match_flag=1 (high confidence, near B)
   - `{tree}_uncertain.ply` — match_flag=0 (fallback to primary, far from any B)
   Per-vertex fields: `(x, y, z, current_id, match_flag, src)`:
    - `current_id`: integer id of the source A file
    - `match_flag`: 1 if A∩B (matched to some candidate); 0 if A-only (fallback to primary)
    - `src`: 1 for A points (B core points are in separate handling)

This creates high-confidence matched outputs (tier 1) separated from fallback-uncertain 
outputs (tier 2) for staged manual review. Unassigned current files are copied through unchanged.

The writer is a **single-phase appender** (no memmap stitching, no temporary .npy chunks),
so I/O is minimal and RAM stays flat. LAS/LAZ reading is chunked via laspy; PLY/A reading
uses array slicing in chunks.

Requirements:
  pip install numpy scipy plyfile laspy tqdm

Usage:
  python seg_mask_from_existing_stream.py \
        /path/to/trees_to_correct \
        -m /path/to/correct_masks \
    -o /path/to/output \
    --gate 1.5 \
    --base-slice 0.10 0.40 \
    --hag-percentile 0.02 \
    --distance 1.0 \
    --k-alt 5 \
    --chunk-size 500000 \
    --workers 1

Notes:
- This script writes **PLY** outputs to support the extra per-vertex fields cleanly.
- Set `--workers` to 1 when running many A files sequentially; if you avoid multiprocessing,
  you can bump it (SciPy parallelizes inside KD queries). Avoid mixing both.
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

@dataclass
class RefTree(TreeInfo):
    kd: Optional[cKDTree] = None


def estimate_hag(xyz: np.ndarray, percentile: float) -> Tuple[np.ndarray, float]:
    if xyz.size == 0:
        return np.empty((0,), dtype=np.float32), 0.0
    z = xyz[:,2]
    z0 = float(np.quantile(z, percentile))
    return (z - z0).astype(np.float32, copy=False), z0


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


def build_ref_index(ref_paths: List[Path], hag_percentile: float,
                    base_low: float, base_high: float,
                    fallback_percent: float,
                    distance: float,
                    verbose: bool = True) -> Tuple[List[RefTree], cKDTree, List[Tuple[np.ndarray,np.ndarray]]]:
    ref_list: List[RefTree] = []
    bases: List[np.ndarray] = []
    bboxes: List[Tuple[np.ndarray,np.ndarray]] = []
    it = ref_paths
    if verbose:
        it = tqdm(it, desc='Loading existing (B) trees', unit='tree')
    for p in it:
        xyz = read_xyz(p)
        hag, z0 = estimate_hag(xyz, hag_percentile)
        base_xy = compute_base_xy(xyz, hag, base_low, base_high, fallback_percent=fallback_percent)
        ref = RefTree(name=p.stem, path=p, xyz=xyz, base_xy=base_xy, z0=z0, hag=hag)
        ref_list.append(ref)
        bases.append(base_xy)
        if xyz.size == 0:
            mins = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
            maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
        else:
            mins = xyz.min(axis=0) - distance
            maxs = xyz.max(axis=0) + distance
        bboxes.append((mins.astype(np.float32), maxs.astype(np.float32)))
    if len(ref_list) == 0:
        raise SystemExit('No usable existing (B) references.')
    bases_arr = np.asarray(bases, dtype=np.float32)
    kd = cKDTree(bases_arr)
    return ref_list, kd, bboxes


def ensure_ref_kd(ref: RefTree) -> cKDTree:
    if ref.kd is None:
        pts = ref.xyz if ref.xyz.size > 0 else np.zeros((1,3), dtype=np.float32)
        ref.kd = cKDTree(pts, leafsize=64, compact_nodes=True, balanced_tree=True)
    return ref.kd

# ---------------- Assignment with bbox + primary-first ----------------

def choose_primary_and_candidates(a_base: np.ndarray, ref_base_kd: cKDTree, total_b: int,
                                  gate: float, k_alt: int,
                                  primary_max_distance: float = 0.2) -> Tuple[int, List[int]]:
    d, primary_idx = ref_base_kd.query(a_base, k=1)
    if d > primary_max_distance:
        primary_idx = None
    else:
        primary_idx = int(primary_idx)

    within = ref_base_kd.query_ball_point(a_base, r=gate) if gate > 0 else []
    if len(within) == 0:
        k = total_b if (k_alt is not None and k_alt <= 0) else min(max(1,k_alt), total_b)
        _, knn = ref_base_kd.query(a_base, k=k)
        if np.isscalar(knn):
            cands = [int(knn)]
        else:
            cands = [int(i) for i in np.atleast_1d(knn).tolist()]
    else:
        cands = [int(i) for i in within]
    if primary_idx not in cands and primary_idx is not None:
        cands = [int(primary_idx)] + cands
    elif primary_idx is None:
        pass
    else:
        cands = [int(primary_idx)] + [i for i in cands if i != int(primary_idx)]

    return primary_idx, cands


def assign_chunk_primary_then_alts(a_xyz: np.ndarray, primary_idx: int, candidates: List[int],
                                   ref_list: List[RefTree], bboxes: List[Tuple[np.ndarray,np.ndarray]],
                                   distance: float, workers: int) -> Tuple[np.ndarray, np.ndarray]:
    """Assign a chunk of A points: try primary first (with bbox+upper_bound), then alternatives.
    Returns (assigned_b, match_flag)."""
    n = a_xyz.shape[0]
    assigned_b = np.full(n, primary_idx, dtype=np.int64)  # default fallback: primary
    match_flag = np.zeros(n, dtype=np.int32)

    # Try all candidates and assign each point to the closest
    best_dist = np.full(n, np.inf, dtype=np.float32)
    best_b = np.full(n, primary_idx, dtype=np.int64)

    for bidx in candidates:
        mins, maxs = bboxes[bidx]
        in_box = np.all((a_xyz >= mins) & (a_xyz <= maxs), axis=1)
        if not np.any(in_box):
            continue
        sub_idx = np.where(in_box)[0]
        sub_pts = a_xyz[sub_idx]
        d, _ = ensure_ref_kd(ref_list[bidx]).query(sub_pts, k=1, distance_upper_bound=distance, workers=workers)
        ok = (d != np.inf)
        if not np.any(ok):
            continue
        ok_idx = sub_idx[ok]
        dists = d[ok]
        better = dists < best_dist[ok_idx]
        if np.any(better):
            take = ok_idx[better]
            best_dist[take] = dists[better]
            best_b[take] = bidx

    matched = best_dist != np.inf
    assigned_b[:] = best_b
    match_flag[matched] = 1
    match_flag[~matched] = 0

    return assigned_b, match_flag

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description='Streaming A→B merge that writes one PLY per B with B core + assigned A points.')
    ap.add_argument('input_folder', type=Path, help='Folder of trees to correct / mask (.ply/.las/.laz)')
    ap.add_argument('-m', '--mask_dir', type=Path, required=True, help='Folder of correct masks to use (.ply/.las/.laz)')
    ap.add_argument('-o', '--output', type=Path, required=True, help='Output folder for per-B PLY files')
    ap.add_argument('-scf', '--save-current-filename', action='store_true', help='Rename the saved output files to the majority contributing current_id filename')

    # HAG & base
    ap.add_argument('--hag-percentile', type=float, default=0.02, help='Percentile of z as ground (default 0.02)')
    ap.add_argument('--base-slice', type=float, nargs=2, default=[0.10, 0.40], metavar=('LOW','HIGH'), help='HAG slice for base centroid')
    ap.add_argument('--fallback-percent', type=float, default=0.05, help='Fallback lowest proportion when base slice sparse')

    # Matching
    ap.add_argument('--gate', type=float, default=1.5, help='Base-XY gating radius for B candidates')
    ap.add_argument('--k-alt', type=int, default=5, help='If no within-gate, use k nearest Bs (<=0 → all)')
    ap.add_argument('--primary-max-distance', type=float, default=0.2,
                    help='Maximum base distance (m) allowed for selecting a primary mask (default 0.2)')

    # Distance & performance
    ap.add_argument('-d', '--distance', type=float, default=0.5, help='NN distance threshold (m) for a good B match')
    ap.add_argument('--chunk-size', type=int, default=500000, help='Points per chunk for A')
    ap.add_argument('--workers', type=int, default=None, help='Threads for processing A files in parallel (default 1, set >1 to parallelize across A files). Default physical cpus')
    ap.add_argument('--threads', type=int, default=None, help='Threads for KD queries inside SciPy (pick 1 if you parallelize elsewhere). Default logical cores - physical cpus')

    # Report
    ap.add_argument('--report', type=Path, default=None, help='Optional CSV report path (default: output/merge_report.csv)')

    args = ap.parse_args()

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set workers
    if args.workers is None:
        import psutil
        args.workers = psutil.cpu_count(logical=False)
    if args.threads is None:
        import psutil
        args.threads = psutil.cpu_count(logical=True) // args.workers
    else:
        args.workers = psutil.cpu_count(logical=True) // args.threads   
    print(f"Using {args.workers} worker threads for A files, {args.threads} threads for KD queries.")

    a_files = list_point_files(args.input_folder)
    b_files = list_point_files(args.mask_dir)
    if not a_files:
        raise SystemExit('No current (A) files found')
    if not b_files:
        raise SystemExit('No existing (B) files found')

    base_low, base_high = args.base_slice

    # Build B index + bboxes
    ref_list, ref_base_kd, bboxes = build_ref_index(
        b_files, args.hag_percentile, base_low, base_high, args.fallback_percent, args.distance, verbose=True
    )

    passthrough_fields = infer_passthrough_fields(a_files)
    if passthrough_fields:
        print('Passthrough fields:', ', '.join(name for name, _ in passthrough_fields))
    else:
        print('Passthrough fields: none')

    # Structured dtype for output
    out_dtype_fields: List[Tuple[str, str]] = [
        ('x','<f4'), ('y','<f4'), ('z','<f4'),
        ('current_id','<i4'), ('match_flag','<i4'), ('src','<i4')
    ]
    out_dtype_fields.extend((name, str(dt)) for name, dt in passthrough_fields)
    out_dtype = np.dtype(out_dtype_fields)

    # Two appenders per B: matched and uncertain tiers
    appenders_matched: List[PlyStructAppender] = []
    appenders_uncertain: List[PlyStructAppender] = []
    counts_matched = np.zeros((len(ref_list),), dtype=np.int64)
    counts_uncertain = np.zeros((len(ref_list),), dtype=np.int64)
    for bidx, ref in enumerate(ref_list):
        app_m = PlyStructAppender(out_dir / f"{ref.name}_matched.ply", out_dtype)
        app_u = PlyStructAppender(out_dir / f"{ref.name}_uncertain.ply", out_dtype)
        appenders_matched.append(app_m)
        appenders_uncertain.append(app_u)

    # current_id to filename mapping
    rct_id_map: Dict[int, str] = {}
    for idx, a_path in enumerate(a_files, start=1):
        rct_id_map[idx] = a_path.name

    print(f'Processing {len(a_files)} current (A) masks ...')
    passthrough_copied = 0
    
    def process_a_file(a_path):
        """Process a single A file and return results to accumulate."""
        ext = a_path.suffix.lower()
        cid = next(k for k, v in rct_id_map.items() if v == a_path.name)

        passthrough_names = [name for name, _ in passthrough_fields]

        def convert_passthrough_chunk(chunk_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            out: Dict[str, np.ndarray] = {}
            for name, out_dt in passthrough_fields:
                vals = chunk_data.get(name)
                if vals is None:
                    continue
                out[name] = np.asarray(vals).astype(out_dt, copy=False)
            return out
        
        # Load base for this A
        if ext in ('.las','.laz'):
            with laspy.open(a_path) as src:
                n = min(int(src.header.point_count), 1000000)
                it = src.chunk_iterator(n)
                first = next(it)
                a_xyz_sample = np.vstack((first.x, first.y, first.z)).T.astype(np.float32, copy=False)
        elif ext == '.ply':
            with open(a_path, 'rb') as f:
                ply = PlyData.read(f)
            v = ply['vertex']
            a_xyz_sample = np.vstack((v['x'], v['y'], v['z'])).T.astype(np.float32, copy=False)
        else:
            return {'mode': 'skip', 'cid': cid, 'path': a_path}
        
        a_hag, _ = estimate_hag(a_xyz_sample, args.hag_percentile)
        a_base = compute_base_xy(a_xyz_sample, a_hag, base_low, base_high, fallback_percent=args.fallback_percent)
        if not np.isfinite(a_base).all():
            a_base = np.mean(a_xyz_sample[:,:2], axis=0)
        primary_idx, candidates = choose_primary_and_candidates(
            a_base, ref_base_kd, len(ref_list), args.gate, args.k_alt, args.primary_max_distance)
        
        if primary_idx is None:
            # No confident primary B for this A: pass file through unchanged.
            return {'mode': 'copy_passthrough', 'cid': cid, 'path': a_path}

        # Collect results: (bidx, pts, match_flag)
        results = [[] for _ in range(len(ref_list))]
        
        if ext in ('.las','.laz'):
            with laspy.open(a_path) as src:
                for chunk in src.chunk_iterator(args.chunk_size):
                    chunk_xyz = np.vstack((chunk.x, chunk.y, chunk.z)).T.astype(np.float32, copy=False)
                    passthrough_chunk = convert_passthrough_chunk({
                        name: np.asarray(getattr(chunk, name)) for name in passthrough_names if hasattr(chunk, name)
                    })
                    assigned_b, match_flag = assign_chunk_primary_then_alts(
                        chunk_xyz, primary_idx, candidates, ref_list, bboxes, args.distance, args.threads)
                    for bidx in np.unique(assigned_b):
                        sel = (assigned_b == bidx)
                        results[bidx].append((
                            chunk_xyz[sel],
                            match_flag[sel],
                            {name: vals[sel] for name, vals in passthrough_chunk.items()}
                        ))
        else:  # PLY
            with open(a_path, 'rb') as f:
                v = PlyData.read(f)['vertex']
            pts = np.vstack((v['x'], v['y'], v['z'])).T.astype(np.float32, copy=False)
            for s in range(0, pts.shape[0], args.chunk_size):
                e = min(s + args.chunk_size, pts.shape[0])
                chunk_xyz = pts[s:e]
                passthrough_chunk = convert_passthrough_chunk({
                    name: np.asarray(v[name][s:e]) for name in passthrough_names if name in v.data.dtype.names
                })
                assigned_b, match_flag = assign_chunk_primary_then_alts(
                    chunk_xyz, primary_idx, candidates, ref_list, bboxes, args.distance, args.threads)
                for bidx in np.unique(assigned_b):
                    sel = (assigned_b == bidx)
                    results[bidx].append((
                        chunk_xyz[sel],
                        match_flag[sel],
                        {name: vals[sel] for name, vals in passthrough_chunk.items()}
                    ))
        
        return {'mode': 'assigned', 'cid': cid, 'results': results}
    
    # Track current_ids per existing mask with point counts per current_id
    current_ids_per_b: List[Dict[int, int]] = [{} for _ in range(len(ref_list))]
    
    # Process A files in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_a_file, p): p for p in a_files}
        for future in tqdm(as_completed(futures), total=len(a_files), desc='Assigning A→B', unit='mask'):
            result = future.result()
            if result is None:
                continue

            mode = result.get('mode')
            if mode == 'copy_passthrough':
                src_path = result['path']
                dst_path = out_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                passthrough_copied += 1
                continue

            if mode != 'assigned':
                continue

            cid = result.get('cid')
            results = result.get('results')
            if cid is None or results is None:
                continue
            for bidx, chunks in enumerate(results):
                if len(chunks) > 0:
                    if cid not in current_ids_per_b[bidx]:
                        current_ids_per_b[bidx][cid] = 0
                for pts, mflags, passthrough_chunk in chunks:
                    if pts.shape[0] == 0:
                        continue
                    recs = np.zeros(pts.shape[0], dtype=out_dtype)
                    recs['x'], recs['y'], recs['z'] = pts[:,0], pts[:,1], pts[:,2]
                    recs['current_id'][:] = cid
                    recs['match_flag'][:] = mflags
                    recs['src'][:] = 1
                    for name, vals in passthrough_chunk.items():
                        recs[name] = vals
                    
                    # Split by confidence tier
                    matched_sel = (mflags == 1)
                    uncertain_sel = (mflags == 0)
                    
                    if np.any(matched_sel):
                        appenders_matched[bidx].append(recs[matched_sel])
                        counts_matched[bidx] += int(matched_sel.sum())
                    if np.any(uncertain_sel):
                        appenders_uncertain[bidx].append(recs[uncertain_sel])
                        counts_uncertain[bidx] += int(uncertain_sel.sum())
                    
                    current_ids_per_b[bidx][cid] += pts.shape[0]

    # Finalize all B outputs (skip empty tiers to avoid empty files)
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

    # current_id_map.csv
    with open(out_dir / 'current_id_map.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['current_id','current_filename'])
        for cid, name in rct_id_map.items():
            w.writerow([cid, name])

    # Report
    report_path = args.report if args.report is not None else (out_dir / 'merge_report.csv')
    with open(report_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['existing_name', 'points_matched', 'points_uncertain', 'current_ids'])
        for bidx, ref in enumerate(ref_list):
            # Sort current_ids by contribution (descending)
            sorted_cids = sorted(current_ids_per_b[bidx].items(), key=lambda x: x[1], reverse=True)
            cid_list = ','.join(f"{cid}({count})" for cid, count in sorted_cids)
            w.writerow([ref.name, int(counts_matched[bidx]), int(counts_uncertain[bidx]), cid_list])

    # If --save-current-filename, merge duplicate-majority outputs and rename by majority current filename
    if args.save_current_filename:
        majority_groups: Dict[str, List[int]] = {}
        for bidx, _ref in enumerate(ref_list):
            if len(current_ids_per_b[bidx]) == 0:
                continue
            majority_cid = max(current_ids_per_b[bidx].items(), key=lambda x: x[1])[0]
            majority_name = rct_id_map.get(majority_cid, f"current_{majority_cid}")
            majority_name = majority_name.replace('.ply', '').replace('.las', '').replace('.laz', '')
            majority_groups.setdefault(majority_name, []).append(bidx)

        for majority_name, bidx_group in majority_groups.items():
            for suffix in ['_matched', '_uncertain']:
                src_paths = [
                    out_dir / f"{ref_list[bidx].name}{suffix}.ply"
                    for bidx in bidx_group
                    if (out_dir / f"{ref_list[bidx].name}{suffix}.ply").exists()
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

        print('Renamed/merged output files based on majority contributing current_id.')

    print(f"Done. Wrote per-existing outputs to {out_dir}.")
    if passthrough_copied > 0:
        print(f"Copied {passthrough_copied} unmatched current file(s) through to output.")
    print(f"Current-ID map: {out_dir / 'current_id_map.csv'}")
    print(f"Report: {report_path}")

if __name__ == '__main__':
    main()
