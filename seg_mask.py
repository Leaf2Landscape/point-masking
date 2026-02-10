#!/usr/bin/env python3
"""
seg_mask.py — Correct initial tree segmentations using a trusted reference set,
with a single Euclidean distance threshold, and write ONE file per A tree.

Overview
--------
Given two sets of tree-level point clouds:
  * A (initial): segmentations you want to correct (e.g., recent drone/TLS trees)
  * B (reference): trusted previous segmentations (e.g., prior TLS, curated set)

Pipeline:
  1) Compute a base (x,y) per tree using a height-above-ground (HAG) slice.
  2) Match A bases to nearest B bases within a gating radius (closest wins).
     If multiple B bases are within the gate, split A by 2D nearest-base (Voronoi).
  3) Correct each matched partition: keep only points whose nearest neighbor
     in the matched B tree is within user-specified **Euclidean** distance `--distance`.
  4) Merge all kept partitions and write ONE .ply per A tree:
       * {Aname}_corrected.ply  for matched A (merged across partitions)
       * {Aname}_unmatched.ply  for unmatched A trees (copy-through)

Safety guard (optional): if A-vs-B agreement near the stem (HAG <= --z-core)
is poor (< --stem-keep-thresh), skip correction for that partition (keep it as-is).

Inputs
------
- A and B folders: .ply, .las, or .laz (XYZ only).

Outputs
-------
- One .ply per A tree:
    - Aname_corrected.ply (if matched) or
    - Aname_unmatched.ply (if not matched)
- A CSV report with match metrics.

Usage
-----
python seg_mask.py \
  -a /path/to/A_initial \
  -b /path/to/B_reference \
  -o /path/to/output \
  --gate 1.5 \
  --base-slice 0.10 0.40 \
  --hag-percentile 0.02 \
  --distance 1.0 \
  --z-core 1.5 \
  --stem-keep-thresh 0.55

Dependencies
------------
  pip install numpy scipy tqdm laspy plyfile
"""

from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
import laspy
from tqdm import tqdm

SUPPORTED_EXTS = {'.ply', '.las', '.laz'}

# ------------------------- IO helpers -------------------------

def read_xyz(path: Path) -> np.ndarray:
    """Read XYZ as (N,3) float64 from .ply/.las/.laz. Returns empty array on failure."""
    ext = path.suffix.lower()
    try:
        if ext == '.ply':
            with open(path, 'rb') as f:
                ply = PlyData.read(f)
            v = ply['vertex']
            return np.vstack((v['x'], v['y'], v['z'])).T.astype(np.float64, copy=False)
        elif ext in ('.las', '.laz'):
            with laspy.open(path) as src:
                las = src.read()
            return np.vstack((las.x, las.y, las.z)).T.astype(np.float64, copy=False)
        else:
            raise ValueError(f'Unsupported extension: {ext}')
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")
        return np.empty((0, 3), dtype=np.float64)


def write_ply_xyz(path: Path, xyz: np.ndarray) -> None:
    """Write (N,3) float array to a binary_little_endian PLY as x,y,z float32."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if xyz.size == 0:
        dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        data = np.empty(0, dtype=dtype)
    else:
        xyz32 = np.asarray(xyz, dtype='<f4')
        dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        data = np.empty(xyz32.shape[0], dtype=dtype)
        data['x'], data['y'], data['z'] = xyz32[:, 0], xyz32[:, 1], xyz32[:, 2]
    el = PlyElement.describe(data, 'vertex')
    PlyData([el], text=False).write(str(path))


# ------------------------- Geometry helpers -------------------------
@dataclass
class TreeInfo:
    name: str
    path: Path
    xyz: np.ndarray
    base_xy: np.ndarray  # (2,)
    z0: float            # estimated ground elevation
    hag: np.ndarray      # per-point height above ground


@dataclass
class RefTree(TreeInfo):
    kd: Optional[cKDTree] = None  # KD-tree on full XYZ (built lazily)


def estimate_hag(xyz: np.ndarray, percentile: float) -> Tuple[np.ndarray, float]:
    """Return (hag, z0) where z0 is the given percentile of z, hag = z - z0."""
    if xyz.size == 0:
        return np.empty((0,), dtype=np.float64), 0.0
    z = xyz[:, 2]
    z0 = float(np.quantile(z, percentile))
    hag = z - z0
    return hag, z0


def compute_base_xy(xyz: np.ndarray, hag: np.ndarray,
                    base_low: float, base_high: float,
                    fallback_percent: float = 0.05,
                    min_pts: int = 30) -> np.ndarray:
    """Compute base (x,y) from HAG slice [base_low, base_high];
    fallback to lowest `fallback_percent` by z if slice is sparse. Returns (2,) float64.
    """
    sel = (hag >= base_low) & (hag <= base_high)
    pts = xyz[sel]
    if pts.shape[0] < min_pts:
        if xyz.shape[0] == 0:
            return np.array([np.nan, np.nan], dtype=np.float64)
        z = xyz[:, 2]
        thr = np.quantile(z, fallback_percent)
        pts = xyz[z <= thr]
        if pts.shape[0] == 0:
            pts = xyz
    return np.mean(pts[:, :2], axis=0).astype(np.float64, copy=False)


# ------------------------- Matching & correction -------------------------
@dataclass
class MatchResult:
    a_name: str
    b_name: Optional[str]
    base_dist: Optional[float]
    num_in: int
    num_out: int
    kept_frac: float
    note: str


def build_ref_index(ref_paths: List[Path], hag_percentile: float,
                    base_low: float, base_high: float,
                    verbose: bool = True) -> Tuple[List[RefTree], cKDTree, np.ndarray]:
    ref_trees: List[RefTree] = []
    bases: List[np.ndarray] = []
    it = tqdm(ref_paths, desc='Loading reference trees', unit='tree') if verbose else ref_paths
    for p in it:
        xyz = read_xyz(p)
        hag, z0 = estimate_hag(xyz, hag_percentile)
        base_xy = compute_base_xy(xyz, hag, base_low, base_high)
        ref_trees.append(RefTree(name=p.stem, path=p, xyz=xyz, base_xy=base_xy, z0=z0, hag=hag))
        bases.append(base_xy)
    bases_arr = np.asarray(bases, dtype=np.float64)
    kd = cKDTree(bases_arr) if len(ref_trees) > 0 else None
    return ref_trees, kd, bases_arr


def split_by_nearest_bases(a_xy: np.ndarray, candidate_bases: np.ndarray) -> np.ndarray:
    """Label each a_xy point by nearest candidate base (returns labels in [0..K-1])."""
    diff = a_xy[:, None, :] - candidate_bases[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return np.argmin(d2, axis=1)


def ensure_ref_kd(ref: RefTree) -> cKDTree:
    if ref.kd is None:
        ref.kd = cKDTree(ref.xyz) if ref.xyz.size > 0 else cKDTree(np.zeros((1, 3)))
    return ref.kd


def stem_agreement_fraction(a_xyz: np.ndarray, a_hag: np.ndarray, ref: RefTree,
                            z_core: float, distance: float) -> float:
    """Fraction of A stem-slice points (HAG <= z_core) whose NN distance to ref <= distance."""
    stem_sel = (a_hag <= z_core)
    if not np.any(stem_sel):
        return 0.0
    kd = ensure_ref_kd(ref)
    pts = a_xyz[stem_sel]
    dists, _ = kd.query(pts, k=1)
    return float(np.sum(dists <= distance) / max(1, pts.shape[0]))


def correct_keep_mask(a_xyz: np.ndarray, ref: RefTree,
                      distance: float, workers: int) -> np.ndarray:
    """Return boolean keep mask: keep points whose NN distance to ref <= distance."""
    if a_xyz.size == 0:
        return np.zeros((0,), dtype=bool)
    kd = ensure_ref_kd(ref)
    dists, _ = kd.query(a_xyz, k=1, workers=workers)
    return (dists <= distance)


# ------------------------- CLI and main pipeline -------------------------

def list_point_files(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in SUPPORTED_EXTS and p.is_file()]


def main():
    ap = argparse.ArgumentParser(
        description='Correct initial tree segmentations using a reference set (base-matched, single distance threshold, one file per A)')
    ap.add_argument('-a', '--init-folder', type=Path, required=True, help='Folder of initial A trees (.ply/.las/.laz)')
    ap.add_argument('-b', '--ref-folder', type=Path, required=True, help='Folder of reference B trees (.ply/.las/.laz)')
    ap.add_argument('-o', '--output', type=Path, required=True, help='Output folder for per-A corrected .ply files')

    # HAG & base detection
    ap.add_argument('--hag-percentile', type=float, default=0.02,
                    help='Percentile of z used as ground per tree (default 0.02 = 2%)')
    ap.add_argument('--base-slice', type=float, nargs=2, default=[0.10, 0.40],
                    metavar=('LOW', 'HIGH'), help='HAG slice (m) to compute base centroid (default 0.10 0.40)')
    ap.add_argument('--fallback-percent', type=float, default=0.05,
                    help='If base slice sparse, fallback to lowest proportion of z (default 0.05)')

    # Matching
    ap.add_argument('--gate', type=float, default=1.5, help='Gating radius (m) for base matching (default 1.5)')

    # Correction distance (single threshold)
    ap.add_argument('--distance', type=float, default=1.0,
                    help='Max Euclidean distance (m) from matched B tree to keep a point')

    # Stem agreement guard
    ap.add_argument('--z-core', type=float, default=1.5, help='Stem slice height threshold (m) for agreement check')
    ap.add_argument('--stem-keep-thresh', type=float, default=0.55,
                    help='If fraction of A stem-slice within --distance of B < this, skip correction (keep all)')

    # System
    ap.add_argument('--workers', type=int, default=-1, help='Threads for KD queries (-1 = all cores)')
    ap.add_argument('--report', type=Path, default=None, help='Optional CSV path for match/correction report')

    args = ap.parse_args()

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    a_files = list_point_files(args.init_folder)
    b_files = list_point_files(args.ref_folder)
    if not a_files:
        raise SystemExit('No A trees found in --init-folder')
    if not b_files:
        raise SystemExit('No B trees found in --ref-folder')

    base_low, base_high = args.base_slice

    # Build reference index
    ref_list, ref_kd, ref_bases = build_ref_index(b_files, args.hag_percentile, base_low, base_high, verbose=True)

    report_rows: List[MatchResult] = []

    print(f'Processing {len(a_files)} initial trees from {args.init_folder} ...')

    for a_path in tqdm(a_files, desc='Correcting A trees', unit='tree'):
        a_xyz = read_xyz(a_path)
        a_hag, a_z0 = estimate_hag(a_xyz, args.hag_percentile)
        a_base = compute_base_xy(a_xyz, a_hag, base_low, base_high,
                                 fallback_percent=args.fallback_percent)

        # Degenerate or no reference
        if not np.isfinite(a_base).all() or ref_kd is None or len(ref_list) == 0:
            out_path = out_dir / f'{a_path.stem}_unchanged.ply'
            write_ply_xyz(out_path, a_xyz)
            report_rows.append(MatchResult(a_path.name, None, None, int(a_xyz.shape[0]), int(a_xyz.shape[0]), 1.0,
                                           note='degenerate_base' if not np.isfinite(a_base).all() else 'no_reference'))
            continue

        # Find B candidates within gate
        cand_idx = ref_kd.query_ball_point(a_base, r=args.gate)
        if len(cand_idx) == 0:
            # Unmatched A: copy-through to single output (manual fix later)
            out_path = out_dir / f'{a_path.stem}_unmatched.ply'
            write_ply_xyz(out_path, a_xyz)
            report_rows.append(MatchResult(a_path.name, None, None, int(a_xyz.shape[0]), int(a_xyz.shape[0]), 1.0,
                                           note='no_match_within_gate'))
            continue

        # Partitions: single or Voronoi by candidate B bases
        if len(cand_idx) == 1:
            parts = [(cand_idx[0], np.arange(a_xyz.shape[0], dtype=np.int64))]
        else:
            cand_bases = ref_bases[cand_idx]
            labels = split_by_nearest_bases(a_xyz[:, :2], cand_bases)
            parts = []
            for k, bidx in enumerate(cand_idx):
                sel = np.where(labels == k)[0]
                if sel.size > 0:
                    parts.append((bidx, sel))

        # Accumulate kept points for this A (merged into one file at the end)
        kept_accum = []
        part_counter = 0

        for b_idx, sel_idx in parts:
            ref = ref_list[b_idx]
            sub_xyz = a_xyz[sel_idx]
            sub_hag = a_hag[sel_idx]

            # Stem agreement guard
            frac = stem_agreement_fraction(sub_xyz, sub_hag, ref, args.z_core, args.distance)
            if frac < args.stem_keep_thresh:
                note = f'skipped_correction_low_stem_agreement({frac:.2f})'
                keep_mask = np.ones(sub_xyz.shape[0], dtype=bool)  # keep all if guard fails
            else:
                note = f'corrected(stem_agree={frac:.2f})'
                keep_mask = correct_keep_mask(sub_xyz, ref, args.distance, workers=args.workers)

            kept_xyz = sub_xyz[keep_mask]
            kept_accum.append(kept_xyz)

            # Report per partition
            base_dist = float(np.linalg.norm(a_base - ref.base_xy))
            report_rows.append(MatchResult(a_path.name, ref.path.name, base_dist,
                                           int(sub_xyz.shape[0]), int(kept_xyz.shape[0]),
                                           0.0 if sub_xyz.shape[0] == 0 else kept_xyz.shape[0] / sub_xyz.shape[0],
                                           note))
            part_counter += 1

        # Merge and write ONE file for this A
        if kept_accum:
            merged_xyz = np.vstack([arr for arr in kept_accum if arr.size > 0]) if any(arr.size for arr in kept_accum) else np.empty((0, 3), dtype=np.float64)
        else:
            merged_xyz = np.empty((0, 3), dtype=np.float64)
        out_path = out_dir / f'{a_path.stem}_corrected.ply'
        write_ply_xyz(out_path, merged_xyz)

    # Report
    report_path = args.report if args.report is not None else (out_dir / 'seg_mask_report.csv')
    with open(report_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['a_name', 'b_name', 'base_dist_m', 'num_in', 'num_out', 'kept_frac', 'note'])
        for r in report_rows:
            w.writerow([r.a_name, r.b_name or '', f'{r.base_dist:.3f}' if r.base_dist is not None else '',
                        r.num_in, r.num_out, f'{r.kept_frac:.3f}', r.note])

    print(f'Done. Wrote per-A outputs to {out_dir}. Report: {report_path}')


if __name__ == '__main__':
    main()