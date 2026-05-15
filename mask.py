#!/usr/bin/env python3
import argparse
import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import laspy
import numpy as np
from laspy import ExtraBytesParams, ScaleAwarePointRecord
from plyfile import PlyData
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree
from tqdm import tqdm

SUPPORTED_EXTS = {".ply", ".las", ".laz"}


@dataclass
class HullInfo:
    tree: cKDTree
    delaunay: Optional[Delaunay]
    mins: np.ndarray
    maxs: np.ndarray


@dataclass
class MaskRecord:
    tree_id: int
    stem_id: int
    points: np.ndarray
    tree: cKDTree
    mins: np.ndarray
    maxs: np.ndarray
    hull: Optional[HullInfo] = None
    las_out: Optional[Path] = None
    ply_out: Optional[Path] = None


class PlyStructAppender:
    """Append structured records to temp body and emit one valid binary PLY."""

    def __init__(self, out_path: Path, dtype: np.dtype):
        self.out_path = out_path
        self.tmp_path = out_path.with_suffix(out_path.suffix + ".bin")
        if self.out_path.exists():
            self.out_path.unlink()
        if self.tmp_path.exists():
            self.tmp_path.unlink()
        self.count = 0
        self.dtype = dtype
        self.tmp_f = open(self.tmp_path, "ab")

    def append(self, recs: np.ndarray):
        if recs is None or recs.size == 0:
            return
        if recs.dtype != self.dtype:
            recs = recs.astype(self.dtype, copy=False)
        recs = np.ascontiguousarray(recs)
        self.tmp_f.write(recs.tobytes())
        self.count += recs.shape[0]

    def close(self):
        if not self.tmp_f.closed:
            self.tmp_f.close()

    def _header(self) -> bytes:
        lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {self.count}",
        ]
        for name, dt in self.dtype.fields.items():
            t = dt[0]
            if t == np.dtype("<f4"):
                lines.append(f"property float {name}")
            elif t == np.dtype("<i4"):
                lines.append(f"property int {name}")
            else:
                raise TypeError(f"Unsupported dtype for PLY field {name}: {t}")
        lines.append("end_header")
        return ("\n".join(lines) + "\n").encode("ascii")

    def finalize(self):
        self.close()
        with open(self.out_path, "wb") as fout:
            fout.write(self._header())
        with open(self.out_path, "ab") as fout, open(self.tmp_path, "rb") as ftmp:
            for chunk in iter(lambda: ftmp.read(1 << 20), b""):
                fout.write(chunk)
        try:
            os.remove(self.tmp_path)
        except OSError:
            pass

    def discard(self):
        self.close()
        try:
            os.remove(self.tmp_path)
        except OSError:
            pass
        try:
            os.remove(self.out_path)
        except OSError:
            pass


def list_point_files(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]


def parse_mask_ids(stem: str) -> Tuple[int, int]:
    """Parse tree/stem ids from {tree_id} or {tree_id}_{stem_id}."""
    parts = stem.split("_")
    if len(parts) == 1:
        tree_s, stem_s = parts[0], "1"
    elif len(parts) == 2:
        tree_s, stem_s = parts
    else:
        raise ValueError(
            f"Invalid mask filename '{stem}'. Expected {{tree_id}} or {{tree_id}}_{{stem_id}}"
        )

    if not tree_s.isdigit() or not stem_s.isdigit():
        raise ValueError(
            f"Invalid mask filename '{stem}'. tree_id/stem_id must be positive integers"
        )

    tree_id = int(tree_s)
    stem_id = int(stem_s)
    if tree_id < 1 or tree_id >= 20000:
        raise ValueError(f"Invalid tree_id={tree_id} in '{stem}'. Expected 1..19999")
    if stem_id < 1 or stem_id >= 27:
        raise ValueError(f"Invalid stem_id={stem_id} in '{stem}'. Expected 1..26")
    return tree_id, stem_id


def get_points(filepath: Path) -> np.ndarray:
    """Read XYZ from PLY/LAS/LAZ."""
    ext = filepath.suffix.lower()
    try:
        if ext == ".ply":
            with open(filepath, "rb") as f:
                ply = PlyData.read(f)
                v = ply["vertex"]
                return np.vstack((v["x"], v["y"], v["z"])).T.astype(np.float32, copy=False)
        if ext in (".las", ".laz"):
            with laspy.open(filepath) as f:
                las = f.read()
                return np.vstack((las.x, las.y, las.z)).T.astype(np.float32, copy=False)
    except Exception as e:
        print(f"Warning: failed to read {filepath}: {e}")
    return np.empty((0, 3), dtype=np.float32)


def build_hull_info(points: np.ndarray, decimation_size: float, vox_mul: float) -> Optional[HullInfo]:
    """Build hull support from voxelized z-slice extreme XY points."""
    if points.shape[0] < 8:
        return None

    grid = float(decimation_size) * float(vox_mul)
    if grid <= 0:
        return None

    mins = np.min(points, axis=0)
    vox = np.floor((points - mins) / grid).astype(np.int32)
    z_ids = vox[:, 2]

    keep_indices: List[int] = []
    for z_val in np.unique(z_ids):
        sel_z = z_ids == z_val
        if not np.any(sel_z):
            continue

        pts_z = points[sel_z]
        vox_z = vox[sel_z]
        global_idx_z = np.where(sel_z)[0]
        centroid_xy = np.mean(pts_z[:, :2], axis=0)

        uniq, inv = np.unique(vox_z, axis=0, return_inverse=True)
        for gid in range(uniq.shape[0]):
            local = np.where(inv == gid)[0]
            if local.size == 1:
                keep_indices.append(int(global_idx_z[local[0]]))
                continue
            group_pts = pts_z[local]
            radial = np.linalg.norm(group_pts[:, :2] - centroid_xy, axis=1)
            pick_local = local[int(np.argmax(radial))]
            keep_indices.append(int(global_idx_z[pick_local]))

    if not keep_indices:
        return None

    support = points[np.unique(np.asarray(keep_indices, dtype=np.int64))]
    if support.shape[0] < 4:
        return None

    try:
        hull = ConvexHull(support, qhull_options="QJ")
    except QhullError:
        return None

    hull_pts = support[hull.vertices]
    if hull_pts.shape[0] < 4:
        return None

    try:
        delaunay = Delaunay(hull_pts, qhull_options="QJ")
    except QhullError:
        delaunay = None

    return HullInfo(
        tree=cKDTree(hull_pts, leafsize=64, compact_nodes=True, balanced_tree=True),
        delaunay=delaunay,
        mins=np.min(hull_pts, axis=0).astype(np.float32),
        maxs=np.max(hull_pts, axis=0).astype(np.float32),
    )


def load_masks(
    mask_folder: Path,
    distance: float,
    use_hull_fill: bool,
    decimation_size: Optional[float],
    vox_mul: float,
    out_dir: Path,
    las_suffix: str,
) -> List[MaskRecord]:
    grouped: Dict[Tuple[int, int], List[np.ndarray]] = {}
    mask_files = list_point_files(mask_folder)
    if not mask_files:
        raise SystemExit("No mask files found.")

    print(f"Loading {len(mask_files)} mask file(s)...")
    for f in tqdm(mask_files, unit="mask"):
        try:
            tree_id, stem_id = parse_mask_ids(f.stem)
        except ValueError as e:
            raise SystemExit(str(e)) from e

        pts = get_points(f)
        if pts.shape[0] == 0:
            continue
        grouped.setdefault((tree_id, stem_id), []).append(pts)

    if not grouped:
        raise SystemExit("No usable mask points found.")

    masks: List[MaskRecord] = []
    for (tree_id, stem_id), point_sets in sorted(grouped.items()):
        pts = np.vstack(point_sets).astype(np.float32, copy=False)
        tree = cKDTree(pts, leafsize=64, compact_nodes=True, balanced_tree=True)
        mins = np.min(pts, axis=0).astype(np.float32) - float(distance)
        maxs = np.max(pts, axis=0).astype(np.float32) + float(distance)
        key_name = f"{tree_id}_{stem_id}"

        hull = None
        if use_hull_fill:
            hull = build_hull_info(pts, float(decimation_size), vox_mul)

        masks.append(
            MaskRecord(
                tree_id=tree_id,
                stem_id=stem_id,
                points=pts,
                tree=tree,
                mins=mins,
                maxs=maxs,
                hull=hull,
                las_out=out_dir / f"{key_name}{las_suffix}",
                ply_out=out_dir / f"{key_name}.ply",
            )
        )

    print(f"Using {len(masks)} unique mask id pair(s).")
    return masks


def process_chunk(
    chunk_xyz: np.ndarray,
    masks: List[MaskRecord],
    distance: float,
    use_hull_fill: bool,
    hull_eps: float,
) -> np.ndarray:
    """Primary nearest-distance assignment, with optional secondary hull fill."""
    n = chunk_xyz.shape[0]
    min_dists = np.full(n, np.inf, dtype=np.float32)
    matched_mask_idx = np.full(n, -1, dtype=np.int64)

    c_min = np.min(chunk_xyz, axis=0)
    c_max = np.max(chunk_xyz, axis=0)

    for mask_idx, mask in enumerate(masks):
        if np.any(c_min > mask.maxs) or np.any(c_max < mask.mins):
            continue

        in_box = np.all((chunk_xyz >= mask.mins) & (chunk_xyz <= mask.maxs), axis=1)
        if not np.any(in_box):
            continue

        candidates = chunk_xyz[in_box]
        dists, _ = mask.tree.query(candidates, k=1, distance_upper_bound=distance, workers=1)
        valid = dists != np.inf
        if not np.any(valid):
            continue

        global_idx = np.where(in_box)[0]
        valid_global = global_idx[valid]
        valid_dists = dists[valid]
        better = valid_dists < min_dists[valid_global]
        if np.any(better):
            idx = valid_global[better]
            min_dists[idx] = valid_dists[better]
            matched_mask_idx[idx] = mask_idx

    if not use_hull_fill:
        return matched_mask_idx

    unmatched = matched_mask_idx < 0
    if not np.any(unmatched):
        return matched_mask_idx

    hull_best = np.full(n, np.inf, dtype=np.float32)

    for mask_idx, mask in enumerate(masks):
        hull = mask.hull
        if hull is None:
            continue

        mins = hull.mins - hull_eps
        maxs = hull.maxs + hull_eps
        in_hull_box = unmatched & np.all((chunk_xyz >= mins) & (chunk_xyz <= maxs), axis=1)
        if not np.any(in_hull_box):
            continue

        test_pts = chunk_xyz[in_hull_box]
        if hull.delaunay is None:
            inside = np.zeros(test_pts.shape[0], dtype=bool)
        else:
            inside = hull.delaunay.find_simplex(test_pts) >= 0

        boundary_dist, _ = hull.tree.query(test_pts, k=1, workers=1)
        near = boundary_dist <= hull_eps
        accepted = inside | near
        if not np.any(accepted):
            continue

        local_idx = np.where(in_hull_box)[0]
        accepted_idx = local_idx[accepted]
        accepted_score = np.where(inside[accepted], 0.0, boundary_dist[accepted]).astype(np.float32)
        better = accepted_score < hull_best[accepted_idx]
        if np.any(better):
            widx = accepted_idx[better]
            matched_mask_idx[widx] = mask_idx
            hull_best[widx] = accepted_score[better]

    return matched_mask_idx


def build_las_header_with_ids(src_header: laspy.LasHeader) -> laspy.LasHeader:
    out_header = src_header.copy()
    dim_names = set(out_header.point_format.dimension_names)
    if "tree_id" not in dim_names:
        out_header.add_extra_dim(ExtraBytesParams(name="tree_id", type=np.int32))
    if "stem_id" not in dim_names:
        out_header.add_extra_dim(ExtraBytesParams(name="stem_id", type=np.int32))
    return out_header


def make_point_record_with_ids(
    chunk_subset,
    out_header: laspy.LasHeader,
    src_dim_names: List[str],
    tree_id: int,
    stem_id: int,
):
    rec = ScaleAwarePointRecord.zeros(len(chunk_subset), header=out_header)
    for dim in src_dim_names:
        if dim in ("tree_id", "stem_id"):
            continue
        if hasattr(chunk_subset, dim):
            setattr(rec, dim, np.asarray(getattr(chunk_subset, dim)))
    rec.tree_id = np.full(len(chunk_subset), tree_id, dtype=np.int32)
    rec.stem_id = np.full(len(chunk_subset), stem_id, dtype=np.int32)
    return rec


def main():
    parser = argparse.ArgumentParser(
        description="Extract points per mask id and write LAS/LAZ with IDs and/or per-mask PLY."
    )
    parser.add_argument("-m", "--mask-folder", required=True, type=Path, help="Folder containing mask files")
    parser.add_argument("-t", "--target", required=True, type=Path, help="Target LAS/LAZ/PLY file or folder")
    parser.add_argument("-o", "--output", type=Path, help="Output directory")
    parser.add_argument("-d", "--distance", type=float, required=True, help="Distance threshold for primary match")
    parser.add_argument("--chunk-size", type=int, default=500000, help="Points per chunk (default: 500,000)")
    parser.add_argument(
        "--ids-only",
        action="store_true",
        help="Only write LAS/LAZ outputs with tree_id and stem_id (skip PLY outputs)",
    )
    parser.add_argument(
        "--ply-only",
        action="store_true",
        help="Only write PLY outputs (skip LAS/LAZ outputs)",
    )
    parser.add_argument(
        "--hull-fill",
        action="store_true",
        help="Enable secondary hull-based inclusion for unmatched points",
    )
    parser.add_argument(
        "--decimation-size",
        type=float,
        default=None,
        help="Base decimation size used with hull fill (required for --hull-fill)",
    )
    parser.add_argument(
        "--vox-mul",
        type=float,
        default=3.0,
        help="Voxel multiplier for hull support grid size: decimation_size * vox_mul (default 3)",
    )
    parser.add_argument(
        "--hull-eps",
        type=float,
        default=0.05,
        help="Small epsilon distance used for near-hull inclusion (default 0.05)",
    )
    args = parser.parse_args()

    if args.ids_only and args.ply_only:
        raise SystemExit("Choose only one of --ids-only or --ply-only.")

    write_las = not args.ply_only
    write_ply = not args.ids_only

    if args.hull_fill and (args.decimation_size is None or args.decimation_size <= 0):
        raise SystemExit("--hull-fill requires --decimation-size > 0")

    if args.output:
        out_dir = args.output
    else:
        out_dir = Path(f"{args.mask_folder.name}_extracted")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.target.is_dir():
        target_files = list_point_files(args.target)
    elif args.target.is_file() and args.target.suffix.lower() in SUPPORTED_EXTS:
        target_files = [args.target]
    else:
        raise SystemExit("Invalid target path. Must be a file or directory with .ply/.las/.laz")

    if not target_files:
        raise SystemExit("No target files found.")

    las_targets = [p for p in target_files if p.suffix.lower() in (".las", ".laz")]
    if write_las and not las_targets:
        raise SystemExit("LAS/LAZ output requested but no LAS/LAZ target inputs were provided.")

    las_suffix = ".laz" if any(p.suffix.lower() == ".laz" for p in las_targets) else ".las"

    masks = load_masks(
        mask_folder=args.mask_folder,
        distance=args.distance,
        use_hull_fill=args.hull_fill,
        decimation_size=args.decimation_size,
        vox_mul=args.vox_mul,
        out_dir=out_dir,
        las_suffix=las_suffix,
    )

    if write_las:
        for m in masks:
            if m.las_out is not None and m.las_out.exists():
                m.las_out.unlink()

    ply_dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("tree_id", "<i4"),
        ("stem_id", "<i4"),
    ])
    ply_appenders: Dict[Tuple[int, int], PlyStructAppender] = {}
    if write_ply:
        for m in masks:
            if m.ply_out is None:
                continue
            ply_appenders[(m.tree_id, m.stem_id)] = PlyStructAppender(m.ply_out, ply_dtype)

    print(f"Processing {len(target_files)} target file(s)")
    print(f"Primary assignment distance: {args.distance}")
    if args.hull_fill:
        print(
            f"Hull fill enabled: decimation_size={args.decimation_size}, "
            f"vox_mul={args.vox_mul}, hull_eps={args.hull_eps}"
        )

    total_points = 0
    assigned_points = 0
    start_time = time.time()

    for target_file in target_files:
        ext = target_file.suffix.lower()
        print(f"Processing target: {target_file.name}")

        if ext in (".las", ".laz"):
            with contextlib.ExitStack() as stack:
                src = stack.enter_context(laspy.open(target_file))
                src_dim_names = list(src.header.point_format.dimension_names)
                out_header = build_las_header_with_ids(src.header) if write_las else None

                las_writers: Dict[Tuple[int, int], object] = {}
                if write_las:
                    for m in masks:
                        out_path = m.las_out
                        if out_path is None:
                            continue
                        mode = "a" if out_path.exists() else "w"
                        las_writers[(m.tree_id, m.stem_id)] = stack.enter_context(
                            laspy.open(out_path, mode=mode, header=out_header)
                        )

                for chunk in tqdm(src.chunk_iterator(args.chunk_size), unit="pts"):
                    chunk_xyz = np.vstack((chunk.x, chunk.y, chunk.z)).T.astype(np.float32, copy=False)
                    if chunk_xyz.shape[0] == 0:
                        continue
                    total_points += chunk_xyz.shape[0]

                    matched_mask_idx = process_chunk(
                        chunk_xyz,
                        masks,
                        args.distance,
                        use_hull_fill=args.hull_fill,
                        hull_eps=args.hull_eps,
                    )

                    for mask_idx, mask in enumerate(masks):
                        sel = matched_mask_idx == mask_idx
                        if not np.any(sel):
                            continue
                        assigned_points += int(np.sum(sel))

                        if write_las:
                            subset = chunk[sel]
                            rec = make_point_record_with_ids(
                                chunk_subset=subset,
                                out_header=out_header,
                                src_dim_names=src_dim_names,
                                tree_id=mask.tree_id,
                                stem_id=mask.stem_id,
                            )
                            las_writers[(mask.tree_id, mask.stem_id)].write_points(rec)

                        if write_ply:
                            recs = np.zeros(int(np.sum(sel)), dtype=ply_dtype)
                            pts = chunk_xyz[sel]
                            recs["x"] = pts[:, 0]
                            recs["y"] = pts[:, 1]
                            recs["z"] = pts[:, 2]
                            recs["tree_id"] = mask.tree_id
                            recs["stem_id"] = mask.stem_id
                            ply_appenders[(mask.tree_id, mask.stem_id)].append(recs)

        elif ext == ".ply":
            with open(target_file, "rb") as f:
                ply = PlyData.read(f)
            pts = np.vstack((ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"])).T.astype(
                np.float32, copy=False
            )

            if write_las:
                raise SystemExit(
                    "LAS/LAZ output is enabled but target contains PLY input. Use --ply-only for PLY targets."
                )

            for start_idx in tqdm(range(0, pts.shape[0], args.chunk_size), unit="pts"):
                end_idx = min(start_idx + args.chunk_size, pts.shape[0])
                chunk_xyz = pts[start_idx:end_idx]
                if chunk_xyz.shape[0] == 0:
                    continue
                total_points += chunk_xyz.shape[0]

                matched_mask_idx = process_chunk(
                    chunk_xyz,
                    masks,
                    args.distance,
                    use_hull_fill=args.hull_fill,
                    hull_eps=args.hull_eps,
                )

                for mask_idx, mask in enumerate(masks):
                    sel = matched_mask_idx == mask_idx
                    if not np.any(sel):
                        continue
                    assigned_points += int(np.sum(sel))
                    if write_ply:
                        recs = np.zeros(int(np.sum(sel)), dtype=ply_dtype)
                        sel_pts = chunk_xyz[sel]
                        recs["x"] = sel_pts[:, 0]
                        recs["y"] = sel_pts[:, 1]
                        recs["z"] = sel_pts[:, 2]
                        recs["tree_id"] = mask.tree_id
                        recs["stem_id"] = mask.stem_id
                        ply_appenders[(mask.tree_id, mask.stem_id)].append(recs)

    if write_ply:
        for app in ply_appenders.values():
            if app.count > 0:
                app.finalize()
            else:
                app.discard()

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f}s. Scanned {total_points} points; assigned {assigned_points}.")
    if write_las:
        print(f"Wrote LAS/LAZ outputs to: {out_dir}")
    if write_ply:
        print(f"Wrote per-mask PLY outputs to: {out_dir}")


if __name__ == "__main__":
    main()
