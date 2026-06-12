#!/usr/bin/env python3
import argparse
import concurrent.futures
import contextlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import laspy
import numpy as np
from laspy import ExtraBytesParams, ScaleAwarePointRecord
from plyfile import PlyData
from scipy.spatial import ConvexHull, QhullError, cKDTree
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


SUPPORTED_EXTS = {".ply", ".las", ".laz"}
PROGRESS_KW = {
    "leave": False,
    "mininterval": 0.5,
    "dynamic_ncols": True,
    "disable": False,
    "file": sys.stdout,
}


def stage(msg: str) -> None:
    print(f"[mask] {msg}", flush=True)


def normalize_cli_path(path: Path, must_exist: bool) -> Path:
    p = path.expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve(strict=must_exist)


def list_point_files(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]


def iter_points(path: Path, chunk_size: int) -> Iterable[np.ndarray]:
    ext = path.suffix.lower()
    if ext in (".las", ".laz"):
        with laspy.open(path) as src:
            for chunk in src.chunk_iterator(chunk_size):
                xyz = np.vstack((chunk.x, chunk.y, chunk.z)).T.astype(np.float32, copy=False)
                if xyz.size > 0:
                    yield xyz
        return

    if ext == ".ply":
        with open(path, "rb") as f:
            ply = PlyData.read(f)
        v = ply["vertex"]
        xyz = np.vstack((v["x"], v["y"], v["z"])).T.astype(np.float32, copy=False)
        if xyz.shape[0] == 0:
            return
        for start in range(0, xyz.shape[0], chunk_size):
            end = min(start + chunk_size, xyz.shape[0])
            yield xyz[start:end]
        return

    raise ValueError(f"Unsupported file extension: {path.suffix}")


def get_points(filepath: Path, chunk_size: int = 500000, progress_desc: Optional[str] = None) -> np.ndarray:
    ext = filepath.suffix.lower()
    try:
        if ext == ".ply":
            with open(filepath, "rb") as f:
                ply = PlyData.read(f)
                v = ply["vertex"]
                return np.vstack((v["x"], v["y"], v["z"])).T.astype(np.float32, copy=False)
        if ext in (".las", ".laz"):
            with laspy.open(filepath) as src:
                if src.header.point_count == 0:
                    return np.empty((0, 3), dtype=np.float32)

                chunks: List[np.ndarray] = []
                total_chunks = max(1, (int(src.header.point_count) + int(chunk_size) - 1) // int(chunk_size))
                chunk_iter = src.chunk_iterator(chunk_size)
                if progress_desc is not None:
                    chunk_iter = tqdm(
                        chunk_iter,
                        total=total_chunks,
                        desc=progress_desc,
                        unit="chunk",
                        **PROGRESS_KW,
                    )

                for chunk in chunk_iter:
                    xyz = np.vstack((chunk.x, chunk.y, chunk.z)).T.astype(np.float32, copy=False)
                    if xyz.shape[0] > 0:
                        chunks.append(xyz)

                if not chunks:
                    return np.empty((0, 3), dtype=np.float32)
                return np.vstack(chunks)
    except Exception as exc:
        stage(f"Warning: failed to read {filepath}: {exc}")
    return np.empty((0, 3), dtype=np.float32)


def parse_mask_ids(stem: str) -> Tuple[int, int, bool]:
    parts = stem.split("_")
    if len(parts) == 1:
        tree_s, stem_s, has_stem_id = parts[0], None, False
    elif len(parts) == 2:
        tree_s, stem_s, has_stem_id = parts[0], parts[1], True
    else:
        raise ValueError(
            f"Invalid mask filename '{stem}'. Expected {{tree_id}} or {{tree_id}}_{{stem_id}}"
        )

    if not tree_s.isdigit() or (stem_s is not None and not stem_s.isdigit()):
        raise ValueError(
            f"Invalid mask filename '{stem}'. tree_id/stem_id must be positive integers"
        )

    tree_id = int(tree_s)
    stem_id = int(stem_s) if stem_s is not None else 0
    if tree_id < 1 or tree_id >= 20000:
        raise ValueError(f"Invalid tree_id={tree_id} in '{stem}'. Expected 1..19999")
    if has_stem_id and (stem_id < 1 or stem_id >= 27):
        raise ValueError(f"Invalid stem_id={stem_id} in '{stem}'. Expected 1..26")
    return tree_id, stem_id, has_stem_id


@dataclass
class HullInfo:
    mins: np.ndarray
    maxs: np.ndarray
    A: np.ndarray
    b: np.ndarray
    hull_points: np.ndarray


@dataclass
class MaskRecord:
    tree_id: int
    stem_id: int
    has_stem_id: bool
    points: np.ndarray
    tree: cKDTree
    mins: np.ndarray
    maxs: np.ndarray
    hull: Optional[HullInfo] = None
    las_out: Optional[Path] = None
    ply_out: Optional[Path] = None


@dataclass
class FileMaskSource:
    points: np.ndarray
    tree: cKDTree
    field_names: List[str]
    field_arrays: Dict[str, np.ndarray]
    raw_xy_bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y) computed BEFORE voxel downsampling


def _no_mask_fill(dtype: np.dtype) -> int:
    return 0 if np.issubdtype(dtype, np.unsignedinteger) else -1


def _field_dtype(header: laspy.LasHeader, field_name: str) -> np.dtype:
    dim_names = list(header.point_format.dimension_names)
    if field_name in dim_names:
        dim = header.point_format.dimension_by_name(field_name)
        if dim.dtype is not None:
            return np.dtype(dim.dtype)
        # Bit-packed sub-fields (e.g. classification) report no scalar dtype; laspy
        # unpacks them into the smallest unsigned int that fits num_bits.
        for nbytes in (1, 2, 4, 8):
            if dim.num_bits <= nbytes * 8:
                return np.dtype(f"u{nbytes}")
        return np.dtype(np.uint64)
    for extra in header.point_format.extra_dimensions:
        if extra.name == field_name:
            return np.dtype(extra.dtype)
    raise SystemExit(
        f"Field '{field_name}' not found in mask file. Available dimensions: {', '.join(dim_names)}"
    )


def load_file_mask(
    path: Path,
    fields: List[str],
    distance: float,
    chunk_size: int,
    mask_voxel_size: float,
    fast_index_build: bool,
) -> FileMaskSource:
    with laspy.open(path) as src:
        N = int(src.header.point_count)
        xyz_buf = np.empty((N, 3), dtype=np.float32)
        field_bufs = {f: np.empty(N, dtype=_field_dtype(src.header, f)) for f in fields}
        offset = 0
        for chunk in src.chunk_iterator(chunk_size):
            n = len(chunk.x)
            xyz_buf[offset:offset + n, 0] = chunk.x
            xyz_buf[offset:offset + n, 1] = chunk.y
            xyz_buf[offset:offset + n, 2] = chunk.z
            for f in fields:
                field_bufs[f][offset:offset + n] = np.asarray(getattr(chunk, f))
            offset += n

    raw_xy_bounds = (
        float(xyz_buf[:offset, 0].min()),
        float(xyz_buf[:offset, 1].min()),
        float(xyz_buf[:offset, 0].max()),
        float(xyz_buf[:offset, 1].max()),
    )

    if mask_voxel_size > 0:
        vox = np.floor(xyz_buf / mask_voxel_size).astype(np.int64)
        _, idx = np.unique(vox, axis=0, return_index=True)
        idx = np.sort(idx)
        xyz_buf = xyz_buf[idx]
        for f in fields:
            field_bufs[f] = field_bufs[f][idx]

    tree = cKDTree(
        xyz_buf,
        leafsize=64,
        compact_nodes=not fast_index_build,
        balanced_tree=not fast_index_build,
    )
    return FileMaskSource(points=xyz_buf, tree=tree, field_names=fields, field_arrays=field_bufs, raw_xy_bounds=raw_xy_bounds)


def assign_fields_for_chunk(
    chunk_xyz: np.ndarray,
    source: FileMaskSource,
    distance: float,
    query_workers: int,
) -> Tuple[Dict[str, np.ndarray], int]:
    n = chunk_xyz.shape[0]
    dists, idxs = source.tree.query(
        chunk_xyz, k=1, distance_upper_bound=distance, workers=query_workers
    )
    matched = dists != np.inf
    match_count = int(np.count_nonzero(matched))
    out: Dict[str, np.ndarray] = {}
    for f in source.field_names:
        arr = np.zeros(n, dtype=source.field_arrays[f].dtype)
        if match_count > 0:
            arr[matched] = source.field_arrays[f][idxs[matched]]
        out[f] = arr
    return out, match_count


def build_mask_record(
    tree_id: int,
    stem_id: int,
    has_stem_id: bool,
    point_sets: List[np.ndarray],
    distance: float,
    use_hull_fill: bool,
    vox_mul: float,
    out_dir: Path,
    las_suffix: str,
    fast_index_build: bool,
) -> MaskRecord:
    pts = np.vstack(point_sets).astype(np.float32, copy=False)
    tree = cKDTree(
        pts,
        leafsize=64,
        compact_nodes=not fast_index_build,
        balanced_tree=not fast_index_build,
    )
    mins = np.min(pts, axis=0).astype(np.float32) - float(distance)
    maxs = np.max(pts, axis=0).astype(np.float32) + float(distance)
    key_name = f"{tree_id}_{stem_id}" if has_stem_id else f"{tree_id}"

    hull = None
    if use_hull_fill:
        hull = build_hull_info(pts, float(distance), vox_mul)

    return MaskRecord(
        tree_id=tree_id,
        stem_id=stem_id,
        has_stem_id=has_stem_id,
        points=pts,
        tree=tree,
        mins=mins,
        maxs=maxs,
        hull=hull,
        ply_out=out_dir / f"{key_name}.ply",
    )


class PlyStructAppender:
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

    def append(self, recs: np.ndarray) -> None:
        if recs is None or recs.size == 0:
            return
        if recs.dtype != self.dtype:
            recs = recs.astype(self.dtype, copy=False)
        recs = np.ascontiguousarray(recs)
        self.tmp_f.write(recs.tobytes())
        self.count += recs.shape[0]

    def close(self) -> None:
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

    def finalize(self) -> None:
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

    def discard(self) -> None:
        self.close()
        for p in (self.tmp_path, self.out_path):
            try:
                os.remove(p)
            except OSError:
                pass


def build_hull_info(points: np.ndarray, distance: float, vox_mul: float) -> Optional[HullInfo]:
    if points.shape[0] < 8:
        return None

    grid = float(distance) * float(vox_mul)
    if grid <= 0:
        return None

    mins = np.min(points, axis=0)
    vox = np.floor((points - mins) / grid).astype(np.int32)
    # One representative per voxel keeps support size bounded with low overhead.
    _, idx = np.unique(vox, axis=0, return_index=True)
    if idx.size == 0:
        return None

    support = points[np.sort(idx)]
    if support.shape[0] < 4:
        return None

    try:
        hull = ConvexHull(support, qhull_options="QJ")
    except QhullError:
        return None

    hull_pts = support[hull.vertices]
    if hull_pts.shape[0] < 4:
        return None

    # Delaunay removed: hull half-space equations enable fast vectorized membership tests.
    A = hull.equations[:, :-1].astype(np.float32, copy=False)
    b = hull.equations[:, -1].astype(np.float32, copy=False)

    return HullInfo(
        mins=np.min(hull_pts, axis=0).astype(np.float32),
        maxs=np.max(hull_pts, axis=0).astype(np.float32),
        A=A,
        b=b,
        hull_points=hull_pts.astype(np.float32, copy=False),
    )


def compute_crop_bounds_file(source: FileMaskSource) -> Tuple[float, float, float, float]:
    return (
        float(source.raw_xy_bounds[0]),
        float(source.raw_xy_bounds[1]),
        float(source.raw_xy_bounds[2]),
        float(source.raw_xy_bounds[3]),
    )


def compute_crop_bounds_folder(masks: List[MaskRecord]) -> Tuple[float, float, float, float]:
    raw_min_x = float(min(float(m.points[:, 0].min()) for m in masks))
    raw_min_y = float(min(float(m.points[:, 1].min()) for m in masks))
    raw_max_x = float(max(float(m.points[:, 0].max()) for m in masks))
    raw_max_y = float(max(float(m.points[:, 1].max()) for m in masks))
    crop_min_x = float(np.floor((raw_min_x - 10.0) / 10.0) * 10.0)
    crop_min_y = float(np.floor((raw_min_y - 10.0) / 10.0) * 10.0)
    crop_max_x = float(np.ceil((raw_max_x + 10.0) / 10.0) * 10.0)
    crop_max_y = float(np.ceil((raw_max_y + 10.0) / 10.0) * 10.0)
    return (crop_min_x, crop_min_y, crop_max_x, crop_max_y)


def load_masks(
    mask_folder: Path,
    distance: float,
    use_hull_fill: bool,
    vox_mul: float,
    out_dir: Path,
    las_suffix: str,
    load_chunk_size: int,
    index_workers: int,
    index_workers_auto: bool,
    fast_index_build: bool,
) -> List[MaskRecord]:
    grouped: Dict[Tuple[int, int], List[np.ndarray]] = {}
    grouped_has_stem: Dict[Tuple[int, int], bool] = {}
    mask_files = list_point_files(mask_folder)
    if not mask_files:
        raise SystemExit("No mask files found.")

    stage(f"Loading {len(mask_files)} mask file(s)")
    for path in tqdm(mask_files, unit="mask", desc="Load masks", **PROGRESS_KW):
        try:
            tree_id, stem_id, has_stem_id = parse_mask_ids(path.stem)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        read_desc = f"Read {path.name}" if path.suffix.lower() in (".las", ".laz") else None
        pts = get_points(path, chunk_size=load_chunk_size, progress_desc=read_desc)
        if pts.shape[0] == 0:
            continue
        key = (tree_id, stem_id)
        grouped.setdefault(key, []).append(pts)
        grouped_has_stem[key] = has_stem_id

    if not grouped:
        raise SystemExit("No usable mask points found.")

    grouped_items = sorted(grouped.items())
    total_mask_points = int(sum(sum(arr.shape[0] for arr in point_sets) for point_sets in grouped.values()))
    stage(
        f"Building spatial indices for {len(grouped_items)} unique mask id pair(s) "
        f"from {total_mask_points} point(s)"
    )
    workers = max(1, int(index_workers))
    workers = min(workers, len(grouped_items))

    if index_workers_auto and workers > 1:
        avg_points_per_mask = total_mask_points / max(1, len(grouped_items))

        # KD-tree construction is typically memory-bandwidth bound for large masks.
        # In those cases, many build threads can be slower than serial.
        if len(grouped_items) < 16 or avg_points_per_mask >= 500000:
            stage(
                "Auto index worker policy selected serial build "
                f"(masks={len(grouped_items)}, avg_points_per_mask={int(avg_points_per_mask)})"
            )
            workers = 1
        else:
            auto_cap = min(workers, 8)
            if auto_cap != workers:
                stage(f"Auto index worker policy capped workers to {auto_cap} to avoid oversubscription")
            workers = auto_cap

    if workers > 1:
        stage(f"Parallel mask index build enabled with {workers} worker(s)")
    if fast_index_build:
        stage("Fast index build enabled (faster startup, potentially slower nearest-neighbor queries)")

    masks: List[MaskRecord] = []
    if workers == 1:
        for (tree_id, stem_id), point_sets in tqdm(
            grouped_items,
            unit="mask",
            desc="Build mask trees",
            **PROGRESS_KW,
        ):
            masks.append(
                build_mask_record(
                    tree_id=tree_id,
                    stem_id=stem_id,
                    has_stem_id=grouped_has_stem[(tree_id, stem_id)],
                    point_sets=point_sets,
                    distance=distance,
                    use_hull_fill=use_hull_fill,
                    vox_mul=vox_mul,
                    out_dir=out_dir,
                    las_suffix=las_suffix,
                    fast_index_build=fast_index_build,
                )
            )
    else:
        futures: Dict[concurrent.futures.Future[MaskRecord], Tuple[int, int]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for (tree_id, stem_id), point_sets in grouped_items:
                fut = pool.submit(
                    build_mask_record,
                    tree_id=tree_id,
                    stem_id=stem_id,
                    has_stem_id=grouped_has_stem[(tree_id, stem_id)],
                    point_sets=point_sets,
                    distance=distance,
                    use_hull_fill=use_hull_fill,
                    vox_mul=vox_mul,
                    out_dir=out_dir,
                    las_suffix=las_suffix,
                    fast_index_build=fast_index_build,
                )
                futures[fut] = (tree_id, stem_id)

            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                unit="mask",
                desc="Build mask trees",
                **PROGRESS_KW,
            ):
                masks.append(fut.result())

        masks.sort(key=lambda m: (m.tree_id, m.stem_id))

    stage(f"Using {len(masks)} unique mask id pair(s)")
    return masks


def resolve_index_workers(index_workers_raw: str) -> int:
    if index_workers_raw is None:
        index_workers_raw = "auto"
    raw = str(index_workers_raw).strip().lower()
    if raw == "auto":
        slurm_raw = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
        if slurm_raw:
            try:
                slurm_cpus = int(slurm_raw)
                if slurm_cpus > 0:
                    return slurm_cpus
            except ValueError:
                pass

        cpu_count = os.cpu_count() or 1
        return max(1, int(cpu_count))

    try:
        workers = int(raw)
    except ValueError as exc:
        raise SystemExit("--index_workers must be a positive integer or 'auto'") from exc
    if workers <= 0:
        raise SystemExit("--index_workers must be > 0")
    return workers


def resolve_query_workers(query_workers_raw: str) -> int:
    if query_workers_raw is None:
        query_workers_raw = "auto"
    raw = str(query_workers_raw).strip().lower()
    if raw == "auto":
        slurm_raw = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
        if slurm_raw:
            try:
                slurm_cpus = int(slurm_raw)
                if slurm_cpus > 0:
                    return slurm_cpus
            except ValueError:
                pass
        cpu_count = os.cpu_count() or 1
        return max(1, int(cpu_count))

    try:
        workers = int(raw)
    except ValueError as exc:
        raise SystemExit("--query_workers must be a positive integer or 'auto'") from exc
    if workers <= 0:
        raise SystemExit("--query_workers must be > 0")
    return workers


def process_chunk(
    chunk_xyz: np.ndarray,
    masks: List[MaskRecord],
    distance: float,
    use_hull_fill: bool,
    hull_eps: float,
    query_workers: int,
) -> np.ndarray:
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
        dists, _ = mask.tree.query(candidates, k=1, distance_upper_bound=distance, workers=query_workers)
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
        # Half-space checks avoid per-point simplex tests and extra KD boundary queries.
        plane_eval = test_pts @ hull.A.T + hull.b
        dist_planes = np.max(plane_eval, axis=1)
        inside = dist_planes <= 0.0
        near = dist_planes <= float(hull_eps)
        accepted = near
        if not np.any(accepted):
            continue

        local_idx = np.where(in_hull_box)[0]
        accepted_idx = local_idx[accepted]
        accepted_score = np.maximum(dist_planes[accepted], 0.0).astype(np.float32, copy=False)
        better = accepted_score < hull_best[accepted_idx]
        if np.any(better):
            widx = accepted_idx[better]
            matched_mask_idx[widx] = mask_idx
            hull_best[widx] = accepted_score[better]

    return matched_mask_idx


def build_las_header_with_ids(src_header: laspy.LasHeader, write_stem_id: bool = True) -> laspy.LasHeader:
    out_header = src_header.copy()
    dim_names = set(out_header.point_format.dimension_names)
    if "tree_id" not in dim_names:
        out_header.add_extra_dim(ExtraBytesParams(name="tree_id", type=np.int32))
    if write_stem_id and "stem_id" not in dim_names:
        out_header.add_extra_dim(ExtraBytesParams(name="stem_id", type=np.int32))
    return out_header


def make_point_record_with_ids(
    chunk_subset,
    out_header: laspy.LasHeader,
    src_dim_names: List[str],
    tree_id,
    stem_id,
    write_stem_id: bool = True,
):
    rec = ScaleAwarePointRecord.zeros(len(chunk_subset), header=out_header)
    for dim in src_dim_names:
        if dim in ("tree_id", "stem_id"):
            continue
        if hasattr(chunk_subset, dim):
            setattr(rec, dim, np.asarray(getattr(chunk_subset, dim)))
    if np.isscalar(tree_id):
        rec.tree_id = np.full(len(chunk_subset), tree_id, dtype=np.int32)
    else:
        rec.tree_id = np.asarray(tree_id, dtype=np.int32)
    if write_stem_id:
        if np.isscalar(stem_id):
            rec.stem_id = np.full(len(chunk_subset), stem_id, dtype=np.int32)
        else:
            rec.stem_id = np.asarray(stem_id, dtype=np.int32)
    return rec


def build_las_header_with_fields(
    src_header: laspy.LasHeader,
    field_names: List[str],
    field_dtypes: Dict[str, np.dtype],
) -> laspy.LasHeader:
    out_header = src_header.copy()
    dim_names = set(out_header.point_format.dimension_names)
    for f in field_names:
        if f not in dim_names:
            out_header.add_extra_dim(ExtraBytesParams(name=f, type=field_dtypes[f]))
    return out_header


def make_point_record_with_fields(
    chunk_subset,
    out_header: laspy.LasHeader,
    src_dim_names: List[str],
    field_value_arrays: Dict[str, np.ndarray],
):
    rec = ScaleAwarePointRecord.zeros(len(chunk_subset), header=out_header)
    for dim in src_dim_names:
        if dim in field_value_arrays:
            continue
        if hasattr(chunk_subset, dim):
            setattr(rec, dim, np.asarray(getattr(chunk_subset, dim)))
    for f in field_value_arrays:
        setattr(rec, f, np.asarray(field_value_arrays[f]))
    return rec


def project_top(points: np.ndarray) -> np.ndarray:
    return points[:, [0, 1]]


def project_front(points: np.ndarray) -> np.ndarray:
    return points[:, [0, 2]]


def project_aerial(points: np.ndarray) -> np.ndarray:
    x = points[:, 0] + 0.35 * points[:, 1]
    y = points[:, 2] - 0.25 * points[:, 1]
    return np.column_stack((x, y))


def bbox_corners(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [mins[0], mins[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=np.float64,
    )


def capped_voxel_sample(
    files: List[Path],
    chunk_size: int,
    voxel_size: float,
    max_points: int,
) -> np.ndarray:
    kept: List[np.ndarray] = []
    total_kept = 0

    for path in files:
        for pts in iter_points(path, chunk_size):
            if voxel_size > 0:
                vox = np.floor(pts / voxel_size).astype(np.int64)
                _, idx = np.unique(vox, axis=0, return_index=True)
                pts = pts[np.sort(idx)]

            if pts.shape[0] == 0:
                continue

            room = max_points - total_kept
            if room <= 0:
                break
            if pts.shape[0] > room:
                pick = np.linspace(0, pts.shape[0] - 1, num=room, dtype=np.int64)
                pts = pts[pick]

            kept.append(pts.astype(np.float32, copy=False))
            total_kept += pts.shape[0]

        if total_kept >= max_points:
            break

    if not kept:
        return np.empty((0, 3), dtype=np.float32)
    return np.vstack(kept)


def sample_points_array(points: np.ndarray, voxel_size: float, max_points: int) -> np.ndarray:
    pts = points.astype(np.float32, copy=False)
    if pts.shape[0] == 0:
        return pts

    if voxel_size > 0:
        vox = np.floor(pts / voxel_size).astype(np.int64)
        _, idx = np.unique(vox, axis=0, return_index=True)
        pts = pts[np.sort(idx)]

    if pts.shape[0] > max_points:
        pick = np.linspace(0, pts.shape[0] - 1, num=max_points, dtype=np.int64)
        pts = pts[pick]

    return pts.astype(np.float32, copy=False)


def subtract_exact_points(points: np.ndarray, remove: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0 or remove.shape[0] == 0:
        return points

    pts = np.ascontiguousarray(points.astype(np.float32, copy=False))
    rem = np.ascontiguousarray(remove.astype(np.float32, copy=False))
    row_dtype = np.dtype((np.void, pts.dtype.itemsize * pts.shape[1]))
    pts_view = pts.view(row_dtype).ravel()
    rem_view = rem.view(row_dtype).ravel()
    keep = ~np.isin(pts_view, rem_view)
    return pts[keep]


def _scatter_compare_view(
    ax,
    title: str,
    background_pts: np.ndarray,
    input_pts: np.ndarray,
    output_pts: np.ndarray,
    proj_fn,
    hull_points: Optional[np.ndarray] = None,
    xlim=None,
    ylim=None,
) -> None:
    if background_pts.shape[0] > 0:
        p = proj_fn(background_pts)
        ax.scatter(p[:, 0], p[:, 1], s=1.1, c="#000000", alpha=0.25, linewidths=0)
    if input_pts.shape[0] > 0:
        p = proj_fn(input_pts)
        ax.scatter(p[:, 0], p[:, 1], s=1.3, c="#2563eb", alpha=0.18, linewidths=0)
    if output_pts.shape[0] > 0:
        p = proj_fn(output_pts)
        ax.scatter(p[:, 0], p[:, 1], s=1.3, c="#f97316", alpha=0.18, linewidths=0)

    if hull_points is not None and hull_points.shape[0] >= 3:
        hp = proj_fn(hull_points)
        if hp.shape[0] >= 3:
            try:
                hp_hull = ConvexHull(hp)
                poly = hp[hp_hull.vertices]
                if poly.shape[0] >= 3:
                    poly = np.vstack((poly, poly[0]))
                    ax.plot(poly[:, 0], poly[:, 1], color="#e11d48", linewidth=1.2, alpha=0.95)
            except QhullError:
                pass

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def _overlay_bounds(
    background_pts: np.ndarray,
    input_pts: np.ndarray,
    output_pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    arrays = [arr for arr in (background_pts, input_pts, output_pts) if arr.shape[0] > 0]
    if arrays:
        combined = np.vstack(arrays)
    else:
        combined = np.empty((0, 3), dtype=np.float32)

    if combined.shape[0] == 0:
        zeros = np.zeros(3, dtype=np.float32)
        return zeros, zeros
    return np.min(combined, axis=0), np.max(combined, axis=0)


def generate_qc_plots(
    masks: List[MaskRecord],
    out_dir: Path,
    rng_seed: int,
    count: int,
    target_plot_points: np.ndarray,
    mask_plot_max_points: int,
    plot_voxel_size: float,
    show_hull_outline: bool,
    selection_state_path: Optional[Path],
    debug_settings: Optional[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[str], bool]:
    if plt is None:
        raise RuntimeError("matplotlib is required for QC plots. Install matplotlib and rerun.")
    if not masks:
        return [], [], False

    chosen: List[MaskRecord] = []
    reused_selection = False
    candidate_map = {f"{m.tree_id}_{m.stem_id}": m for m in masks}

    if selection_state_path is not None and debug_settings is not None and selection_state_path.exists():
        try:
            state = json.loads(selection_state_path.read_text(encoding="utf-8"))
            if state.get("debug_settings") == debug_settings:
                selected_keys = state.get("selected_masks", [])
                if isinstance(selected_keys, list) and selected_keys:
                    restored = [candidate_map[key] for key in selected_keys if key in candidate_map]
                    if len(restored) == len(selected_keys):
                        chosen = restored
                        reused_selection = True
        except Exception:
            pass

    if not chosen:
        rng = random.Random(rng_seed)
        n_pick = min(int(count), len(masks))
        chosen = rng.sample(masks, n_pick)
        if selection_state_path is not None and debug_settings is not None:
            selection_state_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "debug_settings": debug_settings,
                "selected_masks": [f"{m.tree_id}_{m.stem_id}" for m in chosen],
            }
            selection_state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    plot_dir = out_dir / "qc_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    per_mask_stats: List[Dict[str, object]] = []
    overview_data: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]] = []

    for mask in chosen:
        input_pts = sample_points_array(mask.points, plot_voxel_size, mask_plot_max_points)
        mask_bbox_mins = np.min(mask.points, axis=0)
        mask_bbox_maxs = np.max(mask.points, axis=0)
        target_in_mask_box = target_plot_points[
            np.all((target_plot_points >= mask_bbox_mins) & (target_plot_points <= mask_bbox_maxs), axis=1)
        ]
        output_pts = np.empty((0, 3), dtype=np.float32)
        if mask.ply_out is not None and mask.ply_out.exists():
            output_pts = capped_voxel_sample(
                files=[mask.ply_out],
                chunk_size=max(1, mask_plot_max_points),
                voxel_size=plot_voxel_size,
                max_points=mask_plot_max_points,
            )

        background_pts = subtract_exact_points(target_in_mask_box, output_pts)

        hull_pts = mask.hull.hull_points if show_hull_outline and mask.hull is not None else None

        mins, maxs = _overlay_bounds(background_pts, input_pts, output_pts)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=180)
        _scatter_compare_view(
            axes[0],
            "Top (XY)",
            background_pts,
            input_pts,
            output_pts,
            project_top,
            hull_points=hull_pts,
            xlim=(float(mins[0]), float(maxs[0])),
            ylim=(float(mins[1]), float(maxs[1])),
        )
        _scatter_compare_view(
            axes[1],
            "Front (XZ)",
            background_pts,
            input_pts,
            output_pts,
            project_front,
            hull_points=hull_pts,
            xlim=(float(mins[0]), float(maxs[0])),
            ylim=(float(mins[2]), float(maxs[2])),
        )
        corners = bbox_corners(mins, maxs)
        ac = project_aerial(corners)
        _scatter_compare_view(
            axes[2],
            "Aerial (Oblique)",
            background_pts,
            input_pts,
            output_pts,
            project_aerial,
            hull_points=hull_pts,
            xlim=(float(np.min(ac[:, 0])), float(np.max(ac[:, 0]))),
            ylim=(float(np.min(ac[:, 1])), float(np.max(ac[:, 1]))),
        )
        fig.suptitle(
            f"Mask {mask.tree_id}_{mask.stem_id} | black=other target | blue=input mask | orange=output mask",
            fontsize=11,
        )
        fig.tight_layout()
        fig_path = plot_dir / f"{mask.tree_id}_{mask.stem_id}_views.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        overview_data.append((f"{mask.tree_id}_{mask.stem_id}", background_pts, input_pts, output_pts, hull_pts))
        per_mask_stats.append(
            {
                "mask_file": f"{mask.tree_id}_{mask.stem_id}",
                "plot_path": str(fig_path),
                "background_points_sampled": int(background_pts.shape[0]),
                "input_points_sampled": int(input_pts.shape[0]),
                "output_points_sampled": int(output_pts.shape[0]),
            }
        )

    if overview_data:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=180)
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            if i >= len(overview_data):
                ax.axis("off")
                continue
            name, background_pts, input_pts, output_pts, hull_pts = overview_data[i]
            _scatter_compare_view(
                ax,
                f"{name} (Aerial)",
                background_pts,
                input_pts,
                output_pts,
                project_aerial,
                hull_points=hull_pts,
            )
        fig.suptitle(
            "QC sample (2x2): black=other target | blue=input mask | orange=output mask",
            fontsize=12,
        )
        fig.tight_layout()
        grid_path = plot_dir / "qc_overview_2x2.png"
        fig.savefig(grid_path, bbox_inches="tight")
        plt.close(fig)

    return per_mask_stats, [f"{m.tree_id}_{m.stem_id}" for m in chosen], reused_selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract points per mask id and write LAS/LAZ with IDs and/or per-mask PLY."
    )
    parser.add_argument(
        "-m",
        "--mask",
        required=True,
        type=Path,
        help="Folder containing mask files, or a single LAS/LAZ mask file",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default="tree_id,stem_id",
        help="Comma-separated fields to transfer in single-file mask mode (default: tree_id,stem_id)",
    )
    parser.add_argument(
        "--mask_voxel_size",
        type=float,
        default=0.0,
        help="Voxel-downsample the single-file mask before building the KD-tree (default 0 = off)",
    )
    parser.add_argument("-t", "--target", required=True, type=Path, help="Target LAS/LAZ/PLY file or folder")
    parser.add_argument("-o", "--output", type=Path, help="Output directory")
    parser.add_argument("-d", "--distance", type=float, required=True, help="Distance threshold for primary match")
    parser.add_argument("--chunk-size", type=int, default=500000, help="Points per chunk (default: 500,000)")
    parser.add_argument(
        "--ids_only",
        action="store_true",
        help="Only write LAS/LAZ outputs with tree_id and stem_id (skip PLY outputs)",
    )
    parser.add_argument(
        "--ply_only",
        action="store_true",
        help="Only write PLY outputs (skip LAS/LAZ outputs)",
    )
    parser.add_argument(
        "--hull_fill",
        action="store_true",
        help="Enable secondary hull-based inclusion for unmatched points",
    )
    parser.add_argument(
        "--las_out",
        action="store_true",
        help="Save output as uncompressed .las instead of .laz (default: .laz)",
    )
    parser.add_argument(
        "--vox_mul",
        type=float,
        default=3.0,
        help="Voxel multiplier for hull support grid size: distance * vox_mul (default 3)",
    )
    parser.add_argument(
        "--hull_eps",
        type=float,
        help="Small epsilon distance used for near-hull inclusion (default --distance if not set)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug QC plots after mask assignment")
    parser.add_argument("--plot_count", type=int, default=4, help="Number of random masks to plot for QC")
    parser.add_argument("--plot_seed", type=int, default=42, help="Random seed for selecting QC masks")
    parser.add_argument(
        "--plot_match_distance",
        type=float,
        default=None,
        help="Match threshold for QC plot coloring (default: --distance)",
    )
    parser.add_argument(
        "--plot_target_max_points",
        type=int,
        default=300000,
        help="Max target points sampled for QC plots",
    )
    parser.add_argument(
        "--plot_mask_max_points",
        type=int,
        default=120000,
        help="Max mask points sampled per QC plot",
    )
    parser.add_argument(
        "--plot_voxel_size",
        type=float,
        default=0.10,
        help="Voxel size for target/mask sampling during QC plots",
    )
    parser.add_argument(
        "--index_workers",
        type=str,
        default="auto",
        help="Workers for building mask KD-tree/hull indices (positive integer or 'auto'; default auto-detects CPU count or SLURM allocation)",
    )
    parser.add_argument(
        "--query_workers",
        type=str,
        default="auto",
        help="Workers for nearest-neighbor queries during assignment (positive integer or 'auto'; default auto-detects CPU count or SLURM allocation)",
    )
    parser.add_argument(
        "--fast_index_build",
        action="store_true",
        help="Build less-balanced KD-trees to reduce startup time (query phase may be slower)",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Crop target points to the XY bounding box of the mask(s) before assignment",
    )
    args = parser.parse_args()

    args.mask = normalize_cli_path(args.mask, must_exist=True)
    args.target = normalize_cli_path(args.target, must_exist=True)
    if args.output is not None:
        args.output = normalize_cli_path(args.output, must_exist=False)

    mask_is_file = args.mask.is_file()
    mask_is_folder = args.mask.is_dir()
    if not mask_is_file and not mask_is_folder:
        raise SystemExit("Invalid --mask path. Must be a folder or a single LAS/LAZ file")

    field_names = [f.strip() for f in args.fields.split(",") if f.strip()]

    if mask_is_file:
        if args.mask.suffix.lower() not in (".las", ".laz"):
            raise SystemExit("Single-file --mask must be a .las or .laz file")
        if not field_names:
            raise SystemExit("--fields may not be empty")
    else:
        if args.fields != "tree_id,stem_id":
            stage("Warning: --fields is ignored in folder mask mode")
        if args.mask_voxel_size > 0:
            stage("Warning: --mask_voxel_size is ignored in folder mask mode")

    if args.ids_only and args.ply_only:
        raise SystemExit("Choose only one of --ids-only or --ply-only")
    if args.distance <= 0:
        raise SystemExit("--distance must be > 0")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be > 0")
    if args.plot_count < 0:
        raise SystemExit("--plot_count must be >= 0")
    if args.plot_target_max_points < 1000 or args.plot_mask_max_points < 1000:
        raise SystemExit("--plot_target_max_points and --plot_mask_max_points must be >= 1000")
    if args.plot_voxel_size <= 0:
        raise SystemExit("--plot_voxel_size must be > 0")
    if not args.hull_eps:
        args.hull_eps = args.distance

    index_workers_auto = args.index_workers is None or str(args.index_workers).strip().lower() == "auto"
    args.index_workers = resolve_index_workers(args.index_workers)
    stage(
        f"Index workers resolved to {args.index_workers} "
        f"({'auto' if index_workers_auto else 'manual'})"
    )
    args.query_workers = resolve_query_workers(args.query_workers)
    stage(f"Query workers resolved to {args.query_workers}")

    write_las = not args.ply_only
    write_ply = not args.ids_only

    out_dir = args.output if args.output else Path(f"{args.mask.name}_extracted")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.target.is_dir():
        target_files = list_point_files(args.target)
    elif args.target.is_file() and args.target.suffix.lower() in SUPPORTED_EXTS:
        target_files = [args.target]
    else:
        raise SystemExit("Invalid target path. Must be a file or directory with .ply/.las/.laz")
    if not target_files:
        raise SystemExit("No target files found")

    las_targets = [p for p in target_files if p.suffix.lower() in (".las", ".laz")]
    if write_las and not las_targets:
        raise SystemExit("LAS/LAZ output requested but no LAS/LAZ target inputs were provided")
    if args.las_out:
        las_suffix = ".las"
    else:
        las_suffix = ".laz" if any(p.suffix.lower() == ".laz" for p in las_targets) else ".las"

    if mask_is_file:
        if args.hull_fill:
            raise SystemExit("--hull_fill is not supported with a single-file mask")
        if args.ply_only:
            raise SystemExit("--ply_only is not supported with a single-file mask")
        if any(p.suffix.lower() == ".ply" for p in target_files):
            raise SystemExit("Single-file mask mode supports only LAS/LAZ targets, not PLY")
        if args.debug:
            stage("Warning: --debug is ignored in single-file mask mode (QC plots require folder mode)")
            args.debug = False

        with laspy.open(args.mask) as mask_src:
            field_dtypes = {f: _field_dtype(mask_src.header, f) for f in field_names}

        stage(f"Loading single-file mask: {args.mask.name}")
        file_mask_source = load_file_mask(
            path=args.mask,
            fields=field_names,
            distance=args.distance,
            chunk_size=args.chunk_size,
            mask_voxel_size=args.mask_voxel_size,
            fast_index_build=args.fast_index_build,
        )

        crop_xy: Optional[Tuple[float, float, float, float]] = None
        if args.crop:
            crop_xy = compute_crop_bounds_file(file_mask_source)
            stage(
                f"Crop enabled: XY min=({crop_xy[0]:.3f}, {crop_xy[1]:.3f})  "
                f"max=({crop_xy[2]:.3f}, {crop_xy[3]:.3f})"
            )

        stage(f"Processing {len(target_files)} target file(s)")
        stage(f"Primary assignment distance: {args.distance}")
        stage(f"Transferring fields: {', '.join(field_names)}")

        total_points = 0
        assigned_points = 0
        unbound_points = 0
        chunks_seen = 0
        last_heartbeat_t = time.time()
        start_time = time.time()
        las_output_paths: List[Path] = []

        for target_file in target_files:
            stage(f"Target: {target_file.name}")
            with contextlib.ExitStack() as stack:
                src = stack.enter_context(laspy.open(target_file))
                src_dim_names = list(src.header.point_format.dimension_names)
                has_bound_field = "bound" in src_dim_names
                out_header = build_las_header_with_fields(src.header, field_names, field_dtypes)

                masked_out = out_dir / f"{target_file.stem}_masked{las_suffix}"
                if masked_out.exists():
                    masked_out.unlink()
                las_output_paths.append(masked_out)
                las_writer = stack.enter_context(laspy.open(masked_out, mode="w", header=out_header))

                with tqdm(
                    total=int(src.header.point_count),
                    unit="pts",
                    desc=f"Assign {target_file.name}",
                    **PROGRESS_KW,
                ) as pbar_assign:
                    for chunk in src.chunk_iterator(args.chunk_size):
                        n_pts_scanned = int(len(chunk.x))
                        if n_pts_scanned == 0:
                            continue
                        total_points += n_pts_scanned

                        # Split unbound (bound == 0) from bound immediately.
                        # Unbound points stream straight to output; only bound points
                        # go through crop and KD-tree assignment.
                        if has_bound_field:
                            is_unbound = np.asarray(chunk.bound) == 0
                            unbound_chunk = chunk[is_unbound]
                            bound_chunk   = chunk[~is_unbound]
                        else:
                            unbound_chunk = None
                            bound_chunk   = chunk

                        # Stream unbound points directly to output with fill values
                        if unbound_chunk is not None and len(unbound_chunk.x) > 0:
                            n_unbound = len(unbound_chunk.x)
                            fill_vals = {
                                f: np.full(n_unbound, _no_mask_fill(file_mask_source.field_arrays[f].dtype), dtype=file_mask_source.field_arrays[f].dtype)
                                for f in file_mask_source.field_names
                            }
                            las_writer.write_points(make_point_record_with_fields(
                                chunk_subset=unbound_chunk,
                                out_header=out_header,
                                src_dim_names=src_dim_names,
                                field_value_arrays=fill_vals,
                            ))
                            unbound_points += n_unbound

                        # Crop and mask bound points
                        n_matched = 0
                        if len(bound_chunk.x) > 0:
                            bound_xyz = np.vstack((bound_chunk.x, bound_chunk.y, bound_chunk.z)).T.astype(np.float32, copy=False)
                            if crop_xy is not None:
                                in_crop = (
                                    (bound_xyz[:, 0] >= crop_xy[0]) & (bound_xyz[:, 0] <= crop_xy[2]) &
                                    (bound_xyz[:, 1] >= crop_xy[1]) & (bound_xyz[:, 1] <= crop_xy[3])
                                )
                                bound_xyz  = bound_xyz[in_crop]
                                bound_chunk = bound_chunk[in_crop]
                            if len(bound_chunk.x) > 0:
                                field_vals, n_matched = assign_fields_for_chunk(
                                    bound_xyz,
                                    file_mask_source,
                                    args.distance,
                                    query_workers=args.query_workers,
                                )
                                las_writer.write_points(make_point_record_with_fields(
                                    chunk_subset=bound_chunk,
                                    out_header=out_header,
                                    src_dim_names=src_dim_names,
                                    field_value_arrays=field_vals,
                                ))

                        chunks_seen += 1
                        assigned_points += n_matched

                        pbar_assign.update(n_pts_scanned)

                        now_t = time.time()
                        if now_t - last_heartbeat_t >= 20.0:
                            stage(
                                f"Heartbeat: processed {chunks_seen} chunk(s), "
                                f"scanned {total_points} point(s), "
                                f"assigned {assigned_points}, unbound {unbound_points}"
                            )
                            last_heartbeat_t = now_t

        elapsed = time.time() - start_time
        stage(f"Done in {elapsed:.2f}s. Scanned {total_points} points; assigned {assigned_points}, unbound {unbound_points}.")
        stage("Wrote masked LAS/LAZ outputs: " + ", ".join(str(p) for p in las_output_paths))
        return

    masks = load_masks(
        mask_folder=args.mask,
        distance=args.distance,
        use_hull_fill=args.hull_fill,
        vox_mul=args.vox_mul,
        out_dir=out_dir,
        las_suffix=las_suffix,
        load_chunk_size=args.chunk_size,
        index_workers=args.index_workers,
        index_workers_auto=index_workers_auto,
        fast_index_build=args.fast_index_build,
    )

    crop_xy: Optional[Tuple[float, float, float, float]] = None
    if args.crop:
        crop_xy = compute_crop_bounds_folder(masks)
        stage(
            f"Crop enabled: XY min=({crop_xy[0]:.3f}, {crop_xy[1]:.3f})  "
            f"max=({crop_xy[2]:.3f}, {crop_xy[3]:.3f})"
        )

    write_stem_id = any(m.has_stem_id for m in masks)

    mask_tree_ids = np.asarray([m.tree_id for m in masks], dtype=np.int32)
    mask_stem_ids = np.asarray([m.stem_id for m in masks], dtype=np.int32)

    ply_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("tree_id", "<i4")]
    if write_stem_id:
        ply_fields.append(("stem_id", "<i4"))
    ply_dtype = np.dtype(ply_fields)
    ply_appenders: Dict[Tuple[int, int], PlyStructAppender] = {}
    if write_ply:
        for m in masks:
            if m.ply_out is not None:
                ply_appenders[(m.tree_id, m.stem_id)] = PlyStructAppender(m.ply_out, ply_dtype)

    stage(f"Processing {len(target_files)} target file(s)")
    stage(f"Primary assignment distance: {args.distance}")
    if args.hull_fill:
        stage(
            f"Hull fill enabled: support_spacing=distance({args.distance}), "
            f"vox_mul={args.vox_mul}, hull_eps={args.hull_eps}"
        )

    total_points = 0
    assigned_points = 0
    unbound_points = 0
    chunks_seen = 0
    last_heartbeat_t = time.time()
    start_time = time.time()
    las_output_paths: List[Path] = []

    for target_file in target_files:
        ext = target_file.suffix.lower()
        stage(f"Target: {target_file.name}")

        if ext in (".las", ".laz"):
            with contextlib.ExitStack() as stack:
                src = stack.enter_context(laspy.open(target_file))
                src_dim_names = list(src.header.point_format.dimension_names)
                has_bound_field = "bound" in src_dim_names
                out_header = build_las_header_with_ids(src.header, write_stem_id=write_stem_id) if write_las else None

                las_writer = None
                if write_las:
                    masked_out = out_dir / f"{target_file.stem}_masked{las_suffix}"
                    if masked_out.exists():
                        masked_out.unlink()
                    las_output_paths.append(masked_out)
                    las_writer = stack.enter_context(laspy.open(masked_out, mode="w", header=out_header))

                with tqdm(
                    total=int(src.header.point_count),
                    unit="pts",
                    desc=f"Assign {target_file.name}",
                    **PROGRESS_KW,
                ) as pbar_assign:
                    for chunk in src.chunk_iterator(args.chunk_size):
                        chunk_xyz = np.vstack((chunk.x, chunk.y, chunk.z)).T.astype(np.float32, copy=False)
                        if chunk_xyz.shape[0] == 0:
                            continue
                        n_pts_scanned = int(chunk_xyz.shape[0])
                        total_points += n_pts_scanned

                        # Split unbound (bound == 0) from bound immediately.
                        # Unbound points stream straight to output; only bound points
                        # go through crop and KD-tree assignment.
                        if has_bound_field:
                            is_unbound = np.asarray(chunk.bound) == 0
                            unbound_chunk = chunk[is_unbound]
                            bound_chunk = chunk[~is_unbound]
                            bound_xyz = chunk_xyz[~is_unbound]
                        else:
                            unbound_chunk = None
                            bound_chunk = chunk
                            bound_xyz = chunk_xyz

                        # Stream unbound points directly to output with sentinel IDs
                        if unbound_chunk is not None and len(unbound_chunk.x) > 0:
                            n_unbound = len(unbound_chunk.x)
                            if write_las and las_writer is not None:
                                rec = make_point_record_with_ids(
                                    chunk_subset=unbound_chunk,
                                    out_header=out_header,
                                    src_dim_names=src_dim_names,
                                    tree_id=np.full(n_unbound, -1, dtype=np.int32),
                                    stem_id=np.full(n_unbound, -1, dtype=np.int32),
                                    write_stem_id=write_stem_id,
                                )
                                las_writer.write_points(rec)
                            unbound_points += n_unbound

                        # Crop bound points
                        if crop_xy is not None and bound_xyz.shape[0] > 0:
                            in_crop = (
                                (bound_xyz[:, 0] >= crop_xy[0]) & (bound_xyz[:, 0] <= crop_xy[2]) &
                                (bound_xyz[:, 1] >= crop_xy[1]) & (bound_xyz[:, 1] <= crop_xy[3])
                            )
                            bound_xyz = bound_xyz[in_crop]
                            bound_chunk = bound_chunk[in_crop]

                        if bound_xyz.shape[0] > 0:
                            matched_mask_idx = process_chunk(
                                bound_xyz,
                                masks,
                                args.distance,
                                use_hull_fill=args.hull_fill,
                                hull_eps=args.hull_eps,
                                query_workers=args.query_workers,
                            )

                            matched_sel = matched_mask_idx >= 0
                            if write_las and las_writer is not None:
                                tree_ids = np.zeros(bound_xyz.shape[0], dtype=np.int32)
                                stem_ids = np.zeros(bound_xyz.shape[0], dtype=np.int32) if write_stem_id else None
                                if np.any(matched_sel):
                                    tree_ids[matched_sel] = mask_tree_ids[matched_mask_idx[matched_sel]]
                                    if write_stem_id:
                                        stem_ids[matched_sel] = mask_stem_ids[matched_mask_idx[matched_sel]]
                                rec = make_point_record_with_ids(
                                    chunk_subset=bound_chunk,
                                    out_header=out_header,
                                    src_dim_names=src_dim_names,
                                    tree_id=tree_ids,
                                    stem_id=stem_ids,
                                    write_stem_id=write_stem_id,
                                )
                                las_writer.write_points(rec)

                            matched_ids = np.unique(matched_mask_idx[matched_mask_idx >= 0])
                            for mask_idx in matched_ids:
                                mask = masks[int(mask_idx)]
                                sel = matched_mask_idx == mask_idx
                                if not np.any(sel):
                                    continue
                                sel_count = int(np.sum(sel))
                                assigned_points += sel_count

                                if write_ply:
                                    recs = np.zeros(sel_count, dtype=ply_dtype)
                                    pts = bound_xyz[sel]
                                    recs["x"] = pts[:, 0]
                                    recs["y"] = pts[:, 1]
                                    recs["z"] = pts[:, 2]
                                    recs["tree_id"] = mask.tree_id
                                    if write_stem_id:
                                        recs["stem_id"] = mask.stem_id
                                    ply_appenders[(mask.tree_id, mask.stem_id)].append(recs)

                        chunks_seen += 1
                        pbar_assign.update(n_pts_scanned)

                        now_t = time.time()
                        if now_t - last_heartbeat_t >= 20.0:
                            stage(
                                f"Heartbeat: processed {chunks_seen} chunk(s), "
                                f"scanned {total_points} point(s), "
                                f"assigned {assigned_points}, unbound {unbound_points}"
                            )
                            last_heartbeat_t = now_t

        elif ext == ".ply":
            with open(target_file, "rb") as f:
                ply = PlyData.read(f)
            pts = np.vstack((ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"])).T.astype(
                np.float32, copy=False
            )

            if write_las:
                raise SystemExit(
                    "LAS/LAZ output is enabled but target contains PLY input. Use --ply-only for PLY targets"
                )

            with tqdm(
                total=int(pts.shape[0]),
                unit="pts",
                desc=f"Assign {target_file.name}",
                **PROGRESS_KW,
            ) as pbar_assign:
                for start_idx in range(0, pts.shape[0], args.chunk_size):
                    end_idx = min(start_idx + args.chunk_size, pts.shape[0])
                    chunk_xyz = pts[start_idx:end_idx]
                    if chunk_xyz.shape[0] == 0:
                        continue
                    n_chunk_pts = int(chunk_xyz.shape[0])
                    total_points += n_chunk_pts

                    if crop_xy is not None:
                        in_crop = (
                            (chunk_xyz[:, 0] >= crop_xy[0]) & (chunk_xyz[:, 0] <= crop_xy[2]) &
                            (chunk_xyz[:, 1] >= crop_xy[1]) & (chunk_xyz[:, 1] <= crop_xy[3])
                        )
                        if not np.any(in_crop):
                            pbar_assign.update(n_chunk_pts)
                            continue
                        chunk_xyz = chunk_xyz[in_crop]

                    matched_mask_idx = process_chunk(
                        chunk_xyz,
                        masks,
                        args.distance,
                        use_hull_fill=args.hull_fill,
                        hull_eps=args.hull_eps,
                        query_workers=args.query_workers,
                    )
                    chunks_seen += 1

                    matched_ids = np.unique(matched_mask_idx[matched_mask_idx >= 0])
                    for mask_idx in matched_ids:
                        mask = masks[int(mask_idx)]
                        sel = matched_mask_idx == mask_idx
                        if not np.any(sel):
                            continue
                        sel_count = int(np.sum(sel))
                        assigned_points += sel_count
                        if write_ply:
                            recs = np.zeros(sel_count, dtype=ply_dtype)
                            sel_pts = chunk_xyz[sel]
                            recs["x"] = sel_pts[:, 0]
                            recs["y"] = sel_pts[:, 1]
                            recs["z"] = sel_pts[:, 2]
                            recs["tree_id"] = mask.tree_id
                            if write_stem_id:
                                recs["stem_id"] = mask.stem_id
                            ply_appenders[(mask.tree_id, mask.stem_id)].append(recs)

                    pbar_assign.update(n_chunk_pts)

                    now_t = time.time()
                    if now_t - last_heartbeat_t >= 20.0:
                        stage(
                            f"Heartbeat: processed {chunks_seen} chunk(s), "
                            f"scanned {total_points} point(s), assigned {assigned_points}"
                        )
                        last_heartbeat_t = now_t

    if write_ply:
        for app in ply_appenders.values():
            if app.count > 0:
                app.finalize()
            else:
                app.discard()

    elapsed = time.time() - start_time
    stage(f"Done in {elapsed:.2f}s. Scanned {total_points} points; assigned {assigned_points}, unbound {unbound_points}.")
    if write_las:
        stage("Wrote masked LAS/LAZ outputs: " + ", ".join(str(p) for p in las_output_paths))
    if write_ply:
        stage(f"Wrote per-mask PLY outputs to: {out_dir}")

    if args.debug and args.plot_count > 0:
        if not write_ply:
            stage("Debug QC plots require PLY outputs. Re-run without --ids-only.")
            return

        stage("Generating QC plots for input-mask vs output-mask overlays")
        target_plot_points = capped_voxel_sample(
            files=target_files,
            chunk_size=args.chunk_size,
            voxel_size=args.plot_voxel_size,
            max_points=args.plot_target_max_points,
        )
        debug_settings = {
            "target": str(args.target),
            "plot_count": int(args.plot_count),
            "plot_seed": int(args.plot_seed),
            "plot_target_max_points": int(args.plot_target_max_points),
            "plot_mask_max_points": int(args.plot_mask_max_points),
            "plot_voxel_size": float(args.plot_voxel_size),
            "show_hull_outline": bool(args.hull_fill),
        }
        selection_state_path = out_dir / "qc_plots" / "debug_selection.json"

        try:
            qc_stats, qc_selected_masks, qc_selection_reused = generate_qc_plots(
                masks=masks,
                out_dir=out_dir,
                rng_seed=args.plot_seed,
                count=args.plot_count,
                target_plot_points=target_plot_points,
                mask_plot_max_points=args.plot_mask_max_points,
                plot_voxel_size=args.plot_voxel_size,
                show_hull_outline=bool(args.hull_fill),
                selection_state_path=selection_state_path,
                debug_settings=debug_settings,
            )
            stage(
                f"QC plots done: {len(qc_stats)} mask plot(s), selection reused={qc_selection_reused}, "
                f"selected={';'.join(qc_selected_masks)}"
            )
        except Exception as exc:
            stage(f"QC plot generation failed: {exc}")


if __name__ == "__main__":
    main()
