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


def parse_mask_ids(stem: str) -> Tuple[int, int]:
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


@dataclass
class HullInfo:
    mins: np.ndarray
    maxs: np.ndarray
    A: np.ndarray
    b: np.ndarray


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


def build_mask_record(
    tree_id: int,
    stem_id: int,
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
    key_name = f"{tree_id}_{stem_id}"

    hull = None
    if use_hull_fill:
        hull = build_hull_info(pts, float(distance), vox_mul)

    return MaskRecord(
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
    )


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
    mask_files = list_point_files(mask_folder)
    if not mask_files:
        raise SystemExit("No mask files found.")

    stage(f"Loading {len(mask_files)} mask file(s)")
    for path in tqdm(mask_files, unit="mask", desc="Load masks", **PROGRESS_KW):
        try:
            tree_id, stem_id = parse_mask_ids(path.stem)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        read_desc = f"Read {path.name}" if path.suffix.lower() in (".las", ".laz") else None
        pts = get_points(path, chunk_size=load_chunk_size, progress_desc=read_desc)
        if pts.shape[0] == 0:
            continue
        grouped.setdefault((tree_id, stem_id), []).append(pts)

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
                    tree_id,
                    stem_id,
                    point_sets,
                    distance,
                    use_hull_fill,
                    vox_mul,
                    out_dir,
                    las_suffix,
                    fast_index_build,
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


def _scatter_view(
    ax,
    title: str,
    tgt_matched: np.ndarray,
    tgt_unmatched: np.ndarray,
    mask_unmatched: np.ndarray,
    proj_fn,
    xlim=None,
    ylim=None,
) -> None:
    if tgt_unmatched.shape[0] > 0:
        p = proj_fn(tgt_unmatched)
        ax.scatter(p[:, 0], p[:, 1], s=1.0, c="black", alpha=0.55, linewidths=0)
    if tgt_matched.shape[0] > 0:
        p = proj_fn(tgt_matched)
        ax.scatter(p[:, 0], p[:, 1], s=1.2, c="#22c55e", alpha=0.75, linewidths=0)
    if mask_unmatched.shape[0] > 0:
        p = proj_fn(mask_unmatched)
        ax.scatter(p[:, 0], p[:, 1], s=1.8, c="#ef4444", alpha=0.85, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def _extract_plot_sets(
    mask_pts: np.ndarray,
    target_points: np.ndarray,
    match_dist: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mins = np.min(mask_pts, axis=0)
    maxs = np.max(mask_pts, axis=0)
    pad = float(match_dist)
    in_bbox = np.all((target_points >= (mins - pad)) & (target_points <= (maxs + pad)), axis=1)
    target_local = target_points[in_bbox]
    if target_local.shape[0] == 0:
        return target_local, target_local, mask_pts, mins, maxs

    mask_tree = cKDTree(mask_pts)
    d_tgt, _ = mask_tree.query(target_local, k=1, workers=1)
    tgt_matched = target_local[d_tgt <= match_dist]
    tgt_unmatched = target_local[d_tgt > match_dist]

    tgt_tree = cKDTree(target_local)
    d_mask, _ = tgt_tree.query(mask_pts, k=1, workers=1)
    mask_unmatched = mask_pts[d_mask > match_dist]
    return tgt_matched, tgt_unmatched, mask_unmatched, mins, maxs


def generate_qc_plots(
    mask_paths: List[Path],
    target_files: List[Path],
    out_dir: Path,
    rng_seed: int,
    count: int,
    match_dist: float,
    target_plot_max_points: int,
    mask_plot_max_points: int,
    sample_chunk_size: int,
    plot_voxel_size: float,
    selection_state_path: Optional[Path],
    debug_settings: Optional[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[str], bool]:
    if plt is None:
        raise RuntimeError("matplotlib is required for QC plots. Install matplotlib and rerun.")
    if not mask_paths:
        return [], [], False

    chosen: List[Path] = []
    reused_selection = False
    candidate_map = {p.name: p for p in mask_paths}

    if selection_state_path is not None and debug_settings is not None and selection_state_path.exists():
        try:
            state = json.loads(selection_state_path.read_text(encoding="utf-8"))
            if state.get("debug_settings") == debug_settings:
                selected_names = state.get("selected_masks", [])
                if isinstance(selected_names, list) and selected_names:
                    restored = [candidate_map[name] for name in selected_names if name in candidate_map]
                    if len(restored) == len(selected_names):
                        chosen = restored
                        reused_selection = True
        except Exception:
            pass

    if not chosen:
        rng = random.Random(rng_seed)
        n_pick = min(int(count), len(mask_paths))
        chosen = rng.sample(mask_paths, n_pick)
        if selection_state_path is not None and debug_settings is not None:
            selection_state_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "debug_settings": debug_settings,
                "selected_masks": [p.name for p in chosen],
            }
            selection_state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    stage("Sampling target for QC plots")
    target_plot_points = capped_voxel_sample(
        files=target_files,
        chunk_size=sample_chunk_size,
        voxel_size=plot_voxel_size,
        max_points=target_plot_max_points,
    )
    if target_plot_points.shape[0] == 0:
        raise RuntimeError("Could not sample target points for QC plots")

    plot_dir = out_dir / "qc_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    per_mask_stats: List[Dict[str, object]] = []
    overview_data: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    for mask_path in chosen:
        mask_pts = capped_voxel_sample(
            files=[mask_path],
            chunk_size=sample_chunk_size,
            voxel_size=plot_voxel_size,
            max_points=mask_plot_max_points,
        )
        if mask_pts.shape[0] == 0:
            continue

        tgt_match, tgt_unmatch, mask_unmatch, mins, maxs = _extract_plot_sets(
            mask_pts=mask_pts,
            target_points=target_plot_points,
            match_dist=match_dist,
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=180)
        _scatter_view(
            axes[0],
            "Top (XY)",
            tgt_match,
            tgt_unmatch,
            mask_unmatch,
            project_top,
            xlim=(float(mins[0]), float(maxs[0])),
            ylim=(float(mins[1]), float(maxs[1])),
        )
        _scatter_view(
            axes[1],
            "Front (XZ)",
            tgt_match,
            tgt_unmatch,
            mask_unmatch,
            project_front,
            xlim=(float(mins[0]), float(maxs[0])),
            ylim=(float(mins[2]), float(maxs[2])),
        )
        corners = bbox_corners(mins, maxs)
        ac = project_aerial(corners)
        _scatter_view(
            axes[2],
            "Aerial (Oblique)",
            tgt_match,
            tgt_unmatch,
            mask_unmatch,
            project_aerial,
            xlim=(float(np.min(ac[:, 0])), float(np.max(ac[:, 0]))),
            ylim=(float(np.min(ac[:, 1])), float(np.max(ac[:, 1]))),
        )
        fig.suptitle(
            f"Mask {mask_path.name} | green=target matched | black=target unmatched | red=mask unmatched",
            fontsize=11,
        )
        fig.tight_layout()
        fig_path = plot_dir / f"{mask_path.stem}_views.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        overview_data.append((mask_path.name, tgt_match, tgt_unmatch, mask_unmatch))
        per_mask_stats.append(
            {
                "mask_file": mask_path.name,
                "plot_path": str(fig_path),
                "target_points_in_bbox": int(tgt_match.shape[0] + tgt_unmatch.shape[0]),
                "target_matched": int(tgt_match.shape[0]),
                "target_unmatched": int(tgt_unmatch.shape[0]),
                "mask_unmatched": int(mask_unmatch.shape[0]),
            }
        )

    if overview_data:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=180)
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            if i >= len(overview_data):
                ax.axis("off")
                continue
            name, tgt_match, tgt_unmatch, mask_unmatch = overview_data[i]
            _scatter_view(ax, f"{name} (Aerial)", tgt_match, tgt_unmatch, mask_unmatch, project_aerial)
        fig.suptitle(
            "QC sample (2x2): green=target matched | black=target unmatched | red=mask unmatched",
            fontsize=12,
        )
        fig.tight_layout()
        grid_path = plot_dir / "qc_overview_2x2.png"
        fig.savefig(grid_path, bbox_inches="tight")
        plt.close(fig)

    return per_mask_stats, [p.name for p in chosen], reused_selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract points per mask id and write LAS/LAZ with IDs and/or per-mask PLY."
    )
    parser.add_argument("-m", "--mask-folder", required=True, type=Path, help="Folder containing mask files")
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
    args = parser.parse_args()

    args.mask_folder = normalize_cli_path(args.mask_folder, must_exist=True)
    args.target = normalize_cli_path(args.target, must_exist=True)
    if args.output is not None:
        args.output = normalize_cli_path(args.output, must_exist=False)

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

    out_dir = args.output if args.output else Path(f"{args.mask_folder.name}_extracted")
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
    las_suffix = ".laz" if any(p.suffix.lower() == ".laz" for p in las_targets) else ".las"

    masks = load_masks(
        mask_folder=args.mask_folder,
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

    if write_las:
        for m in masks:
            if m.las_out is not None and m.las_out.exists():
                m.las_out.unlink()

    ply_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("tree_id", "<i4"),
            ("stem_id", "<i4"),
        ]
    )
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
    chunks_seen = 0
    last_heartbeat_t = time.time()
    start_time = time.time()

    for target_file in target_files:
        ext = target_file.suffix.lower()
        stage(f"Target: {target_file.name}")

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
                        if out_path.exists():
                            las_writers[(m.tree_id, m.stem_id)] = stack.enter_context(
                                laspy.open(out_path, mode="a")
                            )
                        else:
                            las_writers[(m.tree_id, m.stem_id)] = stack.enter_context(
                                laspy.open(out_path, mode="w", header=out_header)
                            )

                for chunk in tqdm(
                    src.chunk_iterator(args.chunk_size),
                    unit="pts",
                    desc=f"Assign {target_file.name}",
                    **PROGRESS_KW,
                ):
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
                            recs = np.zeros(sel_count, dtype=ply_dtype)
                            pts = chunk_xyz[sel]
                            recs["x"] = pts[:, 0]
                            recs["y"] = pts[:, 1]
                            recs["z"] = pts[:, 2]
                            recs["tree_id"] = mask.tree_id
                            recs["stem_id"] = mask.stem_id
                            ply_appenders[(mask.tree_id, mask.stem_id)].append(recs)

                    now_t = time.time()
                    if now_t - last_heartbeat_t >= 20.0:
                        stage(
                            f"Heartbeat: processed {chunks_seen} chunk(s), "
                            f"scanned {total_points} point(s), assigned {assigned_points}"
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

            for start_idx in tqdm(
                range(0, pts.shape[0], args.chunk_size),
                unit="pts",
                desc=f"Assign {target_file.name}",
                **PROGRESS_KW,
            ):
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
                        recs["stem_id"] = mask.stem_id
                        ply_appenders[(mask.tree_id, mask.stem_id)].append(recs)

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
    stage(f"Done in {elapsed:.2f}s. Scanned {total_points} points; assigned {assigned_points}.")
    if write_las:
        stage(f"Wrote LAS/LAZ outputs to: {out_dir}")
    if write_ply:
        stage(f"Wrote per-mask PLY outputs to: {out_dir}")

    if args.debug and args.plot_count > 0:
        if not write_ply:
            stage("Debug QC plots require PLY outputs. Re-run without --ids-only.")
            return

        stage("Generating QC plots for assigned masks")
        mask_paths = [m.ply_out for m in masks if m.ply_out is not None and m.ply_out.exists()]
        plot_match_distance = (
            float(args.plot_match_distance) if args.plot_match_distance is not None else float(args.distance)
        )
        debug_settings = {
            "target": str(args.target),
            "plot_count": int(args.plot_count),
            "plot_seed": int(args.plot_seed),
            "plot_match_distance": float(plot_match_distance),
            "plot_target_max_points": int(args.plot_target_max_points),
            "plot_mask_max_points": int(args.plot_mask_max_points),
            "plot_voxel_size": float(args.plot_voxel_size),
            "sample_chunk_size": int(args.chunk_size),
        }
        selection_state_path = out_dir / "qc_plots" / "debug_selection.json"

        try:
            qc_stats, qc_selected_masks, qc_selection_reused = generate_qc_plots(
                mask_paths=[p for p in mask_paths if p is not None],
                target_files=target_files,
                out_dir=out_dir,
                rng_seed=args.plot_seed,
                count=args.plot_count,
                match_dist=plot_match_distance,
                target_plot_max_points=args.plot_target_max_points,
                mask_plot_max_points=args.plot_mask_max_points,
                sample_chunk_size=args.chunk_size,
                plot_voxel_size=args.plot_voxel_size,
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
