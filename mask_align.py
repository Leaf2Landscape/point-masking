#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import laspy
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from tqdm import tqdm

SUPPORTED_EXTS = {".ply", ".las", ".laz"}


def list_point_files(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]


def iter_points(path: Path, chunk_size: int) -> Iterable[np.ndarray]:
    ext = path.suffix.lower()
    if ext in (".las", ".laz"):
        with laspy.open(path) as src:
            for chunk in src.chunk_iterator(chunk_size):
                xyz = np.vstack((chunk.x, chunk.y, chunk.z)).T.astype(np.float64, copy=False)
                if xyz.size > 0:
                    yield xyz
        return

    if ext == ".ply":
        with open(path, "rb") as f:
            ply = PlyData.read(f)
        v = ply["vertex"]
        xyz = np.vstack((v["x"], v["y"], v["z"])).T.astype(np.float64, copy=False)
        if xyz.size == 0:
            return
        for start in range(0, xyz.shape[0], chunk_size):
            end = min(start + chunk_size, xyz.shape[0])
            yield xyz[start:end]
        return

    raise ValueError(f"Unsupported file extension: {path.suffix}")


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    homog = np.ones((points.shape[0], 4), dtype=np.float64)
    homog[:, :3] = points
    out = homog @ transform.T
    return out[:, :3]


def read_dat_matrix(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    text = text.replace(",", " ")
    vals = np.fromstring(text, sep=" ", dtype=np.float64)
    if vals.size != 16:
        raise ValueError(f"Expected 16 numeric values in DAT transform, found {vals.size}")
    mat = vals.reshape(4, 4)
    if not np.isfinite(mat).all():
        raise ValueError("DAT transform contains non-finite values")
    return mat


def write_dat_matrix(path: Path, mat: np.ndarray) -> None:
    lines = [" ".join(f"{v:.10f}" for v in row) for row in mat]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def capped_voxel_sample(
    files: List[Path],
    chunk_size: int,
    voxel_size: float,
    max_points: int,
    transform: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Collect a memory-capped sample with per-chunk voxel thinning."""
    kept: List[np.ndarray] = []
    total_kept = 0

    for path in files:
        for pts in iter_points(path, chunk_size):
            if transform is not None:
                pts = apply_transform(pts, transform)

            if voxel_size > 0:
                voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
                _, unique_local = np.unique(voxel_idx, axis=0, return_index=True)
                pts = pts[np.sort(unique_local)]

            if pts.shape[0] == 0:
                continue

            room = max_points - total_kept
            if room <= 0:
                break

            if pts.shape[0] > room:
                pick = np.linspace(0, pts.shape[0] - 1, num=room, dtype=np.int64)
                pts = pts[pick]

            kept.append(pts)
            total_kept += pts.shape[0]

        if total_kept >= max_points:
            break

    if not kept:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(kept)


def rigid_transform_kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_centroid = np.mean(src, axis=0)
    dst_centroid = np.mean(dst, axis=0)

    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid

    h = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    t = dst_centroid - r @ src_centroid

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def run_icp_scipy(
    source: np.ndarray,
    target: np.ndarray,
    init_transform: np.ndarray,
    threshold: float,
    max_iter: int,
    tol: float,
) -> Dict[str, object]:
    if source.shape[0] < 3 or target.shape[0] < 3:
        raise RuntimeError("Need at least 3 points in source and target for ICP")

    tform = init_transform.copy()
    target_tree = cKDTree(target)
    prev_rmse = np.inf
    used_iters = 0

    for i in range(max_iter):
        used_iters = i + 1
        src_t = apply_transform(source, tform)
        dists, nn_idx = target_tree.query(src_t, k=1, workers=1)
        inlier = dists <= threshold

        if np.count_nonzero(inlier) < 3:
            break

        src_in = src_t[inlier]
        dst_in = target[nn_idx[inlier]]

        delta = rigid_transform_kabsch(src_in, dst_in)
        tform = delta @ tform

        rmse = float(np.sqrt(np.mean(dists[inlier] ** 2)))
        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse

    src_final = apply_transform(source, tform)
    dists, _ = target_tree.query(src_final, k=1, workers=1)
    inlier = dists <= threshold
    fitness = float(np.count_nonzero(inlier) / max(1, source.shape[0]))
    rmse = float(np.sqrt(np.mean(dists[inlier] ** 2))) if np.any(inlier) else float("nan")

    return {
        "backend": "scipy",
        "transform": tform,
        "fitness": fitness,
        "inlier_rmse": rmse,
        "iterations": used_iters,
    }


def run_icp(
    source: np.ndarray,
    target: np.ndarray,
    init_transform: np.ndarray,
    threshold: float,
    max_iter: int,
    tol: float,
) -> Dict[str, object]:
    try:
        import open3d as o3d

        src_pc = o3d.geometry.PointCloud()
        src_pc.points = o3d.utility.Vector3dVector(source)

        tgt_pc = o3d.geometry.PointCloud()
        tgt_pc.points = o3d.utility.Vector3dVector(target)

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter,
            relative_fitness=tol,
            relative_rmse=tol,
        )

        reg = o3d.pipelines.registration.registration_icp(
            src_pc,
            tgt_pc,
            threshold,
            init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria,
        )

        return {
            "backend": "open3d",
            "transform": np.asarray(reg.transformation, dtype=np.float64),
            "fitness": float(reg.fitness),
            "inlier_rmse": float(reg.inlier_rmse),
            "iterations": int(max_iter),
        }
    except Exception:
        return run_icp_scipy(source, target, init_transform, threshold, max_iter, tol)


def compute_residual_summary(source_aligned: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    if source_aligned.shape[0] == 0 or target.shape[0] == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }

    tree = cKDTree(target)
    dists, _ = tree.query(source_aligned, k=1, workers=1)
    return {
        "mean": float(np.mean(dists)),
        "median": float(np.median(dists)),
        "p90": float(np.quantile(dists, 0.90)),
        "p95": float(np.quantile(dists, 0.95)),
        "p99": float(np.quantile(dists, 0.99)),
        "max": float(np.max(dists)),
    }


def transform_las_like(in_path: Path, out_path: Path, transform: np.ndarray, chunk_size: int) -> int:
    count = 0
    with laspy.open(in_path) as src:
        out_header = src.header.copy()
        with laspy.open(out_path, mode="w", header=out_header) as dst:
            for chunk in src.chunk_iterator(chunk_size):
                xyz = np.vstack((chunk.x, chunk.y, chunk.z)).T.astype(np.float64, copy=False)
                xyz_t = apply_transform(xyz, transform)
                chunk.x = xyz_t[:, 0]
                chunk.y = xyz_t[:, 1]
                chunk.z = xyz_t[:, 2]
                dst.write_points(chunk)
                count += len(chunk)
    return count


def transform_ply(in_path: Path, out_path: Path, transform: np.ndarray) -> int:
    with open(in_path, "rb") as f:
        ply = PlyData.read(f)

    vertex = np.asarray(ply["vertex"].data)
    if vertex.size == 0:
        PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(out_path)
        return 0

    xyz = np.vstack((vertex["x"], vertex["y"], vertex["z"])).T.astype(np.float64, copy=False)
    xyz_t = apply_transform(xyz, transform)

    out_vertex = np.array(vertex, copy=True)
    out_vertex["x"] = xyz_t[:, 0]
    out_vertex["y"] = xyz_t[:, 1]
    out_vertex["z"] = xyz_t[:, 2]

    PlyData([PlyElement.describe(out_vertex, "vertex")], text=False).write(out_path)
    return out_vertex.shape[0]


def project_top(points: np.ndarray) -> np.ndarray:
    return points[:, [0, 1]]


def project_front(points: np.ndarray) -> np.ndarray:
    return points[:, [0, 2]]


def project_aerial(points: np.ndarray) -> np.ndarray:
    # Oblique "down-angle from side" projection.
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


def sample_points_from_file(path: Path, chunk_size: int, max_points: int) -> np.ndarray:
    """Memory-capped deterministic sampling from one file."""
    chunks: List[np.ndarray] = []
    kept = 0
    for pts in iter_points(path, chunk_size):
        room = max_points - kept
        if room <= 0:
            break
        if pts.shape[0] > room:
            pick = np.linspace(0, pts.shape[0] - 1, num=room, dtype=np.int64)
            pts = pts[pick]
        chunks.append(pts)
        kept += pts.shape[0]
    if not chunks:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(chunks)


def _scatter_view(
    ax,
    title: str,
    tgt_matched: np.ndarray,
    tgt_unmatched: np.ndarray,
    mask_unmatched: np.ndarray,
    proj_fn,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
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
    transformed_mask_paths: List[Path],
    target_file: Path,
    out_dir: Path,
    rng_seed: int,
    count: int,
    match_dist: float,
    target_plot_max_points: int,
    mask_plot_max_points: int,
    sample_chunk_size: int,
    plot_voxel_size: float,
) -> List[Dict[str, object]]:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required for QC plot generation. Install matplotlib and rerun."
        ) from e

    if not transformed_mask_paths:
        return []

    rng = random.Random(rng_seed)
    n_pick = min(int(count), len(transformed_mask_paths))
    chosen = rng.sample(transformed_mask_paths, n_pick)

    print("Sampling target for QC plots...")
    target_plot_points = capped_voxel_sample(
        [target_file],
        chunk_size=sample_chunk_size,
        voxel_size=plot_voxel_size,
        max_points=target_plot_max_points,
        transform=None,
    )
    if target_plot_points.shape[0] == 0:
        raise RuntimeError("Could not sample target points for QC plots")

    plot_dir = out_dir / "qc_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    per_mask_stats: List[Dict[str, object]] = []
    overview_data: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    for mask_path in chosen:
        mask_pts = sample_points_from_file(mask_path, sample_chunk_size, mask_plot_max_points)
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
            _scatter_view(
                ax,
                f"{name} (Aerial)",
                tgt_match,
                tgt_unmatch,
                mask_unmatch,
                project_aerial,
            )
        fig.suptitle(
            "QC sample (2x2): green=target matched | black=target unmatched | red=mask unmatched",
            fontsize=12,
        )
        fig.tight_layout()
        grid_path = plot_dir / "qc_overview_2x2.png"
        fig.savefig(grid_path, bbox_inches="tight")
        plt.close(fig)

    return per_mask_stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Align combined masks to target with optional DAT coarse transform + ICP, then write transformed masks."
    )
    ap.add_argument("target_file", type=Path, help="Target LAS/LAZ/PLY file used as ICP reference")
    ap.add_argument("mask_dir", type=Path, help="Directory containing mask files (.ply/.las/.laz)")
    ap.add_argument(
        "--dat_transform",
        type=Path,
        default=None,
        help="Optional DAT file with a 4x4 coarse transform matrix applied before ICP",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for transformed masks (default: <mask_dir>/transformed_to_target)",
    )
    ap.add_argument(
        "--voxel_size",
        type=float,
        default=0.10,
        help="Voxel size for memory-capped sampling before ICP (default: 0.10)",
    )
    ap.add_argument(
        "--max_points_target",
        type=int,
        default=800000,
        help="Maximum sampled points from target for ICP (default: 800000)",
    )
    ap.add_argument(
        "--max_points_masks",
        type=int,
        default=800000,
        help="Maximum sampled points from combined masks for ICP (default: 800000)",
    )
    ap.add_argument(
        "--sample_chunk_size",
        type=int,
        default=500000,
        help="Chunk size used for streaming point sampling (default: 500000)",
    )
    ap.add_argument(
        "--write_chunk_size",
        type=int,
        default=500000,
        help="Chunk size used when writing transformed LAS/LAZ files (default: 500000)",
    )
    ap.add_argument(
        "--icp_threshold",
        type=float,
        default=1.0,
        help="Max correspondence distance for ICP (default: 1.0)",
    )
    ap.add_argument(
        "--icp_max_iter",
        type=int,
        default=60,
        help="Maximum ICP iterations (default: 60)",
    )
    ap.add_argument(
        "--icp_tol",
        type=float,
        default=1e-6,
        help="ICP convergence tolerance (default: 1e-6)",
    )
    ap.add_argument(
        "--plot_count",
        type=int,
        default=4,
        help="Number of random transformed masks to plot for QC (default: 4, debug only)",
    )
    ap.add_argument(
        "--plot_seed",
        type=int,
        default=42,
        help="Random seed for selecting QC masks (default: 42)",
    )
    ap.add_argument(
        "--plot_match_distance",
        type=float,
        default=None,
        help="Match threshold for QC plot coloring (default: --icp_threshold)",
    )
    ap.add_argument(
        "--plot_target_max_points",
        type=int,
        default=300000,
        help="Max target points sampled for QC plots (default: 300000)",
    )
    ap.add_argument(
        "--plot_mask_max_points",
        type=int,
        default=120000,
        help="Max transformed-mask points sampled per QC plot (default: 120000)",
    )
    ap.add_argument(
        "--plot_voxel_size",
        type=float,
        default=0.10,
        help="Voxel size for target sampling during QC plots (default: 0.10)",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug outputs (QC plots). By default, plots are not generated.",
    )
    ap.add_argument(
        "--reuse-existing-masks",
        action="store_true",
        help="Reuse previously transformed masks in output_dir when available.",
    )
    args = ap.parse_args()

    if not args.target_file.exists() or args.target_file.suffix.lower() not in SUPPORTED_EXTS:
        raise SystemExit("target_file must exist and be .ply/.las/.laz")
    if not args.mask_dir.exists() or not args.mask_dir.is_dir():
        raise SystemExit("mask_dir must exist and be a directory")
    if args.voxel_size <= 0:
        raise SystemExit("--voxel_size must be > 0")
    if args.max_points_target < 1000 or args.max_points_masks < 1000:
        raise SystemExit("--max_points_target and --max_points_masks must be >= 1000")
    if args.plot_count < 0:
        raise SystemExit("--plot_count must be >= 0")
    if args.plot_target_max_points < 1000 or args.plot_mask_max_points < 1000:
        raise SystemExit("--plot_target_max_points and --plot_mask_max_points must be >= 1000")
    if args.plot_voxel_size <= 0:
        raise SystemExit("--plot_voxel_size must be > 0")

    mask_files = list_point_files(args.mask_dir)
    if not mask_files:
        raise SystemExit("No mask files found in mask_dir")

    out_dir = args.output_dir if args.output_dir is not None else (args.mask_dir / "transformed_to_target")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = out_dir / "mask_to_target_transform.DAT"
    report_path = out_dir / "icp_report.json"
    summary_csv_path = out_dir / "icp_summary.csv"

    transformed_paths = [out_dir / p.name for p in mask_files]
    can_reuse_masks = all(p.exists() for p in transformed_paths)
    reuse_mode = bool(args.reuse_existing_masks and can_reuse_masks and matrix_path.exists())

    coarse = np.eye(4, dtype=np.float64)
    icp: Dict[str, object]
    icp_transform = np.eye(4, dtype=np.float64)
    final_transform = np.eye(4, dtype=np.float64)
    residuals: Dict[str, float]
    file_report: List[Dict[str, object]] = []

    if reuse_mode:
        print("Reusing existing transformed masks and transform matrix from output_dir...")
        final_transform = read_dat_matrix(matrix_path)
        for p in transformed_paths:
            file_report.append({"file": p.name, "output": str(p), "points": -1})

        if report_path.exists():
            try:
                old_report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                old_report = {}
        else:
            old_report = {}

        icp = {
            "backend": old_report.get("backend", "reused"),
            "fitness": float(old_report.get("icp_fitness", float("nan"))),
            "inlier_rmse": float(old_report.get("icp_inlier_rmse", float("nan"))),
            "iterations": int(old_report.get("icp_iterations", 0)),
            "transform": np.asarray(old_report.get("icp_transform", np.eye(4)), dtype=np.float64),
        }
        if np.asarray(icp["transform"]).shape == (4, 4):
            icp_transform = np.asarray(icp["transform"], dtype=np.float64)
        else:
            icp_transform = np.eye(4, dtype=np.float64)

        residuals = old_report.get("residual_summary", None)
        if not isinstance(residuals, dict):
            residuals = {
                "mean": float("nan"),
                "median": float("nan"),
                "p90": float("nan"),
                "p95": float("nan"),
                "p99": float("nan"),
                "max": float("nan"),
            }
    else:
        if args.reuse_existing_masks and not reuse_mode:
            print("Reuse requested, but existing transformed masks/transform were incomplete. Running full alignment.")

        if args.dat_transform is not None:
            coarse = read_dat_matrix(args.dat_transform)

        print("Sampling target point cloud for ICP...")
        target_sample = capped_voxel_sample(
            [args.target_file],
            chunk_size=args.sample_chunk_size,
            voxel_size=args.voxel_size,
            max_points=args.max_points_target,
            transform=None,
        )
        if target_sample.shape[0] < 1000:
            raise SystemExit("Insufficient target sample points for ICP")

        print("Sampling combined masks for ICP...")
        mask_sample = capped_voxel_sample(
            mask_files,
            chunk_size=args.sample_chunk_size,
            voxel_size=args.voxel_size,
            max_points=args.max_points_masks,
            transform=coarse,
        )
        if mask_sample.shape[0] < 1000:
            raise SystemExit("Insufficient mask sample points for ICP")

        print(
            f"Running ICP on sampled sets: source={mask_sample.shape[0]}, target={target_sample.shape[0]}"
        )
        icp = run_icp(
            source=mask_sample,
            target=target_sample,
            init_transform=np.eye(4, dtype=np.float64),
            threshold=args.icp_threshold,
            max_iter=args.icp_max_iter,
            tol=args.icp_tol,
        )

        icp_transform = np.asarray(icp["transform"], dtype=np.float64)
        final_transform = icp_transform @ coarse

        aligned_mask_sample = apply_transform(mask_sample, icp_transform)
        residuals = compute_residual_summary(aligned_mask_sample, target_sample)

        print("Writing transformed masks...")
        for src_path in tqdm(mask_files, unit="file"):
            dst_path = out_dir / src_path.name
            ext = src_path.suffix.lower()
            if ext in (".las", ".laz"):
                npts = transform_las_like(src_path, dst_path, final_transform, args.write_chunk_size)
            elif ext == ".ply":
                npts = transform_ply(src_path, dst_path, final_transform)
            else:
                continue
            file_report.append({"file": src_path.name, "output": str(dst_path), "points": int(npts)})

        write_dat_matrix(matrix_path, final_transform)
    plot_match_distance = float(args.plot_match_distance) if args.plot_match_distance is not None else float(args.icp_threshold)

    transformed_paths = [out_dir / row["file"] for row in file_report]
    qc_stats: List[Dict[str, object]] = []
    if args.debug and args.plot_count > 0:
        print("Generating QC plots...")
        qc_stats = generate_qc_plots(
            transformed_mask_paths=transformed_paths,
            target_file=args.target_file,
            out_dir=out_dir,
            rng_seed=args.plot_seed,
            count=args.plot_count,
            match_dist=plot_match_distance,
            target_plot_max_points=args.plot_target_max_points,
            mask_plot_max_points=args.plot_mask_max_points,
            sample_chunk_size=args.sample_chunk_size,
            plot_voxel_size=args.plot_voxel_size,
        )
    elif not args.debug:
        print("Debug mode off; skipping QC plot generation.")

    summary_row = {
        "target_file": str(args.target_file),
        "mask_dir": str(args.mask_dir),
        "output_dir": str(out_dir),
        "backend": icp["backend"],
        "icp_fitness": float(icp["fitness"]),
        "icp_inlier_rmse": float(icp["inlier_rmse"]),
        "icp_iterations": int(icp["iterations"]),
        "residual_mean": residuals["mean"],
        "residual_median": residuals["median"],
        "residual_p90": residuals["p90"],
        "residual_p95": residuals["p95"],
        "residual_p99": residuals["p99"],
        "residual_max": residuals["max"],
        "masks_written": len(file_report),
        "points_written": int(sum(int(r["points"]) for r in file_report if int(r["points"]) >= 0)),
        "reuse_existing_masks": bool(reuse_mode),
        "debug": bool(args.debug),
        "qc_plot_count": int(len(qc_stats)),
        "qc_plot_match_distance": plot_match_distance,
    }
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        w.writeheader()
        w.writerow(summary_row)

    report = {
        "target_file": str(args.target_file),
        "mask_dir": str(args.mask_dir),
        "output_dir": str(out_dir),
        "coarse_transform_path": str(args.dat_transform) if args.dat_transform else None,
        "backend": icp["backend"],
        "icp_fitness": icp["fitness"],
        "icp_inlier_rmse": icp["inlier_rmse"],
        "icp_iterations": icp["iterations"],
        "voxel_size": args.voxel_size,
        "max_points_target": args.max_points_target,
        "max_points_masks": args.max_points_masks,
        "icp_threshold": args.icp_threshold,
        "coarse_transform": coarse.tolist(),
        "icp_transform": icp_transform.tolist(),
        "final_transform": final_transform.tolist(),
        "residual_summary": residuals,
        "files_written": file_report,
        "reuse_existing_masks": bool(reuse_mode),
        "debug": bool(args.debug),
        "summary_csv": str(summary_csv_path),
        "qc_plot_match_distance": plot_match_distance,
        "qc_plots": qc_stats,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Transformed masks: {out_dir}")
    print(f"Final transform DAT: {matrix_path}")
    print(f"ICP summary CSV: {summary_csv_path}")
    print(f"ICP report: {report_path}")


if __name__ == "__main__":
    main()
