#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import laspy
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from tqdm import tqdm


SUPPORTED_EXTS = {".ply", ".las", ".laz"}
PROGRESS_KW = {
    "leave": False,
    "mininterval": 0.5,
    "dynamic_ncols": True,
    "disable": not sys.stderr.isatty(),
}


def stage(msg: str) -> None:
    print(f"[align] {msg}")


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
    text = path.read_text(encoding="utf-8").replace(",", " ")
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
    progress_desc: Optional[str] = None,
) -> np.ndarray:
    kept: List[np.ndarray] = []
    total_kept = 0

    file_iter: Iterable[Path] = files
    if progress_desc is not None:
        file_iter = tqdm(files, desc=progress_desc, unit="file", **PROGRESS_KW)

    for path in file_iter:
        for pts in iter_points(path, chunk_size):
            if transform is not None:
                pts = apply_transform(pts, transform)

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
    src_c = src - src_centroid
    dst_c = dst - dst_centroid

    h = src_c.T @ dst_c
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Align combined masks to a target cloud using optional DAT coarse transform + ICP; "
            "write transformed masks and reports."
        )
    )
    ap.add_argument("target_file", type=Path, help="Target cloud file (.ply/.las/.laz)")
    ap.add_argument("mask_dir", type=Path, help="Directory of mask files to align")
    ap.add_argument(
        "--dat_transform",
        type=Path,
        default=None,
        help="Optional 4x4 DAT matrix applied to masks before ICP",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: mask_dir/transformed_to_target)",
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
    args = ap.parse_args()

    args.target_file = normalize_cli_path(args.target_file, must_exist=True)
    args.mask_dir = normalize_cli_path(args.mask_dir, must_exist=True)
    if args.dat_transform is not None:
        args.dat_transform = normalize_cli_path(args.dat_transform, must_exist=True)
    if args.output_dir is not None:
        args.output_dir = normalize_cli_path(args.output_dir, must_exist=False)

    if args.target_file.suffix.lower() not in SUPPORTED_EXTS:
        raise SystemExit("target_file must be .ply/.las/.laz")
    if not args.mask_dir.is_dir():
        raise SystemExit("mask_dir must be a directory")
    if args.voxel_size <= 0:
        raise SystemExit("--voxel_size must be > 0")
    if args.max_points_target < 1000 or args.max_points_masks < 1000:
        raise SystemExit("--max_points_target and --max_points_masks must be >= 1000")
    if args.sample_chunk_size <= 0 or args.write_chunk_size <= 0:
        raise SystemExit("--sample_chunk_size and --write_chunk_size must be > 0")

    mask_files = list_point_files(args.mask_dir)
    if not mask_files:
        raise SystemExit("No mask files found in mask_dir")

    out_dir = args.output_dir if args.output_dir is not None else (args.mask_dir / "transformed_to_target")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = out_dir / "mask_to_target_transform.DAT"
    report_path = out_dir / "icp_report.json"
    summary_csv_path = out_dir / "icp_summary.csv"

    coarse = np.eye(4, dtype=np.float64)
    if args.dat_transform is not None:
        coarse = read_dat_matrix(args.dat_transform)

    stage("Sampling target point cloud for ICP")
    target_sample = capped_voxel_sample(
        [args.target_file],
        chunk_size=args.sample_chunk_size,
        voxel_size=args.voxel_size,
        max_points=args.max_points_target,
        transform=None,
        progress_desc="Sample target",
    )
    if target_sample.shape[0] < 1000:
        raise SystemExit("Insufficient target sample points for ICP")

    stage("Sampling combined masks for ICP")
    mask_sample = capped_voxel_sample(
        mask_files,
        chunk_size=args.sample_chunk_size,
        voxel_size=args.voxel_size,
        max_points=args.max_points_masks,
        transform=coarse,
        progress_desc="Sample masks",
    )
    if mask_sample.shape[0] < 1000:
        raise SystemExit("Insufficient mask sample points for ICP")

    stage(
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
    residuals: Dict[str, float] = compute_residual_summary(aligned_mask_sample, target_sample)

    stage("Writing transformed masks")
    file_report: List[Dict[str, object]] = []
    for src_path in tqdm(mask_files, desc="Write transformed masks", unit="file", **PROGRESS_KW):
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
        "points_written": int(sum(int(r["points"]) for r in file_report)),
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
        "summary_csv": str(summary_csv_path),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    stage("Done")
    stage(f"Transformed masks: {out_dir}")
    stage(f"Final transform DAT: {matrix_path}")
    stage(f"ICP summary CSV: {summary_csv_path}")
    stage(f"ICP report: {report_path}")


if __name__ == "__main__":
    main()
