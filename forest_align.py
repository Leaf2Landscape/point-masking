#!/usr/bin/env python3

"""
forest_align.py

ForestAlign-style multi-view TLS co-registration for forestry scans.

This refactor replaces the previous sliding-window GICP averaging workflow with
the structural-complexity registration process described in:

    Castorena et al. (2025), ForestAlign: Automatic forest structure-based
    alignment for multi-view TLS and ALS point clouds.

Implemented workflow:
1. Stream and lightly decimate each scan for efficiency.
2. Downsample to a coarse voxel grid.
3. Estimate local plane normals in a spherical neighborhood.
4. Group points by structural complexity using a 3D von Mises-Fisher mixture.
5. Match structural groups between scans.
6. Run incremental ICP from low-complexity groups to high-complexity groups.
7. Apply pairwise registration across the full multi-view scan set using either
   a reference-scan or sequential chaining strategy.
8. Export the merged full-resolution output and optional per-scan transforms.

Registration pipeline:
- (Optional) Pre-coarse stage: user-specified voxel size and ICP distance
- Coarse stage: 0.05m voxel, 0.03m ICP (literature defaults)
- Fine stage: 0.015m voxel, 0.01m ICP (literature defaults)
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import laspy
except ModuleNotFoundError:
    laspy = None

try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None



# Paper-aligned defaults (always used for main two-stage pipeline)
COARSE_VOXEL_SIZE = 0.05
FINE_VOXEL_SIZE = 0.015
NORMAL_RADIUS = 0.25
COARSE_ICP_DIST = 0.03
FINE_ICP_DIST = 0.01
ICP_ITERS = 120

# Efficiency controls
SAMPLE_STEP = 8
MAX_POINTS_PER_SCAN = 250000
LAS_CHUNK_SIZE = 1000000
CACHE_SIZE = 4

# Outlier removal (noise filtering for TLS)
OUTLIER_REMOVAL = False
OUTLIER_NB_NEIGHBORS = 50
OUTLIER_STD_RATIO = 2.0

SCENE_PRESETS = {
    "tls_forest": 3,
    "als_forest": 2,
    "low_vegetation": 2,
    "bare_ground": 1,
}


@dataclass(frozen=True)
class ForestAlignSettings:
    complexity_levels: int
    normal_radius: float = NORMAL_RADIUS
    icp_iters: int = ICP_ITERS
    outlier_removal: bool = OUTLIER_REMOVAL
    outlier_nb_neighbors: int = OUTLIER_NB_NEIGHBORS
    outlier_std_ratio: float = OUTLIER_STD_RATIO
    # Optional pre-coarse stage (None/0 disables it)
    pre_coarse_voxel_size: float = 0.0
    pre_coarse_icp_dist: float = 0.0


@dataclass
class StructuralCloud:
    sampled_pcd: object
    grouped_pcd: object
    labels: np.ndarray
    complexity_scores: np.ndarray


def _require_dependencies():
    missing = []
    if laspy is None:
        missing.append("laspy")
    if o3d is None:
        missing.append("open3d")
    if missing:
        packages = ", ".join(missing)
        raise RuntimeError(
            f"Missing required Python packages: {packages}. Install them in the Python environment used to run this script."
        )


def normalize_cli_path(path: Path, must_exist: bool) -> Path:
    p = path.expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve(strict=must_exist)


def read_dat_matrix(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").replace(",", " ")
    vals = np.fromstring(text, sep=" ", dtype=np.float64)
    if vals.size != 16:
        raise ValueError(f"Expected 16 numeric values in DAT transform, found {vals.size}")
    mat = vals.reshape(4, 4)
    if not np.isfinite(mat).all():
        raise ValueError("DAT transform contains non-finite values")
    return mat


def _ext_family(ext: str) -> str:
    ext_l = ext.lower()
    if ext_l in [".laz", ".las"]:
        return "las"
    if ext_l == ".ply":
        return "ply"
    return "unknown"


def _is_supported_ext(ext: str) -> bool:
    return _ext_family(ext) in {"las", "ply"}


def _stream_las_points(path, sample_step, max_points, chunk_size):
    """Stream LAS/LAZ points in chunks and apply fast stride decimation."""
    pieces = []
    total = 0

    with laspy.open(path) as las:
        for chunk in las.chunk_iterator(chunk_size):
            pts = np.vstack((chunk.x, chunk.y, chunk.z)).T
            if sample_step > 1:
                pts = pts[::sample_step]
            if pts.size == 0:
                continue

            pieces.append(pts)
            total += len(pts)

            if max_points > 0 and total >= max_points:
                break

    if not pieces:
        return np.empty((0, 3), dtype=np.float64)

    pts = np.vstack(pieces)
    if max_points > 0 and len(pts) > max_points:
        pts = pts[:max_points]
    return pts


def load_scan_to_o3d(path, sample_step, max_points, chunk_size):
    """Load a scan as an Open3D point cloud with sampling applied."""
    input_ext = os.path.splitext(path)[1].lower()
    if not _is_supported_ext(input_ext):
        raise RuntimeError(f"Unsupported input extension for scan: {path}")

    if input_ext in [".laz", ".las"]:
        pts = _stream_las_points(path, sample_step, max_points, chunk_size)
        if pts.size == 0:
            raise RuntimeError(f"No points found in scan: {path}")
    else:
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            raise RuntimeError(f"No points found in scan: {path}")
        if sample_step > 1:
            pts = pts[::sample_step]
        if max_points > 0 and len(pts) > max_points:
            pts = pts[:max_points]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def _remove_statistical_outliers(pcd, nb_neighbors, std_ratio):
    """Remove statistical outliers from a point cloud using local density filtering."""
    if len(pcd.points) < nb_neighbors:
        return pcd
    
    pcd_filtered, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    n_removed = len(pcd.points) - len(pcd_filtered.points)
    if n_removed > 0:
        retention = 100.0 * len(pcd_filtered.points) / len(pcd.points)
        print(f"  Outlier removal: {n_removed} points removed, {retention:.1f}% retained")
    return pcd_filtered


def _normalize_rows(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    out = vectors.copy()
    valid = norms[:, 0] > 1e-12
    out[valid] /= norms[valid]
    out[~valid] = np.array([0.0, 0.0, 1.0])
    return out


def _canonicalize_normals(normals):
    normals = _normalize_rows(normals)
    flips = (
        (normals[:, 2] < 0)
        | ((np.isclose(normals[:, 2], 0.0)) & (normals[:, 1] < 0))
        | ((np.isclose(normals[:, 2], 0.0)) & (np.isclose(normals[:, 1], 0.0)) & (normals[:, 0] < 0))
    )
    normals[flips] *= -1.0
    return normals


def _estimate_kappa_3d(r_bar):
    r_bar = np.clip(r_bar, 1e-6, 0.999999)
    kappa = (r_bar * (3.0 - r_bar ** 2)) / np.maximum(1.0 - r_bar ** 2, 1e-6)
    return np.clip(kappa, 1e-3, 1e3)


def _log_sinh(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    hi = x > 20.0
    out[hi] = x[hi] - np.log(2.0)
    x_lo = np.maximum(x[~hi], 1e-8)
    out[~hi] = np.log(np.sinh(x_lo))
    return out


def _log_vmf_c3(kappa):
    kappa = np.asarray(kappa, dtype=np.float64)
    out = np.full_like(kappa, -np.log(4.0 * np.pi), dtype=np.float64)
    mask = kappa > 1e-6
    out[mask] = np.log(kappa[mask]) - np.log(4.0 * np.pi) - _log_sinh(kappa[mask])
    return out


def _spherical_kmeans(normals, n_clusters, max_iters=10):
    if n_clusters <= 1 or len(normals) == 0:
        centers = normals[:1] if len(normals) else np.array([[0.0, 0.0, 1.0]])
        return _normalize_rows(centers), np.zeros(len(normals), dtype=np.int32)

    centers = [normals[np.argmax(normals[:, 2])]]
    for _ in range(1, n_clusters):
        similarities = normals @ np.vstack(centers).T
        next_idx = np.argmin(np.max(similarities, axis=1))
        centers.append(normals[next_idx])

    centers = _normalize_rows(np.vstack(centers))
    labels = np.zeros(len(normals), dtype=np.int32)

    for _ in range(max_iters):
        similarities = normals @ centers.T
        new_labels = np.argmax(similarities, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for idx in range(n_clusters):
            cluster = normals[labels == idx]
            if len(cluster) == 0:
                fallback = np.argmin(np.max(similarities, axis=1))
                centers[idx] = normals[fallback]
            else:
                centers[idx] = _normalize_rows(cluster.mean(axis=0, keepdims=True))[0]

    return centers, labels


def fit_vmf_mixture(normals, n_clusters, max_iters=20, tol=1e-4):
    """Fit a simple EM mixture of 3D von Mises-Fisher distributions."""
    normals = _canonicalize_normals(normals)
    n_points = len(normals)
    if n_points == 0:
        return np.array([1.0]), np.array([[0.0, 0.0, 1.0]]), np.array([1e-3]), np.empty(0, dtype=np.int32)

    n_clusters = max(1, min(n_clusters, n_points))
    centers, labels = _spherical_kmeans(normals, n_clusters)

    weights = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    weights = np.maximum(weights, 1.0)
    weights /= weights.sum()

    kappas = np.full(n_clusters, 10.0, dtype=np.float64)
    mus = centers.copy()

    for cluster_idx in range(n_clusters):
        cluster = normals[labels == cluster_idx]
        if len(cluster) == 0:
            continue
        resultant = cluster.sum(axis=0, keepdims=True)
        mus[cluster_idx] = _normalize_rows(resultant)[0]
        r_bar = np.linalg.norm(resultant) / len(cluster)
        kappas[cluster_idx] = _estimate_kappa_3d(r_bar)

    prev_log_likelihood = None

    for _ in range(max_iters):
        dot = normals @ mus.T
        log_prob = np.log(weights + 1e-12)[None, :] + _log_vmf_c3(kappas)[None, :] + dot * kappas[None, :]

        row_max = np.max(log_prob, axis=1, keepdims=True)
        stable = np.exp(log_prob - row_max)
        denom = np.sum(stable, axis=1, keepdims=True)
        resp = stable / np.maximum(denom, 1e-12)
        log_likelihood = np.sum(row_max[:, 0] + np.log(np.maximum(denom[:, 0], 1e-12)))

        nk = np.sum(resp, axis=0)
        weights = np.maximum(nk, 1e-12)
        weights /= weights.sum()

        resultant = resp.T @ normals
        resultant_norm = np.linalg.norm(resultant, axis=1)
        mus = _normalize_rows(resultant)
        r_bar = resultant_norm / np.maximum(nk, 1e-12)
        kappas = _estimate_kappa_3d(r_bar)

        if prev_log_likelihood is not None:
            delta = abs(log_likelihood - prev_log_likelihood) / max(abs(prev_log_likelihood), 1.0)
            if delta < tol:
                break
        prev_log_likelihood = log_likelihood

    final_dot = normals @ mus.T
    final_log_prob = np.log(weights + 1e-12)[None, :] + _log_vmf_c3(kappas)[None, :] + final_dot * kappas[None, :]
    final_labels = np.argmax(final_log_prob, axis=1).astype(np.int32)
    return weights, mus, kappas, final_labels


def build_structural_cloud(sampled_pcd, settings):
    grouped_pcd = sampled_pcd.voxel_down_sample(COARSE_VOXEL_SIZE)
    if len(grouped_pcd.points) == 0:
        raise RuntimeError("Scan became empty after coarse voxel downsampling")

    grouped_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=settings.normal_radius,
            max_nn=60,
        )
    )
    normals = _canonicalize_normals(np.asarray(grouped_pcd.normals))
    _, _, kappas, labels = fit_vmf_mixture(normals, settings.complexity_levels)

    order = np.argsort(-kappas)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    labels = remap[labels]

    ordered_kappas = kappas[order]
    complexity_scores = 1.0 / np.maximum(ordered_kappas, 1e-6)

    return StructuralCloud(
        sampled_pcd=o3d.geometry.PointCloud(sampled_pcd),
        grouped_pcd=grouped_pcd,
        labels=labels,
        complexity_scores=complexity_scores,
    )


class PointCloudCache:
    """Small LRU cache for sampled scans and their structural groupings."""

    def __init__(self, paths, sample_step, max_points, chunk_size, cache_size, settings):
        self.paths = paths
        self.sample_step = sample_step
        self.max_points = max_points
        self.chunk_size = chunk_size
        self.cache_size = max(1, cache_size)
        self.settings = settings
        self.cache = OrderedDict()

    def get(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]

        sampled = load_scan_to_o3d(
            self.paths[idx],
            self.sample_step,
            self.max_points,
            self.chunk_size,
        )
        
        # Apply outlier removal if enabled
        if self.settings.outlier_removal:
            sampled = _remove_statistical_outliers(
                sampled,
                self.settings.outlier_nb_neighbors,
                self.settings.outlier_std_ratio
            )
        
        model = build_structural_cloud(sampled, self.settings)

        self.cache[idx] = model
        self.cache.move_to_end(idx)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return model


def _build_subset_cloud(structural_cloud, groups, voxel_size):
    points = np.asarray(structural_cloud.grouped_pcd.points)
    mask = np.isin(structural_cloud.labels, np.asarray(groups, dtype=np.int32))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[mask])
    if voxel_size > 0 and len(pcd.points) > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def _match_group_pairs(source_scores, target_scores):
    source_ids = list(range(len(source_scores)))
    target_ids = list(range(len(target_scores)))

    if len(source_ids) <= len(target_ids):
        best_pairs = None
        best_cost = float("inf")
        for target_perm in itertools.permutations(target_ids, len(source_ids)):
            cost = sum(abs(source_scores[s] - target_scores[t]) for s, t in zip(source_ids, target_perm))
            if cost < best_cost:
                best_cost = cost
                best_pairs = list(zip(source_ids, target_perm))
        return best_pairs or []

    best_pairs = None
    best_cost = float("inf")
    for source_perm in itertools.permutations(source_ids, len(target_ids)):
        cost = sum(abs(source_scores[s] - target_scores[t]) for s, t in zip(source_perm, target_ids))
        if cost < best_cost:
            best_cost = cost
            best_pairs = list(zip(source_perm, target_ids))

    return sorted(best_pairs or [], key=lambda pair: source_scores[pair[0]])


def run_icp(src, tgt, init_transform, max_corr_dist, max_iters):
    if len(src.points) == 0 or len(tgt.points) == 0:
        return init_transform

    result = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_corr_dist,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iters),
    )
    return result.transformation


def forestalign_pairwise(source_model, target_model, settings):
    """Run the ForestAlign pairwise workflow: optional pre-coarse, then coarse, then fine."""
    group_pairs = _match_group_pairs(source_model.complexity_scores, target_model.complexity_scores)
    group_pairs = sorted(group_pairs, key=lambda pair: source_model.complexity_scores[pair[0]])

    transform = np.eye(4)
    source_groups = []
    target_groups = []

    # Optional pre-coarse stage (user-specified voxel size and ICP distance)
    if settings.pre_coarse_voxel_size > 0 and settings.pre_coarse_icp_dist > 0:
        for source_group, target_group in group_pairs:
            source_groups.append(source_group)
            target_groups.append(target_group)

            src_stage = _build_subset_cloud(source_model, source_groups, settings.pre_coarse_voxel_size)
            tgt_stage = _build_subset_cloud(target_model, target_groups, settings.pre_coarse_voxel_size)

            transform = run_icp(
                src_stage,
                tgt_stage,
                transform,
                settings.pre_coarse_icp_dist,
                settings.icp_iters,
            )
        
        # Reset groups for main coarse stage
        source_groups = []
        target_groups = []

    # Main coarse stage (literature defaults: 0.05m voxel, 0.03m ICP)
    for source_group, target_group in group_pairs:
        source_groups.append(source_group)
        target_groups.append(target_group)

        src_stage = _build_subset_cloud(source_model, source_groups, COARSE_VOXEL_SIZE)
        tgt_stage = _build_subset_cloud(target_model, target_groups, COARSE_VOXEL_SIZE)

        transform = run_icp(
            src_stage,
            tgt_stage,
            transform,
            COARSE_ICP_DIST,
            settings.icp_iters,
        )

    # Fine stage (literature defaults: 0.015m voxel, 0.01m ICP)
    src_final = o3d.geometry.PointCloud(source_model.sampled_pcd)
    tgt_final = o3d.geometry.PointCloud(target_model.sampled_pcd)
    if FINE_VOXEL_SIZE > 0:
        src_final = src_final.voxel_down_sample(FINE_VOXEL_SIZE)
        tgt_final = tgt_final.voxel_down_sample(FINE_VOXEL_SIZE)

    return run_icp(
        src_final,
        tgt_final,
        transform,
        FINE_ICP_DIST,
        settings.icp_iters,
    )


def _transform_structural_cloud(structural_cloud, transform):
    sampled = o3d.geometry.PointCloud(structural_cloud.sampled_pcd)
    grouped = o3d.geometry.PointCloud(structural_cloud.grouped_pcd)
    sampled.transform(transform)
    grouped.transform(transform)
    return StructuralCloud(
        sampled_pcd=sampled,
        grouped_pcd=grouped,
        labels=structural_cloud.labels.copy(),
        complexity_scores=structural_cloud.complexity_scores.copy(),
    )


def register_masks_to_target(paths, cache, target_index, mask_indices, settings, coarse_transform):
    transforms = [np.eye(4) for _ in paths]
    target_model = cache.get(target_index)

    coarse_is_identity = np.allclose(coarse_transform, np.eye(4))
    combined_sampled = o3d.geometry.PointCloud()

    for idx in tqdm(mask_indices, desc="Building combined mask source cloud"):
        source_model = cache.get(idx)
        sampled = o3d.geometry.PointCloud(source_model.sampled_pcd)
        if not coarse_is_identity:
            sampled.transform(coarse_transform)
        combined_sampled += sampled

    if len(combined_sampled.points) == 0:
        raise RuntimeError("Combined mask source cloud is empty")

    combined_source_model = build_structural_cloud(combined_sampled, settings)
    icp_transform = forestalign_pairwise(combined_source_model, target_model, settings)
    final_transform = icp_transform @ coarse_transform

    for idx in mask_indices:
        transforms[idx] = final_transform

    return transforms


def register_multiview_scans(paths, cache, reference_index, pairing_mode, settings):
    transforms = [np.eye(4) for _ in paths]

    if pairing_mode == "reference":
        target_model = cache.get(reference_index)
        for idx in tqdm(range(len(paths)), desc="ForestAlign multi-view registration"):
            if idx == reference_index:
                continue
            transforms[idx] = forestalign_pairwise(cache.get(idx), target_model, settings)
        return transforms

    for idx in tqdm(range(reference_index + 1, len(paths)), desc="Forward sequential registration"):
        local_transform = forestalign_pairwise(cache.get(idx), cache.get(idx - 1), settings)
        transforms[idx] = transforms[idx - 1] @ local_transform

    for idx in tqdm(range(reference_index - 1, -1, -1), desc="Backward sequential registration"):
        local_transform = forestalign_pairwise(cache.get(idx), cache.get(idx + 1), settings)
        transforms[idx] = transforms[idx + 1] @ local_transform

    return transforms


def _write_cloud(path, pcd, ext):
    """Write a point cloud to .laz/.las/.ply based on extension."""
    if ext in [".laz", ".las"]:
        pts = np.asarray(pcd.points)
        header = laspy.LasHeader(point_format=0)
        header.offsets = np.min(pts, axis=0)
        header.scales = np.array([0.001, 0.001, 0.001])
        las_out = laspy.LasData(header)
        las_out.x = pts[:, 0]
        las_out.y = pts[:, 1]
        las_out.z = pts[:, 2]
        las_out.write(path)
    else:
        o3d.io.write_point_cloud(path, pcd, write_ascii=False, compressed=True)


def _apply_transform_to_xyz(x, y, z, transform):
    """Apply a 4x4 affine transform to coordinate arrays."""
    pts = np.column_stack((x, y, z))
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    pts_h = np.hstack((pts, ones))
    return (transform @ pts_h.T).T[:, :3]


def _write_transform_dat(path, transform):
    """Write a 4x4 transform matrix in DAT text format."""
    with open(path, "w", encoding="ascii") as handle:
        for row in transform:
            handle.write(" ".join(f"{value:.12f}" for value in row) + "\n")


def _transform_scan_to_points(path, transform, scanner_id=0):
    """Load one LAS/LAZ, apply the transform, set scanner ID, and return the full point record."""
    las = laspy.read(path)
    xyz_t = _apply_transform_to_xyz(las.x, las.y, las.z, transform)
    las.x = xyz_t[:, 0]
    las.y = xyz_t[:, 1]
    las.z = xyz_t[:, 2]
    if scanner_id > 0:
        las.point_source_id = np.full(len(las.points), scanner_id, dtype=np.uint16)
    return las.points


def _write_merged_las_preserve_fields(paths, transforms, output_file):
    """Write merged LAS/LAZ preserving all point fields from each scan with scanner IDs."""
    first = laspy.read(paths[0])
    header = laspy.LasHeader(point_format=first.header.point_format, version=first.header.version)
    header.scales = first.header.scales.copy()
    header.offsets = first.header.offsets.copy()

    with laspy.open(output_file, mode="w", header=header) as writer:
        for idx, path in enumerate(tqdm(paths, desc="Transforming full-res and writing merged LAS")):
            scanner_id = _extract_scanner_id(path, idx)
            writer.write_points(_transform_scan_to_points(path, transforms[idx], scanner_id=scanner_id))


def _write_transformed_las_preserve_fields(input_path, transform, output_path, scanner_id=0):
    las = laspy.read(input_path)
    xyz_t = _apply_transform_to_xyz(las.x, las.y, las.z, transform)
    las.x = xyz_t[:, 0]
    las.y = xyz_t[:, 1]
    las.z = xyz_t[:, 2]
    if scanner_id > 0:
        las.point_source_id = np.full(len(las.points), scanner_id, dtype=np.uint16)
    las.write(output_path)


def _load_full_scan_for_output(path, chunk_size):
    """Load the full-resolution scan for final export (no sampling cap)."""
    return load_scan_to_o3d(
        path=path,
        sample_step=1,
        max_points=0,
        chunk_size=chunk_size,
    )


def _extract_scanner_id(filename, index):
    """Extract scanner ID from filename if it matches ScanPos{number} pattern, else use index."""
    import re
    basename = os.path.basename(filename)
    match = re.search(r'ScanPos(\d+)', basename)
    if match:
        return int(match.group(1))
    return index + 1  # 1-based index fallback


def _detect_input_format(project_dir):
    laz_files = glob.glob(os.path.join(project_dir, "*.laz"))
    las_files = glob.glob(os.path.join(project_dir, "*.las"))
    ply_files = glob.glob(os.path.join(project_dir, "*.ply"))

    if laz_files:
        return os.path.join(project_dir, "*.laz"), ".laz"
    if las_files:
        return os.path.join(project_dir, "*.las"), ".las"
    if ply_files:
        return os.path.join(project_dir, "*.ply"), ".ply"
    raise RuntimeError("No .laz, .las, or .ply files found in the project directory")


def _list_supported_point_files(folder: Path) -> list[Path]:
    files = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if _is_supported_ext(p.suffix):
            files.append(p)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "ForestAlign registration for either multi-view scan sets (project_dir) "
            "or plot-to-plot alignment (--target + --mask_dir)."
        )
    )
    parser.add_argument(
        "project_dir",
        nargs="?",
        default=None,
        help="Directory containing multistation input scans (default mode)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Plot-to-plot mode target cloud (.laz/.las/.ply)",
    )
    parser.add_argument(
        "--mask_dir",
        type=Path,
        default=None,
        help="Plot-to-plot mode mask cloud directory (.laz/.las/.ply)",
    )
    parser.add_argument(
        "--dat_transform",
        type=Path,
        default=None,
        help="Optional 4x4 DAT matrix applied to masks before ICP (plot-to-plot mode)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Multistation: merged output file path. Plot-to-plot: per-mask output directory.",
    )
    parser.add_argument(
        "--pairing",
        choices=["reference", "sequential"],
        default="reference",
        help="How pairwise ForestAlign is extended across the multi-view scan set",
    )
    parser.add_argument(
        "--reference-scan",
        type=int,
        default=0,
        help="0-based scan index used as the reference scan or sequential anchor",
    )
    parser.add_argument(
        "--scene-preset",
        choices=sorted(SCENE_PRESETS.keys()),
        default="tls_forest",
        help="Scene preset controlling the number of structural complexity levels",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=SAMPLE_STEP,
        help="Keep every Nth point before voxelization for registration",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=MAX_POINTS_PER_SCAN,
        help="Maximum sampled points per scan used during registration (0 = unlimited)",
    )
    parser.add_argument(
        "--icp-iters",
        type=int,
        default=ICP_ITERS,
        help="Maximum ICP iterations per stage",
    )
    parser.add_argument(
        "--pre-coarse-voxel-size",
        type=float,
        default=0.0,
        help="Optional pre-coarse voxel size (0 = disabled). If set, runs before the main coarse/fine stages.",
    )
    parser.add_argument(
        "--pre-coarse-icp-dist",
        type=float,
        default=0.0,
        help="Optional pre-coarse ICP distance (0 = disabled). Must be set together with --pre-coarse-voxel-size.",
    )
    parser.add_argument(
        "--outlier-removal",
        action="store_true",
        help="Enable statistical outlier removal per-scan to filter wind noise before registration",
    )
    parser.add_argument(
        "--outlier-nb-neighbors",
        type=int,
        default=OUTLIER_NB_NEIGHBORS,
        help="Number of neighbors for statistical outlier detection (larger = more conservative)",
    )
    parser.add_argument(
        "--outlier-std-ratio",
        type=float,
        default=OUTLIER_STD_RATIO,
        help="Standard deviation ratio threshold for outlier removal (larger = fewer points removed)",
    )
    parser.add_argument(
        "--write-transforms",
        action="store_true",
        help="Write per-scan 4x4 transforms as <basename>_transform.DAT",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip the final merged output and force writing per-scan transform DAT files",
    )
    args = parser.parse_args()

    _require_dependencies()

    if args.project_dir is not None:
        args.project_dir = str(normalize_cli_path(Path(args.project_dir), must_exist=True))
    if args.target is not None:
        args.target = normalize_cli_path(args.target, must_exist=True)
    if args.mask_dir is not None:
        args.mask_dir = normalize_cli_path(args.mask_dir, must_exist=True)
    if args.dat_transform is not None:
        args.dat_transform = normalize_cli_path(args.dat_transform, must_exist=True)

    has_plot_inputs = (args.target is not None) or (args.mask_dir is not None)
    if has_plot_inputs and not (args.target is not None and args.mask_dir is not None):
        raise RuntimeError("Provide both --target and --mask_dir for plot-to-plot mode")

    plot_mode = args.target is not None and args.mask_dir is not None

    if not plot_mode:
        if args.project_dir is None:
            raise RuntimeError("project_dir is required unless both --target and --mask_dir are provided")
        if not os.path.isdir(args.project_dir):
            raise RuntimeError(f"Project directory not found: {args.project_dir}")

    if plot_mode:
        if not args.target.is_file():
            raise RuntimeError(f"Target file not found: {args.target}")
        if not args.mask_dir.is_dir():
            raise RuntimeError(f"Mask directory not found: {args.mask_dir}")

    if args.sample_step < 1:
        raise RuntimeError("--sample-step must be >= 1")
    if args.max_points < 0:
        raise RuntimeError("--max-points must be >= 0")
    if args.icp_iters <= 0:
        raise RuntimeError("--icp-iters must be > 0")
    if args.pre_coarse_voxel_size < 0:
        raise RuntimeError("--pre-coarse-voxel-size must be >= 0")
    if args.pre_coarse_icp_dist < 0:
        raise RuntimeError("--pre-coarse-icp-dist must be >= 0")
    if (args.pre_coarse_voxel_size > 0) != (args.pre_coarse_icp_dist > 0):
        raise RuntimeError("--pre-coarse-voxel-size and --pre-coarse-icp-dist must both be set or both disabled")
    if args.outlier_nb_neighbors < 5:
        raise RuntimeError("--outlier-nb-neighbors must be >= 5")
    if args.outlier_std_ratio <= 0:
        raise RuntimeError("--outlier-std-ratio must be > 0")

    coarse_transform = np.eye(4, dtype=np.float64)
    if args.dat_transform is not None:
        coarse_transform = read_dat_matrix(args.dat_transform)

    if plot_mode:
        target_ext = args.target.suffix.lower()
        if not _is_supported_ext(target_ext):
            raise RuntimeError(f"Unsupported target extension: {target_ext}")

        raw_mask_paths = _list_supported_point_files(args.mask_dir)
        target_abs = args.target.resolve()
        mask_paths = [str(p) for p in raw_mask_paths if p.resolve() != target_abs]

        if len(mask_paths) < 1:
            raise RuntimeError("Need at least 1 mask scan in --mask_dir for plot-to-plot mode")

        paths = [str(args.target)] + mask_paths
        target_index = 0
        mask_indices = list(range(1, len(paths)))

        if args.output:
            output_dir = Path(args.output).expanduser()
            if output_dir.suffix.lower() in [".laz", ".las", ".ply"]:
                raise RuntimeError("In plot-to-plot mode, --output must be a directory path")
            if not output_dir.is_absolute():
                output_dir = Path.cwd() / output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = args.mask_dir / "aligned"
            output_dir.mkdir(parents=True, exist_ok=True)

        input_ext = target_ext
        output_ext = None
        output_file = None
    else:
        input_glob, input_ext = _detect_input_format(args.project_dir)
        paths = sorted(glob.glob(input_glob))
        if len(paths) < 2:
            raise RuntimeError("Need at least 2 scans for ForestAlign registration")
        if args.reference_scan < 0 or args.reference_scan >= len(paths):
            raise RuntimeError("--reference-scan must be within the scan index range")

        if args.output:
            output_file = args.output
            _, ext = os.path.splitext(output_file)
            output_ext = ext.lower()
            if output_ext not in [".laz", ".las", ".ply"]:
                raise RuntimeError("Output format must be .laz, .las, or .ply")
        else:
            output_ext = input_ext
            output_file = os.path.join(args.project_dir, f"merged_forestalign_refined{output_ext}")

    settings = ForestAlignSettings(
        complexity_levels=SCENE_PRESETS[args.scene_preset],
        icp_iters=args.icp_iters,
        outlier_removal=args.outlier_removal,
        outlier_nb_neighbors=args.outlier_nb_neighbors,
        outlier_std_ratio=args.outlier_std_ratio,
        pre_coarse_voxel_size=args.pre_coarse_voxel_size,
        pre_coarse_icp_dist=args.pre_coarse_icp_dist,
    )
    cache = PointCloudCache(
        paths=paths,
        sample_step=args.sample_step,
        max_points=args.max_points,
        chunk_size=LAS_CHUNK_SIZE,
        cache_size=CACHE_SIZE,
        settings=settings,
    )

    if plot_mode:
        print(f"Mode: plot-to-plot (target + masks)")
        print("Registration: one-to-one (combined masks -> target)")
        print(f"Target: {args.target}")
        print(f"Found {len(mask_indices)} mask scan(s) (.ply/.las/.laz)")
        print(f"Per-mask outputs directory: {output_dir}")
        if args.dat_transform is not None:
            print(f"DAT coarse transform: {args.dat_transform}")
    else:
        print(f"Mode: multistation")
        print(f"Found {len(paths)} scans ({input_ext})")
    
    pipeline_desc = "Coarse (0.05m/0.03m) → Fine (0.015m/0.01m)"
    if settings.pre_coarse_voxel_size > 0:
        pipeline_desc = f"Pre-coarse ({settings.pre_coarse_voxel_size}m/{settings.pre_coarse_icp_dist}m) → {pipeline_desc}"
    
    if plot_mode:
        print(
            "ForestAlign settings: "
            f"scene_preset={args.scene_preset}, "
            f"complexity_levels={settings.complexity_levels}, "
            f"pipeline={pipeline_desc}, "
            f"icp_iters={settings.icp_iters}, "
            f"sample_step={args.sample_step}, "
            f"max_points={args.max_points}, "
            f"outlier_removal={settings.outlier_removal}, "
            f"outlier_nb_neighbors={settings.outlier_nb_neighbors}, "
            f"outlier_std_ratio={settings.outlier_std_ratio}"
        )

        transforms = register_masks_to_target(
            paths=paths,
            cache=cache,
            target_index=target_index,
            mask_indices=mask_indices,
            settings=settings,
            coarse_transform=coarse_transform,
        )

        if args.write_transforms or args.no_merge:
            print("Writing per-mask transform DAT files")
            for idx in mask_indices:
                basename = os.path.splitext(os.path.basename(paths[idx]))[0]
                dat_path = output_dir / f"{basename}_transform.DAT"
                _write_transform_dat(str(dat_path), transforms[idx])

        print("Writing aligned mask outputs")
        for idx in tqdm(mask_indices, desc="Writing per-mask aligned outputs"):
            in_path = paths[idx]
            in_ext = os.path.splitext(in_path)[1].lower()
            basename = os.path.splitext(os.path.basename(in_path))[0]
            out_path = output_dir / f"{basename}_aligned{in_ext}"

            if in_ext in [".laz", ".las"]:
                scanner_id = _extract_scanner_id(in_path, idx)
                _write_transformed_las_preserve_fields(
                    in_path,
                    transforms[idx],
                    str(out_path),
                    scanner_id=scanner_id,
                )
            else:
                pcd_full = _load_full_scan_for_output(in_path, LAS_CHUNK_SIZE)
                pcd_full.transform(transforms[idx])
                _write_cloud(str(out_path), pcd_full, in_ext)
    else:
        print(
            "ForestAlign settings: "
            f"pairing={args.pairing}, "
            f"reference_scan={args.reference_scan}, "
            f"scene_preset={args.scene_preset}, "
            f"complexity_levels={settings.complexity_levels}, "
            f"pipeline={pipeline_desc}, "
            f"icp_iters={settings.icp_iters}, "
            f"sample_step={args.sample_step}, "
            f"max_points={args.max_points}, "
            f"outlier_removal={settings.outlier_removal}, "
            f"outlier_nb_neighbors={settings.outlier_nb_neighbors}, "
            f"outlier_std_ratio={settings.outlier_std_ratio}"
        )

        transforms = register_multiview_scans(
            paths,
            cache,
            args.reference_scan,
            args.pairing,
            settings,
        )

        if args.write_transforms or args.no_merge:
            print("Writing per-scan transform DAT files")
            for idx, in_path in enumerate(paths):
                basename = os.path.splitext(os.path.basename(in_path))[0]
                dat_path = os.path.join(args.project_dir, f"{basename}_transform.DAT")
                _write_transform_dat(dat_path, transforms[idx])

        if not args.no_merge:
            if output_ext in [".laz", ".las"] and input_ext in [".laz", ".las"]:
                print(f"Writing merged output with preserved fields: {output_file}")
                _write_merged_las_preserve_fields(paths, transforms, output_file)
            else:
                print("Merging refined point clouds")
                merged = o3d.geometry.PointCloud()
                for idx, in_path in enumerate(tqdm(paths, desc="Loading full-res and merging")):
                    pcd_full = _load_full_scan_for_output(in_path, LAS_CHUNK_SIZE)
                    pcd_full.transform(transforms[idx])
                    merged += pcd_full

                print(f"Writing output: {output_file}")
                _write_cloud(output_file, merged, output_ext)

    print("Done.")


if __name__ == "__main__":
    main()