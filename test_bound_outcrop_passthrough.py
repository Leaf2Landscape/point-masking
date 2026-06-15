"""
Test: bound==1 points outside the --crop zone are written to output with tree_id=0
      (not silently dropped), while bound==1 points inside the crop zone get real
      tree_ids and bound==0 points still get the -1 sentinel.

Layout
------
Mask file   : 5 points at (-1..1, -1..1, 0), tree_id=1, stem_id=1
Target file : 3 bound points at (0, 0, z)      -- inside crop zone, matched  → tree_id=1
              2 bound points at (1000, 0, z)    -- outside crop zone, dropped → tree_id=0
              2 unbound points at (2000, 0, z)  -- bound==0, always through  → tree_id=-1
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import laspy
import numpy as np
from laspy import ExtraBytesParams


def make_mask(path: Path) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.array([0.0, 0.0, 0.0])
    header.scales = np.array([0.001, 0.001, 0.001])
    header.add_extra_dim(ExtraBytesParams("tree_id", type=np.int32))
    header.add_extra_dim(ExtraBytesParams("stem_id", type=np.int32))

    las = laspy.LasData(header=header)
    xs = np.array([-1.0, -1.0,  1.0,  1.0, 0.0])
    ys = np.array([-1.0,  1.0, -1.0,  1.0, 0.0])
    las.x = xs
    las.y = ys
    las.z = np.zeros(len(xs))
    las.tree_id = np.ones(len(xs), dtype=np.int32)
    las.stem_id = np.ones(len(xs), dtype=np.int32)
    las.write(str(path))


def make_target(path: Path) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.array([0.0, 0.0, 0.0])
    header.scales = np.array([0.001, 0.001, 0.001])
    header.add_extra_dim(ExtraBytesParams("bound", type=np.uint8))

    las = laspy.LasData(header=header)

    # 3 bound points inside crop zone, close to mask → tree_id=1
    in_x = np.array([0.0, 0.1, -0.1])
    in_y = np.array([0.0, 0.0,  0.0])
    in_z = np.array([0.1, 0.2,  0.3])
    in_b = np.array([1, 1, 1], dtype=np.uint8)

    # 2 bound points outside crop zone (~1000 m away) → tree_id=0
    out_x = np.array([1000.0, 1000.1])
    out_y = np.array([0.0,    0.0   ])
    out_z = np.array([1.0,    2.0   ])
    out_b = np.array([1, 1], dtype=np.uint8)

    # 2 unbound points (~2000 m away) → tree_id=-1
    ub_x = np.array([2000.0, 2000.1])
    ub_y = np.array([0.0,    0.0   ])
    ub_z = np.array([1.0,    2.0   ])
    ub_b = np.array([0, 0], dtype=np.uint8)

    las.x = np.concatenate([in_x, out_x, ub_x])
    las.y = np.concatenate([in_y, out_y, ub_y])
    las.z = np.concatenate([in_z, out_z, ub_z])
    las.bound = np.concatenate([in_b, out_b, ub_b])
    las.write(str(path))


def run_mask(mask_path: Path, target_path: Path, out_dir: Path) -> Path:
    cmd = [
        sys.executable, "mask.py",
        "--mask", str(mask_path),
        "--target", str(target_path),
        "--output", str(out_dir),
        "--distance", "0.5",
        "--crop",
        "--fields", "tree_id,stem_id",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"mask.py exited with code {result.returncode}")
    return out_dir / f"{target_path.stem}_masked.las"


def check(output_path: Path) -> None:
    las = laspy.read(str(output_path))
    xs = np.asarray(las.x)
    tree_ids = np.asarray(las.tree_id)
    stem_ids = np.asarray(las.stem_id)
    bounds = np.asarray(las.bound)

    print(f"Output points : {len(xs)}")
    for i in range(len(xs)):
        print(f"  [{i}] x={xs[i]:.1f}  bound={bounds[i]}  tree_id={tree_ids[i]}  stem_id={stem_ids[i]}")

    # All 7 points must survive
    assert len(xs) == 7, f"Expected 7 points, got {len(xs)}"

    in_crop_mask  = bounds == 1
    out_crop_mask = (bounds == 1) & (xs > 999) & (xs < 1001)
    unbound_mask  = bounds == 0

    # Detect in-crop vs out-of-crop bound points by position
    in_crop_bound  = (bounds == 1) & (xs < 1)    # near origin
    out_crop_bound = (bounds == 1) & (xs > 999)  # ~1000 m away

    # In-crop bound points must get tree_id=1
    assert np.all(tree_ids[in_crop_bound] == 1), \
        f"In-crop bound tree_ids should be 1, got: {tree_ids[in_crop_bound]}"
    assert np.all(stem_ids[in_crop_bound] == 1), \
        f"In-crop bound stem_ids should be 1, got: {stem_ids[in_crop_bound]}"

    # Out-of-crop bound points must get tree_id=0 (fill, not dropped)
    assert np.any(out_crop_bound), "No out-of-crop bound points found in output — they were dropped!"
    assert np.all(tree_ids[out_crop_bound] == 0), \
        f"Out-of-crop bound tree_ids should be 0, got: {tree_ids[out_crop_bound]}"

    # Unbound points must get tree_id=-1
    assert np.all(tree_ids[unbound_mask] == -1), \
        f"Unbound tree_ids should be -1, got: {tree_ids[unbound_mask]}"

    print("All assertions passed.")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mask_path = tmp / "mask.las"
        target_path = tmp / "target.las"
        out_dir = tmp / "out"
        out_dir.mkdir()

        make_mask(mask_path)
        make_target(target_path)
        output_path = run_mask(mask_path, target_path, out_dir)
        check(output_path)


if __name__ == "__main__":
    main()
