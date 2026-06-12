"""
Test: unbound points (bound == 0) pass through unchanged in single-file mode,
      even when --crop is active and they are far outside the crop zone.

Layout
------
Mask file   : 5 points at (0, 0, z), tree_id=1, stem_id=1
Target file : 3 bound points at (0, 0, z)   -- inside crop zone, should be masked
              2 unbound points at (1000, 0, z) -- far outside crop zone, should pass through
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
    # Spread in XY so crop_bounds_file covers [-1, 1] x [-1, 1],
    # ensuring all bound target points (within 0.1 m of origin) are inside the crop zone.
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

    # 3 bound points near the mask (inside crop zone, within 0.5 m in 3D)
    bound_x = np.array([0.0, 0.1, -0.1])
    bound_y = np.array([0.0, 0.0,  0.0])
    bound_z = np.array([0.1, 0.2,  0.3])
    bound_b = np.array([1, 1, 1], dtype=np.uint8)

    # 2 unbound points far away (~1000 m from the mask)
    unbound_x = np.array([1000.0, 1000.1])
    unbound_y = np.array([0.0,    0.0   ])
    unbound_z = np.array([1.0,    2.0   ])
    unbound_b = np.array([0, 0], dtype=np.uint8)

    las.x = np.concatenate([bound_x, unbound_x])
    las.y = np.concatenate([bound_y, unbound_y])
    las.z = np.concatenate([bound_z, unbound_z])
    las.bound = np.concatenate([bound_b, unbound_b])
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

    # All 5 points must survive
    assert len(xs) == 5, f"Expected 5 points, got {len(xs)}"

    bound_mask = bounds != 0
    unbound_mask = bounds == 0

    # Bound points must have tree_id == 1 and stem_id == 1
    assert np.all(tree_ids[bound_mask] == 1), f"Bound tree_ids wrong: {tree_ids[bound_mask]}"
    assert np.all(stem_ids[bound_mask] == 1), f"Bound stem_ids wrong: {stem_ids[bound_mask]}"

    # Unbound points must have sentinel fill value -1 (int32 fields)
    assert np.all(tree_ids[unbound_mask] == -1), f"Unbound tree_ids should be -1, got: {tree_ids[unbound_mask]}"
    assert np.all(stem_ids[unbound_mask] == -1), f"Unbound stem_ids should be -1, got: {stem_ids[unbound_mask]}"

    # Unbound XY must be preserved (~1000 m away)
    unbound_xs = xs[unbound_mask]
    assert np.all(unbound_xs > 999), f"Unbound points should be ~1000 m away, got: {unbound_xs}"

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
