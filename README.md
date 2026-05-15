# Point Masking

This repository contains two related tools:

- `mask.py`: multi-mask extraction from one large target point cloud
- `segfix_trees.py`: per-tree segmentation correction for RayCloudTools outputs using masks from the same site

## Installation

```bash
pip install numpy scipy plyfile tqdm
pip install "laspy[lazrs]"
```

## Tool 1: mask.py

Use `mask.py` when you have:

- a folder of mask files named as `{tree_id}_{stem_id}` or `{tree_id}`
- a target point cloud (single file or folder) to assign against masks
- a need to produce mask-wise outputs with explicit IDs

### What it does

- Parses IDs from mask filenames:
  - `{tree_id}_{stem_id}` -> uses both values
  - `{tree_id}` -> defaults `stem_id=1`
- Validates ID ranges:
  - `tree_id`: `1..19999`
  - `stem_id`: `1..26`
- Groups duplicate mask IDs and treats them as one unique mask.
- Uses nearest-distance search (KD-tree) as the primary inclusion method.
- By default writes both:
  - LAS/LAZ outputs per unique mask with extra dimensions `tree_id` and `stem_id`
  - PLY outputs per unique mask with fields `(x, y, z, tree_id, stem_id)`

### Optional one-action modes

- `--ids-only`: write only LAS/LAZ with `tree_id` and `stem_id`
- `--ply-only`: write only PLY outputs

### Optional occlusion-fill pass

- `--hull-fill` enables a secondary inclusion pass for unmatched points.
- Nearest-distance assignment remains primary; hull fill only adds missed points.
- Hull support points are built from voxelized z-slices using the extreme XY point rule.
- Related flags:
  - `--decimation-size` (required with `--hull-fill`)
  - `--vox-mul` (default `3`, voxel size multiplier)
  - `--hull-eps` (default `0.05`, near-hull tolerance)

### Example (default: write LAS/LAZ + PLY)

```bash
python mask.py \
  --mask-folder ./masks \
  --target ./plot_cloud.laz \
  --distance 0.5 \
  --output ./masked_output \
  --chunk-size 500000
```

### Example (PLY only + hull fill)

```bash
python mask.py \
  --mask-folder ./masks \
  --target ./plot_cloud.ply \
  --distance 0.5 \
  --ply-only \
  --hull-fill \
  --decimation-size 0.03 \
  --vox-mul 3 \
  --hull-eps 0.05
```

### Note for future output policy

Current default is to produce both LAS/LAZ (with ID extras) and PLY per-mask outputs. This can be changed later in `mask.py` by adjusting `write_las` / `write_ply` default selection logic in `main()`.

## Tool 2: segfix_trees.py

Use `segfix_trees.py` when you have RayCloudTools automated segmentation outputs split per tree and want to correct them using an existing mask set from the same site.

### Refactored behavior

- Inputs are target tree files (one tree per file) and mask files.
- Candidate masks are selected only by overlapping bounding boxes.
- For each target point, the script assigns to the closest overlapping mask.
- Confidence is radius-based:
  - matched: closest distance <= `--distance`
  - uncertain: closest distance > `--distance`
- Outputs are written per mask tree:
  - `{mask}_matched.ply`
  - `{mask}_uncertain.ply`
- If a target tree has no overlapping masks, the original target file is copied through unchanged.

This guarantees points are not dropped during assignment for overlap cases: each point is routed to one closest overlapping mask, then split into matched vs uncertain by radius.

### Example

```bash
python segfix_trees.py \
  ./target_trees \
  -m ./site_masks \
  -o ./segfix_output \
  --distance 0.5 \
  --chunk-size 500000 \
  --workers 4 \
  --save-target-filename
```

### Outputs and reports

- `current_id_map.csv`: target ID to filename mapping (legacy filename, target-based columns: `target_id,target_filename`)
- `link_report.csv`: selected mask, overlap count, matched/uncertain totals per target tree
- `merge_report.csv`: matched/uncertain totals per mask tree with contributing target IDs

## Tool 3: mask_align.py

Use `mask_align.py` when you need to align mask coordinates to a target cloud using:

- optional coarse transform from a DAT 4x4 matrix
- fine alignment by ICP on the target and the combined masks

### What it does

- Reads all masks in `mask_dir` and treats them as one combined source for ICP.
- Optionally applies `--dat_transform` first as a coarse source-to-target transform.
- Builds memory-capped sampled point sets (voxel-thinned in chunks) for both source and target.
- Runs ICP on sampled sets (Open3D backend when available, SciPy fallback).
- Applies the final transform to every original mask file and writes transformed copies.

### Outputs

- transformed mask files in `--output_dir` or default `mask_dir/transformed_to_target`
- final transform matrix in DAT format: `mask_to_target_transform.DAT`
- ICP report JSON: `icp_report.json` (fitness, RMSE, residual summary, matrices, file list)
- one-row CSV summary: `icp_summary.csv`
- QC plots under `qc_plots/` (only when `--debug` is enabled):
  - random per-mask 3-view figures (`*_views.png`): top, front, aerial
  - `qc_overview_2x2.png` from 4 randomly selected masks

QC colors:
- green: target points matched to mask
- black: target points in mask bbox but unmatched
- red: mask points with no match in target

### Example

```bash
python mask_align.py \
  ./plot_cloud.laz \
  ./masks \
  --dat_transform ./coarse_align.DAT \
  --output_dir ./masks_aligned \
  --voxel_size 0.10 \
  --max_points_target 800000 \
  --max_points_masks 800000 \
  --icp_threshold 1.0 \
  --icp_max_iter 60
```

### QC plot controls

- `--plot_count` (default `4`): number of random masks to visualize
- `--plot_seed` (default `42`): random seed for reproducible mask selection
- `--plot_match_distance` (default `--icp_threshold`): match threshold used for QC colors
- `--plot_target_max_points` (default `300000`): target sample cap for plots
- `--plot_mask_max_points` (default `120000`): per-mask sample cap for plots
- `--plot_voxel_size` (default `0.10`): voxel thinning size for target plot sampling
- `--debug`: required to generate QC plots

### Reuse mode

- `--reuse-existing-masks`: if transformed masks and `mask_to_target_transform.DAT` already exist in `output_dir`, the script reuses them instead of recomputing alignment and rewriting masks.
- This is useful when rerunning the same site and you want to keep the same transformed masks.

### Notes

- Sampling is memory-capped by `--max_points_target` and `--max_points_masks` for large datasets.
- Larger `--voxel_size` reduces memory/time but may reduce alignment detail.
- If Open3D is not available, the script falls back to a SciPy-based ICP implementation.
