# Point Masking

This repository contains two related tools:

- `mask.py`: multi-mask extraction from one large target point cloud
- `segfix_trees.py`: per-tree segmentation correction for RayCloudTools outputs using masks from the same site

## Installation

Install the package from the repository root:

```bash
python -m pip install .
```

For editable development install:

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install .[plot]
python -m pip install .[all]
```

This package installs two console scripts:

- `point-masking-mask` -> runs `mask.py`
- `point-masking-segfix` -> runs `segfix_trees.py`

If you only need the core dependencies, the following are still required:

```bash
python -m pip install numpy scipy plyfile tqdm
python -m pip install "laspy[lazrs]"
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
- Optional QC/debug plotting (`--debug`) after assignment:
  - per-mask 3-view figures (`*_views.png`): top, front, aerial
  - `qc_overview_2x2.png` from sampled masks
  - persistent selection state in `qc_plots/debug_selection.json` for reproducible debug comparisons

### Optional one-action modes

- `--ids-only`: write only LAS/LAZ with `tree_id` and `stem_id`
- `--ply-only`: write only PLY outputs

### Optional occlusion-fill pass

- `--hull-fill` enables a secondary inclusion pass for unmatched points.
- Nearest-distance assignment remains primary; hull fill only adds missed points.
- Hull support points are built from voxelized z-slices using the extreme XY point rule.
- Related flags:
  - `--distance` (also used as hull support spacing base)
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

