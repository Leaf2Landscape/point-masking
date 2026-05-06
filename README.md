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

- many mask files
- one large target cloud (or a folder of large target files)
- a need to extract points to the closest mask

### What it does

- Loads all masks and builds KD trees.
- Streams each target cloud in chunks.
- Assigns each target point to the closest mask within the configured distance threshold.
- Writes one output per mask.
- Preserves source point attributes for LAS/LAZ workflows.

### Example

```bash
python mask.py \
  --mask-folder ./masks \
  --target ./plot_cloud.laz \
  --distance 0.5 \
  --output ./masked_output \
  --chunk-size 500000
```

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
