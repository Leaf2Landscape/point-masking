# Point Masking

A high-performance Python tool for extracting subsets of points from a massive target point cloud based on proximity to multiple "mask" objects (e.g., extracting individual trees from a forest plot using tree stems as masks).

## Features

- **Single-Pass Multiplexing:** Reads the massive target file once, checking against all masks simultaneously.
- **Attribute Preservation:** Keeps **all** original LAS/LAZ attributes (Intensity, GPS Time, Return Number, Classification, etc.).
- **Optimized Performance:** Uses KD-Tree spatial indexing with "Fast Reject" bounding boxes and C-level distance pruning.
- **Memory Efficient:** Streams the target file in chunks; only mask points are held in RAM.

## Installation

You need Python 3 and a few scientific libraries.

```bash
# Install dependencies
pip install numpy scipy plyfile tqdm

# Install laspy with LAZ compression support
pip install "laspy[lazrs]"
```

## Usage
```
python py_point_mask.py
  --mask-folder ./my_masks/    # Folder containing mask files (.ply, .las, .laz)
  --target ./plot_cloud.laz    # The single large target point cloud
  --distance 0.5               # Distance threshold (in file units)
  --output ./extracted_trees   # (Optional) Output directory
  --chunk-size 500000          # (Optional) Points per chunk (Default: 500000)
```
