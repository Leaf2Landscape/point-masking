#!/usr/bin/env python3
import argparse
import os
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData
import laspy
from tqdm import tqdm
from pathlib import Path
import contextlib

# --- Helper Functions ---

def get_points(filepath):
    """Reads XYZ coordinates from PLY or LAS/LAZ files efficiently."""
    ext = Path(filepath).suffix.lower()
    try:
        if ext == '.ply':
            # PLY: Read using plyfile
            with open(filepath, 'rb') as f:
                ply = PlyData.read(f)
                v = ply['vertex']
                return np.vstack((v['x'], v['y'], v['z'])).T
        elif ext in ['.las', '.laz']:
            # LAS/LAZ: Read using laspy
            with laspy.open(filepath) as f:
                las = f.read()
                return np.vstack((las.x, las.y, las.z)).T
    except Exception as e:
        print(f"Warning: Failed to read {filepath}: {e}")
        return np.array([])
    return np.array([])

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="Optimized Single-Pass Point Mask (Multiplexed)")
    parser.add_argument("-m", "--mask-folder", required=True, type=Path, help="Folder containing mask files")
    parser.add_argument("-t", "--target", required=True, type=Path, help="Large target LAS/LAZ file")
    parser.add_argument("-o", "--output", type=Path, help="Output directory (optional)")
    parser.add_argument("-d", "--distance", type=float, required=True, help="Distance threshold")
    parser.add_argument("--chunk-size", type=int, default=500000, help="Points per chunk (default: 500,000)")
    
    args = parser.parse_args()

    # 1. Setup Directories
    if args.output:
        out_dir = args.output
    else:
        out_dir = Path(f"{args.mask_folder.name}_extracted")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load All Masks into Memory
    masks = []
    mask_files = list(args.mask_folder.glob("*.ply")) + \
                 list(args.mask_folder.glob("*.las")) + \
                 list(args.mask_folder.glob("*.laz"))

    if not mask_files:
        print("No mask files found.")
        sys.exit(1)

    print(f"Loading {len(mask_files)} masks...")
    
    for f in tqdm(mask_files, unit="mask"):
        pts = get_points(f)
        if len(pts) == 0: continue
        
        # Build optimized KD-Tree
        # leafsize=64 is often faster for 3D point data than default
        tree = cKDTree(pts, leafsize=64, compact_nodes=True, balanced_tree=True)
        
        # Calculate bounds for Fast-Reject BBox check
        mins = np.min(pts, axis=0) - args.distance
        maxs = np.max(pts, axis=0) + args.distance
        
        # Determine output filename (MaskName + TargetExtension)
        # e.g. tree_01.ply -> tree_01.las
        out_path = out_dir / (f.stem + args.target.suffix)
        
        masks.append({
            'name': f.name,
            'tree': tree,
            'min': mins,
            'max': maxs,
            'out_path': out_path,
            'writer': None
        })

    print(f"Processing target: {args.target}")
    print(f"Writing outputs to: {out_dir}")
    
    start_time = time.time()

    # 3. Stream Target ONCE
    # ExitStack ensures all file handles (input + all outputs) close safely
    with contextlib.ExitStack() as stack:
        try:
            # Open Input
            src = stack.enter_context(laspy.open(args.target))
            total_points = src.header.point_count
            
            # Open All Outputs
            # Note: This requires system open file limit > len(masks)
            # Linux default is 1024. If you have >1000 masks, run `ulimit -n 4096`
            for m in masks:
                m['writer'] = stack.enter_context(
                    laspy.open(m['out_path'], mode='w', header=src.header)
                )

            # Iterate Chunks
            iterator = src.chunk_iterator(args.chunk_size)
            
            with tqdm(total=total_points, unit="pts") as pbar:
                for chunk in iterator:
                    # Convert chunk to numpy (Nx3)
                    # We do this once per chunk, not per mask
                    chunk_xyz = np.vstack((chunk.x, chunk.y, chunk.z)).T
                    
                    # Calculate Chunk BBox (Fast Reject)
                    c_min = np.min(chunk_xyz, axis=0)
                    c_max = np.max(chunk_xyz, axis=0)

                    # Check against every mask
                    for m in masks:
                        # A. Global BBox Check (Does chunk overlap mask at all?)
                        # Logic: If chunk_min > mask_max OR chunk_max < mask_min, no overlap
                        if np.any(c_min > m['max']) or np.any(c_max < m['min']):
                            continue
                        
                        # B. Fine BBox Filter (Points inside mask bounds)
                        # Create boolean mask for points roughly near the tree
                        in_box = np.all((chunk_xyz >= m['min']) & (chunk_xyz <= m['max']), axis=1)
                        
                        if not np.any(in_box):
                            continue

                        # C. KD-Tree Query (Only on 'in_box' points)
                        candidates = chunk_xyz[in_box]
                        
                        # Optimization: distance_upper_bound
                        # Prunes tree search immediately if point is > distance
                        # Returns 'inf' for points outside distance
                        dists, _ = m['tree'].query(candidates, k=1, distance_upper_bound=args.distance, workers=1)
                        
                        valid_prox = (dists != float('inf'))

                        if np.any(valid_prox):
                            # Map back to original chunk indices
                            final_mask = np.zeros(len(chunk), dtype=bool)
                            final_mask[in_box] = valid_prox
                            
                            # Write points
                            m['writer'].write_points(chunk[final_mask])

                    pbar.update(len(chunk))
                    
        except OSError as e:
            if "Too many open files" in str(e):
                print("\n[Error] Too many open files.")
                print("Run 'ulimit -n 4096' in your terminal and try again.")
            else:
                raise e

    elapsed = time.time() - start_time
    print(f"\nDone. Processed {total_points} points in {elapsed:.2f}s.")

if __name__ == "__main__":
    main()