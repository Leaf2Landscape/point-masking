#!/usr/bin/env python3
import argparse
import os
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
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

def process_chunk(chunk_xyz, masks, distance):
    """Processes a chunk of points against all masks and returns a mapping of mask index to points."""
    min_dists = np.full(len(chunk_xyz), float('inf'))
    matched_mask_idx = np.full(len(chunk_xyz), -1, dtype=int)
    
    for mask_idx, m in enumerate(masks):
        # A. Global BBox Check
        c_min = np.min(chunk_xyz, axis=0)
        c_max = np.max(chunk_xyz, axis=0)
        if np.any(c_min > m['max']) or np.any(c_max < m['min']):
            continue
        
        # B. Fine BBox Filter
        in_box = np.all((chunk_xyz >= m['min']) & (chunk_xyz <= m['max']), axis=1)
        
        if not np.any(in_box):
            continue

        # C. KD-Tree Query
        candidates = chunk_xyz[in_box]
        dists, _ = m['tree'].query(candidates, k=1, distance_upper_bound=distance, workers=1)
        
        valid_prox = (dists != float('inf'))
        
        # Update minimum distances and track which mask is closest
        in_box_indices = np.where(in_box)[0]
        for local_idx, global_idx in enumerate(in_box_indices):
            if valid_prox[local_idx] and dists[local_idx] < min_dists[global_idx]:
                min_dists[global_idx] = dists[local_idx]
                matched_mask_idx[global_idx] = mask_idx
    
    return matched_mask_idx

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="Optimized Single-Pass Point Mask (Multiplexed)")
    parser.add_argument("-m", "--mask-folder", required=True, type=Path, help="Folder containing mask files")
    parser.add_argument("-t", "--target", required=True, type=Path, help="Large target LAS/LAZ/PLY file or folder containing multiple target files")
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

    # 3. Check whether target is a single file or a folder
    if args.target.is_dir():
        target_files = list(args.target.glob("*.ply")) + \
                       list(args.target.glob("*.las")) + \
                       list(args.target.glob("*.laz"))
        if not target_files:
            print("No target files found in the specified folder.")
            sys.exit(1)
        print(f"Found {len(target_files)} target files. Processing each sequentially...")
    elif args.target.is_file():
        target_files = [args.target]
    else:
        print("Invalid target path. Must be a file or directory.")
        sys.exit(1)

    # 4. Clear existing mask outputs if they exist (to avoid appending to old files)
    for m in masks:
        if m['out_path'].exists():
            print(f"Removing existing output file: {m['out_path']}")
            m['out_path'].unlink()

    # 5. For each target, stream ONCE
    # ExitStack ensures all file handles (input + all outputs) close safely
    for target_file in target_files:
        with contextlib.ExitStack() as stack:
            
            try:
                # Open Input
                if target_file.suffix.lower() in ['.las', '.laz']:
                    src = stack.enter_context(laspy.open(target_file))
                    total_points = src.header.point_count
                
                    # Open All Outputs
                    # Note: This requires system open file limit > len(masks)
                    # Linux default is 1024. If you have >1000 masks, run `ulimit -n 4096`
                    for m in masks:
                        file_mode = 'a' if m['out_path'].exists() else 'w'
                        m['writer'] = stack.enter_context(
                            laspy.open(m['out_path'], mode=file_mode, header=src.header)
                        )
                    
                    # Iterate Chunks
                    iterator = src.chunk_iterator(args.chunk_size)
                    
                    with tqdm(total=total_points, unit="pts") as pbar:
                        for chunk in iterator:
                            # Convert chunk to numpy (Nx3)
                            chunk_xyz = np.vstack((chunk.x, chunk.y, chunk.z)).T
                            
                            # Query all masks at once for each point
                            matched_mask_idx = process_chunk(chunk_xyz, masks, args.distance)
                            
                            # Write points to their closest matching mask
                            for mask_idx, m in enumerate(masks):
                                final_mask = matched_mask_idx == mask_idx
                                if np.any(final_mask):
                                    m['writer'].write_points(chunk[final_mask])

                            pbar.update(len(chunk))
                
                elif target_file.suffix.lower() == '.ply':
                    ply = PlyData.read(target_file)
                    total_points = ply['vertex'].count
                    pts = np.vstack((ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z'])).T
                    
                    # Open All Outputs
                    for m in masks:
                        # Append mode if file exists, write mode for first target
                        file_mode = 'ab' if m['out_path'].exists() else 'wb'
                        m['writer'] = stack.enter_context(open(m['out_path'], file_mode))
                        # Only write empty header for new files
                        if file_mode == 'wb':
                            ply['vertex'].data = np.empty(0, dtype=ply['vertex'].dtype())
                            PlyData(ply.elements).write(m['writer'])
                    
                    # Iterate Chunks
                    with tqdm(total=total_points, unit="pts") as pbar:
                        for start_idx in range(0, total_points, args.chunk_size):
                            end_idx = min(start_idx + args.chunk_size, total_points)
                            chunk_xyz = pts[start_idx:end_idx]
                            
                            # Query all masks at once for each point
                            matched_mask_idx = process_chunk(chunk_xyz, masks, args.distance)
                            
                            # Write points to their closest matching mask
                            for mask_idx, m in enumerate(masks):
                                final_mask = matched_mask_idx == mask_idx
                                if np.any(final_mask):
                                    filtered_data = chunk_xyz[final_mask] if chunk_xyz.size > 0 else np.empty((0, 3))
                                    if filtered_data.size > 0:
                                        # Create a new structured array for the filtered points
                                        vertex_dtype = ply['vertex'].dtype()
                                        filtered_structured = np.empty(filtered_data.shape[0], dtype=vertex_dtype)
                                        filtered_structured['x'] = filtered_data[:, 0]
                                        filtered_structured['y'] = filtered_data[:, 1]
                                        filtered_structured['z'] = filtered_data[:, 2]
                                        filtered_vertex = PlyElement.describe(filtered_structured, 'vertex')
                                        PlyData([filtered_vertex]).write(m['writer'])

                            pbar.update(end_idx - start_idx)
                
                else:
                    print(f"Unsupported target file format: {target_file.suffix}")
                    continue
                        
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