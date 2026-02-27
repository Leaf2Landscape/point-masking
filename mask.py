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

# --- PLY additive writer (valid final PLY) ---

class PlyAppender:
    """
    Accumulates vertices in a temporary binary file during the run.
    On finalize(), writes a single valid PLY (binary_little_endian) with the
    correct vertex count and appends the body.
    """
    def __init__(self, out_path: Path):
        self.out_path = out_path
        self.tmp_path = out_path.with_suffix(out_path.suffix + ".bin")
        # Fresh start for this run
        if self.out_path.exists():
            self.out_path.unlink()
        if self.tmp_path.exists():
            self.tmp_path.unlink()
        self.count = 0
        # Open temp body file in append-binary mode
        self.tmp_f = open(self.tmp_path, 'ab')

    def append(self, xyz: np.ndarray):
        """Append an (N,3) float array to the temp body as float32 LE."""
        if xyz is None or xyz.size == 0:
            return
        arr = np.asarray(xyz, dtype='<f4', order='C')  # float32 little-endian
        self.tmp_f.write(arr.tobytes())
        self.count += arr.shape[0]

    def close(self):
        if not self.tmp_f.closed:
            self.tmp_f.close()

    def finalize(self):
        """Write one valid PLY with correct header + body, then cleanup."""
        self.close()
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {self.count}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        ).encode('ascii')
        # Write header
        with open(self.out_path, 'wb') as fout:
            fout.write(header)
        # Append body
        with open(self.out_path, 'ab') as fout, open(self.tmp_path, 'rb') as ftmp:
            for chunk in iter(lambda: ftmp.read(1 << 20), b''):
                fout.write(chunk)
        # Remove temp
        try:
            os.remove(self.tmp_path)
        except OSError:
            pass

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

    if args.target.is_dir():
        target_exts = list(args.target.glob("*.ply")) + list(args.target.glob("*.las")) + list(args.target.glob("*.laz"))
        suffix = target_exts[0].suffix if target_exts else '.ply'
    else:
        suffix = args.target.suffix
    producing_ply = suffix.lower() == '.ply'
    
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

        out_path = out_dir / f"{f.stem}_mask{suffix}"
        
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
    removed_outputs = 0
    for m in masks:
        if m['out_path'].exists():
            m['out_path'].unlink()
            removed_outputs += 1
    if removed_outputs > 0:
        print(f"Removed {removed_outputs} existing output files to start fresh.")

    if producing_ply:
        for m in masks:
            m['writer'] = PlyAppender(m['out_path'])

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
                    
                    # Iterate Chunks
                    with tqdm(total=total_points, unit="pts") as pbar:
                        for start_idx in range(0, total_points, args.chunk_size):
                            end_idx = min(start_idx + args.chunk_size, total_points)
                            chunk_xyz = pts[start_idx:end_idx]
                            
                            # Query all masks at once for each point
                            matched_mask_idx = process_chunk(chunk_xyz, masks, args.distance)
                            
                            # Write points to their closest matching mask
                            for mask_idx, m in enumerate(masks):
                                sel = (matched_mask_idx == mask_idx)
                                if np.any(sel):
                                    m['writer'].append(chunk_xyz[sel])
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
                
    # Finalize all PLY outputs
    for m in masks:
        m['writer'].finalize()

    elapsed = time.time() - start_time
    print(f"\nDone. Processed {total_points} points in {elapsed:.2f}s.")

if __name__ == "__main__":
    main()