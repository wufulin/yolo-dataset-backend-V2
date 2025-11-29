import argparse
import random
import zipfile
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple
import time
from tqdm import tqdm
import hashlib

def parse_size(size_str):
    """Parse size string (e.g., '1GB', '500MB') to bytes."""
    units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}
    size_str = size_str.upper()
    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                return int(float(size_str[:-len(unit)]) * multiplier)
            except ValueError:
                pass
    return int(size_str)

def load_source_data(source_zip: Path) -> Tuple[List[Dict], bytes, str]:
    """Load all source files into memory at once (fixed path matching)"""
    source_pairs = []
    
    with zipfile.ZipFile(source_zip, 'r') as zf:
        all_files = set(zf.namelist())  # Use set for faster lookup
        
        # 1. Find YAML file
        yaml_files = [name for name in all_files if name.endswith(('.yaml', '.yml'))]
        if not yaml_files:
            raise ValueError(f"Could not find dataset YAML file in {source_zip}")
        
        yaml_path_str = yaml_files[0]
        yaml_data = zf.read(yaml_path_str)
        
        # 2. Determine dataset root directory (use forward slashes consistently)
        dataset_root = Path(yaml_path_str).parent
        root_prefix = str(dataset_root).replace('\\', '/') + '/' if str(dataset_root) != '.' else ''
        
        # 3. Collect all image-label pairs
        splits = ['train', 'val']
        
        for split in splits:
            img_dir_prefix = f"{root_prefix}images/{split}/"
            lbl_dir_prefix = f"{root_prefix}labels/{split}/"
            
            # Find all image files in this directory
            for file_path in all_files:
                if file_path.startswith(img_dir_prefix) and file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    # Build label path
                    img_name = Path(file_path).name
                    label_path = f"{lbl_dir_prefix}{Path(file_path).stem}.txt"
                    
                    if label_path in all_files:
                        # Read data into memory
                        img_data = zf.read(file_path)
                        label_data = zf.read(label_path)
                        
                        source_pairs.append({
                            'image_data': img_data,
                            'label_data': label_data,
                            'suffix': Path(file_path).suffix,
                            'split': split,
                            'total_size': len(img_data) + len(label_data)
                        })
        
        print(f"  Found {len(source_pairs)} valid image-label pairs")
        print(f"  Dataset root: {root_prefix if root_prefix else '(archive root)'}")
    
    if not source_pairs:
        raise ValueError("No valid image-label pairs found. Please check if the zip structure is correct.")
    
    return source_pairs, yaml_data, yaml_path_str

def generate_batch_optimized(args) -> List[Tuple[str, bytes]]:
    """Optimized worker process: generate a batch of file data"""
    batch_id, batch_target_size, source_pairs = args
    results = []
    current_size = 0
    
    # Use faster random number generation
    rng = random.Random(batch_id + int(time.time() * 1000) % 10000)
    
    # Precompute file path templates
    path_templates = {
        'train': ('images/train/{}{}', 'labels/train/{}.txt'),
        'val': ('images/val/{}{}', 'labels/val/{}.txt')
    }
    
    # Use simpler filename generation (avoid UUID overhead)
    file_counter = batch_id * 1000000
    
    while current_size < batch_target_size:
        pair = rng.choice(source_pairs)
        file_counter += 1
        
        # Generate simple but unique filename
        file_hash = hashlib.md5(f"{batch_id}_{file_counter}".encode()).hexdigest()[:16]
        
        target_split = pair['split']
        img_template, lbl_template = path_templates[target_split]
        
        # Build file paths
        img_path = img_template.format(file_hash, pair['suffix'])
        lbl_path = lbl_template.format(file_hash)
        
        # Add to results (use original data directly, avoid copying)
        results.append((img_path, pair['image_data']))
        results.append((lbl_path, pair['label_data']))
        
        current_size += pair['total_size']
    
    return results

def create_big_dataset_ultrafast(source_zip, target_size_str, output_zip=None, num_workers=None, compress_level=1):
    """Ultra-fast version: maximum speed optimization"""
    target_size = parse_size(target_size_str)
    
    if output_zip is None:
        output_zip = Path(source_zip).parent / f"coco8-detect-{target_size_str}.zip"
    
    output_path = Path(output_zip)
    print(f"{'='*60}")
    print(f"Source: {source_zip}")
    print(f"Target Size: {target_size_str} ({target_size:,} bytes)")
    print(f"Output: {output_path}")
    print(f"Workers: {num_workers or 'auto'}")
    print(f"Compression: Level {compress_level}")
    print(f"{'='*60}")
    
    # 1. Load all source data into memory at once
    print("\n[Step 1] Loading source data into memory...")
    start = time.time()
    source_pairs, yaml_data, yaml_path_str = load_source_data(Path(source_zip))
    
    total_source_size = sum(p['total_size'] for p in source_pairs)
    avg_pair_size = total_source_size / len(source_pairs)
    needed_count = int(target_size / avg_pair_size)
    
    print(f"  ✓ Loaded {len(source_pairs)} source pairs in {time.time()-start:.2f}s")
    print(f"  ✓ Average pair size: {avg_pair_size/1024:.2f} KB")
    print(f"  ✓ Estimated pairs needed: {needed_count:,}")
    
    # 2. Configure parallel tasks (dynamic adjustment based on data size)
    if num_workers is None:
        # Use more workers for large files
        if target_size > 10 * 1024**3:  # >10GB
            num_workers = min(cpu_count() * 2, 16)
        else:
            num_workers = min(cpu_count(), 8)
    
    # Ensure each worker has sufficient workload
    min_pairs_per_worker = 100
    if needed_count // num_workers < min_pairs_per_worker:
        num_workers = max(1, needed_count // min_pairs_per_worker)
    
    size_per_worker = target_size // num_workers
    print(f"\n[Step 2] Generating data with {num_workers} workers...")
    print(f"  Size per worker: {size_per_worker/1024/1024:.2f} MB")
    
    # 3. Multi-process generation with batch writing
    start_gen = time.time()
    
    with Pool(processes=num_workers) as pool:
        tasks = [(i, size_per_worker, source_pairs) for i in range(num_workers)]
        
        # Use lower compression level for faster writing
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=compress_level) as zf:
            # Write YAML file
            zf.writestr(Path(yaml_path_str).name, yaml_data)
            
            # Pre-create directory structure (optional, but helps with some filesystem optimizations)
            for split in ['train', 'val']:
                zf.writestr(f"images/{split}/", b'')
                zf.writestr(f"labels/{split}/", b'')
            
            # Batch process results
            batch_size = 1000  # Number of files to process per batch
            current_batch = []
            
            with tqdm(total=needed_count, unit='pairs', desc="  Progress") as pbar:
                for worker_results in pool.imap_unordered(generate_batch_optimized, tasks):
                    for file_path, file_data in worker_results:
                        current_batch.append((file_path, file_data))
                        
                        # Batch write
                        if len(current_batch) >= batch_size:
                            for path, data in current_batch:
                                zf.writestr(path, data)
                            pbar.update(len(current_batch) // 2)  # Each pair contains 2 files
                            current_batch = []
                    
                    # Write remaining batch
                    if current_batch:
                        for path, data in current_batch:
                            zf.writestr(path, data)
                        pbar.update(len(current_batch) // 2)
                        current_batch = []
    
    elapsed = time.time() - start_gen
    print(f"\n[Step 3] Zipping completed in {elapsed:.2f}s")
    
    # Verify output
    final_size = output_path.stat().st_size
    print(f"\n{'='*60}")
    print(f"✓ Dataset created successfully!")
    print(f"  Location: {output_path}")
    print(f"  Final size: {final_size / 1024 / 1024:.2f} MB ({final_size:,} bytes)")
    print(f"  Speed: {final_size / 1024 / 1024 / elapsed:.2f} MB/s")
    print(f"  Compression ratio: {final_size/target_size:.2%}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a large YOLO dataset from coco8-detect (Ultra Fast)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_big_datasets.py 10GB
  python create_big_datasets.py 10GB --workers 16 --output mydataset.zip --compress 0
        """
    )
    parser.add_argument("size", help="Target size (e.g., '1GB', '500MB')")
    parser.add_argument("--source", default="../data/coco8-detect.zip", help="Source zip path")
    parser.add_argument("--output", help="Output zip path")
    parser.add_argument("--workers", type=int, help="Number of workers (default: auto)")
    parser.add_argument("--compress", type=int, default=1, help="Compression level (0-9, 0=no compression, 1=fastest)")
    
    args = parser.parse_args()
    
    # Automatically find source file
    source_path = Path(args.source)
    possible_paths = [
        source_path,
        Path(__file__).parent / "data" / "coco8-detect.zip",
        Path(__file__).parent.parent / "data" / "coco8-detect.zip",
        Path("../data/coco8-detect.zip")
    ]
    
    found_source = None
    for p in possible_paths:
        if p.exists():
            found_source = p
            break
    
    if not found_source:
        print(f"Error: Source file not found. Tried: {[str(p) for p in possible_paths]}")
        exit(1)
    
    create_big_dataset_ultrafast(
        source_zip=found_source,
        target_size_str=args.size,
        output_zip=args.output,
        num_workers=args.workers,
        compress_level=args.compress
    )
