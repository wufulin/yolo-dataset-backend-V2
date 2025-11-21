import argparse
import random
import shutil
import uuid
import zipfile
from pathlib import Path


def parse_size(size_str):
    """Parse size string (e.g., '1GB', '500MB') to bytes."""
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3,
        'TB': 1024**4
    }
    size_str = size_str.upper()
    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                number = float(size_str[:-len(unit)])
                return int(number * multiplier)
            except ValueError:
                pass
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"Invalid size format: {size_str}. Use format like '1GB', '500MB'.")

def create_big_dataset(source_zip, target_size_str, output_zip=None):
    # Parse target size
    try:
        target_size = parse_size(target_size_str)
    except ValueError as e:
        print(f"Error: {e}")
        return

    source_path = Path(source_zip)
    if not source_path.exists():
        print(f"Error: Source file {source_zip} not found.")
        return

    if output_zip is None:
        output_zip = source_path.parent / f"coco8-detect-{target_size_str}.zip"

    output_path = Path(output_zip)

    print(f"Source: {source_path}")
    print(f"Target Size: {target_size_str} ({target_size} bytes)")
    print(f"Output: {output_path}")

    # Temporary directory for extraction and construction
    temp_dir = Path("temp_dataset_creation")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    try:
        # 1. Extract source zip
        print("Extracting source dataset...")
        with zipfile.ZipFile(source_path, 'r') as zf:
            z_root = zipfile.Path(zf)
            # Find the root folder inside the zip (usually 'coco8-detect/')
            # Adjust depending on if zip has a top-level folder or not.
            # Based on `ls` output earlier, it seems `coco8-detect` is a directory inside `backend/data`,
            # but the zip likely contains `coco8-detect/` as root or contents directly.
            # Let's safely extract all.
            zf.extractall(temp_dir)

        # Locate the dataset root in extracted files
        # We expect structure: temp_dir/coco8-detect/...
        # Or temp_dir/... (if no root folder)

        # Find 'data.yaml' or 'coco8.yaml' to locate root
        yaml_files = list(temp_dir.rglob("*.yaml"))
        if not yaml_files:
             print("Error: Could not find dataset YAML file in source.")
             return

        dataset_root = yaml_files[0].parent
        print(f"Dataset root found at: {dataset_root}")

        images_dir = dataset_root / "images"
        labels_dir = dataset_root / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            print("Error: 'images' or 'labels' directory not found in dataset.")
            return

        # Collect source files
        splits = ['train', 'val'] # Standard YOLO splits
        source_pairs = [] # (image_path, label_path, split_name)

        for split in splits:
            split_img_dir = images_dir / split
            split_lbl_dir = labels_dir / split

            if not split_img_dir.exists():
                continue

            for img_file in split_img_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # Find corresponding label
                    label_file = split_lbl_dir / (img_file.stem + ".txt")
                    if label_file.exists():
                        source_pairs.append({
                            'image': img_file,
                            'label': label_file,
                            'split': split
                        })

        if not source_pairs:
            print("Error: No valid image-label pairs found.")
            return

        print(f"Found {len(source_pairs)} source pairs.")

        # Calculate average pair size to estimate count needed
        total_pair_size = 0
        for pair in source_pairs:
            total_pair_size += pair['image'].stat().st_size
            total_pair_size += pair['label'].stat().st_size

        avg_pair_size = total_pair_size / len(source_pairs)

        # Estimate overhead (zip compression headers etc, keeping it simple)
        # Target count
        needed_count = int(target_size / avg_pair_size)
        print(f"Estimated pairs needed: {needed_count}")

        # 2. Create new dataset structure
        new_dataset_root_name = output_path.stem
        build_dir = temp_dir / "build" / new_dataset_root_name
        build_dir.mkdir(parents=True)

        (build_dir / "images" / "train").mkdir(parents=True)
        (build_dir / "images" / "val").mkdir(parents=True)
        (build_dir / "labels" / "train").mkdir(parents=True)
        (build_dir / "labels" / "val").mkdir(parents=True)

        # Copy YAML
        yaml_src = yaml_files[0]
        shutil.copy(yaml_src, build_dir / yaml_src.name)

        # 3. Generate Data
        current_size = 0
        files_created = 0

        print("Generating data...")

        # We will keep adding files until we reach target size
        # We can cycle through source pairs randomly

        while current_size < target_size:
            # Pick a random source pair
            pair = random.choice(source_pairs)

            # Generate unique name
            unique_id = str(uuid.uuid4().hex)
            new_name = f"{unique_id}"

            # Determine split (maintain roughly same distribution or just random?
            # User said "copy ... train or val", let's keep the original split of the source file
            # or randomize? "可复制...的train或者val".
            # Let's just stick to the original split of the source file to be safe on distribution,
            # or just assign mostly to train.
            # Let's randomly assign to train/val with 80/20 ratio for variety, or keep original.
            # Keeping original split is safest for class distribution per split.
            target_split = pair['split']

            # Paths
            new_img_path = build_dir / "images" / target_split / (new_name + pair['image'].suffix)
            new_lbl_path = build_dir / "labels" / target_split / (new_name + ".txt")

            # Copy files
            shutil.copy(pair['image'], new_img_path)
            shutil.copy(pair['label'], new_lbl_path)

            # Update size (approximate with file size on disk)
            current_size += new_img_path.stat().st_size
            current_size += new_lbl_path.stat().st_size

            files_created += 1

            if files_created % 1000 == 0:
                print(f"Created {files_created} pairs. Current size: {current_size/1024/1024:.2f} MB")

        print(f"Finished generation. Total pairs: {files_created}. Total raw size: {current_size/1024/1024:.2f} MB")

        # 4. Zip the result
        print("Zipping result...")
        shutil.make_archive(str(output_path.with_suffix('')), 'zip', temp_dir / "build", new_dataset_root_name)
        # shutil.make_archive creates .zip automatically, so we strip suffix if provided

        print(f"Dataset created successfully at: {output_path}")

    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a large YOLO dataset from coco8-detect.")
    parser.add_argument("size", help="Target size of the dataset (e.g., '1GB', '500MB').")
    parser.add_argument("--source", default="../data/coco8-detect.zip", help="Path to source coco8-detect.zip.")
    parser.add_argument("--output", help="Path to output zip file.")

    args = parser.parse_args()

    # Adjust source path if running from different root
    if not Path(args.source).exists():
        # Try finding it relative to script if default
        script_dir = Path(__file__).parent
        potential_source = script_dir.parent / "data" / "coco8-detect.zip"
        if potential_source.exists():
            args.source = str(potential_source)
        else:
             # fallback check relative to workspace root
             potential_source_2 = Path("../data/coco8-detect.zip")
             if potential_source_2.exists():
                 args.source = str(potential_source_2)

    create_big_dataset(args.source, args.size, args.output)

