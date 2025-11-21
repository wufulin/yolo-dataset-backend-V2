"""
YOLO Dataset Initialization Script
Reads YOLO format datasets and inserts them into MongoDB

Usage:
    python scripts/init_dataset.py
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.upload_service import upload_service
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class YOLODatasetImporter:
    """YOLO Dataset Importer"""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset_info = {
            "name": Path(dataset_path).name,
            "description": "",
            "dataset_type": "detect",
            "class_names": []
        }


    def import_dataset(self):
        """Execute complete dataset import workflow"""
        logger.info("=" * 60)
        logger.info("YOLO Dataset Import Tool")
        logger.info("=" * 60)

        try:
            asyncio.run(upload_service.process_dataset(self.dataset_path, self.dataset_info))

            logger.info("\n" + "=" * 60)
            logger.info("✓ Dataset import successful!")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"\n✗ Import failed: {e}", exc_info=True)
            raise e


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Initialize YOLO dataset")
    parser.add_argument("--dataset_path", type=str, default="./data/coco8-detect-1024MB.zip", help="Path to the dataset")

    args = parser.parse_args()

    dataset_path = args.dataset_path

    # Check if path exists
    if not os.path.exists(dataset_path):
        logger.error(f"✗ Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    # Create importer and execute import
    importer = YOLODatasetImporter(dataset_path)
    importer.import_dataset()


if __name__ == "__main__":
    main()
