#!/usr/bin/env python3
"""
Update Datasets Collection Schema - Add 'validating' Status
Maintains backward compatibility while extending the schema with new status value
"""

import os
import sys
from typing import Dict, Any

# Add project root directory to Python path for module resolution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
from pymongo import MongoClient
from pymongo.errors import OperationFailure

# Project internal imports
from app.config import settings
from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def update_dataset_schema() -> bool:
    """
    Update the MongoDB 'datasets' collection schema to include 'validating' status
    
    Extends the status enum with 'validating' while preserving existing validation rules.
    Maintains strict validation level for data integrity.
    
    Returns:
        bool: True if schema update succeeds, False otherwise
    """
    # Initialize MongoDB client
    client = MongoClient(settings.mongodb_url)
    db = client[settings.mongo_db_name]

    try:
        # Define updated schema validator with 'validating' status added
        schema_validator: Dict[str, Any] = {
            '$jsonSchema': {
                'bsonType': 'object',
                'required': ['name', 'dataset_type', 'class_names', 'status', 'created_at', 'updated_at'],
                'properties': {
                    'name': {
                        'bsonType': 'string',
                        'description': 'Dataset name must be a non-empty string (required)'
                    },
                    'dataset_type': {
                        'enum': ['detect', 'obb', 'segment', 'pose', 'classify'],
                        'description': 'Dataset type must be one of the supported detection/classification types'
                    },
                    'class_names': {
                        'bsonType': 'array',
                        'items': {'bsonType': 'string'},
                        'description': 'Class names must be an array of non-empty strings'
                    },
                    'num_images': {
                        'bsonType': 'int',
                        'minimum': 0,
                        'description': 'Number of images must be a non-negative integer'
                    },
                    'num_annotations': {
                        'bsonType': 'int',
                        'minimum': 0,
                        'description': 'Number of annotations must be a non-negative integer'
                    },
                    'status': {
                        'enum': ['processing', 'active', 'validating', 'error', 'deleted'],
                        'description': 'Dataset status - added "validating" for intermediate validation phase'
                    }
                }
            }
        }

        # Execute schema update command
        result = db.command({
            'collMod': 'datasets',
            'validator': schema_validator,
            'validationLevel': 'strict'
        })

        logger.info("✅ Datasets collection schema updated successfully")
        logger.info(f"   Added 'validating' to status enum values")
        return True

    except OperationFailure as e:
        logger.error(f"❌ Schema update failed (MongoDB operation error): {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during schema update", exc_info=True)
        logger.error(f"   Error details: {e}")
        return False
    finally:
        # Ensure MongoDB client connection is closed
        client.close()
        logger.debug("MongoDB client connection closed")


if __name__ == "__main__":
    # Print execution header
    print("=" * 60)
    print("Update Datasets Collection Schema")
    print("=" * 60)
    # Mask credentials in MongoDB URL for security
    db_connection_display = settings.mongodb_url.split('@')[-1] if '@' in settings.mongodb_url else settings.mongodb_url
    print(f"Target Database: {settings.mongo_db_name}")
    print(f"MongoDB Connection: {db_connection_display}")
    print()
    
    # Execute schema update
    update_success = update_dataset_schema()
    
    # Print result summary
    if update_success:
        print("\n✅ Schema update completed successfully!")
        print("\nSupported status values now include:")
        print("  - processing (dataset being processed)")
        print("  - active (dataset ready for use)")
        print("  - validating (new - dataset under validation)")
        print("  - error (dataset processing failed)")
        print("  - deleted (dataset marked as deleted)")
        sys.exit(0)
    else:
        print("\n❌ Schema update failed. Please check application logs for detailed errors.")
        sys.exit(1)