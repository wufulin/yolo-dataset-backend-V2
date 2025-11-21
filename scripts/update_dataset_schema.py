#!/usr/bin/env python3
"""更新 datasets collection 的 schema，添加 'validating' 状态"""
import os
import sys

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymongo import MongoClient
from pymongo.errors import OperationFailure

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def update_dataset_schema():
    """更新 datasets collection 的 schema，添加 'validating' 状态"""
    client = MongoClient(settings.mongodb_url)
    db = client[settings.mongo_db_name]

    try:
        # 更新 schema，添加 'validating' 状态
        validation = {
            '$jsonSchema': {
                'bsonType': 'object',
                'required': ['name', 'dataset_type', 'class_names', 'status', 'created_at', 'updated_at'],
                'properties': {
                    'name': {
                        'bsonType': 'string',
                        'description': 'Dataset name must be a string and is required'
                    },
                    'dataset_type': {
                        'enum': ['detect', 'obb', 'segment', 'pose', 'classify'],
                        'description': 'Dataset type must be one of the supported types'
                    },
                    'class_names': {
                        'bsonType': 'array',
                        'items': {
                            'bsonType': 'string'
                        },
                        'description': 'Class names must be an array of strings'
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
                        'description': 'Status must be one of the allowed values'
                    }
                }
            }
        }

        # 更新 schema
        result = db.command({
            'collMod': 'datasets',
            'validator': validation,
            'validationLevel': 'strict'
        })

        logger.info("✅ Datasets collection schema updated successfully")
        logger.info(f"   Added 'validating' to status enum")
        return True

    except OperationFailure as e:
        logger.error(f"❌ Failed to update schema: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        return False
    finally:
        client.close()


if __name__ == "__main__":
    print("=" * 60)
    print("更新 Datasets Collection Schema")
    print("=" * 60)
    print(f"数据库: {settings.mongo_db_name}")
    print(f"连接: {settings.mongodb_url.split('@')[-1] if '@' in settings.mongodb_url else settings.mongodb_url}")
    print()
    
    success = update_dataset_schema()
    
    if success:
        print("\n✅ Schema 更新成功！")
        print("现在 status 字段支持以下值：")
        print("  - processing")
        print("  - active")
        print("  - validating (新增)")
        print("  - error")
        print("  - deleted")
        sys.exit(0)
    else:
        print("\n❌ Schema 更新失败，请检查错误信息")
        sys.exit(1)

