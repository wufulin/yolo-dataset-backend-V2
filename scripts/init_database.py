"""Database initialization script."""
import logging
import os
import sys

from pymongo import ASCENDING, DESCENDING, TEXT, MongoClient
from pymongo.errors import CollectionInvalid, OperationFailure

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


def init_database():
    """Initialize database with collections and indexes."""
    client = MongoClient(settings.mongodb_url)
    db = client[settings.mongo_db_name]

    # Create collections with validation
    init_datasets_collection(db)
    init_images_collection(db)
    # init_upload_sessions_collection(db)   
    init_dataset_statistics_collection(db)
    init_annotations_collection(db)
    init_annotation_stats_collection(db)
    init_users_collection(db)

    # Create initial admin user
    create_initial_admin(db)

    logger.info("Database initialization completed successfully!")
    client.close()


def init_datasets_collection(db):
    """Initialize datasets collection with schema validation."""
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

    try:
        db.create_collection('datasets')
    except CollectionInvalid:
        pass  # Collection already exists

    # Apply schema validation
    db.command({
        'collMod': 'datasets',
        'validator': validation,
        'validationLevel': 'strict'
    })

    # Create indexes
    db.datasets.create_index([('name', ASCENDING)], unique=True)
    db.datasets.create_index([('dataset_type', ASCENDING), ('created_at', DESCENDING)])
    db.datasets.create_index([('status', ASCENDING), ('created_at', DESCENDING)])
    db.datasets.create_index([('created_at', DESCENDING)])


def init_images_collection(db):
    """Initialize images collection with schema validation."""
    validation = {
        '$jsonSchema': {
            'bsonType': 'object',
            'required': ['dataset_id', 'filename', 'file_path', 'split', 'created_at'],
            'properties': {
                'dataset_id': {
                    'bsonType': 'objectId',
                    'description': 'Dataset reference must be an ObjectId'
                },
                'filename': {
                    'bsonType': 'string',
                    'description': 'Filename must be a string'
                },
                'file_path': {
                    'bsonType': 'string',
                    'description': 'File path must be a string'
                },
                'split': {
                    'enum': ['train', 'val', 'test'],
                    'description': 'Split must be one of train/val/test'
                },
                'width': {
                    'bsonType': 'int',
                    'minimum': 1,
                    'description': 'Image width must be a positive integer'
                },
                'height': {
                    'bsonType': 'int',
                    'minimum': 1,
                    'description': 'Image height must be a positive integer'
                },
                'is_annotated': {
                    'bsonType': 'bool',
                    'description': 'is_annotated must be a boolean'
                }
            }
        }
    }

    try:
        db.create_collection('images')
    except CollectionInvalid:
        pass

    db.command({
        'collMod': 'images',
        'validator': validation,
        'validationLevel': 'strict'
    })

    # Create indexes
    db.images.create_index([('dataset_id', ASCENDING), ('split', ASCENDING), ('created_at', DESCENDING)])
    db.images.create_index([('dataset_id', ASCENDING), ('is_annotated', ASCENDING)])
    db.images.create_index([('dataset_id', ASCENDING), ('filename', ASCENDING)])
    db.images.create_index([('file_hash', ASCENDING)])
    db.images.create_index([('split', ASCENDING)])
    db.images.create_index([('created_at', DESCENDING)])
    db.images.create_index([('filename', TEXT)])


def init_upload_sessions_collection(db):
    """Initialize upload_sessions collection."""
    try:
        db.create_collection('upload_sessions')
    except CollectionInvalid:
        pass

    db.upload_sessions.create_index([('upload_id', ASCENDING)], unique=True)
    db.upload_sessions.create_index([('expires_at', ASCENDING)], expireAfterSeconds=0)
    db.upload_sessions.create_index([('status', ASCENDING), ('created_at', DESCENDING)])
    db.upload_sessions.create_index([('created_at', DESCENDING)])


def init_dataset_statistics_collection(db):
    """Initialize dataset_statistics collection."""
    try:
        db.create_collection('dataset_statistics')
    except CollectionInvalid:
        pass

    db.dataset_statistics.create_index([('dataset_id', ASCENDING), ('date', DESCENDING)])
    db.dataset_statistics.create_index([('date', DESCENDING)])


def init_users_collection(db):
    """Initialize users collection."""
    validation = {
        '$jsonSchema': {
            'bsonType': 'object',
            'required': ['username', 'hashed_password', 'role', 'is_active', 'created_at'],
            'properties': {
                'username': {
                    'bsonType': 'string',
                    'description': 'Username must be a string'
                },
                'email': {
                    'bsonType': 'string',
                    'pattern': '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$',
                    'description': 'Email must be a valid email address'
                },
                'role': {
                    'enum': ['admin', 'user'],
                    'description': 'Role must be admin or user'
                },
                'is_active': {
                    'bsonType': 'bool',
                    'description': 'is_active must be a boolean'
                }
            }
        }
    }

    try:
        db.create_collection('users')
    except CollectionInvalid:
        pass

    db.command({
        'collMod': 'users',
        'validator': validation,
        'validationLevel': 'strict'
    })

    db.users.create_index([('username', ASCENDING)], unique=True)
    db.users.create_index([('email', ASCENDING)], unique=True, sparse=True)


def create_initial_admin(db):
    """Create initial admin user."""
    import hashlib
    from datetime import datetime

    # Simple password hashing using SHA-256 (sufficient for development)
    # For production, use a proper password hashing library with salt
    password = 'admin'
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    admin_user = {
        'username': 'admin',
        'email': 'admin@yolo-datasets.com',
        'hashed_password': hashed_password,
        'role': 'admin',
        'is_active': True,
        'created_at': datetime.utcnow(),
        'last_login': None,
        'preferences': {
            'default_page_size': 20,
            'theme': 'light'
        }
    }

    # Insert if not exists
    if not db.users.find_one({'username': 'admin'}):
        db.users.insert_one(admin_user)
        logger.info("✅ Created initial admin user: admin/admin")
        logger.info("   Password hash: SHA-256 (development only)")
    else:
        logger.info("ℹ️  Admin user already exists")


def init_annotations_collection(db):
    """Initialize annotations collection with schema validation."""
    validation = {
        '$jsonSchema': {
            'bsonType': 'object',
            'required': ['image_id', 'dataset_id', 'class_id', 'class_name', 'annotation_type', 'created_at'],
            'properties': {
                'image_id': {
                    'bsonType': 'objectId',
                    'description': 'Image reference must be an ObjectId'
                },
                'dataset_id': {
                    'bsonType': 'objectId',
                    'description': 'Dataset reference must be an ObjectId'
                },
                'class_id': {
                    'bsonType': 'int',
                    'minimum': 0,
                    'description': 'Class ID must be a non-negative integer'
                },
                'class_name': {
                    'bsonType': 'string',
                    'description': 'Class name must be a string'
                },
                'annotation_type': {
                    'enum': ['detect', 'obb', 'segment', 'pose', 'classify'],
                    'description': 'Annotation type must be one of the supported types'
                },
                'confidence': {
                    'bsonType': 'double',
                    'minimum': 0,
                    'maximum': 1,
                    'description': 'Confidence must be between 0 and 1'
                }
            }
        }
    }

    try:
        db.create_collection('annotations')
    except CollectionInvalid:
        pass

    db.command({
        'collMod': 'annotations',
        'validator': validation,
        'validationLevel': 'strict'
    })

    # Create indexes for annotations
    db.annotations.create_index([('image_id', ASCENDING)])
    db.annotations.create_index([('dataset_id', ASCENDING)])
    db.annotations.create_index([('class_name', ASCENDING)])
    db.annotations.create_index([('annotation_type', ASCENDING)])
    db.annotations.create_index([('confidence', DESCENDING)])
    db.annotations.create_index([('created_at', DESCENDING)])
    db.annotations.create_index([('image_id', ASCENDING), ('annotation_type', ASCENDING)])


def init_annotation_stats_collection(db):
    """Initialize annotation_stats collection."""
    try:
        db.create_collection('annotation_stats')
    except CollectionInvalid:
        pass

    db.annotation_stats.create_index([('dataset_id', ASCENDING), ('date', DESCENDING)])
    db.annotation_stats.create_index([('date', DESCENDING)])


if __name__ == '__main__':
    init_database()
