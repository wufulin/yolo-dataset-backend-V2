# YOLO Dataset Management API v2.0.0

A modern, high-performance backend system for managing YOLO datasets, supporting dataset management, upload, and processing for multiple YOLO task types (detection, segmentation, pose estimation, oriented bounding boxes, classification).

## Features

### Core Features
- ✅ **Dataset Management**: Create, query, and list datasets
- ✅ **Chunked Upload**: Support for large file chunked uploads, up to 100GB
- ✅ **Resume Upload**: Upload session management with support for resuming interrupted uploads
- ✅ **Multi-Task Support**: Support for five task types: detect, obb, segment, pose, classify
- ✅ **YOLO Format Validation**: Automatic validation of uploaded datasets against YOLO format specifications
- ✅ **Image Management**: Image metadata management with pagination and split filtering
- ✅ **Storage Management**: Object storage using MinIO with presigned URL support

### Technical Features
- 🚀 **Async High Performance**: Built on FastAPI with async MongoDB driver
- 🔒 **Authentication & Authorization**: JWT-based user authentication system
- 📊 **Session Management**: Redis-based upload session management with session recovery
- 📝 **Comprehensive Logging**: Structured logging with error tracking
- 🎯 **Exception Handling**: Robust exception handling with unified error response format
- 📈 **Performance Monitoring**: Built-in performance monitoring decorators

## Tech Stack

- **Web Framework**: FastAPI 0.104.1
- **Database**: MongoDB (Motor async driver)
- **Object Storage**: MinIO
- **Cache/Session**: Redis
- **Validation Framework**: Pydantic 2.5.0
- **YOLO Validation**: Ultralytics 8.3.228
- **Python Version**: 3.10+

## Project Structure

```
yolo-dataset-backend-v2/
├── app/
│   ├── api/              # API routes
│   │   ├── datasets.py   # Dataset management endpoints
│   │   └── upload.py    # File upload endpoints
│   ├── auth.py          # Authentication module
│   ├── config.py        # Configuration management
│   ├── core/            # Core functionality modules
│   │   ├── config.py    # Configuration system
│   │   ├── decorators.py # Decorators
│   │   ├── exceptions.py # Exception definitions
│   │   └── ...
│   ├── models/          # Data models
│   ├── schemas/         # Pydantic schemas
│   ├── services/        # Business service layer
│   │   ├── dataset_service.py
│   │   ├── upload_service.py
│   │   ├── minio_service.py
│   │   ├── redis_service.py
│   │   └── ...
│   ├── utils/           # Utility functions
│   │   ├── yolo_validator.py
│   │   └── file_utils.py
│   └── main.py          # Application entry point
├── docker/              # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/                # Documentation
│   ├── system-design.md
│   └── yolo_format_specification.md
├── scripts/             # Utility scripts
├── requirements.txt     # Python dependencies
└── env.example          # Environment variables example
```

## Quick Start

### Prerequisites

- Python 3.10+
- MongoDB
- MinIO
- Redis

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd yolo-dataset-backend-v2
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp env.example .env
# Edit .env file and configure database, MinIO, Redis connection information
```

4. **Initialize database**
```bash
python scripts/init_database.py
```

5. **Start the service**
```bash
# Development mode
python -m app.main

# Or using uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

Start all services with Docker Compose:

```bash
cd docker
docker-compose up -d
```

This will start the following services:
- MongoDB (port 27017)
- Redis (port 6379)
- MinIO (port 9000, console 9001)
- FastAPI application (port 8000)

## Configuration

Main configuration items (set in `.env` file):

### Application Configuration
- `APP_NAME`: Application name
- `APP_VERSION`: Application version
- `DEBUG`: Debug mode
- `SECRET_KEY`: JWT secret key
- `HOST`: Service listening address
- `PORT`: Service port

### MongoDB Configuration
- `MONGODB_URL`: MongoDB connection URL
- `MONGO_DB_NAME`: Database name
- `MONGODB_MAX_POOL_SIZE`: Connection pool size

### MinIO Configuration
- `MINIO_ENDPOINT`: MinIO service address
- `MINIO_ACCESS_KEY`: Access key
- `MINIO_SECRET_KEY`: Secret key
- `MINIO_BUCKET_NAME`: Bucket name
- `MINIO_SECURE`: Whether to use HTTPS

### Redis Configuration
- `REDIS_URL`: Redis connection URL (recommended)
- Or use `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` for separate configuration
- `REDIS_SESSION_TTL`: Session expiration time (seconds)

### Upload Configuration
- `MAX_UPLOAD_SIZE`: Maximum upload file size (bytes, default 100GB)
- `UPLOAD_CHUNK_SIZE`: Chunk size (bytes, default 10MB)
- `TEMP_DIR`: Temporary file directory

For detailed configuration, refer to the `env.example` file.

## API Documentation

After starting the service, visit the following URLs to view API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Main API Endpoints

#### Dataset Management
- `POST /api/v1/datasets` - Create a dataset
- `GET /api/v1/datasets` - Get dataset list (paginated)
- `GET /api/v1/datasets/{dataset_id}` - Get dataset details
- `GET /api/v1/datasets/{dataset_id}/images` - Get dataset image list

#### File Upload
- `POST /api/v1/upload/start` - Start upload session
- `POST /api/v1/upload/chunk/{upload_id}/{chunk_index}` - Upload chunk
- `POST /api/v1/upload/complete/{upload_id}` - Complete upload and process
- `GET /api/v1/upload/status/{upload_id}` - Query upload status
- `POST /api/v1/upload/recover/{upload_id}` - Recover upload session
- `DELETE /api/v1/upload/{upload_id}` - Cancel upload

#### Image Management
- `GET /api/v1/images/{image_id}` - Get image details (with annotations)

## Supported Dataset Types

The system supports the following YOLO task types:

1. **detect** - Object detection
2. **obb** - Oriented bounding box detection
3. **segment** - Instance segmentation
4. **pose** - Pose estimation
5. **classify** - Image classification

Each dataset type must conform to YOLO format specifications. See `docs/yolo_format_specification.md` for details.

## Dataset Format Requirements

### Directory Structure
```
dataset_root/
├── data.yaml          # Dataset configuration file
├── images/
│   ├── train/       # Training set images
│   ├── val/          # Validation set images
│   └── test/         # Test set images (optional)
└── labels/
    ├── train/        # Training set labels
    ├── val/          # Validation set labels
    └── test/         # Test set labels (optional)
```

### data.yaml Example
```yaml
path: dataset_root
train: images/train
val: images/val
names:
  0: person
  1: bicycle
  2: car
```

For detailed format specifications, refer to `docs/yolo_format_specification.md`.

## Upload Workflow

1. **Start Upload**: Call `/api/v1/upload/start` to create an upload session
2. **Upload Chunks**: Loop through calling `/api/v1/upload/chunk/{upload_id}/{chunk_index}` to upload all chunks
3. **Complete Upload**: Call `/api/v1/upload/complete/{upload_id}` to complete upload and trigger dataset processing
4. **Query Status**: Use `/api/v1/upload/status/{upload_id}` to query upload progress

Resume upload is supported: if upload is interrupted, you can call `/api/v1/upload/recover/{upload_id}` to recover the session, then continue uploading missing chunks.

## Architecture & Flow Diagrams

The following diagrams illustrate the system architecture, upload flow, and component interactions. These visualizations help understand the complete data flow from file upload to dataset processing.

### System Components Interaction

This diagram shows how different system components interact during a typical request flow:

![System Components Interaction](docs/System%20Components%20Interaction.svg)

**Key Interactions:**
- **Request Routing**: Nginx load balances requests to backend services
- **Session & State Management**: Redis handles upload session state and caching
- **File Storage Operations**: MinIO manages large file storage with multipart uploads
- **Metadata Management**: MongoDB stores dataset and image metadata
- **Background Processing**: Worker processes handle async file processing tasks
- **Response & Real-time Updates**: WebSocket connections provide real-time status updates

### Dataset Upload System Sequence

This sequence diagram details the complete dataset upload process from initiation to completion:

![Dataset Upload System Sequence](docs/Dataset%20Upload%20System%20Sequence%20Diagram.svg)

**Process Flow:**
1. **File Upload Initiation**: User selects ZIP file (≤100GB), frontend validates file type/size
2. **Chunked Upload Process**: File is split into chunks and uploaded in parallel (5 concurrent chunks)
3. **Archive Processing & Validation**: ZIP is extracted, YOLO format is validated
4. **Parallel Storage Processing**: Images are processed in batches and stored in MinIO, metadata in MongoDB
5. **Finalization**: Statistics are generated, resources are cleaned up, final status is updated

### Enhanced Upload Flow

This comprehensive flowchart shows the detailed upload workflow including error handling and recovery:

![Enhanced Upload Flow](docs/Enhanced%20Upload%20Flow.svg)

**Key Stages:**
- **⓪ Frontend Upload Module**: File validation, chunking, parallel upload, progress tracking
- **① Backend Upload Processing**: Session management, chunk validation, integrity checks
- **② Archive Processing & Discovery**: Streaming extraction, directory traversal, file pairing, YOLO validation
- **③ Parallel Processing Pipeline**: Image processing, storage routing, MinIO upload, metadata caching
- **④ Completion & Analytics**: Statistics generation, performance analytics, resource cleanup
- **⑤ Error Handling & Recovery**: Centralized error handling, automatic recovery, graceful degradation

### Key Status Transitions

This diagram illustrates the status transition flow throughout the upload and processing lifecycle:

![Key Status Transition Sequence](docs/Key%20Status%20Transition%20Sequence.svg)

**Status Flow:**
- `FRONTEND_UPLOAD_COMPLETE` → All chunks uploaded successfully
- `BACKEND_UPLOAD_COMPLETE` → File merged and validated
- `YOLO_VALIDATION_FAILED` → Format validation errors (recoverable)
- `VALIDATION_PASSED` → YOLO format validated successfully
- `STORAGE_SUCCESS` → All files stored in MinIO and MongoDB
- `PROCESSING_COMPLETE` → Dataset ready for use

### Real-time Status Updates

This sequence diagram shows how real-time status updates are propagated to the frontend:

![Real-time Status Updates Sequence](docs/Real-time%20Status%20Updates%20Sequence.svg)

**Update Mechanisms:**
- **Redis Pub/Sub**: Backend publishes status updates to Redis channels
- **WebSocket Connections**: Frontend subscribes to status updates via WebSocket
- **Polling Fallback**: HTTP polling available when WebSocket is unavailable
- **Progress Tracking**: Real-time progress percentage and chunk status updates

### Diagram Files Location

All diagram source files are located in the `docs/` directory:
- `System Components Interaction.svg` - System architecture overview
- `Dataset Upload System Sequence Diagram.svg` - Complete upload sequence
- `Enhanced Upload Flow.svg` - Detailed upload workflow with error handling
- `Key Status Transition Sequence.svg` - Status state machine
- `Real-time Status Updates Sequence.svg` - Real-time update mechanism

For more detailed system design documentation, refer to `docs/system-design.md`.

## Development Guide

### Code Structure

- **API Layer** (`app/api/`): Route definitions and request handling
- **Service Layer** (`app/services/`): Business logic implementation
- **Model Layer** (`app/models/`): Data model definitions
- **Schema Layer** (`app/schemas/`): API request/response schemas
- **Utility Layer** (`app/utils/`): Common utility functions

### Adding New Features

1. Implement business logic in `app/services/`
2. Add routes in `app/api/`
3. Define request/response schemas in `app/schemas/`
4. Update API documentation

### Running Tests

```bash
# Run tests (if available)
pytest

# Code linting
flake8 app/
black app/ --check
```

## Logging

Log file locations:
- Application logs: `logs/app.log`
- Error logs: `logs/error.log`

Log level can be configured via `LOG_LEVEL` environment variable (DEBUG/INFO/WARNING/ERROR/CRITICAL).

## Performance Optimization

- Use async MongoDB driver (Motor) for improved database performance
- Redis caching for upload sessions to reduce database pressure
- Chunked upload support for large files to avoid memory overflow
- MinIO object storage for distributed storage support

## Security Recommendations

1. **Production Environment Configuration**:
   - Change `SECRET_KEY` to a strong random string
   - Configure CORS allowed origins (do not use `allow_origins=["*"]`)
   - Enable HTTPS
   - Configure MongoDB and Redis authentication

2. **File Upload Security**:
   - Validate file types and sizes
   - Scan for malicious files
   - Limit upload frequency

3. **Authentication & Authorization**:
   - Use strong password policies
   - Regularly rotate JWT keys
   - Implement role-based access control

## Troubleshooting

### Common Issues

1. **MongoDB Connection Failed**
   - Check `MONGODB_URL` configuration
   - Verify MongoDB service is running
   - Check network connection and firewall settings

2. **Redis Connection Failed**
   - Check `REDIS_URL` or `REDIS_HOST`/`REDIS_PORT` configuration
   - Verify Redis service is running
   - Application will use in-memory storage without Redis (not recommended for production)

3. **MinIO Upload Failed**
   - Check MinIO service status
   - Verify bucket is created
   - Validate access keys and permissions

4. **Upload Session Lost**
   - Check Redis connection and TTL settings
   - Verify session has not expired
   - Check logs for detailed error information

## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

License: MIT

## Contact

wufulinit@gmail.com

## Changelog

### v2.0.0
- Refactored configuration management system
- Enhanced exception handling mechanism
- Optimized upload session management
- Improved logging
- Support for multiple YOLO task types

