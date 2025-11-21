### 1. Database Overview
**Database Name**: `yolo_datasets`  
**Storage Engine**: WiredTiger (MongoDB default)  
**Character Encoding**: UTF-8

### 2. Collection Design
#### 2.1 datasets Collection
Stores dataset-level metadata information
```javascript
{
  _id: ObjectId,
  name: String,                    // Dataset name, unique
  description: String,             // Dataset description
  dataset_type: String,            // Type: detect/obb/segment/pose/classify
  class_names: [String],           // List of class names
  num_images: Number,              // Total number of images
  num_annotations: Number,         // Total number of annotations
  splits: {                        // Statistics of each split quantity
    train: Number,
    val: Number,
    test: Number
  },
  status: String,                  // Status: processing/active/error
  error_message: String,           // Error message
  file_size: Number,               // Original file size (bytes)
  storage_path: String,            // Storage path
  created_by: String,              // Creator
  created_at: ISODate,
  updated_at: ISODate,
  version: Number                  // Dataset version
}
```

#### 2.2 images Collection
Stores image metadata and annotation information.
```javascript
{
  _id: ObjectId,
  dataset_id: ObjectId,            // Foreign key, references datasets._id
  filename: String,                // Original file name
  file_path: String,               // Storage path in MinIO
  file_size: Number,               // File size (bytes)
  file_hash: String,               // File MD5 hash
  width: Number,                   // Image width
  height: Number,                  // Image height
  channels: Number,                // Number of channels
  format: String,                  // Image format: jpg/png, etc.
  split: String,                   // Split: train/val/test
  annotations: [                   // Annotation array
    {
      annotation_type: String,     // bbox/obb/segment/pose/classify
      class_id: Number,
      class_name: String,
      confidence: Number,          // Confidence (optional)
      image_id: ObjectId,
      dataset_id: ObjectId,
      
      // For bbox
      bbox: {
        x_center: Number,          // Normalized center x-coordinate
        y_center: Number,          // Normalized center y-coordinate  
        width: Number,             // Normalized width
        height: Number             // Normalized height
      },
      
      // For obb
      obb: {
        points: [Number]           // 8 coordinate values [x1,y1,x2,y2,x3,y3,x4,y4]
      },
      
      // For segment
      segment: {
        points: [Number]           // Polygon point coordinates
      },
      
      // For pose
      pose: {
        keypoints: [Number],       // Key points [x1,y1,v1,x2,y2,v2,...]
        skeleton: [Number]         // Skeleton connection relationship (optional)
      },
      
      // For classify
      classification: {
        class_id: Number,
        class_name: String
      },
      
      created_at: ISODate,
      updated_at: ISODate
    }
  ],
  metadata: {                      // Extended metadata
    source: String,                 // source (optional)
    original_path: String         // original_path (optional)
  },
  is_annotated: Boolean,           // Whether annotated
  annotation_count: Number,        // Number of annotations
  created_at: ISODate,
  updated_at: ISODate
}
```

#### 2.3 upload_sessions Collection
Manages file upload sessions.
```javascript
{
  _id: ObjectId,
  upload_id: String,               // Upload session ID
  user_id: String,                 // User ID
  filename: String,                // Original file name
  file_size: Number,               // Total file size
  total_chunks: Number,            // Total number of chunks
  chunk_size: Number,              // Chunk size
  received_chunks: [Number],       // Received chunk indices
  temp_path: String,               // Temporary file path
  status: String,                  // Status: uploading/processing/completed/error
  dataset_id: ObjectId,            // Associated dataset ID (after completion)
  error_message: String,           // Error message
  created_at: ISODate,
  updated_at: ISODate,
  expires_at: ISODate              // Session expiration time
}
```

#### 2.4 dataset_statistics Collection
Stores dataset statistical information for quick query.
```javascript
{
  _id: ObjectId,
  dataset_id: ObjectId,            // Foreign key, references datasets._id
  date: ISODate,                   // Statistical date
  total_images: Number,            // Total number of images
  total_annotations: Number,       // Total number of annotations
  images_by_split: {               // Statistics by split
    train: Number,
    val: Number, 
    test: Number
  },
  annotations_by_class: {          // Annotation count by class
    "class_name": Number
  },
  avg_annotations_per_image: Number, // Average number of annotations per image
  class_distribution: {            // Class distribution
    "class_name": Number           // Number of images per class
  },
  created_at: ISODate
}
```

#### 2.5 users Collection
User management (simplified version).
```javascript
{
  _id: ObjectId,
  username: String,                // Username, unique
  email: String,                   // Email
  hashed_password: String,         // Hashed password
  role: String,                    // Role: admin/user
  is_active: Boolean,              // Whether active
  created_at: ISODate,
  last_login: ISODate,
  preferences: {                   // User preference settings
    default_page_size: Number,
    theme: String
  }
}
```

### 3. Index Design
#### 3.1 datasets Collection Indexes
```javascript
// Unique index
db.datasets.createIndex({ "name": 1 }, { unique: true });

// Compound indexes
db.datasets.createIndex({ "dataset_type": 1, "created_at": -1 });
db.datasets.createIndex({ "status": 1, "created_at": -1 });
db.datasets.createIndex({ "created_by": 1, "created_at": -1 });

// TTL index - Automatically deletes records with status "error" older than 7 days
db.datasets.createIndex({ "status": 1, "updated_at": 1 }, { 
  partialFilterExpression: { status: "error" },
  expireAfterSeconds: 604800 // 7 days
});
```

#### 3.2 images Collection Indexes
```javascript
// Compound indexes - Main query patterns
db.images.createIndex({ "dataset_id": 1, "split": 1, "created_at": -1 });
db.images.createIndex({ "dataset_id": 1, "is_annotated": 1 });
db.images.createIndex({ "dataset_id": 1, "filename": 1 });

// Single-field indexes
db.images.createIndex({ "file_hash": 1 }); // For deduplication
db.images.createIndex({ "split": 1 });
db.images.createIndex({ "created_at": -1 });

// Geospatial index (if using GPS data)
db.images.createIndex({ "metadata.gps": "2dsphere" });

// Text index - Supports filename search
db.images.createIndex({ "filename": "text" });
```

#### 3.3 upload_sessions Collection Indexes
```javascript
// Unique index
db.upload_sessions.createIndex({ "upload_id": 1 }, { unique: true });

// TTL index - Automatically cleans up expired upload sessions (24 hours)
db.upload_sessions.createIndex({ "expires_at": 1 }, { expireAfterSeconds: 0 });

// Status indexes
db.upload_sessions.createIndex({ "status": 1, "created_at": -1 });
db.upload_sessions.createIndex({ "user_id": 1, "created_at": -1 });
```

#### 3.4 dataset_statistics Collection Indexes
```javascript
// Compound indexes
db.dataset_statistics.createIndex({ "dataset_id": 1, "date": -1 });
db.dataset_statistics.createIndex({ "date": -1 });
```

#### 3.5 users Collection Indexes
```javascript
// Unique indexes
db.users.createIndex({ "username": 1 }, { unique: true });
db.users.createIndex({ "email": 1 }, { unique: true, sparse: true });
```

### 4. Data Relationship Design
#### 4.1 Reference Relationships
```text
datasets (1) ←→ (N) images
datasets (1) ←→ (N) dataset_statistics  
datasets (1) ←→ (N) upload_sessions
users (1) ←→ (N) datasets
users (1) ←→ (N) upload_sessions
```

#### 4.2 Data Consistency Strategy
- **Referential Integrity**: Maintained at the application layer, using transactions to ensure data consistency
  
- **Cascading Delete**: Automatically deletes related images and statistical information when deleting a dataset
  
- **Data Validation**: Uses MongoDB schema validation
