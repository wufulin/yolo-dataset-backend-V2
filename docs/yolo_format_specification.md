# Ultralytics YOLO Dataset Format Specifications and Best Practices Blueprint  


## 1. Introduction and Scope of Study  

In the R&D workflow of object detection models, the standardization level of data preparation often determines the efficiency and stability of training, validation, and deployment. The Ultralytics YOLO series adopts a clear set of YOLO-format dataset specifications, enabling different tasks (detection, segmentation, pose estimation, oriented bounding boxes) to be correctly loaded, augmented, and evaluated by the framework under a unified convention. Based on official documentation, this report systematically organizes the directory structure, configuration file (data.yaml) structure, label (.txt) specifications, and naming practices of the Ultralytics YOLO dataset format. It further expands to cover task-specific differences, data validation and caching mechanisms, format conversion tools, common errors, and automated checklists. The goal is to provide computer vision engineers, data annotation and platform operation personnel, MLOps/DevOps engineers, and technical leaders with a set of directly implementable specifications and best practices.  

This report covers the following task types:  
- Object Detection (detect)  
- Instance Segmentation (segment)  
- Pose Estimation (pose)  
- Oriented Bounding Box (OBB)  

The version baseline is based on the current official documentation; given the frequent iterations of Ultralytics, specific details shall be subject to the latest releases, especially the nuances related to task differences and field definitions [^2][^3][^9].  

The deliverables of this report include:  
- Reusable directory templates and data.yaml examples  
- Label file fields, coordinate normalization, and class mapping rules  
- Key explanations of data validation and caching mechanisms  
- A checklist of common errors and recommendations for automated inspection scripts  
- Naming conventions and version management suggestions  
- Format conversion tools and practical guidelines  

Notes on information gaps:  
- The complete syntax details of label files for segmentation, pose estimation, and OBB are not fully presented in the currently available information. This report only provides general principles and boundary descriptions; refer to the official task-specific documentation for complete paradigms.  
- Regarding the details of "path relativity" and "relative root path", although examples and general conventions exist, it is recommended to verify against the latest official explanations and source code.  
- Details such as whether the `nc` (number of classes) field is mandatory in data.yaml and the consistency boundaries between `names` and indices need to be verified with source code or more comprehensive examples.  
- Information on the NDJSON alternative organization method emerging in the community is currently incomplete; if adopted, small-scale pilot verification is recommended first.  
- Automated scripts for LabelStudio export and splitting strategies need to be customized based on team toolchains and scenarios [^2][^3][^6][^7][^9][^10].  


## 2. Overview of Standard Directory Structure  

The core organizational concept of the Ultralytics YOLO format is to manage images and labels separately: **data splits (train/val/test)** serve as the top-level dimension, and **the parallel placement of images and labels** serves as the secondary dimension. This ensures the loader can map each image to its annotation file (.txt) one-to-one via name matching. Under the dataset root directory, there are typically two folders: `images` and `labels`; corresponding images and label files are stored in subdirectories for different splits (training, validation, testing). If an image contains no objects, the corresponding .txt label file may be omitted to avoid unnecessary I/O caused by empty files [^2].  

For directory naming, it is recommended to use lowercase English letters and underscores, avoiding spaces and special characters. This ensures consistency across cross-platform path parsing and version control tools (e.g., Git). A common data splitting strategy is the explicit separation of training, validation, and test sets; in scenarios with limited resources or small-scale data, the `test` split may be omitted, but the `val` split is recommended to remain for stable evaluation and early stopping during training [^2][^6].  

In the Ultralytics YOLO loader, label files and image files use a **name-matching mechanism**: the base name (excluding the extension) of the label file must exactly match the image file name, with the extension `.txt`, and be placed in the corresponding `labels` subdirectory for the same split as the image. This consistency is guaranteed by the framework’s `YOLODataset` class; meanwhile, the framework provides multi-threaded label validation and caching mechanisms to improve loading efficiency and robustness [^3].  

To intuitively illustrate the standard organization, a template is provided below.  

Table 1: Standard Directory Structure Template and Path Segment Description  

| Level | Directory/File | Description |  
|---|---|---|  
| Root | dataset_root/ | Dataset root directory, storing data.yaml, images, and labels |  
| Root | data.yaml | Dataset configuration file, defining path, train, val, test, names, etc. |  
| Image Dir | images/train/ | Training set images (supports common image formats) |  
| Image Dir | images/val/ | Validation set images |  
| Image Dir | images/test/ | Test set images (optional) |  
| Label Dir | labels/train/ | Training set .txt labels, name-matched with images in images/train |  
| Label Dir | labels/val/ | Validation set .txt labels, name-matched with images in images/val |  
| Label Dir | labels/test/ | Test set .txt labels, name-matched with images in images/test |  


### 2.1 Directory Layout Specifications and Template  

The following template is recommended to ensure the framework loader can discover data as required:  

```
dataset_root/
  ├─ data.yaml
  ├─ images/
  │  ├─ train/
  │  │  ├─ <imageA>.<ext>
  │  │  ├─ <imageB>.<ext>
  │  │  └─ …
  │  ├─ val/
  │  │  └─ …
  │  └─ test/
  │     └─ …
  └─ labels/
     ├─ train/
     │  ├─ <imageA>.txt
     │  ├─ <imageB>.txt
     │  └─ …
     ├─ val/
     │  └─ …
     └─ test/
        └─ …
```  

Key requirements:  
- Splits are named using lowercase underscore style: `train`/`val`/`test`.  
- Label files and image files share the same base name (excluding extensions), with the label file extension `.txt`. They are placed in the corresponding `labels` and `images` subdirectories for the same split.  
- Do not generate .txt label files for images with no objects, to avoid additional overhead from empty files [^2].  


## 3. data.yaml Configuration File Format and Mandatory Fields  

YOLO dataset configuration uses a YAML file, customarily named `data.yaml`. This file carries core information for the framework to locate data, identify classes, and organize loading paths. The Ultralytics official specifications regard `path`, `train`, `val`, and `names` as core fields, with `test` being optional; some examples include a `download` configuration for automatic download of sample data. Paths are usually expressed as **relative paths** (relative to the dataset root directory, typically the directory where data.yaml resides) and can be referenced as a whole via the `data` parameter in training commands [^1][^2].  

To clarify the role of each field, a field comparison table is provided below.  

Table 2: data.yaml Fields and Their Meanings  

| Field | Mandatory | Meaning | Example |  
|---|---|---|---|  
| path | Core | Dataset root directory (usually the same as the directory containing data.yaml) | dataset_root |  
| train | Core | Training set image directory (relative to `path`) or a list of image paths | images/train |  
| val | Core | Validation set image directory (relative to `path`) or a list of image paths | images/val |  
| test | Optional | Test set image directory (relative to `path`) or a list of image paths | images/test |  
| names | Core | Class name dictionary, with keys as class indices starting from 0 and values as class names | {0: "person", 1: "bicycle", …} |  
| download | Optional | Automatic download script or URL (for sample datasets) | {url: "…"} |  

In the official COCO8 example, the configuration includes `path`, `train`, `val`, a complete list of 80 class names, and a `download` field pointing to the standard download package. This confirms the relative path convention for `path`, `train`, and `val`, as well as the core role of `names` as the class mapping [^1].  

Examples of references in training and validation:  
- CLI: `yolo detect train data=dataset.yaml model=yolo11n.pt imgsz=640 epochs=100`  
- Python: `model.train(data='dataset.yaml', imgsz=640, epochs=100)`  

The `data` parameter here points to the `data.yaml` file. Ultralytics also demonstrates an alternative organization method using NDJSON in its documentation and examples; hints and notes on this are provided in the "Appendix" and "Information Gaps" sections of this report [^2].  


### 3.1 Path Configuration Rules and Examples  

Official examples generally use relative paths. Taking COCO8 as an example: `path=coco8`, `train=images/train`, `val=images/val`; this means when the training command is executed in the directory containing `dataset.yaml`, the framework will concatenate `path` with `train`/`val` to form the complete path, thereby locating images and labels. This relative path convention ensures good portability: when the entire dataset root directory is moved to a different machine or workspace, there is no need to modify internal paths—you only need to ensure the training command is executed in the directory containing `data.yaml` [^1][^2].  

Additionally, official specifications state that `train`/`val`/`test` can point to:  
1. A directory;  
2. A .txt file containing a list of image paths;  
3. A list of multiple paths.  

This flexibility allows managing sample collections via files or lists in large-scale data scenarios, avoiding I/O pressure caused by excessively deep directory hierarchies or a large number of files concentrated in a single directory [^2].  

Table 3: Three Expression Methods for train/val/test and Their Applicable Scenarios  

| Expression Method | Example | Applicable Scenario | Advantages | Considerations |  
|---|---|---|---|---|  
| Directory | images/train | Small-to-medium datasets with clear directory hierarchies | Intuitive structure, easy manual inspection | I/O pressure for a large number of files in a single directory |  
| File (.txt list) | train.txt (one image path per line) | Large-scale or distributed storage scenarios | Unified list management, easy splitting and versioning | Requires maintenance of list file generation and synchronization |  
| Path list | [p1, p2, p3] | Aggregating multi-source data | Flexible combination | Higher requirements for list updates and consistency management |  

In multi-task scenarios, `data.yaml` assumes the general responsibility of "path and class mapping". The `YOLODataset` will select to load the corresponding fields and formats for detection, segmentation, pose, or OBB labels based on task parameters (e.g., `use_segments`, `use_keypoints`, `use_obb`) [^3].  


## 4. Label File Format (.txt): Coordinates and Class Indices  

YOLO label files use **normalized coordinates in xywh format**: each line represents one object instance, containing the class index, bounding box center coordinates, width, and height—all values are normalized to the range [0, 1]. Class indices use zero-based numbering, corresponding one-to-one with the keys in the `names` dictionary of `data.yaml`. For images with no objects, the corresponding .txt file is omitted to reduce empty files and avoid unnecessary processing by the loader [^2][^3].  

Table 4: Detailed Explanation of YOLO Label Line Fields and Normalization Rules  

| Field | Value Type | Meaning | Normalization Calculation Method |  
|---|---|---|---|  
| class_id | Integer (zero-based) | Object class index, corresponding to the keys in data.yaml’s `names` | Fixed integer, no normalization required |  
| x_center | Float (0–1) | x-coordinate of the bounding box center | x_center / image width |  
| y_center | Float (0–1) | y-coordinate of the bounding box center | y_center / image height |  
| width | Float (0–1) | Bounding box width | width / image width |  
| height | Float (0–1) | Bounding box height | height / image height |  

Example label lines (detection):  
- 0 0.517 0.618 0.204 0.312  
- 1 0.123 0.234 0.056 0.078  

The above examples indicate:  
- An object of class 0 has its center at 51.7% of the image width and 61.8% of the image height, with a bounding box width and height accounting for 20.4% and 31.2% of the image width and height, respectively.  

In multi-object scenarios, each line independently describes one bounding box, arranged in sequence. The label file shares the same base name as the image file but uses the `.txt` extension, and is placed in the corresponding `labels` subdirectory for the same split as the image [^2].  


### 4.1 Coordinate Normalization and Conversion Example  

The basic principle for converting pixel coordinates to normalized coordinates is to divide by the image dimensions:  
1. Given pixel coordinates of the bounding box: top-left (xmin, ymin) and bottom-right (xmax, ymax);  
2. Calculate bounding box width: `w = xmax − xmin`, bounding box height: `h = ymax − ymin`;  
3. Calculate center coordinates: `x_center = xmin + w/2`, `y_center = ymin + h/2`;  
4. Normalize: `x_center_norm = x_center / image_width`, `y_center_norm = y_center / image_height`, `w_norm = w / image_width`, `h_norm = h / image_height`.  

Verification requirements:  
- All values ∈ [0, 1];  
- Bounding box width and height do not exceed the image boundaries (even after normalization, values should not exceed 1);  
- Class indices are within the valid range (consistent with the `names` dictionary) [^2].  

During loading, the framework performs consistency checks between labels and images, including multi-threaded validation of label format and coordinate validity. It also writes label caches to `labels.cache` to improve loading speed and reproducibility in subsequent training [^3].  


## 5. Task Differences: Detection/Segmentation/Pose/OBB  

The Ultralytics `YOLODataset` unified loader supports label formats for multiple tasks; the loader selects which fields to read based on task parameters:  
- Detection (detect): Reads `class_id` and normalized xywh in the .txt line format described above.  
- Segmentation (segment): Each line appends a sequence of segmentation point coordinates (the official documentation available for this report does not provide complete syntax details); lines typically start with `class_id`, followed by a series of normalized point coordinates.  
- Pose Estimation (pose): Each line appends a sequence of keypoint coordinates (and visibility flags, etc.); refer to official task documentation for complete details.  
- Oriented Bounding Box (OBB): Each line appends a rotated box coordinate representation (possibly four corner points or other formats); refer to official task documentation for specific syntax [^3][^9].  

Table 5: Task Types and Main Label File Fields  

| Task Type | Main Fields | Additional Fields | Description |  
|---|---|---|---|  
| Detection (detect) | class_id, x_center, y_center, width, height | None | Normalized xywh, zero-based class indices |  
| Segmentation (segment) | class_id | Segmentation point coordinate sequence | Refer to official documentation for syntax details |  
| Pose Estimation (pose) | class_id | Keypoint coordinates and attributes | Refer to official documentation for syntax details |  
| OBB (Oriented Bounding Box) | class_id | Rotated box coordinate representation | Refer to official documentation for syntax details |  

Principle of data preparation consistency:  
- Ensure the `names` dictionary correctly maps class indices;  
- If additional fields exist for a task, coordinates still need to be normalized;  
- Maintain name matching between label and image files.  

In multi-task data organization, it is recommended to use separate label directories or explicit task configurations for different tasks. Avoid mixing field representations of different tasks in the same label file, as this may cause ambiguity in loader parsing and abnormalities in the training pipeline [^3][^9].  


## 6. Data Validation and Automated Checks (Caching and Quality Control)  

The Ultralytics dataset loader provides data validation and caching mechanisms to ensure data quality before training and improve loading performance:  
- Label Caching (`cache_labels`): By default, parses labels and caches image check results to `labels.cache`, including version and hash verification to ensure consistency and reproducibility.  
- Image Validation: Checks image integrity, readability, and dimensions; counts missing, found, empty, and corrupted files to avoid unexpected read errors during training.  
- Multi-threaded Validation: Executes label and image validation in parallel during dataset initialization, accelerating preparation and early detection of format issues [^3].  

Practical Recommendations:  
- Perform a full validation before the first training or after data updates to ensure consistency between labels and images.  
- Caching can significantly reduce I/O overhead in subsequent runs; however, after data modifications, clear or update the cache to avoid "stale cache" affecting training results.  
- Pay attention to statistical information output by the framework, and promptly locate and fix missing or corrupted samples [^3].  


## 7. Naming Conventions and Organizational Best Practices  

To enhance maintainability for cross-team collaboration and version management, the following conventions are recommended:  
- **File and Directory Naming**: Use a combination of lowercase English letters and underscores; avoid spaces and special characters. Uniformly use `.txt` as the extension for label files.  
- **Class Naming**: Use concise, readable names that are consistent within the team; avoid arbitrary changes that may cause index misalignment. Maintain a unique "name-index" mapping in the `names` dictionary of `data.yaml`.  
- **Data Splitting and Naming**: Explicitly use `train`/`val`/`test` splits. In small-scale or exploratory scenarios, the `test` split may be omitted, but the `val` split is recommended to remain for stable evaluation during training.  
- **Versioning and Reproducibility**: Fix the expressions of `path`, `train`, `val`, and the class set in `data.yaml`. Collaborate with version control systems and data manifests (e.g., `.txt` path lists) to support reproducible experiments and data migration.  
- **CI/CD Integration**: Incorporate data validation and cache updates into pipeline checks to ensure automatic verification of format and integrity after each data update [^2][^3][^6].  


## 8. Format Conversion and Ecosystem Tools  

Ultralytics provides official conversion tools to migrate from common formats (such as COCO) to YOLO format:  
- **COCO → YOLO**: Use the JSON2YOLO tool or Ultralytics data conversion interface to uniformly map polygon/keypoint/box annotations to YOLO conventions. After conversion, it is still necessary to check whether the normalization range and class mapping are correct, and confirm the directory organization for name-matching between labels and images [^2][^10].  

In the annotation tool ecosystem, LabelStudio is a commonly used choice. It can export label and image organizations in YOLO format; combined with automatic splitting strategies (e.g., 7:2:1 split for `train`/`val`/`test`), it can quickly form a trainable dataset framework. Practical recommendations include:  
- Unify the directory organization and naming conventions for `images` and `labels` using project templates.  
- Immediately execute an automated verification script after export to check the range of `class_id`, coordinate normalization, filename consistency, and missing labels.  
- Incorporate export and verification steps into automated pipelines to reduce the risk of human error and inconsistency [^7].  


## 9. Common Errors and Prevention Methods  

High-frequency issues in the data preparation phase typically stem from "directory inconsistency, unnormalized coordinates, incorrect class mapping, mismatched image-label pairs, and stale caches". To reduce rework costs, it is recommended to establish a standardized checklist and perform automatic verification after each data update.  

Table 6: Error - Symptom - Root Cause - Remediation  

| Error | Symptom | Root Cause | Remediation |  
|---|---|---|---|  
| Mismatch between `images` and `labels` | Training logs indicate a large number of missing labels or abnormal metrics | Inconsistent filenames between labels and images, or incorrect directory placement | Standardize base filenames and ensure `labels` and `images` are placed in corresponding directories for the same split |  
| Unnormalized or out-of-bounds coordinates | Low mAP, unstable training, or out-of-bounds output boxes | Use of pixel coordinates or calculation errors | Perform divisive normalization based on image dimensions and verify values are within the [0, 1] range |  
| Out-of-bounds class indices or inconsistent class names | Incorrect class mapping or training errors | `class_id` does not match keys in `names`, or missing entries in `names` | Verify the integrity of the `names` dictionary and continuity of indices; correct `class_id` |  
| Empty `.txt` files for images with no objects | Loader errors or unnecessary I/O overhead | Incorrect creation of empty label files | Delete empty `.txt` files and follow the convention of "no label file for images with no objects" |  
| Stale cache | Training loading process or metrics do not match expectations | `labels.cache` not updated | Clear or update the cache and re-verify data consistency |  
| Incorrect paths in `data.yaml` | Training command fails to locate data | Incorrect relative path baseline or typos | Use the directory containing `data.yaml` as the baseline and correct the paths for `path`/`train`/`val`/`test` |  
| Mixed task label expressions | Abnormal loader parsing | A single label file contains fields for multiple tasks | Clarify task-specific division of labor and directory organization; avoid mixing fields [^2][^3] |  

The key prevention points for the above issues are reflected in official documentation and loader references; establishing automated scripts and integrating them into CI/CD is an effective way to reduce risks [^2][^3].  


## 10. Appendix: Examples and Templates  

To accelerate implementation, the following sections provide configuration key points for the COCO8 dataset, a general `data.yaml` template, and recommendations for standardized directory templates.  

### Example: Configuration Key Points for the COCO8 Dataset  
- `path`: coco8  
- `train`: images/train  
- `val`: images/val  
- `names`: 80 classes (consistent with COCO classes)  
- `download`: Provides a URL for downloading the standard compressed package  

This example fully demonstrates the use of relative path conventions, class dictionaries, and the automatic download field, and can be used as a reference template for new dataset configurations [^1].  


### General `data.yaml` Template (Detection Task)  

```yaml
# Dataset root directory (usually the directory containing data.yaml)
path: dataset_root

# Training/validation/test splits (relative to `path`; can also use file lists or lists of paths)
train: images/train
val: images/val
test: images/test  # Optional

# Class name dictionary (zero-indexed)
names:
  0: person
  1: bicycle
  2: car
  # Append other classes as needed...

# Optional: Automatic download (for sample datasets)
download:
  url: https://example.com/dataset.zip
```  


### Standardized Directory Template (Detection Task)  

```
dataset_root/
├─ data.yaml
├─ images/
│  ├─ train/
│  │  ├─ <image_0001>.<ext>
│  │  └─ …
│  ├─ val/
│  │  └─ …
│  └─ test/  # Optional
│     └─ …
└─ labels/
   ├─ train/
   │  ├─ <image_0001>.txt
   │  └─ …
   ├─ val/
   │  └─ …
   └─ test/  # Optional
      └─ …
```  


### Example Label `.txt` File (Detection Task)  

```txt
# Label file for image_0001.jpg: image_0001.txt
0 0.517 0.618 0.204 0.312
1 0.123 0.234 0.056 0.078
```  

Referencing official examples and source code documentation can help quickly correct directory and configuration details. It is recommended to perform a comprehensive one-time verification based on this template before launching a new dataset to avoid format-related rework during the training phase [^1][^2][^3].  


---

## Conclusion  

The successful implementation of the YOLO format relies on three pillars: strict directory and file naming conventions, correct `data.yaml` configuration and class mapping, and verifiable label coordinate normalization. Building automated data verification and cache update mechanisms around these three pillars can significantly reduce the costs of pre-training data preparation and in-training anomaly troubleshooting. For multi-task expansion (segmentation, pose estimation, OBB), teams should develop internal templates and operation manuals based on the syntax details in official documentation, and ensure unified label expressions and clear task boundaries. Driven by both conventions and toolchains, the dataset organization for Ultralytics YOLO will be more maintainable, more reproducible, and faster to complete the closed loop from annotation to production.  


---

## References  

[^1]: Ultralytics YOLO Dataset Configuration File Example: coco8.yaml. https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml  
[^2]: Object Detection Datasets Overview - Ultralytics YOLO Docs. https://docs.ultralytics.com/datasets/detect/  
[^3]: Reference for ultralytics/data/dataset.py - Ultralytics YOLO Docs. https://docs.ultralytics.com/reference/data/dataset/  
[^4]: Object Detection - Ultralytics YOLO Documentation (Chinese). https://docs.ultralytics.com/zh/tasks/detect/  
[^5]: Model Validation with Ultralytics YOLO (Chinese). https://docs.ultralytics.com/zh/modes/val/  
[^6]: LabelStudio + YOLO Practical Guide: Full Workflow from Annotation to Training (Chinese). https://www.lixueduan.com/posts/ai/09-labelstudio-train-yolo/  
[^7]: Ultralytics Quickstart. https://docs.ultralytics.com/quickstart/  
[^8]: Releases · ultralytics/ultralytics. https://github.com/ultralytics/ultralytics/releases  
[^9]: Pose Estimation Datasets Overview - Ultralytics YOLO Docs. https://docs.ultralytics.com/datasets/pose/  
[^10]: JSON2YOLO Format Conversion Tool - Ultralytics GitHub. https://github.com/ultralytics/JSON2YOLO