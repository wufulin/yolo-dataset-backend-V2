"""Utility functions for file operations"""
import asyncio
import hashlib
import logging
import mimetypes
import os
import shutil
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from app.core.exceptions import (
    FileNotFoundException,
    StorageException,
    ValidationException,
)
# from app.core.decorators import performance_monitor, cache_result, exception_handler  # Avoid circular import
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


@dataclass
class FileInfo:
    """File information data class"""
    path: Path
    size: int
    modified_time: datetime
    mime_type: Optional[str] = None
    hash_md5: Optional[str] = None
    is_dir: bool = False


@dataclass
class ArchiveExtractionResult:
    """Archive extraction result data class"""
    success: bool
    extracted_files: List[Path] = None
    total_size: int = 0
    errors: List[str] = None
    warnings: List[str] = None
    extraction_time: float = 0.0

    def __post_init__(self):
        if self.extracted_files is None:
            self.extracted_files = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class FileOperationError(Exception):
    """File operation base exception"""
    pass


class FileProcessor:
    """File Processor - Provides high-performance file operations"""

    @staticmethod
    def get_file_hash(file_path: Union[str, Path], algorithm: str = "md5", chunk_size: int = 8192) -> str:
        """
        Calculate file hash value

        Args:
            file_path: Path to the file
            algorithm: Hash algorithm (md5, sha1, sha256)
            chunk_size: Reading chunk size

        Returns:
            str: Calculated hash value
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundException(f"File not found: {file_path}")

        if algorithm not in hashlib.algorithms_available:
            raise ValidationException(f"Unsupported hash algorithm: {algorithm}")

        try:
            hash_obj = hashlib.new(algorithm)

            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    hash_obj.update(chunk)

            return hash_obj.hexdigest()

        except Exception as e:
            raise FileOperationError(f"Failed to calculate hash for {file_path}: {e}")

    @staticmethod
    def get_file_info(file_path: Union[str, Path], include_hash: bool = False,
                      hash_algorithm: str = "md5") -> FileInfo:
        """
        Get detailed file information

        Args:
            file_path: Path to the file
            include_hash: Whether to include hash value
            hash_algorithm: Hash algorithm to use if include_hash is True

        Returns:
            FileInfo: File information object
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundException(f"File not found: {file_path}")

        try:
            stat = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            hash_md5 = None

            if include_hash and file_path.is_file():
                hash_md5 = FileProcessor.get_file_hash(file_path, hash_algorithm)

            return FileInfo(
                path=file_path,
                size=stat.st_size,
                modified_time=datetime.fromtimestamp(stat.st_mtime),
                mime_type=mime_type,
                hash_md5=hash_md5,
                is_dir=file_path.is_dir()
            )

        except Exception as e:
            raise FileOperationError(f"Failed to get file info for {file_path}: {e}")

    @staticmethod
    def safe_remove(path: Union[str, Path], force: bool = False) -> bool:
        """
        Safely delete file or directory

        Args:
            path: Path to file or directory
            force: Whether to force deletion (bypass recycle bin etc.)

        Returns:
            bool: True if deletion succeeded, False otherwise
        """
        path = Path(path)

        try:
            if path.is_file():
                path.unlink()
                logger.debug(f"File removed: {path}")
                return True
            elif path.is_dir():
                shutil.rmtree(path)
                logger.debug(f"Directory removed: {path}")
                return True
            else:
                logger.warning(f"Path does not exist: {path}")
                return False

        except Exception as e:
            logger.error(f"Failed to remove path {path}: {e}")
            if force:
                try:
                    # Force deletion attempt
                    if path.is_file():
                        path.unlink(missing_ok=True)
                    elif path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    return True
                except Exception:
                    pass
            return False

    @staticmethod
    def ensure_directory(path: Union[str, Path], mode: int = 0o755) -> Path:
        """
        Ensure directory exists; create if it doesn't

        Args:
            path: Directory path
            mode: Directory permission mode

        Returns:
            Path: Created directory path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True, mode=mode)
        return path

    @staticmethod
    def list_files(directory: Union[str, Path],
                   pattern: Optional[str] = None,
                   recursive: bool = False,
                   include_hidden: bool = False) -> Iterator[FileInfo]:
        """
        List files in directory

        Args:
            directory: Directory path
            pattern: Filename pattern matching
            recursive: Whether to search recursively
            include_hidden: Whether to include hidden files

        Yields:
            FileInfo: File information object for each matching file
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundException(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ValidationException(f"Path is not a directory: {directory}")

        pattern_path = None
        if pattern:
            pattern_path = Path(pattern)

        def match_pattern(filename: str) -> bool:
            if not pattern_path:
                return True

            # Simple pattern matching
            name = filename.lower()
            pattern_str = str(pattern_path).lower()

            if '*' in pattern_str:
                # Convert glob pattern to simple matching
                pattern_parts = pattern_str.split('*')
                if len(pattern_parts) == 2:
                    return (pattern_parts[0] in name or not pattern_parts[0]) and \
                        (pattern_parts[1] in name or not pattern_parts[1])

            return pattern_str in name

        try:
            if recursive:
                for item in directory.rglob('*'):
                    if not item.is_file():
                        continue

                    if not include_hidden and item.name.startswith('.'):
                        continue

                    if match_pattern(item.name):
                        yield FileProcessor.get_file_info(item)
            else:
                for item in directory.iterdir():
                    if not item.is_file():
                        continue

                    if not include_hidden and item.name.startswith('.'):
                        continue

                    if match_pattern(item.name):
                        yield FileProcessor.get_file_info(item)

        except Exception as e:
            raise FileOperationError(f"Error listing files in {directory}: {e}")


class ArchiveProcessor:
    """Archive Processor - Handles archive file operations"""

    @staticmethod
    def extract_zip_safe(zip_path: Union[str, Path],
                         extract_dir: Union[str, Path],
                         root_folder_name: Optional[str] = None,
                         skip_hidden: bool = True) -> ArchiveExtractionResult:
        """
        Safely extract ZIP file, skip specified root directory

        Args:
            zip_path: Path to ZIP file
            extract_dir: Target extraction directory
            root_folder_name: Root directory name to skip (auto-detect if None)
            skip_hidden: Whether to skip hidden files and directories

        Returns:
            ArchiveExtractionResult: Extraction result object
        """
        zip_path = Path(zip_path)
        extract_dir = Path(extract_dir)
        start_time = datetime.now()

        if not zip_path.exists():
            raise FileNotFoundException(f"ZIP file not found: {zip_path}")

        if not zip_path.suffix.lower() in ['.zip', '.jar']:
            raise ValidationException(f"Invalid ZIP file: {zip_path}")

        try:
            FileProcessor.ensure_directory(extract_dir)
            extracted_files = []
            total_size = 0
            errors = []
            warnings = []

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Auto-detect root folder name
                if root_folder_name is None:
                    names = zip_ref.namelist()
                    for name in names:
                        if '/' in name and not name.startswith('__MACOSX') and not name.startswith('._'):
                            root_folder_name = name.split('/')[0]
                            break

                # Extract files
                for member in zip_ref.namelist():
                    # Skip system files
                    if skip_hidden and (member.startswith('__MACOSX/') or member.startswith('._')):
                        continue

                    # Skip root directory entry
                    if root_folder_name and member == root_folder_name + '/':
                        continue

                    # Process file path
                    if root_folder_name and member.startswith(root_folder_name + '/'):
                        new_member = member[len(root_folder_name + '/'):]
                        if not new_member:
                            continue
                    else:
                        new_member = member

                    # Extract file
                    try:
                        target_path = extract_dir / new_member

                        # Ensure target directory exists
                        if '/' in new_member:
                            target_path.parent.mkdir(parents=True, exist_ok=True)

                        # Extract file content
                        with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)

                        extracted_files.append(target_path)
                        total_size += target_path.stat().st_size

                    except Exception as e:
                        errors.append(f"Failed to extract {member}: {e}")

            extraction_time = (datetime.now() - start_time).total_seconds()

            return ArchiveExtractionResult(
                success=len(errors) == 0,
                extracted_files=extracted_files,
                total_size=total_size,
                errors=errors,
                warnings=warnings,
                extraction_time=extraction_time
            )

        except zipfile.BadZipFile:
            raise ValidationException(f"Invalid ZIP file: {zip_path}")
        except Exception as e:
            raise StorageException(f"Failed to extract ZIP file: {e}")

    @staticmethod
    def get_zip_info(zip_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get ZIP file information

        Args:
            zip_path: Path to ZIP file

        Returns:
            Dict: ZIP file metadata
        """
        zip_path = Path(zip_path)

        if not zip_path.exists():
            raise FileNotFoundException(f"ZIP file not found: {zip_path}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                file_count = len([f for f in file_list if not f.endswith('/')])
                dir_count = len([f for f in file_list if f.endswith('/')])

                # Calculate total size
                total_size = sum(zip_ref.getinfo(f).file_size for f in file_list if not f.endswith('/'))

                return {
                    "file_count": file_count,
                    "dir_count": dir_count,
                    "total_size": total_size,
                    "total_size_mb": round(total_size / 1024 / 1024, 2),
                    "compressed_size": zip_path.stat().st_size,
                    "compression_ratio": round((1 - zip_path.stat().st_size / total_size) * 100,
                                               2) if total_size > 0 else 0,
                    "files": file_list[:20],  # First 20 files
                    "has_root_folder": ArchiveProcessor._has_single_root_folder(file_list)
                }

        except Exception as e:
            raise FileOperationError(f"Failed to get ZIP info: {e}")

    @staticmethod
    def _has_single_root_folder(file_list: List[str]) -> bool:
        """Check if ZIP file has a single root folder"""
        if not file_list:
            return False

        # Get first-level directories
        first_level_items = set()
        for item in file_list:
            if '/' in item:
                first_level_items.add(item.split('/')[0])

        # Filter out system folders
        system_folders = {'__MACOSX', '__pycache__', '.DS_Store'}
        first_level_items = first_level_items - system_folders

        return len(first_level_items) == 1


class FileValidator:
    """File Validator - Validates file formats and integrity"""

    # Supported image formats
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}

    # Supported document formats
    SUPPORTED_DOCUMENT_FORMATS = {'.pdf', '.txt', '.json', '.yaml', '.yml', '.csv'}

    # Supported archive formats
    SUPPORTED_ARCHIVE_FORMATS = {'.zip', '.tar', '.gz', '.bz2', '.7z', '.rar'}

    @staticmethod
    def validate_image_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate image file

        Args:
            file_path: Path to image file

        Returns:
            Dict: Validation result
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundException(f"Image file not found: {file_path}")

        if file_path.suffix.lower() not in FileValidator.SUPPORTED_IMAGE_FORMATS:
            raise ValidationException(f"Unsupported image format: {file_path.suffix}")

        try:
            file_info = FileProcessor.get_file_info(file_path, include_hash=False)

            # Basic validation
            if file_info.size == 0:
                raise ValidationException("Image file is empty")

            if file_info.size > 100 * 1024 * 1024:  # 100MB
                raise ValidationException("Image file too large (max 100MB)")

            # Additional format-specific validation can be added here
            # e.g., using PIL to verify image integrity

            return {
                "is_valid": True,
                "file_info": file_info,
                "format": file_path.suffix.lower(),
                "validation_passed": True
            }

        except Exception as e:
            raise FileOperationError(f"Image validation failed: {e}")

    @staticmethod
    def validate_archive_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate archive file

        Args:
            file_path: Path to archive file

        Returns:
            Dict: Validation result
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundException(f"Archive file not found: {file_path}")

        if file_path.suffix.lower() not in FileValidator.SUPPORTED_ARCHIVE_FORMATS:
            raise ValidationException(f"Unsupported archive format: {file_path.suffix}")

        try:
            if file_path.suffix.lower() == '.zip':
                zip_info = ArchiveProcessor.get_zip_info(file_path)

                return {
                    "is_valid": True,
                    "file_info": FileProcessor.get_file_info(file_path),
                    "archive_type": "zip",
                    "info": zip_info,
                    "validation_passed": True
                }
            else:
                # Validation for other formats can be added later
                return {
                    "is_valid": True,
                    "file_info": FileProcessor.get_file_info(file_path),
                    "archive_type": file_path.suffix.lower(),
                    "validation_passed": True
                }

        except Exception as e:
            raise FileOperationError(f"Archive validation failed: {e}")


# Compatibility functions - Maintain backward compatibility
def resolve_target_directory(zip_file_path, target_folder_name=None):
    """Resolve target directory path from ZIP file path"""
    zip_path = Path(zip_file_path)

    if target_folder_name is None:
        target_folder_name = zip_path.stem

    target_dir = zip_path.parent / target_folder_name
    return target_dir


def extract_skip_root_safe(zip_path: str, extract_dir: str, root_folder_name: Optional[str] = None) -> None:
    """Extract ZIP file and skip specified root directory"""
    result = ArchiveProcessor.extract_zip_safe(zip_path, extract_dir, root_folder_name)
    if not result.success and result.errors:
        raise FileOperationError(f"Extraction failed: {'; '.join(result.errors)}")


def ensure_directory(path: str) -> None:
    """Ensure directory exists (compatibility wrapper)"""
    FileProcessor.ensure_directory(path)


def get_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """Calculate file MD5 hash (compatibility wrapper)"""
    return FileProcessor.get_file_hash(file_path, "md5", chunk_size)


def safe_remove(path: str) -> bool:
    """Safely delete file or directory (compatibility wrapper)"""
    return FileProcessor.safe_remove(path)


def get_file_size(file_path: str) -> int:
    """Get file size in bytes (compatibility wrapper)"""
    return FileProcessor.get_file_info(file_path).size


def get_extension(filename: str) -> str:
    """Get file extension (lowercase)"""
    return Path(filename).suffix.lower()


def is_valid_filename(filename: str) -> bool:
    """Check if filename is valid (no invalid characters)"""
    if not filename or filename.startswith('.'):
        return False

    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    return not any(char in filename for char in invalid_chars)
