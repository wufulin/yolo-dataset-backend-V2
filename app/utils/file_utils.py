"""Utility functions for file operations - 改进版本，使用现代Python特性."""
import asyncio
import hashlib
import logging
import mimetypes
import os
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from app.core.exceptions import (
    FileNotFoundException,
    StorageException,
    ValidationException,
)

# from app.core.decorators import performance_monitor, cache_result, exception_handler  # 避免循环导入
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


@dataclass
class FileInfo:
    """文件信息数据类"""
    path: Path
    size: int
    modified_time: datetime
    mime_type: Optional[str] = None
    hash_md5: Optional[str] = None
    is_dir: bool = False


@dataclass
class ArchiveExtractionResult:
    """压缩包提取结果"""
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
    """文件操作异常"""
    pass


class FileProcessor:
    """文件处理器 - 提供高性能的文件操作"""

    @staticmethod
    def get_file_hash(file_path: Union[str, Path], algorithm: str = "md5", chunk_size: int = 8192) -> str:
        """
        计算文件哈希值

        Args:
            file_path: 文件路径
            algorithm: 哈希算法 (md5, sha1, sha256)
            chunk_size: 读取块大小

        Returns:
            str: 哈希值
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
        获取文件详细信息

        Args:
            file_path: 文件路径
            include_hash: 是否包含哈希值
            hash_algorithm: 哈希算法

        Returns:
            FileInfo: 文件信息
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
        安全删除文件或目录

        Args:
            path: 文件或目录路径
            force: 是否强制删除（绕过回收站等）

        Returns:
            bool: 删除是否成功
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
                    # 强制删除
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
        确保目录存在，如果不存在则创建

        Args:
            path: 目录路径
            mode: 目录权限

        Returns:
            Path: 创建的目录路径
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
        列出目录中的文件

        Args:
            directory: 目录路径
            pattern: 文件名模式匹配
            recursive: 是否递归搜索
            include_hidden: 是否包含隐藏文件

        Yields:
            FileInfo: 文件信息
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

            # 简单的模式匹配
            name = filename.lower()
            pattern_str = str(pattern_path).lower()

            if '*' in pattern_str:
                # 转换glob模式为简单匹配
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
    """压缩包处理器"""

    @staticmethod
    def extract_zip_safe(zip_path: Union[str, Path],
                        extract_dir: Union[str, Path],
                        root_folder_name: Optional[str] = None,
                        skip_hidden: bool = True) -> ArchiveExtractionResult:
        """
        安全解压ZIP文件，跳过指定的根目录

        Args:
            zip_path: ZIP文件路径
            extract_dir: 解压目标目录
            root_folder_name: 要跳过的根目录名，None则自动检测
            skip_hidden: 是否跳过隐藏文件和目录

        Returns:
            ArchiveExtractionResult: 解压结果
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
                # 自动检测根目录名
                if root_folder_name is None:
                    names = zip_ref.namelist()
                    for name in names:
                        if '/' in name and not name.startswith('__MACOSX') and not name.startswith('._'):
                            root_folder_name = name.split('/')[0]
                            break

                # 提取文件
                for member in zip_ref.namelist():
                    # 跳过系统文件
                    if skip_hidden and (member.startswith('__MACOSX/') or member.startswith('._')):
                        continue

                    # 跳过根目录条目
                    if root_folder_name and member == root_folder_name + '/':
                        continue

                    # 处理文件路径
                    if root_folder_name and member.startswith(root_folder_name + '/'):
                        new_member = member[len(root_folder_name + '/'):]
                        if not new_member:
                            continue
                    else:
                        new_member = member

                    # 提取文件
                    try:
                        target_path = extract_dir / new_member

                        # 确保目标目录存在
                        if '/' in new_member:
                            target_path.parent.mkdir(parents=True, exist_ok=True)

                        # 提取文件内容
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
        获取ZIP文件信息

        Args:
            zip_path: ZIP文件路径

        Returns:
            Dict: ZIP文件信息
        """
        zip_path = Path(zip_path)

        if not zip_path.exists():
            raise FileNotFoundException(f"ZIP file not found: {zip_path}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                file_count = len([f for f in file_list if not f.endswith('/')])
                dir_count = len([f for f in file_list if f.endswith('/')])

                # 计算总大小
                total_size = sum(zip_ref.getinfo(f).file_size for f in file_list if not f.endswith('/'))

                return {
                    "file_count": file_count,
                    "dir_count": dir_count,
                    "total_size": total_size,
                    "total_size_mb": round(total_size / 1024 / 1024, 2),
                    "compressed_size": zip_path.stat().st_size,
                    "compression_ratio": round((1 - zip_path.stat().st_size / total_size) * 100, 2) if total_size > 0 else 0,
                    "files": file_list[:20],  # 前20个文件
                    "has_root_folder": ArchiveProcessor._has_single_root_folder(file_list)
                }

        except Exception as e:
            raise FileOperationError(f"Failed to get ZIP info: {e}")

    @staticmethod
    def _has_single_root_folder(file_list: List[str]) -> bool:
        """检查是否有单一的根文件夹"""
        if not file_list:
            return False

        # 获取第一级目录
        first_level_items = set()
        for item in file_list:
            if '/' in item:
                first_level_items.add(item.split('/')[0])

        # 过滤掉系统文件夹
        system_folders = {'__MACOSX', '__pycache__', '.DS_Store'}
        first_level_items = first_level_items - system_folders

        return len(first_level_items) == 1


class FileValidator:
    """文件验证器"""

    # 支持的图像格式
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}

    # 支持的文档格式
    SUPPORTED_DOCUMENT_FORMATS = {'.pdf', '.txt', '.json', '.yaml', '.yml', '.csv'}

    # 支持的压缩包格式
    SUPPORTED_ARCHIVE_FORMATS = {'.zip', '.tar', '.gz', '.bz2', '.7z', '.rar'}

    @staticmethod
    def validate_image_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        验证图像文件

        Args:
            file_path: 文件路径

        Returns:
            Dict: 验证结果
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundException(f"Image file not found: {file_path}")

        if file_path.suffix.lower() not in FileValidator.SUPPORTED_IMAGE_FORMATS:
            raise ValidationException(f"Unsupported image format: {file_path.suffix}")

        try:
            file_info = FileProcessor.get_file_info(file_path, include_hash=False)

            # 基本验证
            if file_info.size == 0:
                raise ValidationException("Image file is empty")

            if file_info.size > 100 * 1024 * 1024:  # 100MB
                raise ValidationException("Image file too large (max 100MB)")

            # 可以添加更具体的图像格式验证
            # 例如使用PIL验证图像完整性

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
        验证压缩包文件

        Args:
            file_path: 文件路径

        Returns:
            Dict: 验证结果
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
                # 其他格式的验证可以后续添加
                return {
                    "is_valid": True,
                    "file_info": FileProcessor.get_file_info(file_path),
                    "archive_type": file_path.suffix.lower(),
                    "validation_passed": True
                }

        except Exception as e:
            raise FileOperationError(f"Archive validation failed: {e}")


# 兼容性函数 - 保持向后兼容
def resolve_target_directory(zip_file_path, target_folder_name=None):
    """从zip文件路径解析目标目录路径"""
    zip_path = Path(zip_file_path)

    if target_folder_name is None:
        target_folder_name = zip_path.stem

    target_dir = zip_path.parent / target_folder_name
    return target_dir


def extract_skip_root_safe(zip_path: str, extract_dir: str, root_folder_name: Optional[str] = None) -> None:
    """解压zip文件，跳过指定的根目录"""
    result = ArchiveProcessor.extract_zip_safe(zip_path, extract_dir, root_folder_name)
    if not result.success and result.errors:
        raise FileOperationError(f"Extraction failed: {'; '.join(result.errors)}")


def ensure_directory(path: str) -> None:
    """确保目录存在"""
    FileProcessor.ensure_directory(path)


def get_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """计算文件MD5哈希"""
    return FileProcessor.get_file_hash(file_path, "md5", chunk_size)


def safe_remove(path: str) -> bool:
    """安全删除文件或目录"""
    return FileProcessor.safe_remove(path)


def get_file_size(file_path: str) -> int:
    """获取文件大小"""
    return FileProcessor.get_file_info(file_path).size


def get_extension(filename: str) -> str:
    """获取文件扩展名"""
    return Path(filename).suffix.lower()


def is_valid_filename(filename: str) -> bool:
    """检查文件名是否有效"""
    if not filename or filename.startswith('.'):
        return False

    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    return not any(char in filename for char in invalid_chars)
