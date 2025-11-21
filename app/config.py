"""
Application Configuration File - Backward Compatible Version
Now using the new configuration management system while maintaining backward compatibility
"""

from typing import Dict, Optional, Set

# Import configuration manager from core module
from app.core.config import config_manager

# Backward compatibility: Use new config manager but retain original interface
settings = config_manager.settings

# Supported image formats - Retrieve from new configuration system
# Convert to lowercase to ensure case-insensitive matching
ALLOWED_IMAGE_FORMATS: Set[str] = set(fmt.lower() for fmt in settings.allowed_image_formats)