#!/usr/bin/env python3
"""
🐧 PNGN Terminal Animator - Configuration Module
================================================
Copyright (c) 2025 PNGN-Tec LLC

Centralized Configuration System
=================================
Complete configuration for terminal animation rendering including:
- Terminal dimensions and character sizing
- 32-color PNGN palette system
- Team color definitions (PNGN, KLLR, SHMA)
- Visual effect parameters
- Animation settings
- Scaling system configuration (compatible with PNGN_Scaling)
- RGB color utilities with palette-aware operations

Configuration Overview
======================
This module provides all constants and utilities needed for the terminal
animator to function standalone, including the complete PNGN 32-color
palette, team-specific color schemes, effect character sets, and cache
management for optimal performance.

Scaling System Integration
===========================
This module provides the configuration interface expected by PNGN_Scaling
modules, including:
- CacheConfig for pngn_width and pngn_scale
- RenderingConfig for quality modes
- PerformanceConfig for resource limits
- ScalingAlgorithm and QualityMode enums

Color System
============
The PNGN 32-color palette consists of:
- PNGN Team Colors (indices 0-7): Purple/violet spectrum
- KLLR Team Colors (indices 8-15): Pink/magenta spectrum  
- SHMA Team Colors (indices 16-23): Green/yellow spectrum
- Shared Colors (indices 24-31): Whites, grays, cyans

Each color includes RGB tuple and ANSI escape sequence for maximum
compatibility with terminal and image rendering systems.
"""

import threading
import logging
import os
from typing import Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger('pngn_config')

# Type alias for RGB colors
RGBColor = Tuple[int, int, int]

# ============================================================================
# TERMINAL DIMENSIONS
# ============================================================================

TERMINAL_WIDTH = 46      # Characters wide
TERMINAL_HEIGHT = 23     # Lines tall

# Character cell dimensions (pixels)
CHAR_WIDTH = 8           # Standard width
CHAR_HEIGHT = 16         # Standard height
STANDARD_CHAR_WIDTH = 8
STANDARD_CHAR_HEIGHT = 16

# ============================================================================
# ANIMATION SETTINGS
# ============================================================================

CURSOR_BLINK_FRAMES = 12  # Frames per blink cycle
RGB_SPLIT_OFFSET = 1      # Pixel offset for RGB split effect

# ============================================================================
# EFFECT CHARACTER SETS
# ============================================================================

# Glitch characters for digital corruption
GLITCH_CHARS = "▒▓░█▄▀▌▐┼╬▪▫"

# Void/darkness characters
VOID_CHARS = "○▨◇◆◡◢◣◤◥◦"

# Gradient characters for smooth transitions
GRADIENT_CHARS = " ░▒▓█"

# Organic/flowing characters
ORGANIC_CHARS = "~∼≈≋∿╱"

# Particle effect characters
PARTICLE_CHARS = "·∙•◦○◯⊙⊚⊛"

# Static noise characters
STATIC_CHARS = '.:·∙'

# Corruption characters
CORRUPTION_CHARS = "▨▓▒░▄▀▌▐"

# Base corruption probability
CORRUPTION_CHANCE_BASE = 0.005

# ============================================================================
# CONFIGURATION ENUMS (PNGN_Scaling compatible)
# ============================================================================

class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"


class ScalingAlgorithm(Enum):
    """Image scaling algorithms"""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


class QualityMode(Enum):
    """Rendering quality modes"""
    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"
    EXCELLENT = "excellent"


# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

@dataclass
class CacheConfig:
    """
    Cache configuration parameters.
    
    The max_memory_mb parameter is allocated as follows:
    - pngn_width: max_memory_mb / 5 (20% of total)
    - pngn_scale: max_memory_mb / 2 (50% of total)
    
    Attributes:
        ttl_seconds: Time-to-live for cache entries
        cleanup_interval_seconds: How often to run cache cleanup
        default_size: Default number of cache entries
        high_water_mark: Maximum cache size before eviction
        low_water_mark: Target size after eviction
        max_memory_mb: Base memory allocation value
        eviction_strategy: Strategy for cache eviction
        enable_caching: Master switch for caching
        enable_compression: Whether to compress cached data
    """
    
    # Time-based settings
    ttl_seconds: int = 300
    cleanup_interval_seconds: int = 60
    
    # Size limits
    default_size: int = 50
    high_water_mark: int = 75
    low_water_mark: int = 35
    
    # Memory allocation base value
    max_memory_mb: float = 50.0
    
    # Strategy
    eviction_strategy: CacheStrategy = CacheStrategy.LRU
    
    # Feature flags
    enable_caching: bool = True
    enable_compression: bool = False
    
    def validate(self) -> bool:
        """Validate cache configuration"""
        if self.ttl_seconds <= 0:
            raise ValueError("Cache TTL must be positive")
        if self.high_water_mark <= self.low_water_mark:
            raise ValueError("High water mark must exceed low water mark")
        if self.max_memory_mb <= 0:
            raise ValueError("Cache memory limit must be positive")
        return True


# ============================================================================
# RENDERING CONFIGURATION
# ============================================================================

@dataclass
class RenderingConfig:
    """Rendering and display configuration"""
    
    # Base character dimensions
    base_char_width: int = STANDARD_CHAR_WIDTH
    base_char_height: int = STANDARD_CHAR_HEIGHT
    
    # Base document dimensions
    base_columns: int = TERMINAL_WIDTH
    base_rows: int = TERMINAL_HEIGHT
    
    # Dirty rectangle tracking
    dirty_block_size: int = 8
    enable_dirty_rect: bool = True
    
    # Font settings
    default_font_size: int = 16
    font_cache_size: int = 50
    font_memory_mb: float = 20.0
    
    # Quality settings per mode
    quality_settings: Dict[QualityMode, Dict[str, Any]] = field(default_factory=lambda: {
        QualityMode.SPEED: {
            "char_width": 8,
            "char_height": 16,
            "scaling": ScalingAlgorithm.NEAREST
        },
        QualityMode.BALANCED: {
            "char_width": 10,
            "char_height": 20,
            "scaling": ScalingAlgorithm.BILINEAR
        },
        QualityMode.QUALITY: {
            "char_width": 12,
            "char_height": 24,
            "scaling": ScalingAlgorithm.BICUBIC
        },
        QualityMode.EXCELLENT: {
            "char_width": 12,
            "char_height": 24,
            "scaling": ScalingAlgorithm.LANCZOS
        }
    })
    
    def get_quality_settings(self, mode: QualityMode) -> Dict[str, Any]:
        """Get settings for a quality mode"""
        return self.quality_settings.get(mode, self.quality_settings[QualityMode.BALANCED])
    
    def validate(self) -> bool:
        """Validate rendering configuration"""
        if self.base_char_width <= 0 or self.base_char_height <= 0:
            raise ValueError("Character dimensions must be positive")
        if self.base_columns <= 0 or self.base_rows <= 0:
            raise ValueError("Document dimensions must be positive")
        if self.dirty_block_size <= 0:
            raise ValueError("Dirty block size must be positive")
        if self.font_memory_mb <= 0:
            raise ValueError("Font memory must be positive")
        return True


# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

@dataclass
class PerformanceConfig:
    """Performance and resource limits"""
    
    # Document limits
    max_document_width: int = 10000
    max_document_height: int = 10000
    max_render_pixels: int = 100_000_000
    max_line_length: int = 10000
    
    # Memory thresholds
    memory_warning_threshold: float = 100.0
    memory_critical_threshold: float = 50.0
    
    # Thread settings
    max_worker_threads: int = 4
    cleanup_thread_timeout: float = 5.0
    
    # Batch processing
    batch_size_threshold: int = 10
    max_batch_size: int = 100
    
    # Timeouts
    render_timeout_seconds: float = 30.0
    scale_timeout_seconds: float = 10.0
    
    def validate(self) -> bool:
        """Validate performance configuration"""
        if self.max_render_pixels <= 0:
            raise ValueError("Max render pixels must be positive")
        if self.memory_critical_threshold >= self.memory_warning_threshold:
            raise ValueError("Critical threshold must be less than warning threshold")
        if self.batch_size_threshold <= 0:
            raise ValueError("Batch size threshold must be positive")
        if self.max_worker_threads <= 0:
            raise ValueError("Max worker threads must be positive")
        return True


# ============================================================================
# MAIN CONFIGURATION CLASS
# ============================================================================

@dataclass
class ScalingSystemConfig:
    """Complete system configuration"""
    
    # Sub-configurations
    cache: CacheConfig = field(default_factory=CacheConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # System-wide settings
    debug_mode: bool = False
    log_level: str = "INFO"
    config_file: Optional[str] = None
    auto_reload: bool = False
    
    def validate(self) -> bool:
        """Validate entire configuration"""
        self.cache.validate()
        self.rendering.validate()
        self.performance.validate()
        return True


# ============================================================================
# CONFIGURATION MANAGER (SINGLETON)
# ============================================================================

class ConfigurationManager:
    """
    Singleton configuration manager with runtime reloading.
    Thread-safe management of global configuration with change notifications.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._config = ScalingSystemConfig()
        self._callbacks = []
        self._config_lock = threading.RLock()
        self._load_environment_overrides()
        
        self._initialized = True
        logger.info("Configuration manager initialized")
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables"""
        
        # Cache settings
        if 'PNGN_CACHE_TTL' in os.environ:
            self._config.cache.ttl_seconds = int(os.environ['PNGN_CACHE_TTL'])
        if 'PNGN_CACHE_SIZE' in os.environ:
            self._config.cache.default_size = int(os.environ['PNGN_CACHE_SIZE'])
        if 'PNGN_CACHE_MEMORY' in os.environ:
            self._config.cache.max_memory_mb = float(os.environ['PNGN_CACHE_MEMORY'])
        
        # Rendering settings
        if 'PNGN_FONT_MEMORY' in os.environ:
            self._config.rendering.font_memory_mb = float(os.environ['PNGN_FONT_MEMORY'])
        if 'PNGN_DIRTY_BLOCK_SIZE' in os.environ:
            self._config.rendering.dirty_block_size = int(os.environ['PNGN_DIRTY_BLOCK_SIZE'])
        
        # Performance settings
        if 'PNGN_MAX_THREADS' in os.environ:
            self._config.performance.max_worker_threads = int(os.environ['PNGN_MAX_THREADS'])
        if 'PNGN_BATCH_THRESHOLD' in os.environ:
            self._config.performance.batch_size_threshold = int(os.environ['PNGN_BATCH_THRESHOLD'])
        
        # Debug mode
        if 'PNGN_DEBUG' in os.environ:
            self._config.debug_mode = os.environ['PNGN_DEBUG'].lower() in ('true', '1', 'yes')
    
    @property
    def config(self) -> ScalingSystemConfig:
        """Get current configuration"""
        with self._config_lock:
            return self._config
    
    def reload(self, new_config: Optional[ScalingSystemConfig] = None) -> bool:
        """
        Reload configuration and notify callbacks.
        
        Args:
            new_config: New configuration to apply (reloads from env if None)
            
        Returns:
            True if reload successful
        """
        with self._config_lock:
            old_config = self._config
            
            try:
                if new_config:
                    new_config.validate()
                    self._config = new_config
                else:
                    # Reload from environment
                    self._load_environment_overrides()
                
                # Notify callbacks
                self._notify_callbacks(old_config, self._config)
                
                logger.info("Configuration reloaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Configuration reload failed: {e}")
                self._config = old_config
                return False
    
    def register_callback(self, callback: Callable[[ScalingSystemConfig, ScalingSystemConfig], None]):
        """
        Register callback for configuration changes.
        
        Args:
            callback: Function called with (old_config, new_config)
        """
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable):
        """Remove a registered callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _notify_callbacks(self, old_config: ScalingSystemConfig, new_config: ScalingSystemConfig):
        """Notify all registered callbacks of configuration change"""
        for callback in self._callbacks:
            try:
                callback(old_config, new_config)
            except Exception as e:
                logger.error(f"Callback notification failed: {e}")


# ============================================================================
# PUBLIC API FUNCTIONS (PNGN_Scaling compatible)
# ============================================================================

_manager = ConfigurationManager()

def get_config() -> ScalingSystemConfig:
    """Get current system configuration"""
    return _manager.config

def reload_config(new_config: Optional[ScalingSystemConfig] = None) -> bool:
    """Reload system configuration"""
    return _manager.reload(new_config)

def register_config_callback(callback: Callable[[ScalingSystemConfig, ScalingSystemConfig], None]):
    """Register for configuration change notifications"""
    _manager.register_callback(callback)

def unregister_config_callback(callback: Callable):
    """Unregister a configuration change callback"""
    _manager.unregister_callback(callback)

# Module-specific convenience functions matching what pngn_scale expects
def get_cache_config() -> CacheConfig:
    """Get cache configuration"""
    return _manager.config.cache

def get_rendering_config() -> RenderingConfig:
    """Get rendering configuration"""
    return _manager.config.rendering

def get_performance_config() -> PerformanceConfig:
    """Get performance configuration"""
    return _manager.config.performance

# Backward compatibility alias
register_callback = register_config_callback

# ============================================================================
# PNGN 32-COLOR PALETTE
# ============================================================================

PNGN_32_COLORS = {
    # PNGN TEAM COLORS (0-7)
    0: {'name': 'PNGN Purple', 'rgb': (191, 0, 255), 'ansi': '\033[38;2;191;0;255m'},
    1: {'name': 'Neon Violet', 'rgb': (182, 0, 255), 'ansi': '\033[38;2;175;0;255m'},
    2: {'name': 'Electric Violet', 'rgb': (173, 0, 255), 'ansi': '\033[38;2;159;0;255m'},
    3: {'name': 'Ultra Violet', 'rgb': (164, 0, 255), 'ansi': '\033[38;2;143;0;255m'},
    4: {'name': 'Neon Blue', 'rgb': (127, 0, 255), 'ansi': '\033[38;2;127;0;255m'},
    5: {'name': 'Deep Blue', 'rgb': (118, 0, 255), 'ansi': '\033[38;2;111;0;255m'},
    6: {'name': 'Royal Purple', 'rgb': (135, 0, 247), 'ansi': '\033[38;2;135;0;247m'},
    7: {'name': 'Void Black', 'rgb': (15, 15, 35), 'ansi': '\033[38;2;15;15;35m'},
    
    # KLLR TEAM COLORS (8-15)
    8: {'name': 'KLLR Pink', 'rgb': (255, 0, 215), 'ansi': '\033[38;2;255;0;215m'},
    9: {'name': 'Electric Fuschia', 'rgb': (255, 0, 197), 'ansi': '\033[38;2;255;0;191m'},
    10: {'name': 'Deep Pink', 'rgb': (255, 0, 179), 'ansi': '\033[38;2;255;0;167m'},
    11: {'name': 'Hot Pink', 'rgb': (255, 0, 161), 'ansi': '\033[38;2;255;0;143m'},
    12: {'name': 'Neon Magenta', 'rgb': (255, 32, 191), 'ansi': '\033[38;2;255;32;191m'},
    13: {'name': 'Electric Rose', 'rgb': (255, 32, 173), 'ansi': '\033[38;2;255;48;167m'},
    14: {'name': 'Crimson Pink', 'rgb': (247, 0, 135), 'ansi': '\033[38;2;247;0;135m'},
    15: {'name': 'Blood Black', 'rgb': (26, 0, 16), 'ansi': '\033[38;2;26;0;16m'},
    
    # SHMA TEAM COLORS (16-23)
    16: {'name': 'Neon Green', 'rgb': (0, 255, 0), 'ansi': '\033[38;2;0;255;0m'},
    17: {'name': 'Toxic Lime', 'rgb': (36, 255, 0), 'ansi': '\033[38;2;48;255;0m'},
    18: {'name': 'Radiation Green', 'rgb': (0, 255, 36), 'ansi': '\033[38;2;0;255;48m'},
    19: {'name': 'Acid Lime', 'rgb': (72, 255, 0), 'ansi': '\033[38;2;96;255;0m'},
    20: {'name': 'Electric Yellow', 'rgb': (159, 255, 0), 'ansi': '\033[38;2;159;255;0m'},
    21: {'name': 'Amber Glow', 'rgb': (223, 255, 0), 'ansi': '\033[38;2;223;255;0m'},
    22: {'name': 'Fire Orange', 'rgb': (255, 191, 0), 'ansi': '\033[38;2;255;191;0m'},
    23: {'name': 'Toxic Black', 'rgb': (10, 26, 10), 'ansi': '\033[38;2;10;26;10m'},
    
    # SHARED COLORS (24-31)
    24: {'name': 'White Flash', 'rgb': (255, 255, 255), 'ansi': '\033[38;2;255;255;255m'},
    25: {'name': 'Ghost White', 'rgb': (230, 230, 230), 'ansi': '\033[38;2;240;240;250m'},
    26: {'name': 'Plasma Silver', 'rgb': (205, 205, 205), 'ansi': '\033[38;2;210;210;235m'},
    27: {'name': 'Medium Gray', 'rgb': (180, 180, 180), 'ansi': '\033[38;2;170;170;170m'},
    28: {'name': 'Dark Gray', 'rgb': (155, 155, 155), 'ansi': '\033[38;2;110;110;110m'},
    29: {'name': 'Digital Cyan', 'rgb': (0, 191, 255), 'ansi': '\033[38;2;0;191;255m'},
    30: {'name': 'Aqua Flash', 'rgb': (0, 255, 255), 'ansi': '\033[38;2;0;223;255m'},
    31: {'name': 'Deep Sky Blue', 'rgb': (0, 127, 255), 'ansi': '\033[38;2;0;159;255m'},
}

# ============================================================================
# TEAM COLOR DEFINITIONS
# ============================================================================

TEAM_COLORS = {
    'PNGN': {
        'primary': PNGN_32_COLORS[0]['ansi'],    # PNGN Purple
        'glow': PNGN_32_COLORS[1]['ansi'],       # Neon Violet
        'accent': PNGN_32_COLORS[3]['ansi'],     # Ultra Violet
    },
    'KLLR': {
        'primary': PNGN_32_COLORS[8]['ansi'],    # KLLR Pink
        'glow': PNGN_32_COLORS[9]['ansi'],       # Electric Fuschia
        'accent': PNGN_32_COLORS[12]['ansi'],    # Neon Magenta
    },
    'SHMA': {
        'primary': PNGN_32_COLORS[16]['ansi'],   # Neon Green
        'glow': PNGN_32_COLORS[17]['ansi'],      # Toxic Lime
        'accent': PNGN_32_COLORS[19]['ansi'],    # Acid Lime
    }
}

# ============================================================================
# RGB COLOR UTILITIES
# ============================================================================

class RGBColors:
    """
    RGB color values and utilities for the PNGN 32-color palette.
    
    Provides palette-aware color manipulation ensuring all operations
    stay within the defined PNGN color space.
    """
    
    # Team signature colors (EXACT MATCHES)
    PNGN_PRIMARY = PNGN_32_COLORS[0]['rgb']      # (191, 0, 255)
    KLLR_PRIMARY = PNGN_32_COLORS[8]['rgb']      # (255, 0, 215)
    SHMA_PRIMARY = PNGN_32_COLORS[16]['rgb']     # (0, 255, 0)
    
    # Neon accent variations
    PNGN_ACCENT = PNGN_32_COLORS[2]['rgb']       # Electric Violet
    PNGN_HIGHLIGHT = PNGN_32_COLORS[3]['rgb']    # Ultra Violet
    KLLR_ACCENT = PNGN_32_COLORS[9]['rgb']       # Electric Fuschia
    KLLR_HIGHLIGHT = PNGN_32_COLORS[10]['rgb']   # Deep Pink
    SHMA_ACCENT = PNGN_32_COLORS[18]['rgb']      # Radiation Green
    SHMA_HIGHLIGHT = PNGN_32_COLORS[17]['rgb']   # Toxic Lime
    
    # Pure neon colors
    NEON_GREEN = PNGN_32_COLORS[16]['rgb']
    NEON_PINK = PNGN_32_COLORS[8]['rgb']
    NEON_CYAN = PNGN_32_COLORS[29]['rgb']
    NEON_BLUE = PNGN_32_COLORS[4]['rgb']
    NEON_YELLOW = PNGN_32_COLORS[20]['rgb']
    NEON_ORANGE = PNGN_32_COLORS[22]['rgb']
    NEON_RED = PNGN_32_COLORS[12]['rgb']
    NEON_PURPLE = PNGN_32_COLORS[2]['rgb']
    
    # Special effect colors
    VOID_PURPLE = PNGN_32_COLORS[7]['rgb']
    RADIATION_GREEN = PNGN_32_COLORS[18]['rgb']
    COSMIC_BLUE = PNGN_32_COLORS[30]['rgb']
    GLITCH_STATIC = PNGN_32_COLORS[26]['rgb']
    CORRUPTION_RED = PNGN_32_COLORS[14]['rgb']
    FLESH_PINK = PNGN_32_COLORS[11]['rgb']
    TOXIC_YELLOW = PNGN_32_COLORS[19]['rgb']
    
    # UI colors
    UI_BACKGROUND = PNGN_32_COLORS[7]['rgb']
    UI_FOREGROUND = PNGN_32_COLORS[24]['rgb']
    UI_SUCCESS = PNGN_32_COLORS[16]['rgb']
    UI_WARNING = PNGN_32_COLORS[21]['rgb']
    UI_ERROR = PNGN_32_COLORS[12]['rgb']
    UI_INFO = PNGN_32_COLORS[29]['rgb']
    
    @staticmethod
    def rgb_to_ansi(rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to ANSI color code"""
        r, g, b = rgb
        return f"\033[38;2;{r};{g};{b}m"
    
    @staticmethod
    def brighten(color: RGBColor, factor: float = 1.3) -> RGBColor:
        """Map to semantically brighter palette color"""
        if factor <= 1.0:
            return color
        
        brightness_map = {
            # PNGN TEAM RAMPS
            PNGN_32_COLORS[7]['rgb']: PNGN_32_COLORS[7]['rgb'],
            PNGN_32_COLORS[6]['rgb']: PNGN_32_COLORS[3]['rgb'],
            PNGN_32_COLORS[3]['rgb']: PNGN_32_COLORS[2]['rgb'],
            PNGN_32_COLORS[2]['rgb']: PNGN_32_COLORS[0]['rgb'],
            PNGN_32_COLORS[0]['rgb']: PNGN_32_COLORS[1]['rgb'],
            PNGN_32_COLORS[1]['rgb']: PNGN_32_COLORS[4]['rgb'],
            PNGN_32_COLORS[4]['rgb']: PNGN_32_COLORS[5]['rgb'],
            PNGN_32_COLORS[5]['rgb']: PNGN_32_COLORS[5]['rgb'],
        }
        
        return brightness_map.get(color, color)

    @staticmethod
    def dim(color: RGBColor) -> RGBColor:
        """Map to semantically dimmer palette color"""
        dim_map = {
            PNGN_32_COLORS[5]['rgb']: PNGN_32_COLORS[4]['rgb'],
            PNGN_32_COLORS[4]['rgb']: PNGN_32_COLORS[1]['rgb'],
            PNGN_32_COLORS[1]['rgb']: PNGN_32_COLORS[0]['rgb'],
            PNGN_32_COLORS[0]['rgb']: PNGN_32_COLORS[2]['rgb'],
            PNGN_32_COLORS[2]['rgb']: PNGN_32_COLORS[3]['rgb'],
            PNGN_32_COLORS[3]['rgb']: PNGN_32_COLORS[6]['rgb'],
            PNGN_32_COLORS[6]['rgb']: PNGN_32_COLORS[7]['rgb'],
            PNGN_32_COLORS[7]['rgb']: PNGN_32_COLORS[7]['rgb'],
        }
        
        return dim_map.get(color, color)
    
    @staticmethod
    def blend(color1: RGBColor, color2: RGBColor, ratio: float = 0.5) -> RGBColor:
        """Select between two palette colors based on ratio"""
        if ratio < 0.25:
            return color1
        elif ratio > 0.75:
            return color2
        else:
            return color2 if ratio > 0.5 else color1


# ============================================================================
# BUILDING BLOCKS (for compatibility)
# ============================================================================

class BuildingBlocks:
    """
    Compatibility class for accessing colors and utilities.
    
    Provides a centralized access point for the PNGN color system,
    matching the interface expected by terminal.py and other modules.
    """
    PNGN_32_COLORS = PNGN_32_COLORS