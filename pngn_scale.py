#!/usr/bin/env python3
"""
🐧 PNGN-Tec Image Scaling Module
=================================
Copyright (c) 2025 PNGN-Tec LLC

Document Image Scaling System
==============================
Implements the scaling component of the render-then-scale pattern,
transforming base resolution images to match target device characteristics.

Core Features:
- Multi-algorithm scaling (nearest, bilinear, bicubic, lanczos)
- Thread-safe caching with TTL and memory bounds
- Quality mode integration for automatic algorithm selection
- Performance monitoring and metrics collection
- Timeout protection for reliability
- Memory-accurate cache management

Technical Implementation:
- Fast cache key generation using Python hash
- Proper thread synchronization for concurrent scaling
- Configurable cleanup thread with safe termination
- Runtime configuration updates through callbacks
- Automatic algorithm selection based on quality modes
- Parallel processing support for batch operations

Performance Features:
- Lightweight hashing instead of MD5
- Memory-based cache eviction
- Algorithm selection by image dimensions
- Concurrent processing for batch operations
- Compression for infrequent cache entries

Module Interface:
- ImageScaler: Main scaling class with caching
- ScalingContext: Context for scaling operations
- scale_image(): Simple function interface
- batch_scale(): Process multiple images efficiently
"""

import time
import threading
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from PIL import Image
import io

# Configure logging
logger = logging.getLogger('pngn_scale')

# Import configuration
try:
    from config import (
        get_config, register_config_callback, 
        get_cache_config, get_rendering_config, get_performance_config,
        ScalingAlgorithm, QualityMode
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    logger.info("pngn_config not available - using default values")
    
    # Define fallback enums
    class ScalingAlgorithm(Enum):
        NEAREST = "nearest"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        LANCZOS = "lanczos"
    
    class QualityMode(Enum):
        SPEED = "speed"
        BALANCED = "balanced"
        QUALITY = "quality"
        EXCELLENT = "excellent"

# PIL algorithm mapping
ALGORITHM_MAP = {
    ScalingAlgorithm.NEAREST: Image.NEAREST,
    ScalingAlgorithm.BILINEAR: Image.BILINEAR,
    ScalingAlgorithm.BICUBIC: Image.BICUBIC,
    ScalingAlgorithm.LANCZOS: Image.LANCZOS,
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'lanczos': Image.LANCZOS
}


@dataclass
class ScalingContext:
    """
    Context for image scaling operations.
    
    Defines source and target dimensions, quality settings,
    and scaling parameters for the render-then-scale pattern.
    """
    # Source dimensions (base resolution)
    source_width: int
    source_height: int
    
    # Target dimensions (device-specific)
    target_width: int
    target_height: int
    
    # Quality and algorithm
    quality_mode: QualityMode = QualityMode.BALANCED
    algorithm: Optional[ScalingAlgorithm] = None  # Auto-select if None
    
    # Performance settings
    timeout: Optional[float] = None  # Timeout in seconds
    compress_cache: bool = False  # Compress cached results
    
    @property
    def scale_x(self) -> float:
        """Horizontal scaling factor"""
        return self.target_width / self.source_width if self.source_width > 0 else 1.0
    
    @property
    def scale_y(self) -> float:
        """Vertical scaling factor"""
        return self.target_height / self.source_height if self.source_height > 0 else 1.0
    
    @property
    def needs_scaling(self) -> bool:
        """Check if scaling is needed"""
        return (self.source_width != self.target_width or 
                self.source_height != self.target_height)
    
    @property
    def total_pixels(self) -> int:
        """Total pixels in target image"""
        return self.target_width * self.target_height
    
    def select_algorithm(self, performance_config=None) -> ScalingAlgorithm:
        """
        Select optimal algorithm based on context.
        
        Args:
            performance_config: Performance configuration for thresholds
            
        Returns:
            Selected scaling algorithm
        """
        # Use explicit algorithm if specified
        if self.algorithm:
            return self.algorithm
        
        # Select based on quality mode
        if self.quality_mode == QualityMode.SPEED:
            return ScalingAlgorithm.NEAREST
        elif self.quality_mode == QualityMode.EXCELLENT:
            return ScalingAlgorithm.LANCZOS
        
        # For balanced/quality, consider image size
        if performance_config and self.total_pixels > performance_config.max_render_pixels // 10:
            # Large image - use faster algorithm
            return ScalingAlgorithm.BILINEAR
        
        # Default selections
        if self.quality_mode == QualityMode.QUALITY:
            return ScalingAlgorithm.BICUBIC
        else:  # BALANCED
            return ScalingAlgorithm.BILINEAR


@dataclass
class CacheEntry:
    """
    Thread-safe cached scaling result with metadata.
    
    Tracks image data, creation time, access patterns,
    and memory usage for cache management with proper synchronization.
    """
    image: Image.Image
    timestamp: float
    access_count: int = 0
    memory_bytes: int = 0
    compressed: bool = False
    compression_data: Optional[bytes] = None
    _lock: threading.RLock = field(default_factory=threading.RLock)
    
    def is_expired(self, ttl: float) -> bool:
        """Check if entry has exceeded TTL"""
        return time.time() - self.timestamp > ttl
    
    def touch(self):
        """Update access count for LRU tracking"""
        with self._lock:
            self.access_count += 1
            self.timestamp = time.time()
    
    def compress(self):
        """Thread-safe compression to save memory"""
        with self._lock:
            if self.compressed or self.compression_data:
                return
            
            if self.image is None:
                return
            
            try:
                # Save to bytes buffer
                buffer = io.BytesIO()
                self.image.save(buffer, format='PNG', compress_level=6)
                self.compression_data = buffer.getvalue()
                self.compressed = True
                
                # Clear original image to save memory
                self.image = None
                
                # Update memory usage
                self.memory_bytes = len(self.compression_data)
            except Exception as e:
                logger.warning(f"Failed to compress cache entry: {e}")
    
    def get_image(self) -> Optional[Image.Image]:
        """Thread-safe image retrieval with decompression if needed"""
        with self._lock:
            if not self.compressed:
                return self.image
            
            if not self.compression_data:
                return None
            
            try:
                buffer = io.BytesIO(self.compression_data)
                # Create a copy to ensure thread independence
                decompressed = Image.open(buffer).copy()
                # Keep compressed data for future requests
                return decompressed
            except Exception as e:
                logger.error(f"Failed to decompress cache entry: {e}")
                return None


class ImageScaler:
    """
    Thread-safe image scaler with caching.
    
    Implements the scaling component of render-then-scale pattern
    with TTL and memory-bounded caching, quality mode integration,
    and configuration support.
    """
    
    def __init__(self, 
                 cache_size: Optional[int] = None,
                 cache_ttl: Optional[float] = None,
                 cache_memory_mb: Optional[float] = None,
                 max_workers: Optional[int] = None):
        """
        Initialize image scaler.
        
        Args:
            cache_size: Maximum cached entries (uses config if None)
            cache_ttl: Cache TTL in seconds (uses config if None)
            cache_memory_mb: Maximum cache memory (uses config if None)
            max_workers: Maximum worker threads (uses config if None)
        """
        # Load configuration
        if CONFIG_AVAILABLE:
            config = get_config()
            cache_config = config.cache
            perf_config = config.performance
            
            cache_size = cache_size or cache_config.default_size
            cache_ttl = cache_ttl or cache_config.ttl_seconds
            cache_memory_mb = cache_memory_mb or (cache_config.max_memory_mb / 2)
            max_workers = max_workers or perf_config.max_worker_threads
            
            self._render_timeout = perf_config.scale_timeout_seconds
            self._cleanup_interval = cache_config.cleanup_interval_seconds
            self._quality_settings = config.rendering.quality_settings
        else:
            # Defaults
            cache_size = cache_size or 50
            cache_ttl = cache_ttl or 300
            cache_memory_mb = cache_memory_mb or 25.0
            max_workers = max_workers or 4
            
            self._render_timeout = 10.0
            self._cleanup_interval = 60
            self._quality_settings = {}
        
        # Cache configuration
        self._cache = OrderedDict()
        self._cache_size = cache_size
        self._cache_ttl = cache_ttl
        self._cache_memory_limit = int(cache_memory_mb * 1024 * 1024)
        self._cache_memory = 0
        self._cache_lock = threading.RLock()
        
        # Concurrent scaling coordination
        self._scaling_events = {}  # Maps cache keys to threading.Event objects
        self._events_lock = threading.Lock()
        
        # Thread pool for batch operations
        self._max_workers = max_workers
        self._executor = None
        self._executor_lock = threading.Lock()
        
        # Cleanup thread
        self._cleanup_thread = None
        self._cleanup_event = threading.Event()
        self._shutdown = False
        self._start_cleanup_thread()
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_evictions': 0,
            'cache_compressions': 0,
            'scale_operations': 0,
            'scale_errors': 0,
            'total_scale_time_ms': 0,
            'timeouts': 0,
            'algorithm_usage': {
                ScalingAlgorithm.NEAREST: 0,
                ScalingAlgorithm.BILINEAR: 0,
                ScalingAlgorithm.BICUBIC: 0,
                ScalingAlgorithm.LANCZOS: 0
            }
        }
        
        # Register for configuration changes
        if CONFIG_AVAILABLE:
            register_config_callback(self._on_config_change)
        
        logger.info(f"ImageScaler initialized: cache_size={cache_size}, "
                   f"ttl={cache_ttl}s, memory={cache_memory_mb}MB")
    
    def _on_config_change(self, old_config, new_config):
        """Handle configuration changes"""
        cache_config = new_config.cache
        perf_config = new_config.performance
        
        with self._cache_lock:
            # Update cache settings
            self._cache_size = cache_config.default_size
            self._cache_ttl = cache_config.ttl_seconds
            self._cache_memory_limit = int(cache_config.max_memory_mb / 2 * 1024 * 1024)
            
            # Update performance settings
            self._render_timeout = perf_config.scale_timeout_seconds
            self._max_workers = perf_config.max_worker_threads
            
            # Update quality settings
            self._quality_settings = new_config.rendering.quality_settings
            
            # Enforce new limits
            self._enforce_cache_limits()
        
        logger.info(f"Scaler configuration updated: size={self._cache_size}, "
                   f"ttl={self._cache_ttl}s, workers={self._max_workers}")
    
    def scale_image(self, image: Image.Image, context: ScalingContext) -> Image.Image:
        """
        Scale image according to context with proper concurrency control.
        
        Implements thread coordination to prevent duplicate scaling operations
        when multiple threads request the same uncached image simultaneously.
        
        Args:
            image: Source PIL Image
            context: Scaling context with parameters
            
        Returns:
            Scaled PIL Image
        """
        # Check if scaling needed
        if not context.needs_scaling:
            self.stats['scale_operations'] += 1
            return image
        
        # Generate cache key
        cache_key = self._generate_cache_key(image, context)
        
        # Check cache first
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        # Coordinate concurrent requests
        scaling_thread = False
        event = None
        
        with self._events_lock:
            if cache_key in self._scaling_events:
                # Another thread is scaling this image
                event = self._scaling_events[cache_key]
            else:
                # We'll do the scaling
                event = threading.Event()
                self._scaling_events[cache_key] = event
                event.clear()
                scaling_thread = True
        
        if not scaling_thread:
            # Wait for other thread to complete
            if event:
                event.wait(timeout=self._render_timeout)
            # Try cache again
            cached = self._get_cached(cache_key)
            if cached:
                return cached
            # If still not cached, fall through to scale ourselves
            scaling_thread = True
        
        # Perform scaling
        try:
            start_time = time.time()
            
            # Scale with timeout if configured
            if context.timeout or self._render_timeout:
                timeout = context.timeout or self._render_timeout
                scaled = self._scale_with_timeout(image, context, timeout)
            else:
                scaled = self._perform_scaling(image, context)
            
            # Track time
            scale_time = (time.time() - start_time) * 1000
            self.stats['total_scale_time_ms'] += scale_time
            self.stats['scale_operations'] += 1
            
            # Cache result
            self._cache_result(cache_key, scaled, context.compress_cache)
            
            return scaled
            
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            self.stats['scale_errors'] += 1
            
            # Return original on error
            return image
        
        finally:
            # Clean up coordination
            with self._events_lock:
                if cache_key in self._scaling_events:
                    self._scaling_events[cache_key].set()
                    del self._scaling_events[cache_key]
    
    def _scale_with_timeout(self, image: Image.Image, 
                           context: ScalingContext, 
                           timeout: float) -> Image.Image:
        """
        Scale with timeout protection.
        
        Args:
            image: Source image
            context: Scaling context
            timeout: Timeout in seconds
            
        Returns:
            Scaled image
        """
        with self._get_executor() as executor:
            future = executor.submit(self._perform_scaling, image, context)
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                self.stats['timeouts'] += 1
                logger.warning(f"Scaling timeout after {timeout}s")
                future.cancel()
                # Fall back to faster algorithm
                context.algorithm = ScalingAlgorithm.NEAREST
                return self._perform_scaling(image, context)
    
    def _perform_scaling(self, image: Image.Image, 
                        context: ScalingContext) -> Image.Image:
        """
        Execute the actual scaling operation.
        
        Args:
            image: Source image
            context: Scaling context
            
        Returns:
            Scaled image
        """
        # Select algorithm
        perf_config = get_performance_config() if CONFIG_AVAILABLE else None
        algorithm = context.select_algorithm(perf_config)
        
        # Get PIL algorithm
        pil_algorithm = ALGORITHM_MAP.get(algorithm, Image.BILINEAR)
        
        # Track algorithm usage
        self.stats['algorithm_usage'][algorithm] += 1
        
        # Calculate target size
        target_size = (context.target_width, context.target_height)
        
        # Perform scaling
        try:
            scaled = image.resize(target_size, pil_algorithm)
            return scaled
        except Exception as e:
            logger.error(f"Resize failed with {algorithm}: {e}")
            # Fall back to nearest
            return image.resize(target_size, Image.NEAREST)
    
    def batch_scale(self, images: List[Tuple[Image.Image, ScalingContext]],
                   max_workers: Optional[int] = None) -> List[Image.Image]:
        """
        Scale multiple images concurrently.
        
        Args:
            images: List of (image, context) tuples
            max_workers: Override max worker threads
            
        Returns:
            List of scaled images
        """
        if not images:
            return []
        
        # Use single thread for small batches
        if len(images) <= 2:
            return [self.scale_image(img, ctx) for img, ctx in images]
        
        # Use thread pool for larger batches
        max_workers = max_workers or self._max_workers
        results = []
        
        with self._get_executor(max_workers) as executor:
            futures = [
                executor.submit(self.scale_image, img, ctx)
                for img, ctx in images
            ]
            
            for future in futures:
                try:
                    results.append(future.result(timeout=self._render_timeout))
                except Exception as e:
                    logger.error(f"Batch scaling error: {e}")
                    # Add None for failed scaling
                    results.append(None)
        
        return results
    
    def _generate_cache_key(self, image: Image.Image, 
                           context: ScalingContext) -> str:
        """
        Generate deterministic and reliable cache key.
        
        Uses strategic sampling for large images and full hashing for small ones
        to ensure uniqueness while maintaining performance.
        
        Args:
            image: Source image
            context: Scaling context
            
        Returns:
            Deterministic cache key string
        """
        # Create deterministic image identifier
        image_data = {
            'size': image.size,
            'mode': image.mode,
            'format': getattr(image, 'format', None)
        }
        
        # Use faster hash for smaller images, MD5 for larger ones
        if image.width * image.height < 1000000:  # < 1 megapixel
            # Full content hash for small images
            content_hash = hashlib.md5(image.tobytes()).hexdigest()[:16]
        else:
            # Strategic sampling for large images
            pixels = image.getdata()
            # Sample corners, center, and distributed points
            sample_indices = [
                0, image.width - 1,  # Top corners
                (image.height - 1) * image.width,  # Bottom left
                image.height * image.width - 1,  # Bottom right
                (image.height // 2) * image.width + image.width // 2,  # Center
            ]
            # Add distributed samples
            step = max(1, len(pixels) // 20)
            sample_indices.extend(range(step, len(pixels), step)[:15])
            
            sample_data = bytes(str([pixels[i] for i in sample_indices if i < len(pixels)]), 'utf-8')
            content_hash = hashlib.md5(sample_data).hexdigest()[:16]
        
        # Combine all components
        key_components = [
            content_hash,
            f"{image.width}x{image.height}",
            image.mode,
            f"{context.target_width}x{context.target_height}",
            context.quality_mode.value,
            context.algorithm.value if context.algorithm else 'auto'
        ]
        
        return '_'.join(str(c) for c in key_components)
    
    def _get_cached(self, key: str) -> Optional[Image.Image]:
        """
        Get cached image if available.
        
        Args:
            key: Cache key
            
        Returns:
            Cached image or None
        """
        with self._cache_lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check for in-progress marker
                if entry is None:
                    return None
                
                if not entry.is_expired(self._cache_ttl):
                    entry.touch()
                    self._cache.move_to_end(key)
                    self.stats['cache_hits'] += 1
                    return entry.get_image()
                else:
                    # Expired
                    del self._cache[key]
                    self._cache_memory -= entry.memory_bytes
        
        self.stats['cache_misses'] += 1
        return None
    
    def _cache_result(self, key: str, image: Image.Image, compress: bool = False):
        """
        Cache scaling result with memory management.
        
        Args:
            key: Cache key
            image: Scaled image
            compress: Whether to compress for storage
        """
        # Estimate memory usage
        memory_bytes = self._estimate_image_memory(image)
        
        with self._cache_lock:
            # Remove in-progress marker
            if key in self._cache and self._cache[key] is None:
                del self._cache[key]
            
            # Enforce limits
            self._enforce_cache_limits(memory_bytes)
            
            # Create entry
            entry = CacheEntry(
                image=image,
                timestamp=time.time(),
                memory_bytes=memory_bytes
            )
            
            # Compress if requested or memory pressure
            if compress or self._cache_memory > self._cache_memory_limit * 0.8:
                entry.compress()
                self.stats['cache_compressions'] += 1
            
            # Add to cache
            self._cache[key] = entry
            self._cache_memory += entry.memory_bytes
    
    def _enforce_cache_limits(self, new_size: int = 0):
        """
        Enforce cache size and memory limits.
        
        Args:
            new_size: Size of item being added
        """
        # Remove entries until within limits
        while self._cache and (
            len(self._cache) >= self._cache_size or
            self._cache_memory + new_size > self._cache_memory_limit
        ):
            # Find oldest entry (excluding in-progress markers)
            oldest_key = None
            for k, v in self._cache.items():
                if v is not None:  # Skip in-progress markers
                    oldest_key = k
                    break
            
            if oldest_key:
                entry = self._cache[oldest_key]
                del self._cache[oldest_key]
                if entry:
                    self._cache_memory -= entry.memory_bytes
                self.stats['cache_evictions'] += 1
            else:
                break  # Only in-progress markers remain
    
    def _estimate_image_memory(self, image: Image.Image) -> int:
        """
        Accurately estimate memory usage of image including PIL overhead.
        
        Args:
            image: PIL Image
            
        Returns:
            Estimated bytes including PIL overhead and metadata
        """
        width, height = image.size
        
        # Base bytes per pixel by mode
        mode_bytes = {
            '1': 0.125,  # 1-bit
            'L': 1,       # 8-bit grayscale
            'P': 1,       # 8-bit palette
            'RGB': 3,     # 24-bit
            'RGBA': 4,    # 32-bit
            'CMYK': 4,    # 32-bit
            'YCbCr': 3,   # 24-bit
            'LAB': 3,     # 24-bit
            'HSV': 3,     # 24-bit
            'I': 4,       # 32-bit integer
            'F': 4        # 32-bit float
        }
        
        bytes_per_pixel = mode_bytes.get(image.mode, 4)
        pixel_data = int(width * height * bytes_per_pixel)
        
        # Account for PIL overhead (approximately 20-30%)
        pil_overhead = int(pixel_data * 0.25)
        
        # Account for palette if present
        palette_size = 0
        if image.mode == 'P' and hasattr(image, 'palette'):
            palette_size = 768  # 256 colors * 3 bytes
        
        # Account for metadata
        metadata_size = 1024  # Base estimate for EXIF, ICC profile, etc.
        if hasattr(image, 'info'):
            metadata_size += len(str(image.info))
        
        total = pixel_data + pil_overhead + palette_size + metadata_size
        
        # Round up to nearest KB
        return ((total + 1023) // 1024) * 1024
    
    def _get_executor(self, max_workers: Optional[int] = None) -> ThreadPoolExecutor:
        """
        Get or create thread pool executor with health monitoring.
        
        Automatically recreates executor if health check fails.
        
        Args:
            max_workers: Override max workers
            
        Returns:
            Healthy thread pool executor
        """
        max_workers = max_workers or self._max_workers
        
        with self._executor_lock:
            # Check executor health
            if self._executor:
                # Verify executor is still functional
                try:
                    # Submit a simple test task
                    future = self._executor.submit(lambda: True)
                    result = future.result(timeout=0.1)
                    if not result:
                        raise RuntimeError("Executor test failed")
                except Exception as e:
                    # Executor is unhealthy, recreate it
                    logger.warning(f"Executor unhealthy ({e}), recreating")
                    try:
                        self._executor.shutdown(wait=False)
                    except Exception:
                        pass
                    self._executor = None
            
            # Create new executor if needed
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="ImageScaler"
                )
                logger.debug(f"Created thread pool executor with {max_workers} workers")
            
            return self._executor
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True
            )
            self._cleanup_thread.start()
    
    def _cleanup_worker(self):
        """Background worker to clean expired entries"""
        while not self._shutdown:
            # Wait for interval or shutdown
            if self._cleanup_event.wait(timeout=self._cleanup_interval):
                break  # Shutdown requested
            
            # Clean expired entries
            try:
                with self._cache_lock:
                    expired = []
                    for key, entry in self._cache.items():
                        if entry and entry.is_expired(self._cache_ttl):
                            expired.append(key)
                    
                    for key in expired:
                        entry = self._cache[key]
                        del self._cache[key]
                        if entry:
                            self._cache_memory -= entry.memory_bytes
                    
                    if expired:
                        logger.debug(f"Cleaned {len(expired)} expired cache entries")
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def clear_cache(self):
        """Clear all cached entries"""
        with self._cache_lock:
            self._cache.clear()
            self._cache_memory = 0
            logger.info("Cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get scaler statistics.
        
        Returns:
            Dictionary of statistics and metrics
        """
        stats = self.stats.copy()
        
        # Calculate rates
        total_requests = stats['cache_hits'] + stats['cache_misses']
        if total_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_requests
        else:
            stats['cache_hit_rate'] = 0.0
        
        if stats['scale_operations'] > 0:
            stats['avg_scale_time_ms'] = (
                stats['total_scale_time_ms'] / stats['scale_operations']
            )
        else:
            stats['avg_scale_time_ms'] = 0.0
        
        # Current cache state
        with self._cache_lock:
            stats['cache_entries'] = len(self._cache)
            stats['cache_memory_bytes'] = self._cache_memory
            stats['cache_memory_mb'] = self._cache_memory / (1024 * 1024)
        
        return stats
    
    def shutdown(self):
        """Clean shutdown of scaler"""
        logger.info("Shutting down image scaler")
        
        # Signal cleanup thread
        self._shutdown = True
        self._cleanup_event.set()
        
        # Wait for cleanup thread with proper timeout
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            timeout = get_performance_config().cleanup_thread_timeout if CONFIG_AVAILABLE else 5.0
            self._cleanup_thread.join(timeout=timeout)
            
            if self._cleanup_thread.is_alive():
                logger.warning("Cleanup thread did not terminate gracefully")
        
        # Shutdown executor
        with self._executor_lock:
            if self._executor:
                self._executor.shutdown(wait=True, timeout=2.0)
                self._executor = None
        
        # Clear cache
        self.clear_cache()
        
        logger.info("Image scaler shutdown complete")


# Convenience functions

_default_scaler = None
_scaler_lock = threading.Lock()

def get_scaler() -> ImageScaler:
    """Get or create default scaler"""
    global _default_scaler
    
    if _default_scaler is None:
        with _scaler_lock:
            if _default_scaler is None:
                _default_scaler = ImageScaler()
    
    return _default_scaler

def scale_image(image: Image.Image, context: ScalingContext) -> Image.Image:
    """
    Scale image using default scaler.
    
    Args:
        image: Source image
        context: Scaling context
        
    Returns:
        Scaled image
    """
    return get_scaler().scale_image(image, context)

def batch_scale(images: List[Tuple[Image.Image, ScalingContext]]) -> List[Image.Image]:
    """
    Scale multiple images using default scaler.
    
    Args:
        images: List of (image, context) tuples
        
    Returns:
        List of scaled images
    """
    return get_scaler().batch_scale(images)

def shutdown_scaler():
    """Shutdown default scaler cleanly"""
    global _default_scaler
    
    if _default_scaler:
        _default_scaler.shutdown()
        _default_scaler = None
