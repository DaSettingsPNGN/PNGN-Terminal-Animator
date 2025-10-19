#!/usr/bin/env python3
"""
🐧 PNGN Terminal Animator - Width Calculation Module
====================================================
Copyright (c) 2025 PNGN-Tec LLC

Visual Width Calculation System
================================
Accurate text width measurement for terminal and monospace rendering,
providing the foundation for proper text layout across different displays.

Core Features
=============
- Unicode-aware width calculation (CJK, emoji, combining marks)
- Control character handling with proper exclusion
- Thread-safe caching with memory bounds
- Codepoint-level optimization for common characters
- Fallback support when wcwidth unavailable
- Runtime configuration updates

Technical Implementation
========================
- Uses wcwidth library when available for accurate measurements
- Handles non-BMP characters (emoji, supplementary planes)
- LRU cache with both count and memory limits
- Per-codepoint caching for frequently used characters
- Control character detection and proper width exclusion
- Configuration-based cache management

Primary Use Cases
=================
- Terminal text layout and wrapping
- Monospace font rendering
- Document formatting with structure preservation
- Animation text measurement

Module Interface
================
- WidthCalculator: Main class with full features
- get_width(): Simple function for single strings
- get_widths(): Batch processing for multiple strings
- clear_default_cache(): Clear the default calculator cache

Example Usage
=============
```python
from pngn_width import get_width, get_widths

# Single string
width = get_width("Hello, World!")  # Returns 13

# Multiple strings
widths = get_widths(["Hello", "World"])  # Returns [5, 5]

# Custom calculator with specific cache size
from pngn_width import WidthCalculator
calc = WidthCalculator(cache_size=500, cache_memory_mb=5.0)
width = calc.get_width("Custom text")
```
"""

import threading
import logging
from typing import Optional, List, Dict, Union
from collections import OrderedDict
import unicodedata

# Configure logging
logger = logging.getLogger('pngn_width')

# Import configuration
try:
    from config import get_cache_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    logger.info("config not available - using default values")

# Import wcwidth with fallback
try:
    from wcwidth import wcwidth, wcswidth
    WCWIDTH_AVAILABLE = True
except ImportError:
    WCWIDTH_AVAILABLE = False
    logger.info("wcwidth not installed - using fallback width calculation")
    
    def wcwidth(char: str) -> int:
        """Fallback single character width calculation."""
        if not char:
            return 0
        code = ord(char)
        # Control characters
        if code < 32 or (0x7f <= code < 0xa0):
            return -1
        # Check East Asian width
        ea_width = unicodedata.east_asian_width(char)
        if ea_width in ('F', 'W'):  # Fullwidth or Wide
            return 2
        elif ea_width in ('H', 'Na', 'N'):  # Halfwidth, Narrow, Neutral
            return 1
        else:  # Ambiguous
            return 1
    
    def wcswidth(text: str) -> int:
        """Fallback string width calculation."""
        if not text:
            return 0
        width = 0
        for char in text:
            char_width = wcwidth(char)
            if char_width < 0:
                return -1  # Contains control characters
            width += char_width
        return width


class WidthCalculator:
    """
    Thread-safe text width calculator with caching.
    
    Provides accurate visual width for terminal/monospace rendering,
    handling all Unicode ranges including control characters, combining
    marks, and multi-column characters. Integrates with configuration
    system for runtime updates when available.
    
    Attributes:
        stats: Dictionary containing calculation statistics
        
    Cache Behavior:
    - LRU eviction when size limit reached
    - Memory-bounded with automatic cleanup
    - Thread-safe for concurrent access
    - Separate codepoint cache for hot path optimization
    
    Performance Characteristics:
    - O(1) cache lookup
    - O(n) width calculation where n is string length
    - Minimal allocations for cached results
    - Lock contention minimized with read-write pattern
    """
    
    def __init__(self, 
                 cache_size: Optional[int] = None,
                 cache_memory_mb: Optional[float] = None,
                 use_codepoint_cache: bool = True,
                 enable_cache: bool = True):
        """
        Initialize width calculator.
        
        Args:
            cache_size: Maximum number of cached strings (uses config if None)
            cache_memory_mb: Maximum cache memory in MB (uses config if None)
            use_codepoint_cache: Whether to cache individual codepoint widths
            enable_cache: Whether to enable string caching
        """
        # Load configuration values
        if CONFIG_AVAILABLE and (cache_size is None or cache_memory_mb is None):
            cache_config = get_cache_config()
            
            if cache_size is None:
                cache_size = cache_config.default_size
            if cache_memory_mb is None:
                cache_memory_mb = cache_config.max_memory_mb / 5  # Use 1/5 of total
            
            # Check if caching is globally disabled
            if not cache_config.enable_caching:
                enable_cache = False
                logger.info("Caching disabled by configuration")
        
        # Use defaults if config not available or values not provided
        cache_size = cache_size or 1000
        cache_memory_mb = cache_memory_mb or 10.0
        
        # String cache with LRU eviction
        self._string_cache = OrderedDict()
        self._cache_size = cache_size
        self._cache_memory_limit = int(cache_memory_mb * 1024 * 1024)
        self._cache_memory = 0
        self._cache_enabled = enable_cache
        self._lock = threading.Lock()
        
        # Codepoint cache for frequently used characters
        self._use_codepoint_cache = use_codepoint_cache
        if use_codepoint_cache:
            self._codepoint_cache = self._build_codepoint_cache()
        else:
            self._codepoint_cache = {}
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'calculations': 0,
            'control_chars_handled': 0,
            'cache_evictions': 0,
            'errors': 0
        }
        
        logger.info(f"WidthCalculator initialized with cache_size={cache_size}, "
                   f"memory_limit={cache_memory_mb}MB, cache_enabled={enable_cache}")
    
    def get_width(self, text: str) -> int:
        """
        Get visual width of text in terminal columns.
        
        Args:
            text: Text to measure
            
        Returns:
            Visual width in columns (0 for empty/control-only text)
        """
        if not text:
            return 0
        
        # Check string cache first
        cached = self._get_cached(text)
        if cached is not None:
            return cached
        
        # Calculate width
        try:
            width = self._calculate_width(text)
            self.stats['calculations'] += 1
        except Exception as e:
            logger.error(f"Width calculation error: {e}")
            self.stats['errors'] += 1
            # Fallback to simple length
            width = len(text)
        
        # Cache the result
        self._cache_result(text, width)
        
        return width
    
    def get_widths(self, texts: List[str]) -> List[int]:
        """
        Get widths for multiple strings.
        
        Args:
            texts: List of strings to measure
            
        Returns:
            List of visual widths in columns
        """
        return [self.get_width(text) for text in texts]
    
    def get_line_widths(self, lines: List[str]) -> List[int]:
        """
        Calculate widths for document lines.
        Alias for get_widths() for compatibility.
        
        Args:
            lines: Document lines
            
        Returns:
            List of line widths
        """
        return self.get_widths(lines)
    
    def _get_cached(self, text: str) -> Optional[int]:
        """Get cached width if available."""
        if not self._cache_enabled:
            return None
            
        with self._lock:
            if text in self._string_cache:
                # Move to end for LRU
                self._string_cache.move_to_end(text)
                self.stats['cache_hits'] += 1
                return self._string_cache[text]
        
        self.stats['cache_misses'] += 1
        return None
    
    def _calculate_width(self, text: str) -> int:
        """
        Calculate visual width handling all Unicode cases.
        
        Args:
            text: Text to measure
            
        Returns:
            Visual width in columns
        """
        if not WCWIDTH_AVAILABLE:
            # Simple fallback
            return len(text)
        
        # Try fast path with wcswidth
        width = wcswidth(text)
        
        if width >= 0:
            # Normal case - no control characters
            return width
        
        # Contains control characters - calculate char by char
        self.stats['control_chars_handled'] += 1
        return self._calculate_width_with_control(text)
    
    def _calculate_width_with_control(self, text: str) -> int:
        """
        Calculate width for text containing control characters.
        
        Args:
            text: Text possibly containing control characters
            
        Returns:
            Visual width excluding control characters
        """
        total_width = 0
        
        for char in text:
            # Check codepoint cache first
            code = ord(char)
            if code in self._codepoint_cache:
                total_width += self._codepoint_cache[code]
                continue
            
            # Calculate width
            if code < 0x10000:  # BMP character
                char_width = wcwidth(char) if WCWIDTH_AVAILABLE else 1
            else:  # Non-BMP (emoji, etc.)
                # These often display as 2 columns
                if WCWIDTH_AVAILABLE:
                    char_width = wcwidth(char)
                    if char_width is None:
                        char_width = 2  # Common for emoji
                else:
                    char_width = 2
            
            # Only add positive widths (skip control/zero-width)
            if char_width > 0:
                total_width += char_width
                
                # Cache this codepoint if frequently used
                if self._use_codepoint_cache and code < 0x10000:
                    self._codepoint_cache[code] = char_width
        
        return total_width
    
    def _cache_result(self, text: str, width: int):
        """
        Cache a width calculation with memory management.
        
        Args:
            text: Original text
            width: Calculated width
        """
        if not self._cache_enabled:
            return
            
        # Estimate memory usage
        text_size = len(text.encode('utf-8', errors='ignore'))
        
        with self._lock:
            self._enforce_cache_limits(text_size)
            
            # Add to cache
            self._string_cache[text] = width
            self._cache_memory += text_size
    
    def _enforce_cache_limits(self, new_size: int = 0):
        """
        Enforce cache size and memory limits.
        
        Args:
            new_size: Size of item being added (for preemptive cleanup)
        """
        # Check if we need to evict entries
        while self._string_cache and (
            len(self._string_cache) >= self._cache_size or
            self._cache_memory + new_size > self._cache_memory_limit
        ):
            # Evict oldest (first) entry
            evicted_text, _ = self._string_cache.popitem(last=False)
            self._cache_memory -= len(evicted_text.encode('utf-8', errors='ignore'))
            self.stats['cache_evictions'] += 1
            
            if self._cache_memory < 0:
                self._cache_memory = 0  # Correct any drift
    
    def _build_codepoint_cache(self) -> Dict[int, int]:
        """
        Build cache for common codepoints.
        
        Returns:
            Dictionary mapping codepoint to width
        """
        cache = {}
        
        # ASCII printable characters
        for code in range(32, 127):
            cache[code] = 1
        
        # Common zero-width characters
        zero_width_ranges = [
            (0x0300, 0x036F),  # Combining diacritical marks
            (0x1AB0, 0x1AFF),  # Combining diacritical marks extended
            (0x1DC0, 0x1DFF),  # Combining diacritical marks supplement
            (0x20D0, 0x20FF),  # Combining diacritical marks for symbols
            (0xFE20, 0xFE2F),  # Combining half marks
        ]
        
        for start, end in zero_width_ranges:
            for code in range(start, end + 1):
                cache[code] = 0
        
        # Control characters
        for code in range(0, 32):
            cache[code] = 0
        for code in range(0x7F, 0xA0):
            cache[code] = 0
        
        return cache
    
    def clear_cache(self):
        """Clear all cached widths."""
        with self._lock:
            self._string_cache.clear()
            self._cache_memory = 0
            # Keep codepoint cache as it's static
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get calculator statistics.
        
        Returns:
            Dictionary of statistics including:
            - cache_hits: Number of cache hits
            - cache_misses: Number of cache misses
            - cache_hit_rate: Hit rate as percentage
            - calculations: Total calculations performed
            - control_chars_handled: Strings with control characters
            - cache_evictions: Number of evictions
            - errors: Number of errors encountered
            - cache_entries: Current cache size
            - cache_memory_bytes: Current memory usage
            - cache_enabled: Whether caching is enabled
        """
        stats = self.stats.copy()
        
        # Calculate cache hit rate
        total_requests = stats['cache_hits'] + stats['cache_misses']
        if total_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_requests
        else:
            stats['cache_hit_rate'] = 0.0
        
        # Add current cache info
        with self._lock:
            stats['cache_entries'] = len(self._string_cache)
            stats['cache_memory_bytes'] = self._cache_memory
            stats['cache_enabled'] = self._cache_enabled
        
        return stats
    
    def set_cache_enabled(self, enabled: bool):
        """
        Enable or disable caching at runtime.
        
        Args:
            enabled: Whether caching should be enabled
        """
        with self._lock:
            self._cache_enabled = enabled
            if not enabled:
                # Clear cache if disabling
                self._string_cache.clear()
                self._cache_memory = 0
                logger.info("Cache disabled and cleared")
            else:
                logger.info("Cache enabled")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_default_calculator = None
_calculator_lock = threading.Lock()

def get_width(text: str) -> int:
    """
    Get visual width of text using default calculator.
    
    Args:
        text: Text to measure
        
    Returns:
        Visual width in columns
        
    Example:
        >>> get_width("Hello")
        5
        >>> get_width("你好")  # Chinese characters
        4
        >>> get_width("Hello 👋")  # With emoji
        8
    """
    global _default_calculator
    
    if _default_calculator is None:
        with _calculator_lock:
            if _default_calculator is None:
                # Create with configuration-based defaults
                _default_calculator = WidthCalculator()
    
    return _default_calculator.get_width(text)


def get_widths(texts: List[str]) -> List[int]:
    """
    Get widths for multiple strings using default calculator.
    
    Args:
        texts: List of strings to measure
        
    Returns:
        List of visual widths
        
    Example:
        >>> get_widths(["Hello", "World"])
        [5, 5]
        >>> get_widths(["A", "你", "👋"])
        [1, 2, 2]
    """
    global _default_calculator
    
    if _default_calculator is None:
        with _calculator_lock:
            if _default_calculator is None:
                # Create with configuration-based defaults
                _default_calculator = WidthCalculator()
    
    return _default_calculator.get_widths(texts)


def clear_default_cache():
    """
    Clear the default calculator's cache.
    
    Useful when you want to free memory or after processing a large
    batch of unique strings that won't be reused.
    """
    global _default_calculator
    
    if _default_calculator is not None:
        _default_calculator.clear_cache()


def get_default_stats() -> Dict[str, Union[int, float]]:
    """
    Get statistics from the default calculator.
    
    Returns:
        Dictionary of statistics or empty dict if calculator not initialized
        
    Example:
        >>> stats = get_default_stats()
        >>> print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
    """
    global _default_calculator
    
    if _default_calculator is not None:
        return _default_calculator.get_stats()
    return {}


# For compatibility with existing code
calculate_visual_width = get_width
calculate_visual_widths = get_widths