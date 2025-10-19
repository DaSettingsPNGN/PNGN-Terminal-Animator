#!/usr/bin/env python3
"""
🐧 PNGN Terminal Animator - Deterministic Terminal Renderer
===========================================================
Copyright (c) 2025 PNGN-Tec LLC

Terminal Animation Rendering System
====================================
Professional terminal-style animation renderer with team-themed colors,
visual effects, markdown support, and emoji rendering. Optimized for
creating animated GIFs for Discord and social media.

Core Features
=============
- Three team themes with 32-color PNGN palette
- Markdown rendering (headers, bold, italic, code, links)
- Visual effects (glitch, corruption, static)
- Full emoji support with proper two-cell allocation
- Deterministic animations for reproducible output
- Three-tier caching system for performance
- GIF export optimization for Discord

Technical Implementation
========================
- Pure PIL-based rendering for maximum compatibility
- Frame-based animation system (12 FPS optimized)
- Deterministic effect triggering using content hashing
- Emoji manager with pre-loaded bitmap cache
- Character-level layout engine with font variants
- Team-specific color application and border styling
- ANSI escape sequence parsing and rendering

Color System
============
Uses PNGN 32-color palette with three team themes:
- PNGN (Purple/Digital): Electric purple with glitch aesthetic
- KLLR (Pink/Love): Hot pink with gradient glow
- SHMA (Green/Toxic): Neon green with organic pulsing

Each theme includes:
- Primary color for backgrounds and borders
- Glow colors for text and accents
- Accent colors for effects and highlights
- Corruption and glitch color variants

Visual Effects
==============
Deterministic effect system triggered by content hash:
- Glitch: Brief single-frame digital artifacts
- Corruption: Character replacement during burst windows
- Static: TV-style noise with pulsing intensity

Effects activate at pseudo-random but deterministic moments,
ensuring the same content produces the same animation pattern.

Performance Characteristics
============================
- Frame render time: 10-50ms typical
- Memory usage: ~50MB base + frame buffers
- Cache hit rate: 80%+ for repeated content
- Thread-safe for concurrent rendering

Module Interface
================
- create_terminal(): Factory function for renderer creation
- DeterministicPureTerminalRenderer: Main renderer class
  - render(): Render single frame with effects
  - render_particles(): Render Braille particle effects
  - get_stats(): Get performance statistics

Example Usage
=============
```python
from terminal import create_terminal

# Create renderer
terminal = create_terminal(terminal_width=80, height=24)

# Render frame
frame = terminal.render(
    content="Hello, Terminal! 🎨",
    personality="PNGN",
    effects=["glitch"],
    frame=0
)

# Save as image
frame.save("output.png")

# Create GIF animation
frames = []
for i in range(60):
    frame = terminal.render(
        content=f"Frame {i}",
        personality="PNGN",
        frame=i
    )
    frames.append(frame)

frames[0].save(
    "animation.gif",
    save_all=True,
    append_images=frames[1:],
    duration=83,  # 12 FPS
    loop=0
)
```

Dependencies
============
- numpy: Image buffer manipulation
- Pillow: Image rendering and font handling
- config: Color palette and settings
- pngn_width: Optional width calculation utility

See Also
========
- config.py: Configuration and color system
- pngn_width.py: Unicode width calculator
- example_basic.py: Simple usage example
- example_gif.py: GIF animation examples
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
import re
import time
import math
import hashlib
from pathlib import Path
import logging
from enum import Enum
from collections import defaultdict

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger('PNGN.Terminal.Deterministic')

# ============================================================================
# CONFIGURATION IMPORTS - Standalone Version
# ============================================================================

# Import all configuration from standalone config module
from config import (
    # Terminal dimensions
    TERMINAL_WIDTH, TERMINAL_HEIGHT,
    CHAR_WIDTH, CHAR_HEIGHT,
    STANDARD_CHAR_WIDTH, STANDARD_CHAR_HEIGHT,
    
    # Animation settings
    CURSOR_BLINK_FRAMES,
    RGB_SPLIT_OFFSET,
    
    # Effect character sets
    GLITCH_CHARS,
    VOID_CHARS,
    GRADIENT_CHARS,
    ORGANIC_CHARS,
    PARTICLE_CHARS,
    CORRUPTION_CHARS,
    CORRUPTION_CHANCE_BASE,
    STATIC_CHARS,
    
    # Color system - PNGN 32-color palette
    PNGN_32_COLORS,
    TEAM_COLORS,
    BuildingBlocks,
    RGBColors,
    
    # Cache configuration
    get_cache_config,
)

# Configuration available from standalone module
BUILDING_BLOCKS_AVAILABLE = True

# Simplified cache for standalone operation
# The renderer includes its own internal caching system
CACHE_AVAILABLE = False
UnifiedCacheSystem = None
get_unified_cache = None

logger.info("Using PNGN_32_COLORS from standalone config.py")
logger.info(f"Loaded {len(PNGN_32_COLORS)} colors from PNGN palette")

# ============================================================================
# HYBRID OPTIMIZATION IMPORTS (Optional)
# ============================================================================

# Try to import vectorized modules for acceleration
# These are optional performance enhancements
try:
    from pngn_vector import (
        VectorizedCharacterMetrics,
        VectorizedDirtyTracker,
        get_vectorized_system
    )
    VECTOR_MODULE_AVAILABLE = True
    logger.info("pngn_vector module loaded - vectorized operations enabled")
except ImportError:
    VECTOR_MODULE_AVAILABLE = False
    logger.info("pngn_vector not available - using built-in operations")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_rgb_from_ansi(ansi_code: str) -> Tuple[int, int, int]:
    """
    Extract RGB tuple from ANSI escape sequence.
    
    Args:
        ansi_code: ANSI color code or RGB tuple
        
    Returns:
        RGB color tuple (r, g, b)
        
    Examples:
        >>> extract_rgb_from_ansi('\033[38;2;255;0;255m')
        (255, 0, 255)
        >>> extract_rgb_from_ansi((255, 0, 255))
        (255, 0, 255)
    """
    if isinstance(ansi_code, tuple):
        # Already an RGB tuple
        return ansi_code
    
    if isinstance(ansi_code, str):
        # Parse ANSI escape sequence: \033[38;2;R;G;Bm or \x1b[38;2;R;G;Bm
        if '\033[38;2;' in ansi_code or '\x1b[38;2;' in ansi_code:
            # Remove escape prefix and trailing 'm'
            rgb_part = ansi_code.replace('\033[38;2;', '').replace('\x1b[38;2;', '').replace('m', '')
            try:
                r, g, b = map(int, rgb_part.split(';'))
                return (r, g, b)
            except:
                pass
    
    # Fallback to a default color
    return (139, 0, 255)  # Electric Violet as default


# ============================================================================
# REST OF YOUR TERMINAL.PY CODE CONTINUES HERE...
# ============================================================================
# (Keep all your existing code below this point)

# ============================================================================
# EMOJI MANAGER
# ============================================================================

# Emoji border configuration
EMOJI_BORDER_THICKNESS = 4

# Braille Patterns block - excluded from emoji detection for particle rendering
BRAILLE_RANGE = (0x2800, 0x28FF)

class EmojiManager:
    """Manages pre-loaded and pre-processed emoji bitmaps for maximum performance"""
    
    # Pre-computed set of emoji codepoints for O(1) detection
    EMOJI_CODEPOINTS = set()
    
    # Unicode ranges for emoji detection
    EMOJI_RANGES = [
        (0x1F300, 0x1F5FF),  # Miscellaneous Symbols and Pictographs
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F680, 0x1F6FF),  # Transport and Map Symbols
        (0x1F700, 0x1F77F),  # Alchemical Symbols
        (0x1F780, 0x1F7FF),  # Geometric Shapes Extended
        (0x1F800, 0x1F8FF),  # Supplemental Arrows-C
        (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
        (0x1FA00, 0x1FA6F),  # Chess Symbols
        (0x1FA70, 0x1FAFF),  # Symbols and Pictographs Extended-A
        (0x2600, 0x26FF),    # Miscellaneous Symbols
        (0x2700, 0x27BF),    # Dingbats
        (0x2300, 0x23FF),    # Miscellaneous Technical
        (0xFE00, 0xFE0F),    # Variation Selectors
        (0x1F1E6, 0x1F1FF),  # Regional Indicator Symbols (flags)
    ]
    
    def __init__(self, emoji_dir: Optional[Path] = None):
        """Initialize the emoji manager and pre-load all emojis"""
        self.emoji_dir = emoji_dir or Path(__file__).parent / 'fonts' / 'emoji' / '16x16'
        self.emoji_cache: Dict[str, np.ndarray] = {}
        self.load_count = 0
        self.failed_count = 0
        
        # Build emoji codepoint set for O(1) detection
        if not EmojiManager.EMOJI_CODEPOINTS:
            self._build_emoji_set()
        
        # Pre-load all available emojis
        self._preload_emojis()
        
        logger.info(f"EmojiManager: {self.load_count} emojis pre-loaded, {self.failed_count} failed")
    
    @classmethod
    def _build_emoji_set(cls):
        """Build a set of all emoji codepoints for O(1) detection"""
        for start, end in cls.EMOJI_RANGES:
            for codepoint in range(start, end + 1):
                cls.EMOJI_CODEPOINTS.add(codepoint)
        logger.info(f"Built emoji detection set with {len(cls.EMOJI_CODEPOINTS)} codepoints")
    
    def is_emoji(self, char: str) -> bool:
        """Fast O(1) emoji detection using pre-built set, excluding Braille patterns"""
        if not char:
            return False
        
        codepoint = ord(char[0])
        
        # Exclude Braille Patterns block used for particle rendering
        if BRAILLE_RANGE[0] <= codepoint <= BRAILLE_RANGE[1]:
            return False
        
        return codepoint in EmojiManager.EMOJI_CODEPOINTS
    
    def _preload_emojis(self):
        """Pre-load and pre-process all emoji images at startup"""
        # First try Termux home fonts directory
        termux_emoji_dir = Path('/data/data/com.termux/files/home/fonts/emoji/16x16')
        if termux_emoji_dir.exists():
            self.emoji_dir = termux_emoji_dir
            logger.info(f"Using Termux home emoji directory: {self.emoji_dir}")
        elif not self.emoji_dir.exists():
            logger.warning(f"Emoji directory not found: {self.emoji_dir}")
            return
        
        # Load all PNG files in the emoji directory
        emoji_files = list(self.emoji_dir.glob('*.png'))
        logger.info(f"Pre-loading {len(emoji_files)} emoji files from {self.emoji_dir}...")
        
        for emoji_path in emoji_files:
            try:
                # Extract codepoint from filename
                codepoint_hex = emoji_path.stem
                codepoint = int(codepoint_hex, 16)
                char = chr(codepoint)
                
                # Load and pre-process the emoji once
                emoji_array = self._load_and_process_emoji(emoji_path)
                if emoji_array is not None:
                    self.emoji_cache[char] = emoji_array
                    self.load_count += 1
                else:
                    self.failed_count += 1
                    
            except (ValueError, OverflowError) as e:
                logger.debug(f"Skipping {emoji_path.name}: {e}")
                self.failed_count += 1
                continue
    
    def _load_and_process_emoji(self, emoji_path: Path) -> Optional[np.ndarray]:
        """Load and pre-process a single emoji image with all enhancements"""
        try:
            # Load the image
            img = Image.open(emoji_path)
            
            # Apply enhancement once during loading
            # Boost contrast significantly
            contrast = ImageEnhance.Contrast(img)
            img = contrast.enhance(3.0)
            
            # Boost color saturation
            color = ImageEnhance.Color(img)
            img = color.enhance(2.0)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Handle dimension normalization efficiently
            if img.size != (16, 16):
                width, height = img.size
                
                # Common cases - pad rather than resize
                if width == 16 and height == 8:
                    # Pad vertically
                    padded = Image.new('RGB', (16, 16), (15, 15, 35))  # Void Black
                    padded.paste(img, (0, 4))  # Center vertically
                    img = padded
                elif width == 8 and height == 16:
                    # Pad horizontally
                    padded = Image.new('RGB', (16, 16), (15, 15, 35))
                    padded.paste(img, (4, 0))  # Center horizontally
                    img = padded
                elif width < 16 and height < 16:
                    # Center smaller emoji
                    padded = Image.new('RGB', (16, 16), (15, 15, 35))
                    y_offset = (16 - height) // 2
                    x_offset = (16 - width) // 2
                    padded.paste(img, (x_offset, y_offset))
                    img = padded
                else:
                    # Resize as last resort
                    img = img.resize((16, 16), Image.LANCZOS)
            
            # Convert to numpy array
            return np.array(img, dtype=np.uint8)
            
        except Exception as e:
            logger.debug(f"Failed to load emoji from {emoji_path}: {e}")
            return None
    
    def get_emoji_bitmap(self, char: str, outline_color: Optional[Tuple[int, int, int]] = None) -> Optional[np.ndarray]:
        """Get pre-processed emoji bitmap with optional outline - VERY FAST"""
        if char not in self.emoji_cache:
            # Try loading it on-demand if not pre-loaded
            codepoint_hex = format(ord(char[0]), 'x').lower()
            emoji_path = self.emoji_dir / f'{codepoint_hex}.png'
            if emoji_path.exists():
                emoji_array = self._load_and_process_emoji(emoji_path)
                if emoji_array is not None:
                    self.emoji_cache[char] = emoji_array
                    self.load_count += 1
                else:
                    return None
            else:
                return None
        
        # Return a copy to avoid modifying the cached version
        emoji = self.emoji_cache[char].copy()
        
        # Add simple box outline if requested
        if outline_color:
            emoji[0, :] = outline_color    # Top edge
            emoji[-1, :] = outline_color   # Bottom edge
            emoji[:, 0] = outline_color    # Left edge
            emoji[:, -1] = outline_color   # Right edge
        
        return emoji

# ============================================================================
# MARKDOWN PROCESSING SECTION
# ============================================================================

# Markdown color definitions using PNGN_32_COLORS
MARKDOWN_COLORS = {
    'header1': PNGN_32_COLORS[24]['rgb'],  # White Flash (255, 255, 255)
    'header2': PNGN_32_COLORS[26]['rgb'],  # Plasma Silver (200, 195, 220)
    'header3': PNGN_32_COLORS[26]['rgb'],  # Plasma Silver (200, 195, 220)
    'header4': PNGN_32_COLORS[26]['rgb'],  # Plasma Silver (200, 195, 220)
    'header5': PNGN_32_COLORS[26]['rgb'],  # Plasma Silver (200, 195, 220)
    'header6': PNGN_32_COLORS[26]['rgb'],  # Plasma Silver (200, 195, 220)
    'code': PNGN_32_COLORS[26]['rgb'],     # Plasma Silver (200, 195, 220)
    'link': PNGN_32_COLORS[26]['rgb'],     # Plasma Silver (200, 195, 220)
    'bold': PNGN_32_COLORS[29]['rgb'],     # Digital Cyan (0, 212, 255)
    'italic': PNGN_32_COLORS[29]['rgb'],   # Digital Cyan (0, 212, 255)
}

class MarkdownType(Enum):
    HEADER1 = "header1"
    HEADER2 = "header2"
    HEADER3 = "header3"
    HEADER4 = "header4"
    HEADER5 = "header5"
    HEADER6 = "header6"
    CODE_INLINE = "code_inline"
    CODE_BLOCK = "code_block"
    LINK = "link"
    BOLD = "bold"
    ITALIC = "italic"
    TEXT = "text"
    TABLE = "table"

@dataclass
class MarkdownRegion:
    markdown_type: MarkdownType
    content: str
    line_num: int
    start_col: int
    end_col: int
    fg_color: Tuple[int, int, int]
    bold: bool = False
    italic: bool = False
    underline: bool = False

class MarkdownPattern:
    PATTERN_STRINGS = {
        # Headers
        'header1': r'^(?:.*?)?#\s+(.+?)(?:\s*#*)?$',
        'header2': r'^(?:.*?)?##\s+(.+?)(?:\s*#*)?$',
        'header3': r'^(?:.*?)?###\s+(.+?)(?:\s*#*)?$',
        'header4': r'^(?:.*?)?####\s+(.+?)(?:\s*#*)?$',
        'header5': r'^(?:.*?)?#####\s+(.+?)(?:\s*#*)?$',
        'header6': r'^(?:.*?)?######\s+(.+?)(?:\s*#*)?$',
        'code_block': r'^```[\s\S]*?```$',
        'code_inline': r'`([^`]+)`',
        'link': r'\[([^\]]+)\]\(([^\)]+)\)',
        # Bold and italic with proper precedence handling
        'bold': r'\*\*([^*]+?)\*\*',
        'italic': r'(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)',
    }
    
    PATTERNS = {}

def _compile_markdown_patterns():
    for name, pattern in MarkdownPattern.PATTERN_STRINGS.items():
        try:
            MarkdownPattern.PATTERNS[name] = re.compile(pattern, re.MULTILINE)
        except re.error as e:
            logger.error(f"Failed to compile markdown pattern '{name}': {e}")
            MarkdownPattern.PATTERNS[name] = None

# Compile patterns at module load time
_compile_markdown_patterns()

# ============================================================================
# CACHE SYSTEM
# ============================================================================

# Cache imports - using UnifiedCacheSystem from buffer.py
try:
    from buffer import UnifiedCacheSystem, get_unified_cache
    CACHE_AVAILABLE = True
    logger.info("UnifiedCacheSystem available from buffer.py")
except ImportError:
    CACHE_AVAILABLE = False
    UnifiedCacheSystem = None
    get_unified_cache = None
    logger.warning("UnifiedCacheSystem not available from buffer.py - running without cache")

# ============================================================================
# ANSI CODES AND UTILITIES
# ============================================================================

class ANSI:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    STRIKE = "\033[9m"

def sanitize_terminal_input(text: str) -> str:
    """Remove dangerous control sequences from external user input."""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', text)
    
    # Remove control characters except \n, \r, \t
    cleaned = []
    for char in text:
        if ord(char) >= 32 or char in '\n\r\t':
            cleaned.append(char)
    
    return ''.join(cleaned)

@dataclass
class RenderInstruction:
    """Single character to render with complete styling context"""
    char: str
    x: int
    y: int
    fg_color: Tuple[int, int, int] = (255, 255, 255)
    bg_color: Tuple[int, int, int] = (0, 0, 0)
    effects: Set[str] = field(default_factory=set)
    bold: bool = False
    italic: bool = False
    underline: bool = False
    markdown_type: Optional[MarkdownType] = None

class TextLayoutEngine:
    """
    Simplified text layout engine with persistent cache integration.
    Handles character bitmap retrieval through UnifiedCacheSystem and delegates
    emoji rendering to EmojiManager while supporting font variant selection.
    """
    
    def __init__(self, font: ImageFont.ImageFont, font_path: Optional[str] = None):
        """
        Initialize layout engine with font reference.
        
        Args:
            font: Base font for rendering
            font_path: Path to font file (unused, kept for interface compatibility)
        """
        self.font = font
        self.font_path = font_path
        
        # Font variants and emoji manager will be set by parent renderer
        self.font_variants = {}
        self.emoji_manager = None
        
        # Cache reference will be set by parent renderer
        self.cache = None
        
        # Character dimensions
        self.char_width = 8
        self.char_height = 16
        
        logger.info("TextLayoutEngine initialized for vectorized rendering")
    
    def set_emoji_manager(self, emoji_manager):
        """Set emoji manager reference after initialization"""
        self.emoji_manager = emoji_manager
    
    def set_cache(self, cache):
        """Set cache reference after initialization"""
        self.cache = cache
    
    def set_font_variants(self, variants: Dict[str, ImageFont.ImageFont]):
        """Set font variant dictionary after initialization"""
        self.font_variants = variants
    
    def get_character_bitmap(self, char: str, fg_color: Tuple[int, int, int],
                            bg_color: Tuple[int, int, int], style_flags: int) -> np.ndarray:
        """
        Get character bitmap through persistent cache or render on demand.
        
        Args:
            char: Character to render
            fg_color: Foreground RGB color tuple
            bg_color: Background RGB color tuple
            style_flags: Binary flags (1=bold, 2=italic, 4=underline, 8=strikethrough)
        
        Returns:
            NumPy array of shape (16, 8, 3) containing RGB pixel data
        """
        # Check persistent cache
        char_code = ord(char[0]) if char else 32
        if self.cache:
            cache_key = f"char_{char_code}_{fg_color[0]}_{fg_color[1]}_{fg_color[2]}_{bg_color[0]}_{bg_color[1]}_{bg_color[2]}_{style_flags}"
            cached_frame = self.cache.get_frame(cache_key)
            
            if cached_frame is not None:
                return np.array(cached_frame, dtype=np.uint8)
        
        # Cache miss - render the character
        bitmap = self._render_character(char, fg_color, bg_color, style_flags)
        
        # Store in persistent cache
        if self.cache:
            bitmap_image = Image.fromarray(bitmap.astype(np.uint8))
            cache_key = f"char_{char_code}_{fg_color[0]}_{fg_color[1]}_{fg_color[2]}_{bg_color[0]}_{bg_color[1]}_{bg_color[2]}_{style_flags}"
            self.cache.put_frame(cache_key, bitmap_image)
        
        return bitmap
    
    def _render_character(self, char: str, fg_color: Tuple[int, int, int],
                         bg_color: Tuple[int, int, int], style_flags: int) -> np.ndarray:
        """Render character bitmap using PIL or emoji system"""
        
        # Check if this is an emoji and we have an emoji manager
        char_code = ord(char[0]) if char else 32
        if self.emoji_manager and char_code in EmojiManager.EMOJI_CODEPOINTS:
            # Exclude Braille patterns which are handled as text
            if not (BRAILLE_RANGE[0] <= char_code <= BRAILLE_RANGE[1]):
                # Pass foreground color as outline for visual consistency with text
                emoji_bitmap = self.emoji_manager.get_emoji_bitmap(char, outline_color=fg_color)
                if emoji_bitmap is not None:
                    return emoji_bitmap
        
        # Regular text rendering
        img = Image.new('RGB', (self.char_width, self.char_height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Select font based on style flags
        font_to_use = self.font
        if self.font_variants:
            if (style_flags & 1) and (style_flags & 2):
                font_to_use = self.font_variants.get('bold_italic', self.font)
            elif style_flags & 1:
                font_to_use = self.font_variants.get('bold', self.font)
            elif style_flags & 2:
                font_to_use = self.font_variants.get('italic', self.font)
        
        draw.text((0, 0), char, font=font_to_use, fill=fg_color)
        
        # Apply underline decoration
        if style_flags & 4:
            y_pos = self.char_height - 2
            underline_color = RGBColors.brighten(fg_color, 1.3)
            draw.line([(0, y_pos), (self.char_width - 1, y_pos)], fill=underline_color, width=1)
        
        # Apply strikethrough decoration
        if style_flags & 8:
            y_pos = self.char_height // 2
            draw.line([(0, y_pos), (self.char_width - 1, y_pos)], fill=fg_color, width=1)
        
        return np.array(img, dtype=np.uint8)

class DeterministicPureTerminalRenderer:
    """
    Pure text-to-image renderer with deterministic effect support.
    Complete functionality of original pure renderer.
    Enhanced with PNGN color system, all colors from PNGN_32_COLORS.
    Modified: No inner border, configurable outer glow border.
    Now uses UnifiedCacheSystem from buffer.py for caching.
    OPTIMIZED: Pre-loaded emojis, O(1) detection, simplified caching.
    """
    
    def __init__(self, terminal_width: int = TERMINAL_WIDTH, height: int = TERMINAL_HEIGHT):
        self.terminal_width = terminal_width
        self.height = height
        
        # Always render at desktop resolution
        self.char_width = STANDARD_CHAR_WIDTH  # 8
        self.char_height = STANDARD_CHAR_HEIGHT  # 16
        
        # Border for outer glow effect (configurable)
        self.outer_border_thickness = 0  # Set to non-zero to enable glow border
        
        # Image dimensions (NO inner border padding)
        self.inner_width = terminal_width * self.char_width
        self.inner_height = height * self.char_height
        # Final image size includes outer border
        self.img_width = self.inner_width + (self.outer_border_thickness * 2)
        self.img_height = self.inner_height + (self.outer_border_thickness * 2)
        
        # Default background (will be overridden by team colors during render)
        self.default_bg = PNGN_32_COLORS[7]['rgb']  # Void Black as fallback
        self.buffer = np.full((self.img_height, self.img_width, 3), self.default_bg, dtype=np.uint8)
        
        # ============================================================================
        # HYBRID OPTIMIZATION: Initialize vectorized components if available
        # ============================================================================
        
        # Try to use vectorized character metrics
        if VECTOR_MODULE_AVAILABLE:
            try:
                self.vector_metrics = VectorizedCharacterMetrics(cache_size=20000)
                self.dirty_tracker = VectorizedDirtyTracker(
                    self.inner_width, self.inner_height,
                    block_size=self.char_width * 2  # Track 2x2 character blocks
                )
                logger.info("Vectorized operations initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize vectorized components: {e}")
                self.vector_metrics = None
                self.dirty_tracker = None
        else:
            self.vector_metrics = None
            self.dirty_tracker = None
        
        # Use UnifiedCacheSystem from buffer.py - global instance only
        if CACHE_AVAILABLE:
            try:
                self.cache = get_unified_cache()
                logger.info("Connected to global UnifiedCacheSystem from buffer.py")
            except Exception as e:
                logger.warning(f"Failed to retrieve global cache: {e}")
                self.cache = None
        else:
            self.cache = None
            logger.warning("Running without cache - UnifiedCacheSystem not available")
        
        # Initialize EmojiManager for optimized emoji handling
        self.emoji_manager = EmojiManager()
        logger.info(f"EmojiManager initialized with {len(self.emoji_manager.emoji_cache)} emojis")
        
        # Font loading with path tracking
        font_result = self._load_font_with_path()
        self.font = font_result['font']
        self.font_path = font_result['path']
        
        # Load font variants for markdown styling
        self._load_font_variants()
        
        # Initialize layout engine with loaded font and path
        if self.font_path:
            try:
                self.layout_engine = TextLayoutEngine(self.font, self.font_path)
                self.layout_engine.set_cache(self.cache)
                self.layout_engine.set_font_variants(self.font_variants)
                self.layout_engine.set_emoji_manager(self.emoji_manager)
                logger.info("TextLayoutEngine initialized for vectorized rendering")
            except Exception as e:
                logger.warning(f"Failed to initialize TextLayoutEngine: {e}")
                self.layout_engine = None
        else:
            self.layout_engine = None
        
        # Performance tracking
        self.render_times = []
        self.shutting_down = False
        
        # Memory pressure callback
        self.memory_pressure_callback = None
        self.reduce_quality = False
        
        # For deterministic effects
        self._content_hash = ""
        self._frame_seed = 0
        
        # Access to BuildingBlocks components if available
        self.building_blocks = BuildingBlocks if BUILDING_BLOCKS_AVAILABLE else None
        
        # Markdown stats
        self.markdown_stats = defaultdict(int)
        
        # Track whether we're using optimizations
        self.using_vectorization = (self.vector_metrics is not None)
        self.using_layout_engine = (self.layout_engine is not None)
    
    def _load_font_with_path(self) -> Dict[str, Any]:
        """Load monospace font and return both font object and path"""
        try:
            # Priority 1: Local fonts directory (package fonts) - CHECK FIRST!
            local_fonts_dir = Path(__file__).parent / 'fonts'
            dejavu_local = local_fonts_dir / 'DejaVuSansMono.ttf'
            if dejavu_local.exists():
                font = ImageFont.truetype(str(dejavu_local), 14)
                logger.info("Loaded DejaVu Sans Mono from local fonts")
                return {'font': font, 'path': str(dejavu_local)}
            
            unifont_local = local_fonts_dir / 'unifont.ttf'
            if unifont_local.exists():
                font = ImageFont.truetype(str(unifont_local), 16)
                logger.info("Loaded GNU Unifont from local fonts")
                return {'font': font, 'path': str(unifont_local)}
            
            # Priority 2: Termux home fonts directory
            fonts_dir = Path('/data/data/com.termux/files/home/fonts')
            dejavu_path = fonts_dir / 'DejaVuSansMono.ttf'
            if dejavu_path.exists():
                font = ImageFont.truetype(str(dejavu_path), 14)
                logger.info("Loaded DejaVu Sans Mono from Termux home")
                return {'font': font, 'path': str(dejavu_path)}
            
            # Priority 3: Common Linux system paths
            linux_font_paths = [
                Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'),
                Path('/usr/share/fonts/dejavu/DejaVuSansMono.ttf'),
                Path('/usr/share/fonts/TTF/DejaVuSansMono.ttf'),
            ]
            
            for font_path in linux_font_paths:
                if font_path.exists():
                    font = ImageFont.truetype(str(font_path), 14)
                    logger.info(f"Loaded font from system: {font_path}")
                    return {'font': font, 'path': str(font_path)}
            
            # Priority 4: Termux system (last resort)
            termux_font_dir = Path('/data/data/com.termux/files/usr/share/fonts/TTF')
            dejavu_system = termux_font_dir / 'DejaVuSansMono.ttf'
            if dejavu_system.exists():
                font = ImageFont.truetype(str(dejavu_system), 14)
                logger.info("Loaded DejaVu Sans Mono from Termux system")
                return {'font': font, 'path': str(dejavu_system)}
            
            # Ultimate fallback
            logger.error("No suitable font found - using default")
            return {'font': ImageFont.load_default(), 'path': None}
        
        except Exception as e:
            logger.error(f"Font loading failed: {e}")
            return {'font': ImageFont.load_default(), 'path': None}
    
    def _load_font_variants(self):
        """Load additional font variants for markdown styling"""
        self.font_variants = {}
        
        if not self.font_path:
            return
        
        # Derive variant paths from base font path
        base_path = Path(self.font_path)
        font_dir = base_path.parent
        
        variants = {
            'bold': 'DejaVuSansMono-Bold.ttf',
            'italic': 'DejaVuSansMono-Oblique.ttf',
            'bold_italic': 'DejaVuSansMono-BoldOblique.ttf',
        }
        
        for name, filename in variants.items():
            variant_path = font_dir / filename
            if variant_path.exists():
                try:
                    self.font_variants[name] = ImageFont.truetype(str(variant_path), 14)
                except Exception as e:
                    logger.debug(f"Could not load {name} variant: {e}")
    
    def _create_content_hash(self, content: str) -> str:
        """Create deterministic hash from content for seeding effects"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_pseudo_random(self, x: int, y: int, frame: int, salt: str = "") -> float:
        """Generate deterministic pseudo-random value between 0 and 1"""
        hash_int = int(self._content_hash[:8], 16) if self._content_hash else 0
        salt_hash = hash(salt) if salt else 0
        value = ((x * 31 + y * 37 + frame * 13 + hash_int + salt_hash) % 10000) / 10000.0
        return value
    
    def _deterministic_choice(self, choices: list, x: int, y: int, frame: int, salt: str = ""):
        """Deterministically choose from a list"""
        index = int(self._get_pseudo_random(x, y, frame, salt) * len(choices))
        return choices[min(index, len(choices) - 1)]
    
    def _deterministic_int(self, min_val: int, max_val: int, x: int, y: int, frame: int, salt: str = "") -> int:
        """Deterministically generate an integer in range"""
        pseudo_random = self._get_pseudo_random(x, y, frame, salt)
        return min_val + int(pseudo_random * (max_val - min_val + 1))
    
    def set_memory_pressure_callback(self, callback):
        """Set callback for memory pressure notifications"""
        self.memory_pressure_callback = callback
    
    def handle_memory_pressure(self, pressure_level: float):
        """Handle memory pressure by reducing quality"""
        if pressure_level > 0.8:
            logger.warning(f"High memory pressure: {pressure_level}")
            self.reduce_quality = True
            if self.cache:
                pressure_str = 'CRITICAL' if pressure_level > 0.9 else 'HIGH'
                self.cache.reduce_cache(pressure_str)
        elif pressure_level > 0.5:
            if self.cache:
                self.cache.reduce_cache('MODERATE')
        elif pressure_level < 0.5:
            self.reduce_quality = False
        
        if self.memory_pressure_callback:
            self.memory_pressure_callback(pressure_level)
    
    async def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down terminal renderer...")
        self.shutting_down = True
        
        # Clear resources without shutting down global cache
        self.cache = None
        self.buffer = None
        self.font = None
        self.emoji_manager = None
        
        logger.info("Terminal renderer shutdown complete")

# ============================================================================
    # COLOR ANIMATION AND FRAME VARIATION METHODS
    # ============================================================================
    
    def _get_frame_variations(self, frame: int) -> Dict[str, Any]:
        """
        Get frame variations with single pseudo-random brightness trigger.
        
        Generates one five-frame brightness pulse per animation at a position
        determined by content hash. The pulse timing appears random but remains
        deterministic, ensuring reproducible behavior while creating visual variety.
        
        At 12 FPS with 60-frame animations, the brightness event occurs once per
        five-second cycle at a pseudo-random moment, with the majority of frames
        maintaining stable neutral intensity.
        
        Args:
            frame: Current frame number in animation sequence
            
        Returns:
            Dictionary containing active animation parameters
        """
        # Calculate deterministic trigger window position from content hash
        if hasattr(self, '_content_hash') and self._content_hash:
            hash_int = int(self._content_hash[:8], 16)
            # Position trigger within valid range allowing 5-frame pulse to complete
            trigger_start = (hash_int % 56)  # Range: frame 0-55
        else:
            trigger_start = 20  # Default fallback position
        
        # Check if current frame falls within the five-frame trigger window
        in_trigger_window = trigger_start <= frame < (trigger_start + 5)
        
        if in_trigger_window:
            # Calculate local position within the active window
            local_frame = frame - trigger_start
            
            # Complete sine cycle across 5 frames for sharp brightness pulse
            # Frequency: 2*pi/5 ≈ 1.2566 radians per frame
            glow_intensity = 1.0 + 0.15 * math.sin(local_frame * 1.2566)
        else:
            # Outside trigger window: neutral intensity for stable colors
            glow_intensity = 1.0
        
        return {
            'cursor': '█' if (frame // CURSOR_BLINK_FRAMES) % 2 == 0 else '_',
            'glow_intensity': glow_intensity,
            'corruption_pattern': frame % 5,
        }
    
    def _apply_glow_with_dimming(self, fg: Tuple[int, int, int], variations: Dict[str, Any]) -> Tuple[int, int, int]:
        """
        Apply glow effect with palette-aware brightening and dimming.
        
        Translates glow intensity values into discrete palette color steps using
        threshold-based evaluation. The RGBColors.brighten and RGBColors.dim methods
        move exactly one palette position up or down through the PNGN_32_COLORS palette,
        with double application at extreme thresholds producing two-step movements.
        
        The neutral zone between 0.90 and 1.10 applies no modification, ensuring colors
        remain stable when glow intensity hovers near the baseline value of one.
        
        Args:
            fg: Foreground color RGB tuple from palette
            variations: Dictionary containing glow_intensity parameter
            
        Returns:
            Modified palette color with brightness adjustment applied
        """
        glow = variations.get('glow_intensity', 1.0)
        
        if glow > 1.09:
            if glow > 1.13:
                # Very high glow - move two palette steps brighter
                fg = RGBColors.brighten(RGBColors.brighten(fg, glow), glow)
            else:
                # Medium-high glow - move one palette step brighter
                fg = RGBColors.brighten(fg, glow)
        elif glow < 0.91:
            if glow < 0.87:
                # Very low glow - move two palette steps dimmer
                fg = RGBColors.dim(RGBColors.dim(fg))
            else:
                # Medium-low glow - move one palette step dimmer
                fg = RGBColors.dim(fg)
        
        return fg

# ============================================================================
    # EFFECT APPLICATION METHODS
    # ============================================================================
    
    def _apply_effects(self, instructions: List[RenderInstruction], personality: Optional[str],
                      effects: List[str], frame: int, context: Optional[Dict[str, Any]],
                      rgb_colors: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]]) -> str:
        """
        Apply character-based visual effects with content-based deterministic triggering.
        
        Builds character grid from instructions, applies requested effects as sparse
        deterministic events based on content hash, then converts to ANSI for rendering.
        Only traditional terminal effects supported: corruption, glitch, static.
        """
        # Build grid for effect processing
        grid = [[' ' for _ in range(self.terminal_width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.terminal_width)] for _ in range(self.height)]
        
        # Place instructions in grid
        for inst in instructions:
            if 0 <= inst.x < self.terminal_width and 0 <= inst.y < self.height:
                grid[inst.y][inst.x] = inst.char
                styles[inst.y][inst.x] = inst
        
        # Use variations from context if provided, otherwise generate
        if context and isinstance(context, dict):
            variations = context
        else:
            variations = self._get_frame_variations(frame)
        
        # Reduce effects if memory pressure active
        if self.reduce_quality and len(effects) > 2:
            effects = effects[:2]
        
        # Determine which effects should activate based on content hash
        active_effects = self._select_active_effects(effects)
        
        # Apply only the selected effects as brief events
        if 'corruption' in active_effects:
            self._apply_corruption(grid, styles, frame, variations, personality)
        if 'glitch' in active_effects:
            self._apply_glitch(grid, styles, frame, personality)
        if 'static' in active_effects:
            self._apply_static(grid, styles, frame)
        
        # Convert grid to ANSI
        return self._grid_to_ansi(grid, styles, variations)
    
    def _select_active_effects(self, requested_effects: List[str]) -> Set[str]:
        """
        Deterministically select which effects should activate based on content hash.
        
        Each effect receives independent activation probability around fifty percent,
        creating varied combinations where some content shows multiple effects, others
        show single effects, and some show no effects at all.
        
        Args:
            requested_effects: List of effect names requested by caller
            
        Returns:
            Set of effect names that should activate for this render
        """
        if not hasattr(self, '_content_hash') or not self._content_hash:
            return set()
        
        active = set()
        hash_int = int(self._content_hash[:8], 16)
        
        # Each effect gets independent activation check using different hash bits
        if 'corruption' in requested_effects:
            if (hash_int & 0xFF) > 128:  # Approximately 50% activation rate
                active.add('corruption')
        
        if 'glitch' in requested_effects:
            if ((hash_int >> 8) & 0xFF) > 128:
                active.add('glitch')
        
        if 'static' in requested_effects:
            if ((hash_int >> 16) & 0xFF) > 128:
                active.add('static')
        
        return active
    
    def _apply_corruption(self, grid: List[List[str]], styles: List[List[Optional[RenderInstruction]]], 
                         frame: int, variations: Dict[str, Any], personality: Optional[str] = None):
        """
        Apply corruption effect as brief burst event during deterministic window.
        
        Replaces characters with glitched alternatives and applies team-specific
        corruption colors during a five-frame window positioned by content hash.
        """
        # Calculate corruption event window from content hash
        if hasattr(self, '_content_hash') and self._content_hash:
            hash_int = int(self._content_hash[:8], 16)
            corruption_start = ((hash_int >> 16) % 56)
        else:
            corruption_start = 30
        
        # Only apply corruption during the event window
        in_corruption_window = corruption_start <= frame < (corruption_start + 5)
        
        if not in_corruption_window:
            return
        
        pattern = variations.get('corruption_pattern', 0)
        corruption_color = self._team_colors['corruption'] if hasattr(self, '_team_colors') else PNGN_32_COLORS[21]['rgb']
        
        for y in range(self.height):
            for x in range(self.terminal_width):
                if grid[y][x] != ' ':
                    pseudo_random = self._get_pseudo_random(x, y, frame, "corruption")
                    
                    if pseudo_random < CORRUPTION_CHANCE_BASE:
                        if pattern == 0:
                            char_index = (x + y + frame) % len(CORRUPTION_CHARS)
                            grid[y][x] = CORRUPTION_CHARS[char_index]
                        elif pattern == 1:
                            grid[y][x] = '?' if pseudo_random < (CORRUPTION_CHANCE_BASE / 2) else '!'
                        else:
                            chars = ['▓', '▒', '░']
                            char_index = (x * y + frame) % len(chars)
                            grid[y][x] = chars[char_index]
                        
                        if styles[y][x]:
                            styles[y][x].fg_color = corruption_color
    
    def _apply_glitch(self, grid: List[List[str]], styles: List[List[Optional[RenderInstruction]]], 
                     frame: int, personality: Optional[str] = None):
        """
        Apply glitch effect as single-frame flash events at deterministic moments.
        
        Triggers on two specific frames calculated from content hash, replacing
        characters momentarily with glitch alternatives and vibrant team colors.
        """
        # Calculate glitch trigger frames from content hash
        if hasattr(self, '_content_hash') and self._content_hash:
            hash_int = int(self._content_hash[:8], 16)
            glitch_frame_1 = (hash_int >> 8) % 60
            glitch_frame_2 = ((hash_int >> 16) % 50) + 10
        else:
            glitch_frame_1 = 15
            glitch_frame_2 = 45
        
        # Only trigger on specific frames
        if frame not in [glitch_frame_1, glitch_frame_2]:
            return
        
        glitch_colors = self._team_colors['glitch'] if hasattr(self, '_team_colors') else [PNGN_32_COLORS[31]['rgb']]
        
        # Deterministic character replacement with color
        num_glitches = self._deterministic_int(3, 8, 3, 0, frame, "glitch_count")
        for i in range(num_glitches):
            x = self._deterministic_int(0, self.terminal_width - 1, i, 0, frame, f"glitch_x_{i}")
            y = self._deterministic_int(0, self.height - 1, 0, i, frame, f"glitch_y_{i}")
            if grid[y][x] != ' ':
                char_index = (x + y + frame) % len(GLITCH_CHARS)
                grid[y][x] = GLITCH_CHARS[char_index]
                if styles[y][x]:
                    color_index = (x + y + frame) % len(glitch_colors)
                    styles[y][x].fg_color = glitch_colors[color_index]
    
    def _apply_static(self, grid: List[List[str]], styles: List[List[Optional[RenderInstruction]]], frame: int):
        """
        Apply static noise effect as pulsing intensity window.
        
        Creates brief period of visible static noise with sine-modulated intensity
        during an eight-frame window positioned by content hash.
        """
        # Calculate static intensity window from content hash
        if hasattr(self, '_content_hash') and self._content_hash:
            hash_int = int(self._content_hash[:8], 16)
            static_start = ((hash_int >> 24) % 50) + 5
        else:
            static_start = 25
        
        # Apply static only during intensity window
        in_static_window = static_start <= frame < (static_start + 8)
        
        if not in_static_window:
            return
        
        # Variable intensity within window peaks at center
        local_frame = frame - static_start
        intensity = 0.05 + 0.10 * math.sin(local_frame * 0.785)
        
        static_colors = [
            PNGN_32_COLORS[26]['rgb'],
            PNGN_32_COLORS[24]['rgb'],
            self.default_bg,
        ]
        
        for y in range(self.height):
            for x in range(self.terminal_width):
                pseudo_random = self._get_pseudo_random(x, y, frame, "static")
                if pseudo_random < intensity:
                    char_index = (x * y + frame) % len(STATIC_CHARS)
                    grid[y][x] = STATIC_CHARS[char_index]
                    color_index = (x + y + frame) % len(static_colors)
                    if styles[y][x] is None:
                        styles[y][x] = RenderInstruction(
                            char=grid[y][x],
                            x=x,
                            y=y,
                            fg_color=static_colors[color_index]
                        )
                    else:
                        styles[y][x].fg_color = static_colors[color_index]

# ============================================================================
    # MARKDOWN DETECTION
    # ============================================================================
    
    def _detect_markdown_in_text(self, text: str) -> List[MarkdownRegion]:
        """Detect markdown elements in text using regex patterns"""
        return self._detect_markdown_with_regex(text)
    
    def _detect_markdown_with_regex(self, text: str) -> List[MarkdownRegion]:
        """Detect markdown elements in text and return regions with type flags only"""
        regions = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            # Keep track of what's already been processed
            processed_ranges = set()
            
            # Headers - check from longest to shortest pattern to avoid overlap
            header_matched = False
            for level in range(6, 0, -1):
                header_key = f'header{level}'
                pattern = MarkdownPattern.PATTERNS.get(header_key)
                if pattern and pattern.match(line):
                    match = pattern.match(line)
                    if match:
                        # Apply styling based on header level
                        if level == 1:
                            bold = True
                            underline = True
                        elif level == 2:
                            bold = True
                            underline = True
                        else:  # levels 3-6
                            bold = True
                            underline = False
                        
                        regions.append(MarkdownRegion(
                            markdown_type=MarkdownType[header_key.upper()],
                            content=line,
                            line_num=line_num,
                            start_col=1,
                            end_col=len(line) - 1,
                            fg_color=(0, 0, 0),  # Placeholder, not used
                            bold=bold,
                            underline=underline
                        ))
                        header_matched = True
                        # Mark entire line as processed for headers
                        processed_ranges.update(range(0, len(line)))
                        break
            
            # Check for table rows (simple pipe detection)
            if '|' in line and line.strip().startswith('|'):
                regions.append(MarkdownRegion(
                    markdown_type=MarkdownType.TABLE,
                    content=line,
                    line_num=line_num,
                    start_col=0,
                    end_col=len(line),
                    fg_color=(0, 0, 0)  # Placeholder
                ))
                continue
            
            # Code blocks
            if line.startswith('```'):
                regions.append(MarkdownRegion(
                    markdown_type=MarkdownType.CODE_BLOCK,
                    content=line,
                    line_num=line_num,
                    start_col=0,
                    end_col=len(line),
                    fg_color=(0, 0, 0)  # Placeholder
                ))
                continue
            
            # Inline code - process before bold/italic to avoid conflicts
            code_pattern = MarkdownPattern.PATTERNS.get('code_inline')
            if code_pattern:
                for match in code_pattern.finditer(line):
                    if not any(match.start() >= start and match.end() <= end 
                              for start, end in [(r.start_col, r.end_col) 
                                                for r in regions if r.line_num == line_num]):
                        regions.append(MarkdownRegion(
                            markdown_type=MarkdownType.CODE_INLINE,
                            content=match.group(0),
                            line_num=line_num,
                            start_col=match.start(),
                            end_col=match.end(),
                            fg_color=(0, 0, 0)  # Placeholder
                        ))
                        processed_ranges.update(range(match.start(), match.end()))
            
            # Links
            link_pattern = MarkdownPattern.PATTERNS.get('link')
            if link_pattern:
                for match in link_pattern.finditer(line):
                    if not any(pos in processed_ranges for pos in range(match.start(), match.end())):
                        regions.append(MarkdownRegion(
                            markdown_type=MarkdownType.LINK,
                            content=match.group(1),
                            line_num=line_num,
                            start_col=match.start(),
                            end_col=match.end(),
                            fg_color=(0, 0, 0)  # Placeholder
                        ))
                        processed_ranges.update(range(match.start(), match.end()))
            
            # Process Bold BEFORE Italic (since bold uses ** and italic uses *)
            bold_pattern = MarkdownPattern.PATTERNS.get('bold')
            if bold_pattern:
                for match in bold_pattern.finditer(line):
                    if not any(pos in processed_ranges for pos in range(match.start(), match.end())):
                        regions.append(MarkdownRegion(
                            markdown_type=MarkdownType.BOLD,
                            content=match.group(1),
                            line_num=line_num,
                            start_col=match.start(),
                            end_col=match.end(),
                            fg_color=(0, 0, 0),  # Placeholder
                            bold=True
                        ))
                        # Mark this range as processed so italic doesn't pick it up
                        processed_ranges.update(range(match.start(), match.end()))
            
            # Italic - process after bold, only in unprocessed ranges
            italic_improved_pattern = r'(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)'
            try:
                italic_regex = re.compile(italic_improved_pattern)
                for match in italic_regex.finditer(line):
                    # Only process if this range hasn't been processed yet
                    if not any(pos in processed_ranges for pos in range(match.start(), match.end())):
                        regions.append(MarkdownRegion(
                            markdown_type=MarkdownType.ITALIC,
                            content=match.group(1),
                            line_num=line_num,
                            start_col=match.start(),
                            end_col=match.end(),
                            fg_color=(0, 0, 0),  # Placeholder
                            italic=True
                        ))
                        processed_ranges.update(range(match.start(), match.end()))
            except re.error:
                # Fallback to simpler pattern if the improved one fails
                italic_pattern = MarkdownPattern.PATTERNS.get('italic')
                if italic_pattern:
                    for match in italic_pattern.finditer(line):
                        # Check if this match is NOT within any processed range
                        match_range = set(range(match.start(), match.end()))
                        if not match_range.intersection(processed_ranges):
                            regions.append(MarkdownRegion(
                                markdown_type=MarkdownType.ITALIC,
                                content=match.group(1),
                                line_num=line_num,
                                start_col=match.start(),
                                end_col=match.end(),
                                fg_color=(0, 0, 0),  # Placeholder
                                italic=True
                            ))
                            processed_ranges.update(range(match.start(), match.end()))
        
        return regions
    
    def _text_to_instructions_with_markdown(self, text: str, personality: Optional[str] = None, 
                                           effects: Optional[List[str]] = None,
                                           markdown_regions: List[MarkdownRegion] = None) -> List[RenderInstruction]:
        """Convert text to render instructions with markdown type flags"""
        instructions = []
        
        # Get base team colors
        default_fg = self._team_colors['text'] if hasattr(self, '_team_colors') else PNGN_32_COLORS[31]['rgb']
        
        # Build a map of position to markdown region for quick lookup
        markdown_map = {}
        if markdown_regions:
            for region in markdown_regions:
                for col in range(region.start_col, region.end_col):
                    markdown_map[(region.line_num, col)] = region
        
        x, y = 0, 0
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            for char_idx, char in enumerate(line):
                if x >= self.terminal_width:
                    x = 0
                    y += 1
                
                if y >= self.height:
                    break
                
                # Check if this position has markdown styling
                region = markdown_map.get((line_num, char_idx))
                
                instruction = RenderInstruction(
                    char=char,
                    x=x,
                    y=y,
                    fg_color=default_fg,  # Base color, will be overridden by markdown type
                    effects=set(effects) if effects else set(),
                    bold=region.bold if region else False,
                    italic=region.italic if region else False,
                    underline=region.underline if region else False,
                    markdown_type=region.markdown_type if region else None
                )
                
                instructions.append(instruction)
                x += 1
            
            # Add newline
            x = 0
            y += 1
            
            if y >= self.height:
                break
        
        return instructions
    
    def _text_to_instructions(self, text: str, personality: Optional[str] = None, 
                             effects: Optional[List[str]] = None) -> List[RenderInstruction]:
        """Convert text to render instructions with PNGN_32_COLORS"""
        instructions = []
        
        # Get team colors based on personality
        if personality == 'PNGN':
            fg_color = PNGN_32_COLORS[2]['rgb']  # Neon Purple for text glow
        elif personality == 'KLLR':
            fg_color = PNGN_32_COLORS[8]['rgb']  # Neon Magenta for text glow
        elif personality == 'SHMA':
            fg_color = PNGN_32_COLORS[14]['rgb']  # Neon Green for text
        else:
            fg_color = PNGN_32_COLORS[31]['rgb']  # White Flash for neutral text
        
        x, y = 0, 0
        for char in text:
            if char == '\n':
                x = 0
                y += 1
                continue
            
            if x >= self.terminal_width:
                x = 0
                y += 1
            
            if y >= self.height:
                break
            
            instructions.append(RenderInstruction(
                char=char,
                x=x,
                y=y,
                fg_color=fg_color,
                effects=set(effects) if effects else set()
            ))
            x += 1
        
        return instructions
    
    # ============================================================================
    # GRID TO ANSI CONVERSION
    # ============================================================================
    
    def _grid_to_ansi(self, grid: List[List[str]], styles: List[List[Optional[RenderInstruction]]], 
                     variations: Dict[str, Any]) -> str:
        """Convert grid to ANSI string with markdown-aware coloring"""
        ansi_parts = []
        current_fg = None
        current_bg = None
        current_bold = False
        current_italic = False
        current_underline = False
        
        for y in range(self.height):
            for x in range(self.terminal_width):
                char = grid[y][x]
                style = styles[y][x]
                
                if style:
                    # ENFORCE MARKDOWN COLORS - Direct mapping from MARKDOWN_COLORS dict
                    if style.markdown_type:
                        # Map markdown type directly to predefined color
                        if style.markdown_type == MarkdownType.HEADER1:
                            fg = MARKDOWN_COLORS['header1']
                        elif style.markdown_type == MarkdownType.HEADER2:
                            fg = MARKDOWN_COLORS['header2']
                        elif style.markdown_type == MarkdownType.HEADER3:
                            fg = MARKDOWN_COLORS['header3']
                        elif style.markdown_type == MarkdownType.HEADER4:
                            fg = MARKDOWN_COLORS['header4']
                        elif style.markdown_type == MarkdownType.HEADER5:
                            fg = MARKDOWN_COLORS['header5']
                        elif style.markdown_type == MarkdownType.HEADER6:
                            fg = MARKDOWN_COLORS['header6']
                        elif style.markdown_type == MarkdownType.BOLD:
                            fg = MARKDOWN_COLORS['bold']
                        elif style.markdown_type == MarkdownType.ITALIC:
                            fg = MARKDOWN_COLORS['italic']
                        elif style.markdown_type == MarkdownType.CODE_INLINE:
                            fg = MARKDOWN_COLORS['code']
                        elif style.markdown_type == MarkdownType.CODE_BLOCK:
                            fg = MARKDOWN_COLORS['code']
                        elif style.markdown_type == MarkdownType.LINK:
                            fg = MARKDOWN_COLORS['link']
                        else:
                            fg = style.fg_color
                    else:
                        # Non-markdown text uses personality-based color
                        fg = style.fg_color
                    
                    # Apply glow intensity ONLY to non-markdown text
                    if not style.markdown_type:
                        fg = self._apply_glow_with_dimming(fg, variations)
                    
                    # Handle foreground color change
                    if fg != current_fg:
                        ansi_parts.append(f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m")
                        current_fg = fg
                    
                    # Handle bold
                    if style.bold != current_bold:
                        if style.bold:
                            ansi_parts.append("\033[1m")
                        else:
                            ansi_parts.append("\033[22m")
                        current_bold = style.bold
                    
                    # Handle italic
                    if style.italic != current_italic:
                        if style.italic:
                            ansi_parts.append("\033[3m")
                        else:
                            ansi_parts.append("\033[23m")
                        current_italic = style.italic
                    
                    # Handle underline
                    if style.underline != current_underline:
                        if style.underline:
                            ansi_parts.append("\033[4m")
                        else:
                            ansi_parts.append("\033[24m")
                        current_underline = style.underline
                
                ansi_parts.append(char)
        
        return ''.join(ansi_parts) + ANSI.RESET

# ============================================================================
    # ANSI TO IMAGE RENDERING
    # ============================================================================
    
    def _render_ansi_to_image(self, ansi_text: str, personality: Optional[str] = None, 
                             frame: int = 0) -> Image.Image:
        """Render ANSI text to image with PNGN_32_COLORS support and emoji flag handling"""
        if self.shutting_down:
            return Image.new('RGB', (self.inner_width, self.inner_height), PNGN_32_COLORS[7]['rgb'])
        
        # Get team-specific background color
        team_bg = self._team_colors['background'] if hasattr(self, '_team_colors') else PNGN_32_COLORS[7]['rgb']
        
        # Clear inner buffer with team background (NO border padding)
        inner_buffer = np.full((self.inner_height, self.inner_width, 3), team_bg, dtype=np.uint8)
        
        # Store frame for blink effect
        self.current_frame = frame
        
        # Get default colors based on personality
        default_fg = self._team_colors['text'] if hasattr(self, '_team_colors') else PNGN_32_COLORS[31]['rgb']
        
        # Parse ANSI and render
        x, y = 0, 0
        fg_color = default_fg
        bg_color = team_bg
        bold = False
        dim = False
        italic = False
        underline = False
        blink = False
        reverse = False
        strike = False
        
        # Emoji flag tracking variables
        emoji_start_pending = False
        skip_next_char = False
        
        # Enhanced ANSI color mapping to PNGN_32_COLORS
        ansi_to_palette = {
            30: PNGN_32_COLORS[7]['rgb'],   # Black -> Void Black
            31: PNGN_32_COLORS[12]['rgb'],  # Red -> Pure Red
            32: PNGN_32_COLORS[14]['rgb'],  # Green -> Neon Green
            33: PNGN_32_COLORS[16]['rgb'],  # Yellow -> Electric Yellow
            34: PNGN_32_COLORS[6]['rgb'],   # Blue -> Deep Blue
            35: PNGN_32_COLORS[8]['rgb'],   # Magenta -> Neon Magenta
            36: PNGN_32_COLORS[28]['rgb'],  # Cyan -> Digital Cyan
            37: PNGN_32_COLORS[31]['rgb'],  # White -> White Flash
        }
        
        i = 0
        while i < len(ansi_text):
            # Check if we should skip this character (emoji continuation marker)
            if skip_next_char:
                skip_next_char = False
                x += 1
                i += 1
                continue
            
            if ansi_text[i:i+2] == '\033[':
                # Parse ANSI escape
                j = i + 2
                while j < len(ansi_text) and ansi_text[j] not in 'mHJKDCBA':
                    j += 1
                
                if j < len(ansi_text):
                    codes_str = ansi_text[i+2:j]
                    cmd = ansi_text[j]
                    
                    # Check for emoji flags
                    if codes_str == '998':
                        emoji_start_pending = True
                        i = j + 1
                        continue
                    elif codes_str == '999':
                        skip_next_char = True
                        i = j + 1
                        continue
                    
                    if cmd == 'm':
                        codes = codes_str.split(';')
                        
                        for code in codes:
                            if code == '0':
                                fg_color = default_fg
                                bg_color = team_bg
                                bold = dim = italic = underline = blink = reverse = strike = False
                            elif code == '1':
                                bold = True
                            elif code == '2':
                                dim = True
                            elif code == '3':
                                italic = True
                            elif code == '4':
                                underline = True
                            elif code == '5':
                                blink = True
                            elif code == '7':
                                reverse = True
                            elif code == '9':
                                strike = True
                            elif code == '22':
                                bold = dim = False
                            elif code == '23':
                                italic = False
                            elif code == '24':
                                underline = False
                            elif code == '25':
                                blink = False
                            elif code == '27':
                                reverse = False
                            elif code == '29':
                                strike = False
                            elif code in ['30', '31', '32', '33', '34', '35', '36', '37']:
                                fg_color = ansi_to_palette[int(code)]
                            elif code in ['40', '41', '42', '43', '44', '45', '46', '47']:
                                if code == '40':
                                    bg_color = team_bg  # Black background uses team background
                                else:
                                    bg_color = ansi_to_palette[int(code) - 10]
                        
                        if codes_str.startswith('38;2;'):
                            parts = codes_str.split(';')
                            if len(parts) >= 5:
                                fg_color = (int(parts[2]), int(parts[3]), int(parts[4]))
                        elif codes_str.startswith('48;2;'):
                            parts = codes_str.split(';')
                            if len(parts) >= 5:
                                bg_color = (int(parts[2]), int(parts[3]), int(parts[4]))
                    
                    i = j + 1
                else:
                    i += 1
            
            elif ansi_text[i] == '\n':
                x = 0
                y += 1
                i += 1
            
            elif ansi_text[i] == '\r':
                x = 0
                i += 1
            
            else:
                # Render character
                if 0 <= x < self.terminal_width and 0 <= y < self.height:
                    actual_fg = fg_color
                    actual_bg = bg_color
                    
                    if reverse:
                        actual_fg, actual_bg = actual_bg, actual_fg
                    
                    if dim:
                        actual_fg = RGBColors.dim(actual_fg)
                    elif bold:
                        actual_fg = RGBColors.brighten(actual_fg, 1.5)
                    
                    # Calculate style flags for cache
                    style_flags = 0
                    if bold: style_flags |= 1
                    if italic: style_flags |= 2
                    if underline: style_flags |= 4
                    if strike: style_flags |= 8
                    if blink and ((frame // 6) % 2 == 0):
                        actual_fg = actual_bg
                    
                    # Check if this is an emoji flagged character
                    is_emoji_char = emoji_start_pending
                    if emoji_start_pending:
                        emoji_start_pending = False
                    
                    # Calculate pixel position
                    y_pos = y * self.char_height
                    x_pos = x * self.char_width
                    
                    # Get the bitmap through layout engine
                    if self.layout_engine:
                        bitmap = self.layout_engine.get_character_bitmap(
                            ansi_text[i], actual_fg, actual_bg, style_flags
                        )
                    else:
                        bitmap = self._get_char_bitmap_styled(
                            ansi_text[i], actual_fg, actual_bg, 
                            personality, style_flags, underline, strike
                        )
                    
                    # Check if this is an emoji or was flagged as emoji
                    if self._is_emoji(ansi_text[i]) or is_emoji_char:
                        # Validate emoji dimensions before placement
                        if bitmap.shape != (16, 16, 3):
                            logger.warning(f"Emoji bitmap has wrong dimensions: {bitmap.shape}, creating fallback")
                            # Create a 16x16 fallback
                            fallback = np.full((16, 16, 3), actual_bg, dtype=np.uint8)
                            h, w = bitmap.shape[:2]
                            # Copy what we can
                            copy_h = min(h, 16)
                            copy_w = min(w, 16)
                            y_off = (16 - copy_h) // 2
                            x_off = (16 - copy_w) // 2
                            fallback[y_off:y_off + copy_h, x_off:x_off + copy_w] = bitmap[:copy_h, :copy_w]
                            bitmap = fallback
                        
                        # Place the 16x16 emoji at current position
                        # The emoji will visually overflow into the next cell, but that's expected
                        # since the buffer has already allocated that space
                        if y_pos + 16 <= self.inner_height and x_pos + 16 <= self.inner_width:
                            inner_buffer[y_pos:y_pos + 16, x_pos:x_pos + 16] = bitmap
                        elif y_pos + 16 <= self.inner_height and x_pos < self.inner_width:
                            # Partial placement at right edge
                            available_width = self.inner_width - x_pos
                            inner_buffer[y_pos:y_pos + 16, x_pos:x_pos + available_width] = bitmap[:, :available_width]
                        
                        # Advance x by 2 for emojis
                        x += 2
                        i += 1
                    else:
                        # Regular character - standard placement
                        if y_pos + self.char_height <= self.inner_height and x_pos + self.char_width <= self.inner_width:
                            inner_buffer[y_pos:y_pos + self.char_height,
                                       x_pos:x_pos + self.char_width] = bitmap[:self.char_height, :self.char_width]
                        
                        # Regular character advances by 1
                        x += 1
                    
                    # Handle line wrapping
                    if x >= self.terminal_width:
                        x = 0
                        y += 1
                
                # Always advance the string index by 1
                i += 1
        
        # Create the final image from inner buffer
        img = Image.fromarray(inner_buffer)
        
        return img
    
    def _get_char_bitmap_styled(self, char: str, fg_color: Tuple[int, int, int], 
                               bg_color: Tuple[int, int, int], personality: Optional[str], 
                               style_flags: int, underline: bool, strike: bool) -> np.ndarray:
        """Get character bitmap with style effects using layout engine cache when available"""
        # Use layout engine if available for cached bitmap retrieval
        if self.layout_engine:
            # Combine underline and strike into style_flags
            combined_flags = style_flags
            if underline:
                combined_flags |= 4
            if strike:
                combined_flags |= 8
            
            return self.layout_engine.get_character_bitmap(char, fg_color, bg_color, combined_flags)
        
        # Fallback to direct rendering if layout engine unavailable
        # Check if character is emoji - use pre-loaded bitmap
        if self._is_emoji(char):
            emoji_bitmap = self._render_color_emoji(char, bg_color)
            if emoji_bitmap is not None:
                # Apply style effects to emoji if needed (underline, strike)
                if underline or strike:
                    # Convert to PIL Image for drawing
                    img = Image.fromarray(emoji_bitmap.astype(np.uint8))
                    draw = ImageDraw.Draw(img)
                    
                    if underline:
                        y_pos = 14  # Position underline near bottom of 16x16 emoji
                        underline_color = RGBColors.brighten(fg_color, 1.3)
                        draw.line([(0, y_pos), (15, y_pos)], fill=underline_color, width=1)
                    
                    if strike:
                        y_pos = 8  # Middle of 16x16 emoji
                        draw.line([(0, y_pos), (15, y_pos)], fill=fg_color, width=1)
                    
                    # Convert back to numpy array
                    emoji_bitmap = np.array(img)
                
                # Return the emoji bitmap - it will be cached by the calling method
                return emoji_bitmap
        
        # Regular styled text rendering continues here...
        # Determine actual dimensions based on character type
        char_width = self.char_width
        char_height = self.char_height
        
        img = Image.new('RGB', (char_width, char_height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Select font based on style flags
        font_to_use = self.font
        if hasattr(self, 'font_variants'):
            if (style_flags & 1) and (style_flags & 2):  # Bold + Italic
                font_to_use = self.font_variants.get('bold_italic', self.font)
            elif style_flags & 1:  # Bold
                font_to_use = self.font_variants.get('bold', self.font)
            elif style_flags & 2:  # Italic
                font_to_use = self.font_variants.get('italic', self.font)
        
        draw.text((0, 0), char, font=font_to_use, fill=fg_color)
        
        # Apply underline and strikethrough as needed
        if underline:
            y_pos = char_height - 2
            underline_color = RGBColors.brighten(fg_color, 1.3)
            draw.line([(0, y_pos), (char_width - 1, y_pos)], fill=underline_color, width=1)
        
        if strike:
            y_pos = char_height // 2
            draw.line([(0, y_pos), (char_width - 1, y_pos)], fill=fg_color, width=1)
        
        return np.array(img)
    
    def _is_emoji(self, char: str) -> bool:
        """OPTIMIZED: O(1) emoji detection using EmojiManager"""
        return self.emoji_manager.is_emoji(char)
    
    def _render_color_emoji(self, char: str, bg_color: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """OPTIMIZED: Get pre-loaded emoji bitmap with team-based borders"""
        # Get pre-processed emoji from manager (without outline initially)
        emoji_bitmap = self.emoji_manager.get_emoji_bitmap(char, outline_color=None)
        
        if emoji_bitmap is None:
            logger.debug(f"No pre-loaded emoji for {char} (U+{format(ord(char[0]), 'x').upper()})")
            return None
        
        # Make a copy to avoid modifying the cached version
        emoji_bitmap = emoji_bitmap.copy()
        
        # Apply team-based border styling
        if hasattr(self, '_team_colors'):
            team_colors = self._team_colors
            team = self._current_team if hasattr(self, '_current_team') else 'PNGN'
            
            if team == 'PNGN':
                # Digital glitch style - layered purple borders
                emoji_bitmap[0, :] = team_colors['glow']      # Top outer - Neon Purple
                emoji_bitmap[-1, :] = team_colors['glow']     # Bottom outer - Neon Purple
                emoji_bitmap[:, 0] = team_colors['glow']      # Left outer - Neon Purple
                emoji_bitmap[:, -1] = team_colors['glow']     # Right outer - Neon Purple
                
                # Inner accent line for depth
                emoji_bitmap[1, 1:-1] = team_colors['accent']     # Top inner - Electric Violet
                emoji_bitmap[-2, 1:-1] = team_colors['accent']    # Bottom inner - Electric Violet
                emoji_bitmap[1:-1, 1] = team_colors['accent']     # Left inner - Electric Violet
                emoji_bitmap[1:-1, -2] = team_colors['accent']    # Right inner - Electric Violet
                
            elif team == 'KLLR':
                # Soft vibrant style - gradient pink borders
                emoji_bitmap[0, :] = team_colors['primary']    # Top - KLLR Pink
                emoji_bitmap[-1, :] = team_colors['primary']   # Bottom - KLLR Pink
                emoji_bitmap[:, 0] = team_colors['primary']    # Left - KLLR Pink
                emoji_bitmap[:, -1] = team_colors['primary']   # Right - KLLR Pink
                
                # Brighter glow accent
                emoji_bitmap[1, 1:-1] = team_colors['glow']       # Top inner - Neon Magenta
                emoji_bitmap[-2, 1:-1] = team_colors['glow']      # Bottom inner - Neon Magenta
                emoji_bitmap[1:-1, 1] = team_colors['glow']       # Left inner - Neon Magenta
                emoji_bitmap[1:-1, -2] = team_colors['glow']      # Right inner - Neon Magenta
                
            elif team == 'SHMA':
                # Organic/toxic style - radioactive green borders
                emoji_bitmap[0, :] = team_colors['glow']      # Top - Radiation Green
                emoji_bitmap[-1, :] = team_colors['glow']     # Bottom - Radiation Green
                emoji_bitmap[:, 0] = team_colors['glow']      # Left - Radiation Green
                emoji_bitmap[:, -1] = team_colors['glow']     # Right - Radiation Green
                
                # Accent with primary green
                emoji_bitmap[1, 1:-1] = team_colors['primary']    # Top inner - Neon Green
                emoji_bitmap[-2, 1:-1] = team_colors['primary']   # Bottom inner - Neon Green
                emoji_bitmap[1:-1, 1] = team_colors['primary']    # Left inner - Neon Green
                emoji_bitmap[1:-1, -2] = team_colors['primary']   # Right inner - Neon Green
            
            else:
                # Default neutral border
                outline_color = team_colors.get('text', PNGN_32_COLORS[31]['rgb'])
                emoji_bitmap[0, :] = outline_color
                emoji_bitmap[-1, :] = outline_color
                emoji_bitmap[:, 0] = outline_color
                emoji_bitmap[:, -1] = outline_color
        else:
            # Fallback if no team colors available
            outline_color = PNGN_32_COLORS[31]['rgb']  # White Flash
            emoji_bitmap[0, :] = outline_color
            emoji_bitmap[-1, :] = outline_color
            emoji_bitmap[:, 0] = outline_color
            emoji_bitmap[:, -1] = outline_color
        
        return emoji_bitmap

# ============================================================================
    # TEAM COLOR AND BORDER METHODS
    # ============================================================================
    
    def _get_team_colors(self, team: Optional[str]) -> Dict[str, Tuple[int, int, int]]:
        """Get comprehensive color scheme for a team"""
        if team == 'PNGN':
            return {
                'primary': PNGN_32_COLORS[0]['rgb'],     # PNGN Purple
                'text': PNGN_32_COLORS[2]['rgb'],        # Neon Purple
                'glow': PNGN_32_COLORS[1]['rgb'],        # Neon Purple
                'accent': PNGN_32_COLORS[3]['rgb'],      # Electric Violet
                'background': PNGN_32_COLORS[7]['rgb'],  # Void Black
                'corruption': PNGN_32_COLORS[6]['rgb'],  # Deep Void
                'glitch': [
                    PNGN_32_COLORS[0]['rgb'],   # PNGN Purple
                    PNGN_32_COLORS[1]['rgb'],   # Neon Purple
                    PNGN_32_COLORS[2]['rgb'],   # Electric Violet
                    PNGN_32_COLORS[3]['rgb'],   # Ultra Violet
                ],
            }
        elif team == 'KLLR':
            return {
                'primary': PNGN_32_COLORS[8]['rgb'],      # KLLR Pink
                'text': PNGN_32_COLORS[10]['rgb'],        # Deep Pink
                'glow': PNGN_32_COLORS[9]['rgb'],         # Hot Magenta
                'accent': PNGN_32_COLORS[11]['rgb'],      # Hot Pink
                'background': PNGN_32_COLORS[15]['rgb'],  # Blood Black
                'corruption': PNGN_32_COLORS[14]['rgb'],  # Neon Red
                'glitch': [
                    PNGN_32_COLORS[8]['rgb'],   # KLLR Pink
                    PNGN_32_COLORS[9]['rgb'],   # Hot Magenta
                    PNGN_32_COLORS[10]['rgb'],  # Deep Pink
                    PNGN_32_COLORS[25]['rgb'],  # Ghost White
                ],
            }
        elif team == 'SHMA':
            return {
                'primary': PNGN_32_COLORS[16]['rgb'],    # Neon Green
                'text': PNGN_32_COLORS[18]['rgb'],       # Radiation Green
                'glow': PNGN_32_COLORS[17]['rgb'],       # Toxic Lime
                'accent': PNGN_32_COLORS[19]['rgb'],     # Acid Lime
                'background': PNGN_32_COLORS[23]['rgb'], # Toxic Black
                'corruption': PNGN_32_COLORS[22]['rgb'], # Fire Orange
                'glitch': [
                    PNGN_32_COLORS[16]['rgb'],  # Neon Green
                    PNGN_32_COLORS[17]['rgb'],  # Toxic Lime
                    PNGN_32_COLORS[19]['rgb'],  # Acid Lime
                    PNGN_32_COLORS[20]['rgb'],  # Electric Yellow
                ],
            }
        else:
            # Default/neutral colors
            return {
                'primary': PNGN_32_COLORS[24]['rgb'],    # White Flash
                'text': PNGN_32_COLORS[24]['rgb'],       # White Flash
                'glow': PNGN_32_COLORS[26]['rgb'],       # Plasma Silver
                'accent': PNGN_32_COLORS[26]['rgb'],     # Plasma Silver
                'background': PNGN_32_COLORS[7]['rgb'],  # Void Black
                'corruption': PNGN_32_COLORS[14]['rgb'], # Corruption Crimson
                'glitch': [
                    PNGN_32_COLORS[24]['rgb'],  # White Flash
                    PNGN_32_COLORS[26]['rgb'],  # Plasma Silver
                ],
            }
    
    def _add_team_border(self, img: Image.Image, team: Optional[str], frame: int) -> Image.Image:
        """Add a team-colored OUTER GLOW border around the terminal output - DETERMINISTIC"""
        if not team:
            team = 'PNGN'
        
        # Get team colors from the centralized system
        team_colors = self._get_team_colors(team)
        team_data = {
            'primary': team_colors['primary'],
            'glow': team_colors['glow'],
            'accent': team_colors['accent'],
        }
        
        # Create new image with border
        bordered_img = Image.new('RGB', (self.img_width, self.img_height), team_data['accent'])
        draw = ImageDraw.Draw(bordered_img)
        
        # Draw border based on team style with ENHANCED colors - DETERMINISTIC
        if team.upper() == 'PNGN':
            # Digital glitch style - multiple colored lines with NEON effect
            colors = [
                team_data['primary'],    # PNGN Purple (179, 0, 255)
                team_data['glow'],       # Neon Purple (191, 0, 255)
                team_data['accent'],     # Electric Violet (139, 0, 255)
            ]
            for i in range(self.outer_border_thickness):
                color = colors[i % len(colors)]
                # Add glow effect by brightening - DETERMINISTIC
                if i == 1:
                    color = RGBColors.brighten(color, 1.3)
                
                # Draw rectangle border
                draw.rectangle(
                    [i, i, self.img_width - i - 1, self.img_height - i - 1],
                    outline=color,
                    width=1
                )
        
        elif team.upper() == 'KLLR':
            # Soft gradient style with VIBRANT pink transitions - DETERMINISTIC
            for i in range(self.outer_border_thickness):
                # Blend from hot pink to neon magenta
                ratio = i / max(self.outer_border_thickness - 1, 1)
                color = RGBColors.blend(team_data['primary'], team_data['glow'], ratio)
                
                # Add shimmer effect - DETERMINISTIC based on frame
                if i == self.outer_border_thickness // 2:
                    shimmer = 1.5 + 0.2 * math.sin(frame * 0.1)
                    color = RGBColors.brighten(color, shimmer)
                
                draw.rectangle(
                    [i, i, self.img_width - i - 1, self.img_height - i - 1],
                    outline=color,
                    width=1
                )
        
        elif team.upper() == 'SHMA':
            # Organic/toxic style with RADIOACTIVE glow - DETERMINISTIC
            colors = [
                team_data['primary'],    # Neon Green (0, 255, 0)
                team_data['glow'],       # Radiation Green (0, 255, 127)
                team_data['accent'],     # Toxic Lime (127, 255, 0)
            ]
            for i in range(self.outer_border_thickness):
                base_color = colors[i % len(colors)]
                
                # Add pulsing effect - DETERMINISTIC
                pulse = math.sin(frame * 0.15 + i) * 0.3 + 1.0
                color = RGBColors.brighten(base_color, pulse)
                
                # Add some deterministic "organic" variation
                points = []
                for x in range(0, self.img_width, 10):
                    # Use deterministic offset
                    offset = int(self._get_pseudo_random(x, i, frame, "border_x") * 3 - 1) if i == self.outer_border_thickness - 1 else 0
                    points.extend([
                        (x, i + offset),
                        (x, self.img_height - i - 1 + offset)
                    ])
                for y in range(0, self.img_height, 10):
                    offset = int(self._get_pseudo_random(i, y, frame, "border_y") * 3 - 1) if i == self.outer_border_thickness - 1 else 0
                    points.extend([
                        (i + offset, y),
                        (self.img_width - i - 1 + offset, y)
                    ])
                
                # Draw the organic border
                draw.rectangle(
                    [i, i, self.img_width - i - 1, self.img_height - i - 1],
                    outline=color,
                    width=1
                )
        
        # Paste the original image in the center
        bordered_img.paste(img, (self.outer_border_thickness, self.outer_border_thickness))
        
        return bordered_img
    
    # ============================================================================
    # PARTICLE RENDERING
    # ============================================================================
    
    def render_particles(self, braille_lines: List[str], shade_map: Dict[Tuple[int, int], float], 
                        frame: int = 0, team: str = 'PNGN', base_image: Optional[Image.Image] = None) -> Image.Image:
        """Render Braille particle output with ENHANCED team colors - DETERMINISTIC"""
        # Particle text should be pre-formatted by buffer
        particle_text = '\n'.join(braille_lines)
        
        # If base_image provided, use it; otherwise create new one
        if base_image:
            img = base_image.copy()
            self.buffer = np.array(img)
        else:
            # Clear buffer with team-appropriate background
            self.buffer.fill(0)
            bg_color = PNGN_32_COLORS[7]['rgb']  # Void Black
            bg_color_array = np.array(bg_color, dtype=np.uint8)
            self.buffer[:, :] = bg_color_array
            img = Image.new('RGB', (self.inner_width, self.inner_height), bg_color)
        
        draw = ImageDraw.Draw(img)
        
        # Get team colors
        colors = self._team_colors if hasattr(self, '_team_colors') else self._get_team_colors(team)
        base_color = colors['glow']
        
        # NO inner border offset needed
        x_offset = 0
        y_offset = 0
        
        # Render each Braille character
        lines = particle_text.split('\n')
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char != ' ':
                    # Get brightness from shade map
                    brightness = shade_map.get((x, y), 1.0)
                    
                    # Apply team-specific color effects - DETERMINISTIC selection
                    if team == 'PNGN':
                        # Digital particles with purple-blue gradient
                        if brightness > 0.7:
                            color = PNGN_32_COLORS[2]['rgb']  # Neon Purple
                        elif brightness > 0.4:
                            color = PNGN_32_COLORS[0]['rgb']  # PNGN Purple
                        else:
                            color = PNGN_32_COLORS[6]['rgb']  # Deep Blue
                    elif team == 'KLLR':
                        # Love particles with pink-red gradient
                        if brightness > 0.7:
                            color = PNGN_32_COLORS[8]['rgb']  # Neon Magenta
                        elif brightness > 0.4:
                            color = PNGN_32_COLORS[1]['rgb']  # KLLR Pink
                        else:
                            color = PNGN_32_COLORS[12]['rgb']  # Pure Red
                    else:  # SHMA
                        # Toxic particles with green-yellow gradient
                        if brightness > 0.7:
                            color = PNGN_32_COLORS[16]['rgb']  # Electric Yellow
                        elif brightness > 0.4:
                            color = PNGN_32_COLORS[19]['rgb']  # Radiation Green
                        else:
                            color = PNGN_32_COLORS[14]['rgb']  # Neon Green
                    
                    # Apply brightness
                    color = tuple(int(c * brightness) for c in color)
                    
                    # Draw character without border offset
                    pixel_x = x * self.char_width + x_offset
                    pixel_y = y * self.char_height + y_offset
                    draw.text((pixel_x, pixel_y), char, font=self.font, fill=color)
        
        # Update buffer from image
        self.buffer = np.array(img)
        
        return img
    
    # ============================================================================
    # MAIN RENDER METHOD
    # ============================================================================
    
    def render(self, 
               content: str,
               personality: Optional[str] = None,
               effects: Optional[List[str]] = None,
               frame: int = 0,
               context: Optional[Dict[str, Any]] = None,
               rgb_colors: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]] = None,
               creature_data: Optional[Any] = None,
               user_id: Optional[str] = None,
               reactions: Optional[List[Tuple[str, int, int]]] = None,
               animation_mode: bool = False) -> Image.Image:
        """
        Pure text rendering with deterministic effects and three-tier caching.
        
        Cache Strategy:
        - Level 1: Complete frame cache (highest performance, full pipeline bypass)
        - Level 2: ANSI string cache (bypasses markdown, effects, grid processing)
        - Level 3: Character bitmap cache via layout engine (per-character optimization)
        
        Args:
            content: Text or ANSI content to render
            personality: Team name for coloring and border
            effects: List of visual effects to apply
            frame: Current frame number for animations
            context: Additional context (variations, etc)
            rgb_colors: RGB split colors if needed
            creature_data: Creature data (ignored - compatibility)
            user_id: User ID (ignored - compatibility)
            reactions: Reaction data (ignored - compatibility)
            animation_mode: If True, bypass caching for smooth animations
            
        Returns:
            PIL Image at desktop resolution with team glow border
        """
        
        if self.shutting_down:
            return Image.new('RGB', (self.img_width, self.img_height), PNGN_32_COLORS[8]['rgb'])
        
        start_time = time.time()
        
        # Store animation mode for cache bypass
        self._animation_mode = animation_mode
        
        # Create content hash for deterministic effects and cache keys
        self._content_hash = self._create_content_hash(content)
        self._frame_seed = frame
        self._current_team = personality
        self._team_colors = self._get_team_colors(personality)
        
        # Generate cache keys for three-tier strategy
        effects_hash = hashlib.md5(str(sorted(effects or [])).encode()).hexdigest()[:8]
        frame_cache_key = f"frame_{self._content_hash}_{personality}_{frame}_{effects_hash}"
        ansi_cache_key = f"ansi_{self._content_hash}_{personality}_{frame}_{effects_hash}"
        
        # TIER 1: Check for complete cached frame (highest performance)
        if self.cache and not animation_mode:
            cached_frame = self.cache.get_frame(frame_cache_key)
            if cached_frame is not None:
                logger.debug(f"Frame cache HIT - skipping entire render pipeline")
                render_time = (time.time() - start_time) * 1000
                self.render_times.append(render_time)
                return cached_frame
        
        # TIER 2: Check for cached ANSI (bypasses markdown, effects, grid processing)
        cached_ansi = None
        if self.cache and not animation_mode:
            cached_ansi_bytes = self.cache.get_ansi(ansi_cache_key)
            if cached_ansi_bytes is not None:
                try:
                    cached_ansi = cached_ansi_bytes.decode('utf-8')
                    logger.debug(f"ANSI cache HIT - skipping markdown and effects processing")
                except Exception as e:
                    logger.warning(f"Failed to decode cached ANSI: {e}")
                    cached_ansi = None
        
        # Sanitize input
        content = sanitize_terminal_input(content)
        
        # Generate or use cached ANSI
        if cached_ansi is not None:
            ansi_text = cached_ansi
        else:
            # Full processing pipeline for cache miss
            markdown_regions = self._detect_markdown_in_text(content)
            instructions = self._text_to_instructions_with_markdown(content, personality, effects, markdown_regions)
            ansi_text = self._apply_effects(instructions, personality, effects or [], frame, context, rgb_colors)
            
            # Store ANSI in cache for future renders
            if self.cache and not animation_mode:
                self.cache.put_ansi(ansi_cache_key, ansi_text)
                logger.debug(f"Stored ANSI in cache")
        
        # Render ANSI to image (TIER 3 character cache used internally)
        img = self._render_ansi_to_image(ansi_text, personality, frame)
        
        # Add team-based outer border
        img = self._add_team_border(img, personality, frame)
        
        # Store complete frame in cache
        if self.cache and not animation_mode:
            self.cache.put_frame(frame_cache_key, img)
            logger.debug(f"Stored complete frame in cache")
        
        # Track render time
        render_time = (time.time() - start_time) * 1000
        self.render_times.append(render_time)
        
        # Handle memory pressure if detected
        if self.memory_pressure_callback and render_time > 100:
            self.memory_pressure_callback(render_time / 200)
        
        return img
    
    # ============================================================================
    # STATISTICS AND UTILITY METHODS
    # ============================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get render statistics with emoji optimization info and hybrid status"""
        if not self.render_times:
            return {'status': 'No renders yet'}
        
        stats = {
            'avg_render_time': sum(self.render_times) / len(self.render_times),
            'min_render_time': min(self.render_times),
            'max_render_time': max(self.render_times),
            'renders_completed': len(self.render_times),
            'deterministic': True,
            'uses_pngn_32_colors': True,
            'building_blocks_available': BUILDING_BLOCKS_AVAILABLE,
            'markdown_detected': sum(self.markdown_stats.values()),
            'cache_available': self.cache is not None,
            'emoji_flag_handling': True,
            'emoji_optimization': {
                'pre_loaded': len(self.emoji_manager.emoji_cache),
                'load_failures': self.emoji_manager.failed_count,
                'detection_complexity': 'O(1)',
                'uses_pre_processing': True
            },
            'hybrid_optimizations': {
                'vectorization': self.using_vectorization,
                'layout_engine': self.using_layout_engine,
                'vector_module': VECTOR_MODULE_AVAILABLE,
            }
        }
        
        # Add cache stats if available
        if self.cache:
            stats['cache_type'] = 'UnifiedCacheSystem'
            cache_stats = self.cache.get_stats()
            stats['cache_stats'] = {
                'frame_cache_used': cache_stats['used']['frame'],
                'ansi_cache_used': cache_stats['used']['ansi'],
                'hit_rates': cache_stats['hit_rates']
            }
        
        # Add vectorized metrics stats if available
        if self.using_vectorization and self.vector_metrics:
            stats['vector_metrics'] = self.vector_metrics.get_stats()
        
        # Add layout engine stats if available
        if self.layout_engine:
            stats['layout_engine'] = self.layout_engine.get_stats()
        
        # Add dirty tracking stats if available
        if self.dirty_tracker:
            stats['incremental_rendering'] = {
                'enabled': True,
                'dirty_percentage': self.dirty_tracker.compute_dirty_percentage()
            }
        
        return stats


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_terminal(terminal_width: int = TERMINAL_WIDTH, height: int = TERMINAL_HEIGHT) -> DeterministicPureTerminalRenderer:
    """Factory function for terminal creation"""
    return DeterministicPureTerminalRenderer(terminal_width, height)
