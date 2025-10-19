# 🐧 Changelog

All notable changes to PNGN Terminal Animator.

---

## [1.0.0] - 2025-01-XX

### Added

**Core Rendering**
- Terminal animation renderer with frame-based output
- Three team themes: PNGN (purple), KLLR (pink), SHMA (green)
- Custom 32-color palette system with RGB values and ANSI codes
- Deterministic animation system using content-based hashing
- Frame timing optimized for 12 FPS GIF output

**Markdown Support**
- Headers (levels 1-6)
- Bold and italic text
- Inline code and code blocks
- Links
- Basic table detection

**Visual Effects**
- Glitch effect (brief digital artifacts)
- Corruption effect (character replacement)
- Static effect (TV-style noise)
- Content-based deterministic triggering
- Configurable effect character sets

**Color System**
- PNGN 32-color palette with team-specific ranges
- Palette-aware brighten and dim operations
- Team-specific border styling
- ANSI escape sequence parsing
- Support for 256-color and true-color modes

**Emoji Support**
- Pre-loaded emoji bitmap system
- Proper two-cell width allocation
- 16x16 pixel emoji rendering
- Team-colored borders
- O(1) emoji detection

**Width Calculator (pngn_width.py)**
- Unicode-aware text width measurement
- Thread-safe LRU cache with memory limits
- CJK character support (2-cell width)
- Emoji width calculation
- Combining mark handling
- Fallback mode without wcwidth library

**Configuration (config.py)**
- Terminal dimensions (default 46x23)
- Character cell size (8x16 pixels)
- Animation settings (blink rate, offsets)
- Effect character sets
- Complete color palette definitions
- Cache configuration for width calculator

**Examples**
- example_basic.py: Single frame rendering
- example_gif.py: Three GIF animations (loading, reveal, glitch)
- example_markdown.py: Markdown feature showcase

**Documentation**
- README with usage examples
- Professional module headers with detailed docstrings
- Type hints throughout codebase
- Inline comments for complex logic

**Infrastructure**
- requirements.txt (numpy, Pillow, optional wcwidth)
- MIT License
- Git ignore configuration
- Professional copyright headers

### Features

**Deterministic Output**
Content hash generates pseudo-random seeds ensuring identical content produces identical animations. Useful for testing and version control.

**Discord Optimization**
Default settings target Discord's 8MB file size limit with 12 FPS frame rate and 46x23 character dimensions optimized for inline display.

**Standalone Operation**
Minimal dependencies (numpy, Pillow) with all configuration in single module. Optional wcwidth for enhanced Unicode support.

**Memory Efficiency**
Three-tier caching system (frame cache, ANSI cache, character cache) with configurable limits and LRU eviction.

**Thread Safety**
Width calculator uses proper locking enabling concurrent access patterns.

### Technical Details

**Rendering Pipeline**
Text → Markdown detection → Render instructions → Effect application → ANSI conversion → Bitmap rendering → Border application → Output image

**Effect Timing**
Effects trigger during deterministic windows calculated from content hash. Glitch uses specific frame numbers, corruption uses 5-frame windows, static uses 8-frame intensity windows.

**Font System**
Attempts multiple font paths (Termux home, system, local) with graceful fallback to default. Supports font variants for bold, italic, and combined styling.

**Color Operations**
Brighten and dim functions step through palette rather than performing math, maintaining visual consistency and staying within defined color space.

### Performance

**Typical Metrics**
- Frame render time: 10-50ms
- Cache hit rate: 80%+ for repeated content
- Memory usage: ~50MB base + frame buffers
- Width calculator: O(1) cache lookup, O(n) calculation where n is string length

---