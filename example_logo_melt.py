#!/usr/bin/env python3
"""
🧊 PNGN Terminal Animator - Logo Melt Animation Example
=======================================================
Copyright (c) 2025 PNGN-Tec LLC
"""

import random
import argparse
from typing import List
from PIL import Image
from pngn_terminal import create_terminal

PNGN_LOGO = [
    "▒▓███▒███▄   ▒█▒█████▒███▄   ▒█",
    " ▒██▒ ██▒██ ▀█ ▒██ ██  ▒  ██ ▀█ ▒██",
    " ▒██▒ █▒▒██  ▀█ █  ██▒▄▄▄▒██  ▀█ █",
    " ▒██▄█▓▒▒██▒  ▀███▒▒▓█  ██▒██▒  ▀███▒",
    "▒██▒ ▒ ▒▒██▒   ▓██▒▒▓███▀▒██▒   ▓██▒",
    "▒▓▒▒ ▒ ▒ ▒▒   ▒ ▒ ▒▒   ▒▒▒▒    ▒ ▒▒",
    " ▒ ▒   ▒ ▒▒   ▒ ▒▒ ▒   ▒▒▒▒    ▒ ▒▒",
    " ▒       ▒    ▒ ▒       ▒ ▒    ▒ ▒"
]

BORDERS = {'tl': '╔', 'tr': '╗', 'bl': '╚', 'br': '╝', 'h': '═', 'v': '║'}
BASE_TERMINAL_WIDTH = 46
BASE_TERMINAL_HEIGHT = 23
PHASE_1_IDLE_FRAMES = 12
PHASE_2_TYPING_FRAMES = 24
TOTAL_FRAMES = 36
LOGO_PULSE_FRAMES = 6


class LogoMelter:
    def __init__(self, seed: int = 12345):
        self.seed = seed
        self._current_typing_frame = 0
        self._drip_duration_tracker = {}
    
    def render_frame(self, frame: int) -> List[str]:
        typing_frame_in_phase = frame % PHASE_2_TYPING_FRAMES
        typing_progress = typing_frame_in_phase / PHASE_2_TYPING_FRAMES
        
        if typing_progress < 0.3:
            melt_level = 0
        elif typing_progress < 0.6:
            melt_level = 1
        else:
            melt_level = 2
        
        typing_frame = self._current_typing_frame
        self._current_typing_frame = (typing_frame + 1) % PHASE_2_TYPING_FRAMES
        
        if melt_level == 0:
            if self._drip_duration_tracker:
                self._drip_duration_tracker = {}
            return list(PNGN_LOGO)
        
        melt_random = random.Random(self.seed)
        frame_based_melt = min(3, typing_frame // 6)
        actual_melt_level = min(melt_level, frame_based_melt)
        
        grid = [list(line) for line in PNGN_LOGO]
        max_drip_length = 4 * actual_melt_level
        
        for _ in range(max_drip_length):
            grid.append([' '] * len(grid[0]) if grid else [])
        
        max_degrade_rows = min(3 + actual_melt_level * 2, len(PNGN_LOGO))
        
        for row in range(max_degrade_rows):
            for col in range(len(grid[row]) if row < len(grid) else 0):
                if col < len(grid[row]):
                    char = grid[row][col]
                    if char == ' ':
                        continue
                    
                    base_chance = 1.0 - (row * 0.2)
                    degrade_chance = base_chance * (actual_melt_level / 3.0)
                    
                    degradation_level = 0
                    progression_multiplier = (typing_frame + 1) / PHASE_2_TYPING_FRAMES
                    max_degradation = int(actual_melt_level * progression_multiplier * 3)
                    
                    while degradation_level < max_degradation and melt_random.random() < degrade_chance:
                        if char == '█':
                            char = '▓'
                        elif char == '▓':
                            char = '▒'
                        elif char == '▒':
                            char = '░'
                        elif char == '░' and actual_melt_level >= 3:
                            char = ' '
                        elif char in '▀▄':
                            char = '░'
                        elif char in '╗╔╚║' and actual_melt_level >= 2:
                            char = '░'
                        
                        degradation_level += 1
                        degrade_chance *= 0.7
                    
                    grid[row][col] = char
        
        drip_positions = []
        max_drips = 3 + actual_melt_level * 4
        drip_count = 0
        solid_block = '█'
        
        for col in range(len(grid[0]) if grid else 0):
            if drip_count >= max_drips:
                break
            for row in range(len(PNGN_LOGO) - 1, -1, -1):
                if row < len(grid) and col < len(grid[row]):
                    char = grid[row][col]
                    if char == solid_block:
                        seed_value_local = (col * 7 + row * 13) % 100
                        if seed_value_local < (actual_melt_level * 30):
                            drip_positions.append((row, col))
                            drip_count += 1
                            
                            pos_key = f"{row}_{col}"
                            if pos_key not in self._drip_duration_tracker:
                                self._drip_duration_tracker[pos_key] = 0
                            self._drip_duration_tracker[pos_key] += 1
                            break
        
        progression_factor = (typing_frame / PHASE_2_TYPING_FRAMES)
        
        for i, (row, col) in enumerate(drip_positions):
            pos_key = f"{row}_{col}"
            drip_age = self._drip_duration_tracker.get(pos_key, 1)
            
            base_length = min(1 + actual_melt_level, max_drip_length)
            progression_growth = int(progression_factor * 3)
            age_growth = min(drip_age // 4, 3)
            variance = (col % 3) - 1
            
            drip_length = max(1, base_length + progression_growth + age_growth + variance)
            
            if i < len(drip_positions) // 3:
                drip_length = min(drip_length + 1, max_drip_length)
            
            drip_length = min(drip_length, max_drip_length)
            
            for d in range(drip_length):
                drip_row = row + d + 1
                if drip_row < len(grid) and col < len(grid[drip_row]):
                    if d == 0:
                        drip_char = '╎'
                    elif d == drip_length - 1:
                        drip_char = '·' if drip_length > 2 else '│'
                    else:
                        drip_char = '│'
                    
                    grid[drip_row][col] = drip_char
                    
                    if d > 1 and actual_melt_level >= 2 and i < max_drips // 2:
                        for adj_col in [col - 1, col + 1]:
                            if 0 <= adj_col < len(grid[0]) and adj_col < len(grid[drip_row]):
                                if grid[drip_row][adj_col] == ' ':
                                    spawn_chance = (col + adj_col + d + actual_melt_level) % 5
                                    if spawn_chance == 0 and progression_factor > 0.5:
                                        secondary_length = min(
                                            int((progression_factor - 0.5) * 4) + age_growth,
                                            len(grid) - drip_row - 1
                                        )
                                        for sd in range(secondary_length):
                                            if drip_row + sd + 1 < len(grid) and adj_col < len(grid[drip_row + sd + 1]):
                                                secondary_char = '┆' if sd < secondary_length - 1 else '·'
                                                grid[drip_row + sd + 1][adj_col] = secondary_char
        
        current_drip_keys = {f"{row}_{col}" for row, col in drip_positions}
        keys_to_remove = [k for k in self._drip_duration_tracker if k not in current_drip_keys]
        for key in keys_to_remove:
            del self._drip_duration_tracker[key]
        
        result_lines = []
        for row in grid:
            line = ''.join(row)
            if line.strip() or any(c in '╎│·┆' for c in line):
                result_lines.append(line)
        
        return result_lines


def apply_logo_glitch(logo_lines: List[str], frame: int, seed: int, team: str = 'PNGN') -> List[str]:
    if frame % LOGO_PULSE_FRAMES >= 2:
        return logo_lines
    
    frame_random = random.Random(seed + frame)
    glitched_lines = []
    
    for line in logo_lines:
        if line.strip() and frame_random.random() < 0.3:
            if team == 'PNGN':
                glitched = []
                for char in line:
                    if char in '█▓▒░' and frame_random.random() < 0.2:
                        glitched.append(frame_random.choice(['█', '▓', '▒', '░', '▀', '▄']))
                    elif char != ' ' and frame_random.random() < 0.1:
                        glitched.append(frame_random.choice(['0', '1', '│', '─']))
                    else:
                        glitched.append(char)
                line = ''.join(glitched)
        
        glitched_lines.append(line)
    
    return glitched_lines


def build_terminal_frame(logo_lines: List[str], frame: int) -> str:
    lines = []
    
    title = " PNGN LOGO MELT "
    top_line = BORDERS['tl'] + BORDERS['h'] * (BASE_TERMINAL_WIDTH - 2) + BORDERS['tr']
    title_start = (BASE_TERMINAL_WIDTH - len(title)) // 2
    top_list = list(top_line)
    for i, char in enumerate(title):
        if title_start + i < len(top_list) - 1:
            top_list[title_start + i] = char
    lines.append(''.join(top_list))
    
    lines.append(BORDERS['v'] + ' ' * (BASE_TERMINAL_WIDTH - 2) + BORDERS['v'])
    
    max_content_lines = BASE_TERMINAL_HEIGHT - 5
    
    for logo_line in logo_lines[:max_content_lines]:
        if len(logo_line) > BASE_TERMINAL_WIDTH - 4:
            logo_line = logo_line[:BASE_TERMINAL_WIDTH - 4]
        padding = (BASE_TERMINAL_WIDTH - 2 - len(logo_line)) // 2
        centered = ' ' * padding + logo_line + ' ' * (BASE_TERMINAL_WIDTH - 2 - padding - len(logo_line))
        lines.append(BORDERS['v'] + centered + BORDERS['v'])
    
    while len(lines) < BASE_TERMINAL_HEIGHT - 2:
        lines.append(BORDERS['v'] + ' ' * (BASE_TERMINAL_WIDTH - 2) + BORDERS['v'])
    
    cursor = '█' if (frame // 10) % 2 == 0 else '_'
    cursor_line = f" {cursor} " + ' ' * (BASE_TERMINAL_WIDTH - 5)
    lines.append(BORDERS['v'] + cursor_line + BORDERS['v'])
    
    bottom_line = BORDERS['bl'] + BORDERS['h'] * (BASE_TERMINAL_WIDTH - 2) + BORDERS['br']
    lines.append(bottom_line)
    
    return '\n'.join(lines)


def get_frame_effects(frame: int) -> List[str]:
    effects = ['glow']
    
    if frame >= PHASE_1_IDLE_FRAMES:
        typing_frame = frame - PHASE_1_IDLE_FRAMES
        typing_progress = typing_frame / PHASE_2_TYPING_FRAMES
        
        if typing_progress > 0.2:
            effects.append('corruption')
        
        if typing_progress > 0.4:
            effects.append('glitch')
        
        if typing_progress > 0.6:
            effects.append('static')
    
    return effects


def main():
    parser = argparse.ArgumentParser(description='PNGN Logo Melting Animation')
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--frames', type=int, default=TOTAL_FRAMES)
    args = parser.parse_args()
    
    terminal = create_terminal(terminal_width=BASE_TERMINAL_WIDTH, height=BASE_TERMINAL_HEIGHT)
    melter = LogoMelter(seed=args.seed)
    
    print("🧊 PNGN Logo Melting Animation")
    print("=" * 60)
    print(f"Phase 1 (Idle):   Frames 0-11   (1.0s)")
    print(f"Phase 2 (Typing): Frames 12-35  (2.0s)")
    print(f"Total: {args.frames} frames at 12 FPS ({args.frames / 12:.1f}s)")
    print(f"Seed: {args.seed}")
    print()
    
    frames = []
    
    for frame in range(args.frames):
        if frame < PHASE_1_IDLE_FRAMES:
            logo_lines = list(PNGN_LOGO)
        else:
            typing_frame = frame - PHASE_1_IDLE_FRAMES
            logo_lines = melter.render_frame(typing_frame)
            logo_lines = apply_logo_glitch(logo_lines, typing_frame, args.seed, 'PNGN')
        
        terminal_text = build_terminal_frame(logo_lines, frame)
        effects = get_frame_effects(frame)
        
        img = terminal.render(
            content=terminal_text,
            personality='PNGN',
            effects=effects,
            frame=frame
        )
        
        frames.append(img)
        
        if (frame + 1) % 12 == 0:
            phase_name = "IDLE" if frame < PHASE_1_IDLE_FRAMES else "TYPING"
            print(f"  Frame {frame + 1}/{args.frames} [{phase_name}]")
    
    output = f"pngn_logo_melt_seed{args.seed}.gif"
    frames[0].save(
        output,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=83,
        loop=0,
        optimize=False
    )
    
    print(f"\n✓ Saved {output}")


if __name__ == "__main__":
    main()