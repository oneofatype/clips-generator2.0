"""
Speaker Info Overlay Generator
==============================

Creates animated .ass subtitle files for speaker information overlays.
Uses the EXACT animation style from overlay.py (fly-up with blur focus).

The overlay displays speaker info with a cinematic fly-up animation:
- Name
- Title/Role
- Net worth (with ONLY the $ symbol in red)
"""

import random
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SpeakerInfo:
    """Speaker information for overlay."""
    name: str  # e.g., "Jack Zhang"
    title: str  # e.g., "Co-founder of Airwallex"
    net_worth: str  # e.g., "$8,000,000,000 Net Worth" - ONLY the $ will be red


def create_speaker_info_ass(
    speaker_info: SpeakerInfo,
    output_filename: str = "speaker_info.ass",
    width: int = 1080,
    height: int = 1920,
    start_time_seconds: float = 0.5,
    display_duration: float = 4.0,
    font_size_ratio: float = 0.12,  # Font size asratio of video width (1.6%)
    position_from_bottom_ratio: float = 0.25,  # 25% from bottom
    left_margin: int = 40  # Left margin for alignment
) -> str:
    """
    Create an .ass subtitle file with animated speaker info.
    Uses the EXACT animation style from overlay.py.
    
    Args:
        speaker_info: Speaker information dataclass
        output_filename: Output .ass file path
        width: Video width in pixels
        height: Video height in pixels
        start_time_seconds: When the animation starts
        display_duration: How long the text stays visible
        font_size_ratio: Font size as ratio of video width
        position_from_bottom_ratio: 30% from bottom
        left_margin: Left margin for text alignment
        
    Returns:
        Path to the created .ass file
    """
    # Calculate font size based on video width
    font_size = int(width * font_size_ratio)
    font_family = "Bebas Neue"
    
    # Animation settings - MATCHING overlay.py EXACTLY
    stagger_delay = 0.06  # Delay between each line appearing (same as overlay.py)
    anim_duration = 400   # Animation duration in ms (same as overlay.py)
    
    # Position calculation (30% from bottom)
    base_y = int(height * (1 - position_from_bottom_ratio))
    
    # Line spacing
    line_height = int(font_size * 1.3)
    
    # Colors in ASS format (&HAABBGGRR format - note: it's BGR not RGB!)
    white_color = "&H00FFFFFF"  # White
    red_color = "&H000000FF"    # Red (for $ only)
    
    # Style with NO outline, NO shadow, NO border - clean text only
    # BorderStyle=1 with Outline=0 and Shadow=0 = no border
    # Alignment=1 = left align
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_family},{font_size},{white_color},&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,1,{left_margin},10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    def fmt_time(t: float) -> str:
        """Format time for ASS format."""
        h = int(t / 3600)
        m = int((t % 3600) / 60)
        s = int(t % 60)
        cs = int((t * 100) % 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
    
    events = []
    
    # Prepare lines
    lines = [
        speaker_info.name,
        speaker_info.title,
        speaker_info.net_worth
    ]
    
    # Calculate total height of all lines
    total_lines_height = len(lines) * line_height
    
    # Start Y position (center the block at the specified position)
    start_y = base_y - (total_lines_height // 2)
    
    # X position for left alignment
    x_pos = left_margin
    
    # Character index counter for staggered animation across all lines
    char_index = 0
    
    for line_idx, line_text in enumerate(lines):
        y_end = start_y + (line_idx * line_height)
        
        # Process each character in the line
        char_x = x_pos
        
        # Estimate character width (for positioning)
        char_width = font_size * 0.6  # Approximate width for monospace-ish positioning
        
        for char_idx, char in enumerate(line_text):
            if char == " ":
                char_x += char_width * 0.5  # Smaller space
                char_index += 1
                continue
            
            # Calculate timing with stagger - EXACT same as overlay.py
            delay = (char_index * stagger_delay) + random.uniform(-0.01, 0.01)
            char_start = start_time_seconds + delay
            char_end = char_start + display_duration
            
            t_start = fmt_time(char_start)
            t_end = fmt_time(char_end)
            
            # Animation values - EXACT same as overlay.py
            drop_distance = 150 + random.randint(-20, 20)
            y_start = y_end + drop_distance
            
            # Determine if this character is the $ symbol (make it red)
            if char == '$':
                # Red dollar sign - use inline color override
                char_content = f"{{\\c{red_color}}}{char}{{\\c{white_color}}}"
            else:
                char_content = char
            
            # Create dialogue line with EXACT animation from overlay.py
            # \move(x1, y1, x2, y2, t1, t2) -> Move from start to end position
            # \fad(t1, 0) -> Fade in during t1 ms, no fade out
            # \blur20\t(0,anim_duration,\blur0) -> Start blurred, animate to sharp
            line = (
                f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,"
                f"{{\\pos({char_x},{y_end})"
                f"\\move({char_x},{y_start},{char_x},{y_end},0,{anim_duration})"
                f"\\fad({anim_duration},0)"
                f"\\blur20\\t(0,{anim_duration},\\blur0)"
                f"}}{char_content}"
            )
            events.append(line)
            
            char_x += char_width
            char_index += 1
    
    # Write the file
    with open(output_filename, "w", encoding='utf-8') as f:
        f.write(header + "\n".join(events))
    
    print(f"[INFO] Speaker info .ass file created: {output_filename}")
    print(f"[INFO] Font size: {font_size}px ({font_size_ratio*100:.0f}% of {width}px width)")
    print(f"[INFO] Position: {position_from_bottom_ratio*100:.0f}% from bottom")
    print(f"[INFO] Left-aligned with margin: {left_margin}px")
    
    return output_filename


def apply_speaker_overlay_ffmpeg(
    input_video: str,
    ass_file: str,
    output_video: str,
    ffmpeg_path: str = "ffmpeg"
) -> bool:
    """
    Apply the .ass subtitle overlay to a video using FFmpeg.
    
    Args:
        input_video: Input video path
        ass_file: Path to the .ass subtitle file
        output_video: Output video path
        ffmpeg_path: Path to FFmpeg executable
        
    Returns:
        True if successful, False otherwise
    """
    import subprocess
    import os
    
    # Escape the ass file path for FFmpeg filter
    # On Windows, we need to escape colons and backslashes
    ass_escaped = ass_file.replace('\\', '/').replace(':', '\\:')
    
    cmd = [
        ffmpeg_path,
        '-y',  # Overwrite output
        '-i', input_video,
        '-vf', f"ass='{ass_escaped}'",
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-c:a', 'copy',
        '-movflags', '+faststart',
        output_video
    ]
    
    try:
        print(f"[INFO] Applying speaker overlay with FFmpeg...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[INFO] Speaker overlay applied successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg failed: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"[ERROR] FFmpeg not found at: {ffmpeg_path}")
        return False


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    # Test the speaker info overlay
    speaker = SpeakerInfo(
        name="Jack Zhang",
        title="Co-founder of Airwallex",
        net_worth="($8,000,000,000 Net Worth)"
    )
    
    # Create for a typical vertical video (1080x1920)
    create_speaker_info_ass(
        speaker_info=speaker,
        output_filename="test_speaker.ass",
        width=1080,
        height=1920
    )
    
    print("\nTest .ass file created! Use with FFmpeg:")
    print('ffmpeg -i input.mp4 -vf "ass=test_speaker.ass" output.mp4')
