import random

def create_fast_cinematic_ass(text, output_filename="fast_cinematic.ass", width=1920, height=1080):
    # --- CONFIGURATION ---
    font_size = 120
    font_family = "Arial"
    start_time_seconds = 1.0 
    letter_spacing = 80
    
    # SPEED SETTINGS
    # Lower = Faster sequence (0.05 is very fast, 0.2 is slow)
    stagger_delay = 0.05
    
    # How fast one letter flies up and focuses (in milliseconds)
    # 300ms = snappy/fast, 1000ms = slow/dreamy
    anim_duration = 300
    
    # ---------------------

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_family},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    events = []
    total_width = len(text) * letter_spacing
    start_x = (width - total_width) / 2
    base_y = height / 2

    for i, char in enumerate(text):
        if char == " ":
            continue
            
        # 1. Faster Timing Calculation
        delay = (i * stagger_delay) + random.uniform(-0.01, 0.01)
        char_start = start_time_seconds + delay
        char_end = char_start + 4.0 
        
        def fmt_time(t):
            h = int(t / 3600)
            m = int((t % 3600) / 60)
            s = int(t % 60)
            cs = int((t * 100) % 100)
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        t_start = fmt_time(char_start)
        t_end = fmt_time(char_end)
        
        drop_distance = 150 + random.randint(-20, 20)
        y_start = base_y + drop_distance
        y_end = base_y
        x_pos = start_x + (i * letter_spacing)
        
        # 2. Updated ASS Tags for Speed
        # \move(x1, y1, x2, y2, t1, t2) -> Moves explicitly within the first few ms
        # \fad(t1, 0) -> Fades in quickly
        # \t(t1, t2, \blur0) -> Blurs out quickly
        
        line = (
            f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,"
            f"{{\\move({x_pos},{y_start},{x_pos},{y_end},0,{anim_duration})" 
            f"\\fad({anim_duration},0)"
            f"\\blur20\\t(0,{anim_duration},\\blur0)" 
            f"}}{char}"
        )
        events.append(line)

    with open(output_filename, "w", encoding='utf-8') as f:
        f.write(header + "\n".join(events))
    
    print(f"File '{output_filename}' created. Animation duration: {anim_duration}ms per letter.")

# --- CHANGE YOUR TEXT HERE ---
create_fast_cinematic_ass("FAST MOTION")