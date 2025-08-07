import os
from moviepy import *

def create_caption_text_clips(text_file_path, output_dir, duration=5, size=(720, 1280), fontsize=48, color='white', bg_color='black'):
    """
    Create video clips from lines of text in a file. Each line becomes a video clip.
    If the file does not exist, create a sample file and use it.
    """
    if not os.path.exists(text_file_path):
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write("Sample caption 1\nSample caption 2\nSample caption 3\n")
    with open(text_file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    os.makedirs(output_dir, exist_ok=True)
    clips = []
    for i, line in enumerate(lines):
        txt_clip = TextClip(
            line,
            font_size=fontsize,
            font='BOOKOSB.TTF',
            color=color, 
            size=size,
            bg_color=bg_color,
            method='caption',
            text_align='center',
            stroke_color='black',
            stroke_width=4
        )
        txt_clip = txt_clip.with_duration(duration)
        out_path = os.path.join(output_dir, f"caption_{i+1}.mp4")
        txt_clip.write_videofile(out_path, fps=24)
        clips.append(txt_clip)
    # Optionally concatenate all caption clips into one video
    final = concatenate_videoclips(clips)
    final_out = os.path.join(output_dir, "all_captions.mp4")
    final.write_videofile(final_out, fps=24)
    return final_out
