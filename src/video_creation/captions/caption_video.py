import os
import textwrap
# Updated import for MoviePy v2 compatibility
from moviepy import ImageClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Import animation logic from separate file
from animation_engine import get_animation

# --- ייבוא לספריות עיבוד טקסט Bidi (מימין לשמאל) ---
try:
    from bidi.algorithm import get_display
    from arabic_reshaper import reshape
    BIDI_SUPPORT = True
except ImportError:
    # Print warning if the libraries are missing, but continue execution
    print("Warning: python-bidi or arabic-reshaper not found. Hebrew/Arabic text might not render correctly (RTL issue).")
    BIDI_SUPPORT = False
# ----------------------------------------------------


def get_best_font(font_path, fontsize):
    """Attempts to load a scalable font."""
    if font_path:
        try:
            return ImageFont.truetype(font_path, fontsize)
        except IOError:
            pass

    common_fonts = [
        "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf",
        "FreeSans.ttf", "seguiemj.ttf", "AppleGothic.ttf"
    ]

    for f_name in common_fonts:
        try:
            return ImageFont.truetype(f_name, fontsize)
        except IOError:
            continue

    print("Warning: No scalable font found.")
    return ImageFont.load_default()


def create_text_image(text, size, fontsize=50, color='white', font_path=None, stroke_only=False, stroke_fill_color=None, stroke_width_override=None):
    """
    Creates a transparent image with text using Pillow.
    If stroke_only is True, the fill color is transparent.
    """

    # --- 1. עיבוד טקסט Bidi (עברית / RTL) ---
    if BIDI_SUPPORT:
        # עיבוד ה-reshaper (למעשה עושה את רוב העבודה)
        reshaped_text = reshape(text)
        # שינוי כיוון הצגה (להימנע מהיפוך מילים)
        processed_text = get_display(reshaped_text)
    else:
        processed_text = text
    # ---------------------------------------

    img = Image.new('RGBA', size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    font = get_best_font(font_path, fontsize)
    W, H = size

    # Auto-Wrap Logic
    try:
        avg_char_width = font.getlength("A")
    except AttributeError:
        # Use draw.textlength for older Pillow versions
        avg_char_width = draw.textlength("A", font=font)

    if avg_char_width == 0:
        avg_char_width = 10
    max_chars = int((W * 0.9) / avg_char_width)

    # 2. עטיפת הטקסט (Text Wrapping) - על הטקסט המעובד
    wrapped_text = textwrap.fill(processed_text, width=max_chars)

    # Calculate dynamic stroke width first
    outline_color = stroke_fill_color if stroke_fill_color is not None else (
        'black' if color.lower() in ['white', '#ffffff'] else 'white')
    stroke_width = stroke_width_override if stroke_width_override is not None else max(
        2, int(fontsize / 15))

    # Determine Fill Color
    # Transparent fill for stroke_only
    fill_color = (255, 255, 255, 0) if stroke_only else color

    # Measure with stroke_width included
    try:
        left, top, right, bottom = draw.multiline_textbbox(
            (0, 0),
            wrapped_text,
            font=font,
            align='center',
            stroke_width=stroke_width
        )
        text_w, text_h = right - left, bottom - top
    except AttributeError:
        text_w, text_h = draw.textsize(wrapped_text, font=font)
        text_w += stroke_width * 2
        text_h += stroke_width * 2

    # Create canvas with padding
    canvas_w = int(text_w + 40)
    canvas_h = int(text_h + 40)

    text_canvas = Image.new('RGBA', (canvas_w, canvas_h), (255, 255, 255, 0))
    draw_canvas = ImageDraw.Draw(text_canvas)

    # Draw centered
    draw_canvas.multiline_text(
        (20, 20),
        wrapped_text,
        font=font,
        fill=fill_color,  # Use fill_color (transparent or solid)
        stroke_width=stroke_width,
        stroke_fill=outline_color,
        align='center'
    )

    return np.array(text_canvas)


def generate_video_from_image(image_path, output_path, text_content, duration=5, style_config=None, animation_type='static'):
    """Generates video with animated text."""
    if not style_config:
        style_config = {}

    # 1. Background
    background_clip = ImageClip(image_path).with_duration(duration)

    # 2. Text Generation Setup
    default_fontsize = int(background_clip.h / 10)
    font_size = style_config.get('fontsize', default_fontsize)
    text_color = style_config.get('color', 'white')
    font_path = style_config.get('font_path', None)

    # 3. Positioning Logic (applies to all text clips)
    pos_config = style_config.get('position', 'center')
    screen_w, screen_h = background_clip.size

    final_x = 'center'
    final_y = 'center'

    if pos_config == 'bottom':
        # final_y is calculated as a number here
        final_y = screen_h - (background_clip.h * 0.15)
    elif pos_config == 'top':
        # final_y is calculated as a number here
        final_y = background_clip.h * 0.1
    elif isinstance(pos_config, tuple):
        # final_x and final_y are whatever the user supplied (could be numbers or strings)
        final_x, final_y = pos_config

    clips_to_composite = [background_clip]
    anim_func = get_animation(animation_type)

    # --- SPECIAL EFFECT: Gold Outline/Shadow (Now fades in completely) ---
    if animation_type == 'special_outline':

        # --- Shadow Offset Logic ---
        x_offset = 5
        y_offset = 5

        shadow_x_pos = final_x
        shadow_y_pos = final_y

        # If final_x/y is numeric (from 'top', 'bottom', or (num, num) tuple), apply offset.
        if isinstance(final_x, (int, float)):
            shadow_x_pos = final_x + x_offset

        if isinstance(final_y, (int, float)):
            shadow_y_pos = final_y + y_offset

        # --- Layer 1: Black Shadow Base (Thickest Stroke, Transparent Fill) ---
        shadow_array = create_text_image(
            text=text_content, size=background_clip.size, fontsize=font_size,
            stroke_only=True, stroke_fill_color='#333333', stroke_width_override=int(font_size / 8)
        )
        shadow_clip = ImageClip(shadow_array).with_duration(duration)
        shadow_clip = shadow_clip.with_position((shadow_x_pos, shadow_y_pos))

        # --- Layer 2: Gold/Yellow Outline (Medium Stroke, Transparent Fill) ---
        gold_array = create_text_image(
            text=text_content, size=background_clip.size, fontsize=font_size,
            stroke_only=True, stroke_fill_color='#FFD700', stroke_width_override=int(font_size / 20)
        )
        gold_clip = ImageClip(gold_array).with_duration(duration)
        gold_clip = gold_clip.with_position(
            (final_x, final_y))  # Base position (no offset)

        # --- Layer 3: White Fill (Thin Stroke, Solid Fill) ---
        fill_array = create_text_image(
            text=text_content, size=background_clip.size, fontsize=font_size,
            # Minimal stroke for clean look
            color='white', stroke_fill_color='#FFD700', stroke_width_override=1
        )
        fill_clip = ImageClip(fill_array).with_duration(duration)
        fill_clip = fill_clip.with_position(
            (final_x, final_y))  # Base position (no offset)

        # --- APPLY ANIMATION TO ALL LAYERS (Fade In) ---
        if anim_func:
            # Apply the same animation function (fade_in) to all layers
            shadow_clip = anim_func(
                shadow_clip, duration, screen_h=screen_h, pos=(shadow_x_pos, shadow_y_pos))
            gold_clip = anim_func(gold_clip, duration,
                                  screen_h=screen_h, pos=(final_x, final_y))
            fill_clip = anim_func(fill_clip, duration,
                                  screen_h=screen_h, pos=(final_x, final_y))

        clips_to_composite.append(shadow_clip)
        clips_to_composite.append(gold_clip)
        clips_to_composite.append(fill_clip)

    # --- Typewriter Effect (Fixed: now animates the full text, stroke included) ---
    elif animation_type == 'typewriter':
        # Create single full text clip (fill + stroke)
        txt_img_array = create_text_image(
            text=text_content,
            size=background_clip.size,
            fontsize=font_size,
            color=text_color,
            font_path=font_path
        )
        txt_clip = ImageClip(txt_img_array).with_duration(
            duration).with_position((final_x, final_y))

        # Note: The static stroke_clip is REMOVED, forcing the animation to reveal everything.

        # Apply Typewriter animation to the regular text clip (the full text)
        if anim_func:
            animated_clip = anim_func(
                txt_clip,
                duration,
                screen_h=screen_h,
                pos=(final_x, final_y)
            )
            clips_to_composite.append(animated_clip)

    # --- Standard Animation (Fade In, Slide Up, Pulse) ---
    else:
        # Create full text clip
        txt_img_array = create_text_image(
            text=text_content,
            size=background_clip.size,
            fontsize=font_size,
            color=text_color,
            font_path=font_path
        )
        txt_clip = ImageClip(txt_img_array).with_duration(
            duration).with_position((final_x, final_y))

        # Apply Animation
        if anim_func:
            txt_clip = anim_func(
                txt_clip,
                duration,
                screen_h=screen_h,
                pos=(final_x, final_y)
            )
        clips_to_composite.append(txt_clip)

    # 5. Render
    final_video = CompositeVideoClip(clips_to_composite)

    # Adjust final position based on clip size if 'center' or 'bottom' was used
    if pos_config == 'bottom':
        # Re-calculate position after clip is fully rendered (MoviePy handles 'center' automatically)
        pass  # MoviePy handles position when clips are added to CompositeVideoClip

    final_video.write_videofile(
        output_path, fps=24, codec='libx264', audio=False)
    print(f"Video saved to: {output_path}")


# Example Usage
# if __name__ == "__main__":
#     text1 = 'לכל אחד מגיע, קצת מנוחה..'
#     text2 = 'השלך לים את כל הדאגות ויהיה בסדר'
#     image1 = "C:\\Gituhb\\AI\\AIVidCreator-main\\src\\video_creation\\captions\\openart-image_VwnlIfm6_1750794067702_raw.jpg"
#     image2 = "C:\\Gituhb\\AI\\AIVidCreator-main\\src\\video_creation\\captions\\openart-image_XBTy9gP8_1750794017015_raw.jpg"
#     images = [image1, image2]
#     texts = [text1, text2]
#     for image, text in zip(images, texts):
#         generate_video_from_image(image, image.replace(
#             ".jpg", ".mp4"), text_content=text, animation_type='slide_in')
#     # Ensure you have an image named 'input.jpg' or change this path
#     generate_video_from_image('input.jpg', 'output.mp4', "Hello World", style_config={'position': 'bottom', 'color': 'yellow'})
