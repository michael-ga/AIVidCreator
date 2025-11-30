import math
import numpy as np
import moviepy.video.fx as vfx
# ייבוא ImageClip מכאן עוזר לוודא שהפילטרים נטענים
from moviepy.video.VideoClip import VideoClip, ImageClip


def anim_fade_in(clip, duration, **kwargs):
    """הופעה הדרגתית של הטקסט."""
    # משך ה-fadein הוא שליש מה-duration הכולל
    return clip.with_effects([vfx.CrossFadeIn(duration / 3)])


def anim_fade_out(clip, duration, **kwargs):
    """הופעה הדרגתית של הטקסט."""
    # משך ה-fadein הוא שליש מה-duration הכולל
    return clip.with_effects([vfx.CrossFadeOut(duration / 3)])


def anim_slide_in(clip, duration, **kwargs):
    """הופעה הדרגתית של הטקסט."""
    # משך ה-fadein הוא שליש מה-duration הכולל
    return clip.with_effects([vfx.SlideIn(duration / 3, 'top')])


def anim_slide_up(clip, duration, **kwargs):
    """החלקת טקסט מלמטה למיקום הסופי."""
    w, h = clip.size
    screen_h = kwargs.get('screen_h', 1080)
    target_pos = kwargs.get('pos', ('center', 'center'))

    # Determine final Y
    final_y = screen_h / 2 - h / \
        2 if target_pos == 'center' or target_pos[1] == 'center' else target_pos[1]
    if isinstance(final_y, str):
        final_y = screen_h / 2

    start_y = screen_h + 50

    def pos_func(t):
        t = min(t, 1.0)
        progress = 1 - (1 - t) ** 3
        current_y = start_y + (final_y - start_y) * progress
        x = 'center' if isinstance(
            target_pos, str) or target_pos[0] == 'center' else target_pos[0]
        return (x, current_y)

    return clip.with_position(pos_func)


def anim_pulse(clip, duration, **kwargs):
    """אפקט פעימה עדינה (שינוי גודל)."""
    return clip.with_effects([vfx.Resize(lambda t: 1 + 0.05 * math.sin(2 * math.pi * t))])


def anim_typewriter(clip, duration, **kwargs):
    """
    אנימציית מכונת כתיבה באמצעות חשיפה הדרגתית של הטקסט.
    הפילטר מופעל על ה-make_frame של הקליפ כדי לעקוף בעיות fl/fx ב-ImageClip.
    """
    # ברירת מחדל: הכתיבה מסתיימת ב-70% מזמן הקליפ
    typing_duration = kwargs.get('type_duration', duration * 0.7)

    def frame_level_typewriter_mask(get_frame, t):
        """
        פונקציית פילטר ברמת הפריימ ליצירת אפקט מכונת כתיבה.
        """
        frame = get_frame(t)
        h, w, c = frame.shape

        # חישוב התקדמות (0.0 עד 1.0)
        progress = min(t / typing_duration, 1.0)

        # חישוב רוחב נראה בפיקסלים
        visible_w = int(w * progress)

        new_frame = frame.copy()

        # אם יש ערוץ אלפא (שקיפות), אנו משתמשים בו כדי להסתיר את החלקים הבלתי נראים.
        if c == 4:
            # בשלב אפס או לפני תחילת ההקלדה, כל ה-Alpha הוא 0.
            if t == 0 or progress == 0:
                new_frame[:, :, 3] = 0
            # חשיפת הטקסט משמאל לימין על ידי איפוס ערוץ האלפא (שקיפות) באזור החתוך.
            else:
                if visible_w < w:
                    new_frame[:, visible_w:, 3] = 0

        return new_frame

    # החלת פונקציית הפילטר ברמת הפריימ
    return clip.fl(frame_level_typewriter_mask)


# Registry mapping names to functions
ANIMATIONS = {
    'fade_in': anim_fade_in,
    'fade_out': anim_fade_out,
    'slide_up': anim_slide_up,
    'pulse': anim_pulse,
    'typewriter': anim_typewriter,
    'slide_in': anim_slide_in,
    # שימוש ב-fade_in גם עבור special_outline
    'special_outline': anim_fade_in,
    'static': lambda c, d, **k: c
}


def get_animation(name):
    """Safe getter for animations"""
    return ANIMATIONS.get(name, ANIMATIONS['static'])
