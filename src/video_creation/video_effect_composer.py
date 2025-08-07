"""
Given directory contains zoomed clips or set of clips 

Compose effects of green screen or black screen.
- Randomly
- By config and AI selection
"""

from moviepy import *
from moviepy.video.fx import MaskColor
from moviepy import VideoFileClip, CompositeVideoClip
import cv2
import numpy as np
import os
import random
from video_creation.utils.config_loader import VideoConfig

config = VideoConfig()


class VideoComposer:
    def __init__(self, videos_path):
        self.videos_path = videos_path
        self.videos = [
            os.path.join(videos_path, f).replace("\\", "/")
            for f in os.listdir(videos_path)
            if f.endswith(".mp4")
        ]

    def randomize_effects(self, output_dir):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        effects_list = [
            os.path.join(config.effects_path, f).replace("\\", "/")
            for f in os.listdir(config.effects_path)
            if f.endswith(".mp4")
        ]

        for video in self.videos:
            effect_video_path = random.choice(effects_list)
            output_path = os.path.join(output_dir, os.path.basename(video)).replace(
                "\\", "/"
            )
            if "green" in effect_video_path.lower():
                self.apply_green_screen_to_videos(
                    video, effect_video_path, output_path, green_offset=0
                )
            else:
                self.add_effect(video, effect_video_path, output_path)

    def add_effect(self, main_video_path, effect_video_path, output_path):
        main_clip = VideoFileClip(main_video_path)
        effect_clip = (
            VideoFileClip(effect_video_path)
            .with_duration(main_clip.duration)
            .resized(main_clip.size)
            .with_opacity(0.4)
        )
        composed_clip = CompositeVideoClip(
            [main_clip, effect_clip.with_position(("center", "center"))]
        )
        composed_clip.write_videofile(output_path, codec="libx264", fps=24)

    def apply_green_screen_to_videos(
        self, green_screen_path, background_path, output_path, green_offset=0
    ):
        green_screen_video = VideoFileClip(green_screen_path)
        background_video = VideoFileClip(background_path)
        min_duration = min(green_screen_video.duration, background_video.duration)
        green_screen_video = green_screen_video.subclipped(
            green_offset, min_duration + green_offset
        )
        background_video = background_video.subclipped(0, min_duration)
        final_video = green_screen_video.transform(
            lambda gf, t: apply_green_screen(gf(t), background_video.get_frame(t))
        )
        final_video.write_videofile(output_path, codec="libx264")


def create_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    mask_inv_rgb = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2RGB)
    mask_inv_rgb = mask_inv_rgb / 255.0
    return mask_inv_rgb


def apply_green_screen(frame, background_frame):
    background_frame = cv2.resize(background_frame, (frame.shape[1], frame.shape[0]))
    mask = create_mask(frame)
    foreground = frame * mask
    background = background_frame * (1 - mask)
    combined_frame = cv2.add(foreground, background)
    return combined_frame.astype(np.uint8)
