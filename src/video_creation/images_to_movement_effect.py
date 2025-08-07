import numpy as np
from moviepy import *
import math
from PIL import Image
import os
import random

DEFAULT_DURATION=6
MOVEMENT= "movement"  # Default movement type for output file naming
class ClipMaker:
    def __init__(
        self,
        image_path,
        output_path=None,
        duration=10,
        fps=30,
        zoom_ratio=0.05,
    ):
        self.image_path = image_path
        self.duration = duration
        self.fps = fps
        self.zoom_ratio = zoom_ratio

        # Set the output path based on the provided output_path or the image file name
        if output_path is None:
            base_dir = os.path.dirname(image_path)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            movement_dir = os.path.join(base_dir, MOVEMENT)
            os.makedirs(movement_dir, exist_ok=True)
            self.output_path = os.path.join(movement_dir, f"{base_name}_zoomed.mp4")
        else:
            self.output_path = output_path

    def zoom_in_effect(self, get_frame, t):
        """
        Apply a zoom-in effect to an image frame.

        Parameters:
        - get_frame: Function to get the frame at time t.
        - t: Time in seconds.

        Returns:
        - Zoomed image as a NumPy array.
        """
        img = Image.fromarray(get_frame(t))
        base_size = img.size

        # Calculate new size based on zoom ratio and time
        new_size = [
            math.ceil(base_size[0] * (1 + (self.zoom_ratio * t))),
            math.ceil(base_size[1] * (1 + (self.zoom_ratio * t))),
        ]

        # Ensure the new dimensions are even
        new_size[0] += new_size[0] % 2
        new_size[1] += new_size[1] % 2

        # Resize the image
        img = img.resize(new_size, Image.LANCZOS)

        # Calculate cropping coordinates to maintain the original size
        x = (new_size[0] - base_size[0]) // 2
        y = (new_size[1] - base_size[1]) // 2

        # Crop and resize back to the original size
        img = img.crop((x, y, x + base_size[0], y + base_size[1]))
        result = np.array(img)
        img.close()

        return result

    def create_zoom_in_clip(self, duration=None, output_path=None):
        """
        Create a video clip with a zoom-in effect.

        Parameters:
        - duration: Duration of the video clip in seconds. If None, use the instance's duration.
        - output_path: Path to save the output video file. If None, use the instance's output_path.
        """
        if duration is None:
            duration = self.duration

        if output_path is None:
            output_path = self.output_path

        # Load the image
        image_clip = ImageClip(self.image_path, duration=duration)
        image_clip = image_clip.with_fps(self.fps)

        # Apply the zoom-in effect using the transform method
        zoomed_clip = image_clip.transform(
            lambda get_frame, t: self.zoom_in_effect(get_frame, t)
        )

        # Write the result to a video file
        zoomed_clip.write_videofile(output_path, codec="libx264", fps=self.fps)

    def shake_effect(self, get_frame, t):
        """
        Apply a shake effect by shifting the image randomly at each frame.

        Returns:
        - Shaken image as a NumPy array.
        """
        img = Image.fromarray(get_frame(t))
        width, height = img.size
        max_shift = 10
        shift_x = int(random.uniform(-max_shift, max_shift) * math.sin(t * 2))
        shift_y = int(random.uniform(-max_shift, max_shift) * math.cos(t * 2))
        new_img = Image.new("RGB", (width, height))
        new_img.paste(img, (shift_x, shift_y))
        return np.array(new_img)

    def create_shake_clip(self, duration=None, output_path=None):
        """
        Create a video clip with a shake effect.
        """
        if duration is None:
            duration = self.duration
        if output_path is None:
            output_path = self.output_path.replace(".mp4", "_shake.mp4")
        image_clip = ImageClip(self.image_path, duration=duration).with_fps(self.fps)
        shaken_clip = image_clip.transform(lambda gf, t: self.shake_effect(gf, t))
        shaken_clip.write_videofile(output_path, codec="libx264", fps=self.fps)

    def rotation_effect(self, get_frame, t):
        """
        Apply a rotation effect that rotates left to right, then back to center.

        Returns:
        - Rotated image as a NumPy array.
        """
        img = Image.fromarray(get_frame(t))
        max_angle = 20 * 0.4  # maximum rotation in degrees (left/right), scaled down by 0.4
        # Oscillate angle: goes from 0 -> max_angle -> -max_angle -> 0 over duration
        progress = t / self.duration
        angle = max_angle * math.sin(math.pi * progress)
        rotated = img.rotate(angle, expand=False, resample=Image.BICUBIC)
        return np.array(rotated)

    def create_rotation_clip(self, duration=None, output_path=None):
        """
        Create a video clip with a rotation effect.
        """
        if duration is None:
            duration = self.duration
        if output_path is None:
            output_path = self.output_path.replace(".mp4", "_rotate.mp4")
        image_clip = ImageClip(self.image_path, duration=duration).with_fps(self.fps)
        rotated_clip = image_clip.transform(lambda gf, t: self.rotation_effect(gf, t))
        rotated_clip.write_videofile(output_path, codec="libx264", fps=self.fps)

    def pan_effect(self, get_frame, t):
        """
        Apply a pan effect (Ken Burns style). Move the image faster across the frame.
        """
        img = Image.fromarray(get_frame(t))
        width, height = img.size
        pan_distance = 150  # increased horizontal pan in pixels for faster movement
        progress = t / self.duration
        shift_x = int(pan_distance * progress)
        shift_y = 0
        new_img = Image.new("RGB", (width, height))
        new_img.paste(img, (shift_x, shift_y))
        return np.array(new_img)

    def create_pan_clip(self, duration=None, output_path=None):
        """
        Create a video clip with a pan effect.
        """
        if duration is None:
            duration = self.duration
        if output_path is None:
            output_path = self.output_path.replace(".mp4", "_pan.mp4")
        image_clip = ImageClip(self.image_path, duration=duration).with_fps(self.fps)
        pan_clip = image_clip.transform(lambda gf, t: self.pan_effect(gf, t))
        pan_clip.write_videofile(output_path, codec="libx264", fps=self.fps)

    def create_random_effect(self, duration=None, output_path=None):
        """
        Create a video clip with a randomly selected effect (zoom, shake, rotation, or pan).
        """
        import random
        effects = [
            (self.create_zoom_in_clip, '_zoomed.mp4'),
            (self.create_shake_clip, '_shake.mp4'),
            (self.create_rotation_clip, '_rotate.mp4'),
            (self.create_pan_clip, '_pan.mp4'),
        ]
        effect_func, suffix = random.choice(effects)
        if output_path is None:
            base, _ = os.path.splitext(self.output_path)
            output_path = base + suffix
        effect_func(duration=duration, output_path=output_path)
        print(f"Created clip with effect: {output_path}")


def get_image_paths(directory,mark=True):
    """
    Get a list of image paths from the specified directory.

    Parameters:
    - directory: Path to the directory containing images.

    Returns:
    - List of image paths.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    images =  [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.lower().endswith(valid_extensions)
    ]
    if mark:
        images.sort(key=lambda x: os.path.getmtime(x))
        for i, image in enumerate(images):
            new_name = os.path.join(directory, f"{i+1}{os.path.splitext(image)[1]}")
            os.rename(image, new_name)
        return [os.path.join(directory, f"{i+1}{os.path.splitext(image)[1]}") for i, image in enumerate(images)]


# Example usage for getting image paths
# if __name__ == "__main__":
#     image_directory = r"C:\Side\VidAI-main\fear"
#     image_paths = get_image_paths(image_directory)
#     for image_path in image_paths:
#         clip_maker = ClipMaker(image_path)
#         clip_maker.create_zoom_in_clip(duration=DEFAULT_DURATION)
