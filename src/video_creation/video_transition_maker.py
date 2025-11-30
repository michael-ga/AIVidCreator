"""
Video Transition Maker

This module provides a flexible class for applying transition effects between video clips.
It's designed to work after videos are created with movement effects and voiceover.

Usage:
    transition_maker = VideoTransitionMaker(transition_type='crossfade', duration=1.0)
    final_video = transition_maker.combine_videos(video_paths, output_path)
"""

import os
from typing import List, Optional, Callable, Dict
from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
try:
    import moviepy.video.fx.all as vfx
except Exception:
    try:
        import moviepy.video.fx as vfx
    except Exception:
        vfx = None
try:
    import moviepy.audio.fx as afx
except Exception:
    afx = None
import numpy as np


class VideoTransitionMaker:
    """
    A flexible class for applying transition effects between video clips.

    Designed to make it easy to add new transition implementations.
    """

    # Registry of available transition types
    TRANSITION_REGISTRY: Dict[str, str] = {
        'crossfade': 'Crossfade transition (fade out + fade in)',
        'fadein': 'Fade in transition for second clip',
        'fadeout': 'Fade out transition for first clip',
        'slide_left': 'Slide transition from left',
        'slide_right': 'Slide transition from right',
        'slide_up': 'Slide transition from top',
        'slide_down': 'Slide transition from bottom',
        'zoom_in': 'Zoom in transition',
        'zoom_out': 'Zoom out transition',
        'wipe_left': 'Wipe transition from left',
        'wipe_right': 'Wipe transition from right',
        'dissolve': 'Dissolve transition (similar to crossfade)',
        'none': 'No transition (simple concatenation)'
    }

    def __init__(
        self,
        transition_type: str = 'crossfade',
        duration: float = 1.0,
        fps: Optional[int] = None
    ):
        """
        Initialize the VideoTransitionMaker.

        Args:
            transition_type: Type of transition to apply. See TRANSITION_REGISTRY for options.
            duration: Duration of the transition in seconds.
            fps: Frames per second for output video. If None, uses fps from first clip.
        """
        self.transition_type = transition_type.lower()
        self.duration = duration
        self.fps = fps

        if self.transition_type not in self.TRANSITION_REGISTRY:
            raise ValueError(
                f"Unknown transition type: {transition_type}. "
                f"Available types: {list(self.TRANSITION_REGISTRY.keys())}"
            )

    def combine_videos(
        self,
        video_paths: List[str],
        output_path: str,
        transition_duration: Optional[float] = None
    ) -> Optional[str]:
        """
        Combine multiple video clips with transition effects.

        Args:
            video_paths: List of paths to video files to combine.
            output_path: Path where the final video will be saved.
            transition_duration: Override the default transition duration for this operation.

        Returns:
            Path to the created video file, or None if creation failed.
        """
        if not video_paths:
            print("✗ No video paths provided")
            return None

        if len(video_paths) == 1:
            # Only one video, no transition needed
            return self._copy_single_video(video_paths[0], output_path)

        transition_duration = transition_duration or self.duration

        try:
            print(
                f"\n🎬 Combining {len(video_paths)} clips with '{self.transition_type}' transition...")

            # Load all video clips
            clips = []
            for i, path in enumerate(video_paths):
                if not os.path.exists(path):
                    print(
                        f"⚠ Warning: Video file not found: {path}, skipping...")
                    continue
                clip = VideoFileClip(path)
                clips.append(clip)
                print(
                    f"  Loaded clip {i + 1}/{len(video_paths)}: {os.path.basename(path)}")

            if not clips:
                print("✗ No valid video clips to combine")
                return None

            # Get fps from first clip if not specified
            fps = self.fps or clips[0].fps

            # Apply transitions between clips
            transitioned_clips = []
            for i in range(len(clips)):
                if i == 0:
                    # First clip - no transition before it
                    transitioned_clips.append(clips[i])
                else:
                    # Apply transition between previous and current clip
                    prev_clip = transitioned_clips[-1]
                    curr_clip = clips[i]

                    # Get the transition method
                    transition_method = self._get_transition_method()
                    transitioned_pair = transition_method(
                        prev_clip, curr_clip, transition_duration)

                    # Replace the last clip with the transitioned pair
                    transitioned_clips[-1] = transitioned_pair

            # Concatenate all clips
            if len(transitioned_clips) == 1:
                final_video = transitioned_clips[0]
            else:
                final_video = concatenate_videoclips(
                    transitioned_clips, method="compose")

            # Ensure consistent fps
            final_video = final_video.with_fps(fps)

            # Write the final video
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(
                output_path) else '.', exist_ok=True)
            final_video.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=fps
            )

            # Close all clips
            for clip in clips:
                clip.close()
            final_video.close()

            print(f"✓ Final video created: {output_path}")
            return output_path

        except Exception as e:
            print(f"✗ Error combining videos: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _copy_single_video(self, video_path: str, output_path: str) -> Optional[str]:
        """Copy a single video file to output path."""
        try:
            clip = VideoFileClip(video_path)
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(
                output_path) else '.', exist_ok=True)
            clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=clip.fps
            )
            clip.close()
            print(f"✓ Copied single video: {output_path}")
            return output_path
        except Exception as e:
            print(f"✗ Error copying video: {e}")
            return None

    def _get_transition_method(self) -> Callable:
        """Get the transition method based on transition_type."""
        transition_methods = {
            'crossfade': self._apply_crossfade,
            'fadein': self._apply_fadein,
            'fadeout': self._apply_fadeout,
            'slide_left': self._apply_slide_left,
            'slide_right': self._apply_slide_right,
            'slide_up': self._apply_slide_up,
            'slide_down': self._apply_slide_down,
            'zoom_in': self._apply_zoom_in,
            'zoom_out': self._apply_zoom_out,
            'wipe_left': self._apply_wipe_left,
            'wipe_right': self._apply_wipe_right,
            'dissolve': self._apply_dissolve,
            'none': self._apply_none
        }
        return transition_methods.get(self.transition_type, self._apply_crossfade)

    # ==================== Transition Implementations ====================

    def _apply_crossfade(self, clip1: VideoFileClip, clip2: VideoFileClip, duration: float) -> VideoFileClip:
        """
        Apply a crossfade transition (fade out first clip, fade in second clip).
        This is the most common and smooth transition.
        """
        # Ensure clips have the same size
        clip1, clip2 = self._match_clip_sizes(clip1, clip2)

        # Use helper which builds a CompositeVideoClip performing the crossfade
        return self._crossfade_composite(clip1, clip2, duration)

    def _apply_fadein(self, clip1: VideoFileClip, clip2: VideoFileClip, duration: float) -> VideoFileClip:
        """Apply fade in transition to the second clip."""
        clip1, clip2 = self._match_clip_sizes(clip1, clip2)

        if vfx is None:
            raise RuntimeError(
                "moviepy.video.fx is required for fade-in transitions.")

        fadein_cls = getattr(vfx, 'FadeIn', None) or getattr(
            vfx, 'CrossFadeIn', None)
        if fadein_cls is None:
            raise RuntimeError(
                "FadeIn function not available in moviepy.video.fx.")

        clip2_faded = clip2.with_effects([fadein_cls(duration)])

        if clip2.audio is not None and afx is not None:
            audio_fadein_cls = getattr(afx, 'AudioFadeIn', None)
            if audio_fadein_cls is None:
                raise RuntimeError(
                    "AudioFadeIn not available in moviepy.audio.fx.")
            clip2_faded = clip2_faded.with_audio(
                clip2.audio.with_effects([audio_fadein_cls(duration)]))

        return concatenate_videoclips([clip1, clip2_faded], method="compose")

    def _apply_fadeout(self, clip1: VideoFileClip, clip2: VideoFileClip, duration: float) -> VideoFileClip:
        """Apply fade out transition to the first clip."""
        clip1, clip2 = self._match_clip_sizes(clip1, clip2)

        if vfx is None:
            raise RuntimeError(
                "moviepy.video.fx is required for fade-out transitions.")

        fadeout_cls = getattr(vfx, 'FadeOut', None) or getattr(
            vfx, 'CrossFadeOut', None)
        if fadeout_cls is None:
            raise RuntimeError(
                "FadeOut function not available in moviepy.video.fx.")

        clip1_faded = clip1.with_effects([fadeout_cls(duration)])

        if clip1.audio is not None and afx is not None:
            audio_fadeout_cls = getattr(afx, 'AudioFadeOut', None)
            if audio_fadeout_cls is None:
                raise RuntimeError(
                    "AudioFadeOut not available in moviepy.audio.fx.")
            clip1_faded = clip1_faded.with_audio(
                clip1.audio.with_effects([audio_fadeout_cls(duration)]))

        return concatenate_videoclips([clip1_faded, clip2], method="compose")

    def _apply_dissolve(self, clip1: VideoFileClip, clip2: VideoFileClip, duration: float) -> VideoFileClip:
        """Apply a dissolve transition (similar to crossfade but with different timing)."""
        # Dissolve behaves like crossfade here
        return self._crossfade_composite(clip1, clip2, duration)

    def _crossfade_composite(self, clip1: VideoFileClip, clip2: VideoFileClip, duration: float) -> VideoFileClip:
        """Create a CompositeVideoClip that crossfades clip1 into clip2 over duration seconds.

        If moviepy fx FadeIn/FadeOut are available, use them to create smooth fades. Otherwise
        fall back to a simple overlay (no alpha ramp) to ensure the output is produced.
        """
        clip1, clip2 = self._match_clip_sizes(clip1, clip2)

        if vfx is None:
            raise RuntimeError(
                "moviepy.video.fx is required for crossfade transitions.")

        fadeout_cls = getattr(vfx, 'CrossFadeOut', None) or getattr(
            vfx, 'FadeOut', None)
        fadein_cls = getattr(vfx, 'CrossFadeIn', None) or getattr(
            vfx, 'FadeIn', None)
        if not (fadeout_cls and fadein_cls):
            raise RuntimeError(
                "Crossfade requires CrossFadeIn/CrossFadeOut (or FadeIn/FadeOut) in moviepy.video.fx.")

        audio_fadeout_cls = getattr(
            afx, 'AudioFadeOut', None) if afx is not None else None
        audio_fadein_cls = getattr(
            afx, 'AudioFadeIn', None) if afx is not None else None

        d = max(0.0, min(duration, clip1.duration, clip2.duration))
        if d <= 0:
            return concatenate_videoclips([clip1, clip2], method="compose")

        overlap_start = clip1.duration - d

        c1 = clip1.with_effects([fadeout_cls(d)])
        if clip1.audio is not None and audio_fadeout_cls is not None:
            c1 = c1.with_audio(
                clip1.audio.with_effects([audio_fadeout_cls(d)]))

        c2 = clip2.with_effects([fadein_cls(d)]).with_start(overlap_start)
        if clip2.audio is not None and audio_fadein_cls is not None:
            c2 = c2.with_audio(clip2.audio.with_effects([audio_fadein_cls(d)]))

        total_duration = clip1.duration + clip2.duration - d
        return CompositeVideoClip([c1, c2], size=clip1.size).with_duration(total_duration)

    def _apply_none(self, clip1: VideoFileClip, clip2: VideoFileClip, duration: float) -> VideoFileClip:
        """No transition - simple concatenation."""
        clip1, clip2 = self._match_clip_sizes(clip1, clip2)
        return concatenate_videoclips([clip1, clip2], method="compose")

    def _apply_slide_left(self, clip1: VideoFileClip, clip2: VideoFileClip, duration: float) -> VideoFileClip:
        """Slide transition: second clip slides in from the left."""
        clip1, clip2 = self._match_clip_sizes(clip1, clip2)
        w, h = clip1.size

        def slide_effect(get_frame, t):
            if t < duration:
                # During transition: clip2 slides in from left
                progress = t / duration
                x_offset = int((1 - progress) * w)
                frame1 = get_frame(t)
                frame2 = clip2.get_frame(t)

                # Create composite frame
                result = np.zeros_like(frame1)
                # Place clip1 (fading out)
                alpha1 = 1 - progress
                result = (result * (1 - alpha1) +
                          frame1 * alpha1).astype(np.uint8)
                # Place clip2 (sliding in from left)
                if x_offset < w:
                    result[:, x_offset:] = frame2[:, :w - x_offset]
            else:
                # After transition: show clip2
                return clip2.get_frame(t - duration)
            return result.astype(np.uint8)

        # Create transition clip
        transition_clip = clip1.transform(slide_effect, duration=duration)
        # Concatenate with remaining clip2
        clip2_remaining = clip2.subclipped(duration, clip2.duration)
        return concatenate_videoclips([transition_clip, clip2_remaining], method="compose")

    def _apply_slide_right(self, clip1: VideoFileClip, clip2: VideoFileClip, duration: float) -> VideoFileClip:
        """Slide transition: second clip slides in from the right."""
        clip1, clip2 = self._match_clip_sizes(clip1, clip2)
        w, h = clip1.size

        def slide_effect(get_frame, t):
            if t < duration:
                progress = t / duration
                x_offset = int(progress * w)
                frame1 = get_frame(t)
                frame2 = clip2.get_frame(t)

                result = np.zeros_like(frame1)
                alpha1 = 1 - progress
                result = (result * (1 - alpha1) +
                          frame1 * alpha1).astype(np.uint8)
                if x_offset > 0:
                    result[:, :x_offset] = frame2[:, w - x_offset:]
            else:
                return clip2.get_frame(t - duration)
            return result.astype(np.uint8)

        transition_clip = clip1.transform(slide_effect, duration=duration)
        clip2_remaining = clip2.subclipped(duration, clip2.duration)
        return concatenate_videoclips([transition_clip, clip2_remaining], method="compose")

    def _apply_slide_up(self, clip1: VideoFileClip, clip2: VideoFileClip, duration: float) -> VideoFileClip:
        """Slide transition: second clip slides in from the top."""
        clip1, clip2 = self._match_clip_sizes(clip1, clip2)
        w, h = clip1.size

        def slide_effect(get_frame, t):
            if t < duration:
                progress = t / duration
                y_offset = int((1 - progress) * h)
                frame1 = get_frame(t)
                frame2 = clip2.get_frame(t)

                result = np.zeros_like(frame1)
                alpha1 = 1 - progress
                result = (result * (1 - alpha1) +
                          frame1 * alpha1).astype(np.uint8)
                if y_offset < h:
                    result[y_offset:, :] = frame2[:h - y_offset, :]
            else:
                return clip2.get_frame(t - duration)
            return result.astype(np.uint8)

        transition_clip = clip1.transform(slide_effect, duration=duration)
        clip2_remaining = clip2.subclipped(duration, clip2.duration)
        return concatenate_videoclips([transition_clip, clip2_remaining], method="compose")

    def _apply_slide_down(self, clip1: VideoFileClip, clip2: VideoFileClip, duration: float) -> VideoFileClip:
        """Slide transition: second clip slides in from the bottom."""
        clip1, clip2 = self._match_clip_sizes(clip1, clip2)
        w, h = clip1.size

        def slide_effect(get_frame, t):
            if t < duration:
                progress = t / duration
                y_offset = int(progress * h)
                frame1 = get_frame(t)
                frame2 = clip2.get_frame(t)

                result = np.zeros_like(frame1)
                alpha1 = 1 - progress
                result = (result * (1 - alpha1) +
                          frame1 * alpha1).astype(np.uint8)
                if y_offset > 0:
                    result[:y_offset, :] = frame2[h - y_offset:, :]
            else:
                return clip2.get_frame(t - duration)
            return result.astype(np.uint8)

        transition_clip = clip1.transform(slide_effect, duration=duration)
        clip2_remaining = clip2.subclipped(duration, clip2.duration)
        return concatenate_videoclips([transition_clip, clip2_remaining], method="compose")

    def _apply_zoom_in(self, clip1: VideoFileClip, clip2: VideoFileClip, duration: float) -> VideoFileClip:
        """Zoom in transition: second clip zooms in while first fades out."""
        clip1, clip2 = self._match_clip_sizes(clip1, clip2)
        w, h = clip1.size

        def zoom_effect(get_frame, t):
            if t < duration:
                progress = t / duration
                zoom_factor = 1.0 + (0.3 * progress)  # Zoom from 1.0 to 1.3
                alpha1 = 1 - progress

                frame1 = get_frame(t)
                frame2 = clip2.get_frame(t)

                # Zoom clip2
                from PIL import Image
                img2 = Image.fromarray(frame2)
                new_w = int(w * zoom_factor)
                new_h = int(h * zoom_factor)
                # Use LANCZOS resampling (compatible with older PIL versions)
                try:
                    img2 = img2.resize(
                        (new_w, new_h), Image.Resampling.LANCZOS)
                except AttributeError:
                    img2 = img2.resize((new_w, new_h), Image.LANCZOS)
                x = (new_w - w) // 2
                y = (new_h - h) // 2
                img2 = img2.crop((x, y, x + w, y + h))
                frame2_zoomed = np.array(img2)

                # Composite
                result = (frame1 * alpha1 + frame2_zoomed *
                          (1 - alpha1)).astype(np.uint8)
            else:
                return clip2.get_frame(t - duration)
            return result

        transition_clip = clip1.transform(zoom_effect, duration=duration)
        clip2_remaining = clip2.subclipped(duration, clip2.duration)
        return concatenate_videoclips([transition_clip, clip2_remaining], method="compose")

    def _apply_zoom_out(self, clip1: VideoFileClip, clip2: VideoFileClip, duration: float) -> VideoFileClip:
        """Zoom out transition: first clip zooms out while second fades in."""
        clip1, clip2 = self._match_clip_sizes(clip1, clip2)
        w, h = clip1.size

        def zoom_effect(get_frame, t):
            if t < duration:
                progress = t / duration
                zoom_factor = 1.3 - (0.3 * progress)  # Zoom from 1.3 to 1.0
                alpha2 = progress

                frame1 = get_frame(t)
                frame2 = clip2.get_frame(t)

                # Zoom clip1
                from PIL import Image
                img1 = Image.fromarray(frame1)
                new_w = int(w * zoom_factor)
                new_h = int(h * zoom_factor)
                # Use LANCZOS resampling (compatible with older PIL versions)
                try:
                    img1 = img1.resize(
                        (new_w, new_h), Image.Resampling.LANCZOS)
                except AttributeError:
                    img1 = img1.resize((new_w, new_h), Image.LANCZOS)
                x = (new_w - w) // 2
                y = (new_h - h) // 2
                img1 = img1.crop((x, y, x + w, y + h))
                frame1_zoomed = np.array(img1)

                # Composite
                result = (frame1_zoomed * (1 - alpha2) +
                          frame2 * alpha2).astype(np.uint8)
            else:
                return clip2.get_frame(t - duration)
            return result

        transition_clip = clip1.transform(zoom_effect, duration=duration)
        clip2_remaining = clip2.subclipped(duration, clip2.duration)
        return concatenate_videoclips([transition_clip, clip2_remaining], method="compose")

    def _apply_wipe_left(self, clip1: VideoFileClip, clip2: VideoFileClip, duration: float) -> VideoFileClip:
        """Wipe transition: second clip wipes in from the left."""
        clip1, clip2 = self._match_clip_sizes(clip1, clip2)
        w, h = clip1.size

        def wipe_effect(get_frame, t):
            if t < duration:
                progress = t / duration
                wipe_x = int(progress * w)
                frame1 = get_frame(t)
                frame2 = clip2.get_frame(t)

                result = frame1.copy()
                result[:, :wipe_x] = frame2[:, :wipe_x]
            else:
                return clip2.get_frame(t - duration)
            return result

        transition_clip = clip1.transform(wipe_effect, duration=duration)
        clip2_remaining = clip2.subclipped(duration, clip2.duration)
        return concatenate_videoclips([transition_clip, clip2_remaining], method="compose")

    def _apply_wipe_right(self, clip1: VideoFileClip, clip2: VideoFileClip, duration: float) -> VideoFileClip:
        """Wipe transition: second clip wipes in from the right."""
        clip1, clip2 = self._match_clip_sizes(clip1, clip2)
        w, h = clip1.size

        def wipe_effect(get_frame, t):
            if t < duration:
                progress = t / duration
                wipe_x = int((1 - progress) * w)
                frame1 = get_frame(t)
                frame2 = clip2.get_frame(t)

                result = frame1.copy()
                result[:, wipe_x:] = frame2[:, wipe_x:]
            else:
                return clip2.get_frame(t - duration)
            return result

        transition_clip = clip1.transform(wipe_effect, duration=duration)
        clip2_remaining = clip2.subclipped(duration, clip2.duration)
        return concatenate_videoclips([transition_clip, clip2_remaining], method="compose")

    def _match_clip_sizes(self, clip1: VideoFileClip, clip2: VideoFileClip):
        """Ensure both clips have the same size."""
        size1 = clip1.size
        size2 = clip2.size

        if size1 != size2:
            # Resize clip2 to match clip1
            clip2 = clip2.resized(size1)

        return clip1, clip2

    @classmethod
    def list_transitions(cls) -> Dict[str, str]:
        """List all available transition types."""
        return cls.TRANSITION_REGISTRY.copy()

    @classmethod
    def register_transition(cls, name: str, description: str, method: Callable):
        """
        Register a custom transition method.

        Args:
            name: Name of the transition (e.g., 'custom_fade')
            description: Description of the transition
            method: Method that takes (clip1, clip2, duration) and returns a VideoFileClip

        Example:
            def my_custom_transition(clip1, clip2, duration):
                # Your custom transition logic
                return result_clip

            VideoTransitionMaker.register_transition(
                'custom_fade',
                'My custom fade transition',
                my_custom_transition
            )
        """
        cls.TRANSITION_REGISTRY[name] = description
        setattr(cls, f'_apply_{name}', method)

        # Update the _get_transition_method to include the new transition
        def _get_transition_method_with_custom(self):
            methods = {
                'crossfade': self._apply_crossfade,
                'fadein': self._apply_fadein,
                'fadeout': self._apply_fadeout,
                'slide_left': self._apply_slide_left,
                'slide_right': self._apply_slide_right,
                'slide_up': self._apply_slide_up,
                'slide_down': self._apply_slide_down,
                'zoom_in': self._apply_zoom_in,
                'zoom_out': self._apply_zoom_out,
                'wipe_left': self._apply_wipe_left,
                'wipe_right': self._apply_wipe_right,
                'dissolve': self._apply_dissolve,
                'none': self._apply_none
            }
            # Add custom transition if it exists
            if hasattr(self, f'_apply_{name}'):
                methods[name] = getattr(self, f'_apply_{name}')
            return methods.get(self.transition_type, self._apply_crossfade)

        cls._get_transition_method = _get_transition_method_with_custom


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Example: Combine videos with different transition types

    # List available transitions
    print("Available transitions:")
    for name, desc in VideoTransitionMaker.list_transitions().items():
        print(f"  - {name}: {desc}")

    # Example usage (uncomment and provide actual paths):
    # video_paths = [
    #     "path/to/video1.mp4",
    #     "path/to/video2.mp4",
    #     "path/to/video3.mp4"
    # ]
    #
    # # Create transition maker with crossfade
    # transition_maker = VideoTransitionMaker(
    #     transition_type='crossfade',
    #     duration=1.0
    # )
    #
    # # Combine videos with transitions
    # output_path = transition_maker.combine_videos(
    #     video_paths,
    #     "output/final_video_with_transitions.mp4"
    # )
