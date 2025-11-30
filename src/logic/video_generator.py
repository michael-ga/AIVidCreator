from voice.elevenlabs_api import create_audio
from video_creation.video_transition_maker import VideoTransitionMaker
from video_creation.images_to_movement_effect import ClipMaker
import os
import json
import sys
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

# Add src directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


DEFAULT_DURATION = 3  # default seconds per clip when no voiceover


def load_project_data(project_dir):
    """Load project_data.json from the project directory."""
    project_data_path = os.path.join(project_dir, "project_data.json")
    if not os.path.exists(project_data_path):
        raise FileNotFoundError(
            f"project_data.json not found at: {project_data_path}")

    with open(project_data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_images_from_directory(images_dir):
    """Get all image files from the Images directory."""
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found at: {images_dir}")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    images = [
        os.path.join(images_dir, file)
        for file in os.listdir(images_dir)
        if file.lower().endswith(valid_extensions)
    ]

    # Sort by modification time to maintain order
    images.sort(key=lambda x: os.path.getmtime(x))
    return images


def create_movement_clips(images, output_dir, duration=DEFAULT_DURATION):
    """
    Create video clips with random movement effects for each image.
    Returns list of output video paths.
    """
    movement_dir = os.path.join(output_dir, "movement")
    os.makedirs(movement_dir, exist_ok=True)

    video_clips = []

    for i, image_path in enumerate(images):
        print(
            f"\n📹 Processing image {i + 1}/{len(images)}: {os.path.basename(image_path)}")

        # Create output path for the movement clip
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(movement_dir, f"{base_name}_movement.mp4")

        # Create ClipMaker instance
        # duration may be a single number or a list/dict indexed by i or base_name
        this_duration = duration
        if isinstance(duration, dict):
            # try scene id or base_name
            this_duration = duration.get(
                str(i + 1), duration.get(base_name, DEFAULT_DURATION))
        elif isinstance(duration, list):
            this_duration = duration[i] if i < len(
                duration) else DEFAULT_DURATION

        clip_maker = ClipMaker(
            image_path=image_path,
            output_path=output_path,
            duration=this_duration,
            fps=30
        )

        # Create random movement effect
        clip_maker.create_random_effect(
            duration=this_duration, output_path=output_path)

        if os.path.exists(output_path):
            video_clips.append(output_path)
            print(f"✓ Created movement clip: {output_path}")
        else:
            print(f"✗ Failed to create clip for: {image_path}")

    return video_clips


def create_voiceovers(scenes, output_dir):
    """
    Create voiceover audio files for each scene's text.
    Returns list of audio file paths.
    """
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    audio_files = {}
    scenes_list = scenes.get("scenes", [])

    # We'll update the scenes JSON in-place so we can persist audio paths
    for i, scene in enumerate(scenes_list):
        scene_id = scene.get("id", i + 1)
        text = scene.get("text", "")

        if not text:
            print(f"⚠ Scene {scene_id} has no text, skipping voiceover")
            continue

        print(
            f"\n🎤 Ensuring voiceover for scene {scene_id} ({i + 1}/{len(scenes_list)})")

        # If a voiceover path already exists in scene data and file exists, reuse it
        existing = scene.get("voiceover") or scene.get("audio")
        if existing:
            existing_path = os.path.join(
                output_dir, existing) if not os.path.isabs(existing) else existing
            if os.path.exists(existing_path):
                try:
                    audio_clip = AudioFileClip(existing_path)
                    dur = audio_clip.duration
                    audio_clip.close()
                    audio_files[str(scene_id)] = existing_path
                    scene["voiceover"] = os.path.relpath(
                        existing_path, output_dir)
                    print(
                        f"✓ Reusing existing voiceover for scene {scene_id}: {existing_path} (duration {dur:.2f}s)")
                    continue
                except Exception:
                    print(
                        f"⚠ Could not read existing audio for scene {scene_id}, will recreate")

        audio_filename = f"voiceover_scene_{scene_id}.mp3"
        audio_path = os.path.join(audio_dir, audio_filename)

        try:
            create_audio(text, audio_dir, name=audio_filename)
            if os.path.exists(audio_path):
                # measure duration
                try:
                    audio_clip = AudioFileClip(audio_path)
                    dur = audio_clip.duration
                    audio_clip.close()
                except Exception:
                    dur = None

                audio_files[str(scene_id)] = audio_path
                # store relative path inside scene so project_data.json isn't full of absolute paths
                scene["voiceover"] = os.path.relpath(audio_path, output_dir)
                print(
                    f"✓ Created voiceover: {audio_path} (duration {dur if dur is not None else 'unknown'})")
            else:
                print(f"✗ Failed to create voiceover for scene {scene_id}")
        except Exception as e:
            print(f"✗ Error creating voiceover for scene {scene_id}: {e}")

    return audio_files


def combine_video_with_audio(video_path, audio_path, output_path):
    """Combine a video clip with its corresponding audio."""
    try:
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)

        # Trim audio to match video duration if needed
        if audio.duration > video.duration:
            audio = audio.subclipped(0, video.duration)

        # Set video duration to match audio if audio is shorter
        if audio.duration < video.duration:
            video = video.subclipped(0, audio.duration)

        # Combine video and audio
        final_clip = video.with_audio(audio)
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=video.fps
        )

        video.close()
        audio.close()
        final_clip.close()

        return True
    except Exception as e:
        print(f"✗ Error combining video and audio: {e}")
        return False


def match_scenes_with_images(scenes, images):
    """
    Match scenes with images.
    Returns list of tuples: (image_path, scene_data)
    """
    scenes_list = scenes.get("scenes", [])
    matched = []

    # Match images with scenes by index
    for i, scene in enumerate(scenes_list):
        if i < len(images):
            matched.append((images[i], scene))
        else:
            print(f"⚠ No image available for scene {scene.get('id', i + 1)}")

    return matched


def create_final_video(
    video_clips_with_audio,
    output_dir,
    transition_type: str = 'crossfade',
    transition_duration: float = 1.0
):
    """
    Concatenate all video clips into a final video with transition effects.

    Args:
        video_clips_with_audio: List of video file paths to combine.
        output_dir: Directory where the final video will be saved.
        transition_type: Type of transition to apply between clips.
                        Options: 'crossfade', 'fadein', 'fadeout', 'slide_left',
                                'slide_right', 'slide_up', 'slide_down', 'zoom_in',
                                'zoom_out', 'wipe_left', 'wipe_right', 'dissolve', 'none'
        transition_duration: Duration of each transition in seconds.
    """
    if not video_clips_with_audio:
        print("✗ No video clips to concatenate")
        return None

    print(
        f"\n🎬 Combining {len(video_clips_with_audio)} clips with '{transition_type}' transition...")

    try:
        # Use VideoTransitionMaker to combine videos with transitions
        transition_maker = VideoTransitionMaker(
            transition_type=transition_type,
            duration=transition_duration
        )

        final_output_path = os.path.join(output_dir, "final_video.mp4")
        result_path = transition_maker.combine_videos(
            video_clips_with_audio,
            final_output_path,
            transition_duration=transition_duration
        )

        if result_path:
            print(f"✓ Final video created with transitions: {result_path}")
            return result_path
        else:
            print(
                "⚠ Transitioned final video failed, attempting simple concatenation as fallback...")
            # fallback: try a simple concatenation without fancy transitions
            try:
                clips = []
                for path in video_clips_with_audio:
                    if not os.path.exists(path):
                        print(
                            f"  ⚠ Missing clip for fallback concat: {path}, skipping")
                        continue
                    clips.append(VideoFileClip(path))

                if not clips:
                    print("✗ No valid clips found for fallback concatenation")
                    return None

                final_video = concatenate_videoclips(clips, method="compose")
                fps = clips[0].fps if clips and hasattr(
                    clips[0], 'fps') else 24
                final_video = final_video.with_fps(fps)
                os.makedirs(os.path.dirname(final_output_path) if os.path.dirname(
                    final_output_path) else '.', exist_ok=True)
                final_video.write_videofile(
                    final_output_path,
                    codec="libx264",
                    audio_codec="aac",
                    fps=fps
                )

                for c in clips:
                    try:
                        c.close()
                    except Exception:
                        pass
                final_video.close()
                print(
                    f"✓ Final video created by fallback concatenation: {final_output_path}")
                return final_output_path
            except Exception as e:
                print(f"✗ Fallback concatenation failed: {e}")
                import traceback
                traceback.print_exc()
                return None

    except Exception as e:
        print(f"✗ Error creating final video: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(project_dir):
    """
    Main function to create media files from project data.

    Args:
        project_dir: Path to the project directory containing project_data.json and Images folder
    """
    print(f"🚀 Starting media creation for: {project_dir}")

    # Step 1: Load project data
    print("\n📋 Step 1: Loading project data...")
    project_data = load_project_data(project_dir)
    scenes = project_data.get("scenes", {})
    print(f"✓ Loaded project data with {len(scenes.get('scenes', []))} scenes")

    # Step 2: Get images
    print("\n🖼️ Step 2: Getting images...")
    images_dir = os.path.join(project_dir, "Images")
    images = get_images_from_directory(images_dir)
    print(f"✓ Found {len(images)} images")

    if not images:
        print("✗ No images found. Exiting.")
        return

    # Step 3: Create (or reuse) voiceovers for each scene first
    print("\n🎤 Step 3: Creating/reusing voiceovers for scenes...")
    audio_files = create_voiceovers(scenes, project_dir)
    print(f"✓ Ensured voiceovers for {len(audio_files)} scenes")

    # Persist updated scenes (voiceover paths) back into project_data.json
    project_data_path = os.path.join(project_dir, "project_data.json")
    try:
        # Ensure we update the original project_data structure
        if isinstance(project_data, dict):
            project_data['scenes'] = scenes
            to_write = project_data
        else:
            to_write = {'scenes': scenes}

        with open(project_data_path, 'w', encoding='utf-8') as f:
            json.dump(to_write, f, indent=2, ensure_ascii=False)
        print(
            f"✓ Updated project data with voiceover locations: {project_data_path}")
    except Exception as e:
        print(f"⚠ Could not update project_data.json: {e}")

    # Step 4: Match scenes with images
    print("\n🔗 Step 4: Matching scenes with images...")
    matched = match_scenes_with_images(scenes, images)
    print(f"✓ Matched {len(matched)} scenes with images")

    # Step 5: Compute per-scene durations based on voiceover lengths (+1s) or fallback
    print("\n⏱️ Step 5: Computing per-scene durations from voiceovers...")
    per_scene_duration = {}
    min_duration = None
    for i, (image_path, scene) in enumerate(matched):
        scene_id = scene.get('id', i + 1)
        audio_rel = scene.get('voiceover')
        audio_path = None
        if audio_rel:
            candidate = os.path.join(project_dir, audio_rel)
            if os.path.exists(candidate):
                audio_path = candidate
        # Also check audio_files dict
        if not audio_path and str(scene_id) in audio_files:
            audio_path = audio_files.get(str(scene_id))

        if audio_path and os.path.exists(audio_path):
            try:
                audio_clip = AudioFileClip(audio_path)
                dur = audio_clip.duration
                audio_clip.close()
                scene_duration = max(DEFAULT_DURATION, dur + 1.0)
            except Exception:
                scene_duration = DEFAULT_DURATION
        else:
            scene_duration = DEFAULT_DURATION

        per_scene_duration[str(scene_id)] = scene_duration
        if min_duration is None or scene_duration < min_duration:
            min_duration = scene_duration

    print(
        f"✓ Computed durations for {len(per_scene_duration)} scenes (min={min_duration})")

    # Step 6: Create movement clips with per-scene durations
    print("\n📹 Step 6: Creating movement clips with scene durations...")
    # Build a mapping from base_name or index to duration for create_movement_clips
    duration_map = {}
    for i, (image_path, scene) in enumerate(matched):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        scene_id = scene.get('id', i + 1)
        duration_map[str(scene_id)] = per_scene_duration.get(
            str(scene_id), DEFAULT_DURATION)
        duration_map[base_name] = per_scene_duration.get(
            str(scene_id), DEFAULT_DURATION)

    video_clips = create_movement_clips(
        images, project_dir, duration=duration_map)
    print(f"✓ Created {len(video_clips)} movement clips")

    # Step 7: Combine video clips with audio
    print("\n🎬 Step 7: Combining videos with audio...")
    combined_dir = os.path.join(project_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    # Create a dictionary mapping scene_id to audio_path
    audio_dict = {k: v for k, v in audio_files.items()}

    video_clips_with_audio = []
    for i, (image_path, scene) in enumerate(matched):
        scene_id = scene.get("id", i + 1)
        video_path = video_clips[i] if i < len(video_clips) else None

        # audio lookup tries a few keys
        audio_path = None
        # first try scene.voiceover
        audio_rel = scene.get('voiceover')
        if audio_rel:
            candidate = os.path.join(project_dir, audio_rel)
            if os.path.exists(candidate):
                audio_path = candidate

        if not audio_path:
            audio_path = audio_dict.get(
                str(scene_id)) or audio_dict.get(scene_id)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        combined_output = os.path.join(
            combined_dir, f"{base_name}_with_audio.mp4")

        if video_path and audio_path and os.path.exists(audio_path):
            print(
                f"  Combining clip {i + 1}/{len(matched)} (scene {scene_id})...")
            if combine_video_with_audio(video_path, audio_path, combined_output):
                video_clips_with_audio.append(combined_output)
            else:
                print(
                    f"⚠ Combining failed for scene {scene_id}, using video without audio")
                video_clips_with_audio.append(video_path)
        elif video_path:
            video_clips_with_audio.append(video_path)

    print(
        f"✓ Prepared {len(video_clips_with_audio)} clips (with/without audio)")

    # Step 7: Create final video with transitions
    print("\n🎞️ Step 7: Creating final video with transitions...")
    # You can customize the transition type and duration here
    # Available transitions: 'crossfade', 'fadein', 'fadeout', 'slide_left', 'slide_right',
    #                       'slide_up', 'slide_down', 'zoom_in', 'zoom_out', 'wipe_left',
    #                       'wipe_right', 'dissolve', 'none'
    # Step 8: Create final video with transitions - clamp transition duration
    print("\n🎞️ Step 8: Creating final video with transitions...")
    # Choose desired transition and base duration
    desired_transition = 'crossfade'
    desired_transition_duration = 1.0

    # clamp transition to at most half the shortest clip to avoid overlap errors
    if min_duration is None:
        min_duration = DEFAULT_DURATION
    safe_transition_duration = min(
        desired_transition_duration, max(0.1, min_duration / 2.0))

    final_video_path = create_final_video(
        video_clips_with_audio,
        project_dir,
        transition_type=desired_transition,
        transition_duration=safe_transition_duration
    )

    if final_video_path:
        print(f"\n🎉 Success! Final video created at: {final_video_path}")
    else:
        print("\n⚠ Final video creation failed, but individual clips are available")

    print(f"\n📂 All files saved to: {project_dir}")


if __name__ == "__main__":
    # Default project directory
    project_dir = r"C:\Gituhb\AI\VidAi\content\workitems\courage_video\courage_20251103_231347"

    # Allow command line argument for project directory
    if len(sys.argv) > 1:
        project_dir = sys.argv[1]

    if not os.path.exists(project_dir):
        print(f"✗ Project directory not found: {project_dir}")
        sys.exit(1)

    main(project_dir)
