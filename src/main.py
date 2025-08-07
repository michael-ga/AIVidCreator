import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from video_creation.utils.flow_manager import continue_create_movement
from video_creation.utils.image_fetcher import fetch_recent_images
from video_creation.audio_utils import combine_audio_with_video
from video_creation.images_to_movement_effect import ClipMaker, get_image_paths
from video_creation.video_effect_composer import VideoComposer
from video_creation.caption_text_flow import create_caption_text_clips

DURATION = 3  # Default duration for clips
BASE_DIR = "PATH_TO_USER_DOWNLOAD"
PATH_TO_DEFULT_AUDIO = "PATH_TO_MP3_VIRAL_AUDIO"

path = fetch_recent_images(BASE_DIR, minutes=30)
# If you want to fetch new images, uncomment the above line.
# For now, use the latest workitem folder.
# path = r"C:\Gituhb\AI\VidAi\content\workitems\images_20250531_215247"
movement_path = os.path.join(path, "movement")
images_paths = get_image_paths(path)
if not images_paths:
    print("No images found in the specified directory.")
    exit()

if continue_create_movement(path, images_paths):
    for image_path in images_paths:
        clip_maker = ClipMaker(image_path)
        clip_maker.create_shake_clip(duration=DURATION)
        clip_maker.create_zoom_in_clip(duration=DURATION)

else:
    print("Movement folder already exists and contains all images. Skipping creation.")

video_effect_composer = VideoComposer(movement_path)
with_effects_path = os.path.join(movement_path, "with_effects")
os.makedirs(with_effects_path, exist_ok=True)
video_effect_composer.randomize_effects(with_effects_path)

# --- Audio Handling Example ---
# Combine a viral sound with the first video with effect
viral_audio = PATH_TO_DEFULT_AUDIO
first_video = None
for f in os.listdir(with_effects_path):
    if f.endswith(".mp4"):
        first_video = os.path.join(with_effects_path, f)
        break
if first_video:
    output_video_with_audio = os.path.join(with_effects_path, "final_with_audio.mp4")
    combine_audio_with_video(first_video, viral_audio, output_video_with_audio)
    print(f"Combined video and audio saved to: {output_video_with_audio}")
else:
    print("No video found in with_effects folder to combine with audio.")

# --- Caption Text Flow Example ---
caption_txt = os.path.join(path, "captions.txt")
caption_output_dir = os.path.join(path, "caption_clips")
caption_video = create_caption_text_clips(caption_txt, caption_output_dir)
print(f"Caption video created at: {caption_video}")
