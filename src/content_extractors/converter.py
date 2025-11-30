from moviepy import VideoFileClip
import os


def convert_mov_to_mp4(input_path, output_path):
    """
    Converts a .mov video file to .mp4 format.

    Parameters:
    input_path (str): Path to the input .mov file.
    output_path (str): Path to save the output .mp4 file.
    """
    try:
        # Load the .mov file
        video_clip = VideoFileClip(input_path)

        # Write the video file to .mp4 format
        video_clip.write_videofile(
            output_path,
            codec='libx264',  # Video codec
            audio_codec='aac',  # Audio codec
            # Additional ffmpeg parameters
            ffmpeg_params=['-preset', 'fast', '-crf', '23']
        )
        print(f"Conversion successful: '{output_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# get all mov files from EFFECTS and convert them to mp4
# for f in os.listdir(EFFECTS):
#     if f.endswith(".mov"):
#         input_mov = os.path.join(EFFECTS, f).replace("\\", "/")
#         output_mp4 = os.path.join(EFFECTS, f.replace(".mov", ".mp4")).replace("\\", "/")
#         convert_mov_to_mp4(input_mov, output_mp4)
