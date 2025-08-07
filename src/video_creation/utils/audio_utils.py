from moviepy import *

def combine_audio_with_video(video_path, audio_path, output_path, audio_volume=1.0):
    """
    Combine an audio file with a video file. The audio will be trimmed or looped to match the video duration.
    """
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path).with_volume_scaled(audio_volume)
    if audio.duration < video.duration:
        # Loop audio if it's shorter than video
        n_loops = int(video.duration // audio.duration) + 1
        audio = CompositeAudioClip([audio] * n_loops).subclipped(0, video.duration)
    else:
        audio = audio.subclipped(0, video.duration)
    video = video.with_audio(audio)
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    video.close()
    audio.close()

def crop_audio(audio_path, start_time, end_time, output_path):
    """
    Crop an audio file to a specific time range and save it to a new file.
    """
    if end_time == None:
        end_time = AudioFileClip(audio_path).duration

    audio = AudioFileClip(audio_path).subclipped(start_time, end_time)
    audio.write_audiofile(output_path, codec="mp3")
    audio.close()

def convert_to_wav(audio_path, output_path):
    """
    Convert an audio file to WAV format.
    """
    audio = AudioFileClip(audio_path)
    audio.write_audiofile(output_path, codec="pcm_s16le")
    audio.close()

if __name__ == "__main__":
    # Example usage
    # crop_audio(r"C:\Gituhb\AI\VidAi\content\audio\Cornfield Chase.mp3",33,None,r"C:\Gituhb\AI\VidAi\content\audio\Cornfield Chase cropped.mp3")
#    convert_to_wav(r"C:\Gituhb\AI\VidAi\content\workitems\images_20250608_014745\Leo-7760206.mp3",r"C:\Gituhb\AI\VidAi\content\workitems\images_20250608_014745\Leo-7760206.wav")