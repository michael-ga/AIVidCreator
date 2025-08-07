import os
import whisper

def create_srt_from_voiceover(wav_path, txt_path=None, srt_path=None, language='en', model_size='base'):
    """
    Use OpenAI Whisper to transcribe a WAV/MP3 voiceover and create an SRT file with timestamps.
    
    Note: This function requires ffmpeg to be installed and available in PATH.
    If you encounter errors, install ffmpeg:
    - Windows: winget install "FFmpeg (Essentials Build)" or download from https://ffmpeg.org/
    - Mac: brew install ffmpeg
    - Linux: sudo apt install ffmpeg
    
    wav_path: path to the WAV/MP3 audio file
    txt_path: path to the plain text file (optional, not used with Whisper transcription)
    srt_path: output SRT file path (default: same as wav_path with .srt)
    language: language code (default: 'en')
    model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
    """
    if srt_path is None:
        srt_path = os.path.splitext(wav_path)[0] + ".srt"
    
    try:
        # Load Whisper model
        print(f"Loading Whisper model '{model_size}'...")
        model = whisper.load_model(model_size)
        
        # Transcribe audio with word timestamps
        print(f"Transcribing audio file: {wav_path}")
        result = model.transcribe(wav_path, word_timestamps=True, language=language)
        
        # Convert to SRT format
        srt_content = ""
        subtitle_index = 1
        
        for segment in result['segments']:
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            text = segment['text'].strip()
            
            srt_content += f"{subtitle_index}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"{text}\n\n"
            subtitle_index += 1
        
        # Write SRT file
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        print(f"SRT created successfully at: {srt_path}")
        return srt_path
        
    except FileNotFoundError as e:
        print("ERROR: ffmpeg not found. Please install ffmpeg:")
        print("Windows: winget install 'FFmpeg (Essentials Build)'")
        print("Mac: brew install ffmpeg")
        print("Linux: sudo apt install ffmpeg")
        print(f"Original error: {e}")
        return None
    except Exception as e:
        print(f"Error creating SRT file: {e}")
        return None

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    seconds_int = int(seconds_remainder)
    milliseconds = int((seconds_remainder - seconds_int) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"

def create_demo_srt(srt_path):
    """Create a demo SRT file for testing purposes"""
    demo_content = """1
00:00:00,000 --> 00:00:03,000
This is a demo subtitle file

2
00:00:03,000 --> 00:00:06,000
Created with Whisper-based transcription

3
00:00:06,000 --> 00:00:10,000
Install ffmpeg to enable audio transcription
"""
    
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write(demo_content)
    
    print(f"Demo SRT file created at: {srt_path}")
    return srt_path

# Example usage:
if __name__ == "__main__":
    audio_file = r"C:\Gituhb\AI\VidAi\content\workitems\images_20250608_014745\Leo-7760206.mp3"
    txt = r"C:\Gituhb\AI\VidAi\content\workitems\images_20250608_014745\voiceover.txt"
    
    # Try to create SRT from audio, fallback to demo if ffmpeg not available
    result = create_srt_from_voiceover(audio_file, txt)
    
    if result is None:
        # Create demo SRT file as fallback
        demo_srt_path = os.path.splitext(audio_file)[0] + "_demo.srt"
        create_demo_srt(demo_srt_path)
        print("\nCreated demo SRT file. Install ffmpeg to enable real transcription.")
