ELEVENLABS_API_KEY="sk_054704f4f0e7a4a377ef24a14934b0fafda51341f70c4212"

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import os

load_dotenv()

elevenlabs = ElevenLabs(
  api_key=ELEVENLABS_API_KEY,
)
def create_audio(text_arg, output_path,name="voice.mp3"):
    audio = elevenlabs.text_to_speech.convert(
        text=text_arg,
        voice_id="TxGEqnHWrfWFTfGW9XjX",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    # Save audio stream to file (MP3)
    with open(os.path.join(output_path, name), "wb") as f:
        # The SDK may return an iterator/generator of bytes chunks
        for chunk in audio:
            if chunk:
                f.write(chunk)
    print(f"Saved audio to: {os.path.join(output_path, name)}")

if __name__ == "__main__":
    create_audio("What if courage wasn't something to fear—but something to become?\n\n   In this powerful visual short, we journey through what it truly means to be courageous—not just\n", ".")
