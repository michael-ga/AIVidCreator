## Description 
This tool aimed to make Youtube short combine AI and local edit using moviepy
Input - video topic and optional script as exmaple
output - video with generated images to video with effect, voiceover, and subtitles.

## Roadmap
Video generation flow
- [ ] input the topic idea or example. [UI/ CLI with easy start 1 click]
- [ ] Generate image prompts and transcript, clean for voice over [Design json structure]
- [ ] Generate content [Keep parts and videos, metadata of prompts and data for AI pick scene sound text timing ]
    - [ ] Video description, viral title, tags [1 prompt]
    - [ ] images using image generation 
    - [ ] Optional - video for hook etc..
    - [ ] voice over using elevenlabs API
    - [ ] Transcript to voiceover to srt using wishper model
- [ ] Detect voice over timing srt and plan times to compose scenes
- [ ] Genreate template for text video to compose [Include styles]
- [ ] Compose video with effects, sound, titles

## Notes
Effects file name convention
- include green - for green screen effect
- short description to be used with AI 
- Compress giant effect clips using `convert_mov_to_mp4`

### Example content directory structure
- Add edited viral audios under audio
- Effects base include all effects to be randomized from - Future AI pick by name and image prompt.
```
C:\GITUHB\AI\VIDAI\CONTENT
├───audio
├───effect_base_videos
├───images
├───output
└───workitems
    ├───images_20250530_013035
    │   ├───caption_clips
    │   └───movement
    │       └───with_effects
    ├───images_20250531_215247

    │   └───movement
```
