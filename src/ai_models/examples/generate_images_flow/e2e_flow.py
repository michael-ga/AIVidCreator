"""
Complete Flow with All Movement Effects Combined - WORKING VERSION

This script creates a proper video where:
1. Generates voiceover first to get actual duration
2. Creates 4 movement effects per scene, each taking 1/4 of the total time
3. Combines all 4 effects into one seamless video per scene
4. Adds the scene's audio to its combined video
5. Combines all scene videos into final complete video
6. SAVES ALL PATHS TO JSON IMMEDIATELY

Usage:
    python e2e_flow.py
"""

import json
import os
import sys
import asyncio
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import shutil

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
ai_models_dir = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.dirname(ai_models_dir)
sys.path.extend([ai_models_dir, src_dir])

print("🔧 Loading dependencies...")

try:
    from ai_interfaces.models import ImageGenerationRequest
    from ai_interfaces.image.providers.gemini_image import get_gemini_image_generator
    from ai_interfaces.exceptions import RateLimitError, AIProviderError  
    from config.settings import validate_environment
    print("   ✓ AI interfaces loaded")
    AI_AVAILABLE = True
except ImportError as e:
    print(f"   ⚠️ AI interfaces not available: {e}")
    AI_AVAILABLE = False
    RateLimitError = None  # Fallback
    AIProviderError = None

try:
    from audio_creation.elevenlabs_api import create_audio
    print("   ✓ ElevenLabs API loaded")
    ELEVENLABS_AVAILABLE = True
except ImportError as e:
    print(f"   ⚠️ ElevenLabs not available: {e}")
    ELEVENLABS_AVAILABLE = False

try:
    from video_creation.images_to_movement_effect import ClipMaker
    from video_creation.utils.audio_utils import combine_audio_with_video
    print("   ✓ Video creation modules loaded")
    VIDEO_AVAILABLE = True
except ImportError as e:
    print(f"   ⚠️ Video modules not available: {e}")
    VIDEO_AVAILABLE = False

try:
    from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
    print("   ✓ MoviePy loaded")
    MOVIEPY_AVAILABLE = True
except ImportError as e:
    print(f"   ⚠️ MoviePy not available: {e}")
    MOVIEPY_AVAILABLE = False

ALL_DEPENDENCIES = AI_AVAILABLE and ELEVENLABS_AVAILABLE and VIDEO_AVAILABLE and MOVIEPY_AVAILABLE

if ALL_DEPENDENCIES:
    print("✅ All dependencies loaded successfully!")
else:
    print("⚠️ Some dependencies missing.")


def extract_retry_delay_from_error(error_msg: str) -> float:
    """Extract retry delay from Gemini API error message."""
    try:
        delay_match = re.search(r'retry in ([\d.]+)s', error_msg)
        if delay_match:
            return float(delay_match.group(1))
        retry_match = re.search(r"'retryDelay': '(\d+)s'", error_msg)
        if retry_match:
            return float(retry_match.group(1))
        return 30.0
    except:
        return 30.0


def is_rate_limit_error(error_msg: str) -> bool:
    """Check if error is a rate limit error."""
    error_str = str(error_msg).lower()
    return any(phrase in error_str for phrase in [
        "429", "resource_exhausted", "quota exceeded", "rate limit", "too many requests"
    ])


def get_audio_duration(audio_path: str) -> float:
    """Get duration of an audio file in seconds."""
    try:
        if MOVIEPY_AVAILABLE:
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            audio_clip.close()
            return duration
        else:
            return 3.0
    except Exception as e:
        print(f"⚠️ Could not detect audio duration for {os.path.basename(audio_path)}: {e}")
        return 3.0


def create_placeholder_image(output_path: str, scene_id: int, prompt: str) -> bool:
    """Create a simple placeholder image when Gemini fails."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('RGB', (720, 1280), color='#1a1a2e')
        draw = ImageDraw.Draw(img)
        
        for i in range(1280):
            alpha = i / 1280
            color = (int(26 + alpha * 100), int(26 + alpha * 50), int(46 + alpha * 150))
            draw.line([(0, i), (720, i)], fill=color)
        
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        scene_text = f"Scene {scene_id}"
        draw.text((50, 100), scene_text, fill='white', font=font)
        
        prompt_lines = []
        words = prompt.split()
        current_line = ""
        for word in words:
            if len(current_line + " " + word) < 40:
                current_line += " " + word if current_line else word
            else:
                if current_line:
                    prompt_lines.append(current_line)
                current_line = word
                if len(prompt_lines) >= 5:
                    break
        if current_line and len(prompt_lines) < 5:
            prompt_lines.append(current_line)
        
        y_offset = 200
        for line in prompt_lines:
            draw.text((50, y_offset), line, fill='lightgray', font=font)
            y_offset += 50
        
        draw.text((50, 1200), "Placeholder Image", fill='gray', font=font)
        img.save(output_path)
        return True
        
    except Exception as e:
        print(f"   ⚠️ Could not create placeholder image: {e}")
        return False


class CompleteFlowWithAllEffects:
    """Complete video creation with all movement effects combined per scene."""
    
    def __init__(self, json_file_path: str):
        """Initialize the video creator."""
        self.json_file_path = json_file_path
        self.flow_data: Optional[Dict[str, Any]] = None
        self.output_directory: Optional[str] = None
        self.results = {
            'success': False,
            'images_generated': 0,
            'images_failed': 0,
            'placeholders_created': 0,
            'audios_generated': 0,
            'movement_videos_per_scene': 0,
            'combined_scene_videos': 0,
            'final_video': None,
            'errors': [],
            'step_status': {},
            'retries_performed': 0
        }
    
    def save_json_immediately(self, message: str = "Saving progress..."):
        """Save JSON immediately after each major step - FORCED WRITE."""
        try:
            print(f"\n💾 {message}")
            
            # Update timestamp
            self.flow_data['last_updated'] = datetime.now().isoformat()
            
            # Write to a temporary file first
            temp_path = self.json_file_path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.flow_data, f, indent=2, ensure_ascii=False)
                f.flush()  # Force write to disk
                os.fsync(f.fileno())  # Ensure it's written to disk
            
            # Create backup of current file
            backup_path = self.json_file_path + '.bak'
            if os.path.exists(self.json_file_path):
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                shutil.copy2(self.json_file_path, backup_path)
            
            # Replace original with temp - use shutil.move for Windows compatibility
            shutil.move(temp_path, self.json_file_path)
            
            print(f"   ✓ JSON saved successfully to: {os.path.basename(self.json_file_path)}")
            return True
            
        except Exception as e:
            print(f"   ✗ Error saving JSON: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def load_flow_data(self) -> bool:
        """Load and validate flow description JSON."""
        print("\n📋 Loading flow description...")
        
        try:
            if not os.path.exists(self.json_file_path):
                raise FileNotFoundError(f"Flow file not found: {self.json_file_path}")
                
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.flow_data = json.load(f)
                
            self.output_directory = self.flow_data.get('output_directory')
            if not self.output_directory:
                raise ValueError("No output_directory found in JSON")
                
            scenes = self.flow_data.get('scenes', {}).get('scenes', [])
            if not scenes:
                raise ValueError("No scenes found in flow data")
            
            print(f"✓ Flow loaded: {self.flow_data.get('user_input', 'Unknown')}")
            print(f"✓ Output directory: {self.output_directory}")
            print(f"✓ Found {len(scenes)} scenes")
            
            self.results['step_status']['load_flow'] = True
            return True
            
        except Exception as e:
            error_msg = f"Failed to load flow data: {e}"
            self.results['errors'].append(error_msg)
            print(f"✗ {error_msg}")
            return False
    
    async def generate_single_image_with_retry(self, img_gen, request, scene_id: int, scene_prompt: str, images_dir: str, max_retries: int = 3) -> Optional[str]:
        """Generate a single image with retry logic for rate limits."""
        
        for attempt in range(max_retries):
            try:
                print(f"   Attempting generation (try {attempt + 1}/{max_retries})...")
                response = await img_gen.generate(request)
                
                if response and response.images:
                    image_path = os.path.join(images_dir, f"scene_{scene_id:02d}_000.png")
                    with open(image_path, 'wb') as f:
                        f.write(response.images[0])
                    print(f"   ✓ Image saved: {os.path.basename(image_path)}")
                    self.results['images_generated'] += 1
                    return image_path
                    
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's the DAILY quota (100/day) - no point retrying
                if "quota exceeded" in error_msg.lower() and ("100" in error_msg or "GenerateRequestsPerDay" in error_msg):
                    print(f"\n❌ DAILY QUOTA LIMIT HIT!")
                    print(f"   Gemini free tier allows 100 images per day")
                    print(f"   You've used all 100 requests today")
                    print(f"   Creating placeholder images for remaining scenes...")
                    self.results['images_failed'] += 1
                    return None  # Skip to placeholder immediately
                
                # Check if it's a RateLimitError with retry_after attribute
                if RateLimitError and isinstance(e, RateLimitError):
                    delay = e.retry_after if e.retry_after else 60
                    
                    print(f"\n⚠️ RATE LIMIT HIT!")
                    print(f"   API requires {delay}s wait time")
                    
                    if attempt < max_retries - 1:
                        print(f"   ⏰ Waiting {delay}s before retry {attempt + 2}/{max_retries}...")
                        self.results['retries_performed'] += 1
                        await asyncio.sleep(delay + 2)
                        continue
                    else:
                        print(f"   ✗ Max retries reached. Creating placeholder...")
                        self.results['images_failed'] += 1
                        return None
                
                # Fallback: check error message string for rate limiting
                if is_rate_limit_error(error_msg):
                    delay = extract_retry_delay_from_error(error_msg)
                    print(f"   ⏳ Rate limit detected. Delay: {delay}s")
                    
                    if attempt < max_retries - 1:
                        print(f"   Waiting {delay}s before retry {attempt + 2}/{max_retries}...")
                        self.results['retries_performed'] += 1
                        await asyncio.sleep(delay + 2)
                        continue
                    else:
                        print(f"   ✗ Max retries reached. Creating placeholder...")
                        self.results['images_failed'] += 1
                        return None
                
                # Non-rate-limit errors
                print(f"   ✗ Error on attempt {attempt + 1}: {error_msg[:150]}...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
                else:
                    print(f"   ✗ Non-recoverable error after {max_retries} attempts")
                    self.results['images_failed'] += 1
                    return None
        
        return None
    
    async def generate_all_images(self) -> bool:
        """Generate images for all scenes using Gemini AI with comprehensive retry logic."""
        print("\n🎨 Generating images with Gemini AI...")
        
        if not AI_AVAILABLE:
            print("⚠️ AI services not available - creating placeholder images")
            scenes = self.flow_data.get('scenes', {}).get('scenes', [])
            images_dir = os.path.join(self.output_directory, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            for i, scene in enumerate(scenes):
                scene_id = scene.get('id', i + 1)
                prompt = scene.get('image_prompt', f'Scene {scene_id}')
                placeholder_path = os.path.join(images_dir, f"scene_{scene_id:02d}_placeholder.png")
                
                if create_placeholder_image(placeholder_path, scene_id, prompt):
                    scene['image_path'] = placeholder_path
                    self.results['placeholders_created'] += 1
            
            self.save_json_immediately("Saved placeholder image paths")
            self.results['step_status']['generate_images'] = self.results['placeholders_created'] > 0
            return self.results['placeholders_created'] > 0
        
        try:
            env_status = validate_environment()
            if not env_status.get("google"):
                print("⚠️ Google API key not found. Creating placeholder images...")
                return await self.generate_all_images()
        except Exception as e:
            print(f"⚠️ Environment validation failed: {e}")
            return await self.generate_all_images()
        
        try:
            images_dir = os.path.join(self.output_directory, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            img_gen = get_gemini_image_generator()
            
            scenes = self.flow_data.get('scenes', {}).get('scenes', [])
            generated_count = 0
            failed_count = 0
            
            print(f"Will generate {len(scenes)} images with rate limit handling...")
            
            for i, scene in enumerate(scenes):
                scene_id = scene.get('id', i + 1)
                image_prompt = scene.get('image_prompt', '')
                
                if not image_prompt:
                    print(f"⚠️ No image prompt for scene {scene_id}")
                    failed_count += 1
                    continue
                    
                print(f"\n🖼️ Scene {scene_id}: Generating image...")
                print(f"   Prompt: {image_prompt[:80]}...")
                
                try:
                    request = ImageGenerationRequest(
                        prompt=f"{image_prompt}. Cinematic, high quality, 9:16 aspect ratio, detailed",
                        negative_prompt="blurry, low quality, distorted, watermark, text, ugly",
                        width=720,
                        height=1280,
                        style="cinematic, photorealistic"
                    )
                    
                    image_path = await self.generate_single_image_with_retry(
                        img_gen, request, scene_id, image_prompt, images_dir, max_retries=2
                    )
                    
                    if image_path:
                        scene['image_path'] = image_path
                        # SAVE IMMEDIATELY after each image
                        self.save_json_immediately(f"Saved image path for scene {scene_id}")
                        
                        if 'placeholder' in image_path:
                            print(f"   📋 Using placeholder for scene {scene_id}")
                        else:
                            generated_count += 1
                            print(f"   ✅ Successfully generated image for scene {scene_id}")
                    else:
                        print(f"   ❌ Complete failure for scene {scene_id}")
                        failed_count += 1
                        
                except Exception as e:
                    print(f"   ✗ Unexpected error for scene {scene_id}: {e}")
                    failed_count += 1
                    continue
                
                if i < len(scenes) - 1:
                    delay = min(3.0 + (failed_count * 2), 10.0)
                    print(f"   ⏸️ Waiting {delay:.1f}s before next scene...")
                    await asyncio.sleep(delay)
            
            total_with_content = generated_count + self.results['placeholders_created']
            
            print(f"\n📊 Image Generation Results:")
            print(f"   ✅ Successfully generated: {generated_count}")
            print(f"   📋 Placeholder images: {self.results['placeholders_created']}")
            print(f"   ❌ Complete failures: {failed_count}")
            print(f"   🔄 Total retries performed: {self.results['retries_performed']}")
            
            self.results['images_generated'] = generated_count
            self.results['images_failed'] = failed_count
            self.results['step_status']['generate_images'] = total_with_content > 0
            
            return total_with_content > 0
            
        except Exception as e:
            error_msg = f"Image generation pipeline failed: {e}"
            self.results['errors'].append(error_msg)
            print(f"✗ {error_msg}")
            self.results['step_status']['generate_images'] = False
            return False
    
    def create_all_voiceovers(self) -> bool:
        """Create voiceover audio for all scenes using ElevenLabs."""
        print("\n🎤 Creating voiceovers with ElevenLabs...")
        
        if not ELEVENLABS_AVAILABLE:
            print("⚠️ ElevenLabs not available - using default durations")
            scenes = self.flow_data.get('scenes', {}).get('scenes', [])
            for scene in scenes:
                scene['actual_duration'] = scene.get('duration', 4)
            self.save_json_immediately("Saved default durations")
            self.results['step_status']['create_audio'] = False
            return False
        
        try:
            audio_dir = os.path.join(self.output_directory, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            
            scenes = self.flow_data.get('scenes', {}).get('scenes', [])
            generated_count = 0
            
            for i, scene in enumerate(scenes):
                scene_id = scene.get('id', i + 1)
                scene_text = scene.get('text', '')
                
                if not scene_text:
                    print(f"⚠️ No text for scene {scene_id}")
                    scene['actual_duration'] = scene.get('duration', 4)
                    continue
                    
                print(f"\n🔊 Scene {scene_id}: Creating voiceover...")
                print(f"   Text: {scene_text[:80]}...")
                
                try:
                    audio_filename = f"scene_{scene_id:02d}_voiceover.mp3"
                    
                    create_audio(
                        text_arg=scene_text,
                        output_path=audio_dir,
                        name=audio_filename
                    )
                    
                    audio_path = os.path.join(audio_dir, audio_filename)
                    
                    if os.path.exists(audio_path):
                        scene['audio_path'] = audio_path
                        generated_count += 1
                        
                        duration = get_audio_duration(audio_path)
                        scene['actual_duration'] = duration
                        
                        # SAVE IMMEDIATELY after each audio
                        self.save_json_immediately(f"Saved audio path for scene {scene_id}")
                        
                        print(f"   ✓ Saved: {audio_filename} ({duration:.1f}s)")
                    else:
                        print(f"   ✗ Audio file not created for scene {scene_id}")
                        scene['actual_duration'] = scene.get('duration', 4)
                    
                except Exception as e:
                    print(f"   ✗ Error creating voiceover for scene {scene_id}: {e}")
                    scene['actual_duration'] = scene.get('duration', 4)
                    continue
            
            print(f"\n✓ Generated {generated_count} voiceovers successfully")
            self.results['audios_generated'] = generated_count
            self.results['step_status']['create_audio'] = generated_count > 0
            return generated_count > 0
            
        except Exception as e:
            error_msg = f"Voiceover generation failed: {e}"
            self.results['errors'].append(error_msg)
            print(f"✗ {error_msg}")
            self.results['step_status']['create_audio'] = False
            return False
    
    def create_all_movement_effects_for_scene(self, scene_data: dict, movement_dir: str) -> Tuple[List[str], float]:
        """Create all 4 movement effects for a single scene."""
        scene_id = scene_data.get('id')
        image_path = scene_data.get('image_path')
        total_duration = scene_data.get('actual_duration', scene_data.get('duration', 4))
        
        if not image_path or not os.path.exists(image_path):
            print(f"   ⚠️ No image found for scene {scene_id}")
            return [], 0.0
        
        effect_duration = total_duration / 4.0
        
        print(f"   Image: {os.path.basename(image_path)}")
        print(f"   Total duration: {total_duration:.1f}s")
        print(f"   Per effect: {effect_duration:.1f}s")
        
        try:
            clip_maker = ClipMaker(
                image_path=image_path,
                duration=effect_duration
            )
            
            movement_effects = ['zoom', 'shake', 'pan', 'rotation']
            created_videos = []
            
            for effect in movement_effects:
                try:
                    video_filename = f"scene_{scene_id:02d}_{effect}.mp4"
                    video_path = os.path.join(movement_dir, video_filename)
                    
                    print(f"     Creating {effect} effect ({effect_duration:.1f}s)...")
                    
                    if effect == 'zoom':
                        clip_maker.create_zoom_in_clip(duration=effect_duration, output_path=video_path)
                    elif effect == 'shake':
                        clip_maker.create_shake_clip(duration=effect_duration, output_path=video_path)
                    elif effect == 'pan':
                        clip_maker.create_pan_clip(duration=effect_duration, output_path=video_path)
                    elif effect == 'rotation':
                        clip_maker.create_rotation_clip(duration=effect_duration, output_path=video_path)
                    
                    if os.path.exists(video_path):
                        created_videos.append(video_path)
                        print(f"     ✓ Created: {video_filename}")
                    else:
                        print(f"     ✗ Failed to create {effect} effect")
                    
                except Exception as e:
                    print(f"     ✗ Error creating {effect} effect: {e}")
                    continue
            
            return created_videos, effect_duration
            
        except Exception as e:
            print(f"   ✗ Error creating movement videos for scene {scene_id}: {e}")
            return [], 0.0
    
    def combine_scene_effects_with_transitions(self, effect_videos: List[str], scene_id: int, combined_dir: str) -> Optional[str]:
        """Combine all 4 movement effects for a scene into one video."""
        if not effect_videos or not MOVIEPY_AVAILABLE:
            return None
        
        try:
            print(f"   🔗 Combining {len(effect_videos)} effects for scene {scene_id}...")
            
            combined_filename = f"scene_{scene_id:02d}_combined_effects.mp4"
            combined_path = os.path.join(combined_dir, combined_filename)
            
            clips = []
            for video_path in effect_videos:
                if os.path.exists(video_path):
                    try:
                        clip = VideoFileClip(video_path)
                        clips.append(clip)
                        print(f"     ✓ Loaded: {os.path.basename(video_path)} ({clip.duration:.1f}s)")
                    except Exception as e:
                        print(f"     ✗ Error loading {os.path.basename(video_path)}: {e}")
                        continue
            
            if not clips:
                print(f"     ✗ No clips could be loaded for scene {scene_id}")
                return None
            
            print(f"     Concatenating {len(clips)} clips...")
            final_clip = concatenate_videoclips(clips, method="compose")
            
            print(f"     Writing combined video ({final_clip.duration:.1f}s total)...")
            final_clip.write_videofile(
                combined_path,
                codec='libx264',
                audio_codec='aac',
                logger=None
            )
            
            for clip in clips:
                clip.close()
            final_clip.close()
            
            if os.path.exists(combined_path):
                print(f"   ✓ Combined effects video: {combined_filename}")
                return combined_path
            else:
                print(f"   ✗ Failed to create combined video")
                return None
            
        except Exception as e:
            print(f"   ✗ Error combining effects for scene {scene_id}: {e}")
            return None
    
    def create_and_combine_all_movement_videos(self) -> bool:
        """Create all movement effects and combine them for each scene."""
        print("\n🎬 Creating and combining movement videos...")
        
        if not VIDEO_AVAILABLE or not MOVIEPY_AVAILABLE:
            print("⚠️ Video modules not available")
            self.results['step_status']['create_movement'] = False
            return False
        
        try:
            movement_dir = os.path.join(self.output_directory, "images", "movement")
            combined_dir = os.path.join(self.output_directory, "combined_effects")
            os.makedirs(movement_dir, exist_ok=True)
            os.makedirs(combined_dir, exist_ok=True)
            
            scenes = self.flow_data.get('scenes', {}).get('scenes', [])
            total_videos_created = 0
            combined_videos_created = 0
            
            for i, scene in enumerate(scenes):
                scene_id = scene.get('id', i + 1)
                
                if not scene.get('image_path'):
                    print(f"⚠️ Skipping scene {scene_id} - no image available")
                    continue
                
                print(f"\n📹 Scene {scene_id}: Creating movement effects...")
                
                effect_videos, effect_duration = self.create_all_movement_effects_for_scene(scene, movement_dir)
                
                if effect_videos:
                    total_videos_created += len(effect_videos)
                    scene['movement_videos'] = effect_videos
                    scene['effect_duration'] = effect_duration
                    
                    # SAVE IMMEDIATELY after movement videos
                    self.save_json_immediately(f"Saved movement videos for scene {scene_id}")
                    
                    combined_video = self.combine_scene_effects_with_transitions(effect_videos, scene_id, combined_dir)
                    
                    if combined_video:
                        scene['combined_effects_video'] = combined_video
                        combined_videos_created += 1
                        
                        # SAVE IMMEDIATELY after combined video
                        self.save_json_immediately(f"Saved combined video for scene {scene_id}")
                        
                        print(f"   ✓ Scene {scene_id} complete: {len(effect_videos)} effects combined")
                    else:
                        print(f"   ⚠️ Scene {scene_id}: Effects created but combination failed")
                else:
                    print(f"   ✗ No effects created for scene {scene_id}")
            
            print(f"\n✓ Created {total_videos_created} movement effect videos")
            print(f"✓ Combined {combined_videos_created} scene effect videos")
            
            self.results['movement_videos_per_scene'] = total_videos_created // 4 if total_videos_created > 0 else 0
            self.results['combined_scene_videos'] = combined_videos_created
            self.results['step_status']['create_movement'] = combined_videos_created > 0
            return combined_videos_created > 0
            
        except Exception as e:
            error_msg = f"Movement video creation failed: {e}"
            self.results['errors'].append(error_msg)
            print(f"✗ {error_msg}")
            self.results['step_status']['create_movement'] = False
            return False
    
    def add_audio_to_combined_videos(self) -> bool:
        """Add audio to each scene's combined effects video."""
        print("\n🎵 Adding audio to combined scene videos...")
        
        if not VIDEO_AVAILABLE:
            print("⚠️ Video modules not available")
            self.results['step_status']['add_audio'] = False
            return False
        
        try:
            final_scenes_dir = os.path.join(self.output_directory, "final_scenes")
            os.makedirs(final_scenes_dir, exist_ok=True)
            
            scenes = self.flow_data.get('scenes', {}).get('scenes', [])
            final_scene_videos = []
            
            for i, scene in enumerate(scenes):
                scene_id = scene.get('id', i + 1)
                combined_video = scene.get('combined_effects_video')
                audio_path = scene.get('audio_path')
                
                if not combined_video or not os.path.exists(combined_video):
                    print(f"⚠️ No combined video for scene {scene_id}")
                    continue
                
                print(f"\n🔗 Scene {scene_id}: Adding audio...")
                print(f"   Video: {os.path.basename(combined_video)}")
                
                try:
                    final_filename = f"scene_{scene_id:02d}_final.mp4"
                    final_path = os.path.join(final_scenes_dir, final_filename)
                    
                    if audio_path and os.path.exists(audio_path):
                        print(f"   Audio: {os.path.basename(audio_path)}")
                        combine_audio_with_video(combined_video, audio_path, final_path)
                    else:
                        print(f"   No audio - using video only")
                        import shutil
                        shutil.copy2(combined_video, final_path)
                    
                    if os.path.exists(final_path):
                        scene['final_scene_video'] = final_path
                        final_scene_videos.append(final_path)
                        
                        # SAVE IMMEDIATELY after each final scene video
                        self.save_json_immediately(f"Saved final video for scene {scene_id}")
                        
                        print(f"   ✓ Created: {final_filename}")
                    else:
                        print(f"   ✗ Failed to create final scene video")
                    
                except Exception as e:
                    print(f"   ✗ Error adding audio to scene {scene_id}: {e}")
                    continue
            
            print(f"\n✓ Created {len(final_scene_videos)} final scene videos")
            
            self.flow_data['final_scene_videos'] = final_scene_videos
            self.save_json_immediately("Saved final scene videos list")
            
            self.results['step_status']['add_audio'] = len(final_scene_videos) > 0
            return len(final_scene_videos) > 0
            
        except Exception as e:
            error_msg = f"Adding audio failed: {e}"
            self.results['errors'].append(error_msg)
            print(f"✗ {error_msg}")
            self.results['step_status']['add_audio'] = False
            return False
    
    def create_final_complete_video(self) -> bool:
        """Combine all final scene videos into one complete video."""
        print("\n🎞️ Creating final complete video...")
        
        if not MOVIEPY_AVAILABLE:
            print("⚠️ MoviePy not available")
            self.results['step_status']['final_video'] = False
            return False
        
        try:
            final_scene_videos = self.flow_data.get('final_scene_videos', [])
            
            if not final_scene_videos:
                print("✗ No final scene videos to combine")
                self.results['step_status']['final_video'] = False
                return False
            
            existing_videos = [v for v in final_scene_videos if os.path.exists(v)]
            
            if not existing_videos:
                print("✗ No valid final scene videos found")
                self.results['step_status']['final_video'] = False
                return False
            
            final_video_path = os.path.join(self.output_directory, "final_complete_video.mp4")
            
            if len(existing_videos) == 1:
                import shutil
                shutil.copy2(existing_videos[0], final_video_path)
                print(f"✓ Single video copied as final")
            else:
                print(f"   Combining {len(existing_videos)} final scene videos...")
                
                clips = []
                total_duration = 0
                
                for i, video_path in enumerate(existing_videos):
                    try:
                        clip = VideoFileClip(video_path)
                        clips.append(clip)
                        total_duration += clip.duration
                        print(f"   ✓ Scene {i+1}: {os.path.basename(video_path)} ({clip.duration:.1f}s)")
                    except Exception as e:
                        print(f"   ✗ Error loading {os.path.basename(video_path)}: {e}")
                        continue
                
                if not clips:
                    print("✗ No clips could be loaded")
                    self.results['step_status']['final_video'] = False
                    return False
                
                print(f"   Total duration: {total_duration:.1f}s")
                
                final_clip = concatenate_videoclips(clips, method="compose")
                
                print(f"   Writing: {os.path.basename(final_video_path)}")
                final_clip.write_videofile(
                    final_video_path,
                    codec='libx264',
                    audio_codec='aac',
                    logger=None
                )
                
                for clip in clips:
                    clip.close()
                final_clip.close()
            
            if os.path.exists(final_video_path):
                self.flow_data['final_complete_video'] = final_video_path
                self.results['final_video'] = final_video_path
                self.results['step_status']['final_video'] = True
                
                # FINAL SAVE with complete video path
                self.save_json_immediately("Saved final complete video path")
                
                print(f"\n🎉 FINAL COMPLETE VIDEO CREATED!")
                print(f"   Path: {final_video_path}")
                print(f"   Contains: {len(existing_videos)} scenes")
                print(f"   Each scene has all 4 movement effects combined with audio")
                return True
            else:
                print("✗ Final video file not created")
                self.results['step_status']['final_video'] = False
                return False
                
        except Exception as e:
            error_msg = f"Final video creation failed: {e}"
            self.results['errors'].append(error_msg)
            print(f"✗ {error_msg}")
            self.results['step_status']['final_video'] = False
            return False
    
    async def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete video creation pipeline with all effects combined."""
        print("🚀 STARTING COMPLETE FLOW WITH ALL MOVEMENT EFFECTS")
        print("=" * 70)
        print("Pipeline:")
        print("1. Generate images (with smart retry and fallback to placeholders)")
        print("2. Create voiceovers (to get actual durations)")
        print("3. Create 4 movement effects per scene (divided by duration)")
        print("4. Combine all effects per scene (FIXED - no lambda errors)")
        print("5. Add scene audio to combined effect videos")
        print("6. Combine all scenes into final video")
        print("7. Save JSON IMMEDIATELY after each major step")
        print("=" * 70)
        
        start_time = datetime.now()
        
        if not self.load_flow_data():
            return self.results
        
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Update status to processing
        self.flow_data['status'] = 'processing'
        self.flow_data['started_timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_json_immediately("Started processing")
        
        await self.generate_all_images()
        self.create_all_voiceovers()
        self.create_and_combine_all_movement_videos()
        self.add_audio_to_combined_videos()
        self.create_final_complete_video()
        
        # Final save with all results
        critical_steps = ['create_movement', 'add_audio', 'final_video']
        success_count = sum(1 for step in critical_steps if self.results['step_status'].get(step, False))
        self.results['success'] = success_count >= 2 or self.results['final_video'] is not None
        
        self.flow_data['status'] = 'completed' if self.results['success'] else 'failed'
        self.flow_data['completed_timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.flow_data['processing_results'] = self.results
        
        self.save_json_immediately("FINAL SAVE - Processing complete")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("🎉 COMPLETE FLOW WITH ALL EFFECTS FINISHED!")
        print("=" * 70)
        print(f"⏱️  Total time: {duration:.1f} seconds")
        print(f"✅ Success: {self.results['success']}")
        print(f"🖼️  Images generated: {self.results['images_generated']}")
        print(f"📋 Placeholder images: {self.results['placeholders_created']}")
        print(f"❌ Image failures: {self.results['images_failed']}")
        print(f"🎤 Audio files created: {self.results['audios_generated']}")
        print(f"📹 Scenes with movement videos: {self.results['movement_videos_per_scene']}")
        print(f"🎬 Combined effect videos: {self.results['combined_scene_videos']}")
        print(f"🎵 Final scene videos: {len(self.flow_data.get('final_scene_videos', []))}")
        print(f"🎞️  Final complete video: {'✓' if self.results['final_video'] else '✗'}")
        
        if self.results['retries_performed'] > 0:
            print(f"🔄 API retries performed: {self.results['retries_performed']}")
        
        if self.results['final_video']:
            print(f"\n🎊 YOUR COMPLETE VIDEO IS READY!")
            print(f"📁 {self.results['final_video']}")
            print("\n🎬 This video contains:")
            print("   - All scenes with 4 movement effects each")
            print("   - Scene-specific audio synchronized perfectly")
            print("   - All scenes combined in sequence")
            print(f"\n💾 JSON file updated with ALL paths:")
            print(f"   {self.json_file_path}")
            print("   ✓ Reload the JSON file to see all generated file paths")
        
        if self.results['errors']:
            print(f"\n⚠️ Errors encountered:")
            for error in self.results['errors']:
                print(f"   - {error}")
        
        return self.results


async def main():
    """Main execution function."""
    json_path = "flow_description.json"
    
    if not os.path.exists(json_path):
        print(f"❌ Flow description file not found: {json_path}")
        print("Please ensure the file exists in the current directory.")
        return
    
    creator = CompleteFlowWithAllEffects(json_path)
    results = await creator.run_complete_pipeline()
    
    exit_code = 0 if results['success'] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())