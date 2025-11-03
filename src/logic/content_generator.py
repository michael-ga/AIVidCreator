"""
Content Generator - Clean E2E Content Creation

A concise, directed class that generates complete video content JSON files
with all required data (script, scenes, images, descriptions, titles).
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional

from video_creation.utils.config_loader import VideoConfig
from video_creation.ai.prompts import get_similiar_script, get_scenes, get_description_prompt
from ai_models.config.settings import validate_environment
from ai_models.ai_interfaces.models import ImageGenerationRequest
from ai_models.ai_interfaces.llm.providers.gemini_llm import get_gemini_client
from ai_models.ai_interfaces.image.providers.gemini_image import get_gemini_image_generator


class ContentGenerator:
    """
    Generates complete video content with all required data.

    Creates a ready-to-use JSON file containing:
    - Generated script
    - Scenes with image prompts
    - Video description and titles
    - Generated images
    - Directory structure
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the content generator"""
        self.config = VideoConfig(config_path)
        self.llm_client = None
        self.image_generator = None
        self._setup_clients()

    def _setup_clients(self):
        """Setup AI clients (Gemini LLM + Image Generator)"""
        env_status = validate_environment()

        if env_status["google"]:
            try:
                self.llm_client = get_gemini_client()
                self.image_generator = get_gemini_image_generator()
                print("✓ Content generator initialized with Gemini")
            except Exception as e:
                print(f"⚠ Failed to initialize Gemini: {e}")
        else:
            print("⚠ GOOGLE_API_KEY not found. Please set it in your environment.")

    def _normalize_scenes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize scenes structure - unwrap nested scenes if present"""
        if not isinstance(data, dict):
            return data

        scenes_value = data.get('scenes')
        if isinstance(scenes_value, dict) and isinstance(scenes_value.get('scenes'), list):
            return {"scenes": scenes_value['scenes']}

        return data

    async def _generate_script(self, example_script: str, subject: str) -> str:
        """Generate new script based on example and subject"""
        if not self.llm_client:
            raise RuntimeError("LLM client not available")

        prompt = get_similiar_script(example_script, subject)
        response = await self.llm_client.generate(prompt, model=None, temperature=0.7)

        # Extract script from START/END markers if present
        text = response.text.strip()
        if "<START>" in text and "<END>" in text:
            start_idx = text.find("<START>") + len("<START>")
            end_idx = text.find("<END>")
            text = text[start_idx:end_idx].strip()

        return text

    async def _generate_scenes_data(self, script: str) -> Dict[str, Any]:
        """Generate scenes JSON from script"""
        if not self.llm_client:
            raise RuntimeError("LLM client not available")

        prompt = get_scenes(script)
        response = await self.llm_client.generate(prompt, model=None, temperature=0.5)

        # Extract JSON from response
        text = response.text.strip()

        # Try to extract from START/END markers
        if "<START>" in text and "<END>" in text:
            start_idx = text.find("<START>") + len("<START>")
            end_idx = text.find("<END>")
            text = text[start_idx:end_idx].strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:].strip()
        if text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        try:
            parsed = json.loads(text)
            return self._normalize_scenes(parsed) if isinstance(parsed, dict) else parsed
        except json.JSONDecodeError:
            # Fallback: try to find JSON object in text
            start_idx = text.find('{')
            if start_idx != -1:
                try:
                    # Simple extraction - find matching closing brace
                    brace_count = 0
                    for i in range(start_idx, len(text)):
                        if text[i] == '{':
                            brace_count += 1
                        elif text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                parsed = json.loads(text[start_idx:i + 1])
                                return self._normalize_scenes(parsed) if isinstance(parsed, dict) else parsed
                except Exception:
                    pass

            # Last resort fallback
            return {
                "scenes": [{
                    "id": 1,
                    "duration": 3,
                    "text": script[:100] + "...",
                    "image_prompt": f"Visual representation of: {script[:50]}..."
                }]
            }

    async def _generate_metadata(self, script: str, subject: str) -> Dict[str, Any]:
        """Generate video description and titles"""
        if not self.llm_client:
            return {
                "description": f"Video about {subject}",
                "titles": [f"Understanding {subject}", f"The Power of {subject}"]
            }

        # Generate description
        desc_prompt = get_description_prompt()
        desc_response = await self.llm_client.generate(
            f"{desc_prompt}\n\nScript: {script}",
            model=None,
            temperature=0.7
        )
        description = desc_response.text.strip()

        # Generate titles
        titles_prompt = f"""Based on this video script about {subject}, generate 5 creative and engaging titles.

Script: {script[:200]}...

Return only the titles, one per line, without numbering."""

        titles_response = await self.llm_client.generate(
            titles_prompt,
            model=None,
            temperature=0.4
        )

        titles = [
            title.strip()
            for title in titles_response.text.strip().split('\n')
            if title.strip()
        ][:7]  # Limit to 7 titles

        return {
            "description": description,
            "titles": titles
        }

    async def _generate_images(self, scenes_data: Dict[str, Any], output_dir: str) -> List[Dict[str, str]]:
        """Generate images for each scene"""
        if not self.image_generator:
            print("⚠ Image generator not available. Skipping image generation.")
            return []

        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        images_data = []
        scenes = scenes_data.get("scenes", [])

        for i, scene in enumerate(scenes):
            try:
                print(f"  🎨 Generating image {i + 1}/{len(scenes)}...")

                request = ImageGenerationRequest(
                    prompt=scene.get("image_prompt", f"Scene {i + 1}"),
                    width=1024,
                    height=1024,
                    num_images=1
                )

                response = await self.image_generator.generate(request)

                saved_paths = response.save_all(
                    images_dir,
                    f"scene_{scene.get('id', i + 1)}"
                )

                if saved_paths:
                    images_data.append({
                        "scene_id": scene.get("id", i + 1),
                        "image_path": saved_paths[0],
                        "prompt": scene.get("image_prompt", ""),
                        "text": scene.get("text", "")
                    })
                    print(f"    ✓ Saved: {os.path.basename(saved_paths[0])}")

            except Exception as e:
                print(f"    ❌ Error generating image for scene {i + 1}: {e}")
                continue

        return images_data

    async def generate(
        self,
        subject: str,
        example_script: str,
        output_subfolder: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate complete video content and save to JSON file.

        Args:
            subject: The topic/subject for the video
            example_script: Example script to base the new video on
            output_subfolder: Optional subfolder name for output

        Returns:
            Dictionary containing all generated content with file paths
        """
        print(f"\n🎬 Generating content for: '{subject}'")
        print("=" * 60)

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_subject = "".join(c for c in subject if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_subject = safe_subject.replace(' ', '_')[:30]

        if output_subfolder:
            output_dir = os.path.join(
                self.config.content_workitems,
                output_subfolder,
                f"{safe_subject}_{timestamp}"
            )
        else:
            output_dir = os.path.join(
                self.config.content_workitems,
                f"{safe_subject}_{timestamp}"
            )

        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")

        # Generate all content
        print("\n📝 Step 1: Generating script...")
        script = await self._generate_script(example_script, subject)
        print("✓ Script generated")

        print("\n🎭 Step 2: Creating scenes...")
        scenes_data = await self._generate_scenes_data(script)
        scenes_count = len(scenes_data.get('scenes', []))
        print(f"✓ Generated {scenes_count} scenes")

        print("\n📋 Step 3: Creating metadata...")
        metadata = await self._generate_metadata(script, subject)
        print("✓ Description and titles generated")

        print("\n🖼️ Step 4: Generating images...")
        images_data = await self._generate_images(scenes_data, output_dir)
        print(f"✓ Generated {len(images_data)} images")

        # Compile complete result
        result = {
            "subject": subject,
            "example_script": example_script,
            "generated_script": script,
            "scenes": scenes_data,
            "description": metadata["description"],
            "titles": metadata["titles"],
            "images": images_data,
            "output_directory": output_dir,
            "timestamp": timestamp,
            "status": "completed"
        }

        # Save to JSON file
        json_file = os.path.join(output_dir, "content.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Content saved to: {json_file}")
        print("\n🎉 Content generation completed!")
        print(f"📂 All files in: {output_dir}")

        return result


# Convenience function for async usage
async def generate_content(
    subject: str,
    example_script: str,
    output_subfolder: Optional[str] = None,
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate content.

    Args:
        subject: The topic/subject for the video
        example_script: Example script to base the new video on
        output_subfolder: Optional subfolder name for output
        config_path: Optional path to config file

    Returns:
        Dictionary containing all generated content
    """
    generator = ContentGenerator(config_path)
    return await generator.generate(subject, example_script, output_subfolder)


# if __name__ == "__main__":
#     # Example usage
#     async def main():
#         example_script = """What if responsibility wasn't a burden—but a superpower?

# In this powerful visual short, we journey through what it truly means to be responsible—not just keeping promises, but leading with integrity, owning your role, and rising above blame."""

#         result = await generate_content(
#             subject="courage",
#             example_script=example_script,
#             output_subfolder="test_content"
#         )

#         print("\n" + "=" * 60)
#         print("📊 GENERATION SUMMARY")
#         print("=" * 60)
#         print(f"Subject: {result['subject']}")
#         print(f"Scenes: {len(result['scenes'].get('scenes', []))}")
#         print(f"Images: {len(result['images'])}")
#         print(f"Titles: {len(result['titles'])}")
#         print(f"\nJSON file: {os.path.join(result['output_directory'], 'content.json')}")

#     asyncio.run(main())

