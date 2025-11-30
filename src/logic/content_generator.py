"""
Content Generator - Clean E2E Content Creation

A concise, directed class that generates complete video content JSON files
with all required data (script, scenes, images, descriptions, titles).

Uses LLMInterface protocol for flexible LLM provider selection.
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from video_creation.utils.config_loader import VideoConfig
from video_creation.ai.prompts import get_similiar_script, get_scenes, get_description_prompt
from ai_models.config.settings import validate_environment
from ai_models.ai_interfaces.models import ImageGenerationRequest, LLMResponse
from ai_models.ai_interfaces.llm.protocols import AsyncLLMInterface, LLMInterface
from ai_models.ai_interfaces.image.protocols import ImageGeneratorInterface
from helpers.text_utils import (
    extract_text_from_response,
    extract_json_from_text,
    normalize_scenes_structure,
    parse_titles_from_text,
    clean_filename
)


def get_llm_provider(provider: Optional[str] = None) -> Union[AsyncLLMInterface, LLMInterface]:
    """
    Get LLM provider instance based on provider name.

    Args:
        provider: Provider name ('gemini', 'openai', 'copilot-cli', or None for auto-detect)

    Returns:
        LLM provider instance implementing LLMInterface or AsyncLLMInterface

    Raises:
        ValueError: If provider is not supported or not available
        RuntimeError: If no provider can be initialized
    """
    env_status = validate_environment()

    # Auto-detect if provider not specified
    if provider is None:
        if env_status["google"]:
            provider = "gemini"
        elif env_status["openai"]:
            provider = "openai"
        else:
            # Try copilot-cli as fallback (doesn't need API keys)
            try:
                from ai_models.ai_interfaces.llm.providers.copilot_cli_llm import get_copilot_cli_client
                return get_copilot_cli_client()
            except Exception:
                raise RuntimeError("No LLM provider available. Please set GOOGLE_API_KEY or OPENAI_API_KEY")

    provider = provider.lower().strip()

    if provider == "gemini":
        if not env_status["google"]:
            raise ValueError("Gemini provider requires GOOGLE_API_KEY. Please set it in your environment.")
        from ai_models.ai_interfaces.llm.providers.gemini_llm import get_gemini_client
        return get_gemini_client()

    elif provider == "openai":
        if not env_status["openai"]:
            raise ValueError("OpenAI provider requires OPENAI_API_KEY. Please set it in your environment.")
        from ai_models.ai_interfaces.llm.providers.openai_llm import OpenAILLM
        return OpenAILLM()

    elif provider in ("copilot", "copilot-cli"):
        from ai_models.ai_interfaces.llm.providers.copilot_cli_llm import get_copilot_cli_client
        return get_copilot_cli_client()

    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: 'gemini', 'openai', 'copilot-cli'")


def get_image_provider(provider: Optional[str] = None) -> Optional[ImageGeneratorInterface]:
    """
    Get image generator provider instance.

    Args:
        provider: Provider name ('gemini' or None for auto-detect)

    Returns:
        Image generator instance or None if not available
    """
    env_status = validate_environment()

    if provider is None:
        provider = "gemini" if env_status["google"] else None

    if provider is None:
        return None

    provider = provider.lower().strip()

    if provider == "gemini":
        if not env_status["google"]:
            return None
        from ai_models.ai_interfaces.image.providers.gemini_image import get_gemini_image_generator
        return get_gemini_image_generator()

    else:
        raise ValueError(f"Unknown image provider: {provider}. Supported: 'gemini'")


class ContentGenerator:
    """
    Generates complete video content with all required data.

    Creates a ready-to-use JSON file containing:
    - Generated script
    - Scenes with image prompts
    - Video description and titles
    - Generated images
    - Directory structure

    Uses LLMInterface protocol for flexible LLM provider selection.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        llm_provider: Optional[Union[str, Any]] = None,
        image_provider: Optional[str] = None
    ):
        """
        Initialize the content generator.

        Args:
            config_path: Optional path to video config file
            llm_provider: LLM provider name ('gemini', 'openai', 'copilot-cli'),
                         LLMInterface instance, or None for auto-detect (defaults to gemini)
            image_provider: Image provider name ('gemini') or None for auto-detect
        """
        self.config = VideoConfig(config_path)

        # Setup LLM client
        # Check if llm_provider is already an instance (has generate method)
        if llm_provider is not None and hasattr(llm_provider, 'generate'):
            self.llm_client = llm_provider
            provider_name = getattr(llm_provider, 'provider_name', getattr(llm_provider, '__class__', type(llm_provider)).__name__)
            print(f"✓ Using provided LLM client: {provider_name}")
        else:
            try:
                self.llm_client = get_llm_provider(llm_provider)
                provider_name = llm_provider or "auto-detected"
                print(f"✓ LLM provider initialized: {provider_name}")
            except Exception as e:
                print(f"⚠ Failed to initialize LLM provider: {e}")
                self.llm_client = None

        # Setup image generator
        try:
            self.image_generator = get_image_provider(image_provider)
            if self.image_generator:
                print(f"✓ Image generator initialized: {image_provider or 'auto-detected'}")
        except Exception as e:
            print(f"⚠ Image generator not available: {e}")
            self.image_generator = None

    async def _call_llm(self, prompt: str, model: Optional[str] = None, temperature: float = 0.7) -> LLMResponse:
        """Call LLM with automatic sync/async handling"""
        if not self.llm_client:
            raise RuntimeError("LLM client not available")

        # Check if it's an async interface
        if hasattr(self.llm_client, 'generate') and asyncio.iscoroutinefunction(self.llm_client.generate):
            return await self.llm_client.generate(prompt, model=model, temperature=temperature)
        else:
            # Sync interface - run in executor
            return await asyncio.to_thread(
                self.llm_client.generate,
                prompt,
                model=model,
                temperature=temperature
            )

    async def _generate_script(self, example_script: str, subject: str) -> str:
        """Generate new script based on example and subject"""
        prompt = get_similiar_script(example_script, subject)
        response = await self._call_llm(prompt, model=None, temperature=0.7)

        # Extract script from response using text utilities
        return extract_text_from_response(response.text, use_markers=True, remove_markdown=False)

    async def _generate_scenes_data(self, script: str) -> Dict[str, Any]:
        """Generate scenes JSON from script"""
        prompt = get_scenes(script)
        response = await self._call_llm(prompt, model=None, temperature=0.5)

        # Extract JSON from response using text utilities
        parsed = extract_json_from_text(response.text)

        if parsed and isinstance(parsed, dict):
            return normalize_scenes_structure(parsed)

        # Fallback if extraction fails
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
        desc_response = await self._call_llm(
            f"{desc_prompt}\n\nScript: {script}",
            model=None,
            temperature=0.7
        )
        description = desc_response.text.strip()

        # Generate titles
        titles_prompt = f"""Based on this video script about {subject}, generate 5 creative and engaging titles.

Script: {script[:200]}...

Return only the titles, one per line, without numbering."""

        titles_response = await self._call_llm(
            titles_prompt,
            model=None,
            temperature=0.4
        )

        titles = parse_titles_from_text(titles_response.text, max_titles=7)

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
        safe_subject = clean_filename(subject, max_length=30)

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
    config_path: Optional[str] = None,
    llm_provider: Optional[Union[str, Any]] = None,
    image_provider: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate content.

    Args:
        subject: The topic/subject for the video
        example_script: Example script to base the new video on
        output_subfolder: Optional subfolder name for output
        config_path: Optional path to config file
        llm_provider: LLM provider name or instance (defaults to auto-detect/gemini)
        image_provider: Image provider name (defaults to auto-detect/gemini)

    Returns:
        Dictionary containing all generated content
    """
    generator = ContentGenerator(
        config_path=config_path,
        llm_provider=llm_provider,
        image_provider=image_provider
    )
    return await generator.generate(subject, example_script, output_subfolder)


if __name__ == "__main__":
    # Example usage
    async def main():
        example_script = """What if responsibility wasn't a burden—but a superpower?

In this powerful visual short, we journey through what it truly means to be responsible—not just keeping promises, but leading with integrity, owning your role, and rising above blame."""

        result = await generate_content(
            subject="courage",
            example_script=example_script,
            output_subfolder="test_content"
        )

        print("\n" + "=" * 60)
        print("📊 GENERATION SUMMARY")
        print("=" * 60)
        print(f"Subject: {result['subject']}")
        print(f"Scenes: {len(result['scenes'].get('scenes', []))}")
        print(f"Images: {len(result['images'])}")
        print(f"Titles: {len(result['titles'])}")
        print(f"\nJSON file: {os.path.join(result['output_directory'], 'content.json')}")

    asyncio.run(main())

