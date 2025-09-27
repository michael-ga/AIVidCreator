"""
FREE Gemini Image Generation examples.
"""

import asyncio
import os
from ai_interfaces.models import ImageGenerationRequest
from ai_interfaces.image.providers.gemini_image import get_gemini_image_generator
from config.settings import validate_environment

async def test_free_gemini_generation():
    """Test FREE Gemini image generation"""
    print("=== FREE Gemini Image Generation ===")
    
    env_status = validate_environment()
    if not env_status["google"]:
        print("Google API key not found. Set GOOGLE_API_KEY in .env")
        return
    
    try:
        img_gen = get_gemini_image_generator()
        os.makedirs("generated_images", exist_ok=True)
        
        # Show FREE model info
        model_info = img_gen.get_model_info()
        print(f"Model: {model_info['name']}")
        print(f"Cost: {model_info['cost']}")
        print(f"Quality: {model_info['quality']}")
        
        # Simple FREE generation
        print("\n1. Simple FREE generation:")
        response = await img_gen.generate_simple(
            "A beautiful sunset over mountains, photorealistic",
            width=1024,
            height=768
        )
        
        saved_paths = response.save_all("generated_images", "free_gemini")
        print(f"✅ Generated {len(saved_paths)} FREE images:")
        for i, path in enumerate(saved_paths):
            img = response.images[i]
            print(f"  - {path} ({img.width}x{img.height}, {img.size_mb:.2f}MB)")
        
        # Advanced FREE generation
        print("\n2. Advanced FREE generation with style:")
        request = ImageGenerationRequest(
            prompt="A majestic dragon in a fantasy landscape",
            negative_prompt="blurry, low quality, cartoon",
            width=1024,
            height=1024,
            style="photorealistic fantasy art, detailed"
        )
        
        response = await img_gen.generate(request)
        saved_paths = response.save_all("generated_images", "free_dragon")
        
        print(f"✅ Advanced FREE generation completed:")
        print(f"  - {saved_paths[0]} ({response.images[0].size})")
        print(f"  - Total cost: {response.metadata['cost']}")
        
        print(f"\n🎉 All FREE image generation successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_free_gemini_generation())
