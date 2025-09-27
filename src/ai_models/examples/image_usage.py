"""
Image generation examples.
"""

import os
from ai_interfaces.models import ImageGenerationRequest
from ai_interfaces.image.providers.openai_image import OpenAIImageGenerator
from config.settings import validate_environment

def basic_image_generation():
    """Basic image generation example"""
    print("=== Basic Image Generation ===")
    
    env_status = validate_environment()
    if not env_status["openai"]:
        print("OpenAI API key not found.")
        return
    
    try:
        # Initialize generator
        img_gen = OpenAIImageGenerator()
        
        # Create output directory
        os.makedirs("generated_images", exist_ok=True)
        
        # Simple generation
        response = img_gen.generate_simple(
            "A serene mountain landscape at sunset, digital art style",
            width=1024,
            height=768
        )
        
        # Save images
        saved_paths = response.save_all("generated_images", "landscape")
        print(f"Generated {len(saved_paths)} images:")
        for path in saved_paths:
            print(f"  - {path}")
        
        # Advanced generation
        request = ImageGenerationRequest(
            prompt="A futuristic cyberpunk city with neon lights",
            width=1792,
            height=1024,
            num_images=1
        )
        
        response = img_gen.generate(request)
        print(f"\nAdvanced generation completed. Total size: {response.total_size_mb:.2f}MB")
        
        # Model info
        models = img_gen.get_available_models()
        sizes = img_gen.get_supported_sizes()
        print(f"Available models: {models}")
        print(f"Supported sizes: {sizes}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    basic_image_generation()
