"""
Combined LLM and image generation workflow.
"""

import os
import asyncio
from ai_interfaces import create_user_message
from ai_interfaces.llm.providers.gemini_llm import get_gemini_client
from ai_interfaces.image.providers.openai_image import OpenAIImageGenerator
from config.settings import validate_environment

async def llm_enhanced_image_workflow():
    """Use LLM to enhance prompts for image generation"""
    print("=== LLM-Enhanced Image Generation Workflow ===")
    
    env_status = validate_environment()
    if not env_status["google"] or not env_status["openai"]:
        print("Both Google and OpenAI API keys required for this demo.")
        return
    
    try:
        # Initialize both services
        gemini = get_gemini_client()
        img_gen = OpenAIImageGenerator()
        
        # Original simple prompt
        original_prompt = "a robot in a garden"
        print(f"Original prompt: {original_prompt}")
        
        # Enhance with Gemini
        enhanced, model_used = await gemini.enhance_prompt_text_only(
            original_prompt,
            target_model="DALL-E 3",
            wants_negative=False
        )
        print(f"Enhanced prompt: {enhanced}")
        print(f"Enhancement model: {model_used}")
        
        # Generate image with enhanced prompt
        os.makedirs("generated_images", exist_ok=True)
        
        response = img_gen.generate_simple(enhanced)
        saved_paths = response.save_all("generated_images", "enhanced_robot")
        
        print(f"Generated enhanced image: {saved_paths[0]}")
        
        # Compare with original prompt
        response_original = img_gen.generate_simple(original_prompt)
        saved_paths_original = response_original.save_all("generated_images", "original_robot")
        
        print(f"Generated original image: {saved_paths_original[0]}")
        print("Compare the two images to see the difference!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(llm_enhanced_image_workflow())
