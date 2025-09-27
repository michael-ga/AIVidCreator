"""
LLM usage examples - Clean async version.
"""

import asyncio
from ai_interfaces import create_user_message, create_system_message
from ai_interfaces.llm.providers.gemini_llm import get_gemini_client
from config.settings import validate_environment

async def basic_gemini_usage():
    """Basic Gemini LLM usage - now fully async"""
    print("=== Basic Gemini LLM Usage ===")
    
    try:
        gemini = get_gemini_client()
        
        # Simple generation - just await!
        response = await gemini.generate(
            "What is machine learning?", 
            model="gemini-1.5-flash-latest",
            temperature=0.3
        )
        print(f"Response: {response.text[:200]}...")
        print(f"Model: {response.model}")
        print(f"Tokens: {response.total_tokens}")
        
        # Conversation - just await!
        messages = [
            create_system_message("You are a helpful coding assistant."),
            create_user_message("Explain Python decorators briefly.")
        ]
        response = await gemini.generate(messages, model="gemini-1.5-flash-latest")
        print(f"\nConversation response: {response.text[:200]}...")
        
        # Streaming - async iterator!
        print("\nStreaming: ", end="", flush=True)
        async for chunk in gemini.stream("Tell me about quantum computing", model="gemini-1.5-flash-latest"):
            print(chunk.delta, end="", flush=True)
        print()
        
    except Exception as e:
        print(f"Error: {e}")

async def prompt_enhancement_demo():
    """Demonstrate prompt enhancement - clean async"""
    print("\n=== Prompt Enhancement Demo ===")
    
    try:
        gemini = get_gemini_client()
        
        # Text-only enhancement - just await, no asyncio.run()!
        original_prompt = "a cat sitting on a chair"
        enhanced, model_used = await gemini.enhance_prompt_text_only(
            original_prompt,
            target_model="DALL-E 3",
            wants_negative=True
        )
        print(f"Original: {original_prompt}")
        print(f"Enhanced: {enhanced}")
        print(f"Model used: {model_used}")
        
        # Enhancement via interface - just await!
        response = await gemini.generate(
            "a dragon in the sky",
            target_model="Midjourney",
            wants_negative=True
        )
        print(f"\nInterface enhancement: {response.text}")
        
    except Exception as e:
        print(f"Error: {e}")

async def main():
    """Single async main function - the clean way"""
    env_status = validate_environment()
    
    if env_status["google"]:
        await basic_gemini_usage()
        await prompt_enhancement_demo()
    else:
        print("Google API key not found. Set GOOGLE_API_KEY in .env")

if __name__ == "__main__":
    # Single asyncio.run() call - no more nested loops!
    asyncio.run(main())
