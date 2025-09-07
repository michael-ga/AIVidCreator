"""
Configuration settings for AI providers.
"""

import os
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_api_key(key_name: str) -> Optional[str]:
    """Get API key from environment variables"""
    return os.getenv(key_name)

# API Keys
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
GOOGLE_API_KEY = get_api_key("GOOGLE_API_KEY")

# Default models
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_IMAGE_MODEL = "dall-e-3"
DEFAULT_GEMINI_MODEL = "gemini-1.5-pro-latest"

# Gemini model configurations
GEMINI_VISION_MODELS_TO_TRY = ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest"]
GEMINI_TEXT_MODELS_TO_TRY = [
    "gemini-2.5-flash-lite",
    "gemma-3n-e2b-it",
    "gemma-3n-e4b-it",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
    "gemma-3-1b-it",
    "gemini-2.5-pro-preview-tts",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.0-flash-thinking-exp-1219",
    "gemini-2.0-flash-thinking-exp",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-exp-1206",
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.0-pro-exp",
    "gemini-2.0-flash-lite-preview",
    "gemini-2.0-flash-lite-preview-02-05",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-exp-image-generation",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-03-25",
    "gemini-1.5-flash-8b-latest",
    "gemini-1.5-flash-8b-001",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-1.5-pro-002",
    "gemini-1.5-pro-latest"
]

def validate_environment() -> Dict[str, bool]:
    """Check which API keys are available"""
    return {
        "openai": bool(OPENAI_API_KEY),
        "google": bool(GOOGLE_API_KEY)
    }

def get_provider_config(provider: str) -> Dict[str, Any]:
    """Get configuration for a specific provider"""
    configs = {
        "openai": {
            "api_key": OPENAI_API_KEY,
            "chat_model": DEFAULT_OPENAI_CHAT_MODEL,
            "image_model": DEFAULT_OPENAI_IMAGE_MODEL
        },
        "google": {
            "api_key": GOOGLE_API_KEY,
            "model": DEFAULT_GEMINI_MODEL
        }
    }
    return configs.get(provider, {})
