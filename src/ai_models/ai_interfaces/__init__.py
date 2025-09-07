"""
AI Interfaces Package - Standardized interfaces for AI providers.
"""

# Import LLM components
from .models import ChatMessage, LLMResponse, LLMStreamChunk
from .llm.protocols import LLMInterface, AsyncLLMInterface

# Import Image components
from .models import (
    ImageFormat, ImageGenerationRequest, GeneratedImage, ImageGenerationResponse
)
from .image.protocols import ImageGeneratorInterface

# Import exceptions
from .exceptions import (
    AIProviderError, RateLimitError, ModelNotFoundError, InvalidRequestError, AuthenticationError
)

# Convenience functions
def create_chat_message(role: str, content: str, **metadata) -> ChatMessage:
    """Convenience function to create a ChatMessage"""
    return ChatMessage(role=role, content=content, metadata=metadata or None)

def create_user_message(content: str) -> ChatMessage:
    """Create a user message"""
    return create_chat_message("user", content)

def create_system_message(content: str) -> ChatMessage:
    """Create a system message"""
    return create_chat_message("system", content)

def create_assistant_message(content: str) -> ChatMessage:
    """Create an assistant message"""
    return create_chat_message("assistant", content)

# Public API
__all__ = [
    # LLM exports
    "ChatMessage", "LLMResponse", "LLMStreamChunk",
    "LLMInterface", "AsyncLLMInterface",
    "create_chat_message", "create_user_message", "create_system_message", "create_assistant_message",
    
    # Image exports
    "ImageFormat", "ImageGenerationRequest", "GeneratedImage", "ImageGenerationResponse",
    "ImageGeneratorInterface",
    
    # Exception exports
    "AIProviderError", "RateLimitError", "ModelNotFoundError", "InvalidRequestError", "AuthenticationError"
]
