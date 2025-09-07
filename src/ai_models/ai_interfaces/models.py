"""
Data models for AI interfaces - all @dataclass definitions.
"""

from __future__ import annotations
import base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

# ============================================================================
# LLM Models
# ============================================================================

@dataclass
class ChatMessage:
    """Represents a message in a conversation"""
    role: str  # "system", "user", "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate role after initialization"""
        valid_roles = {"system", "user", "assistant"}
        if self.role not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}, got: {self.role}")

@dataclass
class LLMResponse:
    """Standardized response from LLM"""
    text: str
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: Any = None

    @property
    def total_tokens(self) -> int:
        """Get total tokens used"""
        return self.usage.get('total_tokens', 0)

    @property
    def prompt_tokens(self) -> int:
        """Get prompt tokens used"""
        return self.usage.get('prompt_tokens', 0)

    @property
    def completion_tokens(self) -> int:
        """Get completion tokens used"""
        return self.usage.get('completion_tokens', 0)

@dataclass
class LLMStreamChunk:
    """Represents a chunk in streaming response"""
    delta: str
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_chunk: Any = None

    @property
    def is_final(self) -> bool:
        """Check if this is the final chunk"""
        return self.finish_reason is not None

# ============================================================================
# Image Models
# ============================================================================

class ImageFormat(Enum):
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    
    @classmethod
    def from_string(cls, format_str: str) -> 'ImageFormat':
        """Create ImageFormat from string"""
        format_str = format_str.lower()
        for fmt in cls:
            if fmt.value == format_str:
                return fmt
        raise ValueError(f"Unsupported image format: {format_str}")

@dataclass
class ImageGenerationRequest:
    """Request parameters for image generation"""
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    num_images: int = 1
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    seed: Optional[int] = None
    style: Optional[str] = None
    format: ImageFormat = ImageFormat.PNG
    
    def __post_init__(self):
        """Validate parameters"""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")
        if self.num_images <= 0:
            raise ValueError("Number of images must be positive")
        if not (0.0 <= self.guidance_scale <= 20.0):
            raise ValueError("Guidance scale must be between 0.0 and 20.0")

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio"""
        return self.width / self.height

@dataclass
class GeneratedImage:
    """Represents a generated image with utilities"""
    data: bytes
    format: ImageFormat = ImageFormat.PNG
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_base64(self) -> str:
        """Convert image to base64 string"""
        return base64.b64encode(self.data).decode('utf-8')
    
    def save(self, filepath: str) -> None:
        """Save image to file"""
        with open(filepath, 'wb') as f:
            f.write(self.data)
    
    @property
    def size(self) -> Tuple[int, int]:
        """Get image dimensions as tuple"""
        return (self.width, self.height)
    
    @property
    def size_mb(self) -> float:
        """Get image size in megabytes"""
        return len(self.data) / (1024 * 1024)
    
    def __len__(self) -> int:
        """Get size of image data in bytes"""
        return len(self.data)

@dataclass
class ImageGenerationResponse:
    """Response from image generator"""
    images: List[GeneratedImage]
    model: str
    request: ImageGenerationRequest
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_images(self) -> int:
        """Get number of generated images"""
        return len(self.images)
    
    @property
    def total_size_mb(self) -> float:
        """Get total size of all images in megabytes"""
        return sum(img.size_mb for img in self.images)
    
    def save_all(self, directory: str, prefix: str = "generated") -> List[str]:
        """Save all images to directory"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        saved_paths = []
        for i, image in enumerate(self.images):
            filename = f"{prefix}_{i:03d}.{image.format.value}"
            filepath = os.path.join(directory, filename)
            image.save(filepath)
            saved_paths.append(filepath)
        
        return saved_paths
