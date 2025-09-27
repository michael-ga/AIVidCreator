"""
Protocol definitions for image generation interfaces.
"""

from typing import Protocol, List, Optional, Any, Tuple, Dict
from ..models import ImageGenerationRequest, ImageGenerationResponse, ImageFormat

class ImageGeneratorInterface(Protocol):
    """Protocol defining the interface for AI Image Generators"""
    
    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Generate images from detailed request"""
        ...
    
    def generate_simple(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        **kwargs: Any,
    ) -> ImageGenerationResponse:
        """Simplified image generation method"""
        ...
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        ...
    
    def get_supported_sizes(self) -> List[Tuple[int, int]]:
        """Get list of supported image dimensions"""
        ...
    
    def get_supported_formats(self) -> List[ImageFormat]:
        """Get list of supported output formats"""
        ...
