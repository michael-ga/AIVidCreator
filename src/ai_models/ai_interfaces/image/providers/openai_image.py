"""
OpenAI DALL-E image generator implementation.
"""

from typing import List, Optional, Any, Tuple, Dict
import base64
import logging

from openai import OpenAI

from ...models import ImageGenerationRequest, ImageGenerationResponse, GeneratedImage, ImageFormat
from ...exceptions import AIProviderError, RateLimitError, AuthenticationError
from ..protocols import ImageGeneratorInterface
from config.settings import OPENAI_API_KEY, DEFAULT_OPENAI_IMAGE_MODEL

logger = logging.getLogger(__name__)

class OpenAIImageGenerator:
    """OpenAI DALL-E implementation of ImageGeneratorInterface"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_OPENAI_IMAGE_MODEL
    ):
        try:
            self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
            self.default_model = default_model
            self.provider_name = "OpenAI"
        except Exception as e:
            raise AuthenticationError(f"Failed to initialize OpenAI client: {str(e)}", "openai")
    
    def _handle_openai_error(self, error: Exception) -> None:
        """Convert OpenAI errors to our standard exceptions"""
        error_str = str(error).lower()
        
        if "rate limit" in error_str:
            raise RateLimitError(f"OpenAI rate limit exceeded: {str(error)}", "openai")
        elif "authentication" in error_str or "api key" in error_str:
            raise AuthenticationError(f"OpenAI authentication failed: {str(error)}", "openai")
        else:
            raise AIProviderError(f"OpenAI API error: {str(error)}", "openai")
    
    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        try:
            size_str = f"{request.width}x{request.height}"
            
            response = self.client.images.generate(
                model=self.default_model,
                prompt=request.prompt,
                size=size_str,
                n=request.num_images,
                response_format="b64_json"
            )
            
            images = []
            for img_data in response.data:
                if img_data.b64_json:
                    image_bytes = base64.b64decode(img_data.b64_json)
                    images.append(GeneratedImage(
                        data=image_bytes,
                        format=ImageFormat.PNG,
                        width=request.width,
                        height=request.height,
                        metadata={
                            "revised_prompt": getattr(img_data, "revised_prompt", None),
                            "provider": self.provider_name
                        }
                    ))
            
            return ImageGenerationResponse(
                images=images,
                model=self.default_model,
                request=request,
                metadata={"provider": self.provider_name}
            )
        except Exception as e:
            self._handle_openai_error(e)
    
    def generate_simple(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        **kwargs: Any,
    ) -> ImageGenerationResponse:
        request = ImageGenerationRequest(
            prompt=prompt,
            width=width,
            height=height,
            num_images=num_images,
            **kwargs
        )
        return self.generate(request)
    
    def get_available_models(self) -> List[str]:
        return ["dall-e-2", "dall-e-3"]
    
    def get_supported_sizes(self) -> List[Tuple[int, int]]:
        return [(1024, 1024), (1792, 1024), (1024, 1792)]
    
    def get_supported_formats(self) -> List[ImageFormat]:
        return [ImageFormat.PNG]
