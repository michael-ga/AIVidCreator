"""
Corrected FREE Gemini Image Generator with proper aspect ratio support.
Based on proven ComfyUI implementation patterns.
"""

import asyncio
import io
import logging
from typing import List, Optional, Any, Tuple, Dict
from PIL import Image

from ...models import ImageGenerationRequest, ImageGenerationResponse, GeneratedImage, ImageFormat
from ...exceptions import AIProviderError, RateLimitError, AuthenticationError
from ..protocols import ImageGeneratorInterface
from config.settings import GOOGLE_API_KEY

logger = logging.getLogger(__name__)

class GeminiImageGenerator:
    """
    FREE Gemini Image Generator with proper aspect ratio control.
    Multi-layered approach: strong prompting + optional retry + fallback cropping.
    """
    
    def __init__(self, max_concurrent_requests: int = 2, auto_retry: bool = True, force_aspect: bool = False):
        """
        Initialize with comprehensive aspect ratio control options.
        
        Args:
            max_concurrent_requests: Concurrent API calls limit for free tier
            auto_retry: If True, retry with corrective prompt if aspect ratio doesn't match
            force_aspect: If True, crop to exact dimensions as final fallback
        """
        api_key = GOOGLE_API_KEY
        if not api_key:
            raise AuthenticationError("GOOGLE_API_KEY not found in environment variables", "google")

        try:
            from google import genai
            from google.genai import types
            
            self.client = genai.Client(api_key=api_key)
            self.types = types
            logger.info("Successfully initialized FREE Gemini image generation with aspect ratio control")
            
        except ImportError as e:
            raise RuntimeError(
                "Missing google-genai library. Install with: pip install google-genai"
            ) from e
        except Exception as e:
            raise AuthenticationError(f"Failed to initialize Gemini client: {e}", "google")
        
        self.model_name = "gemini-2.0-flash-preview-image-generation"
        self.provider_name = "Google Gemini"
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.auto_retry = auto_retry
        self.force_aspect = force_aspect

    def _handle_gemini_error(self, error: Exception) -> None:
        """Convert Gemini errors to standard exceptions"""
        error_str = str(error).lower()
        
        if "429" in str(error) or "quota" in error_str or "rate limit" in error_str:
            retry_after = 60
            import re
            delay_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', str(error))
            if delay_match:
                try:
                    retry_after = int(delay_match.group(1))
                except ValueError:
                    pass
            
            raise RateLimitError(f"Gemini free tier rate limit: {str(error)}", "google", retry_after)
        elif "authentication" in error_str or "api key" in error_str:
            raise AuthenticationError(f"Gemini authentication failed: {str(error)}", "google")
        else:
            raise AIProviderError(f"Gemini image generation error: {str(error)}", "google")

    def _build_strong_dimensional_prompt(self, request: ImageGenerationRequest) -> str:
        """Build prompt with explicit dimensional instructions like successful ComfyUI"""
        
        # Calculate and name the aspect ratio
        aspect_ratio = request.width / request.height
        
        # Precise aspect ratio descriptions
        if abs(aspect_ratio - 1.0) < 0.01:
            aspect_desc = "square format"
        elif abs(aspect_ratio - 1.33) < 0.05:  # 4:3
            aspect_desc = "4:3 landscape format" if request.width > request.height else "3:4 portrait format"
        elif abs(aspect_ratio - 1.78) < 0.05:  # 16:9
            aspect_desc = "16:9 widescreen landscape format" if request.width > request.height else "9:16 portrait format"
        else:
            aspect_desc = f"{request.width}:{request.height} aspect ratio"
        
        # Strong, explicit dimensional instructions
        parts = [
            f"Generate a high-quality, photorealistic image",
            f"Output dimensions: exactly {request.width}x{request.height} pixels",
            f"Aspect ratio: {aspect_desc}",
            f"Do not output a square image unless the requested ratio is 1:1",
            f"Content: {request.prompt}"
        ]
        
        if request.style:
            parts.append(f"Style: {request.style}")
        
        if request.negative_prompt:
            parts.append(f"Avoid: {request.negative_prompt}")
        
        parts.append("Professional quality, sharp details, vibrant colors")
        
        return ". ".join(parts)

    def _create_corrective_prompt(self, request: ImageGenerationRequest, actual_width: int, actual_height: int) -> str:
        """Create corrective prompt for retry attempts"""
        return (
            f"The previous image was {actual_width}x{actual_height} which is incorrect. "
            f"Regenerate the image at exactly {request.width}x{request.height} pixels "
            f"(aspect ratio {request.width}:{request.height}), not square. "
            f"Content: {request.prompt}. "
            f"Ensure the output matches the requested dimensions precisely."
        )

    def _aspect_ratios_match(self, actual_w: int, actual_h: int, target_w: int, target_h: int, tolerance: float = 0.02) -> bool:
        """Check if aspect ratios match within tolerance"""
        actual_ratio = actual_w / actual_h
        target_ratio = target_w / target_h
        return abs(actual_ratio - target_ratio) <= tolerance

    def _crop_to_exact_aspect(self, image_data: bytes, target_width: int, target_height: int) -> bytes:
        """Crop image to exact aspect ratio (center crop)"""
        try:
            img = Image.open(io.BytesIO(image_data))
            current_w, current_h = img.size
            target_ratio = target_width / target_height
            current_ratio = current_w / current_h
            
            if abs(current_ratio - target_ratio) < 0.01:
                return image_data
            
            if target_ratio > current_ratio:
                # Target is wider - crop height
                new_height = int(current_w / target_ratio)
                top = (current_h - new_height) // 2
                crop_box = (0, top, current_w, top + new_height)
            else:
                # Target is taller - crop width
                new_width = int(current_h * target_ratio)
                left = (current_w - new_width) // 2
                crop_box = (left, 0, left + new_width, current_h)
            
            cropped = img.crop(crop_box)
            output = io.BytesIO()
            cropped.save(output, format="PNG")
            return output.getvalue()
            
        except Exception as e:
            logger.warning(f"Aspect ratio cropping failed: {e}")
            return image_data

    def _extract_images_from_response(self, response, request: ImageGenerationRequest) -> List[GeneratedImage]:
        """Extract images using proven ComfyUI method with optional processing"""
        images = []
        
        try:
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                try:
                                    # Get image binary data (proven ComfyUI method)
                                    image_binary = part.inline_data.data
                                    
                                    # Process through PIL for consistency
                                    pil_image = Image.open(io.BytesIO(image_binary))
                                    if pil_image.mode != "RGB":
                                        pil_image = pil_image.convert("RGB")
                                    
                                    # Convert to bytes
                                    img_byte_arr = io.BytesIO()
                                    pil_image.save(img_byte_arr, format='PNG')
                                    processed_data = img_byte_arr.getvalue()
                                    
                                    # Apply force_aspect cropping if enabled and needed
                                    was_cropped = False
                                    if self.force_aspect and not self._aspect_ratios_match(
                                        pil_image.width, pil_image.height, request.width, request.height
                                    ):
                                        processed_data = self._crop_to_exact_aspect(
                                            processed_data, request.width, request.height
                                        )
                                        was_cropped = True
                                        # Re-open to get actual dimensions after crop
                                        pil_image = Image.open(io.BytesIO(processed_data))
                                    
                                    images.append(GeneratedImage(
                                        data=processed_data,
                                        format=ImageFormat.PNG,
                                        width=pil_image.width,
                                        height=pil_image.height,
                                        seed=request.seed,
                                        metadata={
                                            "provider": self.provider_name,
                                            "model": self.model_name,
                                            "free_generation": True,
                                            "requested_size": f"{request.width}x{request.height}",
                                            "actual_size": f"{pil_image.width}x{pil_image.height}",
                                            "aspect_match": self._aspect_ratios_match(
                                                pil_image.width, pil_image.height, request.width, request.height
                                            ),
                                            "force_cropped": was_cropped
                                        }
                                    ))
                                    
                                except Exception as img_error:
                                    logger.error(f"Error processing image: {img_error}")
                                    continue
            
            return images
            
        except Exception as e:
            raise AIProviderError(f"Failed to extract images from response: {str(e)}", "google")

    async def _generate_single_image(self, prompt_text: str, request: ImageGenerationRequest) -> List[GeneratedImage]:
        """Generate a single image with given prompt"""
        generation_config = self.types.GenerateContentConfig(
            temperature=min(request.guidance_scale / 10.0, 1.0),
            response_modalities=['Text', 'Image']
        )
        
        parts = [{"text": prompt_text}]
        content_parts = [{"parts": parts}]
        
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model_name,
            contents=content_parts,
            config=generation_config
        )
        
        return self._extract_images_from_response(response, request)

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Generate FREE images with comprehensive aspect ratio control"""
        async with self._semaphore:
            try:
                logger.info(f"Generating {request.num_images} FREE image(s) at {request.width}x{request.height}")
                
                all_images = []
                
                for i in range(request.num_images):
                    try:
                        # First attempt with strong dimensional prompt
                        primary_prompt = self._build_strong_dimensional_prompt(request)
                        logger.debug(f"Primary prompt: {primary_prompt[:150]}...")
                        
                        images = await self._generate_single_image(primary_prompt, request)
                        
                        # Check if retry is needed and enabled
                        if (self.auto_retry and images and 
                            not self._aspect_ratios_match(
                                images[0].width, images[0].height, request.width, request.height
                            )):
                            
                            logger.info(f"First attempt: {images[0].width}x{images[0].height}, retrying for {request.width}x{request.height}")
                            
                            # Retry with corrective prompt
                            corrective_prompt = self._create_corrective_prompt(
                                request, images[0].width, images[0].height
                            )
                            logger.debug(f"Corrective prompt: {corrective_prompt[:150]}...")
                            
                            retry_images = await self._generate_single_image(corrective_prompt, request)
                            if retry_images:
                                images = retry_images  # Use retry result
                                logger.info(f"Retry result: {images[0].width}x{images[0].height}")
                        
                        if images:
                            all_images.extend(images)
                            actual_img = images[0]
                            logger.info(f"Generated image {i+1}/{request.num_images}: "
                                      f"{actual_img.width}x{actual_img.height} "
                                      f"(requested: {request.width}x{request.height})")
                        
                        # Respectful delay for free tier
                        if i < request.num_images - 1:
                            await asyncio.sleep(2.0)
                            
                    except Exception as e:
                        logger.error(f"Failed to generate image {i+1}: {e}")
                        continue
                
                if not all_images:
                    raise RuntimeError("Failed to generate any images")
                
                # Calculate success metrics
                successful_aspects = sum(
                    1 for img in all_images 
                    if self._aspect_ratios_match(img.width, img.height, request.width, request.height)
                )
                
                return ImageGenerationResponse(
                    images=all_images,
                    model=self.model_name,
                    request=request,
                    metadata={
                        "provider": self.provider_name,
                        "generated_count": len(all_images),
                        "requested_count": request.num_images,
                        "aspect_ratio_success_rate": f"{successful_aspects}/{len(all_images)}",
                        "total_size_mb": sum(img.size_mb for img in all_images),
                        "cost": "FREE!",
                        "free_tier": True,
                        "auto_retry_enabled": self.auto_retry,
                        "force_aspect_enabled": self.force_aspect
                    }
                )
                
            except Exception as e:
                self._handle_gemini_error(e)

    async def generate_simple(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        **kwargs: Any,
    ) -> ImageGenerationResponse:
        """Simplified FREE image generation with aspect ratio control"""
        request = ImageGenerationRequest(
            prompt=prompt,
            width=width,
            height=height,
            num_images=num_images,
            **kwargs
        )
        return await self.generate(request)

    def get_available_models(self) -> List[str]:
        return [self.model_name]

    def get_supported_sizes(self) -> List[Tuple[int, int]]:
        """Realistically supported dimensions based on testing"""
        return [
            # Square formats
            (512, 512), (768, 768), (1024, 1024),
            # 4:3 formats (corrected)
            (1024, 768), (768, 1024),
            # 16:9 formats (corrected)
            (1280, 720), (720, 1280),
            (1920, 1080), (1080, 1920),
            # Ultra-wide formats
            (1792, 1024), (1024, 1792)
        ]

    def get_supported_formats(self) -> List[ImageFormat]:
        return [ImageFormat.PNG]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_name,
            "provider": self.provider_name,
            "type": "image_generation",
            "cost": "FREE!",
            "free_tier": True,
            "aspect_ratio_support": "Full support with strong prompting",
            "features": [
                "text_to_image", "style_control", "negative_prompts",
                "aspect_ratio_control", "auto_retry", "force_aspect_cropping"
            ],
            "quality": "High-quality photorealistic image generation - completely FREE!",
            "success_strategies": [
                "Strong dimensional prompting",
                "Auto-retry with corrective prompts", 
                "Optional force-aspect cropping"
            ]
        }

# Global instance management with configuration options
_gemini_image_client = None
_current_config = None

def get_gemini_image_generator(auto_retry: bool = True, force_aspect: bool = False) -> GeminiImageGenerator:
    """
    Get or create FREE Gemini image generator with aspect ratio control.
    
    Args:
        auto_retry: Enable automatic retry with corrective prompts
        force_aspect: Enable force cropping to exact dimensions as fallback
    """
    global _gemini_image_client, _current_config
    
    new_config = (auto_retry, force_aspect)
    if _gemini_image_client is None or _current_config != new_config:
        _gemini_image_client = GeminiImageGenerator(auto_retry=auto_retry, force_aspect=force_aspect)
        _current_config = new_config
    
    return _gemini_image_client
