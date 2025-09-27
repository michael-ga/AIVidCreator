"""
Clean async-first Gemini implementation - much simpler and better for production.
"""

import google.generativeai as genai
import logging
import asyncio
import random
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Tuple
from PIL import Image

from ...models import ChatMessage, LLMResponse, LLMStreamChunk
from ...exceptions import AIProviderError, AuthenticationError, RateLimitError
from ..protocols import LLMInterface, AsyncLLMInterface
from config.settings import GOOGLE_API_KEY, GEMINI_TEXT_MODELS_TO_TRY, GEMINI_VISION_MODELS_TO_TRY

logger = logging.getLogger(__name__)

# Keep your original system prompts unchanged
SYSTEM_PROMPT = """
You are an expert at creating detailed prompts for AI image generation for {model_name} model. Your task is to improve the user's prompt to generate higher quality, more detailed images.

Guidelines:
1. Keep the core meaning and intent of the original prompt
2. Add descriptive details about lighting, composition, and style
3. Include technical photography terms when appropriate
4. Enhance with artistic style suggestions
5. Make the prompt more specific and detailed
6. If an image is provided, consider its content when improving the prompt

{negative_prompt_instruction}

Return only the improved prompt{negative_format_instruction} do not return any other comments or not related text beside the prompts!.
"""

IMAGE_SYSTEM_PROMPT = """
You are a master-level visual prompt engineer for {model_name} model — a state-of-the-art image generation and editing model. Your mission is to transform user intent and reference inputs into high-impact, visually coherent narrative prompts optimized for in-context image synthesis.

## YOUR ENHANCED EDITING FLOW

1. **Extract Creative Intent and Preferences**
   - Identify the user's emotional tone, composition goals, preferred color palettes, and artistic style references (e.g., "baroque oil painting", "cinematic lens flares", "muted pastel tones").
   - If these are unclear or missing, ask direct clarifying questions to ground the creative direction.

2. **Analyze Reference Image (if provided)**
   - Determine the subject, pose, facial identity (if applicable), texture details, environment, lighting setup, and focal depth.
   - Note stylistic traits (e.g., "grainy texture," "soft backlight," "motion blur"), layout balance, and visual storytelling cues.

3. **Align with Desired Edits**
   - Parse the user's prompt to classify the intent:  
     (a) **Style Transfer**  
     (b) **Text Editing**  
     (c) **Object/Clothing Replacement**  
     (d) **Background Swapping**  
     (e) **Character Consistency / Iterative Editing**  
   - Distinguish between local edits (preserve context) and generative edits (rebuild context with fidelity).

4. **Apply Layered Descriptive Detailing**
   - Use clear, actionable phrasing with compound descriptors (e.g., "fractured obsidian armor," "luminescent ink swirling in liquid gravity").
   - Describe the subject first, then environment, then atmosphere, and finally style.

5. **Construct the Prompt Structure**
   Compose a **single cohesive paragraph** that follows this flow:
   - **[Subject & Action]**: "A serene astronaut floats weightlessly…"
   - **[Environment & Context]**: "…inside a stained-glass cathedral orbiting Jupiter."
   - **[Atmosphere & Lighting]**: "Reflections of nebulae ripple across her golden visor under soft, prismatic light."
   - **[Artistic Style & Medium]**: "Hyperreal concept art with painterly textures and soft volumetric fog."

{negative_prompt_instruction}

## OUTPUT GUIDELINES
- Return **only** the narrative prompt{negative_format_instruction} — no commentary, no greetings, no explanations.
- Ensure prompts are compatible with iterative editing and character-preserving workflows.
"""

class GeminiLLM:
    """Clean async-first Gemini implementation - simple and production-ready"""
    
    def __init__(self, max_concurrent_requests: int = 10):
        """Initialize with simple concurrency control"""
        api_key = GOOGLE_API_KEY
        if not api_key:
            raise AuthenticationError("GOOGLE_API_KEY not found in environment variables", "google")

        genai.configure(api_key=api_key)
        
        # Simple concurrency control - no complex ThreadPoolExecutor needed!
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Initialize available models
        self.available_vision_models = self._initialize_models(GEMINI_VISION_MODELS_TO_TRY, "vision")
        self.available_text_models = self._initialize_models(GEMINI_TEXT_MODELS_TO_TRY, "text")
        
        if not self.available_text_models and not self.available_vision_models:
            raise RuntimeError("No Gemini models could be initialized")

    def _initialize_models(self, model_names: List[str], model_type: str) -> List[Dict[str, Any]]:
        """Initialize and validate Gemini models (unchanged)"""
        models_to_try = []
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                models_to_try.append({
                    "name": f"Gemini ({model_name})", 
                    "model": model,
                    "model_name": model_name
                })
                logger.info(f"Successfully initialized {model_type} model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini {model_type} model {model_name}. Error: {e}")
        
        return models_to_try

    def _prepare_messages(self, messages: Union[str, List[ChatMessage]]) -> str:
        """Convert messages to Gemini format (unchanged)"""
        if isinstance(messages, str):
            return messages
        
        formatted_parts = []
        for msg in messages:
            if msg.role == "system":
                formatted_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                formatted_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                formatted_parts.append(f"Assistant: {msg.content}")
        
        return "\n\n".join(formatted_parts)

    def _select_model_for_generation(self, model: Optional[str], needs_vision: bool = False) -> Dict[str, Any]:
        """Select appropriate model for generation (unchanged)"""
        if model:
            all_models = self.available_vision_models if needs_vision else self.available_text_models
            for model_info in all_models:
                if model in model_info["model_name"]:
                    return model_info
        
        if needs_vision and self.available_vision_models:
            return random.choice(self.available_vision_models)
        elif self.available_text_models:
            return random.choice(self.available_text_models)
        else:
            raise RuntimeError("No suitable models available")

    def _handle_gemini_error(self, error: Exception) -> None:
        """Convert Gemini errors to our standard exceptions (unchanged)"""
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
            
            raise RateLimitError(f"Gemini rate limit exceeded: {str(error)}", "google", retry_after)
        elif "authentication" in error_str or "api key" in error_str:
            raise AuthenticationError(f"Gemini authentication failed: {str(error)}", "google")
        else:
            raise AIProviderError(f"Gemini API error: {str(error)}", "google")

    # ============================================================================
    # Clean Async Interface - No More Event Loop Conflicts!
    # ============================================================================

    async def generate(
        self,
        messages: Union[str, List[ChatMessage]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Clean async generate - no more nested asyncio.run() calls!"""
        
        # Check if this is a prompt enhancement request
        if any(key in kwargs for key in ['target_model', 'user_negative_prompt', 'wants_negative']):
            return await self._generate_enhanced_prompt(messages, model, **kwargs)
        
        # Standard generation with simple concurrency control
        async with self._semaphore:
            return await self._generate_internal(messages, model, temperature, max_tokens, **kwargs)

    async def _generate_internal(
        self,
        messages: Union[str, List[ChatMessage]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Internal async generation using modern asyncio.to_thread"""
        prompt = self._prepare_messages(messages)
        needs_vision = 'image_path' in kwargs or 'image' in kwargs
        
        selected_model = self._select_model_for_generation(model, needs_vision)
        
        try:
            generation_config = {"temperature": temperature}
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens

            if needs_vision and 'image_path' in kwargs:
                # Load image asynchronously using modern asyncio.to_thread
                image = await asyncio.to_thread(self._load_image, kwargs['image_path'])
                content = [prompt, image]
            else:
                content = prompt

            # Call Gemini API using modern asyncio.to_thread - much cleaner!
            response = await asyncio.to_thread(
                selected_model["model"].generate_content,
                content,
                generation_config=generation_config
            )
            
            if not response or not response.text:
                raise RuntimeError("Empty response from Gemini")
            
            usage_info = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage_info = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                }
            
            return LLMResponse(
                text=response.text.strip(),
                finish_reason="stop",
                usage=usage_info,
                model=selected_model["model_name"],
                metadata={
                    "provider": "Google Gemini",
                    "temperature": temperature,
                    "multimodal": needs_vision
                },
                raw_response=response
            )
            
        except Exception as e:
            self._handle_gemini_error(e)

    def _load_image(self, image_path: str) -> Image.Image:
        """Load and prepare image - runs in thread via asyncio.to_thread"""
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    async def stream(
        self,
        messages: Union[str, List[ChatMessage]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Async streaming"""
        try:
            response = await self.generate(messages, model, temperature, max_tokens, **kwargs)
            
            # Simulate streaming by chunking response
            text = response.text
            chunk_size = 50
            
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                is_final = (i + chunk_size) >= len(text)
                
                yield LLMStreamChunk(
                    delta=chunk,
                    finish_reason="stop" if is_final else None,
                    metadata={
                        "chunk_index": i // chunk_size,
                        "is_final": is_final,
                        "provider": "Google Gemini"
                    },
                    raw_chunk=None
                )
                
        except Exception as e:
            yield LLMStreamChunk(
                delta="",
                finish_reason="error",
                metadata={"error": str(e), "provider": "Google Gemini"},
                raw_chunk=None
            )

    # ============================================================================
    # Clean Async Prompt Enhancement - No More Complexity!
    # ============================================================================

    async def _generate_enhanced_prompt(
        self, 
        messages: Union[str, List[ChatMessage]], 
        model: Optional[str], 
        **kwargs: Any
    ) -> LLMResponse:
        """Handle prompt enhancement - now purely async"""
        user_prompt = self._prepare_messages(messages)
        
        target_model = kwargs.get('target_model', model)
        user_negative_prompt = kwargs.get('user_negative_prompt')
        wants_negative = kwargs.get('wants_negative', False)
        image_path = kwargs.get('image_path')
        
        try:
            # Just await - no asyncio.run() needed!
            if image_path:
                enhanced_text, used_model = await self.improve_prompt_multimodal(
                    user_prompt, image_path, target_model, 
                    user_negative_prompt, wants_negative
                )
            else:
                enhanced_text, used_model = await self.enhance_prompt_text_only(
                    user_prompt, target_model, 
                    user_negative_prompt, wants_negative
                )
            
            return LLMResponse(
                text=enhanced_text,
                model=used_model or "unknown",
                finish_reason="stop",
                metadata={
                    "provider": "Google Gemini",
                    "operation": "prompt_enhancement",
                    "original_prompt": user_prompt,
                    "target_model": target_model,
                    "multimodal": bool(image_path)
                }
            )
            
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {e}")
            return LLMResponse(
                text=user_prompt,
                model="fallback",
                finish_reason="error",
                metadata={"error": str(e), "fallback": True}
            )

    async def enhance_prompt_text_only(
        self, 
        user_prompt: str, 
        target_model: Optional[str] = None, 
        user_negative_prompt: Optional[str] = None, 
        wants_negative: bool = False
    ) -> Tuple[str, Optional[str]]:
        """Clean async text enhancement using asyncio.to_thread"""
        if not self.available_text_models:
            return user_prompt, None
        
        # Prepare instructions (unchanged)
        if wants_negative:
            if user_negative_prompt:
                negative_instruction = f"7. IMPORTANT: The user provided this negative prompt: '{user_negative_prompt}'. Please enhance and improve this negative prompt as well to better avoid unwanted elements."
                format_instruction = " and enhanced negative prompt"
            else:
                negative_instruction = "7. IMPORTANT: Generate an appropriate negative prompt to avoid common unwanted elements like blur, distortion, poor anatomy, etc."
                format_instruction = " and negative prompt"
        else:
            negative_instruction = "7. IMPORTANT: Do NOT generate any negative prompt. Focus only on enhancing the positive prompt."
            format_instruction = ""
        
        models_to_try = self.available_text_models.copy()
        random.shuffle(models_to_try)
        
        async with self._semaphore:
            for model_info in models_to_try:
                try:
                    formatted_system_prompt = SYSTEM_PROMPT.format(
                        model_name=target_model or "AI image generation",
                        negative_prompt_instruction=negative_instruction,
                        negative_format_instruction=format_instruction
                    )
                    
                    enhancement_instruction = f"{formatted_system_prompt}\n\nUser's prompt: '{user_prompt}'"
                    
                    # Use modern asyncio.to_thread - much cleaner than ThreadPoolExecutor!
                    response = await asyncio.to_thread(
                        model_info["model"].generate_content,
                        enhancement_instruction
                    )
                    
                    if response and response.text and response.text.strip():
                        return response.text.strip(), model_info["model_name"]
                        
                except Exception as e:
                    logger.warning(f"Model {model_info['name']} failed: {e}")
                    continue
        
        return user_prompt, None

    async def improve_prompt_multimodal(
        self, 
        original_prompt: str, 
        image_path: str, 
        target_model: Optional[str] = None, 
        user_negative_prompt: Optional[str] = None, 
        wants_negative: bool = False
    ) -> Tuple[str, Optional[str]]:
        """Clean async multimodal enhancement"""
        if not self.available_vision_models:
            return original_prompt, None
        
        # Prepare instructions (unchanged)
        if wants_negative:
            if user_negative_prompt:
                negative_instruction = f"6. **IMPORTANT**: The user provided this negative prompt: '{user_negative_prompt}'. Please enhance and improve this negative prompt as well to better avoid unwanted elements."
                format_instruction = " and enhanced negative prompt on a separate line starting with 'Negative prompt:'"
            else:
                negative_instruction = "6. **IMPORTANT**: Generate an appropriate negative prompt to avoid common unwanted elements. Always append this on a separate line starting with 'Negative prompt:'"
                format_instruction = " and negative prompt on a separate line starting with 'Negative prompt:'"
        else:
            negative_instruction = "6. **IMPORTANT**: Do NOT generate any negative prompt. Focus only on enhancing the positive prompt."
            format_instruction = ""
        
        try:
            # Load image using asyncio.to_thread
            image = await asyncio.to_thread(self._load_image, image_path)
            
            models_to_try = self.available_vision_models.copy()
            random.shuffle(models_to_try)
            
            async with self._semaphore:
                for model_info in models_to_try:
                    try:
                        formatted_system_prompt = IMAGE_SYSTEM_PROMPT.format(
                            model_name=target_model or "AI image generation",
                            negative_prompt_instruction=negative_instruction,
                            negative_format_instruction=format_instruction
                        )
                        
                        prompt_parts = [f"{formatted_system_prompt}\n\nUser's prompt: '{original_prompt}'", image]
                        
                        # Use asyncio.to_thread for API call
                        response = await asyncio.to_thread(
                            model_info["model"].generate_content,
                            prompt_parts
                        )
                        
                        if response and response.text and response.text.strip():
                            return response.text.strip(), model_info["model_name"]
                            
                    except Exception as e:
                        logger.warning(f"Model {model_info['name']} failed: {e}")
                        continue
            
            return original_prompt, None
                
        except Exception as e:
            logger.error(f"Error in multimodal enhancement: {e}")
            return original_prompt, None

    async def enhance_prompt(
        self, 
        user_prompt: str, 
        image_path: Optional[str] = None, 
        model_name: Optional[str] = None, 
        user_negative_prompt: Optional[str] = None, 
        wants_negative: bool = False
    ) -> Tuple[str, Optional[str]]:
        """Main async prompt enhancement method (unchanged interface)"""
        if image_path:
            return await self.improve_prompt_multimodal(user_prompt, image_path, model_name, user_negative_prompt, wants_negative)
        else:
            return await self.enhance_prompt_text_only(user_prompt, model_name, user_negative_prompt, wants_negative)

    # ============================================================================
    # Utility Methods (unchanged)
    # ============================================================================

    def get_available_models(self) -> List[str]:
        """Get list of available Gemini models"""
        all_models = set()
        for model_info in self.available_text_models + self.available_vision_models:
            all_models.add(model_info["model_name"])
        return sorted(list(all_models))

    def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific model"""
        if not self.available_text_models and not self.available_vision_models:
            return {"error": "No models available"}
        
        default_model = None
        if self.available_text_models:
            default_model = self.available_text_models[0]["model_name"]
        elif self.available_vision_models:
            default_model = self.available_vision_models[0]["model_name"]
            
        target_model = model or default_model
        
        supports_vision = any(target_model == m["model_name"] for m in self.available_vision_models)
        supports_text = any(target_model == m["model_name"] for m in self.available_text_models)
        
        return {
            "name": target_model,
            "provider": "Google Gemini",
            "type": "generative_ai",
            "supports_vision": supports_vision,
            "supports_text": supports_text,
            "context_window": 1000000 if "1.5" in target_model else 32000,
            "specialized_features": ["prompt_enhancement", "multimodal_analysis"]
        }

    def get_model_status(self) -> Dict[str, Any]:
        """Get status information about available models"""
        return {
            "vision_models_available": len(self.available_vision_models),
            "text_models_available": len(self.available_text_models),
            "vision_models": [model["model_name"] for model in self.available_vision_models],
            "text_models": [model["model_name"] for model in self.available_text_models]
        }

# ============================================================================
# Simple Global Instance Management
# ============================================================================

_gemini_client = None

def get_gemini_client() -> GeminiLLM:
    """Get or create async Gemini client instance"""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiLLM()
    return _gemini_client

def get_model_status() -> Dict[str, Any]:
    """Get status of available Gemini models"""
    client = get_gemini_client()
    return client.get_model_status()
