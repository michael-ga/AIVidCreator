"""
Gemini LLM implementation with your original prompt enhancement features.
"""

import google.generativeai as genai
import logging
import asyncio
import random
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple
from PIL import Image

from ...models import ChatMessage, LLMResponse, LLMStreamChunk
from ...exceptions import AIProviderError, AuthenticationError
from ..protocols import LLMInterface
from config.settings import GOOGLE_API_KEY, GEMINI_TEXT_MODELS_TO_TRY, GEMINI_VISION_MODELS_TO_TRY

logger = logging.getLogger(__name__)

# Your original system prompts
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
   - Note stylistic traits (e.g., "grainy texture," "soft backlight", "motion blur"), layout balance, and visual storytelling cues.

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
    """Gemini implementation of LLMInterface with specialized prompt enhancement"""
    
    def __init__(self):
        """Initialize Gemini client with API key from config"""
        api_key = GOOGLE_API_KEY
        if not api_key:
            raise AuthenticationError("GOOGLE_API_KEY not found in environment variables", "google")

        genai.configure(api_key=api_key)

        # Initialize available models
        self.available_vision_models = self._initialize_models(GEMINI_VISION_MODELS_TO_TRY, "vision")
        self.available_text_models = self._initialize_models(GEMINI_TEXT_MODELS_TO_TRY, "text")
        
        if not self.available_text_models and not self.available_vision_models:
            raise RuntimeError("No Gemini models could be initialized")

    def _initialize_models(self, model_names: List[str], model_type: str) -> List[Dict[str, Any]]:
        """Initialize and validate Gemini models"""
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
        
        if not models_to_try:
            logger.error(f"No {model_type} models could be initialized.")
        else:
            logger.info(f"Initialized {len(models_to_try)} {model_type} models.")
            
        return models_to_try

    def _prepare_messages(self, messages: Union[str, List[ChatMessage]]) -> str:
        """Convert messages to Gemini format"""
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
        """Select appropriate model for generation"""
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

    def generate(
        self,
        messages: Union[str, List[ChatMessage]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text response using Gemini models"""
        
        # Check if this is a prompt enhancement request
        if any(key in kwargs for key in ['target_model', 'user_negative_prompt', 'wants_negative']):
            return self._generate_enhanced_prompt(messages, model, **kwargs)
        
        # Standard text generation
        prompt = self._prepare_messages(messages)
        needs_vision = 'image_path' in kwargs or 'image' in kwargs
        
        selected_model = self._select_model_for_generation(model, needs_vision)
        
        try:
            generation_config = {
                "temperature": temperature,
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens

            if needs_vision and 'image_path' in kwargs:
                image = Image.open(kwargs['image_path'])
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                content = [prompt, image]
            else:
                content = prompt

            response = selected_model["model"].generate_content(
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
            raise AIProviderError(f"Gemini generation error: {str(e)}", "google") from e

    def stream(
        self,
        messages: Union[str, List[ChatMessage]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[LLMStreamChunk]:
        """Stream text response from Gemini"""
        try:
            # Generate complete response first (Gemini streaming can be complex)
            response = self.generate(messages, model, temperature, max_tokens, **kwargs)
            
            # Simulate streaming by breaking response into chunks
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
        
        supports_vision = any(
            target_model == m["model_name"] for m in self.available_vision_models
        )
        supports_text = any(
            target_model == m["model_name"] for m in self.available_text_models
        )
        
        return {
            "name": target_model,
            "provider": "Google Gemini",
            "type": "generative_ai",
            "supports_vision": supports_vision,
            "supports_text": supports_text,
            "context_window": 1000000 if "1.5" in target_model else 32000,
            "specialized_features": ["prompt_enhancement", "multimodal_analysis"]
        }

    # ============================================================================
    # Your Original Prompt Enhancement Methods
    # ============================================================================

    def _generate_enhanced_prompt(
        self, 
        messages: Union[str, List[ChatMessage]], 
        model: Optional[str], 
        **kwargs: Any
    ) -> LLMResponse:
        """Handle prompt enhancement via the generate interface"""
        user_prompt = self._prepare_messages(messages)
        
        target_model = kwargs.get('target_model', model)
        user_negative_prompt = kwargs.get('user_negative_prompt')
        wants_negative = kwargs.get('wants_negative', False)
        image_path = kwargs.get('image_path')
        
        try:
            if image_path:
                enhanced_text, used_model = asyncio.run(
                    self.improve_prompt_multimodal(
                        user_prompt, image_path, target_model, 
                        user_negative_prompt, wants_negative
                    )
                )
            else:
                enhanced_text, used_model = asyncio.run(
                    self.enhance_prompt_text_only(
                        user_prompt, target_model, 
                        user_negative_prompt, wants_negative
                    )
                )
            
            return LLMResponse(
                text=enhanced_text,
                model=used_model,
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
        """Enhance a text prompt without image analysis"""
        if not self.available_text_models:
            logger.error("No text models available. Returning original prompt.")
            return user_prompt, None
        
        # Prepare negative prompt instructions
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
        failed_models = set()
        
        for model_info in models_to_try:
            model_name = model_info["name"]
            model_id = model_info["model_name"]
            
            if model_id in failed_models:
                continue
                
            model = model_info["model"]
            
            formatted_system_prompt = SYSTEM_PROMPT.format(
                model_name=target_model or "AI image generation",
                negative_prompt_instruction=negative_instruction,
                negative_format_instruction=format_instruction
            )
            
            enhancement_instruction = f"""
            {formatted_system_prompt}
            
            User's prompt: '{user_prompt}'
            """
            
            logger.info(f"Attempting to enhance text prompt with {model_name} for target model: {target_model}...")
     
            try:
                response = await asyncio.to_thread(
                    model.generate_content,
                    enhancement_instruction
                )
                
                if response and response.text and response.text.strip():
                    enhanced = response.text.strip()
                    logger.info(f"Successfully enhanced text prompt with {model_name}.")
                    return enhanced, model_info["model_name"]
                else:
                    logger.warning(f"{model_name} returned an empty response. Trying next model.")
                    failed_models.add(model_id)
                    
            except Exception as e:
                logger.error(f"Failed to get response from {model_name}. Error: {e}. Trying next model...")
                failed_models.add(model_id)
                continue
        
        logger.critical("All text models failed. Returning the original prompt.")
        return user_prompt, None

    async def improve_prompt_multimodal(
        self, 
        original_prompt: str, 
        image_path: str, 
        target_model: Optional[str] = None, 
        user_negative_prompt: Optional[str] = None, 
        wants_negative: bool = False
    ) -> Tuple[str, Optional[str]]:
        """Improve a prompt using Gemini Vision models"""
        if not self.available_vision_models:
            logger.error("No vision models available. Returning original prompt.")
            return original_prompt, None
        
        # Prepare negative prompt instructions
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
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            models_to_try = self.available_vision_models.copy()
            random.shuffle(models_to_try)
            failed_models = set()
            
            for model_info in models_to_try:
                model_name = model_info["name"]
                model_id = model_info["model_name"]
                
                if model_id in failed_models:
                    continue
                    
                model = model_info["model"]
                
                formatted_system_prompt = IMAGE_SYSTEM_PROMPT.format(
                    model_name=target_model or "AI image generation",
                    negative_prompt_instruction=negative_instruction,
                    negative_format_instruction=format_instruction
                )
                
                prompt_parts = [
                    f"{formatted_system_prompt}\n\nUser's prompt: '{original_prompt}'",
                    image
                ]
                
                logger.info(f"Attempting to improve prompt with {model_name} for target model: {target_model}...")
                try:
                    response = await asyncio.to_thread(
                        model.generate_content,
                        prompt_parts
                    )
                    
                    if response and response.text and response.text.strip():
                        improved_prompt = response.text.strip()
                        logger.info(f"Successfully improved prompt with {model_name}.")
                        return improved_prompt, model_info["model_name"]
                    else:
                        logger.warning(f"{model_name} returned an empty response. Trying next model.")
                        failed_models.add(model_id)
                        
                except Exception as e:
                    logger.error(f"Failed to get response from {model_name}. Error: {e}. Trying next model...")
                    failed_models.add(model_id)
                    continue
            
            logger.critical("All vision models failed. Returning the original prompt.")
            return original_prompt, None
                
        except Exception as e:
            logger.error(f"Error preparing image for multimodal enhancement: {e}")
            return original_prompt, None

    async def enhance_prompt(
        self, 
        user_prompt: str, 
        image_path: Optional[str] = None, 
        model_name: Optional[str] = None, 
        user_negative_prompt: Optional[str] = None, 
        wants_negative: bool = False
    ) -> Tuple[str, Optional[str]]:
        """Main method to enhance prompts - your original method"""
        if image_path:
            return await self.improve_prompt_multimodal(user_prompt, image_path, model_name, user_negative_prompt, wants_negative)
        else:
            return await self.enhance_prompt_text_only(user_prompt, model_name, user_negative_prompt, wants_negative)

    def get_model_status(self) -> Dict[str, Any]:
        """Get status information about available models - your original method"""
        return {
            "vision_models_available": len(self.available_vision_models),
            "text_models_available": len(self.available_text_models),
            "vision_models": [model["model_name"] for model in self.available_vision_models],
            "text_models": [model["model_name"] for model in self.available_text_models]
        }

# Global instance management (your original pattern)
gemini_client = None

def get_gemini_client() -> GeminiLLM:
    """Get or create Gemini client instance"""
    global gemini_client
    if gemini_client is None:
        gemini_client = GeminiLLM()
    return gemini_client

def get_model_status() -> Dict[str, Any]:
    """Get status of available Gemini models"""
    client = get_gemini_client()
    return client.get_model_status()
