"""
OpenAI LLM implementation.
"""

from typing import Union, List, Iterator, Optional, Any, Dict
import logging
from openai import OpenAI

from ...models import ChatMessage, LLMResponse, LLMStreamChunk
from ...exceptions import AIProviderError, RateLimitError, ModelNotFoundError, AuthenticationError
from ..protocols import LLMInterface
from config.settings import OPENAI_API_KEY, DEFAULT_OPENAI_CHAT_MODEL

logger = logging.getLogger(__name__)

class OpenAILLM:
    """OpenAI ChatGPT implementation of LLMInterface"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_OPENAI_CHAT_MODEL,
        base_url: Optional[str] = None
    ):
        try:
            self.client = OpenAI(api_key=api_key or OPENAI_API_KEY, base_url=base_url)
            self.default_model = default_model
            self.provider_name = "OpenAI"
        except Exception as e:
            raise AuthenticationError(f"Failed to initialize OpenAI client: {str(e)}", "openai")
    
    def _prepare_messages(self, messages: Union[str, List[ChatMessage]]) -> List[Dict[str, str]]:
        """Convert messages to OpenAI format"""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    def _handle_openai_error(self, error: Exception) -> None:
        """Convert OpenAI errors to our standard exceptions"""
        error_str = str(error).lower()
        
        if "rate limit" in error_str:
            raise RateLimitError(f"OpenAI rate limit exceeded: {str(error)}", "openai")
        elif "model" in error_str and "not found" in error_str:
            raise ModelNotFoundError(f"OpenAI model not found: {str(error)}", "openai")
        elif "authentication" in error_str or "api key" in error_str:
            raise AuthenticationError(f"OpenAI authentication failed: {str(error)}", "openai")
        else:
            raise AIProviderError(f"OpenAI API error: {str(error)}", "openai")
    
    def generate(
        self,
        messages: Union[str, List[ChatMessage]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        model = model or self.default_model
        formatted_messages = self._prepare_messages(messages)
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            choice = response.choices[0]
            usage_dict = response.usage.model_dump() if response.usage else {}
            
            return LLMResponse(
                text=choice.message.content or "",
                finish_reason=choice.finish_reason,
                usage=usage_dict,
                model=model,
                metadata={"response_id": response.id, "provider": self.provider_name},
                raw_response=response
            )
        except Exception as e:
            self._handle_openai_error(e)
    
    def stream(
        self,
        messages: Union[str, List[ChatMessage]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[LLMStreamChunk]:
        model = model or self.default_model
        formatted_messages = self._prepare_messages(messages)
        
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta_content = choice.delta.content or ""
                    
                    yield LLMStreamChunk(
                        delta=delta_content,
                        finish_reason=choice.finish_reason,
                        metadata={"chunk_id": chunk.id, "provider": self.provider_name},
                        raw_chunk=chunk
                    )
        except Exception as e:
            self._handle_openai_error(e)
    
    def get_available_models(self) -> List[str]:
        """Get available OpenAI models"""
        return ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    
    def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        model = model or self.default_model
        
        model_info = {
            "gpt-4o": {"context_window": 128000, "type": "chat", "multimodal": True},
            "gpt-4o-mini": {"context_window": 128000, "type": "chat", "multimodal": True},
            "gpt-4-turbo": {"context_window": 128000, "type": "chat", "multimodal": True},
            "gpt-3.5-turbo": {"context_window": 16385, "type": "chat", "multimodal": False}
        }
        
        info = model_info.get(model, {"context_window": 4096, "type": "chat", "multimodal": False})
        info.update({
            "name": model,
            "provider": self.provider_name,
        })
        
        return info
