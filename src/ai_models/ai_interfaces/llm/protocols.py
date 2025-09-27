"""
Protocol definitions for LLM interfaces.
"""

from typing import Protocol, Union, List, Optional, Any, Iterator, Dict
from ..models import ChatMessage, LLMResponse, LLMStreamChunk

class LLMInterface(Protocol):
    """Protocol defining the interface for Large Language Models"""
    
    def generate(
        self,
        messages: Union[str, List[ChatMessage]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text response from messages"""
        ...

    def stream(
        self,
        messages: Union[str, List[ChatMessage]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[LLMStreamChunk]:
        """Stream text response from messages"""
        ...
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        ...
    
    def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific model"""
        ...

class AsyncLLMInterface(Protocol):
    """Async version of LLM interface"""
    
    async def generate(
        self,
        messages: Union[str, List[ChatMessage]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async generate method"""
        ...
    
    async def stream(
        self,
        messages: Union[str, List[ChatMessage]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[LLMStreamChunk]:
        """Async stream method"""
        ...
