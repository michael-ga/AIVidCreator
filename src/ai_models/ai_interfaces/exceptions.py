"""
Custom exceptions for AI interfaces.
"""

class AIProviderError(Exception):
    """Base exception for AI provider errors"""
    
    def __init__(self, message: str, provider: str = "unknown", error_code: str = None):
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code

class RateLimitError(AIProviderError):
    """Raised when rate limits are exceeded"""
    
    def __init__(self, message: str, provider: str = "unknown", retry_after: int = None):
        super().__init__(message, provider, "rate_limit")
        self.retry_after = retry_after

class ModelNotFoundError(AIProviderError):
    """Raised when requested model is not available"""
    
    def __init__(self, message: str, provider: str = "unknown", model: str = None):
        super().__init__(message, provider, "model_not_found")
        self.model = model

class InvalidRequestError(AIProviderError):
    """Raised when request parameters are invalid"""
    
    def __init__(self, message: str, provider: str = "unknown", parameter: str = None):
        super().__init__(message, provider, "invalid_request")
        self.parameter = parameter

class AuthenticationError(AIProviderError):
    """Raised when authentication fails"""
    
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider, "authentication_error")
