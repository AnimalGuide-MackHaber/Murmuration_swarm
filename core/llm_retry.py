import asyncio
import logging
import functools
import time
from core.settings_manager import SettingsManager

logger = logging.getLogger("LLMRetry")

def async_retry(func):
    """
    Decorator for retrying asynchronous LLM calls with exponential backoff.
    Reads configuration from SettingsManager.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        settings_manager = SettingsManager()
        settings = settings_manager.get_settings()
        
        retry_config = settings.llm_retry
        max_retries = retry_config.max_retries if retry_config else 10
        base_delay = retry_config.base_delay if retry_config else 5.0
        use_exponential = retry_config.exponential_backoff if retry_config else True
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Determine if we should retry based on the error type/message
                # We want to retry on transient errors (503, 429, connection issues)
                # But maybe not on auth errors (401) or bad requests (400)
                
                should_retry = False
                
                # Common transient error signals
                transient_signals = [
                    "503", "unavailable", "high demand", "overloaded",
                    "429", "rate limit", "too many requests",
                    "connection", "timeout", "network", "socket",
                    "deadline exceeded", "internal server error", "500"
                ]
                
                if any(signal in error_msg for signal in transient_signals):
                    should_retry = True
                
                # If we've exhausted retries or it's not a retryable error, raise
                if not should_retry or attempt >= max_retries:
                    logger.error(f"LLM Call failed after {attempt} retries: {e}")
                    raise last_exception
                
                # Calculate delay
                if use_exponential:
                    delay = base_delay * (2 ** attempt)
                else:
                    delay = base_delay
                
                logger.debug(f"LLM Call failed (Attempt {attempt+1}/{max_retries}). Retrying in {delay:.2f}s... Error: {e}")
                await asyncio.sleep(delay)
                
        return await func(*args, **kwargs) # Should not be reached normally
        
    return wrapper
