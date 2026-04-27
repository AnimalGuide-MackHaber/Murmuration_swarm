import aiohttp
import asyncio
import json
import logging
from settings_manager import SettingsManager

class OllamaResponse:
    """
    Duck-typed response object to mimic GeminiResponse for compatibility.
    """
    def __init__(self, text: str, usage_metadata: dict = None, thinking: str = None):
        self.text = text
        self.usage_metadata = usage_metadata
        self.thinking = thinking

class OllamaWrapper:
    def __init__(self, model_name: str = "deepseek-r1:1.5b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.logger = logging.getLogger("OllamaWrapper")
        
    def configure(self, api_key: str = None):
        # Ollama typically doesn't need an API key for local instances, 
        # but method is kept for interface compatibility.
        pass

    async def generate_content_async(self, prompt: str) -> OllamaResponse:
        """
        Calls the local Ollama API asynchronously.
        """
        url = f"{self.base_url}/api/generate"
        
        # Handle list prompts (history) by joining them if necessary, 
        # though Ollama 'generate' endpoint expects a single prompt string usually.
        # For chat-like history, /api/chat is better, but this wrapper replaces a simple generate interface.
        final_prompt = prompt
        if isinstance(prompt, list):
            final_prompt = "\n".join(prompt)

        payload = {
            "model": self.model_name,
            "prompt": final_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7 
            }
        }
        
        try:
            settings = SettingsManager().get_settings()
            if settings.enable_thinking:
                payload["think"] = True
        except Exception as e:
            self.logger.error(f"Error fetching settings for think mode: {e}")

        try:
            async with aiohttp.ClientSession() as session:
                self.logger.debug(f"Sending request to Ollama ({self.model_name})...")
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"Ollama API Error ({response.status}): {error_text}")
                    
                    data = await response.json()
                    
                    response_text = data.get("response", "")
                    thinking_text = data.get("thinking", None)
                    
                    # Extract usage stats if available
                    # Ollama returns: eval_count, eval_duration, load_duration, prompt_eval_count, etc.
                    usage = {
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                        "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                    }
                    
                    return OllamaResponse(text=response_text, usage_metadata=usage, thinking=thinking_text)

        except aiohttp.ClientError as e:
            self.logger.error(f"Ollama Connection Error: {e}")
            raise RuntimeError(f"Failed to connect to Ollama at {self.base_url}. Is it running?")
        except Exception as e:
            self.logger.error(f"Ollama Error: {e}")
            raise e
