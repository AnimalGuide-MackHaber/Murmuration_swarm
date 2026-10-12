import os
import asyncio
import logging
from typing import Union, List, Optional
from core.llm_retry import async_retry
from core.settings_manager import SettingsManager

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class GeminiResponse:
    """
    Duck-typed response object to match OllamaResponse/NvidiaResponse for compatibility.
    """
    def __init__(self, text: str, usage_metadata: dict = None):
        self.text = text
        self.usage_metadata = usage_metadata

class GeminiLLM:
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None, rate_limiter=None):
        self.model_name = model_name
        self.api_key = api_key
        self.client = None
        self._loop = None
        self.rate_limiter = rate_limiter
        self.logger = logging.getLogger("GeminiLLM")

    def configure(self, api_key: str = None):
        """
        Configures the Google GenAI client with the API key.
        """
        if not HAS_GENAI:
            raise ImportError(
                "google-genai package is not installed. "
                "Install it with: pip install google-genai"
            )

        if api_key:
            self.api_key = api_key
        
        if not self.api_key:
            self.api_key = SettingsManager().get_api_key()

        if not self.api_key:
            self.logger.warning("Google API Key not found. Gemini LLM calls will likely fail.")

        self.client = genai.Client(api_key=self.api_key)
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = None
        self.logger.info(f"GeminiLLM configured for model: {self.model_name}")

    @staticmethod
    def list_models(api_key: Optional[str] = None) -> List[str]:
        """
        Lists available Gemini models using the google-genai SDK.
        """
        # Try to use streamlit caching if available
        try:
            import streamlit as st
            @st.cache_data(ttl=3600) # Cache for 1 hour
            def _cached_list_models(api_key):
                return GeminiLLM._fetch_models(api_key)
            return _cached_list_models(api_key)
        except ImportError:
            return GeminiLLM._fetch_models(api_key)

    @staticmethod
    def _fetch_models(api_key: Optional[str] = None) -> List[str]:
        """
        Internal method to fetch models from the API.
        """
        if not HAS_GENAI:
            return ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]

        if not api_key:
            api_key = SettingsManager().get_api_key()

        if not api_key:
            return ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]

        try:
            client = genai.Client(api_key=api_key)
            models = client.models.list()
            # Filter for models that support generateContent
            # The model objects in google-genai have 'name' and 'supported_generation_methods'
            # Note: client.models.list() returns an iterator of Model objects
            model_names = []
            for m in models:
                if 'generateContent' in m.supported_actions:
                    # Model name is usually like 'models/gemini-1.5-flash'
                    # We might want to strip 'models/' prefix for UI
                    name = m.name
                    if name.startswith("models/"):
                        name = name[7:]
                    model_names.append(name)
            
            if not model_names:
                return ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
                
            return sorted(model_names)
        except Exception as e:
            logging.getLogger("GeminiLLM").error(f"Error listing Gemini models: {e}")
            return ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]

    @async_retry
    async def generate_content_async(self, prompt: Union[str, List[str]]) -> GeminiResponse:
        """
        Calls the Gemini API asynchronously using the google.genai SDK.
        """
        try:
            curr_loop = asyncio.get_event_loop()
        except RuntimeError:
            curr_loop = None

        if not self.client or (self._loop is not None and self._loop != curr_loop):
            self.configure()

        # Handle list prompts by joining
        if isinstance(prompt, list):
            final_prompt = "\n".join(prompt)
        else:
            final_prompt = prompt

        # Rate Limiting
        estimated_tokens = len(final_prompt) // 4
        if self.rate_limiter:
            await self.rate_limiter.acquire(estimated_tokens)

        try:
            self.logger.debug(f"Sending request to Gemini ({self.model_name})...")

            # google.genai uses aio for async operations
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=final_prompt
            )

            response_text = response.text or ""

            # Extract usage stats if available
            usage_dict = None
            actual_tokens = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                actual_tokens = getattr(response.usage_metadata, 'total_token_count', 0)
                usage_dict = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": actual_tokens
                }

            if self.rate_limiter:
                # If we have actual tokens, use them to update the limiter
                if actual_tokens > 0:
                    await self.rate_limiter.update_usage(actual_tokens)
                else:
                    # Fallback to estimate if usage metadata missing
                    await self.rate_limiter.update_usage(estimated_tokens + (len(response_text) // 4))

            if not response_text.strip():
                raise RuntimeError("Empty response from Gemini")

            return GeminiResponse(text=response_text, usage_metadata=usage_dict)

        except Exception as e:
            self.logger.debug(f"Gemini API Error: {e}")
            raise e
