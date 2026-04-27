import os
import logging
from typing import Union, List, Optional

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
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.client = None
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
            self.api_key = os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            self.logger.warning("Google API Key not found. Gemini LLM calls will likely fail.")

        self.client = genai.Client(api_key=self.api_key)
        self.logger.info(f"GeminiLLM configured for model: {self.model_name}")

    async def generate_content_async(self, prompt: Union[str, List[str]]) -> GeminiResponse:
        """
        Calls the Gemini API asynchronously using the google.genai SDK.
        """
        if not self.client:
            self.configure()

        # Handle list prompts by joining
        if isinstance(prompt, list):
            final_prompt = "\n".join(prompt)
        else:
            final_prompt = prompt

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
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage_dict = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                }

            return GeminiResponse(text=response_text, usage_metadata=usage_dict)

        except Exception as e:
            self.logger.error(f"Gemini API Error: {e}")
            raise e
