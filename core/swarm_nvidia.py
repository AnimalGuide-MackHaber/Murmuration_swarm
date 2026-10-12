import os
import asyncio
import logging
from openai import AsyncOpenAI
from typing import List, Union, Optional
from core.llm_retry import async_retry

class NvidiaResponse:
    """
    Duck-typed response object to mimic Gemini/Ollama response for compatibility.
    """
    def __init__(self, text: str, usage_metadata: dict = None):
        self.text = text
        self.usage_metadata = usage_metadata

class NvidiaLLM:
    def __init__(self, model_name: str, base_url: str = "https://integrate.api.nvidia.com/v1", api_key: Optional[str] = None, rate_limiter=None):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.client = None
        self._loop = None
        self.rate_limiter = rate_limiter
        self.logger = logging.getLogger("NvidiaLLM")

    def configure(self, api_key: str = None):
        """
        Configures the AsyncOpenAI client. can be called with an API key, 
        or will use the one provided in init/env.
        """
        if api_key:
            self.api_key = api_key
            
        if not self.api_key:
             self.api_key = os.getenv("NVIDIA_API_KEY")

        if not self.api_key:
            self.logger.warning("Nvidia API Key not found. LLM calls will likely fail.")
            
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = None
        self.logger.info(f"NvidiaLLM configured for model: {self.model_name}")

    @async_retry
    async def generate_content_async(self, prompt: Union[str, List[str]]) -> NvidiaResponse:
        """
        Calls the Nvidia/OpenAI API asynchronously.
        """
        try:
            curr_loop = asyncio.get_event_loop()
        except RuntimeError:
            curr_loop = None

        if not self.client or (self._loop is not None and self._loop != curr_loop):
            self.configure()

        # Handle list prompts (history) by joining them or formatting as messages
        messages = []
        if isinstance(prompt, list):
             final_prompt = "\n".join(prompt)
             messages.append({"role": "user", "content": final_prompt})
        else:
             final_prompt = prompt
             messages.append({"role": "user", "content": final_prompt})

        # Rate Limiting
        estimated_tokens = len(final_prompt) // 4
        if self.rate_limiter:
            await self.rate_limiter.acquire(estimated_tokens)

        try:
            self.logger.debug(f"Sending request to Nvidia ({self.model_name})...")
            
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                stream=False
            )

            response_text = completion.choices[0].message.content
            
            # Extract usage stats
            usage_data = completion.usage
            actual_tokens = usage_data.total_tokens
            usage_dict = {
                "prompt_tokens": usage_data.prompt_tokens,
                "completion_tokens": usage_data.completion_tokens,
                "total_tokens": actual_tokens
            }

            if self.rate_limiter:
                await self.rate_limiter.update_usage(actual_tokens)

            if not response_text.strip():
                raise RuntimeError("Empty response from Nvidia/OpenAI")

            return NvidiaResponse(text=response_text, usage_metadata=usage_dict)

        except Exception as e:
            self.logger.error(f"Nvidia API Error: {e}")
            raise e
