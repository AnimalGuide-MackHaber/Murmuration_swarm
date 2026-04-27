import os
import logging
from openai import AsyncOpenAI
from typing import List, Union, Optional

class NvidiaResponse:
    """
    Duck-typed response object to mimic Gemini/Ollama response for compatibility.
    """
    def __init__(self, text: str, usage_metadata: dict = None):
        self.text = text
        self.usage_metadata = usage_metadata

class NvidiaLLM:
    def __init__(self, model_name: str, base_url: str = "https://integrate.api.nvidia.com/v1", api_key: Optional[str] = None):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.client = None
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
        self.logger.info(f"NvidiaLLM configured for model: {self.model_name}")

    async def generate_content_async(self, prompt: Union[str, List[str]]) -> NvidiaResponse:
        """
        Calls the Nvidia/OpenAI API asynchronously.
        """
        if not self.client:
            self.configure()

        # Handle list prompts (history) by joining them or formatting as messages
        messages = []
        if isinstance(prompt, list):
             # Simple joining for now, or could map to chat roles if the prompt structure allows
             # For this system, the prompt is usually a single constructed string or a list of history.
             # SwarmOrchestrator seems to pass a single string usually.
             final_prompt = "\n".join(prompt)
             messages.append({"role": "user", "content": final_prompt})
        else:
             messages.append({"role": "user", "content": prompt})

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
            usage_dict = {
                "prompt_tokens": usage_data.prompt_tokens,
                "completion_tokens": usage_data.completion_tokens,
                "total_tokens": usage_data.total_tokens
            }

            return NvidiaResponse(text=response_text, usage_metadata=usage_dict)

        except Exception as e:
            self.logger.error(f"Nvidia API Error: {e}")
            raise e
