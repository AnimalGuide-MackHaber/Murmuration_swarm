import json
import os
from typing import Dict, Any

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "maai_settings.json")

class Config:
    def __init__(self, path: str = SETTINGS_PATH):
        self.path = path
        self.settings: Dict[str, Any] = self._load_settings()

    def _load_settings(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding settings JSON: {e}")
                return {}
        return {}

    @property
    def llm_provider(self) -> str:
        return self.settings.get("llm_provider", "gemini")

    @property
    def model_name(self) -> str:
        return self.settings.get("model_name", "gemini-2.5-flash")

    @property
    def embedding_model_name(self) -> str:
        return self.settings.get("embedding_model_name", "text-embedding-004")

    def update_setting(self, key: str, value: Any):
        self.settings[key] = value
        with open(self.path, "w") as f:
            json.dump(self.settings, f, indent=4)

    @property
    def google_api_key(self) -> str:
        # maai_settings.json overrides environment variable
        json_key = self.settings.get("google_api_key", "")
        if json_key:
            return json_key
        return os.environ.get("GEMINI_API_KEY", "")

    @property
    def visitation_threshold(self) -> int:
        return self.settings.get("visitation_threshold", 50)

    @property
    def producers_paused(self) -> bool:
        return self.settings.get("producers_paused", False)

    def set_visitation_threshold(self, value: int):
        self.update_setting("visitation_threshold", value)

    def set_producers_paused(self, value: bool):
        self.update_setting("producers_paused", value)

    def get_rate_limits(self, provider: str = "gemini") -> Dict[str, Any]:
        limits = self.settings.get("rate_limits", {})
        provider_limits = limits.get(provider, {})
        return {
            "requests_per_minute": provider_limits.get("requests_per_minute", 15),
            "tokens_per_minute": provider_limits.get("tokens_per_minute", 1000000),
            "request_delay_seconds": provider_limits.get("request_delay_seconds", 0.05)
        }

config = Config()
