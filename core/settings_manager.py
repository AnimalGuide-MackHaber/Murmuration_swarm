import json
import os
from pydantic import BaseModel, Field
from typing import Optional, Dict
from core.rate_limiter import RateLimiter

# Define the settings structure using Pydantic for validation
class PhysicsWeights(BaseModel):
    separation: float
    alignment: float
    cohesion: float
    goal_seeking: float

class PhysicsSettings(BaseModel):
    visual_range: float
    max_speed: float
    influence_decay_alpha: float
    weights: PhysicsWeights
    dt: float

class SimulationSettings(BaseModel):
    default_max_iterations: int
    parallel_agent_execution: bool = False
    learning_refit_interval: int = 25

class RateLimitingSettings(BaseModel):
    requests_per_minute: int = 60
    tokens_per_minute: int = 32000
    request_delay_seconds: float = 2.0

class PerModelRateLimiting(BaseModel):
    ollama: RateLimitingSettings = Field(default_factory=lambda: RateLimitingSettings(
        requests_per_minute=20000, tokens_per_minute=90000000, request_delay_seconds=0.05))
    nvidia: RateLimitingSettings = Field(default_factory=lambda: RateLimitingSettings(
        requests_per_minute=5000, tokens_per_minute=50000000, request_delay_seconds=0.1))
    gemini: RateLimitingSettings = Field(default_factory=lambda: RateLimitingSettings(
        requests_per_minute=10, tokens_per_minute=250000, request_delay_seconds=6.0))
    doab: RateLimitingSettings = Field(default_factory=lambda: RateLimitingSettings(
        requests_per_minute=30, tokens_per_minute=0, request_delay_seconds=2.0))

class NvidiaSettings(BaseModel):
    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key: Optional[str] = None

class RetrySettings(BaseModel):
    max_retries: int = 10
    base_delay: float = 5.0
    exponential_backoff: bool = True

class CircuitBreakerSettings(BaseModel):
    max_failures: int = 5
    open_duration_seconds: int = 600

class KnowledgeGraphSettings(BaseModel):
    max_retrieval_depth: int = 1
    max_children_per_node: int = 3
    chunk_size: int = 15000
    search_long_term_memory: bool = True

class GlobalSettings(BaseModel):
    physics: PhysicsSettings
    simulation: SimulationSettings
    circuit_breaker: Optional[CircuitBreakerSettings] = Field(default_factory=CircuitBreakerSettings)
    knowledge_graph: Optional[KnowledgeGraphSettings] = Field(default_factory=KnowledgeGraphSettings)
    rate_limiting: Optional[PerModelRateLimiting] = Field(default_factory=PerModelRateLimiting)
    llm_provider: str = Field(default="ollama", description="LLM Provider: 'ollama' or 'nvidia' or 'gemini'")
    enable_thinking: bool = Field(default=False, description="Enable thinking for supported LLM providers (e.g., Ollama)")
    google_api_key: Optional[str] = Field(default=None, description="Gemini API Key")
    model_name: str = Field(default="deepseek-r1:1.5b", description="Model Name")
    nvidia_api: Optional[NvidiaSettings] = Field(default_factory=NvidiaSettings)
    llm_retry: Optional[RetrySettings] = Field(default_factory=RetrySettings)

class SettingsManager:
    _instance = None
    _settings: GlobalSettings = None
    _rate_limiters: Dict[str, RateLimiter] = {}
    
    DEFAULT_SETTINGS_PATH = "swarm_settings.json"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SettingsManager, cls).__new__(cls)
            cls._instance.load_settings()
        return cls._instance

    def load_settings(self, path: str = DEFAULT_SETTINGS_PATH):
        """Loads settings from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Settings file not found at {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
            
        self._settings = GlobalSettings(**data)

    def get_settings(self) -> GlobalSettings:
        """Returns the current global settings."""
        if self._settings is None:
             self.load_settings()
        return self._settings

    def get_api_key(self) -> Optional[str]:
        """
        Retrieves the Google API Key.
        Priority:
        1. Environment Variable 'GOOGLE_API_KEY'
        2. Settings file 'google_api_key'
        """
        settings = self.get_settings()
        if settings.google_api_key:
            return settings.google_api_key

        env_key = os.getenv("GOOGLE_API_KEY")
        if env_key:
            return env_key
            
        return None

    def get_nvidia_api_key(self) -> Optional[str]:
        """
        Retrieves the Nvidia API Key.
        Priority:
        1. Environment Variable 'NVIDIA_API_KEY'
        2. Settings file 'nvidia_api.api_key'
        """
        settings = self.get_settings()
        if settings.nvidia_api and settings.nvidia_api.api_key:
            return settings.nvidia_api.api_key

        env_key = os.getenv("NVIDIA_API_KEY")
        if env_key:
            return env_key
            
        return None

    def get_rate_limiting_for_provider(self, provider: str) -> RateLimitingSettings:
        """
        Returns the rate limiting settings for the given provider.
        Falls back to sensible defaults if the provider key is missing.
        """
        settings = self.get_settings()
        rl = settings.rate_limiting
        if rl is None:
            return RateLimitingSettings()
        
        provider_lower = provider.lower()
        if hasattr(rl, provider_lower):
            return getattr(rl, provider_lower)
        
        # Fallback
        return RateLimitingSettings()

    def get_shared_rate_limiter(self, provider: str) -> RateLimiter:
        """
        Returns a shared RateLimiter instance for the given provider.
        Creates it if it doesn't exist yet.
        """
        provider_lower = provider.lower()
        if provider_lower not in self._rate_limiters:
            rl_settings = self.get_rate_limiting_for_provider(provider_lower)
            self._rate_limiters[provider_lower] = RateLimiter(
                requests_per_minute=rl_settings.requests_per_minute,
                tokens_per_minute=rl_settings.tokens_per_minute
            )
        return self._rate_limiters[provider_lower]

    def update_settings(self, new_settings: GlobalSettings):
        """Updates the global settings in memory (does not persist to disk automatically)."""
        self._settings = new_settings

    def save_settings(self, path: str = DEFAULT_SETTINGS_PATH):
        """Persists the current settings to disk."""
        if self._settings:
            with open(path, 'w') as f:
                f.write(self._settings.model_dump_json(indent=4))
