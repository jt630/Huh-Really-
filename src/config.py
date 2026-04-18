"""
Configuration - loads from .env via pydantic-settings.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Optional: demo mode runs entirely from pre-computed fixtures and does not
    # require a key. Live mode (no fixture_outputs) will raise at the first
    # agent call if this is unset.
    anthropic_api_key: str | None = None
    ncbi_api_key: str | None = None
    census_api_key: str | None = None
    epa_aqs_email: str | None = None
    epa_aqs_key: str | None = None
    claude_model: str = "claude-sonnet-4-6"
    cache_dir: Path = Path("data/cache")
    results_dir: Path = Path("results")
    correlation_alpha: float = 0.05
    correlation_top_n: int = 20
    literature_max_results: int = 50

    def model_post_init(self, __context: object) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
