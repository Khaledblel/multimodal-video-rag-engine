from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    qdrant_url: str = "http://qdrant:6333"
    data_dir: Path = Path("/data")
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
