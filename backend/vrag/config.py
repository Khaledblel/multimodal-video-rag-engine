from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    qdrant_url: str = "http://qdrant:6333"
    data_dir: Path = Path("/data")
    log_level: str = "INFO"

    asr_enabled: bool = True
    asr_model: str = "small"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_language: str = ""

    text_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
