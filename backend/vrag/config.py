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

    frame_embed_enabled: bool = True
    frame_embed_model: str = "google/siglip2-so400m-patch14-384"
    frame_embed_dim: int = 1152
    embed_device: str = "auto"
    embedder_url: str = "http://embedder:8001"
    fusion_text_weight: float = 1.0
    fusion_frame_weight: float = 1.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
