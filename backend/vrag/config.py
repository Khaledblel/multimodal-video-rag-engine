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

    text_embed_model: str = "qwen3-embedding:0.6b"
    text_embed_dim: int = 1024
    index_enabled: bool = True

    frame_embed_enabled: bool = True
    frame_embed_model: str = "google/siglip2-so400m-patch14-384"
    frame_embed_dim: int = 1152
    embed_device: str = "auto"
    embedder_url: str = "http://embedder:8001"
    fusion_text_weight: float = 1.0
    fusion_frame_weight: float = 1.0

    vlm_enabled: bool = True
    vlm_model: str = "qwen3-vl:8b-instruct"
    vlm_num_ctx: int = 16384
    vlm_frame_side: int = 768
    vlm_keep_alive: str = "5m"
    vlm_think: bool = False
    planner_enabled: bool = True
    ollama_url: str = "http://ollama:11434"


@lru_cache
def get_settings() -> Settings:
    return Settings()
