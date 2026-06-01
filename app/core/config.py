"""Application configuration, loaded from environment variables / .env."""

from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalMode(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Elasticsearch
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "documents"

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 200

    # Retrieval
    default_top_k: int = 3
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    rrf_k: int = 60
    rrf_candidates: int = 5

    # HNSW index tuning
    hnsw_type: str = "int8_hnsw"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 512

    # Observability
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
