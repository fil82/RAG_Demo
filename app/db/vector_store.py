from typing import Protocol

from core.config import Settings
from models.models import DocumentChunk, DocumentChunkWithScore


class VectorStore(Protocol):
    def ensure_index(self, recreate: bool = False) -> None:
        """Create the index (and mapping) if it does not already exist."""
        ...

    def upsert(self, chunks: list[DocumentChunk]) -> list[str]:
        """Index the given chunks (with embeddings). Returns the stored ids."""
        ...

    def query(
        self, embedding: list[float], top_k: int
    ) -> list[DocumentChunkWithScore]:
        """Return the ``top_k`` most similar chunks to ``embedding``."""
        ...

    def keyword_query(
        self, query_text: str, top_k: int
    ) -> list[DocumentChunkWithScore]:
        """Return the ``top_k`` chunks matching ``query_text`` lexically (BM25)."""
        ...

    def hybrid_query(
        self, query_text: str, embedding: list[float], top_k: int
    ) -> list[DocumentChunkWithScore]:
        """Return the ``top_k`` chunks fusing dense + BM25 rankings via RRF."""
        ...

    def health(self) -> bool:
        """Return True if the backend is reachable."""
        ...


def get_vector_store(settings: Settings) -> VectorStore:
    from db.providers.elasticsearch_store import ElasticsearchVectorStore

    return ElasticsearchVectorStore(
        url=settings.elasticsearch_url,
        index=settings.elasticsearch_index,
        embedding_dim=settings.embedding_dim,
        rrf_k=settings.rrf_k,
        rrf_candidates=settings.rrf_candidates,
        hnsw_type=settings.hnsw_type,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construction=settings.hnsw_ef_construction,
    )
