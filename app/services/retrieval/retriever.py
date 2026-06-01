"""Retrieval service: embed a question, search, synthesize an answer."""

import logging

from core.config import RetrievalMode
from db.vector_store import VectorStore
from models.models import DocumentChunkWithScore, QueryResult
from services.embed.embedder import EmbeddingService

logger = logging.getLogger(__name__)

_NO_RESULTS = "No relevant information was found in the corpus."


class RetrievalService:
    def __init__(
        self,
        embedder: EmbeddingService,
        store: VectorStore,
        default_top_k: int = 3,
        mode: RetrievalMode = RetrievalMode.HYBRID,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._default_top_k = default_top_k
        self._mode = RetrievalMode(mode)

    def retrieve(self, question: str, top_k: int | None = None) -> QueryResult:
        k = top_k or self._default_top_k
        if self._mode == RetrievalMode.BM25:
            results = self._store.keyword_query(question, k)
        elif self._mode == RetrievalMode.DENSE:
            embedding = self._embedder.embed_query(question)
            results = self._store.query(embedding, k)
        else:  
            embedding = self._embedder.embed_query(question)
            results = self._store.hybrid_query(question, embedding, k)
        logger.info("Retrieved %d chunks for question", len(results))
        return QueryResult(
            query=question,
            results=results,
            answer=self._synthesize(results),
        )

    @staticmethod
    def _synthesize(results: list[DocumentChunkWithScore]) -> str:
        """Return the single best-matching chunk as the answer.

        A real system would feed these chunks to an LLM; the task allows a
        simple response, and this is the seam where an LLM call would slot in.
        """
        if not results:
            return _NO_RESULTS
        return results[0].text
