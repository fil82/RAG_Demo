import logging
from typing import Any, Optional

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from models.models import DocumentChunk, DocumentChunkMetadata, DocumentChunkWithScore

logger = logging.getLogger(__name__)


class ElasticsearchVectorStore:
    """Stores chunk text + embedding and retrieves via the ``knn`` query."""

    def __init__(
        self,
        url: str,
        index: str,
        embedding_dim: int,
        client: Optional[Elasticsearch] = None,
        rrf_k: int = 60,
        rrf_candidates: int = 50,
        hnsw_type: str = "int8_hnsw",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 100,
    ) -> None:
        self._index = index
        self._dim = embedding_dim
        self._rrf_k = rrf_k
        self._rrf_candidates = rrf_candidates
        self._hnsw_type = hnsw_type
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._client = client or Elasticsearch(url, request_timeout=30)

    def health(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            logger.exception("Elasticsearch ping failed")
            return False

    def ensure_index(self, recreate: bool = False) -> None:
        exists = self._client.indices.exists(index=self._index)
        if recreate and exists:
            self._client.indices.delete(index=self._index)
            exists = False
        if exists:
            return
        self._client.indices.create(index=self._index, mappings=self._mapping())
        logger.info("Created index '%s' (dims=%d)", self._index, self._dim)

    def upsert(self, chunks: list[DocumentChunk]) -> list[str]:
        if not chunks:
            return []
        actions = [self._to_action(chunk) for chunk in chunks]
        success, errors = bulk(self._client, actions, refresh=True)
        if errors:
            logger.warning("Bulk index reported errors: %s", errors)
        logger.info("Indexed %d chunks into '%s'", success, self._index)
        return [chunk.id for chunk in chunks if chunk.id]

    def query(
        self, embedding: list[float], top_k: int
    ) -> list[DocumentChunkWithScore]:
        response = self._client.search(
            index=self._index,
            knn={
                "field": "embedding",
                "query_vector": embedding,
                "k": top_k,
                "num_candidates": max(top_k * 10, 50),
            },
            source_excludes=["embedding"],
            size=top_k,
        )
        hits = response["hits"]["hits"]
        return [self._to_chunk(hit) for hit in hits]

    def keyword_query(
        self, query_text: str, top_k: int
    ) -> list[DocumentChunkWithScore]:
        """Lexical BM25 search over the analyzed ``text`` field."""
        response = self._client.search(
            index=self._index,
            query={"match": {"text": query_text}},
            source_excludes=["embedding"],
            size=top_k,
        )
        hits = response["hits"]["hits"]
        return [self._to_chunk(hit) for hit in hits]

    def hybrid_query(
        self, query_text: str, embedding: list[float], top_k: int
    ) -> list[DocumentChunkWithScore]:
        """Combine dense kNN and BM25 rankings via Reciprocal Rank Fusion.

        Runs both searches in a single ``_msearch`` round-trip, then fuses the
        two ranked lists in Python so neither dense cosine scores nor BM25
        scores need normalizing (RRF is rank-based).
        """
        candidates = max(top_k, self._rrf_candidates)
        dense_body = {
            "knn": {
                "field": "embedding",
                "query_vector": embedding,
                "k": candidates,
                "num_candidates": max(candidates * 4, 100),
            },
            "size": candidates,
            "_source": {"excludes": ["embedding"]},
        }
        bm25_body = {
            "query": {"match": {"text": query_text}},
            "size": candidates,
            "_source": {"excludes": ["embedding"]},
        }
        response = self._client.msearch(
            searches=[
                {"index": self._index},
                dense_body,
                {"index": self._index},
                bm25_body,
            ]
        )
        dense_hits = response["responses"][0]["hits"]["hits"]
        bm25_hits = response["responses"][1]["hits"]["hits"]
        return self._rrf_fuse([dense_hits, bm25_hits], top_k)

    def _rrf_fuse(
        self, ranked_lists: list[list[dict[str, Any]]], top_k: int
    ) -> list[DocumentChunkWithScore]:
        """Fuse ranked hit lists with RRF: score = sum 1/(rrf_k + rank)."""
        scores: dict[str, float] = {}
        hits: dict[str, dict[str, Any]] = {}
        for ranked in ranked_lists:
            for rank, hit in enumerate(ranked, start=1):
                doc_id = hit["_id"]
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (
                    self._rrf_k + rank
                )
                hits.setdefault(doc_id, hit)
        ordered = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
        return [self._to_chunk(hits[doc_id], scores[doc_id]) for doc_id in ordered]

    def _mapping(self) -> dict[str, Any]:
        return {
            "properties": {
                "text": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": self._dim,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {
                        "type": self._hnsw_type,
                        "m": self._hnsw_m,
                        "ef_construction": self._hnsw_ef_construction,
                    },
                },
                "metadata": {
                    "properties": {
                        "source": {"type": "keyword"},
                        "source_id": {"type": "keyword"},
                        "document_id": {"type": "keyword"},
                    }
                },
            }
        }

    def _to_action(self, chunk: DocumentChunk) -> dict[str, Any]:
        return {
            "_index": self._index,
            "_id": chunk.id,
            "text": chunk.text,
            "embedding": chunk.embedding,
            "metadata": chunk.metadata.model_dump(mode="json"),
        }

    @staticmethod
    def _to_chunk(
        hit: dict[str, Any], score: Optional[float] = None
    ) -> DocumentChunkWithScore:
        source = hit.get("_source", {})
        return DocumentChunkWithScore(
            id=hit["_id"],
            text=source.get("text", ""),
            metadata=DocumentChunkMetadata(**source.get("metadata", {})),
            score=score if score is not None else hit.get("_score"),
        )
