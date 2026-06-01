import json
import logging
from pathlib import Path

from db.vector_store import VectorStore
from models.models import Document, DocumentMetadata
from services.embed.embedder import EmbeddingService
from services.ingestion.chunking import chunk_document

logger = logging.getLogger(__name__)


def load_documents(path: Path) -> list[Document]:
    """Read a JSONL corpus of documetns."""
    documents: list[Document] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            documents.append(
                Document(
                    id=raw.get("id"),
                    text=raw["text"],
                    metadata=DocumentMetadata(
                        source=raw.get("source"), source_id=raw.get("id")
                    ),
                )
            )
    return documents


class IngestionService:
    """Run the full ingestion pipeline: read -> chunk -> embed -> index."""

    def __init__(
        self,
        embedder: EmbeddingService,
        store: VectorStore,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest(
        self, path: Path, *, recreate_index: bool = False
    ) -> dict[str, int]:
        """Ingest a JSONL corpus."""
        path = Path(path)
        documents = load_documents(path)

        chunks = []
        for document in documents:
            chunks.extend(
                chunk_document(
                    document,
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._chunk_overlap,
                )
            )
        logger.info(
            "Loaded %d documents -> %d chunks", len(documents), len(chunks)
        )

        embeddings = self._embedder.embed_texts(
            [chunk.text for chunk in chunks]
        )
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        self._store.ensure_index(recreate=recreate_index)
        indexed = self._store.upsert(chunks)

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "indexed": len(indexed),
        }
