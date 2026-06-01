"""CLI entrypoint: ``python -m services.ingestion [options]``."""

import argparse
import logging
from pathlib import Path

from core.config import get_settings
from core.logging_config import setup_logging
from db.vector_store import get_vector_store
from services.embed.embedder import EmbeddingService
from services.ingestion.pipeline import IngestionService

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("samples/wikipedia_documents.jsonl"),
        help="Path to a JSONL corpus of {id, source, text} records",
    )
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Delete and recreate the index before ingesting",
    )
    args = parser.parse_args(argv)

    setup_logging(settings.log_level)

    store = get_vector_store(settings)
    embedder = EmbeddingService(settings.embedding_model)

    service = IngestionService(
        embedder,
        store,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    summary = service.ingest(args.path, recreate_index=args.recreate_index)
    logger.info("Ingestion complete: %s", summary)


if __name__ == "__main__":
    main()
