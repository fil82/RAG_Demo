import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.models import Document, DocumentChunk, DocumentChunkMetadata

logger = logging.getLogger(__name__)


def chunk_document(
    document: Document, *, chunk_size: int, chunk_overlap: int
) -> list[DocumentChunk]:
    """Split ``document.text`` into overlapping chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    texts = splitter.split_text(document.text)

    doc_id = document.id or ""
    source = document.metadata.source if document.metadata else None
    source_id = (
        document.metadata.source_id if document.metadata else None
    ) or doc_id

    chunks = [
        DocumentChunk(
            id=f"{doc_id}_chunk_{index}",
            text=text,
            metadata=DocumentChunkMetadata(
                source=source, source_id=source_id, document_id=doc_id
            ),
        )
        for index, text in enumerate(texts)
    ]
    logger.debug("Document '%s' split into %d chunks", doc_id, len(chunks))
    return chunks
