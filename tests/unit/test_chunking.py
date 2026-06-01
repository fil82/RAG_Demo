from models.models import Document, DocumentMetadata, Source
from services.ingestion.chunking import chunk_document


def _document(text: str) -> Document:
    return Document(
        id="doc_1",
        text=text,
        metadata=DocumentMetadata(source=Source.wikipedia, source_id="doc_1"),
    )


def test_splits_into_multiple_chunks():
    doc = _document("abcdefghij" * 100)
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1


def test_chunks_overlap():
    doc = _document("abcdefghij" * 100)
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)
    assert chunks[0].text[-20:] in chunks[1].text


def test_chunk_metadata_and_ids():
    doc = _document("abcdefghij" * 100)
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)
    assert [c.id for c in chunks] == [
        f"doc_1_chunk_{i}" for i in range(len(chunks))
    ]
    assert all(c.metadata.document_id == "doc_1" for c in chunks)
    assert all(c.metadata.source == Source.wikipedia for c in chunks)
    assert all(c.metadata.source_id == "doc_1" for c in chunks)


def test_short_document_single_chunk():
    chunks = chunk_document(_document("short text"), chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"
