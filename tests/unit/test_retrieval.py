from models.models import DocumentChunkMetadata, DocumentChunkWithScore
from services.retrieval.retriever import RetrievalService


class MockEmbedder:
    def __init__(self):
        self.embed_calls = 0

    def embed_query(self, text):
        self.embed_calls += 1
        return [0.1, 0.2, 0.3]


class MockStore:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []

    def query(self, embedding, top_k):
        self.calls.append(("query", embedding, top_k))
        return self._chunks[:top_k]

    def keyword_query(self, query_text, top_k):
        self.calls.append(("keyword_query", query_text, top_k))
        return self._chunks[:top_k]

    def hybrid_query(self, query_text, embedding, top_k):
        self.calls.append(("hybrid_query", query_text, embedding, top_k))
        return self._chunks[:top_k]


def _chunk(text, score):
    return DocumentChunkWithScore(
        id="x",
        text=text,
        metadata=DocumentChunkMetadata(document_id="d"),
        score=score,
    )


def test_retrieve_returns_chunks_and_concatenated_answer():
    store = MockStore([_chunk("alpha", 0.9), _chunk("beta", 0.8)])
    service = RetrievalService(MockEmbedder(), store, mode="dense")

    result = service.retrieve("a question", top_k=2)

    assert result.query == "a question"
    assert [r.text for r in result.results] == ["alpha", "beta"]
    assert result.answer == "alpha\n\nbeta"
    assert store.calls == [("query", [0.1, 0.2, 0.3], 2)]


def test_retrieve_uses_default_top_k():
    store = MockStore([_chunk("alpha", 0.9)])
    service = RetrievalService(
        MockEmbedder(), store, default_top_k=5, mode="dense"
    )

    service.retrieve("q")

    assert store.calls[0][-1] == 5


def test_retrieve_with_no_results():
    service = RetrievalService(MockEmbedder(), MockStore([]), mode="dense")
    result = service.retrieve("q", top_k=3)
    assert result.results == []
    assert "No relevant" in result.answer


def test_hybrid_mode_is_default_and_calls_hybrid_query():
    store = MockStore([_chunk("alpha", 0.9)])
    embedder = MockEmbedder()
    service = RetrievalService(embedder, store)  # default mode

    service.retrieve("q", top_k=1)

    assert store.calls == [("hybrid_query", "q", [0.1, 0.2, 0.3], 1)]
    assert embedder.embed_calls == 1


def test_bm25_mode_skips_embedding():
    store = MockStore([_chunk("alpha", 0.9)])
    embedder = MockEmbedder()
    service = RetrievalService(embedder, store, mode="bm25")

    service.retrieve("q", top_k=1)

    assert store.calls == [("keyword_query", "q", 1)]
    assert embedder.embed_calls == 0
