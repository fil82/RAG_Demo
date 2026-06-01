from services.embed.embedder import EmbeddingService


class MockHFModel:
    """Stand-in for SentenceTransformer; records calls, returns toy vectors."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        self.calls.append(
            {"texts": list(texts), "normalize": normalize_embeddings}
        )
        return [[float(len(t)), 1.0, 0.0] for t in texts]


def test_embed_texts_returns_one_vector_per_text():
    model = MockHFModel()
    service = EmbeddingService("fake-model", model=model)
    vectors = service.embed_texts(["a", "bb"])
    assert vectors == [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]
    assert model.calls[0]["normalize"] is True


def test_embed_query_returns_single_vector():
    service = EmbeddingService("fake-model", model=FakeModel())
    assert service.embed_query("abc") == [3.0, 1.0, 0.0]


def test_embed_empty_input_is_noop():
    model = MockHFModel()
    service = EmbeddingService("fake-model", model=model)
    assert service.embed_texts([]) == []
    assert model.calls == []  # model is never invoked
