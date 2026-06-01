"""Unit tests for Reciprocal Rank Fusion (no Elasticsearch needed)."""

from db.providers.elasticsearch_store import ElasticsearchVectorStore


def _hit(doc_id, text="t", score=1.0):
    return {
        "_id": doc_id,
        "_score": score,
        "_source": {"text": text, "metadata": {"document_id": doc_id}},
    }


def _store(rrf_k=60):
    # client=object() avoids constructing a real Elasticsearch connection.
    return ElasticsearchVectorStore(
        url="http://x", index="i", embedding_dim=3, client=object(), rrf_k=rrf_k
    )


def test_rrf_fuse_orders_by_summed_reciprocal_rank():
    store = _store(rrf_k=60)
    dense = [_hit("a"), _hit("b"), _hit("c")]
    bm25 = [_hit("b"), _hit("d"), _hit("a")]

    fused = store._rrf_fuse([dense, bm25], top_k=4)
    order = [c.id for c in fused]

    # b: 1/61 + 1/61, a: 1/61 + 1/63 -> b first, a second.
    assert order[0] == "b"
    assert order[1] == "a"
    assert set(order) == {"a", "b", "c", "d"}


def test_rrf_fuse_doc_in_both_lists_outranks_singletons():
    store = _store(rrf_k=60)
    dense = [_hit("solo1"), _hit("shared")]
    bm25 = [_hit("solo2"), _hit("shared")]

    fused = store._rrf_fuse([dense, bm25], top_k=3)

    assert fused[0].id == "shared"
    # Fused score is the sum across both lists (rank 2 in each).
    assert fused[0].score == 2 * (1.0 / (60 + 2))


def test_rrf_fuse_respects_top_k():
    store = _store()
    dense = [_hit("a"), _hit("b"), _hit("c")]

    fused = store._rrf_fuse([dense], top_k=2)

    assert len(fused) == 2
    assert [c.id for c in fused] == ["a", "b"]
