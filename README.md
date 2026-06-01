# RAG Demo — Document Q&A API with Vector Search

A small Retrieval-Augmented Generation service. It ingests a corpus of public-domain
text, chunks and embeds it, stores the vectors in **Elasticsearch**, and exposes a
**FastAPI** endpoint that answers questions by **hybrid search** — dense kNN and lexical
BM25 combined with Reciprocal Rank Fusion (RRF).
---

## Architecture

```
                ┌──────────────── ingestion (CLI) ─────────────────┐
                │  read JSONL → chunk (overlap) → embed → index     │
samples/*.jsonl ─┤                                                  ├─► Elasticsearch
                │  RecursiveCharacterTextSplitter   bge-m3 (1024-d) │   (dense_vector +
                └──────────────────────────────────────────────────┘    analyzed text)
                                                                              ▲
   POST /retrieve ─► embed question ─► kNN ┐                                  │
                                           ├─► RRF fuse ─► answer ────────────┘
                  ─► question text ──► BM25 ┘   (hybrid; mode-configurable)
   GET  /gtg      ─► Elasticsearch ping (readiness)
```

Layers depend inward through small interfaces, so each piece is swappable and testable:

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Config / observability | [`app/core/`](app/core/) | Settings, structured logging |
| Models | [`app/models/`](app/models/) | Pydantic schemas for documents, chunks, queries |
| Embeddings | [`app/services/embed/`](app/services/embed/) | `sentence-transformers` wrapper (lazy-loaded) |
| Vector store | [`app/db/`](app/db/) | `VectorStore` protocol + Elasticsearch provider |
| Ingestion | [`app/services/ingestion/`](app/services/ingestion/) | Chunking + pipeline + CLI |
| Retrieval | [`app/services/retrieval/`](app/services/retrieval/) | Embed → search → synthesize |
| API | [`app/server/`](app/server/) | FastAPI app: `/retrieve`, `/gtg` |

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (for local dev)
- Docker + Docker Compose

---

## Quickstart — with Docker

Brings up Elasticsearch, runs ingestion once, then starts the API:

```bash
docker compose up --build
```

Then query it:

```bash
curl localhost:8080/gtg

curl -s localhost:8080/retrieve \
  -H 'content-type: application/json' \
  -d '{"query": "What is the capital of Nepal?", "top_k": 3}' | jq
```

---

## Local development (uv)

All commands run from the `app/` directory (the project root for uv):

```bash
cd app
uv sync --extra ml      # add the embedding model stack when running for real
```

### Ingest

Needs a reachable Elasticsearch (`docker compose up elasticsearch -d`):

```bash
uv run python -m services.ingestion --recreate-index
```

### Serve

```bash
uv run uvicorn server.main:app --reload --port 8080
```

Swagger API docs: <http://localhost:8080/docs>

---

## API

### `POST /retrieve`

Request:
```json
{ "query": "What is the capital of Nepal?", "top_k": 3 }
```

Response:
```json
{
  "query": "What is the capital of Nepal?",
  "results": [
    { "id": "doc_1_chunk_0", "text": "Kathmandu ...", "score": 0.83,
      "metadata": { "source": "wikipedia", "source_id": "doc_1", "document_id": "doc_1" } }
  ],
  "answer": "Kathmandu ...\n\n..."
}
```

### `GET /gtg`

Readiness probe — `200 {"gtg": "OK"}` when Elasticsearch is reachable, `503` otherwise.

---

## Testing

```bash
cd app
uv run pytest                  # unit tests 
uv run pytest -m acceptance    # end-to-end acceptance
```

- **Unit** ([`tests/unit/`](tests/unit/))
- **Acceptance** ([`tests/acceptance_tests/`](tests/acceptance_tests/)) — hits `/gtg` and
  `/retrieve` against `API_BASE_URL` (default `http://localhost:8080`). Run after
  `docker compose up`.

---

## Configuration

All settings have defaults; override via environment or a `.env` file (see
[`app/.env.example`](app/.env.example)).

| Variable | Default                 | Purpose |
|----------|-------------------------|---------|
| `ELASTICSEARCH_URL` | `http://localhost:9200` | Elasticsearch endpoint |
| `ELASTICSEARCH_INDEX` | `documents`             | Index name |
| `EMBEDDING_MODEL` | `BAAI/bge-m3`           | sentence-transformers model |
| `EMBEDDING_DIM` | `1024`                  | Vector dimension (must match the model) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200`          | Chunking |
| `DEFAULT_TOP_K` | `3`                     | Default results per query |
| `RETRIEVAL_MODE` | `hybrid`                | Retrieval strategy: `hybrid` (RRF) / `dense` / `bm25` |
| `RRF_K` | `60`                    | RRF rank constant (larger → flatter rank weighting) |
| `RRF_CANDIDATES` | `50`                    | Depth fetched from each retriever before fusion |
| `HNSW_TYPE` | `int8_hnsw`             | `dense_vector` index kind: `int8_hnsw` / `int4_hnsw` / `hnsw` / `bbq_hnsw` / `flat` |
| `HNSW_M` | `32`                     | HNSW neighbors per node (higher → better recall, more memory) |
| `HNSW_EF_CONSTRUCTION` | `512`                   | HNSW build-time candidate list (higher → better graph, slower indexing) |
| `LOG_LEVEL` | `INFO`                  | Logging level |


---

## Possible improvements

- Real LLM answer generation over the retrieved chunks.
- Agentic RAG
- GraphRAG if fit corpus
- Cross-encoder reranking of the fused candidates for even better precision.
- Auth, rate limiting, and a metrics observation for production readiness.
