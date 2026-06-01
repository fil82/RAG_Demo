import logging

from fastapi import FastAPI, HTTPException

from core.config import get_settings
from core.logging_config import setup_logging
from db.vector_store import get_vector_store
from models.models import Query, QueryResult
from models.troubleshooting import GoodToGoInfo, GoodToGoStatus
from services.embed.embedder import EmbeddingService
from services.retrieval.retriever import RetrievalService

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(title="RAG Demo", version="0.1.0")

    store = get_vector_store(settings)
    embedder = EmbeddingService(settings.embedding_model)
    retrieval = RetrievalService(
        embedder,
        store,
        default_top_k=settings.default_top_k,
        mode=settings.retrieval_mode,
    )

    @app.get("/gtg", response_model=GoodToGoInfo, tags=["health"])
    def good_to_go() -> GoodToGoInfo:
        """Readiness probe: OK only if Elasticsearch is reachable."""
        if not store.health():
            raise HTTPException(status_code=503, detail="Elasticsearch unavailable")
        return GoodToGoInfo(gtg=GoodToGoStatus.OK)

    @app.post("/retrieve", response_model=QueryResult, tags=["retrieval"])
    def retrieve(query: Query) -> QueryResult:
        """Embed the question, search the corpus, and return ranked chunks."""
        return retrieval.retrieve(query.query, query.top_k)

    return app


app = create_app()


def start() -> None:
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8080)
