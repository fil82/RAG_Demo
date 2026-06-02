"""Embedding service wrapping a sentence-transformers model.

The model is loaded at construction time into ``self.model``. Tests inject a
fake ``model`` to avoid downloading anything.
"""

import logging
import time
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        model: Any = None,
        normalize: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._normalize = normalize
        self.model = model if model is not None else self._load_model()

    def _load_model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model '%s'...", self._model_name)
        start = time.perf_counter()
        model = SentenceTransformer(self._model_name, device=self._device)
        logger.info("Loaded embedding model in %.1fs", time.perf_counter() - start)
        return model

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        return [
            vec.tolist() if hasattr(vec, "tolist") else list(vec)
            for vec in vectors
        ]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
