from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from app.embedding.mock_vertex_embedding import TextEmbeddingModel

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    backend: str = "sentence-transformers"
    cache_path: str = ".cache/embedding_cache.json"
    use_cache: bool = True
    local_files_only: bool = True


class EmbeddingService:
    """Generates embeddings with local sentence-transformers or a Vertex-like mock."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._cache: Dict[str, List[float]] = {}
        self._cache_file = Path(self.config.cache_path)
        self._st_model: Any | None = None
        self._mock_model: TextEmbeddingModel | None = None

        if self.config.use_cache:
            self._load_cache()

    def _load_cache(self) -> None:
        if self._cache_file.exists():
            self._cache = json.loads(self._cache_file.read_text(encoding="utf-8"))
            LOGGER.info("Loaded %d embedding cache entries", len(self._cache))

    def save_cache(self) -> None:
        if not self.config.use_cache:
            return
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache_file.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

    def _init_backend(self) -> None:
        if self.config.backend == "mock-vertex":
            if self._mock_model is None:
                self._mock_model = TextEmbeddingModel.from_pretrained("textembedding-gecko-mock")
            return

        if self._st_model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            # Fail fast in offline or certificate-constrained environments.
            self._st_model = SentenceTransformer(
                self.config.model_name,
                local_files_only=self.config.local_files_only,
            )
            LOGGER.info("Loaded sentence-transformer model: %s", self.config.model_name)
        except Exception as exc:  # pragma: no cover - fallback is environment dependent
            LOGGER.warning(
                "Could not load sentence-transformers model (%s). Falling back to mock backend.",
                exc,
            )
            self.config.backend = "mock-vertex"
            self._mock_model = TextEmbeddingModel.from_pretrained("textembedding-gecko-mock")

    def generate_embedding(self, text: str) -> List[float]:
        return self.generate_embeddings([text])[0]

    def generate_embeddings(self, list_of_texts: List[str]) -> List[List[float]]:
        outputs: List[List[float]] = []

        uncached: List[str] = []
        uncached_positions: List[int] = []

        for idx, text in enumerate(list_of_texts):
            if self.config.use_cache and text in self._cache:
                outputs.append(self._cache[text])
            else:
                outputs.append([])
                uncached.append(text)
                uncached_positions.append(idx)

        if not uncached:
            return outputs

        self._init_backend()

        if self.config.backend == "mock-vertex":
            assert self._mock_model is not None
            generated = [item.values for item in self._mock_model.get_embeddings(uncached)]
        else:
            assert self._st_model is not None
            matrix = self._st_model.encode(
                uncached,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            generated = matrix.astype(np.float32).tolist()

        for pos, text, embedding in zip(uncached_positions, uncached, generated):
            outputs[pos] = embedding
            if self.config.use_cache:
                self._cache[text] = embedding

        return outputs
