from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass(slots=True)
class TextEmbedding:
    """Mimics Vertex AI TextEmbedding response object."""

    values: List[float]


class TextEmbeddingModel:
    """Lightweight local mock for vertexai.language_models.TextEmbeddingModel."""

    def __init__(self, model_name: str = "textembedding-gecko", dimension: int = 384) -> None:
        self.model_name = model_name
        self.dimension = dimension

    @classmethod
    def from_pretrained(cls, model_name: str) -> "TextEmbeddingModel":
        return cls(model_name=model_name)

    def _embed_text(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.dimension).astype(np.float32)
        vector /= np.linalg.norm(vector) + 1e-12
        return vector.tolist()

    def get_embeddings(self, texts: Iterable[str]) -> List[TextEmbedding]:
        return [TextEmbedding(values=self._embed_text(text)) for text in texts]
