from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np

from app.vectorstore.similarity import keyword_overlap_score, normalize_vectors


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


@dataclass(slots=True)
class SearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class FAISSVectorStore:
    """FAISS index with chunk metadata and cosine-compatible similarity search."""

    def __init__(self, dimension: int, normalize: bool = True) -> None:
        self.dimension = dimension
        self.normalize = normalize
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[DocumentChunk] = []

    def add_documents(self, documents: List[DocumentChunk], embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError("Embeddings shape does not match index dimension")

        vectors = embeddings.astype(np.float32)
        if self.normalize:
            vectors = normalize_vectors(vectors)

        self.index.add(vectors)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        query = query_embedding.astype(np.float32).reshape(1, -1)
        if self.normalize:
            query = normalize_vectors(query)

        scores, indices = self.index.search(query, top_k)
        results: List[SearchResult] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            results.append(
                SearchResult(
                    chunk_id=doc.chunk_id,
                    text=doc.text,
                    score=float(score),
                    metadata=doc.metadata,
                )
            )
        return results

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 3,
        semantic_weight: float = 0.8,
    ) -> List[SearchResult]:
        semantic_results = self.search(query_embedding=query_embedding, top_k=max(top_k * 2, top_k))
        reranked = []
        for item in semantic_results:
            lexical = keyword_overlap_score(query_text, item.text)
            combined_score = semantic_weight * item.score + (1 - semantic_weight) * lexical
            reranked.append(
                SearchResult(
                    chunk_id=item.chunk_id,
                    text=item.text,
                    score=float(combined_score),
                    metadata={**item.metadata, "semantic_score": item.score, "keyword_score": lexical},
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]

    def save_index(self, directory: str) -> None:
        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(base / "faiss.index"))
        payload = {
            "dimension": self.dimension,
            "normalize": self.normalize,
            "documents": [asdict(doc) for doc in self.documents],
        }
        (base / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_index(cls, directory: str) -> "FAISSVectorStore":
        base = Path(directory)
        payload = json.loads((base / "metadata.json").read_text(encoding="utf-8"))

        store = cls(dimension=int(payload["dimension"]), normalize=bool(payload["normalize"]))
        store.index = faiss.read_index(str(base / "faiss.index"))
        store.documents = [DocumentChunk(**doc) for doc in payload["documents"]]
        return store
