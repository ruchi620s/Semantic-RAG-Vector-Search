from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from app.embedding.embedder import EmbeddingService
from app.retrieval.query_expander import QueryExpansionService
from app.vectorstore.faiss_store import FAISSVectorStore, SearchResult


class Retriever:
    """Implements raw and AI-enhanced retrieval strategies."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
        query_expander: QueryExpansionService,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.query_expander = query_expander

    @staticmethod
    def _serialize_results(results: List[SearchResult]) -> List[Dict[str, Any]]:
        return [
            {
                "chunk_id": item.chunk_id,
                "text": item.text,
                "score": round(item.score, 4),
                "metadata": item.metadata,
            }
            for item in results
        ]

    def raw_search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        start = time.perf_counter()
        query_vector = np.array(self.embedding_service.generate_embedding(query), dtype=np.float32)
        results = self.vector_store.search(query_embedding=query_vector, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "strategy": "raw_vector_search",
            "query": query,
            "latency_ms": round(latency_ms, 2),
            "results": self._serialize_results(results),
        }

    def enhanced_search(self, query: str, top_k: int = 3, hybrid: bool = True) -> Dict[str, Any]:
        start = time.perf_counter()
        expanded_query = self.query_expander.expand_query(query)
        query_vector = np.array(self.embedding_service.generate_embedding(expanded_query), dtype=np.float32)

        if hybrid:
            results = self.vector_store.hybrid_search(
                query_embedding=query_vector,
                query_text=expanded_query,
                top_k=top_k,
            )
            strategy_name = "enhanced_query_expansion_hybrid"
        else:
            results = self.vector_store.search(query_embedding=query_vector, top_k=top_k)
            strategy_name = "enhanced_query_expansion"

        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "strategy": strategy_name,
            "query": query,
            "expanded_query": expanded_query,
            "latency_ms": round(latency_ms, 2),
            "results": self._serialize_results(results),
        }
