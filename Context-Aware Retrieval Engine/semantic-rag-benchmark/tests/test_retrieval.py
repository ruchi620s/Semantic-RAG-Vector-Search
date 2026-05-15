from __future__ import annotations

import numpy as np

from app.embedding.embedder import EmbeddingConfig, EmbeddingService
from app.retrieval.query_expander import QueryExpansionService
from app.retrieval.retriever import Retriever
from app.vectorstore.faiss_store import DocumentChunk, FAISSVectorStore


def _build_retriever() -> Retriever:
    embedder = EmbeddingService(EmbeddingConfig(backend="mock-vertex", use_cache=False))
    docs = [
        DocumentChunk("a", "autoscaling and load balancing protect peak traffic", {"topic": "scale"}),
        DocumentChunk("b", "caching and API gateways improve response speed", {"topic": "perf"}),
        DocumentChunk("c", "circuit breakers isolate failures in microservices", {"topic": "resilience"}),
    ]

    vectors = np.array(embedder.generate_embeddings([d.text for d in docs]), dtype=np.float32)
    store = FAISSVectorStore(dimension=vectors.shape[1])
    store.add_documents(docs, vectors)

    return Retriever(embedder, store, QueryExpansionService())


def test_raw_and_enhanced_search_return_top_k() -> None:
    retriever = _build_retriever()

    raw = retriever.raw_search("How does the system handle peak load?", top_k=2)
    enhanced = retriever.enhanced_search("How does the system handle peak load?", top_k=2)

    assert raw["strategy"] == "raw_vector_search"
    assert enhanced["strategy"] == "enhanced_query_expansion_hybrid"
    assert len(raw["results"]) == 2
    assert len(enhanced["results"]) == 2
    assert enhanced["expanded_query"] != "How does the system handle peak load?"


def test_enhanced_search_includes_latency() -> None:
    retriever = _build_retriever()
    result = retriever.enhanced_search("What improves API response speed?", top_k=3)

    assert result["latency_ms"] >= 0
    assert isinstance(result["results"], list)
