from __future__ import annotations

import numpy as np

from app.embedding.embedder import EmbeddingConfig, EmbeddingService
from app.vectorstore.faiss_store import DocumentChunk, FAISSVectorStore


def test_faiss_store_add_and_search() -> None:
    embedder = EmbeddingService(EmbeddingConfig(backend="mock-vertex", use_cache=False))

    docs = [
        DocumentChunk("c1", "autoscaling handles traffic spikes", {"topic": "scaling"}),
        DocumentChunk("c2", "caching reduces API latency", {"topic": "performance"}),
    ]
    vectors = np.array(embedder.generate_embeddings([d.text for d in docs]), dtype=np.float32)

    store = FAISSVectorStore(dimension=vectors.shape[1])
    store.add_documents(docs, vectors)

    query = np.array(embedder.generate_embedding("traffic spikes and autoscaling"), dtype=np.float32)
    results = store.search(query_embedding=query, top_k=2)

    assert len(results) == 2
    assert all(r.chunk_id in {"c1", "c2"} for r in results)


def test_faiss_store_save_and_load(tmp_path) -> None:
    embedder = EmbeddingService(EmbeddingConfig(backend="mock-vertex", use_cache=False))

    docs = [DocumentChunk("c1", "fault isolation via bulkheads", {"topic": "resilience"})]
    vectors = np.array(embedder.generate_embeddings([docs[0].text]), dtype=np.float32)

    store = FAISSVectorStore(dimension=vectors.shape[1])
    store.add_documents(docs, vectors)

    out_dir = tmp_path / "index"
    store.save_index(str(out_dir))

    loaded = FAISSVectorStore.load_index(str(out_dir))
    query = np.array(embedder.generate_embedding("bulkhead pattern"), dtype=np.float32)
    results = loaded.search(query_embedding=query, top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "c1"
