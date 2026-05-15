from __future__ import annotations

from app.embedding.embedder import EmbeddingConfig, EmbeddingService
from app.embedding.mock_vertex_embedding import TextEmbeddingModel


def test_mock_vertex_embedding_shape_and_determinism() -> None:
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko-mock")
    emb_1 = model.get_embeddings(["distributed systems"])[0].values
    emb_2 = model.get_embeddings(["distributed systems"])[0].values

    assert len(emb_1) == 384
    assert emb_1 == emb_2


def test_embedding_service_generate_embeddings_mock_backend() -> None:
    service = EmbeddingService(
        EmbeddingConfig(backend="mock-vertex", use_cache=False)
    )
    vectors = service.generate_embeddings(["autoscaling", "caching"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert vectors[0] != vectors[1]
