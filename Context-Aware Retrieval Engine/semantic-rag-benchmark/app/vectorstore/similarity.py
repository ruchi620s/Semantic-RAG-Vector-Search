from __future__ import annotations

from typing import Iterable, Set

import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def keyword_overlap_score(query: str, document: str) -> float:
    q_tokens: Set[str] = {tok.lower() for tok in query.split() if len(tok) > 2}
    d_tokens: Set[str] = {tok.lower() for tok in document.split() if len(tok) > 2}
    if not q_tokens:
        return 0.0
    overlap = q_tokens.intersection(d_tokens)
    return len(overlap) / len(q_tokens)


def reciprocal_rank_fusion(ranks: Iterable[int], k: int = 60) -> float:
    return float(sum(1.0 / (k + rank) for rank in ranks))
