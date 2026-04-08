"""
Semantic Intent Inference
-------------------------
Single source of truth for soft intent inference.

Compares the query embedding against SEMANTIC_AXES anchor phrases
via cosine similarity and routes matches into:
  - soft_constraints: axes that steer vector retrieval (emotion, tone)
  - inferred_signals: axes used as metadata signals only (genre, etc.)

Imported by parse_query.py — do NOT duplicate this logic there.
"""

import numpy as np
from embeddings.embedding_singleton import EmbeddingModelSingleton
from .semantic_axes import SEMANTIC_AXES

_model = EmbeddingModelSingleton.get_model("all-MiniLM-L6-v2")

# Axes that directly constrain retrieval (blended into query embedding)
SOFT_CONSTRAINT_AXES = {"emotion", "tone"}


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def infer_soft_intent(query: str, threshold: float = 0.35) -> dict:
    """
    Infers semantic intent from query using SEMANTIC_AXES.

    Returns:
        {
            "soft_constraints": {axis: {"matched_phrase": str, "confidence": float}},
            "inferred_signals":  {axis: {"matched_phrase": str, "confidence": float}}
        }
    """
    if not query.strip():
        return {"soft_constraints": {}, "inferred_signals": {}}

    query_emb = _model.encode(query, normalize_embeddings=True)

    soft_constraints = {}
    inferred_signals = {}

    for axis, phrases in SEMANTIC_AXES.items():
        phrase_embs = _model.encode(phrases, normalize_embeddings=True)
        sims = [_cosine_sim(query_emb, p) for p in phrase_embs]
        max_sim = max(sims)

        if max_sim >= threshold:
            data = {
                "confidence": round(float(max_sim), 3),
                "matched_phrase": phrases[sims.index(max_sim)]
            }
            if axis in SOFT_CONSTRAINT_AXES:
                soft_constraints[axis] = data
            else:
                inferred_signals[axis] = data

    return {
        "soft_constraints": soft_constraints,
        "inferred_signals": inferred_signals
    }
