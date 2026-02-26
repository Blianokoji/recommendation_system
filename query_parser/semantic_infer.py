import numpy as np
from embeddings.embedding_singleton import EmbeddingModelSingleton
from .semantic_axes import SEMANTIC_AXES

_model = EmbeddingModelSingleton.get_model("all-MiniLM-L6-v2")

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def infer_soft_intent(query: str, threshold: float = 0.35):
    """
    Infers soft semantic intent from query using semantic axes.
    Returns open-vocabulary tags.
    """
    query_emb = _model.encode(query, normalize_embeddings=True)

    inferred = {}

    for axis, phrases in SEMANTIC_AXES.items():
        phrase_embs = _model.encode(phrases, normalize_embeddings=True)

        sims = [cosine_sim(query_emb, p) for p in phrase_embs]
        max_sim = max(sims)

        if max_sim >= threshold:
            inferred[axis] = {
                "confidence": round(float(max_sim), 3),
                "matched_phrase": phrases[sims.index(max_sim)]
            }

    return inferred
