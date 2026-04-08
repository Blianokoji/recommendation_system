"""
Intent Gate
-----------
Classifies whether a raw user query is a movie search intent
using a pre-built semantic centroid (no LLM, no keywords).

How it works:
    1. Embed the query with all-MiniLM-L6-v2
    2. cosine_sim(query_emb, movie_intent_centroid)
    3. If >= INTENT_THRESHOLD → movie_search, else → invalid

Confidence = raw cosine similarity (0–1, calibrated).
Threshold tunable via INTENT_THRESHOLD env var (default: 0.40).

Build the centroid once:
    python -m query_parser.build_intent_centroid
"""

import os
import numpy as np
from embeddings.embedding_singleton import EmbeddingModelSingleton

CENTROID_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "movie_intent_centroid.npy"
)
INTENT_THRESHOLD = float(os.getenv("INTENT_THRESHOLD", "0.40"))
EMBED_MODEL = "all-MiniLM-L6-v2"

# ---- lazy-loaded singletons ----
_model = None
_centroid = None


def _load():
    global _model, _centroid
    if _model is None:
        _model = EmbeddingModelSingleton.get_model(EMBED_MODEL)
    if _centroid is None:
        if not os.path.exists(CENTROID_FILE):
            raise FileNotFoundError(
                f"Intent centroid not found at {CENTROID_FILE}. "
                "Run: python -m query_parser.build_intent_centroid"
            )
        _centroid = np.load(CENTROID_FILE)


def classify_intent(query: str) -> dict:
    """
    Args:
        query: raw user query string

    Returns:
        {
            "intent_type": "movie_search" | "invalid",
            "confidence": float   # cosine similarity vs movie centroid
        }
    """
    _load()

    q_emb = _model.encode(query.strip(), normalize_embeddings=True)
    # centroid is already L2-normalised, so dot product == cosine_sim
    confidence = float(np.dot(q_emb, _centroid))
    confidence = round(min(max(confidence, 0.0), 1.0), 3)

    return {
        "intent_type": "movie_search" if confidence >= INTENT_THRESHOLD else "invalid",
        "confidence": confidence,
    }
