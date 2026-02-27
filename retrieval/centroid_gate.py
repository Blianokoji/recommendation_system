import os
import json
import numpy as np
from embeddings.embedding_singleton import EmbeddingModelSingleton
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ PATHS ------------------

BASE_DIR = os.getenv("DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CENTROID_DIR = os.path.join(BASE_DIR, "centroids")
CENTROIDS_FILE = os.path.join(CENTROID_DIR, "actor_centroids.npy")
ACTOR_INDEX_FILE = os.path.join(CENTROID_DIR, "actor_index.json")

EMBED_MODEL = "all-MiniLM-L6-v2"

# ------------------ LAZY LOAD ------------------

_centroids = None
_index_to_actor = None
_model = None

def _load_resources():
    global _centroids, _index_to_actor, _model
    
    if _centroids is None:
        if not os.path.exists(CENTROIDS_FILE):
             raise FileNotFoundError(f"Centroids file missing at {CENTROIDS_FILE}. Have you uploaded the volume?")
        _centroids = np.load(CENTROIDS_FILE)
        
    if _index_to_actor is None:
        if not os.path.exists(ACTOR_INDEX_FILE):
             raise FileNotFoundError(f"Actor index missing at {ACTOR_INDEX_FILE}")
        with open(ACTOR_INDEX_FILE, "r", encoding="utf-8") as f:
            _actor_index = json.load(f)
        _index_to_actor = {v: k for k, v in _actor_index.items()}
        
    if _model is None:
        _model = EmbeddingModelSingleton.get_model(EMBED_MODEL)

# ------------------ CORE ------------------

def get_relevant_actor_centroids(
    query: str,
    top_k: int = 3,
    threshold: float = 0.35
):
    """
    Returns actor names whose centroid is relevant to the query.
    """

    _load_resources()

    q_emb = _model.encode(query, normalize_embeddings=True).reshape(1, -1)
    sims = cosine_similarity(q_emb, _centroids)[0]

    ranked = sorted(
        enumerate(sims),
        key=lambda x: x[1],
        reverse=True
    )

    selected = []
    for idx, score in ranked[:top_k]:
        if score >= threshold:
            selected.append({
                "actor": _index_to_actor[idx],
                "similarity": float(score)
            })

    return selected
