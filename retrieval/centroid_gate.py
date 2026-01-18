import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ PATHS ------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CENTROID_DIR = os.path.join(BASE_DIR, "centroids")
CENTROIDS_FILE = os.path.join(CENTROID_DIR, "actor_centroids.npy")
ACTOR_INDEX_FILE = os.path.join(CENTROID_DIR, "actor_index.json")

EMBED_MODEL = "all-MiniLM-L6-v2"

# ------------------ LOAD ------------------

_centroids = np.load(CENTROIDS_FILE)
with open(ACTOR_INDEX_FILE, "r", encoding="utf-8") as f:
    _actor_index = json.load(f)

_index_to_actor = {v: k for k, v in _actor_index.items()}

_model = SentenceTransformer(EMBED_MODEL)

# ------------------ CORE ------------------

def get_relevant_actor_centroids(
    query: str,
    top_k: int = 3,
    threshold: float = 0.35
):
    """
    Returns actor names whose centroid is relevant to the query.
    """

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
