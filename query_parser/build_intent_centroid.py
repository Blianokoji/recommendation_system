"""
Build Movie Intent Centroid
---------------------------
One-time offline script. Encodes a curated set of known movie search
queries and saves their mean (L2-normalised) embedding as the intent
centroid used by intent_gate.py at query time.

Run once after first checkout:
    python -m query_parser.build_intent_centroid
"""

import os
import sys

# Allow running as standalone script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from embeddings.embedding_singleton import EmbeddingModelSingleton

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "movie_intent_centroid.npy")
EMBED_MODEL = "all-MiniLM-L6-v2"

# ============================================================
# POSITIVE EXAMPLES  (genuine movie search queries)
# ============================================================

POSITIVE_QUERIES = [
    # Direct / explicit
    "action movies with great fight scenes",
    "romantic comedy films",
    "sci-fi movies with time travel",
    "dark psychological thrillers",
    "best movies of 2019",
    "classic films from the 90s",
    "Tom Cruise action films",
    "oscar winning dramatic films",
    "movies with plot twists",
    "feel good family movies",
    "gripping crime thrillers",
    "low budget horror movies",
    "independent art house cinema",
    "blockbuster adventure films",
    "sad movies that make you cry",
    "inspirational sports movies",
    "historical epic films",
    "movies about AI and technology",
    "good mystery films",
    "romantic period drama movies",
    "fantasy films with magic",
    "biopics about famous people",
    "movies about music",
    "crime drama films",
    "suspenseful movies to watch",
    "animated movies for adults",
    "movies from the 80s",
    "top rated movies ever made",
    "movies about friendship",
    "films set in space",
    "comedy movies that are actually funny",
    "underrated films worth watching",
    "best superhero movies",
    # Natural language / implicit
    "show me something funny",
    "good thriller to watch tonight",
    "horror movies that are scary",
    "what should I watch tonight",
    "something light and funny to watch",
    "show me a good film",
    "I want to watch something emotional",
    "recommend me a movie",
    "I need something to watch",
    "suggest a film for the weekend",
    "something with a good story",
    "looking for a feel good film",
    "anything with Tom Hanks",
    "movies like Inception",
    "something similar to The Dark Knight",
    "show me a tearjerker",
]

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(f"[INFO] Loading model: {EMBED_MODEL}")
    model = EmbeddingModelSingleton.get_model(EMBED_MODEL)

    print(f"[INFO] Encoding {len(POSITIVE_QUERIES)} positive movie queries...")
    embeddings = model.encode(POSITIVE_QUERIES, normalize_embeddings=True)

    centroid = embeddings.mean(axis=0)
    # Re-normalise so dot products equal cosine similarity at query time
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

    np.save(OUTPUT_FILE, centroid)
    print(f"[DONE] Intent centroid saved to: {OUTPUT_FILE}")
    print(f"[DONE] Centroid shape: {centroid.shape}")
