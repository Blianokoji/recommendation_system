"""
Candidate Retrieval Module (Fixed)
---------------------------------
Uses structured query intent to perform:

1. Hard constraint filtering (actor, safety)
2. Semantic retrieval using soft intent phrases
3. Local clustering for diversity

NO keyword OR behavior.
NO identity leakage into embeddings.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from vector_store.chroma_client import get_chroma_collection
from query_parser.parse_query import parse_query

# ------------------ CONFIG ------------------

RETRIEVAL_LIMIT = 80
LOCAL_CLUSTERS = 4
RETURN_PER_CLUSTER = 5

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ------------------ MODEL ------------------

_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ------------------ CORE ------------------

def embed_text(text: str) -> np.ndarray:
    return _embedding_model.encode(text, normalize_embeddings=True)


def retrieve_candidates(
    query: str,
    allow_adult_override: bool = None
) -> List[Dict]:

    parsed = parse_query(query)

    if parsed["intent_type"] != "movie_search":
        return []

    collection = get_chroma_collection()

    # ------------------------------------------------
    # HARD CONSTRAINTS → WHERE FILTER
    # ------------------------------------------------

    where_filters = []

    # Adult safety
    allow_adult = parsed["filters"]["allow_adult"]
    if allow_adult_override is not None:
        allow_adult = allow_adult_override

    if not allow_adult:
        where_filters.append({"adult": False})

    if len(where_filters) == 1:
        where_clause = where_filters[0]
    elif len(where_filters) > 1:
        where_clause = {"$and": where_filters}
    else:
        where_clause = None

    # ------------------------------------------------
    # SOFT CONSTRAINT → SEMANTIC QUERY
    # ------------------------------------------------

    soft_intent = parsed["soft_constraints"]

    semantic_phrases = [
        v["matched_phrase"]
        for v in soft_intent.values()
        if "matched_phrase" in v
    ]

    # Fallback if no soft intent
    if semantic_phrases:
        semantic_query = " ".join(semantic_phrases)
    else:
        semantic_query = "movie"

    query_embedding = embed_text(semantic_query)

    # ------------------------------------------------
    # HARD CONSTRAINTS → WHERE_DOCUMENT (Actors)
    # ------------------------------------------------
    
    where_document = None
    actors = parsed["hard_constraints"]["actors"]
    if actors:
        # Relaxed token-based matching (AND logic across tokens)
        # e.g. "Tom Cruise" -> matches "Tom" AND "Cruise" (any order)
        # This prevents "Tom Hardy" from matching "Tom Cruise" while allowing "Cruise, Tom"
        actor_tokens = actors[0].split()
        if len(actor_tokens) == 1:
             where_document = {"$contains": actor_tokens[0]}
        else:
             where_document = {
                 "$and": [{"$contains": token} for token in actor_tokens]
             }

    # ------------------------------------------------
    # VECTOR RETRIEVAL
    # ------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=RETRIEVAL_LIMIT,
        where=where_clause,
        where_document=where_document,
        include=["embeddings", "metadatas", "documents"]
    )

    if not results["ids"][0]:
        return []

    embeddings = np.array(results["embeddings"][0])
    ids = results["ids"][0]
    metas = results["metadatas"][0]
    # Use title from metadata (cleaner display)
    titles = [m["title"] for m in metas]

    # ------------------------------------------------
    # LOCAL CLUSTERING (DIVERSITY)
    # ------------------------------------------------

    local_k = min(LOCAL_CLUSTERS, len(embeddings))
    clusterer = AgglomerativeClustering(n_clusters=local_k)
    labels = clusterer.fit_predict(embeddings)

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(i)

    sorted_clusters = sorted(
        clusters.values(),
        key=len,
        reverse=True
    )

    # ------------------------------------------------
    # FINAL SELECTION
    # ------------------------------------------------

    final = []
    for cluster in sorted_clusters:
        for idx in cluster[:RETURN_PER_CLUSTER]:
            final.append({
                "tmdb_id": ids[idx],
                "title": titles[idx],
                "metadata": metas[idx]
            })

    return final


# ------------------ DEMO ------------------

if __name__ == "__main__":
    query = "emotional Tom Cruise movies"
    results = retrieve_candidates(query)

    print("\nRetrieved Candidates:\n")
    for r in results:
        print(f"{r['tmdb_id']} | {r['title']}")
