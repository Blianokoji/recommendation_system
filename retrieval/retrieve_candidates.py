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

# ------------------------------------------------
# WEIGHTED SCORING (PAPER-ALIGNED)
# ------------------------------------------------

def compute_weighted_scores(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    cluster_labels: np.ndarray
) -> List[float]:
    """
    Computes weighted ranking scores using:
    - Semantic similarity (Q)
    - Cluster coherence (C)

    Hard constraints (actors, safety) are already enforced.
    """

    # ---- weights (as per paper) ----
    W_Q = 0.65   # semantic relevance
    W_C = 0.35   # cluster alignment

    # ---- semantic similarity (cosine) ----
    semantic_scores = embeddings @ query_embedding

    # ---- cluster relevance ----
    dominant_cluster = np.bincount(cluster_labels).argmax()
    cluster_scores = np.array([
        1.0 if label == dominant_cluster else 0.0
        for label in cluster_labels
    ])

    # ---- final weighted score ----
    final_scores = (
        W_Q * semantic_scores +
        W_C * cluster_scores
    )

    return final_scores.tolist()

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

    semantic_embeddings = []

    for key, constraint in soft_intent.items():
        phrase = constraint.get("matched_phrase")
        confidence = constraint.get("confidence", 0.0)

        # Ignore weak semantic signals
        if phrase and confidence >= 0.45:
            semantic_embeddings.append(embed_text(phrase))

    # Fallback if no reliable soft intent
    if semantic_embeddings:
        query_embedding = np.mean(semantic_embeddings, axis=0)
    else:
        query_embedding = embed_text("movie")

    # ------------------------------------------------
    # HARD CONSTRAINTS → WHERE_DOCUMENT (Actors)
    # ------------------------------------------------
    
    where_document = None
    actor_tokens = []
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
    scores = compute_weighted_scores(
        embeddings=embeddings,
        query_embedding=query_embedding,
        cluster_labels=labels
    )

    ranked_indices = sorted(
        range(len(ids)),
        key=lambda i: scores[i],
        reverse=True
    )

    final = []
    for idx in ranked_indices[:RETURN_PER_CLUSTER * LOCAL_CLUSTERS]:
        final.append({
            "tmdb_id": ids[idx],
            "title": titles[idx],
            "metadata": metas[idx],
            "score": round(scores[idx], 4)
        })
    # ------------------------------------------------
    # HARD ACTOR PRIORITY ENFORCEMENT
    # ------------------------------------------------

    if actor_tokens:
        final = [
            m for m in final
            if all(
                token.lower() in (m["metadata"].get("actor", "") or "").lower()
                for token in actor_tokens
            )
        ]
    # ------------------------------------------------
    # EXPLAINABILITY PAYLOAD (NON-INTRUSIVE)
    # ------------------------------------------------

    for i, m in enumerate(final):
        m["explanation"] = {
            "actor_constraint": actors[0] if actors else None,

            "semantic_components": [
                v["matched_phrase"]
            for v in soft_intent.values()
            if v.get("matched_phrase") and v.get("confidence", 0.0) >= 0.45
        ],

        "soft_constraints": soft_intent,

        "filters_applied": where_clause,

        # ---- scoring transparency ----
            "weighted_score": round(m["score"], 4),
            "score_breakdown": {
                "semantic_weight": 0.65,
                "cluster_weight": 0.35,
                "note": "Hard constraints enforced before scoring"
            }
        }



    return final


# ------------------ DEMO ------------------

if __name__ == "__main__":
    query = "emotional Tom Cruise movies"
    results = retrieve_candidates(query)

    print("\nRetrieved Candidates:\n")
    for r in results:
        print(f"{r['tmdb_id']} | {r['title']}")

        if "explanation" in r:
            exp = r["explanation"]
            print("  ↳ Explanation:")

            # Actor constraint
            print(f"     Actor Constraint      : {exp.get('actor_constraint')}")

            # Semantic signals actually used
            semantic_components = exp.get("semantic_components", [])
            if semantic_components:
                print("     Semantic Signals Used :")
                for s in semantic_components:
                    print(f"        - {s}")
            else:
                print("     Semantic Signals Used : None")

            # Soft constraints (axes detected)
            soft_keys = list(exp.get("soft_constraints", {}).keys())
            print(f"     Soft Constraints     : {soft_keys if soft_keys else 'None'}")

            # Filters
            print(f"     Filters Applied      : {exp.get('filters_applied')}")

            # Weighted score
            if "weighted_score" in exp:
                print(f"     Final Weighted Score : {exp['weighted_score']}")

            # Optional score breakdown (paper alignment)
            breakdown = exp.get("score_breakdown")
            if breakdown:
                print("     Score Breakdown      :")
                for k, v in breakdown.items():
                    print(f"        - {k.replace('_', ' ').title()} : {v}")

            print()


