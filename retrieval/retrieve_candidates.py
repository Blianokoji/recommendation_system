"""
Candidate Retrieval Module (Final)
---------------------------------
Uses structured query intent to perform:

1. Hard constraint filtering (actor, safety)
2. Semantic retrieval using dominant soft intent signals
3. Local clustering for diversity
4. Weighted scoring (paper-aligned)
5. Explainability payload

NO keyword OR behavior
NO identity leakage into embeddings
Deterministic, safe, demo-ready
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import List, Dict

from embeddings.embedding_singleton import EmbeddingModelSingleton
from sklearn.cluster import AgglomerativeClustering

from vector_store.chroma_client import get_chroma_collection
from query_parser.parse_query import parse_query

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

RETRIEVAL_LIMIT = 80
LOCAL_CLUSTERS = 4
RETURN_PER_CLUSTER = 5

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Weights as per paper
W_Q = 0.65   # semantic relevance
W_C = 0.35   # cluster coherence

# ------------------------------------------------
# MODEL
# ------------------------------------------------

_embedding_model = EmbeddingModelSingleton.get_model(EMBEDDING_MODEL_NAME)

def embed_text(text: str) -> np.ndarray:
    return _embedding_model.encode(text, normalize_embeddings=True)

# ------------------------------------------------
# WEIGHTED SCORING
# ------------------------------------------------

def compute_weighted_scores(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    cluster_labels: np.ndarray,
    weight_q: float = W_Q,
    weight_c: float = W_C
) -> List[float]:
    """
    Score(m) = w_q * Q(m) + w_c * C(m)

    Q(m): cosine similarity with semantic query
    C(m): dominant local cluster alignment
    """

    semantic_scores = embeddings @ query_embedding

    dominant_cluster = np.bincount(cluster_labels).argmax()
    cluster_scores = np.array([
        1.0 if label == dominant_cluster else 0.0
        for label in cluster_labels
    ])

    final_scores = (
        weight_q * semantic_scores +
        weight_c * cluster_scores
    )

    return final_scores.tolist()

# ------------------------------------------------
# MAIN RETRIEVAL
# ------------------------------------------------

def retrieve_candidates(
    query: str,
    allow_adult_override: bool = None
) -> Dict:

    parsed = parse_query(query)

    # ------------------ INTENT GATE ------------------

    if parsed.get("intent_type") != "movie_search":
        return {
            "intent_passable": False,
            "reason": parsed.get("reason", "Invalid intent"),
            "results": []
        }

    collection = get_chroma_collection()

    # ------------------------------------------------
    # HARD CONSTRAINTS → METADATA FILTER
    # ------------------------------------------------

    where_filters = []

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
    # SOFT CONSTRAINTS → DOMINANT SEMANTIC QUERY
    # ------------------------------------------------

    soft_intent = parsed["soft_constraints"]

    # Determine dominant semantic signal
    max_conf = max(
        (v.get("confidence", 0.0) for v in soft_intent.values()),
        default=0.0
    )

    semantic_embeddings = [embed_text(parsed["original_query"])]
    used_semantic_phrases = [parsed["original_query"]]

    for constraint in soft_intent.values():
        phrase = constraint.get("matched_phrase")
        conf = constraint.get("confidence", 0.0)

        # Only include dominant or near-dominant signals
        if (
            phrase and
            conf >= 0.45 and
            conf >= 0.85 * max_conf
        ):
            semantic_embeddings.append(embed_text(phrase))
            used_semantic_phrases.append(phrase)

    # Blend original query with the dominant semantic signals smoothly
    query_embedding = np.mean(semantic_embeddings, axis=0)

    # ------------------------------------------------
    # HARD CONSTRAINTS → DOCUMENT FILTER (ACTORS)
    # ------------------------------------------------

    where_document = None
    actor_tokens = []
    actors = parsed["hard_constraints"]["actors"]

    if actors:
        actor_tokens = actors[0].split()
        if len(actor_tokens) == 1:
            where_document = {"$contains": actor_tokens[0]}
        else:
            where_document = {
                "$and": [{"$contains": t} for t in actor_tokens]
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
        return {
            "intent_passable": True,
            "intent_confidence": parsed.get("confidence"),
            "parsed_intent": parsed,
            "results": []
        }

    embeddings = np.array(results["embeddings"][0])
    ids = results["ids"][0]
    metas = results["metadatas"][0]
    titles = [m["title"] for m in metas]

    # ------------------------------------------------
    # LOCAL CLUSTERING
    # ------------------------------------------------

    local_k = min(LOCAL_CLUSTERS, len(embeddings))
    clusterer = AgglomerativeClustering(n_clusters=local_k)
    labels = clusterer.fit_predict(embeddings)

    # Dynamic Weights: if no predefined strong emotional anchors were matched,
    # to avoid the dominant local cluster suppressing hyper-specific outlier matches (e.g. "golf")
    # we heavily reduce the cluster weight so the raw semantic similarity determines the winner.
    if len(used_semantic_phrases) == 1 and used_semantic_phrases[0] == parsed["original_query"]:
        active_wq, active_wc = 0.95, 0.05
    else:
        active_wq, active_wc = W_Q, W_C

    scores = compute_weighted_scores(
        embeddings=embeddings,
        query_embedding=query_embedding,
        cluster_labels=labels,
        weight_q=active_wq,
        weight_c=active_wc
    )

    ranked_indices = sorted(
        range(len(ids)),
        key=lambda i: scores[i],
        reverse=True
    )

    # ------------------------------------------------
    # FINAL SELECTION
    # ------------------------------------------------

    final = []
    for idx in ranked_indices[:RETURN_PER_CLUSTER * LOCAL_CLUSTERS]:
        final.append({
            "tmdb_id": ids[idx],
            "title": titles[idx],
            "metadata": metas[idx],
            "score": round(scores[idx], 4)
        })

    # Enforce actor constraint strictly (post-score)
    if actor_tokens:
        final = [
            m for m in final
            if all(
                t.lower() in (m["metadata"].get("actor", "") or "").lower()
                for t in actor_tokens
            )
        ]

    # ------------------------------------------------
    # EXPLAINABILITY PAYLOAD
    # ------------------------------------------------

    for m in final:
        m["explanation"] = {
            "actor_constraint": actors[0] if actors else None,
            "semantic_components": used_semantic_phrases,
            "soft_constraints": soft_intent,
            "filters_applied": where_clause,
            "weighted_score": m["score"],
            "score_breakdown": {
                "semantic_weight": W_Q,
                "cluster_weight": W_C,
                "note": "Hard constraints enforced before scoring"
            }
        }

    return {
        "intent_passable": True,
        "intent_confidence": parsed.get("confidence"),
        "parsed_intent": parsed,
        "results": final
    }

# ------------------------------------------------
# DEMO
# ------------------------------------------------

if __name__ == "__main__":
    query = "emotional Tom Cruise movies"
    response = retrieve_candidates(query)

    print("\nRetrieved Candidates:\n")

    for r in response["results"]:
        print(f"{r['tmdb_id']} | {r['title']}")
        exp = r["explanation"]

        print("  ↳ Explanation:")
        print(f"     Actor Constraint      : {exp['actor_constraint']}")

        if exp["semantic_components"]:
            print("     Semantic Signals Used :")
            for s in exp["semantic_components"]:
                print(f"        - {s}")
        else:
            print("     Semantic Signals Used : None")

        print(f"     Soft Constraints     : {list(exp['soft_constraints'].keys())}")
        print(f"     Filters Applied      : {exp['filters_applied']}")
        print(f"     Final Weighted Score : {exp['weighted_score']}")
        print()
