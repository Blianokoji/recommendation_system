"""
Candidate Retrieval Module
--------------------------
Uses structured query intent to perform:

1. Hard constraint filtering (adult safety, year range, actors)
2. Centroid gate (actor-based hard filter when no actor is named)
3. Semantic retrieval using dominant soft intent signals
4. Local clustering for diversity
5. Fuzzy-aware weighted scoring (semantic + cluster + temporal + actor)
6. Explainability payload

Weights are loaded from models/optimal_weights.json if available.
NO keyword OR behavior
NO identity leakage into embeddings
Deterministic, safe, demo-ready
"""

import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import List, Dict

from embeddings.embedding_singleton import EmbeddingModelSingleton
from sklearn.cluster import AgglomerativeClustering

from vector_store.chroma_client import get_chroma_collection
from query_parser.parse_query import parse_query
from retrieval.centroid_gate import get_relevant_actor_centroids

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

RETRIEVAL_LIMIT = 80
LOCAL_CLUSTERS = 4
RETURN_PER_CLUSTER = 5

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ------------------------------------------------
# WEIGHT LOADING (GA-optimized or defaults)
# ------------------------------------------------

BASE_DIR = os.getenv("DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_FILE = os.path.join(BASE_DIR, "models", "optimal_weights.json")

def _load_weights():
    """Load weights from GA-optimized JSON, or use sensible defaults."""
    defaults = {
        "W_semantic": 0.55,
        "W_cluster": 0.20,
        "W_temporal": 0.15,
        "W_actor": 0.10
    }
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, "r") as f:
                loaded = json.load(f)
            # Validate all keys exist
            for key in defaults:
                if key not in loaded:
                    print(f"[WARN] Missing weight '{key}' in {WEIGHTS_FILE}, using defaults")
                    return defaults
            print(f"[INFO] Loaded GA-optimized weights from {WEIGHTS_FILE}")
            return loaded
        except Exception as e:
            print(f"[WARN] Failed to load weights: {e}, using defaults")
            return defaults
    return defaults

_weights = _load_weights()
W_Q = _weights["W_semantic"]
W_C = _weights["W_cluster"]
W_T = _weights["W_temporal"]
W_A = _weights["W_actor"]

# ------------------------------------------------
# MODEL
# ------------------------------------------------

_embedding_model = EmbeddingModelSingleton.get_model(EMBEDDING_MODEL_NAME)


def embed_text(text: str) -> np.ndarray:
    return _embedding_model.encode(text, normalize_embeddings=True)

# ------------------------------------------------
# WEIGHTED SCORING (4-component fuzzy-aware)
# ------------------------------------------------

def compute_weighted_scores(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    cluster_labels: np.ndarray,
    temporal_scores: np.ndarray,
    actor_scores: np.ndarray,
    weight_q: float = W_Q,
    weight_c: float = W_C,
    weight_t: float = W_T,
    weight_a: float = W_A
) -> List[float]:
    """
    Score(m) = w_q * Q(m) + w_c * C(m) + w_t * T(m) + w_a * A(m)

    Q(m): cosine similarity with semantic query
    C(m): dominant local cluster alignment
    T(m): temporal fuzzy membership (0-1)
    A(m): actor fuzzy membership (0-1)
    """
    semantic_scores = embeddings @ query_embedding

    dominant_cluster = np.bincount(cluster_labels).argmax()
    cluster_scores = np.array([
        1.0 if label == dominant_cluster else 0.0
        for label in cluster_labels
    ])

    final_scores = (
        weight_q * semantic_scores +
        weight_c * cluster_scores +
        weight_t * temporal_scores +
        weight_a * actor_scores
    )
    return final_scores.tolist()

# ------------------------------------------------
# BUILD where_document FOR ACTORS
# ------------------------------------------------

def _build_actor_document_filter(actors: List[str], actor_confidence: Dict[str, float]):
    """
    Build a ChromaDB where_document filter that requires ALL tokens
    of ALL named actors to appear in the document (title + cast).

    Only applies hard filters for actors with confidence == 1.0.
    Fuzzy-matched actors (confidence < 1.0) are scored post-retrieval.
    """
    # Assume any actor parsed with > 0.80 confidence (which includes autocorrects) is a hard constraint
    hard_actors = [a for a in actors if actor_confidence.get(a, 1.0) >= 0.80]

    if not hard_actors:
        return None

    all_tokens = []
    for actor in hard_actors:
        all_tokens.extend(actor.split())

    # Deduplicate tokens while preserving order
    seen = set()
    unique_tokens = []
    for t in all_tokens:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique_tokens.append(t)

    if len(unique_tokens) == 1:
        return {"$contains": unique_tokens[0]}

    return {"$and": [{"$contains": t} for t in unique_tokens]}

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
    # EXTRACT FUZZY CONSTRAINTS
    # ------------------------------------------------

    fuzzy_constraints = parsed.get("fuzzy_constraints", {})
    temporal_fuzzy = fuzzy_constraints.get("temporal")
    fuzzy_actors = fuzzy_constraints.get("actors", {})

    # ------------------------------------------------
    # HARD CONSTRAINTS -> METADATA FILTERS (where)
    # ------------------------------------------------

    where_filters = []

    # Adult safety
    allow_adult = parsed["filters"]["allow_adult"]
    if allow_adult_override is not None:
        allow_adult = allow_adult_override
    if not allow_adult:
        where_filters.append({"adult": {"$eq": False}})

    # Year range — only if NO fuzzy temporal is active
    year = parsed["hard_constraints"].get("year", {})
    year_from = year.get("year_from")
    year_to = year.get("year_to")

    if not temporal_fuzzy:
        if year_from is not None:
            where_filters.append({"release_year": {"$gte": year_from}})
        if year_to is not None:
            where_filters.append({"release_year": {"$lte": year_to}})

    # Combine where filters
    if len(where_filters) == 0:
        where_clause = None
    elif len(where_filters) == 1:
        where_clause = where_filters[0]
    else:
        where_clause = {"$and": where_filters}

    # ------------------------------------------------
    # HARD CONSTRAINTS -> DOCUMENT FILTER (actors)
    # ------------------------------------------------

    actors = parsed["hard_constraints"]["actors"]
    actor_confidence = parsed["hard_constraints"].get("actor_confidence", {})
    where_document = _build_actor_document_filter(actors, actor_confidence)

    # ------------------------------------------------
    # CENTROID GATE — hard filter when no actor named
    # ------------------------------------------------

    centroid_actors_used = []
    if not actors:
        centroid_hits = get_relevant_actor_centroids(
            parsed["original_query"],
            top_k=2,
            threshold=0.65
        )
        if centroid_hits:
            centroid_actors_used = [c["actor"] for c in centroid_hits]
            # Hard filter: require any of the centroid actors in the document
            # Use the most confident centroid actor's tokens
            top_centroid_actor = centroid_hits[0]["actor"]
            tokens = top_centroid_actor.split()
            if len(tokens) == 1:
                where_document = {"$contains": tokens[0]}
            else:
                where_document = {"$and": [{"$contains": t} for t in tokens]}

    # ------------------------------------------------
    # SOFT CONSTRAINTS -> DOMINANT SEMANTIC QUERY
    # ------------------------------------------------

    soft_intent = parsed["soft_constraints"]

    max_conf = max(
        (v.get("confidence", 0.0) for v in soft_intent.values()),
        default=0.0
    )

    semantic_embeddings = [embed_text(parsed["original_query"])]
    used_semantic_phrases = [parsed["original_query"]]

    for constraint in soft_intent.values():
        phrase = constraint.get("matched_phrase")
        conf = constraint.get("confidence", 0.0)
        if phrase and conf >= 0.45 and conf >= 0.85 * max_conf:
            semantic_embeddings.append(embed_text(phrase))
            used_semantic_phrases.append(phrase)

    # Blend and re-normalize (mean of unit vectors is not unit length)
    query_embedding = np.mean(semantic_embeddings, axis=0)
    query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

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
    # COMPUTE FUZZY TEMPORAL SCORES
    # ------------------------------------------------

    n = len(embeddings)

    if temporal_fuzzy and "mu_func" in temporal_fuzzy:
        mu_func = temporal_fuzzy["mu_func"]
        temporal_membership = np.array([
            mu_func(m.get("release_year", 2000)) for m in metas
        ])
    else:
        # No fuzzy temporal — neutral (all 1.0 so it doesn't affect scoring)
        temporal_membership = np.ones(n)

    # ------------------------------------------------
    # COMPUTE FUZZY ACTOR SCORES
    # ------------------------------------------------

    if fuzzy_actors:
        # For fuzzy-matched actors, score each movie by whether the actor
        # name tokens appear in its cast metadata
        actor_membership = np.ones(n)
        for actor_name, conf in fuzzy_actors.items():
            tokens = [t.lower() for t in actor_name.split()]
            for i, m in enumerate(metas):
                cast_str = (m.get("actor", "") or "").lower()
                if all(t in cast_str for t in tokens):
                    actor_membership[i] = max(actor_membership[i], conf)
                else:
                    actor_membership[i] *= 0.3  # penalize non-matching
    else:
        actor_membership = np.ones(n)

    # ------------------------------------------------
    # LOCAL CLUSTERING (with safety guard)
    # ------------------------------------------------

    if n < 2:
        # Can't cluster a single sample — skip, use raw semantic scores
        labels = np.zeros(n, dtype=int)
        active_wq, active_wc = W_Q + W_C, 0.0
    else:
        local_k = min(LOCAL_CLUSTERS, n)
        clusterer = AgglomerativeClustering(n_clusters=local_k)
        labels = clusterer.fit_predict(embeddings)

        # Smooth weight: cluster weight scales linearly with confidence
        if len(used_semantic_phrases) == 1:
            active_wq, active_wc = W_Q + W_C * 0.8, W_C * 0.2
        else:
            active_wc = W_C * max_conf
            active_wq = W_Q + W_C * (1 - max_conf)

    # Determine temporal and actor weights
    active_wt = W_T if temporal_fuzzy else 0.0
    active_wa = W_A if fuzzy_actors else 0.0

    # Redistribute unused fuzzy weight to semantic
    unused_fuzzy_weight = 0.0
    if not temporal_fuzzy:
        unused_fuzzy_weight += W_T
    if not fuzzy_actors:
        unused_fuzzy_weight += W_A
    active_wq += unused_fuzzy_weight

    scores = compute_weighted_scores(
        embeddings=embeddings,
        query_embedding=query_embedding,
        cluster_labels=labels,
        temporal_scores=temporal_membership,
        actor_scores=actor_membership,
        weight_q=active_wq,
        weight_c=active_wc,
        weight_t=active_wt,
        weight_a=active_wa
    )

    ranked_indices = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)

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

    # Post-score actor enforcement — only for hard constraints
    hard_actors = [a for a in actors if actor_confidence.get(a, 1.0) >= 0.80]
    if hard_actors:
        all_tokens = []
        for actor in hard_actors:
            all_tokens.extend([t.lower() for t in actor.split()])

        final = [
            m for m in final
            if all(
                t in (m["metadata"].get("actor", "") or "").lower()
                for t in all_tokens
            )
        ]

    # ------------------------------------------------
    # EXPLAINABILITY PAYLOAD
    # ------------------------------------------------

    for m in final:
        m["explanation"] = {
            "actor_constraint": actors if actors else None,
            "actor_confidence": actor_confidence if actors else None,
            "year_constraint": year if (year_from is not None or year_to is not None) else None,
            "temporal_fuzzy": {
                "label": temporal_fuzzy["label"],
                "matched_word": temporal_fuzzy["matched_word"],
                "mu_value": round(temporal_membership[ranked_indices[0]], 3) if len(ranked_indices) > 0 else None
            } if temporal_fuzzy else None,
            "centroid_actors_applied": centroid_actors_used if centroid_actors_used else None,
            "semantic_components": used_semantic_phrases,
            "soft_constraints": soft_intent,
            "filters_applied": where_clause,
            "weighted_score": m["score"],
            "score_breakdown": {
                "semantic_weight": round(active_wq, 3),
                "cluster_weight": round(active_wc, 3),
                "temporal_weight": round(active_wt, 3),
                "actor_weight": round(active_wa, 3),
                "note": "Fuzzy constraints applied as soft score multipliers"
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
    queries = [
        "emotional Tom Cruise movies",
        "Tom Hanks and Meg Ryan movies",
        "golf movies",
        "happy movies from the 90s",
        "dark sci fi films between 2010 and 2020",
        "latest action movies",
        "classic comedies",
        "old brad pitt movies",
    ]
    for q in queries:
        print(f"\n{'='*60}\nQUERY: {q}")
        response = retrieve_candidates(q)
        print(f"Intent confidence: {response.get('intent_confidence')}")
        print(f"Results ({len(response['results'])}):") 
        for r in response["results"][:5]:
            exp = r["explanation"]
            print(f"  {r['title']}  [{r['score']}]")
            if exp.get("year_constraint"):
                print(f"    year: {exp['year_constraint']}")
            if exp.get("temporal_fuzzy"):
                print(f"    temporal: {exp['temporal_fuzzy']}")
            if exp.get("centroid_actors_applied"):
                print(f"    centroid actors: {exp['centroid_actors_applied']}")
