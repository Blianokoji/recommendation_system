"""
Query Parsing Module
--------------------
Converts raw user queries into structured JSON with
hard (deterministic) and soft (semantic) constraints.

Design principles:
- Identity entities (actors) are enforced as hard constraints
- Abstract intent (emotion, tone) is inferred semantically
- Genre is treated as an inferred signal, not a constraint
- No keyword hardcoding for abstract intent
- Output is retrieval- and reasoning-ready JSON
"""

import os
import pandas as pd
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from .semantic_axes import SEMANTIC_AXES

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ACTOR_STATS_CSV = os.path.join(
    BASE_DIR, "data_stats", "actor_stats.csv"
)

# ============================================================
# LOAD ACTOR VOCABULARY (DATA-DRIVEN)
# ============================================================

def load_actor_vocabulary(csv_path: str) -> List[str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Actor stats file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "actor" not in df.columns:
        raise ValueError("actor_stats.csv must contain 'actor' column")

    actors = (
        df["actor"]
        .dropna()
        .astype(str)
        .str.lower()
        .str.strip()
        .unique()
        .tolist()
    )

    # Longest names first to avoid partial matches
    actors.sort(key=len, reverse=True)
    return actors


KNOWN_ACTORS = load_actor_vocabulary(ACTOR_STATS_CSV)

# ============================================================
# MODEL LOAD (ONCE)
# ============================================================

_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ============================================================
# SOFT INTENT INFERENCE
# ============================================================

def infer_soft_intent(
    text: str,
    threshold: float = 0.35
) -> Dict:
    """
    Infers abstract semantic intent using embedding similarity
    against predefined semantic axes.

    Returns:
        {
          axis_name: {
            "matched_phrase": str,
            "confidence": float
          }
        }
    """
    if not text.strip():
        return {}

    query_emb = _embedding_model.encode(
        text, normalize_embeddings=True
    )

    inferred = {}

    for axis, anchor_phrases in SEMANTIC_AXES.items():
        phrase_embs = _embedding_model.encode(
            anchor_phrases, normalize_embeddings=True
        )

        similarities = [
            _cosine_sim(query_emb, p_emb)
            for p_emb in phrase_embs
        ]

        max_sim = max(similarities)
        if max_sim >= threshold:
            best_idx = similarities.index(max_sim)
            inferred[axis] = {
                "matched_phrase": anchor_phrases[best_idx],
                "confidence": round(max_sim, 3)
            }

    return inferred


# ============================================================
# MAIN QUERY PARSER
# ============================================================

def parse_query(query: str) -> Dict:
    """
    Main entry point.

    Input:
        Raw user query (string)

    Output:
        Structured JSON with:
        - intent_type
        - hard_constraints
        - soft_constraints (emotion, tone)
        - inferred_signals (genre, others)
        - filters
        - confidence
    """

    original_query = query
    q = query.lower().strip()

    # --------------------------------------------------------
    # INTENT CLASSIFICATION (MINIMAL, SAFE)
    # --------------------------------------------------------

    if not any(token in q for token in ["movie", "movies", "film", "films"]):
        return {
            "intent_type": "invalid",
            "reason": "Query does not appear to be movie-related",
            "original_query": original_query
        }

    # --------------------------------------------------------
    # HARD CONSTRAINTS (ACTORS)
    # --------------------------------------------------------

    actors: List[str] = []

    for actor in KNOWN_ACTORS:
        if actor in q:
            actors.append(actor.title())
            # Remove actor mention to prevent semantic pollution
            q = q.replace(actor, " ")

    q = " ".join(q.split())

    # --------------------------------------------------------
    # SOFT + INFERRED INTENT (SEMANTIC)
    # --------------------------------------------------------

    raw_soft_intent = infer_soft_intent(q)

    soft_constraints = {}
    inferred_signals = {}

    for axis, data in raw_soft_intent.items():
        if axis in {"emotion", "tone"}:
            soft_constraints[axis] = data
        else:
            # Genre and other axes are signals, not constraints
            inferred_signals[axis] = data

    # --------------------------------------------------------
    # SAFETY FILTERS
    # --------------------------------------------------------

    allow_adult = not any(
        word in q for word in ["kids", "children", "family"]
    )

    # --------------------------------------------------------
    # CONFIDENCE HEURISTIC
    # --------------------------------------------------------

    confidence = 0.6
    if actors:
        confidence += 0.25
    if soft_constraints:
        confidence += 0.1
    confidence = min(confidence, 1.0)

    # --------------------------------------------------------
    # FINAL STRUCTURED OUTPUT
    # --------------------------------------------------------

    return {
        "intent_type": "movie_search",
        "hard_constraints": {
            "actors": actors
        },
        "soft_constraints": soft_constraints,
        "inferred_signals": inferred_signals,
        "filters": {
            "allow_adult": allow_adult
        },
        "confidence": round(confidence, 2),
        "original_query": original_query
    }


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    demo_queries = [
        "emotional Tom Cruise movies",
        "dark psychological sci fi films",
        "family friendly comedy movies",
        "best movies ever"
    ]

    for q in demo_queries:
        print("\nQUERY:", q)
        print(parse_query(q))
