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
import re
import pandas as pd
from difflib import get_close_matches
from typing import Dict, List

import numpy as np
from embeddings.embedding_singleton import EmbeddingModelSingleton
from .semantic_axes import SEMANTIC_AXES

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.getenv("DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ACTOR_STATS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_stats", "actor_stats.csv"
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

_embedding_model = EmbeddingModelSingleton.get_model("all-MiniLM-L6-v2")


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
    # HARD CONSTRAINTS (ACTORS) WITH FUZZY FALLBACK
    # --------------------------------------------------------

    actors: List[str] = []
    temp_q = q # Use a temporary query string for actor replacement

    # 1. Exact substring match
    found_actors_lower = set()
    for actor in KNOWN_ACTORS:
        if actor.lower() in temp_q:
            found_actors_lower.add(actor.lower())
            actors.append(actor) # Append original cased actor
            # Remove actor mention to prevent semantic pollution
            temp_q = temp_q.replace(actor.lower(), " ")

    # 2. Fuzzy match if no exact match found
    if not actors:
        # Extract word n-grams (1, 2, 3 words) from query to check against actor names
        words = re.findall(r'\b\w+\b', temp_q)
        ngrams = []
        for i in range(len(words)):
            ngrams.append(words[i])
            if i < len(words) - 1:
                ngrams.append(words[i] + " " + words[i+1])
            if i < len(words) - 2:
                ngrams.append(words[i] + " " + words[i+1] + " " + words[i+2])

        # Optimize by caching lowercased mapping
        known_lower_map = {a.lower(): a for a in KNOWN_ACTORS}

        for ngram in ngrams:
            matches = get_close_matches(ngram, list(known_lower_map.keys()), n=1, cutoff=0.85)
            if matches and matches[0] not in found_actors_lower: # Avoid re-adding already found actors
                actors.append(known_lower_map[matches[0]])
                found_actors_lower.add(matches[0])
                # Remove actor mention to prevent semantic pollution
                temp_q = temp_q.replace(ngram, " ") # Replace the ngram that matched

    # Deduplicate and title-case actors
    actors = sorted(list(set([a.title() for a in actors])))

    # Update the main query string 'q' with the cleaned version
    q = " ".join(temp_q.split())

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
