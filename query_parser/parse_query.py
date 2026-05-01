"""
Query Parsing Module
--------------------
Converts raw user queries into structured JSON with
hard (deterministic), soft (semantic), and fuzzy constraints.

Design principles:
- Intent classification: semantic centroid gate (not keywords)
- Confidence: real cosine similarity, not a hand-tuned heuristic
- Identity entities (actors) are enforced as hard constraints
- Year range is enforced as a hard constraint
- Qualitative temporal words (latest, old, classic) use fuzzy membership
- Abstract intent (emotion, tone) is inferred via semantic axes
- Genre is treated as an inferred signal, not a constraint
- No keyword hardcoding for abstract intent
- Output is retrieval- and reasoning-ready JSON
"""

import os
import re
import pandas as pd
from difflib import get_close_matches, SequenceMatcher
from typing import Dict, List, Optional, Tuple

import numpy as np
from embeddings.embedding_singleton import EmbeddingModelSingleton
from .intent_gate import classify_intent
from .semantic_infer import infer_soft_intent
from .fuzzy_temporal import extract_temporal_fuzzy, strip_temporal_qualitative_tokens

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.getenv("DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACTOR_STATS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "centroids", "actor_stats.csv"
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
    # Longest names first to avoid partial matches (e.g. "Will Smith" before "Will")
    actors.sort(key=len, reverse=True)
    return actors


KNOWN_ACTORS = load_actor_vocabulary(ACTOR_STATS_CSV)

# ============================================================
# EMBEDDING MODEL (SINGLETON)
# ============================================================

_embedding_model = EmbeddingModelSingleton.get_model("all-MiniLM-L6-v2")

# ============================================================
# YEAR CONSTRAINT EXTRACTION
# ============================================================

# Decade mappings: two-digit form (e.g. "90") → century
_DECADE_CENTURY = {d: 1900 for d in range(30, 100, 10)}   # 30s–90s → 1900s
_DECADE_CENTURY.update({d: 2000 for d in range(0, 30, 10)})  # 00s–20s → 2000s


def _decade_to_range(raw: str) -> Tuple[int, int]:
    """Convert a matched decade string ('90', '1990', '2000') to (year_from, year_to)."""
    if len(raw) == 2:
        d = int(raw)
        century = _DECADE_CENTURY.get(d, 1900)
        start = century + d
    else:
        start = int(raw)
    return start, start + 9


def extract_year_constraint(q: str) -> Dict:
    """
    Extracts year constraint from query string.

    Supported patterns:
        - Decade:  90s, 80s, 2000s, 2010s, 2020s
        - Range:   2015-2020, 2015–2020, between 2014 and 2020,
                   from 2014 to 2020, from 2014 through 2020
        - After:   after 2010, since 2010, from 2010
        - Before:  before 2000, until 2000, prior to 2000
        - Single:  2019 movies, movies from 2019, in 2019

    Returns:
        {"year_from": int|None, "year_to": int|None}
    """

    # --- Decade: 90s, 80s, 2000s, 2010s ---
    decade_m = re.search(r'\b((?:19|20)\d{2}|[3-9]\d)s\b', q)
    if decade_m:
        yr_from, yr_to = _decade_to_range(decade_m.group(1))
        return {"year_from": yr_from, "year_to": yr_to}

    # --- Range: between X and Y ---
    between_m = re.search(r'\bbetween\s+(\d{4})\s+and\s+(\d{4})\b', q)
    if between_m:
        return {"year_from": int(between_m.group(1)), "year_to": int(between_m.group(2))}

    # --- Range: from X to/through/- Y ---
    from_to_m = re.search(
        r'\bfrom\s+(\d{4})\s+(?:to|through)\s+(\d{4})\b'
        r'|(\d{4})\s*[–\-]\s*(\d{4})',
        q
    )
    if from_to_m:
        g = from_to_m.groups()
        if g[0] and g[1]:
            return {"year_from": int(g[0]), "year_to": int(g[1])}
        if g[2] and g[3]:
            return {"year_from": int(g[2]), "year_to": int(g[3])}

    # --- After / since X ---
    after_m = re.search(r'\b(?:after|since)\s+(\d{4})\b', q)
    if after_m:
        return {"year_from": int(after_m.group(1)), "year_to": None}

    # --- Before / until X ---
    before_m = re.search(r'\b(?:before|until|prior\s+to)\s+(\d{4})\b', q)
    if before_m:
        return {"year_from": None, "year_to": int(before_m.group(1))}

    # --- Single year: "movies from 2019" / "2019 movies" / "in 2019" ---
    single_m = re.search(
        r'(?:^|\bfrom\s+|\bin\s+|\bof\s+|\byear\s+)(\d{4})(?:\s|$)',
        q
    )
    if single_m:
        yr = int(single_m.group(1))
        if 1900 <= yr <= 2030:
            return {"year_from": yr, "year_to": yr}

    return {"year_from": None, "year_to": None}


def _strip_year_tokens(q: str, year_constraint: dict) -> str:
    """Remove year-related tokens from query string to prevent semantic pollution."""
    # Remove decade patterns
    q = re.sub(r'\b((?:19|20)\d{2}|[3-9]\d)s\b', ' ', q)
    # Remove 4-digit years
    q = re.sub(r'\b\d{4}\b', ' ', q)
    # Remove connecting words left behind
    q = re.sub(r'\b(between|from|to|through|after|since|before|until|prior\s+to|in|of)\b', ' ', q)
    return ' '.join(q.split())

# ============================================================
# ADULT CONTENT FILTER
# ============================================================

_FAMILY_WORDS = {"kids", "children", "family", "kid", "toddler", "baby", "child", "babies"}

# Phrases that contain family words but should NOT block adult content
_FAMILY_NEGATION_PATTERNS = [
    r"dysfunctional\s+family",
    r"dark\s+family",
    r"family\s+tragedy",
    r"family\s+crime",
    r"broken\s+family",
]


def _check_allow_adult(q: str) -> bool:
    # First check negation patterns — these override the block
    for pattern in _FAMILY_NEGATION_PATTERNS:
        if re.search(pattern, q):
            return True  # allow adult content

    # Check for family-safe words
    words = set(re.findall(r'\b\w+\b', q))
    if words & _FAMILY_WORDS:
        return False

    return True

# ============================================================
# JARO-WINKLER SIMILARITY (pure Python fallback)
# ============================================================

def _jaro_winkler(s1: str, s2: str) -> float:
    """
    Jaro-Winkler similarity between two strings.
    Returns a value in [0.0, 1.0] where 1.0 is an exact match.
    """
    s1_len = len(s1)
    s2_len = len(s2)
    if s1_len == 0 and s2_len == 0:
        return 1.0
    if s1_len == 0 or s2_len == 0:
        return 0.0

    match_distance = max(s1_len, s2_len) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * s1_len
    s2_matches = [False] * s2_len

    matches = 0
    transpositions = 0

    for i in range(s1_len):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, s2_len)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(s1_len):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / s1_len + matches / s2_len +
            (matches - transpositions / 2) / matches) / 3

    # Winkler bonus for common prefix (up to 4 chars)
    prefix = 0
    for i in range(min(4, s1_len, s2_len)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1 - jaro)

# ============================================================
# ACTOR EXTRACTION (EXACT + FUZZY with Jaro-Winkler confidence)
# ============================================================

def _extract_actors(q: str) -> Tuple[List[Dict], str]:
    """
    Returns (actors_list, cleaned_query_without_actor_tokens).
    Each actor entry is {"name": str, "confidence": float}.
    Exact matches get confidence=1.0; fuzzy matches get Jaro-Winkler score.
    """
    temp_q = q
    actors: List[Dict] = []
    found_lower: set = set()

    # 1. Exact substring match (longest names first prevents partial shadowing)
    for actor in KNOWN_ACTORS:
        if actor.lower() in temp_q:
            found_lower.add(actor.lower())
            actors.append({"name": actor, "confidence": 1.0})
            temp_q = temp_q.replace(actor.lower(), " ")

    # 2. Fuzzy fallback — only if exact match found nothing
    if not actors:
        words = re.findall(r'\b\w+\b', temp_q)
        ngrams = []
        for i in range(len(words)):
            ngrams.append(words[i])
            if i < len(words) - 1:
                ngrams.append(words[i] + " " + words[i + 1])
            if i < len(words) - 2:
                ngrams.append(words[i] + " " + words[i + 1] + " " + words[i + 2])

        known_lower_map = {a.lower(): a for a in KNOWN_ACTORS}

        for ngram in ngrams:
            matches = get_close_matches(ngram, list(known_lower_map.keys()), n=1, cutoff=0.80)
            if matches and matches[0] not in found_lower:
                jw_score = _jaro_winkler(ngram.lower(), matches[0].lower())
                if jw_score >= 0.82:
                    actors.append({"name": known_lower_map[matches[0]], "confidence": round(jw_score, 3)})
                    found_lower.add(matches[0])
                    temp_q = temp_q.replace(ngram, " ")

    # Deduplicate and title-case
    seen = set()
    unique_actors = []
    for a in actors:
        name_tc = a["name"].title()
        if name_tc not in seen:
            seen.add(name_tc)
            unique_actors.append({"name": name_tc, "confidence": a["confidence"]})
    unique_actors.sort(key=lambda x: x["name"])

    cleaned_q = " ".join(temp_q.split())
    return unique_actors, cleaned_q

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
        - hard_constraints  (actors, year)
        - soft_constraints  (emotion, tone — steer the embedding)
        - inferred_signals  (genre, others — metadata signals only)
        - filters           (allow_adult)
        - confidence        (real cosine similarity from intent gate)
        - original_query
    """
    original_query = query
    q = query.lower().strip()

    # --------------------------------------------------------
    # INTENT CLASSIFICATION (semantic centroid gate)
    # --------------------------------------------------------
    intent_result = classify_intent(q)

    if intent_result["intent_type"] != "movie_search":
        return {
            "intent_type": "invalid",
            "reason": "Query does not appear to be movie-related",
            "confidence": intent_result["confidence"],
            "original_query": original_query,
        }

    confidence = intent_result["confidence"]

    # --------------------------------------------------------
    # FUZZY TEMPORAL (must run BEFORE hard year extraction)
    # --------------------------------------------------------
    temporal_fuzzy = extract_temporal_fuzzy(q)
    if temporal_fuzzy:
        q = strip_temporal_qualitative_tokens(q)

    # --------------------------------------------------------
    # YEAR CONSTRAINT (extract + strip from query)
    # --------------------------------------------------------
    year_constraint = extract_year_constraint(q)
    q = _strip_year_tokens(q, year_constraint)

    # If we found a fuzzy temporal AND no explicit numeric year,
    # blank out year so retrieval uses the fuzzy membership instead.
    if temporal_fuzzy and year_constraint.get("year_from") is None and year_constraint.get("year_to") is None:
        fuzzy_constraints_temporal = temporal_fuzzy
    else:
        # Explicit numeric year takes priority over fuzzy
        fuzzy_constraints_temporal = None

    # --------------------------------------------------------
    # HARD CONSTRAINTS — ACTORS (exact -> fuzzy with confidence)
    # --------------------------------------------------------
    actor_matches, q = _extract_actors(q)
    # Separate into names list and confidence map
    actors = [a["name"] for a in actor_matches]
    actor_confidence = {a["name"]: a["confidence"] for a in actor_matches}

    # --------------------------------------------------------
    # SOFT + INFERRED INTENT (semantic, on cleaned query)
    # --------------------------------------------------------
    semantic_q = re.sub(r'\b(movies?|films?|shows?)\b', ' ', q).strip()
    soft_result = infer_soft_intent(semantic_q)
    soft_constraints = soft_result["soft_constraints"]
    inferred_signals = soft_result["inferred_signals"]

    # --------------------------------------------------------
    # SAFETY FILTER
    # --------------------------------------------------------
    allow_adult = _check_allow_adult(q)

    # --------------------------------------------------------
    # BUILD FUZZY CONSTRAINTS PAYLOAD
    # --------------------------------------------------------
    fuzzy_constraints = {}
    if fuzzy_constraints_temporal:
        fuzzy_constraints["temporal"] = {
            "label": fuzzy_constraints_temporal["label"],
            "matched_word": fuzzy_constraints_temporal["matched_word"],
            "mu_func": fuzzy_constraints_temporal["mu_func"]
        }
    # Flag actors with sub-1.0 confidence as fuzzy
    fuzzy_actors = {name: conf for name, conf in actor_confidence.items() if conf < 1.0}
    if fuzzy_actors:
        fuzzy_constraints["actors"] = fuzzy_actors

    # --------------------------------------------------------
    # FINAL STRUCTURED OUTPUT
    # --------------------------------------------------------
    return {
        "intent_type": "movie_search",
        "hard_constraints": {
            "actors": actors,
            "actor_confidence": actor_confidence,
            "year": year_constraint,
        },
        "soft_constraints": soft_constraints,
        "inferred_signals": inferred_signals,
        "fuzzy_constraints": fuzzy_constraints,
        "filters": {
            "allow_adult": allow_adult,
        },
        "confidence": confidence,
        "original_query": original_query,
    }


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    import json

    demo_queries = [
        "emotional Tom Cruise movies",
        "Tom Hanks and Meg Ryan movies",
        "dark psychological sci fi films from the 90s",
        "movies for my kid",
        "dysfunctional family drama",
        "movies between 2014 and 2020",
        "show me something funny",
        "happy movies after 2015",
        "help me write an email",          # should be INVALID
        "good thriller to watch tonight",  # no movie/film keyword — should now PASS
    ]

    for q in demo_queries:
        print(f"\nQUERY: {q}")
        result = parse_query(q)
        print(json.dumps(result, indent=2))
