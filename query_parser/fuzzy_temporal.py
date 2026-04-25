"""
Fuzzy Temporal Inference
------------------------
Detects qualitative temporal words in queries and maps them
to trapezoidal membership functions over the year axis.

Instead of hard year filters like $gte / $lte, this module
outputs a callable mu(year) -> [0.0, 1.0] that scores how
well a movie's release year fits the user's temporal intent.

All boundaries are computed dynamically from the current year.
"""

import re
from datetime import date
from typing import Optional, Tuple, Callable, Dict

# ============================================================
# QUALITATIVE TEMPORAL VOCABULARY
# ============================================================

_RECENT_WORDS = {"latest", "latest", "new", "newest", "recent", "recently", "contemporary", "modern"}
_OLD_WORDS = {"old", "older", "oldest", "classic", "classics", "vintage", "retro", "throwback"}
_MID_WORDS = {"mid", "middle", "neither"}  # reserved for future use

# ============================================================
# TRAPEZOIDAL MEMBERSHIP FUNCTION
# ============================================================

def _trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Standard trapezoidal membership function.

    Shape:
        0                       0
         \   1.0 plateau    /
          a---b---------c---d

    Returns mu(x) in [0.0, 1.0].
    """
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1.0
    elif c < x < d:
        return (d - x) / (d - c)
    return 0.0


def _build_recent_membership() -> Callable[[int], float]:
    """
    'latest' / 'new' / 'recent':
    Full membership for the last 3 years, linear decay over 5 more years.

    Example (current_year=2026):
        mu(2026) = 1.0   (this year)
        mu(2024) = 1.0   (within 2 years)
        mu(2023) = 1.0   (within 3 years)
        mu(2020) = 0.5   (decaying)
        mu(2018) = 0.0   (outside window)
    """
    now = date.today().year
    # Trapezoid: a=now-7, b=now-3, c=now, d=now+1
    a = now - 7
    b = now - 3
    c = now
    d = now + 1  # future-proof for upcoming releases
    return lambda year: _trapezoidal(float(year), a, b, c, d)


def _build_old_membership() -> Callable[[int], float]:
    """
    'old' / 'classic' / 'vintage':
    Full membership for movies older than 20 years, decay toward 6 years ago.

    Example (current_year=2026):
        mu(1990) = 1.0   (classic)
        mu(2000) = 1.0   (still classic)
        mu(2006) = 1.0   (boundary of full membership)
        mu(2013) = 0.5   (decaying)
        mu(2020) = 0.0   (too recent)
    """
    now = date.today().year
    # Trapezoid: a=1899, b=1900, c=now-20, d=now-6
    a = 1899
    b = 1900
    c = now - 20
    d = now - 6
    return lambda year: _trapezoidal(float(year), a, b, c, d)


# ============================================================
# EXTRACTION
# ============================================================

def extract_temporal_fuzzy(query: str) -> Optional[Dict]:
    """
    Scans the query for qualitative temporal words.

    Returns:
        None if no qualitative temporal word found.
        {
            "label": "recent" | "old",
            "matched_word": str,
            "mu_func": Callable[[int], float]
        }
    """
    words = set(re.findall(r'\b\w+\b', query.lower()))

    # Check recent words
    matched_recent = words & _RECENT_WORDS
    if matched_recent:
        return {
            "label": "recent",
            "matched_word": matched_recent.pop(),
            "mu_func": _build_recent_membership()
        }

    # Check old words
    matched_old = words & _OLD_WORDS
    if matched_old:
        return {
            "label": "old",
            "matched_word": matched_old.pop(),
            "mu_func": _build_old_membership()
        }

    return None


def strip_temporal_qualitative_tokens(query: str) -> str:
    """
    Remove qualitative temporal words from the query to prevent
    them from polluting the semantic embedding.
    """
    all_words = _RECENT_WORDS | _OLD_WORDS | _MID_WORDS
    pattern = r'\b(' + '|'.join(re.escape(w) for w in all_words) + r')\b'
    cleaned = re.sub(pattern, ' ', query, flags=re.IGNORECASE)
    return ' '.join(cleaned.split())
