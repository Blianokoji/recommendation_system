"""
Query JSON Builder
------------------
Converts raw user queries into structured JSON
without using an LLM.

This module is intentionally conservative and
acts as a safe precursor to SLM-based parsing.
"""

import re
from typing import Dict


# Minimal lookup tables (expand later or replace with SLM)
KNOWN_GENRES = {
    "comedy", "drama", "thriller", "action", "romance",
    "horror", "sci-fi", "fantasy", "crime"
}

KNOWN_ACTORS = {
    "tom cruise",
    "brad pitt",
    "leonardo dicaprio",
    "meryl streep"
}


def build_query_json(query: str) -> Dict:
    q = query.lower()

    json_out = {
        "entities": {
            "actors": [],
            "directors": [],
            "franchises": []
        },
        "constraints": {
            "genres": [],
            "year_range": {
                "from": None,
                "to": None
            },
            "language": None,
            "adult": False
        },
        "semantic_intent": "",
        "confidence": {
            "entities_explicit": False,
            "constraints_explicit": False
        }
    }

    # ---------- ACTOR EXTRACTION ----------
    for actor in KNOWN_ACTORS:
        if actor in q:
            json_out["entities"]["actors"].append(actor.title())
            q = q.replace(actor, "")
            json_out["confidence"]["entities_explicit"] = True

    # ---------- GENRE EXTRACTION ----------
    for genre in KNOWN_GENRES:
        if genre in q:
            json_out["constraints"]["genres"].append(genre.title())
            q = q.replace(genre, "")
            json_out["confidence"]["constraints_explicit"] = True

    # ---------- YEAR EXTRACTION ----------
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", q)
    if len(years) == 1:
        json_out["constraints"]["year_range"]["from"] = int(years[0])
        json_out["constraints"]["year_range"]["to"] = int(years[0])
        q = q.replace(years[0], "")
    elif len(years) >= 2:
        json_out["constraints"]["year_range"]["from"] = int(min(years))
        json_out["constraints"]["year_range"]["to"] = int(max(years))
        for y in years:
            q = q.replace(y, "")

    # ---------- ADULT FLAG ----------
    if "18+" in query or "adult" in q:
        json_out["constraints"]["adult"] = True

    # ---------- SEMANTIC INTENT ----------
    cleaned = re.sub(r"\s+", " ", q).strip()
    json_out["semantic_intent"] = cleaned if cleaned else query

    return json_out
