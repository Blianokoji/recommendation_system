"""
Main API Entry Point
-------------------
Provides safe and reasoned movie recommendation endpoints.

Design:
- /retrieve/core     → Deterministic retrieval (NO SLM)
- /retrieve/reasoned → SLM-augmented reranking (DISABLED — returns 503)
- /retrieve          → Unified, routes via use_slm flag
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional

import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Core retrieval (already working)
from retrieval.retrieve_candidates import retrieve_candidates

# Optional SLM reasoning layer
from slm.reasoner import rerank_with_slm

# -------------------------------------------------
# FASTAPI APP
# -------------------------------------------------

app = FastAPI(
    title="Safe Semantic Movie Recommendation System",
    description="Deterministic + Reasoned Retrieval API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# REQUEST SCHEMA
# -------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    use_slm: Optional[bool] = False


# -------------------------------------------------
# CORE RETRIEVAL (NO SLM, GUARANTEED)
# -------------------------------------------------

@app.post("/retrieve/core")
def retrieve_core(req: QueryRequest):
    """
    Deterministic retrieval.
    Uses:
    - Structured query parsing
    - Hard constraints
    - Semantic retrieval
    - Weighted scoring
    """

    results = retrieve_candidates(req.query)

    return {
        "mode": "deterministic",
        "query": req.query,
        "intent_passable": results.get("intent_passable", False),
        "intent_confidence": results.get("intent_confidence"),
        "parsed_intent": results.get("parsed_intent"),
        "result_count": len(results.get("results", [])),
        "results": results.get("results", [])
    }



# -------------------------------------------------
# REASONED RETRIEVAL (WITH SLM)
# -------------------------------------------------

@app.post("/retrieve/reasoned")
def retrieve_reasoned(req: QueryRequest):
    """
    Reasoned retrieval.
    SLM is ONLY used for reranking already-safe candidates.
    """
    
    # 1. Get deterministic candidates
    deterministic_results = retrieve_candidates(req.query)
    
    # 2. Check if valid
    if not deterministic_results.get("intent_passable"):
        return {
            "mode": "reasoned",
            "query": req.query,
            "intent_passable": False,
            "reason": deterministic_results.get("reason"),
            "results": []
        }
        
    candidates = deterministic_results.get("results", [])
    parsed_intent = deterministic_results.get("parsed_intent", {})
    
    # 3. Pass more candidates for slm to evaluate (take all returned by default limits)
    # SLM will pick the top 5
    reranked = rerank_with_slm(req.query, parsed_intent, candidates, max_results=5)

    return {
        "mode": "reasoned",
        "query": req.query,
        "intent_passable": True,
        "intent_confidence": deterministic_results.get("intent_confidence"),
        "parsed_intent": parsed_intent,
        "result_count": len(reranked),
        "results": reranked
    }


# -------------------------------------------------
# UNIFIED ENDPOINT (DEMO FRIENDLY)
# -------------------------------------------------

@app.post("/retrieve")
def retrieve(req: QueryRequest):
    """
    Unified endpoint.
    Switches between core and SLM-based retrieval.
    """

    if req.use_slm:
        return retrieve_reasoned(req)

    return retrieve_core(req)


# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)
