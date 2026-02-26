"""
Main API Entry Point
-------------------
Provides safe and reasoned movie recommendation endpoints.

Design:
- /retrieve/core     → Deterministic retrieval (NO SLM)
- /retrieve/reasoned → SLM-augmented reranking (OPTIONAL)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional

import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Core retrieval (already working)
from retrieval.retrieve_candidates import retrieve_candidates

# Optional SLM reasoning layer (safe stub for demo)
# from slm.reasoner import rerank_with_slm

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
    # SLM TEMPORARILY DISABLED
    return {
         "error": "SLM endpoint is temporarily disabled.",
         "mode": "disabled"
    }

    # base_results = retrieve_candidates(req.query)

    # if not base_results:
    #     return {
    #         "mode": "reasoned",
    #         "query": req.query,
    #         "result_count": 0,
    #         "results": []
    #     }

    # reranked = rerank_with_slm(
    #     query=req.query,
    #     candidates=base_results
    # )

    # return {
    #     "mode": "reasoned",
    #     "query": req.query,
    #     "result_count": len(reranked),
    #     "results": reranked
    # }


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
    uvicorn.run(app, host="127.0.0.1", port=8080)
