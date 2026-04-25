"""
SLM Reasoner Module
-------------------
Provides a final reranking layer using Gemini 2.5 Flash.
Takes the top deterministic candidates and uses structured outputs
to return the absolute best matches with plain-english reasoning.
"""

import os
import json
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

# ------------------------------------------------
# SCHEMAS
# ------------------------------------------------

class RerankedMovie(BaseModel):
    tmdb_id: str = Field(description="The TMDB ID of the movie")
    reasoning: str = Field(description="A 1-2 sentence explanation of why this movie perfectly matches the user's explicit and implicit intent. Highlight specific thematic connections or constraints met.")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0 that this movie is exactly what the user is looking for.")

class RerankResponse(BaseModel):
    curated_movies: list[RerankedMovie] = Field(description="The curated and reranked list of movies, sorted by confidence descending.")

# ------------------------------------------------
# SLM LOGIC
# ------------------------------------------------

def rerank_with_slm(query: str, parsed_intent: dict, candidates: List[dict], max_results: int = 5) -> List[dict]:
    """
    Reranks a list of deterministic candidates using Gemini.

    Args:
        query: The raw user query
        parsed_intent: The parsed intent dictionary from the query parser
        candidates: A list of movie dictionaries (tmdb_id, title, metadata, score)
        max_results: The number of top movies to return

    Returns:
        The reranked and filtered list of candidate dictionaries, enriched with 'slm_reasoning'
    """
    if not candidates:
        return []

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        print("[WARN] Gemini API key not found or google-genai not installed. Falling back to deterministic rank.")
        return candidates[:max_results]

    try:
        client = genai.Client(api_key=api_key)
        
        # Prepare the candidate list for the prompt
        candidate_summary = []
        for i, m in enumerate(candidates):
            meta = m.get("metadata", {})
            summary = (
                f"[{i}] TMDB ID: {m['tmdb_id']} | Title: {m['title']} | "
                f"Year: {meta.get('release_year', 'N/A')} | "
                f"Genres: {meta.get('genres', 'N/A')} | "
                f"Cast: {meta.get('actor', 'N/A')}\n"
            )
            candidate_summary.append(summary)
            
        candidate_text = "".join(candidate_summary)

        # Build prompt
        prompt = f"""You are a master movie curator. A user has sent the following query:
"{query}"

The system's deterministic retrieval engine has parsed their intent as:
{json.dumps(parsed_intent, indent=2)}

Here are the top candidates retrieved by the system:
{candidate_text}

Your task:
1. Review the candidates against the user's raw query and parsed intent.
2. Filter out movies that do not genuinely fit the desired tone, temporal constraints, or actors.
3. Select the absolute best {max_results} (or fewer) movies from this list.
4. For each selected movie, provide a brief reasoning explaining why it perfectly matches the user's request.
5. Provide a confidence score (0.0 to 1.0).
6. Return the results sorted by confidence (highest first).
"""

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RerankResponse,
                temperature=0.2, # Low temperature for more analytical ranking
            ),
        )
        
        raw_json = response.text
        if not raw_json:
            return candidates[:max_results]
            
        result_data = json.loads(raw_json)
        curated_items = result_data.get("curated_movies", [])
        
        # Merge SLM results back with the original candidate data
        candidate_dict = {m["tmdb_id"]: m for m in candidates}
        final_list = []
        
        for item in curated_items:
            tid = item["tmdb_id"]
            if tid in candidate_dict:
                movie = candidate_dict[tid].copy()
                movie["slm_reasoning"] = item["reasoning"]
                movie["slm_confidence"] = item["confidence"]
                # Override the system score with SLM confidence so it sorts correctly
                movie["deterministic_score"] = movie["score"]
                movie["score"] = item["confidence"]
                final_list.append(movie)
                
        # If SLM failed to return anything valid or hallucinated IDs, fallback
        if not final_list:
            return candidates[:max_results]
            
        return final_list

    except Exception as e:
        print(f"[ERROR] SLM reranking failed: {e}. Falling back to deterministic.")
        return candidates[:max_results]
