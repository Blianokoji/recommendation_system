<div align="center">

# 🎬 Movie Recommendation System

### Intelligent Semantic Search with Hard Constraints, Centroid Gating & Diversity

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Cloudflare-F38020?style=for-the-badge&logo=cloudflare&logoColor=white" />
</p>

</div>

---

## 📖 Overview

This project is a sophisticated **Hybrid Movie Recommendation Engine** that acts as the backend API for a Vercel-hosted frontend. It combines the power of **Semantic Vector Search** with **Structured Hard Constraints** and **Actor Centroid Gating**. 

Unlike simple KNN lookups, this system understands query intent, enforces strict filters, intuits actor profiles from semantic descriptions, and ensures result diversity through local clustering. It is fully containerized using Docker and securely exposed via a Cloudflare Tunnel.

---

## 🧠 How It Works

The system operates in two distinct phases:

### 🏗️ Stage 1: The Build Phase (Data Pipeline)
Orchestrated by `build_pipeline.py`, this phase runs automatically during the Docker build process to prepare the data.
1. **Data Cleaning**: Strips noise from the raw TMDB dataset and filters adult content.
2. **Embedding Generation**: Uses `all-MiniLM-L6-v2` to convert movie plots into 384-dimensional vectors.
3. **Global Clustering**: Groups movies via MiniBatchKMeans to understand semantic neighborhoods.
4. **Actor Centroids**: Calculates a "mathematical center" for an actor's career by averaging the embeddings of all their movies. This allows the system to understand an actor's "vibe" (e.g., action stars vs rom-com leads).
5. **Vector Ingestion**: Bakes the data into a local ChromaDB instance inside the Docker image.

### 🔍 Stage 2: The Inference Phase (API Retrieval)
Happens in milliseconds when a request hits the `/retrieve` endpoint.
1. **Query Parsing**: Extracts intents, hard constraints (years, adult flags), and soft constraints (plot vibes).
2. **Centroid Gating**: If no actor is named, but the plot description heavily matches an Actor Centroid (e.g., *"movies with an actor who does his own stunts"*), the system secretly applies a hard constraint for that actor.
3. **Vector Retrieval**: Queries ChromaDB using the soft constraint vector while strictly respecting metadata filters.
4. **Fuzzy-Aware Scoring**: Ranks the top 80 candidates using a 4-part formula (Semantic, Cluster, Temporal, Actor).
5. **Diversity Re-Ranking**: Clusters the candidates locally and selects top matches from *different* clusters to prevent returning 10 identical sequels.

---

## 🎓 The Academic Perspective: Why This Beats SOTA

Traditional State-of-the-Art (SOTA) recommendation systems generally fall into two categories: **Collaborative Filtering** (classic recommendation) and **RAG + LLM pipelines** (modern AI search). This system introduces novel architectural patterns to solve the critical flaws in both.

### 1. vs. SOTA Recommender Systems (Collaborative Filtering)
**The SOTA Approach**: Netflix or Amazon rely on Matrix Factorization and Collaborative Filtering (CF). They recommend based on *"Users who liked X also liked Y."*
**The Flaw**: The **Cold Start Problem**. They cannot handle novel, hyper-specific queries like *"a movie where a dog saves the day in a snowy town."* They only understand user history, not content meaning.
**Our Approach (Zero-Shot Semantics)**: By relying entirely on deep semantic embeddings, our system maps the mathematical meaning of the plot directly to the user's conversational intent. It requires zero user history, bypassing the Cold Start problem entirely and enabling highly specific, intent-driven discovery.

### 2. vs. SOTA RAG + LLM Pipelines
**The SOTA Approach**: Standard Retrieval-Augmented Generation grabs the top 10 vector matches and stuffs them into a massive LLM (like GPT-4) to pick the best ones.
**The Flaws**: Extreme latency (3-5 seconds per query), hallucination of constraints (LLMs often ignore exact years), and over-reliance on the LLM to fix a broken retrieval pipeline.
**Our Approach**: We push the intelligence down into a deterministic, sub-100ms retrieval layer. We solve the flaws of standard vector search mathematically *before* any LLM is involved using the three architectural patterns below:

### 3. The "Semantic Trap" vs. Dual-Clustering
**The Problem (SOTA)**: If you query a standard vector database for *"boy wizard goes to school"*, it will mathematically return *Harry Potter* parts 1 through 8. It provides a terrible user experience because there is zero diversity. 
**The Solution (Our Approach)**: We use a Dual-Clustering topology. 
*   **Global Clustering** (Build Phase): Groups the entire database to understand macro-genres.
*   **Local Clustering** (Inference Phase): We take the top 80 semantic matches and run *Agglomerative Clustering* on the fly. We then pluck the highest-scoring movie from *different* local clusters. 
*   *Result*: You get *Harry Potter*, but you also get *The Golden Compass* and *Percy Jackson*.

### 4. Entity Blindness vs. Latent Semantic Profiling (Centroids)
**The Problem (SOTA)**: Traditional systems rely on Named Entity Recognition (NER). If you search *"Tom Cruise space movies"*, NER finds "Tom Cruise" and filters the database. But if you search *"movies with that actor who does crazy stunts hanging from airplanes"*, NER fails completely because there is no explicit name.
**The Solution (Our Approach)**: **Actor Centroid Gating**. During the build phase, we average the 384-dimensional plot vectors of every movie an actor has been in, creating a "Latent Semantic Profile" (Centroid) for their career. 
*   When the query *"actor who does crazy stunts"* comes in, it doesn't match any movie perfectly, but its vector has a **0.72 cosine similarity with Tom Cruise's Centroid**.
*   The system *infers* the actor and secretly applies a hard database filter for "Tom Cruise", returning his exact filmography based purely on a description of his "vibe."

### 5. Rigid Metadata vs. Fuzzy-Aware Scoring
**The Problem (SOTA)**: Standard hybrid search applies rigid boolean filters: `year > 1990 AND year < 2000`. If you ask for a "classic 90s movie" and the best match was made in 1989, it gets ruthlessly eliminated.
**The Solution (Our Approach)**: We model temporal and identity constraints as **Fuzzy Membership Curves**. The year 1989 doesn't get a `0` (eliminated), it gets a `0.85` membership score for the concept of "90s". This score acts as a multiplier in our proprietary 4-part ranking algorithm, mathematically blending structured metadata with raw semantic distance.

### 6. Architectural Trade-offs
No architecture is perfect. Here are the explicit design compromises made for this system:

*   **Tradeoff 1: Heavy Pre-computation (Build Phase)**
    *   *The Cost*: Generating 384-dimensional embeddings, executing global K-Means clustering, and calculating hundreds of Actor Centroids takes significant CPU time (~15 mins locally).
    *   *Verdict: Highly Acceptable*. Movie databases update infrequently. By shifting 99% of the computational weight to an asynchronous build phase, we guarantee sub-100ms latency for the user during inference.
*   **Tradeoff 2: Strict Reliance on Metadata Quality**
    *   *The Cost*: If a movie's TMDB plot summary is vague (e.g., "A man goes on a journey") or the cast list is missing, the embeddings and centroids will fail to capture its true essence.
    *   *Verdict: Acceptable*. TMDB is heavily curated. For enterprise use cases on messy corporate data, this pipeline would require an upfront LLM "data enrichment" pass to generate synthetic, detailed descriptions before embedding.
*   **Tradeoff 3: Dropping Deep Personalization (Passive Discovery)**
    *   *The Cost*: Because we don't track user click history or watch time, the system cannot passively recommend a movie just because "people similar to you liked it."
    *   *Verdict: Acceptable for the Use Case*. This is an **intent-driven search engine**, not a passive scrolling feed. It is designed to perfectly fulfill a user's exact craving in the moment, which CF algorithms fail to do.

---

## 🚀 Setup & Deployment

The system is designed to run locally on your machine (acting as the server) and tunnel out to the internet via Cloudflare, allowing your Vercel frontend to communicate with it securely.

### Prerequisites
- Docker Desktop installed and running.
- A Cloudflare account with a Zero Trust Tunnel created.

### 1. Environment Variables
Create a `.env` file in the root directory:
```env
TMDB_API_KEY=your_tmdb_key
GEMINI_API_KEY=your_gemini_key
CLOUDFLARE_TUNNEL_TOKEN=your_cloudflare_tunnel_token
```

### 2. Build and Run
Because the Dockerfile is heavily optimized with layers, the first build takes 10-20 minutes (ML dependencies + ChromaDB generation). Subsequent builds take seconds.

```bash
# Build the images
docker compose build

# Start the API and the Cloudflare Tunnel sidecar
docker compose up -d

# View logs
docker compose logs -f
```

### 3. Architecture

```text
┌─────────────────── Your Laptop ───────────────────┐
│                                                    │
│  ┌──────────────┐       ┌───────────────────┐     │
│  │  cloudflared  │◄─────│  Cloudflare Edge   │◄───── Vercel Frontend
│  │  (tunnel)     │      │  (reverse proxy)   │     │
│  └──────┬───────┘       └───────────────────┘     │
│         │ http://app:8000                          │
│  ┌──────▼───────┐                                  │
│  │  FastAPI      │                                  │
│  │  recsys-api   │                                  │
│  │  :8000        │                                  │
│  └──────────────┘                                  │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## 🔌 API Endpoints

The API runs on port `8000` (or via your Cloudflare Tunnel URL).

### `POST /retrieve`
Unified endpoint for recommendations.

**Request Body:**
```json
{
  "query": "emotional space movies",
  "use_slm": false
}
```

**Response (Snippet):**
```json
{
  "mode": "deterministic",
  "query": "emotional space movies",
  "intent_passable": true,
  "result_count": 5,
  "results": [
    {
      "tmdb_id": 157336,
      "title": "Interstellar",
      "score": 0.824,
      "explanation": { ... }
    }
  ]
}
```

### `GET /health`
Returns `{"status": "ok"}` if the FastAPI server is running.

---

## 📂 Project Structure

```bash
recommendation_system/
├── 📂 api/                # FastAPI application and routing
├── 📂 centroids/          # Actor centroid generation and matching
├── 📂 clustering/         # Global diversity clustering logic
├── 📂 data_clean/         # Raw TMDB data preprocessing
├── 📂 embeddings/         # SentenceTransformer singleton & generation
├── 📂 models/             # Weights, GA optimization, KMeans models
├── 📂 query_parser/       # Intent extraction & NLP constraint logic
├── 📂 retrieval/          # Core vector search & 4-part scoring engine
├── 📂 vector_store/       # ChromaDB client & ingestion scripts
├── 📄 build_pipeline.py   # Master orchestrator for the build phase
├── 📄 docker-compose.yml  # Multi-container setup (App + Tunnel)
├── 📄 Dockerfile          # Layered image with baked ML artifacts
└── 📄 README.md           # This file
```
