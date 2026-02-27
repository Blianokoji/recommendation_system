# Backend Deployment and Infrastructure Overview

This document describes the backend deployment strategy, infrastructure requirements, and dependency management approach used for the **Safety-Aligned Movie Recommendation System**.  
It is intended for infrastructure, deployment, and operations review.

---

## 1. Deployment Architecture Overview

The system follows a **decoupled frontend-backend architecture**:

- **Frontend**: Deployed on Vercel (stateless, UI-only)
- **Backend**: Deployed as a persistent FastAPI service
- **Vector Store**: ChromaDB with on-disk persistence
- **Model Inference**: SentenceTransformer-based embedding models

Frontend (Vercel)
|
v
Backend API (FastAPI + Retrieval Logic)
|
v
ChromaDB (Persistent Vector Store)


The backend is deployed as a **long-running service** due to:
- Requirement for persistent vector storage
- ML model loading at startup
- Deterministic retrieval behavior

---

## 2. Why Serverless Backends Are Not Used

Serverless platforms (Vercel Functions, Netlify Functions, Cloudflare Workers) were intentionally avoided because:

- They are stateless and cannot persist ChromaDB
- Cold starts negatively impact embedding-based retrieval
- ML inference is not suited for short-lived execution contexts

Hence, the backend is deployed using a **container-based platform with persistent storage**.

---

## 3. Selected Deployment Platform

### Primary Platform: Railway

Railway was selected due to:

- Support for long-running Dockerized services
- Support for persistent disk volumes
- Native HTTPS endpoints
- Compatibility with Python ML workloads
- Low operational overhead for academic projects

Alternative platforms such as Render or AWS EC2 can also be used, but Railway provides the best balance of simplicity and capability for this project.

---

## 4. Dependency Management Using `uv`

The project uses **uv** as the Python package manager instead of pip.

### Reasons for Using uv

- Deterministic dependency resolution via `uv.lock`
- Faster environment setup compared to pip
- Reproducible builds across development and deployment
- Lockfile-based guarantees (important for ML systems)

This ensures that the exact same versions of ML libraries are used during development, deployment, and evaluation.

---

## 5. Dependency Files

The backend uses:

- `pyproject.toml` – Defines project metadata and dependencies
- `uv.lock` – Locks exact dependency versions

No `requirements.txt` file is used.

---

## 6. Docker-Based Deployment (uv-Compatible)

The backend is deployed using Docker with uv for dependency installation.

### Dockerfile

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Persistent storage for ChromaDB
VOLUME /app/chroma_db

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

```
## Key points:

uv sync --frozen enforces exact dependency versions

uv manages the virtual environment internally

The backend runs via uvicorn under uv control

## 7. Persistent Storage Configuration
ChromaDB and massive ML models require persistent storage. Since they total ~100MB and are `.gitignore`d, they will not be pulled automatically during Railway Git builds.

### Volume Configuration
**Mount path inside container:** `/app/data`

This directory stores:
- `chroma_db/` (Vector embeddings, metadata indices)
- `models/` (Incremental Cluster Assignments)
- `embeddings/` (Raw arrays for quick computations)
- `centroids/` (Actor centroids and map files)

### Railway Deployment Setup
On Railway, we bypass volume upload limitations by mirroring the models onto GitHub Releases. 
Because these files total ~116MB, they are zipped on your machine and uploaded out-of-band as a Release Payload. The Dockerfile `wget`s the zip file securely during build time.

1. Compress your output files on your local machine into `ml_data.zip`.
2. Push a GitHub release targeting the `master` branch (e.g., `v1.0.0`) and attach `ml_data.zip`.
3. The Railway builder will download and extract the data dynamically into the `/app/data` folder at deployment time!
*(A detailed step-by-step is available in `railway_volume_instructions.txt` at the root).*

8. Backend API Endpoints
The backend exposes the following endpoints:

Health Check
GET /health
Used to verify service availability and deployment status.

Retrieval Endpoint (Deterministic)
GET /retrieve?query=<user_query>
Uses structured query parsing

Enforces hard constraints (actors, safety)

Applies weighted semantic ranking

Returns explainable results

A second endpoint using SLM-based reranking may be enabled later.

9. CORS Configuration
Since the frontend is hosted on Vercel, CORS is enabled in FastAPI:

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
This allows secure cross-origin communication during demo and evaluation.

10. Summary of Infrastructure Requirements
Container runtime (Docker)

Python 3.11+

Persistent disk storage

Internet access for TMDB API (data ingestion phase)

HTTPS endpoint for frontend integration

11. Deployment Rationale Summary
The chosen deployment strategy ensures:

Deterministic behavior (no hallucinations)

Persistent vector storage

Explainable retrieval

Scalability for demo usage

Reproducibility for academic evaluation

This setup aligns with the project’s emphasis on safety, robustness, and controlled reasoning.

