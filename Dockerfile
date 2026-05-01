FROM python:3.11-slim

# System deps for native extensions (chromadb/hnswlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast, from official image)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# ---- Layer 1: Dependencies ----
# Step A: Install CPU-only PyTorch FIRST from the dedicated index.
# The uv.lock was generated on Windows and contains CUDA torch for Linux.
# By pre-installing CPU torch, uv sync will see it's already satisfied and skip
# the 2GB+ NVIDIA download entirely.
RUN uv pip install --system --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

# Step B: Install everything else from the lock file.
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache -r pyproject.toml

# ---- Layer 2: Pre-built ML artifacts (heavy, rarely change) ----
COPY data_clean/tmdb_cleaned.csv              data_clean/tmdb_cleaned.csv
COPY embeddings/movie_embeddings.npy          embeddings/movie_embeddings.npy
COPY embeddings/embedding_index.csv           embeddings/embedding_index.csv
COPY clustering/tmdb_clustered_incremental.csv clustering/tmdb_clustered_incremental.csv
COPY models/minibatch_kmeans.joblib           models/minibatch_kmeans.joblib
COPY centroids/actor_centroids.npy            centroids/actor_centroids.npy
COPY centroids/actor_cluster_map.json         centroids/actor_cluster_map.json
COPY centroids/actor_index.json               centroids/actor_index.json
COPY centroids/actor_stats.csv                centroids/actor_stats.csv

# ---- Layer 3: Application code (changes frequently) ----
COPY api/           api/
COPY retrieval/     retrieval/
COPY query_parser/  query_parser/
COPY vector_store/  vector_store/
COPY slm/           slm/
COPY optimization/  optimization/
COPY embeddings/embedding_singleton.py embeddings/embedding_singleton.py
COPY build_pipeline.py .
COPY start.sh .

# ---- Layer 4: Build ChromaDB from pre-copied artifacts ----
RUN python build_pipeline.py

RUN chmod +x /app/start.sh

ENV PORT=8000
EXPOSE 8000

CMD ["/app/start.sh"]
