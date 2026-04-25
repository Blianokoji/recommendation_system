#!/bin/bash
set -e

echo "=== System Check ==="
# Check if ChromaDB exists (should be baked into the image)
if [ ! -d "chroma_db" ]; then
  echo "[WARN] chroma_db not found! Attempting emergency build..."
  python build_pipeline.py
else
  echo "[INFO] ML artifacts found in image. Skipping build."
fi

echo "=== Starting API Server ==="
# Run the FastAPI server via Uvicorn
# Link to Render's dynamic PORT
exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
