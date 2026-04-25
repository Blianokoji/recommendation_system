#!/bin/bash
set -e

echo "=== System Initialization ==="
echo "Checking and building ML artifacts (if missing)..."

# Run the idempotent build pipeline
# We assume dependencies are installed in the system path or current environment
python build_pipeline.py

echo "=== Starting API Server ==="
# Run the FastAPI server via Uvicorn
# Use the PORT environment variable, defaulting to 8000
exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
