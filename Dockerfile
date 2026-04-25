FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml uv.lock ./

# Install dependencies into the image (system environment)
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy all project files
COPY . .

# --- BUILD PHASE ---
# Build the vector database and ML artifacts during Docker build
# This ensures the API starts instantly and fits in low-RAM environments (like Render Free)
RUN python build_pipeline.py

# Ensure start.sh is executable
RUN chmod +x /app/start.sh

# Render uses the PORT environment variable
ENV PORT=8000
EXPOSE 8000

# The start.sh script now just starts the server
# (since build_pipeline.py already ran during docker build)
CMD ["/app/start.sh"]
