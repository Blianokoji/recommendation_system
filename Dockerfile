FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml uv.lock ./

# Install dependencies into the image (system environment for simplicity in Docker)
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy all project files
COPY . .

# Ensure the startup script is executable
RUN chmod +x /app/start.sh

# Default environment variables
ENV PORT=8000
EXPOSE 8000

# The start.sh script handles running build_pipeline.py before uvicorn
CMD ["/app/start.sh"]
