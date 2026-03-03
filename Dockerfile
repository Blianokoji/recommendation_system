FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl wget unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Persistent storage logic handled via local GitHub download caching
# Download the model artifacts into a temporary setup block
ARG DATA_URL="https://github.com/Blianokoji/recommendation_system/releases/download/v1.0.0/ml_data.zip"
RUN wget -qO ml_data.zip ${DATA_URL} && \
    unzip -q ml_data.zip -d /app/data && \
    rm ml_data.zip

# Map local data directory logic from earlier
ENV DATA_DIR="/app/data"
ENV CHROMA_DB_PATH="/app/data"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Using shell form to support dynamic $PORT substitution injected by Railway
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}
