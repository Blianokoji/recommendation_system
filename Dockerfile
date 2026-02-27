FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Persistent storage for ChromaDB and ML paths
# (Railway manages volumes externally, so no VOLUME keyword here)

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
