"""
Embedding Generation Script
---------------------------

Generates sentence-level embeddings for cleaned TMDB movie data.
These embeddings are used for:
- semantic retrieval (ChromaDB)
- incremental clustering (MiniBatch K-Means)
- query-time similarity matching

Design principles:
- Offline computation
- Deterministic mapping between tmdb_id and embedding index
- Normalized embeddings for stable similarity
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# ------------------ PATH CONFIG ------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_CSV = os.path.join(BASE_DIR, "data_clean", "tmdb_cleaned.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "embeddings")
EMBEDDING_FILE = os.path.join(OUTPUT_DIR, "movie_embeddings.npy")
INDEX_FILE = os.path.join(OUTPUT_DIR, "embedding_index.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ LOAD DATA ------------------

print("[INFO] Loading cleaned dataset...")
if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(
        f"Cleaned data not found at {DATA_CSV}. "
        "Run data_clean/clean.py first."
    )

df = pd.read_csv(DATA_CSV)

required_cols = {"tmdb_id", "embedding_text"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

texts = df["embedding_text"].astype(str).tolist()

print(f"[INFO] Loaded {len(texts)} movies for embedding.")

# ------------------ LOAD MODEL ------------------

print("[INFO] Loading SentenceTransformer model: all-MiniLM-L6-v2")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ GENERATE EMBEDDINGS ------------------

print("[INFO] Generating embeddings...")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True
)

# ------------------ NORMALIZE ------------------

print("[INFO] Normalizing embeddings...")
embeddings = normalize(embeddings)

# ------------------ SAVE OUTPUTS ------------------

print(f"[INFO] Saving embeddings to {EMBEDDING_FILE}")
np.save(EMBEDDING_FILE, embeddings)

print(f"[INFO] Saving embedding index to {INDEX_FILE}")
df[["tmdb_id"]].to_csv(INDEX_FILE, index=False)

print("[DONE] Embedding generation completed successfully.")
print(f"[DONE] Embedding matrix shape: {embeddings.shape}")
