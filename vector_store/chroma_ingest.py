"""
ChromaDB Ingestion Script
------------------------
Ingests movie embeddings into ChromaDB with metadata for
safe, cluster-aware retrieval.

Compatible with ChromaDB >= 0.4.x
Persistence is automatic when persist_directory is set.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from chroma_client import get_chroma_collection

# ------------------ PATH CONFIG ------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLUSTERED_CSV = os.path.join(
    BASE_DIR, "clustering", "tmdb_clustered_incremental.csv"
)

EMBEDDINGS_FILE = os.path.join(
    BASE_DIR, "embeddings", "movie_embeddings.npy"
)

# ------------------ INGESTION ------------------

def ingest():
    print("[INFO] Loading clustered dataset...")
    df = pd.read_csv(CLUSTERED_CSV)

    print("[INFO] Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_FILE)

    assert len(df) == len(embeddings), "Embedding / data length mismatch"

    collection = get_chroma_collection()

    # Fetch existing IDs to avoid duplication
    existing = collection.get(include=[])
    existing_ids = set(existing["ids"]) if existing and "ids" in existing else set()

    print("[INFO] Ingesting embeddings into ChromaDB...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        doc_id = str(row["tmdb_id"])

        if doc_id in existing_ids:
            continue

        metadata = {
            "title": row["title"],
            "cluster_id": int(row["cluster_id"]),
            "cluster_safe": bool(row["cluster_safe"]),
            "adult": bool(row["adult"]),
            "release_year": int(row["release_year"])
                if not pd.isna(row["release_year"]) else None,
            "genres": row["genres"] or "",
            "actor": row["cast"] or ""
        }

        # Include title and cast in the document text for keyword search
        doc_text = f"{row['title']} {row['cast'] or ''}"

        collection.add(
            ids=[doc_id],
            embeddings=[embeddings[idx].tolist()],
            metadatas=[metadata],
            documents=[doc_text]
        )

    print("[DONE] ChromaDB ingestion completed.")
    print("[DONE] Total vectors in collection:", collection.count())


# ------------------ ENTRY POINT ------------------

if __name__ == "__main__":
    ingest()
