"""
ChromaDB Client (New API)
------------------------
Persistent storage is automatic when persist_directory is set.
"""

import os
import chromadb
from chromadb.config import Settings

BASE_DIR = os.getenv("DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

COLLECTION_NAME = "tmdb_movies"

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("[DEBUG] Chroma persist path:", CHROMA_PATH)
    return collection
