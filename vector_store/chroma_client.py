"""
ChromaDB Client (Singleton)
----------------------------
Persistent storage is automatic when persist_directory is set.
Client and collection are cached as module-level singletons
to avoid re-opening the database on every query.
"""

import os
import chromadb
from chromadb.config import Settings

BASE_DIR = os.getenv("DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(BASE_DIR, "chroma_db"))

COLLECTION_NAME = "tmdb_movies"

_client = None
_collection = None

def get_chroma_collection():
    global _client, _collection

    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=CHROMA_PATH)

    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    return _collection
