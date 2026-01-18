import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# ------------------ PATHS ------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_CSV = os.path.join(BASE_DIR, "data_clean", "tmdb_cleaned.csv")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings", "movie_embeddings.npy")

OUTPUT_DIR = os.path.join(BASE_DIR, "centroids")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CENTROID_FILE = os.path.join(OUTPUT_DIR, "actor_centroids.npy")
INDEX_FILE = os.path.join(OUTPUT_DIR, "actor_index.json")
STATS_FILE = os.path.join(OUTPUT_DIR, "actor_stats.csv")

MIN_MOVIES = 5   # minimum films to form a centroid

# ------------------ LOAD DATA ------------------

print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_CSV)
embeddings = np.load(EMBEDDINGS_FILE)

assert len(df) == len(embeddings), "Data / embedding mismatch"

# ------------------ COLLECT MOVIES PER ACTOR ------------------

actor_to_indices = defaultdict(list)

for idx, cast_str in enumerate(df["cast"].fillna("")):
    actors = [a.strip() for a in cast_str.split(",") if a.strip()]
    for actor in actors:
        actor_to_indices[actor].append(idx)

print(f"[INFO] Found {len(actor_to_indices)} unique actors")

# ------------------ BUILD CENTROIDS ------------------

centroids = []
actor_index = {}
stats = []

for actor, indices in tqdm(actor_to_indices.items(), desc="Building actor centroids"):
    if len(indices) < MIN_MOVIES:
        continue

    vecs = embeddings[indices]
    centroid = vecs.mean(axis=0)

    centroid_id = len(centroids)
    centroids.append(centroid)

    actor_index[actor] = centroid_id

    stats.append({
        "actor": actor,
        "movie_count": len(indices)
    })

# ------------------ SAVE ------------------

centroids = np.vstack(centroids)

np.save(CENTROID_FILE, centroids)

with open(INDEX_FILE, "w", encoding="utf-8") as f:
    json.dump(actor_index, f, indent=2)

pd.DataFrame(stats).sort_values(
    "movie_count", ascending=False
).to_csv(STATS_FILE, index=False)

print("[DONE] Actor centroid construction complete")
print(f"[INFO] Saved {len(centroids)} actor centroids")
