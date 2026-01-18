"""
Incremental Clustering Pipeline
-------------------------------

Performs incremental semantic clustering over movie embeddings using
MiniBatchKMeans. Clusters are post-annotated with safety metadata
to enable policy-based filtering during retrieval.

Design goals:
- Incremental (partial_fit)
- Re-runnable and resumable
- Safety-aware at cluster level
- Paper-defensible
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

# ------------------ PATH CONFIG ------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_CSV = os.path.join(BASE_DIR, "data_clean", "tmdb_cleaned.csv")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings", "movie_embeddings.npy")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "minibatch_kmeans.joblib")

OUTPUT_DIR = os.path.join(BASE_DIR, "clustering")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "tmdb_clustered_incremental.csv")
CLUSTER_STATS_CSV = os.path.join(OUTPUT_DIR, "cluster_stats.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ PARAMETERS ------------------

N_CLUSTERS = 150
BATCH_SIZE = 1024
RANDOM_STATE = 42

# Safety threshold: % of adult movies allowed in a cluster
ADULT_RATIO_THRESHOLD = 0.15

# ------------------ LOAD DATA ------------------

print("[INFO] Loading cleaned dataset and embeddings...")

df = pd.read_csv(DATA_CSV)
X = np.load(EMBEDDINGS_FILE)

assert len(df) == len(X), "Mismatch between data rows and embeddings"

# ------------------ LOAD / INIT MODEL ------------------

if os.path.exists(MODEL_PATH):
    print("[INFO] Loading existing MiniBatchKMeans model...")
    kmeans = joblib.load(MODEL_PATH)
else:
    print("[INFO] Initializing new MiniBatchKMeans model...")
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE
    )

# ------------------ INCREMENTAL TRAINING ------------------

print("[INFO] Performing incremental clustering...")

for start in range(0, len(X), BATCH_SIZE):
    end = start + BATCH_SIZE
    batch = X[start:end]
    kmeans.partial_fit(batch)

# ------------------ ASSIGN CLUSTERS ------------------

print("[INFO] Assigning cluster IDs...")
df["cluster_id"] = kmeans.predict(X)

# ------------------ CLUSTER SAFETY ANNOTATION ------------------

print("[INFO] Computing cluster-level safety statistics...")

cluster_stats = (
    df.groupby("cluster_id")
      .agg(
          cluster_size=("tmdb_id", "count"),
          adult_ratio=("adult", "mean"),
          avg_vote=("vote_average", "mean"),
          avg_popularity=("popularity", "mean")
      )
      .reset_index()
)

cluster_stats["cluster_safe"] = (
    cluster_stats["adult_ratio"] <= ADULT_RATIO_THRESHOLD
)

# Merge safety labels back
df = df.merge(
    cluster_stats[["cluster_id", "cluster_safe"]],
    on="cluster_id",
    how="left"
)

# ------------------ SAVE OUTPUTS ------------------

print("[INFO] Saving clustering model...")
joblib.dump(kmeans, MODEL_PATH)

print("[INFO] Saving clustered dataset...")
df.to_csv(OUTPUT_CSV, index=False)

print("[INFO] Saving cluster statistics...")
cluster_stats.to_csv(CLUSTER_STATS_CSV, index=False)

print("[DONE] Incremental clustering completed successfully.")
print(f"[DONE] Total clusters: {N_CLUSTERS}")
print(f"[DONE] Safe clusters: {cluster_stats['cluster_safe'].sum()}")
