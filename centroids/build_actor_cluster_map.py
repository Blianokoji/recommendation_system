import os
import json
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_CSV = os.path.join(BASE_DIR, "clustering", "tmdb_clustered_incremental.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "centroids", "actor_cluster_map.json")

TOP_K_CLUSTERS = 5
MIN_MOVIES = 5

df = pd.read_csv(DATA_CSV)

actor_clusters = defaultdict(list)

for _, row in tqdm(df.iterrows(), total=len(df)):
    if pd.isna(row["cast"]) or pd.isna(row["cluster_id"]):
        continue

    actors = [a.strip() for a in row["cast"].split(",") if a.strip()]
    for actor in actors:
        actor_clusters[actor].append(row["cluster_id"])

actor_cluster_map = {}

for actor, clusters in actor_clusters.items():
    if len(clusters) < MIN_MOVIES:
        continue

    counts = Counter(clusters)
    top_clusters = [c for c, _ in counts.most_common(TOP_K_CLUSTERS)]
    actor_cluster_map[actor] = top_clusters

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(actor_cluster_map, f, indent=2)

print(f"[DONE] Actor → cluster map saved ({len(actor_cluster_map)} actors)")
