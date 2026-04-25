"""
GA Weight Training Script (Lightweight)
-----------------------------------------
Evolves optimal weights for the 4-component scoring formula
WITHOUT querying ChromaDB. Uses analytical fitness based on
the structure of golden benchmark queries.

Runs in < 5 seconds. Outputs: models/optimal_weights.json
"""

import os
import sys
import json
from datetime import date

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from optimization.ga_optimizer import evolve

# ------------------------------------------------
# PATHS
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_QUERIES_FILE = os.path.join(BASE_DIR, "optimization", "golden_queries.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "models", "optimal_weights.json")

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# ------------------------------------------------
# LOAD GOLDEN QUERIES
# ------------------------------------------------

print("[INFO] Loading golden benchmark queries...")
with open(GOLDEN_QUERIES_FILE, "r", encoding="utf-8") as f:
    golden_queries = json.load(f)

print(f"[INFO] Loaded {len(golden_queries)} benchmark queries")

# ------------------------------------------------
# ANALYTICAL FITNESS (no ChromaDB needed)
# ------------------------------------------------
# The fitness function rewards weight distributions that
# allocate resources proportionally to what the benchmark
# query set actually needs. This is a structural prior.

# Pre-compute query features
n_temporal = sum(1 for q in golden_queries if q.get("temporal"))
n_actor = sum(1 for q in golden_queries if q.get("actor"))
n_general = len(golden_queries) - n_temporal - n_actor
# Some queries have both temporal + actor
n_both = sum(1 for q in golden_queries if q.get("temporal") and q.get("actor"))

temporal_ratio = n_temporal / len(golden_queries)
actor_ratio = n_actor / len(golden_queries)

print(f"[INFO] Query distribution: {n_temporal} temporal, {n_actor} actor, {n_general} general, {n_both} both")


def fitness_function(weights: np.ndarray) -> float:
    """
    Analytical fitness: scores how well a weight vector serves
    the mix of query types in the golden benchmark.

    Rewards:
    - Semantic weight being dominant (it's always useful)
    - Temporal weight being proportional to temporal query frequency
    - Actor weight being proportional to actor query frequency
    - Cluster weight being moderate (diversity matters but isn't primary)
    - Penalizes extreme imbalance
    """
    w_sem, w_clust, w_temp, w_actor = weights

    score = 0.0

    # 1. Semantic should be the backbone (0.4 - 0.7 is ideal)
    if 0.35 <= w_sem <= 0.75:
        score += 0.3
    score += 0.1 * (1.0 - abs(w_sem - 0.55))  # peak at 0.55

    # 2. Temporal weight should reflect temporal query proportion
    ideal_temp = temporal_ratio * 0.5  # scale down since semantic carries too
    score += 0.2 * (1.0 - abs(w_temp - ideal_temp))

    # 3. Actor weight should reflect actor query proportion
    ideal_actor = actor_ratio * 0.35
    score += 0.15 * (1.0 - abs(w_actor - ideal_actor))

    # 4. Cluster weight should be moderate (0.10 - 0.25)
    if 0.08 <= w_clust <= 0.30:
        score += 0.15
    score += 0.05 * (1.0 - abs(w_clust - 0.18))

    # 5. Penalize any single weight dominating > 0.8
    if max(weights) > 0.80:
        score -= 0.3

    # 6. Penalize any weight being negligible (< 0.02)
    if min(weights) < 0.02:
        score -= 0.1

    # 7. For queries with BOTH actor + temporal, reward balanced split
    if n_both > 0:
        balance = 1.0 - abs(w_temp - w_actor)
        score += 0.05 * balance

    return score


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():
    print("[INFO] Starting lightweight GA weight optimization...")

    best_weights, best_fitness = evolve(
        fitness_fn=fitness_function,
        pop_size=40,
        generations=20,
        verbose=True
    )

    result = {
        "W_semantic": round(float(best_weights[0]), 4),
        "W_cluster": round(float(best_weights[1]), 4),
        "W_temporal": round(float(best_weights[2]), 4),
        "W_actor": round(float(best_weights[3]), 4),
        "fitness": round(float(best_fitness), 4),
        "trained_year": date.today().year
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[DONE] Optimal weights saved to {OUTPUT_FILE}")
    print(f"[DONE] {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
