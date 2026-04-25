import os
import sys
import subprocess

def run_script(script_path):
    print(f"\n{'='*50}")
    print(f"-> Running: {script_path}")
    print(f"{'='*50}\n")
    
    # Inject base dir into PYTHONPATH so module imports work
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    
    result = subprocess.run([sys.executable, script_path], env=env)
    
    if result.returncode != 0:
        print(f"\n[Error] Error executing {script_path}. Exiting pipeline.")
        sys.exit(result.returncode)

def main():
    print("=== Starting Recommendation System Build Pipeline ===")
    print("This will regenerate all datasets, embeddings, clusters, and the ChromaDB.")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the precise order of execution and their target artifacts
    pipeline_steps = [
        ("data_clean/clean.py", "data_clean/tmdb_cleaned.csv"),
        ("embeddings/generate_embeddings.py", "embeddings/movie_embeddings.npy"),
        ("models/cluster.py", "clustering/tmdb_clustered_incremental.csv"),
        ("centroids/build_actor_centroids.py", "centroids/actor_centroids.npy"),
        ("centroids/build_actor_cluster_map.py", "centroids/actor_cluster_map.json"),
        ("vector_store/chroma_ingest.py", "chroma_db/chroma.sqlite3"),
        # GA weight optimization — MUST come last (needs populated ChromaDB)
        ("optimization/train_weights.py", "models/optimal_weights.json"),
    ]
    
    # Execute each script if needed
    for step, target in pipeline_steps:
        script_path = os.path.join(base_dir, *step.split("/"))
        target_path = os.path.join(base_dir, *target.split("/"))
        
        if not os.path.exists(script_path):
            print(f"\n[Warning] Script not found at {script_path}. Did you delete it?")
            sys.exit(1)
            
        if os.path.exists(target_path):
            print(f"-> [Skip] {step} (Target already exists: {target})")
            continue
            
        run_script(script_path)
    
    print(f"\n{'='*50}")
    print("[Done] Pipeline execution fully completed! Everything is ready.")
    print(f"{'='*50}\n")
    print("You can now safely run your API server:")
    print("uv run uvicorn api.main:app")

if __name__ == "__main__":
    main()
