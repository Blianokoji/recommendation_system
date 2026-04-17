import os
import subprocess

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

def render_dot(name, dot_src):
    path = os.path.join(OUT, name)
    dot_path = path.replace('.png', '.dot')
    with open(dot_path, 'w', encoding='utf-8') as f:
        f.write(dot_src)
    
    subprocess.run(["dot", "-Tpng", dot_path, "-o", path, "-Gdpi=200"], check=True)
    os.remove(dot_path)
    print(f"  ✓  {name} (Graphviz)")

# ══════════════════════════════════════════════════════════════════════════════
# DFD Level 0 Context Diagram
# ══════════════════════════════════════════════════════════════════════════════
dfd_0 = """
digraph DFD0 {
    fontname="Helvetica,Arial,sans-serif"
    node [fontname="Helvetica,Arial,sans-serif", shape=rect, style="rounded,filled", fillcolor="#e2e8f0", penwidth=1.5]
    edge [fontname="Helvetica,Arial,sans-serif", fontsize=10, penwidth=1.2, color="#374151", minlen=2]
    rankdir=LR

    # External Entities
    User [label="End User\n(Client)", fillcolor="#bbf7d0"]
    TMDB [label="TMDB\nAPI", fillcolor="#fed7aa"]
    Admin [label="Admin /\nDevOps", fillcolor="#e9d5ff"]
    Cloud [label="Docker /\nCloud", fillcolor="#fecaca"]

    # Main System
    System [label="Semantic Movie\nRecommendation\nSystem", shape=circle, style="filled", fillcolor="#bfdbfe", width=1.5]

    # Flows
    TMDB -> System [label="Raw movie metadata"]
    User -> System [label="Natural language query"]
    System -> User [label="JSON recommendations"]
    System -> Cloud [label="Deployed container image"]
    Admin -> System [label="Config / env vars"]
    System -> Admin [label="Logs / health"]
}
"""

# ══════════════════════════════════════════════════════════════════════════════
# DFD Level 1
# ══════════════════════════════════════════════════════════════════════════════
dfd_1 = """
digraph DFD1 {
    fontname="Helvetica,Arial,sans-serif"
    node [fontname="Helvetica,Arial,sans-serif", shape=rect, fillcolor="#e2e8f0", style=filled, penwidth=1.5]
    edge [fontname="Helvetica,Arial,sans-serif", fontsize=10, penwidth=1.2, color="#374151"]
    rankdir=LR
    splines=polyline

    # External Entities
    Ext_TMDB [label="TMDB API", shape=Mrecord, fillcolor="#fed7aa"]
    Ext_User [label="User Client", shape=Mrecord, fillcolor="#bbf7d0"]

    # Processes
    node [shape=circle, fillcolor="#bfdbfe"]
    P1 [label="P1\nData Ingestion"]
    P2 [label="P2\nData Cleaning\n& Embedding"]
    P3 [label="P3\nOffline Clustering"]
    P4 [label="P4\nActor Centroids"]
    P5 [label="P5\nVector DB\nIngestion"]
    P6 [label="P6\nQuery Parser"]
    P7 [label="P7\nRetrieval Engine"]
    P8 [label="P8\nREST API Layer"]

    # Data Stores
    node [shape=box3d, fillcolor="#fcd34d"]
    D1 [label="D1: movie_embeddings.npy"]
    D2 [label="D2: ChromaDB\n(Index + Meta)"]
    D3 [label="D3: actor_centroids.npy"]

    # Flows (Offline)
    Ext_TMDB -> P1 [label="API JSON"]
    P1 -> P2 [label="Raw CSV"]
    P2 -> D1 [label="Cleaned text +\nembeddings"]
    P2 -> P3 [label="Embeddings"]
    P3 -> P4 [label="Cluster mappings"]
    P4 -> D3 [label="Actor representations"]
    P3 -> P5 [label="Clusters / Metadata"]
    P5 -> D2 [label="Populate DB"]

    # Flows (Online)
    Ext_User -> P8 [label="NL Query"]
    P8 -> P6 [label="Query Text"]
    P6 -> P7 [label="Structured Intent\n+ Constraints"]
    D3 -> P7 [label="Soft actor match"]
    P7 -> D2 [label="ANN Search\n+ Filters"]
    D2 -> P7 [label="Retrieved Results"]
    P7 -> P8 [label="Ranked payload"]
    P8 -> Ext_User [label="JSON Output"]
}
"""

# ══════════════════════════════════════════════════════════════════════════════
# DFD Level 2
# ══════════════════════════════════════════════════════════════════════════════
dfd_2 = """
digraph DFD2 {
    fontname="Helvetica,Arial,sans-serif"
    node [fontname="Helvetica,Arial,sans-serif", shape=rect, fillcolor="#e2e8f0", style=filled, penwidth=1.5]
    edge [fontname="Helvetica,Arial,sans-serif", fontsize=10, penwidth=1.2, color="#374151"]
    rankdir=LR

    # External boundary inputs
    Input [label="Raw NL Query\n(From API Layer)", shape=cds, fillcolor="#bbf7d0"]
    Output [label="JSON Response\nResults", shape=cds, fillcolor="#bbf7d0"]

    subgraph cluster_P6 {
        label="P6: Query Parser Process"
        fontweight="bold"
        color="#2563eb"
        bgcolor="#eff6ff"
        
        node [shape=circle, fillcolor="#93c5fd"]
        P61 [label="P6.1\nIntent Gate"]
        P62 [label="P6.2\nYear Regex"]
        P63 [label="P6.3\nActor Extract"]
        P64 [label="P6.4\nSoft Intent\nAxes"]
        P65 [label="P6.5\nSafety Filter"]
        
        Input -> P61, P62, P63, P64, P65
    }

    subgraph cluster_data {
        label="Static Assets / Data"
        color="#a1a1aa"
        style="dashed"
        
        node [shape=box3d, fillcolor="#fde047"]
        DS_Centroid [label="Movie Intent\nCentroid"]
        DS_Vocab [label="Actor Vocab"]
        DS_Axes [label="Semantic\nAxes"]
        DS_ActorDB [label="actor_centroids.npy"]
        DS_Chroma [label="ChromaDB Index"]
        
        DS_Centroid -> P61
        DS_Vocab -> P63
        DS_Axes -> P64
    }

    subgraph cluster_P7 {
        label="P7: Retrieval Engine Process"
        fontweight="bold"
        color="#059669"
        bgcolor="#ecfdf5"
        
        node [shape=circle, fillcolor="#6ee7b7"]
        P71 [label="P7.1\nWhere Clause"]
        P72 [label="P7.2\nCentroid Fallback"]
        P73 [label="P7.3\nQuery Blending"]
        P74 [label="P7.4\nHNSW + Cluster"]
        P75 [label="P7.5\nScore & Verify"]
    }

    # Connections from P6 to P7
    P62 -> P71 [label="Year limits"]
    P63 -> P71 [label="Actor names"]
    P65 -> P71 [label="Adult filter"]
    
    P63 -> P72 [label="(If empty) trig"]
    DS_ActorDB -> P72
    
    P61 -> P73 [label="Base query"]
    P64 -> P73 [label="Soft phrase/conf"]
    
    P71 -> P74 [label="Where dict"]
    P73 -> P74 [label="Blended embed"]
    
    P74 -> DS_Chroma [label="Search request"]
    DS_Chroma -> P74 [label="Candidates"]
    
    P74 -> P75 [label="80 Candidates"]
    P75 -> Output [label="Top 20 reranked"]
}
"""

# ══════════════════════════════════════════════════════════════════════════════
# Architecture Diagram
# ══════════════════════════════════════════════════════════════════════════════
arch = """
digraph Architecture {
    fontname="Helvetica,Arial,sans-serif"
    node [fontname="Helvetica,Arial,sans-serif", shape=rect, fillcolor="#e2e8f0", style="rounded,filled", penwidth=1.5]
    edge [fontname="Helvetica,Arial,sans-serif", fontsize=10, penwidth=1.2, color="#374151"]
    rankdir=TB
    compound=true

    subgraph cluster_Offline {
        label="Offline Data Pipeline (One-Time Execution)"
        fontweight="bold"
        fontsize=14
        color="#d97706"
        bgcolor="#fffbeb"

        TMDB [label="TMDB REST API", shape=Mrecord, fillcolor="#fed7aa"]
        Ingest [label="tmdb_pipeline.py\n(Data Ingestion)"]
        Clean [label="data_clean/clean.py\n(Cleaning & Feature Eng)"]
        
        Embed [label="generate_embeddings.py\n(all-MiniLM-L6-v2)", fillcolor="#c4b5fd"]
        Cluster [label="models/cluster.py\n(MiniBatchKMeans k=150)", fillcolor="#86efac"]
        Actors [label="build_actor_centroids.py\n(Filmography Pooling)", fillcolor="#93c5fd"]
        
        DB_CSV [label="tmdb_movies_demo.csv\n(~45K Movies)", shape=cylinder, fillcolor="#fde047"]
        DB_NPY [label="movie_embeddings.npy", shape=cylinder, fillcolor="#fde047"]
        
        TMDB -> Ingest -> DB_CSV -> Clean
        Clean -> Embed -> DB_NPY -> Cluster
        Cluster -> Actors
    }

    subgraph cluster_Online {
        label="Online Inference Pipeline (Per-Query)"
        fontweight="bold"
        fontsize=14
        color="#2563eb"
        bgcolor="#eff6ff"
        
        FastAPI [label="FastAPI App\n(HTTP Router)", fillcolor="#fca5a5"]
        Parser [label="Query Parser Subsystem\n- Intent Centroid Gate\n- Regex Constraint Parsers\n- Semantic Soft Matching", fillcolor="#93c5fd"]
        Retrieval [label="Retrieval Engine\n- Query Blending\n- Actor Fallback", fillcolor="#86efac"]
        DB_Chroma [label="ChromaDB\n(HNSW Vector Store)", shape=cylinder, fillcolor="#fde047"]
        Reranker [label="Agglomerative Clustering\n+ Weighted Scoring", fillcolor="#c4b5fd"]
        
        FastAPI -> Parser [label="User Query"]
        Parser -> Retrieval [label="Structured Intent/Constraints"]
        Retrieval -> DB_Chroma [label="Blended Vec + Filters"]
        DB_Chroma -> Reranker [label="Top 80 Candidates"]
        Reranker -> FastAPI [label="Ranked Top 20 Payload"]
    }
    
    # Cross boundary
    DB_NPY -> DB_Chroma [label="chroma_ingest.py\n(Vector Indexing)", lhead=cluster_Online]
    Actors -> Retrieval [label="Actor Representations", style=dashed]
    Cluster -> DB_Chroma [label="Cluster Assignments", style=dashed]
}
"""

if __name__ == "__main__":
    print(f"Generating simple, functional DFDs to {OUT}...")
    render_dot("dfd_level0.png", dfd_0)
    render_dot("dfd_level1.png", dfd_1)
    render_dot("dfd_level2.png", dfd_2)
    render_dot("architecture_block.png", arch)
    print("Done generating DFDs.")
