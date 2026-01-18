<div align="center">

# 🎬 Movie Recommendation System

### Intelligent Semantic Search with Hard Constraints & Diversity

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Sentence_Transformers-Embeddings-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Clustering-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

</div>

---

## 📖 Overview

This project is a sophisticated **Hybrid Movie Recommendation Engine** that combines the power of **Semantic Vector Search** with **Structured Hard Constraints**. Unlike simple KNN lookups, this system understands query intent, enforces strict filters (like specific actors), and ensures result diversity through local clustering.

## ✨ Key Features

<table>
  <tr>
    <td width="50%">
      <h3>🧠 Semantic Understanding</h3>
      <p>Uses <b>Sentence Transformers</b> (<code>all-MiniLM-L6-v2</code>) to understand the "vibe" or plot description of a query (e.g., <i>"emotional movies about space"</i>).</p>
    </td>
    <td width="50%">
      <h3>🛡️ Hard Constraints</h3>
      <p>Strictly enforces filters like <b>Actor Name</b> presence (via <code>metadata</code> & <code>document</code> filtering) and <b>Adult Content</b> safety.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🧩 Diversity Clustering</h3>
      <p>Applies <b>Agglomerative Clustering</b> on retrieved candidates to prevent result monotony (e.g., returning 10 similar sequels).</p>
    </td>
    <td width="50%">
      <h3>⚡ Efficient Retrieval</h3>
      <p>Powered by <b>ChromaDB</b> for high-performance vector similarity search with metadata filtering.</p>
    </td>
  </tr>
</table>

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/recommendation-system.git
cd recommendation-system
```

### 2. Set up Environment (using `uv`)
This project uses `uv` for blazing fast dependency management.

```bash
# Initialize venv and install dependencies
uv sync

# Or manually:
uv pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```bash
TMDB_API_KEY=your_api_key_here
```

## 🛠️ Usage

### 📊 Data Pipeline (Ingestion)
To process the dataset and populate the ChromaDB vector store:

```bash
# Clean data, generate embeddings, and ingest into ChromaDB
uv run vector_store/chroma_ingest.py
```
> **Note:** The ingestion script reads from `clustering/tmdb_clustered_incremental.csv` and builds the index with title/cast metadata.

### 🔍 Run Retrieval Demo
To test the recommendation engine with a sample query:

```bash
uv run retrieval/retrieve_candidates.py
```
*Current Query in script:* `"emotional Tom Cruise movies"`

## 📂 Project Structure

```bash
recommendation_system/
├── 📂 vector_store/       # ChromaDB client & ingestion logic
│   ├── chroma_ingest.py
│   └── chroma_client.py
├── 📂 retrieval/          # Core retrieval & filtering logic
│   └── retrieve_candidates.py
├── 📂 query_parser/       # Intent extraction & query understanding
├── 📂 embeddings/         # Embedding generation scripts
├── 📂 models/             # Clustering models (KMeans, etc.)
├── 📂 data_clean/         # Data cleaning utilities
├── 📄 requirements.txt    # Project dependencies
└── 📄 README.md           # Documentation
```

## 🧠 How It Works

1.  **Query Parsing**: The system parses the user string to extract **Intent** (e.g., `movie_search`), **Hard Constraints** (e.g., `actor: "Tom Cruise"`), and **Soft Constraints** (semantic phrases).
2.  **Vector Search**: It queries ChromaDB using the semantic embedding of the soft constraints.
3.  **Hard Filtering**: Simultanously applies rigorous filters (using `where_document`) to ensure required entities (actors) are present.
4.  **Clustering**: Results are clustered locally to select a diverse set of top candidates.

---

