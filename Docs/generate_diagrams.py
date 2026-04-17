"""
Generates all diagrams for the Semantic Movie Recommendation System report.
Saves images to Docs/images/
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
BG   = "#0d1117"
CARD = "#161b22"
ACC1 = "#58a6ff"   # blue
ACC2 = "#3fb950"   # green
ACC3 = "#f78166"   # red/orange
ACC4 = "#d2a8ff"   # purple
ACC5 = "#ffa657"   # amber
GRAY = "#8b949e"
WHITE= "#e6edf3"
DARK = "#21262d"

def savefig(name, fig=None, dpi=200):
    path = os.path.join(OUT, name)
    (fig or plt).savefig(path, dpi=dpi, bbox_inches="tight",
                         facecolor=BG, edgecolor="none")
    plt.close("all")
    print(f"  ✓  {name}")

def styled_ax(fig, ax, title=""):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    if title:
        ax.set_title(title, color=WHITE, fontsize=13, fontweight="bold", pad=10)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: draw a rounded box with label
# ─────────────────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, color=ACC1, fontsize=9, text_color=BG, alpha=1.0, radius=0.03):
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle=f"round,pad={radius}",
                          facecolor=color, edgecolor="none", alpha=alpha, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold", zorder=4,
            multialignment="center", wrap=True)

def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.5, label="", style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0."), zorder=2)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx+0.01, my+0.01, label, color=GRAY, fontsize=7, ha="center", zorder=5)

def section(label):
    print(f"\n{'─'*50}\n  {label}\n{'─'*50}")


# ══════════════════════════════════════════════════════════════════════════════
#  1. DFD LEVEL 0  — Context Diagram
# ══════════════════════════════════════════════════════════════════════════════
section("DFD Level 0 – Context Diagram")

fig, ax = plt.subplots(figsize=(12, 7))
styled_ax(fig, ax, "Data Flow Diagram — Level 0: System Context")
ax.set_xlim(0, 10); ax.set_ylim(0, 7)

# Central process
box(ax, 5, 3.5, 2.8, 1.5,
    "Semantic Movie\nRecommendation\nSystem", color=ACC1, fontsize=11)

# External entities (rounded boxes in different colour)
entities = [
    (1.2, 5.8, "TMDB\nAPI",          ACC5),
    (1.2, 1.2, "End User\n(Client)", ACC2),
    (8.8, 5.8, "Admin /\nDevOps",    ACC4),
    (8.8, 1.2, "Docker /\nCloud",    ACC3),
]
for ex, ey, elabel, ecol in entities:
    box(ax, ex, ey, 1.8, 0.9, elabel, color=ecol, fontsize=9)

# Flows
flows = [
    (1.2, 5.3, 3.6, 4.0,  "Raw movie metadata",    ACC5),
    (3.6, 3.2, 1.2, 1.7,  "No data flow",          GRAY),
    (2.1, 1.2, 3.6, 3.0,  "Natural language query", ACC2),
    (6.4, 3.0, 7.9, 1.5,  "JSON recommendations",  ACC2),
    (7.9, 5.5, 6.4, 4.0,  "Config / Env vars",     ACC4),
    (6.4, 3.5, 7.9, 5.5,  "Logs / Health",         ACC4),
    (6.4, 3.2, 7.9, 1.5,  "Container image",       ACC3),
]
arrow(ax, 1.2, 5.3, 3.6, 4.1, ACC5, label="Raw movie metadata")
arrow(ax, 2.1, 1.5, 3.6, 3.0, ACC2, label="NL Query")
arrow(ax, 6.4, 3.0, 8.1, 1.5, ACC2, label="JSON results")
arrow(ax, 8.1, 5.5, 6.4, 4.1, ACC4, label="Config / env vars")
arrow(ax, 6.4, 3.5, 8.1, 5.3, ACC4, label="Logs / health status")
arrow(ax, 6.4, 3.2, 8.1, 1.5, ACC3, label="Deployed container")

# legend
patches = [mpatches.Patch(color=c, label=l) for c, l in
           [(ACC1,"System Process"),(ACC5,"TMDB API"),(ACC2,"End User"),
            (ACC4,"Admin/DevOps"),(ACC3,"Infrastructure")]]
ax.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
          facecolor=DARK, edgecolor=GRAY, labelcolor=WHITE, framealpha=0.9)

savefig("dfd_level0.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  2. DFD LEVEL 1  — Major Sub-Processes
# ══════════════════════════════════════════════════════════════════════════════
section("DFD Level 1 – Major Sub-Processes")

fig, ax = plt.subplots(figsize=(16, 10))
styled_ax(fig, ax, "Data Flow Diagram — Level 1: Major Sub-Processes")
ax.set_xlim(0, 16); ax.set_ylim(0, 10)

# Data stores (open rectangles)
def datastore(ax, x, y, w, h, label, color=GRAY):
    rx, ry = x-w/2, y-h/2
    ax.add_patch(plt.Rectangle((rx, ry), w, h,
                                facecolor=DARK, edgecolor=color, lw=2, zorder=3))
    ax.plot([rx, rx+w], [ry+h, ry+h], color=color, lw=2, zorder=4)
    ax.text(x, y, label, ha="center", va="center", fontsize=8,
            color=color, fontweight="bold", zorder=5)

# External entities
box(ax, 1.0, 8.5, 1.6, 0.8, "TMDB API",     ACC5, 9)
box(ax, 1.0, 1.5, 1.6, 0.8, "User Client",  ACC2, 9)
box(ax, 15,  5.0, 1.6, 0.8, "HTTP Client",  ACC2, 9)

# Processes
box(ax, 4.5, 8.5, 2.4, 0.9, "P1: Data\nIngestion",       ACC1, 9)
box(ax, 4.5, 6.5, 2.4, 0.9, "P2: Data Cleaning\n& Embedding",   ACC1, 9)
box(ax, 4.5, 4.5, 2.4, 0.9, "P3: Offline\nClustering",   ACC1, 9)
box(ax, 4.5, 2.5, 2.4, 0.9, "P4: Actor Centroid\nConstruction", ACC4, 9)
box(ax, 8.5, 7.5, 2.4, 0.9, "P5: Vector Store\nIngestion",   ACC1, 9)
box(ax, 8.5, 4.5, 2.4, 0.9, "P6: Query\nParser",          ACC2, 9)
box(ax, 8.5, 2.0, 2.4, 0.9, "P7: Retrieval\nEngine",      ACC3, 9)
box(ax, 12.5, 5.0, 2.4, 0.9, "P8: REST\nAPI Layer",       ACC5, 9)

# Data stores
datastore(ax, 6.8, 5.8, 2.0, 0.6, "D1: movie_embeddings.npy", ACC1)
datastore(ax, 6.8, 3.5, 2.0, 0.6, "D2: ChromaDB\n(HNSW Index)", ACC4)
datastore(ax, 6.8, 1.5, 2.0, 0.6, "D3: actor_centroids.npy",  ACC2)

# Arrows - offline pipeline
arrow(ax, 1.8, 8.5,  3.3, 8.5,   ACC5, label="API response JSON")
arrow(ax, 4.5, 8.05, 4.5, 6.95,  ACC1, label="raw CSV")
arrow(ax, 4.5, 6.05, 4.5, 4.95,  ACC1, label="cleaned + embeddings")
arrow(ax, 5.7, 6.5,  5.8, 5.8,   ACC1, label="embeddings")
arrow(ax, 5.7, 4.5,  5.8, 3.5,   ACC4, label="cluster_id")
arrow(ax, 5.8, 5.5,  7.4, 7.5,   ACC1, label="embeddings+metadata")
arrow(ax, 5.8, 3.2,  7.4, 7.3,   ACC4)
arrow(ax, 4.5, 4.05, 4.5, 2.95,  ACC4, label="embeddings")
arrow(ax, 5.7, 2.5,  5.8, 1.5,   ACC2, label="centroids")
arrow(ax, 9.7, 7.5,  11.0, 7.5,  ACC1)
arrow(ax, 11.0, 7.5, 11.0, 5.4,  ACC1, label="indexed")

# Arrows - online pipeline
arrow(ax, 1.8, 1.5,  7.4, 4.3,   ACC2, label="NL query")
arrow(ax, 9.7, 4.5,  11.0, 4.5,  ACC2, label="parsed intent")
arrow(ax, 9.7, 4.2,  7.4, 2.3,   ACC3)
arrow(ax, 7.8, 2.7,  7.8, 3.2,   ACC3, label="ChromaDB\nresults")
arrow(ax, 7.5, 1.5,  7.5, 2.0,   ACC2, label="centroids")
arrow(ax, 9.7, 2.0,  11.3, 4.6,  ACC3, label="scored results")
arrow(ax, 13.7, 5.0, 14.2, 5.0,  ACC2, label="HTTP response")
arrow(ax, 11.0, 4.6, 11.3, 5.0,  ACC5)
arrow(ax, 13.7, 5.0, 14.2, 5.0,  ACC2)

savefig("dfd_level1.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  3. DFD LEVEL 2  — Query Parser Deep-Dive
# ══════════════════════════════════════════════════════════════════════════════
section("DFD Level 2 – Query Parser (P6) Expanded")

fig, ax = plt.subplots(figsize=(16, 11))
styled_ax(fig, ax, "Data Flow Diagram — Level 2: Query Parser & Retrieval Engine (P6 / P7)")
ax.set_xlim(0, 16); ax.set_ylim(0, 11)

# External / boundary
box(ax, 1.0, 9.0, 1.6, 0.8, "User\nQuery",     ACC2, 9)
box(ax, 15,  5.5, 1.6, 0.8, "Retrieval\nEngine", ACC3, 9)

# Sub-processes of P6: Query Parser
box(ax, 4.5, 9.5, 2.8, 0.9, "P6.1: Intent Gate\n(Centroid Cosine)",  ACC4, 8.5)
box(ax, 4.5, 7.5, 2.8, 0.9, "P6.2: Year Constraint\nExtraction (Regex)", ACC1, 8.5)
box(ax, 4.5, 5.5, 2.8, 0.9, "P6.3: Actor\nExtraction",               ACC1, 8.5)
box(ax, 4.5, 3.5, 2.8, 0.9, "P6.4: Soft Intent\nInference (Semantic Axes)", ACC1, 8.5)
box(ax, 4.5, 1.5, 2.8, 0.9, "P6.5: Safety\nFilter",                  ACC5, 8.5)

# Sub-processes of P7: Retrieval Engine
box(ax, 10.5, 9.0, 2.8, 0.9, "P7.1: Build ChromaDB\nWhere Clause",   ACC1, 8.5)
box(ax, 10.5, 7.0, 2.8, 0.9, "P7.2: Actor Centroid\nGate (fallback)", ACC4, 8.5)
box(ax, 10.5, 5.0, 2.8, 0.9, "P7.3: Query Vector\nBlending",          ACC2, 8.5)
box(ax, 10.5, 3.0, 2.8, 0.9, "P7.4: HNSW ANN\n+ Agglomerative Clust.", ACC3, 8.5)
box(ax, 10.5, 1.0, 2.8, 0.9, "P7.5: Weighted Score\n+ Actor Enforce", ACC3, 8.5)

# Data stores
def ds(ax, x, y, label, color=GRAY):
    ax.add_patch(plt.Rectangle((x-1.3, y-0.3), 2.6, 0.6,
                                facecolor=DARK, edgecolor=color, lw=1.5, zorder=3))
    ax.plot([x-1.3, x+1.3], [y+0.3, y+0.3], color=color, lw=1.5, zorder=4)
    ax.text(x, y, label, ha="center", va="center", fontsize=7.5,
            color=color, fontweight="bold", zorder=5)

ds(ax, 7.8, 9.5, "Intent Centroid .npy", ACC4)
ds(ax, 7.8, 5.5, "actor_stats.csv",      ACC1)
ds(ax, 7.8, 3.5, "Semantic Axes",        ACC2)
ds(ax, 7.8, 7.0, "actor_centroids.npy",  ACC4)
ds(ax, 7.8, 1.0, "ChromaDB Index",       ACC3)

# Query input → all sub-processes
for ty in [9.5, 7.5, 5.5, 3.5, 1.5]:
    arrow(ax, 1.8, 9.0, 3.1, ty, ACC2)

# Intent centroid
arrow(ax, 7.8, 9.5, 5.9, 9.5, ACC4, label="centroid vec")

# Actor vocab
arrow(ax, 7.8, 5.5, 5.9, 5.5, ACC1, label="vocab list")

# Semantic axes
arrow(ax, 7.8, 3.5, 5.9, 3.5, ACC2, label="anchor phrases")

# Parser → P7 flows
arrow(ax, 5.9, 9.5, 9.1, 9.0, ACC4, label="intent=movie_search")
arrow(ax, 5.9, 7.5, 9.1, 9.0, ACC1, label="year constraints")
arrow(ax, 5.9, 5.5, 9.1, 9.0, ACC1, label="actor names")
arrow(ax, 5.9, 5.5, 9.1, 7.0, ACC4)
arrow(ax, 5.9, 3.5, 9.1, 5.0, ACC2, label="soft constraints")
arrow(ax, 5.9, 1.5, 9.1, 9.0, ACC5, label="allow_adult flag")

# P7 internal
arrow(ax, 9.1, 7.0, 7.8, 7.0, ACC4, label="centroids")
arrow(ax, 9.1, 1.0, 7.8, 1.0, ACC3, label="index query")
arrow(ax, 10.5, 8.55, 10.5, 7.45, ACC1, label="where clause")
arrow(ax, 10.5, 6.55, 10.5, 5.45, ACC4, label="centroid filter")
arrow(ax, 10.5, 4.55, 10.5, 3.45, ACC2, label="blended query vec")
arrow(ax, 10.5, 2.55, 10.5, 1.45, ACC3, label="candidates + labels")

# Out
arrow(ax, 11.9, 1.0, 14.2, 5.5, ACC3, label="final ranked results")

savefig("dfd_level2.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  4. SYSTEM ARCHITECTURE BLOCK DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
section("System Architecture Block Diagram")

fig, ax = plt.subplots(figsize=(18, 11))
styled_ax(fig, ax, "System Architecture — Semantic Movie Recommendation Engine")
ax.set_xlim(0, 18); ax.set_ylim(0, 11)

# ── Layer labels
def layer_label(ax, y, label, color):
    ax.text(0.2, y, label, color=color, fontsize=8, fontweight="bold",
            va="center", rotation=90, alpha=0.7)
    ax.axhline(y, color=color, lw=0.4, alpha=0.2, xmin=0.02, xmax=0.98)

# ── Offline Pipeline ──────────────────────────────────────
ax.add_patch(plt.Rectangle((0.5, 6.0), 8.5, 4.5,
             facecolor="#1f2937", edgecolor=ACC5, lw=1.5, alpha=0.5, zorder=0))
ax.text(4.75, 10.7, "OFFLINE DATA PIPELINE (one-time)", color=ACC5,
        ha="center", fontsize=10, fontweight="bold")

off_boxes = [
    (1.6,  8.4, "TMDB\nAPI", ACC5),
    (3.4,  8.4, "tmdb_pipeline.py\nP1: Ingestion", ACC1),
    (5.5,  8.4, "data_clean/clean.py\nP2: NLP Feature Eng.", ACC1),
    (7.8,  9.2, "generate_embeddings.py\nP3: MiniLM-L6-v2 (384d)", ACC4),
    (7.8,  7.6, "build_intent_centroid.py\nP3b: Intent Centroid", ACC4),
    (5.5,  6.7, "cluster.py\nP4: MiniBatchKMeans K=150", ACC1),
    (7.8,  6.3, "build_actor_centroids.py\nP5: 4300 Actor Centroids", ACC2),
]
for bx, by, bl, bc in off_boxes:
    box(ax, bx, by, 1.9, 0.7, bl, bc, 7.5)

# Data stores offline
ds_off = [
    (3.4,  6.7, "tmdb_movies_demo.csv\n~45K records"),
    (5.5,  7.6, "movie_embeddings.npy\n(45K×384)"),
    (5.5,  9.2, "tmdb_cleaned.csv"),
]
for dx, dy, dl in ds_off:
    ax.add_patch(plt.Rectangle((dx-1.1, dy-0.35), 2.2, 0.7,
                                facecolor=DARK, edgecolor=GRAY, lw=1, zorder=3))
    ax.plot([dx-1.1, dx+1.1], [dy+0.35, dy+0.35], color=GRAY, lw=1, zorder=4)
    ax.text(dx, dy, dl, ha="center", va="center", fontsize=7, color=GRAY, zorder=5)

arrow(ax, 2.55, 8.4, 2.85, 8.4, ACC5)
arrow(ax, 4.35, 8.4, 4.45, 9.2, ACC1)
arrow(ax, 4.35, 8.4, 4.45, 7.6, ACC1)
arrow(ax, 4.35, 8.4, 4.45, 8.4, ACC1)   # → cleaned CSV
arrow(ax, 6.55, 9.2, 6.85, 9.2, ACC1)
arrow(ax, 6.55, 7.6, 6.85, 7.6, ACC4)
arrow(ax, 6.55, 8.4, 6.55, 6.7, ACC1)
arrow(ax, 6.55, 6.7, 6.85, 6.3, ACC2)

# ── Online Pipeline ──────────────────────────────────────
ax.add_patch(plt.Rectangle((0.5, 0.3), 17.0, 5.5,
             facecolor="#1a2332", edgecolor=ACC1, lw=1.5, alpha=0.5, zorder=0))
ax.text(9.0, 6.0, "ONLINE INFERENCE PIPELINE (per query)", color=ACC1,
        ha="center", fontsize=10, fontweight="bold")

on_boxes = [
    (1.5, 3.5, "Client\n(HTTP POST)", ACC2),
    (3.5, 4.8, "FastAPI\nRouter", ACC5),
    (3.5, 3.5, "P6.1: Intent\nGate", ACC4),
    (3.5, 2.2, "P6.2: Year +\nActor Extract", ACC1),
    (6.0, 4.5, "P6.3: Soft Intent\nInference", ACC1),
    (6.0, 2.8, "P6.4: Safety\nFilter", ACC5),
    (8.8, 4.5, "P7.1: Where\nClause Builder", ACC1),
    (8.8, 2.8, "P7.2: Centroid\nGate (fallback)", ACC4),
    (11.5, 4.5, "P7.3: Query Vec\nBlending", ACC2),
    (11.5, 2.8, "ChromaDB\nHNSW ANN", ACC3),
    (14.0, 4.5, "P7.4: Aggl.\nClustering (k=4)", ACC3),
    (14.0, 2.8, "P7.5: Weighted\nScoring", ACC3),
    (16.5, 3.5, "JSON\nResponse", ACC2),
]
for bx, by, bl, bc in on_boxes:
    box(ax, bx, by, 1.9, 0.75, bl, bc, 7.5)

# Key arrows - online pipeline
arrow(ax, 2.45, 3.5,  2.55, 3.5,  ACC2)
arrow(ax, 2.55, 3.5,  2.55, 4.8,  ACC2)
arrow(ax, 2.55, 4.8,  2.55, 3.5,  ACC2)
arrow(ax, 2.5,  4.8,  2.55, 4.8,  ACC2)
arrow(ax, 4.45, 4.8,  5.05, 4.5,  ACC5)
arrow(ax, 4.45, 3.5,  5.05, 4.5,  ACC4)
arrow(ax, 4.45, 2.2,  5.05, 2.8,  ACC1)
arrow(ax, 7.05, 4.5,  7.85, 4.5,  ACC1)
arrow(ax, 7.05, 2.8,  7.85, 2.8,  ACC5)
arrow(ax, 7.85, 2.8,  7.85, 4.5,  ACC4)
arrow(ax, 9.75, 4.5, 10.55, 4.5,  ACC1)
arrow(ax, 9.75, 2.8, 10.55, 2.8,  ACC4)
arrow(ax, 10.55,2.8, 10.55, 4.5,  ACC3)
arrow(ax,12.45, 4.5, 13.05, 4.5,  ACC2)
arrow(ax,12.45, 2.8, 13.05, 2.8,  ACC3)
arrow(ax,14.95, 4.5, 14.95, 2.8,  ACC3)
arrow(ax,14.95, 2.8, 15.55, 3.5,  ACC3)
arrow(ax,15.55, 3.5, 15.55, 3.5,  ACC2)

# Vector stores used online
ds2 = [
    (8.8, 1.2, "actor_centroids.npy"),
    (11.5, 1.2, "Semantic Axes Cache"),
    (14.0, 1.2, "cluster_stats.csv"),
]
for dx, dy, dl in ds2:
    ax.add_patch(plt.Rectangle((dx-1.1, dy-0.3), 2.2, 0.6,
                                facecolor=DARK, edgecolor=GRAY, lw=1, zorder=3))
    ax.plot([dx-1.1, dx+1.1], [dy+0.3, dy+0.3], color=GRAY, lw=1, zorder=4)
    ax.text(dx, dy, dl, ha="center", va="center", fontsize=7.5, color=GRAY, zorder=5)
    arrow(ax, dx, dy+0.3, dx, dy+0.85, GRAY)

savefig("architecture_block.png", fig)

print("\nPhase 1 complete.")
