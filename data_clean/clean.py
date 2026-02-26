"""
TMDB Data Cleaning Script
------------------------

This script performs conservative preprocessing on the raw TMDB dataset
to prepare it for:
- semantic embeddings
- clustering (HDBSCAN)
- safety analysis
- downstream recommendation tasks

Design principles:
- Do NOT over-clean text (transformers handle noise well)
- Preserve semantic richness
- Enforce safety-critical consistency
- Remove only genuinely unusable rows
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_CSV = os.path.join(BASE_DIR, "tmdb_movies_demo.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data_clean", "tmdb_cleaned.csv")

MIN_OVERVIEW_LENGTH = 30  # characters

# ------------------ LOAD ------------------

print("[INFO] Loading dataset...")
df = pd.read_csv(INPUT_CSV)
csv_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
# writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
# if csv_file.tell() == 0:
#     writer.writeheader()
print(f"[INFO] Initial dataset size: {df.shape}")

# ------------------ BASIC SANITY ------------------

# Drop exact duplicate TMDB IDs (should be rare but safe)
df = df.drop_duplicates(subset=["tmdb_id"])

# ------------------ TEXT CLEANING ------------------

def clean_text(text: str) -> str:
    """
    Conservative text cleaning.
    DO NOT stem or lemmatize.
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.strip().lower()
    return text

# Overview (most important field)
df["overview"] = df["overview"].fillna("").apply(clean_text)

# Drop movies with no usable overview
df = df[df["overview"].str.len() >= MIN_OVERVIEW_LENGTH]

# Title
df["title"] = df["title"].fillna("").apply(clean_text)
df["original_title"] = df["original_title"].fillna("").apply(clean_text)

# Tagline (optional but useful)
df["tagline"] = df["tagline"].fillna("").apply(clean_text)

# ------------------ CATEGORICAL FIELDS ------------------

df["genres"] = df["genres"].fillna("").str.lower().str.strip()
df["original_language"] = df["original_language"].fillna("unknown")

df["spoken_languages"] = (
    df["spoken_languages"]
    .fillna("")
    .str.lower()
    .str.strip()
)

df["production_countries"] = (
    df["production_countries"]
    .fillna("")
    .str.upper()
    .str.strip()
)

df["collection_name"] = (
    df["collection_name"]
    .fillna("")
    .apply(clean_text)
)

# ------------------ NUMERIC FIELDS ------------------

numeric_cols = [
    "runtime_minutes",
    "popularity",
    "vote_average",
    "vote_count"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Runtime: replace impossible values
df.loc[df["runtime_minutes"] <= 0, "runtime_minutes"] = np.nan

# Fill numeric NaNs with medians (robust)
for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# ------------------ DATE FIELDS ------------------

df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
df["release_month"] = pd.to_numeric(df["release_month"], errors="coerce")

# Drop rows without year (very rare, but unusable)
df = df.dropna(subset=["release_year"])

df["release_year"] = df["release_year"].astype(int)
df["release_month"] = df["release_month"].fillna(0).astype(int)

# ------------------ SAFETY FIELD ------------------

df["adult"] = df["adult"].fillna(False).astype(bool)

# ------------------ EMBEDDING TEXT ------------------
# This is the EXACT text used later for embeddings & clustering

print("[INFO] Constructing embedding text...")

df["embedding_text"] = (
    df["title"] + ". " +
    df["overview"] + ". " +
    "genres: " + df["genres"] + ". " +
    "language: " + df["original_language"] + ". " +
    "year: " + df["release_year"].astype(str)
)

# ------------------ FINAL SANITY CHECK ------------------

assert df["embedding_text"].isnull().sum() == 0
assert df["tmdb_id"].isnull().sum() == 0

# ------------------ SAVE ------------------

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

df.to_csv(OUTPUT_CSV, index=False)

print(f"[DONE] Cleaned dataset saved to: {OUTPUT_CSV}")
print(f"[DONE] Final dataset size: {df.shape}")
