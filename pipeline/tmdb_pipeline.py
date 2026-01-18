import os
import csv
import signal
import sys
import time
import calendar
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import date

load_dotenv()

BASE_URL = "https://api.themoviedb.org/3"
API_KEY = os.getenv("TMDB_API_KEY")

OUTPUT_FILE = "tmdb_movies_demo.csv"

FIELDNAMES = [
    "tmdb_id",
    "title",
    "original_title",
    "overview",
    "tagline",
    "release_date",
    "release_year",
    "release_month",
    "original_language",
    "spoken_languages",
    "genres",
    "runtime_minutes",
    "popularity",
    "vote_average",
    "vote_count",
    "adult",
    "cast",
    "director",
    "production_countries",
    "collection_name",
    "retrieval_month"
]

# ------------------ HTTP SESSION ------------------

session = requests.Session()

retries = Retry(
    total=5,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)


def make_request(endpoint, params=None):
    if params is None:
        params = {}

    params["api_key"] = API_KEY

    response = session.get(
        f"{BASE_URL}{endpoint}",
        params=params,
        timeout=30
    )
    response.raise_for_status()
    return response.json()

# ------------------ CSV SETUP ------------------

csv_file = open(OUTPUT_FILE, "a", newline="", encoding="utf-8")
writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)

if csv_file.tell() == 0:
    writer.writeheader()

def handle_exit(signum, frame):
    print("\n[INFO] Graceful shutdown requested. Saving progress...")
    csv_file.flush()
    csv_file.close()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# ------------------ TMDB LOGIC ------------------

def discover_movies(start_date, end_date, max_items=200):
    movies = []
    page = 1

    while len(movies) < max_items:
        params = {
            "primary_release_date.gte": start_date,
            "primary_release_date.lte": end_date,
            "sort_by": "popularity.desc",
            "page": page,
            "include_adult": "true"
        }

        data = make_request("/discover/movie", params)
        results = data.get("results", [])

        if not results:
            break

        for m in results:
            movies.append(m["id"])
            if len(movies) >= max_items:
                break

        page += 1

    return list(set(movies))


def fetch_movie_details(movie_id):
    movie = make_request(f"/movie/{movie_id}")
    credits = make_request(f"/movie/{movie_id}/credits")

    cast = [c["name"] for c in credits.get("cast", [])[:5]]
    director = next(
        (c["name"] for c in credits.get("crew", [])
         if c["job"] == "Director"),
        None
    )

    return {
        "tmdb_id": movie["id"],
        "title": movie["title"],
        "original_title": movie["original_title"],
        "overview": movie.get("overview", ""),
        "tagline": movie.get("tagline", ""),
        "release_date": movie.get("release_date"),
        "release_year": int(movie["release_date"][:4]) if movie.get("release_date") else None,
        "release_month": int(movie["release_date"][5:7]) if movie.get("release_date") else None,
        "original_language": movie.get("original_language"),
        "spoken_languages": ",".join(
            l["iso_639_1"] for l in movie.get("spoken_languages", [])
        ),
        "genres": ",".join(g["name"] for g in movie.get("genres", [])),
        "runtime_minutes": movie.get("runtime"),
        "popularity": movie.get("popularity"),
        "vote_average": movie.get("vote_average"),
        "vote_count": movie.get("vote_count"),
        "adult": movie.get("adult", False),
        "cast": ",".join(cast),
        "director": director,
        "production_countries": ",".join(
            c["iso_3166_1"] for c in movie.get("production_countries", [])
        ),
        "collection_name": (
            movie["belongs_to_collection"]["name"]
            if movie.get("belongs_to_collection")
            else None
        )
    }

# ------------------ MAIN PIPELINE ------------------

def build_dataset(start_year=2012, years=13):
    seen_ids = set()

    for year in range(start_year, start_year + years):
        for month in range(1, 13):
            last_day = calendar.monthrange(year, month)[1]
            start = date(year, month, 1)
            end = date(year, month, last_day)

            ids = discover_movies(start.isoformat(), end.isoformat())

            for movie_id in tqdm(ids, desc=f"{year}-{month:02d}"):
                if movie_id in seen_ids:
                    continue

                try:
                    row = fetch_movie_details(movie_id)
                    row["retrieval_month"] = f"{year}-{month:02d}-01"
                    writer.writerow(row)
                    csv_file.flush()
                    seen_ids.add(movie_id)
                    time.sleep(0.1)  # throttle
                except Exception as e:
                    print(f"[WARN] Failed movie {movie_id}: {e}")
                    continue

    print(f"[DONE] Dataset build completed.")

if __name__ == "__main__":
    build_dataset()
