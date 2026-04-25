# Render Deployment Guide (Free Tier Optimized)

This guide helps you deploy the **Semantic Movie Recommendation System** to Render.com using their **Free Tier**.

## 1. Prepare Your Repository
Ensure your `master` branch is up to date and contains:
- `Dockerfile` (Updated for Render build-time ingestion)
- `start.sh`
- `pyproject.toml` & `uv.lock`
- `tmdb_movies_demo.csv` (The seed data)
- All the logic files (`api/`, `retrieval/`, `slm/`, `query_parser/`, etc.)

## 2. Push to GitHub
```bash
git add .
git commit -m "Optimize for Render deployment"
git push origin master
```

## 3. Create a New Web Service on Render
1. Go to [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. **Service Name**: `movie-recommendation-api`
5. **Environment**: `Docker`
6. **Region**: Choose one (e.g., Oregon - US West).
7. **Instance Type**: `Free` (512 MB RAM).

## 4. Environment Variables
In the **Advanced** or **Environment** section, add the following:
- `TMDB_API_KEY`: your_key_here
- `GEMINI_API_KEY`: your_key_here
- `PYTHONUTF8`: `1`

*(The `PORT` variable is automatically handled by Render and our `start.sh`)*

## 5. Deployment
- Click **Create Web Service**.
- **Build Phase**: Render will build the Docker image. During this phase, it will run `build_pipeline.py`. This step might take 3-5 minutes as it generates the vector index.
- **Runtime**: Once the build starts, Render will deploy the container. Because the database is "baked in", the API will start instantly.

## 6. Access Your API
- **URL**: `https://movie-recommendation-api.onrender.com`
- **Docs**: `https://movie-recommendation-api.onrender.com/docs`

---

### Why this works for the Free Tier:
- **Build-Time Ingestion**: By running the pipeline during the Docker build, we use Render's build-time resources (which are more generous) rather than the restricted runtime memory (512MB).
- **Fast Startup**: The system doesn't need to rebuild the world every time it wakes up from sleep.
- **Zero Cost**: Render Free tier is completely free as long as you don't mind the 15-minute inactivity spin-down.
