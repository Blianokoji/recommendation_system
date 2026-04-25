# Linode Deployment Guide

This guide details how to deploy the **Semantic Movie Recommendation System** to a Linode Virtual Machine (Ubuntu 22.04 LTS recommended).

## 1. Provision Your Linode
- Log in to your [Linode Cloud Manager](https://cloud.linode.com/).
- Click **Create Linode**.
- **Image**: Ubuntu 22.04 LTS
- **Region**: Choose the one closest to your users.
- **Plan**: Shared CPU -> Linode 4GB or 8GB (Recommended, as embedding and clustering are memory-intensive).
- **Label**: `movie-rec-api`
- **Root Password**: Set a secure password.
- Click **Create Linode**.

## 2. Connect and Install Docker
Once your Linode is "Running", SSH into it:
```bash
ssh root@<YOUR_LINODE_IP>
```

Install Docker:
```bash
# Update packages
apt-get update
apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Verify installation
docker --version
```

## 3. Clone and Configure
Clone your repository:
```bash
git clone <YOUR_REPO_URL>
cd recommendation_system
```

Create your `.env` file on the server:
```bash
nano .env
```
Paste your keys:
```text
TMDB_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
PORT=8000
```
*(Press Ctrl+O, Enter, Ctrl+X to save and exit)*

## 4. Build and Run
The `start.sh` script inside the container will automatically run the `build_pipeline.py` to generate the embeddings, clusters, and database from the seed CSV.

```bash
# Build the image
docker build -t movie-rec-api .

# Run the container
# We map host port 80 to container port 8000
docker run -d \
  --name movie-rec-app \
  -p 80:8000 \
  --env-file .env \
  --restart unless-stopped \
  movie-rec-api
```

## 5. Monitor Initialization
The first run will take a few minutes as it builds the ML artifacts. Watch the logs:
```bash
docker logs -f movie-rec-app
```
Once you see `Application startup complete.`, the system is live!

## 6. Access the API
- **Base URL**: `http://<YOUR_LINODE_IP>/`
- **Docs**: `http://<YOUR_LINODE_IP>/docs`
- **Test Query**: 
  ```bash
  curl -X POST "http://<YOUR_LINODE_IP>/retrieve/core" \
       -H "Content-Type: application/json" \
       -d '{"query": "emotional sci-fi"}'
  ```
