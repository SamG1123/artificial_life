# Demo: Hostable Chat-Only Wrapper

Quick instructions to run the lightweight demo server locally or in Docker.

Requirements
- Python 3.11+ and a virtualenv with dependencies installed (`requirements.txt`)
- Optional: Docker + Docker Compose for containerized run

Run locally (python venv)

```powershell
venv\Scripts\activate
python demo_server.py --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

Run with Docker

Build and run (single container):

```bash
docker build -t artificial-life-demo -f Dockerfile.fastapi .
docker run -p 8000:8000 -v $(pwd)/memory_store:/app/memory_store \
  -e GROQ_API_KEY=your_key_here artificial-life-demo
```

Run with Docker Compose (recommended for local dev)

```bash
GROQ_API_KEY=your_key_here docker compose up --build
```

FastAPI WebSocket

The repository now includes a `demo_server_fastapi.py` which provides a WebSocket endpoint at `/ws` for live streaming messages. The compose file builds the FastAPI image by default.

Production notes

- The FastAPI server supports optional API key protection via the `DEMO_API_KEY` environment variable. If set, clients must include the header `x-api-key: <value>` for HTTP requests and include `?api_key=<value>` in the WebSocket URL.
- To run a production container with API key protection:

```bash
DEMO_API_KEY=changeme docker compose up --build
```

CI / Automatic image publish

This repository includes a GitHub Actions workflow that builds and publishes a Docker image on push to `main` or when creating a matching tag. It publishes to GitHub Container Registry (GHCR) by default using `ghcr.io/${{ github.repository }}` and can optionally push to Docker Hub if you add the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets.

Required repository secrets to enable publishing:
- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` (optional) — push to Docker Hub
- The workflow uses `${{ secrets.GITHUB_TOKEN }}` to publish to GHCR; ensure `packages: write` permission is allowed for the workflow (the workflow file already requests it).

After the workflow runs you can pull the image:

```bash
docker pull ghcr.io/<owner>/<repo>:latest
# or (if Docker Hub configured)
docker pull <dockerhub-username>/artificial-life-demo:latest
```


Notes
- The demo uses `memory_store/` for persistence; the compose file mounts this directory as a volume so state survives container restarts.
- Set `GROQ_API_KEY` if your agent requires Groq access; the demo can run headless without it but LLM-backed features will be limited.
- For production, consider switching the demo server to FastAPI + `uvicorn` and adding TLS, authentication, and rate limiting.
