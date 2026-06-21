# TAHNSW — Topology-Aware HNSW

> A custom semantic search algorithm that improves upon HNSW by using **clustering coefficients** and **betweenness centrality** for smarter graph construction — with a full-stack showcase app to compare TAHNSW vs standard HNSW live.

---

## Project Structure

```
el/
├── tahnsw.py              # Core TAHNSW algorithm implementation
├── benchmark.py           # CLI benchmark runner (TAHNSW vs HNSW)
├── requirements.txt       # Python deps for standalone algorithm use
│
├── api/                   # FastAPI backend (semantic search server)
│   ├── main.py            # API entrypoint (FastAPI app + endpoints)
│   ├── corpus.py          # Loads movies, embeds them, builds both indexes
│   ├── requirements.txt   # Backend Python dependencies
│   └── data/
│       └── movies.json    # Movie corpus used for semantic search
│
└── ui/                    # Next.js frontend (interactive showcase)
    ├── app/               # Next.js app directory
    ├── .env.local         # Frontend environment variables
    └── package.json       # Node dependencies
```

---

## Three Core Contributions of TAHNSW

| # | What changes | How |
|---|---|---|
| 1 | **Layer assignment** | Replace random draw with clustering coefficient C(v) |
| 2 | **Adaptive degree M** | Replace fixed M=16 with M(v) from betweenness centrality |
| 3 | **RNG edge pruning** | Remove dominated edges at Layer 0 after neighbor selection |

---

## Prerequisites

- **Python 3.10+** with `pip`
- **Node.js 18+** with `npm`
- A virtual environment is recommended (the repo ships a `.venv/` folder)

---

## Running the Full Stack App

The project has two parts that run simultaneously:

### 1. Backend — FastAPI Server

```bash
# From the project root: /home/Kaiser/clg/el

# Activate the virtual environment (if using .venv)
source .venv/bin/activate

# Install backend dependencies (first time only)
pip install -r api/requirements.txt

# Start the API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at **http://localhost:8000**

> **Note:** On first startup, the server automatically:
> 1. Downloads the `all-MiniLM-L6-v2` sentence-transformer model (~90 MB)
> 2. Embeds all movies in the corpus
> 3. Builds both the HNSW and TAHNSW indexes
>
> This takes a minute. The `/api/status/stream` SSE endpoint streams real-time progress to the frontend.

#### Available API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Current build status |
| `GET` | `/api/status/stream` | SSE stream of build progress |
| `POST` | `/api/search` | Run a semantic search query |
| `GET` | `/api/benchmark` | Recall–QPS curve data |
| `GET` | `/api/info` | Corpus size, build times, TAHNSW stats |

---

### 2. Frontend — Next.js App

Open a **new terminal**:

```bash
# Navigate to the UI directory
cd ui/

# Install Node dependencies (first time only)
npm install

# Start the dev server
npm run dev
```

The frontend will be available at **http://localhost:3000**

The `.env.local` file already points the frontend at the backend:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Standalone Algorithm Usage

Use TAHNSW directly in Python without the web app:

```python
import numpy as np
from tahnsw import TAHNSWIndex, TAHNSWConfig

cfg = TAHNSWConfig(
    cluster_high_thresh = 0.70,   # C(v) > 0.70  →  leaf node  →  Layer 0
    cluster_low_thresh  = 0.20,   # C(v) < 0.20  →  hub  node  →  high layer
    M_alpha             = 1.5,    # hub  gets M × 1.5 connections
    M_beta              = 0.5,    # leaf gets M × 0.5 connections
)

data = np.random.randn(50_000, 128).astype(np.float32)
idx = TAHNSWIndex(dim=128, max_elements=50_000, config=cfg)
idx.init_index(M=16, ef_construction=200)
idx.add_items(data)

queries = np.random.randn(100, 128).astype(np.float32)
labels, dists = idx.knn_query(queries, k=10)
```

Install standalone deps:
```bash
pip install -r requirements.txt
```

---

## Running the Benchmark

```bash
# Quick synthetic benchmark (no downloads needed)
python benchmark.py --mode synthetic --N 30000 --dim 128 --k 10

# Full benchmark on SIFT-128 dataset (download first)
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
python benchmark.py --mode hdf5 --dataset sift-128-euclidean.hdf5 --k 10
```

---

## Hyperparameter Reference

| Parameter | Default | What it does | Range to try |
|---|---|---|---|
| `cluster_high_thresh` | 0.70 | C(v) above = leaf, Layer 0 only | 0.60–0.80 |
| `cluster_low_thresh` | 0.20 | C(v) below = hub, high layer | 0.10–0.30 |
| `M_alpha` | 1.5 | Hub degree multiplier | 1.3–2.0 |
| `M_beta` | 0.5 | Leaf degree fraction | 0.4–0.7 |
| `sketch_S` | 30 | BFS samples for centrality | 20–100 |
| `k_candidates` | 20 | Neighborhood for C(v) | 10–30 |

---

## Algorithm Complexity

| Operation | HNSW | TAHNSW |
|---|---|---|
| Insert one node | O(M · log N) | O(M_eff · log N + k²) |
| Build N nodes | O(N · M · log N) | O(N · M_eff · log N) |
| Search one query | O(log N) | O(log N) — unchanged |
| Space | O(N · M) | O(N · M_eff) |

Where `M_eff ≈ 0.75 × M` on clustered data → ~25% construction speedup.
The `k²` term is a constant (k=20 is fixed) and does not grow with N.

---

## Quick Checklist

| Step | Command | Port |
|---|---|---|
| Start backend | `uvicorn api.main:app --reload --port 8000` | 8000 |
| Start frontend | `cd ui && npm run dev` | 3000 |
| Open app | Browser → `http://localhost:3000` | — |
