# TAHNSW — Topology-Aware HNSW

Full Python implementation of TAHNSW built on top of `hnswlib`.

## Three Contributions

| # | What changes | How |
|---|---|---|
| 1 | Layer assignment | Replace random draw with clustering coefficient C(v) |
| 2 | Adaptive degree M | Replace fixed M=16 with M(v) from betweenness centrality |
| 3 | RNG edge pruning | Remove dominated edges at Layer 0 after neighbor selection |

## Install

```bash
pip install hnswlib numpy matplotlib h5py
```

## Quick Start

```python
import numpy as np
from tahnsw import TAHNSWIndex, TAHNSWConfig

# Configure
cfg = TAHNSWConfig(
    cluster_high_thresh = 0.70,   # C(v) > 0.70  →  leaf node  →  Layer 0
    cluster_low_thresh  = 0.20,   # C(v) < 0.20  →  hub  node  →  high layer
    M_alpha             = 1.5,    # hub  gets M × 1.5 connections
    M_beta              = 0.5,    # leaf gets M × 0.5 connections
)

# Build index
data = np.random.randn(50_000, 128).astype(np.float32)
idx = TAHNSWIndex(dim=128, max_elements=50_000, config=cfg)
idx.init_index(M=16, ef_construction=200)
idx.add_items(data)

# Search — identical to hnswlib
queries = np.random.randn(100, 128).astype(np.float32)
labels, dists = idx.knn_query(queries, k=10)
```

## Benchmark

```bash
# Quick test on synthetic clustered data (no downloads needed)
python benchmark.py --mode synthetic --N 30000 --dim 128 --k 10

# Full benchmark on SIFT-128 (download first)
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
python benchmark.py --mode hdf5 --dataset sift-128-euclidean.hdf5 --k 10
```

## Hyperparameter Tuning Guide

| Parameter | Default | What it does | Range to try |
|---|---|---|---|
| `cluster_high_thresh` | 0.70 | C(v) above = leaf, Layer 0 only | 0.60–0.80 |
| `cluster_low_thresh` | 0.20 | C(v) below = hub, high layer | 0.10–0.30 |
| `M_alpha` | 1.5 | Hub degree multiplier | 1.3–2.0 |
| `M_beta` | 0.5 | Leaf degree fraction | 0.4–0.7 |
| `sketch_S` | 30 | BFS samples for centrality | 20–100 |
| `k_candidates` | 20 | Neighborhood for C(v) | 10–30 |

## File Structure

```
tahnsw/
├── tahnsw.py         # Main implementation — TAHNSWIndex + all modules
├── benchmark.py      # TAHNSW vs HNSW benchmark runner
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Algorithm Complexity

| Operation | HNSW | TAHNSW |
|---|---|---|
| Insert one node | O(M · log N) | O(M_eff · log N + k²) |
| Build N nodes | O(N · M · log N) | O(N · M_eff · log N) |
| Search one query | O(log N) | O(log N) — unchanged |
| Space | O(N · M) | O(N · M_eff) |

Where `M_eff ≈ 0.75 × M` on clustered data → ~25% construction speedup.
The `k²` term is a constant (k=20 is fixed) and does not grow with N.

## Running the Application
Start the backend:

pip install -r api/requirements.txt
uvicorn api.main:app --reload

Start the frontend:

npm i
cd ui
