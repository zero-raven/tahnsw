"""
corpus.py — Loads movies, embeds them, builds HNSW and TAHNSW (C++) indexes.
"""
import json
import os
import sys
import time
import numpy as np

# Add project root to path so tahnsw_cpp.so is importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import hnswlib
import tahnsw_cpp
from sentence_transformers import SentenceTransformer

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "movies.json")
MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384
M = 16
EF_CONSTRUCTION = 200
EF_SEARCH = 80
K = 5

# TAHNSW hyperparameters
M_ALPHA = 1.5
M_BETA  = 0.5
# Init with inflated M so memory is allocated for hub nodes (which get M*alpha edges)
M_INFLATED = int(M * M_ALPHA) + 2   # = 26


class SearchCorpus:
    def __init__(self, progress_callback=None):
        self.model = None
        self.movies = []
        self.embeddings = None
        self.hnsw_index = None
        self.tahnsw_index = None
        self.status = "idle"
        self.hnsw_build_time = 0.0
        self.tahnsw_build_time = 0.0
        self.tahnsw_stats = {}
        self._progress_callback = progress_callback

    def _emit(self, msg: str):
        self.status = msg
        if self._progress_callback:
            self._progress_callback(msg)

    def build(self):
        # 1. Load model
        self._emit("Loading sentence-transformer model…")
        self.model = SentenceTransformer(MODEL_NAME)

        # 2. Load corpus
        self._emit("Loading movie corpus…")
        with open(DATA_PATH) as f:
            self.movies = json.load(f)

        texts = [f"{m['title']}. {m['plot']}" for m in self.movies]
        n = len(texts)

        # 3. Embed
        self._emit(f"Embedding {n} documents…")
        self.embeddings = self.model.encode(
            texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        ).astype(np.float32)

        # 4. Build HNSW (standard hnswlib)
        self._emit("Building HNSW index…")
        t0 = time.perf_counter()
        self.hnsw_index = hnswlib.Index(space="cosine", dim=DIM)
        self.hnsw_index.init_index(max_elements=n, ef_construction=EF_CONSTRUCTION, M=M)
        self.hnsw_index.add_items(self.embeddings, list(range(n)))
        self.hnsw_index.set_ef(EF_SEARCH)
        self.hnsw_build_time = time.perf_counter() - t0

        # 5. Build TAHNSW (C++ topology-aware)
        self._emit("Building TAHNSW (C++) index…")
        t0 = time.perf_counter()

        # Use cosine space — IndexCpp normalises vectors internally for 'cosine'
        self.tahnsw_index = tahnsw_cpp.IndexCpp("cosine", DIM)

        # Allocate with inflated M so hub nodes (which get M*alpha edges) fit in memory
        self.tahnsw_index.init_index(
            max_elements=n,
            M=M_INFLATED,
            ef_construction=EF_CONSTRUCTION,
            random_seed=42,
            allow_replace_deleted=False,
        )

        # Configure TAHNSW — must be called AFTER init_index (appr_alg must exist)
        # but BEFORE add_items (so construction uses topology logic)
        self.tahnsw_index.set_tahnsw_config(
            cluster_high_thresh=0.70,
            cluster_low_thresh=0.20,
            M_alpha=M_ALPHA,
            M_beta=M_BETA,
            M_min=2,
            sketch_width=512,
            sketch_depth=5,
            sketch_S=30,
            prune_layer0_only=True,
            prune_angle_check=False,
            prune_angle_deg=60.0,
            k_candidates=20,
            b_max_window=5000,
            fast_construction=True,
            cluster_radius_mult=1.0,
            heuristic_alpha=1.0,
        )

        self.tahnsw_index.add_items(
            self.embeddings, np.arange(n, dtype=np.int64), num_threads=1
        )
        self.tahnsw_index.set_ef(EF_SEARCH)
        self.tahnsw_build_time = time.perf_counter() - t0

        # 6. Collect stats from C++ index
        raw = self.tahnsw_index.get_tahnsw_stats()
        hub_count  = int(raw["hub_count"])
        leaf_count = int(raw["leaf_count"])
        mid_count  = int(raw["mid_count"])
        edges_saved    = int(raw["edges_saved"])
        edges_baseline = int(raw["edges_baseline"])
        total = max(1, hub_count + leaf_count + mid_count)

        # Approximate mean M_eff from analytical formula (same as Python version)
        mean_M_eff = (
            hub_count  * (M * M_ALPHA) +
            leaf_count * (M * M_BETA) +
            mid_count  * M
        ) / total

        self.tahnsw_stats = {
            "hub_pct":  round(100.0 * hub_count  / total, 1),
            "leaf_pct": round(100.0 * leaf_count / total, 1),
            "normal_pct": round(100.0 * mid_count  / total, 1),
            "mean_M_eff": round(mean_M_eff, 2),
            "edge_reduction_pct": round(
                100.0 * edges_saved / max(1, edges_baseline), 1
            ),
        }

        self._emit("ready")

    def search(self, query: str, k: int = K):
        if self.status != "ready":
            raise RuntimeError("Index not ready")

        qvec = self.model.encode([query], normalize_embeddings=True).astype(np.float32)

        # HNSW search
        t0 = time.perf_counter()
        h_labels, h_dists = self.hnsw_index.knn_query(qvec, k=k)
        hnsw_latency_ms = (time.perf_counter() - t0) * 1000

        # TAHNSW search (C++)
        t0 = time.perf_counter()
        t_labels, t_dists = self.tahnsw_index.knn_query(qvec, k=k)
        tahnsw_latency_ms = (time.perf_counter() - t0) * 1000

        def fmt_results(labels, dists):
            results = []
            for label, dist in zip(labels[0], dists[0]):
                m = self.movies[int(label)]
                results.append({
                    "id": m["id"],
                    "title": m["title"],
                    "genre": m["genre"],
                    "plot": m["plot"],
                    "score": round(float(1 - dist), 4),
                })
            return results

        return {
            "hnsw": {
                "results": fmt_results(h_labels, h_dists),
                "latency_ms": round(hnsw_latency_ms, 3),
            },
            "tahnsw": {
                "results": fmt_results(t_labels, t_dists),
                "latency_ms": round(tahnsw_latency_ms, 3),
            },
        }

    def benchmark_data(self, ef_values=None, k: int = K):
        """Sweep ef_search and return recall-QPS curve data."""
        if self.status != "ready":
            raise RuntimeError("Index not ready")

        ef_values = ef_values or [10, 20, 40, 80, 120, 200, 300]
        n_queries = min(200, len(self.movies))
        rng = np.random.default_rng(42)
        query_idxs = rng.choice(len(self.movies), n_queries, replace=False)
        query_vecs = self.embeddings[query_idxs]

        # Ground truth: brute-force cosine (embeddings are already normalised)
        sims = query_vecs @ self.embeddings.T   # (n_queries, N)
        gt = np.argsort(-sims, axis=1)[:, :k]

        def sweep(index):
            curve = []
            for ef in ef_values:
                index.set_ef(ef)
                t0 = time.perf_counter()
                labels, _ = index.knn_query(query_vecs, k=k)
                elapsed = time.perf_counter() - t0
                qps = n_queries / elapsed
                hits = sum(
                    len(set(labels[i].tolist()) & set(gt[i].tolist()))
                    for i in range(n_queries)
                )
                recall = hits / (n_queries * k)
                curve.append({"ef": ef, "recall": round(recall, 4), "qps": round(qps, 1)})
            index.set_ef(EF_SEARCH)   # reset
            return curve

        return {
            "hnsw": {
                "curve": sweep(self.hnsw_index),
                "build_time": round(self.hnsw_build_time, 3),
            },
            "tahnsw": {
                "curve": sweep(self.tahnsw_index),
                "build_time": round(self.tahnsw_build_time, 3),
                "stats": self.tahnsw_stats,
            },
        }
