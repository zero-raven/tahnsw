"""
benchmark.py  —  TAHNSW vs HNSW recall-QPS benchmark
======================================================
Runs both algorithms on synthetic SIFT-like data and on real HDF5
datasets from ann-benchmarks.com if available.

Usage:
    # Quick test on synthetic data (no downloads needed):
    python benchmark.py --mode synthetic --N 50000 --dim 128 --k 10

    # Full benchmark on SIFT-128 (download first):
    # wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
    python benchmark.py --mode hdf5 --dataset sift-128-euclidean.hdf5

Results are saved to results/benchmark_results.json and a recall-QPS plot.
"""

import argparse
import json
import os
import time
import numpy as np
import sys



try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_synthetic(N: int, dim: int, n_clusters: int = 20, seed: int = 42):
    """
    Generates clustered data that resembles real embedding distributions.
    Clusters are important — random uniform data doesn't stress HNSW's
    topology blind spots the way real clustered embeddings do.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    per_cluster = N // n_clusters
    points = []
    for c in centers:
        noise = rng.standard_normal((per_cluster, dim)).astype(np.float32) * 0.15
        points.append(c + noise)
    data = np.vstack(points)[:N]

    # Queries come from a slightly different distribution
    q_centers = centers + rng.standard_normal(centers.shape).astype(np.float32) * 0.05
    queries = []
    for c in q_centers:
        noise = rng.standard_normal((5, dim)).astype(np.float32) * 0.12
        queries.append(c + noise)
    queries = np.vstack(queries)[:min(1000, N // 10)]

    return data, queries


def load_hdf5(path: str):
    """Load ann-benchmarks HDF5 dataset."""
    if not HAS_H5PY:
        raise ImportError("Install h5py: pip install h5py")
    with h5py.File(path, "r") as f:
        data    = np.array(f["train"], dtype=np.float32)
        queries = np.array(f["test"],  dtype=np.float32)
        gt      = np.array(f["neighbors"])  # ground truth: (n_queries, 100)
    return data, queries, gt


def brute_force_knn(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact kNN via brute force — used for ground truth on synthetic data."""
    print(f"  Computing brute-force ground truth for {len(queries)} queries ...")
    results = np.zeros((len(queries), k), dtype=int)
    for i, q in enumerate(queries):
        dists = np.sum((data - q) ** 2, axis=1)
        results[i] = np.argpartition(dists, k)[:k]
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Recall computation
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """recall@k = fraction of true neighbours correctly returned, averaged."""
    n, k = predicted.shape
    hits = 0
    for i in range(n):
        pred_set = set(predicted[i].tolist())
        true_set = set(ground_truth[i, :k].tolist())
        hits += len(pred_set & true_set)
    return hits / (n * k)


# ─────────────────────────────────────────────────────────────────────────────
# Build + benchmark one index
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_hnsw(
    data:      np.ndarray,
    queries:   np.ndarray,
    gt:        np.ndarray,
    M:         int,
    ef_const:  int,
    ef_search_values: list,
    k:         int,
    space:     str = "l2"
) -> dict:
    """Build standard hnswlib index and measure recall-QPS curve."""
    import hnswlib
    print(f"\n[HNSW] Building index: N={len(data)}, dim={data.shape[1]}, M={M}")
    idx = hnswlib.Index(space=space, dim=data.shape[1])
    idx.init_index(max_elements=len(data), ef_construction=ef_const, M=M)

    t0 = time.perf_counter()
    idx.add_items(data, np.arange(len(data)))
    build_time = time.perf_counter() - t0
    print(f"[HNSW] Build time: {build_time:.2f}s")

    results = {"build_time": build_time, "curve": []}
    for ef in ef_search_values:
        idx.set_ef(ef)
        t0 = time.perf_counter()
        labels, _ = idx.knn_query(queries, k=k)
        elapsed = time.perf_counter() - t0
        qps = len(queries) / elapsed
        rec = recall_at_k(labels, gt)
        results["curve"].append({"ef": ef, "recall": rec, "qps": qps})
        print(f"  ef={ef:4d}  recall={rec:.4f}  QPS={qps:8.1f}")

    return results


def benchmark_tahnsw(
    data:      np.ndarray,
    queries:   np.ndarray,
    gt:        np.ndarray,
    M:         int,
    ef_const:  int,
    ef_search_values: list,
    k:         int,
    space:     str = "l2",
    config     = None
) -> dict:
    """Build TAHNSW index and measure recall-QPS curve."""
    from tahnsw import TAHNSWConfig, TAHNSWIndex
    cfg = config or TAHNSWConfig(verbose=True)
    print(f"\n[TAHNSW] Building index: N={len(data)}, dim={data.shape[1]}, M={M}")
    print(f"  Config: HIGH={cfg.cluster_high_thresh}, LOW={cfg.cluster_low_thresh}, "
          f"alpha={cfg.M_alpha}, beta={cfg.M_beta}")

    idx = TAHNSWIndex(dim=data.shape[1], max_elements=len(data),
                      space=space, config=cfg)
    idx.init_index(M=M, ef_construction=ef_const)

    t0 = time.perf_counter()
    idx.add_items(data, np.arange(len(data), dtype=np.int64))
    build_time = time.perf_counter() - t0
    print(f"[TAHNSW] Build time: {build_time:.2f}s")

    results = {
        "build_time": build_time,
        "curve": [],
        "stats": {
            "hub_pct":      idx.stats.hub_count / max(1, idx.stats.total_inserted),
            "leaf_pct":     idx.stats.leaf_count / max(1, idx.stats.total_inserted),
            "mean_M_eff":   float(np.mean(idx.stats.M_values)) if idx.stats.M_values else M,
            "edge_reduction": idx.stats.edges_saved / max(1, idx.stats.edges_baseline)
        }
    }

    for ef in ef_search_values:
        idx.set_ef(ef)
        t0 = time.perf_counter()
        labels, _ = idx.knn_query(queries, k=k)
        elapsed = time.perf_counter() - t0
        qps = len(queries) / elapsed
        rec = recall_at_k(labels, gt)
        results["curve"].append({"ef": ef, "recall": rec, "qps": qps})
        print(f"  ef={ef:4d}  recall={rec:.4f}  QPS={qps:8.1f}")
    return results

def benchmark_tahnsw_cpp(
    data:      np.ndarray,
    queries:   np.ndarray,
    gt:        np.ndarray,
    M:         int,
    ef_const:  int,
    ef_search_values: list,
    k:         int,
    space:     str = "l2",
    config     = None
) -> dict:
    """Build TAHNSW C++ index and measure recall-QPS curve."""
    import time
    import tahnsw_cpp
    from tahnsw import TAHNSWConfig
    cfg = config or TAHNSWConfig()
    print(f"\n[TAHNSW C++] Building index: N={len(data)}, dim={data.shape[1]}, M={M}")
    
    idx = tahnsw_cpp.IndexCpp(space=space, dim=data.shape[1])
    idx.init_index(max_elements=len(data), ef_construction=ef_const, M=M)
    idx.set_tahnsw_config(
        cluster_high_thresh=cfg.cluster_high_thresh,
        cluster_low_thresh=cfg.cluster_low_thresh,
        M_alpha=cfg.M_alpha,
        M_beta=cfg.M_beta,
        M_min=cfg.M_min,
        sketch_width=cfg.sketch_width,
        sketch_depth=cfg.sketch_depth,
        sketch_S=cfg.sketch_S,
        prune_layer0_only=cfg.prune_layer0_only,
        prune_angle_check=cfg.prune_angle_check,
        prune_angle_deg=cfg.prune_angle_deg,
        k_candidates=cfg.k_candidates,
        b_max_window=cfg.b_max_window
    )

    t0 = time.perf_counter()
    idx.add_items(data, np.arange(len(data)))
    build_time = time.perf_counter() - t0
    print(f"[TAHNSW C++] Build time: {build_time:.2f}s")
    
    stats = idx.get_tahnsw_stats()

    results = {
        "build_time": build_time,
        "curve": [],
        "stats": {
            "hub_pct":      stats["hub_count"] / max(1, len(data)),
            "leaf_pct":     stats["leaf_count"] / max(1, len(data)),
            "mean_M_eff":   M,
            "edge_reduction": stats["edges_saved"] / max(1, stats["edges_baseline"])
        }
    }

    for ef in ef_search_values:
        idx.set_ef(ef)
        t0 = time.perf_counter()
        labels, _ = idx.knn_query(queries, k=k)
        elapsed = time.perf_counter() - t0
        qps = len(queries) / elapsed
        rec = recall_at_k(labels, gt)
        results["curve"].append({"ef": ef, "recall": rec, "qps": qps})
        print(f"  ef={ef:4d}  recall={rec:.4f}  QPS={qps:8.1f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(hnsw_res: dict, tahnsw_res: dict, tahnsw_cpp_res: dict, title: str, outpath: str) -> None:
    if not HAS_MPL:
        print("matplotlib not available — skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── Recall-QPS curve ──────────────────────────────────────────────────────
    ax = axes[0]
    h_recalls = [p["recall"] for p in hnsw_res["curve"]]
    h_qps     = [p["qps"]    for p in hnsw_res["curve"]]
    t_recalls = [p["recall"] for p in tahnsw_res["curve"]]
    t_qps     = [p["qps"]    for p in tahnsw_res["curve"]]
    tc_recalls = [p["recall"] for p in tahnsw_cpp_res["curve"]] if tahnsw_cpp_res else []
    tc_qps     = [p["qps"]    for p in tahnsw_cpp_res["curve"]] if tahnsw_cpp_res else []

    ax.plot(h_recalls, h_qps, "o-", color="#888780", lw=2, ms=6,
            label=f"HNSW  (build {hnsw_res['build_time']:.1f}s)")
    ax.plot(t_recalls, t_qps, "s-", color="#534AB7", lw=2.5, ms=7,
            label=f"TAHNSW (build {tahnsw_res['build_time']:.1f}s)")
    
    if tahnsw_cpp_res:
        ax.plot(tc_recalls, tc_qps, "^-", color="#E63946", lw=2.5, ms=7,
                label=f"TAHNSW C++ (build {tahnsw_cpp_res['build_time']:.1f}s)")

    all_recalls = h_recalls + t_recalls + tc_recalls
    all_qps = h_qps + t_qps + tc_qps
    ax.fill_between(
        [min(all_recalls), max(all_recalls)],
        0, max(all_qps) * 1.1,
        alpha=0
    )

    if t_recalls and h_recalls:
        ax.fill_between(
            sorted(set(h_recalls + t_recalls)),
            np.interp(sorted(set(h_recalls + t_recalls)), h_recalls, h_qps),
            np.interp(sorted(set(h_recalls + t_recalls)), t_recalls, t_qps),
            alpha=0.12, color="#534AB7"
        )

    ax.set_xlabel("Recall @ k", fontsize=12)
    ax.set_ylabel("Queries per second (QPS)", fontsize=12)
    ax.set_title("Recall vs QPS tradeoff\n(higher curve = better)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    # ── Build time bar ────────────────────────────────────────────────────────
    ax2 = axes[1]
    methods  = ["HNSW", "TAHNSW Python"]
    times    = [hnsw_res["build_time"], tahnsw_res["build_time"]]
    colors   = ["#888780", "#534AB7"]
    
    if tahnsw_cpp_res:
        methods.append("TAHNSW C++")
        times.append(tahnsw_cpp_res["build_time"])
        colors.append("#E63946")
    bars     = ax2.bar(methods, times, color=colors, width=0.4, edgecolor="none")
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{t:.1f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")

    delta = 100 * (1 - tahnsw_res["build_time"] / max(1e-9, hnsw_res["build_time"]))
    ax2.set_title(f"Index build time\nTAHNSW is {delta:+.1f}% vs HNSW", fontsize=11)
    ax2.set_ylabel("Seconds", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {outpath}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TAHNSW vs HNSW benchmark")
    parser.add_argument("--mode",    choices=["synthetic", "hdf5"], default="synthetic")
    parser.add_argument("--dataset", default="sift-128-euclidean.hdf5")
    parser.add_argument("--N",       type=int, default=30000)
    parser.add_argument("--dim",     type=int, default=128)
    parser.add_argument("--k",       type=int, default=10)
    parser.add_argument("--M",       type=int, default=16)
    parser.add_argument("--ef_construction", type=int, default=200)
    parser.add_argument("--space",   default="l2", choices=["l2", "cosine"])
    parser.add_argument("--outdir",  default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ef_values = [10, 20, 40, 80, 120, 200, 300]

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.mode == "synthetic":
        print(f"Generating synthetic clustered data: N={args.N}, dim={args.dim}")
        data, queries = load_synthetic(args.N, args.dim)
        gt = brute_force_knn(data, queries, k=args.k)
        title = f"TAHNSW vs HNSW — Synthetic Clustered (N={args.N:,}, d={args.dim})"

    else:
        print(f"Loading HDF5 dataset: {args.dataset}")
        data, queries, gt = load_hdf5(args.dataset)
        title = f"TAHNSW vs HNSW — {os.path.basename(args.dataset)}"

    print(f"Data shape:    {data.shape}")
    print(f"Query shape:   {queries.shape}")
    print(f"GT shape:      {gt.shape}")

    # ── TAHNSW config ─────────────────────────────────────────────────────────
    from tahnsw import TAHNSWConfig
    cfg = TAHNSWConfig(
        cluster_high_thresh = 0.70,
        cluster_low_thresh  = 0.20,
        M_alpha             = 1.5,
        M_beta              = 0.5,
        sketch_width        = 512,
        sketch_depth        = 5,
        sketch_S            = 30,
        k_candidates        = 20,
        verbose             = True
    )

    # ── Run benchmarks ────────────────────────────────────────────────────────
    hnsw_res   = benchmark_hnsw(
        data, queries, gt,
        M=args.M, ef_const=args.ef_construction,
        ef_search_values=ef_values, k=args.k, space=args.space
    )
    tahnsw_res = benchmark_tahnsw(
        data, queries, gt,
        M=args.M, ef_const=args.ef_construction,
        ef_search_values=ef_values, k=args.k, space=args.space, config=cfg
    )
    tahnsw_cpp_res = benchmark_tahnsw_cpp(
        data, queries, gt,
        M=args.M, ef_const=args.ef_construction,
        ef_search_values=ef_values, k=args.k, space=args.space, config=cfg
    )

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "config": {
            "N": len(data), "dim": data.shape[1], "k": args.k,
            "M": args.M, "ef_construction": args.ef_construction,
            "space": args.space
        },
        "hnsw":   hnsw_res,
        "tahnsw": tahnsw_res,
        "tahnsw_cpp": tahnsw_cpp_res
    }
    json_path = os.path.join(args.outdir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'ef':>6}  {'HNSW rec':>10}  {'HNSW QPS':>10}  "
          f"{'TAHNSW rec':>12}  {'TAHNSW QPS':>12}  "
          f"{'C++ rec':>10}  {'C++ QPS':>10}  {'QPS delta (Py)':>15}")
    print("-" * 80)
    for h, t, tc in zip(hnsw_res["curve"], tahnsw_res["curve"], tahnsw_cpp_res["curve"]):
        delta = 100 * (t["qps"] - h["qps"]) / max(h["qps"], 1e-9)
        print(f"{h['ef']:>6}  {h['recall']:>10.4f}  {h['qps']:>10.1f}  "
              f"{t['recall']:>12.4f}  {t['qps']:>12.1f}  "
              f"{tc['recall']:>10.4f}  {tc['qps']:>10.1f}  {delta:>+14.1f}%")

    bt_delta = 100 * (1 - tahnsw_res["build_time"] / max(hnsw_res["build_time"], 1e-9))
    cpp_delta = 100 * (1 - tahnsw_cpp_res["build_time"] / max(tahnsw_res["build_time"], 1e-9))
    print("-"*80)
    print(f"Build time:  HNSW {hnsw_res['build_time']:.2f}s  "
          f"vs  TAHNSW (Py) {tahnsw_res['build_time']:.2f}s  ({bt_delta:+.1f}%)  "
          f"vs  TAHNSW (C++) {tahnsw_cpp_res['build_time']:.2f}s  ({cpp_delta:+.1f}% faster than Py)")
    
    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = os.path.join(args.outdir, "recall_qps_curve.png")
    plot_results(hnsw_res, tahnsw_res, tahnsw_cpp_res, title, plot_path)


if __name__ == "__main__":
    main()
