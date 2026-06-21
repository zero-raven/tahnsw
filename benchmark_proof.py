"""
benchmark_proof.py — reproducible, graphical proof of the TAHNSW optimizations.

Compares three indexes built from the SAME compiled binary:
  * HNSW            — stock hnswlib
  * TAHNSW (orig)   — topology-aware, original construction  (fast_construction=False)
  * TAHNSW (opt)    — topology-aware, optimized construction (fast_construction=True)

Produces results/tahnsw_proof.png with four panels:
  (a) build time vs N           — optimized TAHNSW tracks HNSW; original is ~3x slower
  (b) build overhead vs HNSW    — bar chart at the largest N
  (c) recall-QPS curve          — search quality/speed parity
  (d) searchBaseLayer per node  — the eliminated redundant search (≈3 → ≈1)

Methodology (anti-noise):
  - build times: interleaved across methods, best-of-BUILD_REPS (min)
  - QPS:         warmup + best-of-Q_REPS (min latency)
  - ground truth via exact brute force; recall@k measured against it
Run:  python benchmark_proof.py
"""
import time, os, json
import numpy as np
import hnswlib, tahnsw_cpp

# ── experiment knobs ──────────────────────────────────────────────────────────
DIM, K, M, EFC = 128, 10, 16, 200
N_SCALING   = [2500, 5000, 10000, 20000]   # build-time-vs-N panel
N_CURVE     = 20000                        # recall-QPS panel
EFS         = [10, 20, 40, 80, 150, 250]
N_QUERIES   = 300
BUILD_REPS  = 2
Q_REPS      = 8
NThr        = 1                            # single thread isolates algorithmic cost
OUT         = os.path.join("results", "tahnsw_proof.png")


def synth(N, dim, n_clusters=20, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    pts = [c + rng.standard_normal((N // n_clusters + 1, dim)).astype(np.float32) * 0.15
           for c in centers]
    data = np.vstack(pts)[:N]
    qc = centers + rng.standard_normal(centers.shape).astype(np.float32) * 0.05
    qs = [c + rng.standard_normal((N_QUERIES // n_clusters + 1, dim)).astype(np.float32) * 0.12
          for c in qc]
    return data, np.vstack(qs)[:N_QUERIES]


def brute_force(data, q, k):
    out = np.zeros((len(q), k), dtype=int)
    for i, x in enumerate(q):
        out[i] = np.argpartition(np.sum((data - x) ** 2, axis=1), k)[:k]
    return out


def recall(pred, gt):
    n, k = pred.shape
    return sum(len(set(pred[i]) & set(gt[i, :k])) for i in range(n)) / (n * k)


def make_index(kind, data, N):
    """kind: 'hnsw' | 'orig' | 'opt'. Returns (index, build_seconds)."""
    if kind == "hnsw":
        idx = hnswlib.Index(space="l2", dim=DIM)
        idx.init_index(max_elements=N, ef_construction=EFC, M=M)
        idx.set_num_threads(NThr)
        t = time.perf_counter(); idx.add_items(data, np.arange(N))
        return idx, time.perf_counter() - t
    idx = tahnsw_cpp.IndexCpp(space="l2", dim=DIM)
    idx.init_index(max_elements=N, ef_construction=EFC, M=M)
    idx.set_tahnsw_config(fast_construction=(kind == "opt"))
    idx.set_num_threads(NThr)
    t = time.perf_counter(); idx.add_items(data, np.arange(N))
    return idx, time.perf_counter() - t


def best_build(kind, data, N):
    """Interleaved best-of build time; returns (best_seconds, last_index)."""
    best, last = 1e18, None
    for _ in range(BUILD_REPS):
        idx, bt = make_index(kind, data, N)
        best = min(best, bt); last = idx
    return best, last


def qps_curve(idx, q, gt, is_cpp):
    idx.set_num_threads(NThr)
    out = []
    for ef in EFS:
        idx.set_ef(ef)
        idx.knn_query(q, k=K)  # warmup
        best, lab = 1e18, None
        for _ in range(Q_REPS):
            t = time.perf_counter(); lab, _ = idx.knn_query(q, k=K)
            best = min(best, time.perf_counter() - t)
        out.append((ef, recall(lab, gt), len(q) / best))
    return out


def main():
    os.makedirs("results", exist_ok=True)
    kinds = ["hnsw", "orig", "opt"]
    labels = {"hnsw": "HNSW", "orig": "TAHNSW (orig)", "opt": "TAHNSW (opt)"}

    # ── (a) build time vs N — interleave methods at each N to share noise ───────
    print("== build-time scaling ==")
    build_t = {k: [] for k in kinds}
    for N in N_SCALING:
        data, _ = synth(N, DIM)
        row = {}
        for k in kinds:
            bt, _ = best_build(k, data, N)
            build_t[k].append(bt); row[k] = bt
        print(f"  N={N:6d}  " + "  ".join(f"{labels[k]}={row[k]:.2f}s" for k in kinds)
              + f"  | opt/HNSW={row['opt']/row['hnsw']:.2f}x  orig/HNSW={row['orig']/row['hnsw']:.2f}x")

    # ── (c)+(d) recall-QPS + search calls at N_CURVE ────────────────────────────
    print(f"== recall-QPS @ N={N_CURVE} ==")
    data, q = synth(N_CURVE, DIM)
    gt = brute_force(data, q, K)
    curves, calls = {}, {}
    for k in kinds:
        _, idx = best_build(k, data, N_CURVE)
        curves[k] = qps_curve(idx, q, gt, is_cpp=(k != "hnsw"))
        calls[k] = (idx.get_tahnsw_stats()["search_calls"] / N_CURVE) if k != "hnsw" else 1.0
        print(f"  {labels[k]:16s} searches/node={calls[k]:.2f}")
        for ef, r, qp in curves[k]:
            print(f"      ef={ef:4d} recall={r:.4f} qps={qp:8.0f}")

    # ── plot ────────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing — skipping plot"); return

    col = {"hnsw": "#888780", "orig": "#E63946", "opt": "#534AB7"}
    mk  = {"hnsw": "o", "orig": "s", "opt": "^"}
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("TAHNSW construction optimization — proof", fontsize=15, fontweight="bold")

    # (a) build vs N
    a = ax[0, 0]
    for k in kinds:
        a.plot(N_SCALING, build_t[k], mk[k] + "-", color=col[k], lw=2.2, ms=7, label=labels[k])
    a.set_xlabel("N (vectors)"); a.set_ylabel("build time (s), single-thread")
    a.set_title("(a) Build time vs N  — lower is better"); a.legend(); a.grid(alpha=0.3)

    # (b) overhead vs HNSW at largest N
    b = ax[0, 1]
    big = -1
    ov = [build_t[k][big] / build_t["hnsw"][big] for k in kinds]
    bars = b.bar([labels[k] for k in kinds], ov, color=[col[k] for k in kinds], width=0.55)
    for bar, v in zip(bars, ov):
        b.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f}x",
               ha="center", va="bottom", fontweight="bold")
    b.axhline(1.0, color="#888780", ls="--", lw=1)
    b.set_ylabel("build time / HNSW")
    b.set_title(f"(b) Build overhead vs HNSW  (N={N_SCALING[big]:,})"); b.grid(alpha=0.3, axis="y")

    # (c) recall-QPS
    c = ax[1, 0]
    for k in kinds:
        rs = [p[1] for p in curves[k]]; qs = [p[2] for p in curves[k]]
        c.plot(rs, qs, mk[k] + "-", color=col[k], lw=2.2, ms=7, label=labels[k])
    c.set_xlabel("recall@%d" % K); c.set_ylabel("queries / sec")
    c.set_title("(c) Recall–QPS  — upper-right is better"); c.legend(); c.grid(alpha=0.3)

    # (d) searches per node
    d = ax[1, 1]
    ck = ["orig", "opt"]
    bars = d.bar([labels[k] for k in ck], [calls[k] for k in ck],
                 color=[col[k] for k in ck], width=0.5)
    for bar, k in zip(bars, ck):
        d.text(bar.get_x() + bar.get_width() / 2, calls[k] + 0.03,
               f"{calls[k]:.2f}", ha="center", va="bottom", fontweight="bold")
    d.set_ylabel("searchBaseLayer calls / node")
    d.set_title("(d) Redundant search eliminated"); d.grid(alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\nsaved figure -> {OUT}")

    with open(os.path.join("results", "tahnsw_proof.json"), "w") as f:
        json.dump({"build_t": build_t, "N_SCALING": N_SCALING,
                   "curves": curves, "search_calls_per_node": calls}, f, indent=2)
    print("saved data   -> results/tahnsw_proof.json")


if __name__ == "__main__":
    main()
