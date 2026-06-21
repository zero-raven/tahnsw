"""
benchmark_real_verify.py — rigorous check that TAHNSW(alpha) beats HNSW on the
recall-QPS FRONTIER, not merely by spending more edges.

Controls:
  * HNSW M=16/24/32         — HNSW's own frontier (more M = more edges/memory)
  * TAHNSW alpha=1.0 M=16   — must ~match HNSW M=16 (alpha=1 is neutral)
  * TAHNSW alpha=1.2 M=16   — the candidate winner

If alpha=1.2 M=16 sits above even HNSW M=32, the gain is NOT from extra edges.
Build multithreaded (best-of); QPS single-thread (best-of). Real SIFT-128 subset.
Output: results/real_sift_verify.png / .json
"""
import time, os, json
import numpy as np
import h5py, hnswlib, tahnsw_cpp

DATA, N_BASE, N_QRY, K, EFC = "data_real/sift-128-euclidean.hdf5", 100000, 1000, 10, 200
EFS = [10, 15, 20, 30, 40, 60, 100, 150, 250]
Q_REPS, BUILD_REPS = 8, 2
OUT = "results/real_sift_verify"


def load():
    with h5py.File(DATA, "r") as f:
        return (np.ascontiguousarray(f["train"][:N_BASE]).astype(np.float32),
                np.ascontiguousarray(f["test"][:N_QRY]).astype(np.float32))


def exact_gt(base, qry, k):
    bn = np.einsum("ij,ij->i", base, base); gt = np.zeros((len(qry), k), np.int64)
    for s in range(0, len(qry), 100):
        sc = bn[None, :] - 2.0 * (qry[s:s+100] @ base.T)
        gt[s:s+100] = np.argpartition(sc, k, axis=1)[:, :k]
    return gt


def recall(p, gt):
    n, k = p.shape
    return sum(len(set(p[i]) & set(gt[i, :k])) for i in range(n)) / (n * k)


def build_hnsw(base, M):
    best, idx = 1e18, None
    for _ in range(BUILD_REPS):
        i = hnswlib.Index(space="l2", dim=base.shape[1])
        i.init_index(max_elements=len(base), ef_construction=EFC, M=M); i.set_num_threads(0)
        t = time.perf_counter(); i.add_items(base, np.arange(len(base))); bt = time.perf_counter()-t
        if bt < best: best, idx = bt, i
    return idx, best


def build_tahnsw(base, M, alpha):
    best, idx = 1e18, None
    for _ in range(BUILD_REPS):
        i = tahnsw_cpp.IndexCpp(space="l2", dim=base.shape[1])
        i.init_index(max_elements=len(base), ef_construction=EFC, M=M)
        i.set_tahnsw_config(cluster_low_thresh=0.0, cluster_high_thresh=1.0,
                            M_alpha=1.0, M_beta=1.0, heuristic_alpha=alpha)
        i.set_num_threads(0)
        t = time.perf_counter(); i.add_items(base, np.arange(len(base))); bt = time.perf_counter()-t
        if bt < best: best, idx = bt, i
    return idx, best


def curve(idx, qry, gt):
    idx.set_num_threads(1); out = []
    for ef in EFS:
        idx.set_ef(ef); idx.knn_query(qry, k=K)
        best, lab = 1e18, None
        for _ in range(Q_REPS):
            t = time.perf_counter(); lab, _ = idx.knn_query(qry, k=K); best = min(best, time.perf_counter()-t)
        out.append((ef, recall(lab, gt), len(qry)/best))
    return out


def iso(c, tgt):
    pts = sorted((r, q) for _, r, q in c)
    for i in range(1, len(pts)):
        if pts[i][0] >= tgt:
            (r0, q0), (r1, q1) = pts[i-1], pts[i]
            return q0 + (q1-q0)*(tgt-r0)/max(1e-9, r1-r0)
    return float("nan")


def main():
    os.makedirs("results", exist_ok=True)
    base, qry = load(); print("computing GT..."); gt = exact_gt(base, qry, K)
    specs = [("HNSW M=16", "hnsw", 16, None), ("HNSW M=24", "hnsw", 24, None),
             ("HNSW M=32", "hnsw", 32, None), ("TAHNSW M=16 a=1.0", "ta", 16, 1.0),
             ("TAHNSW M=16 a=1.2", "ta", 16, 1.2)]
    res = {}
    for name, kind, M, a in specs:
        idx, bt = build_hnsw(base, M) if kind == "hnsw" else build_tahnsw(base, M, a)
        res[name] = {"build": bt, "curve": curve(idx, qry, gt)}
        print(f"{name:20s} build={bt:.2f}s")

    base_iso = {t: iso(res["HNSW M=16"]["curve"], t) for t in (0.90, 0.95, 0.99)}
    print(f"\n{'method':20s} {'build':>6s}  iso@0.90        iso@0.95        iso@0.99")
    for name, r in res.items():
        cells = []
        for t in (0.90, 0.95, 0.99):
            v = iso(r["curve"], t); d = 100*(v-base_iso[t])/base_iso[t] if base_iso[t]==base_iso[t] else 0
            cells.append(f"{v:7.0f}({d:+4.0f}%)")
        print(f"{name:20s} {r['build']:6.2f}  " + "  ".join(cells))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"TAHNSW alpha-RNG vs HNSW frontier — real SIFT-128 (N={N_BASE:,})", fontsize=14, fontweight="bold")
    style = {"HNSW M=16": ("o--", "#999"), "HNSW M=24": ("o--", "#666"), "HNSW M=32": ("o--", "#000"),
             "TAHNSW M=16 a=1.0": ("^-", "#7aa"), "TAHNSW M=16 a=1.2": ("^-", "#E63946")}
    for name, r in res.items():
        m, col = style[name]; c = r["curve"]
        lw = 3 if "1.2" in name else 1.8
        ax[0].plot([p[1] for p in c], [p[2] for p in c], m, color=col, lw=lw, ms=6, label=name, zorder=6 if "1.2" in name else 3)
    ax[0].set_xlabel(f"recall@{K}"); ax[0].set_ylabel("queries/sec (single-thread)")
    ax[0].set_title("Recall–QPS frontier  (upper-right better)"); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3); ax[0].set_xlim(0.85, 1.003)
    ts = [0.90, 0.95, 0.99]; x = np.arange(len(ts)); names = list(res.keys()); w = 0.8/len(names)
    for j, name in enumerate(names):
        ax[1].bar(x - 0.4 + (j+0.5)*w, [iso(res[name]["curve"], t) for t in ts], w, color=style[name][1], label=name)
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"recall {t}" for t in ts]); ax[1].set_ylabel("iso-recall QPS")
    ax[1].set_title("QPS at fixed recall (taller=faster)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(OUT+".png", dpi=150, bbox_inches="tight")
    json.dump(res, open(OUT+".json", "w"), indent=2)
    print(f"\nsaved -> {OUT}.png / .json")


if __name__ == "__main__":
    main()
