"""
benchmark_real.py — TAHNSW vs HNSW on the REAL SIFT-128 dataset (ann-benchmarks).

Loads a subset of SIFT-128, recomputes exact ground truth, and sweeps the
alpha-RNG (Vamana/DiskANN RobustPrune) knob at FIXED degree (M=16, same memory
budget as the HNSW baseline) to test whether it shifts the recall-QPS frontier
ABOVE HNSW. Build is multithreaded; QPS is measured single-thread (clean).

Output: results/real_sift_alpha.png + results/real_sift_alpha.json
"""
import time, os, json
import numpy as np
import h5py, hnswlib, tahnsw_cpp

DATA   = "data_real/sift-128-euclidean.hdf5"
N_BASE = 100000
N_QRY  = 1000
K      = 10
M      = 16
EFC    = 200
EFS    = [10, 15, 20, 30, 40, 60, 100, 150, 250]
ALPHAS = [0.9, 1.0, 1.1, 1.2, 1.3]
Q_REPS = 6
OUT    = "results/real_sift_alpha"


def load_subset():
    with h5py.File(DATA, "r") as f:
        base = np.ascontiguousarray(f["train"][:N_BASE]).astype(np.float32)
        qry  = np.ascontiguousarray(f["test"][:N_QRY]).astype(np.float32)
    return base, qry


def exact_gt(base, qry, k):
    bn = np.einsum("ij,ij->i", base, base)          # ||b||^2
    gt = np.zeros((len(qry), k), dtype=np.int64)
    for s in range(0, len(qry), 100):
        q = qry[s:s+100]
        sc = bn[None, :] - 2.0 * (q @ base.T)        # monotone in true L2
        gt[s:s+100] = np.argpartition(sc, k, axis=1)[:, :k]
    return gt


def recall(pred, gt):
    n, k = pred.shape
    return sum(len(set(pred[i]) & set(gt[i, :k])) for i in range(n)) / (n * k)


def hnsw_build(base):
    idx = hnswlib.Index(space="l2", dim=base.shape[1])
    idx.init_index(max_elements=len(base), ef_construction=EFC, M=M)
    idx.set_num_threads(0)
    t = time.perf_counter(); idx.add_items(base, np.arange(len(base)))
    return idx, time.perf_counter() - t


def tahnsw_build(base, alpha):
    idx = tahnsw_cpp.IndexCpp(space="l2", dim=base.shape[1])
    idx.init_index(max_elements=len(base), ef_construction=EFC, M=M)
    # Isolate the alpha knob: no hub/leaf split (all "mid" => HNSW-like layers),
    # no degree adaptation (M_eff == M). Only the selection heuristic changes.
    idx.set_tahnsw_config(cluster_low_thresh=0.0, cluster_high_thresh=1.0,
                          M_alpha=1.0, M_beta=1.0, heuristic_alpha=alpha)
    idx.set_num_threads(0)
    t = time.perf_counter(); idx.add_items(base, np.arange(len(base)))
    return idx, time.perf_counter() - t


def curve(idx, qry, gt):
    idx.set_num_threads(1)
    out = []
    for ef in EFS:
        idx.set_ef(ef); idx.knn_query(qry, k=K)
        best, lab = 1e18, None
        for _ in range(Q_REPS):
            t = time.perf_counter(); lab, _ = idx.knn_query(qry, k=K)
            best = min(best, time.perf_counter() - t)
        out.append((ef, recall(lab, gt), len(qry) / best))
    return out


def iso_qps(c, target):
    pts = sorted((r, q) for _, r, q in c)
    for i in range(1, len(pts)):
        if pts[i][0] >= target:
            (r0, q0), (r1, q1) = pts[i-1], pts[i]
            return q0 + (q1 - q0) * (target - r0) / max(1e-9, r1 - r0)
    return pts[-1][1] if pts[-1][0] >= target else float("nan")


def main():
    os.makedirs("results", exist_ok=True)
    print(f"loading SIFT-128 subset: N_base={N_BASE} N_query={N_QRY} dim=128")
    base, qry = load_subset()
    print("computing exact ground truth ...")
    gt = exact_gt(base, qry, K)

    results = {}
    hi, hb = hnsw_build(base)
    results["HNSW"] = {"build": hb, "curve": curve(hi, qry, gt)}
    print(f"HNSW           build={hb:.2f}s")

    for a in ALPHAS:
        ti, tb = tahnsw_build(base, a)
        results[f"TAHNSW a={a}"] = {"build": tb, "curve": curve(ti, qry, gt), "alpha": a}
        print(f"TAHNSW a={a:<4} build={tb:.2f}s")

    # report
    print(f"\n{'method':16s} {'build':>7s}  " + "  ".join(f"iso@{t}" for t in (0.90, 0.95, 0.99)))
    hrow = results["HNSW"]
    h_iso = {t: iso_qps(hrow["curve"], t) for t in (0.90, 0.95, 0.99)}
    for name, r in results.items():
        iso = {t: iso_qps(r["curve"], t) for t in (0.90, 0.95, 0.99)}
        deltas = "  ".join(
            (f"{iso[t]:7.0f}" + (f"({100*(iso[t]-h_iso[t])/h_iso[t]:+.0f}%)" if name != "HNSW" and h_iso[t]==h_iso[t] else "       "))
            for t in (0.90, 0.95, 0.99))
        print(f"{name:16s} {r['build']:7.2f}  {deltas}")

    # plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"TAHNSW (alpha-RNG) vs HNSW — real SIFT-128 (N={N_BASE:,}, M={M})",
                 fontsize=14, fontweight="bold")
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(ALPHAS)))
    a0 = ax[0]
    hc = results["HNSW"]["curve"]
    a0.plot([p[1] for p in hc], [p[2] for p in hc], "o--", color="black", lw=2.5, ms=7, label="HNSW (M=16)", zorder=5)
    for a, col in zip(ALPHAS, cmap):
        c = results[f"TAHNSW a={a}"]["curve"]
        a0.plot([p[1] for p in c], [p[2] for p in c], "^-", color=col, lw=2, ms=6, label=f"TAHNSW α={a}")
    a0.set_xlabel(f"recall@{K}"); a0.set_ylabel("queries/sec (single-thread)")
    a0.set_title("Recall–QPS frontier  (upper-right is better)"); a0.legend(fontsize=9); a0.grid(alpha=0.3)
    a0.set_xlim(0.80, 1.005)

    a1 = ax[1]
    ts = [0.90, 0.95, 0.99]
    x = np.arange(len(ts)); w = 0.8 / (len(ALPHAS) + 1)
    a1.bar(x - 0.4 + 0.5*w, [h_iso[t] for t in ts], w, color="black", label="HNSW")
    for j, (a, col) in enumerate(zip(ALPHAS, cmap)):
        iso = [iso_qps(results[f"TAHNSW a={a}"]["curve"], t) for t in ts]
        a1.bar(x - 0.4 + (j+1.5)*w, iso, w, color=col, label=f"α={a}")
    a1.set_xticks(x); a1.set_xticklabels([f"recall {t}" for t in ts])
    a1.set_ylabel("iso-recall QPS"); a1.set_title("QPS at fixed recall (taller = faster)")
    a1.legend(fontsize=9); a1.grid(alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
    with open(OUT + ".json", "w") as f:
        json.dump({k: {"build": v["build"], "curve": v["curve"]} for k, v in results.items()}, f, indent=2)
    print(f"\nsaved -> {OUT}.png and {OUT}.json")


if __name__ == "__main__":
    main()
