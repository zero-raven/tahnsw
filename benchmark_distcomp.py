"""
benchmark_distcomp.py — hardware-independent verdict on alpha-RNG.

Wall-clock QPS on this machine has a ~+/-30% noise floor (proven: the alpha=1.0
control, which must equal HNSW, swings that much run-to-run). So we drop the clock
and measure the DETERMINISTIC quantity that actually governs speed: the mean number
of distance computations per query at a given recall. Fewer comps @ same recall =
genuinely faster on ANY hardware.

All indexes are tahnsw_cpp at M=16 with identical config EXCEPT heuristic_alpha,
so the comparison isolates alpha exactly. alpha=1.0 == stock HNSW heuristic.
Output: results/distcomp.png / .json
"""
import os, json
import numpy as np
import h5py, tahnsw_cpp

DATA, N_BASE, N_QRY, K, M, EFC = "data_real/sift-128-euclidean.hdf5", 100000, 1000, 10, 16, 200
EFS = [10, 15, 20, 30, 40, 60, 100, 150, 250, 400]
ALPHAS = [0.9, 1.0, 1.1, 1.2, 1.3]


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


def build(base, alpha):
    i = tahnsw_cpp.IndexCpp(space="l2", dim=base.shape[1])
    i.init_index(max_elements=len(base), ef_construction=EFC, M=M)
    i.set_tahnsw_config(cluster_low_thresh=0.0, cluster_high_thresh=1.0,
                        M_alpha=1.0, M_beta=1.0, heuristic_alpha=alpha)
    i.set_num_threads(0); i.add_items(base, np.arange(len(base)))
    return i


def curve(idx, qry, gt):
    out = []
    for ef in EFS:
        idx.set_ef(ef)
        idx.reset_metrics()
        lab, _ = idx.knn_query(qry, k=K)
        dc = idx.get_tahnsw_stats()["dist_comps"] / len(qry)
        out.append((ef, recall(lab, gt), dc))
    return out


def iso_dc(c, tgt):
    pts = sorted((r, dc) for _, r, dc in c)
    for i in range(1, len(pts)):
        if pts[i][0] >= tgt:
            (r0, d0), (r1, d1) = pts[i-1], pts[i]
            return d0 + (d1-d0)*(tgt-r0)/max(1e-9, r1-r0)
    return float("nan")


def main():
    os.makedirs("results", exist_ok=True)
    base, qry = load(); print("GT..."); gt = exact_gt(base, qry, K)
    res = {}
    for a in ALPHAS:
        res[a] = curve(build(base, a), qry, gt)
        print(f"alpha={a}: built")

    print(f"\nMean distance comps/query at fixed recall (LOWER = faster, hardware-independent)")
    print(f"{'alpha':>6} {'dc@0.90':>9} {'dc@0.95':>9} {'dc@0.99':>9}   (alpha=1.0 == HNSW heuristic)")
    base_a = 1.0
    bdc = {t: iso_dc(res[base_a], t) for t in (0.90, 0.95, 0.99)}
    for a in ALPHAS:
        cells = []
        for t in (0.90, 0.95, 0.99):
            v = iso_dc(res[a], t)
            d = 100*(v-bdc[t])/bdc[t] if bdc[t]==bdc[t] else 0
            cells.append(f"{v:7.0f}({d:+4.0f}%)")
        print(f"{a:>6} " + " ".join(cells))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"alpha-RNG efficiency on real SIFT-128 (N={N_BASE:,}, M={M}) — hardware-independent",
                 fontsize=13, fontweight="bold")
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(ALPHAS)))
    for a, col in zip(ALPHAS, cmap):
        c = res[a]; lab = f"α={a}" + ("  (=HNSW)" if a == 1.0 else "")
        lw = 3 if a == 1.0 else 1.8
        ax[0].plot([p[1] for p in c], [p[2] for p in c], "o-", color=("black" if a==1.0 else col), lw=lw, ms=5, label=lab)
    ax[0].set_xlabel(f"recall@{K}"); ax[0].set_ylabel("mean distance comps / query (lower=faster)")
    ax[0].set_title("Recall vs distance computations  (lower curve = faster)")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3); ax[0].set_xlim(0.85, 1.003)

    ts = [0.90, 0.95, 0.99]; x = np.arange(len(ts)); w = 0.8/len(ALPHAS)
    for j, (a, col) in enumerate(zip(ALPHAS, cmap)):
        ax[1].bar(x - 0.4 + (j+0.5)*w, [iso_dc(res[a], t) for t in ts], w,
                  color=("black" if a==1.0 else col), label=f"α={a}")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"recall {t}" for t in ts])
    ax[1].set_ylabel("distance comps / query at fixed recall (shorter=faster)")
    ax[1].set_title("Cost at fixed recall (shorter = faster)"); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, axis="y")
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig("results/distcomp.png", dpi=150, bbox_inches="tight")
    json.dump({str(a): res[a] for a in ALPHAS}, open("results/distcomp.json", "w"), indent=2)
    print("\nsaved -> results/distcomp.png / .json")


if __name__ == "__main__":
    main()
