"""Noise-resistant HNSW vs TAHNSW-C++ comparison.
Interleaved best-of-N build timing + best-of-N search QPS."""
import time, numpy as np
import hnswlib, tahnsw_cpp

N, DIM, K = 20000, 128, 10
M, EFC = 16, 200
EFS = [10, 40, 80, 120, 200]
NThr = 1
BUILD_REPS = 3
Q_REPS = 8

def synth(N, dim, n_clusters=20, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    pts = [c + rng.standard_normal((N//n_clusters, dim)).astype(np.float32)*0.15 for c in centers]
    data = np.vstack(pts)[:N]
    qc = centers + rng.standard_normal(centers.shape).astype(np.float32)*0.05
    qs = [c + rng.standard_normal((10, dim)).astype(np.float32)*0.12 for c in qc]
    return data, np.vstack(qs)[:200]

def bf(data, q, k):
    out = np.zeros((len(q), k), dtype=int)
    for i, x in enumerate(q):
        out[i] = np.argpartition(np.sum((data-x)**2, axis=1), k)[:k]
    return out

def recall(pred, gt):
    n, k = pred.shape
    return sum(len(set(pred[i]) & set(gt[i,:k])) for i in range(n))/(n*k)

def mk_hnsw(data):
    idx = hnswlib.Index(space="l2", dim=DIM)
    idx.init_index(max_elements=N, ef_construction=EFC, M=M); idx.set_num_threads(NThr)
    t=time.perf_counter(); idx.add_items(data, np.arange(N)); return idx, time.perf_counter()-t

def mk_tahnsw(data):
    idx = tahnsw_cpp.IndexCpp(space="l2", dim=DIM)
    idx.init_index(max_elements=N, ef_construction=EFC, M=M); idx.set_tahnsw_config(); idx.set_num_threads(NThr)
    t=time.perf_counter(); idx.add_items(data, np.arange(N)); return idx, time.perf_counter()-t

def curve(idx, q, gt):
    idx.set_num_threads(NThr); res=[]
    for ef in EFS:
        idx.set_ef(ef); idx.knn_query(q, k=K)
        best=1e9; lab=None
        for _ in range(Q_REPS):
            t=time.perf_counter(); lab,_=idx.knn_query(q,k=K); best=min(best,time.perf_counter()-t)
        res.append((ef, recall(lab,gt), len(q)/best))
    return res

if __name__ == "__main__":
    print(f"N={N} dim={DIM} M={M} efc={EFC} threads={NThr} build_reps={BUILD_REPS} q_reps={Q_REPS}")
    data, q = synth(N, DIM); gt = bf(data, q, K)
    hb, tb = [], []; hi=ti=None
    for r in range(BUILD_REPS):                    # interleave to share noise
        i,t = mk_hnsw(data); hb.append(t); hi=i
        i,t = mk_tahnsw(data); tb.append(t); ti=i
    hbt, tbt = min(hb), min(tb)
    st = ti.get_tahnsw_stats()
    print(f"\nBUILD (best of {BUILD_REPS}): HNSW {hbt:.2f}s | TAHNSW {tbt:.2f}s | ratio {tbt/hbt:.2f}x")
    print(f"  HNSW runs {[round(x,2) for x in hb]}  TAHNSW runs {[round(x,2) for x in tb]}")
    print(f"  stats: hubs={st['hub_count']} leaves={st['leaf_count']} mid={st['mid_count']}")
    hc, tc = curve(hi,q,gt), curve(ti,q,gt)
    print(f"\n{'ef':>5} {'HNSW rec':>9} {'HNSW QPS':>10} {'TA rec':>9} {'TA QPS':>10} {'QPS d%':>8}")
    for (e,hr,hq),(_,tr,tq) in zip(hc,tc):
        print(f"{e:>5} {hr:>9.4f} {hq:>10.0f} {tr:>9.4f} {tq:>10.0f} {100*(tq-hq)/hq:>+7.1f}%")
    # iso-recall: QPS where each first reaches recall>=0.97
    def iso(c, thr=0.97):
        for e,r,qps in c:
            if r>=thr: return e,r,qps
        return c[-1]
    he,hr,hq = iso(hc); te,tr,tq = iso(tc)
    print(f"\nISO-RECALL>=0.97: HNSW ef={he} rec={hr:.3f} {hq:.0f}qps | TAHNSW ef={te} rec={tr:.3f} {tq:.0f}qps | {100*(tq-hq)/hq:+.1f}%")
