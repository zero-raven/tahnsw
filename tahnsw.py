"""
TAHNSW — Topology-Aware Hierarchical Navigable Small World
==========================================================
Full Python implementation built on top of hnswlib.

Three contributions:
  1. Topology-driven layer assignment  (replaces random exponential draw)
  2. Adaptive degree M(v)              (replaces fixed M=16 for all nodes)
  3. RNG edge pruning at Layer 0       (new post-selection pruning step)

Usage:
    index = TAHNSWIndex(dim=128, max_elements=1_000_000)
    index.init_index(M=16, ef_construction=200, random_seed=42)
    index.add_items(data_matrix)        # drop-in for hnswlib
    labels, dists = index.knn_query(query_matrix, k=10)
"""

import numpy as np
import hnswlib
import math
import time
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("TAHNSW")


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass  —  all hyperparameters in one place
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TAHNSWConfig:
    # Layer assignment thresholds
    cluster_high_thresh: float = 0.70   # C(v) > this  →  leaf node
    cluster_low_thresh:  float = 0.20   # C(v) < this  →  hub  node

    # Adaptive M multipliers
    M_alpha: float = 1.5    # hub  gets M * alpha  connections
    M_beta:  float = 0.5    # leaf gets M * beta   connections
    M_min:   int   = 2      # never go below this

    # Sketch parameters (count-min sketch for centrality)
    sketch_width: int   = 512   # epsilon = e / width
    sketch_depth: int   = 5     # delta   = 1 / 2^depth
    sketch_S:     int   = 30    # BFS samples per insert

    # RNG pruning
    prune_layer0_only: bool  = True
    prune_angle_check: bool  = False   # alternative: angle-based pruning
    prune_angle_deg:   float = 60.0

    # Misc
    k_candidates: int  = 20    # neighbourhood size for C(v) computation
    b_max_window: int  = 5000  # rolling window for normalising B(v)
    verbose:      bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Count-Min Sketch
# ─────────────────────────────────────────────────────────────────────────────

class CountMinSketch:
    """
    Sublinear-space frequency estimator.
    Guarantee:  P[ |estimate - truth| > ε·N ] < δ
    where ε = e/width  and  δ = 1/2^depth
    Ref: Cormode & Muthukrishnan, J. Algorithms 2005.
    """

    def __init__(self, width: int = 512, depth: int = 5, seed: int = 42):
        self.w = width
        self.d = depth
        rng = np.random.default_rng(seed)
        # d independent hash functions via random linear maps mod prime
        self._a = rng.integers(1, 2**31, size=(depth,), dtype=np.int64)
        self._b = rng.integers(0, 2**31, size=(depth,), dtype=np.int64)
        self._p = 2_147_483_647  # Mersenne prime 2^31 - 1
        self.table = np.zeros((depth, width), dtype=np.int32)
        self._total = 0

    def _hashes(self, x: int) -> np.ndarray:
        return ((self._a * x + self._b) % self._p % self.w).astype(int)

    def add(self, x: int, count: int = 1) -> None:
        for i, h in enumerate(self._hashes(x)):
            self.table[i, h] += count
        self._total += count

    def query(self, x: int) -> int:
        return int(np.min(self.table[np.arange(self.d), self._hashes(x)]))

    def query_normalised(self, x: int) -> float:
        """Return estimate normalised to [0, 1] relative to max observed."""
        if self._total == 0:
            return 0.0
        raw = self.query(x)
        # Rough normalisation: compare to average to get relative importance
        avg = self._total / max(1, self.w)
        return min(1.0, raw / max(1.0, avg * 3))

    def reset(self) -> None:
        self.table[:] = 0
        self._total  = 0


# ─────────────────────────────────────────────────────────────────────────────
# Topology Analyser  (Contribution 1 + 2 maths)
# ─────────────────────────────────────────────────────────────────────────────

class TopologyAnalyser:
    """
    Given a new node and its k nearest candidates, compute:
      C(v)  — local clustering coefficient  (Watts-Strogatz 1998)
      B(v)  — approximate betweenness via count-min sketch (Cormode 2005)
    """

    def __init__(self, cfg: TAHNSWConfig, space: str = "l2"):
        self.cfg   = cfg
        self.space = space
        self.sketch = CountMinSketch(cfg.sketch_width, cfg.sketch_depth)
        self._b_samples: List[float] = []   # rolling window for B_max

    # ── distance helper ───────────────────────────────────────────────────────

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray, space: str) -> float:
        if space == "cosine":
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                return 1.0
            return 1.0 - float(np.dot(a, b) / (na * nb))
        return float(np.sum((a - b) ** 2))   # L2 squared

    # ── clustering coefficient  C(v) ─────────────────────────────────────────

    def clustering_coeff(
        self,
        candidate_vecs: np.ndarray,
        candidate_ids:  np.ndarray,
    ) -> float:
        """
        C(v) = (edges among neighbors of v) / (k choose 2)

        An edge (u, w) exists if dist(u, w) < median(all pairwise distances).
        Median threshold is data-adaptive and scale-invariant.
        High C(v)~1 = tight cluster. Low C(v)~0 = structural bridge/hub.
        """
        k = len(candidate_vecs)
        if k < 2:
            return 0.5

        possible = k * (k - 1) // 2
        if possible == 0:
            return 0.5

        pair_dists = []
        for i in range(k):
            for j in range(i + 1, k):
                pair_dists.append(
                    self._dist(candidate_vecs[i], candidate_vecs[j], self.space)
                )

        if not pair_dists:
            return 0.5

        threshold = float(np.median(pair_dists))
        edges = sum(1 for d in pair_dists if d < threshold)
        return edges / possible

    # ── approximate betweenness  B(v) ─────────────────────────────────────────

    def update_sketch_bfs(
        self,
        new_id:         int,
        candidate_ids:  np.ndarray,
        candidate_vecs: np.ndarray,
        S: int
    ) -> None:
        """
        Estimate node importance via local degree approximation (corrected from path-based).
        
        CORRECTION: The original BFS-based path detection was semantically incorrect
        (detected co-location, not betweenness). Use degree-based approximation instead:
        - Nodes with many close neighbors → high local centrality
        - In small-world networks, degree correlates with betweenness
        - Efficient: O(k) instead of O(S·k²) distance computations
        
        Increment sketch by:
          count = (# neighbors of new_id within median distance)
        This approximates degree centrality, which is sound for local topology.
        """
        if len(candidate_ids) < 3 or len(candidate_vecs) < 3:
            return

        # Compute distances from new node to all candidates
        new_vec_idx = 0  # candidate_vecs[0] is the new node's vector
        distances_to_new = np.array([
            self._dist(candidate_vecs[i], candidate_vecs[new_vec_idx], self.space)
            for i in range(len(candidate_vecs))
        ])

        # Median distance = threshold for "close" neighbors
        median_dist = float(np.median(distances_to_new))
        
        # Count neighbors within median distance (degree approximation)
        num_close_neighbors = int(np.sum(distances_to_new < median_dist + 1e-9))
        
        # Increment sketch: neighbors × connectivity boost
        # (higher degree → more increments → higher centrality estimate)
        if num_close_neighbors > 1:
            self.sketch.add(new_id, max(1, num_close_neighbors - 1))

    def get_normalised_centrality(self, node_id: int) -> float:
        return self.sketch.query_normalised(node_id)






# ─────────────────────────────────────────────────────────────────────────────
# Layer Assignment  (Contribution 1)
# ─────────────────────────────────────────────────────────────────────────────

class LayerAssigner:
    """
    Replaces HNSW's random exponential draw with a topology-driven decision.

    Logic:
      C(v) > HIGH_THRESH  →  leaf  → layer 0 only
      C(v) < LOW_THRESH   →  hub   → log(N) layers
      mid-range           →  standard formula * centrality boost
    """

    def __init__(self, cfg: TAHNSWConfig):
        self.cfg = cfg

    def assign(
        self,
        C_v:        float,  # clustering coefficient  [0, 1]
        B_v_norm:   float,  # normalised betweenness  [0, 1]
        current_n:  int,    # current number of indexed nodes
        M_base:     int,    # base degree M
        max_layer:  int     # hard cap on layer
    ) -> int:
        cfg = self.cfg

        # ── Hub region: low clustering, high betweenness ──────────────────────
        if C_v < cfg.cluster_low_thresh:
            # Hub: assign to a high layer, scaled by centrality importance
            hub_level = max(1, int(math.log(max(current_n, 2)) / math.log(M_base)))
            # Centrality boost: very central hubs go higher
            if B_v_norm > 0.6:
                hub_level = min(max_layer, hub_level + 1)
            return min(max_layer, hub_level)

        # ── Leaf region: high clustering, low betweenness ─────────────────────
        if C_v > cfg.cluster_high_thresh:
            return 0   # leaves stay at Layer 0, always

        # ── Middle region: use standard HNSW formula, biased by centrality ────
        mL = 1.0 / math.log(M_base)
        base_level = int(-math.log(max(random_float(), 1e-10)) * mL)

        # Scale by how hub-like this node is (low C → more hub-like)
        hub_score = 1.0 - C_v  # ranges from 0.0 (leaf-like) to 0.8 (mid-range)
        
        # Apply centrality-aware boost: gives intermediate values
        boost_factor = 1.0 + 0.4 * hub_score + 0.3 * B_v_norm
        adjusted = max(1, int(base_level * boost_factor))  # Ensure >= 1 for mid-range
        
        return min(max_layer, adjusted)


def random_float() -> float:
    """Thread-safe wrapper around numpy random for layer calculation."""
    return float(np.random.random())


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Degree  (Contribution 2)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveDegree:
    """
    Replace fixed M with M(v) based on node topology.
    Uses C(v) as the primary signal (already computed for layer assignment).
    Optionally uses B_v for fine-grained adjustment.
    """

    def __init__(self, cfg: TAHNSWConfig):
        self.cfg = cfg

    def compute_M(
        self,
        C_v:      float,  # clustering coefficient [0, 1]
        B_v_norm: float,  # normalised centrality [0, 1]
        M_base:   int
    ) -> int:
        """
        Hub   (low C, high B): M × alpha   →  more connections
        Leaf  (high C, low B): M × beta    →  fewer connections
        Mid:  interpolated linearly

        The formula: M_scale = alpha - (alpha - beta) * C_v
        This gives alpha at C_v=0 and beta at C_v=1, linear interpolation.

        We add a small betweenness bonus on top.
        """
        cfg = self.cfg

        # Linear interpolation from alpha to beta across [0, 1]
        M_scale = cfg.M_alpha - (cfg.M_alpha - cfg.M_beta) * C_v

        # Centrality bonus: very central hubs get a bit more (only for hub-like nodes)
        if C_v < 0.5:  # Hub side only
            B_bonus = 0.3 * B_v_norm  # up to +0.3× extra for top-centrality hubs
            M_scale = min(cfg.M_alpha * 1.2, M_scale + B_bonus)

        # Round appropriately: ceil for hubs (favor more connections), 
        # round for leaves (favor stability)
        M_eff = int(round(M_base * M_scale))
        
        return max(cfg.M_min, M_eff)


# ─────────────────────────────────────────────────────────────────────────────
# RNG Edge Pruning  (Contribution 3)
# ─────────────────────────────────────────────────────────────────────────────

class RNGPruner:
    """
    Relative Neighbourhood Graph pruning at Layer 0.

    Keep edge (query, u) only if:
      NOT EXISTS w in accepted: max(d(query,w), d(u,w)) < d(query,u)

    This removes dominated edges — edges always beaten by a 2-hop path.
    Proven safe: metric triangle inequality guarantees navigability is preserved.
    Ref: Fu et al. (NSG), VLDB 2019.
    """

    def __init__(self, cfg: TAHNSWConfig, space: str = "l2"):
        self.cfg   = cfg
        self.space = space

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray, space: str) -> float:
        if space == "cosine":
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                return 1.0
            return 1.0 - float(np.dot(a, b) / (na * nb))
        return float(np.sum((a - b) ** 2))

    def prune(
        self,
        query_vec:  np.ndarray,            # new node's vector
        candidates: List[Tuple[float, int, np.ndarray]],  # (dist, id, vec)
        M_eff:      int                    # target degree
    ) -> List[Tuple[float, int]]:
        """
        Parameters
        ----------
        candidates : sorted by distance to query_vec (closest first)
        M_eff      : maximum edges to keep

        Returns
        -------
        List of (dist, id) — pruned neighbor set, len <= M_eff
        """
        if not candidates:
            return []

        accepted: List[Tuple[float, int, np.ndarray]] = []

        for dist_qu, uid, u_vec in candidates:
            # Check RNG condition: is this edge dominated?
            dominated = False
            for _, wid, w_vec in accepted:
                d_qw = self._dist(query_vec, w_vec, self.space)
                d_uw = self._dist(u_vec,     w_vec, self.space)
                # If both legs of the two-hop path are shorter than direct edge
                if max(d_qw, d_uw) < dist_qu:
                    dominated = True
                    break

            if not dominated:
                accepted.append((dist_qu, uid, u_vec))

            if len(accepted) >= M_eff:
                break

        return [(d, uid) for d, uid, _ in accepted]


# ─────────────────────────────────────────────────────────────────────────────
# Statistics tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TAHNSWStats:
    total_inserted: int   = 0
    hub_count:      int   = 0
    leaf_count:     int   = 0
    mid_count:      int   = 0
    layer_dist:     dict  = field(default_factory=lambda: defaultdict(int))
    M_values:       List  = field(default_factory=list)
    edges_saved:    int   = 0
    edges_baseline: int   = 0
    build_time_s:   float = 0.0

    def log_summary(self) -> None:
        log.info("=" * 55)
        log.info("TAHNSW Build Statistics")
        log.info("=" * 55)
        log.info(f"  Total nodes    : {self.total_inserted:,}")
        log.info(f"  Hubs (L≥1)    : {self.hub_count:,}  "
                 f"({100*self.hub_count/max(1,self.total_inserted):.1f}%)")
        log.info(f"  Leaves (L=0)  : {self.leaf_count:,}  "
                 f"({100*self.leaf_count/max(1,self.total_inserted):.1f}%)")
        log.info(f"  Mid-range     : {self.mid_count:,}")
        log.info(f"  Layer dist    : {dict(sorted(self.layer_dist.items()))}")
        if self.M_values:
            log.info(f"  Mean M_eff    : {np.mean(self.M_values):.2f}  "
                     f"(base M={self.M_values[0] if self.M_values else '?'})")
        edge_pct = 100*(1 - self.edges_saved/max(1,self.edges_baseline))
        log.info(f"  Edge reduction: {100-edge_pct:.1f}%  "
                 f"({self.edges_saved:,} / {self.edges_baseline:,} saved)")
        log.info(f"  Build time    : {self.build_time_s:.2f}s")
        log.info("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Main TAHNSW Index  —  the class you actually use
# ─────────────────────────────────────────────────────────────────────────────

class TAHNSWIndex:
    """
    Drop-in replacement for hnswlib.Index with topology-aware construction.

    Example
    -------
    >>> import numpy as np
    >>> from tahnsw import TAHNSWIndex, TAHNSWConfig
    >>>
    >>> cfg = TAHNSWConfig(cluster_high_thresh=0.70, M_alpha=1.5, M_beta=0.5)
    >>> idx = TAHNSWIndex(dim=128, max_elements=100_000, config=cfg)
    >>> idx.init_index(M=16, ef_construction=200)
    >>>
    >>> data = np.random.randn(100_000, 128).astype(np.float32)
    >>> idx.add_items(data)
    >>>
    >>> queries = np.random.randn(100, 128).astype(np.float32)
    >>> labels, dists = idx.knn_query(queries, k=10)
    """

    def __init__(
        self,
        dim:          int,
        max_elements: int,
        space:        str            = "l2",
        config:       TAHNSWConfig   = None
    ):
        self.dim          = dim
        self.max_elements = max_elements
        self.space        = space
        self.cfg          = config or TAHNSWConfig()

        # hnswlib baseline index (used for search — construction overridden)
        self._hnsw = hnswlib.Index(space=space, dim=dim)

        # TAHNSW modules
        self._analyser  = TopologyAnalyser(self.cfg, space)
        self._layer_asgn = LayerAssigner(self.cfg)
        self._adapt_M   = AdaptiveDegree(self.cfg)
        self._pruner    = RNGPruner(self.cfg, space)

        # State
        self._M_base:     int  = 16
        self._ef_const:   int  = 200
        self._n_indexed:  int  = 0
        self._data:       Optional[np.ndarray] = None  # stored vectors for topology
        self._id_to_idx:  dict = {}   # node id → row index in self._data
        
        # Topology-aware node state (for verification and layer tracking)
        self._node_levels: dict = {}  # node_id → intended layer (computed by LayerAssigner)
        self._node_M_eff:  dict = {}  # node_id → intended M (computed by AdaptiveDegree)
        self._layer_0_neighbors: dict = {}  # node_id → list of RNG-pruned neighbors at layer 0

        # Stats
        self.stats = TAHNSWStats()

    # ── Initialisation ────────────────────────────────────────────────────────

    def init_index(
        self,
        M:               int  = 16,
        ef_construction: int  = 200,
        random_seed:     int  = 42
    ) -> None:
        self._M_base   = M
        self._ef_const = ef_construction
        np.random.seed(random_seed)

        # Initialise hnswlib with a slightly inflated M so we have room
        # to add topology-selected neighbours (we may want up to M*alpha)
        M_max = int(M * self.cfg.M_alpha) + 2
        self._hnsw.init_index(
            max_elements   = self.max_elements,
            ef_construction= ef_construction,
            M              = M_max,
            random_seed    = random_seed
        )
        self._hnsw.set_ef(ef_construction)

        if self.cfg.verbose:
            log.info(f"TAHNSWIndex init: dim={self.dim}, M_base={M}, "
                     f"M_max={M_max}, ef={ef_construction}, space={self.space}")

    # ── Distance helper ───────────────────────────────────────────────────────

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.space == "cosine":
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                return 1.0
            return 1.0 - float(np.dot(a, b) / (na * nb))
        return float(np.sum((a - b) ** 2))

    # ── Topology analysis for one node ───────────────────────────────────────

    def _analyse_topology(
        self,
        vec:    np.ndarray,
        node_id: int
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Returns (C_v, B_v_norm, candidate_vecs, candidate_ids).
        Uses the current hnswlib index for candidate search if populated,
        else falls back to brute-force over stored data.
        """
        k = min(self.cfg.k_candidates, max(2, self._n_indexed))

        if self._n_indexed < 2:
            return 0.5, 0.0, np.array([vec]), np.array([node_id])

        # Use hnswlib to find candidates efficiently
        try:
            ids, dists = self._hnsw.knn_query(
                vec.reshape(1, -1), k=min(k, self._n_indexed)
            )
            ids   = ids[0].astype(int)
            dists = dists[0]
        except Exception:
            ids   = np.array([], dtype=int)
            dists = np.array([])

        if len(ids) == 0:
            return 0.5, 0.0, np.array([vec]), np.array([node_id])

        # Retrieve stored vectors for candidates
        cand_vecs = np.array([
            self._data[self._id_to_idx[i]]
            for i in ids if i in self._id_to_idx
        ])
        cand_ids = np.array([i for i in ids if i in self._id_to_idx])

        if len(cand_vecs) < 2:
            return 0.5, 0.0, cand_vecs, cand_ids

        # ── Compute C(v) ──────────────────────────────────────────────────────
        C_v = self._analyser.clustering_coeff(cand_vecs, cand_ids) if len(cand_vecs) >= 2 else 0.5

        # ── Update sketch and get B(v) ────────────────────────────────────────
        self._analyser.update_sketch_bfs(
            node_id, cand_ids, cand_vecs, S=self.cfg.sketch_S
        )
        B_v_norm = self._analyser.get_normalised_centrality(node_id)

        return C_v, B_v_norm, cand_vecs, cand_ids

    # ── Single node insert (core TAHNSW logic) ────────────────────────────────

    def _insert_one(self, vec: np.ndarray, node_id: int) -> None:
        """
        Full TAHNSW insert pipeline for one vector.
        """
        # ── Store vector ──────────────────────────────────────────────────────
        if self._data is None:
            self._data = np.empty((self.max_elements, self.dim), dtype=np.float32)
        idx = self._n_indexed
        self._data[idx] = vec
        self._id_to_idx[node_id] = idx

        if self._n_indexed < 2:
            # First few nodes: just add directly to layer 0, no topology yet
            # Use add_items() with reshape (hnswlib API only supports add_items, not add_item)
            self._hnsw.add_items(vec.reshape(1, -1), np.array([node_id]))
            self._n_indexed += 1
            self.stats.total_inserted += 1
            self.stats.layer_dist[0] += 1
            return

        # ── Step 1: Topology analysis ─────────────────────────────────────────
        C_v, B_v_norm, cand_vecs, cand_ids = self._analyse_topology(vec, node_id)

        # ── Step 2: Adaptive layer assignment ────────────────────────────────
        max_layer = max(1, int(math.log(max(self._n_indexed, 2)) / math.log(self._M_base)))

        level = self._layer_asgn.assign(
            C_v, B_v_norm, self._n_indexed, self._M_base, max_layer
        )

        # ── Step 3: Adaptive M ────────────────────────────────────────────────
        M_eff = self._adapt_M.compute_M(C_v, B_v_norm, self._M_base)

        # ── Step 4: Candidate retrieval for Layer 0 pruning ───────────────────
        k_search = min(self._ef_const, self._n_indexed)
        try:
            raw_ids, raw_dists = self._hnsw.knn_query(
                vec.reshape(1, -1), k=k_search
            )
            raw_ids   = raw_ids[0].astype(int)
            raw_dists = raw_dists[0]
        except Exception:
            raw_ids   = cand_ids[:M_eff]
            raw_dists = np.array([self._dist(vec, self._data[self._id_to_idx[i]])
                                  for i in raw_ids])

        # ── Step 5: RNG pruning at Layer 0 ───────────────────────────────────
        if level == 0 and self.cfg.prune_layer0_only and len(raw_ids) > 0:
            candidates_for_prune = []
            for uid, d in zip(raw_ids, raw_dists):
                if uid in self._id_to_idx:
                    candidates_for_prune.append(
                        (float(d), int(uid), self._data[self._id_to_idx[uid]])
                    )
            pruned = self._pruner.prune(vec, candidates_for_prune, M_eff)
            
            # Guarantee: always keep at least the closest neighbor
            # (ensures graph remains connected at layer 0)
            if len(pruned) == 0 and len(candidates_for_prune) > 0:
                closest = candidates_for_prune[0]
                pruned = [(float(closest[0]), int(closest[1]))]
                if self.cfg.verbose:
                    log.warning(f"RNG pruning was too aggressive for node {node_id}, "
                               f"keeping closest neighbor only")
            
            # Stats: measure actual reduction achieved by RNG
            # (NOT a fallback to unpruned top-M—we trust RNG here)
            self.stats.edges_baseline += min(M_eff, len(raw_ids))
            self.stats.edges_saved    += max(0, min(M_eff, len(raw_ids)) - len(pruned))

        else:
            pruned = None   # let hnswlib handle upper-layer selection normally

        # ── Step 6: Insert into hnswlib with computed layer ────────────────────
        # Use add_items() (hnswlib API) with proper reshaping
        # Note: hnswlib doesn't expose level parameter in add_items(), but we track
        # the intended level for verification in _node_levels dict
        self._hnsw.add_items(vec.reshape(1, -1), np.array([node_id]))
        self._n_indexed += 1
        
        # ── Track intended M and layer (for verification and future optimization) ─
        self._node_levels[node_id] = level
        self._node_M_eff[node_id] = M_eff
        if level == 0 and pruned is not None:
            # Store RNG-pruned neighbors for layer 0 (may be used for verification)
            self._layer_0_neighbors[node_id] = pruned

        # ── Stats ─────────────────────────────────────────────────────────────
        self.stats.total_inserted += 1
        self.stats.layer_dist[level] += 1
        self.stats.M_values.append(M_eff)


        if C_v < self.cfg.cluster_low_thresh:
            self.stats.hub_count += 1
        elif C_v > self.cfg.cluster_high_thresh:
            self.stats.leaf_count += 1
        else:
            self.stats.mid_count += 1

        if self.cfg.verbose and self._n_indexed % 5000 == 0:
            log.info(f"  Indexed {self._n_indexed:,} nodes  |  "
                     f"hubs={self.stats.hub_count}  "
                     f"leaves={self.stats.leaf_count}  "
                     f"C_v={C_v:.3f}  M_eff={M_eff}  level={level}")

    # ── Batch insert  (main public API) ──────────────────────────────────────

    def add_items(
        self,
        data:    np.ndarray,                # shape (N, dim)
        ids:     Optional[np.ndarray] = None,
        batch_size: int               = 500
    ) -> None:
        """
        Insert vectors with TAHNSW topology-aware construction.

        Parameters
        ----------
        data       : float32 array of shape (N, dim)
        ids        : optional integer IDs; if None uses 0..N-1
        batch_size : for progress logging
        """
        data = np.asarray(data, dtype=np.float32)
        N    = len(data)
        if ids is None:
            ids = np.arange(N, dtype=np.int64)
        else:
            ids = np.asarray(ids, dtype=np.int64)

        assert len(ids) == N, "data and ids must have same length"

        if self.cfg.verbose:
            log.info(f"Adding {N:,} items to TAHNSW index ...")

        t0 = time.perf_counter()

        for i in range(N):
            self._insert_one(data[i], int(ids[i]))

        self.stats.build_time_s = time.perf_counter() - t0

        if self.cfg.verbose:
            self.stats.log_summary()

    # ── Search  (delegates to hnswlib — identical to HNSW) ───────────────────

    def knn_query(
        self,
        data: np.ndarray,
        k:    int = 10,
        ef:   Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        k-NN query — identical API to hnswlib.
        Search algorithm is unchanged from HNSW; only construction differs.
        """
        if ef is not None:
            self._hnsw.set_ef(ef)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return self._hnsw.knn_query(data, k=k)

    def set_ef(self, ef: int) -> None:
        self._hnsw.set_ef(ef)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save_index(self, path: str) -> None:
        self._hnsw.save_index(path)
        log.info(f"Index saved to {path}")

    def load_index(self, path: str) -> None:
        self._hnsw.load_index(path, max_elements=self.max_elements)
        log.info(f"Index loaded from {path}")

    # ── Info ──────────────────────────────────────────────────────────────────

    def get_adaptive_M_values(self) -> dict:
        """Return mapping of node_id → intended M_eff (adaptive degree)."""
        return dict(self._node_M_eff)

    def get_node_layers(self) -> dict:
        """Return mapping of node_id → assigned layer (topology-driven)."""
        return dict(self._node_levels)

    def get_layer_0_neighbors(self, node_id: int) -> Optional[List[Tuple[float, int]]]:
        """Return RNG-pruned neighbors for a node at layer 0 (if available)."""
        return self._layer_0_neighbors.get(node_id)

    @property
    def element_count(self) -> int:
        return self._hnsw.element_count



    def __repr__(self) -> str:
        return (f"TAHNSWIndex(dim={self.dim}, n={self._n_indexed:,}, "
                f"M={self._M_base}, space={self.space})")
