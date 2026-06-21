#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>

// topology_math.h is included from hnswalg.h, which is included from hnswlib.h.
// By the time we get here, hnswlib.h has already defined SpaceInterface, labeltype, etc.
// We only need forward declarations for types defined in hnswalg.h itself
// (which hasn't finished defining yet when we're included).

namespace hnswlib {
    // These are defined in hnswalg.h but we need them here.
    // hnswalg.h will also define them — C++ allows identical typedefs.
    typedef unsigned int tableint;

    // Forward declaration — full definition is in hnswalg.h
    template<typename dist_t> class HierarchicalNSW;
}

namespace tahnsw {

// ─────────────────────────────────────────────────────────────────────────────
// Config structure
// ─────────────────────────────────────────────────────────────────────────────
struct TAHNSWConfig {
    float cluster_high_thresh = 0.70f;
    float cluster_low_thresh = 0.20f;
    float M_alpha = 1.5f;
    float M_beta = 0.5f;
    int M_min = 2;
    
    int sketch_width = 512;
    int sketch_depth = 5;
    int sketch_S = 30;
    
    bool prune_layer0_only = true;
    bool prune_angle_check = false;
    float prune_angle_deg = 60.0f;
    
    int k_candidates = 20;
    int b_max_window = 5000;

    // Neighbourhood-radius multiplier for the clustering coefficient C(v):
    // an edge between two neighbours counts if their distance < mult * mean(d to query).
    float cluster_radius_mult = 1.0f;

    // alpha for the neighbour-selection heuristic (DiskANN/Vamana RobustPrune).
    // A candidate c is dropped if an accepted neighbour w has alpha*d(w,c) < d(query,c).
    // alpha=1.0 == stock HNSW RNG heuristic; alpha>1 keeps more diverse/long-range
    // edges (denser, more navigable); alpha<1 prunes harder (sparser).
    float heuristic_alpha = 1.0f;

    // When true (default) the optimized construction path is used:
    //   - reuse the topology pre-pass candidates instead of a 2nd level-0 search
    //   - keep HNSW's exponential layer distribution (no level>=1 floor)
    //   - cap reverse-link degree per node instead of the inflated global maxM0_
    // Set false to reproduce the original (slow) construction for A/B benchmarks.
    bool fast_construction = true;
};

// ─────────────────────────────────────────────────────────────────────────────
// Count-Min Sketch
// ─────────────────────────────────────────────────────────────────────────────
class CountMinSketch {
    int w, d;
    std::vector<long long> a, b;  // long long to avoid overflow with large prime
    long long p;
    std::vector<std::vector<int>> table;
    long long total;

public:
    CountMinSketch(int width = 512, int depth = 5, int seed = 42) : w(width), d(depth), p(2147483647LL), total(0) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<long long> dist_a(1, p - 1);
        std::uniform_int_distribution<long long> dist_b(0, p - 1);
        
        a.resize(depth);
        b.resize(depth);
        table.resize(depth, std::vector<int>(width, 0));
        
        for (int i = 0; i < depth; ++i) {
            a[i] = dist_a(rng);
            b[i] = dist_b(rng);
        }
    }

    std::vector<int> hashes(int x) const {
        std::vector<int> h(d);
        for (int i = 0; i < d; ++i) {
            long long val = (a[i] * static_cast<long long>(x) + b[i]) % p;
            if (val < 0) val += p;  // ensure positive after modulo
            h[i] = static_cast<int>(val % w);
        }
        return h;
    }

    void add(int x, int count = 1) {
        auto h = hashes(x);
        for (int i = 0; i < d; ++i) {
            table[i][h[i]] += count;
        }
        total += count;
    }

    int query(int x) const {
        auto h = hashes(x);
        int min_val = table[0][h[0]];
        for (int i = 1; i < d; ++i) {
            min_val = std::min(min_val, table[i][h[i]]);
        }
        return min_val;
    }

    float query_normalised(int x) const {
        if (total == 0) return 0.0f;
        int raw = query(x);
        float avg = static_cast<float>(total) / std::max(1, w);
        float norm = static_cast<float>(raw) / std::max(1.0f, avg * 3.0f);
        return std::min(1.0f, norm);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Topology Analyser
// ─────────────────────────────────────────────────────────────────────────────
template<typename dist_t>
class TopologyAnalyser {
    TAHNSWConfig cfg;
    CountMinSketch sketch;
    hnswlib::SpaceInterface<dist_t>* space;

public:
    TopologyAnalyser(TAHNSWConfig config, hnswlib::SpaceInterface<dist_t>* s) 
        : cfg(config), sketch(config.sketch_width, config.sketch_depth), space(s) {}

    float clustering_coeff(const std::vector<std::pair<dist_t, hnswlib::tableint>>& candidates, 
                           hnswlib::HierarchicalNSW<dist_t>* index) {
        int k = static_cast<int>(candidates.size());
        if (k < 2) return 0.5f;

        int possible = k * (k - 1) / 2;
        if (possible == 0) return 0.5f;

        // Scale = the neighbourhood radius, taken from the query→candidate
        // distances the caller already computed (candidates[i].first). Using an
        // ABSOLUTE scale is the whole point: the previous version thresholded on
        // the median of the candidate-pair distances, which by construction makes
        // ~half the pairs "edges" → C(v) ≈ 0.5 for EVERY node (no signal).
        //
        // Two neighbours are "connected" if they sit within the neighbourhood
        // radius of each other. A tight cluster (leaf) → most pairs connected →
        // C(v)→1. A bridge spanning clusters (hub) → cross pairs far apart →
        // C(v)→0. The radius scales by cfg.cluster_radius_mult (tunable).
        dist_t mean_q = 0;
        for (int i = 0; i < k; ++i) mean_q += candidates[i].first;
        mean_q /= static_cast<dist_t>(k);
        if (mean_q <= 0) return 0.5f;
        dist_t threshold = mean_q * static_cast<dist_t>(cfg.cluster_radius_mult);

        int edges = 0;
        for (int i = 0; i < k; ++i) {
            char* data_i = index->getDataByInternalId(candidates[i].second);
            for (int j = i + 1; j < k; ++j) {
                char* data_j = index->getDataByInternalId(candidates[j].second);
                dist_t d = space->get_dist_func()(data_i, data_j, space->get_dist_func_param());
                if (d < threshold) edges++;
            }
        }

        return static_cast<float>(edges) / possible;
    }

    void update_sketch(hnswlib::tableint new_id, const void* new_data, 
                       const std::vector<std::pair<dist_t, hnswlib::tableint>>& candidates,
                       hnswlib::HierarchicalNSW<dist_t>* index) {
        if (candidates.size() < 3) return;

        std::vector<dist_t> dists_to_new;
        dists_to_new.reserve(candidates.size());
        for (const auto& c : candidates) {
            char* cand_data = index->getDataByInternalId(c.second);
            dists_to_new.push_back(space->get_dist_func()(new_data, cand_data, space->get_dist_func_param()));
        }

        std::vector<dist_t> sorted_dists = dists_to_new;
        std::nth_element(sorted_dists.begin(), sorted_dists.begin() + sorted_dists.size() / 2, sorted_dists.end());
        dist_t median_dist = sorted_dists[sorted_dists.size() / 2];

        int num_close = 0;
        for (dist_t d : dists_to_new) {
            if (d < median_dist + 1e-9) num_close++;
        }

        if (num_close > 1) {
            sketch.add(new_id, std::max(1, num_close - 1));
        }
    }

    float get_normalised_centrality(int node_id) const {
        return sketch.query_normalised(node_id);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Layer Assigner
// ─────────────────────────────────────────────────────────────────────────────
class LayerAssigner {
    TAHNSWConfig cfg;
public:
    LayerAssigner(TAHNSWConfig config) : cfg(config) {}

    int assign(float C_v, float B_v_norm, int current_n, int M_base, int max_layer, float random_val) {
        if (C_v < cfg.cluster_low_thresh) {
            int hub_level = std::max(1, static_cast<int>(std::log(std::max(current_n, 2)) / std::log(M_base)));
            if (B_v_norm > 0.6f) {
                hub_level = std::min(max_layer, hub_level + 1);
            }
            return std::min(max_layer, hub_level);
        }

        if (C_v > cfg.cluster_high_thresh) {
            return 0;
        }

        double mL = 1.0 / std::log(M_base);
        int base_level = static_cast<int>(-std::log(std::max(random_val, 1e-10f)) * mL);

        float hub_score = 1.0f - C_v;
        float boost_factor = 1.0f + 0.4f * hub_score + 0.3f * B_v_norm;
        // OPT(build+search): keep HNSW's exponential layer distribution. The old
        // std::max(1, ...) floor forced EVERY mid node to level >= 1, so ~100% of
        // nodes ended up in the upper layers (vs ~1/ln(M) in real HNSW). That made
        // each insert run a full search at level 1 too, and bloated the hierarchy.
        // A base_level draw of 0 (the common case) must stay at level 0; the
        // topology boost only lifts nodes that already drew a higher level.
        // (Legacy path keeps the old std::max(1,...) floor for A/B comparison.)
        int adjusted = cfg.fast_construction
            ? static_cast<int>(base_level * boost_factor)
            : std::max(1, static_cast<int>(base_level * boost_factor));

        return std::min(max_layer, adjusted);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Adaptive Degree
// ─────────────────────────────────────────────────────────────────────────────
class AdaptiveDegree {
    TAHNSWConfig cfg;
public:
    AdaptiveDegree(TAHNSWConfig config) : cfg(config) {}

    int compute_M(float C_v, float B_v_norm, int M_base) {
        float M_scale = cfg.M_alpha - (cfg.M_alpha - cfg.M_beta) * C_v;
        if (C_v < 0.5f) {
            float B_bonus = 0.3f * B_v_norm;
            M_scale = std::min(cfg.M_alpha * 1.2f, M_scale + B_bonus);
        }
        int M_eff = static_cast<int>(std::round(M_base * M_scale));
        return std::max(cfg.M_min, M_eff);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// RNG Pruner
// ─────────────────────────────────────────────────────────────────────────────
template<typename dist_t>
class RNGPruner {
    TAHNSWConfig cfg;
    hnswlib::SpaceInterface<dist_t>* space;
public:
    RNGPruner(TAHNSWConfig config, hnswlib::SpaceInterface<dist_t>* s) : cfg(config), space(s) {}

    std::vector<std::pair<dist_t, hnswlib::tableint>> prune(
        const void* query_vec, 
        const std::vector<std::pair<dist_t, hnswlib::tableint>>& candidates, 
        int M_eff,
        hnswlib::HierarchicalNSW<dist_t>* index) {
        
        if (candidates.empty()) return {};

        std::vector<std::pair<dist_t, hnswlib::tableint>> accepted;
        accepted.reserve(M_eff);

        for (const auto& cand : candidates) {
            bool dominated = false;
            char* u_vec = index->getDataByInternalId(cand.second);

            for (const auto& w : accepted) {
                char* w_vec = index->getDataByInternalId(w.second);
                dist_t d_qw = w.first;
                dist_t d_uw = space->get_dist_func()(u_vec, w_vec, space->get_dist_func_param());

                if (std::max(d_qw, d_uw) < cand.first) {
                    dominated = true;
                    break;
                }
            }

            if (!dominated) {
                accepted.push_back(cand);
            }

            if (static_cast<int>(accepted.size()) >= M_eff) {
                break;
            }
        }
        return accepted;
    }
};

} // namespace tahnsw
