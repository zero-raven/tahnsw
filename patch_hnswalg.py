import re
import sys

def patch():
    with open("src/tahnsw_alg.h", "r") as f:
        content = f.read()

    # Include topology_math.h
    content = content.replace('#include "hnswlib.h"', '#include "hnswlib.h"\n#include "topology_math.h"')

    # Add members to HierarchicalNSW
    members = """
    tahnsw::TAHNSWConfig tahnsw_cfg;
    std::unique_ptr<tahnsw::TopologyAnalyser<dist_t>> topology_analyser;
    std::unique_ptr<tahnsw::LayerAssigner> layer_assigner;
    std::unique_ptr<tahnsw::AdaptiveDegree> adaptive_degree;
    std::unique_ptr<tahnsw::RNGPruner<dist_t>> rng_pruner;

    // TAHNSW stats
    mutable std::atomic<long> tahnsw_hub_count{0};
    mutable std::atomic<long> tahnsw_leaf_count{0};
    mutable std::atomic<long> tahnsw_mid_count{0};
    mutable std::atomic<long> tahnsw_edges_saved{0};
    mutable std::atomic<long> tahnsw_edges_baseline{0};
    std::vector<int> tahnsw_m_values_;

    void set_tahnsw_config(const tahnsw::TAHNSWConfig& cfg, SpaceInterface<dist_t> *s) {
        tahnsw_cfg = cfg;
        topology_analyser.reset(new tahnsw::TopologyAnalyser<dist_t>(tahnsw_cfg, s));
        layer_assigner.reset(new tahnsw::LayerAssigner(tahnsw_cfg));
        adaptive_degree.reset(new tahnsw::AdaptiveDegree(tahnsw_cfg));
        rng_pruner.reset(new tahnsw::RNGPruner<dist_t>(tahnsw_cfg, s));
        
        // ensure M_ is enough for alpha
        int M_max = static_cast<int>(M_ * tahnsw_cfg.M_alpha) + 2;
        maxM_ = M_max;
        maxM0_ = M_max * 2;
    }
"""
    content = content.replace("    bool allow_replace_deleted_ = false;", "    bool allow_replace_deleted_ = false;\n" + members)

    # Initialize vector in constructors
    content = content.replace("element_levels_(max_elements)", "element_levels_(max_elements), tahnsw_m_values_(max_elements)")

    # Modify addPoint signature to optionally skip TAHNSW logic? No, we just inject it.
    
    # Locate addPoint logic
    # Find:
    # int curlevel = getRandomLevel(mult_);
    # if (level > 0)
    #     curlevel = level;
    # element_levels_[cur_c] = curlevel;

    tahnsw_addpoint = """
        int curlevel = 0;
        int M_eff = M_;
        float C_v = 0.5f;
        float B_v_norm = 0.0f;
        std::vector<std::pair<dist_t, tableint>> candidates_for_prune;

        if (topology_analyser && cur_element_count > 2) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            
            // Search down to level 0 without locking global the whole time
            tableint currObj2;
            int maxlevelcopy2;
            {
                std::unique_lock<std::mutex> templock2(global);
                currObj2 = enterpoint_node_;
                maxlevelcopy2 = maxlevel_;
            }
            
            if ((signed)currObj2 != -1) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj2), dist_func_param_);
                for (int level_s = maxlevelcopy2; level_s > 0; level_s--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj2]);
                        data = get_linklist(currObj2, level_s);
                        int size = getListCount(data);
                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj2 = cand;
                                changed = true;
                            }
                        }
                    }
                }
                top_candidates = searchBaseLayer(currObj2, data_point, 0);
            }

            std::vector<std::pair<dist_t, tableint>> cands;
            cands.reserve(top_candidates.size());
            auto temp_q = top_candidates;
            while (!temp_q.empty()) {
                cands.push_back(temp_q.top());
                temp_q.pop();
            }
            std::reverse(cands.begin(), cands.end()); // closest first

            int k_topology = std::min(tahnsw_cfg.k_candidates, (int)cands.size());
            std::vector<std::pair<dist_t, tableint>> top_k_cands(cands.begin(), cands.begin() + k_topology);
            
            C_v = topology_analyser->clustering_coeff(top_k_cands, this);
            topology_analyser->update_sketch(cur_c, data_point, top_k_cands, this);
            B_v_norm = topology_analyser->get_normalised_centrality(cur_c);

            int max_layer_allowed = std::max(1, (int)(std::log(std::max((int)cur_element_count, 2)) / std::log(M_)));
            
            std::uniform_real_distribution<double> dist_unif(0.0, 1.0);
            float random_val = dist_unif(level_generator_);
            
            curlevel = layer_assigner->assign(C_v, B_v_norm, cur_element_count, M_, max_layer_allowed, random_val);
            M_eff = adaptive_degree->compute_M(C_v, B_v_norm, M_);
            
            if (C_v < tahnsw_cfg.cluster_low_thresh) {
                tahnsw_hub_count++;
            } else if (C_v > tahnsw_cfg.cluster_high_thresh) {
                tahnsw_leaf_count++;
            } else {
                tahnsw_mid_count++;
            }

            candidates_for_prune = cands;
        } else {
            curlevel = getRandomLevel(mult_);
            M_eff = M_;
        }
        
        tahnsw_m_values_[cur_c] = M_eff;
        if (level > 0) curlevel = level; // override
        element_levels_[cur_c] = curlevel;
"""
    content = re.sub(
        r"int curlevel = getRandomLevel\(mult_\);\s*if \(level > 0\)\s*curlevel = level;\s*element_levels_\[cur_c\] = curlevel;",
        tahnsw_addpoint,
        content
    )

    # Modify mutuallyConnectNewElement to take M_eff parameter
    content = content.replace(
        "tableint mutuallyConnectNewElement(",
        "tableint mutuallyConnectNewElement_TAHNSW(int M_eff,"
    )

    # In mutuallyConnectNewElement_TAHNSW, replace M_ with M_eff when selecting neighbors
    content = content.replace("getNeighborsByHeuristic2(top_candidates, M_);", "getNeighborsByHeuristic2(top_candidates, M_eff);")
    content = content.replace("if (top_candidates.size() > M_)", "if (top_candidates.size() > M_eff)")
    content = content.replace("selectedNeighbors.reserve(M_);", "selectedNeighbors.reserve(M_eff);")

    # In addPoint, replace calls to mutuallyConnectNewElement with mutuallyConnectNewElement_TAHNSW and RNG prune
    
    rng_prune_logic = """
                if (level == 0 && topology_analyser && tahnsw_cfg.prune_layer0_only && cur_element_count > 2) {
                    std::vector<std::pair<dist_t, tableint>> cands;
                    int initial_candidates_size = top_candidates.size();
                    while (!top_candidates.empty()) {
                        cands.push_back(top_candidates.top());
                        top_candidates.pop();
                    }
                    std::reverse(cands.begin(), cands.end());
                    
                    auto pruned = rng_pruner->prune(data_point, cands, M_eff, this);
                    
                    if (pruned.empty() && !cands.empty()) {
                        pruned.push_back(cands[0]);
                    }
                    
                    tahnsw_edges_baseline += std::min(M_eff, initial_candidates_size);
                    tahnsw_edges_saved += std::max(0, std::min(M_eff, initial_candidates_size) - (int)pruned.size());
                    
                    for (const auto& p : pruned) {
                        top_candidates.push(p);
                    }
                }
                mutuallyConnectNewElement_TAHNSW(M_eff, data_point, cur_c, top_candidates, level, false);
"""
    content = content.replace("mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);", rng_prune_logic)

    with open("src/tahnsw_alg.h", "w") as f:
        f.write(content)

if __name__ == "__main__":
    patch()
