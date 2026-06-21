import re

def patch():
    # Overwrite hnswalg.h with tahnsw_alg.h
    import shutil
    shutil.copy("src/tahnsw_alg.h", "src/hnswlib/hnswlib/hnswalg.h")

    # Now modify bindings.cpp to expose set_tahnsw_config
    with open("src/hnswlib/python_bindings/bindings.cpp", "r") as f:
        content = f.read()

    # Add set_tahnsw_config to the Index class
    set_config_cpp = """
    void set_tahnsw_config(float cluster_high_thresh, float cluster_low_thresh, float M_alpha, float M_beta, int M_min, int sketch_width, int sketch_depth, int sketch_S, bool prune_layer0_only, bool prune_angle_check, float prune_angle_deg, int k_candidates, int b_max_window, bool fast_construction, float cluster_radius_mult, float heuristic_alpha) {
        if (!appr_alg) throw std::runtime_error("Index not initiated");
        tahnsw::TAHNSWConfig cfg;
        cfg.cluster_high_thresh = cluster_high_thresh;
        cfg.cluster_low_thresh = cluster_low_thresh;
        cfg.M_alpha = M_alpha;
        cfg.M_beta = M_beta;
        cfg.M_min = M_min;
        cfg.sketch_width = sketch_width;
        cfg.sketch_depth = sketch_depth;
        cfg.sketch_S = sketch_S;
        cfg.prune_layer0_only = prune_layer0_only;
        cfg.prune_angle_check = prune_angle_check;
        cfg.prune_angle_deg = prune_angle_deg;
        cfg.k_candidates = k_candidates;
        cfg.b_max_window = b_max_window;
        cfg.fast_construction = fast_construction;
        cfg.cluster_radius_mult = cluster_radius_mult;
        cfg.heuristic_alpha = heuristic_alpha;
        appr_alg->set_tahnsw_config(cfg, l2space);
    }

    py::dict get_tahnsw_stats() {
        if (!appr_alg) return py::dict();
        return py::dict(
            "hub_count"_a = appr_alg->tahnsw_hub_count.load(),
            "leaf_count"_a = appr_alg->tahnsw_leaf_count.load(),
            "mid_count"_a = appr_alg->tahnsw_mid_count.load(),
            "edges_saved"_a = appr_alg->tahnsw_edges_saved.load(),
            "edges_baseline"_a = appr_alg->tahnsw_edges_baseline.load(),
            "search_calls"_a = appr_alg->tahnsw_search_calls.load(),
            "dist_comps"_a = appr_alg->metric_distance_computations.load(),
            "hops"_a = appr_alg->metric_hops.load()
        );
    }

    void reset_metrics() {
        if (!appr_alg) return;
        appr_alg->metric_distance_computations = 0;
        appr_alg->metric_hops = 0;
    }

    void build_search_seeds(int n_hub_seeds) {
        if (!appr_alg) throw std::runtime_error("Index not initiated");
        appr_alg->build_search_seeds(n_hub_seeds);
    }

    void clear_search_seeds() {
        if (appr_alg) appr_alg->tahnsw_search_seeds_.clear();
    }
    """

    content = content.replace("void set_ef(size_t ef) {", set_config_cpp + "\n    void set_ef(size_t ef) {")

    # Bind the method
    bind_code = """
        .def("set_ef", &Index<float>::set_ef, py::arg("ef"))
        .def("set_tahnsw_config", &Index<float>::set_tahnsw_config, 
             py::arg("cluster_high_thresh")=0.70f, 
             py::arg("cluster_low_thresh")=0.20f, 
             py::arg("M_alpha")=1.5f, 
             py::arg("M_beta")=0.5f, 
             py::arg("M_min")=2, 
             py::arg("sketch_width")=512, 
             py::arg("sketch_depth")=5, 
             py::arg("sketch_S")=30, 
             py::arg("prune_layer0_only")=true, 
             py::arg("prune_angle_check")=false, 
             py::arg("prune_angle_deg")=60.0f, 
             py::arg("k_candidates")=20,
             py::arg("b_max_window")=5000,
             py::arg("fast_construction")=true,
             py::arg("cluster_radius_mult")=1.0f,
             py::arg("heuristic_alpha")=1.0f)
        .def("get_tahnsw_stats", &Index<float>::get_tahnsw_stats)
        .def("reset_metrics", &Index<float>::reset_metrics)
        .def("build_search_seeds", &Index<float>::build_search_seeds, py::arg("n_hub_seeds")=6)
        .def("clear_search_seeds", &Index<float>::clear_search_seeds)
"""
    content = content.replace('.def("set_ef", &Index<float>::set_ef, py::arg("ef"))', bind_code)

    # Change module name from hnswlib to tahnsw_cpp
    content = content.replace('PYBIND11_PLUGIN(hnswlib) {', 'PYBIND11_PLUGIN(tahnsw_cpp) {')
    content = content.replace('m.def("knn_query", &Index<float>::knnQuery_return_numpy);', '') # remove global knn_query if any
    content = content.replace('py::module m("hnswlib");', 'py::module m("tahnsw_cpp");')
    content = content.replace('PYBIND11_MODULE(hnswlib, m) {', 'PYBIND11_MODULE(tahnsw_cpp, m) {')

    # Wrap classes in namespace tahnsw_cpp_bindings
    content = content.replace('using namespace pybind11::literals;', 'using namespace pybind11::literals;\n\nnamespace tahnsw_cpp_bindings {\n')
    content = content.replace('PYBIND11_PLUGIN(tahnsw_cpp) {', '}\n\nPYBIND11_PLUGIN(tahnsw_cpp) {')
    content = content.replace('PYBIND11_MODULE(tahnsw_cpp, m) {', '}\n\nPYBIND11_MODULE(tahnsw_cpp, m) {')
    content = re.sub(r'(?<!BF)Index<float>', 'tahnsw_cpp_bindings::Index<float>', content)
    content = content.replace('BFIndex<float>', 'tahnsw_cpp_bindings::BFIndex<float>')
    content = content.replace('py::class_<tahnsw_cpp_bindings::Index<float>>(m, "Index")', 'py::class_<tahnsw_cpp_bindings::Index<float>>(m, "IndexCpp")')
    content = content.replace('py::class_<tahnsw_cpp_bindings::BFIndex<float>>(m, "BFIndex")', 'py::class_<tahnsw_cpp_bindings::BFIndex<float>>(m, "BFIndexCpp")')

    with open("src/hnswlib/python_bindings/bindings.cpp", "w") as f:
        f.write(content)

if __name__ == "__main__":
    patch()
