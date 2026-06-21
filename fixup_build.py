"""
fixup_build.py — fix remaining compile errors in the patched C++ files.
Safe to run multiple times (idempotent replacements).
"""

HNSW_H   = "src/hnswlib/hnswlib/hnswalg.h"
BIND_CPP = "src/hnswlib/python_bindings/bindings.cpp"

# ── Fix hnswalg.h ─────────────────────────────────────────────────────────────
with open(HNSW_H, "r") as f:
    h = f.read()

# repairConnectionsForUpdate still uses the old function name
OLD = "mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);"
NEW = "mutuallyConnectNewElement_TAHNSW(M_, dataPoint, dataPointInternalId, filteredTopCandidates, level, true);"
if OLD in h:
    h = h.replace(OLD, NEW)
    print("hnswalg.h: patched repairConnectionsForUpdate call")
else:
    print("hnswalg.h: repairConnectionsForUpdate already patched")

with open(HNSW_H, "w") as f:
    f.write(h)

# ── Fix bindings.cpp ──────────────────────────────────────────────────────────
with open(BIND_CPP, "r") as f:
    b = f.read()

# 1. build_search_seeds — stub out body (appr_alg->build_search_seeds doesn't exist)
b = b.replace(
    'appr_alg->build_search_seeds(n_hub_seeds);',
    '(void)n_hub_seeds; /* no-op: not implemented */',
)

# 2. clear_search_seeds — stub out body (tahnsw_search_seeds_ doesn't exist)
b = b.replace(
    'if (appr_alg) appr_alg->tahnsw_search_seeds_.clear();',
    '/* no-op: not implemented */',
)

# 3. get_tahnsw_stats — remove tahnsw_search_calls (member doesn't exist)
b = b.replace(
    '"search_calls"_a = appr_alg->tahnsw_search_calls.load(),',
    '"search_calls"_a = (long)0,  /* not implemented */',
)

with open(BIND_CPP, "w") as f:
    f.write(b)
print("bindings.cpp: all stubs applied")
