"""
memark/engine/config/default_config.py
======================================
Single source of truth for all cognitive architecture parameters.
Loads from configs/default_config.json if available, with in-code fallbacks.
"""
import os
import json

# ============================================================
# EMBEDDING & VECTOR SEARCH
# ============================================================
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
CONCEPT_SEARCH_K: int = 50

# ============================================================
# SEMANTIC THRESHOLDS
# ============================================================
CONCEPT_MERGE_THRESHOLD: float = 0.45
SUMMARY_THRESHOLD: float = 0.45
CONCEPT_LINK_THRESHOLD: float = 0.15
EPISODIC_DUPLICATE_THRESHOLD: float = 0.92

# ============================================================
# SALIENCE ECOLOGY & DECAY
# ============================================================
DORMANCY_THRESHOLD: float = 0.1
DECAY_RATES: dict = {
    "episodic":         0.05,
    "summary":          0.02,
    "concept":          0.005,
    "plan":             0.01,
    "action":           0.02,
    "predicted_outcome": 0.02,
    "goal":             0.003,
    "edge":             0.03
}
IDENTITY_DECAY_MULTIPLIER: float = 0.1

# ============================================================
# SPREADING ACTIVATION
# ============================================================
BASE_ACTIVATION: float = 0.8
ACTIVATION_DECAY_FACTOR: float = 0.7
ACTIVATION_THRESHOLD: float = 0.1
MAX_SPREAD_DEPTH: int = 3
MAX_CANDIDATE_NEIGHBORS: int = 5
MIN_EDGE_WEIGHT_FOR_PROPAGATION: float = 0.1

# ============================================================
# REINFORCEMENT
# ============================================================
REINFORCEMENT_FACTOR: float = 0.1
PROCEDURAL_COST_DISCOUNT: float = 0.9
MIN_PLAN_ENERGY_COST: float = 5.0

# ============================================================
# DELIBERATIVE PLANNING
# ============================================================
MAX_PLAN_DEPTH: int = 3
MAX_SUBPLAN_BRANCHING: int = 3
MAX_CONCURRENT_PLANS: int = 3
BASE_ENERGY_BUDGET: float = 100.0
PLAN_ACTIVATION_COST: float = 20.0
IDENTITY_OVERFLOW_THRESHOLD: float = 0.85
PLAN_RELIABILITY_WEIGHT: float = 0.3
PLAN_GOAL_ALIGNMENT_WEIGHT: float = 0.7
PLAN_REINFORCEMENT_MIN_RELIABILITY: float = 0.7
PLAN_RELIABILITY_ALPHA: float = 0.3

# ============================================================
# GRAPH MAINTENANCE
# ============================================================
EDGE_PRUNE_THRESHOLD: float = 0.05

# ============================================================
# ABLATION CONTROL FLAGS (For Scientific Evaluation)
# ============================================================
ENABLE_SALIENCE_DECAY: bool = True
ENABLE_SPREADING_ACTIVATION: bool = True
ENABLE_DORMANCY: bool = True
ENABLE_PLANNING: bool = True
ENABLE_IDENTITY_REINFORCEMENT: bool = True
ENABLE_GOAL_BIASING: bool = True
ENABLE_SEMANTIC_MESH_PROPAGATION: bool = True
DISABLE_HIERARCHY_PROPAGATION: bool = False
ENABLE_SALIENCE: bool = True

def _load_config():
    try:
        curr = os.path.abspath(os.path.dirname(__file__))
        root = os.path.abspath(os.path.join(curr, "../../../"))
        config_path = os.path.join(root, "configs", "default_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

# Load and override
_data = _load_config()
for _k, _v in _data.items():
    globals()[_k] = _v
