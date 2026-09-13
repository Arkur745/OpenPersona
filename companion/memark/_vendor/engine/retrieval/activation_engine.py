import time
from typing import List, Dict, Any, Set, Optional

class ActivationEngine:
    """
    Orchestrates spreading activation across the memory graph.
    Prevents explosion via decay and thresholding.
    Supports concept-bounded propagation to keep context relevant.
    """

    def __init__(
        self,
        memory_repository,
        base_activation: Optional[float] = None,
        decay_factor: Optional[float] = None,
        activation_threshold: Optional[float] = None,
        max_depth: Optional[int] = None
    ):
        import engine.config.default_config as _cfg
        self.memory_repository = memory_repository
        self.base_activation = base_activation if base_activation is not None else _cfg.BASE_ACTIVATION
        self.decay_factor = decay_factor if decay_factor is not None else _cfg.ACTIVATION_DECAY_FACTOR
        self.activation_threshold = activation_threshold if activation_threshold is not None else _cfg.ACTIVATION_THRESHOLD
        self.max_depth = max_depth if max_depth is not None else _cfg.MAX_SPREAD_DEPTH

    # ============================================================
    # Propagate Activation
    # ============================================================

    def propagate_activation(
        self,
        start_node,
        allowed_concept_ids: Optional[List[str]] = None,
        trace: Optional[Any] = None
    ):
        """
        Initiates spreading activation from a seed node.
        Returns a dictionary of node_id -> activation_score.
        """
        
        import logging
        import engine.config.default_config as _cfg
        
        spreading_on = getattr(_cfg, "ENABLE_SPREADING_ACTIVATION", True)
        dormancy_on = getattr(_cfg, "ENABLE_DORMANCY", True)
        hierarchy_on = not getattr(_cfg, "DISABLE_HIERARCHY_PROPAGATION", False)
        firewall_on = allowed_concept_ids is not None
        salience_on = getattr(_cfg, "ENABLE_SALIENCE", True)
        
        logging.info(
            f"[EVAL RUNTIME CONFIG] Spreading: {'ON' if spreading_on else 'OFF'} | "
            f"Dormancy: {'ON' if dormancy_on else 'OFF'} | "
            f"Hierarchy: {'ON' if hierarchy_on else 'OFF'} | "
            f"Firewall: {'ON' if firewall_on else 'OFF'} | "
            f"Salience: {'ON' if salience_on else 'OFF'}"
        )

        salience_weight = start_node.salience_score if salience_on else 1.0
        initial_activation = (
            salience_weight 
            * self.base_activation
        )

        activated_nodes = {
            start_node.id: initial_activation
        }

        activated_edges = {}
        visited = set()
        
        if spreading_on:
            self._recursive_propagation(
                current_node=start_node,
                current_activation=initial_activation,
                activated_nodes=activated_nodes,
                activated_edges=activated_edges,
                visited=visited,
                depth=0,
                max_depth=self.max_depth,
                allowed_concept_ids=allowed_concept_ids,
                trace=trace
            )

        return {
            "nodes": activated_nodes,
            "edges": activated_edges
        }

    # ============================================================
    # Recursive Spreading
    # ============================================================

    def _recursive_propagation(
        self,
        current_node,
        current_activation,
        activated_nodes,
        activated_edges,
        visited,
        depth,
        max_depth,
        allowed_concept_ids=None,
        trace=None
    ):
        """
        Core recursive engine for spreading activation.
        """
        
        if depth >= max_depth:
            return

        if current_activation < self.activation_threshold:
            return

        visited.add(current_node.id)

        # ============================================================
        # Bounded Propagation Check
        # ============================================================
        # If we are propagating from an episodic/summary node, 
        # we check if it belongs to an 'allowed' concept domain.
        # This prevents 'topic bleeding' during high-salience recall.
        
        if (
            allowed_concept_ids is not None
            and current_node.node_type in ["summary", "episodic"]
        ):
            # Find the root concept for this node
            root_concept_id = (
                self._get_root_concept_id(
                    current_node
                )
            )
            
            if (
                root_concept_id 
                and root_concept_id not in allowed_concept_ids
            ):
                # Suppress activation for out-of-domain nodes
                if trace:
                    trace.add_suppression_event(
                        target_id=current_node.id,
                        target_root_concept=root_concept_id,
                        suppression_reason="Out of concept domain"
                    )
                return

        # ============================================================
        # Collect Neighbor Activations
        # ============================================================

        candidate_neighbors = []

        for neighbor_id in (
            self.memory_repository.get_neighbors(
                current_node.id
            )
        ):

            edge_data = (
                self.memory_repository.get_edge_data(
                    current_node.id,
                    neighbor_id
                )
            )

            if not edge_data:
                continue

            import engine.config.default_config as _cfg
            rel_type = edge_data.get("relation_type")
            
            # Check semantic mesh ablation
            if not getattr(_cfg, "ENABLE_SEMANTIC_MESH_PROPAGATION", True) and rel_type == "semantic_similarity":
                continue
                
            # Check hierarchy propagation bypass
            if getattr(_cfg, "DISABLE_HIERARCHY_PROPAGATION", False) and rel_type != "semantic_similarity":
                continue

            edge_weight = edge_data.get(
                "weight",
                1.0
            )

            edge_confidence = edge_data.get(
                "confidence",
                1.0
            )

            if edge_weight < 0.1:
                continue

            # Compute new activation strength
            next_activation = (
                current_activation 
                * edge_weight 
                * edge_confidence 
                * self.decay_factor
            )

            if next_activation < self.activation_threshold:
                continue

            neighbor_node = (
                self.memory_repository.get_node(
                    neighbor_id
                )
            )

            if not neighbor_node:
                continue

            # Check Dormancy Gating
            if getattr(_cfg, "ENABLE_DORMANCY", True):
                is_dormant = (
                    getattr(neighbor_node, "metadata", {}).get("is_dormant", False) 
                    or (getattr(neighbor_node, "salience_score", 1.0) < getattr(_cfg, "DORMANCY_THRESHOLD", 0.1))
                )
                if is_dormant and next_activation < 0.8:
                    continue

            candidate_neighbors.append(
                (neighbor_node, next_activation)
            )

        # ============================================================
        # Sort and Limit Propagation (Competetive Inhibition)
        # ============================================================
        # We only propagate to the top most salient neighbors
        # to prevent graph saturation.
        #
        # Sort key includes neighbor_node.id as a deterministic tie-breaker: when two
        # or more candidates share the exact same activation score (common whenever a
        # graph builder assigns fixed, repeated edge weights, e.g. WikiMultihopGraphBuilder's
        # constant 0.9/0.85/0.6), relying on activation score alone left the top-5 cutoff
        # order-dependent on dict/set iteration order, which was not reproducible across
        # process runs. See docs/eval_audit.md Phase 2.5 S1.
        candidate_neighbors.sort(
            key=lambda x: (-x[1], x[0].id)
        )

        top_neighbors = candidate_neighbors[:5]

        for (
            neighbor_node,
            next_activation
        ) in top_neighbors:

            # Update activated nodes map (keep highest activation)
            if neighbor_node.id not in activated_nodes or next_activation > activated_nodes[neighbor_node.id]:
                activated_nodes[neighbor_node.id] = next_activation

            # Track activated edges for reinforcement
            edge_key = (current_node.id, neighbor_node.id)
            if edge_key not in activated_edges or next_activation > activated_edges[edge_key]:
                activated_edges[edge_key] = next_activation

            # Recurse
            if neighbor_node.id not in visited:
                
                if trace:
                    target_root = self._get_root_concept_id(neighbor_node)
                    trace.add_propagation_event(
                        source_id=current_node.id,
                        target_id=neighbor_node.id,
                        target_root_concept=target_root or "None",
                        strength=next_activation,
                        depth=depth + 1
                    )
                
                self._recursive_propagation(
                    current_node=neighbor_node,
                    current_activation=next_activation,
                    activated_nodes=activated_nodes,
                    activated_edges=activated_edges,
                    visited=visited,
                    depth=depth + 1,
                    max_depth=max_depth,
                    allowed_concept_ids=allowed_concept_ids,
                    trace=trace
                )

    # ============================================================
    # Helper: Get Root Concept ID
    # ============================================================

    def _get_root_concept_id(self, node):
        """
        Traverses upward to find the top-level concept node.
        """
        
        curr = node
        visited_up = set()
        
        while curr and curr.node_type != "concept":
            
            if curr.id in visited_up:
                break
            visited_up.add(curr.id)
            
            # Use parent_id for hierarchy traversal
            parent_id = getattr(curr, "parent_id", None)
            if not parent_id:
                break
                
            curr = self.memory_repository.get_node(parent_id)
            
        if curr and curr.node_type == "concept":
            return curr.id
            
        return None
