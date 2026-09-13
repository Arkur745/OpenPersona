class MemoryRepository:

    def __init__(
        self,
        memory_manager
    ):

        self.memory_manager = (
            memory_manager
        )

    # ============================================================
    # Node Operations
    # ============================================================

    def create_node(
        self,
        node_type,
        summary
    ):

        return (
            self.memory_manager.create_node(
                node_type=node_type,
                summary=summary
            )
        )

    def reinforce_node(self, node_id: str, reinforcement_factor: float = 0.1):
        self.memory_manager.reinforce_node(node_id, reinforcement_factor)

    def get_node(
        self,
        node_id
    ):

        return (
            self.memory_manager.get_node(
                node_id
            )
        )

    def get_all_nodes(self):

        return (
            self.memory_manager.nodes.values()
        )

    def get_nodes_by_type(self, node_type: str):
        return [
            node_id
            for node_id, node in self.memory_manager.nodes.items()
            if node.node_type == node_type
        ]

    # ============================================================
    # Edge Operations
    # ============================================================

    def create_semantic_edges_batch(self, edge_list: list):
        if hasattr(self.memory_manager.neo4j_repository, "create_semantic_edges_batch"):
            self.memory_manager.neo4j_repository.create_semantic_edges_batch(edge_list)
        
        # Also update runtime graph
        for edge in edge_list:
            self.create_edge(
                source_node=edge["source_node"],
                target_node=edge["target_node"],
                relation_type="semantic_similarity",
                weight=edge["weight"],
                confidence=edge["confidence"]
            )

    def create_edge(
        self,
        source_node,
        target_node,
        relation_type,
        weight=1.0,
        confidence=1.0
    ):

        return (
            self.memory_manager.create_edge(
                source_node=source_node,
                target_node=target_node,
                relation_type=relation_type,
                weight=weight,
                confidence=confidence
            )
        )

    def get_neighbors(
        self,
        node_id
    ):
        return self.memory_manager.get_neighbors(node_id)

    def get_edge_data(
        self,
        source_node,
        target_node
    ):

        graph = self.memory_manager.graph

        return graph.get_edge_data(
            source_node,
            target_node
        )

    def get_edge_between(
        self,
        source_node,
        target_node
    ):
        return self.memory_manager.get_edge_between(
            source_node,
            target_node
        )

    # ============================================================
    # Graph Operations
    # ============================================================

    def has_edge(
        self,
        source_node,
        target_node
    ):

        return (
            self.memory_manager.graph.has_edge(
                source_node,
                target_node
            )
        )

    def remove_edge(
        self,
        source_node,
        target_node
    ):

        self.memory_manager.graph.remove_edge(
            source_node,
            target_node
        )

    def remove_node(
        self,
        node_id
    ):

        self.memory_manager.graph.remove_node(
            node_id
        )

    def sync_node(self, node_id: str):
        self.memory_manager.sync_node(node_id)

    def sync_edge(self, edge_id: str):
        self.memory_manager.sync_edge(edge_id)

    # ============================================================
    # Semantic Connectivity Logic (Centralized)
    # ============================================================

    def find_related_nodes(
        self,
        embedding,
        top_k=5,
        include_node_types=None
    ):
        if not self.memory_manager.vector_search:
            return []

        filters = None
        if include_node_types:
            # For now, handle single type or list
            if isinstance(include_node_types, str):
                filters = {"node_type": include_node_types}
            elif isinstance(include_node_types, list) and len(include_node_types) == 1:
                filters = {"node_type": include_node_types[0]}
            # Complex logic for multiple types would go here if needed
            # For our current needs (concepts linking to concepts), this is enough.

        results = self.memory_manager.vector_search.search(
            query_embedding=embedding,
            top_k=top_k,
            filters=filters
        )

        return results

    def create_semantic_edges(
        self,
        new_node,
        related_results,
        similarity_threshold=None
    ):
        import engine.config.default_config as _cfg
        if similarity_threshold is None:
            similarity_threshold = _cfg.CONCEPT_LINK_THRESHOLD
        for result in related_results:
            similarity = result["similarity"] if "similarity" in result else result.get("score", 0)
            
            if similarity < similarity_threshold:
                continue

            target_id = result["node_id"]
            if target_id == new_node.id:
                continue

            target_node = self.get_node(target_id)
            if not target_node or target_node.node_type != new_node.node_type:
                continue
            from engine.logging import get_logger
            logger = get_logger("storage")
            logger.info(f"Linking {new_node.node_type} {new_node.id[:8]} <-> {target_id[:8]} (Weight: {similarity:.4f})")
            
            # Bidirectional Edge
            self.create_edge(
                source_node=new_node.id,
                target_node=target_id,
                relation_type="semantic_similarity",
                weight=similarity,
                confidence=1.0
            )
            self.create_edge(
                source_node=target_id,
                target_node=new_node.id,
                relation_type="semantic_similarity",
                weight=similarity,
                confidence=1.0
            )

    def get_all_relationships(self) -> list:
        """Returns all graph edges as standardized dictionaries matching relationship queries."""
        return [
            {
                "id": edge.id,
                "source": edge.source_node,
                "target": edge.target_node,
                "type": edge.relation_type,
                "weight": edge.weight,
                "confidence": edge.confidence
            }
            for edge in self.memory_manager.edges.values()
        ]

