import uuid
import random
import networkx as nx

from typing import Dict, List, Optional
from datetime import datetime

from engine.core.memory_node import MemoryNode
from engine.core.memory_edge import MemoryEdge

from dataclasses import asdict

class GraphMemoryManager:

    def __init__(self,
                 neo4j_repository=None,
                 vector_search=None,
                 id_seed: Optional[int] = None
                 ):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: Dict[str, MemoryEdge] = {}
        self.neo4j_repository = (
            neo4j_repository
        )
        self.vector_search = vector_search

        # Node/edge IDs are normally random uuid4()s (fine for production, where IDs
        # only need to be unique). Benchmark/reproducibility harnesses can pass
        # id_seed to make ID generation deterministic instead: IDs still look like
        # random UUIDs, but two runs constructed with the same id_seed and the same
        # sequence of create_node/create_edge calls produce byte-identical IDs. This
        # matters because ActivationEngine's top-K neighbor cutoff uses node.id as a
        # deterministic tie-breaker (see activation_engine.py) -- that tie-break is
        # only reproducible across runs if the IDs it breaks ties on are themselves
        # reproducible. See docs/eval_audit.md Phase 2.5 S1.
        self._id_rng = random.Random(id_seed) if id_seed is not None else None

    def _new_id(self) -> str:
        if self._id_rng is not None:
            return str(uuid.UUID(int=self._id_rng.getrandbits(128), version=4))
        return str(uuid.uuid4())

    # --------------------------------------------------------
    # Node Management
    # --------------------------------------------------------

    def create_node(
        self,
        node_type: str,
        summary: str,
        importance_score: float = 0.5,
        salience_score: float = 0.5,
        confidence_score: float = 1.0,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:

        node_id = self._new_id()

        node = MemoryNode(
            id=node_id,
            node_type=node_type,
            summary=summary,
            importance_score=importance_score,
            salience_score=salience_score,
            confidence_score=confidence_score,
            parent_id=parent_id,
            metadata=metadata or {}
        )

        self.nodes[node_id] = node
        from engine.logging import get_logger
        logger = get_logger("graph_manager")
        logger.info(f"Created {node_type} node: {node_id[:8]} (Total: {len(self.nodes)})")


        self.graph.add_node(
            node_id,
            **asdict(node)
        )

        # Attach child to parent if parent exists
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].child_ids.append(node_id)
        # ========================================================
        # Persist to Neo4j & Vector Store
        # ========================================================


        if self.neo4j_repository:
            self.neo4j_repository.create_node(asdict(node))
            
        if self.vector_search:
            # Note: Embedding must be added separately via insertion engine
            # but we ensure metadata is initialized
            self.vector_search.update_metadata(node_id, {
                "node_type": node_type,
                "salience": salience_score,
                "is_dormant": node.metadata.get("is_dormant", False)
            })

        return node_id

    def sync_node(self, node_id: str):
        """Synchronizes an in-memory node with Neo4j and Vector Store."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            # Sync to Neo4j
            if self.neo4j_repository:
                self.neo4j_repository.update_node(asdict(node))
                
            # Sync to Vector Substrate (Qdrant/FAISS)
            if self.vector_search:
                self.vector_search.update_metadata(node_id, {
                    "salience": node.salience_score,
                    "is_dormant": node.metadata.get("is_dormant", False),
                    "is_identity_core": node.metadata.get("is_identity_core", False),
                    "is_goal": node.node_type == "goal"
                })


    # --------------------------------------------------------
    # Edge Management
    # --------------------------------------------------------

    def create_edge(
        self,
        source_node: str,
        target_node: str,
        relation_type: str,
        weight: float = 1.0,
        confidence: float = 1.0,
        decay_rate: float = 0.01
    ) -> str:

        if source_node not in self.nodes:
            raise ValueError(f"Source node {source_node} not found")

        if target_node not in self.nodes:
            raise ValueError(f"Target node {target_node} not found")

        edge_id = self._new_id()

        edge = MemoryEdge(
            id=edge_id,
            source_node=source_node,
            target_node=target_node,
            relation_type=relation_type,
            weight=weight,
            confidence=confidence,
            decay_rate=decay_rate
        )

        self.edges[edge_id] = edge

        self.graph.add_edge(
            source_node,
            target_node,
            **asdict(edge)
        )

        # ========================================================
        # Persist to Neo4j
        # ========================================================


        if self.neo4j_repository:
            self.neo4j_repository.create_edge(asdict(edge))
            
        return edge_id

    def sync_edge(self, edge_id: str):
        """Synchronizes an in-memory edge with Neo4j."""
        if self.neo4j_repository and edge_id in self.edges:
            edge = self.edges[edge_id]
            self.neo4j_repository.update_edge(asdict(edge))

    # --------------------------------------------------------
    # Retrieval Helpers
    # --------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> List[MemoryNode]:
        node = self.get_node(node_id)

        if not node:
            return []

        return [self.nodes[child_id] for child_id in node.child_ids]

    def get_connected_nodes(self, node_id: str) -> List[str]:
        if node_id not in self.graph:
            return []

        successors = list(self.graph.successors(node_id))
        predecessors = list(self.graph.predecessors(node_id))

        neighbors = successors + predecessors
        return neighbors

    # --------------------------------------------------------
    # Memory Dynamics
    # --------------------------------------------------------

    def reinforce_node(self, node_id: str, reinforcement_factor: float = 0.1):
        """Boosts a node's salience and importance."""
        node = self.nodes.get(node_id)
        if not node:
            return

        # Boost salience gradually (diminishing returns toward 1.0)
        node.salience_score = node.salience_score + reinforcement_factor * (1 - node.salience_score)
        node.retrieval_count += 1
        node.last_accessed = datetime.utcnow()
        
        self.sync_node(node_id)

    def reinforce_edge(self, edge_id: str, reinforcement_factor: float = 0.1):

        edge = self.edges.get(edge_id)

        if not edge:
            return

        edge.weight = edge.weight + reinforcement_factor * (1 - edge.weight)
        edge.retrieval_count += 1
        edge.last_accessed = datetime.utcnow()
        self.sync_edge(edge_id)

    def decay_edge(self, edge_id: str):

        edge = self.edges.get(edge_id)

        if not edge:
            return

        edge.weight *= (1 - edge.decay_rate)
        self.sync_edge(edge_id)

    def prune_weak_edges(self, threshold: float = 0.05):

        edges_to_remove = []

        for edge_id, edge in self.edges.items():
            if edge.weight < threshold:
                edges_to_remove.append(edge_id)

        for edge_id in edges_to_remove:
            edge = self.edges[edge_id]

            self.graph.remove_edge(edge.source_node, edge.target_node)
            del self.edges[edge_id]

    def prune_pendant_nodes(self):

        nodes_to_remove = []

        for node_id in self.graph.nodes:
            degree = self.graph.degree(node_id)

            if degree == 0:
                nodes_to_remove.append(node_id)

        for node_id in nodes_to_remove:
            self.graph.remove_node(node_id)
            del self.nodes[node_id]

    # --------------------------------------------------------
    # Graph Inspection
    # --------------------------------------------------------

    def print_graph_summary(self):

        from engine.logging import get_logger
        logger = get_logger("graph_manager")
        logger.info("===== MEMORY GRAPH SUMMARY =====")
        logger.info(f"Total Nodes: {len(self.nodes)}")
        logger.info(f"Total Edges: {len(self.edges)}")

        logger.info("Nodes:")
        for node_id, node in self.nodes.items():
            logger.info(f"- {node.node_type}: {node.summary[:50]}")

        logger.info("Edges:")
        for edge_id, edge in self.edges.items():
            logger.info(
                f"- {edge.source_node[:6]} -> {edge.target_node[:6]} "
                f"({edge.relation_type}, weight={edge.weight:.3f})"
            )

    def get_edges_for_node(self, node_id):

        connected_edges = []

        for edge in self.edges.values():

            if (
                edge.source_node == node_id
                or edge.target_node == node_id
            ):

                connected_edges.append(edge)

        return connected_edges
    # --------------------------------------------------------
    # Repository Compatibility Helpers
    # --------------------------------------------------------
    
    
    def get_neighbors(
        self,
        node_id
    ):
    
        if node_id not in self.graph:
            return []
    
        # Return both predecessors and successors for a DiGraph to act as 'neighbors'.
        # Deduplicate via dict.fromkeys (insertion-order-preserving) rather than set():
        # set() ordering depends on Python's per-process string hash randomization
        # (PYTHONHASHSEED), which made ActivationEngine's top-K neighbor cutoff
        # non-reproducible across process runs whenever multiple candidates tied on
        # activation score (see docs/eval_audit.md Phase 2.5 S1).
        successors = list(self.graph.successors(node_id))
        predecessors = list(self.graph.predecessors(node_id))
        return list(dict.fromkeys(successors + predecessors))



    
    
    def get_edge_data(
        self,
        source_node,
        target_node
    ):
    
        return self.graph.get_edge_data(
            source_node,
            target_node
        )
    
    
    def has_edge(
        self,
        source_node,
        target_node
    ):
    
        return self.graph.has_edge(
            source_node,
            target_node
        )
    def get_edge_between(
        self,
        source_node,
        target_node
    ):
    
        for edge in self.edges.values():
        
            if (
                edge.source_node == source_node
                and edge.target_node == target_node
            ):
    
                return edge
    
        return None

# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":

    memory_engine = GraphMemoryManager()

    # Create root node
    root_id = memory_engine.create_node(
        node_type="concept",
        summary="Research on graph-based memory systems"
    )

    # Create child node
    child_id = memory_engine.create_node(
        node_type="summary",
        summary="Discussion about hierarchical semantic memory",
        parent_id=root_id
    )

    # Create connected node
    related_id = memory_engine.create_node(
        node_type="concept",
        summary="Adaptive forgetting and memory pruning"
    )

    # Create relationships
    memory_engine.create_edge(
        source_node=root_id,
        target_node=related_id,
        relation_type="related_to",
        weight=0.8
    )

    # Inspect graph
    memory_engine.print_graph_summary()

