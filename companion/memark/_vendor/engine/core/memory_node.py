from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from engine.core.types import NodeType

@dataclass
class MemoryNode:
    id: str
    node_type: NodeType # episodic, summary, concept, goal, plan, action, predicted_outcome
    summary: str
    embedding: Optional[List[float]] = None
    timestamp_created: datetime = field(default_factory=datetime.utcnow)
    timestamp_updated: datetime = field(default_factory=datetime.utcnow)
    importance_score: float = 0.5
    salience_score: float = 0.5
    confidence_score: float = 1.0
    retrieval_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    parent_id: Optional[str] = None # For goals/plans, this is the Master Goal ID / Parent Plan ID
    child_ids: List[str] = field(default_factory=list) # For goals/plans, these are Subgoals / Subplans
    raw_conversation_refs: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict) 
    # Metadata for goals: status, priority, goal_type
    # Metadata for plans/actions:
    #   status: "proposed", "active", "blocked", "dormant", "completed", "failed", "abandoned"
    #   energy_cost: float
    #   recursion_depth: int

