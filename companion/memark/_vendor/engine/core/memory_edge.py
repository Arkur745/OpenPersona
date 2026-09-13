from dataclasses import dataclass, field
from datetime import datetime
from engine.core.types import RelationType

@dataclass
class MemoryEdge:
    id: str
    source_node: str
    target_node: str
    relation_type: RelationType
    weight: float = 1.0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    retrieval_count: int = 0
    decay_rate: float = 0.01
