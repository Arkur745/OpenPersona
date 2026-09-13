from enum import Enum

class StrEnum(str, Enum):
    """
    Subclass of str and Enum to allow direct string comparisons
    and full JSON serialization compatibility.
    """
    def __str__(self) -> str:
        return self.value

class NodeType(StrEnum):
    CONCEPT = "concept"
    SUMMARY = "summary"
    EPISODIC = "episodic"
    GOAL = "goal"
    PLAN = "plan"
    ACTION = "action"
    PREDICTED_OUTCOME = "predicted_outcome"
    SNAPSHOT = "snapshot"
    IDENTITY = "identity"

class PlanState(StrEnum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    BLOCKED = "blocked"
    DORMANT = "dormant"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"

class RelationType(StrEnum):
    SEMANTIC_SIMILARITY = "semantic_similarity"
    HIERARCHICAL = "hierarchical"
    HIERARCHICAL_GOAL = "hierarchical_goal"
    GENERATES = "GENERATES"
    CONTAINS = "CONTAINS"
    ENABLES = "ENABLES"
    PREDICTS = "PREDICTS"
    RESULTS_IN = "RESULTS_IN"
    MOTIVATES = "motivates"
    BLOCKS = "BLOCKS"
    REQUIRES = "REQUIRES"

