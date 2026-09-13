from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseVectorStore(ABC):
    """
    Abstract base class for vector storage substrates.
    Enables backend-agnostic semantic retrieval.
    """
    
    @abstractmethod
    def add_embedding(self, node_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None):
        """Adds or updates an embedding in the store."""
        pass

    @abstractmethod
    def update_metadata(self, node_id: str, metadata: Dict[str, Any]):
        """Updates metadata for an existing vector without re-indexing."""
        pass

    @abstractmethod
    def search(
        self, 
        query_embedding: List[float], 
        limit: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs a semantic search, optionally filtered by metadata.
        Returns a list of results with scores and payloads.
        """
        pass

    @abstractmethod
    def delete(self, node_id: str):
        """Removes a vector from the store."""
        pass

    @abstractmethod
    def clear(self):
        """Clears all vectors from the store."""
        pass
