import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from .vector_store import BaseVectorStore

class FaissVectorStore(BaseVectorStore):
    """
    FAISS implementation of the VectorStore substrate.
    Provides simple, local, in-memory similarity search.
    Note: Metadata filtering is NOT natively supported and must be handled in post-processing.
    """
    
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.node_ids = []
        self.embeddings = []
        self.metadata_map = {}

    def add_embedding(self, node_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None):
        """Adds an embedding and its metadata."""
        # Convert to numpy
        emb_np = np.array(embedding).astype("float32")
        
        # FAISS doesn't support easy updates, so we check if it exists
        if node_id in self.node_ids:
            idx = self.node_ids.index(node_id)
            self.embeddings[idx] = emb_np
            if metadata:
                self.metadata_map[node_id] = metadata
            # Rebuild index (FAISS FlatL2 is fast for small sets)
            self._rebuild_index()
        else:
            self.node_ids.append(node_id)
            self.embeddings.append(emb_np)
            if metadata:
                self.metadata_map[node_id] = metadata
            self.index.add(np.array([emb_np]))

    def update_metadata(self, node_id: str, metadata: Dict[str, Any]):
        """Updates local metadata map."""
        if node_id in self.metadata_map:
            self.metadata_map[node_id].update(metadata)
        else:
            self.metadata_map[node_id] = metadata

    def search(
        self, 
        query_embedding: List[float], 
        limit: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs search. Filters are applied in POST-PROCESSING for FAISS.
        """
        if not self.node_ids:
            return []
            
        q_emb_np = np.array([query_embedding]).astype("float32")
        limit = min(limit, len(self.node_ids))
        
        distances, indices = self.index.search(q_emb_np, limit)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.node_ids): continue
            
            node_id = self.node_ids[idx]
            payload = self.metadata_map.get(node_id, {})
            
            # Manual Filter Application
            if filters:
                match = True
                for k, v in filters.items():
                    if payload.get(k) != v:
                        match = False
                        break
                if not match: continue

            results.append({
                "node_id": node_id,
                "score": 1 / (1 + float(dist)), # Convert L2 to similarity
                "payload": payload
            })
            
        return results

    def delete(self, node_id: str):
        if node_id in self.node_ids:
            idx = self.node_ids.index(node_id)
            self.node_ids.pop(idx)
            self.embeddings.pop(idx)
            self.metadata_map.pop(node_id, None)
            self._rebuild_index()

    def clear(self):
        self.node_ids = []
        self.embeddings = []
        self.metadata_map = {}
        self.index.reset()

    def _rebuild_index(self):
        if not self.embeddings:
            self.index.reset()
            return
        embeddings_array = np.array(self.embeddings).astype("float32")
        self.index.reset()
        self.index.add(embeddings_array)
