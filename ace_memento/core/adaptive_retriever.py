"""
Adaptive Retriever - CHỈ DÙNG COSINE SIMILARITY (giống CaseBank cũ)
Không có BM25, không có hybrid search.
"""

import numpy as np
from typing import List, Dict, Any, Optional


class AdaptiveRetriever:
    """
    Adaptive Retriever chỉ dùng Cosine Similarity.
    
    GIỐNG CASEBANK CŨ:
    - Embedding model: BAAI/bge-m3
    - Cosine similarity: np.dot(embeddings, query_emb)
    - Normalize embeddings: normalize_embeddings=True
    
    KHÁC CASEBANK CŨ:
    - Hỗ trợ metadata filtering (task_type, reward, error_identification)
    - Round-based retrieval (diverse selection, error-aware selection)
    """
    
    def __init__(
        self,
        embedding_model,
        embedding_vectors: np.ndarray,
        corpus_texts: List[str],
        metadata_list: List[Dict[str, Any]],
        top_k: int = 4,
    ):
        """
        Args:
            embedding_model: SentenceTransformer model
            embedding_vectors: numpy array of embeddings (N, dim)
            corpus_texts: List of text for each entry
            metadata_list: List of metadata for each entry
            top_k: Number of results to return
        """
        self.embedding_model = embedding_model
        self.embedding_vectors = embedding_vectors
        self.corpus_texts = corpus_texts
        self.metadata_list = metadata_list
        self.top_k = top_k
        
        # BM25 disabled - kept for compatibility
        self.bm25_index = None
    
    def set_bm25_index(self, bm25_index):
        """Set BM25 index (disabled - kept for compatibility)"""
        self.bm25_index = bm25_index
    
    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        round_idx: int = 0,
        previous_error: Optional[str] = None,
        k: Optional[int] = None,
        intent: str = "general",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve Top-K cases using Cosine Similarity.
        
        Args:
            query: User query
            query_embedding: Pre-computed embedding (optional)
            round_idx: Current reflection round (0 = initial)
            previous_error: Error from previous round
            k: Number of results
            intent: Search intent (exact, recall, general, explore) - kept for compatibility
        
        Returns:
            List of retrieved cases with scores
        """
        k = k or self.top_k
        
        # If no embeddings or model, return empty
        if self.embedding_vectors is None or self.embedding_model is None:
            return [{"idx": i, "score": 0.0} for i in range(min(k, len(self.corpus_texts)))]
        
        # Get query embedding
        if query_embedding is None:
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )[0]
        
        # Compute cosine similarities
        similarities = np.dot(self.embedding_vectors, query_embedding)
        
        # Get top indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Build results
        results = []
        for idx in top_indices:
            result = {
                "idx": idx,
                "score": float(similarities[idx]),
                "combined_score": float(similarities[idx]),
                "vector_score": float(similarities[idx]),
            }
            
            # Add metadata if available
            if idx < len(self.metadata_list):
                metadata = self.metadata_list[idx]
                result["reward"] = metadata.get("reward", 0)
                result["error_identification"] = metadata.get("error_identification", "")
                result["task_type"] = metadata.get("task_type", "general")
            
            results.append(result)
        
        return results
    
    def _get_similarity(self, idx1: int, idx2: int) -> float:
        """
        Compute cosine similarity between two entries.
        Used for diverse selection (MMR).
        """
        if self.embedding_vectors is None:
            return 0.0
        
        emb1 = self.embedding_vectors[idx1]
        emb2 = self.embedding_vectors[idx2]
        return float(np.dot(emb1, emb2))