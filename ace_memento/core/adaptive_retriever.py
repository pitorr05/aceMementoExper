"""
Adaptive Retriever with Multi-Granular Filtering
Based on AdaMEM 2025 and Amber 2025
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import math

class AdaptiveRetriever:
    """
    Adaptive retrieval with:
    1. Multi-granular filtering (semantic, contextual)
    2. Adaptive strategy (different retrieval per round)
    3. Cosine similarity only (no BM25)
    """
    
    def __init__(
        self,
        embedding_model,
        embedding_vectors: np.ndarray,
        corpus_texts: List[str],
        metadata_list: List[Dict[str, Any]],
        top_k: int = 4,
    ):
        self.embedding_model = embedding_model
        self.embedding_vectors = embedding_vectors
        self.corpus_texts = corpus_texts
        self.metadata_list = metadata_list
        self.top_k = top_k
        
        # BM25 disabled - keeping for compatibility
        self.bm25_index = None
    
    def set_bm25_index(self, bm25_index):
        """Set BM25 index (disabled - kept for compatibility)"""
        self.bm25_index = bm25_index
        # BM25 is disabled, but method kept for compatibility
    
    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        round_idx: int = 0,
        previous_error: Optional[str] = None,
        k: Optional[int] = None,
        intent: str = "general"
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval method with adaptive strategy.
        Uses ONLY Cosine Similarity (no BM25).
        """
        k = k or self.top_k
        
        # --- Step 1: Get all candidates (no BM25 filtering) ---
        candidates = [{"idx": i} for i in range(len(self.corpus_texts))]
        
        # --- Step 2: Semantic filtering (Cosine Similarity) ---
        candidates = self._semantic_filter(query, query_embedding, candidates, limit=k * 2)
        
        # --- Step 3: Contextual filtering ---
        candidates = self._contextual_filter(candidates, round_idx, previous_error)
        
        # --- Step 4: Adaptive Strategy ---
        if round_idx == 0:
            results = self._diverse_selection(candidates, k)
        else:
            results = self._error_aware_selection(candidates, previous_error, k)
        
        return results
    
    def _semantic_filter(
        self, 
        query: str, 
        query_embedding: Optional[np.ndarray],
        candidates: List[Dict[str, Any]], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Vector semantic reranking using Cosine Similarity"""
        if self.embedding_vectors is None or self.embedding_model is None:
            return candidates
        
        # Get query embedding
        if query_embedding is None:
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )[0]
        
        # Compute cosine similarities for all candidates
        for item in candidates:
            idx = item["idx"]
            emb = self.embedding_vectors[idx]
            similarity = np.dot(emb, query_embedding)  # Cosine similarity
            item["vector_score"] = float(similarity)
            item["combined_score"] = similarity  # Only vector score
        
        # Sort by combined score
        sorted_candidates = sorted(
            candidates, 
            key=lambda x: x.get("combined_score", 0), 
            reverse=True
        )
        
        return sorted_candidates[:limit]
    
    def _contextual_filter(
        self,
        candidates: List[Dict[str, Any]],
        round_idx: int,
        previous_error: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Contextual filtering based on metadata"""
        filtered = []
        
        for item in candidates:
            idx = item["idx"]
            metadata = self.metadata_list[idx]
            
            item["task_type"] = metadata.get("task_type", "general")
            item["reward"] = metadata.get("reward", 0)
            
            score_boost = 0.0
            
            # Boost successful cases
            if metadata.get("reward") == 1:
                score_boost += 0.1
            
            # Boost cases with similar error
            if previous_error and metadata.get("error_identification") == previous_error:
                score_boost += 0.2
            
            item["combined_score"] = item.get("combined_score", 0) + score_boost
            filtered.append(item)
        
        filtered.sort(key=lambda x: x["combined_score"], reverse=True)
        return filtered
    
    def _diverse_selection(
        self, 
        candidates: List[Dict[str, Any]], 
        k: int
    ) -> List[Dict[str, Any]]:
        """Select diverse cases (MMR-style)"""
        if len(candidates) <= k:
            return candidates[:k]
        
        selected = []
        remaining = candidates[:]
        
        while len(selected) < k and remaining:
            best_idx = 0
            best_score = -float('inf')
            
            for i, item in enumerate(remaining):
                relevance = item.get("combined_score", 0)
                
                max_similarity = 0
                for sel in selected:
                    sim = self._get_similarity(item["idx"], sel["idx"])
                    max_similarity = max(max_similarity, sim)
                
                mmr_score = 0.7 * relevance - 0.3 * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _error_aware_selection(
        self,
        candidates: List[Dict[str, Any]],
        previous_error: Optional[str],
        k: int
    ) -> List[Dict[str, Any]]:
        """Select cases based on previous error"""
        if previous_error:
            error_cases = [
                item for item in candidates 
                if item.get("error_identification") == previous_error
            ]
            
            top_k_error = min(int(k * 0.7), len(error_cases))
            top_k_diverse = k - top_k_error
            
            results = error_cases[:top_k_error]
            remaining = [item for item in candidates if item not in results]
            results.extend(remaining[:top_k_diverse])
            
            return results
        
        return candidates[:k]
    
    def _get_similarity(self, idx1: int, idx2: int) -> float:
        """Compute cosine similarity between two entries"""
        emb1 = self.embedding_vectors[idx1]
        emb2 = self.embedding_vectors[idx2]
        return float(np.dot(emb1, emb2))