"""
Memory Consolidator for AMORE
Utility-based pruning and semantic abstraction
"""
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

class MemoryConsolidator:
    """
    Manages memory lifecycle:
    1. Utility-based pruning
    2. Semantic abstraction (clustering similar entries)
    """
    
    def __init__(
        self,
        utility_threshold: float = 0.15,
        abstraction_similarity_threshold: float = 0.85,
        min_cluster_size: int = 3,
        alpha_reward: float = 0.5,
        beta_access: float = 0.3,
        gamma_recency: float = 0.2
    ):
        self.utility_threshold = utility_threshold
        self.abstraction_similarity_threshold = abstraction_similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.alpha_reward = alpha_reward
        self.beta_access = beta_access
        self.gamma_recency = gamma_recency
    
    def consolidate(
        self,
        entries: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Main consolidation function
        
        Returns:
            Pruned and abstracted list of entries
        """
        if not entries:
            return []
        
        # --- Step 1: Update utility scores ---
        for entry in entries:
            entry["utility_score"] = self._calculate_utility(entry)
        
        # --- Step 2: Prune low utility entries ---
        pruned = [
            entry for entry in entries 
            if entry["utility_score"] >= self.utility_threshold
        ]
        
        print(f"[Consolidator] Pruned {len(entries) - len(pruned)} entries")
        
        # --- Step 3: Semantic abstraction ---
        if embeddings is not None and len(pruned) >= self.min_cluster_size:
            abstracted = self._abstract_clusters(pruned, embeddings)
        else:
            abstracted = pruned
        
        return abstracted
    
    def _calculate_utility(self, entry: Dict[str, Any]) -> float:
        """
        Utility = α * reward + β * access_freq + γ * recency
        """
        # Reward component
        reward = entry.get("reward", 0)
        reward_score = 1.0 if reward == 1 else 0.3
        
        # Access frequency (capped at 10)
        access_count = entry.get("access_count", 0)
        access_score = min(access_count / 10.0, 1.0)
        
        # Recency
        timestamp_str = entry.get("timestamp", "")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                days_ago = (datetime.now() - timestamp).days
                # Recent entries get higher score (decay over 30 days)
                recency_score = max(0.0, 1.0 - days_ago / 30.0)
            except:
                recency_score = 0.5
        else:
            recency_score = 0.5
        
        # Weighted sum
        utility = (
            self.alpha_reward * reward_score +
            self.beta_access * access_score +
            self.gamma_recency * recency_score
        )
        
        return utility
    
    def _abstract_clusters(
        self,
        entries: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Cluster similar entries and abstract them into prototypes
        """
        from sklearn.cluster import DBSCAN
        
        # Filter entries with embeddings
        valid_indices = []
        valid_embeddings = []
        valid_entries = []
        
        for i, entry in enumerate(entries):
            if entry.get("idx") is not None and entry["idx"] < len(embeddings):
                valid_indices.append(i)
                valid_embeddings.append(embeddings[entry["idx"]])
                valid_entries.append(entry)
        
        if len(valid_entries) < self.min_cluster_size:
            return entries
        
        # Cluster
        valid_embeddings = np.array(valid_embeddings)
        clustering = DBSCAN(
            eps=1 - self.abstraction_similarity_threshold,
            min_samples=self.min_cluster_size,
            metric='cosine'
        ).fit(valid_embeddings)
        
        labels = clustering.labels_
        
        # Track cluster sizes
        from collections import Counter
        cluster_counts = Counter(labels)
        
        abstracted = []
        for i, label in enumerate(labels):
            entry = valid_entries[i]
            
            if label == -1:
                # Not in any cluster
                abstracted.append(entry)
            else:
                # In a cluster
                if cluster_counts[label] >= self.min_cluster_size:
                    # Mark for abstraction (we keep only the prototype)
                    # For now, we just keep the entry with highest reward
                    # In a more advanced version, we'd create a prototype
                    entry["is_abstracted"] = True
                    entry["cluster_id"] = int(label)
                    abstracted.append(entry)
                else:
                    # Cluster too small
                    abstracted.append(entry)
        
        return abstracted