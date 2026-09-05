"""
Adaptive Memory Entry with rich metadata (P, S, O, M)
Based on CBR Review 2025
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
from datetime import datetime

@dataclass
class AdaptiveMemoryEntry:
    """
    A rich memory entry following CBR theory: (P, S, O, M)
    
    P: Problem Space - features of the problem
    S: Solution Space - actions/plan taken
    O: Outcome Space - evaluation metrics
    M: Metadata - temporal markers, conditions
    """
    
    # --- Problem Space (P) ---
    question: str                                    # Raw question
    question_embedding: Optional[np.ndarray] = None  # Semantic vector
    task_type: str = "general"                       # "formula", "finer"
    domain: str = "unknown"                          # "finance", "math"
    difficulty_score: float = 0.5                    # Estimated difficulty
    keywords: List[str] = field(default_factory=list)  # Extracted keywords
    
    # --- Solution Space (S) ---
    plan: str = ""                                   # Generated plan
    bullet_ids_used: List[str] = field(default_factory=list)
    reasoning_trace: str = ""                        # Step-by-step reasoning
    final_answer: str = ""                           # Final answer
    
    # --- Outcome Space (O) ---
    reward: int = 0                                   # 0 (wrong) or 1 (correct)
    confidence_score: float = 0.0                     # From Reflector
    error_identification: str = ""                    # Type of error if wrong
    root_cause: str = ""                             # Why error occurred
    key_insight: str = ""                            # Lesson learned
    
    # --- Metadata (M) ---
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    last_accessed: Optional[str] = None
    consolidation_version: int = 0
    utility_score: float = 0.0                        # Computed dynamically
    
    def update_access(self) -> None:
        """Update access statistics when retrieved"""
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSONL storage"""
        return {
            "question": self.question,
            "plan": self.plan,
            "bullet_ids_used": self.bullet_ids_used,
            "reasoning_trace": self.reasoning_trace,
            "final_answer": self.final_answer,
            "reward": self.reward,
            "confidence_score": self.confidence_score,
            "error_identification": self.error_identification,
            "root_cause": self.root_cause,
            "key_insight": self.key_insight,
            "task_type": self.task_type,
            "domain": self.domain,
            "difficulty_score": self.difficulty_score,
            "keywords": self.keywords,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "consolidation_version": self.consolidation_version,
            "utility_score": self.utility_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveMemoryEntry":
        """Deserialize from dictionary"""
        # Remove None values
        clean_data = {k: v for k, v in data.items() if v is not None}
        return cls(**clean_data)