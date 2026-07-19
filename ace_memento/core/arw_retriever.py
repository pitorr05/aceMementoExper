import numpy as np
from typing import List, Dict, Any, Optional
import json
import os

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


class ARWRetriever:
    """
    Adaptive Retriever Weighting (ARW) for CaseBank.
    Combines BM25, Semantic, Temporal, and Memento retrievers with adaptive weights.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-m3",
        top_k: int = 4,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.top_k = top_k
        self.device = device
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        
        # ARW state: 4 retrievers
        self.num_retrievers = 4
        self.weights = np.ones(self.num_retrievers) / self.num_retrievers
        self.m = np.zeros(self.num_retrievers)  # Momentum
        self.v = np.zeros(self.num_retrievers)  # Variance
        self.t = 0  # Iteration counter
        
        # Retriever names
        self.retriever_names = ["BM25", "Semantic", "Temporal", "Memento"]
        
        # Embedding model
        self._embedding_model = None
        self.embedding_model_name = embedding_model_name
        
        # Cache for cases
        self.cases = []
        self._bm25_index = None
        self._embeddings = None
        self._case_timestamps = []
        
    def _load_embedding_model(self):
        if self._embedding_model is None and EMBEDDING_AVAILABLE:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(
                    self.embedding_model_name, 
                    device=self.device
                )
            except Exception as e:
                print(f"[ARW] Error loading embedding model: {e}")
                
    def _init_bm25(self):
        """Initialize BM25 index from cases."""
        if not BM25_AVAILABLE or not self.cases:
            return
        try:
            tokenized_cases = [c["question"].lower().split() for c in self.cases]
            self._bm25_index = BM25Okapi(tokenized_cases)
        except Exception as e:
            print(f"[ARW] Error initializing BM25: {e}")
            
    def _build_embeddings(self):
        """Build embeddings for all cases."""
        if not EMBEDDING_AVAILABLE or not self.cases:
            return
        self._load_embedding_model()
        if self._embedding_model is None:
            return
        try:
            texts = [c["question"] for c in self.cases]
            self._embeddings = self._embedding_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        except Exception as e:
            print(f"[ARW] Error building embeddings: {e}")
            
    def _get_bm25_scores(self, query: str) -> np.ndarray:
        """Get BM25 scores for all cases."""
        if self._bm25_index is None or not self.cases:
            return np.zeros(len(self.cases))
        try:
            tokenized_query = query.lower().split()
            scores = self._bm25_index.get_scores(tokenized_query)
            # Normalize to [0, 1]
            if scores.max() > 0:
                scores = scores / scores.max()
            return scores
        except Exception:
            return np.zeros(len(self.cases))
            
    def _get_semantic_scores(self, query: str) -> np.ndarray:
        """Get semantic (embedding) scores for all cases."""
        if self._embeddings is None or self._embedding_model is None or not self.cases:
            return np.zeros(len(self.cases))
        try:
            query_emb = self._embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )[0]
            similarities = np.dot(self._embeddings, query_emb)
            return similarities
        except Exception:
            return np.zeros(len(self.cases))
            
    def _get_temporal_scores(self, query: str) -> np.ndarray:
        """Get temporal scores (recency weighting)."""
        if not self._case_timestamps or not self.cases:
            return np.ones(len(self.cases))
        try:
            # Simple recency: newer cases get higher scores
            max_time = max(self._case_timestamps)
            if max_time == 0:
                return np.ones(len(self.cases))
            scores = np.array([t / max_time for t in self._case_timestamps])
            return scores
        except Exception:
            return np.ones(len(self.cases))
            
    def _get_memento_scores(self, query: str) -> np.ndarray:
        """Get Memento scores (based on reward history)."""
        if not self.cases:
            return np.zeros(len(self.cases))
        try:
            # Higher reward = higher score
            scores = np.array([c.get("reward", 0) for c in self.cases])
            # Normalize to [0, 1]
            if scores.max() > 0:
                scores = scores / scores.max()
            return scores
        except Exception:
            return np.zeros(len(self.cases))
            
    def update_weights(self, scores: List[float], reward: int) -> None:
        """
        Update ARW weights based on retrieval feedback.
        scores: list of scores from each retriever for the query
        reward: 1 if correct, 0 if incorrect
        """
        self.t += 1
        g = reward * np.array(scores)
        
        # Adam update
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g**2
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        self.weights += self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        # Softmax normalization
        exp_w = np.exp(self.weights - np.max(self.weights))
        self.weights = exp_w / exp_w.sum()
        
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve Top-K cases using weighted fusion."""
        k = top_k if top_k is not None else self.top_k
        if not self.cases:
            return []
            
        # Get scores from each retriever
        bm25_scores = self._get_bm25_scores(query)
        semantic_scores = self._get_semantic_scores(query)
        temporal_scores = self._get_temporal_scores(query)
        memento_scores = self._get_memento_scores(query)
        
        # Weighted fusion
        all_scores = np.array([bm25_scores, semantic_scores, temporal_scores, memento_scores])
        final_scores = np.dot(self.weights, all_scores)
        
        # Get top-k indices
        top_indices = np.argsort(final_scores)[::-1][:k]
        
        # Build results
        results = []
        for idx in top_indices:
            case = self.cases[idx].copy()
            case["score"] = float(final_scores[idx])
            case["bm25_score"] = float(bm25_scores[idx])
            case["semantic_score"] = float(semantic_scores[idx])
            case["temporal_score"] = float(temporal_scores[idx])
            case["memento_score"] = float(memento_scores[idx])
            results.append(case)
            
        return results
        
    def add_case(self, question: str, plan: str, reward: int) -> None:
        """Add a new case and update indices."""
        import time
        case_entry = {
            "question": question,
            "plan": plan,
            "reward": int(reward)
        }
        self.cases.append(case_entry)
        self._case_timestamps.append(time.time())
        
        # Rebuild indices
        self._init_bm25()
        self._build_embeddings()
        
    def load_cases(self, file_path: str) -> None:
        """Load cases from JSONL file."""
        if not os.path.exists(file_path):
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            case = json.loads(line)
                            self.cases.append(case)
                            self._case_timestamps.append(0)  # Placeholder
                        except Exception:
                            pass
            print(f"[ARW] Loaded {len(self.cases)} cases")
            self._init_bm25()
            self._build_embeddings()
        except Exception as e:
            print(f"[ARW] Error loading cases: {e}")
            
    def format_for_prompt(self, retrieved_cases: List[Dict], max_pos: int = 3, max_neg: int = 3) -> str:
        """Format retrieved cases for prompt."""
        if not retrieved_cases:
            return "No previous cases found."
            
        positive = [c for c in retrieved_cases if c.get("reward", 0) == 1]
        negative = [c for c in retrieved_cases if c.get("reward", 0) == 0]
        
        parts = []
        if positive:
            parts.append("### Successful Examples:")
            for i, case in enumerate(positive[:max_pos], 1):
                parts.append(f"Example {i}:")
                parts.append(f"Question: {case['question']}")
                parts.append(f"Plan: {case['plan']}\n")
                
        if negative:
            parts.append("### Unsuccessful Examples:")
            for i, case in enumerate(negative[:max_neg], 1):
                parts.append(f"Example {i}:")
                parts.append(f"Question: {case['question']}")
                parts.append(f"Plan: {case['plan']}\n")
                
        return "\n".join(parts) if parts else "No structured examples found."