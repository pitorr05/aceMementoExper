import os
import json
import math
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import defaultdict

# Try to import torch and transformers for parametric memory
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch or transformers not available. Parametric retrieval will be disabled.")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("Warning: sentence-transformers not available. Non-parametric retrieval will use keyword overlap.")

# Try to import BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank-bm25 not installed. BM25 search will be disabled. Install with: pip install rank-bm25")

# Global registry for shared embedding models to avoid redundant loads and save RAM/VRAM
_SHARED_MODELS = {}

# --- Hybrid Search Configuration ---
HYBRID_DEFAULT_LIMIT = 20
HYBRID_DEFAULT_SIMILARITY_THRESHOLD = 0.5
HYBRID_OVERSAMPLE_MULTIPLIER = 2
RRF_K = 60

# Intent-based weights: (bm25_weight, vector_weight)
INTENT_WEIGHTS = {
    "recall": {"bm25": 0.6, "vector": 0.4},    # Tìm lại thông tin đã biết
    "explore": {"bm25": 0.3, "vector": 0.7},   # Khám phá khái niệm liên quan
    "exact": {"bm25": 0.8, "vector": 0.2},     # Khớp chính xác từ khóa
    "general": {"bm25": 0.4, "vector": 0.6},   # Cân bằng
}

# --- Helpers for formatting plans ---
def _parse_plan(plan_field: Union[str, dict, list, None]) -> Optional[Union[dict, list]]:
    if plan_field is None:
        return None
    if isinstance(plan_field, (dict, list)):
        return plan_field
    if isinstance(plan_field, str):
        s = plan_field.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return {"plan": [{"description": s}]}
    return None

def _pretty_plan(plan_obj: Union[dict, list]) -> str:
    try:
        steps = []
        if isinstance(plan_obj, dict) and "plan" in plan_obj and isinstance(plan_obj["plan"], list):
            for item in plan_obj["plan"]:
                if isinstance(item, dict):
                    sid = item.get("id")
                    desc = item.get("description") or item.get("desc") or item.get("step") or str(item)
                    steps.append(f"{sid}. {desc}" if sid is not None else f"- {desc}")
                else:
                    steps.append(f"- {str(item)}")
        elif isinstance(plan_obj, list):
            for i, item in enumerate(plan_obj, 1):
                if isinstance(item, dict):
                    desc = item.get("description") or item.get("desc") or item.get("step") or str(item)
                    steps.append(f"{i}. {desc}")
                else:
                    steps.append(f"{i}. {str(item)}")
        else:
            return json.dumps(plan_obj, ensure_ascii=False)
        return "\n".join(steps) if steps else json.dumps(plan_obj, ensure_ascii=False)
    except Exception:
        return json.dumps(plan_obj, ensure_ascii=False)

def build_icl_text(case: str, plan) -> str:
    parts = ["[CASE]", str(case)]
    if plan is not None:
        pobj = _parse_plan(plan)
        parts += ["[PLAN]", _pretty_plan(pobj) if pobj is not None else str(plan)]
    return "\n".join(parts).strip()

# --- Parametric classifier model architecture ---
if TORCH_AVAILABLE:
    class MemoryRetrieverClassifier(nn.Module):
        def __init__(self, sentence_bert: AutoModel):
            super().__init__()
            hidden = sentence_bert.config.hidden_size
            self.sentence_bert = sentence_bert
            self.classifier = nn.Sequential(
                nn.Linear(hidden * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 2)
            )

        def forward(self, ids1, mask1, ids2, mask2):
            o1 = self.sentence_bert(ids1, attention_mask=mask1).last_hidden_state[:, 0]
            o2 = self.sentence_bert(ids2, attention_mask=mask2).last_hidden_state[:, 0]
            return self.classifier(torch.cat([o1, o2], dim=1))
else:
    MemoryRetrieverClassifier = None

class CaseBank:
    """
    Episodic Case Memory với Hybrid Search (BM25 + Vector Similarity).
    
    Hỗ trợ:
        - Hybrid retrieval: Kết hợp BM25 (lexical matching) và Vector Similarity (semantic matching)
        - Reciprocal Rank Fusion (RRF) để kết hợp kết quả tối ưu
        - Intent-aware search: Điều chỉnh trọng số dựa trên intent (recall/explore/exact/general)
        - Parametric retrieval (MemoryRetrieverClassifier neural model) - vẫn giữ nguyên
        - Non-parametric retrieval (SentenceTransformer cosine similarity) - fallback
        - Keyword overlap - fallback cuối cùng
    """
    
    def __init__(
        self,
        memory_jsonl_path: str,
        top_k: int = 4,
        embedding_model_name: str = "BAAI/bge-m3",
        parametric_model_name: str = "princeton-nlp/sup-simcse-roberta-base",
        retriever_model_path: Optional[str] = None,
        device: str = "cpu",
        search_intent: str = "general"
    ):
        self.memory_jsonl_path = memory_jsonl_path
        self.top_k = top_k
        self.embedding_model_name = embedding_model_name
        self.parametric_model_name = parametric_model_name
        self.retriever_model_path = retriever_model_path
        self.device = device if device != "auto" else ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.search_intent = search_intent  # default intent
        
        self.cases: List[Dict[str, Any]] = []
        
        # Lazy loaded components
        self._emb_model = None
        self._embeddings = None  # numpy array (N, dim)
        self._corpus_texts = []  # lưu text cho BM25
        
        # BM25 index
        self._bm25_index = None
        
        # Parametric retriever models
        self._para_tokenizer = None
        self._para_model = None
        
        # Load cases from JSONL
        self.load_cases()
        self._init_parametric_retriever()
    
    def load_cases(self) -> None:
        """Load cases from JSONL file and rebuild all indices."""
        self.cases = []
        if not os.path.exists(self.memory_jsonl_path):
            return
        
        try:
            with open(self.memory_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.cases.append(json.loads(line))
                    except Exception:
                        pass
            print(f"[CaseBank] Loaded {len(self.cases)} cases from {self.memory_jsonl_path}")
            self._rebuild_indices()
        except Exception as e:
            print(f"[CaseBank] Error loading cases: {e}")
    
    def add_case(self, question: str, plan: str, reward: int) -> None:
        """Add a new case and update all indices."""
        case_entry = {
            "question": question,
            "plan": plan,
            "reward": int(reward)
        }
        self.cases.append(case_entry)
        
        # Write to JSONL file
        os.makedirs(os.path.dirname(self.memory_jsonl_path), exist_ok=True)
        try:
            with open(self.memory_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(case_entry, ensure_ascii=False) + "\n")
            print(f"[CaseBank] Case saved successfully (reward={reward})")
            self._update_indices(case_entry)
        except Exception as e:
            print(f"[CaseBank] Error writing case: {e}")
    
    def _init_parametric_retriever(self) -> None:
        """Initialize neural parametric classifier retriever if checkpoint is provided."""
        if not TORCH_AVAILABLE or not self.retriever_model_path or not os.path.exists(self.retriever_model_path):
            return
        
        try:
            print(f"[CaseBank] Loading neural classifier retriever from {self.retriever_model_path}")
            self._para_tokenizer = AutoTokenizer.from_pretrained(self.parametric_model_name)
            backbone = AutoModel.from_pretrained(self.parametric_model_name)
            
            self._para_model = MemoryRetrieverClassifier(backbone).to(self.device)
            self._para_model.load_state_dict(torch.load(self.retriever_model_path, map_location=self.device))
            self._para_model.eval()
            print("[CaseBank] Parametric CaseRetriever loaded successfully")
        except Exception as e:
            print(f"[CaseBank] Error loading parametric retriever model: {e}")
            self._para_model = None
    
    def _load_emb_model(self) -> None:
        """Load shared embedding model."""
        if self._emb_model is None and EMBEDDING_AVAILABLE:
            try:
                key = (self.embedding_model_name, self.device)
                if key not in _SHARED_MODELS:
                    print(f"[CaseBank] Loading shared model: {self.embedding_model_name} on {self.device}")
                    _SHARED_MODELS[key] = SentenceTransformer(self.embedding_model_name, device=self.device)
                self._emb_model = _SHARED_MODELS[key]
            except Exception as e:
                print(f"[CaseBank] Error loading embedding model: {e}")
    
    def _rebuild_indices(self) -> None:
        """Rebuild all indices (BM25, Vector) from scratch."""
        if not self.cases:
            self._embeddings = None
            self._corpus_texts = []
            self._bm25_index = None
            return
        
        # 1. Rebuild BM25 index
        self._corpus_texts = [c["question"] for c in self.cases]
        if BM25_AVAILABLE and len(self.cases) > 0:
            try:
                # Tokenize documents
                tokenized_corpus = [doc.split(" ") for doc in self._corpus_texts]
                self._bm25_index = BM25Okapi(tokenized_corpus)
            except Exception as e:
                print(f"[CaseBank] Error building BM25 index: {e}")
                self._bm25_index = None
        else:
            self._bm25_index = None
        
        # 2. Rebuild vector embeddings
        if EMBEDDING_AVAILABLE and self._emb_model is None:
            self._load_emb_model()
        
        if EMBEDDING_AVAILABLE and self._emb_model is not None:
            try:
                texts = [c["question"] for c in self.cases]
                self._embeddings = self._emb_model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            except Exception as e:
                print(f"[CaseBank] Error encoding cases: {e}")
                self._embeddings = None
        else:
            self._embeddings = None
    
    def _update_indices(self, new_case: Dict[str, Any]) -> None:
        """Update indices when a new case is added."""
        # Update BM25 - rebuild if BM25 is available
        if BM25_AVAILABLE and self._bm25_index is not None:
            self._corpus_texts.append(new_case["question"])
            try:
                tokenized_corpus = [doc.split(" ") for doc in self._corpus_texts]
                self._bm25_index = BM25Okapi(tokenized_corpus)
            except Exception as e:
                print(f"[CaseBank] Error updating BM25 index: {e}")
                self._bm25_index = None
        else:
            self._corpus_texts.append(new_case["question"])
        
        # Update vector embeddings
        if EMBEDDING_AVAILABLE and self._emb_model is not None:
            try:
                new_emb = self._emb_model.encode(
                    [new_case["question"]],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                if self._embeddings is None or len(self._embeddings) == 0:
                    self._embeddings = new_emb
                else:
                    self._embeddings = np.vstack([self._embeddings, new_emb])
            except Exception as e:
                print(f"[CaseBank] Error updating vector index: {e}")
                # Fallback: rebuild all
                self._rebuild_indices()
    
    def retrieve_cases(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        search_intent: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve Top-K relevant cases using Hybrid Search (BM25 + Vector).
        
        Args:
            query: The query string
            top_k: Number of cases to return (default: self.top_k)
            search_intent: Intent type - 'recall', 'explore', 'exact', 'general'
                          (default: self.search_intent)
        
        Returns:
            List of cases with scores
        """
        k = top_k if top_k is not None else self.top_k
        intent = search_intent or self.search_intent
        
        if not self.cases or k <= 0:
            return []
        
        # 1. Try parametric retrieval first (neural classifier)
        if TORCH_AVAILABLE and self._para_model is not None:
            try:
                return self._parametric_retrieval(query, k)
            except Exception as e:
                print(f"[CaseBank] Error in parametric retrieval: {e}")
                # Fall through to hybrid search
        
        # 2. Hybrid Search: BM25 + Vector
        if BM25_AVAILABLE and self._bm25_index is not None and EMBEDDING_AVAILABLE and self._embeddings is not None:
            try:
                return self._hybrid_retrieval(query, k, intent)
            except Exception as e:
                print(f"[CaseBank] Error in hybrid retrieval: {e}")
                # Fall through to vector-only
        
        # 3. Fallback: Vector-only retrieval
        if self._embeddings is not None and EMBEDDING_AVAILABLE:
            try:
                return self._vector_only_retrieval(query, k)
            except Exception as e:
                print(f"[CaseBank] Error in vector retrieval: {e}")
                # Fall through to keyword fallback
        
        # 4. Final fallback: keyword overlap
        return self._keyword_fallback(query, k)
    
    def _parametric_retrieval(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Parametric retrieval using neural classifier."""
        icl_pool = [build_icl_text(c["question"], c["plan"]) for c in self.cases]
        probs = self._score_batch(query, icl_pool)
        
        results = []
        for i, (case, score) in enumerate(zip(self.cases, probs)):
            case_copy = case.copy()
            case_copy["score"] = score
            results.append(case_copy)
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
    
    def _hybrid_retrieval(self, query: str, k: int, intent: str) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining BM25 and Vector similarity."""
        oversample = HYBRID_OVERSAMPLE_MULTIPLIER
        
        # BM25 search
        bm25_results = self._bm25_search(query, limit=k * oversample)
        
        # Vector search
        vector_results = self._vector_search(query, limit=k * oversample)
        
        # Get weights for intent
        weights = INTENT_WEIGHTS.get(intent, INTENT_WEIGHTS["general"])
        bm25_weight = weights["bm25"]
        vector_weight = weights["vector"]
        
        # Combine using RRF
        fused_results = self._reciprocal_rank_fusion(
            bm25_results,
            vector_results,
            bm25_weight,
            vector_weight,
            k
        )
        
        # Format results
        final_cases = []
        for item in fused_results:
            case = self.cases[item["idx"]].copy()
            case["score"] = item["score"]
            final_cases.append(case)
        
        return final_cases
    
    def _bm25_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """BM25 search."""
        if not BM25_AVAILABLE or self._bm25_index is None or not self._corpus_texts:
            return []
        
        tokenized_query = query.split(" ")
        scores = self._bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:limit]
        
        results = []
        for idx in top_indices:
            results.append({"idx": idx, "score": float(scores[idx])})
        return results
    
    def _vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Vector similarity search."""
        if self._embeddings is None or self._emb_model is None:
            return []
        
        query_emb = self._emb_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]
        
        similarities = np.dot(self._embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:limit]
        
        results = []
        for idx in top_indices:
            results.append({"idx": idx, "score": float(similarities[idx])})
        return results
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        bm25_weight: float,
        vector_weight: float,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF) with sigmoid normalization.
        """
        k = RRF_K
        rrf_scores = defaultdict(float)
        memory_map = {}
        
        # Process BM25 results
        for rank, item in enumerate(bm25_results, 1):
            idx = item["idx"]
            rrf_scores[idx] += bm25_weight * (1.0 / (k + rank))
            if idx not in memory_map:
                memory_map[idx] = {"idx": idx}
        
        # Process Vector results
        for rank, item in enumerate(vector_results, 1):
            idx = item["idx"]
            rrf_scores[idx] += vector_weight * (1.0 / (k + rank))
            if idx not in memory_map:
                memory_map[idx] = {"idx": idx}
        
        # Sort and normalize
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        def sigmoid_normalize(raw_score: float) -> float:
            # Sigmoid: 1 / (1 + exp(-k * (x - midpoint)))
            # k=1000 provides good spread, midpoint=0.009 centers on typical RRF range
            return 1.0 / (1.0 + math.exp(-1000 * (raw_score - 0.009)))
        
        final_results = []
        for idx, raw_score in sorted_items[:limit]:
            item = memory_map[idx]
            item["score"] = sigmoid_normalize(raw_score)
            final_results.append(item)
        
        return final_results
    
    def _vector_only_retrieval(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Vector-only retrieval (fallback when BM25 is unavailable)."""
        self._load_emb_model()
        if self._emb_model is None:
            return self._keyword_fallback(query, k)
        
        query_emb = self._emb_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]
        
        similarities = np.dot(self._embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:k]
        
        retrieved = []
        for idx in top_indices:
            case = self.cases[idx].copy()
            case["similarity"] = float(similarities[idx])
            case["score"] = float(similarities[idx])
            retrieved.append(case)
        return retrieved
    
    def _keyword_fallback(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Simple keyword overlap fallback."""
        results = []
        query_words = set(query.lower().split())
        for idx, c in enumerate(self.cases):
            q_words = set(c["question"].lower().split())
            overlap = len(query_words.intersection(q_words))
            results.append((overlap, idx))
        results.sort(key=lambda x: x[0], reverse=True)
        ret_indices = [idx for score, idx in results[:k]]
        return [self.cases[idx] for idx in ret_indices]
    
    @torch.inference_mode()
    def _score_batch(self, query: str, icl_pool: List[str]) -> List[float]:
        """Compute classifier probability scores using parametric neural retriever."""
        if self._para_tokenizer is None or self._para_model is None:
            return []
        
        t1 = self._para_tokenizer(icl_pool, padding=True, truncation=True, return_tensors="pt")
        t2 = self._para_tokenizer([query] * len(icl_pool), padding=True, truncation=True, return_tensors="pt")
        
        ids1 = t1["input_ids"].to(self.device)
        mask1 = t1["attention_mask"].to(self.device)
        ids2 = t2["input_ids"].to(self.device)
        mask2 = t2["attention_mask"].to(self.device)
        
        logits = self._para_model(ids1, mask1, ids2, mask2)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        return probs
    
    def format_cases_for_prompt(self, retrieved_cases: List[Dict[str, Any]], max_pos: int = 3, max_neg: int = 3) -> str:
        """Format retrieved positive and negative cases into a prompt block."""
        if not retrieved_cases:
            return "No previous cases found in Case Memory."
        
        positive_cases = [c for c in retrieved_cases if c.get("reward", 0) == 1]
        negative_cases = [c for c in retrieved_cases if c.get("reward", 0) == 0]
        
        prompt_parts: List[str] = []
        
        if positive_cases:
            prompt_parts.append(f"### Successful Examples (reward=1) - Showing up to {max_pos}:")
            for i, case in enumerate(positive_cases[:max_pos], 1):
                prompt_parts.append(
                    f"Example {i}:\n"
                    f"Question: {case['question']}\n"
                    f"Plan:\n{case['plan']}\n"
                )
        
        if negative_cases:
            prompt_parts.append(f"### Unsuccessful Examples (reward=0) - Showing up to {max_neg}:")
            for i, case in enumerate(negative_cases[:max_neg], 1):
                prompt_parts.append(
                    f"Example {i}:\n"
                    f"Question: {case['question']}\n"
                    f"Plan:\n{case['plan']}\n"
                )
        
        if not prompt_parts:
            return "No structured examples found in Case Memory."
        
        return "\n".join(prompt_parts)
    
    def set_search_intent(self, intent: str) -> None:
        """Set the default search intent."""
        if intent in INTENT_WEIGHTS:
            self.search_intent = intent
            print(f"[CaseBank] Search intent set to: {intent}")
        else:
            print(f"[CaseBank] Unknown intent: {intent}. Available: {list(INTENT_WEIGHTS.keys())}")
