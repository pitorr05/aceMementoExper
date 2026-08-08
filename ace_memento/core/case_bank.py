import os
import json
import math
import numpy as np
import re
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import defaultdict
from datetime import datetime

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
    print("Warning: rank-bm25 not available. BM25 retrieval will be disabled.")
    print("Install with: pip install rank-bm25")

# Global registry for shared embedding models
_SHARED_MODELS = {}

# ====== MIRA-OSS HYBRID SEARCH CONSTANTS ======
# Search defaults
HYBRID_DEFAULT_LIMIT = 20
HYBRID_DEFAULT_SIMILARITY_THRESHOLD = 0.5
HYBRID_DEFAULT_MIN_IMPORTANCE = 0.1
HYBRID_OVERSAMPLE_MULTIPLIER = 2

# Intent-based weights: (bm25_weight, vector_weight)
INTENT_RECALL_BM25 = 0.6
INTENT_RECALL_VECTOR = 0.4
INTENT_EXPLORE_BM25 = 0.3
INTENT_EXPLORE_VECTOR = 0.7
INTENT_EXACT_BM25 = 0.8
INTENT_EXACT_VECTOR = 0.2
INTENT_GENERAL_BM25 = 0.4
INTENT_GENERAL_VECTOR = 0.6

# Reciprocal Rank Fusion
RRF_K = 60


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
    Episodic Case Memory (Memento style).
    
    Hybrid search implementation combining BM25 text search with vector similarity.
    Inspired by mira-OSS hybrid_search.py.
    
    Supports:
      - Parametric retrieval (MemoryRetrieverClassifier neural model)
      - Dense retrieval (SentenceTransformer/BGE-M3)
      - BM25 retrieval (rank_bm25)
      - Hybrid retrieval with Weighted RRF + Sigmoid normalization
      - Intent-aware weights (recall/explore/exact/general)
    """

    def __init__(
        self,
        memory_jsonl_path: str,
        top_k: int = 8,
        embedding_model_name: str = "BAAI/bge-m3",
        parametric_model_name: str = "princeton-nlp/sup-simcse-roberta-base",
        retriever_model_path: Optional[str] = None,
        device: str = "cpu",
        # --- Hybrid retrieval parameters ---
        use_hybrid: bool = True,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        # --- BM25 document type ---
        bm25_doc_type: str = "question_only",  # "question_only" or "question_plan"
        # --- BM25 normalization ---
        bm25_norm: str = "softmax",  # "softmax", "sigmoid", "minmax"
        # --- Default intent (mira-OSS) ---
        default_intent: str = "general",
        # --- Utility (optional) ---
        use_reward_bias: bool = True,
        use_utility: bool = True,
    ):
        self.memory_jsonl_path = memory_jsonl_path
        self.top_k = top_k
        self.embedding_model_name = embedding_model_name
        self.parametric_model_name = parametric_model_name
        self.retriever_model_path = retriever_model_path
        self.device = device if device != "auto" else ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        
        # --- Hybrid retrieval config ---
        self.use_hybrid = use_hybrid
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.bm25_doc_type = bm25_doc_type
        self.bm25_norm = bm25_norm
        self.default_intent = default_intent
        self.use_reward_bias = use_reward_bias
        self.use_utility = use_utility
        
        # Validate weights
        if self.use_hybrid:
            total_weight = self.dense_weight + self.bm25_weight
            if abs(total_weight - 1.0) > 1e-6:
                print(f"[CaseBank] Warning: Weights sum to {total_weight}, normalizing to 1.0")
                self.dense_weight = self.dense_weight / total_weight
                self.bm25_weight = self.bm25_weight / total_weight
            print(f"[CaseBank] Hybrid: dense={self.dense_weight:.2f}, bm25={self.bm25_weight:.2f}")
        
        print(f"[CaseBank] BM25 doc_type: {self.bm25_doc_type}, norm: {self.bm25_norm}")
        print(f"[CaseBank] Default intent: {self.default_intent}")
        
        self.cases: List[Dict[str, Any]] = []

        # Lazy loaded components
        self._emb_model = None
        self._embeddings = None

        # Parametric retriever models
        self._para_tokenizer = None
        self._para_model = None

        # --- BM25 components ---
        self._bm25_index = None
        self._tokenized_corpus = None

        # Load cases from JSONL
        self.load_cases()
        self._init_parametric_retriever()

    # ======================================================================
    # 1. LOAD / SAVE CASES
    # ======================================================================

    def load_cases(self) -> None:
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
                        case = json.loads(line)
                        if "timestamp" not in case:
                            case["timestamp"] = datetime.now().isoformat()
                        if "usage_count" not in case:
                            case["usage_count"] = 0
                        self.cases.append(case)
                    except Exception:
                        pass
            print(f"[CaseBank] Loaded {len(self.cases)} cases from {self.memory_jsonl_path}")
            self._rebuild_embeddings()
            self._build_bm25_index()
        except Exception as e:
            print(f"[CaseBank] Error loading cases: {e}")

    def add_case(self, question: str, plan: str, reward: int) -> None:
        case_entry = {
            "question": question,
            "plan": plan,
            "reward": int(reward),
            "timestamp": datetime.now().isoformat(),
            "usage_count": 1,
        }
        self.cases.append(case_entry)

        os.makedirs(os.path.dirname(self.memory_jsonl_path), exist_ok=True)
        try:
            with open(self.memory_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(case_entry, ensure_ascii=False) + "\n")
            print(f"[CaseBank] Case saved successfully (reward={reward})")
            self._append_embedding(question)
            self._build_bm25_index()
        except Exception as e:
            print(f"[CaseBank] Error writing case: {e}")

    # ======================================================================
    # 2. DENSE EMBEDDING (BGE-M3)
    # ======================================================================

    def _load_emb_model(self) -> None:
        if self._emb_model is None and EMBEDDING_AVAILABLE:
            try:
                key = (self.embedding_model_name, self.device)
                if key not in _SHARED_MODELS:
                    print(f"[CaseBank] Loading shared model: {self.embedding_model_name} on {self.device}")
                    _SHARED_MODELS[key] = SentenceTransformer(self.embedding_model_name, device=self.device)
                self._emb_model = _SHARED_MODELS[key]
            except Exception as e:
                print(f"[CaseBank] Error loading embedding model: {e}")

    def _append_embedding(self, question: str) -> None:
        if not EMBEDDING_AVAILABLE or self.retriever_model_path:
            return
        self._load_emb_model()
        if self._emb_model is None:
            return
        try:
            new_emb = self._emb_model.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            if self._embeddings is None or len(self._embeddings) == 0:
                self._embeddings = new_emb
            else:
                self._embeddings = np.vstack([self._embeddings, new_emb])
        except Exception as e:
            print(f"[CaseBank] Error encoding single case: {e}")
            self._rebuild_embeddings()

    def _rebuild_embeddings(self) -> None:
        if not EMBEDDING_AVAILABLE or not self.cases or self.retriever_model_path:
            self._embeddings = None
            return

        self._load_emb_model()
        if self._emb_model is None:
            return

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

    def _get_dense_scores(self, query: str) -> Optional[np.ndarray]:
        """Get dense scores for all cases."""
        if self._embeddings is None or not EMBEDDING_AVAILABLE:
            return None
        try:
            self._load_emb_model()
            query_emb = self._emb_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )[0]
            scores = np.dot(self._embeddings, query_emb)
            # Normalize from [-1, 1] to [0, 1]
            return (scores + 1) / 2
        except Exception as e:
            print(f"[CaseBank] Error getting dense scores: {e}")
            return None

    # ======================================================================
    # 3. BM25 (Question or Question + Plan)
    # ======================================================================

    def _extract_plan_text(self, plan: Union[str, dict, list, None]) -> str:
        if not plan:
            return ""
        try:
            plan_data = json.loads(plan) if isinstance(plan, str) else plan
            texts = []
            if isinstance(plan_data, dict):
                steps = plan_data.get("plan", [])
                if isinstance(steps, list):
                    for step in steps:
                        if isinstance(step, dict):
                            desc = step.get("description") or step.get("desc") or step.get("step") or ""
                            if desc:
                                texts.append(str(desc))
                    return " ".join(texts)
            elif isinstance(plan_data, list):
                for step in plan_data:
                    if isinstance(step, dict):
                        desc = step.get("description") or step.get("desc") or step.get("step") or ""
                        if desc:
                            texts.append(str(desc))
                return " ".join(texts)
        except (json.JSONDecodeError, TypeError, AttributeError):
            return str(plan)
        return ""

    def _build_document(self, case: Dict[str, Any]) -> str:
        question = case.get("question", "")
        if self.bm25_doc_type == "question_only":
            return question
        plan_text = self._extract_plan_text(case.get("plan", ""))
        if plan_text:
            return f"{question} {plan_text}"
        return question

    def _tokenize_bm25(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r'[a-z0-9_]+', text.lower())

    def _build_bm25_index(self) -> None:
        if not BM25_AVAILABLE or not self.cases:
            return
        try:
            tokenized_corpus = []
            for c in self.cases:
                doc = self._build_document(c)
                tokenized_corpus.append(self._tokenize_bm25(doc))
            self._tokenized_corpus = tokenized_corpus
            self._bm25_index = BM25Okapi(tokenized_corpus)
            print(f"[CaseBank] BM25 index: {len(self.cases)} docs, doc_type={self.bm25_doc_type}")
        except Exception as e:
            print(f"[CaseBank] BM25 build error: {e}")
            self._bm25_index = None

    def _normalize_bm25_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize BM25 scores using specified method."""
        if len(scores) == 0:
            return scores
        if np.max(scores) == 0:
            return np.zeros_like(scores)
        
        if self.bm25_norm == "softmax":
            exp_scores = np.exp(scores - np.max(scores))
            return exp_scores / np.sum(exp_scores)
        elif self.bm25_norm == "sigmoid":
            mean_score = np.mean(scores)
            if mean_score > 0:
                return 1 / (1 + np.exp(-scores / mean_score))
            return scores
        else:  # "minmax"
            min_score = np.min(scores)
            max_score = np.max(scores)
            if max_score > min_score:
                return (scores - min_score) / (max_score - min_score)
            return scores

    def _get_bm25_scores(self, query: str) -> Optional[np.ndarray]:
        if self._bm25_index is None or not BM25_AVAILABLE:
            return None
        try:
            scores = np.array(self._bm25_index.get_scores(self._tokenize_bm25(query)))
            return self._normalize_bm25_scores(scores)
        except Exception as e:
            print(f"[CaseBank] BM25 error: {e}")
            return None

    # ======================================================================
    # 4. PARAMETRIC RETRIEVER
    # ======================================================================

    def _init_parametric_retriever(self) -> None:
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

    @torch.inference_mode()
    def _get_parametric_scores(self, query: str) -> Optional[np.ndarray]:
        if self._para_model is None or self._para_tokenizer is None:
            return None
        try:
            icl_pool = [build_icl_text(c["question"], c["plan"]) for c in self.cases]
            t1 = self._para_tokenizer(icl_pool, padding=True, truncation=True, return_tensors="pt")
            t2 = self._para_tokenizer([query] * len(icl_pool), padding=True, truncation=True, return_tensors="pt")
            ids1, mask1 = t1["input_ids"].to(self.device), t1["attention_mask"].to(self.device)
            ids2, mask2 = t2["input_ids"].to(self.device), t2["attention_mask"].to(self.device)
            logits = self._para_model(ids1, mask1, ids2, mask2)
            return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        except Exception as e:
            print(f"[CaseBank] Error getting parametric scores: {e}")
            return None

    # ======================================================================
    # 5. MIRA-OSS: INTENT WEIGHTS & RRF
    # ======================================================================

    def _get_intent_weights(self, intent: str) -> Tuple[float, float]:
        """Get BM25 and vector weights based on search intent."""
        weights = {
            "recall": (INTENT_RECALL_BM25, INTENT_RECALL_VECTOR),
            "explore": (INTENT_EXPLORE_BM25, INTENT_EXPLORE_VECTOR),
            "exact": (INTENT_EXACT_BM25, INTENT_EXACT_VECTOR),
            "general": (INTENT_GENERAL_BM25, INTENT_GENERAL_VECTOR),
        }
        return weights.get(intent, weights["general"])

    def _sigmoid_normalize_rrf(self, raw_score: float) -> float:
        """Sigmoid normalization for RRF scores (mira-OSS style)."""
        return 1.0 / (1.0 + math.exp(-RRF_K * 10 * (raw_score - 0.009)))

    def _reciprocal_rank_fusion(
        self,
        dense_scores: Optional[np.ndarray],
        bm25_scores: Optional[np.ndarray],
        candidate_indices: List[int],
        search_intent: str = "general",
    ) -> Dict[int, float]:
        """
        Weighted Reciprocal Rank Fusion with sigmoid normalization.
        Inspired by mira-OSS hybrid_search.py.
        """
        if not candidate_indices:
            return {}
        
        # Get intent weights
        bm25_weight, dense_weight = self._get_intent_weights(search_intent)
        
        # Calculate ranks
        dense_rank = {}
        if dense_scores is not None:
            sorted_idx = sorted(candidate_indices, key=lambda i: dense_scores[i], reverse=True)
            for rank, idx in enumerate(sorted_idx, 1):
                dense_rank[idx] = rank
        
        bm25_rank = {}
        if bm25_scores is not None and self.use_hybrid:
            sorted_idx = sorted(candidate_indices, key=lambda i: bm25_scores[i], reverse=True)
            for rank, idx in enumerate(sorted_idx, 1):
                bm25_rank[idx] = rank
        
        # Weighted RRF scores
        rrf_scores = defaultdict(float)
        for idx in candidate_indices:
            if idx in dense_rank:
                rrf_scores[idx] += dense_weight * (1.0 / (RRF_K + dense_rank[idx]))
            if idx in bm25_rank and self.use_hybrid:
                rrf_scores[idx] += bm25_weight * (1.0 / (RRF_K + bm25_rank[idx]))
        
        # Sigmoid normalization
        return {idx: self._sigmoid_normalize_rrf(score) for idx, score in rrf_scores.items()}

    # ======================================================================
    # 6. UTILITY SCORE (optional)
    # ======================================================================

    def _compute_utility(self, idx: int, retrieval_score: float) -> float:
        if not self.use_utility:
            return retrieval_score
        
        case = self.cases[idx]
        utility = retrieval_score
        
        # Reward bonus
        reward = case.get("reward", 0)
        if self.use_reward_bias:
            utility += 0.1 * reward
        
        # Recency bonus
        timestamp = case.get("timestamp")
        if timestamp:
            try:
                case_time = datetime.fromisoformat(timestamp)
                days_old = (datetime.now() - case_time).days
                recency_bonus = 0.05 * max(0, 1 - days_old / 30)
                utility += recency_bonus
            except Exception:
                pass
        
        # Frequency bonus
        usage_count = case.get("usage_count", 0)
        utility += 0.05 * min(usage_count / 10, 1.0)
        
        return utility

    # ======================================================================
    # 7. MAIN RETRIEVE (MIRA-OSS STYLE)
    # ======================================================================

    def retrieve_cases(
        self,
        query: str,
        top_k: Optional[int] = None,
        search_intent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve Top-K relevant cases using hybrid search.
        
        Flow (mira-OSS style):
        1. Parametric (if available)
        2. Get Dense scores + BM25 scores
        3. Candidate generation (oversample: top_k * 2)
        4. Weighted RRF with sigmoid normalization
        5. Utility scoring (optional)
        6. Return Top-K
        """
        k = top_k if top_k is not None else self.top_k
        intent = search_intent if search_intent is not None else self.default_intent
        
        if not self.cases:
            return []

        # ====== 1. PARAMETRIC (if available) ======
        if TORCH_AVAILABLE and self._para_model is not None:
            try:
                param_scores = self._get_parametric_scores(query)
                if param_scores is not None:
                    top_indices = np.argsort(param_scores)[::-1][:k]
                    return self._format_results(top_indices, None, None, "parametric")
            except Exception as e:
                print(f"[CaseBank] Parametric retrieval error: {e}")

        # ====== 2. GET SCORES ======
        dense_scores = self._get_dense_scores(query)
        bm25_scores = self._get_bm25_scores(query) if self.use_hybrid else None
        
        # ====== 3. CANDIDATE GENERATION (mira-OSS oversample) ======
        candidate_pool = set()
        oversample = HYBRID_OVERSAMPLE_MULTIPLIER
        
        if dense_scores is not None:
            top_dense = np.argsort(dense_scores)[::-1][:k * oversample]
            candidate_pool.update(top_dense)
        
        if bm25_scores is not None and self.use_hybrid:
            top_bm25 = np.argsort(bm25_scores)[::-1][:k * oversample]
            candidate_pool.update(top_bm25)
        
        if not candidate_pool:
            # Fallback: dense-only
            if dense_scores is not None:
                top_indices = np.argsort(dense_scores)[::-1][:k]
                return self._format_results(top_indices, dense_scores, None, "dense_fallback")
            return []
        
        candidate_list = list(candidate_pool)
        
        # ====== 4. WEIGHTED RRF WITH SIGMOID NORMALIZATION ======
        all_scores = self._reciprocal_rank_fusion(
            dense_scores,
            bm25_scores,
            candidate_list,
            search_intent=intent,
        )
        
        # ====== 5. UTILITY SCORING (optional) ======
        if self.use_utility:
            for idx in candidate_list:
                all_scores[idx] = self._compute_utility(idx, all_scores[idx])
        
        # ====== 6. TOP-K ======
        top_indices = sorted(candidate_list, key=lambda i: all_scores[i], reverse=True)[:k]
        
        # ====== 7. FORMAT RESULTS ======
        return self._format_results(top_indices, dense_scores, bm25_scores, f"rrf_{intent}", all_scores)

    def _format_results(
        self,
        top_indices: List[int],
        dense_scores: Optional[np.ndarray],
        bm25_scores: Optional[np.ndarray],
        method: str,
        all_scores: Optional[Dict[int, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Format results with full information."""
        retrieved = []
        for idx in top_indices:
            case = self.cases[idx].copy()
            if all_scores is not None:
                case["score"] = float(all_scores[idx])
            elif dense_scores is not None:
                case["score"] = float(dense_scores[idx])
            else:
                case["score"] = 0.0
            case["retrieval_method"] = method
            
            if dense_scores is not None:
                case["dense_score"] = float(dense_scores[idx])
            if bm25_scores is not None:
                case["bm25_score"] = float(bm25_scores[idx])
            
            # Update usage count
            case["usage_count"] = case.get("usage_count", 0) + 1
            self.cases[idx]["usage_count"] = case["usage_count"]
            
            retrieved.append(case)
        
        # Enhanced logging (mira-OSS style)
        if retrieved:
            print(f"[CaseBank] Retrieved {len(retrieved)} cases (method={method}, candidates={len(top_indices)})")
            for rank, case in enumerate(retrieved[:4], 1):
                print(f"  Rank {rank}: score={case['score']:.4f}, reward={case.get('reward', 0)}")
                print(f"    Q: {case.get('question', '')[:60]}...")
                if 'dense_score' in case:
                    print(f"    Dense: {case['dense_score']:.4f}")
                if 'bm25_score' in case:
                    print(f"    BM25: {case['bm25_score']:.4f}")
        
        return retrieved

    # ======================================================================
    # 8. FORMAT CASES FOR PROMPT
    # ======================================================================

    def format_cases_for_prompt(self, retrieved_cases: List[Dict[str, Any]], max_pos: int = 3, max_neg: int = 3) -> str:
        if not retrieved_cases:
            return "No previous cases found in Case Memory."

        positive_cases = [c for c in retrieved_cases if c.get("reward", 0) == 1]
        negative_cases = [c for c in retrieved_cases if c.get("reward", 0) == 0]

        prompt_parts = []

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

        return "\n".join(prompt_parts) if prompt_parts else "No structured examples found."

    # ======================================================================
    # 9. PARAMETRIC RETRIEVE (kept separate)
    # ======================================================================

    def retrieve_parametric_only(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = top_k if top_k is not None else self.top_k
        if not self.cases or self._para_model is None:
            return []
        try:
            param_scores = self._get_parametric_scores(query)
            if param_scores is None:
                return []
            top_indices = np.argsort(param_scores)[::-1][:k]
            retrieved = []
            for idx in top_indices:
                case = self.cases[idx].copy()
                case["score"] = float(param_scores[idx])
                case["retrieval_method"] = "parametric"
                retrieved.append(case)
            return retrieved
        except Exception as e:
            print(f"[CaseBank] Error in parametric retrieval: {e}")
            return []

    # ======================================================================
    # 10. STATS
    # ======================================================================

    def get_retrieval_stats(self) -> Dict[str, Any]:
        return {
            "use_hybrid": self.use_hybrid,
            "dense_weight": self.dense_weight,
            "bm25_weight": self.bm25_weight,
            "bm25_doc_type": self.bm25_doc_type,
            "bm25_norm": self.bm25_norm,
            "default_intent": self.default_intent,
            "use_reward_bias": self.use_reward_bias,
            "use_utility": self.use_utility,
            "top_k": self.top_k,
            "num_cases": len(self.cases),
            "bm25_available": BM25_AVAILABLE and self._bm25_index is not None,
            "embedding_available": EMBEDDING_AVAILABLE and self._embeddings is not None,
            "parametric_available": TORCH_AVAILABLE and self._para_model is not None,
        }
