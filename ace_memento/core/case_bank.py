import os
import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union

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


# Global registry for shared embedding models to avoid redundant loads and save RAM/VRAM
_SHARED_MODELS = {}


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
    Supports both:
      - Non-parametric retrieval (SentenceTransformer/Faiss cosine similarity)
      - Parametric retrieval (MemoryRetrieverClassifier neural model)
    """

    def __init__(
        self,
        memory_jsonl_path: str,
        top_k: int = 4,
        embedding_model_name: str = "BAAI/bge-m3",
        parametric_model_name: str = "princeton-nlp/sup-simcse-roberta-base",
        retriever_model_path: Optional[str] = None,
        device: str = "cpu"
    ):
        self.memory_jsonl_path = memory_jsonl_path
        self.top_k = top_k
        self.embedding_model_name = embedding_model_name
        self.parametric_model_name = parametric_model_name
        self.retriever_model_path = retriever_model_path
        self.device = device if device != "auto" else ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.cases: List[Dict[str, Any]] = []

        # Lazy loaded components
        self._emb_model = None
        self._embeddings = None

        # Parametric retriever models
        self._para_tokenizer = None
        self._para_model = None

        # Load cases from JSONL
        self.load_cases()
        self._init_parametric_retriever()

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
                        self.cases.append(json.loads(line))
                    except Exception:
                        pass
            print(f"[CaseBank] Loaded {len(self.cases)} cases from {self.memory_jsonl_path}")
            self._rebuild_embeddings()
        except Exception as e:
            print(f"[CaseBank] Error loading cases: {e}")

    def add_case(self, question: str, plan: str, reward: int) -> None:
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
            self._append_embedding(question)
        except Exception as e:
            print(f"[CaseBank] Error writing case: {e}")

    def _init_parametric_retriever(self) -> None:
        """Initialize neural parametric classifier retriever if check-point is provided."""
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
            # Fall back to rebuilding if anything goes wrong
            self._rebuild_embeddings()

    def _rebuild_embeddings(self) -> None:
        if not EMBEDDING_AVAILABLE or not self.cases or self.retriever_model_path:
            # Skip if parametric mode is active
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

    @torch.inference_mode()
    def _score_batch(self, query: str, icl_pool: List[str]) -> List[float]:
        """Compute classifier probability scores using parametric neural retriever."""
        t1 = self._para_tokenizer(icl_pool, padding=True, truncation=True, return_tensors="pt")
        t2 = self._para_tokenizer([query] * len(icl_pool), padding=True, truncation=True, return_tensors="pt")
        
        ids1 = t1["input_ids"].to(self.device)
        mask1 = t1["attention_mask"].to(self.device)
        ids2 = t2["input_ids"].to(self.device)
        mask2 = t2["attention_mask"].to(self.device)
        
        logits = self._para_model(ids1, mask1, ids2, mask2)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        return probs

    def retrieve_cases(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve Top-K relevant cases for the query."""
        k = top_k if top_k is not None else self.top_k
        if not self.cases:
            return []

        # 1. Use parametric neural model if available
        if TORCH_AVAILABLE and self._para_model is not None:
            try:
                icl_pool = [build_icl_text(c["question"], c["plan"]) for c in self.cases]
                probs = self._score_batch(query, icl_pool)
                
                results = []
                for i, (case, score) in enumerate(zip(self.cases, probs)):
                    case_copy = case.copy()
                    case_copy["score"] = score
                    results.append(case_copy)
                
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:k]
            except Exception as e:
                print(f"[CaseBank] Error running parametric retrieval: {e}")

        # 2. Fall back to non-parametric embedding retrieval
        if self._embeddings is not None and EMBEDDING_AVAILABLE:
            try:
                self._load_emb_model()
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
            except Exception as e:
                print(f"[CaseBank] Error in non-parametric retrieval: {e}")

        # 3. Simple keyword word-overlap matching fallback
        results = []
        query_words = set(query.lower().split())
        for idx, c in enumerate(self.cases):
            q_words = set(c["question"].lower().split())
            overlap = len(query_words.intersection(q_words))
            results.append((overlap, idx))
        results.sort(key=lambda x: x[0], reverse=True)
        ret_indices = [idx for score, idx in results[:k]]
        return [self.cases[idx] for idx in ret_indices]

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
