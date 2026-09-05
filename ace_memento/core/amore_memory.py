"""
AMORE: Adaptive Memory with Online Refinement & Evolution
Main class that replaces CaseBank
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from .adaptive_memory_entry import AdaptiveMemoryEntry
from .adaptive_retriever import AdaptiveRetriever
from .predict_calibrate_gate import PredictCalibrateGate
from .memory_consolidator import MemoryConsolidator

# Try to import BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank-bm25 not installed. BM25 search will be disabled.")

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("Warning: sentence-transformers not available. Vector search will be disabled.")

# Shared model registry
_SHARED_MODELS = {}

class AMOREMemory:
    """
    Adaptive Memory with Online Refinement & Evolution
    
    A complete memory system that replaces CaseBank with:
    1. Rich memory entries (P, S, O, M)
    2. Predict-Calibrate write gating
    3. Adaptive retrieval with multi-granular filtering
    4. Memory lifecycle with consolidation
    """
    
    def __init__(
        self,
        memory_jsonl_path: str,
        api_client: Any,
        api_provider: str,
        generator_model: str,
        top_k: int = 4,
        embedding_model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        consolidation_frequency: int = 50,
        utility_threshold: float = 0.15,
        uncertainty_threshold: float = 0.3
    ):
        self.memory_jsonl_path = memory_jsonl_path
        self.top_k = top_k
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.consolidation_frequency = consolidation_frequency
        
        # Store entries
        self.entries: List[AdaptiveMemoryEntry] = []
        self._idx_to_entry: Dict[int, AdaptiveMemoryEntry] = {}
        
        # Embedding model
        self._emb_model = None
        self._embeddings = None  # numpy array
        
        # BM25 index
        self._corpus_texts = []
        self._bm25_index = None
        
        # Retrieval
        self._retriever = None
        
        # Write gate
        self.predict_gate = PredictCalibrateGate(
            api_client=api_client,
            api_provider=api_provider,
            model=generator_model,
            uncertainty_threshold=uncertainty_threshold
        )
        
        # Consolidator
        self.consolidator = MemoryConsolidator(
            utility_threshold=utility_threshold
        )
        
        # Tracking
        self.steps_since_consolidation = 0
        self.step_counter = 0
        
        # Load existing entries
        self.load()
    
    def load(self) -> None:
        """Load memory from JSONL file"""
        self.entries = []
        
        if not os.path.exists(self.memory_jsonl_path):
            print(f"[AMORE] Memory file not found: {self.memory_jsonl_path}")
            os.makedirs(os.path.dirname(self.memory_jsonl_path), exist_ok=True)
            return
        
        try:
            with open(self.memory_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = AdaptiveMemoryEntry.from_dict(data)
                        self.entries.append(entry)
                    except Exception as e:
                        print(f"[AMORE] Error loading entry: {e}")
            
            print(f"[AMORE] Loaded {len(self.entries)} entries from {self.memory_jsonl_path}")
            self._rebuild_indices()
        except Exception as e:
            print(f"[AMORE] Error loading memory: {e}")
    def load_cases(self) -> None:
        self.load()
    
    def save(self) -> None:
        """Save memory to JSONL file"""
        os.makedirs(os.path.dirname(self.memory_jsonl_path), exist_ok=True)
        
        try:
            with open(self.memory_jsonl_path, "w", encoding="utf-8") as f:
                for entry in self.entries:
                    f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            print(f"[AMORE] Saved {len(self.entries)} entries to {self.memory_jsonl_path}")
        except Exception as e:
            print(f"[AMORE] Error saving memory: {e}")
    
    def add_case(
        self,
        question: str,
        plan: str,
        reward: int,
        final_answer: str = "",
        reasoning_trace: str = "",
        bullet_ids_used: List[str] = None,
        error_identification: str = "",
        root_cause: str = "",
        key_insight: str = "",
        playbook: str = ""
    ) -> bool:
        """
        Add a new case with Predict-Calibrate gating.
        
        Returns:
            True if stored, False if filtered out
        """
        # --- Step 1: Predict-Calibrate Gate ---
        should_store, confidence, reasoning = self.predict_gate.should_store(
            question=question,
            predicted_answer=final_answer,
            ground_truth=plan if reward == 1 else "INCORRECT",
            current_playbook=playbook,
            confidence_score=1.0 if reward == 1 else 0.0,
            call_id=f"gate_{self.step_counter}"
        )
        
        if not should_store:
            print(f"[AMORE] Skipped storage: {reasoning}")
            return False
        
        # --- Step 2: Create entry with rich metadata ---
        # Get embedding if available
        question_embedding = None
        if EMBEDDING_AVAILABLE and self._emb_model is not None:
            try:
                question_embedding = self._emb_model.encode(
                    [question],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )[0]
            except Exception as e:
                print(f"[AMORE] Error encoding question: {e}")
        
        entry = AdaptiveMemoryEntry(
            question=question,
            question_embedding=question_embedding,
            plan=plan,
            bullet_ids_used=bullet_ids_used or [],
            reasoning_trace=reasoning_trace,
            final_answer=final_answer,
            reward=reward,
            confidence_score=confidence,
            error_identification=error_identification,
            root_cause=root_cause,
            key_insight=key_insight,
            task_type=self._detect_task_type(question),
            keywords=self._extract_keywords(question),
        )
        
        # --- Step 3: Store ---
        self.entries.append(entry)
        self._update_indices(entry)
        self.step_counter += 1
        
        print(f"[AMORE] Stored entry (reward={reward}, confidence={confidence:.2f}): {reasoning}")
        
        # --- Step 4: Auto-consolidate if needed ---
        self.steps_since_consolidation += 1
        if self.steps_since_consolidation >= self.consolidation_frequency:
            self.consolidate()
            self.steps_since_consolidation = 0
        
        self.save()
        return True
    
    def retrieve_cases(
        self,
        query: str,
        top_k: Optional[int] = None,
        round_idx: int = 0,
        previous_error: Optional[str] = None,
        intent: str = "general"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-K relevant cases.
        
        Args:
            query: User query
            top_k: Number of cases to return
            round_idx: Reflection round (0 = initial)
            previous_error: Error from previous round
            intent: Search intent (exact, recall, general, explore)
        
        Returns:
            List of cases with scores
        """
        k = top_k or self.top_k
        
        if not self.entries or k <= 0:
            return []
        
        # Build retriever if not exists
        self._ensure_retriever()
        
        # Get query embedding
        query_embedding = None
        if EMBEDDING_AVAILABLE and self._emb_model is not None:
            try:
                query_embedding = self._emb_model.encode(
                    [query],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )[0]
            except Exception:
                pass
        
        # Retrieve
        results = self._retriever.retrieve(
            query=query,
            query_embedding=query_embedding,
            round_idx=round_idx,
            previous_error=previous_error,
            k=k,
            intent=intent
        )
        
        # Format results
        final_cases = []
        for item in results:
            idx = item["idx"]
            if idx < len(self.entries):
                entry = self.entries[idx]
                entry.update_access()
                
                case = {
                    "question": entry.question,
                    "plan": entry.plan,
                    "reward": entry.reward,
                    "score": item.get("combined_score", 0.0),
                    "error_identification": entry.error_identification,
                    "key_insight": entry.key_insight,
                    "access_count": entry.access_count,
                }
                final_cases.append(case)
        
        # Update access stats and save periodically
        if final_cases:
            self.save()
        
        return final_cases
    
    def consolidate(self) -> int:
        """
        Run memory consolidation: pruning + abstraction.
        
        Returns:
            Number of entries removed
        """
        old_count = len(self.entries)
        
        # Convert entries to dicts for consolidator
        entry_dicts = [entry.to_dict() for entry in self.entries]
        
        # Run consolidation
        consolidated_dicts = self.consolidator.consolidate(
            entry_dicts,
            self._embeddings
        )
        
        # Convert back to entries
        new_entries = []
        for data in consolidated_dicts:
            # Check if this is an abstracted entry
            if data.get("is_abstracted", False):
                # Keep the original entry (for now)
                # In a more advanced version, we'd create a prototype
                pass
            new_entries.append(AdaptiveMemoryEntry.from_dict(data))
        
        # Update
        removed = old_count - len(new_entries)
        self.entries = new_entries
        
        if removed > 0:
            self._rebuild_indices()
            self.save()
            print(f"[AMORE] Consolidated: {old_count} → {len(new_entries)} entries (removed {removed})")
        
        return removed
    
    def _rebuild_indices(self) -> None:
        """Rebuild BM25 and vector indices"""
        if not self.entries:
            self._corpus_texts = []
            self._embeddings = None
            self._bm25_index = None
            self._retriever = None
            return
        
        # Corpus texts for BM25
        self._corpus_texts = [entry.question for entry in self.entries]
        
        # BM25
        if BM25_AVAILABLE:
            try:
                tokenized_corpus = [doc.split(" ") for doc in self._corpus_texts]
                self._bm25_index = BM25Okapi(tokenized_corpus)
            except Exception as e:
                print(f"[AMORE] Error building BM25: {e}")
                self._bm25_index = None
        else:
            self._bm25_index = None
        
        # Vector embeddings
        if EMBEDDING_AVAILABLE:
            self._load_emb_model()
            if self._emb_model is not None:
                try:
                    self._embeddings = self._emb_model.encode(
                        self._corpus_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                except Exception as e:
                    print(f"[AMORE] Error encoding: {e}")
                    self._embeddings = None
        
        # Rebuild retriever
        self._retriever = None
    
    def _update_indices(self, entry: AdaptiveMemoryEntry) -> None:
        """Update indices with new entry"""
        # Update corpus
        self._corpus_texts.append(entry.question)
        
        # Update BM25 (rebuild)
        if BM25_AVAILABLE and self._bm25_index is not None:
            try:
                tokenized_corpus = [doc.split(" ") for doc in self._corpus_texts]
                self._bm25_index = BM25Okapi(tokenized_corpus)
            except Exception:
                self._bm25_index = None
        
        # Update embeddings
        if EMBEDDING_AVAILABLE and self._emb_model is not None:
            if entry.question_embedding is not None:
                if self._embeddings is None or len(self._embeddings) == 0:
                    self._embeddings = np.array([entry.question_embedding])
                else:
                    self._embeddings = np.vstack([self._embeddings, entry.question_embedding])
            else:
                # Rebuild all if no embedding available
                self._rebuild_indices()
        
        # Invalidate retriever
        self._retriever = None
    
    def _ensure_retriever(self) -> None:
        """Ensure retriever is built"""
        if self._retriever is None and self.entries:
            metadata_list = []
            for entry in self.entries:
                metadata_list.append({
                    "task_type": entry.task_type,
                    "reward": entry.reward,
                    "error_identification": entry.error_identification,
                })
            
            self._retriever = AdaptiveRetriever(
                embedding_model=self._emb_model,
                embedding_vectors=self._embeddings,
                corpus_texts=self._corpus_texts,
                metadata_list=metadata_list,
                top_k=self.top_k
            )
            self._retriever.set_bm25_index(self._bm25_index)
    
    def _load_emb_model(self) -> None:
        """Load shared embedding model"""
        if self._emb_model is None and EMBEDDING_AVAILABLE:
            try:
                key = (self.embedding_model_name, self.device)
                if key not in _SHARED_MODELS:
                    print(f"[AMORE] Loading embedding model: {self.embedding_model_name}")
                    _SHARED_MODELS[key] = SentenceTransformer(
                        self.embedding_model_name,
                        device=self.device
                    )
                self._emb_model = _SHARED_MODELS[key]
            except Exception as e:
                print(f"[AMORE] Error loading embedding model: {e}")
                self._emb_model = None
    
    def _detect_task_type(self, question: str) -> str:
        """Detect task type from question"""
        question_lower = question.lower()
        if "formula" in question_lower or "calculate" in question_lower:
            return "formula"
        elif "tag" in question_lower or "us gaap" in question_lower:
            return "finer"
        elif "research" in question_lower or "deep" in question_lower:
            return "deepresearcher"
        return "general"
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract keywords from question"""
        # Simple extraction - can be improved
        words = question.lower().split()
        # Filter common words
        stopwords = {"what", "is", "the", "for", "of", "to", "in", "on", "at", "by"}
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        return keywords[:5]  # Limit to top 5
    
    def format_cases_for_prompt(
        self,
        retrieved_cases: List[Dict[str, Any]],
        max_pos: int = 3,
        max_neg: int = 3
    ) -> str:
        """Format retrieved cases for prompt"""
        if not retrieved_cases:
            return "No previous cases found in Memory."
        
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
                if case.get('key_insight'):
                    prompt_parts.append(f"Key Insight: {case['key_insight']}\n")
        
        if negative_cases:
            prompt_parts.append(f"### Unsuccessful Examples (reward=0) - Showing up to {max_neg}:")
            for i, case in enumerate(negative_cases[:max_neg], 1):
                prompt_parts.append(
                    f"Example {i}:\n"
                    f"Question: {case['question']}\n"
                    f"Plan:\n{case['plan']}\n"
                )
                if case.get('error_identification'):
                    prompt_parts.append(f"Error: {case['error_identification']}\n")
        
        if not prompt_parts:
            return "No structured examples found in Memory."
        
        return "\n".join(prompt_parts)
