"""
AMORE Memory - Adaptive Memory with Online Refinement & Evolution

GIỮ NGUYÊN cấu trúc CaseBank: question, plan, reward
THÊM: Socratic Contradiction Resolution 
THÊM: Conversation Splitting + Chunk Compression

Novelty:
- Learning through contradiction discovery (Socratic method)
- ONE LLM call for contradiction detection and extraction
- Hypothesis vs Evidence comparison
- Fallback: KHÔNG LƯU (chỉ tích lũy buffer)
"""

import os
import json
import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from .processing.conversation_splitter import ConversationSplitter
from .processing.chunk_compressor import ChunkCompressor, ConversationChunk
from .processing.contradiction_resolver import ContradictionResolver

logger = logging.getLogger(__name__)


class AMOREMemory:
    """
    AMORE Memory - CaseBank + Socratic Contradiction Resolution.
    
    GIỮ NGUYÊN:
    - Cấu trúc lưu: question, plan, reward
    - Retrieval: Cosine Similarity
    
    THÊM MỚI:
    - Conversation splitting (detect topic changes)
    - Chunk compression (title + content + raw_conversation)
    - Socratic Contradiction Resolution (hypothesis → contradictions → insights)
    - Write gating: CHỈ LƯU KHI CÓ CONTRADICTIONS
    """
    
    def __init__(
        self,
        memory_jsonl_path: str,
        llm_client,
        llm_provider: str,
        llm_model: str,
        playbook_manager,
        embedding_model,
        top_k: int = 4,
        consolidation_frequency: int = 50,
        device: str = "cpu",
    ):
        self.memory_jsonl_path = memory_jsonl_path
        self.top_k = top_k
        self.consolidation_frequency = consolidation_frequency
        
        # --- CẤU TRÚC CASEBANK (giữ nguyên) ---
        self.cases: List[Dict[str, Any]] = []
        self._emb_model = embedding_model
        self._embeddings = None
        self._corpus_texts = []
        self._retriever = None
        
        # --- PROCESSING MODULES (Socratic Method) ---
        self.splitter = ConversationSplitter(
            llm_client=llm_client,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        
        self.compressor = ChunkCompressor(
            llm_client=llm_client,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        
        self.resolver = ContradictionResolver(
            llm_client=llm_client,
            llm_provider=llm_provider,
            llm_model=llm_model,
            playbook_manager=playbook_manager,
        )
        
        # --- Buffer và chunks ---
        self._buffer: List[Dict[str, Any]] = []
        self._chunks: List[ConversationChunk] = []
        
        # --- Tracking ---
        self.steps_since_consolidation = 0
        self.step_counter = 0
        
        # Load existing cases
        self.load_cases()
    
    def load_cases(self) -> None:
        """Load cases từ JSONL (giống CaseBank cũ)"""
        self.cases = []
        
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
                        self.cases.append(json.loads(line))
                    except Exception as e:
                        print(f"[AMORE] Error loading case: {e}")
            
            print(f"[AMORE] Loaded {len(self.cases)} cases from {self.memory_jsonl_path}")
            self._rebuild_indices()
        except Exception as e:
            print(f"[AMORE] Error loading memory: {e}")
    
    def save(self) -> None:
        """Save cases to JSONL (giống CaseBank cũ)"""
        os.makedirs(os.path.dirname(self.memory_jsonl_path), exist_ok=True)
        
        try:
            with open(self.memory_jsonl_path, "w", encoding="utf-8") as f:
                for case in self.cases:
                    f.write(json.dumps(case, ensure_ascii=False) + "\n")
            print(f"[AMORE] Saved {len(self.cases)} cases")
        except Exception as e:
            print(f"[AMORE] Error saving: {e}")
    
    async def add_case(
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
        playbook: str = "",
    ) -> bool:
        """
        Add case - GIỮ NGUYÊN question, plan, reward + Socratic Contradiction Resolution.
        
        Returns:
            True if stored (with contradictions), False if filtered out
        """
        # --- Step 1: Buffer messages (LƯU CẢ final_answer VÀ plan) ---
        self._buffer.append({"role": "user", "content": question})
        
        # 🔧 Lưu cả final_answer và plan vào buffer
        # Vì plan là JSON string chứa reasoning + final_answer
        assistant_content = plan if plan else final_answer
        self._buffer.append({"role": "assistant", "content": assistant_content})
        
        print(f"[AMORE DEBUG] buffer_size={len(self._buffer)}, step={self.step_counter}")
        
        # --- Step 2: Check if we should split ---
        should_split, _ = await self.splitter.should_split(question, self._buffer)

        if len(self._buffer) >= self.splitter.max_buffer_size:
            should_split = True
            print(f"[AMORE] 🔴 Hard limit reached! buffer_size={len(self._buffer)} (max={self.splitter.max_buffer_size})")
        self.steps_since_consolidation += 1
        if self.steps_since_consolidation >= self.consolidation_frequency:
            should_split = True
            self.steps_since_consolidation = 0
            print(f"[AMORE] 🔄 Consolidation frequency triggered: {self.consolidation_frequency} steps since last consolidation")
        
        # --- Step 3: AMORE Pipeline (chỉ chạy khi đủ điều kiện) ---
        if should_split and len(self._buffer) >= 4:
            print(f"[AMORE] ✅ Pipeline triggered! buffer_size={len(self._buffer)}")
            
            # --- Step 3a: Compress chunk ---
            chunk = await self.compressor.compress(self._buffer)
            self._chunks.append(chunk)
            self._buffer = []  # Reset buffer
            
            # --- Step 3b: Socratic Contradiction Resolution ---
            has_contradiction, insights = await self.resolver.resolve(
                chunk_title=chunk.title,
                raw_conversation=chunk.raw_conversation,
                ground_truth=plan,
            )
            
            if has_contradiction:
                # ✅ LƯU CASE VỚI SOCRATIC INSIGHTS
                case_entry = {
                    # --- Core fields (giống CaseBank cũ) ---
                    "question": chunk.title,
                    "plan": plan,
                    "reward": int(reward),
                    
                    # --- AMORE metadata ---
                    "final_answer": final_answer,
                    "reasoning_trace": reasoning_trace,
                    "bullet_ids_used": bullet_ids_used or [],
                    "error_identification": error_identification,
                    "root_cause": root_cause,
                    "key_insight": key_insight,
                    
                    # --- System fields ---
                    "timestamp": datetime.now().isoformat(),
                    "access_count": 0,
                    "utility_score": 0.0,
                    
                    # --- Chunk fields ---
                    "title": chunk.title,
                    "content": chunk.content,
                    "chunk_content": chunk.content,
                    "raw_conversation": chunk.raw_conversation,
                    
                    # --- Socratic insights (NOVELTY) ---
                    "socratic_insights": insights,
                }
                
                self.cases.append(case_entry)
                self._update_indices(case_entry)
                self.step_counter += 1
                self.save()
                
                print(f"[AMORE] ✅ Stored case with {len(insights)} Socratic insights")
                for insight in insights[:3]:
                    print(f"  - {insight}")
                
                return True
            else:
                # ❌ KHÔNG LƯU (no contradictions)
                print("[AMORE] ⏭️ Skipped: no contradictions found")
                return False
        
        # --- Step 4: FALLBACK - KHÔNG LƯU VÀO CASEBANK ---
        # Chỉ tích lũy buffer
        # Khi đạt hard limit (25 messages), pipeline sẽ tự động chạy
        return False
    
    def retrieve_cases(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve Top-K cases - CHỈ DÙNG COSINE SIMILARITY (giống CaseBank cũ).
        """
        k = top_k or self.top_k
        
        if not self.cases or k <= 0:
            return []
        
        self._ensure_retriever()
        
        # Get query embedding
        query_embedding = None
        if self._emb_model is not None:
            try:
                query_embedding = self._emb_model.encode(
                    [query],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )[0]
            except Exception:
                pass
        
        results = self._retriever.retrieve(
            query=query,
            query_embedding=query_embedding,
            k=k,
        )
        
        final_cases = []
        for item in results:
            idx = item["idx"]
            if idx < len(self.cases):
                case = self.cases[idx].copy()
                case["access_count"] = case.get("access_count", 0) + 1
                case["score"] = item.get("score", 0.0)
                final_cases.append(case)
        
        return final_cases
    
    def format_cases_for_prompt(
        self,
        retrieved_cases: List[Dict[str, Any]],
        max_pos: int = 3,
        max_neg: int = 3
    ) -> str:
        """Format retrieved cases (GIỮ NGUYÊN format)."""
        if not retrieved_cases:
            return "No previous cases found in Memory."
        
        positive_cases = [c for c in retrieved_cases if c.get("reward", 0) == 1]
        negative_cases = [c for c in retrieved_cases if c.get("reward", 0) == 0]
        
        prompt_parts = []
        
        if positive_cases:
            prompt_parts.append(f"### Successful Examples (reward=1):")
            for i, case in enumerate(positive_cases[:max_pos], 1):
                prompt_parts.append(
                    f"Example {i}:\n"
                    f"Question: {case['question']}\n"
                    f"Plan:\n{case['plan']}\n"
                )
                if case.get('key_insight'):
                    prompt_parts.append(f"Key Insight: {case['key_insight']}\n")
        
        if negative_cases:
            prompt_parts.append(f"### Unsuccessful Examples (reward=0):")
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
    
    # ==================== HELPER METHODS ====================
    
    def _rebuild_indices(self):
        """Rebuild embeddings for retrieval."""
        if not self.cases:
            self._embeddings = None
            self._corpus_texts = []
            self._retriever = None
            return
        
        self._corpus_texts = [c["question"] for c in self.cases]
        
        if self._emb_model is not None:
            try:
                self._embeddings = self._emb_model.encode(
                    self._corpus_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                self._retriever = None
            except Exception as e:
                print(f"[AMORE] Error encoding: {e}")
                self._embeddings = None
    
    def _update_indices(self, case_entry):
        """Update indices with new case."""
        self._corpus_texts.append(case_entry["question"])
        
        if self._emb_model is not None and self._embeddings is not None:
            try:
                emb = self._emb_model.encode(
                    [case_entry["question"]],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )[0]
                self._embeddings = np.vstack([self._embeddings, emb])
            except Exception:
                self._rebuild_indices()
        
        self._retriever = None
    
    def _ensure_retriever(self):
        """Ensure retriever is built."""
        if self._retriever is None and self.cases:
            from .adaptive_retriever import AdaptiveRetriever
            metadata_list = [
                {
                    "reward": c.get("reward", 0),
                    "error_identification": c.get("error_identification", ""),
                }
                for c in self.cases
            ]
            self._retriever = AdaptiveRetriever(
                embedding_model=self._emb_model,
                embedding_vectors=self._embeddings,
                corpus_texts=self._corpus_texts,
                metadata_list=metadata_list,
                top_k=self.top_k
            )
