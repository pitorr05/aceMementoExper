"""
FailureMemoryBank for ACE System

Analogical Reflection: stores past failure episodes and retrieves the most
similar ones so the Reflector can reason by analogy — "we've seen this kind
of mistake before, here is what went wrong and why."

Design principle — shared embedding model:
    When RAE (PlaybookRetriever) is enabled the SentenceTransformer is already
    loaded.  FailureMemoryBank accepts an optional `encoder` callable so it can
    delegate to PlaybookRetriever.encode() and avoid loading a second copy of
    the (large) BAAI/bge-m3 model.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Callable

try:
    import faiss
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("Warning: faiss not available for FailureMemoryBank.")
    print("Install with: pip install faiss-cpu")


class FailureMemoryBank:
    """
    Episodic memory of past failure cases for analogical reflection.

    Each stored entry captures one failure episode:
        - question            (used as the semantic retrieval key)
        - predicted_answer
        - ground_truth
        - error_identification  (from the Reflector)
        - root_cause            (from the Reflector)
        - key_insight           (from the Reflector)

    At reflection time the top-K entries most similar to the current question
    are retrieved and injected into the Reflector prompt so the LLM can draw
    analogies with previously seen mistakes.

    Shared embedding model:
        Pass `encoder=playbook_retriever.encode` to reuse the BGE-M3 model
        that is already loaded for RAE.  When `encoder` is None the bank
        lazy-loads its own SentenceTransformer (only if available).
    """

    def __init__(
        self,
        encoder: Optional[Callable[[List[str]], np.ndarray]] = None,
        embedding_dim: int = 1024,
        top_k: int = 3,
        embedding_model_name: str = 'BAAI/bge-m3',
    ):
        """
        Args:
            encoder: A callable that accepts a list of strings and returns a
                     float32 numpy array of shape (N, embedding_dim).
                     Pass ``playbook_retriever.encode`` to share the BGE-M3
                     model that is already loaded for RAE.
                     When None the bank lazy-loads its own
                     SentenceTransformer (requires sentence-transformers).
            embedding_dim: Dimensionality of the embedding vectors.
                           Must match the model (BGE-M3 → 1024).
            top_k: Default number of similar failures to retrieve.
            embedding_model_name: HuggingFace model id used **only** when
                                   `encoder` is None and a standalone model
                                   must be loaded.
        """
        self.embedding_dim = embedding_dim
        self.default_top_k = top_k
        self.embedding_model_name = embedding_model_name

        # Injected or standalone encoder
        self._external_encoder: Optional[Callable] = encoder
        self._standalone_model: Optional[Any] = None  # lazy-loaded fallback

        # FAISS flat inner-product index + raw entries
        self._index: Optional[Any] = None
        self._entries: List[Dict[str, Any]] = []

        if not MEMORY_AVAILABLE:
            print(
                "⚠️  FailureMemoryBank: faiss not available — "
                "analogical reflection disabled."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_standalone_model(self) -> None:
        """Lazy-load a standalone SentenceTransformer when no encoder was injected."""
        if self._standalone_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            print(
                f"[FailureMemory] Loading standalone embedding model: "
                f"{self.embedding_model_name}"
            )
            self._standalone_model = SentenceTransformer(self.embedding_model_name)
        except ImportError:
            print(
                "⚠️  FailureMemoryBank: sentence-transformers not available. "
                "Analogical reflection disabled."
            )

    def _encode(self, texts: List[str]) -> Optional[np.ndarray]:
        """Return L2-normalised float32 embeddings, or None if unavailable."""
        if self._external_encoder is not None:
            return self._external_encoder(texts)
        # Fallback: standalone model
        if not MEMORY_AVAILABLE:
            return None
        self._load_standalone_model()
        if self._standalone_model is None:
            return None
        return self._standalone_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from all stored embeddings."""
        if not MEMORY_AVAILABLE or not self._entries:
            return
        embeddings = np.stack([e['_emb'] for e in self._entries])
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings)
        self._index = index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        question: str,
        predicted_answer: str,
        ground_truth: str,
        error_identification: str = "",
        root_cause: str = "",
        key_insight: str = "",
    ) -> None:
        """
        Store a failure episode in the memory bank.

        Should be called *after* reflection so that the distilled insights
        (error_identification, root_cause, key_insight) are available.
        Only stores episodes where the model was incorrect.
        """
        emb = self._encode([question])
        if emb is None:
            return  # encoding unavailable; silently skip

        entry: Dict[str, Any] = {
            'question': question,
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'error_identification': error_identification,
            'root_cause': root_cause,
            'key_insight': key_insight,
            '_emb': emb[0],  # shape (dim,)
        }
        self._entries.append(entry)
        self._rebuild_index()
        print(f"[FailureMemory] Stored failure #{self.size} | "
              f"q='{question[:80].strip()}...' | bank_size={self.size}")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-K most similar past failures for `query`.

        Returns an empty list when the bank is empty or unavailable.
        Each result dict contains public fields (no '_emb') plus:
            - 'similarity'  cosine similarity score
            - 'rank'        1-based retrieval rank
        """
        if not MEMORY_AVAILABLE or self._index is None or not self._entries:
            return []

        k = min(top_k if top_k is not None else self.default_top_k, len(self._entries))

        query_emb = self._encode([query])
        if query_emb is None:
            return []

        scores, indices = self._index.search(query_emb, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            entry = {key: val for key, val in self._entries[idx].items()
                     if not key.startswith('_')}
            entry['similarity'] = float(score)
            entry['rank'] = rank + 1
            results.append(entry)
        if results:
            top = results[0]
            print(f"[FailureMemory] Retrieved {len(results)} similar failure(s) "
                  f"(bank_size={self.size}) | "
                  f"top similarity={top['similarity']:.3f} | "
                  f"top q='{top['question'][:60].strip()}...'")
        return results

    @staticmethod
    def format_for_prompt(similar_failures: List[Dict[str, Any]]) -> str:
        """
        Render retrieved failures as a human-readable block for the
        Reflector prompt.
        """
        if not similar_failures:
            return "(No similar past failures found)"

        lines: List[str] = []
        for f in similar_failures:
            lines.append(
                f"--- Similar Failure #{f['rank']} "
                f"(similarity={f['similarity']:.3f}) ---"
            )
            lines.append(f"Question: {f['question']}")
            lines.append(f"Wrong Answer: {f['predicted_answer']}")
            if f.get('ground_truth'):
                lines.append(f"Ground Truth: {f['ground_truth']}")
            if f.get('error_identification'):
                lines.append(f"Error: {f['error_identification']}")
            if f.get('root_cause'):
                lines.append(f"Root Cause: {f['root_cause']}")
            if f.get('key_insight'):
                lines.append(f"Key Insight: {f['key_insight']}")
            lines.append("")
        return '\n'.join(lines).strip()

    @property
    def size(self) -> int:
        """Number of failure episodes stored."""
        return len(self._entries)
