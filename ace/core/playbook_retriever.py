"""
PlaybookRetriever Component for ACE System

Retrieval-Augmented Execution (RAE): At generation time, instead of passing
the entire playbook to the LLM, this component retrieves only the Top-K most
relevant bullets using semantic similarity search (BGE-M3 + FAISS).
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    RAE_AVAILABLE = True
except ImportError:
    RAE_AVAILABLE = False
    print("Warning: sentence-transformers or faiss not available for RAE.")
    print("Install with: pip install sentence-transformers faiss-cpu")


# ---------------------------------------------------------------------------
# Reuse the bullet parser from bulletpoint_analyzer (avoid circular import by
# duplicating the tiny regex here rather than importing cross-module).
# ---------------------------------------------------------------------------

_BULLET_PATTERN = re.compile(
    r'\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)'
)
_HEADER_PATTERN = re.compile(r'^##?\s+')


def _parse_bullet(line: str) -> Optional[Dict[str, Any]]:
    """Parse a playbook bullet line into a dict. Returns None for headers/blanks."""
    line = line.strip()
    if not line or _HEADER_PATTERN.match(line):
        return None
    m = _BULLET_PATTERN.match(line)
    if m:
        bullet_id, helpful, harmful, content = m.groups()
        return {
            'id': bullet_id,
            'helpful': int(helpful),
            'harmful': int(harmful),
            'content': content.strip(),
            'original_line': line,
        }
    # Fallback: lines containing '::' but no ID header
    if '::' in line:
        parts = line.split('::', 1)
        return {
            'id': f'unknown-{abs(hash(line)) % 10000:04d}',
            'helpful': 0,
            'harmful': 0,
            'content': parts[1].strip(),
            'original_line': line,
        }
    return None


class PlaybookRetriever:
    """
    Semantic retriever for playbook bullet points.

    Usage:
        retriever = PlaybookRetriever(top_k=10)
        retriever.update_index(playbook_string)   # call after each curator step
        focused_playbook = retriever.retrieve(query)
        # pass focused_playbook to Generator instead of the full playbook
    """

    def __init__(
        self,
        embedding_model_name: str = 'BAAI/bge-m3',
        embedding_dim: int = 1024,
        top_k: int = 10,
    ):
        """
        Initialize the PlaybookRetriever.

        Args:
            embedding_model_name: HuggingFace model id for sentence-transformers.
                                  Defaults to BAAI/bge-m3 (multilingual, dim=1024).
            embedding_dim: Dimensionality of the embedding vectors.
                           Must match the chosen model (bge-m3 → 1024).
            top_k: Default number of bullets to retrieve.
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = embedding_dim
        self.default_top_k = top_k

        # Lazy-loaded embedding model
        self._embedding_model: Optional[Any] = None

        # FAISS index + metadata — populated by update_index()
        self._index: Optional[Any] = None
        self._bullets: List[Dict[str, Any]] = []
        self._section_headers: List[str] = []   # kept to wrap output playbook
        self._raw_playbook: str = ""

        if not RAE_AVAILABLE:
            print("⚠️  PlaybookRetriever initialized but dependencies not available — "
                  "will fall back to full playbook.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazily load the embedding model on first use."""
        if self._embedding_model is None and RAE_AVAILABLE:
            print(f"[RAE] Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to L2-normalized float32 embeddings."""
        self._load_model()
        embeddings = self._embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # bge-m3 recommends normalizing for cosine
            show_progress_bar=False,
        ).astype(np.float32)
        return embeddings

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Public encode method — allows other components (e.g. FailureMemoryBank)
        to share this retriever's embedding model without loading a second copy.
        """
        return self._encode(texts)

    @property
    def embedding_model(self):
        """The underlying SentenceTransformer instance (lazily loaded)."""
        self._load_model()
        return self._embedding_model

    def _parse_playbook(self, playbook: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Parse playbook into section header lines and bullet dicts.

        Returns:
            (header_lines, bullets) where header_lines are '##' lines
            and bullets have keys: id, helpful, harmful, content, original_line.
        """
        header_lines = []
        bullets = []
        for line in playbook.split('\n'):
            if _HEADER_PATTERN.match(line.strip()):
                header_lines.append(line)
            else:
                parsed = _parse_bullet(line)
                if parsed:
                    bullets.append(parsed)
        return header_lines, bullets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_index(self, playbook: str) -> None:
        """
        (Re)build the FAISS index from the current playbook.

        Call this every time the Curator updates the playbook so that
        subsequent retrieve() calls reflect the latest bullets.

        Args:
            playbook: Full playbook string (may have hundreds of bullets).
        """
        if not RAE_AVAILABLE:
            self._raw_playbook = playbook
            return

        self._raw_playbook = playbook
        self._section_headers, self._bullets = self._parse_playbook(playbook)

        if len(self._bullets) == 0:
            self._index = None
            return

        contents = [b['content'] for b in self._bullets]
        embeddings = self._encode(contents)  # (N, dim)

        # Build an inner-product index (cosine sim after L2 norm)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings)
        self._index = index

        print(f"[RAE] Index built: {len(self._bullets)} bullets indexed "
              f"(model={self.embedding_model_name}, dim={self.embedding_dim})")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Retrieve the Top-K most relevant bullets for the given query.

        Falls back to the full playbook when:
        - Dependencies are not available.
        - Index has not been built yet (update_index not called).
        - Playbook has fewer bullets than top_k.

        Args:
            query: The task description / question to use as retrieval query.
            top_k: Override the default top_k. Uses self.default_top_k if None.

        Returns:
            A playbook-formatted string containing only the retrieved bullets,
            with section headers preserved for context.
        """
        k = top_k if top_k is not None else self.default_top_k

        # Fallback conditions
        if not RAE_AVAILABLE or self._index is None or len(self._bullets) == 0:
            return self._raw_playbook

        num_bullets = len(self._bullets)
        if num_bullets <= k:
            # No need to filter — return full playbook as-is
            return self._raw_playbook

        # Encode query
        query_emb = self._encode([query])  # (1, dim)

        # Search
        actual_k = min(k, num_bullets)
        scores, indices = self._index.search(query_emb, actual_k)
        retrieved_indices = set(indices[0].tolist())

        print(f"[RAE] Retrieved {len(retrieved_indices)}/{num_bullets} bullets "
              f"(top_k={k}) | top score={scores[0][0]:.4f}")

        # Rebuild a focused playbook string:
        # Keep all section headers + only retrieved bullets
        focused_lines: List[str] = []
        bullet_rank = {idx: rank for rank, idx in enumerate(indices[0].tolist())}

        # We want to preserve section structure, so walk the original playbook
        bullet_global_idx = 0
        for line in self._raw_playbook.split('\n'):
            stripped = line.strip()
            if _HEADER_PATTERN.match(stripped):
                focused_lines.append(line)
                continue
            parsed = _parse_bullet(line)
            if parsed:
                if bullet_global_idx in retrieved_indices:
                    focused_lines.append(line)
                bullet_global_idx += 1
            else:
                # blank lines / misc — keep for readability
                focused_lines.append(line)

        return '\n'.join(focused_lines)

    @property
    def is_available(self) -> bool:
        """True if RAE dependencies are installed and index has been built."""
        return RAE_AVAILABLE and self._index is not None

    @property
    def num_bullets(self) -> int:
        """Number of bullets currently indexed."""
        return len(self._bullets)
