import re
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False


class PlaybookManager:
    """
    Manages Playbook loading, parsing, and Retrieval-Augmented Execution (RAE).
    """

    BULLET_PATTERN = re.compile(
        r'\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)'
    )
    HEADER_PATTERN = re.compile(r'^##?\s+')

    def __init__(
        self,
        initial_playbook: Optional[str] = None,
        embedding_model_name: str = "BAAI/bge-m3",
        device: str = "cpu"
    ):
        self.embedding_model_name = embedding_model_name
        self.device = device
        self._model = None
        self._bullet_embeddings = None

        if initial_playbook:
            self.playbook = initial_playbook
        else:
            self.playbook = self._empty_playbook()

        self.section_headers, self.bullets = self.parse_playbook(self.playbook)

    def _empty_playbook(self) -> str:
        return """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""

    def parse_playbook(self, playbook_text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Parse playbook string into headers and parsed bullet dicts."""
        headers = []
        bullets = []
        
        for line in playbook_text.split('\n'):
            line_str = line.strip()
            if not line_str:
                continue
            if self.HEADER_PATTERN.match(line_str):
                headers.append(line_str)
            else:
                parsed = self._parse_line(line_str)
                if parsed:
                    bullets.append(parsed)
        return headers, bullets

    def _parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        m = self.BULLET_PATTERN.match(line)
        if m:
            bullet_id, helpful, harmful, content = m.groups()
            return {
                'id': bullet_id,
                'helpful': int(helpful),
                'harmful': int(harmful),
                'content': content.strip(),
                'original_line': line,
            }
        if '::' in line:
            parts = line.split('::', 1)
            return {
                'id': f'misc-{abs(hash(line)) % 10000:04d}',
                'helpful': 0,
                'harmful': 0,
                'content': parts[1].strip(),
                'original_line': line,
            }
        return None

    def _load_model(self) -> None:
        if self._model is None and EMBEDDING_AVAILABLE:
            try:
                from .case_bank import _SHARED_MODELS
                key = (self.embedding_model_name, self.device)
                if key not in _SHARED_MODELS:
                    print(f"[PlaybookManager] Loading shared model: {self.embedding_model_name} on {self.device}")
                    _SHARED_MODELS[key] = SentenceTransformer(self.embedding_model_name, device=self.device)
                self._model = _SHARED_MODELS[key]
            except Exception as e:
                print(f"Warning: Failed to load shared embedding model: {e}")
                self._model = None

    def update_playbook(self, new_playbook: str) -> None:
        """Update the internal playbook representation and re-parse."""
        self.playbook = new_playbook
        self.section_headers, self.bullets = self.parse_playbook(new_playbook)
        self._bullet_embeddings = None

    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        """Encode texts using the shared embedding model."""
        self._load_model()
        if self._model is not None:
            try:
                return self._model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                ).astype(np.float32)
            except Exception as e:
                print(f"[PlaybookManager] Error in encode: {e}")
        return None

    def retrieve_bullets(self, query: str, top_k: int = 10) -> str:
        """
        RAE Retrieval: retrieve Top-K relevant bullets for a query.
        Returns a formatted playbook string preserving sections containing only retrieved bullets.
        """
        if not self.bullets:
            return self.playbook

        if len(self.bullets) <= top_k:
            return self.playbook

        # Load embedding model and compute similarities
        self._load_model()

        if self._model is None or not EMBEDDING_AVAILABLE:
            # Fallback simple keyword search overlap
            results = []
            query_words = set(query.lower().split())
            for idx, b in enumerate(self.bullets):
                words = set(b["content"].lower().split())
                overlap = len(query_words.intersection(words))
                results.append((overlap, idx))
            results.sort(key=lambda x: x[0], reverse=True)
            retrieved_indices = {self.bullets[idx]["id"] for score, idx in results[:top_k]}
        else:
            try:
                query_emb = self.encode([query])[0]

                if self._bullet_embeddings is None:
                    contents = [b["content"] for b in self.bullets]
                    self._bullet_embeddings = self.encode(contents)

                if self._bullet_embeddings is not None:
                    similarities = np.dot(self._bullet_embeddings, query_emb)
                    top_indices = np.argsort(similarities)[::-1][:top_k]
                    retrieved_indices = {self.bullets[idx]["id"] for idx in top_indices}
                else:
                    retrieved_indices = {b["id"] for b in self.bullets[:top_k]}
            except Exception as e:
                print(f"[PlaybookManager] Error in retrieval embedding search: {e}")
                retrieved_indices = {b["id"] for b in self.bullets[:top_k]}

        # Form a playbook string maintaining structure but filtering bullets
        focused_lines: List[str] = []
        for line in self.playbook.split('\n'):
            line_str = line.strip()
            if self.HEADER_PATTERN.match(line_str):
                focused_lines.append(line)
            else:
                parsed = self._parse_line(line_str)
                if parsed:
                    if parsed['id'] in retrieved_indices:
                        focused_lines.append(line)
                else:
                    # preserve blank lines for format
                    if not line_str:
                        focused_lines.append(line)

        return '\n'.join(focused_lines)
