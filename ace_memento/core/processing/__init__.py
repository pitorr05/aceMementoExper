"""
AMORE Processing Pipeline - Socratic Contradiction Resolution

A novel memory management framework inspired by the Socratic method:
- ConversationSplitter: Detects semantic boundaries
- ChunkCompressor: Compresses conversations into structured chunks
- ContradictionResolver: Discovers contradictions between belief and evidence

This is the core of AMORE's novelty - learning through contradiction resolution.
"""

from .conversation_splitter import ConversationSplitter, SplitSignal
from .chunk_compressor import ChunkCompressor, ConversationChunk
from .contradiction_resolver import ContradictionResolver

__all__ = [
    "ConversationSplitter",
    "SplitSignal",
    "ChunkCompressor",
    "ConversationChunk",
    "ContradictionResolver",
]