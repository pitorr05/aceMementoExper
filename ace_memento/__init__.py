from .runner import ACEMementoRunner
from .core.case_bank import CaseBank
from .core.playbook import PlaybookManager
from .core.generator import Generator
from .compat import ACE, HierarchicalClient, QueryRecord, MetaCycle, ExecStep, ToolCallRecord

# --- AMORE exports ---
try:
    from .core.amore_memory import AMOREMemory
    from .core.processing.conversation_splitter import ConversationSplitter, SplitSignal
    from .core.processing.chunk_compressor import ChunkCompressor, ConversationChunk
    from .core.processing.contradiction_resolver import ContradictionResolver
    AMORE_AVAILABLE = True
except ImportError:
    AMORE_AVAILABLE = False
    print("Warning: AMORE modules not found. Only CaseBank mode available.")

__all__ = [
    "ACEMementoRunner",
    "CaseBank",
    "PlaybookManager",
    "Generator",
    "ACE",
    "HierarchicalClient",
    "QueryRecord",
    "MetaCycle",
    "ExecStep",
    "ToolCallRecord",
    # --- AMORE exports ---
    "AMOREMemory",
    "ConversationSplitter",
    "SplitSignal",
    "ChunkCompressor",
    "ConversationChunk",
    "ContradictionResolver",
    "AMORE_AVAILABLE",
]