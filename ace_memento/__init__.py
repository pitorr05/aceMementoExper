# ace_memento/__init__.py

from .runner import ACEMementoRunner
from .core.case_bank import CaseBank
from .core.playbook import PlaybookManager
from .core.generator import Generator
from .compat import ACE, HierarchicalClient, QueryRecord, MetaCycle, ExecStep, ToolCallRecord

# --- AMORE imports ---
try:
    from .core.amore_memory import AMOREMemory
    from .core.adaptive_memory_entry import AdaptiveMemoryEntry
    from .core.adaptive_retriever import AdaptiveRetriever
    from .core.predict_calibrate_gate import PredictCalibrateGate
    from .core.memory_consolidator import MemoryConsolidator
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
    "AdaptiveMemoryEntry",
    "AdaptiveRetriever",
    "PredictCalibrateGate",
    "MemoryConsolidator",
    "AMORE_AVAILABLE",
]