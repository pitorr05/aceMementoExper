"""
Core module for ACE system.
Contains the three main agent classes: Generator, Reflector, and Curator.
"""

from .generator import Generator
from .reflector import Reflector
from .curator import Curator
from .adversarial_agent import AdversarialAgent
from .bulletpoint_analyzer import BulletpointAnalyzer, DEDUP_AVAILABLE
from .playbook_retriever import PlaybookRetriever, RAE_AVAILABLE
from .failure_memory import FailureMemoryBank, MEMORY_AVAILABLE

__all__ = [
    'Generator', 'Reflector', 'Curator', 'AdversarialAgent',
    'BulletpointAnalyzer', 'DEDUP_AVAILABLE',
    'PlaybookRetriever', 'RAE_AVAILABLE',
    'FailureMemoryBank', 'MEMORY_AVAILABLE',
]