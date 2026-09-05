"""
Socratic Contradiction Resolver - Core of AMORE.

Inspired by the Socratic method of learning through questioning:
1. HYPOTHESIS: What does the system believe? (from Playbook)
2. EVIDENCE: What does the conversation show? (raw conversation)
3. CONTRADICTION: Where do they differ?
4. RESOLUTION: Extract new knowledge from the contradiction

This is the novelty of AMORE - learning through contradiction discovery,
NOT through gap scoring or passive storage.
"""

import json
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ContradictionResolver:
    """
    Resolves contradictions between system belief and observed evidence.
    
    Key insight: Learning occurs when we discover what we got wrong.
    Not by measuring how much we got wrong (gap scoring),
    but by directly finding WHAT we got wrong (contradictions).
    
    This is the Socratic method in action.
    """
    
    def __init__(
        self,
        llm_client,
        llm_provider: str,
        llm_model: str,
        playbook_manager,
    ):
        self.llm_client = llm_client
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.playbook_manager = playbook_manager
    
    async def resolve(
        self,
        chunk_title: str,
        raw_conversation: List[Dict[str, Any]],
        ground_truth: str,
    ) -> Tuple[bool, List[str]]:
        """
        Resolve contradictions between belief and evidence.
        
        Returns:
            (has_contradiction, new_insights)
        """
        # Step 1: Form hypothesis from Playbook
        hypothesis = await self._form_hypothesis(chunk_title)
        
        # Step 2: Find contradictions (ONE LLM call)
        contradictions = await self._find_contradictions(
            hypothesis, raw_conversation
        )
        
        # Step 3: Check if any contradiction exists
        has_contradiction = len(contradictions) > 0
        
        if has_contradiction:
            print(f"[ContradictionResolver] Found {len(contradictions)} contradictions")
            for c in contradictions[:3]:
                print(f"  - {c}")
        else:
            print("[ContradictionResolver] No contradictions found")
        
        return has_contradiction, contradictions
    
    async def _form_hypothesis(self, chunk_title: str) -> str:
        """
        Form hypothesis about what the conversation should contain.
        
        This is the "belief" to be tested against evidence.
        """
        from ace_memento.utils.llm import timed_llm_call
        
        # Retrieve relevant bullets from Playbook
        bullets = self.playbook_manager.retrieve_bullets(chunk_title, top_k=5)
        
        if not bullets or bullets == self.playbook_manager.playbook:
            return ""
        
        bullet_contents = [b["content"] for b in self.playbook_manager.bullets[:5]]
        knowledge_text = "\n".join(f"- {k}" for k in bullet_contents)
        
        prompt = f"""
Based on existing knowledge, form a hypothesis about this conversation.

Existing knowledge:
{knowledge_text}

Conversation topic: {chunk_title}

Hypothesize what the conversation likely contains (1-3 sentences).
If no hypothesis can be formed, output "NO_HYPOTHESIS".
"""
        
        response, _ = timed_llm_call(
            self.llm_client,
            self.llm_provider,
            self.llm_model,
            prompt,
            role="hypothesizer",
            call_id="hypothesizer",
            max_tokens=512,
        )
        
        if "NO_HYPOTHESIS" in response.upper():
            return ""
        
        return response.strip()
    
    async def _find_contradictions(
        self,
        hypothesis: str,
        raw_conversation: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Find contradictions between hypothesis and evidence.
        
        This is the Socratic method in action:
        ONE LLM call compares belief vs evidence and extracts contradictions.
        """
        from ace_memento.utils.llm import timed_llm_call
        
        raw_text = self._format_conversation(raw_conversation)
        
        # Cold start: no hypothesis to contradict
        if not hypothesis:
            prompt = f"""
Extract all important facts from this conversation.

CONVERSATION:
{raw_text}

Output each fact as a separate line.
If nothing important, output "NO_FACTS".
"""
        else:
            # Find contradictions between hypothesis and evidence
            prompt = f"""
Find contradictions between the hypothesis and the actual conversation.

HYPOTHESIS (what we believed would be there):
{hypothesis}

EVIDENCE (what the conversation actually shows):
{raw_text}

Extract ONLY the information that CONTRADICTS or is ABSENT from the hypothesis.
These are the learning opportunities - what we got wrong or missed.

Output each contradiction as a separate line.
If no contradictions found, output "NO_CONTRADICTIONS".
"""
        
        response, _ = timed_llm_call(
            self.llm_client,
            self.llm_provider,
            self.llm_model,
            prompt,
            role="contradiction_finder",
            call_id="contradiction_finder",
            max_tokens=1024,
        )
        
        if "NO_CONTRADICTIONS" in response.upper() or "NO_FACTS" in response.upper():
            return []
        
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        return lines
    
    def _format_conversation(self, messages: List[Dict]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f">>> USER: {content}")
            else:
                lines.append(f"ASSISTANT: {content}")
        return "\n".join(lines)
