"""
Conversation Splitter - Semantic boundary detection.

Detects when a conversation shifts to a new topic.
Inspired by cognitive science principles of episode segmentation.
"""

import json
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class SplitSignal:
    """Output of conversation splitting."""
    def __init__(self, should_split: bool, confidence: float, reason: str = None):
        self.should_split = should_split
        self.confidence = confidence
        self.reason = reason


class ConversationSplitter:
    """
    Detects when a conversation should be split into chunks.
    
    Key principles:
    - LLM evaluates semantic coherence between new message and buffer
    - Hard limit prevents buffer overflow (max 25 messages)
    - Confidence threshold controls splitting decisions (default 0.7)
    """
    
    def __init__(
        self,
        llm_client,
        llm_provider: str,
        llm_model: str,
        confidence_threshold: float = 0.6,
        max_buffer_size: int = 19,
    ):
        self.llm_client = llm_client
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.confidence_threshold = confidence_threshold
        self.max_buffer_size = max_buffer_size
    
    async def should_split(
        self,
        new_message: str,
        buffer: List[Dict[str, Any]],
    ) -> Tuple[bool, Optional[SplitSignal]]:
        """
        Determine if a new message should start a new chunk.
        
        Args:
            new_message: The incoming message content
            buffer: Current conversation buffer
        
        Returns:
            (should_split, signal)
        """
        # Hard limit: always split if buffer full
        if len(buffer) >= self.max_buffer_size:
            return True, None
        
        # Empty buffer: don't split, just add
        if not buffer:
            return False, None
        
        # LLM-based split detection
        signal = await self._detect_split(new_message, buffer)
        
        should_split = signal.should_split and signal.confidence >= self.confidence_threshold
        if should_split:
            print(f"[Splitter] Split detected! confidence={signal.confidence:.2f}, reason={signal.reason}")
        
        return should_split, signal
    
    async def _detect_split(
        self,
        new_message: str,
        buffer: List[Dict[str, Any]],
    ) -> SplitSignal:
        """Use LLM to detect if new message crosses a topic boundary."""
        from ace_memento.utils.llm import timed_llm_call
        
        context = self._format_context(buffer)
        
        prompt = f"""
You are analyzing a conversation flow to detect topic boundaries.

Recent conversation context:
{context}

New message:
{new_message}

Determine if this new message starts a NEW topic or continues the current one.

Guidelines:
- Look for topic shifts, new subjects, or unrelated content
- Consider temporal markers ("by the way", "anyway", "on another note")
- Consider intent shifts (question → statement, info-seeking → decision)

Respond with JSON:
{{"should_split": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}

Confidence guide:
- 0.9-1.0: Clearly a new topic
- 0.7-0.9: Likely a new topic  
- 0.5-0.7: Unclear
- 0.0-0.5: Clearly continuing
"""
        
        response, _ = timed_llm_call(
            self.llm_client,
            self.llm_provider,
            self.llm_model,
            prompt,
            role="splitter",
            call_id="splitter",
            max_tokens=256,
        )
        
        return self._parse_response(response)
    
    def _format_context(self, messages: List[Dict], max_messages: int = 10) -> str:
        recent = messages[-max_messages:] if len(messages) > max_messages else messages
        lines = [f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}" 
                 for msg in recent]
        return "\n".join(lines)
    
    def _parse_response(self, response: str) -> SplitSignal:
        try:
            if "```json" in response:
                start = response.index("```json") + 7
                end = response.index("```", start)
                response = response[start:end]
            elif "```" in response:
                start = response.index("```") + 3
                end = response.index("```", start)
                response = response[start:end]
            
            data = json.loads(response.strip())
            
            return SplitSignal(
                should_split=bool(data.get("should_split", False)),
                confidence=float(data.get("confidence", 0.0)),
                reason=data.get("reason"),
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return SplitSignal(
                should_split=False,
                confidence=0.0,
                reason="Failed to parse LLM response",
            )
