"""
Chunk Compressor - Compresses conversations into structured chunks.

Transforms raw conversation segments into:
- Title: Concise descriptive summary
- Content: Third-person narrative
- Raw conversation: Preserved for extraction (never modified)
"""

import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ConversationChunk:
    """Structured chunk with title, content, and raw conversation."""
    def __init__(
        self,
        title: str,
        content: str,
        raw_conversation: List[Dict[str, Any]],
    ):
        self.title = title
        self.content = content
        self.raw_conversation = raw_conversation  # Preserved for extraction


class ChunkCompressor:
    """
    Compresses conversation segments into structured chunks.
    
    Key principles:
    - Raw conversation is preserved for extraction (never use compressed content)
    - Title + Content for retrieval (easier to search)
    - Third-person narrative for content
    """
    
    def __init__(self, llm_client, llm_provider: str, llm_model: str):
        self.llm_client = llm_client
        self.llm_provider = llm_provider
        self.llm_model = llm_model
    
    async def compress(
        self,
        messages: List[Dict[str, Any]],
    ) -> ConversationChunk:
        """Compress a conversation segment into a chunk."""
        from ace_memento.utils.llm import timed_llm_call
        
        conversation = self._format_conversation(messages)
        
        prompt = f"""
Create a structured memory chunk from this conversation.

Conversation:
{conversation}

Output JSON with:
- "title": A short, descriptive title (max 10 words)
- "content": A third-person narrative summary (2-3 sentences)

Requirements:
- Title should be concise and descriptive of the key theme
- Content should be in third-person perspective
- Preserve key information without adding new facts
- Do not include conversational filler

Example:
{{"title": "Operating Margin Calculation Example",
  "content": "The user asked about calculating operating margin. The assistant explained that operating margin equals operating income divided by revenue, and provided a numerical example with Company A having EBIT of 45k and revenue of 300k, resulting in a 15% margin."}}
"""
        
        response, _ = await timed_llm_call(
            self.llm_client,
            self.llm_provider,
            self.llm_model,
            prompt,
            role="compressor",
            max_tokens=1024,
        )
        
        data = self._parse_response(response)
        title = data.get("title", "Untitled Conversation")
        print(f"[Compressor] Created chunk: '{title}'")
        
        return ConversationChunk(
            title=data.get("title", "Untitled Conversation"),
            content=data.get("content", ""),
            raw_conversation=messages.copy(),
        )
    
    def _format_conversation(self, messages: List[Dict]) -> str:
        lines = [f"{m.get('role', 'user').upper()}: {m.get('content', '')}" 
                 for m in messages]
        return "\n".join(lines)
    
    def _parse_response(self, response: str) -> Dict:
        try:
            if "```json" in response:
                start = response.index("```json") + 7
                end = response.index("```", start)
                response = response[start:end]
            elif "```" in response:
                start = response.index("```") + 3
                end = response.index("```", start)
                response = response[start:end]
            return json.loads(response.strip())
        except:
            return {"title": "Untitled", "content": ""}