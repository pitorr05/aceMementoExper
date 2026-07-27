import re
import json
from typing import Dict, List, Tuple, Optional, Any
from ..utils.llm import timed_llm_call
from ..utils.formatting import extract_json_from_text
from .playbook import PlaybookManager

CURATOR_PROMPT = """You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help answer similar questions.
- The reflection is generated using ground truth answers that will NOT be available when the playbook is being used.

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt.
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook.
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook.
- Do NOT regenerate the entire playbook - only provide the additions needed.
- Focus on quality over quantity.
- Format your response as a PURE JSON object with specific sections.

**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Question Context:**
{question_context}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / detailed analysis
- operations: a list of operations to be performed on the playbook
  - type: the type of operation ("ADD")
  - section: the section to add the bullet to (e.g. "formulas_and_calculations", "strategies_and_insights", "common_mistakes_to_avoid", "others")
  - content: the new content of the bullet (without any ID or helpful/harmful prefixes)

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your detailed reasoning process]",
  "operations": [
    {{
      "type": "ADD", 
      "section": "formulas_and_calculations",
      "content": "[New calculation method details...]"
    }}
  ]
}}
"""

CURATOR_PROMPT_NO_GT = """You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help answer similar questions.
- The reflection is generated using environment feedback that will NOT be available when the playbook is being used.

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt.
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook.
- Avoid redundancy.
- Format your response as a PURE JSON object with specific sections.

**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Question Context:**
{question_context}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / detailed analysis
- operations: a list of operations to be performed on the playbook
  - type: the type of operation ("ADD")
  - section: the section to add the bullet to
  - content: the new content of the bullet

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your detailed reasoning process]",
  "operations": [
    {{
      "type": "ADD", 
      "section": "formulas_and_calculations",
      "content": "[New calculation method details...]"
    }}
  ]
}}
"""


class Curator:
    """
    Curator agent updating Playbook via delta operations.
    """

    def __init__(self, api_client: Any, api_provider: str, model: str, max_tokens: int = 4096):
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens

    def curate(
        self,
        current_playbook: str,
        recent_reflection: str,
        question_context: str,
        current_step: int,
        total_samples: int,
        token_budget: int,
        playbook_stats: Dict[str, Any],
        use_ground_truth: bool = True,
        use_json_mode: bool = False,
        call_id: str = "curate",
        log_dir: Optional[str] = None,
        next_global_id: int = 1
    ) -> Tuple[str, int, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Produce delta playbook modifications based on reflection.
        """
        stats_str = json.dumps(playbook_stats, indent=2)

        if use_ground_truth:
            prompt = CURATOR_PROMPT.format(
                current_step=current_step,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats_str,
                recent_reflection=recent_reflection,
                current_playbook=current_playbook,
                question_context=question_context
            )
        else:
            prompt = CURATOR_PROMPT_NO_GT.format(
                current_step=current_step,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats_str,
                recent_reflection=recent_reflection,
                current_playbook=current_playbook,
                question_context=question_context
            )

        response, call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="curator",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode
        )

        operations = []
        updated_playbook = current_playbook

        # Check for empty response error
        if response.startswith("INCORRECT_DUE_TO_EMPTY_RESPONSE"):
            print(f"Skipping curator operation due to empty response")
            return current_playbook, next_global_id, [], call_info

        try:
            parsed = extract_json_from_text(response)
            if parsed and "operations" in parsed:
                operations = parsed["operations"]
                updated_playbook, next_global_id = self.apply_curator_operations(
                    current_playbook, operations, next_global_id
                )
        except Exception as e:
            print(f"[Curator] Error parsing operations: {e}")

        return updated_playbook, next_global_id, operations, call_info

    def get_section_slug(self, section_name: str) -> str:
        """Get 3-letter prefix slug for the section name."""
        slug_map = {
            "strategies_and_insights": "str",
            "formulas_and_calculations": "calc",
            "code_snippets_and_templates": "code",
            "common_mistakes_to_avoid": "err",
            "problem_solving_heuristics": "prob",
            "context_clues_and_indicators": "ctx",
            "others": "misc"
        }
        clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and")
        return slug_map.get(clean_name, "misc")

    def apply_curator_operations(
        self,
        playbook_text: str,
        operations: List[Dict[str, Any]],
        next_id: int
    ) -> Tuple[str, int]:
        """Apply ADD operations to the playbook, keeping layout structure intact."""
        lines = playbook_text.strip().split('\n')
        
        # Build section map
        sections = {}
        current_section = "general"
        
        for i, line in enumerate(lines):
            if line.strip().startswith('##'):
                section_header = line.strip()[2:].strip()
                current_section = section_header.lower().replace(' ', '_').replace('&', 'and')
                if current_section not in sections:
                    sections[current_section] = []
            elif line.strip():
                sections[current_section].append((i, line))
        
        bullets_to_add = []
        for op in operations:
            op_type = op.get('type', 'ADD')
            if op_type == 'ADD':
                section_raw = op.get('section', 'others')
                section = section_raw.lower().replace(' ', '_').replace('&', 'and')
                
                if section not in sections and section != 'general':
                    section = 'others'
                
                slug = self.get_section_slug(section)
                new_id = f"{slug}-{next_id:05d}"
                next_id += 1
                
                content = op.get('content', '')
                new_line = f"[{new_id}] helpful=0 harmful=0 :: {content}"
                bullets_to_add.append((section, new_line))
                print(f"[Curator] Added bullet {new_id} to section {section}")

        # Re-build playbook text
        final_lines = []
        current_section = None
        
        for line in lines:
            if line.strip().startswith('##'):
                if current_section:
                    section_adds = [b for s, b in bullets_to_add if s == current_section]
                    final_lines.extend(section_adds)
                    bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]
                
                section_header = line.strip()[2:].strip()
                current_section = section_header.lower().replace(' ', '_').replace('&', 'and')
            final_lines.append(line)
        
        if current_section:
            section_adds = [b for s, b in bullets_to_add if s == current_section]
            final_lines.extend(section_adds)
            bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]
            
        if bullets_to_add:
            others_bullets = [b for s, b in bullets_to_add]
            others_idx = -1
            for i, line in enumerate(final_lines):
                if line.strip() == "## OTHERS":
                    others_idx = i
                    break
            if others_idx >= 0:
                for i, bullet in enumerate(others_bullets):
                    final_lines.insert(others_idx + 1 + i, bullet)
            else:
                final_lines.extend(others_bullets)

        return '\n'.join(final_lines), next_id

    def update_bullet_counts(self, playbook_text: str, bullet_tags: List[Dict[str, str]]) -> str:
        """Update helpful/harmful counts on playbook lines based on Reflector tags."""
        lines = playbook_text.strip().split('\n')
        updated_lines = []
        
        tag_map = {}
        for tag in bullet_tags:
            bullet_id = tag.get('id') or tag.get('bullet', '')
            tag_value = tag.get('tag', 'neutral')
            if bullet_id:
                tag_map[bullet_id] = tag_value
                
        if not tag_map:
            return playbook_text

        for line in lines:
            line_str = line.strip()
            if line_str.startswith('##') or not line_str:
                updated_lines.append(line)
                continue
                
            m = PlaybookManager.BULLET_PATTERN.match(line_str)
            if m:
                bullet_id, helpful, harmful, content = m.groups()
                helpful = int(helpful)
                harmful = int(harmful)
                
                if bullet_id in tag_map:
                    tag = tag_map[bullet_id]
                    if tag == 'helpful':
                        helpful += 1
                    elif tag == 'harmful':
                        harmful += 1
                
                new_line = f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {content}"
                updated_lines.append(new_line)
            else:
                updated_lines.append(line)
                
        return '\n'.join(updated_lines)

    def get_playbook_stats(self, playbook_text: str) -> Dict[str, Any]:
        """Generate statistics about Playbook performance."""
        lines = playbook_text.strip().split('\n')
        stats = {
            'total_bullets': 0,
            'high_performing': 0,
            'problematic': 0,
            'unused': 0,
            'by_section': {}
        }
        
        current_section = 'general'
        for line in lines:
            line_str = line.strip()
            if line_str.startswith('##'):
                current_section = line_str[2:].strip()
                continue
                
            m = PlaybookManager.BULLET_PATTERN.match(line_str)
            if m:
                bullet_id, helpful, harmful, content = m.groups()
                helpful = int(helpful)
                harmful = int(harmful)
                stats['total_bullets'] += 1
                
                if helpful > 5 and harmful < 2:
                    stats['high_performing'] += 1
                elif harmful >= helpful and harmful > 0:
                    stats['problematic'] += 1
                elif helpful + harmful == 0:
                    stats['unused'] += 1
                
                if current_section not in stats['by_section']:
                    stats['by_section'][current_section] = {'count': 0, 'helpful': 0, 'harmful': 0}
                
                stats['by_section'][current_section]['count'] += 1
                stats['by_section'][current_section]['helpful'] += helpful
                stats['by_section'][current_section]['harmful'] += harmful
                
        return stats
