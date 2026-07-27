import json
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from .planner import Planner
from .executor import Executor
from ..utils.llm import timed_llm_call
from ..utils.formatting import extract_json_from_text

GENERATOR_PROMPT_MEMENTO = """You are an analysis expert tasked with answering questions using your knowledge, a curated playbook of strategies and insights, and previous examples from Case Memory.

**Instructions:**
- Read the playbook carefully and apply relevant strategies, formulas, and insights.
- Pay attention to common mistakes listed in the playbook and avoid them.
- Reference the Case Memory examples to understand the correct logic for similar questions.
- Show your reasoning step-by-step.
- Double-check your calculations and logic before providing the final answer.

Your response MUST be a valid JSON object matching this exact schema:
{{
  "reasoning": "[Your detailed step-by-step reasoning and calculations]",
  "bullet_ids": ["calc-00001", "fin-00002"],
  "final_answer": "[Your concise final answer]"
}}

**Playbook:**
{playbook}

**Case Memory:**
{cases}

**Question:**
{question}

**Context:**
{context}

**Response format (JSON only, no markdown formatting or code blocks):**
"""


class Generator:
    """
    Coordinating Generator agent (combining Meta-Planner and Executor).
    Processes user query and subtasks sequentially.
    """

    def __init__(self, planner: Planner, executor: Executor, max_cycles: int = 3):
        self.planner = planner
        self.executor = executor
        self.max_cycles = max_cycles

    async def generate(
        self,
        question: str,
        playbook: str,
        cases_text: str,
        context: str = "",
        use_json_mode: bool = False,
        call_id: str = "gen",
        log_dir: Optional[str] = None
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Generate answer for a query in a single-step LLM call (similar to standard ACE),
        but including case_bank (Case Memory) examples in the prompt.
        """
        # Format the prompt
        prompt = GENERATOR_PROMPT_MEMENTO.format(
            playbook=playbook,
            cases=cases_text or "No previous cases found in Case Memory.",
            question=question,
            context=context or "(None)"
        )

        # Call the LLM (using the planner's client and model config)
        response_text, call_info = await asyncio.to_thread(
            timed_llm_call,
            self.planner.api_client,
            self.planner.api_provider,
            self.planner.model,
            prompt,
            role="generator",
            call_id=call_id,
            max_tokens=self.planner.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode
        )

        # Parse JSON response to extract final_answer and bullet_ids
        bullet_ids_used = []
        final_answer = ""
        try:
            data = extract_json_from_text(response_text)
            if data:
                final_answer = str(data.get("final_answer", "")).strip()
                bullet_ids_used = data.get("bullet_ids", [])
            
            # Fallback if final_answer is missing, placeholder, or empty
            if not final_answer or final_answer == "No final answer found" or final_answer.startswith("["):
                from utils import extract_answer
                extracted = extract_answer(response_text)
                if extracted and extracted != "No final answer found":
                    final_answer = extracted.strip()
                else:
                    final_answer = response_text.strip()
        except Exception:
            from utils import extract_answer
            extracted = extract_answer(response_text)
            if extracted and extracted != "No final answer found":
                final_answer = extracted.strip()
            else:
                final_answer = response_text.strip()

        # Build trajectory trace matching the expected schema
        trajectory = {
            "question": question,
            "context": context,
            "final_answer": final_answer,
            "plan_json": response_text,
            "meta_trace": [{"cycle": 0, "prompt": prompt, "response": response_text}],
            "executor_trace": [],
            "tool_history": [],
            "bullet_ids_used": bullet_ids_used
        }

        return final_answer, bullet_ids_used, trajectory
