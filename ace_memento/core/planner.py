import json
from typing import Dict, List, Tuple, Any, Optional
from ..utils.llm import timed_llm_call
from ..utils.formatting import META_PLANNER_SYSTEM_PROMPT, META_PLANNER_USER_TEMPLATE, strip_fences, extract_json_from_text


class Planner:
    """
    Planner agent that reads Dual Memory (CBR cases + Playbook rules)
    and decomposes user query into subtasks, or outputs a final answer.
    """

    def __init__(self, api_client: Any, api_provider: str, model: str, max_tokens: int = 4096):
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens

    def plan(
        self,
        question: str,
        playbook: str,
        cases_text: str,
        history_text: str = "(No task history yet)",
        context: str = "",
        use_json_mode: bool = False,
        call_id: str = "plan",
        log_dir: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Produce a plan or final answer based on prompt.
        """
        # Format the user prompt
        user_prompt = META_PLANNER_USER_TEMPLATE.format(
            playbook=playbook,
            cases=cases_text,
            history=history_text,
            question=question,
            context=context
        )

        # Merge system prompt into prompt if sglang or some providers don't separate it nicely,
        # but standard timed_llm_call messages array supports it. Here we construct the unified prompt.
        prompt = f"{META_PLANNER_SYSTEM_PROMPT}\n\n{user_prompt}"

        # We pass use_json_mode=True if the planner expects JSON plan format,
        # but if we expect "FINAL ANSWER:" format, we shouldn't force JSON mode
        # or we should check if the planner output starts with "FINAL ANSWER:"
        response, call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="planner",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode
        )

        return response, call_info
