import json
from typing import Dict, List, Tuple, Optional, Any
from ..utils.llm import timed_llm_call
from ..utils.formatting import extract_json_from_text

REFLECTOR_PROMPT = """You are an expert analyst and educator. Your job is to diagnose why a model's reasoning went wrong by analyzing the gap between the predicted answer and the ground truth.

**Instructions:**
- Carefully analyze the model's reasoning trace and task execution history to identify where it went wrong.
- Compare the predicted answer with the ground truth to understand the gap.
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies.
- Provide actionable insights that could help the model avoid this mistake in the future.
- You will receive the bulletpoints from the Playbook that were used by the planner.
- Analyze these bulletpoints and give a tag for each: 'helpful', 'harmful', or 'neutral'.
- You will also receive similar past failure cases for analogical reasoning. Use these to draw parallels and identify recurring error patterns.

Your output must be a JSON object containing these exact fields:
- reasoning: your chain of thought / detailed analysis
- error_identification: what specifically went wrong?
- root_cause_analysis: why did this error occur?
- correct_approach: what should the model have done instead?
- key_insight: what strategy, formula, or principle should be added or remembered to avoid this error?
- analogical_note: how does this failure compare to similar past failures?
- bullet_tags: a list of objects with "id" and "tag" for each bulletpoint used.

**Question:**
{}

**Model's Trajectory (Plan + Steps + Tool Calls):**
{}

**Model's Predicted Answer:**
{}

**Ground Truth Answer:**
{}

**Environment Feedback:**
{}

**Playbook Bulletpoints Used:**
{}

**Similar Past Failures (for analogical reflection):**
{}

**Answer in this exact JSON format (do not use markdown blocks):**
{{
  "reasoning": "[Your detailed reasoning process]",
  "error_identification": "[Specific error identification]",
  "root_cause_analysis": "[Root cause analysis]",
  "correct_approach": "[Correct approach details]",
  "key_insight": "[Key insight to prevent this mistake]",
  "analogical_note": "[Analogical comparison notes]",
  "bullet_tags": [
    {{"id": "calc-00001", "tag": "helpful"}},
    {{"id": "fin-00002", "tag": "harmful"}}
  ]
}}
"""

REFLECTOR_PROMPT_NO_GT = """You are an expert analyst and educator. Your job is to diagnose why a model's reasoning went wrong when coming up with the predicted answer.

**Instructions:**
- Carefully analyze the model's reasoning trace and task execution history to identify where it went wrong.
- Take the environment feedback into account.
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies.
- Provide actionable insights that could help the model avoid this mistake in the future.
- You will receive the bulletpoints from the Playbook that were used by the planner.
- Analyze these bulletpoints and give a tag for each: 'helpful', 'harmful', or 'neutral'.
- You will also receive similar past failure cases for analogical reasoning.

Your output must be a JSON object containing these exact fields:
- reasoning: your chain of thought / detailed analysis
- error_identification: what specifically went wrong?
- root_cause_analysis: why did this error occur?
- correct_approach: what should the model have done instead?
- key_insight: what strategy, formula, or principle should be added or remembered to avoid this error?
- analogical_note: how does this failure compare to similar past failures?
- bullet_tags: a list of objects with "id" and "tag" for each bulletpoint used.

**Question:**
{}

**Model's Trajectory (Plan + Steps + Tool Calls):**
{}

**Model's Predicted Answer:**
{}

**Environment Feedback:**
{}

**Playbook Bulletpoints Used:**
{}

**Similar Past Failures (for analogical reflection):**
{}

**Answer in this exact JSON format (do not use markdown blocks):**
{{
  "reasoning": "[Your detailed reasoning process]",
  "error_identification": "[Specific error identification]",
  "root_cause_analysis": "[Root cause analysis]",
  "correct_approach": "[Correct approach details]",
  "key_insight": "[Key insight to prevent this mistake]",
  "analogical_note": "[Analogical comparison notes]",
  "bullet_tags": [
    {{"id": "calc-00001", "tag": "helpful"}},
    {{"id": "fin-00002", "tag": "harmful"}}
  ]
}}
"""


class Reflector:
    """
    Reflector agent diagnosing mistakes and tagging playbook bullets.
    """

    def __init__(self, api_client: Any, api_provider: str, model: str, max_tokens: int = 4096):
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens

    def reflect(
        self,
        question: str,
        trajectory_str: str,
        predicted_answer: str,
        ground_truth: Optional[str],
        environment_feedback: str,
        bullets_used_str: str,
        analogical_context: str = "(No similar past failures found)",
        use_ground_truth: bool = True,
        use_json_mode: bool = False,
        call_id: str = "reflect",
        log_dir: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Run the reflector model on a given trajectory.
        """
        if use_ground_truth and ground_truth:
            prompt = REFLECTOR_PROMPT.format(
                question,
                trajectory_str,
                predicted_answer,
                ground_truth,
                environment_feedback,
                bullets_used_str,
                analogical_context
            )
        else:
            prompt = REFLECTOR_PROMPT_NO_GT.format(
                question,
                trajectory_str,
                predicted_answer,
                environment_feedback,
                bullets_used_str,
                analogical_context
            )

        response, call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="reflector",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode
        )

        # Parse bullet tags
        bullet_tags = []
        try:
            parsed = extract_json_from_text(response)
            if parsed and "bullet_tags" in parsed:
                bullet_tags = parsed["bullet_tags"]
        except Exception as e:
            print(f"[Reflector] Error extracting bullet tags from response: {e}")

        return response, bullet_tags, call_info
