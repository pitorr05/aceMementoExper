"""
Adversarial agent for ACE system.
Generates adversarial queries and their intended answers.
"""

import json
from typing import Dict, Optional, Any, Tuple

from ..prompts.adversarial import ADVERSARIAL_PROMPT
from playbook_utils import extract_json_from_text
from llm import timed_llm_call


class AdversarialAgent:
    """
    Adversarial agent that creates mock queries designed to expose
    weaknesses in the current playbook.
    """

    def __init__(self, api_client, api_provider, model: str, max_tokens: int = 4096):
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens

    def generate_attack(
        self,
        playbook: str,
        task_name: str,
        recent_question: str,
        recent_context: str,
        recent_target: str,
        use_json_mode: bool = False,
        call_id: str = "adv",
        log_dir: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a single adversarial sample.

        Returns:
            (attack_dict, call_info)
        """
        prompt = ADVERSARIAL_PROMPT.format(
            playbook=playbook,
            task_name=task_name,
            recent_question=recent_question,
            recent_context=recent_context,
            recent_target=recent_target,
        )

        response, call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="adversarial",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
        )

        attack = extract_json_from_text(response)
        if not attack or not isinstance(attack, dict):
            print("Warning: AdversarialAgent failed to parse attack JSON")
            return None, call_info

        required = ["question", "context", "target", "attack_rationale", "vulnerability_hint"]
        missing = [k for k in required if k not in attack]
        if missing:
            print(f"Warning: AdversarialAgent JSON missing fields: {missing}")
            return None, call_info

        for key in required:
            if attack.get(key) is None:
                attack[key] = ""
            else:
                attack[key] = str(attack[key]).strip()

        if not attack["question"] or not attack["target"]:
            print("Warning: AdversarialAgent produced empty question/target")
            return None, call_info

        return attack, call_info
