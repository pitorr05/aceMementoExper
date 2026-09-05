"""
Predict-Calibrate Gate for intelligent memory storage
Based on Nemori 2025: Only store what cannot be predicted from current knowledge
"""
from typing import Tuple, Optional, Any
from ..utils.llm import timed_llm_call

class PredictCalibrateGate:
    """
    Implements Predict-Calibrate Loop from Nemori paper.
    Only stores cases that the system cannot predict from current knowledge.
    """
    
    def __init__(
        self, 
        api_client: Any, 
        api_provider: str, 
        model: str, 
        max_tokens: int = 4096,
        uncertainty_threshold: float = 0.3
    ):
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
        self.uncertainty_threshold = uncertainty_threshold
    
    def should_store(
        self,
        question: str,
        predicted_answer: str,
        ground_truth: str,
        current_playbook: str,
        confidence_score: float = 0.0,
        call_id: str = "gate"
    ) -> Tuple[bool, float, str]:
        """
        Determine if a case should be stored.
        
        Returns:
            (should_store, confidence, reasoning)
        """
        # --- Step 1: If wrong, always store (but with low confidence) ---
        if predicted_answer != ground_truth:
            return True, 0.3, "Prediction was incorrect. Storing for future correction."
        
        # --- Step 2: If correct, check if it's a knowledge gap ---
        # 2a. If we don't have a playbook, store (high confidence)
        if not current_playbook or current_playbook == "":
            return True, 0.9, "No playbook available. Storing as new knowledge."
        
        # 2b. Estimate prediction gap using LLM
        prediction_gap = self._estimate_prediction_gap(question, current_playbook)
        
        # 2c. Decision based on gap
        if prediction_gap > self.uncertainty_threshold:
            return True, prediction_gap, f"Knowledge gap detected (gap={prediction_gap:.2f}). Storing."
        else:
            return False, 0.0, f"Can be predicted from playbook (gap={prediction_gap:.2f}). Not storing."
    
    def _estimate_prediction_gap(
        self, 
        question: str, 
        playbook: str,
        call_id: str = "gap"
    ) -> float:
        """
        Estimate how much new knowledge this case contains.
        
        Uses LLM to evaluate if the playbook alone can answer the question.
        Returns a score between 0.0 (can answer) and 1.0 (cannot answer).
        """
        prompt = self._build_gap_prompt(question, playbook)
        
        response, _ = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="gate",
            call_id=call_id,
            max_tokens=128,
        )
        
        # Parse response for gap score
        gap_score = self._parse_gap_score(response)
        return gap_score
    
    def _build_gap_prompt(self, question: str, playbook: str) -> str:
        return f"""
You are an evaluator assessing whether a playbook has enough information to answer a question.

Playbook:
{playbook}

Question:
{question}

Rate how well the playbook can answer this question on a scale of 0.0 to 1.0:
- 0.0: The playbook contains all necessary information to answer correctly
- 0.5: The playbook has some relevant information but not enough
- 1.0: The playbook has no relevant information

Output ONLY a single float number between 0.0 and 1.0:
"""
    
    def _parse_gap_score(self, response: str) -> float:
        """Extract gap score from LLM response"""
        try:
            import re
            # Find all numbers in the response
            numbers = re.findall(r'(\d+\.?\d*)', response)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            return 0.5  # Default if parsing fails
        except Exception:
            return 0.5  # Default if parsing fails