"""
DataProcessor cho DeepResearcher dataset
Format: {"question": "...", "ground_truth": "...", "data_source": "..."}
"""
import json
import re
from typing import Dict, List, Any, Optional

class DeepResearcherProcessor:
    """DataProcessor cho dataset DeepResearcher"""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Chuyển đổi dữ liệu raw sang format chuẩn của ACE
        Input: [{"question": ..., "ground_truth": ..., "data_source": ...}]
        Output: [{"question": ..., "context": ..., "target": ..., "others": {...}}]
        """
        processed = []
        for item in raw_data:
            processed.append({
                'question': item.get('question', ''),
                'context': item.get('context', ''),  # DeepResearcher không có context
                'target': item.get('ground_truth', ''),
                'others': {
                    'data_source': item.get('data_source', 'unknown')
                }
            })
        return processed

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        So sánh câu trả lời dự đoán và ground truth
        Linh hoạt cho các kiểu đáp án khác nhau
        """
        if not predicted or not ground_truth:
            return False

        # Chuẩn hóa: loại bỏ khoảng trắng thừa, dấu ngoặc kép, xuống dòng
        pred_clean = self._normalize_answer(predicted)
        gt_clean = self._normalize_answer(ground_truth)

        # 1. Exact match
        if pred_clean == gt_clean:
            return True

        # 2. Chứa nhau (không phân biệt hoa thường)
        if gt_clean.lower() in pred_clean.lower() or pred_clean.lower() in gt_clean.lower():
            return True

        # 3. So sánh từng từ (cho các câu trả lời dài)
        pred_words = set(pred_clean.lower().split())
        gt_words = set(gt_clean.lower().split())
        if pred_words and gt_words:
            # Nếu > 50% từ khớp nhau
            overlap = len(pred_words.intersection(gt_words))
            if overlap / len(gt_words) > 0.5:
                return True

        return False

    def _normalize_answer(self, text: str) -> str:
        """Chuẩn hóa câu trả lời"""
        if not text:
            return ""
        # Loại bỏ dấu ngoặc kép, khoảng trắng thừa
        text = text.strip().strip('"\'')
        # Thay thế nhiều khoảng trắng bằng 1
        text = ' '.join(text.split())
        # Loại bỏ các ký tự đặc biệt (giữ lại chữ, số, dấu câu cơ bản)
        text = re.sub(r'[^\w\s.,;:!?-]', '', text)
        return text.strip()

    def evaluate_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Tính độ chính xác trên toàn bộ tập"""
        if not predictions or not ground_truths:
            return 0.0

        correct = sum(1 for p, g in zip(predictions, ground_truths)
                     if self.answer_is_correct(p, g))
        return correct / len(predictions)

    @staticmethod
    def load_data(file_path: str) -> List[Dict]:
        """Helper function để load dữ liệu JSONL"""
        samples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        samples.append({
                            'question': data.get('question', ''),
                            'context': data.get('context', ''),
                            'target': data.get('ground_truth', ''),
                            'data_source': data.get('data_source', 'unknown')
                        })
            print(f"Loaded {len(samples)} samples from {file_path}")
        except FileNotFoundError:
            print(f" File not found: {file_path}")
        return samples