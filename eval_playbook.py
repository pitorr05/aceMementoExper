#!/usr/bin/env python
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ace_memento import ACE
from eval.deepresearcher.data_processor import DeepResearcherProcessor

PLAYBOOK_PATH = "./results_full/ace_memento_20260712_111916_offline/best_playbook.txt"
DATA_DIR = "./data/deepresearcher_subset"
SAVE_DIR = "./results_test_50"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"

with open(PLAYBOOK_PATH, "r") as f:
    playbook = f.read()

print(f"✅ Loaded playbook: {len(playbook)} chars")

processor = DeepResearcherProcessor()
test_samples = processor.load_data(os.path.join(DATA_DIR, "test.jsonl"))

print(f"✅ Loaded {len(test_samples)} test samples")

ace = ACE(
    api_provider="vllm",
    generator_model=MODEL,
    reflector_model=MODEL,
    curator_model=MODEL,
    max_tokens=4096,
    use_rae=True,
    rae_top_k=10,
    use_failure_memory=True,
    failure_memory_top_k=3,
    device="cuda"
)

config = {
    'task_name': 'deepresearcher',
    'save_dir': SAVE_DIR,
    'test_workers': 10,
    'api_provider': 'vllm',
    'use_rae': True,
    'rae_top_k': 10,
    'use_failure_memory': True,
    'failure_memory_top_k': 3,
    'json_mode': False,
}

print("\n" + "="*60)
print("🧪 Đánh giá playbook trên 50 mẫu test")
print("="*60 + "\n")

results = ace.run(
    mode='eval_only',
    test_samples=test_samples,
    data_processor=processor,
    config=config
)

print("\n" + "="*60)
print("📊 KẾT QUẢ ĐÁNH GIÁ")
print("="*60)
print(json.dumps(results, indent=2))
print("="*60)
