#!/usr/bin/env python
"""
Run ACE gốc (không Memento) trên DeepResearcher dataset
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Thêm path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import ACE gốc (không phải ace_memento)
from ace import ACE
from eval.deepresearcher.data_processor import DeepResearcherProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Run ACE gốc trên DeepResearcher")
    parser.add_argument("--data_dir", type=str, default="./data/deepresearcher",
                        help="Directory containing train.jsonl, val.jsonl, test.jsonl")
    parser.add_argument("--save_dir", type=str, default="./results_ace_goc",
                        help="Directory to save results")
    parser.add_argument("--api_provider", type=str, default="vllm",
                        choices=["vllm", "openai", "sambanova", "together"],
                        help="API provider")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Model name")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:5000/v1",
                        help="vLLM server URL")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--test_workers", type=int, default=30,
                        help="Number of parallel test workers")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set environment variables
    os.environ["VLLM_BASE_URL"] = args.vllm_url
    # KHÔNG set USE_MEMENTO -> dùng ACE gốc

    print("=" * 70)
    print(" ACE GỐC (không Memento) trên DeepResearcher")
    print("=" * 70)
    print(f" Data dir: {args.data_dir}")
    print(f" Save dir: {args.save_dir}")
    print(f" API: {args.api_provider}")
    print(f" Model: {args.model}")
    print("=" * 70)

    # --- Load dữ liệu ---
    print("\n Loading DeepResearcher data...")
    processor = DeepResearcherProcessor()
    
    train_data = processor.load_data(os.path.join(args.data_dir, "train.jsonl"))
    val_data = processor.load_data(os.path.join(args.data_dir, "val.jsonl"))
    test_data = processor.load_data(os.path.join(args.data_dir, "test.jsonl"))

    if not train_data and not test_data:
        print(" No data loaded!")
        sys.exit(1)

    if not val_data and test_data:
        val_data = test_data[:len(test_data)//2]

    print(f" Train: {len(train_data)} samples")
    print(f" Val: {len(val_data)} samples")
    print(f" Test: {len(test_data)} samples")

    # --- Khởi tạo ACE gốc ---
    print("\n Initializing ACE gốc...")
    ace = ACE(
        api_provider=args.api_provider,
        generator_model=args.model,
        reflector_model=args.model,
        curator_model=args.model,
        max_tokens=4096,
        # ACE gốc KHÔNG có các tham số này:
        # use_rae, use_failure_memory, memory_jsonl_path, device
    )

    # --- Cấu hình training ---
    config = {
        'num_epochs': args.num_epochs,
        'max_num_rounds': 3,
        'curator_frequency': 1,
        'eval_steps': 50,
        'save_steps': 25,
        'playbook_token_budget': 8000,
        'task_name': 'deepresearcher',
        'json_mode': False,
        'no_ground_truth': False,
        'save_dir': args.save_dir,
        'test_workers': args.test_workers,
        'api_provider': args.api_provider,
    }

    # --- Chạy offline training ---
    print("\n" + "=" * 70)
    print("🏋️  Starting OFFLINE Training (ACE gốc)")
    print("=" * 70 + "\n")

    results = ace.run(
        mode='offline',
        train_samples=train_data,
        val_samples=val_data,
        test_samples=test_data,
        data_processor=processor,
        config=config
    )

    # --- In kết quả ---
    print("\n" + "=" * 70)
    print(" EXPERIMENT COMPLETE!")
    print("=" * 70)
    if "initial_test_results" in results:
        print(f" Initial Test Accuracy: {results['initial_test_results'].get('accuracy', 0):.4f}")
    if "final_test_results" in results:
        print(f" Final Test Accuracy:   {results['final_test_results'].get('accuracy', 0):.4f}")
    if "best_validation_accuracy" in results:
        print(f" Best Validation Acc:   {results['best_validation_accuracy']:.4f}")
    print(f"\n Results saved to: {args.save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
