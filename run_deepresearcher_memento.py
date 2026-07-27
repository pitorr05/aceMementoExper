#!/usr/bin/env python
"""
Run ACE-Memento on DeepResearcher dataset
End-to-end experiment to evaluate ACE + Memento combination
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Thêm path để import các module
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import từ ACEmento
from ace_memento import ACE
from eval.deepresearcher.data_processor import DeepResearcherProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Run ACE-Memento on DeepResearcher")
    parser.add_argument("--data_dir", type=str, default="./data/deepresearcher",
                        help="Directory containing train.jsonl, val.jsonl, test.jsonl")
    parser.add_argument("--save_dir", type=str, default="./results_deepresearcher",
                        help="Directory to save results")
    parser.add_argument("--api_provider", type=str, default="vllm",
                        choices=["vllm", "openai", "sambanova", "together"],
                        help="API provider")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Model name")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:5000/v1",
                        help="vLLM server URL")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--test_workers", type=int, default=10,
                        help="Number of parallel test workers")
    parser.add_argument("--use_rae", action="store_true", default=True,
                        help="Enable RAE (Retrieval-Augmented Execution)")
    parser.add_argument("--use_failure_memory", action="store_true", default=True,
                        help="Enable Failure Memory (Analogical Reflection)")
    parser.add_argument("--no_rae", action="store_false", dest="use_rae",
                        help="Disable RAE")
    parser.add_argument("--no_failure_memory", action="store_false", dest="use_failure_memory",
                        help="Disable Failure Memory")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use: cuda or cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set environment variables
    os.environ["USE_MEMENTO"] = "1"
    os.environ["VLLM_BASE_URL"] = args.vllm_url

    print("=" * 70)
    print(" ACE-Memento Experiment on DeepResearcher")
    print("=" * 70)
    print(f" Data dir: {args.data_dir}")
    print(f" Save dir: {args.save_dir}")
    print(f" API: {args.api_provider}")
    print(f" Model: {args.model}")
    print(f" RAE: {args.use_rae}")
    print(f" Failure Memory: {args.use_failure_memory}")
    print("=" * 70)

    # --- Load dữ liệu ---
    print("\n Loading DeepResearcher data...")
    train_data = DeepResearcherProcessor.load_data(
        os.path.join(args.data_dir, "train.jsonl")
    )
    val_data = DeepResearcherProcessor.load_data(
        os.path.join(args.data_dir, "val.jsonl")
    )
    test_data = DeepResearcherProcessor.load_data(
        os.path.join(args.data_dir, "test.jsonl")
    )

    # Kiểm tra dữ liệu
    if not train_data and not test_data:
        print(" No data loaded! Check data_dir path.")
        print(f"   Expected files: {args.data_dir}/{{train,val,test}}.jsonl")
        sys.exit(1)

    # Fallback nếu thiếu val
    if not val_data and test_data:
        val_data = test_data[:len(test_data)//2]
        print(f"  No val set found, using first half of test as val")

    print(f" Train: {len(train_data)} samples")
    print(f" Val: {len(val_data)} samples")
    print(f" Test: {len(test_data)} samples")

    # --- Khởi tạo ACE-Memento ---
    print("\n Initializing ACE-Memento...")
    ace = ACE(
        api_provider=args.api_provider,
        generator_model=args.model,
        reflector_model=args.model,
        curator_model=args.model,
        max_tokens=4096,
        use_rae=args.use_rae,
        rae_top_k=10,
        use_failure_memory=args.use_failure_memory,
        failure_memory_top_k=3,
        memory_jsonl_path=os.path.join(args.save_dir, "case_bank.jsonl"),
        device=args.device  
    )

    # --- Cấu hình training ---
    config = {
        'num_epochs': args.num_epochs,
        'max_num_rounds': 3,
        'curator_frequency': 1,
        'eval_steps': 50,
        'online_eval_frequency': 15,
        'save_steps': 25,
        'playbook_token_budget': 8000,
        'task_name': 'deepresearcher',
        'json_mode': False,
        'no_ground_truth': False,
        'save_dir': args.save_dir,
        'test_workers': args.test_workers,
        'use_bulletpoint_analyzer': False,
        'api_provider': args.api_provider,
        'use_rae': args.use_rae,
        'rae_top_k': 10,
        'use_failure_memory': args.use_failure_memory,
        'failure_memory_top_k': 3,
        'device': args.device,
    }

    # --- Tạo DataProcessor ---
    processor = DeepResearcherProcessor()

    # --- Chạy offline training ---
    print("\n" + "=" * 70)
    print("🏋️  Starting OFFLINE Training + Evaluation")
    print("=" * 70 + "\n")

    results = ace.run(
        mode='offline',
        train_samples=train_data if train_data else None,
        val_samples=val_data if val_data else None,
        test_samples=test_data if test_data else val_data,
        data_processor=processor,
        config=config
    )

    # --- In kết quả cuối cùng ---
    print("\n" + "=" * 70)
    print(" EXPERIMENT COMPLETE!")
    print("=" * 70)

    # In kết quả đẹp
    if "initial_test_results" in results:
        print(f" Initial Test Accuracy: {results['initial_test_results'].get('accuracy', 0):.4f}")
    if "final_test_results" in results:
        print(f" Final Test Accuracy:   {results['final_test_results'].get('accuracy', 0):.4f}")
    if "best_validation_accuracy" in results:
        print(f" Best Validation Acc:   {results['best_validation_accuracy']:.4f}")

    print(f"\n Results saved to: {args.save_dir}")
    print("=" * 70)

    # Lưu kết quả tóm tắt
    summary = {
        "config": vars(args),
        "results": results
    }
    summary_path = os.path.join(args.save_dir, "experiment_summary.json")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f" Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
