#!/usr/bin/env python3
"""
Standalone runner for ORIGINAL ACE (no Memento, no CaseBank, no Hybrid Search).

Supports both tasks:
    - formula    (numeric answers)
    - finer_0.5  (US GAAP tags)

Usage examples:
    # Formula (VLLM port 8000)
    VLLM_BASE_URL="http://localhost:8000/v1" \
    python -m eval.finance.run_ace \
        --task_name formula \
        --mode offline \
        --save_path ./results/ace_Goc_formula_ \
        --api_provider vllm \
        --generator_model Qwen/Qwen3-4B-Instruct-2507 \
        --reflector_model Qwen/Qwen3-4B-Instruct-2507 \
        --curator_model Qwen/Qwen3-4B-Instruct-2507 \
        --num_epochs 1 --max_tokens 4096 --test_workers 20 --seed 42

    # Finer 0.5 (VLLM port 8000)
    VLLM_BASE_URL="http://localhost:8000/v1" \
    python -m eval.finance.run_ace \
        --task_name finer_0.5 \
        --mode offline \
        --save_path ./results/ace_Goc_finer05_ \
        --api_provider vllm \
        --generator_model Qwen/Qwen3-4B-Instruct-2507 \
        --reflector_model Qwen/Qwen3-4B-Instruct-2507 \
        --curator_model Qwen/Qwen3-4B-Instruct-2507 \
        --num_epochs 1 --max_tokens 4096 --eval_steps 50 --save_steps 50 \
        --test_workers 10 --seed 42
"""

import os
import json
import argparse
from datetime import datetime
from .data_processor import DataProcessor

# ✅ CHỈ import ACE gốc — KHÔNG import ace_memento
from ace import ACE
from utils import initialize_clients, set_global_seed


# ============================================================================
# ARGUMENT PARSING (chỉ tham số của ACE gốc)
# ============================================================================
def parse_args():
    """Parse command line arguments (ACE original only)."""
    parser = argparse.ArgumentParser(
        description='ACE System (ORIGINAL - No Memento) - Standalone Runner'
    )

    # ---- Task configuration ----
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["formula", "finer", "finer_0.5", "finer_500"],
                        help="Task name (must match key in sample_config.json)")
    parser.add_argument("--initial_playbook_path", type=str, default=None,
                        help="Path to initial playbook (optional)")
    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "online", "eval_only"],
                        help="Run mode")

    # ---- Model configuration ----
    parser.add_argument("--api_provider", type=str, default="sambanova",
                        choices=["sambanova", "together", "openai", "vllm", "sglang"],
                        help="API provider")
    parser.add_argument("--generator_model", type=str, default="DeepSeek-V3.1")
    parser.add_argument("--reflector_model", type=str, default="DeepSeek-V3.1")
    parser.add_argument("--curator_model", type=str, default="DeepSeek-V3.1")

    # ---- Training configuration ----
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_num_rounds", type=int, default=3)
    parser.add_argument("--curator_frequency", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--online_eval_frequency", type=int, default=15)
    parser.add_argument("--save_steps", type=int, default=50)

    # ---- System configuration ----
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--playbook_token_budget", type=int, default=80000)
    parser.add_argument("--test_workers", type=int, default=20)

    # ---- Prompt configuration ----
    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--no_ground_truth", action="store_true")

    # ---- Bulletpoint analyzer ----
    parser.add_argument("--use_bulletpoint_analyzer", action="store_true")
    parser.add_argument("--bulletpoint_analyzer_threshold", type=float, default=0.90)

    # ---- Output ----
    parser.add_argument("--save_path", type=str, default=None,
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data(data_path: str):
    """Load data from a JSONL file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"Loaded {len(data)} samples from {data_path}")
    return data


def preprocess_data(task_name, config, mode):
    """Load and preprocess data for formula / finer / finer_0.5."""
    processor = DataProcessor(task_name=task_name)

    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None

        if "test_data" not in config:
            raise ValueError(f"{mode} mode requires test_data in config.")

        test_samples = load_data(config["test_data"])
        test_samples = processor.process_task_data(test_samples)

        if mode == "online":
            print(f"Online mode: Training and testing on {len(test_samples)} examples")
        else:
            print(f"Eval only mode: Testing on {len(test_samples)} examples")

    else:  # offline
        train_samples = load_data(config["train_data"])
        val_samples = load_data(config["val_data"])
        train_samples = processor.process_task_data(train_samples)
        val_samples = processor.process_task_data(val_samples)

        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            test_samples = []

        print(f"Offline mode: Training on {len(train_samples)} examples, "
              f"validating on {len(val_samples)}, testing on {len(test_samples)}")

    return train_samples, val_samples, test_samples, processor


def load_initial_playbook(path):
    """Load initial playbook if provided."""
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return None


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main execution function for ACE original."""
    args = parse_args()

    if not args.save_path:
        raise ValueError("--save_path must be specified")

    set_global_seed(args.seed)
    print(f"Using seed: {args.seed}")

    # ---- Print banner ----
    print(f"\n{'='*60}")
    print(f"ACE SYSTEM (ORIGINAL - No Memento)")
    print(f"{'='*60}")
    print(f"Task: {args.task_name}")
    print(f"Mode: {args.mode.upper().replace('_', ' ')}")
    print(f"Generator Model: {args.generator_model}")
    print(f"Reflector Model: {args.reflector_model}")
    print(f"Curator Model: {args.curator_model}")
    print(f"{'='*60}\n")

    # ---- Load data ----
    config_path = "./eval/finance/data/sample_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        task_config = json.load(f)

    if args.task_name not in task_config:
        raise ValueError(
            f"Task '{args.task_name}' not found in {config_path}. "
            f"Available: {list(task_config.keys())}"
        )

    train_samples, val_samples, test_samples, data_processor = preprocess_data(
        args.task_name,
        task_config[args.task_name],
        args.mode
    )

    # ---- Load initial playbook ----
    initial_playbook = load_initial_playbook(args.initial_playbook_path)
    if initial_playbook:
        print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
    else:
        print("Using empty playbook as initial playbook\n")

    # ---- Create ACE original (chỉ tham số cơ bản) ----
    ace_system = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=initial_playbook,
        use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
        bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold,
    )

    # ---- Prepare config (không có tham số Memento) ----
    config = {
        'num_epochs': args.num_epochs,
        'max_num_rounds': args.max_num_rounds,
        'curator_frequency': args.curator_frequency,
        'eval_steps': args.eval_steps,
        'online_eval_frequency': args.online_eval_frequency,
        'save_steps': args.save_steps,
        'playbook_token_budget': args.playbook_token_budget,
        'task_name': args.task_name,
        'mode': args.mode,
        'json_mode': args.json_mode,
        'no_ground_truth': args.no_ground_truth,
        'save_dir': args.save_path,
        'test_workers': args.test_workers,
        'initial_playbook_path': args.initial_playbook_path,
        'use_bulletpoint_analyzer': args.use_bulletpoint_analyzer,
        'bulletpoint_analyzer_threshold': args.bulletpoint_analyzer_threshold,
        'api_provider': args.api_provider,
        'seed': args.seed,
    }

    if args.mode == "eval_only" and test_samples:
        test_samples = test_samples[:50]
        print(f">>> [INFO] Limited test samples to first 50 for quick verification")

    # ---- Run (KHÔNG truyền resume_from) ----
    results = ace_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=data_processor,
        config=config,
    )

    print(f"\n{'='*60}")
    print(f"ACE ORIGINAL - RUN COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
