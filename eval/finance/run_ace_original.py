#!/usr/bin/env python3
"""
Run ACE gốc (Princeton) trên Symptom2Disease (hoặc finer / formula).

ACE gốc CHỈ có Playbook (Generator + Reflector + Curator).
KHÔNG có CaseBank, KHÔNG có AMORE, KHÔNG có Socratic Contradiction Resolution.

Khác với run.py (dùng ace_memento.ACE), file này dùng ace.ACE.
"""
import os
import json
import argparse
from datetime import datetime
from .data_processor import DataProcessor
from ace import ACE
from utils import initialize_clients, set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description='ACE gốc - Original ACE System')

    # Task configuration
    parser.add_argument("--task_name", type=str, required=True,
                        help="Name of the task (e.g., 'symptom2disease', 'finer', 'formula')")
    parser.add_argument("--initial_playbook_path", type=str, default=None,
                        help="Path to initial playbook (optional)")
    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "online", "eval_only"],
                        help="Run mode")

    # Model configuration
    parser.add_argument("--api_provider", type=str, default="vllm",
                        choices=["sambanova", "together", "openai", "vllm", "sglang"])
    parser.add_argument("--generator_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--reflector_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--curator_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")

    # Training configuration
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_num_rounds", type=int, default=3)
    parser.add_argument("--curator_frequency", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--online_eval_frequency", type=int, default=15)
    parser.add_argument("--save_steps", type=int, default=50)

    # System configuration
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--playbook_token_budget", type=int, default=80000)
    parser.add_argument("--test_workers", type=int, default=20)

    # Prompt configuration
    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--no_ground_truth", action="store_true")

    # Bulletpoint analyzer configuration
    parser.add_argument("--use_bulletpoint_analyzer", action="store_true")
    parser.add_argument("--bulletpoint_analyzer_threshold", type=float, default=0.90)

    # RAE configuration
    parser.add_argument("--use_rae", action="store_true")
    parser.add_argument("--rae_top_k", type=int, default=10)

    # Failure Memory configuration
    parser.add_argument("--use_failure_memory", action="store_true")
    parser.add_argument("--failure_memory_top_k", type=int, default=3)

    # Adversarial configuration
    parser.add_argument("--use_adversarial", action="store_true")
    parser.add_argument("--adversarial_frequency", type=int, default=10)
    parser.add_argument("--adversarial_model", type=str, default=None)

    # Output configuration
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_data(data_path: str):
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
    processor = DataProcessor(task_name=task_name)

    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None
        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            raise ValueError(f"{mode} mode requires test data in config.")
    else:
        train_samples = load_data(config["train_data"])
        val_samples = load_data(config["val_data"])
        train_samples = processor.process_task_data(train_samples)
        val_samples = processor.process_task_data(val_samples)

        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            test_samples = []

    return train_samples, val_samples, test_samples, processor


def load_initial_playbook(path):
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return None


def main():
    args = parse_args()
    if not args.save_path:
        raise ValueError("--save_path must be specified")

    set_global_seed(args.seed)
    print(f"Using seed: {args.seed}")

    print(f"\n{'='*60}")
    print(f"ACE GỐC (Princeton) — NO CaseBank, NO AMORE")
    print(f"{'='*60}")
    print(f"Task: {args.task_name}")
    print(f"Mode: {args.mode.upper().replace('_', ' ')}")
    print(f"Generator Model: {args.generator_model}")
    print(f"{'='*60}\n")

    # Load config
    with open("./eval/finance/data/sample_config.json", 'r') as f:
        task_config = json.load(f)

    train_samples, val_samples, test_samples, data_processor = preprocess_data(
        args.task_name,
        task_config[args.task_name],
        args.mode
    )

    initial_playbook = load_initial_playbook(args.initial_playbook_path)
    if initial_playbook:
        print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
    else:
        print("Using empty playbook as initial playbook\n")

    # === ACE gốc — CHỈ truyền các tham số mà ace.ACE nhận ===
    ace_system = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=initial_playbook,
        use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
        bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold,
        use_rae=args.use_rae,
        rae_top_k=args.rae_top_k,
        use_failure_memory=args.use_failure_memory,
        failure_memory_top_k=args.failure_memory_top_k,
        use_adversarial=args.use_adversarial,
        adversarial_frequency=args.adversarial_frequency,
        adversarial_model=args.adversarial_model,
    )

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
        'use_rae': args.use_rae,
        'rae_top_k': args.rae_top_k,
        'use_failure_memory': args.use_failure_memory,
        'failure_memory_top_k': args.failure_memory_top_k,
        'api_provider': args.api_provider,
        'use_adversarial': args.use_adversarial,
        'adversarial_frequency': args.adversarial_frequency,
        'seed': args.seed,
    }

    # === ACE gốc: KHÔNG truyền resume_from ===
    results = ace_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=data_processor,
        config=config,
    )


if __name__ == "__main__":
    main()