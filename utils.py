#!/usr/bin/env python3
import os
import re
import json
import random
import openai
import tiktoken
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()


def set_global_seed(seed: int) -> None:
    """Set global random seed across common libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    # Best-effort torch seeding (optional dependency)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def initialize_clients(api_provider):
    """Initialize separate clients for generator, reflector, and curator"""
    if api_provider == "sambanova":
        # Use SambaNova API
        base_url = "https://api.sambanova.ai/v1"
        api_key = os.getenv('SAMBANOVA_API_KEY', '')
        if not api_key:
            raise ValueError("SambaNova api key not found in environment variables")
    elif api_provider == "together":
        # Use Together API
        base_url = "https://api.together.xyz/v1"
        api_key = os.getenv('TOGETHER_API_KEY', '')
        if not api_key:
            raise ValueError("Together api key not found in environment variables")
    elif api_provider == "openai":
        # Use OpenAI API
        base_url = "https://api.openai.com/v1"
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            raise ValueError("OpenAI api key not found in environment variables")
    elif api_provider == "vllm":
        # Use local vLLM server (OpenAI-compatible)
        base_url = os.getenv('VLLM_BASE_URL', 'http://localhost:8000/v1')
        api_key = os.getenv('VLLM_API_KEY', 'EMPTY')
    elif api_provider == "sglang":
        # Use local SGLang server (native /generate endpoint)
        base_url = os.getenv('SGLANG_BASE_URL', 'http://127.0.0.1:62726')
        generator_client = {"base_url": base_url}
        reflector_client = {"base_url": base_url}
        curator_client = {"base_url": base_url}
        print(f"Using {api_provider} API for all models")
        return generator_client, reflector_client, curator_client
    else:
        raise ValueError((f"Invalid api_provider name: {api_provider}. Must be 'sambanova', 'together', 'openai', 'vllm', or 'sglang'"))

    request_timeout = float(os.getenv('LLM_REQUEST_TIMEOUT_SECONDS', '180'))
        
    generator_client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout)
    reflector_client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout)
    curator_client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout)
    
    print(f"Using {api_provider} API for all models")
    return generator_client, reflector_client, curator_client

def get_section_slug(section_name):
    """Convert section name to slug format (3-5 chars)"""
    # Common section mappings - updated to match original sections
    slug_map = {
        "financial_strategies_and_insights": "fin",
        "formulas_and_calculations": "calc",
        "code_snippets_and_templates": "code",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics": "prob",
        "context_clues_and_indicators": "ctx",
        "others": "misc",
        "meta_strategies": "meta"
    }
    
    # Clean and convert to snake_case
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and")
    
    if clean_name in slug_map:
        return slug_map[clean_name]
    
    # Generate slug from first letters
    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    else:
        return "".join(w[0] for w in words[:5])

def extract_boxed_content(text):
    """Helper function to extract content from \\boxed{} format"""
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)
    if not match:
        return None
    
    start = match.end() - 1  # Position of opening brace
    brace_count = 0
    i = start
    
    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start + 1:i]  # Content between braces
        i += 1
    return None

def extract_answer(response):
    """Extract final answer from model response"""
    try:
        # First try JSON parsing
        parsed = json.loads(response)
        answer = str(parsed.get("final_answer", "No final answer found"))
        return answer  
            
    except (json.JSONDecodeError, KeyError, AttributeError):
        # JSON parsing failed, use fallback logic
        matches = re.findall(r"Finish\[(.*?)\]", response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Try to get final answer from JSON style response with regex matching 
        # Try double quotes first
        matches = re.findall(r'"final_answer"\s*:\s*"([^"]*)"', response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Try single quotes
        matches = re.findall(r"'final_answer'\s*:\s*'([^']*)'", response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Handle JSON format without quotes (for simple expressions)
        matches = re.findall(r'[\'"]final_answer[\'"]\s*:\s*([^,}]+)', response)
        if matches:
            answer = matches[-1].strip()
            # Clean up trailing characters
            answer = re.sub(r'[,}]*$', '', answer)
            return answer
        
        # Fallback for "The final answer is: X" pattern with boxed
        final_answer_pattern = r'[Tt]he final answer is:?\s*\$?\\boxed\{'
        match = re.search(final_answer_pattern, response)
        if match:
            # Extract boxed content starting from this match
            remaining_text = response[match.start():]
            boxed_content = extract_boxed_content(remaining_text)
            if boxed_content:
                return boxed_content
        
        # More general pattern for "final answer is X"
        matches = re.findall(r'[Tt]he final answer is:?\s*([^\n.]+)', response)
        if matches:
            answer = matches[-1].strip()
            # Clean up common formatting
            answer = re.sub(r'^\$?\\boxed\{([^}]+)\}\$?$', r'\1', answer)
            answer = answer.replace('$', '').strip()
            if answer:
                return answer
        
        return "No final answer found"
    
enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(prompt: str) -> int:
    return len(enc.encode(prompt))


def evaluate_single_test_sample(args_tuple, data_processor) -> Tuple[Dict, str]:
    """
    Evaluate a single test sample - task-agnostic implementation.

    Args:
        args_tuple: Tuple of (index, task_dict, generator, playbook, max_tokens,
                             log_dir, use_json_mode, retriever)
        data_processor: DataProcessor instance with answer_is_correct method
    """
    (i, task_dict, generator, playbook, max_tokens, log_dir, use_json_mode, retriever) = args_tuple
    try:
        context = task_dict["context"]
        question = task_dict["question"]
        target = task_dict["target"]

        gen_response, bullet_ids, call_info = generator.generate(
            question=question,
            playbook=playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"test_eval_{i}",
            log_dir=log_dir,
            retriever=retriever
        )

        final_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(final_answer, target)

        return {
            "index": i,
            "final_answer": final_answer,
            "target": target,
            "is_correct": is_correct,
            "success": True
        }, None

    except Exception as e:
        return None, f"Error evaluating sample {i}: {type(e).__name__}: {str(e)}"


def evaluate_test_set(data_processor, generator, playbook, test_samples,
                      max_tokens=4096, log_dir=None, max_workers=20,
                      use_json_mode=False, retriever=None) -> Tuple[Dict, Dict]:
    """
    Parallel evaluation of test set - task-agnostic implementation.

    Args:
        data_processor: DataProcessor instance with answer_is_correct and evaluate_accuracy methods
        generator: Generator instance
        playbook: Current playbook string
        test_samples: List of test samples
        max_tokens: Max tokens for generation
        log_dir: Directory for logs
        max_workers: Number of parallel workers
        use_json_mode: Whether to use JSON mode
        retriever: Optional PlaybookRetriever for RAE (Top-K bullet retrieval at generation time)

    Returns:
        Tuple of (results_dict, error_logs_dict)
    """
    print(f"\n{'='*40}")
    print(f"EVALUATING TEST SET - {len(test_samples)} samples, {max_workers} workers")
    print(f"{'='*40}")

    args_list = [
        (i, sample, generator, playbook, max_tokens, log_dir, use_json_mode, retriever)
        for i, sample in enumerate(test_samples)
    ]

    results = {
        "correct": 0, "total": 0, "no_answer": 0,
        "answers": [], "targets": [], "errors": []
    }

    # Use a wrapper to pass data_processor to the evaluation function
    def eval_wrapper(args_tuple):
        return evaluate_single_test_sample(args_tuple, data_processor)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {
            executor.submit(eval_wrapper, args): args 
            for args in args_list
        }

        for i, future in enumerate(as_completed(future_to_args), 1):
            result, error = future.result()
            
            if error:
                print(error)
                continue

            if result and result["success"]:
                results["correct"] += (1 if result["is_correct"] else 0)
                results["total"] += 1
                results["answers"].append(result["final_answer"])
                results["targets"].append(result["target"])
                
                if not result["is_correct"]:
                    results["errors"].append({
                        "index": result["index"],
                        "prediction": result["final_answer"],
                        "ground_truth": result["target"]
                    })
                
                if result["final_answer"] == "No final answer found":
                    results["no_answer"] += 1

            if i % 50 == 0:
                curr_acc = results["correct"] / results["total"] if results["total"] > 0 else 0
                print(f"Progress: {i}/{len(args_list)}, Accuracy: {curr_acc:.3f}")
    
    if results["answers"] and results["targets"]:
        accuracy = data_processor.evaluate_accuracy(results["answers"], results["targets"])
        
        final_results = {
            "accuracy": accuracy,
            "correct": results["correct"],
            "total": results["total"],
            "no_answer": results["no_answer"]
        }
        
        error_logs = {
            "accuracy": accuracy,
            "errors": results["errors"]
        }
        
        print(f"\n📊 Final Accuracy: {accuracy:.3f} ({results['correct']}/{results['total']})")
    else:
        results = {"accuracy": 0.0, "correct": 0, "total": 0}
        error_logs = {}
        print(f"\n📊 No valid results!")
        
    return final_results, error_logs


def calculate_llm_statistics(log_dir: str) -> Dict[str, Any]:
    """Calculate LLM latency and prompt/completion token speeds from JSON logs in log_dir"""
    import os
    import json
    
    total_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_call_time = 0.0
    total_ttft = 0.0
    ttft_count = 0
    total_tpot = 0.0
    tpot_count = 0
    
    if not log_dir or not os.path.exists(log_dir):
        return {
            "total_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_call_time": 0.0,
            "avg_latency": 0.0,
            "avg_prompt_tokens_per_sec": 0.0,
            "avg_completion_tokens_per_sec": 0.0,
            "avg_total_tokens_per_sec": 0.0,
            "avg_ttft": 0.0,
            "avg_tpot": 0.0
        }
        
    for filename in os.listdir(log_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(log_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    call_info = json.load(f)
                
                # Check for LLM call time fields
                call_time = call_info.get("call_time") or call_info.get("total_time") or 0.0
                if call_time <= 0:
                    continue
                    
                prompt_tokens = call_info.get("prompt_num_tokens")
                if prompt_tokens is None:
                    prompt = call_info.get("prompt", "")
                    prompt_tokens = count_tokens(prompt)
                    
                completion_tokens = call_info.get("response_num_tokens")
                if completion_tokens is None:
                    response = call_info.get("response", "")
                    completion_tokens = count_tokens(response)
                    
                total_calls += 1
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_call_time += call_time
                
                ttft = call_info.get("ttft")
                if ttft is not None:
                    total_ttft += ttft
                    ttft_count += 1
                    
                tpot = call_info.get("tpot")
                if tpot is not None:
                    total_tpot += tpot
                    tpot_count += 1
            except Exception as e:
                pass
                
    avg_latency = total_call_time / total_calls if total_calls > 0 else 0.0
    avg_prompt_tokens_per_sec = total_prompt_tokens / total_call_time if total_call_time > 0 else 0.0
    avg_completion_tokens_per_sec = total_completion_tokens / total_call_time if total_call_time > 0 else 0.0
    avg_total_tokens_per_sec = (total_prompt_tokens + total_completion_tokens) / total_call_time if total_call_time > 0 else 0.0
    avg_ttft = total_ttft / ttft_count if ttft_count > 0 else 0.0
    avg_tpot = total_tpot / tpot_count if tpot_count > 0 else 0.0
    
    return {
        "total_calls": total_calls,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_call_time": total_call_time,
        "avg_latency": avg_latency,
        "avg_prompt_tokens_per_sec": avg_prompt_tokens_per_sec,
        "avg_completion_tokens_per_sec": avg_completion_tokens_per_sec,
        "avg_total_tokens_per_sec": avg_total_tokens_per_sec,
        "avg_ttft": avg_ttft,
        "avg_tpot": avg_tpot
    }


def print_eval_statistics_banner(
    total_running_time: float, 
    avg_run_time: float, 
    llm_stats: Dict[str, Any], 
    avg_vram: float = None, 
    gpu_averages: Dict[str, float] = None
) -> None:
    """Print a clean metrics banner with the speed and duration of the evaluation"""
    print(f"\n{'='*60}")
    print(f"EVALUATION METRICS & PERFORMANCE")
    print(f"{'='*60}")
    print(f"Total Running Time:              {total_running_time:.2f} seconds")
    print(f"Average Running Time per Sample:  {avg_run_time:.2f} seconds")
    
    if avg_vram is not None and avg_vram > 0.0:
        if gpu_averages and len(gpu_averages) > 1:
            gpu_str = ", ".join([f"{k}: {v:.1f} MB" for k, v in gpu_averages.items()])
            print(f"Average VRAM Usage:              {avg_vram:.1f} MB ({gpu_str})")
        else:
            print(f"Average VRAM Usage:              {avg_vram:.1f} MB")
            
    print(f"{'-'*60}")
    print(f"Total LLM Calls:                 {llm_stats['total_calls']}")
    print(f"Total LLM Call Duration:         {llm_stats['total_call_time']:.2f} seconds")
    print(f"Average LLM Call Latency:        {llm_stats['avg_latency']:.2f} seconds")
    
    if llm_stats.get("avg_ttft", 0.0) > 0.0 or llm_stats.get("avg_tpot", 0.0) > 0.0:
        print(f"Average TTFT:                    {llm_stats.get('avg_ttft', 0.0):.4f} seconds")
        print(f"Average TPOT:                    {llm_stats.get('avg_tpot', 0.0):.4f} seconds/token")
        
    print(f"{'-'*60}")
    print(f"Total Prompt (Request) Tokens:   {llm_stats['total_prompt_tokens']}")
    print(f"Total Completion (Resp) Tokens:  {llm_stats['total_completion_tokens']}")
    print(f"{'-'*60}")
    print(f"Avg Request (Prompt) Speed:      {llm_stats['avg_prompt_tokens_per_sec']:.2f} tokens/second")
    print(f"Avg Response (Completion) Speed: {llm_stats['avg_completion_tokens_per_sec']:.2f} tokens/second")
    print(f"Avg Total Token Throughput:      {llm_stats['avg_total_tokens_per_sec']:.2f} tokens/second")
    print(f"{'='*60}\n")


class VRAMMonitor:
    """Background GPU memory usage (VRAM) monitor using nvidia-smi"""
    def __init__(self, interval=1.0):
        self.interval = interval
        self.vram_logs = []
        import threading
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.vram_logs = []
        self.stop_event.clear()
        import threading
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        if self.thread:
            self.stop_event.set()
            self.thread.join()
            self.thread = None

    def _monitor(self):
        import subprocess
        import time
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                gpu_usages = []
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        gpu_usages.append(int(line.strip()))
                if gpu_usages:
                    self.vram_logs.append(gpu_usages)
            except Exception:
                pass
            time.sleep(self.interval)

    def get_average_vram(self):
        if not self.vram_logs:
            return 0.0, {}
        
        num_gpus = len(self.vram_logs[0])
        gpu_averages = {}
        for i in range(num_gpus):
            usages = [log[i] for log in self.vram_logs if i < len(log)]
            if usages:
                gpu_averages[f"GPU {i}"] = sum(usages) / len(usages)
        
        overall_avg = sum(gpu_averages.values()) / len(gpu_averages) if gpu_averages else 0.0
        return overall_avg, gpu_averages
