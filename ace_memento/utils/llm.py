import os
import time
import random
import json
from datetime import datetime
from urllib import request, error
import openai
from typing import Dict, List, Tuple, Optional, Any

# Simple logger adaptation for ace_memento
def log_llm_call(log_dir: str, call_info: Dict[str, Any]) -> None:
    if not log_dir:
        return
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{call_info['role']}_{call_info['call_id']}_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)
    call_info['timestamp'] = timestamp
    call_info['datetime'] = datetime.now().isoformat()
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(call_info, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to log LLM call: {e}")

def initialize_clients(api_provider: str) -> Tuple[openai.OpenAI, openai.OpenAI, openai.OpenAI]:
    """Initialize separate clients for generator, reflector, and curator"""
    if api_provider == "sambanova":
        base_url = "https://api.sambanova.ai/v1"
        api_key = os.getenv('SAMBANOVA_API_KEY', '')
        if not api_key:
            raise ValueError("SambaNova api key not found in environment variables")
    elif api_provider == "together":
        base_url = "https://api.together.xyz/v1"
        api_key = os.getenv('TOGETHER_API_KEY', '')
        if not api_key:
            raise ValueError("Together api key not found in environment variables")
    elif api_provider == "openai":
        base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            raise ValueError("OpenAI api key not found in environment variables")
    elif api_provider == "vllm":
        base_url = os.getenv('VLLM_BASE_URL', 'http://localhost:8000/v1')
        api_key = os.getenv('VLLM_API_KEY', 'EMPTY')
    elif api_provider == "sglang":
        base_url = os.getenv('SGLANG_BASE_URL', 'http://127.0.0.1:62726')
        generator_client = {"base_url": base_url}
        reflector_client = {"base_url": base_url}
        curator_client = {"base_url": base_url}
        print(f"Using {api_provider} API for all models")
        return generator_client, reflector_client, curator_client
    else:
        raise ValueError(f"Invalid api_provider name: {api_provider}. Must be 'sambanova', 'together', 'openai', 'vllm', or 'sglang'")

    request_timeout = float(os.getenv('LLM_REQUEST_TIMEOUT_SECONDS', '180'))
    generator_client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout)
    reflector_client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout)
    curator_client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout)
    
    print(f"Using {api_provider} API for all models")
    return generator_client, reflector_client, curator_client


def timed_llm_call(
    client: Any,
    api_provider: str,
    model: str,
    prompt: str,
    role: str,
    call_id: str,
    max_tokens: int = 4096,
    log_dir: Optional[str] = None,
    sleep_seconds: Optional[float] = None,
    retries_on_timeout: Optional[int] = None,
    attempt: int = 1,
    use_json_mode: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    Make a timed LLM call with error handling and retry logic.
    """
    start_time = time.time()
    prompt_time = time.time()
    if sleep_seconds is None:
        sleep_seconds = float(os.getenv("LLM_RETRY_SLEEP_SECONDS", "3"))
    if retries_on_timeout is None:
        retries_on_timeout = int(os.getenv("LLM_RETRIES_ON_TIMEOUT", "8"))
    
    print(f"[{role.upper()}] Starting call {call_id}...")
    
    while True:
        try:
            active_client = client

            if api_provider == "sglang":
                base_url = active_client.get("base_url") if isinstance(active_client, dict) else os.getenv('SGLANG_BASE_URL', 'http://127.0.0.1:62726')
                endpoint = f"{base_url.rstrip('/')}/generate"
                payload = {
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": max_tokens,
                    },
                }
                
                track_ttft = os.getenv("TRACK_TTFT", "0") == "1"
                track_tpot = os.getenv("TRACK_TPOT", "0") == "1"
                use_streaming = track_ttft or track_tpot
                ttft = None
                tpot = None
                prompt_tokens = None
                completion_tokens = None
                
                if use_streaming:
                    try:
                        payload["stream"] = True
                        call_start = time.time()
                        req = request.Request(
                            endpoint,
                            data=json.dumps(payload).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        timeout_seconds = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "180"))
                        first_token_time = None
                        chunks = []
                        
                        with request.urlopen(req, timeout=timeout_seconds) as resp:
                            for line in resp:
                                line = line.decode("utf-8").strip()
                                if not line:
                                    continue
                                if line.startswith("data:"):
                                    data_str = line[5:].strip()
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data_json = json.loads(data_str)
                                        text_delta = data_json.get("text")
                                        if text_delta:
                                            if first_token_time is None:
                                                first_token_time = time.time()
                                            chunks.append(text_delta)
                                    except Exception:
                                        pass
                        call_end = time.time()
                        response_content = "".join(chunks)
                        
                        total_time = call_end - call_start
                        ttft_val = (first_token_time - call_start) if first_token_time else total_time
                        if track_ttft:
                            ttft = ttft_val
                        
                        from utils import count_tokens
                        prompt_tokens = count_tokens(prompt)
                        completion_tokens = count_tokens(response_content)
                        
                        if track_tpot:
                            generation_time = max(0.0, total_time - ttft_val)
                            tpot = (generation_time / completion_tokens) if (completion_tokens and completion_tokens > 0) else 0.0
                    except Exception as e:
                        print(f"[Warning] SGLang streaming failed, falling back to non-stream: {e}")
                        use_streaming = False
                        
                if not use_streaming:
                    call_start = time.time()
                    req = request.Request(
                        endpoint,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    timeout_seconds = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "180"))
                    with request.urlopen(req, timeout=timeout_seconds) as resp:
                        raw = resp.read().decode("utf-8")
                    call_end = time.time()

                    response_json = json.loads(raw)
                    if isinstance(response_json, dict):
                        if isinstance(response_json.get("text"), list) and response_json.get("text"):
                            response_content = response_json["text"][0]
                        else:
                            response_content = response_json.get("text") or response_json.get("generated_text")
                    else:
                        response_content = None

                if response_content is None:
                    raise Exception("Empty response from API")

                response_time = time.time()
                total_time = response_time - start_time
                call_info = {
                    "role": role,
                    "call_id": call_id,
                    "model": model,
                    "prompt": prompt,
                    "response": response_content,
                    "prompt_time": prompt_time - start_time,
                    "response_time": response_time - prompt_time,
                    "total_time": total_time,
                    "call_time": call_end - call_start,
                    "prompt_length": len(prompt),
                    "response_length": len(response_content),
                    "prompt_num_tokens": prompt_tokens,
                    "response_num_tokens": completion_tokens,
                    "ttft": ttft,
                    "tpot": tpot
                }

                print(f"[{role.upper()}] Call {call_id} completed in {total_time:.2f}s")
                if log_dir:
                    log_llm_call(log_dir, call_info)
                return response_content, call_info

            # Standard OpenAI / vLLM / SambaNova / Together client
            if api_provider == "openai":
                max_tokens_key = "max_completion_tokens"
            else:
                max_tokens_key = "max_tokens"

            api_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                max_tokens_key: max_tokens
            }
            
            # Add JSON mode if requested
            if use_json_mode:
                api_params["response_format"] = {"type": "json_object"}
                
            track_ttft = os.getenv("TRACK_TTFT", "0") == "1"
            track_tpot = os.getenv("TRACK_TPOT", "0") == "1"
            use_streaming = track_ttft or track_tpot
            ttft = None
            tpot = None
            prompt_tokens = None
            completion_tokens = None
            
            if use_streaming:
                api_params["stream"] = True
                if api_provider in ["openai", "vllm", "sambanova", "together"]:
                    api_params["stream_options"] = {"include_usage": True}
                    
                call_start = time.time()
                try:
                    response_stream = active_client.chat.completions.create(**api_params)
                except Exception as stream_err:
                    if "stream_options" in api_params:
                        del api_params["stream_options"]
                        response_stream = active_client.chat.completions.create(**api_params)
                    else:
                        raise stream_err
                        
                first_token_time = None
                chunks = []
                usage = None
                
                for chunk in response_stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if first_token_time is None and delta.content:
                            first_token_time = time.time()
                        if delta.content:
                            chunks.append(delta.content)
                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage = chunk.usage
                        
                call_end = time.time()
                response_content = "".join(chunks)
                
                total_time = call_end - call_start
                ttft_val = (first_token_time - call_start) if first_token_time else total_time
                if track_ttft:
                    ttft = ttft_val
                
                prompt_tokens = usage.prompt_tokens if usage else None
                completion_tokens = usage.completion_tokens if usage else None
                
                if prompt_tokens is None:
                    from utils import count_tokens
                    prompt_tokens = count_tokens(prompt)
                if completion_tokens is None:
                    from utils import count_tokens
                    completion_tokens = count_tokens(response_content)
                    
                if track_tpot:
                    generation_time = max(0.0, total_time - ttft_val)
                    tpot = (generation_time / completion_tokens) if (completion_tokens and completion_tokens > 0) else 0.0
                
            else:
                call_start = time.time()
                response = active_client.chat.completions.create(**api_params)
                call_end = time.time()
                
                if not response or not response.choices or len(response.choices) == 0:
                    raise Exception("Empty response from API")
                
                response_content = response.choices[0].message.content
                if response_content is None:
                    raise Exception("API returned None content")
                    
                prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else None
                completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else None
            
            response_time = time.time()
            total_time = response_time - start_time
            
            call_info = {
                "role": role,
                "call_id": call_id,
                "model": model,
                "prompt": prompt,
                "response": response_content,
                "prompt_time": prompt_time - start_time,
                "response_time": response_time - prompt_time,
                "total_time": total_time,
                "call_time": call_end - call_start,
                "prompt_length": len(prompt),
                "response_length": len(response_content),
                "prompt_num_tokens": prompt_tokens,
                "response_num_tokens": completion_tokens,
                "ttft": ttft,
                "tpot": tpot
            }
            
            print(f"[{role.upper()}] Call {call_id} completed in {total_time:.2f}s")
            
            if log_dir:
                log_llm_call(log_dir, call_info)
            
            return response_content, call_info
            
        except Exception as e:
            is_timeout = any(k in str(e).lower() for k in ["timeout", "timed out", "connection"])
            is_rate_limit = any(k in str(e).lower() for k in ["rate limit", "429", "rate_limit_exceeded"])
            is_server_error = any(k in str(e).lower() for k in ["500 internal server error", "internal server error", "502 bad gateway", "503 service unavailable"])
            is_empty_response = "empty response" in str(e).lower() or "api returned none content" in str(e).lower()
            
            if (is_timeout or is_rate_limit or is_server_error or is_empty_response) and attempt < retries_on_timeout:
                attempt += 1
                base_sleep = sleep_seconds * 2 if is_rate_limit else sleep_seconds
                jitter = random.uniform(0.5, 1.5)
                sleep_time = base_sleep * jitter
                print(f"[{role.upper()}] Call {call_id} failed ({e}), sleeping {sleep_time:.1f}s then retrying ({attempt}/{retries_on_timeout})...")
                time.sleep(sleep_time)
                continue
            
            error_time = time.time()
            call_info = {
                "role": role,
                "call_id": call_id,
                "model": model,
                "prompt": prompt,
                "error": str(e),
                "total_time": error_time - start_time,
                "prompt_length": len(prompt),
                "attempt": attempt,
            }
            print(f"[{role.upper()}] Call {call_id} failed after {error_time - start_time:.2f}s: {e}")
            if log_dir:
                log_llm_call(log_dir, call_info)
            raise e
