export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=1 uv run vllm serve "Qwen/Qwen3-4B-Instruct-2507" \
    --dtype auto \
    --gpu-memory-utilization 0.95 \
    --max-model-len 262144 \
    --host 0.0.0.0 \
    --port 5000 \
    --trust-remote-code 

USE_MEMENTO=1 VLLM_BASE_URL=http://localhost:5000/v1 uv run python -m eval.finance.run \
    --task_name finer_0.5 \
    --mode offline \
    --save_path results \
    --api_provider vllm \
    --num_epochs 1 \
    --max_num_rounds 3 \
    --generator_model Qwen/Qwen3-4B-Instruct-2507 \
    --reflector_model Qwen/Qwen3-4B-Instruct-2507 \
    --curator_model Qwen/Qwen3-4B-Instruct-2507 \
    --playbook_token_budget 4000 \
    --max_tokens 2048 \
    --test_workers 5 \
    --seed 42 \
    --eval_steps 50 \
    --save_steps 25