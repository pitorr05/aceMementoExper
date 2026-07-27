#!/bin/bash
set -euo pipefail

echo ">>> 1. Clone Main Repo..."
git clone https://github.com/Hieurezdev/ace.git
cd ace/

echo ">>> 2. Install dependencies with uv..."
uv sync
uv add vllm wrapt transformers

cd ..
git clone https://github.com/Hieurezdev/TokenSelectExperiment.git
cd TokenSelectExperiment/

echo ">>> 3. Install dependencies for TokenSelectExperiment..."
uv sync
uv pip install wrapt
uv pip install --python .venv/bin/python torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python flashinfer==0.1.6+cu121torch2.4 --index-url https://flashinfer.ai/whl/cu121/torch2.4
uv pip install --python .venv/bin/python "setuptools<70.0.0"
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python wheel==0.46.3
uv pip install --python .venv/bin/python flash_attn==2.7.0.post2 --no-build-isolation
uv pip install --python .venv/bin/python git+https://github.com/ozeliger/pyairports.git
uv pip install --python .venv/bin/python evaluate==0.4.6
uv pip install --python .venv/bin/python rouge_score==0.1.2 nltk==3.9.3 absl-py==2.4.0

echo ">>> 4. Start benchmark server..."
uv run python benchmark/serve.py \
    --model-path Qwen/Qwen2-7B-Instruct \
    --dp 1 \
    --port 62726 \
    --disable-cuda-graph \
    --disable-regex-jump-forward \
    --disable-radix-cache \
    --max-running-requests 1 \
    --mem-fraction-static 0.85 \
    --context-length 1048576 \
    --sgl-conf-file config/qwen-token-retrieval.yaml &

cd ..
echo ">>> Waiting for benchmark server to be ready..."
until curl -s http://127.0.0.1:62726/health > /dev/null 2>&1; do
    echo "... not ready yet, retrying in 10s"
    sleep 10
done
echo ">>> Server is up!"

cd ace/

echo ">>> 5. Configure local endpoint for ACE..."
export SGLANG_BASE_URL=http://127.0.0.1:62726
export LLM_RETRIES_ON_TIMEOUT=8
export LLM_RETRY_SLEEP_SECONDS=3
export LLM_REQUEST_TIMEOUT_SECONDS=180

echo ">>> 6. Run evaluation..."
uv run python -m eval.finance.run \
    --task_name formula \
    --mode offline \
    --save_path results \
    --api_provider sglang \
    --use_rae \
    --rae_top_k 10 \
    --num_epochs 1 \
    --max_num_rounds 3 \
    --generator_model Qwen/Qwen2-7B-Instruct \
    --reflector_model Qwen/Qwen2-7B-Instruct \
    --curator_model Qwen/Qwen2-7B-Instruct \
    --playbook_token_budget 4000 \
    --max_tokens 2048 \
    --test_workers 1 \
    --seed 42 \
    --eval_steps 50 \
    --save_steps 25

echo ">>> Done!"
