# Original MCE workflow on the ACE datasets, all roles using Qwen

This entry point calls `run_iteration`, `run_meta_agent`, `run_base_agent`,
workspace setup, skill crossover/history, interface validation and validation
selection from the unchanged MCE source in `third_party/mce`.
It does not call the idea5 bounded-memory or skill-transaction runner.

The original project uses Claude Agent SDK for optimization and DeepSeek/hosted
utilities. Here the Qwen backend replaces that runtime with local file/code tools
and chat-completions calls. Meta/base prompts and orchestration remain upstream.
This is **MCE adapted to Qwen and ACE benchmarks**, not an exact reproduction of
the paper's model configuration or published scores.

## Server setup

In your checkout containing `mce_ace/`, activate the ACE environment:

```bash
source .venv/bin/activate
uv pip install -r requirements-mce-qwen.txt
export VLLM_BASE_URL=http://127.0.0.1:5000/v1
export VLLM_API_KEY=EMPTY
curl -s http://127.0.0.1:5000/v1/models
```

The returned model ID must match `model` in `configs/mce_ace/*.yaml`.
No Claude SDK, Claude subscription or hosted model key is needed for `--backend qwen`.
Default tool transport is JSON over ordinary chat, so the existing server does
not require `--enable-auto-tool-choice` or a tool parser. Optionally set
`tool_transport: native` only on a vLLM server configured for Qwen native tool calls.

## Run each dataset independently

One iteration is one complete training pass, then full validation. These commands
use **one pass**, matching the previous ACE epoch count. There is initial skill
generation; history-based meta skill evolution occurs only if iterations >= 2.

Formula:

```bash
python -u -m mce_ace.run --backend qwen \
  --config configs/mce_ace/formula.yaml \
  --output results/mce_qwen_formula --iterations 1
```

FiNER 0.5:

```bash
python -u -m mce_ace.run --backend qwen \
  --config configs/mce_ace/finer_0.5.yaml \
  --output results/mce_qwen_finer05 --iterations 1
```

Symptom2Disease:

```bash
python -u -m mce_ace.run --backend qwen \
  --config configs/mce_ace/symptom2disease.yaml \
  --output results/mce_qwen_symptom --iterations 1
```

Use a new output directory for each run. To study multiple meta generations,
change `--iterations 1` to `--iterations 3` and use a new output directory.
That means three training passes and more inference calls; it is not the same
training budget as single-epoch ACE. No all-dataset script is installed.

This original MCE runner does not offer the idea5 step-checkpoint resume interface.
It writes completed iteration results as it progresses; a failed run should be
investigated and repeated in a new directory. Do not delete a result directory
while a process is running.

## Protocol and adaptations

- Source revision: `c4b7a7c2ce3ffc4bf4a74c52d2dd8a9a8fb14c30`; source files are
  unchanged and verified with `python -m mce_ace.run --verify-source`.
- Uses existing `sample_config.json`, DataProcessor and ACE Generator prompt/parser.
  Complete train/validation/test splits and file order are retained; no sample caps.
- Formula: 500/300/200; FiNER 0.5: 500/250/441; Symptom: 200/50/212 examples.
- All LLM roles use Qwen3-4B-Instruct-2507, temperature 0, maximum output 4096.
- Batch learning uses 50 examples; validation runs at the end of each complete
  MCE iteration (upstream lifecycle), not at ACE's within-epoch intervals.
- No 32K memory cap or idea5 admission/rollback is imposed on this MCE baseline.
- FiNER validation/test aggregate correct labels over total labels. Other tasks
  use exact legacy example metrics. Evaluation exceptions abort; they do not
  silently reduce the denominator. Upstream aggregate training history still uses
  its batch-size weighting; the final validation/test FiNER metric is label-weighted.
- Meta/base agents can Read, Write, Edit, Glob, Grep, load Skill and run Bash.
  The upstream tool permission callbacks apply to direct file operations. Bash and
  generated Python execute with the server user's permissions, as research code;
  this runtime is not an OS sandbox. No parallel subagent tools are implemented.
- Auxiliary learning `call_llm` is redirected to the same Qwen endpoint. The
  optional hosted embedding helper is replaced with explicitly labeled lexical
  cosine similarity; no separate embedding model is assumed. Both changes are
  recorded baseline adaptations. Exported `get_context` is local file/code logic,
  followed by exactly one Generator call. Training utilities may make extra calls.
- Maximum 30 model turns per meta/base invocation, and upstream validation can
  request up to three repair conversations. Malformed JSON receives feedback;
  exhausting the turn limit or receiving truncated output fails loudly.

## Outputs and debugging

```text
run_config.json                 # data hashes, model/config, source revision, backend
iterations.json                 # completed iteration metrics
final_test_results.json         # full test results at best validation iteration
optimizer_calls.jsonl          # Qwen meta/base prompts, tool requests/results, usage
calls.jsonl                    # Generator calls, responses and usage
workspace/meta_agent/evaluations.json
workspace/meta_agent/skills/iter*/SKILL.md
workspace/iter*_sub*/.claude/skills/learning-context/SKILL.md
workspace/iter*_sub*/context/
workspace/iter*_sub*/interfaces/
workspace/iter*_sub*/data/train.json
```

The model sees training rollouts and previous evaluated skills. Validation labels
and test files are not supplied to optimizer tools. Source datasets outside the
workspace are not granted by direct file tools. These are protocol controls, not
an adversarial isolation guarantee against arbitrary Bash/Python.

Verified locally using a scripted model transport: two upstream iterations, two
batches each, real filesystem tools, generated Python interfaces, original MCE
validation/history, and final test evaluation. No real H100 accuracy is reported.

```bash
python -m unittest discover -s tests -p 'test_mce_qwen.py' -v
```
