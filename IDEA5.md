# Idea5: evidence-guided Playbook refinement

Independent offline runner for Qwen3-4B-Instruct-2507, based on `origin/idea4`
(`d5e88fc`). The legacy ACE and AMORE implementations are retained.
This is a research implementation, not a demonstrated accuracy improvement or
a claim of established novelty.

## Run one dataset

From the root of this checkout on the machine with access to your vLLM endpoint:

```bash
python -m pip install -r requirements-idea5.txt
export VLLM_BASE_URL=http://localhost:5000/v1
```

PowerShell equivalent for the endpoint: `$env:VLLM_BASE_URL = 'http://localhost:5000/v1'`.
The endpoint must serve `Qwen/Qwen3-4B-Instruct-2507`. No paid API or Claude SDK is required.
The endpoint's maximum context must accommodate the compiled Playbook, task and
output; 80,000 below is the legacy cl100k token budget, not a Qwen tokenizer count.

FiNER (the previously used `finer_0.5` variant):

```bash
python -m idea5.run --config configs/idea5/finer_0.5.yaml
```

Formula:

```bash
python -m idea5.run --config configs/idea5/formula.yaml
```

Symptom2Disease:

```bash
python -m idea5.run --config configs/idea5/symptom2disease.yaml
```

Each command runs only its dataset. Default output is `results/idea5_<task>`;
an existing directory is rejected. Use `--output results/my_run` for another run.
There is no all-dataset launcher or automatic baseline sweep.

Inspect paths, hashes and sample counts without a model request:

```bash
python -m idea5.run --config configs/idea5/formula.yaml --check-data
```

Resume the same config and dataset from the last atomic step checkpoint:

```bash
python -m idea5.run --config configs/idea5/formula.yaml --resume results/idea5_formula
```

API/transport errors abort instead of silently skipping evaluation samples.
The interrupted step is retried after resume; calls already spent on it remain
in the call log. Completed training is not repeated. Resuming a completed run
evaluates the selected checkpoint on test again. Do not run two writers against
the same output directory. For exact reproduction, resume on the same revision
and model server settings recorded at run creation.

## Learning workflow

1. Compile the current rules and make one Generator call for the training task.
2. Score with the unchanged dataset evaluator. Persist question, context, raw
   output, parsed answer, rule references and the verified training target.
3. Reflect once into a pending lesson. Reflections are proposals, not verified facts.
4. Every 25 training examples (also at the end), identify a recent failure and
   a target rule: cited rule first, lexical input/rule similarity as fallback.
5. Select up to 16 observed training episodes: anchor, random successful guards
   (up to four), then nearest input neighbors. Similarity includes context, not
   just the shared benchmark instruction. This is a diagnostic heuristic, not
   causal attribution of a failure to a rule.
6. The meta-agent proposes up to two alternative edits. `add` introduces one
   rule; `condition` changes only an existing applicability condition; `split`
   replaces one rule with two conditional rules. `patches: []` keeps state unchanged.
7. Evaluate parent and candidates on identical replay episodes. A candidate must
   strictly improve the dataset metric without reducing the guard-group metric.
   Ties retain the parent. Guard checks are aggregate, not per-example guarantees.
8. Commit the winning edit, or retain pending lessons. Select the final snapshot
   using full validation; freeze all memory writes during validation and test.

Rules contain `id`, `group`, `condition`, `guidance`, evidence IDs, parent and step.
The deterministic compiler groups them and includes their conditions in the
Generator context. It does not make an extra model call or route away rules.
No test labels, generated answer tables, or task-specific percentage heuristics
are inserted by the implementation. Conditions are proposed in natural language;
schema validation cannot prove their semantic correctness or generalization.

The two-level design is inspired by MCE, but v1 evolves a bounded structured
Playbook rather than generating arbitrary Python or evolving unrestricted skills.
The evidence-refinement analogy comes from counterexample-guided abstraction
refinement; the finite replay checker offers no formal correctness guarantee.
Episode evidence adapts the earlier memory idea, with verified targets stored
separately from generated output. This is not an unchanged MCE implementation.

## Benchmark compatibility

The actual `origin/idea4` `eval/finance/data_processor.py` handles preprocessing
and scoring; `utils.extract_answer` handles Generator output parsing. The
Generator uses the original ACE template in `ace/prompts/generator.py` with an
empty reflection field. It does not use the AMORE Case Memory template, so
old AMORE runs can differ in more than the memory mechanism.

Datasets are resolved using the existing `eval/finance/data/sample_config.json`;
optional `data: {train_data: ..., val_data: ..., test_data: ...}` in a config
overrides paths explicitly. Relative data paths are resolved from the repo root.
No truncation or resampling is performed. Training preserves file order.

| Task | Train / validation / test | Validation interval | Workers |
|---|---|---|---|
| finer_0.5 | 500 / 250 / 441 | 50 | 10 |
| formula | 500 / 300 / 200 | 100 | 20 |
| symptom2disease | 200 / 50 / 212 | 50 | 20 |

FiNER uses the legacy aggregate label metric; others use their legacy example
metrics. Malformed answers remain in the denominator. Historical runs that
excluded missing responses therefore need denominator reconciliation before
claiming a directly comparable gain. Recorded data hashes establish the input
files used by idea5, not proof that every older run used those identical bytes.

Common settings: seed 42, one epoch, temperature 0, maximum 4096 output tokens,
80,000-token stored Playbook budget and snapshots every 50 examples.
The old three-round ACE correction process is replaced by this method's
one reflection plus periodic replay; it is not silently run in addition.
All roles use the same configured Qwen model. Model API retries are disabled;
empty/malformed responses use the legacy parser rather than dropping samples.

## Artifacts and modules

- `run_config.json`: resolved config, split SHA-256/counts and code revision.
- `checkpoint.json`: atomic current state, episodes, pending reflections,
  rule lineage, replay predictions and accepted/rejected edits.
- `snapshot_<step>.json`: periodic complete checkpoint.
- `validation_<step>.json`: full ordered predictions and metric.
- `best_playbook.json` / `.txt`: validation-selected rules and compiled context.
- `final_test_results.json`: selected step, full test count, metric and ordered predictions.
- `calls.jsonl`: raw prompts/responses, role, server usage and request duration.
- `usage.json`: total tokens, calls by role and summed request durations
  (not wall-clock runtime when requests overlap).

`agents.py` owns model-facing prompts, `memory.py` validates edits and compiles
rules, `runner.py` owns replay/admission/lifecycle, `data.py` adapts legacy data,
and `run.py` handles config and resume. Meta-agent working traces are bounded to
4,000 characters per response and 2,000 per pending reflection; original traces
remain in the checkpoint. Full task input and target evidence are retained.

## Verification

```bash
python -m unittest discover -s tests -p test_idea5.py -v
```

Tests exercise guarded admission, immutable test evaluation, checkpoint resume,
split lineage, malformed/unsupported edits, token-budget rejection and exclusion
of target labels from Generator prompts. They use a fake model and do not measure
Qwen performance. Real H100 runs are necessary to assess accuracy and runtime.

References informing this research implementation:
- MCE: https://arxiv.org/abs/2601.21557
- ACE: https://arxiv.org/abs/2510.04618
- CEGAR: https://www.cs.cmu.edu/~emc/papers/Conference%20Papers/Counterexample-guided%20Abstraction%20Refinement.pdf
- Related work to distinguish before novelty claims: MemScope
  https://arxiv.org/abs/2610.21170 and Scope Before You Persist
  https://arxiv.org/abs/2609.29144
