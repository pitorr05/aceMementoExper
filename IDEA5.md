# Idea5: bounded knowledge revision (checkpoint schema 2)

This version adds atomic replacement transactions, a bounded readable memory,
and confirmation on later training examples. It is an experimental method;
no Qwen accuracy improvement or publication novelty has been established.

## Run one dataset

Install the client dependencies from the checkout root:

```bash
python -m pip install -r requirements-idea5.txt
export VLLM_BASE_URL=http://localhost:5000/v1
```

PowerShell endpoint setting: `$env:VLLM_BASE_URL = 'http://localhost:5000/v1'`.
Use your existing server serving `Qwen/Qwen3-4B-Instruct-2507`.

Run exactly one of the following commands for the desired dataset:

```bash
python -m idea5.run --config configs/idea5/finer_0.5.yaml
```

```bash
python -m idea5.run --config configs/idea5/formula.yaml
```

```bash
python -m idea5.run --config configs/idea5/symptom2disease.yaml
```

Default output is `results/idea5_<task>`. Existing directories are rejected;
use `--output results/formula_v2` for a new run. No all-dataset launcher exists.

```bash
python -m idea5.run --config configs/idea5/formula.yaml --check-data
python -m idea5.run --config configs/idea5/formula.yaml --resume results/idea5_formula
```

`--check-data` prints full split hashes/counts without model calls. Resume uses
the same config/data and an atomic step checkpoint. A partly completed step is
repeated; already incurred calls remain in the log. Resume a completed run to
reevaluate test. Resume with the same code/model-server settings and one writer.
Schema-1 checkpoints from the earlier runner are rejected: start a fresh run.

## Closed learning loop

1. Compile knowledge and answer the current training task once.
2. Score with the original evaluator; reflect on the input, prediction and target.
3. Archive the complete trace for audit; admit a compact evidence record and
   pending reflection to bounded memory. The learning algorithm never reloads archives.
4. Every 25 examples, if no transaction is under confirmation, assemble up to
   16 retained episodes: a recent failure, up to four randomly chosen past
   successes, and lexical input neighbors. Both question and context are used.
5. The meta-agent proposes up to two alternative replacement transactions.
   Each removes zero to four units and inserts one to four units. This supports
   correction, condition refinement, splitting, merging, and freeing space for
   new knowledge in the same atomic change. Each new unit has conditions,
   guidance, group, source IDs, parent IDs and creation step.
6. Run parent and candidate on identical replay inputs. Accept higher aggregate
   dataset accuracy, or equal accuracy with fewer serialized knowledge tokens,
   provided the successful guard-group score does not fall. Budget checks count
   the candidate AND its rollback parent, selected checkpoint and evidence.
7. Tentatively use the winning transaction on the next eight training episodes.
   Independently run its frozen parent on each of those same inputs. Both calls
   exclude the current target. Only after predictions does the evaluator score them.
8. Confirm if the candidate aggregate metric is at least the parent's; otherwise
   restore the parent. No other transaction can overlap this trial. Incomplete
   trials roll back at stream end or when memory pressure prevents keeping them.
9. Validation selects only confirmed knowledge (the parent while a trial is
   active). Test reads the selected checkpoint with one Generator call per item.

No trial starts unless enough future training episodes remain. Thus end-of-stream
lessons can remain pending without entering the selected knowledge. Later episodes
provide prospective feedback, not an independent final test; repeated or duplicate
dataset items are not claimed to be unseen. All test feedback is excluded from learning.

The compiler renders all admitted units and conditions. Conditions are natural
language, not an executable router. The meta-agent's prompt and operator set are
fixed; this implementation does not evolve arbitrary code or MCE learning skills.

## Fixed readable-memory budget

All three configs use these initial, untuned settings:

| Setting | Default | Includes |
|---|---:|---|
| `memory_token_budget` | 32,000 | Entire serialized persistent checkpoint |
| `knowledge_token_budget` | 4,000 | One serialized knowledge version |
| `evidence_token_budget` | 16,000 | Retained episodes plus pending reflections |
| `confirmation_examples` | 8 | Subsequent training episodes per trial |

The total includes active knowledge, rollback parent, validation-selected knowledge,
retained evidence, pending reflections, bounded confirmation answers and counters.
The allocator reserves space for three knowledge versions. The legacy
`playbook_token_budget: 80000` remains an additional compiled-context ceiling.
Counts use the legacy `cl100k_base` tokenizer for reproducibility, not the actual
Qwen tokenizer. Configure the server context window for full prompts and output.

Generated reasoning traces are removed from readable historical evidence; exact
task input, prediction, target and outcome remain. Legacy `others` metadata,
including duplicated `original_context`, is removed from retained samples; the
Generator question/context and evaluator target remain byte-for-byte unchanged.
Full metadata stays in write-only audit records. The 32K default reserves up to
12K for three knowledge versions, 16K for evidence and roughly 4K for other state.
It is an initial research setting, not a measured optimum. In particular, FiNER's
long inputs may permit fewer than the maximum 16 replay episodes.
Under pressure, oldest pending
reflections are evicted first. Evidence is then evicted by a fixed score favoring
references from active/parent/selected knowledge, failures, lexical distinctness,
and recency. Records are removed whole: inputs and labels are not silently cut.
This retention policy is a heuristic, not a learned optimal allocation policy.
An oversized episode can be excluded from memory while its audit record survives.
A new transaction must retain at least one of its cited evidence records. Older
source IDs can outlive their records; the agent cannot dereference those IDs into
the archive. The knowledge/rollback snapshots are never implicitly evicted.

Current task input, model-generated candidate proposals and temporary execution
buffers are working computation, not persistent memory. Dataset arrays are owned
by the experiment harness; the proposal agent receives only retained past evidence,
never arbitrary past/future dataset access. Audit/call logs can grow on disk but
are not searchable tools for the model. Call logs are read only for usage reporting.

## Compatibility and artifacts

Parsers and metrics come unchanged from `eval/finance/data_processor.py` at the
`origin/idea4` base. Data paths come from `eval/finance/data/sample_config.json`
unless an explicit config `data` mapping overrides train_data/val_data/test_data.
Paths are relative to the checkout root; file order and all samples are preserved.

| Task | Train / validation / test | Validation interval | Workers |
|---|---|---|---|
| finer_0.5 | 500 / 250 / 441 | 50 | 10 |
| formula | 500 / 300 / 200 | 100 | 20 |
| symptom2disease | 200 / 50 / 212 | 50 | 20 |

FiNER retains its aggregate label metric. Other tasks retain their example metrics.
Common settings: one epoch, seed 42, temperature 0, maximum 4096 output tokens,
snapshot interval 50. Generator uses the original ACE prompt and answer parser;
the AMORE Case Memory prompt and three-round ACE correction loop are not run.
Parent confirmation adds training calls only. Empty/malformed outputs remain in
the denominator; API failures abort instead of silently dropping samples. Historical
runs with different missing-answer handling or prompts need reconciliation before
claiming directly comparable gains. New memory allocations are method parameters,
not inherited benchmark settings.

- `run_config.json`: config, data hashes/counts and code revision.
- `checkpoint.json`, `snapshot_<step>.json`: bounded schema-2 state for resume.
- `audit.jsonl`: complete episodes, proposed transactions, replay predictions,
  evictions, confirmation results and rollbacks; write-only to learning.
- `validation_<step>.json`: full validation predictions and metric.
- `best_playbook.json` / `.txt`: selected knowledge and rendered context.
- `final_test_results.json`: selected step, full test metric/count and predictions.
- `calls.jsonl`, `usage.json`: raw model requests and aggregate call/token costs.
  Summed request duration is not wall-clock duration with parallel calls.

`bounded.py` owns the new lifecycle; `budget.py` accounts and prunes readable state;
`memory.py` validates atomic transactions; `agents.py` contains model-facing prompts.
The older `Runner` in `runner.py` remains for legacy regression tests and shared
evaluation utilities; the CLI now uses `BoundedRunner` exclusively.

## Verification

```bash
python -m unittest discover -s tests -p 'test_idea5*.py' -v
```

Tests cover merge/correction atomicity, equal-score compression, memory bounds,
rollback/selected-checkpoint accounting, future-regression rollback, successful
confirmation, interrupted-trial resume, rejection of old checkpoints, exclusion
of labels from Generator prompts, and a single test call per sample. Model calls
are fixtures; no H100 benchmark result is implied.

Research antecedents: [MCE](https://arxiv.org/abs/2601.21557),
[ACE](https://arxiv.org/abs/2510.04618),
[CEGAR](https://www.cs.cmu.edu/~emc/papers/Conference%20Papers/Counterexample-guided%20Abstraction%20Refinement.pdf).
Related work to distinguish before novelty claims includes
[MemScope](https://arxiv.org/abs/2610.21170),
[Scope Before You Persist](https://arxiv.org/abs/2609.29144) and
[Retain or Consolidate?](https://arxiv.org/abs/2607.17545).
