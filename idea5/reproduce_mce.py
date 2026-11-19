"""Run actual vendored MCE orchestration with unchanged ACE task adapters.

Optimization uses upstream Claude Agent SDK. Generator uses the user's vLLM.
This command does NOT pretend that a Qwen chat endpoint is a Claude SDK runtime.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys

from .agents import ROOT, Agents, LocalModel
from .data import load
from .runner import atomic_json
from .upstream import SOURCE, builders


def verify_source():
    manifest = json.loads((SOURCE/'UPSTREAM.json').read_text())
    for name, digest in manifest['files'].items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Upstream source modified: '+name)
    return manifest


def make_environment(task, processor, workers):
    builders()
    from env.base import TaskEnvironment, InterfaceSignature, Sample, EnvironmentResult
    from eval.finance.data_processor import load_data

    class ACEEnvironment(TaskEnvironment):
        def get_interface_signatures(self):
            return [InterfaceSignature('get_context', [('question', 'str', 'Task question'),
                     ('context', 'str', 'Task input')], ('str', 'Knowledge to help answer'),
                     'Return task-specific context using learned files/code; never access target labels.')]

        def get_task_instruction(self):
            return (f'ACE benchmark {task}; preserve task question/context and output format. '
                    'Learn reusable knowledge and implement get_context(question, context). '
                    'Use training feedback only. No task-answer lookup tables. '
                    'Use portable paths based on __file__ for your context files.')

        def load_samples(self, path, limit=None, random_sample=False, shuffle=False):
            # Deliberately align order/splits to ACE; upstream random sampling disabled.
            rows = processor.process_task_data(load_data(path))
            if limit:
                rows = rows[:limit]
            return [Sample(id=i, question=s['question'], context=s['context'], ground_truth=s['target'])
                    for i, s in enumerate(rows)]

        async def aevaluate(self, sample, interfaces, llm_client=None, context_dir=None, log_dir=None):
            get_context = interfaces.get('get_context')
            context = get_context(sample.question, sample.context) if get_context else ''
            if not isinstance(context, str):
                raise TypeError('get_context must return str')
            record = await asyncio.to_thread(llm_client.agents.generate,
                                            dict(question=sample.question, context=sample.context), context)
            prediction = record['prediction']
            if task in ('finer', 'finer_0.5'):
                correct, total = processor._finer_answer_is_correct(prediction, sample.ground_truth, return_counts=True)
            else:
                correct, total = int(processor.answer_is_correct(prediction, sample.ground_truth)), 1
            return EnvironmentResult(feedback=f'Predicted: {prediction}; target: {sample.ground_truth}',
                    ground_truth=sample.ground_truth, metrics=dict(accuracy=correct/total, correct=correct, count=total),
                    trajectory=[dict(prediction=prediction, raw=record['raw'], knowledge=context)])

        def format_result_for_training(self, item):
            sample, evaluation = item['sample'], item['evaluation']
            return dict(sample=sample, evaluation=evaluation)

    return ACEEnvironment


async def execute(config, output, iterations):
    from .upstream import builders
    builders()
    # Import SDK-dependent runtime only in explicit reproduction mode.
    from mce import main as upstream
    from mce.eval import batch_evaluate, load_interfaces
    from mce.logging_utils import setup_logger
    from mce.utils import setup_meta_agent_reference
    from env.registry import EnvironmentRegistry
    import mce.eval as evaluation

    processor, splits, hashes = load(config, ROOT)
    output.mkdir(parents=True, exist_ok=False)
    workspace = output/'workspace'
    workspace.mkdir()
    paths = json.loads((ROOT/'eval/finance/data/sample_config.json').read_text())[config['task']]
    paths = config.get('data', paths)
    paths = {k: str((ROOT/Path(v)).resolve()) if not Path(v).is_absolute() else v for k, v in paths.items()}
    atomic_json(output/'run_config.json', dict(config=config, data=hashes, upstream=verify_source()['revision'],
                    protocol='MCE SDK loop with ACE data/order/metric/Generator adapter', iterations=iterations))
    env_name = 'ace_'+config['task']
    EnvironmentRegistry.register(env_name, make_environment(config['task'], processor, config['workers']))
    # Upstream averages per-example metrics. FiNER requires label-weighted totals.
    def weighted(records):
        metrics = [x['evaluation']['metrics'] for x in records]
        total = sum(x['count'] for x in metrics)
        return dict(accuracy=sum(x['correct'] for x in metrics)/total if total else 0)
    evaluation.compute_avg_metrics = weighted
    evaluation.MAX_CONCURRENCY = config['workers']
    original_batch = upstream.batch_evaluate
    async def checked_batch(*args, **kwargs):
        result = await original_batch(*args, **kwargs)
        if result['summary']['errors']:
            raise RuntimeError('MCE adapter evaluation failed; refusing a reduced denominator')
        return result
    upstream.batch_evaluate = checked_batch
    class EvalClient:
        def __init__(self, model):
            self.agents = Agents(LocalModel(config, output))
    upstream.LLMClient = EvalClient
    logger = setup_logger(name='mce_ace_reproduction', log_dir=str(output/'logs'))
    setup_meta_agent_reference(workspace, paths['train_data'], len(splits['train']), logger)
    results = []
    for iteration in range(1, iterations+1):
        result = await upstream.run_iteration(workspace, iteration, env_name, paths['val_data'], paths['train_data'],
                len(splits['train']), len(splits['val']), config['model'], logger, output,
                train_batch_size=config['update_every'])
        if 'error' in result:
            raise RuntimeError(result['error'])
        results.append(result)
        atomic_json(output/'iterations.json', results)
    best = max(results, key=lambda x: x['val_primary_metric'])
    selected = Path(best['sub_iterations'][-1]['folder'])
    env = EnvironmentRegistry.get(env_name)
    interfaces = load_interfaces(selected, env.get_interface_signatures())
    test = await checked_batch(interfaces, env.load_samples(paths['test_data'], limit=None),
                                env_name, EvalClient(config['model']), selected, output)
    atomic_json(output/'final_test_results.json', dict(selected_iteration=best['iteration'], **test))


def main():
    import yaml
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config')
    parser.add_argument('--output')
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--verify-source', action='store_true')
    args = parser.parse_args()
    if args.verify_source:
        print(json.dumps(dict(revision=verify_source()['revision'], status='verified')))
        return
    if not args.config or not args.output or args.iterations < 1:
        parser.error('Provide --config, --output and positive --iterations')
    if not os.getenv('ANTHROPIC_API_KEY') and not os.getenv('ANTHROPIC_AUTH_TOKEN'):
        parser.error('Original MCE optimizer requires Claude SDK authentication; Qwen-only runs use idea5.run')
    config = yaml.safe_load(Path(args.config).read_text())
    from utils import set_global_seed
    set_global_seed(config['seed'])
    asyncio.run(execute(config, Path(args.output).resolve(), args.iterations))


if __name__ == '__main__':
    main()
