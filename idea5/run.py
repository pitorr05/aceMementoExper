"""Run exactly one configured dataset: python -m idea5.run --config FILE."""
import argparse
import json
from pathlib import Path
import subprocess

from .agents import LocalModel, ROOT
from .data import load
from .runner import atomic_json
from .evolution import EvolutionRunner
from .skills import BaseAgent


def main():
    import yaml
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', help='New output directory; refuses to overwrite a run')
    parser.add_argument('--resume', help='Resume a checkpoint directory with identical config/data')
    parser.add_argument('--check-data', action='store_true', help='Inspect full splits without model calls or writing outputs')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    required = {'task', 'model', 'max_tokens', 'seed', 'num_epochs', 'eval_steps', 'save_steps',
                'workers', 'playbook_token_budget', 'update_every', 'max_candidates', 'replay_size',
                'memory_token_budget', 'knowledge_token_budget', 'evidence_token_budget', 'confirmation_examples',
                'skill_token_budget', 'skill_evolve_every', 'skill_history_size'}
    if required - config.keys():
        raise ValueError('Missing config keys: ' + str(required-config.keys()))
    for key in required-{'task', 'model'}:
        if not isinstance(config[key], int) or config[key] < (0 if key == 'seed' else 1):
            raise ValueError('Invalid positive integer config: ' + key)
    if config['max_candidates'] > 2 or config['replay_size'] < 4:
        raise ValueError('Use at most two candidates and replay_size >= 4')
    if 3*config['knowledge_token_budget'] + config['evidence_token_budget'] >= config['memory_token_budget']:
        raise ValueError('Reserve room for active, rollback, selected knowledge and evidence plus metadata')
    if config['skill_evolve_every'] % config['update_every']:
        raise ValueError('skill_evolve_every must be a multiple of update_every')
    processor, splits, hashes = load(config, ROOT)
    if args.check_data:
        print(json.dumps(hashes, indent=2))
        return
    if bool(args.resume) and bool(args.output):
        parser.error('Choose --output or --resume')
    output = Path(args.resume or args.output or f'results/idea5_{config["task"]}')
    state = None
    if args.resume:
        manifest = json.loads((output/'run_config.json').read_text(encoding='utf-8'))
        if manifest['config'] != config or manifest['data'] != hashes:
            raise ValueError('Resume config/data mismatch')
        state = json.loads((output/'checkpoint.json').read_text(encoding='utf-8'))
    else:
        output.mkdir(parents=True, exist_ok=False)
        revision = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                           '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        from .reproduce_mce import verify_source
        atomic_json(output/'run_config.json', dict(config=config, data=hashes, revision=revision,
                    mce_source_revision=verify_source()['revision'], backend='qwen_skill_evolution'))
    from utils import count_tokens, set_global_seed
    set_global_seed(config['seed'])
    runner = EvolutionRunner(config, processor, splits, BaseAgent(LocalModel(config, output), config['task']),
                             output, count_tokens, state=state)
    print('Final test accuracy:', runner.run())


if __name__ == '__main__':
    main()
