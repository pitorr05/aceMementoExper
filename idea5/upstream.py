"""Load unmodified MCE prompt builders without the Claude SDK runtime."""
from functools import lru_cache
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1]/'third_party/mce'


@lru_cache(maxsize=1)
def builders():
    # Upstream imports use absolute env/mce packages, hence the explicit source root.
    sys.path.insert(0, str(SOURCE))
    from mce.prompts.meta_agent import build_meta_agent_prompt
    from mce.prompts.base_agent import build_base_agent_prompt
    from env.base import InterfaceSignature
    return build_meta_agent_prompt, build_base_agent_prompt, InterfaceSignature


def objective(task):
    return (f'Learn context for the ACE benchmark {task}. Use the unchanged question/context '
            'and evaluator. Distinguish ground-truth feedback from generated output. '
            'Generalize knowledge across inputs; do not build answer lookup tables.')


def meta_prompt(task):
    meta, _, _ = builders()
    # Use upstream role/architecture, but replace filesystem history with bounded
    # explicit history. Never let the agent search archived logs for extra memory.
    original = meta(objective(task), [], '/virtual/iter1', '/virtual')
    return original.split('## Working Directory')[0]


def base_prompt(task):
    _, base, signature = builders()
    sig = signature(name='propose_transaction', inputs=[('memory', 'dict', 'Retained knowledge and evidence')],
                    output=('dict', 'Atomic replacement transactions'),
                    description='Apply the supplied learning skill to propose context updates.')
    original = base(objective(task), [sig], '/virtual/iter1')
    return original.split('## Working Directory')[0]
