"""Two-level skill/context evolution with bounded knowledge transactions."""
from copy import deepcopy
from .bounded import BoundedRunner
from .budget import fit_state
from .skills import MetaAgent, INITIAL_SKILL
from .runner import atomic_json


class EvolutionRunner(BoundedRunner):
    def __init__(self, config, processor, splits, agents, output, token_count, state=None):
        if state is not None and state.get('schema_version') != 3:
            raise ValueError('Skill-evolution runs require schema 3; start a fresh output directory')
        legacy_state = deepcopy(state) if state else None
        if legacy_state:
            legacy_state['schema_version'] = 2
        super().__init__(config, processor, splits, agents, output, token_count, state=legacy_state)
        self.state['schema_version'] = 3
        self.state.setdefault('skill', INITIAL_SKILL)
        self.state.setdefault('skill_history', [])
        self.state.setdefault('best_skill', INITIAL_SKILL)
        self.state.setdefault('skill_generation', 0)
        self.meta = MetaAgent(agents.model, config['task'], token_count, config['skill_token_budget'])
        self.agents.skill = self.state['skill']
        self.enforce()

    def refine(self):
        if self.state['trial'] is not None or not self.state['episodes']:
            return
        old_skill = self.state['skill']
        due = self.state['step'] % self.c['skill_evolve_every'] == 0 or self.state['skill_generation'] == 0
        proposed = old_skill
        if due:
            evidence, _, _ = self.replay_set()
            try:
                result = self.meta.evolve(old_skill, self.state['skill_history'], evidence, self.state['pending'])
                proposed = result['skill_md']
                self.audit('skill_proposed', result)
            except (ValueError, TypeError, AttributeError) as error:
                self.audit('skill_rejected', dict(error=str(error)))
                return
        # These values are counted before accepting a context transaction, including
        # the previous skill needed for prospective rollback.
        self.state['skill'] = proposed
        self.agents.skill = proposed
        self.state['skill_parent'] = old_skill
        try:
            self.enforce()
            super().refine()
        except ValueError as error:
            self.audit('skill_rejected', dict(error=str(error)))
        if self.state['trial'] is None:
            self.state['skill'] = old_skill
            self.agents.skill = old_skill
            self.state.pop('skill_parent', None)
        else:
            self.state['skill_generation'] += int(due)

    def settle_trial(self, incomplete=False):
        if self.state['trial'] is None:
            return
        trial = self.state['trial']
        candidate = self.score(trial['candidate_answers'], trial['targets']) if trial['targets'] else None
        parent = self.score(trial['parent_answers'], trial['targets']) if trial['targets'] else None
        accepted = not incomplete and candidate >= parent
        # Archive and retain a small skill/outcome history for subsequent crossover.
        outcome = dict(skill_md=self.state['skill'], accepted=accepted,
                       candidate_score=candidate, parent_score=parent, evaluated_step=self.state['step'])
        super().settle_trial(incomplete)
        if not accepted:
            self.state['skill'] = self.state.get('skill_parent', self.state['skill'])
        self.state.pop('skill_parent', None)
        self.agents.skill = self.state['skill']
        self.state['skill_history'] = (self.state['skill_history']+[outcome])[-self.c['skill_history_size']:]
        self.audit('skill_evaluated', outcome)

    def validate(self):
        old_best = self.state['best_step']
        super().validate()
        if self.state['best_step'] != old_best:
            self.state['best_skill'] = self.state.get('skill_parent', self.state['skill'])
        self.enforce()

    def run(self):
        score = super().run()
        atomic_json(self.output/'selected_skill.json', dict(skill_md=self.state['best_skill']))
        (self.output/'selected_SKILL.md').write_text(self.state['best_skill'], encoding='utf-8')
        return score

    def save(self):
        super().save()
        if 'skill' not in self.state:
            return
        # Inspectable MCE-style artifacts are exports of the bounded checkpoint;
        # agents cannot read previous iteration directories to bypass its budget.
        from .memory import compile_playbook
        workspace = self.output/'current'
        skill_dir = workspace/'.claude/skills/learning-context'
        context_dir = workspace/'context'
        interface_dir = workspace/'interfaces'
        for directory in (skill_dir, context_dir, interface_dir):
            directory.mkdir(parents=True, exist_ok=True)
        (skill_dir/'SKILL.md').write_text(self.state['skill'], encoding='utf-8')
        atomic_json(context_dir/'knowledge.json', self.state['rules'])
        (context_dir/'knowledge.txt').write_text(compile_playbook(self.state['rules']), encoding='utf-8')
        (interface_dir/'__init__.py').write_text(
            'from pathlib import Path\n\ndef get_context(question: str, context: str) -> str:\n'
            '    """Compiled bounded knowledge; no label or archive access."""\n'
            '    return (Path(__file__).resolve().parents[1]/"context"/"knowledge.txt").read_text(encoding="utf-8")\n',
            encoding='utf-8')
