"""MCE-derived meta/base separation with Qwen-native artifact transport.

The upstream SDK filesystem transport is available separately via reproduce_mce.
This backend evolves declarative learning skills, not unrestricted Python code.
"""
from .agents import Agents, parse_object
from .upstream import meta_prompt, base_prompt
import json


INITIAL_SKILL = '''## Skill Overview
Revise reusable knowledge from contrasting successful and failed training episodes.
## Methodology
1. Inspect retained inputs, model outputs, verified feedback and pending reflections.
2. Identify missing knowledge, incorrect guidance, overbroad conditions and redundancy.
3. Preserve input distinctions and exceptions supported by evidence.
4. Propose compact atomic replacements that correct failures without harming successes.
5. Under capacity pressure, merge redundant guidance to free room before adding knowledge.
'''


class MetaAgent:
    def __init__(self, model, task, count, limit):
        self.model, self.task, self.count, self.limit = model, task, count, limit

    def evolve(self, current, history, evidence, pending):
        prompt = meta_prompt(self.task) + '''
## Qwen artifact transport
No filesystem tools are exposed in this backend. All readable history is supplied below.
Evolve the LEARNING PROCEDURE, not the task answers or context rules themselves.
Use agentic crossover: preserve successful parts of retained skills, change ineffective
steps in light of prospective confirmation/replay feedback. Never invent measured results.
Return JSON {"skill_md":"## Skill Overview\\n...\\n## Methodology\\n...",
"rationale":"what changed and why"}. The skill must stand alone, be concise,
and specify how the base-agent diagnoses, generalizes, consolidates and revises knowledge.
Do not change evaluator, dataset instructions, evidence access or runtime safety gates.
'''
        data = dict(current_skill=current, history=history, evidence=evidence, pending=pending,
                    maximum_skill_tokens=self.limit)
        result = parse_object(self.model.call(prompt+json.dumps(data, ensure_ascii=False), 'meta_skill'))
        skill = result.get('skill_md')
        if (not isinstance(skill, str) or '## Skill Overview' not in skill or
                '## Methodology' not in skill or self.count(skill) > self.limit):
            raise ValueError('Skill needs overview/methodology and must fit skill_token_budget')
        if not isinstance(result.get('rationale'), str) or not result['rationale'].strip():
            raise ValueError('Skill evolution needs a rationale')
        return result


class BaseAgent(Agents):
    """Execute an evolved skill through the existing checked transaction interface."""
    def __init__(self, model, task):
        super().__init__(model)
        self.task = task
        self.skill = INITIAL_SKILL

    def propose(self, rules, evidence, candidates, target, max_candidates):
        return super().propose(rules, evidence, candidates, target, max_candidates)

    def learning_prefix(self):
        return base_prompt(self.task) + '\n## Learning skill (execute this methodology)\n' + self.skill + '\n'
