import json
from pathlib import Path
import tempfile
import unittest

from idea5.skills import BaseAgent
from idea5.evolution import EvolutionRunner
from idea5.reproduce_mce import verify_source, make_environment
from test_idea5 import Metric
import asyncio


class Model:
    def __init__(self, regress=False):
        self.prompts = []
        self.config = {}
        self.regress = regress

    def call(self, prompt, role):
        self.prompts.append((role, prompt))
        if role == 'meta_skill':
            return json.dumps(dict(skill_md='## Skill Overview\nEvolved method.\n## Methodology\nUse contrasting evidence and consolidate.',
                                   rationale='respond to retained feedback'))
        if role == 'base_context':
            return json.dumps({'patches': [dict(operation='replace', remove_ids=[],
                  rules=[dict(group='general', condition='all inputs', guidance='solve carefully')],
                  rationale='learn pattern', evidence_ids=['train-000001'])]})
        if role == 'reflector':
            return '{"lesson":"compare evidence"}'
        if role == 'generator':
            has_rule = '[rule-' in prompt
            future = '**Question:**\ntrain2' in prompt or '**Question:**\ntrain3' in prompt
            return json.dumps(dict(final_answer='yes' if has_rule != (self.regress and future) else 'no', bullet_ids=[]))
        raise AssertionError(role)


class EvolutionTests(unittest.TestCase):
    def runner(self, tmp, model=None, state=None):
        config = dict(task='formula', workers=1, seed=42, replay_size=4, max_candidates=2,
                      playbook_token_budget=5000, num_epochs=1, update_every=2, eval_steps=2,
                      save_steps=2, confirmation_examples=2, memory_token_budget=30000,
                      knowledge_token_budget=5000, evidence_token_budget=8000,
                      skill_token_budget=1000, skill_evolve_every=2, skill_history_size=2)
        make = lambda prefix, n: [dict(question=prefix+str(i), context='input', target='yes') for i in range(n)]
        splits = dict(train=make('train',4), val=make('val',2), test=make('test',2))
        return EvolutionRunner(config, Metric(), splits, BaseAgent(model or Model(), 'formula'), tmp, len, state)

    def test_evolved_skill_reaches_base_and_is_confirmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Model()
            runner = self.runner(tmp, model)
            self.assertEqual(runner.run(), 1)
            self.assertEqual(runner.state['schema_version'], 3)
            self.assertIn('Evolved method', runner.state['skill'])
            self.assertIn('Evolved method', runner.state['best_skill'])
            base = [prompt for role, prompt in model.prompts if role == 'base_context']
            self.assertEqual(len(base), 1)
            self.assertIn('Evolved method', base[0])
            self.assertIn('Context Engineer', base[0])
            self.assertTrue(runner.state['skill_history'][0]['accepted'])
            import runpy
            exported = runpy.run_path(str(Path(tmp)/'current/interfaces/__init__.py'))
            self.assertIn('solve carefully', exported['get_context']('q', 'input'))
            restored = self.runner(tmp, state=runner.state)
            self.assertEqual(restored.run(), 1)
            self.assertEqual(restored.state['skill'], runner.state['skill'])

    def test_skill_and_knowledge_rollback_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(tmp, Model(regress=True))
            original = runner.state['skill']
            runner.run()
            self.assertEqual(runner.state['skill'], original)
            self.assertEqual(runner.state['rules'], [])
            self.assertFalse(runner.state['skill_history'][0]['accepted'])

    def test_malformed_skill_never_reaches_base(self):
        class Bad(Model):
            def call(self, prompt, role):
                if role == 'meta_skill':
                    return '{"skill_md":"missing sections", "rationale":"test"}'
                return super().call(prompt, role)
        with tempfile.TemporaryDirectory() as tmp:
            model = Bad()
            runner = self.runner(tmp, model)
            runner.run()
            self.assertEqual(runner.state['rules'], [])
            self.assertFalse(any(role == 'base_context' for role, _ in model.prompts))

    def test_evolution_uses_feedback_history_and_resumes_active_trial(self):
        class Interrupt(Model):
            def call(self, prompt, role):
                if role == 'reflector' and 'train2' in prompt:
                    raise RuntimeError('interrupted')
                return super().call(prompt, role)
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(tmp, Interrupt())
            with self.assertRaises(RuntimeError):
                runner.run()
            state = json.loads((Path(tmp)/'checkpoint.json').read_text())
            self.assertIsNotNone(state['trial'])
            self.assertIn('Evolved method', state['skill'])
            resumed = self.runner(tmp, state=state)
            resumed.run()
            # Another evolution call receives evaluated skill/outcome history,
            # rather than scanning the write-only audit log.
            resumed.state['step'] = 6
            resumed.refine()
            meta = [p for role, p in resumed.agents.model.prompts if role == 'meta_skill']
            self.assertEqual(len(meta), 1)
            self.assertIn('"accepted": true', meta[0])
            self.assertIn('Evolved method', meta[0])

    def test_upstream_source_integrity(self):
        self.assertEqual(len(verify_source()['revision']), 40)

    def test_upstream_adapter_does_not_expose_gold_to_context(self):
        from env.base import Sample
        env = make_environment('formula', Metric(), 1)()
        model = Model()
        class Client:
            agents = BaseAgent(model, 'formula')
        observed = []
        def get_context(question, context):
            observed.append((question, context))
            return 'knowledge'
        sample = Sample(id=0, question='q', context='input', ground_truth='SECRET')
        result = asyncio.run(env.aevaluate(sample, {'get_context': get_context}, Client()))
        self.assertEqual(observed, [('q', 'input')])
        self.assertTrue(all('SECRET' not in prompt for role, prompt in model.prompts))
        self.assertEqual(result.metrics['count'], 1)


if __name__ == '__main__':
    unittest.main()
