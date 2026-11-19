import json
from pathlib import Path
import tempfile
import unittest

from idea5.memory import apply_patch, compile_playbook
from idea5.runner import Runner


def patch(condition='all'):
    return dict(operation='add', target_id=None, rules=[dict(group='math', condition=condition, guidance='multiply')],
                rationale='training evidence', evidence_ids=['train-000001'])


class Metric:
    def answer_is_correct(self, prediction, target):
        return prediction == target

    def evaluate_accuracy(self, predictions, targets):
        return sum(a == b for a, b in zip(predictions, targets))/len(targets)


class FakeAgents:
    def __init__(self):
        self.reflections = []
        self.calls = []

    def generate(self, sample, playbook):
        self.calls.append(sample['question'])
        return dict(prediction='yes' if playbook else 'no', raw='response', cited=[])

    def reflect(self, episode):
        self.reflections.append(episode['sample']['question'])
        return 'lesson'

    def propose(self, *args):
        return {'patches': [patch()]}


class Tests(unittest.TestCase):
    def test_patch_scope_and_evidence(self):
        rules = apply_patch([], patch(), {'train-000001'}, 1)
        self.assertEqual(len(rules), 1)
        self.assertIn('When: all', compile_playbook(rules))
        with self.assertRaises(ValueError):
            apply_patch([], patch(), set(), 1)
        p = patch()
        p.update(operation='condition', target_id=rules[0]['id'])
        p['rules'][0]['guidance'] = 'unrelated replacement'
        with self.assertRaises(ValueError):
            apply_patch(rules, p, {'train-000001'}, 2)
        self.assertEqual(rules[0]['guidance'], 'multiply')

    def fixture(self, directory, agents=None, token_count=len):
        config = dict(workers=1, seed=42, replay_size=4, max_candidates=2,
                      playbook_token_budget=1000, num_epochs=1, update_every=2, eval_steps=2, save_steps=2)
        samples = lambda prefix, n: [dict(question=f'{prefix}{i}', context='input', target='yes') for i in range(n)]
        splits = dict(train=samples('train', 4), val=samples('val', 2), test=samples('test', 3))
        return Runner(config, Metric(), splits, agents or FakeAgents(), directory, token_count)

    def test_offline_and_resume_are_frozen_at_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            agents = FakeAgents()
            runner = self.fixture(tmp, agents)
            self.assertEqual(runner.run(), 1)
            self.assertEqual(len(agents.reflections), 4)
            self.assertEqual(len([c for c in agents.calls if c.startswith('test')]), 3)
            self.assertEqual(runner.state['best_step'], 2)
            state = json.loads((Path(tmp)/'checkpoint.json').read_text())
            resumed = self.fixture(tmp)
            resumed.state = state
            self.assertEqual(resumed.run(), 1)
            self.assertEqual(resumed.agents.reflections, [])
            self.assertEqual(resumed.state, state)

    def test_budget_rejects_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.fixture(tmp, token_count=lambda _: 2000)
            self.assertEqual(runner.run(), 0)
            self.assertEqual(runner.state['rules'], [])

    def test_guard_blocks_regression(self):
        class GuardAgents(FakeAgents):
            def generate(self, sample, playbook):
                good = sample['question'] == 'guard'
                return dict(prediction='yes' if (good != bool(playbook)) else 'no', raw='', cited=[])
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.fixture(tmp, GuardAgents())
            runner.state['step'] = 2
            runner.state['episodes'] = [dict(id=f'train-{i:06d}', sample=dict(question=q, context='', target='yes'),
                                             correct=(q == 'guard'), cited=[]) for i, q in enumerate(['bad','guard','bad2','bad3'], 1)]
            runner.refine()
            self.assertEqual(runner.state['rules'], [])

    def test_split_is_atomic_and_retains_lineage(self):
        rules = apply_patch([], patch(), {'train-000001'}, 1)
        split = patch()
        split.update(operation='split', target_id=rules[0]['id'])
        split['rules'] = [dict(group='math', condition=c, guidance='compute') for c in ('ratio','percentage')]
        result = apply_patch(rules, split, {'train-000001'}, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual({r['parent'] for r in result}, {rules[0]['id']})
        self.assertEqual(len(rules), 1)

    def test_generator_does_not_receive_gold(self):
        from idea5.agents import Agents
        class Capture:
            def call(self, prompt, role):
                self.prompt = prompt
                return '{"final_answer":"0.5", "bullet_ids":[]}'
        model = Capture()
        result = Agents(model).generate(dict(question='question', context='context', target='SECRET_GOLD'), '')
        self.assertNotIn('SECRET_GOLD', model.prompt)
        self.assertEqual(result['prediction'], '0.5')

    def test_resume_after_interrupted_step(self):
        class Interrupt(FakeAgents):
            def reflect(self, episode):
                if episode['sample']['question'] == 'train2':
                    raise RuntimeError('simulated service outage')
                return super().reflect(episode)
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.fixture(tmp, Interrupt())
            with self.assertRaises(RuntimeError):
                runner.run()
            state = json.loads((Path(tmp)/'checkpoint.json').read_text())
            self.assertEqual(state['step'], 2)
            resumed = self.fixture(tmp)
            resumed.state = state
            resumed.run()
            self.assertEqual(resumed.agents.reflections, ['train2', 'train3'])
            self.assertEqual(len(resumed.state['episodes']), 4)

    def test_malformed_meta_response_keeps_parent(self):
        class Malformed(FakeAgents):
            def propose(self, *args):
                return {'patches': [None, {'operation': 'delete'}]}
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.fixture(tmp, Malformed())
            runner.run()
            self.assertEqual(runner.state['rules'], [])
            self.assertTrue(all('rejected' in c for event in runner.state['updates'] for c in event['candidates']))


if __name__ == '__main__':
    unittest.main()
