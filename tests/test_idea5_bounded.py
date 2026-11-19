import json
from pathlib import Path
import tempfile
import unittest

from idea5.bounded import BoundedRunner
from idea5.budget import fit_state, serialized_tokens
from idea5.memory import apply_patch
from test_idea5 import Metric, FakeAgents, patch


def transaction(remove=None, guidance='use evidence'):
    return dict(operation='replace', remove_ids=remove or [],
                rules=[dict(group='topic', condition='relevant input', guidance=guidance)],
                rationale='revise with retained evidence', evidence_ids=['train-000001'])


class BoundedTests(unittest.TestCase):
    def fixture(self, tmp, agents=None, state=None):
        config = dict(workers=1, seed=42, replay_size=4, max_candidates=2,
                      playbook_token_budget=1000, num_epochs=1, update_every=2,
                      eval_steps=2, save_steps=2, confirmation_examples=2,
                      memory_token_budget=6000, knowledge_token_budget=1000,
                      evidence_token_budget=2000)
        make = lambda p, n: [dict(question=f'{p}{i}', context='input', target='yes') for i in range(n)]
        splits = dict(train=make('train', 4), val=make('val', 2), test=make('test', 3))
        return BoundedRunner(config, Metric(), splits, agents or FakeAgents(), tmp, len, state=state)

    def test_confirmation_uses_future_and_test_is_single_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.fixture(tmp)
            self.assertEqual(r.run(), 1)
            self.assertIsNone(r.state['trial'])
            events = [json.loads(x) for x in (Path(tmp)/'audit.jsonl').read_text().splitlines()]
            confirmed = [x for x in events if x['kind'] == 'confirmed']
            self.assertEqual(len(confirmed), 1)
            self.assertEqual(confirmed[0]['transaction_step'], 2)
            self.assertEqual(confirmed[0]['step'], 4)
            self.assertEqual(len([x for x in r.agents.calls if x.startswith('test')]), 3)
            self.assertLessEqual(serialized_tokens(r.state, len), 6000)
            self.assertNotIn('updates', r.state)

    def test_future_regression_rolls_back(self):
        class Regress(FakeAgents):
            def generate(self, sample, playbook):
                future = sample['question'] in ('train2', 'train3')
                return dict(prediction='yes' if bool(playbook) != future else 'no', raw='', cited=[])
        with tempfile.TemporaryDirectory() as tmp:
            r = self.fixture(tmp, Regress())
            r.run()
            self.assertEqual(r.state['rules'], [])
            self.assertEqual(r.state['best_rules'], [])
            self.assertIn('rolled_back', (Path(tmp)/'audit.jsonl').read_text())

    def test_bounded_retention_no_audit_rehydration(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.fixture(tmp)
            for i in range(100):
                r.state['episodes'].append(dict(id=str(i), sample=dict(question='x'*150, context='y'*150, target='yes'),
                                                prediction='yes', correct=True, cited=[], raw='z'*500))
                r.enforce()
                self.assertLessEqual(serialized_tokens(r.state, len), 6000)
            self.assertLess(len(r.state['episodes']), 100)
            self.assertTrue(all('raw' not in x for x in r.state['episodes']))
            r.save()
            (Path(tmp)/'audit.jsonl').write_text('not valid JSON and not readable memory')
            restored = self.fixture(tmp, state=json.loads((Path(tmp)/'checkpoint.json').read_text()))
            self.assertEqual(restored.state, r.state)

    def test_replace_corrects_and_merges_atomically(self):
        original = apply_patch([], patch(), {'train-000001'}, 1)
        original += apply_patch([], patch(), {'train-000001'}, 2)
        edit = transaction([r['id'] for r in original], 'corrected guidance')
        revised = apply_patch(original, edit, {'train-000001'}, 3)
        self.assertEqual(len(revised), 1)
        self.assertEqual(revised[0]['guidance'], 'corrected guidance')
        self.assertEqual(len(revised[0]['parent']), 2)
        self.assertEqual(len(original), 2)
        edit['remove_ids'].append('nonexistent')
        with self.assertRaises(ValueError):
            apply_patch(original, edit, {'train-000001'}, 4)

    def test_equal_score_compaction_can_release_capacity(self):
        class Compact(FakeAgents):
            def propose(self, rules, *args):
                return {'patches': [transaction([rules[0]['id']], 'short guidance')]}
        with tempfile.TemporaryDirectory() as tmp:
            r = self.fixture(tmp, Compact())
            r.state['rules'] = apply_patch([], transaction(guidance='long guidance '*40), {'train-000001'}, 1)
            r.state['step'] = 2
            r.state['episodes'] = [dict(id='train-000001', sample=r.splits['train'][0],
                                       prediction='yes', correct=True, cited=[])]
            before = serialized_tokens(r.state['rules'], len)
            r.refine()
            self.assertIsNotNone(r.state['trial'])
            self.assertLess(serialized_tokens(r.state['rules'], len), before)

    def test_resume_preserves_trial_and_completes_confirmation(self):
        class Interrupt(FakeAgents):
            def reflect(self, episode):
                if episode['sample']['question'] == 'train2':
                    raise RuntimeError('interrupt')
                return 'lesson'
        with tempfile.TemporaryDirectory() as tmp:
            r = self.fixture(tmp, Interrupt())
            with self.assertRaises(RuntimeError):
                r.run()
            state = json.loads((Path(tmp)/'checkpoint.json').read_text())
            self.assertEqual(state['step'], 2)
            self.assertIsNotNone(state['trial'])
            resumed = self.fixture(tmp, state=state)
            self.assertEqual(resumed.run(), 1)
            self.assertEqual(resumed.agents.reflections, ['train2', 'train3'])

    def test_parent_and_best_count_toward_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.fixture(tmp)
            state = dict(r.state, trial={'parent': 'x'*6000})
            with self.assertRaises(ValueError):
                fit_state(state, r.c, len)

    def test_reject_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.fixture(tmp, state=dict(step=1, rules=[], episodes=[], pending=[]))


if __name__ == '__main__':
    unittest.main()
