"""Offline evolution, guarded training replay, immutable evaluation and resume."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import random

from .memory import apply_patch, compile_playbook, similarity


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


class Runner:
    def __init__(self, config, processor, splits, agents, output, token_count, state=None):
        self.c, self.processor, self.splits, self.agents = config, processor, splits, agents
        self.output, self.token_count = Path(output), token_count
        self.state = state or dict(step=0, rules=[], episodes=[], pending=[], updates=[],
                                   validation=[], best_score=-1, best_rules=[], best_step=0)

    def save(self):
        atomic_json(self.output/'checkpoint.json', self.state)

    def evaluate(self, samples, rules):
        context = compile_playbook(rules)
        with ThreadPoolExecutor(max_workers=self.c['workers']) as pool:
            records = list(pool.map(lambda s: self.agents.generate(s, context), samples))
        score = self.processor.evaluate_accuracy([r['prediction'] for r in records], [s['target'] for s in samples])
        return score, records

    def replay_set(self):
        episodes = self.state['episodes']
        failures = [e for e in episodes if not e['correct']]
        if not failures:
            anchor = episodes[-1]
        else:
            anchor = failures[-1]
        text = lambda e: e['sample']['question'] + '\n' + e['sample']['context']
        order = sorted(episodes, key=lambda e: (-similarity(text(anchor), text(e)), e['id']))
        n = self.c['replay_size']
        # Reserve guards independently of similarity to detect broad regressions.
        successes = [e for e in episodes if e['correct'] and e['id'] != anchor['id']]
        random.Random(self.c['seed'] + self.state['step']).shuffle(successes)
        guards = successes[:n//4]
        selected = [anchor] + guards
        ids = {e['id'] for e in selected}
        for e in order:
            if len(selected) >= n:
                break
            if e['id'] not in ids:
                selected.append(e)
                ids.add(e['id'])
        rules = self.state['rules']
        target = next((r['id'] for r in rules if r['id'] in anchor['cited']), None)
        if target is None and rules:
            target = max(rules, key=lambda r: similarity(text(anchor), r['condition']+' '+r['guidance']))['id']
        return selected, {e['id'] for e in guards}, target

    def refine(self):
        evidence, guards, target = self.replay_set()
        event = dict(step=self.state['step'], evidence_ids=[e['id'] for e in evidence],
                     guard_ids=sorted(guards), target_id=target, candidates=[], accepted=None)
        try:
            proposal = self.agents.propose(self.state['rules'], evidence,
                                          self.state['pending'][-self.c['update_every']:],
                                          target, self.c['max_candidates'])
            patches = proposal.get('patches', [])
            if not isinstance(patches, list):
                raise ValueError('patches must be a list')
        except (ValueError, TypeError) as error:
            event['error'] = str(error)
            self.state['updates'].append(event)
            return
        samples = [e['sample'] for e in evidence]
        parent_score, parent_records = self.evaluate(samples, self.state['rules']) if patches else (None, [])
        best_score, best_rules = parent_score, None
        event['parent_score'] = parent_score
        event['parent_predictions'] = parent_records
        guard_indices = [i for i, e in enumerate(evidence) if e['id'] in guards]
        def guard_score(records):
            return self.processor.evaluate_accuracy([records[i]['prediction'] for i in guard_indices],
                                                    [samples[i]['target'] for i in guard_indices]) if guard_indices else 0
        for patch in patches[:self.c['max_candidates']]:
            report = dict(patch=patch)
            event['candidates'].append(report)
            try:
                if patch.get('operation') != 'add' and patch.get('target_id') != target:
                    raise ValueError('Edit outside selected target')
                if patch.get('operation') == 'split' and not ({e['correct'] for e in evidence} == {True, False}):
                    raise ValueError('Split needs successful and failed evidence')
                candidate = apply_patch(self.state['rules'], patch, set(event['evidence_ids']), self.state['step'])
                if self.token_count(compile_playbook(candidate)) > self.c['playbook_token_budget']:
                    raise ValueError('Playbook token budget exceeded')
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                report['rejected'] = str(error)
                continue
            score, records = self.evaluate(samples, candidate)
            report.update(score=score, guard_score=guard_score(records), predictions=records)
            if score > best_score and guard_score(records) >= guard_score(parent_records):
                best_score, best_rules = score, candidate
                event['accepted'] = len(event['candidates']) - 1
        if best_rules is not None:
            self.state['rules'] = best_rules
            # Pending evidence remains available in episodes; remove only addressed reflections.
            addressed = set(event['candidates'][event['accepted']]['patch']['evidence_ids'])
            self.state['pending'] = [p for p in self.state['pending'] if p['episode_id'] not in addressed]
        self.state['updates'].append(event)

    def run(self):
        train = self.splits['train']
        total = len(train) * self.c['num_epochs']
        self.save()
        for index in range(self.state['step'], total):
            sample = train[index % len(train)]
            result = self.agents.generate(sample, compile_playbook(self.state['rules']))
            episode = dict(id=f'train-{index+1:06d}', sample=sample, **result,
                           correct=bool(self.processor.answer_is_correct(result['prediction'], sample['target'])),
                           rule_ids=[r['id'] for r in self.state['rules']])
            reflection = self.agents.reflect(episode)
            self.state['episodes'].append(episode)
            self.state['pending'].append(dict(episode_id=episode['id'], reflection=reflection))
            self.state['step'] = index + 1
            if (index + 1) % self.c['update_every'] == 0 or index + 1 == total:
                self.refine()
            if (index + 1) % self.c['eval_steps'] == 0 or index + 1 == total:
                score, records = self.evaluate(self.splits['val'], self.state['rules'])
                self.state['validation'].append(dict(step=index+1, accuracy=score))
                atomic_json(self.output/f'validation_{index+1}.json', dict(accuracy=score, predictions=records))
                if score > self.state['best_score']:
                    self.state.update(best_score=score, best_rules=deepcopy(self.state['rules']), best_step=index+1)
            self.save()
            if (index+1) % self.c['save_steps'] == 0:
                atomic_json(self.output/f'snapshot_{index+1}.json', self.state)
            print(f"step {index+1}/{total}, rules={len(self.state['rules'])}, best_val={self.state['best_score']:.4f}", flush=True)
        # Test cannot feed the reflector, proposals, memory, or checkpoint selection.
        score, records = self.evaluate(self.splits['test'], self.state['best_rules'])
        atomic_json(self.output/'final_test_results.json', dict(accuracy=score, total=len(records),
                    selected_step=self.state['best_step'], predictions=records))
        atomic_json(self.output/'best_playbook.json', self.state['best_rules'])
        (self.output/'best_playbook.txt').write_text(compile_playbook(self.state['best_rules']), encoding='utf-8')
        usage = dict(calls=0, prompt_tokens=0, completion_tokens=0, seconds=0.0, by_role={})
        log = self.output/'calls.jsonl'
        if log.exists():
            for line in log.read_text(encoding='utf-8').splitlines():
                record = json.loads(line)
                usage['calls'] += 1
                usage['seconds'] += record['seconds']
                usage['by_role'][record['role']] = usage['by_role'].get(record['role'], 0) + 1
                for key in ('prompt_tokens', 'completion_tokens'):
                    usage[key] += (record.get('usage') or {}).get(key, 0)
        atomic_json(self.output/'usage.json', usage)
        return score
