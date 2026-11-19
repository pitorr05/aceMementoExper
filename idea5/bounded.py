"""Budgeted knowledge transactions with prospective training-only confirmation."""
from copy import deepcopy
import json
from .runner import Runner, atomic_json
from .memory import apply_patch, compile_playbook
from .budget import fit_state, serialized_tokens


class BoundedRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (kwargs.get('state') is not None and self.state.get('schema_version') != 2) or self.state.get('updates'):
            raise ValueError('v1 checkpoint cannot resume into bounded-memory training; start a new run')
        self.state.pop('updates', None)
        self.state.pop('validation', None)
        self.state.setdefault('schema_version', 2)
        self.state.setdefault('trial', None)
        self.enforce()

    def audit(self, kind, record):
        with (self.output/'audit.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(kind=kind, step=self.state['step'], **record), ensure_ascii=False)+'\n')

    def enforce(self):
        self.state, evicted = fit_state(self.state, self.c, self.token_count)
        if evicted:
            self.audit('evidence_evicted', dict(ids=evicted))

    def score(self, predictions, targets):
        return self.processor.evaluate_accuracy(predictions, targets)

    def observe_trial(self, sample, result):
        trial = self.state['trial']
        if trial is None:
            return
        # The task was answered before either branch sees this target/feedback.
        parent = self.agents.generate(sample, compile_playbook(trial['parent']))
        trial['candidate_answers'].append(result['prediction'])
        trial['parent_answers'].append(parent['prediction'])
        trial['targets'].append(sample['target'])
        if len(trial['targets']) >= self.c['confirmation_examples']:
            self.settle_trial()

    def settle_trial(self, incomplete=False):
        trial = self.state['trial']
        if trial is None:
            return
        candidate = self.score(trial['candidate_answers'], trial['targets']) if trial['targets'] else None
        parent = self.score(trial['parent_answers'], trial['targets']) if trial['targets'] else None
        accepted = not incomplete and candidate >= parent
        if not accepted:
            self.state['rules'] = trial['parent']
        self.audit('confirmed' if accepted else 'rolled_back', dict(
            transaction_step=trial['step'], candidate_score=candidate, parent_score=parent,
            incomplete=incomplete, trial=trial))
        self.state['trial'] = None

    def refine(self):
        if self.state['trial'] is not None or not self.state['episodes']:
            return
        evidence, guards, target = self.replay_set()
        event = dict(target=target, evidence_ids=[e['id'] for e in evidence], candidates=[], accepted=None)
        try:
            proposal = self.agents.propose(self.state['rules'], evidence, self.state['pending'], target,
                                          self.c['max_candidates'])
            patches = proposal.get('patches', [])
            if not isinstance(patches, list):
                raise ValueError('patches must be a list')
        except (ValueError, TypeError, AttributeError) as error:
            self.audit('proposal_rejected', dict(error=str(error)))
            return
        if not patches:
            return
        samples = [e['sample'] for e in evidence]
        parent_score, parent_records = self.evaluate(samples, self.state['rules'])
        guard_indices = [i for i, e in enumerate(evidence) if e['id'] in guards]
        def guard_score(records):
            return self.score([records[i]['prediction'] for i in guard_indices],
                              [samples[i]['target'] for i in guard_indices]) if guard_indices else 0
        old_tokens = serialized_tokens(self.state['rules'], self.token_count)
        best_key = (parent_score, -old_tokens)
        best_state = None
        event.update(parent_score=parent_score, parent_predictions=parent_records)
        for patch in patches[:self.c['max_candidates']]:
            report = dict(patch=patch)
            event['candidates'].append(report)
            try:
                candidate = apply_patch(self.state['rules'], patch, set(event['evidence_ids']), self.state['step'])
                if serialized_tokens(candidate, self.token_count) > self.c['knowledge_token_budget']:
                    raise ValueError('Knowledge allocation exceeded; consolidate or replace units')
                if self.token_count(compile_playbook(candidate)) > self.c['playbook_token_budget']:
                    raise ValueError('Compiled knowledge budget exceeded')
                proposed = deepcopy(self.state)
                proposed['rules'] = candidate
                proposed['trial'] = dict(parent=deepcopy(self.state['rules']), step=self.state['step'],
                                          parent_answers=[], candidate_answers=[], targets=[])
                proposed, evicted = fit_state(proposed, self.c, self.token_count)
                retained = {e['id'] for e in proposed['episodes']}
                if not set(patch['evidence_ids']) & retained:
                    raise ValueError('Transaction would lose all its supporting evidence')
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                report['rejected'] = str(error)
                continue
            score, records = self.evaluate(samples, candidate)
            key = (score, -serialized_tokens(candidate, self.token_count))
            report.update(score=score, guard_score=guard_score(records), predictions=records, evicted=evicted)
            if key > best_key and guard_score(records) >= guard_score(parent_records):
                best_key, best_state = key, proposed
                event['accepted'] = len(event['candidates'])-1
        if best_state is not None:
            self.state = best_state
        self.audit('transaction_replay', event)

    def validate(self):
        # Only confirmed knowledge is eligible for checkpoint selection.
        if self.state['trial'] is not None:
            rules = self.state['trial']['parent']
        else:
            rules = self.state['rules']
        score, records = self.evaluate(self.splits['val'], rules)
        atomic_json(self.output/f"validation_{self.state['step']}.json", dict(accuracy=score, predictions=records))
        if score > self.state['best_score']:
            proposed = deepcopy(self.state)
            proposed.update(best_score=score, best_rules=deepcopy(rules), best_step=self.state['step'])
            try:
                proposed, evicted = fit_state(proposed, self.c, self.token_count)
                self.state = proposed
            except ValueError:
                # Roll back an unconfirmed branch to reserve the selected checkpoint.
                self.settle_trial(incomplete=True)
                self.state.update(best_score=score, best_rules=deepcopy(rules), best_step=self.state['step'])
                self.enforce()

    def run(self):
        train = self.splits['train']
        total = len(train)*self.c['num_epochs']
        self.save()
        for index in range(self.state['step'], total):
            sample = train[index % len(train)]
            result = self.agents.generate(sample, compile_playbook(self.state['rules']))
            self.state['step'] = index+1
            self.observe_trial(sample, result)
            episode = dict(id=f'train-{index+1:06d}', sample=sample, **result,
                           correct=bool(self.processor.answer_is_correct(result['prediction'], sample['target'])))
            reflection = self.agents.reflect(episode)
            self.audit('episode', dict(episode=episode, reflection=reflection))
            self.state['episodes'].append(episode)
            self.state['pending'].append(dict(episode_id=episode['id'], reflection=reflection))
            try:
                self.enforce()
            except ValueError:
                self.settle_trial(incomplete=True)
                self.enforce()
            # Do not start a trial with insufficient unseen training examples left.
            if (index+1) % self.c['update_every'] == 0 and total-index-1 >= self.c['confirmation_examples']:
                self.refine()
            if index+1 == total:
                self.settle_trial(incomplete=True)
            if (index+1) % self.c['eval_steps'] == 0 or index+1 == total:
                self.validate()
            self.enforce()
            self.save()
            if (index+1) % self.c['save_steps'] == 0:
                atomic_json(self.output/f'snapshot_{index+1}.json', self.state)
            print(f"step {index+1}/{total}, units={len(self.state['rules'])}, memory_tokens="
                  f"{serialized_tokens(self.state, self.token_count)}/{self.c['memory_token_budget']}", flush=True)
        score, records = self.evaluate(self.splits['test'], self.state['best_rules'])
        atomic_json(self.output/'final_test_results.json', dict(accuracy=score, total=len(records),
                    selected_step=self.state['best_step'], predictions=records))
        atomic_json(self.output/'best_playbook.json', self.state['best_rules'])
        (self.output/'best_playbook.txt').write_text(compile_playbook(self.state['best_rules']), encoding='utf-8')
        usage = dict(calls=0, prompt_tokens=0, completion_tokens=0, seconds=0.0, by_role={})
        log = self.output/'calls.jsonl'
        if log.exists():
            # Reporting only: no audit/call record enters the learning state or prompts.
            with log.open(encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    usage['calls'] += 1
                    usage['seconds'] += record['seconds']
                    usage['by_role'][record['role']] = usage['by_role'].get(record['role'], 0)+1
                    for key in ('prompt_tokens', 'completion_tokens'):
                        usage[key] += (record.get('usage') or {}).get(key, 0)
        atomic_json(self.output/'usage.json', usage)
        return score
