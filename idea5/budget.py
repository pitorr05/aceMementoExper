"""Bound all readable persistent state, including rollback and selected snapshots.

Audit logs are write-only to the learning algorithm. Current task input and model
working buffers are transient; they are not a historical retrieval store.
"""
from copy import deepcopy
import json
from .memory import similarity


def serialized_tokens(value, count):
    return count(json.dumps(value, ensure_ascii=False, separators=(',', ':')))


def fit_state(state, config, count):
    """Return a bounded copy or fail; never discard knowledge/rollback implicitly."""
    result = deepcopy(state)
    evicted = []
    def evidence_cost():
        return serialized_tokens([result['episodes'], result['pending']], count)
    def over():
        return (serialized_tokens(result, count) > config['memory_token_budget'] or
                evidence_cost() > config['evidence_token_budget'])
    # Avoid retaining verbose generated traces as the sole evidence of correctness.
    for episode in result['episodes']:
        episode.pop('raw', None)
        episode.pop('rule_ids', None)
    while over() and result['pending']:
        result['pending'].pop(0)
    while over() and result['episodes']:
        episodes = result['episodes']
        def value(e):
            text = e['sample']['question'] + '\n' + e['sample']['context']
            redundancy = max((similarity(text, x['sample']['question']+'\n'+x['sample']['context'])
                              for x in episodes if x is not e), default=0)
            banks = result['rules'] + result['best_rules'] + (result.get('trial') or {}).get('parent', [])
            referenced = any(e['id'] in r.get('sources', []) for r in banks)
            recency = episodes.index(e) / max(1, len(episodes)-1)
            return 2*referenced + (not e['correct']) + (1-redundancy) + 0.25*recency
        victim = min(episodes, key=value)
        evicted.append(victim['id'])
        episodes.remove(victim)
    if over():
        raise ValueError('Knowledge, selection and rollback state exceed total memory budget')
    return result, evicted
