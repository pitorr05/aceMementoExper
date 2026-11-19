"""Serializable rules, bounded edits and deterministic compilation."""
from copy import deepcopy
import re


def words(text):
    return set(re.findall(r"\w+", text.lower()))


def similarity(a, b):
    a, b = words(a), words(b)
    return len(a & b) / max(1, len(a | b))


def compile_playbook(rules):
    groups = {}
    for rule in rules:
        groups.setdefault(rule['group'], []).append(rule)
    return '\n\n'.join(
        '## ' + group + '\n' + '\n'.join(
            f"[{r['id']}] When: {r['condition']}\nGuidance: {r['guidance']}"
            for r in entries) for group, entries in sorted(groups.items()))


def validate_rule(rule):
    if set(rule) != {'group', 'condition', 'guidance'}:
        raise ValueError('Rule needs exactly group, condition, guidance')
    for key, limit in [('group', 120), ('condition', 1000), ('guidance', 3000)]:
        if not isinstance(rule[key], str) or not rule[key].strip() or len(rule[key]) > limit:
            raise ValueError('Invalid rule field: ' + key)


def apply_patch(rules, patch, evidence_ids, step):
    if not isinstance(patch, dict) or set(patch) != {'operation', 'target_id', 'rules', 'rationale', 'evidence_ids'}:
        raise ValueError('Invalid patch schema')
    if not isinstance(patch['rationale'], str) or not patch['rationale'].strip():
        raise ValueError('Missing rationale')
    sources = patch['evidence_ids']
    if not isinstance(sources, list) or not sources or not all(isinstance(i, str) and i in evidence_ids for i in sources):
        raise ValueError('Evidence must reference supplied training episodes')
    op, target, replacements = patch['operation'], patch['target_id'], patch['rules']
    if op not in {'add', 'condition', 'split'} or not isinstance(replacements, list):
        raise ValueError('Unsupported operation')
    if len(replacements) != (2 if op == 'split' else 1):
        raise ValueError('Invalid number of replacement rules')
    for rule in replacements:
        validate_rule(rule)
    old = next((r for r in rules if r['id'] == target), None)
    if (op == 'add' and target is not None) or (op != 'add' and old is None):
        raise ValueError('Invalid target')
    if op == 'condition' and any(replacements[0][k] != old[k] for k in ('group', 'guidance')):
        raise ValueError('Condition edit cannot change guidance/group')
    result = deepcopy(rules)
    if old:
        result = [r for r in result if r['id'] != target]
    for index, rule in enumerate(replacements):
        result.append(dict(rule, id=f'rule-{step:06d}-{index}', sources=sources,
                           parent=target, created_step=step))
    return result
