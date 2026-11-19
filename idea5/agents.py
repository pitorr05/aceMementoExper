"""Local OpenAI-compatible model calls; legacy Generator prompt and parsing."""
import json
import os
import runpy
import time
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]


def parse_object(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object')
    return value


class LocalModel:
    def __init__(self, config, output):
        from openai import OpenAI
        self.client = OpenAI(base_url=os.environ.get('VLLM_BASE_URL', 'http://localhost:5000/v1'),
                             api_key=os.environ.get('VLLM_API_KEY', 'EMPTY'),
                             max_retries=0, timeout=config.get('timeout_seconds', 300))
        self.config, self.output, self.lock = config, Path(output), Lock()

    def call(self, prompt, role):
        start = time.monotonic()
        result = self.client.chat.completions.create(
            model=self.config['model'], messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0, max_tokens=self.config['max_tokens'])
        text = result.choices[0].message.content or ''
        record = dict(role=role, prompt=prompt, response=text,
                      seconds=time.monotonic()-start,
                      usage=result.usage.model_dump() if result.usage else None)
        with self.lock, (self.output / 'calls.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
        return text


class Agents:
    def __init__(self, model):
        self.model = model
        self.prompt = runpy.run_path(str(ROOT/'ace/prompts/generator.py'))['GENERATOR_PROMPT']

    def generate(self, sample, playbook):
        from utils import extract_answer
        raw = self.model.call(self.prompt.format(playbook, '(empty)', sample['question'], sample['context']), 'generator')
        try:
            cited = parse_object(raw).get('bullet_ids', [])
            if not isinstance(cited, list):
                cited = []
        except (ValueError, TypeError):
            cited = []
        return dict(prediction=extract_answer(raw), raw=raw, cited=cited)

    def reflect(self, episode):
        prompt = ('Analyze this scored TRAINING episode. Distinguish model output from verified target. '
                  'Identify a reusable lesson and an observable input condition. Do not merely memorize '
                  'the answer. Output JSON {"lesson": str, "condition": str, "failure": str}.\n'
                  + json.dumps(episode, ensure_ascii=False))
        return self.model.call(prompt, 'reflector')

    def propose(self, rules, evidence, candidates, target, max_candidates):
        # Full traces remain in the checkpoint; constrain the meta-agent's working input.
        evidence = [dict(e, raw=e.get('raw', '')[:4000]) for e in evidence]
        candidates = [dict(c, reflection=c['reflection'][:2000]) for c in candidates]
        prompt = '''You are the meta-agent learning the structure of a Playbook.
Use scored training episodes and pending reflections to propose at most MAX_PATCHES alternative edits.
Propose atomic replacement transactions. Each transaction removes zero to four existing units
and inserts one to four units. You may correct guidance, refine conditions, split a broad unit,
or merge redundant units to make room for new knowledge. Start with relevant target TARGET,
but include redundant units elsewhere when consolidation releases useful capacity.
Use correct and failed episodes to preserve distinguishing conditions and important exceptions.
Equal-accuracy smaller representations can be accepted; after replay, transactions are tested
on later training episodes and rolled back if they regress. Prefer compact reusable knowledge.
Conditions must use observable question/context features, never labels unavailable at inference.
Keep rules reusable; do not embed a lookup table of training answers. Two different labels alone
do not justify splitting a rule. Explain the distinguishing input feature and evidence.
Proposals are alternatives, not a sequence. If evidence is insufficient, return {"patches": []}.
Output JSON {"patches": [{"operation":"replace", "remove_ids":["existing-rule-id"],
"rules":[{"group":"topic", "condition":"input condition", "guidance":"instruction"}],
"rationale":"why", "evidence_ids":["episode-id"]}]}.
'''.replace('MAX_PATCHES', str(max_candidates)).replace('TARGET', str(target))
        prefix = self.learning_prefix() if hasattr(self, 'learning_prefix') else ''
        return parse_object(self.model.call(prefix + prompt + '\n' + json.dumps(
            dict(playbook=rules, evidence=evidence, pending=candidates,
                 budgets={k: self.model.config.get(k) for k in ('memory_token_budget',
                          'knowledge_token_budget', 'evidence_token_budget')}
                 if hasattr(self.model, 'config') else {}), ensure_ascii=False),
                 'base_context' if prefix else 'meta'))
