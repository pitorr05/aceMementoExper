"""Workspace LLM utility: every auxiliary call uses the same local Qwen endpoint."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from openai import OpenAI


def call_llm(prompts, schema=None):
    single=isinstance(prompts,str)
    prompts=[prompts] if single else prompts
    if not isinstance(prompts,list) or len(prompts)>100:
        raise ValueError('Expected at most 100 prompts')
    def run(prompt):
        if schema:
            prompt+='\nReturn JSON matching this schema: '+json.dumps(schema.model_json_schema())
        with OpenAI(base_url=os.environ['VLLM_BASE_URL'],api_key=os.getenv('VLLM_API_KEY','EMPTY'),
                    max_retries=0,timeout=300) as client:
            response=client.chat.completions.create(model=os.environ['MCE_QWEN_MODEL'],
                messages=[dict(role='user',content=prompt)],temperature=0,
                max_tokens=int(os.getenv('MCE_MAX_TOKENS','4096')))
            text=response.choices[0].message.content or ''
        if schema:
            if text.strip().startswith('```'):
                text=text.strip().split('\n',1)[1].rsplit('```',1)[0]
            return schema.model_validate_json(text)
        return text
    with ThreadPoolExecutor(max_workers=10) as pool:
        results=list(pool.map(run,prompts))
    return results[0] if single else results
