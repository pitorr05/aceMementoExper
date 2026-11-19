import asyncio
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
import logging
from unittest.mock import patch

from mce_ace.runtime import QwenSDKClient, ClaudeAgentOptions


class Message:
    def __init__(self,text):
        self.content=text
        self.tool_calls=[]
    def model_dump(self):
        return dict(content=self.content,tool_calls=[])


class FakeOptimizer:
    requests=[]
    def __init__(self,**kwargs):
        self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def close(self): pass
    def create(self,**kwargs):
        messages=kwargs['messages']
        self.requests.append(json.loads(json.dumps(messages)))
        initial=next(m['content'] for m in messages if m['role']=='user')
        cwd=Path(messages[0]['content'].split('Working directory: ')[1])
        already_wrote=any(m['role']=='user' and m['content'].startswith('Tool result Write') for m in messages)
        if already_wrote:
            result=dict(text='Artifacts completed.',tool_calls=[])
        else:
            if 'Meta-Level Agent' in initial:
                skill_path=re.search(r'`([^`\n]+/\.claude/skills/learning-context/SKILL\.md)`',initial)[1]
                writes=[(skill_path,'## Skill Overview\nLearn reusable patterns.\n## Methodology\nAnalyze errors.')]
            else:
                code='def get_context(question: str, context: str) -> str:\n    return "learned context"\n'
                writes=[(str(cwd/'interfaces/get_context.py'),code),(str(cwd/'interfaces/__init__.py'),code),
                        (str(cwd/'context/notes.md'),'Reusable knowledge')]
            result=dict(text='Writing artifacts.',tool_calls=[dict(name='Write',arguments=dict(file_path=p,content=c)) for p,c in writes])
        return SimpleNamespace(choices=[SimpleNamespace(message=Message(json.dumps(result)),finish_reason='stop')],usage=None)


class LocalGenerator:
    def __init__(self,config,output): pass
    def call(self,prompt,role):
        return '{"final_answer":"1.0", "bullet_ids":[]}'


class QwenMCETests(unittest.TestCase):
    def test_invalid_python_write_preserves_previous_interface(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'get_context.py'
            original = 'def get_context(question, context):\n    return ""\n'
            path.write_text(original, encoding='utf-8')
            client = object.__new__(QwenSDKClient)
            client.cwd = Path(folder)
            client.options = ClaudeAgentOptions(allowed_tools=['Write'])
            with self.assertRaisesRegex(ValueError, 'previous file preserved'):
                asyncio.run(client.tool('Write', dict(file_path=str(path),
                    content=r'def get_context(question, context):\n    return ""')))
            self.assertEqual(path.read_text(encoding='utf-8'), original)

    def tearDown(self):
        for logger in logging.Logger.manager.loggerDict.values():
            if isinstance(logger,logging.Logger):
                for handler in logger.handlers[:]:
                    if isinstance(handler,logging.FileHandler):
                        handler.close()
                        logger.removeHandler(handler)

    def test_original_mce_two_iterations_on_ace_fixture(self):
        from idea5.reproduce_mce import execute
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root=Path(tmp)
            paths={}
            for split,n in [('train',4),('val',2),('test',2)]:
                path=root/(split+'.jsonl')
                path.write_text('\n'.join(json.dumps(dict(context=f'Compute {i}.',target='1.0')) for i in range(n)))
                paths[split+'_data']=str(path)
            config=dict(task='formula',model='Qwen/Qwen3-4B-Instruct-2507',max_tokens=4096,
                        seed=42,workers=1,train_batch_size=2,agent_max_turns=5,tool_transport='json',data=paths)
            FakeOptimizer.requests=[]
            with patch.object(QwenSDKClient,'client_factory',staticmethod(FakeOptimizer)), \
                 patch('idea5.reproduce_mce.LocalModel',LocalGenerator):
                asyncio.run(execute(config,root/'run',iterations=2,backend='qwen'))
            result=json.loads((root/'run/final_test_results.json').read_text())
            self.assertEqual(result['summary']['primary_metric_value'],1)
            self.assertEqual(result['summary']['total'],2)
            self.assertEqual(len(json.loads((root/'run/iterations.json').read_text())),2)
            workspace=root/'run/workspace'
            self.assertTrue((workspace/'iter2_sub1/interfaces/get_context.py').exists())
            self.assertIn('VLLM_BASE_URL',(workspace/'iter2_sub1/utils/llm.py').read_text())
            self.assertIn('lexical',(workspace/'iter2_sub1/utils/embedding.py').read_text())
            evaluations=json.loads((workspace/'meta_agent/evaluations.json').read_text())
            self.assertEqual(set(evaluations),{'iter1','iter2'})
            self.assertTrue(any('Iteration 1' in m['content'] for request in FakeOptimizer.requests
                                for m in request if m['role']=='user'))

    def test_permission_callback_receives_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen=[]
            async def deny(name,args,context):
                seen.append(args)
                return dict(behavior='deny',message='outside scope')
            options=ClaudeAgentOptions(cwd=tmp,allowed_tools=['Write'],can_use_tool=deny)
            with patch.object(QwenSDKClient,'client_factory',staticmethod(FakeOptimizer)):
                QwenSDKClient.config={}
                client=QwenSDKClient(options)
                with self.assertRaises(ValueError):
                    asyncio.run(client.tool('Write',dict(file_path='../escape.txt',content='blocked')))
            self.assertTrue(Path(seen[0]['file_path']).is_absolute())
            self.assertFalse((Path(tmp).parent/'escape.txt').exists())


if __name__=='__main__':
    unittest.main()
