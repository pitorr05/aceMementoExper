"""OpenAI tool-call transport implementing the SDK methods used by upstream MCE.

This is a replacement runtime, not the Claude SDK. It preserves upstream prompts,
file/code outputs, permission callbacks and validation/repair conversations.
Generated Python/Bash executes with the OS user's permissions (not an OS sandbox).
"""
import asyncio
from dataclasses import dataclass
import glob
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from types import ModuleType
from types import SimpleNamespace


@dataclass
class TextBlock:
    text: str


@dataclass
class AssistantMessage:
    content: list
    model: str


class ClaudeAgentOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def schema(name, description, fields, required):
    return dict(type='function', function=dict(name=name, description=description,
                parameters=dict(type='object', properties=fields, required=required, additionalProperties=False)))


STRING = {'type': 'string'}
TOOLS = [
    schema('Read', 'Read a UTF-8 file; offset is one-based line number.',
           dict(file_path=STRING, offset={'type':'integer'}, limit={'type':'integer'}), ['file_path']),
    schema('Write', 'Write a complete UTF-8 file, creating parent directories.',
           dict(file_path=STRING, content=STRING), ['file_path','content']),
    schema('Edit', 'Replace exact old_string in a file. Default requires exactly one occurrence.',
           dict(file_path=STRING, old_string=STRING, new_string=STRING, replace_all={'type':'boolean'}),
           ['file_path','old_string','new_string']),
    schema('Glob', 'Find files under path matching a glob pattern.', dict(path=STRING, pattern=STRING), ['pattern']),
    schema('Grep', 'Search file text with a regular expression.', dict(path=STRING, pattern=STRING), ['pattern']),
    schema('Bash', 'Execute a short shell command in the working directory to inspect or verify generated code.',
           dict(command=STRING, timeout={'type':'integer'}), ['command']),
    schema('Skill', 'Read a skill in .claude/skills. Use learning-context for the learning skill.',
           dict(skill=STRING), ['skill']),
]


class QwenSDKClient:
    config = None
    output = None
    client_factory = None  # Injection point for deterministic transport tests.

    def __init__(self, options):
        from openai import OpenAI
        self.options = options
        self.cwd = Path(options.cwd).resolve()
        self.messages = [dict(role='system', content=(
            'You are the MCE context engineering agent. Use the supplied filesystem tools to complete '
            'the requested artifacts. Use native tool calls, not XML or imagined tool results. '
            'Read the learning-context skill before base-level context engineering. '
            'Only available tools are listed; no subagents are available. Finish only after writing files. '
            'Use Bash for short code checks; write source with Write. Working directory: '+str(self.cwd)))]
        factory = self.client_factory or OpenAI
        self.client = factory(base_url=os.getenv('VLLM_BASE_URL', 'http://127.0.0.1:5000/v1'),
                              api_key=os.getenv('VLLM_API_KEY', 'EMPTY'), max_retries=0,
                              timeout=self.config.get('timeout_seconds',300))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.client.close()

    async def query(self, prompt):
        self.messages.append(dict(role='user', content=prompt))

    def log(self, record):
        with (Path(self.output)/'optimizer_calls.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record,ensure_ascii=False)+'\n')

    async def receive_response(self):
        tools = [t for t in TOOLS if t['function']['name'] in self.options.allowed_tools]
        json_tools = self.config.get('tool_transport','json') == 'json'
        if json_tools:
            instruction = ('Tool protocol: reply with JSON {"text":"optional explanation",'
                           '"tool_calls":[{"name":"Read", "arguments":{"file_path":"..."}}]} '
                           'to run tools. Reply {"text":"finished", "tool_calls":[]} only after writing '
                           'and verifying artifacts. Available tool schemas:\n'+json.dumps(tools))
            if not any(m.get('content') == instruction for m in self.messages):
                self.messages.insert(1,dict(role='system',content=instruction))
        for turn in range(self.config.get('agent_max_turns', 30)):
            start = time.monotonic()
            kwargs = {} if json_tools else dict(tools=tools,tool_choice='auto')
            response = await asyncio.to_thread(self.client.chat.completions.create,
                    model=self.config['model'], messages=self.messages,
                    temperature=0.0, max_tokens=self.config['max_tokens'], **kwargs)
            message = response.choices[0].message
            self.log(dict(cwd=str(self.cwd), turn=turn, seconds=time.monotonic()-start,
                          messages=self.messages, response=message.model_dump(),
                          usage=response.usage.model_dump() if response.usage else None))
            if response.choices[0].finish_reason == 'length':
                raise RuntimeError('Qwen tool response truncated; increase max_tokens or simplify agent work')
            calls = message.tool_calls or []
            text = message.content or ''
            if json_tools:
                from idea5.agents import parse_object
                try:
                    payload=parse_object(text)
                    entries=payload.get('tool_calls',[])
                    if not isinstance(entries,list): raise ValueError('tool_calls must be a list')
                    calls=[SimpleNamespace(id=f'call_{turn}_{i}',function=SimpleNamespace(
                          name=x['name'],arguments=json.dumps(x['arguments']))) for i,x in enumerate(entries)]
                    text=payload.get('text','')
                except (ValueError,TypeError,KeyError) as error:
                    self.messages.extend([dict(role='assistant',content=message.content or ''),
                        dict(role='user',content='Invalid tool JSON: '+str(error)+'. Return the specified JSON tool protocol.')])
                    continue
            assistant = dict(role='assistant',content=message.content or '')
            if calls and not json_tools:
                assistant['tool_calls'] = [c.model_dump() for c in calls]
            self.messages.append(assistant)
            if text:
                yield AssistantMessage([TextBlock(text)], self.config['model'])
            if not calls:
                if not text:
                    raise RuntimeError('Empty Qwen optimizer response')
                return  # Upstream verifies files/interfaces and sends repair feedback if needed.
            for call in calls:
                try:
                    args = json.loads(call.function.arguments)
                    if not isinstance(args,dict):
                        raise ValueError('Tool arguments must be an object')
                    result = await self.tool(call.function.name,args)
                except (ValueError,TypeError,OSError,KeyError,re.error) as error:
                    result = 'TOOL ERROR: '+str(error)
                self.log(dict(tool=call.function.name, arguments=call.function.arguments, result=result))
                if json_tools:
                    self.messages.append(dict(role='user',content='Tool result '+call.function.name+':\n'+result[:16000]))
                else:
                    self.messages.append(dict(role='tool',tool_call_id=call.id,content=result[:16000]))
                yield AssistantMessage([TextBlock(call.function.name+': '+result[:500])],self.config['model'])
        raise RuntimeError('MCE optimizer exceeded agent_max_turns; no silent partial success')

    def path(self,value):
        path = Path(value)
        return (path if path.is_absolute() else self.cwd/path).resolve()

    async def tool(self,name,args):
        if name not in self.options.allowed_tools or name not in {t['function']['name'] for t in TOOLS}:
            raise ValueError('Unavailable tool '+name)
        args = dict(args)
        if name == 'Skill':
            skill = args['skill']
            if not re.fullmatch(r'[\w-]+',skill):
                raise ValueError('Invalid skill name')
            return await self.tool('Read',dict(file_path=str(self.cwd/'.claude/skills'/skill/'SKILL.md')))
        for field in ('file_path','path'):
            if field in args:
                args[field] = str(self.path(args[field]))
        if name in ('Glob','Grep') and 'path' not in args:
            args['path'] = str(self.cwd)
        callback = getattr(self.options,'can_use_tool',None)
        if callback:
            decision = await callback(name,args,{})
            if decision.get('behavior') != 'allow':
                raise ValueError(decision.get('message','Tool denied'))
            args = decision.get('updatedInput',args)
        if name == 'Read':
            lines = Path(args['file_path']).read_text(encoding='utf-8').splitlines()
            offset = max(0,args.get('offset',1)-1)
            return '\n'.join(lines[offset:offset+max(1,args.get('limit',250))])
        if name == 'Write':
            path = Path(args['file_path']); path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(args['content'],encoding='utf-8')
            return 'Written '+str(path)
        if name == 'Edit':
            path = Path(args['file_path']); text = path.read_text(encoding='utf-8')
            old = args['old_string']
            if not old or old not in text or (not args.get('replace_all') and text.count(old)!=1):
                raise ValueError('old_string must match exactly; use unique context')
            path.write_text(text.replace(old,args['new_string'], -1 if args.get('replace_all') else 1),encoding='utf-8')
            return 'Edited '+str(path)
        if name == 'Glob':
            root = self.path(args['path'])
            paths = sorted(root.glob(args['pattern']))
            # Glob patterns cannot escape the authorized root.
            return '\n'.join(str(p) for p in paths if p.resolve().is_relative_to(root))[:16000]
        if name == 'Grep':
            root=self.path(args['path']); pattern=re.compile(args['pattern'])
            lines=[]
            for path in ([root] if root.is_file() else root.rglob('*')):
                if not path.is_file() or not path.resolve().is_relative_to(root if root.is_dir() else root.parent):
                    continue
                try:
                    for i,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
                        if pattern.search(line): lines.append(f'{path}:{i}:{line}')
                except (UnicodeError,OSError): continue
                if sum(map(len,lines))>=16000: break
            return '\n'.join(lines)[:16000]
        if name == 'Bash':
            return await asyncio.to_thread(self.shell,args['command'],min(args.get('timeout',60),120))
        raise ValueError('Unsupported tool '+name)

    def shell(self,command,timeout):
        if os.name == 'nt':
            raise ValueError('Qwen MCE Bash tool requires Linux; filesystem tools still work')
        # Runtime permission callbacks are not an OS isolation boundary, as in
        # upstream MCE's local SDK execution. Terminate the whole job on timeout.
        process = subprocess.Popen(['/bin/bash','-lc',command],cwd=self.cwd,
                                   stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        try:
            output,_=process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.communicate()
            return 'TOOL ERROR: command timeout'
        return f'exit_code={process.returncode}\n'+output[:16000]


def install(config,output):
    """Install only for this explicit Qwen MCE process before upstream imports."""
    QwenSDKClient.config, QwenSDKClient.output = config, output
    shim=ModuleType('claude_agent_sdk')
    shim.ClaudeAgentOptions=ClaudeAgentOptions
    shim.ClaudeSDKClient=QwenSDKClient
    async def query(*args,**kwargs):
        raise NotImplementedError('Upstream streaming query shortcut is not used by this MCE entry point')
    shim.query=query
    sys.modules['claude_agent_sdk']=shim
