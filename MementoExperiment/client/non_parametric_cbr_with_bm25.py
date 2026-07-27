from __future__ import annotations
from tqdm import tqdm
import asyncio
import sys
import json
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from openai import AsyncAzureOpenAI
import tiktoken
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging
import colorlog
from transformers import AutoTokenizer, AutoModel
import numpy as np
from rank_bm25 import BM25Okapi
LOG_FORMAT = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
colorlog.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
MAX_CTX = 175000
EXE_MODEL = os.getenv("EXECUTOR_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")


BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.5"))
USE_HYBRID = os.getenv("USE_HYBRID_RETRIEVAL", "True").lower() == "true"

PROMPT_TPL = '''You will be given a question and its ground truth answer list where each item can be a ground truth answer. Provided a pred_answer, you need to judge if the pred_answer correctly answers the question based on the ground truth answer list.
You should first give your rationale for the judgement, and then give your judgement result (i.e., correct or incorrect).

Here is the criteria for the judgement:
1. The pred_answer doesn't need to be exactly the same as any of the ground truth answers, but should be semantically same for the question.
2. Each item in the ground truth answer list can be viewed as a ground truth answer for the question, and the pred_answer should be semantically same to at least one of them.

question: {question}
ground truth answers: {gt_answer}
pred_answer: {pred_answer}

The output should in the following json format:

{{
  "rationale": "...",
  "judgement": "correct" | "incorrect"
}}
'''


query_list: List[str] = []
ground_truth_map: Dict[str, Any] = {}


DATASET_PATH = os.getenv("DATASET_PATH", "../data/deepresearcher.jsonl")

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 400:
            break
        data = json.loads(line)
        q = data['question']
        query_list.append(q)
        ground_truth_map[q] = data.get("ground_truth", None)


server_paths: list[str] = [
    "../server/code_agent.py",
    "../server/ai_crawl.py",
    "../server/documents_tool.py",
    "../server/image_tool.py",
    "../server/math_tool.py",
    "../server/serp_search.py",
    "../server/video_tool.py",
]

load_dotenv()


META_SYSTEM_PROMPT = (
    "You are the META-PLANNER in a hierarchical AI system. A user will ask a\n"
    "high-level question. **First**: break the problem into a *minimal sequence*\n"
    "of executable tasks. Reply ONLY in JSON with the schema:\n"
    "{ \"plan\": [ {\"id\": INT, \"description\": STRING} … ] }\n\n"
    "After each task is executed by the EXECUTOR you will receive its result.\n"
    "Please carefully consider the descriptions of the time of web pages and events in the task, and take these factors into account when planning and giving the final answer.\n"
    "If the final answer is complete, output it with the template:\n"
    "FINAL ANSWER: <answer>\n\n"
    " YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"
    "Please ensure that the final answer strictly follows the question requirements, without any additional analysis.\n"
    "If the final answer is not complete, emit a *new* JSON plan for the remaining work. Keep cycles as\n"
    "few as possible. Never call tools yourself — that's the EXECUTOR's job."
    "  Reply with *pure JSON only*."
)

EXEC_SYSTEM_PROMPT = (
    "You are the EXECUTOR sub-agent. You receive one task description at a time\n"
    "from the meta-planner. Your job is to complete the task, using available\n"
    "tools via function calling if needed. Always think step by step but reply\n"
    "with the minimal content needed for the meta-planner. If you must call a\n"
    "tool, produce the appropriate function call instead of natural language.\n"
    "When done, output a concise result. Do NOT output FINAL ANSWER."
)

# ============ MEMORY CONFIG ============
MEMORY_JSONL_PATH = os.getenv("MEMORY_JSONL_PATH", "../memory/dummy_memo.jsonl")
MEMORY_KEY_FIELD = os.getenv("MEMORY_KEY_FIELD", "question")
MEMORY_VALUE_FIELD = os.getenv("MEMORY_VALUE_FIELD", "plan")
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "4"))
MEMORY_DEVICE = os.getenv("MEMORY_DEVICE", "auto")
MEMORY_MAX_LENGTH = int(os.getenv("MEMORY_MAX_LENGTH", "256"))
MEMORY_MAX_POS_EXAMPLES = int(os.getenv("MEMORY_MAX_POS_EXAMPLES", str(MEMORY_TOP_K)))
MEMORY_MAX_NEG_EXAMPLES = int(os.getenv("MEMORY_MAX_NEG_EXAMPLES", str(MEMORY_TOP_K)))

# ============ SEMANTIC MODEL (SimCSE) ============
memo_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
memo_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to('cuda')

# ============ PATH SETUP ============
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RETRIEVER_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "retriever"))
AGENTFLY_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if RETRIEVER_DIR not in sys.path:
    sys.path.insert(0, RETRIEVER_DIR)
if AGENTFLY_DIR not in sys.path:
    sys.path.insert(0, AGENTFLY_DIR)

# Import memory utilities
try:
    from memory.np_memory import load_jsonl as mem_load_jsonl, extract_pairs as mem_extract_pairs, retrieve as mem_retrieve
except Exception as _e:
    mem_load_jsonl = mem_extract_pairs = mem_retrieve = None
    logger.warning("Memory retriever not available: %s", _e)


# ============ BM25 RETRIEVER CLASS ============
class BM25Retriever:
    """BM25 retriever for case-based memory retrieval"""
    
    def __init__(self):
        self.index: BM25Okapi | None = None
        self.corpus: List[dict] = []
        self.tokenized_corpus: List[List[str]] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 - simple whitespace tokenizer"""
        return text.lower().split()
    
    def build(self, memory_items: List[dict], key_field: str = "question") -> bool:
        """Build BM25 index from memory items"""
        if not memory_items:
            logger.warning("No memory items to build BM25 index")
            return False
        
        self.corpus = []
        self.tokenized_corpus = []
        
        for item in memory_items:
            text = item.get(key_field, "")
            if text:
                self.corpus.append(item)
                self.tokenized_corpus.append(self._tokenize(text))
        
        if self.tokenized_corpus:
            self.index = BM25Okapi(self.tokenized_corpus)
            logger.info(f"BM25 index built with {len(self.tokenized_corpus)} documents")
            return True
        
        return False
    
    def retrieve(self, query: str, top_k: int = MEMORY_TOP_K) -> List[dict]:
        """Retrieve top-k similar cases using BM25"""
        if not self.index or not self.corpus:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return if score > 0
                case = self.corpus[idx].copy()
                case['_bm25_score'] = scores[idx]
                results.append(case)
        
        return results
    
    def add_single(self, item: dict, key_field: str = "question"):
        """Add a single item to BM25 index (incremental)"""
        text = item.get(key_field, "")
        if not text:
            return
        
        tokens = self._tokenize(text)
        self.corpus.append(item)
        self.tokenized_corpus.append(tokens)
        
        # Rebuild index (simple approach, có thể tối ưu sau)
        if self.tokenized_corpus:
            self.index = BM25Okapi(self.tokenized_corpus)


# ============ UTILITY FUNCTIONS ============
def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()
    m = re.search(r"{[\s\S]*}", text)
    return m.group(0) if m else text


def log_block(title: str, content: Any):
    try:
        if not isinstance(content, str):
            content = json.dumps(content, indent=2, ensure_ascii=False)
    except Exception:
        content = str(content)
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}\n{content}\n")


def _count_tokens(msg: Dict[str, str], enc) -> int:
    role_tokens = 4
    content = msg.get("content") or ""
    return role_tokens + len(enc.encode(content))


def trim_messages(messages: List[Dict[str, str]], max_tokens: int, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    total = sum(_count_tokens(m, enc) for m in messages) + 2
    if total <= max_tokens:
        return messages
    
    system_msg = messages[0]
    kept: List[Dict[str, str]] = [system_msg]
    total = _count_tokens(system_msg, enc) + 2
    
    for msg in reversed(messages[1:]):
        t = _count_tokens(msg, enc)
        if total + t > max_tokens:
            break
        kept.insert(1, msg)
        total += t
    return kept


def build_prompt_from_cases(task_text: str, retrieved_cases: list[dict] | None, original_items: list[dict] | None) -> str:
    """Build prompt from retrieved cases with reward filtering"""
    positive_cases: list[dict] = []
    negative_cases: list[dict] = []
    retrieved_cases = retrieved_cases or []
    original_items = original_items or []
    
    # Tạo mapping nhanh từ question tới reward
    reward_map = {}
    for idx, item in enumerate(original_items):
        q = item.get('question', '')
        reward_map[q] = item.get('reward', 0)
    
    for case in retrieved_cases:
        q = case.get('question', '')
        reward = reward_map.get(q, 0)
        if reward == 1:
            positive_cases.append(case)
        else:
            negative_cases.append(case)
    
    prompt_parts: list[str] = []
    
    if positive_cases:
        prompt_parts.append(
            f"Positive Examples (reward=1) - Showing {min(len(positive_cases), MEMORY_MAX_POS_EXAMPLES)} of {len(positive_cases)}:"
        )
        for i, case in enumerate(positive_cases[:MEMORY_MAX_POS_EXAMPLES], 1):
            try:
                plan_data = json.loads(case['plan'])
                plan_steps = plan_data.get('plan', [])
                plan_text = "\n".join([f"{step['id']}. {step['description']}" for step in plan_steps])
                prompt_parts.append(f"Example {i}:\nQuestion: {case['question']}\nPlan:\n{plan_text}\n")
            except Exception:
                prompt_parts.append(f"Example {i}:\nQuestion: {case.get('question','')}\nPlan: {case.get('plan','')}\n")
    
    if negative_cases:
        prompt_parts.append(
            f"Negative Examples (reward=0) - Showing {min(len(negative_cases), MEMORY_MAX_NEG_EXAMPLES)} of {len(negative_cases)}:"
        )
        for i, case in enumerate(negative_cases[:MEMORY_MAX_NEG_EXAMPLES], 1):
            prompt_parts.append(f"Example {i}:\nQuestion: {case.get('question','')}\nPlan: {case.get('plan','')}\n")
    
    prompt_parts.append(
        "Based on the above examples, please provide a plan for the current task. "
        "Focus on the positive examples and avoid the patterns shown in negative examples.\n\nYour plan:"
    )
    return "\n".join(prompt_parts)


def encode_text(text: str, max_length: int = MEMORY_MAX_LENGTH) -> np.ndarray:
    """Encode text using SimCSE model"""
    inputs = memo_tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = memo_model(**inputs)
    # Use [CLS] token embedding
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# ============ CHAT BACKEND ============
class ChatBackend:
    async def chat(self, *_, **__) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIBackend(ChatBackend):
    def __init__(self, model: str, is_azure: bool):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ) if not is_azure else AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        current_attempt = getattr(self.chat.retry.statistics, 'attempt_number', 0)
        
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        
        try:
            resp = await self.client.chat.completions.create(**payload)
        except Exception as e:
            logger.error(f"API call attempt {current_attempt} failed: {type(e).__name__} - {e}")
            raise
        
        msg = resp.choices[0].message
        raw_calls = getattr(msg, "tool_calls", None)
        tool_calls = None
        if raw_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_calls
            ]
        return {"content": msg.content, "tool_calls": tool_calls}


# ============ DATA CLASSES ============
@dataclass
class MetaCycle:
    cycle: int
    input: List[str]
    output: str


@dataclass
class ExecStep:
    task_id: int
    input: str
    output: str


@dataclass
class ToolCallRecord:
    tool: str
    arguments: Dict[str, Any]
    result: str


@dataclass
class QueryRecord:
    task_id: str
    query: str
    model_output: str
    plan_json: str
    meta_trace: List[MetaCycle]
    executor_trace: List[ExecStep]
    tool_history: List[ToolCallRecord]


MAX_TURNS_MEMORY = 50


# ============ MAIN CLIENT WITH BM25 ============
class HierarchicalClient:
    MAX_CYCLES = 3
    
    def __init__(self, meta_model: str, exec_model: str, is_azure: bool):
        self.meta_llm = OpenAIBackend(meta_model, is_azure)
        self.exec_llm = OpenAIBackend(exec_model, is_azure)
        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}
        self.shared_history: List[Dict[str, str]] = []
        
        # Memory components
        self._memory_items: list[dict] = []
        self._memory_pairs: list[tuple[str, str]] = []
        
        # ===== BM25 RETRIEVER =====
        self.bm25_retriever = BM25Retriever()
        
        # Cache for semantic embeddings (optimization)
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        # Load existing memory
        if mem_load_jsonl and mem_extract_pairs:
            try:
                if os.path.exists(MEMORY_JSONL_PATH):
                    self._memory_items = mem_load_jsonl(MEMORY_JSONL_PATH) or []
                    self._memory_pairs = mem_extract_pairs(self._memory_items, MEMORY_KEY_FIELD, MEMORY_VALUE_FIELD) or []
                    
                    # Build BM25 index from existing memory
                    if self._memory_items:
                        self.bm25_retriever.build(self._memory_items, MEMORY_KEY_FIELD)
                    
                    logger.info(f"Loaded memory JSONL ({len(self._memory_items)} items) from {MEMORY_JSONL_PATH}")
                else:
                    logger.warning(f"MEMORY_JSONL_PATH not found: {MEMORY_JSONL_PATH}")
            except Exception as e:
                logger.warning(f"Failed to load memory: {e}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get cached embedding for text"""
        if text not in self._embedding_cache:
            self._embedding_cache[text] = encode_text(text)
        return self._embedding_cache[text]
    
    def _semantic_retrieve(self, query: str, top_k: int = MEMORY_TOP_K) -> List[dict]:
        """Retrieve cases using semantic similarity (SimCSE)"""
        if not mem_retrieve or not self._memory_pairs:
            return []
        
        try:
            results = mem_retrieve(
                task=query,
                pairs=self._memory_pairs,
                tokenizer=memo_tokenizer,
                model=memo_model,
                device_str=MEMORY_DEVICE,
                top_k=top_k,
                max_length=MEMORY_MAX_LENGTH,
            )
            return results or []
        except Exception as e:
            logger.warning(f"Semantic retrieval failed: {e}")
            return []
    
    def _hybrid_retrieve(self, query: str, top_k: int = MEMORY_TOP_K) -> List[dict]:
        """
        Hybrid retrieval combining BM25 and Semantic search
        Uses BM25_WEIGHT and SEMANTIC_WEIGHT from environment variables
        """
        # Get results from both retrievers
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)  # Get more for merging
        semantic_results = self._semantic_retrieve(query, top_k=top_k * 2)
        
        if not bm25_results and not semantic_results:
            return []
        
        # If only one has results, return that
        if not bm25_results:
            return semantic_results[:top_k]
        if not semantic_results:
            return bm25_results[:top_k]
        
        # === MERGE WITH WEIGHTS ===
        # Create mapping from question to combined score
        combined: Dict[str, dict] = {}
        
        # Add BM25 results with BM25 weight
        for case in bm25_results:
            q = case.get('question', '')
            if q:
                score = case.get('_bm25_score', 0.0) * BM25_WEIGHT
                combined[q] = {
                    'case': case,
                    'score': score,
                    'source': 'bm25'
                }
        
        # Add semantic results with semantic weight
        for case in semantic_results:
            q = case.get('question', '')
            if q:
                # Semantic scores are already normalized in mem_retrieve
                semantic_score = case.get('_similarity', 0.5) * SEMANTIC_WEIGHT
                if q in combined:
                    combined[q]['score'] += semantic_score
                    combined[q]['source'] = 'hybrid'
                else:
                    combined[q] = {
                        'case': case,
                        'score': semantic_score,
                        'source': 'semantic'
                    }
        
        # Sort by combined score and return top_k
        sorted_items = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        results = [item['case'] for item in sorted_items[:top_k]]
        
        logger.debug(f"Hybrid retrieval: BM25={len(bm25_results)}, Semantic={len(semantic_results)}, "
                    f"Merged={len(results)}")
        
        return results
    
    def _memory_prompt_for(self, task_text: str) -> str | None:
        """Retrieve relevant cases using hybrid BM25+Semantic retrieval"""
        if not self._memory_items:
            return None
        
        # Choose retrieval method based on configuration
        if USE_HYBRID:
            retrieved_cases = self._hybrid_retrieve(task_text, top_k=MEMORY_TOP_K)
        else:
            # Fallback to semantic only (original behavior)
            retrieved_cases = self._semantic_retrieve(task_text, top_k=MEMORY_TOP_K)
        
        if not retrieved_cases:
            return None
        
        return build_prompt_from_cases(task_text, retrieved_cases, self._memory_items)
    
    def _add_to_history(self, role: str, content: str):
        self.shared_history.append({"role": role, "content": content})
        if len(self.shared_history) > MAX_TURNS_MEMORY:
            self.shared_history.pop(0)
    
    async def connect_to_servers(self, scripts: List[str]):
        for script in scripts:
            path = Path(script)
            if path.suffix not in {".py", ".js"}:
                raise ValueError("Server script must be .py or .js → " + script)
            cmd = "python" if path.suffix == ".py" else "node"
            params = StdioServerParameters(command=cmd, args=[str(path)])
            stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            for tool in (await session.list_tools()).tools:
                if tool.name in self.sessions:
                    raise RuntimeError(f"Duplicate tool name '{tool.name}'.")
                self.sessions[tool.name] = session
        print("Connected tools:", list(self.sessions.keys()))
    
    async def _tools_schema(self) -> List[Dict[str, Any]]:
        result, cached = [], {}
        for session in self.sessions.values():
            tools_resp = cached.get(id(session)) or await session.list_tools()
            cached[id(session)] = tools_resp
            for tool in tools_resp.tools:
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
        return result
    
    async def process_query(self, query: str, task_id: str) -> QueryRecord:
        self.shared_history = []
        tools_schema = await self._tools_schema()
        
        self._add_to_history("user", query)
        
        # Get memory prompt with BM25+Semantic hybrid retrieval
        mem_prompt = self._memory_prompt_for(query)
        if mem_prompt:
            self._add_to_history("user", mem_prompt)
        
        planner_msgs = [{"role": "system", "content": META_SYSTEM_PROMPT}] + self.shared_history
        
        meta_trace: List[MetaCycle] = []
        executor_trace: List[ExecStep] = []
        tool_history: List[ToolCallRecord] = []
        final_answer: str = ""
        latest_plan_json: str = ""
        
        for cycle in range(self.MAX_CYCLES):
            meta_reply = await self.meta_llm.chat(planner_msgs)
            meta_content = meta_reply["content"] or ""
            meta_trace.append(MetaCycle(cycle, [m["content"] for m in planner_msgs], meta_content))
            self._add_to_history("assistant", meta_content)
            
            if meta_content.startswith("FINAL ANSWER:"):
                final_answer = meta_content[len("FINAL ANSWER:"):].strip()
                break
            
            try:
                stripped = _strip_fences(meta_content)
                _ = json.loads(stripped)["plan"]
                latest_plan_json = stripped
            except Exception as e:
                final_answer = f"[planner error] {e}: {meta_content}"
                break
            
            # Execute tasks
            tasks = json.loads(latest_plan_json)["plan"]
            for task in tasks:
                task_desc = f"Task {task['id']}: {task['description']}"
                exec_msgs = (
                    [{"role": "system", "content": EXEC_SYSTEM_PROMPT}] + self.shared_history + [{"role": "user", "content": task_desc}]
                )
                
                while True:
                    exec_msgs = trim_messages(exec_msgs, MAX_CTX)
                    exec_reply = await self.exec_llm.chat(exec_msgs, tools_schema)
                    if exec_reply["content"]:
                        result_text = str(exec_reply["content"])
                        executor_trace.append(ExecStep(task_id=task["id"], input=task_desc, output=result_text))
                        exec_msgs.append({"role": "assistant", "content": result_text})
                        self._add_to_history("assistant", f"Task {task['id']} result: {result_text}")
                        break
                    
                    for call in exec_reply.get("tool_calls") or []:
                        t_name = call["function"]["name"]
                        t_args = json.loads(call["function"].get("arguments") or "{}")
                        session = self.sessions[t_name]
                        result_msg = await session.call_tool(t_name, t_args)
                        result_text = str(result_msg.content)
                        tool_history.append(ToolCallRecord(tool=t_name, arguments=t_args, result=result_text))
                        exec_msgs.extend(
                            [
                                {"role": "assistant", "content": None, "tool_calls": [call]},
                                {"role": "tool", "tool_call_id": call["id"], "name": t_name, "content": result_text},
                            ]
                        )
            
            planner_msgs = [{"role": "system", "content": META_SYSTEM_PROMPT}] + self.shared_history
        else:
            final_answer = meta_content.strip()
        
        self.shared_history.clear()
        
        return QueryRecord(
            task_id=task_id,
            query=query,
            model_output=final_answer,
            plan_json=latest_plan_json,
            meta_trace=meta_trace,
            executor_trace=executor_trace,
            tool_history=tool_history,
        )
    
    async def cleanup(self):
        await self.exit_stack.aclose()
    
    def update_memory(self, question: str, plan: str, reward: int):
        """Update memory with new case and rebuild BM25 index"""
        new_entry = {
            "question": question,
            "plan": plan,
            "reward": reward
        }
        
        # Add to memory items
        self._memory_items.append(new_entry)
        
        # Update BM25 index incrementally
        self.bm25_retriever.add_single(new_entry, MEMORY_KEY_FIELD)
        
        # Update semantic pairs
        if mem_extract_pairs:
            self._memory_pairs.append((question, plan))


# ============ LLM JUDGE ============
JUDGE_CLIENT = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


def _ensure_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (str, int, float, bool)):
        return [str(x)]
    try:
        return [json.dumps(x, ensure_ascii=False)]
    except Exception:
        return [str(x)]


async def llm_judge(question: str, ground_truth: Any, pred_answer: str) -> Dict[str, Any]:
    gt_list = _ensure_list(ground_truth)
    prompt = PROMPT_TPL.format(
        question=question,
        gt_answer=json.dumps(gt_list, ensure_ascii=False),
        pred_answer=pred_answer,
    )
    try:
        resp = await JUDGE_CLIENT.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        content = resp.choices[0].message.content or ""
        content = _strip_fences(content)
        data = json.loads(content)
        judgement = str(data.get("judgement", "incorrect")).lower().strip()
        if judgement not in ("correct", "incorrect"):
            judgement = "incorrect"
        rationale = str(data.get("rationale", ""))
        return {"judgement": judgement, "rationale": rationale}
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return {"judgement": "incorrect", "rationale": f"judge failed: {e}"}


# ============ MAIN FUNCTION ============
async def main():
    if not query_list:
        print("⚠️  query_list is empty – add questions to process.")
        return
    
    # Load existing finished tasks
    finished_task = []
    result_path = "../result/result_round_0.jsonl"
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                    finished_task.append(record.get('query') or record.get('question'))
                except Exception:
                    continue
    
    print(f"\n{'='*60}")
    print(f"🚀 MEMENTO with BM25+Semantic Hybrid Retrieval")
    print(f"   - BM25 Weight: {BM25_WEIGHT}")
    print(f"   - Semantic Weight: {SEMANTIC_WEIGHT}")
    print(f"   - Hybrid Mode: {USE_HYBRID}")
    print(f"   - Top-K: {MEMORY_TOP_K}")
    print(f"   - Total queries: {len(query_list)}")
    print(f"{'='*60}\n")
    
    client = HierarchicalClient(
        os.getenv("META_MODEL", os.getenv("PLANNER_MODEL", "Qwen/Qwen3-4B-Instruct-2507")),
        os.getenv("EXEC_MODEL", os.getenv("EXECUTOR_MODEL", "Qwen/Qwen3-4B-Instruct-2507")),
        os.getenv("USE_AZURE_OPENAI") == "True",
    )
    
    # Connect to tools (uncomment khi có server)
    # await client.connect_to_servers(server_paths)
    
    try:
        # Statistics
        correct_count = 0
        total_count = 0
        
        for task_id, q in enumerate(tqdm(query_list, total=len(query_list), desc="Processing"), start=0):
            if q in finished_task:
                print(f"Task '{q[:50]}...' already finished, skipping...")
                continue
            
            try:
                print(f"\n{'─'*40}")
                print(f" Task {task_id}: {q[:100]}...")
                
                rec = await client.process_query(q, task_id)
                
                pred_answer = rec.model_output
                gt = ground_truth_map.get(q)
                
                judge_res = await llm_judge(q, gt, pred_answer)
                reward = 1 if judge_res["judgement"] == "correct" else 0
                
                if reward == 1:
                    correct_count += 1
                total_count += 1
                
                # Update memory with BM25 index
                client.update_memory(q, rec.plan_json, reward)
                
                rec_dict = asdict(rec)
                rec_dict.update({
                    "question": q,
                    "plan": rec.plan_json,
                    "ground_truth": gt,
                    "pred_answer": pred_answer,
                    "judgement": judge_res["judgement"],
                    "rationale": judge_res["rationale"],
                    "reward": reward,
                })
                
                print(f"    Judgement: {judge_res['judgement']} (Reward: {reward})")
                print(f"    Running accuracy: {correct_count}/{total_count} = {correct_count/total_count*100:.2f}%")
                print(f"    Memory size: {len(client._memory_items)}")
                
                with open(result_path, "a", encoding="utf-8") as fh:
                    json_line = json.dumps(rec_dict, ensure_ascii=False, default=str)
                    fh.write(json_line + "\n")
                
                # Also save to memory JSONL
                try:
                    mem_path = MEMORY_JSONL_PATH
                    os.makedirs(os.path.dirname(mem_path), exist_ok=True)
                    mem_entry = {
                        "question": q,
                        "plan": rec.plan_json or "",
                        "reward": reward
                    }
                    with open(mem_path, "a", encoding="utf-8") as mf:
                        mf.write(json.dumps(mem_entry, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.warning(f"Failed to write memory file: {e}")
                
            except Exception as e:
                print(f"    [ERROR]: {e}")
                continue
        
        # Final summary
        print(f"\n{'='*60}")
        print(f" BENCHMARK COMPLETED")
        print(f"   Total queries: {total_count}")
        print(f"   Correct: {correct_count}")
        print(f"   Accuracy: {correct_count/total_count*100:.2f}%")
        print(f"   Result saved to: {result_path}")
        print(f"{'='*60}\n")
        
    finally:
        await client.cleanup()
if __name__ == "__main__":
    import torch
    
    asyncio.run(main())
