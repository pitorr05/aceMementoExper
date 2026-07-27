import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: mcp not available. Executor will run without external MCP tools.")

from ..utils.llm import timed_llm_call
from ..utils.formatting import EXEC_SYSTEM_PROMPT


class Executor:
    """
    Executor agent that resolves a single subtask.
    Optionally connects to MCP tool servers and executes function calls.
    """

    def __init__(
        self,
        api_client: Any,
        api_provider: str,
        model: str,
        max_tokens: int = 4096,
        server_scripts: Optional[List[str]] = None
    ):
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
        self.server_scripts = server_scripts or []
        self.sessions: Dict[str, Any] = {}
        self.exit_stack = None

    async def connect_mcp_servers(self) -> None:
        """Connect to stdio MCP tool servers."""
        if not MCP_AVAILABLE or not self.server_scripts:
            return

        from contextlib import AsyncExitStack
        self.exit_stack = AsyncExitStack()
        
        for script in self.server_scripts:
            path = Path(script)
            if not path.exists():
                print(f"[Executor] Server script not found, skipping: {script}")
                continue
            
            cmd = "python" if path.suffix == ".py" else "node"
            params = StdioServerParameters(command=cmd, args=[str(path)])
            try:
                stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
                session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                await session.initialize()
                
                tools_list = await session.list_tools()
                for tool in tools_list.tools:
                    if tool.name in self.sessions:
                        print(f"Warning: Duplicate tool name '{tool.name}' from {script}")
                    self.sessions[tool.name] = session
                print(f"[Executor] Connected to MCP tools from {script}")
            except Exception as e:
                print(f"[Executor] Error connecting to {script}: {e}")

    async def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get OpenAI tool schemas for connected MCP tools."""
        if not self.sessions:
            return []

        result = []
        cached = {}
        for name, session in self.sessions.items():
            if id(session) in cached:
                continue
            try:
                tools_resp = await session.list_tools()
                cached[id(session)] = tools_resp
                for tool in tools_resp.tools:
                    result.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        }
                    })
            except Exception as e:
                print(f"Error fetching schema for session: {e}")
        return result

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call connected MCP tool and return result string."""
        if tool_name not in self.sessions:
            return f"Error: Tool '{tool_name}' not available."

        try:
            session = self.sessions[tool_name]
            result_msg = await session.call_tool(tool_name, arguments)
            return str(result_msg.content)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"

    async def execute_task(
        self,
        task_desc: str,
        history: List[Dict[str, str]],
        call_id: str = "exec",
        log_dir: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Execute subtask step-by-step.
        Runs tool loop until the LLM returns text output.
        """
        tools_schema = await self.get_tools_schema()
        
        # Build prompt messages array
        # First the EXEC system prompt, then the plan history, then current subtask
        messages = [{"role": "system", "content": EXEC_SYSTEM_PROMPT}] + history + [{"role": "user", "content": task_desc}]
        
        executor_steps = []
        tool_calls_made = []

        # We construct prompt as string for timed_llm_call
        step_count = 1
        while True:
            # Format history messages into a single prompt for timed_llm_call
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content")
                if content:
                    prompt_parts.append(f"[{role.upper()}]: {content}")
                elif msg.get("tool_calls"):
                    # Append tool call representations
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        prompt_parts.append(f"[ASSISTANT CALLS TOOL]: {func.get('name')}({func.get('arguments')})")
            
            prompt = "\n".join(prompt_parts)

            # Check if there are tool calls to make. Wait, timed_llm_call in ace uses client.chat.completions.create
            # which does standard tool calling. timed_llm_call does not support sending tools directly,
            # so we'll call the client completions API directly here, keeping the exact logging behavior.
            try:
                if self.api_provider == "sglang":
                    # sglang doesn't support tools easily; return text
                    response_text, call_info = await asyncio.to_thread(
                        timed_llm_call,
                        self.api_client,
                        self.api_provider,
                        self.model,
                        prompt,
                        role="executor",
                        call_id=f"{call_id}_step_{step_count}",
                        max_tokens=self.max_tokens,
                        log_dir=log_dir
                    )
                    executor_steps.append({"input": task_desc, "output": response_text})
                    return response_text, executor_steps, tool_calls_made

                # Prepare API call
                max_tokens_key = "max_completion_tokens" if self.api_provider == "openai" else "max_tokens"
                api_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.0,
                    max_tokens_key: self.max_tokens
                }
                if tools_schema:
                    api_params["tools"] = tools_schema
                    api_params["tool_choice"] = "auto"

                # Run chat completion
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.api_client.chat.completions.create(**api_params)
                )

                choice = response.choices[0]
                content = choice.message.content
                tool_calls = getattr(choice.message, "tool_calls", None)

                if content:
                    # Completed subtask
                    executor_steps.append({"input": task_desc, "output": content})
                    return content, executor_steps, tool_calls_made

                if tool_calls:
                    # Execute tool calls
                    assistant_msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                }
                            }
                            for tc in tool_calls
                        ]
                    }
                    messages.append(assistant_msg)

                    for tc in tool_calls:
                        t_name = tc.function.name
                        t_args = json.loads(tc.function.arguments or "{}")
                        print(f"[Executor] Tool Call: {t_name}({t_args})")
                        
                        tool_res = await self.call_tool(t_name, t_args)
                        print(f"[Executor] Tool Result: {tool_res[:200]}...")

                        tool_calls_made.append({
                            "tool": t_name,
                            "arguments": t_args,
                            "result": tool_res
                        })

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": t_name,
                            "content": tool_res
                        })
                else:
                    # Fallback text
                    return "No output generated", executor_steps, tool_calls_made

            except Exception as e:
                print(f"[Executor] LLM Error: {e}")
                return f"Execution Error: {e}", executor_steps, tool_calls_made

            step_count += 1

    async def cleanup(self) -> None:
        if self.exit_stack:
            await self.exit_stack.aclose()
            print("[Executor] Cleaned up MCP tool sessions.")
