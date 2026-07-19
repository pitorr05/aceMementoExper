import re
import os
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from .runner import ACEMementoRunner


# Dataclass definitions matching Memento's trace logs
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
    retrieved_cases: Optional[List[Dict[str, Any]]] = None


class ACE(ACEMementoRunner):
    """
    Drop-in replacement for ACE class in original ACE benchmark.
    """

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        reflector_model: str,
        curator_model: str,
        max_tokens: int = 4096,
        initial_playbook: Optional[str] = None,
        use_bulletpoint_analyzer: bool = False,
        bulletpoint_analyzer_threshold: float = 0.90,
        use_rae: bool = False,
        rae_top_k: int = 10,
        use_failure_memory: bool = False,
        failure_memory_top_k: int = 3,
        use_adversarial: bool = False,
        adversarial_frequency: int = 10,
        adversarial_model: Optional[str] = None,
        memory_jsonl_path: str = "./results/case_bank.jsonl",
        case_bank_top_k: int = 4,
        device: str = "cpu",
        parametric_model_name: str = "princeton-nlp/sup-simcse-roberta-base",
        retriever_model_path: Optional[str] = None,
        use_arw: bool = False,
    ):
        super().__init__(
            api_provider=api_provider,
            generator_model=generator_model,
            reflector_model=reflector_model,
            curator_model=curator_model,
            memory_jsonl_path=memory_jsonl_path,
            max_tokens=max_tokens,
            initial_playbook=initial_playbook,
            use_rae=use_rae,
            rae_top_k=rae_top_k,
            case_bank_top_k=case_bank_top_k,
            use_failure_memory=use_failure_memory,
            failure_memory_top_k=failure_memory_top_k,
            use_adversarial=use_adversarial,
            adversarial_frequency=adversarial_frequency,
            adversarial_model=adversarial_model,
            device=device,
            parametric_model_name=parametric_model_name,
            retriever_model_path=retriever_model_path,
	    use_arw=use_arw,
        )


class HierarchicalClient:
    """
    Drop-in replacement for HierarchicalClient in original Memento benchmarks.
    """

    def __init__(
        self,
        meta_model: str,
        exec_model: str,
        is_azure: bool = False,
        api_provider: str = "openai",
        memory_jsonl_path: str = "./results/case_bank.jsonl",
        playbook_path: Optional[str] = None,
        use_rae: bool = True,
        rae_top_k: int = 10,
        case_bank_top_k: int = 4,
        device: str = "cpu",
        parametric_model_name: str = "princeton-nlp/sup-simcse-roberta-base",
        retriever_model_path: Optional[str] = None,
    ):
        # We reuse ACEMementoRunner internally
        # Load initial playbook if path is provided
        initial_playbook = None
        if playbook_path and os.path.exists(playbook_path):
            with open(playbook_path, "r", encoding="utf-8") as f:
                initial_playbook = f.read()

        self.runner = ACEMementoRunner(
            api_provider=api_provider,
            generator_model=meta_model,
            reflector_model=meta_model,
            curator_model=meta_model,
            memory_jsonl_path=memory_jsonl_path,
            initial_playbook=initial_playbook,
            use_rae=use_rae,
            rae_top_k=rae_top_k,
            case_bank_top_k=case_bank_top_k,
            device=device,
            parametric_model_name=parametric_model_name,
            retriever_model_path=retriever_model_path,
        )
        self._memory_items = []
        self._memory_pairs = []

    def _load_memory(self) -> None:
        """Reload cases from database."""
        self.runner.case_bank.load_cases()

    async def connect_to_servers(self, scripts: List[str]) -> None:
        """Connect to stdio MCP server scripts."""
        self.runner.executor.server_scripts = scripts
        await self.runner.executor.connect_mcp_servers()

    async def process_query(
        self,
        query: str,
        file_path_or_task_id: str,
        task_id: Optional[str] = None
    ) -> QueryRecord:
        """
        Process the query conforming to both:
          - process_query(self, query, task_id)
          - process_query(self, query, file_path, task_id)
        """
        # Determine arguments
        if task_id is None:
            actual_task_id = file_path_or_task_id
            context = ""
        else:
            actual_task_id = task_id
            context = f"File: {file_path_or_task_id}"

        # Retrieve dual memory contexts
        retrieved_cases = self.runner.case_bank.retrieve_cases(query)
        cases_text = self.runner.case_bank.format_cases_for_prompt(retrieved_cases)
        
        playbook = self.runner.playbook_manager.retrieve_bullets(query, self.runner.rae_top_k) if self.runner.use_rae else self.runner.playbook_manager.playbook

        # Generate plan & execution steps
        final_answer, _, trajectory = await self.runner.generator.generate(
            question=query,
            playbook=playbook,
            cases_text=cases_text,
            context=context,
            call_id=f"memento_{actual_task_id}"
        )

        # Convert traces into expected formats
        meta_trace = [
            MetaCycle(cycle=m["cycle"], input=[m["prompt"]], output=m["response"])
            for m in trajectory.get("meta_trace", [])
        ]
        
        executor_trace = [
            ExecStep(task_id=es["task_id"] if "task_id" in es else 1, input=es["input"], output=es["output"])
            for es in trajectory.get("executor_trace", [])
        ]

        tool_history = [
            ToolCallRecord(tool=th["tool"], arguments=th["arguments"], result=th["result"])
            for th in trajectory.get("tool_history", [])
        ]

        return QueryRecord(
            task_id=actual_task_id,
            query=query,
            model_output=final_answer,
            plan_json=trajectory.get("plan_json", ""),
            meta_trace=meta_trace,
            executor_trace=executor_trace,
            tool_history=tool_history,
            retrieved_cases=retrieved_cases
        )

    async def cleanup(self) -> None:
        """Cleanup mcp sessions."""
        await self.runner.executor.cleanup()
