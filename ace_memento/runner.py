import re
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from .core.case_bank import CaseBank
from .core.playbook import PlaybookManager
from .core.planner import Planner
from .core.executor import Executor
from .core.generator import Generator
from .core.reflector import Reflector
from .core.curator import Curator
from .core.failure_memory import FailureMemoryBank
from .core.adversarial_agent import AdversarialAgent
from .utils.llm import initialize_clients


def find_latest_playbook(run_path: str) -> Tuple[Optional[str], int, int]:
    """
    Scans run_path for saved playbook files and finds the one corresponding to the largest step.
    Returns (playbook_filepath, epoch, step).
    If no playbook file is found, returns (None, 1, 0).
    """
    max_epoch = 1
    max_step = 0
    latest_file = None

    if not os.path.exists(run_path):
        return None, 1, 0

    for fname in os.listdir(run_path):
        # Match epoch_{epoch}_step_{step}_playbook.txt
        m = re.match(r"^epoch_(\d+)_step_(\d+)_playbook\.txt$", fname)
        if m:
            epoch = int(m.group(1))
            step = int(m.group(2))
            if (epoch > max_epoch) or (epoch == max_epoch and step > max_step):
                max_epoch = epoch
                max_step = step
                latest_file = os.path.join(run_path, fname)
            continue

        # Match step_{step}_playbook.txt (online mode)
        m = re.match(r"^step_(\d+)_playbook\.txt$", fname)
        if m:
            step = int(m.group(1))
            if step > max_step:
                max_epoch = 1
                max_step = step
                latest_file = os.path.join(run_path, fname)
            continue

        # Match playbook_epoch_{epoch}.txt
        m = re.match(r"^playbook_epoch_(\d+)\.txt$", fname)
        if m:
            epoch = int(m.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                max_step = 0
                latest_file = os.path.join(run_path, fname)
            continue

    return latest_file, max_epoch, max_step



class ACEMementoRunner:
    """
    Main ACE-Memento Orchestrator.
    Manages the continual learning loop coordinating Planner, Executor, Reflector, Curator,
    evolving both Episodic memory (Case Bank) and Semantic memory (Playbook) simultaneously.
    """

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        reflector_model: str,
        curator_model: str,
        memory_jsonl_path: str = "./results/case_bank.jsonl",
        max_tokens: int = 4096,
        initial_playbook: Optional[str] = None,
        use_rae: bool = False,
        rae_top_k: int = 10,
        case_bank_top_k: int = 4,
        use_failure_memory: bool = False,
        failure_memory_top_k: int = 10,
        use_adversarial: bool = False,
        adversarial_frequency: int = 10,
        adversarial_model: Optional[str] = None,
        server_scripts: Optional[List[str]] = None,
        device: str = "cuda",
        parametric_model_name: str = "princeton-nlp/sup-simcse-roberta-base",
        retriever_model_path: Optional[str] = None
    ):
        self.api_provider = api_provider
        self.generator_model = generator_model
        self.reflector_model = reflector_model
        self.curator_model = curator_model
        self.max_tokens = max_tokens
        self.use_rae = use_rae
        self.rae_top_k = rae_top_k
        self.case_bank_top_k = case_bank_top_k
        self.parametric_model_name = parametric_model_name
        self.retriever_model_path = retriever_model_path

        # Initialize clients
        generator_client, reflector_client, curator_client = initialize_clients(api_provider)

        # 1. Playbook Manager (Semantic Memory)
        self.playbook_manager = PlaybookManager(
            initial_playbook=initial_playbook,
            device=device
        )

        # 2. Case Bank (Episodic Memory)
        self.case_bank = CaseBank(
            memory_jsonl_path=memory_jsonl_path,
            top_k=case_bank_top_k,
            parametric_model_name=parametric_model_name,
            retriever_model_path=retriever_model_path,
            device=device
        )

        # 3. Core agents
        self.planner = Planner(generator_client, api_provider, generator_model, max_tokens)
        self.executor = Executor(generator_client, api_provider, generator_model, max_tokens, server_scripts)
        self.generator = Generator(self.planner, self.executor)
        
        self.reflector = Reflector(reflector_client, api_provider, reflector_model, max_tokens)
        self.curator = Curator(curator_client, api_provider, curator_model, max_tokens)

        # Initialize FailureMemoryBank (Analogical Reflection)
        self.use_failure_memory = use_failure_memory
        self.failure_memory_top_k = failure_memory_top_k
        if use_failure_memory:
            shared_encoder = self.playbook_manager.encode
            self.failure_memory = FailureMemoryBank(
                encoder=shared_encoder,
                top_k=failure_memory_top_k,
            )
            print(f"✓ FailureMemoryBank initialized (top_k={failure_memory_top_k})")
        else:
            self.failure_memory = None

        self.use_adversarial = use_adversarial
        self.adversarial_frequency = adversarial_frequency
        adversarial_model_name = adversarial_model or generator_model
        self.adversarial_agent = AdversarialAgent(
            generator_client, api_provider, adversarial_model_name, max_tokens
        )

        self.next_global_id = 1
        self._recompute_next_global_id()

    def _recompute_next_global_id(self) -> None:
        """Find the next ID to assign to playbook bullets."""
        max_id = 0
        for b in self.playbook_manager.bullets:
            id_match = re.search(r'-(\d+)$', b['id'])
            if id_match:
                num = int(id_match.group(1))
                max_id = max(max_id, num)
        self.next_global_id = max_id + 1

    def run(
        self,
        mode: str,
        train_samples: Optional[List[Dict[str, Any]]] = None,
        val_samples: Optional[List[Dict[str, Any]]] = None,
        test_samples: Optional[List[Dict[str, Any]]] = None,
        data_processor: Any = None,
        config: Optional[Dict[str, Any]] = None,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper around run_async to execute the runner loop.
        """
        return asyncio.run(self.run_async(mode, train_samples, val_samples, test_samples, data_processor, config, resume_from))

    async def run_async(
        self,
        mode: str,
        train_samples: Optional[List[Dict[str, Any]]] = None,
        val_samples: Optional[List[Dict[str, Any]]] = None,
        test_samples: Optional[List[Dict[str, Any]]] = None,
        data_processor: Any = None,
        config: Optional[Dict[str, Any]] = None,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main run entry point for offline/online training or evaluation.
        """
        config = config or {}
        resume_epoch = 1
        resume_step = 0

        if resume_from:
            run_path = resume_from
            run_config_path = os.path.join(run_path, "run_config.json")
            if os.path.exists(run_config_path):
                try:
                    with open(run_config_path, "r", encoding="utf-8") as f:
                        loaded_config = json.load(f)
                    # Merge loaded config with passed config (passed config overrides)
                    config = {**loaded_config, **config}
                    print(f"Loaded config from {run_config_path}")
                except Exception as e:
                    print(f"Warning: Failed to load config from {run_config_path}: {e}")

            latest_playbook_path, resume_epoch, resume_step = find_latest_playbook(run_path)
            if latest_playbook_path:
                print(f"Resuming from epoch {resume_epoch}, step {resume_step} using playbook {latest_playbook_path}")
                try:
                    with open(latest_playbook_path, "r", encoding="utf-8") as f:
                        resume_playbook = f.read()
                    self.playbook_manager.update_playbook(resume_playbook)
                    self._recompute_next_global_id()
                except Exception as e:
                    print(f"Error reading playbook {latest_playbook_path}: {e}")
            else:
                print("No intermediate playbook found in resume folder. Starting from epoch 1, step 0.")
                resume_epoch = 1
                resume_step = 0
        else:
            save_dir = config.get("save_dir", "./results")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_folder = f"ace_memento_{timestamp}_{mode}"
            run_path = os.path.join(save_dir, run_folder)
            os.makedirs(run_path, exist_ok=True)

            # Save configuration for future resumes
            run_config = {
                "api_provider": self.api_provider,
                "generator_model": self.generator_model,
                "reflector_model": self.reflector_model,
                "curator_model": self.curator_model,
                "max_tokens": self.max_tokens,
                "use_rae": self.use_rae,
                "rae_top_k": self.rae_top_k,
                "case_bank_top_k": self.case_bank_top_k,
                "use_failure_memory": self.use_failure_memory,
                "failure_memory_top_k": self.failure_memory_top_k,
                "use_adversarial": self.use_adversarial,
                "adversarial_frequency": self.adversarial_frequency,
                "parametric_model_name": self.parametric_model_name,
                "retriever_model_path": self.retriever_model_path,
                **config
            }
            run_config_path = os.path.join(run_path, "run_config.json")
            try:
                with open(run_config_path, "w", encoding="utf-8") as f:
                    json.dump(run_config, f, indent=2)
            except Exception as e:
                print(f"Warning: Failed to save config to {run_config_path}: {e}")

        log_dir = os.path.join(run_path, "logs")
        os.makedirs(log_dir, exist_ok=True)

        num_epochs = config.get("num_epochs", 1)
        max_num_rounds = config.get("max_num_rounds", 3)
        token_budget = config.get("playbook_token_budget", 80000)

        # Isolate CaseBank to the run directory
        run_case_bank_path = os.path.join(run_path, "case_bank.jsonl")
        if os.path.exists(self.case_bank.memory_jsonl_path) and not os.path.exists(run_case_bank_path):
            try:
                import shutil
                os.makedirs(os.path.dirname(run_case_bank_path), exist_ok=True)
                shutil.copy(self.case_bank.memory_jsonl_path, run_case_bank_path)
                print(f"Copied base case bank to {run_case_bank_path}")
            except Exception as e:
                print(f"Warning: Failed to copy case bank to run dir: {e}")
        
        self.case_bank.memory_jsonl_path = run_case_bank_path

        # Truncate case bank to remove duplicate steps if resuming
        if resume_from and os.path.exists(self.case_bank.memory_jsonl_path):
            if mode == "offline":
                total_steps_completed = (resume_epoch - 1) * len(train_samples or []) + resume_step
            else:
                total_steps_completed = resume_step

            try:
                with open(self.case_bank.memory_jsonl_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                if len(lines) > total_steps_completed:
                    print(f"Truncating case bank to {total_steps_completed} cases to prevent duplicate records upon resuming.")
                    with open(self.case_bank.memory_jsonl_path, "w", encoding="utf-8") as f:
                        f.writelines(lines[:total_steps_completed])
            except Exception as e:
                print(f"Warning: Failed to truncate case bank: {e}")

        # Reload cases from the run-isolated file
        self.case_bank.load_cases()

        # Connect executor stdio servers
        await self.executor.connect_mcp_servers()

        results = {}
        best_accuracy = 0.0
        best_playbook = self.playbook_manager.playbook

        if mode == "offline":
            initial_test_accuracy = 0.0
            # Skip initial test on resume to save time and API costs
            if test_samples and not resume_from:
                initial_test_res = await self._run_test(
                    test_samples=test_samples,
                    data_processor=data_processor,
                    playbook=self.playbook_manager.playbook,
                    config=config,
                    log_dir=log_dir,
                    prefix="initial"
                )
                initial_test_accuracy = initial_test_res["accuracy"]
                results["initial_test_results"] = initial_test_res

            print(f"--- Starting Offline Training Epochs={num_epochs} ---")
            train_results = []
            
            for epoch in range(1, num_epochs + 1):
                if epoch < resume_epoch:
                    print(f"Skipping Epoch {epoch} (already completed)")
                    continue
                print(f"--- Epoch {epoch}/{num_epochs} ---")
                
                for step, sample in enumerate(train_samples or [], 1):
                    if epoch == resume_epoch and step <= resume_step:
                        print(f"Skipping Step {step} (already completed)")
                        continue
                    print(f"\n--- Train Step {step}/{len(train_samples)} ---")
                    
                    # 1. Retrieve dual memory contexts
                    query = sample.get("question", "")
                    context = sample.get("context", "")
                    target = sample.get("target", "")

                    retrieved_cases = self.case_bank.retrieve_cases(query)
                    cases_text = self.case_bank.format_cases_for_prompt(retrieved_cases)
                    
                    playbook = self.playbook_manager.retrieve_bullets(query, self.rae_top_k) if self.use_rae else self.playbook_manager.playbook

                    # 2. Run generator (Planner + Executor)
                    final_answer, bullet_ids_used, trajectory = await self.generator.generate(
                        question=query,
                        playbook=playbook,
                        cases_text=cases_text,
                        context=context,
                        use_json_mode=config.get("json_mode", False),
                        call_id=f"train_e{epoch}_s{step}",
                        log_dir=log_dir
                    )
                    initial_answer = final_answer

                    # 3. Evaluate accuracy (reward)
                    is_correct = data_processor.answer_is_correct(final_answer, target)
                    reward = 1 if is_correct else 0
                    print(f"Predicted answer: {final_answer} | Target: {target} | Correct: {is_correct}")

                    # 4. Write case to episodic memory (Memento CASE WRITE)
                    self.case_bank.add_case(query, trajectory["plan_json"], reward)

                    # 5. Reflect and Curate (ACE context engineering)
                    trajectory_str = json.dumps(trajectory, indent=2)
                    bullets_used_str = "\n".join([b["original_line"] for b in self.playbook_manager.bullets if b["id"] in bullet_ids_used])
                    
                    if is_correct:
                        # Correct: reinforce helpful bullets
                        _, bullet_tags, _ = self.reflector.reflect(
                            question=query,
                            trajectory_str=trajectory_str,
                            predicted_answer=final_answer,
                            ground_truth=target,
                            environment_feedback="Predicted answer matches ground truth",
                            bullets_used_str=bullets_used_str,
                            use_ground_truth=True,
                            call_id=f"reflect_s{step}",
                            log_dir=log_dir
                        )
                        # Apply updates to bullet counts
                        updated_playbook = self.curator.update_bullet_counts(self.playbook_manager.playbook, bullet_tags)
                        self.playbook_manager.update_playbook(updated_playbook)
                    else:
                        # Incorrect: run reflection rounds
                        reflection = ""
                        for r in range(max_num_rounds):
                            # Retrieve negative cases for analogical context
                            if self.use_failure_memory and self.failure_memory:
                                similar_failures = self.failure_memory.retrieve(query)
                                analogical_context = self.failure_memory.format_for_prompt(similar_failures)
                            else:
                                neg_cases = [c for c in retrieved_cases if c.get("reward") == 0]
                                analogical_context = self.case_bank.format_cases_for_prompt(neg_cases)
                            
                            reflection, bullet_tags, _ = self.reflector.reflect(
                                question=query,
                                trajectory_str=trajectory_str,
                                predicted_answer=final_answer,
                                ground_truth=target,
                                environment_feedback="Predicted answer does not match ground truth",
                                bullets_used_str=bullets_used_str,
                                analogical_context=analogical_context,
                                use_ground_truth=True,
                                call_id=f"reflect_s{step}_r{r}",
                                log_dir=log_dir
                            )

                            # Tag and reinforce counts
                            updated_playbook = self.curator.update_bullet_counts(self.playbook_manager.playbook, bullet_tags)
                            self.playbook_manager.update_playbook(updated_playbook)

                            # Try to regenerate
                            final_answer, bullet_ids_used, trajectory = await self.generator.generate(
                                question=query,
                                playbook=self.playbook_manager.retrieve_bullets(query, self.rae_top_k) if self.use_rae else self.playbook_manager.playbook,
                                cases_text=cases_text,
                                context=context,
                                use_json_mode=config.get("json_mode", False),
                                call_id=f"train_e{epoch}_s{step}_r{r}",
                                log_dir=log_dir
                            )
                            if data_processor.answer_is_correct(final_answer, target):
                                print(f"Corrected reasoning on round {r}!")
                                break

                        # Store distilled insights from the last reflection into memory
                        if self.use_failure_memory and self.failure_memory and reflection not in ("(empty)", ""):
                            try:
                                parsed = json.loads(reflection) if isinstance(reflection, str) else {}
                            except (json.JSONDecodeError, TypeError):
                                parsed = {}
                            self.failure_memory.add(
                                question=query,
                                predicted_answer=initial_answer,
                                ground_truth=target,
                                error_identification=parsed.get("error_identification", ""),
                                root_cause=parsed.get("root_cause_analysis", ""),
                                key_insight=parsed.get("key_insight", ""),
                            )

                        # Curate: evolve semantic playbook rules
                        stats = self.curator.get_playbook_stats(self.playbook_manager.playbook)
                        updated_playbook, self.next_global_id, operations, _ = self.curator.curate(
                            current_playbook=self.playbook_manager.playbook,
                            recent_reflection=reflection,
                            question_context=context,
                            current_step=step,
                            total_samples=len(train_samples),
                            token_budget=token_budget,
                            playbook_stats=stats,
                            call_id=f"curate_s{step}",
                            log_dir=log_dir,
                            next_global_id=self.next_global_id
                        )
                        self.playbook_manager.update_playbook(updated_playbook)

                    # Run adversarial episode
                    if self.use_adversarial and self.adversarial_agent:
                        await self._run_adversarial_episode(
                            step_id=f"train_e{epoch}_s{step}",
                            epoch=epoch,
                            step=step,
                            log_dir=log_dir,
                            config=config,
                            total_samples=len(train_samples or []),
                            base_question=query,
                            base_context=context,
                            base_target=target,
                            data_processor=data_processor
                        )

                    # Save intermediate playbook at each step
                    save_steps = config.get("save_steps", 50)
                    if step % save_steps == 0:
                        step_playbook_path = os.path.join(run_path, f"epoch_{epoch}_step_{step}_playbook.txt")
                        with open(step_playbook_path, "w", encoding="utf-8") as f:
                            f.write(self.playbook_manager.playbook)

                    # Periodic validation evaluation
                    eval_steps = config.get("eval_steps", 50)
                    if step % eval_steps == 0 and val_samples:
                        val_accuracy = await self._evaluate_validation_set(
                            val_samples=val_samples,
                            data_processor=data_processor,
                            config=config,
                            log_dir=log_dir
                        )
                        # Track best playbook
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            best_playbook = self.playbook_manager.playbook
                            print(f"🎉 New best validation accuracy: {best_accuracy:.4f}")
                            best_playbook_path = os.path.join(run_path, "best_playbook.txt")
                            with open(best_playbook_path, "w", encoding="utf-8") as f:
                                f.write(best_playbook)

                        # Save validation results to a tracking file
                        val_results_path = os.path.join(run_path, "val_results.json")
                        val_history = []
                        if os.path.exists(val_results_path):
                            try:
                                with open(val_results_path, "r") as f:
                                    val_history = json.load(f)
                            except Exception:
                                pass
                        val_history.append({"epoch": epoch, "step": step, "validation_accuracy": val_accuracy})
                        with open(val_results_path, "w") as f:
                            json.dump(val_history, f, indent=2)

                # Save intermediate playbooks
                epoch_playbook_path = os.path.join(run_path, f"playbook_epoch_{epoch}.txt")
                with open(epoch_playbook_path, "w", encoding="utf-8") as f:
                    f.write(self.playbook_manager.playbook)

            # Save final playbook and case bank
            final_playbook_path = os.path.join(run_path, "final_playbook.txt")
            with open(final_playbook_path, "w", encoding="utf-8") as f:
                f.write(self.playbook_manager.playbook)

            # Save best playbook
            best_playbook_path = os.path.join(run_path, "best_playbook.txt")
            with open(best_playbook_path, "w", encoding="utf-8") as f:
                f.write(best_playbook)

            # Final Test
            final_test_accuracy = 0.0
            if test_samples:
                final_test_res = await self._run_test(
                    test_samples=test_samples,
                    data_processor=data_processor,
                    playbook=best_playbook,
                    config=config,
                    log_dir=log_dir,
                    prefix="final"
                )
                final_test_accuracy = final_test_res["accuracy"]
                results["final_test_results"] = final_test_res
            
            results["training"] = "completed"
            results["best_validation_accuracy"] = best_accuracy
            
            # Print final summary banner matching standard ACE format
            print(f"\n{'='*60}")
            print(f"RUN COMPLETE")
            print(f"{'='*60}")
            print(f"Mode: {mode.upper()}")
            print(f"Best Validation Accuracy: {best_accuracy:.3f}")
            if test_samples:
                print(f"Initial Test Accuracy: {initial_test_accuracy:.3f}")
                print(f"Final Test Accuracy: {final_test_accuracy:.3f}")
            print(f"Results saved to: {run_path}")
            print(f"{'='*60}\n")

        elif mode == "online":
            print(f"--- Starting Online Training on {len(test_samples or [])} samples ---")
            initial_test_accuracy = 0.0
            # Skip initial test on resume to save time and API costs
            if test_samples and not resume_from:
                initial_test_res = await self._run_test(
                    test_samples=test_samples,
                    data_processor=data_processor,
                    playbook=self.playbook_manager.playbook,
                    config=config,
                    log_dir=log_dir,
                    prefix="initial"
                )
                initial_test_accuracy = initial_test_res["accuracy"]
                results["initial_test_results"] = initial_test_res

            online_eval_freq = config.get("online_eval_frequency", 15)
            num_windows = (len(test_samples) + online_eval_freq - 1) // online_eval_freq
            correct_count = 0
            total_count = 0
            
            if resume_step > 0:
                # Reconstruct correct_count and total_count from CaseBank
                completed_cases = self.case_bank.cases[:resume_step]
                correct_count = sum(1 for c in completed_cases if c.get("reward") == 1)
                total_count = len(completed_cases)
                print(f"Reconstructed online test progress from case bank: {correct_count}/{total_count} correct so far.")

            for win_idx in range(num_windows):
                start_idx = win_idx * online_eval_freq
                end_idx = min(start_idx + online_eval_freq, len(test_samples))
                win_samples = test_samples[start_idx:end_idx]
                
                # Filter unseen samples in this window for testing to prevent double-counting
                win_samples_test = []
                for local_idx, sample in enumerate(win_samples, 1):
                    global_step = start_idx + local_idx
                    if global_step > resume_step:
                        win_samples_test.append(sample)
                
                # If the entire window is already trained, skip it entirely
                if not win_samples_test:
                    continue

                print(f"\n--- Window {win_idx + 1}/{num_windows} (Samples {start_idx} to {end_idx - 1}, remaining test: {len(win_samples_test)}) ---")
                
                # Test on window (before training on it)
                win_test_res = await self._run_test(
                    test_samples=win_samples_test,
                    data_processor=data_processor,
                    playbook=self.playbook_manager.playbook,
                    config=config,
                    log_dir=log_dir,
                    prefix=f"online_win_{win_idx + 1}"
                )
                correct_count += win_test_res["correct"]
                total_count += win_test_res["total"]
                
                # Train on window
                for local_step, sample in enumerate(win_samples, 1):
                    global_step = start_idx + local_step
                    if global_step <= resume_step:
                        continue
                    print(f"\n--- Window {win_idx + 1}, Step {local_step}/{len(win_samples)} (Global step {global_step}) ---")
                    
                    query = sample.get("question", "")
                    context = sample.get("context", "")
                    target = sample.get("target", "")

                    retrieved_cases = self.case_bank.retrieve_cases(query)
                    cases_text = self.case_bank.format_cases_for_prompt(retrieved_cases)
                    playbook = self.playbook_manager.retrieve_bullets(query, self.rae_top_k) if self.use_rae else self.playbook_manager.playbook

                    final_answer, bullet_ids_used, trajectory = await self.generator.generate(
                        question=query,
                        playbook=playbook,
                        cases_text=cases_text,
                        context=context,
                        use_json_mode=config.get("json_mode", False),
                        call_id=f"online_s{global_step}",
                        log_dir=log_dir
                    )
                    initial_answer = final_answer

                    is_correct = data_processor.answer_is_correct(final_answer, target)
                    reward = 1 if is_correct else 0
                    print(f"Predicted: {final_answer} | Target: {target} | Correct: {is_correct}")

                    self.case_bank.add_case(query, trajectory["plan_json"], reward)

                    trajectory_str = json.dumps(trajectory, indent=2)
                    bullets_used_str = "\n".join([b["original_line"] for b in self.playbook_manager.bullets if b["id"] in bullet_ids_used])
                    
                    if is_correct:
                        _, bullet_tags, _ = self.reflector.reflect(
                            question=query,
                            trajectory_str=trajectory_str,
                            predicted_answer=final_answer,
                            ground_truth=target,
                            environment_feedback="Predicted answer matches ground truth",
                            bullets_used_str=bullets_used_str,
                            use_ground_truth=True,
                            call_id=f"online_reflect_s{global_step}",
                            log_dir=log_dir
                        )
                        updated_playbook = self.curator.update_bullet_counts(self.playbook_manager.playbook, bullet_tags)
                        self.playbook_manager.update_playbook(updated_playbook)
                    else:
                        reflection = ""
                        for r in range(max_num_rounds):
                            if self.use_failure_memory and self.failure_memory:
                                similar_failures = self.failure_memory.retrieve(query)
                                analogical_context = self.failure_memory.format_for_prompt(similar_failures)
                            else:
                                neg_cases = [c for c in retrieved_cases if c.get("reward") == 0]
                                analogical_context = self.case_bank.format_cases_for_prompt(neg_cases)
                            
                            reflection, bullet_tags, _ = self.reflector.reflect(
                                question=query,
                                trajectory_str=trajectory_str,
                                predicted_answer=final_answer,
                                ground_truth=target,
                                environment_feedback="Predicted answer does not match ground truth",
                                bullets_used_str=bullets_used_str,
                                analogical_context=analogical_context,
                                use_ground_truth=True,
                                call_id=f"online_reflect_s{global_step}_r{r}",
                                log_dir=log_dir
                            )

                            updated_playbook = self.curator.update_bullet_counts(self.playbook_manager.playbook, bullet_tags)
                            self.playbook_manager.update_playbook(updated_playbook)

                            final_answer, bullet_ids_used, trajectory = await self.generator.generate(
                                question=query,
                                playbook=self.playbook_manager.retrieve_bullets(query, self.rae_top_k) if self.use_rae else self.playbook_manager.playbook,
                                cases_text=cases_text,
                                context=context,
                                use_json_mode=config.get("json_mode", False),
                                call_id=f"online_s{global_step}_r{r}",
                                log_dir=log_dir
                            )
                            if data_processor.answer_is_correct(final_answer, target):
                                print(f"Corrected reasoning on round {r}!")
                                break

                        # Store distilled insights from the last reflection into memory
                        if self.use_failure_memory and self.failure_memory and reflection not in ("(empty)", ""):
                            try:
                                parsed = json.loads(reflection) if isinstance(reflection, str) else {}
                            except (json.JSONDecodeError, TypeError):
                                parsed = {}
                            self.failure_memory.add(
                                question=query,
                                predicted_answer=initial_answer,
                                ground_truth=target,
                                error_identification=parsed.get("error_identification", ""),
                                root_cause=parsed.get("root_cause_analysis", ""),
                                key_insight=parsed.get("key_insight", ""),
                            )

                        stats = self.curator.get_playbook_stats(self.playbook_manager.playbook)
                        updated_playbook, self.next_global_id, operations, _ = self.curator.curate(
                            current_playbook=self.playbook_manager.playbook,
                            recent_reflection=reflection,
                            question_context=context,
                            current_step=global_step,
                            total_samples=len(test_samples),
                            token_budget=token_budget,
                            playbook_stats=stats,
                            call_id=f"online_curate_s{global_step}",
                            log_dir=log_dir,
                            next_global_id=self.next_global_id
                        )
                        self.playbook_manager.update_playbook(updated_playbook)

                    # Run adversarial episode
                    if self.use_adversarial and self.adversarial_agent:
                        await self._run_adversarial_episode(
                            step_id=f"online_s{global_step}",
                            epoch=win_idx + 1,
                            step=global_step,
                            log_dir=log_dir,
                            config=config,
                            total_samples=len(test_samples or []),
                            base_question=query,
                            base_context=context,
                            base_target=target,
                            data_processor=data_processor
                        )

                    # Save intermediate playbook at each step
                    save_steps = config.get("save_steps", 50)
                    if global_step % save_steps == 0:
                        step_playbook_path = os.path.join(run_path, f"step_{global_step}_playbook.txt")
                        with open(step_playbook_path, "w", encoding="utf-8") as f:
                            f.write(self.playbook_manager.playbook)
            
            final_test_accuracy = correct_count / total_count if total_count > 0 else 0.0
            results["online_test_results"] = {
                "accuracy": final_test_accuracy,
                "correct": correct_count,
                "total": total_count
            }

            # Save final playbook and case bank
            final_playbook_path = os.path.join(run_path, "final_playbook.txt")
            with open(final_playbook_path, "w", encoding="utf-8") as f:
                f.write(self.playbook_manager.playbook)
            
            # Print final summary banner matching standard ACE format
            print(f"\n{'='*60}")
            print(f"ONLINE TRAIN AND TEST COMPLETE")
            print(f"{'='*60}")
            print(f"Mode: {mode.upper()}")
            print(f"Initial Test Accuracy: {initial_test_accuracy:.3f}")
            print(f"Final Test Accuracy: {final_test_accuracy:.3f}")
            print(f"Results saved to: {run_path}")
            print(f"{'='*60}\n")

        elif mode == "eval_only":
            print(f"--- Starting Evaluation on {len(test_samples or [])} samples ---")
            answers = []
            targets = []
            for step, sample in enumerate(test_samples or [], 1):
                query = sample.get("question", "")
                context = sample.get("context", "")
                target = sample.get("target", "")

                retrieved_cases = self.case_bank.retrieve_cases(query)
                cases_text = self.case_bank.format_cases_for_prompt(retrieved_cases)
                playbook = self.playbook_manager.retrieve_bullets(query, self.rae_top_k) if self.use_rae else self.playbook_manager.playbook

                final_answer, _, _ = await self.generator.generate(
                    question=query,
                    playbook=playbook,
                    cases_text=cases_text,
                    context=context,
                    use_json_mode=config.get("json_mode", False),
                    call_id=f"eval_s{step}",
                    log_dir=log_dir
                )

                is_correct = data_processor.answer_is_correct(final_answer, target)
                answers.append(final_answer)
                targets.append(target)
                print(f"Eval {step}: Pred={final_answer} | Target={target} | Correct={is_correct}")

            accuracy = data_processor.evaluate_accuracy(answers, targets) if test_samples else 0.0
            print(f"Evaluation Accuracy: {accuracy:.4f}")
            results["accuracy"] = accuracy

        # Save consolidated results
        final_results_path = os.path.join(run_path, "final_results.json")
        with open(final_results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        await self.executor.cleanup()
        return results

    async def _evaluate_validation_set(
        self,
        val_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
        log_dir: str
    ) -> float:
        """Evaluate validation set concurrently and return accuracy."""
        print(f"\n--- Running Parallel Validation on {len(val_samples)} samples ---")
        use_json_mode = config.get("json_mode", False)
        
        async def eval_single(step, sample):
            query = sample.get("question", "")
            context = sample.get("context", "")
            target = sample.get("target", "")

            retrieved_cases = self.case_bank.retrieve_cases(query)
            cases_text = self.case_bank.format_cases_for_prompt(retrieved_cases)
            playbook = self.playbook_manager.retrieve_bullets(query, self.rae_top_k) if self.use_rae else self.playbook_manager.playbook

            try:
                final_answer, _, _ = await self.generator.generate(
                    question=query,
                    playbook=playbook,
                    cases_text=cases_text,
                    context=context,
                    use_json_mode=use_json_mode,
                    call_id=f"val_s{step}",
                    log_dir=log_dir
                )
                return final_answer, target
            except Exception as e:
                print(f"Error evaluating validation sample {step}: {e}")
                return "No final answer found", target

        # Run concurrently using a Semaphore to respect config.test_workers
        test_workers = config.get("test_workers", 5)
        sem = asyncio.Semaphore(test_workers)
        async def sem_eval(step, sample):
            async with sem:
                return await eval_single(step, sample)

        tasks = [sem_eval(step, sample) for step, sample in enumerate(val_samples, 1)]
        results = await asyncio.gather(*tasks)
        
        predictions = [r[0] for r in results]
        targets = [r[1] for r in results]
        
        accuracy = data_processor.evaluate_accuracy(predictions, targets) if val_samples else 0.0
        correct = sum(1 for p, t in zip(predictions, targets) if data_processor.answer_is_correct(p, t))
        print(f"Validation Accuracy: {accuracy:.4f} ({correct}/{len(val_samples)} samples correct)")
        return accuracy

    async def _run_test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        playbook: str,
        config: Dict[str, Any],
        log_dir: str,
        prefix: str = "test"
    ) -> Dict[str, Any]:
        """Run evaluation on test set concurrently."""
        print(f"\n--- Running Parallel Test ({prefix}) on {len(test_samples)} samples ---")
        use_json_mode = config.get("json_mode", False)
        
        async def eval_single(step, sample):
            query = sample.get("question", "")
            context = sample.get("context", "")
            target = sample.get("target", "")

            retrieved_cases = self.case_bank.retrieve_cases(query)
            cases_text = self.case_bank.format_cases_for_prompt(retrieved_cases)
            p = self.playbook_manager.retrieve_bullets(query, self.rae_top_k) if self.use_rae else playbook

            try:
                final_answer, _, _ = await self.generator.generate(
                    question=query,
                    playbook=p,
                    cases_text=cases_text,
                    context=context,
                    use_json_mode=use_json_mode,
                    call_id=f"{prefix}_s{step}",
                    log_dir=log_dir
                )
                return final_answer, target
            except Exception as e:
                print(f"Error evaluating test sample {step}: {e}")
                return "No final answer found", target

        test_workers = config.get("test_workers", 5)
        sem = asyncio.Semaphore(test_workers)
        async def sem_eval(step, sample):
            async with sem:
                return await eval_single(step, sample)

        tasks = [sem_eval(step, sample) for step, sample in enumerate(test_samples, 1)]
        results = await asyncio.gather(*tasks)
        
        predictions = [r[0] for r in results]
        targets = [r[1] for r in results]
        
        accuracy = data_processor.evaluate_accuracy(predictions, targets) if test_samples else 0.0
        correct = sum(1 for p, t in zip(predictions, targets) if data_processor.answer_is_correct(p, t))
        print(f"Test Accuracy ({prefix}): {accuracy:.4f} ({correct}/{len(test_samples)} samples correct)")
        return {"accuracy": accuracy, "correct": correct, "total": len(test_samples)}

    async def _run_adversarial_episode(
        self,
        step_id: str,
        epoch: int,
        step: int,
        log_dir: str,
        config: Dict[str, Any],
        total_samples: int,
        base_question: str,
        base_context: str,
        base_target: str,
        data_processor: Any,
    ) -> Optional[Dict[str, Any]]:
        if not self.use_adversarial or not self.adversarial_agent:
            return None

        adversarial_frequency = self.adversarial_frequency
        if adversarial_frequency <= 0 or step % adversarial_frequency != 0:
            return None

        print("\n--- Running Adversarial Agent ---")

        use_json_mode = config.get("json_mode", False)
        token_budget = config.get("playbook_token_budget", 80000)
        task_name = config.get("task_name", "default")

        attack, _ = self.adversarial_agent.generate_attack(
            playbook=self.playbook_manager.playbook,
            task_name=task_name,
            recent_question=base_question,
            recent_context=base_context,
            recent_target=base_target,
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_adv_generate",
            log_dir=log_dir,
        )

        if not attack:
            return None

        adv_question = attack.get("question", "")
        adv_context = attack.get("context", "")
        adv_target = attack.get("target", "")
        attack_rationale = attack.get("attack_rationale", "")
        vulnerability_hint = attack.get("vulnerability_hint", "")

        playbook = self.playbook_manager.retrieve_bullets(adv_question, self.rae_top_k) if self.use_rae else self.playbook_manager.playbook
        retrieved_cases = self.case_bank.retrieve_cases(adv_question)
        cases_text = self.case_bank.format_cases_for_prompt(retrieved_cases)

        adv_response, adv_bullet_ids, trajectory = await self.generator.generate(
            question=adv_question,
            playbook=playbook,
            cases_text=cases_text,
            context=adv_context,
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_adv_exec",
            log_dir=log_dir,
        )

        adv_answer = adv_response
        adv_correct = data_processor.answer_is_correct(adv_answer, adv_target)
        reflection_content = "(empty)"

        if not adv_correct:
            bullets_used_str = "\n".join([b["original_line"] for b in self.playbook_manager.bullets if b["id"] in adv_bullet_ids])
            environment_feedback = "Adversarial test: predicted answer does not match adversarial target."
            if attack_rationale:
                environment_feedback += f" Intended trap: {attack_rationale}"
            if vulnerability_hint:
                environment_feedback += f" Vulnerability hint: {vulnerability_hint}"

            if self.use_failure_memory and self.failure_memory:
                similar_failures = self.failure_memory.retrieve(adv_question)
                analogical_context = self.failure_memory.format_for_prompt(similar_failures)
            else:
                neg_cases = [c for c in retrieved_cases if c.get("reward") == 0]
                analogical_context = self.case_bank.format_cases_for_prompt(neg_cases)

            reflection_content, bullet_tags, _ = self.reflector.reflect(
                question=adv_question,
                trajectory_str=json.dumps(trajectory, indent=2),
                predicted_answer=adv_answer,
                ground_truth=adv_target,
                environment_feedback=environment_feedback,
                bullets_used_str=bullets_used_str,
                analogical_context=analogical_context,
                use_ground_truth=True,
                use_json_mode=use_json_mode,
                call_id=f"{step_id}_adv_reflect",
                log_dir=log_dir,
            )

            updated_playbook = self.curator.update_bullet_counts(self.playbook_manager.playbook, bullet_tags)
            self.playbook_manager.update_playbook(updated_playbook)

            if self.use_failure_memory and self.failure_memory and reflection_content not in ("(empty)", ""):
                try:
                    parsed = json.loads(reflection_content) if isinstance(reflection_content, str) else {}
                except (json.JSONDecodeError, TypeError):
                    parsed = {}
                self.failure_memory.add(
                    question=adv_question,
                    predicted_answer=adv_answer,
                    ground_truth=adv_target,
                    error_identification=parsed.get("error_identification", ""),
                    root_cause=parsed.get("root_cause_analysis", ""),
                    key_insight=parsed.get("key_insight", ""),
                )

            stats = self.curator.get_playbook_stats(self.playbook_manager.playbook)
            question_context = (
                f"Adversarial question: {adv_question}\n"
                f"Context: {adv_context}\n"
                f"Attack rationale: {attack_rationale}\n"
                f"Vulnerability hint: {vulnerability_hint}"
            )
            updated_playbook, self.next_global_id, _, _ = self.curator.curate(
                current_playbook=self.playbook_manager.playbook,
                recent_reflection=reflection_content,
                question_context=question_context,
                current_step=step,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats,
                call_id=f"{step_id}_adv_curate",
                log_dir=log_dir,
                next_global_id=self.next_global_id,
            )
            self.playbook_manager.update_playbook(updated_playbook)

        return {
            "question": adv_question,
            "context": adv_context,
            "target": adv_target,
            "predicted_answer": adv_answer,
            "is_correct": adv_correct,
            "attack_rationale": attack_rationale,
            "vulnerability_hint": vulnerability_hint,
        }


