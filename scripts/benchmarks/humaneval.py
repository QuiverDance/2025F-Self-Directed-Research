#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmarks/humaneval.py
- HumanEval pass@1 evaluator with vLLM.
- Dataset: "openai_humaneval" (test split). Each sample contains:
    * task_id (e.g., "HumanEval/0")
    * prompt (function signature + docstring)
    * entry_point (function name to implement)
    * test (python assertions that call the entry_point)
- Generation: complete the prompt to produce the function body.
- Scoring: execute prompt+completion+test in a subprocess; success => pass.
- Records KV meter (engine.reset_kv_meter / engine.get_kv_meter_summary).
"""

import os
import re
import sys
import time
import json
import tempfile
import random
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

from vllm import LLM, SamplingParams
from datasets import load_dataset

@dataclass
class Task:
    task_id: str
    prompt: str
    entry_point: str
    test: str

def _load_tasks() -> List[Task]:
    ds = load_dataset("openai_humaneval", split="test")
    tasks = []
    for row in ds:
        tasks.append(Task(
            task_id=row["task_id"],
            prompt=row["prompt"],
            entry_point=row["entry_point"],
            test=row["test"],
        ))
    return tasks

def _strip_code_fences(text: str) -> str:
    if text is None:
        return ""
    out = text
    out = re.sub(r"^\s*```(?:python)?\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s*```\s*$", "", out)
    return out

def _build_program(prompt: str, completion: str, test: str) -> str:
    body = _strip_code_fences(completion)
    return f"# -*- coding: utf-8 -*-\n{prompt}{body}\n\n{test}\n"

def _run_test_in_subprocess(program_text: str, timeout_s: int = 5) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="humaneval_") as td:
        prog_path = os.path.join(td, "prog.py")
        with open(prog_path, "w", encoding="utf-8") as f:
            f.write(program_text)
        try:
            p = subprocess.run(
                [sys.executable, prog_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
                env={**os.environ, "PYTHONPATH": ""},
            )
            passed = (p.returncode == 0)
            err_tail = (p.stderr or "")[-500:]
            return passed, err_tail
        except subprocess.TimeoutExpired as e:
            return False, f"TIMEOUT after {timeout_s}s"
        except Exception as e:
            return False, f"ERROR: {e}"

def run(model_path: str,
        results_dir: str,
        num_samples: int = 164,
        kshot: int = 0,
        batch_size: int = 1,
        max_new_tokens: int = 512,
        seed: int = 2025,
        tag: str = "") -> Dict:
    random.seed(seed)

    tasks = _load_tasks()
    if not tasks:
        raise RuntimeError("No HumanEval data loaded. Ensure 'datasets' can access 'openai_humaneval'.")
    random.shuffle(tasks)
    eval_set = tasks[:num_samples]

    prompts = [t.prompt for t in eval_set]

    print(f"[vLLM][HumanEval] Loading model: {model_path}")
    t_load0 = time.time()
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="float16",
        gpu_memory_utilization=0.92,
        disable_log_stats=False,
        path_debug=False,
    )
    print(f"[vLLM][HumanEval] Load done in {time.time() - t_load0:.1f}s")

    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        stop=None,
    )

    engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
    assert engine is not None, "Could not access llm_engine/engine from LLM"
    run_id = f"humaneval@{int(time.time())}"
    if tag:
        run_id += f"@{tag}"
    engine.reset_kv_meter(run_id=run_id)

    started_at = time.time()
    per_task = []
    passed_cnt = 0

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        outs = llm.generate(batch_prompts, sampling)
        for j, out in enumerate(outs):
            t = eval_set[i + j]
            if not out.outputs:
                per_task.append({"task_id": t.task_id, "pass": False, "err": "NO_OUTPUT"})
                continue
            completion = out.outputs[0].text or ""
            program = _build_program(t.prompt, completion, t.test)
            ok, err_tail = _run_test_in_subprocess(program_text=program, timeout_s=10)
            per_task.append({"task_id": t.task_id, "pass": ok, "err": err_tail})
            if ok:
                passed_cnt += 1

    duration_sec = time.time() - started_at
    pass_at_1 = passed_cnt / len(eval_set)

    kv_summary = engine.get_kv_meter_summary()

    result = {
        "bench": "humaneval",
        "model": model_path,
        "num_samples": len(eval_set),
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "started_at": started_at,
        "duration_sec": duration_sec,
        "score": pass_at_1,
        "score_display": f"pass@1={pass_at_1*100:.2f}% ({passed_cnt}/{len(eval_set)})",
        "kv_meter_summary": kv_summary,
        "run_id": run_id,
        "per_task": per_task,
    }
    print(f"[HumanEval] {result['score_display']} | N={len(eval_set)} | time={duration_sec:.2f}s")
    return result

