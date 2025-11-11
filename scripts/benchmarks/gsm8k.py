#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmarks/gsm8k.py
- GSM8K evaluator with vLLM, 1-shot Chain-of-Thought (CoT) by default.
- Dataset: "gsm8k", config "main"; we use train for the 1-shot demo and test for eval.
- Score: exact match on the final numeric answer parsed from the last '#### <ans>'.
- Records KV meter (engine.reset_kv_meter / engine.get_kv_meter_summary).
"""

import os
import re
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

from vllm import LLM, SamplingParams
import gc, time
from datasets import load_dataset

@dataclass
class Example:
    question: str
    answer: str

def _load_one_demo(seed: int = 42) -> Optional[Example]:
    ds = load_dataset("gsm8k", "main", split="train")
    if len(ds) == 0:
        return None
    rnd = random.Random(seed)
    row = ds[rnd.randrange(0, len(ds))]
    return Example(row["question"].strip(), row["answer"].strip())

def _iter_test_examples():
    ds = load_dataset("gsm8k", "main", split="test")
    for row in ds:
        yield Example(row["question"].strip(), row["answer"].strip())

def _normalize_final_number(text: str) -> str:
    import re
    if not text:
        return ""
    m = None
    for m in re.finditer(r"####\s*([\-]?\d+(?:\.\d+)?)", text):
        pass
    if m:
        return m.group(1).replace(",", "")
    m2 = None
    for m2 in re.finditer(r"([\-]?\d+(?:\.\d+)?)", text):
        pass
    return m2.group(1).replace(",", "") if m2 else ""

def _format_demo_block(demo: Example) -> str:
    return f"Problem:\n{demo.question}\n\nSolution:\n{demo.answer}\n"

def _format_query(ex: Example, demo: Optional[Example], kshot: int) -> str:
    parts = []
    if demo is not None and kshot >= 1:
        parts.append(_format_demo_block(demo))
    parts.append(f"Problem:\n{ex.question}\n\nSolution:\n")
    return "\n".join(parts)

def _score_batch(outputs, gold_answers: List[str]) -> Tuple[int, List[str]]:
    correct = 0
    preds_norm = []
    for out, gold_raw in zip(outputs, gold_answers):
        gold_norm = _normalize_final_number(gold_raw)
        if not out.outputs:
            preds_norm.append("")
            continue
        pred_text = (out.outputs[0].text or "").strip()
        pred_norm = _normalize_final_number(pred_text)
        preds_norm.append(pred_norm)
        if pred_norm == gold_norm and pred_norm != "":
            correct += 1
    return correct, preds_norm

def run(model_path: str,
        results_dir: str,
        num_samples: int = 100,
        kshot: int = 1,
        batch_size: int = 4,
        max_new_tokens: int = 256,
        seed: int = 2025,
        tag: str = "") -> Dict:
    random.seed(seed)

    demo = _load_one_demo(seed=seed)
    test_pool = list(_iter_test_examples())
    if not test_pool:
        raise RuntimeError("No GSM8K data loaded. Ensure 'datasets' can access 'gsm8k' (config 'main').")

    random.shuffle(test_pool)
    eval_set = test_pool[:num_samples]

    prompts = [_format_query(ex, demo if kshot >= 1 else None, kshot) for ex in eval_set]
    gold_answers = [ex.answer for ex in eval_set]

    print(f"[vLLM][GSM8K] Loading model: {model_path}")
    llm = None
    try:
        t_load0 = time.time()
        llm = LLM(
            model=model_path,
            tokenizer=model_path,
            gpu_memory_utilization=0.92,
            disable_log_stats=False,
            path_debug=False,
        )
        print(f"[vLLM][GSM8K] Load done in {time.time() - t_load0:.1f}s")
    
        sampling = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )
    
        engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
        assert engine is not None, "Could not access llm_engine/engine from LLM"
        run_id = f"gsm8k@{int(time.time())}"
        if tag:
            run_id += f"@{tag}"
        engine.reset_kv_meter(run_id=run_id)
    
        started_at = time.time()
        total_correct = 0
        for i in range(0, len(prompts), batch_size):
            outs = llm.generate(prompts[i:i+batch_size], sampling)
            c, _ = _score_batch(outs, gold_answers[i:i+batch_size])
            total_correct += c
        duration_sec = time.time() - started_at
        acc = total_correct / len(eval_set)
    
        kv_summary = engine.get_kv_meter_summary()
        
        result = {
            "bench": "gsm8k",
            "model": model_path,
            "kshot": kshot,
            "num_samples": len(eval_set),
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "started_at": started_at,
            "duration_sec": duration_sec,
            "score": acc,
            "score_display": f"acc={acc*100:.2f}%",
            "kv_meter_summary": kv_summary,
            "run_id": run_id,
        }
        print(f"[GSM8K] {result['score_display']} | N={len(eval_set)} | time={duration_sec:.2f}s")
        return result
    finally:
        engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as e:
                print(f"[GSM8K] llm.shutdown() error: {e}")
        del llm
        gc.collect()
        time.sleep(10)

