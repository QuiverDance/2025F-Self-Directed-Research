#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmarks/needle.py
- Needle-in-a-Haystack (NIAH) benchmark.
- Build a long document of ~T tokens and insert a unique needle value at a specified depth.
- Ask the model to return exactly the needle value.
- Metric: accuracy (exact value match after light normalization).

Standard choices:
- Document length given in *tokens* (approx, tokenizer-aware).
- Needle format: 'NEEDLE: <value>' where <value> is a deterministic code.
- Prompt requires "Output ONLY the value".
- Location: by depth fraction (0.0=begin, 0.5=middle, 1.0=end) or modes: head/middle/tail/random.

We enforce tokenizer-aware budgets using the model's max context length.
"""

import os
import re
import time
import math
import random
import string
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

from vllm import LLM, SamplingParams
import gc, time

@dataclass
class Sample:
    prompt: str
    gold_value: str

COMMON_WORDS = [
    "the","of","and","to","in","a","is","it","you","that","he","was","for","on","are","as","with",
    "his","they","I","at","be","this","have","from","or","one","had","by","word","but","not","what",
    "all","were","we","when","your","can","said","there","use","an","each","which","do","how","their",
    "if","will","up","other","about","out","many","then","them","these","so","some","her","would","make",
    "like","him","into","time","has","look","two","more","write","go","see","number","no","way","could",
]

def _rand_word_random(rng: random.Random, min_len=3, max_len=9) -> str:
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))

def _rand_word_common(rng: random.Random) -> str:
    return rng.choice(COMMON_WORDS)

def _make_tokens_to_budget(tokenizer, target_tokens: int, rng: random.Random, vocab_mode: str) -> str:
    """Generate text whose tokenized length is close to target_tokens."""
    parts = []
    # Add words by sentences to amortize encode calls
    sent_words = rng.randint(7, 14)
    while True:
        if vocab_mode == "common":
            words = [_rand_word_common(rng) for _ in range(sent_words)]
        else:
            words = [_rand_word_random(rng) for _ in range(sent_words)]
        parts.append(" ".join(words) + ". ")
        txt = "".join(parts)
        if len(tokenizer.encode(txt)) >= target_tokens:
            return txt

def _insert_needle_by_depth(text: str, tokenizer, needle_sentence: str, depth: float) -> str:
    """Insert needle at desired depth (fraction of tokens) preserving overall order."""
    ids = tokenizer.encode(text)
    k = max(0, min(len(ids), int(round(depth * len(ids)))))
    head = tokenizer.decode(ids[:k], skip_special_tokens=True)
    tail = tokenizer.decode(ids[k:], skip_special_tokens=True)
    return head + needle_sentence + " " + tail

def _build_prompt(doc: str, value: str) -> str:
    instr = (
        "You will read a long document. Hidden within it is a sentence of the form 'NEEDLE: <value>'.\n"
        "Return EXACTLY the <value>. Do not add any extra words or punctuation. Output only the value.\n"
    )
    q = "Question: What is the hidden value?\n"
    return f"{instr}\nDocument:\n{doc}\n\n{q}Answer:"

def _normalize_value(s: str) -> str:
    if s is None:
        return ""
    # strip quotes and whitespace/punctuation at ends
    s = s.strip().strip("\"'`.,;: ")
    return s

def _extract_first_token_text(output_text: str) -> str:
    """Take model text; if it contains whitespace, take the first token-ish segment; else full."""
    if not output_text:
        return ""
    # Some models echo "The hidden value is 1234" despite instruction; try last token group too
    t = output_text.strip()
    # Prefer the last token group if colon pattern present
    m = re.search(r"([A-Za-z0-9\-]{2,})", t)
    return m.group(1) if m else t.split()[0]

def _score_batch(outputs, gold_values: List[str]) -> Tuple[int, List[str]]:
    correct = 0
    preds = []
    for out, gold in zip(outputs, gold_values):
        if not out.outputs:
            preds.append("")
            continue
        pred_text = (out.outputs[0].text or "").strip()
        pred_val = _normalize_value(pred_text)
        # If model produced a phrase, try to extract a code-like token
        if len(pred_val.split()) > 1:
            pred_val = _extract_first_token_text(pred_val)
        preds.append(pred_val)
        if pred_val == gold:
            correct += 1
    return correct, preds

def run(model_path: str,
        results_dir: str,
        num_samples: int = 20,
        target_tokens: int = 12000,
        depth: float = 0.5,
        depth_mode: str = "fraction",   # "fraction" | "head" | "middle" | "tail" | "random"
        vocab_mode: str = "random",     # "random" (default) | "common"
        batch_size: int = 1,
        max_new_tokens: int = 8,
        ctx_margin_tokens: int = 128,
        seed: int = 2025,
        tag: str = "") -> Dict:
    """Run NIAH and return a result dict with accuracy."""
    rng = random.Random(seed)

    # Init vLLM
    print(f"[vLLM][NIAH] Loading model: {model_path}")
    llm = None
    try:
        t_load0 = time.time()
        llm = LLM(
            model=model_path,
            tokenizer=model_path,
            dtype="float16",
            gpu_memory_utilization=0.92,
            disable_log_stats=False,
            path_debug=False,
        )
        print(f"[vLLM][NIAH] Load done in {time.time() - t_load0:.1f}s")
    
        tokenizer = llm.get_tokenizer()
        engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
        assert engine is not None, "Could not access llm_engine/engine from LLM"
        max_len = getattr(getattr(engine, "model_config", None), "max_model_len", None) or 32768
        # Total budget for prompt (leave margin + generation)
        budget = max(256, max_len - ctx_margin_tokens - max_new_tokens)
    
        # Determine depth fraction
        if depth_mode != "fraction":
            if depth_mode == "head":
                depth_frac = 0.05
            elif depth_mode == "middle":
                depth_frac = 0.5
            elif depth_mode == "tail":
                depth_frac = 0.95
            elif depth_mode == "random":
                depth_frac = rng.random()
            else:
                depth_frac = 0.5
        else:
            depth_frac = float(max(0.0, min(1.0, depth)))
    
        # Build samples
        samples: List[Sample] = []
        for i in range(num_samples):
            # choose deterministic needle value for reproducibility across seeds
            rng_i = random.Random(seed + 100003 * i)
            value = f"N{rng_i.randint(100000, 999999)}"
            needle_sentence = f"NEEDLE: {value}."
            # allocate near-target token length but cap by budget
            target = min(target_tokens, budget - 64)  # keep some room for headers
            if target <= 64:
                target = max(64, budget // 2)
            hay = _make_tokens_to_budget(tokenizer, target, rng_i, vocab_mode=vocab_mode)
            doc = _insert_needle_by_depth(hay, tokenizer, needle_sentence, depth_frac)
            prompt = _build_prompt(doc, value)
            samples.append(Sample(prompt=prompt, gold_value=value))
    
        prompts = [s.prompt for s in samples]
        golds = [s.gold_value for s in samples]
    
        sampling = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=None,
        )
    
        # Reset KV meter
        run_id = f"needle@{int(time.time())}"
        if tag:
            run_id += f"@{tag}"
        engine.reset_kv_meter(run_id=run_id)
    
        # Generate & score
        started_at = time.time()
        total_correct = 0
        for i in range(0, len(prompts), batch_size):
            outs = llm.generate(prompts[i:i+batch_size], sampling)
            c, _ = _score_batch(outs, golds[i:i+batch_size])
            total_correct += c
        duration_sec = time.time() - started_at
        acc = total_correct / len(samples)
    
        kv_summary = engine.get_kv_meter_summary()
    
        result = {
            "bench": "needle",
            "model": model_path,
            "num_samples": len(samples),
            "seed": seed,
            "config": {
                "target_tokens": target_tokens,
                "depth": depth,
                "depth_mode": depth_mode,
                "vocab_mode": vocab_mode,
                "batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
            },
            "started_at": started_at,
            "duration_sec": duration_sec,
            "score": acc,
            "score_display": f"acc={acc*100:.2f}% ({total_correct}/{len(samples)})",
            "kv_meter_summary": kv_summary,
            "run_id": run_id,
        }
        print(f"[NIAH] {result['score_display']} | N={len(samples)} | time={duration_sec:.2f}s | budget={budget} tok")
        return result
    finally:
        engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as e:
                print(f"[NIAH] llm.shutdown() error: {e}")
        del llm
        gc.collect()
        time.sleep(10)
