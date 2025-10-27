#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmarks/line_retrieval.py
- Synthetic Line Retrieval benchmark (standard long-context retrieval).
- Build a numbered list of L lines; ask the model to output exactly the text of a target line.
- Score: exact match after strip (no extra words, no "Line i:" prefix).

New in this patch:
- Tokenizer-aware context budgeting: ensure each prompt fits the model's max context.
- Token-friendly vocabulary: optional 'common' mode uses frequent short English words
  to avoid BPE fragmentation (which can explode token length).
- Auto-shrinking: if a prompt would exceed budget, the number of lines is reduced
  proportionally (with a small margin) and rebuilt.

Config knobs:
- num_lines: how many numbered lines (L)
- min_words, max_words: number of words per line
- vocab_mode: "random" (default, random lowercase strings) | "common" (frequent short words)
- target_mode: "random" | "head" | "middle" | "tail"
- ctx_budget_tokens: if None, use (engine.max_model_len - ctx_margin_tokens)
- ctx_margin_tokens: reserved tokens below max context (default: 128)
- max_new_tokens: decoding cap

Records KV meter: engine.reset_kv_meter(...) / engine.get_kv_meter_summary().
"""

import os
import time
import math
import random
import string
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

from vllm import LLM, SamplingParams

@dataclass
class Sample:
    prompt: str
    gold: str

COMMON_WORDS = [
    # short, frequent tokens to keep BPE lengths minimal
    "the","of","and","to","in","a","is","it","you","that","he","was","for","on","are","as","with",
    "his","they","I","at","be","this","have","from","or","one","had","by","word","but","not","what",
    "all","were","we","when","your","can","said","there","use","an","each","which","do","how","their",
    "if","will","up","other","about","out","many","then","them","these","so","some","her","would","make",
    "like","him","into","time","has","look","two","more","write","go","see","number","no","way","could",
]

def _rand_word_random(rng: random.Random, min_len=3, max_len=8) -> str:
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))

def _rand_word_common(rng: random.Random) -> str:
    return rng.choice(COMMON_WORDS)

def _make_corpus(num_lines: int, min_words: int, max_words: int, seed: int, vocab_mode: str) -> List[str]:
    rng = random.Random(seed)
    lines = []
    for _ in range(num_lines):
        k = rng.randint(min_words, max_words)
        if vocab_mode == "random":
            words = [_rand_word_random(rng) for _ in range(k)]
        else:
            words = [_rand_word_common(rng) for _ in range(k)]
        lines.append(" ".join(words))
    return lines

def _choose_target_index(num_lines: int, mode: str, rng: random.Random) -> int:
    if mode == "head":
        return 1
    if mode == "middle":
        return (num_lines + 1) // 2
    if mode == "tail":
        return num_lines
    return rng.randint(1, num_lines)

def _build_prompt(lines: List[str], target_idx: int) -> Tuple[str, str]:
    """Return (prompt, gold_text) where gold_text is the text of the target line."""
    L = len(lines)
    header = (
        "You will be shown a list of numbered lines. "
        f"There are {L} lines in total. "
        f"Return EXACTLY the text of the line with number {target_idx}. "
        "Do not include the line number or any extra words. "
        "Output only the line text.\n\n"
        "Numbered lines:\n"
    )
    body_lines = [f"Line {i+1}: {lines[i]}" for i in range(L)]
    prompt = header + "\n".join(body_lines) + "\n\nAnswer:"
    gold = lines[target_idx - 1]
    return prompt, gold

def _count_tokens(tokenizer, text: str) -> int:
    # vLLM tokenizer returns a list of ids when called as encode()
    return len(tokenizer.encode(text))

def _score_batch(outputs, gold_texts: List[str]) -> Tuple[int, List[str]]:
    correct = 0
    preds = []
    for out, gold in zip(outputs, gold_texts):
        if not out.outputs:
            preds.append("")
            continue
        pred = (out.outputs[0].text or "").strip()
        preds.append(pred)
        if pred.strip() == gold.strip():
            correct += 1
    return correct, preds

def run(model_path: str,
        results_dir: str,
        num_samples: int = 20,
        num_lines: int = 2000,
        min_words: int = 5,
        max_words: int = 9,
        vocab_mode: str = "random",
        target_mode: str = "random",
        batch_size: int = 1,
        max_new_tokens: int = 64,
        ctx_budget_tokens: Optional[int] = None,
        ctx_margin_tokens: int = 128,
        seed: int = 42,
        tag: str = "") -> Dict:
    """Run Line Retrieval benchmark and return a result dict.

    Returns dict contains:
      - bench, model, num_samples, config, score(acc), kv_meter_summary, ...
    """
    rng = random.Random(seed)

    # Initialize vLLM first so we can use its tokenizer and model max len
    print(f"[vLLM][LineRetrieval] Loading model: {model_path}")
    t_load0 = time.time()
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="float16",
        gpu_memory_utilization=0.92,
        disable_log_stats=False,
        path_debug=False,
    )
    print(f"[vLLM][LineRetrieval] Load done in {time.time() - t_load0:.1f}s")

    tokenizer = llm.get_tokenizer()
    engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
    assert engine is not None, "Could not access llm_engine/engine from LLM"
    # Determine token budget
    model_max = getattr(getattr(engine, "model_config", None), "max_model_len", None)
    if model_max is None:
        model_max = 32768  # sensible default if not exposed
    budget = ctx_budget_tokens if ctx_budget_tokens is not None else max(256, model_max - ctx_margin_tokens)

    # Build samples while enforcing the token budget
    samples: List[Sample] = []
    for s in range(num_samples):
        # Base corpus
        lines = _make_corpus(
            num_lines=num_lines,
            min_words=min_words,
            max_words=max_words,
            seed=seed + s * 9973,
            vocab_mode=vocab_mode,
        )
        target_idx = _choose_target_index(num_lines, target_mode, rng)

        # Try full size; if too long, shrink lines count proportionally
        attempt = 0
        cur_num = num_lines
        prompt, gold = _build_prompt(lines, target_idx)
        tok_len = _count_tokens(tokenizer, prompt)
        while tok_len > budget and attempt < 4 and cur_num > 1:
            # Estimate new line count
            shrink_ratio = budget / float(tok_len)
            new_num = max(1, int(math.floor(cur_num * shrink_ratio * 0.95)))  # keep a small safety margin
            if new_num == cur_num:
                new_num = max(1, cur_num - 1)
            # Rebuild smaller corpus deterministically
            lines = lines[:new_num]
            # keep target within range
            target_idx = min(target_idx, new_num)
            prompt, gold = _build_prompt(lines, target_idx)
            tok_len = _count_tokens(tokenizer, prompt)
            cur_num = new_num
            attempt += 1

        samples.append(Sample(prompt=prompt, gold=gold))

    prompts = [s.prompt for s in samples]
    golds = [s.gold for s in samples]

    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    # Reset KV meter
    run_id = f"line_retrieval@{int(time.time())}"
    if tag:
        run_id += f"@{tag}"
    engine.reset_kv_meter(run_id=run_id)

    # Generate
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
        "bench": "line_retrieval",
        "model": model_path,
        "num_samples": len(samples),
        "seed": seed,
        "config": {
            "num_lines": num_lines,
            "min_words": min_words,
            "max_words": max_words,
            "vocab_mode": vocab_mode,
            "target_mode": target_mode,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "ctx_budget_tokens": budget,
            "ctx_margin_tokens": ctx_margin_tokens,
        },
        "started_at": started_at,
        "duration_sec": duration_sec,
        "score": acc,
        "score_display": f"acc={acc*100:.2f}% ({total_correct}/{len(samples)})",
        "kv_meter_summary": kv_summary,
        "run_id": run_id,
    }
    print(f"[LineRetrieval] {result['score_display']} | N={len(samples)} | time={duration_sec:.2f}s | budget={budget} tok")
    return result

