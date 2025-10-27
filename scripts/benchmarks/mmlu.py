#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmarks/mmlu.py
- MMLU evaluator with vLLM (1-shot default), following the standard
  multiple-choice scoring protocol.
- Prefer the modern Hugging Face "cais/mmlu" dataset (dev/test on the hub).
  Fall back to legacy "hendrycks_test" only if absolutely necessary.
- Records KV meter (engine.reset_kv_meter / engine.get_kv_meter_summary).
- Returns a dict to be written by the orchestrator.

Dependencies
- vllm, datasets
"""
import os
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")    # suppress noisy INFO logs
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")    # disable vLLM default logging setup

from vllm import LLM, SamplingParams
from datasets import load_dataset

# 57 MMLU subjects
MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

CHOICES = ["A", "B", "C", "D"]

@dataclass
class Example:
    subject: str
    question: str
    choices: List[str]  # exactly 4
    answer: str         # one of "A","B","C","D"


# -------------------------
# Row normalization helpers
# -------------------------
def _choices_from_row(row: Dict[str, Any]) -> List[str]:
    """
    Normalize 'choices' to a list[str] length-4 in A,B,C,D order.
    - 'cais/mmlu': choices is a list[str]
    - 'hendrycks_test': choices is a dict with keys 'A','B','C','D'
    """
    raw = row.get("choices")
    if isinstance(raw, dict):
        # legacy schema with keys "A","B","C","D"
        return [str(raw[k]).strip() for k in CHOICES]
    # otherwise assume list-like
    lst = list(raw) if raw is not None else []
    if len(lst) < 4:
        raise ValueError(f"Malformed choices: expected 4, got {len(lst)}")
    return [str(lst[i]).strip() for i in range(4)]


def _answer_letter_from_row(row: Dict[str, Any]) -> str:
    """
    Normalize 'answer' to one of 'A','B','C','D'.
    - 'cais/mmlu': answer is usually an int index (0..3)
    - 'hendrycks_test': answer is a letter 'A'..'D'
    Also accept common fallbacks: 'target', 'label', 'correct', 'answer_idx'.
    """
    ans_raw = None
    for key in ("answer", "target", "label", "correct", "answer_idx"):
        if key in row:
            ans_raw = row[key]
            break
    # try integer index
    try:
        idx = int(ans_raw)
        if 0 <= idx < 4:
            return CHOICES[idx]
    except Exception:
        pass
    # try letter
    if isinstance(ans_raw, str):
        s = ans_raw.strip().upper()
        if s in CHOICES:
            return s
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < 4:
                return CHOICES[idx]
    raise ValueError(f"Unrecognized answer format: {ans_raw!r}")

# -------------------------
# Robust dataset loaders
# -------------------------

def _load_split(subject: str, split: str):
    """
    Try loading from modern repo first, then legacy:
      1) "cais/mmlu" (subject configs, dev/test available on the Hub)
      2) "hendrycks_test" (may require external tar download; avoid if possible)
    """
    # modern (preferred): on-Hub parquet w/ dev/test
    try:
        return load_dataset("cais/mmlu", subject, split=split)
    except Exception:
        pass

    # legacy (fallback): might try to fetch Berkeley tar externally
    try:
        return load_dataset("hendrycks_test", subject, split=split)
    except Exception:
        pass

    return None


def _build_subject_demo(subject: str) -> Optional[Example]:
    """Pick one dev example to serve as the in-context demo for this subject."""
    ds = _load_split(subject, "dev")
    if ds is None or len(ds) == 0:
        return None
    row = random.choice(ds)
    q = str(row["question"]).strip()
    choices = _choices_from_row(row)
    ans = _answer_letter_from_row(row)
    return Example(subject, q, choices, ans)


def _iter_test_examples(subject: str):
    """Yield test examples for a subject as Example objects."""
    ds = _load_split(subject, "test")
    if ds is None:
        return
    for row in ds:
        q = str(row["question"]).strip()
        choices = _choices_from_row(row)
        ans = _answer_letter_from_row(row)
        yield Example(subject, q, choices, ans)


def _format_demo_block(example: Example) -> str:
    """Format a single demo example as (Q, choices, Answer: X)."""
    lines = [f"Question: {example.question}"]
    for i, ch in enumerate(CHOICES):
        lines.append(f"{ch}. {example.choices[i]}")
    lines.append(f"Answer: {example.answer}")
    return "\n".join(lines)


def _format_query(example: Example, demo: Optional[Example], kshot: int) -> str:
    """Build the final prompt with (k-shot) demos for the same subject."""
    parts = [f"Subject: {example.subject.replace('_',' ')}\n"]
    if demo is not None and kshot >= 1:
        parts.append(_format_demo_block(demo))
        parts.append("")  # blank line before the actual query
    lines = [f"Question: {example.question}"]
    for i, ch in enumerate(CHOICES):
        lines.append(f"{ch}. {example.choices[i]}")
    lines.append("Answer:")
    parts.append("\n".join(lines))
    return "\n".join(parts)


def _pick_letter_from_logprobs(g0) -> str:
    """Extract top letter A/B/C/D from the first generated token's logprobs.
    This is robust to tokenizers that include a leading space.
    """
    if not g0 or ("logprob" not in g0):
        return ""
    token = g0.get("token", "")
    stripped = token.strip().upper()
    if stripped in CHOICES:
        return stripped
    for ch in CHOICES:
        if stripped.startswith(ch):
            return ch
    return ""


def _score_batch_text(outputs, gold_letters: List[str]) -> Tuple[int, List[str]]:
    """Return (#correct, list_of_pred_letters) using vLLM outputs with logprobs."""
    correct = 0
    preds = []
    for out, gold in zip(outputs, gold_letters):
        if not out.outputs:
            preds.append("")
            continue
        cand = out.outputs[0]
        logprob_entries = cand.logprobs or []
        first_tok = logprob_entries[0] if logprob_entries else None
        pred = _pick_letter_from_logprobs(first_tok)
        if not pred:
            txt = (cand.text or "").strip().upper()
            pred = txt[:1] if txt[:1] in CHOICES else ""
        preds.append(pred)
        if pred == gold:
            correct += 1
    return correct, preds


def run(model_path: str,
        results_dir: str,
        num_samples: int = 100,
        kshot: int = 1,
        batch_size: int = 8,
        max_new_tokens: int = 2,
        seed: int = 2025,
        tag: str = "") -> Dict:
    """Run MMLU with vLLM and return a result dict for run_benchmark.py.

    Returns dict contains:
      - bench, model, kshot, num_samples, seed, started_at, duration_sec
      - score (macro_subject_acc), micro_acc
      - per_subject: {subject: {"n": int, "acc": float}}
      - kv_meter_summary: dict (from engine.get_kv_meter_summary())
    """
    assert kshot in (0, 1), "This simple runner currently supports kshot in {0,1}"
    random.seed(seed)

    # Prepare subjects and demos
    subject_to_demo = {}
    subject_to_examples = {}
    for s in MMLU_SUBJECTS:
        demo = _build_subject_demo(s)
        if demo is not None:
            subject_to_demo[s] = demo
        examples = list(_iter_test_examples(s))
        subject_to_examples[s] = examples

    # Pool examples across subjects and sample num_samples
    pool: List[Example] = []
    for s in MMLU_SUBJECTS:
        pool.extend(subject_to_examples.get(s, []))
    if not pool:
        raise RuntimeError(
            "No MMLU data loaded. Prefer 'cais/mmlu' which hosts dev/test on the Hub. "
            "Check your network or set HF_DATASETS_OFFLINE=1 with a prefilled cache."
        )

    random.shuffle(pool)
    eval_set = pool[:num_samples]

    # Group by subject for macro averaging
    by_subject: Dict[str, List[Example]] = {}
    for ex in eval_set:
        by_subject.setdefault(ex.subject, []).append(ex)

    # Build all prompts and gold labels
    prompts: List[str] = []
    gold_letters: List[str] = []
    for ex in eval_set:
        demo = subject_to_demo.get(ex.subject, None)
        prompt = _format_query(ex, demo, kshot)
        prompts.append(prompt)
        gold_letters.append(ex.answer)

    # Initialize vLLM
    print(f"[vLLM][MMLU] Loading model: {model_path}")
    t_load0 = time.time()
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="float16",
        gpu_memory_utilization=0.92,
        disable_log_stats=False,
        path_debug=False,
    )
    print(f"[vLLM][MMLU] Load done in {time.time() - t_load0:.1f}s")

    # Sampling params: single token, deterministic
    sampling = SamplingParams(
        max_tokens=2,
        temperature=0.0,
        top_p=1.0,
        logprobs=5,
        stop=["\n"],
    )

    # Reset KV meter for this run
    engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
    assert engine is not None, "Could not access llm_engine/engine from LLM"
    run_id = f"mmlu@{int(time.time())}"
    if tag:
        run_id += f"@{tag}"
    engine.reset_kv_meter(run_id=run_id)

    # Generate in batches
    started_at = time.time()
    total_correct = 0
    preds_all: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_gold = gold_letters[i:i+batch_size]
        outs = llm.generate(batch_prompts, sampling)
        n_correct, preds = _score_batch_text(outs, batch_gold)
        total_correct += n_correct
        preds_all.extend(preds)

    duration_sec = time.time() - started_at
    micro_acc = total_correct / len(eval_set)

    # Macro (subject-averaged) accuracy
    subj_scores = {}
    for subj, exs in by_subject.items():
        idxs = [j for j, e in enumerate(eval_set) if e.subject == subj]
        if not idxs:
            continue
        c = sum(1 for j in idxs if preds_all[j] == gold_letters[j])
        subj_scores[subj] = {"n": len(idxs), "acc": (c / len(idxs)) if len(idxs) else 0.0}
    macro_acc = (sum(v["acc"] for v in subj_scores.values()) / len(subj_scores)) if subj_scores else 0.0

    # KV summary
    kv_summary = engine.get_kv_meter_summary()
    
    # Assemble result
    result = {
        "bench": "mmlu",
        "model": model_path,
        "kshot": kshot,
        "num_samples": len(eval_set),
        "seed": seed,
        "started_at": started_at,
        "duration_sec": duration_sec,
        "score": macro_acc,
        "score_display": f"macro={macro_acc*100:.2f}%, micro={micro_acc*100:.2f}%",
        "micro_acc": micro_acc,
        "per_subject": subj_scores,
        "kv_meter_summary": kv_summary,
        "run_id": run_id,
    }
    print(f"[MMLU] {result['score_display']} | N={len(eval_set)} | time={duration_sec:.2f}s")
    return result

