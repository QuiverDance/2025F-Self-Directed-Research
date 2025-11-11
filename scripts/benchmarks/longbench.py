#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmarks/longbench.py
- Unified LongBench runner for: qasper, hotpotqa, 2wikimqa, musique.
- Standard extractive QA evaluation: token-level F1 (primary), EM (aux).
- Robust data loader that *does not rely on HF dataset scripts* (datasets>=4 compatible).
  It can:
    1) Load local JSONL files if provided via `data_dir` or env LONGEBENCH_DATA_DIR/LONGBENCH_DATA_DIR.
    2) Download `data.zip` from HF repos (THUDM/LongBench, zai-org/LongBench, yanbingzheng/LongBench)
       via huggingface_hub, then read the JSONL inside the zip.
- Handles long inputs safely with tokenizer-aware truncation that preserves the tail (keeps the question).

Usage (module API):
    run(model_path=..., results_dir=..., dataset="qasper", num_samples=100, data_dir=None, ...)

Returned dict:
    {
      "bench": "longbench",
      "dataset": "<name>",
      "score": <avg_f1>,
      "score_display": "F1=..%, EM=..%",
      "micro": {"f1": ..., "em": ...},
      "kv_meter_summary": {...},
      "per_item": [ { "id": ..., "f1": ..., "em": ..., "pred": "..."} ... ]
    }
"""

import os
import re
import io
import zipfile
import time
import math
import random
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

from vllm import LLM, SamplingParams
import gc, time
from datasets import load_dataset
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None  # optional

# --------------------
# Config & templates
# --------------------

VALID_DATASETS = ["qasper", "hotpotqa", "2wikimqa", "musique"]

INSTRUCTIONS = {
    "qasper": (
        "You are given a scientific article and a question.\n"
        "Answer ONLY with exact phrase(s) copied verbatim from the article.\n"
        "Do NOT add, paraphrase, or infer any new words.\n"
        "If the question cannot be answered from the article, write: unanswerable.\n"
        "If it is yes/no, answer: yes, no, or unanswerable.\n"
        "Output ONLY the answer text, without quotes or extra words.\n"
    ),
    "hotpotqa": (
        "You are given a long passage and a multi-hop question.\n"
        "Answer ONLY with the minimal text span from the passage.\n"
        "If it is yes/no, answer: yes or no. If unanswerable, write: unanswerable.\n"
        "Output ONLY the answer text.\n"
    ),
    "2wikimqa": (
        "You are given a long passage derived from multiple Wikipedia pages and a question.\n"
        "Answer ONLY with the minimal text span from the passage (no extra words).\n"
        "If it is yes/no, answer: yes or no. If unanswerable, write: unanswerable.\n"
        "Output ONLY the answer text.\n"
    ),
    "musique": (
        "You are given a long passage and a question requiring multi-step reasoning.\n"
        "Answer ONLY with a short span copied from the passage.\n"
        "If it is yes/no, answer: yes or no. If unanswerable, write: unanswerable.\n"
        "Output ONLY the answer text.\n"
    ),
}

# --------------------
# Data model
# --------------------

@dataclass
class Item:
    ex_id: str
    context: str
    question: str
    answers: List[str]

# --------------------
# Utilities
# --------------------

_ARTICLES = {"a", "an", "the"}
_PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    # remove punctuation
    s = re.sub(f"[{re.escape(_PUNCT)}]", " ", s)
    # remove articles
    s = " ".join(w for w in s.split() if w not in _ARTICLES)
    # squeeze spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _f1_em(pred: str, gold_list: List[str]) -> Tuple[float, float]:
    """Return (max_f1, max_em) over multiple gold strings."""
    pn = _normalize_text(pred)
    if pn == "" and any(_normalize_text(g) == "" for g in gold_list):
        return 1.0, 1.0
    best_f1 = 0.0
    best_em = 0.0
    for g in gold_list:
        gn = _normalize_text(g)
        if gn == pn:
            best_em = max(best_em, 1.0)
        # token F1
        ptoks = pn.split()
        gtoks = gn.split()
        if len(ptoks) == 0 and len(gtoks) == 0:
            f1 = 1.0
        elif len(ptoks) == 0 or len(gtoks) == 0:
            f1 = 0.0
        else:
            common = 0
            count_p = {}
            for t in ptoks:
                count_p[t] = count_p.get(t, 0) + 1
            count_g = {}
            for t in gtoks:
                count_g[t] = count_g.get(t, 0) + 1
            for t in count_p:
                if t in count_g:
                    common += min(count_p[t], count_g[t])
            if common == 0:
                f1 = 0.0
            else:
                prec = common / len(ptoks)
                rec = common / len(gtoks)
                f1 = 2 * prec * rec / (prec + rec)
        best_f1 = max(best_f1, f1)
    return best_f1, best_em

def _safe_tail_truncate(tokenizer, text: str, budget: int) -> str:
    """Keep the tail tokens within budget (preserve the question at the end)."""
    ids = tokenizer.encode(text)
    if len(ids) <= budget:
        return text
    keep = ids[-budget:]
    return tokenizer.decode(keep, skip_special_tokens=True)

def _build_prompt(dataset: str, item: Item) -> str:
    instr = INSTRUCTIONS[dataset]
    return f"{instr}\nDocument:\n{item.context}\n\nQuestion:\n{item.question}\n\nAnswer:"

# --------------------
# Data loading (no dataset scripts)
# --------------------

CANDIDATE_REPOS = [
    "THUDM/LongBench",        # official
    "zai-org/LongBench",      # data mirror
    "yanbingzheng/LongBench", # community mirror
    "Leooyii/longbench",      # another mirror
]

def _find_jsonl_in_zip(zf: zipfile.ZipFile, dataset: str) -> Optional[str]:
    """Heuristically find a JSONL file for `dataset` in the provided zip."""
    names = zf.namelist()
    # Prefer exact names
    exact = [n for n in names if n.endswith(f"/{dataset}.jsonl") or n.endswith(f"{dataset}.jsonl")]
    if exact:
        return exact[0]
    # Next, look for folders like dataset/test.jsonl
    test_like = [n for n in names if dataset in n and n.endswith(".jsonl") and ("test" in n or "dev" in n or "val" in n)]
    if test_like:
        return test_like[0]
    # Fallback: any jsonl containing dataset substring
    any_like = [n for n in names if dataset in n and n.endswith(".jsonl")]
    if any_like:
        return any_like[0]
    return None

def _load_items_from_jsonl_bytes(b: bytes, dataset: str) -> List[Item]:
    items: List[Item] = []
    for ln in b.decode("utf-8").splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        ex_id = str(obj.get("_id") or obj.get("id") or obj.get("example_id") or obj.get("task_id") or "")
        context = obj.get("context") or obj.get("passage") or obj.get("document") or ""
        question = obj.get("input") or obj.get("question") or ""
        answers = obj.get("answers") or obj.get("answer") or obj.get("outputs") or []
        if isinstance(answers, str):
            answers = [answers]
        # Some LongBench rows wrap answers as list of lists; flatten
        if answers and isinstance(answers[0], list):
            answers = [x for sub in answers for x in (sub if isinstance(sub, list) else [sub])]
        items.append(Item(ex_id=ex_id, context=context, question=question, answers=answers))
    return items

def _load_longbench(dataset: str, data_dir: Optional[str] = None) -> List[Item]:
    """Load LongBench without relying on dataset scripts."""
    # 1) Local dir provided?
    env_dir = os.getenv("LONGEBENCH_DATA_DIR") or os.getenv("LONGBENCH_DATA_DIR")
    base_dir = data_dir or env_dir
    if base_dir:
        # look for files like <base>/<dataset>.jsonl or <base>/<dataset>/test.jsonl
        candidates = [
            os.path.join(base_dir, f"{dataset}.jsonl"),
            os.path.join(base_dir, dataset, "test.jsonl"),
            os.path.join(base_dir, dataset, f"{dataset}.jsonl"),
            os.path.join(base_dir, f"{dataset}_test.jsonl"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                ds = load_dataset("json", data_files=p, split="train")
                return [Item(
                    ex_id=str(r.get("_id") or r.get("id") or ""),
                    context=r.get("context") or r.get("passage") or "",
                    question=r.get("input") or r.get("question") or "",
                    answers=(r.get("answers") or r.get("answer") or [])
                ) for r in ds]
        # As a fallback, scan all jsonl files under base_dir and pick the first that contains dataset name
        for root, _, files in os.walk(base_dir):
            for fn in files:
                if fn.endswith(".jsonl") and dataset in fn.lower():
                    p = os.path.join(root, fn)
                    ds = load_dataset("json", data_files=p, split="train")
                    return [Item(
                        ex_id=str(r.get("_id") or r.get("id") or ""),
                        context=r.get("context") or r.get("passage") or "",
                        question=r.get("input") or r.get("question") or "",
                        answers=(r.get("answers") or r.get("answer") or [])
                    ) for r in ds]

    # 2) Try to download a zip from HF datasets repos
    if hf_hub_download is not None:
        for repo in CANDIDATE_REPOS:
            try:
                zip_path = hf_hub_download(repo_id=repo, filename="data.zip", repo_type="dataset")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    inner = _find_jsonl_in_zip(zf, dataset)
                    if not inner:
                        continue
                    with zf.open(inner, "r") as f:
                        b = f.read()
                        items = _load_items_from_jsonl_bytes(b, dataset)
                        if items:
                            return items
            except Exception:
                continue

    # 3) As a last resort, instruct user to provide local data
    raise RuntimeError(
        "Failed to load LongBench data. Provide a local folder via `data_dir` or env "
        "LONGEBENCH_DATA_DIR with JSONL files (e.g., qasper.jsonl or qasper/test.jsonl). "
        "Alternatively, download `data.zip` from the THUDM/LongBench dataset page on Hugging Face "
        "and point to the extracted folder."
    )

# --------------------
# Runner
# --------------------

def run(model_path: str,
        results_dir: str,
        dataset: str = "qasper",
        num_samples: int = 100,
        batch_size: int = 4,
        max_new_tokens: int = 32,
        seed: int = 2025,
        tag: str = "",
        ctx_margin_tokens: int = 128,
        data_dir: Optional[str] = None) -> Dict:
    """Run LongBench <dataset> with vLLM.
    Primary metric: average token F1; also reports EM.
    """
    assert dataset in VALID_DATASETS, f"dataset must be one of {VALID_DATASETS}"
    random.seed(seed)

    # Load data (no scripts)
    pool = _load_longbench(dataset, data_dir=data_dir)
    random.shuffle(pool)
    eval_set = pool[:num_samples]

    # Initialize vLLM
    print(f"[vLLM][LongBench:{dataset}] Loading model: {model_path}")
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
        print(f"[vLLM][LongBench:{dataset}] Load done in {time.time() - t_load0:.1f}s")
    
        sampling = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=None,
        )
    
        tokenizer = llm.get_tokenizer()
        engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
        assert engine is not None, "Could not access llm_engine/engine from LLM"
        max_len = getattr(getattr(engine, "model_config", None), "max_model_len", None) or 32768
        # Reserve margin for generation and safety
        budget = max(256, max_len - ctx_margin_tokens - max_new_tokens)
    
        # Build prompts with tail-preserving truncation
        prompts = []
        gold_all = []
        for item in eval_set:
            raw_prompt = _build_prompt(dataset, item)
            ids = tokenizer.encode(raw_prompt)
            if len(ids) <= budget:
                prompts.append(raw_prompt)
            else:
                # Truncate ONLY the Document while preserving the question tail
                ctx_budget = max(64, budget // 2)  # allocate at least half to context
                ctx_trunc = _safe_tail_truncate(tokenizer, item.context, ctx_budget)
                patched = Item(ex_id=item.ex_id, context=ctx_trunc, question=item.question, answers=item.answers)
                prompts.append(_build_prompt(dataset, patched))
            gold_all.append(item.answers)
    
        # Reset KV meter
        run_id = f"longbench:{dataset}@{int(time.time())}"
        if tag:
            run_id += f"@{tag}"
        engine.reset_kv_meter(run_id=run_id)
    
        # Batched generation
        started_at = time.time()
        preds = []
        for i in range(0, len(prompts), batch_size):
            outs = llm.generate(prompts[i:i+batch_size], sampling)
            for out in outs:
                text = (out.outputs[0].text if out.outputs else "") or ""
                preds.append(text.strip())
    
        duration_sec = time.time() - started_at
    
        # Scoring
        f1s = []
        ems = []
        per_item = []
        for pid, (pred, golds, item) in enumerate(zip(preds, gold_all, eval_set)):
            f1, em = _f1_em(pred, golds)
            f1s.append(f1)
            ems.append(em)
            per_item.append({"id": item.ex_id, "f1": f1, "em": em, "pred": pred})
    
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        avg_em = sum(ems) / len(ems) if ems else 0.0
    
        kv_summary = engine.get_kv_meter_summary()
    
        result = {
            "bench": "longbench",
            "dataset": dataset,
            "model": model_path,
            "num_samples": len(eval_set),
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "started_at": started_at,
            "duration_sec": duration_sec,
            "score": avg_f1,
            "score_display": f"F1={avg_f1*100:.2f}%, EM={avg_em*100:.2f}%",
            "micro": {"f1": avg_f1, "em": avg_em},
            "kv_meter_summary": kv_summary,
            "run_id": run_id,
            "per_item": per_item,
        }
        print(f"[LongBench:{dataset}] {result['score_display']} | N={len(eval_set)} | time={duration_sec:.2f}s")
        return result
    finally:
        engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as e:
                print(f"[LongBench] llm.shutdown() error: {e}")
        del llm
        gc.collect()
        time.sleep(10)

