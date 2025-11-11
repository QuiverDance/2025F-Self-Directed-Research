#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_benchmark.py
- Orchestrator to run one or more benchmark suites on a vLLM-backed model.
- Results are written under ./results/<bench>.json (one file per bench run).
- Prints each bench's wall time and score, and includes KV meter summary.
"""

import os, sys, json, time, argparse, gc
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULT_RESULTS_DIR = HERE / "results"
SEED = 2025
ALL = 999999
# ------------------------------------------------------------------
# Global config used when --bench all
# You can tune per-benchmark parameters here at one place.
# ------------------------------------------------------------------
BENCH_ALL_CONFIG = {
    "mmlu":   {"num_samples": ALL, "kshot": 1, "batch_size": 128, "max_new_tokens": 2, "seed": SEED, "tag": ""},
    "gsm8k":  {"num_samples": 1319, "kshot": 1, "batch_size": 8, "max_new_tokens": 256, "seed": SEED, "tag": ""},
    "humaneval": {"num_samples": 164, "kshot": 0, "batch_size": 1, "max_new_tokens": 512, "seed": SEED, "tag": ""},
    "line_retrieval": {"num_samples": 200, "batch_size": 1, "max_new_tokens": 64,
                       "lr_num_lines": 2000, "lr_min_words": 5, "lr_max_words": 9, "lr_target_mode": "random",
                       "seed": SEED, "tag": ""},
    "longbench_qasper":  {"lb_dataset": "qasper",   "num_samples": 200, "batch_size": 1, "max_new_tokens": 32, "seed": SEED, "tag": "", "lb_data_dir": ""},
    "longbench_hotpotqa":{"lb_dataset": "hotpotqa", "num_samples": 200, "batch_size": 1, "max_new_tokens": 32, "seed": SEED, "tag": "", "lb_data_dir": ""},
    "longbench_2wikimqa":{"lb_dataset": "2wikimqa", "num_samples": 200, "batch_size": 1, "max_new_tokens": 32, "seed": SEED, "tag": "", "lb_data_dir": ""},
    "longbench_musique": {"lb_dataset": "musique",  "num_samples": 200, "batch_size": 1, "max_new_tokens": 32, "seed": SEED, "tag": "", "lb_data_dir": ""},
    "needle": {"num_samples": 100, "batch_size": 1, "max_new_tokens": 8,
               "nh_target_tokens": 12000, "nh_depth_mode": "random",
               "nh_vocab_mode": "random", "seed": SEED, "tag": ""},
}

def _print_agg_kv(summary: dict) -> None:
    agg = (summary or {}).get("aggregate", {})
    print("  [KV Aggregate]")
    print(f"    run_id                        : {summary.get('run_id')}")
    print(f"    num_requests                  : {summary.get('num_requests')}")
    print(f"    total_kv_capacity             : {summary.get('total_kv_capacity_readable')}")
    print(f"    - kv_usage_max                : {agg.get('kv_usage_max')}")
    print(f"    - kv_usage_avg_time_weighted  : {agg.get('kv_usage_avg_time_weighted_mean')}")
    print(f"    - kv_bytes_max                : {agg.get('kv_bytes_max_readable', 'n/a')}")
    print(f"    - kv_bytes_avg_time_weighted  : {agg.get('kv_bytes_avg_time_weighted_mean_readable', 'n/a')}")
    print(f"    - ttft_mean                   : {agg.get('ttft_mean_ms_readable', 'n/a')}")
    print(f"    - decode_tpt_mean             : {agg.get('decode_tpt_ms_mean_readable', 'n/a')}")
    print(f"    - decode_tps_mean             : {agg.get('decode_tps_mean_readable', 'n/a')}")

def run_mmlu(model: str, results_dir: Path, num_samples: int, kshot: int, batch_size: int, max_new_tokens:int, seed: int, tag: str):
    from benchmarks.mmlu import run as mmlu_run
    return mmlu_run(
        model_path=model,
        results_dir=str(results_dir),
        num_samples=num_samples,
        kshot=kshot,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        seed=seed,
        tag=tag,
    )

def run_gsm8k(model: str, results_dir: Path, num_samples: int, kshot: int, batch_size: int, max_new_tokens: int, seed: int, tag: str):
    from benchmarks.gsm8k import run as gsm8k_run
    return gsm8k_run(
        model_path=model,
        results_dir=str(results_dir),
        num_samples=num_samples,
        kshot=kshot,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        seed=seed,
        tag=tag,
    )

def run_humaneval(model: str, results_dir: Path, num_samples: int, batch_size: int, max_new_tokens: int, seed: int, tag: str):
    from benchmarks.humaneval import run as he_run
    return he_run(
        model_path=model,
        results_dir=str(results_dir),
        num_samples=num_samples,
        kshot=0,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        seed=seed,
        tag=tag,
    )

def run_line_retrieval(model: str, results_dir: Path,
                       num_samples: int, batch_size: int, max_new_tokens: int,
                       lr_num_lines: int, lr_min_words: int, lr_max_words: int, lr_target_mode: str,
                       seed: int, tag: str):
    from benchmarks.line_retrieval import run as lr_run
    return lr_run(
        model_path=model,
        results_dir=str(results_dir),
        num_samples=num_samples,
        num_lines=lr_num_lines, min_words=lr_min_words, max_words=lr_max_words, target_mode=lr_target_mode,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        seed=seed, tag=tag,
    )

def run_longbench(model: str, results_dir: Path, dataset: str,
                  num_samples: int, batch_size: int, max_new_tokens: int, seed: int, tag: str,
                  lb_data_dir: str = ""):
    from benchmarks.longbench import run as lb_run
    return lb_run(
        model_path=model,
        results_dir=str(results_dir),
        dataset=dataset,
        num_samples=num_samples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        seed=seed, tag=tag,
        data_dir=(lb_data_dir or None),
    )

def run_needle(model: str, results_dir: Path,
               num_samples: int, batch_size: int, max_new_tokens: int,
               nh_target_tokens: int, nh_depth: float, nh_depth_mode: str, nh_vocab_mode: str,
               seed: int, tag: str):
    from benchmarks.needle import run as needle_run
    return needle_run(
        model_path=model,
        results_dir=str(results_dir),
        num_samples=num_samples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        target_tokens=nh_target_tokens,
        depth=nh_depth, depth_mode=nh_depth_mode, vocab_mode=nh_vocab_mode,
        seed=seed, tag=tag,
    )

def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks orchestrator")
    parser.add_argument("--model", type=str, required=True, help="Path or HF id of the model to run with vLLM")
    parser.add_argument("--bench", type=str, default="mmlu", 
                        help="Which benchmark to run: mmlu | gsm8k | humaneval | line_retrieval | longbench | needle | all")
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Directory to write result JSONs (default: ./results)")
    parser.add_argument("--num_samples", type=int, default=100, help="Per-bench sample limit")
    parser.add_argument("--kshot", type=int, default=1, help="In-context examples per query")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens for generative benches (e.g., GSM8K)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag appended to run_id and filename")

    # Line Retrieval specific knobs (prefixed)
    parser.add_argument("--lr_num_lines", type=int, default=2000, help="(LineRetrieval) number of lines in the context")
    parser.add_argument("--lr_min_words", type=int, default=5, help="(LineRetrieval) min words per line")
    parser.add_argument("--lr_max_words", type=int, default=9, help="(LineRetrieval) max words per line")
    parser.add_argument("--lr_target_mode", type=str, default="random",
                        choices=["random", "head", "middle", "tail"],
                        help="(LineRetrieval) where the target line index is located")
    
    # LongBench specific: dataset choice for single-run
    parser.add_argument("--lb_dataset", type=str, default="qasper",
                        choices=["qasper", "hotpotqa", "2wikimqa", "musique"], help="(LongBench) which dataset to run")
    parser.add_argument("--lb_data_dir", type=str, default="",
                        help="(LongBench) local folder with JSONL (e.g., qasper.jsonl). If empty, try HF data.zip.")
    
    # Needle-in-a-Haystack specific knobs (prefixed)
    parser.add_argument("--nh_target_tokens", type=int, default=12000, help="(Needle) target document length in tokens")
    parser.add_argument("--nh_depth", type=float, default=0.5, help="(Needle) depth fraction (0..1) if depth_mode=fraction")
    parser.add_argument("--nh_depth_mode", type=str, default="fraction",
                        choices=["fraction", "head", "middle", "tail", "random"], help="(Needle) needle location mode")
    parser.add_argument("--nh_vocab_mode", type=str, default="random",
                        choices=["random", "common"], help="(Needle) filler vocabulary mode")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    def run_one(bench: str, cfg: dict):
        print(f"\n=== Running bench: {bench} ===")
        t0 = time.time()
        if bench == "mmlu":
            result = run_mmlu(
                model=args.model, results_dir=results_dir,
                num_samples=cfg.get("num_samples", args.num_samples),
                kshot=cfg.get("kshot", args.kshot),
                batch_size=cfg.get("batch_size", args.batch_size),
                max_new_tokens=cfg.get("max_new_tokens", args.max_new_tokens),
                seed=cfg.get("seed", args.seed),
                tag=cfg.get("tag", args.tag),
            )
        elif bench == "gsm8k":
            result = run_gsm8k(
                model=args.model, results_dir=results_dir,
                num_samples=cfg.get("num_samples", args.num_samples),
                kshot=cfg.get("kshot", args.kshot),
                batch_size=cfg.get("batch_size", args.batch_size),
                max_new_tokens=cfg.get("max_new_tokens", args.max_new_tokens),
                seed=cfg.get("seed", args.seed),
                tag=cfg.get("tag", args.tag),
            )
        elif bench == "humaneval":
            result = run_humaneval(
                model=args.model, results_dir=results_dir,
                num_samples=cfg.get("num_samples", args.num_samples),
                batch_size=cfg.get("batch_size", args.batch_size),
                max_new_tokens=cfg.get("max_new_tokens", args.max_new_tokens),
                seed=cfg.get("seed", args.seed),
                tag=cfg.get("tag", args.tag),
            )
        elif bench == "line_retrieval":
            result = run_line_retrieval(
                model=args.model, results_dir=results_dir,
                num_samples=cfg.get("num_samples", args.num_samples),
                batch_size=cfg.get("batch_size", args.batch_size),
                max_new_tokens=cfg.get("max_new_tokens", args.max_new_tokens),
                lr_num_lines=cfg.get("lr_num_lines", args.lr_num_lines),
                lr_min_words=cfg.get("lr_min_words", args.lr_min_words),
                lr_max_words=cfg.get("lr_max_words", args.lr_max_words),
                lr_target_mode=cfg.get("lr_target_mode", args.lr_target_mode),
                seed=cfg.get("seed", args.seed),
                tag=cfg.get("tag", args.tag),
            )
        elif bench == "longbench" or bench.startswith("longbench_"):
            # Determine dataset: prefer cfg.lb_dataset; otherwise parse suffix or use CLI arg
            ds = cfg.get("lb_dataset")
            if not ds and "_" in bench:
                ds = bench.split("_", 1)[1]
            if not ds:
                ds = args.lb_dataset
            result = run_longbench(
                model=args.model, results_dir=results_dir, dataset=ds,
                num_samples=cfg.get("num_samples", args.num_samples),
                batch_size=cfg.get("batch_size", args.batch_size),
                max_new_tokens=cfg.get("max_new_tokens", args.max_new_tokens),
                lb_data_dir=cfg.get("lb_data_dir", args.lb_data_dir),
                seed=cfg.get("seed", args.seed),
                tag=cfg.get("tag", args.tag),
            )
        elif bench == "needle":
            result = run_needle(
                model=args.model, results_dir=results_dir,
                num_samples=cfg.get("num_samples", args.num_samples),
                batch_size=cfg.get("batch_size", args.batch_size),
                max_new_tokens=cfg.get("max_new_tokens", args.max_new_tokens),
                nh_target_tokens=cfg.get("nh_target_tokens", args.nh_target_tokens),
                nh_depth=cfg.get("nh_depth", args.nh_depth),
                nh_depth_mode=cfg.get("nh_depth_mode", args.nh_depth_mode),
                nh_vocab_mode=cfg.get("nh_vocab_mode", args.nh_vocab_mode),
                seed=cfg.get("seed", args.seed),
                tag=cfg.get("tag", args.tag),
            )
        else:
            raise NotImplementedError(f"Bench '{bench}' not implemented.")
        dt = time.time() - t0
        score_str = result.get("score_display", "")
        print(f"[{bench}] done in {dt:.2f}s | score={score_str}")
        kv_summary = result.get("kv_meter_summary", {})
        _print_agg_kv(kv_summary)

        # Choose result filename:
        # - For 'longbench' single-run, include dataset suffix, e.g., longbench_qasper.json
        # - For 'longbench_*' presets, keep the bench name as-is (already specific)
        outfile = f"{bench}.json"
        if bench == "longbench":
            ds = (result or {}).get("dataset")
            if ds:
                outfile = f"longbench_{ds}.json"
        # (Other benches unchanged)
        out_path = results_dir / outfile

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[{bench}] results written to: {out_path}")
        
        gc.collect()
        time.sleep(30.0)

        return bench, {"score": result.get("score"), "score_display": result.get("score_display"),
                       "duration_sec": dt, "result_path": str(out_path)}

    overall = {}
    started = time.time()
    if args.bench == "all":
        for bname in BENCH_ALL_CONFIG:
            bench, meta = run_one(bname, BENCH_ALL_CONFIG[bname])
            overall[bench] = meta
    else:
        bench, meta = run_one(args.bench, {
            "num_samples": args.num_samples,
            "kshot": args.kshot,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "tag": args.tag,
        })
        overall[bench] = meta

    overall["total_wall_time_sec"] = time.time() - started
    print("\n=== Summary ===")
    for k, v in overall.items():
        if k == "total_wall_time_sec": continue
        print(f"- {k}: {v['score_display']} in {v['duration_sec']:.2f}s -> {v['result_path']}")
    print(f"Total wall time: {overall['total_wall_time_sec']:.2f}s")

if __name__ == "__main__":
    main()

