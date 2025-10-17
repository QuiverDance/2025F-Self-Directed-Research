# vllm/instrumentation/kv_metrics.py
# Unified request-scoped metrics for KV cache & latency.
# - Minimal overhead, one JSONL append per request at completion.
# - Two snapshots only: "prefill" (at first token) and "decode" (at request end).
# - Engine-global KV snapshots (GPU/CPU/Total) at those two moments.

from __future__ import annotations
import json
import os
import threading
import time
import atexit
import math
from statistics import median
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

def _now_ms_monotonic() -> float:
    """Monotonic milliseconds to avoid wall-clock jumps."""
    return time.monotonic_ns() / 1e6

def _fmt_bytes(n: Optional[int]) -> Optional[str]:
    if n is None:
        return None
    try:
        v = float(int(n))
    except Exception:
        return None
    units = ["B","KB","MB","GB","TB","PB"]
    i = 0
    while v >= 1024.0 and i < len(units)-1:
        v /= 1024.0
        i += 1
    return f"{v:.2f} {units[i]}"

@dataclass
class RequestLog:
    # Required idenfitifiers / meta
    request_id: str
    model: Optional[str] = None
    dtype: Optional[str] = None
    num_devices: Optional[int] = None
    block_size: Optional[int] = None

    # Optional run/bench meta (if provided via meta dict at enqueue)
    run_id: Optional[str] = None
    bench_id: Optional[str] = None
    case_id: Optional[str] = None
    trial: Optional[int] = None
    rank: Optional[int] = None
    is_warmup: Optional[bool] = None

    # Tokens counts
    context_len: int = 0
    generated_len: int = 0

    # Timing metrics (in ms)
    ttft_ms: Optional[float] = None  # Time to first token
    prefill_ms: Optional[float] = None  # Prefill time
    decode_ms: Optional[float] = None  # Decode time

    # Tokens per second
    tps_prefill: Optional[float] = None
    tps_decode: Optional[float] = None

    #  KV cache metrics
    kv_bytes_total_at_prefill: Optional[int] = None
    kv_bytes_gpu_at_prefill: Optional[int] = None
    kv_bytes_cpu_at_prefill: Optional[int] = None

    kv_bytes_total_at_decode: Optional[int] = None
    kv_bytes_gpu_at_decode: Optional[int] = None
    kv_bytes_cpu_at_decode: Optional[int] = None

    # Token-based KV size (for transparency; not used for scheduling)
    kv_token_bytes_est_at_prefill: Optional[int] = None
    kv_token_bytes_est_at_decode: Optional[int] = None

    kv_bytes_total_peak_alloc: int | None = None
    kv_bytes_gpu_peak_alloc:   int | None = None
    kv_bytes_cpu_peak_alloc:   int | None = None

class _SafeJsonWriter:
    """Thread-safe append-only JSONL writer (dir + filename)."""

    def __init__(self, log_dir: str, filename: str):

        self._dir = log_dir or "."
        os.makedirs(self._dir, exist_ok=True)

        # Do NOT auto-append rank suffix by default.
        # If you really want it, set VLLM_KV_METRICS_RANK_SUFFIX=on.
        base, ext = os.path.splitext(filename)
        add_rank = (os.getenv("VLLM_KV_METRICS_RANK_SUFFIX", "off").strip().lower()
                    in {"on", "true", "1", "yes"})
        if add_rank and ext == ".jsonl":
            rank = os.getenv("RANK") or os.getenv("LOCAL_RANK")
            if rank and not base.endswith(f".rank{rank}"):
                filename = f"{base}.rank{rank}{ext}"

        self._file = filename
        self._full_path = os.path.join(self._dir, self._file)

        # Touch the file
        open(self._full_path, "a", encoding="utf-8").close()
        self._lock = threading.Lock()
    
    @property
    def full_path(self) -> str:
        return self._full_path

    @property
    def directory(self) -> str:
        return self._dir

    @property
    def filename(self) -> str:
        return self._file

    def write(self, data: Dict[str, Any]):
        """Append one JSON line to the log file, and mirror to stdout."""
        line = json.dumps(data, ensure_ascii=False)
        with self._lock:
            with open(self._full_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        # mirror to stdout (toggle with VLLM_KV_METRICS_STDOUT=on/off)
        try:
            if (os.getenv("VLLM_KV_METRICS_STDOUT", "on").lower()
                    in {"on", "true", "1", "yes"}):
                print(line)
        except Exception:
            pass

class KVMetricsConfig:
    """Config driven by CLI/env.
    Enablement accepts: on/true/1/yes (case-insensitive)."""
    def __init__(self,
                 mode: Optional[str] = None,
                 out_dir: Optional[str] = None,
                 out_file: Optional[str] = None,
                 out_path: Optional[str] = None,
                 summary_file: Optional[str] = None):

        env_mode = (os.getenv("VLLM_KV_METRICS") or "").strip().lower()
        mode = (mode or "").strip().lower() or env_mode
        truthy = {"on", "true", "1", "yes"}
        self._enabled = mode in truthy

        # Backward-compat path (file path wins)
        env_path = os.getenv("VLLM_KV_METRICS_PATH")
        path = out_path or env_path  # may be None

        # New-style dir/file
        env_dir = os.getenv("VLLM_KV_METRICS_DIR")
        env_file = os.getenv("VLLM_KV_METRICS_FILE")

        if path:
            # Split path into dir + file
            d, f = os.path.split(path)
            self.out_dir = d or "."
            self.out_file = f or "kv_metrics.jsonl"
        else:
            self.out_dir = out_dir or env_dir or "./runs"
            self.out_file = out_file or env_file or "kv_metrics.jsonl"

        # Summary toggle (default: on) and filename
        self._summary_enabled = (os.getenv("VLLM_KV_METRICS_SUMMARY", "on")
                                 .strip().lower() in truthy)
        env_sum_file = os.getenv("VLLM_KV_METRICS_SUMMARY_FILE")
        self.summary_file = summary_file or env_sum_file or "summary.json"


    @property
    def enabled(self) -> bool:
        return bool(self._enabled)

    @property
    def summary_enabled(self) -> bool:
        return bool(self._summary_enabled)

class KVMetricsCollector:
    """
    Per-process singleton to collect request-scoped baseline metrics.

    Integration points:
      Engine / server entrypoints call:
        - link_engine(engine_like)
        - on_enqueue(request_id, context_len, meta)
        - on_first_token(request_id)
        - on_prefill_end(request_id)  # alias of on_first_token for clarity
        - snapshot_kv('prefill'|'decode', request_id)
        - count_generated_token(request_id, is_eos=False)
        - on_stream_end(request_id)
    """    

    _instance: Optional["KVMetricsCollector"] = None
    _init_lock = threading.Lock()
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for tests)."""
        with cls._init_lock:
            cls._instance = None

        inst = cls._get_or_create_singleton()
        try:
            agg = getattr(inst, "_agg", None) or getattr(inst, "aggregator", None)
            if agg is not None and hasattr(agg, "reset"):
                agg.reset()
        except Exception:
            pass

    @classmethod
    def get(cls, config: Optional[KVMetricsConfig] = None) -> "KVMetricsCollector":
        """Return singleton; if config differs, reconfigure the existing instance."""
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = KVMetricsCollector(config or KVMetricsConfig())
            else:
                if config is not None and cls._instance._config_differs_from(config):
                    cls._instance._reconfigure(config)
        return cls._instance

    def _config_differs_from(self, new_cfg: KVMetricsConfig) -> bool:
        old = self.cfg
        return (
            bool(old.enabled) != bool(new_cfg.enabled) or
            (getattr(old, "out_dir", None) != getattr(new_cfg, "out_dir", None)) or
            (getattr(old, "out_file", None) != getattr(new_cfg, "out_file", None)) or
            (getattr(old, "summary_enabled", None) != getattr(new_cfg, "summary_enabled", None)) or
            (getattr(old, "summary_file", None) != getattr(new_cfg, "summary_file", None))
        )

    def _reconfigure(self, new_cfg: KVMetricsConfig) -> None:
        """Rebuild writer/aggregator when env or args changed at runtime."""
        self.cfg = new_cfg
        self._writer = (_SafeJsonWriter(self.cfg.out_dir, self.cfg.out_file)
                        if self.cfg.enabled else None)
        # rebuild aggregator + (idempotent) atexit
        self._agg = _RunAggregator() if (self.cfg.enabled and self.cfg.summary_enabled) else None
        if self._agg is not None and not getattr(self, "_atexit_registered", False):
            atexit.register(self._write_summary_at_exit)
            self._atexit_registered = True

    def __init__(self, config: KVMetricsConfig):
        self.cfg = config
        self._writer = (_SafeJsonWriter(self.cfg.out_dir, self.cfg.out_file)
                        if self.cfg.enabled else None)
        self._engine_ref: Any = None
        self._req: Dict[str, RequestLog] = {}
        self._t_start: Dict[str, float] = {}
        self._t_first: Dict[str, float] = {}
        self._t_end: Dict[str, float] = {}
        self._finished: set[str] = set()  # guard double-flush
        self._lock = threading.Lock()
        # Track inflight requests so we can write summary immediately when all finish.
        self._inflight: set[str] = set()
        # per-request baseline of engine-global KV usage (bytes)
        # format: {request_id: {"total": int, "gpu": int, "cpu": int}}
        self._baseline: Dict[str, Dict[str, int]] = {}

        self._agg = _RunAggregator() if (self.cfg.enabled and self.cfg.summary_enabled) else None
        # register atexit once per process
        self._atexit_registered = False
        if self._agg is not None and not self._atexit_registered:
            atexit.register(self._write_summary_at_exit)
            self._atexit_registered = True

    # ---------- Engine linking & KV snapshot ----------
    def link_engine(self, engine_like: Any) -> None:
        """Keep a weak-ish reference to engine internals for KV snapshots."""
        self._engine_ref = engine_like

    def _kv_bytes_snapshot(self) -> Dict[str, int]:
        """Compute KV bytes from (a) direct counters or (b) used_blocks*block_bytes.
        Always initialize locals to avoid UnboundLocalError.
        """
        eng = getattr(self, "_engine_ref", None)
        if eng is None:
            _kvdbg("snapshot", "no engine_ref")
            return {"total": 0, "gpu": 0, "cpu": 0}

        # mgr = _locate_kv_manager(eng)
        _, _, mgr = _get_core_scheduler_manager(eng)
        if mgr is None:
            _kvdbg("snapshot", "no manager")
            return {"total": 0, "gpu": 0, "cpu": 0}

        # 1) Direct byte counters (best possible, if exposed by the manager)
        gpu_b = _get_attr_any(mgr, ["gpu_bytes", "gpu_used_bytes", "gpu_allocated_bytes"]) if mgr else None
        cpu_b = _get_attr_any(mgr, ["cpu_bytes", "cpu_used_bytes", "host_bytes", "offload_bytes"]) if mgr else None
        if isinstance(gpu_b, int) or isinstance(cpu_b, int):
            g = int(gpu_b or 0); c = int(cpu_b or 0)
            out = {"total": g + c, "gpu": g, "cpu": c}
            _kvdbg("snapshot.direct", out)
            return {"total": g + c, "gpu": g, "cpu": c}

        # 2) Fallback: used_blocks × bytes_per_block (works with BlockPool)
        block_bytes = _infer_block_bytes(eng)
        _kvdbg("snapshot.block_bytes_raw", block_bytes)
        if block_bytes <= 0 or mgr is None:
            return {"total": 0, "gpu": 0, "cpu": 0}

        g_blocks = _get_used_blocks(mgr, "gpu")
        c_blocks = _get_used_blocks(mgr, "cpu")
        g = int(g_blocks) * int(block_bytes)
        c = int(c_blocks) * int(block_bytes)

        _kvdbg("snapshot.blocks", {"g_blocks": g_blocks, "c_blocks": c_blocks})
        totals = {"total": g + c, "gpu": g, "cpu": c}
        _kvdbg("snapshot.bytes", totals)

        return {"total": g + c, "gpu": g, "cpu": c}

    def snapshot_kv(self, phase: str, request_id: str) -> None:
        """Store engine-global KV snapshot for the given phase ('prefill'|'decode')."""
        if not self.cfg.enabled:
            return

        snap = self._kv_bytes_snapshot()
        cur_total = int(snap.get("total", 0))
        cur_gpu   = int(snap.get("gpu",   0))
        cur_cpu   = int(snap.get("cpu",   0))
        base = self._baseline.get(request_id, {"total": 0, "gpu": 0, "cpu": 0})
        # per-request incremental usage (clamped >= 0)
        total = max(0, cur_total - int(base.get("total", 0)))
        gpu   = max(0, cur_gpu   - int(base.get("gpu",   0)))
        cpu   = max(0, cur_cpu   - int(base.get("cpu",   0)))

        key_total = f"kv_bytes_total_at_{phase}"
        key_gpu   = f"kv_bytes_gpu_at_{phase}"
        key_cpu   = f"kv_bytes_cpu_at_{phase}"

        with self._lock:
            rec = self._req.get(request_id)
            if not rec:
                _kvdbg("snapshot_kv.missing_rec", {"rid": request_id})
                return
            # --- token-based estimation using known token counts in rec ---
            ctx = int(getattr(rec, "context_len", 0) or 0)
            gen = int(getattr(rec, "generated_len", 0) or 0)
            _, est_prefill, est_decode = _est_bytes_from_tokens(self._engine_ref, ctx, gen)
            est_token_bytes = est_prefill if phase == "prefill" else est_decode

            try:
                blk_bytes = _infer_block_bytes(self._engine_ref)
                _kvdbg("snapshot.block_bytes", f"{blk_bytes} (token≈{int(est_token_bytes)})")
            except Exception:
                pass

            # Write into the record (dict-first; fallback to attribute if not a dict)
            if isinstance(rec, dict):
                rec[key_total] = total
                rec[key_gpu]   = gpu
                rec[key_cpu]   = cpu
                rec[f"kv_token_bytes_est_at_{phase}"] = int(est_token_bytes)
            else:
                setattr(rec, key_total, total)
                setattr(rec, key_gpu,   gpu)
                setattr(rec, key_cpu,   cpu)
                setattr(rec, f"kv_token_bytes_est_at_{phase}", int(est_token_bytes))

            # Maintain peak for summary
            try:
                prev_peak = int(self._agg.get("peak_kv_bytes_total", 0))
                if total > prev_peak:
                    self._agg["peak_kv_bytes_total"] = total
            except Exception:
                pass

            _kvdbg("snapshot_kv.write", {
                "rid": request_id, "phase": phase,
                key_total: total, key_gpu: gpu, key_cpu: cpu,
                f"kv_token_bytes_est_at_{phase}": int(est_token_bytes),
            })

    
    def on_enqueue(self, request_id: str, context_len: int, meta: Optional[Dict[str, Any]] = None) -> None:
        """Create per-request record with context length and static meta."""
        if not self.cfg.enabled:
            return
        now = _now_ms_monotonic()
        with self._lock:
            if request_id in self._req:
                self._req[request_id].context_len = max(self._req[request_id].context_len, int(context_len or 0))
            else:
                base = {"request_id": request_id, "context_len": int(context_len or 0)}
                if meta:
                    for k in ("model", "dtype", "num_devices", "block_size",
                              "run_id", "bench_id", "case_id", "trial", "rank", "is_warmup"):
                        if k in meta:
                            base[k] = meta[k]
                self._req[request_id] = RequestLog(**base)
            self._t_start[request_id] = now
            # capture engine-global KV usage as baseline for this request
            try:
                snap0 = self._kv_bytes_snapshot()  # {"total","gpu","cpu"}
                self._baseline[request_id] = {
                    "total": int(snap0.get("total", 0) or 0),
                    "gpu":   int(snap0.get("gpu",   0) or 0),
                    "cpu":   int(snap0.get("cpu",   0) or 0),
                }
            except Exception:
                print("[KVCHK] baseline snapshot failed for", request_id)
                self._baseline[request_id] = {"total": 0, "gpu": 0, "cpu": 0}
            self._inflight.add(request_id)
    
    def on_first_token(self, request_id: str) -> None:
        """Mark first token emission (TTFT & prefill_ms)."""
        if not self.cfg.enabled:
            return
        now = _now_ms_monotonic()
        with self._lock:
            if request_id in self._t_first:
                return  # idempotent
            self._t_first[request_id] = now
            t0 = self._t_start.get(request_id)
            rec = self._req.get(request_id)
            if rec and t0 is not None:
                dt = max(0.0, now - t0)
                rec.ttft_ms = dt
                rec.prefill_ms = dt  # unified: prefill ends at first token

    def on_prefill_end(self, request_id: str) -> None:
        """Alias kept for clarity with engine hooks."""
        self.on_first_token(request_id)

    def on_stream_end(self, request_id: str) -> None:
        # print("[KVCHK-WRITE] writer?", bool(self._writer), "summary?", self.cfg.summary_enabled)
        """Finalize request record, compute decode_ms/TPS, flush JSONL once."""
        if not self.cfg.enabled:
            return
        now = _now_ms_monotonic()
        with self._lock:
            if request_id in self._finished:
                return  # double-flush guard
            self._t_end[request_id] = now
            t_first = self._t_first.get(request_id)
            rec = self._req.get(request_id)
            if rec and t_first is not None:
                rec.decode_ms = max(0.0, now - t_first)
                if rec.prefill_ms and rec.prefill_ms >= 1.0:
                    rec.tps_prefill = rec.context_len / (rec.prefill_ms / 1000.0)
                else:
                    rec.tps_prefill = None
                if rec.decode_ms and rec.decode_ms >= 1.0:
                    rec.tps_decode = rec.generated_len / (rec.decode_ms / 1000.0)
                else:
                    rec.tps_decode = None
            if self._writer and rec:
                if self._agg is not None:
                    self._agg.observe_request(rec)
                payload = asdict(rec)
                payload["ts"] = int(time.time() * 1000)  # ordering aid  
                # Add human-readable fields alongside byte counts
                for k, v in list(payload.items()):
                    if (isinstance(v, int) and
                        (k.startswith("kv_bytes_") or k.startswith("kv_token_bytes_est_") or k.endswith("_peak_alloc"))):
                        hv = _fmt_bytes(v)
                        if hv is not None:
                            payload[k + "_human"] = hv

                # print("KV metrics.on_stream_end writing", payload)
                self._writer.write(payload)

            self._finished.add(request_id)
            # drop baseline after finishing this request
            try:
                self._baseline.pop(request_id, None)
            except Exception:
                pass
            self._inflight.discard(request_id)
            # If no more inflight, write summary right now (no need to wait for atexit).
            if self._agg is not None and self.cfg.summary_enabled and not self._inflight:
                try:
                    self._write_summary_at_exit()
                except Exception:
                    pass
            for d in (self._req, self._t_start, self._t_first, self._t_end):
                d.pop(request_id, None)
            print("[KVCHK-WRITE] wrote jsonl line for", request_id)

    def count_generated_token(self, request_id: str, is_eos: bool = False) -> None:
        """Increment generated_len for non-EOS tokens."""
        if not self.cfg.enabled:
            return
        if is_eos:
            return
        with self._lock:
            rec = self._req.get(request_id)
            if rec:
                rec.generated_len += 1

    def bump_peak_alloc(self, request_id: str) -> None:
        """Update per-request peak of block-based allocated bytes (GPU/CPU/Total)."""
        if not self.cfg.enabled:
            return
        snap = self._kv_bytes_snapshot()
        cur_total = int(snap.get("total", 0))
        cur_gpu   = int(snap.get("gpu",   0))
        cur_cpu   = int(snap.get("cpu",   0))
        base = self._baseline.get(str(request_id), {"total": 0, "gpu": 0, "cpu": 0})
        total = max(0, cur_total - int(base.get("total", 0)))
        gpu   = max(0, cur_gpu   - int(base.get("gpu",   0)))
        cpu   = max(0, cur_cpu   - int(base.get("cpu",   0)))

        with self._lock:
            rec = self._req.get(str(request_id))
            if rec is None:
                return
            rec.kv_bytes_total_peak_alloc = max(int(rec.kv_bytes_total_peak_alloc or 0), total)
            rec.kv_bytes_gpu_peak_alloc   = max(int(rec.kv_bytes_gpu_peak_alloc or 0), gpu)
            rec.kv_bytes_cpu_peak_alloc   = max(int(rec.kv_bytes_cpu_peak_alloc or 0), cpu)
        _kvdbg("peak.bump", {"rid": str(request_id), "total": total, "gpu": gpu, "cpu": cpu})

    def _summary_path(self) -> str:
        """Return path to the summary file in the configured directory."""
        if self._writer is not None:
            base_dir = self._writer.directory
        else:
            base_dir = self.cfg.out_dir or "."
        return os.path.join(base_dir, self.cfg.summary_file)

    def _write_summary_at_exit(self) -> None:
        if not self.cfg.summary_enabled or self._agg is None:
            return
        try:
            payload = self._agg.build_summary()
            spath = self._summary_path()
            d = os.path.dirname(spath)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(spath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            # mirror to stdout
            try:
                if (os.getenv("VLLM_KV_METRICS_STDOUT", "on").lower()
                        in {"on", "true", "1", "yes"}):
                    print(json.dumps(payload, ensure_ascii=False))
            except Exception:
                pass
        except Exception:
            pass

class _RunAggregator:
    """Lightweight in-process aggregator for summary.json."""
    
    def __init__(self):
        self.total_context = 0
        self.total_prefill_ms = 0.0
        self.total_generated = 0
        self.total_decode_ms = 0.0
        self.ttft_list = []  # for p50/p90
        self.peak_kv_total = 0
        # optional counts
        self._num_requests = 0
        self._req_stats = []

    def reset(self) -> None:
        self.total_context = 0
        self.total_prefill_ms = 0.0
        self.total_generated = 0
        self.total_decode_ms = 0.0
        self.ttft_list = []
        self.peak_kv_total = 0
        self._num_requests = 0
        self._req_stats = []

    def observe_request(self, rec: "RequestLog"):
        self._num_requests += 1
        self.total_context += int(rec.context_len or 0)
        self.total_generated += int(rec.generated_len or 0)
        if rec.prefill_ms is not None:
            self.total_prefill_ms += float(rec.prefill_ms)
        if rec.decode_ms is not None:
            self.total_decode_ms += float(rec.decode_ms)
        if rec.ttft_ms is not None:
            self.ttft_list.append(float(rec.ttft_ms))
        # peak KV
        peak = int(getattr(rec, "kv_bytes_total_peak_alloc", 0) or 0)
        if peak > self.peak_kv_total:
            self.peak_kv_total = peak

        self._req_stats.append({
            "context_tokens": int(rec.context_len or 0),
            "generated_tokens": int(rec.generated_len or 0),
            "prefill_ms": float(rec.prefill_ms) if rec.prefill_ms is not None else None,
            "decode_ms": float(rec.decode_ms) if rec.decode_ms is not None else None,
            "ttft_ms": float(rec.ttft_ms) if rec.ttft_ms is not None else None,
            "kv_peak_bytes": peak,
        })

    def build_summary(self) -> dict:

        def _pct(arr, q):
            if not arr:
                return None
            arr = sorted(arr)
            k = (len(arr) - 1) * q
            f = math.floor(k); c = math.ceil(k)
            if f == c:
                return arr[int(k)]
            return arr[f] * (c - k) + arr[c] * (k - f)

        def _valid(nums):
            out = []
            for x in nums:
                if x is None:
                    continue
                if isinstance(x, (int, float)):
                    if math.isfinite(float(x)):
                        out.append(float(x))
            return out

        def _mean(x):
            return (sum(x) / len(x)) if x else None

        # token-weighted TPS
        tps_prefill = (self.total_context / (self.total_prefill_ms / 1000.0)
                       if self.total_context > 0 and self.total_prefill_ms >= 1.0 else None)
        tps_decode = (self.total_generated / (self.total_decode_ms / 1000.0)
                      if self.total_generated > 0 and self.total_decode_ms >= 1.0 else None)
        
        ttft_list = list(getattr(self, "ttft_list", []) or [])
        ttft_p50 = _pct(ttft_list, 0.50) if ttft_list else None
        ttft_p90 = _pct(ttft_list, 0.90) if ttft_list else None
        
        reqs = [r for r in getattr(self, "_req_stats", []) if isinstance(r, dict)]
        prefill_tps_per_req, decode_tps_per_req, latency_ms = [], [], []
        ctx_list = _valid([r.get("context_tokens") for r in reqs])
        gen_list = _valid([r.get("generated_tokens") for r in reqs])
        kvp_list = _valid([r.get("kv_peak_bytes") for r in reqs])
        ttft_req = _valid([r.get("ttft_ms") for r in reqs])

        for r in reqs:
            pre_ms, dec_ms = r.get("prefill_ms"), r.get("decode_ms")
            ctok, gtok = r.get("context_tokens"), r.get("generated_tokens")
            if pre_ms and pre_ms > 0 and ctok:
                prefill_tps_per_req.append(ctok / (pre_ms / 1000.0))
            if dec_ms and dec_ms > 0 and gtok:
                decode_tps_per_req.append(gtok / (dec_ms / 1000.0))
            if pre_ms is not None or dec_ms is not None:
                a = float(pre_ms or 0.0) + float(dec_ms or 0.0)
                latency_ms.append(a)

        out = {
            "num_requests": getattr(self, "_num_requests", len(reqs)),
            "tps_prefill_token_weighted": tps_prefill,
            "tps_decode_token_weighted": tps_decode,
            "ttft_ms_p50": ttft_p50,
            "ttft_ms_p90": ttft_p90,
            "peak_kv_bytes_total": getattr(self, "peak_kv_total", None),

            "avg_prefill_tps": _mean(prefill_tps_per_req),
            "avg_decode_tps": _mean(decode_tps_per_req),

            "avg_ttft_ms_per_req": _mean(ttft_req),
            "p50_ttft_ms_per_req": _pct(ttft_req, 0.50) if ttft_req else None,
            "p90_ttft_ms_per_req": _pct(ttft_req, 0.90) if ttft_req else None,

            "avg_latency_ms_per_req": _mean(latency_ms),
            "p50_latency_ms_per_req": _pct(latency_ms, 0.50) if latency_ms else None,
            "p90_latency_ms_per_req": _pct(latency_ms, 0.90) if latency_ms else None,

            "avg_context_tokens_per_req": _mean(ctx_list),
            "avg_generated_tokens_per_req": _mean(gen_list),

            "avg_kv_peak_bytes_per_req": _mean(kvp_list),
            "max_kv_peak_bytes_per_req": (max(kvp_list) if kvp_list else None),
        }

        for _k in ("peak_kv_bytes_total", "avg_kv_peak_bytes_per_req", "max_kv_peak_bytes_per_req"):
            _v = out.get(_k)
            try:
                if _v is not None:
                    out[_k + "_human"] = _fmt_bytes(int(float(_v)))
            except Exception:
                pass

        return out


def _get_attr_any(obj: Any, names: list[str]):
    """Return first existing attribute or callable() result among names."""
    for n in names:
        if not hasattr(obj, n):
            continue
        v = getattr(obj, n)
        if callable(v):
            try:
                v = v()
            except Exception:
                continue
        return v
    return None

def _unwrap_engine_layers(eng: Any) -> list[Any]:
    """Yield plausible containers that might carry the KV manager."""
    if eng is None:
        return []
    layers = [eng]
    # common wrappers in v1
    for n in ("engine_core", "engine", "core"):
        x = getattr(eng, n, None)
        if x is not None:
            layers.append(x)
            # one more hop (e.g., engine_core.engine_core)
            y = getattr(x, n, None)
            if y is not None:
                layers.append(y)
    return layers

def _locate_kv_manager(eng: Any) -> Any:
    """Try multiple paths to find a cache/block manager object."""
    layers = [
        eng,
        getattr(eng, "engine_core", None),
        getattr(eng, "core", None),
        getattr(eng, "engine", None),
    ]
    layers = [x for x in layers if x is not None]
    for layer in layers:
        # Try under scheduler first
        sched = getattr(layer, "scheduler", None)
        if sched is not None:
            for name in ("kv_cache_manager", "block_manager", "cache_manager"):
                m = getattr(sched, name, None)
                if m is not None:
                    return m
        # Then try layer itself
        for name in ("kv_cache_manager", "block_manager", "cache_manager"):
            m = getattr(layer, name, None)
            if m is not None:
                return m
    return None

def _get_core_scheduler_manager(eng):
    """Return (core, scheduler, kv_cache_manager) using vLLM v1 fixed path."""
    # Normalize engine core (v1 stacks can be eng.engine_core.engine_core)
    core = getattr(eng, "engine_core", eng)
    # unwrap multiple layers
    while core is not None and hasattr(core, "engine_core"):
        nxt = getattr(core, "engine_core")
        # Break potential self-cycles defensively
        if nxt is core:
            break
        core = nxt

    sched = getattr(core, "scheduler", None)
    if sched is None and hasattr(core, "model_executor"):
        sched = getattr(core.model_executor, "scheduler", None)
    
    mgr = None
    if sched is not None:
        for name in ("kv_cache_manager", "block_manager", "cache_manager"):
            mgr = getattr(sched, name, None)
            if mgr is not None:
                break
    if mgr is None and core is not None:
        for name in ("kv_cache_manager", "block_manager", "cache_manager"):
            mgr = getattr(core, name, None)
            if mgr is not None:
                break

    # for debugging
    _kvdbg("resolve.core", type(core).__name__ if core else None)
    _kvdbg("resolve.sched", type(sched).__name__ if sched else None)
    _kvdbg("resolve.mgr", type(mgr).__name__ if mgr else None)

    return core, sched, mgr

def _dtype_bytes(dtype_str: str | None) -> int:
    if not dtype_str:
        return 0
    s = dtype_str.lower()
    if s in ("fp16", "float16", "half"):
        return 2
    if s in ("bf16", "bfloat16"):
        return 2
    if s in ("fp32", "float32", "float"):
        return 4
    if s in ("fp8", "e4m3", "e5m2"):
        return 1
    return 0

def _est_bytes_from_tokens(eng: Any, context_len: int, generated_len: int) -> tuple[int, int, int]:
    """Return (block_size, est_prefill_bytes, est_decode_bytes) using ceil-div block math."""
    block_bytes = _infer_block_bytes(eng)
    vcfg = getattr(eng, "vllm_config", None)
    cc   = getattr(vcfg, "cache_config", None) if vcfg else None
    block_size = getattr(cc, "block_size", 16) if cc else 16
    # ceil division
    def cdiv(a, b): return (int(a) + int(b) - 1) // int(b)
    pre_blocks = cdiv(int(context_len), int(block_size))
    dec_blocks = cdiv(int(context_len) + int(generated_len), int(block_size))
    return int(block_size), pre_blocks * block_bytes, dec_blocks * block_bytes

def _infer_block_bytes(eng: Any) -> int:
    """Compute bytes per KV block from model/cache config."""
    core = getattr(eng, "engine_core", eng)
    vcfg = getattr(core, "vllm_config", None)
    mconf = getattr(vcfg, "model_config", None) if vcfg else None
    hf   = getattr(mconf, "hf_config", None)   if mconf else None
    cc   = getattr(vcfg, "cache_config", None) if vcfg else None

    # --- block size ---
    block_size = getattr(cc, "block_size", None)
    try:
        block_size = int(block_size) if block_size is not None else 16
    except Exception:
        block_size = 16

    # --- model params ---
    def g(o, *keys):
        for k in keys:
            if o is not None and hasattr(o, k):
                return getattr(o, k)
        return None
    num_layers = g(hf, "num_hidden_layers", "n_layer") or g(mconf, "num_hidden_layers") or 0
    n_heads    = g(hf, "num_attention_heads", "n_head") or g(mconf, "num_attention_heads") or 0
    n_kv       = g(hf, "num_key_value_heads", "n_kv_heads") or g(mconf, "num_key_value_heads") or 0
    head_dim   = g(hf, "head_dim")
    if head_dim is None:
        hidden_size = g(hf, "hidden_size", "n_embd") or g(mconf, "hidden_size")
        if hidden_size and n_heads:
            head_dim = int(hidden_size) // int(n_heads)
    head_dim = int(head_dim or 0)

    # --- dtype -> bytes per element ---
    kv_dtype = g(cc, "kv_cache_dtype") or g(cc, "dtype")
    bpe = 2  # default fp16/bf16
    if kv_dtype:
        s = str(kv_dtype).lower()
        if "32" in s: bpe = 4
        elif "8" in s or "fp8" in s or "e4m3" in s or "e5m2" in s or "int8" in s: bpe = 1

    bytes_per_token = int(num_layers) * int(n_kv) * head_dim * 2 * int(bpe)
    block_bytes = int(block_size) * bytes_per_token

    # for debugging
    _kvdbg("block_bytes.inputs", {
        "block_size": block_size,
        "num_layers": int(num_layers),
        "n_kv_heads": int(n_kv),
        "n_heads": int(n_heads),
        "head_dim": int(head_dim),
        "dtype_bytes": int(bpe),
    })
    return int(block_bytes)

def _get_used_blocks(mgr: Any, device: str) -> int:
    """Return number of used KV blocks for the given device.
    Supports vLLM v1 BlockPool: GPU used = num_gpu_blocks - get_num_free_blocks().
    """
    dev = (device or "gpu").lower()
    bp = getattr(mgr, "block_pool", None) or getattr(mgr, "pool", None)

    # for debugging
    if bp is None:
        _kvdbg(f"used_blocks[{dev}]", "no block_pool")

    if dev == "gpu" and bp is not None:
        total = getattr(bp, "num_gpu_blocks", None)
        get_free = getattr(bp, "get_num_free_blocks", None)
        if isinstance(total, int) and callable(get_free):
            try:
                free = int(get_free())
                used = max(0, int(total) - free)
                _kvdbg("used_blocks[gpu]", {"total": int(total), "free": free, "used": used})
                return max(0, int(total) - free)
            except Exception as e:
                _kvdbg("used_blocks[gpu].err", str(e))
                return 0

    if dev == "cpu" and bp is not None:
        total = getattr(bp, "num_cpu_blocks", None) or getattr(bp, "cpu_num_blocks", None)
        get_free = getattr(bp, "get_num_cpu_free_blocks", None)
        if isinstance(total, int) and callable(get_free):
            try:
                free = int(get_free())
                return max(0, int(total) - free)
            except Exception:
                return 0
        return 0

    _kvdbg(f"used_blocks[{dev}]", "unsupported device")
    return 0



def _kvdbg(tag: str, payload=None):
    return
    if payload is None:
        print(f"[KVDBG] {tag}", flush=True)
    else:
        try:
            # compact JSON for structures
            if isinstance(payload, (dict, list, tuple)):
                s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                print(f"[KVDBG] {tag} {s}", flush=True)
            else:
                print(f"[KVDBG] {tag} {payload}", flush=True)
        except Exception:
            print(f"[KVDBG] {tag} {payload}", flush=True)