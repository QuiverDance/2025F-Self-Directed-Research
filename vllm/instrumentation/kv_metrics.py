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
        """Read global KV usage (bytes) from engine-like object."""

        eng = self._engine_ref
        if eng is None:
            return {"total": 0, "gpu": 0, "cpu": 0}
        
        mgr = _locate_kv_manager(eng)
        if mgr is not None:
            gpu_b = _get_attr_any(mgr, ["gpu_bytes", "gpu_used_bytes", "gpu_allocated_bytes"])
            cpu_b = _get_attr_any(mgr, ["cpu_bytes", "cpu_used_bytes", "host_bytes", "offload_bytes"])
        
        # If byte counters are available, use them directly.
        if isinstance(gpu_b, int) or isinstance(cpu_b, int):
            g = int(gpu_b or 0); c = int(cpu_b or 0)
            return {"total": g + c, "gpu": g, "cpu": c}

        # Otherwise compute bytes = used_blocks * bytes_per_block
        block_bytes = _infer_block_bytes(eng)
        if block_bytes <= 0 or mgr is None:
            return {"total": 0, "gpu": 0, "cpu": 0}

        gpu_used = _get_used_blocks(mgr, "gpu")
        cpu_used = _get_used_blocks(mgr, "cpu")

        g = int(gpu_used) * block_bytes
        c = int(cpu_used) * block_bytes
        return {"total": g + c, "gpu": g, "cpu": c}

    def snapshot_kv(self, phase: str, request_id: str) -> None:
        """Store engine-global KV snapshot for the given phase ('prefill'|'decode')."""
        if not self.cfg.enabled:
            return
        snap = self._kv_bytes_snapshot()
        with self._lock:
            rec = self._req.get(request_id)
            if not rec:
                return
            if phase == "prefill":
                rec.kv_bytes_total_at_prefill = snap["total"]
                rec.kv_bytes_gpu_at_prefill = snap["gpu"]
                rec.kv_bytes_cpu_at_prefill = snap["cpu"]
            elif phase == "decode":
                rec.kv_bytes_total_at_decode = snap["total"]
                rec.kv_bytes_gpu_at_decode = snap["gpu"]
                rec.kv_bytes_cpu_at_decode = snap["cpu"]

    
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
        print("KV metrics.on_stream_end start", request_id)
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
                print("KV metrics.on_stream_end writing", payload)
                self._writer.write(payload)

            self._finished.add(request_id)
            self._inflight.discard(request_id)
            # If no more inflight, write summary right now (no need to wait for atexit).
            if self._agg is not None and self.cfg.summary_enabled and not self._inflight:
                try:
                    self._write_summary_at_exit()
                except Exception:
                    pass
            for d in (self._req, self._t_start, self._t_first, self._t_end):
                d.pop(request_id, None)
            print("KV metrics.on_stream_end done", request_id)

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
            # ensure directory exists
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
            # never raise on exit
            pass

    def write_summary_now(self) -> Optional[str]:
        """Manually write summary.json immediately. Returns path if written."""
        if not self.cfg.summary_enabled or self._agg is None:
            return None
        try:
            payload = self._agg.build_summary()
            spath = self._summary_path()
            d = os.path.dirname(spath)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(spath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return spath
        except Exception:
            return None

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
        for v in (rec.kv_bytes_total_at_prefill, rec.kv_bytes_total_at_decode):
            if isinstance(v, int) and v > self.peak_kv_total:
                self.peak_kv_total = v

    def build_summary(self) -> dict:
        # token-weighted TPS
        tps_prefill = (self.total_context / (self.total_prefill_ms / 1000.0)
                       if self.total_context > 0 and self.total_prefill_ms >= 1.0 else None)
        tps_decode = (self.total_generated / (self.total_decode_ms / 1000.0)
                      if self.total_generated > 0 and self.total_decode_ms >= 1.0 else None)
        ttft_p50 = median(self.ttft_list) if self.ttft_list else None
        # p90 (simple, robust even for small N)
        ttft_p90 = None
        if self.ttft_list:
            arr = sorted(self.ttft_list)
            idx = max(0, int(round(0.90 * (len(arr) - 1))))
            ttft_p90 = arr[idx]
        return {
            "num_requests": self._num_requests,
            "tps_prefill_token_weighted": tps_prefill,
            "tps_decode_token_weighted": tps_decode,
            "ttft_ms_p50": ttft_p50,
            "ttft_ms_p90": ttft_p90,
            "peak_kv_bytes_total": self.peak_kv_total,
        }


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

    try:
        core = (getattr(eng, "engine_core", None)
                or getattr(eng, "core", None)
                or getattr(eng, "engine", None))
        if core is not None:
            sched = getattr(core, "scheduler", None)
            if sched is not None:
                m = getattr(sched, "kv_cache_manager", None)
                if m is not None:
                    return m
    except Exception:
        pass
    return None

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

def _infer_block_bytes(eng: Any) -> int:
    """Compute bytes per KV block from model/cache config."""
    # block_size
    bs = _get_attr_any(eng, ["cache_config", "vllm_config"])
    block_size = None
    if bs is not None and hasattr(bs, "block_size"):
        block_size = getattr(bs, "block_size")
    if block_size is None:
        block_size = _get_attr_any(eng, ["block_size", "kv_block_size", "cache_block_size"])
    try:
        block_size = int(block_size) if block_size is not None else 16
    except Exception:
        block_size = 16

    # model config / hf config (layers, heads, head_dim)
    mc = _get_attr_any(eng, ["model_config"]) or _get_attr_any(eng, ["vllm_config"])
    hf = None
    for name in ("hf_config", "config"):
        if mc is not None and hasattr(mc, name):
            hf = getattr(mc, name)
            break

    num_layers = _get_attr_any(hf, ["num_hidden_layers", "n_layer"]) or \
                 _get_attr_any(mc, ["num_hidden_layers"])
    num_layers = int(num_layers) if num_layers is not None else 0

    n_kv_heads = _get_attr_any(hf, ["num_key_value_heads", "n_kv_heads"]) or \
                 _get_attr_any(mc, ["num_key_value_heads", "n_kv_heads"])
    n_heads = _get_attr_any(hf, ["num_attention_heads", "n_head"]) or \
              _get_attr_any(mc, ["num_attention_heads", "n_head"])
    if n_kv_heads is None and n_heads is not None:
        n_kv_heads = int(n_heads)  # fallback (no GQA info)
    n_kv_heads = int(n_kv_heads) if n_kv_heads is not None else 0

    head_dim = _get_attr_any(hf, ["head_dim"])
    if head_dim is None:
        hidden = _get_attr_any(hf, ["hidden_size", "n_embd"])
        if hidden is not None and n_heads:
            try:
                head_dim = int(hidden) // int(n_heads)
            except Exception:
                head_dim = None
    head_dim = int(head_dim) if head_dim is not None else 0

    # KV dtype: prefer explicit kv_cache dtype; else engine/meta dtype
    kv_dtype = None
    cc = _get_attr_any(eng, ["cache_config"])
    if cc is not None:
        kv_dtype = _get_attr_any(cc, ["kv_cache_dtype"]) or _get_attr_any(cc, ["dtype"])
    if kv_dtype is None:
        kv_dtype = _get_attr_any(mc, ["dtype"])
    byte_per = _dtype_bytes(str(kv_dtype) if kv_dtype else None)
    if byte_per == 0:
        # In practice KV cache is fp16/bf16 even for AWQ; default to 2B.
        byte_per = 2

    # 2 (K & V) * heads * head_dim * bytes * block_size * num_layers
    try:
        return int(2 * n_kv_heads * head_dim * byte_per * block_size * num_layers)
    except Exception:
        return 0

def _get_used_blocks(mgr: Any, dev: str) -> int:
    """Try to read used-block count for 'gpu' or 'cpu' from manager."""
    # direct “used blocks”
    v = _get_attr_any(mgr, [f"{dev}_used_blocks", f"used_{dev}_blocks",
                            f"{dev}_active_blocks", f"active_{dev}_blocks"])
    if isinstance(v, int):
        return v
    # total - free
    total = _get_attr_any(mgr, [f"{dev}_num_blocks", f"num_{dev}_blocks",
                                f"total_{dev}_blocks", f"{dev}_total_blocks"])
    free = _get_attr_any(mgr, [f"{dev}_free_blocks", f"free_{dev}_blocks",
                               f"num_free_{dev}_blocks"])
    try:
        if total is not None and free is not None:
            return max(0, int(total) - int(free))
    except Exception:
        pass
    # nested pool/stats guesses
    pool = _get_attr_any(mgr, [f"{dev}_pool", f"{dev}_allocator", f"{dev}_cache"])
    if pool is not None:
        v2 = _get_attr_any(pool, ["used_blocks", "active_blocks", "num_used_blocks"])
        if isinstance(v2, int):
            return v2
        tot = _get_attr_any(pool, ["total_blocks", "num_blocks"])
        fre = _get_attr_any(pool, ["free_blocks", "num_free_blocks"])
        try:
            if tot is not None and fre is not None:
                return max(0, int(tot) - int(fre))
        except Exception:
            pass
    return 0
