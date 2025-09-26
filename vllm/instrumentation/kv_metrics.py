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
    """Thread-safe append-only JSONL writer."""

    def __init__(self, path: str):

        # If a directory was given, write to kv_metrics.jsonl inside it.
        if path.endswith(os.sep) or (os.path.isdir(path) and not os.path.isfile(path)):
            path = os.path.join(path, "kv_metrics.jsonl")
        # Add rank suffix automatically to avoid contention in multi-proc runs.
        rank = os.getenv("RANK") or os.getenv("LOCAL_RANK")
        if rank is not None and path.endswith(".jsonl") and f".rank{rank}" not in path:
            base, ext = os.path.splitext(path)
            path = f"{base}.rank{rank}{ext}"

        self._path = path

        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        open(self._path, "a", encoding="utf-8").close()
        
        self._lock = threading.Lock()

    def write(self, data: Dict[str, Any]):
        line = json.dumps(data, ensure_ascii=False)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

class KVMetricsConfig:
    """Config driven by CLI/env.
    Enablement accepts: on/true/1/yes (case-insensitive)."""
    def __init__(self, mode: Optional[str] = None, out_path:Optional[str]=None):

        env_mode = (os.getenv("VLLM_KV_METRICS") or "").strip().lower()
        mode = (mode or "").strip().lower() or env_mode
        truthy = {"on", "true", "1", "yes"}
        self._enabled = mode in truthy
        env_path = os.getenv("VLLM_KV_METRICS_PATH")
        self.out_path = out_path or env_path or "./runs/kv_metrics.jsonl"


    @property
    def enabled(self) -> bool:
        return bool(self._enabled)


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
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = KVMetricsCollector(config or KVMetricsConfig())
        return cls._instance

    def __init__(self, config: KVMetricsConfig):
        self.cfg = config
        self._writer = _SafeJSONLWriter(self.cfg.out_path) if self.cfg.enabled else None
        self._engine_ref: Any = None
        self._req: Dict[str, RequestLog] = {}
        self._t_start: Dict[str, float] = {}
        self._t_first: Dict[str, float] = {}
        self._t_end: Dict[str, float] = {}
        self._finished: set[str] = set()  # guard double-flush
        self._lock = threading.Lock()

    # ---------- Engine linking & KV snapshot ----------
    def link_engine(self, engine_like: Any) -> None:
        """Keep a weak-ish reference to engine internals for KV snapshots."""
        self._engine_ref = engine_like

    def _kv_bytes_snapshot(self) -> Dict[str, int]:
        """Read global KV usage (bytes) from engine-like object."""
        gpu_bytes = 0
        cpu_bytes = 0
        eng = self._engine_ref
        if eng is None:
            return {"total": 0, "gpu": 0, "cpu": 0}
        candidates = ["block_manager", "kv_cache_manager", "cache_manager", "block_space_manager"]
        mgr = None
        for name in candidates:
            mgr = getattr(eng, name, None)
            if mgr is not None:
                break
        def _try_int(x) -> int:
            try:
                return int(x)
            except Exception:
                return 0
        if mgr is not None:
            for attr in ("gpu_bytes", "gpu_used_bytes", "gpu_allocated_bytes"):
                val = getattr(mgr, attr, None)
                if val is not None:
                    gpu_bytes = _try_int(val)
                    break
            for attr in ("cpu_bytes", "cpu_used_bytes", "host_bytes", "offload_bytes"):
                val = getattr(mgr, attr, None)
                if val is not None:
                    cpu_bytes = _try_int(val)
                    break
        return {"total": gpu_bytes + cpu_bytes, "gpu": gpu_bytes, "cpu": cpu_bytes}

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
                payload = asdict(rec)
                payload["ts"] = int(time.time() * 1000)  # ordering aid
                self._writer.write(payload)
            self._finished.add(request_id)
            for d in (self._req, self._t_start, self._t_first, self._t_end):
                d.pop(request_id, None)

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