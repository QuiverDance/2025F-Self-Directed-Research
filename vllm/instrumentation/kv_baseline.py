# vllm/instrumentation/kv_baseline.py
# Baseline request-scoped metrics for KV cache & latency.
# - Minimal overhead, single JSONL append per request.

from __future__ import annotations
import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

def _now_ms_monotonic() -> float:
    return time.monotonic_ns() / 1e6

@dataclass
class RequestLog:
    # Required idenfitifiers / meta
    request_id: str
    model: Optional[str] = None
    dtype: Optional[str] = None
    num_devices: Optional[int] = None
    block_size: Optional[int] = None

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
        self._path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        open(self._path, "a", encoding="utf-8").close()  # Ensure file exists

    def write(self, data: Dict[str, Any]):
        line = json.dumps(data, ensure_ascii=False)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

class KVMetricsConfig:
    """Runtime config from CLI/env."""
    def __init__(self, mode:str="off", out_path:Optional[str]=None):
        # mode: "off" | "baseline"
        self.mode = (mode or "off").lower()
        env_mode = os.getenv("VLLM_KV_METRICS", "").lower()
        if env_mode:
            self.mode = env_mode
        self.out_path = out_path or os.getenv("VLLM_KV_METRICS_PATH", "")
        if not self.out_path:
            self.out_path = ".runs/kv_baseline.jsonl"


    @property
    def enabled(self) -> bool:
        return self.mode == "baseline"


class KVMetricsCollector:
    """
    Per-process singleton to collect request-scoped baseline metrics.

    Integration points:
    - Engine:
        * on_enqueue(request_id, context_len, meta)
        * on_first_token(request_id)
        * on_prefill_end(request_id)  # baseline: same moment as first token
        * on_stream_end(request_id)
        * snapshot_kv(phase, request_id)
      Requires `link_engine(engine)` so we can read KV pool bytes.

    Notes:
    - KV snapshots are GLOBAL to the engine at that moment, not per-request.
    - generated_len excludes EOS by convention.
    """    

    _instance: Optional["KVMetricsCollector"] = None
    _init_lock = threading.Lock()

    @classmethod
    def get(cls, config: Optional[KVMetricsConfig] = None) -> "KVMetricsCollector":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = KVMetricsCollector(config or KVMetricsConfig())
        return cls._instance

    def __init__(self, config: KVMetricsConfig):
        self.cfg = config
        self._writer = _SafeJsonWriter(self.cfg.out_path) if self.cfg.enabled else None
        self._engine = None  # weak reference to engine-like object
        self._req: Dict[str, RequestLog] = {}
        self._t_start: Dict[str, float] = {}
        self._t_first: Dict[str, float] = {}
        self._t_end: Dict[str, float] = {}
        self._lock = threading.Lock()

    # ---------- Engine linking & KV snapshot ----------
    def link_engine(self, engine_like: Any):
        """
        Give us access to engine internals for KV snapshots & meta.
        We will try to discover block manager & meta across versions.
        """
        self._engine_ref = engine_like

    def _kv_bytes_snapshot(self) -> Dict[str, int]:
        """
        Read global KV usage (bytes) from engine. We try several attribute names
        to be robust across vLLM versions. If we cannot find them, we return zeros.
        """
        gpu_bytes = 0
        cpu_bytes = 0
        total = 0

        eng = self._engine_ref
        if eng is None:
            return dict(total=0, gpu=0, cpu=0)

        # Heuristic lookup: block manager / cache manager
        candidates = [
            # common names to try (across versions)
            "block_manager", "kv_cache_manager", "cache_manager", "block_space_manager"
        ]
        mgr = None
        for name in candidates:
            mgr = getattr(eng, name, None)
            if mgr is not None:
                break

        # try to read pools/counters
        def _try_int(x) -> int:
            try:
                return int(x)
            except Exception:
                return 0
        
        if mgr is not None:
            # Typical attributes to try. Fall back gracefully.
            # gpu
            for attr in ["gpu_bytes", "gpu_used_bytes", "gpu_allocated_bytes"]:
                val = getattr(mgr, attr, None)
                if val is not None:
                    gpu_bytes = _try_int(val)
                    break
            # cpu/offload
            for attr in ["cpu_bytes", "cpu_used_bytes", "host_bytes", "offload_bytes"]:
                val = getattr(mgr, attr, None)
                if val is not None:
                    cpu_bytes = _try_int(val)
                    break
        
        total = gpu_bytes + cpu_bytes
        return dict(total=total, gpu=gpu_bytes, cpu=cpu_bytes)

    def snapshot_kv(self, phase: str, request_id: str):
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

    
    # ---------- Request lifecycle (engine-side) ----------
    def on_enqueue(self, request_id: str, context_len: int, meta: Optional[Dict[str, Any]] = None):
        if not self.cfg.enabled:
            return
        now = _now_ms_monotonic()
        with self._lock:
            if request_id not in self._req:
                self._req[request_id] = RequestLog(
                    request_id=request_id,
                    model=(meta or {}).get("model"),
                    dtype=(meta or {}).get("dtype"),
                    num_devices=(meta or {}).get("num_devices"),
                    block_size=(meta or {}).get("block_size"),
                    context_len=int(context_len or 0),
                    generated_len=0,
                )
            self._t_start[request_id] = now
    
    def on_first_token(self, request_id: str):
        if not self.cfg.enabled:
            return
        now = _now_ms_monotonic()
        with self._lock:
            if request_id in self._t_first:
                return  # already marked
            self._t_first[request_id] = now
            t0 = self._t_start.get(request_id)
            rec = self._req.get(request_id)
            if rec and t0 is not None:
                rec.ttft_ms = max(0.0, now - t0)
                # Baseline: prefill_end == first_token moment
                rec.prefill_ms = max(0.0, now - t0)

    def on_prefill_end(self, request_id: str):
        # For baseline, this is equivalent to on_first_token. Retained for clarity.
        self.on_first_token(request_id)

    def on_stream_end(self, request_id: str):
        if not self.cfg.enabled:
            return
        now = _now_ms_monotonic()
        with self._lock:
            self._t_end[request_id] = now
            t_first = self._t_first.get(request_id)
            rec = self._req.get(request_id)
            if rec and t_first is not None:
                rec.decode_ms = max(0.0, now - t_first)
                # TPS with guards (<1ms => null)
                if rec.prefill_ms and rec.prefill_ms >= 1.0:
                    rec.tps_prefill = rec.context_len / (rec.prefill_ms / 1000.0)
                else:
                    rec.tps_prefill = None
                if rec.decode_ms and rec.decode_ms >= 1.0:
                    rec.tps_decode = rec.generated_len / (rec.decode_ms / 1000.0)
                else:
                    rec.tps_decode = None

            # Flush and clean
            if self._writer and rec:
                payload = asdict(rec)
                # Optional ordering ts (wall-clock) for easier log reading
                payload["ts"] = int(time.time() * 1000)
                self._writer.write(payload)

            # GC request state
            for d in (self._req, self._t_start, self._t_first, self._t_end):
                if request_id in d:
                    del d[request_id]

    # ---------- Token counting ----------
    def count_generated_token(self, request_id: str, is_eos: bool = False):
        if not self.cfg.enabled:
            return
        if is_eos:
            return  # EOS excluded by convention
        with self._lock:
            rec = self._req.get(request_id)
            if rec:
                rec.generated_len += 1