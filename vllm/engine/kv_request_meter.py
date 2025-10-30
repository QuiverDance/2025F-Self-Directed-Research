from __future__ import annotations

import json
import os
import re
import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Set, List, Any


# ----------------------------- helpers (single-path) -----------------------
def _read_hf_config_from_model_dir(mc) -> Optional[dict]:
    """Single-path: use engine.model_config.model to find <model_dir>/config.json."""
    try:
        model_dir = getattr(mc, "model")
        if not isinstance(model_dir, str):
            return None
        cfg_path = os.path.join(model_dir, "config.json")
        if not os.path.isfile(cfg_path):
            return None
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _parse_block_size_from_metrics_info(cc) -> Optional[int]:
    s = None
    try:
        s = getattr(cc, "metrics_info")()
    except Exception:
        return None
    if not s:
        return None
    m = re.search(r"'block_size'\s*:\s*'(\d+)'", str(s))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _human_bytes(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    v = float(x); i = 0
    while v >= 1024.0 and i < len(units) - 1:
        v /= 1024.0; i += 1
    return f"{v:.2f} {units[i]}"

def _bpe_from_dtype_str(s: Optional[str]) -> int:
    """Return bytes-per-element; default to FP16 if unknown."""
    if not s:
        return 2
    t = str(s).lower()
    if "float32" in t or "fp32" in t: return 4
    if "bfloat16" in t or "bf16" in t: return 2
    if "float16" in t or "fp16" in t or "half" in t: return 2
    if "fp8" in t or "e4m3" in t or "e5m2" in t: return 1
    if "int8" in t or "uint8" in t: return 1
    return 2  # safe default

def _safe_call0(obj, name: str):
    """Call zero-arg getter by exact name; return None if missing/fails."""
    try:
        fn = getattr(obj, name)
    except Exception:
        return None
    try:
        return fn() if callable(fn) else fn
    except TypeError:
        return None
    except Exception:
        return None

def _safe_attr(obj, name: str):
    """Get attribute by exact name; return None if missing/fails."""
    try:
        return getattr(obj, name)
    except Exception:
        return None

def _parse_num_hidden_layers_from_repr(mc) -> Optional[int]:
    """Use exactly one path: parse validate_model_config_after() string."""
    s = _safe_call0(mc, "validate_model_config_after")
    if not s:
        return None
    # single regex pattern; do NOT try multiple names
    m = re.search(r"num_hidden_layers\s*=\s*(\d+)", str(s))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def _parse_block_size_from_metrics_info(cc) -> Optional[int]:
    """Use exactly one path: parse metrics_info() string for 'block_size'."""
    s = _safe_call0(cc, "metrics_info")
    if not s:
        return None
    m = re.search(r"'block_size'\s*:\s*'(\d+)'", str(s))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def _parse_model_dtype_from_repr(mc) -> Optional[str]:
    """Use exactly one path: parse validate_model_config_after() for dtype=..."""
    s = _safe_call0(mc, "validate_model_config_after")
    if not s:
        return None
    m = re.search(r"dtype\s*=\s*([A-Za-z0-9\._]+)", str(s))
    if m:
        return m.group(1)
    return None


# ----------------------------- data model ----------------------------------

@dataclass
class _ReqStat:
    req_id: str
    t_add: float
    t_first_seen: Optional[float] = None
    t_finish: Optional[float] = None
    auc_kv: float = 0.0
    max_kv: float = 0.0
    auc_bytes: float = 0.0
    max_bytes: float = 0.0
    last_ts: Optional[float] = None
    steps: int = 0


# ----------------------------- main meter ----------------------------------

class KVRequestMeter:
    """Request-scoped KV meter (strict single-name logic, no multi-name guessing).

    Exact sources (per your dump):
      - ModelConfig: get_hidden_size(), get_head_size(), get_total_num_kv_heads()
      - Model layers: validate_model_config_after() string → parse 'num_hidden_layers'
      - CacheConfig: block_size attribute; else metrics_info() string → 'block_size'
      - DType: kv_cache_dtype attribute (if present and not 'None'/'auto');
               else parse dtype from validate_model_config_after(); else FP16
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engine = None
        self._run_id: Optional[str] = None

        # capacity metadata
        self._bytes_per_block: Optional[float] = None
        self._tokens_per_block: Optional[int] = None
        self._total_blocks: Optional[int] = None
        self._total_bytes: Optional[float] = None

        # per-request stats
        self._reqs: Dict[str, _ReqStat] = {}

    # ---- lifecycle ----

    def attach(self, engine) -> None:
        with self._lock:
            self._engine = engine

    def reset(self, run_id: Optional[str] = None) -> None:
        with self._lock:
            self._run_id = run_id or f"run@{int(time.time() * 1000)}"
            self._reqs.clear()
            self._bytes_per_block = None
            self._tokens_per_block = None
            self._total_blocks = None
            self._total_bytes = None

    # ---- capacity derivation (single-path) ----

    def _ensure_bytes_per_block(self) -> None:
        """Compute bytes_per_block once using a single source: HF config.json."""
        if self._bytes_per_block is not None or self._engine is None:
            return
    
        mc = getattr(self._engine, "model_config", None)
        cc = getattr(self._engine, "cache_config", None)
        if mc is None or cc is None:
            return

        # ---- Single path: read <model_dir>/config.json
        cfg = _read_hf_config_from_model_dir(mc)
        if cfg is None:
            return
    
        # Exact fields from config.json (no guessing)
        hidden_size = cfg.get("hidden_size")
        n_heads = cfg.get("num_attention_heads")
        n_kv_heads = cfg.get("num_key_value_heads") or n_heads
        n_layers = cfg.get("num_hidden_layers")
        torch_dtype = cfg.get("torch_dtype")  # e.g., "bfloat16", "float16", "float32"
    
        ok_dims = all(isinstance(x, int) and x > 0 for x in [hidden_size, n_heads, n_kv_heads, n_layers])
        if not ok_dims:
            return
    
        head_dim = hidden_size // n_heads
    
        # Tokens per block: prefer attribute; if absent, parse metrics_info() once.
        tokens_per_block = getattr(cc, "block_size", None)
        if not isinstance(tokens_per_block, int) or tokens_per_block <= 0:
            tokens_per_block = _parse_block_size_from_metrics_info(cc)
        if not isinstance(tokens_per_block, int) or tokens_per_block <= 0:
            return
    
        # DType: prefer cache-specific if explicit; else use HF torch_dtype; else default fp16.
        cc_dtype = getattr(cc, "kv_cache_dtype", None)
        if cc_dtype and str(cc_dtype).lower() not in ("none", "auto"):
            bpe = _bpe_from_dtype_str(str(cc_dtype))
        else:
            bpe = _bpe_from_dtype_str(torch_dtype)
    
        # bytes per token / block (layers * 2(K,V) * kv_heads * head_dim * bytes/elem)
        bytes_per_token = int(n_layers) * 2 * int(n_kv_heads) * int(head_dim) * int(bpe)
        self._bytes_per_block = float(bytes_per_token * int(tokens_per_block))
        self._tokens_per_block = int(tokens_per_block)
    
        # finalize total_bytes if total_blocks already known
        if self._total_blocks is not None:
            self._total_bytes = self._bytes_per_block * self._total_blocks
    
    # ---- per-step / per-req ----

    def note_request_added(self, req_id: str, ts: float | None = None) -> None:
        now = ts if ts is not None else time.monotonic()
        with self._lock:
            if req_id not in self._reqs:
                self._reqs[req_id] = _ReqStat(req_id=req_id, t_add=now)
    
    def get_open_request_ids(self) -> set[str]:
        with self._lock:
            return {rid for rid, rs in self._reqs.items() if rs.t_finish is None}

    def note_step(
        self,
        active_req_ids: Set[str],
        kv_usage_perc: Optional[float],
        ts: Optional[float],
        used_blocks: Optional[int] = None,  # ignored (single-path design)
        total_blocks: Optional[int] = None,
    ) -> None:
        if ts is None:
            return
        with self._lock:
            # Accept total_blocks from engine (locked path in llm_engine)
            if isinstance(total_blocks, int) and total_blocks > 0:
                if self._total_blocks != int(total_blocks):
                    self._total_blocks = int(total_blocks)
                    if self._bytes_per_block is not None:
                        self._total_bytes = self._bytes_per_block * self._total_blocks

            # Ensure capacity math ready
            self._ensure_bytes_per_block()
            
            if kv_usage_perc is not None and self._total_bytes is not None:
                cur_bytes = float(kv_usage_perc) * float(self._total_bytes)
            
            for rid in active_req_ids:
                rs = self._reqs.get(rid)
                if rs is None:
                    rs = _ReqStat(req_id=rid, t_add=ts)
                    self._reqs[rid] = rs

                if rs.t_first_seen is None:
                    rs.t_first_seen = ts
                    rs.last_ts = ts

                if rs.last_ts is not None and ts >= rs.last_ts:
                    dt = ts - rs.last_ts
                    if kv_usage_perc is not None:
                        rs.auc_kv += float(kv_usage_perc) * dt
                    if cur_bytes is not None:
                        rs.auc_bytes += float(cur_bytes) * dt
                    rs.last_ts = ts

                if kv_usage_perc is not None:
                    rs.max_kv = max(rs.max_kv, float(kv_usage_perc))
                if cur_bytes is not None:
                    rs.max_bytes = max(rs.max_bytes, float(cur_bytes))

                rs.steps += 1

    def note_request_finished(self, req_id: str, ts: Optional[float] = None) -> None:
        t = ts or time.monotonic()
        with self._lock:
            rs = self._reqs.get(req_id)
            if rs is None:
                rs = _ReqStat(req_id=req_id, t_add=t)
                self._reqs[req_id] = rs
            rs.t_finish = t

    # ---- reporting ----

    def get_total_kv_bytes(self) -> Optional[float]:
        with self._lock:
            self._ensure_bytes_per_block()
            return self._total_bytes

    def get_run_summary(self) -> dict:
        with self._lock:
            self._ensure_bytes_per_block()

            per_req: List[dict] = []
            # pooled (ratio-of-sums) accumulators
            total_auc_kv = 0.0
            total_auc_bytes = 0.0
            total_dur = 0.0
            total_decode_time = 0.0
            total_decode_steps = 0
            for rs in self._reqs.values():
                dur = None
                if rs.t_first_seen is not None and rs.t_finish is not None:
                    dur = max(0.0, rs.t_finish - rs.t_first_seen)
    
                avg_kv = None
                avg_bytes = None
                if dur and dur > 0:
                    avg_kv = rs.auc_kv / dur
                    avg_bytes = rs.auc_bytes / dur
                    total_auc_kv += rs.auc_kv
                    total_auc_bytes += rs.auc_bytes
                    total_dur += dur
                
                ttft = None
                if rs.t_first_seen is not None:
                    ttft = max(0.0, rs.t_first_seen - rs.t_add)
                
                decode_time = dur if dur is not None else None
                decode_steps = rs.steps  # step-based approximation of #decoded tokens
                decode_tps = None
                decode_tpt_ms = None
                if decode_time is not None and decode_time > 0 and decode_steps > 0:
                    decode_tps = decode_steps / decode_time
                    decode_tpt_ms = 1000.0 * (decode_time / decode_steps)
                    total_decode_time += decode_time
                    total_decode_steps += decode_steps

                peak_kv_bytes = rs.max_bytes if rs.max_bytes is not None else None
                peak_kv_usage = rs.max_kv if rs.max_kv is not None else None

                item = {
                    "request_id": rs.req_id,
                    "t_add": rs.t_add,
                    "t_first_seen": rs.t_first_seen,
                    "t_finish": rs.t_finish,
                    "duration_decode_like": dur,
                    "kv_usage_max": rs.max_kv,
                    "kv_usage_avg_time_weighted": avg_kv,
                    "kv_bytes_max": rs.max_bytes,
                    "kv_bytes_avg_time_weighted": avg_bytes,
                    "kv_bytes_auc": rs.auc_bytes,
                    "steps_seen": rs.steps,
                    "ttft_seconds": ttft,
                    "decode_tps": decode_tps,     
                    "decode_tpt_ms": decode_tpt_ms,
                    "peak_kv_bytes": peak_kv_bytes,
                    "peak_kv_usage": peak_kv_usage,
                }
                if item["kv_bytes_max"] is not None:
                    item["kv_bytes_max_readable"] = _human_bytes(item["kv_bytes_max"])
                if item["kv_bytes_avg_time_weighted"] is not None:
                    item["kv_bytes_avg_time_weighted_readable"] = _human_bytes(item["kv_bytes_avg_time_weighted"])
                if item["peak_kv_bytes"] is not None:
                    item["peak_kv_bytes_readable"] = _human_bytes(item["peak_kv_bytes"])
                if item["ttft_seconds"] is not None:
                    item["ttft_ms_readable"] = f"{item['ttft_seconds']*1000.0:.1f} ms"
                if item["decode_tpt_ms"] is not None:
                    item["decode_tpt_ms_readable"] = f"{item['decode_tpt_ms']:.2f} ms/token"
                if item["decode_tps"] is not None:
                    item["decode_tps_readable"] = f"{item['decode_tps']:.2f} tok/s"
   
                per_req.append(item)

            n = len(per_req)
            agg = {}
            if n > 0:
                try:
                    import statistics as st
                    max_perc = [x["kv_usage_max"] for x in per_req if x["kv_usage_max"] is not None]
                    avg_perc = [x["kv_usage_avg_time_weighted"] for x in per_req if x["kv_usage_avg_time_weighted"] is not None]
                    max_bytes = [x["kv_bytes_max"] for x in per_req if x["kv_bytes_max"] is not None]
                    avg_bytes = [x["kv_bytes_avg_time_weighted"] for x in per_req if x["kv_bytes_avg_time_weighted"] is not None]
                    ttft_list = [x["ttft_seconds"] for x in per_req if x["ttft_seconds"] is not None]
                    tps_list  = [x["decode_tps"] for x in per_req if x["decode_tps"] is not None]
                    tpt_list  = [x["decode_tpt_ms"] for x in per_req if x["decode_tpt_ms"] is not None]
                    # --- pooled / overall (ratio-of-sums) primary, fallback to simple mean if not computable ---
                    pooled_kv_usage_avg = (total_auc_kv / total_dur) if total_dur > 0 else (st.mean(avg_perc) if avg_perc else None)
                    pooled_kv_bytes_avg = (total_auc_bytes / total_dur) if total_dur > 0 else (st.mean(avg_bytes) if avg_bytes else None)
                    pooled_decode_tps   = (total_decode_steps / total_decode_time) if total_decode_time > 0 else (st.mean(tps_list) if tps_list else None)
                    pooled_decode_tpt   = (1000.0 * (total_decode_time / total_decode_steps)) if total_decode_steps > 0 else (st.mean(tpt_list) if tpt_list else None)
                    
                    agg = {
                        "num_requests": n,
                        "kv_usage_max": max(max_perc) if max_perc else None,
                        "kv_usage_avg_time_weighted_mean": pooled_kv_usage_avg,
                        "kv_bytes_max": max(max_bytes) if max_bytes else None,
                        "kv_bytes_avg_time_weighted_mean": pooled_kv_bytes_avg,
                        "ttft_mean_seconds": st.mean(ttft_list) if ttft_list else None,
                        "decode_tps_mean": pooled_decode_tps,
                        "decode_tpt_ms_mean": pooled_decode_tpt,
                    }
                except Exception:
                    agg = {"num_requests": n}

            total_bytes = self._total_bytes
            out = {
                "run_id": self._run_id,
                "num_requests": n,
                "total_kv_capacity_bytes": total_bytes,
                "total_kv_capacity_readable": _human_bytes(total_bytes),
                "per_request": per_req,
                "aggregate": agg,
            }
            if agg.get("kv_bytes_max") is not None:
                out["aggregate"]["kv_bytes_max_readable"] = _human_bytes(agg["kv_bytes_max"])
            if agg.get("kv_bytes_avg_time_weighted_mean") is not None:
                out["aggregate"]["kv_bytes_avg_time_weighted_mean_readable"] = _human_bytes(
                    agg["kv_bytes_avg_time_weighted_mean"]
                )
            if agg.get("ttft_mean_seconds") is not None:
               out["aggregate"]["ttft_mean_ms_readable"] = f"{agg['ttft_mean_seconds']*1000.0:.1f} ms"
            if agg.get("decode_tpt_ms_mean") is not None:
                out["aggregate"]["decode_tpt_ms_mean_readable"] = f"{agg['decode_tpt_ms_mean']:.2f} ms/token"
            if agg.get("decode_tps_mean") is not None:
                out["aggregate"]["decode_tps_mean_readable"] = f"{agg['decode_tps_mean']:.2f} tok/s"
            return out


# Module-level singleton
kv_meter = KVRequestMeter()

