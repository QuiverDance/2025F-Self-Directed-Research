# vllm/v1/metrics/kv_quant.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import json
import time
from pathlib import Path

@dataclass
class KVQuantLayerRecord:
    layer: int
    bits_k: int
    bits_v: int
    kv_bytes_total_packed: int
    kv_bytes_scales: int
    tt: float

class KVQuantMetricsLogger:
    """Append-only JSONL metrics logger for KV quantization."""
    def __init__(self, log_path: Optional[str], enabled: bool):
        self.enabled = enabled and bool(log_path)
        self.log_path = Path(log_path) if log_path else None
        if self.enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_layer(self, rec: KVQuantLayerRecord):
        if not self.enabled:
            return
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def log_summary(self, total_bytes: int, scale_bytes: int):
        if not self.enabled:
            return
        payload = {"type": "kv_quant_summary", "kv_bytes_total_packed": int(total_bytes),
                   "kv_bytes_scales": int(scale_bytes), "tt": time.time()}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
