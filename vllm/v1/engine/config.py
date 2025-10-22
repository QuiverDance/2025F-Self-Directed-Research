# vllm/v1/engine/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import json
import re

_VALID_BITS = {2, 4, 8}
_DEFAULT_POLICY = {"K": 8, "V": 8, "group_size": 64, "mode_k": "asymmetric_channel", "mode_v": "asymmetric_token"}

def _expand_layers_key(key: str) -> List[int]:
    """Parse layer range string like '0-3,6,8-10' into indices."""
    out: List[int] = []
    for part in key.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(list(range(int(a), int(b) + 1)))
        elif part:
            out.append(int(part))
    return out

@dataclass
class LayerPolicy:
    bits_k: int
    bits_v: int
    group_size: int = 64
    mode_k: str = "asymmetric_channel" # "symmetric_channel" | "symmetric_token" | "asymmetric_channel" | "asymmetric_token"
    mode_v: str = "asymmetric_token"

@dataclass
class KVQuantConfig:
    enable: bool = False
    fused_attn: bool = False
    validate: bool = False
    log_path: Optional[str] = None
    log_interval: int = 128
    block_size: int = 16
    default_policy: LayerPolicy = field(default_factory=lambda: LayerPolicy(**_DEFAULT_POLICY))
    per_layer: Dict[int, LayerPolicy] = field(default_factory=dict)

    @classmethod
    def from_args(cls, args) -> "KVQuantConfig":
        enable = bool(getattr(args, "kv_quant_enable", False))
        fused = bool(getattr(args, "kv_quant_fused_attn", False))
        validate = bool(getattr(args, "kv_quant_validate", False))
        log_path = getattr(args, "kv_quant_log_path", None)
        log_interval = int(getattr(args, "kv_quant_log_interval", 128))
        block_size = int(getattr(args, "kv_quant_block_size", 16))

        cfg_path = getattr(args, "kv_quant_config", None)
        default_policy = LayerPolicy(**_DEFAULT_POLICY)

        per_layer: Dict[int, LayerPolicy] = {}
        if cfg_path:
            with open(cfg_path, "r") as f:
                raw = json.load(f)
            base = {**_DEFAULT_POLICY, **(raw.get("default", {}))}
            default_policy = LayerPolicy(
                bits_k=int(base.get("K", 8)), bits_v=int(base.get("V", 8)),
                group_size=int(base.get("group_size", 64)),
                mode_k=str(base.get("mode_k", "asymmetric_channel")),
                mode_v=str(base.get("mode_v", "asymmetric_token")),
            )
            layers = raw.get("layers", {})
            for spec, pol in layers.items():
                bits_k = int(pol.get("K", default_policy.bits_k))
                bits_v = int(pol.get("V", default_policy.bits_v))
                gsz = int(pol.get("group_size", default_policy.group_size))
                mode_k = str(pol.get("mode_k", default_policy.mode_k))
                mode_v = str(pol.get("mode_v", default_policy.mode_v))
                for li in _expand_layers_key(spec):
                    per_layer[li] = LayerPolicy(bits_k, bits_v, gsz, mode_k, mode_v)

        # validate bits
        if enable:
            for li, p in per_layer.items():
                if p.bits_k not in _VALID_BITS or p.bits_v not in _VALID_BITS:
                    raise ValueError(f"Layer {li}: invalid bits (K={p.bits_k}, V={p.bits_v}); allowed: {_VALID_BITS}")
            if default_policy.bits_k not in _VALID_BITS or default_policy.bits_v not in _VALID_BITS:
                raise ValueError(f"default: invalid bits (K={default_policy.bits_k}, V={default_policy.bits_v})")

        return cls(enable=enable, fused_attn=fused, validate=validate,
                   log_path=log_path, log_interval=log_interval, block_size=block_size,
                   default_policy=default_policy, per_layer=per_layer)

    def policy_for(self, layer_idx: int) -> LayerPolicy:
        return self.per_layer.get(layer_idx, self.default_policy)
