#/vllm/v1/core/kvtuner_quantizer.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal

import torch
from vllm.v1.core.kvtuner_math import KVTunerMath, QuantMeta

_GLOBAL_QUANTIZER: Optional["KVTunerQuantizer"] = None

def set_global_quantizer(q: Optional["KVTunerQuantizer"]) -> None:
    """Install/uninstall the global KVTuner quantizer instance."""
    global _GLOBAL_QUANTIZER
    _GLOBAL_QUANTIZER = q
    if q is None:
        print("[KVTuner] Global quantizer cleared.")
    else:
        print("[KVTuner] Global quantizer set.")

def get_global_quantizer() -> Optional["KVTunerQuantizer"]:
    return _GLOBAL_QUANTIZER


@dataclass
class PackedKV:
    """Container for packed KV bytes + meta."""
    packed: torch.Tensor         # uint8 flat buffer
    meta: Dict[str, Any]         # scales/zps/shape/axis/bits etc.


class KVTunerQuantizer:
    """
    KVTuner-style layer-wise KV cache quantizer.

    Spec (요점):
      - External JSON `kvtuner_config`:
          * groups["per-token-asym"]: list[list[int]]  (layer index groups)
          * candidates_index_map: dict[str|int, int]   (e.g., {"0":8,"1":4,...})
          * group_choice: list[int]                    (e.g., [2,2,0,2,1])
          * optional: bit_candidates: [8,4,2,1]
          * optional: max_per_layer_scale (default 2.0)
      - Granularity:
          * Key:   per-channel, axis=1, q_group_size=32, residual_length=32
          * Value: per-token,   axis=0
      - Asymmetric quant (min/max), ties-to-even rounding (torch.round).
      - Scales/zero-points are stored in FP16/BF16 only (strict no-FP32).
      - Write path stores packed bytes + meta (16B align is write-path 일).
      - Read path dequant to FP16/BF16 only.
    """
    def __init__(self, cfg: Dict[str, Any], model_dtype: torch.dtype = torch.float16):
        self.cfg = cfg
        self.model_dtype = model_dtype if model_dtype in (torch.float16, torch.bfloat16) else torch.float16
        self.eps: float = 1e-8
        self.max_per_layer_scale: float = float(cfg.get("max_per_layer_scale", 2.0))

        groups = (cfg.get("groups") or {})
        self.groups_pta: list[list[int]] = groups.get("per-token-asym", [])
        if not isinstance(self.groups_pta, list):
            raise ValueError("kvtuner_config.groups['per-token-asym'] must be a list of lists.")

        # ---- candidates: support int OR {"k":int,"v":int} forms ----
        # ---- candidates: be liberal in what we accept ----
        def _norm_kv_pair(v: Any) -> tuple[int, int]:
            """Normalize one 'candidate' entry to (k_bits, v_bits).
            Accepts:
              - scalar: 8  -> (8, 8)
              - {"k":8,"v":4} or {"key":8,"value":4}
              - {"nbits_key":8,"nbits_value":4}
              - {"bits":4}  -> (4,4)
              - {"k_bits":8,"v_bits":4} / {"key_bits":8,"value_bits":4}
            """
            if isinstance(v, dict):
                # try a bunch of common aliases (spec + practical variants)
                aliases_k = ("k", "key", "nbits_key", "k_bits", "key_bits")
                aliases_v = ("v", "value", "nbits_value", "v_bits", "value_bits")
                kb = None
                vb = None
                for a in aliases_k:
                    if a in v:
                        kb = v[a]; break
                for a in aliases_v:
                    if a in v:
                        vb = v[a]; break
                # {"bits": X} -> both K/V use X
                if kb is None and vb is None and "bits" in v:
                    b = int(v["bits"])
                    return b, b
                # If one side is missing, mirror the other if present
                if kb is not None and vb is None:
                    vb = kb
                if vb is not None and kb is None:
                    kb = vb
                if kb is None or vb is None:
                    raise ValueError(
                        "Invalid candidate entry: expected scalar or a dict with "
                        "k/v (or key/value / nbits_key/nbits_value). Got: "
                        f"{v!r}"
                    )
                return int(kb), int(vb)
            # scalar -> same for k and v
            return int(v), int(v)

        cand_map = cfg.get("candidates_index_map") or {}
        self.candidates_index_map_k: Dict[int, int] = {}
        self.candidates_index_map_v: Dict[int, int] = {}
        self.bit_candidates_k: list[int] = []
        self.bit_candidates_v: list[int] = []

        if isinstance(cand_map, dict) and len(cand_map) > 0:
            # normalize keys to int, values to (k,v)
            tmp: Dict[int, tuple[int, int]] = {int(k): _norm_kv_pair(v) for k, v in cand_map.items()}
            for i in sorted(tmp.keys()):
                kb, vb = tmp[i]
                self.candidates_index_map_k[i] = kb
                self.candidates_index_map_v[i] = vb
                self.bit_candidates_k.append(kb)
                self.bit_candidates_v.append(vb)
        else:
            # derive from bit_candidates (int list OR dict list)
            raw_bc = cfg.get("bit_candidates", [8, 4])
            if not isinstance(raw_bc, list) or len(raw_bc) == 0:
                raw_bc = [8, 4]
            for i, entry in enumerate(raw_bc):
                kb, vb = _norm_kv_pair(entry)
                self.candidates_index_map_k[i] = kb
                self.candidates_index_map_v[i] = vb
                self.bit_candidates_k.append(kb)
                self.bit_candidates_v.append(vb)

        # ---- group_choice: support list OR {"per-token-asym":[...]} ----
        _gc = cfg.get("group_choice", [])
        if isinstance(_gc, dict):
            _gc = _gc.get("per-token-asym", [])
        self.group_choice: list[int] = list(map(int, _gc))
        if len(self.group_choice) != len(self.groups_pta):
            raise ValueError(
                f"group_choice length ({len(self.group_choice)}) "
                f"must match groups['per-token-asym'] length ({len(self.groups_pta)})."
            )

        # Build layer->group map for fast lookup
        self.layer_to_group: Dict[int, int] = {}
        for gid, layer_list in enumerate(self.groups_pta):
            if not isinstance(layer_list, list):
                raise ValueError("Each group in groups['per-token-asym'] must be a list of layer indices.")
            for L in layer_list:
                self.layer_to_group[int(L)] = gid

        # Quantization axes & groupsizes from spec
        self.key_axis: int = 1
        self.key_q_group_size: int = int(cfg.get("key_q_group_size", 32))
        self.key_residual_len: int = int(cfg.get("key_residual_length", 32))

        self.val_axis: int = 0
        self.val_q_group_size: Optional[int] = None  # per-token: no grouping unless specified
        self.val_residual_len: Optional[int] = None

        # Always-on debug summary
        print("[KVTuner] Quantizer init")
        print("  - model_dtype:", str(self.model_dtype).replace("torch.", ""))
        print("  - bit_candidates_k:", self.bit_candidates_k)
        print("  - bit_candidates_v:", self.bit_candidates_v)
        print("  - groups['per-token-asym']:", [len(g) for g in self.groups_pta], "(layers per group)")
        print("  - group_choice (indexes):", self.group_choice)
        print("  - candidates_index_map_k:", self.candidates_index_map_k)
        print("  - candidates_index_map_v:", self.candidates_index_map_v)
        print("  - max_per_layer_scale:", self.max_per_layer_scale)

        # Validate group_choice indices are in candidates_index_map
        valid_idxs = set(self.candidates_index_map_k.keys())
        bad = [i for i in self.group_choice if i not in valid_idxs]
        if bad:
            raise ValueError(
                f"group_choice contains invalid candidate indices {bad}; "
                f"allowed indices: {sorted(valid_idxs)}"
            )

    # ------------ Public API (used by write/read paths later) ------------
    def get_bits(self, layer_idx: int, kind: Literal["k", "v"] = "k") -> int:
        """Return selected bitwidth for given layer (K and V handled separately)."""
        gid = self.layer_to_group.get(int(layer_idx))
        if gid is None or gid < 0 or gid >= len(self.group_choice):
            # default to highest precision if out-of-range
            b = max(self.bit_candidates_k) if kind == "k" else max(self.bit_candidates_v)
        else:
            cand_idx = int(self.group_choice[gid])
            if kind == "k":
                b = int(self.candidates_index_map_k.get(cand_idx, max(self.bit_candidates_k)))
            else:
                b = int(self.candidates_index_map_v.get(cand_idx, max(self.bit_candidates_v)))
        return b

    # --- Key: per-channel (axis=1), grouped by q_group_size ---
    def quantize_k(self, layer_idx: int, K: torch.Tensor) -> PackedKV:
        bits = self.get_bits(layer_idx, "k")
        return self._quantize_generic(
            X=K, bits=bits, axis=self.key_axis, q_group_size=self.key_q_group_size,
            kind="k", layer_idx=layer_idx
        )

    def dequantize_k(self, packed: PackedKV) -> torch.Tensor:
        return self._dequantize_generic(packed)

    # --- Value: per-token (axis=0) ---
    def quantize_v(self, layer_idx: int, V: torch.Tensor) -> PackedKV:
        bits = self.get_bits(layer_idx, "v")
        return self._quantize_generic(
            X=V, bits=bits, axis=self.val_axis, q_group_size=self.val_q_group_size,
            kind="v", layer_idx=layer_idx
        )

    def dequantize_v(self, packed: PackedKV) -> torch.Tensor:
        return self._dequantize_generic(packed)

    # ---- Helpers for size accounting ----
    def estimate_packed_nbytes(self, shape: Tuple[int, ...], bits: int) -> int:
        numel = 1
        for d in shape:
            numel *= int(d)
        qbytes = (numel * bits + 7) // 8
        # Meta (very rough, dominated by scales/zps) – actual accounting handled by allocator.
        return int(qbytes)

    # ---------------- Internal implementation ----------------
    def _quantize_generic(
        self,
        X: torch.Tensor,
        bits: int,
        axis: int,
        q_group_size: Optional[int],
        kind: Literal["k", "v"],
        layer_idx: int,
    ) -> PackedKV:
        x = X.to(dtype=self.model_dtype)
        q_int, meta = KVTunerMath.quantize(
            x, bits,
            axis=axis,
            q_group_size=q_group_size,
            clamp_max=self.max_per_layer_scale,
            eps=self.eps,
            model_dtype=self.model_dtype,
        )
        flat_q = q_int.reshape(-1).to(torch.int32)

        packed = self._pack_nbits(flat_q, bits).to(torch.uint8)

        meta = {
            "kind": kind,
            "layer_idx": int(layer_idx),
            "bits": int(bits),
            "axis": int(meta.axis),
            "orig_shape": tuple(int(d) for d in meta.orig_shape),
            "group_size": int(meta.group_size),
            "groups": int(meta.groups),
            "scale": meta.scale,   # [..., G]
            "zp": meta.zp,         # [..., G]
            "dtype": str(self.model_dtype).replace("torch.", ""),
        }
        print(f"[KVTuner] Quantized {kind.upper()} L{layer_idx} -> {bits}-bit, "
              f"axis={meta.axis}, groups={meta.groups}, packed_bytes={packed.numel()}")

        return PackedKV(packed=packed, meta=meta)

    def _dequantize_generic(self, obj: PackedKV) -> torch.Tensor:
        meta = obj.meta
        bits = int(meta["bits"])
        shape = tuple(meta["orig_shape"])
        numel = 1
        for d in shape:
            numel *= d
        q_flat = self._unpack_nbits(obj.packed.to(torch.uint8), bits, numel)
        q_int = q_flat.view(*shape).to(torch.int32)
        qm = QuantMeta(
            axis=int(meta["axis"]),
            orig_shape=shape,
            group_size=int(meta["group_size"]),
            groups=int(meta["groups"]),
            scale=meta["scale"],
            zp=meta["zp"],
        )
        out = KVTunerMath.dequantize(q_int, bits, qm, model_dtype=self.model_dtype)
        return out

    @staticmethod
    def _pack_nbits(vals: torch.Tensor, bits: int) -> torch.Tensor:
        """Pack int tensor (non-negative) into a uint8 buffer."""
        assert vals.dtype in (torch.int32, torch.int16, torch.int8)
        if bits == 8:
            return vals.to(torch.uint8)
        # generic pack: little-endian within byte
        per_byte = 8 // bits
        # pad to multiple of per_byte
        n = vals.numel()
        pad = (per_byte - (n % per_byte)) % per_byte
        if pad:
            vals = torch.nn.functional.pad(vals, (0, pad), value=0)
        vals = vals.view(-1, per_byte).to(torch.int32)
        packed = torch.zeros(vals.shape[0], dtype=torch.int32, device=vals.device)
        for i in range(per_byte):
            packed |= (vals[:, i] & ((1 << bits) - 1)) << (i * bits)
        return packed.to(torch.uint8)

    @staticmethod
    def _unpack_nbits(buf: torch.Tensor, bits: int, out_numel: int) -> torch.Tensor:
        if bits == 8:
            out = buf.to(torch.int32)
            return out[:out_numel]
        per_byte = 8 // bits
        n_groups = (out_numel + per_byte - 1) // per_byte
        # ensure buf has n_groups bytes
        if buf.numel() < n_groups:
            raise ValueError("Packed buffer is shorter than expected.")
        b = buf[:n_groups].to(torch.int32)
        outs = []
        mask = (1 << bits) - 1
        for i in range(per_byte):
            outs.append((b >> (i * bits)) & mask)
        out = torch.stack(outs, dim=1).reshape(-1)
        return out[:out_numel]

