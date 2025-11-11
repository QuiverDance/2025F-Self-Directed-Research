#/vllm/v1/core/kvtuner_quantizer.py
from __future__ import annotations
import os, json, math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal

import torch
from vllm.v1.core.kvtuner_math import KVTunerMath, QuantMeta
from vllm.v1.serial_utils import pack_nbits, unpack_nbits
from vllm.v1.pool.kv_meta import build_kv_meta, parse_kv_meta
import os, sys
_GLOBAL_QUANTIZER: Optional["KVTunerQuantizer"] = None

def set_global_quantizer(q: Optional["KVTunerQuantizer"]) -> None:
    """Install/uninstall the global KVTuner quantizer instance."""
    global _GLOBAL_QUANTIZER
    _GLOBAL_QUANTIZER = q

    print(
        "[KVTuner] Global quantizer set." if q is not None else "[KVTuner] Global quantizer cleared.",
        "pid=", os.getpid(),
        "module=", __name__, "file=", __file__,
        "mod_id=", id(sys.modules[__name__]),
        "obj_id=", id(_GLOBAL_QUANTIZER),
        flush=True,
    )
    if q is None:
        print("[KVTuner] Global quantizer cleared.")
    else:
        print("[KVTuner] Global quantizer set.")

def get_global_quantizer() -> Optional["KVTunerQuantizer"]:
    print(
        "[KVTuner] get_global_quantizer",
        "pid=", os.getpid(),
        "module=", __name__, "file=", __file__,
        "mod_id=", id(sys.modules[__name__]),
        "obj_id=", id(_GLOBAL_QUANTIZER),
        "_GLOBAL_QUANTIZER=", _GLOBAL_QUANTIZER,
        flush=True,
    )
    return _GLOBAL_QUANTIZER

# --- Local 4-bit pack/unpack for V (lo-first) to avoid nibble-order mismatch ---
def _pack4_lohi(q_flat_int32: torch.Tensor) -> torch.ByteTensor:
    """Pack int32 values in [0..15] into uint8 bytes: low nibble = q[0], high = q[1]."""
    q = q_flat_int32.to(torch.int32)
    if q.numel() % 2 != 0:
        # Pad one zero to make even length so each byte has two nibbles.
        q = torch.cat([q, torch.zeros(1, dtype=torch.int32, device=q.device)], dim=0)
    q0 = q[0::2].to(torch.uint8)              # low nibble values
    q1 = q[1::2].to(torch.uint8)              # high nibble values
    packed = (q0 & 0x0F) | ((q1 & 0x0F) << 4)
    return packed.contiguous()

def _unpack4_lohi(packed_u8: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack uint8 bytes into int32 low-first sequence of length `numel`."""
    x = packed_u8.to(torch.uint8)
    lo = (x & 0x0F).to(torch.int32)
    hi = ((x >> 4) & 0x0F).to(torch.int32)
    q = torch.empty(lo.numel() * 2, dtype=torch.int32, device=x.device)
    q[0::2] = lo
    q[1::2] = hi
    if numel < q.numel():
        q = q[:numel]
    return q
@dataclass
class PackedKV:
    """Container for packed KV bytes + meta."""
    packed: torch.Tensor         # uint8 flat buffer (device = X.device)
    meta_bytes: bytes            # 16B-aligned serialized meta (CPU bytes)


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
        self.enable = (bool)(cfg.get("enable", False))
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
        print("  - enable:", self.enable)
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
    def quantize_v(self, layer_idx: int, v: torch.Tensor) -> PackedKV:
        """
        Quantize V-cache tensor.
        This now uses the generic quantization path for consistency and robustness.
        """
        bits = self.get_bits(layer_idx, "v")
        # Use the same generic implementation as quantize_k, but with parameters
        # specific to V-cache (per-token quantization means axis=0 and no grouping).
        return self._quantize_generic(
            X=v,
            bits=bits,
            axis=self.val_axis,
            q_group_size=self.val_q_group_size,
            kind="v",
            layer_idx=layer_idx
        )

    def dequantize_v(self, obj: PackedKV) -> torch.Tensor:
        """
        Dequantize V-cache tensor.
        This uses the generic dequantization path which correctly handles metadata.
        """
        # The generic dequantizer correctly parses the metadata created by
        # the generic quantizer, ensuring symmetry.
        return self._dequantize_generic(obj)

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

        # =========== START: ADD DEBUGGING CODE ===========
        # Perform an immediate dequantization to check the quantization quality.
        # This helps verify if the quantization math itself is correct.
        try:
            x_hat = KVTunerMath.dequantize(q_int.clone(), bits, meta, model_dtype=self.model_dtype)
            # Calculate the L-infinity norm (max absolute error)
            error = torch.max(torch.abs(x - x_hat)).item()
            # print(f"[KVTuner:DBG][Q-CHECK] kind={kind.upper()} L{layer_idx} bits={bits} "
            #       f"Quantization L-inf error: {error:.4f}")
        except Exception as e:
            print(f"[KVTuner:DBG][Q-CHECK] ERROR during round-trip check for {kind.upper()} L{layer_idx}: {e}")
        # =========== END: ADD DEBUGGING CODE =============

        flat_q = q_int.reshape(-1).to(torch.int32)
        # V(4-bit) uses local lo->hi nibble packing to match dequant path exactly.
        packed = (_pack4_lohi(flat_q) if (kind == "v" and bits == 4)
                  else pack_nbits(flat_q, bits)).to(torch.uint8)

        meta_bytes = build_kv_meta(
            kind=kind, bits=int(bits), axis=int(meta.axis),
            orig_shape=tuple(int(d) for d in meta.orig_shape),
            group_size=int(meta.group_size), groups=int(meta.groups),
            arrays_dtype=self.model_dtype,
            scale=meta.scale, zp=meta.zp,
        )
        # print(f"[KVTuner] Quantized {kind.upper()} L{layer_idx} -> {bits}-bit, "
        #       f"axis={meta.axis}, groups={meta.groups}, packed_bytes={packed.numel()}, meta_len={len(meta_bytes)}")
        return PackedKV(packed=packed, meta_bytes=meta_bytes)

    def _dequantize_generic(self, obj: PackedKV) -> torch.Tensor:
        md = parse_kv_meta(obj.meta_bytes, device=obj.packed.device)
        bits = int(md["bits"])
        shape = tuple(md["orig_shape"])
        numel = 1
        for d in shape:
            numel *= d
        # V(4-bit) uses local lo->hi nibble unpacking (symmetric to _pack4_lohi).
        q_flat = (_unpack4_lohi(obj.packed.to(torch.uint8), numel)
                  if (md.get("kind", "k") == "v" and bits == 4)
                  else unpack_nbits(obj.packed.to(torch.uint8), bits, numel))
        q_int = q_flat.view(*shape).to(torch.int32)
        qm = QuantMeta(
            axis=int(md["axis"]),
            orig_shape=shape,
            group_size=int(md["group_size"]),
            groups=int(md["groups"]),
            scale=md["scale"],
            zp=md["zp"],
        )
        out = KVTunerMath.dequantize(q_int, bits, qm, model_dtype=self.model_dtype)
        return out
    

# ---------------- Self-check main ----------------
'''Usage example:
python -m vllm.v1.core.kvtuner_quantizer \
  --kvtuner_config_path ~/2025F-Self-Directed-Research/scripts/configs/layer_quantization_config.json \
  --dtype bf16 --device cuda --seed 0 \
  --shape_k 1,128,8,128 --shape_v 128,8,128
'''
if __name__ == "__main__":
    import argparse, json, os, ast
    import torch

    ap = argparse.ArgumentParser(
        description="KVTuner pack/meta/depack self-check with external config.")
    ap.add_argument("--kvtuner_config_path", type=str, default=None,
                    help="Path to kvtuner_config JSON file.")
    ap.add_argument("--kvtuner_config", type=str, default=None,
                    help="Inline JSON (or Python dict literal) for kvtuner_config.")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16",
                    help="Model dtype for quant/dequant.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--shape_k", type=str, default="2,70,19,64",
                    help="Comma-separated shape for test K (e.g., 2,70,19,64).")
    ap.add_argument("--shape_v", type=str, default="23,3,80",
                    help="Comma-separated shape for test V (e.g., 23,3,80).")
    args = ap.parse_args()

    # 1) Load config from path / inline / ENV / fallback
    cfg = None
    if args.kvtuner_config_path is not None:
        with open(args.kvtuner_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    elif args.kvtuner_config is not None:
        s = args.kvtuner_config.strip()
        try:
            cfg = json.loads(s)
        except json.JSONDecodeError:
            # allow Python dict literal too
            cfg = ast.literal_eval(s)
    elif "KVTUNER_CONFIG" in os.environ:
        s = os.environ["KVTUNER_CONFIG"]
        try:
            cfg = json.loads(s)
        except json.JSONDecodeError:
            cfg = ast.literal_eval(s)
    else:
        # tiny fallback for convenience
        cfg = {
            "groups": {"per-token-asym": [[0]]},
            "group_choice": [0],
            "bit_candidates": [4],
            "max_per_layer_scale": 2.0,
        }

    # 2) Dtype/device/shapes
    model_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(args.device)
    shape_k = tuple(int(x) for x in args.shape_k.split(",") if x)
    shape_v = tuple(int(x) for x in args.shape_v.split(",") if x)

    # 3) Seed & tensors
    torch.manual_seed(args.seed)
    Xk = (torch.randn(*shape_k, dtype=model_dtype, device=device) * 0.5).contiguous()
    Xv = (torch.randn(*shape_v, dtype=model_dtype, device=device) * 0.5).contiguous()

    # 4) Run
    qtz = KVTunerQuantizer(cfg, model_dtype=model_dtype)
    pk = qtz.quantize_k(0, Xk)
    pv = qtz.quantize_v(0, Xv)
    Xk_hat = qtz.dequantize_k(pk)
    Xv_hat = qtz.dequantize_v(pv)

    # 5) Checks
    assert Xk_hat.shape == Xk.shape and Xv_hat.shape == Xv.shape, "shape mismatch"
    linf = (Xk_hat - Xk).abs().max().item() + (Xv_hat - Xv).abs().max().item()
    print(f"[KVTuner] pack/meta/depack OK, Linf sum≈{linf:.3e}")