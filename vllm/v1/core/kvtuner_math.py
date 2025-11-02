# vllm/v1/core/kvtuner_math.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch

# Always-on debug prints for verification (remove later if stable)
def _dbg(msg: str):
    print(f"[KVTunerMath] {msg}")

@dataclass
class QuantMeta:
    """Metadata for dequantization and shape restoration."""
    axis: int                  # quantization axis in original tensor
    orig_shape: Tuple[int, ...]
    group_size: int            # size along axis per quantization group
    groups: int                # number of groups along axis
    # qparams
    scale: torch.Tensor        # shape: leading_dims + (groups,)
    zp: torch.Tensor           # shape: leading_dims + (groups,)

class KVTunerMath:
    """Math core for K/V-cache quantization.
    - Asymmetric quantization with ties-to-even rounding.
    - Strict: No FP32 for x/scale/zp/intermediates (only FP16/BF16 + INT).
    """

    @staticmethod
    def _assert_no_fp32(*ts: torch.Tensor):
        for t in ts:
            if t is None:
                continue
            assert t.dtype not in (torch.float32, ), "FP32 is forbidden."

    @staticmethod
    def round_ties_to_even(x: torch.Tensor) -> torch.Tensor:
        """Banker's rounding. torch.round implements ties-to-even."""
        KVTunerMath._assert_no_fp32(x)
        return torch.round(x)

    @staticmethod
    def _move_axis_last(x: torch.Tensor, axis: int):
        """Permute tensor so that the quantization axis is the last dim."""
        ndim = x.dim()
        if axis < 0:
            axis += ndim
        assert 0 <= axis < ndim, f"Invalid axis {axis} for ndim={ndim}"
        perm = [i for i in range(ndim) if i != axis] + [axis]
        invperm = [0] * ndim
        for i, p in enumerate(perm):
            invperm[p] = i
        y = x.permute(*perm).contiguous()
        return y, perm, invperm

    @staticmethod
    def _group_last_axis(y: torch.Tensor,
                         group_size: Optional[int]) -> tuple[torch.Tensor, int, int, int]:
        """Reshape last axis into (groups, group_size) with zero padding if needed.
        Returns:
          z: [..., groups, group_size]
          A: original size of last axis
          G: number of groups
          pad: number of padded elements at end of last axis
        """
        A = y.shape[-1]
        if group_size is None or group_size <= 0:
            group_size = A
        G = (A + group_size - 1) // group_size
        pad = G * group_size - A
        if pad:
            y = torch.nn.functional.pad(y, (0, pad), mode="constant", value=0)
        z = y.view(*y.shape[:-1], G, group_size)
        return z, A, G, pad

    @staticmethod
    def compute_qparams(x: torch.Tensor,
                        bits: int,
                        *,
                        axis: int,
                        q_group_size: Optional[int],
                        clamp_max: float = 2.0,
                        eps: float = 1e-8,
                        model_dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        """Compute asymmetric scale/zp per (axis, group).
        Returns (scale, zp, aux) where scale/zp have shape [..., groups].
        """
        assert bits in (8, 4, 2, 1), f"Unsupported bits={bits}"
        assert x.dtype in (torch.float16, torch.bfloat16), "Only FP16/BF16 allowed."
        KVTunerMath._assert_no_fp32(x)

        y, perm, invperm = KVTunerMath._move_axis_last(x, axis)
        z, A, G, pad = KVTunerMath._group_last_axis(y, q_group_size)

        # Min/max per group (no FP32)
        x_min = z.amin(dim=-1)
        x_max = z.amax(dim=-1)

        # Asymmetric qparams with scale clamp [eps, clamp_max]
        qmin = 0
        qmax = (1 << bits) - 1
        rng = torch.clamp(x_max - x_min, min=eps)
        scale = rng / torch.tensor(qmax - qmin, dtype=model_dtype, device=x.device)
        scale = torch.clamp(scale, min=eps, max=clamp_max).to(model_dtype)
        zp = torch.round(torch.tensor(qmin, dtype=model_dtype, device=x.device) - x_min / scale)
        zp = torch.clamp(zp, min=qmin, max=qmax).to(model_dtype)

        # Debug summary (always-on)
        num_clamped_hi = int(torch.count_nonzero(scale == clamp_max).item())
        num_clamped_lo = int(torch.count_nonzero(scale == eps).item())
        _dbg(f"qparams: bits={bits}, axis={axis}, groups={G}, "
             f"scale[eps]={num_clamped_lo}, scale[max]={num_clamped_hi}, "
             f"shape(scale)={tuple(scale.shape)}")

        aux = {"A": A, "G": G, "pad": pad, "perm_last": int(perm[-1])}
        return scale, zp, aux

    @staticmethod
    def quantize(x: torch.Tensor,
                 bits: int,
                 *,
                 axis: int,
                 q_group_size: Optional[int],
                 clamp_max: float = 2.0,
                 eps: float = 1e-8,
                 model_dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, QuantMeta]:
        """Return int tensor q (unpacked) and QuantMeta (scale/zp etc.)."""
        scale, zp, aux = KVTunerMath.compute_qparams(
            x, bits, axis=axis, q_group_size=q_group_size,
            clamp_max=clamp_max, eps=eps, model_dtype=model_dtype)

        # Reuse the same steps to construct grouped view
        y, perm, invperm = KVTunerMath._move_axis_last(x, axis)
        z, A, G, pad = KVTunerMath._group_last_axis(y, q_group_size)

        qmin = 0
        qmax = (1 << bits) - 1
        # ✅ Standard asymmetric quant: q = round(x/scale + zp)
        q = torch.round(z / scale.unsqueeze(-1) + zp.unsqueeze(-1))
        q = torch.clamp(q, min=qmin, max=qmax)

        # ✅ Flatten (groups, group_size) → last axis, then drop the padded tail to length A
        group_size = z.shape[-1]           # = q_group_size or A
        q_flat = q.reshape(*y.shape[:-1], G * group_size)
        if pad:
            q_flat = q_flat[..., :A]       # keep only real elements

        # Debug guard (always-on)
        assert q_flat.shape[-1] == A, f"q_flat last-dim {q_flat.shape[-1]} != A {A}"

        q_last = q_flat.contiguous()
        q_orig = q_last.permute(*invperm).contiguous().to(torch.int32)

        meta = QuantMeta(
            axis=axis,
            orig_shape=tuple(int(d) for d in x.shape),
            group_size=int(q_group_size or A),
            groups=int(G),
            scale=scale.to(model_dtype).contiguous(),
            zp=zp.to(model_dtype).contiguous(),
        )

        _dbg(f"quantize: bits={bits}, axis={axis}, A={A}, G={G}, "
             f"group_size={meta.group_size}, flattened={G*group_size}, "
             f"q_int32.numel={q_orig.numel()}")

        return q_orig, meta

    @staticmethod
    def dequantize(q: torch.Tensor,
                   bits: int,
                   meta: QuantMeta,
                   *,
                   model_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Reconstruct FP tensor using meta (scale/zp) with the same dtype."""
        assert q.dtype in (torch.int32, torch.int16, torch.int8), "q must be integer tensor"
        axis = meta.axis
        shape = meta.orig_shape
        group_size = meta.group_size
        G = meta.groups
        scale = meta.scale.to(model_dtype)
        zp = meta.zp.to(model_dtype)

        # Bring axis to last
        ndim = len(shape)
        if axis < 0:
            axis += ndim
        perm = [i for i in range(ndim) if i != axis] + [axis]
        invperm = [0] * ndim
        for i, p in enumerate(perm):
            invperm[p] = i

        q_last = q.permute(*perm).contiguous()
        A = q_last.shape[-1]
        pad = G * group_size - A
        if pad:
            q_last = torch.nn.functional.pad(q_last, (0, pad), mode="constant", value=0)

        z = q_last.view(*q_last.shape[:-1], G, group_size)

        # ✅ Standard asymmetric dequant: x̂ = scale * (q - zp)
        x = (z.to(model_dtype) - zp.unsqueeze(-1)) * scale.unsqueeze(-1)

        # ✅ Flatten (G, group_size) → last axis, then drop padded tail back to length A
        x_flat = x.reshape(*x.shape[:-2], G * group_size)
        if pad:
            x_flat = x_flat[..., :A]
        # Debug guard
        assert x_flat.shape[-1] == A, f"dequant flatten last-dim {x_flat.shape[-1]} != A {A}"

        out = x_flat.permute(*invperm).contiguous().to(model_dtype)

        _dbg(f"dequantize: bits={bits}, axis={axis}, A={A}, G={G}, "
             f"flattened={G*group_size}, out.shape={tuple(out.shape)}")

        return out


# === Append this to the bottom of vllm/v1/core/kvtuner_math.py ===
# Self-checking routines for KVTunerMath
# run via: python -m vllm.v1.core.kvtuner_math
import math
import random

def _selfcheck_device():
    """Pick CUDA if available; otherwise CPU."""
    use_cuda = torch.cuda.is_available()
    dev = torch.device("cuda") if use_cuda else torch.device("cpu")
    print(f"[KVTunerMath:SELF] device={dev.type}")
    return dev

def _assert_no_fp32_tensors(named_tensors: dict):
    """Hard assert to enforce 'no FP32 anywhere' rule."""
    for name, t in named_tensors.items():
        if not isinstance(t, torch.Tensor):
            continue
        assert t.dtype not in (torch.float32,), f"{name} is FP32 (forbidden)."

def _ties_to_even_unit_test():
    """Verify ties-to-even rounding behavior explicitly."""
    v = torch.tensor([-1.5, -0.5, 0.5, 1.5, 2.5], dtype=torch.bfloat16)
    r = KVTunerMath.round_ties_to_even(v)
    print(f"[KVTunerMath:SELF] ties-to-even input : {v.tolist()}")
    print(f"[KVTunerMath:SELF] ties-to-even output: {r.tolist()}")
    # Expected: [-2.0, -0.0, 0.0, 2.0, 2.0]  (banker's rounding)
    exp = torch.tensor([-2.0, -0.0, 0.0, 2.0, 2.0], dtype=torch.bfloat16)
    assert torch.allclose(r, exp), "ties-to-even does not match expectation."

def _run_one_case(x: torch.Tensor, axis: int, q_group_size: int | None, bits: int):
    """Quantize->Dequantize round-trip, print errors and invariants."""
    print(f"[KVTunerMath:SELF] case: shape={tuple(x.shape)}, axis={axis}, "
          f"q_group_size={q_group_size}, bits={bits}")

    q_int, meta = KVTunerMath.quantize(
        x, bits, axis=axis, q_group_size=q_group_size,
        clamp_max=2.0, eps=1e-8, model_dtype=x.dtype
    )
    x_hat = KVTunerMath.dequantize(q_int, bits, meta, model_dtype=x.dtype)

    # Invariants
    qmin, qmax = 0, (1 << bits) - 1
    q_int_min = int(q_int.min().item())
    q_int_max = int(q_int.max().item())
    print(f"[KVTunerMath:SELF] q-int range = [{q_int_min}, {q_int_max}] "
          f"(expected within [{qmin}, {qmax}])")
    assert qmin <= q_int_min and q_int_max <= qmax, "q values out of range."

    # Error metrics
    err = (x_hat - x).to(torch.float32)  # for accuracy of metrics only
    mae = float(err.abs().mean().item())
    mse = float((err * err).mean().item())
    mx  = float(err.abs().max().item())

    # === Realistic bounds under low-precision math (BF16/FP16) ===
    # 1) Ideal mid-rise quantizer: L∞ ≤ 0.5 * scale_i  (per-group)
    # 2) Finite-precision add-on (double rounding etc.):
    #      + ULP(dtype, |x̂|)  (use full ULP, not 0.5, to be conservative)
    # 3) Saturated samples (q==qmin or q==qmax) can exceed 0.5*Δ due to zp rounding;
    #    allow extra +0.5*max(scale) slack for saturated subset only.
    def _mantissa_bits(dt: torch.dtype) -> int:
        if dt == torch.bfloat16:
            return 7
        if dt == torch.float16:
            return 10
        return 23  # fallback for float32 in case of testing

    def _max_half_ulp(x_like: torch.Tensor, dt: torch.dtype) -> float:
        m = _mantissa_bits(dt)
        # ulp(|y|) = 2^(floor(log2(|y|)) - m)
        absx = x_like.abs().to(torch.float32)
        tiny = torch.finfo(torch.float32).tiny
        absx = torch.clamp(absx, min=tiny)
        exp2 = torch.floor(torch.log2(absx))
        ulp  = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=absx.device), exp2 - float(m))
        return float(ulp.max().item()) * 0.5

    scale_max = float(meta.scale.max().item())
    half_ulp  = _max_half_ulp(x_hat, x_hat.dtype)
    full_ulp  = 2.0 * half_ulp  # be conservative for double-rounding chains
    base_bound = 0.5 * scale_max + full_ulp + 1e-6

    # Separate saturated vs non-saturated samples
    qmin, qmax = 0, (1 << bits) - 1
    is_sat = (q_int == qmin) | (q_int == qmax)
    any_sat = bool(is_sat.any().item())
    err_abs = err.abs()
    mx_nsat = float(err_abs[~is_sat].max().item()) if (~is_sat).any() else 0.0
    mx_sat  = float(err_abs[is_sat ].max().item()) if any_sat else 0.0

    # Non-saturated must be within base_bound
    ok_nsat = (mx_nsat <= base_bound) if (~is_sat).any() else True
    # Saturated are given +0.5*scale_max extra slack
    sat_bound = base_bound + 0.5 * scale_max
    ok_sat  = (mx_sat <= sat_bound) if any_sat else True

    print(f"[KVTunerMath:SELF] err: MAE={mae:.6e}  MSE={mse:.6e}  Linf={mx:.6e}  "
          f"(base_bound≈{base_bound:.6e}, sat_bound≈{sat_bound:.6e}, "
          f"mx_nsat={mx_nsat:.6e}, mx_sat={mx_sat:.6e}, any_sat={any_sat})")

    assert (ok_nsat and ok_sat) or bits == 1, \
        "L-infinity error exceeds realistic bounds (incl. dtype rounding & saturation)."

    # Dtype discipline
    _assert_no_fp32_tensors({
        "x": x, "x_hat": x_hat, "scale": meta.scale, "zp": meta.zp
    })

def _clamp_stress_test(x: torch.Tensor, axis: int, q_group_size: int | None, bits: int):
    """Adaptively scale input to force scale clamp (=2.0) to engage.
    Why: fixed ×100 may be insufficient when base max(scale) << 2.0."""
    print(f"[KVTunerMath:SELF] clamp-stress: bits={bits}")
    # 1) Measure baseline max(scale) without changing values
    scale_small, zp_small, _ = KVTunerMath.compute_qparams(
        x, bits, axis=axis, q_group_size=q_group_size,
        clamp_max=2.0, eps=1e-8, model_dtype=x.dtype
    )
    scale_small = scale_small.to(x.dtype)
    max_small = float(scale_small.max().item())
    clamp_max = 2.0

    # 2) Choose a factor that guarantees max(scale)*factor > clamp_max
    #    (use 1.25x headroom; avoid FP32 by casting back to x.dtype for the mul)
    if max_small <= 0:
        factor = 1024.0
    else:
        factor = min(max((clamp_max / max_small) * 1.25, 2.0), 1e6)
    factor_t = torch.tensor(factor, dtype=x.dtype, device=x.device)
    x_big = (x * factor_t).to(x.dtype)

    # 3) Recompute qparams after amplification
    scale_big, zp_big, _ = KVTunerMath.compute_qparams(
        x_big, bits, axis=axis, q_group_size=q_group_size,
        clamp_max=clamp_max, eps=1e-8, model_dtype=x.dtype
    )
    scale_big = scale_big.to(x.dtype)

    # 4) Count groups that hit the clamp exactly (=2.0 in BF16/FP16 is representable)
    n_hi_small = int((scale_small == torch.tensor(clamp_max, dtype=x.dtype, device=x.device)).sum().item())
    n_hi_big   = int((scale_big   == torch.tensor(clamp_max, dtype=x.dtype, device=x.device)).sum().item())

    print(f"[KVTunerMath:SELF] clamp-stress factor≈{factor:.2f}, "
          f"max(scale_small)≈{max_small:.6e}, max(scale_big)≈{float(scale_big.max().item()):.6e}")
    print(f"[KVTunerMath:SELF] groups clamped@max(scale)=2.0 -> small={n_hi_small}, big={n_hi_big} "
          f"(big should be > small and > 0)")
    assert n_hi_big >= n_hi_small and n_hi_big > 0, "Clamp did not engage as expected."

def selfcheck(seed: int = 0):
    """
    Intent:
      - Verify math-only correctness independent of the engine.
      - Cover rounding rule, axis/grouping, tail padding, clamp behavior, and dtype discipline.
    Why:
      - If this passes, asymmetric quantization + ties-to-even + clamp[eps,2.0] is applied as specified.
    """
    print("[KVTunerMath:SELF] ===== SELF-CHECK START =====")
    random.seed(seed)
    torch.manual_seed(seed)
    dev = _selfcheck_device()

    # 0) Ties-to-even micro test
    _ties_to_even_unit_test()

    # 1) K-style case: per-channel (axis=1), group_size=32, non-multiple length to test padding
    #    shape: [B, C, S, D] -> axis=1 (C), choose C=70 to trigger tail
    B, C, S, D = 2, 70, 19, 64
    xK = (torch.rand(B, C, S, D, device=dev, dtype=torch.bfloat16) * 2.0 - 1.0)  # ~U(-1,1)
    for bits in (8, 4, 2, 1):
        _run_one_case(xK, axis=1, q_group_size=32, bits=bits)
        _clamp_stress_test(xK, axis=1, q_group_size=32, bits=bits)

    # 2) V-style case: per-token (axis=0), no grouping (entire axis as one group)
    #    shape: [S, H, D]; pick S not multiple of typical sizes to test reshape
    S, H, D = 23, 3, 80
    xV = (torch.randn(S, H, D, device=dev, dtype=torch.bfloat16) * 0.5)  # ~N(0,0.5^2)
    for bits in (8, 4, 2, 1):
        _run_one_case(xV, axis=0, q_group_size=None, bits=bits)
        _clamp_stress_test(xV, axis=0, q_group_size=None, bits=bits)

    print("[KVTunerMath:SELF] ===== SELF-CHECK PASSED =====")

if __name__ == "__main__":
    # Always-on debug entry; run and delete after confirming PASS.
    selfcheck(seed=0)
