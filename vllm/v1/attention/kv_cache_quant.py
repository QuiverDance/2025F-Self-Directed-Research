# vllm/v1/attention/kv_cache_quant.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import torch
import math
from vllm.v1.metrics.kv_quant import KVQuantLayerRecord
# -------------------------
# Helpers for (de)quant
# -------------------------
T_TILE = 256  # T-tiling for memory-safe quantization

def _cuda_free_bytes(device="cuda"):
    import torch
    free, total = torch.cuda.mem_get_info()
    return int(free)

def _mb(x): return int(x) >> 20

def _iter_covering(chunks, start: int, stop: int):
    """Yield (idx, off, take, dst) covering [start:stop) over a list of T-major chunks.
       dst: destination start index inside the requested slice buffer (0-based).
    """
    t = 0
    filled = 0
    for i, ch in enumerate(chunks):
        clen = int(ch.shape[0])
        if t + clen <= start:
            t += clen
            continue
        if t >= stop:
            break
        off  = max(0, start - t)
        take = min(clen - off, stop - (t + off))
        yield i, off, take, filled
        filled += take
        t += clen

def _amax_channelwise(x: torch.Tensor, dim: int) -> torch.Tensor:
    # amax along specified dim (keepdim=True for broadcasting)
    return x.abs().amax(dim=dim, keepdim=True).clamp_min_(1e-8)

def _quantize_symmetric(x: torch.Tensor, *, bits: int, group_size: int, mode: str,
                        t_tile: int = T_TILE) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-safe symmetric quantization (all compute in fp16) with T-tiling.
      - mode='symmetric_channel' : time-invariant scale per (H, D-group)
      - mode='symmetric_token'   : per-token scale per (T, H, D-group)
    Returns: (q_packed, scale)
    """
    assert x.dim() == 3 and mode in ("symmetric_channel", "symmetric_token")
    T, H, D = x.shape
    # signed symmetric code range (2's complement for packing)
    if bits == 8:
        qmin, qmax = -128, 127
    elif bits == 4:
        qmin, qmax = -8, 7
    elif bits == 2:
        qmin, qmax = -2, 1
    else:
        raise ValueError(f"Unsupported bits={bits}")

    # group split (Mistral: D=128, group_size=64 -> G=2; Generally D%group_size==0)
    G = (D + group_size - 1) // group_size
    xg = x.view(T, H, G, group_size)

    # output buffers
    if bits == 8:
        q_out = torch.empty((T, H, D), dtype=torch.int8, device=x.device)
    elif bits == 4:
        q_out = torch.empty((T, H, (D + 1) // 2), dtype=torch.uint8, device=x.device)  # nibble packed
    else:
        q_out = torch.empty((T, H, (D + 3) // 4), dtype=torch.uint8, device=x.device)  # 2bit packed

    eps = 1e-8
    if mode == "symmetric_channel":
        # scale: [H,G] (time-invariant)
        amax = xg.abs().amax(dim=(0, 3))                           # [H,G]
        scale = (amax / float(max(abs(qmin), qmax))).clamp_min(eps).to(torch.float16)  # [H,G]
        s = scale.view(1, H, G, 1).to(x.dtype)                     # fp16 broadcast

        for t0 in range(0, T, t_tile):
            t1 = min(T, t0 + t_tile)
            xt = xg[t0:t1]                                         # [t,H,G,gs]
            tmp = xt.div(s)                                        # fp16
            torch.round(tmp, out=tmp)
            tmp.clamp_(qmin, qmax)
            qi8 = tmp.view(t1 - t0, H, G * group_size)[:, :, :D].to(torch.int8)
            if bits == 8:
                q_out[t0:t1] = qi8
            elif bits == 4:
                q_out[t0:t1] = _pack_nibble_signed(qi8)
            else:
                q_out[t0:t1] = _pack_2bit_signed(qi8)

    else:  # symmetric_token
        # scale: [T,H,G] (time-varying). Allocate final scale once
        scale = torch.empty((T, H, G), dtype=torch.float16, device=x.device)
        for t0 in range(0, T, t_tile):
            t1 = min(T, t0 + t_tile)
            xt = xg[t0:t1]                                         # [t,H,G,gs]
            amax_t = xt.abs().amax(dim=3)                          # [t,H,G]
            st = (amax_t / float(max(abs(qmin), qmax))).clamp_min(eps).to(torch.float16)
            scale[t0:t1] = st
            s = st.view(t1 - t0, H, G, 1).to(x.dtype)              # fp16
            tmp = xt.div(s)
            torch.round(tmp, out=tmp)
            tmp.clamp_(qmin, qmax)
            qi8 = tmp.view(t1 - t0, H, G * group_size)[:, :, :D].to(torch.int8)
            if bits == 8:
                q_out[t0:t1] = qi8
            elif bits == 4:
                q_out[t0:t1] = _pack_nibble_signed(qi8)
            else:
                q_out[t0:t1] = _pack_2bit_signed(qi8)

    return q_out.contiguous(), (scale if mode == "symmetric_token" else scale.contiguous())

def _dequantize_symmetric(q_packed, scale: torch.Tensor, *, bits: int,
                          group_size: int, mode: str, T: int, H: int, D: int) -> torch.Tensor:
    """
    Inverse of symmetric quantization.
      - symmetric_channel : scale [H,G]
      - symmetric_token   : scale [T,H,G]
    All math in fp16. Unpackers return int8 then cast to fp16.
    """
    assert mode in ("symmetric_channel", "symmetric_token")
    # unpack → int8
    if bits == 8:
        qi8 = q_packed.view(T, H, D).to(torch.int8)                # [T,H,D]
    elif bits == 4:
        qi8 = _unpack_nibble_signed_i8(q_packed, T, H, D)          # int8 [T,H,D]
    elif bits == 2:
        qi8 = _unpack_2bit_signed_i8(q_packed, T, H, D)            # int8 [T,H,D]
    else:
        raise ValueError(f"Unsupported bits={bits}")

    # group view
    Dp = ((D + group_size - 1) // group_size) * group_size
    G  = Dp // group_size
    if Dp != D:
        # pad tail to full group width (int8)
        qi8 = torch.nn.functional.pad(qi8, (0, Dp - D))
    qg = qi8.view(T, H, G, group_size)                              # int8

    # broadcast scale
    if mode == "symmetric_channel":
        s = scale.view(1, H, G, 1).to(torch.float16)
    else:
        s = scale.view(T, H, G, 1).to(torch.float16)

    x = qg.to(torch.float16) * s
    return x.view(T, H, Dp)[..., :D].contiguous()

def _unpack_nibble_signed_i8(packed: torch.Tensor, T: int, H: int, D: int) -> torch.Tensor:
    """Unpack uint8 -> signed int4 in int8 tensor [-8..7], shape [T,H,D]."""
    b = packed.view(T, H, -1)
    lo = (b & 0x0F).to(torch.int8) - 8
    hi = ((b >> 4) & 0x0F).to(torch.int8) - 8
    out = torch.empty((T, H, b.shape[-1] * 2), dtype=torch.int8, device=packed.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out[..., :D].contiguous()

def _unpack_2bit_signed_i8(packed: torch.Tensor, T: int, H: int, D: int) -> torch.Tensor:
    """Unpack uint8 -> signed 2-bit in int8 tensor [-2..1], shape [T,H,D]."""
    b = packed.view(T, H, -1)
    a0 = ( b        & 0x03).to(torch.int8) - 2
    a1 = ((b >> 2)  & 0x03).to(torch.int8) - 2
    a2 = ((b >> 4)  & 0x03).to(torch.int8) - 2
    a3 = ((b >> 6)  & 0x03).to(torch.int8) - 2
    out = torch.empty((T, H, b.shape[-1] * 4), dtype=torch.int8, device=packed.device)
    out[..., 0::4] = a0
    out[..., 1::4] = a1
    out[..., 2::4] = a2
    out[..., 3::4] = a3
    return out[..., :D].contiguous()


def _asym_qparams(x: torch.Tensor, bits: int, reduce_over_time: bool, group_size: int):
    """
    Compute per-group (scale, zp) for asymmetric quant.
    - If reduce_over_time=True  -> reduce over (T, group_size) ==> per-channel-asym
    - Else (False)              -> reduce over (group_size)     ==> per-token-asym
    Returns: (scale, zp, Dg)
      scale: fp16 (shape [H,Dg] or [T,H,Dg])
      zp   : uint8 (same shape as scale) with range [0, 2^bits-1]
      Dg   : #groups along D (ceil_div(D, group_size))
    """
    import torch
    T, H, D = x.shape
    qmax = (1 << bits) - 1
    Dg = (D + group_size - 1) // group_size

    # reshape to group
    xg = x.view(T, H, Dg, group_size)

    if reduce_over_time:
        x_min = xg.amin(dim=(0, 3))        # [H,Dg]
        x_max = xg.amax(dim=(0, 3))        # [H,Dg]
        reduce_shape = (H, Dg)
    else:
        x_min = xg.amin(dim=3)             # [T,H,Dg]
        x_max = xg.amax(dim=3)             # [T,H,Dg]
        reduce_shape = (T, H, Dg)

    # avoid degenerate ranges
    eps = 1e-8
    scale = (x_max - x_min).clamp_min(eps) / float(qmax)  # fp16 later
    zp = torch.round(-x_min / scale).clamp_(0, qmax).to(torch.uint8)

    return scale.to(torch.float16), zp, Dg, reduce_shape

def _quantize_asymmetric(x: torch.Tensor, *, bits: int, group_size: int,
                         mode: str, t_tile: int = T_TILE) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Memory-safe asymmetric quantization (all math in fp16) with T-tiling.
      - mode='asymmetric_channel' : scale/zp [H,Dg]
      - mode='asymmetric_token'   : scale/zp [T,H,Dg]
    Returns (q_packed, scale(fp16), zp(uint8)).
    """
    assert mode in ("asymmetric_channel", "asymmetric_token")
    T, H, D = x.shape
    Dg = (D + group_size - 1) // group_size
    xg = x.view(T, H, Dg, group_size)

    qmax = (1 << bits) - 1
    eps = 1e-8

    if mode == "asymmetric_channel":
        x_min = xg.amin(dim=(0, 3))                                  # [H,Dg]
        x_max = xg.amax(dim=(0, 3))                                  # [H,Dg]
        scale = ((x_max - x_min).clamp_min(eps) / float(qmax)).to(torch.float16)
        zp    = torch.round((-x_min / scale).clamp_(0, qmax)).to(torch.uint8)

        # outputs
        if bits == 8:
            q_out = torch.empty((T, H, D), dtype=torch.uint8, device=x.device)
        elif bits == 4:
            q_out = torch.empty((T, H, D // 2), dtype=torch.uint8, device=x.device)
        else:
            q_out = torch.empty((T, H, D // 4), dtype=torch.uint8, device=x.device)

        s = scale.view(1, H, Dg, 1).to(x.dtype)                       # fp16
        z = zp.view(1, H, Dg, 1).to(x.dtype)                          # fp16
        for t0 in range(0, T, t_tile):
            t1 = min(T, t0 + t_tile)
            xt = xg[t0:t1]
            tmp = xt.div(s).add_(z)                                   # fp16
            torch.round(tmp, out=tmp)
            tmp.clamp_(0, qmax)
            qt = tmp.view(t1 - t0, H, D).to(torch.uint8)
            if bits == 8:
                q_out[t0:t1] = qt
            elif bits == 4:
                q_out[t0:t1] = _pack_nibble_u4(qt)
            else:
                q_out[t0:t1] = _pack_2bit_u2(qt)

        return q_out.contiguous(), scale.contiguous(), zp.contiguous()

    else:
        # per-token stats
        x_min = xg.amin(dim=3)                                        # [T,H,Dg]
        x_max = xg.amax(dim=3)                                        # [T,H,Dg]
        scale = ((x_max - x_min).clamp_min(eps) / float(qmax)).to(torch.float16)
        zp    = torch.round((-x_min / scale).clamp_(0, qmax)).to(torch.uint8)

        if bits == 8:
            q_out = torch.empty((T, H, D), dtype=torch.uint8, device=x.device)
        elif bits == 4:
            q_out = torch.empty((T, H, D // 2), dtype=torch.uint8, device=x.device)
        else:
            q_out = torch.empty((T, H, D // 4), dtype=torch.uint8, device=x.device)

        for t0 in range(0, T, t_tile):
            t1 = min(T, t0 + t_tile)
            xt = xg[t0:t1]
            st = scale[t0:t1].view(t1 - t0, H, Dg, 1).to(x.dtype)
            zt = zp[t0:t1].view(t1 - t0, H, Dg, 1).to(x.dtype)
            tmp = xt.div(st).add_(zt)
            torch.round(tmp, out=tmp)
            tmp.clamp_(0, qmax)
            qt = tmp.view(t1 - t0, H, D).to(torch.uint8)
            if bits == 8:
                q_out[t0:t1] = qt
            elif bits == 4:
                q_out[t0:t1] = _pack_nibble_u4(qt)
            else:
                q_out[t0:t1] = _pack_2bit_u2(qt)

        return q_out.contiguous(), scale.contiguous(), zp.contiguous()

def _dequantize_asymmetric(q_packed: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor,
                           *, bits: int, group_size: int, mode: str,
                           T: int, H: int, D: int) -> torch.Tensor:
    """
    Inverse of _quantize_asymmetric. All math in fp16.
    """
    assert mode in ("asymmetric_channel", "asymmetric_token")
    # unpack to uint8 then fp16
    if bits == 8:
        q = q_packed
    elif bits == 4:
        q = _unpack_nibble_u4(q_packed, D)
    elif bits == 2:
        q = _unpack_2bit_u2(q_packed, D)
    else:
        raise ValueError(f"Unsupported bits={bits}")

    Dg = (D + group_size - 1) // group_size
    qf = q.view(T, H, Dg, group_size).to(torch.float16)

    if mode == "asymmetric_channel":
        s = scale.view(1, H, Dg, 1).to(torch.float16)
        z = zp.view(1, H, Dg, 1).to(torch.float16)
    else:
        s = scale.view(T, H, Dg, 1).to(torch.float16)
        z = zp.view(T, H, Dg, 1).to(torch.float16)

    x = (qf - z) * s
    return x.view(T, H, D).contiguous()
    
# ----- packing utils (signed) -----

def _pack_nibble_signed(q_int8: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values (range [-8,7]) into uint8 (2 per byte).
    Assumes last dim packs by pairs. Returns [T,H,D//2] uint8."""
    # shift by +8 to make unsigned nibble [0,15]
    u = (q_int8 + 8).to(torch.uint8)
    T, H, D = u.shape
    if D % 2 != 0:
        u = torch.nn.functional.pad(u, (0, 1))
        D += 1
    u = u.view(T, H, D // 2, 2)
    packed = (u[..., 0] & 0x0F) | ((u[..., 1] & 0x0F) << 4)
    return packed.contiguous()

def _unpack_nibble_signed(packed: torch.Tensor, T: int, H: int, D: int) -> torch.Tensor:
    """Unpack uint8 -> int4 signed in float32."""
    lo = (packed & 0x0F).to(torch.int16) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    out = torch.empty((T, H, (D + (D % 2))), dtype=torch.int16, device=packed.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out[..., :D].to(torch.float32)

def _pack_2bit_signed(q_int8: torch.Tensor) -> torch.Tensor:
    """Pack signed 2-bit values (range [-2,1]) into uint8 (4 per byte)."""
    u = (q_int8 + 2).to(torch.uint8)  # [0..3]
    T, H, D = u.shape
    rem = D % 4
    if rem != 0:
        u = torch.nn.functional.pad(u, (0, 4 - rem))
        D += (4 - rem)
    u = u.view(T, H, D // 4, 4)
    packed = (u[..., 0] & 0x03) | ((u[..., 1] & 0x03) << 2) | ((u[..., 2] & 0x03) << 4) | ((u[..., 3] & 0x03) << 6)
    return packed.contiguous()

def _unpack_2bit_signed(packed: torch.Tensor, T: int, H: int, D: int) -> torch.Tensor:
    """Unpack uint8 -> 2bit signed float32."""
    b = packed
    a0 = (b & 0x03).to(torch.int16) - 2
    a1 = ((b >> 2) & 0x03).to(torch.int16) - 2
    a2 = ((b >> 4) & 0x03).to(torch.int16) - 2
    a3 = ((b >> 6) & 0x03).to(torch.int16) - 2
    out = torch.empty((packed.shape[0], packed.shape[1], D), dtype=torch.int16, device=packed.device)
    out[..., 0::4] = a0
    out[..., 1::4] = a1
    out[..., 2::4] = a2
    out[..., 3::4] = a3
    return out.to(torch.float32)

# ---------- Unsigned nibble/2bit packers for asymmetric quant ----------

def _pack_nibble_u4(x: torch.Tensor) -> torch.Tensor:
    """Pack uint4 (0..15) pairs into uint8. Last dim must be even."""
    assert x.dtype == torch.uint8
    T, H, D = x.shape
    assert D % 2 == 0, "D must be even for u4 pack"
    x = x.view(T, H, D // 2, 2)
    hi = (x[..., 0] & 0x0F) << 4
    lo = (x[..., 1] & 0x0F)
    out = (hi | lo).contiguous()
    return out.view(T, H, D // 2)

def _unpack_nibble_u4(x: torch.Tensor, D: int) -> torch.Tensor:
    """Unpack uint8 into uint4 along last dim."""
    assert x.dtype == torch.uint8
    T, H, D2 = x.shape
    assert D2 * 2 == D, "Mismatched D for u4 unpack"
    x = x.view(T, H, D2, 1)
    hi = (x[..., 0] >> 4) & 0x0F
    lo = x[..., 0] & 0x0F
    out = torch.stack([hi, lo], dim=-1).reshape(T, H, D)
    return out.contiguous()

def _pack_2bit_u2(x: torch.Tensor) -> torch.Tensor:
    """Pack uint2 (0..3) quadruples into uint8."""
    assert x.dtype == torch.uint8
    T, H, D = x.shape
    assert D % 4 == 0, "D must be multiple of 4 for u2 pack"
    x = x.view(T, H, D // 4, 4)
    out = ( (x[..., 0] & 0x03)
          | ((x[..., 1] & 0x03) << 2)
          | ((x[..., 2] & 0x03) << 4)
          | ((x[..., 3] & 0x03) << 6) )
    return out.contiguous().view(T, H, D // 4)

def _unpack_2bit_u2(x: torch.Tensor, D: int) -> torch.Tensor:
    """Unpack uint8 into 4x uint2 along last dim."""
    assert x.dtype == torch.uint8
    T, H, D4 = x.shape
    assert D4 * 4 == D, "Mismatched D for u2 unpack"
    b = x.view(T, H, D4, 1)[..., 0]
    a0 =  b        & 0x03
    a1 = (b >> 2)  & 0x03
    a2 = (b >> 4)  & 0x03
    a3 = (b >> 6)  & 0x03
    out = torch.stack([a0, a1, a2, a3], dim=-1).reshape(T, H, D)
    return out.contiguous()

def _infer_teff_from_padding(K: torch.Tensor, *, every_n: int = 1) -> int:
    """Heuristically infer effective T (#valid tokens) by detecting trailing
    all-zero rows along time axis. Works when compiler pads to a fixed T (e.g., 8192).
    Cost: O(T*H*D) read once per layer at first prefill (and rare repeats)."""
    # K: [T, H, D] on CUDA. We only need to know which rows are all-zeros.
    # Downsample over time if needed (every_n>1) for very long T, then refine.
    T = K.shape[0]
    # Fast path: reduce over feature dims → [T]
    # NOTE: use fp16->fp32 reduction to avoid overflow/underflow issues.
    flat = K.abs().reshape(T, -1).sum(dim=1)  # [T], fp16->fp16 sum is fine here
    nz = (flat != 0)
    if not bool(nz.any()):
        return 0
    teff = int(nz.nonzero()[-1].item()) + 1  # last nonzero + 1
    return teff

# -------------------------
# Quantized KV cache
# -------------------------

@dataclass
class LayerKVStore:
    bits_k: int
    bits_v: int
    group_size: int
    mode_k: str = "symmetric"
    mode_v: str = "symmetric"
    # packed storage and scales
    K_packed: Optional[torch.Tensor] = None
    V_packed: Optional[torch.Tensor] = None
    K_scale: Optional[torch.Tensor] = None
    V_scale: Optional[torch.Tensor] = None
    K_zp: Optional[torch.Tensor] = None   # uint8, shape depends on mode_k
    V_zp: Optional[torch.Tensor] = None   # uint8, shape depends on mode_v
    T: int = 0
    H: int = 0
    D: int = 0
    K_chunks: list[torch.Tensor] = field(default_factory=list)
    V_chunks: list[torch.Tensor] = field(default_factory=list)
    Ks_chunks: list[torch.Tensor] = field(default_factory=list)  # for per_token_head
    Vs_chunks: list[torch.Tensor] = field(default_factory=list)  # for per_token_head

class PagedKVCacheQuantized:
    """Reference quantized KV cache that packs K/V per layer with given policy.
    This implementation keeps a contiguous [T,H,D] history per layer for clarity.
    Replace with paged allocator bindings for production.
    """

    def __init__(self, n_layers: int, policies, device: torch.device, debug: bool = False, debug_interval: int = 128):
        self.device = device
        self.layers: Dict[int, LayerKVStore] = {}
        self.policies = policies  # callable: layer_idx -> LayerPolicy

        self.lowmem_guard_bytes = 768 << 20   # 768 MiB threshold
        self.min_t_tile         = 64          # TILE under limit

        # stats
        self.bytes_total = 0
        self.bytes_scales = 0
        self.bytes_zp = 0

        self.debug = debug
        self.debug_interval = max(1, int(debug_interval))

        for li in range(n_layers):
            p = policies(li)
            self.layers[li] = LayerKVStore(
                bits_k=p.bits_k,
                bits_v=p.bits_v,
                group_size=p.group_size,
                mode_k=getattr(p, "mode_k", "asymmetric_channel"),
                mode_v=getattr(p, "mode_v", "asymmetric_token"),
            )

    @staticmethod
    def _pack_bits(bits: int, q: torch.Tensor, scale: torch.Tensor, is_k: bool, D: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return q, scale

    def append_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Append one step of K,V: accepts [H,D] or [T,H,D]; always uses _append_core."""
        st = self.layers[layer_idx]
        if k.dim() == 2:
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        if self.debug and (st.T == 0 or st.T % self.debug_interval == 0):
            print(f"[KVQDBG] L{layer_idx} append@T={st.T}: incoming {tuple(k.shape)}", flush=True)
        self._append_core(layer_idx, k.contiguous(), v.contiguous())
        if self.debug:
            print(_cuda_mem_snapshot(f"after-append L{layer_idx}"), flush=True)

    def append_kv_prefill(self, layer_idx: int, K: torch.Tensor, V: torch.Tensor) -> None:
        """
        Append-only for prefill phase with padding-aware effective length.
        """
        st = self.layers[layer_idx]
        T_in = int(K.shape[0])
        # approximate T_eff by checking trailing all-zero rows in K
        T_eff = _infer_teff_from_padding(K)
        if self.debug:
            print(f"[KVQDBG] L{layer_idx} PREFILL infer T_eff={T_eff} (T_in={T_in})", flush=True)
        if T_eff == 0:
            return  # nothing to store
        # if already have >= T_eff, skip
        T_have = int(st.T)
        if T_eff <= T_have:
            if self.debug:
                print(f"[KVQDBG] L{layer_idx} PREFILL skip (dup): have={T_have}, incoming={T_eff}", flush=True)
            return
        # save K,V from T_have to T_eff
        K_use = K[T_have:T_eff].contiguous()
        V_use = V[T_have:T_eff].contiguous()
        self._append_core(layer_idx, K_use, V_use)

    def _append_core(self, layer_idx: int, K: torch.Tensor, V: torch.Tensor) -> None:
        """Quantize & append K/V (time-axis) into per-layer packed storage (CHUNKED)."""
        import torch, math
        st = self.layers[layer_idx]

        # --- validate & init H/D ---
        K = K.contiguous(); V = V.contiguous()
        T_new, H_in, D_in = K.shape
        if int(getattr(st, "H", 0) or 0) > 0 and int(getattr(st, "D", 0) or 0) > 0:
            assert (H_in, D_in) == (st.H, st.D), f"Shape mismatch: got ({H_in},{D_in}) expected ({st.H},{st.D})"
        else:
            st.H, st.D = int(H_in), int(D_in)

        # ---- Quantize & pack ----
        # K
        mk = getattr(st, "mode_k", "symmetric_channel")
        if mk == "asymmetric_channel":
            qk, sk, zpk = _quantize_asymmetric(K, bits=st.bits_k, group_size=st.group_size,
                                            mode="asymmetric_channel", t_tile=T_TILE)
            # per-channel stats stored once
            if st.K_scale is None: st.K_scale = sk.contiguous()
            if st.K_zp    is None: st.K_zp    = zpk.contiguous()
            sk_to_store, zpk_to_store = None, None  # don't store per-chunk
        elif mk == "asymmetric_token":
            qk, sk, zpk = _quantize_asymmetric(K, bits=st.bits_k, group_size=st.group_size,
                                            mode="asymmetric_token", t_tile=T_TILE)
            sk_to_store, zpk_to_store = sk, zpk     # store per-chunk
        elif mk in ("symmetric_channel"):
            qk, sk = _quantize_symmetric(K, bits=st.bits_k, group_size=st.group_size,
                                        mode="symmetric_channel", t_tile=T_TILE)
            zpk, sk_to_store, zpk_to_store = None, None, None
            if st.K_scale is None: st.K_scale = sk.contiguous()
        elif mk == "symmetric_token":
            qk, sk = _quantize_symmetric(K, bits=st.bits_k, group_size=st.group_size,
                                        mode="symmetric_token", t_tile=T_TILE)
            zpk, sk_to_store, zpk_to_store = None, sk, None
        else:
            raise ValueError(f"Unknown mode_k={mk}")

        # V
        mv = getattr(st, "mode_v", "symmetric_channel")
        if mv == "asymmetric_channel":
            qv, sv, zpv = _quantize_asymmetric(V, bits=st.bits_v, group_size=st.group_size,
                                            mode="asymmetric_channel", t_tile=T_TILE)
            if st.V_scale is None: st.V_scale = sv.contiguous()
            if st.V_zp    is None: st.V_zp    = zpv.contiguous()
            sv_to_store, zpv_to_store = None, None
        elif mv == "asymmetric_token":
            qv, sv, zpv = _quantize_asymmetric(V, bits=st.bits_v, group_size=st.group_size,
                                            mode="asymmetric_token", t_tile=T_TILE)
            sv_to_store, zpv_to_store = sv, zpv
        elif mv in ("symmetric_channel"):
            qv, sv = _quantize_symmetric(V, bits=st.bits_v, group_size=st.group_size,
                                        mode="symmetric_channel", t_tile=T_TILE)
            zpv, sv_to_store, zpv_to_store = None, None, None
            if st.V_scale is None: st.V_scale = sv.contiguous()
        elif mv == "symmetric_token":
            qv, sv = _quantize_symmetric(V, bits=st.bits_v, group_size=st.group_size,
                                        mode="symmetric_token", t_tile=T_TILE)
            zpv, sv_to_store, zpv_to_store = None, sv, None
        else:
            raise ValueError(f"Unknown mode_v={mv}")

        # ---- append into chunk-lists (NO torch.cat) ----
        st.K_chunks.append(qk.contiguous())
        st.V_chunks.append(qv.contiguous())
        if sk_to_store is not None: st.Ks_chunks.append(sk_to_store.contiguous())
        if sv_to_store is not None: st.Vs_chunks.append(sv_to_store.contiguous())
        if zpk_to_store is not None:
            if getattr(st, "Kzp_chunks", None) is None: st.Kzp_chunks = []
            st.Kzp_chunks.append(zpk_to_store.contiguous())
        if zpv_to_store is not None:
            if getattr(st, "Vzp_chunks", None) is None: st.Vzp_chunks = []
            st.Vzp_chunks.append(zpv_to_store.contiguous())

        # ---- update T & byte counters ----
        st.T = int(st.T) + int(T_new)
        self.bytes_total += int(qk.numel() * qk.element_size()) + int(qv.numel() * qv.element_size())

        # K bytes (scales/zp)
        if mk == "symmetric_token" and sk_to_store is not None:
            self.bytes_scales += int(sk_to_store.numel() * sk_to_store.element_size())
        elif mk in ("symmetric_channel", "symmetric") and int(st.T) == T_new:
            self.bytes_scales += int(st.K_scale.numel() * st.K_scale.element_size())
        elif mk == "asymmetric_token":
            self.bytes_scales += int(sk_to_store.numel() * sk_to_store.element_size())
            self.bytes_zp     += int(zpk_to_store.numel())
        elif mk == "asymmetric_channel" and int(st.T) == T_new:
            self.bytes_scales += int(st.K_scale.numel() * st.K_scale.element_size())
            self.bytes_zp     += int(st.K_zp.numel())

        # V bytes (scales/zp)
        if mv == "symmetric_token" and sv_to_store is not None:
            self.bytes_scales += int(sv_to_store.numel() * sv_to_store.element_size())
        elif mv in ("symmetric_channel", "symmetric") and int(st.T) == T_new:
            self.bytes_scales += int(st.V_scale.numel() * st.V_scale.element_size())
        elif mv == "asymmetric_token":
            self.bytes_scales += int(sv_to_store.numel() * sv_to_store.element_size())
            self.bytes_zp     += int(zpv_to_store.numel())
        elif mv == "asymmetric_channel" and int(st.T) == T_new:
            self.bytes_scales += int(st.V_scale.numel() * st.V_scale.element_size())
            self.bytes_zp     += int(st.V_zp.numel())

        self._maybe_release_cuda_cache(f"PREFILL after-append L{layer_idx}")
        if getattr(self, "debug", False):
            add_bytes = (qk.numel()*qk.element_size() + qv.numel()*qv.element_size())
            print(f"[KVQDBG] L{layer_idx} append_chunked: +T={T_new} +packed={add_bytes}B (tot_packed={self.bytes_total}B)", flush=True)

    def dequant_slice(self, layer_idx: int, t_slice: slice) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return fp16 K,V for given time slice [start:stop]. Shapes [T,H,D].
        Works over chunked storage without allocating a big intermediate.
        """
        st = self.layers[layer_idx]
        start, stop, _ = t_slice.indices(st.T)
        Twant = stop - start
        H, D = st.H, st.D
        dev = self.device

        K_out = torch.empty((Twant, H, D), dtype=torch.float16, device=dev)
        V_out = torch.empty((Twant, H, D), dtype=torch.float16, device=dev)

        # ---- K ----
        mk = getattr(st, "mode_k", "symmetric_channel")
        if mk == "asymmetric_channel":
            Ks, Kzp = st.K_scale, st.K_zp
        elif mk in ("symmetric_channel"):
            Ks = st.K_scale

        for i, off, take, dst in _iter_covering(st.K_chunks, start, stop):
            Kp = st.K_chunks[i][off:off+take]
            if mk == "asymmetric_token":
                Ks = st.Ks_chunks[i][off:off+take]; Kzp = st.Kzp_chunks[i][off:off+take]
                Kt = _dequantize_asymmetric(Kp, Ks, Kzp, bits=st.bits_k, group_size=st.group_size,
                                            mode="asymmetric_token", T=take, H=H, D=D)
            elif mk == "asymmetric_channel":
                Kt = _dequantize_asymmetric(Kp, Ks, Kzp, bits=st.bits_k, group_size=st.group_size,
                                            mode="asymmetric_channel", T=take, H=H, D=D)
            elif mk == "symmetric_token":
                Ks = st.Ks_chunks[i][off:off+take]
                Kt = _dequantize_symmetric(Kp, Ks, bits=st.bits_k, group_size=st.group_size,
                                        mode="symmetric_token", T=take, H=H, D=D)
            else:  # symmetric_channel or symmetric
                Kt = _dequantize_symmetric(Kp, Ks, bits=st.bits_k, group_size=st.group_size,
                                        mode="symmetric_channel", T=take, H=H, D=D)
            K_out[dst:dst+take].copy_(Kt, non_blocking=True)

        # ---- V ----
        mv = getattr(st, "mode_v", "symmetric_channel")
        if mv == "asymmetric_channel":
            Vs, Vzp = st.V_scale, st.V_zp
        elif mv in ("symmetric_channel"):
            Vs = st.V_scale

        for i, off, take, dst in _iter_covering(st.V_chunks, start, stop):
            Vp = st.V_chunks[i][off:off+take]
            if mv == "asymmetric_token":
                Vs = st.Vs_chunks[i][off:off+take]; Vzp = st.Vzp_chunks[i][off:off+take]
                Vt = _dequantize_asymmetric(Vp, Vs, Vzp, bits=st.bits_v, group_size=st.group_size,
                                            mode="asymmetric_token", T=take, H=H, D=D)
            elif mv == "asymmetric_channel":
                Vt = _dequantize_asymmetric(Vp, Vs, Vzp, bits=st.bits_v, group_size=st.group_size,
                                            mode="asymmetric_channel", T=take, H=H, D=D)
            elif mv == "symmetric_token":
                Vs = st.Vs_chunks[i][off:off+take]
                Vt = _dequantize_symmetric(Vp, Vs, bits=st.bits_v, group_size=st.group_size,
                                        mode="symmetric_token", T=take, H=H, D=D)
            else:
                Vt = _dequantize_symmetric(Vp, Vs, bits=st.bits_v, group_size=st.group_size,
                                        mode="symmetric_channel", T=take, H=H, D=D)
            V_out[dst:dst+take].copy_(Vt, non_blocking=True)

        self._maybe_release_cuda_cache(f"dequant_slice L{layer_idx} [{start}:{stop}]")
        return K_out, V_out


    def dequant_slice_into(self, layer_idx: int, t_slice: slice, out_k: torch.Tensor, out_v: torch.Tensor):
        """Dequantize [start:stop] directly into provided GPU scratch buffers."""
        if self.debug:
            start, stop, _ = t_slice.indices(self.layers[layer_idx].T)
            Twant = stop - start
            print(f"[KVQDBG] L{layer_idx} dequant-into: slice=[{start}:{stop}] (T={Twant}) -> out_k/out_v on {out_k.device}", flush=True)
            print(_cuda_mem_snapshot(f"before-dequant L{layer_idx}"), flush=True)

        K, V = self.dequant_slice(layer_idx, t_slice)
        if out_k.shape != K.shape: out_k.resize_(K.shape)
        if out_v.shape != V.shape: out_v.resize_(V.shape)
        out_k.copy_(K, non_blocking=True); out_v.copy_(V, non_blocking=True)
        
        if self.debug:
            print(_cuda_mem_snapshot(f"after-dequant L{layer_idx}"), flush=True)
            # show scratch sizes
            print(f"[KVQDBG] L{layer_idx} scratch shapes: K{tuple(out_k.shape)} V{tuple(out_v.shape)}", flush=True)
        return out_k, out_v

    def _maybe_release_cuda_cache(self, where: str):
        import torch
        free_b = _cuda_free_bytes()
        if free_b < self.lowmem_guard_bytes:
            if getattr(self, "debug", False):
                print(f"[KVQDBG][LOWMEM] free={_mb(free_b)}MiB < {_mb(self.lowmem_guard_bytes)}MiB @ {where} -> empty_cache()+shrink T_TILE", flush=True)
            torch.cuda.empty_cache()
            # shrink T_TILE by half, but not below min_t_tile
            if hasattr(self, "t_tile"):
                self.t_tile = max(self.min_t_tile, int(self.t_tile // 2) or self.min_t_tile)

    # Simple metric helpers
    def bytes_summary(self) -> Dict[str, int]:
        return {"kv_bytes_total_packed": int(self.bytes_total), "kv_bytes_scales": int(self.bytes_scales)}


def _cuda_mem_snapshot(tag: str):
    """Return a compact GPU memory snapshot string."""
    try:
        import torch
        if not torch.cuda.is_available():
            return f"[{tag}] cuda not available"
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated()
        reserv = torch.cuda.memory_reserved()
        return (f"[{tag}] free={free/2**20:.0f}MiB alloc={alloc/2**20:.0f}MiB "
                f"reserved={reserv/2**20:.0f}MiB total={total/2**20:.0f}MiB")
    except Exception as e:
        return f"[{tag}] mem-snapshot-error: {e!r}"
