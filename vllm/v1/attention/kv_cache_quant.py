# vllm/v1/attention/kv_cache_quant.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import math
from vllm.v1.metrics.kv_quant import KVQuantLayerRecord
# -------------------------
# Helpers for (de)quant
# -------------------------

def _amax_channelwise(x: torch.Tensor, dim: int) -> torch.Tensor:
    # amax along specified dim (keepdim=True for broadcasting)
    return x.abs().amax(dim=dim, keepdim=True).clamp_min_(1e-8)

def _quantize_symmetric(x: torch.Tensor, bits: int, group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-channel(grouped) quantization (time-invariant scales).
    Returns (q_packed, scale). For 8-bit: int8; for 4/2-bit: uint8 packed.
    x: [T,H,D]
    """
    assert x.dim() == 3, "Expected [T, H, D]"
    T, H, D = x.shape

    if bits == 8:
        levels = 127.0
    elif bits == 4:
        levels = 7.0
    elif bits == 2:
        levels = 1.0
    else:
        raise ValueError(f"Unsupported bits={bits}")

    # pad D to multiple of group_size if needed
    if D % group_size != 0:
        pad = group_size - (D % group_size)
        x_pad = torch.nn.functional.pad(x, (0, pad))
        Dp = x_pad.shape[-1]
    else:
        x_pad = x
        Dp = D
    G = Dp // group_size

    # scale per (H,G), reduced over time & group
    x_view = x_pad.view(T, H, G, group_size)
    amax_hg = x_view.abs().amax(dim=(0, 3)).amax(dim=0)     # [H,G]
    scale = (amax_hg / levels).clamp_min(1e-8)              # [H,G]

    # quantize
    q = (x_pad.view(T, H, G, group_size) / scale.unsqueeze(0).unsqueeze(-1)).round()
    if bits == 8:
        q = q.clamp_(-128, 127).to(torch.int8).view(T, H, Dp)
        return q[..., :D].contiguous(), scale.contiguous()
    elif bits == 4:
        q = q.clamp_(-8, 7).to(torch.int8)
        return _pack_nibble_signed(q.view(T, H, Dp))[..., : (D // 2)], scale.contiguous()
    else:
        q = q.clamp_(-2, 1).to(torch.int8)
        return _pack_2bit_signed(q.view(T, H, Dp))[..., : (D // 4)], scale.contiguous()

def _dequantize_symmetric(q, scale, bits: int, group_size: int, T: int, H: int, D: int) -> torch.Tensor:
    """Inverse of per-channel(grouped) symmetric quantization. Returns fp16 [T,H,D]."""
    if bits == 8:
        xq = q.to(torch.float32)
    elif bits == 4:
        xq = _unpack_nibble_signed(q, T, H, D)  # float32
    else:
        xq = _unpack_2bit_signed(q, T, H, D)    # float32

    Dp = ((D + group_size - 1) // group_size) * group_size
    G  = Dp // group_size
    xq = torch.nn.functional.pad(xq, (0, Dp - D)).view(T, H, G, group_size)
    x  = xq * scale.view(1, H, G, 1).to(torch.float16)      # time-invariant scales
    return x.view(T, H, Dp)[..., :D].to(torch.float16).contiguous()

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
                         mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric quantization:
      mode='asymmetric_channel' -> per-channel-asym (reduce over T + group_size)
      mode='asymmetric_token'   -> per-token-asym   (reduce over group_size)
    Returns (q_packed, scale, zp). q_packed dtype=uint8 (packed if bits<8).
    """
    import torch
    assert mode in ("asymmetric_channel", "asymmetric_token")
    reduce_over_time = (mode == "asymmetric_channel")
    scale, zp, Dg, _ = _asym_qparams(x, bits, reduce_over_time, group_size)

    # broadcast to [T,H,D] with grouped zp/scale
    T, H, D = x.shape
    xg = x.view(T, H, Dg, group_size)
    if reduce_over_time:
        s = scale.view(1, H, Dg, 1).to(x.dtype)
        z = zp.view(1, H, Dg, 1).to(torch.int32)
    else:
        s = scale.view(T, H, Dg, 1).to(x.dtype)
        z = zp.view(T, H, Dg, 1).to(torch.int32)

    q = torch.round((xg / s) + z).clamp_(0, (1 << bits) - 1).to(torch.uint8)
    q = q.view(T, H, D)

    # pack along last dim
    if bits == 8:
        q_packed = q
    elif bits == 4:
        q_packed = _pack_nibble_u4(q)
    elif bits == 2:
        q_packed = _pack_2bit_u2(q)
    else:
        raise ValueError(f"Unsupported bits={bits} for asymmetric")
    return q_packed.contiguous(), scale.contiguous(), zp.contiguous()

def _dequantize_asymmetric(q_packed: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor,
                           *, bits: int, group_size: int, mode: str,
                           T: int, H: int, D: int) -> torch.Tensor:
    """
    Inverse of _quantize_asymmetric. Returns fp16 tensor [T,H,D].
    """
    import torch
    assert mode in ("asymmetric_channel", "asymmetric_token")
    # unpack
    if bits == 8:
        q = q_packed
    elif bits == 4:
        q = _unpack_nibble_u4(q_packed, D)
    elif bits == 2:
        q = _unpack_2bit_u2(q_packed, D)
    else:
        raise ValueError(f"Unsupported bits={bits}")

    Dg = (D + group_size - 1) // group_size
    qg = q.view(T, H, Dg, group_size).to(torch.int32)

    # broadcast zp/scale
    if mode == "asymmetric_channel":
        s = scale.view(1, H, Dg, 1).to(torch.float16)
        z = zp.view(1, H, Dg, 1).to(torch.int32)
    else:
        s = scale.view(T, H, Dg, 1).to(torch.float16)
        z = zp.view(T, H, Dg, 1).to(torch.int32)

    x = (qg.to(torch.float16) - z.to(torch.float16)) * s
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

class PagedKVCacheQuantized:
    """Reference quantized KV cache that packs K/V per layer with given policy.
    This implementation keeps a contiguous [T,H,D] history per layer for clarity.
    Replace with paged allocator bindings for production.
    """

    def __init__(self, n_layers: int, policies, device: torch.device, debug: bool = False, debug_interval: int = 128):
        self.device = device
        self.layers: Dict[int, LayerKVStore] = {}
        self.policies = policies  # callable: layer_idx -> LayerPolicy

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
        """Quantize & append K/V (time-axis) into per-layer packed storage."""
        import torch
        st = self.layers[layer_idx]

        # --- validate & init H/D ---
        K = K.contiguous(); V = V.contiguous()
        T_new, H_in, D_in = K.shape
        if int(getattr(st, "H", 0) or 0) > 0 and int(getattr(st, "D", 0) or 0) > 0:
            assert (H_in, D_in) == (st.H, st.D), f"Shape mismatch: got ({H_in},{D_in}) expected ({st.H},{st.D})"
        else:
            st.H, st.D = int(H_in), int(D_in)

        # ==== K: choose quant path ====
        if getattr(st, "mode_k", "symmetric") == "asymmetric_channel":
            qk, sk, zpk = _quantize_asymmetric(K, bits=st.bits_k, group_size=st.group_size, mode="asymmetric_channel")
        elif getattr(st, "mode_k", "symmetric") == "asymmetric_token":
            qk, sk, zpk = _quantize_asymmetric(K, bits=st.bits_k, group_size=st.group_size, mode="asymmetric_token")
        else:
            qk, sk = _quantize_symmetric(K, bits=st.bits_k, group_size=st.group_size)
            zpk = None

        # ==== V: choose quant path ====
        if getattr(st, "mode_v", "symmetric") == "asymmetric_token":
            qv, sv, zpv = _quantize_asymmetric(V, bits=st.bits_v, group_size=st.group_size, mode="asymmetric_token")
        elif getattr(st, "mode_v", "symmetric") == "asymmetric_channel":
            qv, sv, zpv = _quantize_asymmetric(V, bits=st.bits_v, group_size=st.group_size, mode="asymmetric_channel")
        else:
            qv, sv = _quantize_symmetric(V, bits=st.bits_v, group_size=st.group_size)
            zpv = None

        def _cat0(old, new):
            if old is None or (isinstance(old, torch.Tensor) and old.numel() == 0):
                return new.contiguous()
            return torch.cat([old, new], dim=0).contiguous()

        # ---- append packed ----
        st.K_packed = _cat0(st.K_packed, qk)
        st.V_packed = _cat0(st.V_packed, qv)

        # ---- scales/zp handling ----
        # per-token-* : append along time
        def _append_scale_zp(scale_old, zp_old, scale_new, zp_new, per_token: bool):
            if per_token:
                scale_old = _cat0(scale_old, scale_new)
                zp_old    = _cat0(zp_old,    zp_new)
            else:
                if scale_old is None or (isinstance(scale_old, torch.Tensor) and scale_old.numel() == 0):
                    scale_old = scale_new.contiguous()
                if zp_new is not None:
                    if zp_old is None or (isinstance(zp_old, torch.Tensor) and zp_old.numel() == 0):
                        zp_old = zp_new.contiguous()
            return scale_old, zp_old

        # K
        if zpk is None:
            # symmetric(per-channel only): scale is time-invariant → set once
            if st.K_scale is None or (isinstance(st.K_scale, torch.Tensor) and st.K_scale.numel() == 0):
                st.K_scale = sk.contiguous()
        else:
            per_token = (st.mode_k == "asymmetric_token")
            st.K_scale, st.K_zp = _append_scale_zp(st.K_scale, st.K_zp, sk, zpk, per_token)

        # V
        if zpv is None:
            if st.V_scale is None or (isinstance(st.V_scale, torch.Tensor) and st.V_scale.numel() == 0):
                st.V_scale = sv.contiguous()
        else:
            per_token = (st.mode_v == "asymmetric_token")
            st.V_scale, st.V_zp = _append_scale_zp(st.V_scale, st.V_zp, sv, zpv, per_token)

        # ---- advance T ----
        st.T = int(st.T) + int(T_new)

        # ---- byte accounting (packed + scale + zp) ----
        self.bytes_total += int(qk.numel() * qk.element_size()) + int(qv.numel() * qv.element_size())

        # K scales/zp
        if zpk is None:
            # symmetric per-channel: count once on first write into empty store
            if int(st.T) == T_new:
                self.bytes_scales += int(sk.numel() * sk.element_size())
        else:
            if st.mode_k == "asymmetric_token":
                self.bytes_scales += int(sk.numel() * sk.element_size())
                self.bytes_zp     += int(zpk.numel())
            else:
                if int(st.T) == T_new:
                    self.bytes_scales += int(sk.numel() * sk.element_size())
                    self.bytes_zp     += int(zpk.numel())

        # V scales/zp
        if zpv is None:
            if int(st.T) == T_new:
                self.bytes_scales += int(sv.numel() * sv.element_size())
        else:
            if st.mode_v == "asymmetric_token":
                self.bytes_scales += int(sv.numel() * sv.element_size())
                self.bytes_zp     += int(zpv.numel())
            else:
                if int(st.T) == T_new:
                    self.bytes_scales += int(sv.numel() * sv.element_size())
                    self.bytes_zp     += int(zpv.numel())
        

        if getattr(self, "debug", False):
            add_bytes = (qk.numel()*qk.element_size() + qv.numel()*qv.element_size())
            print(f"[KVQDBG] L{layer_idx} _append_core: +packed={add_bytes}B (tot_packed={self.bytes_total}B)", flush=True)



    def dequant_slice(self, layer_idx: int, t_slice: slice) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return fp16 K,V for given time slice [start:stop]. Shapes [T,H,D]."""
        st = self.layers[layer_idx]
        start, stop, _ = t_slice.indices(st.T)
        T = stop - start

        # ---- K ----
        Kp = st.K_packed[start:stop]
        if getattr(st, "mode_k", "symmetric").startswith("asymmetric"):
            if st.mode_k == "asymmetric_token":
                Ks = st.K_scale[start:stop]; Kzp = st.K_zp[start:stop]
            else:
                Ks = st.K_scale; Kzp = st.K_zp
            K = _dequantize_asymmetric(Kp, Ks, Kzp, bits=st.bits_k, group_size=st.group_size,
                                    mode=st.mode_k, T=T, H=st.H, D=st.D)
        else:
            Ks = st.K_scale
            K  = _dequantize_symmetric(Kp, Ks, bits=st.bits_k, group_size=st.group_size,
                               T=T, H=st.H, D=st.D)

        # ---- V ----
        Vp = st.V_packed[start:stop]
        if getattr(st, "mode_v", "symmetric").startswith("asymmetric"):
            if st.mode_v == "asymmetric_token":
                Vs = st.V_scale[start:stop]; Vzp = st.V_zp[start:stop]
            else:
                Vs = st.V_scale; Vzp = st.V_zp
            V = _dequantize_asymmetric(Vp, Vs, Vzp, bits=st.bits_v, group_size=st.group_size,
                                    mode=st.mode_v, T=T, H=st.H, D=st.D)
        else:
            Vs = st.V_scale
            V  = _dequantize_symmetric(Vp, Vs, bits=st.bits_v, group_size=st.group_size,
                               T=T, H=st.H, D=st.D)

        return K, V

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
