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

def _quantize_symmetric(x: torch.Tensor, bits: int, group_size: int, granularity: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize x with symmetric per-channel(group) or per-token-head granularity.
    Returns (q_packed, scale). For 8-bit: int8 tensor; for 4/2-bit: uint8 packed.
    Layout assumption:
      x shape: [T, H, D] (token, head, head_dim)
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

    if granularity == "per_token_head":
        # scale shape: [T, H, ceil(D/group)]
        if D % group_size != 0:
            # pad D to multiple of group_size for scale computation; pad with zeros
            pad = group_size - (D % group_size)
            x_pad = torch.nn.functional.pad(x, (0, pad))
            Dp = x_pad.shape[-1]
        else:
            x_pad = x
            Dp = D
        G = Dp // group_size
        x_view = x_pad.view(T, H, G, group_size)
        # scale per group
        scale = _amax_channelwise(x_view, dim=-1) / levels  # [T,H,G,1]
        scale = scale.squeeze(-1)                             # [T,H,G]
        # quantize
        q = (x_pad.view(T, H, G, group_size) / scale.unsqueeze(-1)).round()
        if bits == 8:
            q = q.clamp_(-128, 127).to(torch.int8).view(T, H, Dp)
            return q[..., :D].contiguous(), scale
        elif bits == 4:
            q = q.clamp_(-8, 7).to(torch.int8)  # store as signed nibble before pack
            return _pack_nibble_signed(q.view(T, H, Dp))[..., : (D // 2)], scale
        else:
            q = q.clamp_(-2, 1).to(torch.int8)  # store as 2-bit signed before pack
            return _pack_2bit_signed(q.view(T, H, Dp))[..., : (D // 4)], scale
    else:
        # per_channel(group) over D only (shared across tokens)
        if D % group_size != 0:
            pad = group_size - (D % group_size)
            x_pad = torch.nn.functional.pad(x, (0, pad))
            Dp = x_pad.shape[-1]
        else:
            x_pad = x
            Dp = D
        G = Dp // group_size
        # compute scale across T dimension as well to be safe (amax over T)
        x_view = x_pad.view(T, H, G, group_size)
        amax = x_view.abs().amax(dim=(0, 2, 3), keepdim=False)  # [H]
        # use per-head global amax for stability; could be per (H, G) optionally
        # switch to (H, G)
        amax_hg = x_view.abs().amax(dim=(0, 3)).amax(dim=0)     # [H,G]
        scale = (amax_hg / levels).clamp_min(1e-8)              # [H,G]
        # quantize
        q = (x_pad.view(T, H, G, group_size) / scale.unsqueeze(0).unsqueeze(-1)).round()
        if bits == 8:
            q = q.clamp_(-128, 127).to(torch.int8).view(T, H, Dp)
            return q[..., :D].contiguous(), scale
        elif bits == 4:
            q = q.clamp_(-8, 7).to(torch.int8)
            return _pack_nibble_signed(q.view(T, H, Dp))[..., : (D // 2)], scale
        else:
            q = q.clamp_(-2, 1).to(torch.int8)
            return _pack_2bit_signed(q.view(T, H, Dp))[..., : (D // 4)], scale

def _dequantize_symmetric(q, scale, bits: int, group_size: int, granularity: str, T: int, H: int, D: int) -> torch.Tensor:
    """Undo _quantize_symmetric. Returns fp16 tensor [T,H,D]."""
    if bits == 8:
        xq = q.to(torch.float32)
        if granularity == "per_token_head":
            # scale: [T,H,G], expand to [T,H,G,group]
            Dp = ( (D + group_size - 1) // group_size ) * group_size
            G = Dp // group_size
            xq = torch.nn.functional.pad(xq, (0, Dp - D)).view(T, H, G, group_size)
            x = xq * scale.unsqueeze(-1)
            return x.view(T, H, Dp)[..., :D].to(torch.float16).contiguous()
        else:
            Dp = ( (D + group_size - 1) // group_size ) * group_size
            G = Dp // group_size
            xq = torch.nn.functional.pad(xq, (0, Dp - D)).view(T, H, G, group_size)
            x = xq * scale.unsqueeze(0).unsqueeze(-1)
            return x.view(T, H, Dp)[..., :D].to(torch.float16).contiguous()
    elif bits == 4:
        # unpack to signed int4 in [-8,7]
        xq = _unpack_nibble_signed(q, T, H, D)  # float32
    else:
        xq = _unpack_2bit_signed(q, T, H, D)    # float32

    if granularity == "per_token_head":
        Dp = ( (D + group_size - 1) // group_size ) * group_size
        G = Dp // group_size
        xq = torch.nn.functional.pad(xq, (0, Dp - D)).view(T, H, G, group_size)
        x = xq * scale.unsqueeze(-1)
        return x.view(T, H, Dp)[..., :D].to(torch.float16).contiguous()
    else:
        Dp = ( (D + group_size - 1) // group_size ) * group_size
        G = Dp // group_size
        xq = torch.nn.functional.pad(xq, (0, Dp - D)).view(T, H, G, group_size)
        x = xq * scale.unsqueeze(0).unsqueeze(-1)
        return x.view(T, H, Dp)[..., :D].to(torch.float16).contiguous()

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

# -------------------------
# Quantized KV cache
# -------------------------

@dataclass
class LayerKVStore:
    bits_k: int
    bits_v: int
    group_size: int
    granularity: str
    symmetric: bool = True
    # packed storage and scales
    K_packed: Optional[torch.Tensor] = None
    V_packed: Optional[torch.Tensor] = None
    K_scale: Optional[torch.Tensor] = None
    V_scale: Optional[torch.Tensor] = None
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

        self.debug = debug
        self.debug_interval = max(1, int(debug_interval))

        for li in range(n_layers):
            p = policies(li)
            self.layers[li] = LayerKVStore(
                bits_k=p.bits_k, bits_v=p.bits_v, group_size=p.group_size, granularity=p.granularity, symmetric=p.symmetric
            )

    @staticmethod
    def _pack_bits(bits: int, q: torch.Tensor, scale: torch.Tensor, is_k: bool, D: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return q, scale

    def append_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Append one step of K,V: shapes [H,D] (single token) or [T,H,D]."""
        # === DEBUG: before-append snapshot ===
        if self.debug and (self.layers[layer_idx].T == 0):
            print(f"[KVQDBG] L{layer_idx} first-append: K/V shape={tuple(k.shape)} device={k.device} dtype={k.dtype}", flush=True)
            print(_cuda_mem_snapshot(f"before-append L{layer_idx}"), flush=True)
        elif self.debug and (self.layers[layer_idx].T % self.debug_interval == 0):
            print(f"[KVQDBG] L{layer_idx} append@T={self.layers[layer_idx].T}: incoming {tuple(k.shape)}", flush=True)

        st = self.layers[layer_idx]
        assert k.shape == v.shape and k.dim() in (2,3)
        # normalize to [T,H,D]
        if k.dim() == 2:
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        T, H, D = k.shape
        st.H = H; st.D = D

        # Quantize K
        qk, sk = _quantize_symmetric(k, st.bits_k, st.group_size, st.granularity)
        qv, sv = _quantize_symmetric(v, st.bits_v, st.group_size, st.granularity)

        # Append along T
        def _cat(old, new):
            if old is None:
                return new.contiguous()
            return torch.cat([old, new], dim=0).contiguous()

        st.K_packed = _cat(st.K_packed, qk)
        st.V_packed = _cat(st.V_packed, qv)
        st.K_scale = _cat(st.K_scale, sk) if st.granularity == "per_token_head" else sk
        first_append = (st.T == 0)
        st.V_scale = _cat(st.V_scale, sv) if st.granularity == "per_token_head" else sv
        st.T += T

        # book-keep bytes
        self.bytes_total += int(qk.element_size() * qk.numel())
        self.bytes_total += int(qv.element_size() * qv.numel())

        # scales bytes        
        if st.granularity == "per_token_head":
            self.bytes_scales += int(sk.element_size() * sk.numel())
            self.bytes_scales += int(sv.element_size() * sv.numel())
        else:
            if first_append:
                self.bytes_scales += int(sk.element_size() * sk.numel())
                self.bytes_scales += int(sv.element_size() * sv.numel())

        # per-layer cumulative snapshot
        if hasattr(self, "logger") and self.logger:
            k_bytes = int(st.K_packed.element_size() * st.K_packed.numel()) if st.K_packed is not None else 0
            v_bytes = int(st.V_packed.element_size() * st.V_packed.numel()) if st.V_packed is not None else 0
            ks_bytes = int(st.K_scale.element_size() * st.K_scale.numel()) if st.K_scale is not None else 0
            vs_bytes = int(st.V_scale.element_size() * st.V_scale.numel()) if st.V_scale is not None else 0
            self.logger.log_layer(KVQuantLayerRecord(
                layer=layer_idx,
                bits_k=st.bits_k,
                bits_v=st.bits_v,
                kv_bytes_total_packed=k_bytes + v_bytes,
                kv_bytes_scales=ks_bytes + vs_bytes,
                tt=0.0,
            ))
        
        # === DEBUG: after-append snapshot ===
        if self.debug:
            bs = self.bytes_summary()
            packed = bs.get("kv_bytes_total_packed", 0)
            scales = bs.get("kv_bytes_scales", 0)
            print(f"[KVQDBG] L{layer_idx} after-append: packed={packed} bytes, scales={scales} bytes", flush=True)
            print(_cuda_mem_snapshot(f"after-append L{layer_idx}"), flush=True)


    def dequant_slice(self, layer_idx: int, t_slice: slice) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return fp16 K,V for given time slice [start:stop]. Shapes [T,H,D]."""
        st = self.layers[layer_idx]
        start, stop, step = t_slice.indices(st.T)
        # select packed
        Kp = st.K_packed[start:stop]
        Vp = st.V_packed[start:stop]
        T = Kp.shape[0]
        # select scales
        if st.granularity == "per_token_head":
            Ks = st.K_scale[start:stop]
            Vs = st.V_scale[start:stop]
        else:
            Ks = st.K_scale
            Vs = st.V_scale
        K = _dequantize_symmetric(Kp, Ks, st.bits_k, st.group_size, st.granularity, T=T, H=st.H, D=st.D)
        V = _dequantize_symmetric(Vp, Vs, st.bits_v, st.group_size, st.granularity, T=T, H=st.H, D=st.D)
        return K, V

    def dequant_slice_into(
        self, layer_idx: int, t_slice: slice,
        out_k: torch.Tensor, out_v: torch.Tensor
    ):
        """Dequantize [start:stop] directly into provided GPU scratch buffers."""
        if self.debug:
            start, stop, _ = t_slice.indices(self.layers[layer_idx].T)
            Twant = stop - start
            print(f"[KVQDBG] L{layer_idx} dequant-into: slice=[{start}:{stop}] (T={Twant}) -> out_k/out_v on {out_k.device}", flush=True)
            print(_cuda_mem_snapshot(f"before-dequant L{layer_idx}"), flush=True)

        st = self.layers[layer_idx]
        start, stop, _ = t_slice.indices(st.T)
        T = stop - start
        # (packed/scales)
        Kp = st.K_packed[start:stop]; Vp = st.V_packed[start:stop]
        if st.granularity == "per_token_head":
            Ks = st.K_scale[start:stop]; Vs = st.V_scale[start:stop]
        else:
            Ks = st.K_scale; Vs = st.V_scale
        K = _dequantize_symmetric(Kp, Ks, st.bits_k, st.group_size, st.granularity,
                                  T=T, H=st.H, D=st.D)
        V = _dequantize_symmetric(Vp, Vs, st.bits_v, st.group_size, st.granularity,
                                  T=T, H=st.H, D=st.D)
        # copy into out scratch
        if out_k.shape != K.shape: out_k.resize_(K.shape)
        if out_v.shape != V.shape: out_v.resize_(V.shape)
        out_k.copy_(K, non_blocking=True)
        out_v.copy_(V, non_blocking=True)
        
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
