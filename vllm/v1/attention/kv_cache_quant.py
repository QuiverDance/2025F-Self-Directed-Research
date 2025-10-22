# vllm/v1/attention/kv_cache_quant.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import torch
import math

# Reuse the public record type for optional layer-wise logging (already used by metrics)
from vllm.v1.metrics.kv_quant import KVQuantLayerRecord

# ======================
# Global / Epoch control
# ======================
T_TILE = 256  # T-tiling for memory-safe (de)quant
_KVQ_ACTIVE_EPOCH = 0

def kvq_new_task():
    """Bump global epoch to signal cache reset for a new REQUEST."""
    global _KVQ_ACTIVE_EPOCH
    _KVQ_ACTIVE_EPOCH += 1

# ======================
# Quant helpers (reuse)
# ======================
def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def _pack_bits(xq: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Pack low-bit integers along the last dimension 8//bits at a time.
    xq: uint8 tensor with values in [0, 2^bits-1]
    Returns a uint8 packed tensor.
    """
    assert xq.dtype == torch.uint8
    step = 8 // bits
    pad = (-xq.shape[-1]) % step
    if pad:
        xq = torch.nn.functional.pad(xq, (0, pad))
    xq = xq.view(*xq.shape[:-1], -1, step)
    out = torch.zeros(xq.shape[:-1], dtype=torch.uint8, device=xq.device)
    for i in range(step):
        out |= (xq[..., i] & ((1 << bits) - 1)) << (i * bits)
    return out

def _unpack_bits(packed: torch.Tensor, bits: int, last_dim: int) -> torch.Tensor:
    """
    Inverse of _pack_bits.
    last_dim: number of original items along the packed dim before packing.
    """
    assert packed.dtype == torch.uint8
    step = 8 // bits
    out = torch.empty((*packed.shape, step), dtype=torch.uint8, device=packed.device)
    for i in range(step):
        out[..., i] = (packed >> (i * bits)) & ((1 << bits) - 1)
    out = out.view(*out.shape[:-1], -1)[..., :last_dim]
    return out

def _compute_groupwise_scale_zp(
    x: torch.Tensor, bits: int, group_size: int, asymmetric: bool, reduce_over_time: bool
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """
    Compute per-group (scale[, zp]) for symmetric/asymmetric quant.
    - If reduce_over_time=True  -> reduce over (T, group_size)  => per-channel stats
    - Else (False)              -> reduce over (group_size)     => per-token stats
    Returns:
      scale: fp16, shape [H,Dg] or [T,H,Dg]
      zp   : uint8 (same shape) if asymmetric else None
      Dg   : number of groups on D axis
    """
    T, H, D = x.shape
    Dg = _ceil_div(D, group_size)
    # reshape to (..., Dg, group_size)
    pad = (-D) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    xg = x.view(T, H, Dg, group_size)
    if reduce_over_time:
        xred = xg.float().amax(dim=(0, 3)) - xg.float().amin(dim=(0, 3))
        # avoid zero-scale
        scale = (xred / ((1 << bits) - 1)).clamp_min(1e-8).to(torch.float16)  # [H,Dg]
        zp = None
        return scale, zp, Dg
    else:
        xmax = xg.float().amax(dim=3)
        xmin = xg.float().amin(dim=3)
        scale = ((xmax - xmin) / ((1 << bits) - 1)).clamp_min(1e-8).to(torch.float16)  # [T,H,Dg]
        zp = torch.clamp((-xmin / scale).round(), 0, (1 << bits) - 1).to(torch.uint8)
        return scale, zp, Dg

@dataclass
class LayerPolicy:
    bits_k: int = 8
    bits_v: int = 8
    group_size: int = 64
    mode_k: str = "asymmetric_channel"  # {"symmetric_channel","asymmetric_channel","asymmetric_token"}
    mode_v: str = "asymmetric_token"

@dataclass
class _BlockStore:
    """Quantized storage for one block."""
    # Packed low-bit tensors, laid out as [Tblk, H, DgPacked] where last dim is packed
    K_packed: Optional[torch.Tensor] = None
    V_packed: Optional[torch.Tensor] = None
    # Scales and optional zp
    K_scale: Optional[torch.Tensor] = None
    V_scale: Optional[torch.Tensor] = None
    K_zp: Optional[torch.Tensor] = None
    V_zp: Optional[torch.Tensor] = None
    # Meta
    T_filled: int = 0  # how many tokens in this block are valid (<= block_size)

@dataclass
class _LayerState:
    """State for one transformer layer."""
    policy: LayerPolicy
    H: Optional[int] = None
    D: Optional[int] = None
    T_total: int = 0                         # total tokens ever appended
    blocks: Dict[int, _BlockStore] = field(default_factory=dict)

class PagedKVCacheQuantized:
    """
    Block/page-aligned quantized KV cache.

    - No duplicate FP16 KV: native vLLM KV allocation can be set to 0 by caller.
    - Appends are written into block-local packed buffers keyed by block_id.
    - Decoding dequantizes only the requested [start:stop) range, walking blocks.
    - Layer policies (bits/mode/group_size) come from KVQuantConfig.
    """

    def __init__(self, num_layers: int, device: torch.device, policies: Dict[int, LayerPolicy],
                 block_size: int = 16, validate: bool = False, debug: bool = False, log_interval: int = 128) -> None:
        self.num_layers = num_layers
        self.block_size = int(block_size)
        self.device = device
        self.layers: Dict[int, _LayerState] = {
            li: _LayerState(policy=policies.get(li, LayerPolicy())) for li in range(num_layers)
        }
        self.validate = bool(validate)
        self.debug = bool(debug)
        self.log_interval = int(log_interval)

        # accounting
        self.bytes_packed_total = 0
        self.bytes_scales_total = 0
        self.bytes_zp_total = 0
        self._epoch = _KVQ_ACTIVE_EPOCH

    # --------- internal helpers ---------
    def _ensure_shape(self, li: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[int,int,int]:
        """Initialize layer shape (H,D) the first time we see it."""
        st = self.layers[li]
        T, H, D = int(k.shape[0]), int(k.shape[1]), int(k.shape[2])
        if st.H is None:
            st.H, st.D = H, D
        else:
            if H != st.H or D != st.D:
                raise ValueError(f"[KVQ] Layer {li}: shape changed: was (H={st.H},D={st.D}), now (H={H},D={D})")
        return T, H, D

    def _get_block(self, st: _LayerState, block_id: int) -> _BlockStore:
        if block_id not in st.blocks:
            st.blocks[block_id] = _BlockStore()
        return st.blocks[block_id]

    def _quant_into_block(self, li: int, st: _LayerState, blk: _BlockStore,
                          t0_blk: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Quantize [Tb,H,D] (Tb<=block_size) chunk and write into block 'blk' at offset t0_blk.
        k, v shapes: [Tb, H, D], contiguous on device(self.device).
        """
        pol = st.policy
        Tb, H, D = k.shape
        assert v.shape == (Tb, H, D)

        # --- K ---
        asym_k = "asymmetric" in pol.mode_k
        per_token_k = (pol.mode_k == "asymmetric_token")
        scale_k, zp_k, Dg = _compute_groupwise_scale_zp(
            k, pol.bits_k, pol.group_size, asymmetric=asym_k, reduce_over_time=not per_token_k
        )
        # quantize / pack
        if per_token_k:
            # scale_k: [T,H,Dg], zp_k: [T,H,Dg]
            qk = torch.clamp((k.float() / scale_k.float()).round() + zp_k.float(), 0, (1 << pol.bits_k) - 1).to(torch.uint8)
        else:
            # scale_k: [H,Dg], symmetric/asymmetric-channel => zp is None or broadcastable const
            if zp_k is None:
                qk = torch.clamp((k.float() / scale_k.float()).round() + (1 << (pol.bits_k - 1)), 0, (1 << pol.bits_k) - 1).to(torch.uint8)
            else:
                qk = torch.clamp((k.float() / scale_k.float()).round() + zp_k.float(), 0, (1 << pol.bits_k) - 1).to(torch.uint8)
        qk = qk.view(Tb, H, Dg, -1)  # last dim: group_size
        qk_packed = _pack_bits(qk.view(Tb, H, -1), pol.bits_k)  # pack along last

        # --- V ---
        asym_v = "asymmetric" in pol.mode_v
        per_token_v = (pol.mode_v == "asymmetric_token")
        scale_v, zp_v, Dg_v = _compute_groupwise_scale_zp(
            v, pol.bits_v, pol.group_size, asymmetric=asym_v, reduce_over_time=not per_token_v
        )
        assert Dg_v == Dg
        if per_token_v:
            qv = torch.clamp((v.float() / scale_v.float()).round() + zp_v.float(), 0, (1 << pol.bits_v) - 1).to(torch.uint8)
        else:
            if zp_v is None:
                qv = torch.clamp((v.float() / scale_v.float()).round() + (1 << (pol.bits_v - 1)), 0, (1 << pol.bits_v) - 1).to(torch.uint8)
            else:
                qv = torch.clamp((v.float() / scale_v.float()).round() + zp_v.float(), 0, (1 << pol.bits_v) - 1).to(torch.uint8)
        qv = qv.view(Tb, H, Dg, -1)
        qv_packed = _pack_bits(qv.view(Tb, H, -1), pol.bits_v)

        # allocate or grow storage in the block and write at [t0_blk:t0_blk+Tb)
        def _write(dst: Optional[torch.Tensor], src: torch.Tensor) -> torch.Tensor:
            if dst is None:
                # allocate full block, then partially fill
                full = torch.empty((self.block_size, H, src.shape[-1]), dtype=src.dtype, device=self.device)
                full.zero_()
                full[t0_blk:t0_blk + Tb].copy_(src)
                return full
            else:
                dst[t0_blk:t0_blk + Tb].copy_(src)
                return dst

        blk.K_packed = _write(blk.K_packed, qk_packed)
        blk.V_packed = _write(blk.V_packed, qv_packed)
        blk.T_filled = max(blk.T_filled, t0_blk + Tb)

        # store scales / zp
        def _store_stats(name: str, cur: Optional[torch.Tensor], val: torch.Tensor) -> torch.Tensor:
            if cur is None:
                full = torch.empty((self.block_size, *val.shape[1:]), dtype=val.dtype, device=self.device)
                full.zero_()
                full[t0_blk:t0_blk + Tb].copy_(val)
                return full
            else:
                cur[t0_blk:t0_blk + Tb].copy_(val)
                return cur

        if per_token_k:
            blk.K_scale = _store_stats("K_scale", blk.K_scale, scale_k)
            blk.K_zp    = _store_stats("K_zp",    blk.K_zp,    zp_k)
        else:
            # per-channel: save once at row 0, broadcast on dequant
            blk.K_scale = scale_k
            blk.K_zp    = zp_k
        if per_token_v:
            blk.V_scale = _store_stats("V_scale", blk.V_scale, scale_v)
            blk.V_zp    = _store_stats("V_zp",    blk.V_zp,    zp_v)
        else:
            blk.V_scale = scale_v
            blk.V_zp    = zp_v

        # accounting
        self.bytes_packed_total += int(qk_packed.numel() + qv_packed.numel())
        if per_token_k:
            self.bytes_scales_total += int(scale_k.numel() * 2)  # fp16
            self.bytes_zp_total     += int(zp_k.numel()) if zp_k is not None else 0
        else:
            self.bytes_scales_total += int(scale_k.numel() * 2)
            self.bytes_zp_total     += int(scale_k.numel() if (zp_k is not None) else 0)
        if per_token_v:
            self.bytes_scales_total += int(scale_v.numel() * 2)
            self.bytes_zp_total     += int(zp_v.numel()) if zp_v is not None else 0
        else:
            self.bytes_scales_total += int(scale_v.numel() * 2)
            self.bytes_zp_total     += int(scale_v.numel() if (zp_v is not None) else 0)

    # --------- public API (called by wrapper) ---------
    def append_kv_prefill(self, li: int, K: torch.Tensor, V: torch.Tensor) -> None:
        """
        Prefill append: write [T,H,D] into block-aligned storage.
        NOTE: Fix original bug (lowercase 'k'): use 'K' consistently.
        """
        if self._epoch != _KVQ_ACTIVE_EPOCH:
            self.reset_all()
            self._epoch = _KVQ_ACTIVE_EPOCH

        T, H, D = self._ensure_shape(li, K, V)
        st = self.layers[li]
        # Write the whole T into successive blocks.
        t_written = 0
        while t_written < T:
            blk_idx = (st.T_total + t_written) // self.block_size
            t0_blk  = (st.T_total + t_written) %  self.block_size
            n_blk   = min(self.block_size - t0_blk, T - t_written)
            blk = self._get_block(st, blk_idx)
            self._quant_into_block(li, st, blk, t0_blk,
                                   K[t_written:t_written + n_blk].contiguous(),
                                   V[t_written:t_written + n_blk].contiguous())
            t_written += n_blk
        st.T_total += T

    def append_kv(self, li: int, K: torch.Tensor, V: torch.Tensor) -> None:
        """Decode-time small appends (usually T<=1)."""
        self.append_kv_prefill(li, K, V)

    def dequant_slice_into(self, li: int, sl: slice, Kdst: torch.Tensor, Vdst: torch.Tensor) -> None:
        """
        Dequantize [start:stop) into the provided scratch buffers (on device).
        Kdst/Vdst have shape [Tneed,H,D].
        """
        st = self.layers[li]
        if st.H is None:
            raise RuntimeError(f"[KVQ] Layer {li} has no data.")
        start = 0 if (sl.start is None) else int(sl.start)
        stop  = st.T_total if (sl.stop  is None) else int(sl.stop)
        assert 0 <= start <= stop <= st.T_total
        Tneed = stop - start
        assert Kdst.shape[0] == Vdst.shape[0] == Tneed
        assert Kdst.shape[1] == Vdst.shape[1] == st.H
        assert Kdst.shape[2] == Vdst.shape[2] == st.D

        pol = st.policy
        bits_k, bits_v = pol.bits_k, pol.bits_v
        Dg = _ceil_div(st.D, pol.group_size)

        t_read = 0
        while t_read < Tneed:
            g_t = start + t_read
            blk_idx = g_t // self.block_size
            off     = g_t %  self.block_size
            n_blk   = min(self.block_size - off, Tneed - t_read)
            blk = self._get_block(st, blk_idx)

            # --- unpack K ---
            kpk = blk.K_packed[off:off + n_blk]  # [Tb,H,Packed]
            k_unp = _unpack_bits(kpk, bits_k, Dg * pol.group_size).view(n_blk, st.H, Dg, pol.group_size)[..., :st.D]
            if pol.mode_k == "asymmetric_token":
                scale_k = blk.K_scale[off:off + n_blk]  # [Tb,H,Dg]
                zp_k    = blk.K_zp[off:off + n_blk]
                kf = (k_unp.float() - zp_k.float()) * scale_k.float()
            else:
                scale_k = blk.K_scale  # [H,Dg]
                if blk.K_zp is None:   # symmetric-channel
                    zp = (1 << (bits_k - 1))
                    kf = (k_unp.float() - zp) * scale_k.float()
                else:                   # asymmetric-channel
                    kf = (k_unp.float() - blk.K_zp.float()) * scale_k.float()
            Kdst[t_read:t_read + n_blk].copy_(kf.view(n_blk, st.H, st.D))

            # --- unpack V ---
            vpk = blk.V_packed[off:off + n_blk]
            v_unp = _unpack_bits(vpk, bits_v, Dg * pol.group_size).view(n_blk, st.H, Dg, pol.group_size)[..., :st.D]
            if pol.mode_v == "asymmetric_token":
                scale_v = blk.V_scale[off:off + n_blk]
                zp_v    = blk.V_zp[off:off + n_blk]
                vf = (v_unp.float() - zp_v.float()) * scale_v.float()
            else:
                scale_v = blk.V_scale
                if blk.V_zp is None:
                    zp = (1 << (bits_v - 1))
                    vf = (v_unp.float() - zp) * scale_v.float()
                else:
                    vf = (v_unp.float() - blk.V_zp.float()) * scale_v.float()
            Vdst[t_read:t_read + n_blk].copy_(vf.view(n_blk, st.H, st.D))

            t_read += n_blk

    def bytes_summary(self) -> Dict[str, int]:
        """Return compressed bytes (packed + scales + zps) for metrics."""
        return {
            "kv_bytes_total_packed": int(self.bytes_packed_total),
            "kv_bytes_scales": int(self.bytes_scales_total),
            "kv_bytes_zp": int(self.bytes_zp_total),
        }

    # ---------- maintenance ----------
    def reset_all(self) -> None:
        """Reset all layer storages (fix: iterate dict values; correct field names)."""
        for st in self.layers.values():
            st.blocks.clear()
            st.T_total = 0
            # shapes keep remembered (H,D)
        self.bytes_packed_total = 0
        self.bytes_scales_total = 0
        self.bytes_zp_total = 0