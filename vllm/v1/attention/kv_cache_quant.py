# vllm/v1/attention/kv_cache_quant.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import torch
import math
import logging
from types import MethodType

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

# ==========================================================
# Quant attach shim for vLLM BlockManager (paged KV backend)
# ==========================================================
def attach_quant_to_block_manager(scheduler, kvq_cfg, policy_fn=None) -> bool:
    """Attach a write/read shim to the BlockManager so that KV is stored
    in quantized form per layer/block, while the BlockPool scheduler,
    allocation and eviction logic remain unchanged.

    We do NOT touch CUDA kernels: at read, we dequantize only the requested
    token slice into a scratch FP16 buffer passed to the kernel.
    """
    logger = logging.getLogger(__name__)

    # Resolve manager
    mgr = None
    for name in ("kv_cache_manager", "block_manager", "cache_manager"):
        mgr = getattr(scheduler, name, None) or mgr
    if mgr is None or getattr(mgr, "_kvq_wrapped", False):
        return False

    # Layer policy resolver
    if policy_fn is None:
        policy_fn = getattr(kvq_cfg, "policy_for", None)
    if policy_fn is None:
        return False

    # Debug & logging controls
    mgr._kvq_debug = bool(getattr(kvq_cfg, "debug", False) or getattr(kvq_cfg, "kv_quant_debug", False))
    mgr._kvq_log_interval = int(getattr(kvq_cfg, "log_interval", 0) or getattr(kvq_cfg, "kv_quant_log_interval", 0) or 0)

    # Sidecar store & stats
    mgr._kvq_wrapped = True
    mgr._kvq_store = {}
    mgr._kvq_stats = {"bytes_packed": 0, "bytes_scales": 0, "bytes_zp": 0}
    mgr._kvq_token_count = 0

    def _key(layer_idx: int, block_id: int):
        return (int(layer_idx), int(block_id))

    def _quant_pack(x: torch.Tensor, bits: int, group: int, mode: str):
        x2 = x.contiguous().view(-1, x.shape[-2], x.shape[-1])  # [*, H, D]
        scale, zp = _compute_groupwise_scale_zp(x2, bits, group, mode)
        q = _quantize_u8(x2, scale, zp, bits, mode)
        packed = _pack_bits(q, bits)
        return packed, scale, zp

    def _dequant_slice(packed, scale, zp, bits, mode, last_dim, out_shape):
        unp = _unpack_bits(packed, bits, last_dim).view(*out_shape)
        return _dequantize_u8(unp, scale, zp, bits, mode)

    # ---- append wrapper: write quantized payload into sidecar
    orig_append = getattr(mgr, "append_kv", None) or getattr(mgr, "append", None)
    if orig_append is not None:
        def _wrap_append(self, layer_idx, block_id, k, v, *args, **kwargs):
            pol = policy_fn(int(layer_idx))
            H, Dk = int(k.shape[-2]), int(k.shape[-1]); Dv = int(v.shape[-1])
            kpk, ksc, kzp = _quant_pack(k, pol.bits_k, pol.group_size, pol.mode_k)
            vpk, vsc, vzp = _quant_pack(v, pol.bits_v, pol.group_size, pol.mode_v)
            self._kvq_store[_key(layer_idx, block_id)] = {
                "K_packed": kpk, "K_scale": ksc, "K_zp": kzp,
                "V_packed": vpk, "V_scale": vsc, "V_zp": vzp,
                "H": torch.tensor(H), "Dk": torch.tensor(Dk), "Dv": torch.tensor(Dv),
            }
            self._kvq_stats["bytes_packed"] += int(kpk.numel() + vpk.numel())
            self._kvq_stats["bytes_scales"] += int(ksc.numel() + vsc.numel())
            self._kvq_stats["bytes_zp"]     += int((kzp.numel() if kzp is not None else 0)
                                                   + (vzp.numel() if vzp is not None else 0))
            # interval debug
            self._kvq_token_count += int(k.shape[0])
            if self._kvq_debug and (self._kvq_log_interval == 0 or
                                    (self._kvq_token_count % self._kvq_log_interval == 0)):
                logger.info("[KVQ] append layer=%s block=%s T+=%s stats(packed=%d, scales=%d, zp=%d) items=%d",
                            int(layer_idx), int(block_id), int(k.shape[0]),
                            self._kvq_stats["bytes_packed"], self._kvq_stats["bytes_scales"],
                            self._kvq_stats["bytes_zp"], len(self._kvq_store))
            return {"block_id": int(block_id), "kvq": True}
        mgr._kvq_orig_append = orig_append
        if hasattr(mgr, "append_kv"):
            mgr.append_kv = _wrap_append.__get__(mgr, mgr.__class__)
        else:
            mgr.append = _wrap_append.__get__(mgr, mgr.__class__)

    # ---- gather wrapper: on-demand dequant for requested token window
    orig_gather = getattr(mgr, "gather_kv_slices", None) or getattr(mgr, "gather", None)
    if orig_gather is not None:
        def _wrap_gather(self, layer_idx, block_id, t_offset, t_len, device=None, *args, **kwargs):
            pay = self._kvq_store.get(_key(layer_idx, block_id))
            if pay is None:
                return self._kvq_orig_gather(layer_idx, block_id, t_offset, t_len, device, *args, **kwargs)
            H = int(pay["H"]); Dk = int(pay["Dk"]); Dv = int(pay["Dv"])
            dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            K = torch.empty((t_len, H, Dk), dtype=torch.float16, device=dev)
            V = torch.empty((t_len, H, Dv), dtype=torch.float16, device=dev)
            kpk = pay["K_packed"][t_offset : t_offset + t_len]
            vpk = pay["V_packed"][t_offset : t_offset + t_len]
            ksc = pay["K_scale"]; kzp = pay["K_zp"]
            vsc = pay["V_scale"]; vzp = pay["V_zp"]
            pol = policy_fn(int(layer_idx))
            K.copy_(_dequant_slice(kpk, ksc, kzp, pol.bits_k, pol.mode_k, last_dim=Dk, out_shape=(t_len, H, Dk)))
            V.copy_(_dequant_slice(vpk, vsc, vzp, pol.bits_v, pol.mode_v, last_dim=Dv, out_shape=(t_len, H, Dv)))
            if self._kvq_debug and (self._kvq_log_interval == 0):
                logger.info("[KVQ] gather layer=%s block=%s t=[%s:%s) -> dequant T=%s",
                            int(layer_idx), int(block_id), int(t_offset), int(t_offset + t_len), int(t_len))
            return K, V
        mgr._kvq_orig_gather = orig_gather
        if hasattr(mgr, "gather_kv_slices"):
            mgr.gather_kv_slices = _wrap_gather.__get__(mgr, mgr.__class__)
        else:
            mgr.gather = _wrap_gather.__get__(mgr, mgr.__class__)

    return True

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
    x: torch.Tensor, bits: int, group_size: int, mode: str
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, bool, bool]:
    """
    Compute per-group scale (and optional zp) for the given mode.
    Supported modes:
      - "symmetric_channel"   : scale [H,Dg], zp=None (use mid-offset at quant/dequant)
      - "asymmetric_channel"  : scale [H,Dg], zp [H,Dg]
      - "symmetric_token"     : scale [T,H,Dg], zp=None (use mid-offset)
      - "asymmetric_token"    : scale [T,H,Dg], zp [T,H,Dg]

    Returns:
      scale, zp, Dg, is_symmetric(bool), is_per_token(bool)
    """
    T, H, D = x.shape
    Dg = _ceil_div(D, group_size)

    # Pad D to multiple of group_size for group-wise reduction
    pad = (-D) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    xg = x.view(T, H, Dg, group_size).float()

    is_per_token = mode.endswith("_token")
    is_symmetric = mode.startswith("symmetric")
    Qmax_asym = (1 << bits) - 1
    Qmax_sym  = (1 << (bits - 1)) - 1

    if is_per_token:
        # Reduce over group dimension only -> [T,H,Dg]
        xmax = xg.amax(dim=3)
        xmin = xg.amin(dim=3)
        if is_symmetric:
            # symmetric: use max-abs, zp implicit (mid-offset at use sites)
            s = torch.maximum(xmax.abs(), xmin.abs())
            scale = (s / max(Qmax_sym, 1)).clamp_min(1e-8).to(torch.float16)   # [T,H,Dg]
            zp = None
        else:
            # asymmetric: use range and zp from min
            rng = (xmax - xmin)
            scale = (rng / max(Qmax_asym, 1)).clamp_min(1e-8).to(torch.float16)  # [T,H,Dg]
            zp = torch.clamp((-xmin / scale).round(), 0, Qmax_asym).to(torch.uint8)
    else:
        # per-channel: reduce over time AND group_size -> [H,Dg]
        xmax = xg.amax(dim=(0, 3))
        xmin = xg.amin(dim=(0, 3))
        if is_symmetric:
            s = torch.maximum(xmax.abs(), xmin.abs())
            scale = (s / max(Qmax_sym, 1)).clamp_min(1e-8).to(torch.float16)   # [H,Dg]
            zp = None
        else:
            rng = (xmax - xmin)
            scale = (rng / max(Qmax_asym, 1)).clamp_min(1e-8).to(torch.float16)  # [H,Dg]
            zp = torch.clamp((-xmin / scale).round(), 0, Qmax_asym).to(torch.uint8)

    return scale, zp, Dg, is_symmetric, is_per_token

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
        self.policies = policies
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
        scale_k, zp_k, Dg, sym_k, per_token_k = _compute_groupwise_scale_zp(
            k, pol.bits_k, pol.group_size, pol.mode_k
        )
        # Quantize with proper padding/reshape/broadcast to groups
        padD = (-D) % pol.group_size
        k_pad = torch.nn.functional.pad(k, (0, padD)) if padD else k
        k4 = k_pad.view(Tb, H, Dg, pol.group_size)         # [Tb,H,Dg,G]
        if per_token_k:
            sk = scale_k[..., None].float()                # [T,H,Dg,1]
            if sym_k:
                offset = (1 << (pol.bits_k - 1))
                qk4 = torch.clamp((k4.float() / sk).round() + offset, 0, (1 << pol.bits_k) - 1).to(torch.uint8)
            else:
                zk = zp_k[..., None].float()               # [T,H,Dg,1]
                qk4 = torch.clamp((k4.float() / sk).round() + zk, 0, (1 << pol.bits_k) - 1).to(torch.uint8)
        else:
            sk = scale_k[None, ..., None].float()          # [1,H,Dg,1]
            if sym_k:
                offset = (1 << (pol.bits_k - 1))
                qk4 = torch.clamp((k4.float() / sk).round() + offset, 0, (1 << pol.bits_k) - 1).to(torch.uint8)
            else:
                zk = zp_k[None, ..., None].float()         # [1,H,Dg,1]
                qk4 = torch.clamp((k4.float() / sk).round() + zk, 0, (1 << pol.bits_k) - 1).to(torch.uint8)
        qk_packed = _pack_bits(qk4.view(Tb, H, -1), pol.bits_k)  # pack along last

        # --- V ---
        scale_v, zp_v, Dg_v, sym_v, per_token_v = _compute_groupwise_scale_zp(
            v, pol.bits_v, pol.group_size, pol.mode_v
        )
        assert Dg_v == Dg
        # Same padding/reshape/broadcast for V
        v_pad = torch.nn.functional.pad(v, (0, padD)) if padD else v
        v4 = v_pad.view(Tb, H, Dg, pol.group_size)         # [Tb,H,Dg,G]
        if per_token_v:
            sv = scale_v[..., None].float()
            if sym_v:
                offset_v = (1 << (pol.bits_v - 1))
                qv4 = torch.clamp((v4.float() / sv).round() + offset_v, 0, (1 << pol.bits_v) - 1).to(torch.uint8)
            else:
                zv = zp_v[..., None].float()
                qv4 = torch.clamp((v4.float() / sv).round() + zv, 0, (1 << pol.bits_v) - 1).to(torch.uint8)
        else:
            sv = scale_v[None, ..., None].float()
            if sym_v:
                offset_v = (1 << (pol.bits_v - 1))
                qv4 = torch.clamp((v4.float() / sv).round() + offset_v, 0, (1 << pol.bits_v) - 1).to(torch.uint8)
            else:
                zv = zp_v[None, ..., None].float()
                qv4 = torch.clamp((v4.float() / sv).round() + zv, 0, (1 << pol.bits_v) - 1).to(torch.uint8)
        qv_packed = _pack_bits(qv4.view(Tb, H, -1), pol.bits_v)

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
            # symmetric_token: zp not stored (use mid-offset); asymmetric_token: store zp
            if not sym_k:
                blk.K_zp = _store_stats("K_zp", blk.K_zp, zp_k)
            else:
                blk.K_zp = None
        else:
            # per-channel: store per-layer stats once; symmetric has no zp
            blk.K_scale = scale_k
            blk.K_zp    = None if sym_k else zp_k
        if per_token_v:
            blk.V_scale = _store_stats("V_scale", blk.V_scale, scale_v)
            if not sym_v:
                blk.V_zp = _store_stats("V_zp", blk.V_zp, zp_v)
            else:
                blk.V_zp = None
        else:
            blk.V_scale = scale_v
            blk.V_zp    = None if sym_v else zp_v

        # accounting
        self.bytes_packed_total += int(qk_packed.numel() + qv_packed.numel())
        if per_token_k:
            self.bytes_scales_total += int(scale_k.numel() * 2)  # fp16
            if not sym_k:
                self.bytes_zp_total += int(zp_k.numel())
        else:
            self.bytes_scales_total += int(scale_k.numel() * 2)
            if not sym_k:
                self.bytes_zp_total += int(scale_k.numel())
        if per_token_v:
            self.bytes_scales_total += int(scale_v.numel() * 2)
            if not sym_v:
                self.bytes_zp_total += int(zp_v.numel())
        else:
            self.bytes_scales_total += int(scale_v.numel() * 2)
            if not sym_v:
                self.bytes_zp_total += int(scale_v.numel())

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
            if pol.mode_k.endswith("_token"):
                # per-token stats: [Tb,H,Dg] -> add last singleton dim for group axis
                scale_k = blk.K_scale[off:off + n_blk][..., None].float()  # [Tb,H,Dg,1]
                if pol.mode_k.startswith("asymmetric"):
                    zp_k = blk.K_zp[off:off + n_blk][..., None].float()     # [Tb,H,Dg,1]
                    kf = (k_unp.float() - zp_k) * scale_k
                else:
                    zp = float(1 << (bits_k - 1))                           # scalar mid-offset
                    kf = (k_unp.float() - zp) * scale_k
            else:
                # per-channel stats: [H,Dg] -> expand to [1,H,Dg,1]
                scale_k = blk.K_scale[None, ..., None].float()              # [1,H,Dg,1]
                if pol.mode_k.startswith("asymmetric"):
                    zp_k = blk.K_zp[None, ..., None].float()                # [1,H,Dg,1]
                    kf = (k_unp.float() - zp_k) * scale_k
                else:
                    zp = float(1 << (bits_k - 1))
                    kf = (k_unp.float() - zp) * scale_k
            Kdst[t_read:t_read + n_blk].copy_(kf.view(n_blk, st.H, st.D))

            # --- unpack V ---
            vpk = blk.V_packed[off:off + n_blk]
            v_unp = _unpack_bits(vpk, bits_v, Dg * pol.group_size).view(n_blk, st.H, Dg, pol.group_size)[..., :st.D]
            if pol.mode_v.endswith("_token"):
                scale_v = blk.V_scale[off:off + n_blk][..., None].float()   # [Tb,H,Dg,1]
                if pol.mode_v.startswith("asymmetric"):
                    zp_v = blk.V_zp[off:off + n_blk][..., None].float()     # [Tb,H,Dg,1]
                    vf = (v_unp.float() - zp_v) * scale_v
                else:
                    zp = float(1 << (bits_v - 1))
                    vf = (v_unp.float() - zp) * scale_v
            else:
                scale_v = blk.V_scale[None, ..., None].float()              # [1,H,Dg,1]
                if pol.mode_v.startswith("asymmetric"):
                    zp_v = blk.V_zp[None, ..., None].float()                # [1,H,Dg,1]
                    vf = (v_unp.float() - zp_v) * scale_v
                else:
                    zp = float(1 << (bits_v - 1))
                    vf = (v_unp.float() - zp) * scale_v
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