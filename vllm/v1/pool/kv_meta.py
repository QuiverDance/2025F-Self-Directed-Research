# /vllm/v1/pool/kv_meta.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Literal
import struct
import numpy as np
import torch
from vllm.v1.serial_utils import align16

# Fixed header (exactly 16B) + 32B TOC -> 48B aligned prefix
# [0:16)  Fixed header:
#   0..3   magic = b'KVT1'
#   4..5   version (u16) = 1
#   6..7   flags (u16): low nibble(kind: 0=K,1=V), next nibble(dtype:0=fp16,1=bf16)
#   8      bits (u8)
#   9      axis (u8)
#   10..11 q_group_size (u16)
#   12..15 groups (u32)
#
# [16:48) Table-Of-Contents (TOC), little-endian int32:
#   16..19  orig_ndim
#   20..23  off_shape
#   24..27  shape_count (=orig_ndim)
#   28..31  off_scales
#   32..35  n_scales
#   36..39  off_zps
#   40..43  n_zps
#   44..47  total_len (bytes, incl. padding to 16B)
#
# Sections (each starts at 16B boundary):
#   [off_shape, off_shape+4*orig_ndim) : int32[orig_ndim]
#   [off_scales, off_scales+sizeof(scales)) : raw BF16/FP16 bytes
#   [off_zps, off_zps+sizeof(zps))         : raw BF16/FP16 bytes

MAGIC = b"KVT1"
VERSION = 1

def _dtype_code(dt: torch.dtype) -> int:
    if dt == torch.float16:
        return 0
    if dt == torch.bfloat16:
        return 1
    raise ValueError(f"Unsupported dtype for meta arrays: {dt}")

def _code_to_dtype(code: int) -> torch.dtype:
    return torch.float16 if code == 0 else torch.bfloat16

def build_kv_meta(
    *,
    kind: Literal["k", "v"],
    bits: int,
    axis: int,
    orig_shape: Tuple[int, ...],
    group_size: int,
    groups: int,
    arrays_dtype: torch.dtype,     # {fp16,bf16}
    scale: torch.Tensor,           # stored as raw bytes of arrays_dtype
    zp: torch.Tensor,              # stored as raw bytes of arrays_dtype
) -> bytes:
    assert bits in (1, 2, 4, 8)
    kind_code = 0 if kind == "k" else 1
    flags = (kind_code & 0xF) | ((_dtype_code(arrays_dtype) & 0xF) << 4)

    # Fixed header (16B)
    hdr = struct.pack(
        "<4sH H B B H I",
        MAGIC, VERSION, flags, bits & 0xFF, axis & 0xFF,
        group_size & 0xFFFF, groups & 0xFFFFFFFF,
    )
    assert len(hdr) == 16

    # Prepare sections
    shape = tuple(int(x) for x in orig_shape)
    shape_bytes = struct.pack("<" + "i" * len(shape), *shape)

    # scale/zp → raw bytes in arrays_dtype (no FP32 allowed)
    scale_b = scale.to(arrays_dtype).contiguous().view(torch.uint8).cpu().numpy().tobytes()
    zp_b    = zp.to(arrays_dtype).contiguous().view(torch.uint8).cpu().numpy().tobytes()

    # TOC (we fill offsets after placing sections)
    # Reserve 32B
    toc = bytearray(32)
    toc_view = memoryview(toc)
    # orig_ndim
    struct.pack_into("<i", toc_view, 0, len(shape))

    # Layout after 48B prefix
    off = 48
    # shape
    off_shape = off
    off += len(shape_bytes)
    off = align16(off)
    # scales
    off_scales = off
    off += len(scale_b)
    off = align16(off)
    # zps
    off_zps = off
    off += len(zp_b)
    off = align16(off)
    total_len = off

    # Fill TOC
    struct.pack_into("<i", toc_view,  4, off_shape)
    struct.pack_into("<i", toc_view,  8, len(shape))
    struct.pack_into("<i", toc_view, 12, off_scales)
    struct.pack_into("<i", toc_view, 16, scale.numel())
    struct.pack_into("<i", toc_view, 20, off_zps)
    struct.pack_into("<i", toc_view, 24, zp.numel())
    struct.pack_into("<i", toc_view, 28, total_len)

    # Assemble
    buf = bytearray(total_len)
    buf[0:16] = hdr
    buf[16:48] = toc
    buf[off_shape:off_shape + len(shape_bytes)] = shape_bytes
    s0 = off_scales
    buf[s0:s0 + len(scale_b)] = scale_b
    z0 = off_zps
    buf[z0:z0 + len(zp_b)] = zp_b
    return bytes(buf)

def parse_kv_meta(meta: bytes, device: torch.device | None = None) -> Dict[str, Any]:
    """Parse meta bytes → dict with header fields and tensors."""
    assert len(meta) >= 48
    # Header
    (magic, version, flags, bits, axis, qgs, groups) = struct.unpack_from(
        "<4sH H B B H I", meta, 0)
    assert magic == MAGIC and version == VERSION
    kind_code = flags & 0xF
    arrays_dtype = _code_to_dtype((flags >> 4) & 0xF)
    kind = "k" if kind_code == 0 else "v"
    # TOC
    (orig_ndim, off_shape, shape_cnt, off_scales,
     n_scales, off_zps, n_zps, total_len) = struct.unpack_from("<iiiiiiii", meta, 16)
    assert shape_cnt == orig_ndim and total_len <= len(meta)
    # Sections
    shape = tuple(struct.unpack_from("<" + "i" * orig_ndim, meta, off_shape))

    # Raw → tensors
    def _bytes_to_tensor(offset: int, count: int, dt: torch.dtype) -> torch.Tensor:
        nbytes = count * torch.tensor([], dtype=dt).element_size()
        raw = memoryview(meta)[offset: offset + nbytes]
        np_arr = np.frombuffer(raw, dtype=np.uint8).copy()  # own the memory
        t = torch.from_numpy(np_arr).to(dtype=torch.uint8)
        t = t.contiguous()  # ensure contiguous before view
        try:
            t = t.view(dt)  # reinterpret to fp16/bf16 (no FP32)
            t = t.clone()  # ensure we own the memory
            if device is not None:
                t = t.to(device=device, non_blocking=True)  # allow async transfer
            return t
        except RuntimeError as e:
            raise RuntimeError(f"Failed to convert bytes to tensor. Shape: {t.shape}, "
                             f"Target dtype: {dt}, Count: {count}, Bytes: {nbytes}") from e

    scale = _bytes_to_tensor(off_scales, n_scales, arrays_dtype)
    zp    = _bytes_to_tensor(off_zps,    n_zps,    arrays_dtype)

    # # =========== START: ADD DEBUGGING CODE ===========
    # print(f"[KVTuner:DBG][PARSE_META] kind={kind}, axis={axis}, groups={groups}, orig_shape={shape}")
    # print(f"[KVTuner:DBG][PARSE_META] Raw scale shape: {scale.shape}, zp shape: {zp.shape}")
    # =========== END: ADD DEBUGGING CODE =============

    # === Restore shapes for broadcasting ===
    # FIX: Restore the special handling for V-cache (per-token, axis=0)
    if kind == "v" and axis == 0:
        # For V-cache, we expect scale/zp to have a 1D shape of [T], where T is the number of groups.
        assert scale.numel() == groups, f"V-cache scale.numel() {scale.numel()} != groups {groups}"
        assert zp.numel() == groups, f"V-cache zp.numel() {zp.numel()} != groups {groups}"
        # Ensure contiguous and proper shape [groups]
        scale = scale.contiguous().view(groups)
        zp = zp.contiguous().view(groups)
    else:
        # Generic handling for K-cache (per-channel)
        ndim = len(shape)
        ax = axis if axis >= 0 else axis + ndim
        other_dims = [shape[i] for i in range(ndim) if i != ax]
        target_shape = tuple(other_dims + [int(groups)])
        
        expected_numel = int(np.prod(other_dims, dtype=np.int64)) * int(groups)
        assert scale.numel() == expected_numel, \
            f"scale.numel({scale.numel()}) != prod(other_dims)*G ({expected_numel})"
        assert zp.numel() == scale.numel(), "zp/scale numel mismatch"
        
        scale = scale.contiguous().view(*target_shape)
        zp = zp.contiguous().view(*target_shape)

    # =========== START: ADD DEBUGGING CODE ===========
    # print(f"[KVTuner:DBG][PARSE_META] Final scale shape: {scale.shape}, zp shape: {zp.shape}")
    # =========== END: ADD DEBUGGING CODE =============

    return {
        "kind": kind, "bits": bits, "axis": axis,
        "group_size": qgs, "groups": groups,
        "orig_shape": shape,
        "arrays_dtype": arrays_dtype,
        "scale": scale, "zp": zp,
    }
