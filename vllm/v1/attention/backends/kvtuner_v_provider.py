# vllm/v1/attention/backends/kvtuner_v_provider.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch

_ALIGN = 16

def _align16(n: int) -> int:
    r = n % _ALIGN
    return n if r == 0 else n + (_ALIGN - r)

class KVTunerVProvider:
    """
    Process-local store for packed V and on-the-fly dequantization.
    Stores (packed, meta) per physical block id. Meta is 16B-aligned.
    """

    def __init__(self) -> None:
        # block_id -> (packed:uint8 [B], meta:uint8 [B_aligned],
        #              scale:fp16/bf16 [T], zp:uint8 [T])
        self._store: Dict[int, Tuple[torch.ByteTensor,
                                     torch.ByteTensor,
                                     torch.Tensor,
                                     torch.ByteTensor]] = {}

    @torch.no_grad()
    def note_write(
        self,
        *,
        quantizer,               # KVTunerQuantizer
        layer_idx: int,
        block_ids: torch.Tensor, # [T]
        V_tokens: torch.Tensor,  # [T, H, D] bf16/fp16
    ) -> None:
        """Quantize V (per-token asym), build 16B-aligned meta, store per block."""
        assert V_tokens.dtype in (torch.float16, torch.bfloat16)
        T, H, D = int(V_tokens.shape[0]), int(V_tokens.shape[1]), int(V_tokens.shape[2])

        # Quantize
        V_flat = V_tokens.reshape(T, H * D).contiguous()
        packed, scale, zp = quantizer.quantize_v(layer_idx, V_flat)  # returns (uint8, fp16/bf16, uint8)

        # Build minimal meta blob (16B-aligned): [T(int32), H(int32), D(int32), reserved(int32),
        #                                          then scale(fp16/bf16) * T, then zp(u8) * T, plus pad].
        meta_hdr = torch.tensor([T, H, D, 0], dtype=torch.int32, device=V_tokens.device).view(torch.uint8)
        scale_bytes = scale.view(torch.uint8)             # 2 bytes per token (fp16/bf16)
        zp_bytes = zp.view(torch.uint8)                   # 1 byte per token
        meta = torch.cat([meta_hdr, scale_bytes, zp_bytes], dim=0)
        pad = _align16(int(meta.numel())) - int(meta.numel())
        if pad:
            meta = torch.cat([meta, torch.zeros(pad, dtype=torch.uint8, device=meta.device)], dim=0)

        # Save the same segment for all blocks touched in this call (safe E2E).
        for bid in torch.unique(block_ids).tolist():
            self._store[int(bid)] = (packed, meta, scale, zp)

    @torch.no_grad()
    def dequantize_value_cache(
        self,
        *,
        quantizer,             # KVTunerQuantizer
        block_table,           # attn_metadata.block_table
        value_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
        out_dtype: torch.dtype,
        layer_idx: int,
    ) -> torch.Tensor:
        """Swap value_cache with on-the-fly dequant view if packed exists for current block."""
        # Expect current physical block id to be exposed by block_table.
        bid = getattr(block_table, "current_block_id", None)
        if bid is None:
            return value_cache
        item = self._store.get(int(bid))
        if item is None:
            return value_cache
        packed, _meta, scale, zp = item
        T = int(scale.shape[0])
        shape = (T, num_kv_heads, head_size)
        V = quantizer.dequantize_v(
            layer_idx=layer_idx,  # exact layer index (impl-injected)
            packed=packed,
            scale=scale,
            zp=zp,
            out_dtype=out_dtype,
            shape=shape,
        )
        return V
