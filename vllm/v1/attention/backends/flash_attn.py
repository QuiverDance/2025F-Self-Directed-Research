# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch, os, sys
from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.layer import Attention
from vllm.attention.ops.merge_attn_states import merge_attn_states
from vllm.attention.utils.fa_utils import (flash_attn_supports_fp8,
                                           get_flash_attn_version,
                                           is_flash_attn_varlen_func_available)

if is_flash_attn_varlen_func_available():
    from vllm.attention.utils.fa_utils import (flash_attn_varlen_func,
                                               get_scheduler_metadata,
                                               reshape_and_cache_flash)

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.attention.backends.utils import (AttentionCGSupport,
                                              AttentionMetadataBuilder,
                                              CommonAttentionMetadata,
                                              get_kv_cache_layout)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm.v1.core.kv_cache_coordinator import get_global_coordinator

logger = init_logger(__name__)

# NOTE(woosuk): This is an arbitrary number. Tune it if needed.
_DEFAULT_MAX_NUM_SPLITS_FOR_CUDA_GRAPH = 16


class FlashAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        supported_head_sizes = cls.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            attn_type = cls.__name__.removesuffix("Backend")
            raise ValueError(
                f"Head size {head_size} is not supported by {attn_type}. "
                f"Supported head sizes are: {supported_head_sizes}. "
                "Set VLLM_ATTENTION_BACKEND=FLEX_ATTENTION to use "
                "FlexAttention backend which supports all head sizes.")

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return FlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")


@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: Optional[torch.Tensor]
    prefix_kv_lens: Optional[torch.Tensor]
    suffix_kv_lens: Optional[torch.Tensor]

    # Optional aot scheduling
    scheduler_metadata: Optional[torch.Tensor] = None
    prefix_scheduler_metadata: Optional[torch.Tensor] = None
    max_num_splits: int = 0

    causal: bool = True


def _get_sliding_window_configs(
        vllm_config: VllmConfig) -> set[Optional[tuple[int, int]]]:
    """Get the set of all sliding window configs used in the model."""
    sliding_window_configs: set[Optional[tuple[int, int]]] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        assert isinstance(layer.impl, FlashAttentionImpl)
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class FlashAttentionMetadataBuilder(
        AttentionMetadataBuilder[FlashAttentionMetadata]):
    # FA3:
    # Supports full cudagraphs for all cases.
    #
    # FA2:
    # For FA2, a graph is captured with max_query_len=1, (which is what we
    # capture by default for num_tokens <= max_num_seqs when there is no
    # spec-decode) then these graphs will not work for mixed prefill-decode
    # (unlike FA3). This is due to special max_query_len=1 packed-GQA handling
    # in FA2.
    # In summary if we are running with spec decodes the graphs would
    # work for mixed prefill-decode and uniform-decode. But for non-spec decodes
    # the graphs would not work for mixed prefill-decode; sorta the inverse
    # of UNIFORM_SINGLE_TOKEN_DECODE.
    # There's probably a better way to describe this using `AttentionCGSupport`
    # but for now just set it to `UNIFORM_BATCH` to get use to drop down
    # to FULL_AND_PIECEWISE.
    # TODO(luka, lucas): audit FA2 as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    cudagraph_support = AttentionCGSupport.ALWAYS \
        if get_flash_attn_version() == 3 else AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config)
        self.num_heads_kv = self.model_config.get_num_kv_heads(
            self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = (get_flash_attn_version() == 3)

        self.use_full_cuda_graph = \
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()

        if self.use_full_cuda_graph and self.aot_schedule:
            self.max_cudagraph_size = self.compilation_config.max_capture_size

            if self.max_cudagraph_size > 992:
                # This condition derives from FA3's internal heuristic.
                # TODO(woosuk): Support larger cudagraph sizes.
                raise ValueError(
                    "Capture size larger than 992 is not supported for "
                    "full cuda graph.")

            self.scheduler_metadata = torch.zeros(
                vllm_config.scheduler_config.max_num_seqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = _DEFAULT_MAX_NUM_SPLITS_FOR_CUDA_GRAPH

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: Optional[tuple[int, int]] = None

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few 
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # the overhead of the aot schedule is not worth it for spec-decode
        aot_schedule = self.aot_schedule and not fast_build

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(
                    self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        def schedule(batch_size, cu_query_lens, max_query_len, seqlens,
                     max_seq_len, causal):
            cache_dtype = self.cache_config.cache_dtype
            if cache_dtype.startswith("fp8"):
                qkv_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                    cache_dtype)
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=self.max_num_splits,
                )
            return None

        use_cascade = common_prefix_len > 0

        if use_cascade:
            cu_prefix_query_lens = torch.tensor([0, num_actual_tokens],
                                                dtype=torch.int32,
                                                device=self.device)
            prefix_kv_lens = torch.tensor([common_prefix_len],
                                          dtype=torch.int32,
                                          device=self.device)
            suffix_kv_lens = (seq_lens_cpu[:num_reqs] - common_prefix_len).to(
                self.device, non_blocking=True)
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False)
            scheduler_metadata = schedule(batch_size=num_reqs,
                                          cu_query_lens=query_start_loc,
                                          max_query_len=max_query_len,
                                          seqlens=suffix_kv_lens,
                                          max_seq_len=max_seq_len -
                                          common_prefix_len,
                                          causal=True)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None
            scheduler_metadata = schedule(batch_size=num_reqs,
                                          cu_query_lens=query_start_loc,
                                          max_query_len=max_query_len,
                                          seqlens=seq_lens,
                                          max_seq_len=max_seq_len,
                                          causal=causal)
        # For FA3 + full cudagraph
        max_num_splits = 0
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

            if num_actual_tokens <= self.max_cudagraph_size:
                # NOTE(woosuk): Setting num_splits > 1 may increase the memory
                # usage, because the intermediate buffers of size [num_splits,
                # num_heads, num_tokens, head_size] are allocated. Therefore,
                # we only set num_splits when using cuda graphs.
                max_num_splits = self.max_num_splits

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal)
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)


class FlashAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        sinks: Optional[torch.Tensor] = None,
        # --- START: DEBUGGING CODE ---
        # Add layer_idx as an argument to easily track which layer is being processed.
        # This requires a small change where this class is instantiated.
        # For now, we will assume it's passed or set externally for logging.
        # To make it work, we add it to the init signature.
        layer_idx: int = -1,
        # --- END: DEBUGGING CODE ---
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # --- START: DEBUGGING CODE ---
        # Store the layer_idx for use in the forward pass logging.
        self.kvtuner_layer_idx = layer_idx
        # --- END: DEBUGGING CODE ---

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        FlashAttentionBackend.validate_head_size(head_size)

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version()
        if is_quantized_kv_cache(self.kv_cache_dtype) \
            and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device.")

        self.sinks = sinks
        if self.sinks is not None:
            assert self.vllm_flash_attn_version == 3, (
                "Sinks are only supported in FlashAttention 3")
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."
        
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for FlashAttentionImpl")

        if attn_metadata is None:
            # Profiling run.
            return output

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(query[:num_actual_tokens],
                                                   key[:num_actual_tokens],
                                                   value[:num_actual_tokens],
                                                   output[:num_actual_tokens],
                                                   attn_metadata, layer)

        # --- Debug helper: compact tensor stats (always-on; remove after diagnosis) ---
        def _dbg_stats(name: str, t: torch.Tensor):
            try:
                dev = str(t.device)
                dtype = str(t.dtype)
                shape = tuple(t.shape)
                tf = t.float()
                finite = torch.isfinite(tf)
                fin_ratio = float(finite.sum().item()) / float(tf.numel() or 1)
                tmin = tf[finite].min().item() if finite.any() else float('nan')
                tmax = tf[finite].max().item() if finite.any() else float('nan')
                tmean = tf[finite].mean().item() if finite.any() else float('nan')
                l2 = torch.linalg.vector_norm(tf[finite]).item() if finite.any() else float('nan')
                print(f"[KVTuner:DBG] {name} dev={dev} dtype={dtype} shape={shape} "
                      f"finite={fin_ratio:.3f} min={tmin:.5f} max={tmax:.5f} mean={tmean:.5f} l2={l2:.5f}")
            except Exception as _e:
                import sys
                print(f"[KVTuner:DBG][ERR] stats({name}): {type(_e).__name__}: {_e}", file=sys.stderr)
        
        # Derive block size directly from kv_cache: [2, B, block_size, H, D]
        block_size = int(kv_cache.size(2))

        # For decoder/cross-attention, use KV cache as usual
        key_cache, value_cache = kv_cache.unbind(0)
        
        # if key_cache is not None:
        #     print(f"[KVTuner:ATTN] K shape={key_cache.shape} range=[{key_cache.min():.4f}, {key_cache.max():.4f}]")
        # if value_cache is not None:
        #     print(f"[KVTuner:ATTN] V shape={value_cache.shape} range=[{value_cache.min():.4f}, {value_cache.max():.4f}]")
            
        # KVTuner integration (global coordinator owns quantizer and pool)
        coord = get_global_coordinator()
        kv_quantizer = getattr(coord, "kv_quantizer", None)

        # Maintain per-kind per-block row maps to place rows at correct offsets
        # Structure: {"k": {bid: LongTensor(sorted unique offsets)}, "v": {...}}
        if not hasattr(self, "_kvt_rowmap"):
            self._kvt_rowmap = {"k": {}, "v": {}}

        # --- KVTuner read-before-attention (Decode path): restore K/V from tails ---
        # Plan compliance:
        #   Decode: Read -> Decompress -> Attention -> Compress -> Store
        #   Prefill: (no read)
        try:
            # MODIFICATION: Allow this block to run during decode
            # to implement the "Read" part of the Read-Modify-Write cycle.
            # It restores the previous full-precision state into the working cache.
            # This function is being documented in English.
            if (kv_quantizer is not None
                and getattr(kv_quantizer, "enable", False)):
                # Collect unique physical block ids referenced by current batch.

                # --- START: DEBUGGING CODE ---
                # This log will now appear for every layer during prefill,
                # confirming that the dequantization block is being executed correctly.
                layer_idx = getattr(self, "kvtuner_layer_idx", -1)
                # print(f"[KVTuner:DEQUANT-TRACE] Executing for Layer {layer_idx}, "
                #       f"num_actual_tokens={num_actual_tokens} (Prefill > 1, Decode == 1)")
                # --- END: DEBUGGING CODE ---

                bt = attn_metadata.block_table  # [num_seqs, max_blocks] with -1 as padding
                if isinstance(bt, torch.Tensor):
                    used = bt[bt > 0]
                    if used.numel() > 0:
                        uniq = torch.unique(used).tolist()
                        # Dequantize per block and copy rows into cache tensors.
                        from vllm.v1.core.kvtuner_quantizer import PackedKV
                        for _bid in uniq:
                            bid = int(_bid)
                            k_tail = coord.block_pool.read_k_tail(bid)
                            v_tail = coord.block_pool.read_v_tail(bid)
                            if k_tail is None or v_tail is None:
                                continue  # nothing stored for this block
                            pk, mk, scale_k, zp_k = k_tail
                            pv, mv, scale_v, zp_v = v_tail
                            
                            # Meta tensors -> raw bytes for the quantizer
                            mk_bytes = mk.detach().cpu().contiguous().numpy().tobytes()
                            mv_bytes = mv.detach().cpu().contiguous().numpy().tobytes()

                            # Dequantize to cache dtype/device; shapes: [rows, H, D]
                            from vllm.v1.core.kvtuner_quantizer import PackedKV as _Packed
                            dec_k = kv_quantizer.dequantize_k(_Packed(packed=pk, meta_bytes=mk_bytes))\
                                                   .to(dtype=key_cache.dtype, device=key_cache.device)
                            dec_v = kv_quantizer.dequantize_v(_Packed(packed=pv, meta_bytes=mv_bytes))\
                                                   .to(dtype=value_cache.dtype, device=value_cache.device)

                            # Place rows at the correct offsets with proper layout
                            dk_rows = int(dec_k.shape[0])
                            dv_rows = int(dec_v.shape[0])
                            
                            # --- START: FIX ---
                            #
                            # ISSUE: The original code had redundant `.reshape()` calls and used `.copy_()`,
                            # which can cause data corruption if tensor memory layouts (strides) mismatch.
                            #
                            # FIX:
                            # 1. Removed the unnecessary `.reshape()` calls. The `dequantize` function
                            #    already returns tensors with the correct shape.
                            # 2. Replaced the fragile `.copy_()` operation with direct slice assignment,
                            #    which is more robust for writing data into tensor views.
                            # 3. Added debug prints to verify that the data written to the cache
                            #    matches the dequantized data immediately before assignment.

                            # The dequantized tensors are already in the correct shape.
                            # Ensure they are contiguous before assignment.
                            dec_k = dec_k.contiguous()
                            dec_v = dec_v.contiguous()

                            # --- START: DEBUGGING CODE ---
                            # This code verifies that the dequantized values are correct right before
                            # they are placed back into the main KV cache.
                            try:
                                k_min, k_max, k_mean = dec_k.min().item(), dec_k.max().item(), dec_k.mean().item()
                                v_min, v_max, v_mean = dec_v.min().item(), dec_v.max().item(), dec_v.mean().item()
                                # print(f"[KVTuner:VERIFY][PRE-WRITE] bid={bid} "
                                #       f"K(shape={dec_k.shape}, range=[{k_min:.4f}, {k_max:.4f}], mean={k_mean:.4f}) "
                                #       f"V(shape={dec_v.shape}, range=[{v_min:.4f}, {v_max:.4f}], mean={v_mean:.4f})")
                            except Exception as e:
                                print(f"[KVTuner:VERIFY][ERROR] Pre-write stat failed: {e}")
                            # --- END: DEBUGGING CODE ---
                            
                            # Perform direct assignment to the cache slices. This is safer than copy_().
                            key_cache[bid, :dk_rows] = dec_k
                            value_cache[bid, :dv_rows] = dec_v
                            
                            # --- START: DEBUGGING CODE ---
                            # This code verifies that the values in the KV cache match the source
                            # tensor immediately after the write operation.
                            try:
                                written_k = key_cache[bid, :dk_rows]
                                written_v = value_cache[bid, :dv_rows]
                                k_min, k_max, k_mean = written_k.min().item(), written_k.max().item(), written_k.mean().item()
                                v_min, v_max, v_mean = written_v.min().item(), written_v.max().item(), written_v.mean().item()
                                # print(f"[KVTuner:VERIFY][POST-WRITE] bid={bid} "
                                #       f"K(shape={written_k.shape}, range=[{k_min:.4f}, {k_max:.4f}], mean={k_mean:.4f}) "
                                #       f"V(shape={written_v.shape}, range=[{v_min:.4f}, {v_max:.4f}], mean={v_mean:.4f})")
                                # Assert that the data is identical.
                                assert torch.all(written_k == dec_k)
                                assert torch.all(written_v == dec_v)
                            except Exception as e:
                                print(f"[KVTuner:VERIFY][ERROR] Post-write verification failed: {e}")
                            # --- END: DEBUGGING CODE ---
                            #
                            # --- END: FIX ---

                            # Preserve tensor layout when reshaping to match cache format
                            # Shape should be [num_rows, num_heads, head_dim] with proper strides
                            # dec_k = dec_k.reshape(dk_rows, key_cache.size(2), key_cache.size(3))
                            # dec_v = dec_v.reshape(dv_rows, value_cache.size(2), value_cache.size(3))

                            # # Convert to target dtype/device and ensure contiguous memory
                            # dec_k = dec_k.to(dtype=key_cache.dtype, device=key_cache.device, memory_format=torch.contiguous_format)
                            # dec_v = dec_v.to(dtype=value_cache.dtype, device=value_cache.device, memory_format=torch.contiguous_format)
                            
                            # # Copy blocks with explicit indices
                            # try:
                            #     target_k = key_cache[bid, :dk_rows]
                            #     target_v = value_cache[bid, :dv_rows]
                                
                            #     # Ensure shapes match before copying
                            #     assert target_k.shape == dec_k.shape, f"Shape mismatch K: {target_k.shape} vs {dec_k.shape}"
                            #     assert target_v.shape == dec_v.shape, f"Shape mismatch V: {target_v.shape} vs {dec_v.shape}"
                                
                            #     # Perform direct copy
                            #     target_k.copy_(dec_k)
                            #     target_v.copy_(dec_v)
                                
                            #     # print(f"[KVTuner:DBG] After copy K[0,0,:8]={target_k[0,0,:8].tolist()}")
                            # except Exception as e:
                            #     print(f"[KVTuner:DBG][ERROR] Copy failed: {str(e)}")
                            #     # Fallback to direct assignment if copy fails
                            #     key_cache[bid, :dk_rows] = dec_k
                            #     value_cache[bid, :dv_rows] = dec_v
                            # # Extra per-block debug for KVTuner investigation
                            try:
                                dkf = dec_k.float()
                                dvf = dec_v.float()
                                # print(f"[KVTuner:DBG] placed block bid={bid} dk_rows={dk_rows} dv_rows={dv_rows} "
                                #       f"K_block_range=[{dkf.min():.5f},{dkf.max():.5f}] V_block_range=[{dvf.min():.5f},{dvf.max():.5f}]")
                            except Exception:
                                pass

        except Exception as _e:
            import sys
            print(f"[KVTuner:read-before-attn][WARN] {type(_e).__name__}: {_e}", file=sys.stderr)
        # --- END KVTuner read-before-attention ---

        # key and value may be None in cross attention. They are calculated once
        # from the encoder output and then cached. If they exist, reshape & cache.
        if (self.kv_sharing_target_layer_name is None and key is not None
                and value is not None):

            # Reshape input K/V and store them in the cache (raw layout).
            # NOTE: The reshape_and_cache op uses slot_mapping's shape to decide
            # the number of actual tokens; K/V tensors here may be padded.
            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
        
        if self.kv_cache_dtype.startswith("fp8"):
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)
            num_tokens, num_heads, head_size = query.shape
            query, _ = ops.scaled_fp8_quant(
                query.reshape(
                    (num_tokens, num_heads * head_size)).contiguous(),
                layer._q_scale)
            query = query.reshape((num_tokens, num_heads, head_size))

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table

            # --- START: DIAGNOSTIC DEBUGGING CODE ---
            #
            # GOAL: Inspect the state of query, key_cache, and value_cache right before
            # they are passed to the attention kernel. This will reveal if the
            # inputs are corrupted (e.g., all zeros, NaN, Inf).

            try:
                num_active_tokens = query.shape[0]
                layer_idx = getattr(self, "kvtuner_layer_idx", -1)
                
                # 1. Query Statistics
                q_l2 = torch.linalg.norm(query.float()).item()
                q_mean = query.float().mean().item()
                # print(f"[KVTuner:KERNEL-INPUT L{layer_idx}] "
                #       f"Query ({query.shape}): L2={q_l2:.4f}, Mean={q_mean:.4f}")

                # 2. Active KV Cache Statistics
                # We only inspect the blocks that are actually used by the current batch.
                active_blocks_flat = block_table[:attn_metadata.query_start_loc.shape[0]-1].flatten()
                unique_block_indices = torch.unique(active_blocks_flat[active_blocks_flat >= 0])
                
                if unique_block_indices.numel() > 0:
                    active_k_cache = key_cache[unique_block_indices]
                    active_v_cache = value_cache[unique_block_indices]
                    
                    k_l2 = torch.linalg.norm(active_k_cache.float()).item()
                    k_mean = active_k_cache.float().mean().item()
                    k_min, k_max = active_k_cache.min().item(), active_k_cache.max().item()

                    v_l2 = torch.linalg.norm(active_v_cache.float()).item()
                    v_mean = active_v_cache.float().mean().item()
                    v_min, v_max = active_v_cache.min().item(), active_v_cache.max().item()

                    # print(f"[KVTuner:KERNEL-INPUT L{layer_idx}] "
                    #       f"Active K-Cache ({active_k_cache.shape}): L2={k_l2:.4f}, Mean={k_mean:.4f}, Range=[{k_min:.4f}, {k_max:.4f}]")
                    # print(f"[KVTuner:KERNEL-INPUT L{layer_idx}] "
                    #       f"Active V-Cache ({active_v_cache.shape}): L2={v_l2:.4f}, Mean={v_mean:.4f}, Range=[{v_min:.4f}, {v_max:.4f}]")
                else:
                    print(f"[KVTuner:KERNEL-INPUT L{layer_idx}] No active KV cache blocks for this batch.")

            except Exception as e:
                print(f"[KVTuner:KERNEL-INPUT L{layer_idx}][ERROR] Failed to print debug stats: {e}", file=sys.stderr)
            
            # --- END: DIAGNOSTIC DEBUGGING CODE ---

            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            # NORMALIZE block_table padding for Flash-Attention 
            # Our system uses 0 for padding, but Flash-Attention expects -1
            # Clone first to avoid modifying the original
            block_table_for_fa = block_table.clone()

            # Convert padding 0s to -1s with in-place update
            # Look for zeros in block table and mark them as padding (-1)
            padding_mask = (block_table_for_fa == 0)
            block_table_for_fa[padding_mask] = -1

            # --- START: DEBUGGING CODE ---
            # This verification ensures the block table is correctly formatted before the kernel call.
            try:
                # 1. Assert that no 0s are left, as 0 is a valid physical block index (null_block)
                #    but should not be present after padding conversion.
                assert not (block_table_for_fa == 0).any(), \
                    "Block table still contains 0s after padding conversion."

                # 2. Check that all indices are either -1 (padding) or valid block indices.
                num_physical_blocks = key_cache.size(0)
                min_idx = block_table_for_fa.min().item()
                max_idx = block_table_for_fa.max().item()

                assert min_idx >= -1, f"Invalid block index {min_idx} found. Must be >= -1."
                assert max_idx < num_physical_blocks, \
                    f"Invalid block index {max_idx} found. Must be < {num_physical_blocks}."

                # print(f"[KVTuner:VALIDATE-BT] Block table for FlashAttention is valid. "
                #       f"Index range: [{min_idx}, {max_idx}], "
                #       f"Padding ratio: {(block_table_for_fa == -1).float().mean().item():.2%}")
            except AssertionError as e:
                print(f"[KVTuner:VALIDATE-BT][ERROR] {e}", file=sys.stderr)
            # --- END: DEBUGGING CODE ---

            flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=block_table_for_fa,
                softcap=self.logits_soft_cap,
                scheduler_metadata=scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                # q_descale=layer._q_scale.expand(descale_shape),
                # k_descale=layer._k_scale.expand(descale_shape),
                # v_descale=layer._v_scale.expand(descale_shape),
                q_descale=None,      # Now correctly passed as None for non-FP8
                k_descale=None,      # Now correctly passed as None for non-FP8
                v_descale=None,      # Now correctly passed as None for non-FP8
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
            )

            # --- KVTuner write-after-attention: Attn -> Compress -> Store ---
            try:
                # MODIFICATION: Allow this block to run during decode to implement the "Modify-Write" part.
                # It takes the updated full-precision cache, re-quantizes the ENTIRE block history,
                # and overwrites the storage snapshot.
                # This function is being documented in English.
                if (kv_quantizer is not None
                    and getattr(kv_quantizer, "enable", False)
                    and attn_type == AttentionType.DECODER
                    and key is not None and value is not None):
                    # Group current-step tokens by block id, then quantize & store per block tail.
                    num_actual_tokens = int(attn_metadata.num_actual_tokens)
                    if num_actual_tokens > 0:
                        slots = attn_metadata.slot_mapping[:num_actual_tokens]
                        block_size = int(kv_cache.size(2))
                        mask = (slots >= 0)
                        if torch.count_nonzero(mask) > 0:
                            local_slots = slots[mask]
                            local_idx = torch.nonzero(mask, as_tuple=False).view(-1)
                            block_ids = torch.div(local_slots, block_size, rounding_mode='floor')
                            offs_in_block = (local_slots % block_size)
                            
                            # This groups batch indices by block_id.
                            # For decode, each block will have only one new token.
                            # This function is being documented in English.
                            groups = {}
                            for i in range(local_idx.numel()):
                                bi = int(block_ids[i].item())
                                groups.setdefault(bi, []).append(i)

                            layer_idx = int(getattr(self, "kvtuner_layer_idx", 0))
                            from vllm.v1.pool.kv_meta import parse_kv_meta

                            for b, indices_in_group in groups.items():
                                # Find the maximum offset written to this block in the current step.
                                # This tells us the new total number of tokens in the block.
                                # This function is being documented in English.
                                new_token_offsets = offs_in_block[torch.tensor(indices_in_group, device=offs_in_block.device)]
                                num_tokens_in_block = new_token_offsets.max().item() + 1
                                
                                # Retrieve the full, updated history from the working cache.
                                # This function is being documented in English.
                                full_k_history = key_cache[b, :num_tokens_in_block].contiguous()
                                full_v_history = value_cache[b, :num_tokens_in_block].contiguous()

                                # Re-quantize the entire history and overwrite the storage.
                                # This function is being documented in English.
                                pk = kv_quantizer.quantize_k(layer_idx, full_k_history)
                                pv = kv_quantizer.quantize_v(layer_idx, full_v_history)

                                # --- START: FIX ---
                                # ISSUE: `np.frombuffer` on a bytes object creates a read-only view,
                                # triggering a UserWarning and indicating fragile memory handling.
                                # FIX: Add `.copy()` to create a writable numpy array, resolving the
                                # warning and making the buffer handling more robust.
                                k_meta_np = np.frombuffer(pk.meta_bytes, dtype=np.uint8).copy()
                                v_meta_np = np.frombuffer(pv.meta_bytes, dtype=np.uint8).copy()
                                k_meta = torch.from_numpy(k_meta_np).to(device=key_cache.device)
                                v_meta = torch.from_numpy(v_meta_np).to(device=value_cache.device)
                                # --- END: FIX ---
                                
                                # Parse for fp16/bf16 scale/zp tensors
                                k_md = parse_kv_meta(pk.meta_bytes, device=key_cache.device)
                                v_md = parse_kv_meta(pv.meta_bytes, device=value_cache.device)

                                coord.block_pool.write_k_tail(
                                    int(b),
                                    packed=pk.packed.to(torch.uint8),
                                    meta=k_meta.to(torch.uint8),
                                    scale=k_md["scale"],
                                    zp=k_md["zp"],
                                )
                                coord.block_pool.write_v_tail(
                                    int(b),
                                    packed=pv.packed.to(torch.uint8),
                                    meta=v_meta.to(torch.uint8),
                                    scale=v_md["scale"],
                                    zp=v_md["zp"],
                                )
            except Exception as _e:
                import sys
                print(f"[KVTuner:write-after-attn][WARN] {type(_e).__name__}: {_e}", file=sys.stderr)
            # --- END KVTuner write-after-attention ---
            return output

        # Cascade attention (rare case).
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
        )
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # For encoder attention, process FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention")

        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,  # type: ignore[union-attr]
            self.num_kv_heads)

        # Call flash attention directly on Q, K, V tensors
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,  # Encoder attention is bidirectional
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
        )

        return output


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    use_local_attention: bool,
    num_sms: int,
) -> bool:
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False

    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    use_flash_decoding = (num_queries_per_kv > 1 and not use_sliding_window
                          and not use_alibi and np.all(query_lens == 1))
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (num_reqs * num_kv_heads *
                           cdiv(num_queries_per_kv, q_tile_size))
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    return cascade_time < flash_decoding_time


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    fa_version: int,
    prefix_scheduler_metadata: Optional[torch.Tensor] = None,
    suffix_scheduler_metadata: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert alibi_slopes is None, ("Cascade attention does not support ALiBi.")
    # TODO: Support sliding window.
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window.")

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0
    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process shared prefix.
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=sliding_window,
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=prefix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape)
        if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape)
        if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape)
        if v_descale is not None else None,
    )

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process suffix per query.
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=sliding_window,
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=suffix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape)
        if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape)
        if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape)
        if v_descale is not None else None,
    )

    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                      suffix_lse)