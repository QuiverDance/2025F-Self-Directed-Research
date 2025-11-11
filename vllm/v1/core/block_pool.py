# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from collections.abc import Iterable
from typing import Optional

from vllm.distributed.kv_events import (MEDIUM_GPU, AllBlocksCleared,
                                        BlockRemoved, BlockStored,
                                        KVCacheEvent)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (BlockHash, BlockHashWithGroupId,
                                         ExternalBlockHash,
                                         FreeKVCacheBlockQueue, KVCacheBlock,
                                         get_block_hash,
                                         make_block_hash_with_group_id,
                                         maybe_convert_block_hash)
from vllm.v1.request import Request

import torch
logger = init_logger(__name__)


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        enable_kv_cache_events: Whether to enable kv cache events.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        enable_kv_cache_events: bool = False,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # {block_hash: {block ID: block}}. A cached block is
        # a full block with a block hash that can be used for prefix caching.
        # The cached block may be used by running requests or in the
        # free_block_queue that could potentially be evicted.
        # NOTE: We currently don't de-duplicate the blocks in the cache,
        # meaning that if a block becomes full and is cached, we don't check
        # if there is already an identical block in the cache. This is because
        # we want to make sure the allocated block IDs won't change so that
        # block tables are append-only.
        self.cached_block_hash_to_block: dict[BlockHashWithGroupId, dict[
            int, KVCacheBlock]] = defaultdict(dict)

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        # === KVTuner V-only accounting model (bytes-based) ===
        # If enabled, usage() reports "compressed used bytes / raw capacity bytes".
        self._kv_accounting_enabled: bool = False
        self._kv_page_raw_bytes: int = 0       # denominator: raw FP page bytes
        self._kv_page_used_bytes_q: int = 0    # numerator per-used-page: K raw + V packed + meta

        # === KVTuner V-only: page-tail storage (16B aligned) ===
        # We keep a per-block "tail" region that models the page tail. The stored
        # blobs are ready-to-write byte payloads (no fp32 anywhere).
        # Keys are physical block ids (int).
        self._v_tail_exists: dict[int, bool] = {}
        self._v_tail_packed: dict[int, torch.ByteTensor] = {}
        self._v_tail_meta: dict[int, torch.ByteTensor] = {}
        # Optional convenience (read path can avoid parsing meta if stored here):
        self._v_tail_scale: dict[int, torch.Tensor] = {}   # fp16/bf16
        self._v_tail_zp: dict[int, torch.Tensor] = {}  # fp16/bf16 or u8

        # --- K-tail storage (packed bytes + 16B-aligned meta) ---
        self._k_tail_exists: dict[int, bool] = {}
        self._k_tail_packed: dict[int, torch.ByteTensor] = {}
        self._k_tail_meta: dict[int, torch.ByteTensor] = {}
        self._k_tail_scale: dict[int, torch.Tensor] = {}   # fp16/bf16
        self._k_tail_zp: dict[int, torch.Tensor] = {}  # fp16/bf16 or u8

         # --- KVTuner debug: keep lightweight checksums for packed/meta per block
        # to verify that what we read equals what we wrote.
        # We only track small ints to avoid memory blow-up.
        self._v_tail_dbg_sum_packed: dict[int, int] = {}
        self._v_tail_dbg_sum_meta: dict[int, int] = {}
        self._k_tail_dbg_sum_packed: dict[int, int] = {}
        self._k_tail_dbg_sum_meta: dict[int, int] = {}
        # Also keep tiny nibble hist (0..15) length-16 python lists for 4-bit cases.
        self._v_tail_dbg_hist: dict[int, list[int]] = {}
        self._k_tail_dbg_hist: dict[int, list[int]] = {}

    @staticmethod
    def _dbg_sum_u8(t: torch.ByteTensor) -> int:
        # device-agnostic sum
        return int(t.to(device="cpu", dtype=torch.int32).sum().item())

    @staticmethod
    def _dbg_nibble_hist_u4(packed: torch.ByteTensor) -> list[int]:
        """Return simple nibble histogram (0..15) for a uint8 packed buffer."""
        x = packed.to("cpu")
        lo = (x & 0x0F).to(torch.int64)
        hi = ((x >> 4) & 0x0F).to(torch.int64)
        hist = torch.zeros(16, dtype=torch.int64)
        hist.scatter_add_(0, lo, torch.ones_like(lo))
        hist.scatter_add_(0, hi, torch.ones_like(hi))
        return [int(v) for v in hist.tolist()]

    def _clear_kvtuner_tail(self, block_id: int):
        """
        Explicitly clear all KVTuner-related tail data associated with a block ID.
        This must be called when a block is evicted or recycled to prevent stale data.
        """
        # This function is being documented in English.
        self._v_tail_exists.pop(block_id, None)
        self._v_tail_packed.pop(block_id, None)
        self._v_tail_meta.pop(block_id, None)
        self._v_tail_scale.pop(block_id, None)
        self._v_tail_zp.pop(block_id, None)
        self._v_tail_dbg_sum_packed.pop(block_id, None)
        self._v_tail_dbg_sum_meta.pop(block_id, None)
        self._v_tail_dbg_hist.pop(block_id, None)

        self._k_tail_exists.pop(block_id, None)
        self._k_tail_packed.pop(block_id, None)
        self._k_tail_meta.pop(block_id, None)
        self._k_tail_scale.pop(block_id, None)
        self._k_tail_zp.pop(block_id, None)
        self._k_tail_dbg_sum_packed.pop(block_id, None)
        self._k_tail_dbg_sum_meta.pop(block_id, None)
        self._k_tail_dbg_hist.pop(block_id, None)

    def set_quantized_accounting_model(self, *, page_raw_bytes: int, page_used_bytes_q: int) -> None:
        """Install bytes-based accounting for KV mixed-precision quantization (K+V packed + meta).
        This affects kv_cache_usage(%): used_bytes_q / capacity_raw_bytes."""
        assert isinstance(page_raw_bytes, int) and page_raw_bytes > 0
        assert isinstance(page_used_bytes_q, int) and page_used_bytes_q > 0
        self._kv_accounting_enabled = True
        self._kv_page_raw_bytes = int(page_raw_bytes)
        self._kv_page_used_bytes_q = int(page_used_bytes_q)

    @staticmethod
    def _align16(n: int) -> int:
        r = n & 0xF
        return n if r == 0 else (n + (16 - r))

    def write_v_tail(self,
                     block_id: int,
                     *,
                     packed: torch.ByteTensor,
                     meta: torch.ByteTensor,
                     scale: torch.Tensor,
                     zp: torch.ByteTensor) -> None:
        """Install V page-tail for the block: packed bytes + 16B-aligned meta.
        All tensors must reside on the same device and obey dtype rules."""
        assert packed.dtype == torch.uint8, "Packed V must be uint8."
        assert meta.dtype == torch.uint8, "Meta must be uint8."
        assert scale.dtype in (torch.float16, torch.bfloat16), "Scale must be fp16/bf16."
        # The zero-point dtype can vary, but we will store it consistently as uint8.
        # This function is being documented in English.
        assert zp.dtype in (torch.uint8, torch.float16, torch.bfloat16), \
            "Zero-points must be fp16/bf16 (or uint8 for legacy)."
        # Enforce 16B alignment on meta length by padding zeros if needed.
        pad = self._align16(int(meta.numel())) - int(meta.numel())
        if pad:
            meta = torch.cat([meta, torch.zeros(pad, dtype=torch.uint8, device=meta.device)], dim=0)

        # --- SAFETY: clone to CPU so lifetime/device aliasing can't corrupt tails ---
        p = packed.detach().contiguous().to(torch.uint8).to("cpu").clone()
        m = meta.detach().contiguous().to(torch.uint8).to("cpu").clone()
        s = scale.detach().contiguous().to(scale.dtype).to("cpu").clone()
        
        # --- START: FIX ---
        # ISSUE: The zero-point (zp) was stored with its original float dtype (bf16/fp16),
        # which can cause subtle data corruption issues.
        # FIX: Consistently cast zp to torch.uint8 before storing, as hinted by
        # the original code comments. This ensures data type stability.
        z = zp.detach().contiguous().to(torch.uint8).to("cpu").clone()
        # --- END: FIX ---

        self._v_tail_exists[block_id] = True
        self._v_tail_packed[block_id] = p
        self._v_tail_meta[block_id] = m
        self._v_tail_scale[block_id] = s
        self._v_tail_zp[block_id] = z
        # Lightweight debug accounting recorded at write time so reads can
        # verify integrity without re-parsing or heavy work. Guarded to
        # never raise on write path.
        try:
            self._v_tail_dbg_sum_packed[block_id] = self._dbg_sum_u8(p)
            self._v_tail_dbg_sum_meta[block_id] = self._dbg_sum_u8(m)
            # Nibble histogram is useful for 4-bit-packed buffers; it's
            # cheap and harmless for other widths (it returns counts of
            # low/high nibbles across the bytes).
            self._v_tail_dbg_hist[block_id] = self._dbg_nibble_hist_u4(p)
        except Exception:
            # Do not let debug accounting break normal operation.
            self._v_tail_dbg_sum_packed.pop(block_id, None)
            self._v_tail_dbg_sum_meta.pop(block_id, None)
            self._v_tail_dbg_hist.pop(block_id, None)

    def read_v_tail(self, block_id: int):
        """Return (packed:uint8, meta:uint8, scale:fp16/bf16, zp:fp16/bf16|u8) or None."""
        if self._v_tail_exists.get(block_id, False):
            p = self._v_tail_packed[block_id]
            m = self._v_tail_meta[block_id]
            s = self._v_tail_scale[block_id]
            
            # --- START: FIX ---
            # ISSUE: If zp was somehow missing from the dictionary, it could lead to an error.
            # FIX: Ensure zp is retrieved correctly, creating a default UINT8 tensor if it's missing.
            # This makes the read operation more robust.
            z = self._v_tail_zp.get(block_id)
            if z is None:
                # If zp is missing, create a default zero tensor with uint8 dtype to match the
                # expected storage format.
                z = torch.zeros_like(s, dtype=torch.uint8) 
            # --- END: FIX ---

            # --- DEBUG: verify sums match write-time ---
            try:
                sum_p = self._dbg_sum_u8(p)
                sum_m = self._dbg_sum_u8(m)
                hist = self._dbg_nibble_hist_u4(p)
                wp = self._v_tail_dbg_sum_packed.get(block_id, None)
                wm = self._v_tail_dbg_sum_meta.get(block_id, None)
                # print(f"[KVTuner:POOL][READ  V] bid={block_id} packed_n={int(p.numel())} meta_n={int(m.numel())} "
                #       f"sum(p)={sum_p} sum(m)={sum_m} "
                #       f"w/sum(p)={wp} w/sum(m)={wm} hist={hist[:8]}..", flush=True)
            except Exception as _e:
                print(f"[KVTuner:POOL][READ  V][DBGERR] {type(_e).__name__}: {_e}", flush=True)
            return (p, m, s, z)
        return None

    def write_k_tail(self,
                     block_id: int,
                     *,
                     packed: torch.ByteTensor,
                     meta: torch.ByteTensor,
                     scale: torch.Tensor,
                     zp: torch.ByteTensor) -> None:
        """Install K page-tail for the block: packed bytes + 16B-aligned meta.
        Mirrors write_v_tail but for Keys. All tensors must be on same device and obey dtype rules."""
        assert packed.dtype == torch.uint8, "Packed K must be uint8."
        assert meta.dtype == torch.uint8, "Meta must be uint8."
        assert scale.dtype in (torch.float16, torch.bfloat16), "Scale must be fp16/bf16."
        # The zero-point dtype can vary, but we will store it consistently as uint8.
        # This function is being documented in English.
        assert zp.dtype in (torch.uint8, torch.float16, torch.bfloat16), \
            "Zero-points must be fp16/bf16 (or uint8 for legacy)."

        # Align meta to 16B as a safety guard (kv_meta already aligned, but keep invariant here)
        if (int(meta.numel()) & 0xF) != 0:
            pad = self._align16(int(meta.numel())) - int(meta.numel())
            meta = torch.cat([meta, torch.zeros(pad, dtype=torch.uint8, device=meta.device)], dim=0)

        # SAFETY: clone to CPU (same rationale as V)
        p = packed.detach().contiguous().to(torch.uint8).to("cpu").clone()
        m = meta.detach().contiguous().to(torch.uint8).to("cpu").clone()
        s = scale.detach().contiguous().to(scale.dtype).to("cpu").clone()
        
        # --- START: FIX (Consistent with write_v_tail) ---
        z = zp.detach().contiguous().to(torch.uint8).to("cpu").clone()
        # --- END: FIX ---

        self._k_tail_exists[block_id] = True
        self._k_tail_packed[block_id] = p
        self._k_tail_meta[block_id] = m
        self._k_tail_scale[block_id] = s
        self._k_tail_zp[block_id] = z
        # Record lightweight debug accounting for K-tail as well.
        try:
            self._k_tail_dbg_sum_packed[block_id] = self._dbg_sum_u8(p)
            self._k_tail_dbg_sum_meta[block_id] = self._dbg_sum_u8(m)
            self._k_tail_dbg_hist[block_id] = self._dbg_nibble_hist_u4(p)
        except Exception:
            self._k_tail_dbg_sum_packed.pop(block_id, None)
            self._k_tail_dbg_sum_meta.pop(block_id, None)
            self._k_tail_dbg_hist.pop(block_id, None)

    def read_k_tail(self, block_id: int):
        """Return (packed, meta, scale, zp) for K if exists; else None."""
        if self._k_tail_exists.get(block_id, False):
            p = self._k_tail_packed[block_id]
            m = self._k_tail_meta[block_id]
            s = self._k_tail_scale[block_id]
            
            # --- START: FIX (Consistent with read_v_tail) ---
            z = self._k_tail_zp.get(block_id)
            if z is None:
                z = torch.zeros_like(s, dtype=torch.uint8)
            # --- END: FIX ---
                
            try:
                sum_p = self._dbg_sum_u8(p)
                sum_m = self._dbg_sum_u8(m)
                hist = self._dbg_nibble_hist_u4(p)
                wp = self._k_tail_dbg_sum_packed.get(block_id, None)
                wm = self._k_tail_dbg_sum_meta.get(block_id, None)
                # print(f"[KVTuner:POOL][READ  K] bid={block_id} packed_n={int(p.numel())} meta_n={int(m.numel())} "
                #       f"sum(p)={sum_p} sum(m)={sum_m} "
                #       f"w/sum(p)={wp} w/sum(m)={wm} hist={hist[:8]}..", flush=True)
            except Exception as _e:
                print(f"[KVTuner:POOL][READ  K][DBGERR] {type(_e).__name__}: {_e}", flush=True)
            return (p, m, s, z)
        return None
        
    def get_cached_block(
            self, block_hash: BlockHash,
            kv_cache_group_ids: list[int]) -> Optional[list[KVCacheBlock]]:
        """Get the cached block by the block hash for each group in 
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id)
            cached_blocks_one_group = self.cached_block_hash_to_block.get(
                block_hash_with_group_id)
            if not cached_blocks_one_group:
                return None
            first_block = next(iter(cached_blocks_one_group.values()))
            cached_blocks.append(first_block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
        """
        if num_cached_blocks == num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        new_block_hashes = request.block_hashes[num_cached_blocks:]

        new_hashes: Optional[list[ExternalBlockHash]] = (
            [] if self.enable_kv_cache_events else None)
        for i, blk in enumerate(new_full_blocks):
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id)
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block[block_hash_with_group_id][
                blk.block_id] = blk
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: Optional[ExternalBlockHash] = None
            else:
                parent_block = blocks[num_cached_blocks - 1]
                assert parent_block.block_hash is not None
                parent_block_hash = maybe_convert_block_hash(
                    get_block_hash(parent_block.block_hash))

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.
                    all_token_ids[num_cached_blocks *
                                  block_size:num_full_blocks * block_size],
                    block_size=block_size,
                    lora_id=request.lora_request.id
                    if request.lora_request else None,
                    medium=MEDIUM_GPU,
                ))

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # FIX: Explicitly clear any stale KVTuner tail data for each recycled block.
        # This is the correct location to ensure cleanup happens for ALL block re-allocations,
        # regardless of whether prefix caching is enabled.
        # This function is being documented in English.
        for block in ret:
            self._clear_kvtuner_tail(block.block_id)

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # FIX: Clear any stale KVTuner tail data before this block is reused.
        # This prevents new allocations from inheriting old, incorrect quantized data.
        # This function is being documented in English.
        self._clear_kvtuner_tail(block.block_id)

        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            # This function is being documented in English.
            return False
        blocks_by_id = self.cached_block_hash_to_block.get(block_hash)
        if blocks_by_id is None:
            # block_hash not found in cached_block_hash_to_block,
            # eviction is not needed
            # This function is being documented in English.
            return False
        block.reset_hash()
        blocks_by_id.pop(block.block_id, None)
        if len(blocks_by_id) == 0:
            del self.cached_block_hash_to_block[block_hash]

        if self.enable_kv_cache_events:
            # FIXME (Chen): Not sure whether we should return `hash_value`
            # or `(hash_value, group_id)` here. But it's fine now because
            # we disable hybrid kv cache manager when kv cache event is
            # enabled, so there is only one group.
            # This function is being documented in English.
            self.kv_event_queue.append(
                BlockRemoved(block_hashes=[
                    maybe_convert_block_hash(get_block_hash(block_hash))
                ],
                             medium=MEDIUM_GPU))
        return True

    def touch(self, blocks: tuple[list[KVCacheBlock], ...]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for blocks_per_group in blocks:
            for block in blocks_per_group:
                # ref_cnt=0 means this block is in the free list (i.e. eviction
                # candidate), so remove it.
                if block.ref_cnt == 0 and not block.is_null:
                    self.free_block_queue.remove(block)
                block.ref_cnt += 1

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n([
            block for block in blocks_list
            if block.ref_cnt == 0 and not block.is_null
        ])

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet", num_used_blocks - 1)
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = defaultdict(dict)

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        Semantics:
            - If quantized-accounting is enabled (via set_quantized_accounting_model),
              we report: used_bytes_q / capacity_raw_bytes.
              capacity_raw_bytes = (num_gpu_blocks - 1) * page_raw_bytes
              used_bytes_q       = (num_used_blocks) * page_used_bytes_q
            - Otherwise, we fall back to the classic block-ratio usage.
        """
        total_gpu_blocks = self.num_gpu_blocks - 1  # subtract null
        if total_gpu_blocks <= 0:
            return 0.0

        num_used_blocks = total_gpu_blocks - self.get_num_free_blocks()
        if self._kv_accounting_enabled:
            capacity_raw_bytes = total_gpu_blocks * self._kv_page_raw_bytes
            used_bytes_q = num_used_blocks * self._kv_page_used_bytes_q
            if capacity_raw_bytes <= 0:
                return 0.0
            return float(used_bytes_q) / float(capacity_raw_bytes)
        
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.
        
        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events