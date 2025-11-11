# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Optional

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager, FullAttentionManager, get_manager_for_kv_cache_spec)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.request import Request

from vllm.v1.core.kvtuner_quantizer import KVTunerQuantizer
from vllm.utils import get_dtype_size

_GLOBAL_COORD: Optional["KVCacheCoordinator"] = None

class KVCacheCoordinator(ABC):
    """
    Coordinate the KV cache of different KV cache groups.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        quantizer: Optional[KVTunerQuantizer] = None
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching

        self.block_pool = BlockPool(kv_cache_config.num_blocks, enable_caching,
                                    enable_kv_cache_events)

        # --- KVTuner KV bytes-based accounting (exact enough, no globals) ---
        # usage% = used_bytes_q / capacity_raw_bytes
        # capacity_raw_bytes = (num_blocks-1)*page_raw_bytes(raw K+V)
        # used_bytes_q       = num_used_blocks*(K packed + V packed + meta(K)+meta(V))
        self.kv_quantizer = quantizer
        if quantizer is not None and getattr(quantizer, "enable") and len(kv_cache_config.kv_cache_groups) >= 1:
            try:
                spec0 = kv_cache_config.kv_cache_groups[0].kv_cache_spec
                if isinstance(spec0, FullAttentionSpec) and not spec0.use_mla:

                    dtype_size = get_dtype_size(spec0.dtype)
                    elems = spec0.block_size * spec0.num_kv_heads * spec0.head_size
                    page_raw_bytes = int(spec0.page_size_bytes)  # raw K + raw V (denominator)

                    # Bits per layer/group (use layer 0's group as representative)
                    vbits = int(quantizer.get_bits(0, "v"))
                    kbits = int(quantizer.get_bits(0, "k"))
                    # Packed payload sizes
                    v_packed = (elems * vbits + 7) // 8
                    k_packed = (elems * kbits + 7) // 8
                    # Meta bytes approximation (16B-aligned)
                    #  V (axis=0): per row (token*head) have 1 group → (2B scale + 1B zp)
                    meta_v_core = 16 + (spec0.block_size * spec0.num_kv_heads) * (2 + 1)
                    meta_v_bytes = ((meta_v_core + 15) // 16) * 16
                    #  K (axis=1, q_group_size=32): groups along head_size
                    qgs = int(getattr(quantizer, "key_q_group_size", 32))
                    g_per_row = (spec0.head_size + qgs - 1) // qgs
                    meta_k_core = 16 + (spec0.block_size * spec0.num_kv_heads * g_per_row) * (2 + 1)
                    meta_k_bytes = ((meta_k_core + 15) // 16) * 16
                    page_used_bytes_q = int(k_packed + v_packed + meta_k_bytes + meta_v_bytes)

                    # KVTUNER_DEBUG_START
                    import os, sys
                    try:
                        print(f"[KVTuner:acct] pid={os.getpid()} "
                              f"k_packed={k_packed} v_packed={v_packed} "
                              f"meta_k={meta_k_bytes} meta_v={meta_v_bytes} "
                              f"denom_raw={page_raw_bytes} used_q={page_used_bytes_q}",
                              flush=True, file=sys.stderr)
                    except Exception:
                        pass
                    # KVTUNER_DEBUG_END

                    if page_raw_bytes > 0 and page_used_bytes_q > 0:

                        self.block_pool.set_quantized_accounting_model(
                            page_raw_bytes=page_raw_bytes,
                            page_used_bytes_q=page_used_bytes_q)
                        # Always-on debug
                        print("[KVTuner][Coordinator] accounting installed",
                              f"quantizer_id={id(quantizer)} dtype_size={dtype_size}",
                              f"block_size={spec0.block_size} n_kv_heads={spec0.num_kv_heads} head_size={spec0.head_size}",
                              f"elems={elems} kbits={kbits} vbits={vbits} "
                              f"k_packed={k_packed} v_packed={v_packed} "
                              f"meta_k={meta_k_bytes} meta_v={meta_v_bytes}",
                              f"page_raw_bytes={page_raw_bytes} page_used_bytes_q={page_used_bytes_q}",
                              flush=True)
            except Exception as e:
                print("[KVTuner][Coordinator] Warning: could not set quantized ")
                pass

        # Needs special handling for find_longest_cache_hit if eagle is enabled
        self.use_eagle = use_eagle
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
            ) for i, kv_cache_group in enumerate(
                self.kv_cache_config.kv_cache_groups))

    def get_num_blocks_to_allocate(self, request_id: str, num_tokens: int,
                                   new_computed_blocks: tuple[
                                       list[KVCacheBlock], ...],
                                   num_encoder_tokens: int) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.

        Returns:
            The number of blocks.
        """
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                # For cross-attention, we issue a single static allocation
                # of blocks based on the number of encoder input tokens.
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id, num_encoder_tokens, [])
            else:
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id, num_tokens, new_computed_blocks[i])
        return num_blocks_to_allocate

    def save_new_computed_blocks(
            self, request_id: str,
            new_computed_blocks: tuple[list[KVCacheBlock], ...]) -> None:
        """
        Add the new computed blocks to the request.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
        """
        for i, manager in enumerate(self.single_type_managers):
            manager.save_new_computed_blocks(request_id,
                                             new_computed_blocks[i])

    def allocate_new_blocks(
            self,
            request_id: str,
            num_tokens: int,
            num_encoder_tokens: int = 0) -> tuple[list[KVCacheBlock], ...]:
        """
        Allocate new blocks for the request to give it at least `num_tokens` 
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 
                tokens that are already allocated).
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.

        Returns:
            The new allocated blocks.
        """
        return tuple(
            manager.allocate_new_blocks(
                request_id, num_encoder_tokens if isinstance(
                    manager, CrossAttentionManager) else num_tokens)
            for manager in self.single_type_managers)

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_computed_tokens: The total number of tokens
                that need to be cached
                (including tokens that are already cached).
        """
        for manager in self.single_type_managers:
            manager.cache_blocks(request, num_computed_tokens)

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> list[int]:
        """
        Get the number of common prefix blocks for all requests in the RUNNING
        state for each kv cache group.

        Args:
            request_id: The request ID.
            num_running_requests: The total number of requests in the RUNNING
                state.

        Returns:
            list[int]: The number of common prefix blocks for all requests in
                the RUNNING state for each kv cache group.
        """
        num_blocks_per_group = [
            manager.get_num_common_prefix_blocks(request_id,
                                                 num_running_requests)
            for manager in self.single_type_managers
        ]
        return num_blocks_per_group

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and replace 
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            num_computed_tokens: The number of tokens that have been computed.
        """
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(request_id, num_computed_tokens)

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the blocks for the request.
        """
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers)

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        pass


class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    """
    KV cache coordinator to use if prefix caching is disabled or unsupported.
    In contrast to UnitaryKVCacheCoordinator and HybridKVCacheCoordinator,
    supports arbitrary numbers of KV cache groups (including 0 groups).
    Does not implement any features related to prefix caching.
    """

    def __init__(self, kv_cache_config: KVCacheConfig, max_model_len: int,
                 use_eagle: bool, enable_kv_cache_events: bool,
                 dcp_world_size: int,
                 quantizer: Optional[KVTunerQuantizer] = None):
        super().__init__(kv_cache_config,
                         max_model_len,
                         use_eagle,
                         False,
                         enable_kv_cache_events,
                         dcp_world_size=dcp_world_size,
                         quantizer=quantizer)
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager))
        return blocks, 0


class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for models with only one KV cache group. This is the
    case for models with only one KV cache type, e.g., all attention layers use
    full attention or all attention layers use sliding window attention.
    """

    def __init__(self, kv_cache_config: KVCacheConfig, max_model_len: int,
                 use_eagle: bool, enable_caching: bool,
                 enable_kv_cache_events: bool, dcp_world_size: int,
                 quantizer: Optional[KVTunerQuantizer] = None):
        super().__init__(kv_cache_config,
                         max_model_len,
                         use_eagle,
                         enable_caching,
                         enable_kv_cache_events,
                         dcp_world_size=dcp_world_size,
                         quantizer=quantizer)
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[
            0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group")

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
            dcp_world_size=self.dcp_world_size,
        )
        return hit_blocks, len(hit_blocks[0]) * self.block_size


class HybridKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    To simplify `find_longest_cache_hit`, it only supports the combination of 
    two types of KV cache groups, and one of them must be full attention.
    May extend to more general cases in the future.
    """

    def __init__(self, kv_cache_config: KVCacheConfig, max_model_len: int,
                 use_eagle: bool, enable_caching: bool,
                 enable_kv_cache_events: bool, dcp_world_size: int,
                 quantizer: Optional[KVTunerQuantizer] = None):
        super().__init__(kv_cache_config,
                         max_model_len,
                         use_eagle,
                         enable_caching,
                         enable_kv_cache_events,
                         dcp_world_size=dcp_world_size,
                         quantizer=quantizer)
        assert dcp_world_size == 1, "DCP not support hybrid attn now."
        self.verify_and_split_kv_cache_groups()

    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Verifies that the model has exactly two types of KV cache groups, and 
        one of them is full attention. Then, split the kv cache groups into full
        attention groups and other groups.
        """
        full_attention_spec: Optional[FullAttentionSpec] = None
        other_spec: Optional[KVCacheSpec] = None
        self.full_attention_group_ids: list[int] = []
        self.other_group_ids: list[int] = []
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                if full_attention_spec is None:
                    full_attention_spec = g.kv_cache_spec
                else:
                    assert full_attention_spec == g.kv_cache_spec, (
                        "HybridKVCacheCoordinator assumes exactly one type of "
                        "full attention groups now.")
                self.full_attention_group_ids.append(i)
            else:
                if other_spec is None:
                    other_spec = g.kv_cache_spec
                else:
                    assert other_spec == g.kv_cache_spec, (
                        "HybridKVCacheCoordinator assumes "
                        "exactly one other type of groups now.")
                self.other_group_ids.append(i)

        assert full_attention_spec is not None, (
            "HybridKVCacheCoordinator assumes exactly one type of full "
            "attention groups now.")
        assert other_spec is not None, (
            "HybridKVCacheCoordinator assumes exactly one type of other "
            "groups now.")

        self.full_attention_manager_cls = FullAttentionManager
        self.other_attention_cls = self.single_type_managers[
            self.other_group_ids[0]].__class__
        self.full_attention_spec = full_attention_spec
        self.other_spec = other_spec
        self.full_attention_block_size = self.full_attention_spec.block_size
        self.other_block_size = self.other_spec.block_size

        if self.enable_caching:
            # this requirement is only needed for the prefix caching logic
            divisible = self.other_block_size % self.full_attention_block_size
            assert divisible == 0, (
                "KVCacheCoordinator assumes the block_size of full "
                "attention layers is divisible by other layers now.")

        if max(self.full_attention_group_ids) < min(self.other_group_ids):
            self.full_attn_first = True
        elif max(self.other_group_ids) < min(self.full_attention_group_ids):
            self.full_attn_first = False
        else:
            raise ValueError(
                "HybridKVCacheCoordinator assumes the full "
                "attention group ids and other attention group ids "
                "do not interleave, either full attention group ids "
                "are before other attention group ids or vice versa."
                "This is for simplifying merging hit_blocks_full_attn and "
                "hit_blocks_other_attn to hit_blocks.")

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Find the longest cache hit for the request.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A list of the cache hit blocks for each single type manager.
                - The number of tokens of the longest cache hit.
        """
        # First, find the longest cache hit for full attention.
        hit_blocks_full_attn = (
            self.full_attention_manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=max_cache_hit_length,
                kv_cache_group_ids=self.full_attention_group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=self.full_attention_spec,
                use_eagle=self.use_eagle,
            ))
        hit_length = len(
            hit_blocks_full_attn[0]) * self.full_attention_block_size

        # Next, find the cache hit for the other attention WITHIN
        # the cache hit of full attention.
        hit_blocks_other_attn = (
            self.other_attention_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=hit_length,
                kv_cache_group_ids=self.other_group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=self.other_spec,
                use_eagle=self.use_eagle,
            ))
        hit_length = len(hit_blocks_other_attn[0]) * self.other_block_size

        # NOTE: the prefix cache hit length must be a multiple of block_size as
        # we don't support partial block cache hit yet. The cache hit length
        # of other attention is ensured to be a multiple of the block size of
        # full attention layers in current implementation, because hit_length is
        # a multiple of other attention's block size, and other attention's
        # block size is a multiple of full attention's block size (verified in
        # `verify_and_split_kv_cache_groups`).
        assert hit_length % self.full_attention_block_size == 0

        # Truncate the full attention cache hit to the length of the
        # cache hit of the other attention.
        for group_hit_blocks in hit_blocks_full_attn:
            del group_hit_blocks[hit_length // self.full_attention_block_size:]

        # Merge the hit blocks of full attention and other attention.
        if self.full_attn_first:
            hit_blocks = hit_blocks_full_attn + hit_blocks_other_attn
        else:
            hit_blocks = hit_blocks_other_attn + hit_blocks_full_attn
        return hit_blocks, hit_length


def get_kv_cache_coordinator(kv_cache_config: KVCacheConfig,
                             max_model_len: int, use_eagle: bool,
                             enable_caching: bool,
                             enable_kv_cache_events: bool,
                             dcp_world_size: int,
                             quantizer=None) -> KVCacheCoordinator:
    """Factory that returns a process-local singleton coordinator.

    Rules:
      - caching OFF or no groups  -> NoPrefix
      - all groups FullAttention  -> Unitary (even if >1 groups)
      - mixed (full + other)      -> Hybrid
    """
    from vllm.v1.core.kvtuner_quantizer import get_global_quantizer
    try:
        from vllm.v1.kv_cache_interface import FullAttentionSpec
    except Exception:
        FullAttentionSpec = None  # fallback; shouldn't happen

    global _GLOBAL_COORD
    if _GLOBAL_COORD is not None:
        return _GLOBAL_COORD

    # Resolve quantizer
    if quantizer is None:
        quantizer = get_global_quantizer()

    groups = getattr(kv_cache_config, "kv_cache_groups", []) or []
    n_groups = len(groups)

    # 1) No caching at all OR no groups -> NoPrefix
    if not enable_caching or n_groups == 0:
        _GLOBAL_COORD = KVCacheCoordinatorNoPrefixCache(
            kv_cache_config, max_model_len, use_eagle,
            enable_kv_cache_events, dcp_world_size=dcp_world_size,
            quantizer=quantizer)
        return _GLOBAL_COORD

    # 2) All groups are FullAttention -> Unitary (even if >1)
    all_full = (FullAttentionSpec is not None) and all(
        isinstance(g.kv_cache_spec, FullAttentionSpec) for g in groups
    )
    if n_groups == 1 or all_full:
        _GLOBAL_COORD = UnitaryKVCacheCoordinator(
            kv_cache_config, max_model_len, use_eagle,
            enable_caching, enable_kv_cache_events,
            dcp_world_size=dcp_world_size, quantizer=quantizer)
        return _GLOBAL_COORD

    # 3) Mixed types -> Hybrid
    _GLOBAL_COORD = HybridKVCacheCoordinator(
        kv_cache_config, max_model_len, use_eagle,
        enable_caching, enable_kv_cache_events,
        dcp_world_size=dcp_world_size, quantizer=quantizer)
    return _GLOBAL_COORD

def get_global_coordinator() -> KVCacheCoordinator:
    return _GLOBAL_COORD