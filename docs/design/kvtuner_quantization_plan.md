# Proposal: Integrating KVTunner Layer-wise KV Cache Quantization in vLLM 0.10.2

## Objectives
* Add a new quantization mode that enables KVTunner-style per-layer key/value cache compression while keeping compatibility with the existing quantization registry and configuration loading flow. The new mode must allow independently selecting 8-bit, 4-bit, or 2-bit storage for the K and V projections of every transformer block.
* Ensure the runtime can apply heterogeneous KV cache bit-widths without breaking existing attention backends, cache management, or the v1 engine scheduler.
* Provide a foundation for checkpoint metadata (e.g., JSON files) that records the layer-specific bit allocation so the loader and inference engine can consume it automatically.

## Quantization configuration plumbing
* Extend the quantization registry with a `kvtuner` entry that maps to a dedicated config class responsible for loading layer-wise K/V bit schedules and scales. This mirrors how other quantization backends are registered today.【F:vllm/model_executor/layers/quantization/__init__.py†L9-L149】
* Implement `KVTunerConfig` (new file) deriving from `QuantizationConfig` so it can parse a KVTunner-specific JSON manifest, expose per-layer K/V bit widths, and hand back the correct `QuantizeMethodBase` implementation. The class should follow the pattern used by `Fp8Config`, including hook points for mapping HuggingFace module names to the internal layout.【F:vllm/model_executor/layers/quantization/base_config.py†L18-L124】【F:vllm/model_executor/layers/quantization/fp8.py†L45-L134】
* Update `get_quant_config` to look for the new manifest filename and feed it into `KVTunerConfig.from_config`, similar to the existing discovery path for other quantization methods.【F:vllm/model_executor/model_loader/weight_utils.py†L184-L265】
* Extend `CacheConfig` so users can select the new behavior via CLI/constructor flags. This likely requires broadening `CacheDType` to accept symbolic values that mean "delegate to KVTunner schedule" while preserving existing fp8-related semantics.【F:vllm/config/cache.py†L24-L149】

## Layer instrumentation and metadata propagation
* Introduce a `KVTunerKVCacheMethod` (subclass of `BaseKVCacheMethod`) that records the per-layer bit allocation and scale tensors on each `Attention` module. It should attach host/device metadata (e.g., `_k_bits`, `_v_bits`, optional per-group overrides) alongside the existing `_k_scale` and `_v_scale` buffers used during runtime.【F:vllm/model_executor/layers/quantization/kv_cache.py†L14-L143】
* Expand the `Attention` constructor to recognize the new quantization config, surface layer-specific bit widths, and make them available to attention backends. This includes storing attributes for K/V bit depth and updating any logic that assumes a single global `kv_cache_dtype` string.【F:vllm/attention/layer.py†L79-L195】
* Ensure downstream attention backends (FlashAttention, xFormers, Triton MLA, etc.) can query the per-layer bit spec so they can pick the right kernels or conversion paths. Their guardrails currently assume fp8-only quantization and will need to be generalized.【F:vllm/attention/backends/flash_attn.py†L560-L760】

## Runtime cache operations and kernels
* Generalize the Python cache helpers to accept K/V bit width metadata when writing or reading cache pages. `PagedAttention.write_to_paged_cache` and the decode paths should forward the schedule to the custom ops in addition to the existing scale tensors.【F:vllm/attention/ops/paged_attn.py†L73-L198】
* Update the custom operator wrappers in `_custom_ops.py` so that the CUDA/ROCm kernels receive the new arguments (e.g., integer bit widths, optional per-head layout selectors) instead of relying solely on a string-based `kv_cache_dtype`. This may require new overloads for both paged attention variants.【F:vllm/_custom_ops.py†L38-L127】
* Extend the C++ cache API (`csrc/cache.h`) and its CUDA implementations to support mixed-bit quantization. The current kernels are hard-wired to `Fp8KVCacheDataType`; we need a more general enum/templating strategy that can encode 8/4/2-bit packing for keys and values independently, along with helper routines for packing/unpacking during `reshape_and_cache*`, `concat_and_cache_mla`, and `gather_and_maybe_dequant_cache`.【F:csrc/cache.h†L21-L50】【F:csrc/cache_kernels.cu†L210-L415】
* Adjust paged attention kernels (both CUDA and ROCm) to understand the new enum and apply the correct dequantization per layer when computing attention scores. This involves plumbing the bit widths through to the math loops similar to how `Fp8KVCacheDataType` gates the existing fp8 path.【F:csrc/cache_kernels.cu†L212-L415】【F:csrc/attention/dtype_fp8.cuh†L13-L32】

## Engine and scheduler integration
* Propagate the richer KV cache metadata through the v1 engine: the GPU runner currently collapses `cache_config.cache_dtype` into a single torch dtype and allocates homogeneous tensors. It will need to recognize the KVTunner schedule, allocate appropriately packed buffers, and retain per-layer descriptors so attention groups apply the right quantization on read/write.【F:vllm/v1/worker/gpu_model_runner.py†L185-L259】
* Update the engine’s serialization/reporting utilities so usage stats and telemetry reflect the new mixed-bit mode rather than a misleading single dtype string.【F:vllm/v1/utils.py†L335-L365】
* Audit v1 attention backends for assumptions that `kv_cache_dtype.startswith("fp8")` implies quantization; rework those checks to branch on the new enum instead, and ensure the per-layer bit schedule is forwarded alongside scale tensors when invoking backend-specific kernels.【F:vllm/v1/attention/backends/flash_attn.py†L190-L603】【F:vllm/v1/attention/backends/triton_attn.py†L218-L389】

## Testing strategy
* Add targeted unit tests that exercise loading a synthetic KVTunner manifest, verifying that `Attention` layers expose the correct per-layer bit widths and that cache tensors are sized/packed as expected.
* Introduce integration tests that run short decode/prefill traces under each attention backend while mixing 2/4/8-bit assignments across layers, asserting numerical stability (e.g., compare against fp16 outputs within a tolerance) and that cache readback yields the expected dtype/packing.
* Expand existing quantization CI coverage to include a configuration that enables KVTunner on a small model, ensuring both the legacy engine and v1 engine can execute end-to-end with layer-wise K/V quantization enabled.
