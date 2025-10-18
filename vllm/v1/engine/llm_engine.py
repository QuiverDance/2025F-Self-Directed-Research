# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from copy import copy
from typing import Any, Callable, Optional, Union

from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tracing import init_tracer
from vllm.transformers_utils.tokenizer_group import (
    TokenizerGroup, init_tokenizer_from_configs)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Device
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import (PrometheusStatLogger, StatLoggerBase,
                                     StatLoggerFactory)
from vllm.v1.metrics.reader import Metric, get_metrics_snapshot
from vllm.v1.metrics.stats import IterationStats

from vllm.v1.engine.config import KVQuantConfig
from vllm.v1.metrics.kv_quant import KVQuantMetricsLogger, KVQuantLayerRecord
from vllm.v1.attention.kv_cache_quant import PagedKVCacheQuantized

import time as _time, uuid as _uuid, os as _os

# --- KV instrumentation (request-scoped, unified) ---
from vllm.instrumentation.kv_metrics import (
    KVMetricsCollector,
    KVMetricsConfig,
)

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        multiprocess_mode: bool = False,
    ) -> None:
        if not envs.VLLM_USE_V1:
            raise ValueError(
                "Using V1 LLMEngine, but envs.VLLM_USE_V1=False. "
                "This should not happen. As a workaround, try using "
                "LLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        if stat_loggers is not None:
            raise NotImplementedError(
                "Passing StatLoggers to LLMEngine in V1 is not yet supported. "
                "Set VLLM_USE_V1=0 and file and issue on Github.")

        self.vllm_config = vllm_config
        self.observability_config = vllm_config.observability_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        self.log_stats = log_stats
        self.stat_logger: Optional[StatLoggerBase] = None
        if self.log_stats:
            self.stat_logger = PrometheusStatLogger(vllm_config)

        try:
            KVMetricsCollector.reset()
            self._run_id = _time.strftime("%Y%m%d-%H%M%S") + "-" + _uuid.uuid4().hex[:6]
        except Exception:
            self._run_id = None

        # important: init dp group before init the engine_core
        # In the decoupled engine case this is handled in EngineCoreProc.
        parallel_config = vllm_config.parallel_config
        if not multiprocess_mode and parallel_config.data_parallel_size > 1:
            self.dp_group = parallel_config.stateless_init_dp_group()
        else:
            self.dp_group = None
        self.should_execute_dummy_batch = False

        if self.model_config.skip_tokenizer_init:
            self.tokenizer = None
        else:
            # Tokenizer (+ ensure liveness if running in another process).
            self.tokenizer = init_tokenizer_from_configs(
                model_config=vllm_config.model_config,
                scheduler_config=vllm_config.scheduler_config,
                lora_config=vllm_config.lora_config)

        # Processor (convert Inputs --> EngineCoreRequests)
        self.processor = Processor(vllm_config=vllm_config,
                                   tokenizer=self.tokenizer,
                                   mm_registry=mm_registry)

        # OutputProcessor (convert EngineCoreOutputs --> RequestOutput).
        self.output_processor = OutputProcessor(self.tokenizer,
                                                log_stats=self.log_stats)
        if self.observability_config.otlp_traces_endpoint is not None:
            tracer = init_tracer(
                "vllm.llm_engine",
                self.observability_config.otlp_traces_endpoint)
            self.output_processor.tracer = tracer

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )

        if not multiprocess_mode:
            # for v0 compatibility
            self.model_executor = self.engine_core.engine_core.model_executor  # type: ignore

        # --- [KV] Wire up request-scoped metrics (env-driven enable) --------
        # === attach KV metrics collector to the actual engine core ===
        try:
            col = KVMetricsCollector.get()   # env-based config
            eng_like = self.engine_core
            # In single-process, EngineCoreClient exposes the real EngineCore here:
            if hasattr(eng_like, "engine_core"):
                eng_like = eng_like.engine_core
            col.link_engine(eng_like or self)
            self._kv_metrics = col
            print("[KVDBG] linked metrics to", type(eng_like).__name__, flush=True)
        except Exception as e:
            print("[KVDBG] link_engine failed:", repr(e), flush=True)

        # Best-effort static meta recorded with each request log (repro hints).
        self._kv_meta: dict[str, Any] = {
            "model": getattr(self.model_config, "model", None)
                     or getattr(self.model_config, "model_name", None),
            "dtype": None,
            "num_devices": None,
            "block_size": None,
            "run_id": getattr(self, "_run_id", None),
        }
        # dtype
        try:
            dtype_from_model = getattr(self.model_config, "dtype", None) \
                               or getattr(self.vllm_config.model_config, "dtype", None)
            if dtype_from_model:
                self._kv_meta["dtype"] = str(dtype_from_model).lower()
        except Exception:
            pass
        if not self._kv_meta["dtype"]:
            try:
                self._kv_meta["dtype"] = str(
                    getattr(self, "dtype", None) or getattr(self, "layer_dtype", None) or ""
                ).lower() or None
            except Exception:
                pass

        # num_devices
        try:
            import torch
            self._kv_meta["num_devices"] = torch.cuda.device_count()
        except Exception:
            pass

        # block_size
        try:
            bs = getattr(self.cache_config, "block_size", None) \
                 or getattr(self.vllm_config.cache_config, "block_size", None)
            if bs is not None:
                self._kv_meta["block_size"] = int(bs)
        except Exception:
            pass
        if self._kv_meta["block_size"] is None:
            try:
                for name in ("block_size", "kv_block_size", "cache_block_size"):
                    val = getattr(self, name, None)
                    if val is not None:
                        self._kv_meta["block_size"] = int(val)
                        break
                if self._kv_meta["block_size"] is None:
                    mgr = getattr(self, "block_manager", None) or getattr(self, "kv_cache_manager", None)
                    if mgr is not None:
                        for name in ("block_size", "kv_block_size"):
                            val = getattr(mgr, name, None)
                            if val is not None:
                                self._kv_meta["block_size"] = int(val)
                                break
            except Exception:
                pass

        # rank
        try:
            import os as _os
            r = _os.getenv("RANK") or _os.getenv("LOCAL_RANK")
            if r is not None:
                self._kv_meta["rank"] = int(r)
        except Exception:
            pass

        # Per-request streaming state for first-token detection & counting.
        self._kv_seen_first: set[str] = set()
        self._kv_last_len: dict[str, int] = {}


        # Don't keep the dummy data in memory
        self.reset_mm_cache()

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        return cls(vllm_config=vllm_config,
                   executor_class=Executor.get_class(vllm_config),
                   log_stats=(not disable_log_stats),
                   usage_context=usage_context,
                   stat_loggers=stat_loggers,
                   multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        enable_multiprocessing: bool = False,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.debug("Enabling multiprocessing for LLMEngine.")
            enable_multiprocessing = True

        # build engine instance first
        engine = cls(vllm_config=vllm_config,
                 executor_class=executor_class,
                 log_stats=not engine_args.disable_log_stats,
                 usage_context=usage_context,
                 stat_loggers=stat_loggers,
                 multiprocess_mode=enable_multiprocessing)

        return engine

    def get_num_unfinished_requests(self) -> int:
        return self.output_processor.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        has_unfinished = self.output_processor.has_unfinished_requests()
        if self.dp_group is None:
            return has_unfinished or self.engine_core.dp_engines_running()
        return self.has_unfinished_requests_dp(has_unfinished)

    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:
        aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, has_unfinished)
        if not has_unfinished and aggregated_has_unfinished:
            self.should_execute_dummy_batch = True
        return aggregated_has_unfinished

    @classmethod
    def validate_outputs(cls, outputs, output_type):
        return outputs

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.engine_core.get_supported_tasks()

    def abort_request(self, request_ids: list[str]) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        request_ids = self.output_processor.abort_requests(request_ids)
        self.engine_core.abort_requests(request_ids)

        # --- [KV] Mark 'decode' snapshot & flush for aborted requests -------
        for rid in request_ids:
            try:
                self._kv_on_request_end(rid)
                self._kv_last_len.pop(rid, None)
                self._kv_seen_first.discard(rid)
            except Exception:
                pass

    def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> None:
        # Validate the request_id type.
        if not isinstance(request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request_id)}")

        # Process raw inputs into the request.
        prompt_str, request = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            tokenization_kwargs, trace_headers, priority)

        # --- [KV] Enqueue hook -----------------------
        try:
            context_len = self._kv_compute_context_len(prompt_str, request)
            self._kv_on_enqueue(request_id, context_len)
        except Exception:
            pass

        n = params.n if isinstance(params, SamplingParams) else 1

        if n == 1:
            # Make a new RequestState and queue.
            self.output_processor.add_request(request, prompt_str, None, 0)

            # Add the request to EngineCore.
            self.engine_core.add_request(request)
            return

        # Fan out child requests (for n>1).
        parent_req = ParentRequest(request_id, params)
        for idx in range(n):
            request_id, params = parent_req.get_child_info(idx)
            child_request = request if idx == n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = params

            # Make a new RequestState and queue.
            self.output_processor.add_request(child_request, prompt_str,
                                              parent_req, idx)

            # Add the request to EngineCore.
            self.engine_core.add_request(child_request)

    def step(self) -> Union[list[RequestOutput], list[PoolingRequestOutput]]:
        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            self.engine_core.execute_dummy_batch()
            return []

        # 1) Get EngineCoreOutput from the EngineCore.
        outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs.
        iteration_stats = IterationStats() if self.log_stats else None
        processed_outputs = self.output_processor.process_outputs(
            outputs.outputs,
            engine_core_timestamp=outputs.timestamp,
            iteration_stats=iteration_stats)

        # --- [KV] Observe streaming to drive unified metrics ----------------
        try:
            self._kv_observe_processed_outputs(processed_outputs)
        except Exception as e:
            logger.warning(f"KV metrics observe error: {e}")
            pass

        # 3) Abort any reqs that finished due to stop strings.
        self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # 4) Record stats
        if self.stat_logger is not None:
            assert outputs.scheduler_stats is not None
            self.stat_logger.record(scheduler_stats=outputs.scheduler_stats,
                                    iteration_stats=iteration_stats)

        return processed_outputs.request_outputs

    def get_vllm_config(self):
        return self.vllm_config

    def get_model_config(self):
        return self.model_config

    def start_profile(self):
        self.engine_core.profile(True)

    def stop_profile(self):
        self.engine_core.profile(False)

    def reset_mm_cache(self):
        self.processor.clear_cache()
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(self, device: Optional[Device] = None):
        self.engine_core.reset_prefix_cache()

    def reset_for_next_run(
        self,
        reset_metrics: bool = True,
        relink_metrics: bool = True,
        new_run_id: bool = True,
        empty_cuda_cache: bool = False,
    ) -> None:
        """Soft reset between runs without destroying the engine.
        1) Abort ALL active requests
        2) Drain scheduler steps until requests & KV frees settle
        3) Clear MM/prefix caches (return KV blocks to free pool)
        4) Deep-free KV pool if any block remains
        5) (Optional) Reset metrics, stamp new run_id, relink collector
        6) (Optional) Empty CUDA cache (engine stays alive)
        """
        self._begin_new_run()

        import time as _t, gc as _gc

        try:
            ids = []
            core = getattr(self, "engine_core", self)
            if hasattr(core, "engine_core"):
                core = core.engine_core
            sched = getattr(core, "scheduler", None) or getattr(
                getattr(core, "model_executor", None), "scheduler", None)

            containers = ("running","_running","waiting","_waiting",
                        "paused","_paused","swapped","_swapped",
                        "req_queue","request_queue")
            seen = set()
            for attr in containers:
                q = getattr(sched, attr, None)
                if q is None:
                    continue
                try:
                    it = list(q.values()) if hasattr(q, "values") else list(q)
                except Exception:
                    it = []
                for grp in it:
                    rid = getattr(grp, "request_id", None)
                    if rid: seen.add(rid)
                    rids = getattr(grp, "request_ids", None)
                    if rids:
                        try: seen.update(list(rids))
                        except Exception: pass
                    for field in ("seqs","sequences","_seqs"):
                        seqs = getattr(grp, field, None)
                        if not seqs: continue
                        try:
                            for s in list(seqs):
                                rr = getattr(s, "request_id", None)
                                if rr: seen.add(rr)
                        except Exception:
                            pass
            ids = list(seen)
            if ids:
                try:
                    self.abort_request(ids)
                except Exception:
                    self.engine_core.abort_requests(ids)
        except Exception:
            pass

        try:
            max_steps = 16
            for _ in range(max_steps):
                more = False
                try:
                    if self.has_unfinished_requests():
                        more = True
                except Exception:
                    pass
                try:
                    def _used_blocks() -> int:
                        try:
                            core2 = getattr(self, "engine_core", self)
                            if hasattr(core2, "engine_core"):
                                core2 = core2.engine_core
                            sched2 = getattr(core2, "scheduler", None) or getattr(
                                getattr(core2, "model_executor", None), "scheduler", None)
                            mgr2 = None
                            if sched2 is not None:
                                for name in ("kv_cache_manager","block_manager","cache_manager"):
                                    mgr2 = getattr(sched2, name, None) or mgr2
                            bp2 = getattr(mgr2, "block_pool", None)
                            if bp2 is not None:
                                total = getattr(bp2, "num_gpu_blocks", None) or getattr(bp2, "num_blocks", None) or 0
                                free = 0
                                if hasattr(bp2, "get_num_free_blocks"):
                                    free = int(bp2.get_num_free_blocks())
                                elif hasattr(bp2, "num_free_gpu_blocks"):
                                    free = int(getattr(bp2, "num_free_gpu_blocks"))
                                return int(total) - int(free)
                        except Exception:
                            pass
                        return 0
                    if _used_blocks() > 0:
                        more = True
                except Exception:
                    pass

                if not more:
                    break

                try:
                    self.step()
                except Exception:
                    try:
                        self.wake_up()
                    except Exception:
                        pass
                    _t.sleep(0.01)
            _t.sleep(0.01)
        except Exception:
            pass

        for fn in ("reset_mm_cache", "reset_prefix_cache"):
            try:
                getattr(self, fn)()
            except Exception:
                pass

        try:
            core3 = getattr(self, "engine_core", self)
            if hasattr(core3, "engine_core"):
                core3 = core3.engine_core
            sched3 = getattr(core3, "scheduler", None) or getattr(
                getattr(core3, "model_executor", None), "scheduler", None)
            mgr3 = None
            if sched3 is not None:
                for name in ("kv_cache_manager","block_manager","cache_manager"):
                    mgr3 = getattr(sched3, name, None) or mgr3
            for cand in ("reset","reset_cache","reset_kv_cache","free_all_blocks"):
                if hasattr(mgr3, cand):
                    try: getattr(mgr3, cand)()
                    except Exception: pass
            bp3 = getattr(mgr3, "block_pool", None)
            for cand in ("reset","reset_cache","reset_kv_cache","free_all_blocks"):
                if hasattr(bp3, cand):
                    try: getattr(bp3, cand)()
                    except Exception: pass
        except Exception:
            pass

        try:
            if reset_metrics:
                from vllm.instrumentation.kv_metrics import KVMetricsCollector
                KVMetricsCollector.reset()
            if new_run_id:
                rid = _time.strftime("%Y%m%d-%H%M%S") + "-" + _uuid.uuid4().hex[:6]
                setattr(self, "_run_id", rid)
                if hasattr(self, "_kv_meta") and isinstance(self._kv_meta, dict):
                    self._kv_meta["run_id"] = rid
            if relink_metrics:
                from vllm.instrumentation.kv_metrics import KVMetricsCollector
                col = KVMetricsCollector.get()
                core4 = getattr(self, "engine_core", self)
                if hasattr(core4, "engine_core"):
                    core4 = core4.engine_core
                try:
                    col.link_engine(core4 or self)
                    setattr(self, "_kv_metrics", col)
                except Exception:
                    pass
        except Exception:
            pass

        if empty_cuda_cache:
            try:
                import torch
                _gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception:
                pass


    def _begin_new_run(self) -> None:
        """Always reset metrics and stamp a new run_id. No flags needed."""
        try:
            from vllm.instrumentation.kv_metrics import KVMetricsCollector
            import time as _time, uuid as _uuid
            # 1) reset metrics singleton (hard)
            KVMetricsCollector.reset()

            # 2) new run id
            self._run_id = _time.strftime("%Y%m%d-%H%M%S") + "-" + _uuid.uuid4().hex[:6]
            if hasattr(self, "_kv_meta") and isinstance(getattr(self, "_kv_meta"), dict):
                self._kv_meta["run_id"] = self._run_id

            # 3) relink collector to current engine core
            col = KVMetricsCollector.get()
            eng_like = getattr(self, "engine_core", None) or self
            if hasattr(eng_like, "engine_core"):
                eng_like = eng_like.engine_core
            try:
                col.link_engine(eng_like)
                self._kv_metrics = col
            except Exception:
                pass
        except Exception:
            pass

    def sleep(self, level: int = 1):
        self.engine_core.sleep(level)

    def wake_up(self, tags: Optional[list[str]] = None):
        self.engine_core.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def get_metrics(self) -> list[Metric]:
        assert self.log_stats, "Stat logging disabled"
        return get_metrics_snapshot()

    def get_tokenizer_group(self) -> TokenizerGroup:
        if self.tokenizer is None:
            raise ValueError("Unable to get tokenizer because "
                             "skip_tokenizer_init is True")

        return self.tokenizer

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return self.engine_core.pin_lora(lora_id)

    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def __del__(self):
        try:
            self.shutdown(empty_cuda_cache=True, reset_metrics=False)
        except Exception:
            pass

        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    # --- [KV] Helpers ------------------------------------
    def _kv_compute_context_len(self, prompt_str: Optional[str], request: Any) -> int:
        """Compute prompt token length (BPE) best-effort."""
        try:
            ids = getattr(request, "prompt_token_ids", None)
            if ids is not None:
                return int(len(ids))
        except Exception:
            pass
        try:
            if isinstance(prompt_str, str) and self.tokenizer is not None:
                enc = getattr(self.tokenizer, "encode", None)
                if callable(enc):
                    return int(len(enc(prompt_str)))
        except Exception:
            pass
        return 0

    def _kv_on_enqueue(self, request_id: str, context_len: int) -> None:
        """Record enqueue with context length; no 'start' snapshot by design."""
        cfg = getattr(self._kv_metrics, "cfg", None)
        if not cfg or not cfg.enabled:
            return
        self._kv_metrics.on_enqueue(
            request_id,
            context_len=context_len,
            meta=self._kv_meta,
        )

    def _kv_on_first_token(self, request_id: str) -> None:
        """Mark first token (TTFT & prefill_ms) and take 'prefill' snapshot."""
        cfg = getattr(self._kv_metrics, "cfg", None)
        if not cfg or not cfg.enabled:
            return
        self._kv_metrics.on_first_token(request_id)
        self._kv_metrics.on_prefill_end(request_id)
        self._kv_metrics.snapshot_kv("prefill", request_id)

    def _kv_on_request_end(self, request_id: str) -> None:
        """Take 'decode' snapshot and flush the record."""
        # print("KV: request end start", request_id)
        cfg = getattr(self._kv_metrics, "cfg", None)
        if not cfg or not cfg.enabled:
            return
        self._kv_metrics.snapshot_kv("decode", request_id)
        # print("[KVCHK-END] begin", request_id)
        self._kv_metrics.on_stream_end(request_id)
        # print("[KVCHK-END] done", request_id)

    def _kv_observe_processed_outputs(self, processed_outputs: Any) -> None:
        # print("[KVCHK-OBS] enter, n_ro=", len(getattr(processed_outputs, "request_outputs", []) or []))
        """Observe streaming to (a) detect first token, (b) count tokens, and
        (c) detect completion to flush metrics. Uses minimal assumptions about
        RequestOutput structure for version tolerance."""
        cfg = getattr(self._kv_metrics, "cfg", None)
        if not cfg or not cfg.enabled:
            return

        req_outputs = getattr(processed_outputs, "request_outputs", []) or []
        for ro in req_outputs:
            rid = getattr(ro, "request_id", None) or getattr(ro, "id", None)
            if not isinstance(rid, str):
                continue

            # Try to read current generated token ids length.
            cur_len = None
            out0 = None
            outs = getattr(ro, "outputs", None)
            if outs:
                try:
                    out0 = outs[0]
                except Exception:
                    out0 = None
            if out0 is not None:
                ids = getattr(out0, "token_ids", None)
                if ids is not None:
                    try:
                        cur_len = int(len(ids))
                    except Exception:
                        cur_len = None

            # First token detection + counting (EOS excluded by the collector).
            if cur_len is not None:
                last = self._kv_last_len.get(rid, 0)
                delta = max(0, cur_len - last)
                if rid not in self._kv_seen_first and delta > 0:
                    self._kv_seen_first.add(rid)
                    self._kv_on_first_token(rid)
                if delta > 0:
                    for _ in range(delta):
                        self._kv_metrics.count_generated_token(rid, is_eos=False)
                    self._kv_metrics.bump_peak_alloc(rid)
                    
                self._kv_last_len[rid] = cur_len

            # Finish detection: explicit flag or finish_reason.
            finished = bool(getattr(ro, "finished", False))
            if not finished and out0 is not None:
                fr = getattr(out0, "finish_reason", None)
                if fr in ("eos_token", "stop", "length", "end_of_sequence"):
                    finished = True

            if not finished:
                try:
                    if not self.has_unfinished_requests():
                        finished = True
                except Exception:
                    pass
            # print("[KVCHK-OBS] rid=", rid, "finished=", finished,"fr=", getattr(out0, "finish_reason", None))

            if finished:
                self._kv_on_request_end(rid)
                self._kv_last_len.pop(rid, None)
                self._kv_seen_first.discard(rid)
    
    def shutdown(self, empty_cuda_cache: bool = True, reset_metrics: bool = True) -> None:
        """Best-effort cleanup to release KV cache & GPU memory between runs."""
        try:
            # Abort any dangling requests (if API present)
            try:
                self.engine_core.abort_requests([])  # no-op guard
            except Exception:
                pass

            # Clear caches
            try:
                self.reset_mm_cache()
            except Exception:
                pass
            try:
                self.reset_prefix_cache()
            except Exception:
                pass

            # Downstream shutdown hooks if they exist
            for name in ("shutdown", "close"):
                if hasattr(self.engine_core, name):
                    try:
                        getattr(self.engine_core, name)()
                    except Exception:
                        pass

            # Drop refs + CUDA cache
            import gc
            try:
                delattr(self, "engine_core")
            except Exception:
                pass
            gc.collect()
            if empty_cuda_cache:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            if reset_metrics:
                try:
                    from vllm.instrumentation.kv_metrics import KVMetricsCollector
                    KVMetricsCollector.reset()
                except Exception:
                    pass
        except Exception:
            # never raise in cleanup
            pass
    
    def kv_used_gpu_blocks(self) -> int:
        """Return current number of used GPU KV blocks (0 means fully freed)."""
        try:
            core = getattr(self.engine_core, "engine_core", self.engine_core)
            sched = getattr(core, "scheduler", None) or getattr(
                getattr(core, "model_executor", None), "scheduler", None)
            mgr = None
            if sched is not None:
                for name in ("kv_cache_manager", "block_manager", "cache_manager"):
                    mgr = getattr(sched, name, None) or mgr
            bp = getattr(mgr, "block_pool", None)
            if bp and hasattr(bp, "num_gpu_blocks") and hasattr(bp, "get_num_free_blocks"):
                return int(bp.num_gpu_blocks) - int(bp.get_num_free_blocks())
        except Exception:
            pass
        return 0
