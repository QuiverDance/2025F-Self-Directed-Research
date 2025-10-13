# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KVTuner layer-wise KV cache quantization configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.models.utils import extract_layer_index

logger = init_logger(__name__)


def _validate_bits(name: str, value: int) -> int:
    if value not in (2, 4, 8):
        raise ValueError(
            f"Unsupported {name} bit-width {value}. "
            "Only 2-, 4- and 8-bit K/V caches are supported by the KVTuner "
            "integration.")
    return value


def _normalize_layers(raw_layers: Any) -> Iterable[int]:
    if isinstance(raw_layers, int):
        yield raw_layers
        return
    if isinstance(raw_layers, Sequence):
        if len(raw_layers) == 0:
            raise ValueError("Layer override cannot be empty")
        if len(raw_layers) == 2 and all(isinstance(x, int)
                                        for x in raw_layers):
            start, end = raw_layers
            if end < start:
                raise ValueError(
                    f"Invalid layer range {raw_layers}: end < start")
            yield from range(start, end + 1)
            return
        for item in raw_layers:
            if not isinstance(item, int):
                raise ValueError(
                    "Layer overrides must be integers or two-integer "
                    "ranges")
            yield item
        return
    raise TypeError(
        "Layer overrides must be an integer, a two-integer range or an "
        "iterable of integers.")


@dataclass
class _LayerSchedule:
    k_bits: int
    v_bits: int


class KVTunerKVCacheMethod(BaseKVCacheMethod):
    """Quantization method that records K/V bit-width metadata."""

    def __init__(self, quant_config: "KVTunerConfig", layer_prefix: str):
        super().__init__(quant_config)
        self._layer_prefix = layer_prefix

    def create_weights(self, layer: torch.nn.Module):  # type: ignore[override]
        super().create_weights(layer)

        schedule = self.quant_config.get_schedule_for_prefix(self._layer_prefix)
        k_bits, v_bits = schedule.k_bits, schedule.v_bits

        # Keep both Python ints for convenience and register buffers to allow
        # attention backends to consume them without additional device copies.
        layer.k_bits = k_bits
        layer.v_bits = v_bits
        layer.register_buffer("_k_bits", torch.tensor(k_bits, dtype=torch.int32),
                              persistent=False)
        layer.register_buffer("_v_bits", torch.tensor(v_bits, dtype=torch.int32),
                              persistent=False)

    def process_weights_after_loading(self,
                                      layer: torch.nn.Module) -> None:  # noqa: D401
        """Finalize metadata after checkpoints are materialized."""

        super().process_weights_after_loading(layer)
        if not hasattr(layer, "k_bits") or not hasattr(layer, "v_bits"):
            logger.warning_once(
                "Attention layer %s is missing K/V bit metadata; falling back "
                "to default 8-bit cache.", self._layer_prefix)
            layer.k_bits = 8
            layer.v_bits = 8
            layer._k_bits.copy_(torch.tensor(8, dtype=torch.int32))
            layer._v_bits.copy_(torch.tensor(8, dtype=torch.int32))


class KVTunerConfig(QuantizationConfig):
    """Quantization configuration for KVTuner-style KV cache schedules."""

    def __init__(self, default_k_bits: int, default_v_bits: int,
                 layer_schedules: dict[int, _LayerSchedule],
                 module_overrides: dict[str, _LayerSchedule]):
        super().__init__()
        self.default_schedule = _LayerSchedule(default_k_bits, default_v_bits)
        self.layer_schedules = layer_schedules
        self.module_overrides = module_overrides

    # QuantizationConfig API -------------------------------------------------
    def get_name(self) -> str:
        return "kvtuner"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # Activations remain in fp16/bf16; quantization only applies to KV.
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # The schedule is agnostic to GPU architecture; kernels validate later.
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["kvtuner_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "KVTunerConfig":
        default_k_bits = _validate_bits(
            "default_k_bits", config.get("default_k_bits", 8))
        default_v_bits = _validate_bits(
            "default_v_bits", config.get("default_v_bits", 8))

        layer_schedules: dict[int, _LayerSchedule] = {}
        for entry in config.get("layer_bits", []):
            if not isinstance(entry, dict):
                raise TypeError(
                    "Each layer_bits entry must be a dictionary.")
            if "layer" in entry and "layers" in entry:
                raise ValueError(
                    "Specify either 'layer' or 'layers' in a layer_bits "
                    "entry, not both.")
            if "layer" in entry:
                layers = [entry["layer"]]
            elif "layers" in entry:
                layers = list(_normalize_layers(entry["layers"]))
            else:
                raise ValueError(
                    "Layer override entry must contain 'layer' or 'layers'.")

            k_bits = _validate_bits("k_bits", entry.get("k_bits", default_k_bits))
            v_bits = _validate_bits("v_bits", entry.get("v_bits", default_v_bits))

            for layer_idx in layers:
                if not isinstance(layer_idx, int):
                    raise TypeError("Layer indices must be integers.")
                layer_schedules[layer_idx] = _LayerSchedule(k_bits, v_bits)

        module_overrides: dict[str, _LayerSchedule] = {}
        for module_name, override in config.get("module_bits", {}).items():
            if not isinstance(override, dict):
                raise TypeError("module_bits entries must be dictionaries.")
            k_bits = _validate_bits("k_bits",
                                    override.get("k_bits", default_k_bits))
            v_bits = _validate_bits("v_bits",
                                    override.get("v_bits", default_v_bits))
            module_overrides[module_name] = _LayerSchedule(k_bits, v_bits)

        return cls(default_k_bits, default_v_bits, layer_schedules,
                   module_overrides)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        # Only attach to attention layers, whose prefixes end with '.attn'.
        if not prefix.endswith(".attn"):
            return None
        return KVTunerKVCacheMethod(self, prefix)

    # KVTuner specific helpers ----------------------------------------------
    def get_schedule_for_prefix(self, prefix: str) -> _LayerSchedule:
        if prefix in self.module_overrides:
            return self.module_overrides[prefix]

        try:
            layer_idx = extract_layer_index(prefix)
        except AssertionError:
            return self.default_schedule

        return self.layer_schedules.get(layer_idx, self.default_schedule)

    def apply_vllm_mapper(self, hf_to_vllm_mapper):  # type: ignore[override]
        if not self.module_overrides:
            return

        remapped: dict[str, _LayerSchedule] = {}
        for module_name, schedule in self.module_overrides.items():
            mapped = hf_to_vllm_mapper.apply_list([module_name])
            if not mapped:
                logger.warning_once(
                    "Skipping module override %s because it was removed by "
                    "the weights mapper.", module_name)
                continue
            remapped[mapped[0]] = schedule

        self.module_overrides = remapped


__all__ = ["KVTunerConfig", "KVTunerKVCacheMethod"]

