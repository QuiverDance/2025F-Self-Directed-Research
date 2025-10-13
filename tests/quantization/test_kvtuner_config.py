# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.quantization.kvtuner import KVTunerConfig


def make_config(**kwargs):
    base = {
        "default_k_bits": 8,
        "default_v_bits": 8,
    }
    base.update(kwargs)
    return base


def test_kvtuner_layer_overrides():
    cfg = KVTunerConfig.from_config(
        make_config(layer_bits=[{"layer": 0, "k_bits": 4, "v_bits": 2},
                                {
                                    "layers": [3, 5],
                                    "k_bits": 2,
                                    "v_bits": 4
                                }]))

    schedule_0 = cfg.get_schedule_for_prefix("model.layers.0.self_attn.attn")
    assert schedule_0.k_bits == 4
    assert schedule_0.v_bits == 2

    schedule_4 = cfg.get_schedule_for_prefix("model.layers.4.self_attn.attn")
    assert schedule_4.k_bits == 2
    assert schedule_4.v_bits == 4

    fallback = cfg.get_schedule_for_prefix("model.layers.9.self_attn.attn")
    assert fallback.k_bits == 8
    assert fallback.v_bits == 8


def test_kvtuner_module_overrides_and_mapper():
    cfg = KVTunerConfig.from_config(
        make_config(module_bits={
            "encoder.layers.0.self_attn.attn": {
                "k_bits": 2,
                "v_bits": 4,
            }
        }))

    class DummyMapper:

        def apply_list(self, modules):
            return [m.replace("encoder", "model") for m in modules]

    cfg.apply_vllm_mapper(DummyMapper())
    schedule = cfg.get_schedule_for_prefix("model.layers.0.self_attn.attn")
    assert schedule.k_bits == 2
    assert schedule.v_bits == 4


@pytest.mark.parametrize("bad_bits", [1, 3, 16])
def test_kvtuner_rejects_invalid_bits(bad_bits):
    with pytest.raises(ValueError):
        KVTunerConfig.from_config(
            make_config(layer_bits=[{"layer": 0, "k_bits": bad_bits}]))
