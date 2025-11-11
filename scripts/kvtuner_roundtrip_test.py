#!/usr/bin/env python3
"""
Lightweight test to exercise KVTuner quantize->pack->unpack->dequant roundtrip for K and V paths.
Run from repo root after activating your `vllm` conda env:

  conda activate vllm
  python3 scripts/kvtuner_roundtrip_test.py

It prints max/mean reconstruction errors and exits with non-zero code if errors exceed thresholds.
"""
import sys
import os, json
import math
import traceback
import argparse  # Add this import

# Ensure repo root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

try:
    # Import the quantizer and helper modules from the repo
    from vllm.v1.core.kvtuner_quantizer import KVTunerQuantizer
    from vllm.v1.core.kvtuner_math import KVTunerMath
    from vllm.v1 import core as vllm_core  # just to ensure package import
    from vllm.v1.serial_utils import pack_nbits, unpack_nbits
    from vllm.v1.pool.kv_meta import build_kv_meta, parse_kv_meta
except Exception:
    print("ERROR: failed to import vllm modules; make sure you run this with the vllm conda env activated.")
    traceback.print_exc()
    sys.exit(2)

def parse_args():
    parser = argparse.ArgumentParser(description='KVTuner quantize/pack/unpack/dequant roundtrip test')
    parser.add_argument('--config', '-c', 
                       default=os.path.join(REPO_ROOT, "2025F-Self-Directed-Research/scripts", "layer_quantization_config.json"),
                       help='Path to KVTuner config JSON file')
    return parser.parse_args()

torch.manual_seed(0)

def test_pack_unpack(bits):
    print(f"\nTesting pack/unpack for bits={bits}")
    per_byte = 8 // bits
    n = 1000 + (per_byte - 1)  # make length non-multiple sometimes
    vals = torch.randint(0, 2**bits, (n,), dtype=torch.int32)
    packed = pack_nbits(vals, bits)
    out = unpack_nbits(packed, bits, out_numel=n)
    maxe, mean = norm_err(vals.to(out.dtype), out)
    print(f"  pack/unpack: max err={maxe:.3f} mean err={mean:.6f} len={n}")
    return maxe, mean

def norm_err(a, b):
    """Calculate max and mean absolute error between tensors."""
    diff = (a - b).to(torch.float32)  # Use float32 for error calc
    return diff.abs().max().item(), diff.abs().mean().item()

def verify_tail_roundtrip(x, layer_idx, quantizer, pool):
    """Verify KV cache tail storage/retrieval."""
    # Quantize and store
    packed = quantizer.quantize_k(K=x, layer_idx=layer_idx)
    pool.write_v_tail(0,
                     packed=packed.packed,
                     meta=packed.meta,
                     scale=packed.scale,
                     zp=packed.zp)

    # Read and verify
    read_result = pool.read_v_tail(0)
    meta = parse_kv_meta(read_result.meta)
    x_de = quantizer.dequantize_k(read_result.packed, meta)
    return norm_err(x, x_de)

def test_quant_roundtrip_k():
    print("\nTesting K quantize->pack->unpack->dequant roundtrip (axis=1, example keys)")
    # Simulate key tensor: (num_heads, seq_len, head_dim) typical: axis=1 is per-row grouping
    num_heads = 4
    seq_len = 16
    head_dim = 64
    x = torch.randn(num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    args = parse_args()
    cfg_path = args.config
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            kvt_cfg = json.load(f)
            
        # Print layer group info from config
        if "groups" in kvt_cfg:
            print("\nLayer groups from config:")
            for group_name, group_info in kvt_cfg["groups"].items():
                print(f"  {group_name}: {group_info}")
        if "bit_candidates_k" in kvt_cfg:
            print(f"  K bits candidates: {kvt_cfg['bit_candidates_k']}")
        if "bit_candidates_v" in kvt_cfg:
            print(f"  V bits candidates: {kvt_cfg['bit_candidates_v']}")
    else:
        print(f"Warning: Config file not found at {cfg_path}, using default values")
        kvt_cfg = {"bits": 4, "q_group_size": 16, "axis": 1}

    try:
        quantizer = KVTunerQuantizer(kvt_cfg)
    except TypeError:
        try:
            quantizer = KVTunerQuantizer(config=kvt_cfg)
        except TypeError:
            quantizer = KVTunerQuantizer(
                bits=kvt_cfg.get("bits", 4),
                q_group_size=kvt_cfg.get("q_group_size", 16),
                axis=kvt_cfg.get("axis", 1),
            )

    # Test with a few representative layers
    for layer_idx in [0, 1, 4, 8]:
        packed_k = quantizer.quantize_k(K=x, layer_idx=layer_idx)
        meta = parse_kv_meta(packed_k.meta_bytes)
        x_de = quantizer.dequantize_k(packed_k)
        maxe, mean = norm_err(x, x_de)
        print(f"  Layer {layer_idx} K dequant: max err={maxe:.6f} mean err={mean:.8f}")
        # Also print the actual bits used (from meta)
        print(f"    Used {meta.get('bits', 'unknown')} bits")

    return maxe, mean


def test_quant_roundtrip_v():
    print("\nTesting V quantize->pack->unpack->dequant roundtrip (axis= -1 V path)")
    # V tensors often have shape (seq_len, num_heads, head_dim) or (batch, seq_len, dim)
    # We'll choose a 2D example and axis=-1 semantics inside quantizer
    seq = 16
    num_heads = 4
    head_dim = 64
    x = torch.randn(seq, num_heads, head_dim, dtype=torch.bfloat16)
    
    args = parse_args()
    cfg_path = args.config
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            kvt_cfg = json.load(f)
    else:
        print(f"Warning: Config file not found at {cfg_path}, using default values")
        kvt_cfg = {"bits": 4, "q_group_size": 16, "axis": -1}
    

    try:
        quantizer = KVTunerQuantizer(kvt_cfg)
    except TypeError:
        try:
            quantizer = KVTunerQuantizer(config=kvt_cfg)
        except TypeError:
            quantizer = KVTunerQuantizer(
                bits=kvt_cfg.get("bits", 4),
                q_group_size=kvt_cfg.get("q_group_size", 16),
                axis=kvt_cfg.get("axis", -1),
            )

    packed_v = quantizer.quantize_v(v=x, layer_idx=0)
    meta_bytes = packed_v.meta_bytes
    packed_bytes = packed_v.packed

    meta = parse_kv_meta(meta_bytes)
    x_de = quantizer.dequantize_v(packed_v)

    maxe, mean = norm_err(x, x_de)
    print(f"  V dequant: max err={maxe:.6f} mean err={mean:.8f}")
    return maxe, mean


def main():
    args = parse_args()
    print(f"Using config file: {args.config}")
    exit_code = 0

    # Load and print config summary
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
            print("\nKVTuner config summary:")
            for key, value in config.items():
                if isinstance(value, dict):
                    print(f"  {key}: <dict with {len(value)} items>")
                elif isinstance(value, list):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")

    # Basic pack/unpack tests
    for bits in (1, 2, 4, 8):
        maxe, mean = test_pack_unpack(bits)
        if maxe > 0:
            print(f"WARNING: Non-zero pack/unpack error for {bits} bits")
            exit_code = 1

    # Quantizer roundtrip tests
    maxk, meank = test_quant_roundtrip_k()
    maxv, meanv = test_quant_roundtrip_v()

    # Check for concerning error levels
    if maxk > 0.1 or maxv > 0.1:  # 10% error threshold
        print("WARNING: Large quantization errors detected")
        exit_code = 2

    print(f"\nDone. Exit code: {exit_code}")
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
