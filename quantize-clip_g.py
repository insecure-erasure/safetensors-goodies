#!/usr/bin/env python3
"""
quantize-clip_g.py
----------------------
Mixed FP16/FP8 (E4M3) quantization for a CLIP-G text encoder extracted
from an SDXL checkpoint in OpenCLIP format (transformer.resblocks.N.*).

Quantization strategy (based on ViT-bigG-14, 32 transformer blocks):

  By default (conservative, no structural changes to the state dict):
    - MLP weights (c_fc, c_proj) in intermediate blocks  -> float8_e4m3fn
    - Attention out_proj.weight in intermediate blocks    -> float16 (opt-in FP8 via --attn-out-fp8)
    - in_proj_weight (fused Q/K/V) kept intact            -> float16
    - First N blocks (--first-blocks-keep) + last block   -> float16 (preserved)
    - All biases                                          -> float16
    - LayerNorm, embeddings, projection, logit_scale      -> float16

  With --split-attn-qkv (splits in_proj_weight into separate Q/K/V tensors):
    - K, V projections in intermediate blocks             -> float8_e4m3fn
    - Q projection in intermediate blocks                 -> float16 (opt-in FP8 via --attn-q-fp8)
    - in_proj_bias is also split into q/k/v biases        -> float16
    - Use --keep-attn-kv-fp16 to keep K/V at FP16 after the split
      (useful for isolating the impact of the split vs K/V quantization)

  Note: --attn-out-fp8 works independently of --split-attn-qkv because
  out_proj.weight is already a separate tensor in the original state dict.

The output safetensors can be loaded directly by ComfyUI-GGUF's
DualCLIPLoaderGGUF node, which respects per-tensor dtypes.

Requirements:
    pip install safetensors torch

Usage:
    # Conservative: only MLP weights quantized to FP8
    python quantize-clip_g .py -i clip_g.safetensors -o clip_g_fp8.safetensors

    # Split attention and quantize K/V to FP8
    python quantize-clip_g .py -i clip_g.safetensors -o clip_g_fp8.safetensors --split-attn-qkv

    # Split attention but keep K/V at FP16 (diagnostic: isolate split vs quantization)
    python quantize-clip_g .py -i clip_g.safetensors -o clip_g_fp8.safetensors --split-attn-qkv --keep-attn-kv-fp16

    # Also quantize out_proj (independent of split)
    python quantize-clip_g .py -i clip_g.safetensors -o clip_g_fp8.safetensors --attn-out-fp8

    # Maximum quantization: split + all attention weights to FP8
    python quantize-clip_g .py -i clip_g.safetensors -o clip_g_fp8.safetensors --split-attn-qkv --attn-q-fp8 --attn-out-fp8

    # Dry run with verbose per-tensor mapping
    python quantize-clip_g .py -i clip_g.safetensors -o clip_g_fp8.safetensors --split-attn-qkv --dry-run --verbose

    # Adjust number of preserved initial blocks
    python quantize-clip_g .py -i clip_g.safetensors -o clip_g_fp8.safetensors --first-blocks-keep 8

    # Analyze tensors to identify sensitive blocks before quantizing
    python quantize-clip_g .py -i clip_g.safetensors --analyze
"""

import argparse
import sys
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOTAL_BLOCKS      = 32   # transformer blocks in ViT-bigG-14 text encoder
DEFAULT_KEEP      = 7    # first N blocks kept at FP16 (0-indexed: 0..N-1)
                         # last block (index 31) is always kept at FP16

# Maximum representable value in float8_e4m3fn
FP8_E4M3_MAX      = 448.0

# MLP weight suffixes: always quantized to FP8 in intermediate blocks.
# Biases are intentionally excluded: they are tiny and numerically sensitive.
FP8_MLP_SUFFIXES = {
    "mlp.c_fc.weight",    # MLP first linear
    "mlp.c_proj.weight",  # MLP second linear
}

# Attention weight suffixes added to the FP8 set when --split-attn-qkv is active
FP8_ATTN_KV_SUFFIXES = {
    "attn.k_proj.weight", # Attention key projection  (from split in_proj)
    "attn.v_proj.weight", # Attention value projection (from split in_proj)
}

# Optional suffixes controlled by individual CLI flags
FP8_ATTN_OUT_SUFFIX = "attn.out_proj.weight"  # --attn-out-fp8
FP8_ATTN_Q_SUFFIX   = "attn.q_proj.weight"    # --attn-q-fp8

# Regex to detect a resblock key and extract block index + suffix
# Matches: transformer.resblocks.{N}.{suffix}
RESBLOCK_RE = re.compile(r"^transformer\.resblocks\.(\d+)\.(.+)$")

# in_proj_weight holds [Q;K;V] concatenated — needs special handling
IN_PROJ_WEIGHT = "attn.in_proj_weight"
IN_PROJ_BIAS   = "attn.in_proj_bias"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_fp8_support() -> bool:
    """Return True if this PyTorch build exposes float8_e4m3fn."""
    return hasattr(torch, "float8_e4m3fn")


def check_outliers(tensor: torch.Tensor, key: str, verbose: bool) -> None:
    """
    Check if a tensor has values outside the representable range of FP8 E4M3
    (±448). Emits a warning with tensor name, max absolute value, and
    percentage of outlier values.
    """
    abs_vals = tensor.abs().float()
    max_val  = abs_vals.max().item()
    if max_val > FP8_E4M3_MAX:
        outlier_count = (abs_vals > FP8_E4M3_MAX).sum().item()
        outlier_pct   = outlier_count / tensor.numel() * 100.0
        print(f"  WARNING: {key}")
        print(f"    max |value| = {max_val:.4f} (FP8 E4M3 max = {FP8_E4M3_MAX})")
        print(f"    outliers    = {outlier_count} ({outlier_pct:.4f}%)")
    elif verbose:
        print(f"    outlier check OK: max |value| = {max_val:.4f}")


def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    """Cast a FP16 weight tensor to float8_e4m3fn via FP32 intermediate."""
    return tensor.to(torch.float32).to(torch.float8_e4m3fn)


def build_fp8_suffixes(split_attn: bool, keep_kv_fp16: bool,
                       attn_out_fp8: bool, attn_q_fp8: bool) -> set:
    """
    Return the full set of weight suffixes to quantize to FP8 based on
    active flags.

    Base set: MLP weights (always included).
    Conditionally added:
      - K/V projections: when --split-attn-qkv is set and --keep-attn-kv-fp16 is not
      - Q projection:    when --split-attn-qkv and --attn-q-fp8 are both set
      - out_proj:        when --attn-out-fp8 is set (independent of split)
    """
    suffixes = set(FP8_MLP_SUFFIXES)
    if split_attn and not keep_kv_fp16:
        suffixes.update(FP8_ATTN_KV_SUFFIXES)
    if split_attn and attn_q_fp8:
        suffixes.add(FP8_ATTN_Q_SUFFIX)
    if attn_out_fp8:
        suffixes.add(FP8_ATTN_OUT_SUFFIX)
    return suffixes


def is_fp8_candidate(block_idx: int, suffix: str, first_keep: int, fp8_suffixes: set) -> bool:
    """
    Return True if this (block_idx, suffix) pair should be quantized to FP8.
    Conditions:
      - Block is in the intermediate range [first_keep, TOTAL_BLOCKS-1)
      - Suffix is in the FP8 candidate set
    """
    last_block = TOTAL_BLOCKS - 1
    if block_idx < first_keep or block_idx == last_block:
        return False
    return suffix in fp8_suffixes


def split_in_proj(tensor: torch.Tensor):
    """
    Split a fused in_proj_weight [Q+K+V, dim] or in_proj_bias [Q+K+V]
    into three equal chunks.
    For weights: returns (q, k, v) each of shape [dim, embed_dim].
    For biases:  returns (q, k, v) each of shape [dim].
    """
    chunk = tensor.shape[0] // 3
    return tensor[:chunk], tensor[chunk:2*chunk], tensor[2*chunk:]


def log_tensor(key: str, src_dtype: torch.dtype, dst_dtype: torch.dtype, verbose: bool) -> None:
    """Print per-tensor dtype mapping when verbose mode is active."""
    if verbose:
        src = str(src_dtype).replace("torch.", "")
        dst = str(dst_dtype).replace("torch.", "")
        if src == dst:
            print(f"  {key}: {src} (unchanged)")
        else:
            print(f"  {key}: {src} -> {dst}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

# Weight suffixes to include in per-block analysis.
# These are all the weight tensors that could potentially be quantized.
ANALYSIS_WEIGHT_SUFFIXES = {
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
    "attn.in_proj_weight",   # fused Q/K/V (will be analyzed as a whole and split)
    "attn.out_proj.weight",
}


def analyze_tensor(tensor: torch.Tensor):
    """
    Compute quantization sensitivity metrics for a single weight tensor.
    Returns a dict with:
      - params:       number of parameters
      - norm:         Frobenius norm of the tensor
      - max_abs:      maximum absolute value
      - std:          standard deviation
      - outliers:     number of values outside FP8 E4M3 representable range (±448)
      - outlier_pct:  percentage of outlier values
      - quant_error:  relative quantization error (NRMSE) from FP16 -> FP8 -> FP16 roundtrip
    """
    t = tensor.float()
    abs_vals = t.abs()
    max_abs  = abs_vals.max().item()

    outlier_count = (abs_vals > FP8_E4M3_MAX).sum().item()
    outlier_pct   = outlier_count / t.numel() * 100.0

    # Roundtrip quantization error: FP16 -> FP32 -> FP8 -> FP32
    t_fp8       = t.to(torch.float8_e4m3fn).float()
    error       = (t - t_fp8)
    rmse        = error.pow(2).mean().sqrt().item()
    t_norm      = t.pow(2).mean().sqrt().item()
    quant_error = rmse / t_norm if t_norm > 0 else 0.0

    return {
        "params":      tensor.numel(),
        "norm":        t.norm().item(),
        "max_abs":     max_abs,
        "std":         t.std().item(),
        "outliers":    outlier_count,
        "outlier_pct": outlier_pct,
        "quant_error": quant_error,
    }


def analyze(input_path: Path):
    """
    Analyze all weight tensors in the model to identify blocks that are
    sensitive to FP8 quantization. Prints per-block metrics and a
    recommendation of which blocks to protect.
    """
    if not check_fp8_support():
        print("ERROR: float8_e4m3fn is not available in this PyTorch build.")
        print("       Requires PyTorch >= 2.1 compiled with FP8 support.")
        sys.exit(1)

    print(f"Loading: {input_path}")
    sd = load_file(str(input_path))
    print(f"  Tensors loaded: {len(sd)}")
    print()

    # Collect per-block, per-component metrics
    # Structure: block_data[block_idx][component_name] = metrics dict
    block_data = {}

    for key, tensor in sd.items():
        m = RESBLOCK_RE.match(key)
        if m is None:
            continue

        block_idx = int(m.group(1))
        suffix    = m.group(2)

        if suffix not in ANALYSIS_WEIGHT_SUFFIXES:
            continue

        if block_idx not in block_data:
            block_data[block_idx] = {}

        if suffix == IN_PROJ_WEIGHT:
            # Analyze the fused tensor as a whole
            block_data[block_idx]["in_proj (fused)"] = analyze_tensor(tensor)
            # Also analyze Q/K/V components separately
            q, k, v = split_in_proj(tensor)
            block_data[block_idx]["attn Q"] = analyze_tensor(q)
            block_data[block_idx]["attn K"] = analyze_tensor(k)
            block_data[block_idx]["attn V"] = analyze_tensor(v)
        else:
            # Friendly name for display
            name_map = {
                "mlp.c_fc.weight":       "MLP c_fc",
                "mlp.c_proj.weight":     "MLP c_proj",
                "attn.out_proj.weight":  "attn out_proj",
            }
            name = name_map.get(suffix, suffix)
            block_data[block_idx][name] = analyze_tensor(tensor)

    if not block_data:
        print("No resblock weight tensors found in the file.")
        return

    # --- Per-block detailed report ---
    print("=" * 90)
    print(f"{'Block':>5} {'Component':<18} {'Params':>10} {'Norm':>10} "
          f"{'Max|W|':>10} {'Std':>10} {'Outliers':>10} {'QError%':>10}")
    print("-" * 90)

    # Collect per-block aggregate quantization error for ranking
    block_agg_error = {}

    for block_idx in sorted(block_data.keys()):
        components = block_data[block_idx]
        weighted_error_sum = 0.0
        total_params = 0

        for comp_name, m in components.items():
            # Skip the fused in_proj from the aggregate since we already
            # count Q/K/V separately
            if comp_name == "in_proj (fused)":
                continue
            weighted_error_sum += m["quant_error"] * m["params"]
            total_params += m["params"]

        block_agg_error[block_idx] = weighted_error_sum / total_params if total_params > 0 else 0.0

        for comp_name, m in components.items():
            outlier_str = f"{m['outliers']}" if m["outliers"] == 0 else f"{m['outliers']} ({m['outlier_pct']:.3f}%)"
            print(f"{block_idx:>5} {comp_name:<18} {m['params']:>10,} {m['norm']:>10.2f} "
                  f"{m['max_abs']:>10.4f} {m['std']:>10.6f} {outlier_str:>10} "
                  f"{m['quant_error']*100:>9.4f}%")
        print()

    # --- Block ranking by quantization sensitivity ---
    print("=" * 90)
    print("Block sensitivity ranking (weighted average quantization error, higher = more sensitive)")
    print("-" * 90)

    ranked = sorted(block_agg_error.items(), key=lambda x: x[1], reverse=True)

    # Compute mean and standard deviation for threshold
    errors = [e for _, e in ranked]
    mean_error = sum(errors) / len(errors)
    std_error  = (sum((e - mean_error) ** 2 for e in errors) / len(errors)) ** 0.5

    # Blocks with error > mean + 1 std are flagged as sensitive
    threshold = mean_error + std_error

    sensitive_blocks = []
    for block_idx, err in ranked:
        flag = " << SENSITIVE" if err > threshold else ""
        print(f"  Block {block_idx:>2}: {err*100:.4f}%{flag}")
        if err > threshold:
            sensitive_blocks.append(block_idx)

    print()
    print(f"  Mean quantization error : {mean_error*100:.4f}%")
    print(f"  Std deviation           : {std_error*100:.4f}%")
    print(f"  Sensitivity threshold   : {threshold*100:.4f}% (mean + 1 std)")
    print()

    # --- Recommendation ---
    if sensitive_blocks:
        sensitive_blocks.sort()
        block_list = ", ".join(str(b) for b in sensitive_blocks)
        print(f"  Recommendation: consider protecting blocks {block_list}")
        print(f"  These blocks show significantly higher quantization error than average.")

        # Suggest --first-blocks-keep value if sensitive blocks form a prefix
        contiguous_from_zero = 0
        for i, b in enumerate(sensitive_blocks):
            if b == i:
                contiguous_from_zero = i + 1
            else:
                break

        if contiguous_from_zero > 0:
            print(f"  Suggested --first-blocks-keep: {contiguous_from_zero} "
                  f"(covers initial sensitive blocks 0-{contiguous_from_zero-1})")

        # Check for sensitive blocks near the end
        last_block = TOTAL_BLOCKS - 1
        sensitive_tail = [b for b in sensitive_blocks if b >= last_block - 3 and b != last_block]
        if sensitive_tail:
            tail_list = ", ".join(str(b) for b in sensitive_tail)
            print(f"  Note: blocks {tail_list} near the end are also sensitive "
                  f"(block {last_block} is always protected).")
    else:
        print("  No blocks show significantly elevated quantization error.")
        print("  The default --first-blocks-keep value should be adequate.")

    print()


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(input_path: Path, output_path: Path, first_keep: int, dry_run: bool,
            split_attn: bool, keep_kv_fp16: bool, attn_q_fp8: bool, attn_out_fp8: bool,
            verbose: bool):
    if not check_fp8_support():
        print("ERROR: float8_e4m3fn is not available in this PyTorch build.")
        print("       Requires PyTorch >= 2.1 compiled with FP8 support.")
        sys.exit(1)

    fp8_suffixes = build_fp8_suffixes(split_attn, keep_kv_fp16, attn_out_fp8, attn_q_fp8)

    print(f"Loading: {input_path}")
    sd_in = load_file(str(input_path))
    print(f"  Tensors loaded : {len(sd_in)}")
    print(f"  FP16 preserved : blocks 0-{first_keep-1} and block {TOTAL_BLOCKS-1}")

    if not split_attn:
        fp8_desc = "MLP weights"
        if attn_out_fp8:
            fp8_desc += " + attn out_proj"
        print(f"  FP8 candidates : blocks {first_keep}-{TOTAL_BLOCKS-2} ({fp8_desc})")
        print(f"  in_proj_weight : kept fused in FP16 (use --split-attn-qkv to split)")
    else:
        fp8_desc = "MLP weights"
        if keep_kv_fp16:
            fp8_desc += " (K/V split but kept FP16)"
        else:
            fp8_desc += " + attn K/V"
        if attn_q_fp8:
            fp8_desc += " + attn Q"
        if attn_out_fp8:
            fp8_desc += " + attn out_proj"
        print(f"  FP8 candidates : blocks {first_keep}-{TOTAL_BLOCKS-2} ({fp8_desc})")
        print(f"  Split attn     : enabled (in_proj_weight -> Q/K/V)")
        if keep_kv_fp16:
            print(f"  Keep K/V FP16  : enabled (--keep-attn-kv-fp16)")
    print()

    if verbose:
        print("--- Per-tensor mapping ---")

    sd_out   = {}
    stats    = {"fp16": 0, "fp8": 0, "split_qkv": 0}
    bytes_in = sum(t.numel() * t.element_size() for t in sd_in.values())

    for key, tensor in sd_in.items():
        m = RESBLOCK_RE.match(key)

        # ----------------------------------------------------------------
        # Tensors outside resblocks: always FP16
        # ----------------------------------------------------------------
        if m is None:
            sd_out[key] = tensor.to(torch.float16)
            log_tensor(key, tensor.dtype, torch.float16, verbose)
            stats["fp16"] += 1
            continue

        block_idx = int(m.group(1))
        suffix    = m.group(2)

        # ----------------------------------------------------------------
        # Handle fused in_proj_weight
        # ----------------------------------------------------------------
        if suffix == IN_PROJ_WEIGHT:
            if not split_attn:
                # Default: keep the fused tensor intact at FP16
                sd_out[key] = tensor.to(torch.float16)
                log_tensor(key, tensor.dtype, torch.float16, verbose)
                stats["fp16"] += 1
                continue

            # --split-attn-qkv: split Q/K/V and apply per-component strategy
            q, k, v = split_in_proj(tensor)
            last_block = TOTAL_BLOCKS - 1
            eligible = (block_idx >= first_keep) and (block_idx != last_block)

            base = f"transformer.resblocks.{block_idx}"

            if eligible:
                # Q: FP8 only if --attn-q-fp8 is set, otherwise FP16
                q_key = f"{base}.attn.q_proj.weight"
                if attn_q_fp8:
                    check_outliers(q, q_key, verbose)
                    sd_out[q_key] = to_fp8(q)
                    log_tensor(q_key, tensor.dtype, torch.float8_e4m3fn, verbose)
                    stats["fp8"] += 1
                else:
                    sd_out[q_key] = q.to(torch.float16)
                    log_tensor(q_key, tensor.dtype, torch.float16, verbose)
                    stats["fp16"] += 1

                # K and V: FP8 by default, FP16 if --keep-attn-kv-fp16 is set
                k_key = f"{base}.attn.k_proj.weight"
                v_key = f"{base}.attn.v_proj.weight"
                if keep_kv_fp16:
                    sd_out[k_key] = k.to(torch.float16)
                    sd_out[v_key] = v.to(torch.float16)
                    log_tensor(k_key, tensor.dtype, torch.float16, verbose)
                    log_tensor(v_key, tensor.dtype, torch.float16, verbose)
                    stats["fp16"] += 2
                else:
                    check_outliers(k, k_key, verbose)
                    check_outliers(v, v_key, verbose)
                    sd_out[k_key] = to_fp8(k)
                    sd_out[v_key] = to_fp8(v)
                    log_tensor(k_key, tensor.dtype, torch.float8_e4m3fn, verbose)
                    log_tensor(v_key, tensor.dtype, torch.float8_e4m3fn, verbose)
                    stats["fp8"] += 2
                stats["split_qkv"] += 1
            else:
                # Protected block: keep fused in_proj_weight intact at FP16
                sd_out[key] = tensor.to(torch.float16)
                log_tensor(key, tensor.dtype, torch.float16, verbose)
                stats["fp16"] += 1
            continue

        # ----------------------------------------------------------------
        # Handle fused in_proj_bias
        # ----------------------------------------------------------------
        if suffix == IN_PROJ_BIAS:
            if not split_attn:
                # Default: keep the fused bias intact at FP16
                sd_out[key] = tensor.to(torch.float16)
                log_tensor(key, tensor.dtype, torch.float16, verbose)
                stats["fp16"] += 1
                continue

            # --split-attn-qkv: split bias consistently with weights
            last_block = TOTAL_BLOCKS - 1
            eligible = (block_idx >= first_keep) and (block_idx != last_block)

            if eligible:
                q_bias, k_bias, v_bias = split_in_proj(tensor)
                base = f"transformer.resblocks.{block_idx}"
                q_bias_key = f"{base}.attn.q_proj.bias"
                k_bias_key = f"{base}.attn.k_proj.bias"
                v_bias_key = f"{base}.attn.v_proj.bias"

                sd_out[q_bias_key] = q_bias.to(torch.float16)
                sd_out[k_bias_key] = k_bias.to(torch.float16)
                sd_out[v_bias_key] = v_bias.to(torch.float16)
                log_tensor(q_bias_key, tensor.dtype, torch.float16, verbose)
                log_tensor(k_bias_key, tensor.dtype, torch.float16, verbose)
                log_tensor(v_bias_key, tensor.dtype, torch.float16, verbose)
                stats["fp16"] += 3
            else:
                # Protected block: keep fused bias at FP16
                sd_out[key] = tensor.to(torch.float16)
                log_tensor(key, tensor.dtype, torch.float16, verbose)
                stats["fp16"] += 1
            continue

        # ----------------------------------------------------------------
        # Regular resblock tensor
        # ----------------------------------------------------------------
        if is_fp8_candidate(block_idx, suffix, first_keep, fp8_suffixes):
            check_outliers(tensor, key, verbose)
            sd_out[key] = to_fp8(tensor)
            log_tensor(key, tensor.dtype, torch.float8_e4m3fn, verbose)
            stats["fp8"] += 1
        else:
            sd_out[key] = tensor.to(torch.float16)
            log_tensor(key, tensor.dtype, torch.float16, verbose)
            stats["fp16"] += 1

    if verbose:
        print("--- End per-tensor mapping ---")
        print()

    # ----------------------------------------------------------------
    # Tensor count verification
    # ----------------------------------------------------------------
    if split_attn:
        # Each split replaces 1 weight + 1 bias with 3 + 3 = +4 net tensors
        expected_tensors = len(sd_in) + stats["split_qkv"] * 4
    else:
        expected_tensors = len(sd_in)
    actual_tensors = len(sd_out)

    if actual_tensors != expected_tensors:
        print(f"WARNING: Tensor count mismatch!")
        print(f"  Expected : {expected_tensors}")
        print(f"  Actual   : {actual_tensors}")
    elif verbose:
        print(f"Tensor count verified: {actual_tensors} (expected {expected_tensors})")

    # ----------------------------------------------------------------
    # Size report
    # ----------------------------------------------------------------
    bytes_out = sum(t.numel() * t.element_size() for t in sd_out.values())
    mb_in     = bytes_in  / (1024 ** 2)
    mb_out    = bytes_out / (1024 ** 2)
    reduction = (1.0 - bytes_out / bytes_in) * 100.0

    print()
    print("=== Summary ===")
    print(f"  FP16 tensors kept       : {stats['fp16']}")
    print(f"  FP8  tensors quantized  : {stats['fp8']}")
    print(f"  in_proj blocks split    : {stats['split_qkv']}")
    print(f"  Output tensors (total)  : {actual_tensors}")
    print(f"  Input size  (estimated) : {mb_in:.1f} MB")
    print(f"  Output size (estimated) : {mb_out:.1f} MB")
    print(f"  Size reduction          : {reduction:.1f}%")
    print()

    if dry_run:
        print("Dry run — no file written.")
        return

    # ----------------------------------------------------------------
    # Sanity check: no NaN or Inf in FP16 tensors
    # (FP8 tensors cannot be checked with isfinite directly)
    # ----------------------------------------------------------------
    issues = [
        k for k, t in sd_out.items()
        if t.dtype == torch.float16 and not torch.isfinite(t).all()
    ]
    if issues:
        print(f"WARNING: {len(issues)} FP16 tensor(s) contain NaN or Inf:")
        for k in issues[:5]:
            print(f"  {k}")

    print(f"Saving: {output_path}")
    save_file(sd_out, str(output_path))
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mixed FP16/FP8 quantization for CLIP-G (OpenCLIP format) safetensors"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input safetensors file (CLIP-G in FP16, OpenCLIP format)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output safetensors file with mixed FP16/FP8 precision "
             "(not required when using --analyze)"
    )
    parser.add_argument(
        "--first-blocks-keep", "-k", type=int, default=DEFAULT_KEEP,
        metavar="N",
        help=f"Number of initial blocks to keep at FP16 (default: {DEFAULT_KEEP}). "
             f"Block {TOTAL_BLOCKS-1} is always kept at FP16."
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Analyze all weight tensors and report per-block quantization sensitivity. "
             "Prints metrics (norm, outliers, roundtrip error) and recommends which "
             "blocks to protect. Does not write any file. Only requires --input."
    )

    # -- Attention split and quantization flags --
    parser.add_argument(
        "--split-attn-qkv", action="store_true",
        help="Split fused in_proj_weight into separate Q/K/V tensors (and biases). "
             "By default K/V are quantized to FP8 in intermediate blocks; Q stays FP16. "
             "Without this flag, in_proj_weight is kept intact in FP16."
    )
    parser.add_argument(
        "--keep-attn-kv-fp16", action="store_true",
        help="When splitting attention (--split-attn-qkv), keep K and V at FP16 "
             "instead of quantizing them to FP8. Useful for diagnosing whether "
             "quality changes come from the split itself or from K/V quantization. "
             "Requires --split-attn-qkv."
    )
    parser.add_argument(
        "--attn-q-fp8", action="store_true",
        help="Quantize the attention Q projection to FP8 in intermediate blocks. "
             "Q is the most sensitive attention component — validate results carefully. "
             "Requires --split-attn-qkv."
    )
    parser.add_argument(
        "--attn-out-fp8", action="store_true",
        help="Quantize attention out_proj.weight to FP8 in intermediate blocks. "
             "Conservative extra savings with low quality risk. "
             "Does not require --split-attn-qkv (out_proj is already a separate tensor)."
    )

    # -- Output control --
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-tensor dtype mapping and detailed outlier checks"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print statistics without writing any file"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # --analyze mode: run analysis and exit
    if args.analyze:
        analyze(input_path)
        return

    # Quantization mode: --output is required
    if args.output is None:
        print("ERROR: --output is required (unless using --analyze).", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)

    if args.first_blocks_keep < 1 or args.first_blocks_keep >= TOTAL_BLOCKS - 1:
        print(
            f"ERROR: --first-blocks-keep must be between 1 and {TOTAL_BLOCKS - 2}",
            file=sys.stderr
        )
        sys.exit(1)

    if args.keep_attn_kv_fp16 and not args.split_attn_qkv:
        print(
            "ERROR: --keep-attn-kv-fp16 requires --split-attn-qkv. "
            "Without the split, K/V don't exist as separate tensors.",
            file=sys.stderr
        )
        sys.exit(1)

    if args.attn_q_fp8 and not args.split_attn_qkv:
        print(
            "ERROR: --attn-q-fp8 requires --split-attn-qkv. "
            "Without the split, Q doesn't exist as a separate tensor.",
            file=sys.stderr
        )
        sys.exit(1)

    convert(input_path, output_path, args.first_blocks_keep, args.dry_run,
            args.split_attn_qkv, args.keep_attn_kv_fp16, args.attn_q_fp8,
            args.attn_out_fp8, args.verbose)


if __name__ == "__main__":
    main()
