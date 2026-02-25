#!/usr/bin/env python3
"""
quantize-clip_g.py
----------------------
Mixed FP16/FP8 (E4M3) quantization for a CLIP-G text encoder extracted
from an SDXL checkpoint in OpenCLIP format (transformer.resblocks.N.*).

Quantization strategy (based on ViT-bigG-14, 32 transformer blocks):

  By default (no structural changes to the state dict):
    - MLP weights (c_fc, c_proj) in intermediate blocks  -> float8_e4m3fn
    - in_proj_weight (fused Q/K/V) in intermediate blocks -> float8_e4m3fn
    - Attention out_proj.weight in intermediate blocks    -> float16 (opt-in FP8 via --attn-out-fp8)
    - First N blocks (--keep-first-blocks) + last block   -> float16 (preserved)
    - All biases                                          -> float16
    - LayerNorm, embeddings, projection, logit_scale      -> float16

  With --split-attn-qkv [mode] (splits in_proj_weight into separate Q/K/V):
    Splits in_proj_weight and in_proj_bias into separate Q/K/V tensors
    with configurable precision per component:
      q16,kv8  - Q FP16, K/V FP8 (default if no mode given)
      q8,kv8   - Q, K, V all FP8
      q8,kv16  - Q FP8, K/V FP16
      q16,kv16 - Q, K, V all FP16 (split only, no quantization)

  Note: --attn-out-fp8 works independently of --split-attn-qkv because
  out_proj.weight is already a separate tensor in the original state dict.

The output safetensors can be loaded directly by ComfyUI-GGUF's
DualCLIPLoaderGGUF node, which respects per-tensor dtypes.

Requirements:
    pip install safetensors torch

Usage:
    # Default: MLP weights + fused in_proj_weight quantized to FP8
    python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors

    # Protect specific blocks using ranges/enumerations
    python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors -k 0-6
    python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors -k 0-3,5-6
    python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors -k 0,1,4,5

    # A single integer n means protect blocks 0..n-1 (legacy behavior)
    python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors -k 7

    # Split attention: Q FP16, K/V FP8 (default split mode)
    python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors --split-attn-qkv

    # Split attention: all Q/K/V to FP8
    python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors --split-attn-qkv q8,kv8

    # Split attention: split only, no quantization of Q/K/V
    python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors --split-attn-qkv q16,kv16

    # Also quantize out_proj (independent of split)
    python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors --attn-out-fp8

    # Maximum quantization: split Q/K/V all FP8 + out_proj FP8
    python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors --split-attn-qkv q8,kv8 --attn-out-fp8

    # Dry run with verbose per-tensor mapping
    python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors --split-attn-qkv --dry-run --verbose

    # Adjust number of preserved initial blocks
    python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors --keep-first-blocks 8

    # Analyze tensors to identify sensitive blocks before quantizing
    python quantize-clip_g.py -i clip_g.safetensors --analyze
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
FP8_ATTN_Q_SUFFIX   = "attn.q_proj.weight"    # --split-attn-qkv q8,*
FP8_IN_PROJ_SUFFIX  = "attn.in_proj_weight"   # default (fused, no split)

# Regex to detect a resblock key and extract block index + suffix
# Matches: transformer.resblocks.{N}.{suffix}
RESBLOCK_RE = re.compile(r"^transformer\.resblocks\.(\d+)\.(.+)$")

# in_proj_weight holds [Q;K;V] concatenated — needs special handling
IN_PROJ_WEIGHT = "attn.in_proj_weight"
IN_PROJ_BIAS   = "attn.in_proj_bias"


# ---------------------------------------------------------------------------
# Block-keep set parsing
# ---------------------------------------------------------------------------

def parse_keep_spec(spec: str) -> set:
    """
    Parse a block-keep specification into a set of block indices to protect.

    Accepts:
      - A single integer n  -> protect blocks 0..n-1  (legacy behavior)
      - A comma-separated list of integers and/or ranges, e.g.:
          0,1,2-4,6   ->  {0, 1, 2, 3, 4, 6}
          0-6         ->  {0, 1, 2, 3, 4, 5, 6}
          0-3,5-6     ->  {0, 1, 2, 3, 5, 6}
          0,1,4,5     ->  {0, 1, 4, 5}

    The last block (TOTAL_BLOCKS - 1) is always protected regardless of spec.
    Raises ValueError with a descriptive message on invalid input.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("Block-keep specification is empty.")

    # Detect legacy single-integer mode: no commas, no dashes (or leading dash for negative)
    # A pure integer string means "protect 0..n-1"
    if re.fullmatch(r'\d+', spec):
        n = int(spec)
        if n < 1 or n > TOTAL_BLOCKS - 1:
            raise ValueError(
                f"Single integer must be between 1 and {TOTAL_BLOCKS - 1} "
                f"(got {n}). It means 'protect blocks 0..n-1'."
            )
        return set(range(n))

    # General range/enumeration parsing
    indices = set()
    parts = spec.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            raise ValueError(f"Empty segment in block-keep specification: '{spec}'")
        range_match = re.fullmatch(r'(\d+)-(\d+)', part)
        if range_match:
            lo, hi = int(range_match.group(1)), int(range_match.group(2))
            if lo > hi:
                raise ValueError(
                    f"Invalid range '{part}': start ({lo}) must be <= end ({hi})."
                )
            if hi >= TOTAL_BLOCKS:
                raise ValueError(
                    f"Block index {hi} in range '{part}' exceeds maximum "
                    f"block index {TOTAL_BLOCKS - 1}."
                )
            indices.update(range(lo, hi + 1))
        elif re.fullmatch(r'\d+', part):
            idx = int(part)
            if idx >= TOTAL_BLOCKS:
                raise ValueError(
                    f"Block index {idx} exceeds maximum block index {TOTAL_BLOCKS - 1}."
                )
            indices.add(idx)
        else:
            raise ValueError(
                f"Cannot parse segment '{part}' in block-keep specification '{spec}'. "
                f"Expected an integer or a range like '2-5'."
            )
    return indices


def is_block_protected(block_idx: int, protected_blocks: set) -> bool:
    """Return True if this block should be kept at FP16."""
    last_block = TOTAL_BLOCKS - 1
    return block_idx == last_block or block_idx in protected_blocks


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


def build_fp8_suffixes(split_mode: str, attn_out_fp8: bool) -> set:
    """
    Return the full set of weight suffixes to quantize to FP8 based on
    active flags.

    Base set: MLP weights (always included).
    Conditionally added:
      - in_proj_weight:  when split_mode is None (fused, always FP8 by default)
      - K/V projections: when split_mode includes kv8
      - Q projection:    when split_mode includes q8
      - out_proj:        when --attn-out-fp8 is set (independent of split)
    """
    suffixes = set(FP8_MLP_SUFFIXES)
    if split_mode is None:
        # No split: fused in_proj_weight goes to FP8 by default
        suffixes.add(FP8_IN_PROJ_SUFFIX)
    else:
        # Split active: check per-component precision from mode string
        if "kv8" in split_mode:
            suffixes.update(FP8_ATTN_KV_SUFFIXES)
        if split_mode.startswith("q8"):
            suffixes.add(FP8_ATTN_Q_SUFFIX)
    if attn_out_fp8:
        suffixes.add(FP8_ATTN_OUT_SUFFIX)
    return suffixes


def is_fp8_candidate(block_idx: int, suffix: str, protected_blocks: set, fp8_suffixes: set) -> bool:
    """
    Return True if this (block_idx, suffix) pair should be quantized to FP8.
    Conditions:
      - Block is not in the protected set (which always includes the last block)
      - Suffix is in the FP8 candidate set
    """
    if is_block_protected(block_idx, protected_blocks):
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
# Covers both fused format (attn.in_proj_weight) and pre-split format
# (attn.{q,k,v}_proj.weight) so that files produced with --split-attn-qkv
# are analyzed correctly.
ANALYSIS_WEIGHT_SUFFIXES = {
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
    "attn.in_proj_weight",    # fused Q/K/V  (original / protected-block format)
    "attn.q_proj.weight",     # split Q      (produced by --split-attn-qkv)
    "attn.k_proj.weight",     # split K
    "attn.v_proj.weight",     # split V
    "attn.out_proj.weight",
}

# Suffixes that represent the already-split Q/K/V projections
SPLIT_QKV_SUFFIXES = {
    "attn.q_proj.weight",
    "attn.k_proj.weight",
    "attn.v_proj.weight",
}

# Friendly display names for components
COMPONENT_DISPLAY_NAMES = {
    "mlp.c_fc.weight":       "MLP.c_fc",
    "mlp.c_proj.weight":     "MLP.c_proj",
    "attn.out_proj.weight":  "attn.out_proj",
}

# Display names for the in_proj children (shown indented under attn.in_proj_weight)
IN_PROJ_CHILD_NAMES = {
    "attn.q_proj.weight": "attn.q_proj",
    "attn.k_proj.weight": "attn.k_proj",
    "attn.v_proj.weight": "attn.v_proj",
}

# QError thresholds for per-component warnings (as fractions, not percentages)
WARN_THRESHOLD_OUT_PROJ = 0.08   # 8%: elevated risk for out_proj
WARN_THRESHOLD_HIGH     = 0.12   # 12%: high risk, strongly discouraged


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
    block_data    = {}
    # Also keep a raw-suffix -> metrics mapping for flag-specific analysis
    block_raw     = {}   # block_raw[block_idx][suffix] = metrics dict

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
            block_raw[block_idx]  = {}

        if suffix == IN_PROJ_WEIGHT:
            # Fused format: analyze the whole tensor and derive Q/K/V by splitting
            block_data[block_idx]["attn.in_proj_weight"] = analyze_tensor(tensor)
            q, k, v = split_in_proj(tensor)
            block_data[block_idx]["attn.q_proj.weight"] = analyze_tensor(q)
            block_data[block_idx]["attn.k_proj.weight"] = analyze_tensor(k)
            block_data[block_idx]["attn.v_proj.weight"] = analyze_tensor(v)
            block_raw[block_idx]["attn.q_proj.weight"] = block_data[block_idx]["attn.q_proj.weight"]
            block_raw[block_idx]["attn.k_proj.weight"] = block_data[block_idx]["attn.k_proj.weight"]
            block_raw[block_idx]["attn.v_proj.weight"] = block_data[block_idx]["attn.v_proj.weight"]
            block_data[block_idx]["_in_proj_source"] = "fused"
        elif suffix in SPLIT_QKV_SUFFIXES:
            # Pre-split format: Q/K/V arrive as separate tensors.
            metrics = analyze_tensor(tensor)
            block_data[block_idx][suffix] = metrics
            block_raw[block_idx][suffix]  = metrics
            # Accumulate tensors to build a synthetic fused summary row once all three are seen.
            staging = block_data[block_idx].setdefault("_qkv_staging", {})
            staging[suffix] = tensor
            if len(staging) == 3:
                fused = torch.cat([
                    staging["attn.q_proj.weight"],
                    staging["attn.k_proj.weight"],
                    staging["attn.v_proj.weight"],
                ], dim=0)
                block_data[block_idx]["attn.in_proj_weight"] = analyze_tensor(fused)
                block_data[block_idx]["_in_proj_source"] = "split"
        else:
            name = COMPONENT_DISPLAY_NAMES.get(suffix, suffix)
            metrics = analyze_tensor(tensor)
            block_data[block_idx][name] = metrics
            block_raw[block_idx][suffix] = metrics

    if not block_data:
        print("No resblock weight tensors found in the file.")
        return

    # Remove internal staging keys used during ingestion
    for bd in block_data.values():
        bd.pop("_qkv_staging", None)
        bd.pop("_in_proj_source", None)

    # --- Per-block detailed report ---
    # Column widths: component column needs extra space for indented children
    COL_COMP = 22
    print("=" * 94)
    print(f"{'Block':>5} {'Component':<{COL_COMP}} {'Params':>10} {'Norm':>10} "
          f"{'Max|W|':>10} {'Std':>10} {'Outliers':>10} {'QError%':>10}")
    print("-" * 94)

    # Ordered rendering: attn.in_proj_weight first, then its children, then the rest
    IN_PROJ_KEY  = "attn.in_proj_weight"
    IN_PROJ_CHILDREN = ["attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight"]
    RENDER_ORDER = [IN_PROJ_KEY] + IN_PROJ_CHILDREN + ["attn.out_proj", "MLP.c_fc", "MLP.c_proj"]

    # Collect per-block aggregate quantization error for ranking
    block_agg_error = {}

    for block_idx in sorted(block_data.keys()):
        components = block_data[block_idx]
        weighted_error_sum = 0.0
        total_params = 0

        for comp_key, m in components.items():
            # Skip the fused in_proj from the aggregate since we already
            # count Q/K/V separately
            if comp_key == IN_PROJ_KEY:
                continue
            weighted_error_sum += m["quant_error"] * m["params"]
            total_params += m["params"]

        block_agg_error[block_idx] = weighted_error_sum / total_params if total_params > 0 else 0.0

        # Build ordered list of (display_name, metrics, is_child)
        rows = []
        for comp_key, m in components.items():
            if comp_key == IN_PROJ_KEY:
                rows.append(("attn.in_proj_weight", m, False, True))  # (name, metrics, is_child, is_parent)
            elif comp_key in IN_PROJ_CHILDREN:
                display = IN_PROJ_CHILD_NAMES[comp_key]
                rows.append((display, m, True, False))
            else:
                rows.append((comp_key, m, False, False))

        # Sort by a canonical order
        order_map = {
            "attn.in_proj_weight": 0,
            "attn.q_proj": 1, "attn.k_proj": 2, "attn.v_proj": 3,
            "attn.out_proj": 4, "MLP.c_fc": 5, "MLP.c_proj": 6,
        }
        rows.sort(key=lambda r: order_map.get(r[0], 99))

        for i, (disp_name, m, is_child, is_parent) in enumerate(rows):
            outlier_str = (f"{m['outliers']}" if m["outliers"] == 0
                           else f"{m['outliers']} ({m['outlier_pct']:.3f}%)")
            if is_child:
                # Determine tree character: last child gets └─, others get ├─
                child_rows = [r for r in rows if r[2]]  # all children
                is_last = (disp_name == child_rows[-1][0])
                tree = "  └─ " if is_last else "  ├─ "
                comp_col = f"{tree}{disp_name}"
            else:
                comp_col = disp_name

            print(f"{block_idx:>5} {comp_col:<{COL_COMP}} {m['params']:>10,} {m['norm']:>10.2f} "
                  f"{m['max_abs']:>10.4f} {m['std']:>10.6f} {outlier_str:>10} "
                  f"{m['quant_error']*100:>9.4f}%")
        print()

    # --- Block ranking by quantization sensitivity ---
    # Compute mean and standard deviation for threshold
    errors = list(block_agg_error.values())
    mean_error = sum(errors) / len(errors)
    std_error  = (sum((e - mean_error) ** 2 for e in errors) / len(errors)) ** 0.5

    # Blocks with error > mean + 1 std are flagged as sensitive
    threshold = mean_error + std_error

    sensitive_blocks = sorted(
        b for b, e in block_agg_error.items() if e > threshold
    )

    print("=" * 94)
    print("Block sensitivity ranking (weighted average quantization error) — ordered by block index")
    print("-" * 94)

    # Display rows ordered by block index (ascending)
    for block_idx in sorted(block_agg_error.keys()):
        err  = block_agg_error[block_idx]
        flag = " << SENSITIVE" if err > threshold else ""
        print(f"  Block {block_idx:>2}: {err*100:.4f}%{flag}")

    print()
    print(f"  Mean quantization error : {mean_error*100:.4f}%")
    print(f"  Std deviation           : {std_error*100:.4f}%")
    print(f"  Sensitivity threshold   : {threshold*100:.4f}% (mean + 1 std)")
    print()

    # --- General recommendation ---
    if sensitive_blocks:
        block_list = ", ".join(str(b) for b in sensitive_blocks)
        print(f"  Recommendation: consider protecting blocks {block_list}")
        print(f"  These blocks show significantly higher quantization error than average.")

        # Recommend the minimum --keep-first-blocks that covers all sensitive blocks.
        # If all sensitive blocks are below TOTAL_BLOCKS - 1 (last block is always protected),
        # suggest a contiguous range 0..max_sensitive to keep the recommendation simple.
        non_last_sensitive = [b for b in sensitive_blocks if b < TOTAL_BLOCKS - 1]
        if non_last_sensitive:
            suggested_keep = max(non_last_sensitive) + 1
            # Check if the sensitive set forms a contiguous prefix from 0
            is_prefix = all(b < suggested_keep for b in non_last_sensitive) and \
                        len(non_last_sensitive) == suggested_keep
            if is_prefix:
                print(f"  Suggested --keep-first-blocks: {suggested_keep} "
                      f"(covers sensitive blocks 0-{suggested_keep - 1})")
            else:
                # Sensitive blocks are not a contiguous prefix: recommend the covering
                # contiguous range and also show the exact set for -k enumeration.
                print(f"  Suggested --keep-first-blocks: {suggested_keep} "
                      f"(contiguous range 0-{suggested_keep - 1} covering all sensitive blocks)")
                exact_spec = ",".join(str(b) for b in non_last_sensitive)
                print(f"  Alternatively, protect only sensitive blocks with: -k {exact_spec}")

        # Note about sensitive blocks near the end
        last_block = TOTAL_BLOCKS - 1
        sensitive_tail = [b for b in sensitive_blocks if b >= last_block - 3 and b != last_block]
        if sensitive_tail:
            tail_list = ", ".join(str(b) for b in sensitive_tail)
            print(f"  Note: blocks {tail_list} near the end are also sensitive "
                  f"(block {last_block} is always protected).")
    else:
        print("  No blocks show significantly elevated quantization error.")
        print("  The default --keep-first-blocks value should be adequate.")

    print()

    # --- --split-attn-qkv K/V quantization impact analysis (shown first) ---
    _analyze_attn_kv_fp8(block_raw, sensitive_blocks)

    # --- --attn-out-fp8 impact analysis (shown second) ---
    _analyze_attn_out_fp8(block_raw, sensitive_blocks)

    print()


def _analyze_attn_out_fp8(block_raw: dict, sensitive_blocks: list):
    """
    Report the per-block impact of enabling --attn-out-fp8, highlighting
    blocks where out_proj quantization error is elevated or high.
    """
    last_block = TOTAL_BLOCKS - 1

    # Collect out_proj errors for all intermediate blocks
    out_proj_errors = {}
    for block_idx in sorted(block_raw.keys()):
        if block_idx == last_block:
            continue
        suffix_data = block_raw[block_idx]
        if "attn.out_proj.weight" in suffix_data:
            out_proj_errors[block_idx] = suffix_data["attn.out_proj.weight"]["quant_error"]

    if not out_proj_errors:
        return

    elevated = {b: e for b, e in out_proj_errors.items() if e >= WARN_THRESHOLD_OUT_PROJ}
    high     = {b: e for b, e in out_proj_errors.items() if e >= WARN_THRESHOLD_HIGH}

    print("  attn.out_proj quantization impact analysis")
    print("  " + "-" * 60)

    if not elevated:
        print("  All intermediate blocks show acceptable out_proj QError (<8%).")
        print("  --attn-out-fp8 is safe to use with any --keep-first-blocks value.")
        print()
        return

    # Show per-block out_proj error table for intermediate blocks
    print(f"  {'Block':>5}  {'out_proj QError%':>16}  {'Risk':>12}  {'Note'}")
    print(f"  {'-'*5}  {'-'*16}  {'-'*12}  {'-'*30}")
    for block_idx in sorted(out_proj_errors.keys()):
        err = out_proj_errors[block_idx]
        pct = err * 100
        if err >= WARN_THRESHOLD_HIGH:
            risk = "HIGH"
            note = "strongly discouraged"
        elif err >= WARN_THRESHOLD_OUT_PROJ:
            risk = "ELEVATED"
            note = "review carefully"
        else:
            risk = "ok"
            note = ""
        marker = " <<" if err >= WARN_THRESHOLD_OUT_PROJ else ""
        print(f"  {block_idx:>5}  {pct:>15.4f}%  {risk:>12}  {note}{marker}")

    print()

    # Determine which elevated blocks would be exposed for different keep values
    # Find the minimum --keep-first-blocks that covers all high-risk out_proj blocks
    high_blocks   = sorted(high.keys())
    elev_blocks   = sorted(elevated.keys())

    if high_blocks:
        min_keep_high = max(high_blocks) + 1
        print(f"  WARNING: {len(high_blocks)} block(s) with out_proj QError ≥ 12%: "
              f"{', '.join(str(b) for b in high_blocks)}")
        print(f"           --attn-out-fp8 is strongly discouraged unless "
              f"--keep-first-blocks >= {min_keep_high}.")

    if elev_blocks:
        min_keep_elev = max(elev_blocks) + 1
        print(f"  CAUTION:  {len(elev_blocks)} block(s) with out_proj QError ≥ 8%: "
              f"{', '.join(str(b) for b in elev_blocks)}")
        print(f"           To safely use --attn-out-fp8, set --keep-first-blocks >= {min_keep_elev}.")

    # Cross-reference with general --keep-first-blocks recommendation
    if sensitive_blocks:
        non_last_sensitive = [b for b in sensitive_blocks if b < TOTAL_BLOCKS - 1]
        if non_last_sensitive:
            effective_keep = max(non_last_sensitive) + 1
            remaining_elev = [b for b in elev_blocks if b >= effective_keep]
            if remaining_elev:
                print(f"  With the suggested --keep-first-blocks {effective_keep}, blocks "
                      f"{', '.join(str(b) for b in remaining_elev)} would still be exposed "
                      f"to out_proj quantization.")
                print(f"  Consider omitting --attn-out-fp8 or raising --keep-first-blocks "
                      f"to {max(remaining_elev) + 1}.")

    print()


def _analyze_attn_kv_fp8(block_raw: dict, sensitive_blocks: list):
    """
    Report the per-block impact of enabling --split-attn-qkv with Q/K/V quantization,
    highlighting blocks where Q, K or V projection quantization error is elevated.
    """
    last_block = TOTAL_BLOCKS - 1

    WARN_KV = 0.06  # 6%: K/V are somewhat more sensitive than MLP

    qkv_errors = {}
    for block_idx in sorted(block_raw.keys()):
        if block_idx == last_block:
            continue
        suffix_data = block_raw[block_idx]
        q_err = suffix_data.get("attn.q_proj.weight", {}).get("quant_error")
        k_err = suffix_data.get("attn.k_proj.weight", {}).get("quant_error")
        v_err = suffix_data.get("attn.v_proj.weight", {}).get("quant_error")
        if q_err is not None and k_err is not None and v_err is not None:
            qkv_errors[block_idx] = {
                "q": q_err,
                "k": k_err,
                "v": v_err,
                "max": max(q_err, k_err, v_err),
            }

    if not qkv_errors:
        return

    elevated_qkv = {b: d for b, d in qkv_errors.items() if d["max"] >= WARN_KV}

    print("  attn.in_proj split quantization impact analysis (--split-attn-qkv)")
    print("  " + "-" * 60)

    if not elevated_qkv:
        print("  All intermediate blocks show acceptable Q/K/V QError (<6%).")
        print("  --split-attn-qkv with Q/K/V FP8 is safe across intermediate blocks.")
        print()
        return

    print(f"  {'Block':>5}  {'Q QError%':>10}  {'K QError%':>10}  {'V QError%':>10}  {'Risk':>10}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for block_idx in sorted(qkv_errors.keys()):
        d = qkv_errors[block_idx]
        risk = "ELEVATED" if d["max"] >= WARN_KV else "ok"
        marker = " <<" if d["max"] >= WARN_KV else ""
        print(f"  {block_idx:>5}  {d['q']*100:>9.4f}%  {d['k']*100:>9.4f}%  {d['v']*100:>9.4f}%  {risk:>10}{marker}")

    print()
    elev_blocks = sorted(elevated_qkv.keys())
    min_keep_kv = max(elev_blocks) + 1
    print(f"  CAUTION: {len(elev_blocks)} block(s) with elevated Q/K/V QError (≥6%): "
          f"{', '.join(str(b) for b in elev_blocks)}")
    print(f"           To safely use Q/K/V FP8, set --keep-first-blocks >= {min_keep_kv},")
    print(f"           or use --split-attn-qkv q16,kv16 to keep Q/K/V at FP16 after the split.")
    print()


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(input_path: Path, output_path: Path, protected_blocks: set, dry_run: bool,
            split_mode: str, attn_out_fp8: bool, verbose: bool):
    if not check_fp8_support():
        print("ERROR: float8_e4m3fn is not available in this PyTorch build.")
        print("       Requires PyTorch >= 2.1 compiled with FP8 support.")
        sys.exit(1)

    split_attn = split_mode is not None
    attn_q_fp8 = split_attn and split_mode.startswith("q8")
    keep_kv_fp16 = split_attn and "kv16" in split_mode

    fp8_suffixes = build_fp8_suffixes(split_mode, attn_out_fp8)

    last_block = TOTAL_BLOCKS - 1

    # Build a human-readable description of protected blocks for the summary
    protected_non_last = sorted(b for b in protected_blocks if b != last_block)
    if protected_non_last:
        # Check if it's a contiguous prefix
        if protected_non_last == list(range(len(protected_non_last))):
            protected_desc = f"blocks 0-{protected_non_last[-1]} and block {last_block}"
        else:
            protected_desc = f"blocks {','.join(str(b) for b in protected_non_last)} and block {last_block}"
    else:
        protected_desc = f"block {last_block} only"

    print(f"Loading: {input_path}")
    sd_in = load_file(str(input_path))
    print(f"  Tensors loaded : {len(sd_in)}")
    print(f"  FP16 preserved : {protected_desc}")

    if not split_attn:
        fp8_desc = "MLP weights + attn.in_proj_weight (fused)"
        if attn_out_fp8:
            fp8_desc += " + attn out_proj"
        print(f"  FP8 candidates : unprotected blocks ({fp8_desc})")
        print(f"  in_proj_weight : fused FP8 (default)")
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
        print(f"  FP8 candidates : unprotected blocks ({fp8_desc})")
        print(f"  Split attn     : enabled (in_proj_weight -> Q/K/V, mode: {split_mode})")
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
                if not is_block_protected(block_idx, protected_blocks):
                    # Default: quantize fused tensor directly to FP8
                    check_outliers(tensor, key, verbose)
                    sd_out[key] = to_fp8(tensor)
                    log_tensor(key, tensor.dtype, torch.float8_e4m3fn, verbose)
                    stats["fp8"] += 1
                else:
                    # Protected block: keep the fused tensor at FP16
                    sd_out[key] = tensor.to(torch.float16)
                    log_tensor(key, tensor.dtype, torch.float16, verbose)
                    stats["fp16"] += 1
                continue

            # --split-attn-qkv: split Q/K/V and apply per-component strategy
            q, k, v = split_in_proj(tensor)

            base = f"transformer.resblocks.{block_idx}"

            if not is_block_protected(block_idx, protected_blocks):
                # Q: FP8 only if split mode starts with q8, otherwise FP16
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

                # K and V: FP8 by default, FP16 if split mode includes kv16
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
            if not is_block_protected(block_idx, protected_blocks):
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
        if is_fp8_candidate(block_idx, suffix, protected_blocks, fp8_suffixes):
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
    # Valid modes for --split-attn-qkv
    SPLIT_MODES = ["q16,kv8", "q8,kv8", "q8,kv16", "q16,kv16"]
    DEFAULT_SPLIT_MODE = "q16,kv8"

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
        "--keep-first-blocks", "-k", type=str, default=str(DEFAULT_KEEP),
        metavar="SPEC",
        help=(
            f"Blocks to keep at FP16. Accepts a single integer n (protect blocks 0..n-1), "
            f"or a comma-separated list of indices and/or ranges, e.g.: "
            f"7  |  0-6  |  0-3,5-6  |  0,1,4,5  |  0,1,2-4,6. "
            f"Block {TOTAL_BLOCKS-1} is always kept at FP16 regardless of this setting. "
            f"(default: {DEFAULT_KEEP}, i.e. blocks 0-{DEFAULT_KEEP-1})"
        )
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Analyze all weight tensors and report per-block quantization sensitivity. "
             "Prints metrics (norm, outliers, roundtrip error) and recommends which "
             "blocks to protect. Does not write any file. Only requires --input."
    )

    # -- Attention split and quantization flags --
    parser.add_argument(
        "--split-attn-qkv", nargs="?", const=DEFAULT_SPLIT_MODE, default=None,
        choices=SPLIT_MODES, metavar="MODE",
        help="Split fused in_proj_weight into separate Q/K/V tensors (and biases), "
             "with precision control per component. "
             "Without this flag, in_proj_weight is quantized to FP8 as a fused tensor. "
             "Options: "
             f"{DEFAULT_SPLIT_MODE} - Q FP16, K/V FP8 (default if no value given); "
             "q8,kv8 - Q, K, V all FP8; "
             "q8,kv16 - Q FP8, K/V FP16; "
             "q16,kv16 - Q, K, V all FP16 (split only, no quantization)"
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

    # Parse block-keep specification
    try:
        protected_blocks = parse_keep_spec(args.keep_first_blocks)
    except ValueError as e:
        print(f"ERROR: Invalid --keep-first-blocks value: {e}", file=sys.stderr)
        sys.exit(1)

    # Always protect the last block
    protected_blocks.add(TOTAL_BLOCKS - 1)

    # Inform the user about the split mode when using the default
    split_mode = args.split_attn_qkv
    if split_mode is not None:
        idx = None
        for i, arg in enumerate(sys.argv):
            if arg == "--split-attn-qkv":
                idx = i
                break
        explicit = (idx is not None
                    and idx + 1 < len(sys.argv)
                    and sys.argv[idx + 1] in SPLIT_MODES)
        if not explicit and split_mode == DEFAULT_SPLIT_MODE:
            print(f"NOTE: --split-attn-qkv used without explicit mode. "
                  f"Defaulting to {DEFAULT_SPLIT_MODE} (Q FP16, K/V FP8).")
            print()

    convert(input_path, output_path, protected_blocks, args.dry_run,
            split_mode, args.attn_out_fp8, args.verbose)


if __name__ == "__main__":
    main()
