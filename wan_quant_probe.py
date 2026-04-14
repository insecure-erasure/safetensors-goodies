#!/usr/bin/env python3
"""
Quantization Sensitivity Analysis for safetensors models (Wan architecture).

Analyzes weight tensors of transformer blocks to estimate sensitivity to
quantization, helping decide which layers should be kept in BF16 (*KEEP*),
quantized to FP8, or quantized to NVFP4.

A hard floor on excess kurtosis (--kurtosis-keep, default 8.0) forces *KEEP*
regardless of the percentile-based score when a tensor's distribution is
extremely leptokurtic. This protects layers that the IQR clipping in norm_iqr()
would otherwise under-penalise.

Targets the following layer types across all transformer blocks:
  - cross_attn: k, v, q, o, k_img, v_img
  - self_attn:  k, v, q, o
  - ffn:        0, 2

Metrics computed per tensor (on GPU when available):
  - Excess kurtosis:  indicates heavy-tailed distributions (0 = normal); corrected
  - Dynamic range:    abs(max) - abs(min), within-type min-max normalized for scoring
  - Std deviation:    overall weight dispersion
  - Outliers %:       fraction of values exceeding N standard deviations
  - Aspect ratio:     max(rows, cols) / min(rows, cols), within-type normalized

Recommendation score is a weighted combination of three components:
  - Excess kurtosis (IQR-normalized globally)
  - Dynamic range (within-type min-max normalized)
  - Aspect ratio (within-type min-max normalized; inactive when all tensors
    of the same type share the same shape, which is the norm in transformer
    architectures — effective score budget is then 0.9 instead of 1.0)

Outlier% is computed and displayed in tables for reference but does not
contribute to the score, as it is strongly correlated with excess kurtosis.

Thresholds are derived automatically from the model's own score distribution
using configurable percentiles.

Extreme block ranges are derived automatically from the total number of blocks
detected in the model, using a configurable percentage (default: 10%).

Usage:
    python analyze_quant_sensitivity.py model.safetensors
    python analyze_quant_sensitivity.py model.safetensors --csv results.csv
    python analyze_quant_sensitivity.py model.safetensors --device cpu
    python analyze_quant_sensitivity.py model.safetensors --outlier-sigma 4
    python analyze_quant_sensitivity.py model.safetensors --fp8-percentile 70 --keep-percentile 90
    python analyze_quant_sensitivity.py model.safetensors --extreme-pct 15
    python analyze_quant_sensitivity.py model.safetensors --kurtosis-weight 0.5 --range-weight 0.4 --ar-weight 0.1
    python analyze_quant_sensitivity.py model.safetensors --lowram

--lowram mode reads the safetensors header once and then loads each tensor
individually by byte offset, avoiding the full-file mmap that safe_open
performs. Use this on systems with limited RAM (e.g. 16 GB without swap)
where mmap-ing a 28 GB model file would exhaust virtual memory.

Compatibility with convert_to_quant protected layers:
convert_to_quant protects two sets of layers by design:
  - Excluded by partial name match: norm, bias, img_emb.proj.*,
    patch_embedding, k_norm, q_norm
  - Kept in high precision by exact name: text_embedding.*, time_embedding.*,
    time_projection.*, head.head, head.modulation

LAYER_PATTERNS targets only blocks.<idx>.(cross_attn|self_attn|ffn).*.weight,
which does not intersect with either protected set. No explicit protection
mechanism is needed: the scope of analysis is disjoint from what
convert_to_quant protects by construction.
"""

import argparse
import csv
import json
import re
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors is required. Install with: pip install safetensors")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Layer filter configuration
# ---------------------------------------------------------------------------

# Regex patterns for layers to analyze, keyed by display group name.
# Each pattern captures the block index in group 1.
#
# Scope is deliberately limited to transformer block weights of the form
# blocks.<idx>.(cross_attn|self_attn|ffn).*.weight. This does not intersect
# with the layers that convert_to_quant protects by design:
#   - Excluded by partial name match: norm, bias, img_emb.proj.*,
#     patch_embedding, k_norm, q_norm
#   - Kept in high precision by exact name: text_embedding.*, time_embedding.*,
#     time_projection.*, head.head, head.modulation
# No explicit guard is needed: the disjointness is a consequence of the
# patterns themselves.
LAYER_PATTERNS = {
    "cross_attn.k":     re.compile(r"blocks\.(\d+)\.cross_attn\.k\.weight$"),
    "cross_attn.v":     re.compile(r"blocks\.(\d+)\.cross_attn\.v\.weight$"),
    "cross_attn.q":     re.compile(r"blocks\.(\d+)\.cross_attn\.q\.weight$"),
    "cross_attn.o":     re.compile(r"blocks\.(\d+)\.cross_attn\.o\.weight$"),
    "cross_attn.k_img": re.compile(r"blocks\.(\d+)\.cross_attn\.k_img\.weight$"),
    "cross_attn.v_img": re.compile(r"blocks\.(\d+)\.cross_attn\.v_img\.weight$"),
    "self_attn.k":      re.compile(r"blocks\.(\d+)\.self_attn\.k\.weight$"),
    "self_attn.v":      re.compile(r"blocks\.(\d+)\.self_attn\.v\.weight$"),
    "self_attn.q":      re.compile(r"blocks\.(\d+)\.self_attn\.q\.weight$"),
    "self_attn.o":      re.compile(r"blocks\.(\d+)\.self_attn\.o\.weight$"),
    "ffn.0":            re.compile(r"blocks\.(\d+)\.ffn\.0\.weight$"),
    "ffn.2":            re.compile(r"blocks\.(\d+)\.ffn\.2\.weight$"),
}

# Detail sections: which layer groups to break down by block position
DETAIL_GROUPS = {
    "cross_attn": ["cross_attn.k", "cross_attn.v", "cross_attn.q", "cross_attn.o",
                   "cross_attn.k_img", "cross_attn.v_img"],
    "self_attn":  ["self_attn.k", "self_attn.v", "self_attn.q", "self_attn.o"],
    "ffn":        ["ffn.0", "ffn.2"],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TensorMetrics:
    """Metrics for a single weight tensor."""
    key: str                   # full safetensors key
    layer_type: str            # e.g. "cross_attn.k"
    block_idx: int             # transformer block index
    shape: Tuple[int, ...]     # tensor shape
    excess_kurtosis: float     # kurtosis - 3; 0 for normal distribution
    dynamic_range: float       # abs(max) - abs(min)
    std: float                 # standard deviation
    outlier_pct: float         # % of values beyond outlier_sigma * std
    aspect_ratio: float        # max(rows, cols) / min(rows, cols)
    score: float = 0.0         # combined sensitivity score (filled after normalization)
    recommendation: str = ""   # *KEEP*, FP8, or NVFP4 (filled after thresholds)
    reason: str = ""           # decision reason (filled alongside recommendation)


@dataclass
class AggregatedMetrics:
    """Mean metrics for a group of tensors."""
    layer_type: str
    block_range: str           # e.g. "0–39" or "0–3"
    count: int
    excess_kurtosis: float
    kurtosis_max: float        # max excess kurtosis within the group; flags outlier tensors
    dynamic_range: float
    std: float
    outlier_pct: float
    aspect_ratio: float
    score: float = 0.0
    recommendation: str = ""
    reason: str = ""           # decision reason (mirrors TensorMetrics.reason for aggregates)
    spread_filtered: bool = False  # True when recommendation was changed by the spread filter
                                   # (FP8 → NVFP4, or *KEEP* → FP8)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(tensor: torch.Tensor, outlier_sigma: float) -> Dict[str, float]:
    """
    Compute quantization sensitivity metrics for a 2D weight tensor.

    Excess kurtosis is used (kurtosis - 3), so a normal distribution gives 0.
    Positive values indicate heavier tails than normal (more outlier-prone).
    Negative values indicate lighter tails (platykurtic distributions).

    Args:
        tensor:        2D float tensor (already on target device)
        outlier_sigma: standard deviation multiplier for outlier detection

    Returns:
        Dictionary with keys: excess_kurtosis, dynamic_range, std,
                               outlier_pct, aspect_ratio
    """
    t = tensor.to(torch.float32).flatten()

    mean = t.mean()
    std  = t.std()

    # Excess kurtosis: subtract 3 so normal distribution gives 0
    if std > 0:
        excess_kurtosis = ((t - mean) ** 4).mean() / (std ** 4) - 3.0
    else:
        excess_kurtosis = torch.tensor(0.0)

    # Dynamic range: difference between max and min absolute values
    dynamic_range = t.abs().max() - t.abs().min()

    # Outlier percentage: fraction of values beyond outlier_sigma standard deviations
    if std > 0:
        outlier_mask = (t - mean).abs() > outlier_sigma * std
        outlier_pct  = outlier_mask.float().mean() * 100.0
    else:
        outlier_pct = torch.tensor(0.0)

    # Aspect ratio
    rows, cols = tensor.shape
    aspect_ratio = max(rows, cols) / min(rows, cols)

    return {
        "excess_kurtosis": excess_kurtosis.item(),
        "dynamic_range":   dynamic_range.item(),
        "std":             std.item(),
        "outlier_pct":     outlier_pct.item(),
        "aspect_ratio":    aspect_ratio,
    }


# ---------------------------------------------------------------------------
# Score computation (combined sensitivity score)
# ---------------------------------------------------------------------------

def compute_scores(
    all_metrics:     List[TensorMetrics],
    kurtosis_weight: float,
    range_weight:    float,
    ar_weight:       float,
) -> None:
    """
    Compute a combined sensitivity score for each tensor and store it in-place.

    Excess kurtosis is normalized globally using IQR (interquartile range),
    which is more robust than min-max against extreme outlier tensors.

    Dynamic range and aspect ratio are normalized within each layer type
    using min-max, so they are only compared against tensors of the same kind.

    Args:
        all_metrics:      list of TensorMetrics to score (modified in-place)
        kurtosis_weight:  weight for IQR-normalized excess kurtosis
        range_weight:     weight for within-type min-max normalized dynamic range
        ar_weight:        weight for within-type normalized aspect ratio
    """
    # --- IQR normalization for excess kurtosis (global, robust to outliers) ---
    kurt_vals = torch.tensor([m.excess_kurtosis for m in all_metrics])
    q25 = torch.quantile(kurt_vals, 0.25).item()
    q75 = torch.quantile(kurt_vals, 0.75).item()
    iqr = q75 - q25

    def norm_iqr(val: float) -> float:
        # Clip to [q25 - 1.5*IQR, q75 + 1.5*IQR] then normalize to [0, 1]
        lo = q25 - 1.5 * iqr
        hi = q75 + 1.5 * iqr
        clipped = max(lo, min(hi, val))
        return (clipped - lo) / (hi - lo) if hi > lo else 0.0

    # --- Within-type min-max normalization for dynamic range and aspect ratio ---
    by_type: Dict[str, List[TensorMetrics]] = defaultdict(list)
    for m in all_metrics:
        by_type[m.layer_type].append(m)

    type_range_bounds: Dict[str, Tuple[float, float]] = {}
    type_ar_bounds:    Dict[str, Tuple[float, float]] = {}

    for layer_type, group in by_type.items():
        dr_vals = [m.dynamic_range for m in group]
        ar_vals = [m.aspect_ratio  for m in group]
        type_range_bounds[layer_type] = (min(dr_vals), max(dr_vals))
        type_ar_bounds[layer_type]    = (min(ar_vals), max(ar_vals))

    # --- Assign scores ---
    for m in all_metrics:
        n_kurt = norm_iqr(m.excess_kurtosis)

        dr_lo, dr_hi = type_range_bounds[m.layer_type]
        n_range = (m.dynamic_range - dr_lo) / (dr_hi - dr_lo) if dr_hi > dr_lo else 0.0

        ar_lo, ar_hi = type_ar_bounds[m.layer_type]
        n_ar = (m.aspect_ratio - ar_lo) / (ar_hi - ar_lo) if ar_hi > ar_lo else 0.0

        m.score = (
            kurtosis_weight * n_kurt +
            range_weight    * n_range +
            ar_weight       * n_ar
        )


# ---------------------------------------------------------------------------
# Threshold and recommendation logic
# ---------------------------------------------------------------------------

def compute_auto_thresholds(
    all_metrics:     List[TensorMetrics],
    fp8_percentile:  float,
    keep_percentile: float,
) -> Tuple[float, float]:
    """
    Derive FP8 and *KEEP* score thresholds from the model's own score distribution.

    Args:
        all_metrics:     list of TensorMetrics with scores already computed
        fp8_percentile:  score percentile above which FP8 is recommended
        keep_percentile: score percentile above which *KEEP* is recommended

    Returns:
        Tuple of (fp8_threshold, keep_threshold) as score values
    """
    scores = torch.tensor([m.score for m in all_metrics])
    fp8_threshold  = torch.quantile(scores, fp8_percentile  / 100.0).item()
    keep_threshold = torch.quantile(scores, keep_percentile / 100.0).item()
    return fp8_threshold, keep_threshold


def assign_recommendation(
    score:           float,
    fp8_threshold:   float,
    keep_threshold:  float,
    fp8_min_score:   float = 0.0,
    excess_kurtosis: float = 0.0,
    kurtosis_keep:   float = float('inf'),
) -> Tuple[str, str]:
    """
    Return (*KEEP*|FP8|NVFP4, reason) based on score thresholds.

    FP8 requires both the percentile threshold and the absolute minimum score
    to be met. This prevents percentile-based thresholds from recommending FP8
    when all scores are clustered in a narrow range and no layer is genuinely
    sensitive.

    A hard floor on excess kurtosis (kurtosis_keep) forces *KEEP* when a
    tensor's distribution is extremely leptokurtic, regardless of its
    percentile-based score. This protects layers that norm_iqr() under-penalises
    because IQR clipping saturates the kurtosis contribution before the score
    can reach the keep_threshold.

    Args:
        score:           combined sensitivity score for the tensor or aggregate
        fp8_threshold:   percentile-derived score threshold for FP8
        keep_threshold:  percentile-derived score threshold for *KEEP*
        fp8_min_score:   absolute minimum score required for any FP8 recommendation
        excess_kurtosis: raw excess kurtosis for the tensor or group mean/max
        kurtosis_keep:   absolute excess kurtosis above which *KEEP* is forced
                         regardless of score (default: inf, i.e. disabled)

    Returns:
        Tuple of (recommendation, reason) where reason is one of:
          kurtosis_floor    — forced *KEEP* due to excess_kurtosis >= kurtosis_keep
          score_percentile  — *KEEP* or FP8 driven by percentile threshold
          score_below_fp8_min — NVFP4 because score >= fp8_threshold but
                                 < fp8_min_score (fp8_min_score guard active)
          default           — NVFP4 because score is below all thresholds
    """
    if excess_kurtosis >= kurtosis_keep:
        return "*KEEP*", "kurtosis_floor"
    if score >= keep_threshold:
        return "*KEEP*", "score_percentile"
    elif score >= fp8_threshold and score >= fp8_min_score:
        return "FP8", "score_percentile"
    elif score >= fp8_threshold and score < fp8_min_score:
        return "NVFP4", "score_below_fp8_min"
    else:
        return "NVFP4", "default"


# ---------------------------------------------------------------------------
# Low-RAM safetensors helpers
# ---------------------------------------------------------------------------

# Safetensors dtype codes to torch dtype mapping
_ST_DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "F16":  torch.float16,
    "F32":  torch.float32,
    "F64":  torch.float64,
    "I8":   torch.int8,
    "I16":  torch.int16,
    "I32":  torch.int32,
    "I64":  torch.int64,
    "U8":   torch.uint8,
}


def read_safetensors_header(model_path: str) -> Tuple[Dict, int]:
    """
    Read the safetensors header without mmap-ing the data region.

    The safetensors format starts with an 8-byte little-endian uint64
    indicating the header length, followed by that many bytes of UTF-8
    JSON. The data region starts immediately after.

    Args:
        model_path: path to the safetensors file

    Returns:
        Tuple of (header_dict, data_offset) where data_offset is the byte
        position at which the tensor data region begins.
    """
    with open(model_path, "rb") as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            raise RuntimeError("File too short to be a valid safetensors file.")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_json = f.read(header_size)

    header = json.loads(header_json.decode("utf-8"))
    data_offset = 8 + header_size
    return header, data_offset


def load_tensor_lowram(
    model_path:  str,
    key:         str,
    header:      Dict,
    data_offset: int,
    device:      torch.device,
) -> torch.Tensor:
    """
    Load a single tensor from a safetensors file by reading only its byte range.

    Avoids mmap-ing the full file. The tensor is read as raw bytes, interpreted
    as the declared dtype, reshaped, and moved to the target device.

    Args:
        model_path:  path to the safetensors file
        key:         tensor key as it appears in the header
        header:      parsed header dict (from read_safetensors_header)
        data_offset: byte offset where the data region starts
        device:      torch device to place the tensor on

    Returns:
        Tensor with the declared shape and dtype, on the target device.
    """
    meta = header[key]
    dtype_str   = meta["dtype"]
    shape       = meta["shape"]
    byte_start, byte_end = meta["data_offsets"]

    torch_dtype = _ST_DTYPE_MAP.get(dtype_str)
    if torch_dtype is None:
        raise RuntimeError(f"Unsupported safetensors dtype: {dtype_str}")

    n_bytes = byte_end - byte_start
    with open(model_path, "rb") as f:
        f.seek(data_offset + byte_start)
        raw = f.read(n_bytes)

    tensor = torch.frombuffer(bytearray(raw), dtype=torch_dtype).reshape(shape)
    return tensor.to(device=device)


# ---------------------------------------------------------------------------
# Analysis pass
# ---------------------------------------------------------------------------

def analyze_model(
    model_path:    str,
    device:        torch.device,
    outlier_sigma: float,
    low_ram:       bool = False,
) -> List[TensorMetrics]:
    """
    Stream through the model file and compute metrics for all target tensors.

    Tensors are loaded one at a time, moved to the target device for
    computation, then immediately released to keep memory usage minimal.

    When low_ram=True, the safetensors header is parsed manually and each
    tensor is read by byte offset, avoiding the full-file mmap that
    safe_open performs internally. Use this on systems with limited RAM
    where mmap-ing the full model file would exhaust virtual memory.

    Args:
        model_path:    path to the BF16 safetensors file
        device:        torch device for computation
        outlier_sigma: sigma multiplier for outlier detection
        low_ram:       if True, use byte-range reads instead of safe_open mmap

    Returns:
        List of TensorMetrics, one per analyzed tensor
    """
    results: List[TensorMetrics] = []

    # Identify matching keys without loading any tensors
    matching: List[Tuple[str, str, int]] = []  # (key, layer_type, block_idx)

    if low_ram:
        # Parse header manually: reads only the JSON header (a few MB),
        # never mmap-ing the data region.
        header, data_offset = read_safetensors_header(model_path)
        all_keys = [k for k in header.keys() if k != "__metadata__"]
    else:
        with safe_open(model_path, framework="pt", device="cpu") as f:
            all_keys = list(f.keys())

    for key in all_keys:
        for layer_type, pattern in LAYER_PATTERNS.items():
            m = pattern.search(key)
            if m:
                block_idx = int(m.group(1))
                matching.append((key, layer_type, block_idx))
                break

    total = len(matching)
    print(f"  Target tensors found: {total}")
    print()

    if low_ram:
        # Load each tensor individually by byte offset; no persistent file handle.
        for i, (key, layer_type, block_idx) in enumerate(matching, 1):
            print(f"\r  Analyzing {i}/{total}: {key[:70]:<70}", end="", flush=True)

            tensor = load_tensor_lowram(model_path, key, header, data_offset, device)

            if tensor.ndim != 2:
                del tensor
                continue

            metrics = compute_metrics(tensor, outlier_sigma)

            results.append(TensorMetrics(
                key             = key,
                layer_type      = layer_type,
                block_idx       = block_idx,
                shape           = tuple(tensor.shape),
                excess_kurtosis = metrics["excess_kurtosis"],
                dynamic_range   = metrics["dynamic_range"],
                std             = metrics["std"],
                outlier_pct     = metrics["outlier_pct"],
                aspect_ratio    = metrics["aspect_ratio"],
            ))

            del tensor

    else:
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for i, (key, layer_type, block_idx) in enumerate(matching, 1):
                print(f"\r  Analyzing {i}/{total}: {key[:70]:<70}", end="", flush=True)

                tensor = f.get_tensor(key).to(device=device)

                if tensor.ndim != 2:
                    del tensor
                    continue

                metrics = compute_metrics(tensor, outlier_sigma)

                results.append(TensorMetrics(
                    key             = key,
                    layer_type      = layer_type,
                    block_idx       = block_idx,
                    shape           = tuple(tensor.shape),
                    excess_kurtosis = metrics["excess_kurtosis"],
                    dynamic_range   = metrics["dynamic_range"],
                    std             = metrics["std"],
                    outlier_pct     = metrics["outlier_pct"],
                    aspect_ratio    = metrics["aspect_ratio"],
                ))

                del tensor
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print(f"\r  {'Done':<80}")
    return results


# ---------------------------------------------------------------------------
# Block position classification
# ---------------------------------------------------------------------------

def compute_extreme_ranges(
    total_blocks: int,
    extreme_pct:  float,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Compute extreme block index ranges based on a percentage of total blocks.

    Args:
        total_blocks: total number of transformer blocks detected in the model
        extreme_pct:  percentage of blocks to consider extreme at each end

    Returns:
        Tuple of ((low_start, low_end), (high_start, high_end))
    """
    n_extreme = max(1, round(total_blocks * extreme_pct / 100.0))
    low  = (0, n_extreme - 1)
    high = (total_blocks - n_extreme, total_blocks - 1)
    return low, high


def classify_block(
    block_idx:    int,
    extreme_low:  Tuple[int, int],
    extreme_high: Tuple[int, int],
) -> str:
    """Return 'extreme_low', 'extreme_high', or 'middle'."""
    if extreme_low[0] <= block_idx <= extreme_low[1]:
        return "extreme_low"
    if extreme_high[0] <= block_idx <= extreme_high[1]:
        return "extreme_high"
    return "middle"


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def block_range_label(block_indices: List[int], total_blocks: int) -> str:
    """Return a compact range string like '0–3', '4–35', '0–39'."""
    lo, hi = min(block_indices), max(block_indices)
    if lo == 0 and hi == total_blocks - 1:
        return f"0–{hi}"
    return f"{lo}–{hi}"


def aggregate(
    metrics_list: List[TensorMetrics],
    layer_type:   str,
    block_range:  str,
) -> AggregatedMetrics:
    """Compute mean metrics across a list of TensorMetrics."""
    n = len(metrics_list)
    if n == 0:
        return AggregatedMetrics(layer_type, block_range, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return AggregatedMetrics(
        layer_type      = layer_type,
        block_range     = block_range,
        count           = n,
        excess_kurtosis = sum(m.excess_kurtosis for m in metrics_list) / n,
        kurtosis_max    = max(m.excess_kurtosis for m in metrics_list),
        dynamic_range   = sum(m.dynamic_range   for m in metrics_list) / n,
        std             = sum(m.std             for m in metrics_list) / n,
        outlier_pct     = sum(m.outlier_pct     for m in metrics_list) / n,
        aspect_ratio    = metrics_list[0].aspect_ratio,  # same shape within type
        score           = sum(m.score           for m in metrics_list) / n,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

SEP = "─" * 110

def fmt_row_summary(agg: AggregatedMetrics) -> str:
    return (
        f"  {agg.layer_type:<22} {agg.block_range:<10} "
        f"{agg.excess_kurtosis:>14.3f}  {agg.dynamic_range:>10.3f}  "
        f"{agg.std:>6.4f}  {agg.outlier_pct:>9.1f}%  "
        f"{agg.aspect_ratio:>5.2f}  {agg.score:>6.3f}  {agg.recommendation}"
    )


def fmt_row_detail(agg: AggregatedMetrics) -> str:
    if agg.spread_filtered:
        marker = "  [→NVFP4 ~spread]" if agg.recommendation == "FP8" else "  [→FP8 ~spread]"
    else:
        marker = ""
    return (
        f"  {agg.layer_type:<22} {agg.block_range:<12} "
        f"{agg.excess_kurtosis:>14.3f}  {agg.kurtosis_max:>10.3f}  {agg.dynamic_range:>10.3f}  "
        f"{agg.std:>6.4f}  {agg.outlier_pct:>9.1f}%  "
        f"{agg.score:>6.3f}  {agg.recommendation}{marker}"
    )


def print_summary_table(aggregated: List[AggregatedMetrics]) -> None:
    print()
    print(SEP)
    print("  SUMMARY BY LAYER TYPE — weight tensors")
    print(SEP)
    print(
        f"  {'Layer':<22} {'Blocks':<10} {'Excess kurtosis':>14}  {'Dyn. range':>10}  "
        f"{'Std':>6}  {'Outliers%':>9}  {'AR':>5}  {'Score':>6}  Recommendation"
    )
    print(SEP)
    for agg in aggregated:
        print(fmt_row_summary(agg))
    print(SEP)


def print_detail_table(group_name: str, rows: List[AggregatedMetrics]) -> None:
    print()
    print(SEP)
    print(f"  DETAIL: {group_name} — weight tensors by block position")
    print(SEP)
    print(
        f"  {'Layer':<22} {'Blocks':<12} {'Excess kurtosis':>14}  {'Kurt. max':>10}  {'Dyn. range':>10}  "
        f"{'Std':>6}  {'Outliers%':>9}  {'Score':>6}  Recommendation"
    )
    print(SEP)
    for row in rows:
        print(fmt_row_detail(row))
    print(SEP)


# ---------------------------------------------------------------------------
# Suggested convert_to_quant parameters
# ---------------------------------------------------------------------------

def _layer_type_to_key_pattern(layer_type: str) -> str:
    """Return the safetensors key fragment (without block prefix) for a layer type."""
    mapping = {
        "cross_attn.k":     r"cross_attn\.k\.weight",
        "cross_attn.v":     r"cross_attn\.v\.weight",
        "cross_attn.q":     r"cross_attn\.q\.weight",
        "cross_attn.o":     r"cross_attn\.o\.weight",
        "cross_attn.k_img": r"cross_attn\.k_img\.weight",
        "cross_attn.v_img": r"cross_attn\.v_img\.weight",
        "self_attn.k":      r"self_attn\.k\.weight",
        "self_attn.v":      r"self_attn\.v\.weight",
        "self_attn.q":      r"self_attn\.q\.weight",
        "self_attn.o":      r"self_attn\.o\.weight",
        "ffn.0":            r"ffn\.0\.weight",
        "ffn.2":            r"ffn\.2\.weight",
    }
    return mapping.get(layer_type, layer_type)


def _block_range_to_indices(block_range: str) -> List[int]:
    """
    Parse a block range label such as '0–3' or '4–35' into a list of integers.
    Handles both ASCII hyphen and en-dash.
    """
    # Normalize en-dash and em-dash to hyphen
    normalized = block_range.replace("–", "-").replace("—", "-")
    parts = normalized.split("-")
    lo, hi = int(parts[0]), int(parts[1])
    return list(range(lo, hi + 1))


def _blocks_to_alternation(indices: List[int]) -> str:
    """Return a regex alternation string for a list of block indices, e.g. '0|1|2|3'."""
    return "|".join(str(i) for i in sorted(indices))


def build_convert_to_quant_params(
    all_detail_rows:      List[AggregatedMetrics],
    all_metrics:          List[TensorMetrics],
    min_group_spread:     float,
    spread_filter_exempt: set,
) -> Tuple[List[Tuple[str, List[int]]], List[Tuple[str, List[int]]]]:
    """
    Build FP8 and *KEEP* recommendations for convert_to_quant.

    FP8 / NVFP4 decisions use group-level recommendations with the spread
    filter applied:
      - FP8 groups whose layer type has insufficient score spread across
        position groups are downgraded to NVFP4 (spread too low to trust
        per-group signal). Affected rows are marked spread_filtered=True.
      - *KEEP* groups in low-spread types are demoted to FP8 (score still
        warrants conservative quantization, but not full BF16 retention).

    Layer types listed in spread_filter_exempt are never subject to the
    spread filter in either direction. Their per-tensor individual
    recommendation is always used as-is, making the group position
    irrelevant for those types.

    *KEEP* decisions are resolved at individual tensor level: when a group
    recommendation is *KEEP*, only the tensors that are individually *KEEP*
    are added to keep_entries. Tensors in the same group whose individual
    recommendation is FP8 are added to fp8_entries instead. Tensors that
    are individually NVFP4 within a *KEEP* group need no entry.
    This prevents over-protection of blocks that don't individually warrant
    BF16 retention (e.g. when one outlier tensor pulls the group mean above
    the *KEEP* threshold while its neighbours are unremarkable).

    Args:
        all_detail_rows:      AggregatedMetrics rows from all detail tables,
                              with recommendations already assigned
        all_metrics:          per-tensor TensorMetrics with individual
                              recommendations already assigned
        min_group_spread:     minimum score spread across position groups required
                              to use per-group FP8 recommendations
        spread_filter_exempt: set of layer_type strings whose tensors always use
                              their individual recommendation, bypassing the spread
                              filter entirely in both directions

    Returns:
        Tuple of (fp8_entries, keep_entries), each a list of
        (layer_type, block_indices) tuples.
    """
    # Index individual tensor recommendations for *KEEP* resolution and exempt handling
    individual_rec: Dict[Tuple[str, int], str] = {
        (m.layer_type, m.block_idx): m.recommendation for m in all_metrics
    }

    # Group rows by layer type to compute per-type spread
    by_layer_type: Dict[str, List[AggregatedMetrics]] = defaultdict(list)
    for row in all_detail_rows:
        by_layer_type[row.layer_type].append(row)

    # Identify layer types with insufficient spread, excluding exempt types
    low_spread_types: set = set()
    for layer_type, rows in by_layer_type.items():
        if layer_type in spread_filter_exempt:
            continue
        scores = [r.score for r in rows]
        spread = max(scores) - min(scores)
        if spread < min_group_spread:
            low_spread_types.add(layer_type)

    fp8_entries:  List[Tuple[str, List[int]]] = []
    keep_entries: List[Tuple[str, List[int]]] = []

    for row in all_detail_rows:
        indices = _block_range_to_indices(row.block_range)

        if row.layer_type in spread_filter_exempt:
            # Exempt layer type: always use individual tensor recommendations,
            # bypassing group-level spread filter in both directions.
            keep_idxs = []
            fp8_idxs  = []
            for idx in indices:
                rec = individual_rec.get((row.layer_type, idx), "NVFP4")
                if rec == "*KEEP*":
                    keep_idxs.append(idx)
                elif rec == "FP8":
                    fp8_idxs.append(idx)
                # NVFP4 tensors need no entry
            if keep_idxs:
                keep_entries.append((row.layer_type, keep_idxs))
            if fp8_idxs:
                fp8_entries.append((row.layer_type, fp8_idxs))

        elif row.layer_type in low_spread_types and row.recommendation == "FP8":
            row.spread_filtered = True
            # downgraded to NVFP4 — do not add to fp8_entries

        elif row.layer_type in low_spread_types and row.recommendation == "*KEEP*":
            row.spread_filtered = True
            # demoted from *KEEP* to FP8 — spread too low to justify BF16 retention.
            # Always FP8 regardless of fp8_min_score: *KEEP* score is by definition
            # above the FP8 threshold.
            fp8_entries.append((row.layer_type, indices))

        elif row.recommendation == "FP8":
            fp8_entries.append((row.layer_type, indices))

        elif row.recommendation == "*KEEP*":
            # Resolve at individual tensor level to avoid over-protecting blocks
            # that don't individually warrant BF16.
            keep_idxs = []
            fp8_idxs  = []
            for idx in indices:
                rec = individual_rec.get((row.layer_type, idx), "NVFP4")
                if rec == "*KEEP*":
                    keep_idxs.append(idx)
                elif rec == "FP8":
                    fp8_idxs.append(idx)
                # NVFP4 tensors within a *KEEP* group need no entry
            if keep_idxs:
                keep_entries.append((row.layer_type, keep_idxs))
            if fp8_idxs:
                fp8_entries.append((row.layer_type, fp8_idxs))

    return fp8_entries, keep_entries


def build_detail_regex(entries: List[Tuple[str, List[int]]]) -> Optional[str]:
    r"""
    Build a single alternation regex from a list of (layer_type, block_indices) tuples.

    Each entry produces a pattern of the form:
        blocks\.(0|1|2|3)\.cross_attn\.k\.weight

    All patterns are joined with '|'.

    Args:
        entries: list of (layer_type, block_indices) tuples

    Returns:
        Combined regex string, or None if entries is empty.
    """
    if not entries:
        return None
    patterns = []
    for layer_type, indices in entries:
        key_pat   = _layer_type_to_key_pattern(layer_type)
        block_alt = _blocks_to_alternation(indices)
        patterns.append(rf"blocks\.({block_alt})\.{key_pat}")
    return "|".join(patterns)


def print_suggested_params(
    fp8_entries:      List[Tuple[str, List[int]]],
    keep_entries:     List[Tuple[str, List[int]]],
    fp8_min_score:    float,
    min_group_spread: float,
) -> None:
    print()
    print(SEP)
    print("  SUGGESTED convert_to_quant PARAMETERS")
    print(SEP)
    print("  Based on detail-level analysis (by block position group):")
    print(f"  Filters active — FP8 min score: {fp8_min_score}  |  min group spread: {min_group_spread}")
    print()

    fp8_regex = build_detail_regex(fp8_entries)
    if fp8_regex:
        print(f'  --custom-layers "{fp8_regex}"')
        print( "  --custom-type fp8")
    else:
        print("  No layers recommended as FP8 (all NVFP4, or filtered by min score / spread)")

    print()

    keep_regex = build_detail_regex(keep_entries)
    if keep_regex:
        print(f'  --exclude-layers "{keep_regex}"')
    else:
        print("  No additional --exclude-layers needed (--wan covers existing exclusions)")

    print(SEP)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(
    all_metrics:      List[TensorMetrics],
    output_path:      str,
    effective_rec:    Dict[Tuple[str, int], str],
    effective_reason: Dict[Tuple[str, int], str],
) -> None:
    """Export per-tensor metrics to a CSV file.

    Includes both the raw per-tensor recommendation (assigned before the spread
    filter runs) and the effective recommendation after the spread filter has
    been applied at group level. The two columns differ only for tensors whose
    layer type was affected by the spread filter:
      - recommendation:           raw value from assign_recommendation
      - effective_recommendation: NVFP4 if the group was FP8->NVFP4 by spread,
                                  FP8   if the group was *KEEP*->FP8 by spread,
                                  same as recommendation otherwise
      - reason: decision reason for the effective recommendation; one of:
          kurtosis_floor       *KEEP* forced because excess kurtosis >= kurtosis_keep
          score_percentile     *KEEP* or FP8 driven by percentile threshold
          score_below_fp8_min  NVFP4 because fp8_min_score guard blocked FP8
          default              NVFP4 because score is below all thresholds
          group_keep_resolved   tensor was individually *KEEP* but demoted to FP8/NVFP4
                                during per-tensor resolution within a *KEEP* group
          group_fp8_promotion   tensor was individually NVFP4 but carried up to FP8
                                because its block-position group scored FP8
          group_spread_demotion tensor was individually FP8 or *KEEP* but brought down
                                by the group's spread-filter result (spread_filtered
                                flag not set because the group rec is already NVFP4)
          spread_demotion       FP8->NVFP4 or *KEEP*->FP8 by spread filter
    """
    fieldnames = [
        "key", "layer_type", "block_idx",
        "rows", "cols",
        "excess_kurtosis", "dynamic_range", "std", "outlier_pct",
        "aspect_ratio", "score", "recommendation", "effective_recommendation",
        "reason",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            key = (m.layer_type, m.block_idx)
            eff = effective_rec.get(key, m.recommendation)
            rsn = effective_reason.get(key, m.reason)
            writer.writerow({
                "key":                      m.key,
                "layer_type":               m.layer_type,
                "block_idx":                m.block_idx,
                "rows":                     m.shape[0],
                "cols":                     m.shape[1],
                "excess_kurtosis":          f"{m.excess_kurtosis:.4f}",
                "dynamic_range":            f"{m.dynamic_range:.4f}",
                "std":                      f"{m.std:.6f}",
                "outlier_pct":              f"{m.outlier_pct:.2f}",
                "aspect_ratio":             f"{m.aspect_ratio:.2f}",
                "score":                    f"{m.score:.4f}",
                "recommendation":           m.recommendation,
                "effective_recommendation": eff,
                "reason":                   rsn,
            })
    print(f"\n  CSV exported to: {output_path}")


# ---------------------------------------------------------------------------
# Output file size estimation
# ---------------------------------------------------------------------------

# Bytes per weight element for each effective format.
# NVFP4: 4 bits data + 1 byte scale per group of 16 weights = 4.5 bits/weight
#        = 0.5625 bytes/weight (assumes group size 16, standard for NVIDIA NVFP4).
_BYTES_PER_WEIGHT: Dict[str, float] = {
    "*KEEP*": 2.0,    # BF16
    "FP8":    1.0,
    "NVFP4":  0.5625,
}


def estimate_output_size(
    all_metrics:      List[TensorMetrics],
    effective_rec:    Dict[Tuple[str, int], str],
    original_bytes:   int,
) -> Dict:
    """
    Estimate the output file size if all recommendations are adopted.

    The 480 tensors in scope are re-sized according to their effective
    recommendation. The remaining tensors (out of scope) are assumed to
    stay in BF16; their aggregate size is derived by subtracting the
    in-scope BF16 footprint from the original file size.

    Args:
        all_metrics:    per-tensor TensorMetrics with shapes known
        effective_rec:  (layer_type, block_idx) -> effective recommendation
        original_bytes: original file size in bytes (from stat)

    Returns:
        Dict with keys:
          in_scope_original_bytes   — in-scope tensors at BF16
          in_scope_estimated_bytes  — in-scope tensors after quantization
          per_format_bytes          — dict {format: estimated_bytes}
          per_format_counts         — dict {format: tensor_count}
          out_of_scope_bytes        — out-of-scope tensors (unchanged, BF16)
          total_estimated_bytes     — full estimated output file size
          original_bytes            — original file size (passed through)
    """
    in_scope_original = 0
    in_scope_estimated = 0
    per_format_bytes:  Dict[str, float] = {"*KEEP*": 0.0, "FP8": 0.0, "NVFP4": 0.0}
    per_format_counts: Dict[str, int]   = {"*KEEP*": 0,   "FP8": 0,   "NVFP4": 0}

    for m in all_metrics:
        n_weights = m.shape[0] * m.shape[1]
        in_scope_original += n_weights * 2  # BF16 = 2 bytes/weight

        fmt = effective_rec.get((m.layer_type, m.block_idx), m.recommendation)
        bpw = _BYTES_PER_WEIGHT.get(fmt, 2.0)
        tensor_bytes = n_weights * bpw
        in_scope_estimated          += tensor_bytes
        per_format_bytes[fmt]        = per_format_bytes.get(fmt, 0.0) + tensor_bytes
        per_format_counts[fmt]       = per_format_counts.get(fmt, 0)  + 1

    out_of_scope = original_bytes - in_scope_original
    total_estimated = out_of_scope + in_scope_estimated

    return {
        "in_scope_original_bytes":  in_scope_original,
        "in_scope_estimated_bytes": in_scope_estimated,
        "per_format_bytes":         per_format_bytes,
        "per_format_counts":        per_format_counts,
        "out_of_scope_bytes":       out_of_scope,
        "total_estimated_bytes":    total_estimated,
        "original_bytes":           original_bytes,
    }


def print_size_estimate(est: Dict) -> None:
    """Print the estimated output file size block."""
    def gb(b: float) -> str:
        return f"{b / 1024**3:.2f} GB"

    original  = est["original_bytes"]
    total     = est["total_estimated_bytes"]
    delta     = total - original
    delta_pct = delta / original * 100.0
    pfb       = est["per_format_bytes"]
    pfc       = est["per_format_counts"]
    n_scope   = sum(pfc.values())

    sign      = "+" if delta >= 0 else "−"
    abs_delta = abs(delta) / 1024**3

    print()
    print(SEP)
    print("  ESTIMATED OUTPUT FILE SIZE")
    print(SEP)
    print(f"  Tensors in scope ({n_scope})  —  original BF16: {gb(est['in_scope_original_bytes'])}")
    for fmt in ("*KEEP*", "FP8", "NVFP4"):
        label = f"    {fmt} ({'BF16' if fmt == '*KEEP*' else fmt})"
        print(f"  {label:<38}  {pfc[fmt]:>4} tensors  →  {gb(pfb[fmt]):>9}")
    print(f"  {'  Subtotal after quantization':<38}  {gb(est['in_scope_estimated_bytes']):>9}")
    print(f"  {'Tensors out of scope (BF16, unchanged)':<38}  {gb(est['out_of_scope_bytes']):>9}")
    print(f"  {'─' * 60}")
    print(f"  {'Total estimated':<38}  {gb(total):>9}")
    print(f"  {'Original file size':<38}  {gb(original):>9}")
    print(f"  {'Delta':<38}  {sign}{abs_delta:.2f} GB  ({sign}{abs(delta_pct):.1f}%)")
    print(SEP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantization sensitivity analysis for Wan safetensors models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Layer types analyzed across all transformer blocks:
  cross_attn: k, v, q, o, k_img, v_img
  self_attn:  k, v, q, o
  ffn:        0, 2

Score composition:
  - Excess kurtosis (IQR-normalized globally)       default weight: 0.6
  - Dynamic range (within-type IQR normalized)      default weight: 0.3
  - Aspect ratio  (within-type min-max normalized)  default weight: 0.1
  Outlier% is shown in tables but excluded from score (correlated with kurtosis).
  Note: aspect ratio contributes 0 when all tensors of a type share the same
  shape (standard in transformer architectures); a warning is shown at runtime.

Thresholds and extreme block ranges are derived automatically from the model.
""",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Path to the BF16 safetensors model file.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="OUTPUT.csv",
        help="Export per-tensor metrics to a CSV file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for computation: 'cpu' or 'cuda' (default: auto-detect).",
    )
    parser.add_argument(
        "--outlier-sigma",
        type=float,
        default=3.0,
        metavar="N",
        help="Standard deviation multiplier for outlier detection (default: 3.0).",
    )
    parser.add_argument(
        "--fp8-percentile",
        type=float,
        default=75.0,
        metavar="P",
        help="Score percentile threshold for FP8 recommendation (default: 75).",
    )
    parser.add_argument(
        "--keep-percentile",
        type=float,
        default=90.0,
        metavar="P",
        help="Score percentile threshold for *KEEP* recommendation (default: 90).",
    )
    parser.add_argument(
        "--extreme-pct",
        type=float,
        default=10.0,
        metavar="P",
        help="Percentage of blocks to consider extreme at each end (default: 10).",
    )
    parser.add_argument(
        "--kurtosis-weight",
        type=float,
        default=0.6,
        metavar="W",
        help="Weight for IQR-normalized excess kurtosis in combined score (default: 0.6).",
    )
    parser.add_argument(
        "--range-weight",
        type=float,
        default=0.3,
        metavar="W",
        help="Weight for within-type dynamic range in combined score (default: 0.3).",
    )
    parser.add_argument(
        "--ar-weight",
        type=float,
        default=0.1,
        metavar="W",
        help="Weight for within-type aspect ratio in combined score (default: 0.1).",
    )
    parser.add_argument(
        "--fp8-min-score",
        type=float,
        default=0.50,
        metavar="S",
        help=(
            "Absolute minimum score required for an FP8 recommendation, regardless "
            "of the percentile threshold. Prevents percentile-based thresholds from "
            "recommending FP8 when all scores are clustered in a narrow range. "
            "Default 0.50 is calibrated from Wan 2.1 14B score distribution; "
            "set to 0.0 to disable. (default: 0.50)"
        ),
    )
    parser.add_argument(
        "--min-group-spread",
        type=float,
        default=0.06,
        metavar="S",
        help=(
            "Minimum score spread across block position groups required for "
            "per-group FP8 recommendations within a layer type. If the difference "
            "between the highest and lowest group score is below this threshold, "
            "variation between groups is considered noise and all groups of that "
            "type are treated as NVFP4. Rows downgraded by this filter are marked "
            "[~spread] in the detail tables. Set to 0.0 to disable. (default: 0.06)"
        ),
    )
    parser.add_argument(
        "--spread-filter-exempt",
        nargs="*",
        default=[],
        metavar="LAYER_TYPE",
        help=(
            "Layer types that bypass the spread filter entirely, always using "
            "their per-tensor individual recommendation regardless of group-level "
            "score spread. Applies in both directions: neither FP8→NVFP4 demotion "
            "nor NVFP4→FP8 promotion by group position will affect these types. "
            "Recommended for layers that are numerically sensitive in DiT cross-attention, "
            "where group averaging may mask individual tensor behaviour. "
            "Example: --spread-filter-exempt cross_attn.k cross_attn.q "
            "(default: none)"
        ),
    )
    parser.add_argument(
        "--kurtosis-keep",
        type=float,
        default=8.0,
        metavar="K",
        help=(
            "Hard floor on excess kurtosis: any layer (or group, using kurtosis_max) "
            "with excess kurtosis >= K is forced to *KEEP* regardless of its "
            "percentile-based score. Protects extremely leptokurtic tensors that "
            "norm_iqr() clips before they can reach keep_threshold. "
            "Default 8.0 captures the blocks.12 and blocks.20 cross_attn.o outliers "
            "found in Wan 2.1 14B LightX2V distilled; set to inf to disable. "
            "(default: 8.0)"
        ),
    )
    parser.add_argument(
        "--lowram",
        action="store_true",
        default=False,
        help=(
            "Avoid mmap-ing the full model file. Reads the safetensors header "
            "once and loads each tensor individually by byte offset. Use this "
            "on systems with limited RAM (e.g. 16 GB without swap) where a "
            "full-file mmap would exhaust virtual memory."
        ),
    )

    args = parser.parse_args()

    # Validate input file
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}")
        sys.exit(1)

    # Validate weights sum to 1
    total_weight = args.kurtosis_weight + args.range_weight + args.ar_weight
    if abs(total_weight - 1.0) > 1e-6:
        print(f"Error: --kurtosis-weight + --range-weight + --ar-weight must sum to 1.0 (got {total_weight:.4f})")
        sys.exit(1)

    # Validate spread_filter_exempt layer types
    valid_layer_types = set(LAYER_PATTERNS.keys())
    for lt in args.spread_filter_exempt:
        if lt not in valid_layer_types:
            print(f"Error: --spread-filter-exempt: unknown layer type '{lt}'. "
                  f"Valid types: {', '.join(sorted(valid_layer_types))}")
            sys.exit(1)

    # Resolve device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Device info string
    if device.type == "cuda":
        props   = torch.cuda.get_device_properties(device)
        vram_mb = props.total_memory // (1024 * 1024)
        device_str = f"CUDA ({props.name}, {vram_mb} MB)"
    else:
        device_str = "CPU"

    print()
    print(f"  Quantization Sensitivity Analysis — {model_path.name}")
    print(f"  Device: {device_str}")
    exempt_str = ", ".join(sorted(args.spread_filter_exempt)) if args.spread_filter_exempt else "none"
    print(f"  Outlier sigma: {args.outlier_sigma}  |  "
          f"FP8 percentile: {args.fp8_percentile}  |  "
          f"Keep percentile: {args.keep_percentile}  |  "
          f"FP8 min score: {args.fp8_min_score}  |  "
          f"Min group spread: {args.min_group_spread}  |  "
          f"Spread filter exempt: {exempt_str}  |  "
          f"Extreme blocks: {args.extreme_pct}%  |  "
          f"Kurtosis keep: {args.kurtosis_keep}  |  "
          f"Low-RAM: {'yes' if args.lowram else 'no'}")
    print()
    print("  Scanning model...")

    # Analysis pass
    all_metrics = analyze_model(str(model_path), device, args.outlier_sigma, low_ram=args.lowram)

    if not all_metrics:
        print("Error: no target tensors found in model.")
        sys.exit(1)

    # Determine total number of blocks from detected tensors
    total_blocks = max(m.block_idx for m in all_metrics) + 1

    print(f"  Tensors analyzed: {len(all_metrics)} | Blocks: {total_blocks}")

    # Detect whether AR will contribute to scores: within-type AR variation
    # is zero when all tensors of a type share the same shape (universal in
    # Wan and most transformer architectures). Report this before scoring so
    # the user knows the effective weight budget is 0.9, not 1.0.
    by_type_check: Dict[str, List[TensorMetrics]] = defaultdict(list)
    for m in all_metrics:
        by_type_check[m.layer_type].append(m)
    ar_inactive = all(
        max(m.aspect_ratio for m in g) == min(m.aspect_ratio for m in g)
        for g in by_type_check.values()
    )
    ar_note = " [inactive: no within-type AR variation — effective budget: 0.9]" if ar_inactive else ""
    print(f"  Score weights — kurtosis (IQR): {args.kurtosis_weight}  |  "
          f"dyn. range: {args.range_weight}  |  "
          f"aspect ratio: {args.ar_weight}{ar_note}  "
          f"(outliers% shown but not scored)")

    # Compute extreme block ranges from model size
    extreme_low, extreme_high = compute_extreme_ranges(total_blocks, args.extreme_pct)
    print(f"  Extreme blocks — low: {extreme_low[0]}–{extreme_low[1]}  |  "
          f"high: {extreme_high[0]}–{extreme_high[1]}")

    # Compute combined sensitivity scores
    compute_scores(
        all_metrics,
        kurtosis_weight = args.kurtosis_weight,
        range_weight    = args.range_weight,
        ar_weight       = args.ar_weight,
    )

    # Compute auto thresholds from score distribution
    fp8_threshold, keep_threshold = compute_auto_thresholds(
        all_metrics, args.fp8_percentile, args.keep_percentile
    )
    # Indicate which parameter is actually controlling the FP8 threshold so
    # the user knows whether adjusting --fp8-percentile or --fp8-min-score
    # will have effect.
    if args.fp8_min_score > fp8_threshold:
        fp8_driver = f"*KEEP*: score >= {keep_threshold:.3f}"
        print(f"  Auto thresholds — FP8: score >= {args.fp8_min_score:.3f} "
              f"[fp8-min-score active; percentile gave {fp8_threshold:.3f}]  |  "
              f"{fp8_driver}")
    else:
        print(f"  Auto thresholds — FP8: score >= {fp8_threshold:.3f} [percentile]  |  "
              f"*KEEP*: score >= {keep_threshold:.3f}")

    # Assign recommendations
    for m in all_metrics:
        m.recommendation, m.reason = assign_recommendation(
            m.score, fp8_threshold, keep_threshold, args.fp8_min_score,
            excess_kurtosis=m.excess_kurtosis,
            kurtosis_keep=args.kurtosis_keep,
        )

    # ---------------------------------------------------------------------------
    # Build summary table (aggregated across all blocks per layer type)
    # ---------------------------------------------------------------------------
    by_layer: Dict[str, List[TensorMetrics]] = defaultdict(list)
    for m in all_metrics:
        by_layer[m.layer_type].append(m)

    summary_aggs: List[AggregatedMetrics] = []
    for layer_type in LAYER_PATTERNS.keys():
        group = by_layer.get(layer_type, [])
        if not group:
            continue
        block_indices = [m.block_idx for m in group]
        agg = aggregate(group, layer_type, block_range_label(block_indices, total_blocks))
        agg.recommendation, agg.reason = assign_recommendation(
            agg.score, fp8_threshold, keep_threshold, args.fp8_min_score,
            excess_kurtosis=agg.kurtosis_max,
            kurtosis_keep=args.kurtosis_keep,
        )
        summary_aggs.append(agg)

    print_summary_table(summary_aggs)

    # ---------------------------------------------------------------------------
    # Build detail rows (by block position group per layer type)
    # ---------------------------------------------------------------------------
    all_detail_rows: List[AggregatedMetrics] = []
    detail_rows_by_group: Dict[str, List[AggregatedMetrics]] = {}

    for group_name, layer_types in DETAIL_GROUPS.items():
        group_rows: List[AggregatedMetrics] = []

        for layer_type in layer_types:
            group = by_layer.get(layer_type, [])
            if not group:
                continue

            extreme_low_group  = [m for m in group
                                   if classify_block(m.block_idx, extreme_low, extreme_high) == "extreme_low"]
            middle_group       = [m for m in group
                                   if classify_block(m.block_idx, extreme_low, extreme_high) == "middle"]
            extreme_high_group = [m for m in group
                                   if classify_block(m.block_idx, extreme_low, extreme_high) == "extreme_high"]

            for subset in [extreme_low_group, middle_group, extreme_high_group]:
                if not subset:
                    continue
                indices = [m.block_idx for m in subset]
                label   = block_range_label(indices, total_blocks)
                agg     = aggregate(subset, layer_type, label)
                agg.recommendation, agg.reason = assign_recommendation(
                    agg.score, fp8_threshold, keep_threshold, args.fp8_min_score,
                    excess_kurtosis=agg.kurtosis_max,
                    kurtosis_keep=args.kurtosis_keep,
                )
                group_rows.append(agg)

        if group_rows:
            detail_rows_by_group[group_name] = group_rows
            all_detail_rows.extend(group_rows)

    # ---------------------------------------------------------------------------
    # Suggested convert_to_quant parameters
    # (must run before printing detail tables so spread_filtered flags are set)
    # ---------------------------------------------------------------------------
    fp8_entries, keep_entries = build_convert_to_quant_params(
        all_detail_rows, all_metrics, args.min_group_spread,
        spread_filter_exempt=set(args.spread_filter_exempt),
    )

    # ---------------------------------------------------------------------------
    # Print detail tables (spread_filtered flags now set)
    # ---------------------------------------------------------------------------
    for group_name, group_rows in detail_rows_by_group.items():
        print_detail_table(group_name, group_rows)

    print_suggested_params(fp8_entries, keep_entries, args.fp8_min_score, args.min_group_spread)

    # ---------------------------------------------------------------------------
    # CSV export
    # ---------------------------------------------------------------------------
    if args.csv:
        # Build a (layer_type, block_idx) -> effective_recommendation index from
        # the detail rows, where spread_filtered rows carry the post-filter decision.
        # For *KEEP* groups, resolution is done at individual tensor level to match
        # the behaviour of build_convert_to_quant_params: only tensors that are
        # individually *KEEP* are kept in BF16; tensors in the same group whose
        # individual recommendation is FP8 or NVFP4 are recorded accordingly.
        # Tensors not present in any detail row keep their raw recommendation.
        individual_rec: Dict[Tuple[str, int], str] = {
            (m.layer_type, m.block_idx): m.recommendation for m in all_metrics
        }
        effective_rec: Dict[Tuple[str, int], str] = {}
        effective_reason: Dict[Tuple[str, int], str] = {}

        # Build an individual reason index for fast lookup below
        individual_reason: Dict[Tuple[str, int], str] = {
            (m.layer_type, m.block_idx): m.reason for m in all_metrics
        }

        spread_exempt = set(args.spread_filter_exempt)

        for row in all_detail_rows:
            if row.layer_type in spread_exempt:
                # Exempt layer type: each tensor uses its individual recommendation
                # and reason regardless of group position or spread.
                for idx in _block_range_to_indices(row.block_range):
                    ind_rec = individual_rec.get((row.layer_type, idx), "NVFP4")
                    effective_rec[(row.layer_type, idx)] = ind_rec
                    # reason stays as the individual tensor's reason

            elif row.spread_filtered:
                # spread filter changed FP8→NVFP4 or *KEEP*→FP8 at group level
                eff = "NVFP4" if row.recommendation == "FP8" else "FP8"
                for idx in _block_range_to_indices(row.block_range):
                    effective_rec[(row.layer_type, idx)]    = eff
                    effective_reason[(row.layer_type, idx)] = "spread_demotion"
            elif row.recommendation == "*KEEP*":
                # Resolve at individual tensor level, matching build_convert_to_quant_params.
                # Tensors that are individually *KEEP* keep their own reason.
                # Tensors that are individually FP8/NVFP4 within a *KEEP* group are
                # labelled group_keep_resolved: the group pulled them up to *KEEP* at
                # aggregate level, but individual resolution brings them back down.
                for idx in _block_range_to_indices(row.block_range):
                    ind_rec = individual_rec.get((row.layer_type, idx), row.recommendation)
                    effective_rec[(row.layer_type, idx)] = ind_rec
                    if ind_rec == "*KEEP*":
                        # Genuinely kept: reason is the individual tensor's own reason
                        pass  # effective_reason falls back to m.reason in export_csv
                    else:
                        # Demoted from group *KEEP* to individual FP8/NVFP4
                        effective_reason[(row.layer_type, idx)] = "group_keep_resolved"
            else:
                # Group recommendation is FP8 or NVFP4 (not spread-filtered).
                # Three sub-cases depending on direction:
                #   group_fp8_promotion: individual was NVFP4/below-FP8, group carries it up to FP8
                #   group_spread_demotion: individual was FP8 or *KEEP*, group brings it down
                #                         (spread filter acted but spread_filtered flag not set,
                #                          e.g. when the group rec itself is already NVFP4)
                #   no change: individual and group agree
                for idx in _block_range_to_indices(row.block_range):
                    effective_rec[(row.layer_type, idx)] = row.recommendation
                    ind_rec = individual_rec.get((row.layer_type, idx), row.recommendation)
                    if ind_rec == row.recommendation:
                        pass  # reason stays as the individual tensor's reason
                    elif row.recommendation == "FP8" and ind_rec in ("NVFP4",):
                        effective_reason[(row.layer_type, idx)] = "group_fp8_promotion"
                    else:
                        # Individual was FP8 or *KEEP* but group is lower: demotion by spread
                        effective_reason[(row.layer_type, idx)] = "group_spread_demotion"

        export_csv(all_metrics, args.csv, effective_rec, effective_reason)

    # ---------------------------------------------------------------------------
    # Estimated output file size
    # ---------------------------------------------------------------------------
    # Build effective_rec for size estimation regardless of --csv flag.
    # Reuse the dict built above if CSV was requested; otherwise build it now.
    if not args.csv:
        individual_rec_sz: Dict[Tuple[str, int], str] = {
            (m.layer_type, m.block_idx): m.recommendation for m in all_metrics
        }
        effective_rec_sz: Dict[Tuple[str, int], str] = {}
        spread_exempt_sz = set(args.spread_filter_exempt)
        for row in all_detail_rows:
            if row.layer_type in spread_exempt_sz:
                for idx in _block_range_to_indices(row.block_range):
                    effective_rec_sz[(row.layer_type, idx)] = individual_rec_sz.get(
                        (row.layer_type, idx), "NVFP4"
                    )
            elif row.spread_filtered:
                eff = "NVFP4" if row.recommendation == "FP8" else "FP8"
                for idx in _block_range_to_indices(row.block_range):
                    effective_rec_sz[(row.layer_type, idx)] = eff
            elif row.recommendation == "*KEEP*":
                for idx in _block_range_to_indices(row.block_range):
                    effective_rec_sz[(row.layer_type, idx)] = individual_rec_sz.get(
                        (row.layer_type, idx), row.recommendation
                    )
            else:
                for idx in _block_range_to_indices(row.block_range):
                    effective_rec_sz[(row.layer_type, idx)] = row.recommendation
    else:
        effective_rec_sz = effective_rec  # already built above

    original_bytes = model_path.stat().st_size
    size_est = estimate_output_size(all_metrics, effective_rec_sz, original_bytes)
    print_size_estimate(size_est)


if __name__ == "__main__":
    main()
