#!/usr/bin/env python3
"""
Quantization Sensitivity Analysis for safetensors models.

Analyzes weight tensors of transformer blocks to estimate sensitivity to
quantization, helping decide which layers should be kept in BF16 (*KEEP*),
quantized to FP8, or quantized to NVFP4.

Supported models (--model):
  wan      - Wan 2.1 video diffusion model (blocks.* architecture)
  zimage   - Z-Image / Z-Image Turbo image diffusion model (layers.* architecture)

A hard floor on excess kurtosis (--kurtosis-keep, default 8.0) forces *KEEP*
regardless of the percentile-based score when a tensor's distribution is
extremely leptokurtic.

For Wan 2.1, targets the following layer types across all transformer blocks:
  - cross_attn: k, v, q, o, k_img, v_img
  - self_attn:  k, v, q, o
  - ffn:        0, 2

For Z-Image, targets the following layer types:
  Main blocks (layers.0-N):
  - attention: qkv, out
  - feed_forward: w1, w2, w3
  - adaLN_modulation: 0
  Refiners (context_refiner.0-1, noise_refiner.0-1) — group flat, no
  positional classification:
  - attention: qkv, out
  - feed_forward: w1, w2, w3

  The following layers from ZIMAGE_LAYER_KEYNAMES are intentionally excluded
  from analysis: x_embedder, clip_text_pooled_proj, final_layer,
  cap_embedder.1, adaLN_modulation (global), t_embedder, time_text_embed.
  These receive special treatment in convert_to_quant (--zimage /
  --zimage_refiner) and are assumed to reflect an informed architectural
  decision by the model author. Use convert_to_quant documentation for
  details on their handling.

Metrics computed per tensor (on GPU when available):
  - Excess kurtosis:  indicates heavy-tailed distributions (0 = normal)
  - Dynamic range:    abs(max) - abs(min), within-type normalized for scoring
  - Std deviation:    overall weight dispersion
  - Outliers %:       fraction of values exceeding N standard deviations
  - Aspect ratio:     max(rows, cols) / min(rows, cols), within-type normalized

Thresholds are derived automatically from the model's own score distribution
using configurable percentiles.

Usage:
    python quant_probe.py model.safetensors --model wan
    python quant_probe.py model.safetensors --model zimage
    python quant_probe.py model.safetensors --model zimage --csv results.csv
    python quant_probe.py model.safetensors --model wan --device cpu
    python quant_probe.py model.safetensors --model wan --lowram

Compatibility with convert_to_quant:
  Wan 2.1:  use --wan flag
  Z-Image:  use --zimage (quantize refiners) or --zimage_refiner (keep refiners
            in BF16). The --exclude-layers regex output from this script is
            compatible with both flags.
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
# Model configuration registry
# ---------------------------------------------------------------------------

# Default args calibrated from Wan 2.1 14B score distribution.
# Z-Image defaults are marked None — uncalibrated, use Wan values as fallback
# until empirical calibration is performed on a real Z-Image model.
_WAN_DEFAULT_ARGS = {
    "fp8_percentile":  75.0,
    "keep_percentile": 90.0,
    "kurtosis_keep":   8.0,
    "fp8_min_score":   0.50,
    "min_group_spread": 0.06,
}

_ZIMAGE_DEFAULT_ARGS = {
    "fp8_percentile":   75.0,
    "keep_percentile":  90.0,
    "kurtosis_keep":    8.0,
    "fp8_min_score":    0.0,
    "min_group_spread": 0.20,
}

# Regex patterns for Wan 2.1 — keyed by display group name.
# Each pattern captures the block index in group 1.
_WAN_LAYER_PATTERNS = {
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

_WAN_DETAIL_GROUPS = {
    "cross_attn": ["cross_attn.k", "cross_attn.v", "cross_attn.q", "cross_attn.o",
                   "cross_attn.k_img", "cross_attn.v_img"],
    "self_attn":  ["self_attn.k", "self_attn.v", "self_attn.q", "self_attn.o"],
    "ffn":        ["ffn.0", "ffn.2"],
}

# Regex patterns for Z-Image main blocks (layers.0-N).
# Strips optional model.diffusion_model. prefix via (?:...) non-capturing group.
_ZIMAGE_LAYER_PATTERNS = {
    "attention.qkv":        re.compile(r"(?:model\.diffusion_model\.)?layers\.(\d+)\.attention\.qkv\.weight$"),
    "attention.out":        re.compile(r"(?:model\.diffusion_model\.)?layers\.(\d+)\.attention\.out\.weight$"),
    "feed_forward.w1":      re.compile(r"(?:model\.diffusion_model\.)?layers\.(\d+)\.feed_forward\.w1\.weight$"),
    "feed_forward.w2":      re.compile(r"(?:model\.diffusion_model\.)?layers\.(\d+)\.feed_forward\.w2\.weight$"),
    "feed_forward.w3":      re.compile(r"(?:model\.diffusion_model\.)?layers\.(\d+)\.feed_forward\.w3\.weight$"),
    "adaLN_modulation.0":   re.compile(r"(?:model\.diffusion_model\.)?layers\.(\d+)\.adaLN_modulation\.0\.weight$"),
}

_ZIMAGE_DETAIL_GROUPS = {
    "attention":     ["attention.qkv", "attention.out"],
    "feed_forward":  ["feed_forward.w1", "feed_forward.w2", "feed_forward.w3"],
    "adaLN":         ["adaLN_modulation.0"],
}

# Regex patterns for Z-Image refiners.
# These use a separate block namespace: context_refiner and noise_refiner.
# Block index captured in group 1, subgraph name in group 2 (for display).
_ZIMAGE_REFINER_PATTERNS = {
    "refiner.attention.qkv":   re.compile(r"(?:model\.diffusion_model\.)?(context_refiner|noise_refiner)\.(\d+)\.attention\.qkv\.weight$"),
    "refiner.attention.out":   re.compile(r"(?:model\.diffusion_model\.)?(context_refiner|noise_refiner)\.(\d+)\.attention\.out\.weight$"),
    "refiner.feed_forward.w1": re.compile(r"(?:model\.diffusion_model\.)?(context_refiner|noise_refiner)\.(\d+)\.feed_forward\.w1\.weight$"),
    "refiner.feed_forward.w2": re.compile(r"(?:model\.diffusion_model\.)?(context_refiner|noise_refiner)\.(\d+)\.feed_forward\.w2\.weight$"),
    "refiner.feed_forward.w3": re.compile(r"(?:model\.diffusion_model\.)?(context_refiner|noise_refiner)\.(\d+)\.feed_forward\.w3\.weight$"),
}

_ZIMAGE_REFINER_DETAIL_GROUPS = {
    "refiner.attention":    ["refiner.attention.qkv", "refiner.attention.out"],
    "refiner.feed_forward": ["refiner.feed_forward.w1", "refiner.feed_forward.w2", "refiner.feed_forward.w3"],
}

# Layers from ZIMAGE_LAYER_KEYNAMES excluded from analysis.
# These receive highprec treatment in convert_to_quant and are assumed to
# reflect an informed architectural decision by the model author.
ZIMAGE_HIGHPREC_EXCLUDED = [
    "x_embedder",
    "clip_text_pooled_proj",
    "final_layer",
    "cap_embedder.1",
    "adaLN_modulation",   # global (not layers.*)
    "t_embedder",
    "time_text_embed",
]

MODEL_CONFIGS = {
    "wan": {
        "description":        "Wan 2.1 video diffusion model",
        "convert_flag":       "--wan",
        "layer_patterns":     _WAN_LAYER_PATTERNS,
        "detail_groups":      _WAN_DETAIL_GROUPS,
        "refiner_patterns":   None,
        "refiner_detail_groups": None,
        "default_args":       _WAN_DEFAULT_ARGS,
        "highprec_excluded":  None,
    },
    "zimage": {
        "description":        "Z-Image / Z-Image Turbo image diffusion model",
        "convert_flag":       "--zimage / --zimage_refiner",
        "layer_patterns":     _ZIMAGE_LAYER_PATTERNS,
        "detail_groups":      _ZIMAGE_DETAIL_GROUPS,
        "refiner_patterns":   _ZIMAGE_REFINER_PATTERNS,
        "refiner_detail_groups": _ZIMAGE_REFINER_DETAIL_GROUPS,
        "default_args":       _ZIMAGE_DEFAULT_ARGS,
        "highprec_excluded":  ZIMAGE_HIGHPREC_EXCLUDED,
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TensorMetrics:
    """Metrics for a single weight tensor."""
    key: str                   # full safetensors key
    layer_type: str            # e.g. "attention.qkv"
    block_idx: int             # transformer block index
    subgraph: str              # e.g. "layers", "context_refiner", "noise_refiner"
    shape: Tuple[int, ...]     # tensor shape
    excess_kurtosis: float
    dynamic_range: float
    std: float
    outlier_pct: float
    aspect_ratio: float
    score: float = 0.0
    recommendation: str = ""
    reason: str = ""


@dataclass
class AggregatedMetrics:
    """Mean metrics for a group of tensors."""
    layer_type: str
    block_range: str
    subgraph: str
    count: int
    excess_kurtosis: float
    kurtosis_max: float
    dynamic_range: float
    std: float
    outlier_pct: float
    aspect_ratio: float
    score: float = 0.0
    recommendation: str = ""
    reason: str = ""
    spread_filtered: bool = False


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(tensor: torch.Tensor, outlier_sigma: float) -> Dict[str, float]:
    """
    Compute quantization sensitivity metrics for a 2D weight tensor.

    Excess kurtosis is used (kurtosis - 3), so a normal distribution gives 0.
    """
    t = tensor.to(torch.float32).flatten()

    mean = t.mean()
    std  = t.std()

    if std > 0:
        excess_kurtosis = ((t - mean) ** 4).mean() / (std ** 4) - 3.0
    else:
        excess_kurtosis = torch.tensor(0.0)

    dynamic_range = t.abs().max() - t.abs().min()

    if std > 0:
        outlier_mask = (t - mean).abs() > outlier_sigma * std
        outlier_pct  = outlier_mask.float().mean() * 100.0
    else:
        outlier_pct = torch.tensor(0.0)

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
# Score computation
# ---------------------------------------------------------------------------

def compute_scores(
    all_metrics:     List[TensorMetrics],
    kurtosis_weight: float,
    range_weight:    float,
    ar_weight:       float,
) -> None:
    """Compute combined sensitivity score for each tensor (in-place)."""
    kurt_vals = torch.tensor([m.excess_kurtosis for m in all_metrics])
    q25 = torch.quantile(kurt_vals, 0.25).item()
    q75 = torch.quantile(kurt_vals, 0.75).item()
    iqr = q75 - q25

    def norm_iqr(val: float) -> float:
        lo = q25 - 1.5 * iqr
        hi = q75 + 1.5 * iqr
        clipped = max(lo, min(hi, val))
        return (clipped - lo) / (hi - lo) if hi > lo else 0.0

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
    """Return (*KEEP*|FP8|NVFP4, reason) based on score thresholds."""
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

def _match_key(
    key: str,
    layer_patterns: Dict,
    refiner_patterns: Optional[Dict],
) -> Optional[Tuple[str, int, str]]:
    """
    Try to match a safetensors key against all configured patterns.

    Returns (layer_type, block_idx, subgraph) or None if no match.
    For refiner patterns, subgraph is "context_refiner" or "noise_refiner".
    For main patterns, subgraph is "layers" (Wan: "blocks").
    """
    for layer_type, pattern in layer_patterns.items():
        m = pattern.search(key)
        if m:
            block_idx = int(m.group(1))
            # Determine subgraph from key prefix
            if "context_refiner" in key:
                subgraph = "context_refiner"
            elif "noise_refiner" in key:
                subgraph = "noise_refiner"
            elif "blocks." in key:
                subgraph = "blocks"
            else:
                subgraph = "layers"
            return layer_type, block_idx, subgraph

    if refiner_patterns:
        for layer_type, pattern in refiner_patterns.items():
            m = pattern.search(key)
            if m:
                subgraph  = m.group(1)   # "context_refiner" or "noise_refiner"
                block_idx = int(m.group(2))
                return layer_type, block_idx, subgraph

    return None


def analyze_model(
    model_path:      str,
    device:          torch.device,
    outlier_sigma:   float,
    layer_patterns:  Dict,
    refiner_patterns: Optional[Dict],
    low_ram:         bool = False,
) -> List[TensorMetrics]:
    """
    Stream through the model file and compute metrics for all target tensors.
    """
    results: List[TensorMetrics] = []

    if low_ram:
        header, data_offset = read_safetensors_header(model_path)
        all_keys = [k for k in header.keys() if k != "__metadata__"]
    else:
        with safe_open(model_path, framework="pt", device="cpu") as f:
            all_keys = list(f.keys())

    # Identify matching keys
    matching: List[Tuple[str, str, int, str]] = []  # (key, layer_type, block_idx, subgraph)
    for key in all_keys:
        result = _match_key(key, layer_patterns, refiner_patterns)
        if result:
            layer_type, block_idx, subgraph = result
            matching.append((key, layer_type, block_idx, subgraph))

    total = len(matching)
    print(f"  Target tensors found: {total}")
    print()

    if low_ram:
        for i, (key, layer_type, block_idx, subgraph) in enumerate(matching, 1):
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
                subgraph        = subgraph,
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
            for i, (key, layer_type, block_idx, subgraph) in enumerate(matching, 1):
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
                    subgraph        = subgraph,
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
    n_extreme = max(1, round(total_blocks * extreme_pct / 100.0))
    low  = (0, n_extreme - 1)
    high = (total_blocks - n_extreme, total_blocks - 1)
    return low, high


def classify_block(
    block_idx:    int,
    extreme_low:  Tuple[int, int],
    extreme_high: Tuple[int, int],
) -> str:
    if extreme_low[0] <= block_idx <= extreme_low[1]:
        return "extreme_low"
    if extreme_high[0] <= block_idx <= extreme_high[1]:
        return "extreme_high"
    return "middle"


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def block_range_label(block_indices: List[int], total_blocks: int) -> str:
    lo, hi = min(block_indices), max(block_indices)
    if lo == 0 and hi == total_blocks - 1:
        return f"0–{hi}"
    return f"{lo}–{hi}"


def aggregate(
    metrics_list: List[TensorMetrics],
    layer_type:   str,
    block_range:  str,
    subgraph:     str,
) -> AggregatedMetrics:
    n = len(metrics_list)
    if n == 0:
        return AggregatedMetrics(layer_type, block_range, subgraph, 0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return AggregatedMetrics(
        layer_type      = layer_type,
        block_range     = block_range,
        subgraph        = subgraph,
        count           = n,
        excess_kurtosis = sum(m.excess_kurtosis for m in metrics_list) / n,
        kurtosis_max    = max(m.excess_kurtosis for m in metrics_list),
        dynamic_range   = sum(m.dynamic_range   for m in metrics_list) / n,
        std             = sum(m.std             for m in metrics_list) / n,
        outlier_pct     = sum(m.outlier_pct     for m in metrics_list) / n,
        aspect_ratio    = metrics_list[0].aspect_ratio,
        score           = sum(m.score           for m in metrics_list) / n,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

SEP = "─" * 115


def fmt_row_summary(agg: AggregatedMetrics) -> str:
    return (
        f"  {agg.layer_type:<28} {agg.block_range:<10} "
        f"{agg.excess_kurtosis:>14.3f}  {agg.dynamic_range:>10.3f}  "
        f"{agg.std:>6.4f}  {agg.outlier_pct:>9.1f}%  "
        f"{agg.aspect_ratio:>5.2f}  {agg.score:>6.3f}  {agg.recommendation}"
    )


def fmt_row_detail(agg: AggregatedMetrics) -> str:
    marker = ""
    if agg.spread_filtered:
        marker = "  [→NVFP4 ~spread]" if agg.recommendation == "FP8" else "  [→FP8 ~spread]"
    return (
        f"  {agg.layer_type:<28} {agg.block_range:<12} "
        f"{agg.excess_kurtosis:>14.3f}  {agg.kurtosis_max:>10.3f}  {agg.dynamic_range:>10.3f}  "
        f"{agg.std:>6.4f}  {agg.outlier_pct:>9.1f}%  "
        f"{agg.score:>6.3f}  {agg.recommendation}{marker}"
    )


def print_summary_table(aggregated: List[AggregatedMetrics], title_suffix: str = "") -> None:
    print()
    print(SEP)
    title = "  SUMMARY BY LAYER TYPE — weight tensors"
    if title_suffix:
        title += f"  [{title_suffix}]"
    print(title)
    print(SEP)
    print(
        f"  {'Layer':<28} {'Blocks':<10} {'Excess kurtosis':>14}  {'Dyn. range':>10}  "
        f"{'Std':>6}  {'Outliers%':>9}  {'AR':>5}  {'Score':>6}  Recommendation"
    )
    print(SEP)
    for agg in aggregated:
        print(fmt_row_summary(agg))
    print(SEP)


def print_detail_table(group_name: str, rows: List[AggregatedMetrics],
                       title_suffix: str = "") -> None:
    print()
    print(SEP)
    title = f"  DETAIL: {group_name} — weight tensors by block position"
    if title_suffix:
        title += f"  [{title_suffix}]"
    print(title)
    print(SEP)
    print(
        f"  {'Layer':<28} {'Blocks':<12} {'Excess kurtosis':>14}  {'Kurt. max':>10}  {'Dyn. range':>10}  "
        f"{'Std':>6}  {'Outliers%':>9}  {'Score':>6}  Recommendation"
    )
    print(SEP)
    for row in rows:
        print(fmt_row_detail(row))
    print(SEP)


def print_highprec_notice(excluded: List[str], convert_flag: str) -> None:
    """Print notice about layers excluded from analysis due to highprec treatment."""
    print()
    print(SEP)
    print("  LAYERS EXCLUDED FROM ANALYSIS")
    print(SEP)
    print(f"  The following layers are NOT included in this analysis because they")
    print(f"  receive special high-precision treatment in convert_to_quant")
    print(f"  ({convert_flag}) and are assumed to reflect an informed architectural")
    print(f"  decision by the model author. Consult convert_to_quant documentation")
    print(f"  for details on their handling.")
    print()
    for name in excluded:
        print(f"    - {name}")
    print(SEP)


# ---------------------------------------------------------------------------
# Layer type to key pattern (for regex output)
# ---------------------------------------------------------------------------

def _layer_type_to_key_pattern_wan(layer_type: str) -> str:
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


def _layer_type_to_key_pattern_zimage(layer_type: str) -> str:
    mapping = {
        "attention.qkv":      r"attention\.qkv\.weight",
        "attention.out":      r"attention\.out\.weight",
        "feed_forward.w1":    r"feed_forward\.w1\.weight",
        "feed_forward.w2":    r"feed_forward\.w2\.weight",
        "feed_forward.w3":    r"feed_forward\.w3\.weight",
        "adaLN_modulation.0": r"adaLN_modulation\.0\.weight",
        # Refiner types
        "refiner.attention.qkv":   r"attention\.qkv\.weight",
        "refiner.attention.out":   r"attention\.out\.weight",
        "refiner.feed_forward.w1": r"feed_forward\.w1\.weight",
        "refiner.feed_forward.w2": r"feed_forward\.w2\.weight",
        "refiner.feed_forward.w3": r"feed_forward\.w3\.weight",
    }
    return mapping.get(layer_type, layer_type)


def _block_range_to_indices(block_range: str) -> List[int]:
    normalized = block_range.replace("–", "-").replace("—", "-")
    parts = normalized.split("-")
    lo, hi = int(parts[0]), int(parts[1])
    return list(range(lo, hi + 1))


def _blocks_to_alternation(indices: List[int]) -> str:
    return "|".join(str(i) for i in sorted(indices))


# ---------------------------------------------------------------------------
# Suggested convert_to_quant parameters
# ---------------------------------------------------------------------------

def build_convert_to_quant_params(
    all_detail_rows:      List[AggregatedMetrics],
    all_metrics:          List[TensorMetrics],
    min_group_spread:     float,
    spread_filter_exempt: set,
    model:                str,
) -> Tuple[List[Tuple[str, List[int], str]], List[Tuple[str, List[int], str]]]:
    """
    Build FP8 and *KEEP* recommendations for convert_to_quant.

    Returns lists of (layer_type, block_indices, subgraph) tuples.
    """
    individual_rec: Dict[Tuple[str, int, str], str] = {
        (m.layer_type, m.block_idx, m.subgraph): m.recommendation for m in all_metrics
    }

    by_layer_type: Dict[str, List[AggregatedMetrics]] = defaultdict(list)
    for row in all_detail_rows:
        by_layer_type[row.layer_type].append(row)

    low_spread_types: set = set()
    for layer_type, rows in by_layer_type.items():
        if layer_type in spread_filter_exempt:
            continue
        scores = [r.score for r in rows]
        spread = max(scores) - min(scores)
        if spread < min_group_spread:
            low_spread_types.add(layer_type)

    fp8_entries:  List[Tuple[str, List[int], str]] = []
    keep_entries: List[Tuple[str, List[int], str]] = []

    for row in all_detail_rows:
        indices  = _block_range_to_indices(row.block_range)
        subgraph = row.subgraph

        if row.layer_type in spread_filter_exempt:
            keep_idxs, fp8_idxs = [], []
            for idx in indices:
                rec = individual_rec.get((row.layer_type, idx, subgraph), "NVFP4")
                if rec == "*KEEP*":
                    keep_idxs.append(idx)
                elif rec == "FP8":
                    fp8_idxs.append(idx)
            if keep_idxs:
                keep_entries.append((row.layer_type, keep_idxs, subgraph))
            if fp8_idxs:
                fp8_entries.append((row.layer_type, fp8_idxs, subgraph))

        elif row.layer_type in low_spread_types and row.recommendation == "FP8":
            row.spread_filtered = True

        elif row.layer_type in low_spread_types and row.recommendation == "*KEEP*":
            row.spread_filtered = True
            fp8_entries.append((row.layer_type, indices, subgraph))

        elif row.recommendation == "FP8":
            fp8_entries.append((row.layer_type, indices, subgraph))

        elif row.recommendation == "*KEEP*":
            keep_idxs, fp8_idxs = [], []
            for idx in indices:
                rec = individual_rec.get((row.layer_type, idx, subgraph), row.recommendation)
                if rec == "*KEEP*":
                    keep_idxs.append(idx)
                elif rec == "FP8":
                    fp8_idxs.append(idx)
            if keep_idxs:
                keep_entries.append((row.layer_type, keep_idxs, subgraph))
            if fp8_idxs:
                fp8_entries.append((row.layer_type, fp8_idxs, subgraph))

    return fp8_entries, keep_entries


def _build_regex_for_entries(
    entries:  List[Tuple[str, List[int], str]],
    model:    str,
) -> Optional[str]:
    """Build a single alternation regex from (layer_type, block_indices, subgraph) tuples."""
    if not entries:
        return None

    if model == "wan":
        layer_type_to_key = _layer_type_to_key_pattern_wan
        block_container   = "blocks"
    else:
        layer_type_to_key = _layer_type_to_key_pattern_zimage
        block_container   = None  # varies by subgraph

    patterns = []
    for layer_type, indices, subgraph in entries:
        key_pat   = layer_type_to_key(layer_type)
        block_alt = _blocks_to_alternation(indices)

        if model == "wan":
            patterns.append(rf"blocks\.({block_alt})\.{key_pat}")
        else:
            if subgraph in ("context_refiner", "noise_refiner"):
                patterns.append(rf"{subgraph}\.({block_alt})\.{key_pat}")
            else:
                patterns.append(rf"layers\.({block_alt})\.{key_pat}")

    return "|".join(patterns)


def print_suggested_params(
    fp8_entries:      List[Tuple[str, List[int], str]],
    keep_entries:     List[Tuple[str, List[int], str]],
    fp8_min_score:    float,
    min_group_spread: float,
    model:            str,
    convert_flag:     str,
) -> None:
    print()
    print(SEP)
    print("  SUGGESTED convert_to_quant PARAMETERS")
    print(SEP)
    print(f"  Model: {model}  |  convert_to_quant flag: {convert_flag}")
    print(f"  Filters active — FP8 min score: {fp8_min_score}  |  min group spread: {min_group_spread}")
    print()

    fp8_regex = _build_regex_for_entries(fp8_entries, model)
    if fp8_regex:
        print(f'  --custom-layers "{fp8_regex}"')
        print( "  --custom-type fp8")
    else:
        print("  No layers recommended as FP8 (all NVFP4, or filtered by min score / spread)")

    print()

    keep_regex = _build_regex_for_entries(keep_entries, model)
    if keep_regex:
        print(f'  --exclude-layers "{keep_regex}"')
    else:
        print("  No additional --exclude-layers needed (model flag covers existing exclusions)")

    print(SEP)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(
    all_metrics:      List[TensorMetrics],
    output_path:      str,
    effective_rec:    Dict[Tuple[str, int, str], str],
    effective_reason: Dict[Tuple[str, int, str], str],
) -> None:
    fieldnames = [
        "key", "layer_type", "block_idx", "subgraph",
        "rows", "cols",
        "excess_kurtosis", "dynamic_range", "std", "outlier_pct",
        "aspect_ratio", "score", "recommendation", "effective_recommendation",
        "reason",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            key = (m.layer_type, m.block_idx, m.subgraph)
            eff = effective_rec.get(key, m.recommendation)
            rsn = effective_reason.get(key, m.reason)
            writer.writerow({
                "key":                      m.key,
                "layer_type":               m.layer_type,
                "block_idx":                m.block_idx,
                "subgraph":                 m.subgraph,
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

_BYTES_PER_WEIGHT: Dict[str, float] = {
    "*KEEP*": 2.0,
    "FP8":    1.0,
    "NVFP4":  0.5625,
}


def estimate_output_size(
    all_metrics:    List[TensorMetrics],
    effective_rec:  Dict[Tuple[str, int, str], str],
    original_bytes: int,
) -> Dict:
    in_scope_original  = 0
    in_scope_estimated = 0
    per_format_bytes:  Dict[str, float] = {"*KEEP*": 0.0, "FP8": 0.0, "NVFP4": 0.0}
    per_format_counts: Dict[str, int]   = {"*KEEP*": 0,   "FP8": 0,   "NVFP4": 0}

    for m in all_metrics:
        n_weights = m.shape[0] * m.shape[1]
        in_scope_original += n_weights * 2

        fmt = effective_rec.get((m.layer_type, m.block_idx, m.subgraph), m.recommendation)
        bpw = _BYTES_PER_WEIGHT.get(fmt, 2.0)
        tensor_bytes = n_weights * bpw
        in_scope_estimated          += tensor_bytes
        per_format_bytes[fmt]        = per_format_bytes.get(fmt, 0.0) + tensor_bytes
        per_format_counts[fmt]       = per_format_counts.get(fmt, 0)  + 1

    out_of_scope    = original_bytes - in_scope_original
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
# Effective recommendation builder (shared between CSV and size estimate)
# ---------------------------------------------------------------------------

def build_effective_rec(
    all_detail_rows:      List[AggregatedMetrics],
    all_metrics:          List[TensorMetrics],
    spread_filter_exempt: set,
) -> Tuple[Dict, Dict]:
    """Build (effective_rec, effective_reason) dicts keyed by (layer_type, block_idx, subgraph)."""
    individual_rec: Dict[Tuple[str, int, str], str] = {
        (m.layer_type, m.block_idx, m.subgraph): m.recommendation for m in all_metrics
    }
    effective_rec:    Dict[Tuple[str, int, str], str] = {}
    effective_reason: Dict[Tuple[str, int, str], str] = {}

    for row in all_detail_rows:
        indices  = _block_range_to_indices(row.block_range)
        subgraph = row.subgraph

        if row.layer_type in spread_filter_exempt:
            for idx in indices:
                k = (row.layer_type, idx, subgraph)
                effective_rec[k] = individual_rec.get(k, "NVFP4")

        elif row.spread_filtered:
            eff = "NVFP4" if row.recommendation == "FP8" else "FP8"
            for idx in indices:
                k = (row.layer_type, idx, subgraph)
                effective_rec[k]    = eff
                effective_reason[k] = "spread_demotion"

        elif row.recommendation == "*KEEP*":
            for idx in indices:
                k       = (row.layer_type, idx, subgraph)
                ind_rec = individual_rec.get(k, row.recommendation)
                effective_rec[k] = ind_rec
                if ind_rec != "*KEEP*":
                    effective_reason[k] = "group_keep_resolved"

        else:
            for idx in indices:
                k = (row.layer_type, idx, subgraph)
                effective_rec[k] = row.recommendation
                ind_rec = individual_rec.get(k, row.recommendation)
                if ind_rec != row.recommendation:
                    if row.recommendation == "FP8" and ind_rec == "NVFP4":
                        effective_reason[k] = "group_fp8_promotion"
                    else:
                        effective_reason[k] = "group_spread_demotion"

    return effective_rec, effective_reason


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantization sensitivity analysis for safetensors models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported models:
  wan     - Wan 2.1 video diffusion model
  zimage  - Z-Image / Z-Image Turbo image diffusion model

Score composition:
  - Excess kurtosis (IQR-normalized globally)       default weight: 0.6
  - Dynamic range (within-type min-max normalized)  default weight: 0.3
  - Aspect ratio  (within-type min-max normalized)  default weight: 0.1
  Outlier% is shown in tables but excluded from score (correlated with kurtosis).

Thresholds and extreme block ranges are derived automatically from the model.

Default thresholds are calibrated per model:
  wan    - calibrated from Wan 2.1 14B score distribution
  zimage - calibrated empirically from Z-Image Turbo score distribution
""",
    )
    parser.add_argument("model_path", type=str,
                        help="Path to the BF16 safetensors model file.")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model architecture to analyze.")
    parser.add_argument("--csv", type=str, default=None, metavar="OUTPUT.csv",
                        help="Export per-tensor metrics to a CSV file.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for computation: 'cpu' or 'cuda' (default: auto-detect).")
    parser.add_argument("--outlier-sigma", type=float, default=3.0, metavar="N",
                        help="Standard deviation multiplier for outlier detection (default: 3.0).")
    parser.add_argument("--fp8-percentile", type=float, default=None, metavar="P",
                        help="Score percentile threshold for FP8 recommendation (default: model-specific).")
    parser.add_argument("--keep-percentile", type=float, default=None, metavar="P",
                        help="Score percentile threshold for *KEEP* recommendation (default: model-specific).")
    parser.add_argument("--extreme-pct", type=float, default=10.0, metavar="P",
                        help="Percentage of blocks to consider extreme at each end (default: 10).")
    parser.add_argument("--kurtosis-weight", type=float, default=0.6, metavar="W",
                        help="Weight for IQR-normalized excess kurtosis in combined score (default: 0.6).")
    parser.add_argument("--range-weight", type=float, default=0.3, metavar="W",
                        help="Weight for within-type dynamic range in combined score (default: 0.3).")
    parser.add_argument("--ar-weight", type=float, default=0.1, metavar="W",
                        help="Weight for within-type aspect ratio in combined score (default: 0.1).")
    parser.add_argument("--fp8-min-score", type=float, default=None, metavar="S",
                        help="Absolute minimum score required for FP8 (default: model-specific).")
    parser.add_argument("--min-group-spread", type=float, default=None, metavar="S",
                        help="Minimum score spread across block position groups for per-group FP8 (default: model-specific).")
    parser.add_argument("--spread-filter-exempt", nargs="*", default=[],
                        metavar="LAYER_TYPE",
                        help="Layer types that bypass the spread filter.")
    parser.add_argument("--kurtosis-keep", type=float, default=None, metavar="K",
                        help="Hard floor on excess kurtosis: forces *KEEP* (default: model-specific).")
    parser.add_argument("--lowram", action="store_true", default=False,
                        help="Avoid mmap-ing the full model file.")

    args = parser.parse_args()

    # Validate input file
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}")
        sys.exit(1)

    # Validate weights sum to 1
    total_weight = args.kurtosis_weight + args.range_weight + args.ar_weight
    if abs(total_weight - 1.0) > 1e-6:
        print(f"Error: --kurtosis-weight + --range-weight + --ar-weight must sum to 1.0 (got {total_weight:.4f})")
        sys.exit(1)

    # Load model config
    cfg         = MODEL_CONFIGS[args.model]
    defaults    = cfg["default_args"]
    convert_flag = cfg["convert_flag"]

    # Resolve defaults: CLI > model-specific > Wan fallback (with warning)
    def resolve(cli_val, key):
        if cli_val is not None:
            return cli_val, False   # (value, used_fallback)
        model_val = defaults[key]
        if model_val is not None:
            return model_val, False
        # Fall back to Wan calibrated values
        wan_val = _WAN_DEFAULT_ARGS[key]
        return wan_val, True

    fp8_percentile,   _ = resolve(args.fp8_percentile,   "fp8_percentile")
    keep_percentile,  _ = resolve(args.keep_percentile,  "keep_percentile")
    kurtosis_keep,    _ = resolve(args.kurtosis_keep,     "kurtosis_keep")
    fp8_min_score,    _ = resolve(args.fp8_min_score,     "fp8_min_score")
    min_group_spread, _ = resolve(args.min_group_spread,  "min_group_spread")

    # Validate spread_filter_exempt layer types
    all_layer_types = set(cfg["layer_patterns"].keys())
    if cfg["refiner_patterns"]:
        all_layer_types |= set(cfg["refiner_patterns"].keys())
    for lt in args.spread_filter_exempt:
        if lt not in all_layer_types:
            print(f"Error: --spread-filter-exempt: unknown layer type '{lt}' for model '{args.model}'.")
            print(f"       Valid types: {', '.join(sorted(all_layer_types))}")
            sys.exit(1)

    # Resolve device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        props      = torch.cuda.get_device_properties(device)
        vram_mb    = props.total_memory // (1024 * 1024)
        device_str = f"CUDA ({props.name}, {vram_mb} MB)"
    else:
        device_str = "CPU"

    print()
    print(f"  Quantization Sensitivity Analysis — {model_path.name}")
    print(f"  Model: {args.model}  ({cfg['description']})")
    print(f"  Device: {device_str}")

    exempt_str = ", ".join(sorted(args.spread_filter_exempt)) if args.spread_filter_exempt else "none"
    print(f"  Outlier sigma: {args.outlier_sigma}  |  "
          f"FP8 percentile: {fp8_percentile}  |  "
          f"Keep percentile: {keep_percentile}  |  "
          f"FP8 min score: {fp8_min_score}  |  "
          f"Min group spread: {min_group_spread}  |  "
          f"Spread filter exempt: {exempt_str}  |  "
          f"Extreme blocks: {args.extreme_pct}%  |  "
          f"Kurtosis keep: {kurtosis_keep}  |  "
          f"Low-RAM: {'yes' if args.lowram else 'no'}")
    print()

    # Print highprec excluded notice for models that have it
    if cfg["highprec_excluded"]:
        print_highprec_notice(cfg["highprec_excluded"], convert_flag)

    print()
    print("  Scanning model...")

    # Analysis pass
    all_metrics = analyze_model(
        str(model_path),
        device,
        args.outlier_sigma,
        cfg["layer_patterns"],
        cfg["refiner_patterns"],
        low_ram=args.lowram,
    )

    if not all_metrics:
        print("Error: no target tensors found in model.")
        sys.exit(1)

    # Split metrics by subgraph
    main_subgraphs    = {"blocks", "layers"}
    refiner_subgraphs = {"context_refiner", "noise_refiner"}

    main_metrics    = [m for m in all_metrics if m.subgraph in main_subgraphs]
    refiner_metrics = [m for m in all_metrics if m.subgraph in refiner_subgraphs]

    # Total blocks from main subgraph only (for extreme range computation)
    if main_metrics:
        total_blocks = max(m.block_idx for m in main_metrics) + 1
    else:
        total_blocks = 0

    print(f"  Tensors analyzed: {len(all_metrics)} | Main blocks: {total_blocks}"
          + (f" | Refiner tensors: {len(refiner_metrics)}" if refiner_metrics else ""))

    # AR inactive check
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
          f"aspect ratio: {args.ar_weight}{ar_note}")

    # Extreme block ranges (main blocks only)
    if total_blocks > 0:
        extreme_low, extreme_high = compute_extreme_ranges(total_blocks, args.extreme_pct)
        print(f"  Extreme blocks — low: {extreme_low[0]}–{extreme_low[1]}  |  "
              f"high: {extreme_high[0]}–{extreme_high[1]}")

    # Compute scores (globally across all metrics)
    compute_scores(all_metrics, args.kurtosis_weight, args.range_weight, args.ar_weight)

    # Compute thresholds
    fp8_threshold, keep_threshold = compute_auto_thresholds(
        all_metrics, fp8_percentile, keep_percentile
    )
    if fp8_min_score > fp8_threshold:
        print(f"  Auto thresholds — FP8: score >= {fp8_min_score:.3f} "
              f"[fp8-min-score active; percentile gave {fp8_threshold:.3f}]  |  "
              f"*KEEP*: score >= {keep_threshold:.3f}")
    else:
        print(f"  Auto thresholds — FP8: score >= {fp8_threshold:.3f} [percentile]  |  "
              f"*KEEP*: score >= {keep_threshold:.3f}")

    # Assign recommendations
    for m in all_metrics:
        m.recommendation, m.reason = assign_recommendation(
            m.score, fp8_threshold, keep_threshold, fp8_min_score,
            excess_kurtosis=m.excess_kurtosis,
            kurtosis_keep=kurtosis_keep,
        )

    # -----------------------------------------------------------------------
    # Summary table — main blocks
    # -----------------------------------------------------------------------
    by_layer: Dict[str, List[TensorMetrics]] = defaultdict(list)
    for m in main_metrics:
        by_layer[m.layer_type].append(m)

    summary_aggs: List[AggregatedMetrics] = []
    for layer_type in cfg["layer_patterns"].keys():
        group = by_layer.get(layer_type, [])
        if not group:
            continue
        block_indices = [m.block_idx for m in group]
        agg = aggregate(group, layer_type,
                        block_range_label(block_indices, total_blocks),
                        "layers")
        agg.recommendation, agg.reason = assign_recommendation(
            agg.score, fp8_threshold, keep_threshold, fp8_min_score,
            excess_kurtosis=agg.kurtosis_max,
            kurtosis_keep=kurtosis_keep,
        )
        summary_aggs.append(agg)

    print_summary_table(summary_aggs, title_suffix="main blocks")

    # -----------------------------------------------------------------------
    # Summary table — refiners (if present)
    # -----------------------------------------------------------------------
    refiner_summary_aggs: List[AggregatedMetrics] = []
    if refiner_metrics and cfg["refiner_patterns"]:
        by_refiner_layer: Dict[str, List[TensorMetrics]] = defaultdict(list)
        for m in refiner_metrics:
            by_refiner_layer[m.layer_type].append(m)

        for layer_type in cfg["refiner_patterns"].keys():
            group = by_refiner_layer.get(layer_type, [])
            if not group:
                continue
            block_indices = [m.block_idx for m in group]
            # Use total refiner blocks per subgraph for label (max 2)
            total_refiner = max(block_indices) + 1
            agg = aggregate(group, layer_type,
                            block_range_label(block_indices, total_refiner),
                            "refiners")
            agg.recommendation, agg.reason = assign_recommendation(
                agg.score, fp8_threshold, keep_threshold, fp8_min_score,
                excess_kurtosis=agg.kurtosis_max,
                kurtosis_keep=kurtosis_keep,
            )
            refiner_summary_aggs.append(agg)

        print_summary_table(refiner_summary_aggs,
                            title_suffix="context_refiner + noise_refiner")

    # -----------------------------------------------------------------------
    # Detail tables — main blocks (with positional classification)
    # -----------------------------------------------------------------------
    all_detail_rows: List[AggregatedMetrics] = []
    detail_rows_by_group: Dict[str, List[AggregatedMetrics]] = {}

    if total_blocks > 0:
        for group_name, layer_types in cfg["detail_groups"].items():
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
                    agg     = aggregate(subset, layer_type, label, "layers")
                    agg.recommendation, agg.reason = assign_recommendation(
                        agg.score, fp8_threshold, keep_threshold, fp8_min_score,
                        excess_kurtosis=agg.kurtosis_max,
                        kurtosis_keep=kurtosis_keep,
                    )
                    group_rows.append(agg)

            if group_rows:
                detail_rows_by_group[group_name] = group_rows
                all_detail_rows.extend(group_rows)

    # -----------------------------------------------------------------------
    # Detail tables — refiners (flat, no positional classification)
    # -----------------------------------------------------------------------
    refiner_detail_rows: List[AggregatedMetrics] = []
    if refiner_metrics and cfg["refiner_detail_groups"]:
        by_refiner_layer_sg: Dict[Tuple[str, str], List[TensorMetrics]] = defaultdict(list)
        for m in refiner_metrics:
            by_refiner_layer_sg[(m.layer_type, m.subgraph)].append(m)

        refiner_detail_by_group: Dict[str, List[AggregatedMetrics]] = {}

        for group_name, layer_types in cfg["refiner_detail_groups"].items():
            group_rows = []
            for layer_type in layer_types:
                for subgraph in ("context_refiner", "noise_refiner"):
                    group = by_refiner_layer_sg.get((layer_type, subgraph), [])
                    if not group:
                        continue
                    block_indices  = [m.block_idx for m in group]
                    total_r        = max(block_indices) + 1
                    label          = block_range_label(block_indices, total_r)
                    agg            = aggregate(group, layer_type, label, subgraph)
                    agg.recommendation, agg.reason = assign_recommendation(
                        agg.score, fp8_threshold, keep_threshold, fp8_min_score,
                        excess_kurtosis=agg.kurtosis_max,
                        kurtosis_keep=kurtosis_keep,
                    )
                    group_rows.append(agg)

            if group_rows:
                refiner_detail_by_group[group_name] = group_rows
                refiner_detail_rows.extend(group_rows)

        all_detail_rows.extend(refiner_detail_rows)

    # -----------------------------------------------------------------------
    # Suggested parameters
    # (must run before printing detail tables so spread_filtered flags are set)
    # -----------------------------------------------------------------------
    fp8_entries, keep_entries = build_convert_to_quant_params(
        all_detail_rows, all_metrics,
        min_group_spread,
        spread_filter_exempt=set(args.spread_filter_exempt),
        model=args.model,
    )

    # Print detail tables (main blocks)
    for group_name, group_rows in detail_rows_by_group.items():
        print_detail_table(group_name, group_rows, title_suffix="main blocks")

    # Print detail tables (refiners)
    if refiner_metrics and cfg["refiner_detail_groups"]:
        for group_name, group_rows in refiner_detail_by_group.items():
            print_detail_table(group_name, group_rows,
                               title_suffix="refiners — no positional classification")

    print_suggested_params(
        fp8_entries, keep_entries,
        fp8_min_score, min_group_spread,
        args.model, convert_flag,
    )

    # -----------------------------------------------------------------------
    # Effective recommendation (shared for CSV and size estimate)
    # -----------------------------------------------------------------------
    effective_rec, effective_reason = build_effective_rec(
        all_detail_rows, all_metrics,
        spread_filter_exempt=set(args.spread_filter_exempt),
    )

    # CSV export
    if args.csv:
        export_csv(all_metrics, args.csv, effective_rec, effective_reason)

    # Size estimate
    original_bytes = model_path.stat().st_size
    size_est = estimate_output_size(all_metrics, effective_rec, original_bytes)
    print_size_estimate(size_est)


if __name__ == "__main__":
    main()
