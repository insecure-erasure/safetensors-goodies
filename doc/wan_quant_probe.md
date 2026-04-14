# wan_quant_probe.py

A command-line tool to analyze weight tensors in Wan 2.x diffusion model checkpoints and recommend a quantization format for each layer — `*KEEP*` (BF16), FP8, or NVFP4 — before running `convert_to_quant`. Designed to feed directly into [ComfyUI](https://github.com/comfyanonymous/ComfyUI) quantization workflows.

## Features

- **Per-tensor sensitivity scoring** based on excess kurtosis, dynamic range, and aspect ratio
- **Automatic thresholds** derived from the model's own score distribution — no manual tuning required
- **Kurtosis hard floor** (`--kurtosis-keep`) to protect extremely leptokurtic tensors that score-based thresholds would miss
- **Spread filter** to suppress per-group FP8 recommendations when score variation across block positions is too low to be meaningful
- **Spread filter exemptions** (`--spread-filter-exempt`) for layer types where individual tensor decisions should always take precedence over group-level ones
- **Suggested `convert_to_quant` parameters** printed directly — `--custom-layers` and `--exclude-layers` regexes ready to copy
- **Estimated output file size** after applying recommendations, broken down by format
- **CSV export** with per-tensor metrics, raw recommendation, effective recommendation after all filters, and decision reason
- **`--lowram` mode** for systems where mmap-ing the full model file would exhaust virtual memory
- GPU-accelerated metric computation with automatic CPU fallback

## Requirements

```
torch
safetensors
```

```
pip install torch safetensors
```

## Usage

### Basic analysis

```
python wan_quant_probe.py model.safetensors
```

### Export metrics to CSV

```
python wan_quant_probe.py model.safetensors --csv results.csv
```

### Protect cross-attention K and Q layers from group-level overrides

Based on the Wan 2.1 paper, cross-attention K and Q projections are numerically sensitive — the $QK^T$ operation is the primary source of precision loss in quantized attention. Exempting them from the spread filter ensures their quantization format is always decided at individual tensor level:

```
python wan_quant_probe.py model.safetensors --csv results.csv \
    --spread-filter-exempt cross_attn.k cross_attn.q
```

### Run on CPU or limit to CPU

```
python wan_quant_probe.py model.safetensors --device cpu
```

### Low-RAM mode

Use this on systems without enough memory to mmap the full model file (e.g. 16 GB RAM without swap when analyzing a 28 GB model):

```
python wan_quant_probe.py model.safetensors --lowram
```

### Adjust quantization thresholds

```
# More conservative: protect more layers as FP8 instead of NVFP4
python wan_quant_probe.py model.safetensors --fp8-percentile 60

# Less conservative: protect fewer layers in BF16
python wan_quant_probe.py model.safetensors --keep-percentile 95
```

## Scope

The script analyzes the following layer types across all transformer blocks:

| Group | Layers |
| --- | --- |
| `cross_attn` | `k`, `v`, `q`, `o`, `k_img`, `v_img` |
| `self_attn` | `k`, `v`, `q`, `o` |
| `ffn` | `0`, `2` |

This scope is disjoint from the layers that `convert_to_quant` protects by construction (norm, bias, patch embedding, text/time embeddings, head). No explicit guards are needed.

## How recommendations are assigned

Each tensor receives a sensitivity score combining three metrics:

- **Excess kurtosis** (weight 0.6) — IQR-normalized globally; detects heavy-tailed distributions prone to quantization error
- **Dynamic range** (weight 0.3) — within-type min-max normalized; detects wide value spreads
- **Aspect ratio** (weight 0.1) — within-type normalized; inactive when all tensors of a type share the same shape, which is standard in transformer architectures

Scores are compared against two percentile-derived thresholds:

- Score ≥ `keep-percentile` → `*KEEP*` (stay in BF16)
- Score ≥ `fp8-percentile` and ≥ `fp8-min-score` → FP8
- Otherwise → NVFP4

Two additional guards override the score:

- **Kurtosis hard floor** — tensors with excess kurtosis ≥ `--kurtosis-keep` are forced to `*KEEP*` regardless of score. Protects layers whose distribution is so extreme that IQR normalization saturates before the score can reach the keep threshold.
- **Spread filter** — when score variation across block position groups (first blocks, middle, last blocks) is below `--min-group-spread`, per-group recommendations are suppressed and all tensors of that type fall back to NVFP4. Layer types listed in `--spread-filter-exempt` bypass this filter entirely.

## CSV output

The exported CSV includes one row per analyzed tensor with the following columns:

| Column | Description |
| --- | --- |
| `key` | Full safetensors key |
| `layer_type` | Layer group (e.g. `cross_attn.k`) |
| `block_idx` | Transformer block index |
| `rows`, `cols` | Tensor shape |
| `excess_kurtosis` | Corrected excess kurtosis (0 = normal distribution) |
| `dynamic_range` | abs(max) − abs(min) |
| `std` | Standard deviation |
| `outlier_pct` | % of values beyond `outlier-sigma` × std |
| `aspect_ratio` | max(rows, cols) / min(rows, cols) |
| `score` | Combined sensitivity score |
| `recommendation` | Raw recommendation before group-level filters |
| `effective_recommendation` | Final recommendation after all filters |
| `reason` | Decision reason (see below) |

### `reason` values

| Value | Meaning |
| --- | --- |
| `score_percentile` | `*KEEP*` or FP8 driven by percentile threshold |
| `score_below_fp8_min` | NVFP4 because `fp8-min-score` guard blocked FP8 |
| `default` | NVFP4 because score is below all thresholds |
| `kurtosis_floor` | `*KEEP*` forced because excess kurtosis ≥ `kurtosis-keep` |
| `group_keep_resolved` | Tensor was individually `*KEEP*` but demoted during per-tensor resolution within a `*KEEP*` group |
| `group_fp8_promotion` | Tensor was individually NVFP4 but carried up to FP8 by its block-position group |
| `group_spread_demotion` | Tensor was individually FP8 or `*KEEP*` but brought down by the group's spread filter result |
| `spread_demotion` | FP8→NVFP4 or `*KEEP*`→FP8 by spread filter directly |

## CLI reference

| Argument | Default | Description |
| --- | --- | --- |
| `model` | *(required)* | Path to the BF16 safetensors model file |
| `--csv` | — | Export per-tensor metrics to a CSV file |
| `--device` | auto | Computation device: `cpu` or `cuda` |
| `--lowram` | off | Avoid full-file mmap; load tensors individually by byte offset |
| `--fp8-percentile` | 75 | Score percentile threshold for FP8 recommendation |
| `--keep-percentile` | 90 | Score percentile threshold for `*KEEP*` recommendation |
| `--fp8-min-score` | 0.5 | Absolute minimum score required for FP8; set to 0.0 to disable |
| `--kurtosis-keep` | 8.0 | Excess kurtosis hard floor for forced `*KEEP*`; set to `inf` to disable |
| `--min-group-spread` | 0.06 | Minimum score spread across block position groups for per-group FP8; set to 0.0 to disable |
| `--spread-filter-exempt` | — | Layer types that bypass the spread filter (e.g. `cross_attn.k cross_attn.q`) |
| `--extreme-pct` | 10 | Percentage of blocks considered extreme at each end |
| `--outlier-sigma` | 3.0 | Standard deviation multiplier for outlier detection |
| `--kurtosis-weight` | 0.6 | Score weight for excess kurtosis |
| `--range-weight` | 0.3 | Score weight for dynamic range |
| `--ar-weight` | 0.1 | Score weight for aspect ratio |
