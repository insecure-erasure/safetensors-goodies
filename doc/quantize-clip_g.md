# quantize-clip_g.py

Mixed FP16/FP8 (E4M3) quantization for a CLIP-G text encoder extracted from an SDXL checkpoint in OpenCLIP format (`transformer.resblocks.N.*`).

The output safetensors can be loaded directly by ComfyUI-GGUF's `DualCLIPLoaderGGUF` node, which respects per-tensor dtypes.

## Requirements

```
torch >= 2.1  (FP8 support required)
safetensors
```

```
pip install torch safetensors
```

---

## Quantization strategy

The script targets **ViT-bigG-14** (32 transformer blocks). Tensors fall into one of two categories: preserved at FP16, or quantized to `float8_e4m3fn`.

### Always preserved at FP16

- All tensors outside `transformer.resblocks.*`: embeddings, positional encodings, LayerNorm parameters, final projection, `logit_scale`.
- All **biases** — they are numerically sensitive and small enough to make quantization savings negligible.
- The **first N blocks** (controlled by `--keep-first-blocks`, default: 7), i.e. blocks 0–6.
- The **last block** (index 31), always.

### Quantized to FP8 (intermediate blocks only)

The following tensors in blocks `first_keep` through 30 are candidates for FP8 quantization, subject to the active flags:

| Tensor | Default | Flag required |
|---|---|---|
| `mlp.c_fc.weight` | ✅ FP8 | — |
| `mlp.c_proj.weight` | ✅ FP8 | — |
| `attn.in_proj_weight` *(fused)* | ✅ FP8 | — |
| `attn.out_proj.weight` | FP16 | `--attn-out-fp8` |
| `attn.k_proj.weight` *(split)* | FP8 | `--split-attn-qkv [q16,kv8 \| q8,kv8 \| q8,kv16]` |
| `attn.v_proj.weight` *(split)* | FP8 | `--split-attn-qkv [q16,kv8 \| q8,kv8 \| q8,kv16]` |
| `attn.q_proj.weight` *(split)* | FP16 | `--split-attn-qkv [q8,kv8 \| q8,kv16]` |

### Fused attention weights (`in_proj_weight`)

The original OpenCLIP format stores Q, K, and V projections as a single fused tensor (`attn.in_proj_weight`). By default the script quantizes this tensor to FP8 directly, without splitting.

With `--split-attn-qkv`, the fused tensor is split into three separate tensors (`attn.q_proj.weight`, `attn.k_proj.weight`, `attn.v_proj.weight`) and the corresponding bias is split into `attn.q_proj.bias`, `attn.k_proj.bias`, `attn.v_proj.bias`. Splitting replaces 2 tensors (weight + bias) with 6, for a net gain of +4 tensors per eligible block. The precision of each component is controlled by the split mode (see below).

---

## Usage

### Default (MLP + fused in_proj_weight to FP8)

```bash
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors
```

Quantizes `mlp.c_fc.weight`, `mlp.c_proj.weight`, and `attn.in_proj_weight` (fused) in intermediate blocks. All other attention tensors remain at FP16.

### Split attention: Q FP16, K/V FP8 (default split mode)

```bash
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors --split-attn-qkv
```

Splits `in_proj_weight` into Q/K/V. K and V are quantized to FP8; Q stays at FP16. Equivalent to `--split-attn-qkv q16,kv8`.

### Split attention modes

```bash
# Q FP16, K/V FP8 (default when no mode is specified)
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors --split-attn-qkv q16,kv8

# Q, K, V all FP8
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors --split-attn-qkv q8,kv8

# Q FP8, K/V FP16
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors --split-attn-qkv q8,kv16

# Split only, no quantization of Q/K/V
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors --split-attn-qkv q16,kv16
```

### Add out_proj quantization (independent of split)

```bash
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors --attn-out-fp8
```

### Maximum quantization

```bash
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors \
    --split-attn-qkv q8,kv8 --attn-out-fp8
```

### Protect specific blocks using ranges or enumerations

```bash
# Single integer n: protect blocks 0..n-1 (legacy behaviour, default is 7)
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors -k 7

# Contiguous range
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors -k 0-6

# Non-contiguous ranges
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors -k 0-3,5-6

# Explicit enumeration
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors -k 0,1,4,5
```

Block 31 is always protected regardless of the `-k` value.

### Analyze before quantizing

```bash
python quantize-clip_g.py -i clip_g.safetensors --analyze
```

Prints per-block quantization sensitivity metrics and recommendations. Does not write any file. Only `--input` is required.

### Adjust the number of protected initial blocks

```bash
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors \
    --keep-first-blocks 8
```

### Dry run with verbose per-tensor mapping

```bash
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors \
    --split-attn-qkv --dry-run --verbose
```

---

## CLI reference

| Flag | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | *(required)* | Input safetensors file (CLIP-G, OpenCLIP format, FP16) |
| `--output` | `-o` | *(required unless `--analyze`)* | Output safetensors file |
| `--keep-first-blocks` | `-k` | `7` | Blocks to preserve at FP16. Accepts a single integer `n` (protects blocks 0 to n−1), or a comma-separated list of indices and/or ranges (e.g. `0-3,5-6`, `0,1,4,5`). Block 31 is always preserved regardless of this setting. |
| `--analyze` | | off | Analyze quantization sensitivity per block. Prints metrics and recommendations. Does not write any file. |
| `--split-attn-qkv [MODE]` | | off | Split fused `in_proj_weight` into separate Q/K/V tensors. Optional mode controls per-component precision (see table below). When omitted, `in_proj_weight` is quantized to FP8 as a fused tensor. |
| `--attn-out-fp8` | | off | Quantize `out_proj.weight` to FP8 in intermediate blocks. Does not require `--split-attn-qkv`. |
| `--dry-run` | | off | Compute and print statistics without writing the output file. |
| `--verbose` | `-v` | off | Print per-tensor dtype mapping and detailed outlier information. |

### --split-attn-qkv modes

| Mode | Q | K | V | Notes |
|---|---|---|---|---|
| `q16,kv8` | FP16 | FP8 | FP8 | Default when no mode is specified |
| `q8,kv8` | FP8 | FP8 | FP8 | Maximum attention quantization |
| `q8,kv16` | FP8 | FP16 | FP16 | Q only |
| `q16,kv16` | FP16 | FP16 | FP16 | Split only, no quantization |

When `--split-attn-qkv` is used without an explicit mode, the script prints a note confirming the default (`q16,kv8`).

---

## Analysis mode

`--analyze` computes the following metrics for each weight tensor in every transformer block:

| Metric | Description |
|---|---|
| `Params` | Number of parameters in the tensor |
| `Norm` | Frobenius norm |
| `Max\|W\|` | Maximum absolute value |
| `Std` | Standard deviation |
| `Outliers` | Values outside the FP8 E4M3 representable range (±448) |
| `QError%` | Roundtrip quantization error — NRMSE from FP16 → FP8 → FP16 |

Blocks are ranked by their **weighted average quantization error** (across all quantizable tensors), displayed in **block index order** with sensitive blocks flagged inline. Blocks with error above mean + 1 standard deviation are flagged as sensitive.

The recommendation section suggests a `--keep-first-blocks` value covering all sensitive blocks. When sensitive blocks do not form a contiguous prefix, both the covering contiguous range and the exact `-k` enumeration are shown.

### Files produced with --split-attn-qkv

`--analyze` handles both formats transparently:

- **Fused format** (original files or protected blocks): `attn.in_proj_weight` is split internally into Q/K/V for per-component metrics.
- **Pre-split format** (files produced with `--split-attn-qkv`): Q/K/V arrive as separate tensors. The script accumulates them and synthesizes a fused `attn.in_proj_weight` summary row by concatenating the three tensors along dimension 0.

In both cases the per-block table shows the same set of rows.

After the general block ranking, the analysis mode emits two additional sections:

### attn.in_proj split quantization impact analysis

Reports per-block quantization error for `attn.q_proj.weight`, `attn.k_proj.weight`, and `attn.v_proj.weight` (derived from splitting `in_proj_weight`) across all intermediate blocks. Blocks where any of Q, K, or V exceeds 6% QError are flagged as elevated. The section recommends the minimum `--keep-first-blocks` to cover them, or suggests using `--split-attn-qkv q16,kv16` as an alternative.

### attn.out_proj quantization impact analysis

Reports per-block quantization error for `attn.out_proj.weight` across all intermediate blocks, with risk classification:

| Threshold | Risk level | Meaning |
|---|---|---|
| QError < 8% | ok | Safe to quantize |
| QError ≥ 8% | ELEVATED | Review carefully |
| QError ≥ 12% | HIGH | Strongly discouraged |

For blocks with elevated or high risk, the section reports the minimum `--keep-first-blocks` value needed to exclude them from quantization, and cross-references it with the general block sensitivity recommendation.

---

## Output

After conversion, the script prints a summary:

```
=== Summary ===
  FP16 tensors kept       : …
  FP8  tensors quantized  : …
  in_proj blocks split    : …
  Output tensors (total)  : …
  Input size  (estimated) : … MB
  Output size (estimated) : … MB
  Size reduction          : …%
```

A **tensor count verification** checks that the number of output tensors matches the expected count, accounting for the tensors added by `--split-attn-qkv`.

A **NaN/Inf sanity check** is run on all FP16 tensors before saving. FP8 tensors cannot be checked directly with `torch.isfinite`.

---

## Notes

- FP8 quantization requires **PyTorch ≥ 2.1** compiled with FP8 support. The script checks for `torch.float8_e4m3fn` at startup and exits with an error if it is not available.
- Quantization uses a FP32 intermediate: `tensor.to(torch.float32).to(torch.float8_e4m3fn)`.
- Outlier warnings are emitted for any tensor with values exceeding ±448 before quantization, indicating possible clipping.
- Q is considered the most sensitive attention component. Validate output quality carefully when using `--attn-q-fp8`.
