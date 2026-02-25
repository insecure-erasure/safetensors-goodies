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
- The **first N blocks** (controlled by `--first-blocks-keep`, default: 7), i.e. blocks 0–6.
- The **last block** (index 31), always.

### Quantized to FP8 (intermediate blocks only)

The following tensors in blocks `first_keep` through 30 are candidates for FP8 quantization, subject to the active flags:

| Tensor | Default | Flag required |
|---|---|---|
| `mlp.c_fc.weight` | ✅ FP8 | — |
| `mlp.c_proj.weight` | ✅ FP8 | — |
| `attn.out_proj.weight` | FP16 | `--attn-out-fp8` |
| `attn.k_proj.weight` *(split)* | FP8 | `--split-attn-qkv` |
| `attn.v_proj.weight` *(split)* | FP8 | `--split-attn-qkv` |
| `attn.q_proj.weight` *(split)* | FP16 | `--split-attn-qkv --attn-q-fp8` |

### Fused attention weights (`in_proj_weight`)

The original OpenCLIP format stores Q, K, and V projections as a single fused tensor (`attn.in_proj_weight`). By default the script leaves this tensor intact at FP16.

With `--split-attn-qkv`, the fused tensor is split into three separate tensors (`attn.q_proj.weight`, `attn.k_proj.weight`, `attn.v_proj.weight`) and the corresponding bias is split into `attn.q_proj.bias`, `attn.k_proj.bias`, `attn.v_proj.bias`. Splitting replaces 2 tensors (weight + bias) with 6, for a net gain of +4 tensors per eligible block.

---

## Usage

### Conservative (MLP only)

```bash
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors
```

Quantizes only `mlp.c_fc.weight` and `mlp.c_proj.weight` in intermediate blocks. All attention tensors remain at FP16, fused.

### Split attention and quantize K/V

```bash
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors --split-attn-qkv
```

Splits `in_proj_weight` into Q/K/V tensors. K and V are quantized to FP8; Q stays at FP16.

### Isolate the impact of the split vs K/V quantization

```bash
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors \
    --split-attn-qkv --keep-attn-kv-fp16
```

Splits the fused tensor but keeps all resulting Q/K/V tensors at FP16. Useful for diagnosing whether quality changes come from the structural split or from K/V quantization.

### Add out_proj quantization (independent of split)

```bash
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors --attn-out-fp8
```

`out_proj.weight` is already a separate tensor in the original format, so `--attn-out-fp8` works independently of `--split-attn-qkv`.

### Maximum quantization

```bash
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors \
    --split-attn-qkv --attn-q-fp8 --attn-out-fp8
```

Quantizes all weight tensors that can be quantized in intermediate blocks: MLP c_fc, MLP c_proj, Q, K, V, and out_proj.

### Analyze before quantizing

```bash
python quantize-clip_g.py -i clip_g.safetensors --analyze
```

Prints per-block quantization sensitivity metrics (norm, max absolute value, std, outlier count, roundtrip error) and recommends which blocks to protect. Does not write any file. Only `--input` is required.

### Adjust the number of protected initial blocks

```bash
python quantize-clip_g.py -i clip_g.safetensors -o clip_g_fp8.safetensors \
    --first-blocks-keep 8
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
| `--first-blocks-keep` | `-k` | `7` | Number of initial blocks to preserve at FP16 (blocks 0 to N−1). Block 31 is always preserved. |
| `--analyze` | | off | Analyze quantization sensitivity per block. Prints metrics and recommendations. Does not write any file. |
| `--split-attn-qkv` | | off | Split fused `in_proj_weight` into separate Q/K/V tensors. Enables K/V FP8 quantization. |
| `--keep-attn-kv-fp16` | | off | When splitting, keep K and V at FP16 instead of quantizing them. Requires `--split-attn-qkv`. |
| `--attn-q-fp8` | | off | Quantize the Q projection to FP8 in intermediate blocks. Requires `--split-attn-qkv`. |
| `--attn-out-fp8` | | off | Quantize `out_proj.weight` to FP8 in intermediate blocks. Does not require `--split-attn-qkv`. |
| `--dry-run` | | off | Compute and print statistics without writing the output file. |
| `--verbose` | `-v` | off | Print per-tensor dtype mapping and detailed outlier information. |

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

Blocks are ranked by their **weighted average quantization error** (across all quantizable tensors). Blocks with error above mean + 1 standard deviation are flagged as sensitive, and the script emits a recommendation for `--first-blocks-keep`.

The fused `in_proj_weight` is analyzed both as a whole and split into Q, K, V components. The Q/K/V rows are rendered indented under `attn.in_proj_weight` using a tree-style layout (`├─` / `└─`).

After the general block ranking, the analysis mode emits two additional sections:

### --attn-out-fp8 impact analysis

Reports per-block quantization error for `attn.out_proj.weight` across all intermediate blocks, with risk classification:

| Threshold | Risk level | Meaning |
|---|---|---|
| QError < 8% | ok | Safe to quantize |
| QError ≥ 8% | ELEVATED | Review carefully |
| QError ≥ 12% | HIGH | Strongly discouraged |

For blocks with elevated or high risk, the section reports the minimum `--first-blocks-keep` value needed to exclude them from quantization, and cross-references it with the general block sensitivity recommendation.

### --split-attn-qkv K/V quantization impact analysis

Reports per-block quantization error for `attn.k_proj.weight` and `attn.v_proj.weight` (derived from splitting `in_proj_weight`) across all intermediate blocks. Blocks where either K or V exceeds 6% QError are flagged as elevated. The section recommends the minimum `--first-blocks-keep` to cover them, or suggests using `--keep-attn-kv-fp16` as an alternative.

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
