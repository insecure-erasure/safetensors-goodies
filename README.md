# safetensors-extract

A command-line utility to extract individual components from SDXL/Pony diffusion model checkpoints into separate `.safetensors` files, with optional precision conversion. Designed for workflows that quantize components independently before loading them in [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Features

- Extracts **UNet**, **CLIP-L**, **CLIP-G**, and **VAE** from a single checkpoint
- Outputs keys in **ComfyUI-compatible standalone format** (prefixes stripped automatically)
- Supports per-component **precision conversion**: `fp32`, `fp16`, `fp8`, `fp4`
- Analyze-only mode to inspect checkpoint structure without extracting anything
- Selective extraction: extract only the components you need

## Requirements

```
torch
safetensors
```

Install with:

```bash
pip install torch safetensors
```

> **Note:** fp4 is not natively supported by PyTorch. Tensors requested at fp4 precision will silently fall back to fp8 (`torch.float8_e4m3fn`).

## Usage

### Extract all components

```bash
python safetensors-extract.py -i my_model.safetensors -d ./output
```

### Extract specific components

```bash
python safetensors-extract.py -i my_model.safetensors -d ./output -c unet -c vae
```

### Extract with precision conversion

```bash
python safetensors-extract.py -i my_model.safetensors -d ./output \
    --vae-precision fp16 \
    --clip-l-precision fp8 \
    --clip-g-precision fp8
```

> Precision can only be converted **downward** (e.g. fp32 → fp16). Attempting to upscale precision will emit a warning and keep the original dtype.

### Analyze checkpoint structure

```bash
python safetensors-extract.py -i my_model.safetensors --analyze
```

### Keep original keys (debugging)

```bash
python safetensors-extract.py -i my_model.safetensors -d ./output --keep-original-keys
```

## Output naming

Extracted files follow this naming convention:

```
{model_name}_{component}.safetensors
{model_name}_{component}.{precision}.safetensors   # when precision is specified
```

Example:

```
ponyDiffusionV6XL_unet.safetensors
ponyDiffusionV6XL_vae.fp16.safetensors
ponyDiffusionV6XL_clip_l.fp8.safetensors
```

## Supported model architectures

| Architecture | UNet | CLIP-L | CLIP-G | VAE |
|---|---|---|---|---|
| SDXL | ✓ | ✓ | ✓ | ✓ |
| Pony Diffusion | ✓ | ✓ | ✓ | ✓ |
| SD 1.x / 2.x (partial) | ✓ | ✓ | — | ✓ |

## Key transformation

When a checkpoint is loaded, keys are automatically stripped of their component-level prefixes so the resulting file can be loaded as a standalone model in ComfyUI:

| Original key prefix | Component | Output key prefix |
|---|---|---|
| `model.diffusion_model.*` | UNet | *(removed)* |
| `first_stage_model.*` | VAE | *(removed)* |
| `conditioner.embedders.0.transformer.*` | CLIP-L | *(removed)* |
| `conditioner.embedders.1.model.*` | CLIP-G | *(removed)* |

Use `--keep-original-keys` to skip this transformation.

## CLI reference

| Argument | Description |
|---|---|
| `-i`, `--input` | Path to the source `.safetensors` file *(required)* |
| `-d`, `--output-dir` | Output directory for extracted files |
| `-c`, `--component` | Component to extract: `unet`, `vae`, `clip_l`, `clip_g` (repeatable) |
| `--vae-precision` | Target precision for VAE: `fp4`, `fp8`, `fp16`, `fp32` |
| `--clip-l-precision` | Target precision for CLIP-L |
| `--clip-g-precision` | Target precision for CLIP-G |
| `--analyze` | Inspect checkpoint structure without extracting |
| `--keep-original-keys` | Preserve original key names (disables ComfyUI key transforms) |
