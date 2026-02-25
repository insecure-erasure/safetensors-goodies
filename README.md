# safetensors-extract

A command-line utility to extract individual components from diffusion model checkpoints into separate `.safetensors` files, with automatic architecture detection and optional precision conversion. Designed for workflows that quantize components independently before loading them in [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Features

- **Automatic architecture detection** — no need to specify the model type manually
- Extracts components (UNet/Transformer/DiT, VAE, text encoders) with **ComfyUI-compatible key naming**
- **Default precision policy**: fp32 tensors are downscaled to fp16 automatically; fp8/fp16/bf16 are kept as-is; VAE is always exempt
- Per-component **precision overrides**: `fp32`, `fp16`, `bf16`, `fp8`
- Analyze and list modes to inspect checkpoints without extracting anything
- `--force-architecture` to override detection when needed

## Supported architectures

| Architecture | Diffusion model | VAE | Text encoder(s) |
|---|---|---|---|
| SDXL / Pony | UNet | ✓ | CLIP-L + CLIP-G |
| SD 1.5 / 2.x | UNet | ✓ | CLIP |
| Flux / Chroma | Transformer | ✓ | CLIP-L + T5-XXL |
| Lumina / zImage | DiT | ✓ | Text encoder |
| PixArt | DiT | ✓ | Text encoder |
| HunyuanDiT | DiT | ✓ | Text encoder + Text encoder 2 |
| Unknown | generic fallback | — | — |

## Requirements

```
torch
safetensors
```

Install with:

```bash
pip install torch safetensors
```

## Usage

### Inspect a checkpoint without extracting

```bash
# Detailed structure with key transformation preview
python safetensors-extract.py -i my_model.safetensors --analyze

# Quick component list
python safetensors-extract.py -i my_model.safetensors --list
```

### Extract all components

```bash
python safetensors-extract.py -i my_model.safetensors -d ./output
```

fp32 tensors are automatically downscaled to fp16. VAE is kept at its original precision.

### Keep original precision

```bash
python safetensors-extract.py -i my_model.safetensors -d ./output -k
```

### Extract specific components

```bash
python safetensors-extract.py -i my_model.safetensors -d ./output -c unet -c vae
```

### Override precision for specific components

```bash
# Keep precision globally, but force VAE to 16-bit
python safetensors-extract.py -i my_model.safetensors -d ./output -k --vae-precision 16

# Explicit precision for everything
python safetensors-extract.py -i my_model.safetensors -d ./output \
    --unet-precision 16 \
    --clip-l-precision 16 \
    --clip-g-precision 16 \
    --vae-precision 16
```

Precision flags accept `16` or `32`. For 16-bit targets, the script automatically chooses fp16 or bf16 based on value range analysis. Upscaling is never performed.

### Allow mixed fp16/bf16 per tensor

```bash
python safetensors-extract.py -i my_model.safetensors -d ./output -m
```

By default the fp16/bf16 decision is made once per component. With `-m`, each tensor is evaluated individually. This may produce files with mixed dtypes, which not all model loaders handle correctly.

### Force architecture

```bash
python safetensors-extract.py -i my_model.safetensors -d ./output --force-architecture Flux
```

## Precision policy

| Scenario | Behavior |
|---|---|
| Default (no flags) | fp32 → 16-bit adaptive (fp16 if all values fit within ±65504, else bf16); float8 and VAE always unchanged |
| `-k` / `--keep-precision` | All tensors keep original dtype |
| `-m` / `--mixed-dtype` | Per-tensor fp16/bf16 decision instead of per-component; may produce mixed-dtype files |
| `--{component}-precision 16` | That component uses adaptive 16-bit conversion (float8 tensors still kept as-is) |
| `--{component}-precision 32` | That component keeps fp32 (upscaling is never done) |

After every conversion, the script automatically checks for new infinities (overflow), values flushed to zero, and new NaNs. Issues are reported as warnings but do not abort extraction. These checks are skipped for float8 tensors, as PyTorch does not implement `isinf`/`isnan` for those types.

## Output naming

```
{model_name}_{component}.safetensors
{model_name}_{component}.{precision}.safetensors   # when precision override is used
```

## CLI reference

| Argument | Description |
|---|---|
| `-i`, `--input` | Path to the source `.safetensors` file *(required)* |
| `-d`, `--output-dir` | Output directory for extracted files |
| `-c`, `--component` | Component to extract (repeatable). Default: all detected |
| `-k`, `--keep-precision` | Preserve original dtypes (default: downscale fp32 → 16-bit adaptive) |
| `-m`, `--mixed-dtype` | Per-tensor fp16/bf16 decision (default: per-component) |
| `--force-architecture` | Override auto-detection: `SDXL`, `SD15`, `Flux`, `Lumina`, `PixArt`, `HunyuanDiT` |
| `--analyze` | Inspect checkpoint structure without extracting |
| `--list` | Print detected architecture and component names |
| `--keep-original-keys` | Preserve original key names (disables ComfyUI key transforms) |
| `--vae-precision` | Precision override for VAE: `16` or `32` |
| `--unet-precision` | Precision override for UNet |
| `--transformer-precision` | Precision override for Transformer (Flux) |
| `--dit-precision` | Precision override for DiT (Lumina, PixArt, HunyuanDiT) |
| `--clip-precision` | Precision override for CLIP (SD15, Flux) |
| `--clip-l-precision` | Precision override for CLIP-L (SDXL) |
| `--clip-g-precision` | Precision override for CLIP-G (SDXL) |
| `--t5-precision` | Precision override for T5 (Flux, generic) |
| `--t5xxl-precision` | Precision override for T5-XXL (Flux) |
| `--text-encoder-precision` | Precision override for text encoder |
| `--text-encoder-2-precision` | Precision override for second text encoder |
