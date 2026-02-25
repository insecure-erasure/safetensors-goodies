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
| Flux | Transformer | ✓ | CLIP + T5 |
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
# Keep precision globally, but force VAE to fp16
python safetensors-extract.py -i my_model.safetensors -d ./output -k --vae-precision fp16

# Explicit precision for everything
python safetensors-extract.py -i my_model.safetensors -d ./output \
    --unet-precision fp16 \
    --clip-l-precision fp8 \
    --clip-g-precision fp8 \
    --vae-precision fp16
```

Precision can only be converted **downward**. Attempting to upscale will emit a warning and keep the original dtype.

### Force architecture

```bash
python safetensors-extract.py -i my_model.safetensors -d ./output --force-architecture Flux
```

## Precision policy

| Scenario | Behavior |
|---|---|
| Default (no flags) | fp32 → fp16; fp16/bf16/fp8 → unchanged; VAE always unchanged |
| `-k` / `--keep-precision` | All tensors keep original dtype |
| `--{component}-precision X` | That component uses precision X, regardless of `-k` |
| Upscale attempt (e.g. fp16 → fp32) | Warning emitted, original dtype kept |

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
| `-k`, `--keep-precision` | Preserve original dtypes (default: downscale fp32 → fp16) |
| `--force-architecture` | Override auto-detection: `SDXL`, `SD15`, `Flux`, `Lumina`, `PixArt`, `HunyuanDiT` |
| `--analyze` | Inspect checkpoint structure without extracting |
| `--list` | Print detected architecture and component names |
| `--keep-original-keys` | Preserve original key names (disables ComfyUI key transforms) |
| `--vae-precision` | Precision override for VAE: `fp32`, `fp16`, `bf16`, `fp8` |
| `--unet-precision` | Precision override for UNet |
| `--transformer-precision` | Precision override for Transformer (Flux) |
| `--dit-precision` | Precision override for DiT (Lumina, PixArt, HunyuanDiT) |
| `--clip-precision` | Precision override for CLIP (SD15, Flux) |
| `--clip-l-precision` | Precision override for CLIP-L (SDXL) |
| `--clip-g-precision` | Precision override for CLIP-G (SDXL) |
| `--t5-precision` | Precision override for T5 (Flux) |
| `--text-encoder-precision` | Precision override for text encoder |
| `--text-encoder-2-precision` | Precision override for second text encoder |
