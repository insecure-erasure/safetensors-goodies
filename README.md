# safetensors-goodies

A collection of command-line utilities for working with diffusion model files in `.safetensors` format.

## Scripts

### [safetensors-extract.py](doc/safetensors-extract.md)

Extracts individual components (UNet/Transformer/DiT, VAE, text encoders) from a full diffusion model checkpoint into separate `.safetensors` files. Supports automatic architecture detection, ComfyUI-compatible key renaming, and optional precision conversion. Designed for workflows that quantize components independently before loading them in ComfyUI.

### [quantize-clip_g.py](doc/quantize-clip_g.md)

Applies mixed FP16/FP8 quantization to a CLIP-G encoder (OpenCLIP format). Allows fine-grained control over which transformer blocks are kept at FP16, with optional splitting of fused attention QKV tensors and per-component precision assignment. Includes an analysis mode that reports per-block quantization sensitivity and recommends which blocks to protect.

## Requirements

```
torch
safetensors
```

```
pip install torch safetensors
```
