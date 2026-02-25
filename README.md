# safetensors-goodies

A collection of command-line utilities for working with diffusion model files in `.safetensors` format, aimed at technically proficient users comfortable with CLI tools.

## Scripts

### [safetensors-extract.py](doc/safetensors-extract.md)

Extracts individual components (UNet/Transformer/DiT, VAE, text encoders) from a full diffusion model checkpoint into separate `.safetensors` files, with automatic architecture detection and ComfyUI-compatible key renaming.

The main use case is preparing components for independent quantization. The typical workflow is to extract the components with this script, convert them to GGUF format using the conversion tools bundled with [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), and then quantize them with `llama-quantize` from [llama.cpp](https://github.com/ggerganov/llama.cpp) using [city96's diffusion model patch](https://github.com/city96/ComfyUI-GGUF/blob/main/tools/README.md). This makes it possible to produce any K-quant variant (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, etc.) for any component of any supported architecture — something that checkpoint authors rarely provide beyond a handful of FP8 or INT4 releases.

### [quantize-clip_g.py](doc/quantize-clip_g.md)

Applies mixed FP16/FP8 quantization to a CLIP-G text encoder (OpenCLIP format). CLIP-G is one of the text encoders used in SDXL-based models and, at around 1.3 GB, it is a non-trivial memory cost for users with limited VRAM. This script is an experiment in selective quantization: rather than applying a uniform precision reduction, it keeps the most sensitive transformer blocks at FP16 and quantizes only the intermediate blocks where the quality impact is lower, with fine-grained control over which weight tensors are quantized. The goal is to reduce memory footprint with minimal quality loss.

The script includes an analysis mode (`--analyze`) that measures per-block quantization sensitivity and recommends which blocks to protect, making it easier to find a good precision/quality trade-off for a specific model.

Quality comparisons against the full-precision baseline are planned.

## Requirements

```
torch
safetensors
```

```
pip install torch safetensors
```
