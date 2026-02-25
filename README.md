# safetensors-goodies

A collection of command-line utilities for working with diffusion model files in `.safetensors` format.

## Scripts

### [safetensors-extract.py](doc/safetensors-extract.md)

Extracts individual components (UNet/Transformer/DiT, VAE, text encoders) from a full diffusion model checkpoint into separate `.safetensors` files. Supports automatic architecture detection, ComfyUI-compatible key renaming, and optional precision conversion. Designed for workflows that quantize components independently before loading them in ComfyUI.

```
usage: safetensors-extract.py [-h] -i INPUT [-d OUTPUT_DIR] [-c COMPONENTS] [--analyze] [--list]
                              [-k] [-m]
                              [--force-architecture {SDXL,SD15,Flux,Lumina,PixArt,HunyuanDiT}]
                              [--keep-original-keys] [--vae-precision {16,32}]
                              [--unet-precision {16,32}] [--transformer-precision {16,32}]
                              [--dit-precision {16,32}] [--clip-precision {16,32}]
                              [--clip-l-precision {16,32}] [--clip-g-precision {16,32}]
                              [--t5-precision {16,32}] [--text-encoder-precision {16,32}]
                              [--text-encoder-2-precision {16,32}]

Universal safetensors component extractor

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Input .safetensors file
  -d, --output-dir OUTPUT_DIR
                        Output directory for extracted components
  -c, --component COMPONENTS
                        Component to extract (can be repeated). Default: all detected
  --analyze             Analyze without extracting
  --list                List available components
  -k, --keep-precision  Keep original precision (default: downscale fp32 to 16-bit)
  -m, --mixed-dtype     Allow mixed fp16/bf16 per tensor (may cause compatibility issues)
  --force-architecture {SDXL,SD15,Flux,Lumina,PixArt,HunyuanDiT}
                        Override architecture detection
  --keep-original-keys  Keep original keys (for debugging)
  --vae-precision {16,32}
                        Target precision for vae (16 or 32 bits)
  --unet-precision {16,32}
                        Target precision for unet (16 or 32 bits)
  --transformer-precision {16,32}
                        Target precision for transformer (16 or 32 bits)
  --dit-precision {16,32}
                        Target precision for dit (16 or 32 bits)
  --clip-precision {16,32}
                        Target precision for clip (16 or 32 bits)
  --clip-l-precision {16,32}
                        Target precision for clip_l (16 or 32 bits)
  --clip-g-precision {16,32}
                        Target precision for clip_g (16 or 32 bits)
  --t5-precision {16,32}
                        Target precision for T5-XXL encoder (16 or 32 bits) (Flux, Chroma)
  --text-encoder-precision {16,32}
                        Target precision for text_encoder (16 or 32 bits)
  --text-encoder-2-precision {16,32}
                        Target precision for text_encoder_2 (16 or 32 bits)

Examples:
  # Analyze checkpoint structure
  safetensors-extract.py -i model.safetensors --analyze

  # Extract all components (fp32 auto-downscaled to 16-bit)
  safetensors-extract.py -i model.safetensors -d ./extracted

  # Extract keeping original precision
  safetensors-extract.py -i model.safetensors -d ./extracted -k

  # Extract only VAE and UNet
  safetensors-extract.py -i model.safetensors -d ./extracted -c vae -c unet

  # Force VAE to 16-bit precision
  safetensors-extract.py -i model.safetensors -d ./extracted --vae-precision 16

  # Allow mixed fp16/bf16 per tensor
  safetensors-extract.py -i model.safetensors -d ./extracted -m

  # List available components
  safetensors-extract.py -i model.safetensors --list

Precision policy:
  - Default: fp32 tensors are downscaled to 16-bit (fp16 if values fit, bf16 otherwise)
  - Exception: VAE always keeps original precision (for quality)
  - With -k/--keep-precision: all tensors keep original precision
  - Explicit --*-precision flags override the policy for that component
  - With -m/--mixed-dtype: per-tensor fp16/bf16 decision (may cause compatibility issues)

Supported architectures: SDXL, SD15, Flux, Lumina, PixArt, HunyuanDiT
Unknown architectures are handled with generic pattern matching.
```

### [quantize-clip_g.py](doc/quantize-clip_g.md)

Applies mixed FP16/FP8 quantization to a CLIP-G encoder (OpenCLIP format). Allows fine-grained control over which transformer blocks are kept at FP16, with optional splitting of fused attention QKV tensors and per-component precision assignment. Includes an analysis mode that reports per-block quantization sensitivity and recommends which blocks to protect.

```
usage: quantize-clip_g.py [-h] --input INPUT [--output OUTPUT] [--keep-first-blocks SPEC]
                          [--analyze] [--split-attn-qkv [MODE]] [--attn-out-fp8] [--verbose]
                          [--dry-run]

Mixed FP16/FP8 quantization for CLIP-G (OpenCLIP format) safetensors

options:
  -h, --help            show this help message and exit
  --input, -i INPUT     Input safetensors file (CLIP-G in FP16, OpenCLIP format)
  --output, -o OUTPUT   Output safetensors file with mixed FP16/FP8 precision (not required when
                        using --analyze)
  --keep-first-blocks, -k SPEC
                        Blocks to keep at FP16. Accepts a single integer n (protect blocks 0..n-1),
                        or a comma-separated list of indices and/or ranges, e.g.: 7 | 0-6 | 0-3,5-6
                        | 0,1,4,5 | 0,1,2-4,6. Block 31 is always kept at FP16 regardless of this
                        setting. (default: 7, i.e. blocks 0-6)
  --analyze             Analyze all weight tensors and report per-block quantization sensitivity.
                        Prints metrics (norm, outliers, roundtrip error) and recommends which blocks
                        to protect. Does not write any file. Only requires --input.
  --split-attn-qkv [MODE]
                        Split fused in_proj_weight into separate Q/K/V tensors (and biases), with
                        precision control per component. Without this flag, in_proj_weight is
                        quantized to FP8 as a fused tensor. Options: q16,kv8 - Q FP16, K/V FP8
                        (default if no value given); q8,kv8 - Q, K, V all FP8; q8,kv16 - Q FP8, K/V
                        FP16; q16,kv16 - Q, K, V all FP16 (split only, no quantization)
  --attn-out-fp8        Quantize attention out_proj.weight to FP8 in intermediate blocks.
                        Conservative extra savings with low quality risk. Does not require --split-
                        attn-qkv (out_proj is already a separate tensor).
  --verbose, -v         Print per-tensor dtype mapping and detailed outlier checks
  --dry-run             Compute and print statistics without writing any file
```

## Requirements

```
torch
safetensors
```

```
pip install torch safetensors
```
