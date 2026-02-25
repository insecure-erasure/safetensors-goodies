# safetensors-goodies

A collection of command-line utilities for working with diffusion model files in `.safetensors` format, aimed at technically proficient users comfortable with CLI tools.


## [safetensors-extract.py](doc/safetensors-extract.md)

Extracts individual components (UNet/Transformer/DiT, VAE, text encoders) from a full diffusion model checkpoint into separate `.safetensors` files, with automatic architecture detection and ComfyUI-compatible key renaming.

The main use case is preparing components for independent quantization. The typical workflow is to extract the components with this script, convert them to GGUF format using the conversion tools bundled with [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), and then quantize them with `llama-quantize` from [llama.cpp](https://github.com/ggerganov/llama.cpp) using [city96's diffusion model patch](https://github.com/city96/ComfyUI-GGUF/blob/main/tools/README.md). This makes it possible to produce any K-quant variant (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, etc.) for any component of any supported architecture — something that checkpoint authors rarely provide beyond a handful of FP8 or INT4 releases.

### SDXL Pipeline Comparison: Checkpoint vs GGUF + VAE Configurations

This section validates that different SDXL pipeline configurations produce equivalent results.

All three images were generated using the same prompt, seed, and sampler settings.

* Checkpoint: [cyberrealisticXL v9.0](https://civitai.com/models/312530?modelVersionId=2611295) by [Cyberdelia](https://civitai.com/user/Cyberdelia)
* Prompt: ultra detailed bust portrait of a 28-year-old woman, medium shot, wearing turtle neck sweater, subject centered, natural framing, 50mm lens, realistic skin texture, detailed eyes, soft cinematic lighting, professional photography, high dynamic range, sharp focus, natural colors
* Negative prompt: lowres, blurry, oversharpened, jpeg artifacts, bad anatomy, extra fingers, deformed hands, cross-eye, plastic skin, overexposed, underexposed, watermark, text, logo
* Sampler: dpmpp_2m
* Scheduler: karras
* Steps: 30
* Denoise: 1.0

| Image | Configuration |
|:-----:|:--------------|
| [![Full checkpoint FP16](assets/sdxl_checkpoint-fp16.png)](assets/sdxl_checkpoint-fp16.png) | **Full checkpoint — FP16**<br>Standard SDXL checkpoint with the VAE baked in at FP16. Used as the reference output. |
| [![GGUF FP16 + VAE from checkpoint](assets/sdxl_gguf-fp16_clip_vae.png)](assets/sdxl_gguf-fp16_clip_vae.png) | **GGUF FP16 UNet + CLIP and VAE extracted from the checkpoint**<br>The UNet is loaded in GGUF FP16 format. The VAE and CLIP are the originals extracted from the same checkpoint, also in FP16. Pixel-perfect identical to the full checkpoint output (mean difference: 0.0). |
| [![GGUF FP16 + VAE fp16-fix FP16](assets/sdxl_gguf-fp16_clip_vae-fp16-fix-fp16.png)](assets/sdxl_gguf-fp16_clip_vae-fp16-fix-fp16.png) | **GGUF FP16 UNet + CLIP from checkpoint + madebyollin's VAE fp16-fix converted to FP16**<br>Same as above but using [madebyollin's sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) as the decoder. The original weights are FP32; they were converted to FP16 prior to this test. The conversion produced no overflow-to-inf cases and only 405 underflow-to-zero values out of millions of parameters — all negligible. Output is pixel-perfect identical to the reference (mean difference: 0.0). |

## [quantize-clip_g.py](doc/quantize-clip_g.md)

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
