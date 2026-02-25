# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project uses a simple `major.minor` versioning scheme.

---

## [1.0] - Initial release

### Added

- Extraction of **UNet**, **CLIP-L**, **CLIP-G**, and **VAE** components from SDXL/Pony `.safetensors` checkpoints into separate files.
- Automatic **key transformation** to ComfyUI-compatible standalone model format (strips component-level prefixes such as `model.diffusion_model.`, `first_stage_model.`, `conditioner.embedders.0.transformer.`, etc.).
- Per-component **precision conversion** via CLI flags: `--vae-precision`, `--clip-l-precision`, `--clip-g-precision`. Supported formats: `fp32`, `fp16`, `fp8`, `fp4` (fp4 falls back to fp8 with a warning, as PyTorch has no native fp4 dtype).
- Guard against **precision upscaling**: tensors cannot be converted to a higher bit-width than their source; a `UserWarning` is emitted and the original dtype is preserved.
- **Selective extraction** via `-c`/`--component` (repeatable flag). When omitted, all four components are extracted.
- **Analyze mode** (`--analyze`): inspects and prints checkpoint key structure grouped by prefix without writing any files.
- `--keep-original-keys` flag to skip key transformation (useful for debugging or non-ComfyUI workflows).
- Output files are named `{model_stem}_{component}.safetensors` or `{model_stem}_{component}.{precision}.safetensors` when a precision target is specified.
- Per-component extraction summary: tensor count, parameter count, in-memory size, and three sample output keys.
- Warning display for unrecognized tensors (classified as `unknown`).
- Partial support for **SD 1.x/2.x** checkpoint layouts via `cond_stage_model.*` key detection.
