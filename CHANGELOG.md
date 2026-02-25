# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project uses a simple `major.minor` versioning scheme.

---

## [2.1] - Adaptive 16-bit conversion and post-conversion validation

### Added

- **Adaptive 16-bit conversion**: when downscaling from fp32, the script now inspects the actual tensor values before choosing a target dtype. If all values fit within the fp16 range (±65504), fp16 is used; otherwise the component falls back to bf16. The decision is made per-component by default, or per-tensor with `--mixed-dtype`.
- `tensor_fits_fp16()`: checks whether a tensor's values fall within fp16 range.
- `analyze_component_for_fp16()`: scans all tensors in a component and returns a list of those that exceed fp16 range, with their max absolute value.
- **Post-conversion validation** via `validate_conversion()`: runs automatically after every conversion and checks for three issues — new infinity values (overflow), values flushed to zero (underflow), and new NaNs. Reports via `UserWarning` if thresholds are exceeded.
- `-m` / `--mixed-dtype` flag: enables per-tensor fp16/bf16 decision instead of the default per-component decision. Emits a compatibility warning when the output file ends up with mixed dtypes.

### Changed

- `--*-precision` flags now accept integers (`16` or `32`) instead of strings (`fp16`, `bf16`, `fp8`). The script determines the specific 16-bit dtype (fp16 or bf16) automatically based on value range analysis.
- Default precision policy message updated to reflect adaptive behavior: `fp32 → 16-bit adaptive (fp16 if fits, else bf16)`.
- `apply_precision_policy()` rewritten around the new `convert_tensor_to_16bit()` helper, which encapsulates the fp16/bf16 selection logic.
- `PRECISION_MAP` removed; dtype selection is now handled directly via `convert_tensor_to_16bit()` and the `DTYPE_BITS` lookup.

### Removed

- `fp8`, `bf16`, and `fp32`/`fp16` as explicit string choices for `--*-precision` flags — replaced by integer `16`/`32`.
- `fp8` as a conversion target for explicit precision overrides.
- `convert_tensor_precision()` — replaced by `convert_tensor_to_16bit()` and inline upscale guard in `apply_precision_policy()`.
- `DEFAULT_DOWNSCALE_PRECISION` and `LOW_PRECISION_BITS` constants.
- `fp8`/`fp5m2` entries from `DTYPE_BITS` (only standard float and int dtypes remain).


## [2.0] - Universal extractor with architecture auto-detection

Complete rewrite. The script is no longer SDXL/Pony-specific; it now detects the model architecture automatically and adapts its classification and key transformation logic accordingly.

### Added

- **Architecture auto-detection** based on tensor key scoring. Supported architectures: SDXL, SD 1.5, Flux, Lumina, PixArt, HunyuanDiT. Unknown architectures fall back to generic pattern matching.
- `--force-architecture` flag to override auto-detection when needed.
- `--list` mode: prints detected architecture and available components without loading weights.
- `-k` / `--keep-precision` flag to preserve all original dtypes. When omitted, the new default precision policy applies (see below).
- **Default precision policy**: fp32 tensors are automatically downscaled to fp16 on extraction. fp8, fp16, and bf16 tensors are kept as-is. VAE is always exempt from automatic downscaling.
- `bf16` as a valid precision target (replaces the removed `fp4`).
- Per-component precision flags for all component types across all supported architectures: `--vae-precision`, `--unet-precision`, `--transformer-precision`, `--dit-precision`, `--clip-precision`, `--clip-l-precision`, `--clip-g-precision`, `--t5-precision`, `--text-encoder-precision`, `--text-encoder-2-precision`.
- `--component` / `-c` flag is now free-form (no fixed choices), matching whatever components the detected architecture exposes.
- `list_components()` as a standalone utility function.
- Usage examples and precision policy documentation in `--help` output.

### Changed

- Key classification logic replaced by architecture-specific regex pattern matching (`ARCHITECTURE_PATTERNS` dict) with a generic fallback (`GENERIC_COMPONENT_PATTERNS`).
- Key transformation is now driven per-architecture and per-component from `ARCHITECTURE_PATTERNS`, replacing the static `KEY_TRANSFORMS` dict.
- Wrapper prefix detection (`model.diffusion_model.`, `model.`, `diffusion_model.`) is now automatic and applied before classification and transformation.
- `analyze_checkpoint()` rewritten to use `safe_open` (no weights loaded) and to show key transformation previews alongside component groupings.
- `extract_components()` now uses `safe_open` for initial key scanning and only loads full weights for the extraction step.
- Precision conversion consolidated into `apply_precision_policy()`, which handles the default policy, `--keep-precision`, and explicit overrides in a single pass with per-component conversion stats.
- Upscaling warning is now only shown when the upscale was explicitly requested, not silently skipped.
- Output summary shows GB instead of MB for file size reporting at the top level.
- Code reorganized into clearly delimited sections: precision handling, architecture detection, extraction logic, CLI.

### Removed

- `fp4` precision option (PyTorch has no native fp4 dtype; it was silently aliasing to fp8 anyway).
- Hard-coded component list `['unet', 'vae', 'clip_l', 'clip_g']` — components are now derived from architecture detection.
- `identify_component()` and the static `KEY_TRANSFORMS` dict.
- `get_tensor_bits()` (replaced by `DTYPE_BITS` lookup dict used directly).

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
