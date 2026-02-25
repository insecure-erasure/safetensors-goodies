# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.5] - Analyze mode handles pre-split safetensors files

### Added (`quantize-clip_g.py`)

- `--analyze` now works correctly on files produced with `--split-attn-qkv`, where `attn.q_proj.weight`, `attn.k_proj.weight`, and `attn.v_proj.weight` exist as separate tensors instead of a fused `attn.in_proj_weight`. The three tensors are accumulated via a `_qkv_staging` dict and concatenated into a synthetic fused tensor to produce the `attn.in_proj_weight` summary row, keeping the per-block table layout consistent across both formats.
- `SPLIT_QKV_SUFFIXES` constant: set of the three split projection suffixes, used to detect pre-split blocks during analysis ingestion.
- `attn.q_proj.weight`, `attn.k_proj.weight`, and `attn.v_proj.weight` added to `ANALYSIS_WEIGHT_SUFFIXES` so they are picked up when iterating the state dict.
- Internal `_qkv_staging` and `_in_proj_source` keys are cleaned up from `block_data` after ingestion.

---

## [1.1.4] - Flexible block protection and analysis output improvements

### Added (`quantize-clip_g.py`)

- **`--first-blocks-keep` / `-k` now accepts ranges and enumerations** in addition to a single integer. Valid formats: `7` (protect blocks 0–6, legacy), `0-6` (explicit range), `0-3,5-6` (multiple ranges), `0,1,4,5` (enumeration). The last block (31) is always protected regardless of the spec.
- `parse_keep_spec()`: parses a `-k` spec string into a set of block indices. Raises `ValueError` with a descriptive message on invalid input.
- `is_block_protected()`: replaces the previous `block_idx < first_keep or block_idx == last_block` inline checks throughout `convert()`.
- When sensitive blocks in `--analyze` do not form a contiguous prefix, the recommendation now shows both the covering contiguous range and the exact `-k` enumeration for surgical protection.
- `_analyze_attn_kv_fp8()` now includes the **Q projection** column in its per-block table and flags blocks where any of Q, K, or V exceeds 6% QError (previously only K and V were checked).

### Changed (`quantize-clip_g.py`)

- `convert()` signature: `first_keep: int` replaced by `protected_blocks: set`. All internal eligibility checks now use `is_block_protected()`.
- `is_fp8_candidate()` signature: `first_keep: int` replaced by `protected_blocks: set`.
- Block sensitivity ranking in `--analyze` is now displayed in **ascending block index order** instead of descending error order. Sensitive blocks are still flagged inline with `<< SENSITIVE`.
- `--analyze` sections reordered: the split Q/K/V impact analysis is shown before the `out_proj` impact analysis.
- `_analyze_attn_out_fp8()` and `_analyze_attn_kv_fp8()` section headers updated to `attn.out_proj quantization impact analysis` and `attn.in_proj split quantization impact analysis (--split-attn-qkv)`.
- Startup summary now prints a human-readable description of protected blocks (e.g. `blocks 0-6 and block 31`, or `blocks 0,1,4,5 and block 31`) instead of the previous fixed `blocks 0-{first_keep-1} and block 31`.
- FP8 candidate summary line changed from `blocks {first_keep}-{TOTAL_BLOCKS-2}` to `unprotected blocks`.

---

## [1.1.3] - Consolidate attention flags into --split-attn-qkv mode argument

### Changed (`quantize-clip_g.py`)

- **`in_proj_weight` is now quantized to FP8 by default** (without any flag). Previously it was kept at FP16 unless `--in-proj-fp8` was passed.
- **`--split-attn-qkv` now accepts an optional mode argument** controlling per-component precision. Valid modes: `q16,kv8` (default), `q8,kv8`, `q8,kv16`, `q16,kv16`. When used without an explicit mode, the script prints a note confirming the default.
- `build_fp8_suffixes()` signature simplified to `(split_mode, attn_out_fp8)`. The `split_mode` string (or `None` for no split) replaces the previous `split_attn`, `keep_kv_fp16`, `attn_q_fp8`, and `in_proj_fp8` boolean parameters.
- `convert()` signature simplified to `(... split_mode, attn_out_fp8, verbose)`. The `split_attn`, `keep_kv_fp16`, `attn_q_fp8`, and `in_proj_fp8` parameters are derived internally from `split_mode`.
- Startup summary now reports `in_proj_weight : fused FP8 (default)` in the no-split path, and `Split attn : enabled (in_proj_weight -> Q/K/V, mode: {mode})` in the split path.
- `_analyze_attn_kv_fp8()` recommendation updated to suggest `--split-attn-qkv q16,kv16` instead of the removed `--keep-attn-kv-fp16`.

### Removed (`quantize-clip_g.py`)

- `--keep-attn-kv-fp16` flag — replaced by `--split-attn-qkv q16,kv16` (or `q8,kv16` to keep only K/V at FP16 while quantizing Q).
- `--attn-q-fp8` flag — replaced by `--split-attn-qkv q8,kv8` or `--split-attn-qkv q8,kv16`.
- `--in-proj-fp8` flag — fused `in_proj_weight` quantization is now the default behaviour in the no-split path.
- Validation errors for `--keep-attn-kv-fp16` and `--attn-q-fp8` without `--split-attn-qkv`, and for `--in-proj-fp8` combined with `--split-attn-qkv`.

---

## [1.1.2] - Add --in-proj-fp8 flag to quantize-clip_g.py

### Added (`quantize-clip_g.py`)

- **`--in-proj-fp8`**: quantizes `attn.in_proj_weight` to FP8 as a fused tensor, without splitting into Q/K/V. Applies only to intermediate blocks (respects `--first-blocks-keep` and the always-protected last block). Incompatible with `--split-attn-qkv`; an explicit error is raised if both flags are passed together.
- `FP8_IN_PROJ_SUFFIX` constant for the fused `attn.in_proj_weight` suffix.
- `build_fp8_suffixes()` now accepts `in_proj_fp8` parameter; adds `FP8_IN_PROJ_SUFFIX` to the candidate set when `--in-proj-fp8` is active and `--split-attn-qkv` is not.
- Startup summary now reports `in_proj_weight : fused FP8 (--in-proj-fp8)` when the flag is active, instead of the default FP16 notice.

---

## [1.1.1] - Richer analyze mode output for quantize-clip_g.py

### Changed (`quantize-clip_g.py`)

- Component names in the per-block analysis table now use dot-notation consistent with the tensor key names (`MLP.c_fc`, `MLP.c_proj`, `attn.out_proj`, `attn.in_proj_weight`, `attn.q_proj.weight`, `attn.k_proj.weight`, `attn.v_proj.weight`) instead of the previous informal labels (`MLP c_fc`, `attn Q`, `in_proj (fused)`, etc.).
- Q/K/V rows in the analysis table are now rendered indented under `attn.in_proj_weight` using a tree-style layout (`├─` / `└─`).
- Analysis table column width increased to accommodate the longer component names.

### Added (`quantize-clip_g.py`)

- **`--attn-out-fp8` impact analysis** section in `--analyze` mode: reports per-block quantization error for `attn.out_proj.weight` across all intermediate blocks, classifies risk (ok / ELEVATED ≥8% / HIGH ≥12%), and recommends the minimum `--first-blocks-keep` value needed to safely use `--attn-out-fp8`. Cross-references with the general block sensitivity recommendation.
- **`--split-attn-qkv` K/V impact analysis** section in `--analyze` mode: reports per-block quantization error for K and V projections (derived from `in_proj_weight`) across all intermediate blocks, flags blocks with QError ≥ 6% as elevated, and recommends `--first-blocks-keep` or `--keep-attn-kv-fp16` as mitigations.
- `COMPONENT_DISPLAY_NAMES` and `IN_PROJ_CHILD_NAMES` dicts: centralize component display name mapping used in the analysis output.
- `WARN_THRESHOLD_OUT_PROJ` (0.08) and `WARN_THRESHOLD_HIGH` (0.12) constants for `--attn-out-fp8` risk classification.
- `block_raw` dict in `analyze()`: parallel per-block storage keyed by raw suffix, used by the flag-specific analysis helpers.
- `_analyze_attn_out_fp8()` and `_analyze_attn_kv_fp8()` as standalone helper functions called at the end of `analyze()`.

---

## [1.1.0] - Initial release of quantize-clip_g.py

### Added

- **`quantize-clip_g.py`**: mixed FP16/FP8 (`float8_e4m3fn`) quantization for a CLIP-G text encoder extracted from an SDXL checkpoint in OpenCLIP format (`transformer.resblocks.N.*`). Output is compatible with ComfyUI-GGUF's `DualCLIPLoaderGGUF` node.

- **Conservative quantization mode** (default): quantizes only `mlp.c_fc.weight` and `mlp.c_proj.weight` in intermediate blocks. All attention tensors, biases, LayerNorm parameters, embeddings, and projection tensors remain at FP16. First N blocks (`--first-blocks-keep`, default 7) and the last block (index 31) are always preserved at FP16.

- **`--split-attn-qkv`**: splits the fused `attn.in_proj_weight` [Q;K;V] tensor into separate `attn.q_proj.weight`, `attn.k_proj.weight`, and `attn.v_proj.weight` tensors. Also splits `attn.in_proj_bias` into `attn.q_proj.bias`, `attn.k_proj.bias`, and `attn.v_proj.bias`. By default K and V are quantized to FP8; Q stays at FP16.

- **`--keep-attn-kv-fp16`**: when splitting attention, keeps K and V at FP16. Useful for isolating the impact of the structural split vs K/V quantization. Requires `--split-attn-qkv`.

- **`--attn-q-fp8`**: quantizes the Q projection to FP8 in intermediate blocks. Q is the most sensitive attention component. Requires `--split-attn-qkv`.

- **`--attn-out-fp8`**: quantizes `attn.out_proj.weight` to FP8 in intermediate blocks. Works independently of `--split-attn-qkv` because `out_proj` is already a separate tensor in the original format.

- **`--analyze` mode**: analyzes all weight tensors per block and prints quantization sensitivity metrics (Frobenius norm, max absolute value, std, outlier count and percentage relative to FP8 E4M3 range, and roundtrip NRMSE from FP16 → FP8 → FP16). Ranks blocks by weighted average quantization error, flags sensitive blocks (error > mean + 1 std), and emits a recommendation for `--first-blocks-keep`. The fused `in_proj_weight` is analyzed both as a whole and split into Q/K/V components.

- **`--first-blocks-keep` / `-k`**: sets the number of initial blocks to preserve at FP16 (default: 7).

- **`--dry-run`**: computes and prints the conversion summary without writing any file.

- **`--verbose` / `-v`**: prints per-tensor dtype mapping and detailed outlier information.

- **Tensor count verification**: after conversion, checks that the number of output tensors matches the expected count, accounting for the +4 net tensors per eligible block when `--split-attn-qkv` is active.

- **NaN/Inf sanity check**: runs on all FP16 tensors before saving.

- **Outlier warnings**: emitted for any tensor with values outside ±448 before FP8 quantization.

---

## [1.0.6] - T5 precision flag consolidation

### Changed

- `--t5-precision` is now the single canonical flag for controlling T5-XXL encoder
  precision. `--t5xxl-precision` has been removed.
- Internally, the CLI value is remapped to the `t5xxl` component key when building
  the precision map, so the override reaches `apply_precision_policy` correctly.

### Fixed

- `--t5-precision` was inert since v2.2: it wrote to `precision_map['t5']` but no
  extracted component carries that name (the component was renamed to `t5xxl` in
  v2.2). The flag is now functional.

## [1.0.5] - Chroma distilled model support

### Fixed

- Flux `transformer` component now recognises `distilled_guidance_layer.*` keys, present in Chroma distilled model variants. Without this pattern those keys were left unclassified and dropped from the extracted file.


## [1.0.4] - Richer analyze mode output

### Changed

- `--analyze` now displays **checkpoint metadata** at the top of the output (before the component breakdown), with individual values truncated to 200 characters. Previously metadata was shown as a single count at the bottom.
- Each component in the breakdown now includes a **dtype distribution summary** (e.g. `312 fp16, 4 fp32`), derived by reading tensor metadata via `safe_open` without loading weights into memory. Float8 variants are shown with their full type name (`fp8_e4m3fn`, `fp8_e5m2`).
- Sample key display increased from 3 to 10 keys per component.


## [1.0.3] - Flux key transform fixes for nested transformer prefix

### Fixed

- Flux `clip_l` and `t5xxl` key transforms now also strip the intermediate `.transformer.` level present in some Flux checkpoint variants. Added transform entries for `text_encoders.clip_l.transformer.*`, `text_encoder.clip_l.transformer.*`, `clip_l.transformer.*`, and the equivalent patterns for `t5xxl`. Previously, keys from these checkpoints would pass through untransformed and fail to load as standalone models in ComfyUI.


## [1.0.2] - Float8 support and Flux text encoder fixes

### Added

- **Float8 dtype support** (`torch.float8_e4m3fn`, `torch.float8_e5m2`): entries added to `DTYPE_BITS` conditionally at import time to maintain compatibility with PyTorch < 2.1, which does not have these types.
- `is_float8_dtype()` helper: centralizes float8 detection for code paths that need to special-case these dtypes (PyTorch does not implement `isinf`/`isnan` for float8).
- `--t5xxl-precision` flag for explicit precision control over the T5-XXL encoder in Flux models.

### Changed

- **Float8 tensors are never upsampled**: the default precision policy and explicit `--*-precision 16` both skip float8 tensors (keeping them as-is). Upsampling is only possible by passing `allow_upsampling=True` to `convert_tensor_to_16bit()` directly.
- `validate_conversion()` skips post-conversion checks when either the source or target dtype is float8, as `torch.isinf` and `torch.isnan` are not implemented for those types.
- `needs_16bit_conversion` detection now checks dtype directly (`float32`/`float64`) instead of comparing bit width, preventing float8 tensors from being mistakenly treated as candidates for downscaling.
- Output precision suffix is now derived from the **actual dtypes present in the output dict** rather than from conversion bookkeeping. New possible suffixes: `fp8` (pure float8 output), `fp8_mixed` (float8 + 16-bit in the same file), `fp32` (no conversion performed, all fp32).
- Extraction now prints `No precision conversion needed` when a component required no dtype changes.
- **Flux architecture**: `clip` component renamed to `clip_l`; `t5` renamed to `t5xxl`. Patterns updated to match the `text_encoders.clip_l.*` and `text_encoders.t5xxl.*` key layout used by Flux checkpoints, in addition to flatter alternatives (`clip_l.*`, `t5xxl.*`).
- `--t5xxl-precision` added to the component precision flag list in the CLI (alongside the existing `--t5-precision` for backward compatibility).


## [1.0.1] - Adaptive 16-bit conversion and post-conversion validation

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


## [1.0.0] - Universal extractor with architecture auto-detection

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

## [0.1.0] - Initial release

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
