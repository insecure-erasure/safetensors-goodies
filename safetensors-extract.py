#!/usr/bin/env python3
"""
Universal extractor for safetensors checkpoint files.
Automatically detects architecture and extracts components (UNet/Transformer, VAE, text encoders).
Supports: SDXL, SD 1.5/2.x, Flux, Lumina/zImage, PixArt, HunyuanDiT, and unknown architectures.

Key features:
- Auto-detection of model architecture
- Dynamic component classification
- Key transformation for standalone ComfyUI-compatible loading
- Adaptive precision conversion (fp16/bf16) with validation
- Fallback extraction for unknown architectures
"""

import argparse
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch


# =============================================================================
# PRECISION HANDLING
# =============================================================================

# FP16 max representable value
FP16_MAX = 65504.0

DTYPE_BITS = {
    torch.float32: 32,
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.float64: 64,
    torch.int8: 8,
    torch.uint8: 8,
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
}

# Float8 dtypes (added in PyTorch 2.1+)
# These are used in some quantized models like Flux
try:
    DTYPE_BITS[torch.float8_e4m3fn] = 8
    DTYPE_BITS[torch.float8_e5m2] = 8
except AttributeError:
    pass  # Older PyTorch versions don't have float8


def is_float8_dtype(dtype):
    """Check if dtype is a float8 variant (not supported by isinf/isnan)."""
    try:
        return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    except AttributeError:
        return False


def get_tensor_bits(dtype):
    """Returns the bit size of a tensor's dtype."""
    return DTYPE_BITS.get(dtype, 32)


def tensor_fits_fp16(tensor):
    """
    Check if all values in a tensor fit within fp16 range (±65504).
    Returns (fits, max_abs_value)
    """
    if tensor.dtype in (torch.float32, torch.float64):
        max_abs = tensor.abs().max().item()
        return max_abs <= FP16_MAX, max_abs
    # Float8 types always fit in fp16 range (they have smaller range)
    if is_float8_dtype(tensor.dtype):
        return True, 0.0
    # Already fp16 or lower precision
    return True, 0.0


def validate_conversion(original, converted, key, component_name):
    """
    Validates precision conversion detecting potential issues.
    Always runs after conversion. Reports warnings but doesn't block.

    Returns list of warning messages.
    """
    warnings_list = []

    # Skip validation for float8 dtypes (isinf/isnan not implemented)
    if is_float8_dtype(original.dtype) or is_float8_dtype(converted.dtype):
        return warnings_list

    # 1. Check for new infinities (overflow)
    orig_inf_count = torch.isinf(original).sum().item()
    conv_inf_count = torch.isinf(converted).sum().item()
    if conv_inf_count > orig_inf_count:
        new_infs = conv_inf_count - orig_inf_count
        warnings_list.append(
            f"[{component_name}] '{key}': {new_infs} new inf values (overflow)"
        )

    # 2. Check for flush-to-zero (non-zero becoming zero)
    orig_nonzero_mask = original != 0
    orig_nonzero_count = orig_nonzero_mask.sum().item()
    if orig_nonzero_count > 0:
        # Compare in same dtype to avoid precision issues in comparison
        conv_zero_from_nonzero = (orig_nonzero_mask & (converted == 0)).sum().item()
        if conv_zero_from_nonzero > 0:
            pct = 100 * conv_zero_from_nonzero / orig_nonzero_count
            if pct > 0.001:  # Report if more than 0.001%
                warnings_list.append(
                    f"[{component_name}] '{key}': {conv_zero_from_nonzero} values "
                    f"flushed to zero ({pct:.4f}%)"
                )

    # 3. Check for new NaNs
    orig_nan = torch.isnan(original).sum().item()
    conv_nan = torch.isnan(converted).sum().item()
    if conv_nan > orig_nan:
        warnings_list.append(
            f"[{component_name}] '{key}': {conv_nan - orig_nan} new NaN values"
        )

    return warnings_list


def convert_tensor_to_16bit(tensor, force_bf16=False, allow_upsampling=False):
    """
    Convert a tensor to 16-bit precision.

    Args:
        tensor: Input tensor
        force_bf16: If True, always use bf16. If False, use fp16 if values fit.
        allow_upsampling: If False, don't convert lower precision (e.g., float8) to higher.

    Returns:
        Converted tensor (fp16 or bf16), or original if no conversion needed/allowed
    """
    if tensor.dtype in (torch.float16, torch.bfloat16):
        # Already 16-bit
        return tensor

    # Float8 types: only convert if upsampling is explicitly allowed
    if is_float8_dtype(tensor.dtype):
        if allow_upsampling:
            return tensor.to(torch.float16)
        else:
            # Keep original float8 precision
            return tensor

    if tensor.dtype not in (torch.float32, torch.float64):
        # Not a float type we convert (e.g., int tensors)
        return tensor

    if force_bf16:
        return tensor.to(torch.bfloat16)

    # Check if fp16 is safe
    fits, _ = tensor_fits_fp16(tensor)
    if fits:
        return tensor.to(torch.float16)
    else:
        return tensor.to(torch.bfloat16)


def analyze_component_for_fp16(tensors):
    """
    Analyze all tensors in a component to determine if fp16 is safe for all.

    Args:
        tensors: Dict of {key: tensor}

    Returns:
        (all_fit_fp16, problematic_keys)
        - all_fit_fp16: True if all tensors fit in fp16 range
        - problematic_keys: List of (key, max_abs_value) for tensors that don't fit
    """
    problematic = []
    for key, tensor in tensors.items():
        # Float8 always fits in fp16
        if is_float8_dtype(tensor.dtype):
            continue
        if tensor.dtype in (torch.float32, torch.float64):
            fits, max_abs = tensor_fits_fp16(tensor)
            if not fits:
                problematic.append((key, max_abs))

    return len(problematic) == 0, problematic


# =============================================================================
# ARCHITECTURE DETECTION
# =============================================================================

WRAPPER_PREFIXES = [
    'model.diffusion_model.',
    'model.',
    'diffusion_model.',
]

ARCHITECTURE_PATTERNS = {
    'SDXL': {
        'indicators': ['conditioner.embedders.0', 'conditioner.embedders.1'],
        'required_negative': [],
        'components': {
            'unet': {
                'patterns': [
                    r'^input_blocks\.',
                    r'^output_blocks\.',
                    r'^middle_block\.',
                    r'^time_embed\.',
                    r'^out\.',
                    r'^label_emb\.',
                ],
                'key_transforms': [
                    ('model.diffusion_model.', ''),
                ],
            },
            'clip_l': {
                'patterns': [
                    r'^conditioner\.embedders\.0\.',
                ],
                'key_transforms': [
                    ('conditioner.embedders.0.transformer.', ''),
                    ('conditioner.embedders.0.', ''),
                ],
            },
            'clip_g': {
                'patterns': [
                    r'^conditioner\.embedders\.1\.',
                ],
                'key_transforms': [
                    ('conditioner.embedders.1.model.', ''),
                    ('conditioner.embedders.1.', ''),
                ],
            },
            'vae': {
                'patterns': [
                    r'^first_stage_model\.',
                ],
                'key_transforms': [
                    ('first_stage_model.', ''),
                ],
            },
        },
    },
    'SD15': {
        'indicators': ['cond_stage_model.transformer'],
        'required_negative': ['conditioner.embedders.1'],
        'components': {
            'unet': {
                'patterns': [
                    r'^input_blocks\.',
                    r'^output_blocks\.',
                    r'^middle_block\.',
                    r'^time_embed\.',
                    r'^out\.',
                ],
                'key_transforms': [
                    ('model.diffusion_model.', ''),
                ],
            },
            'clip': {
                'patterns': [
                    r'^cond_stage_model\.',
                ],
                'key_transforms': [
                    ('cond_stage_model.transformer.', ''),
                    ('cond_stage_model.', ''),
                ],
            },
            'vae': {
                'patterns': [
                    r'^first_stage_model\.',
                ],
                'key_transforms': [
                    ('first_stage_model.', ''),
                ],
            },
        },
    },
    'Flux': {
        'indicators': ['double_blocks', 'single_blocks', 'img_in', 'txt_in'],
        'required_negative': [],
        'components': {
            'transformer': {
                'patterns': [
                    r'^double_blocks\.',
                    r'^single_blocks\.',
                    r'^img_in\.',
                    r'^txt_in\.',
                    r'^time_in\.',
                    r'^vector_in\.',
                    r'^guidance_in\.',
                    r'^final_layer\.',
                ],
                'key_transforms': [
                    ('model.diffusion_model.', ''),
                ],
            },
            'clip_l': {
                'patterns': [
                    r'^text_encoders\.clip_l\.',
                    r'^text_encoder\.clip_l\.',
                    r'^clip_l\.',
                    r'^clip\.',
                ],
                'key_transforms': [
                    ('text_encoders.clip_l.transformer.', ''),
                    ('text_encoder.clip_l.transformer.', ''),
                    ('text_encoders.clip_l.', ''),
                    ('text_encoder.clip_l.', ''),
                    ('clip_l.transformer.', ''),
                    ('clip_l.', ''),
                ],
            },
            't5xxl': {
                'patterns': [
                    r'^text_encoders\.t5xxl\.',
                    r'^text_encoder\.t5xxl\.',
                    r'^t5xxl\.',
                    r'^t5\.',
                    r'^text_encoder_2\.',
                ],
                'key_transforms': [
                    ('text_encoders.t5xxl.transformer.', ''),
                    ('text_encoder.t5xxl.transformer.', ''),
                    ('text_encoders.t5xxl.', ''),
                    ('text_encoder.t5xxl.', ''),
                    ('t5xxl.transformer.', ''),
                    ('t5xxl.', ''),
                    ('text_encoder_2.', ''),
                ],
            },
            'vae': {
                'patterns': [
                    r'^vae\.',
                    r'^first_stage_model\.',
                ],
                'key_transforms': [
                    ('first_stage_model.', ''),
                    ('vae.', ''),
                ],
            },
        },
    },
    'Lumina': {
        'indicators': ['cap_embedder', 'noise_refiner', 'context_refiner'],
        'secondary_indicators': ['layers.', 'final_layer', 't_embedder'],
        'required_negative': ['double_blocks', 'single_blocks'],
        'components': {
            'dit': {
                'patterns': [
                    r'^layers\.',
                    r'^noise_refiner\.',
                    r'^context_refiner\.',
                    r'^final_layer\.',
                    r'^norm_final',
                    r'^cap_embedder\.',
                    r'^t_embedder\.',
                    r'^x_embedder\.',
                    r'^cap_pad_token',
                    r'^x_pad_token',
                ],
                'key_transforms': [],
            },
            'text_encoder': {
                'patterns': [
                    r'^text_model\.',
                    r'^text_encoder\.',
                    r'^clip\.',
                ],
                'key_transforms': [],
            },
            'vae': {
                'patterns': [
                    r'^vae\.',
                    r'^decoder\.',
                    r'^encoder\.',
                    r'^first_stage_model\.',
                ],
                'key_transforms': [
                    ('first_stage_model.', ''),
                ],
            },
        },
    },
    'PixArt': {
        'indicators': ['adaln_single', 'y_embedder'],
        'secondary_indicators': ['blocks.'],
        'required_negative': ['cap_embedder', 'noise_refiner'],
        'components': {
            'dit': {
                'patterns': [
                    r'^blocks\.',
                    r'^adaln_single\.',
                    r'^y_embedder\.',
                    r'^x_embedder\.',
                    r'^t_embedder\.',
                    r'^t_block\.',
                    r'^pos_embed',
                    r'^final_layer\.',
                ],
                'key_transforms': [],
            },
            'text_encoder': {
                'patterns': [
                    r'^text_encoder\.',
                ],
                'key_transforms': [],
            },
            'vae': {
                'patterns': [
                    r'^vae\.',
                ],
                'key_transforms': [],
            },
        },
    },
    'HunyuanDiT': {
        'indicators': ['pooler', 'style_embedder', 'extra_embedder'],
        'required_negative': [],
        'components': {
            'dit': {
                'patterns': [
                    r'^blocks\.',
                    r'^pooler\.',
                    r'^style_embedder\.',
                    r'^extra_embedder\.',
                    r'^x_embedder\.',
                    r'^t_embedder\.',
                    r'^mlp_t5\.',
                    r'^final_layer\.',
                ],
                'key_transforms': [],
            },
            'text_encoder': {
                'patterns': [
                    r'^text_encoder\.',
                ],
                'key_transforms': [],
            },
            'text_encoder_2': {
                'patterns': [
                    r'^text_encoder_2\.',
                ],
                'key_transforms': [],
            },
            'vae': {
                'patterns': [
                    r'^vae\.',
                ],
                'key_transforms': [],
            },
        },
    },
}

# Generic fallback patterns for unknown architectures
GENERIC_COMPONENT_PATTERNS = {
    'unet': [
        (r'input_blocks', 'UNet input blocks'),
        (r'output_blocks', 'UNet output blocks'),
        (r'middle_block', 'UNet middle block'),
        (r'time_embed', 'Time embedding'),
        (r'diffusion_model', 'Diffusion model'),
    ],
    'transformer': [
        (r'double_blocks', 'DiT double blocks'),
        (r'single_blocks', 'DiT single blocks'),
        (r'layers\.', 'Transformer layers'),
        (r'blocks\.', 'Transformer blocks'),
        (r'final_layer', 'Final layer'),
    ],
    'text_encoder': [
        (r'text_model', 'Text model'),
        (r'text_encoder', 'Text encoder'),
        (r'clip', 'CLIP'),
        (r'cond_stage', 'Conditioning stage'),
        (r'conditioner\.embedders', 'Conditioner embedders'),
        (r't5', 'T5'),
    ],
    'vae': [
        (r'vae', 'VAE'),
        (r'first_stage', 'First stage (VAE)'),
        (r'decoder\.', 'Decoder'),
        (r'encoder\.', 'Encoder'),
        (r'quant_conv', 'Quantization conv'),
        (r'post_quant', 'Post-quantization'),
    ],
}


def strip_wrapper_prefix(key):
    """Remove common wrapper prefixes from a key."""
    for prefix in WRAPPER_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def detect_architecture(keys):
    """
    Detect the model architecture based on tensor key patterns.
    Returns (architecture_name, confidence, wrapper_prefix).
    """
    # Detect wrapper prefix
    wrapper_prefix = None
    for prefix in WRAPPER_PREFIXES:
        matching = sum(1 for k in keys if k.startswith(prefix))
        if matching > len(keys) * 0.3:
            wrapper_prefix = prefix
            break

    # Strip wrapper for detection
    stripped_keys = [strip_wrapper_prefix(k) for k in keys]
    key_set = set(stripped_keys)

    scores = {}
    for arch_name, arch_info in ARCHITECTURE_PATTERNS.items():
        score = 0
        total_weight = 0

        # Primary indicators (weight 2)
        for indicator in arch_info['indicators']:
            total_weight += 2
            if any(indicator in k for k in stripped_keys):
                score += 2

        # Secondary indicators (weight 1)
        if 'secondary_indicators' in arch_info:
            for indicator in arch_info['secondary_indicators']:
                total_weight += 1
                if any(indicator in k for k in stripped_keys):
                    score += 1

        # Negative indicators (disqualify)
        if 'required_negative' in arch_info:
            for neg in arch_info['required_negative']:
                if any(neg in k for k in stripped_keys):
                    score = 0
                    break

        if score > 0 and total_weight > 0:
            scores[arch_name] = score / total_weight

    if scores:
        best = max(scores, key=scores.get)
        return best, scores[best], wrapper_prefix

    return 'Unknown', 0.0, wrapper_prefix


def classify_tensor_by_architecture(key, architecture, wrapper_prefix=None):
    """
    Classify a tensor key into a component based on detected architecture.
    Returns (component_name, stripped_key).
    """
    stripped_key = key
    if wrapper_prefix and key.startswith(wrapper_prefix):
        stripped_key = key[len(wrapper_prefix):]

    if architecture in ARCHITECTURE_PATTERNS:
        components = ARCHITECTURE_PATTERNS[architecture]['components']
        for comp_name, comp_info in components.items():
            for pattern in comp_info['patterns']:
                if re.match(pattern, stripped_key):
                    return comp_name, stripped_key

    return None, stripped_key


def classify_tensor_generic(key, wrapper_prefix=None):
    """
    Generic classification for unknown architectures.
    Returns (component_name, stripped_key).
    """
    stripped_key = key
    if wrapper_prefix and key.startswith(wrapper_prefix):
        stripped_key = key[len(wrapper_prefix):]

    key_lower = stripped_key.lower()

    # Check each category
    for comp_name, patterns in GENERIC_COMPONENT_PATTERNS.items():
        for pattern, _ in patterns:
            if re.search(pattern, key_lower):
                return comp_name, stripped_key

    return 'unknown', stripped_key


def transform_key(key, component, architecture):
    """
    Transform a checkpoint key to standalone format.
    """
    if architecture not in ARCHITECTURE_PATTERNS:
        # For unknown architectures, just strip wrapper
        return strip_wrapper_prefix(key)

    comp_info = ARCHITECTURE_PATTERNS[architecture]['components'].get(component)
    if not comp_info:
        return strip_wrapper_prefix(key)

    # First strip wrapper
    transformed = strip_wrapper_prefix(key)

    # Then apply component-specific transforms
    for old_prefix, new_prefix in comp_info.get('key_transforms', []):
        if transformed.startswith(old_prefix):
            transformed = new_prefix + transformed[len(old_prefix):]
            break

    return transformed


# =============================================================================
# PRECISION CONVERSION
# =============================================================================

def apply_precision_policy(
    tensors,
    component_name,
    explicit_precision=None,
    keep_precision=False,
    mixed_dtype=False
):
    """
    Apply precision policy to a dict of tensors.

    Policy:
    1. If explicit_precision is set (16 or 32): apply it
    2. If keep_precision is True: keep original
    3. Default: downscale fp32 to 16-bit (except VAE)

    For 16-bit conversion:
    - By default, analyze all tensors and pick fp16 or bf16 for the whole component
    - If mixed_dtype is True, decide per-tensor (may produce mixed fp16/bf16)

    Args:
        tensors: Dict of {key: tensor}
        component_name: Component name for logging
        explicit_precision: User-requested precision (16 or 32) or None
        keep_precision: If True, preserve original precision
        mixed_dtype: If True, allow mixed fp16/bf16 per tensor

    Returns:
        (converted_tensors, precision_suffix)
        precision_suffix is the string to add to filename, or None
    """
    converted = {}
    conversion_stats = defaultdict(int)
    precision_suffix = None
    all_warnings = []

    # Determine if we need to do 16-bit conversion
    needs_16bit_conversion = False
    if explicit_precision == 16:
        needs_16bit_conversion = True
    elif not keep_precision and not explicit_precision:
        # Default policy: downscale fp32 to 16-bit, except VAE
        # Note: we no longer upscale float8 to fp16 (that would increase file size)
        if component_name != 'vae':
            # Check if any tensor is fp32/fp64 (higher than 16-bit)
            for tensor in tensors.values():
                if tensor.dtype in (torch.float32, torch.float64):
                    needs_16bit_conversion = True
                    break

    if needs_16bit_conversion and not mixed_dtype:
        # Analyze component to decide fp16 vs bf16
        all_fit, problematic = analyze_component_for_fp16(tensors)
        if all_fit:
            target_dtype = torch.float16
            precision_suffix = 'fp16'
            print(f"    All tensors fit in fp16 range")
        else:
            target_dtype = torch.bfloat16
            precision_suffix = 'bf16'
            print(f"    {len(problematic)} tensor(s) exceed fp16 range, using bf16 for component")
            for key, max_val in problematic[:3]:
                print(f"      - {key}: max |value| = {max_val:.2e}")
            if len(problematic) > 3:
                print(f"      ... and {len(problematic) - 3} more")

    # Track dtypes used (for mixed mode)
    dtypes_used = set()

    for key, tensor in tensors.items():
        original_dtype = tensor.dtype
        source_bits = get_tensor_bits(tensor.dtype)

        if explicit_precision == 32:
            # Keep as fp32 (or upscale if lower, which we don't do)
            if source_bits < 32:
                warnings.warn(
                    f"[{component_name}] '{key}' is {source_bits}-bit. "
                    f"Cannot upscale to 32-bit. Keeping original.",
                    UserWarning
                )
                new_tensor = tensor
            else:
                new_tensor = tensor
            precision_suffix = 'fp32'

        elif explicit_precision == 16 or (needs_16bit_conversion and not keep_precision):
            # Convert to 16-bit (only downscale, never upscale)
            if is_float8_dtype(tensor.dtype):
                # Float8: keep as is (no upsampling to fp16)
                new_tensor = tensor
            elif source_bits <= 16:
                # Already 16-bit (fp16/bf16), keep as is
                new_tensor = tensor
            else:
                # fp32/fp64: downscale to 16-bit
                if mixed_dtype:
                    # Per-tensor decision
                    new_tensor = convert_tensor_to_16bit(tensor, force_bf16=False)
                else:
                    # Use component-wide decision
                    new_tensor = tensor.to(target_dtype)

                # Validate conversion
                conv_warnings = validate_conversion(tensor, new_tensor, key, component_name)
                all_warnings.extend(conv_warnings)

            dtypes_used.add(new_tensor.dtype)

        elif keep_precision:
            # Keep original precision
            new_tensor = tensor

        else:
            # Default: keep as is (VAE or already low precision)
            new_tensor = tensor

        converted[key] = new_tensor

        # Track conversions
        if new_tensor.dtype != original_dtype:
            conversion_stats[f"{original_dtype} → {new_tensor.dtype}"] += 1

    # Determine precision suffix based on actual dtypes in output
    # Check for float8 types
    has_float8_output = any(is_float8_dtype(t.dtype) for t in converted.values())
    has_fp16_output = any(t.dtype == torch.float16 for t in converted.values())
    has_bf16_output = any(t.dtype == torch.bfloat16 for t in converted.values())
    has_fp32_output = any(t.dtype == torch.float32 for t in converted.values())

    # Build suffix based on what's actually in the output
    if has_float8_output:
        if has_fp16_output or has_bf16_output:
            # Mixed float8 + 16-bit
            precision_suffix = 'fp8_mixed'
        else:
            # Pure float8
            precision_suffix = 'fp8'
    elif mixed_dtype and needs_16bit_conversion:
        if has_fp16_output and has_bf16_output:
            precision_suffix = 'fp16_bf16_mixed'
            warnings.warn(
                f"[{component_name}] Output file contains mixed fp16/bf16 tensors. "
                "This may not be compatible with all model loaders.",
                UserWarning
            )
        elif has_bf16_output:
            precision_suffix = 'bf16'
        elif has_fp16_output:
            precision_suffix = 'fp16'
    elif has_fp32_output and not has_fp16_output and not has_bf16_output:
        precision_suffix = 'fp32'
    # else: precision_suffix keeps the value set earlier (fp16/bf16 from component analysis)

    # Log conversions
    if conversion_stats:
        for conv, count in conversion_stats.items():
            print(f"    Converted {count} tensors: {conv}")
    else:
        print(f"    No precision conversion needed")

    # Log validation warnings
    for w in all_warnings:
        warnings.warn(w, UserWarning)

    return converted, precision_suffix


# =============================================================================
# EXTRACTION LOGIC
# =============================================================================

def analyze_checkpoint(input_path):
    """
    Analyze checkpoint structure without extracting.
    """
    print(f"Analyzing: {input_path}")
    print(f"File size: {os.path.getsize(input_path) / (1024**3):.3f} GB\n")

    with safe_open(input_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        metadata = f.metadata()

    architecture, confidence, wrapper_prefix = detect_architecture(keys)

    print("=" * 70)
    print(f"Detected architecture: {architecture} (confidence: {confidence:.0%})")
    if wrapper_prefix:
        print(f"Wrapper prefix: '{wrapper_prefix}'")
    print("=" * 70)

    # Classify all tensors
    components = defaultdict(list)
    for key in keys:
        if architecture != 'Unknown':
            comp, stripped = classify_tensor_by_architecture(key, architecture, wrapper_prefix)
        else:
            comp, stripped = classify_tensor_generic(key, wrapper_prefix)

        if comp is None:
            comp, _ = classify_tensor_generic(key, wrapper_prefix)

        components[comp or 'unknown'].append(key)

    print("\nComponent breakdown:")
    for comp, comp_keys in sorted(components.items()):
        print(f"\n  {comp}: {len(comp_keys)} tensors")
        # Show sample keys
        for k in comp_keys[:3]:
            transformed = transform_key(k, comp, architecture)
            if k != transformed:
                print(f"    {k}")
                print(f"      → {transformed}")
            else:
                print(f"    {k}")
        if len(comp_keys) > 3:
            print(f"    ... and {len(comp_keys) - 3} more")

    if metadata:
        print(f"\nMetadata: {len(metadata)} entries")

    return architecture, confidence, wrapper_prefix, components


def extract_components(
    input_path,
    output_dir,
    components_to_extract=None,
    precision_map=None,
    keep_precision=False,
    keep_original_keys=False,
    force_architecture=None,
    mixed_dtype=False
):
    """
    Extract components from a checkpoint file.

    Args:
        input_path: Path to .safetensors file
        output_dir: Output directory
        components_to_extract: List of components to extract (None = all detected)
        precision_map: Dict mapping component names to explicit target precision (16 or 32)
        keep_precision: If True, preserve original precision (unless overridden by precision_map)
        keep_original_keys: Don't transform keys
        force_architecture: Override auto-detection
        mixed_dtype: Allow mixed fp16/bf16 per tensor
    """
    precision_map = precision_map or {}

    print(f"Loading: {input_path}")
    print(f"File size: {os.path.getsize(input_path) / (1024**3):.3f} GB\n")

    # Get keys for detection
    with safe_open(input_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

    # Detect or use forced architecture
    if force_architecture:
        architecture = force_architecture
        confidence = 1.0
        wrapper_prefix = None
        for prefix in WRAPPER_PREFIXES:
            if sum(1 for k in keys if k.startswith(prefix)) > len(keys) * 0.3:
                wrapper_prefix = prefix
                break
        print(f"Using forced architecture: {architecture}")
    else:
        architecture, confidence, wrapper_prefix = detect_architecture(keys)
        print(f"Detected architecture: {architecture} (confidence: {confidence:.0%})")

    if wrapper_prefix:
        print(f"Wrapper prefix: '{wrapper_prefix}'")

    # Show precision policy
    if keep_precision:
        print("Precision policy: keep original")
    else:
        print("Precision policy: downscale fp32 → 16-bit adaptive (fp16 if fits, else bf16)")
        print("                  float8 kept as-is (no upsampling)")
        print("                  VAE keeps original precision")

    if mixed_dtype:
        print("Mixed dtype mode: enabled (per-tensor fp16/bf16 decision)")

    # Load full state dict
    print("\nLoading tensors...")
    state_dict = load_file(input_path)

    # Classify all tensors
    component_tensors = defaultdict(dict)
    unknown_keys = []

    for key, tensor in state_dict.items():
        if architecture != 'Unknown':
            comp, stripped = classify_tensor_by_architecture(key, architecture, wrapper_prefix)
        else:
            comp, stripped = classify_tensor_generic(key, wrapper_prefix)

        if comp is None:
            comp, stripped = classify_tensor_generic(key, wrapper_prefix)

        if comp == 'unknown':
            unknown_keys.append(key)
        else:
            # Transform key unless keeping originals
            if keep_original_keys:
                new_key = key
            else:
                new_key = transform_key(key, comp, architecture)

            component_tensors[comp][new_key] = tensor

    # Summary
    print("\nClassification summary:")
    all_components = list(component_tensors.keys())
    if unknown_keys:
        all_components.append('unknown')

    for comp in sorted(all_components):
        if comp == 'unknown':
            tensors = {k: state_dict[k] for k in unknown_keys}
        else:
            tensors = component_tensors[comp]

        total_params = sum(t.numel() for t in tensors.values())
        total_mb = sum(t.numel() * t.element_size() for t in tensors.values()) / (1024**2)

        extract_marker = "✓" if (components_to_extract is None or comp in components_to_extract) else "✗"
        print(f"  [{extract_marker}] {comp}: {len(tensors)} tensors, {total_params:,} params, {total_mb:.1f} MB")

    if unknown_keys:
        print(f"\n⚠ {len(unknown_keys)} unclassified tensors:")
        for k in unknown_keys[:5]:
            print(f"    - {k}")
        if len(unknown_keys) > 5:
            print(f"    ... and {len(unknown_keys) - 5} more")

    # Determine what to extract
    if components_to_extract is None:
        components_to_extract = list(component_tensors.keys())

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(input_path).stem

    # Extract each component
    print("\nExtracting components...")
    extracted_files = []

    for comp in components_to_extract:
        if comp not in component_tensors:
            print(f"  ⚠ {comp}: not found in checkpoint")
            continue

        tensors = component_tensors[comp]
        if not tensors:
            continue

        print(f"  Processing {comp}...")

        # Apply precision policy
        explicit_precision = precision_map.get(comp)
        converted_tensors, precision_suffix = apply_precision_policy(
            tensors,
            comp,
            explicit_precision=explicit_precision,
            keep_precision=keep_precision,
            mixed_dtype=mixed_dtype
        )

        # Build filename
        if precision_suffix:
            filename = f"{base_name}_{comp}.{precision_suffix}.safetensors"
        else:
            filename = f"{base_name}_{comp}.safetensors"

        output_path = os.path.join(output_dir, filename)
        print(f"    Saving → {filename}")

        save_file(converted_tensors, output_path)

        size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"    ✓ {size_mb:.1f} MB, {len(converted_tensors)} tensors")

        # Show sample keys
        sample = list(converted_tensors.keys())[:2]
        print(f"    Sample keys: {sample}")

        extracted_files.append(output_path)

    print(f"\n✓ Extraction complete. {len(extracted_files)} files created.")
    return extracted_files


def list_components(input_path):
    """
    List available components in a checkpoint.
    """
    with safe_open(input_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

    architecture, confidence, wrapper_prefix = detect_architecture(keys)

    components = set()
    for key in keys:
        if architecture != 'Unknown':
            comp, _ = classify_tensor_by_architecture(key, architecture, wrapper_prefix)
        else:
            comp, _ = classify_tensor_generic(key, wrapper_prefix)

        if comp is None:
            comp, _ = classify_tensor_generic(key, wrapper_prefix)

        components.add(comp or 'unknown')

    return architecture, sorted(components)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Universal safetensors component extractor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze checkpoint structure
  %(prog)s -i model.safetensors --analyze

  # Extract all components (fp32 auto-downscaled to 16-bit)
  %(prog)s -i model.safetensors -d ./extracted

  # Extract keeping original precision
  %(prog)s -i model.safetensors -d ./extracted -k

  # Extract only VAE and UNet
  %(prog)s -i model.safetensors -d ./extracted -c vae -c unet

  # Force VAE to 16-bit precision
  %(prog)s -i model.safetensors -d ./extracted --vae-precision 16

  # Allow mixed fp16/bf16 per tensor
  %(prog)s -i model.safetensors -d ./extracted -m

  # List available components
  %(prog)s -i model.safetensors --list

Precision policy:
  - Default: fp32 tensors are downscaled to 16-bit (fp16 if values fit, bf16 otherwise)
  - Exception: VAE always keeps original precision (for quality)
  - With -k/--keep-precision: all tensors keep original precision
  - Explicit --*-precision flags override the policy for that component
  - With -m/--mixed-dtype: per-tensor fp16/bf16 decision (may cause compatibility issues)

Supported architectures: SDXL, SD15, Flux, Lumina, PixArt, HunyuanDiT
Unknown architectures are handled with generic pattern matching.
        """
    )

    parser.add_argument('-i', '--input', required=True, help='Input .safetensors file')
    parser.add_argument('-d', '--output-dir', help='Output directory for extracted components')

    parser.add_argument(
        '-c', '--component',
        action='append',
        dest='components',
        help='Component to extract (can be repeated). Default: all detected'
    )

    parser.add_argument('--analyze', action='store_true', help='Analyze without extracting')
    parser.add_argument('--list', action='store_true', help='List available components')

    parser.add_argument(
        '-k', '--keep-precision',
        action='store_true',
        help='Keep original precision (default: downscale fp32 to 16-bit)'
    )

    parser.add_argument(
        '-m', '--mixed-dtype',
        action='store_true',
        help='Allow mixed fp16/bf16 per tensor (may cause compatibility issues)'
    )

    parser.add_argument(
        '--force-architecture',
        choices=list(ARCHITECTURE_PATTERNS.keys()),
        help='Override architecture detection'
    )

    parser.add_argument(
        '--keep-original-keys',
        action='store_true',
        help='Keep original keys (for debugging)'
    )

    # Precision options (now accept 16 or 32)
    for comp in ['vae', 'unet', 'transformer', 'dit', 'clip', 'clip_l', 'clip_g',
                 't5', 't5xxl', 'text_encoder', 'text_encoder_2']:
        parser.add_argument(
            f'--{comp.replace("_", "-")}-precision',
            type=int,
            choices=[16, 32],
            help=f'Target precision for {comp} (16 or 32 bits)'
        )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"✗ Error: File not found: {args.input}")
        return 1

    if not args.input.endswith('.safetensors'):
        print(f"✗ Error: File must be .safetensors")
        return 1

    # List mode
    if args.list:
        arch, components = list_components(args.input)
        print(f"Architecture: {arch}")
        print(f"Components: {', '.join(components)}")
        return 0

    # Analyze mode
    if args.analyze:
        analyze_checkpoint(args.input)
        return 0

    # Extract mode requires output dir
    if not args.output_dir:
        print("✗ Error: --output-dir required for extraction")
        return 1

    # Build precision map from args
    precision_map = {}
    for comp in ['vae', 'unet', 'transformer', 'dit', 'clip', 'clip_l', 'clip_g',
                 't5', 't5xxl', 'text_encoder', 'text_encoder_2']:
        attr = f'{comp}_precision'
        val = getattr(args, attr, None)
        if val:
            precision_map[comp] = val

    try:
        extract_components(
            args.input,
            args.output_dir,
            components_to_extract=args.components,
            precision_map=precision_map,
            keep_precision=args.keep_precision,
            keep_original_keys=args.keep_original_keys,
            force_architecture=args.force_architecture,
            mixed_dtype=args.mixed_dtype
        )
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
