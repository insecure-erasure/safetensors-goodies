#!/usr/bin/env python3
"""
Universal extractor for safetensors checkpoint files.
Automatically detects architecture and extracts components (UNet/Transformer, VAE, text encoders).
Supports: SDXL, SD 1.5/2.x, Flux, Lumina/zImage, PixArt, HunyuanDiT, and unknown architectures.

Key features:
- Auto-detection of model architecture
- Dynamic component classification
- Key transformation for standalone ComfyUI-compatible loading
- Precision conversion with validation
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

PRECISION_MAP = {
    'fp32': (torch.float32, 32),
    'fp16': (torch.float16, 16),
    'bf16': (torch.bfloat16, 16),
    'fp8': (torch.float8_e4m3fn, 8),
}

DTYPE_BITS = {
    torch.float32: 32,
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.float64: 64,
    torch.float8_e4m3fn: 8,
    torch.float8_e5m2: 8,
    torch.int8: 8,
    torch.uint8: 8,
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
}


def get_tensor_bits(dtype):
    """Returns the bit size of a tensor's dtype."""
    return DTYPE_BITS.get(dtype, 32)


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
            'clip': {
                'patterns': [
                    r'^text_encoders?\.',
                    r'^clip\.',
                ],
                'key_transforms': [
                    ('text_encoder.', ''),
                ],
            },
            't5': {
                'patterns': [
                    r'^t5\.',
                    r'^text_encoder_2\.',
                ],
                'key_transforms': [
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

# Precisions that are considered "low" and should be preserved by default
LOW_PRECISION_BITS = {8, 16}

# Default downscale target when fp32 is found and --keep-precision is not set
DEFAULT_DOWNSCALE_PRECISION = 'fp16'


def convert_tensor_precision(tensor, target_precision, component_name, key, explicit_request=False):
    """
    Converts a tensor to target precision with validation.

    Args:
        tensor: Input tensor
        target_precision: Target precision string ('fp32', 'fp16', 'bf16', 'fp8')
        component_name: Component name for warnings
        key: Tensor key for warnings
        explicit_request: If True, this was explicitly requested by user

    Returns:
        Converted tensor or original if conversion is invalid
    """
    if target_precision not in PRECISION_MAP:
        return tensor

    target_dtype, target_bits = PRECISION_MAP[target_precision]
    source_bits = get_tensor_bits(tensor.dtype)

    # Don't upscale - warn if explicitly requested
    if target_bits > source_bits:
        if explicit_request:
            warnings.warn(
                f"[{component_name}] '{key}' is {source_bits}-bit. "
                f"Cannot upscale to {target_bits}-bit ({target_precision}). Keeping original.",
                UserWarning
            )
        return tensor

    # Same precision, no conversion needed
    if tensor.dtype == target_dtype:
        return tensor

    # Convert
    return tensor.to(target_dtype)


def apply_precision_policy(
    tensors,
    component_name,
    explicit_precision=None,
    keep_precision=False
):
    """
    Apply precision policy to a dict of tensors.

    Policy:
    1. If explicit_precision is set: use it (unless it's an upscale)
    2. If keep_precision is True: keep original
    3. Default: downscale fp32 to fp16, keep fp8/fp16 as-is

    Args:
        tensors: Dict of {key: tensor}
        component_name: Component name for logging
        explicit_precision: User-requested precision (or None)
        keep_precision: If True, preserve original precision

    Returns:
        (converted_tensors, precision_suffix)
        precision_suffix is the string to add to filename, or None
    """
    converted = {}
    conversion_stats = defaultdict(int)
    precision_suffix = None

    for key, tensor in tensors.items():
        source_bits = get_tensor_bits(tensor.dtype)
        original_dtype = tensor.dtype

        if explicit_precision:
            # User explicitly requested a precision
            new_tensor = convert_tensor_precision(
                tensor, explicit_precision, component_name, key, explicit_request=True
            )
            precision_suffix = explicit_precision

        elif keep_precision:
            # Keep original precision
            new_tensor = tensor

        else:
            # Default policy: downscale fp32 to fp16, except VAE
            if source_bits > 16 and component_name != 'vae':
                new_tensor = convert_tensor_precision(
                    tensor, DEFAULT_DOWNSCALE_PRECISION, component_name, key, explicit_request=False
                )
            else:
                # fp16, bf16, fp8, or VAE - keep as is
                new_tensor = tensor

        converted[key] = new_tensor

        # Track conversions
        if new_tensor.dtype != original_dtype:
            conversion_stats[f"{original_dtype} → {new_tensor.dtype}"] += 1

    # Log conversions
    if conversion_stats:
        for conv, count in conversion_stats.items():
            print(f"    Converted {count} tensors: {conv}")

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
    force_architecture=None
):
    """
    Extract components from a checkpoint file.

    Args:
        input_path: Path to .safetensors file
        output_dir: Output directory
        components_to_extract: List of components to extract (None = all detected)
        precision_map: Dict mapping component names to explicit target precision
        keep_precision: If True, preserve original precision (unless overridden by precision_map)
        keep_original_keys: Don't transform keys
        force_architecture: Override auto-detection
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
        print(f"Precision policy: downscale fp32 → {DEFAULT_DOWNSCALE_PRECISION} (except VAE)")

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
            keep_precision=keep_precision
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

  # Extract all components (fp32 auto-downscaled to fp16)
  %(prog)s -i model.safetensors -d ./extracted

  # Extract keeping original precision
  %(prog)s -i model.safetensors -d ./extracted -k

  # Extract only VAE and UNet
  %(prog)s -i model.safetensors -d ./extracted -c vae -c unet

  # Keep precision but force VAE to fp16
  %(prog)s -i model.safetensors -d ./extracted -k --vae-precision fp16

  # List available components
  %(prog)s -i model.safetensors --list

Precision policy:
  - Default: fp32 tensors are downscaled to fp16, fp8/fp16 are kept as-is
  - Exception: VAE always keeps original precision (for quality)
  - With -k/--keep-precision: all tensors keep original precision
  - Explicit --*-precision flags override the policy for that component
  - Upscaling (e.g., fp16 to fp32) is never done; a warning is shown

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
        help='Keep original precision (default: downscale fp32 to fp16)'
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

    # Precision options
    for comp in ['vae', 'unet', 'transformer', 'dit', 'clip', 'clip_l', 'clip_g',
                 't5', 'text_encoder', 'text_encoder_2']:
        parser.add_argument(
            f'--{comp.replace("_", "-")}-precision',
            choices=['fp32', 'fp16', 'bf16', 'fp8'],
            help=f'Target precision for {comp}'
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
                 't5', 'text_encoder', 'text_encoder_2']:
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
            force_architecture=args.force_architecture
        )
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
