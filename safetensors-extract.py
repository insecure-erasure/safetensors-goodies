#!/usr/bin/env python3
"""
Extracts individual components (UNet, CLIP-L, CLIP-G, VAE) from SDXL/Pony models
Saves each component in separate safetensors files with appropriate suffixes
Supports independent extraction and multiple precision formats (fp4, fp8, fp16, fp32)

FIXED: Transforms keys to ComfyUI-compatible format for standalone component loading
"""

import argparse
import os
from pathlib import Path
from safetensors.torch import load_file, save_file
import torch
import warnings


# Precision mapping with bit sizes
PRECISION_MAP = {
    'fp32': (torch.float32, 32),
    'fp16': (torch.float16, 16),
    'fp8': (torch.float8_e4m3fn, 8),
    'fp4': (torch.float8_e4m3fn, 4)  # Note: PyTorch doesn't have native fp4, using fp8 as fallback
}

# Key prefix mappings for ComfyUI compatibility
# Maps checkpoint prefixes to standalone model prefixes
KEY_TRANSFORMS = {
    'vae': [
        ('first_stage_model.', ''),  # Remove first_stage_model prefix
    ],
    'clip_l': [
        # SDXL format: conditioner.embedders.0.transformer -> clip_l format
        ('conditioner.embedders.0.transformer.', ''),
        # Some models use cond_stage_model
        ('cond_stage_model.transformer.', ''),
        ('cond_stage_model.', ''),
    ],
    'clip_g': [
        # SDXL format: conditioner.embedders.1.model -> clip_g format
        ('conditioner.embedders.1.model.', ''),
        ('conditioner.embedders.1.', ''),
    ],
    'unet': [
        ('model.diffusion_model.', ''),  # Remove model.diffusion_model prefix
    ],
}


def get_tensor_bits(dtype):
    """Returns the bit size of a tensor's dtype"""
    dtype_bits = {
        torch.float32: 32,
        torch.float16: 16,
        torch.bfloat16: 16,
        torch.float8_e4m3fn: 8,
        torch.float8_e5m2: 8,
        torch.int8: 8,
        torch.uint8: 8,
    }
    return dtype_bits.get(dtype, 32)  # Default to 32 if unknown


def identify_component(key):
    """
    Identifies which component a state dict key belongs to.

    SDXL checkpoint structure:
    - model.diffusion_model.* -> UNet
    - first_stage_model.* -> VAE
    - conditioner.embedders.0.* -> CLIP-L (text encoder 1)
    - conditioner.embedders.1.* -> CLIP-G (text encoder 2, OpenCLIP ViT-bigG)
    """
    # Order matters! Check most specific patterns first

    # CLIP-G: Second text encoder (OpenCLIP ViT-bigG)
    if key.startswith('conditioner.embedders.1.'):
        return 'clip_g'

    # CLIP-L: First text encoder
    if key.startswith('conditioner.embedders.0.'):
        return 'clip_l'

    # Alternative CLIP location (some SD 1.x/2.x models)
    if key.startswith('cond_stage_model.'):
        return 'clip_l'

    # VAE: First stage model (autoencoder)
    if key.startswith('first_stage_model.'):
        return 'vae'

    # UNet: Diffusion model
    if key.startswith('model.diffusion_model.'):
        return 'unet'

    # Fallback for alternative naming conventions
    key_lower = key.lower()

    if 'diffusion_model' in key_lower or key_lower.startswith('unet.'):
        return 'unet'

    if key_lower.startswith('vae.'):
        return 'vae'

    return 'unknown'


def transform_key(key, component):
    """
    Transforms a checkpoint key to the standalone format expected by ComfyUI.

    Args:
        key: Original key from checkpoint
        component: Component name ('vae', 'clip_l', 'clip_g', 'unet')

    Returns:
        Transformed key for standalone model
    """
    if component not in KEY_TRANSFORMS:
        return key

    transforms = KEY_TRANSFORMS[component]
    for old_prefix, new_prefix in transforms:
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix):]

    return key


def convert_tensor_precision(tensor, target_precision, source_bits, component_name, key):
    """
    Converts a tensor to target precision if valid

    Args:
        tensor: Input tensor
        target_precision: Target dtype string ('fp32', 'fp16', 'fp8', 'fp4')
        source_bits: Original bit size of the tensor
        component_name: Name of the component for warning messages
        key: Tensor key for warning messages

    Returns:
        Converted tensor or original tensor if conversion is invalid
    """
    if target_precision not in PRECISION_MAP:
        return tensor

    target_dtype, target_bits = PRECISION_MAP[target_precision]

    # Check if trying to upscale precision
    if target_bits > source_bits:
        warnings.warn(
            f"[{component_name}] Tensor '{key}' has {source_bits} bits. "
            f"Cannot upscale to {target_bits} bits ({target_precision}). Keeping original precision.",
            UserWarning
        )
        return tensor

    # FP4 handling (PyTorch doesn't have native fp4)
    if target_precision == 'fp4':
        warnings.warn(
            f"[{component_name}] PyTorch doesn't have native fp4. Using fp8 instead for tensor '{key}'.",
            UserWarning
        )
        return tensor.to(torch.float8_e4m3fn) if tensor.dtype != torch.float8_e4m3fn else tensor

    # Convert to target precision
    if tensor.dtype != target_dtype:
        return tensor.to(target_dtype)

    return tensor


def extract_components(input_path, output_dir, components_to_extract=None, precision_map=None, keep_original_keys=False):
    """
    Extracts components from the model and saves them separately

    Args:
        input_path: Path to the .safetensors model file
        output_dir: Directory where extracted components will be saved
        components_to_extract: List of components to extract (None = all)
        precision_map: Dict mapping component names to target precision
        keep_original_keys: If True, don't transform keys (for debugging)
    """
    if components_to_extract is None:
        components_to_extract = ['unet', 'clip_l', 'clip_g', 'vae']

    if precision_map is None:
        precision_map = {}

    print(f"Loading model from: {input_path}")
    state_dict = load_file(input_path)

    # Dictionaries to store components
    components = {
        'unet': {},
        'clip_l': {},
        'clip_g': {},
        'vae': {},
        'unknown': {}
    }

    # Classify tensors by component
    print("Classifying components...")
    for key, tensor in state_dict.items():
        component = identify_component(key)

        # Transform key to standalone format
        if keep_original_keys:
            new_key = key
        else:
            new_key = transform_key(key, component)

        components[component][new_key] = tensor

    # Show classification summary
    print("\nClassification summary:")
    for comp_name, comp_dict in components.items():
        if comp_dict:
            total_params = sum(t.numel() for t in comp_dict.values())
            total_size_mb = sum(t.numel() * t.element_size() for t in comp_dict.values()) / (1024**2)
            will_extract = "✓" if comp_name in components_to_extract else "✗"
            print(f"  [{will_extract}] {comp_name}: {len(comp_dict)} tensors, {total_params:,} parameters, {total_size_mb:.2f} MB")

    # Show unknown tensors if any
    if components['unknown']:
        print(f"\n⚠️  Unknown tensors ({len(components['unknown'])}):")
        for key in list(components['unknown'].keys())[:10]:
            print(f"    - {key}")
        if len(components['unknown']) > 10:
            print(f"    ... and {len(components['unknown']) - 10} more")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(input_path).stem

    # Save each component
    print("\nSaving components...")
    for comp_name in ['unet', 'vae', 'clip_l', 'clip_g']:
        comp_dict = components[comp_name]

        if not comp_dict:
            print(f"  ⚠️  {comp_name}: No tensors found, skipping")
            continue

        # Skip if not in extraction list
        if comp_name not in components_to_extract:
            print(f"  Skipping {comp_name} (not selected for extraction)")
            continue

        # Determine target precision for this component
        target_precision = precision_map.get(comp_name, None)

        # Convert precision if requested
        if target_precision:
            print(f"  Converting {comp_name} to {target_precision}...")
            comp_dict_converted = {}
            for k, v in comp_dict.items():
                source_bits = get_tensor_bits(v.dtype)
                comp_dict_converted[k] = convert_tensor_precision(
                    v, target_precision, source_bits, comp_name, k
                )
            comp_dict = comp_dict_converted
            filename = f"{base_name}_{comp_name}.{target_precision}.safetensors"
        else:
            filename = f"{base_name}_{comp_name}.safetensors"

        output_path = os.path.join(output_dir, filename)

        print(f"  Saving {comp_name} -> {output_path}")
        save_file(comp_dict, output_path)

        # Show file size and sample keys
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"    ✓ Size: {size_mb:.2f} MB")

        # Show first few keys for verification
        sample_keys = list(comp_dict.keys())[:3]
        print(f"    Sample keys: {sample_keys}")

    print("\n✓ Extraction completed successfully")


def analyze_checkpoint(input_path):
    """
    Analyzes a checkpoint and shows its structure without extracting.
    Useful for debugging key naming conventions.
    """
    print(f"Analyzing checkpoint: {input_path}")
    state_dict = load_file(input_path)

    # Group keys by prefix
    prefixes = {}
    for key in state_dict.keys():
        # Get first two levels of the key
        parts = key.split('.')
        if len(parts) >= 2:
            prefix = '.'.join(parts[:2])
        else:
            prefix = parts[0]

        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(key)

    print(f"\nCheckpoint structure ({len(state_dict)} total tensors):\n")

    for prefix in sorted(prefixes.keys()):
        keys = prefixes[prefix]
        component = identify_component(keys[0])
        print(f"  {prefix}.* ({len(keys)} tensors) -> {component}")
        # Show a sample key
        print(f"    Example: {keys[0]}")

    return prefixes


def main():
    parser = argparse.ArgumentParser(
        description='Extracts components from SDXL/Pony models into separate safetensors files'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to the SDXL/Pony .safetensors model file'
    )
    parser.add_argument(
        '-d', '--output-dir',
        help='Directory where extracted components will be saved'
    )
    parser.add_argument(
        '-c', '--component',
        action='append',
        dest='components',
        choices=['unet', 'vae', 'clip_l', 'clip_g'],
        help='Component to extract (can be used multiple times). Default: all'
    )
    parser.add_argument(
        '--vae-precision',
        choices=['fp4', 'fp8', 'fp16', 'fp32'],
        help='Target precision for VAE component'
    )
    parser.add_argument(
        '--clip-l-precision',
        choices=['fp4', 'fp8', 'fp16', 'fp32'],
        help='Target precision for CLIP-L component'
    )
    parser.add_argument(
        '--clip-g-precision',
        choices=['fp4', 'fp8', 'fp16', 'fp32'],
        help='Target precision for CLIP-G component'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Only analyze the checkpoint structure without extracting'
    )
    parser.add_argument(
        '--keep-original-keys',
        action='store_true',
        help='Keep original checkpoint keys (for debugging, may break ComfyUI compatibility)'
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"✗ Error: File not found: {args.input}")
        return 1

    if not args.input.endswith('.safetensors'):
        print(f"✗ Error: File must have .safetensors extension")
        return 1

    # Analyze mode
    if args.analyze:
        analyze_checkpoint(args.input)
        return 0

    # Extraction requires output directory
    if not args.output_dir:
        print("✗ Error: --output-dir is required for extraction")
        return 1

    # Process components
    components_list = args.components if args.components else ['unet', 'vae', 'clip_l', 'clip_g']

    # Build precision map
    precision_map = {}
    if args.vae_precision:
        precision_map['vae'] = args.vae_precision
    if args.clip_l_precision:
        precision_map['clip_l'] = args.clip_l_precision
    if args.clip_g_precision:
        precision_map['clip_g'] = args.clip_g_precision

    # Execute extraction
    try:
        extract_components(
            args.input,
            args.output_dir,
            components_list,
            precision_map,
            keep_original_keys=args.keep_original_keys
        )
        return 0
    except Exception as e:
        print(f"✗ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
