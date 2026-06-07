#!/usr/bin/env python3
"""
quantize_lora.py — Quantize LoRA safetensors files to lower precision.

Usage:
    python quantize_lora.py input.safetensors [--output output.safetensors] [--dtype bf16]
    python quantize_lora.py input.safetensors --dtype bf16 --check

Arguments:
    input.safetensors       Path to the safetensors file to quantize.

Options:
    -o, --output PATH       Output path. Default: <input>_<dtype>.safetensors
    -d, --dtype DTYPE       Target dtype (default: bf16). Options: bf16, fp16
    -f, --force             Force conversion even if input is not FP32.
    --check                 Only check the dtype of the input file, don't convert.
    -v, --verbose           Print detailed tensor info per layer.
    --dry-run               Show what would be done without writing the output.

Examples:
    # Check what dtype a LoRA is
    python quantize_lora.py foo.safetensors --check

    # Convert FP32 LoRA to BF16
    python quantize_lora.py foo.safetensors -o foo_bf16.safetensors

    # Convert FP32 LoRA to FP16
    python quantize_lora.py foo.safetensors -d fp16

    # Dry-run: see size savings without writing
    python quantize_lora.py foo.safetensors --dry-run
"""

import argparse
import json
import os
import sys
import time

import torch
import safetensors
from safetensors.torch import load_file, save_file


# ── dtype helpers ──────────────────────────────────────────────────────────────

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "f32":  torch.float32,
    "f16":  torch.float16,
    "bf16": torch.bfloat16,
}

TYPENAME_MAP = {
    torch.float32:   "FP32",
    torch.bfloat16:  "BF16",
    torch.float16:   "FP16",
    torch.float64:   "FP64",
}

BYTES_PER_ELEMENT = {
    torch.float32:  4,
    torch.bfloat16: 2,
    torch.float16:  2,
    torch.float64:  8,
}


# ── inspection ─────────────────────────────────────────────────────────────────

def inspect_file(path: str, verbose: bool = False):
    """Load safetensors header and return (tensors_dict, per_tensor_dtypes)."""
    print(f"📂 File: {path}")
    size_bytes = os.path.getsize(path)
    print(f"   Size: {size_bytes / 1e6:.2f} MB  ({size_bytes:,} bytes)")

    # Read just the header first (faster for huge files)
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))

    tensor_names = sorted(k for k in header if k != "__metadata__")
    num_tensors = len(tensor_names)
    print(f"   Tensors: {num_tensors}")

    # Count dtypes from header
    dtype_counts = {}
    total_params = 0
    for name in tensor_names:
        info = header[name]
        dt = info["dtype"]
        shape = info["shape"]
        n_elems = 1
        for s in shape:
            n_elems *= s
        total_params += n_elems
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1

    print(f"   Parameters: {total_params:,}")
    print(f"   Dtype distribution (from header):")
    for dt, count in sorted(dtype_counts.items()):
        print(f"      {dt}: {count} tensors")

    # Infer the likely actual torch dtype
    dtypes_set = set(dtype_counts.keys())
    if dtypes_set == {"F32"} or dtypes_set == {"Float32"}:
        inferred = torch.float32
    elif dtypes_set == {"BF16"} or dtypes_set == {"BFloat16"} or dtypes_set == {"BFloat16"}:
        inferred = torch.bfloat16
    elif dtypes_set == {"F16"} or dtypes_set == {"Float16"}:
        inferred = torch.float16
    else:
        inferred = None

    # Also load a sample tensor to confirm
    if inferred is not None and not verbose:
        sample = tensor_names[0]
        tensors = {}
        tensors[sample] = torch.zeros(1)  # dummy to get load_file to work
        del tensors[sample]

    if verbose and num_tensors > 0:
        print(f"\n   Tensor list (first {min(20, num_tensors)}):")
        for i, name in enumerate(tensor_names[:20]):
            info = header[name]
            shape_str = "x".join(str(s) for s in info["shape"])
            print(f"      [{i:3d}] {name}: {info['dtype']} [{shape_str}]")
        if num_tensors > 20:
            print(f"      ... and {num_tensors - 20} more")

    # Metadata
    if "__metadata__" in header:
        meta = header["__metadata__"]
        print(f"\n   Metadata: {json.dumps(meta, indent=2)[:500]}")

    return inferred


# ── quantization ───────────────────────────────────────────────────────────────

def quantize(
    input_path: str,
    output_path: str,
    target_dtype: torch.dtype,
    force: bool = False,
    verbose: bool = False,
    dry_run: bool = False,
):
    """Load a safetensors file, convert all tensors to target_dtype, save."""

    if not os.path.exists(input_path):
        print(f"❌ Error: file not found: {input_path}")
        sys.exit(1)

    # 1. Inspect
    inferred = inspect_file(input_path, verbose=verbose)
    print()

    if inferred is None:
        print("⚠️  Could not determine a single dtype for this file.")
        print(f"   Proceeding with conversion to {TYPENAME_MAP[target_dtype]}...")
    elif inferred == target_dtype:
        print(f"✅ File is already {TYPENAME_MAP[target_dtype]}. Nothing to do.")
        sys.exit(0)
    elif inferred != torch.float32 and not force:
        print(
            f"⚠️  Input is {TYPENAME_MAP[inferred]}, not FP32.\n"
            f"   Converting from {TYPENAME_MAP[inferred]} to "
            f"{TYPENAME_MAP[target_dtype]} may cause quality loss.\n"
            f"   Use --force to override."
        )
        sys.exit(1)

    # 2. Load
    src_name = TYPENAME_MAP.get(inferred, "unknown")
    dst_name = TYPENAME_MAP[target_dtype]

    print(f"🔄 Loading tensors from '{input_path}'...")
    t0 = time.time()
    tensors = load_file(input_path, device="cpu")
    load_time = time.time() - t0
    print(f"   Loaded {len(tensors)} tensors in {load_time:.2f}s")

    # 3. Analyze
    in_size = sum(t.numel() * t.element_size() for t in tensors.values())
    out_elem_size = BYTES_PER_ELEMENT[target_dtype]
    out_size = sum(t.numel() * out_elem_size for t in tensors.values())
    savings = in_size - out_size

    print(f"\n   📊 Size analysis:")
    print(f"      Input:  {in_size / 1e6:.2f} MB  ({src_name})")
    print(f"      Output: {out_size / 1e6:.2f} MB  ({dst_name})")
    print(f"      Saving: {savings / 1e6:.2f} MB  ({savings/in_size*100:.1f}%)")

    if dry_run:
        print(f"\n⏸️  Dry-run: no file written.")
        print(f"   Would write to: {output_path}")
        return

    # 4. Convert
    print(f"\n🔄 Converting to {dst_name}...")
    t0 = time.time()
    converted = {}
    for name, tensor in tensors.items():
        converted[name] = tensor.to(target_dtype)
    convert_time = time.time() - t0
    print(f"   Converted {len(converted)} tensors in {convert_time:.2f}s")

    # 5. Save
    print(f"💾 Saving to '{output_path}'...")
    t0 = time.time()
    save_file(converted, output_path)
    save_time = time.time() - t0
    print(f"   Saved in {save_time:.2f}s")

    # 6. Verify
    out_size_disk = os.path.getsize(output_path)
    print(f"\n✅ Done! Output file: {output_path}")
    print(f"   Size on disk: {out_size_disk / 1e6:.2f} MB")
    print(f"   Total time: {load_time + convert_time + save_time:.2f}s")

    # Quick sanity check: load a tensor back
    verify = load_file(output_path, device="cpu")
    first_key = next(k for k in verify if k != "__metadata__")
    actual_dtype = verify[first_key].dtype
    print(f"   Verified: first tensor '{first_key}' is {TYPENAME_MAP.get(actual_dtype, str(actual_dtype))}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantize LoRA safetensors files to lower precision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Path to input .safetensors file")
    parser.add_argument("-o", "--output", default=None, help="Output path")
    parser.add_argument(
        "-d", "--dtype", default="bf16", choices=["bf16", "fp16"],
        help="Target precision (default: bf16)",
    )
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force conversion even if input is not FP32")
    parser.add_argument("--check", action="store_true",
                        help="Only check dtype, don't convert")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print detailed tensor info")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing")

    args = parser.parse_args()

    target_dtype = DTYPE_MAP[args.dtype]

    # ── check mode ──
    if args.check:
        inferred = inspect_file(args.input, verbose=args.verbose)
        print()
        if inferred is not None:
            print(f"✅ Result: {TYPENAME_MAP.get(inferred, str(inferred))}")
        else:
            print("⚠️  Mixed or unknown dtypes.")
        return

    # ── output path ──
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_{args.dtype}{ext}"

    if os.path.exists(args.output) and args.input != args.output:
        print(f"⚠️  Output file already exists: {args.output}")
        resp = input("   Overwrite? [y/N] ")
        if resp.lower() != "y":
            print("Aborted.")
            sys.exit(1)

    # ── quantize ──
    quantize(args.input, args.output, target_dtype,
             force=args.force, verbose=args.verbose, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
