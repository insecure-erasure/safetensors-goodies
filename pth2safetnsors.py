#!/usr/bin/env python3
"""
Convert PyTorch checkpoint files (.pth, .pt) to SafeTensors format while preserving metadata.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
from safetensors.torch import save_file, safe_open


class CheckpointConverter:
    """Handles conversion of PyTorch checkpoints to SafeTensors format."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def log(self, message: str, level: str = "INFO"):
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{level}] {message}")
    
    def load_checkpoint(self, input_path: Path) -> Dict[str, Any]:
        """Load PyTorch checkpoint file."""
        self.log(f"Loading checkpoint from {input_path}")
        
        try:
            # Try loading with weights_only=True first (safer)
            checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)
            self.log("Loaded checkpoint with weights_only=True")
        except Exception as e:
            self.log(f"Failed to load with weights_only=True: {e}", "WARNING")
            self.log("Attempting to load with weights_only=False", "WARNING")
            try:
                checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
                self.log("Loaded checkpoint with weights_only=False")
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        return checkpoint
    
    def extract_state_dict(self, checkpoint: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Extract state_dict and metadata from checkpoint.
        
        Returns:
            Tuple of (state_dict, metadata)
        """
        state_dict = None
        metadata = {}
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Common keys for state_dict
            state_dict_keys = ["state_dict", "model_state_dict", "model"]
            
            for key in state_dict_keys:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    self.log(f"Found state_dict under key '{key}'")
                    break
            
            # If no state_dict found, assume the checkpoint itself is the state_dict
            if state_dict is None:
                # Check if this looks like a state_dict (all values are tensors)
                if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    state_dict = checkpoint
                    self.log("Checkpoint appears to be a direct state_dict")
                else:
                    # Extract tensors and non-tensors separately
                    state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
                    metadata_raw = {k: v for k, v in checkpoint.items() if not isinstance(v, torch.Tensor)}
                    
                    if state_dict:
                        self.log(f"Extracted {len(state_dict)} tensors from checkpoint")
                        # Store non-tensor items as metadata
                        for k, v in metadata_raw.items():
                            metadata[k] = self._serialize_value(v)
                    else:
                        raise ValueError("Could not find state_dict in checkpoint")
            else:
                # Extract metadata from other keys
                for k, v in checkpoint.items():
                    if k not in state_dict_keys and not isinstance(v, torch.Tensor):
                        metadata[k] = self._serialize_value(v)
        else:
            raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")
        
        self.log(f"Extracted {len(state_dict)} tensors and {len(metadata)} metadata entries")
        
        return state_dict, metadata
    
    def _serialize_value(self, value: Any) -> str:
        """Convert a value to string for SafeTensors metadata."""
        if isinstance(value, (int, float, bool)):
            return str(value)
        elif isinstance(value, str):
            return value
        elif isinstance(value, (list, tuple, dict)):
            return json.dumps(value)
        elif isinstance(value, torch.Tensor):
            # For small tensors, serialize as list
            if value.numel() <= 100:
                return json.dumps(value.tolist())
            else:
                return f"<Tensor: shape={list(value.shape)}, dtype={value.dtype}>"
        else:
            return str(value)
    
    def prepare_tensors(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare tensors for SafeTensors (make contiguous, handle shared memory)."""
        self.log("Preparing tensors for SafeTensors format")
        
        prepared = {}
        cloned_count = 0
        
        for key, tensor in state_dict.items():
            # Make tensor contiguous
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            
            # Check for shared storage and clone if necessary
            # This prevents "tensors share memory" errors
            needs_clone = False
            for existing_key, existing_tensor in prepared.items():
                if tensor.data_ptr() == existing_tensor.data_ptr():
                    needs_clone = True
                    self.log(f"Tensor '{key}' shares memory with '{existing_key}', cloning", "WARNING")
                    break
            
            if needs_clone:
                tensor = tensor.clone()
                cloned_count += 1
            
            prepared[key] = tensor
        
        if cloned_count > 0:
            self.log(f"Cloned {cloned_count} tensors to resolve shared memory")
        
        return prepared
    
    def add_optimizer_metadata(self, checkpoint: Dict[str, Any], metadata: Dict[str, str], 
                               preserve_optimizer: bool) -> Dict[str, str]:
        """Add optimizer state information to metadata if requested."""
        if not preserve_optimizer:
            return metadata
        
        optimizer_keys = ["optimizer_state_dict", "optimizer"]
        for key in optimizer_keys:
            if key in checkpoint:
                self.log(f"Found optimizer state under '{key}'")
                # Serialize optimizer state (can be large)
                opt_state = checkpoint[key]
                if isinstance(opt_state, dict):
                    # Store summary information
                    metadata[f"{key}_keys"] = json.dumps(list(opt_state.keys()))
                    
                    # Try to serialize compact representation
                    try:
                        metadata[key] = json.dumps(opt_state, default=str)
                        self.log(f"Serialized optimizer state ({len(metadata[key])} chars)")
                    except Exception as e:
                        self.log(f"Could not fully serialize optimizer state: {e}", "WARNING")
                        metadata[f"{key}_summary"] = str(opt_state)[:1000] + "..."
                break
        
        return metadata
    
    def save_safetensors(self, state_dict: Dict[str, torch.Tensor], metadata: Dict[str, str], 
                        output_path: Path):
        """Save state_dict and metadata to SafeTensors file."""
        self.log(f"Saving to {output_path}")
        
        try:
            save_file(state_dict, str(output_path), metadata=metadata)
            self.log(f"Successfully saved {len(state_dict)} tensors with {len(metadata)} metadata entries")
        except Exception as e:
            raise RuntimeError(f"Failed to save SafeTensors file: {e}")
    
    def verify_conversion(self, input_path: Path, output_path: Path):
        """Verify that conversion was successful by comparing tensors and metadata."""
        self.log("Verifying conversion...")
        
        # Load original
        original = self.load_checkpoint(input_path)
        original_state_dict, original_metadata = self.extract_state_dict(original)
        
        # Load converted
        converted_state_dict = {}
        converted_metadata = {}
        
        with safe_open(str(output_path), framework="pt", device="cpu") as f:
            converted_metadata = f.metadata() or {}
            for key in f.keys():
                converted_state_dict[key] = f.get_tensor(key)
        
        # Compare tensors
        if set(original_state_dict.keys()) != set(converted_state_dict.keys()):
            missing = set(original_state_dict.keys()) - set(converted_state_dict.keys())
            extra = set(converted_state_dict.keys()) - set(original_state_dict.keys())
            
            if missing:
                print(f"[ERROR] Missing tensors in converted file: {missing}")
            if extra:
                print(f"[ERROR] Extra tensors in converted file: {extra}")
            return False
        
        # Compare shapes and dtypes
        all_match = True
        for key in original_state_dict.keys():
            orig_tensor = original_state_dict[key]
            conv_tensor = converted_state_dict[key]
            
            if orig_tensor.shape != conv_tensor.shape:
                print(f"[ERROR] Shape mismatch for '{key}': {orig_tensor.shape} vs {conv_tensor.shape}")
                all_match = False
            
            if orig_tensor.dtype != conv_tensor.dtype:
                print(f"[ERROR] Dtype mismatch for '{key}': {orig_tensor.dtype} vs {conv_tensor.dtype}")
                all_match = False
            
            # Compare values (sample check for large tensors)
            if not torch.allclose(orig_tensor, conv_tensor, rtol=1e-5, atol=1e-8):
                print(f"[ERROR] Value mismatch for '{key}'")
                all_match = False
        
        if all_match:
            print("[OK] All tensors match")
        
        # Compare metadata
        print(f"\nMetadata comparison:")
        print(f"  Original: {len(original_metadata)} entries")
        print(f"  Converted: {len(converted_metadata)} entries")
        
        if converted_metadata:
            print(f"\nConverted metadata keys: {list(converted_metadata.keys())}")
        
        return all_match
    
    def list_metadata(self, input_path: Path):
        """List all metadata found in the checkpoint without converting."""
        checkpoint = self.load_checkpoint(input_path)
        _, metadata = self.extract_state_dict(checkpoint)
        
        print(f"\nCheckpoint: {input_path}")
        print(f"Found {len(metadata)} metadata entries:\n")
        
        for key, value in metadata.items():
            # Truncate long values
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            print(f"  {key}: {value_str}")
        
        # Also show checkpoint structure
        print(f"\nCheckpoint structure:")
        if isinstance(checkpoint, dict):
            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor{list(value.shape)}")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with {len(value)} entries")
                else:
                    print(f"  {key}: {type(value).__name__}")
    
    def convert(self, input_path: Path, output_path: Optional[Path] = None, 
                preserve_optimizer: bool = False) -> Path:
        """
        Main conversion method.
        
        Args:
            input_path: Path to input .pth/.pt file
            output_path: Path to output .safetensors file (optional)
            preserve_optimizer: Whether to preserve optimizer state in metadata
        
        Returns:
            Path to output file
        """
        # Determine output path
        if output_path is None:
            output_path = input_path.with_suffix(".safetensors")
        
        # Load checkpoint
        checkpoint = self.load_checkpoint(input_path)
        
        # Extract state_dict and metadata
        state_dict, metadata = self.extract_state_dict(checkpoint)
        
        # Add optimizer metadata if requested
        metadata = self.add_optimizer_metadata(checkpoint, metadata, preserve_optimizer)
        
        # Prepare tensors
        state_dict = self.prepare_tensors(state_dict)
        
        # Save to SafeTensors
        self.save_safetensors(state_dict, metadata, output_path)
        
        # Print summary
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"\nConversion completed successfully!")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Size:   {file_size:.2f} MB")
        print(f"  Tensors: {len(state_dict)}")
        print(f"  Metadata entries: {len(metadata)}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoint files (.pth, .pt) to SafeTensors format while preserving metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Basic conversion
  python convert_to_safetensors.py model.pth
  
  # Convert with custom output path
  python convert_to_safetensors.py model.pth -o converted/model.safetensors
  
  # Convert and verify
  python convert_to_safetensors.py checkpoint.pth --verify --verbose
  
  # List metadata without converting
  python convert_to_safetensors.py checkpoint.pth --list-metadata
        """
    )
    
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input .pth or .pt file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output .safetensors file path (default: same name as input with .safetensors extension)"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the conversion by comparing tensor shapes and metadata"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output showing detailed conversion process"
    )
    
    parser.add_argument(
        "--preserve-optimizer",
        action="store_true",
        help="Attempt to preserve optimizer state as metadata (warning: may significantly increase file size)"
    )
    
    parser.add_argument(
        "--list-metadata",
        action="store_true",
        help="List all metadata found in the checkpoint without converting"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file '{args.input}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not args.input.suffix in [".pth", ".pt"]:
        print(f"Warning: Input file '{args.input}' does not have .pth or .pt extension", file=sys.stderr)
    
    # Create converter
    converter = CheckpointConverter(verbose=args.verbose)
    
    try:
        # List metadata mode
        if args.list_metadata:
            converter.list_metadata(args.input)
            return
        
        # Convert
        output_path = converter.convert(
            args.input,
            args.output,
            preserve_optimizer=args.preserve_optimizer
        )
        
        # Verify if requested
        if args.verify:
            print("\n" + "="*60)
            success = converter.verify_conversion(args.input, output_path)
            print("="*60)
            
            if not success:
                print("\n[WARNING] Verification found differences!")
                sys.exit(1)
            else:
                print("\n[OK] Verification passed!")
    
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
