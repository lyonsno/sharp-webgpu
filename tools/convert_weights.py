#!/usr/bin/env python3
"""
Convert Apple SHARP PyTorch checkpoint to flat binary format for WebGPU.

Usage:
    # Download and convert the default model:
    python tools/convert_weights.py --output public/weights.bin

    # Use a local checkpoint:
    python tools/convert_weights.py --checkpoint path/to/sharp.pt --output public/weights.bin

    # List tensors without converting:
    python tools/convert_weights.py --list-only

Output format (same as moge-webgpu but with SHRP magic):
    Header:
        4 bytes: magic "SHRP"
        4 bytes: version (1)
        4 bytes: num_tensors
        4 bytes: header_size

    Tensor table (repeated num_tensors times):
        64 bytes: name (null-padded ASCII)
        4 bytes: dtype (0=fp32, 1=fp16)
        4 bytes: ndim
        16 bytes: shape (4 x u32, padded)
        4 bytes: offset into weight data section
        4 bytes: size in bytes

    Weight data:
        Packed contiguous tensors in row-major order
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import torch
import numpy as np


DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

MAGIC = b"SHRP"
VERSION = 1
MAX_NAME_LEN = 128
MAX_DIMS = 4


def load_checkpoint(checkpoint_path: str | None) -> dict:
    """Load SHARP checkpoint from local path or download default."""
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading local checkpoint: {checkpoint_path}")
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    print(f"Downloading default model from {DEFAULT_MODEL_URL}")
    return torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)


def build_state_dict(checkpoint) -> dict:
    """Build a flat state dict from the SHARP checkpoint.

    SHARP's checkpoint is a direct state_dict (not wrapped in a dict).
    We instantiate the model to get the full state_dict with proper keys.
    """
    # Import SHARP's model factory
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ml-sharp" / "src"))
    from sharp.models import PredictorParams, create_predictor

    model = create_predictor(PredictorParams())
    model.load_state_dict(checkpoint)
    return model.state_dict()


# Linear weight names that need transposition from [out, in] to [in, out]
# for the WebGPU shader convention (row-major, multiply input @ weight)
LINEAR_WEIGHT_SUFFIXES = (
    'attn.qkv.weight', 'attn.proj.weight',
    'mlp.fc1.weight', 'mlp.fc2.weight',
    'mlp.w12.weight', 'mlp.w3.weight',  # SwiGLU variant if present
)


def should_transpose(name: str, arr: np.ndarray) -> bool:
    """Decide if a weight tensor needs transposition for WebGPU."""
    if arr.ndim != 2:
        return False
    return any(name.endswith(s) for s in LINEAR_WEIGHT_SUFFIXES)


def convert(state_dict: dict, output_path: str, dtype: str = "fp16"):
    """Convert state dict to flat binary."""
    tensor_entries = []
    weight_data = bytearray()

    dtype_code = 0 if dtype == "fp32" else 1
    np_dtype = np.float32 if dtype == "fp32" else np.float16

    # Filter out non-tensor entries
    skip_prefixes = ('init_model.', 'gaussian_composer.')  # no learned weights

    for name, tensor in sorted(state_dict.items()):
        if any(name.startswith(p) for p in skip_prefixes):
            continue

        arr = tensor.detach().float().numpy()

        # Transpose 2D linear weights for WebGPU
        if should_transpose(name, arr):
            arr = arr.T.copy()

        arr = arr.astype(np_dtype)
        data = arr.tobytes()

        shape = list(arr.shape)
        if len(shape) > MAX_DIMS:
            shape = [int(np.prod(shape[:-3]))] + list(shape[-3:])
            arr = arr.reshape(shape)
            data = arr.tobytes()

        offset = len(weight_data)
        size = len(data)
        weight_data.extend(data)

        # Pad to 16-byte alignment
        pad = (16 - (len(weight_data) % 16)) % 16
        weight_data.extend(b"\x00" * pad)

        tensor_entries.append({
            "name": name,
            "dtype": dtype_code,
            "shape": shape,
            "offset": offset,
            "size": size,
        })

    # Build header
    num_tensors = len(tensor_entries)
    ENTRY_SIZE = 160  # 128 (name) + 4 (dtype) + 4 (ndim) + 16 (shape) + 4 (offset) + 4 (size)
    header_size = 16 + num_tensors * ENTRY_SIZE

    # Adjust offsets to be absolute
    for entry in tensor_entries:
        entry["offset"] += header_size

    # Write binary
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", num_tensors))
        f.write(struct.pack("<I", header_size))

        for entry in tensor_entries:
            name_bytes = entry["name"].encode("ascii")[:MAX_NAME_LEN]
            f.write(name_bytes.ljust(MAX_NAME_LEN, b"\x00"))
            f.write(struct.pack("<I", entry["dtype"]))
            ndim = len(entry["shape"])
            f.write(struct.pack("<I", ndim))
            shape_padded = entry["shape"] + [0] * (MAX_DIMS - ndim)
            for s in shape_padded:
                f.write(struct.pack("<I", s))
            f.write(struct.pack("<I", entry["offset"]))
            f.write(struct.pack("<I", entry["size"]))

        f.write(weight_data)

    total_mb = len(weight_data) / (1024 * 1024)
    print(f"Weights written to {output_path}")
    print(f"  Tensors: {num_tensors}")
    print(f"  Size: {total_mb:.1f} MB ({dtype})")
    print(f"  Header: {header_size} bytes")

    # Write JSON sidecar with tensor manifest
    manifest = {
        "magic": "SHRP",
        "version": VERSION,
        "dtype": dtype,
        "num_tensors": num_tensors,
        "total_size_mb": round(total_mb, 1),
        "tensors": {
            entry["name"]: {
                "shape": entry["shape"],
                "offset": entry["offset"],
                "size": entry["size"],
            }
            for entry in tensor_entries
        },
    }
    manifest_path = Path(output_path).with_suffix(".json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert SHARP weights for WebGPU")
    parser.add_argument("--checkpoint", default=None,
                        help="Local checkpoint .pt path (downloads default if omitted)")
    parser.add_argument("--output", default="public/weights.bin",
                        help="Output binary file path")
    parser.add_argument("--dtype", default="fp16", choices=["fp32", "fp16"],
                        help="Weight data type (default: fp16)")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list tensor names and shapes")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)

    # SHARP checkpoint is a direct state_dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if args.list_only:
        print(f"\nState dict ({len(state_dict)} tensors):")
        total_params = 0
        for name, tensor in sorted(state_dict.items()):
            shape = list(tensor.shape)
            params = tensor.numel()
            total_params += params
            shape_str = "x".join(str(s) for s in shape)
            print(f"  {name:64s} {shape_str:>20s}  {params:>12,d}")
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Total size (fp32): {total_params * 4 / 1024 / 1024:.1f} MB")
        print(f"Total size (fp16): {total_params * 2 / 1024 / 1024:.1f} MB")
        return

    convert(state_dict, args.output, args.dtype)


if __name__ == "__main__":
    main()
