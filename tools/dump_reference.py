#!/usr/bin/env python3
"""
Dump SHARP PyTorch reference intermediate tensors for numerical parity comparison.

Runs the reference model in fp16 on MPS (matching WebGPU's fp16 weight precision)
and saves intermediate tensors at each pipeline stage as flat binary files.

Usage:
    python tools/dump_reference.py --image public/samples/sample_1.jpg --output public/reference_dumps/

The WebGPU comparator can then load these dumps and compare against its own
intermediate values at each stage.

Dump stages:
    1. input_normalized.bin     — [3, 1536, 1536] normalized image [-1, 1]
    2. spn_feature_*.bin        — 5 SPN output feature maps
    3. monodepth_disparity.bin  — [2, 1536, 1536] disparity
    4. feature_input.bin        — [5, 1536, 1536] Gaussian decoder input
    5. geom_deltas.bin          — [6, 768, 768] geometry prediction deltas
    6. tex_deltas.bin           — [22, 768, 768] texture prediction deltas
    7. gaussians_ndc.bin        — [1179648, 14] composed Gaussians in NDC
    8. gaussians_world.bin      — [1179648, 14] unprojected world-space Gaussians
    9. manifest.json            — metadata for all dumps
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description="Dump SHARP reference intermediates")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="public/reference_dumps/", help="Output directory")
    parser.add_argument("--dtype", default="fp16", choices=["fp32", "fp16"],
                        help="Model precision (default: fp16 to match WebGPU)")
    args = parser.parse_args()

    # Import SHARP
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ml-sharp" / "src"))
    from sharp.models import PredictorParams, create_predictor
    from sharp.models.predictor import RGBGaussianPredictor
    from sharp.utils import io as sharp_io
    from sharp.utils.gaussians import Gaussians3D, unproject_gaussians

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Precision: {args.dtype}")

    # Load model
    print("Loading SHARP model...")
    state_dict = torch.hub.load_state_dict_from_url(
        "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt",
        progress=True,
    )
    model = create_predictor(PredictorParams())
    model.load_state_dict(state_dict)
    model.eval()

    if args.dtype == "fp16":
        model = model.half()
    model = model.to(device)

    # Load and preprocess image (matching predict.py)
    print(f"Loading image: {args.image}")
    image_np, _, f_px = sharp_io.load_rgb(Path(args.image))
    height, width = image_np.shape[:2]
    print(f"  Original size: {width}x{height}, focal: {f_px:.1f}px")

    internal_shape = (1536, 1536)
    model_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    image_pt = torch.from_numpy(image_np.copy()).float().to(device).permute(2, 0, 1) / 255.0
    disparity_factor = torch.tensor([f_px / width], dtype=model_dtype, device=device)

    image_resized = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    ).to(model_dtype)

    manifest = {
        "image": args.image,
        "dtype": args.dtype,
        "device": device,
        "original_width": width,
        "original_height": height,
        "focal_px": float(f_px),
        "internal_shape": list(internal_shape),
        "disparity_factor": float(disparity_factor.item()),
        "dumps": {},
    }

    def save(name, tensor, description=""):
        """Save a tensor as flat binary and record in manifest."""
        arr = tensor.detach().float().cpu().numpy()
        path = output_dir / f"{name}.bin"
        arr.tofile(path)
        manifest["dumps"][name] = {
            "file": f"{name}.bin",
            "shape": list(arr.shape),
            "dtype": "float32",
            "size_bytes": arr.nbytes,
            "description": description,
        }
        print(f"  Saved {name}: shape={list(arr.shape)}, {arr.nbytes / 1024:.1f} KB")

    # --- Stage 1: Normalized input ---
    print("\nStage 1: Input")
    save("input_normalized", image_resized[0], "Normalized input image [-1, 1] CHW")

    # --- Stage 2: Run model with hooks to capture intermediates ---
    print("\nStage 2: Running model with intermediate capture...")

    # Hook into the monodepth model to capture SPN features and disparity
    spn_features = []
    monodepth_disparity = [None]
    feature_input_capture = [None]
    monodepth_output_capture = [None]

    # We need to trace through the model manually to capture intermediates
    with torch.no_grad():
        # Step 1: Monodepth forward (captures SPN features internally)
        monodepth_output = model.monodepth_model(image_resized)
        monodepth_disp = monodepth_output.disparity

        save("monodepth_disparity", monodepth_disp[0],
             f"Raw disparity [{monodepth_disp.shape[1]} channels, {monodepth_disp.shape[2]}x{monodepth_disp.shape[3]}]")

        # Step 2: Depth computation
        disp_factor = disparity_factor[:, None, None, None]
        monodepth = disp_factor / monodepth_disp.clamp(min=1e-4, max=1e4)

        # Step 3: Depth alignment (no-op at inference)
        monodepth, _ = model.depth_alignment(monodepth, None, monodepth_output.decoder_features)

        save("depth_after_alignment", monodepth[0], "Depth after alignment (no-op at inference)")

        # Step 4: Initializer
        init_output = model.init_model(image_resized, monodepth)

        save("feature_input", init_output.feature_input[0],
             f"Feature input to Gaussian decoder [{init_output.feature_input.shape[1]} channels]")

        if init_output.global_scale is not None:
            save("global_scale", init_output.global_scale,
                 f"Global scale from depth normalization: {init_output.global_scale.item():.6f}")

        # Save base Gaussian values
        bv = init_output.gaussian_base_values
        save("base_mean_x_ndc", bv.mean_x_ndc[0], "Base mean X in NDC")
        save("base_mean_y_ndc", bv.mean_y_ndc[0], "Base mean Y in NDC")
        save("base_mean_inv_z", bv.mean_inverse_z_ndc[0], "Base mean inverse Z (disparity)")
        save("base_scales", bv.scales[0], "Base Gaussian scales")
        save("base_colors", bv.colors[0], "Base colors (avg-pooled image)")

        # Step 5: Gaussian decoder (feature_model)
        image_features = model.feature_model(
            init_output.feature_input, encodings=monodepth_output.output_features
        )

        save("texture_features", image_features.texture_features[0],
             f"Texture features [{image_features.texture_features.shape[1]}ch]")
        save("geometry_features", image_features.geometry_features[0],
             f"Geometry features [{image_features.geometry_features.shape[1]}ch]")

        # Step 6: Prediction head
        delta_values = model.prediction_head(image_features)

        # delta_values shape: [B, 14, num_layers, H, W]
        save("delta_values", delta_values[0], f"Raw delta values {list(delta_values.shape[1:])}")

        # Split geometry and texture deltas (matching DirectPredictionHead.forward)
        geom_deltas = delta_values[:, :3]  # [B, 3, num_layers, H, W]
        tex_deltas = delta_values[:, 3:]   # [B, 11, num_layers, H, W]

        # Flatten to [B, C*layers, H, W] to match WebGPU output format
        B, C_g, L, H, W = geom_deltas.shape
        geom_flat = geom_deltas.reshape(B, C_g * L, H, W)
        tex_flat = tex_deltas.reshape(B, (14 - 3) * L, H, W)

        save("geom_deltas", geom_flat[0], f"Geometry deltas [{geom_flat.shape[1]}ch, {H}x{W}]")
        save("tex_deltas", tex_flat[0], f"Texture deltas [{tex_flat.shape[1]}ch, {H}x{W}]")

        # Step 7: Composer
        gaussians_ndc = model.gaussian_composer(
            delta=delta_values,
            base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale,
        )

        save("gaussians_ndc_means", gaussians_ndc.mean_vectors[0], "NDC Gaussian means [N, 3]")
        save("gaussians_ndc_scales", gaussians_ndc.singular_values[0], "NDC Gaussian scales [N, 3]")
        save("gaussians_ndc_quats", gaussians_ndc.quaternions[0], "NDC Gaussian quaternions [N, 4]")
        save("gaussians_ndc_colors", gaussians_ndc.colors[0], "NDC Gaussian colors [N, 3]")
        save("gaussians_ndc_opacities", gaussians_ndc.opacities[0], "NDC Gaussian opacities [N]")

        # Step 8: Unprojection (matching predict.py)
        intrinsics = torch.tensor([
            [f_px, 0, width / 2, 0],
            [0, f_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32, device=device)
        intrinsics_resized = intrinsics.clone()
        intrinsics_resized[0] *= internal_shape[0] / width
        intrinsics_resized[1] *= internal_shape[1] / height

        gaussians_world = unproject_gaussians(
            gaussians_ndc, torch.eye(4, device=device), intrinsics_resized, internal_shape
        )

        save("gaussians_world_means", gaussians_world.mean_vectors[0], "World Gaussian means [N, 3]")
        save("gaussians_world_scales", gaussians_world.singular_values[0], "World Gaussian scales [N, 3]")
        save("gaussians_world_quats", gaussians_world.quaternions[0], "World Gaussian quaternions [N, 4]")
        save("gaussians_world_colors", gaussians_world.colors[0], "World Gaussian colors [N, 3]")
        save("gaussians_world_opacities", gaussians_world.opacities[0], "World Gaussian opacities [N]")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")

    # Print summary
    total_size = sum(d["size_bytes"] for d in manifest["dumps"].values())
    print(f"\nTotal: {len(manifest['dumps'])} dumps, {total_size / 1024 / 1024:.1f} MB")
    print(f"Gaussian count: {gaussians_ndc.mean_vectors.shape[1]:,}")


if __name__ == "__main__":
    main()
