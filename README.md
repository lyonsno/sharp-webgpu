# SHARP-WebGPU

A complete port of Apple's [SHARP](https://github.com/apple/ml-sharp) (single-image 3D Gaussian Splat generation) from PyTorch to WebGPU compute shaders. No server, no WASM, no ONNX runtime — pure GPU compute shaders dispatched from JavaScript.

## What it does

Drop an image in the browser and get **1.18 million 3D Gaussian Splats** in ~25 seconds on Apple M4 Max. Download as a standard .ply file compatible with any 3DGS viewer.

| Component | Status |
|-----------|--------|
| ViT-Large backbone (DINOv2, patch_size=16) | Working (~480ms per patch) |
| SPN encoder (35 batched ViT passes) | Working (~18s) |
| Monodepth decoder (depth map) | Working |
| Gaussian decoder (texture + geometry features) | Working |
| Prediction head (14-channel deltas) | Working |
| Initializer + Composer (base + deltas → 3DGS) | Working |
| PLY export | Working (1.18M splats, ~66 MB) |
| Weight conversion (PyTorch → flat binary) | Working (702M params, 1.25 GB fp16) |

## Architecture

SHARP predicts 3D Gaussian Splats from a single image in a single feedforward pass:

```
Image (1536x1536)
  → SlidingPyramidNetwork
      → 3-level pyramid (1536 → 768 → 384)
      → 35 overlapping patches (5x5 + 3x3 + 1x1) through DINOv2 ViT-Large
      → overlap merge + upsample fusion → 5 multi-resolution feature maps
  → MonodepthDPT (MultiresConvDecoder + disparity head → depth map)
  → GaussianDecoder DPT (MultiresConvDecoder + SkipConvBackbone → features)
  → DirectPredictionHead (14-channel deltas per Gaussian layer)
  → MultiLayerInitializer (base Gaussians from image + depth)
  → GaussianComposer (base + deltas → 1.18M 3D Gaussian Splats)
  → PLY export (standard 3DGS format)
```

14 WGSL compute shaders handle all operations: patch embedding, layer norm, multi-head self-attention, linear projection, GELU MLP, layer scale, conv2d, conv1x1, conv_transpose2d, bilinear upsample, pixel shuffle, group norm, and activations (ReLU, SiLU, sigmoid, softplus).

## Setup

```bash
git clone https://github.com/lyonsno/sharp-webgpu.git
cd sharp-webgpu
npm install
```

### Convert weights

Requires a Python environment with PyTorch. The converter downloads the default SHARP checkpoint from Apple's CDN (~2.7 GB PyTorch, converts to ~1.25 GB fp16 binary):

```bash
# If you have ml-sharp's venv:
path/to/python tools/convert_weights.py --output public/weights.bin --dtype fp16

# Or install torch separately:
pip install torch
python tools/convert_weights.py --output public/weights.bin --dtype fp16
```

To just inspect the tensor list without downloading:
```bash
python tools/convert_weights.py --list-only
```

### Run

```bash
npm run dev
# Open http://localhost:5175/
```

Check "Run full SPN", click a sample image or drop your own. After ~25 seconds you'll see a depth map and a download link for the .ply file containing 1.18M Gaussian Splats.

The .ply file loads in any standard 3DGS viewer (SuperSplat, Kaminos, etc.).

## Browser requirements

- Chrome 113+ or Edge 113+ (WebGPU enabled)
- Firefox 141+ (WebGPU enabled via `dom.webgpu.enabled` in about:config)
- GPU with WebGPU support

## Performance

On Apple M4 Max (128 GB):

| Stage | Time |
|-------|------|
| Weight loading (first run) | ~5s |
| SPN encoder (35 ViT passes) | ~18s |
| Monodepth decoder | ~2s |
| Gaussian decoder + compose | ~3s |
| **Total** | **~25s** |

## Kernel reuse

~90% of the ViT compute shaders are shared with [moge-webgpu](https://github.com/lyonsno/moge-webgpu) (MoGe-2 depth estimation in WebGPU). Both models use DINOv2 ViT-Large backbones — SHARP uses patch_size=16 (vs MoGe's 14), but the attention, layernorm, linear, and MLP shaders are identical.

## Tools

- `tools/convert_weights.py` — Convert SHARP PyTorch checkpoint to WebGPU binary format
- `tools/witness.mjs` — Automated inference witness (headless Chrome + WebGPU)
- `tools/backbone_smoke.mjs` — Backbone-only smoke test
- `tools/demo_smoke.mjs` — Demo UI smoke test

## License

This port is provided under the same terms as Apple's original:

- **Code**: [Apple License](https://github.com/apple/ml-sharp/blob/main/LICENSE) (use, reproduce, modify, redistribute with attribution)
- **Model weights**: [Apple ML Research Model License](https://github.com/apple/ml-sharp/blob/main/MODEL_LICENSE) (non-commercial research only)

The WGSL compute shaders and JavaScript inference code in this repository are original work.

## References

- [SHARP: Sharp Monocular View Synthesis in Less Than a Second](https://apple.github.io/ml-sharp/) — Mescheder et al., Apple Research
- [apple/ml-sharp](https://github.com/apple/ml-sharp) — Original PyTorch implementation
- [arXiv:2512.10685](https://arxiv.org/abs/2512.10685) — Paper
