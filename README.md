# SHARP-WebGPU

**Work in progress.** A port of Apple's [SHARP](https://github.com/apple/ml-sharp) (single-image 3D Gaussian Splat generation) from PyTorch to WebGPU compute shaders. No server, no WASM, no ONNX runtime — pure GPU compute shaders dispatched from JavaScript.

## Current status

The DINOv2 ViT-Large backbone runs end-to-end in WebGPU. The full SHARP pipeline is not yet implemented.

| Component | Status |
|-----------|--------|
| ViT-Large backbone (DINOv2, patch_size=16) | Working (~480ms on M4 Max) |
| Weight conversion (PyTorch → flat binary) | Working (702M params, 1.34 GB fp16) |
| SPN encoder (multi-scale pyramid + batched ViT) | Not started |
| Monodepth decoder (DPT + disparity head) | Not started |
| Gaussian decoder (DPT + prediction head) | Not started |
| Initializer + Composer (base Gaussians + deltas) | Not started |
| 3DGS output / visualization | Not started |

## What works now

Drop an image (or click a sample) and the app runs the full ViT-Large forward pass through 24 transformer blocks, producing intermediate features at layers [5, 11, 17, 23]. You'll see timing and token grid stats. No visual output yet — that requires the full pipeline.

## Architecture

SHARP predicts 3D Gaussian Splats from a single image in a single feedforward pass:

```
Image (1536x1536)
  → DINOv2 ViT-Large (patch_size=16, embed_dim=1024, depth=24)    ← working
  → SlidingPyramidNetwork (multi-scale patches → batched ViT)
  → MonodepthDPT (depth estimation)
  → GaussianDecoder DPT (feature extraction)
  → DirectPredictionHead (14-channel deltas: 3 mean + 3 scale + 4 quat + 3 color + 1 opacity)
  → MultiLayerInitializer (base Gaussians from image + depth)
  → GaussianComposer (base + deltas → final 3D Gaussian Splats)
```

14 WGSL compute shaders handle all operations: patch embedding, layer norm, multi-head self-attention, linear projection, GELU MLP, layer scale, conv2d, conv1x1, conv_transpose2d, bilinear upsample, pixel shuffle, group norm, and activations (ReLU, SiLU, sigmoid, softplus).

## Setup

```bash
git clone https://github.com/lyonsno/sharp-webgpu.git
cd sharp-webgpu
npm install
```

### Convert weights

Requires a Python environment with PyTorch. The converter downloads the default SHARP checkpoint from Apple's CDN (~2.7 GB PyTorch, converts to ~1.34 GB fp16 binary):

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

Click a sample image or drop your own. The backbone will run and show timing results.

## Browser requirements

- Chrome 113+ or Edge 113+ (WebGPU enabled)
- Firefox 141+ (WebGPU enabled via `dom.webgpu.enabled` in about:config)
- GPU with WebGPU support

## Kernel reuse

~90% of the ViT compute shaders are shared with [moge-webgpu](https://github.com/lyonsno/moge-webgpu) (MoGe-2 depth estimation in WebGPU). Both models use DINOv2 ViT-Large backbones — SHARP uses patch_size=16 (vs MoGe's 14), but the attention, layernorm, linear, and MLP shaders are identical.

## Tools

- `tools/convert_weights.py` — Convert SHARP PyTorch checkpoint to WebGPU binary format
- `tools/backbone_smoke.mjs` — Puppeteer-based automated backbone smoke test

## License

This port is provided under the same terms as Apple's original:

- **Code**: [Apple License](https://github.com/apple/ml-sharp/blob/main/LICENSE) (use, reproduce, modify, redistribute with attribution)
- **Model weights**: [Apple ML Research Model License](https://github.com/apple/ml-sharp/blob/main/MODEL_LICENSE) (non-commercial research only)

The WGSL compute shaders and JavaScript inference code in this repository are original work.

## References

- [SHARP: Sharp Monocular View Synthesis in Less Than a Second](https://apple.github.io/ml-sharp/) — Mescheder et al., Apple Research
- [apple/ml-sharp](https://github.com/apple/ml-sharp) — Original PyTorch implementation
- [arXiv:2512.10685](https://arxiv.org/abs/2512.10685) — Paper
