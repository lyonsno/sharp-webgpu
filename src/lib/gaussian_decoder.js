/**
 * gaussian_decoder.js — Gaussian prediction pipeline for SHARP-WebGPU.
 *
 * Takes SPN features + monodepth disparity → Gaussian Splat parameters.
 *
 * Pipeline:
 *   1. Initializer: image + depth → base Gaussians + feature_input [5, H, W]
 *   2. GaussianDecoder: MultiresConvDecoder(SPN features) + SkipConvBackbone(feature_input) → fused features
 *   3. Texture/Geometry heads: features → [32, H/2, W/2] each
 *   4. Prediction head: geometry→3ch deltas, texture→11ch deltas → [14, 2, H/2, W/2]
 *   5. Composer: base Gaussians + deltas → final 3D Gaussian Splats
 */

import { createStorageBuffer, createEmptyBuffer, readBuffer } from './gpu.js';
import {
  dispatchConv2d,
  dispatchConv1x1,
  dispatchConvTranspose2d,
  dispatchActivation,
  dispatchGroupNorm,
} from './shader_ops.js';

const breathe = () => new Promise(r => setTimeout(r, 0));

// --- Shared decoder dispatch (used by both monodepth and gaussian decoders) ---

function dispatchResidualBlock(device, inputBuf, C, H, W, prefix, raw) {
  const count = C * H * W;
  const enc = device.createCommandEncoder();

  const relu1 = dispatchActivation(device, enc, inputBuf, null, count, 0);
  const conv1 = dispatchConv2d(device, enc, relu1,
    raw.get(`${prefix}.residual.1.weight`), raw.get(`${prefix}.residual.1.bias`),
    { inC: C, inH: H, inW: W, outC: C, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });
  const relu2 = dispatchActivation(device, enc, conv1.buffer, null, count, 0);
  const conv2 = dispatchConv2d(device, enc, relu2,
    raw.get(`${prefix}.residual.3.weight`), raw.get(`${prefix}.residual.3.bias`),
    { inC: C, inH: H, inW: W, outC: C, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });
  const sumBuf = dispatchActivation(device, enc, inputBuf, conv2.buffer, count, 2);

  device.queue.submit([enc.finish()]);
  return sumBuf;
}

function dispatchFusionBlock(device, x0Buf, x1Buf, C, H, W, prefix, raw, hasDeconv) {
  let currentBuf = x0Buf;
  let currentH = H, currentW = W;

  if (x1Buf) {
    const res1Buf = dispatchResidualBlock(device, x1Buf, C, currentH, currentW, `${prefix}.resnet1`, raw);
    const enc = device.createCommandEncoder();
    currentBuf = dispatchActivation(device, enc, currentBuf, res1Buf, C * currentH * currentW, 2);
    device.queue.submit([enc.finish()]);
  }

  currentBuf = dispatchResidualBlock(device, currentBuf, C, currentH, currentW, `${prefix}.resnet2`, raw);

  const enc = device.createCommandEncoder();
  if (hasDeconv) {
    const deconvResult = dispatchConvTranspose2d(device, enc, currentBuf,
      raw.get(`${prefix}.deconv.weight`), null,
      { inC: C, inH: currentH, inW: currentW, outC: C, stride: 2 });
    currentBuf = deconvResult.buffer;
    currentH *= 2;
    currentW *= 2;
  }

  const outResult = dispatchConv1x1(device, enc, currentBuf,
    raw.get(`${prefix}.out_conv.weight`), raw.get(`${prefix}.out_conv.bias`),
    { inC: C, outC: C, H: currentH, W: currentW });
  device.queue.submit([enc.finish()]);

  return { buffer: outResult.buffer, H: currentH, W: currentW };
}

/**
 * Dispatch a GroupNorm residual block (used in texture/geometry heads).
 *
 * Structure: GN(inC)→ReLU→Conv3x3(inC→hidden)→GN(hidden)→ReLU→Conv3x3(hidden→outC) + skip
 * Weight indices: .0=GN, .1=ReLU(no w), .2=Conv, .3=GN, .4=ReLU(no w), .5=Conv
 */
function dispatchGroupNormResidualBlock(device, inputBuf, inC, outC, hiddenC, H, W, prefix, raw, numGroups) {
  const enc = device.createCommandEncoder();

  // GroupNorm(inC)
  const gn1 = dispatchGroupNorm(device, enc, inputBuf,
    raw.get(`${prefix}.residual.0.weight`), raw.get(`${prefix}.residual.0.bias`),
    { C: inC, H, W, numGroups, eps: 1e-5 });

  // ReLU
  const relu1 = dispatchActivation(device, enc, gn1, null, inC * H * W, 0);

  // Conv3x3(inC → hiddenC)
  const conv1 = dispatchConv2d(device, enc, relu1,
    raw.get(`${prefix}.residual.2.weight`), raw.get(`${prefix}.residual.2.bias`),
    { inC, inH: H, inW: W, outC: hiddenC, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });

  // GroupNorm(hiddenC)
  const gn2 = dispatchGroupNorm(device, enc, conv1.buffer,
    raw.get(`${prefix}.residual.3.weight`), raw.get(`${prefix}.residual.3.bias`),
    { C: hiddenC, H, W, numGroups: Math.min(numGroups, hiddenC), eps: 1e-5 });

  // ReLU
  const relu2 = dispatchActivation(device, enc, gn2, null, hiddenC * H * W, 0);

  // Conv3x3(hiddenC → outC)
  const conv2 = dispatchConv2d(device, enc, relu2,
    raw.get(`${prefix}.residual.5.weight`), raw.get(`${prefix}.residual.5.bias`),
    { inC: hiddenC, inH: H, inW: W, outC: outC, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });

  device.queue.submit([enc.finish()]);

  // Skip connection (identity if inC == outC)
  if (inC === outC) {
    const addEnc = device.createCommandEncoder();
    const sumBuf = dispatchActivation(device, addEnc, inputBuf, conv2.buffer, outC * H * W, 2);
    device.queue.submit([addEnc.finish()]);
    return sumBuf;
  }
  return conv2.buffer;
}

/**
 * Dispatch a texture/geometry head.
 * Structure: GNResBlock → GNResBlock → ReLU → Conv1x1(128→32) → ReLU
 */
function dispatchHead(device, inputBuf, C, H, W, prefix, raw, numGroups) {
  // Block 0: residual with GN
  let features = dispatchGroupNormResidualBlock(device, inputBuf, C, C, C / 2, H, W, `${prefix}.0`, raw, numGroups);
  // Block 1: residual with GN
  features = dispatchGroupNormResidualBlock(device, features, C, C, C / 2, H, W, `${prefix}.1`, raw, numGroups);

  const enc = device.createCommandEncoder();
  // ReLU
  const relu = dispatchActivation(device, enc, features, null, C * H * W, 0);
  // Conv1x1(128 → 32)
  const conv = dispatchConv1x1(device, enc, relu,
    raw.get(`${prefix}.3.weight`), raw.get(`${prefix}.3.bias`),
    { inC: C, outC: 32, H, W });
  // ReLU
  const out = dispatchActivation(device, enc, conv.buffer, null, 32 * H * W, 0);
  device.queue.submit([enc.finish()]);

  return out;
}

export class GaussianPipeline {
  constructor(device) {
    this.device = device;
  }

  /**
   * Run the full Gaussian prediction pipeline.
   *
   * @param {GPUBuffer[]} spnFeatures - 5 multi-resolution feature maps from SPN
   * @param {{C,H,W}[]} spnDims - dimensions per feature map
   * @param {GPUBuffer} disparityBuf - [2, H, W] disparity from monodepth
   * @param {number} dispH - disparity height (1536)
   * @param {number} dispW - disparity width (1536)
   * @param {Float32Array} chwImage - [3, 1536, 1536] normalized image ([-1, 1])
   * @param {Object} weights - weights with .raw accessor
   * @returns {Promise<{ gaussians: Float32Array, numGaussians: number, numLayers: number, H: number, W: number }>}
   */
  async run(spnFeatures, spnDims, disparityBuf, dispH, dispW, chwImage, weights) {
    const device = this.device;
    const raw = weights.raw;
    const fmPrefix = 'feature_model';
    const phPrefix = 'prediction_head';
    const decoderDim = 128;
    const numGroups = 8;
    const numLayers = 2; // Gaussian layers
    const stride = 2;    // Output stride

    console.log('[Gaussian] Running initializer...');

    // --- Step 1: Initializer (pure math, no weights) ---
    // Read disparity from GPU
    const dispData = await readBuffer(device, disparityBuf, 2 * dispH * dispW * 4);

    // Convert image from [-1,1] to [0,1] for initializer
    const imgSize = dispH; // 1536
    const img01 = new Float32Array(3 * imgSize * imgSize);
    for (let i = 0; i < chwImage.length; i++) {
      img01[i] = (chwImage[i] + 1.0) * 0.5;
    }

    // Create feature_input: cat(image[0,1], normalized_disparity) → [5, H, W], then 2*x - 1
    const featureInput = new Float32Array(5 * imgSize * imgSize);
    const HW = imgSize * imgSize;
    // Copy image channels (already in [0,1])
    featureInput.set(img01.subarray(0, 3 * HW));
    // Compute normalized disparity from depth (disparity_factor / depth)
    // disparity_factor = 1.0 (default), depth = disparity_factor / disparity
    // So normalized_disparity = disparity (channels 0 and 1)
    for (let c = 0; c < 2; c++) {
      for (let i = 0; i < HW; i++) {
        featureInput[(3 + c) * HW + i] = dispData[c * HW + i];
      }
    }
    // Normalize to [-1, 1]
    for (let i = 0; i < featureInput.length; i++) {
      featureInput[i] = 2.0 * featureInput[i] - 1.0;
    }

    await breathe();

    // --- Step 2: Gaussian decoder (feature_model) ---
    console.log('[Gaussian] Running decoder (MultiresConvDecoder)...');

    // The Gaussian decoder's MultiresConvDecoder uses the SAME SPN features
    // as the monodepth decoder, just with 128-dim instead of 256-dim.
    // dims_encoder = [256, 256, 512, 1024, 1024], dims_decoder = [128]*5
    const projected = [];

    // convs[0]: Conv2d(256→128, 1x1, bias=false)
    let enc = device.createCommandEncoder();
    const conv0 = dispatchConv1x1(device, enc, spnFeatures[0],
      raw.get(`${fmPrefix}.decoder.convs.0.weight`), null,
      { inC: spnDims[0].C, outC: decoderDim, H: spnDims[0].H, W: spnDims[0].W });
    device.queue.submit([enc.finish()]);
    projected[0] = { buffer: conv0.buffer, C: decoderDim, H: spnDims[0].H, W: spnDims[0].W };

    // convs[1-4]: Conv2d(inC→128, 3x3, bias=false)
    for (let i = 1; i <= 4; i++) {
      enc = device.createCommandEncoder();
      const result = dispatchConv2d(device, enc, spnFeatures[i],
        raw.get(`${fmPrefix}.decoder.convs.${i}.weight`), null,
        { inC: spnDims[i].C, inH: spnDims[i].H, inW: spnDims[i].W,
          outC: decoderDim, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });
      device.queue.submit([enc.finish()]);
      projected[i] = { buffer: result.buffer, C: decoderDim, H: result.outH, W: result.outW };
    }
    await breathe();

    // Fuse from lowest to highest resolution
    let features = dispatchFusionBlock(device,
      projected[4].buffer, null, decoderDim, projected[4].H, projected[4].W,
      `${fmPrefix}.decoder.fusions.4`, raw, true);
    await breathe();

    for (let i = 3; i >= 0; i--) {
      features = dispatchFusionBlock(device,
        features.buffer, projected[i].buffer,
        decoderDim, features.H, features.W,
        `${fmPrefix}.decoder.fusions.${i}`, raw, i > 0);
      await breathe();
    }

    console.log(`[Gaussian]   Decoder output: [${decoderDim}, ${features.H}, ${features.W}]`);

    // upsample = identity (stride_out = 2)

    // --- Step 3: Image encoder (SkipConvBackbone) ---
    // Conv2d(5→128, kernel_size=2, stride=2, bias=true)
    const featureInputBuf = createStorageBuffer(device, featureInput);
    enc = device.createCommandEncoder();
    const skipResult = dispatchConv2d(device, enc, featureInputBuf,
      raw.get(`${fmPrefix}.image_encoder.conv.weight`),
      raw.get(`${fmPrefix}.image_encoder.conv.bias`),
      { inC: 5, inH: imgSize, inW: imgSize, outC: decoderDim,
        kH: 2, kW: 2, padH: 0, padW: 0, strideH: 2, strideW: 2 });
    device.queue.submit([enc.finish()]);
    console.log(`[Gaussian]   Image encoder: [5, ${imgSize}, ${imgSize}] → [${decoderDim}, ${skipResult.outH}, ${skipResult.outW}]`);
    featureInputBuf.destroy();
    await breathe();

    // --- Step 4: Fusion block (decoder + skip) ---
    const fused = dispatchFusionBlock(device,
      features.buffer, skipResult.buffer,
      decoderDim, features.H, features.W,
      `${fmPrefix}.fusion`, raw, false);
    console.log(`[Gaussian]   Fusion: [${decoderDim}, ${fused.H}, ${fused.W}]`);
    await breathe();

    // --- Step 5: Texture and geometry heads ---
    console.log('[Gaussian] Running texture/geometry heads...');
    const textureFeatures = dispatchHead(device, fused.buffer, decoderDim, fused.H, fused.W,
      `${fmPrefix}.texture_head`, raw, numGroups);
    await breathe();

    const geometryFeatures = dispatchHead(device, fused.buffer, decoderDim, fused.H, fused.W,
      `${fmPrefix}.geometry_head`, raw, numGroups);
    await breathe();

    console.log(`[Gaussian]   Heads output: texture=[32, ${fused.H}, ${fused.W}] geometry=[32, ${fused.H}, ${fused.W}]`);

    // --- Step 6: Prediction head (DirectPredictionHead) ---
    // geometry: Conv2d(32, 3*numLayers=6, 1x1)
    // texture: Conv2d(32, 11*numLayers=22, 1x1)
    enc = device.createCommandEncoder();
    const geomDeltas = dispatchConv1x1(device, enc, geometryFeatures,
      raw.get(`${phPrefix}.geometry_prediction_head.weight`),
      raw.get(`${phPrefix}.geometry_prediction_head.bias`),
      { inC: 32, outC: 3 * numLayers, H: fused.H, W: fused.W });
    const texDeltas = dispatchConv1x1(device, enc, textureFeatures,
      raw.get(`${phPrefix}.texture_prediction_head.weight`),
      raw.get(`${phPrefix}.texture_prediction_head.bias`),
      { inC: 32, outC: 11 * numLayers, H: fused.H, W: fused.W });
    device.queue.submit([enc.finish()]);

    console.log(`[Gaussian]   Prediction head: geometry=[${3 * numLayers}, ${fused.H}, ${fused.W}] texture=[${11 * numLayers}, ${fused.H}, ${fused.W}]`);

    // Store delta buffers for downstream compose step
    this._geomDeltasBuf = geomDeltas.buffer;
    this._texDeltasBuf = texDeltas.buffer;

    // Read back deltas
    const geomData = await readBuffer(device, geomDeltas.buffer, 3 * numLayers * fused.H * fused.W * 4);
    const texData = await readBuffer(device, texDeltas.buffer, 11 * numLayers * fused.H * fused.W * 4);

    const outH = fused.H, outW = fused.W;
    const numGaussians = numLayers * outH * outW;

    console.log(`[Gaussian] Output: ${numGaussians} Gaussians (${numLayers} layers × ${outH}×${outW})`);

    // --- Step 7: Compose Gaussians (CPU for now) ---
    // Combine geometry (3 mean deltas) + texture (3 scale + 4 quat + 3 color + 1 opacity deltas)
    // into a flat array of Gaussian parameters
    // Each Gaussian: [x, y, z, sx, sy, sz, qw, qx, qy, qz, r, g, b, opacity] = 14 floats
    const gaussians = new Float32Array(numGaussians * 14);
    const HW2 = outH * outW;

    for (let layer = 0; layer < numLayers; layer++) {
      for (let p = 0; p < HW2; p++) {
        const gIdx = (layer * HW2 + p) * 14;

        // Geometry deltas: [3, numLayers, H, W] but stored as [3*numLayers, H, W]
        // After unflatten: channel c, layer l → index (c * numLayers + l) * HW + p
        for (let c = 0; c < 3; c++) {
          gaussians[gIdx + c] = geomData[(c * numLayers + layer) * HW2 + p];
        }

        // Texture deltas: [11, numLayers, H, W] stored as [11*numLayers, H, W]
        for (let c = 0; c < 11; c++) {
          gaussians[gIdx + 3 + c] = texData[(c * numLayers + layer) * HW2 + p];
        }
      }
    }

    return {
      gaussians,
      numGaussians,
      numLayers,
      H: outH,
      W: outW,
    };
  }
}
