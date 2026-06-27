/**
 * spn.js — SlidingPyramidNetwork for SHARP-WebGPU.
 *
 * Takes a 1536x1536 image and produces 5 multi-resolution feature maps:
 *   [0] latent0: [256, ~768, ~768] from layer 5 intermediates, 5x5 patches, 3x upsample
 *   [1] latent1: [256, ~384, ~384] from layer 11 intermediates, 5x5 patches, 2x upsample
 *   [2] x0:      [512, ~192, ~192] from final output, 5x5 patches, 1x upsample
 *   [3] x1:      [1024, ~96, ~96]  from final output, 3x3 patches, 1x upsample
 *   [4] fused:   [1024, ~48, ~48]  from final output 1x1 + image encoder, fused
 *
 * Pipeline:
 *   1. Create 3-level pyramid: 1536 → 768 → 384
 *   2. Extract overlapping patches: 5x5 (25) + 3x3 (9) + 1x1 (1) = 35 patches
 *   3. Run patch_encoder on each patch (sequential, 35 dispatches)
 *   4. Extract intermediate features at layers [5, 11] from first 25 patches
 *   5. Merge overlapping patches (trim-based, no blending)
 *   6. Run image_encoder on 384x384 low-res image
 *   7. Upsample all features through fusion layers
 *   8. Fuse lowest level with image encoder output
 */

import { createStorageBuffer, createEmptyBuffer, readBuffer } from './gpu.js';
import { ViTEncoder, VIT_CONFIG } from './backbone.js';
import {
  dispatchConv1x1,
  dispatchConvTranspose2d,
} from './shader_ops.js';

const SPN_CONFIG = {
  inputSize: 1536,       // full pipeline input size
  patchSize: 384,        // ViT input size
  tokenSize: 24,         // 384 / 16 = 24 tokens per side
  D: 1024,               // embed dim
  dimsEncoder: [256, 256, 512, 1024, 1024], // output channel dims per level
  intermediateLayers: [5, 11, 17, 23],
};

/**
 * Extract overlapping patches from an image on the CPU.
 * Returns array of Float32Array in CHW layout, each 3×384×384.
 */
function extractPatches(chwData, imgSize, overlapRatio, patchSize) {
  const patchStride = Math.floor(patchSize * (1 - overlapRatio));
  const steps = Math.ceil((imgSize - patchSize) / patchStride) + 1;
  const C = 3;
  const patches = [];

  for (let j = 0; j < steps; j++) {
    const j0 = j * patchStride;
    for (let i = 0; i < steps; i++) {
      const i0 = i * patchStride;
      const patch = new Float32Array(C * patchSize * patchSize);
      for (let c = 0; c < C; c++) {
        for (let py = 0; py < patchSize; py++) {
          for (let px = 0; px < patchSize; px++) {
            patch[c * patchSize * patchSize + py * patchSize + px] =
              chwData[c * imgSize * imgSize + (j0 + py) * imgSize + (i0 + px)];
          }
        }
      }
      patches.push(patch);
    }
  }

  return { patches, steps };
}

/**
 * Bilinear downsample a CHW float32 image on the CPU.
 * align_corners=false to match PyTorch F.interpolate default.
 */
function bilinearDownsample(chwData, srcSize, dstSize) {
  const C = 3;
  const out = new Float32Array(C * dstSize * dstSize);
  const scale = srcSize / dstSize;

  for (let c = 0; c < C; c++) {
    for (let dy = 0; dy < dstSize; dy++) {
      for (let dx = 0; dx < dstSize; dx++) {
        // align_corners=false: src = (dst + 0.5) * scale - 0.5
        const sy = (dy + 0.5) * scale - 0.5;
        const sx = (dx + 0.5) * scale - 0.5;

        const y0 = Math.max(0, Math.floor(sy));
        const x0 = Math.max(0, Math.floor(sx));
        const y1 = Math.min(srcSize - 1, y0 + 1);
        const x1 = Math.min(srcSize - 1, x0 + 1);

        const fy = sy - y0;
        const fx = sx - x0;

        const srcBase = c * srcSize * srcSize;
        const v00 = chwData[srcBase + y0 * srcSize + x0];
        const v01 = chwData[srcBase + y0 * srcSize + x1];
        const v10 = chwData[srcBase + y1 * srcSize + x0];
        const v11 = chwData[srcBase + y1 * srcSize + x1];

        out[c * dstSize * dstSize + dy * dstSize + dx] =
          v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
          v10 * (1 - fx) * fy + v11 * fx * fy;
      }
    }
  }
  return out;
}

/**
 * Merge overlapping patch features by trimming overlap regions.
 * Input: array of feature buffers, each [D, tokenH, tokenW] in CHW.
 * Output: single merged buffer [D, mergedH, mergedW].
 */
function mergeFeaturesCPU(featureArrays, steps, D, tokenSize, padding) {
  // Compute merged spatial size
  if (padding === 0) {
    const mergedSize = steps * tokenSize;
    const merged = new Float32Array(D * mergedSize * mergedSize);
    let idx = 0;
    for (let j = 0; j < steps; j++) {
      for (let i = 0; i < steps; i++) {
        const feat = featureArrays[idx++];
        for (let c = 0; c < D; c++) {
          for (let py = 0; py < tokenSize; py++) {
            for (let px = 0; px < tokenSize; px++) {
              merged[c * mergedSize * mergedSize + (j * tokenSize + py) * mergedSize + (i * tokenSize + px)] =
                feat[c * tokenSize * tokenSize + py * tokenSize + px];
            }
          }
        }
      }
    }
    return { data: merged, H: mergedSize, W: mergedSize };
  }

  // With overlap trimming
  // Each patch contributes a trimmed region. Edge patches lose `padding` on the exterior side only.
  const trimmedSizes = [];
  for (let s = 0; s < steps; s++) {
    let h = tokenSize;
    if (s > 0) h -= padding;           // trim top/left interior edge
    if (s < steps - 1) h -= padding;   // trim bottom/right interior edge
    trimmedSizes.push(h);
  }
  const mergedSize = trimmedSizes.reduce((a, b) => a + b, 0);

  const merged = new Float32Array(D * mergedSize * mergedSize);
  let idx = 0;
  let dstY = 0;

  for (let j = 0; j < steps; j++) {
    const rowStartY = (j > 0) ? padding : 0;
    const rowEndY = tokenSize - ((j < steps - 1) ? padding : 0);
    const rowH = rowEndY - rowStartY;
    let dstX = 0;

    for (let i = 0; i < steps; i++) {
      const colStartX = (i > 0) ? padding : 0;
      const colEndX = tokenSize - ((i < steps - 1) ? padding : 0);
      const colW = colEndX - colStartX;
      const feat = featureArrays[idx++];

      for (let c = 0; c < D; c++) {
        for (let py = 0; py < rowH; py++) {
          for (let px = 0; px < colW; px++) {
            merged[c * mergedSize * mergedSize + (dstY + py) * mergedSize + (dstX + px)] =
              feat[c * tokenSize * tokenSize + (rowStartY + py) * tokenSize + (colStartX + px)];
          }
        }
      }
      dstX += colW;
    }
    dstY += rowH;
  }

  return { data: merged, H: mergedSize, W: mergedSize };
}

/**
 * Strip CLS token and reshape [N, D] → [D, tokenH, tokenW] (CHW).
 * Returns Float32Array.
 */
function reshapeFeature(tokenData, D, tokenH, tokenW) {
  const numPatches = tokenH * tokenW;
  const out = new Float32Array(D * numPatches);
  // Input: [N, D] where N = numPatches + 1 (CLS at index 0)
  // Output: [D, tokenH, tokenW] (CHW)
  for (let d = 0; d < D; d++) {
    for (let p = 0; p < numPatches; p++) {
      // input token index is p+1 (skip CLS), dimension d
      out[d * numPatches + p] = tokenData[(p + 1) * D + d];
    }
  }
  return out;
}

export class SlidingPyramidNetwork {
  constructor(device) {
    this.device = device;
    this.vitEncoder = new ViTEncoder(device);
  }

  init(weights) {
    this.vitEncoder.init();
    this.weights = weights;

    // Remap patch encoder block weights
    const patchPrefix = 'monodepth_model.monodepth_predictor.encoder.patch_encoder.';
    this._patchWeights = {
      patchEmbed: weights.patchEncoder.patchEmbed,
      posEmbed: weights.patchEncoder.posEmbed,
      clsToken: weights.patchEncoder.clsToken,
      norm: weights.patchEncoder.norm,
      blockWeights: {},
    };
    for (const [key, buf] of Object.entries(weights.patchEncoder.blockWeights)) {
      this._patchWeights.blockWeights[key.replace(patchPrefix, '')] = buf;
    }

    // Remap image encoder block weights
    const imagePrefix = 'monodepth_model.monodepth_predictor.encoder.image_encoder.';
    this._imageWeights = {
      patchEmbed: weights.imageEncoder.patchEmbed,
      posEmbed: weights.imageEncoder.posEmbed,
      clsToken: weights.imageEncoder.clsToken,
      norm: weights.imageEncoder.norm,
      blockWeights: {},
    };
    for (const [key, buf] of Object.entries(weights.imageEncoder.blockWeights)) {
      this._imageWeights.blockWeights[key.replace(imagePrefix, '')] = buf;
    }
  }

  /**
   * Run the full SPN pipeline.
   * @param {Float32Array} chwImage - [3, 1536, 1536] normalized to [-1, 1]
   * @returns {Promise<{ features: GPUBuffer[], featureDims: {C,H,W}[] }>}
   */
  async run(chwImage) {
    const device = this.device;
    const { inputSize, patchSize, tokenSize, D, dimsEncoder } = SPN_CONFIG;

    console.log('[SPN] Creating image pyramid...');
    // Step 0: pyramid
    const img1536 = chwImage; // 1536x1536
    const img768 = bilinearDownsample(img1536, 1536, 768);
    const img384 = bilinearDownsample(img1536, 1536, 384);

    // Step 1: extract patches
    console.log('[SPN] Extracting patches...');
    const x0 = extractPatches(img1536, 1536, 0.25, patchSize); // 5x5 = 25
    const x1 = extractPatches(img768, 768, 0.5, patchSize);    // 3x3 = 9
    const x2 = { patches: [img384], steps: 1 };                 // 1x1 = 1
    const padding = 3;

    const allPatches = [...x0.patches, ...x1.patches, ...x2.patches];
    console.log(`[SPN] ${allPatches.length} patches (${x0.steps}x${x0.steps} + ${x1.steps}x${x1.steps} + 1x1)`);

    // Step 2: run patch encoder on all 35 patches in chunks with yields
    // Processing in chunks of CHUNK_SIZE with a yield between chunks prevents
    // GPU saturation from starving the system / hanging the box.
    const CHUNK_SIZE = 4;
    console.log('[SPN] Running patch encoder on 35 patches (chunks of ' + CHUNK_SIZE + ')...');
    const tokenH = tokenSize, tokenW = tokenSize;
    const N = tokenH * tokenW + 1; // 577

    const patchOutputs = [];      // final normed tokens per patch
    const layer5Features = [];     // intermediate layer 5 for first 25 patches
    const layer11Features = [];    // intermediate layer 11 for first 25 patches

    for (let chunkStart = 0; chunkStart < allPatches.length; chunkStart += CHUNK_SIZE) {
      const chunkEnd = Math.min(chunkStart + CHUNK_SIZE, allPatches.length);

      for (let p = chunkStart; p < chunkEnd; p++) {
        const patchBuf = createStorageBuffer(device, allPatches[p]);

        const enc = device.createCommandEncoder();
        const result = this.vitEncoder.encode(enc, patchBuf, this._patchWeights, tokenH, tokenW);
        device.queue.submit([enc.finish()]);

        // Read back final tokens and destroy the GPU buffer
        const finalData = await readBuffer(device, result.finalTokensBuf, N * D * 4);
        result.finalTokensBuf.destroy();
        patchOutputs.push(finalData);

        // Read back intermediate features for first 25 patches (high-res x0 only)
        // Only layers 5 and 11 are needed — skip reading layers 17 and 23
        for (const snap of result.intermediateFeatures) {
          if (p < x0.patches.length) {
            if (snap.layerIdx === SPN_CONFIG.intermediateLayers[0]) {
              const snapData = await readBuffer(device, snap.buffer, N * D * 4);
              layer5Features.push(snapData);
            } else if (snap.layerIdx === SPN_CONFIG.intermediateLayers[1]) {
              const snapData = await readBuffer(device, snap.buffer, N * D * 4);
              layer11Features.push(snapData);
            }
          }
          // Always destroy intermediate buffers after reading (or skipping)
          snap.buffer.destroy();
          snap._destroyed = true;
        }

        patchBuf.destroy();
      }

      console.log(`[SPN]   Patch ${chunkEnd}/${allPatches.length} done`);

      // Yield between chunks to let the GPU/system breathe
      if (chunkEnd < allPatches.length) {
        await new Promise(r => setTimeout(r, 0));
      }
    }

    // Step 3: reshape features (strip CLS, reshape to CHW) and merge
    console.log('[SPN] Merging features...');

    // Reshape all patch outputs to [D, tokenH, tokenW]
    const reshapedOutputs = patchOutputs.map(data => reshapeFeature(data, D, tokenH, tokenW));

    // Latent0: layer 5 from first 25 patches → merge 5x5 → [D, 96, 96]
    const latent0Reshaped = layer5Features.map(data => reshapeFeature(data, D, tokenH, tokenW));
    const latent0Merged = mergeFeaturesCPU(latent0Reshaped, x0.steps, D, tokenSize, padding);

    // Latent1: layer 11 from first 25 patches → merge 5x5 → [D, 96, 96]
    const latent1Reshaped = layer11Features.map(data => reshapeFeature(data, D, tokenH, tokenW));
    const latent1Merged = mergeFeaturesCPU(latent1Reshaped, x0.steps, D, tokenSize, padding);

    // x0: final output from first 25 patches → merge 5x5 → [D, 96, 96]
    const x0Merged = mergeFeaturesCPU(reshapedOutputs.slice(0, 25), x0.steps, D, tokenSize, padding);

    // x1: final output from next 9 patches → merge 3x3 → [D, 48, 48]
    const x1Merged = mergeFeaturesCPU(reshapedOutputs.slice(25, 34), x1.steps, D, tokenSize, 2 * padding);

    // x2: final output from last patch → [D, 24, 24] (no merge needed)
    const x2Feature = reshapedOutputs[34];

    console.log(`[SPN] Merged: latent0=[${D},${latent0Merged.H},${latent0Merged.W}] latent1=[${D},${latent1Merged.H},${latent1Merged.W}] x0=[${D},${x0Merged.H},${x0Merged.W}] x1=[${D},${x1Merged.H},${x1Merged.W}] x2=[${D},${tokenSize},${tokenSize}]`);

    // Step 4: run image encoder on 384x384
    console.log('[SPN] Running image encoder...');
    const imgBuf384 = createStorageBuffer(device, img384);
    const imgEnc = device.createCommandEncoder();
    const imgResult = this.vitEncoder.encode(imgEnc, imgBuf384, this._imageWeights, tokenH, tokenW);
    device.queue.submit([imgEnc.finish()]);
    const imgTokens = await readBuffer(device, imgResult.finalTokensBuf, N * D * 4);
    const imgFeature = reshapeFeature(imgTokens, D, tokenH, tokenW); // [D, 24, 24]
    imgBuf384.destroy();
    // Clean up image encoder buffers (no intermediates needed)
    imgResult.finalTokensBuf.destroy();
    for (const snap of imgResult.intermediateFeatures) {
      if (!snap._destroyed) { snap.buffer.destroy(); snap._destroyed = true; }
    }

    // Step 5: upsample all features through fusion layers
    console.log('[SPN] Running upsample fusion...');
    const raw = this.weights.raw;

    // Upload merged features to GPU
    const latent0Buf = createStorageBuffer(device, latent0Merged.data);
    const latent1Buf = createStorageBuffer(device, latent1Merged.data);
    const x0Buf = createStorageBuffer(device, x0Merged.data);
    const x1Buf = createStorageBuffer(device, x1Merged.data);
    const x2Buf = createStorageBuffer(device, new Float32Array(x2Feature));
    const imgFeatureBuf = createStorageBuffer(device, new Float32Array(imgFeature));

    const prefix = 'monodepth_model.monodepth_predictor.encoder.';

    // Upsample latent0: 1x1 conv (1024→256) + 3x ConvTranspose2d (256→256, stride=2)
    let feat0 = this._dispatchUpsampleBlock(latent0Buf, latent0Merged.H, latent0Merged.W,
      `${prefix}upsample_latent0`, [1024, 256, 256, 256], [256, 256, 256, 256], 4);

    // Upsample latent1: 1x1 conv (1024→256) + 2x ConvTranspose2d
    let feat1 = this._dispatchUpsampleBlock(latent1Buf, latent1Merged.H, latent1Merged.W,
      `${prefix}upsample_latent1`, [1024, 256, 256], [256, 256, 256], 3);

    // Upsample0: 1x1 conv (1024→512) + 1x ConvTranspose2d
    let feat2 = this._dispatchUpsampleBlock(x0Buf, x0Merged.H, x0Merged.W,
      `${prefix}upsample0`, [1024, 512], [512, 512], 2);

    // Upsample1: 1x1 conv (1024→1024) + 1x ConvTranspose2d
    let feat3 = this._dispatchUpsampleBlock(x1Buf, x1Merged.H, x1Merged.W,
      `${prefix}upsample1`, [1024, 1024], [1024, 1024], 2);

    // Upsample2: 1x1 conv (1024→1024) + 1x ConvTranspose2d
    let feat4x2 = this._dispatchUpsampleBlock(x2Buf, tokenSize, tokenSize,
      `${prefix}upsample2`, [1024, 1024], [1024, 1024], 2);

    // Upsample lowres: single ConvTranspose2d (1024→1024, stride=2, bias=true)
    const lowresEnc = device.createCommandEncoder();
    const lowresResult = dispatchConvTranspose2d(device, lowresEnc, imgFeatureBuf,
      raw.get(`${prefix}upsample_lowres.weight`),
      raw.get(`${prefix}upsample_lowres.bias`),
      { inC: 1024, inH: tokenSize, inW: tokenSize, outC: 1024, stride: 2 });
    device.queue.submit([lowresEnc.finish()]);

    // Fuse lowres: concat(x2_upsampled, lowres_upsampled) → 1x1 conv (2048→1024)
    // Concatenate along channel dimension on CPU for simplicity
    const x2UpData = await readBuffer(device, feat4x2.buffer, feat4x2.C * feat4x2.H * feat4x2.W * 4);
    const lowresData = await readBuffer(device, lowresResult.buffer, lowresResult.C * lowresResult.H * lowresResult.W * 4);
    const fusedH = Math.min(feat4x2.H, lowresResult.H);
    const fusedW = Math.min(feat4x2.W, lowresResult.W);
    const concatData = new Float32Array(2048 * fusedH * fusedW);
    // First 1024 channels from x2_upsampled
    concatData.set(x2UpData.subarray(0, 1024 * fusedH * fusedW));
    // Next 1024 channels from lowres_upsampled
    for (let i = 0; i < 1024 * fusedH * fusedW; i++) {
      concatData[1024 * fusedH * fusedW + i] = lowresData[i];
    }
    const concatBuf = createStorageBuffer(device, concatData);

    const fuseEnc = device.createCommandEncoder();
    const fusedResult = dispatchConv1x1(device, fuseEnc, concatBuf,
      raw.get(`${prefix}fuse_lowres.weight`),
      raw.get(`${prefix}fuse_lowres.bias`),
      { inC: 2048, outC: 1024, H: fusedH, W: fusedW });
    device.queue.submit([fuseEnc.finish()]);

    const features = [feat0, feat1, feat2, feat3, fusedResult];
    const featureDims = features.map(f => ({ C: f.C, H: f.H, W: f.W }));

    console.log('[SPN] Output feature maps:');
    for (let i = 0; i < 5; i++) {
      console.log(`  [${i}] C=${featureDims[i].C} H=${featureDims[i].H} W=${featureDims[i].W}`);
    }

    // Cleanup intermediate buffers
    latent0Buf.destroy();
    latent1Buf.destroy();
    x0Buf.destroy();
    x1Buf.destroy();
    x2Buf.destroy();
    imgFeatureBuf.destroy();
    concatBuf.destroy();
    feat4x2.buffer.destroy();
    lowresResult.buffer.destroy();

    return { features: features.map(f => f.buffer), featureDims };
  }

  /**
   * Dispatch a sequential upsample block: 1x1 conv + N ConvTranspose2d layers.
   * All layers have bias=false.
   */
  _dispatchUpsampleBlock(inputBuf, H, W, prefix, inChannels, outChannels, numLayers) {
    const device = this.device;
    const raw = this.weights.raw;
    let currentBuf = inputBuf;
    let currentH = H, currentW = W;
    let currentC = inChannels[0];

    for (let i = 0; i < numLayers; i++) {
      const weight = raw.get(`${prefix}.${i}.weight`);
      const enc = device.createCommandEncoder();
      let result;

      if (i === 0) {
        // First layer: 1x1 Conv2d projection (no bias)
        result = dispatchConv1x1(device, enc, currentBuf, weight, null,
          { inC: inChannels[i], outC: outChannels[i], H: currentH, W: currentW });
        currentC = outChannels[i];
      } else {
        // Subsequent layers: ConvTranspose2d stride=2 (no bias)
        result = dispatchConvTranspose2d(device, enc, currentBuf, weight, null,
          { inC: inChannels[i], inH: currentH, inW: currentW, outC: outChannels[i], stride: 2 });
        currentH *= 2;
        currentW *= 2;
        currentC = outChannels[i];
      }

      device.queue.submit([enc.finish()]);
      // Destroy previous intermediate buffer (not the original input — caller owns that)
      if (currentBuf !== inputBuf) currentBuf.destroy();
      currentBuf = result.buffer;
    }

    return { buffer: currentBuf, C: currentC, H: currentH, W: currentW };
  }
}

export { SPN_CONFIG };
