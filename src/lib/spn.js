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

import { createStorageBuffer } from './gpu.js';
import { ViTEncoder, VIT_CONFIG } from './backbone.js';
import {
  dispatchConv1x1,
  dispatchConvTranspose2d,
  dispatchConcatChannels,
  dispatchMergeTokenPatches,
} from './shader_ops.js';
import { planSpnFusionChunks, schedulerYield } from './scheduler.js';

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
  async run(chwImage, options = {}) {
    const device = this.device;
    const { inputSize, patchSize, tokenSize, D, dimsEncoder } = SPN_CONFIG;
    const scheduler = options.scheduler || null;
    const effective = scheduler?.effective || {};
    const telemetry = options.telemetry || null;

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
    // Processing in scheduler-sized chunks with a yield between chunks prevents
    // GPU saturation from starving the system / hanging the box.
    const patchChunkSize = effective.spnPatchChunkSize || 4;
    console.log('[SPN] Running patch encoder on 35 patches (chunks of ' + patchChunkSize + ')...');
    const tokenH = tokenSize, tokenW = tokenSize;
    const N = tokenH * tokenW + 1; // 577

    const patchTokenBuffers = [];   // final normed tokens per patch
    const layer5TokenBuffers = [];  // intermediate layer 5 for first 25 patches
    const layer11TokenBuffers = []; // intermediate layer 11 for first 25 patches

    for (let chunkStart = 0; chunkStart < allPatches.length; chunkStart += patchChunkSize) {
      const chunkEnd = Math.min(chunkStart + patchChunkSize, allPatches.length);

      for (let p = chunkStart; p < chunkEnd; p++) {
        const patchBuf = createStorageBuffer(device, allPatches[p]);

        const result = await this.vitEncoder.encode(patchBuf, this._patchWeights, tokenH, tokenW, {
          scheduler,
          telemetry,
          encoderLabel: 'patch',
          patchIndex: p,
          retainOutputs: true,
        });

        patchTokenBuffers.push(result.finalTokensBuf);

        for (const snap of result.intermediateFeatures) {
          let retained = false;
          if (p < x0.patches.length) {
            if (snap.layerIdx === SPN_CONFIG.intermediateLayers[0]) {
              layer5TokenBuffers.push(snap.buffer);
              retained = true;
            } else if (snap.layerIdx === SPN_CONFIG.intermediateLayers[1]) {
              layer11TokenBuffers.push(snap.buffer);
              retained = true;
            }
          }
          if (!retained) {
            snap.buffer.destroy();
            snap._destroyed = true;
          }
        }

        patchBuf.destroy();
      }

      console.log(`[SPN]   Patch ${chunkEnd}/${allPatches.length} done`);

      // Yield between chunks to let the GPU/system breathe
      if (chunkEnd < allPatches.length) {
        await schedulerYield(scheduler, device, telemetry, 'spn-patch-chunk', { chunkStart, chunkEnd, totalPatches: allPatches.length });
      }
    }

    // Step 3: strip CLS, trim, and merge patch-token features on GPU.
    console.log('[SPN] Merging features on GPU...');
    const mergeEnc = device.createCommandEncoder();
    const latent0Merged = dispatchMergeTokenPatches(device, mergeEnc, layer5TokenBuffers,
      { steps: x0.steps, D, tokenH, tokenW, padding });
    const latent1Merged = dispatchMergeTokenPatches(device, mergeEnc, layer11TokenBuffers,
      { steps: x0.steps, D, tokenH, tokenW, padding });
    const x0Merged = dispatchMergeTokenPatches(device, mergeEnc, patchTokenBuffers.slice(0, 25),
      { steps: x0.steps, D, tokenH, tokenW, padding });
    const x1Merged = dispatchMergeTokenPatches(device, mergeEnc, patchTokenBuffers.slice(25, 34),
      { steps: x1.steps, D, tokenH, tokenW, padding: 2 * padding });
    const x2Merged = dispatchMergeTokenPatches(device, mergeEnc, patchTokenBuffers.slice(34, 35),
      { steps: 1, D, tokenH, tokenW, padding: 0 });
    device.queue.submit([mergeEnc.finish()]);
    await schedulerYield(scheduler, device, telemetry, 'spn-fusion', {
      block: 'spn-patch-merge-gpu',
      patchTokenBuffers: patchTokenBuffers.length,
      intermediateTokenBuffers: layer5TokenBuffers.length + layer11TokenBuffers.length,
    });

    for (const buf of patchTokenBuffers) buf.destroy();
    for (const buf of layer5TokenBuffers) buf.destroy();
    for (const buf of layer11TokenBuffers) buf.destroy();
    for (const merged of [latent0Merged, latent1Merged, x0Merged, x1Merged, x2Merged]) {
      for (const scratch of merged.scratchBuffers || []) scratch.destroy();
    }

    console.log(`[SPN] Merged: latent0=[${D},${latent0Merged.H},${latent0Merged.W}] latent1=[${D},${latent1Merged.H},${latent1Merged.W}] x0=[${D},${x0Merged.H},${x0Merged.W}] x1=[${D},${x1Merged.H},${x1Merged.W}] x2=[${D},${tokenSize},${tokenSize}]`);

    // Step 4: run image encoder on 384x384
    console.log('[SPN] Running image encoder...');
    const imgBuf384 = createStorageBuffer(device, img384);
    const imgResult = await this.vitEncoder.encode(imgBuf384, this._imageWeights, tokenH, tokenW, {
      scheduler,
      telemetry,
      encoderLabel: 'image',
      retainOutputs: true,
    });
    const imageMergeEnc = device.createCommandEncoder();
    const imgFeature = dispatchMergeTokenPatches(device, imageMergeEnc, [imgResult.finalTokensBuf],
      { steps: 1, D, tokenH, tokenW, padding: 0 });
    device.queue.submit([imageMergeEnc.finish()]);
    await schedulerYield(scheduler, device, telemetry, 'spn-image-encoder', { tokenCount: N, block: 'image-token-merge-gpu' });
    imgBuf384.destroy();
    imgResult.finalTokensBuf.destroy();
    for (const scratch of imgFeature.scratchBuffers || []) scratch.destroy();
    // Clean up image encoder buffers (no intermediates needed)
    for (const snap of imgResult.intermediateFeatures) {
      if (!snap._destroyed) { snap.buffer.destroy(); snap._destroyed = true; }
    }

    // Step 5: upsample all features through fusion layers
    console.log('[SPN] Running upsample fusion...');
    const raw = this.weights.raw;

    const latent0Buf = latent0Merged.buffer;
    const latent1Buf = latent1Merged.buffer;
    const x0Buf = x0Merged.buffer;
    const x1Buf = x1Merged.buffer;
    const x2Buf = x2Merged.buffer;
    const imgFeatureBuf = imgFeature.buffer;

    const prefix = 'monodepth_model.monodepth_predictor.encoder.';

    // Upsample latent0: 1x1 conv (1024→256) + 3x ConvTranspose2d (256→256, stride=2)
    let feat0 = await this._dispatchUpsampleBlock(latent0Buf, latent0Merged.H, latent0Merged.W,
      `${prefix}upsample_latent0`, [1024, 256, 256, 256], [256, 256, 256, 256], 4, 'upsample_latent0', scheduler, telemetry);
    await schedulerYield(scheduler, device, telemetry, 'spn-fusion', { block: 'upsample_latent0', role: 'group-complete', layerCount: 4 });

    // Upsample latent1: 1x1 conv (1024→256) + 2x ConvTranspose2d
    let feat1 = await this._dispatchUpsampleBlock(latent1Buf, latent1Merged.H, latent1Merged.W,
      `${prefix}upsample_latent1`, [1024, 256, 256], [256, 256, 256], 3, 'upsample_latent1', scheduler, telemetry);
    await schedulerYield(scheduler, device, telemetry, 'spn-fusion', { block: 'upsample_latent1', role: 'group-complete', layerCount: 3 });

    // Upsample0: 1x1 conv (1024→512) + 1x ConvTranspose2d
    let feat2 = await this._dispatchUpsampleBlock(x0Buf, x0Merged.H, x0Merged.W,
      `${prefix}upsample0`, [1024, 512], [512, 512], 2, 'upsample0', scheduler, telemetry);
    await schedulerYield(scheduler, device, telemetry, 'spn-fusion', { block: 'upsample0', role: 'group-complete', layerCount: 2 });

    // Upsample1: 1x1 conv (1024→1024) + 1x ConvTranspose2d
    let feat3 = await this._dispatchUpsampleBlock(x1Buf, x1Merged.H, x1Merged.W,
      `${prefix}upsample1`, [1024, 1024], [1024, 1024], 2, 'upsample1', scheduler, telemetry);
    await schedulerYield(scheduler, device, telemetry, 'spn-fusion', { block: 'upsample1', role: 'group-complete', layerCount: 2 });

    // Upsample2: 1x1 conv (1024→1024) + 1x ConvTranspose2d
    let feat4x2 = await this._dispatchUpsampleBlock(x2Buf, tokenSize, tokenSize,
      `${prefix}upsample2`, [1024, 1024], [1024, 1024], 2, 'upsample2', scheduler, telemetry);
    await schedulerYield(scheduler, device, telemetry, 'spn-fusion', { block: 'upsample2', role: 'group-complete', layerCount: 2 });

    // Upsample lowres: single ConvTranspose2d (1024→1024, stride=2, bias=true)
    const lowresResult = await this._dispatchChunkedConvTranspose2d({
      inputBuf: imgFeatureBuf,
      weightBuf: raw.get(`${prefix}upsample_lowres.weight`),
      biasBuf: raw.get(`${prefix}upsample_lowres.bias`),
      inC: 1024,
      inH: tokenSize,
      inW: tokenSize,
      outC: 1024,
      stride: 2,
      blockLabel: 'upsample-lowres',
      parentBlock: null,
      scheduler,
      telemetry,
    });

    // Fuse lowres: concat(x2_upsampled, lowres_upsampled) → 1x1 conv (2048→1024)
    // Keep the concat GPU-resident so this midstream wall does not force a readback/upload pair.
    const fusedH = Math.min(feat4x2.H, lowresResult.H);
    const fusedW = Math.min(feat4x2.W, lowresResult.W);
    const concatEnc = device.createCommandEncoder();
    const concatResult = dispatchConcatChannels(device, concatEnc, feat4x2.buffer, lowresResult.buffer,
      { aC: 1024, bC: 1024, H: fusedH, W: fusedW });
    device.queue.submit([concatEnc.finish()]);
    await schedulerYield(scheduler, device, telemetry, 'spn-fusion', { block: 'gpu-concat-lowres' });

    const fuseEnc = device.createCommandEncoder();
    const fusedResult = dispatchConv1x1(device, fuseEnc, concatResult.buffer,
      raw.get(`${prefix}fuse_lowres.weight`),
      raw.get(`${prefix}fuse_lowres.bias`),
      { inC: 2048, outC: 1024, H: fusedH, W: fusedW });
    device.queue.submit([fuseEnc.finish()]);
    await schedulerYield(scheduler, device, telemetry, 'spn-fusion', { block: 'fuse-lowres' });

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
    concatResult.buffer.destroy();
    feat4x2.buffer.destroy();
    lowresResult.buffer.destroy();

    return { features: features.map(f => f.buffer), featureDims };
  }

  async _dispatchChunkedConvTranspose2d({
    inputBuf,
    weightBuf,
    biasBuf = null,
    inC,
    inH,
    inW,
    outC,
    stride,
    blockLabel,
    parentBlock,
    layerIndex = null,
    layerCount = null,
    scheduler,
    telemetry,
  }) {
    const device = this.device;
    const effective = scheduler?.effective || {};
    const outH = inH * stride;
    const outW = inW * stride;
    const totalOutputItems = outC * outH * outW;
    const outputChunks = planSpnFusionChunks(totalOutputItems, effective.spnFusionChunkItems || 0);
    let result = null;
    let outputBuffer = null;

    for (const outputChunk of outputChunks) {
      const enc = device.createCommandEncoder();
      result = dispatchConvTranspose2d(device, enc, inputBuf, weightBuf, biasBuf, {
        inC,
        inH,
        inW,
        outC,
        stride,
        outputBuffer,
        outputStart: outputChunk.outputStart,
        outputCount: outputChunk.outputCount,
      });
      outputBuffer = result.buffer;
      device.queue.submit([enc.finish()]);

      const isFinalChunk = outputChunk.chunkIndex === outputChunk.chunkCount - 1;
      await schedulerYield(scheduler, device, telemetry, 'spn-fusion', {
        block: isFinalChunk ? blockLabel : `${blockLabel}.output-chunk-${outputChunk.chunkIndex}`,
        parentBlock: isFinalChunk ? parentBlock : blockLabel,
        ...(isFinalChunk
          ? { role: 'wait-bearing-layer', chunkRole: 'spn-fusion-output-chunk' }
          : { role: 'spn-fusion-output-chunk' }),
        layerIndex,
        layerCount,
        op: 'conv-transpose2d',
        C: outC,
        H: outH,
        W: outW,
        outputChunkIndex: outputChunk.chunkIndex,
        outputChunkCount: outputChunk.chunkCount,
        outputStart: outputChunk.outputStart,
        outputEnd: outputChunk.outputEnd,
        outputCount: outputChunk.outputCount,
        totalOutputItems,
      });
    }

    return result;
  }

  /**
   * Dispatch a sequential upsample block: 1x1 conv + N ConvTranspose2d layers.
   * All layers have bias=false.
   */
  async _dispatchUpsampleBlock(inputBuf, H, W, prefix, inChannels, outChannels, numLayers, blockLabel, scheduler, telemetry) {
    const device = this.device;
    const raw = this.weights.raw;
    let currentBuf = inputBuf;
    let currentH = H, currentW = W;
    let currentC = inChannels[0];

    for (let i = 0; i < numLayers; i++) {
      const weight = raw.get(`${prefix}.${i}.weight`);
      let result;

      if (i === 0) {
        // First layer: 1x1 Conv2d projection (no bias)
        const enc = device.createCommandEncoder();
        result = dispatchConv1x1(device, enc, currentBuf, weight, null,
          { inC: inChannels[i], outC: outChannels[i], H: currentH, W: currentW });
        device.queue.submit([enc.finish()]);
        currentC = outChannels[i];
      } else {
        // Subsequent layers: ConvTranspose2d stride=2 (no bias)
        result = await this._dispatchChunkedConvTranspose2d({
          inputBuf: currentBuf,
          weightBuf: weight,
          inC: inChannels[i],
          inH: currentH,
          inW: currentW,
          outC: outChannels[i],
          stride: 2,
          blockLabel: `${blockLabel}.layer-${i}`,
          parentBlock: blockLabel,
          layerIndex: i,
          layerCount: numLayers,
          scheduler,
          telemetry,
        });
        currentH = result.H;
        currentW = result.W;
        currentC = outChannels[i];
      }

      if (i === 0) {
        await schedulerYield(scheduler, device, telemetry, 'spn-fusion', {
          block: `${blockLabel}.layer-${i}`,
          parentBlock: blockLabel,
          role: 'wait-bearing-layer',
          layerIndex: i,
          layerCount: numLayers,
          op: 'conv1x1',
          C: currentC,
          H: currentH,
          W: currentW,
        });
      }
      // Destroy previous intermediate buffer (not the original input — caller owns that)
      if (currentBuf !== inputBuf) currentBuf.destroy();
      currentBuf = result.buffer;
    }

    return { buffer: currentBuf, C: currentC, H: currentH, W: currentW };
  }
}

export { SPN_CONFIG };
