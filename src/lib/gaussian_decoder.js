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

import { createStorageBuffer } from './gpu.js';
import {
  dispatchActivation,
  dispatchGroupNorm,
  dispatchGaussianInitializerFeatureInput,
} from './shader_ops.js';
import {
  createDecoderAdaptiveDuty,
  dispatchTiledConv1x1,
  dispatchTiledConv2d,
  dispatchTiledConvTranspose2d,
  dispatchTiledGroupNormRelu,
} from './decoder_duties.js';
import { schedulerYield } from './scheduler.js';

const breathe = () => new Promise(r => setTimeout(r, 0));

// --- Shared decoder dispatch (used by both monodepth and gaussian decoders) ---

async function dispatchResidualBlock(device, inputBuf, C, H, W, prefix, raw, boundaryYield, label, decoderKernelChunkItems, decoderAdaptiveDuty) {
  const count = C * H * W;
  const convParams = {
    inC: C, inH: H, inW: W, outC: C,
    kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1,
  };
  const conv1 = await dispatchTiledConv2d({
    device,
    inputBuf,
    weightBuf: raw.get(`${prefix}.residual.1.weight`),
    biasBuf: raw.get(`${prefix}.residual.1.bias`),
    params: convParams,
    chunkItems: decoderKernelChunkItems,
    adaptiveDuty: decoderAdaptiveDuty,
    phase: 'residual-conv1',
    details: { label, C, H, W },
    boundaryYield,
    prepareInput: (encoder, source) => dispatchActivation(device, encoder, source, null, count, 0),
  });
  const conv2 = await dispatchTiledConv2d({
    device,
    inputBuf: conv1.buffer,
    weightBuf: raw.get(`${prefix}.residual.3.weight`),
    biasBuf: raw.get(`${prefix}.residual.3.bias`),
    params: convParams,
    chunkItems: decoderKernelChunkItems,
    adaptiveDuty: decoderAdaptiveDuty,
    phase: 'residual-conv2',
    details: { label, C, H, W },
    boundaryYield,
    prepareInput: (encoder, source) => dispatchActivation(device, encoder, source, null, count, 0),
  });

  const enc = device.createCommandEncoder();
  const sumBuf = dispatchActivation(device, enc, inputBuf, conv2.buffer, count, 2);
  device.queue.submit([enc.finish()]);
  await boundaryYield('residual-skip-add', { label, C, H, W });

  return sumBuf;
}

async function dispatchFusionBlock(device, x0Buf, x1Buf, C, H, W, prefix, raw, hasDeconv, boundaryYield, label, decoderKernelChunkItems, decoderAdaptiveDuty) {
  let currentBuf = x0Buf;
  let currentH = H, currentW = W;

  if (x1Buf) {
    const res1Buf = await dispatchResidualBlock(
      device,
      x1Buf,
      C,
      currentH,
      currentW,
      `${prefix}.resnet1`,
      raw,
      boundaryYield,
      `${label}.resnet1`,
      decoderKernelChunkItems,
      decoderAdaptiveDuty
    );
    const enc = device.createCommandEncoder();
    currentBuf = dispatchActivation(device, enc, currentBuf, res1Buf, C * currentH * currentW, 2);
    device.queue.submit([enc.finish()]);
    await boundaryYield('fusion-skip-add', { label, C, H: currentH, W: currentW });
  }

  currentBuf = await dispatchResidualBlock(
    device,
    currentBuf,
    C,
    currentH,
    currentW,
    `${prefix}.resnet2`,
    raw,
    boundaryYield,
    `${label}.resnet2`,
    decoderKernelChunkItems,
    decoderAdaptiveDuty
  );

  if (hasDeconv) {
    const deconvResult = await dispatchTiledConvTranspose2d({
      device,
      inputBuf: currentBuf,
      weightBuf: raw.get(`${prefix}.deconv.weight`),
      biasBuf: null,
      params: { inC: C, inH: currentH, inW: currentW, outC: C, stride: 2 },
      chunkItems: decoderKernelChunkItems,
      adaptiveDuty: decoderAdaptiveDuty,
      phase: 'fusion-deconv',
      details: { label, C, H: currentH * 2, W: currentW * 2 },
      boundaryYield,
    });
    currentBuf = deconvResult.buffer;
    currentH *= 2;
    currentW *= 2;
  }

  const outResult = await dispatchTiledConv1x1({
    device,
    inputBuf: currentBuf,
    weightBuf: raw.get(`${prefix}.out_conv.weight`),
    biasBuf: raw.get(`${prefix}.out_conv.bias`),
    params: { inC: C, outC: C, H: currentH, W: currentW },
    chunkItems: decoderKernelChunkItems,
    adaptiveDuty: decoderAdaptiveDuty,
    phase: 'fusion-out-conv',
    details: { label, C, H: currentH, W: currentW },
    boundaryYield,
  });

  return { buffer: outResult.buffer, H: currentH, W: currentW };
}

/**
 * Dispatch a GroupNorm residual block (used in texture/geometry heads).
 *
 * Structure: GN(inC)→ReLU→Conv3x3(inC→hidden)→GN(hidden)→ReLU→Conv3x3(hidden→outC) + skip
 * Weight indices: .0=GN, .1=ReLU(no w), .2=Conv, .3=GN, .4=ReLU(no w), .5=Conv
 */
async function dispatchGroupNormResidualBlock(device, inputBuf, inC, outC, hiddenC, H, W, prefix, raw, numGroups, boundaryYield, label, decoderKernelChunkItems, decoderAdaptiveDuty) {
  let conv1Input = inputBuf;
  let prepareConv1Input = null;
  if (decoderKernelChunkItems > 0) {
    const normalized = await dispatchTiledGroupNormRelu({
      device,
      inputBuf,
      scaleBuf: raw.get(`${prefix}.residual.0.weight`),
      biasBuf: raw.get(`${prefix}.residual.0.bias`),
      params: { C: inC, H, W, numGroups, eps: 1e-5 },
      chunkItems: decoderKernelChunkItems,
      adaptiveDuty: decoderAdaptiveDuty,
      phase: 'head-gn1',
      details: { label, inC, outC, hiddenC, H, W },
      boundaryYield,
    });
    conv1Input = normalized.buffer;
  } else {
    prepareConv1Input = encoder => {
      const gn1 = dispatchGroupNorm(device, encoder, inputBuf,
        raw.get(`${prefix}.residual.0.weight`), raw.get(`${prefix}.residual.0.bias`),
        { C: inC, H, W, numGroups, eps: 1e-5 });
      return dispatchActivation(device, encoder, gn1, null, inC * H * W, 0);
    };
  }
  const conv1 = await dispatchTiledConv2d({
    device,
    inputBuf: conv1Input,
    weightBuf: raw.get(`${prefix}.residual.2.weight`),
    biasBuf: raw.get(`${prefix}.residual.2.bias`),
    params: { inC, inH: H, inW: W, outC: hiddenC, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 },
    chunkItems: decoderKernelChunkItems,
    adaptiveDuty: decoderAdaptiveDuty,
    phase: 'head-gn-conv1',
    details: { label, inC, outC, hiddenC, H, W },
    boundaryYield,
    prepareInput: prepareConv1Input,
  });
  let conv2Input = conv1.buffer;
  let prepareConv2Input = null;
  if (decoderKernelChunkItems > 0) {
    const normalized = await dispatchTiledGroupNormRelu({
      device,
      inputBuf: conv1.buffer,
      scaleBuf: raw.get(`${prefix}.residual.3.weight`),
      biasBuf: raw.get(`${prefix}.residual.3.bias`),
      params: { C: hiddenC, H, W, numGroups: Math.min(numGroups, hiddenC), eps: 1e-5 },
      chunkItems: decoderKernelChunkItems,
      adaptiveDuty: decoderAdaptiveDuty,
      phase: 'head-gn2',
      details: { label, inC, outC, hiddenC, H, W },
      boundaryYield,
    });
    conv2Input = normalized.buffer;
  } else {
    prepareConv2Input = encoder => {
      const gn2 = dispatchGroupNorm(device, encoder, conv1.buffer,
        raw.get(`${prefix}.residual.3.weight`), raw.get(`${prefix}.residual.3.bias`),
        { C: hiddenC, H, W, numGroups: Math.min(numGroups, hiddenC), eps: 1e-5 });
      return dispatchActivation(device, encoder, gn2, null, hiddenC * H * W, 0);
    };
  }
  const conv2 = await dispatchTiledConv2d({
    device,
    inputBuf: conv2Input,
    weightBuf: raw.get(`${prefix}.residual.5.weight`),
    biasBuf: raw.get(`${prefix}.residual.5.bias`),
    params: { inC: hiddenC, inH: H, inW: W, outC, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 },
    chunkItems: decoderKernelChunkItems,
    adaptiveDuty: decoderAdaptiveDuty,
    phase: 'head-gn-conv2',
    details: { label, inC, outC, hiddenC, H, W },
    boundaryYield,
    prepareInput: prepareConv2Input,
  });

  // Skip connection (identity if inC == outC)
  if (inC === outC) {
    const addEnc = device.createCommandEncoder();
    const sumBuf = dispatchActivation(device, addEnc, inputBuf, conv2.buffer, outC * H * W, 2);
    device.queue.submit([addEnc.finish()]);
    await boundaryYield('residual-skip-add', { label, C: outC, H, W });
    return sumBuf;
  }
  return conv2.buffer;
}

/**
 * Dispatch a texture/geometry head.
 * Structure: GNResBlock → GNResBlock → ReLU → Conv1x1(128→32) → ReLU
 */
async function dispatchHead(device, inputBuf, C, H, W, prefix, raw, numGroups, boundaryYield, label, decoderKernelChunkItems, decoderAdaptiveDuty) {
  // Block 0: residual with GN
  let features = await dispatchGroupNormResidualBlock(
    device,
    inputBuf,
    C,
    C,
    C / 2,
    H,
    W,
    `${prefix}.0`,
    raw,
    numGroups,
    boundaryYield,
    `${label}.0`,
    decoderKernelChunkItems,
    decoderAdaptiveDuty
  );
  // Block 1: residual with GN
  features = await dispatchGroupNormResidualBlock(
    device,
    features,
    C,
    C,
    C / 2,
    H,
    W,
    `${prefix}.1`,
    raw,
    numGroups,
    boundaryYield,
    `${label}.1`,
    decoderKernelChunkItems,
    decoderAdaptiveDuty
  );

  let enc = device.createCommandEncoder();
  const relu = dispatchActivation(device, enc, features, null, C * H * W, 0);
  const reluSubmittedAtMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
  device.queue.submit([enc.finish()]);
  await boundaryYield('head-preprojection-relu', { label, C, H, W, commandSubmittedAtMs: reluSubmittedAtMs });

  const conv = await dispatchTiledConv1x1({
    device,
    inputBuf: relu,
    weightBuf: raw.get(`${prefix}.3.weight`),
    biasBuf: raw.get(`${prefix}.3.bias`),
    params: { inC: C, outC: 32, H, W },
    chunkItems: decoderKernelChunkItems,
    adaptiveDuty: decoderAdaptiveDuty,
    phase: 'head-projection',
    details: { label, C, H, W, outC: 32 },
    boundaryYield,
  });

  enc = device.createCommandEncoder();
  const out = dispatchActivation(device, enc, conv.buffer, null, 32 * H * W, 0);
  const finalSubmittedAtMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
  device.queue.submit([enc.finish()]);
  await boundaryYield('head-final', { label, C, H, W, commandSubmittedAtMs: finalSubmittedAtMs });

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
  async run(spnFeatures, spnDims, disparityBuf, dispH, dispW, chwImage, weights, options = {}) {
    const device = this.device;
    const scheduler = options.scheduler || null;
    const telemetry = options.telemetry || null;
    const gaussianPhaseYieldMs = scheduler?.effective?.gaussianPhaseYieldMs ?? scheduler?.effective?.yieldMs ?? 0;
    const gaussianPhaseYield = (phase, details = {}, queueCompletionFence = null) => scheduler
      ? schedulerYield(
          scheduler,
          device,
          telemetry,
          'gaussian-phase',
          { phase, ...details },
          gaussianPhaseYieldMs,
          queueCompletionFence,
        )
      : breathe();
    const decoderKernelChunkItems = scheduler?.effective?.decoderKernelChunkItems || 0;
    const decoderAdaptiveDuty = createDecoderAdaptiveDuty(scheduler, telemetry, 'gaussian');
    const raw = weights.raw;
    const fmPrefix = 'feature_model';
    const phPrefix = 'prediction_head';
    const decoderDim = 128;
    const numGroups = 8;
    const numLayers = 2; // Gaussian layers
    const stride = 2;    // Output stride

    console.log('[Gaussian] Running initializer...');

    // --- Step 1: Initializer (pure math, no weights) ---
    const imgSize = dispH; // 1536

    // Create feature_input: cat(image[0,1], normalized_disparity) → [5, H, W], then 2*x - 1
    //
    // Reference flow (initializer.py prepare_feature_input):
    //   1. depth = disparity_factor / clamp(disparity, 1e-4, 1e4)
    //   2. _rescale_depth: depth *= 1/(min(depth)+1e-6), clamp to 100
    //   3. normalized_disparity = disparity_factor / rescaled_depth = 1.0 / rescaled_depth
    //   4. feature_input = cat(image, normalized_disparity)
    //   5. feature_input = 2 * feature_input - 1
    // The first three channels become the original normalized CHW image after
    // the reference [0,1] -> [-1,1] roundtrip, so only the image upload remains.
    let enc = device.createCommandEncoder();
    const imageInputBuf = createStorageBuffer(device, chwImage);
    const featureInput = dispatchGaussianInitializerFeatureInput(device, enc, imageInputBuf, disparityBuf,
      { H: imgSize, W: imgSize });
    device.queue.submit([enc.finish()]);
    console.log(`[Gaussian]   Initializer feature_input: [${featureInput.C}, ${featureInput.H}, ${featureInput.W}]`);
    imageInputBuf.destroy();
    for (const scratch of featureInput.scratchBuffers) scratch.destroy();
    await gaussianPhaseYield('initializer-feature-input', { dispH, dispW, inputChannels: 5, H: imgSize, W: imgSize });

    // --- Step 2: Gaussian decoder (feature_model) ---
    console.log('[Gaussian] Running decoder (MultiresConvDecoder)...');

    // The Gaussian decoder's MultiresConvDecoder uses the SAME SPN features
    // as the monodepth decoder, just with 128-dim instead of 256-dim.
    // dims_encoder = [256, 256, 512, 1024, 1024], dims_decoder = [128]*5
    const projected = [];

    // convs[0]: Conv2d(256→128, 1x1, bias=false)
    const conv0 = await dispatchTiledConv1x1({
      device,
      inputBuf: spnFeatures[0],
      weightBuf: raw.get(`${fmPrefix}.decoder.convs.0.weight`),
      biasBuf: null,
      params: { inC: spnDims[0].C, outC: decoderDim, H: spnDims[0].H, W: spnDims[0].W },
      chunkItems: decoderKernelChunkItems,
      adaptiveDuty: decoderAdaptiveDuty,
      phase: 'project-feature',
      details: {
        index: 0,
        inC: spnDims[0].C,
        outC: decoderDim,
        H: spnDims[0].H,
        W: spnDims[0].W,
      },
      boundaryYield: gaussianPhaseYield,
    });
    projected[0] = { buffer: conv0.buffer, C: decoderDim, H: spnDims[0].H, W: spnDims[0].W };

    // convs[1-4]: Conv2d(inC→128, 3x3, bias=false)
    for (let i = 1; i <= 4; i++) {
      const result = await dispatchTiledConv2d({
        device,
        inputBuf: spnFeatures[i],
        weightBuf: raw.get(`${fmPrefix}.decoder.convs.${i}.weight`),
        biasBuf: null,
        params: { inC: spnDims[i].C, inH: spnDims[i].H, inW: spnDims[i].W,
          outC: decoderDim, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 },
        chunkItems: decoderKernelChunkItems,
        adaptiveDuty: decoderAdaptiveDuty,
        phase: 'project-feature',
        details: { index: i, inC: spnDims[i].C, outC: decoderDim, H: spnDims[i].H, W: spnDims[i].W },
        boundaryYield: gaussianPhaseYield,
      });
      projected[i] = { buffer: result.buffer, C: decoderDim, H: result.outH, W: result.outW };
    }
    await gaussianPhaseYield('project-features', { count: projected.length });

    // Fuse from lowest to highest resolution
    let features = await dispatchFusionBlock(device,
      projected[4].buffer, null, decoderDim, projected[4].H, projected[4].W,
      `${fmPrefix}.decoder.fusions.4`, raw, true, gaussianPhaseYield, 'decoder.fusion.4', decoderKernelChunkItems, decoderAdaptiveDuty);
    await gaussianPhaseYield('decoder-fusion', { index: 4, H: features.H, W: features.W });

    for (let i = 3; i >= 0; i--) {
      features = await dispatchFusionBlock(device,
        features.buffer, projected[i].buffer,
        decoderDim, features.H, features.W,
        `${fmPrefix}.decoder.fusions.${i}`, raw, i > 0, gaussianPhaseYield, `decoder.fusion.${i}`, decoderKernelChunkItems, decoderAdaptiveDuty);
      await gaussianPhaseYield('decoder-fusion', { index: i, H: features.H, W: features.W });
    }

    console.log(`[Gaussian]   Decoder output: [${decoderDim}, ${features.H}, ${features.W}]`);

    // upsample = identity (stride_out = 2)

    // --- Step 3: Image encoder (SkipConvBackbone) ---
    // Conv2d(5→128, kernel_size=2, stride=2, bias=true)
    const skipResult = await dispatchTiledConv2d({
      device,
      inputBuf: featureInput.buffer,
      weightBuf: raw.get(`${fmPrefix}.image_encoder.conv.weight`),
      biasBuf: raw.get(`${fmPrefix}.image_encoder.conv.bias`),
      params: { inC: 5, inH: imgSize, inW: imgSize, outC: decoderDim,
        kH: 2, kW: 2, padH: 0, padW: 0, strideH: 2, strideW: 2 },
      chunkItems: decoderKernelChunkItems,
      adaptiveDuty: decoderAdaptiveDuty,
      phase: 'image-encoder',
      details: { inputChannels: 5, H: imgSize, W: imgSize },
      boundaryYield: gaussianPhaseYield,
    });
    console.log(`[Gaussian]   Image encoder: [5, ${imgSize}, ${imgSize}] → [${decoderDim}, ${skipResult.outH}, ${skipResult.outW}]`);
    featureInput.buffer.destroy();

    // --- Step 4: Fusion block (decoder + skip) ---
    const fused = await dispatchFusionBlock(device,
      features.buffer, skipResult.buffer,
      decoderDim, features.H, features.W,
      `${fmPrefix}.fusion`, raw, false, gaussianPhaseYield, 'skip-fusion', decoderKernelChunkItems, decoderAdaptiveDuty);
    console.log(`[Gaussian]   Fusion: [${decoderDim}, ${fused.H}, ${fused.W}]`);
    await gaussianPhaseYield('skip-fusion', { H: fused.H, W: fused.W });

    // --- Step 5: Texture and geometry heads ---
    console.log('[Gaussian] Running texture/geometry heads...');
    const textureFeatures = await dispatchHead(
      device,
      fused.buffer,
      decoderDim,
      fused.H,
      fused.W,
      `${fmPrefix}.texture_head`,
      raw,
      numGroups,
      gaussianPhaseYield,
      'texture-head',
      decoderKernelChunkItems,
      decoderAdaptiveDuty
    );
    await gaussianPhaseYield('texture-head', { H: fused.H, W: fused.W });

    const geometryFeatures = await dispatchHead(
      device,
      fused.buffer,
      decoderDim,
      fused.H,
      fused.W,
      `${fmPrefix}.geometry_head`,
      raw,
      numGroups,
      gaussianPhaseYield,
      'geometry-head',
      decoderKernelChunkItems,
      decoderAdaptiveDuty
    );
    await gaussianPhaseYield('geometry-head', { H: fused.H, W: fused.W });

    console.log(`[Gaussian]   Heads output: texture=[32, ${fused.H}, ${fused.W}] geometry=[32, ${fused.H}, ${fused.W}]`);

    // --- Step 6: Prediction head (DirectPredictionHead) ---
    // geometry: Conv2d(32, 3*numLayers=6, 1x1)
    // texture: Conv2d(32, 11*numLayers=22, 1x1)
    const geomDeltas = await dispatchTiledConv1x1({
      device,
      inputBuf: geometryFeatures,
      weightBuf: raw.get(`${phPrefix}.geometry_prediction_head.weight`),
      biasBuf: raw.get(`${phPrefix}.geometry_prediction_head.bias`),
      params: { inC: 32, outC: 3 * numLayers, H: fused.H, W: fused.W },
      chunkItems: decoderKernelChunkItems,
      adaptiveDuty: decoderAdaptiveDuty,
      phase: 'prediction-geometry',
      details: { H: fused.H, W: fused.W, outC: 3 * numLayers },
      boundaryYield: gaussianPhaseYield,
    });

    const texDeltas = await dispatchTiledConv1x1({
      device,
      inputBuf: textureFeatures,
      weightBuf: raw.get(`${phPrefix}.texture_prediction_head.weight`),
      biasBuf: raw.get(`${phPrefix}.texture_prediction_head.bias`),
      params: { inC: 32, outC: 11 * numLayers, H: fused.H, W: fused.W },
      chunkItems: decoderKernelChunkItems,
      adaptiveDuty: decoderAdaptiveDuty,
      phase: 'prediction-texture',
      details: { H: fused.H, W: fused.W, outC: 11 * numLayers },
      boundaryYield: gaussianPhaseYield,
    });

    console.log(`[Gaussian]   Prediction head: geometry=[${3 * numLayers}, ${fused.H}, ${fused.W}] texture=[${11 * numLayers}, ${fused.H}, ${fused.W}]`);

    // Store delta buffers for downstream compose step
    this._geomDeltasBuf = geomDeltas.buffer;
    this._texDeltasBuf = texDeltas.buffer;

    const outH = fused.H, outW = fused.W;
    const numGaussians = numLayers * outH * outW;

    console.log(`[Gaussian] Output: ${numGaussians} Gaussians (${numLayers} layers × ${outH}×${outW})`);

    return {
      numGaussians,
      numLayers,
      H: outH,
      W: outW,
    };
  }
}
