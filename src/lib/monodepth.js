/**
 * monodepth.js — Monodepth decoder for SHARP-WebGPU.
 *
 * Takes 5 multi-resolution feature maps from SPN and produces a disparity map.
 *
 * Architecture (MultiresConvDecoder + head):
 *   1. Project each SPN output to decoder dim (256) via 1x1 or 3x3 conv
 *   2. Fuse features from lowest to highest resolution through FeatureFusionBlocks
 *   3. Each fusion block: optional resnet1(skip) + add, resnet2, deconv(2x), out_conv(1x1)
 *   4. Disparity head: conv3x3(256→128) → deconv2x → conv3x3(128→32) → ReLU → conv1x1(32→2) → ReLU
 *
 * Output: [2, 1536, 1536] disparity (2-layer depth)
 *
 * GPU pressure management: yields between fusion blocks to prevent system freeze.
 */

import { createStorageBuffer, createEmptyBuffer, readBuffer } from './gpu.js';
import {
  dispatchConv1x1,
  dispatchActivation,
} from './shader_ops.js';
import { dispatchTiledConv2d, dispatchTiledConvTranspose2d } from './decoder_duties.js';
import { schedulerYield } from './scheduler.js';

/** Yield to let the GPU/system breathe. */
const breathe = () => new Promise(r => setTimeout(r, 0));

/**
 * Dispatch a residual block: residual = ReLU→Conv3x3→ReLU→Conv3x3; out = input + residual
 *
 * Batches all 5 operations into a single command encoder submission
 * to reduce submit overhead while keeping them in one GPU batch.
 */
async function dispatchResidualBlock(device, inputBuf, C, H, W, prefix, raw, boundaryYield, label, decoderKernelChunkItems) {
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
    phase: 'residual-conv2',
    details: { label, C, H, W },
    boundaryYield,
    prepareInput: (encoder, source) => dispatchActivation(device, encoder, source, null, count, 0),
  });

  const enc = device.createCommandEncoder();
  // Skip connection: input + residual
  const sumBuf = dispatchActivation(device, enc, inputBuf, conv2.buffer, count, 2);
  device.queue.submit([enc.finish()]);
  await boundaryYield('residual-skip-add', { label, C, H, W });

  return sumBuf;
}

/**
 * Dispatch a FeatureFusionBlock2d.
 * Batches operations within the block, yields between blocks at the caller level.
 */
async function dispatchFusionBlock(device, x0Buf, x1Buf, C, H, W, prefix, raw, hasDeconv, boundaryYield, label, decoderKernelChunkItems) {
  let currentBuf = x0Buf;
  let currentH = H, currentW = W;

  // If x1 provided: resnet1(x1) + x0
  if (x1Buf) {
    const res1Buf = await dispatchResidualBlock(device, x1Buf, C, currentH, currentW, `${prefix}.resnet1`, raw, boundaryYield, `${label}.resnet1`, decoderKernelChunkItems);
    await boundaryYield('fusion-resnet1', {
      label,
      C,
      H: currentH,
      W: currentW,
      role: 'group-complete',
      waitBearingBoundaries: ['residual-conv1', 'residual-conv2', 'residual-skip-add'],
    });
    const enc = device.createCommandEncoder();
    currentBuf = dispatchActivation(device, enc, currentBuf, res1Buf, C * currentH * currentW, 2);
    device.queue.submit([enc.finish()]);
    await boundaryYield('fusion-skip-add', { label, C, H: currentH, W: currentW });
  }

  // resnet2
  currentBuf = await dispatchResidualBlock(device, currentBuf, C, currentH, currentW, `${prefix}.resnet2`, raw, boundaryYield, `${label}.resnet2`, decoderKernelChunkItems);
  await boundaryYield('fusion-resnet2', {
    label,
    C,
    H: currentH,
    W: currentW,
    role: 'group-complete',
    waitBearingBoundaries: ['residual-conv1', 'residual-conv2', 'residual-skip-add'],
  });

  // deconv (2x upsample) + out_conv batched together
  if (hasDeconv) {
    const deconvResult = await dispatchTiledConvTranspose2d({
      device,
      inputBuf: currentBuf,
      weightBuf: raw.get(`${prefix}.deconv.weight`),
      biasBuf: null,
      params: { inC: C, inH: currentH, inW: currentW, outC: C, stride: 2 },
      chunkItems: decoderKernelChunkItems,
      phase: 'fusion-deconv',
      details: { label, C, H: currentH * 2, W: currentW * 2 },
      boundaryYield,
    });
    currentBuf = deconvResult.buffer;
    currentH *= 2;
    currentW *= 2;
  }

  const enc = device.createCommandEncoder();
  const outResult = dispatchConv1x1(device, enc, currentBuf,
    raw.get(`${prefix}.out_conv.weight`),
    raw.get(`${prefix}.out_conv.bias`),
    { inC: C, outC: C, H: currentH, W: currentW });
  device.queue.submit([enc.finish()]);
  await boundaryYield('fusion-out-conv', { label, C, H: currentH, W: currentW, hasDeconv });

  return { buffer: outResult.buffer, H: currentH, W: currentW };
}

export class MonodepthDecoder {
  constructor(device) {
    this.device = device;
  }

  /**
   * Run the monodepth decoder.
   * @param {GPUBuffer[]} spnFeatures - 5 feature map buffers from SPN
   * @param {{C,H,W}[]} spnDims - dimensions of each feature map
   * @param {Object} weights - weights object with .raw accessor
   * @returns {Promise<{ disparityBuf: GPUBuffer, H: number, W: number, C: number }>}
   */
  async run(spnFeatures, spnDims, weights, options = {}) {
    const device = this.device;
    const scheduler = options.scheduler || null;
    const telemetry = options.telemetry || null;
    const monodepthPhaseYield = (phase, details = {}) => scheduler
      ? schedulerYield(scheduler, device, telemetry, 'monodepth-phase', { phase, ...details })
      : breathe();
    const decoderKernelChunkItems = scheduler?.effective?.decoderKernelChunkItems || 0;
    const raw = weights.raw;
    const prefix = 'monodepth_model.monodepth_predictor';
    const decoderDim = 256;

    console.log('[Monodepth] Running decoder...');

    // Step 1: Project SPN features to decoder dim
    const projected = [];
    projected[0] = { buffer: spnFeatures[0], C: decoderDim, H: spnDims[0].H, W: spnDims[0].W };

    for (let i = 1; i <= 4; i++) {
      const result = await dispatchTiledConv2d({
        device,
        inputBuf: spnFeatures[i],
        weightBuf: raw.get(`${prefix}.decoder.convs.${i}.weight`),
        biasBuf: null,
        params: { inC: spnDims[i].C, inH: spnDims[i].H, inW: spnDims[i].W,
          outC: decoderDim, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 },
        chunkItems: decoderKernelChunkItems,
        phase: 'project-feature',
        details: { index: i, inC: spnDims[i].C, outC: decoderDim, H: spnDims[i].H, W: spnDims[i].W },
        boundaryYield: monodepthPhaseYield,
      });
      projected[i] = { buffer: result.buffer, C: decoderDim, H: result.outH, W: result.outW };
      console.log(`[Monodepth]   convs[${i}]: [${spnDims[i].C},${spnDims[i].H},${spnDims[i].W}] → [${decoderDim},${result.outH},${result.outW}]`);
    }

    // Step 2: Fuse from lowest to highest resolution
    let features = await dispatchFusionBlock(device,
      projected[4].buffer, null,
      decoderDim, projected[4].H, projected[4].W,
      `${prefix}.decoder.fusions.4`, raw, true, monodepthPhaseYield, 'decoder.fusions.4', decoderKernelChunkItems);
    console.log(`[Monodepth]   fusions[4]: → [${decoderDim},${features.H},${features.W}]`);

    for (let i = 3; i >= 0; i--) {
      const hasDeconv = i > 0;
      features = await dispatchFusionBlock(device,
        features.buffer, projected[i].buffer,
        decoderDim, features.H, features.W,
        `${prefix}.decoder.fusions.${i}`, raw, hasDeconv, monodepthPhaseYield, `decoder.fusions.${i}`, decoderKernelChunkItems);
      console.log(`[Monodepth]   fusions[${i}]: → [${decoderDim},${features.H},${features.W}]`);
    }

    // Step 3: Disparity head
    console.log('[Monodepth] Running disparity head...');

    // Batch the head ops: conv3x3 → deconv → conv3x3 → relu → conv1x1 → relu
    // head.0: Conv2d(256→128, 3x3, pad=1)
    const head0 = await dispatchTiledConv2d({
      device,
      inputBuf: features.buffer,
      weightBuf: raw.get(`${prefix}.head.0.weight`),
      biasBuf: raw.get(`${prefix}.head.0.bias`),
      params: { inC: 256, inH: features.H, inW: features.W,
        outC: 128, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 },
      chunkItems: decoderKernelChunkItems,
      phase: 'head-conv0',
      details: { H: features.H, W: features.W, inC: 256, outC: 128 },
      boundaryYield: monodepthPhaseYield,
    });

    // head.1: ConvTranspose2d(128→128, 2x2, stride=2, bias=true)
    const head1 = await dispatchTiledConvTranspose2d({
      device,
      inputBuf: head0.buffer,
      weightBuf: raw.get(`${prefix}.head.1.weight`),
      biasBuf: raw.get(`${prefix}.head.1.bias`),
      params: { inC: 128, inH: head0.outH, inW: head0.outW, outC: 128, stride: 2 },
      chunkItems: decoderKernelChunkItems,
      phase: 'head-deconv',
      details: { H: head0.outH * 2, W: head0.outW * 2, inC: 128, outC: 128 },
      boundaryYield: monodepthPhaseYield,
    });

    // head.2: Conv2d(128→32, 3x3, pad=1)
    const head2 = await dispatchTiledConv2d({
      device,
      inputBuf: head1.buffer,
      weightBuf: raw.get(`${prefix}.head.2.weight`),
      biasBuf: raw.get(`${prefix}.head.2.bias`),
      params: { inC: 128, inH: head1.H, inW: head1.W,
        outC: 32, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 },
      chunkItems: decoderKernelChunkItems,
      phase: 'head-conv2',
      details: { H: head1.H, W: head1.W, inC: 128, outC: 32 },
      boundaryYield: monodepthPhaseYield,
    });

    let enc = device.createCommandEncoder();
    // head.3: ReLU
    const head3 = dispatchActivation(device, enc, head2.buffer, null, 32 * head2.outH * head2.outW, 0);
    device.queue.submit([enc.finish()]);
    await monodepthPhaseYield('head-relu3', { H: head2.outH, W: head2.outW, C: 32 });

    enc = device.createCommandEncoder();
    // head.4: Conv2d(32→2, 1x1)
    const head4 = dispatchConv1x1(device, enc, head3,
      raw.get(`${prefix}.head.4.weight`),
      raw.get(`${prefix}.head.4.bias`),
      { inC: 32, outC: 2, H: head2.outH, W: head2.outW });
    device.queue.submit([enc.finish()]);
    await monodepthPhaseYield('head-conv4', { H: head4.H, W: head4.W, inC: 32, outC: 2 });

    enc = device.createCommandEncoder();
    // head.5: ReLU
    const disparityBuf = dispatchActivation(device, enc, head4.buffer, null, 2 * head4.H * head4.W, 0);
    device.queue.submit([enc.finish()]);
    await monodepthPhaseYield('head-final', { H: head4.H, W: head4.W, outC: 2 });

    const outH = head4.H, outW = head4.W;
    console.log(`[Monodepth] Output disparity: [2, ${outH}, ${outW}]`);

    return { disparityBuf, H: outH, W: outW, C: 2 };
  }
}
