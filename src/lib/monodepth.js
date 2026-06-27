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
  dispatchConv2d,
  dispatchConv1x1,
  dispatchConvTranspose2d,
  dispatchActivation,
} from './shader_ops.js';

/** Yield to let the GPU/system breathe. */
const breathe = () => new Promise(r => setTimeout(r, 0));

/**
 * Dispatch a residual block: residual = ReLU→Conv3x3→ReLU→Conv3x3; out = input + residual
 *
 * Batches all 5 operations into a single command encoder submission
 * to reduce submit overhead while keeping them in one GPU batch.
 */
function dispatchResidualBlock(device, inputBuf, C, H, W, prefix, raw) {
  const count = C * H * W;
  const enc = device.createCommandEncoder();

  // ReLU → Conv3x3 (first)
  const relu1 = dispatchActivation(device, enc, inputBuf, null, count, 0);
  const conv1 = dispatchConv2d(device, enc, relu1,
    raw.get(`${prefix}.residual.1.weight`),
    raw.get(`${prefix}.residual.1.bias`),
    { inC: C, inH: H, inW: W, outC: C, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });

  // ReLU → Conv3x3 (second)
  const relu2 = dispatchActivation(device, enc, conv1.buffer, null, count, 0);
  const conv2 = dispatchConv2d(device, enc, relu2,
    raw.get(`${prefix}.residual.3.weight`),
    raw.get(`${prefix}.residual.3.bias`),
    { inC: C, inH: H, inW: W, outC: C, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });

  // Skip connection: input + residual
  const sumBuf = dispatchActivation(device, enc, inputBuf, conv2.buffer, count, 2);

  device.queue.submit([enc.finish()]);
  return sumBuf;
}

/**
 * Dispatch a FeatureFusionBlock2d.
 * Batches operations within the block, yields between blocks at the caller level.
 */
function dispatchFusionBlock(device, x0Buf, x1Buf, C, H, W, prefix, raw, hasDeconv) {
  let currentBuf = x0Buf;
  let currentH = H, currentW = W;

  // If x1 provided: resnet1(x1) + x0
  if (x1Buf) {
    const res1Buf = dispatchResidualBlock(device, x1Buf, C, currentH, currentW, `${prefix}.resnet1`, raw);
    const enc = device.createCommandEncoder();
    currentBuf = dispatchActivation(device, enc, currentBuf, res1Buf, C * currentH * currentW, 2);
    device.queue.submit([enc.finish()]);
  }

  // resnet2
  currentBuf = dispatchResidualBlock(device, currentBuf, C, currentH, currentW, `${prefix}.resnet2`, raw);

  // deconv (2x upsample) + out_conv batched together
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
    raw.get(`${prefix}.out_conv.weight`),
    raw.get(`${prefix}.out_conv.bias`),
    { inC: C, outC: C, H: currentH, W: currentW });
  device.queue.submit([enc.finish()]);

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
  async run(spnFeatures, spnDims, weights) {
    const device = this.device;
    const raw = weights.raw;
    const prefix = 'monodepth_model.monodepth_predictor';
    const decoderDim = 256;

    console.log('[Monodepth] Running decoder...');

    // Step 1: Project SPN features to decoder dim
    const projected = [];
    projected[0] = { buffer: spnFeatures[0], C: decoderDim, H: spnDims[0].H, W: spnDims[0].W };

    for (let i = 1; i <= 4; i++) {
      const enc = device.createCommandEncoder();
      const result = dispatchConv2d(device, enc, spnFeatures[i],
        raw.get(`${prefix}.decoder.convs.${i}.weight`), null,
        { inC: spnDims[i].C, inH: spnDims[i].H, inW: spnDims[i].W,
          outC: decoderDim, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });
      device.queue.submit([enc.finish()]);
      projected[i] = { buffer: result.buffer, C: decoderDim, H: result.outH, W: result.outW };
      console.log(`[Monodepth]   convs[${i}]: [${spnDims[i].C},${spnDims[i].H},${spnDims[i].W}] → [${decoderDim},${result.outH},${result.outW}]`);
    }

    await breathe();

    // Step 2: Fuse from lowest to highest resolution
    let features = dispatchFusionBlock(device,
      projected[4].buffer, null,
      decoderDim, projected[4].H, projected[4].W,
      `${prefix}.decoder.fusions.4`, raw, true);
    console.log(`[Monodepth]   fusions[4]: → [${decoderDim},${features.H},${features.W}]`);
    await breathe();

    for (let i = 3; i >= 0; i--) {
      const hasDeconv = i > 0;
      features = dispatchFusionBlock(device,
        features.buffer, projected[i].buffer,
        decoderDim, features.H, features.W,
        `${prefix}.decoder.fusions.${i}`, raw, hasDeconv);
      console.log(`[Monodepth]   fusions[${i}]: → [${decoderDim},${features.H},${features.W}]`);

      // Yield between fusion blocks — critical for high-res levels
      await breathe();
    }

    // Step 3: Disparity head
    console.log('[Monodepth] Running disparity head...');

    // Batch the head ops: conv3x3 → deconv → conv3x3 → relu → conv1x1 → relu
    let enc = device.createCommandEncoder();

    // head.0: Conv2d(256→128, 3x3, pad=1)
    const head0 = dispatchConv2d(device, enc, features.buffer,
      raw.get(`${prefix}.head.0.weight`),
      raw.get(`${prefix}.head.0.bias`),
      { inC: 256, inH: features.H, inW: features.W,
        outC: 128, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });

    device.queue.submit([enc.finish()]);
    await breathe();

    // head.1: ConvTranspose2d(128→128, 2x2, stride=2, bias=true)
    enc = device.createCommandEncoder();
    const head1 = dispatchConvTranspose2d(device, enc, head0.buffer,
      raw.get(`${prefix}.head.1.weight`),
      raw.get(`${prefix}.head.1.bias`),
      { inC: 128, inH: head0.outH, inW: head0.outW, outC: 128, stride: 2 });

    // head.2: Conv2d(128→32, 3x3, pad=1)
    const head2 = dispatchConv2d(device, enc, head1.buffer,
      raw.get(`${prefix}.head.2.weight`),
      raw.get(`${prefix}.head.2.bias`),
      { inC: 128, inH: head1.H, inW: head1.W,
        outC: 32, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });

    // head.3: ReLU
    const head3 = dispatchActivation(device, enc, head2.buffer, null, 32 * head2.outH * head2.outW, 0);

    // head.4: Conv2d(32→2, 1x1)
    const head4 = dispatchConv1x1(device, enc, head3,
      raw.get(`${prefix}.head.4.weight`),
      raw.get(`${prefix}.head.4.bias`),
      { inC: 32, outC: 2, H: head2.outH, W: head2.outW });

    // head.5: ReLU
    const disparityBuf = dispatchActivation(device, enc, head4.buffer, null, 2 * head4.H * head4.W, 0);

    device.queue.submit([enc.finish()]);

    const outH = head4.H, outW = head4.W;
    console.log(`[Monodepth] Output disparity: [2, ${outH}, ${outW}]`);

    return { disparityBuf, H: outH, W: outW, C: 2 };
  }
}
