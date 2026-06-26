/**
 * backbone.js — DINOv2 ViT-Large backbone dispatch for SHARP.
 *
 * SHARP uses two ViT-Large encoders inside its SlidingPyramidNetwork:
 *   - patch_encoder: processes high-res patches (patch_size=16, img_size=384)
 *   - image_encoder: processes the low-res full image (patch_size=16, img_size=384)
 *
 * Both share the same architecture (DINOv2 ViT-Large, dinov2l16_384):
 *   embed_dim=1024, depth=24, num_heads=16, patch_size=16
 *   intermediate_features_ids=[5, 11, 17, 23]
 *
 * This module provides a single ViT encoder that can run against either weight
 * set. The SPN module will orchestrate calling it for patches vs full image.
 *
 * For the initial backbone smoke, we run just the patch encoder on a 384x384 image.
 */

import { createStorageBuffer, createEmptyBuffer, readBuffer } from './gpu.js';

import patchEmbedWGSL from '../shaders/patch_embed_dinov2.wgsl?raw';
import layerNormWGSL from '../shaders/layernorm_vit.wgsl?raw';
import attentionWGSL from '../shaders/attention.wgsl?raw';
import linearWGSL from '../shaders/linear.wgsl?raw';
import linearGeluWGSL from '../shaders/linear_gelu.wgsl?raw';
import layerscaleWGSL from '../shaders/layerscale.wgsl?raw';
import transposeWGSL from '../shaders/transpose_nd.wgsl?raw';

const MAX_WG = 65535;
function splitWG(total) {
  if (total <= MAX_WG) return [total, 1];
  return [MAX_WG, Math.ceil(total / MAX_WG)];
}
function ceilDiv(a, b) { return Math.ceil(a / b); }

function makeUniform(device, data) {
  const buf = device.createBuffer({
    size: Math.max(data.byteLength, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer || data));
  buf.unmap();
  return buf;
}

function uniformKey(data) {
  const bytes = new Uint8Array(data.buffer || data);
  let h = 0;
  for (let i = 0; i < bytes.length; i++) h = (h * 31 + bytes[i]) | 0;
  return `u_${bytes.length}_${h}`;
}

// SHARP's ViT-Large config (dinov2l16_384)
const VIT_CONFIG = {
  dim: 1024,
  numHeads: 16,
  headDim: 64,
  numLayers: 24,
  patchSize: 16,
  channels: 3,
  intermediateLayers: [5, 11, 17, 23],
  mlpHiddenDim: 4096,
  scale: 1.0 / Math.sqrt(64),
  eps: 1e-6,
};

/**
 * A single DINOv2 ViT-Large encoder that can run against any weight set.
 * Used by the SPN to run both the patch encoder and image encoder.
 */
class ViTEncoder {
  constructor(device) {
    this.device = device;
    this.pipelines = {};
    this._uniformCache = new Map();
  }

  init() {
    const device = this.device;
    const make = (code, entry) => device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code }), entryPoint: entry },
    });

    this.pipelines.patchEmbed = make(patchEmbedWGSL, 'main');
    this.pipelines.layerNorm = make(layerNormWGSL, 'main');
    this.pipelines.attnScores = make(attentionWGSL, 'computeScores');
    this.pipelines.attnSoftmax = make(attentionWGSL, 'softmax');
    this.pipelines.attnApply = make(attentionWGSL, 'applyAttn');
    this.pipelines.linear = make(linearWGSL, 'main');
    this.pipelines.linearGelu = make(linearGeluWGSL, 'main');
    this.pipelines.layerScale = make(layerscaleWGSL, 'main');
    this.pipelines.transpose = make(transposeWGSL, 'main');

    // QKV split shader
    this.pipelines.splitQKV = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: `
            struct P { N: u32, D: u32, numWgX: u32 }
            @group(0) @binding(0) var<uniform> p: P;
            @group(0) @binding(1) var<storage, read> qkv: array<f32>;
            @group(0) @binding(2) var<storage, read_write> q: array<f32>;
            @group(0) @binding(3) var<storage, read_write> k: array<f32>;
            @group(0) @binding(4) var<storage, read_write> v: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
              let idx = (wgid.x + wgid.y * p.numWgX) * 256u + lid.x;
              if (idx >= p.N * p.D) { return; }
              let row = idx / p.D;
              let col = idx % p.D;
              let D3 = p.D * 3u;
              q[idx] = qkv[row * D3 + col];
              k[idx] = qkv[row * D3 + p.D + col];
              v[idx] = qkv[row * D3 + 2u * p.D + col];
            }
          `,
        }),
        entryPoint: 'main',
      },
    });

  }

  _cachedUniform(data) {
    const key = uniformKey(data);
    if (this._uniformCache.has(key)) return this._uniformCache.get(key);
    const buf = makeUniform(this.device, data);
    this._uniformCache.set(key, buf);
    return buf;
  }

  /**
   * Run ViT encoder on an image buffer.
   *
   * @param {GPUCommandEncoder} encoder
   * @param {GPUBuffer} imageBuf - [3, imgH, imgW] normalized to [-1, 1] CHW
   * @param {Object} vitWeights - { patchEmbed, posEmbed, clsToken, norm, blockWeights }
   * @param {number} tokenH - imgH / patchSize
   * @param {number} tokenW - imgW / patchSize
   * @returns {{
   *   finalTokensBuf: GPUBuffer,      // [N, D] post-final-norm tokens (CLS at index 0)
   *   intermediateFeatures: Array,     // pre-final-norm snapshots at configured layers
   *   tokenH, tokenW, numPatches, N
   * }}
   */
  encode(encoder, imageBuf, vitWeights, tokenH, tokenW) {
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const numPatches = tokenH * tokenW;
    const N = numPatches + 1;
    const T = N * D;

    // Pre-allocate work buffers (reused across calls for same grid size)
    this._ensureWorkBuffers(tokenH, tokenW);
    const wb = this._wb;

    // Destroy previous intermediate feature snapshots
    if (this._prevIntermediates) {
      for (const snap of this._prevIntermediates) snap.buffer.destroy();
    }
    if (this._prevFinalBuf) this._prevFinalBuf.destroy();

    // --- Patch embedding ---
    this._encodePatchEmbed(encoder, imageBuf, vitWeights, wb.tokenBufA, tokenH, tokenW);

    // --- Transformer blocks ---
    const intermediateFeatures = [];
    let currentTokens = wb.tokenBufA;

    for (let l = 0; l < VIT_CONFIG.numLayers; l++) {
      // LayerNorm1
      this._encodeLayerNorm(encoder, currentTokens, wb.normBuf, vitWeights, l, 'norm1', N);

      // Attention
      this._encodeQKV(encoder, wb.normBuf, wb.qBuf, wb.kBuf, wb.vBuf, vitWeights, l, N, wb.qkvWorkBuf);
      this._encodeAttnScores(encoder, wb.qBuf, wb.kBuf, wb.scoreBuf, N);
      this._encodeAttnSoftmax(encoder, wb.scoreBuf, N);
      this._encodeAttnApply(encoder, wb.scoreBuf, wb.vBuf, wb.attnOutBuf, N);
      this._encodeLinearByKey(encoder, wb.attnOutBuf, wb.projOutBuf, vitWeights, l, 'attn.proj', N, D, D);

      // LayerScale1 + residual
      const attnOut = (currentTokens === wb.tokenBufA) ? wb.tokenBufB : wb.tokenBufA;
      this._encodeLayerScaleResidual(encoder, wb.projOutBuf, currentTokens, attnOut, vitWeights, l, 'ls1', T, D);
      currentTokens = attnOut;

      // LayerNorm2
      this._encodeLayerNorm(encoder, currentTokens, wb.normBuf, vitWeights, l, 'norm2', N);

      // MLP
      this._encodeLinearGelu(encoder, wb.normBuf, wb.hiddenBuf, vitWeights, l, 'mlp.fc1', N, D, VIT_CONFIG.mlpHiddenDim);
      this._encodeLinearByKey(encoder, wb.hiddenBuf, wb.ffnOutBuf, vitWeights, l, 'mlp.fc2', N, VIT_CONFIG.mlpHiddenDim, D);

      // LayerScale2 + residual
      const ffnOut = (currentTokens === wb.tokenBufA) ? wb.tokenBufB : wb.tokenBufA;
      this._encodeLayerScaleResidual(encoder, wb.ffnOutBuf, currentTokens, ffnOut, vitWeights, l, 'ls2', T, D);
      currentTokens = ffnOut;

      // Capture intermediate features (pre-final-norm snapshots — downstream
      // consumers apply their own per-level processing on raw block output)
      if (VIT_CONFIG.intermediateLayers.includes(l)) {
        const snapBuf = createEmptyBuffer(device, T * 4, GPUBufferUsage.COPY_DST);
        encoder.copyBufferToBuffer(currentTokens, 0, snapBuf, 0, T * 4);
        intermediateFeatures.push({ buffer: snapBuf, layerIdx: l });
      }
    }

    // Final norm (applied to all tokens including CLS)
    const finalNormedBuf = createEmptyBuffer(device, T * 4);
    this._encodeLayerNormFinal(encoder, currentTokens, finalNormedBuf, vitWeights, N);

    // Track for cleanup on next call
    this._prevIntermediates = intermediateFeatures;
    this._prevFinalBuf = finalNormedBuf;

    return {
      finalTokensBuf: finalNormedBuf,
      intermediateFeatures,
      tokenH,
      tokenW,
      numPatches,
      N,
    };
  }

  _ensureWorkBuffers(tokenH, tokenW) {
    if (this._wb && this._wbTokenH === tokenH && this._wbTokenW === tokenW) return;
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const numPatches = tokenH * tokenW;
    const N = numPatches + 1;
    const T = N * D;

    // Destroy old buffers if grid size changed
    if (this._wb) {
      for (const buf of Object.values(this._wb)) buf.destroy();
    }

    this._wb = {
      tokenBufA: createEmptyBuffer(device, T * 4),
      tokenBufB: createEmptyBuffer(device, T * 4),
      normBuf: createEmptyBuffer(device, T * 4),
      qBuf: createEmptyBuffer(device, T * 4),
      kBuf: createEmptyBuffer(device, T * 4),
      vBuf: createEmptyBuffer(device, T * 4),
      scoreBuf: createEmptyBuffer(device, VIT_CONFIG.numHeads * N * N * 4),
      attnOutBuf: createEmptyBuffer(device, T * 4),
      projOutBuf: createEmptyBuffer(device, T * 4),
      hiddenBuf: createEmptyBuffer(device, N * VIT_CONFIG.mlpHiddenDim * 4),
      ffnOutBuf: createEmptyBuffer(device, T * 4),
      qkvWorkBuf: createEmptyBuffer(device, N * 3 * D * 4),
    };
    this._wbTokenH = tokenH;
    this._wbTokenW = tokenW;
  }

  // --- Private dispatch methods ---
  // Weight access uses vitWeights.blockWeights[`blocks.${l}.${suffix}`]

  _getBlockWeight(vitWeights, layerIdx, suffix) {
    return vitWeights.blockWeights[`blocks.${layerIdx}.${suffix}`];
  }

  _encodePatchEmbed(encoder, imageBuf, vitWeights, outputBuf, tokenH, tokenW) {
    const D = VIT_CONFIG.dim;
    const ps = VIT_CONFIG.patchSize;
    const numTokens = tokenH * tokenW + 1;
    const totalWG = ceilDiv(numTokens * D, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([tokenH * ps, tokenW * ps, ps, tokenH, tokenW, 3, D, numTokens, wgX]);
    const paramsBuf = this._cachedUniform(paramsData);

    const bg = this.device.createBindGroup({
      layout: this.pipelines.patchEmbed.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: imageBuf } },
        { binding: 2, resource: { buffer: vitWeights.patchEmbed.weight } },
        { binding: 3, resource: { buffer: vitWeights.patchEmbed.bias } },
        { binding: 4, resource: { buffer: vitWeights.clsToken } },
        { binding: 5, resource: { buffer: vitWeights.posEmbed } },
        { binding: 6, resource: { buffer: outputBuf } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.patchEmbed);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeLayerNorm(enc, input, output, vitWeights, layerIdx, normName, N) {
    const D = VIT_CONFIG.dim;
    const paramsData = new ArrayBuffer(16);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, D, true);
    v.setFloat32(8, VIT_CONFIG.eps, true);
    const paramsBuf = this._cachedUniform(new Uint8Array(paramsData));

    const gamma = this._getBlockWeight(vitWeights, layerIdx, `${normName}.weight`);
    const beta = this._getBlockWeight(vitWeights, layerIdx, `${normName}.bias`);
    if (!gamma || !beta) throw new Error(`Missing LayerNorm weights: blocks.${layerIdx}.${normName}`);

    const bg = this.device.createBindGroup({
      layout: this.pipelines.layerNorm.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: gamma } },
        { binding: 3, resource: { buffer: beta } },
        { binding: 4, resource: { buffer: output } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.layerNorm);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(N);
    pass.end();
  }

  _encodeLayerNormFinal(enc, input, output, vitWeights, N) {
    const D = VIT_CONFIG.dim;
    const paramsData = new ArrayBuffer(16);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, D, true);
    v.setFloat32(8, VIT_CONFIG.eps, true);
    const paramsBuf = this._cachedUniform(new Uint8Array(paramsData));

    const bg = this.device.createBindGroup({
      layout: this.pipelines.layerNorm.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: vitWeights.norm.weight } },
        { binding: 3, resource: { buffer: vitWeights.norm.bias } },
        { binding: 4, resource: { buffer: output } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.layerNorm);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(N);
    pass.end();
  }

  _encodeQKV(enc, input, qBuf, kBuf, vBuf, vitWeights, layerIdx, N, qkvWorkBuf) {
    const D = VIT_CONFIG.dim;
    const D3 = 3 * D;
    const qkvWeight = this._getBlockWeight(vitWeights, layerIdx, 'attn.qkv.weight');
    const qkvBias = this._getBlockWeight(vitWeights, layerIdx, 'attn.qkv.bias');
    if (!qkvWeight || !qkvBias) throw new Error(`Missing QKV weights: blocks.${layerIdx}.attn.qkv`);

    this._encodeLinearFull(enc, input, qkvWorkBuf, qkvWeight, qkvBias, N, D, D3);
    this._encodeSplitQKV(enc, qkvWorkBuf, qBuf, kBuf, vBuf, N, D);
  }

  _encodeLinearFull(enc, input, output, weight, bias, numRows, inDim, outDim) {
    const totalWG = ceilDiv(numRows * outDim, 256);
    const [wgX, wgY] = splitWG(totalWG);
    const paramsBuf = this._cachedUniform(new Uint32Array([numRows, inDim, outDim, wgX]));

    const bg = this.device.createBindGroup({
      layout: this.pipelines.linear.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: weight } },
        { binding: 3, resource: { buffer: bias } },
        { binding: 4, resource: { buffer: output } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.linear);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeSplitQKV(enc, qkvBuf, qBuf, kBuf, vBuf, N, D) {
    const total = N * D;
    const totalWG = ceilDiv(total, 256);
    const [wgX, wgY] = splitWG(totalWG);
    const paramsBuf = this._cachedUniform(new Uint32Array([N, D, wgX]));

    const bg = this.device.createBindGroup({
      layout: this.pipelines.splitQKV.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: qkvBuf } },
        { binding: 2, resource: { buffer: qBuf } },
        { binding: 3, resource: { buffer: kBuf } },
        { binding: 4, resource: { buffer: vBuf } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.splitQKV);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeLinearByKey(enc, input, output, vitWeights, layerIdx, suffix, numRows, inDim, outDim) {
    const weight = this._getBlockWeight(vitWeights, layerIdx, `${suffix}.weight`);
    const bias = this._getBlockWeight(vitWeights, layerIdx, `${suffix}.bias`);
    if (!weight || !bias) throw new Error(`Missing linear weights: blocks.${layerIdx}.${suffix}`);
    this._encodeLinearFull(enc, input, output, weight, bias, numRows, inDim, outDim);
  }

  _encodeAttnScores(enc, qBuf, kBuf, scoreBuf, N) {
    const { dim, numHeads, headDim, scale } = VIT_CONFIG;
    const total = numHeads * N * N;
    const totalWG = ceilDiv(total, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new ArrayBuffer(24);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, dim, true);
    v.setUint32(8, numHeads, true);
    v.setUint32(12, headDim, true);
    v.setFloat32(16, scale, true);
    v.setUint32(20, wgX, true);
    const paramsBuf = this._cachedUniform(new Uint8Array(paramsData));

    const bg = this.device.createBindGroup({
      layout: this.pipelines.attnScores.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: qBuf } },
        { binding: 2, resource: { buffer: kBuf } },
        { binding: 3, resource: { buffer: scoreBuf } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.attnScores);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeAttnSoftmax(enc, scoreBuf, N) {
    const totalRows = VIT_CONFIG.numHeads * N;
    const totalWG = ceilDiv(totalRows, 256);
    const [wgX, wgY] = splitWG(totalWG);
    const paramsBuf = this._cachedUniform(new Uint32Array([N, VIT_CONFIG.numHeads, wgX]));

    const bg = this.device.createBindGroup({
      layout: this.pipelines.attnSoftmax.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: scoreBuf } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.attnSoftmax);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeAttnApply(enc, scoreBuf, vBuf, output, N) {
    const D = VIT_CONFIG.dim;
    const totalWG = ceilDiv(N * D, 256);
    const [wgX, wgY] = splitWG(totalWG);
    const paramsBuf = this._cachedUniform(new Uint32Array([N, D, VIT_CONFIG.numHeads, VIT_CONFIG.headDim, wgX]));

    const bg = this.device.createBindGroup({
      layout: this.pipelines.attnApply.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: scoreBuf } },
        { binding: 2, resource: { buffer: vBuf } },
        { binding: 3, resource: { buffer: output } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.attnApply);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeLayerScaleResidual(enc, input, residual, output, vitWeights, layerIdx, lsName, count, D) {
    const totalWG = ceilDiv(count, 256);
    const [wgX, wgY] = splitWG(totalWG);
    const paramsBuf = this._cachedUniform(new Uint32Array([count, D, wgX]));

    const gamma = this._getBlockWeight(vitWeights, layerIdx, `${lsName}.gamma`);
    if (!gamma) throw new Error(`Missing LayerScale gamma: blocks.${layerIdx}.${lsName}`);

    const bg = this.device.createBindGroup({
      layout: this.pipelines.layerScale.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: gamma } },
        { binding: 3, resource: { buffer: residual } },
        { binding: 4, resource: { buffer: output } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.layerScale);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeLinearGelu(enc, input, output, vitWeights, layerIdx, suffix, numRows, inDim, outDim) {
    const weight = this._getBlockWeight(vitWeights, layerIdx, `${suffix}.weight`);
    const bias = this._getBlockWeight(vitWeights, layerIdx, `${suffix}.bias`);
    if (!weight || !bias) throw new Error(`Missing MLP weights: blocks.${layerIdx}.${suffix}`);

    const totalWG = ceilDiv(numRows * outDim, 256);
    const [wgX, wgY] = splitWG(totalWG);
    const paramsBuf = this._cachedUniform(new Uint32Array([numRows, inDim, outDim, wgX]));

    const bg = this.device.createBindGroup({
      layout: this.pipelines.linearGelu.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: weight } },
        { binding: 3, resource: { buffer: bias } },
        { binding: 4, resource: { buffer: output } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.linearGelu);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }
}

/**
 * SharpBackbone — high-level interface for the backbone smoke.
 * Runs a single ViT encoder (patch_encoder) on a resized 384x384 image.
 */
export class SharpBackbone {
  constructor(device) {
    this.device = device;
    this.vitEncoder = new ViTEncoder(device);
  }

  init(weights) {
    this.vitEncoder.init();
    this.weights = weights;

    // Remap weight keys for the ViT encoder interface.
    // Weight loader stores block weights as full state_dict keys like:
    //   monodepth_model.monodepth_predictor.encoder.patch_encoder.blocks.0.attn.qkv.weight
    // The ViT encoder expects:
    //   blocks.0.attn.qkv.weight
    const prefix = 'monodepth_model.monodepth_predictor.encoder.patch_encoder.';
    this._patchEncoderWeights = {
      patchEmbed: weights.patchEncoder.patchEmbed,
      posEmbed: weights.patchEncoder.posEmbed,
      clsToken: weights.patchEncoder.clsToken,
      norm: weights.patchEncoder.norm,
      blockWeights: {},
    };

    // Remap block weight keys: strip prefix
    for (const [key, buf] of Object.entries(weights.patchEncoder.blockWeights)) {
      const short = key.replace(prefix, '');
      this._patchEncoderWeights.blockWeights[short] = buf;
    }
  }

  /**
   * Run backbone on an image blob.
   * @param {Blob} blob - image blob (any format the browser can decode)
   * @returns {{ tokenH, tokenW, dim, numPatches, clsSample, hasNaN, finalTokensBuf, intermediateFeatures }}
   */
  async run(blob) {
    const device = this.device;

    // Resize to 384x384 for the backbone
    const targetSize = 384;
    const ps = VIT_CONFIG.patchSize;
    const tokenH = targetSize / ps;
    const tokenW = targetSize / ps;

    const bitmap = await createImageBitmap(blob, { resizeWidth: targetSize, resizeHeight: targetSize });
    const canvas = new OffscreenCanvas(targetSize, targetSize);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0);
    const resized = ctx.getImageData(0, 0, targetSize, targetSize);

    // Convert RGBA HWC to CHW float32, normalize to [-1, 1]
    // SHARP applies AffineRangeNormalizer(input_range=(0,1), output_range=(-1,1))
    // before the SPN encoder: output = input * 2.0 - 1.0
    const chw = new Float32Array(3 * targetSize * targetSize);
    for (let y = 0; y < targetSize; y++) {
      for (let x = 0; x < targetSize; x++) {
        const srcIdx = (y * targetSize + x) * 4;
        const dstBase = y * targetSize + x;
        chw[0 * targetSize * targetSize + dstBase] = resized.data[srcIdx] / 127.5 - 1.0;     // R
        chw[1 * targetSize * targetSize + dstBase] = resized.data[srcIdx + 1] / 127.5 - 1.0; // G
        chw[2 * targetSize * targetSize + dstBase] = resized.data[srcIdx + 2] / 127.5 - 1.0; // B
      }
    }

    // Destroy previous image buffer
    if (this._prevImageBuf) this._prevImageBuf.destroy();
    const imageBuf = createStorageBuffer(device, chw);
    this._prevImageBuf = imageBuf;

    // Run ViT encoder
    const enc = device.createCommandEncoder();
    const result = this.vitEncoder.encode(enc, imageBuf, this._patchEncoderWeights, tokenH, tokenW);
    device.queue.submit([enc.finish()]);

    // Read back CLS token for validation
    const D = VIT_CONFIG.dim;
    const clsSample = await readBuffer(device, result.finalTokensBuf, D * 4);

    // Validate output: check for NaN/Infinity
    let hasNaN = false;
    let nanCount = 0;
    for (let i = 0; i < clsSample.length; i++) {
      if (!isFinite(clsSample[i])) { hasNaN = true; nanCount++; }
    }

    console.log(`Backbone: ${result.intermediateFeatures.length} intermediate features captured`);
    console.log(`  Token grid: ${tokenH}x${tokenW} = ${result.numPatches} patches + 1 CLS`);
    console.log(`  CLS token [0:8]:`, Array.from(clsSample.slice(0, 8)).map(v => v.toFixed(4)));
    if (hasNaN) console.error(`  OUTPUT INVALID: ${nanCount}/${clsSample.length} non-finite values in CLS token`);

    return {
      tokenH,
      tokenW,
      dim: D,
      numPatches: result.numPatches,
      clsSample: Array.from(clsSample.slice(0, 8)),
      hasNaN,
      finalTokensBuf: result.finalTokensBuf,
      intermediateFeatures: result.intermediateFeatures,
    };
  }
}

export { VIT_CONFIG, ViTEncoder };
