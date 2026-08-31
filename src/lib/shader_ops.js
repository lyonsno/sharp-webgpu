/**
 * shader_ops.js — WebGPU compute dispatch wrappers for each shader.
 *
 * Each function creates a pipeline, binds buffers, and dispatches.
 * Pipelines are cached by device for reuse.
 */

import conv2dWGSL from '../shaders/conv2d.wgsl?raw';
import conv1x1WGSL from '../shaders/conv1x1.wgsl?raw';
import convTranspose2dWGSL from '../shaders/conv_transpose2d.wgsl?raw';
import activationsWGSL from '../shaders/activations.wgsl?raw';
import groupnormWGSL from '../shaders/groupnorm.wgsl?raw';
import pixelshuffleWGSL from '../shaders/pixelshuffle.wgsl?raw';
import upsampleWGSL from '../shaders/upsample.wgsl?raw';
import concatChannelsWGSL from '../shaders/concat_channels.wgsl?raw';
import tokenPatchMergeWGSL from '../shaders/token_patch_merge.wgsl?raw';
import gaussianInitializerFeatureInputWGSL from '../shaders/gaussian_initializer_feature_input.wgsl?raw';
import gaussianInitializerReduceMinWGSL from '../shaders/gaussian_initializer_reduce_min.wgsl?raw';

import { createStorageBuffer, createEmptyBuffer } from './gpu.js';

const pipelineCache = new WeakMap();
const uniformCache = new WeakMap();
const dummyBiasCache = new WeakMap();
const MAX_WG_DIM = 65535;

function exactUniformKey(bytes) {
  let key = `${bytes.byteLength}:`;
  for (let index = 0; index < bytes.byteLength; index++) {
    key += String.fromCharCode(bytes[index]);
  }
  return key;
}

function cachedUniform(device, data) {
  const bytes = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data);
  let deviceCache = uniformCache.get(device);
  if (!deviceCache) {
    deviceCache = new Map();
    uniformCache.set(device, deviceCache);
  }
  const key = exactUniformKey(bytes);
  if (deviceCache.has(key)) return deviceCache.get(key);
  const buf = device.createBuffer({
    size: Math.max(bytes.byteLength, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buf.getMappedRange()).set(bytes);
  buf.unmap();
  deviceCache.set(key, buf);
  return buf;
}

function getDummyBias(device) {
  if (!dummyBiasCache.has(device)) {
    dummyBiasCache.set(device, createStorageBuffer(device, new Float32Array([0])));
  }
  return dummyBiasCache.get(device);
}

/**
 * Split a total workgroup count into 2D dispatch (x, y) to stay within limits.
 * Returns [wgX, wgY] where wgX * wgY >= totalWG and wgX <= MAX_WG_DIM.
 */
function splitWorkgroups(totalWG) {
  if (totalWG <= MAX_WG_DIM) return [totalWG, 1];
  const wgX = MAX_WG_DIM;
  const wgY = Math.ceil(totalWG / MAX_WG_DIM);
  return [wgX, wgY];
}

function getOrCreatePipeline(device, key, code, entryPoint) {
  let deviceCache = pipelineCache.get(device);
  if (!deviceCache) {
    deviceCache = new Map();
    pipelineCache.set(device, deviceCache);
  }
  if (deviceCache.has(key)) return deviceCache.get(key);
  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint },
  });
  deviceCache.set(key, pipeline);
  return pipeline;
}

function ceil(a, b) { return Math.ceil(a / b); }

/**
 * Dispatch conv2d (3x3 or arbitrary kernel).
 * Returns output buffer [outC, outH, outW].
 */
export function dispatchConv2d(device, encoder, inputBuf, weightBuf, biasBuf, params) {
  const { inC, inH, inW, outC, kH, kW, padH, padW, strideH, strideW } = params;
  const outH = Math.floor((inH + 2 * padH - kH) / strideH) + 1;
  const outW = Math.floor((inW + 2 * padW - kW) / strideW) + 1;
  const hasBias = biasBuf ? 1 : 0;
  const totalOutputItems = outC * outH * outW;
  if (!Number.isSafeInteger(totalOutputItems) || totalOutputItems <= 0) {
    throw new RangeError('conv2d output dimensions must produce a positive safe-integer size');
  }
  const outputStart = params.outputStart ?? 0;
  const outputCount = params.outputCount ?? totalOutputItems;
  if (!Number.isSafeInteger(outputStart) || outputStart < 0 || outputStart >= totalOutputItems) {
    throw new RangeError('conv2d outputStart must be a non-negative safe integer');
  }
  if (!Number.isSafeInteger(outputCount) || outputCount <= 0 || outputCount > totalOutputItems - outputStart) {
    throw new RangeError('conv2d output range must be non-empty and within the output tensor');
  }
  const tiled = params.outputStart !== undefined || params.outputCount !== undefined;

  const pipeline = getOrCreatePipeline(
    device,
    tiled ? 'conv2d_tiled' : 'conv2d',
    conv2dWGSL,
    tiled ? 'conv2d_tiled_main' : 'conv2d_main'
  );

  const totalWG = ceil(outputCount, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);
  const uniformData = new Uint32Array([
    inC, inH, inW, outC, outH, outW, kH, kW,
    padH, padW, strideH, strideW, hasBias, wgX, outputStart, outputCount,
  ]);
  const uniformBuf = cachedUniform(device, uniformData);

  const dummyBias = biasBuf || getDummyBias(device);
  const outputBuf = params.outputBuffer || createEmptyBuffer(device, totalOutputItems * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: weightBuf } },
      { binding: 3, resource: { buffer: dummyBias } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass(params.computePassDescriptor);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  if (tiled) {
    pass.dispatchWorkgroups(wgX, wgY);
  } else {
    pass.dispatchWorkgroups(ceil(outW, 16), ceil(outH, 16), outC);
  }
  pass.end();

  return { buffer: outputBuf, outC, outH, outW };
}

/**
 * Dispatch 1x1 conv.
 */
export function dispatchConv1x1(device, encoder, inputBuf, weightBuf, biasBuf, params) {
  const { inC, outC, H, W } = params;
  const hasBias = biasBuf ? 1 : 0;
  const totalOutputItems = outC * H * W;
  if (!Number.isSafeInteger(totalOutputItems) || totalOutputItems <= 0) {
    throw new RangeError('conv1x1 output dimensions must produce a positive safe-integer size');
  }
  const outputStart = params.outputStart ?? 0;
  const outputCount = params.outputCount ?? totalOutputItems;
  if (!Number.isSafeInteger(outputStart) || outputStart < 0 || outputStart >= totalOutputItems) {
    throw new RangeError('conv1x1 outputStart must be a non-negative safe integer');
  }
  if (!Number.isSafeInteger(outputCount) || outputCount <= 0 || outputCount > totalOutputItems - outputStart) {
    throw new RangeError('conv1x1 output range must be non-empty and within the output tensor');
  }
  const tiled = params.outputStart !== undefined || params.outputCount !== undefined;

  const pipeline = getOrCreatePipeline(
    device,
    tiled ? 'conv1x1_tiled' : 'conv1x1',
    conv1x1WGSL,
    tiled ? 'conv1x1_tiled_main' : 'conv1x1_main',
  );

  const totalWG = ceil(outputCount, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);
  const uniformData = new Uint32Array([inC, outC, H, W, hasBias, wgX, outputStart, outputCount]);
  const uniformBuf = cachedUniform(device, uniformData);

  const dummyBias = biasBuf || getDummyBias(device);
  const outputBuf = params.outputBuffer || createEmptyBuffer(device, totalOutputItems * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: weightBuf } },
      { binding: 3, resource: { buffer: dummyBias } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass(params.computePassDescriptor);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  return { buffer: outputBuf, C: outC, H, W };
}

/**
 * Dispatch element-wise activation.
 * op: 0=relu, 1=silu, 2=add, 3=add_relu, 4=sigmoid
 */
export function dispatchActivation(device, encoder, inputA, inputB, count, op) {
  const pipeline = getOrCreatePipeline(device, 'activation', activationsWGSL, 'activation_main');

  const totalWG = ceil(count, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);
  const uniformData = new Uint32Array([count, op, wgX]);
  const uniformBuf = cachedUniform(device, uniformData);

  const dummyB = inputB || getDummyBias(device);
  const outputBuf = createEmptyBuffer(device, count * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputA } },
      { binding: 2, resource: { buffer: dummyB } },
      { binding: 3, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  return outputBuf;
}

/**
 * Dispatch GroupNorm (two-pass: stats then normalize).
 */
export function dispatchGroupNorm(device, encoder, inputBuf, scaleBuf, biasBuf, params) {
  const { C, H, W, numGroups, eps = 1e-5 } = params;

  const statsPipeline = getOrCreatePipeline(device, 'gn_stats', groupnormWGSL, 'groupnorm_stats');
  const normPipeline = getOrCreatePipeline(device, 'gn_norm', groupnormWGSL, 'groupnorm_normalize');

  // Uniform: C, H, W, numGroups, eps (f32), numWorkgroupsX (u32)
  const normTotalWG = ceil(C * H * W, 256);
  const [normWgX, normWgY] = splitWorkgroups(normTotalWG);
  const uniformArr = new ArrayBuffer(64);
  const u32View = new Uint32Array(uniformArr);
  const f32View = new Float32Array(uniformArr);
  u32View[0] = C; u32View[1] = H; u32View[2] = W; u32View[3] = numGroups;
  f32View[4] = eps;
  u32View[5] = normWgX;

  const uniformBuf = cachedUniform(device, new Uint8Array(uniformArr));

  const statsBuf = createEmptyBuffer(device, numGroups * 2 * 4);
  const outputBuf = createEmptyBuffer(device, C * H * W * 4);

  // Pass 1: compute stats (only uses bindings 0, 1, 5 — params, input, stats)
  const statsBindGroup = device.createBindGroup({
    layout: statsPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 5, resource: { buffer: statsBuf } },
    ],
  });

  const pass1 = encoder.beginComputePass();
  pass1.setPipeline(statsPipeline);
  pass1.setBindGroup(0, statsBindGroup);
  pass1.dispatchWorkgroups(ceil(numGroups, 256));
  pass1.end();

  // Pass 2: normalize
  const normBindGroup = device.createBindGroup({
    layout: normPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: scaleBuf } },
      { binding: 3, resource: { buffer: biasBuf } },
      { binding: 4, resource: { buffer: outputBuf } },
      { binding: 5, resource: { buffer: statsBuf } },
    ],
  });

  const pass2 = encoder.beginComputePass();
  pass2.setPipeline(normPipeline);
  pass2.setBindGroup(0, normBindGroup);
  pass2.dispatchWorkgroups(normWgX, normWgY);
  pass2.end();

  return outputBuf;
}

/**
 * Compute bounded partial GroupNorm means and M2 accumulators. One workgroup
 * owns one contiguous partial inside one group.
 */
export function dispatchGroupNormPartialStats(device, encoder, inputBuf, params) {
  const {
    C, H, W, numGroups, partialElements, partialStart, partialCount,
    partialsPerGroup, totalPartials, partialBuffer = null,
  } = params;
  const pipeline = getOrCreatePipeline(device, 'gn_partial_stats', groupnormWGSL, 'groupnorm_partial_stats');
  const [wgX, wgY] = splitWorkgroups(partialCount);
  const uniformData = new Uint32Array([
    C, H, W, numGroups,
    0, wgX, 0, 0,
    partialElements, partialStart, partialCount, partialsPerGroup,
    totalPartials, 0, 0, 0,
  ]);
  const uniformBuf = cachedUniform(device, uniformData);
  const output = partialBuffer || createEmptyBuffer(device, totalPartials * 2 * 4);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 5, resource: { buffer: output } },
    ],
  });
  const pass = encoder.beginComputePass(params.computePassDescriptor);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();
  return output;
}

/** Reduce partial GroupNorm Welford states to one mean and variance pair per group. */
export function dispatchGroupNormReduceStats(device, encoder, partialBuffer, params) {
  const { C, H, W, numGroups, partialsPerGroup, partialElements } = params;
  const pipeline = getOrCreatePipeline(device, 'gn_reduce_stats', groupnormWGSL, 'groupnorm_reduce_stats');
  const uniformData = new Uint32Array([
    C, H, W, numGroups,
    0, 0, 0, 0,
    partialElements, 0, 0, partialsPerGroup,
    numGroups * partialsPerGroup, 0, 0, 0,
  ]);
  const uniformBuf = cachedUniform(device, uniformData);
  const statsBuffer = createEmptyBuffer(device, numGroups * 2 * 4);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: partialBuffer } },
      { binding: 5, resource: { buffer: statsBuffer } },
    ],
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(numGroups);
  pass.end();
  return statsBuffer;
}

/** Normalize one exact output range from reduced GroupNorm statistics and apply ReLU. */
export function dispatchGroupNormNormalizeRelu(device, encoder, inputBuf, scaleBuf, biasBuf, statsBuf, params) {
  const { C, H, W, numGroups, eps = 1e-5 } = params;
  const totalOutputItems = C * H * W;
  const outputStart = params.outputStart ?? 0;
  const outputCount = params.outputCount ?? totalOutputItems;
  if (!Number.isSafeInteger(outputStart) || outputStart < 0 || outputStart >= totalOutputItems) {
    throw new RangeError('groupnorm outputStart must be a non-negative safe integer');
  }
  if (!Number.isSafeInteger(outputCount) || outputCount <= 0 || outputCount > totalOutputItems - outputStart) {
    throw new RangeError('groupnorm output range must be non-empty and within the output tensor');
  }
  const pipeline = getOrCreatePipeline(device, 'gn_normalize_relu_tiled', groupnormWGSL, 'groupnorm_normalize_tiled');
  const [wgX, wgY] = splitWorkgroups(ceil(outputCount, 256));
  const uniformArr = new ArrayBuffer(64);
  const u32View = new Uint32Array(uniformArr);
  const f32View = new Float32Array(uniformArr);
  u32View[0] = C; u32View[1] = H; u32View[2] = W; u32View[3] = numGroups;
  f32View[4] = eps;
  u32View[5] = wgX; u32View[6] = outputStart; u32View[7] = outputCount;
  const uniformBuf = cachedUniform(device, new Uint8Array(uniformArr));
  const outputBuf = params.outputBuffer || createEmptyBuffer(device, totalOutputItems * 4);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: scaleBuf } },
      { binding: 3, resource: { buffer: biasBuf } },
      { binding: 4, resource: { buffer: outputBuf } },
      { binding: 5, resource: { buffer: statsBuf } },
    ],
  });
  const pass = encoder.beginComputePass(params.computePassDescriptor);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();
  return outputBuf;
}

/**
 * Dispatch PixelShuffle.
 */
export function dispatchPixelShuffle(device, encoder, inputBuf, params) {
  const { inC, inH, inW, scaleFactor } = params;
  const outC = inC / (scaleFactor * scaleFactor);
  const outH = inH * scaleFactor;
  const outW = inW * scaleFactor;

  const pipeline = getOrCreatePipeline(device, 'pixelshuffle', pixelshuffleWGSL, 'pixelshuffle_main');

  const totalWG = ceil(outC * outH * outW, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);
  const uniformData = new Uint32Array([inC, inH, inW, outC, scaleFactor, wgX]);
  const uniformBuf = cachedUniform(device, uniformData);

  const outputBuf = createEmptyBuffer(device, outC * outH * outW * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  return { buffer: outputBuf, C: outC, H: outH, W: outW };
}

/**
 * Dispatch bilinear/nearest upsample.
 */
export function dispatchUpsample(device, encoder, inputBuf, params) {
  const { C, inH, inW, outH, outW, mode = 1 } = params; // mode: 0=nearest, 1=bilinear

  const pipeline = getOrCreatePipeline(device, 'upsample', upsampleWGSL, 'upsample_main');

  const totalWG = ceil(C * outH * outW, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);
  const uniformData = new Uint32Array([C, inH, inW, outH, outW, mode, wgX]);
  const uniformBuf = cachedUniform(device, uniformData);

  const outputBuf = createEmptyBuffer(device, C * outH * outW * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  return { buffer: outputBuf, C, H: outH, W: outW };
}

/**
 * Dispatch channel concat for two CHW buffers with matching H/W.
 * Returns output buffer [aC + bC, H, W].
 */
export function dispatchConcatChannels(device, encoder, inputABuf, inputBBuf, params) {
  const { aC, bC, H, W } = params;
  const outC = aC + bC;

  const pipeline = getOrCreatePipeline(device, 'concat_channels', concatChannelsWGSL, 'concat_channels_main');

  const totalWG = ceil(outC * H * W, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);
  const uniformData = new Uint32Array([aC, bC, H, W, wgX]);
  const uniformBuf = cachedUniform(device, uniformData);

  const outputBuf = createEmptyBuffer(device, outC * H * W * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputABuf } },
      { binding: 2, resource: { buffer: inputBBuf } },
      { binding: 3, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  return { buffer: outputBuf, C: outC, H, W };
}

function mergedPatchSize(steps, tokenSize, padding) {
  if (padding === 0) return steps * tokenSize;
  let mergedSize = 0;
  for (let s = 0; s < steps; s++) {
    let size = tokenSize;
    if (s > 0) size -= padding;
    if (s < steps - 1) size -= padding;
    mergedSize += size;
  }
  return mergedSize;
}

/**
 * Merge ViT token buffers into one CHW feature map on GPU.
 * Token buffers are [N, D] with CLS at token 0; output is [D, mergedH, mergedW].
 */
export function dispatchMergeTokenPatches(device, encoder, tokenBuffers, params) {
  const { steps, D, tokenH, tokenW, padding = 0 } = params;
  const patchCount = tokenBuffers.length;
  const tokenCount = tokenH * tokenW + 1;
  const tokenBytes = tokenCount * D * 4;
  const mergedSize = mergedPatchSize(steps, tokenH, padding);

  if (patchCount !== steps * steps) {
    throw new Error(`dispatchMergeTokenPatches expected ${steps * steps} buffers, got ${patchCount}`);
  }
  if (tokenH !== tokenW) {
    throw new Error('dispatchMergeTokenPatches currently expects square token grids');
  }

  const pipeline = getOrCreatePipeline(device, 'token_patch_merge', tokenPatchMergeWGSL, 'token_patch_merge_main');
  const stackedBuf = createEmptyBuffer(device, patchCount * tokenBytes);
  for (let i = 0; i < patchCount; i++) {
    encoder.copyBufferToBuffer(tokenBuffers[i], 0, stackedBuf, i * tokenBytes, tokenBytes);
  }

  const outputBuf = createEmptyBuffer(device, D * mergedSize * mergedSize * 4);
  const totalWG = ceil(D * mergedSize * mergedSize, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);
  const uniformData = new Uint32Array([patchCount, steps, tokenH, tokenW, D, padding, mergedSize, wgX]);
  const uniformBuf = cachedUniform(device, uniformData);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: stackedBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  return { buffer: outputBuf, C: D, H: mergedSize, W: mergedSize, scratchBuffers: [stackedBuf] };
}

/**
 * Dispatch Gaussian initializer feature_input construction on GPU.
 * Produces [5, H, W] in CHW order:
 *   channels 0-2: original normalized image channels
 *   channels 3-4: normalized disparity channels after depth min/rescale
 */
export function dispatchGaussianInitializerFeatureInput(device, encoder, imageBuf, disparityBuf, params) {
  const { H, W } = params;
  const HW = H * W;
  const reduceFromDisparityPipeline = getOrCreatePipeline(
    device,
    'gaussian_depth_min_from_disparity',
    gaussianInitializerReduceMinWGSL,
    'depth_min_from_disparity_main'
  );
  const reduceMinPipeline = getOrCreatePipeline(
    device,
    'gaussian_reduce_min',
    gaussianInitializerReduceMinWGSL,
    'reduce_min_main'
  );
  const featureInputPipeline = getOrCreatePipeline(
    device,
    'gaussian_initializer_feature_input',
    gaussianInitializerFeatureInputWGSL,
    'feature_input_main'
  );

  const scratchBuffers = [];
  let inputBuf = disparityBuf;
  let inputCount = 2 * HW;
  let firstPass = true;

  while (inputCount > 1) {
    const groupCount = ceil(inputCount, 256);
    const [wgX, wgY] = splitWorkgroups(groupCount);
    const uniformData = new Uint32Array([inputCount, wgX, 0, 0]);
    const uniformBuf = cachedUniform(device, uniformData);
    const outputBuf = createEmptyBuffer(device, groupCount * 4);
    const pipeline = firstPass ? reduceFromDisparityPipeline : reduceMinPipeline;

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: inputBuf } },
        { binding: 2, resource: { buffer: outputBuf } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();

    scratchBuffers.push(outputBuf);
    inputBuf = outputBuf;
    inputCount = groupCount;
    firstPass = false;
  }

  const depthMinBuf = inputBuf;
  const outputBuf = createEmptyBuffer(device, 5 * HW * 4);
  const totalWG = ceil(5 * HW, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);
  const uniformData = new Uint32Array([H, W, wgX, 0]);
  const uniformBuf = cachedUniform(device, uniformData);

  const bindGroup = device.createBindGroup({
    layout: featureInputPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: imageBuf } },
      { binding: 2, resource: { buffer: disparityBuf } },
      { binding: 3, resource: { buffer: depthMinBuf } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(featureInputPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  return { buffer: outputBuf, C: 5, H, W, scratchBuffers };
}

/**
 * Dispatch transposed 2D convolution (deconvolution).
 * ConvTranspose2d(inC, outC, kernel_size=stride, stride=stride)
 */
export function dispatchConvTranspose2d(device, encoder, inputBuf, weightBuf, biasBuf, params) {
  const { inC, inH, inW, outC, stride } = params;
  const kH = stride, kW = stride;
  const outH = inH * stride;
  const outW = inW * stride;
  const hasBias = biasBuf ? 1 : 0;
  const totalOutputItems = outC * outH * outW;
  const outputStart = params.outputStart ?? 0;
  const outputCount = params.outputCount ?? totalOutputItems;
  if (!Number.isSafeInteger(outputStart) || outputStart < 0) {
    throw new RangeError('conv-transpose outputStart must be a non-negative safe integer');
  }
  if (!Number.isSafeInteger(outputCount) || outputCount <= 0 || outputStart + outputCount > totalOutputItems) {
    throw new RangeError('conv-transpose output range must be non-empty and within the output tensor');
  }

  const pipeline = getOrCreatePipeline(device, 'conv_transpose2d', convTranspose2dWGSL, 'conv_transpose2d_main');

  const totalWG = ceil(outputCount, 256);
  const [wgX, wgY] = splitWorkgroups(totalWG);
  const uniformData = new Uint32Array([inC, inH, inW, outC, outH, outW, kH, kW, stride, stride, hasBias, wgX, outputStart, outputCount]);
  const uniformBuf = cachedUniform(device, uniformData);

  const dummyBias = biasBuf || getDummyBias(device);
  const outputBuf = params.outputBuffer || createEmptyBuffer(device, totalOutputItems * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: weightBuf } },
      { binding: 3, resource: { buffer: dummyBias } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass(params.computePassDescriptor);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  return { buffer: outputBuf, C: outC, H: outH, W: outW };
}
