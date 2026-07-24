/**
 * weights.js — Load SHARP weights from flat binary format.
 *
 * Binary format (from convert_weights.py):
 *   Header: 4 (magic) + 4 (version) + 4 (num_tensors) + 4 (header_size) = 16 bytes
 *   Tensor table: num_tensors x 160 bytes each
 *     128 bytes: name (null-padded ASCII)
 *     4 bytes: dtype (0=fp32, 1=fp16)
 *     4 bytes: ndim
 *     16 bytes: shape (4 x u32)
 *     4 bytes: offset
 *     4 bytes: size
 *   Weight data: packed tensors
 */

import { createStorageBuffer } from './gpu.js';

const MAGIC = 0x50524853; // "SHRP" in little-endian
const ENTRY_SIZE = 160; // 128 (name) + 4 (dtype) + 4 (ndim) + 16 (shape) + 4 (offset) + 4 (size)

function nowMs() {
  return globalThis.performance?.now?.() ?? Date.now();
}

function emitWeightPhase(onPhase, phase, status, startedAtMs, details = {}) {
  const endedAtMs = nowMs();
  onPhase?.({
    phase,
    status,
    intervalStartMs: startedAtMs,
    intervalEndMs: endedAtMs,
    durationMs: endedAtMs - startedAtMs,
    ...details,
  });
}

function donateTaskTurn(event) {
  return new Promise(resolve => setTimeout(() => resolve(event), 0));
}

/**
 * Fetch through response headers while preserving failures that occur before a
 * body stream exists.
 */
export async function fetchWeightResponse(url, options = {}) {
  const { onPhase } = options;
  const fetchImpl = options.fetchImpl || globalThis.fetch;
  const startedAtMs = nowMs();
  onPhase?.({ phase: 'fetch-response', status: 'started', intervalStartMs: startedAtMs });
  try {
    if (typeof fetchImpl !== 'function') throw new Error('Fetch is unavailable for weight loading');
    const response = await fetchImpl(url);
    if (!response?.ok) {
      throw new Error(`Failed to fetch weights: ${response?.status ?? 'unknown status'}`);
    }
    emitWeightPhase(onPhase, 'fetch-response', 'completed', startedAtMs, {
      responseStatus: response.status ?? null,
    });
    return response;
  } catch (error) {
    emitWeightPhase(onPhase, 'fetch-response', 'failed', startedAtMs, {
      error: error instanceof Error ? error.message : String(error),
    });
    throw error;
  }
}

/**
 * Assemble a streamed weight response without a second full-file copy when the
 * server declares a trustworthy byte length.
 */
export async function readWeightResponse(response, options = {}) {
  const { onPhase, onProgress } = options;
  const yieldControl = options.yieldControl || donateTaskTurn;
  const contentLengthValue = response?.headers?.get?.('content-length');
  const contentEncoding = response?.headers?.get?.('content-encoding');
  const parsedContentLength = Number(contentLengthValue);
  const hasIdentityEncoding = !contentEncoding || contentEncoding.toLowerCase() === 'identity';
  const declaredBytes = hasIdentityEncoding
    && Number.isSafeInteger(parsedContentLength)
    && parsedContentLength > 0
    ? parsedContentLength
    : 0;
  const reader = response?.body?.getReader?.();
  if (!reader) throw new Error('Weight response body is not a readable stream');

  const fetchStartedAtMs = nowMs();
  onPhase?.({ phase: 'fetch-stream', status: 'started', intervalStartMs: fetchStartedAtMs });
  let receivedBytes = 0;
  let target = null;
  const chunks = [];

  try {
    if (declaredBytes > 0) target = new Uint8Array(new ArrayBuffer(declaredBytes));
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!(value instanceof Uint8Array)) {
        throw new Error('Weight response stream returned a non-byte chunk');
      }
      if (declaredBytes > 0 && receivedBytes + value.byteLength > declaredBytes) {
        throw new Error(
          `Weight response declared ${declaredBytes} bytes but exceeded that length while streaming`,
        );
      }
      if (target) target.set(value, receivedBytes);
      else chunks.push(value);
      receivedBytes += value.byteLength;
      onProgress?.(receivedBytes, declaredBytes);
    }
    if (declaredBytes > 0 && receivedBytes !== declaredBytes) {
      throw new Error(`Weight response declared ${declaredBytes} bytes but received ${receivedBytes}`);
    }
  } catch (error) {
    emitWeightPhase(onPhase, 'fetch-stream', 'failed', fetchStartedAtMs, {
      receivedBytes,
      declaredBytes: declaredBytes || null,
      error: error instanceof Error ? error.message : String(error),
    });
    throw error;
  }

  emitWeightPhase(onPhase, 'fetch-stream', 'completed', fetchStartedAtMs, {
    receivedBytes,
    declaredBytes: declaredBytes || null,
  });

  let assemblyMode;
  let postDownloadCopyBytes;
  if (target) {
    assemblyMode = 'preallocated-content-length';
    postDownloadCopyBytes = 0;
    emitWeightPhase(onPhase, 'buffer-consolidation', 'skipped', nowMs(), {
      reason: 'streamed-directly-into-declared-size-buffer',
      receivedBytes,
    });
  } else {
    assemblyMode = 'chunk-consolidation';
    postDownloadCopyBytes = receivedBytes;
    const consolidationStartedAtMs = nowMs();
    onPhase?.({
      phase: 'buffer-consolidation',
      status: 'started',
      intervalStartMs: consolidationStartedAtMs,
      receivedBytes,
    });
    try {
      target = new Uint8Array(new ArrayBuffer(receivedBytes));
      let offset = 0;
      for (const chunk of chunks) {
        target.set(chunk, offset);
        offset += chunk.byteLength;
      }
      emitWeightPhase(onPhase, 'buffer-consolidation', 'completed', consolidationStartedAtMs, {
        receivedBytes,
        copiedBytes: receivedBytes,
      });
    } catch (error) {
      emitWeightPhase(onPhase, 'buffer-consolidation', 'failed', consolidationStartedAtMs, {
        receivedBytes,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  await yieldControl({
    phase: 'fetch-stream-complete',
    receivedBytes,
    declaredBytes: declaredBytes || null,
    assemblyMode,
  });
  return {
    buffer: target.buffer,
    receivedBytes,
    declaredBytes,
    assemblyMode,
    postDownloadCopyBytes,
  };
}

/**
 * Parse the binary header and tensor table.
 */
function parseHeader(buffer) {
  const view = new DataView(buffer);

  const magic = view.getUint32(0, true);
  if (magic !== MAGIC) {
    throw new Error(`Invalid weight file magic: 0x${magic.toString(16)}`);
  }

  const version = view.getUint32(4, true);
  if (version !== 1) {
    throw new Error(`Unsupported weight file version: ${version}`);
  }

  const numTensors = view.getUint32(8, true);
  const headerSize = view.getUint32(12, true);

  const expectedHeaderSize = 16 + numTensors * ENTRY_SIZE;
  if (expectedHeaderSize > buffer.byteLength) {
    throw new Error(`Corrupt weight file: header claims ${numTensors} tensors but file is only ${buffer.byteLength} bytes`);
  }

  const tensors = new Map();
  for (let i = 0; i < numTensors; i++) {
    const entryOffset = 16 + i * ENTRY_SIZE;

    const nameBytes = new Uint8Array(buffer, entryOffset, 128);
    let nameEnd = nameBytes.indexOf(0);
    if (nameEnd === -1) nameEnd = 128;
    const name = new TextDecoder().decode(nameBytes.slice(0, nameEnd));

    const dtype = view.getUint32(entryOffset + 128, true);
    const ndim = view.getUint32(entryOffset + 132, true);
    const shape = [];
    for (let d = 0; d < ndim; d++) {
      shape.push(view.getUint32(entryOffset + 136 + d * 4, true));
    }
    const offset = view.getUint32(entryOffset + 152, true);
    const size = view.getUint32(entryOffset + 156, true);

    tensors.set(name, { dtype, shape, offset, size });
  }

  return { tensors, headerSize };
}

/**
 * Convert fp16 (as uint16) to fp32.
 */
function fp16ToFp32(h) {
  const sign = (h >> 15) & 1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;

  if (exp === 0) {
    if (mant === 0) return sign ? -0.0 : 0.0;
    let val = mant / 1024.0 * Math.pow(2, -14);
    return sign ? -val : val;
  }
  if (exp === 31) {
    return mant === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }

  const val = Math.pow(2, exp - 15) * (1 + mant / 1024.0);
  return sign ? -val : val;
}

function normalizeWriteBytes(value) {
  const requested = Number(value);
  const bytes = Number.isSafeInteger(requested) && requested >= 4
    ? requested
    : 4 * 1024 * 1024;
  return Math.max(4, bytes - (bytes % 4));
}

/**
 * Convert one tensor into its final mapped GPU allocation through bounded host
 * slices. This preserves the fast-path buffer representation without requiring
 * a whole-tensor temporary fp32 allocation or one uninterrupted CPU wall.
 */
export async function extractTensorCooperatively(
  device,
  buffer,
  tensorInfo,
  options = {},
) {
  const { dtype, offset, size } = tensorInfo;
  if (offset + size > buffer.byteLength) {
    throw new Error(`Tensor at offset ${offset} with size ${size} exceeds buffer length ${buffer.byteLength}`);
  }
  if ((dtype !== 0 && dtype !== 1) || size % (dtype === 0 ? 4 : 2) !== 0) {
    throw new Error(`Unsupported or misaligned tensor dtype ${dtype} with size ${size}`);
  }
  if (typeof device?.createBuffer !== 'function') {
    throw new TypeError('Cooperative tensor materialization requires device.createBuffer');
  }

  const tensorName = options.tensorName || 'unnamed-tensor';
  const yieldControl = options.yieldControl || donateTaskTurn;
  const maxWriteBytes = normalizeWriteBytes(options.maxWriteBytes);
  const totalBytes = dtype === 0 ? size : size * 2;
  const totalChunks = Math.ceil(totalBytes / maxWriteBytes);
  const gpuBuffer = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  let mapped = true;

  try {
    const fp16 = dtype === 1 ? new Uint16Array(buffer, offset, size / 2) : null;
    const mappedRange = gpuBuffer.getMappedRange();
    const mappedBytes = new Uint8Array(mappedRange);
    const mappedFp32 = dtype === 1 ? new Float32Array(mappedRange) : null;
    for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex += 1) {
      const outputStart = chunkIndex * maxWriteBytes;
      const outputEnd = Math.min(totalBytes, outputStart + maxWriteBytes);
      const outputBytes = outputEnd - outputStart;
      if (dtype === 0) {
        mappedBytes.set(
          new Uint8Array(buffer, offset + outputStart, outputBytes),
          outputStart,
        );
      } else {
        const elementStart = outputStart / 4;
        const elementEnd = outputEnd / 4;
        for (let index = elementStart; index < elementEnd; index += 1) {
          mappedFp32[index] = fp16ToFp32(fp16[index]);
        }
      }
      if (chunkIndex + 1 < totalChunks) {
        await yieldControl({
          phase: 'initial-gpu-materialization',
          step: 'tensor-upload-chunk',
          tensorName,
          dtype: dtype === 0 ? 'fp32' : 'fp16',
          sourceBytes: size,
          completedBytes: outputEnd,
          totalBytes,
          chunkIndex: chunkIndex + 1,
          totalChunks,
          maxWriteBytes,
        });
      }
    }
    gpuBuffer.unmap();
    mapped = false;
    return gpuBuffer;
  } catch (error) {
    if (mapped) {
      try {
        gpuBuffer.unmap();
      } catch {
        // Destruction below is authoritative cleanup for a failed mapped buffer.
      }
    }
    gpuBuffer.destroy?.();
    throw error;
  }
}

/**
 * Extract a tensor from the binary buffer as a GPU storage buffer.
 */
function extractTensor(device, buffer, tensorInfo) {
  const { dtype, offset, size } = tensorInfo;

  if (offset + size > buffer.byteLength) {
    throw new Error(`Tensor at offset ${offset} with size ${size} exceeds buffer length ${buffer.byteLength}`);
  }

  if (dtype === 0) {
    const data = new Float32Array(buffer, offset, size / 4);
    return createStorageBuffer(device, data);
  } else {
    const fp16 = new Uint16Array(buffer, offset, size / 2);
    const fp32 = new Float32Array(fp16.length);
    for (let i = 0; i < fp16.length; i++) {
      fp32[i] = fp16ToFp32(fp16[i]);
    }
    return createStorageBuffer(device, fp32);
  }
}

/**
 * Extract tensor data as CPU Float32Array.
 */
function extractTensorCPU(buffer, tensorInfo) {
  const { dtype, offset, size } = tensorInfo;
  if (dtype === 0) {
    return new Float32Array(buffer.slice(offset, offset + size));
  } else {
    const fp16 = new Uint16Array(buffer, offset, size / 2);
    const fp32 = new Float32Array(fp16.length);
    for (let i = 0; i < fp16.length; i++) {
      fp32[i] = fp16ToFp32(fp16[i]);
    }
    return fp32;
  }
}

/**
 * Create stable accessors for immutable model weights. GPU buffers are created
 * at most once per tensor name and then shared by every model stage.
 */
export function createWeightTensorAccessors(device, buffer, tensors, options = {}) {
  const extractGpuTensor = options.extractGpuTensor || extractTensor;
  const extractGpuTensorCooperatively = options.extractGpuTensorCooperatively
    || extractTensorCooperatively;
  const extractCpuTensor = options.extractCpuTensor || extractTensorCPU;
  const gpuTensorCache = new Map();
  const materializationFlights = new Map();

  const getInfo = (name) => {
    const info = tensors.get(name);
    if (!info) throw new Error(`Missing weight info: ${name}`);
    return info;
  };

  const get = (name) => {
    if (gpuTensorCache.has(name)) return gpuTensorCache.get(name);
    if (materializationFlights.has(name)) {
      throw new Error(`Weight materialization is in progress: ${name}; await materialize()`);
    }
    const info = tensors.get(name);
    if (!info) throw new Error(`Missing weight: ${name}`);
    const gpuTensor = extractGpuTensor(device, buffer, info);
    gpuTensorCache.set(name, gpuTensor);
    return gpuTensor;
  };

  const tryGet = (name) => (tensors.has(name) ? get(name) : null);
  const materialize = async (name, materializationOptions = {}) => {
    if (gpuTensorCache.has(name)) return gpuTensorCache.get(name);
    if (materializationFlights.has(name)) return materializationFlights.get(name);
    const info = tensors.get(name);
    if (!info) throw new Error(`Missing weight: ${name}`);
    const flight = extractGpuTensorCooperatively(
      device,
      buffer,
      info,
      {
        ...materializationOptions,
        tensorName: name,
      },
    );
    materializationFlights.set(name, flight);
    try {
      const gpuTensor = await flight;
      gpuTensorCache.set(name, gpuTensor);
      return gpuTensor;
    } finally {
      materializationFlights.delete(name);
    }
  };
  const tryMaterialize = (name, materializationOptions = {}) => (
    tensors.has(name) ? materialize(name, materializationOptions) : Promise.resolve(null)
  );

  return {
    get,
    tryGet,
    materialize,
    tryMaterialize,
    getInfo,
    extractTensorCPU: name => extractCpuTensor(buffer, getInfo(name)),
  };
}

async function runWeightLoadPhase(onPhase, phase, task) {
  const startedAtMs = nowMs();
  onPhase?.({ phase, status: 'started', intervalStartMs: startedAtMs });
  try {
    const result = await task();
    emitWeightPhase(onPhase, phase, 'completed', startedAtMs);
    return result;
  } catch (error) {
    emitWeightPhase(onPhase, phase, 'failed', startedAtMs, {
      error: error instanceof Error ? error.message : String(error),
    });
    throw error;
  }
}

/**
 * Load SHARP weights from binary file.
 *
 * State dict key structure (from RGBGaussianPredictor):
 *   monodepth_model.monodepth_predictor.encoder.patch_encoder.*  — SPN patch ViT
 *   monodepth_model.monodepth_predictor.encoder.image_encoder.*  — SPN image ViT
 *   monodepth_model.monodepth_predictor.encoder.upsample*        — SPN fusion layers
 *   monodepth_model.monodepth_predictor.encoder.fuse_lowres.*    — SPN patch+image fusion
 *   monodepth_model.monodepth_predictor.decoder.*                — Monodepth MultiresConvDecoder
 *   monodepth_model.monodepth_predictor.head.*                   — Disparity head
 *   monodepth_model.monodepth_predictor.normalizer.*             — AffineRangeNormalizer
 *   feature_model.*                                              — Gaussian DPT decoder
 *   prediction_head.*                                            — DirectPredictionHead (1x1 convs)
 *   depth_alignment.*                                            — DepthAlignment / scale_map_estimator
 *
 * init_model (MultiLayerInitializer) and gaussian_composer have NO learned weights.
 */
export async function loadWeights(device, url, onProgress, options = {}) {
  const response = await fetchWeightResponse(url, { onPhase: options.onPhase });

  const yieldControl = options.yieldControl || donateTaskTurn;
  const { buffer } = await readWeightResponse(response, {
    onProgress,
    onPhase: options.onPhase,
    yieldControl,
  });
  const { tensors } = await runWeightLoadPhase(
    options.onPhase,
    'header-parse',
    () => parseHeader(buffer),
  );
  const accessors = createWeightTensorAccessors(device, buffer, tensors);
  const {
    materialize,
  } = accessors;
  const materializationWriteBytes = normalizeWriteBytes(options.materializationWriteBytes);
  const materializeWeight = async (name, details = {}) => {
    const info = accessors.getInfo(name);
    const startedAtMs = nowMs();
    options.onPhase?.({
      phase: 'initial-gpu-materialization',
      status: 'started',
      intervalStartMs: startedAtMs,
      tensorName: name,
      dtype: info.dtype === 0 ? 'fp32' : 'fp16',
      sourceBytes: info.size,
      maxWriteBytes: materializationWriteBytes,
      ...details,
    });
    try {
      const result = await materialize(name, {
        maxWriteBytes: materializationWriteBytes,
        yieldControl: receipt => yieldControl({
          ...details,
          ...receipt,
        }),
      });
      emitWeightPhase(
        options.onPhase,
        'initial-gpu-materialization',
        'completed',
        startedAtMs,
        {
          tensorName: name,
          dtype: info.dtype === 0 ? 'fp32' : 'fp16',
          sourceBytes: info.size,
          outputBytes: info.dtype === 0 ? info.size : info.size * 2,
          maxWriteBytes: materializationWriteBytes,
          role: 'cooperative-duty-interval',
          ...details,
        },
      );
      return result;
    } catch (error) {
      emitWeightPhase(
        options.onPhase,
        'initial-gpu-materialization',
        'failed',
        startedAtMs,
        {
          tensorName: name,
          dtype: info.dtype === 0 ? 'fp32' : 'fp16',
          sourceBytes: info.size,
          maxWriteBytes: materializationWriteBytes,
          role: 'cooperative-duty-interval',
          error: error instanceof Error ? error.message : String(error),
          ...details,
        },
      );
      throw error;
    }
  };
  const tryMaterializeWeight = async (name, details = {}) => {
    if (!tensors.has(name)) return null;
    return materializeWeight(name, details);
  };

  // --- Build ViT block weights for a given encoder prefix ---
  async function buildViTBlocks(prefix, numLayers, encoder) {
    const blocks = {};
    for (let l = 0; l < numLayers; l++) {
      const bp = `${prefix}.blocks.${l}`;
      const blockTensorNames = [
        'attn.qkv.weight', 'attn.qkv.bias',
        'attn.proj.weight', 'attn.proj.bias',
        'norm1.weight', 'norm1.bias',
        'norm2.weight', 'norm2.bias',
        'ls1.gamma', 'ls2.gamma',
        'mlp.fc1.weight', 'mlp.fc1.bias',
        'mlp.fc2.weight', 'mlp.fc2.bias',
        // GluMlp variant (defensive — not used by default dinov2l16_384 preset)
        'mlp.w12.weight', 'mlp.w12.bias',
        'mlp.w3.weight', 'mlp.w3.bias',
      ];
      for (let tensorIndex = 0; tensorIndex < blockTensorNames.length; tensorIndex += 1) {
        const name = blockTensorNames[tensorIndex];
        const fullName = `${bp}.${name}`;
        const buf = await tryMaterializeWeight(fullName, {
          encoder,
          blockIndex: l,
          tensorIndex,
          tensorsPerBlock: blockTensorNames.length,
        });
        if (buf) blocks[fullName] = buf;
      }
      options.onPhase?.({
        phase: 'initial-gpu-materialization',
        status: 'progress',
        encoder,
        completedBlocks: l + 1,
        totalBlocks: numLayers,
      });
      await yieldControl({
        phase: 'initial-gpu-materialization',
        step: 'vit-block',
        encoder,
        completedBlocks: l + 1,
        totalBlocks: numLayers,
      });
    }
    return blocks;
  }

  const { patchEncoder, imageEncoder, predictionHead } = await runWeightLoadPhase(
    options.onPhase,
    'initial-gpu-materialization',
    async () => {
      // --- Patch encoder (ViT in SPN) ---
      const patchEncoderPrefix = 'monodepth_model.monodepth_predictor.encoder.patch_encoder';
      const patchEncoder = {
        patchEmbed: {
          weight: await materializeWeight(`${patchEncoderPrefix}.patch_embed.proj.weight`, {
            encoder: 'patch',
            role: 'patch-embed',
          }),
          bias: await materializeWeight(`${patchEncoderPrefix}.patch_embed.proj.bias`, {
            encoder: 'patch',
            role: 'patch-embed',
          }),
        },
        posEmbed: await materializeWeight(`${patchEncoderPrefix}.pos_embed`, {
          encoder: 'patch',
          role: 'embedding',
        }),
        clsToken: await materializeWeight(`${patchEncoderPrefix}.cls_token`, {
          encoder: 'patch',
          role: 'embedding',
        }),
        norm: {
          weight: await materializeWeight(`${patchEncoderPrefix}.norm.weight`, {
            encoder: 'patch',
            role: 'final-norm',
          }),
          bias: await materializeWeight(`${patchEncoderPrefix}.norm.bias`, {
            encoder: 'patch',
            role: 'final-norm',
          }),
        },
        blockWeights: await buildViTBlocks(patchEncoderPrefix, 24, 'patch'),
      };

      // --- Image encoder (ViT in SPN) ---
      const imageEncoderPrefix = 'monodepth_model.monodepth_predictor.encoder.image_encoder';
      const imageEncoder = {
        patchEmbed: {
          weight: await materializeWeight(`${imageEncoderPrefix}.patch_embed.proj.weight`, {
            encoder: 'image',
            role: 'patch-embed',
          }),
          bias: await materializeWeight(`${imageEncoderPrefix}.patch_embed.proj.bias`, {
            encoder: 'image',
            role: 'patch-embed',
          }),
        },
        posEmbed: await materializeWeight(`${imageEncoderPrefix}.pos_embed`, {
          encoder: 'image',
          role: 'embedding',
        }),
        clsToken: await materializeWeight(`${imageEncoderPrefix}.cls_token`, {
          encoder: 'image',
          role: 'embedding',
        }),
        norm: {
          weight: await materializeWeight(`${imageEncoderPrefix}.norm.weight`, {
            encoder: 'image',
            role: 'final-norm',
          }),
          bias: await materializeWeight(`${imageEncoderPrefix}.norm.bias`, {
            encoder: 'image',
            role: 'final-norm',
          }),
        },
        blockWeights: await buildViTBlocks(imageEncoderPrefix, 24, 'image'),
      };

      // --- Prediction head ---
      const predictionHead = {
        geometry: {
          weight: await materializeWeight('prediction_head.geometry_prediction_head.weight', {
            role: 'prediction-head',
          }),
          bias: await materializeWeight('prediction_head.geometry_prediction_head.bias', {
            role: 'prediction-head',
          }),
        },
        texture: {
          weight: await materializeWeight('prediction_head.texture_prediction_head.weight', {
            role: 'prediction-head',
          }),
          bias: await materializeWeight('prediction_head.texture_prediction_head.bias', {
            role: 'prediction-head',
          }),
        },
      };
      await yieldControl({ phase: 'initial-gpu-materialization', step: 'complete' });
      return { patchEncoder, imageEncoder, predictionHead };
    },
  );

  const weights = {
    patchEncoder,
    imageEncoder,
    predictionHead,
    // SPN fusion, decoder, feature_model, depth_alignment weights will be
    // wired as we implement each stage. For now, store raw tensor map for
    // incremental bring-up.
    raw: { tensors, buffer, ...accessors },
  };

  console.log(`Loaded ${tensors.size} tensors from SHARP weight file`);
  return weights;
}
