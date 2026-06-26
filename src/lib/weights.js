/**
 * weights.js — Load SHARP weights from flat binary format.
 *
 * Binary format (from convert_weights.py):
 *   Header: 4 (magic) + 4 (version) + 4 (num_tensors) + 4 (header_size) = 16 bytes
 *   Tensor table: num_tensors x 96 bytes each
 *     64 bytes: name (null-padded ASCII)
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
 * Load SHARP weights from binary file.
 *
 * State dict key structure (from RGBGaussianPredictor):
 *   monodepth_model.dpt.encoder.patch_encoder.*   — SPN patch ViT
 *   monodepth_model.dpt.encoder.image_encoder.*   — SPN image ViT
 *   monodepth_model.dpt.encoder.*                 — SPN fusion layers
 *   monodepth_model.dpt.decoder.*                 — Monodepth MultiresConvDecoder
 *   monodepth_model.dpt.head.*                    — Disparity head
 *   monodepth_model.dpt.normalizer.*              — AffineRangeNormalizer
 *   feature_model.*                               — Gaussian DPT decoder
 *   prediction_head.*                             — DirectPredictionHead (1x1 convs)
 *   depth_alignment.*                             — DepthAlignment / scale_map_estimator
 *
 * init_model (MultiLayerInitializer) and gaussian_composer have NO learned weights.
 */
export async function loadWeights(device, url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch weights: ${response.status}`);
  }

  const contentLength = parseInt(response.headers.get('content-length') || '0');
  const reader = response.body.getReader();

  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (onProgress) onProgress(received, contentLength);
  }

  const buffer = new ArrayBuffer(received);
  const uint8 = new Uint8Array(buffer);
  let pos = 0;
  for (const chunk of chunks) {
    uint8.set(chunk, pos);
    pos += chunk.length;
  }

  const { tensors } = parseHeader(buffer);

  const get = (name) => {
    const info = tensors.get(name);
    if (!info) throw new Error(`Missing weight: ${name}`);
    return extractTensor(device, buffer, info);
  };

  const tryGet = (name) => {
    const info = tensors.get(name);
    if (!info) return null;
    return extractTensor(device, buffer, info);
  };

  const getInfo = (name) => {
    const info = tensors.get(name);
    if (!info) throw new Error(`Missing weight info: ${name}`);
    return info;
  };

  // --- Build ViT block weights for a given encoder prefix ---
  function buildViTBlocks(prefix, numLayers) {
    const blocks = {};
    for (let l = 0; l < numLayers; l++) {
      const bp = `${prefix}.blocks.${l}`;
      for (const name of [
        'attn.qkv.weight', 'attn.qkv.bias',
        'attn.proj.weight', 'attn.proj.bias',
        'norm1.weight', 'norm1.bias',
        'norm2.weight', 'norm2.bias',
        'ls1.gamma', 'ls2.gamma',
        'mlp.fc1.weight', 'mlp.fc1.bias',
        'mlp.fc2.weight', 'mlp.fc2.bias',
      ]) {
        const fullName = `${bp}.${name}`;
        const buf = tryGet(fullName);
        if (buf) blocks[fullName] = buf;
      }
    }
    return blocks;
  }

  // --- Patch encoder (ViT in SPN) ---
  // Actual key prefix from checkpoint: monodepth_model.monodepth_predictor.encoder.patch_encoder
  const patchEncoderPrefix = 'monodepth_model.monodepth_predictor.encoder.patch_encoder';
  const patchEncoder = {
    patchEmbed: {
      weight: get(`${patchEncoderPrefix}.patch_embed.proj.weight`),
      bias: get(`${patchEncoderPrefix}.patch_embed.proj.bias`),
    },
    posEmbed: get(`${patchEncoderPrefix}.pos_embed`),
    clsToken: get(`${patchEncoderPrefix}.cls_token`),
    norm: {
      weight: get(`${patchEncoderPrefix}.norm.weight`),
      bias: get(`${patchEncoderPrefix}.norm.bias`),
    },
    blockWeights: buildViTBlocks(patchEncoderPrefix, 24),
  };

  // --- Image encoder (ViT in SPN) ---
  const imageEncoderPrefix = 'monodepth_model.monodepth_predictor.encoder.image_encoder';
  const imageEncoder = {
    patchEmbed: {
      weight: get(`${imageEncoderPrefix}.patch_embed.proj.weight`),
      bias: get(`${imageEncoderPrefix}.patch_embed.proj.bias`),
    },
    posEmbed: get(`${imageEncoderPrefix}.pos_embed`),
    clsToken: get(`${imageEncoderPrefix}.cls_token`),
    norm: {
      weight: get(`${imageEncoderPrefix}.norm.weight`),
      bias: get(`${imageEncoderPrefix}.norm.bias`),
    },
    blockWeights: buildViTBlocks(imageEncoderPrefix, 24),
  };

  // --- Prediction head ---
  const predictionHead = {
    geometry: {
      weight: get('prediction_head.geometry_prediction_head.weight'),
      bias: get('prediction_head.geometry_prediction_head.bias'),
    },
    texture: {
      weight: get('prediction_head.texture_prediction_head.weight'),
      bias: get('prediction_head.texture_prediction_head.bias'),
    },
  };

  const weights = {
    patchEncoder,
    imageEncoder,
    predictionHead,
    // SPN fusion, decoder, feature_model, depth_alignment weights will be
    // wired as we implement each stage. For now, store raw tensor map for
    // incremental bring-up.
    raw: { tensors, buffer, get, tryGet, getInfo, extractTensorCPU: (name) => extractTensorCPU(buffer, getInfo(name)) },
  };

  console.log(`Loaded ${tensors.size} tensors from SHARP weight file`);
  return weights;
}
