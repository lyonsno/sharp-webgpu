import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import * as gpuModule from '../src/lib/gpu.js';

assert.equal(
  typeof gpuModule.sharpOptionalDeviceFeatures,
  'function',
  'SHARP must expose deterministic optional device-feature negotiation',
);
assert.deepEqual(
  gpuModule.sharpOptionalDeviceFeatures({ features: new Set(['shader-f16', 'timestamp-query']) }),
  ['timestamp-query'],
  'timestamp-query must be requested when the adapter supports GPU-authoritative adaptive timing',
);
assert.deepEqual(
  gpuModule.sharpOptionalDeviceFeatures({ features: new Set(['shader-f16']) }),
  [],
  'unsupported timestamp-query must remain unavailable rather than making device creation fail',
);

const adaptiveScheduler = {
  effective: {
    decoderKernelTargetDurationMs: 12,
    decoderKernelMinChunkItems: 65536,
    decoderKernelMaxChunkItems: 8388608,
  },
};
assert.throws(
  () => gpuModule.validateAdaptiveDecoderDeviceCapabilities(
    { features: new Set() },
    adaptiveScheduler,
  ),
  /adaptive decoder scheduling requires the timestamp-query device feature/,
  'adaptive scheduling must reject an injected device without timestamp-query before model work',
);
assert.deepEqual(
  gpuModule.validateAdaptiveDecoderDeviceCapabilities(
    { features: new Set(['timestamp-query']) },
    adaptiveScheduler,
  ),
  {
    adaptiveDecoderEnabled: true,
    timestampQuery: 'available',
  },
  'adaptive scheduling must admit a device carrying the negotiated timestamp-query feature',
);
assert.deepEqual(
  gpuModule.validateAdaptiveDecoderDeviceCapabilities(
    { features: new Set() },
    { effective: { decoderKernelTargetDurationMs: 0 } },
  ),
  {
    adaptiveDecoderEnabled: false,
    timestampQuery: 'unavailable',
  },
  'fixed decoder scheduling must not require timestamp-query',
);

const mainSource = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
assert.match(
  mainSource,
  /requestedFeatures:\s*featureNames\(gpu\?\.requestedFeatures\)/,
  'backend identity must preserve the exact optional features requested at device creation',
);
const adaptiveDevicePreflightIndex = mainSource.indexOf(
  'validateAdaptiveDecoderDeviceCapabilities(\n        gpu.device,\n        currentScheduler,',
);
const weightsLoadIndex = mainSource.indexOf('weights = await loadWeights(');
assert.ok(adaptiveDevicePreflightIndex >= 0, 'the product route must invoke adaptive device admission');
assert.ok(
  weightsLoadIndex > adaptiveDevicePreflightIndex,
  'the product route must reject an incompatible adaptive device before loading model weights',
);

const browserHarnessSource = readFileSync(
  new URL('../tools/test_decoder_kernel_tiling_browser.mjs', import.meta.url),
  'utf8',
);
assert.match(
  browserHarnessSource,
  /adapter\.features\.has\(['"]timestamp-query['"]\)[\s\S]{0,240}requiredFeatures:\s*\[['"]timestamp-query['"]\]/,
  'the direct adaptive browser witness must negotiate timestamp-query explicitly',
);

assert.equal(
  typeof gpuModule.validateSharpDeviceCapabilities,
  'function',
  'SHARP must expose a caller-owned device capability validator',
);

const sufficient = gpuModule.validateSharpDeviceCapabilities({
  limits: {
    maxBufferSize: 1_073_741_824,
    maxStorageBufferBindingSize: 1_073_741_824,
  },
});
assert.equal(sufficient.status, 'sufficient');
assert.equal(sufficient.schema, 'sharp-webgpu.device-capability.v0');

assert.throws(
  () => gpuModule.validateSharpDeviceCapabilities({
    limits: {
      maxBufferSize: 1_073_741_824,
      maxStorageBufferBindingSize: 134_217_728,
    },
  }),
  /SHARP device capability insufficient.*maxStorageBufferBindingSize/,
  'SHARP must fail before dispatch when an injected device cannot bind its largest tensor',
);

assert.equal(
  typeof gpuModule.copyMappedBytesCooperatively,
  'function',
  'large mapped readbacks must expose a cooperative byte-for-byte copy boundary',
);
const sourceBytes = Uint8Array.from({ length: 11 }, (_, index) => index * 17);
const copyChunks = [];
const copied = await gpuModule.copyMappedBytesCooperatively(sourceBytes, {
  chunkBytes: 4,
  onChunk: event => copyChunks.push(event),
});
assert.deepEqual(Array.from(copied), Array.from(sourceBytes));
assert.deepEqual(copyChunks.map(event => [event.startByte, event.endByte]), [[0, 4], [4, 8], [8, 11]]);

const defaultCopyChunks = [];
const defaultCopied = await gpuModule.copyMappedBytesCooperatively(sourceBytes, {
  chunkBytes: 0,
  onChunk: event => defaultCopyChunks.push(event),
});
assert.deepEqual(Array.from(defaultCopied), Array.from(sourceBytes));
assert.deepEqual(
  defaultCopyChunks.map(event => [event.startByte, event.endByte]),
  [[0, sourceBytes.byteLength]],
  'zero chunk size must mean one unchunked copy, not one awaited callback per byte',
);
