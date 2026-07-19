import assert from 'node:assert/strict';

import * as gpuModule from '../src/lib/gpu.js';

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
