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
