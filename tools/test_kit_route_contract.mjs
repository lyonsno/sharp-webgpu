import assert from 'node:assert/strict';
import {
  addStagedSubmitStage,
  createSharpImageToSplatRouteDefinition,
  createSharpImageToSplatRouteReceipt,
  createStagedSubmitProfile,
  createWebGpuBackendIdentity,
  SHARP_IMAGE_TO_SPLAT_ROUTE_ID,
  WEBGPU_INFERENCE_KIT_VERSION,
  validateRouteReceipt,
} from '@kaminos/webgpu-inference-kit';

assert.equal(WEBGPU_INFERENCE_KIT_VERSION, '0.1.1');

const requiredStages = ['spn', 'monodepth', 'gaussian-decoder', 'compose-ply', 'output-capture'];

const definition = createSharpImageToSplatRouteDefinition({
  kernel: {
    kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
    profile: 'spn-dinov2l16-monodepth-gaussian-ply',
    commit: 'sharp-webgpu-kit-contract-smoke',
  },
});

assert.equal(SHARP_IMAGE_TO_SPLAT_ROUTE_ID, 'sharp.image-to-splat.webgpu-local.v0');
assert.equal(definition.routeId, SHARP_IMAGE_TO_SPLAT_ROUTE_ID);
assert.deepEqual(definition.requiredStages, requiredStages);
assert.deepEqual(
  definition.outputRoles.filter(output => output.required).map(output => output.role),
  ['splat-candidate', 'depth-map', 'sharp-webgpu-metadata'],
);

const backend = createWebGpuBackendIdentity({
  adapterName: 'contract-test-webgpu-adapter',
  browser: 'node-contract-smoke',
  requestedFeatures: ['timestamp-query'],
  effectiveFeatures: ['timestamp-query'],
  limits: {
    maxBufferSize: 1024,
    maxStorageBufferBindingSize: 1024,
    maxComputeInvocationsPerWorkgroup: 256,
  },
  timestampQuery: 'requested',
});

const profile = createStagedSubmitProfile({
  route: SHARP_IMAGE_TO_SPLAT_ROUTE_ID,
  timingSource: 'adapter-phase-wall-clock',
  requiredStages,
});
for (const [index, name] of requiredStages.entries()) {
  addStagedSubmitStage(profile, { name, ms: index + 1 });
}

const receipt = createSharpImageToSplatRouteReceipt({
  input: {
    artifactId: 'source-image:test',
    sha256: 'sha256-source-image',
    shape: [768, 768, 4],
  },
  outputs: {
    splat: {
      artifactId: 'splat-candidate:test',
      sha256: 'sha256-splat',
      shape: [1179648, 14],
    },
    depthMap: {
      artifactId: 'depth-map:test',
      sha256: 'sha256-depth',
      shape: [768, 768, 4],
    },
    metadata: {
      artifactId: 'sharp-metadata:test',
      sha256: 'sha256-metadata',
      shape: [1],
    },
  },
  backend,
  model: {
    revision: 'local-sharp-webgpu',
    weightsHash: 'sha256-weights',
  },
  kernel: {
    kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
    profile: 'spn-dinov2l16-monodepth-gaussian-ply',
    commit: 'sharp-webgpu-kit-contract-smoke',
  },
  profile,
});

const result = validateRouteReceipt(receipt);
assert.equal(result.ok, true, result.errors.join('; '));
assert.equal(receipt.requestedRouteId, SHARP_IMAGE_TO_SPLAT_ROUTE_ID);
assert.deepEqual(receipt.outputs.map(output => output.role), [
  'splat-candidate',
  'depth-map',
  'sharp-webgpu-metadata',
]);

console.log('SHARP kit route contract passed');
