import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  SHARP_IMAGE_TO_SPLAT_ROUTE_ID,
  WEBGPU_INFERENCE_KIT_VERSION,
  WEBGPU_INFERENCE_RUNTIME_SCHEMA,
  WEBGPU_RUNTIME_PROFILE_SCHEMA,
  createRouteInvocationRequest,
  createRouteWorkerResult,
  createSharpImageToSplatRouteDefinition,
  createSharpImageToSplatRouteReceipt,
  validateRouteWorkerResult,
  validateWebGpuRuntimeProfile,
} from '@kaminos/webgpu-inference-kit';
import {
  SHARP_ROUTE_RUNTIME_LABEL,
  createSharpRouteRuntime,
  finishSharpRouteRuntimeProfile,
} from '../src/lib/route_runtime.js';

const [kitMajor, kitMinor, kitPatch] = WEBGPU_INFERENCE_KIT_VERSION.split('.').map(Number);
assert.deepEqual([kitMajor, kitMinor], [0, 1]);
assert.ok(kitPatch >= 9, `SHARP route runtime bridge requires kit >=0.1.9, got ${WEBGPU_INFERENCE_KIT_VERSION}`);

const definition = createSharpImageToSplatRouteDefinition({
  kernel: {
    kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
    profile: 'spn-dinov2l16-monodepth-gaussian-ply',
    commit: 'sharp-webgpu-route-runtime-contract',
  },
});

const fakeQueue = {
  submit() {},
  async onSubmittedWorkDone() {},
};
const fakeDevice = {
  queue: fakeQueue,
  features: new Set(['webgpu-core']),
  limits: {
    maxBufferSize: 1024,
    maxStorageBufferBindingSize: 1024,
    maxComputeInvocationsPerWorkgroup: 256,
  },
};
const fakeAdapter = {
  info: {
    description: 'contract-test-webgpu-adapter',
  },
  features: new Set(['webgpu-core']),
  limits: fakeDevice.limits,
};

let nowMs = 100;
const runtime = await createSharpRouteRuntime({
  device: fakeDevice,
  adapter: fakeAdapter,
}, {
  routeDefinition: definition,
  browser: 'node-sharp-runtime-contract',
  now: () => {
    nowMs += 5;
    return nowMs;
  },
});

assert.equal(runtime.schema, WEBGPU_INFERENCE_RUNTIME_SCHEMA);
assert.equal(runtime.routeId, SHARP_IMAGE_TO_SPLAT_ROUTE_ID);
assert.equal(runtime.runtimeLabel, SHARP_ROUTE_RUNTIME_LABEL);
assert.deepEqual(runtime.profile.requiredStages, definition.requiredStages);

for (const stageName of definition.requiredStages) {
  const result = await runtime.runStage(stageName, async stage => {
    assert.equal(typeof stage.yieldToBrowser, 'function');
    return `${stageName}:ok`;
  }, {
    routeStage: stageName,
  });
  assert.equal(result, `${stageName}:ok`);
}

const runtimeProfile = finishSharpRouteRuntimeProfile(runtime);
assert.equal(runtimeProfile.schema, WEBGPU_RUNTIME_PROFILE_SCHEMA);
assert.equal(runtimeProfile.routeId, SHARP_IMAGE_TO_SPLAT_ROUTE_ID);
assert.equal(runtimeProfile.runtimeLabel, SHARP_ROUTE_RUNTIME_LABEL);
assert.equal(runtimeProfile.timingSource, definition.timingSource);
assert.equal(runtimeProfile.profile.timingSource, definition.timingSource);
assert.deepEqual(runtimeProfile.profile.stageNames, definition.requiredStages);
assert.equal(runtimeProfile.profile.stages.length, definition.requiredStages.length);
assert.equal(runtimeProfile.profile.stages[0].metadata.routeStage, 'spn');
assert.equal(runtimeProfile.evidence.mode, 'live');
assert.equal(runtimeProfile.evidence.source, 'sharp-webgpu-browser-route');
assert.deepEqual(validateWebGpuRuntimeProfile(runtimeProfile), { ok: true, errors: [] });

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
      artifactId: 'sharp-webgpu-metadata:test',
      sha256: 'sha256-metadata',
      shape: [1],
    },
  },
  backend: runtimeProfile.backend,
  model: {
    revision: 'local-sharp-webgpu',
    weightsHash: 'sha256-weights',
  },
  kernel: definition.kernel,
  profile: runtimeProfile.profile,
});
receipt.metadataPayload = {
  routeTailTimings: [
    {
      stage: 'compose-ply',
      step: 'compose-export',
      ms: 12.5,
    },
  ],
};
const request = createRouteInvocationRequest(definition, {
  requestId: 'sharp-route-runtime-contract',
  inputs: {
    'source-image': {
      artifactId: 'source-image:test',
      sha256: 'sha256-source-image',
      shape: [768, 768, 4],
    },
  },
  outputs: {
    'splat-candidate': {
      artifactId: 'splat-candidate:test',
      shape: [1179648, 14],
    },
    'depth-map': {
      artifactId: 'depth-map:test',
      shape: [768, 768, 4],
    },
    'sharp-webgpu-metadata': {
      artifactId: 'sharp-webgpu-metadata:test',
      shape: [1],
    },
  },
});
const workerResult = createRouteWorkerResult(definition, { request, receipt });
assert.equal(workerResult.receipt.metadataPayload.routeTailTimings[0].stage, 'compose-ply');
assert.equal(workerResult.receipt.metadataPayload.routeTailTimings[0].step, 'compose-export');
assert.deepEqual(
  validateRouteWorkerResult(workerResult, definition),
  { ok: true, errors: [] },
  'runtime-backed SHARP receipt must satisfy the stricter route worker contract'
);

const root = new URL('..', import.meta.url).pathname;
const mainSource = readFileSync(join(root, 'src', 'main.js'), 'utf8');
assert.match(mainSource, /createSharpRouteRuntime/, 'browser route must create the SHARP kit runtime wrapper');
assert.match(mainSource, /runRouteStage\([^)]*'spn'/s, 'SPN stage must execute through the kit runtime stage wrapper');
assert.match(mainSource, /runRouteStage\([^)]*'monodepth'/s, 'monodepth stage must execute through the kit runtime stage wrapper');
assert.match(mainSource, /runRouteStage\([^)]*'output-capture'/s, 'output-capture stage must execute through the kit runtime stage wrapper');
assert.match(mainSource, /runRouteStage\([^)]*'gaussian-decoder'/s, 'Gaussian decoder stage must execute through the kit runtime stage wrapper');
assert.match(mainSource, /runRouteStage\([^)]*'compose-ply'/s, 'compose-ply stage must execute through the kit runtime stage wrapper');
assert.match(mainSource, /finishSharpRouteRuntimeProfile/, 'browser route must finish and expose the kit runtime profile');
assert.match(mainSource, /runtimeProfile:\s*null/, 'run debug must expose runtimeProfile without inventing it before execution');

console.log('SHARP route runtime contract passed');
