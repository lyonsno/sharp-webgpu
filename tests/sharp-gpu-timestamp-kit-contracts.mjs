import assert from 'node:assert/strict';

import {
  WEBGPU_INFERENCE_KIT_VERSION,
  createWebGpuAdaptiveCommandDutyPlanner,
} from '@kaminos/webgpu-inference-kit';

assert.equal(
  WEBGPU_INFERENCE_KIT_VERSION,
  '0.1.46',
  'the Chrome 151 assay must exercise the exact canonical producer kit',
);

function planner() {
  return createWebGpuAdaptiveCommandDutyPlanner({
    plannerId: 'sharp:decoder:gpu-timestamp-consumer',
    unit: 'output-item',
    totalItems: 100,
    initialChunkItems: 20,
    targetDurationMs: 10,
    adjustmentGain: 1,
    bounds: {
      minChunkItems: 5,
      maxChunkItems: 100,
    },
    retention: 'uncapped',
    metadata: {
      routeId: 'sharp.image-to-splat.webgpu-local.v0',
      timingSourceSchema: 'sharp-webgpu.gpu-timestamp-range.v0',
    },
  });
}

const adaptive = planner();
const firstRange = adaptive.nextRange();
const gpuObservation = adaptive.observeRange({
  rangeId: firstRange.rangeId,
  observedDurationMs: 20,
  timingAuthority: 'gpu-timestamp-query',
});

assert.equal(gpuObservation.status, 'range-observed');
assert.equal(gpuObservation.timingAuthority, 'gpu-timestamp-query');
assert.equal(gpuObservation.observedDurationMs, 20);
assert.equal(gpuObservation.nextChunkItems, 10);
assert.equal(adaptive.snapshot().ranges[0].timingAuthority, 'gpu-timestamp-query');

const zeroDuration = planner();
const zeroRange = zeroDuration.nextRange();
assert.throws(
  () => zeroDuration.observeRange({
    rangeId: zeroRange.rangeId,
    observedDurationMs: 0,
    timingAuthority: 'gpu-timestamp-query',
  }),
  /gpu-timestamp-query.*greater than zero|observedDurationMs.*greater than zero/,
  'GPU timestamp observations must preserve the timer contract of an ordered positive range',
);
assert.equal(zeroDuration.snapshot().completedItems, 0);

const falseAuthority = planner();
const falseAuthorityRange = falseAuthority.nextRange();
assert.throws(
  () => falseAuthority.observeRange({
    rangeId: falseAuthorityRange.rangeId,
    observedDurationMs: 20,
    timingAuthority: 'wall-clock-around-yield',
  }),
  /timingAuthority/,
  'unmeasured wall-clock authority must not advance adaptive coverage',
);
assert.equal(falseAuthority.snapshot().completedItems, 0);

console.log('SHARP GPU timestamp kit contracts passed');
