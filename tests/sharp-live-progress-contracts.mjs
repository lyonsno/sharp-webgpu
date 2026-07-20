import assert from 'node:assert/strict';

import * as schedulerModule from '../src/lib/scheduler.js';

assert.equal(
  typeof schedulerModule.createSharpProgressTracker,
  'function',
  'SHARP must expose one runtime progress tracker that keeps projection and liveness authority distinct',
);

let timestampMs = 1000;
const initializingTracker = schedulerModule.createSharpProgressTracker({
  now: () => timestampMs,
});
const initializing = initializingTracker.emitRouteProgress(0.01, 'SHARP is initializing WebGPU.', {
  phase: 'initializing',
});
assert.equal(
  initializing.livenessAuthority,
  'no-scheduler-boundary-observed',
  'route setup must not impersonate a completed scheduler boundary before inference work reports one',
);
const forgedLifecycle = initializingTracker.emitRouteProgress(0.02, 'Truthful lifecycle message.', {
  phase: 'source-load',
  schema: 'forged.progress.v9',
  progress: 0.99,
  message: 'Forged completion.',
  progressAuthority: 'wall-time',
  completionAuthority: 'wall-time',
  livenessAuthority: 'completed-scheduler-boundary',
  livenessOrdinal: 99,
  timestampMs: -1,
});
assert.equal(forgedLifecycle.schema, 'sharp-webgpu.progress.v0');
assert.equal(forgedLifecycle.progress, 0.02);
assert.equal(forgedLifecycle.message, 'Truthful lifecycle message.');
assert.equal(forgedLifecycle.progressAuthority, 'stage-weighted-work-projection');
assert.equal(forgedLifecycle.completionAuthority, 'not-wall-time');
assert.equal(forgedLifecycle.livenessAuthority, 'no-scheduler-boundary-observed');
assert.equal(forgedLifecycle.livenessOrdinal, 0);
assert.equal(forgedLifecycle.timestampMs, 1000);
assert.equal(forgedLifecycle.phase, 'source-load', 'lawful caller phase metadata must remain compatible');

const tracker = schedulerModule.createSharpProgressTracker({
  now: () => timestampMs,
});

const first = tracker.reportSchedulerBoundary({
  phase: 'vit-block-microphase',
  boundary: 'after-submit',
  details: {
    encoder: 'patch',
    patchIndex: 17,
    blockIndex: 12,
    totalBlocks: 24,
    microphase: 'attention-scores',
    microdutyMode: 'dispatch-major',
  },
});

assert.equal(first.schema, 'sharp-webgpu.progress.v0');
assert.equal(first.progressAuthority, 'stage-weighted-work-projection');
assert.equal(first.completionAuthority, 'not-wall-time');
assert.equal(first.livenessAuthority, 'completed-scheduler-boundary');
assert.equal(first.livenessOrdinal, 1);
assert.equal(first.phaseWorkOrdinal, 1);
assert.equal(first.workOrdinal, 1, 'existing consumers must retain the original per-phase workOrdinal field');
assert.equal(first.phase, 'vit-block-microphase');
assert.equal(first.boundary, 'after-submit');
assert.deepEqual(first.work, {
  encoder: 'patch',
  patchIndex: 17,
  blockIndex: 12,
  totalBlocks: 24,
  microphase: 'attention-scores',
  microdutyMode: 'dispatch-major',
});
assert.match(first.message, /patch 18/i);
assert.match(first.message, /block 13\/24/i);
assert.match(first.message, /attention scores/i);

timestampMs = 1250;
const second = tracker.reportSchedulerBoundary({
  phase: 'vit-block-microphase',
  boundary: 'after-submit',
  details: {
    encoder: 'patch',
    patchIndex: 17,
    blockIndex: 12,
    totalBlocks: 24,
    microphase: 'attention-softmax',
    microdutyMode: 'dispatch-major',
  },
});

assert.ok(second.progress >= first.progress, 'projected progress must remain monotonic');
assert.equal(second.livenessOrdinal, 2, 'liveness must advance even when the displayed percentage rounds to the same value');
assert.equal(second.phaseWorkOrdinal, 2);
assert.equal(second.timestampMs, 1250);
assert.match(second.message, /attention softmax/i);

timestampMs = 1500;
const patchChunk = tracker.reportSchedulerBoundary({
  phase: 'spn-patch-chunk',
  boundary: 'after-submit',
  details: {
    chunkStart: 16,
    chunkEnd: 18,
    totalPatches: 35,
  },
});

assert.equal(patchChunk.livenessOrdinal, 3);
assert.equal(patchChunk.phaseWorkOrdinal, 1, 'phase work ordinals must remain phase-local');
assert.deepEqual(patchChunk.exactWork, {
  completed: 18,
  total: 35,
  unit: 'patch',
  authority: 'scheduler-range',
});
assert.match(patchChunk.message, /patches 18\/35/i);

let latestInvalidRange = null;
for (const [name, event] of [
  ['completion beyond total', {
    phase: 'spn-patch-chunk',
    details: { chunkStart: 34, chunkEnd: 36, totalPatches: 35 },
  }],
  ['zero total', {
    phase: 'route-tail',
    details: { processedItems: 5, totalItems: 0 },
  }],
  ['negative completion', {
    phase: 'spn-fusion',
    details: { outputStart: 0, outputEnd: -1, totalOutputItems: 10 },
  }],
  ['end before start', {
    phase: 'spn-fusion',
    details: { outputStart: 8, outputEnd: 5, totalOutputItems: 10 },
  }],
]) {
  latestInvalidRange = tracker.reportSchedulerBoundary(event);
  assert.equal(latestInvalidRange.exactWork, null, `${name} must not receive exact scheduler-range authority`);
  if (event.phase === 'spn-patch-chunk') {
    assert.doesNotMatch(
      latestInvalidRange.message,
      new RegExp(`${event.details.chunkEnd}/${event.details.totalPatches}`),
      `${name} must not survive as authoritative-looking patch progress text`,
    );
  }
}

const lifecycle = tracker.emitRouteProgress(0.01, 'SHARP is initializing WebGPU.', {
  phase: 'initializing',
});
assert.equal(
  lifecycle.livenessOrdinal,
  latestInvalidRange.livenessOrdinal,
  'non-boundary lifecycle projection must not impersonate another completed scheduler work item',
);
assert.equal(lifecycle.progress, latestInvalidRange.progress, 'late lifecycle projection must not move route progress backwards');

console.log('SHARP live progress contracts passed');
