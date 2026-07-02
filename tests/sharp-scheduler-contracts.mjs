import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  createSharpRunTelemetry,
  parseSharpSchedulerConfig,
  recordSchedulerEvent,
  schedulerYield,
  schedulerTelemetrySnapshot,
} from '../src/lib/scheduler.js';

const root = new URL('..', import.meta.url).pathname;
const schedulerPath = join(root, 'src', 'lib', 'scheduler.js');
const mainPath = join(root, 'src', 'main.js');
const spnPath = join(root, 'src', 'lib', 'spn.js');
const gaussianPath = join(root, 'src', 'lib', 'gaussian_decoder.js');

assert.ok(existsSync(schedulerPath), 'SHARP-WebGPU must expose a scheduler contract module');

const requested = {
  mode: 'cooperative',
  spnPatchChunkSize: 1,
  yieldMs: 5,
  waitForSubmittedWorkDone: true,
  gaussianPhaseYieldMs: 7,
  vitBlockChunkSize: 2,
};
const scheduler = parseSharpSchedulerConfig({ sharpScheduler: requested });
assert.equal(scheduler.schema, 'sharp-webgpu.scheduler-config.v0');
assert.equal(scheduler.requested.spnPatchChunkSize, 1);
assert.equal(scheduler.effective.spnPatchChunkSize, 1);
assert.equal(scheduler.effective.yieldMs, 5);
assert.equal(scheduler.effective.waitForSubmittedWorkDone, true);
assert.equal(scheduler.effective.gaussianPhaseYieldMs, 7);
assert.equal(scheduler.effective.vitBlockChunkSize, null, 'unfused ViT block chunking must not look effective before it is implemented');
assert.deepEqual(scheduler.unsupportedFields, ['vitBlockChunkSize']);

const telemetry = createSharpRunTelemetry(scheduler, { runId: 'contract-run' });
recordSchedulerEvent(telemetry, 'spn-patch-chunk', {
  chunkStart: 0,
  chunkEnd: 1,
  waitedForSubmittedWorkDone: true,
  yieldMs: 5,
});
const snapshot = schedulerTelemetrySnapshot(telemetry);
assert.equal(snapshot.schema, 'sharp-webgpu.scheduler-telemetry.v0');
assert.equal(snapshot.runId, 'contract-run');
assert.equal(snapshot.requestedScheduler.spnPatchChunkSize, 1);
assert.equal(snapshot.effectiveScheduler.spnPatchChunkSize, 1);
assert.deepEqual(snapshot.unsupportedFields, ['vitBlockChunkSize']);
assert.equal(snapshot.events[0].phase, 'spn-patch-chunk');
assert.notEqual(snapshot.status, 'verified', 'unsupported requested scheduler fields must not produce verified scheduler telemetry');

const proofScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    spnPatchChunkSize: 1,
    yieldMs: 5,
    waitForSubmittedWorkDone: true,
  },
});
const proofTelemetry = createSharpRunTelemetry(proofScheduler, { runId: 'proof-run' });
let queueWaitCount = 0;
await schedulerYield(
  proofScheduler,
  { queue: { onSubmittedWorkDone: async () => { queueWaitCount += 1; } } },
  proofTelemetry,
  'spn-patch-chunk',
  { chunkStart: 0, chunkEnd: 1, totalPatches: 35 }
);
const proofSnapshot = schedulerTelemetrySnapshot(proofTelemetry);
assert.equal(queueWaitCount, 1, 'schedulerYield must call queue.onSubmittedWorkDone when requested');
assert.equal(proofSnapshot.status, 'verified', 'observed queue/yield events plus matching boundary assertion may verify');
assert.equal(proofSnapshot.eventTrace.schema, 'kaminos.webgpu-scheduler-event-trace.v0');
assert.equal(proofSnapshot.eventTrace.timingAuthority, 'browser-wall-clock');
assert.ok(proofSnapshot.eventTrace.events.some(event => event.kind === 'chunk-start' && event.boundary === 'spn-patch-chunk'));
assert.ok(proofSnapshot.eventTrace.events.some(event => event.kind === 'queue-work-done-start' && event.boundary === 'spn-patch-chunk'));
assert.ok(proofSnapshot.eventTrace.events.some(event => event.kind === 'queue-work-done-end' && event.boundary === 'spn-patch-chunk'));
assert.ok(proofSnapshot.eventTrace.events.some(event => event.kind === 'js-yield-start' && event.boundary === 'spn-patch-chunk'));
assert.ok(proofSnapshot.eventTrace.events.some(event => event.kind === 'js-yield-end' && event.boundary === 'spn-patch-chunk'));
assert.deepEqual(proofSnapshot.events, proofSnapshot.eventTrace.events, 'legacy events alias must match eventTrace.events');
assert.equal(proofSnapshot.boundaryAssertions[0].field, 'phaseChunkSize.spnPatch');
assert.equal(proofSnapshot.boundaryAssertions[0].status, 'verified');
assert.equal(proofSnapshot.boundaryAssertions[0].observedBoundary, 'spn-patch-chunk');
assert.equal(proofSnapshot.boundaryAssertions[0].observedCount, 1);
assert.equal(proofSnapshot.boundaryAssertions[0].observedQueueWaitCount, 1);
assert.equal(proofSnapshot.boundaryAssertions[0].observedYieldCount, 1);

const mainSource = readFileSync(mainPath, 'utf8');
assert.match(mainSource, /parseSharpSchedulerConfig/, 'main entry must parse caller scheduler config at run time');
assert.match(mainSource, /window\.__SHARP_LAST_RUN_TELEMETRY__/, 'browser route must expose last scheduler telemetry for Kaminos');
assert.match(mainSource, /schedulerTelemetrySnapshot/, 'main entry must publish a normalized scheduler telemetry snapshot');

const spnSource = readFileSync(spnPath, 'utf8');
assert.doesNotMatch(spnSource, /const\s+CHUNK_SIZE\s*=\s*4/, 'SPN patch chunking must not be a hidden singleton constant');
assert.match(spnSource, /effective\.spnPatchChunkSize/, 'SPN patch chunking must use the effective scheduler config');
assert.match(spnSource, /spn-patch-chunk/, 'SPN must record breathing evidence around patch chunks');

const gaussianSource = readFileSync(gaussianPath, 'utf8');
assert.match(gaussianSource, /gaussianPhaseYieldMs/, 'Gaussian decoder phase breathing must use the scheduler config');
assert.match(gaussianSource, /gaussian-phase/, 'Gaussian decoder must record phase-level breathing evidence');
