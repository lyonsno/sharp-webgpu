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
const backbonePath = join(root, 'src', 'lib', 'backbone.js');
const gaussianPath = join(root, 'src', 'lib', 'gaussian_decoder.js');
const contentionWitnessPath = join(root, 'tools', 'contention_witness.mjs');

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
assert.equal(scheduler.effective.vitBlockChunkSize, 2, 'requested ViT block chunking must become effective scheduler config');
assert.deepEqual(scheduler.unsupportedFields, []);

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
assert.deepEqual(snapshot.unsupportedFields, []);
assert.equal(snapshot.events[0].phase, 'spn-patch-chunk');
assert.equal(snapshot.status, 'scheduler-unverified', 'requested ViT block chunking must not verify without observed ViT block boundaries');
const missingVitAssertion = snapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseChunkSize.vitBlock');
assert.ok(missingVitAssertion, 'requested ViT block chunking must produce a boundary assertion');
assert.equal(missingVitAssertion.status, 'unverified');
assert.equal(missingVitAssertion.observedBoundary, 'vit-block-chunk');
assert.equal(missingVitAssertion.observedYieldCount, 0);

const uncappedTimingScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    spnPatchChunkSize: 1,
    yieldMs: 5000,
    gaussianPhaseYieldMs: 5000,
  },
});
assert.equal(uncappedTimingScheduler.requested.yieldMs, 5000, 'requested yieldMs must preserve caller intent');
assert.equal(uncappedTimingScheduler.effective.yieldMs, 5000, 'effective yieldMs must not silently cap below caller intent');
assert.equal(uncappedTimingScheduler.requested.gaussianPhaseYieldMs, 5000, 'requested Gaussian yield must preserve caller intent');
assert.equal(uncappedTimingScheduler.effective.gaussianPhaseYieldMs, 5000, 'effective Gaussian yield must not silently cap below caller intent');
assert.deepEqual(uncappedTimingScheduler.unsupportedFields, [], 'supported timing fields must not be laundered through hidden caps');

const defaultScheduler = parseSharpSchedulerConfig();
const defaultTelemetry = createSharpRunTelemetry(defaultScheduler, { runId: 'default-yield-run' });
let defaultTimerFired = false;
setTimeout(() => { defaultTimerFired = true; }, 0);
await schedulerYield(
  defaultScheduler,
  {},
  defaultTelemetry,
  'spn-patch-chunk',
  { chunkStart: 0, chunkEnd: defaultScheduler.effective.spnPatchChunkSize, totalPatches: 35 }
);
const defaultYieldSnapshot = schedulerTelemetrySnapshot(defaultTelemetry);
assert.equal(defaultTimerFired, true, 'default SPN chunk boundary must preserve an actual task yield');
assert.equal(defaultYieldSnapshot.boundaryAssertions[0].observedYieldCount, 1, 'default SPN chunk proof must record the preserved task yield');

const partialDefaultTelemetry = createSharpRunTelemetry(defaultScheduler, { runId: 'partial-default-run' });
recordSchedulerEvent(partialDefaultTelemetry, 'spn-patch-chunk', {
  boundary: 'spn-patch-chunk',
  kind: 'chunk-start',
});
const partialDefaultSnapshot = schedulerTelemetrySnapshot(partialDefaultTelemetry);
assert.equal(partialDefaultSnapshot.status, 'scheduler-unverified', 'default SPN scheduler proof must reject chunk telemetry without observed JS yield');
assert.equal(partialDefaultSnapshot.boundaryAssertions[0].status, 'unverified');
assert.equal(partialDefaultSnapshot.boundaryAssertions[0].observedYieldCount, 0);

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

const missingQueueTelemetry = createSharpRunTelemetry(proofScheduler, { runId: 'missing-queue-run' });
recordSchedulerEvent(missingQueueTelemetry, 'spn-patch-chunk', {
  boundary: 'spn-patch-chunk',
  kind: 'chunk-start',
});
recordSchedulerEvent(missingQueueTelemetry, 'spn-patch-chunk', {
  boundary: 'spn-patch-chunk',
  kind: 'js-yield-start',
  yieldMs: 5,
});
recordSchedulerEvent(missingQueueTelemetry, 'spn-patch-chunk', {
  boundary: 'spn-patch-chunk',
  kind: 'js-yield-end',
  yieldMs: 5,
});
const missingQueueSnapshot = schedulerTelemetrySnapshot(missingQueueTelemetry);
assert.equal(missingQueueSnapshot.status, 'scheduler-unverified', 'requested queue wait must keep telemetry unverified when queue events are absent');
assert.equal(missingQueueSnapshot.boundaryAssertions[0].status, 'unverified');
assert.equal(missingQueueSnapshot.boundaryAssertions[0].observedQueueWaitCount, 0);

const missingYieldTelemetry = createSharpRunTelemetry(proofScheduler, { runId: 'missing-yield-run' });
recordSchedulerEvent(missingYieldTelemetry, 'spn-patch-chunk', {
  boundary: 'spn-patch-chunk',
  kind: 'chunk-start',
});
recordSchedulerEvent(missingYieldTelemetry, 'spn-patch-chunk', {
  boundary: 'spn-patch-chunk',
  kind: 'queue-work-done-start',
});
recordSchedulerEvent(missingYieldTelemetry, 'spn-patch-chunk', {
  boundary: 'spn-patch-chunk',
  kind: 'queue-work-done-end',
});
const missingYieldSnapshot = schedulerTelemetrySnapshot(missingYieldTelemetry);
assert.equal(missingYieldSnapshot.status, 'scheduler-unverified', 'requested JS yield must keep telemetry unverified when yield events are absent');
assert.equal(missingYieldSnapshot.boundaryAssertions[0].status, 'unverified');
assert.equal(missingYieldSnapshot.boundaryAssertions[0].observedYieldCount, 0);

const gaussianProofScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    spnPatchChunkSize: 1,
    gaussianPhaseYieldMs: 7,
  },
});
const missingGaussianTelemetry = createSharpRunTelemetry(gaussianProofScheduler, { runId: 'missing-gaussian-run' });
recordSchedulerEvent(missingGaussianTelemetry, 'spn-patch-chunk', {
  boundary: 'spn-patch-chunk',
  kind: 'chunk-start',
});
const missingGaussianSnapshot = schedulerTelemetrySnapshot(missingGaussianTelemetry);
const missingGaussianAssertion = missingGaussianSnapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseYieldMs.gaussianPhase');
assert.equal(missingGaussianSnapshot.status, 'scheduler-unverified', 'requested Gaussian phase yield must keep telemetry unverified when Gaussian events are absent');
assert.ok(missingGaussianAssertion, 'requested Gaussian phase yield must produce a boundary assertion');
assert.equal(missingGaussianAssertion.status, 'unverified');
assert.equal(missingGaussianAssertion.observedBoundary, 'gaussian-phase');
assert.equal(missingGaussianAssertion.observedYieldCount, 0);

const gaussianProofTelemetry = createSharpRunTelemetry(gaussianProofScheduler, { runId: 'gaussian-proof-run' });
await schedulerYield(
  gaussianProofScheduler,
  {},
  gaussianProofTelemetry,
  'spn-patch-chunk',
  { chunkStart: 0, chunkEnd: gaussianProofScheduler.effective.spnPatchChunkSize, totalPatches: 35 }
);
await schedulerYield(
  gaussianProofScheduler,
  {},
  gaussianProofTelemetry,
  'gaussian-phase',
  { phase: 'prediction-head' },
  gaussianProofScheduler.effective.gaussianPhaseYieldMs
);
const gaussianProofSnapshot = schedulerTelemetrySnapshot(gaussianProofTelemetry);
const gaussianProofAssertion = gaussianProofSnapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseYieldMs.gaussianPhase');
assert.equal(gaussianProofSnapshot.status, 'verified', 'observed SPN chunk and Gaussian phase yield events may verify');
assert.equal(gaussianProofAssertion.status, 'verified');
assert.equal(gaussianProofAssertion.observedYieldCount, 1);

const vitProofScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    spnPatchChunkSize: 1,
    vitBlockChunkSize: 2,
    yieldMs: 0,
  },
});
assert.equal(vitProofScheduler.effective.vitBlockChunkSize, 2, 'ViT block chunking must be effective independently of nonzero yieldMs');
assert.deepEqual(vitProofScheduler.unsupportedFields, [], 'ViT block chunking must not be reported as unsupported');
const vitProofTelemetry = createSharpRunTelemetry(vitProofScheduler, { runId: 'vit-proof-run' });
await schedulerYield(
  vitProofScheduler,
  {},
  vitProofTelemetry,
  'spn-patch-chunk',
  { chunkStart: 0, chunkEnd: 1, totalPatches: 35 }
);
await schedulerYield(
  vitProofScheduler,
  {},
  vitProofTelemetry,
  'vit-block-chunk',
  { encoder: 'patch', blockStart: 0, blockEnd: 2, totalBlocks: 24, tokenCount: 577 }
);
const vitProofSnapshot = schedulerTelemetrySnapshot(vitProofTelemetry);
const vitProofAssertion = vitProofSnapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseChunkSize.vitBlock');
assert.equal(vitProofSnapshot.status, 'verified', 'observed SPN and ViT block chunk yields may verify requested scheduler boundaries');
assert.equal(vitProofAssertion.status, 'verified');
assert.equal(vitProofAssertion.effective, 2);
assert.equal(vitProofAssertion.observedBoundary, 'vit-block-chunk');
assert.equal(vitProofAssertion.observedCount, 1);
assert.equal(vitProofAssertion.observedYieldCount, 1);

const missingVitTelemetry = createSharpRunTelemetry(vitProofScheduler, { runId: 'missing-vit-run' });
await schedulerYield(
  vitProofScheduler,
  {},
  missingVitTelemetry,
  'spn-patch-chunk',
  { chunkStart: 0, chunkEnd: 1, totalPatches: 35 }
);
const missingVitSnapshot = schedulerTelemetrySnapshot(missingVitTelemetry);
const missingVitProofAssertion = missingVitSnapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseChunkSize.vitBlock');
assert.equal(missingVitSnapshot.status, 'scheduler-unverified', 'requested ViT block chunking must remain unverified when only patch chunk events are observed');
assert.equal(missingVitProofAssertion.status, 'unverified');
assert.equal(missingVitProofAssertion.observedCount, 0);
assert.equal(missingVitProofAssertion.observedYieldCount, 0);

const mainSource = readFileSync(mainPath, 'utf8');
assert.match(mainSource, /parseSharpSchedulerConfig/, 'main entry must parse caller scheduler config at run time');
assert.match(mainSource, /window\.__SHARP_LAST_RUN_TELEMETRY__/, 'browser route must expose last scheduler telemetry for Kaminos');
assert.match(mainSource, /schedulerTelemetrySnapshot/, 'main entry must publish a normalized scheduler telemetry snapshot');
assert.match(mainSource, /scheduler:\s*sharpRouteDefinition\.scheduler/, 'route debug scheduler must preserve the route scheduler receipt shape for contention witnesses');
assert.match(mainSource, /sharpScheduler:\s*null/, 'route debug must expose the SHARP scheduler config on a distinct field');
assert.doesNotMatch(mainSource, /runDebug\.scheduler\s*=\s*currentScheduler/, 'SHARP scheduler config must not overwrite the route scheduler debug shape');
assert.match(mainSource, /runDebug\.sharpScheduler\s*=\s*currentScheduler/, 'main entry must bind the per-run SHARP scheduler config to sharpScheduler');

const spnSource = readFileSync(spnPath, 'utf8');
assert.doesNotMatch(spnSource, /const\s+CHUNK_SIZE\s*=\s*4/, 'SPN patch chunking must not be a hidden singleton constant');
assert.match(spnSource, /effective\.spnPatchChunkSize/, 'SPN patch chunking must use the effective scheduler config');
assert.match(spnSource, /spn-patch-chunk/, 'SPN must record breathing evidence around patch chunks');

const backboneSource = readFileSync(backbonePath, 'utf8');
assert.match(backboneSource, /schedulerYield/, 'ViT encoder must use the scheduler yield primitive');
assert.match(backboneSource, /effective\.vitBlockChunkSize/, 'ViT encoder must use the effective scheduler block chunk size');
assert.match(backboneSource, /vit-block-chunk/, 'ViT encoder must record breathing evidence around block chunks');

const gaussianSource = readFileSync(gaussianPath, 'utf8');
assert.match(gaussianSource, /gaussianPhaseYieldMs/, 'Gaussian decoder phase breathing must use the scheduler config');
assert.match(gaussianSource, /gaussian-phase/, 'Gaussian decoder must record phase-level breathing evidence');

const contentionWitnessSource = readFileSync(contentionWitnessPath, 'utf8');
assert.match(contentionWitnessSource, /--sharp-scheduler/, 'contention witness must expose the SHARP scheduler query config as an invocation parameter');
assert.match(contentionWitnessSource, /searchParams\.set\('sharpScheduler'/, 'contention witness must pass the requested scheduler to the browser route URL');
