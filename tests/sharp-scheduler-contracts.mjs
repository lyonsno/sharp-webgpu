import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  attachSharpLiveScheduler,
  createSharpRunTelemetry,
  createSharpRuntimeDutyMap,
  parseSharpSchedulerConfig,
  recordSchedulerEvent,
  schedulerYield,
  schedulerTelemetrySnapshot,
} from '../src/lib/scheduler.js';
import * as schedulerModule from '../src/lib/scheduler.js';

const root = new URL('..', import.meta.url).pathname;
const schedulerPath = join(root, 'src', 'lib', 'scheduler.js');
const mainPath = join(root, 'src', 'main.js');
const spnPath = join(root, 'src', 'lib', 'spn.js');
const monodepthPath = join(root, 'src', 'lib', 'monodepth.js');
const backbonePath = join(root, 'src', 'lib', 'backbone.js');
const gaussianPath = join(root, 'src', 'lib', 'gaussian_decoder.js');
const composePath = join(root, 'src', 'lib', 'compose.js');
const shaderOpsPath = join(root, 'src', 'lib', 'shader_ops.js');
const convTransposeShaderPath = join(root, 'src', 'shaders', 'conv_transpose2d.wgsl');
const concatChannelsShaderPath = join(root, 'src', 'shaders', 'concat_channels.wgsl');
const tokenPatchMergeShaderPath = join(root, 'src', 'shaders', 'token_patch_merge.wgsl');
const gaussianInitializerShaderPath = join(root, 'src', 'shaders', 'gaussian_initializer_feature_input.wgsl');
const gaussianInitializerReduceShaderPath = join(root, 'src', 'shaders', 'gaussian_initializer_reduce_min.wgsl');
const contentionWitnessPath = join(root, 'tools', 'contention_witness.mjs');

assert.ok(existsSync(schedulerPath), 'SHARP-WebGPU must expose a scheduler contract module');
assert.equal(
  typeof schedulerModule.schedulerTelemetrySnapshotCooperatively,
  'function',
  'route finalization must expose a task-yielding snapshot for uncapped telemetry',
);
const schedulerSource = readFileSync(schedulerPath, 'utf8');
assert.doesNotMatch(
  schedulerSource,
  /return JSON\.parse\(JSON\.stringify\(telemetry\)\)/,
  'uncapped telemetry snapshots must not serialize the whole event corpus on UI-critical route finalization',
);

const requested = {
  mode: 'cooperative',
  spnPatchChunkSize: 1,
  yieldMs: 5,
  waitForSubmittedWorkDone: true,
  gaussianPhaseYieldMs: 7,
  vitBlockChunkSize: 2,
  spnFusionChunkItems: 2097152,
};
const scheduler = parseSharpSchedulerConfig({ sharpScheduler: requested });
assert.equal(scheduler.schema, 'sharp-webgpu.scheduler-config.v0');
assert.equal(scheduler.requested.spnPatchChunkSize, 1);
assert.equal(scheduler.effective.spnPatchChunkSize, 1);
assert.equal(scheduler.effective.yieldMs, 5);
assert.equal(scheduler.effective.waitForSubmittedWorkDone, true);
assert.equal(scheduler.effective.gaussianPhaseYieldMs, 7);
assert.equal(scheduler.effective.vitBlockChunkSize, 2, 'requested ViT block chunking must become effective scheduler config');
assert.equal(scheduler.requested.spnFusionChunkItems, 2097152, 'requested SPN fusion chunking must remain visible');
assert.equal(scheduler.effective.spnFusionChunkItems, 2097152, 'requested SPN fusion chunking must become effective scheduler config');
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
assert.equal(snapshot.eventTrace.clock.clockId, 'sharp-webgpu-performance-clock');
assert.equal(snapshot.eventTrace.clock.source, 'performance.now');
assert.equal(snapshot.requestedScheduler.spnPatchChunkSize, 1);
assert.equal(snapshot.effectiveScheduler.spnPatchChunkSize, 1);
assert.deepEqual(snapshot.unsupportedFields, []);
assert.equal(snapshot.events[0].phase, 'spn-patch-chunk');
const snapshotEventPhase = snapshot.events[0].phase;
telemetry.eventTrace.events[0].phase = 'mutated-after-snapshot';
assert.equal(snapshot.events[0].phase, snapshotEventPhase, 'snapshot events must remain isolated from live telemetry mutation');
telemetry.eventTrace.events[0].phase = snapshotEventPhase;
assert.equal(snapshot.events, snapshot.eventTrace.events, 'snapshot event aliases must share one isolated event corpus');
assert.equal(snapshot.status, 'scheduler-unverified', 'requested ViT block chunking must not verify without observed ViT block boundaries');
const missingVitAssertion = snapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseChunkSize.vitBlock');
assert.ok(missingVitAssertion, 'requested ViT block chunking must produce a boundary assertion');
assert.equal(missingVitAssertion.status, 'unverified');
assert.equal(missingVitAssertion.observedBoundary, 'vit-block-chunk');
assert.equal(missingVitAssertion.observedYieldCount, 0);

const cooperativeSnapshotTelemetry = createSharpRunTelemetry(scheduler, { runId: 'cooperative-snapshot-run' });
for (let index = 0; index < 1025; index += 1) {
  recordSchedulerEvent(cooperativeSnapshotTelemetry, 'route-tail', {
    kind: 'chunk-start',
    role: 'cpu-materialization-chunk',
    workOrdinal: index,
  });
}
const cooperativeSourceEventCount = cooperativeSnapshotTelemetry.events.length;
let cooperativeTaskYieldCount = 0;
let injectedAfterSnapshotStart = false;
const cooperativeSnapshot = await schedulerModule.schedulerTelemetrySnapshotCooperatively(
  cooperativeSnapshotTelemetry,
  'verified',
  {
    chunkEvents: 128,
    taskYield: async () => {
      cooperativeTaskYieldCount += 1;
      if (!injectedAfterSnapshotStart) {
        injectedAfterSnapshotStart = true;
        recordSchedulerEvent(cooperativeSnapshotTelemetry, 'route-tail', { kind: 'post-snapshot-start' });
      }
      await new Promise(resolve => setTimeout(resolve, 0));
    },
  },
);
assert.ok(cooperativeTaskYieldCount > 0, 'large telemetry snapshots must donate real browser tasks between bounded clone batches');
assert.equal(cooperativeSnapshot.snapshotProcess.mode, 'cooperative-fixed-prefix');
assert.equal(cooperativeSnapshot.snapshotProcess.taskYieldCount, cooperativeTaskYieldCount);
assert.equal(cooperativeSnapshot.events.length, cooperativeSourceEventCount, 'snapshot must preserve the fixed event prefix present at invocation');
assert.equal(cooperativeSnapshotTelemetry.events.length, cooperativeSourceEventCount + 1, 'live telemetry may continue after snapshot invocation');
cooperativeSnapshotTelemetry.events[0].phase = 'mutated-after-cooperative-snapshot';
assert.equal(cooperativeSnapshot.events[0].phase, 'route-tail', 'cooperative snapshot events must remain isolated from live telemetry mutation');

const providerFailureScheduler = parseSharpSchedulerConfig({ sharpScheduler: requested });
const providerFailureTelemetry = createSharpRunTelemetry(providerFailureScheduler, { runId: 'provider-failure-run' });
attachSharpLiveScheduler(providerFailureScheduler, {
  runtime: {
    device: {},
    queue: {},
    requestForegroundOpportunity() {
      return { requestId: 'foreground-request-1', completion: Promise.resolve({}) };
    },
    foregroundOpportunitySnapshot() {
      return { pendingRequestCount: 1 };
    },
    foregroundOpportunities: {
      async serviceAtBoundary() {
        return {
          status: 'failed',
          capturedRequestCount: 1,
          servicedRequestCount: 0,
          failures: [{ failure: { error: { message: 'foreground opportunity service failed' } } }],
        };
      },
    },
  },
  invocation: {
    bounds: { phaseChunkSize: { spnPatch: { min: 1, max: 35, stepFactor: 2 } } },
    getControl: () => 1,
  },
  stage: 'spn',
  foregroundOpportunityHook: () => ({
    requestId: 'foreground-request-1',
    metadata: {},
    run: async () => ({}),
  }),
});
await assert.rejects(
  () => schedulerYield(providerFailureScheduler, {}, providerFailureTelemetry, 'spn-patch-chunk'),
  error => error?.message === 'foreground opportunity service failed',
  'the provider error path must preserve the actual foreground failure instead of throwing a scope ReferenceError',
);

const lateStageScheduler = parseSharpSchedulerConfig({ sharpScheduler: requested });
const lateStageTelemetry = createSharpRunTelemetry(lateStageScheduler, { runId: 'late-stage-foreground-run' });
const lateStageRequests = [];
const lateStageServices = [];
const lateStageProgress = [];
Object.defineProperty(lateStageScheduler, 'progressReporter', {
  value: event => lateStageProgress.push(event),
  configurable: true,
});
attachSharpLiveScheduler(lateStageScheduler, {
  runtime: {
    device: {},
    queue: {},
    requestForegroundOpportunity(input) {
      lateStageRequests.push(input);
      return {
        requestId: 'late-stage-request-1',
        completion: Promise.resolve({
          requestId: 'late-stage-request-1',
          status: 'completed',
          submissionCount: 1,
        }),
      };
    },
    foregroundOpportunitySnapshot() {
      return { pendingRequestCount: 1 };
    },
    foregroundOpportunities: {
      async serviceAtBoundary(descriptor) {
        lateStageServices.push(descriptor);
        return {
          status: 'completed',
          capturedRequestCount: 1,
          servicedRequestCount: 1,
        };
      },
    },
  },
  invocation: {
    invocationId: 'late-stage-invocation',
    bounds: { phaseChunkSize: { spnPatch: { min: 1, max: 35, stepFactor: 2 } } },
    getControl: () => 1,
  },
  stage: 'monodepth',
  foregroundOpportunityHook: () => ({
    requestId: 'late-stage-request-1',
    metadata: {},
    run: async () => ({}),
  }),
});
await schedulerYield(
  lateStageScheduler,
  {},
  lateStageTelemetry,
  'monodepth-phase',
  { phase: 'head-final' },
  0,
);
assert.equal(lateStageRequests.length, 1, 'monodepth boundaries must request foreground work just like SPN controls');
assert.equal(lateStageServices.length, 1, 'monodepth boundaries must service pending foreground work before the next encode');
assert.equal(lateStageServices[0].position, 'before-encode', 'late-stage foreground service must retain its exact boundary position');
assert.equal(lateStageProgress.length, 1, 'each cooperative boundary must publish live model progress');
assert.equal(lateStageProgress[0].phase, 'monodepth-phase');
assert.equal(lateStageProgress[0].details.phase, 'head-final');

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

const uncappedFusionScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    spnFusionChunkItems: 123456789,
  },
});
assert.equal(uncappedFusionScheduler.requested.spnFusionChunkItems, 123456789, 'requested SPN fusion chunk size must preserve caller intent');
assert.equal(uncappedFusionScheduler.effective.spnFusionChunkItems, 123456789, 'effective SPN fusion chunk size must not silently cap below caller intent');

const defaultScheduler = parseSharpSchedulerConfig();
assert.equal(defaultScheduler.effective.spnFusionChunkItems, 0, 'default scheduler must preserve the existing single-dispatch SPN fusion path');
assert.equal(defaultScheduler.effective.vitMicroduty, false, 'default inference must not pay sub-block scheduling overhead without caller intent');
const cooperativeMicrodutyScheduler = parseSharpSchedulerConfig({
  sharpScheduler: { mode: 'cooperative', vitMicroduty: true },
});
assert.equal(cooperativeMicrodutyScheduler.effective.vitMicroduty, true, 'a cooperative caller must be able to opt into sub-block ViT duties explicitly');
assert.equal(typeof schedulerModule.planSpnFusionChunks, 'function', 'SPN fusion output chunk planning must be directly testable');
assert.equal(typeof schedulerModule.planNextSpnFusionChunk, 'function', 'adaptive SPN fusion must plan one exact next range from the current control');
assert.equal(typeof schedulerModule.planNextVitBlockChunk, 'function', 'adaptive ViT dispatch must plan one exact next block range from the current control');
assert.equal(typeof schedulerModule.planVitBlockMicroduties, 'function', 'one ViT block must expose ordered attention and MLP microduties');
if (typeof schedulerModule.planSpnFusionChunks === 'function') {
  assert.deepEqual(
    schedulerModule.planSpnFusionChunks(10, 0),
    [{ chunkIndex: 0, chunkCount: 1, outputStart: 0, outputEnd: 10, outputCount: 10 }],
    'disabled SPN fusion chunking must preserve one exact output range'
  );
  assert.deepEqual(
    schedulerModule.planSpnFusionChunks(10, 4),
    [
      { chunkIndex: 0, chunkCount: 3, outputStart: 0, outputEnd: 4, outputCount: 4 },
      { chunkIndex: 1, chunkCount: 3, outputStart: 4, outputEnd: 8, outputCount: 4 },
      { chunkIndex: 2, chunkCount: 3, outputStart: 8, outputEnd: 10, outputCount: 2 },
    ],
    'SPN fusion chunk ranges must cover the output exactly once including the remainder'
  );
}
if (typeof schedulerModule.planNextSpnFusionChunk === 'function') {
  const firstAdaptiveRange = schedulerModule.planNextSpnFusionChunk(10, 0, 4, 0);
  const reducedAdaptiveRange = schedulerModule.planNextSpnFusionChunk(10, firstAdaptiveRange.outputEnd, 2, 1);
  const relaxedAdaptiveRange = schedulerModule.planNextSpnFusionChunk(10, reducedAdaptiveRange.outputEnd, 4, 2);
  assert.deepEqual(
    [firstAdaptiveRange, reducedAdaptiveRange, relaxedAdaptiveRange],
    [
      { chunkIndex: 0, projectedChunkCount: 3, outputStart: 0, outputEnd: 4, outputCount: 4 },
      { chunkIndex: 1, projectedChunkCount: 4, outputStart: 4, outputEnd: 6, outputCount: 2 },
      { chunkIndex: 2, projectedChunkCount: 3, outputStart: 6, outputEnd: 10, outputCount: 4 },
    ],
    'adaptive range planning must preserve exact coverage while the live control shrinks and relaxes',
  );
  assert.deepEqual(
    schedulerModule.planNextSpnFusionChunk(Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER - 2, Number.MAX_SAFE_INTEGER, 1),
    {
      chunkIndex: 1,
      projectedChunkCount: 2,
      outputStart: Number.MAX_SAFE_INTEGER - 2,
      outputEnd: Number.MAX_SAFE_INTEGER,
      outputCount: 2,
    },
    'uncapped next-range arithmetic must preserve the exact safe-integer remainder without overflow',
  );
  assert.throws(
    () => schedulerModule.planNextSpnFusionChunk(10, 10, 4, 2),
    /output start must identify a remaining safe-integer range/,
    'adaptive planning must reject a range start after all output has been covered',
  );
}
if (typeof schedulerModule.planNextVitBlockChunk === 'function') {
  const firstVitRange = schedulerModule.planNextVitBlockChunk(24, 0, 4, 0);
  const reducedVitRange = schedulerModule.planNextVitBlockChunk(24, firstVitRange.blockEnd, 2, 1);
  const relaxedVitRange = schedulerModule.planNextVitBlockChunk(24, reducedVitRange.blockEnd, 8, 2);
  assert.deepEqual(firstVitRange, {
    blockChunkIndex: 0,
    blockStart: 0,
    blockEnd: 4,
    blockCount: 4,
    totalBlocks: 24,
  });
  assert.deepEqual(reducedVitRange, {
    blockChunkIndex: 1,
    blockStart: 4,
    blockEnd: 6,
    blockCount: 2,
    totalBlocks: 24,
  });
  assert.deepEqual(relaxedVitRange, {
    blockChunkIndex: 2,
    blockStart: 6,
    blockEnd: 14,
    blockCount: 8,
    totalBlocks: 24,
  });
}
if (typeof schedulerModule.planVitBlockMicroduties === 'function') {
  assert.deepEqual(
    schedulerModule.planVitBlockMicroduties({
      blockChunkIndex: 1,
      blockStart: 4,
      blockEnd: 6,
      blockCount: 2,
      totalBlocks: 24,
    }),
    [
      { microdutyIndex: 0, blockIndex: 4, microphase: 'attention-residual' },
      { microdutyIndex: 1, blockIndex: 4, microphase: 'mlp-residual' },
      { microdutyIndex: 2, blockIndex: 5, microphase: 'attention-residual' },
      { microdutyIndex: 3, blockIndex: 5, microphase: 'mlp-residual' },
    ],
    'microduty planning must preserve attention-before-MLP order for every exact block range',
  );
}
const fusionChunkScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    spnFusionChunkItems: 4,
    waitForSubmittedWorkDone: true,
    yieldMs: 0,
  },
});
const fusionChunkTelemetry = createSharpRunTelemetry(fusionChunkScheduler, { runId: 'fusion-chunk-run' });
await schedulerYield(
  fusionChunkScheduler,
  { queue: { onSubmittedWorkDone: async () => {} } },
  fusionChunkTelemetry,
  'spn-fusion',
  {
    block: 'upsample_latent0.layer-3.output-chunk-0',
    parentBlock: 'upsample_latent0.layer-3',
    role: 'spn-fusion-output-chunk',
    outputChunkIndex: 0,
    outputChunkCount: 2,
    outputStart: 0,
    outputEnd: 4,
    outputCount: 4,
    totalOutputItems: 8,
  }
);
await schedulerYield(
  fusionChunkScheduler,
  { queue: { onSubmittedWorkDone: async () => {} } },
  fusionChunkTelemetry,
  'spn-fusion',
  {
    block: 'upsample_latent0.layer-3',
    parentBlock: 'upsample_latent0',
    role: 'wait-bearing-layer',
    chunkRole: 'spn-fusion-output-chunk',
    outputChunkIndex: 1,
    outputChunkCount: 2,
    outputStart: 4,
    outputEnd: 8,
    outputCount: 4,
    totalOutputItems: 8,
  }
);
const fusionChunkSnapshot = schedulerTelemetrySnapshot(fusionChunkTelemetry);
const fusionChunkAssertion = fusionChunkSnapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseChunkSize.spnFusionOutputItems');
assert.ok(fusionChunkAssertion, 'requested SPN fusion chunking must produce a boundary assertion');
assert.equal(fusionChunkAssertion.status, 'verified', 'observed submitted and yielded SPN output chunks must verify requested chunking');
assert.equal(fusionChunkAssertion.effective, 4);
assert.equal(fusionChunkAssertion.observedRole, 'spn-fusion-output-chunk');
assert.equal(fusionChunkAssertion.observedCount, 2, 'verification must include the final tile carried by the legacy layer wait');

const ineffectiveFusionScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    spnFusionChunkItems: 100,
    waitForSubmittedWorkDone: true,
    yieldMs: 0,
  },
});
const ineffectiveFusionTelemetry = createSharpRunTelemetry(ineffectiveFusionScheduler, { runId: 'ineffective-fusion-chunk-run' });
await schedulerYield(
  ineffectiveFusionScheduler,
  { queue: { onSubmittedWorkDone: async () => {} } },
  ineffectiveFusionTelemetry,
  'spn-fusion',
  {
    block: 'upsample-lowres',
    role: 'wait-bearing-layer',
    chunkRole: 'spn-fusion-output-chunk',
    outputChunkIndex: 0,
    outputChunkCount: 1,
    outputStart: 0,
    outputEnd: 8,
    outputCount: 8,
    totalOutputItems: 8,
  }
);
const ineffectiveFusionSnapshot = schedulerTelemetrySnapshot(ineffectiveFusionTelemetry);
const ineffectiveFusionAssertion = ineffectiveFusionSnapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseChunkSize.spnFusionOutputItems');
assert.equal(ineffectiveFusionAssertion.status, 'unverified', 'one output dispatch must not falsely verify requested SPN fusion tiling');
assert.equal(ineffectiveFusionAssertion.observedCount, 0, 'only multi-chunk output ranges count as observed SPN fusion tiling');
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

const backgroundScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'background',
  },
});
assert.equal(backgroundScheduler.effective.mode, 'background', 'background mode must stay visible in effective scheduler identity');
assert.equal(backgroundScheduler.effective.spnPatchChunkSize, 1, 'background mode must default to one SPN patch per chunk');
assert.equal(backgroundScheduler.effective.vitBlockChunkSize, 1, 'background mode must default to one ViT block per chunk');
assert.equal(backgroundScheduler.effective.vitMicroduty, true, 'background mode must split one-block ViT work into foreground-serviceable microduties');
assert.equal(backgroundScheduler.effective.waitForSubmittedWorkDone, true, 'background mode must wait for submitted work before yielding');
assert.ok(backgroundScheduler.effective.yieldMs >= 8, 'background mode must donate real event-loop time, not setTimeout(0)');
assert.ok(backgroundScheduler.effective.gaussianPhaseYieldMs >= 8, 'background mode must donate real Gaussian phase yield time');
assert.ok(backgroundScheduler.effective.routeTailYieldMs >= 8, 'background mode must donate real route-tail yield time');
assert.ok(backgroundScheduler.effective.cpuChunkItems > 0, 'background mode must define CPU materialization chunk size');

const backgroundMissingVitTelemetry = createSharpRunTelemetry(backgroundScheduler, { runId: 'background-missing-vit-run' });
await schedulerYield(
  backgroundScheduler,
  { queue: { onSubmittedWorkDone: async () => {} } },
  backgroundMissingVitTelemetry,
  'spn-patch-chunk',
  { chunkStart: 0, chunkEnd: 1, totalPatches: 35 }
);
await schedulerYield(
  backgroundScheduler,
  { queue: { onSubmittedWorkDone: async () => {} } },
  backgroundMissingVitTelemetry,
  'gaussian-phase',
  { phase: 'prediction-head' },
  backgroundScheduler.effective.gaussianPhaseYieldMs
);
await schedulerYield(
  backgroundScheduler,
  { queue: { onSubmittedWorkDone: async () => {} } },
  backgroundMissingVitTelemetry,
  'route-tail',
  { role: 'cpu-materialization-chunk', stage: 'output-capture', step: 'depth-preview-pixels', processedItems: backgroundScheduler.effective.cpuChunkItems }
);
const backgroundMissingVitSnapshot = schedulerTelemetrySnapshot(backgroundMissingVitTelemetry);
const backgroundMissingVitAssertion = backgroundMissingVitSnapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseChunkSize.vitBlock');
assert.equal(backgroundMissingVitSnapshot.status, 'scheduler-unverified', 'background preset ViT chunking must remain unverified without observed ViT block boundaries');
assert.ok(backgroundMissingVitAssertion, 'background preset ViT chunking must produce a boundary assertion');
assert.equal(backgroundMissingVitAssertion.status, 'unverified');
assert.equal(backgroundMissingVitAssertion.effective, 1);

const backgroundMissingRouteTailTelemetry = createSharpRunTelemetry(backgroundScheduler, { runId: 'background-missing-route-tail-run' });
await schedulerYield(
  backgroundScheduler,
  { queue: { onSubmittedWorkDone: async () => {} } },
  backgroundMissingRouteTailTelemetry,
  'spn-patch-chunk',
  { chunkStart: 0, chunkEnd: 1, totalPatches: 35 }
);
await schedulerYield(
  backgroundScheduler,
  { queue: { onSubmittedWorkDone: async () => {} } },
  backgroundMissingRouteTailTelemetry,
  'vit-block-chunk',
  { encoder: 'patch', blockStart: 0, blockEnd: 1, totalBlocks: 24, tokenCount: 577 }
);
await schedulerYield(
  backgroundScheduler,
  { queue: { onSubmittedWorkDone: async () => {} } },
  backgroundMissingRouteTailTelemetry,
  'gaussian-phase',
  { phase: 'prediction-head' },
  backgroundScheduler.effective.gaussianPhaseYieldMs
);
const backgroundMissingRouteTailSnapshot = schedulerTelemetrySnapshot(backgroundMissingRouteTailTelemetry);
const backgroundRouteTailAssertion = backgroundMissingRouteTailSnapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseYieldMs.routeTail');
const backgroundCpuChunkAssertion = backgroundMissingRouteTailSnapshot.boundaryAssertions.find(assertion => assertion.field === 'cpuMaterializationChunkItems');
assert.equal(backgroundMissingRouteTailSnapshot.status, 'scheduler-unverified', 'background route-tail duties must remain unverified without observed route-tail CPU chunk events');
assert.ok(backgroundRouteTailAssertion, 'background route-tail yield must produce a boundary assertion');
assert.equal(backgroundRouteTailAssertion.status, 'unverified');
assert.ok(backgroundCpuChunkAssertion, 'background CPU materialization chunking must produce a boundary assertion');
assert.equal(backgroundCpuChunkAssertion.status, 'unverified');

async function recordBackgroundDutiesExceptSpn(schedulerConfig, telemetrySink) {
  const queueDevice = { queue: { onSubmittedWorkDone: async () => {} } };
  await schedulerYield(
    schedulerConfig,
    queueDevice,
    telemetrySink,
    'vit-block-chunk',
    { encoder: 'patch', blockStart: 0, blockEnd: 1, totalBlocks: 24, tokenCount: 577 }
  );
  await schedulerYield(
    schedulerConfig,
    queueDevice,
    telemetrySink,
    'gaussian-phase',
    { phase: 'prediction-head' },
    schedulerConfig.effective.gaussianPhaseYieldMs
  );
  await schedulerYield(
    schedulerConfig,
    queueDevice,
    telemetrySink,
    'route-tail',
    { role: 'cpu-materialization-chunk', stage: 'output-capture', step: 'depth-preview-pixels', processedItems: schedulerConfig.effective.cpuChunkItems }
  );
}

const backgroundZeroSpnScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'background',
    spnPatchChunkSize: 0,
  },
});
assert.equal(backgroundZeroSpnScheduler.requested.spnPatchChunkSize, 0, 'zero SPN caller input must stay visible as requested metadata');
assert.equal(backgroundZeroSpnScheduler.effective.spnPatchChunkSize, 1, 'zero SPN caller input is clamped to an effective chunking obligation');
const backgroundZeroSpnTelemetry = createSharpRunTelemetry(backgroundZeroSpnScheduler, { runId: 'background-zero-spn-run' });
await recordBackgroundDutiesExceptSpn(backgroundZeroSpnScheduler, backgroundZeroSpnTelemetry);
const backgroundZeroSpnSnapshot = schedulerTelemetrySnapshot(backgroundZeroSpnTelemetry);
const backgroundZeroSpnAssertion = backgroundZeroSpnSnapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseChunkSize.spnPatch');
assert.equal(backgroundZeroSpnSnapshot.status, 'scheduler-unverified', 'clamped zero SPN chunking must remain unverified without observed SPN patch boundaries');
assert.ok(backgroundZeroSpnAssertion, 'clamped zero SPN chunking must produce an effective SPN assertion');
assert.equal(backgroundZeroSpnAssertion.status, 'unverified');
assert.equal(backgroundZeroSpnAssertion.requested, 0);
assert.equal(backgroundZeroSpnAssertion.effective, 1);

const backgroundNullSpnScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'background',
    spnPatchChunkSize: null,
  },
});
assert.equal(backgroundNullSpnScheduler.requested.spnPatchChunkSize, null, 'null SPN caller input must stay visible as requested metadata');
assert.equal(backgroundNullSpnScheduler.effective.spnPatchChunkSize, 1, 'null SPN caller input is clamped to an effective chunking obligation');
const backgroundNullSpnTelemetry = createSharpRunTelemetry(backgroundNullSpnScheduler, { runId: 'background-null-spn-run' });
await recordBackgroundDutiesExceptSpn(backgroundNullSpnScheduler, backgroundNullSpnTelemetry);
const backgroundNullSpnSnapshot = schedulerTelemetrySnapshot(backgroundNullSpnTelemetry);
const backgroundNullSpnAssertion = backgroundNullSpnSnapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseChunkSize.spnPatch');
assert.equal(backgroundNullSpnSnapshot.status, 'scheduler-unverified', 'clamped null SPN chunking must remain unverified without observed SPN patch boundaries');
assert.ok(backgroundNullSpnAssertion, 'clamped null SPN chunking must produce an effective SPN assertion');
assert.equal(backgroundNullSpnAssertion.status, 'unverified');
assert.equal(backgroundNullSpnAssertion.requested, null);
assert.equal(backgroundNullSpnAssertion.effective, 1);

const explicitBackgroundScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'background',
    yieldMs: 3,
    waitForSubmittedWorkDone: false,
    routeTailYieldMs: 5,
    cpuChunkItems: 1234,
  },
});
assert.equal(explicitBackgroundScheduler.effective.yieldMs, 3, 'explicit background yieldMs must override the mode preset');
assert.equal(explicitBackgroundScheduler.effective.waitForSubmittedWorkDone, false, 'explicit background queue-wait choice must override the mode preset');
assert.equal(explicitBackgroundScheduler.effective.routeTailYieldMs, 5, 'explicit routeTailYieldMs must override the mode preset');
assert.equal(explicitBackgroundScheduler.effective.cpuChunkItems, 1234, 'explicit cpuChunkItems must override the mode preset');

const dutyMap = createSharpRuntimeDutyMap();
assert.equal(dutyMap.schema, 'sharp-webgpu.background-duty-map.v0');
assert.ok(dutyMap.generatedAt, 'duty map must preserve generation time');
for (const requiredStep of [
  'spn.patch-final-token-readback',
  'spn.patch-intermediate-feature-readback',
  'spn.lowres-fusion-readback',
  'gaussian.initializer-disparity-readback',
  'output-capture.disparity-readback',
  'compose-ply.geometry-delta-readback',
  'compose-ply.texture-delta-readback',
  'compose-ply.compose-export',
  'compose-ply.depth-normalize',
  'compose-ply.depth-min',
  'compose-ply.depth-rescale',
  'compose-ply.base-disparity',
  'compose-ply.base-grid',
  'compose-ply.base-color',
  'compose-ply.ply-data-allocation',
  'compose-ply.gaussian-activation-setup',
  'compose-ply.ply-blob-assembly',
  'compose-ply.object-url-create',
  'compose-ply.output-bind',
  'compose-ply.inference-window-finalize',
]) {
  assert.ok(
    dutyMap.steps.some(step => step.id === requiredStep),
    `background duty map must classify ${requiredStep}`
  );
}
const midstreamSteps = dutyMap.steps.filter(step => step.syncClass === 'midstream-sync');
const finalMaterializationSteps = dutyMap.steps.filter(step => step.syncClass === 'final-materialization');
assert.ok(midstreamSteps.length >= 4, 'duty map must identify multiple midstream sync walls');
assert.ok(finalMaterializationSteps.length >= 3, 'duty map must identify final materialization work separately');
assert.ok(
  dutyMap.steps.some(step => step.id === 'gaussian.initializer-disparity-readback' && step.nextAction === 'move-to-gpu'),
  'Gaussian initializer disparity readback must be classified as a GPU-residency target'
);
assert.ok(
  dutyMap.steps.some(step => step.id === 'compose-ply.compose-export' && step.productHandling === 'materialize-visibly'),
  'PLY export must be classed as visible materialization rather than hidden route wait'
);

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
assert.equal(
  proofSnapshot.eventTrace.clock?.schema,
  'kaminos.browser-epoch-monotonic-clock.v0',
  'scheduler telemetry must expose a cross-page comparable epoch-monotonic clock',
);
assert.equal(proofSnapshot.eventTrace.clock?.relativeClock, 'performance.now');
assert.equal(proofSnapshot.eventTrace.clock?.epochClock, 'performance.timeOrigin+performance.now');
assert.ok(Number.isFinite(proofSnapshot.eventTrace.clock?.timeOriginEpochMs));
assert.ok(proofSnapshot.eventTrace.events.some(event => event.kind === 'chunk-start' && event.boundary === 'spn-patch-chunk'));
assert.ok(proofSnapshot.eventTrace.events.some(event => event.kind === 'queue-work-done-start' && event.boundary === 'spn-patch-chunk'));
assert.ok(proofSnapshot.eventTrace.events.some(event => event.kind === 'queue-work-done-end' && event.boundary === 'spn-patch-chunk'));
assert.ok(proofSnapshot.eventTrace.events.some(event => event.kind === 'js-yield-start' && event.boundary === 'spn-patch-chunk'));
assert.ok(proofSnapshot.eventTrace.events.some(event => event.kind === 'js-yield-end' && event.boundary === 'spn-patch-chunk'));
const proofQueueStart = proofSnapshot.eventTrace.events.find(event => event.kind === 'queue-work-done-start');
const proofQueueEnd = proofSnapshot.eventTrace.events.find(event => event.kind === 'queue-work-done-end');
assert.ok(proofQueueStart?.dutyId, 'queue-drain start must carry a stable duty id');
assert.equal(proofQueueEnd?.dutyId, proofQueueStart?.dutyId, 'queue-drain endpoints must share one duty id');
assert.ok(Number.isFinite(proofQueueStart?.epochMs), 'queue-drain start must carry an epoch timestamp');
assert.ok(Number.isFinite(proofQueueEnd?.epochMs), 'queue-drain end must carry an epoch timestamp');
assert.ok(proofQueueEnd.epochMs >= proofQueueStart.epochMs, 'queue-drain epoch endpoints must be ordered');
for (const event of proofSnapshot.eventTrace.events) {
  assert.equal(event.runId, 'proof-run', `scheduler event ${event.kind} must carry the active run id`);
  assert.ok(Number.isFinite(event.epochMs), `scheduler event ${event.kind} must carry epochMs`);
  assert.ok(
    Math.abs((event.epochMs - proofSnapshot.eventTrace.clock.timeOriginEpochMs) - event.tMs) < 1,
    `scheduler event ${event.kind} must use the declared time origin`,
  );
}
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
assert.match(mainSource, /sharpRuntimeGlobal\.__SHARP_LAST_RUN_TELEMETRY__/, 'browser and callable routes must expose last scheduler telemetry for Kaminos');
assert.match(
  mainSource,
  /markInferenceStart\?\.\(currentSchedulerTelemetry\.runId\)/,
  'contention inference windows must bind to the active scheduler telemetry run id',
);
assert.match(
  mainSource,
  /markInferenceEnd\?\.\(currentSchedulerTelemetry\.runId\)/,
  'contention inference closure must preserve the active scheduler telemetry run id',
);
assert.match(mainSource, /schedulerTelemetrySnapshot/, 'main entry must publish a normalized scheduler telemetry snapshot');
assert.equal(
  [...mainSource.matchAll(/await schedulerTelemetrySnapshotCooperatively\(currentSchedulerTelemetry, '(?:verified|failed)'\)/g)].length,
  3,
  'both success returns and failure finalization must use task-yielding telemetry snapshots',
);
assert.match(mainSource, /scheduler:\s*sharpRouteDefinition\.scheduler/, 'route debug scheduler must preserve the route scheduler receipt shape for contention witnesses');
assert.match(mainSource, /sharpScheduler:\s*null/, 'route debug must expose the SHARP scheduler config on a distinct field');
assert.doesNotMatch(mainSource, /runDebug\.scheduler\s*=\s*currentScheduler/, 'SHARP scheduler config must not overwrite the route scheduler debug shape');
assert.match(mainSource, /runDebug\.sharpScheduler\s*=\s*currentScheduler/, 'main entry must bind the per-run SHARP scheduler config to sharpScheduler');
assert.match(mainSource, /monodepth\.run\([\s\S]*weights,\s*\{\s*scheduler:\s*currentScheduler,\s*telemetry:\s*currentSchedulerTelemetry,\s*\}/, 'main entry must pass scheduler telemetry into monodepth');
assert.match(mainSource, /schedulerYield/, 'main route tail must use schedulerYield for cooperative tail checkpoints');
assert.match(mainSource, /routeTailTimings/, 'main route tail must record per-step route tail timing deltas');
assert.match(mainSource, /routeTailTimings:\s*runDebug\.routeTailTimings/, 'route receipt metadata must preserve route-tail timing deltas');
assert.match(mainSource, /backgroundDutyMap:\s*createSharpRuntimeDutyMap\(\)/, 'run debug must expose the SHARP background duty map');
assert.match(mainSource, /backgroundDutyMap:\s*runDebug\.backgroundDutyMap/, 'route receipt metadata must preserve the background duty map');
assert.match(mainSource, /recordCpuDutyChunk/, 'main route tail must chunk CPU materialization work under scheduler control');
assert.match(mainSource, /receipt\.metadataPayload\s*=\s*metadata/, 'route receipt object must carry the concrete route-tail metadata payload, not only a metadata hash');
assert.match(mainSource, /['"]route-tail['"]/, 'main route tail must emit route-tail scheduler telemetry');
for (const step of ['object-url-create', 'output-bind']) {
  assert.match(
    mainSource,
    new RegExp(`step:\\s*['"]${escapeRegExp(step)}['"]`),
    `main route tail must emit a named blocking interval for ${step}`,
  );
}
assert.match(mainSource, /step:\s*['"]inference-window-finalize['"]/, 'main route must name the post-bind inference finalization envelope');
assert.match(mainSource, /role:\s*['"]localization-envelope['"]/, 'post-bind timing must remain an envelope rather than claiming a specific blocking operation');
assert.match(
  mainSource,
  /inferenceFinalizeStartMs[\s\S]*markInferenceEnd[\s\S]*step:\s*['"]inference-window-finalize['"]/,
  'the finalization envelope must begin from the prior duty end and include contention-probe inference closure',
);
const composeSource = readFileSync(composePath, 'utf8');
assert.match(composeSource, /step:\s*['"]ply-data-allocation['"]/, 'PLY data allocation must emit its own blocking interval');
assert.match(composeSource, /step:\s*['"]gaussian-activation-setup['"]/, 'Gaussian activation setup must emit its own blocking interval');
assert.match(composeSource, /step:\s*['"]ply-blob-assembly['"]/, 'PLY assembly must emit its named blocking interval');
assert.match(mainSource, /intervalStartMs/, 'route-tail telemetry must preserve interval starts for blocking duties');
assert.match(mainSource, /intervalEndMs/, 'route-tail telemetry must preserve interval ends for blocking duties');
for (const [stage, steps] of [
  ['output-capture', ['disparity-readback', 'depth-preview-render']],
  ['compose-ply', ['geometry-delta-readback', 'texture-delta-readback', 'compose-export']],
]) {
  assert.match(mainSource, new RegExp(`stage:\\s*['"]${escapeRegExp(stage)}['"]`), `main route tail must record ${stage} timing stage identity`);
  for (const step of steps) {
    assert.match(mainSource, new RegExp(`step:\\s*['"]${escapeRegExp(step)}['"]`), `main route tail must record ${stage}/${step} timing identity`);
  }
}

const spnSource = readFileSync(spnPath, 'utf8');
const convTransposeOpsSource = readFileSync(shaderOpsPath, 'utf8');
const convTransposeShaderSource = readFileSync(convTransposeShaderPath, 'utf8');
assert.doesNotMatch(spnSource, /const\s+CHUNK_SIZE\s*=\s*4/, 'SPN patch chunking must not be a hidden singleton constant');
assert.match(spnSource, /effective\.spnPatchChunkSize/, 'SPN patch chunking must use the effective scheduler config');
assert.match(spnSource, /spn-patch-chunk/, 'SPN must record breathing evidence around patch chunks');
assert.match(spnSource, /effective\.spnFusionChunkItems/, 'SPN fusion dispatch chunking must use explicit effective scheduler config');
assert.match(spnSource, /planNextSpnFusionChunk/, 'SPN fusion dispatches must replan one exact range from the current live control');
assert.match(spnSource, /outputChunkCountAuthority:\s*['"]projection-at-encode['"]/, 'adaptive SPN telemetry must label projected chunk counts');
assert.match(spnSource, /outputChunkActualCount/, 'adaptive SPN telemetry must preserve final actual chunk count');
assert.match(spnSource, /role:\s*['"]spn-fusion-output-chunk['"]/, 'SPN fusion chunks must emit distinct wait-bearing telemetry');
assert.match(spnSource, /chunkRole:\s*['"]spn-fusion-output-chunk['"]/, 'the final layer wait must retain explicit final-tile identity');
assert.match(spnSource, /_dispatchChunkedConvTranspose2d/, 'all SPN transposed convolutions must share one range-submission helper');
assert.match(
  spnSource,
  /_dispatchChunkedConvTranspose2d\(\{[\s\S]*?blockLabel:\s*['"]upsample-lowres['"][\s\S]*?parentBlock:\s*null/,
  'the standalone SPN lowres transposed convolution must use the shared range-submission helper'
);
assert.equal(
  findAllMatches(spnSource, /dispatchConvTranspose2d\(/g).length,
  1,
  'SPN must dispatch transposed convolutions only inside the shared range-submission helper'
);
assert.match(convTransposeOpsSource, /outputStart/, 'conv-transpose dispatch wrapper must accept an output range start');
assert.match(convTransposeOpsSource, /outputCount/, 'conv-transpose dispatch wrapper must accept an output range length');
assert.match(convTransposeShaderSource, /outputStart/, 'conv-transpose shader must offset each chunk into the shared output tensor');
assert.match(convTransposeShaderSource, /outputCount/, 'conv-transpose shader must reject invocations outside the requested chunk');

const executableSpnSource = spnSource
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/\/\/.*$/gm, '');

function assertSpnFusionYieldAfterSubmit(block) {
  const pattern = new RegExp(
    `await\\s+schedulerYield\\(\\s*scheduler\\s*,\\s*device\\s*,\\s*telemetry\\s*,\\s*['"]spn-fusion['"]\\s*,\\s*\\{\\s*block:\\s*['"]${escapeRegExp(block)}['"]\\s*\\}`,
    'g'
  );
  const matches = findAllMatches(executableSpnSource, pattern);
  assert.equal(matches.length, 1, `SPN must record one spn-fusion scheduler boundary for ${block}`);
  const precedingSource = executableSpnSource.slice(Math.max(0, matches[0].index - 1000), matches[0].index);
  const lastSubmit = precedingSource.lastIndexOf('device.queue.submit([');
  assert.notEqual(lastSubmit, -1, `SPN must submit GPU work before awaiting ${block}`);
  const afterSubmit = precedingSource.slice(lastSubmit);
  assert.match(
    afterSubmit,
    /^device\.queue\.submit\(\[[a-zA-Z0-9_$]+\.finish\(\)\]\);/,
    `SPN must submit a finished command buffer before awaiting ${block}`
  );
  assert.doesNotMatch(
    afterSubmit,
    /await\s+schedulerYield\(/,
    `SPN must not record another scheduler boundary between submit and ${block}`
  );
}

function assertSpnYieldAfterReadback(varName, block) {
  const assignmentPattern = new RegExp(
    `const\\s+${escapeRegExp(varName)}\\s*=\\s*await\\s+readBuffer\\(`,
    'g'
  );
  const yieldPattern = new RegExp(
    `await\\s+schedulerYield\\(\\s*scheduler\\s*,\\s*device\\s*,\\s*telemetry\\s*,\\s*['"]spn-fusion['"]\\s*,\\s*\\{\\s*block:\\s*['"]${escapeRegExp(block)}['"]\\s*\\}`,
    'g'
  );
  const assignment = findAllMatches(executableSpnSource, assignmentPattern)[0];
  const yieldMatch = findAllMatches(executableSpnSource, yieldPattern)[0];
  assert.ok(assignment, `SPN must read back ${varName} explicitly`);
  assert.ok(yieldMatch, `SPN must yield after ${varName} readback using ${block}`);
  assert.ok(yieldMatch.index > assignment.index, `SPN must await ${block} after ${varName} readback`);
}

function assertSpnYieldAfterUpload(varName, block) {
  const uploadPattern = new RegExp(
    `const\\s+${escapeRegExp(varName)}\\s*=\\s*createStorageBuffer\\(\\s*device\\s*,\\s*concatData\\s*\\)`,
    'g'
  );
  const yieldPattern = new RegExp(
    `await\\s+schedulerYield\\(\\s*scheduler\\s*,\\s*device\\s*,\\s*telemetry\\s*,\\s*['"]spn-fusion['"]\\s*,\\s*\\{\\s*block:\\s*['"]${escapeRegExp(block)}['"]\\s*\\}`,
    'g'
  );
  const upload = findAllMatches(executableSpnSource, uploadPattern)[0];
  const yieldMatch = findAllMatches(executableSpnSource, yieldPattern)[0];
  assert.ok(upload, `SPN must upload ${varName} from concatData explicitly`);
  assert.ok(yieldMatch, `SPN must yield after ${varName} upload using ${block}`);
  assert.ok(yieldMatch.index > upload.index, `SPN must await ${block} after ${varName} upload`);
}

function assertSpnUpsampleInternalLayer(block, minLayerCount) {
  const callPattern = new RegExp(
    `await\\s+this\\._dispatchUpsampleBlock\\([\\s\\S]*?['"]${escapeRegExp(block)}['"]\\s*,\\s*scheduler\\s*,\\s*telemetry\\s*\\)`,
    'g'
  );
  assert.equal(
    findAllMatches(executableSpnSource, callPattern).length,
    1,
    `SPN ${block} must call the upsample helper with scheduler telemetry and block identity`
  );
  const layerBoundaryPattern = new RegExp(
    `block:\\s*\`\\$\\{blockLabel\\}\\.layer-\\$\\{i\\}\`[\\s\\S]*parentBlock:\\s*blockLabel[\\s\\S]*role:\\s*['"]wait-bearing-layer['"]`,
    'm'
  );
  assert.match(
    executableSpnSource,
    layerBoundaryPattern,
    'SPN upsample helper must emit wait-bearing per-layer block telemetry'
  );
  const markerPattern = new RegExp(
    `await\\s+schedulerYield\\(\\s*scheduler\\s*,\\s*device\\s*,\\s*telemetry\\s*,\\s*['"]spn-fusion['"]\\s*,\\s*\\{[\\s\\S]*?block:\\s*['"]${escapeRegExp(block)}['"][\\s\\S]*?role:\\s*['"]group-complete['"][\\s\\S]*?layerCount:\\s*${minLayerCount}`,
    'm'
  );
  assert.match(
    executableSpnSource,
    markerPattern,
    `SPN ${block} must preserve a coarse group-complete marker with layerCount ${minLayerCount}`
  );
}

for (const block of ['fuse-lowres']) {
  assertSpnFusionYieldAfterSubmit(block);
}
for (const [block, layerCount] of [
  ['upsample_latent0', 4],
  ['upsample_latent1', 3],
  ['upsample0', 2],
  ['upsample1', 2],
  ['upsample2', 2],
]) {
  assertSpnUpsampleInternalLayer(block, layerCount);
}
assert.doesNotMatch(
  executableSpnSource,
  /await\s+readBuffer\(\s*device\s*,\s*feat4x2\.buffer/,
  'SPN lowres fusion must not read back x2 upsampled features for CPU concat'
);
assert.doesNotMatch(
  executableSpnSource,
  /await\s+readBuffer\(\s*device\s*,\s*lowresResult\.buffer/,
  'SPN lowres fusion must not read back lowres upsampled features for CPU concat'
);
assert.doesNotMatch(
  executableSpnSource,
  /concatData|cpu-concat-lowres|concat-upload/,
  'SPN lowres fusion must not allocate CPU concatData or expose CPU concat/upload boundaries'
);
assert.match(
  executableSpnSource,
  /dispatchConcatChannels\(\s*device\s*,\s*concatEnc\s*,\s*feat4x2\.buffer\s*,\s*lowresResult\.buffer/,
  'SPN lowres fusion must concatenate x2 and lowres features on GPU'
);
assert.match(
  executableSpnSource,
  /block:\s*['"]gpu-concat-lowres['"]/,
  'SPN lowres fusion must expose a GPU concat scheduler boundary'
);
assert.doesNotMatch(
  executableSpnSource,
  /await\s+readBuffer\(\s*device\s*,\s*result\.finalTokensBuf/,
  'SPN patch final tokens must stay GPU-resident through merge'
);
assert.doesNotMatch(
  executableSpnSource,
  /await\s+readBuffer\(\s*device\s*,\s*snap\.buffer/,
  'SPN patch intermediate snapshots must stay GPU-resident through merge'
);
assert.doesNotMatch(
  executableSpnSource,
  /const\s+imgTokens\s*=\s*await\s+readBuffer\(/,
  'SPN image encoder tokens must stay GPU-resident through lowres fusion'
);
assert.doesNotMatch(
  executableSpnSource,
  /mergeFeaturesCPU\(/,
  'SPN run path must not merge patch features on CPU'
);
assert.doesNotMatch(
  executableSpnSource,
  /createStorageBuffer\(\s*device\s*,\s*(?:latent0Merged|latent1Merged|x0Merged|x1Merged)\.data/,
  'SPN run path must not upload CPU-merged patch features'
);
assert.doesNotMatch(
  executableSpnSource,
  /createStorageBuffer\(\s*device\s*,\s*new\s+Float32Array\(\s*(?:x2Feature|imgFeature)\s*\)/,
  'SPN run path must not upload CPU-reshaped single-patch features'
);
assert.match(
  executableSpnSource,
  /dispatchMergeTokenPatches\(\s*device\s*,\s*mergeEnc\s*,\s*layer5TokenBuffers/,
  'SPN must merge layer-5 patch intermediates on GPU'
);
assert.match(
  executableSpnSource,
  /dispatchMergeTokenPatches\(\s*device\s*,\s*mergeEnc\s*,\s*layer11TokenBuffers/,
  'SPN must merge layer-11 patch intermediates on GPU'
);
assert.match(
  executableSpnSource,
  /dispatchMergeTokenPatches\(\s*device\s*,\s*mergeEnc\s*,\s*patchTokenBuffers\.slice\(\s*0\s*,\s*25\s*\)/,
  'SPN must merge first 25 patch final-token buffers on GPU'
);
assert.match(
  executableSpnSource,
  /dispatchMergeTokenPatches\(\s*device\s*,\s*mergeEnc\s*,\s*patchTokenBuffers\.slice\(\s*25\s*,\s*34\s*\)/,
  'SPN must merge 3x3 patch final-token buffers on GPU'
);
assert.match(
  executableSpnSource,
  /block:\s*['"]spn-patch-merge-gpu['"]/,
  'SPN must expose a GPU patch-merge scheduler boundary'
);
assert.match(
  executableSpnSource,
  /this\.vitEncoder\.encode\(\s*patchBuf[\s\S]*?retainOutputs:\s*true/,
  'SPN patch encoder calls must request caller-retained token buffers before delayed GPU merge'
);
assert.match(
  executableSpnSource,
  /this\.vitEncoder\.encode\(\s*imgBuf384[\s\S]*?retainOutputs:\s*true/,
  'SPN image encoder calls must request caller-retained tokens before GPU lowres merge'
);

const shaderOpsSource = readFileSync(shaderOpsPath, 'utf8');
assert.match(shaderOpsSource, /concat_channels\.wgsl\?raw/, 'shader ops must import a GPU channel-concat shader');
assert.match(shaderOpsSource, /export function dispatchConcatChannels/, 'shader ops must expose dispatchConcatChannels');
assert.ok(existsSync(concatChannelsShaderPath), 'SPN GPU lowres concat must have a WGSL shader source');
assert.match(shaderOpsSource, /token_patch_merge\.wgsl\?raw/, 'shader ops must import a GPU token patch-merge shader');
assert.match(shaderOpsSource, /export function dispatchMergeTokenPatches/, 'shader ops must expose dispatchMergeTokenPatches');
assert.ok(existsSync(tokenPatchMergeShaderPath), 'SPN GPU patch merge must have a WGSL shader source');
assert.match(shaderOpsSource, /gaussian_initializer_feature_input\.wgsl\?raw/, 'shader ops must import a GPU Gaussian initializer shader');
assert.match(shaderOpsSource, /gaussian_initializer_reduce_min\.wgsl\?raw/, 'shader ops must import a GPU Gaussian initializer min-reduction shader');
assert.match(shaderOpsSource, /export function dispatchGaussianInitializerFeatureInput/, 'shader ops must expose dispatchGaussianInitializerFeatureInput');
assert.ok(existsSync(gaussianInitializerShaderPath), 'Gaussian initializer GPU residency must have a WGSL shader source');
assert.ok(existsSync(gaussianInitializerReduceShaderPath), 'Gaussian initializer GPU residency must have a min-reduction WGSL shader source');

const monodepthSource = readFileSync(monodepthPath, 'utf8');
assert.match(monodepthSource, /schedulerYield/, 'Monodepth decoder must use the scheduler yield primitive');
assert.match(monodepthSource, /monodepth-phase/, 'Monodepth decoder must record phase-level breathing evidence');
assert.match(monodepthSource, /options\s*=\s*\{\}/, 'Monodepth decoder must accept per-run scheduler options');

const executableMonodepthSource = monodepthSource
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/\/\/.*$/gm, '');

function assertMonodepthYieldAfterSubmit(boundary, minCount = 1) {
  const pattern = new RegExp(
    `await\\s+(?:monodepthPhaseYield|boundaryYield)\\(\\s*['"]${escapeRegExp(boundary)}['"]`,
    'g'
  );
  const matches = findAllMatches(executableMonodepthSource, pattern);
  assert.ok(
    matches.length >= minCount,
    `Monodepth decoder must await monodepthPhaseYield('${boundary}') at least ${minCount} time(s)`
  );
  for (const match of matches) {
    const precedingSource = executableMonodepthSource.slice(Math.max(0, match.index - 900), match.index);
    const lastSubmit = precedingSource.lastIndexOf('device.queue.submit([');
    const lastResidualDispatch = precedingSource.lastIndexOf('dispatchResidualBlock(');
    assert.ok(
      lastSubmit !== -1 || lastResidualDispatch !== -1,
      `Monodepth decoder must submit GPU work before awaiting ${boundary}`
    );
    const afterSubmit = lastResidualDispatch > lastSubmit
      ? precedingSource.slice(lastResidualDispatch)
      : precedingSource.slice(lastSubmit);
    if (lastResidualDispatch <= lastSubmit) {
      assert.match(
        afterSubmit,
        /^device\.queue\.submit\(\[[a-zA-Z0-9_$]+\.finish\(\)\]\);/,
        `Monodepth decoder must submit a finished command buffer before awaiting ${boundary}`
      );
    }
    assert.doesNotMatch(
      afterSubmit,
      /await\s+(?:monodepthPhaseYield|boundaryYield)\(/,
      `Monodepth decoder must not record another monodepth scheduler boundary between submit and ${boundary}`
    );
  }
}

function assertMonodepthContinuityMarker(boundary) {
  const markerPattern = new RegExp(
    `await\\s+boundaryYield\\(\\s*['"]${escapeRegExp(boundary)}['"]\\s*,\\s*\\{([\\s\\S]*?)\\}\\s*\\);`
  );
  const marker = executableMonodepthSource.match(markerPattern);
  assert.ok(marker, `Monodepth must preserve coarse ${boundary} coverage labels for Wake`);
  assert.match(marker[1], /role:\s*['"]group-complete['"]/, `Monodepth ${boundary} must declare itself as a group-complete continuity marker`);
  const childList = marker[1].match(/waitBearingBoundaries:\s*\[([\s\S]*?)\]/);
  assert.ok(
    childList,
    `Monodepth ${boundary} must name wait-bearing child boundaries`
  );
  for (const childBoundary of ['residual-conv1', 'residual-conv2', 'residual-skip-add']) {
    assert.match(
      childList[1],
      new RegExp(`['"]${escapeRegExp(childBoundary)}['"]`),
      `Monodepth ${boundary} must name wait-bearing child boundary ${childBoundary}`
    );
  }
}

assert.match(
  executableMonodepthSource,
  /function\s+dispatchResidualBlock[\s\S]*device\.queue\.submit\(\[[a-zA-Z0-9_$]+\.finish\(\)\]\);[\s\S]*return\s+sumBuf;/,
  'Monodepth residual helper must submit GPU work before caller-level fusion residual yields'
);

for (const [boundary, minCount] of [
  ['project-feature', 1],
  ['residual-conv1', 1],
  ['residual-conv2', 1],
  ['residual-skip-add', 1],
  ['fusion-skip-add', 1],
  ['fusion-deconv', 1],
  ['fusion-out-conv', 1],
  ['head-conv0', 1],
  ['head-deconv', 1],
  ['head-conv2', 1],
  ['head-relu3', 1],
  ['head-conv4', 1],
  ['head-final', 1],
]) {
  assertMonodepthYieldAfterSubmit(boundary, minCount);
}

assert.match(executableMonodepthSource, /await\s+boundaryYield\(\s*['"]fusion-resnet1['"]/, 'Monodepth must preserve coarse fusion-resnet1 coverage labels for Wake');
assert.match(executableMonodepthSource, /await\s+boundaryYield\(\s*['"]fusion-resnet2['"]/, 'Monodepth must preserve coarse fusion-resnet2 coverage labels for Wake');
assertMonodepthContinuityMarker('fusion-resnet1');
assertMonodepthContinuityMarker('fusion-resnet2');
assert.match(executableMonodepthSource, /await\s+monodepthPhaseYield\(\s*['"]head-final['"]/, 'Monodepth must preserve coarse head-final coverage label for Wake');

const backboneSource = readFileSync(backbonePath, 'utf8');
assert.match(backboneSource, /schedulerYield/, 'ViT encoder must use the scheduler yield primitive');
assert.match(backboneSource, /effective\.vitBlockChunkSize/, 'ViT encoder must use the effective scheduler block chunk size');
assert.match(backboneSource, /vit-block-chunk/, 'ViT encoder must record breathing evidence around block chunks');
assert.match(backboneSource, /retainOutputs/, 'ViT encoder must expose a caller-retained output mode for delayed GPU consumers');
assert.match(
  backboneSource,
  /if\s*\(\s*!retainOutputs\s*\)[\s\S]*this\._prevIntermediates\s*=\s*intermediateFeatures[\s\S]*this\._prevFinalBuf\s*=\s*finalNormedBuf/,
  'ViT encoder must not track retained outputs for destruction on the next encode call'
);

const gaussianSource = readFileSync(gaussianPath, 'utf8');
assert.match(gaussianSource, /gaussianPhaseYieldMs/, 'Gaussian decoder phase breathing must use the scheduler config');
assert.match(gaussianSource, /gaussian-phase/, 'Gaussian decoder must record phase-level breathing evidence');

const executableGaussianSource = gaussianSource
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/\/\/.*$/gm, '');

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function findAllMatches(source, pattern) {
  return Array.from(source.matchAll(pattern));
}

function assertAwaitedYieldAfterSubmit(source, yieldName, boundary, minCount = 1) {
  const pattern = new RegExp(`await\\s+${yieldName}\\(\\s*['"]${escapeRegExp(boundary)}['"]`, 'g');
  const matches = findAllMatches(source, pattern);
  assert.ok(
    matches.length >= minCount,
    `Gaussian decoder must await ${yieldName}('${boundary}') at least ${minCount} time(s)`
  );

  for (const match of matches) {
    const precedingSource = source.slice(Math.max(0, match.index - 800), match.index);
    const lastSubmit = precedingSource.lastIndexOf('device.queue.submit([');
    assert.notEqual(
      lastSubmit,
      -1,
      `Gaussian decoder must submit a finished command buffer before awaiting ${boundary}`
    );
    const afterSubmit = precedingSource.slice(lastSubmit);
    assert.match(
      afterSubmit,
      /^device\.queue\.submit\(\[[a-zA-Z0-9_$]+\.finish\(\)\]\);/,
      `Gaussian decoder must submit a finished command buffer before awaiting ${boundary}`
    );
    assert.doesNotMatch(
      afterSubmit,
      /await\s+(?:boundaryYield|gaussianPhaseYield)\(/,
      `Gaussian decoder must not record another scheduler boundary between submit and ${boundary}`
    );
  }
}

for (const [boundary, minCount] of [
  ['residual-conv1', 1],
  ['residual-conv2', 1],
  ['residual-skip-add', 2],
  ['fusion-skip-add', 1],
  ['fusion-deconv', 1],
  ['fusion-out-conv', 1],
  ['head-gn-conv1', 1],
  ['head-gn-conv2', 1],
  ['head-final', 1],
]) {
  assertAwaitedYieldAfterSubmit(executableGaussianSource, 'boundaryYield', boundary, minCount);
}
assertAwaitedYieldAfterSubmit(executableGaussianSource, 'gaussianPhaseYield', 'project-feature', 2);
assertAwaitedYieldAfterSubmit(executableGaussianSource, 'gaussianPhaseYield', 'prediction-geometry');
assertAwaitedYieldAfterSubmit(executableGaussianSource, 'gaussianPhaseYield', 'prediction-texture');

assert.doesNotMatch(
  executableGaussianSource,
  /await\s+readBuffer\(\s*device\s*,\s*disparityBuf/,
  'Gaussian initializer must not read back the full disparity buffer to CPU'
);
assert.doesNotMatch(
  executableGaussianSource,
  /new\s+Float32Array\(\s*5\s*\*\s*imgSize\s*\*\s*imgSize\s*\)/,
  'Gaussian initializer must not build the full feature_input tensor on CPU'
);
assert.doesNotMatch(
  executableGaussianSource,
  /new\s+Float32Array\(\s*2\s*\*\s*HW\s*\)/,
  'Gaussian initializer must not build the full depth tensor on CPU'
);
assert.match(
  executableGaussianSource,
  /dispatchGaussianInitializerFeatureInput\(\s*device\s*,\s*enc\s*,\s*imageInputBuf\s*,\s*disparityBuf/,
  'Gaussian initializer must construct feature_input from image and disparity on GPU'
);
assert.match(
  executableGaussianSource,
  /inputChannels:\s*5[\s\S]*?H:\s*imgSize[\s\S]*?W:\s*imgSize/,
  'Gaussian image encoder boundary must continue to report the 5-channel feature input shape'
);
assertAwaitedYieldAfterSubmit(executableGaussianSource, 'gaussianPhaseYield', 'initializer-feature-input');

assert.match(gaussianSource, /async function dispatchResidualBlock/, 'Gaussian residual blocks must be split by awaitable submit/yield boundaries');
assert.match(gaussianSource, /async function dispatchFusionBlock/, 'Gaussian fusion blocks must be split by awaitable submit/yield boundaries');
assert.match(gaussianSource, /async function dispatchGroupNormResidualBlock/, 'Gaussian head residual blocks must be split by awaitable submit/yield boundaries');
assert.match(gaussianSource, /let\s+features\s*=\s*await\s+dispatchFusionBlock/, 'Gaussian decoder must await the initial decoder fusion block');
assert.match(gaussianSource, /features\s*=\s*await\s+dispatchFusionBlock/, 'Gaussian decoder must await decoder fusion loop blocks');
assert.match(gaussianSource, /const\s+fused\s*=\s*await\s+dispatchFusionBlock/, 'Gaussian decoder must await skip-fusion block');
assert.match(gaussianSource, /const\s+textureFeatures\s*=\s*await\s+dispatchHead/, 'Gaussian decoder must await texture head');
assert.match(gaussianSource, /const\s+geometryFeatures\s*=\s*await\s+dispatchHead/, 'Gaussian decoder must await geometry head');
for (const line of executableGaussianSource.split('\n')) {
  if (
    /dispatch(?:ResidualBlock|FusionBlock|GroupNormResidualBlock|Head)\(/.test(line) &&
    !/async function/.test(line)
  ) {
    assert.match(line, /\bawait\s+dispatch/, `Gaussian helper call must be awaited: ${line.trim()}`);
  }
}

const contentionWitnessSource = readFileSync(contentionWitnessPath, 'utf8');
assert.match(
  contentionWitnessSource,
  /if\s*\(Number\.isFinite\(window\.endMs\)\)\s*return/,
  'contention inference closure must be first-write authoritative',
);
assert.match(contentionWitnessSource, /--sharp-scheduler/, 'contention witness must expose the SHARP scheduler query config as an invocation parameter');
assert.match(contentionWitnessSource, /searchParams\.set\('sharpScheduler'/, 'contention witness must pass the requested scheduler to the browser route URL');
