import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  FOREGROUND_BUDGET_GOVERNOR_SCHEMA,
  SHARP_IMAGE_TO_SPLAT_ROUTE_ID,
  WEBGPU_COMMAND_DUTY_REPORT_SCHEMA,
  WEBGPU_HOST_PHASE,
  WEBGPU_HOST_PHASE_RECORDER_SCHEMA,
  WEBGPU_INFERENCE_KIT_VERSION,
  WEBGPU_INFERENCE_RUNTIME_SCHEMA,
  WEBGPU_SCHEDULER_APPLICATION_SCHEMA,
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
import {
  attachSharpLiveScheduler,
  createSharpRunTelemetry,
  detachSharpLiveScheduler,
  parseSharpSchedulerConfig,
  schedulerTelemetrySnapshot,
  schedulerYield,
} from '../src/lib/scheduler.js';
import * as sharpSchedulerModule from '../src/lib/scheduler.js';

const [kitMajor, kitMinor, kitPatch] = WEBGPU_INFERENCE_KIT_VERSION.split('.').map(Number);
assert.deepEqual([kitMajor, kitMinor], [0, 1]);
assert.ok(kitPatch >= 26, `SHARP foreground opportunity bridge requires kit >=0.1.26, got ${WEBGPU_INFERENCE_KIT_VERSION}`);

const schedulerSource = readFileSync(join(new URL('..', import.meta.url).pathname, 'src', 'lib', 'scheduler.js'), 'utf8');
assert.match(
  schedulerSource,
  /prepareCommandDutyAtBoundary/,
  'SHARP scheduler must await the foreground-aware command-duty boundary API before the next encodable duty',
);
assert.match(
  schedulerSource,
  /requestForegroundOpportunity/,
  'SHARP scheduler must route kiln-frame foreground demand through the kit foreground interlock',
);
assert.doesNotMatch(
  schedulerSource,
  /runtime\.prepareCommandDuty\(/,
  'SHARP scheduler must not use synchronous command-duty preparation at foreground-capable boundaries',
);
assert.equal(typeof sharpSchedulerModule.prepareSchedulerDutyBeforeEncode, 'function');
assert.equal(typeof sharpSchedulerModule.submitPreparedSchedulerDuty, 'function');
assert.equal(typeof sharpSchedulerModule.failPreparedSchedulerDuty, 'function');

const definition = createSharpImageToSplatRouteDefinition({
  kernel: {
    kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
    profile: 'spn-dinov2l16-monodepth-gaussian-ply',
    commit: 'sharp-webgpu-route-runtime-contract',
  },
});

let fakeQueueSubmitCount = 0;
const fakeQueue = {
  submit() { fakeQueueSubmitCount += 1; },
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
const routeRunClock = {
  clockId: 'sharp-route-runtime-contract-clock',
  source: 'performance.now',
  timeOriginEpochMs: 1_700_000_100_000,
};
const routeRunId = 'sharp-route-runtime-contract-run';
const runtime = await createSharpRouteRuntime({
  device: fakeDevice,
  adapter: fakeAdapter,
}, {
  routeDefinition: definition,
  browser: 'node-sharp-runtime-contract',
  runId: routeRunId,
  clock: routeRunClock,
  scheduler: {
    mode: 'cooperative',
    yieldMs: 3,
    waitForSubmittedWorkDone: true,
    phaseChunkSize: {
      spnPatch: 1,
      vitBlock: 2,
    },
  },
  schedulerBounds: {
    yieldMs: { min: 0, max: 20, step: 1 },
    phaseChunkSize: {
      spnPatch: { min: 1, max: 35, stepFactor: 2 },
      vitBlock: { min: 1, max: 24, stepFactor: 2 },
    },
  },
  now: () => {
    nowMs += 5;
    return nowMs;
  },
});

assert.equal(runtime.schema, WEBGPU_INFERENCE_RUNTIME_SCHEMA);
assert.equal(runtime.routeId, SHARP_IMAGE_TO_SPLAT_ROUTE_ID);
assert.equal(runtime.runtimeLabel, SHARP_ROUTE_RUNTIME_LABEL);
assert.deepEqual(runtime.profile.requiredStages, definition.requiredStages);
assert.equal(runtime.hostPhases.schema, WEBGPU_HOST_PHASE_RECORDER_SCHEMA);
assert.equal(runtime.hostPhases.runId, routeRunId);
assert.deepEqual(runtime.hostPhases.clock, routeRunClock);
assert.equal(runtime.commandDuties.runId, routeRunId);
assert.equal(runtime.schedulerApplication.snapshot().schema, WEBGPU_SCHEDULER_APPLICATION_SCHEMA);
assert.equal(runtime.schedulerApplication.snapshot().revision, 0);
assert.equal(runtime.schedulerApplication.snapshot().scheduler.phaseChunkSize.spnPatch, 1);
assert.equal(
  Object.hasOwn(runtime.schedulerApplication.snapshot().scheduler.phaseChunkSize, 'spnFusionOutputItems'),
  false,
  'zero-disabled SPN tiling must omit the live control and its positive bounds',
);
assert.equal(typeof runtime.requestForegroundOpportunity, 'function');
assert.equal(typeof runtime.prepareCommandDutyAtBoundary, 'function');
assert.equal(typeof runtime.finishForegroundOpportunities, 'function');

const fusionRuntime = await createSharpRouteRuntime({
  device: fakeDevice,
  adapter: fakeAdapter,
}, {
  routeDefinition: definition,
  browser: 'node-sharp-runtime-adaptive-fusion-contract',
  runId: 'sharp-route-adaptive-fusion-run',
  clock: routeRunClock,
  scheduler: {
    mode: 'cooperative',
    yieldMs: 0,
    waitForSubmittedWorkDone: true,
    spnFusionChunkItems: 8,
    phaseChunkSize: {
      spnPatch: 1,
      vitBlock: 2,
    },
  },
  schedulerBounds: {
    yieldMs: { min: 0, max: 20, step: 1 },
    phaseChunkSize: {
      spnPatch: { min: 1, max: 35, stepFactor: 2 },
      vitBlock: { min: 1, max: 24, stepFactor: 2 },
      spnFusionOutputItems: { min: 1, max: Number.MAX_SAFE_INTEGER, stepFactor: 2 },
    },
  },
  now: () => {
    nowMs += 5;
    return nowMs;
  },
});
const fusionApplication = fusionRuntime.schedulerApplication.snapshot();
assert.equal(fusionApplication.scheduler.phaseChunkSize.spnFusionOutputItems, 8);
assert.deepEqual(
  fusionApplication.bounds.phaseChunkSize.spnFusionOutputItems,
  { min: 1, max: Number.MAX_SAFE_INTEGER, stepFactor: 2 },
  'positive SPN tiling must declare an uncapped live-control range',
);

const adaptiveFusionScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    spnFusionChunkItems: 8,
    yieldMs: 0,
    waitForSubmittedWorkDone: true,
  },
});
const adaptiveFusionTelemetry = createSharpRunTelemetry(adaptiveFusionScheduler, {
  runId: 'sharp-adaptive-fusion-control-run',
});
await fusionRuntime.runInvocation({ invocationId: 'sharp-adaptive-fusion-invocation' }, async invocation => {
  attachSharpLiveScheduler(adaptiveFusionScheduler, {
    runtime: fusionRuntime,
    invocation,
    runId: 'sharp-route-adaptive-fusion-run',
    stage: 'adaptive-spn-fusion-contract',
  });
  fusionRuntime.applySchedulerDecision({
    schema: FOREGROUND_BUDGET_GOVERNOR_SCHEMA,
    routeId: fusionRuntime.routeId,
    status: 'adjusted',
    action: 'reduce-phase-chunk',
    target: 'spnFusionOutputItems',
    schedulerChanged: true,
    applicationAuthority: 'decision-state-only-not-runtime-application',
    revision: 1,
    observation: {
      episodeId: 'sharp-adaptive-fusion',
      episodeEpochId: 'sharp-adaptive-fusion-epoch',
      firingId: 'sharp-adaptive-fusion-reduction',
      maxFrameGapMs: 50,
      targetFrameGapMs: 16,
    },
    previousScheduler: fusionApplication.scheduler,
    effectiveScheduler: {
      ...fusionApplication.scheduler,
      phaseChunkSize: {
        ...fusionApplication.scheduler.phaseChunkSize,
        spnFusionOutputItems: 4,
      },
    },
    failures: [],
  });
  const chunkDetails = {
    role: 'spn-fusion-output-chunk',
    outputStart: 0,
    outputEnd: 4,
    outputCount: 4,
    totalOutputItems: 16,
  };
  try {
    const submitCountBeforePrepare = fakeQueueSubmitCount;
    const liveDuty = await sharpSchedulerModule.prepareSchedulerDutyBeforeEncode(
      adaptiveFusionScheduler,
      adaptiveFusionTelemetry,
      'spn-fusion',
      { role: 'spn-fusion-before-next-output-encode', totalOutputItems: 16 },
    );
    assert.equal(adaptiveFusionScheduler.effective.spnFusionChunkItems, 4, 'the next SPN range must consume the refreshed live control');
    assert.equal(fakeQueueSubmitCount, submitCountBeforePrepare, 'preparing the next duty must not submit or claim command work');
    assert.equal(fusionRuntime.commandDuties.snapshot().submissionCount, 0, 'preparation alone must leave command-duty submission evidence empty');
    assert.equal(
      fusionRuntime.schedulerApplication.snapshot().boundaries.at(-1)?.status,
      'pending-encode-validation',
      'the refreshed boundary must remain pending until the exact next range is encoded',
    );

    await sharpSchedulerModule.submitPreparedSchedulerDuty(
      adaptiveFusionScheduler,
      fakeDevice,
      adaptiveFusionTelemetry,
      liveDuty,
      [{}],
      'spn-fusion',
      chunkDetails,
    );
    assert.equal(fakeQueueSubmitCount, submitCountBeforePrepare + 1, 'the prepared duty must wrap one real queue.submit call');
    assert.equal(fusionRuntime.commandDuties.snapshot().submissionCount, 1);
    assert.equal(fusionRuntime.commandDuties.snapshot().submissions[0].outcome, 'succeeded');
    assert.equal(fusionRuntime.schedulerApplication.snapshot().boundaries.at(-1)?.status, 'encoded');

    await schedulerYield(
      adaptiveFusionScheduler,
      fakeDevice,
      adaptiveFusionTelemetry,
      'spn-fusion',
      chunkDetails,
      0,
    );
    assert.equal(fusionRuntime.commandDuties.snapshot().submissionCount, 1, 'queue drain must not manufacture another command-duty submission');

    await schedulerYield(
      adaptiveFusionScheduler,
      fakeDevice,
      adaptiveFusionTelemetry,
      'spn-fusion',
      { block: 'upsample_latent0', role: 'group-complete', layerCount: 4 },
      0,
    );
    assert.equal(fakeQueueSubmitCount, submitCountBeforePrepare + 1, 'group-complete drain must not claim or perform another submit');
    assert.equal(fusionRuntime.commandDuties.snapshot().submissionCount, 1, 'plain production drain must not manufacture a command-duty row');
  } finally {
    detachSharpLiveScheduler(adaptiveFusionScheduler);
  }
});
const adaptiveFusionPrepared = schedulerTelemetrySnapshot(adaptiveFusionTelemetry).events.find(
  event => event.kind === 'live-scheduler-duty-prepared',
);
assert.equal(adaptiveFusionPrepared?.controlId, 'spnFusionOutputItems');
assert.equal(adaptiveFusionPrepared?.current, 4);

const vitRuntime = await createSharpRouteRuntime({
  device: fakeDevice,
  adapter: fakeAdapter,
}, {
  routeDefinition: definition,
  browser: 'node-sharp-runtime-adaptive-vit-contract',
  runId: 'sharp-route-adaptive-vit-run',
  clock: routeRunClock,
  scheduler: {
    mode: 'cooperative',
    yieldMs: 0,
    waitForSubmittedWorkDone: true,
    phaseChunkSize: { spnPatch: 1, vitBlock: 4 },
  },
  schedulerBounds: {
    yieldMs: { min: 0, max: 20, step: 1 },
    phaseChunkSize: {
      spnPatch: { min: 1, max: 35, stepFactor: 2 },
      vitBlock: { min: 1, max: 24, stepFactor: 2 },
    },
  },
});
const adaptiveVitScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    vitBlockChunkSize: 4,
    yieldMs: 0,
    waitForSubmittedWorkDone: true,
  },
});
const adaptiveVitTelemetry = createSharpRunTelemetry(adaptiveVitScheduler, {
  runId: 'sharp-adaptive-vit-control-run',
});
await vitRuntime.runInvocation({ invocationId: 'sharp-adaptive-vit-invocation' }, async invocation => {
  attachSharpLiveScheduler(adaptiveVitScheduler, {
    runtime: vitRuntime,
    invocation,
    runId: 'sharp-route-adaptive-vit-run',
    stage: 'adaptive-vit-contract',
  });
  try {
    const firstDuty = await sharpSchedulerModule.prepareSchedulerDutyBeforeEncode(
      adaptiveVitScheduler,
      adaptiveVitTelemetry,
      'vit-block-chunk',
      { role: 'vit-before-next-block-encode', blockStart: 0, totalBlocks: 24 },
    );
    const firstRange = sharpSchedulerModule.planNextVitBlockChunk(
      24,
      0,
      adaptiveVitScheduler.effective.vitBlockChunkSize,
      0,
    );
    assert.deepEqual(firstRange, {
      blockChunkIndex: 0,
      blockStart: 0,
      blockEnd: 4,
      blockCount: 4,
      totalBlocks: 24,
    });
    await sharpSchedulerModule.submitPreparedSchedulerDuty(
      adaptiveVitScheduler,
      fakeDevice,
      adaptiveVitTelemetry,
      firstDuty,
      [{}],
      'vit-block-chunk',
      firstRange,
    );

    const previousVitScheduler = vitRuntime.schedulerSnapshot().scheduler;
    vitRuntime.applySchedulerDecision({
      schema: FOREGROUND_BUDGET_GOVERNOR_SCHEMA,
      routeId: vitRuntime.routeId,
      status: 'adjusted',
      action: 'reduce-phase-chunk',
      target: 'vitBlock',
      schedulerChanged: true,
      applicationAuthority: 'decision-state-only-not-runtime-application',
      revision: 1,
      observation: {
        episodeId: 'sharp-adaptive-vit',
        episodeEpochId: 'sharp-adaptive-vit-epoch',
        firingId: 'sharp-adaptive-vit-reduction',
        maxFrameGapMs: 50,
        targetFrameGapMs: 16,
      },
      previousScheduler: previousVitScheduler,
      effectiveScheduler: {
        ...previousVitScheduler,
        phaseChunkSize: {
          ...previousVitScheduler.phaseChunkSize,
          vitBlock: 2,
        },
      },
      failures: [],
    });
    const secondDuty = await sharpSchedulerModule.prepareSchedulerDutyBeforeEncode(
      adaptiveVitScheduler,
      adaptiveVitTelemetry,
      'vit-block-chunk',
      { role: 'vit-before-next-block-encode', blockStart: firstRange.blockEnd, totalBlocks: 24 },
    );
    assert.equal(adaptiveVitScheduler.effective.vitBlockChunkSize, 2);
    const secondRange = sharpSchedulerModule.planNextVitBlockChunk(
      24,
      firstRange.blockEnd,
      adaptiveVitScheduler.effective.vitBlockChunkSize,
      1,
    );
    assert.deepEqual(secondRange, {
      blockChunkIndex: 1,
      blockStart: 4,
      blockEnd: 6,
      blockCount: 2,
      totalBlocks: 24,
    });
    await sharpSchedulerModule.submitPreparedSchedulerDuty(
      adaptiveVitScheduler,
      fakeDevice,
      adaptiveVitTelemetry,
      secondDuty,
      [{}],
      'vit-block-chunk',
      secondRange,
    );
  } finally {
    detachSharpLiveScheduler(adaptiveVitScheduler);
  }
});
assert.equal(vitRuntime.commandDuties.snapshot().submissionCount, 2);
assert.deepEqual(
  vitRuntime.commandDuties.snapshot().submissions.map(row => [
    row.descriptor.metadata.blockStart,
    row.descriptor.metadata.blockEnd,
  ]),
  [[0, 4], [4, 6]],
  'live ViT reduction must preserve exact contiguous submitted ranges',
);

const fourStageScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    vitBlockChunkSize: 1,
    vitMicroduty: true,
    vitMicrodutyMode: 'four-stage',
    yieldMs: 0,
    waitForSubmittedWorkDone: true,
  },
});
const fourStageTelemetry = createSharpRunTelemetry(fourStageScheduler, {
  runId: 'sharp-four-stage-vit-run',
});
const fourStageSubmissionStart = vitRuntime.commandDuties.snapshot().submissionCount;
await vitRuntime.runInvocation({ invocationId: 'sharp-four-stage-vit-invocation' }, async invocation => {
  attachSharpLiveScheduler(fourStageScheduler, {
    runtime: vitRuntime,
    invocation,
    runId: 'sharp-route-four-stage-vit-run',
    stage: 'four-stage-vit-contract',
  });
  try {
    const range = sharpSchedulerModule.planNextVitBlockChunk(24, 0, 1, 0);
    const microduties = sharpSchedulerModule.planVitBlockMicroduties(
      range,
      fourStageScheduler.effective.vitMicrodutyMode,
    );
    for (const microduty of microduties) {
      const dutyPhase = microduty.microdutyIndex === 0
        ? 'vit-block-chunk'
        : 'vit-block-microphase';
      const duty = await sharpSchedulerModule.prepareSchedulerDutyBeforeEncode(
        fourStageScheduler,
        fourStageTelemetry,
        dutyPhase,
        {
          role: 'vit-before-next-microphase-encode',
          microdutyMode: fourStageScheduler.effective.vitMicrodutyMode,
          ...range,
          ...microduty,
        },
      );
      await sharpSchedulerModule.submitPreparedSchedulerDuty(
        fourStageScheduler,
        fakeDevice,
        fourStageTelemetry,
        duty,
        [{}],
        dutyPhase,
        {
          role: 'vit-block-microduty',
          microdutyMode: fourStageScheduler.effective.vitMicrodutyMode,
          ...range,
          ...microduty,
        },
      );
    }
  } finally {
    detachSharpLiveScheduler(fourStageScheduler);
  }
});
const fourStageSubmissions = vitRuntime.commandDuties.snapshot().submissions.slice(fourStageSubmissionStart);
assert.equal(fourStageSubmissions.length, 4, 'one four-stage ViT block must produce four real command-duty submissions');
assert.deepEqual(
  fourStageSubmissions.map(row => [
    row.descriptor.metadata.microdutyMode,
    row.descriptor.metadata.blockIndex,
    row.descriptor.metadata.microphase,
  ]),
  [
    ['four-stage', 0, 'norm1-qkv'],
    ['four-stage', 0, 'attention-projection-residual'],
    ['four-stage', 0, 'norm2-fc1'],
    ['four-stage', 0, 'fc2-residual'],
  ],
  'four-stage command evidence must preserve exact mode, block, and dependency order',
);

const dispatchMajorScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    vitBlockChunkSize: 1,
    vitMicroduty: true,
    vitMicrodutyMode: 'dispatch-major',
    yieldMs: 0,
    waitForSubmittedWorkDone: true,
  },
});
const dispatchMajorTelemetry = createSharpRunTelemetry(dispatchMajorScheduler, {
  runId: 'sharp-dispatch-major-vit-run',
});
const dispatchMajorSubmissionStart = vitRuntime.commandDuties.snapshot().submissionCount;
await vitRuntime.runInvocation({ invocationId: 'sharp-dispatch-major-vit-invocation' }, async invocation => {
  attachSharpLiveScheduler(dispatchMajorScheduler, {
    runtime: vitRuntime,
    invocation,
    runId: 'sharp-route-dispatch-major-vit-run',
    stage: 'dispatch-major-vit-contract',
  });
  try {
    const patchDetails = {
      role: 'vit-patch-embed-duty',
      microdutyMode: 'dispatch-major',
      microdutyIndex: -1,
      blockIndex: null,
      microphase: 'patch-embed',
    };
    const patchDuty = await sharpSchedulerModule.prepareSchedulerDutyBeforeEncode(
      dispatchMajorScheduler,
      dispatchMajorTelemetry,
      'vit-patch-embed',
      patchDetails,
    );
    await sharpSchedulerModule.submitPreparedSchedulerDuty(
      dispatchMajorScheduler,
      fakeDevice,
      dispatchMajorTelemetry,
      patchDuty,
      [{}],
      'vit-patch-embed',
      patchDetails,
    );
    await schedulerYield(
      dispatchMajorScheduler,
      fakeDevice,
      dispatchMajorTelemetry,
      'vit-patch-embed',
      patchDetails,
    );

    const range = sharpSchedulerModule.planNextVitBlockChunk(24, 0, 1, 0);
    const microduties = sharpSchedulerModule.planVitBlockMicroduties(
      range,
      dispatchMajorScheduler.effective.vitMicrodutyMode,
    );
    for (const microduty of microduties) {
      const dutyPhase = microduty.microdutyIndex === 0
        ? 'vit-block-chunk'
        : 'vit-block-microphase';
      const details = {
        role: 'vit-block-microduty',
        microdutyMode: dispatchMajorScheduler.effective.vitMicrodutyMode,
        ...range,
        ...microduty,
      };
      const duty = await sharpSchedulerModule.prepareSchedulerDutyBeforeEncode(
        dispatchMajorScheduler,
        dispatchMajorTelemetry,
        dutyPhase,
        details,
      );
      await sharpSchedulerModule.submitPreparedSchedulerDuty(
        dispatchMajorScheduler,
        fakeDevice,
        dispatchMajorTelemetry,
        duty,
        [{}],
        dutyPhase,
        details,
      );
      await schedulerYield(
        dispatchMajorScheduler,
        fakeDevice,
        dispatchMajorTelemetry,
        dutyPhase,
        details,
      );
    }
  } finally {
    detachSharpLiveScheduler(dispatchMajorScheduler);
  }
});
const dispatchMajorSubmissions = vitRuntime.commandDuties.snapshot().submissions.slice(dispatchMajorSubmissionStart);
assert.equal(dispatchMajorSubmissions.length, 13, 'dispatch-major ViT must submit patch embedding plus twelve block duties');
assert.deepEqual(
  dispatchMajorSubmissions.map(row => [
    row.descriptor.metadata.microdutyMode,
    row.descriptor.metadata.blockIndex,
    row.descriptor.metadata.microphase,
  ]),
  [
    ['dispatch-major', null, 'patch-embed'],
    ['dispatch-major', 0, 'norm1'],
    ['dispatch-major', 0, 'qkv-projection'],
    ['dispatch-major', 0, 'qkv-split'],
    ['dispatch-major', 0, 'attention-scores'],
    ['dispatch-major', 0, 'attention-softmax'],
    ['dispatch-major', 0, 'attention-apply'],
    ['dispatch-major', 0, 'attention-projection'],
    ['dispatch-major', 0, 'attention-residual'],
    ['dispatch-major', 0, 'norm2'],
    ['dispatch-major', 0, 'fc1'],
    ['dispatch-major', 0, 'fc2'],
    ['dispatch-major', 0, 'mlp-residual'],
  ],
  'dispatch-major command evidence must retain patch and dominant dispatch dependency order',
);

const tiledDispatchScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    vitBlockChunkSize: 1,
    vitMicroduty: true,
    vitMicrodutyMode: 'dispatch-major',
    vitLinearTileItems: 6,
    vitAttentionTileItems: 3,
    vitSoftmaxTileRows: 2,
    vitNormTileRows: 1,
    yieldMs: 0,
    waitForSubmittedWorkDone: true,
  },
});
const tiledDispatchTelemetry = createSharpRunTelemetry(tiledDispatchScheduler, {
  runId: 'sharp-tiled-dispatch-vit-run',
});
const tiledSubmissionStart = vitRuntime.commandDuties.snapshot().submissionCount;
const tiledQueueSubmitStart = fakeQueueSubmitCount;
await vitRuntime.runInvocation({ invocationId: 'sharp-tiled-dispatch-vit-invocation' }, async invocation => {
  attachSharpLiveScheduler(tiledDispatchScheduler, {
    runtime: vitRuntime,
    invocation,
    runId: 'sharp-route-tiled-dispatch-vit-run',
    stage: 'tiled-dispatch-vit-contract',
  });
  try {
    const range = sharpSchedulerModule.planNextVitBlockChunk(24, 0, 1, 0);
    const microduties = sharpSchedulerModule.planVitBlockMicroduties(
      range,
      tiledDispatchScheduler.effective.vitMicrodutyMode,
      {
        tokenCount: 2,
        dim: 4,
        mlpHiddenDim: 8,
        numHeads: 2,
        linearTileItems: tiledDispatchScheduler.effective.vitLinearTileItems,
        attentionTileItems: tiledDispatchScheduler.effective.vitAttentionTileItems,
        softmaxTileRows: tiledDispatchScheduler.effective.vitSoftmaxTileRows,
        normTileRows: tiledDispatchScheduler.effective.vitNormTileRows,
      },
    );
    for (const microduty of microduties) {
      const dutyPhase = microduty.microdutyIndex === 0
        ? 'vit-block-chunk'
        : 'vit-block-microphase';
      const details = {
        role: 'vit-block-microduty',
        microdutyMode: tiledDispatchScheduler.effective.vitMicrodutyMode,
        ...range,
        ...microduty,
      };
      const duty = await sharpSchedulerModule.prepareSchedulerDutyBeforeEncode(
        tiledDispatchScheduler,
        tiledDispatchTelemetry,
        dutyPhase,
        details,
      );
      await sharpSchedulerModule.submitPreparedSchedulerDuty(
        tiledDispatchScheduler,
        fakeDevice,
        tiledDispatchTelemetry,
        duty,
        [{}],
        dutyPhase,
        details,
      );
      await schedulerYield(
        tiledDispatchScheduler,
        fakeDevice,
        tiledDispatchTelemetry,
        dutyPhase,
        details,
      );
    }
  } finally {
    detachSharpLiveScheduler(tiledDispatchScheduler);
  }
});
const tiledDispatchSubmissions = vitRuntime.commandDuties.snapshot().submissions.slice(tiledSubmissionStart);
assert.equal(tiledDispatchSubmissions.length, 26, 'one tiny tiled block plan must produce 26 real command-duty submissions');
assert.equal(fakeQueueSubmitCount - tiledQueueSubmitStart, 26, 'every tiled command row must correspond one-for-one with a queue submission');
const tiledRows = tiledDispatchSubmissions.filter(row => Number.isSafeInteger(row.descriptor.metadata.tileIndex));
assert.equal(tiledRows.length, 23, 'only the dominant dispatch and LayerNorm classes may expand into tile rows');
assert.deepEqual(
  tiledRows.filter(row => row.descriptor.metadata.microphase === 'attention-softmax').map(row => [
    row.descriptor.metadata.tileIndex,
    row.descriptor.metadata.tileStart,
    row.descriptor.metadata.tileEnd,
    row.descriptor.metadata.tileUnit,
  ]),
  [[0, 0, 2, 'row'], [1, 2, 4, 'row']],
  'real command evidence must preserve complete-row softmax tile identity',
);

const prepareFailureScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    vitBlockChunkSize: 1,
    vitMicroduty: true,
    vitMicrodutyMode: 'four-stage',
    yieldMs: 0,
    waitForSubmittedWorkDone: true,
  },
});
const prepareFailureTelemetry = createSharpRunTelemetry(prepareFailureScheduler, {
  runId: 'sharp-four-stage-prepare-failure-run',
});
const prepareFailure = new Error('forced live microphase prepare failure');
let prepareFailureCalls = 0;
const queueSubmissionsBeforePrepareFailure = fakeQueueSubmitCount;
attachSharpLiveScheduler(prepareFailureScheduler, {
  runtime: {
    routeId: 'sharp-four-stage-prepare-failure-route',
    device: fakeDevice,
    queue: fakeQueue,
    async prepareCommandDutyAtBoundary() {
      prepareFailureCalls += 1;
      throw prepareFailure;
    },
  },
  invocation: {
    invocationId: 'sharp-four-stage-prepare-failure-invocation',
    bounds: { phaseChunkSize: {} },
  },
  runId: 'sharp-four-stage-prepare-failure-route-run',
  stage: 'four-stage-vit-prepare-failure-contract',
});
try {
  await assert.rejects(
    async () => {
      const duty = await sharpSchedulerModule.prepareSchedulerDutyBeforeEncode(
        prepareFailureScheduler,
        prepareFailureTelemetry,
        'vit-block-microphase',
        {
          role: 'vit-before-next-microphase-encode',
          microdutyMode: 'four-stage',
          blockIndex: 0,
          microphase: 'attention-projection-residual',
        },
      );
      await sharpSchedulerModule.submitPreparedSchedulerDuty(
        prepareFailureScheduler,
        fakeDevice,
        prepareFailureTelemetry,
        duty,
        [{}],
        'vit-block-microphase',
        {
          role: 'vit-block-microduty',
          microdutyMode: 'four-stage',
          blockIndex: 0,
          microphase: 'attention-projection-residual',
        },
      );
      await schedulerYield(
        prepareFailureScheduler,
        fakeDevice,
        prepareFailureTelemetry,
        'vit-block-microphase',
        {
          role: 'vit-block-microduty',
          microdutyMode: 'four-stage',
          blockIndex: 0,
          microphase: 'attention-projection-residual',
        },
      );
    },
    error => error === prepareFailure,
    'a live microphase prepare failure must reject with source identity before encode, submit, or yield',
  );
} finally {
  detachSharpLiveScheduler(prepareFailureScheduler);
}
assert.equal(prepareFailureCalls, 1, 'the failing live runtime must receive exactly one prepare attempt');
assert.equal(fakeQueueSubmitCount, queueSubmissionsBeforePrepareFailure, 'failed live prepare must submit zero untracked GPU work');
const prepareFailureEvents = schedulerTelemetrySnapshot(prepareFailureTelemetry).events;
assert.equal(
  prepareFailureEvents.filter(event => event.kind === 'live-scheduler-duty-failed').length,
  1,
  'failed live prepare must retain one explicit scheduler failure event',
);
assert.equal(
  prepareFailureEvents.filter(event => event.kind === 'chunk-start').length,
  0,
  'failed live prepare must not continue into normal yield telemetry',
);

const staticMicrophaseScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    vitMicroduty: true,
    vitMicrodutyMode: 'four-stage',
  },
});
const staticMicrophaseTelemetry = createSharpRunTelemetry(staticMicrophaseScheduler, {
  runId: 'sharp-four-stage-static-run',
});
const staticMicrophaseDuty = await sharpSchedulerModule.prepareSchedulerDutyBeforeEncode(
  staticMicrophaseScheduler,
  staticMicrophaseTelemetry,
  'vit-block-microphase',
  {
    role: 'vit-before-next-microphase-encode',
    microdutyMode: 'four-stage',
    blockIndex: 0,
    microphase: 'attention-projection-residual',
  },
);
assert.equal(staticMicrophaseDuty, null, 'standalone SHARP must preserve the no-live-runtime duty sentinel');
const queueSubmissionsBeforeStaticMicrophase = fakeQueueSubmitCount;
await sharpSchedulerModule.submitPreparedSchedulerDuty(
  staticMicrophaseScheduler,
  fakeDevice,
  staticMicrophaseTelemetry,
  staticMicrophaseDuty,
  [{}],
  'vit-block-microphase',
  {
    role: 'vit-block-microduty',
    microdutyMode: 'four-stage',
    blockIndex: 0,
    microphase: 'attention-projection-residual',
  },
);
assert.equal(
  fakeQueueSubmitCount,
  queueSubmissionsBeforeStaticMicrophase + 1,
  'standalone SHARP must continue to submit directly without a live command-duty runtime',
);

const failedPlanScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    spnFusionChunkItems: 4,
    yieldMs: 0,
  },
});
const failedPlanTelemetry = createSharpRunTelemetry(failedPlanScheduler, {
  runId: 'sharp-adaptive-fusion-failed-plan-run',
});
await fusionRuntime.runInvocation({ invocationId: 'sharp-adaptive-fusion-failed-plan-invocation' }, async invocation => {
  attachSharpLiveScheduler(failedPlanScheduler, {
    runtime: fusionRuntime,
    invocation,
    runId: 'sharp-route-adaptive-fusion-run',
    stage: 'adaptive-spn-fusion-failed-plan-contract',
  });
  try {
    const liveDuty = await sharpSchedulerModule.prepareSchedulerDutyBeforeEncode(
      failedPlanScheduler,
      failedPlanTelemetry,
      'spn-fusion',
      { role: 'spn-fusion-before-next-output-encode', totalOutputItems: 16 },
    );
    sharpSchedulerModule.failPreparedSchedulerDuty(
      failedPlanScheduler,
      failedPlanTelemetry,
      liveDuty,
      'spn-fusion',
      new Error('range planning failed'),
    );
  } finally {
    detachSharpLiveScheduler(failedPlanScheduler);
  }
});
assert.equal(fusionRuntime.schedulerApplication.snapshot().boundaries.at(-1)?.status, 'failed-before-encode');
assert.equal(fusionRuntime.commandDuties.snapshot().submissionCount, 1, 'failed planning must not add command-duty submission evidence');

const failedSubmitScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    spnFusionChunkItems: 4,
    yieldMs: 0,
  },
});
const failedSubmitTelemetry = createSharpRunTelemetry(failedSubmitScheduler, {
  runId: 'sharp-adaptive-fusion-failed-submit-run',
});
await fusionRuntime.runInvocation({ invocationId: 'sharp-adaptive-fusion-failed-submit-invocation' }, async invocation => {
  attachSharpLiveScheduler(failedSubmitScheduler, {
    runtime: fusionRuntime,
    invocation,
    runId: 'sharp-route-adaptive-fusion-run',
    stage: 'adaptive-spn-fusion-failed-submit-contract',
  });
  try {
    const liveDuty = await sharpSchedulerModule.prepareSchedulerDutyBeforeEncode(
      failedSubmitScheduler,
      failedSubmitTelemetry,
      'spn-fusion',
      { role: 'spn-fusion-before-next-output-encode', totalOutputItems: 16 },
    );
    await assert.rejects(
      () => sharpSchedulerModule.submitPreparedSchedulerDuty(
        failedSubmitScheduler,
        { queue: { submit() { throw new Error('queue submit failed'); } } },
        failedSubmitTelemetry,
        liveDuty,
        [{}],
        'spn-fusion',
        { role: 'spn-fusion-output-chunk', outputStart: 0, outputEnd: 4, outputCount: 4, totalOutputItems: 16 },
      ),
      /queue submit failed/,
    );
  } finally {
    detachSharpLiveScheduler(failedSubmitScheduler);
  }
});
const failedSubmissionReport = fusionRuntime.commandDuties.snapshot();
assert.equal(fusionRuntime.schedulerApplication.snapshot().boundaries.at(-1)?.status, 'encoded', 'finished command encoding remains distinct from failed submission');
assert.ok(failedSubmissionReport.failure, 'failed queue submission must make the recorder failure explicit before finish');
assert.equal(failedSubmissionReport.submissions.at(-1)?.outcome, 'failed');
assert.equal(failedSubmissionReport.submissions.at(-1)?.descriptor.metadata.outputStart, 0);
assert.equal(failedSubmissionReport.submissions.at(-1)?.descriptor.metadata.outputEnd, 4);

const missingFusionBoundsScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    spnFusionChunkItems: 8,
    yieldMs: 0,
  },
});
const missingFusionBoundsTelemetry = createSharpRunTelemetry(missingFusionBoundsScheduler, {
  runId: 'sharp-missing-fusion-bounds-run',
});
await runtime.runInvocation({ invocationId: 'sharp-missing-fusion-bounds-invocation' }, async invocation => {
  attachSharpLiveScheduler(missingFusionBoundsScheduler, {
    runtime,
    invocation,
    runId: routeRunId,
    stage: 'missing-fusion-bounds-contract',
  });
  try {
    await assert.rejects(
      () => sharpSchedulerModule.prepareSchedulerDutyBeforeEncode(
        missingFusionBoundsScheduler,
        missingFusionBoundsTelemetry,
        'spn-fusion',
        { role: 'spn-fusion-output-chunk', outputStart: 0, outputEnd: 8, outputCount: 8, totalOutputItems: 16 },
      ),
      /undeclared scheduler control spnFusionOutputItems/,
      'enabled adaptive tiling must fail loud when runtime invocation bounds omit its control',
    );
  } finally {
    detachSharpLiveScheduler(missingFusionBoundsScheduler);
  }
});
const missingFusionBoundsEvents = schedulerTelemetrySnapshot(missingFusionBoundsTelemetry).events;
assert.ok(missingFusionBoundsEvents.some(event => event.kind === 'live-scheduler-duty-failed'));
assert.equal(missingFusionBoundsEvents.some(event => event.kind === 'chunk-start'), false);

const preprocessed = await runtime.runHostPhase(
  WEBGPU_HOST_PHASE.cpuPreprocess,
  async () => 'preprocessed',
  { detail: { operation: 'source-image-normalization', imageWidth: 1536 } },
);
assert.equal(preprocessed, 'preprocessed');

const previousScheduler = {
  mode: 'cooperative',
  yieldMs: 3,
  waitForSubmittedWorkDone: true,
  phaseChunkSize: { spnPatch: 1, vitBlock: 2 },
};
const effectiveScheduler = {
  mode: 'cooperative',
  yieldMs: 3,
  waitForSubmittedWorkDone: true,
  phaseChunkSize: { spnPatch: 2, vitBlock: 2 },
};
runtime.applySchedulerDecision({
  schema: FOREGROUND_BUDGET_GOVERNOR_SCHEMA,
  routeId: runtime.routeId,
  status: 'relaxed',
  action: 'relax-phase-chunk',
  target: 'spnPatch',
  schedulerChanged: true,
  applicationAuthority: 'decision-state-only-not-runtime-application',
  revision: 1,
  observation: {
    episodeId: 'sharp-contract-live-scheduler',
    episodeEpochId: 'sharp-contract-live-scheduler-epoch',
    firingId: 'sharp-contract-live-spn-relaxation',
    maxFrameGapMs: 40,
    targetFrameGapMs: 16,
  },
  previousScheduler,
  effectiveScheduler,
  failures: [],
});

for (const stageName of definition.requiredStages) {
  const result = await runtime.runStage(stageName, async stage => {
    assert.equal(typeof stage.yieldToBrowser, 'function');
    return `${stageName}:ok`;
  }, {
    routeStage: stageName,
  });
  assert.equal(result, `${stageName}:ok`);
}

let allBoundaryForegroundTelemetry;
await runtime.runInvocation({ invocationId: 'sharp-contract-live-scheduler-invocation' }, async invocation => {
  assert.equal(invocation.schedulerRevision, 1, 'new invocation must consume the scheduler decision revision');
  const foregroundRequest = runtime.requestForegroundOpportunity({
    requestId: 'sharp-contract-kiln-frame-demand',
    metadata: {
      source: 'contract-kiln-frame-demand',
      reason: 'foreground-frame-before-next-sharp-command',
      kind: 'kiln-frame',
      targetFrameBudgetMs: 16.7,
    },
    run: async service => {
      assert.equal(service.routeId, runtime.routeId);
      assert.equal(service.runId, routeRunId);
      service.submit([{}], {
        submissionId: 'sharp-contract-kiln-frame-submit',
        metadata: { kind: 'kiln-frame' },
      });
      return { serviced: true };
    },
  });
  const descriptor = await runtime.prepareCommandDutyAtBoundary({
    phase: 'spn',
    kind: 'compute',
    chunkControl: {
      controlId: 'spnPatch',
      unit: 'patch',
      current: invocation.getControl('spnPatch'),
      bounds: { min: 1, max: 35, stepFactor: 2 },
    },
  }, invocation);
  const foregroundReceipt = await foregroundRequest.completion;
  assert.equal(foregroundReceipt.status, 'completed', 'foreground demand must settle before the next duty is encoded');
  assert.equal(descriptor.chunkControl.current, 2, 'next duty must be constructed from the refreshed scheduler control');
  assert.equal(descriptor.metadata.schedulerBoundary.status, 'pending-encode-validation');
  assert.equal(descriptor.metadata.schedulerBoundary.effectiveSchedulerRevision, 1);
  assert.equal(descriptor.metadata.foregroundOpportunityService.status, 'serviced');
  assert.equal(descriptor.metadata.foregroundOpportunityService.servicedRequestCount, 1);
  runtime.settleCommandDuty(descriptor, { status: 'encoded' });
  await runtime.commandDuties.measureSubmission(descriptor, () => {
    fakeDevice.queue.submit([{}]);
  });

  const controlLessScheduler = parseSharpSchedulerConfig({
    sharpScheduler: {
      mode: 'cooperative',
      yieldMs: 0,
      waitForSubmittedWorkDone: true,
      gaussianPhaseYieldMs: 0,
      routeTailYieldMs: 0,
    },
  });
  const controlLessTelemetry = createSharpRunTelemetry(controlLessScheduler, {
    runId: 'sharp-contract-all-boundary-foreground-run',
  });
  let foregroundSequence = 0;
  attachSharpLiveScheduler(controlLessScheduler, {
    runtime,
    invocation,
    runId: routeRunId,
    stage: 'all-boundary-contract',
    foregroundOpportunityHook({ boundary }) {
      foregroundSequence += 1;
      return {
        requestId: `sharp-contract-${boundary}-foreground-${foregroundSequence}`,
        metadata: { boundary, kind: 'kiln-frame' },
        async run(service) {
          service.submit([{}], {
            submissionId: `sharp-contract-${boundary}-submit-${foregroundSequence}`,
            metadata: { boundary, kind: 'kiln-frame' },
          });
          return { servicedBoundary: boundary };
        },
      };
    },
  });
  try {
    for (const boundary of [
      'spn-fusion',
      'spn-image-encoder',
      'monodepth-phase',
      'gaussian-phase',
      'route-tail',
    ]) {
      await schedulerYield(
        controlLessScheduler,
        fakeDevice,
        controlLessTelemetry,
        boundary,
        { contractBoundary: boundary },
        0,
      );
    }
  } finally {
    detachSharpLiveScheduler(controlLessScheduler);
  }
  allBoundaryForegroundTelemetry = schedulerTelemetrySnapshot(controlLessTelemetry);

  const failingForegroundScheduler = parseSharpSchedulerConfig({
    sharpScheduler: { mode: 'cooperative', yieldMs: 0 },
  });
  const failingForegroundTelemetry = createSharpRunTelemetry(failingForegroundScheduler, {
    runId: 'sharp-contract-failing-foreground-run',
  });
  attachSharpLiveScheduler(failingForegroundScheduler, {
    runtime,
    invocation,
    runId: routeRunId,
    stage: 'foreground-failure-contract',
    foregroundOpportunityHook() {
      throw new Error('foreground callback exploded');
    },
  });
  try {
    await assert.rejects(
      () => schedulerYield(
        failingForegroundScheduler,
        fakeDevice,
        failingForegroundTelemetry,
        'spn-fusion',
        { contractBoundary: 'spn-fusion-failure' },
        0,
      ),
      /foreground callback exploded/,
      'foreground callback setup failure must reject before the next inference encode',
    );
  } finally {
    detachSharpLiveScheduler(failingForegroundScheduler);
  }
  const failureEvents = schedulerTelemetrySnapshot(failingForegroundTelemetry).events;
  assert.ok(failureEvents.some(event => event.kind === 'foreground-opportunity-request-failed'));
  assert.equal(failureEvents.some(event => event.kind === 'live-scheduler-duty-prepared'), false);
  assert.equal(failureEvents.some(event => event.kind === 'chunk-start'), false);

  const noDemandScheduler = parseSharpSchedulerConfig({
    sharpScheduler: { mode: 'cooperative', yieldMs: 0 },
  });
  const noDemandTelemetry = createSharpRunTelemetry(noDemandScheduler, {
    runId: 'sharp-contract-no-foreground-demand-run',
  });
  attachSharpLiveScheduler(noDemandScheduler, {
    runtime,
    invocation,
    runId: routeRunId,
    stage: 'foreground-no-demand-contract',
    foregroundOpportunityHook() { return null; },
  });
  try {
    await schedulerYield(
      noDemandScheduler,
      fakeDevice,
      noDemandTelemetry,
      'spn-fusion',
      { contractBoundary: 'spn-fusion-no-demand' },
      0,
    );
  } finally {
    detachSharpLiveScheduler(noDemandScheduler);
  }
  const noDemandEvents = schedulerTelemetrySnapshot(noDemandTelemetry).events;
  assert.equal(noDemandEvents.some(event => event.kind === 'foreground-opportunity-request-failed'), false);
  assert.equal(noDemandEvents.some(event => event.kind === 'foreground-opportunity-requested'), false);
  assert.equal(noDemandEvents.some(event => event.kind === 'foreground-opportunity-serviced'), false);
  assert.equal(noDemandEvents.some(event => event.kind === 'live-scheduler-duty-prepared'), false);
});

for (const boundary of [
  'spn-fusion',
  'spn-image-encoder',
  'monodepth-phase',
  'gaussian-phase',
  'route-tail',
]) {
  const events = allBoundaryForegroundTelemetry.events.filter(event => event.boundary === boundary);
  assert.ok(
    events.some(event => event.kind === 'foreground-opportunity-requested'),
    `${boundary} must request foreground work before the next inference encode`,
  );
  assert.ok(
    events.some(event => event.kind === 'foreground-opportunity-serviced'),
    `${boundary} must await foreground service before the next inference encode`,
  );
  assert.equal(
    events.some(event => event.kind === 'live-scheduler-duty-prepared'),
    false,
    `${boundary} drain must not impersonate a command duty`,
  );
}

const commandDutyReport = runtime.finishCommandDuties();
assert.equal(commandDutyReport.schema, WEBGPU_COMMAND_DUTY_REPORT_SCHEMA);
assert.equal(commandDutyReport.status, 'succeeded');
assert.equal(commandDutyReport.retention, 'uncapped');
assert.equal(commandDutyReport.submissionCount, 1, 'foreground service and queue drains must not manufacture inference submissions');
assert.equal(commandDutyReport.submissions[0].descriptor.chunkControl.current, 2);
assert.equal(commandDutyReport.submissions[0].descriptor.metadata.schedulerBoundary.status, 'encoded');

const foregroundInterlockReport = runtime.finishForegroundOpportunities();
assert.equal(foregroundInterlockReport.retention, 'uncapped');
assert.equal(foregroundInterlockReport.receipts[0].requestId, 'sharp-contract-kiln-frame-demand');
assert.equal(foregroundInterlockReport.receipts[0].status, 'completed');
assert.equal(foregroundInterlockReport.receipts[0].submissionCount, 1);
assert.equal(foregroundInterlockReport.services[0].routeId, runtime.routeId);
assert.equal(foregroundInterlockReport.services[0].runId, routeRunId);

const hostPhaseReport = runtime.finishHostPhases();
assert.equal(hostPhaseReport.status, 'succeeded');
assert.deepEqual(
  hostPhaseReport.intervals.map(interval => interval.phase),
  ['cpu-preprocess'],
  'SHARP-owned CPU preprocessing must be an explicit route/run/clock-bound host phase',
);

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
