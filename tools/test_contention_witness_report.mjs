import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';

import {
  SHARP_CONTENTION_WITNESS_SCHEMA,
  createSharpContentionWitnessFailureReport,
  createSharpBackgroundHeartbeatReport,
  validateSharpContentionWitnessReport,
} from './contention_witness_report.mjs';
import {
  addStagedSubmitStage,
  classifyWebGpuRouteReceiptEvidence,
  createSharpImageToSplatRouteReceipt,
  createStagedSubmitProfile,
  createWebGpuBackendIdentity,
} from '@kaminos/webgpu-inference-kit';

const EPISODE_ID = 'sharp-contention:test-run';
const PLY_BYTE_LENGTH = 66060836;
const PLY_GAUSSIAN_COUNT = 1179648;
const PLY_SHA256 = 'a'.repeat(64);
const EVENT_SEQUENCE_ENVELOPE = {
  firstSequence: 0,
  lastSequence: 3,
  nextSequence: 4,
  eventCount: 4,
};

function sha256Json(value) {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

function createMetadataPayload(overrides = {}) {
  return {
    schema: 'sharp.webgpu-route-metadata.v0',
    episodeId: EPISODE_ID,
    terminalOutput: {
      plySha256: PLY_SHA256,
      plyByteLength: PLY_BYTE_LENGTH,
      numGaussians: PLY_GAUSSIAN_COUNT,
      completeness: 'complete',
    },
    schedulerTrace: {
      runId: EPISODE_ID,
      eventSequence: EVENT_SEQUENCE_ENVELOPE,
    },
    ...overrides,
  };
}

function createReceipt(overrides = {}) {
  const metadataPayload = overrides.metadataPayload || createMetadataPayload();
  const profile = createStagedSubmitProfile({
    route: 'sharp.image-to-splat.webgpu-local.v0',
    timingSource: 'adapter-phase-wall-clock',
    requiredStages: ['spn', 'monodepth', 'gaussian-decoder', 'compose-ply', 'output-capture'],
  });
  for (const [index, name] of profile.requiredStages.entries()) {
    addStagedSubmitStage(profile, { name, ms: index + 1 });
  }

  const receipt = createSharpImageToSplatRouteReceipt({
    input: {
      artifactId: 'source-image:test',
      sha256: 'sha256-source',
      shape: [768, 768, 4],
    },
    outputs: {
      splat: {
        artifactId: 'splat-candidate:test',
        sha256: PLY_SHA256,
        shape: [PLY_GAUSSIAN_COUNT, 14],
      },
      depthMap: {
        artifactId: 'depth-map:test',
        sha256: 'sha256-depth',
        shape: [768, 768, 1],
      },
      metadata: {
        artifactId: 'sharp-metadata:test',
        sha256: sha256Json(metadataPayload),
        shape: [1],
      },
    },
    backend: createWebGpuBackendIdentity({
      adapterName: 'test-webgpu-adapter',
      browser: 'node-contract-smoke',
      requestedFeatures: [],
      effectiveFeatures: ['timestamp-query'],
      limits: {
        maxBufferSize: 1024,
        maxStorageBufferBindingSize: 1024,
        maxComputeInvocationsPerWorkgroup: 256,
      },
      timestampQuery: 'available',
    }),
    model: {
      revision: 'local-sharp-webgpu',
      weightsHash: 'sha256-weights',
    },
    kernel: {
      kitVersion: '0.1.4',
      profile: 'spn-dinov2l16-monodepth-gaussian-ply',
      commit: 'sharp-webgpu-contention-witness-test',
    },
    profile,
    ...overrides,
  });

  return {
    ...receipt,
    metadataPayload,
    ...(overrides.receiptFields || {}),
  };
}

const routeReceipt = createReceipt();

const baseReport = {
  schema: SHARP_CONTENTION_WITNESS_SCHEMA,
  runId: EPISODE_ID,
  createdAt: '2026-07-02T00:00:00.000Z',
  route: {
    requestedRouteId: 'sharp.image-to-splat.webgpu-local.v0',
    effectiveRouteId: 'sharp.image-to-splat.webgpu-local.v0',
    receipt: routeReceipt,
    evidence: classifyWebGpuRouteReceiptEvidence(routeReceipt, {
      expectedRouteId: 'sharp.image-to-splat.webgpu-local.v0',
    }),
  },
  mode: 'contention',
  input: {
    source: 'sample',
    artifactId: 'public/samples/sample_1.jpg',
  },
  inference: {
    ok: true,
    valid: 'OK',
    timeMs: 28750,
    model: 'DINOv2 ViT-Large (dinov2l16_384)',
    weights: '1280 MB (fp16)',
    outputs: {
      numGaussians: PLY_GAUSSIAN_COUNT,
      plyAvailable: true,
      plyByteLength: PLY_BYTE_LENGTH,
      plySha256: PLY_SHA256,
      completeness: 'complete',
    },
  },
  responsiveness: {
    rafFrames: 3,
    maxFrameGapMs: 380,
    p95FrameGapMs: 380,
    longFrameCount: 2,
  },
  backgroundHeartbeat: {
    schema: 'sharp-webgpu.background-heartbeat.v0',
    timingAuthority: 'browser-raf-and-scheduler-event-trace',
    schedulerMode: 'background',
    verificationState: 'verified',
    requestedScheduler: {
      mode: 'background',
      waitForSubmittedWorkDone: true,
      yieldMs: 16,
    },
    effectiveScheduler: {
      mode: 'background',
      waitForSubmittedWorkDone: true,
      yieldMs: 16,
      vitBlockChunkSize: 1,
      spnPatchChunkSize: 1,
    },
    responsiveness: {
      rafFrames: 3,
      maxFrameGapMs: 380,
      p95FrameGapMs: 380,
      longFrameCount: 2,
    },
    eventTrace: {
      schema: 'kaminos.webgpu-scheduler-event-trace.v0',
      timingAuthority: 'browser-wall-clock',
      sequenceEnvelope: {
        firstSequence: 0,
        lastSequence: 1,
        nextSequence: 2,
        eventCount: 2,
      },
      clock: {
        schema: 'kaminos.browser-epoch-monotonic-clock.v0',
        relativeClock: 'performance.now',
        epochClock: 'performance.timeOrigin+performance.now',
        timeOriginEpochMs: 1770000000000,
      },
      uncapped: true,
      eventCount: 4,
      sequenceEnvelope: EVENT_SEQUENCE_ENVELOPE,
      boundaries: ['spn-patch-chunk', 'vit-block-chunk'],
      events: [
        { sequence: 0, runId: EPISODE_ID, phase: 'spn-patch-chunk', boundary: 'spn-patch-chunk', kind: 'queue-work-done-start', dutyId: `${EPISODE_ID}:spn-patch-chunk:0`, tMs: 1010, epochMs: 1770000001010 },
        { sequence: 1, runId: EPISODE_ID, phase: 'spn-patch-chunk', boundary: 'spn-patch-chunk', kind: 'queue-work-done-end', dutyId: `${EPISODE_ID}:spn-patch-chunk:0`, tMs: 1200, epochMs: 1770000001200 },
        {
          sequence: 2,
          runId: EPISODE_ID,
          phase: 'vit-block-chunk',
          boundary: 'vit-block-chunk',
          kind: 'boundary-start',
          tMs: 1300,
          epochMs: 1770000001300,
          intervalStartMs: 1250,
          intervalEndMs: 1300,
          durationMs: 50,
          intervalStartEpochMs: 1770000001250,
          intervalEndEpochMs: 1770000001300,
        },
        { sequence: 3, runId: EPISODE_ID, phase: 'vit-block-chunk', boundary: 'vit-block-chunk', kind: 'boundary-end', tMs: 1400, epochMs: 1770000001400 },
      ],
    },
    crossPageClock: {
      schema: 'kaminos.browser-epoch-monotonic-clock.v0',
      timingAuthority: 'performance-time-origin-plus-now',
      runId: EPISODE_ID,
      timeOriginEpochMs: 1770000000000,
      inferenceWindowStartEpochMs: 1770000000900,
      inferenceWindowEndEpochMs: 1770000001500,
    },
    gpuDutyIntervals: {
      schema: 'sharp-webgpu.submitted-work-drain-intervals.v0',
      timingAuthority: 'queue-on-submitted-work-done-host-await-not-gpu-exclusive',
      runId: EPISODE_ID,
      count: 1,
      intervals: [{
        runId: EPISODE_ID,
        dutyId: `${EPISODE_ID}:spn-patch-chunk:0`,
        startEventSequence: 0,
        endEventSequence: 1,
        phase: 'spn-patch-chunk',
        boundary: 'spn-patch-chunk',
        kind: 'submitted-work-drain-interval',
        startMs: 1010,
        endMs: 1200,
        durationMs: 190,
        startEpochMs: 1770000001010,
        endEpochMs: 1770000001200,
      }],
    },
    inferenceWindow: {
      runId: EPISODE_ID,
      startMs: 900,
      endMs: 1500,
      durationMs: 600,
    },
    rafIntervalTrace: {
      schema: 'sharp-webgpu.raf-interval-trace.v0',
      timingAuthority: 'browser-request-animation-frame-performance-now',
      runId: EPISODE_ID,
      uncapped: true,
      count: 3,
      timeOriginEpochMs: 1770000000000,
      intervals: [
        { startMs: 900, endMs: 920, durationMs: 20, startEpochMs: 1770000000900, endEpochMs: 1770000000920 },
        { startMs: 920, endMs: 1120, durationMs: 200, startEpochMs: 1770000000920, endEpochMs: 1770000001120 },
        { startMs: 1120, endMs: 1500, durationMs: 380, startEpochMs: 1770000001120, endEpochMs: 1770000001500 },
      ],
    },
    worstFrameGaps: [
      {
        startMs: 1120,
        endMs: 1500,
        durationMs: 380,
        overlapClassification: 'scheduler-event-overlap',
        overlappedEvents: [
          {
            phase: 'spn-patch-chunk',
            boundary: 'spn-patch-chunk',
            kind: 'queue-work-done-start',
            tMs: 1200,
          },
        ],
      },
    ],
  },
  contender: {
    enabled: true,
    submitted: 52,
    completed: 49,
    inferenceWindow: {
      submittedAtStart: 10,
      completedAtStart: 9,
      submittedAtEnd: 52,
      completedAtEnd: 49,
      submittedDelta: 42,
      completedDelta: 40,
    },
    progressDuringInference: true,
    errors: [],
  },
  scheduler: {
    runId: EPISODE_ID,
    mode: 'throughput',
    verificationState: 'scheduler-unverified',
  },
};

assert.deepEqual(validateSharpContentionWitnessReport(baseReport), {
  ok: true,
  errors: [],
  warnings: [],
});

const reportModule = await import('./contention_witness_report.mjs');
assert.equal(
  typeof reportModule.renderSharpContentionTraceSvg,
  'function',
  'the evidence spine must expose a source-validated compact trace renderer',
);
const traceSvg = reportModule.renderSharpContentionTraceSvg(baseReport);
assert.match(traceSvg, /<svg\b/);
assert.match(traceSvg, new RegExp(EPISODE_ID));
assert.match(traceSvg, /splat-candidate/);
assert.equal(
  (traceSvg.match(/data-raf-interval=/g) || []).length,
  baseReport.backgroundHeartbeat.rafIntervalTrace.count,
  'the compact trace must project every uncapped rAF interval',
);
assert.equal(
  (traceSvg.match(/data-scheduler-boundary=/g) || []).length,
  baseReport.backgroundHeartbeat.eventTrace.boundaries.length,
  'the compact trace must aggregate scheduler events by exact semantic boundary',
);
assert.doesNotMatch(traceSvg, /data-scheduler-event=/, 'the compact projection must not allocate one SVG node per scheduler event');

const scaleReport = structuredClone(baseReport);
const scaleEventCount = 10000;
scaleReport.backgroundHeartbeat.eventTrace.events = Array.from({ length: scaleEventCount }, (_, sequence) => {
  if (sequence < 2) return structuredClone(baseReport.backgroundHeartbeat.eventTrace.events[sequence]);
  const tMs = 1200 + (sequence / scaleEventCount) * 200;
  return {
    sequence,
    runId: EPISODE_ID,
    phase: sequence % 2 ? 'spn-patch-chunk' : 'vit-block-chunk',
    boundary: sequence % 2 ? 'spn-patch-chunk' : 'vit-block-chunk',
    kind: 'boundary-sample',
    tMs,
    epochMs: 1770000000000 + tMs,
  };
});
scaleReport.backgroundHeartbeat.eventTrace.eventCount = scaleEventCount;
scaleReport.backgroundHeartbeat.eventTrace.sequenceEnvelope = {
  firstSequence: 0,
  lastSequence: scaleEventCount - 1,
  nextSequence: scaleEventCount,
  eventCount: scaleEventCount,
};
scaleReport.route.receipt.metadataPayload.schedulerTrace.eventSequence = structuredClone(
  scaleReport.backgroundHeartbeat.eventTrace.sequenceEnvelope,
);
scaleReport.route.receipt.outputs.find(output => output.role === 'sharp-webgpu-metadata').sha256 = sha256Json(
  scaleReport.route.receipt.metadataPayload,
);
const scaleSvg = reportModule.renderSharpContentionTraceSvg(scaleReport);
assert.equal(
  (scaleSvg.match(/data-scheduler-boundary=/g) || []).length,
  2,
  'ten thousand uncapped scheduler events must remain two exact boundary aggregates in the SVG',
);
assert.ok(scaleSvg.length < 100000, 'scheduler projection size must depend on boundary cardinality, not raw event count');
const invalidTraceReport = structuredClone(baseReport);
delete invalidTraceReport.backgroundHeartbeat.rafIntervalTrace;
assert.throws(
  () => reportModule.renderSharpContentionTraceSvg(invalidTraceReport),
  /cannot render invalid SHARP contention report.*rafIntervalTrace/,
);
if (process.env.SHARP_TRACE_FIXTURE_OUT) {
  fs.writeFileSync(process.env.SHARP_TRACE_FIXTURE_OUT, `${traceSvg}\n`);
}

const constructedHeartbeat = createSharpBackgroundHeartbeatReport({
  scheduler: {
    runId: 'proof-run',
    status: 'verified',
    requestedScheduler: baseReport.backgroundHeartbeat.requestedScheduler,
    effectiveScheduler: baseReport.backgroundHeartbeat.effectiveScheduler,
    eventTrace: {
      schema: 'kaminos.webgpu-scheduler-event-trace.v0',
      timingAuthority: 'browser-wall-clock',
      events: [
        { phase: 'spn-patch-chunk', boundary: 'spn-patch-chunk', kind: 'chunk-start', tMs: 1008 },
        { runId: 'proof-run', phase: 'vit-block-chunk', boundary: 'vit-block-chunk', kind: 'queue-work-done-start', tMs: 1120 },
      ],
    },
  },
  probe: {
    ...baseReport.responsiveness,
    inferenceWindow: {
      runId: 'proof-run',
      startMs: 900,
      endMs: 1450,
    },
    worstFrameGaps: [
      { startMs: 100, endMs: 800, durationMs: 700 },
      { startMs: 1000, endMs: 1400, durationMs: 400 },
      { startMs: 1500, endMs: 1520, durationMs: 20 },
    ],
  },
});
assert.equal(constructedHeartbeat.schema, 'sharp-webgpu.background-heartbeat.v0');
assert.equal(constructedHeartbeat.eventTrace.eventCount, 2);
assert.deepEqual(constructedHeartbeat.eventTrace.boundaries, ['spn-patch-chunk', 'vit-block-chunk']);
assert.deepEqual(constructedHeartbeat.inferenceWindow, {
  runId: 'proof-run',
  startMs: 900,
  endMs: 1450,
  durationMs: 550,
});
assert.equal(constructedHeartbeat.worstFrameGaps.length, 1, 'heartbeat must exclude gaps outside the measured inference window');
assert.equal(constructedHeartbeat.worstFrameGaps[0].durationMs, 400);
assert.equal(constructedHeartbeat.worstFrameGaps[0].overlapClassification, 'scheduler-event-overlap');
assert.equal(constructedHeartbeat.worstFrameGaps[0].overlappedEvents.length, 2);

const crossPageHeartbeat = createSharpBackgroundHeartbeatReport({
  scheduler: {
    runId: 'proof-run',
    status: 'verified',
    requestedScheduler: baseReport.backgroundHeartbeat.requestedScheduler,
    effectiveScheduler: baseReport.backgroundHeartbeat.effectiveScheduler,
    eventTrace: {
      schema: 'kaminos.webgpu-scheduler-event-trace.v0',
      timingAuthority: 'browser-wall-clock',
      clock: {
        schema: 'kaminos.browser-epoch-monotonic-clock.v0',
        relativeClock: 'performance.now',
        epochClock: 'performance.timeOrigin+performance.now',
        timeOriginEpochMs: 1770000000000,
      },
      events: [
        {
          sequence: 0,
          runId: 'proof-run',
          phase: 'spn-patch-chunk',
          boundary: 'spn-patch-chunk',
          kind: 'queue-work-done-start',
          dutyId: 'proof:spn-patch-chunk:0',
          tMs: 1010,
          epochMs: 1770000001010,
        },
        {
          sequence: 1,
          runId: 'proof-run',
          phase: 'spn-patch-chunk',
          boundary: 'spn-patch-chunk',
          kind: 'queue-work-done-end',
          dutyId: 'proof:spn-patch-chunk:0',
          tMs: 1200,
          epochMs: 1770000001200,
          queueDoneMs: 190,
        },
      ],
    },
  },
  probe: {
    inferenceWindow: {
      runId: 'proof-run',
      startMs: 900,
      endMs: 1500,
      startEpochMs: 1770000000900,
      endEpochMs: 1770000001500,
    },
    rafFrames: 2,
    maxFrameGapMs: 200,
    p95FrameGapMs: 200,
    longFrameCount: 1,
    worstFrameGaps: [{ startMs: 1050, endMs: 1250, durationMs: 200 }],
  },
});
assert.deepEqual(crossPageHeartbeat.crossPageClock, {
  schema: 'kaminos.browser-epoch-monotonic-clock.v0',
  timingAuthority: 'performance-time-origin-plus-now',
  runId: 'proof-run',
  timeOriginEpochMs: 1770000000000,
  inferenceWindowStartEpochMs: 1770000000900,
  inferenceWindowEndEpochMs: 1770000001500,
});
assert.deepEqual(crossPageHeartbeat.gpuDutyIntervals, {
  schema: 'sharp-webgpu.submitted-work-drain-intervals.v0',
  timingAuthority: 'queue-on-submitted-work-done-host-await-not-gpu-exclusive',
  runId: 'proof-run',
  count: 1,
  intervals: [{
    runId: 'proof-run',
    dutyId: 'proof:spn-patch-chunk:0',
    phase: 'spn-patch-chunk',
    boundary: 'spn-patch-chunk',
    kind: 'submitted-work-drain-interval',
    startEventSequence: 0,
    endEventSequence: 1,
    startMs: 1010,
    endMs: 1200,
    durationMs: 190,
    startEpochMs: 1770000001010,
    endEpochMs: 1770000001200,
    stage: null,
    step: null,
    role: null,
  }],
});

const staleRunDutyHeartbeat = createSharpBackgroundHeartbeatReport({
  scheduler: {
    runId: 'current-run',
    status: 'verified',
    requestedScheduler: baseReport.backgroundHeartbeat.requestedScheduler,
    effectiveScheduler: baseReport.backgroundHeartbeat.effectiveScheduler,
    eventTrace: {
      schema: 'kaminos.webgpu-scheduler-event-trace.v0',
      timingAuthority: 'browser-wall-clock',
      clock: {
        schema: 'kaminos.browser-epoch-monotonic-clock.v0',
        relativeClock: 'performance.now',
        epochClock: 'performance.timeOrigin+performance.now',
        timeOriginEpochMs: 1770000000000,
      },
      events: [
        {
          runId: 'stale-run',
          phase: 'spn-fusion', boundary: 'spn-fusion', kind: 'queue-work-done-start',
          dutyId: 'stale-run:spn-fusion:0', tMs: 1000, epochMs: 1770000001000,
        },
        {
          runId: 'stale-run',
          phase: 'spn-fusion', boundary: 'spn-fusion', kind: 'queue-work-done-end',
          dutyId: 'stale-run:spn-fusion:0', tMs: 1100, epochMs: 1770000001100,
        },
      ],
    },
  },
  probe: {
    inferenceWindow: {
      runId: 'current-run',
      startMs: 900,
      endMs: 1200,
      startEpochMs: 1770000000900,
      endEpochMs: 1770000001200,
    },
    worstFrameGaps: [{ startMs: 1000, endMs: 1100, durationMs: 100 }],
  },
});
assert.match(
  staleRunDutyHeartbeat.gpuDutyIntervals.pairingFailures.join('\n'),
  /runId|stale-run|current-run/,
  'stale queue-drain endpoints must fail rather than inherit the current run envelope',
);
assert.equal(
  staleRunDutyHeartbeat.gpuDutyIntervals.intervals.length,
  0,
  'stale queue-drain endpoints must not become current-run intervals',
);

const mismatchedDutyHeartbeat = createSharpBackgroundHeartbeatReport({
  scheduler: {
    runId: 'mismatched-duty-run',
    requestedScheduler: baseReport.backgroundHeartbeat.requestedScheduler,
    effectiveScheduler: baseReport.backgroundHeartbeat.effectiveScheduler,
    eventTrace: {
      clock: {
        schema: 'kaminos.browser-epoch-monotonic-clock.v0',
        timeOriginEpochMs: 1770000000000,
      },
      events: [
        {
          runId: 'mismatched-duty-run',
          phase: 'spn-fusion', boundary: 'spn-fusion', kind: 'queue-work-done-start',
          dutyId: 'mismatched-duty-run:0', tMs: 1000, epochMs: 1770000001000,
        },
        {
          runId: 'mismatched-duty-run',
          phase: 'monodepth-phase', boundary: 'monodepth-phase', kind: 'queue-work-done-end',
          dutyId: 'mismatched-duty-run:0', tMs: 1100, epochMs: 1770000001100,
        },
      ],
    },
  },
  probe: {
    inferenceWindow: {
      runId: 'mismatched-duty-run',
      startMs: 900,
      endMs: 1200,
      startEpochMs: 1770000000900,
      endEpochMs: 1770000001200,
    },
    worstFrameGaps: [{ startMs: 1000, endMs: 1100, durationMs: 100 }],
  },
});
assert.match(
  mismatchedDutyHeartbeat.gpuDutyIntervals.pairingFailures.join('\n'),
  /endpoints disagree/,
  'cross-phase queue-drain endpoints must fail rather than inherit the start label',
);

const intervalHeartbeat = createSharpBackgroundHeartbeatReport({
  scheduler: {
    status: 'verified',
    requestedScheduler: baseReport.backgroundHeartbeat.requestedScheduler,
    effectiveScheduler: baseReport.backgroundHeartbeat.effectiveScheduler,
    eventTrace: {
      schema: 'kaminos.webgpu-scheduler-event-trace.v0',
      timingAuthority: 'browser-wall-clock',
      events: [{
        phase: 'route-tail',
        boundary: 'route-tail',
        kind: 'duty-interval',
        tMs: 1400,
        intervalStartMs: 1000,
        intervalEndMs: 1400,
        durationMs: 400,
        stage: 'compose-ply',
        step: 'ply-blob-assembly',
      }],
    },
  },
  probe: {
    inferenceWindow: { startMs: 900, endMs: 1500 },
    rafFrames: 2,
    maxFrameGapMs: 200,
    p95FrameGapMs: 200,
    longFrameCount: 1,
    worstFrameGaps: [{ startMs: 1100, endMs: 1300, durationMs: 200 }],
  },
});
assert.equal(
  intervalHeartbeat.worstFrameGaps[0].overlapClassification,
  'scheduler-event-overlap',
  'a blocking duty interval must overlap the RAF gap it spans even when neither endpoint falls inside the gap',
);
assert.deepEqual(
  intervalHeartbeat.worstFrameGaps[0].overlappedEvents[0],
  {
    phase: 'route-tail',
    boundary: 'route-tail',
    kind: 'duty-interval',
    tMs: 1400,
    intervalStartMs: 1000,
    intervalEndMs: 1400,
    durationMs: 400,
    stage: 'compose-ply',
    step: 'ply-blob-assembly',
    role: null,
  },
  'heartbeat evidence must preserve the named interval that explains the blocked frame',
);

const intervalOverlapReport = structuredClone(baseReport);
intervalOverlapReport.backgroundHeartbeat.worstFrameGaps[0].overlappedEvents = [
  intervalHeartbeat.worstFrameGaps[0].overlappedEvents[0],
];
assert.equal(
  validateSharpContentionWitnessReport(intervalOverlapReport).ok,
  true,
  'validated heartbeat receipts must accept a truthful spanning interval whose end timestamp falls outside the RAF gap',
);

const durationOnlyHeartbeat = createSharpBackgroundHeartbeatReport({
  scheduler: {
    status: 'verified',
    requestedScheduler: baseReport.backgroundHeartbeat.requestedScheduler,
    effectiveScheduler: baseReport.backgroundHeartbeat.effectiveScheduler,
    eventTrace: {
      schema: 'kaminos.webgpu-scheduler-event-trace.v0',
      timingAuthority: 'browser-wall-clock',
      events: [{ phase: 'spn-patch-chunk', boundary: 'spn-patch-chunk', kind: 'chunk-start', tMs: 0 }],
    },
  },
  probe: {
    ...baseReport.responsiveness,
    inferenceWindow: {
      startMs: 0,
      endMs: 500,
    },
    worstFrameGaps: [{ durationMs: 400 }],
  },
});
assert.equal(durationOnlyHeartbeat.worstFrameGaps.length, 0, 'duration-only gaps must not be normalized into fake intervals');

const fractionalBoundaryHeartbeat = createSharpBackgroundHeartbeatReport({
  scheduler: {
    status: 'verified',
    requestedScheduler: baseReport.backgroundHeartbeat.requestedScheduler,
    effectiveScheduler: baseReport.backgroundHeartbeat.effectiveScheduler,
    eventTrace: {
      schema: 'kaminos.webgpu-scheduler-event-trace.v0',
      timingAuthority: 'browser-wall-clock',
      events: [{ phase: 'spn', boundary: 'spn', kind: 'chunk-start', tMs: 901 }],
    },
  },
  probe: {
    inferenceWindow: { startMs: 900.0006, endMs: 1000.0004 },
    rafFrames: 2,
    maxFrameGapMs: 49.9994,
    p95FrameGapMs: 49.9994,
    longFrameCount: 0,
    frameGapIntervals: [
      { startMs: 900.0006, endMs: 950, durationMs: 49.9994 },
      { startMs: 975, endMs: 1000.0004, durationMs: 25.0004 },
    ],
    worstFrameGaps: [
      { startMs: 900.0006, endMs: 950, durationMs: 49.9994 },
      { startMs: 975, endMs: 1000.0004, durationMs: 25.0004 },
    ],
  },
});
assert.equal(fractionalBoundaryHeartbeat.worstFrameGaps.length, 2, 'rounding must not drop raw boundary-clipped gaps');
assert.equal(fractionalBoundaryHeartbeat.rafIntervalTrace.count, 2, 'rounding must not drop uncapped raw rAF intervals');
assert.equal(fractionalBoundaryHeartbeat.worstFrameGaps[0].startMs, fractionalBoundaryHeartbeat.inferenceWindow.startMs);
assert.equal(fractionalBoundaryHeartbeat.worstFrameGaps[1].endMs, fractionalBoundaryHeartbeat.inferenceWindow.endMs);

const appFailure = createSharpContentionWitnessFailureReport({
  candidateReport: {
    ...structuredClone(baseReport),
    inference: { ...baseReport.inference, ok: false, error: 'device lost during monodepth' },
  },
  failurePhase: 'app-inference',
  error: 'device lost during monodepth',
  validation: { ok: false, errors: ['inference failed'], warnings: [] },
});
assert.equal(appFailure.schema, 'sharp.webgpu-contention-witness-failure.v0');
assert.equal(appFailure.failurePhase, 'app-inference');
assert.equal(appFailure.lastTrustworthyEvidence.inference.ok, false);
assert.equal(
  appFailure.lastTrustworthyEvidence.backgroundHeartbeat.rafIntervalTrace.count,
  3,
  'failure artifacts must preserve the uncapped rAF and shared-clock heartbeat prefix',
);
assert.equal('backgroundHeartbeat' in appFailure, false, 'failure artifacts must not expose a normal heartbeat receipt');

function addSecondAuthenticatedDuty(report, { includeDerivedInterval = true } = {}) {
  const startEvent = {
    sequence: 4,
    runId: EPISODE_ID,
    phase: 'monodepth-phase',
    boundary: 'monodepth-phase',
    kind: 'queue-work-done-start',
    dutyId: `${EPISODE_ID}:monodepth-phase:1`,
    tMs: 1410,
    epochMs: 1770000001410,
    stage: 'monodepth',
    step: 1,
    role: 'decoder',
  };
  const endEvent = {
    ...startEvent,
    sequence: 5,
    kind: 'queue-work-done-end',
    tMs: 1450,
    epochMs: 1770000001450,
  };
  report.backgroundHeartbeat.eventTrace.events.push(startEvent, endEvent);
  report.backgroundHeartbeat.eventTrace.eventCount = 6;
  report.backgroundHeartbeat.eventTrace.sequenceEnvelope = {
    firstSequence: 0,
    lastSequence: 5,
    nextSequence: 6,
    eventCount: 6,
  };
  report.backgroundHeartbeat.eventTrace.boundaries = [
    ...report.backgroundHeartbeat.eventTrace.boundaries,
    'monodepth-phase',
  ].sort();
  report.route.receipt.metadataPayload.schedulerTrace.eventSequence = structuredClone(
    report.backgroundHeartbeat.eventTrace.sequenceEnvelope,
  );
  report.route.receipt.outputs.find(output => output.role === 'sharp-webgpu-metadata').sha256 = sha256Json(
    report.route.receipt.metadataPayload,
  );
  if (includeDerivedInterval) {
    report.backgroundHeartbeat.gpuDutyIntervals.intervals.push({
      runId: EPISODE_ID,
      dutyId: startEvent.dutyId,
      phase: startEvent.phase,
      boundary: startEvent.boundary,
      kind: 'submitted-work-drain-interval',
      startEventSequence: startEvent.sequence,
      endEventSequence: endEvent.sequence,
      startMs: startEvent.tMs,
      endMs: endEvent.tMs,
      durationMs: 40,
      startEpochMs: startEvent.epochMs,
      endEpochMs: endEvent.epochMs,
      stage: startEvent.stage,
      step: startEvent.step,
      role: startEvent.role,
    });
    report.backgroundHeartbeat.gpuDutyIntervals.count = 2;
  }
}

const completeTwoDutyReport = structuredClone(baseReport);
addSecondAuthenticatedDuty(completeTwoDutyReport);
assert.deepEqual(
  validateSharpContentionWitnessReport(completeTwoDutyReport),
  { ok: true, errors: [], warnings: [] },
  'a complete authenticated two-duty report must remain valid',
);

for (const [name, mutate, pattern] of [
  [
    'missing route identity',
    report => { delete report.route; },
    /route/,
  ],
  [
    'fallback route',
    report => { report.route.effectiveRouteId = 'fallback.cpu.fixture.v0'; },
    /effectiveRouteId/,
  ],
  [
    'fallback route receipt preserving route strings',
    report => {
      report.route.receipt = createReceipt({
        status: 'fallback',
        fallbackReason: 'fixture-smoke',
      });
      report.route.evidence = classifyWebGpuRouteReceiptEvidence(report.route.receipt, {
        expectedRouteId: 'sharp.image-to-splat.webgpu-local.v0',
      });
    },
    /authoritative|fallback/,
  ],
  [
    'missing latency',
    report => { report.inference.timeMs = null; },
    /timeMs/,
  ],
  [
    'terminal output without exact byte identity',
    report => { delete report.inference.outputs.plyByteLength; },
    /plyByteLength/,
  ],
  [
    'terminal output byte length spliced after receipt publication',
    report => { report.inference.outputs.plyByteLength += 1024; },
    /plyByteLength|metadata|terminal output/i,
  ],
  [
    'terminal output without exact digest identity',
    report => { delete report.inference.outputs.plySha256; },
    /plySha256/,
  ],
  [
    'terminal output digest contradicts route receipt',
    report => { report.inference.outputs.plySha256 = 'b'.repeat(64); },
    /plySha256|route receipt/,
  ],
  [
    'witness episode id spliced across scheduler and route receipt',
    report => { report.runId = 'sharp-contention:other-episode'; },
    /runId|episode/i,
  ],
  [
    'route metadata episode id changed without rehashing',
    report => { report.route.receipt.metadataPayload.episodeId = 'sharp-contention:other-episode'; },
    /metadata|sha256|episode/i,
  ],
  [
    'route metadata payload hash changed independently',
    report => {
      const metadata = report.route.receipt.outputs.find(output => output.role === 'sharp-webgpu-metadata');
      metadata.sha256 = 'b'.repeat(64);
    },
    /metadata|sha256/i,
  ],
  [
    'hidden invalid output',
    report => { report.inference.valid = 'INVALID'; },
    /valid/,
  ],
  [
    'contended run with no contender progress',
    report => {
      report.contender.completed = 0;
      report.contender.inferenceWindow.completedAtEnd = report.contender.inferenceWindow.completedAtStart;
      report.contender.inferenceWindow.completedDelta = 0;
      report.contender.progressDuringInference = false;
    },
    /progress/,
  ],
  [
    'contender progress outside inference window',
    report => {
      report.contender.submitted = 52;
      report.contender.completed = 49;
      report.contender.inferenceWindow.submittedAtStart = 52;
      report.contender.inferenceWindow.completedAtStart = 49;
      report.contender.inferenceWindow.submittedAtEnd = 52;
      report.contender.inferenceWindow.completedAtEnd = 49;
      report.contender.inferenceWindow.submittedDelta = 0;
      report.contender.inferenceWindow.completedDelta = 0;
      report.contender.progressDuringInference = false;
    },
    /inference window|progress/,
  ],
  [
    'missing frame-tail evidence',
    report => {
      report.responsiveness.rafFrames = 0;
      report.responsiveness.maxFrameGapMs = 0;
      report.responsiveness.p95FrameGapMs = 0;
      report.responsiveness.longFrameCount = 0;
    },
    /rafFrames|frame-tail/,
  ],
  [
    'missing background heartbeat receipt',
    report => { delete report.backgroundHeartbeat; },
    /backgroundHeartbeat/,
  ],
  [
    'heartbeat without uncapped raw rAF intervals',
    report => { delete report.backgroundHeartbeat.rafIntervalTrace; },
    /rafIntervalTrace|uncapped/,
  ],
  [
    'heartbeat with clipped rAF interval retention',
    report => {
      report.backgroundHeartbeat.rafIntervalTrace.intervals.pop();
      report.backgroundHeartbeat.rafIntervalTrace.count -= 1;
    },
    /rafIntervalTrace|rafFrames|uncapped/,
  ],
  [
    'heartbeat with consistently repaired rAF prefix truncation',
    report => {
      report.backgroundHeartbeat.rafIntervalTrace.intervals.shift();
      report.backgroundHeartbeat.rafIntervalTrace.count = 2;
      report.backgroundHeartbeat.responsiveness.rafFrames = 2;
      report.responsiveness.rafFrames = 2;
    },
    /rafIntervalTrace|inferenceWindow|first/i,
  ],
  [
    'heartbeat with consistently repaired rAF suffix truncation',
    report => {
      report.backgroundHeartbeat.rafIntervalTrace.intervals.pop();
      report.backgroundHeartbeat.rafIntervalTrace.count = 2;
      Object.assign(report.backgroundHeartbeat.responsiveness, {
        rafFrames: 2,
        maxFrameGapMs: 200,
        p95FrameGapMs: 200,
        longFrameCount: 1,
      });
      Object.assign(report.responsiveness, report.backgroundHeartbeat.responsiveness);
      report.backgroundHeartbeat.worstFrameGaps = [{
        startMs: 920,
        endMs: 1120,
        durationMs: 200,
        overlapClassification: 'scheduler-event-overlap',
        overlappedEvents: [{
          phase: 'spn-patch-chunk',
          boundary: 'spn-patch-chunk',
          kind: 'queue-work-done-start',
          tMs: 1010,
        }],
      }];
    },
    /rafIntervalTrace|inferenceWindow|last/i,
  ],
  [
    'heartbeat without worst-gap intervals',
    report => { report.backgroundHeartbeat.worstFrameGaps = []; },
    /worstFrameGaps/,
  ],
  [
    'heartbeat without scheduler event trace',
    report => { report.backgroundHeartbeat.eventTrace.eventCount = 0; },
    /eventTrace/,
  ],
  [
    'heartbeat with clipped scheduler event retention',
    report => { report.backgroundHeartbeat.eventTrace.events.pop(); },
    /eventTrace|uncapped|eventCount/,
  ],
  [
    'heartbeat with consistently repaired scheduler event deletion',
    report => {
      report.backgroundHeartbeat.eventTrace.events.splice(2, 1);
      report.backgroundHeartbeat.eventTrace.eventCount = 3;
      report.backgroundHeartbeat.eventTrace.boundaries = ['spn-patch-chunk', 'vit-block-chunk'];
    },
    /eventTrace|sequence|metadata/i,
  ],
  [
    'scheduler interval without epoch endpoints',
    report => {
      delete report.backgroundHeartbeat.eventTrace.events[2].intervalStartEpochMs;
      delete report.backgroundHeartbeat.eventTrace.events[2].intervalEndEpochMs;
    },
    /intervalStartEpochMs|intervalEndEpochMs|epoch/i,
  ],
  [
    'scheduler interval with mixed epoch origin',
    report => { report.backgroundHeartbeat.eventTrace.events[2].intervalStartEpochMs += 5000; },
    /intervalStartEpochMs|timeOrigin|epoch/i,
  ],
  [
    'scheduler interval with reversed epoch bounds',
    report => {
      const event = report.backgroundHeartbeat.eventTrace.events[2];
      event.intervalStartEpochMs = 1770000001350;
      event.intervalEndEpochMs = 1770000001300;
    },
    /intervalStartEpochMs|intervalEndEpochMs|epoch/i,
  ],
  [
    'heartbeat without inference window',
    report => { delete report.backgroundHeartbeat.inferenceWindow; },
    /inferenceWindow/,
  ],
  [
    'heartbeat without cross-page clock',
    report => { delete report.backgroundHeartbeat.crossPageClock; },
    /crossPageClock/,
  ],
  [
    'heartbeat without submitted-work drain intervals',
    report => { delete report.backgroundHeartbeat.gpuDutyIntervals; },
    /gpuDutyIntervals/,
  ],
  [
    'heartbeat clock and duty intervals from different scheduler runs',
    report => { report.backgroundHeartbeat.gpuDutyIntervals.runId = 'stale-run'; },
    /runId|gpuDutyIntervals|crossPageClock/,
  ],
  [
    'heartbeat interval from a stale scheduler run',
    report => { report.backgroundHeartbeat.gpuDutyIntervals.intervals[0].runId = 'stale-run'; },
    /runId|gpuDutyIntervals|crossPageClock/,
  ],
  [
    'heartbeat duty interval count mismatch',
    report => { report.backgroundHeartbeat.gpuDutyIntervals.count = 2; },
    /gpuDutyIntervals.*count/,
  ],
  [
    'heartbeat duty interval with mixed clock origin',
    report => { report.backgroundHeartbeat.gpuDutyIntervals.intervals[0].startEpochMs += 5000; },
    /timeOrigin|startEpochMs|clock/,
  ],
  [
    'heartbeat duty interval with reversed epoch bounds',
    report => {
      report.backgroundHeartbeat.gpuDutyIntervals.intervals[0].startEpochMs = 1770000001300;
      report.backgroundHeartbeat.gpuDutyIntervals.intervals[0].endEpochMs = 1770000001200;
    },
    /startEpochMs|endEpochMs|durationMs/,
  ],
  [
    'heartbeat duty interval outside inference epoch window',
    report => {
      const interval = report.backgroundHeartbeat.gpuDutyIntervals.intervals[0];
      interval.startMs = 100;
      interval.endMs = 200;
      interval.durationMs = 100;
      interval.startEpochMs = 1770000000100;
      interval.endEpochMs = 1770000000200;
    },
    /inferenceWindow|gpuDutyIntervals/,
  ],
  [
    'heartbeat derived duty timing shifted away from retained endpoint events',
    report => {
      const interval = report.backgroundHeartbeat.gpuDutyIntervals.intervals[0];
      interval.startMs += 50;
      interval.endMs += 50;
      interval.startEpochMs += 50;
      interval.endEpochMs += 50;
    },
    /gpuDutyIntervals|source|startMs|endpoint/i,
  ],
  [
    'heartbeat derived duty phase and boundary disagree with retained endpoint events',
    report => {
      const interval = report.backgroundHeartbeat.gpuDutyIntervals.intervals[0];
      interval.phase = 'monodepth-phase';
      interval.boundary = 'monodepth-phase';
    },
    /gpuDutyIntervals|source|phase|boundary|endpoint/i,
  ],
  [
    'heartbeat omits one authenticated complete duty pair with repaired derived count',
    report => { addSecondAuthenticatedDuty(report, { includeDerivedInterval: false }); },
    /gpuDutyIntervals|missing|source|endpoint|bijection/i,
  ],
  [
    'heartbeat duplicates one derived duty interval for one retained endpoint pair',
    report => {
      report.backgroundHeartbeat.gpuDutyIntervals.intervals.push(structuredClone(
        report.backgroundHeartbeat.gpuDutyIntervals.intervals[0],
      ));
      report.backgroundHeartbeat.gpuDutyIntervals.count = 2;
    },
    /gpuDutyIntervals|unique|duplicate|bijection/i,
  ],
  [
    'heartbeat gap before inference window',
    report => {
      report.backgroundHeartbeat.worstFrameGaps[0].startMs = 100;
      report.backgroundHeartbeat.worstFrameGaps[0].endMs = 480;
      report.backgroundHeartbeat.worstFrameGaps[0].overlappedEvents[0].tMs = 120;
    },
    /inferenceWindow|worstFrameGaps/,
  ],
  [
    'heartbeat gap after inference window',
    report => {
      report.backgroundHeartbeat.worstFrameGaps[0].startMs = 1600;
      report.backgroundHeartbeat.worstFrameGaps[0].endMs = 1980;
      report.backgroundHeartbeat.worstFrameGaps[0].overlappedEvents[0].tMs = 1700;
    },
    /inferenceWindow|worstFrameGaps/,
  ],
  [
    'heartbeat max gap disagrees with scoped worst interval',
    report => { report.backgroundHeartbeat.responsiveness.maxFrameGapMs = 999; },
    /maxFrameGapMs|worstFrameGaps/,
  ],
  [
    'top-level responsiveness disagrees with heartbeat window',
    report => { report.responsiveness.longFrameCount = 99; },
    /responsiveness|backgroundHeartbeat/,
  ],
  [
    'heartbeat scheduler overlap with no events',
    report => {
      report.backgroundHeartbeat.worstFrameGaps[0].overlapClassification = 'scheduler-event-overlap';
      report.backgroundHeartbeat.worstFrameGaps[0].overlappedEvents = [];
    },
    /overlappedEvents|scheduler-event-overlap/,
  ],
  [
    'heartbeat uninstrumented gap with events',
    report => {
      report.backgroundHeartbeat.worstFrameGaps[0].overlapClassification = 'uninstrumented-gap';
    },
    /uninstrumented-gap|overlappedEvents/,
  ],
  [
    'heartbeat invalid overlap classification',
    report => {
      report.backgroundHeartbeat.worstFrameGaps[0].overlapClassification = 'maybe-overlap';
    },
    /overlapClassification/,
  ],
  [
    'heartbeat zero-length interval with positive duration',
    report => {
      report.backgroundHeartbeat.worstFrameGaps[0].startMs = 1000;
      report.backgroundHeartbeat.worstFrameGaps[0].endMs = 1000;
      report.backgroundHeartbeat.worstFrameGaps[0].durationMs = 380;
      report.backgroundHeartbeat.worstFrameGaps[0].overlappedEvents[0].tMs = 1000;
    },
    /durationMs|endMs/,
  ],
  [
    'heartbeat duration does not match interval',
    report => {
      report.backgroundHeartbeat.worstFrameGaps[0].startMs = 1000;
      report.backgroundHeartbeat.worstFrameGaps[0].endMs = 1200;
      report.backgroundHeartbeat.worstFrameGaps[0].durationMs = 380;
      report.backgroundHeartbeat.worstFrameGaps[0].overlappedEvents[0].tMs = 1100;
    },
    /durationMs|endMs/,
  ],
  [
    'heartbeat overlap event outside interval',
    report => {
      report.backgroundHeartbeat.worstFrameGaps[0].overlappedEvents[0].tMs = 5000;
    },
    /overlappedEvents|tMs/,
  ],
  [
    'heartbeat overlap event without timing',
    report => {
      report.backgroundHeartbeat.worstFrameGaps[0].overlappedEvents = [{}];
    },
    /overlappedEvents|tMs/,
  ],
  [
    'heartbeat overlap interval with reversed bounds',
    report => {
      report.backgroundHeartbeat.worstFrameGaps[0].overlappedEvents = [{
        phase: 'route-tail',
        boundary: 'route-tail',
        kind: 'duty-interval',
        tMs: 1400,
        intervalStartMs: 1400,
        intervalEndMs: 1000,
        durationMs: 400,
      }];
    },
    /intervalStartMs|intervalEndMs|durationMs/,
  ],
  [
    'heartbeat overlap interval with partial timing',
    report => {
      report.backgroundHeartbeat.worstFrameGaps[0].overlappedEvents = [{
        phase: 'route-tail',
        boundary: 'route-tail',
        kind: 'duty-interval',
        tMs: 1200,
        intervalStartMs: 1000,
        durationMs: 200,
      }];
    },
    /intervalStartMs|intervalEndMs/,
  ],
  [
    'heartbeat overlap interval without duration',
    report => {
      report.backgroundHeartbeat.worstFrameGaps[0].overlappedEvents = [{
        phase: 'route-tail',
        boundary: 'route-tail',
        kind: 'duty-interval',
        tMs: 1400,
        intervalStartMs: 1000,
        intervalEndMs: 1400,
      }];
    },
    /durationMs/,
  ],
]) {
  const report = structuredClone(baseReport);
  mutate(report);
  const result = validateSharpContentionWitnessReport(report);
  assert.equal(result.ok, false, name);
  assert.match(result.errors.join('\n'), pattern, name);
}

const baseline = structuredClone(baseReport);
baseline.mode = 'baseline';
baseline.contender.enabled = false;
baseline.contender.submitted = 0;
baseline.contender.completed = 0;
baseline.contender.inferenceWindow = {
  submittedAtStart: 0,
  completedAtStart: 0,
  submittedAtEnd: 0,
  completedAtEnd: 0,
  submittedDelta: 0,
  completedDelta: 0,
};
baseline.contender.progressDuringInference = false;
assert.equal(validateSharpContentionWitnessReport(baseline).ok, true);

console.log('contention witness report contract passed');
