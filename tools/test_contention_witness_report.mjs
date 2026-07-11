import assert from 'node:assert/strict';

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

function createReceipt(overrides = {}) {
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
        sha256: 'sha256-splat',
        shape: [1179648, 14],
      },
      depthMap: {
        artifactId: 'depth-map:test',
        sha256: 'sha256-depth',
        shape: [768, 768, 1],
      },
      metadata: {
        artifactId: 'sharp-metadata:test',
        sha256: 'sha256-metadata',
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
    ...(overrides.receiptFields || {}),
  };
}

const routeReceipt = createReceipt();

const baseReport = {
  schema: SHARP_CONTENTION_WITNESS_SCHEMA,
  runId: 'sharp-contention:test',
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
      numGaussians: 1179648,
      plyAvailable: true,
    },
  },
  responsiveness: {
    rafFrames: 240,
    maxFrameGapMs: 380,
    p95FrameGapMs: 92,
    longFrameCount: 8,
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
      rafFrames: 240,
      maxFrameGapMs: 380,
      p95FrameGapMs: 92,
      longFrameCount: 8,
    },
    eventTrace: {
      schema: 'kaminos.webgpu-scheduler-event-trace.v0',
      timingAuthority: 'browser-wall-clock',
      eventCount: 4,
      boundaries: ['spn-patch-chunk', 'vit-block-chunk'],
    },
    inferenceWindow: {
      startMs: 900,
      endMs: 1500,
      durationMs: 600,
    },
    worstFrameGaps: [
      {
        startMs: 1000,
        endMs: 1380,
        durationMs: 380,
        overlapClassification: 'scheduler-event-overlap',
        overlappedEvents: [
          {
            phase: 'spn-patch-chunk',
            boundary: 'spn-patch-chunk',
            kind: 'queue-work-done-start',
            tMs: 1012,
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
    mode: 'throughput',
    verificationState: 'scheduler-unverified',
  },
};

assert.deepEqual(validateSharpContentionWitnessReport(baseReport), {
  ok: true,
  errors: [],
  warnings: [],
});

const constructedHeartbeat = createSharpBackgroundHeartbeatReport({
  scheduler: {
    status: 'verified',
    requestedScheduler: baseReport.backgroundHeartbeat.requestedScheduler,
    effectiveScheduler: baseReport.backgroundHeartbeat.effectiveScheduler,
    eventTrace: {
      schema: 'kaminos.webgpu-scheduler-event-trace.v0',
      timingAuthority: 'browser-wall-clock',
      events: [
        { phase: 'spn-patch-chunk', boundary: 'spn-patch-chunk', kind: 'chunk-start', tMs: 1008 },
        { phase: 'vit-block-chunk', boundary: 'vit-block-chunk', kind: 'queue-work-done-start', tMs: 1120 },
      ],
    },
  },
  probe: {
    ...baseReport.responsiveness,
    inferenceWindow: {
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
  startMs: 900,
  endMs: 1450,
  durationMs: 550,
});
assert.equal(constructedHeartbeat.worstFrameGaps.length, 1, 'heartbeat must exclude gaps outside the measured inference window');
assert.equal(constructedHeartbeat.worstFrameGaps[0].durationMs, 400);
assert.equal(constructedHeartbeat.worstFrameGaps[0].overlapClassification, 'scheduler-event-overlap');
assert.equal(constructedHeartbeat.worstFrameGaps[0].overlappedEvents.length, 2);

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
    worstFrameGaps: [
      { startMs: 900.0006, endMs: 950, durationMs: 49.9994 },
      { startMs: 975, endMs: 1000.0004, durationMs: 25.0004 },
    ],
  },
});
assert.equal(fractionalBoundaryHeartbeat.worstFrameGaps.length, 2, 'rounding must not drop raw boundary-clipped gaps');
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
assert.equal('backgroundHeartbeat' in appFailure, false, 'failure artifacts must not expose a normal heartbeat receipt');

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
    'heartbeat without inference window',
    report => { delete report.backgroundHeartbeat.inferenceWindow; },
    /inferenceWindow/,
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
