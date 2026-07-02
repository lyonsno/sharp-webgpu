import assert from 'node:assert/strict';

import {
  SHARP_CONTENTION_WITNESS_SCHEMA,
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
