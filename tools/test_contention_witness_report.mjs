import assert from 'node:assert/strict';

import {
  SHARP_CONTENTION_WITNESS_SCHEMA,
  validateSharpContentionWitnessReport,
} from './contention_witness_report.mjs';

const baseReport = {
  schema: SHARP_CONTENTION_WITNESS_SCHEMA,
  runId: 'sharp-contention:test',
  createdAt: '2026-07-02T00:00:00.000Z',
  route: {
    requestedRouteId: 'sharp.image-to-splat.webgpu-local.v0',
    effectiveRouteId: 'sharp.image-to-splat.webgpu-local.v0',
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
      report.contender.progressDuringInference = false;
    },
    /progress/,
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
baseline.contender.progressDuringInference = false;
assert.equal(validateSharpContentionWitnessReport(baseline).ok, true);

console.log('contention witness report contract passed');
