import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
  buildGateAFailureReport,
  EXPECTED_SOURCE_SHA256,
  EXPECTED_WEIGHTS_SHA256,
  GATE_A_CHUNK_ITEMS,
  RAF_THRESHOLDS_MS,
  summarizeRafGaps,
  validateGateAReport,
} from '../tools/decoder_chunk_ladder_contract.mjs';

assert.equal(EXPECTED_SOURCE_SHA256, '134136dd4086cfc1b887ab0a134c4a2b906223762a0d5959a8b90cc68f11f4f0');
assert.equal(EXPECTED_WEIGHTS_SHA256, '98212168b105c4027aff54c635fe01f547974911deb0c1109d8c05df68a01caf');

assert.deepEqual(
  GATE_A_CHUNK_ITEMS,
  [65_536, 262_144, 1_048_576, 4_194_304, 8_388_608],
  'Gate A must exercise Wake’s complete fixed ladder without a hidden subset',
);
assert.deepEqual(RAF_THRESHOLDS_MS, [16.67, 25, 33.34, 50, 100]);

assert.deepEqual(summarizeRafGaps([10, 20, 30, 40, 110]), {
  sampleCount: 5,
  p95Ms: 110,
  p99Ms: 110,
  maxMs: 110,
  thresholdCounts: {
    '16.67': 4,
    '25': 3,
    '33.34': 2,
    '50': 1,
    '100': 1,
  },
});

const routeIdentity = {
  source: {
    path: '/fixture/17_img.png',
    sha256: EXPECTED_SOURCE_SHA256,
  },
  weights: {
    path: '/fixture/weights.bin',
    sha256: EXPECTED_WEIGHTS_SHA256,
    tensor: 'feature_model.image_encoder.conv.weight',
  },
  browser: {
    requestedExecutable: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    effectiveVersion: 'Chrome/149',
    headed: true,
  },
  adapter: {
    requestedPowerPreference: 'high-performance',
    effective: { vendor: 'apple', architecture: 'metal-3' },
  },
  device: {
    limits: { maxBufferSize: 1_073_741_824 },
    timestampQuery: {
      requested: true,
      effective: false,
      authority: 'cpu-wall-fallback',
    },
  },
};

const makeRun = chunkItems => {
  const itemCount = 75_497_472;
  const ranges = [];
  for (let itemStart = 0, rangeIndex = 0; itemStart < itemCount; rangeIndex += 1) {
    const rangeItemCount = Math.min(chunkItems, itemCount - itemStart);
    ranges.push({
      rangeIndex,
      itemStart,
      itemEnd: itemStart + rangeItemCount,
      itemCount: rangeItemCount,
      commandAllocationAndEncodingMs: 0.2,
      submitCallMs: 0.01,
      submitToQueueCompletionMs: 1.5,
      browserYieldMs: 16.7,
    });
    itemStart += rangeItemCount;
  }
  return {
    chunkItems,
    itemCount,
    dispatchCount: ranges.length,
    ranges,
    totals: {
      commandAllocationAndEncodingMs: 1,
      submitCallMs: 0.1,
      submitToQueueCompletionMs: 10,
      browserYieldMs: 20,
      wallMs: 31.1,
      itemsPerSecond: 2_427_571,
    },
    exactOutput: {
      comparedItemCount: itemCount,
      bitMismatchCount: 0,
      authority: 'gpu-u32-full-buffer-compare',
    },
    raf: summarizeRafGaps([16, 17, 18]),
    hostObservations: [{
      observedAt: '2026-07-25T00:00:00.000Z',
      process: { cpuPercent: 90, residentBytes: 1000 },
      hostMemory: { compressedBytes: 0, swapUsedBytes: 0 },
      thermalPower: { status: 'available', raw: 'CPU_Speed_Limit = 100' },
      gpu: { status: 'unavailable', reason: 'no-unprivileged-counter' },
    }],
  };
};

const validReport = {
  schema: 'sharp-webgpu.decoder-chunk-ladder-report.v0',
  status: 'complete',
  retention: 'uncapped',
  hiddenTimeoutMs: null,
  routeIdentity,
  fixture: {
    authority: 'source-derived-production-shape-not-full-model-intermediate',
    operation: 'feature_model.image_encoder.conv',
    params: {
      inC: 5,
      inH: 1536,
      inW: 1536,
      outC: 128,
      kH: 2,
      kW: 2,
      strideH: 2,
      strideW: 2,
    },
    totalOutputItems: 75_497_472,
  },
  singleDispatchControl: {
    itemCount: 75_497_472,
    dispatchCount: 1,
  },
  requestedChunkItems: [...GATE_A_CHUNK_ITEMS],
  runs: GATE_A_CHUNK_ITEMS.map(makeRun),
};

assert.deepEqual(validateGateAReport(validReport), []);
assert.match(
  validateGateAReport({
    ...validReport,
    requestedChunkItems: GATE_A_CHUNK_ITEMS.slice(1),
  }).join('\n'),
  /fixed chunk ladder/,
);
assert.match(
  validateGateAReport({
    ...validReport,
    routeIdentity: {
      ...routeIdentity,
      browser: { ...routeIdentity.browser, headed: false },
    },
  }).join('\n'),
  /headed browser/,
);
assert.match(
  validateGateAReport({
    ...validReport,
    routeIdentity: {
      ...routeIdentity,
      source: { ...routeIdentity.source, sha256: '1'.repeat(64) },
    },
  }).join('\n'),
  /frozen source hash/,
);
assert.match(
  validateGateAReport({
    ...validReport,
    routeIdentity: {
      ...routeIdentity,
      weights: { ...routeIdentity.weights, sha256: '2'.repeat(64) },
    },
  }).join('\n'),
  /frozen weights hash/,
);
assert.match(
  validateGateAReport({
    ...validReport,
    runs: validReport.runs.map((run, index) => index === 2
      ? { ...run, exactOutput: { ...run.exactOutput, bitMismatchCount: 1 } }
      : run),
  }).join('\n'),
  /exact numerical output/,
);
assert.match(
  validateGateAReport({
    ...validReport,
    runs: validReport.runs.map((run, index) => index === 0
      ? { ...run, hostObservations: [] }
      : run),
  }).join('\n'),
  /host observation/,
);
assert.match(
  validateGateAReport({ ...validReport, hiddenTimeoutMs: 30_000 }).join('\n'),
  /hidden timeout/,
);

const lateFailure = buildGateAFailureReport({
  activeReport: {
    ...validReport,
    status: 'running',
    phase: 'ladder-8388608-complete',
    runs: validReport.runs.slice(0, 4),
  },
  phase: 'validation',
  error: new Error('simulated late validation failure'),
  failedAt: '2026-07-25T00:01:00.000Z',
  lastTrustworthyEvidence: {
    completedRungCount: 4,
    lastCompletedChunkItems: 4_194_304,
  },
});
assert.equal(lateFailure.status, 'failed');
assert.equal(lateFailure.phase, 'validation');
assert.equal(lateFailure.runs.length, 4, 'late failure must preserve every completed rung');
assert.equal(lateFailure.lastTrustworthyEvidence.completedRungCount, 4);
assert.match(lateFailure.error, /simulated late validation failure/);

const runner = await readFile(new URL('../tools/benchmark_decoder_chunk_ladder_browser.mjs', import.meta.url), 'utf8');
assert.match(runner, /headless:\s*false/, 'Gate A must launch visible Chrome');
assert.doesNotMatch(runner, /--headless/, 'Gate A must not smuggle a headless browser flag into Chrome');
assert.match(runner, /setDefaultTimeout\(0\)/, 'Gate A must not install a hidden Puppeteer timeout');
assert.match(runner, /primaryOutputWritten/, 'Gate A failure must preserve a durable report even before completion');

console.log('decoder chunk ladder contracts passed');
