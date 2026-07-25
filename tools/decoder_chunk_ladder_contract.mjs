export const GATE_A_CHUNK_ITEMS = Object.freeze([
  65_536,
  262_144,
  1_048_576,
  4_194_304,
  8_388_608,
]);

export const EXPECTED_SOURCE_SHA256 = '134136dd4086cfc1b887ab0a134c4a2b906223762a0d5959a8b90cc68f11f4f0';
export const EXPECTED_WEIGHTS_SHA256 = '98212168b105c4027aff54c635fe01f547974911deb0c1109d8c05df68a01caf';

export const RAF_THRESHOLDS_MS = Object.freeze([16.67, 25, 33.34, 50, 100]);

function percentile(sorted, percentileValue) {
  if (!sorted.length) return null;
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil((percentileValue / 100) * sorted.length) - 1),
  );
  return sorted[index];
}

export function summarizeRafGaps(gaps) {
  const finite = Array.isArray(gaps)
    ? gaps.filter(value => Number.isFinite(value) && value >= 0)
    : [];
  const sorted = [...finite].sort((a, b) => a - b);
  return {
    sampleCount: finite.length,
    p95Ms: percentile(sorted, 95),
    p99Ms: percentile(sorted, 99),
    maxMs: sorted.length ? sorted[sorted.length - 1] : null,
    thresholdCounts: Object.fromEntries(
      RAF_THRESHOLDS_MS.map(threshold => [
        String(threshold),
        finite.filter(value => value >= threshold).length,
      ]),
    ),
  };
}

export function buildGateAFailureReport({
  activeReport = null,
  phase,
  error,
  failedAt = new Date().toISOString(),
  lastTrustworthyEvidence = null,
  requested = null,
  startedAt = null,
}) {
  return {
    ...(activeReport || {
      schema: 'sharp-webgpu.decoder-chunk-ladder-report.v0',
      startedAt,
      retention: 'uncapped',
      hiddenTimeoutMs: null,
      requestedChunkItems: [...GATE_A_CHUNK_ITEMS],
      requested,
      runs: [],
    }),
    status: 'failed',
    phase,
    failedAt,
    primaryOutputWritten: false,
    error: error?.stack || error?.message || String(error),
    lastTrustworthyEvidence,
  };
}

function sameIntegerArray(actual, expected) {
  return Array.isArray(actual)
    && actual.length === expected.length
    && actual.every((value, index) => value === expected[index]);
}

function validTiming(value) {
  return Number.isFinite(value) && value >= 0;
}

export function validateGateAReport(report) {
  const failures = [];
  if (report?.schema !== 'sharp-webgpu.decoder-chunk-ladder-report.v0') {
    failures.push('wrong Gate A report schema');
  }
  if (report?.retention !== 'uncapped') failures.push('range retention must be uncapped');
  if (report?.hiddenTimeoutMs !== null) failures.push('Gate A must not impose a hidden timeout');
  if (!sameIntegerArray(report?.requestedChunkItems, GATE_A_CHUNK_ITEMS)) {
    failures.push('requested fixed chunk ladder does not match Wake’s five rungs');
  }
  if (report?.routeIdentity?.browser?.headed !== true) {
    failures.push('Gate A requires a headed browser');
  }
  if (report?.routeIdentity?.source?.sha256 !== EXPECTED_SOURCE_SHA256) {
    failures.push('route does not match Wake’s frozen source hash');
  }
  if (report?.routeIdentity?.weights?.sha256 !== EXPECTED_WEIGHTS_SHA256) {
    failures.push('route does not match Wake’s frozen weights hash');
  }
  if (!report?.routeIdentity?.adapter?.effective) {
    failures.push('effective adapter identity is missing');
  }
  if (!report?.routeIdentity?.device?.limits) {
    failures.push('effective device limits are missing');
  }
  if (!report?.routeIdentity?.device?.timestampQuery?.authority) {
    failures.push('timestamp-query identity and timing authority are missing');
  }
  if (report?.fixture?.authority !== 'source-derived-production-shape-not-full-model-intermediate') {
    failures.push('fixture authority must distinguish representative source-derived input from a full model intermediate');
  }
  const totalOutputItems = report?.fixture?.totalOutputItems;
  if (!Number.isSafeInteger(totalOutputItems) || totalOutputItems <= GATE_A_CHUNK_ITEMS.at(-1)) {
    failures.push('production fixture must exceed the largest ladder rung');
  }
  if (
    report?.singleDispatchControl?.dispatchCount !== 1
    || report?.singleDispatchControl?.itemCount !== totalOutputItems
  ) {
    failures.push('single-dispatch control identity is missing or contradictory');
  }
  if (!Array.isArray(report?.runs) || report.runs.length !== GATE_A_CHUNK_ITEMS.length) {
    failures.push('Gate A must contain exactly one run per fixed rung');
    return failures;
  }
  for (let index = 0; index < GATE_A_CHUNK_ITEMS.length; index += 1) {
    const expectedChunkItems = GATE_A_CHUNK_ITEMS[index];
    const run = report.runs[index];
    if (run?.chunkItems !== expectedChunkItems) {
      failures.push(`run ${index} did not preserve chunk size ${expectedChunkItems}`);
    }
    if (run?.itemCount !== totalOutputItems) {
      failures.push(`run ${expectedChunkItems} item count does not match the production fixture`);
    }
    if (!Number.isSafeInteger(run?.dispatchCount) || run.dispatchCount <= 0) {
      failures.push(`run ${expectedChunkItems} dispatch count is missing`);
    }
    if (!Array.isArray(run?.ranges) || run.ranges.length === 0) {
      failures.push(`run ${expectedChunkItems} retained no range observations`);
    } else {
      let cursor = 0;
      for (let rangeIndex = 0; rangeIndex < run.ranges.length; rangeIndex += 1) {
        const range = run.ranges[rangeIndex];
        if (
          range?.rangeIndex !== rangeIndex
          || range?.itemStart !== cursor
          || !Number.isSafeInteger(range?.itemEnd)
          || range.itemEnd <= cursor
          || range.itemCount !== range.itemEnd - range.itemStart
        ) {
          failures.push(`run ${expectedChunkItems} range ${rangeIndex} is discontinuous`);
          break;
        }
        cursor = range.itemEnd;
        for (const timingField of [
          'commandAllocationAndEncodingMs',
          'submitCallMs',
          'submitToQueueCompletionMs',
          'browserYieldMs',
        ]) {
          if (!validTiming(range?.[timingField])) {
            failures.push(`run ${expectedChunkItems} range ${rangeIndex} lacks ${timingField}`);
          }
        }
      }
      if (cursor !== totalOutputItems) {
        failures.push(`run ${expectedChunkItems} range coverage ended at ${cursor}, not ${totalOutputItems}`);
      }
    }
    if (
      run?.exactOutput?.authority !== 'gpu-u32-full-buffer-compare'
      || run?.exactOutput?.comparedItemCount !== totalOutputItems
      || run?.exactOutput?.bitMismatchCount !== 0
    ) {
      failures.push(`run ${expectedChunkItems} did not preserve exact numerical output`);
    }
    if (!run?.raf || !Number.isSafeInteger(run.raf.sampleCount)) {
      failures.push(`run ${expectedChunkItems} lacks rAF distribution evidence`);
    }
    if (!Array.isArray(run?.hostObservations) || run.hostObservations.length === 0) {
      failures.push(`run ${expectedChunkItems} lacks a host observation`);
    }
    for (const timingField of [
      'commandAllocationAndEncodingMs',
      'submitCallMs',
      'submitToQueueCompletionMs',
      'browserYieldMs',
      'wallMs',
      'itemsPerSecond',
    ]) {
      if (!validTiming(run?.totals?.[timingField])) {
        failures.push(`run ${expectedChunkItems} totals lack ${timingField}`);
      }
    }
  }
  return failures;
}
