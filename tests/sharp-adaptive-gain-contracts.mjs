import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  createSharpRunTelemetry,
  parseSharpSchedulerConfig,
  schedulerTelemetrySnapshot,
} from '../src/lib/scheduler.js';

const requestedGain = 0.375;
const adaptiveRequest = {
  decoderKernelChunkItems: 262144,
  decoderKernelMinChunkItems: 65536,
  decoderKernelMaxChunkItems: 8388608,
  decoderKernelTargetDurationMs: 12,
  decoderKernelAdjustmentGain: requestedGain,
  waitForSubmittedWorkDone: true,
};

const scheduler = parseSharpSchedulerConfig({ sharpScheduler: adaptiveRequest });
assert.equal(
  scheduler.requested.decoderKernelAdjustmentGain,
  requestedGain,
  'the requested adaptive gain must remain visible',
);
assert.equal(
  scheduler.effective.decoderKernelAdjustmentGain,
  requestedGain,
  'the explicit SHARP adaptive profile gain must become effective unchanged',
);
assert.equal(
  scheduler.effective.decoderKernelTargetDurationMs,
  12,
  'gain uptake must preserve the measured 12ms target',
);
assert.equal(scheduler.effective.decoderKernelMinChunkItems, 65536);
assert.equal(scheduler.effective.decoderKernelMaxChunkItems, 8388608);
assert.deepEqual(
  scheduler.unsupportedFields,
  [],
  'adaptive gain must be a first-class scheduler field',
);
const telemetry = schedulerTelemetrySnapshot(
  createSharpRunTelemetry(scheduler, { runId: 'sharp-adaptive-gain-contract' }),
);
assert.equal(telemetry.requestedScheduler.decoderKernelAdjustmentGain, requestedGain);
assert.equal(telemetry.effectiveScheduler.decoderKernelAdjustmentGain, requestedGain);

const defaultGainScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    ...adaptiveRequest,
    decoderKernelAdjustmentGain: undefined,
  },
});
assert.equal(
  defaultGainScheduler.effective.decoderKernelAdjustmentGain,
  1,
  'omitted gain must preserve the package full-gain default',
);

for (const invalidGain of [null, '0.375', false, 0, -0.1, 1.01, Number.NaN, Infinity]) {
  assert.throws(
    () => parseSharpSchedulerConfig({
      sharpScheduler: {
        ...adaptiveRequest,
        decoderKernelAdjustmentGain: invalidGain,
      },
    }),
    /decoderKernelAdjustmentGain/,
    `invalid adaptive gain ${String(invalidGain)} must fail loud`,
  );
}

const decoderDutySource = readFileSync(
  new URL('../src/lib/decoder_duties.js', import.meta.url),
  'utf8',
);
assert.match(
  decoderDutySource,
  /adjustmentGain:\s*effective\.decoderKernelAdjustmentGain/,
  'the decoder duty must carry the effective gain into planner creation',
);
assert.match(
  decoderDutySource,
  /adjustmentGain:\s*adaptiveDuty\.adjustmentGain/,
  'planner creation must forward the effective gain to the public kit planner',
);
assert.match(
  decoderDutySource,
  /adjustment:\s*observation\.adjustment/,
  'SHARP telemetry must preserve the planner full-gain and effective correction object',
);

console.log('SHARP adaptive gain contracts passed');
