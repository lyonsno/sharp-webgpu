import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { createWebGpuAdaptiveCommandDutyPlanner } from '@kaminos/webgpu-inference-kit';
import { createDecoderAdaptiveDuty } from '../src/lib/decoder_adaptive_profile.js';
import {
  adaptiveDecoderObservationTelemetryDetails,
  createSharpRunTelemetry,
  parseSharpSchedulerConfig,
  recordSchedulerEvent,
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
assert.equal(
  defaultGainScheduler.requested.decoderKernelAdjustmentGain,
  1,
  'explicit undefined must serialize as the same requested full-gain default as omission',
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
  /adjustmentGain:\s*adaptiveDuty\.adjustmentGain/,
  'planner creation must forward the effective gain to the public kit planner',
);
assert.match(
  decoderDutySource,
  /adaptiveDecoderObservationTelemetryDetails\(observation\)/,
  'the production decoder path must preserve the complete planner observation',
);

const executableRequest = {
  decoderKernelChunkItems: 100,
  decoderKernelMinChunkItems: 50,
  decoderKernelMaxChunkItems: 500,
  decoderKernelTargetDurationMs: 12,
  decoderKernelAdjustmentGain: requestedGain,
  waitForSubmittedWorkDone: true,
};
const executableScheduler = parseSharpSchedulerConfig({
  sharpScheduler: executableRequest,
});
const adaptiveDuty = createDecoderAdaptiveDuty(
  executableScheduler,
  createSharpRunTelemetry(executableScheduler, { runId: 'sharp-partial-gain-duty' }),
  'gaussian',
);
const planner = createWebGpuAdaptiveCommandDutyPlanner({
  plannerId: 'sharp-adaptive-gain-executable-contract',
  unit: adaptiveDuty.unit,
  totalItems: 1000,
  initialChunkItems: adaptiveDuty.initialChunkItems,
  targetDurationMs: adaptiveDuty.targetDurationMs,
  adjustmentGain: adaptiveDuty.adjustmentGain,
  bounds: adaptiveDuty.bounds,
  retention: 'uncapped',
});
const range = planner.nextRange();
const observation = planner.observeRange({
  rangeId: range.rangeId,
  observedDurationMs: 48,
  timingAuthority: 'queue-work-done',
});
const observationDetails = adaptiveDecoderObservationTelemetryDetails(observation);
assert.equal(observationDetails.requestedAdjustmentGain, requestedGain);
assert.equal(observationDetails.effectiveAdjustmentGain, requestedGain);
assert.equal(observationDetails.observedDurationMs, 48);
assert.equal(observationDetails.targetDurationMs, 12);
assert.equal(observationDetails.fullGainCorrectionRatio, 0.25);
assert.ok(
  Math.abs(observationDetails.effectiveCorrectionRatio - (0.25 ** requestedGain)) < 1e-12,
);
assert.equal(observationDetails.observedChunkItems, 100);
assert.equal(observationDetails.rawNextChunkItems, 25);
assert.ok(
  Math.abs(observationDetails.effectiveRawNextChunkItems - (100 * (0.25 ** requestedGain))) < 1e-10,
);
assert.equal(observationDetails.nextChunkItems, 59);
assert.equal(observationDetails.adjustment, 'decrease');
assert.equal(observationDetails.boundApplication, null);

const fullGainRequest = {
  ...executableRequest,
  decoderKernelAdjustmentGain: 1,
};
const fullGainScheduler = parseSharpSchedulerConfig({
  sharpScheduler: fullGainRequest,
});
const fullGainDuty = createDecoderAdaptiveDuty(
  fullGainScheduler,
  createSharpRunTelemetry(fullGainScheduler, { runId: 'sharp-full-gain-duty' }),
  'gaussian',
);
const fullGainPlanner = createWebGpuAdaptiveCommandDutyPlanner({
  plannerId: 'sharp-full-gain-compatibility-contract',
  unit: fullGainDuty.unit,
  totalItems: 1000,
  initialChunkItems: fullGainDuty.initialChunkItems,
  targetDurationMs: fullGainDuty.targetDurationMs,
  adjustmentGain: fullGainDuty.adjustmentGain,
  bounds: fullGainDuty.bounds,
  retention: 'uncapped',
});
const fullGainRange = fullGainPlanner.nextRange();
const fullGainObservation = fullGainPlanner.observeRange({
  rangeId: fullGainRange.rangeId,
  observedDurationMs: 48,
  timingAuthority: 'queue-work-done',
});
assert.equal(fullGainObservation.fullGainCorrectionRatio, 0.25);
assert.equal(fullGainObservation.effectiveCorrectionRatio, 0.25);
assert.equal(fullGainObservation.rawNextChunkItems, 25);
assert.equal(fullGainObservation.effectiveRawNextChunkItems, 25);
assert.equal(fullGainObservation.nextChunkItems, 50);
assert.equal(fullGainObservation.boundApplication, 'minChunkItems');

const observationTelemetry = createSharpRunTelemetry(scheduler, {
  runId: 'sharp-adaptive-observation-contract',
});
recordSchedulerEvent(observationTelemetry, 'gaussian-decode-phase', {
  phase: 'decoder.fusion.4.resnet1.conv1',
  role: 'decoder-kernel-output-tile-observation',
  ...observationDetails,
});
const serializedObservationTelemetry = JSON.parse(JSON.stringify(
  schedulerTelemetrySnapshot(observationTelemetry),
));
const serializedObservation = serializedObservationTelemetry.events[0];
for (const field of [
  'requestedAdjustmentGain',
  'effectiveAdjustmentGain',
  'observedDurationMs',
  'targetDurationMs',
  'fullGainCorrectionRatio',
  'effectiveCorrectionRatio',
  'observedChunkItems',
  'rawNextChunkItems',
  'effectiveRawNextChunkItems',
  'nextChunkItems',
  'adjustment',
  'boundApplication',
]) {
  assert.ok(
    Object.prototype.hasOwnProperty.call(serializedObservation, field),
    `serialized scheduler observation must retain ${field}`,
  );
}

console.log('SHARP adaptive gain contracts passed');
