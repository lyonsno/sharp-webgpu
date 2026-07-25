import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
  createSharpRunTelemetry,
  parseSharpSchedulerConfig,
  recordSchedulerEvent,
} from '../src/lib/scheduler.js';

const observed = [];
const telemetry = createSharpRunTelemetry(
  parseSharpSchedulerConfig({
    sharpScheduler: {
      mode: 'cooperative',
      decoderKernelChunkItems: 262_144,
      decoderKernelMinChunkItems: 65_536,
      decoderKernelMaxChunkItems: 8_388_608,
      decoderKernelTargetDurationMs: 12,
      decoderKernelAdjustmentGain: 0.375,
      waitForSubmittedWorkDone: true,
    },
  }),
  {
    runId: 'live-event-sink-contract',
    onEvent: event => observed.push(event),
  },
);
const event = recordSchedulerEvent(telemetry, 'gaussian-phase', {
  kind: 'decoder-kernel-range-observed',
  role: 'decoder-kernel-output-tile-observation',
  rangeId: 'sharp:live-event-sink-contract:gaussian:0:image-encoder:range:0',
  rangeIndex: 0,
  outputStart: 0,
  outputEnd: 262_144,
  outputCount: 262_144,
});
assert.equal(observed.length, 1);
assert.equal(observed[0], event, 'the live sink must receive the exact event retained by telemetry');
assert.equal(telemetry.events[0], event, 'live streaming must not substitute the terminal archive row');

const manyObserved = [];
const manyTelemetry = createSharpRunTelemetry(
  parseSharpSchedulerConfig({ sharpScheduler: { mode: 'default' } }),
  { onEvent: row => manyObserved.push(row) },
);
for (let index = 0; index < 10_000; index += 1) {
  recordSchedulerEvent(manyTelemetry, 'contract', { ordinal: index });
}
assert.equal(manyObserved.length, 10_000, 'the event sink must not impose a hidden row cap');
assert.equal(manyTelemetry.events.length, 10_000);

const failingTelemetry = createSharpRunTelemetry(
  parseSharpSchedulerConfig({ sharpScheduler: { mode: 'default' } }),
  {
    onEvent() {
      throw new Error('injected live telemetry sink failure');
    },
  },
);
assert.throws(
  () => recordSchedulerEvent(failingTelemetry, 'contract', {}),
  /injected live telemetry sink failure/,
  'a broken evidence sink must fail loud instead of letting an unjournaled route continue',
);

const main = await readFile(new URL('../src/main.js', import.meta.url), 'utf8');
assert.match(
  main,
  /createSharpRunTelemetry\(currentScheduler,\s*\{[\s\S]{0,300}onEvent:\s*options\.onTelemetry/,
  'the product entry point must connect the caller telemetry sink before scheduler work begins',
);

console.log('SHARP live event sink contracts passed');
