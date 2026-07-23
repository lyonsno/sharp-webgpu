import assert from 'node:assert/strict';

import * as schedulerModule from '../src/lib/scheduler.js';

const {
  createSharpRunTelemetry,
  parseSharpSchedulerConfig,
  recordSchedulerEvent,
  schedulerTelemetrySnapshotCooperatively,
} = schedulerModule;

const scheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    decoderKernelChunkItems: 262144,
    decoderKernelMinChunkItems: 65536,
    decoderKernelMaxChunkItems: 8388608,
    decoderKernelTargetDurationMs: 12,
    waitForSubmittedWorkDone: true,
  },
});
const telemetry = createSharpRunTelemetry(scheduler, {
  runId: 'terminal-transfer-contract-run',
  eventCustody: 'sealed-transfer',
});

for (let index = 0; index < 4096; index += 1) {
  recordSchedulerEvent(telemetry, 'monodepth-phase', {
    kind: 'queue-work-done-end',
    role: 'decoder-kernel-output-tile',
    rangeId: `terminal-transfer-contract-run:range:${index}`,
    rangeIndex: index,
    outputStart: index * 16,
    outputEnd: (index + 1) * 16,
    outputCount: 16,
    totalOutputItems: 4096 * 16,
    detail: {
      nested: {
        index,
        label: 'retained-without-terminal-copy',
      },
    },
  });
}

const sourceEvents = telemetry.eventTrace.events;
let taskYieldCount = 0;
const snapshot = await schedulerTelemetrySnapshotCooperatively(
  telemetry,
  'verified',
  {
    chunkEvents: 256,
    eventCustody: 'sealed-transfer',
    jsonProjection: 'compact',
    taskYield: async () => {
      taskYieldCount += 1;
      assert.throws(
        () => sourceEvents.push({ phase: 'late-direct-append' }),
        TypeError,
        'sealed terminal custody must reject direct array appends',
      );
      assert.throws(
        () => {
          sourceEvents[2] = { ...sourceEvents[2], phase: 'same-length-replacement' };
        },
        TypeError,
        'sealed terminal custody must reject same-length event replacement',
      );
      assert.throws(
        () => {
          sourceEvents[1].phase = 'direct-property-mutation';
        },
        TypeError,
        'sealed terminal custody must reject direct event-property mutation',
      );
      assert.throws(
        () => {
          sourceEvents[3].detail.nested.label = 'nested-property-mutation';
        },
        TypeError,
        'sealed terminal custody must reject nested event-property mutation',
      );
      await Promise.resolve();
    },
  },
);

assert.ok(taskYieldCount > 0, 'terminal event indexing must remain cooperative');
assert.ok(
  snapshot.events === sourceEvents,
  'sealed terminal custody must reuse the authoritative event corpus',
);
assert.ok(
  snapshot.eventTrace.events === sourceEvents,
  'event aliases must share the one authoritative corpus',
);
assert.equal(snapshot.snapshotProcess.mode, 'cooperative-sealed-transfer');
assert.equal(snapshot.snapshotProcess.copiedEventCount, 0);
assert.equal(snapshot.snapshotProcess.sourceEventCount, sourceEvents.length);
assert.throws(
  () => recordSchedulerEvent(telemetry, 'route-tail', { kind: 'late-terminal-write' }),
  /scheduler telemetry event corpus is sealed/,
  'terminal transfer must reject later writers instead of mutating archived truth',
);

const projected = JSON.parse(JSON.stringify(snapshot));
assert.equal(projected.schema, snapshot.schema);
assert.equal(projected.runId, snapshot.runId);
assert.equal(projected.eventArchive.retention, 'uncapped');
assert.equal(projected.eventArchive.status, 'resident-sealed');
assert.equal(projected.eventArchive.eventCount, sourceEvents.length);
assert.equal(projected.eventTrace.events, undefined, 'compact JSON must not duplicate the event corpus');
assert.equal(projected.events, undefined, 'compact JSON must not duplicate the legacy event alias');
assert.ok(
  JSON.stringify(projected).length < 100_000,
  'compact terminal JSON must remain independent of the full event payload',
);

assert.equal(
  typeof schedulerModule.createSchedulerTelemetryArchive,
  'function',
  'scheduler must expose an explicit one-ledger archive handoff',
);
const archive = schedulerModule.createSchedulerTelemetryArchive(snapshot);
assert.equal(archive.retention, 'uncapped');
assert.equal(archive.status, 'sealed');
assert.equal(archive.eventCount, sourceEvents.length);
assert.ok(
  archive.events === sourceEvents,
  'archive handoff must retain the exact authoritative event corpus',
);
assert.ok(archive.eventTrace.events === sourceEvents);
assert.equal(
  JSON.parse(JSON.stringify(archive)).events.length,
  sourceEvents.length,
  'explicit archive serialization must retain every event without a hidden cap',
);

const unpreparedTelemetry = createSharpRunTelemetry(scheduler, {
  runId: 'unprepared-sealed-transfer-run',
});
recordSchedulerEvent(unpreparedTelemetry, 'route-tail', { kind: 'boundary-event' });
await assert.rejects(
  schedulerTelemetrySnapshotCooperatively(unpreparedTelemetry, 'verified', {
    eventCustody: 'sealed-transfer',
    jsonProjection: 'compact',
  }),
  /must be prepared when telemetry is created/,
  'late sealed-transfer requests must fail instead of claiming custody over mutable events',
);

console.log('scheduler terminal finalization contracts passed');
