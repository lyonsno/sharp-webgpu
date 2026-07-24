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
assert.equal(
  snapshot.snapshotProcess.proofIndexPassCount,
  1,
  'terminal boundary proof must share one event-indexing pass with cooperative transfer',
);
assert.equal(
  snapshot.snapshotProcess.proofIndexedEventCount,
  sourceEvents.length,
  'the one-pass proof index must account for the complete uncapped event corpus',
);

const proofDutyTelemetry = createSharpRunTelemetry(scheduler, {
  runId: 'terminal-proof-duty-chunk-contract',
  eventCustody: 'sealed-transfer',
});
for (let dutyIndex = 0; dutyIndex < 6; dutyIndex += 1) {
  const identity = {
    boundary: 'monodepth-phase',
    dutyId: `terminal-proof-duty-chunk-contract:${dutyIndex}`,
    role: 'decoder-kernel-output-tile',
    rangeId: `terminal-proof-duty-chunk-contract:range:${dutyIndex}`,
    outputStart: dutyIndex,
    outputEnd: dutyIndex + 1,
  };
  for (const kind of [
    'chunk-start',
    'queue-work-done-start',
    'queue-work-done-end',
    'js-yield-start',
    'js-yield-end',
  ]) {
    recordSchedulerEvent(proofDutyTelemetry, 'monodepth-phase', {
      ...identity,
      kind,
    });
  }
}
const proofDutyYields = [];
const proofDutySnapshot = await schedulerTelemetrySnapshotCooperatively(
  proofDutyTelemetry,
  'verified',
  {
    chunkEvents: 1000,
    proofChunkDuties: 2,
    eventCustody: 'sealed-transfer',
    jsonProjection: 'compact',
    taskYield: async receipt => {
      proofDutyYields.push(receipt);
      await Promise.resolve();
    },
  },
);
assert.equal(proofDutySnapshot.snapshotProcess.proofDutyCount, 6);
assert.equal(proofDutySnapshot.snapshotProcess.proofDutyChunkSize, 2);
assert.equal(proofDutySnapshot.snapshotProcess.proofDutyTaskYieldCount, 3);
assert.equal(
  proofDutyYields.filter(receipt => Number.isSafeInteger(receipt.proofDutyEnd)).length,
  3,
  'large exact-duty ledgers must yield during proof finalization, not only event indexing',
);

const mutablePrefixTelemetry = createSharpRunTelemetry(scheduler, {
  runId: 'mutable-fixed-prefix-contract',
});
let mutableSecondQueueEnd = null;
for (let dutyIndex = 0; dutyIndex < 2; dutyIndex += 1) {
  const identity = {
    boundary: 'spn-patch-chunk',
    dutyId: `mutable-fixed-prefix-contract:${dutyIndex}`,
    chunkStart: dutyIndex,
    chunkEnd: dutyIndex + 1,
  };
  for (const kind of [
    'chunk-start',
    'queue-work-done-start',
    'queue-work-done-end',
    'js-yield-start',
    'js-yield-end',
  ]) {
    const event = recordSchedulerEvent(mutablePrefixTelemetry, 'spn-patch-chunk', {
      ...identity,
      kind,
    });
    if (dutyIndex === 1 && kind === 'queue-work-done-end') {
      mutableSecondQueueEnd = event;
    }
  }
}
let mutablePrefixMutationApplied = false;
const mutablePrefixSnapshot = await schedulerTelemetrySnapshotCooperatively(
  mutablePrefixTelemetry,
  'verified',
  {
    chunkEvents: 1000,
    proofChunkDuties: 1,
    taskYield: async receipt => {
      if (!mutablePrefixMutationApplied && Number.isSafeInteger(receipt.proofDutyEnd)) {
        mutableSecondQueueEnd.boundary = 'mutated-after-fixed-prefix-clone';
        mutablePrefixMutationApplied = true;
      }
      await Promise.resolve();
    },
  },
);
assert.equal(mutablePrefixMutationApplied, true);
assert.equal(mutableSecondQueueEnd.boundary, 'mutated-after-fixed-prefix-clone');
assert.equal(
  mutablePrefixSnapshot.events.find(event => (
    event.dutyId === 'mutable-fixed-prefix-contract:1'
    && event.kind === 'queue-work-done-end'
  )).boundary,
  'spn-patch-chunk',
  'a non-sealed cooperative snapshot must retain its cloned fixed prefix',
);
assert.equal(
  mutablePrefixSnapshot.boundaryAssertions.find(
    assertion => assertion.field === 'phaseChunkSize.spnPatch',
  ).status,
  'verified',
  'boundary assertions must derive from the returned fixed-prefix events, not later live mutation',
);

const fixedDecoderScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    decoderKernelChunkItems: 2,
    decoderKernelTargetDurationMs: 0,
    waitForSubmittedWorkDone: true,
    yieldMs: 4,
  },
});
const mutableDecoderTelemetry = createSharpRunTelemetry(fixedDecoderScheduler, {
  runId: 'mutable-decoder-fixed-prefix-contract',
});
let mutableDecoderStart = null;
for (let tileIndex = 0; tileIndex < 2; tileIndex += 1) {
  const identity = {
    boundary: 'monodepth-phase',
    dutyId: `mutable-decoder-fixed-prefix-contract:${tileIndex}`,
    role: 'decoder-kernel-output-tile',
    tileIndex,
    tileTotal: 2,
    outputStart: tileIndex * 2,
    outputEnd: (tileIndex + 1) * 2,
    outputCount: 2,
    totalOutputItems: 4,
    configuredChunkItems: 2,
  };
  for (const kind of [
    'chunk-start',
    'queue-work-done-start',
    'queue-work-done-end',
    'js-yield-start',
    'js-yield-end',
  ]) {
    const event = recordSchedulerEvent(mutableDecoderTelemetry, 'monodepth-phase', {
      ...identity,
      kind,
    });
    if (tileIndex === 0 && kind === 'chunk-start') mutableDecoderStart = event;
  }
}
let mutableDecoderMutationApplied = false;
const mutableDecoderSnapshot = await schedulerTelemetrySnapshotCooperatively(
  mutableDecoderTelemetry,
  'verified',
  {
    chunkEvents: 1000,
    proofChunkDuties: 1,
    taskYield: async receipt => {
      if (!mutableDecoderMutationApplied && Number.isSafeInteger(receipt.proofDutyEnd)) {
        mutableDecoderStart.configuredChunkItems = 999;
        mutableDecoderMutationApplied = true;
      }
      await Promise.resolve();
    },
  },
);
assert.equal(mutableDecoderMutationApplied, true);
assert.equal(mutableDecoderStart.configuredChunkItems, 999);
assert.equal(
  mutableDecoderSnapshot.events.find(event => (
    event.dutyId === 'mutable-decoder-fixed-prefix-contract:0'
    && event.kind === 'chunk-start'
  )).configuredChunkItems,
  2,
  'decoder coverage must retain its cloned fixed-prefix configuration',
);
assert.equal(
  mutableDecoderSnapshot.boundaryAssertions.find(
    assertion => assertion.field === 'decoderKernelChunkItems',
  ).status,
  'verified',
  'decoder coverage assertions must derive from fixed-prefix events after task yields',
);
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
