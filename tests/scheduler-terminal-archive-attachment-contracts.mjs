import assert from 'node:assert/strict';

import * as schedulerModule from '../src/lib/scheduler.js';

const scheduler = schedulerModule.parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    waitForSubmittedWorkDone: true,
  },
});
const telemetry = schedulerModule.createSharpRunTelemetry(scheduler, {
  runId: 'terminal-archive-attachment-run',
  eventCustody: 'sealed-transfer',
});
schedulerModule.recordSchedulerEvent(telemetry, 'route-tail', {
  kind: 'boundary-event',
});
const snapshot = await schedulerModule.schedulerTelemetrySnapshotCooperatively(
  telemetry,
  'failed',
  {
    eventCustody: 'sealed-transfer',
    jsonProjection: 'compact',
  },
);

assert.equal(
  typeof schedulerModule.attachSchedulerTelemetryArchive,
  'function',
  'scheduler must expose reusable archive attachment for returned and thrown failures',
);

const runDebug = { schedulerTelemetry: snapshot };
const returnedFailure = schedulerModule.attachSchedulerTelemetryArchive({
  ok: false,
  error: 'expected failure',
  runDebug,
}, snapshot);
const archive = returnedFailure.schedulerTelemetryArchive;
assert.equal(archive.eventCount, 1);
assert.ok(archive.events === snapshot.events);
assert.ok(runDebug.schedulerTelemetryArchive === archive);

const thrownFailure = new Error('expected thrown failure');
thrownFailure.sharpRunDebug = runDebug;
schedulerModule.attachSchedulerTelemetryArchive(
  thrownFailure,
  snapshot,
  { debugTarget: runDebug },
);
assert.ok(
  thrownFailure.schedulerTelemetryArchive === archive,
  'thrown and returned failure surfaces must reuse one archive object',
);
assert.ok(
  thrownFailure.sharpRunDebug.schedulerTelemetryArchive === archive,
  'thrown debug custody must retain the explicit uncapped archive',
);
assert.equal(
  Object.keys(thrownFailure).includes('schedulerTelemetryArchive'),
  false,
  'archive attachment must stay outside routine enumerable error JSON',
);
assert.equal(
  JSON.stringify(returnedFailure).includes('terminal-archive-attachment-run'),
  true,
  'routine failure JSON must retain the compact scheduler identity',
);
assert.ok(
  JSON.stringify(returnedFailure).length < 100_000,
  'routine failure JSON must not traverse the explicit archive',
);

console.log('scheduler terminal archive attachment contracts passed');
