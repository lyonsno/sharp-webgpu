import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  createSharpRunTelemetry,
  recordSchedulerEvent,
  schedulerTelemetrySnapshot,
} from '../src/lib/scheduler.js';
import { createAuthenticatedSharpRouteMetadata } from '../src/lib/route_runtime.js';

const episodeId = 'firing-composed-world-contract-001';
const scheduler = {
  requested: { mode: 'cooperative' },
  effective: { mode: 'cooperative' },
  unsupportedFields: [],
};
const telemetry = createSharpRunTelemetry(scheduler, { runId: episodeId });
recordSchedulerEvent(telemetry, 'source-preprocess', { kind: 'duty-start' });
recordSchedulerEvent(telemetry, 'compose-ply', { kind: 'duty-end' });
const snapshot = schedulerTelemetrySnapshot(telemetry, 'verified');
const terminalOutput = {
  plySha256: 'a'.repeat(64),
  plyByteLength: 66_060_836,
  numGaussians: 1_179_648,
  completeness: 'complete',
};

const metadata = createAuthenticatedSharpRouteMetadata({
  episodeId,
  routeId: 'sharp.image-to-splat.webgpu-local.v0',
  schedulerTelemetry: snapshot,
  terminalOutput,
  routeEvidence: {
    phases: [{ name: 'spn', ms: 10 }],
    elapsedMs: 10,
  },
});

assert.equal(metadata.schema, 'sharp.webgpu-route-metadata.v0');
assert.equal(metadata.episodeId, episodeId);
assert.deepEqual(metadata.terminalOutput, terminalOutput);
assert.equal(metadata.schedulerTrace.runId, episodeId);
assert.deepEqual(metadata.schedulerTrace.eventSequence, {
  firstSequence: 0,
  lastSequence: 1,
  nextSequence: 2,
  eventCount: 2,
});
assert.equal(metadata.routeId, 'sharp.image-to-splat.webgpu-local.v0');
assert.deepEqual(metadata.phases, [{ name: 'spn', ms: 10 }]);
assert.equal(Object.isFrozen(metadata), true);
assert.equal(Object.isFrozen(metadata.terminalOutput), true);

assert.throws(
  () => createAuthenticatedSharpRouteMetadata({
    episodeId: 'firing-other-episode',
    routeId: metadata.routeId,
    schedulerTelemetry: snapshot,
    terminalOutput,
  }),
  /episode|run/i,
  'a valid foreign scheduler episode must not be spliceable into this route receipt',
);
assert.throws(
  () => createAuthenticatedSharpRouteMetadata({
    episodeId,
    routeId: metadata.routeId,
    schedulerTelemetry: snapshot,
    terminalOutput,
    routeEvidence: { episodeId: 'firing-route-evidence-overwrite' },
  }),
  /reserved|episode/i,
  'generic route evidence must not overwrite the authenticated episode envelope',
);

const mainSource = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
assert.match(
  mainSource,
  /createSharpRunTelemetry\(currentScheduler, \{[\s\S]{0,300}runId: options\.episodeId/,
  'the callable product route must make the invocation-owned episode id the SHARP scheduler run id',
);
assert.match(
  mainSource,
  /createAuthenticatedSharpRouteMetadata\(\{[\s\S]{0,500}episodeId: runDebug\.schedulerTelemetry\?\.runId[\s\S]{0,500}plyByteLength:/,
  'the route receipt must authenticate the scheduler episode and terminal PLY identity together',
);
assert.match(
  mainSource,
  /finishRouteRun\(runDebug, 'real', \{[\s\S]{0,300}plyByteLength:\s*composed\.plyBlob\.size/,
  'the canonical run-debug output must preserve the same terminal PLY byte length authenticated by the route receipt',
);

console.log('SHARP composed-world product episode contracts passed');
