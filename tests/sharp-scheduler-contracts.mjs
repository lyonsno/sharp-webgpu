import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  createSharpRunTelemetry,
  parseSharpSchedulerConfig,
  recordSchedulerEvent,
  schedulerTelemetrySnapshot,
} from '../src/lib/scheduler.js';

const root = new URL('..', import.meta.url).pathname;
const schedulerPath = join(root, 'src', 'lib', 'scheduler.js');
const mainPath = join(root, 'src', 'main.js');
const spnPath = join(root, 'src', 'lib', 'spn.js');
const backbonePath = join(root, 'src', 'lib', 'backbone.js');
const gaussianPath = join(root, 'src', 'lib', 'gaussian_decoder.js');
const measurementToolPath = join(root, 'tools', 'scheduler_measurement.mjs');

assert.ok(existsSync(schedulerPath), 'SHARP-WebGPU must expose a scheduler contract module');

const requested = {
  mode: 'cooperative',
  spnPatchChunkSize: 1,
  yieldMs: 5,
  waitForSubmittedWorkDone: true,
  gaussianPhaseYieldMs: 7,
  vitBlockChunkSize: 2,
};
const scheduler = parseSharpSchedulerConfig({ sharpScheduler: requested });
assert.equal(scheduler.schema, 'sharp-webgpu.scheduler-config.v0');
assert.equal(scheduler.requested.spnPatchChunkSize, 1);
assert.equal(scheduler.effective.spnPatchChunkSize, 1);
assert.equal(scheduler.effective.yieldMs, 5);
assert.equal(scheduler.effective.waitForSubmittedWorkDone, true);
assert.equal(scheduler.effective.gaussianPhaseYieldMs, 7);
assert.equal(scheduler.effective.vitBlockChunkSize, 2, 'requested ViT block chunking must become effective only on the unfusing branch');
assert.deepEqual(scheduler.unsupportedFields, []);

const telemetry = createSharpRunTelemetry(scheduler, { runId: 'contract-run' });
recordSchedulerEvent(telemetry, 'spn-patch-chunk', {
  chunkStart: 0,
  chunkEnd: 1,
  waitedForSubmittedWorkDone: true,
  yieldMs: 5,
});
const snapshot = schedulerTelemetrySnapshot(telemetry);
assert.equal(snapshot.schema, 'sharp-webgpu.scheduler-telemetry.v0');
assert.equal(snapshot.runId, 'contract-run');
assert.equal(snapshot.requestedScheduler.spnPatchChunkSize, 1);
assert.equal(snapshot.effectiveScheduler.spnPatchChunkSize, 1);
assert.equal(snapshot.effectiveScheduler.vitBlockChunkSize, 2);
assert.deepEqual(snapshot.unsupportedFields, []);
assert.equal(snapshot.events[0].phase, 'spn-patch-chunk');
assert.equal(snapshot.status, 'verified');

const mainSource = readFileSync(mainPath, 'utf8');
assert.match(mainSource, /parseSharpSchedulerConfig/, 'main entry must parse caller scheduler config at run time');
assert.match(mainSource, /window\.__SHARP_LAST_RUN_TELEMETRY__/, 'browser route must expose last scheduler telemetry for Kaminos');
assert.match(mainSource, /schedulerTelemetrySnapshot/, 'main entry must publish a normalized scheduler telemetry snapshot');

const spnSource = readFileSync(spnPath, 'utf8');
assert.doesNotMatch(spnSource, /const\s+CHUNK_SIZE\s*=\s*4/, 'SPN patch chunking must not be a hidden singleton constant');
assert.match(spnSource, /effective\.spnPatchChunkSize/, 'SPN patch chunking must use the effective scheduler config');
assert.match(spnSource, /spn-patch-chunk/, 'SPN must record breathing evidence around patch chunks');
assert.match(spnSource, /effective\.vitBlockChunkSize/, 'SPN must choose split ViT encoding from effective scheduler config');
assert.match(spnSource, /encodeSplit/, 'SPN must call the split ViT encoder only when scheduler config requests it');

const backboneSource = readFileSync(backbonePath, 'utf8');
assert.match(backboneSource, /async\s+encodeSplit\(/, 'ViTEncoder must expose an opt-in split encoder path');
assert.match(backboneSource, /vit-block-segment/, 'split ViT encoder must record block-segment scheduler telemetry');
assert.match(backboneSource, /segmentStartLayer/, 'split ViT telemetry must record the block segment start layer');
assert.match(backboneSource, /segmentEndLayer/, 'split ViT telemetry must record the block segment end layer');
assert.match(backboneSource, /queue\.submit/, 'split ViT encoder must submit command buffers at segment boundaries');

const gaussianSource = readFileSync(gaussianPath, 'utf8');
assert.match(gaussianSource, /gaussianPhaseYieldMs/, 'Gaussian decoder phase breathing must use the scheduler config');
assert.match(gaussianSource, /gaussian-phase/, 'Gaussian decoder must record phase-level breathing evidence');

assert.ok(existsSync(measurementToolPath), 'SHARP scheduler branch must ship a JSON-writing measurement witness');
const measurementToolSource = readFileSync(measurementToolPath, 'utf8');
assert.match(measurementToolSource, /--scheduler/, 'measurement witness must accept scheduler config as an invocation input');
assert.match(measurementToolSource, /--out/, 'measurement witness must write to a caller-chosen report path');
assert.match(measurementToolSource, /__SHARP_LAST_RUN_TELEMETRY__/, 'measurement witness must read browser scheduler telemetry');
assert.match(measurementToolSource, /vit-block-segment/, 'measurement witness must summarize split ViT block segment events');
assert.match(measurementToolSource, /scheduler-unverified/, 'measurement witness must fail loud when browser telemetry is missing');
