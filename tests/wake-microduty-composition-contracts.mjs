import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { copyMappedBytesCooperatively } from '../src/lib/gpu.js';
import * as schedulerModule from '../src/lib/scheduler.js';

const {
  parseSharpSchedulerConfig,
  schedulerTelemetrySnapshotCooperatively,
} = schedulerModule;

const scheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    spnFusionChunkItems: 524288,
    vitBlockChunkSize: 1,
    vitMicroduty: true,
  },
});

assert.equal(scheduler.effective.spnFusionChunkItems, 524288);
assert.equal(scheduler.effective.vitBlockChunkSize, 1);
assert.equal(scheduler.effective.vitMicroduty, true);
assert.deepEqual(scheduler.unsupportedFields, []);

assert.equal(typeof schedulerModule.planSpnFusionChunks, 'function');
assert.deepEqual(schedulerModule.planSpnFusionChunks(10, 4), [
  { chunkIndex: 0, chunkCount: 3, outputStart: 0, outputEnd: 4, outputCount: 4 },
  { chunkIndex: 1, chunkCount: 3, outputStart: 4, outputEnd: 8, outputCount: 4 },
  { chunkIndex: 2, chunkCount: 3, outputStart: 8, outputEnd: 10, outputCount: 2 },
]);
assert.equal(typeof schedulerModule.planVitBlockMicroduties, 'function');
assert.deepEqual(schedulerModule.planVitBlockMicroduties({
  blockChunkIndex: 4,
  blockStart: 4,
  blockEnd: 5,
  blockCount: 1,
  totalBlocks: 24,
}), [
  { microdutyIndex: 0, blockIndex: 4, microphase: 'attention-residual' },
  { microdutyIndex: 1, blockIndex: 4, microphase: 'mlp-residual' },
]);

const spnSource = readFileSync(new URL('../src/lib/spn.js', import.meta.url), 'utf8');
assert.equal(
  [...spnSource.matchAll(/recordSchedulerEvent\(telemetry, 'spn-fusion-host-dispatch'/g)].length,
  2,
  'both tiled ConvTranspose and initial 1x1 submissions must retain named host-dispatch intervals',
);

let defaultCopyCallbacks = 0;
const sourceBytes = new Uint8Array([1, 2, 3, 4]);
const copiedBytes = await copyMappedBytesCooperatively(sourceBytes, {
  chunkBytes: 0,
  onChunk: async () => {
    defaultCopyCallbacks += 1;
  },
});
assert.deepEqual(copiedBytes, sourceBytes);
assert.equal(defaultCopyCallbacks, 1, 'zero/default readback must remain one unchunked copy');

assert.equal(
  typeof schedulerTelemetrySnapshotCooperatively,
  'function',
  'combined route must retain cooperative telemetry finalization',
);
const mainSource = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
assert.equal(
  [...mainSource.matchAll(/await schedulerTelemetrySnapshotCooperatively\(\s*currentSchedulerTelemetry,\s*'(?:verified|failed)',\s*TERMINAL_TELEMETRY_OPTIONS,\s*\)/g)].length,
  3,
  'success, backbone-only, and failure finalization must all await sealed cooperative snapshots',
);
