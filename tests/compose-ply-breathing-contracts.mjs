import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import * as composeModule from '../src/lib/compose.js';
import * as schedulerModule from '../src/lib/scheduler.js';

function legacyWritePLY(plyData, numGaussians, imgW, imgH, focalPx) {
  const header = `ply
format binary_little_endian 1.0
element vertex ${numGaussians}
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
element intrinsic 9
property float intrinsic
element image_size 2
property uint image_size
element color_space 1
property uchar color_space
end_header
`;
  const parts = [
    new TextEncoder().encode(header),
    new Uint8Array(plyData.buffer, plyData.byteOffset, plyData.byteLength),
    new Uint8Array(new Float32Array([focalPx, 0, imgW * 0.5, 0, focalPx, imgH * 0.5, 0, 0, 1]).buffer),
    new Uint8Array(new Uint32Array([imgW, imgH]).buffer),
    new Uint8Array([1]),
  ];
  const totalSize = parts.reduce((sum, part) => sum + part.byteLength, 0);
  const combined = new Uint8Array(totalSize);
  let offset = 0;
  for (const part of parts) {
    combined.set(part, offset);
    offset += part.byteLength;
  }
  return combined;
}

assert.equal(typeof composeModule.writePLY, 'function', 'PLY assembly must be independently testable');

const plyData = new Float32Array(Array.from({ length: 28 }, (_, index) => index * 0.125 - 1));
const legacyBytes = legacyWritePLY(plyData, 2, 640, 480, 640);
const multipartBlob = composeModule.writePLY(plyData, 2, 640, 480, 640);
assert.deepEqual(
  new Uint8Array(await multipartBlob.arrayBuffer()),
  legacyBytes,
  'multipart PLY assembly must remain byte-identical to the legacy combined-buffer layout',
);

const imgH = 4;
const imgW = 4;
const outH = 2;
const outW = 2;
const dispData = new Float32Array(2 * imgH * imgW).fill(0.5);
const geomDeltas = new Float32Array(6 * outH * outW);
const texDeltas = new Float32Array(22 * outH * outW);
const img01 = new Float32Array(3 * imgH * imgW).fill(0.5);
const chunks = [];

const pendingCompose = composeModule.composeAndExport(
  dispData,
  geomDeltas,
  texDeltas,
  img01,
  imgH,
  imgW,
  outH,
  outW,
  640,
  480,
  640,
  {
    chunkItems: 2,
    onChunk: async chunk => { chunks.push(chunk); },
  },
);
assert.ok(pendingCompose instanceof Promise, 'cooperative composition must be asynchronous');
const composed = await pendingCompose;
assert.equal(composed.numGaussians, 8);
assert.ok(composed.plyBlob.size > 0);
for (const step of ['depth-normalize', 'depth-min', 'depth-rescale', 'base-disparity', 'base-grid', 'base-color']) {
  assert.equal(
    chunks.filter(chunk => chunk.step === step).length,
    1,
    `cooperative composition must donate one boundary after ${step}, not yield on every CPU chunk`,
  );
}
assert.equal(chunks.filter(chunk => chunk.step === 'gaussian-compose').length, 4, 'only the heavy Gaussian loop should use repeated item chunks');
for (const chunk of chunks) {
  assert.ok(chunk.processedItems > 0);
  assert.ok(chunk.totalItems >= chunk.processedItems);
}

const nonDivisorChunks = [];
await composeModule.composeAndExport(
  dispData,
  geomDeltas,
  texDeltas,
  img01,
  imgH,
  imgW,
  outH,
  outW,
  640,
  480,
  640,
  {
    chunkItems: 5,
    onChunk: async chunk => { nonDivisorChunks.push(chunk); },
  },
);
const prepChunks = nonDivisorChunks.filter(chunk => chunk.phaseComplete === true);
assert.equal(prepChunks.length, 6, 'non-divisor chunk sizes must still emit every prep phase completion');

const mainSource = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
assert.equal(typeof schedulerModule.classifyCpuDutyCheckpoint, 'function', 'CPU duty checkpoint authority must be directly testable');

const scheduler = { effective: { cpuChunkItems: 5 } };
assert.deepEqual(
  schedulerModule.classifyCpuDutyCheckpoint(scheduler, {
    stage: 'compose-ply',
    step: 'depth-normalize',
    phaseComplete: true,
  }, 32),
  { eligible: true, phaseComplete: true },
  'trusted compose preparation completion must bypass non-divisor modulo alignment',
);
assert.deepEqual(
  schedulerModule.classifyCpuDutyCheckpoint(scheduler, {
    stage: 'compose-ply',
    step: 'gaussian-compose',
    phaseComplete: true,
  }, 3),
  { eligible: false, phaseComplete: false },
  'gaussian chunks must not spoof phase completion',
);
assert.deepEqual(
  schedulerModule.classifyCpuDutyCheckpoint(scheduler, {
    stage: 'output-capture',
    step: 'depth-preview-pixels',
    phaseComplete: true,
  }, 3),
  { eligible: false, phaseComplete: false },
  'output-capture chunks must not spoof compose phase completion',
);
assert.deepEqual(
  schedulerModule.classifyCpuDutyCheckpoint(scheduler, {
    stage: 'compose-ply',
    step: 'gaussian-compose',
  }, 5),
  { eligible: true, phaseComplete: false },
  'ordinary repeated chunks remain eligible on true modulo boundaries',
);
assert.match(mainSource, /classifyCpuDutyCheckpoint/, 'main route must consume the tested checkpoint authority classifier');

console.log('compose/PLY breathing contracts passed');
