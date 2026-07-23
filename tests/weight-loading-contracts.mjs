import assert from 'node:assert/strict';

import * as weightsModule from '../src/lib/weights.js';

assert.equal(
  typeof weightsModule.readWeightResponse,
  'function',
  'weight loading must expose deterministic streamed response assembly',
);
assert.equal(
  typeof weightsModule.createWeightTensorAccessors,
  'function',
  'weight loading must expose immutable GPU tensor access with cache identity',
);

function responseFromChunks(chunks, declaredBytes, contentEncoding = null) {
  let index = 0;
  return {
    headers: {
      get(name) {
        if (name.toLowerCase() === 'content-length') return String(declaredBytes);
        if (name.toLowerCase() === 'content-encoding') return contentEncoding;
        return null;
      },
    },
    body: {
      getReader() {
        return {
          async read() {
            if (index >= chunks.length) return { done: true, value: undefined };
            const value = chunks[index];
            index += 1;
            return { done: false, value };
          },
        };
      },
    },
  };
}

const phaseEvents = [];
const yieldEvents = [];
const streamed = await weightsModule.readWeightResponse(
  responseFromChunks([
    new Uint8Array([1, 2, 3]),
    new Uint8Array([4, 5]),
  ], 5),
  {
    onPhase: event => phaseEvents.push(event),
    yieldControl: async event => yieldEvents.push(event),
  },
);
assert.deepEqual([...new Uint8Array(streamed.buffer)], [1, 2, 3, 4, 5]);
assert.equal(streamed.receivedBytes, 5);
assert.equal(streamed.declaredBytes, 5);
assert.equal(streamed.assemblyMode, 'preallocated-content-length');
assert.equal(streamed.postDownloadCopyBytes, 0, 'trusted content length must eliminate the post-download full-buffer recopy');
assert.deepEqual(
  phaseEvents.map(event => [event.phase, event.status]),
  [
    ['fetch-stream', 'started'],
    ['fetch-stream', 'completed'],
    ['buffer-consolidation', 'skipped'],
  ],
  'stream telemetry must distinguish network receipt from skipped consolidation',
);
assert.equal(yieldEvents.at(-1)?.phase, 'fetch-stream-complete', 'weight loading must yield after network receipt before parsing or upload');

const fallbackPhases = [];
const fallback = await weightsModule.readWeightResponse(
  responseFromChunks([
    new Uint8Array([9, 8]),
    new Uint8Array([7]),
  ], null),
  { onPhase: event => fallbackPhases.push(event) },
);
assert.deepEqual([...new Uint8Array(fallback.buffer)], [9, 8, 7]);
assert.equal(fallback.declaredBytes, 0);
assert.equal(fallback.assemblyMode, 'chunk-consolidation');
assert.equal(fallback.postDownloadCopyBytes, 3);
assert.deepEqual(
  fallbackPhases.map(event => [event.phase, event.status]),
  [
    ['fetch-stream', 'started'],
    ['fetch-stream', 'completed'],
    ['buffer-consolidation', 'started'],
    ['buffer-consolidation', 'completed'],
  ],
  'missing content length must preserve an uncapped, explicitly measured consolidation path',
);

const encoded = await weightsModule.readWeightResponse(
  responseFromChunks([new Uint8Array([6, 5, 4])], 2, 'gzip'),
);
assert.equal(encoded.declaredBytes, 0, 'compressed transfer length must not be mistaken for decoded body length');
assert.equal(encoded.assemblyMode, 'chunk-consolidation');
assert.deepEqual([...new Uint8Array(encoded.buffer)], [6, 5, 4]);

const failureEvents = [];
await assert.rejects(
  weightsModule.readWeightResponse(
    responseFromChunks([new Uint8Array([1, 2, 3])], 5),
    { onPhase: event => failureEvents.push(event) },
  ),
  /declared 5 bytes but received 3/,
);
assert.equal(failureEvents.at(-1)?.phase, 'fetch-stream');
assert.equal(failureEvents.at(-1)?.status, 'failed', 'stream failure must preserve its exact phase before primary output exists');

const tensors = new Map([
  ['shared.weight', { dtype: 0, offset: 0, size: 4 }],
]);
const extractions = [];
const accessors = weightsModule.createWeightTensorAccessors(
  {},
  new ArrayBuffer(4),
  tensors,
  {
    extractGpuTensor(device, buffer, info) {
      const extracted = { device, buffer, info, ordinal: extractions.length };
      extractions.push(extracted);
      return extracted;
    },
    extractCpuTensor: () => new Float32Array([7]),
  },
);
const firstWeight = accessors.get('shared.weight');
const secondWeight = accessors.get('shared.weight');
assert.equal(firstWeight, secondWeight, 'immutable weight buffers must retain stable GPU identity');
assert.equal(accessors.tryGet('shared.weight'), firstWeight);
assert.equal(extractions.length, 1, 'repeated tensor access must not remap and recopy the same immutable weight');
assert.equal(accessors.tryGet('missing.weight'), null);
assert.throws(() => accessors.get('missing.weight'), /Missing weight: missing\.weight/);
assert.deepEqual([...accessors.extractTensorCPU('shared.weight')], [7]);

console.log('weight loading contracts passed');
