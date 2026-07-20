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
assert.equal(typeof composeModule.writePLYAsync, 'function', 'PLY assembly must expose an asynchronous worker path');

const plyData = new Float32Array(Array.from({ length: 28 }, (_, index) => index * 0.125 - 1));
const legacyBytes = legacyWritePLY(plyData, 2, 640, 480, 640);
const multipartBlob = composeModule.writePLY(plyData, 2, 640, 480, 640);
assert.deepEqual(
  new Uint8Array(await multipartBlob.arrayBuffer()),
  legacyBytes,
  'multipart PLY assembly must remain byte-identical to the legacy combined-buffer layout',
);

class InlinePlyWorker {
  constructor({ fail = false, synchronous = false, throwOnTerminate = false } = {}) {
    this.fail = fail;
    this.synchronous = synchronous;
    this.throwOnTerminate = throwOnTerminate;
    this.terminated = false;
    this.terminateCalls = 0;
    this.onmessage = null;
    this.onerror = null;
    this.onmessageerror = null;
  }

  postMessage(message, transfers) {
    assert.equal(message.type, 'assemble-ply');
    assert.deepEqual(transfers, [message.plyBuffer], 'worker assembly must transfer, not clone, the PLY float buffer');
    if (this.fail) {
      const fail = () => this.onerror?.({ message: 'synthetic worker failure', error: new Error('synthetic worker failure') });
      if (this.synchronous) fail();
      else setTimeout(fail, 0);
      return;
    }
    const ownedMessage = structuredClone(message, { transfer: transfers });
    const complete = () => {
      const ownedPlyData = new Float32Array(
        ownedMessage.plyBuffer,
        ownedMessage.plyByteOffset,
        ownedMessage.plyLength,
      );
      const plyBlob = composeModule.writePLY(
        ownedPlyData,
        ownedMessage.numGaussians,
        ownedMessage.imgW,
        ownedMessage.imgH,
        ownedMessage.focalPx,
      );
      this.onmessage?.({
        data: {
          type: 'ply-assembled',
          requestId: ownedMessage.requestId,
          plyBlob,
          bytes: plyBlob.size,
        },
      });
    };
    if (this.synchronous) complete();
    else setTimeout(complete, 0);
  }

  terminate() {
    this.terminateCalls += 1;
    this.terminated = true;
    if (this.throwOnTerminate) throw new Error('synthetic terminate failure');
  }
}

const workerPlyData = new Float32Array(plyData);
let worker = null;
let foregroundTurnObserved = false;
setTimeout(() => { foregroundTurnObserved = true; }, 0);
const pendingWorkerPly = composeModule.writePLYAsync(workerPlyData, 2, 640, 480, 640, {
  mode: 'worker',
  workerFactory: () => {
    worker = new InlinePlyWorker();
    return worker;
  },
});
assert.ok(pendingWorkerPly instanceof Promise, 'worker PLY assembly must return a promise immediately');
assert.equal(workerPlyData.byteLength, 0, 'worker PLY assembly must transfer ownership of the source allocation');
const workerBlob = await pendingWorkerPly;
assert.equal(foregroundTurnObserved, true, 'worker PLY assembly must leave an event-loop opportunity before completion');
assert.equal(worker.terminated, true, 'one-shot PLY workers must terminate after successful output');
assert.deepEqual(
  new Uint8Array(await workerBlob.arrayBuffer()),
  legacyBytes,
  'worker PLY assembly must remain byte-identical to the synchronous output',
);

let failedWorker = null;
await assert.rejects(
  composeModule.writePLYAsync(new Float32Array(plyData), 2, 640, 480, 640, {
    mode: 'worker',
    workerFactory: () => {
      failedWorker = new InlinePlyWorker({ fail: true });
      return failedWorker;
    },
  }),
  /PLY worker failed during ply-blob-assembly: synthetic worker failure/,
  'worker failures must reject without silently falling back to main-thread assembly',
);
assert.equal(failedWorker.terminated, true, 'failed PLY workers must terminate');

let throwingSuccessWorker = null;
const cleanupSafeSuccess = await Promise.race([
  composeModule.writePLYAsync(new Float32Array(plyData), 2, 640, 480, 640, {
    mode: 'worker',
    workerFactory: () => {
      throwingSuccessWorker = new InlinePlyWorker({ synchronous: true, throwOnTerminate: true });
      return throwingSuccessWorker;
    },
  }),
  new Promise((_, reject) => setTimeout(() => reject(new Error('worker success did not settle after cleanup failure')), 50)),
]);
assert.equal(cleanupSafeSuccess.size, multipartBlob.size, 'cleanup failure must not suppress a valid worker result');
assert.equal(throwingSuccessWorker.terminateCalls, 1, 'success cleanup must attempt termination exactly once');

let throwingFailureWorker = null;
await assert.rejects(
  Promise.race([
    composeModule.writePLYAsync(new Float32Array(plyData), 2, 640, 480, 640, {
      mode: 'worker',
      workerFactory: () => {
        throwingFailureWorker = new InlinePlyWorker({ fail: true, synchronous: true, throwOnTerminate: true });
        return throwingFailureWorker;
      },
    }),
    new Promise((_, reject) => setTimeout(() => reject(new Error('worker failure did not settle after cleanup failure')), 50)),
  ]),
  error => {
    assert.match(error.message, /PLY worker failed during ply-blob-assembly: synthetic worker failure/);
    assert.deepEqual(error.cleanupError, {
      name: 'Error',
      message: 'synthetic terminate failure',
    });
    return true;
  },
  'cleanup failure must not replace the primary worker error or leave rejection unsettled',
);
assert.equal(throwingFailureWorker.terminateCalls, 1, 'failure cleanup must attempt termination exactly once');
await assert.rejects(
  composeModule.writePLYAsync(new Float32Array(plyData), 2, 640, 480, 640, { mode: 'mystery' }),
  /Unsupported PLY assembly mode: mystery/,
  'unknown output modes must fail before materialization',
);
let malformedTerminateCalls = 0;
await assert.rejects(
  composeModule.writePLYAsync(new Float32Array(plyData), 2, 640, 480, 640, {
    mode: 'worker',
    workerFactory: () => ({ terminate: () => { malformedTerminateCalls += 1; } }),
  }),
  /PLY worker failed during ply-blob-assembly: worker factory returned an invalid Worker/,
);
assert.equal(malformedTerminateCalls, 1, 'invalid terminable worker-like objects must be retired before rejection');
await assert.rejects(
  composeModule.writePLYAsync(new Float32Array(plyData), 2, 640, 480, 640, { mode: '' }),
  /Unsupported PLY assembly mode:/,
  'an explicit blank output mode must not silently collapse to the default',
);
await assert.rejects(
  composeModule.writePLYAsync(new Float32Array(plyData), 2, 640, 480, 640, {
    mode: 'worker',
    workerFactory: () => ({}),
  }),
  /PLY worker failed during ply-blob-assembly: worker factory returned an invalid Worker/,
  'a malformed worker factory must fail with output-phase identity before transferring data',
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
const intervals = [];

const failedAssemblyIntervals = [];
await assert.rejects(
  composeModule.composeAndExport(
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
      plyAssemblyMode: 'worker',
      plyWorkerFactory: () => new InlinePlyWorker({ fail: true }),
      onInterval: interval => { failedAssemblyIntervals.push(interval); },
    },
  ),
  /PLY worker failed during ply-blob-assembly: synthetic worker failure/,
);
assert.deepEqual(
  failedAssemblyIntervals.at(-1),
  {
    step: 'ply-blob-assembly',
    status: 'failed',
    assemblyMode: 'worker',
    executionThread: 'worker',
    lastTrustworthyStep: 'gaussian-compose',
    intervalStartMs: failedAssemblyIntervals.at(-1).intervalStartMs,
    intervalEndMs: failedAssemblyIntervals.at(-1).intervalEndMs,
    durationMs: failedAssemblyIntervals.at(-1).durationMs,
    error: {
      name: 'PlyWorkerError',
      message: 'PLY worker failed during ply-blob-assembly: synthetic worker failure',
    },
  },
  'failed output assembly must preserve phase, effective mode, duration, and source error before rejecting',
);
assert.ok(failedAssemblyIntervals.at(-1).intervalEndMs >= failedAssemblyIntervals.at(-1).intervalStartMs);

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
    onInterval: interval => { intervals.push(interval); },
  },
);
assert.ok(pendingCompose instanceof Promise, 'cooperative composition must be asynchronous');
const composed = await pendingCompose;
assert.equal(composed.numGaussians, 8);
assert.ok(composed.plyBlob.size > 0);
assert.equal(composed.plyAssemblyMode, 'main-thread', 'default composition must report its effective output mode');
assert.deepEqual(
  intervals.map(interval => interval.step),
  [
    'depth-normalize',
    'depth-min',
    'depth-rescale',
    'base-disparity',
    'base-grid',
    'base-color',
    'ply-data-allocation',
    'gaussian-activation-setup',
    'ply-blob-assembly',
  ],
  'composition must expose every preparation phase before allocation/setup and final Blob assembly',
);
for (const interval of intervals) {
  assert.ok(Number.isFinite(interval.intervalStartMs));
  assert.ok(Number.isFinite(interval.intervalEndMs));
  assert.ok(interval.intervalEndMs >= interval.intervalStartMs);
  assert.equal(interval.durationMs, interval.intervalEndMs - interval.intervalStartMs);
}
const allocationInterval = intervals.find(interval => interval.step === 'ply-data-allocation');
const setupInterval = intervals.find(interval => interval.step === 'gaussian-activation-setup');
const assemblyInterval = intervals.find(interval => interval.step === 'ply-blob-assembly');
assert.equal(allocationInterval.bytes, 8 * 14 * Float32Array.BYTES_PER_ELEMENT);
assert.equal(assemblyInterval.assemblyMode, 'main-thread');
assert.equal(assemblyInterval.executionThread, 'main');
assert.ok(allocationInterval.intervalEndMs <= setupInterval.intervalStartMs);
assert.ok(setupInterval.intervalEndMs <= chunks.find(chunk => chunk.step === 'gaussian-compose').intervalStartMs);
for (const step of ['depth-normalize', 'depth-min', 'depth-rescale', 'base-disparity', 'base-grid', 'base-color']) {
  const interval = intervals.find(candidate => candidate.step === step);
  const checkpoint = chunks.find(chunk => chunk.step === step);
  assert.ok(interval.intervalEndMs <= checkpoint.intervalStartMs || !Number.isFinite(checkpoint.intervalStartMs));
}
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
for (const chunk of chunks.filter(chunk => chunk.step === 'gaussian-compose')) {
  assert.ok(Number.isFinite(chunk.intervalStartMs), 'Gaussian checkpoints must expose CPU work interval starts');
  assert.ok(Number.isFinite(chunk.intervalEndMs), 'Gaussian checkpoints must expose CPU work interval ends');
  assert.ok(chunk.intervalEndMs >= chunk.intervalStartMs);
  assert.equal(chunk.durationMs, chunk.intervalEndMs - chunk.intervalStartMs);
}

const subRowChunks = [];
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
    chunkItems: 1,
    onChunk: async chunk => { subRowChunks.push(chunk); },
  },
);
const subRowGaussianChunks = subRowChunks.filter(chunk => chunk.step === 'gaussian-compose');
assert.deepEqual(
  subRowGaussianChunks.map(chunk => chunk.processedItems),
  [2, 4, 6, 8],
  'sub-row thresholds must emit one interval per actual completed row, not post-work nominal checkpoint slivers',
);
assert.deepEqual(
  subRowGaussianChunks.map(chunk => [chunk.segmentStartProcessedItems, chunk.segmentEndProcessedItems]),
  [[0, 2], [2, 4], [4, 6], [6, 8]],
  'row-batched interval authority must name the actual work bounds between yields',
);
assert.ok(subRowGaussianChunks.every(chunk => chunk.granularity === 'row-batched'));
assert.ok(subRowGaussianChunks.every(chunk => chunk.segmentEndProcessedItems > chunk.segmentStartProcessedItems));

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
assert.equal(prepChunks.length, 7, 'non-divisor chunk sizes must emit every prep completion plus the Gaussian remainder completion');
assert.deepEqual(
  nonDivisorChunks.filter(chunk => chunk.step === 'gaussian-compose').map(chunk => chunk.processedItems),
  [6, 8],
  'Gaussian interval evidence must cover the final non-divisor remainder instead of stopping at the last modulo checkpoint',
);
assert.equal(
  nonDivisorChunks.filter(chunk => chunk.step === 'gaussian-compose').at(-1).phaseComplete,
  true,
  'the final Gaussian remainder interval must be explicitly complete',
);

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
    stage: 'compose-ply',
    step: 'gaussian-compose',
    phaseComplete: true,
    totalItems: 8,
  }, 8),
  { eligible: true, phaseComplete: true },
  'Gaussian phase completion is authoritative only when processedItems equals the declared total',
);
assert.deepEqual(
  schedulerModule.classifyCpuDutyCheckpoint(scheduler, {
    stage: 'compose-ply',
    step: 'gaussian-compose',
    checkpointItems: 5,
    segmentStartProcessedItems: 0,
    segmentEndProcessedItems: 6,
    granularity: 'row-batched',
  }, 6),
  { eligible: true, phaseComplete: false },
  'row-batched authority must accept the actual completed segment that crosses a nominal threshold',
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
assert.match(mainSource, /assemblyMode:\s*details\.assemblyMode/, 'route-tail telemetry must preserve effective PLY assembly mode');
assert.match(mainSource, /status:\s*details\.status/, 'route-tail telemetry must preserve PLY assembly completion or failure status');
assert.match(mainSource, /error:\s*details\.error/, 'route-tail telemetry must preserve PLY worker failure identity');
assert.match(mainSource, /plyAssemblyMode:\s*composed\.plyAssemblyMode/, 'completed route outputs must preserve the effective PLY assembly mode');

console.log('compose/PLY breathing contracts passed');
