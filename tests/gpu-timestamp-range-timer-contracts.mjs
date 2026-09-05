import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { createGpuTimestampRangeTimer } from '../src/lib/gpu_timestamp_range_timer.js';

const usageDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'GPUBufferUsage');
const mapModeDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'GPUMapMode');

function timestampBytes(startedAtNs, completedAtNs) {
  const bytes = new ArrayBuffer(16);
  const view = new DataView(bytes);
  view.setBigUint64(0, BigInt(startedAtNs), true);
  view.setBigUint64(8, BigInt(completedAtNs), true);
  return bytes;
}

try {
  Object.defineProperty(globalThis, 'GPUBufferUsage', {
    configurable: true,
    value: { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, QUERY_RESOLVE: 512 },
  });
  Object.defineProperty(globalThis, 'GPUMapMode', {
    configurable: true,
    value: { READ: 1 },
  });

  let mappedBytes = timestampBytes(1_000_000n, 1_065_536n);
  let mapCount = 0;
  let unmapCount = 0;
  const destroyed = [];
  const querySet = { destroy: () => destroyed.push('query-set') };
  const resolveBuffer = { destroy: () => destroyed.push('resolve-buffer') };
  const readBuffer = {
    async mapAsync(mode, offset, size) {
      assert.equal(mode, GPUMapMode.READ);
      assert.equal(offset, 0);
      assert.equal(size, 16);
      mapCount += 1;
    },
    getMappedRange(offset, size) {
      assert.equal(offset, 0);
      assert.equal(size, 16);
      return mappedBytes;
    },
    unmap() { unmapCount += 1; },
    destroy: () => destroyed.push('read-buffer'),
  };
  const device = {
    features: new Set(['timestamp-query']),
    createQuerySet(descriptor) {
      assert.deepEqual(descriptor, {
        type: 'timestamp',
        count: 2,
        label: 'contract-range:query-set',
      });
      return querySet;
    },
    createBuffer(descriptor) {
      assert.equal(descriptor.size, 16);
      return descriptor.usage & GPUBufferUsage.MAP_READ ? readBuffer : resolveBuffer;
    },
  };
  const commands = [];
  const encoder = {
    resolveQuerySet(observedQuerySet, firstQuery, queryCount, destination, offset) {
      assert.equal(observedQuerySet, querySet);
      assert.equal(destination, resolveBuffer);
      commands.push(`resolve:${firstQuery}:${queryCount}:${offset}`);
    },
    copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
      assert.equal(source, resolveBuffer);
      assert.equal(destination, readBuffer);
      commands.push(`copy:${sourceOffset}:${destinationOffset}:${size}`);
    },
  };
  assert.equal('writeTimestamp' in encoder, false, 'the current GPUCommandEncoder contract has no writeTimestamp method');

  const timer = createGpuTimestampRangeTimer(device, { label: 'contract-range' });
  const computePassDescriptor = timer.begin();
  assert.deepEqual(computePassDescriptor, {
    timestampWrites: {
      querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    },
  });
  commands.push('decoder-commands');
  timer.end(encoder);
  assert.deepEqual(commands, [
    'decoder-commands',
    'resolve:0:2:0',
    'copy:0:0:16',
  ]);
  const measurement = await timer.read();
  assert.equal(measurement.authority, 'timestamp-query-inside-submitted-command-buffer');
  assert.equal(measurement.measurementStatus, 'observed-positive-range');
  assert.equal(measurement.rawDurationNs, '65536');
  assert.equal(measurement.durationNs, '65536');
  assert.equal(measurement.durationMs, 0.065536);
  assert.equal(measurement.resolutionUpperBoundNs, null);
  assert.equal(mapCount, 1);
  assert.equal(unmapCount, 1);

  mappedBytes = timestampBytes(8_000_000n, 8_000_000n);
  timer.begin();
  timer.end(encoder);
  const censoredMeasurement = await timer.read();
  assert.equal(censoredMeasurement.authority, 'timestamp-query-inside-submitted-command-buffer');
  assert.equal(censoredMeasurement.measurementStatus, 'resolution-censored-upper-bound');
  assert.equal(censoredMeasurement.startedAtNs, '8000000');
  assert.equal(censoredMeasurement.completedAtNs, '8000000');
  assert.equal(censoredMeasurement.rawDurationNs, '0');
  assert.equal(censoredMeasurement.durationNs, '65536');
  assert.equal(censoredMeasurement.durationMs, 0.065536);
  assert.equal(censoredMeasurement.resolutionUpperBoundNs, '65536');
  assert.equal(unmapCount, 2);

  mappedBytes = timestampBytes(8_000_001n, 8_000_000n);
  timer.begin();
  timer.end(encoder);
  await assert.rejects(
    () => timer.read(),
    /reversed/,
    'reversed timestamp endpoints remain fatal',
  );
  assert.equal(unmapCount, 3);

  timer.destroy();
  assert.deepEqual(destroyed.sort(), ['query-set', 'read-buffer', 'resolve-buffer']);
  timer.destroy();

  const coldQuerySet = { destroy() {} };
  const coldResolveBuffer = { destroy() {} };
  const coldReadBuffer = {
    async mapAsync() {},
    getMappedRange() { return timestampBytes(9_000_000n, 9_000_000n); },
    unmap() {},
    destroy() {},
  };
  const coldDevice = {
    features: new Set(['timestamp-query']),
    createQuerySet() { return coldQuerySet; },
    createBuffer(descriptor) {
      return descriptor.usage & GPUBufferUsage.MAP_READ ? coldReadBuffer : coldResolveBuffer;
    },
  };
  const coldEncoder = {
    resolveQuerySet() {},
    copyBufferToBuffer() {},
  };
  const coldTimer = createGpuTimestampRangeTimer(coldDevice, { label: 'cold-zero-range' });
  coldTimer.begin();
  coldTimer.end(coldEncoder);
  await assert.rejects(
    () => coldTimer.read(),
    /without a prior positive same-device resolution bound/,
    'a zero range without prior same-device timing evidence remains fatal',
  );
  coldTimer.destroy();

  assert.throws(
    () => createGpuTimestampRangeTimer({ features: new Set() }),
    /requires the timestamp-query device feature/,
    'adaptive timing fails loud rather than falling back to host callback timing',
  );

  const decoderSource = readFileSync(new URL('../src/lib/decoder_duties.js', import.meta.url), 'utf8');
  assert.match(
    decoderSource,
    /encodeTile\([\s\S]{0,200}gpuTimestampTimer\?\.begin\(\)[\s\S]{0,700}gpuTimestampTimer\?\.end\(encoder\)/,
    'GPU pass timestamps must bracket the adaptive decoder kernel before query resolution',
  );
  assert.match(
    decoderSource,
    /gpuTimestampTimer\.read\(\)[\s\S]{0,200}adaptiveDecoderTimingObservation\(yieldReceipt,\s*gpuTimestampRange\)/,
    'adaptive planning must consume the resolved GPU timestamp measurement',
  );
  const shaderOpsSource = readFileSync(new URL('../src/lib/shader_ops.js', import.meta.url), 'utf8');
  for (const operation of [
    'dispatchConv2d',
    'dispatchConv1x1',
    'dispatchGroupNormPartialStats',
    'dispatchGroupNormNormalizeRelu',
    'dispatchConvTranspose2d',
  ]) {
    assert.match(
      shaderOpsSource,
      new RegExp(`function ${operation}\\([\\s\\S]{0,7000}beginComputePass\\(params\\.computePassDescriptor\\)`),
      `${operation} must attach adaptive timestamps through the compute-pass descriptor`,
    );
  }
} finally {
  if (usageDescriptor) Object.defineProperty(globalThis, 'GPUBufferUsage', usageDescriptor);
  else delete globalThis.GPUBufferUsage;
  if (mapModeDescriptor) Object.defineProperty(globalThis, 'GPUMapMode', mapModeDescriptor);
  else delete globalThis.GPUMapMode;
}

console.log('GPU timestamp range timer contracts passed');
