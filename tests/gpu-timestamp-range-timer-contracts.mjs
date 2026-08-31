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

  let mappedBytes = timestampBytes(1_000_000n, 4_500_000n);
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
    writeTimestamp(observedQuerySet, index) {
      assert.equal(observedQuerySet, querySet);
      commands.push(`timestamp:${index}`);
    },
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

  const timer = createGpuTimestampRangeTimer(device, { label: 'contract-range' });
  timer.begin(encoder);
  commands.push('decoder-commands');
  timer.end(encoder);
  assert.deepEqual(commands, [
    'timestamp:0',
    'decoder-commands',
    'timestamp:1',
    'resolve:0:2:0',
    'copy:0:0:16',
  ]);
  const measurement = await timer.read();
  assert.equal(measurement.authority, 'timestamp-query-inside-submitted-command-buffer');
  assert.equal(measurement.durationNs, '3500000');
  assert.equal(measurement.durationMs, 3.5);
  assert.equal(mapCount, 1);
  assert.equal(unmapCount, 1);

  mappedBytes = timestampBytes(8_000_000n, 8_000_000n);
  timer.begin(encoder);
  timer.end(encoder);
  await assert.rejects(
    () => timer.read(),
    /strictly positive ordered duration/,
    'quantized or coalesced zero timestamp ranges cannot become adaptive evidence',
  );
  assert.equal(unmapCount, 2);

  timer.destroy();
  assert.deepEqual(destroyed.sort(), ['query-set', 'read-buffer', 'resolve-buffer']);
  timer.destroy();

  assert.throws(
    () => createGpuTimestampRangeTimer({ features: new Set() }),
    /requires the timestamp-query device feature/,
    'adaptive timing fails loud rather than falling back to host callback timing',
  );

  const decoderSource = readFileSync(new URL('../src/lib/decoder_duties.js', import.meta.url), 'utf8');
  assert.match(
    decoderSource,
    /gpuTimestampTimer\?\.begin\(encoder\)[\s\S]{0,300}encodeTile\([\s\S]{0,700}gpuTimestampTimer\?\.end\(encoder\)/,
    'GPU timestamps must bracket only the decoder commands inside the submitted command buffer',
  );
  assert.match(
    decoderSource,
    /gpuTimestampTimer\.read\(\)[\s\S]{0,200}adaptiveDecoderTimingObservation\(yieldReceipt,\s*gpuTimestampRange\)/,
    'adaptive planning must consume the resolved GPU timestamp measurement',
  );
} finally {
  if (usageDescriptor) Object.defineProperty(globalThis, 'GPUBufferUsage', usageDescriptor);
  else delete globalThis.GPUBufferUsage;
  if (mapModeDescriptor) Object.defineProperty(globalThis, 'GPUMapMode', mapModeDescriptor);
  else delete globalThis.GPUMapMode;
}

console.log('GPU timestamp range timer contracts passed');
