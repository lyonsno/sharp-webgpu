import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

globalThis.GPUBufferUsage = {
  STORAGE: 1,
  COPY_SRC: 2,
  COPY_DST: 4,
};

const weightsModule = await import('../src/lib/weights.js');
const schedulerModule = await import('../src/lib/scheduler.js');
const mainSource = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');

assert.equal(
  typeof weightsModule.extractTensorCooperatively,
  'function',
  'weight loading must expose bounded asynchronous GPU materialization',
);

const MiB = 1024 * 1024;

function createFakeDevice() {
  const writes = [];
  const buffers = [];
  return {
    writes,
    buffers,
    device: {
      createBuffer(descriptor) {
        const bytes = new Uint8Array(descriptor.size);
        const buffer = {
          bytes,
          descriptor,
          destroyed: false,
          mapped: descriptor.mappedAtCreation,
          unmapCount: 0,
          getMappedRange() {
            if (!this.mapped) throw new Error('buffer is not mapped');
            return this.bytes.buffer;
          },
          unmap() {
            if (!this.mapped) throw new Error('buffer is not mapped');
            this.mapped = false;
            this.unmapCount += 1;
          },
          destroy() {
            this.destroyed = true;
            this.mapped = false;
          },
        };
        buffers.push(buffer);
        return buffer;
      },
      queue: {
        writeBuffer(buffer, bufferOffset, data) {
          const source = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
          buffer.bytes.set(source, bufferOffset);
          writes.push({
            bufferOffset,
            byteLength: source.byteLength,
          });
        },
      },
    },
  };
}

const fp16Source = new Uint16Array([
  0x0000,
  0x3c00,
  0xc000,
  0x3800,
  0x4000,
  0x4200,
  0x4400,
  0x4500,
  0x4600,
  0x4700,
]);
const fp16Fixture = fp16Source.buffer.slice(0);
const fp16Device = createFakeDevice();
const fp16Donations = [];
const fp16Buffer = await weightsModule.extractTensorCooperatively(
  fp16Device.device,
  fp16Fixture,
  {
    dtype: 1,
    offset: 0,
    size: fp16Fixture.byteLength,
  },
  {
    tensorName: 'encoder.blocks.0.attn.qkv.weight',
    maxWriteBytes: 16,
    yieldControl: async receipt => {
      fp16Donations.push(receipt);
    },
  },
);

assert.deepEqual(
  {
    mappedAtCreation: fp16Buffer.descriptor.mappedAtCreation,
    usage: fp16Buffer.descriptor.usage,
    queueWriteCount: fp16Device.writes.length,
    unmapCount: fp16Buffer.unmapCount,
  },
  {
    mappedAtCreation: true,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    queueWriteCount: 0,
    unmapCount: 1,
  },
  'cooperative conversion must preserve the fast mapped storage-buffer representation',
);
assert.equal(fp16Donations.length, 2, 'every non-terminal conversion chunk must donate');
assert.deepEqual(
  fp16Donations.map(receipt => [
    receipt.tensorName,
    receipt.completedBytes,
    receipt.totalBytes,
    receipt.chunkIndex,
    receipt.totalChunks,
  ]),
  [
    ['encoder.blocks.0.attn.qkv.weight', 16, 40, 1, 3],
    ['encoder.blocks.0.attn.qkv.weight', 32, 40, 2, 3],
  ],
  'chunk donation receipts must expose exact tensor and total progress identity',
);
assert.deepEqual(
  [...new Float32Array(fp16Buffer.bytes.buffer)],
  [0, 1, -2, 0.5, 2, 3, 4, 5, 6, 7],
  'cooperative fp16 conversion must preserve exact fp32 values',
);

let queueDrainCount = 0;
const noDrainScheduler = schedulerModule.parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    waitForSubmittedWorkDone: true,
    yieldMs: 0,
  },
});
const noDrainTelemetry = schedulerModule.createSharpRunTelemetry(noDrainScheduler, {
  runId: 'weight-upload-no-drain',
});
const noDrainReceipt = await schedulerModule.schedulerYield(
  noDrainScheduler,
  {
    queue: {
      async onSubmittedWorkDone() {
        queueDrainCount += 1;
      },
    },
  },
  noDrainTelemetry,
  'weights-load',
  {
    stage: 'route-setup',
    step: 'tensor-upload-chunk',
    role: 'cooperative-weight-materialization',
  },
  0,
  null,
  { waitForSubmittedWorkDone: false },
);

const defaultChunkOutputBytes = 10 * MiB;
const defaultChunkSource = new Uint16Array(defaultChunkOutputBytes / Float32Array.BYTES_PER_ELEMENT);
const defaultChunkDevice = createFakeDevice();
const defaultChunkDonations = [];
await weightsModule.extractTensorCooperatively(
  defaultChunkDevice.device,
  defaultChunkSource.buffer,
  {
    dtype: 1,
    offset: 0,
    size: defaultChunkSource.byteLength,
  },
  {
    tensorName: 'encoder.blocks.0.mlp.fc1.weight',
    yieldControl: async receipt => {
      defaultChunkDonations.push(receipt);
    },
  },
);

assert.deepEqual(
  {
    queueDrainCount,
    queueWriteCount: defaultChunkDevice.writes.length,
    mappedAtCreation: defaultChunkDevice.buffers[0].descriptor.mappedAtCreation,
    unmapCount: defaultChunkDevice.buffers[0].unmapCount,
    donationCount: defaultChunkDonations.length,
  },
  {
    queueDrainCount: 0,
    queueWriteCount: 0,
    mappedAtCreation: true,
    unmapCount: 1,
    donationCount: 2,
  },
  'intra-tensor conversion must donate at 4 MiB boundaries without queue writes or queue drains',
);
assert.equal(
  noDrainReceipt.waitedForSubmittedWorkDone,
  false,
  'the explicit no-drain boundary must report that it did not settle the queue',
);
assert.equal(
  noDrainTelemetry.events.some(event => event.kind === 'queue-work-done-start'),
  false,
  'the explicit no-drain boundary must not emit queue-settlement evidence',
);
assert.match(
  mainSource,
  /details\.step === 'tensor-upload-chunk'\s*\?\s*\{\s*waitForSubmittedWorkDone:\s*false\s*\}/,
  'the SHARP product route must skip queue settlement only for intra-tensor upload donations',
);

const fp32Source = new Float32Array([1.25, -2.5, 3.75, 5]);
const fp32Device = createFakeDevice();
const fp32Buffer = await weightsModule.extractTensorCooperatively(
  fp32Device.device,
  fp32Source.buffer,
  {
    dtype: 0,
    offset: 0,
    size: fp32Source.byteLength,
  },
  {
    tensorName: 'encoder.norm.weight',
    maxWriteBytes: 8,
    yieldControl: async () => {},
  },
);
assert.deepEqual(
  [...new Float32Array(fp32Buffer.bytes.buffer)],
  [...fp32Source],
  'cooperative fp32 upload must preserve source bytes',
);

const failedDevice = createFakeDevice();
let failedDonationCount = 0;
await assert.rejects(
  weightsModule.extractTensorCooperatively(
    failedDevice.device,
    fp16Fixture,
    { dtype: 1, offset: 0, size: fp16Fixture.byteLength },
    {
      tensorName: 'failed.weight',
      maxWriteBytes: 16,
      yieldControl: async () => {
        failedDonationCount += 1;
        throw new Error('planted donation failure');
      },
    },
  ),
  /planted donation failure/,
);
assert.equal(failedDonationCount, 1);
assert.equal(
  failedDevice.buffers[0].destroyed,
  true,
  'failed cooperative materialization must destroy its partial GPU allocation',
);

let extractionCount = 0;
let releaseExtraction;
const accessors = weightsModule.createWeightTensorAccessors(
  {},
  new ArrayBuffer(4),
  new Map([['shared.weight', { dtype: 0, offset: 0, size: 4 }]]),
  {
    extractGpuTensor: () => ({ mode: 'sync' }),
    async extractGpuTensorCooperatively() {
      extractionCount += 1;
      await new Promise(resolve => {
        releaseExtraction = resolve;
      });
      return { mode: 'cooperative' };
    },
  },
);

assert.equal(typeof accessors.materialize, 'function');
assert.equal(typeof accessors.tryMaterialize, 'function');
const firstMaterialization = accessors.materialize('shared.weight');
const secondMaterialization = accessors.materialize('shared.weight');
assert.equal(extractionCount, 1, 'concurrent materialization must share one in-flight extraction');
releaseExtraction();
const [firstResult, secondResult] = await Promise.all([
  firstMaterialization,
  secondMaterialization,
]);
assert.equal(firstResult, secondResult, 'concurrent materialization must preserve GPU object identity');
assert.equal(accessors.get('shared.weight'), firstResult);
assert.equal(await accessors.tryMaterialize('missing.weight'), null);

console.log('cooperative weight materialization contracts passed');
