import assert from 'node:assert/strict';

globalThis.GPUBufferUsage = {
  STORAGE: 1,
  COPY_SRC: 2,
  COPY_DST: 4,
};

const weightsModule = await import('../src/lib/weights.js');

assert.equal(
  typeof weightsModule.extractTensorCooperatively,
  'function',
  'weight loading must expose bounded asynchronous GPU materialization',
);

function createFakeDevice({ failWriteAt = null } = {}) {
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
          destroy() {
            this.destroyed = true;
          },
        };
        buffers.push(buffer);
        return buffer;
      },
      queue: {
        writeBuffer(buffer, bufferOffset, data) {
          if (failWriteAt !== null && writes.length === failWriteAt) {
            throw new Error('planted queue write failure');
          }
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
  fp16Device.writes.map(write => write.byteLength),
  [16, 16, 8],
  'one large fp16 tensor must be uploaded through bounded writes',
);
assert.equal(fp16Donations.length, 2, 'every non-terminal upload chunk must donate');
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

const failedDevice = createFakeDevice({ failWriteAt: 1 });
await assert.rejects(
  weightsModule.extractTensorCooperatively(
    failedDevice.device,
    fp16Fixture,
    { dtype: 1, offset: 0, size: fp16Fixture.byteLength },
    {
      tensorName: 'failed.weight',
      maxWriteBytes: 16,
      yieldControl: async () => {},
    },
  ),
  /planted queue write failure/,
);
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
