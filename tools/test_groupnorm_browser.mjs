#!/usr/bin/env node
import assert from 'node:assert/strict';
import puppeteer from 'puppeteer-core';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const url = process.argv[2] || 'http://127.0.0.1:5188/';

const browser = await puppeteer.launch({
  executablePath: CHROME_PATH,
  headless: true,
  args: [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--disable-gpu-sandbox',
    '--no-sandbox',
    '--disable-gpu-shader-disk-cache',
  ],
});

try {
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: 'networkidle0' });
  const result = await page.evaluate(async () => {
    const { dispatchGroupNorm } = await import('/src/lib/shader_ops.js');
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const device = await adapter.requestDevice();

    const C = 4;
    const H = 16;
    const W = 16;
    const numGroups = 2;
    const eps = 1e-5;
    const inputValues = Array.from(
      { length: C * H * W },
      (_, index) => Math.sin(index * 0.17) * 2.5 + (index % 7) * 0.125,
    );
    const scaleValues = [0.75, -1.25, 1.5, 0.5];
    const biasValues = [0.1, -0.2, 0.3, -0.4];

    const upload = values => {
      const data = new Float32Array(values);
      const buffer = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const read = async (buffer, count) => {
      const staging = device.createBuffer({
        size: count * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, staging, 0, count * 4);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const values = Array.from(new Float32Array(staging.getMappedRange().slice(0)));
      staging.unmap();
      staging.destroy();
      return values;
    };

    device.pushErrorScope('validation');
    const encoder = device.createCommandEncoder();
    const output = dispatchGroupNorm(
      device,
      encoder,
      upload(inputValues),
      upload(scaleValues),
      upload(biasValues),
      { C, H, W, numGroups, eps },
    );
    device.queue.submit([encoder.finish()]);
    const actual = await read(output, inputValues.length);
    const validationError = await device.popErrorScope();

    const expected = new Array(inputValues.length);
    const channelsPerGroup = C / numGroups;
    const spatialSize = H * W;
    for (let group = 0; group < numGroups; group += 1) {
      const start = group * channelsPerGroup * spatialSize;
      const end = start + channelsPerGroup * spatialSize;
      const values = inputValues.slice(start, end).map(Math.fround);
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
      const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
      for (let index = start; index < end; index += 1) {
        const channel = Math.floor(index / spatialSize);
        expected[index] = ((Math.fround(inputValues[index]) - mean) / Math.sqrt(variance + eps))
          * scaleValues[channel] + biasValues[channel];
      }
    }
    return {
      backend: 'webgpu',
      validationError: validationError?.message || null,
      actual,
      expected,
    };
  });

  assert.equal(result.backend, 'webgpu', 'GroupNorm parity must use the requested WebGPU backend');
  assert.equal(result.validationError, null, `WebGPU validation failed: ${result.validationError}`);
  let maxAbsoluteError = 0;
  for (let index = 0; index < result.expected.length; index += 1) {
    maxAbsoluteError = Math.max(
      maxAbsoluteError,
      Math.abs(result.actual[index] - result.expected[index]),
    );
  }
  assert.ok(maxAbsoluteError <= 2e-5, `parallel GroupNorm max absolute error ${maxAbsoluteError} exceeded 2e-5`);
  console.log(`parallel GroupNorm browser parity passed (max abs error ${maxAbsoluteError})`);
} finally {
  await browser.close();
}
