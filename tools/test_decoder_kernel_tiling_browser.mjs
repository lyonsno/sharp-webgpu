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
    const { dispatchConv2d, dispatchConvTranspose2d } = await import('/src/lib/shader_ops.js');
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const device = await adapter.requestDevice();

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
      const values = Array.from(new Uint32Array(staging.getMappedRange().slice(0)));
      staging.unmap();
      staging.destroy();
      return values;
    };

    device.pushErrorScope('validation');
    const input = upload(Array.from({ length: 32 }, (_, index) => ((index % 11) - 5) / 7));
    const convWeights = upload(Array.from({ length: 54 }, (_, index) => ((index % 13) - 6) / 17));
    const convBias = upload([0.25, -0.5, 0.75]);
    const convParams = {
      inC: 2, inH: 4, inW: 4, outC: 3,
      kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1,
    };

    const fullEncoder = device.createCommandEncoder();
    const fullConv = dispatchConv2d(device, fullEncoder, input, convWeights, convBias, convParams);
    device.queue.submit([fullEncoder.finish()]);

    const firstConvEncoder = device.createCommandEncoder();
    const tiledConv = dispatchConv2d(device, firstConvEncoder, input, convWeights, convBias, {
      ...convParams,
      outputStart: 0,
      outputCount: 17,
    });
    device.queue.submit([firstConvEncoder.finish()]);
    const secondConvEncoder = device.createCommandEncoder();
    dispatchConv2d(device, secondConvEncoder, input, convWeights, convBias, {
      ...convParams,
      outputStart: 17,
      outputCount: 31,
      outputBuffer: tiledConv.buffer,
    });
    device.queue.submit([secondConvEncoder.finish()]);

    const deconvWeights = upload(Array.from({ length: 24 }, (_, index) => ((index % 9) - 4) / 11));
    const deconvBias = upload([0.125, -0.25, 0.5]);
    const deconvParams = { inC: 2, inH: 2, inW: 2, outC: 3, stride: 2 };
    const deconvInput = upload(Array.from({ length: 8 }, (_, index) => (index - 3) / 5));

    const fullDeconvEncoder = device.createCommandEncoder();
    const fullDeconv = dispatchConvTranspose2d(device, fullDeconvEncoder, deconvInput, deconvWeights, deconvBias, deconvParams);
    device.queue.submit([fullDeconvEncoder.finish()]);
    const firstDeconvEncoder = device.createCommandEncoder();
    const tiledDeconv = dispatchConvTranspose2d(device, firstDeconvEncoder, deconvInput, deconvWeights, deconvBias, {
      ...deconvParams,
      outputStart: 0,
      outputCount: 19,
    });
    device.queue.submit([firstDeconvEncoder.finish()]);
    const secondDeconvEncoder = device.createCommandEncoder();
    dispatchConvTranspose2d(device, secondDeconvEncoder, deconvInput, deconvWeights, deconvBias, {
      ...deconvParams,
      outputStart: 19,
      outputCount: 29,
      outputBuffer: tiledDeconv.buffer,
    });
    device.queue.submit([secondDeconvEncoder.finish()]);

    await device.queue.onSubmittedWorkDone();
    const [fullConvBits, tiledConvBits, fullDeconvBits, tiledDeconvBits] = await Promise.all([
      read(fullConv.buffer, 48),
      read(tiledConv.buffer, 48),
      read(fullDeconv.buffer, 48),
      read(tiledDeconv.buffer, 48),
    ]);
    const validationError = await device.popErrorScope();
    return {
      validationError: validationError?.message || null,
      fullConvBits,
      tiledConvBits,
      fullDeconvBits,
      tiledDeconvBits,
    };
  });

  assert.equal(result.validationError, null, `WebGPU validation failed: ${result.validationError}`);
  assert.deepEqual(result.tiledConvBits, result.fullConvBits, 'tiled Conv2d must be bit-identical to the original full dispatch');
  assert.deepEqual(result.tiledDeconvBits, result.fullDeconvBits, 'tiled ConvTranspose2d must be bit-identical to the original full dispatch');
  console.log('decoder kernel tiling browser parity passed');
} finally {
  await browser.close();
}
