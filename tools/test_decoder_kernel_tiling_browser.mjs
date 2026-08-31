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
    const {
      dispatchActivation,
      dispatchConv1x1,
      dispatchConv2d,
      dispatchConvTranspose2d,
      dispatchGroupNorm,
    } = await import('/src/lib/shader_ops.js');
    const {
      createDecoderAdaptiveDuty,
      dispatchTiledConv2d,
      dispatchTiledGroupNormRelu,
    } = await import('/src/lib/decoder_duties.js');
    const {
      createSharpRunTelemetry,
      parseSharpSchedulerConfig,
      schedulerYield,
    } = await import('/src/lib/scheduler.js');
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    if (!adapter.features.has('timestamp-query')) {
      throw new Error('adaptive browser witness requires timestamp-query adapter support');
    }
    const device = await adapter.requestDevice({
      requiredFeatures: ['timestamp-query'],
    });

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
    const readFloats = async (buffer, count) => {
      const bits = await read(buffer, count);
      return Array.from(new Float32Array(new Uint32Array(bits).buffer));
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

    const pointWeights = upload(Array.from({ length: 6 }, (_, index) => ((index % 5) - 2) / 9));
    const pointBias = upload([0.125, -0.25, 0.5]);
    const pointParams = { inC: 2, outC: 3, H: 4, W: 4 };
    const fullPointEncoder = device.createCommandEncoder();
    const fullPoint = dispatchConv1x1(device, fullPointEncoder, input, pointWeights, pointBias, pointParams);
    device.queue.submit([fullPointEncoder.finish()]);
    const firstPointEncoder = device.createCommandEncoder();
    const tiledPoint = dispatchConv1x1(device, firstPointEncoder, input, pointWeights, pointBias, {
      ...pointParams,
      outputStart: 0,
      outputCount: 17,
    });
    device.queue.submit([firstPointEncoder.finish()]);
    const secondPointEncoder = device.createCommandEncoder();
    dispatchConv1x1(device, secondPointEncoder, input, pointWeights, pointBias, {
      ...pointParams,
      outputStart: 17,
      outputCount: 31,
      outputBuffer: tiledPoint.buffer,
    });
    device.queue.submit([secondPointEncoder.finish()]);

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
    const adaptiveScheduler = parseSharpSchedulerConfig({
      sharpScheduler: {
        decoderKernelChunkItems: 17,
        decoderKernelMinChunkItems: 5,
        decoderKernelMaxChunkItems: 48,
        decoderKernelTargetDurationMs: 1000,
        waitForSubmittedWorkDone: true,
        yieldMs: 0,
      },
    });
    const adaptiveTelemetry = createSharpRunTelemetry(adaptiveScheduler, { runId: 'browser-adaptive-conv' });
    const adaptiveDuty = createDecoderAdaptiveDuty(adaptiveScheduler, adaptiveTelemetry, 'browser-conv');
    const adaptiveBoundaries = [];
    const adaptiveConv = await dispatchTiledConv2d({
      device,
      inputBuf: input,
      weightBuf: convWeights,
      biasBuf: convBias,
      params: convParams,
      chunkItems: adaptiveScheduler.effective.decoderKernelChunkItems,
      adaptiveDuty,
      phase: 'browser-adaptive-conv',
      details: { fixture: 'adaptive-conv2d' },
      boundaryYield: async (phase, details, queueCompletionFence) => {
        const receipt = await schedulerYield(
          adaptiveScheduler,
          device,
          adaptiveTelemetry,
          'gaussian-phase',
          { phase, ...details },
          null,
          queueCompletionFence,
        );
        adaptiveBoundaries.push({ phase, ...details, receipt });
        return receipt;
      },
    });

    const gnC = 4;
    const gnH = 48;
    const gnW = 48;
    const gnCount = gnC * gnH * gnW;
    const gnInput = upload(Array.from({ length: gnCount }, (_, index) => 10000 + (((index * 17) % 101 - 50) / 23)));
    const gnScale = upload([0.75, 1.25, -0.5, 0.875]);
    const gnBias = upload([0.1, -0.2, 0.3, -0.4]);
    const serialGnEncoder = device.createCommandEncoder();
    const serialGn = dispatchGroupNorm(device, serialGnEncoder, gnInput, gnScale, gnBias, {
      C: gnC, H: gnH, W: gnW, numGroups: 2, eps: 1e-5,
    });
    const serialGnRelu = dispatchActivation(device, serialGnEncoder, serialGn, null, gnCount, 0);
    device.queue.submit([serialGnEncoder.finish()]);

    const groupNormScheduler = parseSharpSchedulerConfig({
      sharpScheduler: {
        decoderKernelChunkItems: 4096,
        decoderKernelMinChunkItems: 4096,
        decoderKernelMaxChunkItems: 8192,
        decoderKernelTargetDurationMs: 1000,
        waitForSubmittedWorkDone: true,
        yieldMs: 0,
      },
    });
    const groupNormTelemetry = createSharpRunTelemetry(groupNormScheduler, { runId: 'browser-adaptive-groupnorm' });
    const groupNormAdaptiveDuty = createDecoderAdaptiveDuty(groupNormScheduler, groupNormTelemetry, 'browser-groupnorm');
    const groupNormBoundaries = [];
    const parallelGn = await dispatchTiledGroupNormRelu({
      device,
      inputBuf: gnInput,
      scaleBuf: gnScale,
      biasBuf: gnBias,
      params: { C: gnC, H: gnH, W: gnW, numGroups: 2, eps: 1e-5 },
      chunkItems: 4096,
      adaptiveDuty: groupNormAdaptiveDuty,
      phase: 'browser-gn',
      details: { fixture: 'parallel-groupnorm' },
      boundaryYield: async (phase, details, queueCompletionFence) => {
        const receipt = await schedulerYield(
          groupNormScheduler,
          device,
          groupNormTelemetry,
          'gaussian-phase',
          { phase, ...details },
          null,
          queueCompletionFence,
        );
        groupNormBoundaries.push({ phase, ...details, receipt });
        return receipt;
      },
    });

    await device.queue.onSubmittedWorkDone();
    const [fullConvBits, tiledConvBits, adaptiveConvBits, fullPointBits, tiledPointBits, fullDeconvBits, tiledDeconvBits, serialGnValues, parallelGnValues] = await Promise.all([
      read(fullConv.buffer, 48),
      read(tiledConv.buffer, 48),
      read(adaptiveConv.buffer, 48),
      read(fullPoint.buffer, 48),
      read(tiledPoint.buffer, 48),
      read(fullDeconv.buffer, 48),
      read(tiledDeconv.buffer, 48),
      readFloats(serialGnRelu, gnCount),
      readFloats(parallelGn.buffer, gnCount),
    ]);
    const validationError = await device.popErrorScope();
    return {
      requestedFeatures: ['timestamp-query'],
      effectiveFeatures: Array.from(device.features).sort(),
      validationError: validationError?.message || null,
      fullConvBits,
      tiledConvBits,
      adaptiveConvBits,
      fullPointBits,
      tiledPointBits,
      fullDeconvBits,
      tiledDeconvBits,
      serialGnValues,
      parallelGnValues,
      adaptiveConvPlanner: adaptiveConv.adaptivePlanner,
      adaptiveBoundaries,
      groupNormPartialPlanner: parallelGn.partialAdaptivePlanner,
      groupNormNormalizePlanner: parallelGn.normalizeAdaptivePlanner,
      groupNormBoundaries,
    };
  });

  assert.deepEqual(result.requestedFeatures, ['timestamp-query']);
  assert.ok(result.effectiveFeatures.includes('timestamp-query'));
  assert.equal(result.validationError, null, `WebGPU validation failed: ${result.validationError}`);
  assert.deepEqual(result.tiledConvBits, result.fullConvBits, 'tiled Conv2d must be bit-identical to the original full dispatch');
  assert.deepEqual(result.adaptiveConvBits, result.fullConvBits, 'adaptive Conv2d must be bit-identical to the original full dispatch');
  assert.equal(result.adaptiveConvPlanner.status, 'complete');
  assert.equal(result.adaptiveConvPlanner.actualRangeCount, 2, 'under-target Conv2d must grow from 17 items and finish in one exact final range');
  assert.deepEqual(
    result.adaptiveConvPlanner.ranges.map(range => [range.itemStart, range.itemEnd]),
    [[0, 17], [17, 48]],
    'adaptive Conv2d ranges must cover the output exactly once',
  );
  assert.ok(result.adaptiveBoundaries.every(event => event.rangeTotal === null), 'live adaptive boundaries must not project the final range total');
  assert.ok(result.adaptiveBoundaries.every(event => event.receipt.timingAuthority === 'queue-work-done'));
  assert.ok(result.adaptiveBoundaries.every(event => event.receipt.queueWorkAttribution === 'paired-host-fence-settlement'));
  assert.ok(result.adaptiveConvPlanner.ranges.every(range => range.timingAuthority === 'gpu-timestamp-query'));
  assert.deepEqual(result.tiledPointBits, result.fullPointBits, 'tiled Conv1x1 must be bit-identical to the original full dispatch');
  assert.deepEqual(result.tiledDeconvBits, result.fullDeconvBits, 'tiled ConvTranspose2d must be bit-identical to the original full dispatch');
  let maxGroupNormDelta = 0;
  for (let index = 0; index < result.serialGnValues.length; index += 1) {
    const actual = result.parallelGnValues[index];
    const expected = result.serialGnValues[index];
    assert.ok(Number.isFinite(actual), `parallel GroupNorm output ${index} must be finite`);
    maxGroupNormDelta = Math.max(maxGroupNormDelta, Math.abs(actual - expected));
  }
  assert.ok(maxGroupNormDelta <= 2e-3, `parallel GroupNorm max absolute delta ${maxGroupNormDelta} exceeds 2e-3`);
  assert.equal(result.groupNormPartialPlanner.status, 'complete');
  assert.equal(result.groupNormNormalizePlanner.status, 'complete');
  assert.ok(result.groupNormPartialPlanner.actualRangeCount >= 2, 'adaptive GroupNorm partial statistics must expose multiple exact ranges');
  assert.ok(result.groupNormNormalizePlanner.actualRangeCount >= 2, 'adaptive GroupNorm normalization must expose multiple exact ranges');
  assert.ok(result.groupNormBoundaries.every(event => event.receipt.timingAuthority === 'queue-work-done'));
  assert.ok(
    result.groupNormBoundaries
      .filter(event => event.role !== 'groupnorm-stats-reduction')
      .every(event => event.receipt.queueWorkAttribution === 'paired-host-fence-settlement'),
    'adaptive GroupNorm tile boundaries must retain paired host-fence diagnostics',
  );
  assert.ok(
    result.groupNormBoundaries
      .filter(event => event.role === 'groupnorm-stats-reduction')
      .every(event => event.receipt.queueWorkAttribution === 'submitted-range-plus-shared-queue-work'),
    'the fixed GroupNorm reduction boundary must retain ordinary shared-queue attribution',
  );
  assert.ok(result.groupNormPartialPlanner.ranges.every(range => range.timingAuthority === 'gpu-timestamp-query'));
  assert.ok(result.groupNormNormalizePlanner.ranges.every(range => range.timingAuthority === 'gpu-timestamp-query'));
  assert.ok(result.groupNormBoundaries.filter(event => event.role === 'groupnorm-partial-stats-tile').length > 1, 'GroupNorm fixture must submit multiple partial-statistics duties');
  assert.equal(result.groupNormBoundaries.filter(event => event.role === 'groupnorm-stats-reduction').length, 1, 'GroupNorm fixture must submit one bounded statistics reduction');
  assert.ok(result.groupNormBoundaries.filter(event => event.role === 'groupnorm-normalize-relu-tile').length > 1, 'GroupNorm fixture must submit multiple normalization duties');
  console.log(`decoder kernel tiling browser parity passed (parallel GroupNorm max delta ${maxGroupNormDelta})`);
} finally {
  await browser.close();
}
