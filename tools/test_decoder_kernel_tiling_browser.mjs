#!/usr/bin/env node
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import {
  mkdtempSync,
  mkdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

import {
  retainNegotiatedWitnessIdentity,
  retainWitnessNavigationEvidence,
  runNegotiatedWitnessConformance,
  runWitnessAssertions,
  validateWitnessNavigation,
} from './decoder_kernel_witness_contract.mjs';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const REPO_ROOT = fileURLToPath(new URL('..', import.meta.url));
const WITNESS_ENTRY_POINT = 'sharp-webgpu-root-v1';

function gitOutput(args) {
  try {
    return execFileSync('git', args, {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
  } catch {
    return '';
  }
}

const expectedSourceRevision = gitOutput(['rev-parse', 'HEAD']);
const expectedSourceState = gitOutput(['status', '--porcelain']) === '' ? 'clean' : 'dirty';
const outputArgIndex = process.argv.indexOf('--output');
const reportPath = outputArgIndex >= 0
  ? (process.argv[outputArgIndex + 1] || '/tmp/sharp-decoder-kernel-timestamp-conformance-report.json')
  : '/tmp/sharp-decoder-kernel-timestamp-conformance-report.json';
const exerciseFailureArgIndex = process.argv.indexOf('--exercise-failure-phase');
const exerciseFailurePhase = exerciseFailureArgIndex >= 0
  ? process.argv[exerciseFailureArgIndex + 1]
  : null;
const positionalUrl = process.argv.slice(2).find(argument => !argument.startsWith('--') && argument !== reportPath && argument !== exerciseFailurePhase);
const requestedRoute = positionalUrl || 'http://127.0.0.1:5188/';
const requestedBrowser = {
  executablePath: CHROME_PATH,
  product: 'Google Chrome',
  channel: 'installed-stable',
};
const requestedFeatures = ['timestamp-query'];
const runId = `sharp-decoder-timestamp-${Date.now().toString(36)}-${process.pid}`;
const startedAt = new Date().toISOString();
let browser;
let page;
let userDataDir;
let effectiveBrowser = null;
let effectiveRoute = null;
let failurePhase = 'preflight';
let lastTrustworthyEvidence = {
  profileCreated: false,
  browserLaunched: false,
  routeLoaded: false,
  conformanceCompleted: false,
};

mkdirSync(dirname(reportPath), { recursive: true });
writeFileSync(reportPath, `${JSON.stringify({
  schema: 'sharp-webgpu.decoder-kernel-timestamp-conformance.v0',
  status: 'running',
  runId,
  startedAt,
  requestedRoute,
  effectiveRoute: null,
  expectedSourceRevision,
  expectedSourceState: 'clean',
  effectiveSourceRevision: null,
  effectiveSourceState: null,
  expectedEntryPoint: WITNESS_ENTRY_POINT,
  effectiveEntryPoint: null,
  requestedBrowser,
  effectiveBrowser,
  requestedFeatures,
  effectiveFeatures: null,
  currentPhase: failurePhase,
  lastTrustworthyEvidence,
}, null, 2)}\n`);

try {
  if (exerciseFailurePhase === 'preflight') {
    throw new Error('forced witness preflight failure');
  }
  if (exerciseFailurePhase === 'post-negotiation-assertion') {
    lastTrustworthyEvidence = retainNegotiatedWitnessIdentity(lastTrustworthyEvidence, {
      adapterInfo: {
        vendor: 'fixture',
        architecture: 'fixture',
        device: 'fixture',
        description: 'forced post-negotiation failure fixture',
        isFallbackAdapter: false,
      },
      adapterAdmission: {
        status: 'non-fallback-admitted',
        authority: 'webgpu-adapter-info-isFallbackAdapter',
      },
      effectiveFeatures: ['timestamp-query'],
    });
    failurePhase = 'conformance-assertion';
    throw new Error('forced post-negotiation assertion failure');
  }
  if (!expectedSourceRevision) {
    throw new Error('decoder witness could not resolve the expected Git revision');
  }
  if (expectedSourceState !== 'clean') {
    throw new Error('decoder witness requires a clean expected source worktree');
  }
  userDataDir = mkdtempSync(join(tmpdir(), 'sharp-decoder-timestamp-chrome-'));
  lastTrustworthyEvidence = { ...lastTrustworthyEvidence, profileCreated: true };
  failurePhase = 'browser-launch';
  browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    userDataDir,
    headless: false,
    args: [
      '--enable-unsafe-webgpu',
      '--disable-gpu-shader-disk-cache',
      '--use-mock-keychain',
      '--no-first-run',
      '--no-default-browser-check',
    ],
  });
  effectiveBrowser = {
    version: await browser.version(),
    profileAuthority: 'isolated-disposable-mock-keychain',
  };
  lastTrustworthyEvidence = { ...lastTrustworthyEvidence, browserLaunched: true };
  page = await browser.newPage();
  failurePhase = 'route-load';
  const navigationResponse = await page.goto(requestedRoute, { waitUntil: 'networkidle0', timeout: 30_000 });
  effectiveRoute = page.url();
  const responseHeaders = navigationResponse?.headers() || {};
  const navigationEvidence = {
    requestedRoute,
    effectiveRoute,
    status: navigationResponse?.status() ?? null,
    ok: navigationResponse?.ok() ?? false,
    expectedSourceRevision,
    effectiveSourceRevision: responseHeaders['x-sharp-source-revision'] || null,
    effectiveSourceState: responseHeaders['x-sharp-source-state'] || null,
    expectedEntryPoint: WITNESS_ENTRY_POINT,
    effectiveEntryPoint: responseHeaders['x-sharp-entrypoint'] || null,
  };
  lastTrustworthyEvidence = retainWitnessNavigationEvidence(
    lastTrustworthyEvidence,
    navigationEvidence,
  );
  const routeAdmission = validateWitnessNavigation(navigationEvidence);
  effectiveBrowser.userAgent = await page.evaluate(() => navigator.userAgent);
  lastTrustworthyEvidence = { ...lastTrustworthyEvidence, routeLoaded: true, routeAdmission };
  failurePhase = 'webgpu-timestamp-conformance';
  const result = await runNegotiatedWitnessConformance({
    negotiate: async () => page.evaluate(async () => {
      const {
        normalizeWitnessAdapterInfo,
        validateNativeWitnessAdapter,
      } = await import('/tools/decoder_kernel_witness_contract.mjs');
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('WebGPU adapter unavailable');
      const adapterInfo = normalizeWitnessAdapterInfo({
        vendor: adapter.info?.vendor,
        architecture: adapter.info?.architecture,
        device: adapter.info?.device,
        description: adapter.info?.description,
        isFallbackAdapter: adapter.info?.isFallbackAdapter,
      });
      const adapterAdmission = validateNativeWitnessAdapter(adapterInfo);
      if (!adapter.features.has('timestamp-query')) {
        throw new Error('adaptive browser witness requires timestamp-query adapter support');
      }
      const device = await adapter.requestDevice({
        requiredFeatures: ['timestamp-query'],
      });
      globalThis.__sharpDecoderTimestampWitnessV0 = { device };
      return {
        requestedFeatures: ['timestamp-query'],
        effectiveFeatures: Array.from(device.features).sort(),
        adapterInfo,
        adapterAdmission,
      };
    }),
    retainNegotiated: negotiatedIdentity => {
      lastTrustworthyEvidence = retainNegotiatedWitnessIdentity(
        lastTrustworthyEvidence,
        negotiatedIdentity,
      );
    },
    conform: async negotiatedIdentity => {
      const conformanceResult = await page.evaluate(async () => {
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
        const device = globalThis.__sharpDecoderTimestampWitnessV0?.device;
        if (!device) throw new Error('negotiated WebGPU device unavailable to conformance phase');

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
      return { ...negotiatedIdentity, ...conformanceResult };
    },
  });

  const { maxGroupNormDelta } = runWitnessAssertions({
    setFailurePhase: phase => {
      failurePhase = phase;
    },
    assertConformance: () => {
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
  let observedMaxGroupNormDelta = 0;
  for (let index = 0; index < result.serialGnValues.length; index += 1) {
    const actual = result.parallelGnValues[index];
    const expected = result.serialGnValues[index];
    assert.ok(Number.isFinite(actual), `parallel GroupNorm output ${index} must be finite`);
    observedMaxGroupNormDelta = Math.max(observedMaxGroupNormDelta, Math.abs(actual - expected));
  }
  assert.ok(observedMaxGroupNormDelta <= 2e-3, `parallel GroupNorm max absolute delta ${observedMaxGroupNormDelta} exceeds 2e-3`);
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
      return { maxGroupNormDelta: observedMaxGroupNormDelta };
    },
  });
  lastTrustworthyEvidence = {
    ...lastTrustworthyEvidence,
    conformanceCompleted: true,
  };
  failurePhase = 'report-write';
  const report = {
    schema: 'sharp-webgpu.decoder-kernel-timestamp-conformance.v0',
    status: 'passed',
    runId,
    startedAt,
    completedAt: new Date().toISOString(),
    requestedRoute,
    effectiveRoute,
    expectedSourceRevision,
    expectedSourceState: 'clean',
    effectiveSourceRevision: lastTrustworthyEvidence.navigationResponse?.effectiveSourceRevision || null,
    effectiveSourceState: lastTrustworthyEvidence.navigationResponse?.effectiveSourceState || null,
    expectedEntryPoint: WITNESS_ENTRY_POINT,
    effectiveEntryPoint: lastTrustworthyEvidence.navigationResponse?.effectiveEntryPoint || null,
    requestedBrowser,
    effectiveBrowser,
    requestedFeatures,
    effectiveFeatures: result.effectiveFeatures,
    adapterInfo: result.adapterInfo,
    adapterAdmission: result.adapterAdmission,
    routeAdmission: lastTrustworthyEvidence.routeAdmission,
    validationError: result.validationError,
    parity: {
      tiledConv2dBitExact: true,
      adaptiveConv2dBitExact: true,
      tiledConv1x1BitExact: true,
      tiledConvTranspose2dBitExact: true,
      parallelGroupNormMaxAbsoluteDelta: maxGroupNormDelta,
    },
    adaptiveConvPlanner: result.adaptiveConvPlanner,
    adaptiveBoundaries: result.adaptiveBoundaries,
    groupNormPartialPlanner: result.groupNormPartialPlanner,
    groupNormNormalizePlanner: result.groupNormNormalizePlanner,
    groupNormBoundaries: result.groupNormBoundaries,
  };
  writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(`decoder kernel tiling browser parity passed (parallel GroupNorm max delta ${maxGroupNormDelta})`);
  console.log(`Report: ${reportPath}`);
} catch (error) {
  writeFileSync(reportPath, `${JSON.stringify({
    schema: 'sharp-webgpu.decoder-kernel-timestamp-conformance.v0',
    status: 'failed',
    runId,
    startedAt,
    completedAt: new Date().toISOString(),
    requestedRoute,
    effectiveRoute,
    expectedSourceRevision,
    expectedSourceState: 'clean',
    effectiveSourceRevision: lastTrustworthyEvidence.navigationResponse?.effectiveSourceRevision || null,
    effectiveSourceState: lastTrustworthyEvidence.navigationResponse?.effectiveSourceState || null,
    expectedEntryPoint: WITNESS_ENTRY_POINT,
    effectiveEntryPoint: lastTrustworthyEvidence.navigationResponse?.effectiveEntryPoint || null,
    requestedBrowser,
    effectiveBrowser,
    requestedFeatures,
    effectiveFeatures: lastTrustworthyEvidence.effectiveFeatures || null,
    failurePhase,
    lastTrustworthyEvidence,
    error: {
      name: error?.name || 'Error',
      message: error?.message || String(error),
      stack: error?.stack || null,
    },
  }, null, 2)}\n`);
  throw error;
} finally {
  if (browser) await browser.close();
  if (userDataDir) rmSync(userDataDir, { recursive: true, force: true });
}
