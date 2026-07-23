#!/usr/bin/env node

import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { createServer } from 'node:net';

import puppeteer from 'puppeteer-core';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const outputArgIndex = process.argv.indexOf('--output');
const reportPath = outputArgIndex >= 0
  ? (process.argv[outputArgIndex + 1] || '/tmp/sharp-vit-four-stage-parity-report.json')
  : '/tmp/sharp-vit-four-stage-parity-report.json';
const exerciseFailureArgIndex = process.argv.indexOf('--exercise-failure-phase');
const exerciseFailurePhase = exerciseFailureArgIndex >= 0
  ? process.argv[exerciseFailureArgIndex + 1]
  : null;

async function reservePort() {
  const server = createServer();
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const { port } = server.address();
  await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  return port;
}

async function waitForServer(url, child, logs) {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(`Vite exited with ${child.exitCode}:\n${logs.join('')}`);
    }
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // Vite is still binding.
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`Vite did not answer at ${url}:\n${logs.join('')}`);
}

const runId = `sharp-vit-four-stage-parity-${Date.now().toString(36)}-${process.pid}`;
const startedAt = new Date().toISOString();
const serverLogs = [];
let url = null;
let vite;
let browser;
let page;
let failurePhase = 'port-reserve';
let lastTrustworthyEvidence = {
  portReserved: false,
  viteStarted: false,
  browserLaunched: false,
  routeLoaded: false,
  parityCompleted: false,
};
writeFileSync(reportPath, `${JSON.stringify({
  schema: 'sharp-webgpu.vit-four-stage-production-parity.v0',
  status: 'running',
  runId,
  startedAt,
  requestedRoute: null,
  effectiveRoute: null,
  currentPhase: failurePhase,
  lastTrustworthyEvidence,
}, null, 2)}\n`);
try {
  if (exerciseFailurePhase === 'port-reserve') {
    throw new Error('forced port reservation failure');
  }
  const port = await reservePort();
  url = `http://127.0.0.1:${port}/`;
  lastTrustworthyEvidence = { ...lastTrustworthyEvidence, portReserved: true };
  failurePhase = 'vite-start';
  vite = spawn(
    process.platform === 'win32' ? 'npm.cmd' : 'npm',
    ['run', 'dev', '--', '--host', '127.0.0.1', '--port', String(port), '--strictPort'],
    { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'] },
  );
  vite.stdout.on('data', chunk => serverLogs.push(chunk.toString()));
  vite.stderr.on('data', chunk => serverLogs.push(chunk.toString()));
  lastTrustworthyEvidence = { ...lastTrustworthyEvidence, viteStarted: true };
  await waitForServer(url, vite, serverLogs);
  lastTrustworthyEvidence = { ...lastTrustworthyEvidence, viteAnswered: true };
  failurePhase = 'browser-launch';
  browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    protocolTimeout: 600_000,
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--disable-gpu-sandbox',
      '--no-sandbox',
      '--disable-gpu-shader-disk-cache',
    ],
  });
  lastTrustworthyEvidence = { ...lastTrustworthyEvidence, browserLaunched: true };
  page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  failurePhase = 'route-load';
  await page.goto(url, { waitUntil: 'networkidle0', timeout: 30_000 });
  lastTrustworthyEvidence = {
    ...lastTrustworthyEvidence,
    routeLoaded: true,
    effectiveRoute: page.url(),
  };

  failurePhase = 'production-webgpu-parity';
  const result = await page.evaluate(async () => {
    const { ViTEncoder, VIT_CONFIG } = await import('/src/lib/backbone.js');
    const { createStorageBuffer, readBuffer } = await import('/src/lib/gpu.js');
    const {
      createSharpRunTelemetry,
      parseSharpSchedulerConfig,
    } = await import('/src/lib/scheduler.js');

    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });
    const validationErrors = [];
    device.addEventListener('uncapturederror', event => {
      validationErrors.push(event.error?.message || String(event.error));
    });

    const ownedBuffers = new Set();
    const patternedBuffer = (length, scale, phase = 0) => {
      const values = new Float32Array(length);
      for (let index = 0; index < length; index += 1) {
        values[index] = (((index + phase) % 31) - 15) * scale;
      }
      const buffer = createStorageBuffer(device, values);
      ownedBuffers.add(buffer);
      return buffer;
    };
    const constantBuffer = (length, value) => {
      const values = new Float32Array(length);
      values.fill(value);
      const buffer = createStorageBuffer(device, values);
      ownedBuffers.add(buffer);
      return buffer;
    };

    const D = VIT_CONFIG.dim;
    const H = VIT_CONFIG.mlpHiddenDim;
    const tokenH = 1;
    const tokenW = 1;
    const tokenCount = 2;
    const shared = {
      normWeight: constantBuffer(D, 1),
      normBias: patternedBuffer(D, 0.0002, 1),
      qkvWeight: patternedBuffer(D * 3 * D, 0.00001, 2),
      qkvBias: patternedBuffer(3 * D, 0.0001, 3),
      projectionWeight: patternedBuffer(D * D, 0.00001, 4),
      projectionBias: patternedBuffer(D, 0.0001, 5),
      layerScale: constantBuffer(D, 0.01),
      fc1Weight: patternedBuffer(D * H, 0.000006, 6),
      fc1Bias: patternedBuffer(H, 0.00008, 7),
      fc2Weight: patternedBuffer(H * D, 0.000006, 8),
      fc2Bias: patternedBuffer(D, 0.00008, 9),
    };
    const blockWeights = {};
    for (let layer = 0; layer < VIT_CONFIG.numLayers; layer += 1) {
      const prefix = `blocks.${layer}`;
      blockWeights[`${prefix}.norm1.weight`] = shared.normWeight;
      blockWeights[`${prefix}.norm1.bias`] = shared.normBias;
      blockWeights[`${prefix}.attn.qkv.weight`] = shared.qkvWeight;
      blockWeights[`${prefix}.attn.qkv.bias`] = shared.qkvBias;
      blockWeights[`${prefix}.attn.proj.weight`] = shared.projectionWeight;
      blockWeights[`${prefix}.attn.proj.bias`] = shared.projectionBias;
      blockWeights[`${prefix}.ls1.gamma`] = shared.layerScale;
      blockWeights[`${prefix}.norm2.weight`] = shared.normWeight;
      blockWeights[`${prefix}.norm2.bias`] = shared.normBias;
      blockWeights[`${prefix}.mlp.fc1.weight`] = shared.fc1Weight;
      blockWeights[`${prefix}.mlp.fc1.bias`] = shared.fc1Bias;
      blockWeights[`${prefix}.mlp.fc2.weight`] = shared.fc2Weight;
      blockWeights[`${prefix}.mlp.fc2.bias`] = shared.fc2Bias;
      blockWeights[`${prefix}.ls2.gamma`] = shared.layerScale;
    }
    const weights = {
      patchEmbed: {
        weight: patternedBuffer(D * 3 * VIT_CONFIG.patchSize * VIT_CONFIG.patchSize, 0.00002, 10),
        bias: patternedBuffer(D, 0.0001, 11),
      },
      clsToken: patternedBuffer(D, 0.001, 12),
      posEmbed: patternedBuffer(tokenCount * D, 0.0005, 13),
      norm: {
        weight: shared.normWeight,
        bias: shared.normBias,
      },
      blockWeights,
    };
    const image = patternedBuffer(3 * VIT_CONFIG.patchSize * VIT_CONFIG.patchSize, 0.01, 14);

    const requestedBase = {
      mode: 'cooperative',
      yieldMs: 4,
      waitForSubmittedWorkDone: true,
      vitBlockChunkSize: 1,
      vitMicroduty: true,
    };
    const encoder = new ViTEncoder(device);
    encoder.init();
    let runSequence = 0;
    const run = async (mode, schedulerRequest = requestedBase, captureTensors = true) => {
      const scheduler = parseSharpSchedulerConfig({
        sharpScheduler: { ...schedulerRequest, vitMicrodutyMode: mode },
      });
      const telemetry = createSharpRunTelemetry(scheduler, {
        runId: `vit-four-stage-parity-${mode}-${runSequence}`,
      });
      runSequence += 1;
      const startedAt = performance.now();
      const output = await encoder.encode(image, weights, tokenH, tokenW, {
        scheduler,
        telemetry,
        encoderLabel: `parity-${mode}`,
        retainOutputs: true,
      });
      await device.queue.onSubmittedWorkDone();
      const durationMs = performance.now() - startedAt;
      const tensors = captureTensors
        ? [
            await readBuffer(device, output.finalTokensBuf, tokenCount * D * 4),
            ...await Promise.all(output.intermediateFeatures.map(
              feature => readBuffer(device, feature.buffer, tokenCount * D * 4),
            )),
          ]
        : [];
      const queueSettlements = telemetry.events.filter(
        event => event.kind === 'queue-work-done-end',
      );
      const microphaseSequence = queueSettlements
        .map(event => event.microphase)
        .filter(Boolean);
      for (const feature of output.intermediateFeatures) feature.buffer.destroy();
      output.finalTokensBuf.destroy();
      return {
        durationMs,
        tensors,
        queueSettlementCount: queueSettlements.length,
        microphaseSequence,
        requestedMode: scheduler.requested.vitMicrodutyMode,
        effectiveMode: scheduler.effective.vitMicrodutyMode,
      };
    };

    const warmup = {
      ...requestedBase,
      yieldMs: 0,
      waitForSubmittedWorkDone: false,
      vitBlockChunkSize: VIT_CONFIG.numLayers,
      vitMicroduty: false,
    };
    await run('two-stage', warmup, false);
    const measurementOrder = ['dispatch-major', 'four-stage', 'four-stage', 'dispatch-major'];
    const measuredRuns = [];
    for (const mode of measurementOrder) measuredRuns.push(await run(mode));

    const reference = measuredRuns[0];
    const comparisons = measuredRuns.slice(1).flatMap((candidate, candidateIndex) => (
      reference.tensors.map((expected, tensorIndex) => {
        const actual = candidate.tensors[tensorIndex];
        let bitMismatchCount = 0;
        let maxAbsError = 0;
        let nonFiniteCount = 0;
        const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
        const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
        for (let index = 0; index < expected.length; index += 1) {
          if (!Number.isFinite(expected[index]) || !Number.isFinite(actual[index])) {
            nonFiniteCount += 1;
          }
          if (expectedBits[index] !== actualBits[index]) bitMismatchCount += 1;
          maxAbsError = Math.max(maxAbsError, Math.abs(expected[index] - actual[index]));
        }
        return {
          comparisonMode: candidate.effectiveMode,
          comparisonRunIndex: candidateIndex + 1,
          tensorIndex,
          elementCount: expected.length,
          bitMismatchCount,
          maxAbsError,
          nonFiniteCount,
        };
      })
    ));
    const median = values => {
      const sorted = [...values].sort((a, b) => a - b);
      const middle = Math.floor(sorted.length / 2);
      return sorted.length % 2
        ? sorted[middle]
        : (sorted[middle - 1] + sorted[middle]) / 2;
    };
    const summarizeMode = mode => {
      const runs = measuredRuns.filter(runResult => runResult.effectiveMode === mode);
      const durationSamplesMs = runs.map(runResult => runResult.durationMs);
      return {
        durationSamplesMs,
        medianDurationMs: median(durationSamplesMs),
        queueSettlementCounts: runs.map(runResult => runResult.queueSettlementCount),
        requestedMode: runs[0].requestedMode,
        effectiveMode: runs[0].effectiveMode,
        firstBlockMicrophases: runs[0].microphaseSequence.slice(0, 4),
      };
    };
    const dispatchMajor = summarizeMode('dispatch-major');
    const fourStage = summarizeMode('four-stage');
    const sample = Array.from(
      measuredRuns.find(runResult => runResult.effectiveMode === 'four-stage').tensors[0].slice(0, 8),
    );

    for (const buffer of Object.values(encoder._wb || {})) buffer.destroy();
    for (const buffer of encoder._uniformCache.values()) buffer.destroy();
    for (const buffer of ownedBuffers) buffer.destroy();
    await new Promise(resolve => setTimeout(resolve, 50));
    return {
      schema: 'sharp-webgpu.vit-four-stage-production-parity.v0',
      productionClass: ViTEncoder.name,
      config: {
        dim: D,
        hiddenDim: H,
        layerCount: VIT_CONFIG.numLayers,
        tokenCount,
        yieldMs: requestedBase.yieldMs,
        measurementOrder,
      },
      dispatchMajor,
      fourStage,
      comparisons,
      sample,
      validationErrors,
    };
  });

  assert.equal(pageErrors.length, 0, `page errors: ${pageErrors.join('; ')}`);
  assert.equal(result.productionClass, 'ViTEncoder');
  assert.deepEqual(result.validationErrors, []);
  assert.equal(result.dispatchMajor.requestedMode, 'dispatch-major');
  assert.equal(result.dispatchMajor.effectiveMode, 'dispatch-major');
  assert.equal(result.fourStage.requestedMode, 'four-stage');
  assert.equal(result.fourStage.effectiveMode, 'four-stage');
  assert.deepEqual(result.dispatchMajor.queueSettlementCounts, [289, 289]);
  assert.deepEqual(result.fourStage.queueSettlementCounts, [96, 96]);
  assert.deepEqual(result.fourStage.firstBlockMicrophases, [
    'norm1-qkv',
    'attention-projection-residual',
    'norm2-fc1',
    'fc2-residual',
  ]);
  assert.equal(result.comparisons.length, 15);
  for (const comparison of result.comparisons) {
    assert.equal(comparison.nonFiniteCount, 0, `tensor ${comparison.tensorIndex} must stay finite`);
    assert.equal(comparison.bitMismatchCount, 0, `tensor ${comparison.tensorIndex} must be bit-identical`);
    assert.equal(comparison.maxAbsError, 0, `tensor ${comparison.tensorIndex} must have zero error`);
  }
  assert.ok(result.sample.some(value => Math.abs(value) > 0.01), 'parity input must produce non-trivial output');

  failurePhase = 'report-write';
  const savedMsPerEncoder =
    result.dispatchMajor.medianDurationMs - result.fourStage.medianDurationMs;
  const report = {
    status: 'passed',
    runId,
    startedAt,
    completedAt: new Date().toISOString(),
    requestedRoute: url,
    effectiveRoute: page.url(),
    ...result,
    observed: {
      projectionAuthority: 'synthetic-scheduler-overhead-projection',
      projectionBasis: 'paired median from warmed two-token production ViTEncoder runs',
      fullRouteWallTimeMeasured: false,
      projectedEncoderInvocationCount: 36,
      savedMsPerEncoder,
      projectedSavedMsAcross36Encoders: savedMsPerEncoder * 36,
      removedQueueSettlementsPerEncoder:
        result.dispatchMajor.queueSettlementCounts[0] - result.fourStage.queueSettlementCounts[0],
      projectedRemovedQueueSettlementsAcross36Encoders:
        (result.dispatchMajor.queueSettlementCounts[0] - result.fourStage.queueSettlementCounts[0]) * 36,
    },
  };
  lastTrustworthyEvidence = {
    ...lastTrustworthyEvidence,
    parityCompleted: true,
    comparisonCount: result.comparisons.length,
    validationErrorCount: result.validationErrors.length,
  };
  writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify(report, null, 2));
  console.log(`Report: ${reportPath}`);
} catch (error) {
  const failureReport = {
    schema: 'sharp-webgpu.vit-four-stage-production-parity.v0',
    status: 'failed',
    runId,
    startedAt,
    completedAt: new Date().toISOString(),
    requestedRoute: url,
    effectiveRoute: page?.url() || null,
    failurePhase,
    lastTrustworthyEvidence,
    error: {
      name: error?.name || 'Error',
      message: error?.message || String(error),
      stack: error?.stack || null,
    },
  };
  writeFileSync(reportPath, `${JSON.stringify(failureReport, null, 2)}\n`);
  throw error;
} finally {
  if (browser) await browser.close();
  if (vite?.exitCode === null) {
    vite.kill('SIGTERM');
    await new Promise(resolve => {
      vite.once('exit', resolve);
      setTimeout(resolve, 2_000);
    });
  }
}
