/**
 * SHARP-WebGPU — Main entry point
 *
 * Apple SHARP (single-image 3D Gaussian Splat generation) in WebGPU compute.
 *
 * Current status: backbone + SPN encoder.
 * Full pipeline (monodepth decoder, Gaussian decoder, 3DGS output) not yet implemented.
 */

import { initGPU, readBuffer, validateSharpDeviceCapabilities } from './lib/gpu.js';
import { loadWeights } from './lib/weights.js';
import { SharpBackbone } from './lib/backbone.js';
import { SlidingPyramidNetwork } from './lib/spn.js';
import { MonodepthDecoder } from './lib/monodepth.js';
import { GaussianPipeline } from './lib/gaussian_decoder.js';
import { composeAndExport, validateSharpDisparityPlausibility } from './lib/compose.js';
import {
  attachSharpLiveScheduler,
  createSharpRunTelemetry,
  createSharpRuntimeDutyMap,
  classifyCpuDutyCheckpoint,
  detachSharpLiveScheduler,
  parseSharpSchedulerConfig,
  recordSchedulerEvent,
  schedulerYield,
  schedulerTelemetrySnapshot,
  schedulerTelemetrySnapshotCooperatively,
} from './lib/scheduler.js';
import {
  createSharpRouteRuntime,
  finishSharpRouteRuntimeProfile,
} from './lib/route_runtime.js';
import {
  SHARP_IMAGE_TO_SPLAT_ROUTE_ID,
  WEBGPU_HOST_PHASE,
  WEBGPU_INFERENCE_KIT_VERSION,
  addStagedSubmitStage,
  classifyWebGpuRouteReceiptEvidence,
  createSharpImageToSplatRouteDefinition,
  createSharpImageToSplatRouteReceipt,
  createStagedSubmitProfile,
  createWebGpuBackendIdentity,
} from '@kaminos/webgpu-inference-kit';

let gpu = null;
let weights = null;
let backbone = null;
let spn = null;
let monodepth = null;
let gaussianPipeline = null;
let weightsLoadedMB = 0;

const sharpElementPrefix = globalThis.__kaminosSharpElementPrefix || '';
const sharpElementRoot = globalThis.__kaminosSharpElementRoot
  || (typeof document !== 'undefined' ? document : null);
const sharpRuntimeGlobal = globalThis.window || globalThis;

export function resolveSharpElement(root, prefix, id) {
  if (!root) return null;
  const elementId = `${prefix}${id}`;
  if (typeof root.getElementById === 'function') return root.getElementById(elementId);
  if (typeof root.querySelectorAll !== 'function') return null;
  return Array.from(root.querySelectorAll('[id]')).find(element => element.id === elementId) || null;
}

function sharpElement(id) {
  return resolveSharpElement(sharpElementRoot, sharpElementPrefix, id);
}

const dropZone = sharpElement('drop-zone');
const fileInput = sharpElement('file-input');
const statusEl = sharpElement('status');
const errorEl = sharpElement('error');
const outputEl = sharpElement('output');
const resultsEl = sharpElement('results');
const sharpRouteDefinition = createSharpImageToSplatRouteDefinition({
  kernel: {
    kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
    profile: 'spn-dinov2l16-monodepth-gaussian-ply',
    commit: 'sharp-webgpu-browser-runtime',
  },
});

sharpRuntimeGlobal.__sharpDebug = {
  schema: 'sharp.webgpu-route-debug.v0',
  kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
  routeDefinition: sharpRouteDefinition,
  lastRun: null,
};

function createRouteRunDebug(mode) {
  return {
    schema: 'sharp.webgpu-route-run-debug.v0',
    route: {
      requestedRouteId: SHARP_IMAGE_TO_SPLAT_ROUTE_ID,
      effectiveRouteId: mode === 'spn' ? SHARP_IMAGE_TO_SPLAT_ROUTE_ID : 'sharp.backbone-smoke.webgpu-local.v0',
      receipt: null,
      evidence: null,
      receiptError: null,
    },
    mode,
    status: 'running',
    startedAt: new Date().toISOString(),
    startMs: performance.now(),
    endMs: null,
    elapsedMs: null,
    phases: [],
    scheduler: sharpRouteDefinition.scheduler,
    routeScheduler: sharpRouteDefinition.scheduler,
    sharpScheduler: null,
    backpressure: sharpRouteDefinition.backpressure,
    runtimeProfile: null,
    routeTailTimings: [],
    backgroundDutyMap: createSharpRuntimeDutyMap(),
    progressEvents: [],
    outputs: {},
    error: null,
  };
}

function featureNames(features) {
  return Array.from(features || []).map(String).sort();
}

function browserBackendIdentity() {
  const adapterInfo = gpu?.adapter?.info || {};
  const deviceFeatures = featureNames(gpu?.device?.features);
  const adapterFeatures = featureNames(gpu?.adapter?.features);
  const effectiveFeatures = deviceFeatures.length
    ? deviceFeatures
    : (adapterFeatures.length ? adapterFeatures : ['webgpu-core']);

  return createWebGpuBackendIdentity({
    adapterName: adapterInfo.description || adapterInfo.device || adapterInfo.vendor || 'unknown-webgpu-adapter',
    browser: navigator.userAgent,
    requestedFeatures: [],
    effectiveFeatures,
    limits: gpu?.device?.limits || gpu?.adapter?.limits || {},
    timestampQuery: effectiveFeatures.includes('timestamp-query') ? 'available' : 'unavailable',
  });
}

async function sha256Hex(value) {
  let buffer;
  if (value instanceof Blob) {
    buffer = await value.arrayBuffer();
  } else if (ArrayBuffer.isView(value)) {
    buffer = value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  } else if (value instanceof ArrayBuffer) {
    buffer = value;
  } else if (typeof value === 'string') {
    buffer = new TextEncoder().encode(value);
  } else {
    buffer = new TextEncoder().encode(JSON.stringify(value));
  }
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}

async function createExecutionRouteReceipt({ blob, bitmap, depthResult, dispData, composed, runDebug }) {
  const profile = runDebug.runtimeProfile?.profile || createStagedSubmitProfile({
    route: SHARP_IMAGE_TO_SPLAT_ROUTE_ID,
    timingSource: 'adapter-phase-wall-clock',
    requiredStages: sharpRouteDefinition.requiredStages,
  });
  if (!runDebug.runtimeProfile?.profile) {
    for (const phase of runDebug.phases) {
      addStagedSubmitStage(profile, {
        name: phase.name,
        ms: phase.ms,
      });
    }
  }

  const sourceHash = await sha256Hex(blob);
  const splatHash = await sha256Hex(composed.plyBlob);
  const depthHash = await sha256Hex(dispData);
  const metadata = {
    routeId: SHARP_IMAGE_TO_SPLAT_ROUTE_ID,
    phases: runDebug.phases,
    runtimeProfile: runDebug.runtimeProfile,
    routeTailTimings: runDebug.routeTailTimings,
    backgroundDutyMap: runDebug.backgroundDutyMap,
    elapsedMs: runDebug.inferenceElapsedMs,
    outputs: runDebug.outputs,
  };
  const metadataHash = await sha256Hex(metadata);

  const receipt = createSharpImageToSplatRouteReceipt({
    input: {
      artifactId: `source-image:browser:${sourceHash.slice(0, 16)}`,
      sha256: sourceHash,
      shape: [bitmap.height, bitmap.width, 4],
    },
    outputs: {
      splat: {
        artifactId: `splat-candidate:ply:${splatHash.slice(0, 16)}`,
        sha256: splatHash,
        shape: [composed.numGaussians, 14],
      },
      depthMap: {
        artifactId: `depth-map:disparity:${depthHash.slice(0, 16)}`,
        sha256: depthHash,
        shape: [depthResult.H, depthResult.W, depthResult.C || 1],
      },
      metadata: {
        artifactId: `sharp-webgpu-metadata:${metadataHash.slice(0, 16)}`,
        sha256: metadataHash,
        shape: [1],
      },
    },
    backend: browserBackendIdentity(),
    model: {
      revision: 'local-sharp-webgpu',
      weightsHash: `weights.bin:size-mb-${weightsLoadedMB}:sha256-not-collected`,
    },
    kernel: {
      kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
      profile: 'spn-dinov2l16-monodepth-gaussian-ply',
      commit: 'sharp-webgpu-browser-runtime',
    },
    profile,
  });
  receipt.metadataPayload = metadata;
  return receipt;
}

function finishRoutePhase(run, name, startMs) {
  run.phases.push({
    name,
    ms: performance.now() - startMs,
  });
}

async function runRouteStage(routeRuntime, run, name, fn, metadata = {}) {
  const {
    schedulerConfig,
    schedulerTelemetry,
    ...stageMetadata
  } = metadata;
  const invokeStage = () => routeRuntime.runStage(name, fn, {
    routeStage: name,
    ...stageMetadata,
  });
  const result = schedulerConfig && schedulerTelemetry && typeof routeRuntime.runInvocation === 'function'
    ? await routeRuntime.runInvocation({ invocationId: `${schedulerTelemetry.runId}:${name}` }, async invocation => {
      attachSharpLiveScheduler(schedulerConfig, {
        runtime: routeRuntime,
        invocation,
        stage: name,
      });
      try {
        return await invokeStage();
      } finally {
        detachSharpLiveScheduler(schedulerConfig);
      }
    })
    : await invokeStage();
  const stage = routeRuntime.profile.stages[routeRuntime.profile.stages.length - 1];
  run.phases.push({
    name,
    ms: Number.isFinite(stage?.ms) ? stage.ms : 0,
  });
  return result;
}

async function recordRouteTailStep(run, scheduler, telemetry, device, details, fn) {
  const startedAtMs = performance.now();
  const result = await fn();
  const ms = Number((performance.now() - startedAtMs).toFixed(3));
  const entry = {
    stage: details.stage,
    step: details.step,
    ms,
    ...(Number.isFinite(details.bytes) ? { bytes: details.bytes } : {}),
    ...(Number.isFinite(details.pixels) ? { pixels: details.pixels } : {}),
  };
  run.routeTailTimings.push(entry);
  await schedulerYield(scheduler, device, telemetry, 'route-tail', {
    ...entry,
    role: 'route-tail-checkpoint',
  });
  return result;
}

function recordRouteTailInterval(run, telemetry, details, fn) {
  const intervalStartMs = performance.now();
  const result = fn();
  const intervalEndMs = performance.now();
  const entry = {
    stage: details.stage,
    step: details.step,
    role: details.role || 'blocking-duty-interval',
    intervalStartMs,
    intervalEndMs,
    durationMs: intervalEndMs - intervalStartMs,
    ...(Number.isFinite(details.bytes) ? { bytes: details.bytes } : {}),
  };
  run.routeTailTimings.push(entry);
  recordSchedulerEvent(telemetry, 'route-tail', {
    ...entry,
    kind: 'duty-interval',
  });
  details.onMeasured?.(entry);
  return result;
}

function recordObservedRouteTailInterval(run, telemetry, details) {
  const entry = {
    stage: details.stage,
    step: details.step,
    role: details.role || 'blocking-duty-interval',
    intervalStartMs: details.intervalStartMs,
    intervalEndMs: details.intervalEndMs,
    durationMs: details.durationMs,
    ...(Number.isFinite(details.bytes) ? { bytes: details.bytes } : {}),
  };
  run.routeTailTimings.push(entry);
  recordSchedulerEvent(telemetry, 'route-tail', {
    ...entry,
    kind: 'duty-interval',
  });
}

async function recordCpuDutyChunk(run, scheduler, telemetry, device, details, processedItems) {
  const checkpoint = classifyCpuDutyCheckpoint(scheduler, details, processedItems);
  if (!checkpoint.eligible) return;
  const entry = {
    stage: details.stage,
    step: details.step,
    role: 'cpu-materialization-chunk',
    processedItems,
    phaseComplete: checkpoint.phaseComplete,
    ...(Number.isFinite(details.pixels) ? { pixels: details.pixels } : {}),
    ...(Number.isFinite(details.totalItems) ? { totalItems: details.totalItems } : {}),
    ...(Number.isFinite(details.checkpointItems) ? { checkpointItems: details.checkpointItems } : {}),
    ...(Number.isFinite(details.segmentStartProcessedItems) ? { segmentStartProcessedItems: details.segmentStartProcessedItems } : {}),
    ...(Number.isFinite(details.segmentEndProcessedItems) ? { segmentEndProcessedItems: details.segmentEndProcessedItems } : {}),
    ...(details.granularity ? { granularity: details.granularity } : {}),
    ...(Number.isFinite(details.intervalStartMs) ? { intervalStartMs: details.intervalStartMs } : {}),
    ...(Number.isFinite(details.intervalEndMs) ? { intervalEndMs: details.intervalEndMs } : {}),
    ...(Number.isFinite(details.durationMs) ? { durationMs: details.durationMs } : {}),
  };
  run.routeTailTimings.push(entry);
  const hasWorkInterval = Number.isFinite(entry.intervalStartMs)
    && Number.isFinite(entry.intervalEndMs)
    && Number.isFinite(entry.durationMs);
  if (hasWorkInterval) {
    recordSchedulerEvent(telemetry, 'route-tail', {
      ...entry,
      kind: 'duty-interval',
    });
  }
  const isFinalGaussianRemainder = entry.step === 'gaussian-compose'
    && entry.phaseComplete
    && processedItems % scheduler.effective.cpuChunkItems !== 0;
  if (isFinalGaussianRemainder) return;
  const {
    intervalStartMs: _intervalStartMs,
    intervalEndMs: _intervalEndMs,
    durationMs: _durationMs,
    ...yieldEntry
  } = entry;
  await schedulerYield(scheduler, device, telemetry, 'route-tail', yieldEntry);
}

function finishRouteRun(run, status, outputs = {}) {
  run.status = status;
  run.endMs = performance.now();
  run.elapsedMs = run.endMs - run.startMs;
  run.endedAt = new Date().toISOString();
  run.outputs = { ...run.outputs, ...outputs };
}

function setStatus(msg) {
  if (statusEl) statusEl.textContent = msg;
  if (errorEl) errorEl.style.display = 'none';
}

function setError(msg) {
  if (errorEl) {
    errorEl.textContent = msg;
    errorEl.style.display = 'block';
  }
  if (statusEl) statusEl.textContent = '';
}

function showResults(result, elapsed, mode) {
  const setText = (id, value) => {
    const element = sharpElement(id);
    if (element) element.textContent = value;
  };
  setText('r-model', 'DINOv2 ViT-Large (dinov2l16_384)');
  setText('r-weights', `${weightsLoadedMB} MB (fp16)`);
  setText('r-patch', '16x16');

  if (mode === 'spn') {
    setText('r-title', 'Full Route Results');
    setText('r-time-label', 'Full route time');
    setText('r-grid', 'SPN: 35 patches (5x5 + 3x3 + 1x1)');
    const gaussStr = result.numGaussians ? ` → ${(result.numGaussians / 1000).toFixed(0)}K Gaussians` : '';
    setText('r-features', `${result.featureDims.length} multi-res outputs${gaussStr}`);
  } else {
    setText('r-title', 'Backbone Results');
    setText('r-time-label', 'Backbone time');
    setText('r-grid', `${result.tokenH}x${result.tokenW} = ${result.numPatches} patches + 1 CLS`);
    setText('r-features', `${result.intermediateFeatures.length} layers`);
  }

  setText('r-time', `${elapsed.toFixed(0)} ms`);

  const validEl = sharpElement('r-valid');
  if (validEl) {
    if (result.hasNaN) {
      validEl.textContent = 'INVALID (NaN/Inf in output)';
      validEl.style.color = '#f66';
    } else {
      validEl.textContent = 'OK';
      validEl.style.color = '#6f6';
    }
  }

  resultsEl?.classList.add('visible');
}

// --- Drop zone ---
function runSharpStandaloneBlob(blob) {
  return runSharpImageToSplat(blob, {
    mode: sharpElement('use-spn')?.checked === false ? 'backbone' : 'spn',
  });
}

if (dropZone && fileInput) {
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropZone.classList.add('dragover');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropZone.classList.remove('dragover');
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) runSharpStandaloneBlob(file);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) runSharpStandaloneBlob(fileInput.files[0]);
  });
}

// --- Sample image clicks ---
if (sharpElementRoot && typeof sharpElementRoot.querySelectorAll === 'function') {
  sharpElementRoot.querySelectorAll('.sample-thumb').forEach(thumb => {
    thumb.addEventListener('click', async () => {
      const url = thumb.dataset.full;
      try {
        setStatus('Loading sample image...');
        const resp = await fetch(url);
        const blob = await resp.blob();
        await runSharpStandaloneBlob(blob);
      } catch (err) {
        setError(`Failed to load sample: ${err.message}`);
      }
    });
  });
}

export function resolveSharpRunMode(options = {}) {
  const mode = options.mode || 'spn';
  if (mode !== 'spn' && mode !== 'backbone') throw new Error(`unsupported SHARP run mode: ${mode}`);
  return mode;
}

export async function runSharpImageToSplat(blob, options = {}) {
  const runMode = resolveSharpRunMode(options);
  const runDebug = createRouteRunDebug(runMode);
  const currentScheduler = parseSharpSchedulerConfig({
    sharpScheduler: options.scheduler || options.sharpScheduler,
  });
  const currentSchedulerTelemetry = createSharpRunTelemetry(currentScheduler, { mode: runMode });
  const progressCounts = new Map();
  let lastProgress = 0;
  const emitProgress = (progress, message, details = {}) => {
    const nextProgress = Math.max(lastProgress, Math.min(0.93, Number(progress) || 0));
    lastProgress = nextProgress;
    const event = {
      schema: 'sharp-webgpu.progress.v0',
      progress: nextProgress,
      message,
      progressAuthority: 'stage-weighted-work-projection',
      timestampMs: performance.now(),
      ...details,
    };
    runDebug.progressEvents.push(event);
    options.onProgress?.(event);
  };
  Object.defineProperty(currentScheduler, 'progressReporter', {
    configurable: true,
    value: event => {
      const count = (progressCounts.get(event.phase) || 0) + 1;
      progressCounts.set(event.phase, count);
      const projection = event.phase === 'monodepth-phase'
        ? { floor: 0.66, ceiling: 0.76, divisor: 28, message: 'SHARP is resolving scene depth.' }
        : event.phase === 'gaussian-phase'
          ? { floor: 0.79, ceiling: 0.89, divisor: 34, message: 'SHARP is predicting Gaussian geometry.' }
          : event.phase === 'route-tail'
            ? { floor: 0.90, ceiling: 0.925, divisor: 18, message: 'SHARP is assembling the splat artifact.' }
            : { floor: 0.20, ceiling: 0.64, divisor: 170, message: 'SHARP is extracting image features.' };
      const progress = projection.floor
        + (projection.ceiling - projection.floor) * (1 - Math.exp(-count / projection.divisor));
      emitProgress(progress, projection.message, {
        phase: event.phase,
        boundary: event.boundary,
        workOrdinal: count,
      });
    },
  });
  runDebug.sharpScheduler = currentScheduler;
  sharpRuntimeGlobal.__sharpDebug.lastRun = runDebug;
  sharpRuntimeGlobal.__SHARP_LAST_RUN_TELEMETRY__ = schedulerTelemetrySnapshot(currentSchedulerTelemetry, 'running');

  try {
    emitProgress(0.01, 'SHARP is initializing WebGPU.', { phase: 'initializing' });
    setStatus('Initializing WebGPU...');
    const injectedGpu = options.gpuContext
      || globalThis.__kaminosSharpInjectedGpu
      || null;
    if (injectedGpu && (!injectedGpu.device || injectedGpu.queue !== injectedGpu.device.queue)) {
      throw new Error('injected SHARP GPU queue does not belong to the injected device');
    }
    if (!gpu) {
      gpu = options.gpuContext
        || globalThis.__kaminosSharpInjectedGpu
        || await initGPU();
    } else if (injectedGpu && (gpu.device !== injectedGpu.device || gpu.device.queue !== injectedGpu.queue)) {
      throw new Error('SHARP route is already bound to a different GPU device or queue');
    }
    try {
      runDebug.outputs.deviceCapability = validateSharpDeviceCapabilities(gpu.device);
    } catch (error) {
      runDebug.outputs.deviceCapability = error.deviceCapability || null;
      throw error;
    }
    const routeRuntime = await createSharpRouteRuntime(gpu, {
      routeDefinition: sharpRouteDefinition,
      browser: navigator.userAgent,
      runId: currentSchedulerTelemetry.runId,
      clock: currentSchedulerTelemetry.eventTrace.clock,
      scheduler: currentScheduler.effective,
      schedulerBounds: {
        yieldMs: { min: 0, max: 1_000, step: 1 },
        phaseChunkSize: {
          spnPatch: { min: 1, max: 35, stepFactor: 2 },
          vitBlock: { min: 1, max: 24, stepFactor: 2 },
        },
      },
      now: () => performance.now(),
    });

    // Show input preview (preserve aspect ratio)
    emitProgress(0.03, 'SHARP is loading the source image.', { phase: 'source-load' });
    setStatus('Loading image...');
    const bitmap = await createImageBitmap(blob);
    const inputCanvas = sharpElement('input-canvas');
    const maxDisplay = 384;
    const scale = Math.min(maxDisplay / bitmap.width, maxDisplay / bitmap.height);
    if (inputCanvas) {
      inputCanvas.width = Math.round(bitmap.width * scale);
      inputCanvas.height = Math.round(bitmap.height * scale);
      const ctx = inputCanvas.getContext('2d');
      ctx.drawImage(bitmap, 0, 0, inputCanvas.width, inputCanvas.height);
    }
    outputEl?.classList.add('visible');

    if (!weights) {
      setStatus('Loading SHARP weights (~1.25 GB, first load only)...');
      const weightsUrl = options.weightsUrl
        || globalThis.__kaminosSharpWeightsUrl
        || '/weights.bin';
      const weightsLoadStartMs = performance.now();
      try {
        weights = await loadWeights(gpu.device, weightsUrl, (received, total) => {
          const mb = (received / 1024 / 1024).toFixed(0);
          weightsLoadedMB = mb;
          const totalMb = total ? (total / 1024 / 1024).toFixed(0) : '?';
          setStatus(`Loading weights: ${mb} / ${totalMb} MB`);
          const fraction = total > 0 ? Math.min(1, received / total) : 0;
          emitProgress(0.05 + 0.10 * fraction, 'SHARP is loading model weights.', {
            phase: 'weights-load',
            bytesReceived: received,
            bytesTotal: total || null,
          });
        });
      } finally {
        const weightsLoadEndMs = performance.now();
        recordSchedulerEvent(currentSchedulerTelemetry, 'weights-load', {
          boundary: 'weights-load',
          kind: 'duty-interval',
          stage: 'route-setup',
          step: 'fetch-decode-upload',
          role: 'blocking-duty-interval',
          intervalStartMs: weightsLoadStartMs,
          intervalEndMs: weightsLoadEndMs,
          durationMs: weightsLoadEndMs - weightsLoadStartMs,
        });
      }
    }

    // Use SPN for full pipeline, backbone for quick smoke
    const useSPN = runMode === 'spn';

    if (useSPN) {
      if (!spn) {
        spn = new SlidingPyramidNetwork(gpu.device);
        spn.init(weights);
      }

      setStatus('Running SPN (35 ViT passes, may take 15-30s)...');
      emitProgress(0.18, 'SHARP is preparing image features.', { phase: 'source-preprocess' });

      let chw;
      const sourcePreprocessStartMs = performance.now();
      try {
        ({ chw } = await routeRuntime.runHostPhase(
          WEBGPU_HOST_PHASE.cpuPreprocess,
          async () => {
          // Resize to 1536x1536 and normalize to [-1, 1] CHW
          const spnSize = 1536;
          const spnBitmap = await createImageBitmap(blob, { resizeWidth: spnSize, resizeHeight: spnSize });
          const spnCanvas = new OffscreenCanvas(spnSize, spnSize);
          const spnCtx = spnCanvas.getContext('2d');
          spnCtx.drawImage(spnBitmap, 0, 0);
          const spnImageData = spnCtx.getImageData(0, 0, spnSize, spnSize);

          const normalizedChw = new Float32Array(3 * spnSize * spnSize);
          for (let y = 0; y < spnSize; y++) {
            for (let x = 0; x < spnSize; x++) {
              const srcIdx = (y * spnSize + x) * 4;
              const dstBase = y * spnSize + x;
              normalizedChw[0 * spnSize * spnSize + dstBase] = spnImageData.data[srcIdx] / 127.5 - 1.0;
              normalizedChw[1 * spnSize * spnSize + dstBase] = spnImageData.data[srcIdx + 1] / 127.5 - 1.0;
              normalizedChw[2 * spnSize * spnSize + dstBase] = spnImageData.data[srcIdx + 2] / 127.5 - 1.0;
            }
          }
          return { chw: normalizedChw };
        },
        {
          detail: {
            operation: 'source-image-resize-normalize',
            targetWidth: 1536,
            targetHeight: 1536,
          },
          }
        ));
      } finally {
        const sourcePreprocessEndMs = performance.now();
        recordSchedulerEvent(currentSchedulerTelemetry, 'source-preprocess', {
          boundary: 'source-preprocess',
          kind: 'duty-interval',
          stage: 'route-setup',
          step: 'resize-normalize',
          role: 'blocking-duty-interval',
          intervalStartMs: sourcePreprocessStartMs,
          intervalEndMs: sourcePreprocessEndMs,
          durationMs: sourcePreprocessEndMs - sourcePreprocessStartMs,
        });
      }

      const foregroundHandoffStartMs = performance.now();
      try {
        await options.beforeInference?.({
          runId: currentSchedulerTelemetry.runId,
          mode: runMode,
          routeId: SHARP_IMAGE_TO_SPLAT_ROUTE_ID,
        });
      } finally {
        const foregroundHandoffEndMs = performance.now();
        recordSchedulerEvent(currentSchedulerTelemetry, 'foreground-handoff', {
          boundary: 'foreground-handoff',
          kind: 'duty-interval',
          stage: 'product-composition',
          step: 'lease-activation',
          role: 'blocking-duty-interval',
          intervalStartMs: foregroundHandoffStartMs,
          intervalEndMs: foregroundHandoffEndMs,
          durationMs: foregroundHandoffEndMs - foregroundHandoffStartMs,
        });
      }
      sharpRuntimeGlobal.__sharpContentionProbe?.markInferenceStart?.(currentSchedulerTelemetry.runId);
      const t0 = performance.now();
      const spnResult = await runRouteStage(routeRuntime, runDebug, 'spn', () => spn.run(chw, {
        scheduler: currentScheduler,
        telemetry: currentSchedulerTelemetry,
      }), {
        schedulerMode: currentScheduler.effective?.mode,
        schedulerConfig: currentScheduler,
        schedulerTelemetry: currentSchedulerTelemetry,
      });
      emitProgress(0.65, 'SHARP finished image feature extraction.', { phase: 'spn-complete' });

      // Run monodepth decoder
      if (!monodepth) {
        monodepth = new MonodepthDecoder(gpu.device);
      }
      setStatus('Running monodepth decoder...');
      emitProgress(0.66, 'SHARP is resolving scene depth.', { phase: 'monodepth' });
      const depthResult = await runRouteStage(routeRuntime, runDebug, 'monodepth', () => (
        monodepth.run(spnResult.features, spnResult.featureDims, weights, {
          scheduler: currentScheduler,
          telemetry: currentSchedulerTelemetry,
        })
      ), {
        schedulerMode: currentScheduler.effective?.mode,
        schedulerConfig: currentScheduler,
        schedulerTelemetry: currentSchedulerTelemetry,
      });
      const elapsed = performance.now() - t0;

      // Read back disparity and visualize
      const dispData = await runRouteStage(routeRuntime, runDebug, 'output-capture', async () => {
        const disparityBytes = depthResult.C * depthResult.H * depthResult.W * 4;
        const data = await recordRouteTailStep(
          runDebug,
          currentScheduler,
          currentSchedulerTelemetry,
          gpu.device,
          { stage: 'output-capture', step: 'disparity-readback', bytes: disparityBytes },
          () => readBuffer(gpu.device, depthResult.disparityBuf, disparityBytes)
        );

        // Render depth map (channel 0 of 2-channel disparity)
        const depthCanvas = sharpElement('depth-canvas');
        if (depthCanvas) {
          await recordRouteTailStep(
            runDebug,
            currentScheduler,
            currentSchedulerTelemetry,
            gpu.device,
            { stage: 'output-capture', step: 'depth-preview-render', pixels: depthResult.H * depthResult.W },
            async () => {
              const dH = depthResult.H, dW = depthResult.W;
              // Downsample for display if needed
              const maxDisp = 768;
              const dispScale = Math.min(1, maxDisp / Math.max(dH, dW));
              const dispH = Math.round(dH * dispScale);
              const dispW = Math.round(dW * dispScale);
              depthCanvas.width = dispW;
              depthCanvas.height = dispH;
              const ctx = depthCanvas.getContext('2d');
              const imgData = ctx.createImageData(dispW, dispH);

              // Find min/max for normalization (channel 0 only)
              let dMin = Infinity, dMax = -Infinity;
              const cpuChunkItems = currentScheduler.effective?.cpuChunkItems || 0;
              for (let i = 0; i < dH * dW; i++) {
                const v = data[i]; // channel 0
                if (isFinite(v)) {
                  if (v < dMin) dMin = v;
                  if (v > dMax) dMax = v;
                }
                if (cpuChunkItems && (i + 1) % cpuChunkItems === 0) {
                  await recordCpuDutyChunk(
                    runDebug,
                    currentScheduler,
                    currentSchedulerTelemetry,
                    gpu.device,
                    { stage: 'output-capture', step: 'depth-preview-minmax', pixels: dH * dW },
                    i + 1
                  );
                }
              }
              const dRange = dMax - dMin || 1;

              for (let y = 0; y < dispH; y++) {
                for (let x = 0; x < dispW; x++) {
                  // Nearest-neighbor sample from full res
                  const sy = Math.min(Math.floor(y / dispScale), dH - 1);
                  const sx = Math.min(Math.floor(x / dispScale), dW - 1);
                  const v = data[sy * dW + sx]; // channel 0
                  const norm = Math.max(0, Math.min(1, (v - dMin) / dRange));
                  // Turbo-ish colormap for depth
                  const r = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * norm - 3))));
                  const g = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * norm - 2))));
                  const b = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * norm - 1))));
                  const idx = (y * dispW + x) * 4;
                  imgData.data[idx] = r;
                  imgData.data[idx + 1] = g;
                  imgData.data[idx + 2] = b;
                  imgData.data[idx + 3] = 255;
                  const processedPixels = y * dispW + x + 1;
                  if (cpuChunkItems && processedPixels % cpuChunkItems === 0) {
                    await recordCpuDutyChunk(
                      runDebug,
                      currentScheduler,
                      currentSchedulerTelemetry,
                      gpu.device,
                      { stage: 'output-capture', step: 'depth-preview-pixels', pixels: dispH * dispW },
                      processedPixels
                    );
                  }
                }
              }
              ctx.putImageData(imgData, 0, 0);
            }
          );
        }
        return data;
      }, {
        shape: [depthResult.C, depthResult.H, depthResult.W],
        schedulerConfig: currentScheduler,
        schedulerTelemetry: currentSchedulerTelemetry,
      });
      try {
        runDebug.outputs.disparityPlausibility = validateSharpDisparityPlausibility(dispData, {
          channels: depthResult.C,
          height: depthResult.H,
          width: depthResult.W,
        });
      } catch (error) {
        runDebug.outputs.disparityPlausibility = error.disparityPlausibility || null;
        throw error;
      }
      emitProgress(0.78, 'SHARP validated the scene-depth field.', { phase: 'depth-validated' });

      // Run Gaussian prediction pipeline
      if (!gaussianPipeline) {
        gaussianPipeline = new GaussianPipeline(gpu.device);
      }
      setStatus('Running Gaussian prediction...');
      emitProgress(0.79, 'SHARP is predicting Gaussian geometry.', { phase: 'gaussian-decoder' });
      const gaussResult = await runRouteStage(routeRuntime, runDebug, 'gaussian-decoder', () => gaussianPipeline.run(
        spnResult.features, spnResult.featureDims,
        depthResult.disparityBuf, depthResult.H, depthResult.W,
        chw, weights,
        {
          scheduler: currentScheduler,
          telemetry: currentSchedulerTelemetry,
        }
      ), {
        schedulerMode: currentScheduler.effective?.mode,
        gaussianPhaseYieldMs: currentScheduler.effective?.gaussianPhaseYieldMs,
        schedulerConfig: currentScheduler,
        schedulerTelemetry: currentSchedulerTelemetry,
      });

      console.log(`[Main] ${gaussResult.numGaussians} Gaussians predicted (${gaussResult.numLayers} layers × ${gaussResult.H}×${gaussResult.W})`);

      // Compose final Gaussians and generate PLY
      setStatus('Composing Gaussians + PLY export...');
      emitProgress(0.90, 'SHARP is composing the splat artifact.', { phase: 'compose-ply' });

      // Reuse disparity data from depth visualization (avoid redundant GPU readback)
      // Convert image from [-1,1] to [0,1] for initializer
      const img01 = new Float32Array(chw.length);
      const cpuChunkItems = currentScheduler.effective?.cpuChunkItems || 0;
      for (let i = 0; i < chw.length; i++) {
        img01[i] = (chw[i] + 1.0) * 0.5;
        if (cpuChunkItems && (i + 1) % cpuChunkItems === 0) {
          await recordCpuDutyChunk(
            runDebug,
            currentScheduler,
            currentSchedulerTelemetry,
            gpu.device,
            { stage: 'compose-ply', step: 'image-normalize', pixels: chw.length / 3 },
            i + 1
          );
        }
      }

      // Read raw deltas from stored GPU buffers
      const composed = await runRouteStage(routeRuntime, runDebug, 'compose-ply', async () => {
        const geomBytes = 6 * gaussResult.H * gaussResult.W * 4;
        const texBytes = 22 * gaussResult.H * gaussResult.W * 4;
        const geomDeltas = await recordRouteTailStep(
          runDebug,
          currentScheduler,
          currentSchedulerTelemetry,
          gpu.device,
          { stage: 'compose-ply', step: 'geometry-delta-readback', bytes: geomBytes },
          () => readBuffer(gpu.device, gaussianPipeline._geomDeltasBuf, geomBytes, {
            chunkBytes: cpuChunkItems > 0
              ? cpuChunkItems * Float32Array.BYTES_PER_ELEMENT
              : geomBytes,
            onChunk: cpuChunkItems > 0
              ? details => schedulerYield(currentScheduler, gpu.device, currentSchedulerTelemetry, 'route-tail', {
                  stage: 'compose-ply',
                  step: 'geometry-delta-readback-copy',
                  role: 'cpu-materialization-chunk',
                  ...details,
                })
              : null,
          })
        );
        const texDeltas = await recordRouteTailStep(
          runDebug,
          currentScheduler,
          currentSchedulerTelemetry,
          gpu.device,
          { stage: 'compose-ply', step: 'texture-delta-readback', bytes: texBytes },
          () => readBuffer(gpu.device, gaussianPipeline._texDeltasBuf, texBytes, {
            chunkBytes: cpuChunkItems > 0
              ? cpuChunkItems * Float32Array.BYTES_PER_ELEMENT
              : texBytes,
            onChunk: cpuChunkItems > 0
              ? details => schedulerYield(currentScheduler, gpu.device, currentSchedulerTelemetry, 'route-tail', {
                  stage: 'compose-ply',
                  step: 'texture-delta-readback-copy',
                  role: 'cpu-materialization-chunk',
                  ...details,
                })
              : null,
          })
        );

        return recordRouteTailStep(
          runDebug,
          currentScheduler,
          currentSchedulerTelemetry,
          gpu.device,
          { stage: 'compose-ply', step: 'compose-export', pixels: gaussResult.H * gaussResult.W },
          () => composeAndExport(
            dispData, geomDeltas, texDeltas,
            img01, 1536, 1536, gaussResult.H, gaussResult.W,
            bitmap.width, bitmap.height, // original image dims for unprojection
            undefined,
            {
              chunkItems: cpuChunkItems,
              onChunk: cpuChunkItems
                ? chunk => recordCpuDutyChunk(
                  runDebug,
                  currentScheduler,
                  currentSchedulerTelemetry,
                  gpu.device,
                  {
                    stage: 'compose-ply',
                    step: chunk.step,
                    pixels: chunk.totalItems,
                    phaseComplete: chunk.phaseComplete,
                    totalItems: chunk.totalItems,
                    checkpointItems: chunk.checkpointItems,
                    segmentStartProcessedItems: chunk.segmentStartProcessedItems,
                    segmentEndProcessedItems: chunk.segmentEndProcessedItems,
                    granularity: chunk.granularity,
                    intervalStartMs: chunk.intervalStartMs,
                    intervalEndMs: chunk.intervalEndMs,
                    durationMs: chunk.durationMs,
                  },
                  chunk.processedItems,
                )
                : null,
              onInterval: interval => recordObservedRouteTailInterval(
                runDebug,
                currentSchedulerTelemetry,
                { stage: 'compose-ply', ...interval },
              ),
            },
          )
        );
      }, {
        shape: [gaussResult.numGaussians, 14],
        schedulerConfig: currentScheduler,
        schedulerTelemetry: currentSchedulerTelemetry,
      });

      // Create download link
      const downloadLink = sharpElement('download-ply');
      let outputBindEndMs = null;
      if (downloadLink) {
        const url = recordRouteTailInterval(
          runDebug,
          currentSchedulerTelemetry,
          { stage: 'compose-ply', step: 'object-url-create', bytes: composed.plyBlob.size },
          () => URL.createObjectURL(composed.plyBlob),
        );
        recordRouteTailInterval(
          runDebug,
          currentSchedulerTelemetry,
          {
            stage: 'compose-ply',
            step: 'output-bind',
            onMeasured: entry => { outputBindEndMs = entry.intervalEndMs; },
          },
          () => {
            downloadLink.href = url;
            downloadLink.download = 'sharp_gaussians.ply';
            downloadLink.style.display = 'inline-block';
            downloadLink.textContent = `Download PLY (${(composed.plyBlob.size / 1024 / 1024).toFixed(1)} MB, ${(composed.numGaussians / 1000).toFixed(0)}K splats)`;
          },
        );
      }

      const inferenceFinalizeStartMs = outputBindEndMs ?? performance.now();
      const elapsed2 = performance.now() - t0;
      sharpRuntimeGlobal.__sharpContentionProbe?.markInferenceEnd?.(currentSchedulerTelemetry.runId);
      const inferenceFinalizeEndMs = performance.now();
      recordObservedRouteTailInterval(runDebug, currentSchedulerTelemetry, {
        stage: 'compose-ply',
        step: 'inference-window-finalize',
        role: 'localization-envelope',
        intervalStartMs: inferenceFinalizeStartMs,
        intervalEndMs: inferenceFinalizeEndMs,
        durationMs: inferenceFinalizeEndMs - inferenceFinalizeStartMs,
      });
      runDebug.schedulerTelemetry = await schedulerTelemetrySnapshotCooperatively(currentSchedulerTelemetry, 'verified');
      sharpRuntimeGlobal.__SHARP_LAST_RUN_TELEMETRY__ = runDebug.schedulerTelemetry;
      runDebug.schedulerApplication = routeRuntime.schedulerSnapshot();
      runDebug.commandDutyReport = routeRuntime.finishCommandDuties();
      runDebug.hostPhaseReport = routeRuntime.finishHostPhases();
      runDebug.foregroundOpportunityReport = routeRuntime.finishForegroundOpportunities();
      spnResult.hasNaN = false;
      spnResult.numGaussians = composed.numGaussians;
      runDebug.inferenceElapsedMs = elapsed2;
      finishRouteRun(runDebug, 'real', {
        numGaussians: composed.numGaussians,
        plyAvailable: Boolean(downloadLink?.href),
        depthShape: [depthResult.H, depthResult.W],
        splatShape: [composed.numGaussians, 14],
      });
      runDebug.runtimeProfile = finishSharpRouteRuntimeProfile(routeRuntime);
      try {
        const receipt = await createExecutionRouteReceipt({
          blob,
          bitmap,
          depthResult,
          dispData,
          composed,
          runDebug,
        });
        runDebug.route.receipt = receipt;
        runDebug.route.evidence = classifyWebGpuRouteReceiptEvidence(receipt, {
          expectedRouteId: SHARP_IMAGE_TO_SPLAT_ROUTE_ID,
        });
      } catch (receiptError) {
        runDebug.route.receiptError = receiptError?.message || String(receiptError);
        console.error('[Main] Failed to build SHARP route receipt:', receiptError);
      }
      setStatus('');
      showResults(spnResult, elapsed2, 'spn');
      emitProgress(0.93, 'SHARP produced a validated splat artifact.', { phase: 'complete' });
      return {
        ok: true,
        mode: 'spn',
        plyBlob: composed.plyBlob,
        numGaussians: composed.numGaussians,
        runDebug,
      };

    } else {
      if (!backbone) {
        backbone = new SharpBackbone(gpu.device);
        backbone.init(weights);
      }

      setStatus('Running ViT-Large backbone...');
      const t0 = performance.now();
      const result = await backbone.run(blob);
      const elapsed = performance.now() - t0;
      result.schedulerTelemetry = await schedulerTelemetrySnapshotCooperatively(currentSchedulerTelemetry, 'verified');
      sharpRuntimeGlobal.__SHARP_LAST_RUN_TELEMETRY__ = result.schedulerTelemetry;

      finishRoutePhase(runDebug, 'backbone', t0);
      runDebug.inferenceElapsedMs = elapsed;
      finishRouteRun(runDebug, 'partial', {
        numGaussians: null,
        plyAvailable: false,
      });
      setStatus('');
      showResults(result, elapsed, 'backbone');
      return {
        ok: true,
        mode: 'backbone',
        plyBlob: null,
        runDebug,
      };
    }

  } catch (err) {
    sharpRuntimeGlobal.__sharpContentionProbe?.markInferenceEnd?.(currentSchedulerTelemetry.runId);
    if (currentSchedulerTelemetry) {
      currentSchedulerTelemetry.error = err.message;
      sharpRuntimeGlobal.__SHARP_LAST_RUN_TELEMETRY__ = await schedulerTelemetrySnapshotCooperatively(currentSchedulerTelemetry, 'failed');
    }
    runDebug.status = 'error';
    runDebug.error = err?.message || String(err);
    finishRouteRun(runDebug, 'error', runDebug.outputs || {});
    setError(err.message);
    console.error(err);
    if (options.throwOnError === true) {
      err.sharpRunDebug = runDebug;
      throw err;
    }
    return { ok: false, error: err?.message || String(err), runDebug };
  }
}

globalThis.__kaminosRunSharpImageToSplat = runSharpImageToSplat;
