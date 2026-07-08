/**
 * SHARP-WebGPU — Main entry point
 *
 * Apple SHARP (single-image 3D Gaussian Splat generation) in WebGPU compute.
 *
 * Current status: backbone + SPN encoder.
 * Full pipeline (monodepth decoder, Gaussian decoder, 3DGS output) not yet implemented.
 */

import { initGPU, readBuffer } from './lib/gpu.js';
import { loadWeights } from './lib/weights.js';
import { SharpBackbone } from './lib/backbone.js';
import { SlidingPyramidNetwork } from './lib/spn.js';
import { MonodepthDecoder } from './lib/monodepth.js';
import { GaussianPipeline } from './lib/gaussian_decoder.js';
import { composeAndExport } from './lib/compose.js';
import {
  createSharpRunTelemetry,
  parseSharpSchedulerConfig,
  schedulerYield,
  schedulerTelemetrySnapshot,
} from './lib/scheduler.js';
import {
  createSharpRouteRuntime,
  finishSharpRouteRuntimeProfile,
} from './lib/route_runtime.js';
import {
  SHARP_IMAGE_TO_SPLAT_ROUTE_ID,
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

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('error');
const outputEl = document.getElementById('output');
const resultsEl = document.getElementById('results');
const sharpRouteDefinition = createSharpImageToSplatRouteDefinition({
  kernel: {
    kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
    profile: 'spn-dinov2l16-monodepth-gaussian-ply',
    commit: 'sharp-webgpu-browser-runtime',
  },
});

window.__sharpDebug = {
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
    elapsedMs: runDebug.inferenceElapsedMs,
    outputs: runDebug.outputs,
  };
  const metadataHash = await sha256Hex(metadata);

  return createSharpImageToSplatRouteReceipt({
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
}

function finishRoutePhase(run, name, startMs) {
  run.phases.push({
    name,
    ms: performance.now() - startMs,
  });
}

async function runRouteStage(routeRuntime, run, name, fn, metadata = {}) {
  const result = await routeRuntime.runStage(name, fn, {
    routeStage: name,
    ...metadata,
  });
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

function finishRouteRun(run, status, outputs = {}) {
  run.status = status;
  run.endMs = performance.now();
  run.elapsedMs = run.endMs - run.startMs;
  run.endedAt = new Date().toISOString();
  run.outputs = outputs;
}

function setStatus(msg) {
  statusEl.textContent = msg;
  errorEl.style.display = 'none';
}

function setError(msg) {
  errorEl.textContent = msg;
  errorEl.style.display = 'block';
  statusEl.textContent = '';
}

function showResults(result, elapsed, mode) {
  document.getElementById('r-model').textContent = 'DINOv2 ViT-Large (dinov2l16_384)';
  document.getElementById('r-weights').textContent = `${weightsLoadedMB} MB (fp16)`;
  document.getElementById('r-patch').textContent = '16x16';

  if (mode === 'spn') {
    document.getElementById('r-title').textContent = 'Full Route Results';
    document.getElementById('r-time-label').textContent = 'Full route time';
    document.getElementById('r-grid').textContent = `SPN: 35 patches (5x5 + 3x3 + 1x1)`;
    const gaussStr = result.numGaussians ? ` → ${(result.numGaussians / 1000).toFixed(0)}K Gaussians` : '';
    document.getElementById('r-features').textContent = `${result.featureDims.length} multi-res outputs${gaussStr}`;
  } else {
    document.getElementById('r-title').textContent = 'Backbone Results';
    document.getElementById('r-time-label').textContent = 'Backbone time';
    document.getElementById('r-grid').textContent = `${result.tokenH}x${result.tokenW} = ${result.numPatches} patches + 1 CLS`;
    document.getElementById('r-features').textContent = `${result.intermediateFeatures.length} layers`;
  }

  document.getElementById('r-time').textContent = `${elapsed.toFixed(0)} ms`;

  const validEl = document.getElementById('r-valid');
  if (validEl) {
    if (result.hasNaN) {
      validEl.textContent = 'INVALID (NaN/Inf in output)';
      validEl.style.color = '#f66';
    } else {
      validEl.textContent = 'OK';
      validEl.style.color = '#6f6';
    }
  }

  resultsEl.classList.add('visible');
}

// --- Drop zone ---
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleBlob(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleBlob(fileInput.files[0]);
});

// --- Sample image clicks ---
document.querySelectorAll('.sample-thumb').forEach(thumb => {
  thumb.addEventListener('click', async () => {
    const url = thumb.dataset.full;
    try {
      setStatus('Loading sample image...');
      const resp = await fetch(url);
      const blob = await resp.blob();
      await handleBlob(blob);
    } catch (err) {
      setError(`Failed to load sample: ${err.message}`);
    }
  });
});

async function handleBlob(blob) {
  const runMode = (document.getElementById('use-spn')?.checked ?? false) ? 'spn' : 'backbone';
  const runDebug = createRouteRunDebug(runMode);
  const currentScheduler = parseSharpSchedulerConfig();
  const currentSchedulerTelemetry = createSharpRunTelemetry(currentScheduler, { mode: runMode });
  runDebug.sharpScheduler = currentScheduler;
  window.__sharpDebug.lastRun = runDebug;
  window.__SHARP_LAST_RUN_TELEMETRY__ = schedulerTelemetrySnapshot(currentSchedulerTelemetry, 'running');

  try {
    setStatus('Initializing WebGPU...');
    if (!gpu) {
      gpu = await initGPU();
    }
    const routeRuntime = await createSharpRouteRuntime(gpu, {
      routeDefinition: sharpRouteDefinition,
      browser: navigator.userAgent,
      now: () => performance.now(),
    });

    // Show input preview (preserve aspect ratio)
    setStatus('Loading image...');
    const bitmap = await createImageBitmap(blob);
    const inputCanvas = document.getElementById('input-canvas');
    const maxDisplay = 384;
    const scale = Math.min(maxDisplay / bitmap.width, maxDisplay / bitmap.height);
    inputCanvas.width = Math.round(bitmap.width * scale);
    inputCanvas.height = Math.round(bitmap.height * scale);
    const ctx = inputCanvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0, inputCanvas.width, inputCanvas.height);
    outputEl.classList.add('visible');

    if (!weights) {
      setStatus('Loading SHARP weights (~1.25 GB, first load only)...');
      weights = await loadWeights(gpu.device, '/weights.bin', (received, total) => {
        const mb = (received / 1024 / 1024).toFixed(0);
        weightsLoadedMB = mb;
        const totalMb = total ? (total / 1024 / 1024).toFixed(0) : '?';
        setStatus(`Loading weights: ${mb} / ${totalMb} MB`);
      });
    }

    // Use SPN for full pipeline, backbone for quick smoke
    const useSPN = runMode === 'spn';

    if (useSPN) {
      if (!spn) {
        spn = new SlidingPyramidNetwork(gpu.device);
        spn.init(weights);
      }

      setStatus('Running SPN (35 ViT passes, may take 15-30s)...');

      // Resize to 1536x1536 and normalize to [-1, 1] CHW
      const spnSize = 1536;
      const spnBitmap = await createImageBitmap(blob, { resizeWidth: spnSize, resizeHeight: spnSize });
      const spnCanvas = new OffscreenCanvas(spnSize, spnSize);
      const spnCtx = spnCanvas.getContext('2d');
      spnCtx.drawImage(spnBitmap, 0, 0);
      const spnImageData = spnCtx.getImageData(0, 0, spnSize, spnSize);

      const chw = new Float32Array(3 * spnSize * spnSize);
      for (let y = 0; y < spnSize; y++) {
        for (let x = 0; x < spnSize; x++) {
          const srcIdx = (y * spnSize + x) * 4;
          const dstBase = y * spnSize + x;
          chw[0 * spnSize * spnSize + dstBase] = spnImageData.data[srcIdx] / 127.5 - 1.0;
          chw[1 * spnSize * spnSize + dstBase] = spnImageData.data[srcIdx + 1] / 127.5 - 1.0;
          chw[2 * spnSize * spnSize + dstBase] = spnImageData.data[srcIdx + 2] / 127.5 - 1.0;
        }
      }

      window.__sharpContentionProbe?.markInferenceStart?.();
      const t0 = performance.now();
      const spnResult = await runRouteStage(routeRuntime, runDebug, 'spn', () => spn.run(chw, {
        scheduler: currentScheduler,
        telemetry: currentSchedulerTelemetry,
      }), {
        schedulerMode: currentScheduler.effective?.mode,
      });

      // Run monodepth decoder
      if (!monodepth) {
        monodepth = new MonodepthDecoder(gpu.device);
      }
      setStatus('Running monodepth decoder...');
      const depthResult = await runRouteStage(routeRuntime, runDebug, 'monodepth', () => (
        monodepth.run(spnResult.features, spnResult.featureDims, weights, {
          scheduler: currentScheduler,
          telemetry: currentSchedulerTelemetry,
        })
      ), {
        schedulerMode: currentScheduler.effective?.mode,
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
        const depthCanvas = document.getElementById('depth-canvas');
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
              for (let i = 0; i < dH * dW; i++) {
                const v = data[i]; // channel 0
                if (isFinite(v)) {
                  if (v < dMin) dMin = v;
                  if (v > dMax) dMax = v;
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
                }
              }
              ctx.putImageData(imgData, 0, 0);
            }
          );
        }
        return data;
      }, {
        shape: [depthResult.C, depthResult.H, depthResult.W],
      });

      // Run Gaussian prediction pipeline
      if (!gaussianPipeline) {
        gaussianPipeline = new GaussianPipeline(gpu.device);
      }
      setStatus('Running Gaussian prediction...');
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
      });

      console.log(`[Main] ${gaussResult.numGaussians} Gaussians predicted (${gaussResult.numLayers} layers × ${gaussResult.H}×${gaussResult.W})`);

      // Compose final Gaussians and generate PLY
      setStatus('Composing Gaussians + PLY export...');

      // Reuse disparity data from depth visualization (avoid redundant GPU readback)
      // Convert image from [-1,1] to [0,1] for initializer
      const img01 = new Float32Array(chw.length);
      for (let i = 0; i < chw.length; i++) img01[i] = (chw[i] + 1.0) * 0.5;

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
          () => readBuffer(gpu.device, gaussianPipeline._geomDeltasBuf, geomBytes)
        );
        const texDeltas = await recordRouteTailStep(
          runDebug,
          currentScheduler,
          currentSchedulerTelemetry,
          gpu.device,
          { stage: 'compose-ply', step: 'texture-delta-readback', bytes: texBytes },
          () => readBuffer(gpu.device, gaussianPipeline._texDeltasBuf, texBytes)
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
            bitmap.width, bitmap.height  // original image dims for unprojection
          )
        );
      }, {
        shape: [gaussResult.numGaussians, 14],
      });

      // Create download link
      const downloadLink = document.getElementById('download-ply');
      if (downloadLink) {
        const url = URL.createObjectURL(composed.plyBlob);
        downloadLink.href = url;
        downloadLink.download = 'sharp_gaussians.ply';
        downloadLink.style.display = 'inline-block';
        downloadLink.textContent = `Download PLY (${(composed.plyBlob.size / 1024 / 1024).toFixed(1)} MB, ${(composed.numGaussians / 1000).toFixed(0)}K splats)`;
      }

      const elapsed2 = performance.now() - t0;
      window.__sharpContentionProbe?.markInferenceEnd?.();
      runDebug.schedulerTelemetry = schedulerTelemetrySnapshot(currentSchedulerTelemetry, 'verified');
      window.__SHARP_LAST_RUN_TELEMETRY__ = runDebug.schedulerTelemetry;
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

    } else {
      if (!backbone) {
        backbone = new SharpBackbone(gpu.device);
        backbone.init(weights);
      }

      setStatus('Running ViT-Large backbone...');
      const t0 = performance.now();
      const result = await backbone.run(blob);
      const elapsed = performance.now() - t0;
      result.schedulerTelemetry = schedulerTelemetrySnapshot(currentSchedulerTelemetry, 'verified');
      window.__SHARP_LAST_RUN_TELEMETRY__ = result.schedulerTelemetry;

      finishRoutePhase(runDebug, 'backbone', t0);
      runDebug.inferenceElapsedMs = elapsed;
      finishRouteRun(runDebug, 'partial', {
        numGaussians: null,
        plyAvailable: false,
      });
      setStatus('');
      showResults(result, elapsed, 'backbone');
    }

  } catch (err) {
    window.__sharpContentionProbe?.markInferenceEnd?.();
    if (currentSchedulerTelemetry) {
      currentSchedulerTelemetry.error = err.message;
      window.__SHARP_LAST_RUN_TELEMETRY__ = schedulerTelemetrySnapshot(currentSchedulerTelemetry, 'failed');
    }
    runDebug.status = 'error';
    runDebug.error = err?.message || String(err);
    finishRouteRun(runDebug, 'error', runDebug.outputs || {});
    setError(err.message);
    console.error(err);
  }
}
