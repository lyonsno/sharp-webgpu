/**
 * compose.js — Initializer + Composer + PLY export for SHARP-WebGPU.
 *
 * Takes monodepth disparity + prediction deltas + input image and produces
 * final 3D Gaussian Splat parameters, then exports as .ply for standard viewers.
 *
 * All CPU math — no GPU shaders needed for this stage.
 */

// Default SHARP params (from PredictorParams)
const PARAMS = {
  numLayers: 2,
  stride: 2,
  scaleFactor: 1.0,
  disparityFactor: 1.0,
  baseDepth: 10.0,
  normalizeDepth: true,
  colorOption: 'all_layers',
  baseScaleOnPredictedMean: true,
  deltaFactor: { xy: 0.001, z: 0.001, color: 0.1, opacity: 1.0, scale: 1.0, quaternion: 1.0 },
  minScale: 0.0,
  maxScale: 10.0,
};

function softplus(x) { return x > 20 ? x : Math.log(1 + Math.exp(x)); }
function inverseSoftplus(x) { return x > 20 ? x : Math.log(Math.exp(Math.max(x, 1e-6)) - 1); }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function inverseSigmoid(x) {
  const c = Math.max(1e-6, Math.min(1 - 1e-6, x));
  return Math.log(c / (1 - c));
}
function sRGB2linear(x) { return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4); }
function linear2sRGB(x) { return x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055; }

export function validateSharpDisparityPlausibility(dispData, dimensions = {}) {
  const channels = Number(dimensions.channels || 0);
  const height = Number(dimensions.height || 0);
  const width = Number(dimensions.width || 0);
  const expectedValues = channels * height * width;
  if (!(dispData instanceof Float32Array) || expectedValues <= 0 || dispData.length !== expectedValues) {
    throw new Error(`invalid SHARP disparity shape: expected ${expectedValues || 'positive'} float values, got ${dispData?.length ?? 'none'}`);
  }

  let finiteCount = 0;
  let positiveCount = 0;
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumSquares = 0;
  for (const value of dispData) {
    if (!Number.isFinite(value)) continue;
    finiteCount += 1;
    if (value > 1e-4) positiveCount += 1;
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    sumSquares += value * value;
  }
  const positiveFraction = positiveCount / expectedValues;
  const range = max - min;
  const mean = finiteCount ? sum / finiteCount : NaN;
  const std = finiteCount ? Math.sqrt(Math.max(0, sumSquares / finiteCount - mean * mean)) : NaN;
  const report = {
    schema: 'sharp-webgpu.disparity-plausibility.v0',
    status: 'plausible',
    shape: [channels, height, width],
    values: expectedValues,
    finiteCount,
    positiveCount,
    positiveFraction,
    min,
    max,
    range,
    mean,
    std,
  };
  const collapsed = finiteCount !== expectedValues
    || positiveFraction < 0.01
    || max <= 1e-3
    || range <= 1e-5;
  if (collapsed) {
    report.status = 'collapsed';
    const error = new Error(
      `collapsed disparity: positive ${(positiveFraction * 100).toFixed(3)}%, range ${Number.isFinite(range) ? range.toExponential(3) : 'non-finite'}, max ${Number.isFinite(max) ? max.toExponential(3) : 'non-finite'}`,
    );
    error.disparityPlausibility = report;
    throw error;
  }
  return report;
}

/**
 * Compose final Gaussians from base values + deltas.
 *
 * @param {Float32Array} dispData - [2, H, W] disparity from monodepth
 * @param {Float32Array} geomDeltas - [6, outH, outW] geometry deltas (3 mean × 2 layers)
 * @param {Float32Array} texDeltas - [22, outH, outW] texture deltas (11 × 2 layers)
 * @param {Float32Array} img01 - [3, H, W] image in [0, 1]
 * @param {number} imgH - image height (1536)
 * @param {number} imgW - image width (1536)
 * @param {number} outH - output height (768)
 * @param {number} outW - output width (768)
 * @param {number} origW - original image width (for unprojection)
 * @param {number} origH - original image height (for unprojection)
 * @param {number} [focalPx] - focal length in pixels (default: max(origW, origH))
 * @param {{ chunkItems?: number, onChunk?: (chunk: object) => Promise<void>, onInterval?: (interval: object) => void, plyAssemblyMode?: 'main-thread'|'worker', plyWorkerFactory?: () => Worker }} [options]
 * @returns {Promise<{ plyBlob: Blob, plySha256: string|null, numGaussians: number, plyAssemblyMode: string }>}
 */
export async function composeAndExport(dispData, geomDeltas, texDeltas, img01, imgH, imgW, outH, outW, origW, origH, focalPx, options = {}) {
  // Focal length default: max dimension (matches reference load_rgb default)
  if (!focalPx) focalPx = Math.max(origW || imgW, origH || imgH);
  if (!origW) origW = imgW;
  if (!origH) origH = imgH;
  const { numLayers, stride, scaleFactor, disparityFactor, normalizeDepth, baseDepth,
    baseScaleOnPredictedMean, deltaFactor, minScale, maxScale } = PARAMS;
  const chunkItems = Number.isFinite(options.chunkItems) && options.chunkItems > 0
    ? Math.floor(options.chunkItems)
    : 0;
  const onChunk = typeof options.onChunk === 'function' ? options.onChunk : null;
  const onInterval = typeof options.onInterval === 'function' ? options.onInterval : null;
  const emitInterval = (step, intervalStartMs, details = {}) => {
    const intervalEndMs = performance.now();
    onInterval?.({
      step,
      intervalStartMs,
      intervalEndMs,
      durationMs: intervalEndMs - intervalStartMs,
      ...details,
    });
  };
  const phaseCheckpoint = async (step, totalItems) => {
    if (!chunkItems || !onChunk) return;
    await onChunk({ step, processedItems: totalItems, totalItems, phaseComplete: true });
  };

  const HW = imgH * imgW;
  const baseH = imgH / stride;  // 768
  const baseW = imgW / stride;  // 768
  const baseHW = baseH * baseW;

  console.log('[Compose] Building base Gaussians...');

  // --- Step 1: Depth normalization ---
  // depth = disparityFactor / clamp(disparity, 1e-4, 1e4)
  // dispData has 2 channels — use both for 2-layer depth
  const depthNormalizeStartMs = performance.now();
  const depth = new Float32Array(2 * HW);
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < HW; i++) {
      const disp = Math.max(1e-4, Math.min(1e4, dispData[c * HW + i]));
      depth[c * HW + i] = disparityFactor / disp;
    }
  }
  emitInterval('depth-normalize', depthNormalizeStartMs, { items: 2 * HW });
  if (chunkItems && onChunk) await phaseCheckpoint('depth-normalize', 2 * HW);

  let globalScale = 1.0;
  if (normalizeDepth) {
    // Rescale depth so min = 1.0
    const depthMinStartMs = performance.now();
    let depthMin = Infinity;
    for (let i = 0; i < 2 * HW; i++) {
      if (depth[i] < depthMin) depthMin = depth[i];
    }
    emitInterval('depth-min', depthMinStartMs, { items: 2 * HW });
    if (chunkItems && onChunk) await phaseCheckpoint('depth-min', 2 * HW);
    const depthRescaleStartMs = performance.now();
    const depthFactor = 1.0 / (depthMin + 1e-6);
    for (let i = 0; i < 2 * HW; i++) {
      depth[i] = Math.min(depth[i] * depthFactor, 100);
    }
    emitInterval('depth-rescale', depthRescaleStartMs, { items: 2 * HW });
    if (chunkItems && onChunk) await phaseCheckpoint('depth-rescale', 2 * HW);
    globalScale = 1.0 / depthFactor;
  }

  // --- Step 2: Base values ---
  // Base XY in NDC: [-1, 1] grid at stride=2
  // Base inverse Z: from depth via max_pool2d (surface_min → use max of 1/depth)
  // disparity[layer] at base resolution via max_pool(1/depth, stride)
  const baseDisparityStartMs = performance.now();
  const baseDisparity = new Float32Array(numLayers * baseHW);
  for (let layer = 0; layer < numLayers; layer++) {
    const depthChannel = layer === 0 ? 0 : 1;
    for (let by = 0; by < baseH; by++) {
      for (let bx = 0; bx < baseW; bx++) {
        // Max pool (of 1/depth = disparity) over stride×stride window
        let maxDisp = -Infinity;
        for (let sy = 0; sy < stride; sy++) {
          for (let sx = 0; sx < stride; sx++) {
            const iy = by * stride + sy;
            const ix = bx * stride + sx;
            const d = depth[depthChannel * HW + iy * imgW + ix];
            const disp = 1.0 / d;
            if (disp > maxDisp) maxDisp = disp;
          }
        }
        baseDisparity[layer * baseHW + by * baseW + bx] = maxDisp;
      }
    }
  }
  emitInterval('base-disparity', baseDisparityStartMs, { items: numLayers * baseHW });
  if (chunkItems && onChunk) await phaseCheckpoint('base-disparity', numLayers * baseHW);

  // Base XY NDC
  const baseGridStartMs = performance.now();
  const baseX = new Float32Array(baseHW);
  const baseY = new Float32Array(baseHW);
  for (let by = 0; by < baseH; by++) {
    for (let bx = 0; bx < baseW; bx++) {
      baseX[by * baseW + bx] = 2 * (bx * stride + 0.5 * stride) / imgW - 1.0;
      baseY[by * baseW + bx] = 2 * (by * stride + 0.5 * stride) / imgH - 1.0;
    }
  }
  emitInterval('base-grid', baseGridStartMs, { items: baseHW });
  if (chunkItems && onChunk) await phaseCheckpoint('base-grid', baseHW);

  // Base scales
  const dispScaleFactor = 2 * scaleFactor * stride / imgW;

  // Base colors: avg_pool'd image (all_layers)
  const baseColorStartMs = performance.now();
  const baseColors = new Float32Array(3 * baseHW);
  for (let c = 0; c < 3; c++) {
    for (let by = 0; by < baseH; by++) {
      for (let bx = 0; bx < baseW; bx++) {
        let sum = 0;
        for (let sy = 0; sy < stride; sy++) {
          for (let sx = 0; sx < stride; sx++) {
            sum += img01[c * HW + (by * stride + sy) * imgW + (bx * stride + sx)];
          }
        }
        baseColors[c * baseHW + by * baseW + bx] = sum / (stride * stride);
      }
    }
  }
  emitInterval('base-color', baseColorStartMs, { items: 3 * baseHW });
  if (chunkItems && onChunk) await phaseCheckpoint('base-color', 3 * baseHW);

  // --- Step 3: Compose Gaussians ---
  console.log('[Compose] Composing Gaussians...');
  const numGaussians = numLayers * baseHW;
  // PLY fields: x,y,z, f_dc_0/1/2, opacity, scale_0/1/2, rot_0/1/2/3 = 14 floats
  const plyDataAllocationStartMs = performance.now();
  const plyData = new Float32Array(numGaussians * 14);
  const plyDataAllocationEndMs = performance.now();
  onInterval?.({
    step: 'ply-data-allocation',
    intervalStartMs: plyDataAllocationStartMs,
    intervalEndMs: plyDataAllocationEndMs,
    durationMs: plyDataAllocationEndMs - plyDataAllocationStartMs,
    bytes: plyData.byteLength,
  });

  // Scale activation constants
  const activationSetupStartMs = performance.now();
  const scaleConstA = (maxScale - minScale) / (1 - minScale) / (maxScale - 1);
  const scaleConstB = inverseSigmoid((1.0 - minScale) / (maxScale - minScale));

  const outHW = outH * outW;
  let nextGaussianCheckpoint = chunkItems;
  let lastGaussianCheckpoint = 0;
  let gaussianSegmentStartItems = 0;
  const activationSetupEndMs = performance.now();
  onInterval?.({
    step: 'gaussian-activation-setup',
    intervalStartMs: activationSetupStartMs,
    intervalEndMs: activationSetupEndMs,
    durationMs: activationSetupEndMs - activationSetupStartMs,
  });
  let gaussianWorkStartMs = performance.now();

  for (let layer = 0; layer < numLayers; layer++) {
    for (let py = 0; py < baseH; py++) {
      for (let px = 0; px < baseW; px++) {
        const baseIdx = py * baseW + px;
        const gIdx = (layer * baseHW + baseIdx) * 14;

        // Delta indices: [C, numLayers, H, W] stored as [C*numLayers, H, W]
        // channel c, layer l → (c * numLayers + l) * outHW + baseIdx
        const dg = (c, l) => geomDeltas[(c * numLayers + l) * outHW + baseIdx];
        const dt = (c, l) => texDeltas[(c * numLayers + l) * outHW + baseIdx];

        // --- Mean activation ---
        const bx = baseX[baseIdx];
        const by = baseY[baseIdx];
        const bInvZ = baseDisparity[layer * baseHW + baseIdx];

        const dx = deltaFactor.xy * dg(0, layer);
        const dy = deltaFactor.xy * dg(1, layer);
        const dz = deltaFactor.z * dg(2, layer);

        const xx = bx + dx;
        const yy = by + dy;
        const invZZ = softplus(inverseSoftplus(bInvZ) + dz);
        const zz = 1.0 / (invZZ + 1e-3);

        let meanX = zz * xx;
        let meanY = zz * yy;
        let meanZ = zz;

        // --- Scale activation ---
        const bScale = (1.0 / baseDisparity[layer * baseHW + baseIdx]) * dispScaleFactor;
        const adjustedBaseScale = baseScaleOnPredictedMean ? bScale * bInvZ * meanZ : bScale;

        const scales = [];
        for (let s = 0; s < 3; s++) {
          const delta = deltaFactor.scale * dt(s, layer);
          const scaleFact = (maxScale - minScale) * sigmoid(scaleConstA * delta + scaleConstB) + minScale;
          scales.push(adjustedBaseScale * scaleFact);
        }

        // --- Quaternion activation ---
        const qw = 1.0 + deltaFactor.quaternion * dt(3, layer);
        const qx = 0.0 + deltaFactor.quaternion * dt(4, layer);
        const qy = 0.0 + deltaFactor.quaternion * dt(5, layer);
        const qz = 0.0 + deltaFactor.quaternion * dt(6, layer);

        // --- Color activation (sigmoid) ---
        const colors = [];
        for (let c = 0; c < 3; c++) {
          const baseCol = Math.max(0.01, Math.min(0.99, baseColors[c * baseHW + baseIdx]));
          const col = sigmoid(inverseSigmoid(baseCol) + deltaFactor.color * dt(7 + c, layer));
          // Convert to linearRGB
          colors.push(sRGB2linear(col));
        }

        // --- Opacity activation (sigmoid) ---
        const baseOpacity = Math.min(1.0 / numLayers, 0.5);
        const opacity = sigmoid(inverseSigmoid(baseOpacity) + deltaFactor.opacity * dt(10, layer));

        // Apply global scale (NDC → metric)
        meanX *= globalScale;
        meanY *= globalScale;
        meanZ *= globalScale;
        const sv0 = scales[0] * globalScale;
        const sv1 = scales[1] * globalScale;
        const sv2 = scales[2] * globalScale;

        // --- NDC-to-world unprojection ---
        // Reference: unproject_gaussians() in gaussians.py
        // unprojection = inv(ndc_matrix @ intrinsics_resized @ extrinsics)
        // For identity extrinsics and square internal image (1536x1536):
        //   scale_x = origW / (2 * focalPx)
        //   scale_y = origH / (2 * focalPx)
        //   scale_z = 1.0
        const unprojX = origW / (2 * focalPx);
        const unprojY = origH / (2 * focalPx);
        const worldX = meanX * unprojX;
        const worldY = meanY * unprojY;
        const worldZ = meanZ;
        const worldSV0 = sv0 * unprojX;
        const worldSV1 = sv1 * unprojY;
        const worldSV2 = sv2; // z scale unchanged

        // --- Write PLY fields ---
        // For standard 3DGS PLY: xyz, f_dc (SH0), opacity (logit), scale (log), quaternion
        const SH0_COEFF = Math.sqrt(1.0 / (4 * Math.PI));

        plyData[gIdx + 0] = worldX;                         // x
        plyData[gIdx + 1] = worldY;                         // y
        plyData[gIdx + 2] = worldZ;                         // z
        plyData[gIdx + 3] = (linear2sRGB(colors[0]) - 0.5) / SH0_COEFF;  // f_dc_0
        plyData[gIdx + 4] = (linear2sRGB(colors[1]) - 0.5) / SH0_COEFF;  // f_dc_1
        plyData[gIdx + 5] = (linear2sRGB(colors[2]) - 0.5) / SH0_COEFF;  // f_dc_2
        plyData[gIdx + 6] = inverseSigmoid(Math.max(1e-6, Math.min(1 - 1e-6, opacity)));  // opacity logit
        plyData[gIdx + 7] = Math.log(Math.max(1e-10, worldSV0));  // scale_0
        plyData[gIdx + 8] = Math.log(Math.max(1e-10, worldSV1));  // scale_1
        plyData[gIdx + 9] = Math.log(Math.max(1e-10, worldSV2));  // scale_2
        plyData[gIdx + 10] = qw;  // rot_0
        plyData[gIdx + 11] = qx;  // rot_1
        plyData[gIdx + 12] = qy;  // rot_2
        plyData[gIdx + 13] = qz;  // rot_3
      }
      const processedGaussians = layer * baseHW + (py + 1) * baseW;
      if (chunkItems && onChunk && processedGaussians >= nextGaussianCheckpoint) {
        const intervalEndMs = performance.now();
        await onChunk({
          step: 'gaussian-compose',
          processedItems: processedGaussians,
          totalItems: numGaussians,
          checkpointItems: nextGaussianCheckpoint,
          segmentStartProcessedItems: gaussianSegmentStartItems,
          segmentEndProcessedItems: processedGaussians,
          granularity: 'row-batched',
          phaseComplete: processedGaussians === numGaussians,
          intervalStartMs: gaussianWorkStartMs,
          intervalEndMs,
          durationMs: intervalEndMs - gaussianWorkStartMs,
        });
        lastGaussianCheckpoint = processedGaussians;
        gaussianSegmentStartItems = processedGaussians;
        while (nextGaussianCheckpoint <= processedGaussians) nextGaussianCheckpoint += chunkItems;
        gaussianWorkStartMs = performance.now();
      }
    }
  }
  if (chunkItems && onChunk && lastGaussianCheckpoint < numGaussians) {
    const intervalEndMs = performance.now();
    await onChunk({
      step: 'gaussian-compose',
      processedItems: numGaussians,
      totalItems: numGaussians,
      checkpointItems: numGaussians,
      segmentStartProcessedItems: gaussianSegmentStartItems,
      segmentEndProcessedItems: numGaussians,
      granularity: 'row-batched',
      phaseComplete: true,
      intervalStartMs: gaussianWorkStartMs,
      intervalEndMs,
      durationMs: intervalEndMs - gaussianWorkStartMs,
    });
  }

  console.log(`[Compose] ${numGaussians} Gaussians composed`);

  // --- Step 4: Write PLY ---
  console.log('[Compose] Writing PLY...');
  const plyAssemblyStartMs = performance.now();
  const plyAssemblyMode = options.plyAssemblyMode ?? 'main-thread';
  const executionThread = plyAssemblyMode === 'worker' ? 'worker' : 'main';
  let plyBlob;
  let plySha256 = null;
  try {
    plyBlob = await writePLYAsync(plyData, numGaussians, origW, origH, focalPx, {
      mode: plyAssemblyMode,
      workerFactory: options.plyWorkerFactory,
      requireSha256: plyAssemblyMode === 'worker',
      onArtifactIdentity: identity => {
        plySha256 = identity.sha256;
      },
    });
  } catch (error) {
    const plyAssemblyEndMs = performance.now();
    onInterval?.({
      step: 'ply-blob-assembly',
      status: 'failed',
      assemblyMode: plyAssemblyMode,
      executionThread,
      lastTrustworthyStep: 'gaussian-compose',
      intervalStartMs: plyAssemblyStartMs,
      intervalEndMs: plyAssemblyEndMs,
      durationMs: plyAssemblyEndMs - plyAssemblyStartMs,
      error: {
        name: error?.name || 'Error',
        message: error?.message || String(error),
      },
    });
    throw error;
  }
  const plyAssemblyEndMs = performance.now();
  onInterval?.({
    step: 'ply-blob-assembly',
    status: 'completed',
    intervalStartMs: plyAssemblyStartMs,
    intervalEndMs: plyAssemblyEndMs,
    durationMs: plyAssemblyEndMs - plyAssemblyStartMs,
    bytes: plyBlob.size,
    assemblyMode: plyAssemblyMode,
    executionThread,
    artifactIdentity: plySha256 ? 'worker-sha256' : 'not-collected',
  });

  return { plyBlob, plySha256, numGaussians, plyAssemblyMode };
}

let nextPlyWorkerRequestId = 1;

function defaultPlyWorkerFactory() {
  if (typeof Worker !== 'function') {
    throw new Error('Web Worker is unavailable');
  }
  return new Worker(new URL('../workers/ply_writer.js', import.meta.url), {
    type: 'module',
    name: 'sharp-ply-writer',
  });
}

function plyWorkerError(message, cause = null) {
  const error = new Error(`PLY worker failed during ply-blob-assembly: ${message}`);
  error.name = 'PlyWorkerError';
  error.phase = 'ply-blob-assembly';
  if (cause) error.cause = cause;
  return error;
}

function workerCleanupErrorDetails(error) {
  return error ? {
    name: error?.name || 'Error',
    message: error?.message || String(error),
  } : null;
}

function cleanupPlyWorker(worker, terminateWorker) {
  let cleanupError = null;
  for (const handler of ['onmessage', 'onerror', 'onmessageerror']) {
    try {
      worker[handler] = null;
    } catch (error) {
      cleanupError ||= error;
    }
  }
  if (terminateWorker) {
    try {
      Reflect.apply(terminateWorker, worker, []);
    } catch (error) {
      cleanupError ||= error;
    }
  }
  return cleanupError;
}

function attachWorkerCleanupError(error, cleanupError) {
  const details = workerCleanupErrorDetails(cleanupError);
  if (details) error.cleanupError = details;
  return error;
}

function attachWorkerCleanupUnavailable(error, cleanupUnavailable) {
  const details = workerCleanupErrorDetails(cleanupUnavailable);
  if (details) error.cleanupUnavailable = details;
  return error;
}

function capturePlyWorkerCapability(worker, capability) {
  try {
    return { value: worker?.[capability] ?? null, error: null };
  } catch (error) {
    return { value: null, error };
  }
}

/**
 * Assemble an exact PLY either on the calling thread or in a one-shot Worker.
 * Worker mode transfers ownership of plyData.buffer and never silently falls
 * back to the main thread.
 */
export function writePLYAsync(plyData, numGaussians, imgW, imgH, focalPx, options = {}) {
  const mode = options.mode ?? 'main-thread';
  if (mode === 'main-thread') {
    return Promise.resolve(writePLY(plyData, numGaussians, imgW, imgH, focalPx));
  }
  if (mode !== 'worker') {
    return Promise.reject(new RangeError(`Unsupported PLY assembly mode: ${mode}`));
  }
  if (!(plyData instanceof Float32Array)) {
    return Promise.reject(new TypeError('Worker PLY assembly requires Float32Array data'));
  }

  let worker;
  try {
    worker = (options.workerFactory || defaultPlyWorkerFactory)();
  } catch (error) {
    return Promise.reject(plyWorkerError(error?.message || String(error), error));
  }
  const postMessageCapability = capturePlyWorkerCapability(worker, 'postMessage');
  const terminateCapability = capturePlyWorkerCapability(worker, 'terminate');
  const postWorkerMessage = typeof postMessageCapability.value === 'function'
    ? postMessageCapability.value
    : null;
  const terminateWorker = typeof terminateCapability.value === 'function'
    ? terminateCapability.value
    : null;
  const capabilityError = postMessageCapability.error || terminateCapability.error;
  if (!worker || capabilityError || !postWorkerMessage || !terminateWorker) {
    let cleanupError = null;
    if (worker && terminateWorker) {
      try {
        Reflect.apply(terminateWorker, worker, []);
      } catch (error) {
        cleanupError = error;
      }
    }
    const primaryError = capabilityError
      ? plyWorkerError(capabilityError?.message || String(capabilityError), capabilityError)
      : plyWorkerError('worker factory returned an invalid Worker');
    attachWorkerCleanupError(primaryError, cleanupError);
    attachWorkerCleanupUnavailable(primaryError, terminateCapability.error);
    return Promise.reject(primaryError);
  }

  return new Promise((resolve, reject) => {
    const requestId = `sharp-ply-${nextPlyWorkerRequestId++}`;
    let settled = false;
    const fail = (message, cause = null) => {
      if (settled) return;
      settled = true;
      const primaryError = plyWorkerError(message, cause);
      const cleanupError = cleanupPlyWorker(worker, terminateWorker);
      reject(attachWorkerCleanupError(primaryError, cleanupError));
    };

    const handleMessage = event => {
      if (settled) return;
      const result = event?.data;
      if (result?.requestId !== requestId) {
        fail('worker response request identity mismatch');
        return;
      }
      if (result.type === 'ply-error') {
        fail(result.error?.message || 'worker reported an unknown error');
        return;
      }
      if (result.type !== 'ply-assembled' || !(result.plyBlob instanceof Blob)) {
        fail('worker returned an invalid PLY result');
        return;
      }
      if (result.bytes !== result.plyBlob.size) {
        fail('worker byte count does not match returned PLY Blob');
        return;
      }
      const hasValidSha256 = /^[0-9a-f]{64}$/.test(result.sha256 || '');
      if (options.requireSha256 && !hasValidSha256) {
        fail('worker returned missing or invalid PLY SHA-256');
        return;
      }
      if (hasValidSha256) {
        options.onArtifactIdentity?.({
          algorithm: 'sha256',
          authority: 'worker-assembled-artifact',
          sha256: result.sha256,
          bytes: result.bytes,
        });
      }
      settled = true;
      cleanupPlyWorker(worker, terminateWorker);
      resolve(result.plyBlob);
    };
    const handleError = event => fail(event?.message || 'worker execution error', event?.error || null);
    const handleMessageError = () => fail('worker response could not be deserialized');

    try {
      worker.onmessage = handleMessage;
      worker.onerror = handleError;
      worker.onmessageerror = handleMessageError;
      Reflect.apply(postWorkerMessage, worker, [{
        type: 'assemble-ply',
        requestId,
        plyBuffer: plyData.buffer,
        plyByteOffset: plyData.byteOffset,
        plyLength: plyData.length,
        numGaussians,
        imgW,
        imgH,
        focalPx,
      }, [plyData.buffer]]);
    } catch (error) {
      fail(error?.message || String(error), error);
    }
  });
}

/**
 * Write standard 3DGS PLY format.
 */
export function writePLY(plyData, numGaussians, imgW, imgH, focalPx) {
  // Vertex data
  const header = `ply
format binary_little_endian 1.0
element vertex ${numGaussians}
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
element intrinsic 9
property float intrinsic
element image_size 2
property uint image_size
element color_space 1
property uchar color_space
end_header
`;

  const headerBytes = new TextEncoder().encode(header);
  const vertexBytes = new Uint8Array(plyData.buffer, plyData.byteOffset, plyData.byteLength);

  // Intrinsics: 3x3 matrix flattened [fx, 0, cx, 0, fy, cy, 0, 0, 1]
  const intrinsics = new Float32Array([focalPx, 0, imgW * 0.5, 0, focalPx, imgH * 0.5, 0, 0, 1]);
  const intrinsicBytes = new Uint8Array(intrinsics.buffer);

  // Image size: [width, height] as uint32
  const imageSize = new Uint32Array([imgW, imgH]);
  const imageSizeBytes = new Uint8Array(imageSize.buffer);

  // Color space: 1 = sRGB (matching reference save_ply)
  const colorSpace = new Uint8Array([1]);

  return new Blob(
    [headerBytes, vertexBytes, intrinsicBytes, imageSizeBytes, colorSpace],
    { type: 'application/octet-stream' },
  );
}
