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

// --- Covariance-faithful unprojection helpers (mirror ml-sharp apply_transform) ---

// Rotation matrix from (w, x, y, z) quaternion; normalizes first.
function quatToRotationMatrix(qw, qx, qy, qz, out) {
  const n = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) || 1;
  const w = qw / n, x = qx / n, y = qy / n, z = qz / n;
  out[0] = 1 - 2 * (y * y + z * z); out[1] = 2 * (x * y - w * z); out[2] = 2 * (x * z + w * y);
  out[3] = 2 * (x * y + w * z); out[4] = 1 - 2 * (x * x + z * z); out[5] = 2 * (y * z - w * x);
  out[6] = 2 * (x * z - w * y); out[7] = 2 * (y * z + w * x); out[8] = 1 - 2 * (x * x + y * y);
}

// Jacobi eigendecomposition of a symmetric 3x3 matrix.
// Fills eigenvalues (descending) into evals[3] and column eigenvectors into evecs[9].
function symmetricEigen3(m, evals, evecs) {
  let a00 = m[0], a01 = m[1], a02 = m[2], a11 = m[4], a12 = m[5], a22 = m[8];
  let v = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let sweep = 0; sweep < 24; sweep++) {
    const off = a01 * a01 + a02 * a02 + a12 * a12;
    if (off < 1e-30) break;
    for (let pq = 0; pq < 3; pq++) {
      let p, q, apq, app, aqq;
      if (pq === 0) { p = 0; q = 1; apq = a01; app = a00; aqq = a11; }
      else if (pq === 1) { p = 0; q = 2; apq = a02; app = a00; aqq = a22; }
      else { p = 1; q = 2; apq = a12; app = a11; aqq = a22; }
      if (Math.abs(apq) < 1e-30) continue;
      const theta = (aqq - app) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      // Apply rotation to the symmetric matrix elements
      if (pq === 0) {
        const n00 = c * c * a00 - 2 * s * c * a01 + s * s * a11;
        const n11 = s * s * a00 + 2 * s * c * a01 + c * c * a11;
        const n02 = c * a02 - s * a12, n12 = s * a02 + c * a12;
        a00 = n00; a11 = n11; a01 = 0; a02 = n02; a12 = n12;
      } else if (pq === 1) {
        const n00 = c * c * a00 - 2 * s * c * a02 + s * s * a22;
        const n22 = s * s * a00 + 2 * s * c * a02 + c * c * a22;
        const n01 = c * a01 - s * a12, n12 = s * a01 + c * a12;
        a00 = n00; a22 = n22; a02 = 0; a01 = n01; a12 = n12;
      } else {
        const n11 = c * c * a11 - 2 * s * c * a12 + s * s * a22;
        const n22 = s * s * a11 + 2 * s * c * a12 + c * c * a22;
        const n01 = c * a01 - s * a02, n02 = s * a01 + c * a02;
        a11 = n11; a22 = n22; a12 = 0; a01 = n01; a02 = n02;
      }
      // Accumulate eigenvectors: v = v @ J(p,q,c,s)
      for (let r = 0; r < 3; r++) {
        const vp = v[r * 3 + p], vq = v[r * 3 + q];
        v[r * 3 + p] = c * vp - s * vq;
        v[r * 3 + q] = s * vp + c * vq;
      }
    }
  }
  // Sort eigenvalues descending (matches torch.linalg.svd ordering)
  const order = [[a00, 0], [a11, 1], [a22, 2]].sort((u, w) => w[0] - u[0]);
  for (let i = 0; i < 3; i++) {
    evals[i] = order[i][0];
    const src = order[i][1];
    evecs[0 * 3 + i] = v[0 * 3 + src];
    evecs[1 * 3 + i] = v[1 * 3 + src];
    evecs[2 * 3 + i] = v[2 * 3 + src];
  }
  // Ensure a proper rotation (det +1), mirroring the reference SVD reflection fix
  const det =
    evecs[0] * (evecs[4] * evecs[8] - evecs[5] * evecs[7]) -
    evecs[1] * (evecs[3] * evecs[8] - evecs[5] * evecs[6]) +
    evecs[2] * (evecs[3] * evecs[7] - evecs[4] * evecs[6]);
  if (det < 0) { evecs[2] *= -1; evecs[5] *= -1; evecs[8] *= -1; }
}

// (w, x, y, z) quaternion from a rotation matrix (Shepperd's method).
function rotationMatrixToQuat(r, out) {
  const trace = r[0] + r[4] + r[8];
  let w, x, y, z;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    w = 0.25 * s; x = (r[7] - r[5]) / s; y = (r[2] - r[6]) / s; z = (r[3] - r[1]) / s;
  } else if (r[0] > r[4] && r[0] > r[8]) {
    const s = Math.sqrt(1 + r[0] - r[4] - r[8]) * 2;
    w = (r[7] - r[5]) / s; x = 0.25 * s; y = (r[1] + r[3]) / s; z = (r[2] + r[6]) / s;
  } else if (r[4] > r[8]) {
    const s = Math.sqrt(1 + r[4] - r[0] - r[8]) * 2;
    w = (r[2] - r[6]) / s; x = (r[1] + r[3]) / s; y = 0.25 * s; z = (r[5] + r[7]) / s;
  } else {
    const s = Math.sqrt(1 + r[8] - r[0] - r[4]) * 2;
    w = (r[3] - r[1]) / s; x = (r[2] + r[6]) / s; y = (r[5] + r[7]) / s; z = 0.25 * s;
  }
  out[0] = w; out[1] = x; out[2] = y; out[3] = z;
}
function inverseSoftplus(x) { return x > 20 ? x : Math.log(Math.exp(Math.max(x, 1e-6)) - 1); }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function inverseSigmoid(x) {
  const c = Math.max(1e-6, Math.min(1 - 1e-6, x));
  return Math.log(c / (1 - c));
}
function sRGB2linear(x) { return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4); }
function linear2sRGB(x) { return x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055; }

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
 * @param {{ chunkItems?: number, onChunk?: (chunk: object) => Promise<void>, onInterval?: (interval: object) => void }} [options]
 * @returns {Promise<{ plyBlob: Blob, numGaussians: number }>}
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

  const ROT_TMP = new Float64Array(9);
  const COV_TMP = new Float64Array(9);
  const EVAL_TMP = new Float64Array(3);
  const EVEC_TMP = new Float64Array(9);
  const QUAT_TMP = new Float64Array(4);

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

        // Covariance-faithful transform (ml-sharp apply_transform):
        //   Σ_ndc = R diag(s²) Rᵀ;  Σ_world = M Σ_ndc Mᵀ with M = diag(unprojX, unprojY, 1)
        // then eigendecompose Σ_world back into world scales + rotation.
        // The previous per-axis scale shortcut (s_i * unproj_i) is only valid
        // for axis-aligned Gaussians and skewed both scales and orientations.
        quatToRotationMatrix(qw, qx, qy, qz, ROT_TMP);
        const s0sq = sv0 * sv0, s1sq = sv1 * sv1, s2sq = sv2 * sv2;
        // Σ = R diag(s²) Rᵀ (symmetric)
        for (let r = 0; r < 3; r++) {
          for (let c = r; c < 3; c++) {
            COV_TMP[r * 3 + c] = COV_TMP[c * 3 + r] =
              ROT_TMP[r * 3] * ROT_TMP[c * 3] * s0sq +
              ROT_TMP[r * 3 + 1] * ROT_TMP[c * 3 + 1] * s1sq +
              ROT_TMP[r * 3 + 2] * ROT_TMP[c * 3 + 2] * s2sq;
          }
        }
        // M Σ Mᵀ for diagonal M
        const mDiag0 = unprojX, mDiag1 = unprojY, mDiag2 = 1.0;
        COV_TMP[0] *= mDiag0 * mDiag0; COV_TMP[1] *= mDiag0 * mDiag1; COV_TMP[2] *= mDiag0 * mDiag2;
        COV_TMP[3] *= mDiag1 * mDiag0; COV_TMP[4] *= mDiag1 * mDiag1; COV_TMP[5] *= mDiag1 * mDiag2;
        COV_TMP[6] *= mDiag2 * mDiag0; COV_TMP[7] *= mDiag2 * mDiag1; COV_TMP[8] *= mDiag2 * mDiag2;
        symmetricEigen3(COV_TMP, EVAL_TMP, EVEC_TMP);
        const worldSV0 = Math.sqrt(Math.max(0, EVAL_TMP[0]));
        const worldSV1 = Math.sqrt(Math.max(0, EVAL_TMP[1]));
        const worldSV2 = Math.sqrt(Math.max(0, EVAL_TMP[2]));
        rotationMatrixToQuat(EVEC_TMP, QUAT_TMP);

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
        plyData[gIdx + 10] = QUAT_TMP[0];  // rot_0 (w)
        plyData[gIdx + 11] = QUAT_TMP[1];  // rot_1 (x)
        plyData[gIdx + 12] = QUAT_TMP[2];  // rot_2 (y)
        plyData[gIdx + 13] = QUAT_TMP[3];  // rot_3 (z)
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
  const plyBlob = writePLY(plyData, numGaussians, origW, origH, focalPx);
  const plyAssemblyEndMs = performance.now();
  onInterval?.({
    step: 'ply-blob-assembly',
    intervalStartMs: plyAssemblyStartMs,
    intervalEndMs: plyAssemblyEndMs,
    durationMs: plyAssemblyEndMs - plyAssemblyStartMs,
    bytes: plyBlob.size,
  });

  return { plyBlob, numGaussians };
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
