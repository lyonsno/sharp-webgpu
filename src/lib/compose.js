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
 * @returns {{ plyBlob: Blob, numGaussians: number }}
 */
export function composeAndExport(dispData, geomDeltas, texDeltas, img01, imgH, imgW, outH, outW) {
  const { numLayers, stride, scaleFactor, disparityFactor, normalizeDepth, baseDepth,
    baseScaleOnPredictedMean, deltaFactor, minScale, maxScale } = PARAMS;

  const HW = imgH * imgW;
  const baseH = imgH / stride;  // 768
  const baseW = imgW / stride;  // 768
  const baseHW = baseH * baseW;

  console.log('[Compose] Building base Gaussians...');

  // --- Step 1: Depth normalization ---
  // depth = disparityFactor / clamp(disparity, 1e-4, 1e4)
  // dispData has 2 channels — use both for 2-layer depth
  const depth = new Float32Array(2 * HW);
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < HW; i++) {
      const disp = Math.max(1e-4, Math.min(1e4, dispData[c * HW + i]));
      depth[c * HW + i] = disparityFactor / disp;
    }
  }

  let globalScale = 1.0;
  if (normalizeDepth) {
    // Rescale depth so min = 1.0
    let depthMin = Infinity;
    for (let i = 0; i < 2 * HW; i++) {
      if (depth[i] < depthMin) depthMin = depth[i];
    }
    const depthFactor = 1.0 / (depthMin + 1e-6);
    for (let i = 0; i < 2 * HW; i++) {
      depth[i] = Math.min(depth[i] * depthFactor, 100);
    }
    globalScale = 1.0 / depthFactor;
  }

  // --- Step 2: Base values ---
  // Base XY in NDC: [-1, 1] grid at stride=2
  // Base inverse Z: from depth via max_pool2d (surface_min → use max of 1/depth)
  // disparity[layer] at base resolution via max_pool(1/depth, stride)
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

  // Base XY NDC
  const baseX = new Float32Array(baseHW);
  const baseY = new Float32Array(baseHW);
  for (let by = 0; by < baseH; by++) {
    for (let bx = 0; bx < baseW; bx++) {
      baseX[by * baseW + bx] = 2 * (bx * stride + 0.5 * stride) / imgW - 1.0;
      baseY[by * baseW + bx] = 2 * (by * stride + 0.5 * stride) / imgH - 1.0;
    }
  }

  // Base scales
  const dispScaleFactor = 2 * scaleFactor * stride / imgW;

  // Base colors: avg_pool'd image (all_layers)
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

  // --- Step 3: Compose Gaussians ---
  console.log('[Compose] Composing Gaussians...');
  const numGaussians = numLayers * baseHW;
  // PLY fields: x,y,z, f_dc_0/1/2, opacity, scale_0/1/2, rot_0/1/2/3 = 14 floats
  const plyData = new Float32Array(numGaussians * 14);

  // Scale activation constants
  const scaleConstA = (maxScale - minScale) / (1 - minScale) / (maxScale - 1);
  const scaleConstB = inverseSigmoid((1.0 - minScale) / (maxScale - minScale));

  const outHW = outH * outW;

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

        // Apply global scale
        meanX *= globalScale;
        meanY *= globalScale;
        meanZ *= globalScale;
        const sv0 = scales[0] * globalScale;
        const sv1 = scales[1] * globalScale;
        const sv2 = scales[2] * globalScale;

        // --- Write PLY fields ---
        // For standard 3DGS PLY: xyz, f_dc (SH0), opacity (logit), scale (log), quaternion
        const SH0_COEFF = Math.sqrt(1.0 / (4 * Math.PI));

        plyData[gIdx + 0] = meanX;                          // x
        plyData[gIdx + 1] = meanY;                          // y
        plyData[gIdx + 2] = meanZ;                          // z
        plyData[gIdx + 3] = (linear2sRGB(colors[0]) - 0.5) / SH0_COEFF;  // f_dc_0
        plyData[gIdx + 4] = (linear2sRGB(colors[1]) - 0.5) / SH0_COEFF;  // f_dc_1
        plyData[gIdx + 5] = (linear2sRGB(colors[2]) - 0.5) / SH0_COEFF;  // f_dc_2
        plyData[gIdx + 6] = inverseSigmoid(Math.max(1e-6, Math.min(1 - 1e-6, opacity)));  // opacity logit
        plyData[gIdx + 7] = Math.log(Math.max(1e-10, sv0));  // scale_0
        plyData[gIdx + 8] = Math.log(Math.max(1e-10, sv1));  // scale_1
        plyData[gIdx + 9] = Math.log(Math.max(1e-10, sv2));  // scale_2
        plyData[gIdx + 10] = qw;  // rot_0
        plyData[gIdx + 11] = qx;  // rot_1
        plyData[gIdx + 12] = qy;  // rot_2
        plyData[gIdx + 13] = qz;  // rot_3
      }
    }
  }

  console.log(`[Compose] ${numGaussians} Gaussians composed`);

  // --- Step 4: Write PLY ---
  console.log('[Compose] Writing PLY...');
  const plyBlob = writePLY(plyData, numGaussians, imgH, imgW);

  return { plyBlob, numGaussians };
}

/**
 * Write standard 3DGS PLY format.
 */
function writePLY(plyData, numGaussians, imgH, imgW) {
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
end_header
`;

  const headerBytes = new TextEncoder().encode(header);
  const dataBytes = new Uint8Array(plyData.buffer);
  const totalSize = headerBytes.length + dataBytes.length;
  const combined = new Uint8Array(totalSize);
  combined.set(headerBytes);
  combined.set(dataBytes, headerBytes.length);

  return new Blob([combined], { type: 'application/octet-stream' });
}
