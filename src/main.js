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
    document.getElementById('r-grid').textContent = `SPN: 35 patches (5x5 + 3x3 + 1x1)`;
    const gaussStr = result.numGaussians ? ` → ${(result.numGaussians / 1000).toFixed(0)}K Gaussians` : '';
    document.getElementById('r-features').textContent = `${result.featureDims.length} multi-res outputs${gaussStr}`;
  } else {
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
  try {
    setStatus('Initializing WebGPU...');
    if (!gpu) {
      gpu = await initGPU();
    }

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
    const useSPN = document.getElementById('use-spn')?.checked ?? false;

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

      const t0 = performance.now();
      const spnResult = await spn.run(chw);

      // Run monodepth decoder
      if (!monodepth) {
        monodepth = new MonodepthDecoder(gpu.device);
      }
      setStatus('Running monodepth decoder...');
      const depthResult = await monodepth.run(spnResult.features, spnResult.featureDims, weights);
      const elapsed = performance.now() - t0;

      // Read back disparity and visualize
      const dispData = await readBuffer(gpu.device, depthResult.disparityBuf, depthResult.C * depthResult.H * depthResult.W * 4);

      // Render depth map (channel 0 of 2-channel disparity)
      const depthCanvas = document.getElementById('depth-canvas');
      if (depthCanvas) {
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
          const v = dispData[i]; // channel 0
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
            const v = dispData[sy * dW + sx]; // channel 0
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

      // Run Gaussian prediction pipeline
      if (!gaussianPipeline) {
        gaussianPipeline = new GaussianPipeline(gpu.device);
      }
      setStatus('Running Gaussian prediction...');
      const gaussResult = await gaussianPipeline.run(
        spnResult.features, spnResult.featureDims,
        depthResult.disparityBuf, depthResult.H, depthResult.W,
        chw, weights
      );

      console.log(`[Main] ${gaussResult.numGaussians} Gaussians predicted (${gaussResult.numLayers} layers × ${gaussResult.H}×${gaussResult.W})`);

      // Compose final Gaussians and generate PLY
      setStatus('Composing Gaussians + PLY export...');

      // Reuse disparity data from depth visualization (avoid redundant GPU readback)
      // Convert image from [-1,1] to [0,1] for initializer
      const img01 = new Float32Array(chw.length);
      for (let i = 0; i < chw.length; i++) img01[i] = (chw[i] + 1.0) * 0.5;

      // Read raw deltas from stored GPU buffers
      const geomDeltas = await readBuffer(gpu.device, gaussianPipeline._geomDeltasBuf, 6 * gaussResult.H * gaussResult.W * 4);
      const texDeltas = await readBuffer(gpu.device, gaussianPipeline._texDeltasBuf, 22 * gaussResult.H * gaussResult.W * 4);

      const composed = composeAndExport(
        dispData, geomDeltas, texDeltas,
        img01, 1536, 1536, gaussResult.H, gaussResult.W,
        bitmap.width, bitmap.height  // original image dims for unprojection
      );

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
      spnResult.hasNaN = false;
      spnResult.numGaussians = composed.numGaussians;
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

      setStatus('');
      showResults(result, elapsed, 'backbone');
    }

  } catch (err) {
    setError(err.message);
    console.error(err);
  }
}
