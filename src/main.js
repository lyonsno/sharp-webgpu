/**
 * SHARP-WebGPU — Main entry point
 *
 * Apple SHARP (single-image 3D Gaussian Splat generation) in WebGPU compute.
 *
 * Current status: backbone only (ViT-Large, DINOv2, patch_size=16).
 * Full pipeline (SPN, monodepth, Gaussian decoder, 3DGS) not yet implemented.
 */

import { initGPU } from './lib/gpu.js';
import { loadWeights } from './lib/weights.js';
import { SharpBackbone } from './lib/backbone.js';

let gpu = null;
let weights = null;
let backbone = null;
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

function showResults(result, elapsed) {
  document.getElementById('r-model').textContent = 'DINOv2 ViT-Large (dinov2l16_384)';
  document.getElementById('r-weights').textContent = `${weightsLoadedMB} MB (fp16)`;
  document.getElementById('r-patch').textContent = '16x16';
  document.getElementById('r-grid').textContent = `${result.tokenH}x${result.tokenW} = ${result.numPatches} patches + 1 CLS`;
  document.getElementById('r-features').textContent = `${result.intermediateFeatures.length} layers`;
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
      setStatus('Loading SHARP weights (~1.3 GB, first load only)...');
      weights = await loadWeights(gpu.device, '/weights.bin', (received, total) => {
        const mb = (received / 1024 / 1024).toFixed(0);
        weightsLoadedMB = mb;
        const totalMb = total ? (total / 1024 / 1024).toFixed(0) : '?';
        setStatus(`Loading weights: ${mb} / ${totalMb} MB`);
      });
    }

    if (!backbone) {
      backbone = new SharpBackbone(gpu.device);
      backbone.init(weights);
    }

    // Pass the original blob to backbone — it handles resize + normalization
    setStatus('Running ViT-Large backbone...');
    const t0 = performance.now();
    const result = await backbone.run(blob);
    const elapsed = performance.now() - t0;

    setStatus('');
    showResults(result, elapsed);

  } catch (err) {
    setError(err.message);
    console.error(err);
  }
}
