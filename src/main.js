/**
 * SHARP-WebGPU — Main entry point
 *
 * Apple SHARP (single-image 3D Gaussian Splat generation) in WebGPU compute.
 * Architecture: DINOv2 ViT-Large → SPN → MonodepthDPT + GaussianDecoder → 3DGS output
 */

import { initGPU } from './lib/gpu.js';
import { loadWeights } from './lib/weights.js';
import { SharpBackbone } from './lib/backbone.js';

let gpu = null;
let weights = null;
let backbone = null;

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('error');
const outputEl = document.getElementById('output');
const statsEl = document.getElementById('stats');

function setStatus(msg) {
  statusEl.textContent = msg;
  errorEl.style.display = 'none';
}

function setError(msg) {
  errorEl.textContent = msg;
  errorEl.style.display = 'block';
  statusEl.textContent = '';
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
  if (file && file.type.startsWith('image/')) handleImage(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleImage(fileInput.files[0]);
});

async function handleImage(file) {
  try {
    setStatus('Initializing WebGPU...');
    if (!gpu) {
      gpu = await initGPU();
    }

    setStatus('Loading image...');
    const bitmap = await createImageBitmap(file);

    const inputCanvas = document.getElementById('input-canvas');
    inputCanvas.width = bitmap.width;
    inputCanvas.height = bitmap.height;
    const ctx = inputCanvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0);

    if (!weights) {
      setStatus('Loading SHARP weights (first load only)...');
      weights = await loadWeights(gpu.device, '/weights.bin', (received, total) => {
        const mb = (received / 1024 / 1024).toFixed(0);
        const totalMb = total ? (total / 1024 / 1024).toFixed(0) : '?';
        setStatus(`Loading weights: ${mb} / ${totalMb} MB`);
      });
    }

    if (!backbone) {
      backbone = new SharpBackbone(gpu.device);
      backbone.init(weights);
    }

    // --- Backbone smoke: run patch encoder on the image ---
    setStatus('Running backbone smoke...');
    const imageData = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    const t0 = performance.now();

    const result = await backbone.run(imageData, bitmap.width, bitmap.height);
    const elapsed = performance.now() - t0;

    setStatus('');
    statsEl.textContent = `Backbone: ${elapsed.toFixed(0)}ms | Output tokens: ${result.tokenH}x${result.tokenW} | Features: [${result.dim}]`;

    // Visualize depth (placeholder — just show we got output)
    const depthCanvas = document.getElementById('depth-canvas');
    depthCanvas.width = result.tokenW;
    depthCanvas.height = result.tokenH;
    outputEl.classList.add('visible');

  } catch (err) {
    setError(err.message);
    console.error(err);
  }
}
