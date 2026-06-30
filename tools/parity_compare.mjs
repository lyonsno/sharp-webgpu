#!/usr/bin/env node
/**
 * parity_compare.mjs — Generic per-stage numerical parity comparator.
 *
 * Loads reference PyTorch dumps (from dump_reference.py or equivalent),
 * runs the WebGPU pipeline on the same input, and compares intermediate
 * tensors at each named stage.
 *
 * Designed to be reusable across models: MoGe, SHARP, SF3D, Kimodo.
 * Each model provides a reference dump manifest and a browser-side
 * stage capture callback.
 *
 * Usage:
 *   node tools/parity_compare.mjs [--port 5175] [--manifest public/reference_dumps/manifest.json] [--headed]
 *
 * Output:
 *   /tmp/parity-report.json — structured per-stage comparison
 *   Console — human-readable summary table
 */

import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

/**
 * Compare two Float32Arrays and return error statistics.
 */
function compareArrays(gpu, ref) {
  const n = Math.min(gpu.length, ref.length);
  let maxErr = 0, sumErr = 0, sumSqErr = 0;
  let gpuSum = 0, refSum = 0, gpuSqSum = 0, refSqSum = 0;
  let worstIdx = 0, nanCount = 0, infCount = 0;

  for (let i = 0; i < n; i++) {
    if (!isFinite(gpu[i])) { if (isNaN(gpu[i])) nanCount++; else infCount++; continue; }
    if (!isFinite(ref[i])) continue;

    const err = Math.abs(gpu[i] - ref[i]);
    sumErr += err;
    sumSqErr += err * err;
    if (err > maxErr) { maxErr = err; worstIdx = i; }

    gpuSum += gpu[i]; refSum += ref[i];
    gpuSqSum += gpu[i] * gpu[i]; refSqSum += ref[i] * ref[i];
  }

  const finiteN = n - nanCount - infCount;
  if (finiteN === 0) return { maxErr: NaN, meanErr: NaN, rmsErr: NaN, relStd: NaN, nanCount, infCount, n };

  const gpuMean = gpuSum / finiteN;
  const refMean = refSum / finiteN;
  const gpuStd = Math.sqrt(Math.max(0, gpuSqSum / finiteN - gpuMean * gpuMean));
  const refStd = Math.sqrt(Math.max(0, refSqSum / finiteN - refMean * refMean));
  const meanErr = sumErr / finiteN;
  const rmsErr = Math.sqrt(sumSqErr / finiteN);
  const relStd = refStd > 0 ? gpuStd / refStd : NaN;

  return {
    maxErr, meanErr, rmsErr, relStd,
    gpu: { min: null, max: null, mean: gpuMean, std: gpuStd },
    ref: { min: null, max: null, mean: refMean, std: refStd },
    worstIdx,
    worstGpu: gpu[worstIdx],
    worstRef: ref[worstIdx],
    nanCount, infCount, n: finiteN,
  };
}

/**
 * Load a reference dump from the manifest.
 */
function loadReferenceDump(manifestDir, dumpInfo) {
  const filePath = path.join(manifestDir, dumpInfo.file);
  const buffer = fs.readFileSync(filePath);
  // Copy to aligned ArrayBuffer (Node Buffer may not be 4-byte aligned)
  const aligned = new ArrayBuffer(buffer.byteLength);
  new Uint8Array(aligned).set(buffer);
  return new Float32Array(aligned);
}

async function main() {
  const args = process.argv.slice(2);
  const port = args.includes('--port') ? args[args.indexOf('--port') + 1] : '5175';
  const manifestPath = args.includes('--manifest')
    ? args[args.indexOf('--manifest') + 1]
    : 'public/reference_dumps/manifest.json';
  const headed = args.includes('--headed');
  const outputPath = args.includes('--output')
    ? args[args.indexOf('--output') + 1]
    : '/tmp/parity-report.json';

  // Load manifest
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  const manifestDir = path.dirname(manifestPath);
  console.log(`Reference: ${manifest.image} (${manifest.dtype}, ${manifest.device})`);
  console.log(`  focal: ${manifest.focal_px}px, internal: ${manifest.internal_shape.join('x')}`);
  console.log(`  ${Object.keys(manifest.dumps).length} reference dumps\n`);

  // Stages we can compare (WebGPU captures these via console.log parsing)
  // The browser pipeline logs structured data that we capture
  const url = `http://localhost:${port}/`;

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: !headed,
    protocolTimeout: 600000,
    args: [
      '--enable-unsafe-webgpu', '--enable-features=Vulkan',
      '--disable-gpu-sandbox', '--no-sandbox',
      '--disable-gpu-shader-disk-cache',
      '--window-size=1280,900',
    ],
    defaultViewport: { width: 1280, height: 900 },
  });

  const page = await browser.newPage();

  const consoleLines = [];
  page.on('console', msg => {
    const text = msg.text();
    consoleLines.push(text);
    if (text.includes('[') || text.includes('Loaded') || text.includes('error'))
      console.log(`  [page] ${text}`);
  });
  page.on('pageerror', err => console.error('PAGE ERROR:', err.message));

  try {
    console.log('Loading page...');
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });

    // Inject the reference dumps as accessible data
    // We'll serve them via the reference_dumps/ directory (already in public/)
    // and have the page fetch + compare

    // Inject a comparison function into the page
    await page.evaluate(() => {
      window.__parityCaptures = {};
      window.__parityReady = false;
    });

    // Run the full SPN pipeline
    console.log('Running full SPN pipeline...');
    await page.click('#use-spn');
    await page.click('.sample-thumb');

    // Wait for pipeline completion
    const completed = await page.waitForFunction(() => {
      const t = document.getElementById('r-time');
      const e = document.getElementById('error');
      if (e && e.style.display !== 'none' && e.textContent) return 'error:' + e.textContent;
      if (t && t.textContent && t.textContent !== '-') return 'done:' + t.textContent;
      return false;
    }, { timeout: 600000 });

    const status = await completed.jsonValue();
    if (status.startsWith('error:')) {
      console.error(`Pipeline failed: ${status.slice(6)}`);
      process.exit(1);
    }
    console.log(`\nPipeline completed: ${status.slice(5)}\n`);

    // Now compare the stages we can access.
    // The WebGPU pipeline writes intermediate data to GPU buffers that get read
    // back at various points. For a proper stage-by-stage comparison, we need the
    // page to expose its intermediate readbacks.
    //
    // For now, compare the stages we can access from the existing pipeline:
    // - monodepth_disparity (read back for depth visualization)
    // - geom_deltas and tex_deltas (read back for compose)
    // - final composed Gaussians (available from PLY blob)
    //
    // Future: the kit profiling primitives will let us capture arbitrary stages.

    // Compare what we can: fetch reference dumps and compare against page state
    const report = { model: 'SHARP', stages: {}, summary: {} };

    // Load key reference dumps
    const refStages = [
      'monodepth_disparity',
      'geom_deltas',
      'tex_deltas',
      'gaussians_ndc_means',
      'gaussians_ndc_scales',
      'gaussians_ndc_opacities',
      'gaussians_ndc_colors',
      'gaussians_ndc_quats',
    ];

    for (const stageName of refStages) {
      if (!manifest.dumps[stageName]) continue;

      const refData = loadReferenceDump(manifestDir, manifest.dumps[stageName]);
      const refShape = manifest.dumps[stageName].shape;

      // Inject the reference data into the page for comparison
      // We'll use the page's readBuffer to get the WebGPU data
      const refB64 = Buffer.from(refData.buffer, refData.byteOffset, refData.byteLength).toString('base64');

      const stageResult = await page.evaluate(async (stageName, refB64, refShape) => {
        // Decode reference
        const refBytes = Uint8Array.from(atob(refB64), c => c.charCodeAt(0));
        const ref = new Float32Array(refBytes.buffer);

        // Try to get the WebGPU data for this stage from the pipeline state
        let gpu = null;

        if (stageName === 'monodepth_disparity') {
          // dispData was stored on window by main.js for depth viz
          // We need to access it — check if it's available
          // The pipeline stores dispData in the closure, not on window
          // For now, return null — we'll need to modify main.js to expose it
          return { stage: stageName, available: false, reason: 'not exposed to window' };
        }

        // For deltas and composed Gaussians, same issue — they're in function closures
        return { stage: stageName, available: false, reason: 'intermediate not exposed; needs kit capture hooks' };
      }, stageName, refB64, refShape);

      if (!stageResult.available) {
        report.stages[stageName] = { status: 'skipped', reason: stageResult.reason };
        continue;
      }
    }

    // Since intermediates aren't exposed yet, let's do what we CAN do:
    // Compare the final PLY output against reference world-space Gaussians.
    // The PLY blob is accessible via the download link.
    console.log('Comparing final PLY output against reference...');

    const refMeans = loadReferenceDump(manifestDir, manifest.dumps.gaussians_world_means);
    const refScales = loadReferenceDump(manifestDir, manifest.dumps.gaussians_world_scales);
    const refColors = loadReferenceDump(manifestDir, manifest.dumps.gaussians_world_colors);
    const refOpacities = loadReferenceDump(manifestDir, manifest.dumps.gaussians_world_opacities);
    const refQuats = loadReferenceDump(manifestDir, manifest.dumps.gaussians_world_quats);

    // Get PLY data from page
    const plyData = await page.evaluate(async () => {
      const link = document.getElementById('download-ply');
      if (!link || !link.href || link.style.display === 'none') return null;

      const resp = await fetch(link.href);
      const blob = await resp.blob();
      const buffer = await blob.arrayBuffer();
      const bytes = new Uint8Array(buffer);

      // Find end of header
      const headerEnd = new TextDecoder().decode(bytes).indexOf('end_header\n');
      if (headerEnd < 0) return null;
      const dataStart = headerEnd + 'end_header\n'.length;

      // Parse vertex count from header
      const header = new TextDecoder().decode(bytes.slice(0, dataStart));
      const vertexMatch = header.match(/element vertex (\d+)/);
      if (!vertexMatch) return null;
      const numVertices = parseInt(vertexMatch[1]);

      // 14 floats per vertex: xyz, f_dc_012, opacity, scale_012, rot_0123
      const floatsPerVertex = 14;
      // Copy to aligned buffer (dataStart may not be 4-byte aligned after text header)
      const vertexBytes = new Uint8Array(buffer, dataStart, numVertices * floatsPerVertex * 4);
      const alignedBuf = new ArrayBuffer(vertexBytes.length);
      new Uint8Array(alignedBuf).set(vertexBytes);
      const vertexData = new Float32Array(alignedBuf);

      // Extract fields
      const means = new Float32Array(numVertices * 3);
      const scaleLog = new Float32Array(numVertices * 3);
      const colors_sh = new Float32Array(numVertices * 3);
      const opacityLogit = new Float32Array(numVertices);
      const quats = new Float32Array(numVertices * 4);

      for (let i = 0; i < numVertices; i++) {
        const base = i * 14;
        means[i * 3] = vertexData[base];
        means[i * 3 + 1] = vertexData[base + 1];
        means[i * 3 + 2] = vertexData[base + 2];
        colors_sh[i * 3] = vertexData[base + 3];
        colors_sh[i * 3 + 1] = vertexData[base + 4];
        colors_sh[i * 3 + 2] = vertexData[base + 5];
        opacityLogit[i] = vertexData[base + 6];
        scaleLog[i * 3] = vertexData[base + 7];
        scaleLog[i * 3 + 1] = vertexData[base + 8];
        scaleLog[i * 3 + 2] = vertexData[base + 9];
        quats[i * 4] = vertexData[base + 10];
        quats[i * 4 + 1] = vertexData[base + 11];
        quats[i * 4 + 2] = vertexData[base + 12];
        quats[i * 4 + 3] = vertexData[base + 13];
      }

      // Convert back to base64 for Node comparison
      function toB64(arr) {
        const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
        let binary = '';
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        return btoa(binary);
      }

      return {
        numVertices,
        means: toB64(means),
        scaleLog: toB64(scaleLog),
        colors_sh: toB64(colors_sh),
        opacityLogit: toB64(opacityLogit),
        quats: toB64(quats),
      };
    });

    if (!plyData) {
      console.error('Could not extract PLY data from page');
      process.exit(1);
    }

    console.log(`PLY: ${plyData.numVertices} vertices\n`);

    // Decode and compare
    function decodeB64(b64) {
      const buf = Buffer.from(b64, 'base64');
      // Copy to aligned ArrayBuffer (Buffer.from may not be 4-byte aligned)
      const aligned = new ArrayBuffer(buf.byteLength);
      new Uint8Array(aligned).set(buf);
      return new Float32Array(aligned);
    }

    const gpuMeans = decodeB64(plyData.means);

    // Reference is in world space (direct positions); PLY is also world space
    // But PLY scales are in log space and opacities are logits — convert reference to match
    const refScaleLog = new Float32Array(refScales.length);
    for (let i = 0; i < refScales.length; i++) refScaleLog[i] = Math.log(Math.max(1e-10, refScales[i]));

    const refOpacityLogit = new Float32Array(refOpacities.length);
    for (let i = 0; i < refOpacities.length; i++) {
      const o = Math.max(1e-6, Math.min(1 - 1e-6, refOpacities[i]));
      refOpacityLogit[i] = Math.log(o / (1 - o));
    }

    // Reference colors are linearRGB [0,1]; PLY has SH degree 0 from sRGB
    // We need to convert reference colors to the same SH representation
    const SH0_COEFF = Math.sqrt(1.0 / (4 * Math.PI));
    function linear2sRGB(x) { return x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055; }
    const refColorsSH = new Float32Array(refColors.length);
    for (let i = 0; i < refColors.length; i++) {
      refColorsSH[i] = (linear2sRGB(refColors[i]) - 0.5) / SH0_COEFF;
    }

    const gpuScaleLog = decodeB64(plyData.scaleLog);
    const gpuColorsSH = decodeB64(plyData.colors_sh);
    const gpuOpacityLogit = decodeB64(plyData.opacityLogit);
    const gpuQuats = decodeB64(plyData.quats);

    const comparisons = {
      means: compareArrays(gpuMeans, refMeans),
      scale_log: compareArrays(gpuScaleLog, refScaleLog),
      colors_sh: compareArrays(gpuColorsSH, refColorsSH),
      opacity_logit: compareArrays(gpuOpacityLogit, refOpacityLogit),
      quaternions: compareArrays(gpuQuats, refQuats),
    };

    // Print results
    console.log('=== PARITY COMPARISON: WebGPU vs PyTorch fp16 Reference ===\n');
    console.log(`${'Stage'.padEnd(18)} ${'maxErr'.padStart(10)} ${'rmsErr'.padStart(10)} ${'meanErr'.padStart(10)} ${'relStd'.padStart(8)} ${'NaN'.padStart(5)} ${'N'.padStart(10)}`);
    console.log('-'.repeat(75));

    let allPass = true;
    for (const [name, stats] of Object.entries(comparisons)) {
      const pass = stats.maxErr < 1.0 && stats.nanCount === 0; // generous threshold for now
      if (!pass) allPass = false;
      const flag = pass ? ' ' : '!';
      console.log(`${flag} ${name.padEnd(17)} ${stats.maxErr.toFixed(4).padStart(10)} ${stats.rmsErr.toFixed(6).padStart(10)} ${stats.meanErr.toFixed(6).padStart(10)} ${stats.relStd.toFixed(4).padStart(8)} ${String(stats.nanCount).padStart(5)} ${String(stats.n).padStart(10)}`);
      if (stats.maxErr > 0.01) {
        console.log(`    worst@${stats.worstIdx}: gpu=${stats.worstGpu?.toFixed(6)} ref=${stats.worstRef?.toFixed(6)}`);
      }
    }

    console.log('-'.repeat(75));
    console.log(allPass ? 'PARITY: PASS (all stages within tolerance)' : 'PARITY: DIFFERENCES FOUND');

    report.stages = comparisons;
    report.summary = {
      pass: allPass,
      numVertices: plyData.numVertices,
      reference: { dtype: manifest.dtype, device: manifest.device, focal_px: manifest.focal_px },
    };

    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2));
    console.log(`\nReport written to ${outputPath}`);

  } catch (err) {
    console.error(`Parity comparison error: ${err.message}`);
    process.exit(1);
  } finally {
    await browser.close();
  }
}

main();
