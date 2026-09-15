#!/usr/bin/env node
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { compareWebGpuParityArrays } from '@kaminos/webgpu-inference-kit';
import { compareStage, readCaptureChunks } from './parity_transport.mjs';
import { createParityReport, finishParityReport, writeParityReport } from './parity_report.mjs';

function loadReferenceDump(directory, info) {
  if (!info?.file) throw new Error('reference dump is missing');
  const bytes = fs.readFileSync(path.join(directory, info.file));
  if (!bytes.length || bytes.length % 4) throw new Error(`invalid Float32 reference length: ${info.file}`);
  return new Float32Array(Uint8Array.from(bytes).buffer);
}

const stages = [
  'input_normalized', 'monodepth_disparity',
  'spn_encoding_0', 'spn_encoding_1', 'spn_encoding_2', 'spn_encoding_3', 'spn_encoding_4',
  'feature_input', 'gd_decoder_out', 'gd_skip_out', 'gd_fusion_out',
  'geometry_features', 'texture_features', 'geom_deltas', 'tex_deltas',
];

async function main() {
  const args = process.argv.slice(2);
  const option = (name, fallback) => args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
  const outputPath = option('--output', '/tmp/parity-report.json');
  const manifestPath = option('--manifest', 'public/reference_dumps/manifest.json');
  const report = createParityReport({ runId: randomUUID() });
  let browser;
  let page;
  try {
    // Replace any previous report even when setup fails.
    writeParityReport(outputPath, report);
    const stride = Number(option('--stride', '1'));
    if (!Number.isSafeInteger(stride) || stride <= 0) throw new Error('--stride must be a positive integer');
    const sampling = stride === 1 ? { mode: 'all' } : { mode: 'stride', stride, offset: 0 };
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const manifestDir = path.dirname(manifestPath);
    report.reference = { manifest: path.resolve(manifestPath), dtype: manifest.dtype, device: manifest.device, image: manifest.image };
    report.sampling = sampling;
    const url = `http://localhost:${option('--port', '5175')}/`;
    report.requestedRoute = url;
    browser = await puppeteer.launch({
      executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      headless: !args.includes('--headed'),
      protocolTimeout: 600000,
      args: ['--enable-unsafe-webgpu', '--no-sandbox', '--window-size=1280,900'],
    });
    page = await browser.newPage();
    page.on('pageerror', error => console.error('PAGE ERROR:', error.message));
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });
    report.effectiveRoute = page.url();
    report.browser = await browser.version();
    await page.evaluate(runId => {
      window.__enableParityCapture = true;
      window.__sharpParityRequestedRunId = runId;
    }, report.runId);
    report.failurePhase = 'inference';
    writeParityReport(outputPath, report);
    await page.$eval('#use-spn', element => { element.checked = true; });
    await page.click('.sample-thumb');
    await page.waitForFunction(runId => {
      const status = window.__sharpParityStatus;
      return status?.runId === runId && status.status !== 'running';
    }, { timeout: 600000 }, report.runId);
    const status = await page.evaluate(() => window.__sharpParityStatus);
    if (status.status !== 'completed') throw new Error(status.error || 'inference failed');
    console.log('Inference completed; comparing captured stages.');

    report.failurePhase = 'stage-comparison';
    for (const stageId of stages) {
      const description = await page.evaluate(stage => window.__sharpParity.describe(stage), stageId);
      if (!manifest.dumps[stageId] || !description) {
        report.stages[stageId] = {
          status: 'missing',
          reason: !manifest.dumps[stageId] ? 'reference missing' : 'capture missing',
        };
        writeParityReport(outputPath, report);
        continue;
      }
      const reference = loadReferenceDump(manifestDir, manifest.dumps[stageId]);
      // This dump convention differs from the tensor supplied to SHARP.
      if (stageId === 'input_normalized') {
        for (let i = 0; i < reference.length; i++) reference[i] = 2 * reference[i] - 1;
      }
      report.stages[stageId] = { status: 'comparing', capture: description };
      writeParityReport(outputPath, report);
      const comparison = await compareStage(page, { runId: report.runId, stageId }, reference, {
        sampling, interior: stageId === 'geom_deltas' || stageId === 'tex_deltas',
      });
      report.stages[stageId] = {
        status: 'compared', capture: description, comparison,
        referenceNormalization: stageId === 'input_normalized' ? 'zero-one-to-minus-one-one' : 'none',
      };
      report.lastCompletedStage = stageId;
      writeParityReport(outputPath, report);
      console.log(`${stageId}: max=${comparison.metrics.maxAbsoluteError} rms=${comparison.metrics.rootMeanSquareError} n=${comparison.comparedElementCount}/${comparison.sourceElementCount}`);
    }

    if (args.includes('--save-captures')) {
      report.failurePhase = 'raw-export';
      const saveDir = option('--save-captures');
      fs.mkdirSync(saveDir, { recursive: true });
      const wanted = option('--capture-stages', stages.join(',')).split(',');
      report.exports = {};
      for (const stageId of wanted) {
        if (!stages.includes(stageId)) throw new Error(`unknown capture stage: ${stageId}`);
        const description = await page.evaluate(stage => window.__sharpParity.describe(stage), stageId);
        if (!description) {
          report.exports[stageId] = { status: 'missing' };
          continue;
        }
        const outPath = path.join(saveDir, `${stageId}.bin`);
        const file = fs.openSync(outPath, 'w');
        let written = 0;
        report.exports[stageId] = { status: 'writing', path: path.resolve(outPath), byteLength: description.byteLength, writtenBytes: 0 };
        writeParityReport(outputPath, report);
        try {
          for await (const bytes of readCaptureChunks(page, { runId: report.runId, stageId }, description.byteLength)) {
            let offset = 0;
            while (offset < bytes.length) offset += fs.writeSync(file, bytes, offset, bytes.length - offset);
            written += bytes.length;
            report.exports[stageId].writtenBytes = written;
          }
          report.exports[stageId].status = 'completed';
        } finally {
          fs.closeSync(file);
          writeParityReport(outputPath, report);
        }
      }
    }

    report.failurePhase = 'ply-comparison';
    writeParityReport(outputPath, report);
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
      throw new Error('Could not extract PLY data from page');
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

    // Quaternion sign is decomposition-arbitrary (q and -q are the same
    // rotation): align each GPU quat's sign to the reference before comparing.
    for (let i = 0; i + 3 < gpuQuats.length; i += 4) {
      const dot = gpuQuats[i] * refQuats[i] + gpuQuats[i + 1] * refQuats[i + 1] +
                  gpuQuats[i + 2] * refQuats[i + 2] + gpuQuats[i + 3] * refQuats[i + 3];
      if (dot < 0) {
        gpuQuats[i] *= -1; gpuQuats[i + 1] *= -1; gpuQuats[i + 2] *= -1; gpuQuats[i + 3] *= -1;
      }
    }

    const comparisons = {
      means: compareWebGpuParityArrays(gpuMeans, refMeans),
      scale_log: compareWebGpuParityArrays(gpuScaleLog, refScaleLog),
      colors_sh: compareWebGpuParityArrays(gpuColorsSH, refColorsSH),
      opacity_logit: compareWebGpuParityArrays(gpuOpacityLogit, refOpacityLogit),
      quaternions: compareWebGpuParityArrays(gpuQuats, refQuats),
    };


    finishParityReport(report, comparisons, plyData.numVertices);
    for (const [name, result] of Object.entries(comparisons)) {
      console.log(`${name}: max=${result.metrics.maxAbsoluteError} rms=${result.metrics.rootMeanSquareError}`);
    }
    console.log(`Comparison completed: ${report.summary.comparedStages} intermediate stages, ${report.summary.missingStages.length} missing. Report: ${outputPath}`);
  } catch (error) {
    report.status = 'failed';
    report.error = { name: error.name, message: error.message };
    for (const result of Object.values(report.stages)) {
      if (result.status === 'comparing') { result.status = 'failed'; result.error = error.message; }
    }
    for (const result of Object.values(report.exports || {})) {
      if (result.status === 'writing') { result.status = 'failed'; result.error = error.message; }
    }
    process.exitCode = 1;
    console.error(error.message);
  } finally {
    writeParityReport(outputPath, report);
    try {
      if (page) await page.evaluate(runId => {
        if (window.__sharpParity?.runId === runId) window.__sharpParity.clear();
      }, report.runId);
    } catch (error) {
      report.cleanupError = error.message;
      writeParityReport(outputPath, report);
    } finally {
      if (browser) await browser.close();
    }
  }
}

await main();
