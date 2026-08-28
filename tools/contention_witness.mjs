#!/usr/bin/env node
/**
 * Run a SHARP browser inference with optional same-page WebGPU contention and
 * write a validated contention witness report.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

import {
  createSharpBackgroundHeartbeatReport,
  createSharpContentionWitnessFailureReport,
  SHARP_CONTENTION_WITNESS_SCHEMA,
  SHARP_ROUTE_ID,
  validateSharpContentionWitnessReport,
} from './contention_witness_report.mjs';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const __dirname = path.dirname(fileURLToPath(import.meta.url));

function argValue(args, name, fallback = null) {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : fallback;
}

function parseArgs() {
  const args = process.argv.slice(2);
  const mode = argValue(args, '--mode', 'contention');
  return {
    mode,
    port: argValue(args, '--port', '5175'),
    out: argValue(args, '--out', `/tmp/sharp-contention-witness-${mode}.json`),
    screenshot: argValue(args, '--screenshot', `/tmp/sharp-contention-witness-${mode}.png`),
    image: argValue(args, '--image', null),
    sharpScheduler: argValue(args, '--sharp-scheduler', null),
    headed: args.includes('--headed'),
    timeoutMs: Number(argValue(args, '--timeout-ms', '600000')),
    episodeId: argValue(args, '--episode-id', null),
  };
}

function buildUrl(opts) {
  const url = new URL(`http://localhost:${opts.port}/`);
  if (opts.sharpScheduler) url.searchParams.set('sharpScheduler', opts.sharpScheduler);
  if (opts.episodeId) url.searchParams.set('sharpRunId', opts.episodeId);
  return url.toString();
}

async function installProbe(page, mode) {
  await page.evaluate(async ({ mode }) => {
    const probe = {
      mode,
      running: true,
      timeOriginEpochMs: performance.timeOrigin,
      startedAt: performance.now(),
      rafFrames: 0,
      frameGaps: [],
      frameGapIntervals: [],
      lastFrameAt: performance.now(),
      contender: {
        enabled: mode !== 'baseline',
        submitted: 0,
        completed: 0,
        errors: [],
        inferenceWindow: null,
      },
    };

    function frame(now) {
      if (!probe.running) return;
      probe.rafFrames += 1;
      const gap = now - probe.lastFrameAt;
      probe.frameGaps.push(gap);
      probe.frameGapIntervals.push({
        startMs: probe.lastFrameAt,
        endMs: now,
        durationMs: gap,
      });
      probe.lastFrameAt = now;
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);

    async function runContender() {
      try {
        if (!navigator.gpu) throw new Error('WebGPU unavailable for contender');
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) throw new Error('No contender WebGPU adapter');
        const device = await adapter.requestDevice();
        const module = device.createShaderModule({
          code: `
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
              var x = data[id.x];
              for (var i: u32 = 0u; i < 64u; i = i + 1u) {
                x = (x * 1.0001221) + f32((i & 7u) + 1u) * 0.00003125;
              }
              data[id.x] = x;
            }
          `,
        });
        const pipeline = device.createComputePipeline({
          layout: 'auto',
          compute: { module, entryPoint: 'main' },
        });
        const size = 65536 * 4;
        const buffer = device.createBuffer({
          size,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer } }],
        });
        while (probe.running) {
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(1024);
          pass.end();
          device.queue.submit([encoder.finish()]);
          probe.contender.submitted += 1;
          await device.queue.onSubmittedWorkDone();
          probe.contender.completed += 1;
          await new Promise(resolve => setTimeout(resolve, 0));
        }
        buffer.destroy();
      } catch (error) {
        probe.contender.errors.push(error?.message || String(error));
      }
    }

    if (probe.contender.enabled) runContender();

    window.__sharpContentionProbe = {
      markInferenceStart(runId) {
        const startMs = performance.now();
        probe.contender.inferenceWindow = {
          runId,
          startMs,
          startEpochMs: performance.timeOrigin + startMs,
          endMs: null,
          endEpochMs: null,
          submittedAtStart: probe.contender.submitted,
          completedAtStart: probe.contender.completed,
          submittedAtEnd: probe.contender.submitted,
          completedAtEnd: probe.contender.completed,
        };
      },
      markInferenceEnd(runId) {
        const window = probe.contender.inferenceWindow || {
          runId,
          submittedAtStart: probe.contender.submitted,
          completedAtStart: probe.contender.completed,
        };
        if (window.runId !== runId) {
          throw new Error(`inference window run id mismatch: expected ${window.runId}, got ${runId}`);
        }
        if (Number.isFinite(window.endMs)) return;
        window.endMs = performance.now();
        window.endEpochMs = performance.timeOrigin + window.endMs;
        window.submittedAtEnd = probe.contender.submitted;
        window.completedAtEnd = probe.contender.completed;
        probe.contender.inferenceWindow = window;
      },
      stop() {
        probe.running = false;
      },
      snapshot() {
        const inferenceWindow = probe.contender.inferenceWindow || {
          startMs: null,
          endMs: null,
          submittedAtStart: probe.contender.submitted,
          completedAtStart: probe.contender.completed,
        };
        const scopedGapIntervals = Number.isFinite(inferenceWindow.startMs) && Number.isFinite(inferenceWindow.endMs)
          ? probe.frameGapIntervals
            .map(gap => {
              const startMs = Math.max(gap.startMs, inferenceWindow.startMs);
              const endMs = Math.min(gap.endMs, inferenceWindow.endMs);
              return { startMs, endMs, durationMs: endMs - startMs };
            })
            .filter(gap => gap.durationMs > 0)
          : [];
        const gaps = scopedGapIntervals.map(gap => gap.durationMs).sort((a, b) => a - b);
        const p95Index = gaps.length ? Math.min(gaps.length - 1, Math.floor(gaps.length * 0.95)) : 0;
        const window = probe.contender.inferenceWindow || {
          startMs: null,
          endMs: null,
          submittedAtStart: probe.contender.submitted,
          completedAtStart: probe.contender.completed,
          submittedAtEnd: probe.contender.submitted,
          completedAtEnd: probe.contender.completed,
        };
        const submittedDelta = window.submittedAtEnd - window.submittedAtStart;
        const completedDelta = window.completedAtEnd - window.completedAtStart;
        const worstFrameGaps = scopedGapIntervals
          .slice()
          .sort((a, b) => b.durationMs - a.durationMs)
          .slice(0, 8);
        return {
          startedAtMs: probe.startedAt,
          timeOriginEpochMs: probe.timeOriginEpochMs,
          inferenceWindow: {
            runId: window.runId,
            startMs: window.startMs,
            endMs: window.endMs,
            startEpochMs: window.startEpochMs,
            endEpochMs: window.endEpochMs,
          },
          rafFrames: scopedGapIntervals.length,
          frameGapIntervals: scopedGapIntervals,
          maxFrameGapMs: gaps.length ? gaps[gaps.length - 1] : 0,
          p95FrameGapMs: gaps.length ? gaps[p95Index] : 0,
          longFrameCount: gaps.filter(gap => gap > 50).length,
          worstFrameGaps,
          contender: {
            ...probe.contender,
            inferenceWindow: {
              runId: window.runId,
              startMs: window.startMs,
              endMs: window.endMs,
              startEpochMs: window.startEpochMs,
              endEpochMs: window.endEpochMs,
              submittedAtStart: window.submittedAtStart,
              completedAtStart: window.completedAtStart,
              submittedAtEnd: window.submittedAtEnd,
              completedAtEnd: window.completedAtEnd,
              submittedDelta,
              completedDelta,
            },
            progressDuringInference: completedDelta > 0,
          },
        };
      },
    };
  }, { mode });
}

async function triggerInference(page, image) {
  await page.evaluate(() => {
    const checkbox = document.getElementById('use-spn');
    if (!checkbox) throw new Error('use-spn checkbox missing');
    checkbox.checked = true;
  });

  if (image) {
    const fileInput = await page.$('#file-input');
    if (!fileInput) throw new Error('file input missing');
    await fileInput.uploadFile(image);
    return { source: 'file', artifactId: image };
  }

  const thumb = await page.$('.sample-thumb');
  if (!thumb) throw new Error('sample thumbnail missing');
  await page.click('.sample-thumb');
  return { source: 'sample', artifactId: 'public/samples/sample_1.jpg' };
}

function parseMs(text) {
  const match = String(text || '').match(/([0-9]+(?:\.[0-9]+)?)/);
  return match ? Number(match[1]) : null;
}

function parseGaussians(text) {
  const kMatch = String(text || '').match(/([0-9]+(?:\.[0-9]+)?)K Gaussians/i);
  if (kMatch) return Math.round(Number(kMatch[1]) * 1000);
  const match = String(text || '').match(/([0-9][0-9,]*)\s+Gaussians/i);
  return match ? Number(match[1].replaceAll(',', '')) : null;
}

async function collectReport(page, opts, input) {
  const data = await page.evaluate(() => {
    const debug = window.__sharpDebug?.lastRun || null;
    const probe = window.__sharpContentionProbe?.snapshot?.() || null;
    const download = document.getElementById('download-ply');
    return {
      debug,
      probe,
      dom: {
        model: document.getElementById('r-model')?.textContent || null,
        weights: document.getElementById('r-weights')?.textContent || null,
        features: document.getElementById('r-features')?.textContent || null,
        time: document.getElementById('r-time')?.textContent || null,
        valid: document.getElementById('r-valid')?.textContent || null,
        plyAvailable: Boolean(download?.href),
      },
    };
  });

  const debug = data.debug || {};
  const probe = data.probe || { contender: {} };
  const numGaussians = debug.outputs?.numGaussians || parseGaussians(data.dom.features);
  const timeMs = Number.isFinite(debug.inferenceElapsedMs) ? debug.inferenceElapsedMs : parseMs(data.dom.time);
  const scheduler = debug.schedulerTelemetry || debug.sharpScheduler || {};
  const receiptSplat = Array.isArray(debug.route?.receipt?.outputs)
    ? debug.route.receipt.outputs.find(output => output?.role === 'splat-candidate')
    : null;
  const responsiveness = {
    rafFrames: probe.rafFrames || 0,
    maxFrameGapMs: probe.maxFrameGapMs || 0,
    p95FrameGapMs: probe.p95FrameGapMs || 0,
    longFrameCount: probe.longFrameCount || 0,
  };
  const backgroundHeartbeat = createSharpBackgroundHeartbeatReport({
    scheduler,
    probe,
    responsiveness,
  });

  return {
    schema: SHARP_CONTENTION_WITNESS_SCHEMA,
    runId: scheduler.runId || opts.episodeId,
    createdAt: new Date().toISOString(),
    route: {
      requestedRouteId: debug.route?.requestedRouteId || SHARP_ROUTE_ID,
      effectiveRouteId: debug.route?.effectiveRouteId || null,
      receipt: debug.route?.receipt || null,
      evidence: debug.route?.evidence || null,
      receiptError: debug.route?.receiptError || null,
    },
    mode: opts.mode,
    input,
    inference: {
      ok: debug.status === 'real' && data.dom.valid === 'OK',
      error: debug.error || null,
      valid: data.dom.valid,
      timeMs,
      model: data.dom.model,
      weights: data.dom.weights,
      phases: debug.phases || [],
      outputs: {
        numGaussians,
        plyAvailable: Boolean(debug.outputs?.plyAvailable || data.dom.plyAvailable),
        plyByteLength: debug.outputs?.plyByteLength || null,
        plySha256: receiptSplat?.sha256 || null,
        completeness: debug.status === 'real' && receiptSplat ? 'complete' : 'partial',
      },
    },
    responsiveness: {
      ...responsiveness,
    },
    backgroundHeartbeat,
    contender: {
      enabled: opts.mode !== 'baseline',
      submitted: probe.contender?.submitted || 0,
      completed: probe.contender?.completed || 0,
      inferenceWindow: probe.contender?.inferenceWindow || {
        submittedAtStart: 0,
        completedAtStart: 0,
        submittedAtEnd: 0,
        completedAtEnd: 0,
        submittedDelta: 0,
        completedDelta: 0,
      },
      progressDuringInference: Boolean(probe.contender?.progressDuringInference),
      errors: probe.contender?.errors || [],
    },
    scheduler: {
      runId: scheduler.runId || null,
      mode: scheduler.requestedScheduler?.mode || scheduler.effectiveScheduler?.mode || scheduler.mode || 'unknown',
      verificationState: scheduler.verificationState || scheduler.status || 'scheduler-unverified',
      requestedScheduler: scheduler.requestedScheduler || scheduler.requested || null,
      effectiveScheduler: scheduler.effectiveScheduler || scheduler.effective || null,
      unsupportedFields: scheduler.unsupportedFields || [],
    },
  };
}

function writeJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
}

export async function runContentionWitness(rawOpts, {
  launchBrowser = launchOptions => puppeteer.launch(launchOptions),
} = {}) {
  const opts = {
    ...rawOpts,
    episodeId: rawOpts.episodeId || `sharp-contention:${rawOpts.mode}:${Date.now()}`,
  };
  const url = buildUrl(opts);
  let browser = null;
  let page = null;
  let failurePhase = 'browser-launch';
  const consoleLines = [];
  const pageErrors = [];
  let input = { source: 'unknown', artifactId: 'unknown' };
  try {
    browser = await launchBrowser({
      executablePath: CHROME_PATH,
      headless: !opts.headed,
      args: [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan',
        '--disable-gpu-sandbox',
        '--no-sandbox',
        '--disable-gpu-shader-disk-cache',
        '--window-size=1280,900',
      ],
      defaultViewport: { width: 1280, height: 900 },
    });

    failurePhase = 'page-create';
    page = await browser.newPage();
    page.on('console', msg => {
      const text = msg.text();
      consoleLines.push(text);
      console.log(`[page] ${text}`);
    });
    page.on('pageerror', error => {
      const text = error?.message || String(error);
      pageErrors.push(text);
      console.error(`[pageerror] ${text}`);
    });

    failurePhase = 'navigation';
    console.log(`[contention] Loading ${url}`);
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });
    failurePhase = 'probe-install';
    await installProbe(page, opts.mode);
    failurePhase = 'inference-trigger';
    input = await triggerInference(page, opts.image);

    failurePhase = 'app-inference';
    await page.waitForFunction(() => {
      const errorEl = document.getElementById('error');
      if (errorEl && errorEl.style.display !== 'none' && errorEl.textContent) {
        return true;
      }
      const debug = window.__sharpDebug?.lastRun;
      return (debug?.status === 'real' && debug.route?.receipt && debug.route?.evidence)
        || debug?.status === 'error';
    }, { timeout: opts.timeoutMs });

    await page.evaluate(() => window.__sharpContentionProbe?.stop?.());
    await new Promise(resolve => setTimeout(resolve, 100));
    failurePhase = 'screenshot';
    await page.screenshot({ path: opts.screenshot, fullPage: true });

    failurePhase = 'evidence-collection';
    const report = await collectReport(page, opts, input);
    if (pageErrors.length) report.pageErrors = pageErrors;
    failurePhase = 'validating-report';
    const validation = validateSharpContentionWitnessReport(report);
    if (!validation.ok) {
      const failurePhase = report.inference.ok
        ? 'validating-report'
        : (report.backgroundHeartbeat?.inferenceWindow ? 'app-inference' : 'before-inference-window');
      const failure = createSharpContentionWitnessFailureReport({
        candidateReport: report,
        failurePhase,
        error: report.inference.error || validation.errors.join('; ') || 'SHARP contention witness validation failed',
        validation,
      });
      writeJson(opts.out, failure);
      console.error(`[contention] FAIL: ${failure.error}`);
      console.error(`[contention] Failure report: ${opts.out}`);
      return { ok: false, report: failure, out: opts.out };
    }
    report.validation = validation;
    writeJson(opts.out, report);

    console.log(`[contention] Report: ${opts.out}`);
    console.log(`[contention] Screenshot: ${opts.screenshot}`);
    console.log(JSON.stringify({
      ok: validation.ok,
      mode: report.mode,
      timeMs: report.inference.timeMs,
      completed: report.contender.completed,
      maxFrameGapMs: report.responsiveness.maxFrameGapMs,
      errors: validation.errors,
      warnings: validation.warnings,
    }, null, 2));
    return { ok: true, report, out: opts.out };
  } catch (error) {
    let candidateReport = {
      runId: opts.episodeId,
      createdAt: new Date().toISOString(),
      mode: opts.mode,
      input,
    };
    let evidenceCollectionError = null;
    try {
      if (page) candidateReport = await collectReport(page, opts, input);
    } catch (collectionError) {
      evidenceCollectionError = collectionError?.message || String(collectionError);
    }
    const failure = createSharpContentionWitnessFailureReport({
      candidateReport,
      failurePhase,
      error: error?.message || String(error),
    });
    failure.consoleTail = consoleLines.slice(-60);
    failure.pageErrors = pageErrors;
    if (evidenceCollectionError) failure.evidenceCollectionError = evidenceCollectionError;
    writeJson(opts.out, failure);
    if (page) await page.screenshot({ path: opts.screenshot, fullPage: true }).catch(() => {});
    console.error(`[contention] FAIL: ${failure.error}`);
    console.error(`[contention] Failure report: ${opts.out}`);
    return { ok: false, report: failure, out: opts.out };
  } finally {
    if (browser) await browser.close().catch(() => {});
  }
}

async function main() {
  const result = await runContentionWitness(parseArgs());
  process.exitCode = result.ok ? 0 : 1;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  main().catch(error => {
    console.error(error);
    process.exitCode = 1;
  });
}
