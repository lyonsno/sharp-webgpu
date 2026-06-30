#!/usr/bin/env node
/**
 * scheduler_measurement.mjs — JSON witness for SHARP scheduler experiments.
 *
 * Requires a running SHARP-WebGPU dev server. It drives Chrome through one
 * inference run and writes the browser-reported scheduler telemetry to --out.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import puppeteer from 'puppeteer-core';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

function parseArgs() {
  const args = new Map();
  for (let i = 2; i < process.argv.length; i += 1) {
    const key = process.argv[i];
    if (!key.startsWith('--')) continue;
    const value = process.argv[i + 1];
    if (value && !value.startsWith('--')) {
      args.set(key, value);
      i += 1;
    } else {
      args.set(key, '1');
    }
  }
  return {
    port: args.get('--port') || '5175',
    image: args.get('--image') ? resolve(args.get('--image')) : null,
    out: args.get('--out') ? resolve(args.get('--out')) : null,
    scheduler: args.get('--scheduler') || '{}',
    headed: args.has('--headed'),
    timeoutMs: Number(args.get('--timeout-ms') || 600000),
    spn: args.get('--spn') !== '0',
  };
}

function parseScheduler(value) {
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch (error) {
    throw new Error(`--scheduler must be a JSON object: ${error.message}`);
  }
}

function summarizeTelemetry(telemetry) {
  const events = Array.isArray(telemetry?.events) ? telemetry.events : [];
  const vitSegments = events.filter(event => event.phase === 'vit-block-segment');
  const measurements = events.filter(event => event.phase === 'vit-segment-measurement');
  const byEncoder = new Map();
  for (const event of vitSegments) {
    const key = event.encoderLabel || 'unknown';
    if (!byEncoder.has(key)) byEncoder.set(key, []);
    byEncoder.get(key).push({
      segmentStartLayer: event.segmentStartLayer,
      segmentEndLayer: event.segmentEndLayer,
      blockCount: event.blockCount,
      yieldMs: event.yieldMs,
      waitedForSubmittedWorkDone: event.waitedForSubmittedWorkDone,
      durationMs: event.durationMs,
    });
  }
  return {
    status: telemetry ? telemetry.status || 'verified' : 'scheduler-unverified',
    requestedScheduler: telemetry?.requestedScheduler || null,
    effectiveScheduler: telemetry?.effectiveScheduler || null,
    unsupportedFields: telemetry?.unsupportedFields || [],
    eventCount: events.length,
    vitBlockSegmentCount: vitSegments.length,
    vitMeasurementCount: measurements.length,
    encoderSegments: Object.fromEntries(byEncoder),
    totalMeasuredSegmentMs: Number(measurements.reduce((sum, event) => sum + (Number(event.durationMs) || 0), 0).toFixed(3)),
  };
}

function writeReport(path, report) {
  if (!path) return;
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(report, null, 2)}\n`);
}

async function main() {
  const opts = parseArgs();
  if (!opts.out) throw new Error('expected --out <report.json>');
  const scheduler = parseScheduler(opts.scheduler);
  const url = new URL(`http://127.0.0.1:${opts.port}/`);
  url.searchParams.set('sharpScheduler', JSON.stringify(scheduler));
  const reportBase = {
    schema: 'sharp-webgpu.scheduler-measurement-witness.v0',
    ok: false,
    appUrl: url.href,
    requestedScheduler: scheduler,
    input: opts.image ? { path: opts.image } : { sample: true },
    spn: opts.spn,
  };

  const browser = await puppeteer.launch({
    executablePath: process.env.SHARP_WEBGPU_CHROME || CHROME_PATH,
    headless: opts.headed ? false : 'new',
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

  const page = await browser.newPage();
  const browserLogs = [];
  page.on('console', msg => browserLogs.push({ type: msg.type(), text: msg.text() }));
  page.on('pageerror', error => browserLogs.push({ type: 'pageerror', text: error?.message || String(error) }));

  try {
    await page.goto(url.href, { waitUntil: 'networkidle0', timeout: 30000 });
    if (opts.spn) {
      await page.$eval('#use-spn', element => {
        element.checked = true;
        element.dispatchEvent(new Event('change', { bubbles: true }));
      });
    }

    if (opts.image) {
      const fileInput = await page.$('#file-input');
      if (!fileInput) throw new Error('file input not found');
      await fileInput.uploadFile(opts.image);
    } else {
      await page.click('.sample-thumb');
    }

    const outcome = await page.waitForFunction(() => {
      const errorEl = document.getElementById('error');
      if (errorEl && errorEl.style.display !== 'none' && errorEl.textContent.trim()) {
        return JSON.stringify({ ok: false, error: errorEl.textContent.trim(), schedulerTelemetry: window.__SHARP_LAST_RUN_TELEMETRY__ || null });
      }
      const link = document.getElementById('download-ply');
      const validEl = document.getElementById('r-valid');
      const timeEl = document.getElementById('r-time');
      if ((link?.href?.startsWith('blob:') || !document.getElementById('use-spn')?.checked) && timeEl?.textContent && timeEl.textContent !== '-') {
        return JSON.stringify({
          ok: validEl?.textContent === 'OK',
          valid: validEl?.textContent || 'unknown',
          model: document.getElementById('r-model')?.textContent || null,
          grid: document.getElementById('r-grid')?.textContent || null,
          features: document.getElementById('r-features')?.textContent || null,
          time: timeEl.textContent,
          downloadText: link?.textContent || null,
          schedulerTelemetry: window.__SHARP_LAST_RUN_TELEMETRY__ || null,
        });
      }
      return false;
    }, { timeout: opts.timeoutMs });

    const result = JSON.parse(await outcome.jsonValue());
    const telemetry = result.schedulerTelemetry || null;
    const summary = summarizeTelemetry(telemetry);
    const ok = result.ok && summary.status !== 'scheduler-unverified';
    const report = {
      ...reportBase,
      ok,
      status: ok ? 'complete' : summary.status,
      result,
      summary,
      telemetry,
      browserLogs: browserLogs.slice(-80),
    };
    if (!telemetry) report.error = 'scheduler-unverified: browser did not expose window.__SHARP_LAST_RUN_TELEMETRY__';
    if (result.ok === false && result.error) report.error = result.error;
    writeReport(opts.out, report);
    if (!ok) process.exitCode = 1;
  } catch (error) {
    writeReport(opts.out, {
      ...reportBase,
      ok: false,
      status: 'failed',
      error: error?.message || String(error),
      browserLogs: browserLogs.slice(-80),
    });
    process.exitCode = 1;
  } finally {
    await browser.close().catch(() => {});
  }
}

main().catch(error => {
  console.error(error?.stack || error);
  process.exit(1);
});
