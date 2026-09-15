import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import puppeteer from 'puppeteer-core';
import { createServer } from 'vite';
import { compareStage, readCaptureChunks, PARITY_CHUNK_BYTES } from '../tools/parity_transport.mjs';
import { createParityReport, finishParityReport, writeParityReport } from '../tools/parity_report.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const outputPath = process.env.PARITY_BROWSER_SMOKE_OUTPUT || '/tmp/sharp-parity-browser.json';
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const report = { status: 'failed', failurePhase: 'setup', route: null, browser: null, checks: [] };
let server;
let browser;
try {
  writeParityReport(outputPath, report);
  const kitPath = path.join(root, 'node_modules/@kaminos/webgpu-inference-kit/src/parity-primitives.js');
  report.sources = {
    kit: digest(fs.readFileSync(kitPath)),
    session: digest(fs.readFileSync(path.join(root, 'src/lib/parity_capture.js'))),
    transport: digest(fs.readFileSync(path.join(root, 'tools/parity_transport.mjs'))),
  };
  server = await createServer({ root, configFile: false, server: { host: '127.0.0.1', port: 0, open: false } });
  await server.listen();
  report.route = `http://127.0.0.1:${server.httpServer.address().port}/tests/parity-browser.html`;
  browser = await puppeteer.launch({
    executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true, args: ['--no-sandbox', '--disable-gpu'],
  });
  report.browser = await browser.version();
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto(report.route);
  await page.waitForFunction(() => window.__parityTestLoaded);
  report.effectiveRoute = page.url();
  report.failurePhase = 'multi-message-comparison';
  // Larger than two normal CDP chunks, without model or GPU execution.
  const count = PARITY_CHUNK_BYTES / 4 * 2 + 3;
  const values = Float32Array.from({ length: count }, (_, i) => (i % 1024) / 16);
  await page.evaluate(count => {
    window.__sharpParity = window.createSharpParitySession('browser-run');
    window.__sharpParity.capture('encoder', Float32Array.from({ length: count }, (_, i) => (i % 1024) / 16), {
      shape: [count], layout: 'N',
    });
  }, count);
  const identity = { runId: 'browser-run', stageId: 'encoder' };
  const comparison = await compareStage(page, identity, values, { sampling: { mode: 'stride', stride: 3 } });
  assert.equal(comparison.metrics.exactMatch, true);
  assert.equal(comparison.sourceElementCount, count);
  assert.equal(comparison.comparedElementCount, Math.ceil(count / 3));
  report.checks.push('multi-message reference comparison preserves source count and stride');

  const exported = createHash('sha256');
  let exportedBytes = 0;
  let chunkCount = 0;
  for await (const bytes of readCaptureChunks(page, identity, values.byteLength)) {
    exported.update(bytes); exportedBytes += bytes.length; chunkCount++;
  }
  assert.equal(exportedBytes, values.byteLength);
  assert.equal(chunkCount, 3);
  const expectedSha = digest(new Uint8Array(values.buffer));
  assert.equal(exported.digest('hex'), expectedSha);
  report.transfer = { byteLength: exportedBytes, chunkCount, sha256: expectedSha, comparison };
  report.checks.push('raw export is byte-identical across three messages');
  await assert.rejects(compareStage(page, { ...identity, runId: 'stale' }, values), /run/);
  await page.evaluate(identity => {
    window.__sharpParity.beginReference({ ...identity, byteLength: window.__sharpParity.describe(identity.stageId).byteLength });
    window.__sharpParity.appendReference({ ...identity, byteOffset: 0, payloadBase64: 'AAAAAA==' });
  }, identity);
  await assert.rejects(page.evaluate(identity => window.__sharpParity.compareReference(identity), identity), /incomplete/);
  await page.evaluate(() => window.__sharpParity.discardReference());
  report.checks.push('stale run and incomplete reference reject');

  await page.evaluate(() => {
    const session = window.__sharpParity;
    window.__sharpParity = { ...session, readCaptureRange(input) {
      const result = session.readCaptureRange(input);
      return { ...result, payloadBase64: 'AAAAAA==' };
    } };
  });
  await assert.rejects(async () => {
    for await (const unused of readCaptureChunks(page, identity, values.byteLength)) void unused;
  }, /range mismatch/);
  report.checks.push('short raw export response rejects');

  const saved = createParityReport({ runId: identity.runId });
  saved.stages.encoder = { status: 'compared', comparison };
  finishParityReport(saved, { means: comparison }, 1);
  const consumerPath = `${outputPath}.consumer.json`;
  writeParityReport(consumerPath, saved);
  assert.equal(JSON.parse(fs.readFileSync(consumerPath)).stages.encoder.comparison.sourceElementCount, count);
  const failure = spawnSync(process.execPath, ['tools/parity_compare.mjs', '--manifest', `${outputPath}.missing-manifest`, '--output', consumerPath], {
    cwd: root, encoding: 'utf8',
  });
  assert.equal(failure.status, 1);
  const failed = JSON.parse(fs.readFileSync(consumerPath));
  assert.equal(failed.status, 'failed');
  assert.equal(failed.failurePhase, 'setup');
  assert.deepEqual(failed.stages, {});
  report.checks.push('CLI setup failure replaces stale success with failure report');
  await page.evaluate(() => window.__sharpParity.clear());
  assert.deepEqual(await page.evaluate(() => window.__sharpParity.stageIds()), []);
  assert.deepEqual(pageErrors, []);
  report.status = 'succeeded';
  report.failurePhase = null;
} catch (error) {
  report.error = { name: error.name, message: error.message, stack: error.stack };
  process.exitCode = 1;
  console.error(error.message);
} finally {
  writeParityReport(outputPath, report);
  if (browser) await browser.close();
  if (server) await server.close();
}
console.log(`SHARP parity browser ${report.status}: ${outputPath}`);
