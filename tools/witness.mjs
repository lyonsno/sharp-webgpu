#!/usr/bin/env node
/**
 * witness.mjs — Tight-loop inference witness for SHARP-WebGPU development.
 *
 * Launches headless Chrome with WebGPU, loads the app, runs inference,
 * reads back GPU buffer contents, and reports structured results to stdout.
 * Captures all console output, errors, and screenshots.
 *
 * Usage:
 *   node tools/witness.mjs [--port 5175] [--headed] [--image path/to/image.jpg]
 *
 * Output:
 *   - All page console output (prefixed with [page])
 *   - Structured JSON results on success
 *   - Screenshot at /tmp/sharp-witness.png
 *   - Exit code 0 on success, 1 on failure
 *
 * The agent can call this after every change to verify correctness without
 * requiring operator visual inspection.
 */

import puppeteer from 'puppeteer-core';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const __dirname = path.dirname(fileURLToPath(import.meta.url));

function parseArgs() {
  const args = process.argv.slice(2);
  return {
    port: args.includes('--port') ? args[args.indexOf('--port') + 1] : '5175',
    headed: args.includes('--headed'),
    image: args.includes('--image') ? args[args.indexOf('--image') + 1] : null,
  };
}

async function main() {
  const opts = parseArgs();
  const url = `http://localhost:${opts.port}/`;

  const browser = await puppeteer.launch({
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

  const page = await browser.newPage();

  const consoleLines = [];
  const errors = [];

  page.on('console', msg => {
    const text = msg.text();
    consoleLines.push(text);
    // Print everything — the agent needs full visibility
    console.log(`[page] ${text}`);
  });

  page.on('pageerror', err => {
    const text = `PAGE_ERROR: ${err.message}`;
    errors.push(text);
    console.error(text);
  });

  let exitCode = 0;

  try {
    // Load page
    console.log(`[witness] Loading ${url}`);
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });

    // Trigger inference: either upload a file or click sample
    if (opts.image) {
      console.log(`[witness] Uploading image: ${opts.image}`);
      const fileInput = await page.$('#file-input');
      await fileInput.uploadFile(opts.image);
    } else {
      // Click sample thumbnail
      const hasSample = await page.$('.sample-thumb');
      if (hasSample) {
        console.log('[witness] Clicking sample image');
        await page.click('.sample-thumb');
      } else {
        // Fallback: upload moge test fixture
        const mogePath = path.resolve(__dirname, '../../moge-webgpu/public/test_fixtures/input.png');
        if (fs.existsSync(mogePath)) {
          console.log(`[witness] Uploading fallback: ${mogePath}`);
          const fileInput = await page.$('#file-input');
          await fileInput.uploadFile(mogePath);
        } else {
          throw new Error('No sample image available and no --image flag');
        }
      }
    }

    // Wait for results or error — check both timing AND output validation
    console.log('[witness] Waiting for inference...');
    const outcome = await page.waitForFunction(() => {
      const timeEl = document.getElementById('r-time');
      const validEl = document.getElementById('r-valid');
      const errorEl = document.getElementById('error');

      if (errorEl && errorEl.style.display !== 'none' && errorEl.textContent) {
        return JSON.stringify({ ok: false, error: errorEl.textContent });
      }
      if (timeEl && timeEl.textContent && timeEl.textContent !== '-') {
        return JSON.stringify({
          ok: true,
          model: document.getElementById('r-model')?.textContent,
          weights: document.getElementById('r-weights')?.textContent,
          grid: document.getElementById('r-grid')?.textContent,
          features: document.getElementById('r-features')?.textContent,
          time: document.getElementById('r-time')?.textContent,
          valid: validEl?.textContent || 'unknown',
        });
      }
      return false;
    }, { timeout: 300000 });

    const result = JSON.parse(await outcome.jsonValue());

    // Check output validation — fail on anything that isn't OK
    if (result.ok && result.valid && result.valid !== 'OK') {
      result.ok = false;
      result.error = `Output validation failed: ${result.valid}`;
    }

    // Screenshot
    await page.screenshot({ path: '/tmp/sharp-witness.png', fullPage: true });

    if (!result.ok) {
      console.error(`\n[witness] FAIL: ${result.error}`);
      exitCode = 1;
    } else {
      console.log('\n[witness] PASS');
      console.log(JSON.stringify(result, null, 2));
    }

    // Dump any page errors
    if (errors.length > 0) {
      console.log(`\n[witness] ${errors.length} page error(s):`);
      for (const e of errors) console.log(`  ${e}`);
      exitCode = 1;
    }

  } catch (err) {
    console.error(`\n[witness] EXCEPTION: ${err.message}`);
    await page.screenshot({ path: '/tmp/sharp-witness-error.png', fullPage: true }).catch(() => {});
    exitCode = 1;

    // Dump console for debugging
    if (consoleLines.length > 0) {
      console.log('\n[witness] Console dump:');
      for (const line of consoleLines.slice(-30)) {
        console.log(`  ${line}`);
      }
    }
  } finally {
    await browser.close();
  }

  process.exit(exitCode);
}

main();
