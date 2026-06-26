#!/usr/bin/env node
/**
 * Backbone smoke test: load page, drop test image, verify ViT forward pass.
 *
 * Usage:
 *   node tools/backbone_smoke.mjs [--port 5176] [--headed]
 */

import puppeteer from 'puppeteer-core';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Find a test image — use moge-webgpu's if available, otherwise create a synthetic one
function findTestImage() {
  const mogePath = path.resolve(__dirname, '../../moge-webgpu/public/test_fixtures/input.png');
  if (fs.existsSync(mogePath)) return mogePath;
  // Fallback: any png/jpg in the current tree
  return null;
}

async function main() {
  const args = process.argv.slice(2);
  const port = args.includes('--port') ? args[args.indexOf('--port') + 1] : '5176';
  const headed = args.includes('--headed');
  const url = `http://localhost:${port}/`;

  const testImage = findTestImage();
  if (!testImage) {
    console.error('No test image found. Place an image at public/test_fixtures/input.png or use moge-webgpu fixture.');
    process.exit(1);
  }

  console.log(`SHARP backbone smoke — ${url}`);
  console.log(`Test image: ${testImage}\n`);

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: !headed,
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

  const consoleOutput = [];
  page.on('console', msg => {
    const text = msg.text();
    consoleOutput.push(text);
    console.log(`  [page] ${text}`);
  });
  page.on('pageerror', err => {
    console.error('PAGE ERROR:', err.message);
    consoleOutput.push(`ERROR: ${err.message}`);
  });

  try {
    console.log('Loading page...');
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });

    // Programmatically trigger file input with test image
    console.log('Triggering inference with test image...');
    const fileInput = await page.$('#file-input');
    await fileInput.uploadFile(testImage);

    // Wait for inference — the weights are 1.3 GB so this could take a while
    console.log('Waiting for backbone inference (weights: 1.3 GB, may take 30-120s on first load)...');

    // Poll for completion: check results table and validation
    const result = await page.waitForFunction(() => {
      const timeEl = document.getElementById('r-time');
      const validEl = document.getElementById('r-valid');
      const errorEl = document.getElementById('error');
      if (errorEl && errorEl.style.display !== 'none' && errorEl.textContent) {
        return JSON.stringify({ ok: false, error: errorEl.textContent });
      }
      if (timeEl && timeEl.textContent && timeEl.textContent !== '-') {
        return JSON.stringify({
          ok: true,
          time: timeEl.textContent,
          grid: document.getElementById('r-grid')?.textContent,
          valid: validEl?.textContent || 'unknown',
        });
      }
      return false;
    }, { timeout: 300000 });

    const resultValue = JSON.parse(await result.jsonValue());

    // Check output validation
    if (resultValue.ok && resultValue.valid && resultValue.valid !== 'OK') {
      resultValue.ok = false;
      resultValue.error = `Output validation: ${resultValue.valid}`;
    }

    if (!resultValue.ok) {
      console.error(`\nBACKBONE SMOKE FAILED: ${resultValue.error}`);
      process.exit(1);
    }

    console.log(`\nBACKBONE SMOKE PASSED`);
    console.log(`  Time: ${resultValue.time} | Grid: ${resultValue.grid} | Valid: ${resultValue.valid}`);

    // Screenshot
    await page.screenshot({ path: '/tmp/sharp-backbone-smoke.png' });
    console.log(`  Screenshot: /tmp/sharp-backbone-smoke.png`);

  } catch (err) {
    console.error(`\nSmoke test error: ${err.message}`);
    // Dump collected console output
    console.log('\n--- Page console output ---');
    for (const line of consoleOutput) {
      console.log(`  ${line}`);
    }
    await page.screenshot({ path: '/tmp/sharp-backbone-smoke-error.png' });
    console.log(`  Error screenshot: /tmp/sharp-backbone-smoke-error.png`);
    process.exit(1);
  } finally {
    await browser.close();
  }
}

main();
