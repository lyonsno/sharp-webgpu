#!/usr/bin/env node
/**
 * Quick demo smoke: load page, click sample image, verify backbone runs.
 */
import puppeteer from 'puppeteer-core';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

async function main() {
  const port = process.argv[2] || '5176';
  const url = `http://localhost:${port}/`;

  console.log(`Demo smoke — ${url}\n`);

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
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
  const pageErrors = [];
  page.on('console', msg => console.log(`  [page] ${msg.text()}`));
  page.on('pageerror', err => {
    console.error('  PAGE ERROR:', err.message);
    pageErrors.push(err.message);
  });

  try {
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });
    console.log('Page loaded. Clicking sample image...');

    // Click the sample thumbnail
    await page.click('.sample-thumb');

    // Wait for results table and validation
    const result = await page.waitForFunction(() => {
      const time = document.getElementById('r-time');
      const valid = document.getElementById('r-valid');
      const error = document.getElementById('error');
      if (error && error.style.display !== 'none' && error.textContent) {
        return JSON.stringify({ ok: false, error: error.textContent });
      }
      if (time && time.textContent && time.textContent !== '-') {
        return JSON.stringify({
          ok: true,
          model: document.getElementById('r-model')?.textContent,
          weights: document.getElementById('r-weights')?.textContent,
          grid: document.getElementById('r-grid')?.textContent,
          features: document.getElementById('r-features')?.textContent,
          time: time.textContent,
          valid: valid?.textContent || 'unknown',
        });
      }
      return false;
    }, { timeout: 300000 });

    const r = JSON.parse(await result.jsonValue());

    // Fail on invalid output
    if (r.ok && r.valid && r.valid !== 'OK') {
      r.ok = false;
      r.error = `Output validation: ${r.valid}`;
    }

    if (!r.ok) {
      console.error(`\nDEMO SMOKE FAILED: ${r.error}`);
      process.exit(1);
    }

    // Recheck for late-arriving GPU errors
    await new Promise(resolve => setTimeout(resolve, 100));
    const lateError = await page.evaluate(() => {
      const el = document.getElementById('error');
      return (el && el.style.display !== 'none' && el.textContent) ? el.textContent : null;
    });
    if (lateError) {
      console.error(`\nDEMO SMOKE FAILED (late GPU error): ${lateError}`);
      process.exit(1);
    }
    if (pageErrors.length > 0) {
      console.error(`\nDEMO SMOKE FAILED: ${pageErrors.length} page error(s):`);
      for (const e of pageErrors) console.error(`  ${e}`);
      process.exit(1);
    }

    console.log(`\nDEMO SMOKE PASSED`);
    console.log(`  Model:    ${r.model}`);
    console.log(`  Weights:  ${r.weights}`);
    console.log(`  Grid:     ${r.grid}`);
    console.log(`  Features: ${r.features}`);
    console.log(`  Time:     ${r.time}`);
    console.log(`  Valid:    ${r.valid}`);

    await page.screenshot({ path: '/tmp/sharp-demo-smoke.png' });
    console.log(`  Screenshot: /tmp/sharp-demo-smoke.png`);

  } catch (err) {
    console.error(`Error: ${err.message}`);
    await page.screenshot({ path: '/tmp/sharp-demo-smoke-error.png' }).catch(() => {});
    process.exit(1);
  } finally {
    await browser.close();
  }
}

main();
