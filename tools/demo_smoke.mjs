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
  page.on('console', msg => console.log(`  [page] ${msg.text()}`));
  page.on('pageerror', err => console.error('  PAGE ERROR:', err.message));

  try {
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });
    console.log('Page loaded. Clicking sample image...');

    // Click the sample thumbnail
    await page.click('.sample-thumb');

    // Wait for results table to appear
    const result = await page.waitForFunction(() => {
      const time = document.getElementById('r-time');
      const error = document.getElementById('error');
      if (error && error.style.display !== 'none' && error.textContent) {
        return { error: error.textContent };
      }
      if (time && time.textContent && time.textContent !== '-') {
        return {
          model: document.getElementById('r-model').textContent,
          weights: document.getElementById('r-weights').textContent,
          grid: document.getElementById('r-grid').textContent,
          features: document.getElementById('r-features').textContent,
          time: document.getElementById('r-time').textContent,
        };
      }
      return false;
    }, { timeout: 300000 });

    const r = await result.jsonValue();

    if (r.error) {
      console.error(`\nDEMO SMOKE FAILED: ${r.error}`);
      process.exit(1);
    }

    console.log(`\nDEMO SMOKE PASSED`);
    console.log(`  Model:    ${r.model}`);
    console.log(`  Weights:  ${r.weights}`);
    console.log(`  Grid:     ${r.grid}`);
    console.log(`  Features: ${r.features}`);
    console.log(`  Time:     ${r.time}`);

    await page.screenshot({ path: '/tmp/sharp-demo-smoke.png' });
    console.log(`  Screenshot: /tmp/sharp-demo-smoke.png`);

  } catch (err) {
    console.error(`Error: ${err.message}`);
    await page.screenshot({ path: '/tmp/sharp-demo-smoke-error.png' });
    process.exit(1);
  } finally {
    await browser.close();
  }
}

main();
