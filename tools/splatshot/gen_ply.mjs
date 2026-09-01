// Run the SHARP-WebGPU pipeline headless on a given image and save the PLY.
// Usage: node gen_ply.mjs <imagePath> <outPly> [port]
import puppeteer from 'puppeteer-core';
import fs from 'fs';

const [imagePath, outPly, port = '5175'] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  protocolTimeout: 900000,
  args: ['--enable-unsafe-webgpu', '--no-sandbox', '--disable-gpu-sandbox'],
});
const page = await browser.newPage();
page.on('console', m => { const t = m.text(); if (t.includes('[Main]') || t.includes('[Compose]')) console.log(' ', t); });

await page.goto(`http://localhost:${port}/`, { waitUntil: 'networkidle0', timeout: 60000 });
await page.click('#use-spn');
const fileInput = await page.$('#file-input');
await fileInput.uploadFile(imagePath);

const done = await page.waitForFunction(() => {
  const t = document.getElementById('r-time');
  const e = document.getElementById('error');
  if (e && e.style.display !== 'none' && e.textContent) return 'error:' + e.textContent;
  if (t && t.textContent && t.textContent !== '-') return 'done';
  return false;
}, { timeout: 600000 });
if ((await done.jsonValue()).startsWith('error:')) { console.error(await done.jsonValue()); process.exit(1); }

const size = await page.evaluate(async () => {
  const link = document.getElementById('download-ply');
  const buf = await (await fetch(link.href)).arrayBuffer();
  window.__ply = new Uint8Array(buf);
  return buf.byteLength;
});
console.log('PLY bytes:', size);

const CHUNK = 16 * 1024 * 1024;
const parts = [];
for (let off = 0; off < size; off += CHUNK) {
  const b64 = await page.evaluate((off, len) => {
    const slice = window.__ply.subarray(off, off + len);
    let s = '';
    for (let i = 0; i < slice.length; i += 0x8000) s += String.fromCharCode(...slice.subarray(i, i + 0x8000));
    return btoa(s);
  }, off, Math.min(CHUNK, size - off));
  parts.push(Buffer.from(b64, 'base64'));
}
fs.writeFileSync(outPly, Buffer.concat(parts));
console.log('wrote', outPly);
await browser.close();
