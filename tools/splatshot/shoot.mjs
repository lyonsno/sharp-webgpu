import puppeteer from 'puppeteer-core';
import http from 'http';
import fs from 'fs';
import path from 'path';

// Serve from the repo root (cwd) so node_modules and asset paths resolve;
// the viewer page itself lives next to this script.
const ROOT = process.cwd();
const VIEWER_DIR = path.dirname(new URL(import.meta.url).pathname);
const PORT = 8791;
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.ply': 'application/octet-stream' };

const server = http.createServer((req, res) => {
  const urlPath = decodeURIComponent(req.url.split('?')[0]);
  const file = urlPath === '/' || urlPath === '/viewer.html'
    ? path.join(VIEWER_DIR, 'viewer.html')
    : path.join(ROOT, urlPath);
  if (!(file.startsWith(ROOT) || file.startsWith(VIEWER_DIR)) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    res.writeHead(404); res.end('nope'); return;
  }
  res.writeHead(200, { 'content-type': MIME[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
server.listen(PORT);

const shots = JSON.parse(process.argv[2] || '[]');

const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  args: ['--no-sandbox', '--enable-unsafe-webgpu', '--use-angle=metal', '--window-size=1400,1000'],
});
const page = await browser.newPage();
// Square viewport: the viewer's camera aspect is not reconciled with the
// canvas, so a non-square viewport stretches the render horizontally.
await page.setViewport({ width: 1024, height: 1024, deviceScaleFactor: 2 });
page.on('pageerror', e => console.error('[pageerror]', e.message));

for (const s of shots) {
  const q = new URLSearchParams(s.params).toString();
  await page.goto(`http://localhost:${PORT}/viewer.html?${q}`, { waitUntil: 'load' });
  await page.waitForFunction('window.__ready === true || window.__error', { timeout: 120000 });
  const err = await page.evaluate('window.__error');
  if (err) { console.error('shot failed:', s.out, err); continue; }
  await page.screenshot({ path: s.out });
  console.log('wrote', s.out);
}
await browser.close();
server.close();
