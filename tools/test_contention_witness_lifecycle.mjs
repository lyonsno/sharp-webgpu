import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { runContentionWitness } from './contention_witness.mjs';

async function assertDurableEarlyFailure(name, launchBrowser, expectedPhase) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'sharp-contention-lifecycle-'));
  const out = path.join(dir, 'report.json');
  const result = await runContentionWitness({
    mode: 'contention',
    port: '5175',
    out,
    screenshot: path.join(dir, 'screen.png'),
    image: null,
    sharpScheduler: null,
    headed: false,
    timeoutMs: 1000,
    episodeId: `sharp-contention:${name}`,
  }, { launchBrowser });

  assert.equal(result.ok, false, `${name} must return a failed witness result`);
  assert.equal(fs.existsSync(out), true, `${name} must write a durable report`);
  const report = JSON.parse(fs.readFileSync(out, 'utf8'));
  assert.equal(report.schema, 'sharp.webgpu-contention-witness-failure.v0');
  assert.equal(report.runId, `sharp-contention:${name}`);
  assert.equal(report.failurePhase, expectedPhase);
  assert.match(report.error, new RegExp(name));
}

await assertDurableEarlyFailure(
  'launch-rejected',
  async () => { throw new Error('launch-rejected'); },
  'browser-launch',
);

await assertDurableEarlyFailure(
  'new-page-rejected',
  async () => ({
    newPage: async () => { throw new Error('new-page-rejected'); },
    close: async () => {},
  }),
  'page-create',
);

console.log('contention witness lifecycle contract passed');
