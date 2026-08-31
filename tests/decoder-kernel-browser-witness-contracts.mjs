import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import {
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const witnessUrl = new URL('../tools/test_decoder_kernel_tiling_browser.mjs', import.meta.url);
const witnessPath = fileURLToPath(witnessUrl);
const witnessSource = readFileSync(witnessUrl, 'utf8');

assert.match(
  witnessSource,
  /sharp-decoder-kernel-timestamp-conformance-report\.json/,
  'the timestamp-query witness must have a deterministic durable report path',
);
assert.match(
  witnessSource,
  /requestedRoute[\s\S]*effectiveRoute/,
  'the witness report must distinguish requested and effective route identity',
);
assert.match(
  witnessSource,
  /requestedBrowser[\s\S]*effectiveBrowser/,
  'the witness report must distinguish requested and effective browser identity',
);
assert.match(
  witnessSource,
  /status:\s*['"]failed['"][\s\S]*failurePhase[,:][\s\S]*lastTrustworthyEvidence[,:]/,
  'failure before conformance completion must retain its phase and last trustworthy evidence',
);
assert.match(
  witnessSource,
  /userDataDir[\s\S]*--use-mock-keychain/,
  'Chrome must use an isolated profile and mock keychain for the witness',
);
assert.doesNotMatch(
  witnessSource,
  /--enable-features=Vulkan/,
  'the macOS conformance route must not force the Linux Vulkan backend',
);
assert.match(
  witnessSource,
  /requestedFeatures[\s\S]*effectiveFeatures/,
  'the report must preserve exact timestamp-query feature negotiation identity',
);

const failureRoot = mkdtempSync(join(tmpdir(), 'sharp-decoder-timestamp-preflight-'));
const failureReportPath = join(failureRoot, 'report.json');
try {
  writeFileSync(failureReportPath, `${JSON.stringify({
    schema: 'sharp-webgpu.decoder-kernel-timestamp-conformance.v0',
    status: 'passed',
    runId: 'stale-prior-run',
  })}\n`);
  const forcedFailure = spawnSync(
    process.execPath,
    [
      witnessPath,
      '--output',
      failureReportPath,
      '--exercise-failure-phase',
      'preflight',
    ],
    {
      cwd: fileURLToPath(new URL('..', import.meta.url)),
      encoding: 'utf8',
    },
  );
  assert.notEqual(forcedFailure.status, 0, 'forced preflight failure must exit nonzero');
  const failureReport = JSON.parse(readFileSync(failureReportPath, 'utf8'));
  assert.equal(failureReport.status, 'failed');
  assert.equal(failureReport.failurePhase, 'preflight');
  assert.notEqual(failureReport.runId, 'stale-prior-run');
  assert.equal(failureReport.lastTrustworthyEvidence?.routeLoaded, false);
  assert.equal(failureReport.lastTrustworthyEvidence?.conformanceCompleted, false);
  assert.equal(failureReport.requestedRoute, 'http://127.0.0.1:5188/');
  assert.equal(failureReport.effectiveRoute, null);
  assert.equal(failureReport.requestedBrowser.product, 'Google Chrome');
  assert.equal(failureReport.effectiveBrowser, null);
  assert.deepEqual(failureReport.requestedFeatures, ['timestamp-query']);
  assert.equal(failureReport.effectiveFeatures, null);
} finally {
  rmSync(failureRoot, { recursive: true, force: true });
}

console.log('SHARP decoder timestamp browser witness contracts passed');
