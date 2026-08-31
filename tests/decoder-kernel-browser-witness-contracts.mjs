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

import {
  retainNegotiatedWitnessIdentity,
  validateNativeWitnessAdapter,
  validateWitnessNavigation,
} from '../tools/decoder_kernel_witness_contract.mjs';

const witnessUrl = new URL('../tools/test_decoder_kernel_tiling_browser.mjs', import.meta.url);
const witnessPath = fileURLToPath(witnessUrl);
const witnessSource = readFileSync(witnessUrl, 'utf8');

assert.match(
  witnessSource,
  /sharp-decoder-kernel-timestamp-conformance-report\.json/,
  'the timestamp-query witness must have a deterministic durable report path',
);

assert.deepEqual(
  validateWitnessNavigation({
    requestedRoute: 'http://127.0.0.1:5188/',
    effectiveRoute: 'http://127.0.0.1:5188/',
    status: 200,
    ok: true,
  }),
  {
    status: 'admitted',
    requestedRoute: 'http://127.0.0.1:5188/',
    effectiveRoute: 'http://127.0.0.1:5188/',
    httpStatus: 200,
  },
);
assert.throws(
  () => validateWitnessNavigation({
    requestedRoute: 'http://127.0.0.1:5188/',
    effectiveRoute: 'http://127.0.0.1:5188/',
    status: 404,
    ok: false,
  }),
  /unsuccessful HTTP status 404/,
);
assert.throws(
  () => validateWitnessNavigation({
    requestedRoute: 'http://127.0.0.1:5188/',
    effectiveRoute: 'http://127.0.0.1:5188/fallback',
    status: 200,
    ok: true,
  }),
  /effective route mismatch/,
);
assert.throws(
  () => validateNativeWitnessAdapter({ isFallbackAdapter: true }),
  /refuses a fallback GPU adapter/,
);
assert.equal(
  validateNativeWitnessAdapter({ isFallbackAdapter: false }).status,
  'non-fallback-admitted',
);
assert.deepEqual(
  retainNegotiatedWitnessIdentity(
    { conformanceCompleted: false },
    {
      adapterInfo: { isFallbackAdapter: false },
      adapterAdmission: { status: 'non-fallback-admitted' },
      effectiveFeatures: ['timestamp-query'],
    },
  ),
  {
    conformanceCompleted: false,
    adapterInfo: { isFallbackAdapter: false },
    adapterAdmission: { status: 'non-fallback-admitted' },
    effectiveFeatures: ['timestamp-query'],
  },
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
assert.match(
  witnessSource,
  /navigationResponse[\s\S]*validateWitnessNavigation/,
  'the witness must admit a successful canonical navigation before WebGPU evaluation',
);
assert.match(
  witnessSource,
  /adapter\.info[\s\S]*isFallbackAdapter[\s\S]*validateNativeWitnessAdapter/,
  'the witness must reject a fallback adapter before requesting a device',
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

const assertionFailureRoot = mkdtempSync(join(tmpdir(), 'sharp-decoder-timestamp-assertion-'));
const assertionFailureReportPath = join(assertionFailureRoot, 'report.json');
try {
  const forcedFailure = spawnSync(
    process.execPath,
    [
      witnessPath,
      '--output',
      assertionFailureReportPath,
      '--exercise-failure-phase',
      'post-negotiation-assertion',
    ],
    {
      cwd: fileURLToPath(new URL('..', import.meta.url)),
      encoding: 'utf8',
    },
  );
  assert.notEqual(forcedFailure.status, 0, 'forced post-negotiation failure must exit nonzero');
  const failureReport = JSON.parse(readFileSync(assertionFailureReportPath, 'utf8'));
  assert.equal(failureReport.status, 'failed');
  assert.equal(failureReport.failurePhase, 'conformance-assertion');
  assert.deepEqual(failureReport.effectiveFeatures, ['timestamp-query']);
  assert.equal(failureReport.lastTrustworthyEvidence.adapterInfo.isFallbackAdapter, false);
  assert.equal(failureReport.lastTrustworthyEvidence.conformanceCompleted, false);
} finally {
  rmSync(assertionFailureRoot, { recursive: true, force: true });
}

console.log('SHARP decoder timestamp browser witness contracts passed');
