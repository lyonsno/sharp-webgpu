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
  classifyWitnessSourceIdentity,
  retainNegotiatedWitnessIdentity,
  retainWitnessNavigationEvidence,
  resolveWitnessKitVersion,
  runNegotiatedWitnessConformance,
  runWitnessAssertions,
  validateNativeWitnessAdapter,
  validateWitnessKitIdentity,
  validateWitnessNavigation,
} from '../tools/decoder_kernel_witness_contract.mjs';

const witnessUrl = new URL('../tools/test_decoder_kernel_tiling_browser.mjs', import.meta.url);
const witnessPath = fileURLToPath(witnessUrl);
const witnessSource = readFileSync(witnessUrl, 'utf8');
const witnessContractSource = readFileSync(
  new URL('../tools/decoder_kernel_witness_contract.mjs', import.meta.url),
  'utf8',
);
const viteSource = readFileSync(new URL('../vite.config.js', import.meta.url), 'utf8');
const sourceRevision = '1896be13ca401bda8d03791779c7b4158649e917';
const sourceRoot = '/private/tmp/sharp-source-root';
const expectedKitVersion = '0.1.45-sharp-gpu-timestamp-assay.0';

assert.equal(
  await resolveWitnessKitVersion(async () => ({
    WEBGPU_INFERENCE_KIT_VERSION: expectedKitVersion,
  })),
  expectedKitVersion,
);
await assert.rejects(
  resolveWitnessKitVersion(async () => ({})),
  /did not export a version/,
  'a malformed kit must fail inside guarded witness execution',
);

assert.deepEqual(
  validateWitnessKitIdentity({
    expectedKitVersion,
    effectiveKitVersion: expectedKitVersion,
  }),
  {
    status: 'admitted',
    expectedKitVersion,
    effectiveKitVersion: expectedKitVersion,
  },
);
assert.throws(
  () => validateWitnessKitIdentity({
    expectedKitVersion,
    effectiveKitVersion: '0.1.44',
  }),
  /kit version mismatch/,
  'a stale installed kit must not satisfy the Chrome timestamp witness',
);

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
    expectedSourceRevision: sourceRevision,
    effectiveSourceRevision: sourceRevision,
    effectiveSourceState: 'clean',
    expectedSourceRoot: sourceRoot,
    effectiveSourceRoot: sourceRoot,
    expectedEntryPoint: 'sharp-webgpu-root-v1',
    effectiveEntryPoint: 'sharp-webgpu-root-v1',
  }),
  {
    status: 'admitted',
    requestedRoute: 'http://127.0.0.1:5188/',
    effectiveRoute: 'http://127.0.0.1:5188/',
    httpStatus: 200,
    sourceRevision,
    sourceState: 'clean',
    sourceRoot,
    entryPoint: 'sharp-webgpu-root-v1',
  },
);
assert.throws(
  () => validateWitnessNavigation({
    requestedRoute: 'http://127.0.0.1:5188/',
    effectiveRoute: 'http://127.0.0.1:5188/',
    status: 404,
    ok: false,
    expectedSourceRevision: sourceRevision,
    effectiveSourceRevision: sourceRevision,
    effectiveSourceState: 'clean',
    expectedSourceRoot: sourceRoot,
    effectiveSourceRoot: sourceRoot,
    expectedEntryPoint: 'sharp-webgpu-root-v1',
    effectiveEntryPoint: 'sharp-webgpu-root-v1',
  }),
  /unsuccessful HTTP status 404/,
);
assert.throws(
  () => validateWitnessNavigation({
    requestedRoute: 'http://127.0.0.1:5188/',
    effectiveRoute: 'http://127.0.0.1:5188/fallback',
    status: 200,
    ok: true,
    expectedSourceRevision: sourceRevision,
    effectiveSourceRevision: sourceRevision,
    effectiveSourceState: 'clean',
    expectedSourceRoot: sourceRoot,
    effectiveSourceRoot: sourceRoot,
    expectedEntryPoint: 'sharp-webgpu-root-v1',
    effectiveEntryPoint: 'sharp-webgpu-root-v1',
  }),
  /effective route mismatch/,
);
assert.throws(
  () => validateWitnessNavigation({
    requestedRoute: 'http://127.0.0.1:5188/',
    effectiveRoute: 'http://127.0.0.1:5188/',
    status: 200,
    ok: true,
    expectedSourceRevision: sourceRevision,
    effectiveSourceRevision: 'f58eeceb60830ccc072e6f72da13d6b5ded6b534',
    effectiveSourceState: 'clean',
    expectedSourceRoot: sourceRoot,
    effectiveSourceRoot: sourceRoot,
    expectedEntryPoint: 'sharp-webgpu-root-v1',
    effectiveEntryPoint: 'sharp-webgpu-root-v1',
  }),
  /source revision mismatch/,
);
assert.throws(
  () => validateWitnessNavigation({
    requestedRoute: 'http://127.0.0.1:5188/',
    effectiveRoute: 'http://127.0.0.1:5188/',
    status: 200,
    ok: true,
    expectedSourceRevision: sourceRevision,
    effectiveSourceRevision: sourceRevision,
    effectiveSourceState: 'dirty',
    expectedSourceRoot: sourceRoot,
    effectiveSourceRoot: sourceRoot,
    expectedEntryPoint: 'sharp-webgpu-root-v1',
    effectiveEntryPoint: 'sharp-webgpu-root-v1',
  }),
  /source state dirty/,
);
assert.throws(
  () => validateWitnessNavigation({
    requestedRoute: 'http://127.0.0.1:5188/',
    effectiveRoute: 'http://127.0.0.1:5188/',
    status: 200,
    ok: true,
    expectedSourceRevision: sourceRevision,
    effectiveSourceRevision: sourceRevision,
    effectiveSourceState: 'clean',
    expectedSourceRoot: sourceRoot,
    effectiveSourceRoot: sourceRoot,
    expectedEntryPoint: 'sharp-webgpu-root-v1',
    effectiveEntryPoint: 'vite-spa-fallback',
  }),
  /entry point mismatch/,
);
assert.throws(
  () => validateWitnessNavigation({
    requestedRoute: 'http://127.0.0.1:5188/',
    effectiveRoute: 'http://127.0.0.1:5188/',
    status: 200,
    ok: true,
    expectedSourceRevision: sourceRevision,
    effectiveSourceRevision: sourceRevision,
    effectiveSourceState: 'clean',
    expectedSourceRoot: sourceRoot,
    effectiveSourceRoot: '/private/tmp/other-sharp-worktree',
    expectedEntryPoint: 'sharp-webgpu-root-v1',
    effectiveEntryPoint: 'sharp-webgpu-root-v1',
  }),
  /source root mismatch/,
);

assert.deepEqual(
  classifyWitnessSourceIdentity({
    expectedRoot: sourceRoot,
    rootResult: { ok: true, output: sourceRoot },
    revisionResult: { ok: true, output: sourceRevision },
    statusResult: { ok: true, output: '' },
  }),
  {
    sourceRevision,
    sourceRoot,
    sourceState: 'clean',
  },
);
assert.equal(
  classifyWitnessSourceIdentity({
    expectedRoot: sourceRoot,
    rootResult: { ok: true, output: sourceRoot },
    revisionResult: { ok: true, output: sourceRevision },
    statusResult: { ok: false, output: '' },
  }).sourceState,
  'unverifiable',
);
assert.equal(
  classifyWitnessSourceIdentity({
    expectedRoot: sourceRoot,
    rootResult: { ok: true, output: '/private/tmp/other-sharp-worktree' },
    revisionResult: { ok: true, output: sourceRevision },
    statusResult: { ok: true, output: '' },
  }).sourceState,
  'root-mismatch',
);
const rejectedNavigationEvidence = retainWitnessNavigationEvidence(
  { routeLoaded: false },
  {
    requestedRoute: 'http://127.0.0.1:5188/',
    effectiveRoute: 'http://127.0.0.1:5188/',
    status: 200,
    ok: true,
    expectedSourceRevision: sourceRevision,
    effectiveSourceRevision: 'stale-source-revision',
    effectiveSourceState: 'clean',
    expectedSourceRoot: sourceRoot,
    effectiveSourceRoot: sourceRoot,
    expectedEntryPoint: 'sharp-webgpu-root-v1',
    effectiveEntryPoint: 'sharp-webgpu-root-v1',
  },
);
assert.throws(
  () => validateWitnessNavigation(rejectedNavigationEvidence.navigationResponse),
  /source revision mismatch/,
);
assert.equal(rejectedNavigationEvidence.routeLoaded, false);
assert.equal(
  rejectedNavigationEvidence.navigationResponse.effectiveSourceRevision,
  'stale-source-revision',
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
      effectiveKitVersion: expectedKitVersion,
    },
  ),
  {
    conformanceCompleted: false,
    adapterInfo: { isFallbackAdapter: false },
    adapterAdmission: { status: 'non-fallback-admitted' },
    effectiveFeatures: ['timestamp-query'],
    effectiveKitVersion: expectedKitVersion,
  },
);

let retainedBeforeConformanceFailure = null;
await assert.rejects(
  runNegotiatedWitnessConformance({
    negotiate: async () => ({
      adapterInfo: { isFallbackAdapter: false },
      adapterAdmission: { status: 'non-fallback-admitted' },
      effectiveFeatures: ['timestamp-query'],
      effectiveKitVersion: expectedKitVersion,
    }),
    retainNegotiated: identity => {
      retainedBeforeConformanceFailure = identity;
    },
    conform: async () => {
      throw new Error('forced page workload failure');
    },
  }),
  /forced page workload failure/,
);
assert.deepEqual(retainedBeforeConformanceFailure?.effectiveFeatures, ['timestamp-query']);

let assertionFailurePhase = null;
assert.throws(
  () => runWitnessAssertions({
    setFailurePhase: phase => {
      assertionFailurePhase = phase;
    },
    assertConformance: () => {
      throw new Error('forced host assertion failure');
    },
  }),
  /forced host assertion failure/,
);
assert.equal(assertionFailurePhase, 'conformance-assertion');

assert.match(viteSource, /x-sharp-source-revision/);
assert.match(viteSource, /x-sharp-source-state/);
assert.match(viteSource, /x-sharp-source-root/);
assert.match(viteSource, /x-sharp-entrypoint/);
assert.ok(
  witnessSource.indexOf("status: 'running'")
    < witnessSource.indexOf('const expectedSourceIdentity = resolveExpectedSourceIdentity();'),
  'expected-source cleanliness must be resolved after the running report is written',
);
assert.doesNotMatch(
  witnessContractSource,
  /^import\s+[^;]*['"]@kaminos\/webgpu-inference-kit['"];?$/m,
  'the local witness contract must not resolve the external kit during ESM bootstrap',
);
assert.ok(
  witnessSource.indexOf("status: 'running'")
    < witnessSource.indexOf("failurePhase = 'kit-resolution';"),
  'the running report must exist before the witness resolves the external kit',
);
assert.match(
  witnessSource,
  /lastTrustworthyEvidence\s*=\s*\{\s*\.\.\.lastTrustworthyEvidence,\s*effectiveHostKitVersion,\s*\};\s*lastTrustworthyEvidence\s*=\s*\{[\s\S]{0,300}hostKitAdmission/,
  'an observed host kit version must be retained before exact-version admission can throw',
);
assert.match(
  witnessSource,
  /failurePhase\s*=\s*['"]source-validation['"];[\s\S]{0,300}const expectedSourceIdentity = resolveExpectedSourceIdentity\(\)/,
  'source identity failures must have their own durable failure phase',
);
assert.match(
  witnessSource,
  /failurePhase\s*=\s*['"]source-revalidation['"][\s\S]*resolveExpectedSourceIdentity\(\)/,
  'the witness must revalidate expected source identity immediately before success',
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
  /expectedKitVersion[\s\S]*effectiveKitVersion[\s\S]*validateWitnessKitIdentity/,
  'the report must bind the expected and browser-effective kit package identity',
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
  assert.equal(failureReport.expectedKitVersion, expectedKitVersion);
  assert.equal(failureReport.effectiveKitVersion, null);
} finally {
  rmSync(failureRoot, { recursive: true, force: true });
}

const kitFailureRoot = mkdtempSync(join(tmpdir(), 'sharp-decoder-timestamp-kit-resolution-'));
const kitFailureReportPath = join(kitFailureRoot, 'report.json');
try {
  const forcedFailure = spawnSync(
    process.execPath,
    [
      witnessPath,
      '--output',
      kitFailureReportPath,
      '--exercise-failure-phase',
      'kit-resolution',
    ],
    {
      cwd: fileURLToPath(new URL('..', import.meta.url)),
      encoding: 'utf8',
    },
  );
  assert.notEqual(forcedFailure.status, 0, 'forced kit resolution failure must exit nonzero');
  const failureReport = JSON.parse(readFileSync(kitFailureReportPath, 'utf8'));
  assert.equal(failureReport.status, 'failed');
  assert.equal(failureReport.failurePhase, 'kit-resolution');
  assert.match(failureReport.error.message, /forced witness kit resolution failure/);
  assert.equal(failureReport.effectiveHostKitVersion, null);
  assert.equal(failureReport.effectiveKitVersion, null);
  assert.equal(failureReport.lastTrustworthyEvidence?.browserLaunched, false);
} finally {
  rmSync(kitFailureRoot, { recursive: true, force: true });
}

const staleKitRoot = mkdtempSync(join(tmpdir(), 'sharp-decoder-timestamp-stale-kit-'));
const staleKitReportPath = join(staleKitRoot, 'report.json');
try {
  const forcedFailure = spawnSync(
    process.execPath,
    [
      witnessPath,
      '--output',
      staleKitReportPath,
      '--exercise-failure-phase',
      'host-kit-version-mismatch',
    ],
    {
      cwd: fileURLToPath(new URL('..', import.meta.url)),
      encoding: 'utf8',
    },
  );
  assert.notEqual(forcedFailure.status, 0, 'stale host kit identity must exit nonzero');
  const failureReport = JSON.parse(readFileSync(staleKitReportPath, 'utf8'));
  assert.equal(failureReport.status, 'failed');
  assert.equal(failureReport.failurePhase, 'kit-resolution');
  assert.match(failureReport.error.message, /kit version mismatch/);
  assert.equal(failureReport.effectiveHostKitVersion, '0.1.44-forced-stale-kit');
  assert.equal(
    failureReport.lastTrustworthyEvidence?.effectiveHostKitVersion,
    '0.1.44-forced-stale-kit',
  );
  assert.equal(failureReport.hostKitAdmission, null);
  assert.equal(failureReport.lastTrustworthyEvidence?.browserLaunched, false);
} finally {
  rmSync(staleKitRoot, { recursive: true, force: true });
}

const sourceFailureRoot = mkdtempSync(join(tmpdir(), 'sharp-decoder-timestamp-source-validation-'));
const sourceFailureReportPath = join(sourceFailureRoot, 'report.json');
try {
  const forcedFailure = spawnSync(
    process.execPath,
    [
      witnessPath,
      '--output',
      sourceFailureReportPath,
      '--exercise-failure-phase',
      'source-validation',
    ],
    {
      cwd: fileURLToPath(new URL('..', import.meta.url)),
      encoding: 'utf8',
    },
  );
  assert.notEqual(forcedFailure.status, 0, 'forced source validation failure must exit nonzero');
  const failureReport = JSON.parse(readFileSync(sourceFailureReportPath, 'utf8'));
  assert.equal(failureReport.status, 'failed');
  assert.equal(failureReport.failurePhase, 'source-validation');
  assert.match(failureReport.error.message, /forced witness source validation failure/);
  assert.equal(failureReport.effectiveHostKitVersion, expectedKitVersion);
  assert.equal(failureReport.hostKitAdmission.status, 'admitted');
  assert.equal(failureReport.lastTrustworthyEvidence?.browserLaunched, false);
} finally {
  rmSync(sourceFailureRoot, { recursive: true, force: true });
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
  assert.equal(failureReport.expectedKitVersion, expectedKitVersion);
  assert.equal(failureReport.effectiveKitVersion, expectedKitVersion);
  assert.equal(failureReport.lastTrustworthyEvidence.adapterInfo.isFallbackAdapter, false);
  assert.equal(failureReport.lastTrustworthyEvidence.conformanceCompleted, false);
} finally {
  rmSync(assertionFailureRoot, { recursive: true, force: true });
}

console.log('SHARP decoder timestamp browser witness contracts passed');
