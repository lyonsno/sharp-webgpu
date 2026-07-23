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

const witnessUrl = new URL('../tools/test_vit_four_stage_parity_browser.mjs', import.meta.url);
const witnessPath = fileURLToPath(witnessUrl);
const witnessSource = readFileSync(witnessUrl, 'utf8');

assert.match(
  witnessSource,
  /sharp-vit-four-stage-parity-report\.json/,
  'the browser witness must have a deterministic durable report path',
);
assert.match(
  witnessSource,
  /requestedRoute:[\s\S]*effectiveRoute:/,
  'the report must distinguish requested and effective browser route identity',
);
assert.match(
  witnessSource,
  /status:\s*['"]failed['"][\s\S]*failurePhase[,:][\s\S]*lastTrustworthyEvidence[,:]/,
  'a failure before parity completion must still report its phase and last trustworthy evidence',
);
assert.match(
  witnessSource,
  /writeFileSync\([\s\S]*JSON\.stringify/,
  'both successful and failed witness execution must write structured evidence',
);
assert.match(
  witnessSource,
  /measurementOrder\s*=\s*\[\s*['"]dispatch-major['"]\s*,\s*['"]four-stage['"]\s*,\s*['"]four-stage['"]\s*,\s*['"]dispatch-major['"]\s*\]/,
  'timing must use paired forward and reverse mode order',
);
assert.match(
  witnessSource,
  /warmup[\s\S]*yieldMs:\s*0[\s\S]*waitForSubmittedWorkDone:\s*false/,
  'timing must follow an unmeasured production-encoder shader warmup',
);
assert.match(
  witnessSource,
  /durationSamplesMs/,
  'the report must retain the underlying per-mode timing samples',
);
assert.match(
  witnessSource,
  /projectionAuthority:\s*['"]synthetic-scheduler-overhead-projection['"]/,
  'the report must not present synthetic scheduler timing as measured full-route wall time',
);

const failureRoot = mkdtempSync(join(tmpdir(), 'sharp-four-stage-preflight-'));
const failureReportPath = join(failureRoot, 'report.json');
try {
  writeFileSync(failureReportPath, `${JSON.stringify({
    schema: 'sharp-webgpu.vit-four-stage-production-parity.v0',
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
      'port-reserve',
    ],
    {
      cwd: fileURLToPath(new URL('..', import.meta.url)),
      encoding: 'utf8',
    },
  );
  assert.notEqual(forcedFailure.status, 0, 'forced preflight failure must exit nonzero');
  const failureReport = JSON.parse(readFileSync(failureReportPath, 'utf8'));
  assert.equal(failureReport.status, 'failed');
  assert.equal(failureReport.failurePhase, 'port-reserve');
  assert.notEqual(failureReport.runId, 'stale-prior-run');
  assert.equal(failureReport.lastTrustworthyEvidence?.parityCompleted, false);
  assert.equal(failureReport.requestedRoute, null);
  assert.equal(failureReport.effectiveRoute, null);
} finally {
  rmSync(failureRoot, { recursive: true, force: true });
}

console.log('SHARP four-stage witness contracts passed');
