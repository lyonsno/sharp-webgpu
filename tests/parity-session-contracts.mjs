import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const { createSharpParitySession } = await import('../src/lib/parity_capture.js');
const { createParityReport, finishParityReport, writeParityReport } = await import('../tools/parity_report.mjs');

const session = createSharpParitySession('run-a');
session.capture('encoder', new Float32Array([1, 2, 3, 4]), { shape: [4], layout: 'N' });
const identity = { runId: 'run-a', stageId: 'encoder' };
const reference = Buffer.from(new Float32Array([1, 2, 3, 5]).buffer);
session.beginReference({ ...identity, byteLength: reference.length });
assert.throws(() => session.appendReference({ ...identity, runId: 'stale', byteOffset: 0, payloadBase64: 'AAAA' }), /run/);
assert.throws(() => session.appendReference({ ...identity, byteOffset: 4, payloadBase64: 'AAAA' }), /offset/);
session.appendReference({ ...identity, byteOffset: 0, payloadBase64: reference.subarray(0, 5).toString('base64') });
assert.throws(() => session.compareReference(identity), /incomplete/);
session.appendReference({ ...identity, byteOffset: 5, payloadBase64: reference.subarray(5).toString('base64') });
const result = session.compareReference(identity);
assert.equal(result.metrics.maxAbsoluteError, 1);
assert.equal(result.sourceElementCount, 4);
assert.equal(result.stageId, 'encoder');
assert.equal(result.runId, 'run-a');
assert.throws(() => session.compareReference(identity), /reference/);
assert.throws(() => session.beginReference({ ...identity, byteLength: 4 }), /length/);
assert.throws(() => session.readCaptureRange({ ...identity, runId: 'wrong', byteOffset: 0, byteLength: 4 }), /run/);
const range = session.readCaptureRange({ ...identity, byteOffset: 4, byteLength: 8 });
assert.equal(range.runId, 'run-a');
assert.equal(range.byteOffset, 4);
assert.deepEqual(new Float32Array(Uint8Array.from(Buffer.from(range.payloadBase64, 'base64')).buffer), new Float32Array([2, 3]));
session.clear();
assert.equal(session.describe('encoder'), null);

const planes = new Float32Array(2 * 18 * 18).fill(1);
const expectedPlanes = planes.slice();
planes[0] = 99;
session.capture('deltas', planes, { shape: [2, 18, 18], layout: 'CHW' });
const deltaIdentity = { runId: 'run-a', stageId: 'deltas' };
session.beginReference({ ...deltaIdentity, byteLength: expectedPlanes.byteLength });
session.appendReference({ ...deltaIdentity, byteOffset: 0,
  payloadBase64: Buffer.from(expectedPlanes.buffer).toString('base64') });
const deltas = session.compareReference(deltaIdentity, { interior: true });
assert.equal(deltas.metrics.maxAbsoluteError, 98);
assert.equal(deltas.interior.borderPixels, 8);
assert.equal(deltas.interior.comparison.sourceElementCount, 8);
assert.equal(deltas.interior.comparison.metrics.exactMatch, true);
session.clear();

const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'sharp-parity-report-'));
try {
  const reportPath = path.join(directory, 'report.json');
  const report = createParityReport({ runId: 'run-a', reference: { dtype: 'fp32' } });
  report.stages.encoder = { status: 'compared', comparison: result };
  finishParityReport(report, { means: { metrics: { maxAbsoluteError: 0 } } }, 1);
  assert.equal(report.stages.encoder.comparison.metrics.maxAbsoluteError, 1);
  assert.equal(report.status, 'completed');
  assert.equal(report.summary.comparedStages, 1);
  report.status = 'failed';
  report.failurePhase = 'ply-comparison';
  report.error = 'no PLY';
  writeParityReport(reportPath, report);
  const saved = JSON.parse(fs.readFileSync(reportPath));
  assert.equal(saved.stages.encoder.comparison.metrics.maxAbsoluteError, 1);
  assert.equal(saved.ply.means.metrics.maxAbsoluteError, 0);
  assert.equal(saved.status, 'failed');
  assert.equal(saved.failurePhase, 'ply-comparison');
} finally {
  fs.rmSync(directory, { recursive: true, force: true });
}
console.log('SHARP parity session and report contracts passed');
