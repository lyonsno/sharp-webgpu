import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  createSharpRunTelemetry,
  createSharpRuntimeDutyMap,
  parseSharpSchedulerConfig,
  recordSchedulerEvent,
  schedulerYield,
  schedulerTelemetrySnapshot,
} from '../src/lib/scheduler.js';

const root = new URL('..', import.meta.url).pathname;
const schedulerPath = join(root, 'src', 'lib', 'scheduler.js');
const mainPath = join(root, 'src', 'main.js');
const spnPath = join(root, 'src', 'lib', 'spn.js');
const monodepthPath = join(root, 'src', 'lib', 'monodepth.js');
const backbonePath = join(root, 'src', 'lib', 'backbone.js');
const gaussianPath = join(root, 'src', 'lib', 'gaussian_decoder.js');
const contentionWitnessPath = join(root, 'tools', 'contention_witness.mjs');

assert.ok(existsSync(schedulerPath), 'SHARP-WebGPU must expose a scheduler contract module');

const requested = {
  mode: 'cooperative',
  spnPatchChunkSize: 1,
  yieldMs: 5,
  waitForSubmittedWorkDone: true,
  gaussianPhaseYieldMs: 7,
  vitBlockChunkSize: 2,
};
const scheduler = parseSharpSchedulerConfig({ sharpScheduler: requested });
assert.equal(scheduler.schema, 'sharp-webgpu.scheduler-config.v0');
assert.equal(scheduler.requested.spnPatchChunkSize, 1);
assert.equal(scheduler.effective.spnPatchChunkSize, 1);
assert.equal(scheduler.effective.yieldMs, 5);
assert.equal(scheduler.effective.waitForSubmittedWorkDone, true);
assert.equal(scheduler.effective.gaussianPhaseYieldMs, 7);
assert.equal(scheduler.effective.vitBlockChunkSize, 2, 'requested ViT block chunking must become effective scheduler config');
assert.deepEqual(scheduler.unsupportedFields, []);

const telemetry = createSharpRunTelemetry(scheduler, { runId: 'contract-run' });
recordSchedulerEvent(telemetry, 'spn-patch-chunk', {
  chunkStart: 0,
  chunkEnd: 1,
  waitedForSubmittedWorkDone: true,
  yieldMs: 5,
});
const snapshot = schedulerTelemetrySnapshot(telemetry);
assert.equal(snapshot.schema, 'sharp-webgpu.scheduler-telemetry.v0');
assert.equal(snapshot.runId, 'contract-run');
assert.equal(snapshot.requestedScheduler.spnPatchChunkSize, 1);
assert.equal(snapshot.effectiveScheduler.spnPatchChunkSize, 1);
assert.deepEqual(snapshot.unsupportedFields, []);
assert.equal(snapshot.events[0].phase, 'spn-patch-chunk');
assert.equal(snapshot.status, 'scheduler-unverified', 'requested ViT block chunking must not verify without observed ViT block boundaries');
const missingVitAssertion = snapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseChunkSize.vitBlock');
assert.ok(missingVitAssertion, 'requested ViT block chunking must produce a boundary assertion');
assert.equal(missingVitAssertion.status, 'unverified');
assert.equal(missingVitAssertion.observedBoundary, 'vit-block-chunk');
assert.equal(missingVitAssertion.observedYieldCount, 0);

const uncappedTimingScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    spnPatchChunkSize: 1,
    yieldMs: 5000,
    gaussianPhaseYieldMs: 5000,
  },
});
assert.equal(uncappedTimingScheduler.requested.yieldMs, 5000, 'requested yieldMs must preserve caller intent');
assert.equal(uncappedTimingScheduler.effective.yieldMs, 5000, 'effective yieldMs must not silently cap below caller intent');
assert.equal(uncappedTimingScheduler.requested.gaussianPhaseYieldMs, 5000, 'requested Gaussian yield must preserve caller intent');
assert.equal(uncappedTimingScheduler.effective.gaussianPhaseYieldMs, 5000, 'effective Gaussian yield must not silently cap below caller intent');
assert.deepEqual(uncappedTimingScheduler.unsupportedFields, [], 'supported timing fields must not be laundered through hidden caps');

const defaultScheduler = parseSharpSchedulerConfig();
const defaultTelemetry = createSharpRunTelemetry(defaultScheduler, { runId: 'default-yield-run' });
let defaultTimerFired = false;
setTimeout(() => { defaultTimerFired = true; }, 0);
await schedulerYield(
  defaultScheduler,
  {},
  defaultTelemetry,
  'spn-patch-chunk',
  { chunkStart: 0, chunkEnd: defaultScheduler.effective.spnPatchChunkSize, totalPatches: 35 }
);
const defaultYieldSnapshot = schedulerTelemetrySnapshot(defaultTelemetry);
assert.equal(defaultTimerFired, true, 'default SPN chunk boundary must preserve an actual task yield');
assert.equal(defaultYieldSnapshot.boundaryAssertions[0].observedYieldCount, 1, 'default SPN chunk proof must record the preserved task yield');

const backgroundScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'background',
  },
});
assert.equal(backgroundScheduler.effective.mode, 'background', 'background mode must stay visible in effective scheduler identity');
assert.equal(backgroundScheduler.effective.spnPatchChunkSize, 1, 'background mode must default to one SPN patch per chunk');
assert.equal(backgroundScheduler.effective.vitBlockChunkSize, 1, 'background mode must default to one ViT block per chunk');
assert.equal(backgroundScheduler.effective.waitForSubmittedWorkDone, true, 'background mode must wait for submitted work before yielding');
assert.ok(backgroundScheduler.effective.yieldMs >= 8, 'background mode must donate real event-loop time, not setTimeout(0)');
assert.ok(backgroundScheduler.effective.gaussianPhaseYieldMs >= 8, 'background mode must donate real Gaussian phase yield time');
assert.ok(backgroundScheduler.effective.routeTailYieldMs >= 8, 'background mode must donate real route-tail yield time');
assert.ok(backgroundScheduler.effective.cpuChunkItems > 0, 'background mode must define CPU materialization chunk size');

const explicitBackgroundScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'background',
    yieldMs: 3,
    waitForSubmittedWorkDone: false,
    routeTailYieldMs: 5,
    cpuChunkItems: 1234,
  },
});
assert.equal(explicitBackgroundScheduler.effective.yieldMs, 3, 'explicit background yieldMs must override the mode preset');
assert.equal(explicitBackgroundScheduler.effective.waitForSubmittedWorkDone, false, 'explicit background queue-wait choice must override the mode preset');
assert.equal(explicitBackgroundScheduler.effective.routeTailYieldMs, 5, 'explicit routeTailYieldMs must override the mode preset');
assert.equal(explicitBackgroundScheduler.effective.cpuChunkItems, 1234, 'explicit cpuChunkItems must override the mode preset');

const dutyMap = createSharpRuntimeDutyMap();
assert.equal(dutyMap.schema, 'sharp-webgpu.background-duty-map.v0');
assert.ok(dutyMap.generatedAt, 'duty map must preserve generation time');
for (const requiredStep of [
  'spn.patch-final-token-readback',
  'spn.patch-intermediate-feature-readback',
  'spn.lowres-fusion-readback',
  'gaussian.initializer-disparity-readback',
  'output-capture.disparity-readback',
  'compose-ply.geometry-delta-readback',
  'compose-ply.texture-delta-readback',
  'compose-ply.compose-export',
]) {
  assert.ok(
    dutyMap.steps.some(step => step.id === requiredStep),
    `background duty map must classify ${requiredStep}`
  );
}
const midstreamSteps = dutyMap.steps.filter(step => step.syncClass === 'midstream-sync');
const finalMaterializationSteps = dutyMap.steps.filter(step => step.syncClass === 'final-materialization');
assert.ok(midstreamSteps.length >= 4, 'duty map must identify multiple midstream sync walls');
assert.ok(finalMaterializationSteps.length >= 3, 'duty map must identify final materialization work separately');
assert.ok(
  dutyMap.steps.some(step => step.id === 'gaussian.initializer-disparity-readback' && step.nextAction === 'move-to-gpu'),
  'Gaussian initializer disparity readback must be classified as a GPU-residency target'
);
assert.ok(
  dutyMap.steps.some(step => step.id === 'compose-ply.compose-export' && step.productHandling === 'materialize-visibly'),
  'PLY export must be classed as visible materialization rather than hidden route wait'
);

const partialDefaultTelemetry = createSharpRunTelemetry(defaultScheduler, { runId: 'partial-default-run' });
recordSchedulerEvent(partialDefaultTelemetry, 'spn-patch-chunk', {
  boundary: 'spn-patch-chunk',
  kind: 'chunk-start',
});
const partialDefaultSnapshot = schedulerTelemetrySnapshot(partialDefaultTelemetry);
assert.equal(partialDefaultSnapshot.status, 'scheduler-unverified', 'default SPN scheduler proof must reject chunk telemetry without observed JS yield');
assert.equal(partialDefaultSnapshot.boundaryAssertions[0].status, 'unverified');
assert.equal(partialDefaultSnapshot.boundaryAssertions[0].observedYieldCount, 0);

const proofScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    spnPatchChunkSize: 1,
    yieldMs: 5,
    waitForSubmittedWorkDone: true,
  },
});
const proofTelemetry = createSharpRunTelemetry(proofScheduler, { runId: 'proof-run' });
let queueWaitCount = 0;
await schedulerYield(
  proofScheduler,
  { queue: { onSubmittedWorkDone: async () => { queueWaitCount += 1; } } },
  proofTelemetry,
  'spn-patch-chunk',
  { chunkStart: 0, chunkEnd: 1, totalPatches: 35 }
);
const proofSnapshot = schedulerTelemetrySnapshot(proofTelemetry);
assert.equal(queueWaitCount, 1, 'schedulerYield must call queue.onSubmittedWorkDone when requested');
assert.equal(proofSnapshot.status, 'verified', 'observed queue/yield events plus matching boundary assertion may verify');
assert.equal(proofSnapshot.eventTrace.schema, 'kaminos.webgpu-scheduler-event-trace.v0');
assert.equal(proofSnapshot.eventTrace.timingAuthority, 'browser-wall-clock');
assert.ok(proofSnapshot.eventTrace.events.some(event => event.kind === 'chunk-start' && event.boundary === 'spn-patch-chunk'));
assert.ok(proofSnapshot.eventTrace.events.some(event => event.kind === 'queue-work-done-start' && event.boundary === 'spn-patch-chunk'));
assert.ok(proofSnapshot.eventTrace.events.some(event => event.kind === 'queue-work-done-end' && event.boundary === 'spn-patch-chunk'));
assert.ok(proofSnapshot.eventTrace.events.some(event => event.kind === 'js-yield-start' && event.boundary === 'spn-patch-chunk'));
assert.ok(proofSnapshot.eventTrace.events.some(event => event.kind === 'js-yield-end' && event.boundary === 'spn-patch-chunk'));
assert.deepEqual(proofSnapshot.events, proofSnapshot.eventTrace.events, 'legacy events alias must match eventTrace.events');
assert.equal(proofSnapshot.boundaryAssertions[0].field, 'phaseChunkSize.spnPatch');
assert.equal(proofSnapshot.boundaryAssertions[0].status, 'verified');
assert.equal(proofSnapshot.boundaryAssertions[0].observedBoundary, 'spn-patch-chunk');
assert.equal(proofSnapshot.boundaryAssertions[0].observedCount, 1);
assert.equal(proofSnapshot.boundaryAssertions[0].observedQueueWaitCount, 1);
assert.equal(proofSnapshot.boundaryAssertions[0].observedYieldCount, 1);

const missingQueueTelemetry = createSharpRunTelemetry(proofScheduler, { runId: 'missing-queue-run' });
recordSchedulerEvent(missingQueueTelemetry, 'spn-patch-chunk', {
  boundary: 'spn-patch-chunk',
  kind: 'chunk-start',
});
recordSchedulerEvent(missingQueueTelemetry, 'spn-patch-chunk', {
  boundary: 'spn-patch-chunk',
  kind: 'js-yield-start',
  yieldMs: 5,
});
recordSchedulerEvent(missingQueueTelemetry, 'spn-patch-chunk', {
  boundary: 'spn-patch-chunk',
  kind: 'js-yield-end',
  yieldMs: 5,
});
const missingQueueSnapshot = schedulerTelemetrySnapshot(missingQueueTelemetry);
assert.equal(missingQueueSnapshot.status, 'scheduler-unverified', 'requested queue wait must keep telemetry unverified when queue events are absent');
assert.equal(missingQueueSnapshot.boundaryAssertions[0].status, 'unverified');
assert.equal(missingQueueSnapshot.boundaryAssertions[0].observedQueueWaitCount, 0);

const missingYieldTelemetry = createSharpRunTelemetry(proofScheduler, { runId: 'missing-yield-run' });
recordSchedulerEvent(missingYieldTelemetry, 'spn-patch-chunk', {
  boundary: 'spn-patch-chunk',
  kind: 'chunk-start',
});
recordSchedulerEvent(missingYieldTelemetry, 'spn-patch-chunk', {
  boundary: 'spn-patch-chunk',
  kind: 'queue-work-done-start',
});
recordSchedulerEvent(missingYieldTelemetry, 'spn-patch-chunk', {
  boundary: 'spn-patch-chunk',
  kind: 'queue-work-done-end',
});
const missingYieldSnapshot = schedulerTelemetrySnapshot(missingYieldTelemetry);
assert.equal(missingYieldSnapshot.status, 'scheduler-unverified', 'requested JS yield must keep telemetry unverified when yield events are absent');
assert.equal(missingYieldSnapshot.boundaryAssertions[0].status, 'unverified');
assert.equal(missingYieldSnapshot.boundaryAssertions[0].observedYieldCount, 0);

const gaussianProofScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    spnPatchChunkSize: 1,
    gaussianPhaseYieldMs: 7,
  },
});
const missingGaussianTelemetry = createSharpRunTelemetry(gaussianProofScheduler, { runId: 'missing-gaussian-run' });
recordSchedulerEvent(missingGaussianTelemetry, 'spn-patch-chunk', {
  boundary: 'spn-patch-chunk',
  kind: 'chunk-start',
});
const missingGaussianSnapshot = schedulerTelemetrySnapshot(missingGaussianTelemetry);
const missingGaussianAssertion = missingGaussianSnapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseYieldMs.gaussianPhase');
assert.equal(missingGaussianSnapshot.status, 'scheduler-unverified', 'requested Gaussian phase yield must keep telemetry unverified when Gaussian events are absent');
assert.ok(missingGaussianAssertion, 'requested Gaussian phase yield must produce a boundary assertion');
assert.equal(missingGaussianAssertion.status, 'unverified');
assert.equal(missingGaussianAssertion.observedBoundary, 'gaussian-phase');
assert.equal(missingGaussianAssertion.observedYieldCount, 0);

const gaussianProofTelemetry = createSharpRunTelemetry(gaussianProofScheduler, { runId: 'gaussian-proof-run' });
await schedulerYield(
  gaussianProofScheduler,
  {},
  gaussianProofTelemetry,
  'spn-patch-chunk',
  { chunkStart: 0, chunkEnd: gaussianProofScheduler.effective.spnPatchChunkSize, totalPatches: 35 }
);
await schedulerYield(
  gaussianProofScheduler,
  {},
  gaussianProofTelemetry,
  'gaussian-phase',
  { phase: 'prediction-head' },
  gaussianProofScheduler.effective.gaussianPhaseYieldMs
);
const gaussianProofSnapshot = schedulerTelemetrySnapshot(gaussianProofTelemetry);
const gaussianProofAssertion = gaussianProofSnapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseYieldMs.gaussianPhase');
assert.equal(gaussianProofSnapshot.status, 'verified', 'observed SPN chunk and Gaussian phase yield events may verify');
assert.equal(gaussianProofAssertion.status, 'verified');
assert.equal(gaussianProofAssertion.observedYieldCount, 1);

const vitProofScheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    spnPatchChunkSize: 1,
    vitBlockChunkSize: 2,
    yieldMs: 0,
  },
});
assert.equal(vitProofScheduler.effective.vitBlockChunkSize, 2, 'ViT block chunking must be effective independently of nonzero yieldMs');
assert.deepEqual(vitProofScheduler.unsupportedFields, [], 'ViT block chunking must not be reported as unsupported');
const vitProofTelemetry = createSharpRunTelemetry(vitProofScheduler, { runId: 'vit-proof-run' });
await schedulerYield(
  vitProofScheduler,
  {},
  vitProofTelemetry,
  'spn-patch-chunk',
  { chunkStart: 0, chunkEnd: 1, totalPatches: 35 }
);
await schedulerYield(
  vitProofScheduler,
  {},
  vitProofTelemetry,
  'vit-block-chunk',
  { encoder: 'patch', blockStart: 0, blockEnd: 2, totalBlocks: 24, tokenCount: 577 }
);
const vitProofSnapshot = schedulerTelemetrySnapshot(vitProofTelemetry);
const vitProofAssertion = vitProofSnapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseChunkSize.vitBlock');
assert.equal(vitProofSnapshot.status, 'verified', 'observed SPN and ViT block chunk yields may verify requested scheduler boundaries');
assert.equal(vitProofAssertion.status, 'verified');
assert.equal(vitProofAssertion.effective, 2);
assert.equal(vitProofAssertion.observedBoundary, 'vit-block-chunk');
assert.equal(vitProofAssertion.observedCount, 1);
assert.equal(vitProofAssertion.observedYieldCount, 1);

const missingVitTelemetry = createSharpRunTelemetry(vitProofScheduler, { runId: 'missing-vit-run' });
await schedulerYield(
  vitProofScheduler,
  {},
  missingVitTelemetry,
  'spn-patch-chunk',
  { chunkStart: 0, chunkEnd: 1, totalPatches: 35 }
);
const missingVitSnapshot = schedulerTelemetrySnapshot(missingVitTelemetry);
const missingVitProofAssertion = missingVitSnapshot.boundaryAssertions.find(assertion => assertion.field === 'phaseChunkSize.vitBlock');
assert.equal(missingVitSnapshot.status, 'scheduler-unverified', 'requested ViT block chunking must remain unverified when only patch chunk events are observed');
assert.equal(missingVitProofAssertion.status, 'unverified');
assert.equal(missingVitProofAssertion.observedCount, 0);
assert.equal(missingVitProofAssertion.observedYieldCount, 0);

const mainSource = readFileSync(mainPath, 'utf8');
assert.match(mainSource, /parseSharpSchedulerConfig/, 'main entry must parse caller scheduler config at run time');
assert.match(mainSource, /window\.__SHARP_LAST_RUN_TELEMETRY__/, 'browser route must expose last scheduler telemetry for Kaminos');
assert.match(mainSource, /schedulerTelemetrySnapshot/, 'main entry must publish a normalized scheduler telemetry snapshot');
assert.match(mainSource, /scheduler:\s*sharpRouteDefinition\.scheduler/, 'route debug scheduler must preserve the route scheduler receipt shape for contention witnesses');
assert.match(mainSource, /sharpScheduler:\s*null/, 'route debug must expose the SHARP scheduler config on a distinct field');
assert.doesNotMatch(mainSource, /runDebug\.scheduler\s*=\s*currentScheduler/, 'SHARP scheduler config must not overwrite the route scheduler debug shape');
assert.match(mainSource, /runDebug\.sharpScheduler\s*=\s*currentScheduler/, 'main entry must bind the per-run SHARP scheduler config to sharpScheduler');
assert.match(mainSource, /monodepth\.run\([\s\S]*weights,\s*\{\s*scheduler:\s*currentScheduler,\s*telemetry:\s*currentSchedulerTelemetry,\s*\}/, 'main entry must pass scheduler telemetry into monodepth');
assert.match(mainSource, /schedulerYield/, 'main route tail must use schedulerYield for cooperative tail checkpoints');
assert.match(mainSource, /routeTailTimings/, 'main route tail must record per-step route tail timing deltas');
assert.match(mainSource, /routeTailTimings:\s*runDebug\.routeTailTimings/, 'route receipt metadata must preserve route-tail timing deltas');
assert.match(mainSource, /backgroundDutyMap:\s*createSharpRuntimeDutyMap\(\)/, 'run debug must expose the SHARP background duty map');
assert.match(mainSource, /backgroundDutyMap:\s*runDebug\.backgroundDutyMap/, 'route receipt metadata must preserve the background duty map');
assert.match(mainSource, /recordCpuDutyChunk/, 'main route tail must chunk CPU materialization work under scheduler control');
assert.match(mainSource, /receipt\.metadataPayload\s*=\s*metadata/, 'route receipt object must carry the concrete route-tail metadata payload, not only a metadata hash');
assert.match(mainSource, /['"]route-tail['"]/, 'main route tail must emit route-tail scheduler telemetry');
for (const [stage, steps] of [
  ['output-capture', ['disparity-readback', 'depth-preview-render']],
  ['compose-ply', ['geometry-delta-readback', 'texture-delta-readback', 'compose-export']],
]) {
  assert.match(mainSource, new RegExp(`stage:\\s*['"]${escapeRegExp(stage)}['"]`), `main route tail must record ${stage} timing stage identity`);
  for (const step of steps) {
    assert.match(mainSource, new RegExp(`step:\\s*['"]${escapeRegExp(step)}['"]`), `main route tail must record ${stage}/${step} timing identity`);
  }
}

const spnSource = readFileSync(spnPath, 'utf8');
assert.doesNotMatch(spnSource, /const\s+CHUNK_SIZE\s*=\s*4/, 'SPN patch chunking must not be a hidden singleton constant');
assert.match(spnSource, /effective\.spnPatchChunkSize/, 'SPN patch chunking must use the effective scheduler config');
assert.match(spnSource, /spn-patch-chunk/, 'SPN must record breathing evidence around patch chunks');

const executableSpnSource = spnSource
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/\/\/.*$/gm, '');

function assertSpnFusionYieldAfterSubmit(block) {
  const pattern = new RegExp(
    `await\\s+schedulerYield\\(\\s*scheduler\\s*,\\s*device\\s*,\\s*telemetry\\s*,\\s*['"]spn-fusion['"]\\s*,\\s*\\{\\s*block:\\s*['"]${escapeRegExp(block)}['"]\\s*\\}`,
    'g'
  );
  const matches = findAllMatches(executableSpnSource, pattern);
  assert.equal(matches.length, 1, `SPN must record one spn-fusion scheduler boundary for ${block}`);
  const precedingSource = executableSpnSource.slice(Math.max(0, matches[0].index - 1000), matches[0].index);
  const lastSubmit = precedingSource.lastIndexOf('device.queue.submit([');
  assert.notEqual(lastSubmit, -1, `SPN must submit GPU work before awaiting ${block}`);
  const afterSubmit = precedingSource.slice(lastSubmit);
  assert.match(
    afterSubmit,
    /^device\.queue\.submit\(\[[a-zA-Z0-9_$]+\.finish\(\)\]\);/,
    `SPN must submit a finished command buffer before awaiting ${block}`
  );
  assert.doesNotMatch(
    afterSubmit,
    /await\s+schedulerYield\(/,
    `SPN must not record another scheduler boundary between submit and ${block}`
  );
}

function assertSpnYieldAfterReadback(varName, block) {
  const assignmentPattern = new RegExp(
    `const\\s+${escapeRegExp(varName)}\\s*=\\s*await\\s+readBuffer\\(`,
    'g'
  );
  const yieldPattern = new RegExp(
    `await\\s+schedulerYield\\(\\s*scheduler\\s*,\\s*device\\s*,\\s*telemetry\\s*,\\s*['"]spn-fusion['"]\\s*,\\s*\\{\\s*block:\\s*['"]${escapeRegExp(block)}['"]\\s*\\}`,
    'g'
  );
  const assignment = findAllMatches(executableSpnSource, assignmentPattern)[0];
  const yieldMatch = findAllMatches(executableSpnSource, yieldPattern)[0];
  assert.ok(assignment, `SPN must read back ${varName} explicitly`);
  assert.ok(yieldMatch, `SPN must yield after ${varName} readback using ${block}`);
  assert.ok(yieldMatch.index > assignment.index, `SPN must await ${block} after ${varName} readback`);
}

function assertSpnYieldAfterUpload(varName, block) {
  const uploadPattern = new RegExp(
    `const\\s+${escapeRegExp(varName)}\\s*=\\s*createStorageBuffer\\(\\s*device\\s*,\\s*concatData\\s*\\)`,
    'g'
  );
  const yieldPattern = new RegExp(
    `await\\s+schedulerYield\\(\\s*scheduler\\s*,\\s*device\\s*,\\s*telemetry\\s*,\\s*['"]spn-fusion['"]\\s*,\\s*\\{\\s*block:\\s*['"]${escapeRegExp(block)}['"]\\s*\\}`,
    'g'
  );
  const upload = findAllMatches(executableSpnSource, uploadPattern)[0];
  const yieldMatch = findAllMatches(executableSpnSource, yieldPattern)[0];
  assert.ok(upload, `SPN must upload ${varName} from concatData explicitly`);
  assert.ok(yieldMatch, `SPN must yield after ${varName} upload using ${block}`);
  assert.ok(yieldMatch.index > upload.index, `SPN must await ${block} after ${varName} upload`);
}

function assertSpnUpsampleInternalLayer(block, minLayerCount) {
  const callPattern = new RegExp(
    `await\\s+this\\._dispatchUpsampleBlock\\([\\s\\S]*?['"]${escapeRegExp(block)}['"]\\s*,\\s*scheduler\\s*,\\s*telemetry\\s*\\)`,
    'g'
  );
  assert.equal(
    findAllMatches(executableSpnSource, callPattern).length,
    1,
    `SPN ${block} must call the upsample helper with scheduler telemetry and block identity`
  );
  const layerBoundaryPattern = new RegExp(
    `block:\\s*\`\\$\\{blockLabel\\}\\.layer-\\$\\{i\\}\`[\\s\\S]*parentBlock:\\s*blockLabel[\\s\\S]*role:\\s*['"]wait-bearing-layer['"]`,
    'm'
  );
  assert.match(
    executableSpnSource,
    layerBoundaryPattern,
    'SPN upsample helper must emit wait-bearing per-layer block telemetry'
  );
  const markerPattern = new RegExp(
    `await\\s+schedulerYield\\(\\s*scheduler\\s*,\\s*device\\s*,\\s*telemetry\\s*,\\s*['"]spn-fusion['"]\\s*,\\s*\\{[\\s\\S]*?block:\\s*['"]${escapeRegExp(block)}['"][\\s\\S]*?role:\\s*['"]group-complete['"][\\s\\S]*?layerCount:\\s*${minLayerCount}`,
    'm'
  );
  assert.match(
    executableSpnSource,
    markerPattern,
    `SPN ${block} must preserve a coarse group-complete marker with layerCount ${minLayerCount}`
  );
}

for (const block of ['upsample-lowres', 'fuse-lowres']) {
  assertSpnFusionYieldAfterSubmit(block);
}
for (const [block, layerCount] of [
  ['upsample_latent0', 4],
  ['upsample_latent1', 3],
  ['upsample0', 2],
  ['upsample1', 2],
  ['upsample2', 2],
]) {
  assertSpnUpsampleInternalLayer(block, layerCount);
}
assertSpnYieldAfterReadback('x2UpData', 'readback-x2-upsampled');
assertSpnYieldAfterReadback('lowresData', 'readback-lowres');
assert.match(executableSpnSource, /block:\s*['"]cpu-concat-lowres['"]/, 'SPN must expose a scheduler boundary after CPU lowres concat work');
assertSpnYieldAfterUpload('concatBuf', 'concat-upload');

const monodepthSource = readFileSync(monodepthPath, 'utf8');
assert.match(monodepthSource, /schedulerYield/, 'Monodepth decoder must use the scheduler yield primitive');
assert.match(monodepthSource, /monodepth-phase/, 'Monodepth decoder must record phase-level breathing evidence');
assert.match(monodepthSource, /options\s*=\s*\{\}/, 'Monodepth decoder must accept per-run scheduler options');

const executableMonodepthSource = monodepthSource
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/\/\/.*$/gm, '');

function assertMonodepthYieldAfterSubmit(boundary, minCount = 1) {
  const pattern = new RegExp(
    `await\\s+(?:monodepthPhaseYield|boundaryYield)\\(\\s*['"]${escapeRegExp(boundary)}['"]`,
    'g'
  );
  const matches = findAllMatches(executableMonodepthSource, pattern);
  assert.ok(
    matches.length >= minCount,
    `Monodepth decoder must await monodepthPhaseYield('${boundary}') at least ${minCount} time(s)`
  );
  for (const match of matches) {
    const precedingSource = executableMonodepthSource.slice(Math.max(0, match.index - 900), match.index);
    const lastSubmit = precedingSource.lastIndexOf('device.queue.submit([');
    const lastResidualDispatch = precedingSource.lastIndexOf('dispatchResidualBlock(');
    assert.ok(
      lastSubmit !== -1 || lastResidualDispatch !== -1,
      `Monodepth decoder must submit GPU work before awaiting ${boundary}`
    );
    const afterSubmit = lastResidualDispatch > lastSubmit
      ? precedingSource.slice(lastResidualDispatch)
      : precedingSource.slice(lastSubmit);
    if (lastResidualDispatch <= lastSubmit) {
      assert.match(
        afterSubmit,
        /^device\.queue\.submit\(\[[a-zA-Z0-9_$]+\.finish\(\)\]\);/,
        `Monodepth decoder must submit a finished command buffer before awaiting ${boundary}`
      );
    }
    assert.doesNotMatch(
      afterSubmit,
      /await\s+(?:monodepthPhaseYield|boundaryYield)\(/,
      `Monodepth decoder must not record another monodepth scheduler boundary between submit and ${boundary}`
    );
  }
}

function assertMonodepthContinuityMarker(boundary) {
  const markerPattern = new RegExp(
    `await\\s+boundaryYield\\(\\s*['"]${escapeRegExp(boundary)}['"]\\s*,\\s*\\{([\\s\\S]*?)\\}\\s*\\);`
  );
  const marker = executableMonodepthSource.match(markerPattern);
  assert.ok(marker, `Monodepth must preserve coarse ${boundary} coverage labels for Wake`);
  assert.match(marker[1], /role:\s*['"]group-complete['"]/, `Monodepth ${boundary} must declare itself as a group-complete continuity marker`);
  const childList = marker[1].match(/waitBearingBoundaries:\s*\[([\s\S]*?)\]/);
  assert.ok(
    childList,
    `Monodepth ${boundary} must name wait-bearing child boundaries`
  );
  for (const childBoundary of ['residual-conv1', 'residual-conv2', 'residual-skip-add']) {
    assert.match(
      childList[1],
      new RegExp(`['"]${escapeRegExp(childBoundary)}['"]`),
      `Monodepth ${boundary} must name wait-bearing child boundary ${childBoundary}`
    );
  }
}

assert.match(
  executableMonodepthSource,
  /function\s+dispatchResidualBlock[\s\S]*device\.queue\.submit\(\[[a-zA-Z0-9_$]+\.finish\(\)\]\);[\s\S]*return\s+sumBuf;/,
  'Monodepth residual helper must submit GPU work before caller-level fusion residual yields'
);

for (const [boundary, minCount] of [
  ['project-feature', 1],
  ['residual-conv1', 1],
  ['residual-conv2', 1],
  ['residual-skip-add', 1],
  ['fusion-skip-add', 1],
  ['fusion-deconv', 1],
  ['fusion-out-conv', 1],
  ['head-conv0', 1],
  ['head-deconv', 1],
  ['head-conv2', 1],
  ['head-relu3', 1],
  ['head-conv4', 1],
  ['head-final', 1],
]) {
  assertMonodepthYieldAfterSubmit(boundary, minCount);
}

assert.match(executableMonodepthSource, /await\s+boundaryYield\(\s*['"]fusion-resnet1['"]/, 'Monodepth must preserve coarse fusion-resnet1 coverage labels for Wake');
assert.match(executableMonodepthSource, /await\s+boundaryYield\(\s*['"]fusion-resnet2['"]/, 'Monodepth must preserve coarse fusion-resnet2 coverage labels for Wake');
assertMonodepthContinuityMarker('fusion-resnet1');
assertMonodepthContinuityMarker('fusion-resnet2');
assert.match(executableMonodepthSource, /await\s+monodepthPhaseYield\(\s*['"]head-final['"]/, 'Monodepth must preserve coarse head-final coverage label for Wake');

const backboneSource = readFileSync(backbonePath, 'utf8');
assert.match(backboneSource, /schedulerYield/, 'ViT encoder must use the scheduler yield primitive');
assert.match(backboneSource, /effective\.vitBlockChunkSize/, 'ViT encoder must use the effective scheduler block chunk size');
assert.match(backboneSource, /vit-block-chunk/, 'ViT encoder must record breathing evidence around block chunks');

const gaussianSource = readFileSync(gaussianPath, 'utf8');
assert.match(gaussianSource, /gaussianPhaseYieldMs/, 'Gaussian decoder phase breathing must use the scheduler config');
assert.match(gaussianSource, /gaussian-phase/, 'Gaussian decoder must record phase-level breathing evidence');

const executableGaussianSource = gaussianSource
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/\/\/.*$/gm, '');

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function findAllMatches(source, pattern) {
  return Array.from(source.matchAll(pattern));
}

function assertAwaitedYieldAfterSubmit(source, yieldName, boundary, minCount = 1) {
  const pattern = new RegExp(`await\\s+${yieldName}\\(\\s*['"]${escapeRegExp(boundary)}['"]`, 'g');
  const matches = findAllMatches(source, pattern);
  assert.ok(
    matches.length >= minCount,
    `Gaussian decoder must await ${yieldName}('${boundary}') at least ${minCount} time(s)`
  );

  for (const match of matches) {
    const precedingSource = source.slice(Math.max(0, match.index - 800), match.index);
    const lastSubmit = precedingSource.lastIndexOf('device.queue.submit([');
    assert.notEqual(
      lastSubmit,
      -1,
      `Gaussian decoder must submit a finished command buffer before awaiting ${boundary}`
    );
    const afterSubmit = precedingSource.slice(lastSubmit);
    assert.match(
      afterSubmit,
      /^device\.queue\.submit\(\[[a-zA-Z0-9_$]+\.finish\(\)\]\);/,
      `Gaussian decoder must submit a finished command buffer before awaiting ${boundary}`
    );
    assert.doesNotMatch(
      afterSubmit,
      /await\s+(?:boundaryYield|gaussianPhaseYield)\(/,
      `Gaussian decoder must not record another scheduler boundary between submit and ${boundary}`
    );
  }
}

for (const [boundary, minCount] of [
  ['residual-conv1', 1],
  ['residual-conv2', 1],
  ['residual-skip-add', 2],
  ['fusion-skip-add', 1],
  ['fusion-deconv', 1],
  ['fusion-out-conv', 1],
  ['head-gn-conv1', 1],
  ['head-gn-conv2', 1],
  ['head-final', 1],
]) {
  assertAwaitedYieldAfterSubmit(executableGaussianSource, 'boundaryYield', boundary, minCount);
}
assertAwaitedYieldAfterSubmit(executableGaussianSource, 'gaussianPhaseYield', 'project-feature', 2);
assertAwaitedYieldAfterSubmit(executableGaussianSource, 'gaussianPhaseYield', 'prediction-geometry');
assertAwaitedYieldAfterSubmit(executableGaussianSource, 'gaussianPhaseYield', 'prediction-texture');

assert.match(gaussianSource, /async function dispatchResidualBlock/, 'Gaussian residual blocks must be split by awaitable submit/yield boundaries');
assert.match(gaussianSource, /async function dispatchFusionBlock/, 'Gaussian fusion blocks must be split by awaitable submit/yield boundaries');
assert.match(gaussianSource, /async function dispatchGroupNormResidualBlock/, 'Gaussian head residual blocks must be split by awaitable submit/yield boundaries');
assert.match(gaussianSource, /let\s+features\s*=\s*await\s+dispatchFusionBlock/, 'Gaussian decoder must await the initial decoder fusion block');
assert.match(gaussianSource, /features\s*=\s*await\s+dispatchFusionBlock/, 'Gaussian decoder must await decoder fusion loop blocks');
assert.match(gaussianSource, /const\s+fused\s*=\s*await\s+dispatchFusionBlock/, 'Gaussian decoder must await skip-fusion block');
assert.match(gaussianSource, /const\s+textureFeatures\s*=\s*await\s+dispatchHead/, 'Gaussian decoder must await texture head');
assert.match(gaussianSource, /const\s+geometryFeatures\s*=\s*await\s+dispatchHead/, 'Gaussian decoder must await geometry head');
for (const line of executableGaussianSource.split('\n')) {
  if (
    /dispatch(?:ResidualBlock|FusionBlock|GroupNormResidualBlock|Head)\(/.test(line) &&
    !/async function/.test(line)
  ) {
    assert.match(line, /\bawait\s+dispatch/, `Gaussian helper call must be awaited: ${line.trim()}`);
  }
}

const contentionWitnessSource = readFileSync(contentionWitnessPath, 'utf8');
assert.match(contentionWitnessSource, /--sharp-scheduler/, 'contention witness must expose the SHARP scheduler query config as an invocation parameter');
assert.match(contentionWitnessSource, /searchParams\.set\('sharpScheduler'/, 'contention witness must pass the requested scheduler to the browser route URL');
