import assert from 'node:assert/strict';

import {
  createSharpRunTelemetry,
  parseSharpSchedulerConfig,
  recordSchedulerEvent,
  schedulerTelemetrySnapshot,
  schedulerYield,
} from '../src/lib/scheduler.js';

const originalRequestAnimationFrame = globalThis.requestAnimationFrame;
const originalCancelAnimationFrame = globalThis.cancelAnimationFrame;
const originalSetTimeout = globalThis.setTimeout;

let frameCallback = null;
let cancelledFrameId = null;
let configuredDonationCount = 0;

globalThis.requestAnimationFrame = callback => {
  frameCallback = callback;
  return 41;
};
globalThis.cancelAnimationFrame = frameId => {
  cancelledFrameId = frameId;
};
globalThis.setTimeout = (callback, delay, ...args) => {
  if (delay === 5) configuredDonationCount += 1;
  return originalSetTimeout(callback, delay === 5 ? 0 : delay, ...args);
};

try {
  const scheduler = parseSharpSchedulerConfig({
    sharpScheduler: {
      mode: 'cooperative',
      yieldMs: 5,
      waitForSubmittedWorkDone: true,
    },
  });

  let resolveQueue;
  const device = {
    queue: {
      onSubmittedWorkDone() {
        return new Promise(resolve => {
          resolveQueue = resolve;
        });
      },
    },
  };
  const telemetry = createSharpRunTelemetry(scheduler, {
    runId: 'frame-opportunity-observed',
  });
  const yieldPromise = schedulerYield(
    scheduler,
    device,
    telemetry,
    'spn-patch-chunk',
    {
      chunkStart: 0,
      chunkEnd: 1,
      totalPatches: 1,
      commandSubmittedAtMs: performance.now(),
    },
  );

  await Promise.resolve();
  await Promise.resolve();
  assert.equal(typeof frameCallback, 'function', 'the boundary must arm a browser frame observation');
  assert.equal(typeof resolveQueue, 'function', 'the boundary must wait for the submitted queue prefix');
  frameCallback(performance.now());
  resolveQueue();

  const receipt = await yieldPromise;
  const snapshot = schedulerTelemetrySnapshot(telemetry);
  assert.equal(configuredDonationCount, 0, 'an observed frame during queue wait must skip the redundant timer');
  assert.equal(receipt.requestedYieldMs, 5);
  assert.equal(receipt.appliedYieldMs, 0);
  assert.equal(receipt.frameOpportunityArmed, true);
  assert.equal(receipt.frameOpportunityObserved, true);
  assert.equal(receipt.donationDecision, 'skipped-after-frame-opportunity');
  assert.ok(
    snapshot.eventTrace.events.some(event => (
      event.kind === 'js-yield-skipped'
      && event.requestedYieldMs === 5
      && event.appliedYieldMs === 0
      && event.frameOpportunityObserved === true
    )),
    'telemetry must distinguish a proven redundant timer from an applied donation',
  );
  assert.equal(
    snapshot.boundaryAssertions[0].status,
    'verified',
    'a frame observed during the queue wait must satisfy the donation proof without pretending a timer ran',
  );

  frameCallback = null;
  configuredDonationCount = 0;
  const fallbackTelemetry = createSharpRunTelemetry(scheduler, {
    runId: 'frame-opportunity-not-observed',
  });
  const fallbackReceipt = await schedulerYield(
    scheduler,
    { queue: { onSubmittedWorkDone: async () => {} } },
    fallbackTelemetry,
    'spn-patch-chunk',
    {
      chunkStart: 0,
      chunkEnd: 1,
      totalPatches: 1,
      commandSubmittedAtMs: performance.now(),
    },
  );
  assert.equal(configuredDonationCount, 1, 'no observed frame must retain the complete configured timer');
  assert.equal(fallbackReceipt.requestedYieldMs, 5);
  assert.equal(fallbackReceipt.appliedYieldMs, 5);
  assert.equal(fallbackReceipt.frameOpportunityArmed, true);
  assert.equal(fallbackReceipt.frameOpportunityObserved, false);
  assert.equal(fallbackReceipt.donationDecision, 'applied-no-frame-opportunity');
  assert.equal(cancelledFrameId, 41, 'an unobserved frame callback must be cancelled instead of accumulating');

  frameCallback = null;
  const zeroDelayScheduler = parseSharpSchedulerConfig({
    sharpScheduler: {
      mode: 'cooperative',
      yieldMs: 0,
      waitForSubmittedWorkDone: true,
    },
  });
  const zeroDelayReceipt = await schedulerYield(
    zeroDelayScheduler,
    { queue: { onSubmittedWorkDone: async () => {} } },
    createSharpRunTelemetry(zeroDelayScheduler, { runId: 'zero-delay-no-probe' }),
    'spn-patch-chunk',
    {
      chunkStart: 0,
      chunkEnd: 1,
      totalPatches: 1,
      commandSubmittedAtMs: performance.now(),
    },
  );
  assert.equal(frameCallback, null, 'a zero-delay boundary must not manufacture a RAF probe');
  assert.equal(zeroDelayReceipt.frameOpportunityArmed, false);
  assert.equal(zeroDelayReceipt.appliedYieldMs, 0);
  assert.equal(zeroDelayReceipt.donationDecision, 'applied-zero-delay-boundary');

  const forgedTelemetry = createSharpRunTelemetry(scheduler, {
    runId: 'forged-frame-opportunity-skip',
  });
  const forgedIdentity = {
    boundary: 'spn-patch-chunk',
    dutyId: 'forged-frame-opportunity-skip:spn-patch-chunk:0',
    chunkStart: 0,
    chunkEnd: 1,
    totalPatches: 2,
  };
  recordSchedulerEvent(forgedTelemetry, 'spn-patch-chunk', {
    ...forgedIdentity,
    kind: 'chunk-start',
  });
  recordSchedulerEvent(forgedTelemetry, 'spn-patch-chunk', {
    ...forgedIdentity,
    kind: 'queue-work-done-start',
  });
  recordSchedulerEvent(forgedTelemetry, 'spn-patch-chunk', {
    ...forgedIdentity,
    kind: 'queue-work-done-end',
  });
  recordSchedulerEvent(forgedTelemetry, 'spn-patch-chunk', {
    ...forgedIdentity,
    kind: 'js-yield-skipped',
    requestedYieldMs: 5,
    appliedYieldMs: 5,
    frameOpportunityArmed: true,
    frameOpportunitySupported: true,
    frameOpportunityObserved: false,
    donationDecision: 'applied-no-frame-opportunity',
  });
  const forgedSnapshot = schedulerTelemetrySnapshot(forgedTelemetry);
  assert.equal(
    forgedSnapshot.boundaryAssertions[0].status,
    'unverified',
    'an event kind alone must not launder a malformed frame-opportunity skip into proof',
  );
  assert.equal(forgedSnapshot.boundaryAssertions[0].observedFrameOpportunitySkipCount, 0);

  const partialTelemetry = createSharpRunTelemetry(scheduler, {
    runId: 'partial-donation-coverage',
  });
  const recordDuty = ({ dutyId, chunkStart, donation }) => {
    const identity = {
      boundary: 'spn-patch-chunk',
      dutyId,
      chunkStart,
      chunkEnd: chunkStart + 1,
      totalPatches: 2,
    };
    recordSchedulerEvent(partialTelemetry, 'spn-patch-chunk', {
      ...identity,
      kind: 'chunk-start',
    });
    recordSchedulerEvent(partialTelemetry, 'spn-patch-chunk', {
      ...identity,
      kind: 'queue-work-done-start',
    });
    recordSchedulerEvent(partialTelemetry, 'spn-patch-chunk', {
      ...identity,
      kind: 'queue-work-done-end',
    });
    if (donation === 'timer') {
      recordSchedulerEvent(partialTelemetry, 'spn-patch-chunk', {
        ...identity,
        kind: 'js-yield-start',
        requestedYieldMs: 5,
        appliedYieldMs: 5,
        donationDecision: 'applied-no-frame-opportunity',
      });
      recordSchedulerEvent(partialTelemetry, 'spn-patch-chunk', {
        ...identity,
        kind: 'js-yield-end',
        requestedYieldMs: 5,
        appliedYieldMs: 5,
        donationDecision: 'applied-no-frame-opportunity',
      });
    }
  };
  recordDuty({
    dutyId: 'partial-donation-coverage:spn-patch-chunk:0',
    chunkStart: 0,
    donation: 'timer',
  });
  recordDuty({
    dutyId: 'partial-donation-coverage:spn-patch-chunk:1',
    chunkStart: 1,
    donation: null,
  });
  const partialSnapshot = schedulerTelemetrySnapshot(partialTelemetry);
  assert.equal(partialSnapshot.boundaryAssertions[0].observedCount, 2);
  assert.equal(
    partialSnapshot.boundaryAssertions[0].status,
    'unverified',
    'one timer pair must not verify two covered duties',
  );

  const secondDutyIdentity = {
    boundary: 'spn-patch-chunk',
    dutyId: 'partial-donation-coverage:spn-patch-chunk:1',
    chunkStart: 1,
    chunkEnd: 2,
    totalPatches: 2,
  };
  recordSchedulerEvent(partialTelemetry, 'spn-patch-chunk', {
    ...secondDutyIdentity,
    kind: 'js-yield-skipped',
    requestedYieldMs: 5,
    appliedYieldMs: 0,
    frameOpportunityArmed: true,
    frameOpportunitySupported: true,
    frameOpportunityObserved: true,
    donationDecision: 'skipped-after-frame-opportunity',
  });
  const completeSnapshot = schedulerTelemetrySnapshot(partialTelemetry);
  assert.equal(completeSnapshot.boundaryAssertions[0].observedYieldCount, 1);
  assert.equal(completeSnapshot.boundaryAssertions[0].observedFrameOpportunitySkipCount, 1);
  assert.equal(
    completeSnapshot.boundaryAssertions[0].status,
    'verified',
    'one exact timer pair plus one valid same-duty skip must verify two covered duties',
  );

  const unbrandedTelemetry = createSharpRunTelemetry(scheduler, {
    runId: 'unbranded-duty-coverage',
  });
  const brandedDuty = {
    boundary: 'spn-patch-chunk',
    dutyId: 'unbranded-duty-coverage:spn-patch-chunk:0',
    chunkStart: 0,
    chunkEnd: 1,
    totalPatches: 2,
  };
  for (const kind of [
    'chunk-start',
    'queue-work-done-start',
    'queue-work-done-end',
    'js-yield-start',
    'js-yield-end',
  ]) {
    recordSchedulerEvent(unbrandedTelemetry, 'spn-patch-chunk', {
      ...brandedDuty,
      kind,
    });
  }
  recordSchedulerEvent(unbrandedTelemetry, 'spn-patch-chunk', {
    boundary: 'spn-patch-chunk',
    kind: 'chunk-start',
    chunkStart: 1,
    chunkEnd: 2,
    totalPatches: 2,
  });
  const unbrandedSnapshot = schedulerTelemetrySnapshot(unbrandedTelemetry);
  assert.equal(
    unbrandedSnapshot.boundaryAssertions[0].observedCount,
    2,
    'every observed duty start must count even when its identity is malformed',
  );
  assert.equal(
    unbrandedSnapshot.boundaryAssertions[0].status,
    'unverified',
    'a verifier must not discard an unbranded duty and certify the well-formed subset',
  );
} finally {
  globalThis.requestAnimationFrame = originalRequestAnimationFrame;
  globalThis.cancelAnimationFrame = originalCancelAnimationFrame;
  globalThis.setTimeout = originalSetTimeout;
}

console.log('SHARP frame-opportunity donation contracts passed.');
