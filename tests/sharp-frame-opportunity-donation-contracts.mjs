import assert from 'node:assert/strict';

import {
  createSharpRunTelemetry,
  parseSharpSchedulerConfig,
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
} finally {
  globalThis.requestAnimationFrame = originalRequestAnimationFrame;
  globalThis.cancelAnimationFrame = originalCancelAnimationFrame;
  globalThis.setTimeout = originalSetTimeout;
}

console.log('SHARP frame-opportunity donation contracts passed.');
