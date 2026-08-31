import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  adaptiveDecoderTimingObservation,
  captureQueueCompletionFence,
  captureQueueCompletionFencePair,
  createSharpRunTelemetry,
  parseSharpSchedulerConfig,
  schedulerYield,
} from '../src/lib/scheduler.js';

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

async function flushPromiseCallbacks() {
  await Promise.resolve();
  await Promise.resolve();
}

async function withFakeClock(callback) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'performance');
  let now = 0;
  Object.defineProperty(globalThis, 'performance', {
    configurable: true,
    value: {
      timeOrigin: 1_800_000_000_000,
      now: () => now,
    },
  });
  try {
    await callback(value => { now = value; });
  } finally {
    if (descriptor) Object.defineProperty(globalThis, 'performance', descriptor);
    else delete globalThis.performance;
  }
}

const scheduler = parseSharpSchedulerConfig({
  sharpScheduler: {
    mode: 'cooperative',
    waitForSubmittedWorkDone: true,
    yieldMs: 0,
  },
});

await withFakeClock(async setNow => {
  const completions = [deferred(), deferred()];
  let completionQueryCount = 0;
  const queue = {
    onSubmittedWorkDone() {
      const completion = completions[completionQueryCount];
      completionQueryCount += 1;
      return completion.promise;
    },
  };
  const device = { queue };

  setNow(100);
  const preSubmitFence = captureQueueCompletionFence(device);
  setNow(101);
  const commandSubmittedAtMs = performance.now();
  setNow(102);
  const fencePair = captureQueueCompletionFencePair(device, preSubmitFence, commandSubmittedAtMs);
  assert.equal(completionQueryCount, 2, 'the pair captures exactly one pre-submit and one post-submit queue prefix');

  setNow(120);
  completions[0].resolve();
  await flushPromiseCallbacks();
  setNow(123);
  completions[1].resolve();
  await flushPromiseCallbacks();

  setNow(124);
  const receipt = await schedulerYield(
    scheduler,
    device,
    createSharpRunTelemetry(scheduler, { runId: 'dual-fence-primary' }),
    'gaussian-phase',
    {
      commandSubmittedAtMs,
      requestedQueueTimingAuthority: 'incremental-submitted-range',
    },
    0,
    fencePair,
  );
  assert.equal(completionQueryCount, 2, 'consuming the pair must not issue a third queue completion query');
  assert.equal(receipt.prePrefixCompletedAtMs, 120);
  assert.equal(receipt.queueCompletedAtMs, 123);
  assert.equal(receipt.prePrefixWallMs, 20);
  assert.equal(receipt.submitToQueueDoneMs, 22);
  assert.equal(receipt.incrementalSubmitToQueueDoneMs, 3);
  assert.equal(receipt.requestedQueueTimingAuthority, 'incremental-submitted-range');
  assert.equal(receipt.effectiveQueueTimingAuthority, 'incremental-submitted-range');
  assert.equal(receipt.queueTimingFallbackReason, null);
  assert.equal(receipt.queueWorkAttribution, 'incremental-submitted-range');

  const timing = adaptiveDecoderTimingObservation(receipt);
  assert.equal(timing.observedDurationMs, 3, 'the planner consumes only the incremental SHARP interval');
  assert.equal(timing.rawSubmitToQueueDoneMs, 22, 'the raw shared-prefix wall remains visible');
  assert.equal(timing.prePrefixWallMs, 20, 'the preceding shared-prefix wall remains visible');
  assert.equal(timing.effectiveQueueTimingAuthority, 'incremental-submitted-range');

  await assert.rejects(
    () => schedulerYield(
      scheduler,
      device,
      createSharpRunTelemetry(scheduler, { runId: 'dual-fence-replay' }),
      'gaussian-phase',
      { commandSubmittedAtMs, requestedQueueTimingAuthority: 'incremental-submitted-range' },
      0,
      fencePair,
    ),
    /queue completion fence pair has already been consumed/,
    'a paired timing witness is single-use',
  );
});

await withFakeClock(async setNow => {
  const preQueue = { onSubmittedWorkDone: () => Promise.resolve() };
  const postQueue = { onSubmittedWorkDone: () => Promise.resolve() };
  setNow(200);
  const preSubmitFence = captureQueueCompletionFence({ queue: preQueue });
  setNow(201);
  assert.throws(
    () => captureQueueCompletionFencePair({ queue: postQueue }, preSubmitFence, performance.now()),
    /same active device queue/,
    'a mixed-queue pair fails before it can become adaptive evidence',
  );
  assert.throws(
    () => captureQueueCompletionFencePair({ queue: preQueue }, null, performance.now()),
    /valid pre-submit queue completion fence/,
    'a missing pre-submit fence cannot masquerade as incremental timing',
  );
});

await withFakeClock(async setNow => {
  const completions = [deferred(), deferred()];
  let index = 0;
  const queue = { onSubmittedWorkDone: () => completions[index++].promise };
  const device = { queue };
  setNow(300);
  const preSubmitFence = captureQueueCompletionFence(device);
  setNow(301);
  const commandSubmittedAtMs = performance.now();
  setNow(302);
  const pair = captureQueueCompletionFencePair(device, preSubmitFence, commandSubmittedAtMs);
  setNow(320);
  completions[1].resolve();
  await flushPromiseCallbacks();
  setNow(321);
  completions[0].resolve();
  await flushPromiseCallbacks();
  await assert.rejects(
    () => schedulerYield(
      scheduler,
      device,
      createSharpRunTelemetry(scheduler, { runId: 'dual-fence-misordered' }),
      'gaussian-phase',
      { commandSubmittedAtMs, requestedQueueTimingAuthority: 'incremental-submitted-range' },
      0,
      pair,
    ),
    /completion order is invalid/,
    'a post-prefix timestamp preceding the pre-prefix timestamp fails loud',
  );
});

await withFakeClock(async setNow => {
  const completions = [deferred(), deferred()];
  let index = 0;
  const queue = { onSubmittedWorkDone: () => completions[index++].promise };
  const device = { queue };
  setNow(400);
  const preSubmitFence = captureQueueCompletionFence(device);
  setNow(401);
  const commandSubmittedAtMs = performance.now();
  setNow(402);
  const pair = captureQueueCompletionFencePair(device, preSubmitFence, commandSubmittedAtMs);
  setNow(420);
  completions[0].reject(new Error('pre-prefix device loss'));
  await flushPromiseCallbacks();
  setNow(421);
  completions[1].resolve();
  await flushPromiseCallbacks();
  await assert.rejects(
    () => schedulerYield(
      scheduler,
      device,
      createSharpRunTelemetry(scheduler, { runId: 'dual-fence-rejected-pre' }),
      'gaussian-phase',
      { commandSubmittedAtMs, requestedQueueTimingAuthority: 'incremental-submitted-range' },
      0,
      pair,
    ),
    /pre-submit queue completion fence failed/,
    'a rejected pre-prefix cannot silently fall back after claiming paired authority',
  );
});

await withFakeClock(async setNow => {
  const completions = [deferred(), deferred()];
  let index = 0;
  const queue = { onSubmittedWorkDone: () => completions[index++].promise };
  const device = { queue };
  setNow(500);
  const preSubmitFence = captureQueueCompletionFence(device);
  setNow(501);
  const commandSubmittedAtMs = performance.now();
  setNow(502);
  const pair = captureQueueCompletionFencePair(device, preSubmitFence, commandSubmittedAtMs);
  setNow(510);
  completions[0].resolve();
  await flushPromiseCallbacks();
  setNow(511);
  completions[1].resolve();
  await flushPromiseCallbacks();
  await assert.rejects(
    () => schedulerYield(
      scheduler,
      device,
      createSharpRunTelemetry(scheduler, { runId: 'dual-fence-wrong-command' }),
      'gaussian-phase',
      { commandSubmittedAtMs: commandSubmittedAtMs + 1, requestedQueueTimingAuthority: 'incremental-submitted-range' },
      0,
      pair,
    ),
    /does not belong to the submitted command/,
    'a pair bound around another command cannot become timing evidence',
  );
});

await withFakeClock(async setNow => {
  const completion = deferred();
  let completionQueryCount = 0;
  const queue = {
    onSubmittedWorkDone() {
      completionQueryCount += 1;
      return completion.promise;
    },
  };
  const device = { queue };
  setNow(600);
  const commandSubmittedAtMs = performance.now();
  setNow(601);
  const singleFence = captureQueueCompletionFence(device);
  setNow(640);
  completion.resolve();
  await flushPromiseCallbacks();
  const receipt = await schedulerYield(
    scheduler,
    device,
    createSharpRunTelemetry(scheduler, { runId: 'single-fence-fallback' }),
    'gaussian-phase',
    { commandSubmittedAtMs, requestedQueueTimingAuthority: 'incremental-submitted-range' },
    0,
    singleFence,
  );
  assert.equal(completionQueryCount, 1);
  assert.equal(receipt.requestedQueueTimingAuthority, 'incremental-submitted-range');
  assert.equal(receipt.effectiveQueueTimingAuthority, 'submitted-range-prefix');
  assert.equal(receipt.queueTimingFallbackReason, 'dual-fence-unavailable');
  assert.equal(receipt.incrementalSubmitToQueueDoneMs, null);
  const timing = adaptiveDecoderTimingObservation(receipt);
  assert.equal(timing.observedDurationMs, 40, 'the explicit lower-authority fallback retains raw prefix timing');
  assert.equal(timing.effectiveQueueTimingAuthority, 'submitted-range-prefix');
  assert.equal(timing.queueTimingFallbackReason, 'dual-fence-unavailable');
});

const decoderSource = readFileSync(new URL('../src/lib/decoder_duties.js', import.meta.url), 'utf8');
assert.match(
  decoderSource,
  /captureQueueCompletionFence\(device\)[\s\S]{0,300}device\.queue\.submit[\s\S]{0,300}captureQueueCompletionFencePair/,
  'adaptive decoder duties capture same-queue completion immediately before and after submission',
);
assert.match(
  decoderSource,
  /adaptiveDecoderTimingObservation\(yieldReceipt\)[\s\S]{0,500}planner\.observeRange/,
  'adaptive planning consumes the authority-resolved timing observation',
);

console.log('sharp dual queue fence contracts passed');
