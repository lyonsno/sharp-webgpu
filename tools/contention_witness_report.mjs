export const SHARP_CONTENTION_WITNESS_SCHEMA = 'sharp.webgpu-contention-witness.v0';
export const SHARP_BACKGROUND_HEARTBEAT_SCHEMA = 'sharp-webgpu.background-heartbeat.v0';
export const SHARP_ROUTE_ID = 'sharp.image-to-splat.webgpu-local.v0';
export const CROSS_PAGE_CLOCK_SCHEMA = 'kaminos.browser-epoch-monotonic-clock.v0';
export const GPU_DUTY_INTERVALS_SCHEMA = 'sharp-webgpu.submitted-work-drain-intervals.v0';
export const RAF_INTERVAL_TRACE_SCHEMA = 'sharp-webgpu.raf-interval-trace.v0';

import { createHash } from 'node:crypto';
import {
  classifyWebGpuRouteReceiptEvidence,
} from '@kaminos/webgpu-inference-kit';

const VALID_MODES = new Set(['baseline', 'contention', 'cooperative']);
const VALID_HEARTBEAT_OVERLAP_CLASSIFICATIONS = new Set(['scheduler-event-overlap', 'uninstrumented-gap']);

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function isFiniteNonNegative(value) {
  return Number.isFinite(value) && value >= 0;
}

function requireString(errors, value, path) {
  if (!isNonEmptyString(value)) errors.push(`${path} must be a non-empty string`);
}

function requireFinitePositive(errors, value, path) {
  if (!Number.isFinite(value) || value <= 0) errors.push(`${path} must be a positive finite number`);
}

function schedulerEvents(scheduler) {
  return Array.isArray(scheduler?.eventTrace?.events)
    ? scheduler.eventTrace.events
    : (Array.isArray(scheduler?.events) ? scheduler.events : []);
}

function finiteOrZero(value) {
  return Number.isFinite(value) ? value : 0;
}

function summarizeBoundaries(events) {
  return [...new Set(events.map(event => event?.boundary || event?.phase).filter(Boolean))].sort();
}

function sha256Json(value) {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

function eventOverlapForGap(events, gap) {
  const startMs = gap.startMs;
  const endMs = gap.endMs;
  return events
    .filter(event => {
      const pointOverlap = Number.isFinite(event?.tMs) && event.tMs >= startMs && event.tMs <= endMs;
      const intervalOverlap = Number.isFinite(event?.intervalStartMs)
        && Number.isFinite(event?.intervalEndMs)
        && event.intervalEndMs >= event.intervalStartMs
        && event.intervalStartMs <= endMs
        && event.intervalEndMs >= startMs;
      return pointOverlap || intervalOverlap;
    })
    .slice(0, 20)
    .map(event => ({
      phase: event.phase || 'unknown',
      boundary: event.boundary || event.phase || 'unknown',
      kind: event.kind || 'boundary-event',
      tMs: event.tMs,
      ...(Number.isFinite(event.intervalStartMs) ? { intervalStartMs: event.intervalStartMs } : {}),
      ...(Number.isFinite(event.intervalEndMs) ? { intervalEndMs: event.intervalEndMs } : {}),
      ...(Number.isFinite(event.durationMs) ? { durationMs: event.durationMs } : {}),
      stage: event.stage || null,
      step: event.step || null,
      role: event.role || null,
    }));
}

function normalizeInferenceWindow(window, timeOriginEpochMs = null) {
  if (!isObject(window) || !Number.isFinite(window.startMs) || !Number.isFinite(window.endMs) || window.endMs <= window.startMs) {
    return null;
  }
  const startMs = Number(window.startMs.toFixed(3));
  const endMs = Number(window.endMs.toFixed(3));
  const normalized = {
    ...(isNonEmptyString(window.runId) ? { runId: window.runId } : {}),
    startMs,
    endMs,
    durationMs: Number((endMs - startMs).toFixed(3)),
  };
  const startEpochMs = Number.isFinite(window.startEpochMs)
    ? window.startEpochMs
    : (Number.isFinite(timeOriginEpochMs) ? timeOriginEpochMs + startMs : null);
  const endEpochMs = Number.isFinite(window.endEpochMs)
    ? window.endEpochMs
    : (Number.isFinite(timeOriginEpochMs) ? timeOriginEpochMs + endMs : null);
  if (Number.isFinite(startEpochMs) && Number.isFinite(endEpochMs)) {
    normalized.startEpochMs = Number(startEpochMs.toFixed(3));
    normalized.endEpochMs = Number(endEpochMs.toFixed(3));
  }
  return normalized;
}

function normalizeCrossPageClock(clock, inferenceWindow, schedulerRunId) {
  if (!isObject(clock)
    || clock.schema !== CROSS_PAGE_CLOCK_SCHEMA
    || !Number.isFinite(clock.timeOriginEpochMs)
    || !isNonEmptyString(schedulerRunId)
    || inferenceWindow?.runId !== schedulerRunId
    || !Number.isFinite(inferenceWindow?.startEpochMs)
    || !Number.isFinite(inferenceWindow?.endEpochMs)) {
    return null;
  }
  return {
    schema: CROSS_PAGE_CLOCK_SCHEMA,
    timingAuthority: 'performance-time-origin-plus-now',
    runId: schedulerRunId,
    timeOriginEpochMs: clock.timeOriginEpochMs,
    inferenceWindowStartEpochMs: inferenceWindow.startEpochMs,
    inferenceWindowEndEpochMs: inferenceWindow.endEpochMs,
  };
}

function createRafIntervalTrace(probe, inferenceWindow, schedulerRunId, timeOriginEpochMs) {
  const rawIntervals = Array.isArray(probe.frameGapIntervals) ? probe.frameGapIntervals : [];
  const intervals = rawIntervals
    .filter(interval => (
      Number.isFinite(interval?.startMs)
      && Number.isFinite(interval?.endMs)
      && interval.endMs >= interval.startMs
      && inferenceWindow
      && interval.startMs <= inferenceWindow.endMs
      && interval.endMs >= inferenceWindow.startMs
    ))
    .map(interval => {
      const startMs = Math.max(Number(interval.startMs.toFixed(3)), inferenceWindow.startMs);
      const endMs = Math.min(Number(interval.endMs.toFixed(3)), inferenceWindow.endMs);
      return {
        startMs,
        endMs,
        durationMs: Number((endMs - startMs).toFixed(3)),
        ...(Number.isFinite(timeOriginEpochMs) ? {
          startEpochMs: Number((timeOriginEpochMs + startMs).toFixed(3)),
          endEpochMs: Number((timeOriginEpochMs + endMs).toFixed(3)),
        } : {}),
      };
    })
    .filter(interval => interval.endMs >= interval.startMs);
  return {
    schema: RAF_INTERVAL_TRACE_SCHEMA,
    timingAuthority: 'browser-request-animation-frame-performance-now',
    runId: schedulerRunId || null,
    uncapped: true,
    count: intervals.length,
    timeOriginEpochMs: Number.isFinite(timeOriginEpochMs) ? timeOriginEpochMs : null,
    intervals,
  };
}

function createGpuDutyIntervals(events, inferenceWindow, schedulerRunId) {
  const starts = new Map();
  const intervals = [];
  const pairingFailures = [];

  for (const event of events) {
    if (event?.kind !== 'queue-work-done-start' && event?.kind !== 'queue-work-done-end') continue;
    if (!isNonEmptyString(event.runId) || event.runId !== schedulerRunId) {
      pairingFailures.push(`queue-work-done event runId ${event.runId || '<missing>'} does not match scheduler runId ${schedulerRunId || '<missing>'}`);
      continue;
    }
    if (!isNonEmptyString(event.dutyId)) {
      pairingFailures.push('queue-work-done event missing dutyId');
      continue;
    }
    if (event.kind === 'queue-work-done-start') {
      if (starts.has(event.dutyId)) pairingFailures.push(`duplicate queue-work-done start for ${event.dutyId}`);
      starts.set(event.dutyId, event);
      continue;
    }

    const start = starts.get(event.dutyId);
    if (!start) {
      pairingFailures.push(`queue-work-done end without start for ${event.dutyId}`);
      continue;
    }
    starts.delete(event.dutyId);
    if ((start.phase || 'unknown') !== (event.phase || 'unknown')
      || (start.boundary || start.phase || 'unknown') !== (event.boundary || event.phase || 'unknown')) {
      pairingFailures.push(`queue-work-done endpoints disagree for ${event.dutyId}`);
    }
    const interval = {
      runId: start.runId,
      dutyId: event.dutyId,
      phase: start.phase || event.phase || 'unknown',
      boundary: start.boundary || event.boundary || start.phase || event.phase || 'unknown',
      kind: 'submitted-work-drain-interval',
      startEventSequence: start.sequence,
      endEventSequence: event.sequence,
      startMs: start.tMs,
      endMs: event.tMs,
      durationMs: Number.isFinite(start.tMs) && Number.isFinite(event.tMs)
        ? Number((event.tMs - start.tMs).toFixed(3))
        : null,
      startEpochMs: start.epochMs,
      endEpochMs: event.epochMs,
      stage: start.stage || event.stage || null,
      step: start.step || event.step || null,
      role: start.role || event.role || null,
    };
    const insideInferenceWindow = Number.isFinite(inferenceWindow?.startMs)
      && Number.isFinite(inferenceWindow?.endMs)
      && Number.isFinite(interval.startMs)
      && Number.isFinite(interval.endMs)
      && interval.startMs >= inferenceWindow.startMs
      && interval.endMs <= inferenceWindow.endMs;
    if (insideInferenceWindow) intervals.push(interval);
  }
  for (const dutyId of starts.keys()) pairingFailures.push(`queue-work-done start without end for ${dutyId}`);

  return {
    schema: GPU_DUTY_INTERVALS_SCHEMA,
    timingAuthority: 'queue-on-submitted-work-done-host-await-not-gpu-exclusive',
    runId: schedulerRunId || null,
    count: intervals.length,
    intervals,
    ...(pairingFailures.length ? { pairingFailures } : {}),
  };
}

function gpuDutyIntervalSourceKey(interval) {
  return JSON.stringify([
    interval?.runId ?? null,
    interval?.dutyId ?? null,
    interval?.startEventSequence ?? null,
    interval?.endEventSequence ?? null,
  ]);
}

export function createSharpBackgroundHeartbeatReport({ scheduler = {}, probe = {}, responsiveness = null } = {}) {
  const events = schedulerEvents(scheduler);
  const rawInferenceWindow = probe.inferenceWindow || probe.contender?.inferenceWindow;
  const eventClock = scheduler.eventTrace?.clock;
  const inferenceWindow = normalizeInferenceWindow(rawInferenceWindow, eventClock?.timeOriginEpochMs);
  const crossPageClock = normalizeCrossPageClock(eventClock, inferenceWindow, scheduler.runId);
  const gpuDutyIntervals = createGpuDutyIntervals(events, inferenceWindow, scheduler.runId);
  const rafIntervalTrace = createRafIntervalTrace(
    probe,
    inferenceWindow,
    scheduler.runId,
    probe.timeOriginEpochMs ?? eventClock?.timeOriginEpochMs,
  );
  const probeResponsiveness = responsiveness || {
    rafFrames: probe.rafFrames || 0,
    maxFrameGapMs: probe.maxFrameGapMs || 0,
    p95FrameGapMs: probe.p95FrameGapMs || 0,
    longFrameCount: probe.longFrameCount || 0,
  };
  const rawGaps = Array.isArray(probe.worstFrameGaps)
    ? probe.worstFrameGaps
    : (Array.isArray(probe.frameGapIntervals) ? probe.frameGapIntervals : []);
  const worstFrameGaps = rawGaps
    .filter(gap => (
      Number.isFinite(gap?.startMs)
      && Number.isFinite(gap?.endMs)
      && Number.isFinite(gap?.durationMs)
      && gap.endMs >= gap.startMs
      && inferenceWindow
      && gap.startMs >= rawInferenceWindow.startMs
      && gap.endMs <= rawInferenceWindow.endMs
    ))
    .slice()
    .sort((a, b) => b.durationMs - a.durationMs)
    .slice(0, 8)
    .map(gap => {
      const emittedGap = {
        startMs: Math.max(gap.startMs, inferenceWindow.startMs),
        endMs: Math.min(gap.endMs, inferenceWindow.endMs),
      };
      emittedGap.durationMs = emittedGap.endMs - emittedGap.startMs;
      const overlappedEvents = eventOverlapForGap(events, emittedGap);
      return {
        startMs: Number(finiteOrZero(emittedGap.startMs).toFixed(3)),
        endMs: Number(finiteOrZero(emittedGap.endMs).toFixed(3)),
        durationMs: Number(finiteOrZero(emittedGap.durationMs).toFixed(3)),
        overlapClassification: overlappedEvents.length ? 'scheduler-event-overlap' : 'uninstrumented-gap',
        overlappedEvents,
      };
    });

  return {
    schema: SHARP_BACKGROUND_HEARTBEAT_SCHEMA,
    timingAuthority: events.length ? 'browser-raf-and-scheduler-event-trace' : 'browser-raf-only',
    schedulerMode: scheduler.requestedScheduler?.mode || scheduler.effectiveScheduler?.mode || scheduler.mode || 'unknown',
    verificationState: scheduler.verificationState || scheduler.status || 'scheduler-unverified',
    requestedScheduler: scheduler.requestedScheduler || scheduler.requested || null,
    effectiveScheduler: scheduler.effectiveScheduler || scheduler.effective || null,
    responsiveness: probeResponsiveness,
    eventTrace: {
      schema: scheduler.eventTrace?.schema || 'kaminos.webgpu-scheduler-event-trace.v0',
      timingAuthority: scheduler.eventTrace?.timingAuthority || (events.length ? 'browser-wall-clock' : 'not-observed'),
      clock: isObject(eventClock) ? eventClock : null,
      uncapped: true,
      eventCount: events.length,
      sequenceEnvelope: isObject(scheduler.eventTrace?.sequenceEnvelope)
        ? { ...scheduler.eventTrace.sequenceEnvelope }
        : null,
      boundaries: summarizeBoundaries(events),
      events: events.map(event => ({ ...event })),
    },
    crossPageClock,
    gpuDutyIntervals,
    inferenceWindow,
    rafIntervalTrace,
    worstFrameGaps,
  };
}

export function createSharpContentionWitnessFailureReport({
  candidateReport = {},
  failurePhase = 'unknown',
  error = 'SHARP contention witness failed',
  validation = null,
} = {}) {
  return {
    schema: 'sharp.webgpu-contention-witness-failure.v0',
    runId: candidateReport.runId || `sharp-contention:failure:${Date.now()}`,
    createdAt: candidateReport.createdAt || new Date().toISOString(),
    mode: candidateReport.mode || 'unknown',
    input: candidateReport.input || null,
    failurePhase,
    error,
    validation,
    lastTrustworthyEvidence: {
      route: candidateReport.route || null,
      inference: candidateReport.inference || null,
      responsiveness: candidateReport.responsiveness || null,
      backgroundHeartbeat: candidateReport.backgroundHeartbeat || null,
      scheduler: candidateReport.scheduler || null,
      inferenceWindow: candidateReport.backgroundHeartbeat?.inferenceWindow || null,
    },
  };
}

function validateBackgroundHeartbeat(errors, heartbeat) {
  if (!isObject(heartbeat)) {
    errors.push('backgroundHeartbeat must be an object');
    return;
  }
  if (heartbeat.schema !== SHARP_BACKGROUND_HEARTBEAT_SCHEMA) {
    errors.push(`backgroundHeartbeat.schema must be ${SHARP_BACKGROUND_HEARTBEAT_SCHEMA}`);
  }
  requireString(errors, heartbeat.timingAuthority, 'backgroundHeartbeat.timingAuthority');
  requireString(errors, heartbeat.schedulerMode, 'backgroundHeartbeat.schedulerMode');
  requireString(errors, heartbeat.verificationState, 'backgroundHeartbeat.verificationState');
  if (!isObject(heartbeat.requestedScheduler)) errors.push('backgroundHeartbeat.requestedScheduler must be an object');
  if (!isObject(heartbeat.effectiveScheduler)) errors.push('backgroundHeartbeat.effectiveScheduler must be an object');
  validateResponsiveness(errors, [], heartbeat.responsiveness);

  if (!isObject(heartbeat.eventTrace)) {
    errors.push('backgroundHeartbeat.eventTrace must be an object');
  } else {
    requireString(errors, heartbeat.eventTrace.schema, 'backgroundHeartbeat.eventTrace.schema');
    requireString(errors, heartbeat.eventTrace.timingAuthority, 'backgroundHeartbeat.eventTrace.timingAuthority');
    if (!isFiniteNonNegative(heartbeat.eventTrace.eventCount) || heartbeat.eventTrace.eventCount <= 0) {
      errors.push('backgroundHeartbeat.eventTrace.eventCount must be positive');
    }
    if (!Array.isArray(heartbeat.eventTrace.boundaries) || heartbeat.eventTrace.boundaries.length === 0) {
      errors.push('backgroundHeartbeat.eventTrace.boundaries must be a non-empty array');
    }
    if (heartbeat.eventTrace.uncapped !== true) {
      errors.push('backgroundHeartbeat.eventTrace.uncapped must be true');
    }
    if (!Array.isArray(heartbeat.eventTrace.events) || heartbeat.eventTrace.events.length === 0) {
      errors.push('backgroundHeartbeat.eventTrace.events must be a non-empty uncapped array');
    } else {
      if (heartbeat.eventTrace.eventCount !== heartbeat.eventTrace.events.length) {
        errors.push('backgroundHeartbeat.eventTrace.eventCount must match uncapped events length');
      }
      const derivedBoundaries = summarizeBoundaries(heartbeat.eventTrace.events);
      if (JSON.stringify(derivedBoundaries) !== JSON.stringify(heartbeat.eventTrace.boundaries)) {
        errors.push('backgroundHeartbeat.eventTrace.boundaries must match uncapped events');
      }
      for (const [index, event] of heartbeat.eventTrace.events.entries()) {
        const path = `backgroundHeartbeat.eventTrace.events[${index}]`;
        if (!isObject(event)) {
          errors.push(`${path} must be an object`);
          continue;
        }
        requireString(errors, event.runId, `${path}.runId`);
        if (!Number.isInteger(event.sequence) || event.sequence < 0) {
          errors.push(`${path}.sequence must be a non-negative integer`);
        }
        if (isObject(heartbeat.crossPageClock) && event.runId !== heartbeat.crossPageClock.runId) {
          errors.push(`${path}.runId must match backgroundHeartbeat.crossPageClock.runId`);
        }
        if (!isFiniteNonNegative(event.tMs)) errors.push(`${path}.tMs must be a finite non-negative number`);
        if (!isFiniteNonNegative(event.epochMs)) errors.push(`${path}.epochMs must be a finite non-negative number`);
        const traceClock = heartbeat.eventTrace.clock;
        if (isObject(traceClock) && Number.isFinite(event.tMs) && Number.isFinite(event.epochMs)
          && Math.abs((traceClock.timeOriginEpochMs + event.tMs) - event.epochMs) > 1) {
          errors.push(`${path}.epochMs must use backgroundHeartbeat.eventTrace.clock.timeOriginEpochMs`);
        }
        const hasAnyInterval = event.intervalStartMs !== undefined
          || event.intervalEndMs !== undefined
          || event.intervalStartEpochMs !== undefined
          || event.intervalEndEpochMs !== undefined;
        if (hasAnyInterval) {
          if (!isFiniteNonNegative(event.intervalStartMs) || !isFiniteNonNegative(event.intervalEndMs) || !isFiniteNonNegative(event.durationMs)) {
            errors.push(`${path} interval timing must be complete and finite`);
          } else if (event.intervalEndMs < event.intervalStartMs
            || Math.abs((event.intervalEndMs - event.intervalStartMs) - event.durationMs) > 1) {
            errors.push(`${path} interval timing must be ordered and duration-consistent`);
          }
          if (!isFiniteNonNegative(event.intervalStartEpochMs) || !isFiniteNonNegative(event.intervalEndEpochMs)) {
            errors.push(`${path} intervalStartEpochMs and intervalEndEpochMs must be complete and finite`);
          } else {
            if (event.intervalEndEpochMs < event.intervalStartEpochMs) {
              errors.push(`${path} interval epoch timing must be ordered`);
            }
            if (isObject(heartbeat.eventTrace.clock)) {
              if (Math.abs((heartbeat.eventTrace.clock.timeOriginEpochMs + event.intervalStartMs) - event.intervalStartEpochMs) > 1) {
                errors.push(`${path}.intervalStartEpochMs must use backgroundHeartbeat.eventTrace.clock.timeOriginEpochMs`);
              }
              if (Math.abs((heartbeat.eventTrace.clock.timeOriginEpochMs + event.intervalEndMs) - event.intervalEndEpochMs) > 1) {
                errors.push(`${path}.intervalEndEpochMs must use backgroundHeartbeat.eventTrace.clock.timeOriginEpochMs`);
              }
            }
          }
        }
      }
      const envelope = heartbeat.eventTrace.sequenceEnvelope;
      if (!isObject(envelope)) {
        errors.push('backgroundHeartbeat.eventTrace.sequenceEnvelope must be an object');
      } else {
        for (const field of ['firstSequence', 'lastSequence', 'nextSequence', 'eventCount']) {
          if (!Number.isInteger(envelope[field]) || envelope[field] < 0) {
            errors.push(`backgroundHeartbeat.eventTrace.sequenceEnvelope.${field} must be a non-negative integer`);
          }
        }
        const events = heartbeat.eventTrace.events;
        if (envelope.eventCount !== events.length) {
          errors.push('backgroundHeartbeat.eventTrace.sequenceEnvelope.eventCount must match retained events length');
        }
        if (events.length) {
          if (envelope.firstSequence !== events[0].sequence) {
            errors.push('backgroundHeartbeat.eventTrace.sequenceEnvelope.firstSequence must match the first retained event');
          }
          if (envelope.lastSequence !== events[events.length - 1].sequence) {
            errors.push('backgroundHeartbeat.eventTrace.sequenceEnvelope.lastSequence must match the last retained event');
          }
          if (envelope.nextSequence !== envelope.lastSequence + 1) {
            errors.push('backgroundHeartbeat.eventTrace.sequenceEnvelope.nextSequence must follow lastSequence');
          }
          for (const [index, event] of events.entries()) {
            if (event.sequence !== envelope.firstSequence + index) {
              errors.push(`backgroundHeartbeat.eventTrace.events[${index}].sequence must be contiguous`);
            }
          }
        }
      }
    }
  }

  const crossPageClock = heartbeat.crossPageClock;
  if (!isObject(crossPageClock)) {
    errors.push('backgroundHeartbeat.crossPageClock must be an object');
  } else {
    if (crossPageClock.schema !== CROSS_PAGE_CLOCK_SCHEMA) {
      errors.push(`backgroundHeartbeat.crossPageClock.schema must be ${CROSS_PAGE_CLOCK_SCHEMA}`);
    }
    if (crossPageClock.timingAuthority !== 'performance-time-origin-plus-now') {
      errors.push('backgroundHeartbeat.crossPageClock.timingAuthority must be performance-time-origin-plus-now');
    }
    requireString(errors, crossPageClock.runId, 'backgroundHeartbeat.crossPageClock.runId');
    for (const field of ['timeOriginEpochMs', 'inferenceWindowStartEpochMs', 'inferenceWindowEndEpochMs']) {
      if (!isFiniteNonNegative(crossPageClock[field])) {
        errors.push(`backgroundHeartbeat.crossPageClock.${field} must be a finite non-negative number`);
      }
    }
    if (Number.isFinite(crossPageClock.inferenceWindowStartEpochMs)
      && Number.isFinite(crossPageClock.inferenceWindowEndEpochMs)
      && crossPageClock.inferenceWindowEndEpochMs <= crossPageClock.inferenceWindowStartEpochMs) {
      errors.push('backgroundHeartbeat.crossPageClock inference window epoch bounds must be ordered');
    }
    const traceClock = heartbeat.eventTrace?.clock;
    if (!isObject(traceClock)
      || traceClock.schema !== CROSS_PAGE_CLOCK_SCHEMA
      || traceClock.timeOriginEpochMs !== crossPageClock.timeOriginEpochMs) {
      errors.push('backgroundHeartbeat.eventTrace.clock must match backgroundHeartbeat.crossPageClock time origin');
    }
  }

  if (!isObject(heartbeat.inferenceWindow)) {
    errors.push('backgroundHeartbeat.inferenceWindow must be an object');
  } else {
    requireString(errors, heartbeat.inferenceWindow.runId, 'backgroundHeartbeat.inferenceWindow.runId');
    for (const field of ['startMs', 'endMs', 'durationMs']) {
      if (!isFiniteNonNegative(heartbeat.inferenceWindow[field])) {
        errors.push(`backgroundHeartbeat.inferenceWindow.${field} must be a finite non-negative number`);
      }
    }
    if (Number.isFinite(heartbeat.inferenceWindow.startMs) && Number.isFinite(heartbeat.inferenceWindow.endMs)) {
      if (heartbeat.inferenceWindow.endMs <= heartbeat.inferenceWindow.startMs) {
        errors.push('backgroundHeartbeat.inferenceWindow.endMs must be greater than startMs');
      }
      if (Number.isFinite(heartbeat.inferenceWindow.durationMs)
        && Math.abs((heartbeat.inferenceWindow.endMs - heartbeat.inferenceWindow.startMs) - heartbeat.inferenceWindow.durationMs) > 1) {
        errors.push('backgroundHeartbeat.inferenceWindow.durationMs must match endMs - startMs');
      }
    }
  }

  if (isObject(crossPageClock) && isObject(heartbeat.inferenceWindow)) {
    if (crossPageClock.runId !== heartbeat.inferenceWindow.runId) {
      errors.push('backgroundHeartbeat.crossPageClock.runId must match backgroundHeartbeat.inferenceWindow.runId');
    }
    const startEpochFromRelative = crossPageClock.timeOriginEpochMs + heartbeat.inferenceWindow.startMs;
    const endEpochFromRelative = crossPageClock.timeOriginEpochMs + heartbeat.inferenceWindow.endMs;
    if (Number.isFinite(startEpochFromRelative)
      && Math.abs(startEpochFromRelative - crossPageClock.inferenceWindowStartEpochMs) > 1) {
      errors.push('backgroundHeartbeat.crossPageClock inferenceWindowStartEpochMs must use the declared time origin');
    }
    if (Number.isFinite(endEpochFromRelative)
      && Math.abs(endEpochFromRelative - crossPageClock.inferenceWindowEndEpochMs) > 1) {
      errors.push('backgroundHeartbeat.crossPageClock inferenceWindowEndEpochMs must use the declared time origin');
    }
  }

  const rafIntervalTrace = heartbeat.rafIntervalTrace;
  if (!isObject(rafIntervalTrace)) {
    errors.push('backgroundHeartbeat.rafIntervalTrace must be an object with uncapped raw rAF intervals');
  } else {
    if (rafIntervalTrace.schema !== RAF_INTERVAL_TRACE_SCHEMA) {
      errors.push(`backgroundHeartbeat.rafIntervalTrace.schema must be ${RAF_INTERVAL_TRACE_SCHEMA}`);
    }
    if (rafIntervalTrace.timingAuthority !== 'browser-request-animation-frame-performance-now') {
      errors.push('backgroundHeartbeat.rafIntervalTrace.timingAuthority must identify browser requestAnimationFrame performance.now timing');
    }
    if (rafIntervalTrace.uncapped !== true) {
      errors.push('backgroundHeartbeat.rafIntervalTrace.uncapped must be true');
    }
    requireString(errors, rafIntervalTrace.runId, 'backgroundHeartbeat.rafIntervalTrace.runId');
    if (!Array.isArray(rafIntervalTrace.intervals) || rafIntervalTrace.intervals.length === 0) {
      errors.push('backgroundHeartbeat.rafIntervalTrace.intervals must be a non-empty uncapped array');
    } else {
      if (rafIntervalTrace.count !== rafIntervalTrace.intervals.length) {
        errors.push('backgroundHeartbeat.rafIntervalTrace.count must match intervals length');
      }
      if (rafIntervalTrace.count !== heartbeat.responsiveness?.rafFrames) {
        errors.push('backgroundHeartbeat.rafIntervalTrace.count must match responsiveness.rafFrames; clipped retention is invalid');
      }
      const durations = [];
      let previousEndMs = null;
      for (const [index, interval] of rafIntervalTrace.intervals.entries()) {
        const path = `backgroundHeartbeat.rafIntervalTrace.intervals[${index}]`;
        if (!isObject(interval)) {
          errors.push(`${path} must be an object`);
          continue;
        }
        for (const field of ['startMs', 'endMs', 'durationMs', 'startEpochMs', 'endEpochMs']) {
          if (!isFiniteNonNegative(interval[field])) errors.push(`${path}.${field} must be a finite non-negative number`);
        }
        if (Number.isFinite(interval.startMs) && Number.isFinite(interval.endMs) && interval.endMs < interval.startMs) {
          errors.push(`${path}.endMs must be >= startMs`);
        }
        if (Number.isFinite(interval.startMs) && Number.isFinite(interval.endMs) && Number.isFinite(interval.durationMs)
          && Math.abs((interval.endMs - interval.startMs) - interval.durationMs) > 1) {
          errors.push(`${path}.durationMs must match endMs - startMs`);
        }
        if (Number.isFinite(previousEndMs) && Number.isFinite(interval.startMs) && Math.abs(previousEndMs - interval.startMs) > 1) {
          errors.push(`${path}.startMs must continue the uncapped preceding rAF interval`);
        }
        previousEndMs = interval.endMs;
        if (isObject(heartbeat.inferenceWindow)
          && Number.isFinite(interval.startMs)
          && Number.isFinite(interval.endMs)
          && (interval.startMs < heartbeat.inferenceWindow.startMs || interval.endMs > heartbeat.inferenceWindow.endMs)) {
          errors.push(`${path} must fall inside backgroundHeartbeat.inferenceWindow`);
        }
        if (isObject(crossPageClock)
          && Number.isFinite(interval.startEpochMs)
          && Number.isFinite(interval.endEpochMs)
          && (Math.abs((crossPageClock.timeOriginEpochMs + interval.startMs) - interval.startEpochMs) > 1
            || Math.abs((crossPageClock.timeOriginEpochMs + interval.endMs) - interval.endEpochMs) > 1)) {
          errors.push(`${path} epoch bounds must use backgroundHeartbeat.crossPageClock.timeOriginEpochMs`);
        }
        if (Number.isFinite(interval.durationMs)) durations.push(interval.durationMs);
      }
      const firstInterval = rafIntervalTrace.intervals[0];
      const lastInterval = rafIntervalTrace.intervals[rafIntervalTrace.intervals.length - 1];
      if (isObject(heartbeat.inferenceWindow)
        && Number.isFinite(firstInterval?.startMs)
        && Math.abs(firstInterval.startMs - heartbeat.inferenceWindow.startMs) > 0.001) {
        errors.push('backgroundHeartbeat.rafIntervalTrace first interval must start at the inferenceWindow boundary');
      }
      if (isObject(heartbeat.inferenceWindow)
        && Number.isFinite(lastInterval?.endMs)
        && Math.abs(lastInterval.endMs - heartbeat.inferenceWindow.endMs) > 0.001) {
        errors.push('backgroundHeartbeat.rafIntervalTrace last interval must end at the inferenceWindow boundary');
      }
      durations.sort((a, b) => a - b);
      const p95Index = durations.length ? Math.min(durations.length - 1, Math.floor(durations.length * 0.95)) : 0;
      const derived = {
        maxFrameGapMs: durations.length ? durations[durations.length - 1] : 0,
        p95FrameGapMs: durations.length ? durations[p95Index] : 0,
        longFrameCount: durations.filter(durationMs => durationMs > 50).length,
      };
      for (const [field, value] of Object.entries(derived)) {
        if (Number.isFinite(heartbeat.responsiveness?.[field]) && Math.abs(heartbeat.responsiveness[field] - value) > 1) {
          errors.push(`backgroundHeartbeat.responsiveness.${field} must be derived from rafIntervalTrace.intervals`);
        }
      }
    }
    if (isObject(crossPageClock)) {
      if (rafIntervalTrace.runId !== crossPageClock.runId) {
        errors.push('backgroundHeartbeat.rafIntervalTrace.runId must match backgroundHeartbeat.crossPageClock.runId');
      }
      if (rafIntervalTrace.timeOriginEpochMs !== crossPageClock.timeOriginEpochMs) {
        errors.push('backgroundHeartbeat.rafIntervalTrace.timeOriginEpochMs must match backgroundHeartbeat.crossPageClock.timeOriginEpochMs');
      }
    }
  }

  const gpuDutyIntervals = heartbeat.gpuDutyIntervals;
  if (!isObject(gpuDutyIntervals)) {
    errors.push('backgroundHeartbeat.gpuDutyIntervals must be an object');
  } else {
    if (gpuDutyIntervals.schema !== GPU_DUTY_INTERVALS_SCHEMA) {
      errors.push(`backgroundHeartbeat.gpuDutyIntervals.schema must be ${GPU_DUTY_INTERVALS_SCHEMA}`);
    }
    if (gpuDutyIntervals.timingAuthority !== 'queue-on-submitted-work-done-host-await-not-gpu-exclusive') {
      errors.push('backgroundHeartbeat.gpuDutyIntervals.timingAuthority must preserve host-await scope');
    }
    requireString(errors, gpuDutyIntervals.runId, 'backgroundHeartbeat.gpuDutyIntervals.runId');
    if (isObject(crossPageClock) && gpuDutyIntervals.runId !== crossPageClock.runId) {
      errors.push('backgroundHeartbeat.gpuDutyIntervals.runId must match backgroundHeartbeat.crossPageClock.runId');
    }
    if (!Array.isArray(gpuDutyIntervals.intervals)) {
      errors.push('backgroundHeartbeat.gpuDutyIntervals.intervals must be an array');
    } else {
      if (gpuDutyIntervals.count !== gpuDutyIntervals.intervals.length) {
        errors.push('backgroundHeartbeat.gpuDutyIntervals.count must match intervals length');
      }
      if (heartbeat.effectiveScheduler?.waitForSubmittedWorkDone === true && gpuDutyIntervals.intervals.length === 0) {
        errors.push('backgroundHeartbeat.gpuDutyIntervals must be non-empty when submitted-work waiting is effective');
      }
      const sourceDutyIntervals = createGpuDutyIntervals(
        Array.isArray(heartbeat.eventTrace?.events) ? heartbeat.eventTrace.events : [],
        heartbeat.inferenceWindow,
        gpuDutyIntervals.runId,
      );
      for (const failure of sourceDutyIntervals.pairingFailures || []) {
        errors.push(`backgroundHeartbeat retained queue-work endpoints cannot form a complete duty bijection: ${failure}`);
      }
      const sourceIntervalsByKey = new Map(sourceDutyIntervals.intervals.map(interval => [
        gpuDutyIntervalSourceKey(interval),
        interval,
      ]));
      const derivedSourceKeys = new Set();
      const dutyIds = new Set();
      for (const [index, interval] of gpuDutyIntervals.intervals.entries()) {
        const path = `backgroundHeartbeat.gpuDutyIntervals.intervals[${index}]`;
        if (!isObject(interval)) {
          errors.push(`${path} must be an object`);
          continue;
        }
        requireString(errors, interval.dutyId, `${path}.dutyId`);
        requireString(errors, interval.runId, `${path}.runId`);
        if (interval.runId !== gpuDutyIntervals.runId) {
          errors.push(`${path}.runId must match backgroundHeartbeat.gpuDutyIntervals.runId`);
        }
        if (isObject(crossPageClock) && interval.runId !== crossPageClock.runId) {
          errors.push(`${path}.runId must match backgroundHeartbeat.crossPageClock.runId`);
        }
        requireString(errors, interval.phase, `${path}.phase`);
        requireString(errors, interval.boundary, `${path}.boundary`);
        if (interval.kind !== 'submitted-work-drain-interval') {
          errors.push(`${path}.kind must be submitted-work-drain-interval`);
        }
        if (dutyIds.has(interval.dutyId)) errors.push(`${path}.dutyId must be unique`);
        dutyIds.add(interval.dutyId);
        const sourceKey = gpuDutyIntervalSourceKey(interval);
        const sourceInterval = sourceIntervalsByKey.get(sourceKey);
        if (derivedSourceKeys.has(sourceKey)) {
          errors.push(`${path} duplicates one retained queue-work endpoint pair`);
        }
        derivedSourceKeys.add(sourceKey);
        if (!Number.isInteger(interval.startEventSequence) || !Number.isInteger(interval.endEventSequence)) {
          errors.push(`${path} must identify its scheduler start and end event sequences`);
        }
        if (!sourceInterval) {
          errors.push(`${path} must resolve to exactly one eligible retained queue-work endpoint pair`);
        } else {
          for (const field of ['runId', 'dutyId', 'phase', 'boundary', 'kind', 'startEventSequence', 'endEventSequence']) {
            if (interval[field] !== sourceInterval[field]) {
              errors.push(`${path}.${field} must match its retained queue-work endpoint source`);
            }
          }
          for (const field of ['startMs', 'endMs', 'durationMs', 'startEpochMs', 'endEpochMs']) {
            if (!Number.isFinite(interval[field])
              || !Number.isFinite(sourceInterval[field])
              || Math.abs(interval[field] - sourceInterval[field]) > 1) {
              errors.push(`${path}.${field} must match its retained queue-work endpoint source`);
            }
          }
          for (const field of ['stage', 'step', 'role']) {
            if ((interval[field] ?? null) !== (sourceInterval[field] ?? null)) {
              errors.push(`${path}.${field} must match its retained queue-work endpoint source`);
            }
          }
        }
        for (const field of ['startMs', 'endMs', 'durationMs', 'startEpochMs', 'endEpochMs']) {
          if (!isFiniteNonNegative(interval[field])) errors.push(`${path}.${field} must be a finite non-negative number`);
        }
        if (Number.isFinite(interval.startMs) && Number.isFinite(interval.endMs) && interval.endMs < interval.startMs) {
          errors.push(`${path}.endMs must be >= startMs`);
        }
        if (Number.isFinite(interval.startEpochMs) && Number.isFinite(interval.endEpochMs) && interval.endEpochMs < interval.startEpochMs) {
          errors.push(`${path}.endEpochMs must be >= startEpochMs`);
        }
        if (Number.isFinite(interval.startMs) && Number.isFinite(interval.endMs) && Number.isFinite(interval.durationMs)
          && Math.abs((interval.endMs - interval.startMs) - interval.durationMs) > 1) {
          errors.push(`${path}.durationMs must match endMs - startMs`);
        }
        if (isObject(crossPageClock)
          && Number.isFinite(interval.startMs)
          && Number.isFinite(interval.endMs)
          && Number.isFinite(interval.startEpochMs)
          && Number.isFinite(interval.endEpochMs)) {
          if (Math.abs((crossPageClock.timeOriginEpochMs + interval.startMs) - interval.startEpochMs) > 1) {
            errors.push(`${path}.startEpochMs must use the declared time origin`);
          }
          if (Math.abs((crossPageClock.timeOriginEpochMs + interval.endMs) - interval.endEpochMs) > 1) {
            errors.push(`${path}.endEpochMs must use the declared time origin`);
          }
          if (interval.startEpochMs < crossPageClock.inferenceWindowStartEpochMs
            || interval.endEpochMs > crossPageClock.inferenceWindowEndEpochMs) {
            errors.push(`${path} must fall inside the cross-page inferenceWindow`);
          }
        }
      }
      for (const [sourceKey, sourceInterval] of sourceIntervalsByKey.entries()) {
        if (!derivedSourceKeys.has(sourceKey)) {
          errors.push(`backgroundHeartbeat.gpuDutyIntervals is missing the retained queue-work endpoint pair for duty ${sourceInterval.dutyId}`);
        }
      }
    }
    if (Array.isArray(gpuDutyIntervals.pairingFailures) && gpuDutyIntervals.pairingFailures.length > 0) {
      errors.push('backgroundHeartbeat.gpuDutyIntervals contains pairing failures');
    }
  }

  if (!Array.isArray(heartbeat.worstFrameGaps) || heartbeat.worstFrameGaps.length === 0) {
    errors.push('backgroundHeartbeat.worstFrameGaps must be a non-empty array');
    return;
  }
  const scopedMaxGapMs = Math.max(...heartbeat.worstFrameGaps.map(gap => gap?.durationMs).filter(Number.isFinite));
  if (Number.isFinite(scopedMaxGapMs)
    && Number.isFinite(heartbeat.responsiveness?.maxFrameGapMs)
    && Math.abs(scopedMaxGapMs - heartbeat.responsiveness.maxFrameGapMs) > 1) {
    errors.push('backgroundHeartbeat.responsiveness.maxFrameGapMs must match the largest scoped worstFrameGaps interval');
  }
  for (const [index, gap] of heartbeat.worstFrameGaps.entries()) {
    if (!isObject(gap)) {
      errors.push(`backgroundHeartbeat.worstFrameGaps[${index}] must be an object`);
      continue;
    }
    for (const field of ['startMs', 'endMs', 'durationMs']) {
      if (!isFiniteNonNegative(gap[field])) {
        errors.push(`backgroundHeartbeat.worstFrameGaps[${index}].${field} must be a finite non-negative number`);
      }
    }
    if (Number.isFinite(gap.endMs) && Number.isFinite(gap.startMs) && gap.endMs < gap.startMs) {
      errors.push(`backgroundHeartbeat.worstFrameGaps[${index}].endMs must be >= startMs`);
    }
    if (Number.isFinite(gap.endMs) && Number.isFinite(gap.startMs) && Number.isFinite(gap.durationMs)) {
      const intervalDurationMs = gap.endMs - gap.startMs;
      if (Math.abs(intervalDurationMs - gap.durationMs) > 1) {
        errors.push(`backgroundHeartbeat.worstFrameGaps[${index}].durationMs must match endMs - startMs`);
      }
    }
    if (isObject(heartbeat.inferenceWindow)
      && Number.isFinite(gap.startMs)
      && Number.isFinite(gap.endMs)
      && Number.isFinite(heartbeat.inferenceWindow.startMs)
      && Number.isFinite(heartbeat.inferenceWindow.endMs)
      && (gap.startMs < heartbeat.inferenceWindow.startMs || gap.endMs > heartbeat.inferenceWindow.endMs)) {
      errors.push(`backgroundHeartbeat.worstFrameGaps[${index}] must fall inside backgroundHeartbeat.inferenceWindow`);
    }
    requireString(errors, gap.overlapClassification, `backgroundHeartbeat.worstFrameGaps[${index}].overlapClassification`);
    if (!VALID_HEARTBEAT_OVERLAP_CLASSIFICATIONS.has(gap.overlapClassification)) {
      errors.push(`backgroundHeartbeat.worstFrameGaps[${index}].overlapClassification must be one of ${[...VALID_HEARTBEAT_OVERLAP_CLASSIFICATIONS].join(', ')}`);
    }
    if (!Array.isArray(gap.overlappedEvents)) {
      errors.push(`backgroundHeartbeat.worstFrameGaps[${index}].overlappedEvents must be an array`);
    } else if (gap.overlapClassification === 'scheduler-event-overlap' && gap.overlappedEvents.length === 0) {
      errors.push(`backgroundHeartbeat.worstFrameGaps[${index}].overlappedEvents must be non-empty for scheduler-event-overlap`);
    } else if (gap.overlapClassification === 'uninstrumented-gap' && gap.overlappedEvents.length !== 0) {
      errors.push(`backgroundHeartbeat.worstFrameGaps[${index}].overlappedEvents must be empty for uninstrumented-gap`);
    } else if (gap.overlapClassification === 'scheduler-event-overlap') {
      for (const [eventIndex, event] of gap.overlappedEvents.entries()) {
        if (!isObject(event)) {
          errors.push(`backgroundHeartbeat.worstFrameGaps[${index}].overlappedEvents[${eventIndex}] must be an object`);
          continue;
        }
        const hasPointTiming = Number.isFinite(event.tMs);
        const hasIntervalTiming = Number.isFinite(event.intervalStartMs) && Number.isFinite(event.intervalEndMs);
        const hasAnyIntervalField = event.intervalStartMs !== undefined
          || event.intervalEndMs !== undefined
          || event.durationMs !== undefined;
        if (!hasPointTiming) {
          errors.push(`backgroundHeartbeat.worstFrameGaps[${index}].overlappedEvents[${eventIndex}].tMs must be finite`);
        }
        if (hasAnyIntervalField && !hasIntervalTiming) {
          errors.push(`backgroundHeartbeat.worstFrameGaps[${index}].overlappedEvents[${eventIndex}] intervalStartMs and intervalEndMs must both be finite`);
        } else if (hasIntervalTiming) {
          if (event.intervalEndMs < event.intervalStartMs) {
            errors.push(`backgroundHeartbeat.worstFrameGaps[${index}].overlappedEvents[${eventIndex}].intervalEndMs must be >= intervalStartMs`);
          }
          if (!Number.isFinite(event.durationMs) || event.durationMs < 0) {
            errors.push(`backgroundHeartbeat.worstFrameGaps[${index}].overlappedEvents[${eventIndex}].durationMs must be finite and non-negative for interval evidence`);
          } else if (Math.abs((event.intervalEndMs - event.intervalStartMs) - event.durationMs) > 1) {
            errors.push(`backgroundHeartbeat.worstFrameGaps[${index}].overlappedEvents[${eventIndex}].durationMs must match intervalEndMs - intervalStartMs`);
          }
          if (Number.isFinite(gap.startMs) && Number.isFinite(gap.endMs)
            && (event.intervalStartMs > gap.endMs || event.intervalEndMs < gap.startMs)) {
            errors.push(`backgroundHeartbeat.worstFrameGaps[${index}].overlappedEvents[${eventIndex}] interval must overlap the gap interval`);
          }
        } else if (hasPointTiming && Number.isFinite(gap.startMs) && Number.isFinite(gap.endMs) && (event.tMs < gap.startMs || event.tMs > gap.endMs)) {
          errors.push(`backgroundHeartbeat.worstFrameGaps[${index}].overlappedEvents[${eventIndex}].tMs must fall inside the gap interval`);
        }
        if (!isNonEmptyString(event.phase) && !isNonEmptyString(event.boundary)) {
          errors.push(`backgroundHeartbeat.worstFrameGaps[${index}].overlappedEvents[${eventIndex}] must include phase or boundary`);
        }
      }
    }
  }
}

function validateRoute(errors, route) {
  if (!isObject(route)) {
    errors.push('route must be an object');
    return;
  }
  if (route.requestedRouteId !== SHARP_ROUTE_ID) {
    errors.push(`route.requestedRouteId must be ${SHARP_ROUTE_ID}`);
  }
  if (route.effectiveRouteId !== SHARP_ROUTE_ID) {
    errors.push(`route.effectiveRouteId must be ${SHARP_ROUTE_ID}`);
  }
  if (!isObject(route.receipt)) {
    errors.push('route.receipt must be an object');
    return;
  }

  const evidence = classifyWebGpuRouteReceiptEvidence(route.receipt, {
    expectedRouteId: SHARP_ROUTE_ID,
  });
  if (evidence.authoritative !== true) {
    errors.push(`route receipt must classify as authoritative-live-webgpu; got ${evidence.classification}: ${evidence.reasons.join('; ')}`);
  }
  if (!isObject(route.evidence)) {
    errors.push('route.evidence must be an object');
  } else {
    if (route.evidence.authoritative !== evidence.authoritative) {
      errors.push('route.evidence.authoritative must match computed route receipt evidence');
    }
    if (route.evidence.classification !== evidence.classification) {
      errors.push('route.evidence.classification must match computed route receipt evidence');
    }
    if (route.evidence.effectiveRouteId !== evidence.effectiveRouteId) {
      errors.push('route.evidence.effectiveRouteId must match computed route receipt evidence');
    }
  }
}

function validateInference(errors, inference, route) {
  if (!isObject(inference)) {
    errors.push('inference must be an object');
    return;
  }
  if (inference.ok !== true) errors.push('inference.ok must be true');
  if (inference.valid !== 'OK') errors.push('inference.valid must be OK');
  requireFinitePositive(errors, inference.timeMs, 'inference.timeMs');
  requireString(errors, inference.model, 'inference.model');
  requireString(errors, inference.weights, 'inference.weights');
  if (!isObject(inference.outputs)) {
    errors.push('inference.outputs must be an object');
    return;
  }
  requireFinitePositive(errors, inference.outputs.numGaussians, 'inference.outputs.numGaussians');
  if (inference.outputs.plyAvailable !== true) errors.push('inference.outputs.plyAvailable must be true');
  requireFinitePositive(errors, inference.outputs.plyByteLength, 'inference.outputs.plyByteLength');
  if (!/^[a-f0-9]{64}$/i.test(inference.outputs.plySha256 || '')) {
    errors.push('inference.outputs.plySha256 must be a 64-character SHA-256 digest');
  }
  if (inference.outputs.completeness !== 'complete') {
    errors.push('inference.outputs.completeness must be complete');
  }
  const receiptSplat = Array.isArray(route?.receipt?.outputs)
    ? route.receipt.outputs.find(output => output?.role === 'splat-candidate')
    : null;
  if (!receiptSplat) {
    errors.push('route receipt must contain the terminal splat-candidate output');
  } else {
    if (inference.outputs.plySha256 !== receiptSplat.sha256) {
      errors.push('inference.outputs.plySha256 must match the authoritative route receipt splat-candidate SHA-256');
    }
    if (Number.isFinite(inference.outputs.numGaussians) && receiptSplat.shape?.[0] !== inference.outputs.numGaussians) {
      errors.push('inference.outputs.numGaussians must match the authoritative route receipt splat-candidate shape');
    }
  }
}

function validateEpisodeBinding(errors, report) {
  const receipt = report.route?.receipt;
  const payload = receipt?.metadataPayload;
  const metadataOutput = Array.isArray(receipt?.outputs)
    ? receipt.outputs.find(output => output?.role === 'sharp-webgpu-metadata')
    : null;
  if (!isObject(payload)) {
    errors.push('route.receipt.metadataPayload must be an authenticated episode envelope');
    return;
  }
  if (payload.schema !== 'sharp.webgpu-route-metadata.v0') {
    errors.push('route.receipt.metadataPayload.schema must be sharp.webgpu-route-metadata.v0');
  }
  if (!metadataOutput || !/^[a-f0-9]{64}$/i.test(metadataOutput.sha256 || '')) {
    errors.push('route receipt must contain a SHA-256-authenticated sharp-webgpu-metadata output');
  } else if (metadataOutput.sha256 !== sha256Json(payload)) {
    errors.push('route receipt sharp-webgpu-metadata SHA-256 must authenticate metadataPayload');
  }

  requireString(errors, payload.episodeId, 'route.receipt.metadataPayload.episodeId');
  for (const [path, value] of [
    ['runId', report.runId],
    ['scheduler.runId', report.scheduler?.runId],
    ['backgroundHeartbeat.crossPageClock.runId', report.backgroundHeartbeat?.crossPageClock?.runId],
    ['backgroundHeartbeat.inferenceWindow.runId', report.backgroundHeartbeat?.inferenceWindow?.runId],
    ['backgroundHeartbeat.eventTrace run id', report.backgroundHeartbeat?.eventTrace?.events?.[0]?.runId],
    ['metadataPayload.schedulerTrace.runId', payload.schedulerTrace?.runId],
  ]) {
    if (value !== payload.episodeId) errors.push(`${path} must match route.receipt.metadataPayload.episodeId`);
  }

  const terminal = payload.terminalOutput;
  if (!isObject(terminal)) {
    errors.push('route.receipt.metadataPayload.terminalOutput must be an object');
  } else {
    for (const field of ['plySha256', 'plyByteLength', 'numGaussians', 'completeness']) {
      if (terminal[field] !== report.inference?.outputs?.[field]) {
        errors.push(`inference.outputs.${field} must match authenticated metadata terminal output`);
      }
    }
  }

  const authenticatedSequence = payload.schedulerTrace?.eventSequence;
  const retainedSequence = report.backgroundHeartbeat?.eventTrace?.sequenceEnvelope;
  if (!isObject(authenticatedSequence)
    || !isObject(retainedSequence)
    || JSON.stringify(authenticatedSequence) !== JSON.stringify(retainedSequence)) {
    errors.push('backgroundHeartbeat.eventTrace.sequenceEnvelope must match authenticated metadata scheduler trace');
  }
}

function validateResponsiveness(errors, warnings, responsiveness) {
  if (!isObject(responsiveness)) {
    errors.push('responsiveness must be an object');
    return;
  }
  for (const field of ['rafFrames', 'maxFrameGapMs', 'p95FrameGapMs', 'longFrameCount']) {
    if (!isFiniteNonNegative(responsiveness[field])) {
      errors.push(`responsiveness.${field} must be a finite non-negative number`);
    }
  }
  if (responsiveness.rafFrames === 0) {
    errors.push('responsiveness.rafFrames must be positive; frame-tail evidence was not observed');
  }
}

function validateContender(errors, report) {
  const { contender, mode } = report;
  if (!isObject(contender)) {
    errors.push('contender must be an object');
    return;
  }
  if (typeof contender.enabled !== 'boolean') errors.push('contender.enabled must be boolean');
  for (const field of ['submitted', 'completed']) {
    if (!isFiniteNonNegative(contender[field])) {
      errors.push(`contender.${field} must be a finite non-negative number`);
    }
  }
  if (!isObject(contender.inferenceWindow)) {
    errors.push('contender.inferenceWindow must be an object');
  } else {
    for (const field of ['submittedAtStart', 'completedAtStart', 'submittedAtEnd', 'completedAtEnd', 'submittedDelta', 'completedDelta']) {
      if (!isFiniteNonNegative(contender.inferenceWindow[field])) {
        errors.push(`contender.inferenceWindow.${field} must be a finite non-negative number`);
      }
    }
    if (Number.isFinite(contender.inferenceWindow.submittedAtStart) && Number.isFinite(contender.inferenceWindow.submittedAtEnd)) {
      const submittedDelta = contender.inferenceWindow.submittedAtEnd - contender.inferenceWindow.submittedAtStart;
      if (submittedDelta !== contender.inferenceWindow.submittedDelta) {
        errors.push('contender.inferenceWindow.submittedDelta must match submittedAtEnd - submittedAtStart');
      }
    }
    if (Number.isFinite(contender.inferenceWindow.completedAtStart) && Number.isFinite(contender.inferenceWindow.completedAtEnd)) {
      const completedDelta = contender.inferenceWindow.completedAtEnd - contender.inferenceWindow.completedAtStart;
      if (completedDelta !== contender.inferenceWindow.completedDelta) {
        errors.push('contender.inferenceWindow.completedDelta must match completedAtEnd - completedAtStart');
      }
    }
  }
  if (typeof contender.progressDuringInference !== 'boolean') {
    errors.push('contender.progressDuringInference must be boolean');
  }
  if (!Array.isArray(contender.errors)) errors.push('contender.errors must be an array');
  if (mode !== 'baseline') {
    if (contender.enabled !== true) errors.push('contender.enabled must be true for contended/cooperative runs');
    if (contender.submitted <= 0) errors.push('contender.submitted must be positive for contended/cooperative runs');
    if (contender.completed <= 0 || contender.progressDuringInference !== true) {
      errors.push('contender progress must be observed during contended/cooperative inference');
    }
    if (!isObject(contender.inferenceWindow) || contender.inferenceWindow.submittedDelta <= 0 || contender.inferenceWindow.completedDelta <= 0) {
      errors.push('contender progress must be observed inside the measured inference window');
    }
  }
}

export function validateSharpContentionWitnessReport(report) {
  const errors = [];
  const warnings = [];

  if (!isObject(report)) return { ok: false, errors: ['report must be an object'], warnings };
  if (report.schema !== SHARP_CONTENTION_WITNESS_SCHEMA) {
    errors.push(`schema must be ${SHARP_CONTENTION_WITNESS_SCHEMA}`);
  }
  requireString(errors, report.runId, 'runId');
  requireString(errors, report.createdAt, 'createdAt');
  if (!VALID_MODES.has(report.mode)) errors.push(`mode must be one of ${[...VALID_MODES].join(', ')}`);
  validateRoute(errors, report.route);
  if (!isObject(report.input)) {
    errors.push('input must be an object');
  } else {
    requireString(errors, report.input.source, 'input.source');
    requireString(errors, report.input.artifactId, 'input.artifactId');
  }
  validateInference(errors, report.inference, report.route);
  validateResponsiveness(errors, warnings, report.responsiveness);
  validateBackgroundHeartbeat(errors, report.backgroundHeartbeat);
  validateEpisodeBinding(errors, report);
  if (isObject(report.responsiveness) && isObject(report.backgroundHeartbeat?.responsiveness)) {
    for (const field of ['rafFrames', 'maxFrameGapMs', 'p95FrameGapMs', 'longFrameCount']) {
      const routeValue = report.responsiveness[field];
      const heartbeatValue = report.backgroundHeartbeat.responsiveness[field];
      if (Number.isFinite(routeValue) && Number.isFinite(heartbeatValue) && Math.abs(routeValue - heartbeatValue) > 1e-6) {
        errors.push(`responsiveness.${field} must match backgroundHeartbeat.responsiveness.${field}`);
      }
    }
  }
  validateContender(errors, report);
  if (!isObject(report.scheduler)) {
    errors.push('scheduler must be an object');
  } else {
    requireString(errors, report.scheduler.runId, 'scheduler.runId');
    requireString(errors, report.scheduler.mode, 'scheduler.mode');
    requireString(errors, report.scheduler.verificationState, 'scheduler.verificationState');
  }

  return { ok: errors.length === 0, errors, warnings };
}

function escapeXml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

function frameGapColor(durationMs) {
  if (durationMs > 100) return '#b42318';
  if (durationMs > 50) return '#d97706';
  if (durationMs > 16.7) return '#ca8a04';
  return '#15803d';
}

export function renderSharpContentionTraceSvg(report) {
  const validation = validateSharpContentionWitnessReport(report);
  if (!validation.ok) {
    throw new Error(`cannot render invalid SHARP contention report: ${validation.errors.join('; ')}`);
  }

  const heartbeat = report.backgroundHeartbeat;
  const window = heartbeat.inferenceWindow;
  const frameIntervals = heartbeat.rafIntervalTrace.intervals;
  const schedulerEvents = heartbeat.eventTrace.events;
  const splat = report.route.receipt.outputs.find(output => output?.role === 'splat-candidate');
  const width = 1440;
  const left = 80;
  const right = 40;
  const plotWidth = width - left - right;
  const windowDuration = window.endMs - window.startMs;
  const frameTop = 170;
  const frameHeight = 160;
  const maxGap = Math.max(16.7, ...frameIntervals.map(interval => interval.durationMs));
  const xFor = tMs => left + ((tMs - window.startMs) / windowDuration) * plotWidth;
  const boundaryMap = new Map();
  for (const event of schedulerEvents) {
    const name = event.boundary || event.phase || 'unknown';
    const startMs = Number.isFinite(event.intervalStartMs) ? event.intervalStartMs : event.tMs;
    const endMs = Number.isFinite(event.intervalEndMs) ? event.intervalEndMs : event.tMs;
    const summary = boundaryMap.get(name) || {
      name,
      count: 0,
      firstSequence: event.sequence,
      lastSequence: event.sequence,
      firstEpochMs: event.epochMs,
      lastEpochMs: event.epochMs,
      startMs,
      endMs,
    };
    summary.count += 1;
    summary.firstSequence = Math.min(summary.firstSequence, event.sequence);
    summary.lastSequence = Math.max(summary.lastSequence, event.sequence);
    summary.firstEpochMs = Math.min(summary.firstEpochMs, event.epochMs);
    summary.lastEpochMs = Math.max(summary.lastEpochMs, event.epochMs);
    summary.startMs = Math.min(summary.startMs, startMs);
    summary.endMs = Math.max(summary.endMs, endMs);
    boundaryMap.set(name, summary);
  }
  const boundarySummaries = [...boundaryMap.values()].sort((a, b) => a.firstSequence - b.firstSequence);
  const boundaryColumns = 3;
  const boundaryRows = Math.ceil(boundarySummaries.length / boundaryColumns);
  const boundaryLegendTop = frameTop + frameHeight + 46;
  const height = Math.max(430, boundaryLegendTop + boundaryRows * 20 + 24);

  const frameBars = frameIntervals.map((interval, index) => {
    const x = xFor(interval.startMs);
    const intervalWidth = Math.max(1, xFor(interval.endMs) - x);
    const barHeight = Math.max(1, (interval.durationMs / maxGap) * frameHeight);
    const y = frameTop + frameHeight - barHeight;
    return `<rect data-raf-interval="${index}" x="${x.toFixed(3)}" y="${y.toFixed(3)}" width="${intervalWidth.toFixed(3)}" height="${barHeight.toFixed(3)}" fill="${frameGapColor(interval.durationMs)}"><title>${escapeXml(`${interval.durationMs.toFixed(3)}ms at ${interval.startEpochMs.toFixed(3)}`)}</title></rect>`;
  }).join('');

  const boundaryBands = boundarySummaries.map((summary, index) => {
    const startMs = Math.max(window.startMs, summary.startMs);
    const endMs = Math.min(window.endMs, summary.endMs);
    if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs < startMs) return '';
    const x = xFor(startMs);
    const bandWidth = Math.max(1, xFor(endMs) - x);
    const opacity = 0.08 + (index % 3) * 0.035;
    return `<rect data-scheduler-boundary="${escapeXml(summary.name)}" data-event-count="${summary.count}" data-first-sequence="${summary.firstSequence}" data-last-sequence="${summary.lastSequence}" x="${x.toFixed(3)}" y="${frameTop}" width="${bandWidth.toFixed(3)}" height="${frameHeight}" fill="#2563eb" opacity="${opacity.toFixed(3)}"><title>${escapeXml(`${summary.name}: ${summary.count} events, sequence ${summary.firstSequence}-${summary.lastSequence}, epoch ${summary.firstEpochMs}-${summary.lastEpochMs}`)}</title></rect>`;
  }).join('');

  const boundaryLabels = boundarySummaries.map((summary, index) => {
    const column = index % boundaryColumns;
    const row = Math.floor(index / boundaryColumns);
    const x = left + column * (plotWidth / boundaryColumns);
    const y = boundaryLegendTop + row * 20;
    return `<text x="${x.toFixed(3)}" y="${y}" font-family="ui-monospace, monospace" font-size="11" fill="#1d4ed8">${escapeXml(`${summary.name} (${summary.count}; seq ${summary.firstSequence}-${summary.lastSequence})`)}</text>`;
  }).join('');

  const ticks = [0, 0.25, 0.5, 0.75, 1].map(fraction => {
    const x = left + fraction * plotWidth;
    const elapsedSeconds = (fraction * windowDuration) / 1000;
    return `<line x1="${x}" y1="${frameTop}" x2="${x}" y2="${frameTop + frameHeight}" stroke="#d1d5db" stroke-width="1"/><text x="${x}" y="${frameTop + frameHeight + 18}" text-anchor="middle" font-family="ui-monospace, monospace" font-size="11" fill="#4b5563">${elapsedSeconds.toFixed(1)}s</text>`;
  }).join('');

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc" data-source-run-id="${escapeXml(report.runId)}">
  <title id="title">SHARP composed-world contention trace</title>
  <desc id="desc">Uncapped browser animation-frame intervals and exact per-boundary aggregates of the uncapped SHARP scheduler event trace on one validated browser epoch-monotonic clock.</desc>
  <rect width="${width}" height="${height}" fill="#ffffff"/>
  <text x="${left}" y="30" font-family="system-ui, sans-serif" font-size="20" font-weight="700" fill="#111827">SHARP composed-world evidence trace</text>
  <text x="${left}" y="54" font-family="ui-monospace, monospace" font-size="12" fill="#374151">run ${escapeXml(report.runId)} | route ${escapeXml(report.route.effectiveRouteId)} | mode ${escapeXml(report.mode)}</text>
  <text x="${left}" y="74" font-family="ui-monospace, monospace" font-size="12" fill="#374151">splat-candidate ${escapeXml(splat.sha256)}</text>
  <text x="${left}" y="94" font-family="ui-monospace, monospace" font-size="12" fill="#374151">${report.inference.outputs.plyByteLength.toLocaleString('en-US')} bytes | ${report.inference.outputs.numGaussians.toLocaleString('en-US')} Gaussians | output complete</text>
  <text x="${left}" y="114" font-family="ui-monospace, monospace" font-size="12" fill="#374151">clock origin ${heartbeat.crossPageClock.timeOriginEpochMs.toFixed(3)} | ${frameIntervals.length} uncapped rAF intervals | ${schedulerEvents.length} uncapped scheduler events</text>
  <rect x="${left}" y="132" width="12" height="12" fill="#15803d"/><text x="${left + 18}" y="142" font-family="system-ui, sans-serif" font-size="11" fill="#374151">at or below 16.7ms</text>
  <rect x="${left + 142}" y="132" width="12" height="12" fill="#ca8a04"/><text x="${left + 160}" y="142" font-family="system-ui, sans-serif" font-size="11" fill="#374151">16.7-50ms</text>
  <rect x="${left + 252}" y="132" width="12" height="12" fill="#d97706"/><text x="${left + 270}" y="142" font-family="system-ui, sans-serif" font-size="11" fill="#374151">50-100ms</text>
  <rect x="${left + 362}" y="132" width="12" height="12" fill="#b42318"/><text x="${left + 380}" y="142" font-family="system-ui, sans-serif" font-size="11" fill="#374151">over 100ms</text>
  <line x1="${left}" y1="${frameTop + frameHeight}" x2="${left + plotWidth}" y2="${frameTop + frameHeight}" stroke="#111827" stroke-width="1"/>
  ${ticks}
  ${boundaryBands}
  ${frameBars}
  <text x="18" y="${frameTop + frameHeight / 2}" transform="rotate(-90 18 ${frameTop + frameHeight / 2})" text-anchor="middle" font-family="system-ui, sans-serif" font-size="12" fill="#374151">rAF interval (max ${maxGap.toFixed(1)}ms)</text>
  ${boundaryLabels}
</svg>`;
}
