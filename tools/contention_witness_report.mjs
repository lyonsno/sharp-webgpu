export const SHARP_CONTENTION_WITNESS_SCHEMA = 'sharp.webgpu-contention-witness.v0';
export const SHARP_BACKGROUND_HEARTBEAT_SCHEMA = 'sharp-webgpu.background-heartbeat.v0';
export const SHARP_ROUTE_ID = 'sharp.image-to-splat.webgpu-local.v0';

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

function normalizeInferenceWindow(window) {
  if (!isObject(window) || !Number.isFinite(window.startMs) || !Number.isFinite(window.endMs) || window.endMs <= window.startMs) {
    return null;
  }
  const startMs = Number(window.startMs.toFixed(3));
  const endMs = Number(window.endMs.toFixed(3));
  return {
    startMs,
    endMs,
    durationMs: Number((endMs - startMs).toFixed(3)),
  };
}

export function createSharpBackgroundHeartbeatReport({ scheduler = {}, probe = {}, responsiveness = null } = {}) {
  const events = schedulerEvents(scheduler);
  const rawInferenceWindow = probe.inferenceWindow || probe.contender?.inferenceWindow;
  const inferenceWindow = normalizeInferenceWindow(rawInferenceWindow);
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
      eventCount: events.length,
      boundaries: summarizeBoundaries(events),
    },
    inferenceWindow,
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
  }

  if (!isObject(heartbeat.inferenceWindow)) {
    errors.push('backgroundHeartbeat.inferenceWindow must be an object');
  } else {
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

function validateInference(errors, inference) {
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
  validateInference(errors, report.inference);
  validateResponsiveness(errors, warnings, report.responsiveness);
  validateBackgroundHeartbeat(errors, report.backgroundHeartbeat);
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
    requireString(errors, report.scheduler.mode, 'scheduler.mode');
    requireString(errors, report.scheduler.verificationState, 'scheduler.verificationState');
  }

  return { ok: errors.length === 0, errors, warnings };
}
