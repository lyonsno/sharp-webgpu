const DEFAULT_SCHEDULER = {
  mode: 'default',
  spnPatchChunkSize: 4,
  spnFusionChunkItems: 0,
  yieldMs: 0,
  waitForSubmittedWorkDone: false,
  gaussianPhaseYieldMs: 0,
  vitBlockChunkSize: null,
  vitMicroduty: false,
  vitMicrodutyMode: 'two-stage',
  vitLinearTileItems: 0,
  vitAttentionTileItems: 0,
  vitSoftmaxTileRows: 0,
  vitNormTileRows: 0,
  decoderKernelChunkItems: 0,
  routeTailYieldMs: 0,
  cpuChunkItems: 0,
  plyAssemblyMode: 'main-thread',
  retirePostInferenceBuffers: false,
};

const SUPPORTED_FIELDS = new Set([
  'mode',
  'spnPatchChunkSize',
  'spnFusionChunkItems',
  'yieldMs',
  'waitForSubmittedWorkDone',
  'gaussianPhaseYieldMs',
  'vitBlockChunkSize',
  'vitMicroduty',
  'vitMicrodutyMode',
  'vitLinearTileItems',
  'vitAttentionTileItems',
  'vitSoftmaxTileRows',
  'vitNormTileRows',
  'decoderKernelChunkItems',
  'routeTailYieldMs',
  'cpuChunkItems',
  'plyAssemblyMode',
  'retirePostInferenceBuffers',
]);

const INT_FIELDS = new Set(['spnPatchChunkSize', 'spnFusionChunkItems', 'yieldMs', 'gaussianPhaseYieldMs', 'vitBlockChunkSize', 'vitLinearTileItems', 'vitAttentionTileItems', 'vitSoftmaxTileRows', 'vitNormTileRows', 'decoderKernelChunkItems', 'routeTailYieldMs', 'cpuChunkItems']);
const VIT_MICRODUTY_MODES = new Set([
  'two-stage',
  'split-attention',
  'split-mlp',
  'four-stage',
  'dispatch-major',
]);
const PLY_ASSEMBLY_MODES = new Set(['main-thread', 'worker']);
const EVENT_TRACE_SCHEMA = 'kaminos.webgpu-scheduler-event-trace.v0';
const EVENT_CLOCK_SCHEMA = 'kaminos.browser-epoch-monotonic-clock.v0';
const LIVE_SCHEDULER_CONTROLS = {
  'spn-patch-chunk': {
    controlId: 'spnPatch',
    legacyField: 'spnPatchChunkSize',
    unit: 'patch',
  },
  'vit-block-chunk': {
    controlId: 'vitBlock',
    legacyField: 'vitBlockChunkSize',
    unit: 'vit-block',
  },
  'spn-fusion': {
    controlId: 'spnFusionOutputItems',
    legacyField: 'spnFusionChunkItems',
    unit: 'output-item',
    optional: true,
  },
};
const COMPOSE_PHASE_COMPLETION_STEPS = new Set([
  'depth-normalize',
  'depth-min',
  'depth-rescale',
  'base-disparity',
  'base-grid',
  'base-color',
]);

const MODE_PRESETS = {
  background: {
    spnPatchChunkSize: 1,
    yieldMs: 16,
    waitForSubmittedWorkDone: true,
    gaussianPhaseYieldMs: 16,
    vitBlockChunkSize: 1,
    vitMicroduty: true,
    routeTailYieldMs: 16,
    cpuChunkItems: 65536,
  },
  furnace: {
    spnPatchChunkSize: 1,
    yieldMs: 16,
    waitForSubmittedWorkDone: true,
    gaussianPhaseYieldMs: 16,
    vitBlockChunkSize: 1,
    vitMicroduty: true,
    routeTailYieldMs: 16,
    cpuChunkItems: 65536,
  },
};

function nowMs() {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function validProgressOrdinal(index, total = null) {
  if (!Number.isSafeInteger(index) || index < 0) return false;
  if (total === null || total === undefined) return true;
  return Number.isSafeInteger(total) && total > 0 && index < total;
}

function progressMessageForBoundary(phase, details = {}, fallback) {
  if (phase === 'spn-patch-chunk'
      && validExactWorkRange(
        details.chunkEnd,
        details.totalPatches,
        details.chunkStart ?? null,
      )) {
    return `SHARP is extracting image features (patches ${details.chunkEnd}/${details.totalPatches}).`;
  }
  if (phase === 'vit-patch-embed' || phase === 'vit-block-chunk' || phase === 'vit-block-microphase') {
    const identity = [];
    if (details.encoder === 'patch'
        && validProgressOrdinal(details.patchIndex, details.totalPatches)
        && Number.isSafeInteger(details.totalPatches)) {
      identity.push(`patch ${details.patchIndex + 1}/${details.totalPatches}`);
    } else if (details.encoder === 'image') {
      identity.push('image encoder');
    }
    if (validProgressOrdinal(details.blockIndex, details.totalBlocks)) {
      identity.push(Number.isSafeInteger(details.totalBlocks)
        ? `block ${details.blockIndex + 1}/${details.totalBlocks}`
        : `block ${details.blockIndex + 1}`);
    }
    if (typeof details.microphase === 'string' && details.microphase) {
      identity.push(details.microphase.replaceAll('-', ' '));
    }
    return identity.length
      ? `SHARP is extracting image features (${identity.join(', ')}).`
      : fallback;
  }
  if (phase === 'spn-fusion' && typeof details.block === 'string') {
    const chunk = validProgressOrdinal(details.outputChunkIndex, details.outputChunkCount)
      ? `, output chunk ${details.outputChunkIndex + 1}${Number.isSafeInteger(details.outputChunkCount) ? `/${details.outputChunkCount}` : ''}`
      : '';
    return `SHARP is fusing image features (${details.block}${chunk}).`;
  }
  if ((phase === 'monodepth-phase' || phase === 'gaussian-phase')
      && typeof details.phase === 'string') {
    return `${fallback.replace(/\.$/, '')} (${details.phase.replaceAll('-', ' ')}).`;
  }
  if (phase === 'route-tail' && typeof details.step === 'string') {
    return `SHARP is assembling the splat artifact (${details.step.replaceAll('-', ' ')}).`;
  }
  return fallback;
}

function validExactWorkRange(completed, total, start = null) {
  return Number.isSafeInteger(completed)
    && Number.isSafeInteger(total)
    && total > 0
    && completed >= 0
    && completed <= total
    && (start === null
      || (Number.isSafeInteger(start) && start >= 0 && start <= completed));
}

function exactWorkForBoundary(phase, details = {}) {
  if (phase === 'spn-patch-chunk'
      && validExactWorkRange(
        details.chunkEnd,
        details.totalPatches,
        details.chunkStart ?? null,
      )) {
    return {
      completed: details.chunkEnd,
      total: details.totalPatches,
      unit: 'patch',
      authority: 'scheduler-range',
    };
  }
  if (validExactWorkRange(
    details.outputEnd,
    details.totalOutputItems,
    details.outputStart ?? null,
  )) {
    return {
      completed: details.outputEnd,
      total: details.totalOutputItems,
      unit: 'output-item',
      authority: 'scheduler-range',
    };
  }
  if (validExactWorkRange(
    details.processedItems,
    details.totalItems,
    details.segmentStartProcessedItems ?? null,
  )) {
    return {
      completed: details.processedItems,
      total: details.totalItems,
      unit: 'item',
      authority: 'scheduler-range',
    };
  }
  return null;
}

function exactVitProgressForBoundary(phase, details = {}) {
  if (phase !== 'vit-patch-embed'
      && phase !== 'vit-block-chunk'
      && phase !== 'vit-block-microphase') return null;
  if (!Number.isSafeInteger(details.totalPatches) || details.totalPatches <= 0) return null;

  const totalEncoders = Number.isSafeInteger(details.totalEncoders)
    ? details.totalEncoders
    : details.totalPatches + 1;
  if (totalEncoders !== details.totalPatches + 1) return null;
  const encoderIndex = details.encoder === 'image'
    ? details.totalPatches
    : details.patchIndex;
  if (!validProgressOrdinal(encoderIndex, totalEncoders)) return null;

  let encoderProgress = 0;
  if (phase === 'vit-patch-embed') {
    encoderProgress = 0;
  } else if (validProgressOrdinal(details.blockIndex, details.totalBlocks)
      && validProgressOrdinal(details.microdutyIndex, details.totalMicroduties)
      && Number.isSafeInteger(details.totalBlocks)
      && Number.isSafeInteger(details.totalMicroduties)) {
    const encoderDutyCount = 1 + details.totalBlocks * details.totalMicroduties;
    const completedDutyCount = 1
      + details.blockIndex * details.totalMicroduties
      + details.microdutyIndex
      + 1;
    encoderProgress = completedDutyCount / encoderDutyCount;
  } else if (validExactWorkRange(details.blockEnd, details.totalBlocks, details.blockStart ?? null)) {
    encoderProgress = details.blockEnd / details.totalBlocks;
  } else {
    return null;
  }

  const featureWorkProgress = (encoderIndex + Math.min(1, encoderProgress)) / totalEncoders;
  return 0.20 + (0.64 - 0.20) * featureWorkProgress;
}

export function createSharpProgressTracker({ now = nowMs } = {}) {
  if (typeof now !== 'function') throw new TypeError('SHARP progress clock must be callable');
  const phaseCounts = new Map();
  let lastProgress = 0;
  let livenessOrdinal = 0;

  const buildRouteProgress = (progress, message, details = {}, authority = {}) => {
    const nextProgress = Math.max(lastProgress, Math.min(0.93, Number(progress) || 0));
    lastProgress = nextProgress;
    return {
      ...details,
      schema: 'sharp-webgpu.progress.v0',
      progress: nextProgress,
      message,
      progressAuthority: 'stage-weighted-work-projection',
      completionAuthority: 'not-wall-time',
      livenessAuthority: authority.livenessAuthority
        || (livenessOrdinal > 0
          ? 'last-completed-scheduler-boundary'
          : 'no-scheduler-boundary-observed'),
      livenessOrdinal,
      timestampMs: now(),
    };
  };
  const emitRouteProgress = (progress, message, details = {}) => buildRouteProgress(
    progress,
    message,
    details,
  );

  const reportSchedulerBoundary = event => {
    if (!event || typeof event !== 'object' || typeof event.phase !== 'string') {
      throw new TypeError('SHARP scheduler progress requires a phase-bearing boundary event');
    }
    livenessOrdinal += 1;
    const phaseWorkOrdinal = (phaseCounts.get(event.phase) || 0) + 1;
    phaseCounts.set(event.phase, phaseWorkOrdinal);
    const projection = event.phase === 'monodepth-phase'
      ? { floor: 0.66, ceiling: 0.76, divisor: 28, message: 'SHARP is resolving scene depth.' }
      : event.phase === 'gaussian-phase'
        ? { floor: 0.79, ceiling: 0.89, divisor: 34, message: 'SHARP is predicting Gaussian geometry.' }
        : event.phase === 'route-tail'
          ? { floor: 0.90, ceiling: 0.925, divisor: 18, message: 'SHARP is assembling the splat artifact.' }
          : { floor: 0.20, ceiling: 0.64, divisor: 170, message: 'SHARP is extracting image features.' };
    const work = event.details && typeof event.details === 'object'
      ? { ...event.details }
      : {};
    const exactVitProgress = exactVitProgressForBoundary(event.phase, work);
    const progress = exactVitProgress ?? (
      projection.floor
      + (projection.ceiling - projection.floor) * (1 - Math.exp(-phaseWorkOrdinal / projection.divisor))
    );
    return buildRouteProgress(
      progress,
      progressMessageForBoundary(event.phase, work, projection.message),
      {
        kind: 'scheduler-boundary',
        phase: event.phase,
        boundary: event.boundary,
        phaseWorkOrdinal,
        workOrdinal: phaseWorkOrdinal,
        projectionWorkAuthority: exactVitProgress === null
          ? 'phase-event-count-fallback'
          : 'exact-vit-encoder-block-work',
        work,
        exactWork: exactWorkForBoundary(event.phase, work),
      },
      { livenessAuthority: 'completed-scheduler-boundary' },
    );
  };

  return {
    emitRouteProgress,
    reportSchedulerBoundary,
  };
}

function timeOriginEpochMs() {
  if (typeof performance !== 'undefined' && Number.isFinite(performance.timeOrigin)) {
    return performance.timeOrigin;
  }
  return 0;
}

function createEventClock() {
  const source = typeof performance !== 'undefined' ? 'performance.now' : 'date-now';
  return {
    schema: EVENT_CLOCK_SCHEMA,
    clockId: 'sharp-webgpu-performance-clock',
    source,
    relativeClock: source,
    epochClock: typeof performance !== 'undefined' ? 'performance.timeOrigin+performance.now' : 'date-now',
    timeOriginEpochMs: timeOriginEpochMs(),
  };
}

function createEventTrace(events = []) {
  return {
    schema: EVENT_TRACE_SCHEMA,
    clock: createEventClock(),
    timingAuthority: events.length ? 'browser-wall-clock' : 'not-observed',
    events,
  };
}

function boundaryForPhase(phase) {
  if (phase === 'spn-patch-chunk') return 'spn-patch-chunk';
  if (phase === 'spn-image-encoder') return 'spn-image-encoder';
  if (phase === 'spn-fusion') return 'spn-fusion';
  if (phase === 'gaussian-phase') return 'gaussian-phase';
  return phase || 'unknown';
}

function liveControlForBoundary(boundary) {
  return LIVE_SCHEDULER_CONTROLS[boundary] || null;
}

function liveSchedulerFailure(telemetry, phase, boundary, dutyId, error) {
  recordSchedulerEvent(telemetry, phase, {
    boundary,
    kind: 'live-scheduler-duty-failed',
    dutyId,
    error: {
      name: error?.name || 'Error',
      message: error?.message || String(error),
    },
  });
}

function nextSchedulerDutyId(telemetry, boundary) {
  const dutySequence = Number.isInteger(telemetry?._nextDutySequence)
    ? telemetry._nextDutySequence
    : (telemetry?.eventTrace?.events?.filter(event => event?.kind === 'queue-work-done-start').length || 0);
  if (telemetry) telemetry._nextDutySequence = dutySequence + 1;
  return `${telemetry?.runId || 'sharp-webgpu'}:${boundary}:${dutySequence}`;
}

function setLegacySchedulerControl(scheduler, telemetry, legacyField, value) {
  if (!Number.isInteger(value)) return;
  if (scheduler?.effective && typeof scheduler.effective === 'object') {
    scheduler.effective[legacyField] = value;
  }
  if (telemetry?.effectiveScheduler && typeof telemetry.effectiveScheduler === 'object') {
    telemetry.effectiveScheduler[legacyField] = value;
  }
}

function foregroundOpportunityHook(live) {
  return live?.foregroundOpportunityHook
    || (typeof globalThis !== 'undefined' ? globalThis.__kaminosSharpForegroundOpportunity : null);
}

function requestLiveForegroundOpportunity({ live, telemetry, phase, boundary, dutyId, details }) {
  const hook = foregroundOpportunityHook(live);
  if (!hook || typeof live?.runtime?.requestForegroundOpportunity !== 'function') return null;
  try {
    const requestInput = hook({
      routeId: live.runtime.routeId,
      runId: live.runId || telemetry?.runId || null,
      stage: live.stage || null,
      phase,
      boundary,
      dutyId,
      details,
      device: live.runtime.device,
      queue: live.runtime.queue,
    });
    if (!requestInput) return null;
    const request = live.runtime.requestForegroundOpportunity(requestInput);
    recordSchedulerEvent(telemetry, phase, {
      ...details,
      boundary,
      kind: 'foreground-opportunity-requested',
      dutyId,
      requestId: request.requestId,
      source: 'kaminos-sharp-foreground-hook',
    });
    return request;
  } catch (error) {
    recordSchedulerEvent(telemetry, phase, {
      ...details,
      boundary,
      kind: 'foreground-opportunity-request-failed',
      dutyId,
      error: {
        name: error?.name || 'Error',
        message: error?.message || String(error),
      },
    });
    throw new Error(
      `foreground opportunity request failed: ${error?.message || String(error)}`,
      { cause: error },
    );
  }
}

async function serviceLiveForegroundOpportunity({ scheduler, telemetry, phase, boundary, dutyId, details }) {
  const live = scheduler?.liveScheduler;
  if (!live?.runtime || !live?.invocation) return null;
  const foregroundRequest = requestLiveForegroundOpportunity({
    live,
    telemetry,
    phase,
    boundary,
    dutyId,
    details,
  });
  const snapshot = typeof live.runtime.foregroundOpportunityPressureSnapshot === 'function'
    ? live.runtime.foregroundOpportunityPressureSnapshot()
    : live.runtime.foregroundOpportunitySnapshot?.();
  if (!snapshot || snapshot.pendingRequestCount === 0) return null;
  const interlock = live.runtime.foregroundOpportunities;
  if (typeof interlock?.serviceAtBoundary !== 'function') {
    throw new Error('foreground opportunity service is unavailable at SHARP scheduler boundary');
  }
  const service = await interlock.serviceAtBoundary({
    invocationId: live.invocation.invocationId,
    boundaryId: `${live.invocation.invocationId}:sharp-yield:${dutyId}`,
    dutyId,
    phase,
    position: 'before-encode',
    metadata: {
      runtimeLabel: live.runtime.runtimeLabel,
      schedulerRevision: live.invocation.schedulerRevision ?? null,
      boundary,
      stage: live.stage || null,
      ...details,
    },
  });
  const receipt = foregroundRequest?.completion
    ? await foregroundRequest.completion
    : null;
  recordSchedulerEvent(telemetry, phase, {
    ...details,
    boundary,
    kind: 'foreground-opportunity-serviced',
    dutyId,
    requestId: receipt?.requestId || null,
    foregroundStatus: receipt?.status || service.status,
    submissionCount: receipt?.submissionCount ?? null,
    capturedRequestCount: service.capturedRequestCount,
    servicedRequestCount: service.servicedRequestCount,
  });
  if (service.status === 'failed') {
    const error = new Error(
      service.failures?.[0]?.failure?.error?.message || 'foreground opportunity service failed',
    );
    recordSchedulerEvent(telemetry, phase, {
      ...details,
      boundary,
      kind: 'foreground-opportunity-service-failed',
      dutyId,
      error: {
        name: error.name,
        message: error.message,
      },
    });
    throw error;
  }
  return service;
}

async function prepareLiveSchedulerDuty({ scheduler, telemetry, phase, boundary, dutyId, details }) {
  const live = scheduler?.liveScheduler;
  let control = liveControlForBoundary(boundary);
  if (!live?.runtime || !live?.invocation) return null;
  let bounds = control
    ? live.invocation.bounds?.phaseChunkSize?.[control.controlId]
    : null;
  const optionalControlEnabled = Boolean(
    control?.optional
    && Number.isInteger(scheduler?.effective?.[control.legacyField])
    && scheduler.effective[control.legacyField] > 0,
  );
  if (control?.optional && !bounds && !optionalControlEnabled) {
    control = null;
    bounds = null;
  }
  if (control && !bounds) {
    const error = new Error(`undeclared scheduler control ${control.controlId}`);
    liveSchedulerFailure(telemetry, phase, boundary, dutyId, error);
    throw error;
  }
  let foregroundRequest = null;
  let duty = null;
  try {
    const current = control
      ? (Number.isInteger(scheduler?.effective?.[control.legacyField])
          ? scheduler.effective[control.legacyField]
          : live.invocation.getControl(control.controlId))
      : null;
    foregroundRequest = requestLiveForegroundOpportunity({
      live,
      telemetry,
      phase,
      boundary,
      dutyId,
      details,
    });
    const dutyInput = {
      dutyId,
      phase,
      kind: 'compute',
      metadata: {
        boundary,
        stage: live.stage || null,
        operation: 'sharp-scheduler-yield-before-next-encode',
        ...details,
      },
    };
    if (control) {
      dutyInput.chunkControl = {
        controlId: control.controlId,
        unit: control.unit,
        current,
        bounds,
      };
    }
    duty = await live.runtime.prepareCommandDutyAtBoundary(dutyInput, live.invocation);
    if (foregroundRequest?.completion) {
      const receipt = await foregroundRequest.completion;
      recordSchedulerEvent(telemetry, phase, {
        ...details,
        boundary,
        kind: 'foreground-opportunity-serviced',
        dutyId,
        requestId: receipt.requestId,
        foregroundStatus: receipt.status,
        submissionCount: receipt.submissionCount,
      });
    }
    if (control) {
      setLegacySchedulerControl(scheduler, telemetry, control.legacyField, duty.chunkControl.current);
    }
    recordSchedulerEvent(telemetry, phase, {
      ...details,
      boundary,
      kind: 'live-scheduler-duty-prepared',
      dutyId,
      controlId: control?.controlId || null,
      current: control ? duty.chunkControl.current : null,
      schedulerRevision: duty.metadata?.schedulerBoundary?.effectiveSchedulerRevision ?? null,
      schedulerChanged: Boolean(duty.metadata?.schedulerBoundary?.schedulerChanged),
    });
    return duty;
  } catch (error) {
    liveSchedulerFailure(telemetry, phase, boundary, dutyId, error);
    if (duty) {
      try {
        live.runtime.settleCommandDuty(duty, {
          status: 'failed-before-encode',
          phase: `${phase}-prepare`,
          error,
        });
      } catch (settlementError) {
        liveSchedulerFailure(telemetry, phase, boundary, dutyId, settlementError);
        throw settlementError;
      }
    }
    throw error;
  }
}

export async function prepareSchedulerDutyBeforeEncode(scheduler, telemetry, phase, details = {}) {
  const boundary = boundaryForPhase(phase);
  const dutyId = nextSchedulerDutyId(telemetry, boundary);
  return prepareLiveSchedulerDuty({
    scheduler,
    telemetry,
    phase,
    boundary,
    dutyId,
    details,
  });
}

export function failPreparedSchedulerDuty(scheduler, telemetry, liveDuty, phase, error) {
  if (!liveDuty) return null;
  const boundary = boundaryForPhase(phase);
  try {
    return scheduler.liveScheduler.runtime.settleCommandDuty(liveDuty, {
      status: 'failed-before-encode',
      phase: `${phase}-plan-or-encode`,
      error,
    });
  } catch (settlementError) {
    liveSchedulerFailure(telemetry, phase, boundary, liveDuty.dutyId, settlementError);
    throw settlementError;
  }
}

export async function submitPreparedSchedulerDuty(
  scheduler,
  device,
  telemetry,
  liveDuty,
  commandBuffers,
  phase,
  details = {},
) {
  if (!Array.isArray(commandBuffers) || commandBuffers.length === 0) {
    throw new TypeError('prepared scheduler duty submission requires command buffers');
  }
  const boundary = boundaryForPhase(phase);
  if (!liveDuty) {
    device.queue.submit(commandBuffers);
    return null;
  }
  liveDuty.metadata = {
    ...(liveDuty.metadata || {}),
    ...details,
  };
  scheduler.liveScheduler.runtime.settleCommandDuty(liveDuty, { status: 'encoded' });
  try {
    await scheduler.liveScheduler.runtime.commandDuties.measureSubmission(
      liveDuty,
      () => device.queue.submit(commandBuffers),
    );
  } catch (error) {
    recordSchedulerEvent(telemetry, phase, {
      ...details,
      boundary,
      kind: 'live-scheduler-duty-submit-failed',
      dutyId: liveDuty.dutyId,
      error: {
        name: error?.name || 'Error',
        message: error?.message || String(error),
      },
    });
    throw error;
  }
  recordSchedulerEvent(telemetry, phase, {
    ...details,
    boundary,
    kind: 'live-scheduler-duty-submitted',
    dutyId: liveDuty.dutyId,
  });
  return liveDuty;
}

export function attachSharpLiveScheduler(scheduler, liveScheduler) {
  if (!scheduler || typeof scheduler !== 'object') return scheduler;
  Object.defineProperty(scheduler, 'liveScheduler', {
    value: liveScheduler,
    writable: true,
    configurable: true,
    enumerable: false,
  });
  return scheduler;
}

export function detachSharpLiveScheduler(scheduler) {
  if (scheduler && Object.prototype.hasOwnProperty.call(scheduler, 'liveScheduler')) {
    delete scheduler.liveScheduler;
  }
  return scheduler;
}

function countEvents(events, boundary, kind) {
  return events.filter(event => event?.boundary === boundary && (!kind || event?.kind === kind)).length;
}

function countEventsMatching(events, boundary, kind, predicate) {
  return events.filter(event => (
    event?.boundary === boundary
    && (!kind || event?.kind === kind)
    && (!predicate || predicate(event))
  )).length;
}

function schedulerEventCountKey(boundary, kind, role = '') {
  return `${boundary || ''}\u0000${kind || ''}\u0000${role || ''}`;
}

function recordSchedulerEventCount(index, event) {
  const baseKey = schedulerEventCountKey(event?.boundary, event?.kind);
  index.set(baseKey, (index.get(baseKey) || 0) + 1);
  if (event?.role) {
    const roleKey = schedulerEventCountKey(event.boundary, event.kind, event.role);
    index.set(roleKey, (index.get(roleKey) || 0) + 1);
  }
}

function indexedSchedulerEventCount(index, boundary, kind, role = '') {
  return index?.get(schedulerEventCountKey(boundary, kind, role)) || 0;
}

function schedulerWaitRequested(requested, effective) {
  return Boolean(requested.waitForSubmittedWorkDone || effective.waitForSubmittedWorkDone);
}

function schedulerYieldRequested(requested, effective) {
  return Number(requested.yieldMs || 0) > 0 || Number(effective.yieldMs || 0) > 0;
}

function schedulerGaussianYieldRequested(requested, effective) {
  return Number(requested.gaussianPhaseYieldMs || 0) > 0 || Number(effective.gaussianPhaseYieldMs || 0) > 0;
}

function schedulerRouteTailYieldRequested(requested, effective) {
  return Number(requested.routeTailYieldMs || 0) > 0 || Number(effective.routeTailYieldMs || 0) > 0;
}

function boundaryProofStatus({ unsupported, observedCount, observedQueueWaitCount, observedYieldCount, queueWaitRequested, yieldRequested }) {
  if (unsupported) return 'unsupported';
  const queueSatisfied = !queueWaitRequested || observedQueueWaitCount > 0;
  const yieldSatisfied = !yieldRequested || observedYieldCount > 0;
  return observedCount > 0 && queueSatisfied && yieldSatisfied ? 'verified' : 'unverified';
}

function requestedBoundaryAssertions(telemetry, eventCountIndex = null) {
  const requested = telemetry?.requestedScheduler || {};
  const effective = telemetry?.effectiveScheduler || {};
  const unsupportedFields = new Set(telemetry?.unsupportedFields || []);
  const events = telemetry?.eventTrace?.events || telemetry?.events || [];
  const queueWaitRequested = schedulerWaitRequested(requested, effective);
  const yieldRequested = schedulerYieldRequested(requested, effective);
  const gaussianYieldRequested = schedulerGaussianYieldRequested(requested, effective);
  const assertions = [];
  const count = (boundary, kind, role = '') => eventCountIndex
    ? indexedSchedulerEventCount(eventCountIndex, boundary, kind, role)
    : role
      ? countEventsMatching(events, boundary, kind, event => event?.role === role)
      : countEvents(events, boundary, kind);

  if (Number.isFinite(effective.spnPatchChunkSize) && effective.spnPatchChunkSize > 0) {
    const boundary = 'spn-patch-chunk';
    const observedCount = count(boundary, 'chunk-start');
    const observedQueueWaitCount = Math.min(
      count(boundary, 'queue-work-done-start'),
      count(boundary, 'queue-work-done-end')
    );
    const observedYieldCount = Math.min(
      count(boundary, 'js-yield-start'),
      count(boundary, 'js-yield-end')
    );
    const unsupported = unsupportedFields.has('spnPatchChunkSize') || unsupportedFields.has('phaseChunkSize.spnPatch') || unsupportedFields.has('phaseChunkSize');
    assertions.push({
      field: 'phaseChunkSize.spnPatch',
      requested: Number.isFinite(requested.spnPatchChunkSize) ? requested.spnPatchChunkSize : null,
      effective: Number.isFinite(effective.spnPatchChunkSize) ? effective.spnPatchChunkSize : null,
      status: boundaryProofStatus({
        unsupported,
        observedCount,
        observedQueueWaitCount,
        observedYieldCount,
        queueWaitRequested,
        yieldRequested: true,
      }),
      observedBoundary: boundary,
      observedCount,
      expectedMinimumCount: 1,
      observedQueueWaitCount,
      observedYieldCount,
      unsupportedReason: unsupported ? 'effective scheduler declared this field unsupported' : null,
    });
  }

  if (Number.isFinite(effective.spnFusionChunkItems) && effective.spnFusionChunkItems > 0) {
    const boundary = 'spn-fusion';
    const role = 'spn-fusion-output-chunk';
    const unsupported = unsupportedFields.has('spnFusionChunkItems') || unsupportedFields.has('phaseChunkSize.spnFusionOutputItems') || unsupportedFields.has('phaseChunkSize');
    const observedChunk = event => (
      (event?.role === role || event?.chunkRole === role)
      && Number(event?.outputChunkCount || 0) > 1
    );
    const observedCount = countEventsMatching(events, boundary, 'chunk-start', observedChunk);
    const observedQueueWaitCount = Math.min(
      countEventsMatching(events, boundary, 'queue-work-done-start', observedChunk),
      countEventsMatching(events, boundary, 'queue-work-done-end', observedChunk)
    );
    const observedYieldCount = Math.min(
      countEventsMatching(events, boundary, 'js-yield-start', observedChunk),
      countEventsMatching(events, boundary, 'js-yield-end', observedChunk)
    );
    assertions.push({
      field: 'phaseChunkSize.spnFusionOutputItems',
      requested: Number.isFinite(requested.spnFusionChunkItems) ? requested.spnFusionChunkItems : null,
      effective: effective.spnFusionChunkItems,
      status: boundaryProofStatus({
        unsupported,
        observedCount,
        observedQueueWaitCount,
        observedYieldCount,
        queueWaitRequested,
        yieldRequested: true,
      }),
      observedBoundary: boundary,
      observedRole: role,
      observedCount,
      expectedMinimumCount: 1,
      observedQueueWaitCount,
      observedYieldCount,
      unsupportedReason: unsupported ? 'effective scheduler declared this field unsupported' : null,
    });
  }

  if (Number.isFinite(effective.decoderKernelChunkItems) && effective.decoderKernelChunkItems > 0) {
    const role = 'decoder-kernel-output-tile';
    const observedCount = count('monodepth-phase', 'chunk-start', role)
      + count('gaussian-phase', 'chunk-start', role);
    const observedQueueWaitCount = Math.min(
      count('monodepth-phase', 'queue-work-done-start', role)
        + count('gaussian-phase', 'queue-work-done-start', role),
      count('monodepth-phase', 'queue-work-done-end', role)
        + count('gaussian-phase', 'queue-work-done-end', role)
    );
    const observedYieldCount = Math.min(
      count('monodepth-phase', 'js-yield-start', role)
        + count('gaussian-phase', 'js-yield-start', role),
      count('monodepth-phase', 'js-yield-end', role)
        + count('gaussian-phase', 'js-yield-end', role)
    );
    const unsupported = unsupportedFields.has('decoderKernelChunkItems');
    assertions.push({
      field: 'decoderKernelChunkItems',
      requested: Number.isFinite(requested.decoderKernelChunkItems) ? requested.decoderKernelChunkItems : null,
      effective: effective.decoderKernelChunkItems,
      status: boundaryProofStatus({
        unsupported,
        observedCount: observedCount >= 2 ? observedCount : 0,
        observedQueueWaitCount,
        observedYieldCount,
        queueWaitRequested,
        yieldRequested: true,
      }),
      observedBoundaries: ['monodepth-phase', 'gaussian-phase'],
      observedRole: role,
      observedCount,
      expectedMinimumCount: 2,
      observedQueueWaitCount,
      observedYieldCount,
      unsupportedReason: unsupported ? 'effective scheduler declared this field unsupported' : null,
    });
  }

  if (Number.isFinite(effective.vitBlockChunkSize) && effective.vitBlockChunkSize > 0) {
    const boundary = 'vit-block-chunk';
    const unsupported = unsupportedFields.has('vitBlockChunkSize') || unsupportedFields.has('phaseChunkSize.vitBlock') || unsupportedFields.has('phaseChunkSize');
    const observedCount = count(boundary, 'chunk-start');
    const observedQueueWaitCount = Math.min(
      count(boundary, 'queue-work-done-start'),
      count(boundary, 'queue-work-done-end')
    );
    const observedYieldCount = Math.min(
      count(boundary, 'js-yield-start'),
      count(boundary, 'js-yield-end')
    );
    assertions.push({
      field: 'phaseChunkSize.vitBlock',
      requested: Number.isFinite(requested.vitBlockChunkSize) ? requested.vitBlockChunkSize : null,
      effective: Number.isFinite(effective.vitBlockChunkSize) ? effective.vitBlockChunkSize : null,
      status: boundaryProofStatus({
        unsupported,
        observedCount,
        observedQueueWaitCount,
        observedYieldCount,
        queueWaitRequested,
        yieldRequested: true,
      }),
      observedBoundary: boundary,
      observedCount,
      expectedMinimumCount: 1,
      observedQueueWaitCount,
      observedYieldCount,
      unsupportedReason: unsupported ? 'effective scheduler declared this field unsupported' : null,
    });
  }

  if (schedulerRouteTailYieldRequested(requested, effective)) {
    const boundary = 'route-tail';
    const unsupported = unsupportedFields.has('routeTailYieldMs') || unsupportedFields.has('phaseYieldMs.routeTail') || unsupportedFields.has('phaseYieldMs');
    const observedCount = count(boundary, 'chunk-start');
    const observedQueueWaitCount = Math.min(
      count(boundary, 'queue-work-done-start'),
      count(boundary, 'queue-work-done-end')
    );
    const observedYieldCount = Math.min(
      count(boundary, 'js-yield-start'),
      count(boundary, 'js-yield-end')
    );
    assertions.push({
      field: 'phaseYieldMs.routeTail',
      requested: Number.isFinite(requested.routeTailYieldMs) ? requested.routeTailYieldMs : null,
      effective: Number.isFinite(effective.routeTailYieldMs) ? effective.routeTailYieldMs : null,
      status: boundaryProofStatus({
        unsupported,
        observedCount,
        observedQueueWaitCount,
        observedYieldCount,
        queueWaitRequested,
        yieldRequested: true,
      }),
      observedBoundary: boundary,
      observedCount,
      expectedMinimumCount: 1,
      observedQueueWaitCount,
      observedYieldCount,
      unsupportedReason: unsupported ? 'effective scheduler declared this field unsupported' : null,
    });
  }

  if (Number.isFinite(effective.cpuChunkItems) && effective.cpuChunkItems > 0) {
    const boundary = 'route-tail';
    const unsupported = unsupportedFields.has('cpuChunkItems') || unsupportedFields.has('cpuMaterializationChunkItems');
    const observedCount = count(boundary, 'chunk-start', 'cpu-materialization-chunk');
    const observedQueueWaitCount = Math.min(
      count(boundary, 'queue-work-done-start', 'cpu-materialization-chunk'),
      count(boundary, 'queue-work-done-end', 'cpu-materialization-chunk')
    );
    const observedYieldCount = Math.min(
      count(boundary, 'js-yield-start', 'cpu-materialization-chunk'),
      count(boundary, 'js-yield-end', 'cpu-materialization-chunk')
    );
    assertions.push({
      field: 'cpuMaterializationChunkItems',
      requested: Number.isFinite(requested.cpuChunkItems) ? requested.cpuChunkItems : null,
      effective: effective.cpuChunkItems,
      status: boundaryProofStatus({
        unsupported,
        observedCount,
        observedQueueWaitCount,
        observedYieldCount,
        queueWaitRequested,
        yieldRequested: schedulerRouteTailYieldRequested(requested, effective),
      }),
      observedBoundary: boundary,
      observedRole: 'cpu-materialization-chunk',
      observedCount,
      expectedMinimumCount: 1,
      observedQueueWaitCount,
      observedYieldCount,
      unsupportedReason: unsupported ? 'effective scheduler declared this field unsupported' : null,
    });
  }

  if (gaussianYieldRequested) {
    const boundary = 'gaussian-phase';
    const observedCount = count(boundary, 'chunk-start');
    const observedQueueWaitCount = Math.min(
      count(boundary, 'queue-work-done-start'),
      count(boundary, 'queue-work-done-end')
    );
    const observedYieldCount = Math.min(
      count(boundary, 'js-yield-start'),
      count(boundary, 'js-yield-end')
    );
    assertions.push({
      field: 'phaseYieldMs.gaussianPhase',
      requested: Number.isFinite(requested.gaussianPhaseYieldMs) ? requested.gaussianPhaseYieldMs : null,
      effective: Number.isFinite(effective.gaussianPhaseYieldMs) ? effective.gaussianPhaseYieldMs : null,
      status: boundaryProofStatus({
        unsupported: false,
        observedCount,
        observedQueueWaitCount,
        observedYieldCount,
        queueWaitRequested,
        yieldRequested: true,
      }),
      observedBoundary: boundary,
      observedCount,
      expectedMinimumCount: 1,
      observedQueueWaitCount,
      observedYieldCount,
      unsupportedReason: null,
    });
  }

  return assertions;
}

function derivedTelemetryStatus(telemetry, requestedStatus) {
  if (requestedStatus === 'running' || requestedStatus === 'failed') return requestedStatus;
  if (telemetry?.unsupportedFields?.length) return 'unsupported';
  const assertions = telemetry?.boundaryAssertions || [];
  if (!assertions.length) return 'scheduler-unverified';
  if (assertions.some(assertion => assertion?.status === 'unsupported')) return 'unsupported';
  if (assertions.every(assertion => assertion?.status === 'verified')) return 'verified';
  return 'scheduler-unverified';
}

function parseSchedulerPayload(value) {
  if (!value) return {};
  if (typeof value === 'object' && !Array.isArray(value)) return value;
  if (typeof value !== 'string') return {};
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function queryPayload(options = {}) {
  if (options.sharpScheduler) return parseSchedulerPayload(options.sharpScheduler);
  const search = options.search ?? (typeof window !== 'undefined' ? window.location.search : '');
  const params = options.params || new URLSearchParams(search || '');
  const payload = parseSchedulerPayload(params.get('sharpScheduler'));
  for (const key of Object.keys(DEFAULT_SCHEDULER)) {
    if (!params.has(key)) continue;
    payload[key] = params.get(key);
  }
  const globalConfig = options.globalConfig ?? (typeof window !== 'undefined' ? window.__SHARP_SCHEDULER__ : null);
  return { ...parseSchedulerPayload(globalConfig), ...payload };
}

function normalizeInt(value, fallback, { min = 0, max = Number.MAX_SAFE_INTEGER } = {}) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, Math.round(number)));
}

function normalizeBool(value, fallback = false) {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;
  if (typeof value === 'string') return /^(1|true|yes|on)$/i.test(value);
  return fallback;
}

function normalizeRetirementBool(value, fallback = false) {
  if (value === undefined || value === null) return fallback;
  if (typeof value === 'boolean') return value;
  if (value === 1 || /^(1|true|yes|on)$/i.test(String(value))) return true;
  if (value === 0 || /^(0|false|no|off)$/i.test(String(value))) return false;
  throw new RangeError('retirePostInferenceBuffers must be a boolean');
}

function normalizeVitMicrodutyMode(value, fallback = DEFAULT_SCHEDULER.vitMicrodutyMode) {
  const mode = value === undefined || value === null ? fallback : String(value);
  if (!VIT_MICRODUTY_MODES.has(mode)) {
    throw new RangeError(`ViT microduty mode must be one of: ${[...VIT_MICRODUTY_MODES].join(', ')}`);
  }
  return mode;
}

function normalizePlyAssemblyMode(value, fallback = DEFAULT_SCHEDULER.plyAssemblyMode) {
  const mode = value === undefined || value === null ? fallback : String(value);
  if (!PLY_ASSEMBLY_MODES.has(mode)) {
    throw new RangeError(`Unsupported PLY assembly mode: ${mode}`);
  }
  return mode;
}

export function classifyCpuDutyCheckpoint(scheduler, details = {}, processedItems = 0) {
  const chunkItems = scheduler?.effective?.cpuChunkItems || 0;
  const prepPhaseComplete = details.phaseComplete === true
    && details.stage === 'compose-ply'
    && COMPOSE_PHASE_COMPLETION_STEPS.has(details.step);
  const gaussianPhaseComplete = details.phaseComplete === true
    && details.stage === 'compose-ply'
    && details.step === 'gaussian-compose'
    && Number.isFinite(details.totalItems)
    && details.totalItems > 0
    && processedItems === details.totalItems;
  const rowBatchedCheckpoint = details.stage === 'compose-ply'
    && details.step === 'gaussian-compose'
    && details.granularity === 'row-batched'
    && Number.isFinite(details.checkpointItems)
    && Number.isFinite(details.segmentStartProcessedItems)
    && Number.isFinite(details.segmentEndProcessedItems)
    && details.segmentStartProcessedItems < details.checkpointItems
    && details.checkpointItems <= details.segmentEndProcessedItems
    && processedItems === details.segmentEndProcessedItems;
  const phaseComplete = prepPhaseComplete || gaussianPhaseComplete;
  return {
    eligible: Boolean(
      chunkItems
      && processedItems > 0
      && (phaseComplete || rowBatchedCheckpoint || processedItems % chunkItems === 0)
    ),
    phaseComplete,
  };
}

export function planSpnFusionChunks(totalOutputItems, chunkItems = 0) {
  if (!Number.isSafeInteger(totalOutputItems) || totalOutputItems <= 0) {
    throw new RangeError('SPN fusion total output items must be a positive safe integer');
  }
  if (!Number.isSafeInteger(chunkItems) || chunkItems < 0) {
    throw new RangeError('SPN fusion chunk items must be a non-negative safe integer');
  }
  const chunks = [];
  let outputStart = 0;
  let chunkIndex = 0;
  while (outputStart < totalOutputItems) {
    const next = planNextSpnFusionChunk(totalOutputItems, outputStart, chunkItems, chunkIndex);
    chunks.push({
      chunkIndex: next.chunkIndex,
      chunkCount: next.projectedChunkCount,
      outputStart: next.outputStart,
      outputEnd: next.outputEnd,
      outputCount: next.outputCount,
    });
    outputStart = next.outputEnd;
    chunkIndex += 1;
  }
  return chunks;
}

export function planNextSpnFusionChunk(totalOutputItems, outputStart, chunkItems = 0, chunkIndex = 0) {
  if (!Number.isSafeInteger(totalOutputItems) || totalOutputItems <= 0) {
    throw new RangeError('SPN fusion total output items must be a positive safe integer');
  }
  if (!Number.isSafeInteger(outputStart) || outputStart < 0 || outputStart >= totalOutputItems) {
    throw new RangeError('SPN fusion output start must identify a remaining safe-integer range');
  }
  if (!Number.isSafeInteger(chunkItems) || chunkItems < 0) {
    throw new RangeError('SPN fusion chunk items must be a non-negative safe integer');
  }
  if (!Number.isSafeInteger(chunkIndex) || chunkIndex < 0) {
    throw new RangeError('SPN fusion chunk index must be a non-negative safe integer');
  }
  const effectiveChunkItems = chunkItems > 0 ? chunkItems : totalOutputItems;
  const outputCount = Math.min(totalOutputItems - outputStart, effectiveChunkItems);
  const outputEnd = outputStart + outputCount;
  return {
    chunkIndex,
    projectedChunkCount: chunkIndex + Math.ceil((totalOutputItems - outputStart) / effectiveChunkItems),
    outputStart,
    outputEnd,
    outputCount,
  };
}

export function planNextVitBlockChunk(totalBlocks, blockStart, chunkBlocks, blockChunkIndex = 0) {
  if (!Number.isSafeInteger(totalBlocks) || totalBlocks <= 0) {
    throw new RangeError('ViT total blocks must be a positive safe integer');
  }
  if (!Number.isSafeInteger(blockStart) || blockStart < 0 || blockStart >= totalBlocks) {
    throw new RangeError('ViT block start must identify a remaining safe-integer range');
  }
  if (!Number.isSafeInteger(chunkBlocks) || chunkBlocks <= 0) {
    throw new RangeError('ViT chunk blocks must be a positive safe integer');
  }
  if (!Number.isSafeInteger(blockChunkIndex) || blockChunkIndex < 0) {
    throw new RangeError('ViT block chunk index must be a non-negative safe integer');
  }
  const blockCount = Math.min(totalBlocks - blockStart, chunkBlocks);
  return {
    blockChunkIndex,
    blockStart,
    blockEnd: blockStart + blockCount,
    blockCount,
    totalBlocks,
  };
}

export function planDecoderKernelChunks(totalOutputItems, chunkItems = 0) {
  if (!Number.isSafeInteger(totalOutputItems) || totalOutputItems <= 0) {
    throw new RangeError('decoder kernel total output items must be a positive safe integer');
  }
  if (!Number.isSafeInteger(chunkItems) || chunkItems < 0) {
    throw new RangeError('decoder kernel chunk items must be a non-negative safe integer');
  }
  const effectiveChunkItems = chunkItems > 0 ? chunkItems : totalOutputItems;
  const tileTotal = Math.ceil(totalOutputItems / effectiveChunkItems);
  const tiles = [];
  let outputStart = 0;
  for (let tileIndex = 0; outputStart < totalOutputItems; tileIndex++) {
    const outputCount = Math.min(totalOutputItems - outputStart, effectiveChunkItems);
    const outputEnd = outputStart + outputCount;
    tiles.push({
      tileIndex,
      tileTotal,
      outputStart,
      outputEnd,
      outputCount,
      totalOutputItems,
      tileUnit: 'output-item',
    });
    outputStart = outputEnd;
  }
  return tiles;
}

function planExactVitDispatchTiles(totalItems, requestedTileItems, tileUnit) {
  if (!Number.isSafeInteger(totalItems) || totalItems <= 0) {
    throw new RangeError('ViT dispatch tile total must be a positive safe integer');
  }
  if (!Number.isSafeInteger(requestedTileItems) || requestedTileItems < 0) {
    throw new RangeError('ViT dispatch tile size must be a non-negative safe integer');
  }
  if (requestedTileItems === 0) return null;
  const effectiveTileItems = Math.min(totalItems, requestedTileItems);
  const tileTotal = Math.ceil(totalItems / effectiveTileItems);
  const tiles = [];
  for (let tileIndex = 0, tileStart = 0; tileStart < totalItems; tileIndex++) {
    const tileEnd = Math.min(totalItems, tileStart + effectiveTileItems);
    tiles.push({
      tileIndex,
      tileTotal,
      tileStart,
      tileEnd,
      tileItemCount: tileEnd - tileStart,
      totalTileItems: totalItems,
      tileUnit,
      requestedTileItems,
      effectiveTileItems,
    });
    tileStart = tileEnd;
  }
  return tiles;
}

function planVitMicrophaseTiles(microphase, workload) {
  const tokenCount = workload.tokenCount;
  const dim = workload.dim;
  const mlpHiddenDim = workload.mlpHiddenDim;
  const numHeads = workload.numHeads;
  if (microphase === 'qkv-projection') {
    return planExactVitDispatchTiles(tokenCount * dim * 3, workload.linearTileItems, 'output-item');
  }
  if (microphase === 'attention-scores') {
    return planExactVitDispatchTiles(numHeads * tokenCount * tokenCount, workload.attentionTileItems, 'output-item');
  }
  if (microphase === 'attention-softmax') {
    return planExactVitDispatchTiles(numHeads * tokenCount, workload.softmaxTileRows, 'row');
  }
  if (microphase === 'norm1' || microphase === 'norm2' || microphase === 'final-norm') {
    return planExactVitDispatchTiles(tokenCount, workload.normTileRows, 'row');
  }
  if (microphase === 'attention-apply' || microphase === 'attention-projection' || microphase === 'fc2') {
    const requestedTileItems = microphase === 'attention-apply'
      ? workload.attentionTileItems
      : workload.linearTileItems;
    return planExactVitDispatchTiles(tokenCount * dim, requestedTileItems, 'output-item');
  }
  if (microphase === 'fc1') {
    return planExactVitDispatchTiles(tokenCount * mlpHiddenDim, workload.linearTileItems, 'output-item');
  }
  return null;
}

export function planVitBlockMicroduties(range, mode = DEFAULT_SCHEDULER.vitMicrodutyMode, workload = null) {
  if (!range || typeof range !== 'object') {
    throw new TypeError('ViT block range must be an object');
  }
  const { blockStart, blockEnd, blockCount, totalBlocks } = range;
  if (!Number.isSafeInteger(totalBlocks) || totalBlocks <= 0) {
    throw new RangeError('ViT total blocks must be a positive safe integer');
  }
  if (!Number.isSafeInteger(blockStart) || !Number.isSafeInteger(blockEnd)
      || blockStart < 0 || blockStart >= blockEnd || blockEnd > totalBlocks) {
    throw new RangeError('ViT block range must identify an exact remaining interval');
  }
  if (blockCount !== blockEnd - blockStart) {
    throw new RangeError('ViT block range count must equal its exact interval');
  }
  const effectiveMode = normalizeVitMicrodutyMode(mode);
  if (workload !== null) {
    if (!workload || typeof workload !== 'object') {
      throw new TypeError('ViT dispatch tile workload must be an object');
    }
    for (const field of ['tokenCount', 'dim', 'mlpHiddenDim', 'numHeads']) {
      if (!Number.isSafeInteger(workload[field]) || workload[field] <= 0) {
        throw new RangeError(`ViT dispatch tile workload ${field} must be a positive safe integer`);
      }
    }
    for (const field of ['linearTileItems', 'attentionTileItems', 'softmaxTileRows', 'normTileRows']) {
      if (!Number.isSafeInteger(workload[field]) || workload[field] < 0) {
        throw new RangeError(`ViT dispatch tile workload ${field} must be a non-negative safe integer`);
      }
    }
  }
  const phases = effectiveMode === 'dispatch-major'
    ? [
        'norm1',
        'qkv-projection',
        'qkv-split',
        'attention-scores',
        'attention-softmax',
        'attention-apply',
        'attention-projection',
        'attention-residual',
        'norm2',
        'fc1',
        'fc2',
        'mlp-residual',
      ]
    : effectiveMode === 'four-stage'
      ? ['norm1-qkv', 'attention-projection-residual', 'norm2-fc1', 'fc2-residual']
    : effectiveMode === 'split-attention'
      ? ['norm1-qkv', 'attention-projection-residual', 'mlp-residual']
      : effectiveMode === 'split-mlp'
        ? ['attention-residual', 'norm2-fc1', 'fc2-residual']
        : ['attention-residual', 'mlp-residual'];
  const duties = [];
  for (let blockIndex = blockStart; blockIndex < blockEnd; blockIndex++) {
    const blockPhases = effectiveMode === 'dispatch-major' && workload?.normTileRows > 0
      && blockIndex === totalBlocks - 1
      ? [...phases, 'final-norm']
      : phases;
    for (const microphase of blockPhases) {
      const tiles = effectiveMode === 'dispatch-major' && workload
        ? planVitMicrophaseTiles(microphase, workload)
        : null;
      for (const tile of tiles || [null]) {
        duties.push({
          microdutyIndex: duties.length,
          blockIndex,
          microphase,
          ...(tile || {}),
        });
      }
    }
  }
  return duties;
}

export function parseSharpSchedulerConfig(options = {}) {
  const payload = queryPayload(options);
  const requested = { ...DEFAULT_SCHEDULER, ...payload };
  const unsupportedFields = Object.keys(requested)
    .filter(key => !SUPPORTED_FIELDS.has(key) && requested[key] !== undefined && requested[key] !== null)
    .sort();
  const preset = MODE_PRESETS[String(requested.mode || DEFAULT_SCHEDULER.mode)] || {};
  const hasPayload = key => Object.prototype.hasOwnProperty.call(payload, key);
  const fieldValue = key => hasPayload(key) ? requested[key] : (Object.prototype.hasOwnProperty.call(preset, key) ? preset[key] : DEFAULT_SCHEDULER[key]);

  const effective = {
    mode: String(requested.mode || DEFAULT_SCHEDULER.mode),
    spnPatchChunkSize: normalizeInt(fieldValue('spnPatchChunkSize'), DEFAULT_SCHEDULER.spnPatchChunkSize, { min: 1, max: 35 }),
    spnFusionChunkItems: normalizeInt(fieldValue('spnFusionChunkItems'), DEFAULT_SCHEDULER.spnFusionChunkItems, { min: 0 }),
    yieldMs: normalizeInt(fieldValue('yieldMs'), DEFAULT_SCHEDULER.yieldMs, { min: 0 }),
    waitForSubmittedWorkDone: normalizeBool(fieldValue('waitForSubmittedWorkDone'), DEFAULT_SCHEDULER.waitForSubmittedWorkDone),
    gaussianPhaseYieldMs: normalizeInt(fieldValue('gaussianPhaseYieldMs'), DEFAULT_SCHEDULER.gaussianPhaseYieldMs, { min: 0 }),
    vitBlockChunkSize: fieldValue('vitBlockChunkSize') === null || fieldValue('vitBlockChunkSize') === undefined
      ? DEFAULT_SCHEDULER.vitBlockChunkSize
      : normalizeInt(fieldValue('vitBlockChunkSize'), DEFAULT_SCHEDULER.vitBlockChunkSize, { min: 1 }),
    vitMicroduty: normalizeBool(fieldValue('vitMicroduty'), DEFAULT_SCHEDULER.vitMicroduty),
    vitMicrodutyMode: normalizeVitMicrodutyMode(fieldValue('vitMicrodutyMode')),
    vitLinearTileItems: normalizeInt(fieldValue('vitLinearTileItems'), DEFAULT_SCHEDULER.vitLinearTileItems, { min: 0 }),
    vitAttentionTileItems: normalizeInt(fieldValue('vitAttentionTileItems'), DEFAULT_SCHEDULER.vitAttentionTileItems, { min: 0 }),
    vitSoftmaxTileRows: normalizeInt(fieldValue('vitSoftmaxTileRows'), DEFAULT_SCHEDULER.vitSoftmaxTileRows, { min: 0 }),
    vitNormTileRows: normalizeInt(fieldValue('vitNormTileRows'), DEFAULT_SCHEDULER.vitNormTileRows, { min: 0 }),
    decoderKernelChunkItems: normalizeInt(fieldValue('decoderKernelChunkItems'), DEFAULT_SCHEDULER.decoderKernelChunkItems, { min: 0 }),
    routeTailYieldMs: normalizeInt(fieldValue('routeTailYieldMs'), DEFAULT_SCHEDULER.routeTailYieldMs, { min: 0 }),
    cpuChunkItems: normalizeInt(fieldValue('cpuChunkItems'), DEFAULT_SCHEDULER.cpuChunkItems, { min: 0 }),
    plyAssemblyMode: normalizePlyAssemblyMode(fieldValue('plyAssemblyMode')),
    retirePostInferenceBuffers: normalizeRetirementBool(fieldValue('retirePostInferenceBuffers'), DEFAULT_SCHEDULER.retirePostInferenceBuffers),
  };

  if ((effective.vitLinearTileItems > 0 || effective.vitAttentionTileItems > 0 || effective.vitSoftmaxTileRows > 0 || effective.vitNormTileRows > 0)
      && (!effective.vitMicroduty || effective.vitMicrodutyMode !== 'dispatch-major')) {
    throw new RangeError('ViT dispatch tiling requires vitMicroduty=true and vitMicrodutyMode=dispatch-major');
  }

  return {
    schema: 'sharp-webgpu.scheduler-config.v0',
    requested: Object.fromEntries(Object.entries(requested).map(([key, value]) => [
      key,
      INT_FIELDS.has(key) && value !== null
        ? normalizeInt(value, DEFAULT_SCHEDULER[key] ?? null, { min: key === 'vitBlockChunkSize' ? 1 : 0 })
        : value,
    ])),
    effective,
    unsupportedFields,
  };
}

export function createSharpRuntimeDutyMap({ generatedAt = new Date().toISOString() } = {}) {
  return {
    schema: 'sharp-webgpu.background-duty-map.v0',
    generatedAt,
    objective: 'continuous-background-inference-with-visible-materialization',
    steps: [
      {
        id: 'spn.patch-final-token-readback',
        stage: 'spn',
        step: 'patch-final-token-readback',
        kind: 'gpu-readback',
        syncClass: 'midstream-sync',
        requiredFor: 'spn-feature-merge',
        nextAction: 'keep-gpu-resident-or-batch-readback',
      },
      {
        id: 'spn.patch-intermediate-feature-readback',
        stage: 'spn',
        step: 'patch-intermediate-feature-readback',
        kind: 'gpu-readback',
        syncClass: 'midstream-sync',
        requiredFor: 'spn-feature-merge',
        nextAction: 'keep-gpu-resident-or-batch-readback',
      },
      {
        id: 'spn.lowres-fusion-readback',
        stage: 'spn',
        step: 'lowres-fusion-readback',
        kind: 'gpu-readback',
        syncClass: 'midstream-sync',
        requiredFor: 'lowres-concat-upload',
        nextAction: 'move-concat-to-gpu',
      },
      {
        id: 'gaussian.initializer-disparity-readback',
        stage: 'gaussian-decoder',
        step: 'initializer-disparity-readback',
        kind: 'gpu-readback',
        syncClass: 'midstream-sync',
        requiredFor: 'feature-input-construction',
        nextAction: 'move-to-gpu',
      },
      {
        id: 'output-capture.disparity-readback',
        stage: 'output-capture',
        step: 'disparity-readback',
        kind: 'gpu-readback',
        syncClass: 'preview-materialization',
        requiredFor: 'depth-preview-and-receipt',
        productHandling: 'materialize-visibly',
        nextAction: 'defer-or-preview-chunk',
      },
      {
        id: 'output-capture.depth-preview-render',
        stage: 'output-capture',
        step: 'depth-preview-render',
        kind: 'cpu-loop',
        syncClass: 'preview-materialization',
        requiredFor: 'operator-preview',
        productHandling: 'materialize-visibly',
        nextAction: 'chunk-on-main-or-worker',
      },
      {
        id: 'compose-ply.geometry-delta-readback',
        stage: 'compose-ply',
        step: 'geometry-delta-readback',
        kind: 'gpu-readback',
        syncClass: 'final-materialization',
        requiredFor: 'ply-composition',
        productHandling: 'materialize-visibly',
        nextAction: 'defer-export-or-stream-readback',
      },
      {
        id: 'compose-ply.texture-delta-readback',
        stage: 'compose-ply',
        step: 'texture-delta-readback',
        kind: 'gpu-readback',
        syncClass: 'final-materialization',
        requiredFor: 'ply-composition',
        productHandling: 'materialize-visibly',
        nextAction: 'defer-export-or-stream-readback',
      },
      {
        id: 'compose-ply.compose-export',
        stage: 'compose-ply',
        step: 'compose-export',
        kind: 'cpu-loop',
        syncClass: 'final-materialization',
        requiredFor: 'ply-output',
        productHandling: 'materialize-visibly',
        nextAction: 'chunk-on-main-or-worker',
      },
      ...[
        'depth-normalize',
        'depth-min',
        'depth-rescale',
        'base-disparity',
        'base-grid',
        'base-color',
      ].map(step => ({
        id: `compose-ply.${step}`,
        stage: 'compose-ply',
        step,
        kind: 'cpu-preparation-phase',
        syncClass: 'final-materialization',
        requiredFor: 'gaussian-composition',
        productHandling: 'measure-before-smaller-duty-smoke',
        nextAction: 'split-only-if-live-gap-overlaps',
      })),
      {
        id: 'compose-ply.ply-data-allocation',
        stage: 'compose-ply',
        step: 'ply-data-allocation',
        kind: 'typed-array-allocation',
        syncClass: 'final-materialization',
        requiredFor: 'gaussian-composition-output',
        productHandling: 'measure-before-segmented-buffer-surgery',
        nextAction: 'segment-allocation-if-live-gap-overlaps',
      },
      {
        id: 'compose-ply.gaussian-activation-setup',
        stage: 'compose-ply',
        step: 'gaussian-activation-setup',
        kind: 'cpu-setup',
        syncClass: 'final-materialization',
        requiredFor: 'gaussian-composition',
        productHandling: 'measure-before-surgery',
        nextAction: 'retain-if-negligible',
      },
      {
        id: 'compose-ply.ply-blob-assembly',
        stage: 'compose-ply',
        step: 'ply-blob-assembly',
        kind: 'browser-blob-assembly',
        syncClass: 'final-materialization',
        requiredFor: 'ply-output',
        productHandling: 'materialize-visibly',
        nextAction: 'measure-before-worker-or-streaming-surgery',
      },
      {
        id: 'compose-ply.object-url-create',
        stage: 'compose-ply',
        step: 'object-url-create',
        kind: 'browser-object-url',
        syncClass: 'final-materialization',
        requiredFor: 'ply-download',
        productHandling: 'materialize-visibly',
        nextAction: 'measure-before-deferral',
      },
      {
        id: 'compose-ply.output-bind',
        stage: 'compose-ply',
        step: 'output-bind',
        kind: 'dom-bind',
        syncClass: 'final-materialization',
        requiredFor: 'ply-download-action',
        productHandling: 'materialize-visibly',
        nextAction: 'measure-before-deferral',
      },
      {
        id: 'compose-ply.inference-window-finalize',
        stage: 'compose-ply',
        step: 'inference-window-finalize',
        kind: 'localization-envelope',
        syncClass: 'final-materialization',
        requiredFor: 'contention-window-closure',
        productHandling: 'measure-without-operation-attribution',
        nextAction: 'split-only-after-envelope-localizes-a-stall',
      },
    ],
  };
}

export function createSharpRunTelemetry(scheduler, context = {}) {
  const telemetry = {
    schema: 'sharp-webgpu.scheduler-telemetry.v0',
    status: 'scheduler-unverified',
    runId: context.runId || `sharp-webgpu-${Date.now().toString(36)}`,
    startedAt: new Date().toISOString(),
    completedAt: null,
    requestedScheduler: { ...scheduler.requested },
    effectiveScheduler: { ...scheduler.effective },
    unsupportedFields: [...scheduler.unsupportedFields],
    eventTrace: createEventTrace(),
    boundaryAssertions: [],
    events: [],
  };
  Object.defineProperty(telemetry, '_nextDutySequence', {
    value: 0,
    writable: true,
    enumerable: false,
  });
  return telemetry;
}

export function recordSchedulerEvent(telemetry, phase, details = {}) {
  if (!telemetry) return null;
  if (!telemetry.eventTrace) {
    telemetry.eventTrace = createEventTrace();
  }
  if (!telemetry.eventTrace.clock || typeof telemetry.eventTrace.clock !== 'object') {
    telemetry.eventTrace.clock = createEventClock();
  }
  const tMs = Number(nowMs().toFixed(3));
  const timeOriginMs = telemetry.eventTrace.clock.timeOriginEpochMs;
  const event = {
    phase,
    boundary: details.boundary || boundaryForPhase(phase),
    kind: details.kind || 'boundary-event',
    ...details,
    runId: telemetry.runId,
    tMs,
    epochMs: Number((timeOriginMs + tMs).toFixed(3)),
  };
  if (Number.isFinite(details.intervalStartMs) && Number.isFinite(details.intervalEndMs)) {
    event.intervalStartEpochMs = Number((timeOriginMs + details.intervalStartMs).toFixed(3));
    event.intervalEndEpochMs = Number((timeOriginMs + details.intervalEndMs).toFixed(3));
  }
  telemetry.eventTrace.events.push(event);
  telemetry.eventTrace.timingAuthority = 'browser-wall-clock';
  telemetry.events = telemetry.eventTrace.events;
  return event;
}

function cloneTelemetryDetail(value) {
  if (Array.isArray(value)) return value.map(cloneTelemetryDetail);
  if (!value || typeof value !== 'object') return value;
  const clone = {};
  for (const [key, child] of Object.entries(value)) clone[key] = cloneTelemetryDetail(child);
  return clone;
}

function cloneTelemetryEvent(event) {
  const clone = { ...event };
  for (const [key, value] of Object.entries(event)) {
    if (value && typeof value === 'object') clone[key] = cloneTelemetryDetail(value);
  }
  return clone;
}

function ensureTelemetryEventTrace(telemetry) {
  if (!telemetry.eventTrace) {
    telemetry.eventTrace = createEventTrace(Array.isArray(telemetry.events) ? telemetry.events : []);
  }
  if (!telemetry.eventTrace.clock || typeof telemetry.eventTrace.clock !== 'object') {
    telemetry.eventTrace.clock = createEventClock();
  }
  telemetry.events = telemetry.eventTrace.events;
}

function finalizeTelemetrySnapshotState(telemetry, status, eventCountIndex = null) {
  telemetry.boundaryAssertions = requestedBoundaryAssertions(telemetry, eventCountIndex);
  telemetry.status = derivedTelemetryStatus(telemetry, status);
  if (status !== 'running' && !telemetry.completedAt) telemetry.completedAt = new Date().toISOString();
}

function assembleTelemetrySnapshot(telemetry, events, snapshotProcess = null) {
  return {
    ...telemetry,
    requestedScheduler: cloneTelemetryDetail(telemetry.requestedScheduler),
    effectiveScheduler: cloneTelemetryDetail(telemetry.effectiveScheduler),
    unsupportedFields: [...telemetry.unsupportedFields],
    eventTrace: {
      ...telemetry.eventTrace,
      clock: cloneTelemetryDetail(telemetry.eventTrace.clock),
      events,
    },
    boundaryAssertions: telemetry.boundaryAssertions.map(cloneTelemetryEvent),
    events,
    ...(snapshotProcess ? { snapshotProcess } : {}),
  };
}

export function schedulerTelemetrySnapshot(telemetry, status = telemetry?.status || 'verified') {
  if (!telemetry) return null;
  ensureTelemetryEventTrace(telemetry);
  finalizeTelemetrySnapshotState(telemetry, status);
  const events = telemetry.eventTrace.events.map(cloneTelemetryEvent);
  return assembleTelemetrySnapshot(telemetry, events, {
    schema: 'sharp-webgpu.scheduler-snapshot-process.v0',
    mode: 'synchronous',
    sourceEventCount: events.length,
    chunkEvents: events.length,
    taskYieldCount: 0,
  });
}

export async function schedulerTelemetrySnapshotCooperatively(
  telemetry,
  status = telemetry?.status || 'verified',
  options = {},
) {
  if (!telemetry) return null;
  ensureTelemetryEventTrace(telemetry);
  const sourceEvents = telemetry.eventTrace.events;
  const sourceEventCount = sourceEvents.length;
  const requestedChunkEvents = Number(options.chunkEvents);
  const chunkEvents = Number.isFinite(requestedChunkEvents) && requestedChunkEvents > 0
    ? Math.floor(requestedChunkEvents)
    : 512;
  const taskYield = typeof options.taskYield === 'function'
    ? options.taskYield
    : () => new Promise(resolve => setTimeout(resolve, 0));
  const events = new Array(sourceEventCount);
  const eventCountIndex = new Map();
  let taskYieldCount = 0;
  for (let start = 0; start < sourceEventCount; start += chunkEvents) {
    const end = Math.min(sourceEventCount, start + chunkEvents);
    for (let index = start; index < end; index += 1) {
      const event = sourceEvents[index];
      events[index] = cloneTelemetryEvent(event);
      recordSchedulerEventCount(eventCountIndex, event);
    }
    if (end < sourceEventCount) {
      taskYieldCount += 1;
      await taskYield({ startEvent: start, endEvent: end, sourceEventCount });
    }
  }
  finalizeTelemetrySnapshotState(telemetry, status, eventCountIndex);
  return assembleTelemetrySnapshot(telemetry, events, {
    schema: 'sharp-webgpu.scheduler-snapshot-process.v0',
    mode: 'cooperative-fixed-prefix',
    sourceEventCount,
    chunkEvents,
    taskYieldCount,
  });
}

export async function schedulerYield(scheduler, device, telemetry, phase, details = {}, yieldMsOverride = null) {
  const effective = scheduler?.effective || DEFAULT_SCHEDULER;
  const defaultYieldMs = phase === 'route-tail'
    ? (effective.routeTailYieldMs ?? effective.yieldMs ?? 0)
    : (effective.yieldMs ?? 0);
  const yieldMs = yieldMsOverride ?? defaultYieldMs;
  const boundary = boundaryForPhase(phase);
  const startedAtMs = nowMs();
  let waitedForSubmittedWorkDone = false;
  const dutyId = nextSchedulerDutyId(telemetry, boundary);
  await serviceLiveForegroundOpportunity({
    scheduler,
    telemetry,
    phase,
    boundary,
    dutyId,
    details,
  });
  recordSchedulerEvent(telemetry, phase, {
    ...details,
    boundary,
    kind: 'chunk-start',
  });
  if (effective.waitForSubmittedWorkDone && device?.queue?.onSubmittedWorkDone) {
    const queueStartMs = nowMs();
    recordSchedulerEvent(telemetry, phase, {
      ...details,
      boundary,
      kind: 'queue-work-done-start',
      dutyId,
    });
    await device.queue.onSubmittedWorkDone();
    const queueEndMs = nowMs();
    recordSchedulerEvent(telemetry, phase, {
      ...details,
      boundary,
      kind: 'queue-work-done-end',
      dutyId,
      queueDoneMs: Number((queueEndMs - queueStartMs).toFixed(3)),
    });
    waitedForSubmittedWorkDone = true;
  }
  recordSchedulerEvent(telemetry, phase, {
    ...details,
    boundary,
    kind: 'js-yield-start',
    yieldMs,
  });
  await new Promise(resolve => setTimeout(resolve, yieldMs));
  recordSchedulerEvent(telemetry, phase, {
    ...details,
    boundary,
    kind: 'js-yield-end',
    yieldMs,
  });
  const endedAtMs = nowMs();
  recordSchedulerEvent(telemetry, phase, {
    ...details,
    boundary,
    kind: 'chunk-end',
    yieldMs,
    waitedForSubmittedWorkDone,
    durationMs: Number((endedAtMs - startedAtMs).toFixed(3)),
  });
  scheduler?.progressReporter?.({
    phase,
    boundary,
    details: { ...details },
    timestampMs: endedAtMs,
  });
}
