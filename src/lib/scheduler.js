const DEFAULT_SCHEDULER = {
  mode: 'default',
  spnPatchChunkSize: 4,
  spnFusionChunkItems: 0,
  yieldMs: 0,
  waitForSubmittedWorkDone: false,
  gaussianPhaseYieldMs: 0,
  vitBlockChunkSize: null,
  routeTailYieldMs: 0,
  cpuChunkItems: 0,
};

const SUPPORTED_FIELDS = new Set([
  'mode',
  'spnPatchChunkSize',
  'spnFusionChunkItems',
  'yieldMs',
  'waitForSubmittedWorkDone',
  'gaussianPhaseYieldMs',
  'vitBlockChunkSize',
  'routeTailYieldMs',
  'cpuChunkItems',
]);

const INT_FIELDS = new Set(['spnPatchChunkSize', 'spnFusionChunkItems', 'yieldMs', 'gaussianPhaseYieldMs', 'vitBlockChunkSize', 'routeTailYieldMs', 'cpuChunkItems']);
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
    routeTailYieldMs: 16,
    cpuChunkItems: 65536,
  },
  furnace: {
    spnPatchChunkSize: 1,
    yieldMs: 16,
    waitForSubmittedWorkDone: true,
    gaussianPhaseYieldMs: 16,
    vitBlockChunkSize: 1,
    routeTailYieldMs: 16,
    cpuChunkItems: 65536,
  },
};

function nowMs() {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function timeOriginEpochMs() {
  if (typeof performance !== 'undefined' && Number.isFinite(performance.timeOrigin)) {
    return performance.timeOrigin;
  }
  return 0;
}

function createEventClock() {
  return {
    schema: EVENT_CLOCK_SCHEMA,
    relativeClock: typeof performance !== 'undefined' ? 'performance.now' : 'date-now',
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

function isForegroundOpportunityError(error) {
  return /foreground opportunity/i.test(error?.message || String(error));
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
  const snapshot = live.runtime.foregroundOpportunitySnapshot?.();
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
    if (optionalControlEnabled) throw error;
    return null;
  }
  let foregroundRequest = null;
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
    const duty = await live.runtime.prepareCommandDutyAtBoundary(dutyInput, live.invocation);
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
    if (foregroundRequest || isForegroundOpportunityError(error)) throw error;
    return null;
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

function requestedBoundaryAssertions(telemetry) {
  const requested = telemetry?.requestedScheduler || {};
  const effective = telemetry?.effectiveScheduler || {};
  const unsupportedFields = new Set(telemetry?.unsupportedFields || []);
  const events = telemetry?.eventTrace?.events || telemetry?.events || [];
  const queueWaitRequested = schedulerWaitRequested(requested, effective);
  const yieldRequested = schedulerYieldRequested(requested, effective);
  const gaussianYieldRequested = schedulerGaussianYieldRequested(requested, effective);
  const assertions = [];

  if (Number.isFinite(effective.spnPatchChunkSize) && effective.spnPatchChunkSize > 0) {
    const boundary = 'spn-patch-chunk';
    const observedCount = countEvents(events, boundary, 'chunk-start');
    const observedQueueWaitCount = Math.min(
      countEvents(events, boundary, 'queue-work-done-start'),
      countEvents(events, boundary, 'queue-work-done-end')
    );
    const observedYieldCount = Math.min(
      countEvents(events, boundary, 'js-yield-start'),
      countEvents(events, boundary, 'js-yield-end')
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

  if (Number.isFinite(effective.vitBlockChunkSize) && effective.vitBlockChunkSize > 0) {
    const boundary = 'vit-block-chunk';
    const unsupported = unsupportedFields.has('vitBlockChunkSize') || unsupportedFields.has('phaseChunkSize.vitBlock') || unsupportedFields.has('phaseChunkSize');
    const observedCount = countEvents(events, boundary, 'chunk-start');
    const observedQueueWaitCount = Math.min(
      countEvents(events, boundary, 'queue-work-done-start'),
      countEvents(events, boundary, 'queue-work-done-end')
    );
    const observedYieldCount = Math.min(
      countEvents(events, boundary, 'js-yield-start'),
      countEvents(events, boundary, 'js-yield-end')
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
    const observedCount = countEvents(events, boundary, 'chunk-start');
    const observedQueueWaitCount = Math.min(
      countEvents(events, boundary, 'queue-work-done-start'),
      countEvents(events, boundary, 'queue-work-done-end')
    );
    const observedYieldCount = Math.min(
      countEvents(events, boundary, 'js-yield-start'),
      countEvents(events, boundary, 'js-yield-end')
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
    const observedCount = countEventsMatching(
      events,
      boundary,
      'chunk-start',
      event => event?.role === 'cpu-materialization-chunk'
    );
    const observedQueueWaitCount = Math.min(
      countEventsMatching(events, boundary, 'queue-work-done-start', event => event?.role === 'cpu-materialization-chunk'),
      countEventsMatching(events, boundary, 'queue-work-done-end', event => event?.role === 'cpu-materialization-chunk')
    );
    const observedYieldCount = Math.min(
      countEventsMatching(events, boundary, 'js-yield-start', event => event?.role === 'cpu-materialization-chunk'),
      countEventsMatching(events, boundary, 'js-yield-end', event => event?.role === 'cpu-materialization-chunk')
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
    const observedCount = countEvents(events, boundary, 'chunk-start');
    const observedQueueWaitCount = Math.min(
      countEvents(events, boundary, 'queue-work-done-start'),
      countEvents(events, boundary, 'queue-work-done-end')
    );
    const observedYieldCount = Math.min(
      countEvents(events, boundary, 'js-yield-start'),
      countEvents(events, boundary, 'js-yield-end')
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
    routeTailYieldMs: normalizeInt(fieldValue('routeTailYieldMs'), DEFAULT_SCHEDULER.routeTailYieldMs, { min: 0 }),
    cpuChunkItems: normalizeInt(fieldValue('cpuChunkItems'), DEFAULT_SCHEDULER.cpuChunkItems, { min: 0 }),
  };

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

export function schedulerTelemetrySnapshot(telemetry, status = telemetry?.status || 'verified') {
  if (!telemetry) return null;
  if (!telemetry.eventTrace) {
    telemetry.eventTrace = createEventTrace(Array.isArray(telemetry.events) ? telemetry.events : []);
  }
  if (!telemetry.eventTrace.clock || typeof telemetry.eventTrace.clock !== 'object') {
    telemetry.eventTrace.clock = createEventClock();
  }
  telemetry.events = telemetry.eventTrace.events;
  telemetry.boundaryAssertions = requestedBoundaryAssertions(telemetry);
  telemetry.status = derivedTelemetryStatus(telemetry, status);
  if (status !== 'running' && !telemetry.completedAt) telemetry.completedAt = new Date().toISOString();
  return JSON.parse(JSON.stringify(telemetry));
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
}
