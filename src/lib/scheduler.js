const DEFAULT_SCHEDULER = {
  mode: 'default',
  spnPatchChunkSize: 4,
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
  'yieldMs',
  'waitForSubmittedWorkDone',
  'gaussianPhaseYieldMs',
  'vitBlockChunkSize',
  'routeTailYieldMs',
  'cpuChunkItems',
]);

const INT_FIELDS = new Set(['spnPatchChunkSize', 'yieldMs', 'gaussianPhaseYieldMs', 'vitBlockChunkSize', 'routeTailYieldMs', 'cpuChunkItems']);
const EVENT_TRACE_SCHEMA = 'kaminos.webgpu-scheduler-event-trace.v0';
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

function boundaryForPhase(phase) {
  if (phase === 'spn-patch-chunk') return 'spn-patch-chunk';
  if (phase === 'spn-image-encoder') return 'spn-image-encoder';
  if (phase === 'spn-fusion') return 'spn-fusion';
  if (phase === 'gaussian-phase') return 'gaussian-phase';
  return phase || 'unknown';
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
  const phaseComplete = details.phaseComplete === true
    && details.stage === 'compose-ply'
    && COMPOSE_PHASE_COMPLETION_STEPS.has(details.step);
  return {
    eligible: Boolean(
      chunkItems
      && processedItems > 0
      && (phaseComplete || processedItems % chunkItems === 0)
    ),
    phaseComplete,
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
    ],
  };
}

export function createSharpRunTelemetry(scheduler, context = {}) {
  return {
    schema: 'sharp-webgpu.scheduler-telemetry.v0',
    status: 'scheduler-unverified',
    runId: context.runId || `sharp-webgpu-${Date.now().toString(36)}`,
    startedAt: new Date().toISOString(),
    completedAt: null,
    requestedScheduler: { ...scheduler.requested },
    effectiveScheduler: { ...scheduler.effective },
    unsupportedFields: [...scheduler.unsupportedFields],
    eventTrace: {
      schema: EVENT_TRACE_SCHEMA,
      clock: 'performance.now',
      timingAuthority: 'not-observed',
      events: [],
    },
    boundaryAssertions: [],
    events: [],
  };
}

export function recordSchedulerEvent(telemetry, phase, details = {}) {
  if (!telemetry) return null;
  if (!telemetry.eventTrace) {
    telemetry.eventTrace = {
      schema: EVENT_TRACE_SCHEMA,
      clock: 'performance.now',
      timingAuthority: 'not-observed',
      events: [],
    };
  }
  const event = {
    phase,
    boundary: details.boundary || boundaryForPhase(phase),
    kind: details.kind || 'boundary-event',
    tMs: Number(nowMs().toFixed(3)),
    ...details,
  };
  telemetry.eventTrace.events.push(event);
  telemetry.eventTrace.timingAuthority = 'browser-wall-clock';
  telemetry.events = telemetry.eventTrace.events;
  return event;
}

export function schedulerTelemetrySnapshot(telemetry, status = telemetry?.status || 'verified') {
  if (!telemetry) return null;
  if (!telemetry.eventTrace) {
    telemetry.eventTrace = {
      schema: EVENT_TRACE_SCHEMA,
      clock: 'performance.now',
      timingAuthority: telemetry.events?.length ? 'browser-wall-clock' : 'not-observed',
      events: Array.isArray(telemetry.events) ? telemetry.events : [],
    };
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
    });
    await device.queue.onSubmittedWorkDone();
    const queueEndMs = nowMs();
    recordSchedulerEvent(telemetry, phase, {
      ...details,
      boundary,
      kind: 'queue-work-done-end',
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
