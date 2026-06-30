const DEFAULT_SCHEDULER = {
  mode: 'default',
  spnPatchChunkSize: 4,
  yieldMs: 0,
  waitForSubmittedWorkDone: false,
  gaussianPhaseYieldMs: 0,
  vitBlockChunkSize: null,
};

const SUPPORTED_FIELDS = new Set([
  'mode',
  'spnPatchChunkSize',
  'yieldMs',
  'waitForSubmittedWorkDone',
  'gaussianPhaseYieldMs',
]);

const INT_FIELDS = new Set(['spnPatchChunkSize', 'yieldMs', 'gaussianPhaseYieldMs', 'vitBlockChunkSize']);

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

function normalizeInt(value, fallback, { min = 0, max = 10000 } = {}) {
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

export function parseSharpSchedulerConfig(options = {}) {
  const requested = { ...DEFAULT_SCHEDULER, ...queryPayload(options) };
  const unsupportedFields = Object.keys(requested)
    .filter(key => !SUPPORTED_FIELDS.has(key) && requested[key] !== undefined && requested[key] !== null)
    .sort();

  const effective = {
    mode: String(requested.mode || DEFAULT_SCHEDULER.mode),
    spnPatchChunkSize: normalizeInt(requested.spnPatchChunkSize, DEFAULT_SCHEDULER.spnPatchChunkSize, { min: 1, max: 35 }),
    yieldMs: normalizeInt(requested.yieldMs, DEFAULT_SCHEDULER.yieldMs, { min: 0, max: 1000 }),
    waitForSubmittedWorkDone: normalizeBool(requested.waitForSubmittedWorkDone, DEFAULT_SCHEDULER.waitForSubmittedWorkDone),
    gaussianPhaseYieldMs: normalizeInt(requested.gaussianPhaseYieldMs, DEFAULT_SCHEDULER.gaussianPhaseYieldMs, { min: 0, max: 1000 }),
    vitBlockChunkSize: null,
  };

  return {
    schema: 'sharp-webgpu.scheduler-config.v0',
    requested: Object.fromEntries(Object.entries(requested).map(([key, value]) => [
      key,
      INT_FIELDS.has(key) && value !== null ? normalizeInt(value, DEFAULT_SCHEDULER[key] ?? null, { min: 0, max: 10000 }) : value,
    ])),
    effective,
    unsupportedFields,
  };
}

export function createSharpRunTelemetry(scheduler, context = {}) {
  return {
    schema: 'sharp-webgpu.scheduler-telemetry.v0',
    status: 'verified',
    runId: context.runId || `sharp-webgpu-${Date.now().toString(36)}`,
    startedAt: new Date().toISOString(),
    completedAt: null,
    requestedScheduler: { ...scheduler.requested },
    effectiveScheduler: { ...scheduler.effective },
    unsupportedFields: [...scheduler.unsupportedFields],
    events: [],
  };
}

export function recordSchedulerEvent(telemetry, phase, details = {}) {
  if (!telemetry) return null;
  const event = {
    phase,
    atMs: typeof performance !== 'undefined' ? performance.now() : Date.now(),
    ...details,
  };
  telemetry.events.push(event);
  return event;
}

export function schedulerTelemetrySnapshot(telemetry, status = telemetry?.status || 'verified') {
  if (!telemetry) return null;
  telemetry.status = status;
  if (status !== 'running' && !telemetry.completedAt) telemetry.completedAt = new Date().toISOString();
  return JSON.parse(JSON.stringify(telemetry));
}

export async function schedulerYield(scheduler, device, telemetry, phase, details = {}, yieldMsOverride = null) {
  const effective = scheduler?.effective || DEFAULT_SCHEDULER;
  const yieldMs = yieldMsOverride ?? effective.yieldMs ?? 0;
  const startedAtMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
  let waitedForSubmittedWorkDone = false;
  if (effective.waitForSubmittedWorkDone && device?.queue?.onSubmittedWorkDone) {
    await device.queue.onSubmittedWorkDone();
    waitedForSubmittedWorkDone = true;
  }
  await new Promise(resolve => setTimeout(resolve, yieldMs));
  const endedAtMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
  recordSchedulerEvent(telemetry, phase, {
    ...details,
    yieldMs,
    waitedForSubmittedWorkDone,
    durationMs: Number((endedAtMs - startedAtMs).toFixed(3)),
  });
}
