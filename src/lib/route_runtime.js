import {
  SHARP_IMAGE_TO_SPLAT_ROUTE_ID,
  WEBGPU_INFERENCE_KIT_VERSION,
  createWebGpuInferenceRuntime,
  validateWebGpuRuntimeProfile,
} from '@kaminos/webgpu-inference-kit';

export const SHARP_ROUTE_RUNTIME_LABEL = 'sharp-webgpu-route-runtime';

const DEFAULT_KERNEL = {
  kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
  profile: 'spn-dinov2l16-monodepth-gaussian-ply',
  commit: 'sharp-webgpu-browser-runtime',
};

function featureNames(features) {
  return Array.from(features || []).map(String).sort();
}

function adapterName(gpu) {
  const info = gpu?.adapter?.info || {};
  return info.description || info.device || info.vendor || 'unknown-webgpu-adapter';
}

function routeRunId(options = {}) {
  return options.runId || `sharp-route-run-${Date.now().toString(36)}`;
}

function routeClock(options = {}) {
  return options.clock || {
    clockId: options.clockId || 'sharp-webgpu-performance-clock',
    source: 'performance.now',
    timeOriginEpochMs: globalThis.performance?.timeOrigin ?? Date.now(),
  };
}

function kitPhaseChunkSize(input = {}) {
  const phaseChunkSize = input.phaseChunkSize || {};
  return {
    spnPatch: input.spnPatch ?? input.spnPatchChunkSize ?? phaseChunkSize.spnPatch ?? phaseChunkSize.spnPatchChunkSize ?? 35,
    vitBlock: input.vitBlock ?? input.vitBlockChunkSize ?? phaseChunkSize.vitBlock ?? phaseChunkSize.vitBlockChunkSize ?? 24,
  };
}

function kitScheduler(input = {}) {
  return {
    mode: 'cooperative',
    yieldMs: input.yieldMs ?? 0,
    waitForSubmittedWorkDone: Boolean(input.waitForSubmittedWorkDone),
    phaseChunkSize: kitPhaseChunkSize(input),
  };
}

function kitIntegerRange(input, fallback) {
  return {
    min: input?.min ?? fallback.min,
    max: input?.max ?? fallback.max,
    stepFactor: input?.stepFactor ?? fallback.stepFactor,
  };
}

function kitSchedulerBounds(input = {}) {
  return {
    yieldMs: {
      min: input.yieldMs?.min ?? 0,
      max: input.yieldMs?.max ?? 1_000,
      step: input.yieldMs?.step ?? 1,
    },
    phaseChunkSize: {
      spnPatch: kitIntegerRange(input.phaseChunkSize?.spnPatch || input.spnPatch, {
        min: 1,
        max: 35,
        stepFactor: 2,
      }),
      vitBlock: kitIntegerRange(input.phaseChunkSize?.vitBlock || input.vitBlock, {
        min: 1,
        max: 24,
        stepFactor: 2,
      }),
    },
  };
}

export async function createSharpRouteRuntime(gpu, options = {}) {
  if (!gpu?.device) throw new Error('SHARP route runtime requires an existing WebGPU device');
  const routeDefinition = options.routeDefinition || {};
  const routeId = routeDefinition.routeId || SHARP_IMAGE_TO_SPLAT_ROUTE_ID;
  const runId = routeRunId(options);
  const clock = routeClock(options);
  const scheduler = kitScheduler(options.scheduler || routeDefinition.scheduler?.defaultScheduler || {});
  const schedulerBounds = kitSchedulerBounds(options.schedulerBounds || routeDefinition.scheduler?.bounds || {});
  const requiredStages = Array.isArray(routeDefinition.requiredStages)
    ? routeDefinition.requiredStages
    : ['spn', 'monodepth', 'gaussian-decoder', 'compose-ply', 'output-capture'];
  const deviceFeatures = featureNames(gpu.device.features);
  const adapterFeatures = featureNames(gpu.adapter?.features);
  const effectiveFeatures = deviceFeatures.length ? deviceFeatures : adapterFeatures;

  return createWebGpuInferenceRuntime({
    routeId,
    runtimeLabel: options.runtimeLabel || SHARP_ROUTE_RUNTIME_LABEL,
    device: gpu.device,
    adapter: gpu.adapter || null,
    adapterName: options.adapterName || adapterName(gpu),
    browser: options.browser || globalThis.navigator?.userAgent || null,
    requestedFeatures: options.requestedFeatures || [],
    effectiveFeatures: effectiveFeatures.length ? effectiveFeatures : ['webgpu-core'],
    limits: options.limits || gpu.device.limits || gpu.adapter?.limits || {},
    timestampQuery: options.timestampQuery || (effectiveFeatures.includes('timestamp-query') ? 'available' : 'unavailable'),
    requiredStages,
    timingSource: options.timingSource || routeDefinition.timingSource || 'host-stage-timer',
    kernel: {
      ...DEFAULT_KERNEL,
      ...(routeDefinition.kernel || {}),
      ...(options.kernel || {}),
    },
    waitForSubmittedWorkDone: options.waitForSubmittedWorkDone,
    yieldMs: options.yieldMs,
    now: options.now,
    hostPhases: {
      runId,
      clock,
    },
    commandDuties: {
      runId,
      clock,
    },
    schedulerApplication: {
      routeId,
      scheduler,
      bounds: schedulerBounds,
    },
    foregroundOpportunities: options.foregroundOpportunities || {
      runId,
    },
    evidence: options.evidence || {
      mode: 'live',
      source: 'sharp-webgpu-browser-route',
    },
  });
}

export function finishSharpRouteRuntimeProfile(runtime, options = {}) {
  if (!runtime || typeof runtime.finishProfile !== 'function') {
    throw new Error('SHARP route runtime profile requires a kit runtime');
  }
  const runtimeProfile = runtime.finishProfile({
    evidence: options.evidence || {
      mode: 'live',
      source: 'sharp-webgpu-browser-route',
    },
    createdAt: options.createdAt,
  });
  const result = validateWebGpuRuntimeProfile(runtimeProfile);
  if (!result.ok) {
    throw new Error(`invalid SHARP route runtime profile: ${result.errors.join('; ')}`);
  }
  return runtimeProfile;
}
