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

export async function createSharpRouteRuntime(gpu, options = {}) {
  if (!gpu?.device) throw new Error('SHARP route runtime requires an existing WebGPU device');
  const routeDefinition = options.routeDefinition || {};
  const requiredStages = Array.isArray(routeDefinition.requiredStages)
    ? routeDefinition.requiredStages
    : ['spn', 'monodepth', 'gaussian-decoder', 'compose-ply', 'output-capture'];
  const deviceFeatures = featureNames(gpu.device.features);
  const adapterFeatures = featureNames(gpu.adapter?.features);
  const effectiveFeatures = deviceFeatures.length ? deviceFeatures : adapterFeatures;

  return createWebGpuInferenceRuntime({
    routeId: routeDefinition.routeId || SHARP_IMAGE_TO_SPLAT_ROUTE_ID,
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
