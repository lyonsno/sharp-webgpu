/**
 * WebGPU initialization and device management.
 */

let deviceLost = false;
const retiredGpuBuffers = new WeakSet();

function bufferRetirementFailure(report) {
  const error = new Error(`GPU buffer retirement failed for ${report.failedCount} target(s)`);
  error.name = 'GpuBufferRetirementError';
  error.retirementReport = report;
  return error;
}

/**
 * Destroy uniquely owned GPU buffers after their final submitted/readback use.
 * Allocation bytes are caller-known accounting, not observed physical release.
 */
export function retireGpuBuffers(targets, { enabled = false, requested = enabled } = {}) {
  if (!Array.isArray(targets)) {
    throw new TypeError('GPU buffer retirement targets must be an array');
  }

  const uniqueTargets = [];
  const seen = new Map();
  let aliasCount = 0;
  let knownAllocationBytes = 0;
  for (const [index, target] of targets.entries()) {
    if (!target || (typeof target.buffer !== 'object' && typeof target.buffer !== 'function')) {
      throw new TypeError(`GPU buffer retirement target ${index} has no buffer object`);
    }
    const bytes = target.bytes;
    if (!Number.isSafeInteger(bytes) || bytes < 0) {
      throw new TypeError(`GPU buffer retirement target ${index} has invalid known allocation bytes`);
    }
    const prior = seen.get(target.buffer);
    if (prior) {
      if (prior.bytes !== bytes) {
        throw new TypeError(`GPU buffer retirement aliases disagree on bytes: ${prior.label} and ${target.label || index}`);
      }
      aliasCount += 1;
      continue;
    }
    const normalized = {
      label: String(target.label || `buffer-${index}`),
      buffer: target.buffer,
      bytes,
    };
    seen.set(target.buffer, normalized);
    uniqueTargets.push(normalized);
    knownAllocationBytes += bytes;
  }

  const report = {
    schema: 'sharp.webgpu-buffer-retirement.v0',
    requested,
    effective: false,
    status: enabled ? 'running' : 'disabled',
    targetCount: targets.length,
    uniqueTargetCount: uniqueTargets.length,
    aliasCount,
    destroyedCount: 0,
    alreadyRetiredCount: 0,
    failedCount: 0,
    knownAllocationBytes,
    destroyedKnownAllocationBytes: 0,
    observedMemoryReleaseBytes: null,
    failures: [],
  };
  if (!enabled) return report;

  for (const target of uniqueTargets) {
    if (retiredGpuBuffers.has(target.buffer)) {
      report.alreadyRetiredCount += 1;
      continue;
    }
    let destroyBuffer = null;
    try {
      destroyBuffer = target.buffer.destroy;
      if (typeof destroyBuffer !== 'function') {
        throw new TypeError('buffer has no callable destroy capability');
      }
      Reflect.apply(destroyBuffer, target.buffer, []);
      retiredGpuBuffers.add(target.buffer);
      report.destroyedCount += 1;
      report.destroyedKnownAllocationBytes += target.bytes;
    } catch (error) {
      report.failedCount += 1;
      report.failures.push({
        label: target.label,
        name: error?.name || 'Error',
        message: error?.message || String(error),
      });
    }
  }

  if (report.failedCount) {
    report.status = 'failed';
    throw bufferRetirementFailure(report);
  }
  report.status = 'completed';
  report.effective = true;
  return report;
}

export const SHARP_LARGEST_STORAGE_BINDING_BYTES = 256 * 768 * 768 * Float32Array.BYTES_PER_ELEMENT;

export function validateSharpDeviceCapabilities(device) {
  const limits = device?.limits || {};
  const required = {
    maxBufferSize: SHARP_LARGEST_STORAGE_BINDING_BYTES,
    maxStorageBufferBindingSize: SHARP_LARGEST_STORAGE_BINDING_BYTES,
  };
  const effective = {
    maxBufferSize: Number(limits.maxBufferSize) || 0,
    maxStorageBufferBindingSize: Number(limits.maxStorageBufferBindingSize) || 0,
  };
  const insufficient = Object.keys(required).filter(name => effective[name] < required[name]);
  const report = {
    schema: 'sharp-webgpu.device-capability.v0',
    status: insufficient.length ? 'insufficient' : 'sufficient',
    required,
    effective,
    insufficient,
  };
  if (insufficient.length) {
    const error = new Error(`SHARP device capability insufficient: ${insufficient.join(', ')}`);
    error.deviceCapability = report;
    throw error;
  }
  return report;
}

export async function copyMappedBytesCooperatively(mappedBytes, options = {}) {
  if (!(mappedBytes instanceof Uint8Array)) {
    throw new TypeError('mapped readback copy requires Uint8Array source bytes');
  }
  const requestedChunkBytes = Number(options.chunkBytes);
  const chunkBytes = Number.isFinite(requestedChunkBytes) && requestedChunkBytes > 0
    ? Math.floor(requestedChunkBytes)
    : Math.max(1, mappedBytes.byteLength);
  const copiedBytes = new Uint8Array(mappedBytes.byteLength);
  for (let startByte = 0; startByte < mappedBytes.byteLength; startByte += chunkBytes) {
    const endByte = Math.min(mappedBytes.byteLength, startByte + chunkBytes);
    copiedBytes.set(mappedBytes.subarray(startByte, endByte), startByte);
    await options.onChunk?.({
      startByte,
      endByte,
      copiedBytes: endByte,
      totalBytes: mappedBytes.byteLength,
    });
  }
  return copiedBytes;
}

export function sharpOptionalDeviceFeatures(adapter) {
  return adapter?.features?.has?.('timestamp-query') ? ['timestamp-query'] : [];
}

export function validateAdaptiveDecoderDeviceCapabilities(device, scheduler) {
  const adaptiveDecoderEnabled = Number(scheduler?.effective?.decoderKernelTargetDurationMs) > 0;
  const timestampQuery = device?.features?.has?.('timestamp-query')
    ? 'available'
    : 'unavailable';
  const report = Object.freeze({ adaptiveDecoderEnabled, timestampQuery });
  if (adaptiveDecoderEnabled && timestampQuery !== 'available') {
    const error = new Error(
      'adaptive decoder scheduling requires the timestamp-query device feature',
    );
    error.adaptiveDecoderDeviceCapability = report;
    throw error;
  }
  return report;
}

export async function initGPU() {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported in this browser. Try Chrome 113+ or Edge 113+.');
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (!adapter) {
    throw new Error('No WebGPU adapter found. Your GPU may not support WebGPU.');
  }

  const requestedFeatures = sharpOptionalDeviceFeatures(adapter);

  // Request max limits for large model inference
  const device = await adapter.requestDevice({
    requiredFeatures: requestedFeatures,
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
    },
  });

  // Surface device loss as a visible error
  device.lost.then((info) => {
    deviceLost = true;
    const msg = `WebGPU device lost: ${info.message} (reason: ${info.reason})`;
    console.error(msg);
    // Surface in UI if error element exists
    const errorEl = document.getElementById('error');
    if (errorEl) {
      errorEl.textContent = msg;
      errorEl.style.display = 'block';
    }
  });

  // Surface uncaptured WebGPU validation/shader errors in UI
  device.addEventListener('uncapturederror', (event) => {
    const msg = `WebGPU error: ${event.error.message}`;
    console.error(msg);
    const errorEl = document.getElementById('error');
    if (errorEl) {
      errorEl.textContent = msg;
      errorEl.style.display = 'block';
    }
  });

  return { adapter, device, requestedFeatures };
}

/** Check if device is still alive. */
export function isDeviceLost() { return deviceLost; }

/**
 * Create a storage buffer initialized with data.
 */
export function createStorageBuffer(device, data, usage = 0) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | usage,
    mappedAtCreation: true,
  });
  new (data.constructor)(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

/**
 * Create an empty storage buffer.
 */
export function createEmptyBuffer(device, size, usage = 0) {
  return device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | usage,
    mappedAtCreation: false,
  });
}

/**
 * Read back buffer contents to CPU.
 */
export async function readBuffer(device, buffer, size, options = {}) {
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  try {
    const mappedBytes = new Uint8Array(staging.getMappedRange());
    const copiedBytes = await copyMappedBytesCooperatively(mappedBytes, options);
    return new Float32Array(copiedBytes.buffer);
  } finally {
    staging.unmap();
    staging.destroy();
  }
}
