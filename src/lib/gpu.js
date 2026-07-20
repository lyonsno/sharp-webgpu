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

  // Request max limits for large model inference
  const device = await adapter.requestDevice({
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

  return { adapter, device };
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
export async function readBuffer(device, buffer, size) {
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return result;
}
