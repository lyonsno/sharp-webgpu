/**
 * WebGPU initialization and device management.
 */

let deviceLost = false;

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
  const chunkBytes = Number.isFinite(options.chunkBytes)
    ? Math.max(1, Math.floor(options.chunkBytes))
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
