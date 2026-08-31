const GPU_TIMESTAMP_RANGE_SCHEMA = 'sharp-webgpu.gpu-timestamp-range.v0';
const QUERY_BYTE_LENGTH = 16;

function requireWebGpuConstant(group, name) {
  const value = globalThis[group]?.[name];
  if (!Number.isSafeInteger(value)) {
    throw new Error(`adaptive GPU timestamp timing requires ${group}.${name}`);
  }
  return value;
}

function destroyQuietly(resource) {
  if (typeof resource?.destroy !== 'function') return;
  try {
    resource.destroy();
  } catch {
    // Cleanup must not replace the timing failure that triggered it.
  }
}

export function createGpuTimestampRangeTimer(device, { label = 'sharp-adaptive-range' } = {}) {
  if (!device?.features?.has?.('timestamp-query')) {
    throw new Error('adaptive decoder GPU timing requires the timestamp-query device feature');
  }
  if (typeof device.createQuerySet !== 'function' || typeof device.createBuffer !== 'function') {
    throw new TypeError('adaptive decoder GPU timing requires query-set and buffer creation');
  }

  const queryResolve = requireWebGpuConstant('GPUBufferUsage', 'QUERY_RESOLVE');
  const copySource = requireWebGpuConstant('GPUBufferUsage', 'COPY_SRC');
  const copyDestination = requireWebGpuConstant('GPUBufferUsage', 'COPY_DST');
  const mapRead = requireWebGpuConstant('GPUBufferUsage', 'MAP_READ');
  const mapReadMode = requireWebGpuConstant('GPUMapMode', 'READ');

  let querySet = null;
  let resolveBuffer = null;
  let readBuffer = null;
  try {
    querySet = device.createQuerySet({
      type: 'timestamp',
      count: 2,
      label: `${label}:query-set`,
    });
    resolveBuffer = device.createBuffer({
      size: QUERY_BYTE_LENGTH,
      usage: queryResolve | copySource,
      label: `${label}:resolve-buffer`,
    });
    readBuffer = device.createBuffer({
      size: QUERY_BYTE_LENGTH,
      usage: copyDestination | mapRead,
      label: `${label}:read-buffer`,
    });
  } catch (error) {
    destroyQuietly(readBuffer);
    destroyQuietly(resolveBuffer);
    destroyQuietly(querySet);
    throw new Error('adaptive decoder GPU timestamp resource creation failed', { cause: error });
  }

  let state = 'idle';

  function begin(encoder) {
    if (state !== 'idle') throw new Error(`GPU timestamp timer cannot begin from ${state}`);
    if (typeof encoder?.writeTimestamp !== 'function') {
      throw new TypeError('adaptive decoder command encoder must support writeTimestamp');
    }
    encoder.writeTimestamp(querySet, 0);
    state = 'encoding';
  }

  function end(encoder) {
    if (state !== 'encoding') throw new Error(`GPU timestamp timer cannot end from ${state}`);
    if (typeof encoder?.writeTimestamp !== 'function'
        || typeof encoder.resolveQuerySet !== 'function'
        || typeof encoder.copyBufferToBuffer !== 'function') {
      throw new TypeError('adaptive decoder command encoder lacks timestamp resolution support');
    }
    encoder.writeTimestamp(querySet, 1);
    encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
    encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, QUERY_BYTE_LENGTH);
    state = 'pending-read';
  }

  async function read() {
    if (state !== 'pending-read') throw new Error(`GPU timestamp timer cannot read from ${state}`);
    if (typeof readBuffer?.mapAsync !== 'function'
        || typeof readBuffer.getMappedRange !== 'function'
        || typeof readBuffer.unmap !== 'function') {
      throw new TypeError('adaptive decoder GPU timestamp readback buffer is not mappable');
    }
    try {
      await readBuffer.mapAsync(mapReadMode, 0, QUERY_BYTE_LENGTH);
      const view = new DataView(readBuffer.getMappedRange(0, QUERY_BYTE_LENGTH));
      const startedAtNs = view.getBigUint64(0, true);
      const completedAtNs = view.getBigUint64(8, true);
      if (completedAtNs <= startedAtNs) {
        throw new Error('GPU timestamp range must have a strictly positive ordered duration');
      }
      const durationNs = completedAtNs - startedAtNs;
      const durationMs = Number(durationNs) / 1_000_000;
      if (!Number.isFinite(durationMs) || durationMs <= 0) {
        throw new Error('GPU timestamp range produced an invalid duration');
      }
      return Object.freeze({
        schema: GPU_TIMESTAMP_RANGE_SCHEMA,
        authority: 'timestamp-query-inside-submitted-command-buffer',
        startedAtNs: startedAtNs.toString(),
        completedAtNs: completedAtNs.toString(),
        durationNs: durationNs.toString(),
        durationMs,
      });
    } finally {
      try {
        readBuffer.unmap();
      } finally {
        state = 'idle';
      }
    }
  }

  function destroy() {
    if (state === 'destroyed') return;
    destroyQuietly(readBuffer);
    destroyQuietly(resolveBuffer);
    destroyQuietly(querySet);
    state = 'destroyed';
  }

  return Object.freeze({
    schema: GPU_TIMESTAMP_RANGE_SCHEMA,
    begin,
    end,
    read,
    destroy,
  });
}

export { GPU_TIMESTAMP_RANGE_SCHEMA };
