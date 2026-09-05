const GPU_TIMESTAMP_RANGE_SCHEMA = 'sharp-webgpu.gpu-timestamp-range.v0';
const QUERY_BYTE_LENGTH = 16;
const timestampResolutionUpperBoundNsByDevice = new WeakMap();

function parseNanoseconds(value) {
  return typeof value === 'string' && /^(0|[1-9]\d*)$/.test(value)
    ? BigInt(value)
    : null;
}

export function validateGpuTimestampRangeMeasurement(measurement) {
  const snapshot = {
    schema: measurement?.schema,
    authority: measurement?.authority,
    measurementStatus: measurement?.measurementStatus,
    startedAtNs: measurement?.startedAtNs,
    completedAtNs: measurement?.completedAtNs,
    rawDurationNs: measurement?.rawDurationNs,
    durationNs: measurement?.durationNs,
    durationMs: measurement?.durationMs,
    resolutionUpperBoundNs: measurement?.resolutionUpperBoundNs,
  };
  const startedAtNs = parseNanoseconds(snapshot.startedAtNs);
  const completedAtNs = parseNanoseconds(snapshot.completedAtNs);
  const rawDurationNs = parseNanoseconds(snapshot.rawDurationNs);
  const durationNs = parseNanoseconds(snapshot.durationNs);
  const resolutionUpperBoundNs = snapshot.resolutionUpperBoundNs === null
    ? null
    : parseNanoseconds(snapshot.resolutionUpperBoundNs);
  const durationMs = durationNs === null ? null : Number(durationNs) / 1_000_000;
  const observedPositiveRecord = snapshot.measurementStatus === 'observed-positive-range'
    && startedAtNs !== null
    && completedAtNs !== null
    && rawDurationNs !== null
    && durationNs !== null
    && completedAtNs > startedAtNs
    && rawDurationNs === completedAtNs - startedAtNs
    && durationNs === rawDurationNs
    && snapshot.resolutionUpperBoundNs === null;
  const resolutionCensoredRecord = snapshot.measurementStatus === 'resolution-censored-upper-bound'
    && startedAtNs !== null
    && completedAtNs !== null
    && rawDurationNs === 0n
    && durationNs !== null
    && durationNs > 0n
    && completedAtNs === startedAtNs
    && resolutionUpperBoundNs === durationNs;
  if (snapshot.schema !== GPU_TIMESTAMP_RANGE_SCHEMA
      || snapshot.authority !== 'timestamp-query-inside-submitted-command-buffer'
      || (!observedPositiveRecord && !resolutionCensoredRecord)
      || !Number.isFinite(snapshot.durationMs)
      || snapshot.durationMs <= 0
      || snapshot.durationMs !== durationMs) {
    throw new Error('invalid GPU timestamp-query measurement record');
  }
  return Object.freeze({
    schema: GPU_TIMESTAMP_RANGE_SCHEMA,
    authority: 'timestamp-query-inside-submitted-command-buffer',
    measurementStatus: snapshot.measurementStatus,
    startedAtNs: startedAtNs.toString(),
    completedAtNs: completedAtNs.toString(),
    rawDurationNs: rawDurationNs.toString(),
    durationNs: durationNs.toString(),
    durationMs,
    resolutionUpperBoundNs: resolutionUpperBoundNs?.toString() ?? null,
  });
}

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

  function begin() {
    if (state !== 'idle') throw new Error(`GPU timestamp timer cannot begin from ${state}`);
    state = 'encoding';
    return Object.freeze({
      timestampWrites: Object.freeze({
        querySet,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      }),
    });
  }

  function end(encoder) {
    if (state !== 'encoding') throw new Error(`GPU timestamp timer cannot end from ${state}`);
    if (typeof encoder?.resolveQuerySet !== 'function'
        || typeof encoder.copyBufferToBuffer !== 'function') {
      throw new TypeError('adaptive decoder command encoder lacks timestamp resolution support');
    }
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
      if (completedAtNs < startedAtNs) {
        throw new Error(`GPU timestamp range endpoints are reversed: ${startedAtNs} > ${completedAtNs}`);
      }
      const rawDurationNs = completedAtNs - startedAtNs;
      let durationNs = rawDurationNs;
      let measurementStatus = 'observed-positive-range';
      let resolutionUpperBoundNs = null;
      if (rawDurationNs === 0n) {
        resolutionUpperBoundNs = timestampResolutionUpperBoundNsByDevice.get(device) || null;
        if (resolutionUpperBoundNs === null) {
          throw new Error(
            `GPU timestamp range is zero without a prior positive same-device resolution bound: ${startedAtNs} == ${completedAtNs}`,
          );
        }
        durationNs = resolutionUpperBoundNs;
        measurementStatus = 'resolution-censored-upper-bound';
      } else {
        const priorUpperBoundNs = timestampResolutionUpperBoundNsByDevice.get(device);
        if (priorUpperBoundNs === undefined || durationNs < priorUpperBoundNs) {
          timestampResolutionUpperBoundNsByDevice.set(device, durationNs);
        }
      }
      const durationMs = Number(durationNs) / 1_000_000;
      if (!Number.isFinite(durationMs) || durationMs <= 0) {
        throw new Error('GPU timestamp range produced an invalid duration');
      }
      return validateGpuTimestampRangeMeasurement({
        schema: GPU_TIMESTAMP_RANGE_SCHEMA,
        authority: 'timestamp-query-inside-submitted-command-buffer',
        measurementStatus,
        startedAtNs: startedAtNs.toString(),
        completedAtNs: completedAtNs.toString(),
        rawDurationNs: rawDurationNs.toString(),
        durationNs: durationNs.toString(),
        durationMs,
        resolutionUpperBoundNs: resolutionUpperBoundNs?.toString() || null,
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
