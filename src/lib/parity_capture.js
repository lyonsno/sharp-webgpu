import { compareWebGpuParityArrays, createWebGpuParityCaptureRegistry } from '@kaminos/webgpu-inference-kit';

function interiorValues(values, [planes, height, width], border) {
  const result = new Float32Array(planes * (height - 2 * border) * (width - 2 * border));
  let offset = 0;
  for (let plane = 0; plane < planes; plane++) {
    for (let y = border; y < height - border; y++) {
      const row = (plane * height + y) * width;
      const part = values.subarray(row + border, row + width - border);
      result.set(part, offset);
      offset += part.length;
    }
  }
  return result;
}

function toBase64(bytes) {
  let binary = '';
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
}

// One invocation, one sequential reference transfer. The store owns captures;
// this adapter owns only the temporary reference received through CDP.
export function createSharpParitySession(runId) {
  const captures = createWebGpuParityCaptureRegistry({ runId });
  let reference = null;
  function checkRun(identity) {
    if (identity.runId !== runId) throw new Error('parity run identity mismatch');
  }
  function requireReference(identity) {
    checkRun(identity);
    if (!reference || reference.stageId !== identity.stageId) throw new Error('no matching reference transfer');
    return reference;
  }
  return Object.freeze({
    runId,
    capture: captures.capture,
    describe: captures.describe,
    stageIds: captures.stageIds,
    beginReference(identity) {
      checkRun(identity);
      const capture = captures.describe(identity.stageId);
      if (!capture) throw new Error(`missing capture: ${identity.stageId}`);
      if (capture.typedArrayConstructor !== 'Float32Array') throw new Error('SHARP reference requires Float32Array');
      if (identity.byteLength !== capture.byteLength) throw new Error('reference length must match capture');
      if (reference) throw new Error('previous reference transfer has not completed');
      reference = { stageId: identity.stageId, bytes: new Uint8Array(identity.byteLength), offset: 0 };
    },
    appendReference({ runId: incomingRun, stageId, byteOffset, payloadBase64 }) {
      const transfer = requireReference({ runId: incomingRun, stageId });
      if (byteOffset !== transfer.offset) throw new Error('reference byte offset must be contiguous');
      if (typeof payloadBase64 !== 'string') throw new TypeError('reference payload must be base64');
      const binary = atob(payloadBase64);
      if (binary.length === 0 || binary.length > transfer.bytes.length - transfer.offset) {
        throw new RangeError('reference chunk length exceeds remaining bytes or is empty');
      }
      for (let i = 0; i < binary.length; i++) transfer.bytes[transfer.offset + i] = binary.charCodeAt(i);
      transfer.offset += binary.length;
      return transfer.offset;
    },
    compareReference(identity, options = {}) {
      const transfer = requireReference(identity);
      if (transfer.offset !== transfer.bytes.length) throw new Error('reference transfer is incomplete');
      reference = null;
      const values = new Float32Array(transfer.bytes.buffer);
      const comparison = captures.compare(identity.stageId, values, options);
      if (!options.interior) return comparison;
      const description = captures.describe(identity.stageId);
      const shape = description.shape;
      const border = 8;
      if (shape?.length !== 3 || shape[1] <= 2 * border || shape[2] <= 2 * border) {
        throw new Error('interior comparison requires a CHW plane larger than its border');
      }
      const actual = new Float32Array(captures.readBytes(identity.stageId, {
        byteOffset: 0, byteLength: description.byteLength,
      }).buffer);
      return {
        ...comparison,
        interior: {
          borderPixels: border,
          comparison: compareWebGpuParityArrays(interiorValues(actual, shape, border), interiorValues(values, shape, border), {
            stageId: identity.stageId,
          }),
        },
      };
    },
    readCaptureRange(identity) {
      checkRun(identity);
      const { stageId, byteOffset, byteLength } = identity;
      const bytes = captures.readBytes(stageId, { byteOffset, byteLength });
      return { runId, stageId, byteOffset, byteLength: bytes.length, payloadBase64: toBase64(bytes) };
    },
    discardReference() { reference = null; },
    release: captures.release,
    clear() { reference = null; captures.clear(); },
  });
}
