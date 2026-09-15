// The source harness used 18 MiB payloads to stay below its observed CDP
// message ceiling. This controls message size, never total tensor size.
export const PARITY_CHUNK_BYTES = 18 * 1024 * 1024;

function checkChunkSize(value) {
  if (!Number.isSafeInteger(value) || value <= 0) throw new RangeError('chunkBytes must be a positive integer');
}

export async function compareStage(page, identity, reference, { chunkBytes = PARITY_CHUNK_BYTES, sampling, interior = false } = {}) {
  checkChunkSize(chunkBytes);
  if (!(reference instanceof Float32Array)) throw new TypeError('SHARP reference must be Float32Array');
  const bytes = Buffer.from(reference.buffer, reference.byteOffset, reference.byteLength);
  await page.evaluate(input => window.__sharpParity.beginReference(input), { ...identity, byteLength: bytes.length });
  try {
    for (let offset = 0; offset < bytes.length; offset += chunkBytes) {
      const payload = bytes.subarray(offset, offset + chunkBytes);
      const received = await page.evaluate(input => window.__sharpParity.appendReference(input), {
        ...identity, byteOffset: offset, payloadBase64: payload.toString('base64'),
      });
      if (received !== offset + payload.length) throw new Error('reference receiver byte count mismatch');
    }
    return await page.evaluate((input, options) => window.__sharpParity.compareReference(input, options), identity, { sampling, interior });
  } catch (error) {
    await page.evaluate(runId => {
      if (window.__sharpParity?.runId === runId) window.__sharpParity.discardReference();
    }, identity.runId).catch(() => {});
    throw error;
  }
}

export async function* readCaptureChunks(page, identity, byteLength, { chunkBytes = PARITY_CHUNK_BYTES } = {}) {
  checkChunkSize(chunkBytes);
  if (!Number.isSafeInteger(byteLength) || byteLength <= 0) throw new RangeError('capture byteLength must be positive');
  for (let offset = 0; offset < byteLength; offset += chunkBytes) {
    const length = Math.min(chunkBytes, byteLength - offset);
    const result = await page.evaluate(input => window.__sharpParity.readCaptureRange(input), {
      ...identity, byteOffset: offset, byteLength: length,
    });
    const bytes = Buffer.from(result.payloadBase64, 'base64');
    if (result.runId !== identity.runId || result.stageId !== identity.stageId
      || result.byteOffset !== offset || result.byteLength !== length || bytes.length !== length) {
      throw new Error('capture response identity or byte range mismatch');
    }
    yield bytes;
  }
}
