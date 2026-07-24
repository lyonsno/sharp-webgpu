import { writePLY } from '../lib/compose.js';

function digestHex(digest) {
  return Array.from(
    new Uint8Array(digest),
    byte => byte.toString(16).padStart(2, '0'),
  ).join('');
}

self.onmessage = async event => {
  const request = event?.data;
  if (request?.type !== 'assemble-ply') return;

  try {
    const plyData = new Float32Array(
      request.plyBuffer,
      request.plyByteOffset,
      request.plyLength,
    );
    const plyBlob = writePLY(
      plyData,
      request.numGaussians,
      request.imgW,
      request.imgH,
      request.focalPx,
    );
    const plySha256 = digestHex(
      await crypto.subtle.digest('SHA-256', await plyBlob.arrayBuffer()),
    );
    self.postMessage({
      type: 'ply-assembled',
      requestId: request.requestId,
      plyBlob,
      bytes: plyBlob.size,
      sha256: plySha256,
    });
  } catch (error) {
    self.postMessage({
      type: 'ply-error',
      requestId: request?.requestId,
      error: {
        name: error?.name || 'Error',
        message: error?.message || String(error),
      },
    });
  }
};
