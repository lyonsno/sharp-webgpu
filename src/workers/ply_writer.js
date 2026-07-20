import { writePLY } from '../lib/compose.js';

self.onmessage = event => {
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
    self.postMessage({
      type: 'ply-assembled',
      requestId: request.requestId,
      plyBlob,
      bytes: plyBlob.size,
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
