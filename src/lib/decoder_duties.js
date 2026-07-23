import { planDecoderKernelChunks } from './scheduler.js';
import { dispatchConv2d, dispatchConvTranspose2d } from './shader_ops.js';

function positiveSafeProduct(values, label) {
  let product = 1;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value <= 0 || product > Number.MAX_SAFE_INTEGER / value) {
      throw new RangeError(`${label} dimensions must produce a positive safe-integer output size`);
    }
    product *= value;
  }
  return product;
}

async function dispatchKernelTiles({
  device,
  totalOutputItems,
  chunkItems,
  phase,
  details,
  boundaryYield,
  encodeTile,
}) {
  if (!device?.queue || typeof device.createCommandEncoder !== 'function') {
    throw new TypeError('decoder kernel tiling requires a WebGPU device and queue');
  }
  if (typeof boundaryYield !== 'function') {
    throw new TypeError('decoder kernel tiling requires an awaitable boundaryYield callback');
  }
  if (typeof phase !== 'string' || !phase) {
    throw new TypeError('decoder kernel tiling requires a non-empty telemetry phase');
  }

  const tiles = planDecoderKernelChunks(totalOutputItems, chunkItems);
  const tiled = chunkItems > 0;
  let result = null;
  let outputBuffer = null;

  for (const tile of tiles) {
    const encoder = device.createCommandEncoder();
    result = encodeTile(encoder, tiled ? tile : null, outputBuffer);
    if (!result?.buffer) {
      throw new TypeError('decoder kernel tile encoder must return its output buffer');
    }
    if (outputBuffer && result.buffer !== outputBuffer) {
      throw new Error('decoder kernel tiles must retain one shared output buffer');
    }
    outputBuffer = result.buffer;
    device.queue.submit([encoder.finish()]);
    await boundaryYield(phase, tiled
      ? {
          ...details,
          role: 'decoder-kernel-output-tile',
          configuredChunkItems: chunkItems,
          tileIndex: tile.tileIndex,
          tileTotal: tile.tileTotal,
          outputStart: tile.outputStart,
          outputEnd: tile.outputEnd,
          outputCount: tile.outputCount,
          totalOutputItems: tile.totalOutputItems,
          tileUnit: tile.tileUnit,
        }
      : details);
  }

  return {
    ...result,
    tileCount: tiles.length,
    configuredChunkItems: chunkItems,
  };
}

export async function dispatchTiledConv2d({
  device,
  inputBuf,
  weightBuf,
  biasBuf,
  params,
  chunkItems = 0,
  phase,
  details = {},
  boundaryYield,
  prepareInput = null,
}) {
  const outH = Math.floor((params.inH + 2 * params.padH - params.kH) / params.strideH) + 1;
  const outW = Math.floor((params.inW + 2 * params.padW - params.kW) / params.strideW) + 1;
  const totalOutputItems = positiveSafeProduct([params.outC, outH, outW], 'conv2d');
  let preparedInput = inputBuf;
  let inputPrepared = false;

  return dispatchKernelTiles({
    device,
    totalOutputItems,
    chunkItems,
    phase,
    details,
    boundaryYield,
    encodeTile(encoder, tile, outputBuffer) {
      if (!inputPrepared && prepareInput) {
        preparedInput = prepareInput(encoder, inputBuf);
        inputPrepared = true;
      }
      return dispatchConv2d(device, encoder, preparedInput, weightBuf, biasBuf, {
        ...params,
        ...(tile ? {
          outputStart: tile.outputStart,
          outputCount: tile.outputCount,
          outputBuffer,
        } : {}),
      });
    },
  });
}

export async function dispatchTiledConvTranspose2d({
  device,
  inputBuf,
  weightBuf,
  biasBuf,
  params,
  chunkItems = 0,
  phase,
  details = {},
  boundaryYield,
}) {
  const outH = params.inH * params.stride;
  const outW = params.inW * params.stride;
  const totalOutputItems = positiveSafeProduct([params.outC, outH, outW], 'conv-transpose');

  return dispatchKernelTiles({
    device,
    totalOutputItems,
    chunkItems,
    phase,
    details,
    boundaryYield,
    encodeTile(encoder, tile, outputBuffer) {
      return dispatchConvTranspose2d(device, encoder, inputBuf, weightBuf, biasBuf, {
        ...params,
        ...(tile ? {
          outputStart: tile.outputStart,
          outputCount: tile.outputCount,
          outputBuffer,
        } : {}),
      });
    },
  });
}
