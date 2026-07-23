import { planDecoderKernelChunks } from './scheduler.js';
import {
  dispatchConv1x1,
  dispatchConv2d,
  dispatchConvTranspose2d,
  dispatchGroupNormNormalizeRelu,
  dispatchGroupNormPartialStats,
  dispatchGroupNormReduceStats,
} from './shader_ops.js';

const GROUPNORM_PARTIAL_ELEMENTS = 4096;

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

export async function dispatchTiledConv1x1({
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
  const totalOutputItems = positiveSafeProduct([params.outC, params.H, params.W], 'conv1x1');
  return dispatchKernelTiles({
    device,
    totalOutputItems,
    chunkItems,
    phase,
    details,
    boundaryYield,
    encodeTile(encoder, tile, outputBuffer) {
      return dispatchConv1x1(device, encoder, inputBuf, weightBuf, biasBuf, {
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

export async function dispatchTiledGroupNormRelu({
  device,
  inputBuf,
  scaleBuf,
  biasBuf,
  params,
  chunkItems,
  phase,
  details = {},
  boundaryYield,
}) {
  if (!Number.isSafeInteger(chunkItems) || chunkItems <= 0) {
    throw new RangeError('parallel GroupNorm requires a positive safe-integer decoder chunk size');
  }
  if (!device?.queue || typeof device.createCommandEncoder !== 'function') {
    throw new TypeError('parallel GroupNorm requires a WebGPU device and queue');
  }
  if (typeof boundaryYield !== 'function') {
    throw new TypeError('parallel GroupNorm requires an awaitable boundaryYield callback');
  }
  const { C, H, W, numGroups, eps = 1e-5 } = params;
  if (!Number.isSafeInteger(numGroups) || numGroups <= 0 || C % numGroups !== 0) {
    throw new RangeError('parallel GroupNorm requires numGroups to divide C exactly');
  }
  const totalOutputItems = positiveSafeProduct([C, H, W], 'groupnorm');
  const groupSize = totalOutputItems / numGroups;
  const partialsPerGroup = Math.ceil(groupSize / GROUPNORM_PARTIAL_ELEMENTS);
  const totalPartials = positiveSafeProduct([numGroups, partialsPerGroup], 'groupnorm partials');
  const partialsPerTile = Math.max(1, Math.floor(chunkItems / GROUPNORM_PARTIAL_ELEMENTS));
  const partialTiles = planDecoderKernelChunks(totalPartials, partialsPerTile);
  let partialBuffer = null;

  for (const tile of partialTiles) {
    const encoder = device.createCommandEncoder();
    partialBuffer = dispatchGroupNormPartialStats(device, encoder, inputBuf, {
      C, H, W, numGroups,
      partialElements: GROUPNORM_PARTIAL_ELEMENTS,
      partialStart: tile.outputStart,
      partialCount: tile.outputCount,
      partialsPerGroup,
      totalPartials,
      partialBuffer,
    });
    device.queue.submit([encoder.finish()]);
    await boundaryYield(`${phase}-partial-stats`, {
      ...details,
      role: 'groupnorm-partial-stats-tile',
      configuredChunkItems: chunkItems,
      partialElements: GROUPNORM_PARTIAL_ELEMENTS,
      partialIndex: tile.tileIndex,
      partialTotal: tile.tileTotal,
      partialStart: tile.outputStart,
      partialEnd: tile.outputEnd,
      partialCount: tile.outputCount,
      totalPartials,
      totalOutputItems,
    });
  }

  const reductionEncoder = device.createCommandEncoder();
  const statsBuffer = dispatchGroupNormReduceStats(device, reductionEncoder, partialBuffer, {
    C, H, W, numGroups, partialsPerGroup, partialElements: GROUPNORM_PARTIAL_ELEMENTS,
  });
  device.queue.submit([reductionEncoder.finish()]);
  await boundaryYield(`${phase}-stats-reduction`, {
    ...details,
    role: 'groupnorm-stats-reduction',
    configuredChunkItems: chunkItems,
    numGroups,
    partialsPerGroup,
    totalPartials,
    totalOutputItems,
  });

  const outputTiles = planDecoderKernelChunks(totalOutputItems, chunkItems);
  let outputBuffer = null;
  for (const tile of outputTiles) {
    const encoder = device.createCommandEncoder();
    const nextOutputBuffer = dispatchGroupNormNormalizeRelu(
      device,
      encoder,
      inputBuf,
      scaleBuf,
      biasBuf,
      statsBuffer,
      {
        C, H, W, numGroups, eps,
        outputStart: tile.outputStart,
        outputCount: tile.outputCount,
        outputBuffer,
      },
    );
    if (outputBuffer && nextOutputBuffer !== outputBuffer) {
      throw new Error('GroupNorm normalization tiles must retain one shared output buffer');
    }
    outputBuffer = nextOutputBuffer;
    device.queue.submit([encoder.finish()]);
    await boundaryYield(`${phase}-normalize-relu`, {
      ...details,
      role: 'groupnorm-normalize-relu-tile',
      configuredChunkItems: chunkItems,
      tileIndex: tile.tileIndex,
      tileTotal: tile.tileTotal,
      outputStart: tile.outputStart,
      outputEnd: tile.outputEnd,
      outputCount: tile.outputCount,
      totalOutputItems,
      tileUnit: tile.tileUnit,
    });
  }

  return {
    buffer: outputBuffer,
    partialBuffer,
    statsBuffer,
    partialTileCount: partialTiles.length,
    normalizeTileCount: outputTiles.length,
    configuredChunkItems: chunkItems,
  };
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
