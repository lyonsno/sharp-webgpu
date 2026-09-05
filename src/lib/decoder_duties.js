import { createWebGpuAdaptiveCommandDutyPlanner } from '@kaminos/webgpu-inference-kit';
import {
  adaptiveDecoderTimingObservation,
  adaptiveDecoderObservationTelemetryDetails,
  captureQueueCompletionFence,
  captureQueueCompletionFencePair,
  planDecoderKernelChunks,
} from './scheduler.js';
import { createGpuTimestampRangeTimer } from './gpu_timestamp_range_timer.js';
export { createDecoderAdaptiveDuty } from './decoder_adaptive_profile.js';
import {
  dispatchConv1x1,
  dispatchConv2d,
  dispatchConvTranspose2d,
  dispatchGroupNormNormalizeRelu,
  dispatchGroupNormPartialStats,
  dispatchGroupNormReduceStats,
} from './shader_ops.js';

const GROUPNORM_PARTIAL_ELEMENTS = 4096;

function nowMs() {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function scaledAdaptiveDuty(adaptiveDuty, divisor) {
  if (!adaptiveDuty) return null;
  return {
    ...adaptiveDuty,
    unit: 'groupnorm-partial',
    initialChunkItems: Math.max(1, Math.floor(adaptiveDuty.initialChunkItems / divisor)),
    bounds: {
      minChunkItems: Math.max(1, Math.floor(adaptiveDuty.bounds.minChunkItems / divisor)),
      maxChunkItems: Math.max(1, Math.floor(adaptiveDuty.bounds.maxChunkItems / divisor)),
    },
  };
}

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
  adaptiveDuty = null,
  phase,
  details,
  rangeRole = 'decoder-kernel-output-tile',
  describeRange = null,
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

  const planner = adaptiveDuty
    ? createWebGpuAdaptiveCommandDutyPlanner({
        plannerId: adaptiveDuty.nextPlannerId(phase),
        unit: adaptiveDuty.unit || 'output-item',
        totalItems: totalOutputItems,
        initialChunkItems: adaptiveDuty.initialChunkItems,
        targetDurationMs: adaptiveDuty.targetDurationMs,
        adjustmentGain: adaptiveDuty.adjustmentGain,
        bounds: adaptiveDuty.bounds,
        retention: 'uncapped',
        metadata: { phase, ...details },
      })
    : null;
  const fixedTiles = planner ? null : planDecoderKernelChunks(totalOutputItems, chunkItems);
  const tiled = Boolean(planner) || chunkItems > 0;
  let result = null;
  let outputBuffer = null;
  let fixedTileIndex = 0;
  const gpuTimestampTimer = planner
    ? createGpuTimestampRangeTimer(device, { label: `sharp:${phase}:adaptive-range` })
    : null;

  try {
    while (true) {
    if (!planner && fixedTileIndex >= fixedTiles.length) break;
    const range = planner ? planner.nextRange() : null;
    if (planner && !range) break;
    const tile = range
      ? {
          tileIndex: range.rangeIndex,
          tileTotal: range.rangeTotal,
          outputStart: range.itemStart,
          outputEnd: range.itemEnd,
          outputCount: range.itemCount,
          totalOutputItems: range.totalItems,
          tileUnit: range.unit,
        }
      : fixedTiles[fixedTileIndex];
    fixedTileIndex += planner ? 0 : 1;
    let rangeObserved = false;

    try {
      const encoder = device.createCommandEncoder();
      result = encodeTile(
        encoder,
        tiled ? tile : null,
        outputBuffer,
        gpuTimestampTimer?.begin() || null,
      );
      if (!result?.buffer) {
        throw new TypeError('decoder kernel tile encoder must return its output buffer');
      }
      if (outputBuffer && result.buffer !== outputBuffer) {
        throw new Error('decoder kernel tiles must retain one shared output buffer');
      }
      outputBuffer = result.buffer;
      gpuTimestampTimer?.end(encoder);
      const preSubmitQueueCompletionFence = planner
        ? captureQueueCompletionFence(device)
        : null;
      const commandSubmittedAtMs = nowMs();
      device.queue.submit([encoder.finish()]);
      const queueCompletionFence = planner
        ? captureQueueCompletionFencePair(device, preSubmitQueueCompletionFence, commandSubmittedAtMs)
        : null;
      const yieldReceipt = await boundaryYield(phase, tiled
        ? {
            ...details,
            role: rangeRole,
            configuredChunkItems: chunkItems,
            adaptive: Boolean(planner),
            rangeId: range?.rangeId || null,
            rangeIndex: range?.rangeIndex ?? tile.tileIndex,
            rangeTotal: range ? range.rangeTotal : tile.tileTotal,
            rangeCountAuthority: range?.rangeCountAuthority || 'fixed-before-dispatch',
            plannedChunkItems: range?.plannedChunkItems ?? chunkItems,
            targetDurationMs: range?.targetDurationMs ?? null,
            bounds: range?.bounds ?? null,
            tileIndex: tile.tileIndex,
            tileTotal: tile.tileTotal,
            outputStart: tile.outputStart,
            outputEnd: tile.outputEnd,
            outputCount: tile.outputCount,
            totalOutputItems: tile.totalOutputItems,
            tileUnit: tile.tileUnit,
            commandSubmittedAtMs,
            requestedQueueTimingAuthority: planner ? 'gpu-timestamp-query' : null,
            ...(describeRange ? describeRange(tile, range) : {}),
          }
        : { ...details, commandSubmittedAtMs }, queueCompletionFence);

      if (planner) {
        const gpuTimestampRange = await gpuTimestampTimer.read();
        const timingObservation = adaptiveDecoderTimingObservation(yieldReceipt, gpuTimestampRange);
        const observation = planner.observeRange({
          rangeId: range.rangeId,
          observedDurationMs: timingObservation.observedDurationMs,
          timingAuthority: 'gpu-timestamp-query',
        });
        rangeObserved = true;
        adaptiveDuty.recordObservation(phase, {
          ...details,
          role: `${rangeRole}-observation`,
          rangeId: range.rangeId,
          rangeIndex: range.rangeIndex,
          rangeTotal: range.rangeTotal,
          rangeCountAuthority: observation.rangeCountAuthority,
          outputStart: range.itemStart,
          outputEnd: range.itemEnd,
          outputCount: range.itemCount,
          totalOutputItems: range.totalItems,
          timingAuthority: 'gpu-timestamp-query',
          queueWorkAttribution: timingObservation.queueWorkAttribution,
          rawSubmitToQueueDoneMs: timingObservation.rawSubmitToQueueDoneMs,
          prePrefixWallMs: timingObservation.prePrefixWallMs,
          hostFenceSettlementDeltaMs: timingObservation.hostFenceSettlementDeltaMs,
          incrementalSubmitToQueueDoneMs: timingObservation.incrementalSubmitToQueueDoneMs,
          gpuTimestampStartedAtNs: timingObservation.gpuTimestampStartedAtNs,
          gpuTimestampCompletedAtNs: timingObservation.gpuTimestampCompletedAtNs,
          gpuTimestampRawDurationNs: timingObservation.gpuTimestampRawDurationNs,
          gpuTimestampDurationNs: timingObservation.gpuTimestampDurationNs,
          gpuTimestampDurationMs: timingObservation.gpuTimestampDurationMs,
          gpuTimestampMeasurementStatus: timingObservation.gpuTimestampMeasurementStatus,
          gpuTimestampResolutionUpperBoundNs: timingObservation.gpuTimestampResolutionUpperBoundNs,
          requestedQueueTimingAuthority: timingObservation.requestedQueueTimingAuthority,
          effectiveQueueTimingAuthority: timingObservation.effectiveQueueTimingAuthority,
          queueTimingFallbackReason: timingObservation.queueTimingFallbackReason,
          foregroundServiceStatus: yieldReceipt.foregroundServiceStatus,
          ...adaptiveDecoderObservationTelemetryDetails(observation),
          completedItems: observation.completedItems,
          progress: observation.progress,
          actualRangeCount: observation.actualRangeCount,
          ...(describeRange ? describeRange(tile, range) : {}),
        });
      }
    } catch (error) {
      if (planner && !rangeObserved) {
        const failure = planner.failRange({ rangeId: range.rangeId, phase, error });
        adaptiveDuty.recordObservation(phase, {
          ...details,
          role: `${rangeRole}-failed`,
          rangeId: range.rangeId,
          rangeIndex: range.rangeIndex,
          rangeTotal: range.rangeTotal,
          rangeCountAuthority: failure.rangeCountAuthority,
          actualRangeCount: failure.actualRangeCount,
          failure: failure.failure,
        });
      }
      throw error;
    }
  }

    const plannerSnapshot = planner?.snapshot() || null;
    return {
      ...result,
      tileCount: plannerSnapshot?.actualRangeCount ?? fixedTiles.length,
      configuredChunkItems: chunkItems,
      adaptivePlanner: plannerSnapshot,
    };
  } finally {
    gpuTimestampTimer?.destroy();
  }
}

export async function dispatchTiledConv2d({
  device,
  inputBuf,
  weightBuf,
  biasBuf,
  params,
  chunkItems = 0,
  adaptiveDuty = null,
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
    adaptiveDuty,
    phase,
    details,
    boundaryYield,
    encodeTile(encoder, tile, outputBuffer, computePassDescriptor) {
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
          computePassDescriptor,
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
  adaptiveDuty = null,
  phase,
  details = {},
  boundaryYield,
}) {
  const totalOutputItems = positiveSafeProduct([params.outC, params.H, params.W], 'conv1x1');
  return dispatchKernelTiles({
    device,
    totalOutputItems,
    chunkItems,
    adaptiveDuty,
    phase,
    details,
    boundaryYield,
    encodeTile(encoder, tile, outputBuffer, computePassDescriptor) {
      return dispatchConv1x1(device, encoder, inputBuf, weightBuf, biasBuf, {
        ...params,
        ...(tile ? {
          outputStart: tile.outputStart,
          outputCount: tile.outputCount,
          outputBuffer,
          computePassDescriptor,
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
  adaptiveDuty = null,
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
  const partialResult = await dispatchKernelTiles({
    device,
    totalOutputItems: totalPartials,
    chunkItems: partialsPerTile,
    adaptiveDuty: scaledAdaptiveDuty(adaptiveDuty, GROUPNORM_PARTIAL_ELEMENTS),
    phase: `${phase}-partial-stats`,
    details: {
      ...details,
      configuredChunkItems: chunkItems,
      partialElements: GROUPNORM_PARTIAL_ELEMENTS,
      totalPartials,
      groupNormTotalOutputItems: totalOutputItems,
    },
    rangeRole: 'groupnorm-partial-stats-tile',
    describeRange: tile => ({
      partialIndex: tile.tileIndex,
      partialTotal: tile.tileTotal,
      partialStart: tile.outputStart,
      partialEnd: tile.outputEnd,
      partialCount: tile.outputCount,
    }),
    boundaryYield,
    encodeTile(encoder, tile, partialBuffer, computePassDescriptor) {
      return {
        buffer: dispatchGroupNormPartialStats(device, encoder, inputBuf, {
          C, H, W, numGroups,
          partialElements: GROUPNORM_PARTIAL_ELEMENTS,
          partialStart: tile.outputStart,
          partialCount: tile.outputCount,
          partialsPerGroup,
          totalPartials,
          partialBuffer,
          computePassDescriptor,
        }),
      };
    },
  });
  const partialBuffer = partialResult.buffer;

  const reductionEncoder = device.createCommandEncoder();
  const statsBuffer = dispatchGroupNormReduceStats(device, reductionEncoder, partialBuffer, {
    C, H, W, numGroups, partialsPerGroup, partialElements: GROUPNORM_PARTIAL_ELEMENTS,
  });
  const reductionSubmittedAtMs = nowMs();
  device.queue.submit([reductionEncoder.finish()]);
  await boundaryYield(`${phase}-stats-reduction`, {
    ...details,
    role: 'groupnorm-stats-reduction',
    configuredChunkItems: chunkItems,
    numGroups,
    partialsPerGroup,
    totalPartials,
    totalOutputItems,
    commandSubmittedAtMs: reductionSubmittedAtMs,
  });

  const normalizeResult = await dispatchKernelTiles({
    device,
    totalOutputItems,
    chunkItems,
    adaptiveDuty,
    phase: `${phase}-normalize-relu`,
    details,
    rangeRole: 'groupnorm-normalize-relu-tile',
    boundaryYield,
    encodeTile(encoder, tile, outputBuffer, computePassDescriptor) {
      return {
        buffer: dispatchGroupNormNormalizeRelu(
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
            computePassDescriptor,
          },
        ),
      };
    },
  });

  return {
    buffer: normalizeResult.buffer,
    partialBuffer,
    statsBuffer,
    partialTileCount: partialResult.tileCount,
    normalizeTileCount: normalizeResult.tileCount,
    configuredChunkItems: chunkItems,
    partialAdaptivePlanner: partialResult.adaptivePlanner,
    normalizeAdaptivePlanner: normalizeResult.adaptivePlanner,
  };
}

export async function dispatchTiledConvTranspose2d({
  device,
  inputBuf,
  weightBuf,
  biasBuf,
  params,
  chunkItems = 0,
  adaptiveDuty = null,
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
    adaptiveDuty,
    phase,
    details,
    boundaryYield,
    encodeTile(encoder, tile, outputBuffer, computePassDescriptor) {
      return dispatchConvTranspose2d(device, encoder, inputBuf, weightBuf, biasBuf, {
        ...params,
        ...(tile ? {
          outputStart: tile.outputStart,
          outputCount: tile.outputCount,
          outputBuffer,
          computePassDescriptor,
        } : {}),
      });
    },
  });
}
