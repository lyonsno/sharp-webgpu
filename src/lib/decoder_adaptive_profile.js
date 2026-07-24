import { recordSchedulerEvent } from './scheduler.js';

export function createDecoderAdaptiveDuty(scheduler, telemetry, stage) {
  const effective = scheduler?.effective;
  if (!effective || effective.decoderKernelTargetDurationMs <= 0) return null;
  let plannerOrdinal = 0;
  return Object.freeze({
    unit: 'output-item',
    initialChunkItems: effective.decoderKernelChunkItems,
    targetDurationMs: effective.decoderKernelTargetDurationMs,
    adjustmentGain: effective.decoderKernelAdjustmentGain,
    bounds: Object.freeze({
      minChunkItems: effective.decoderKernelMinChunkItems,
      maxChunkItems: effective.decoderKernelMaxChunkItems,
    }),
    nextPlannerId(phase) {
      const ordinal = plannerOrdinal;
      plannerOrdinal += 1;
      return `sharp:${telemetry?.runId || 'unidentified-run'}:${stage}:${ordinal}:${phase}`;
    },
    recordObservation(phase, details) {
      recordSchedulerEvent(telemetry, `${stage}-phase`, {
        phase,
        ...details,
        kind: 'decoder-kernel-range-observed',
      });
    },
  });
}
