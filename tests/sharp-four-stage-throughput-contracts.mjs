import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  parseSharpSchedulerConfig,
  planNextVitBlockChunk,
  planVitBlockMicroduties,
} from '../src/lib/scheduler.js';

const friendlyScheduler = {
  mode: 'cooperative',
  spnPatchChunkSize: 1,
  yieldMs: 4,
  waitForSubmittedWorkDone: true,
  gaussianPhaseYieldMs: 4,
  vitBlockChunkSize: 1,
  vitMicroduty: true,
  vitMicrodutyMode: 'dispatch-major',
  cpuChunkItems: 16384,
  routeTailYieldMs: 3,
  spnFusionChunkItems: 524288,
  decoderKernelChunkItems: 262144,
  decoderKernelMinChunkItems: 65536,
  decoderKernelMaxChunkItems: 8388608,
  decoderKernelTargetDurationMs: 12,
  plyAssemblyMode: 'worker',
  retirePostInferenceBuffers: true,
};
const fourStageRequest = {
  ...friendlyScheduler,
  vitMicrodutyMode: 'four-stage',
};

const dispatchMajor = parseSharpSchedulerConfig({
  sharpScheduler: friendlyScheduler,
});
const fourStage = parseSharpSchedulerConfig({
  sharpScheduler: fourStageRequest,
});

assert.equal(dispatchMajor.requested.vitMicrodutyMode, 'dispatch-major');
assert.equal(dispatchMajor.effective.vitMicrodutyMode, 'dispatch-major');
assert.equal(fourStage.requested.vitMicrodutyMode, 'four-stage');
assert.equal(fourStage.effective.vitMicrodutyMode, 'four-stage');
assert.deepEqual(
  Object.fromEntries(Object.entries(fourStage.requested).filter(([key]) => key !== 'vitMicrodutyMode')),
  Object.fromEntries(Object.entries(dispatchMajor.requested).filter(([key]) => key !== 'vitMicrodutyMode')),
  'the requested four-stage candidate must change only the caller-owned ViT mode',
);
assert.deepEqual(
  Object.fromEntries(Object.entries(fourStage.effective).filter(([key]) => key !== 'vitMicrodutyMode')),
  Object.fromEntries(Object.entries(dispatchMajor.effective).filter(([key]) => key !== 'vitMicrodutyMode')),
  'the effective four-stage candidate must preserve every non-ViT scheduler control',
);

const expectedOrder = [
  'norm1-qkv',
  'attention-projection-residual',
  'norm2-fc1',
  'fc2-residual',
];
const encoderCount = 36;
const blocksPerEncoder = 24;
let dispatchMajorDutyCount = 0;
let fourStageDutyCount = 0;

for (let encoderIndex = 0; encoderIndex < encoderCount; encoderIndex += 1) {
  for (let blockIndex = 0; blockIndex < blocksPerEncoder; blockIndex += 1) {
    const range = planNextVitBlockChunk(
      blocksPerEncoder,
      blockIndex,
      fourStage.effective.vitBlockChunkSize,
      blockIndex,
    );
    const dispatchDuties = planVitBlockMicroduties(
      range,
      dispatchMajor.effective.vitMicrodutyMode,
    );
    const fourStageDuties = planVitBlockMicroduties(
      range,
      fourStage.effective.vitMicrodutyMode,
    );
    assert.deepEqual(
      fourStageDuties.map(duty => duty.microphase),
      expectedOrder,
      `encoder ${encoderIndex} block ${blockIndex} must preserve four-stage dependency order`,
    );
    assert.ok(
      fourStageDuties.every(duty => duty.blockIndex === blockIndex),
      'every four-stage duty must retain exact block identity',
    );
    dispatchMajorDutyCount += dispatchDuties.length;
    fourStageDutyCount += fourStageDuties.length;
  }
}

assert.equal(dispatchMajorDutyCount, 10_368);
assert.equal(fourStageDutyCount, 3_456);
assert.equal(dispatchMajorDutyCount - fourStageDutyCount, 6_912);
assert.equal((dispatchMajorDutyCount - fourStageDutyCount) * friendlyScheduler.yieldMs, 27_648);
const backboneSource = readFileSync(new URL('../src/lib/backbone.js', import.meta.url), 'utf8');
assert.match(
  backboneSource,
  /export\s*\{\s*VIT_CONFIG\s*,\s*ViTEncoder\s*\}/,
  'the real-WebGPU parity witness must execute the production ViT encoder rather than copied shader logic',
);

console.log('SHARP four-stage throughput contracts passed');
