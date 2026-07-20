import assert from 'node:assert/strict';

import {
  parseSharpSchedulerConfig,
  planVitBlockMicroduties,
} from '../src/lib/scheduler.js';

const requested = {
  mode: 'cooperative',
  vitBlockChunkSize: 1,
  vitMicroduty: true,
  vitMicrodutyMode: 'dispatch-major',
};
const scheduler = parseSharpSchedulerConfig({ sharpScheduler: requested });

assert.equal(scheduler.requested.vitMicrodutyMode, 'dispatch-major');
assert.equal(scheduler.effective.vitMicrodutyMode, 'dispatch-major');
assert.deepEqual(scheduler.unsupportedFields, []);

const range = { blockStart: 4, blockEnd: 5, blockCount: 1, totalBlocks: 24 };
assert.deepEqual(
  planVitBlockMicroduties(range, scheduler.effective.vitMicrodutyMode).map(duty => duty.microphase),
  [
    'norm1',
    'qkv-projection',
    'qkv-split',
    'attention-scores',
    'attention-softmax',
    'attention-apply',
    'attention-projection',
    'attention-residual',
    'norm2',
    'fc1',
    'fc2',
    'mlp-residual',
  ],
  'dispatch-major must preserve the exact reviewed duty order for each transformer block',
);
assert.deepEqual(
  planVitBlockMicroduties(range).map(duty => duty.microphase),
  ['attention-residual', 'mlp-residual'],
  'the zero-configuration path must preserve the inherited two-stage microduty contract',
);
assert.throws(
  () => parseSharpSchedulerConfig({
    sharpScheduler: { ...requested, vitMicrodutyMode: 'dispatch-mostly' },
  }),
  /ViT microduty mode must be one of/,
  'unknown microduty modes must fail before inference instead of falling back',
);

console.log('Wake dispatch-major composition contracts passed');
