import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const spnSource = readFileSync(new URL('../src/lib/spn.js', import.meta.url), 'utf8');
const methodStart = spnSource.indexOf('async _dispatchUpsampleBlock(');
assert.notEqual(methodStart, -1, 'SPN must retain the upsample block runtime');

const bodyStart = spnSource.indexOf('{', methodStart);
let depth = 0;
let methodEnd = -1;
for (let index = bodyStart; index < spnSource.length; index += 1) {
  if (spnSource[index] === '{') depth += 1;
  if (spnSource[index] === '}') depth -= 1;
  if (depth === 0) {
    methodEnd = index + 1;
    break;
  }
}
assert.notEqual(methodEnd, -1, 'SPN upsample block must have a complete method body');

const callableSource = spnSource
  .slice(methodStart, methodEnd)
  .replace('async _dispatchUpsampleBlock(', 'async function _dispatchUpsampleBlock(');
const recordedEvents = [];
const dispatchResult = {
  buffer: { destroy() {} },
  H: 1,
  W: 1,
};
const dispatchUpsampleBlock = Function(
  'dispatchConv1x1',
  'recordSchedulerEvent',
  'schedulerYield',
  `return (${callableSource});`,
)(
  () => dispatchResult,
  (_telemetry, boundary, details) => recordedEvents.push({ boundary, ...details }),
  async () => {},
);

const device = {
  createCommandEncoder: () => ({ finish: () => ({}) }),
  queue: { submit() {} },
};
const inputBuffer = { destroy() {} };
const result = await dispatchUpsampleBlock.call(
  { device, weights: { raw: new Map([['fusion.0.weight', {}]]) } },
  inputBuffer,
  1,
  1,
  'fusion',
  [1],
  [1],
  1,
  'feature-fusion',
  null,
  [],
);

assert.equal(result.buffer, dispatchResult.buffer);
assert.equal(recordedEvents.length, 1, 'the initial fusion submission must record one completed interval');
assert.equal(recordedEvents[0].boundary, 'spn-fusion-host-dispatch');
assert.equal(recordedEvents[0].op, 'conv1x1');
assert.ok(Number.isFinite(recordedEvents[0].intervalStartMs));
assert.ok(Number.isFinite(recordedEvents[0].intervalEndMs));
assert.ok(Number.isFinite(recordedEvents[0].durationMs));
assert.ok(recordedEvents[0].durationMs >= 0);

console.log('SPN fusion runtime contracts passed');
