import assert from 'node:assert/strict';

const gpuModule = await import('../src/lib/gpu.js');

assert.equal(
  typeof gpuModule.retireGpuBuffers,
  'function',
  'GPU lifecycle helpers must expose explicit post-inference buffer retirement',
);

if (typeof gpuModule.retireGpuBuffers === 'function') {
  function fakeBuffer(label, { throwOnDestroy = false } = {}) {
    return {
      label,
      destroyCalls: 0,
      destroy() {
        this.destroyCalls += 1;
        if (throwOnDestroy) throw new Error(`destroy ${label} failure`);
      },
    };
  }

  const disabledBuffer = fakeBuffer('disabled');
  const disabled = gpuModule.retireGpuBuffers([
    { label: 'disabled', buffer: disabledBuffer, bytes: 64 },
  ], { enabled: false });
  assert.equal(disabled.status, 'disabled');
  assert.equal(disabled.requested, false);
  assert.equal(disabled.effective, false);
  assert.equal(disabled.destroyedCount, 0);
  assert.equal(disabled.knownAllocationBytes, 64);
  assert.equal(disabled.observedMemoryReleaseBytes, null);
  assert.equal(disabledBuffer.destroyCalls, 0, 'disabled retirement must not touch GPU buffers');

  const first = fakeBuffer('first');
  const second = fakeBuffer('second');
  const completed = gpuModule.retireGpuBuffers([
    { label: 'spn-0', buffer: first, bytes: 100 },
    { label: 'spn-0-alias', buffer: first, bytes: 100 },
    { label: 'disparity', buffer: second, bytes: 300 },
  ], { enabled: true, requested: 'true' });
  assert.equal(completed.status, 'completed');
  assert.equal(completed.requested, 'true', 'retirement reports must preserve caller request identity separately from application');
  assert.equal(completed.effective, true);
  assert.equal(completed.targetCount, 3);
  assert.equal(completed.uniqueTargetCount, 2);
  assert.equal(completed.aliasCount, 1);
  assert.equal(completed.destroyedCount, 2);
  assert.equal(completed.alreadyRetiredCount, 0);
  assert.equal(completed.knownAllocationBytes, 400, 'known bytes must deduplicate aliases');
  assert.equal(completed.destroyedKnownAllocationBytes, 400);
  assert.equal(completed.observedMemoryReleaseBytes, null, 'destroy calls must not impersonate observed browser memory release');
  assert.equal(first.destroyCalls, 1);
  assert.equal(second.destroyCalls, 1);

  const repeated = gpuModule.retireGpuBuffers([
    { label: 'spn-0', buffer: first, bytes: 100 },
    { label: 'disparity', buffer: second, bytes: 300 },
  ], { enabled: true });
  assert.equal(repeated.status, 'completed');
  assert.equal(repeated.destroyedCount, 0);
  assert.equal(repeated.alreadyRetiredCount, 2, 'repeat retirement must remain visible without double destruction');
  assert.equal(repeated.destroyedKnownAllocationBytes, 0);
  assert.equal(first.destroyCalls, 1);
  assert.equal(second.destroyCalls, 1);

  const failing = fakeBuffer('failing', { throwOnDestroy: true });
  const survivor = fakeBuffer('survivor');
  assert.throws(
    () => gpuModule.retireGpuBuffers([
      { label: 'failing', buffer: failing, bytes: 500 },
      { label: 'survivor', buffer: survivor, bytes: 700 },
    ], { enabled: true }),
    error => {
      assert.equal(error.name, 'GpuBufferRetirementError');
      assert.equal(error.retirementReport.status, 'failed');
      assert.equal(error.retirementReport.effective, false);
      assert.equal(error.retirementReport.destroyedCount, 1, 'retirement must continue after one target fails');
      assert.equal(error.retirementReport.failedCount, 1);
      assert.equal(error.retirementReport.knownAllocationBytes, 1200);
      assert.equal(error.retirementReport.destroyedKnownAllocationBytes, 700);
      assert.deepEqual(error.retirementReport.failures, [{
        label: 'failing',
        name: 'Error',
        message: 'destroy failing failure',
      }]);
      return true;
    },
    'requested retirement failure must fail loud with partial-work custody',
  );
  assert.equal(failing.destroyCalls, 1);
  assert.equal(survivor.destroyCalls, 1);

  const witnessedBytes = [
    256 * 768 * 768 * 4,
    256 * 384 * 384 * 4,
    512 * 192 * 192 * 4,
    1024 * 96 * 96 * 4,
    1024 * 48 * 48 * 4,
    2 * 1536 * 1536 * 4,
    6 * 768 * 768 * 4,
    22 * 768 * 768 * 4,
  ];
  const witnessedTargets = witnessedBytes.map((bytes, index) => ({
    label: `witnessed-${index}`,
    buffer: fakeBuffer(`witnessed-${index}`),
    bytes,
  }));
  const witnessed = gpuModule.retireGpuBuffers(witnessedTargets, { enabled: true });
  assert.equal(witnessed.targetCount, 8);
  assert.equal(witnessed.destroyedCount, 8);
  assert.equal(witnessed.knownAllocationBytes, 962592768);
  assert.equal(witnessed.knownAllocationBytes / (1024 * 1024), 918);
  assert.equal(witnessed.destroyedKnownAllocationBytes, 962592768);
  assert.equal(witnessed.observedMemoryReleaseBytes, null);
}

console.log('GPU buffer retirement contracts passed');
