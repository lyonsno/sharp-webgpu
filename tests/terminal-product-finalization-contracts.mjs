import assert from 'node:assert/strict';
import fs from 'node:fs';

import { writePLYAsync } from '../src/lib/compose.js';

const mainSource = fs.readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
const composeSource = fs.readFileSync(new URL('../src/lib/compose.js', import.meta.url), 'utf8');
const workerSource = fs.readFileSync(new URL('../src/workers/ply_writer.js', import.meta.url), 'utf8');

assert.match(
  mainSource,
  /terminalFinalizationTimings:\s*\[\]/,
  'the route report must retain post-seal terminal attribution outside the sealed scheduler corpus',
);
for (const step of [
  'scheduler-snapshot',
  'runtime-report-finalization',
  'route-receipt',
  'result-presentation',
]) {
  assert.match(
    mainSource,
    new RegExp(`step:\\s*['"]${step}['"]`),
    `terminal attribution must name ${step}`,
  );
}

assert.match(
  workerSource,
  /crypto\.subtle\.digest\(\s*['"]SHA-256['"]/,
  'worker PLY assembly must compute artifact identity away from the main thread',
);
assert.match(
  workerSource,
  /sha256:\s*plySha256/,
  'the worker result must carry its PLY digest with the assembled artifact',
);
assert.match(
  composeSource,
  /plySha256 = identity\.sha256/,
  'worker digest authority must survive composition into the route result',
);
assert.match(
  mainSource,
  /splatHash = composed\.plySha256/,
  'the product receipt must consume the worker-owned digest without rematerializing the PLY Blob',
);
assert.doesNotMatch(
  mainSource,
  /const splatHash = await sha256Hex\(composed\.plyBlob\)/,
  'the product route must not copy and hash the complete PLY again on the main thread',
);

function createDigestWorker(sha256) {
  return {
    onmessage: null,
    onerror: null,
    onmessageerror: null,
    postMessage(message) {
      const plyBlob = new Blob(['ply']);
      queueMicrotask(() => this.onmessage?.({
        data: {
          type: 'ply-assembled',
          requestId: message.requestId,
          plyBlob,
          bytes: plyBlob.size,
          sha256,
        },
      }));
    },
    terminate() {},
  };
}

await assert.rejects(
  writePLYAsync(new Float32Array(14), 1, 1, 1, 1, {
    mode: 'worker',
    requireSha256: true,
    workerFactory: () => createDigestWorker(null),
  }),
  /missing or invalid PLY SHA-256/,
  'a missing worker digest must fail before route-receipt authority can be claimed',
);

let propagatedIdentity = null;
await writePLYAsync(new Float32Array(14), 1, 1, 1, 1, {
  mode: 'worker',
  requireSha256: true,
  workerFactory: () => createDigestWorker('a'.repeat(64)),
  onArtifactIdentity: identity => {
    propagatedIdentity = identity;
  },
});
assert.deepEqual(
  propagatedIdentity,
  {
    algorithm: 'sha256',
    authority: 'worker-assembled-artifact',
    sha256: 'a'.repeat(64),
    bytes: 3,
  },
  'worker digest identity must retain algorithm, authority, digest, and exact bytes',
);

console.log('terminal product finalization contracts passed');
