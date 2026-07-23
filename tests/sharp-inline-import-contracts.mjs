import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const moduleUrl = pathToFileURL(new URL('../dist-inline/sharp-inline.js', import.meta.url).pathname);
const moduleSource = readFileSync(moduleUrl, 'utf8');
assert.match(
  moduleSource,
  /\/sharp-inline\/assets\/ply_writer-[A-Za-z0-9_-]+\.js/,
  'inline worker chunks must resolve through the owning Kaminos SHARP route',
);
moduleUrl.searchParams.set('contract', `${Date.now()}`);
const sharp = await import(moduleUrl.href);

assert.equal(typeof sharp.runSharpImageToSplat, 'function', 'inline build must import without a standalone DOM');
assert.equal(
  typeof sharp.sharpKitRuntimeIdentity,
  'function',
  'inline build must expose the effective kit source and foreground-pressure mode',
);
const expectedKitSourceRevision = process.env.SHARP_EXPECTED_KIT_SOURCE_REVISION || null;
if (expectedKitSourceRevision) {
  assert.equal(
    globalThis.__sharpDebug?.kitSourceRevision,
    expectedKitSourceRevision,
    'inline build must carry the exact kit source revision supplied at build time',
  );
}
assert.equal(
  sharp.sharpKitRuntimeIdentity({ foregroundOpportunityPressureSnapshot() {} }).foregroundPressureMode,
  'counter-snapshot',
);
assert.equal(
  sharp.sharpKitRuntimeIdentity({}).foregroundPressureMode,
  'full-history-snapshot-fallback',
);
assert.equal(sharp.resolveSharpRunMode({}), 'spn', 'callable image-to-splat must default to the full SPN route');
assert.equal(sharp.resolveSharpRunMode({ mode: 'backbone' }), 'backbone', 'standalone UI may explicitly request backbone smoke mode');
assert.throws(
  () => sharp.resolveSharpRunMode({ mode: 'unknown' }),
  /unsupported SHARP run mode/,
  'unknown callable route modes must fail loud',
);

const documentElement = { id: 'document-element' };
assert.equal(
  sharp.resolveSharpElement({ getElementById: id => id === 'sharp-status' ? documentElement : null }, 'sharp-', 'status'),
  documentElement,
  'Document and ShadowRoot lookup must use the supplied root',
);
const hostElement = { id: 'host-element' };
assert.equal(
  sharp.resolveSharpElement({ querySelectorAll: selector => selector === '[id]' ? [hostElement] : [] }, 'host-', 'element'),
  hostElement,
  'HTMLElement lookup must remain inside the supplied caller host',
);

console.log('SHARP inline import contracts passed');
