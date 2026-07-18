import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';

const moduleUrl = pathToFileURL(new URL('../dist-inline/sharp-inline.js', import.meta.url).pathname);
moduleUrl.searchParams.set('contract', `${Date.now()}`);
const sharp = await import(moduleUrl.href);

assert.equal(typeof sharp.runSharpImageToSplat, 'function', 'inline build must import without a standalone DOM');
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
