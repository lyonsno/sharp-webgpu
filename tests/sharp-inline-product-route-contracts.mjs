import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const main = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
const spn = readFileSync(new URL('../src/lib/spn.js', import.meta.url), 'utf8');
const vite = readFileSync(new URL('../vite.config.js', import.meta.url), 'utf8');
const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));

assert.match(
  main,
  /export async function runSharpImageToSplat\(blob, options = \{\}\)/,
  'SHARP must expose its existing full route as a callable product-realm function',
);
assert.match(
  main,
  /options\.gpuContext[\s\S]{0,500}injected SHARP GPU queue does not belong to the injected device/,
  'the inline route must reject a mismatched caller-owned queue before inference',
);
assert.match(
  main,
  /gpu = options\.gpuContext[\s\S]{0,250}\|\| globalThis\.__kaminosSharpInjectedGpu[\s\S]{0,250}\|\| await initGPU\(\)/,
  'the product-owned GPU must take precedence while standalone SHARP retains its existing fallback',
);
assert.match(
  main,
  /options\.weightsUrl[\s\S]{0,250}__kaminosSharpWeightsUrl[\s\S]{0,250}'\/weights\.bin'/,
  'the embedded route must use an explicit product-served weights URL without changing standalone defaults',
);
assert.match(
  main,
  /weightsLoadStartMs = performance\.now\(\)[\s\S]{0,1200}weights = await loadWeights[\s\S]{0,1200}recordSchedulerEvent\(currentSchedulerTelemetry, 'weights-load',[\s\S]{0,700}step: 'fetch-decode-upload'/,
  'first-load weight fetch, decode, and upload must remain a named host duty interval',
);
assert.match(
  main,
  /sourcePreprocessStartMs = performance\.now\(\)[\s\S]{0,2200}source-image-resize-normalize[\s\S]{0,1200}recordSchedulerEvent\(currentSchedulerTelemetry, 'source-preprocess',[\s\S]{0,700}step: 'resize-normalize'/,
  'source resize and normalization must remain a named host duty interval',
);
assert.match(
  main,
  /await options\.beforeInference\?\.\(\{[\s\S]{0,500}runId: currentSchedulerTelemetry\.runId[\s\S]{0,500}mode: runMode[\s\S]{0,500}\}\);[\s\S]{0,1500}markInferenceStart/,
  'the callable route must expose an awaited boundary after setup and immediately before real inference',
);
assert.match(
  main,
  /foregroundHandoffStartMs = performance\.now\(\)[\s\S]{0,500}await options\.beforeInference[\s\S]{0,500}recordSchedulerEvent\(currentSchedulerTelemetry, 'foreground-handoff',[\s\S]{0,700}kind: 'duty-interval'[\s\S]{0,500}step: 'lease-activation'/,
  'the awaited product handoff must remain a named scheduler duty interval',
);
assert.match(
  spn,
  /pyramidStartMs = performance\.now\(\)[\s\S]{0,1000}bilinearDownsample[\s\S]{0,700}recordSchedulerEvent\(telemetry, 'spn-host-setup',[\s\S]{0,600}step: 'pyramid-build'/,
  'SPN pyramid construction must expose its synchronous host duty interval',
);
assert.match(
  spn,
  /patchExtractionStartMs = performance\.now\(\)[\s\S]{0,1000}extractPatches[\s\S]{0,700}recordSchedulerEvent\(telemetry, 'spn-host-setup',[\s\S]{0,600}step: 'patch-extraction'/,
  'SPN patch extraction must expose its synchronous host duty interval',
);
assert.match(
  spn,
  /fusionDispatchStartMs = performance\.now\(\)[\s\S]{0,1800}device\.queue\.submit[\s\S]{0,700}recordSchedulerEvent\(telemetry, 'spn-fusion-host-dispatch',[\s\S]{0,800}step: 'layer-dispatch-preparation'/,
  'each SPN fusion layer must expose synchronous dispatch preparation before its queue wait',
);
assert.match(
  main,
  /return \{[\s\S]{0,500}ok: true,[\s\S]{0,500}plyBlob: composed\.plyBlob,[\s\S]{0,500}runDebug/,
  'the callable route must return the real PLY blob and canonical debug envelope to Kaminos',
);
assert.match(
  main,
  /globalThis\.__kaminosRunSharpImageToSplat = runSharpImageToSplat/,
  'the stable library build must expose one explicit callable browser socket',
);
assert.match(
  main,
  /const sharpElementPrefix = globalThis\.__kaminosSharpElementPrefix \|\| ''[\s\S]{0,300}document\.getElementById\(`\$\{sharpElementPrefix\}\$\{id\}`\)/,
  'the embedded route must resolve every UI element through a caller-owned prefix without changing standalone IDs',
);
assert.equal(
  main.match(/document\.getElementById\(/g)?.length,
  1,
  'only the prefix-aware resolver may call document.getElementById',
);
assert.match(
  main,
  /const sharpElementRoot = globalThis\.__kaminosSharpElementRoot \|\| document/,
  'the embedded route must accept a caller-owned SHARP element root',
);
assert.match(
  main,
  /sharpElementRoot\.querySelectorAll\('\.sample-thumb'\)/,
  'sample bindings must stay inside the caller-owned SHARP host when the route is embedded',
);
assert.match(
  vite,
  /SHARP_INLINE_BUILD[\s\S]{0,1200}entry:\s*resolve\(process\.cwd\(\), 'src\/main\.js'\)[\s\S]{0,500}fileName:\s*'sharp-inline'/,
  'the inline build must emit a stable ESM entry from the reviewed route implementation',
);
assert.match(
  vite,
  /publicDir:\s*inlineBuild\s*\?\s*false/,
  'the inline library build must not copy the 1.25 GB weights file into a generated dist tree',
);
assert.equal(pkg.scripts['build:inline'], 'SHARP_INLINE_BUILD=1 vite build', 'SHARP must expose the deterministic inline library build command');

console.log('SHARP inline product route contracts passed');
