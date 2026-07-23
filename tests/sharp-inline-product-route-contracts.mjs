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
  /const runMode = resolveSharpRunMode\(options\)/,
  'the callable product route must resolve its mode from caller options rather than standalone checkbox state',
);
assert.match(
  main,
  /runSharpStandaloneBlob[\s\S]{0,500}mode: sharpElement\('use-spn'\)\?\.checked === false \? 'backbone' : 'spn'/,
  'the standalone UI must preserve checkbox mode through an explicit product call option',
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
  /progressEvents:\s*\[\][\s\S]{0,15000}runDebug\.progressEvents\.push\(event\)/,
  'successful SHARP reports must preserve the uncapped progress events shown to the product UI',
);
assert.match(
  main,
  /function finishRouteRun\([\s\S]{0,500}run\.outputs\s*=\s*\{\s*\.\.\.run\.outputs,\s*\.\.\.outputs\s*\}/,
  'route completion must merge final artifact fields without erasing capability and plausibility evidence',
);
assert.match(
  main,
  /readBuffer\(gpu\.device, gaussianPipeline\._texDeltasBuf, texBytes, \{[\s\S]{0,1000}onChunk:[\s\S]{0,1000}schedulerYield\([\s\S]{0,1000}step: 'texture-delta-readback-copy'/,
  'the 52 MiB texture readback must copy cooperatively through foreground-aware route-tail boundaries',
);
assert.match(
  main,
  /weightsLoadStartMs = performance\.now\(\)[\s\S]{0,1200}weights = await loadWeights[\s\S]{0,7000}recordSchedulerEvent\(currentSchedulerTelemetry, 'weights-load',[\s\S]{0,700}step: 'fetch-decode-upload'/,
  'first-load weight fetch, decode, and upload must remain a named host duty interval',
);
assert.match(
  main,
  /onPhase\(event\)[\s\S]{0,3500}recordSchedulerEvent\(currentSchedulerTelemetry, `weights-load-\$\{phase\}`,[\s\S]{0,700}step: phase/,
  'weight loading must expose fetch, consolidation, parse, and materialization as separate duty intervals',
);
assert.match(
  main,
  /completedBlocksOverall: completedBlocks[\s\S]{0,200}totalBlocksOverall: totalBlocks/,
  'weight materialization progress must pair every partial count with an overall total',
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
  /_dispatchChunkedConvTranspose2d\([\s\S]{0,5000}fusionDispatchStartMs = performance\.now\(\)[\s\S]{0,2200}submitPreparedSchedulerDuty\([\s\S]{0,900}recordSchedulerEvent\(telemetry, 'spn-fusion-host-dispatch',[\s\S]{0,800}step: 'layer-dispatch-preparation'/,
  'each tiled SPN fusion submission must expose host dispatch preparation with exact range identity',
);
assert.match(
  spn,
  /if \(i === 0\) \{[\s\S]{0,600}fusionDispatchStartMs = performance\.now\(\)[\s\S]{0,1200}device\.queue\.submit[\s\S]{0,900}recordSchedulerEvent\(telemetry, 'spn-fusion-host-dispatch',[\s\S]{0,800}step: 'layer-dispatch-preparation'/,
  'the initial SPN fusion projection must retain its named host dispatch interval',
);
assert.match(
  main,
  /return attachSchedulerTelemetryArchive\(\{[\s\S]{0,500}ok: true,[\s\S]{0,500}plyBlob: composed\.plyBlob,[\s\S]{0,500}runDebug[\s\S]{0,200}runDebug\.schedulerTelemetry\)/,
  'the callable route must return the real PLY blob, canonical debug envelope, and explicit telemetry archive to Kaminos',
);
assert.match(
  main,
  /Object\.defineProperty\(result, 'schedulerTelemetryArchive', \{[\s\S]{0,200}enumerable: false/,
  'the uncapped telemetry archive must remain opt-in instead of multiplying routine result JSON',
);
assert.match(
  main,
  /globalThis\.__kaminosRunSharpImageToSplat = runSharpImageToSplat/,
  'the stable library build must expose one explicit callable browser socket',
);
assert.match(
  main,
  /export function resolveSharpElement\(root, prefix, id\)[\s\S]{0,600}root\.getElementById[\s\S]{0,600}root\.querySelector/,
  'the embedded route resolver must honor both Document/ShadowRoot and caller-owned host roots',
);
assert.match(
  main,
  /const sharpElementRoot = globalThis\.__kaminosSharpElementRoot[\s\S]{0,200}typeof document !== 'undefined' \? document : null/,
  'the embedded route must accept a caller-owned root without making document an import prerequisite',
);
assert.match(
  main,
  /sharpElementRoot\.querySelectorAll\('\.sample-thumb'\)/,
  'sample bindings must stay inside the caller-owned SHARP host when the route is embedded',
);
assert.match(
  main,
  /if \(dropZone && fileInput\)[\s\S]{0,1800}dropZone\.addEventListener[\s\S]{0,1800}fileInput\.addEventListener/,
  'importing the callable library must not require standalone drop-zone elements',
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
