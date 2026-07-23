import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const witnessSource = readFileSync(
  new URL('../tools/test_vit_four_stage_parity_browser.mjs', import.meta.url),
  'utf8',
);

assert.match(
  witnessSource,
  /sharp-vit-four-stage-parity-report\.json/,
  'the browser witness must have a deterministic durable report path',
);
assert.match(
  witnessSource,
  /requestedRoute:[\s\S]*effectiveRoute:/,
  'the report must distinguish requested and effective browser route identity',
);
assert.match(
  witnessSource,
  /status:\s*['"]failed['"][\s\S]*failurePhase[,:][\s\S]*lastTrustworthyEvidence[,:]/,
  'a failure before parity completion must still report its phase and last trustworthy evidence',
);
assert.match(
  witnessSource,
  /writeFileSync\([\s\S]*JSON\.stringify/,
  'both successful and failed witness execution must write structured evidence',
);
assert.match(
  witnessSource,
  /measurementOrder\s*=\s*\[\s*['"]dispatch-major['"]\s*,\s*['"]four-stage['"]\s*,\s*['"]four-stage['"]\s*,\s*['"]dispatch-major['"]\s*\]/,
  'timing must use paired forward and reverse mode order',
);
assert.match(
  witnessSource,
  /warmup[\s\S]*yieldMs:\s*0[\s\S]*waitForSubmittedWorkDone:\s*false/,
  'timing must follow an unmeasured production-encoder shader warmup',
);
assert.match(
  witnessSource,
  /durationSamplesMs/,
  'the report must retain the underlying per-mode timing samples',
);
assert.match(
  witnessSource,
  /projectionAuthority:\s*['"]synthetic-scheduler-overhead-projection['"]/,
  'the report must not present synthetic scheduler timing as measured full-route wall time',
);

console.log('SHARP four-stage witness contracts passed');
