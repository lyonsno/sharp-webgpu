export const SHARP_CONTENTION_WITNESS_SCHEMA = 'sharp.webgpu-contention-witness.v0';
export const SHARP_ROUTE_ID = 'sharp.image-to-splat.webgpu-local.v0';

const VALID_MODES = new Set(['baseline', 'contention', 'cooperative']);

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function isFiniteNonNegative(value) {
  return Number.isFinite(value) && value >= 0;
}

function requireString(errors, value, path) {
  if (!isNonEmptyString(value)) errors.push(`${path} must be a non-empty string`);
}

function requireFinitePositive(errors, value, path) {
  if (!Number.isFinite(value) || value <= 0) errors.push(`${path} must be a positive finite number`);
}

function validateRoute(errors, route) {
  if (!isObject(route)) {
    errors.push('route must be an object');
    return;
  }
  if (route.requestedRouteId !== SHARP_ROUTE_ID) {
    errors.push(`route.requestedRouteId must be ${SHARP_ROUTE_ID}`);
  }
  if (route.effectiveRouteId !== SHARP_ROUTE_ID) {
    errors.push(`route.effectiveRouteId must be ${SHARP_ROUTE_ID}`);
  }
}

function validateInference(errors, inference) {
  if (!isObject(inference)) {
    errors.push('inference must be an object');
    return;
  }
  if (inference.ok !== true) errors.push('inference.ok must be true');
  if (inference.valid !== 'OK') errors.push('inference.valid must be OK');
  requireFinitePositive(errors, inference.timeMs, 'inference.timeMs');
  requireString(errors, inference.model, 'inference.model');
  requireString(errors, inference.weights, 'inference.weights');
  if (!isObject(inference.outputs)) {
    errors.push('inference.outputs must be an object');
    return;
  }
  requireFinitePositive(errors, inference.outputs.numGaussians, 'inference.outputs.numGaussians');
  if (inference.outputs.plyAvailable !== true) errors.push('inference.outputs.plyAvailable must be true');
}

function validateResponsiveness(errors, warnings, responsiveness) {
  if (!isObject(responsiveness)) {
    errors.push('responsiveness must be an object');
    return;
  }
  for (const field of ['rafFrames', 'maxFrameGapMs', 'p95FrameGapMs', 'longFrameCount']) {
    if (!isFiniteNonNegative(responsiveness[field])) {
      errors.push(`responsiveness.${field} must be a finite non-negative number`);
    }
  }
  if (responsiveness.rafFrames === 0) {
    warnings.push('responsiveness.rafFrames is zero; main-thread breathing was not observed');
  }
}

function validateContender(errors, report) {
  const { contender, mode } = report;
  if (!isObject(contender)) {
    errors.push('contender must be an object');
    return;
  }
  if (typeof contender.enabled !== 'boolean') errors.push('contender.enabled must be boolean');
  for (const field of ['submitted', 'completed']) {
    if (!isFiniteNonNegative(contender[field])) {
      errors.push(`contender.${field} must be a finite non-negative number`);
    }
  }
  if (typeof contender.progressDuringInference !== 'boolean') {
    errors.push('contender.progressDuringInference must be boolean');
  }
  if (!Array.isArray(contender.errors)) errors.push('contender.errors must be an array');
  if (mode !== 'baseline') {
    if (contender.enabled !== true) errors.push('contender.enabled must be true for contended/cooperative runs');
    if (contender.submitted <= 0) errors.push('contender.submitted must be positive for contended/cooperative runs');
    if (contender.completed <= 0 || contender.progressDuringInference !== true) {
      errors.push('contender progress must be observed during contended/cooperative inference');
    }
  }
}

export function validateSharpContentionWitnessReport(report) {
  const errors = [];
  const warnings = [];

  if (!isObject(report)) return { ok: false, errors: ['report must be an object'], warnings };
  if (report.schema !== SHARP_CONTENTION_WITNESS_SCHEMA) {
    errors.push(`schema must be ${SHARP_CONTENTION_WITNESS_SCHEMA}`);
  }
  requireString(errors, report.runId, 'runId');
  requireString(errors, report.createdAt, 'createdAt');
  if (!VALID_MODES.has(report.mode)) errors.push(`mode must be one of ${[...VALID_MODES].join(', ')}`);
  validateRoute(errors, report.route);
  if (!isObject(report.input)) {
    errors.push('input must be an object');
  } else {
    requireString(errors, report.input.source, 'input.source');
    requireString(errors, report.input.artifactId, 'input.artifactId');
  }
  validateInference(errors, report.inference);
  validateResponsiveness(errors, warnings, report.responsiveness);
  validateContender(errors, report);
  if (!isObject(report.scheduler)) {
    errors.push('scheduler must be an object');
  } else {
    requireString(errors, report.scheduler.mode, 'scheduler.mode');
    requireString(errors, report.scheduler.verificationState, 'scheduler.verificationState');
  }

  return { ok: errors.length === 0, errors, warnings };
}
