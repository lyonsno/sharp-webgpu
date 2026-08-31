export async function resolveWitnessKitVersion(
  loadKit = () => import('@kaminos/webgpu-inference-kit'),
) {
  const kit = await loadKit();
  const version = kit?.WEBGPU_INFERENCE_KIT_VERSION;
  if (typeof version !== 'string' || version.length === 0) {
    throw new Error('decoder witness kit did not export a version');
  }
  return version;
}

function canonicalRoute(route) {
  try {
    return new URL(route).href;
  } catch (error) {
    throw new Error(`invalid decoder witness route: ${route}`, { cause: error });
  }
}

export function validateWitnessKitIdentity({ expectedKitVersion, effectiveKitVersion }) {
  if (typeof expectedKitVersion !== 'string' || expectedKitVersion.length === 0) {
    throw new Error('decoder witness expected kit version is missing');
  }
  if (effectiveKitVersion !== expectedKitVersion) {
    throw new Error(
      `decoder witness kit version mismatch: expected ${expectedKitVersion}, effective ${effectiveKitVersion || 'missing'}`,
    );
  }
  return Object.freeze({
    status: 'admitted',
    expectedKitVersion,
    effectiveKitVersion,
  });
}

export function validateWitnessNavigation({
  requestedRoute,
  effectiveRoute,
  status,
  ok,
  expectedSourceRevision,
  effectiveSourceRevision,
  effectiveSourceState,
  expectedSourceRoot,
  effectiveSourceRoot,
  expectedEntryPoint,
  effectiveEntryPoint,
}) {
  const requested = canonicalRoute(requestedRoute);
  const effective = canonicalRoute(effectiveRoute);
  if (ok !== true || !Number.isInteger(status) || status < 200 || status >= 400) {
    throw new Error(`decoder witness route returned unsuccessful HTTP status ${status}`);
  }
  if (effective !== requested) {
    throw new Error(`decoder witness effective route mismatch: requested ${requested}, effective ${effective}`);
  }
  if (typeof expectedSourceRevision !== 'string' || expectedSourceRevision.length === 0) {
    throw new Error('decoder witness expected source revision is missing');
  }
  if (effectiveSourceRevision !== expectedSourceRevision) {
    throw new Error(`decoder witness source revision mismatch: expected ${expectedSourceRevision}, effective ${effectiveSourceRevision || 'missing'}`);
  }
  if (effectiveSourceState !== 'clean') {
    throw new Error(`decoder witness source state ${effectiveSourceState || 'missing'} is not clean`);
  }
  if (typeof expectedSourceRoot !== 'string' || expectedSourceRoot.length === 0) {
    throw new Error('decoder witness expected source root is missing');
  }
  if (effectiveSourceRoot !== expectedSourceRoot) {
    throw new Error(`decoder witness source root mismatch: expected ${expectedSourceRoot}, effective ${effectiveSourceRoot || 'missing'}`);
  }
  if (typeof expectedEntryPoint !== 'string' || expectedEntryPoint.length === 0) {
    throw new Error('decoder witness expected entry point is missing');
  }
  if (effectiveEntryPoint !== expectedEntryPoint) {
    throw new Error(`decoder witness entry point mismatch: expected ${expectedEntryPoint}, effective ${effectiveEntryPoint || 'missing'}`);
  }
  return Object.freeze({
    status: 'admitted',
    requestedRoute: requested,
    effectiveRoute: effective,
    httpStatus: status,
    sourceRevision: effectiveSourceRevision,
    sourceState: effectiveSourceState,
    sourceRoot: effectiveSourceRoot,
    entryPoint: effectiveEntryPoint,
  });
}

export function classifyWitnessSourceIdentity({
  expectedRoot,
  rootResult,
  revisionResult,
  statusResult,
}) {
  const sourceRoot = rootResult?.ok === true && rootResult.output
    ? rootResult.output
    : null;
  const sourceRevision = revisionResult?.ok === true && revisionResult.output
    ? revisionResult.output
    : null;
  let sourceState = 'unverifiable';
  if (
    typeof expectedRoot === 'string'
    && expectedRoot.length > 0
    && sourceRoot
    && sourceRevision
    && statusResult?.ok === true
  ) {
    sourceState = sourceRoot !== expectedRoot
      ? 'root-mismatch'
      : statusResult.output === '' ? 'clean' : 'dirty';
  }
  return Object.freeze({ sourceRevision, sourceRoot, sourceState });
}

export function normalizeWitnessAdapterInfo(info = {}) {
  return Object.freeze({
    vendor: typeof info.vendor === 'string' ? info.vendor : '',
    architecture: typeof info.architecture === 'string' ? info.architecture : '',
    device: typeof info.device === 'string' ? info.device : '',
    description: typeof info.description === 'string' ? info.description : '',
    isFallbackAdapter: info.isFallbackAdapter,
  });
}

export function validateNativeWitnessAdapter(info) {
  const adapterInfo = normalizeWitnessAdapterInfo(info);
  if (typeof adapterInfo.isFallbackAdapter !== 'boolean') {
    throw new Error('decoder witness adapter did not expose authoritative fallback identity');
  }
  if (adapterInfo.isFallbackAdapter) {
    throw new Error('decoder witness refuses a fallback GPU adapter');
  }
  return Object.freeze({
    status: 'non-fallback-admitted',
    authority: 'webgpu-adapter-info-isFallbackAdapter',
    adapterInfo,
  });
}

export function retainNegotiatedWitnessIdentity(lastTrustworthyEvidence, result) {
  return {
    ...lastTrustworthyEvidence,
    adapterInfo: result.adapterInfo,
    adapterAdmission: result.adapterAdmission,
    effectiveFeatures: [...result.effectiveFeatures],
    effectiveKitVersion: result.effectiveKitVersion,
  };
}

export function retainWitnessNavigationEvidence(lastTrustworthyEvidence, evidence) {
  return {
    ...lastTrustworthyEvidence,
    navigationResponse: Object.freeze({ ...evidence }),
  };
}

export async function runNegotiatedWitnessConformance({
  negotiate,
  retainNegotiated,
  conform,
}) {
  const negotiatedIdentity = await negotiate();
  retainNegotiated(negotiatedIdentity);
  return conform(negotiatedIdentity);
}

export function runWitnessAssertions({ setFailurePhase, assertConformance }) {
  setFailurePhase('conformance-assertion');
  return assertConformance();
}
