function canonicalRoute(route) {
  try {
    return new URL(route).href;
  } catch (error) {
    throw new Error(`invalid decoder witness route: ${route}`, { cause: error });
  }
}

export function validateWitnessNavigation({ requestedRoute, effectiveRoute, status, ok }) {
  const requested = canonicalRoute(requestedRoute);
  const effective = canonicalRoute(effectiveRoute);
  if (ok !== true || !Number.isInteger(status) || status < 200 || status >= 400) {
    throw new Error(`decoder witness route returned unsuccessful HTTP status ${status}`);
  }
  if (effective !== requested) {
    throw new Error(`decoder witness effective route mismatch: requested ${requested}, effective ${effective}`);
  }
  return Object.freeze({
    status: 'admitted',
    requestedRoute: requested,
    effectiveRoute: effective,
    httpStatus: status,
  });
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
  };
}
