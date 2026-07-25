#!/usr/bin/env node
import {
  closeSync,
  createReadStream,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  readSync,
  renameSync,
  writeFileSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';
import puppeteer from 'puppeteer-core';

import {
  buildGateAFailureReport,
  EXPECTED_SOURCE_SHA256,
  EXPECTED_WEIGHTS_SHA256,
  GATE_A_CHUNK_ITEMS,
  validateGateAReport,
} from './decoder_chunk_ladder_contract.mjs';

function parseArgs(argv) {
  const parsed = new Map();
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (!key.startsWith('--')) continue;
    const next = argv[index + 1];
    const value = next && !next.startsWith('--') ? next : true;
    parsed.set(key, value);
    if (value !== true) index += 1;
  }
  return parsed;
}

function requiredPath(args, key) {
  const value = args.get(key);
  if (typeof value !== 'string' || !value) throw new Error(`${key} is required`);
  const resolved = path.resolve(value);
  if (!existsSync(resolved)) throw new Error(`${key} does not exist: ${resolved}`);
  return resolved;
}

function sha256File(filePath) {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(filePath);
    stream.on('error', reject);
    stream.on('data', chunk => hash.update(chunk));
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

function atomicWriteJson(filePath, value) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.tmp-${process.pid}`;
  writeFileSync(temporaryPath, `${JSON.stringify(value, null, 2)}\n`);
  const fileDescriptor = openSync(temporaryPath, 'r');
  fsyncSync(fileDescriptor);
  closeSync(fileDescriptor);
  renameSync(temporaryPath, filePath);
  const directoryDescriptor = openSync(path.dirname(filePath), 'r');
  fsyncSync(directoryDescriptor);
  closeSync(directoryDescriptor);
}

function readCommand(command, args = []) {
  try {
    return {
      status: 'available',
      command: [command, ...args],
      output: execFileSync(command, args, { encoding: 'utf8' }).trim(),
    };
  } catch (error) {
    return {
      status: 'unavailable',
      command: [command, ...args],
      reason: error?.message || String(error),
    };
  }
}

function parseProcessRows(text) {
  return String(text || '')
    .split('\n')
    .map(line => {
      const match = line.trim().match(/^(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(.+)$/);
      if (!match) return null;
      return {
        pid: Number(match[1]),
        ppid: Number(match[2]),
        cpuPercent: Number(match[3]),
        memoryPercent: Number(match[4]),
        residentBytes: Number(match[5]) * 1024,
        virtualBytes: Number(match[6]) * 1024,
        command: match[7],
      };
    })
    .filter(Boolean);
}

function descendantProcesses(rows, rootPid) {
  const selected = new Set([rootPid]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const row of rows) {
      if (!selected.has(row.pid) && selected.has(row.ppid)) {
        selected.add(row.pid);
        changed = true;
      }
    }
  }
  return rows.filter(row => selected.has(row.pid));
}

function parseVmStat(text) {
  const pageSizeMatch = String(text).match(/page size of (\d+) bytes/);
  const pageSize = Number(pageSizeMatch?.[1] || 0);
  const pages = {};
  for (const line of String(text).split('\n')) {
    const match = line.match(/^([^:]+):\s+(\d+)\./);
    if (match) pages[match[1].trim()] = Number(match[2]);
  }
  const bytes = label => Number.isFinite(pages[label]) && pageSize > 0
    ? pages[label] * pageSize
    : null;
  return {
    pageSize,
    freeBytes: bytes('Pages free'),
    activeBytes: bytes('Pages active'),
    inactiveBytes: bytes('Pages inactive'),
    wiredBytes: bytes('Pages wired down'),
    compressedBytes: bytes('Pages occupied by compressor'),
    swapInPages: pages['Swapins'] ?? null,
    swapOutPages: pages['Swapouts'] ?? null,
  };
}

function parseSwapUsage(text) {
  const match = String(text).match(/total = ([\d.]+)M\s+used = ([\d.]+)M\s+free = ([\d.]+)M/);
  if (!match) return { raw: String(text).trim() };
  return {
    totalBytes: Number(match[1]) * 1024 * 1024,
    usedBytes: Number(match[2]) * 1024 * 1024,
    freeBytes: Number(match[3]) * 1024 * 1024,
  };
}

function parseAgxObservation(text) {
  const fields = {};
  for (const field of [
    'Device Utilization %',
    'Renderer Utilization %',
    'Tiler Utilization %',
    'In use system memory',
    'Allocated system memory',
  ]) {
    const match = String(text).match(new RegExp(`"${field.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"=(\\d+)`));
    fields[field] = match ? Number(match[1]) : null;
  }
  return {
    status: Object.values(fields).some(Number.isFinite) ? 'available' : 'unavailable',
    authority: 'system-global-ioreg-agxaccelerator-not-process-exclusive',
    fields,
  };
}

function sampleHost(browserPid) {
  const ps = readCommand('ps', ['-axo', 'pid=,ppid=,%cpu=,%mem=,rss=,vsz=,comm=']);
  const processRows = ps.status === 'available' ? parseProcessRows(ps.output) : [];
  const browserProcesses = descendantProcesses(processRows, browserPid);
  const vmStat = readCommand('vm_stat');
  const swap = readCommand('sysctl', ['-n', 'vm.swapusage']);
  const therm = readCommand('pmset', ['-g', 'therm']);
  const battery = readCommand('pmset', ['-g', 'batt']);
  const agx = readCommand('ioreg', ['-r', '-c', 'AGXAccelerator', '-d', '1', '-l']);
  return {
    observedAt: new Date().toISOString(),
    process: {
      rootPid: browserPid,
      rows: browserProcesses,
      aggregateCpuPercent: browserProcesses.reduce((sum, row) => sum + row.cpuPercent, 0),
      aggregateResidentBytes: browserProcesses.reduce((sum, row) => sum + row.residentBytes, 0),
    },
    hostMemory: {
      vmStat: vmStat.status === 'available' ? parseVmStat(vmStat.output) : vmStat,
      swap: swap.status === 'available' ? parseSwapUsage(swap.output) : swap,
    },
    thermalPower: {
      thermal: therm,
      powerSource: battery,
    },
    gpu: agx.status === 'available' ? parseAgxObservation(agx.output) : agx,
  };
}

function float16ToFloat32(bits) {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function readFloat16Tensor(weightsPath, descriptor) {
  if (
    !descriptor
    || !Number.isSafeInteger(descriptor.offset)
    || !Number.isSafeInteger(descriptor.size)
    || descriptor.offset < 0
    || descriptor.size <= 0
    || descriptor.size % 2 !== 0
  ) {
    throw new Error('invalid fp16 tensor descriptor');
  }
  const bytes = Buffer.alloc(descriptor.size);
  const fileDescriptor = openSync(weightsPath, 'r');
  const readBytes = readSync(fileDescriptor, bytes, 0, bytes.length, descriptor.offset);
  closeSync(fileDescriptor);
  if (readBytes !== bytes.length) {
    throw new Error(`partial tensor read at ${descriptor.offset}: ${readBytes}/${bytes.length}`);
  }
  const values = new Array(bytes.length / 2);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = float16ToFloat32(bytes.readUInt16LE(index * 2));
  }
  return values;
}

const args = parseArgs(process.argv.slice(2));
const usage = 'benchmark_decoder_chunk_ladder_browser.mjs --url <vite-url> --source <17_img.png> --weights <weights.bin> --weights-manifest <weights.json> --out <report.json> [--chrome <executable>]';
if (args.has('--help')) {
  console.log(usage);
  process.exit(0);
}

const url = String(args.get('--url') || 'http://127.0.0.1:5173/');
const sourcePath = requiredPath(args, '--source');
const weightsPath = requiredPath(args, '--weights');
const weightsManifestPath = requiredPath(args, '--weights-manifest');
const outPath = path.resolve(String(args.get('--out') || '/tmp/sharp-decoder-chunk-ladder-report.json'));
const chromePath = path.resolve(String(
  args.get('--chrome') || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
));
const tensorName = 'feature_model.image_encoder.conv.weight';
const biasName = 'feature_model.image_encoder.conv.bias';
const startedAt = new Date().toISOString();
let phase = 'preflight';
let browser = null;
let primaryOutputWritten = false;
let lastTrustworthyEvidence = null;
let activeReport = null;

atomicWriteJson(outPath, {
  schema: 'sharp-webgpu.decoder-chunk-ladder-report.v0',
  status: 'starting',
  phase,
  startedAt,
  primaryOutputWritten,
  hiddenTimeoutMs: null,
  requestedChunkItems: [...GATE_A_CHUNK_ITEMS],
  requested: {
    url,
    sourcePath,
    weightsPath,
    weightsManifestPath,
    chromePath,
  },
  lastTrustworthyEvidence,
});

try {
  const [sourceSha256, weightsSha256, weightsManifestSha256] = await Promise.all([
    sha256File(sourcePath),
    sha256File(weightsPath),
    sha256File(weightsManifestPath),
  ]);
  if (sourceSha256 !== EXPECTED_SOURCE_SHA256) {
    throw new Error(`source hash mismatch: expected ${EXPECTED_SOURCE_SHA256}, observed ${sourceSha256}`);
  }
  if (weightsSha256 !== EXPECTED_WEIGHTS_SHA256) {
    throw new Error(`weights hash mismatch: expected ${EXPECTED_WEIGHTS_SHA256}, observed ${weightsSha256}`);
  }
  const weightsManifest = JSON.parse(readFileSync(weightsManifestPath, 'utf8'));
  const weightDescriptor = weightsManifest.tensors?.[tensorName];
  const biasDescriptor = weightsManifest.tensors?.[biasName];
  const weights = readFloat16Tensor(weightsPath, weightDescriptor);
  const bias = readFloat16Tensor(weightsPath, biasDescriptor);
  const sourceMime = path.extname(sourcePath).toLowerCase() === '.png' ? 'image/png' : 'application/octet-stream';
  const sourceDataUrl = `data:${sourceMime};base64,${readFileSync(sourcePath).toString('base64')}`;

  phase = 'browser-launch';
  browser = await puppeteer.launch({
    executablePath: chromePath,
    headless: false,
    args: [
      '--enable-unsafe-webgpu',
      '--disable-gpu-sandbox',
      '--no-first-run',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
      '--window-size=1280,900',
    ],
  });
  const browserPid = browser.process()?.pid;
  if (!Number.isSafeInteger(browserPid)) throw new Error('headed Chrome process id unavailable');
  const page = await browser.newPage();
  page.setDefaultTimeout(0);
  page.setDefaultNavigationTimeout(0);
  await page.goto(url, { waitUntil: 'networkidle0', timeout: 0 });
  await page.bringToFront();

  phase = 'browser-setup';
  const setup = await page.evaluate(async ({
    sourceDataUrl: encodedSource,
    sourceIdentity,
    weightValues,
    biasValues,
    requestedChunkItems,
  }) => {
    const { dispatchConv2d } = await import('/src/lib/shader_ops.js');
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('high-performance WebGPU adapter unavailable');
    const timestampQueryAvailable = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({
      requiredFeatures: timestampQueryAvailable ? ['timestamp-query'] : [],
    });
    const uncapturedErrors = [];
    device.addEventListener('uncapturederror', event => {
      uncapturedErrors.push({
        observedAt: new Date().toISOString(),
        message: event.error?.message || String(event.error || 'uncaptured WebGPU error'),
      });
    });
    const deviceLoss = {
      state: 'pending',
      reason: null,
      message: null,
      observedAt: null,
    };
    device.lost.then(info => {
      deviceLoss.state = 'resolved';
      deviceLoss.reason = info?.reason || null;
      deviceLoss.message = info?.message || null;
      deviceLoss.observedAt = new Date().toISOString();
    });

    const image = new Image();
    image.src = encodedSource;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = 1536;
    canvas.height = 1536;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const plane = canvas.width * canvas.height;
    const inputValues = new Float32Array(5 * plane);
    for (let index = 0; index < plane; index += 1) {
      const red = rgba[index * 4] / 127.5 - 1;
      const green = rgba[index * 4 + 1] / 127.5 - 1;
      const blue = rgba[index * 4 + 2] / 127.5 - 1;
      const luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
      inputValues[index] = red;
      inputValues[plane + index] = green;
      inputValues[2 * plane + index] = blue;
      inputValues[3 * plane + index] = luma;
      inputValues[4 * plane + index] = -luma;
    }

    const createUploadedBuffer = values => {
      const typed = values instanceof Float32Array ? values : new Float32Array(values);
      const buffer = device.createBuffer({
        size: typed.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer, 0, typed);
      return buffer;
    };
    const inputBuffer = createUploadedBuffer(inputValues);
    const weightBuffer = createUploadedBuffer(weightValues);
    const biasBuffer = createUploadedBuffer(biasValues);
    const fixtureParams = {
      inC: 5,
      inH: 1536,
      inW: 1536,
      outC: 128,
      kH: 2,
      kW: 2,
      padH: 0,
      padW: 0,
      strideH: 2,
      strideW: 2,
    };
    const outH = 768;
    const outW = 768;
    const totalOutputItems = fixtureParams.outC * outH * outW;
    const outputBytes = totalOutputItems * Float32Array.BYTES_PER_ELEMENT;
    if (device.limits.maxBufferSize < outputBytes) {
      throw new Error(`device maxBufferSize ${device.limits.maxBufferSize} cannot hold ${outputBytes}-byte output`);
    }
    if (device.limits.maxStorageBufferBindingSize < outputBytes) {
      throw new Error(`device maxStorageBufferBindingSize ${device.limits.maxStorageBufferBindingSize} cannot bind ${outputBytes}-byte output`);
    }

    const rafGaps = [];
    let previousRaf = null;
    let rafActive = true;
    const rafTick = timestamp => {
      if (!rafActive) return;
      if (previousRaf !== null) rafGaps.push(timestamp - previousRaf);
      previousRaf = timestamp;
      requestAnimationFrame(rafTick);
    };
    requestAnimationFrame(rafTick);

    const compareShader = `
      struct CompareParams {
        itemCount: u32,
        workgroupsX: u32,
      };
      @group(0) @binding(0) var<uniform> params: CompareParams;
      @group(0) @binding(1) var<storage, read> reference: array<u32>;
      @group(0) @binding(2) var<storage, read> candidate: array<u32>;
      @group(0) @binding(3) var<storage, read_write> mismatch: atomic<u32>;
      @compute @workgroup_size(256)
      fn main(
        @builtin(workgroup_id) workgroupId: vec3<u32>,
        @builtin(local_invocation_id) localId: vec3<u32>,
      ) {
        let workgroupIndex = workgroupId.y * params.workgroupsX + workgroupId.x;
        let index = workgroupIndex * 256u + localId.x;
        if (index < params.itemCount && reference[index] != candidate[index]) {
          atomicAdd(&mismatch, 1u);
        }
      }
    `;
    const comparePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: compareShader }),
        entryPoint: 'main',
      },
    });

    const compareOutputs = async (referenceBuffer, candidateBuffer) => {
      const mismatchBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(mismatchBuffer, 0, new Uint32Array([0]));
      const paramsBuffer = device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const totalWorkgroups = Math.ceil(totalOutputItems / 256);
      const workgroupsX = Math.min(totalWorkgroups, device.limits.maxComputeWorkgroupsPerDimension);
      const workgroupsY = Math.ceil(totalWorkgroups / workgroupsX);
      device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([totalOutputItems, workgroupsX]));
      const readBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(comparePipeline);
      pass.setBindGroup(0, device.createBindGroup({
        layout: comparePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: referenceBuffer } },
          { binding: 2, resource: { buffer: candidateBuffer } },
          { binding: 3, resource: { buffer: mismatchBuffer } },
        ],
      }));
      pass.dispatchWorkgroups(workgroupsX, workgroupsY);
      pass.end();
      encoder.copyBufferToBuffer(mismatchBuffer, 0, readBuffer, 0, 4);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      await readBuffer.mapAsync(GPUMapMode.READ);
      const bitMismatchCount = new Uint32Array(readBuffer.getMappedRange().slice(0))[0];
      readBuffer.unmap();
      readBuffer.destroy();
      paramsBuffer.destroy();
      mismatchBuffer.destroy();
      return {
        authority: 'gpu-u32-full-buffer-compare',
        comparedItemCount: totalOutputItems,
        bitMismatchCount,
      };
    };

    const yieldForegroundFrame = async () => {
      const startedAtMs = performance.now();
      await new Promise(resolve => requestAnimationFrame(resolve));
      return performance.now() - startedAtMs;
    };

    const controlStartedAtMs = performance.now();
    const controlEncodeStartedAtMs = performance.now();
    const controlEncoder = device.createCommandEncoder();
    const controlResult = dispatchConv2d(
      device,
      controlEncoder,
      inputBuffer,
      weightBuffer,
      biasBuffer,
      fixtureParams,
    );
    const controlCommands = controlEncoder.finish();
    const controlEncodeEndedAtMs = performance.now();
    const controlSubmitStartedAtMs = performance.now();
    device.queue.submit([controlCommands]);
    const controlSubmitEndedAtMs = performance.now();
    await device.queue.onSubmittedWorkDone();
    const controlQueueEndedAtMs = performance.now();
    const controlYieldMs = await yieldForegroundFrame();
    const controlEndedAtMs = performance.now();

    window.__sharpGateA = {
      device,
      dispatchConv2d,
      inputBuffer,
      weightBuffer,
      biasBuffer,
      fixtureParams,
      totalOutputItems,
      outputBytes,
      referenceOutput: controlResult.buffer,
      compareOutputs,
      yieldForegroundFrame,
      rafGaps,
      rafCursor: rafGaps.length,
      uncapturedErrors,
      deviceLoss,
      requestedChunkItems,
      activeGpuResources: {
        knownBufferCount: 4,
        knownDeclaredBytes: inputValues.byteLength
          + weightValues.length * 4
          + biasValues.length * 4
          + outputBytes,
        scope: 'benchmark-owned input, weight, bias, and reference output buffers; shader cache internals excluded',
      },
      cumulativeGpuResources: {
        createdKnownBufferCount: 4,
        declaredBytes: inputValues.byteLength
          + weightValues.length * 4
          + biasValues.length * 4
          + outputBytes,
      },
      stop() {
        rafActive = false;
        inputBuffer.destroy();
        weightBuffer.destroy();
        biasBuffer.destroy();
        controlResult.buffer.destroy();
      },
    };

    const adapterInfo = adapter.info || {};
    return {
      sourceIdentity,
      browser: {
        headed: !document.hidden,
        visibilityState: document.visibilityState,
        userAgent: navigator.userAgent,
      },
      adapter: {
        requestedPowerPreference: 'high-performance',
        effective: {
          vendor: adapterInfo.vendor || null,
          architecture: adapterInfo.architecture || null,
          device: adapterInfo.device || null,
          description: adapterInfo.description || null,
        },
      },
      device: {
        limits: {
          maxBufferSize: device.limits.maxBufferSize,
          maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
          maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
          maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
          maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
        },
        timestampQuery: {
          requested: true,
          adapterFeatureAvailable: timestampQueryAvailable,
          effective: device.features.has('timestamp-query'),
          authority: 'cpu-performance-now-wall; timestamp-query identity recorded but shader timestamps are not used',
        },
      },
      fixture: {
        authority: 'source-derived-production-shape-not-full-model-intermediate',
        operation: 'feature_model.image_encoder.conv',
        inputDerivation: 'RGB normalized from exact source; channel 3 is normalized luma and channel 4 is negative luma',
        params: fixtureParams,
        totalOutputItems,
        outputBytes,
      },
      control: {
        itemCount: totalOutputItems,
        dispatchCount: 1,
        commandAllocationAndEncodingMs: controlEncodeEndedAtMs - controlEncodeStartedAtMs,
        submitCallMs: controlSubmitEndedAtMs - controlSubmitStartedAtMs,
        submitToQueueCompletionMs: controlQueueEndedAtMs - controlSubmitEndedAtMs,
        browserYieldMs: controlYieldMs,
        wallMs: controlEndedAtMs - controlStartedAtMs,
      },
      gpuResources: {
        active: window.__sharpGateA.activeGpuResources,
        cumulative: window.__sharpGateA.cumulativeGpuResources,
      },
      deviceLoss: { ...deviceLoss },
      uncapturedErrors: [...uncapturedErrors],
    };
  }, {
    sourceDataUrl,
    sourceIdentity: {
      path: sourcePath,
      sha256: sourceSha256,
    },
    weightValues: weights,
    biasValues: bias,
    requestedChunkItems: [...GATE_A_CHUNK_ITEMS],
  });

  const browserVersion = await browser.version();
  const repoCommit = execFileSync('git', ['rev-parse', 'HEAD'], {
    cwd: path.resolve(new URL('..', import.meta.url).pathname),
    encoding: 'utf8',
  }).trim();
  const routeIdentity = {
    repo: {
      path: path.resolve(new URL('..', import.meta.url).pathname),
      commit: repoCommit,
    },
    source: {
      path: sourcePath,
      sha256: sourceSha256,
    },
    weights: {
      path: weightsPath,
      sha256: weightsSha256,
      manifestPath: weightsManifestPath,
      manifestSha256: weightsManifestSha256,
      dtype: weightsManifest.dtype || null,
      tensor: tensorName,
      tensorDescriptor: weightDescriptor,
      biasTensor: biasName,
      biasDescriptor,
    },
    browser: {
      requestedExecutable: chromePath,
      effectiveVersion: browserVersion,
      headed: setup.browser.headed,
      visibilityState: setup.browser.visibilityState,
      userAgent: setup.browser.userAgent,
      processId: browserPid,
    },
    adapter: setup.adapter,
    device: setup.device,
  };

  const report = {
    schema: 'sharp-webgpu.decoder-chunk-ladder-report.v0',
    status: 'running',
    phase: 'single-dispatch-control-complete',
    retention: 'uncapped',
    hiddenTimeoutMs: null,
    startedAt,
    primaryOutputWritten,
    routeIdentity,
    fixture: setup.fixture,
    singleDispatchControl: setup.control,
    requestedChunkItems: [...GATE_A_CHUNK_ITEMS],
    runs: [],
    deviceLoss: setup.deviceLoss,
    uncapturedErrors: setup.uncapturedErrors,
    lastTrustworthyEvidence: {
      phase: 'single-dispatch-control-complete',
      completedRungCount: 0,
      singleDispatchControl: setup.control,
    },
  };
  activeReport = report;
  lastTrustworthyEvidence = report.lastTrustworthyEvidence;
  atomicWriteJson(outPath, report);

  for (const chunkItems of GATE_A_CHUNK_ITEMS) {
    phase = `ladder-${chunkItems}`;
    const hostBefore = sampleHost(browserPid);
    const run = await page.evaluate(async requestedChunkItems => {
      const state = window.__sharpGateA;
      if (!state) throw new Error('Gate A browser state missing');
      if (!state.requestedChunkItems.includes(requestedChunkItems)) {
        throw new Error(`unrequested Gate A chunk size ${requestedChunkItems}`);
      }
      const ranges = [];
      let outputBuffer = null;
      const runRafStart = state.rafGaps.length;
      const wallStartedAtMs = performance.now();
      let itemStart = 0;
      let rangeIndex = 0;
      let commandAllocationAndEncodingMs = 0;
      let submitCallMs = 0;
      let submitToQueueCompletionMs = 0;
      let browserYieldMs = 0;

      while (itemStart < state.totalOutputItems) {
        const itemCount = Math.min(requestedChunkItems, state.totalOutputItems - itemStart);
        const encodeStartedAtMs = performance.now();
        const encoder = state.device.createCommandEncoder();
        const result = state.dispatchConv2d(
          state.device,
          encoder,
          state.inputBuffer,
          state.weightBuffer,
          state.biasBuffer,
          {
            ...state.fixtureParams,
            outputStart: itemStart,
            outputCount: itemCount,
            ...(outputBuffer ? { outputBuffer } : {}),
          },
        );
        outputBuffer = result.buffer;
        const commands = encoder.finish();
        const encodeEndedAtMs = performance.now();
        const submitStartedAtMs = performance.now();
        state.device.queue.submit([commands]);
        const submitEndedAtMs = performance.now();
        await state.device.queue.onSubmittedWorkDone();
        const queueEndedAtMs = performance.now();
        const yieldedMs = await state.yieldForegroundFrame();

        const range = {
          rangeIndex,
          itemStart,
          itemEnd: itemStart + itemCount,
          itemCount,
          plannedChunkItems: requestedChunkItems,
          commandAllocationAndEncodingMs: encodeEndedAtMs - encodeStartedAtMs,
          submitCallMs: submitEndedAtMs - submitStartedAtMs,
          submitToQueueCompletionMs: queueEndedAtMs - submitEndedAtMs,
          browserYieldMs: yieldedMs,
        };
        ranges.push(range);
        commandAllocationAndEncodingMs += range.commandAllocationAndEncodingMs;
        submitCallMs += range.submitCallMs;
        submitToQueueCompletionMs += range.submitToQueueCompletionMs;
        browserYieldMs += range.browserYieldMs;
        itemStart += itemCount;
        rangeIndex += 1;
      }

      const exactOutput = await state.compareOutputs(state.referenceOutput, outputBuffer);
      const wallEndedAtMs = performance.now();
      const rawGapsMs = state.rafGaps.slice(runRafStart);
      state.cumulativeGpuResources.createdKnownBufferCount += 1;
      state.cumulativeGpuResources.declaredBytes += state.outputBytes;
      const activeBeforeDestroy = {
        knownBufferCount: state.activeGpuResources.knownBufferCount + 1,
        knownDeclaredBytes: state.activeGpuResources.knownDeclaredBytes + state.outputBytes,
        scope: state.activeGpuResources.scope,
      };
      outputBuffer.destroy();
      return {
        chunkItems: requestedChunkItems,
        itemCount: state.totalOutputItems,
        dispatchCount: ranges.length,
        ranges,
        totals: {
          commandAllocationAndEncodingMs,
          submitCallMs,
          submitToQueueCompletionMs,
          browserYieldMs,
          wallMs: wallEndedAtMs - wallStartedAtMs,
          itemsPerSecond: state.totalOutputItems / ((wallEndedAtMs - wallStartedAtMs) / 1000),
        },
        exactOutput,
        rafRawGapsMs: rawGapsMs,
        gpuResources: {
          activeBeforeCandidateDestroy: activeBeforeDestroy,
          activeAfterCandidateDestroy: { ...state.activeGpuResources },
          cumulative: { ...state.cumulativeGpuResources },
        },
        deviceLoss: { ...state.deviceLoss },
        uncapturedErrors: [...state.uncapturedErrors],
      };
    }, chunkItems);
    const hostAfter = sampleHost(browserPid);
    const { summarizeRafGaps } = await import('./decoder_chunk_ladder_contract.mjs');
    run.raf = {
      ...summarizeRafGaps(run.rafRawGapsMs),
      rawGapsMs: run.rafRawGapsMs,
      retention: 'uncapped',
    };
    delete run.rafRawGapsMs;
    run.hostObservations = [hostBefore, hostAfter];
    report.runs.push(run);
    report.phase = `${phase}-complete`;
    report.deviceLoss = run.deviceLoss;
    report.uncapturedErrors = run.uncapturedErrors;
    report.lastTrustworthyEvidence = {
      phase: report.phase,
      completedRungCount: report.runs.length,
      lastCompletedChunkItems: chunkItems,
      lastRangeIndex: run.ranges.at(-1)?.rangeIndex ?? null,
      lastItemEnd: run.ranges.at(-1)?.itemEnd ?? null,
      exactOutput: run.exactOutput,
    };
    lastTrustworthyEvidence = report.lastTrustworthyEvidence;
    activeReport = report;
    atomicWriteJson(outPath, report);
  }

  phase = 'validation';
  report.status = 'complete';
  report.phase = 'complete';
  report.completedAt = new Date().toISOString();
  report.validationFailures = validateGateAReport(report);
  if (report.deviceLoss?.state !== 'pending') {
    report.validationFailures.push(`WebGPU device loss resolved: ${report.deviceLoss.reason || 'unknown'}`);
  }
  if (report.uncapturedErrors?.length) {
    report.validationFailures.push(`WebGPU uncaptured errors: ${report.uncapturedErrors.length}`);
  }
  if (report.validationFailures.length) {
    report.status = 'failed';
    throw new Error(`Gate A validation failed: ${report.validationFailures.join('; ')}`);
  }
  primaryOutputWritten = true;
  report.primaryOutputWritten = true;
  activeReport = report;
  await page.evaluate(() => window.__sharpGateA?.stop?.());
  atomicWriteJson(outPath, report);
  console.log(JSON.stringify({
    status: report.status,
    reportPath: outPath,
    routeIdentity: report.routeIdentity,
    rungCount: report.runs.length,
  }, null, 2));
} catch (error) {
  const failureReport = buildGateAFailureReport({
    activeReport,
    phase,
    error,
    startedAt,
    requested: {
      url,
      sourcePath,
      weightsPath,
      weightsManifestPath,
      chromePath,
    },
    lastTrustworthyEvidence,
  });
  atomicWriteJson(outPath, failureReport);
  console.error(JSON.stringify(failureReport, null, 2));
  process.exitCode = 1;
} finally {
  if (browser) await browser.close();
}
