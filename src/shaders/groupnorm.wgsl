// groupnorm.wgsl — Group Normalization compute shader
//
// Implements nn.GroupNorm: divide channels into groups, normalize each group
// independently over (C/num_groups, H, W), then apply learnable scale+bias.
//
// Special case: num_groups=1 → LayerNorm over spatial+channel
// Special case: num_groups=C → InstanceNorm
//
// MoGe-2 uses:
//   - GroupNorm(C//32, C) → 32 channels per group
//   - GroupNorm(1, C) → "layer norm" mode (all channels in one group)
//
// Two-pass approach:
//   Pass 1: compute mean and variance per group
//   Pass 2: normalize and apply scale+bias
//
// Memory layout (CHW, row-major):
//   input:   [C, H, W]      — f32
//   scale:   [C]             — f32 (learnable gamma)
//   bias:    [C]             — f32 (learnable beta)
//   output:  [C, H, W]      — f32

struct GroupNormParams {
  C: u32,
  H: u32,
  W: u32,
  numGroups: u32,
  eps: f32,
  numWorkgroupsX: u32,
  outputStart: u32,
  outputCount: u32,
  partialElements: u32,
  partialStart: u32,
  partialCount: u32,
  partialsPerGroup: u32,
  totalPartials: u32,
  padding0: u32,
  padding1: u32,
  padding2: u32,
};

@group(0) @binding(0) var<uniform> params: GroupNormParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> gnbias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Intermediate buffer for per-group mean and variance
// Layout: [numGroups * 2] — first numGroups entries are means, next are vars
@group(0) @binding(5) var<storage, read_write> stats: array<f32>;

const WG_SIZE: u32 = 256;
var<workgroup> reduceCount: array<u32, 256>;
var<workgroup> reduceMean: array<f32, 256>;
var<workgroup> reduceM2: array<f32, 256>;

// Pass 1: Compute mean and variance for each group
@compute @workgroup_size(WG_SIZE)
fn groupnorm_stats(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let groupIdx = gid.x;
  if (groupIdx >= params.numGroups) {
    return;
  }

  let channelsPerGroup = params.C / params.numGroups;
  let spatialSize = params.H * params.W;
  let groupSize = channelsPerGroup * spatialSize;

  let startCh = groupIdx * channelsPerGroup;

  // Compute mean
  var sum: f32 = 0.0;
  for (var c: u32 = 0; c < channelsPerGroup; c++) {
    let ch = startCh + c;
    for (var sp: u32 = 0; sp < spatialSize; sp++) {
      sum += input[ch * spatialSize + sp];
    }
  }
  let mean = sum / f32(groupSize);
  stats[groupIdx] = mean;

  // Compute variance
  var varSum: f32 = 0.0;
  for (var c: u32 = 0; c < channelsPerGroup; c++) {
    let ch = startCh + c;
    for (var sp: u32 = 0; sp < spatialSize; sp++) {
      let diff = input[ch * spatialSize + sp] - mean;
      varSum += diff * diff;
    }
  }
  stats[params.numGroups + groupIdx] = varSum / f32(groupSize);
}

// Pass 2: Normalize each element using group stats, apply scale+bias
@compute @workgroup_size(WG_SIZE)
fn groupnorm_normalize(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  let idx = linearWG * WG_SIZE + lid.x;
  let totalSize = params.C * params.H * params.W;
  if (idx >= totalSize) {
    return;
  }

  let spatialSize = params.H * params.W;
  let ch = idx / spatialSize;
  let channelsPerGroup = params.C / params.numGroups;
  let groupIdx = ch / channelsPerGroup;

  let mean = stats[groupIdx];
  let variance = stats[params.numGroups + groupIdx];
  let invStd = 1.0 / sqrt(variance + params.eps);

  let normalized = (input[idx] - mean) * invStd;
  output[idx] = normalized * scale[ch] + gnbias[ch];
}

// One workgroup computes a stable Welford state for one bounded partial.
@compute @workgroup_size(WG_SIZE)
fn groupnorm_partial_stats(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  if (linearWG >= params.partialCount) {
    return;
  }
  let partialIndex = params.partialStart + linearWG;
  let groupIndex = partialIndex / params.partialsPerGroup;
  let partialInGroup = partialIndex % params.partialsPerGroup;
  let channelsPerGroup = params.C / params.numGroups;
  let groupSize = channelsPerGroup * params.H * params.W;
  let partialOffset = partialInGroup * params.partialElements;
  let remaining = groupSize - min(groupSize, partialOffset);
  let elementCount = min(params.partialElements, remaining);
  let groupOffset = groupIndex * groupSize;

  var count: u32 = 0u;
  var mean: f32 = 0.0;
  var m2: f32 = 0.0;
  for (var index = lid.x; index < elementCount; index += WG_SIZE) {
    let value = input[groupOffset + partialOffset + index];
    count += 1u;
    let delta = value - mean;
    mean += delta / f32(count);
    let deltaAfter = value - mean;
    m2 += delta * deltaAfter;
  }
  reduceCount[lid.x] = count;
  reduceMean[lid.x] = mean;
  reduceM2[lid.x] = m2;
  workgroupBarrier();

  var stride: u32 = WG_SIZE / 2u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (lid.x < stride) {
      let rightCount = reduceCount[lid.x + stride];
      if (rightCount > 0u) {
        let leftCount = reduceCount[lid.x];
        let totalCount = leftCount + rightCount;
        let delta = reduceMean[lid.x + stride] - reduceMean[lid.x];
        reduceMean[lid.x] += delta * f32(rightCount) / f32(totalCount);
        reduceM2[lid.x] += reduceM2[lid.x + stride]
          + delta * delta * f32(leftCount) * f32(rightCount) / f32(totalCount);
        reduceCount[lid.x] = totalCount;
      }
    }
    workgroupBarrier();
    stride /= 2u;
  }

  if (lid.x == 0u) {
    stats[partialIndex] = reduceMean[0];
    stats[params.totalPartials + partialIndex] = reduceM2[0];
  }
}

// One workgroup combines all partials for one group. Tensor elements are not
// rescanned here; each thread consumes a bounded strided subset of partials.
@compute @workgroup_size(WG_SIZE)
fn groupnorm_reduce_stats(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let groupIndex = wgid.x;
  if (groupIndex >= params.numGroups) {
    return;
  }
  let partialBase = groupIndex * params.partialsPerGroup;
  let groupSize = (params.C / params.numGroups) * params.H * params.W;
  var count: u32 = 0u;
  var mean: f32 = 0.0;
  var m2: f32 = 0.0;
  for (var partial = lid.x; partial < params.partialsPerGroup; partial += WG_SIZE) {
    let partialIndex = partialBase + partial;
    let partialOffset = partial * params.partialElements;
    let partialCount = min(params.partialElements, groupSize - min(groupSize, partialOffset));
    if (partialCount > 0u) {
      let partialMean = input[partialIndex];
      let partialM2 = input[params.totalPartials + partialIndex];
      let totalCount = count + partialCount;
      let delta = partialMean - mean;
      mean += delta * f32(partialCount) / f32(totalCount);
      m2 += partialM2 + delta * delta * f32(count) * f32(partialCount) / f32(totalCount);
      count = totalCount;
    }
  }
  reduceCount[lid.x] = count;
  reduceMean[lid.x] = mean;
  reduceM2[lid.x] = m2;
  workgroupBarrier();

  var stride: u32 = WG_SIZE / 2u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (lid.x < stride) {
      let rightCount = reduceCount[lid.x + stride];
      if (rightCount > 0u) {
        let leftCount = reduceCount[lid.x];
        let totalCount = leftCount + rightCount;
        let delta = reduceMean[lid.x + stride] - reduceMean[lid.x];
        reduceMean[lid.x] += delta * f32(rightCount) / f32(totalCount);
        reduceM2[lid.x] += reduceM2[lid.x + stride]
          + delta * delta * f32(leftCount) * f32(rightCount) / f32(totalCount);
        reduceCount[lid.x] = totalCount;
      }
    }
    workgroupBarrier();
    stride /= 2u;
  }

  if (lid.x == 0u) {
    stats[groupIndex] = reduceMean[0];
    stats[params.numGroups + groupIndex] = max(reduceM2[0] / f32(reduceCount[0]), 0.0);
  }
}

@compute @workgroup_size(WG_SIZE)
fn groupnorm_normalize_tiled(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  let localIndex = linearWG * WG_SIZE + lid.x;
  if (localIndex >= params.outputCount) {
    return;
  }
  let index = params.outputStart + localIndex;
  let spatialSize = params.H * params.W;
  let channel = index / spatialSize;
  let channelsPerGroup = params.C / params.numGroups;
  let groupIndex = channel / channelsPerGroup;
  let mean = stats[groupIndex];
  let variance = stats[params.numGroups + groupIndex];
  let normalized = (input[index] - mean) * (1.0 / sqrt(variance + params.eps));
  output[index] = max(normalized * scale[channel] + gnbias[channel], 0.0);
}
