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
var<workgroup> partialMeans: array<f32, 256>;
var<workgroup> partialM2s: array<f32, 256>;
var<workgroup> partialCounts: array<u32, 256>;

// Pass 1: Compute mean and variance for each group
@compute @workgroup_size(WG_SIZE)
fn groupnorm_stats(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let groupIdx = wgid.x;
  if (groupIdx >= params.numGroups) {
    return;
  }

  let channelsPerGroup = params.C / params.numGroups;
  let spatialSize = params.H * params.W;
  let groupSize = channelsPerGroup * spatialSize;
  let groupStart = groupIdx * groupSize;
  let lane = lid.x;

  // Welford partials keep the parallel reduction stable without a second full read.
  var localMean: f32 = 0.0;
  var localM2: f32 = 0.0;
  var localCount: u32 = 0;
  for (var element = lane; element < groupSize; element += WG_SIZE) {
    let value = input[groupStart + element];
    localCount += 1u;
    let delta = value - localMean;
    localMean += delta / f32(localCount);
    let delta2 = value - localMean;
    localM2 += delta * delta2;
  }

  partialMeans[lane] = localMean;
  partialM2s[lane] = localM2;
  partialCounts[lane] = localCount;
  workgroupBarrier();

  var stride = WG_SIZE / 2u;
  while (stride > 0u) {
    if (lane < stride) {
      let rightCount = partialCounts[lane + stride];
      if (rightCount > 0u) {
        let leftCount = partialCounts[lane];
        let combinedCount = leftCount + rightCount;
        let meanDelta = partialMeans[lane + stride] - partialMeans[lane];
        partialMeans[lane] += meanDelta * f32(rightCount) / f32(combinedCount);
        partialM2s[lane] += partialM2s[lane + stride]
          + meanDelta * meanDelta * f32(leftCount) * f32(rightCount) / f32(combinedCount);
        partialCounts[lane] = combinedCount;
      }
    }
    workgroupBarrier();
    stride /= 2u;
  }

  if (lane == 0u) {
    stats[groupIdx] = partialMeans[0];
    stats[params.numGroups + groupIdx] = partialM2s[0] / f32(partialCounts[0]);
  }
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
