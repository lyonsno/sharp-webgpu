// conv1x1.wgsl — Pointwise (1x1) convolution compute shader
//
// Optimized for the extremely common 1x1 conv case in ConvStack:
//   - input_blocks: project feature dims
//   - output_blocks: project to output dims
//   - DINOv2 output projections
//
// This is effectively a batched matrix multiply over spatial positions.
// Each thread computes one output element (outCh, y, x).
//
// Memory layout (CHW, row-major):
//   input:   [C_in, H, W]     — f32
//   weight:  [C_out, C_in]    — f32 (kH=kW=1, no spatial dims)
//   bias:    [C_out]          — f32
//   output:  [C_out, H, W]   — f32

struct Conv1x1Params {
  inC: u32,
  outC: u32,
  H: u32,
  W: u32,
  hasBias: u32,
  numWorkgroupsX: u32,
  outputStart: u32,
  outputCount: u32,
};

@group(0) @binding(0) var<uniform> params: Conv1x1Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 256;

fn conv1x1_value(idx: u32) -> f32 {
  let spatialSize = params.H * params.W;
  let oc = idx / spatialSize;
  let sp = idx % spatialSize;

  var sum: f32 = 0.0;
  for (var ic: u32 = 0; ic < params.inC; ic++) {
    let inputVal = input[ic * spatialSize + sp];
    let weightVal = weight[oc * params.inC + ic];
    sum += inputVal * weightVal;
  }

  if (params.hasBias != 0) {
    sum += bias[oc];
  }
  return sum;
}

@compute @workgroup_size(WG_SIZE)
fn conv1x1_main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  // 2D dispatch: linearize from workgroup_id.x + workgroup_id.y * numWorkgroupsX
  let spatialSize = params.H * params.W;
  let totalWork = params.outC * spatialSize;
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  let idx = linearWG * WG_SIZE + lid.x;

  if (idx >= totalWork) {
    return;
  }

  output[idx] = conv1x1_value(idx);
}

@compute @workgroup_size(WG_SIZE)
fn conv1x1_tiled_main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  let localIdx = linearWG * WG_SIZE + lid.x;
  if (localIdx >= params.outputCount) {
    return;
  }
  let idx = params.outputStart + localIdx;
  output[idx] = conv1x1_value(idx);
}
