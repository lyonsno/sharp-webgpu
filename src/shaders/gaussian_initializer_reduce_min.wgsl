// gaussian_initializer_reduce_min.wgsl — Reduce Gaussian initializer disparity depths to min depth.

struct ReduceParams {
  count: u32,
  numWorkgroupsX: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> reduceParams: ReduceParams;
@group(0) @binding(1) var<storage, read> reduceInput: array<f32>;
@group(0) @binding(2) var<storage, read_write> reduceOutput: array<f32>;

const WG_SIZE: u32 = 256;
const INF_F32: f32 = 3.402823e38;

var<workgroup> partialMins: array<f32, 256>;

fn linear_workgroup_id(wgid: vec3<u32>, numWorkgroupsX: u32) -> u32 {
  return wgid.x + wgid.y * numWorkgroupsX;
}

fn disparity_to_depth(disparity: f32) -> f32 {
  let clamped = clamp(disparity, 1e-4, 1e4);
  return 1.0 / clamped;
}

fn reduce_workgroup_min(linearWG: u32, lid: vec3<u32>, value: f32) {
  partialMins[lid.x] = value;
  workgroupBarrier();

  var stride = WG_SIZE / 2u;
  loop {
    if (lid.x < stride) {
      partialMins[lid.x] = min(partialMins[lid.x], partialMins[lid.x + stride]);
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }

  if (lid.x == 0u) {
    reduceOutput[linearWG] = partialMins[0];
  }
}

@compute @workgroup_size(WG_SIZE)
fn depth_min_from_disparity_main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = linear_workgroup_id(wgid, reduceParams.numWorkgroupsX);
  let idx = linearWG * WG_SIZE + lid.x;
  var value = INF_F32;
  if (idx < reduceParams.count) {
    value = disparity_to_depth(reduceInput[idx]);
  }
  reduce_workgroup_min(linearWG, lid, value);
}

@compute @workgroup_size(WG_SIZE)
fn reduce_min_main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = linear_workgroup_id(wgid, reduceParams.numWorkgroupsX);
  let idx = linearWG * WG_SIZE + lid.x;
  var value = INF_F32;
  if (idx < reduceParams.count) {
    value = reduceInput[idx];
  }
  reduce_workgroup_min(linearWG, lid, value);
}
