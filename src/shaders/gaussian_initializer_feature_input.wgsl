// gaussian_initializer_feature_input.wgsl — Build SHARP Gaussian initializer feature_input on GPU.

struct FeatureInputParams {
  H: u32,
  W: u32,
  numWorkgroupsX: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<uniform> featureParams: FeatureInputParams;
@group(0) @binding(1) var<storage, read> imageInput: array<f32>;
@group(0) @binding(2) var<storage, read> disparityInput: array<f32>;
@group(0) @binding(3) var<storage, read> depthMinInput: array<f32>;
@group(0) @binding(4) var<storage, read_write> featureOutput: array<f32>;

const WG_SIZE: u32 = 256;

fn linear_workgroup_id(wgid: vec3<u32>, numWorkgroupsX: u32) -> u32 {
  return wgid.x + wgid.y * numWorkgroupsX;
}

fn disparity_to_depth(disparity: f32) -> f32 {
  let clamped = clamp(disparity, 1e-4, 1e4);
  return 1.0 / clamped;
}

@compute @workgroup_size(WG_SIZE)
fn feature_input_main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let hw = featureParams.H * featureParams.W;
  let total = 5u * hw;
  let linearWG = linear_workgroup_id(wgid, featureParams.numWorkgroupsX);
  let idx = linearWG * WG_SIZE + lid.x;

  if (idx >= total) {
    return;
  }

  let channel = idx / hw;
  let spatial = idx % hw;

  if (channel < 3u) {
    featureOutput[idx] = imageInput[idx];
    return;
  }

  let disparityChannel = channel - 3u;
  let depth = disparity_to_depth(disparityInput[disparityChannel * hw + spatial]);
  let depthFactor = 1.0 / (depthMinInput[0] + 1e-6);
  let rescaledDepth = min(depth * depthFactor, 100.0);
  let normalizedDisparity = 1.0 / rescaledDepth;
  featureOutput[idx] = 2.0 * normalizedDisparity - 1.0;
}
