// concat_channels.wgsl — Concatenate two CHW tensors along channel dimension.

struct ConcatChannelsParams {
  aC: u32,
  bC: u32,
  H: u32,
  W: u32,
  numWorkgroupsX: u32,
};

@group(0) @binding(0) var<uniform> params: ConcatChannelsParams;
@group(0) @binding(1) var<storage, read> inputA: array<f32>;
@group(0) @binding(2) var<storage, read> inputB: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 256;

@compute @workgroup_size(WG_SIZE)
fn concat_channels_main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let spatialSize = params.H * params.W;
  let outC = params.aC + params.bC;
  let total = outC * spatialSize;
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  let idx = linearWG * WG_SIZE + lid.x;

  if (idx >= total) {
    return;
  }

  let channel = idx / spatialSize;
  let spatial = idx % spatialSize;

  if (channel < params.aC) {
    output[idx] = inputA[channel * spatialSize + spatial];
  } else {
    let bChannel = channel - params.aC;
    output[idx] = inputB[bChannel * spatialSize + spatial];
  }
}
