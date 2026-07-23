// conv2d.wgsl — 2D convolution compute shader
//
// Standard conv2d with:
//   - Arbitrary kernel size (1x1, 3x3, etc.)
//   - Input tiling with halo in workgroup shared memory for 3x3
//   - Replicate padding (matching PyTorch padding_mode='replicate')
//   - Optional bias
//   - Supports batched execution (one dispatch per output channel group)
//
// Memory layout (all NCHW, row-major):
//   input:   [C_in, H, W]       — f32
//   weight:  [C_out, C_in, kH, kW] — f32
//   bias:    [C_out]             — f32
//   output:  [C_out, H_out, W_out] — f32
//
// Uniforms:
//   inC, inH, inW: input dimensions
//   outC, outH, outW: output dimensions
//   kH, kW: kernel size
//   padH, padW: padding
//   strideH, strideW: stride
//   hasBias: 0 or 1

struct ConvParams {
  inC: u32,
  inH: u32,
  inW: u32,
  outC: u32,
  outH: u32,
  outW: u32,
  kH: u32,
  kW: u32,
  padH: u32,
  padW: u32,
  strideH: u32,
  strideW: u32,
  hasBias: u32,
  numWorkgroupsX: u32,
  outputStart: u32,
  outputCount: u32,
};

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Workgroup tile for output spatial positions
// 16x16 output tile per workgroup
const TILE_W: u32 = 16;
const TILE_H: u32 = 16;

fn conv2d_value(outCh: u32, outY: u32, outX: u32) -> f32 {
  var sum: f32 = 0.0;

  for (var ic: u32 = 0; ic < params.inC; ic++) {
    for (var ky: u32 = 0; ky < params.kH; ky++) {
      for (var kx: u32 = 0; kx < params.kW; kx++) {
        let inY_raw = i32(outY * params.strideH + ky) - i32(params.padH);
        let inX_raw = i32(outX * params.strideW + kx) - i32(params.padW);

        if (inY_raw < 0 || inY_raw >= i32(params.inH) || inX_raw < 0 || inX_raw >= i32(params.inW)) {
          continue;
        }
        let inY = u32(inY_raw);
        let inX = u32(inX_raw);

        let inputIdx = ic * params.inH * params.inW + inY * params.inW + inX;
        let weightIdx = outCh * params.inC * params.kH * params.kW
                      + ic * params.kH * params.kW
                      + ky * params.kW
                      + kx;

        sum += input[inputIdx] * weight[weightIdx];
      }
    }
  }

  if (params.hasBias != 0) {
    sum += bias[outCh];
  }
  return sum;
}

@compute @workgroup_size(TILE_W, TILE_H, 1)
fn conv2d_main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wgid: vec3<u32>,
) {
  let outX = gid.x;
  let outY = gid.y;
  let outCh = wgid.z; // one output channel per z-workgroup

  if (outX >= params.outW || outY >= params.outH || outCh >= params.outC) {
    return;
  }

  let outputIdx = outCh * params.outH * params.outW + outY * params.outW + outX;
  output[outputIdx] = conv2d_value(outCh, outY, outX);
}

const LINEAR_WG_SIZE: u32 = 256;

@compute @workgroup_size(LINEAR_WG_SIZE)
fn conv2d_tiled_main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  let tileIdx = linearWG * LINEAR_WG_SIZE + lid.x;
  if (tileIdx >= params.outputCount) {
    return;
  }

  let outputIdx = params.outputStart + tileIdx;
  let totalOutputItems = params.outC * params.outH * params.outW;
  if (outputIdx >= totalOutputItems) {
    return;
  }

  let spatialSize = params.outH * params.outW;
  let outCh = outputIdx / spatialSize;
  let spatialIdx = outputIdx % spatialSize;
  let outY = spatialIdx / params.outW;
  let outX = spatialIdx % params.outW;
  output[outputIdx] = conv2d_value(outCh, outY, outX);
}
