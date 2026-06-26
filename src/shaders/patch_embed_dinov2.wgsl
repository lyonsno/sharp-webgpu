// DINOv2 patch embedding compute shader.
// Takes an image [3, H, W] and produces (N+1, D) token embeddings:
//   N patches (tokenH x tokenW grid of patchSize x patchSize patches) + 1 CLS token.
//   Each patch is flattened (patchSize*patchSize*3) then linearly projected to D.
//   Position embeddings are added directly (no interpolation when token count matches pretrained).
//
// Parameterized for any patch size via params.patchSize.
// SHARP uses patchSize=16 (dinov2l16_384): 384/16 = 24x24 = 576 patches + 1 CLS = 577 tokens.
// MoGe uses patchSize=14 (dinov2_vitl14): variable grid.

struct Params {
  imgH: u32,      // image height (tokenH * patchSize)
  imgW: u32,      // image width (tokenW * patchSize)
  patchSize: u32,  // 16 for SHARP, 14 for MoGe
  tokenH: u32,
  tokenW: u32,
  channels: u32,   // 3
  D: u32,          // model dim (1024)
  numTokens: u32,  // tokenH * tokenW + 1 (including CLS)
  numWorkgroupsX: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> image: array<f32>;       // [3, imgH, imgW] CHW
@group(0) @binding(2) var<storage, read> projWeight: array<f32>;  // [D, C, patchSize, patchSize]
@group(0) @binding(3) var<storage, read> projBias: array<f32>;    // [D]
@group(0) @binding(4) var<storage, read> clsToken: array<f32>;    // [1, 1, D]
@group(0) @binding(5) var<storage, read> posEmbed: array<f32>;    // [1, 1+numPatchesPretrained, D]
@group(0) @binding(6) var<storage, read_write> output: array<f32>; // [numTokens, D]

const WG_SIZE: u32 = 256;

@compute @workgroup_size(WG_SIZE)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  let idx = linearWG * WG_SIZE + lid.x;
  let totalElements = params.numTokens * params.D;

  if (idx >= totalElements) { return; }

  let token = idx / params.D;
  let d = idx % params.D;

  var val = 0.0;

  if (token == 0u) {
    // CLS token
    val = clsToken[d];
  } else {
    // Patch embedding
    let patchIdx = token - 1u;
    let patchRow = patchIdx / params.tokenW;
    let patchCol = patchIdx % params.tokenW;
    let startY = patchRow * params.patchSize;
    let startX = patchCol * params.patchSize;

    // Conv2d-style patch projection: weight is [D, C, patchSize, patchSize]
    val = projBias[d];
    for (var c = 0u; c < params.channels; c++) {
      for (var py = 0u; py < params.patchSize; py++) {
        for (var px = 0u; px < params.patchSize; px++) {
          let imgY = startY + py;
          let imgX = startX + px;
          // Image is CHW
          let pixelVal = image[c * params.imgH * params.imgW + imgY * params.imgW + imgX];
          // Weight is [D, C, pH, pW] → index [d, c, py, px]
          let wIdx = d * params.channels * params.patchSize * params.patchSize
                   + c * params.patchSize * params.patchSize
                   + py * params.patchSize + px;
          val += pixelVal * projWeight[wIdx];
        }
      }
    }
  }

  // Add position embedding (CLS pos is at index 0, patch pos follow)
  // For now: use position embedding directly if token count matches,
  // otherwise skip (interpolation would need a separate pass)
  val += posEmbed[idx];

  output[idx] = val;
}
