// token_patch_merge.wgsl - Strip CLS and merge trimmed ViT patch tokens into CHW.

struct TokenPatchMergeParams {
  patchCount: u32,
  steps: u32,
  tokenH: u32,
  tokenW: u32,
  D: u32,
  padding: u32,
  mergedSize: u32,
  numWorkgroupsX: u32,
};

@group(0) @binding(0) var<uniform> params: TokenPatchMergeParams;
@group(0) @binding(1) var<storage, read> tokens: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 256u;

fn trim_start(step: u32) -> u32 {
  if (step > 0u) {
    return params.padding;
  }
  return 0u;
}

fn trim_end(step: u32) -> u32 {
  if (step + 1u < params.steps) {
    return params.padding;
  }
  return 0u;
}

fn merged_axis_to_patch(pos: u32) -> vec2<u32> {
  var remaining = pos;
  for (var step = 0u; step < params.steps; step++) {
    let start = trim_start(step);
    let end = trim_end(step);
    let size = params.tokenH - start - end;
    if (remaining < size) {
      return vec2<u32>(step, remaining + start);
    }
    remaining = remaining - size;
  }
  return vec2<u32>(params.steps - 1u, params.tokenH - 1u);
}

@compute @workgroup_size(WG_SIZE)
fn token_patch_merge_main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let spatialSize = params.mergedSize * params.mergedSize;
  let total = params.D * spatialSize;
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  let idx = linearWG * WG_SIZE + lid.x;

  if (idx >= total) {
    return;
  }

  let c = idx / spatialSize;
  let spatial = idx % spatialSize;
  let dstY = spatial / params.mergedSize;
  let dstX = spatial % params.mergedSize;

  let rowMap = merged_axis_to_patch(dstY);
  let colMap = merged_axis_to_patch(dstX);
  let patchRow = rowMap.x;
  let patchCol = colMap.x;
  let localY = rowMap.y;
  let localX = colMap.y;
  let patchIndex = patchRow * params.steps + patchCol;

  if (patchIndex >= params.patchCount || localY >= params.tokenH || localX >= params.tokenW) {
    return;
  }

  let tokenCount = params.tokenH * params.tokenW + 1u;
  let patchTokenOffset = patchIndex * tokenCount * params.D;
  let tokenIndex = localY * params.tokenW + localX + 1u;
  let sourceIndex = patchTokenOffset + tokenIndex * params.D + c;

  output[idx] = tokens[sourceIndex];
}
