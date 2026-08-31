# Vendored Smoke Dependency

`kaminos-webgpu-inference-kit-0.1.38.tgz` is the verified
registry artifact for
[`@kaminos/webgpu-inference-kit@0.1.38`](https://www.npmjs.com/package/@kaminos/webgpu-inference-kit/v/0.1.38),
built from Kaminos commit `5878abf050405bab8218c193f51867f1568b6127`.

SHA-256:
`396b3ea0718117a6cc440ce95853a8fa8851290ca383ab529e358cc4953bdd33`.

The vendored copy keeps SHARP route smokes reproducible without relying on
registry availability. Its bytes are identical to the published package; the
Kaminos main commit and npm registry identity remain publication authority.

## Chrome 151 GPU Timestamp Assay

`kaminos-webgpu-inference-kit-0.1.45-sharp-gpu-timestamp-assay.0.tgz` is the
provisional, unpublished package built from reviewed Kaminos commit
`fd5daefd485287c9d1836891ba1facfb51a9e354` on branch
`cc/wake-kit-gpu-timestamp-assay-0831`.

SHA-256:
`2fb03c6a260e7af7eb5513b6398aa141ecfdf58943dccaab799f43287b88abf9`.

It admits positive in-command-buffer `gpu-timestamp-query` duration as a
distinct adaptive-planner authority while preserving the existing
`queue-work-done` contract. This artifact carries experimental compatibility
authority for SHARP's Chrome 151 no-inference witness only; it is not an npm
publication or a Kaminos main release.
