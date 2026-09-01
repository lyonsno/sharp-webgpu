# splatshot

Headless splat screenshot rig: point it at a `.ply` (3DGS format), give it a
JSON list of camera shots, get PNG frames back. Built for agent visual
verification — the camera is computed from the data, and the frames are meant
to be *read* (inspected) before anything ships.

## Usage

From the repo root (needs `@mkkellogg/gaussian-splats-3d` + `three` +
`puppeteer-core` installed, and Chrome at the standard macOS path):

```bash
node tools/splatshot/shoot.mjs '[
  {"params": {"ply": "assets/cake.ply", "az": 16, "el": 4, "pz": 0.62, "dist": 1.1}, "out": "shot.png"}
]'
```

- `ply` — path relative to the repo root (served over localhost)
- `az`, `el` — orbit azimuth/elevation in degrees around the pivot
- `pz` — orbit pivot depth along +z. Don't guess it: parse the PLY and use the
  opacity-weighted median z of the visible splats.
- `dist` — orbit radius multiplier relative to `pz`

`gen_ply.mjs <image> <out.ply> [port]` runs the SHARP-WebGPU pipeline headless
against a running vite dev server and saves the exported PLY.

## Method notes (the part that matters)

1. **Compute the camera from the data.** Pivot at the opacity-weighted median
   depth; frame from the scene bounds. Eyeballed pivots put the camera inside
   the scene.
2. **Look at every probe frame before committing to a sweep.** Single-image
   splat scenes have failure geographies you can only find empirically — e.g.
   SHARP puts a matte of near-field splats in front of the source viewpoint,
   so views within ~10° of straight-on are fogged; the crisp band for the cake
   scene was 12–24° off-axis.
3. **Shoot a control against ground truth.** The capture viewport must be
   square (see comment in shoot.mjs); this was caught by rendering a square
   control frame and comparing proportions against the input photo.

## Known limits / roadmap

- PLY (3DGS) only. GLB/mesh support would come via a second viewer page
  (three.js GLTFLoader) behind the same shot-list contract.
- Up axis is hardcoded `[0,-1,0]` (SHARP/OpenCV convention). Arbitrary-
  orientation inputs need an `--up` param or an auto-orient probe (PCA on
  visible splat positions is usually enough to find the ground plane).
- Chrome path is hardcoded for macOS.
