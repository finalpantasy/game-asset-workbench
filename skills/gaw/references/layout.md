# Layout (pattern, relayout) and paint bake

Commands: `pattern`, `relayout`, `bake`. Flags: `gaw help`. Output folders must be new or empty.

## Why re-lay-out

Stock game UVs fold and stack: mirrored limbs share texels, a waist overlaps itself, a texture is 512 px. Paint on a shared texel shows on every surface that uses it. A unique layout gives each surface its own texels at a higher size, with the old paint copied across, so a person or an AI can paint one piece without side effects.

## pattern (built in, no Blender, seconds)

`gaw pattern in.glb --out-dir NEW --size 4096`. It cuts pieces by body part (bone weights), flattens each piece from the 3D surface, turns pieces upright, puts left/right pairs side by side, numbers the seams, packs every sheet at one texel density, and copies the paint across. It writes the GLB, new sheets, `*_pattern.png` (pieces, names, numbered seams, dotted material borders), part/position/normal maps, `legend.json`, `checks.json`, `report.json`.

Choose flags by goal (exact flags in `gaw help`):

- Default is fine for a first pass.
- `--v8`: bone-oriented pieces, exact mirror pairs, metrics, ID maps (for painting tools and AI). `--v9` adds low-stretch relaxing; use it when `checks.json` shows high anisotropy.
- `--keep-stock` / `--keep-blink`: pin boxes the game reads at fixed texels (FF9 blink). See `references/ff9.md`.
- `--guide-size N` changes only the guide and ID maps, not the layout.

Checks to read in `checks.json`: `passed`, same skeleton, clips and triangles, `unique` (no shared texels except planned kept boxes), `looks_same` (render difference under 6 of 255 mean and at most 1% of pixels over 24). The Vivi and Zidane runs passed at 4096 with render difference around 2 of 255.

Gotchas:

- Closed rings (hats, sleeves, coat tails, boots) cannot flatten in one piece. The tool cuts them until nothing folds. If a result shows a gray patch, a piece collapsed to a point: it was not edge-connected to its neighbors. Report it as a tool bug.
- Back faces share a front face's texels only when they read the same texels. A back face with its own paint (dark inside of a hair card) gets its own place.
- One density for every sheet means a sheet with few pieces stays partly empty. That is expected.
- The head and hands get a density boost (default 1.5 and 1.75), so body density drops a few percent.
- Overlay meshes (battle-only weapons) must be in the input GLB, or the importer adds them back with stock UVs and they sample the wrong texels. Export with overlays first (`references/ff9.md`).
- `relayout STOCK --check CAND` compares to stock; a candidate that differs in clips or meshes (overlays, extra `stand` clip) fails "clips" or "triangles" for that reason alone. Check against the stock that has the same meshes.

## relayout (Blender, minutes)

Dense unwrap through Blender in the background. Use only when `pattern` cannot cut a model. It needs Blender (`--blender PATH`); it runs at below-normal priority and stops after ten minutes. `--check done.glb` runs only the checks.

## bake (paint projection)

Steps, in order, each into its own new folder: `guide` (render views with ID and depth maps), then paint those pictures (outside gaw, by hand or AI), `register` (align the painted picture to the guide; it refuses a bad scale), `bake` (project the picture onto the atlas under a mask), `demo` (self-test).

- Use at least four views; two leave side and armpit texels unpainted.
- Always pass a mask (`--mask-part`, `--mask-mesh`, `--mask-png`). A bake with no mask hits millions of texels outside the area you meant; with a mask the changed texels outside it are 0.
- "torso" includes the belt and waist. Seams are only matched inside one mesh. Registration is one global similarity transform, so a local AI warp is not corrected.
- `--leak-check` and `FINAL-VERIFICATION.json` report what changed outside the mask. Read them.
- `bake --step roundtrip` on stock is the quality floor: it bakes stock views onto a blanked atlas and measures the error. Eyes and brows stay soft.
- The atlas must be blanked before a round-trip test, or stock colors leak into the "baked" result.
- Brightness: FF9's field shader multiplies texture by vertex color times 2. Output no `COLOR_0`; if one is needed, 0.5 grey with alpha 1.0. See `references/ff9.md`.
