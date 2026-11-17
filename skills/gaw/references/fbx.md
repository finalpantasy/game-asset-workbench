# fbx: GLB to the FBX Memoria loads

`gaw fbx FILE.glb --out DIR` writes a binary FBX 7.4 plus the texture PNGs it names. Memoria (the FF9 mod loader) reads `<model path>.fbx` from a mod folder before the stock prefab (`ModelFactory.cs`), so an edited mesh loads as a custom model with no other change. Flags: `gaw help`.

## Run it

1. `gaw fbx cand.glb --out NEWDIR --readback`. The summary lists bones, meshes, vertex and triangle counts and the two self-check errors.
2. Read `NEWDIR/readback/*-diff.png` next to `-glb.png` and `-fbx.png`, and `readback.json`. The FBX is read back with Memoria's own rules (int32 ids, connections, Model order), rebuilt into a model and drawn from the same cameras as the GLB (front, side, back, and the walk clip at 35 percent). The command fails if more than 0.1 percent of pixels differ.
3. For a mod folder use `--mod-layout`: `DIR/Models/<type>/<id>/<id>.fbx` and the PNGs beside it. `<type>` comes from the GLB's `ff9_geo` name (1 acc, 2 main, 3 mon, 4 npc, 5 sub) and `<id>` from `ff9_geo_id` or `--geo-id`. Zidane is `GEO_MAIN_F0_ZDN`, id 98, so `Models/2/98/98.fbx`, textures `98_0.png` (head, hair) and `98_1.png` (body).
4. Test only in a private game copy, never the installed game. Copy the folder in by hand or with the test harness; `gaw` never writes to a game folder.

## What it writes (so you can tell a wrong file from a right one)

- The data is Unity-space, because Memoria's FBX importer applies no axis, winding, UV or scale conversion. From the GLB (ff9mapkit convention, inverted): Y mirrored, positions divided by `ff9_scale` (0.01), winding reversed, V flipped, rotation quaternion `(-x, y, -z, w)`.
- Bones are `Root` (bone000) and `LimbNode`s named `bone000`..`boneNNN` in the original FF9 order and hierarchy, under an identity `Armature` Null (the importer reads `bone.Parent` of the root). Bone numbers must be contiguous from 0. The rest pose is exact: Memoria rebuilds bind poses from the bones' local translation, rotation (Euler XYZ degrees) and scale, so the writer checks the Euler round trip (`rotation round trip error`, about 1e-16) and that each joint's rest world times its inverse bind is identity in the GLB (`bind pose error`, about 1e-7; above 1e-3 it refuses unless `--force`).
- One Geometry, Model, Skin, Material (Phong, which Memoria maps to shader `Default`) and Texture per GLB mesh; one Cluster per mesh and bone with up to 4 weights per vertex, renormalized. No vertex colors (the decided policy), normals and UVs per vertex.
- `battle_model*` overlay meshes (battle-only daggers) are skipped; `--keep-overlays` keeps them. A field Zidane has none.
- Mesh order is fixed: `battle_model*`, then `field_model*`, then the rest in GLB order. Memoria wires the blink to a material by its index in `GetComponentsInChildren<SkinnedMeshRenderer>()` order, never by name, and stock Zidane's order is `battle_model0, battle_model1, mesh0, mesh1` (the blink is index 3, `mesh1`, the head). An FBX with only `mesh0` and `mesh1` puts index 3 out of range, and the game logs `[GEOTEXHEADER] Invalid mapping for texture animation of Models/GeoTexAnim/GEO_MAIN_F0_ZDN.tab` and never blinks. So for a model that must blink, export with `--keep-overlays` (the GLB must carry the overlay meshes, for example `exports/Zidane-idle-overlays.glb` or a GLB built with `plugins/HDCharacters/tools/import_keep_overlays.py`). The HD Characters plugin hides the daggers in the field.
- The texture is the material's `RelativeFilename`, resolved against the FBX folder. The PNG bytes are the GLB's own, unchanged.

## Checks that exist

- `cargo test -p wb-core --test fbx_export`: builds tiny skinned GLBs in the test (one bone, two bones with blended weights, a bad skeleton that must be refused, the writer against the parser) and, when the Modding files exist, exports stock Zidane, the belt edit and the stock-mesh hair. Each export is read back by the workbench's own Memoria-rule reader and by `ufbx` (a separate reader, dev-dependency only): vertex counts, positions, UVs, winding, weight sums, bone names and order, bind poses.
- `gaw fbx ... --readback`: the picture comparison above.

## Never

- Never claim a custom model loaded because the file exists. Prove the game took it (the harness `state` lists each renderer's vertex and triangle counts and uvHash; they must match the FBX, not stock) and look at a walk clip for stretched triangles.
- Never put the FBX in a game folder from `gaw`. Output goes to a new folder.
- Do not link AGPL FBX code into the workbench. `ufbx` is MIT or Unlicense and is only used by tests.

## In-game proof (private copy only)

Checked 2026-11-16 with the belt edit (`experiments/zidane-belt-strap-v3/boundary/combined-2-3-4/Zidane-belt-strap-v3-v13-boundary-leftline.glb`): `Models/2/98/98.fbx` plus its 8K PNG in the private runtime's mod folder. The game loaded the FBX: field Zidane's renderers reported 3664 and 1598 vertices (stock is 2293 and 1537), and the same on a second `GEO_MAIN_F0_ZDN` built through `ModelFactory.CreateModel`. The `ANH_MAIN_F0_ZDN_WALK` clip, frozen at five points, showed normal proportions, alternating legs and the new belt, no stretching. Pictures and `report.json`: `experiments/fbx-export-claude-v1/review/belt/` (`walk-close.png`, `walk-sheet.png`). Runner: `experiments/fbx-export-claude-v1/run_ingame.py` (it adds an `anim` command to a copy of the truth harness and puts the original DLL back).

Notes from that run:
- Memoria gives FBX textures an empty name. The blink does not need the name; it needs the mesh index above.
- `gaw render`/`views` index textures by image, and `compare`-style tools that use `model_textures` per mesh only look right when image and mesh order agree. The `--readback` check uses `image_textures` for that reason.

## Blink and battle proof (2026-11-16, private copy, hidden, muted, HD Characters plugin on)

`gaw fbx exports/Zidane-idle-overlays.glb --out DIR --mod-layout --keep-overlays --readback` (stock layout plus daggers; 0 differing pixels in 5 views), closed-eye box of `98_0.png` painted pure blue (that run predates the open-eye probe; see below), one launch (`experiments/fbx-export-claude-v1/run_blink_battle.py overlays-probe`):
- Field: no `Invalid mapping`, texanim invalid flags all false, render texture on renderer index 3. Blue showed in 5 of 150 closeups (3 blinks: frames 25, 105-106, 141-142). Daggers hidden by the plugin.
- Battle (first listed battle): `Battle Central Root/98` loaded the FBX (mesh1 1537, mesh0 2293, battle_model1 28, battle_model0 75 vertices). `battle-solo` pictures show correct skin, hair, clothes and both daggers, no smeared texels.
- Pictures and `report.json`: `experiments/fbx-export-claude-v1/review/overlays-probe-blink-battle/`.

Limits:
- The older belt GLB (one relaid 8K sheet, no overlay meshes) cannot blink in game: it has two meshes, so the blink index is out of range, and its sheet has no closed-eye art at the blink boxes (source 224,216-272,248 to target 32,116-80,148 at 512 px). Use the stock-layout rebuild below instead.
- The `98_0`/`98_1` sheet split is not an exporter option: the writer uses the GLB's own images, and the stock-layout GLB already has that split.
- Not checked: the hair FBX in game, vertex colors (none written).

## Belt on the stock layout (2026-11-16): ships with blink

The relaid 8K sheet cannot blink (no eye art at the blink boxes), so the belt edit was rebuilt on the stock `98_0`/`98_1` split. Scripts: `experiments/fbx-export-claude-v1/belt-stock/` (`build.py OUT_DIR [N=4096]`, helpers `transfer.py`, `glbio.py`). Inputs: `exports/Zidane-idle-overlays.glb` (stock layout, overlays, blink art) and the belt GLB `experiments/zidane-belt-strap-v3/boundary/combined-2-3-4/Zidane-belt-strap-v3-v13-boundary-leftline.glb`.

1. Triangle match: stock `mesh0` and the belt GLB share the body surface with byte-identical vertex positions; triangles match by sorted rounded positions. The stock card (22 triangles) is what the strap replaced.
2. Belt triangles: those whose own texels differ from the upscaled stock by more than 5, inside the belt region (y 1.5 to 2.6, |x| < 0.45), plus the card. That is 75 triangles. Sleeve cuffs and hands also differ (resampling noise, not paint) and stay stock.
3. Paint: each belt triangle is rasterized into the stock `98_1` at 4096 and takes the 8K color at the same barycentric point (bilinear). Everything else stays the Lanczos-upscaled stock `98_1`. The paint grows 10 texels into unused gutter.
4. Geometry: stock `mesh0` minus the 22 card triangles, plus the belt GLB's strap triangles and vertices (position, normal, joints, weights). Strap UVs come from locating the belt-sheet UV inside a card triangle and applying the same barycentric to the stock card triangle (4 vertices lie just outside the card).
5. `gaw fbx belt-stock/out/Zidane-belt-stock-layout.glb --out DIR --mod-layout --keep-overlays --readback`.

Texel need: none missing. The 75 belt triangles share 0 texels with other triangles, 0 paint conflicts, and every belt triangle has at least 4.1 stock texels (4096) per 8K texel (median 9.0), so the stock layout holds the paint at its source detail. Front and back renders match the 8K belt (strap with buckle, knot, white tail).

In game (one hidden launch, private copy, HD Characters plugin, blink box of `98_0.png` painted blue; runner `experiments/fbx-export-claude-v1/run_ship_check.py`, pictures in `review/belt-stock-probe-ship/`):
- Field: valid mapping (texanim invalid flags false, render texture on renderer 3), mesh0 3605 vertices (the belt, vs stock 2293), blue on the eyes in 7 of 150 closeups (4 blinks).
- Walk: a spawned `GEO_MAIN_F0_ZDN` frozen at 5 points of `ANH_MAIN_F0_ZDN_WALK`; no stretching, belt strap and buckle visible.
- Battle: `Battle Central Root/98` loads the FBX (mesh0 3605, mesh1 1537, daggers 75/28 vertices); the solo pictures show the new belt and the daggers.

Probe colors: the head has three texture animations (closed-eye box 224,216-272,248, open-eye box 200,184-248,216, target 32,116-80,148 at 512 px), so `probe_paint.py` paints the closed-eye box blue and the open-eye box red. Blue on the eyes only at the blinks and red the rest of the time shows both boxes are in use; anything else on open eyes means the open-eye box does not hold open-eye art. `run_ship_check.py` and `run_blink_battle.py` count both colors.

Checked 2026-11-17 (belt on the stock layout, two-color probe, one hidden launch, `review/belt-stock-probe2-ship/`): red on the eyes in 144 of 150 closeups, blue in 6 (3 blinks), none with neither color.
