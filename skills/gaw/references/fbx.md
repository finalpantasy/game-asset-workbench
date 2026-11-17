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
- Memoria gives FBX textures an empty name, so the harness's `98_0`/`98_1` texture matching and the blink texture animation do not see them: the log has `[GEOTEXHEADER] Invalid mapping for texture animation of Models/GeoTexAnim/GEO_MAIN_F0_ZDN.tab`. Blink on a custom FBX is not checked. The belt GLB uses one 8K sheet for both meshes (`Zidane_STOCK_Organized_8K.png`), not the stock `98_0`/`98_1` pair.
- `gaw render`/`views` index textures by image, and `compare`-style tools that use `model_textures` per mesh only look right when image and mesh order agree. The `--readback` check uses `image_textures` for that reason.
- Not checked: battle (a different model), the hair FBX in game, vertex colors (none written), HD Characters plugin interaction.
