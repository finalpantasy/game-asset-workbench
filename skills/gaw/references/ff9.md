# Final Fantasy IX (Memoria)

Everything here is specific to FF9 character exports. The rest of the skill is game-agnostic.

## Files and names

- Stock Zidane export: `Zidane-idle.glb`. Overlay export (battle daggers as `battle_model0` and `battle_model1`): `Zidane-idle-overlays.glb`. Vivi: `Vivi-idle.glb`. Textures are named by model id: Zidane `98_0` (head and hair) and `98_1` (body), Vivi `8_0` and `8_1`.
- The UV kit (`parts` folder) holds the part-id images for `98_0` and `98_1`. Sheets without a map borrow the parent's map; a sheet with a new layout reads "outside any part", so give judge `--parts DIR` or let it fall back to bone groups.
- Stock textures are 512 px. New layouts are 4096.
- The game folder is not a workspace. Never write to it. Test in a private game copy (`local/notes.md` names it on this machine).

## Blink boxes (do not move)

The game blinks by copying one texture box over another at fixed texels. A layout that moves those boxes breaks blinking.

- Zidane: `--keep-blink` (on `pattern` and `hair`) pins three boxes on `98_0` at 512 px: closed-eye source 224,216-272,248, open-eye source 200,184-248,216 and eye target 32,116-80,148 (`wb_core::blink`). The head has three texture animations: anim 0 copies the closed box onto the target, anim 1 copies the open box, anim 2 alternates them (the blink). Paint in the open-eye box shows over the eyes whenever they are open, so it must stay open-eye art; a layout that put hair there showed hair over open eyes in game. Both eyes read the same stock texels, so the pinned pieces share texels on purpose. `checks.json` counts them apart (`kept_stock_shared_texels`) and does not fail on them. Every other shared texel still fails.
- Vivi: the blink copies across sheets. `--keep-stock 8_0:64,104,176,128 --keep-stock 8_1:456,180,512,204`.
- Any other model: find the boxes in the game's texture-animation data (FF9: `GeoTexAnim` `.tab` plus `GeoTexAnimIndex` csv; `targetuv` is pixels from the top left, `rectuvs` are UVs) and pass them with `--keep-stock SHEET:x0,y0,x1,y1`. Pixels are stock-sheet pixels.
- Pick stock by the candidate's meshes: a candidate with `battle_model0` and `battle_model1` is compared with `Zidane-idle-overlays.glb`, otherwise `Zidane-idle.glb`.
- To check a candidate, read the `checks.json` that `pattern` wrote in its build folder: `passed: true`, `unique: true`, and `kept_stock_shared_texels` greater than 0 on the blink sheet's mesh (14,742 for Zidane's `98_0`). `relayout STOCK.glb --check CAND.glb` does not know the kept boxes: on a good `--keep-blink` layout it reports `unique: false` and `passed: false` with the same 14,742 texels as unplanned, and `--keep-blink` has no effect there. Without the build folder, rebuild with the same flags or test in the game. For a real check, count the blink in the game (a test harness paints the closed-eye box blue and the open-eye box red and counts closeup frames showing each: blue only at the blinks, red on the eyes the rest of the time; the layout passes when the counts and the screen spot match stock, and any other color on open eyes means the open box was not kept). A pass in `checks.json` proves the boxes were kept, not that the game blinks.
- Shipping: the blink sheet's render texture is rebuilt by the HD plugin. See `references/ship.md`.

## Overlay (battle-only) parts

Zidane's daggers exist only as `battle_model0` and `battle_model1`. A plain field export has none, and the importer adds them back with stock UVs, which on a new atlas samples wrong texels (orange smears on the blade). Export with overlays (`plugins/HDCharacters/tools/export_with_overlays.py`), run `pattern` on that GLB, import with `import_keep_overlays.py`. A plain `ff9mapkit` import drops the blink and the overlays.

## Brightness and vertex color

FF9's field shader draws `texture * vertexColor * 2` (0.5 equals 1.0, vertex alpha multiplies cutout alpha). The battle shader ignores vertex color. Stock Zidane and Vivi have no color channel, so write no `COLOR_0`. If one is needed, use 0.5 grey with alpha 1.0: white doubles brightness and alpha 0.5 kills the mesh. AI-generated GLBs often carry `COLOR_0`; strip it. `pattern`, `bake` and `judge REF TEST` print `SHADING WARNING:` lines for this.

## Battle size

The field view is 237 px tall in a 896 px frame (measured). The battle view of 160 px is an estimate. Small defects that vanish at 237 px may not matter; judge at game size before polishing.
