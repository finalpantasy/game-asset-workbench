# Hair, straps, thin lines

Commands: `hair`, `judge --strands`, `shells`, `strap-rebuild`, `strap-check`, `strap-sweep`, `line-repaint`, `boundary-check`, `paint-fix`. Flags: `gaw help`. Outputs go to new folders or new files.

## hair

`gaw hair file.glb --out-dir NEW` retextures hair cards from parameters: a flow field over the cards, strand-aligned UVs, a strand synthesizer, checks and turntable sheets. `--rebuild-model` also rebuilds the card geometry (clean strips along the flow, fold rows, tips trimmed to the stock alpha outline).

- Pick the hair triangles with `--part NAME` (kit part map) or `--mesh NAME`; give the texture with `--sheet`.
- `--look stock|warm` is a palette choice, a taste decision for the user.
- Strand coordinates must come from the stock solve (`--model-param stock`, the default). Re-solving on a finer mesh gave 9 to 22 flow breaks. Leave flow smoothing at its default.
- Read the check lines: flow breaks across joins, Lab color across joins, fringe at every mip level, duplicate faces. Zero breaks and zero fringe is the bar.
- `--rebuild-model` writes the stock-mesh version to `OUT/stock-mesh` as a fallback. Keep it.
- `--backfaces drop` saves triangles but only if the game culls back faces. Ask before using it.
- Form comes from a frequency split. The lock layout, crevice darkness and shading come from a guide in the stock UV layout (the stock sheet by default, or `--guide PNG`, such as a painted or `bake` sheet), blurred along the surface and regraded by `--look`. The synthesizer adds only fine strands. `--guide none` brings back the synthesizer-only look, which reads flat. The `form:` line compares low-pass lightness with stock: correlation near 0.9 and a lock contrast ratio near 1 mean the lock design survived. The guide is blurred inside groups of locks joined by welded edges (`hair::guide_groups`), not inside one lock, so a lock that meets another one keeps the same light and dark across the join.
- AO (`--ao`) and a strand highlight (`--highlight`, 0 turns it off) are baked into the texture because FF9's shader is unlit. Keep `--view normal` (the default). A fixed front viewer puts a pale, strandless smear on the back of the head.
- Hairline triangles outside the part (Zidane's olive widow's peak) need `--grow-part`, or `--include` with triangle indices. The run prints the triangles it added. Grown triangles become their own lock, so they cannot bend the cards' flow. When they were merged into the front lock, the run had 6 to 23 flow breaks. For paint they still join the cards: with surface strands they take the guide of the locks they weld to. Before v5 they got none and rendered as flat orange wedges at the peak.
- Stair-stepped tips: `--tip-refine` (subdivision near the alpha cut) and `--tip-blur` (smooths the cut line) under `--rebuild-model`.
- Every layer is computed on the 3D surface and baked into each chart, and gutter texels take their color from the triangle across the seam. So the two sides of a seam match by construction. The `seam:` line measures Lab at the seam itself; `--seam-tol` only nudges gutter texels where the sides still differ (a safety net).
- The alpha channel is a signed distance to the cut line (`--sdf-spread`, in texels), so the game's 0.5 cutout stays smooth when magnified and at mips. The `alpha coverage` line should stay nearly flat from mip 0 down.
- One flow field covers all the hair: nearby layered cards of other locks pull toward each other (`--couple`; 0 = per lock), and the `flow over the whole hair` line gives their angle before and after. It also names locks whose flow runs away from their faded tips (`--tip-flip` turns them).
- Combing: `--comb strokes.json` takes guide strokes that a person drew on the surface. The flow follows them where they lie (`--comb-radius`). The `hair` run also writes `OUT/flow.json`, the solved flow, for overlays.
- To find leftover fringe texels, set `GAW_FRINGE_AT=1` before the run. Each one is printed with its level, position and color.

### Close look

Zero breaks and zero fringe do not mean a person will accept the hair. Close up, one screen pixel covers about 7 texels of a 4096 sheet (mip 2 to 3), so detail finer than about 15 texels averages into blur. The `close look` line renders 8 close-up views of ours and stock, compares them, and fails the run (`checks.json` key `close_look`) on any of these:

- **Specks and tiny bits**: small pieces of one lock floating next to the hair (`--look-small`, px). More than stock fails. A stipple of mismatched paint is two locks fighting for depth. Our rebuild caused it by snapping gap vertices onto another lock's card. Gap closing now stops short, tip outline vertices stay put, and `--layer-sep` keeps stacked cards apart.
- **Holes**: see-through spots inside the hair. More than stock fails.
- **Sharpness across strands**: lightness change one pixel across the strand direction. Below `--sharp-target` times stock fails. The upsampled stock layer alone scores about half of stock. The fiber layer (`--strand-gain`, `--strand-width`, `--strand-fine-width`, `--strand-fine`) adds strands 20 to 30 texels wide with crisp edges so they survive the close-up mip.
  `--fiber-space surface` (the default) traces the strands on the 3D surface, so one strand runs on across chart seams and card edges. `--fiber-space chart` draws them per chart, as before v4. With surface strands the defaults are `--strand-gain 6` and `--strand-fine 0.15`: stock's broad lock light and dark (the guide) stays at full strength and the fibers sit quietly on top. Gain 14 and fine 0.5 read as wood grain (fiber share 45.8% of tiles against stock's 10.6%). The quiet fibers fail this sharpness bar and the flat bar below (v5: 1.40 and 0.0448 against stock's 0.0373). The gate's lock-scale rule is the check that matches the eye here.
  `--strand-smooth N` (default 16) diffuses the flow over N rings of neighbors inside a lock before the strands are traced. Neighbors that point more than about 70° away (a parting) are left out. The rebuilt strips bend by 20 to 55° at some welded rows, and a strand traced over such a bend changes angle at a straight line, which reads as a cut. 6 rings left the back-center step; 16 removed it. A bend that follows the rebuild's strip layout (mark 7) is still there at 16.
  Two neighbor fibers can leave a dark gap 1 to 2 texels wide, which renders as a one-pixel line and counts as dots in `gaw defects`. Gaps narrower than 0.4 of a fiber cell are filled. Wider fibers (34 texels), softer fiber edges and weaker fine fibers were all tried against `gaw defects`; each one did worse on specks, flat patches or close-look sharpness.
- **Flat patches**: 7×7 windows with lightness std below 1.6 in pieces of 60 px or more. A larger share than stock fails. This is the plastic look. Bright strands at the top of the sRGB gamut used to clip to one cream tone. Lightness now passes through a soft knee under the brightest in-gamut L, and fibers thin out along their length but never vanish.
- **Layer jumps**: where one lock's edge crosses another, strand direction differs by more than 40°. Counted per 1000 hair pixels; more than `--jump-tolerance` over stock fails. Strand direction comes from the flow, so a flow that drifts from the stock paint shows as diagonal strands meeting vertical ones. `--paint-strong` trusts the paint more where it runs close to the away-from-root direction. A higher value or a wider `--paint-strong-deg` brought back flow breaks at the crown sides (0.7 gave 1 to 26 breaks).
- **Texture islands and holes**: alpha pieces under `--island-small` texels inside a chart. Any fails. The cutout is cleaned before and after the tip notches (`--alpha-min-area`, `--alpha-min-width`).

`close-look-layers.png` colors each lock in the close-up views (row 1 stock, row 2 ours; back faces darker; gray is another surface). Look at it when specks or jumps fail: a mixed dot pattern of two colors is depth fighting.

The detectors live in `wb_core::hairlook` (`measure` for a render with a layer id per pixel, `alpha_islands` for a sheet with a chart id per texel) and work on any part, not only hair.

Grown widow's-peak triangles have no stock paint to follow. With surface strands they take the guide of the cards they weld to, and the bangs shadow (`--grown-shade`), center crevice (`--crevice`) and root line (`--root-line`) default to 0. With `--fiber-space chart` those default to 20, 26 and 14.

Then run the gate and the close-look sheet from the Modding folder (`references/defects.md`, "The gate"):

    gaw gate --spec skills/gaw/references/zidane-hair-gate.json --out-dir NEW --ref-run experiments/goldens/calibration/gate-stock/candidate OUT/Zidane-hair.glb
    gaw closelook --marks skills/gaw/references/zidane-hair-marks.json --reference exports/Zidane-idle.glb --candidates OLD.glb OUT/Zidane-hair.glb --labels stock,old,new --out-dir NEW2

Two counts in the gate include things that are not cuts on the rebuilt mesh (found on v5, 2026-11-16). Direction-break counts a front face and its back face as a pair: 79 of v5's 104. Layer-direction-cut fires on welded inner edges where the texture runs on and a strand edge happens to lie along the triangle edge: 87 of v5's 102. Before tuning against those counts, check where the boxes are; `experiments/zidane-hair-tool-v5/REVIEW.md` has the split. The gate renders with the mip picked from triangle area, so a card seen nearly edge-on reads as one flat tone and crisp strands on it can turn into one-pixel lines.

Still crop the sheets at full resolution and look. Passing these checks is necessary, not sufficient.

### Comb and flow files

`strokes.json` (written by the comb tool, read by `hair --comb`):

```json
{"gaw_strokes": 1, "source": "path/to/model.glb",
 "strokes": [{"mesh": 1, "weight": 1.0, "radius": 0.05,
              "points": [{"tri": 812, "bary": [0.2, 0.3, 0.5]}, {"tri": 815, "bary": [0.6, 0.2, 0.2]}],
              "positions": [[0.1, 4.2, -0.5], [0.1, 4.0, -0.5]]}]}
```

- `mesh` is the GLB mesh index. `tri` indexes that mesh's triangles in file order (indices 3k to 3k+2), not the hair tool's own numbering. `bary` sums to 1.
- Points run from root to tip. Each segment between two points sets the strand direction where it lies.
- `weight` (default 1.0) says how strongly the stroke overrides the stock paint. `radius` (model units; default `--comb-radius`) says how far it spreads.
- `positions` holds rest-pose points for readers without the GLB. `hair` ignores it.

`flow.json` (written by `hair` to its output folder, from the stock-mesh pass):

```json
{"gaw_flow": 1, "source": "path/to/model.glb", "mesh": 1, "comb_strokes": 0,
 "triangles": [{"tri": 812, "dir": [0.0, -0.9, 0.3], "region": 4, "coherence": 0.8, "ambiguous": false}],
 "breaks": [{"a": 785, "b": 807, "angle_deg": 79.4}]}
```

`dir` is the unit root-to-tip tangent in model space. `breaks` lists the joins the check calls a flow break.

`gaw judge --strands MODEL.glb [--ref STOCK.glb] [--out-dir D]` measures the direction painted strands run and scores how well they continue across UV seams. It separates bad hair layouts from good ones, but it also flags stock hair, so use it to compare versions, not as a pass/fail gate. Seam maps in `--out-dir`: green continues, red breaks.

## shells

`gaw shells file.glb` lists each connected piece of every mesh: closed or open, thickness, bones, nearest piece, and LOOSE or LOOSE CARD status. Run it after any geometry edit. A new LOOSE piece is a floating part.

## belt strap

1. `strap-rebuild file.glb --out new.glb` replaces the flat strap card with a solid strap tucked under the waistband. Give the exit by the user's marked edge (`--exit-edge`, `--exit-curve`, `--exit-down`); with no flag it uses the waist paint.
2. `strap-check new.glb --out-dir DIR [--reference ref.glb]` draws close-ups (front, sides, 3/4, underside) at rest and in clips, with intersections, cracks and open edges named by triangle. `--reference` also checks the protected part of a reference strap.
3. `strap-sweep before.glb after.glb --out-dir DIR` draws the strap through each clip. Count flagged poses; look at the sheets.

A request that marks a strap edge gives you the exit position directly (`references/mark-and-fix.md`).

## Thin lines and seams in paint

- `line-repaint file.glb --out new.glb --uv u0,v0,u1,v1 --pad 8` repaints thin dark horizontal lines in one UV box from the texels above and below. Only that image changes. A line that comes from an island ending over a dark gutter needs `--pad` so filtering at the island edge stops showing the gutter.
- `boundary-check` labels each thin line or speck in the views (uv seam, clothing edge, intentional overlap, accidental paint mark, silhouette) and renders under texel, smooth and gameplay-distance filtering. Run it before deciding a line is a bug. `--probe x,y` prints the triangle, UV and texel under a screen pixel of the first view, which is how you find the UV box for a fix.
- Read the labels with their limits. Same-paint strokes under 12 pixels count as grain, long dark strokes as painted creases (overlap), and light strokes that are a lighter shade of the paint beside them as painted highlights. A different material painted in (a spill), and its soft edge, always stays a mark. The tuning flags are in `gaw help`.
- `paint-fix --op marks|edge|seams|pad` edits paint while keeping intentional overlaps and clothing edges, and writes a before/after/changed sheet. It refuses to write when a protected texel changed or nothing changed. Protect hand-picked texels with keep boxes; fix a speck the finder misses (one joined to a hem by a thin neck) with a hand-marked box.
- A dotted light or tan line along a UV seam usually means the island's paint stops one or two texels short of the triangle edge, so the edge texels show gutter or stray color. Use `--op pad` with `--reclaim 2` on each island side of the seam, one box per island, then check the view again. A seam can run across several islands: probe along the line and cover each one.
