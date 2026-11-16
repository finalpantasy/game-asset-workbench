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
- `boundary-check` labels each thin line or speck in the views (uv seam, clothing edge, intentional overlap, accidental paint mark, silhouette) and renders under texel, smooth and gameplay-distance filtering. Run it before deciding a line is a bug.
- `paint-fix --op marks|edge|seams|pad` edits paint while keeping intentional overlaps and clothing edges, and writes a before/after/changed sheet.
