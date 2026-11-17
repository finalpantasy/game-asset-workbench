# defects: where exactly is it messy?

`gaw defects` turns "it looks messy" into a list of specific places to fix. Give it a candidate GLB, optionally a reference GLB and a part. It draws close-ups and game-distance views, runs eleven detectors, and writes `defects.json` plus one crop sheet per kind. Flags: `gaw help`. Three flags help when a number is not enough: `--all-views` writes every view plus `views/overview-<view>.png` with each defect boxed in its kind's color (yellow jagged, magenta discoloration, cyan stripe, orange direction, blue gutter, red blur, green flat, white speck, pink layer cut, lime lock scale), `--probe VIEW:X:Y` prints what sits under one pixel (mesh, triangle, island, UV, texel, alpha around it, whether it is in the part), and `--dump-tiles` writes the per-tile numbers to `tiles.json`.

```
gaw defects cand.glb --part hair --ref stock.glb --out-dir OUT
```

`--out-dir` must be new or empty. `--part` takes `hair`, a kit part, a bone group (`head`, `torso`, `arm_L`) or a mesh name; several names separated by commas. A candidate with a new atlas has no kit parts, so use a bone group there. `--ref` fits the cameras to the reference's part, so two models are drawn the same way and their loads compare.

## The eleven detectors

| Kind | What it measures | Needs |
|---|---|---|
| `jagged-edge` | Stair steps along alpha-cutout and silhouette contours: contour roughness after smoothing, and how many up-and-down steps a stretch has | close views |
| `discoloration` | Patches of the part far from the part's own palette and from the ring around them | close and game views |
| `seam-stripe` | A color step across a UV seam that is continuous in 3D (the two sides should match and do not) | mesh and texture |
| `direction-break` | Hair strands turning across a seam, or inside a card (same measure as `judge --strands`) | `--part hair` |
| `dark-gutter` | Texels just outside an island that are much darker or lighter than the island's border (padding and mip bleed) | mesh and texture |
| `smear-blur` | Tiles (16 px) with much less contrast across the strands than the reference's median: strands blurred into a smear | close views, `--ref` |
| `flat-patch` | Tiles (8 px) or whole triangles with almost no colour variance next to the reference's median: plastic wedges and one-tone cards | close views, `--ref` |
| `speck` | Dots (a pixel that differs from its 3 x 3 median or from all four neighbours, so a dither checker counts) and cut-off 5 to 60 px fragments of the part | close views |
| `alpha-stipple` | Tiny closed holes and tiny opaque scraps in the alpha of the texels the part covers (pin holes, stipple at the cutout) | texture |
| `layer-direction-cut` | A run of pixels where one triangle ends and the next starts, with the colour two pixels either side differing hard: a layer that stops dead | close views |
| `lock-scale` | 48 px tiles whose share of fibre-scale energy (1 to 3 px) among all four scales (fibre, strand, lock, broad) is at least 2.0x the reference's median for the view: fine wood-grain stripes where the reference has broad locks | close views, `--ref` |

Every defect has a score. 1.0 sits on the kind's limit, higher is worse. The limit constants live in `wb-core/src/defects.rs` and are provisional (calibration below).

## Reading the output

- Summary lines, one per kind: defect count, worst score, load (the share of the measured edge or area that is flagged), where the worst one is. With `--ref`, the reference's count and load follow after `| reference:`. Compare loads, not counts: counts depend on how a run of bad edges splits.
- `sheet-<kind>.png`: numbered rows. Left, the render crop with the screen box. Right, the texture crop with the texel box. The number matches the id suffix in `defects.json`.
- `defects.json`: each defect has `id`, `kind`, `score`, `severity` (low under 1.3, medium to 2.0, high above), `measure`, `note` (the plain reason), `marked` (triangles by mesh, the same shape as `request.json`), `texel_box` and `texel_boxes` (`[x0,y0,x1,y1]`, end exclusive, with the sheet name), `island`, `view`, `picture` (`views/<view>.png`), `screen_box` (pixels of that picture), `camera` (same shape as a request's view camera), `seen_in` and `seen_boxes` (the screen box of the same place in each other view it was found in). A defect the drawn views cannot see has `view: null`: use the texel box.
- Open the sheets and the view pictures. The numbers rank places; the picture says whether the place is a defect or a design feature (a white strap stripe on a blue vest reads as a seam stripe and is not one).

## Using it

1. Run on the candidate with `--ref` stock and `--part`. Read the summary, open the sheet of the worst kind.
2. Fix the worst few places by id. The `texel_box` is where to paint, the `marked` triangles are what to edit, the `camera` is where to look.
3. Run again into a new folder. The load of the kind you worked on should fall; the others should not rise.
4. To hand places to the user as marks, see `references/mark-and-fix.md` (load `defects.json` as marks).

## What it can and cannot tell

- It ranks versions of one part. Load per kind falls as hair gets cleaned up (calibration table), and it separates the rejected hair v2 from the accepted later passes on stripes, directions and discoloration.
- Stock and procedural hair also score high on stripes. It says "rougher than", not "good".
- Direction breaks are measured in texture space (`strands.rs`), not as a screen-space structure tensor. Screen-space strand orientation is not built.
- Blur and flat compare with the reference's median for the same view; with no `--ref` they use the part's own median, so a part that is smeared everywhere will not show.
- Layer cuts and specks are ranked by place, not by load. Stock has more hard layer edges than the rebuilt hair, so its load is higher; read the overview pictures.
- Reference comparison is by load per kind. There is no per-pixel match, because hair geometry differs between versions.
- Mesh-based defects hidden from every drawn view carry texel boxes only.
- Limits are provisional until more known-good and known-bad pairs have been run. Re-run `bash experiments/goldens/calibrate-defects.sh` after changing a limit and update the table.

## Calibration (reference = stock, `--ref` on)

Cell = defects, worst score, load. Loads: stripe and direction are the share of seam or strand edges flagged; discoloration is the share of part pixels; jagged is the share of contour pixels; gutter is the share of border texels. Hair is `--part hair`, belt is `--part torso`. Reproduce with `bash experiments/goldens/calibrate-defects.sh` (about 1 minute a model); it writes `experiments/goldens/calibration/defects-<name>/` and the tables.

| Hair version | Jagged | Discoloration | Seam stripe | Direction | Gutter |
|---|---|---|---|---|---|
| stock | 5, 1.7x, 0.2% | 2, 1.3x, 0.0% | 5, 2.3x, 67.9% | 7, 1.3x, 13.7% | 3, 1.7x, 5.7% |
| v2 (rejected) | 9, 2.3x, 0.6% | 9, 2.2x, 0.1% | 6, 1.3x, 28.1% | 14, 1.9x, 19.1% | 0 |
| v2-b (rejected) | 9, 2.3x, 0.6% | 9, 1.5x, 0.1% | 6, 1.3x, 28.1% | 14, 1.9x, 19.1% | 0 |
| v4e | 9, 1.9x, 0.1% | 3, 1.3x, 0.1% | 4, 1.5x, 1.4% | 25, 1.8x, 7.2% | 0 |
| v6j | 8, 2.0x, 0.2% | 0 | 3, 1.9x, 1.1% | 28, 1.8x, 8.6% | 1, 1.2x, 0.2% |
| v7z | 8, 2.0x, 0.2% | 1, 1.2x, 0.0% | 3, 1.9x, 0.9% | 28, 1.8x, 8.7% | 0 |
| v8i | 8, 2.0x, 0.2% | 3, 1.3x, 0.0% | 3, 1.5x, 0.7% | 22, 1.8x, 6.1% | 0 |
| v10d | 8, 2.0x, 0.2% | 1, 1.3x, 0.0% | 3, 1.6x, 0.7% | 21, 1.8x, 5.8% | 1, 1.4x, 0.1% |
| v11k | 8, 2.0x, 0.2% | 3, 1.4x, 0.1% | 4, 1.6x, 1.3% | 15, 1.5x, 4.0% | 3, 1.5x, 0.1% |
| v12d | 8, 1.9x, 0.2% | 1, 1.3x, 0.0% | 4, 1.6x, 1.3% | 15, 1.5x, 3.9% | 3, 1.5x, 0.1% |
| codex-zidane-hair-v1 | 42, 3.0x, 3.1% | 0 | 17, 1.9x, 62.8% | 54, 1.8x, 21.8% | 0 |

| Belt version | Jagged | Discoloration | Seam stripe | Gutter |
|---|---|---|---|---|
| stock | 2, 1.8x, 0.0% | 22, 1.3x, 2.5% | 6, 2.1x, 26.6% | 1, 1.2x, 4.0% |
| stock belt painted v13 | 2, 1.8x, 0.0% | 4, 1.1x, 0.1% | 3, 3.8x, 6.3% | 3, 1.9x, 0.9% |
| strap v2 v13 | 2, 1.8x, 0.0% | 4, 1.1x, 0.1% | 3, 3.8x, 6.3% | 3, 1.9x, 1.0% |
| strap v3 v13 | 2, 1.8x, 0.0% | 4, 1.1x, 0.1% | 3, 3.8x, 6.3% | 3, 1.9x, 1.0% |
| v3 shirtfix | 2, 1.8x, 0.0% | 4, 1.1x, 0.1% | 3, 3.8x, 6.3% | 1, 1.9x, 0.4% |
| v3 boundary combined | 2, 1.8x, 0.0% | 4, 1.1x, 0.1% | 3, 3.8x, 4.6% | 1, 1.9x, 0.4% |
| v3 boundary combined 2-3-4 | 2, 1.8x, 0.0% | 4, 1.1x, 0.1% | 3, 3.9x, 4.6% | 0, 0.2% |

Does it rank like a person?

- Hair, mostly yes. The rejected v2 and v2-b carry 28% stripe load and 19% direction load against 1% and 4 to 9% for the accepted passes, plus the worst discoloration (2.2x). v11k and v12d have the lowest direction load of the cleaned versions (4%), v4e to v7z the highest (7 to 9%). The procedural codex-v1 is the worst on everything but discoloration.
- Stock hair scores high (68% stripe load). Its atlas has real color steps at card edges. A person may call stock acceptable; the detector says only "rougher than the repaints".
- Jagged edges do not separate the cleaned passes: 8 to 9 defects at 1.9 to 2.0x each. They share one silhouette. Only codex-v1 stands out (42, 3.0x).
- Belt, in the order a person would expect: the gutter and stripe loads fall from v13 (1.0%, 6.3%) through shirtfix (0.4%, 6.3%) to combined (0.4%, 4.6%) to combined 2-3-4 (0.2%, 4.6%). The stripe worst stays near 3.8x in every repaint: it is the cream and brown belt seam on the front, still a real step (open `sheet-seam-stripe.png`). v13 against v2-v13 against v3-v13 do not separate: those differ in the strap, not in paint.
- The white strap stripes on the blue vest are a design feature that reads as a stripe at 3.8x. Read the sheet before fixing a top-ranked stripe.

## The close-look marks: what it finds (hair v3, `--ref` stock)

The hair session's nine marks on `close-look-coordinator.png`, scored against a run into `experiments/goldens/calibration/defects2-v3e/`. A place counts as found when a box of the right kind lands on it in the overview picture, or when its texel box covers the texels `--probe` reports.

| Mark | Found as |
|---|---|
| 1 flat wedge | `flat-patch` (two flat triangles at the bottom of the forehead V), plus `layer-direction-cut` |
| 2 and 8 specks | `speck` (dither and dots on flat cards); speck-001 holds the worst, with texel box over the dotted card |
| 3 and 4 shards | `flat-patch` (one-tone cards), `speck` (dotted shard edges), `direction-break` |
| 5 stipple | `speck` (a dithered patch beside the ear; alpha is 255 there, so `alpha-stipple` stays 0: the noise is colour, not alpha) |
| 6 and 9 smear | `smear-blur` |
| 7 hard cut | `layer-direction-cut` plus `direction-break` |

Loads against stock, hair v3 and earlier versions (defects, load; stock row is the reference itself):

| Hair | smear-blur | flat-patch | speck | layer cut |
|---|---|---|---|---|
| stock | 12, 4.1% | 9, 0.7% | 27, 0.1% | 29, 18.1% |
| v2 (rejected) | 3, 0.6% | 10, 0.7% | 23, 0.0% | 28, 5.6% |
| v12d | 23, 15.5% | 36, 2.8% | 19, 0.0% | 83, 5.0% |
| v3 | 26, 19.0% | 73, 2.3% | 37, 0.1% | 89, 9.2% |

Blur and flat separate v3 and v12d from stock and v2 by count and load. Speck count barely moves, so read the ranking and the overview. Layer-cut load is lower for the rebuilt hair than for stock, so use its places, not its load.

## The gate: pass or fail against stock

`gaw gate --spec GATE.json --out-dir NEW_DIR [CANDIDATE.glb]` runs `gaw defects` for the reference and the candidate, applies the spec's rules, prints PASS or FAIL for each with the numbers, writes `gate-report.json` (numbers, rule results, both `defects.json` runs under `candidate/` and `reference/`) and exits 1 on any FAIL. `--ref-run DIR` reuses a finished reference run. A run takes about 30 s. The hair spec is `skills/gaw/references/zidane-hair-gate.json`; marks come from `zidane-hair-marks.json` next to it (the nine close-look marks in orbit-view pixels). Run it from the Modding folder: the spec's paths are relative to the current folder.

Two rule types:

- `ratio`: `kinds` (summed), `measure` (`count`, `load` or `score`), `max_x` (the candidate may have at most this many times the reference's value), optional `floor` (the reference value is raised to at least this before multiplying, so a reference near 0 does not make the limit 0).
- `marks`: `kinds`, `marks` (ids from the marks file), `allow`: `reference` (the candidate may have no more boxes of those kinds in a circle than the reference has there) or `none` (zero). A box counts when it touches the circle in the mark's view, whether it was found there or seen there (`seen_boxes`). A candidate that is the reference passes every `reference` rule.

Hair gate on `--ref exports/Zidane-idle.glb` (stock passes against itself):

| Rule | stock | v3 | v4 |
|---|---|---|---|
| smear-blur load, limit 1.0x | 4.1% pass | 19.0% FAIL | 0.1% pass |
| flat-patch load, limit 1.0x | 0.7% pass | 2.3% FAIL | 0.0% pass |
| direction-break count, limit 1.5x | 7 pass | 113 FAIL | 144 FAIL |
| layer-direction-cut count, limit 1.5x | 29 pass | 89 FAIL | 165 FAIL |
| lock-scale load, limit 1.0x | 10.6% pass | 0.0% pass | 45.8% FAIL |
| marks 1, 6, 7, 8, 9 (cut, smear, flat) | pass | FAIL (all five) | FAIL (all five) |
| marks 2, 3, 4, 5, 8 (speck) | pass | FAIL (2, 4, 5) | FAIL (2, 5) |

v4 passed the old numbers (smear and flat near zero) and failed at full size: straight cuts across the strands where cards meet (marks 1, 6, 7, 8, 9) and fine wood-grain stripes where stock has broad locks. The gate now fails it on both. The cut boxes come from `layer-direction-cut` (marks 1, 6, 7, 9) and one `direction-break` (mark 8); lock scale is the new kind.

## close-look sheets: the same crop for every model

`gaw closelook --marks MARKS.json --reference REF.glb --candidates A.glb B.glb --out-dir NEW_DIR` draws each mark's crop from the same orbit view of every model (cameras fitted to the reference's part, so crops line up between models and with the `screen_box` values in `defects.json`), reference first, side by side with labels. It writes `closelook.png` plus `closelook-items-A-B.png` splits (5 marks each, `--rows`). Use it after every gate run: the gate says where, the sheet says whether you agree. Crops come from the orbit views, so they frame a mark slightly differently than the hair session's hand-made pairs (`close-look-pairs-*.png`, drawn with another camera); the places are the same.

Reference run for the hair marks: `experiments/goldens/calibration/closelook-stock-v3-v4/`.

## lock-scale: how it works

For each 48 px tile fully inside the part (85% of pixels at least 4 px from the edge), luminance is box-averaged at radii 1, 3, 8, 20 and 48 px, and the squared differences between neighbouring radii give four energies: fibre, strand, lock and broad shading. The tile's fibre share is `fibre / (fibre + strand + lock + broad)`. Tiles with 2.0x the reference's median share for that view, in groups of 3 or more connected tiles, become boxes. Load is the share of measured tiles flagged. Stock against its own median flags 10.6% (the reference's own spread, so the gate compares against that figure, not against zero); v3 0.0% (blurred, no fine stripes), v4 45.8%. The first try, lock energy over fibre energy, did not separate v4 from stock (24% of tiles against 17%); the fibre share did. Limits: `LOCK_TILE` 48, `LOCK_RADII` 1, 3, 8, 20, 48, `LOCK_RATIO` 2.0, `LOCK_MIN_TILES` 3 in `wb-core/src/defects.rs`.
