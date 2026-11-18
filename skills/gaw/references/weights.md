# weights, deform-check and deform: fit a new mesh to the stock skeleton, then test it moving

Two commands for a new or remeshed character that must run on FF9's 27-bone field skeleton and then go through `gaw fbx`. `gaw weights` gives the new mesh its skin weights from a stock model. `gaw deform-check` poses the result through every clip and fails it if it tears, collapses, passes through itself or carries bad weights. Flags: `gaw help`.

## Order

1. Fit the new mesh to the stock rest pose in Blender: same scale (the mesh as tall as stock's), standing where stock stands, apply all transforms. Two ways in:
   - Mesh only (what an outside modeler or an image-to-3D tool gives you): no armature needed, `gaw weights` copies stock's in (`--adopt-skeleton`).
   - A GLB that already carries stock's 27 joints (same names, order, parents and rest pose).
2. `gaw weights new.glb --from stock.glb --out new-weighted.glb --report weights.json`, plus `--adopt-skeleton` for a mesh-only file.
3. `gaw deform-check new-weighted.glb --ref stock.glb --out-dir NEW_DIR`. Always pass `--ref`; see "Why --ref" below.
4. Open `NEW_DIR/sheet-*.png` and look at the numbered markers. Fix the cause, rerun.
5. `gaw fbx new-weighted.glb --out DIR --readback` (`references/fbx.md`).

## weights

Each new vertex takes the weights of the closest point on the reference surface (barycentric blend of that triangle's corners). Result: at most 4 influences, sum 1, bone names and order kept.

No smoothing by default. Stock's weights have hard cuts (thigh to shin to foot are one body region but move as separate rigid pieces), and every smoothing ring blurs them. Measured with `deform-check --ref stock` on the stock mesh onto itself, the remesh and the noisy remesh (see the calibration section): rings 0 passes all three; rings 1 to 3 each fail at least one (stock itself fails at 3: stretch 4.22 against 3.69, squash 3.66 against 2.74). `--rings N` (strength `--smooth`) adds N rings over the new mesh's welded topology, mixing only vertices of the same body region (L_arm, R_arm, L_leg, R_leg, head, tail, torso). Use it only when a sheet shows jagged weight noise, and run `deform-check` after.

- Refuses, with a problem list, if the skeleton is not the reference's: joint count, names and order, parents, rest pose, inverse bind matrices (tolerance 0.001). A different character (Vivi onto Zidane) stops with `refusing: ... does not have the skeleton of ... (12 problem(s))` and lists the first joints that differ. Nothing is written.
- `--bones a,b` lets only those bones take weight; weight the reference puts on others goes to the nearest allowed parent. `--zero-bones tail` is the reverse (names, joint numbers, or text a name contains).
- `--max-dist D` (default 5 percent of the reference's height) flags vertices that sit far from the reference surface. They still take the closest point's weights; look at them in deform-check. `weights.json` lists them with mesh, vertex, distance and region.
- Writes only JOINTS_0 and WEIGHTS_0 (plus the skin); the rest of the file keeps its bytes. Never overwrites an input or an earlier output (`--force` replaces an earlier output).
- `battle_model*` overlay meshes of the reference are not used as surface unless `--use-battle-meshes`.
- It is a transfer, not a rigging tool. A new surface that is not near stock's (a bigger coat, a cape, long hair) gets the weights of whatever stock surface is closest, which is often wrong. Check `weights.json` distances and the deform sheets; paint those parts by hand.

### --adopt-skeleton: a mesh with no armature

`gaw weights bare.glb --from stock.glb --adopt-skeleton --out OUT.glb` copies stock's joint nodes (and the nodes above them), skin with inverse bind matrices, and clips into the mesh file, points every mesh node at the skin, then transfers as above. The mesh, materials and textures keep their bytes. Before copying it checks the mesh against stock's rest-pose box and refuses with the reason:

- height, width (x) and depth (z) each within `--fit-tol` (default 0.10, 10 percent) after scaling to stock's height, and the bounding box center within that fraction of stock's height on each axis;
- a wrong size prints the scale to apply: `height 453.1295 against the reference's 4.5313 (+9900.0%): scale the mesh by 0.0100 about the origin in Blender (Apply Scale), then export again`, with a hint when the factor is near 100 (centimeters against meters, or FF9's 0.01 model scale). Stock Zidane is 4.53 units tall in the exports;
- a file that already has a skin (drop the flag, the normal skeleton check applies);
- mesh nodes moved by their own or a parent's transform: a skinned mesh ignores node transforms, so the mesh would jump. Apply the transforms in Blender and export again.

`--adopt-skeleton REF.glb` also reads (REF then needs no `--from`). The adopted skeleton is read back and compared with stock's before anything else runs. `--fit-tol` widens the check; a model with no tail, or a much wider cape, needs it. The check cannot tell a shape that is the right size but the wrong pose (arms raised); look at a render.

### A body that is not stock's: --radial, --repair, --polish

Plain closest point assumes the new surface sits on stock's. An original character on the stock rig (Corvin v9: 14,670 vertices, two layers of skin and clothes, thicker limbs, its own `stand` clip, bone-heat weights from Blender) does not. Default transfer on it fails `deform-check --ref stock` worse than the bone-heat weights it was meant to replace.

Why: stock's cuts between bones sit where stock's surface changes bone, which on another body is somewhere else. The forearm-to-hand cut crossed 232 triangles against 14 for bone heat, and every crossing edge is a rigid cut that the `stand` pose tears open. Skin and clothes also land on different stock surfaces, so a collar vertex can take spine where bone heat gives chest or neck. Smoothing rings do not help (they blur the places that were fine).

What to run, in order of how far the body is from stock:

- Body close to stock (remesh, noisy remesh): plain transfer, then `--repair 40 --repair-reach 1 --repair-anchor 0.2` if a sheet shows tears. On the noisy remesh this took stretch 2.67 to 0.35 and squash 1.98 to 0.44, still PASS.
- Body of other proportions: `--radial 0.25 --soften 0 --repair 40 --repair-reach 1 --repair-anchor 0.2`. `--radial R` matches each vertex to stock near the same bone (distance along the bone in full, across it times R), moves stock's cuts onto the new joint planes (`--keep-cuts` leaves them), and `--soften K` averages reference triangles about as close as the best one. Do not use `--radial` on a body that matches stock: on the noisy remesh `--radial 0.25 --soften 0` alone gave volume 130.77 against 90.88 and FAIL.
- Weights you already like (Blender bone heat, hand painting) that fail only in places: `--polish --repair 40 --repair-reach 1 --repair-anchor 0.2` on the file, with `--from stock.glb`. No transfer happens; only flagged vertices change. Keeps what bone heat did well (intersect 7.19 to 7.61, volume 73.1 to 72.9).

`--repair N` poses the model in every clip (stock's when it has none), flags vertices on edges past deform-check's 1.5 stretch or 0.5 squash, and for each tries candidates: blends with its neighbors' weights and with its own three main bones. It keeps the one with the least stretch and squash over all frames, less `--repair-anchor` times how far it moved from the start. After each pass it scores the whole model (stretch and squash per body region, joint volume, new crossings, each against 1.5x stock's value, steep past 85 percent of the limit) and keeps the best pass, so it cannot end worse than it began. `--repair-reach N` lets N rings of neighbors around each flagged vertex change too. `--repair-mode soften` blurs toward the neighbors instead (the first version; it cannot lower a stretch caused by two legs pulling a shared vertex apart). 30 to 60 s on this body. `GAW_WEIGHTS_DEBUG=1` prints the score per pass.

Corvin v9 through `deform-check --ref stock --clips all` (72 frames), value / limit:

| Weights | stretch | squash | volume | intersect | verdict |
|---|---|---|---|---|---|
| bone heat (v9) | 6.13 / 3.69 (R_leg) | 2.83 / 2.74 (R_leg) | 73.13 / 90.88 | 7.19 / 9.66 | FAIL |
| transfer, defaults (rings 0) | 11.2 / 2.80 (L_arm) | 4.57 / 2.59 (R_arm) | 55.5 / 90.88 | 8.52 / 9.66 | FAIL |
| transfer, `--radial 0.25 --soften 0 --repair 40 --repair-reach 1 --repair-anchor 0.2` | 3.55 / 3.69 | 1.59 / 2.97 | 47.93 / 90.88 | 8.98 / 9.66 | PASS |
| bone heat, `--polish --repair 40 --repair-reach 1 --repair-anchor 0.2` | 3.58 / 3.69 | 1.59 / 2.97 | 72.93 / 90.88 | 7.61 / 9.66 | PASS |

Stretch and intersect of the transfer row pass by little (0.14 and 0.68 under the limit). The polished file keeps more room on intersect and is the one to ship; the transfer row shows the stock-weight route now works on a body that differs.

The `stand` thigh and knee, within 4 influences: yes for the knee and most of the thigh, no for one crotch triangle. Bone heat's inner thigh (mesh0 triangle 5639, R_thigh, `stand` at 0.74 s) stretched an edge 4.19x. Changing the weights of the vertices around it (664 welded places in the whole polished file) brought it to 2.37x, R_leg stretch from 6.13 to 3.58 and R_leg squash (back of the knee) from 2.83 to 0.76. The three vertices of that triangle already carry both hips and both thighs, all four slots, and the legs move apart in `stand`, so no weight vector gets it under 1.5x. It stays inside the limit because stock's own R_leg stretch is 2.46. Getting it under 1.5x takes mesh changes at the crotch, not weights, and the engine has no corrective shapes.

The repair only sees frames of the clips. A pose outside them is not tested. Two limits seen: it cannot fix a crossing between two layers by itself (a pull toward the crossing partner raised volume loss past the limit and was dropped), and the transfer row's intersect is about the collar and shoulder, where stock's spine weight reaches higher than Corvin's neck.


## deform-check

Poses every clip (idle, walk, run, turn_l, turn_r) at 12 fixed steps and measures, against the model's own rest pose:

| Kind | What it counts |
|---|---|
| stretch | share of surface area in triangles whose edges grew past 1.5x (candy-wrapper twist, spikes) |
| squash | same for edges that shrank below 0.5x |
| volume | volume lost in the cone around each joint (elbow, knee, hip collapse) |
| intersect | new triangle crossings between body parts since rest (hand in hip, arm in torso); crossings that exist at rest are ignored |
| weights | vertices whose weights do not sum to 1, use more than 4 bones, or name a missing joint |

Stretch and squash are measured on the whole surface and on each body region; the worst scope is the kind's value, and the kind passes only if every scope passes (a bad forearm is a small share of the whole body and would hide otherwise). An edge only counts if it changed by at least 0.6 percent of the model's height, which keeps remesh slivers from counting.

Output in `--out-dir`: `deform.json`, one `sheet-<kind>.png` per kind (numbered markers on posed close-ups, red triangle outline, whole-body inset, caption with bone, clip and time), printed PASS or FAIL per kind. Exit code 1 on FAIL.

`deform.json`: `verdict`, `measures[]` (value, reference value, limit and the rule that set it, `worst[]` with mesh, triangle, bone, clip, time, position and the reference's value at the same place and frame), `limits`, and `summary[]` shaped like `defects.json`'s so a gate can read it.

### Why --ref

Stock has its own flaws: it passes through itself at the pockets and thighs, and its torso stretches in run. With `--ref`, every measure is also taken on the reference at the same frames and the limit is `max(1.5 x reference, floor)`; floors: region 2.5 percent, whole-surface stretch and squash 1, volume 10, intersect 1; weights may not exceed the reference's. Worst places are ranked by how much they exceed the reference at the same place and frame, so a spot that stock also has ranks low.

Without `--ref` the limits are absolute, calibrated on stock: stretch 3.0 (region 9.0), squash 1.5 (region 4.0), volume loss 85, intersect 10, weights 0. The absolute limits are loose because stock's torso already uses 5.77 of the 9.0. They catch big breaks (hand in hip, five influences) and miss small ones (the forearm case below). Use them only when there is no reference.

### Calibration (run on this machine, 2026-11-16)

All with the stock Zidane GLB (`exports/Zidane-idle.glb`) as reference. Value / limit per kind; the stretch and squash value is the worst scope.

| Case | Verdict | stretch | squash | volume | intersect | weights |
|---|---|---|---|---|---|---|
| stock vs itself, `--ref` | PASS | 5.77 / 8.66 | 1.82 / 2.74 | 60.58 / 90.88 | 6.44 / 9.66 | 0 / 0 |
| stock, no `--ref` | PASS | 5.77 / 9.00 | 2.48 / 4.00 | 60.58 / 85.00 | 6.44 / 10.00 | 0 / 0 |
| forearm weights moved to the upper arm | FAIL (`--ref` only) | **3.38 / 2.80** (L_arm) | 1.82 / 2.74 | 60.58 / 90.88 | 6.32 / 9.66 | 0 / 0 |
| 13 chest vertices with 5 influences | FAIL | 2.85 / 2.80 | 1.87 / 2.57 | 60.58 / 90.88 | 6.44 / 9.66 | **13 / 0** |
| left hand pushed into the hip (walk clip) | FAIL | 1.88 / 2.80 | 2.00 / 2.97 | 60.58 / 90.88 | **13.19 / 9.66** | 0 / 0 |
| all three together | FAIL | 4.35 / 2.80 | 1.87 / 2.57 | 60.58 / 90.88 | 8.57 / 9.66 | 13 / 0 |

Markers land where the damage is: the forearm case on the L hand and wrist junction, the influence case on the 13 chest vertices (triangle and bone listed), the hand-hip case on the torso and left forearm triangles at the walk frames where the hand is inside the hip. The numbers above are in `experiments/weights-deform-v1/calib/*/deform.json`; sheets sit beside them.

The stock rows are the point of calibration: stock passes itself, so its own stretch and crossings do not fail anything, and each broken copy fails on the kind it was broken in.

### Transfer cases

Inputs live in `experiments/weights-deform-v1/`: stock subdivided once and decimated to half in Blender (surface kept, 6081 vertices against 3830), the same with every vertex moved by up to 1.2 percent of the model's height (a rough surface), and bare-mesh copies of stock and of the remesh made by `strip_to_mesh_only.py` (no skin, no clips, no bone nodes). Each gets stock's weights with the default settings, then `deform-check --ref stock`. Value / limit:

| Case | Verdict | stretch | squash | volume | intersect | weights |
|---|---|---|---|---|---|---|
| remesh | PASS | 3.59 / 3.84 | 1.81 / 2.74 | 54.65 / 90.88 | 5.88 / 9.66 | 0 |
| remesh, rough surface | PASS | 2.67 / 3.84 | 1.98 / 2.74 | 55.09 / 90.88 | 5.35 / 9.66 | 0 |
| remesh, `--zero-bones tail` | PASS | 3.59 / 3.84 | 1.81 / 2.74 | 54.65 / 90.88 | 4.91 / 9.66 | 0 |
| bare stock mesh, `--adopt-skeleton` | PASS | 5.77 / 8.66 | 1.82 / 2.74 | 60.58 / 90.88 | 6.44 / 9.66 | 0 |
| bare remesh, `--adopt-skeleton` | PASS | 3.59 / 3.84 | 1.81 / 2.74 | 54.65 / 90.88 | 5.88 / 9.66 | 0 |
| bare remesh, `--rings 1` | FAIL | 3.47 / 2.50 (tail) | 2.44 / 2.74 | 54.23 / 90.88 | 6.44 / 9.66 | 0 |
| bare stock mesh, `--rings 3` | FAIL | 4.22 / 3.69 (R_leg) | 3.66 / 2.74 (R_leg) | 59.86 / 90.88 | 8.10 / 9.66 | 0 |

The adopted stock mesh matches stock's own numbers exactly (the transfer reproduces stock's weights on stock's own surface). Walk renders at 15, 35 and 55 percent of the walk clip, stock against the adopted remesh (`renders/walk-compare-stock-adopted-remesh-diff.png`: left stock, middle remesh, right 2x difference): silhouette overlap 0.9993 or better, mean difference 1.6 of 765, 0.3 percent of model pixels differ by more than 90, all thin shading lines where the triangulation differs. The adopted stock mesh renders pixel for pixel like stock. Same for the plain remesh (`renders/walk-compare-stock-remesh-diff.png`).

Tailless: `--zero-bones tail` leaves 23 of 27 bones able to take weight and the four tail bones with no vertex (their weight went to the hips), still 27 joints and max 4 influences. The tail triangles stay in that test file, rigid on the hips; a real tailless mesh has none, and its tail region is left out of the scopes. A first version of `deform-check` matched regions by position and read the torso's reference as the tail's 0.00, which failed this file wrongly; regions now come from the skeleton, and `cargo test -p wb-core --test weights` covers it.

Refusals worth knowing (bare copies of stock, `--adopt-skeleton`): scaled by 100 gives one message, `scale the mesh by 0.0100` with the centimeter hint; scaled by 0.8 gives `scale the mesh by 1.2500`; moved up 0.15 (3 percent of the height, inside the 10 percent limit) is accepted and its largest distance to stock's surface is 0.15, which `weights.json` and the deform sheets show.

## deform (rigkit): pinch and fold count, stock beside candidate

`gaw deform STOCK.glb CAND.glb` plays both models through every clip with rigkit's `check::stretch` and prints one row per clip, stock beside candidate. Use it after `weights` or a rebuild when you want the plain counts (how bad is the worst pinch, how many triangles turn inside out, where) rather than `deform-check`'s pass or fail. It writes nothing unless `--render-dir NEW` is given. The command is built on rigkit (`crates/rigkit` in this repo); gaw only reads GLBs with wb-core and hands rigkit plain arrays.

Each model is measured against its own rest pose. Columns: worst edge ratio (always at least 1; `pinch` means the worst edge got shorter, `pull` longer), worst area ratio, flipped triangles as `all/hard`, and a verdict. A hard flip also changed its worst edge by 1.5x or more; a flip with edges nearly unchanged is a near-degenerate sliver and does not count toward the verdict. WORSE means more hard flips than stock or a worst edge over 1.25x stock's; better means under 0.8x with no more hard flips. Under the table, each clip lists the worst triangles of each side by mesh, triangle number, dominant joint, pose time and ratios, then a per-mesh line so hair cards are judged apart from the body.

A candidate with no clip of the same name (or no clips) is posed with stock's clip on the same joint order. Vertices with no weight, or naming a joint the skeleton lacks, are held at rest and counted in the header. `--json` prints everything (including `by_mesh` and the flipped lists); `--strict` exits 1 when any clip is WORSE; `--steps`, `--top`, `--min-edge`, `--clips` tune it (`gaw help`).

`--render-dir NEW` writes `deform-stock.png` and `deform-cand.png`: per clip the worst triangle and the worst flipped one, outlined in red, posed on the left and at rest on the right, same camera. Open them. The ratio says a triangle collapsed; the picture says whether the collapse is a visible crumple or a sliver nobody sees.

Known limits:

- Stock has its own flaws (run: 28 flipped triangles, 24 hard, mostly upper arms; a 7x pull at the hips). That is why the output is a comparison. A candidate that equals stock passes.
- rigkit records a triangle's flipped flag only when an edge changed, so a pure mirror flip is in the total but cannot be named (`flipped_unlisted` in the JSON, a note in the table). It is never a hard flip.
- It measures edges and area, not crossings or volume. `deform-check` covers those.
- The default minimum edge is 0.2 percent of stock's height, the same number for both models. A denser candidate has more small edges, so its worst ratio can look larger for the same real motion. Check the picture before blaming the weights.

Measured on this machine (2026-11-17), idle, walk, run, turn_l, turn_r:

| Candidate | Read |
|---|---|
| stock against itself | identical on every clip (the test holds this) |
| `claude-zidane-whole-fable-v1` | WORSE in walk and run. Walk: 12.6x pinch at the right shin (stock 6.5x pull at the hips), 4 hard flips against stock's 2. Run: 20x pinch at the shins, 239 hard flips against stock's 24, on shins and upper arms. The render shows the trouser leg folding into the boot. Idle and turns: same as stock, apart from 2 sliver flips. |
| `claude-zidane-hair-ref-v1/models/Zidane-hair-clean-v2.glb` | same as stock on every clip. Body is stock's; the hair cards move rigidly (worst edge 1.00x). |

Tests: `cargo test -p zidane-inspector rkdeform` (pinch, flip, mirror, unweighted vertices, verdict, and rigkit against wb-core's own skinner on the worst stock triangle) and `--test deform_rk` (stock against itself is zero everywhere, clip filter, refused input, `--render-dir` never reuses a folder).


## Using it as a gate rule

`gaw gate` can read `deform.json`: a `ratio` rule with `"check": "deform"` runs `gaw deform-check CAND --ref REF --out-dir OUT/deform` once and reads its `summary[]` (kinds `stretch`, `squash`, `volume`, `intersect`, `weights`; fields `defects`, `worst_score`, `load_percent`, `reference{..}`) the way it reads `defects.json`. Add `"defects": false` to the spec to skip the defects runs. The ready spec is `skills/gaw/references/deform-gate.json` (each kind at most 1.5x stock, `weights` none beyond stock): `gaw gate --spec skills/gaw/references/deform-gate.json --out-dir NEW CAND.glb` from the Modding folder. Stock passes against itself; `broken-a-forearm.glb` fails on stretch and `broken-all.glb` on stretch and weights. Marks rules do not apply to deform. Details: `references/defects.md` (gate section).

## Never

- Never call a model rigged because `weights` exited 0. Open the five sheets.
- Never run `deform-check` without `--ref` on a model that is meant to match stock. The absolute limits miss small breaks.
- Never feed `weights` a mesh whose skeleton was rebuilt by hand. Export it with stock's armature or the refusal list is long.
- Never read a PASS as "good animation". It says no measured defect beyond stock's own; it does not see a wrong silhouette, a collar that clips the neck in a clip it does not have, or a wrong rest pose.

## Reproduce

Calibration: `bash experiments/weights-deform-v1/run_calibration.sh` (set `GAW=` to the exe) rebuilds the broken copies with `make_broken.py` and writes `calib/<case>-absolute` and `calib/<case>-vs-stock`. Transfer cases: `bash experiments/weights-deform-v1/run_transfer_cases.sh` writes `transfer/<case>/` (weighted.glb, weights.json, deform/). Both need Python with `glbtool.py` beside it (no `-I`). The remesh comes from `remesh_blender.py` (Blender) and `assemble_remesh.py`; the bare meshes from `strip_to_mesh_only.py IN OUT [scale [dx dy dz]]`.

Corvin cases: `bash experiments/weights-deform-v1/corvin/run.sh DIR NAME [weights args]` (waits for `workbench/logs/HEAVY.lock`) writes DIR/NAME.glb and DIR/dNAME/deform.json from `experiments/original-test-character-v1/build/v9/paint/Corvin.glb`; the shipped results are `Corvin-gaw-transfer.glb` and `Corvin-gaw-polish.glb` next to it.

Tests: `cargo test -p wb-core --test weights` (transfer follows the closest surface, bone-space matching stays valid off the surface, repair says so without clips, disallowed bones hand weight to their parent, written file reads back valid, five kinds of skeleton mismatch refused, regions follow the skeleton, a bare mesh adopts the armature and gets the reference's weights back, wrong size and place refused with the scale, skin and node-transform refusals), `cargo test -p wb-core deform` and `bvh` (unit tests of the measures and the triangle tree).

## Not built

- Adopting with the new mesh's own armature kept (a file that already has a skin is refused).
