# Mark & fix requests

The user marks what is wrong in the window (Mark & fix page) and presses Save request. That writes a folder under the workbench's `requests/` and puts "Look at workbench request <folder>" on the clipboard. Your job is to read it and act.

## Read the request

```
gaw requests                      # list, newest first
gaw requests show latest --json   # request.json itself
gaw requests show latest          # readable: note, marked triangles by part, edge loops with positions
```

`latest` means newest saved. If the user pasted a folder name or path, pass that instead; it may not be the newest.

A request holds: the user's note (this is the task), the source file, marked triangles by mesh and kit part, edge loops with 3D positions, a mask PNG per texture, texture crops, a screenshot with the area outlined, and locked places. Open the screenshot and the masks. The numbers say where; the picture says what the user meant.

Kinds: `mark` (point at a problem) and `fix my edit` (the request carries the original and the user's rough edit; clean the edit up).

`gaw requests show` lists extra files under "attached":

- `strokes.json` (`gaw_strokes`): comb strokes, the way the user wants hair or fur to flow, root to tip. Each stroke has `mesh` and `points` (`tri` is an index into that mesh's triangles, `bary` the barycentric weights of its three corners) plus `positions` (rest-pose places, which the hair tool ignores). Points run root to tip. Optional per-stroke `weight` (default 1.0) and `radius` (model units) tune how strongly and how far a stroke overrides the paint. Feed them to the hair tool as the flow guide; the hair session owns that side (`references/hair-strap.md`).
- `defects.json` (`workbench_defects`): the defects the user picked from a `gaw defects` run, in that command's shape (kind, score, note, `marked.meshes`, `texel_boxes`, camera). The request's marked triangles and mask already cover them; the note lists them by id.

## Edit, then prove it

1. Edit into a new file. Never edit the source in place.
2. `gaw compare source.glb cand.glb`: the changed parts must match the marked parts.
3. `gaw requests locks latest cand.glb`: prints "Locks held", or lists moved places and exits 1. Locks compare by mesh and vertex number, so a candidate with different meshes or vertex counts cannot be checked place by place; the command says so. Then compare renders instead.
4. `gaw views` lists saved close-up views. `gaw views render cand.glb --out-dir DIR` draws the candidate from each one (posed, at the saved time). Render the stock file the same way and look at both pictures.
5. `gaw judge` against goldens (`references/judge.md`), with `--only` set to the marked parts.

## Defects as marks

`gaw defects` (`references/defects.md`) writes `defects.json` whose entries carry the same `marked` (triangles by mesh) and `camera` shapes as `request.json`, so a place it finds can be loaded as a mark: select `marked.meshes[].triangles`, fly to `camera`, show `texel_box` on its sheet. Each entry also names its `kind`, `score` and plain-language `note`. A defect with `view: null` has no camera; use its texel box.

## Hair flow and defects on the page

- Comb tool (Hair group): one drag per stroke, Ctrl+click removes one. **Flow arrows** shows strand direction on the model and the texture, breaks over 60 degrees in red. **Load flow...** reads the `flow.json` that `gaw hair` writes (`{"gaw_flow":1, "mesh", "comb_strokes", "triangles":[{"tri", "dir":[x,y,z], "ambiguous"}], "breaks":[{"a", "b", "angle_deg"}]}`, source-mesh triangles). **Quick estimate** is a rough solve from the marked hair's paint with default settings; `gaw hair` adds root drape, part relock and the comb, so judge flow from its file. Comb points against the flow turn orange.
- Defects card: **Load defects...** reads `gaw defects` output (also the hand-made `gaw_defects` draft shape with `triangles:[{mesh, tri}]` and `texels:[{image, box}]`). A row click flies the camera to the defect's `camera`, marks its triangles and boxes its texels. **Make request** turns ticked rows into one request. A file made from another GLB than the one open gets a warning, since its triangle numbers may not line up.

## Scripted marking (tests, no mouse)

`gaw view --page mark --mark-script steps.txt --screenshot out.png` runs one command per line (tool, click, stroke, select-tris, lock, save-request, save-glb, check-locks, view, overlay, probe, flow, defects, defect, defects-request, save-strokes). The command table is in the workbench README under "Mark & fix scripts". Gotchas:

- Zidane-class models are low-poly. A shape brush needs a large radius (about 100 px in close-up) or it holds no vertex; the page prints "Nothing moved".
- A scripted run prints the request prompt instead of copying it.
- Save as in the window runs `judge` and takes minutes; wait for it before the screenshot.
- Open edges and cracks are found on the welded model. A break that is a texture seam, or a gap only a posed model opens, does not show; use the Depth overlay, the Look tool and a saved posed view.
