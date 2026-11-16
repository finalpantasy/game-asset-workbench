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

## Edit, then prove it

1. Edit into a new file. Never edit the source in place.
2. `gaw compare source.glb cand.glb`: the changed parts must match the marked parts.
3. `gaw requests locks latest cand.glb`: prints "Locks held", or lists moved places and exits 1. Locks compare by mesh and vertex number, so a candidate with different meshes or vertex counts cannot be checked place by place; the command says so. Then compare renders instead.
4. `gaw views` lists saved close-up views. `gaw views render cand.glb --out-dir DIR` draws the candidate from each one (posed, at the saved time). Render the stock file the same way and look at both pictures.
5. `gaw judge` against goldens (`references/judge.md`), with `--only` set to the marked parts.

## Scripted marking (tests, no mouse)

`gaw view --page mark --mark-script steps.txt --screenshot out.png` runs one command per line (tool, click, stroke, select-tris, lock, save-request, save-glb, check-locks, view, overlay, probe). The command table is in the workbench README under "Mark & fix scripts". Gotchas:

- Zidane-class models are low-poly. A shape brush needs a large radius (about 100 px in close-up) or it holds no vertex; the page prints "Nothing moved".
- A scripted run prints the request prompt instead of copying it.
- Save as in the window runs `judge` and takes minutes; wait for it before the screenshot.
- Open edges and cracks are found on the welded model. A break that is a texture seam, or a gap only a posed model opens, does not show; use the Depth overlay, the Look tool and a saved posed view.
