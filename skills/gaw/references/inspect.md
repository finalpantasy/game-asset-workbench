# Inspect, compare, swap

Commands: `compare`, `report`, `uv`, `swap`, `roundtrip`, `render`, and the window (`gaw view`). Flags: `gaw help`.

## compare: what did the candidate change?

`gaw compare stock.glb cand.glb` prints per texture (changed pixels, bounding box, top parts, untouched parts), per mesh, skeleton and clips. Add `--json` to parse it.

- Textures match by image name. Meshes match by name. A mesh that exists in only one file shows "only in B".
- If the two textures have different sizes, the smaller is scaled up and almost every pixel shows as changed. Read that as "different sizes", not "everything repainted". Compare same-size files, or use `judge`.
- "different shape" on a mesh means the vertex count differs (a new layout splits vertices at new seams). Moved and new-UV counts are only computed when counts match, so "moved 0 uv 0" next to "different shape" is not proof nothing moved. Triangle counts and the skeleton are the comparable numbers.
- "unassigned" in the part list means texels outside any kit part, including unused sheet area.
- `--tol N` ignores per-channel changes up to N (use for lossy re-encodes).
- Compare against the stock that matches the candidate's mesh set. A candidate built from a file with extra overlay meshes (FF9 daggers) reads "only in B" against plain stock. Use the overlay stock as A.

## report, uv

- `report --coverage 4096` lists dead triangles by body region (texture area under 0.01 texel²) and unused texels. Dead triangles are normal in stock; compare counts, do not chase zero.
- `uv file.glb --part NAME` lists UV islands, texels shared by two surfaces, and what painting that part touches. Run it before painting: a part that shares texels with another part changes that other part too. Stock layouts fold limbs and the waist onto themselves, so paint there appears twice.

## swap: new paint, nothing else

`gaw swap in.glb --image NAME --png painted.png --out new.glb` replaces one texture and verifies every other byte block, the JSON and every mesh are identical. The PNG must have the same shape as the texture it replaces; any size works. It never overwrites and refuses an output outside the workspace folder. Read the "changed pixels by part" list: paint landing on parts you did not mean to touch is the usual failure, and it comes from shared texels (see `uv`).

## roundtrip, render

- `roundtrip --out copy.glb` writes the parsed file back and checks byte equality. Use it to prove the tool reads a new kind of file safely.
- `render --out view.png --part NAME --clip walk --time 0.5` saves a model-plus-texture picture without a window.

## The window

`gaw` (no arguments) opens Compare. `gaw view --page compare|inspect|swap|mark ...` opens a page with files and part preselected. `--screenshot out.png` saves the window and exits. Windows are for people and for screenshots on a hidden desktop. When a command can answer, use the command.
