---
name: gaw
description: Use when working on a 3D game asset (character GLB, UVs, textures, hair, straps) with the Game Asset Workbench program `gaw`, or when the user says "Mark & fix request", "Look at workbench request", "compare against stock", "judge", "sewing pattern UVs", "ship textures", "blink boxes", or names any gaw command.
---

# gaw: Game Asset Workbench

`gaw` is a Rust command-line tool and window for remastering game models and textures. It reads GLBs, never overwrites an input, and checks every output. Final Fantasy IX (Memoria) is the first supported game: see `references/ff9.md`.

**`gaw help` is the source of truth for flags.** This skill says which command to use and in what order. It does not list flags. Before running a command, read its block: `gaw help` and find `gaw <command>`. Running a command with no required arguments prints its usage error.

Run it as `gaw` if it is on PATH, otherwise `target/release/gaw.exe` inside the workbench build (`cargo build --release`). If `local/notes.md` exists it names this machine's copy.

## Which command

| You need to | Command | Details |
|---|---|---|
| See what a candidate changed against stock | `compare A.glb B.glb` | `references/inspect.md` |
| Count triangles, dead triangles, unused texels | `report` | `references/inspect.md` |
| See UV islands, shared texels, what paint touches | `uv --part NAME` | `references/inspect.md` |
| Put new paint into a file, nothing else changes | `swap` | `references/inspect.md` |
| Read what the user marked in the window | `requests show latest --json` | `references/mark-and-fix.md` |
| Check an edit left locked places alone | `requests locks latest cand.glb` | `references/mark-and-fix.md` |
| Draw a candidate from saved close-up views | `views render` | `references/mark-and-fix.md` |
| Turn "it looks messy" into places to fix (jagged edges, discoloration, seam stripes, direction breaks, dark gutters, smear, flat plastic, specks, stipple, hard layer cuts, hard flush card edges, fine wood-grain stripes) | `defects cand.glb --part NAME --ref stock.glb` | `references/defects.md` |
| Pass or fail a candidate against stock on the defect numbers and the user's marked places (hair spec in `references/`; `deform-gate.json` is the skinned-model spec, reading `deform-check` with `"check": "deform"`; `new-model-gate-head.json` and `-torso.json` judge a new painted model by its own paint with `"baseline": "self"` and `"cut_regions": "skip"`) | `gate --spec GATE.json cand.glb` | `references/defects.md` (gate section) |
| Look at the same close-up of every marked place for stock and each candidate, side by side (orbit or `gaw hair` turntable cameras, optional defect boxes) | `closelook --marks M.json --reference stock.glb --candidates A.glb B.glb` | `references/defects.md` (close-look section) |
| Lay out UVs so nothing shares a texel | `pattern` (built in), `relayout` (Blender) | `references/layout.md` |
| Paint on rendered pictures, bake to the atlas | `bake` | `references/layout.md` |
| Score a candidate against stock by eye-like difference | `judge` | `references/judge.md` |
| Retexture or rebuild hair cards | `hair`, `judge --strands` | `references/hair-strap.md` |
| Fix a belt strap or thin lines in paint | `strap-*`, `line-repaint`, `boundary-check`, `paint-fix`, `shells` | `references/hair-strap.md` |
| Write textures the game loads fast | `ship` | `references/ship.md` |
| Make an edited mesh load in the game as a custom model (FBX for Memoria) | `fbx cand.glb --out NEW --readback` | `references/fbx.md` |
| Give a new mesh its skin weights from stock: a bare mesh with no armature (`--adopt-skeleton` copies stock's 27 bones in), or one already on the stock skeleton (new Zidane, remeshed copy, tailless model) | `weights new.glb --from stock.glb --out OUT.glb [--adopt-skeleton] [--radial 0.25 --soften 0 --repair 40]` (a body that is not stock's: also `--polish` on weights from Blender) | `references/weights.md` |
| Check a skinned model through every clip before export: candy-wrapper twists, volume loss at joints, arm through torso, bad weights | `deform-check cand.glb --ref stock.glb --out-dir NEW` | `references/weights.md` (deform-check section) |

## Core loops

**Edit loop.** `report` and `uv` on the input, then `requests show latest --json` for the user's marks. Edit. `compare stock cand` (did only the intended parts change?). `judge --golden-dir ... cand` (does it still look like the original?). `views render cand` (look at the saved close-ups yourself). `requests locks latest cand` (did locked places stay put?). Stop at the first red check and fix that before going on.

**Layout loop.** `pattern` (or `relayout`) writes a new folder and runs its own checks. Read `checks.json`. Then `judge` the new GLB against goldens. Then paint on the sheets or `bake` pictures onto them, then `judge` again.

**Ship loop.** `ship` the final sheets, read `ship-report.json`, test in a private game copy.

## Never

- Never write to a game folder, an input file, or the user's only copy. Outputs go to a new folder; `--out-dir` must be new or empty.
- Never call a result good from a pass line alone. Open the review sheets or renders and look. A green check that cannot catch the problem is not evidence.
- Never run a game, window or Blender job in the foreground without a time limit. Run long jobs in the background and tell the user what is running.
- Never move or repack a texture box the game reads at fixed texels (FF9 blink boxes) without `--keep-stock` or `--keep-blink`.
- Never commit or publish model files, game textures or screenshots of game assets unless the user asked.
- Never kill a process you did not start. A running `gaw.exe` may belong to another chat.

## When gaw surprises you

If a command fails, lies, needs a flag this skill does not mention, or you lose time to something not written here, append one line to `skills/gaw/local/feedback.md` (create the file if it is missing): `date | command | what happened | the fix`. Then continue the task. `MAINTAINING.md` explains how those lines become skill edits or code fixes.

## If present

- `local/notes.md`: this machine's paths, test runtime and game-test rules.
- `local/feedback.md`: past surprises, newest last. Skim it before a long run.
