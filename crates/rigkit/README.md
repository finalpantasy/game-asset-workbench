# rigkit

CPU skinning, skin-weight transfer and motion checks for skinned meshes, in Rust. No engine and no
file format: callers read their own glTF (or anything else) and pass plain arrays.

Used by a whitetail hunting game and by the Game Asset Workbench (`gaw`).

## What it does

| Module | Does |
|---|---|
| `skin` | Linear blend skinning: rest positions plus one matrix per bone gives posed positions |
| `clip` | A baked cycle (one matrix per bone per frame), sampled the way a player lerps it |
| `weights::Source` | Copies skin weights from a rigged mesh onto a new mesh (four nearest vertices, hash grid) |
| `Source::confine`, `fade_past`, `smooth` | Repairs a rigger's weights before transfer: clears stray bones, stops a bone at its joint, evens out hard seams between bones |
| `check::stretch` | Plays poses over a mesh and reports the triangles whose edges or area change most, and the ones that turn inside out |
| `check::contact` | Plays a walk with the body moving and reports foot slide, skid at touchdown, landing speed, and kicks at the hip or shoulder |

## Conventions

- Points are `[f32; 3]`. Bone matrices are `[f32; 16]`, column-major (glTF order), rest to posed.
- Triangles are a flat index list, three per triangle.
- glam and rayon are used inside only, so callers can use any glam version or none.

## Checks give numbers, not pictures

Every check returns figures a test can hold to a limit, so a bad deform fails a build instead of
passing a glance at a thumbnail. Stretch ratios are always at least 1; `shrunk` says whether the
worst edge got shorter (pinched or folded skin) or longer (pulled skin).

## Test

```
cargo test --release
```
