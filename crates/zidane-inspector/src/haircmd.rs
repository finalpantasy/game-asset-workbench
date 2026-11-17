//! `gaw hair`: hair texturing from parameters (see `wb_core::hair`).
//!
//! Game-agnostic: point it at any GLB, sheet and hair selection. FF9-only details are options
//! (`--keep-blink` for field Zidane's eye-blink boxes, the UV kit's part map for `--part`).
use crate::Args;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use wb_core::analysis::Parts;
use wb_core::hair_model::{self, Backfaces, ModelOpts};
use wb_core::hairfiber as hair_fiber;
use wb_core::hair::{self, BakeIn, Chart, Flow, FlowOpts, HairSet, JoinIn, JoinKind, Palette, Root, Select, Sheet, Style, SynthCtx, V3};
use wb_core::render::{self, Camera, Filter};
use wb_core::hairlook::{LookOpts, LookStats, View};
use wb_core::{font, image_textures, Model, Texture};

pub const HAIR_USAGE: &str = "  gaw hair file.glb --out-dir NEW_FOLDER [options]
      Retextures hair cards from parameters: flow field, strand-aligned UVs, strand synthesizer,
      checks (flow across joins, Lab across joins, fringe at every mip, duplicate faces) and
      turntable sheets. Options: --sheet NAME (98_0), --mesh NAME, --part NAME [--parts DIR]
      (hair triangles from a part map), --size 4096, --pad 8, --root x,y,z:x,y,z (where hair grows
      from; default a front-to-back line over the top), --paint-weight 0.3 (how much to follow the stock
      paint strokes), --part-angle 70, --smooth 100 (flow smoothing passes), --seed N, --no-root-split, --root-drape (lay the root line onto the hair top), --keep-box x0,y0,x1,y1@SCALE (repeatable),
      --keep-blink (FF9 field Zidane), --compare NAME=FILE.glb (repeatable, turntable rows),
      --name OUT.glb.
      --rebuild-model: also rebuilds the hair geometry (generic card hair): clean strips along the
      flow, vertex rows at folds, tips trimmed to the stock alpha outline, optional gap closing, then
      flow UVs and texture on the new model. The stock mesh version goes to OUT/stock-mesh as a
      fallback. Options: --model-tris 3000, --row-stretch 1.5, --fold-deg 20, --alpha-cut 0.5
      (0 = keep the alpha cutout), --gap 0.5 (grid spacings; 0 = off), --model-smooth N (flow smoothing passes on the new mesh; 0, or 30 with --model-param solve), --model-param stock|solve (carry the stock strand solve over, or re-solve), --backfaces stock|drop|all.
      Look: --look stock|warm (warm = hue -15, chroma x1.25, L +12, contrast 0.6, clumps x1.5),
      --hue-shift DEG, --chroma-scale X, --lightness L, --contrast X, --clump W, --clump-jitter X,
      --wave A, --wave-len L (override the look; warm also sets clump jitter 0.8, wave x1.5 and
      wave length x1.6).
      Form (frequency split): the low frequency (lock layout, crevices, shading) comes from a
      guide in the stock UV layout, blurred along the surface and regraded by the look; the
      synthesizer adds only fine strands. --guide PNG|none (default the stock sheet; a painted or
      `gaw bake` sheet works too), --guide-radius 1.5 (stock texels, surface blur), --guide-blur
      0.75 (guide px), --guide-contrast 1.0, --detail 1.0 (strand strength), --detail-clumps 0.4,
      --tip-depth X. Baked light: --ao 0.45 (surface AO strength from the hair, head and body;
      --ao-spacing 0.02, --ao-rays 32, --ao-reach 0.35, fractions of the hair size),
      --highlight 10 (Kajiya-Kay band, Lab L; 0 = off), --light x,y,z (0,0.6,-0.8), --view
      normal|x,y,z (normal = each side sees its own band), --highlight-power 40,
      --highlight-shift 0.25.
      Hair part: --include I,J,... (mesh triangle indices), --grow-part (adds edge neighbors in
      hair color: --grow-hue 12 deg, --grow-chroma 0.6, --grow-light 0.65; prints the triangles
      added; they stay their own lock and keep stock geometry under --rebuild-model).
      Tips (--rebuild-model): --tip-refine 2 (subdivision levels near the alpha cut), --tip-blur
      1.0 (stock px; smooths the cut line), --layer-sep 0.05 (grid spacings; cards of different
      locks lying on each other are kept this far apart so they stop fighting for depth; 0 = off).
      Widow's peak (grown triangles): --root-line 14, --grown-shade 20 (bangs shadow, Lab L),
      --crevice 26 (dark center line, Lab L); all 0 by default with surface strands, where the
      grown part takes its own stock paint through the guide, averaged with the cards it meets.
      Close-up strands: --strand-gain 14 (Lab L; 6 with surface strands), --strand-width 26 and
      --strand-fine-width 9 (texels), --strand-fine 0.5 (share of the fine layer; 0.15 with
      surface strands); 0 gain = off. --fiber-space surface|chart: surface (default) traces
      strands over the 3D surface so they run on across chart seams and flow regions, and sets
      --detail and --detail-clumps to 0; --strand-smooth 16 (rings of flow diffusion first, so a
      strand bends over several strip rows instead of kinking at one); chart draws them in each
      chart's own coordinates (v4).
      Cutout cleanup: --alpha-min-area 160 (texels; smaller pieces go, smaller holes fill),
      --alpha-min-width 4 (texels; thinner slivers go); 0 area = off.
      Edges and seams: --sdf-spread 2 (alpha ramp half-width in texels around the cutout line;
      0.6 = hard edge), --seam-tol 2 (Lab; gutter texels on chart seams are nudged where the two
      sides differ by more; 0 = measure only). Logs gutter, seam and alpha coverage per mip.
      Flow over the whole hair: --couple 0.5 (pull toward nearby layered cards of other locks;
      0 = per lock), --couple-radius (model units, default hair size / 30), --comb strokes.json
      (guide strokes, schema in the gaw skill), --comb-radius (default hair size / 20),
      --tip-flip (turn locks whose flow runs away from their faded tips), --paint-strong 0.6 and
      --paint-strong-deg 40 (paint weight where the stock paint runs within that angle of the
      away-from-root direction on a card that faces it; 0 = --paint-weight everywhere). Writes
      OUT/flow.json.
      Close look (8 close-up views, ours and stock): --look-small 30 (px; specks and holes),
      --sharp-target 1.25 (x stock sharpness across strands), --island-small 64 (texels),
      --jump-tolerance 0.10 (layer jumps per hair pixel over stock). Writes close-look-layers.png.";

fn opt<'a>(args: &'a Args, k: &str) -> Option<&'a str> {
    args.0.iter().position(|a| a == k).and_then(|i| args.0.get(i + 1)).map(String::as_str)
}
fn opts<'a>(args: &'a Args, k: &str) -> Vec<&'a str> {
    args.0.windows(2).filter(|w| w[0] == k).map(|w| w[1].as_str()).collect()
}
/// Reads `--comb strokes.json` (schema in the gaw skill, hair-strap.md): strokes on `mesh` become
/// rest-pose polylines; strokes on other meshes are counted and skipped.
fn read_comb(path: &str, model: &Model, mesh: usize, radius: f64) -> Result<(Vec<hair::CombStroke>, usize), String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("--comb {path}: {e}"))?;
    let v: Value = serde_json::from_str(&txt).map_err(|e| format!("--comb {path}: {e}"))?;
    if v.get("gaw_strokes").and_then(|x| x.as_u64()) != Some(1) {
        return Err(format!("--comb {path}: not a gaw_strokes 1 file"));
    }
    let (mut out, mut skipped) = (Vec::new(), 0);
    for (si, s) in v.get("strokes").and_then(|x| x.as_array()).ok_or(format!("--comb {path}: no strokes"))?.iter().enumerate() {
        let m = s.get("mesh").and_then(|x| x.as_u64()).ok_or(format!("--comb: stroke {si} has no mesh"))? as usize;
        if m != mesh {
            skipped += 1;
            continue;
        }
        let me = &model.meshes[m];
        let mut pts = Vec::new();
        for (pi, p) in s.get("points").and_then(|x| x.as_array()).ok_or(format!("--comb: stroke {si} has no points"))?.iter().enumerate() {
            let t = p.get("tri").and_then(|x| x.as_u64()).ok_or(format!("--comb: stroke {si} point {pi} has no tri"))? as usize;
            let b: Vec<f64> = p.get("bary").and_then(|x| x.as_array()).map(|a| a.iter().filter_map(|x| x.as_f64()).collect()).unwrap_or_default();
            if t >= me.tris.len() || b.len() != 3 {
                return Err(format!("--comb: stroke {si} point {pi}: tri {t} out of range ({} triangles) or bary not 3 numbers", me.tris.len()));
            }
            let c = me.tris[t].map(|k| me.pos[k as usize]);
            pts.push(std::array::from_fn(|a| (0..3).map(|k| b[k] * c[k][a] as f64).sum::<f64>()));
        }
        if pts.len() >= 2 {
            out.push(hair::CombStroke { points: pts, weight: s.get("weight").and_then(|x| x.as_f64()).unwrap_or(1.0), radius: s.get("radius").and_then(|x| x.as_f64()).unwrap_or(radius) });
        }
    }
    Ok((out, skipped))
}

fn num<T: std::str::FromStr>(args: &Args, k: &str, d: T) -> Result<T, String> {
    match opt(args, k) {
        Some(s) => s.parse().map_err(|_| format!("{k}: cannot read {s}")),
        None => Ok(d),
    }
}
fn v3(s: &str) -> Result<V3, String> {
    let p: Vec<f64> = s.split(',').map(|x| x.trim().parse::<f64>().map_err(|_| format!("not a number: {x}"))).collect::<Result<_, _>>()?;
    if p.len() != 3 {
        return Err(format!("{s}: need x,y,z"));
    }
    Ok([p[0], p[1], p[2]])
}

fn hue(i: usize) -> [u8; 3] {
    let h = (i as f32 * 0.618_034).fract() * 6.0;
    let x = 1.0 - ((h % 2.0) - 1.0).abs();
    let (r, g, b) = match h as i32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    [(40.0 + 215.0 * r) as u8, (40.0 + 215.0 * g) as u8, (40.0 + 215.0 * b) as u8]
}

/// Orthographic camera on the hair: yaw/pitch in degrees from the front view; `half` = half the
/// view height in model units.
pub fn hair_cam(hs: &HairSet, yaw: f32, pitch: f32, half: f32) -> Camera {
    let (lo, hi) = hs.bounds();
    let target = std::array::from_fn(|k| (0.5 * (lo[k] + hi[k])) as f32);
    let ext = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max) as f32;
    Camera { yaw: std::f32::consts::PI + yaw.to_radians(), pitch: pitch.to_radians(), dist: ext * 8.0, target, fov_y: 30f32.to_radians(), ortho: true, ortho_half: half }
}

fn save_png(path: &Path, w: usize, h: usize, rgba: &[u8]) -> Result<(), String> {
    image::save_buffer(path, rgba, w as u32, h as u32, image::ExtendedColorType::Rgba8).map_err(|e| format!("{}: {e}", path.display()))
}

/// Pastes `src` (sw x sh) into `dst` (dw wide) at (ox, oy), enlarged `k` times (nearest).
fn paste(dst: &mut [u8], dw: usize, ox: usize, oy: usize, src: &[u8], sw: usize, sh: usize, k: usize) {
    let dh = dst.len() / 4 / dw;
    for y in 0..sh * k {
        for x in 0..sw * k {
            let (dx, dy) = (ox + x, oy + y);
            if dx >= dw || dy >= dh {
                continue;
            }
            let s = ((y / k) * sw + x / k) * 4;
            let d = (dy * dw + dx) * 4;
            dst[d..d + 4].copy_from_slice(&src[s..s + 4]);
        }
    }
}

const VIEWS: [(&str, f32, f32); 8] =
    [("front", 0.0, 0.0), ("front-left", 45.0, 0.0), ("left", 90.0, 0.0), ("back-left", 135.0, 0.0), ("back", 180.0, 0.0), ("right", -90.0, 0.0), ("top", 0.0, 75.0), ("below", 0.0, -45.0)];

/// A turntable sheet: one row per model, one column per view. `px` = render size, `k` = enlargement.
fn turntable(rows: &[(String, &Model, &[Option<Texture>])], hs: &HairSet, half: f32, px: usize, k: usize, title: &str) -> (usize, usize, Vec<u8>) {
    let tile = px * k;
    let (lw, th) = (150, 30);
    let w = lw + VIEWS.len() * (tile + 4);
    let h = th + rows.len() * (tile + 4);
    let mut out = vec![0u8; w * h * 4];
    for p in out.chunks_mut(4) {
        p.copy_from_slice(&[22, 22, 26, 255]);
    }
    font::draw(&mut out, w, h, 8, 8, title, 2, [230, 230, 230]);
    for (vi, (name, _, _)) in VIEWS.iter().enumerate() {
        font::draw(&mut out, w, h, (lw + vi * (tile + 4) + 4) as i64, 8, name, 2, [200, 200, 200]);
    }
    for (ri, (label, model, tex)) in rows.iter().enumerate() {
        let oy = th + ri * (tile + 4);
        font::draw(&mut out, w, h, 6, (oy + tile / 2) as i64, label, 2, [230, 230, 230]);
        for (vi, (_, yaw, pitch)) in VIEWS.iter().enumerate() {
            let o = render::Options { cam: hair_cam(hs, *yaw, *pitch, half), w: px, h: px, background: ([58, 64, 76], [32, 34, 40]), filter: Filter::Smooth, positions: None };
            let f = render::render(model, tex, &o, &|_, _| None);
            paste(&mut out, w, lw + vi * (tile + 4), oy, &f.rgba, px, px, k);
        }
    }
    (w, h, out)
}

/// Field Zidane's eye-blink boxes on 98_0, in 512-pixel units: the closed-eye source and the eye target.
const BLINK: [[f64; 4]; 2] = [[224.0, 216.0, 272.0, 248.0], [32.0, 116.0, 80.0, 148.0]];

/// Mean Lab of a triangle's paint (texels with alpha >= 128), sampled at 15 points.
fn tri_lab(model: &Model, mesh: usize, t: usize, tex: &Texture) -> Option<[f64; 3]> {
    let m = &model.meshes[mesh];
    let uv = m.tris[t].map(|v| m.uv[v as usize]);
    let (mut acc, mut n) = ([0.0f64; 3], 0.0f64);
    for a in 0..5 {
        for b in 0..(5 - a) {
            let w = [(a as f64 + 0.3) / 5.0, (b as f64 + 0.3) / 5.0];
            let w2 = 1.0 - w[0] - w[1];
            if w2 < 0.0 {
                continue;
            }
            let u = w[0] * uv[0][0] as f64 + w[1] * uv[1][0] as f64 + w2 * uv[2][0] as f64;
            let v = w[0] * uv[0][1] as f64 + w[1] * uv[1][1] as f64 + w2 * uv[2][1] as f64;
            let x = ((u * tex.w as f64) as i64).clamp(0, tex.w as i64 - 1) as usize;
            let y = ((v * tex.h as f64) as i64).clamp(0, tex.h as i64 - 1) as usize;
            let p = &tex.rgba[(y * tex.w + x) * 4..(y * tex.w + x) * 4 + 4];
            if p[3] < 128 {
                continue;
            }
            for c in 0..3 {
                acc[c] += p[c] as f64;
            }
            n += 1.0;
        }
    }
    (n > 0.0).then(|| hair::lab([acc[0] / n, acc[1] / n, acc[2] / n]))
}

/// Grows a triangle selection over edge-connected neighbors painted like it: hue within `hue_tol`
/// degrees of the selection's median, chroma at least `chroma_frac` and lightness at least
/// `light_frac` of its medians (darker hairline shadow and skin stay out). Returns the added
/// triangles (mesh indices).
fn grow_part(model: &Model, mesh: usize, sel: &mut [bool], tex: &Texture, hue_tol: f64, chroma_frac: f64, light_frac: f64) -> Vec<usize> {
    let m = &model.meshes[mesh];
    let lch = |c: [f64; 3]| (c[0], c[2].atan2(c[1]).to_degrees(), (c[1] * c[1] + c[2] * c[2]).sqrt());
    let cols: Vec<Option<(f64, f64, f64)>> = (0..m.tris.len()).map(|t| tri_lab(model, mesh, t, tex).map(lch)).collect();
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    };
    let selc: Vec<(f64, f64, f64)> = (0..m.tris.len()).filter(|&t| sel[t]).filter_map(|t| cols[t]).collect();
    if selc.is_empty() {
        return Vec::new();
    }
    let (l0, h0, c0) = (med(selc.iter().map(|c| c.0).collect()), med(selc.iter().map(|c| c.1).collect()), med(selc.iter().map(|c| c.2).collect()));
    let q = |v: u32| {
        let p = m.pos[v as usize];
        ((p[0] * 1e4).round() as i64, (p[1] * 1e4).round() as i64, (p[2] * 1e4).round() as i64)
    };
    let mut edges: std::collections::HashMap<((i64, i64, i64), (i64, i64, i64)), Vec<usize>> = std::collections::HashMap::new();
    let ek = |a: u32, b: u32| {
        let (x, y) = (q(a), q(b));
        if x < y { (x, y) } else { (y, x) }
    };
    for (t, tri) in m.tris.iter().enumerate() {
        for k in 0..3 {
            edges.entry(ek(tri[k], tri[(k + 1) % 3])).or_default().push(t);
        }
    }
    let mut added = Vec::new();
    loop {
        let mut more = Vec::new();
        for (t, tri) in m.tris.iter().enumerate() {
            if sel[t] {
                continue;
            }
            let Some((l, h, c)) = cols[t] else { continue };
            let dh = ((h - h0 + 540.0) % 360.0 - 180.0).abs();
            if dh > hue_tol || c < chroma_frac * c0 || l < light_frac * l0 {
                continue;
            }
            if (0..3).any(|k| edges[&ek(tri[k], tri[(k + 1) % 3])].iter().any(|&o| o != t && sel[o])) {
                more.push(t);
            }
        }
        if more.is_empty() {
            break;
        }
        for t in more {
            sel[t] = true;
            added.push(t);
        }
    }
    added.sort();
    added
}

pub fn hair_cmd(args: &Args) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let input = args.positionals().into_iter().next().ok_or("hair needs an input GLB")?;
    let out = PathBuf::from(opt(args, "--out-dir").ok_or("hair needs --out-dir NEW_FOLDER")?);
    if out.exists() && std::fs::read_dir(&out).map_err(|e| e.to_string())?.next().is_some() {
        return Err(format!("{} is not empty; pick a new folder", out.display()));
    }
    let model = Model::open(&input)?;
    let sheet_name = opt(args, "--sheet").unwrap_or("98_0");
    let image = model.images.iter().position(|i| i.name.trim_end_matches(".png") == sheet_name).ok_or(format!("no image named {sheet_name}"))?;
    let mesh = match opt(args, "--mesh") {
        Some(n) => model.meshes.iter().position(|m| m.name == n).ok_or(format!("no mesh named {n}"))?,
        None => (0..model.meshes.len()).filter(|&i| model.meshes[i].image == Some(image)).max_by_key(|&i| model.meshes[i].tris.len()).ok_or(format!("no mesh uses {sheet_name}"))?,
    };
    let mut sel = match opt(args, "--part") {
        Some(part) => {
            let dir = PathBuf::from(opt(args, "--parts").unwrap_or(crate::KIT_PARTS));
            let parts = Parts::load(&dir)?;
            let id = parts.id_by_name(part).ok_or(format!("no part named {part}"))?;
            Select::Mask(parts.triangle_parts(&model, mesh).iter().map(|&p| p == id).collect())
        }
        None => Select::All,
    };
    // Grow the hair part: --include lists mesh triangles; --grow-part adds neighbors painted like
    // hair (hue, chroma and lightness near the hair's own) by edge connectivity.
    let mut grown = vec![false; model.meshes[mesh].tris.len()];
    {
        let mut sel_mask: Vec<bool> = match &sel {
            Select::All => vec![true; grown.len()],
            Select::Mask(m) => m.clone(),
        };
        for s in opts(args, "--include") {
            for t in s.split(',').filter(|x| !x.trim().is_empty()) {
                let t: usize = t.trim().parse().map_err(|_| format!("--include {t}: not a triangle index"))?;
                if t >= grown.len() {
                    return Err(format!("--include {t}: mesh has {} triangles", grown.len()));
                }
                if !sel_mask[t] {
                    sel_mask[t] = true;
                    grown[t] = true;
                }
            }
        }
        if args.flag("--grow-part") {
            let tex = Texture::decode(&model.images[image].bytes)?;
            let added = grow_part(&model, mesh, &mut sel_mask, &tex, num(args, "--grow-hue", 12.0)?, num(args, "--grow-chroma", 0.6)?, num(args, "--grow-light", 0.65)?);
            for &t in &added {
                grown[t] = true;
            }
        }
        let list: Vec<usize> = (0..grown.len()).filter(|&t| grown[t]).collect();
        if !list.is_empty() {
            let m = &model.meshes[mesh];
            let cen: Vec<String> = list.iter().map(|&t| {
                let c = (0..3).fold([0.0f32; 3], |a, k| { let p = m.pos[m.tris[t][k] as usize]; [a[0] + p[0] / 3.0, a[1] + p[1] / 3.0, a[2] + p[2] / 3.0] });
                format!("{t} ({:.2},{:.2},{:.2})", c[0], c[1], c[2])
            }).collect();
            println!("hair part grown by {} triangles (kept as stock geometry, textured as hair): {}", list.len(), cen.join(", "));
        }
        if !list.is_empty() {
            sel = Select::Mask(sel_mask);
        }
    }
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_owned();
    let name = opt(args, "--name").map(str::to_owned).unwrap_or_else(|| format!("{stem}-hair.glb"));
    if !args.flag("--rebuild-model") {
        let r = run(args, &RunIn { input: &input, model: &model, orig: &model, mesh, sel, out: &out, name, root: None, flow: None, param: None, opaque: false, extra: Vec::new(), label: "hair tool", model_report: None, renders: true, grown: grown.clone(), stock_hs: None, t0 })?;
        return if r.passed { Ok(()) } else { Err("hair checks failed (see checks.json)".into()) };
    }
    // Fallback first: the stock mesh with flow UVs and the strand texture.
    let fb_name = name.replace(".glb", "-stock-mesh.glb");
    let fb = run(args, &RunIn { input: &input, model: &model, orig: &model, mesh, sel, out: &out.join("stock-mesh"), name: fb_name, root: None, flow: None, param: None, opaque: false, extra: Vec::new(), label: "stock mesh", model_report: None, renders: false, grown: grown.clone(), stock_hs: None, t0 })?;
    let _ = std::fs::copy(out.join("stock-mesh").join("flow.json"), out.join("flow.json"));
    // Better model, built on the stock analysis.
    let stock_tex = Texture::decode(&model.images[image].bytes)?;
    let stock = Sheet { w: stock_tex.w, h: stock_tex.h, rgba: &stock_tex.rgba };
    let backfaces = match opt(args, "--backfaces").unwrap_or("stock") {
        "drop" => Backfaces::Drop,
        "stock" => Backfaces::Stock,
        "all" => Backfaces::All,
        b => return Err(format!("--backfaces {b}: drop, stock or all")),
    };
    let mo = ModelOpts {
        tris: num(args, "--model-tris", 3000)?,
        row_stretch: num(args, "--row-stretch", 1.5)?,
        fold_deg: num(args, "--fold-deg", 20.0)?,
        alpha_cut: num(args, "--alpha-cut", 0.5)?,
        gap: num(args, "--gap", 0.5)?,
        backfaces,
        keep: fb.hs.tris.iter().map(|&t| grown[t]).collect(),
        tip_refine: num(args, "--tip-refine", 2)?,
        tip_blur: num(args, "--tip-blur", 1.0)?,
        layer_sep: num(args, "--layer-sep", 0.05)?,
    };
    let hm = hair_model::remesh(&model, &fb.hs, &fb.fl, &fb.pr, &stock, &mo)?;
    let r = &hm.report;
    println!(
        "  [{:5.1} s] model: {} front + {} back triangles, {} vertices (stock {} front + {} back); spacing {:.4}; {} regions remeshed whole, {} per triangle ({} folded triangles in strand space); {} fold edges kept; {} constraints skipped, {} points merged; area stock {:.3}, remeshed {:.3}, after alpha trim {:.3}; gaps closed at {} vertices (max move {:.4}), boundary distance to other locks in spacings [0,.25,.5,1,2,4+): {:?}; stacked cards separated at {} vertices (max move {:.4})",
        t0.elapsed().as_secs_f64(), r.front, r.back, r.vertices, r.stock_front, r.stock_back, r.spacing, r.regions_remeshed, r.regions_per_triangle, r.flipped_tris, r.fold_edges, r.constraints_skipped, r.merged_points, r.area_stock, r.area_before_trim, r.area_after_trim, r.gap_vertices, r.gap_max_move, r.gap_hist, r.sep_vertices, r.sep_max_move
    );
    let (bytes, mask, hm_of) = hair_model::write(&model, &fb.hs, &hm)?;
    std::fs::create_dir_all(out.join("model")).map_err(|e| e.to_string())?;
    let untextured = out.join("model").join(name.replace(".glb", "-model-stock-texture.glb"));
    std::fs::write(&untextured, &bytes).map_err(|e| e.to_string())?;
    let m2 = Model::from_bytes(&bytes)?;
    let fb_model = Model::open(&fb.glb)?;
    let fl_old = &fb.fl;
    // Smoothing would pull the flow off the carried-over stock coordinates (22 breaks at 30 passes
    // on stock Zidane, 0 at 0), so it is on only for --model-param solve by default.
    let resolve_default = opt(args, "--model-param") == Some("solve");
    let flow_smooth: usize = num(args, "--model-smooth", if resolve_default { 30 } else { 0 })?;
    let flow_fn = |hs2: &HairSet| hair_model::transfer_flow(hs2, &hm, &hm_of, fl_old, flow_smooth);
    // Strand coordinates: carry the stock solve over (default) or re-solve on the new mesh.
    let resolve = match opt(args, "--model-param").unwrap_or("stock") {
        "stock" => false,
        "solve" => true,
        o => return Err(format!("--model-param {o}: expected stock or solve")),
    };
    let param_fn = |hs2: &HairSet, fl2: &Flow| hair_model::transfer_param(hs2, &hm, &hm_of, fl2);
    let model_report = json!({
        "front_triangles": r.front, "back_triangles": r.back, "vertices": r.vertices,
        "stock_front_triangles": r.stock_front, "stock_back_triangles": r.stock_back,
        "spacing": r.spacing, "regions_remeshed": r.regions_remeshed, "regions_per_triangle": r.regions_per_triangle,
        "folded_triangles_in_strand_space": r.flipped_tris, "fold_edges": r.fold_edges, "constraints_skipped": r.constraints_skipped,
        "merged_points": r.merged_points, "area_stock": r.area_stock, "area_remeshed": r.area_before_trim, "area_after_alpha_trim": r.area_after_trim,
        "gap_vertices": r.gap_vertices, "gap_max_move": r.gap_max_move, "sep_vertices": r.sep_vertices, "sep_max_move": r.sep_max_move, "boundary_gap_hist_spacings": r.gap_hist,
        "kept_grown_triangles": mo.keep.iter().filter(|&&k| k).count(), "tip_refine": mo.tip_refine, "tip_blur": mo.tip_blur,
        "options": { "model_tris": mo.tris, "row_stretch": mo.row_stretch, "fold_deg": mo.fold_deg, "alpha_cut": mo.alpha_cut, "gap": mo.gap, "backfaces": format!("{:?}", mo.backfaces) },
        "stock_texture_model": untextured.display().to_string(),
    });
    let res = run(
        args,
        &RunIn {
            input: &input,
            model: &m2,
            orig: &model,
            mesh,
            sel: Select::Mask(mask),
            out: &out,
            name,
            root: Some(Root::line(fb.root.a, fb.root.b)),
            flow: Some(&flow_fn),
            param: if resolve { None } else { Some(&param_fn) },
            opaque: mo.alpha_cut > 0.0,
            extra: vec![("tool, stock mesh".into(), fb_model)],
            label: "tool, rebuilt",
            model_report: Some(model_report),
            renders: true,
            grown: (0..hm_of.len()).map(|m| hm_of[m] != usize::MAX && grown[fb.hs.tris[hm.tris[hm_of[m]].parent]]).collect(),
            stock_hs: Some(&fb.hs),
            t0,
        },
    )?;
    if res.passed && fb.passed {
        Ok(())
    } else {
        Err("hair checks failed (see checks.json)".into())
    }
}

pub struct RunIn<'a> {
    pub input: &'a Path,
    pub model: &'a Model,
    /// The untouched input, for the "stock" turntable row.
    pub orig: &'a Model,
    pub mesh: usize,
    pub sel: Select,
    pub out: &'a Path,
    pub name: String,
    pub root: Option<Root>,
    pub flow: Option<&'a dyn Fn(&HairSet) -> Flow>,
    /// Strand-coordinate override (the rebuilt model carries the stock solve over).
    pub param: Option<&'a dyn Fn(&HairSet, &Flow) -> hair::Param>,
    /// Geometry carries the outline: no cutout from the stock alpha.
    pub opaque: bool,
    /// Turntables, render-space fringe and debug views (off for the fallback run).
    pub renders: bool,
    pub extra: Vec<(String, Model)>,
    pub label: &'a str,
    pub model_report: Option<Value>,
    /// Per mesh triangle of `model`: grown into the hair part (stock geometry, guide paint not used).
    pub grown: Vec<bool>,
    /// The stock hair set on `orig`, for the close-look baseline (None: `model` has stock geometry).
    pub stock_hs: Option<&'a HairSet>,
    pub t0: std::time::Instant,
}

pub struct RunOut {
    pub hs: HairSet,
    pub fl: Flow,
    pub pr: hair::Param,
    pub root: Root,
    pub glb: PathBuf,
    pub passed: bool,
}

fn run(args: &Args, ri: &RunIn) -> Result<RunOut, String> {
    let t0 = ri.t0;
    let input = ri.input;
    let out = ri.out;
    let model = ri.model;
    let mesh = ri.mesh;
    let sheet_name = opt(args, "--sheet").unwrap_or("98_0");
    let image = model.images.iter().position(|i| i.name.trim_end_matches(".png") == sheet_name).ok_or(format!("no image named {sheet_name}"))?;
    let size: usize = num(args, "--size", 4096)?;
    let pad: usize = num(args, "--pad", 8)?;
    let mut style = Style::default();
    style.seed = num(args, "--seed", style.seed)?;
    {
        let warm = opt(args, "--look") == Some("warm");
        let contrast: f64 = num(args, "--contrast", if warm { 0.6 } else { 1.0 })?;
        style.clump_contrast *= contrast;
        style.strand_contrast *= contrast;
        style.clump_edge_dark *= contrast;
        style.root_dark *= contrast;
        style.bundle *= contrast;
        style.clump = num(args, "--clump", if warm { style.clump * 1.5 } else { style.clump })?;
        style.clump_jitter = num(args, "--clump-jitter", if warm { 0.8 } else { style.clump_jitter })?;
        style.wave_amp = num(args, "--wave", if warm { style.wave_amp * 1.5 } else { style.wave_amp })?;
        style.wave_len = num(args, "--wave-len", if warm { style.wave_len * 1.6 } else { style.wave_len })?;
        // With a guide the stock alpha already shapes the tips; deep notches read as stair steps.
        let guided = opt(args, "--guide") != Some("none");
        style.tip_depth = num(args, "--tip-depth", if guided { 0.015 } else { style.tip_depth })?;
        style.ao = num(args, "--ao", style.ao)?;
    }
    let mut hs = HairSet::build(model, mesh, &ri.sel)?;
    {
        let apart: Vec<bool> = hs.tris.iter().map(|&t| ri.grown.get(t).copied().unwrap_or(false)).collect();
        if apart.iter().any(|&g| g) {
            hs.relock(&apart);
        }
    }
    let root = match (&ri.root, opt(args, "--root")) {
        (Some(r), _) => Root::line(r.a, r.b),
        (None, Some(s)) => {
            let (a, b) = s.split_once(':').unwrap_or((s, s));
            Root::line(v3(a)?, v3(b)?)
        }
        (None, None) => Root::auto(&hs),
    };
    let root = if args.flag("--root-drape") && ri.root.is_none() { root.draped(&hs) } else { root };
    println!("== {} ==", ri.label);
    let mut keep: Vec<[f64; 4]> = Vec::new();
    for k in opts(args, "--keep-box") {
        let (b, sc) = k.split_once('@').ok_or("--keep-box needs x0,y0,x1,y1@SCALE (pixels at that sheet size)")?;
        let sc: f64 = sc.parse().map_err(|_| format!("--keep-box scale {sc}"))?;
        let p: Vec<f64> = b.split(',').map(|x| x.parse::<f64>().map_err(|_| format!("--keep-box {b}"))).collect::<Result<_, _>>()?;
        if p.len() != 4 {
            return Err(format!("--keep-box {b}: need x0,y0,x1,y1"));
        }
        keep.push([p[0] / sc, p[1] / sc, p[2] / sc, p[3] / sc]);
    }
    if args.flag("--keep-blink") {
        keep.extend(BLINK.iter().map(|b| [b[0] / 512.0, b[1] / 512.0, b[2] / 512.0, b[3] / 512.0]));
    }
    let say = |s: String| println!("  [{:5.1} s] {s}", t0.elapsed().as_secs_f64());
    say(format!(
        "{}: mesh {} ({}), {} hair triangles ({} degenerate dropped), duplicate faces {} same-winding + {} back faces, {} locks, {} non-manifold edges",
        input.display(),
        mesh,
        model.meshes[mesh].name,
        hs.tris.len(),
        hs.degenerate,
        hs.dup_same,
        hs.dup_flipped,
        hs.nlocks,
        hs.nonmanifold_edges
    ));
    std::fs::create_dir_all(out.join("textures")).map_err(|e| e.to_string())?;

    // Stock sheet.
    let stock_tex = Texture::decode(&model.images[image].bytes)?;
    let stock = Sheet { w: stock_tex.w, h: stock_tex.h, rgba: &stock_tex.rgba };
    // Duplicate faces whose stock UVs differ from their primary (separately painted back faces).
    let own_uv_dups = (0..hs.tris.len())
        .filter(|&i| !hs.is_primary(i))
        .filter(|&i| {
            let p = hs.primary[i];
            let (a, b) = (hair::tri_uv(&model, &hs, i), hair::tri_uv(&model, &hs, p));
            let pw = hs.wt[p];
            (0..3).any(|c| {
                let k = pw.iter().position(|&w| w == hs.wt[i][c]).unwrap_or(c);
                (a[c][0] - b[k][0]).abs() + (a[c][1] - b[k][1]).abs() > 1e-4
            })
        })
        .count();

    // 1-2: flow.
    let fo = FlowOpts { root, paint_weight: num(args, "--paint-weight", 0.3)?, smooth_iters: num(args, "--smooth", 100)?, part_angle: num(args, "--part-angle", 70.0)? };
    let (blo, bhi) = hs.bounds();
    let span = (0..3).map(|k| bhi[k] - blo[k]).fold(0.0, f64::max);
    let comb = match opt(args, "--comb") {
        Some(p) if ri.flow.is_none() => read_comb(p, model, mesh, num(args, "--comb-radius", span / 20.0)?)?,
        _ => (Vec::new(), 0),
    };
    let gf = hair::GlobalFlow { couple: num(args, "--couple", 0.5)?, couple_radius: num(args, "--couple-radius", span / 30.0)?, comb: comb.0, tip_flip: args.flag("--tip-flip"), paint_strong: num(args, "--paint-strong", 0.6)?, paint_strong_deg: num(args, "--paint-strong-deg", 40.0)? };
    let (fl, split, agree) = match ri.flow {
        Some(f) => (f(&hs), 0, None),
        None => {
            let (mut fl, ag): (Flow, hair::FlowAgree) = hair::flow_global(model, &hs, &stock, &fo, &gf);
            let split = if args.flag("--no-root-split") { 0 } else { hair::split_at_root_plane(&hs, &mut fl, &root) };
            (fl, split, Some(ag))
        }
    };
    if let Some(a) = &agree {
        say(format!(
            "flow over the whole hair: {} layered pairs from different locks; angle between them median {:.1} p90 {:.1} ({} over 45) per lock, median {:.1} p90 {:.1} ({} over 45) solved together (--couple {}); {} comb strokes ({} skipped: other mesh), {} triangles combed; {} locks run away from their faded tips{}",
            a.pairs, a.median_before, a.p90_before, a.over45_before, a.median_after, a.p90_after, a.over45_after, gf.couple, gf.comb.len(), comb.1, a.combed, a.tip_against.len(),
            if a.tip_flipped > 0 { format!(" ({} flipped)", a.tip_flipped) } else if a.tip_against.is_empty() { String::new() } else { format!(" ({:?}; --tip-flip turns them)", a.tip_against) }
        ));
    }
    let ambiguous_locks: Vec<(usize, usize, usize)> = {
        let mut per = vec![(0usize, 0usize); hs.nlocks];
        for i in 0..hs.tris.len() {
            if hs.is_primary(i) {
                per[hs.lock[i]].1 += 1;
                per[hs.lock[i]].0 += fl.ambiguous[i] as usize;
            }
        }
        per.iter().enumerate().filter(|(_, p)| p.0 * 4 > p.1).map(|(l, p)| (l, p.0, p.1)).collect()
    };
    say(format!("flow: {split} regions split at the root line's plane (a center part)"));
    say(format!("flow: {} regions, {} parting edges, {} ambiguous triangles, {} locks mostly ambiguous", fl.region_lock.len(), fl.partings, fl.ambiguous.iter().zip(0..).filter(|(a, i)| **a && hs.is_primary(*i)).count(), ambiguous_locks.len()));

    // 3: strand coordinates.
    let pr = match ri.param {
        Some(f) => f(&hs, &fl),
        None => hair::param(&hs, &fl, &root),
    };
    let prim: Vec<usize> = (0..hs.tris.len()).filter(|&i| hs.is_primary(i)).collect();
    let grad_len = |i: usize, c: usize| {
        let (g, _) = hair::basis_grads(hs.corners(i));
        hair::len((0..3).fold([0.0; 3], |a, k| hair::add(a, hair::mul(g[k], pr.uv[i][k][c]))))
    };
    say(format!("gradient length median: u {:.3}, v {:.3} (1 = model units)", { let mut v: Vec<f64> = prim.iter().map(|&i| grad_len(i, 0)).collect(); v.sort_by(f64::total_cmp); v[v.len() / 2] }, { let mut v: Vec<f64> = prim.iter().map(|&i| grad_len(i, 1)).collect(); v.sort_by(f64::total_cmp); v[v.len() / 2] }));
    if std::env::var("HAIR_DEBUG").is_ok() {
        for &i in &prim {
            if pr.u_err[i] > 45.0 {
                let c = hs.centroid(i);
                eprintln!("bad tri {} region {} c [{:.2} {:.2} {:.2}] flow [{:.2} {:.2} {:.2}] away [{:.2} {:.2} {:.2}] n [{:.2} {:.2} {:.2}] uerr {:.0} coh {:.2} dens {:.3?}", hs.tris[i], fl.region[i], c[0], c[1], c[2], fl.dir[i][0], fl.dir[i][1], fl.dir[i][2], fl.away[i][0], fl.away[i][1], fl.away[i][2], hs.normal(i)[0], hs.normal(i)[1], hs.normal(i)[2], pr.u_err[i], fl.coherence[i], pr.dens[i]);
            }
        }
    }
    let v_bad = prim.iter().filter(|&&i| pr.v_err[i] > 45.0).count();
    let u_bad = prim.iter().filter(|&&i| pr.u_err[i] > 45.0).count();
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    };
    say(format!(
        "strand coordinates: v off the flow median {:.1} deg ({} triangles over 45), u off median {:.1} deg ({} over 45)",
        med(prim.iter().map(|&i| pr.v_err[i]).collect()),
        v_bad,
        med(prim.iter().map(|&i| pr.u_err[i]).collect()),
        u_bad
    ));

    // 4: charts and packing.
    let (lo, hi) = hs.bounds();
    let hair_size = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max);
    let (mut charts, chart_of): (Vec<Chart>, Vec<usize>) = hair::charts(&hs, &fl, &pr, hair_size / 600.0);
    let occ = hair::occupied(&model, image, &hs, size, &keep);
    let scale = hair::fit_scale(&mut charts, &occ, size, pad)?;
    let hair_area: f64 = prim.iter().map(|&i| hs.area(i)).sum();
    say(format!("{} charts packed at {:.1} texels per unit ({:.0} hair texels on the {size} sheet)", charts.len(), scale, hair_area * scale * scale));

    // Palette from the stock: hair texels by v / v_max.
    let vmax = prim.iter().flat_map(|&i| pr.uv[i].iter().map(|q| q[1])).fold(0.0, f64::max).max(1e-9);
    let mut samples = Vec::new();
    for &i in &prim {
        let su = hair::tri_uv(&model, &hs, i);
        for a in 0..6 {
            for b in 0..(6 - a) {
                let bc = [(a as f64 + 0.33) / 6.0, (b as f64 + 0.33) / 6.0, 1.0 - (a as f64 + 0.33) / 6.0 - (b as f64 + 0.33) / 6.0];
                if bc[2] < 0.0 {
                    continue;
                }
                let (u, v) = ((0..3).map(|k| bc[k] * su[k][0]).sum::<f64>(), (0..3).map(|k| bc[k] * su[k][1]).sum::<f64>());
                let c = stock.px((u * stock.w as f64) as i64, (v * stock.h as f64) as i64);
                if c[3] < 128.0 {
                    continue;
                }
                let sv = (0..3).map(|k| bc[k] * pr.uv[i][k][1]).sum::<f64>();
                samples.push((sv / vmax, hair::lab([c[0], c[1], c[2]])));
            }
        }
    }
    let mut pal = Palette::from_samples(&samples, 8)?;
    // Look: stock colors by default; --look warm is a lighter, warmer, softer grade (honey blond).
    let warm = match opt(args, "--look").unwrap_or("stock") {
        "stock" => false,
        "warm" => true,
        o => return Err(format!("--look {o}: expected stock or warm")),
    };
    let (hs_d, cs_d, ls_d) = if warm { (-15.0, 1.25, 12.0) } else { (0.0, 1.0, 0.0) };
    let (hue_shift, chroma_scale, l_shift) = (num(args, "--hue-shift", hs_d)?, num(args, "--chroma-scale", cs_d)?, num(args, "--lightness", ls_d)?);
    if hue_shift != 0.0 || chroma_scale != 1.0 || l_shift != 0.0 {
        pal.grade(hue_shift, chroma_scale, l_shift);
        say(format!("palette graded: hue {hue_shift:+} deg, chroma x{chroma_scale}, L {l_shift:+}"));
    }
    say(format!("palette from {} stock texels: root L {:.0}, mid L {:.0}, tip L {:.0}, hue {:.0} deg, chroma {:.0}, L IQR {:.1}", samples.len(), pal.at(0.05)[0], pal.at(0.5)[0], pal.at(0.95)[0], pal.hue, pal.chroma, pal.l_iqr));

    // Ambient occlusion against the whole model (rest pose).
    let mut occluders: Vec<[V3; 3]> = Vec::new();
    for m in &model.meshes {
        for t in &m.tris {
            occluders.push(t.map(|v| [m.pos[v as usize][0] as f64, m.pos[v as usize][1] as f64, m.pos[v as usize][2] as f64]));
        }
    }
    let ao = hair::ambient_occlusion(&hs, &occluders, 48, hair_size * 0.35);
    // Form layers: guide low-pass (stock sheet unless --guide), AO per surface node, highlight.
    let guide_tex = match opt(args, "--guide") {
        None => None,
        Some("none") => None,
        Some(p) => Some(Texture::decode(&std::fs::read(p).map_err(|e| format!("--guide {p}: {e}"))?)?),
    };
    let surface_fibers = match opt(args, "--fiber-space").unwrap_or("surface") {
        "surface" => true,
        "chart" => false,
        s => return Err(format!("--fiber-space {s}: use surface or chart")),
    };
    // Surface strands are quieter by default: the guide carries the lock-scale light and dark.
    let strand_gain: f64 = num(args, "--strand-gain", if surface_fibers { 6.0 } else { 14.0 })?;
    let surf = surface_fibers && strand_gain > 0.0;
    let grown_hs: Vec<bool> = hs.tris.iter().map(|&t| ri.grown.get(t).copied().unwrap_or(false)).collect();
    let use_tri: Vec<bool> = grown_hs.iter().map(|g| !g).collect();
    // With surface strands the grown hairline keeps its own stock paint in the guide, averaged
    // together with the cards it meets, so no color step runs along its outline.
    let guide_use: Vec<bool> = if surf { vec![true; grown_hs.len()] } else { use_tri.clone() };
    let guide_group: Vec<usize> = if surf { hair::guide_groups(&hs) } else { hs.lock.clone() };
    // Size of one stock texel on the surface (model units).
    let stock_texels: f64 = prim.iter().filter(|&&i| use_tri[i]).map(|&i| {
        let q = hair::tri_uv(&model, &hs, i);
        0.5 * ((q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1])).abs() * (stock.w * stock.h) as f64
    }).sum();
    let tex_unit = (prim.iter().filter(|&&i| use_tri[i]).map(|&i| hs.area(i)).sum::<f64>() / stock_texels.max(1.0)).sqrt();
    let color_layer = if opt(args, "--guide") == Some("none") {
        None
    } else {
        let gsheet = match &guide_tex {
            Some(g) => Sheet { w: g.w, h: g.h, rgba: &g.rgba },
            None => Sheet { w: stock.w, h: stock.h, rgba: &stock_tex.rgba },
        };
        let radius = num(args, "--guide-radius", 1.5)? * tex_unit;
        let sigma = num(args, "--guide-blur", 0.75)? * gsheet.w as f64 / stock.w as f64;
        let grid = hair::TriGrid::new(&hs, radius * 0.5, |i| hs.is_primary(i));
        let low = hair::guide_lowpass_grouped(&model, &hs, &gsheet, &guide_use, &guide_group, sigma, radius, &grid);
        let mut ls: Vec<f64> = low.iter().flatten().map(|c| c[0]).collect();
        ls.sort_by(f64::total_cmp);
        let l_mid = ls.get(ls.len() / 2).copied().unwrap_or(50.0);
        let contrast = num(args, "--guide-contrast", 1.0)?;
        let mut filled = 0usize;
        let mut vals = vec![[0.0; 3]; grid.len];
        for i in 0..hs.tris.len() {
            for (ni, bc) in grid.nodes(i) {
                vals[ni] = match low[ni] {
                    Some(c) => {
                        let c = hair::grade_lab(c, hue_shift, chroma_scale, l_shift);
                        let lm = l_mid + l_shift;
                        [lm + (c[0] - lm) * contrast, c[1], c[2]]
                    }
                    None => {
                        filled += 1;
                        let v = (0..3).map(|k| bc[k] * pr.uv[i][k][1]).sum::<f64>();
                        pal.at((v / vmax).clamp(0.0, 1.0))
                    }
                };
            }
        }
        say(format!("guide low-pass: {} ({}x{}), radius {:.4} (stock texel {:.4}), blur {:.2} px, contrast {contrast}; {} nodes, {} from the palette (no guide paint near)", opt(args, "--guide").unwrap_or("stock sheet"), gsheet.w, gsheet.h, radius, tex_unit, sigma, grid.len, filled));
        Some((grid, vals))
    };
    let ao_layer = {
        let grid = hair::TriGrid::new(&hs, hair_size * num(args, "--ao-spacing", 0.02)?, |i| hs.is_primary(i));
        let vals = hair::surface_ao(&hs, &grid, &occluders, num(args, "--ao-rays", 32)?, hair_size * num(args, "--ao-reach", 0.35)?);
        let mut v = vals.clone();
        v.sort_by(f64::total_cmp);
        say(format!("surface AO: {} nodes, open fraction p10 {:.2} median {:.2}, strength {}", grid.len, v[v.len() / 10], v[v.len() / 2], style.ao));
        Some((grid, vals))
    };
    let hl_gain: f64 = num(args, "--highlight", 10.0)?;
    let light = (hl_gain > 0.0).then(|| -> Result<hair::Light, String> {
        Ok(hair::Light {
            to_light: v3(opt(args, "--light").unwrap_or("0,0.6,-0.8"))?,
            to_view: match opt(args, "--view").unwrap_or("normal") {
                "normal" => [0.0; 3],
                s => v3(s)?,
            },
            power: num(args, "--highlight-power", 40.0)?,
            gain: hl_gain,
            shift: num(args, "--highlight-shift", 0.25)?,
        })
    }).transpose()?;
    let fibers3 = (strand_gain > 0.0 && surface_fibers).then(|| -> Result<hair_fiber::Fibers3, String> {
        let (w, fw, fine) = (num(args, "--strand-width", 26.0)?, num(args, "--strand-fine-width", 9.0)?, num(args, "--strand-fine", 0.15)?);
        let smooth = num(args, "--strand-smooth", 16)?;
        let f = hair_fiber::Fibers3::build(&hs, &fl, strand_gain, w, fw, fine, scale, style.seed ^ 0xF1BE, smooth);
        say(format!("surface strands: {} lines ({} samples) {:.4} apart, fine {} lines {:.4} apart; bare share {:.3}", f.main.lines, f.main.samples, f.main.sep, f.fine_field.as_ref().map_or(0, |x| x.lines), f.fine_field.as_ref().map_or(0.0, |x| x.sep), f.bare_share(&hs)));
        Ok(f)
    }).transpose()?;
    let fibers = (strand_gain > 0.0 && !surface_fibers).then(|| {
        hair::Fibers::new(strand_gain, num(args, "--strand-width", 26.0).unwrap_or(26.0), num(args, "--strand-fine-width", 9.0).unwrap_or(9.0), num(args, "--strand-fine", 0.5).unwrap_or(0.5), style.seed ^ 0xF1BE)
    });
    if let Some(f) = &fibers {
        say(format!("strand layer: gain {} L, width {} texels, fine {} x at {} texels", f.gain, f.width, f.fine, f.fine_width));
    }
    // Surface strands replace the chart detail strands too (those restart at every chart seam).
    let detail_default = if fibers3.is_some() { 0.0 } else { 1.0 };
    let look = hair::Look { color: color_layer, detail: hair::Look::detail_style(&style, num(args, "--detail", detail_default)?, num(args, "--detail-clumps", if fibers3.is_some() { 0.0 } else { 0.4 })?), ao: ao_layer, light, fibers, fibers3 };
    say("ambient occlusion baked".into());

    // 5: the sheet. Stock pixels kept (nearest, so non-hair texels look exactly as before).
    let mut rgba = vec![0u8; size * size * 4];
    for y in 0..size {
        for x in 0..size {
            let s = ((y * stock.h / size) * stock.w + x * stock.w / size) * 4;
            rgba[(y * size + x) * 4..(y * size + x) * 4 + 4].copy_from_slice(&stock.rgba[s..s + 4]);
        }
    }
    // Lightly blurred stock alpha for the cutout field.
    let mut sa = vec![0.0; stock.w * stock.h];
    for y in 0..stock.h as i64 {
        for x in 0..stock.w as i64 {
            let mut acc = 0.0;
            for (dx, dy, w) in [(0, 0, 4.0), (1, 0, 2.0), (-1, 0, 2.0), (0, 1, 2.0), (0, -1, 2.0), (1, 1, 1.0), (-1, 1, 1.0), (1, -1, 1.0), (-1, -1, 1.0)] {
                acc += w * stock.px(x + dx, y + dy)[3] / 255.0;
            }
            sa[y as usize * stock.w + x as usize] = if ri.opaque { 1.0 } else { acc / 16.0 };
        }
    }
    if !ri.opaque {
        let tb: f64 = num(args, "--tip-blur", 1.0)?;
        if tb > 0.6 {
            let mut f: Vec<f32> = (0..stock.w * stock.h).map(|k| stock.rgba[k * 4 + 3] as f32 / 255.0).collect();
            hair::blur(&mut f, stock.w, stock.h, 1, tb);
            sa = f.into_iter().map(|x| x as f64).collect();
        }
    }
    let ctx = SynthCtx { style: &style, pal: &pal, size: hair_size, vmax };
    let bin = BakeIn { model: &model, hs: &hs, fl: &fl, pr: &pr, charts: &charts, scale, size, pad, stock: Sheet { w: stock.w, h: stock.h, rgba: &stock_tex.rgba }, stock_alpha: sa, ao, ctx, look: Some(&look), sdf_spread: num(args, "--sdf-spread", 2.0)?, grown: grown_hs.clone(), root_line: num(args, "--root-line", if surf { 0.0 } else { 14.0 })?, grown_shade: num(args, "--grown-shade", if surf { 0.0 } else { 20.0 })?, crevice: num(args, "--crevice", if surf { 0.0 } else { 26.0 })?, alpha_min_area: num(args, "--alpha-min-area", 160)?, alpha_min_width: num(args, "--alpha-min-width", 4.0)? };
    let map = hair::bake(&bin, &mut rgba);
    say(format!("gutter texels: {}, {} taken from the surface across welded edges", map.gutter, map.gutter_surface));
    let suv: Vec<[[f64; 2]; 3]> = (0..hs.tris.len())
        .map(|i| std::array::from_fn(|c| {
            let q = hair::sheet_uv(&charts[chart_of[i]], scale, size, pr.uv[i][c]);
            [q[0] as f64, q[1] as f64]
        }))
        .collect();
    // Seam stitch: a safety net. Surface-space baking should leave it little to do.
    let pre_stitch = rgba.clone();
    let mut seam = hair::seam_stitch(&hs, &fl, &suv, &chart_of, &map.inside, &mut rgba, size, num(args, "--seam-tol", 2.0)?);
    // A stitched texel must stay in the hair palette (the bake's limits); otherwise keep the bake.
    let mut reverted = 0usize;
    for i in 0..size * size {
        if rgba[i * 4..i * 4 + 3] != pre_stitch[i * 4..i * 4 + 3] && rgba[i * 4 + 3] >= 128 {
            let c = [rgba[i * 4] as f64, rgba[i * 4 + 1] as f64, rgba[i * 4 + 2] as f64];
            if pal.is_fringe_by(c, 28.0, 0.36, 14.0) {
                rgba[i * 4..i * 4 + 3].copy_from_slice(&pre_stitch[i * 4..i * 4 + 3]);
                reverted += 1;
            }
        }
    }
    if reverted > 0 {
        let again = hair::seam_stitch(&hs, &fl, &suv, &chart_of, &map.inside, &mut rgba, size, 0.0);
        seam.mean_after = again.mean_before;
        seam.max_after = again.max_before;
        seam.over_after = again.over_before;
        seam.texels_changed -= reverted.min(seam.texels_changed);
        say(format!("seam: {reverted} stitched texels left the palette and were put back"));
    }
    say(format!(
        "seam: {} points on chart seams; Lab at the seam mean {:.2} max {:.2} ({} over 2) before the stitch, mean {:.2} max {:.2} ({} over) after; stitch changed {} gutter texels by {:.1} on average",
        seam.points, seam.mean_before, seam.max_before, seam.over_before, seam.mean_after, seam.max_after, seam.over_after, seam.texels_changed, seam.mean_change
    ));
    // Free texels (not used by anything else, not hair) get hair-ward colors at alpha 0.
    let valid: Vec<bool> = (0..size * size).map(|i| map.chart[i] != usize::MAX || (occ[i] && rgba[i * 4 + 3] >= 128)).collect();
    let fill: Vec<bool> = (0..size * size).map(|i| !occ[i] && map.chart[i] == usize::MAX).collect();
    hair::push_pull(&mut rgba, size, &valid, &fill);
    say("sheet painted".into());
    let png = {
        let mut b = Vec::new();
        image::codecs::png::PngEncoder::new(&mut b)
            .write_image(&rgba, size as u32, size as u32, image::ExtendedColorType::Rgba8)
            .map_err(|e| e.to_string())?;
        b
    };
    std::fs::write(out.join("textures").join(format!("{sheet_name}.png")), &png).map_err(|e| e.to_string())?;

    // New model: hair UVs, then the sheet.
    let nm = hair::new_meshes(&model, &hs, &pr, &charts, &chart_of, scale, size);
    let rebuilt = wb_core::rebuild::rebuild(&model, &nm)?;
    let mid = Model::from_bytes(&rebuilt)?;
    let swapped = wb_core::swap::swap(&mid, sheet_name, &png)?;
    let glb_path = out.join(&ri.name);
    std::fs::write(&glb_path, &swapped.bytes).map_err(|e| e.to_string())?;
    let new_model = Model::from_bytes(&swapped.bytes)?;
    say(format!("wrote {}", glb_path.display()));

    // 6: checks.
    let new_tex = Texture::new(size, size, rgba.clone());
    let mip_level = 3usize;
    let lv = |t: &Texture, l: usize| -> (usize, usize, Vec<u8>) {
        let (w, h, p) = t.level(l);
        (w, h, p.to_vec())
    };
    let (m0w, m0h, m0) = lv(&new_tex, 0);
    let (mw, mh, mp) = lv(&new_tex, mip_level);
    let ours = hair::joins(&JoinIn { hs: &hs, fl: &fl, suv: suv.clone(), chart_of: chart_of.clone(), level0: (m0w, m0h, &m0), mip: (mw, mh, &mp) });
    // The same edges on the stock model, for comparison (stock UV islands as charts).
    let stock_suv: Vec<[[f64; 2]; 3]> = (0..hs.tris.len()).map(|i| hair::tri_uv(&model, &hs, i)).collect();
    let stock_island = stock_islands(&model, &hs);
    let smip = stock_tex.level(mip_level.min(stock_tex.levels() - 1));
    let stock_joins = hair::joins(&JoinIn { hs: &hs, fl: &fl, suv: stock_suv, chart_of: stock_island, level0: (stock_tex.w, stock_tex.h, &stock_tex.rgba), mip: (smip.0, smip.1, smip.2) });
    let breaks = ours.iter().filter(|j| j.kind == JoinKind::Break).count();
    if ri.flow.is_none() {
        // The solved flow, for overlays and the comb tool (schema in the gaw skill, hair-strap.md).
        let fj = json!({
            "gaw_flow": 1, "source": ri.input.display().to_string(), "mesh": mesh, "comb_strokes": gf.comb.len(),
            "triangles": (0..hs.tris.len()).map(|i| json!({
                "tri": hs.tris[i], "dir": fl.dir[i].map(|x| (x * 1e4).round() / 1e4), "region": fl.region[i],
                "coherence": (fl.coherence[i] * 1000.0).round() / 1000.0, "ambiguous": fl.ambiguous[i]
            })).collect::<Vec<_>>(),
            "breaks": ours.iter().filter(|j| j.kind == JoinKind::Break).map(|j| json!({"a": hs.tris[j.a], "b": hs.tris[j.b], "angle_deg": (j.turn * 10.0).round() / 10.0})).collect::<Vec<_>>(),
        });
        std::fs::write(out.join("flow.json"), serde_json::to_string(&fj).unwrap()).map_err(|e| e.to_string())?;
    }
    let count = |js: &[hair::Join], k: JoinKind| js.iter().filter(|j| j.kind == k).count();
    let de = |js: &[hair::Join]| {
        let v: Vec<f64> = js.iter().filter(|j| j.samples > 0).map(|j| j.de0).collect();
        (med(v.clone()), v.iter().cloned().fold(0.0, f64::max), v.len())
    };
    let (d_med, d_max, d_n) = de(&ours);
    let (s_med, s_max, s_n) = de(&stock_joins);
    say(format!(
        "joins: {} between charts: {} continuous, {} partings, {} fold tips, {} merges, {} BREAKS; Lab across joins median {:.2} max {:.2} ({} measured). Stock on the same edges: median {:.2} max {:.2} ({} measured, {} breaks by the same test)",
        ours.len(),
        count(&ours, JoinKind::Continuous),
        count(&ours, JoinKind::Parting),
        count(&ours, JoinKind::FoldTip),
        count(&ours, JoinKind::Merge),
        breaks,
        d_med,
        d_max,
        d_n,
        s_med,
        s_max,
        s_n,
        count(&stock_joins, JoinKind::Break)
    ));
    let hair_mask: Vec<bool> = map.chart.iter().map(|&c| c != usize::MAX).collect();
    let fr = hair::fringe(&new_tex, &hair_mask, &pal, 7);
    let cov_mask: Vec<bool> = map.inside.clone();
    let cov = hair::alpha_coverage(&new_tex, &cov_mask, 7);
    say(format!("alpha coverage (share at alpha 0.5 or more) by mip: {}", cov.iter().map(|c| format!("{c:.3}")).collect::<Vec<_>>().join(" ")));
    // Stock fringe on the stock hair texels, for comparison.
    let mut stock_mask = vec![false; stock_tex.w * stock_tex.h];
    for i in 0..hs.tris.len() {
        let p = hair::tri_uv(&model, &hs, i).map(|q| [q[0] * stock_tex.w as f64, q[1] * stock_tex.h as f64]);
        wb_core::uv::raster(p, stock_tex.w, stock_tex.h, 0.0, |x, y| stock_mask[y * stock_tex.w + x] = true);
    }
    let sfr = hair::fringe(&stock_tex, &stock_mask, &pal, 5);
    let fringe_total: usize = fr.iter().map(|f| f.1).sum();
    // Form: our lightness averaged per stock texel against the stock's (both about one stock texel
    // of low-pass). Correlation says the lock layout survived; the spread ratio says how much of
    // the crevice-to-center contrast did.
    let form = {
        let (w, h) = (stock_tex.w, stock_tex.h);
        let mut ls = vec![0f32; w * h * 2];
        for i in 0..w * h {
            if stock_mask[i] && stock_tex.rgba[i * 4 + 3] >= 128 {
                let p = &stock_tex.rgba[i * 4..i * 4 + 3];
                ls[i * 2] = hair::lab([p[0] as f64, p[1] as f64, p[2] as f64])[0] as f32;
                ls[i * 2 + 1] = 1.0;
            }
        }
        hair::blur(&mut ls, w, h, 2, 0.75);
        let pairs: Vec<(f64, f64)> = (0..w * h)
            .filter(|&i| map.form[i].1 >= 3 && ls[i * 2 + 1] > 0.3)
            .map(|i| (map.form[i].0 / map.form[i].1 as f64, (ls[i * 2] / ls[i * 2 + 1]) as f64))
            .collect();
        let n = pairs.len() as f64;
        let (ma, mb) = (pairs.iter().map(|p| p.0).sum::<f64>() / n.max(1.0), pairs.iter().map(|p| p.1).sum::<f64>() / n.max(1.0));
        let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
        for p in &pairs {
            sab += (p.0 - ma) * (p.1 - mb);
            saa += (p.0 - ma) * (p.0 - ma);
            sbb += (p.1 - mb) * (p.1 - mb);
        }
        let r = sab / (saa * sbb).sqrt().max(1e-9);
        let spread = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            if v.is_empty() { 0.0 } else { v[v.len() * 9 / 10] - v[v.len() / 10] }
        };
        let (so, ss) = (spread(pairs.iter().map(|p| p.0).collect()), spread(pairs.iter().map(|p| p.1).collect()));
        json!({ "lowpass_l_correlation": (r * 1000.0).round() / 1000.0, "spread_p90_p10_ours": (so * 10.0).round() / 10.0, "spread_p90_p10_stock": (ss * 10.0).round() / 10.0,
                "spread_ratio": ((so / ss.max(1e-9)) * 100.0).round() / 100.0, "mean_l_ours": (ma * 10.0).round() / 10.0, "mean_l_stock": (mb * 10.0).round() / 10.0, "stock_texels": pairs.len() })
    };
    say(format!("form: low-pass L correlation with stock {}, lock contrast (P90-P10 L) ours {} vs stock {} (ratio {}), over {} stock texels", form["lowpass_l_correlation"], form["spread_p90_p10_ours"], form["spread_p90_p10_stock"], form["spread_ratio"], form["stock_texels"]));
    say(format!("fringe texels (level, count, shown): ours {:?}; stock {:?}", fr, sfr));

    // Renders: what the eye sees.
    let new_textures = image_textures(&new_model);
    let stock_textures = image_textures(model);
    let orig_textures = image_textures(ri.orig);
    let mut compares: Vec<(String, Model)> = Vec::new();
    for c in opts(args, "--compare") {
        let (name, file) = c.split_once('=').unwrap_or(("compare", c));
        compares.push((name.to_owned(), Model::open(Path::new(file))?));
    }
    let compare_tex: Vec<Vec<Option<Texture>>> = compares.iter().map(|c| image_textures(&c.1)).collect();
    let extra_tex: Vec<Vec<Option<Texture>>> = ri.extra.iter().map(|c| image_textures(&c.1)).collect();
    let mut rows: Vec<(String, &Model, &[Option<Texture>])> = vec![("stock".into(), ri.orig, &orig_textures)];
    for (k, c) in compares.iter().enumerate() {
        rows.push((c.0.clone(), &c.1, &compare_tex[k]));
    }
    for (k, c) in ri.extra.iter().enumerate() {
        rows.push((c.0.clone(), &c.1, &extra_tex[k]));
    }
    rows.push((ri.label.into(), &new_model, &new_textures));
    let (blo, bhi) = render::bounds(model);
    let model_h = (bhi[1] - blo[1]) as f32;
    // Game distance: the whole character about 300 px tall (a 1080p field view), shown 3x.
    let ppu = 300.0 / model_h;
    let head_px = 128usize;
    if ri.renders {
    let (w, h, img) = turntable(&rows, &hs, head_px as f32 * 0.5 / ppu, head_px, 3, "game distance (character ~300 px tall), shown 3x, mipmapped");
    save_png(&out.join("turntable-game.png"), w, h, &img)?;
    let (w, h, img) = turntable(&rows, &hs, hair_size as f32 * 0.62, 480, 1, "close-up, mipmapped");
    save_png(&out.join("turntable-close.png"), w, h, &img)?;
    }
    // Render-space fringe: hair pixels in the close-up and game views of our model.
    let mut render_fringe = 0usize;
    let mut render_hair_px = 0usize;
    let offsets = render::tri_offsets(&new_model);
    let hair_tris_new: std::collections::HashSet<usize> = {
        let mut s = std::collections::HashSet::new();
        // New mesh keeps the triangle order, so the hair triangles keep their indices.
        for &t in &hs.tris {
            s.insert(offsets[mesh] + t);
        }
        s
    };
    let halves: Vec<f32> = if ri.renders { vec![hair_size as f32 * 0.62, head_px as f32 * 0.5 / ppu] } else { Vec::new() };
    for half in halves {
        let px = if half > 1.0 && half == hair_size as f32 * 0.62 { 480 } else { head_px };
        for (_, yaw, pitch) in VIEWS {
            let o = render::Options { cam: hair_cam(&hs, yaw, pitch, half), w: px, h: px, background: ([0, 0, 0], [0, 0, 0]), filter: Filter::Smooth, positions: None };
            let f = render::render(&new_model, &new_textures, &o, &|_, _| None);
            for (i, &id) in f.pick.iter().enumerate() {
                if id >= 0 && hair_tris_new.contains(&(id as usize)) {
                    render_hair_px += 1;
                    if pal.is_fringe([f.rgba[i * 4] as f64, f.rgba[i * 4 + 1] as f64, f.rgba[i * 4 + 2] as f64]) {
                        render_fringe += 1;
                    }
                }
            }
        }
    }
    say(format!("render-space fringe: {render_fringe} of {render_hair_px} hair pixels"));
    // Close look: specks, holes, sharpness across strands, flat patches and direction jumps at lock
    // edges, ours against stock on the same close-up views.
    let lo = LookOpts { small: num(args, "--look-small", 30)?, ..LookOpts::default() };
    let mut look_views: Vec<Value> = Vec::new();
    let (mut look_ours, mut look_stock) = (LookStats::default(), LookStats::default());
    if ri.renders {
        let half = hair_size as f32 * 0.62;
        let shs = ri.stock_hs.unwrap_or(&hs);
        let (lw, lh) = (480 * VIEWS.len(), 960);
        let mut sheet = vec![0u8; lw * lh * 4];
        for (vi, (vname, yaw, pitch)) in VIEWS.iter().copied().enumerate() {
            let (a, ia) = close_look(&new_model, &new_textures, &hs, &hs, yaw, pitch, half, 480, &lo);
            let (b, ib) = close_look(ri.orig, &orig_textures, shs, &hs, yaw, pitch, half, 480, &lo);
            for (row, img) in [(0usize, &ib), (1, &ia)] {
                for y in 0..480 {
                    let o = ((row * 480 + y) * lw + vi * 480) * 4;
                    sheet[o..o + 480 * 4].copy_from_slice(&img[y * 480 * 4..(y + 1) * 480 * 4]);
                }
            }
            look_views.push(json!({ "view": vname, "ours": look_json(&a), "stock": look_json(&b) }));
            look_ours.add(&a);
            look_stock.add(&b);
        }
        save_png(&out.join("close-look-layers.png"), lw, lh, &sheet)?;
    }
    let flat_frac = |s: &LookStats| s.flat_px as f64 / s.part_px.max(1) as f64;
    // Layer jumps per 1000 hair pixels (not per layer-edge pixel: removing a stipple removes
    // edge pixels with no jump, which would raise a per-edge share).
    let jump_frac = |s: &LookStats| 1000.0 * s.jump_px as f64 / s.part_px.max(1) as f64;
    let sharp_target = num(args, "--sharp-target", 1.25)?;
    let jump_tol = num(args, "--jump-tolerance", 0.10)?;
    // Texture space: cutout islands and holes per chart, ours (inside texels) and stock (stock islands).
    let isl_small = num(args, "--island-small", 64usize)?;
    let ours_isl = wb_core::hairlook::alpha_islands(&new_tex.rgba, size, size, &map.chart.iter().zip(&map.inside).map(|(&c, &i)| if i { c } else { usize::MAX }).collect::<Vec<_>>(), isl_small);
    let stock_isl = {
        let mut ch = vec![usize::MAX; stock_tex.w * stock_tex.h];
        let si = stock_islands(&model, &hs);
        for i in 0..hs.tris.len() {
            let p = hair::tri_uv(&model, &hs, i).map(|q| [q[0] * stock_tex.w as f64, q[1] * stock_tex.h as f64]);
            wb_core::uv::raster(p, stock_tex.w, stock_tex.h, 0.0, |x, y| ch[y * stock_tex.w + x] = si[i]);
        }
        // Stock texels are 8x larger in each direction at 4096 vs 512: scale the size bar.
        let k = (size / stock_tex.w.max(1)).max(1);
        wb_core::hairlook::alpha_islands(&stock_tex.rgba, stock_tex.w, stock_tex.h, &ch, (isl_small / (k * k)).max(2))
    };
    let look_fail: Vec<String> = if ri.renders {
        let mut f = Vec::new();
        if look_ours.specks > look_stock.specks {
            f.push(format!("specks {} > stock {}", look_ours.specks, look_stock.specks));
        }
        if look_ours.tiny > look_stock.tiny {
            f.push(format!("tiny bits (stipple) {} > stock {}", look_ours.tiny, look_stock.tiny));
        }
        if look_ours.holes > look_stock.holes {
            f.push(format!("holes {} > stock {}", look_ours.holes, look_stock.holes));
        }
        if look_ours.sharpness < sharp_target * look_stock.sharpness {
            f.push(format!("sharpness {:.3} < {:.2} x stock {:.3}", look_ours.sharpness, sharp_target, look_stock.sharpness));
        }
        if flat_frac(&look_ours) > flat_frac(&look_stock).max(0.002) {
            f.push(format!("flat {:.4} > stock {:.4}", flat_frac(&look_ours), flat_frac(&look_stock)));
        }
        if jump_frac(&look_ours) > (1.0 + jump_tol) * jump_frac(&look_stock) {
            f.push(format!("layer jumps {:.2} > {:.2} x stock {:.2} per 1000 hair px", jump_frac(&look_ours), 1.0 + jump_tol, jump_frac(&look_stock)));
        }
        if ours_isl.0 + ours_isl.2 > 0 {
            f.push(format!("texture islands {} / holes {}", ours_isl.0, ours_isl.2));
        }
        f
    } else {
        Vec::new()
    };
    if ri.renders {
        say(format!(
            "close look (ours / stock, close-up 8 views): specks {} / {} ({} / {} px), tiny bits {} / {}, holes {} / {} ({} / {} px), sharpness across strands {:.3} / {:.3}, flat share {:.4} / {:.4} (largest {} / {} px), layer jumps {:.2} / {:.2} per 1000 hair px ({} / {} px); texture islands {} ({} texels) / {}, holes {} ({} texels) / {}{}",
            look_ours.specks, look_stock.specks, look_ours.speck_px, look_stock.speck_px, look_ours.tiny, look_stock.tiny, look_ours.holes, look_stock.holes, look_ours.hole_px, look_stock.hole_px,
            look_ours.sharpness, look_stock.sharpness, flat_frac(&look_ours), flat_frac(&look_stock), look_ours.flat_max, look_stock.flat_max,
            jump_frac(&look_ours), jump_frac(&look_stock), look_ours.jump_px, look_stock.jump_px,
            ours_isl.0, ours_isl.1, stock_isl.0, ours_isl.2, ours_isl.3, stock_isl.2,
            if look_fail.is_empty() { "; pass".to_string() } else { format!("; FAIL: {}", look_fail.join(", ")) }
        ));
    }

    // Debug views: locks and flow regions; the sheet with charts outlined.
    let mut hair_of = vec![usize::MAX; model.meshes[mesh].tris.len()];
    for (i, &t) in hs.tris.iter().enumerate() {
        hair_of[t] = i;
    }
    if ri.renders {
    let dbg_px = 384;
    let mut dbg = vec![0u8; dbg_px * 4 * dbg_px * 3 * 4];
    for (k, (_, yaw, pitch)) in [VIEWS[0], VIEWS[2], VIEWS[4], VIEWS[6]].iter().enumerate() {
        for (row, mode) in [0usize, 1, 2].iter().enumerate() {
            let o = render::Options { cam: hair_cam(&hs, *yaw, *pitch, hair_size as f32 * 0.62), w: dbg_px, h: dbg_px, background: ([30, 30, 36], [30, 30, 36]), filter: Filter::Texels, positions: None };
            let f = render::render(&model, &stock_textures, &o, &|mi, ti| {
                if mi == mesh && hair_of[ti] != usize::MAX {
                    let i = hair_of[ti];
                    match mode {
                        0 => Some(hue(hs.lock[i])),
                        1 => Some(hue(fl.region[i] * 7 + 3)),
                        _ => {
                            let e = pr.u_err[i];
                            Some(if e > 45.0 { [230, 40, 40] } else if e > 20.0 { [230, 200, 40] } else { [60, 190, 80] })
                        }
                    }
                } else {
                    Some(render::DIM)
                }
            });
            paste(&mut dbg, dbg_px * 4, k * dbg_px, row * dbg_px, &f.rgba, dbg_px, dbg_px, 1);
        }
    }
    save_png(&out.join("locks-regions-strand-error.png"), dbg_px * 4, dbg_px * 3, &dbg)?;
    }
    let small = 1024;
    let mut prev = vec![0u8; small * small * 4];
    let (pw, ph, pp) = new_tex.level((size / small).trailing_zeros() as usize);
    for y in 0..ph.min(small) {
        for x in 0..pw.min(small) {
            let o = (y * pw + x) * 4;
            let a = pp[o + 3] as f32 / 255.0;
            for c in 0..3 {
                prev[(y * small + x) * 4 + c] = (pp[o + c] as f32 * a + 255.0 * (1.0 - a) * if (x / 16 + y / 16) % 2 == 0 { 0.8 } else { 0.6 }) as u8;
            }
            prev[(y * small + x) * 4 + 3] = 255;
        }
    }
    for (ci, c) in charts.iter().enumerate() {
        let k = size / small;
        let (x0, y0) = (c.at[0] as usize / k, c.at[1] as usize / k);
        let (x1, y1) = (((c.at[0] + (c.hi[0] - c.lo[0]) * scale) as usize / k).min(small - 1), ((c.at[1] + (c.hi[1] - c.lo[1]) * scale) as usize / k).min(small - 1));
        let col = hue(ci);
        for x in x0..=x1 {
            for y in [y0, y1] {
                prev[(y * small + x) * 4..(y * small + x) * 4 + 3].copy_from_slice(&col);
            }
        }
        for y in y0..=y1 {
            for x in [x0, x1] {
                prev[(y * small + x) * 4..(y * small + x) * 4 + 3].copy_from_slice(&col);
            }
        }
    }
    save_png(&out.join("textures").join("sheet-preview.png"), small, small, &prev)?;

    // Report.
    let mut join_list: Vec<&hair::Join> = ours.iter().collect();
    join_list.sort_by(|a, b| b.de0.total_cmp(&a.de0));
    let join_rows: Vec<Value> = join_list
        .iter()
        .map(|j| {
            let mut m = json!({
                "tris": [hs.tris[j.a], hs.tris[j.b]], "charts": [j.chart_a, j.chart_b], "kind": j.kind.label(),
                "turn_deg": (j.turn * 10.0).round() / 10.0, "tex_vs_flow_deg": (j.tex_vs_flow * 10.0).round() / 10.0,
            });
            let o = m.as_object_mut().unwrap();
            o.insert("regions".into(), json!([fl.region[j.a], fl.region[j.b]]));
            o.insert("locks".into(), json!([hs.lock[j.a], hs.lock[j.b]]));
            o.insert("normals_deg".into(), json!((hair::ang(hs.normal(j.a), hs.normal(j.b)) * 10.0).round() / 10.0));
            o.insert("lab".into(), json!((j.de0 * 100.0).round() / 100.0));
            o.insert("lab_max".into(), json!((j.de0_max * 100.0).round() / 100.0));
            o.insert("lab_mip3".into(), json!((j.de_mip * 100.0).round() / 100.0));
            o.insert("samples".into(), json!(j.samples));
            let c = hs.centroid(j.a);
            o.insert("at".into(), json!([(c[0] * 1000.0).round() / 1000.0, (c[1] * 1000.0).round() / 1000.0, (c[2] * 1000.0).round() / 1000.0]));
            o.insert("v".into(), json!([(pr.uv[j.a].iter().map(|q| q[1]).sum::<f64>() / 3.0 * 1000.0).round() / 1000.0, (pr.uv[j.b].iter().map(|q| q[1]).sum::<f64>() / 3.0 * 1000.0).round() / 1000.0]));
            o.insert("u_err".into(), json!([pr.u_err[j.a].round(), pr.u_err[j.b].round()]));
            m
        })
        .collect();
    let j: Value = json!({
        "input": input.display().to_string(),
        "output": glb_path.display().to_string(),
        "sheet": sheet_name, "size": size, "mesh": model.meshes[mesh].name,
        "hair_triangles": hs.tris.len(),
        "duplicates": { "same_winding": hs.dup_same, "back_faces": hs.dup_flipped, "with_own_stock_uv": own_uv_dups, "note": "duplicates share their primary's texels (they sit at the same place, so different paint would flicker)" },
        "locks": hs.nlocks, "regions": fl.region_lock.len(), "parting_edges": fl.partings, "nonmanifold_edges": hs.nonmanifold_edges,
        "ambiguous_locks": ambiguous_locks.iter().map(|a| json!({"lock": a.0, "ambiguous": a.1, "triangles": a.2})).collect::<Vec<_>>(),
        "root": { "a": root.a, "b": root.b },
        "v_off_flow_over_45": v_bad, "u_off_flow_over_45": u_bad,
        "charts": charts.len(), "texels_per_unit": scale,
        "palette": { "bands_lab": pal.bands, "hue": pal.hue, "chroma": pal.chroma, "l_iqr": pal.l_iqr },
        "joins": {
            "total": ours.len(), "continuous": count(&ours, JoinKind::Continuous), "partings": count(&ours, JoinKind::Parting),
            "fold_tips": count(&ours, JoinKind::FoldTip), "merges": count(&ours, JoinKind::Merge), "flow_breaks": breaks,
            "lab_median": d_med, "lab_max": d_max, "stock_lab_median": s_med, "stock_lab_max": s_max, "stock_flow_breaks_same_test": count(&stock_joins, JoinKind::Break),
            "list": join_rows
        },
        "fringe": { "ours": fr.iter().map(|f| json!({"level": f.0, "fringe": f.1, "shown": f.2})).collect::<Vec<_>>(),
                    "stock": sfr.iter().map(|f| json!({"level": f.0, "fringe": f.1, "shown": f.2})).collect::<Vec<_>>(),
                    "render_space": { "fringe_pixels": render_fringe, "hair_pixels": render_hair_px } },
        "form": form,
        "seam": { "points": seam.points, "lab_mean_before": seam.mean_before, "lab_max_before": seam.max_before, "over_before": seam.over_before,
                  "lab_mean_after": seam.mean_after, "lab_max_after": seam.max_after, "over_after": seam.over_after,
                  "stitch_texels_changed": seam.texels_changed, "stitch_mean_change": seam.mean_change },
        "gutter": { "texels": map.gutter, "from_surface": map.gutter_surface },
        "flow_global": agree.as_ref().map(|a| json!({ "layered_pairs": a.pairs, "couple": gf.couple, "couple_radius": gf.couple_radius,
            "angle_median_per_lock": a.median_before, "angle_p90_per_lock": a.p90_before, "over45_per_lock": a.over45_before,
            "angle_median_global": a.median_after, "angle_p90_global": a.p90_after, "over45_global": a.over45_after,
            "comb_strokes": gf.comb.len(), "combed_triangles": a.combed, "locks_against_tips": a.tip_against, "locks_flipped": a.tip_flipped })),
        "close_look": { "opts": { "small_px": lo.small, "flat_std": lo.flat_std, "flat_min": lo.flat_min, "jump_deg": lo.jump_deg, "sharp_target": sharp_target, "island_small": isl_small, "jump_tolerance": jump_tol },
            "ours": look_json(&look_ours), "stock": look_json(&look_stock), "views": look_views,
            "texture": { "ours": { "islands": ours_isl.0, "island_texels": ours_isl.1, "holes": ours_isl.2, "hole_texels": ours_isl.3 },
                         "stock": { "islands": stock_isl.0, "island_texels": stock_isl.1, "holes": stock_isl.2, "hole_texels": stock_isl.3 } },
            "failures": look_fail, "passed": look_fail.is_empty() },
        "alpha_coverage_by_mip": cov.iter().map(|c| (c * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        "grown_triangles": (0..ri.grown.len()).filter(|&t| ri.grown[t]).collect::<Vec<_>>(),
        "style": format!("{:?}", style),
        "seconds": t0.elapsed().as_secs_f64(),
    });
    let mut j = j;
    j.as_object_mut().unwrap().insert("label".into(), json!(ri.label));
    if let Some(mr) = &ri.model_report {
        j.as_object_mut().unwrap().insert("model".into(), mr.clone());
    }
    std::fs::write(out.join("checks.json"), serde_json::to_string_pretty(&j).unwrap()).map_err(|e| e.to_string())?;
    println!("{}: flow breaks: {breaks}; fringe texels (all mips): {fringe_total}; render-space fringe: {render_fringe}; Lab across joins median {d_med:.2}, max {d_max:.2}", ri.label);
    if !look_fail.is_empty() {
        println!("{}: close look FAILED: {}", ri.label, look_fail.join(", "));
    }
    let passed = breaks == 0 && fringe_total == 0 && render_fringe == 0 && look_fail.is_empty();
    Ok(RunOut { hs, fl, pr, root, glb: glb_path, passed })
}

/// One close-up view through the close-look detectors: the part is `hs`'s triangles (on `model`),
/// the layer per pixel is the lock. `cam_hs` frames the camera (the same for ours and stock).
#[allow(clippy::too_many_arguments)]
fn close_look(model: &Model, tex: &[Option<Texture>], hs: &HairSet, cam_hs: &HairSet, yaw: f32, pitch: f32, half: f32, px: usize, lo: &LookOpts) -> (LookStats, Vec<u8>) {
    let o = render::Options { cam: hair_cam(cam_hs, yaw, pitch, half), w: px, h: px, background: ([0, 0, 0], [0, 0, 0]), filter: Filter::Smooth, positions: None };
    let f = render::render(model, tex, &o, &|_, _| None);
    let off = render::tri_offsets(model)[hs.mesh];
    let mut lock_of = std::collections::HashMap::new();
    for (i, &t) in hs.tris.iter().enumerate() {
        lock_of.insert(off + t, hs.lock[i] as i64);
    }
    let layer: Vec<i64> = f.pick.iter().map(|&id| if id >= 0 { lock_of.get(&(id as usize)).copied().unwrap_or(-1) } else { -2 }).collect();
    let part: Vec<bool> = layer.iter().map(|&l| l >= 0).collect();
    let st = wb_core::hairlook::measure(&View { w: px, h: px, rgba: &f.rgba, part: &part, layer: &layer }, lo);
    // Debug picture: each lock its own color; a front and a back face of one lock differ in
    // brightness (pick id parity is not meaningful, so use the triangle's facing instead).
    let mut img = vec![0u8; px * px * 4];
    for i in 0..px * px {
        let c: [u8; 3] = if layer[i] >= 0 {
            let hsh = ((layer[i] + 1) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40;
            let back = f.pick[i] >= 0 && hs.tris.binary_search(&(f.pick[i] as usize - off)).map(|k| !hs.is_primary(k)).unwrap_or(false);
            let k = if back { 0.45 } else { 1.0 };
            [((60 + (hsh & 0xBF)) as f64 * k) as u8, ((60 + ((hsh >> 8) & 0xBF)) as f64 * k) as u8, ((60 + ((hsh >> 16) & 0xBF)) as f64 * k) as u8]
        } else if layer[i] == -1 {
            [40, 40, 40]
        } else {
            [0, 0, 0]
        };
        img[i * 4..i * 4 + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
    }
    (st, img)
}

fn look_json(s: &LookStats) -> Value {
    json!({ "hair_px": s.part_px, "specks": s.specks, "speck_px": s.speck_px, "holes": s.holes, "hole_px": s.hole_px, "tiny": s.tiny,
            "sharpness": (s.sharpness * 1000.0).round() / 1000.0, "sharp_px": s.sharp_px, "flat_px": s.flat_px, "flat_max": s.flat_max,
            "layer_edge_px": s.layer_edge_px, "jump_px": s.jump_px,
            "places": s.places.iter().take(30).map(|p| json!({"kind": p.0, "x": p.1, "y": p.2, "size": p.3})).collect::<Vec<_>>() })
}

/// Stock UV islands of the hair triangles (edge-connected in weld id + UV).
fn stock_islands(model: &Model, hs: &HairSet) -> Vec<usize> {
    let m = &model.meshes[hs.mesh];
    let n = hs.tris.len();
    let mut par: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let key = |v: u32| {
        let uv = m.uv[v as usize];
        (hs.wid[v as usize], (uv[0] * 1e5).round() as i64, (uv[1] * 1e5).round() as i64)
    };
    let mut seen: std::collections::HashMap<((usize, i64, i64), (usize, i64, i64)), usize> = std::collections::HashMap::new();
    for i in 0..n {
        let t = m.tris[hs.tris[i]];
        for k in 0..3 {
            let (a, b) = (key(t[k]), key(t[(k + 1) % 3]));
            let e = if a < b { (a, b) } else { (b, a) };
            match seen.get(&e) {
                Some(&j) => {
                    let (x, y) = (find(&mut par, i), find(&mut par, j));
                    par[x] = y;
                }
                None => {
                    seen.insert(e, i);
                }
            }
        }
    }
    (0..n).map(|i| find(&mut par, i)).collect()
}

use image::ImageEncoder;
