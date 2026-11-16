//! `zidane-inspector hair`: hair texturing from parameters (see `wb_core::hair`).
//!
//! Game-agnostic: point it at any GLB, sheet and hair selection. FF9-only details are options
//! (`--keep-blink` for field Zidane's eye-blink boxes, the UV kit's part map for `--part`).
use crate::Args;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use wb_core::analysis::Parts;
use wb_core::hair_model::{self, Backfaces, ModelOpts};
use wb_core::hair::{self, BakeIn, Chart, Flow, FlowOpts, HairSet, JoinIn, JoinKind, Palette, Root, Select, Sheet, Style, SynthCtx, V3};
use wb_core::render::{self, Camera, Filter};
use wb_core::{font, image_textures, Model, Texture};

pub const HAIR_USAGE: &str = "  zidane-inspector hair file.glb --out-dir NEW_FOLDER [options]
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
      wave length x1.6).";

fn opt<'a>(args: &'a Args, k: &str) -> Option<&'a str> {
    args.0.iter().position(|a| a == k).and_then(|i| args.0.get(i + 1)).map(String::as_str)
}
fn opts<'a>(args: &'a Args, k: &str) -> Vec<&'a str> {
    args.0.windows(2).filter(|w| w[0] == k).map(|w| w[1].as_str()).collect()
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
    let sel = match opt(args, "--part") {
        Some(part) => {
            let dir = PathBuf::from(opt(args, "--parts").unwrap_or(crate::KIT_PARTS));
            let parts = Parts::load(&dir)?;
            let id = parts.id_by_name(part).ok_or(format!("no part named {part}"))?;
            Select::Mask(parts.triangle_parts(&model, mesh).iter().map(|&p| p == id).collect())
        }
        None => Select::All,
    };
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_owned();
    let name = opt(args, "--name").map(str::to_owned).unwrap_or_else(|| format!("{stem}-hair.glb"));
    if !args.flag("--rebuild-model") {
        let r = run(args, &RunIn { input: &input, model: &model, orig: &model, mesh, sel, out: &out, name, root: None, flow: None, param: None, opaque: false, extra: Vec::new(), label: "hair tool", model_report: None, renders: true, t0 })?;
        return if r.passed { Ok(()) } else { Err("hair checks failed (see checks.json)".into()) };
    }
    // Fallback first: the stock mesh with flow UVs and the strand texture.
    let fb_name = name.replace(".glb", "-stock-mesh.glb");
    let fb = run(args, &RunIn { input: &input, model: &model, orig: &model, mesh, sel, out: &out.join("stock-mesh"), name: fb_name, root: None, flow: None, param: None, opaque: false, extra: Vec::new(), label: "stock mesh", model_report: None, renders: false, t0 })?;
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
    };
    let hm = hair_model::remesh(&model, &fb.hs, &fb.fl, &fb.pr, &stock, &mo)?;
    let r = &hm.report;
    println!(
        "  [{:5.1} s] model: {} front + {} back triangles, {} vertices (stock {} front + {} back); spacing {:.4}; {} regions remeshed whole, {} per triangle ({} folded triangles in strand space); {} fold edges kept; {} constraints skipped, {} points merged; area stock {:.3}, remeshed {:.3}, after alpha trim {:.3}; gaps closed at {} vertices (max move {:.4}), boundary distance to other locks in spacings [0,.25,.5,1,2,4+): {:?}",
        t0.elapsed().as_secs_f64(), r.front, r.back, r.vertices, r.stock_front, r.stock_back, r.spacing, r.regions_remeshed, r.regions_per_triangle, r.flipped_tris, r.fold_edges, r.constraints_skipped, r.merged_points, r.area_stock, r.area_before_trim, r.area_after_trim, r.gap_vertices, r.gap_max_move, r.gap_hist
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
        "gap_vertices": r.gap_vertices, "gap_max_move": r.gap_max_move, "boundary_gap_hist_spacings": r.gap_hist,
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
    }
    let hs = HairSet::build(model, mesh, &ri.sel)?;
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
    let (fl, split) = match ri.flow {
        Some(f) => (f(&hs), 0),
        None => {
            let mut fl: Flow = hair::flow(model, &hs, &stock, &fo);
            let split = if args.flag("--no-root-split") { 0 } else { hair::split_at_root_plane(&hs, &mut fl, &root) };
            (fl, split)
        }
    };
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
    let ctx = SynthCtx { style: &style, pal: &pal, size: hair_size, vmax };
    let bin = BakeIn { model: &model, hs: &hs, fl: &fl, pr: &pr, charts: &charts, scale, size, pad, stock: Sheet { w: stock.w, h: stock.h, rgba: &stock_tex.rgba }, stock_alpha: sa, ao, ctx };
    let map = hair::bake(&bin, &mut rgba);
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
    let suv: Vec<[[f64; 2]; 3]> = (0..hs.tris.len())
        .map(|i| std::array::from_fn(|c| {
            let q = hair::sheet_uv(&charts[chart_of[i]], scale, size, pr.uv[i][c]);
            [q[0] as f64, q[1] as f64]
        }))
        .collect();
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
    // Stock fringe on the stock hair texels, for comparison.
    let mut stock_mask = vec![false; stock_tex.w * stock_tex.h];
    for i in 0..hs.tris.len() {
        let p = hair::tri_uv(&model, &hs, i).map(|q| [q[0] * stock_tex.w as f64, q[1] * stock_tex.h as f64]);
        wb_core::uv::raster(p, stock_tex.w, stock_tex.h, 0.0, |x, y| stock_mask[y * stock_tex.w + x] = true);
    }
    let sfr = hair::fringe(&stock_tex, &stock_mask, &pal, 5);
    let fringe_total: usize = fr.iter().map(|f| f.1).sum();
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
    let passed = breaks == 0 && fringe_total == 0 && render_fringe == 0;
    Ok(RunOut { hs, fl, pr, root, glb: glb_path, passed })
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
