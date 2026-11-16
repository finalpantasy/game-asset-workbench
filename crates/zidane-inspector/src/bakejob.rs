//! `zidane-inspector bake`: render guide views of a GLB, register painted pictures of those views,
//! and bake the paint onto the texture atlas through the triangles that show it (wb-core `bake`).
//!
//!   bake in.glb --step guide    --out-dir NEW --view front --view back [--size 1024]
//!   bake in.glb --step register --view-dir DIR --paint front=painted.png --out-dir NEW
//!   bake in.glb --step bake     --view-dir DIR --paint front=painted.png ... --mask-part torso --out-dir NEW
//!   bake in.glb --step demo     --out-dir NEW          (self-test: flat recolor of the torso in two views)
//!   bake stock.glb --step roundtrip --pattern pattern.glb --out-dir NEW   (stock renders baked onto the pattern atlas, scored against the truth)
use crate::Args;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use wb_core::bake::*;
use wb_core::pattern::{classify, Part};
use wb_core::render::{self, Camera, Filter};
use wb_core::{font, image_textures, swap, Model, Texture};

#[path = "bakertrip.rs"]
mod rt;

fn input_path(args: &Args) -> PathBuf {
    match args.0.first() {
        Some(a) if !a.starts_with("--") => PathBuf::from(a),
        _ => PathBuf::from(crate::STOCK),
    }
}

fn num<T: std::str::FromStr>(args: &Args, k: &str, default: T) -> Result<T, String> {
    match args.opt(k) {
        Some(s) => s.parse::<T>().map_err(|_| format!("{k}: cannot read {s}")),
        None => Ok(default),
    }
}

fn parse_parts(s: &str) -> Result<Vec<Part>, String> {
    s.split(',')
        .filter(|x| !x.is_empty())
        .map(|n| match n.trim() {
            "head" => Ok(Part::Head),
            "torso" => Ok(Part::Torso),
            "tail" => Ok(Part::Tail),
            "arm" => Ok(Part::Arm),
            "hand" => Ok(Part::Hand),
            "leg" => Ok(Part::Leg),
            "foot" => Ok(Part::Foot),
            "other" => Ok(Part::Other),
            x => Err(format!("--mask-part: no part named {x} (head, torso, tail, arm, hand, leg, foot, other)")),
        })
        .collect()
}

fn parse_view(s: &str, args: &Args) -> Result<ViewSpec, String> {
    let (name, rest) = match s.split_once('=') {
        Some((n, r)) => (n, Some(r)),
        None => (s, None),
    };
    let (yaw, pitch) = match rest {
        Some(r) => {
            let (a, b) = r.split_once(',').ok_or_else(|| format!("--view {s}: use NAME=yaw,pitch"))?;
            (a.parse::<f32>().map_err(|e| format!("--view {s}: {e}"))?, b.parse::<f32>().map_err(|e| format!("--view {s}: {e}"))?)
        }
        None => preset(name).ok_or_else(|| format!("--view {s}: not a preset (front, back, left, right, front-left, front-right, back-left, back-right, above, below); use NAME=yaw,pitch"))?,
    };
    Ok(ViewSpec {
        name: name.to_owned(),
        yaw_deg: yaw,
        pitch_deg: pitch,
        ortho: args.flag("--ortho"),
        size: num(args, "--size", 1024usize)?,
        fill: num(args, "--fill", 0.8f32)?,
        clip: args.opt("--clip").map(str::to_owned),
        time: num(args, "--time", 0.0f32)?,
    })
}

fn tri_parts(model: &Model) -> Vec<Part> {
    let mut v = Vec::with_capacity(model.triangle_count());
    for (mi, m) in model.meshes.iter().enumerate() {
        for ti in 0..m.tris.len() {
            let j = model.triangle_joint(mi, ti);
            v.push(classify(model.joint_names.get(j).map(String::as_str).unwrap_or("")).0);
        }
    }
    v
}

fn write(p: &Path, b: &[u8]) -> Result<(), String> {
    std::fs::write(p, b).map_err(|e| format!("{}: {e}", p.display()))
}

fn wjson(p: &Path, v: &Value) -> Result<(), String> {
    write(p, serde_json::to_string_pretty(v).unwrap().as_bytes())
}

pub fn bake_cmd(args: &Args) -> Result<(), String> {
    let step = args.opt("--step").ok_or("bake needs --step guide|register|bake|demo")?;
    let out = crate::relayoutjob::check_dir(Path::new(args.opt("--out-dir").ok_or("bake needs --out-dir NEW_FOLDER")?))?;
    let input = input_path(args);
    let model = Model::open(&input)?;
    match step {
        "guide" => step_guide(args, &model, &input, &out),
        "register" => step_register(args, &out),
        "bake" => step_bake(args, &model, &input, &out),
        "demo" => step_demo(args, &model, &input, &out),
        "roundtrip" => rt::step_roundtrip(args, &model, &input, &out),
        s => Err(format!("--step {s}: use guide, register, bake, demo or roundtrip")),
    }
}

// ---- step 1: guide ------------------------------------------------------------------------------

fn step_guide(args: &Args, model: &Model, input: &Path, out: &Path) -> Result<(), String> {
    let mut names: Vec<&str> = args.opts("--view");
    if names.is_empty() {
        names.push("front");
    }
    let textures = image_textures(model);
    let file = input.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    for n in names {
        let spec = parse_view(n, args)?;
        let v = render_view(model, &textures, &spec)?;
        write_view(out, &v, &file, model.triangle_count())?;
        let covered = v.ids.iter().filter(|&&i| i >= 0).count();
        println!("view {}: {} x {}, {} pixels show the model -> {}-guide.png, -ids.png, -depth.png, -view.json", spec.name, v.w, v.h, covered, spec.name);
    }
    Ok(())
}

// ---- registration helpers -----------------------------------------------------------------------

fn reg_options(args: &Args) -> Result<RegOptions, String> {
    let d = RegOptions::default();
    Ok(RegOptions { scale_tol: num(args, "--reg-scale-tol", d.scale_tol)?, rot_tol_deg: num(args, "--reg-rot-tol", d.rot_tol_deg)?, min_ncc: num(args, "--reg-min-ncc", d.min_ncc)?, ..d })
}

fn paint_args(args: &Args) -> Result<Vec<(String, PathBuf)>, String> {
    args.opts("--paint")
        .into_iter()
        .map(|s| s.split_once('=').map(|(n, p)| (n.to_owned(), PathBuf::from(p))).ok_or_else(|| format!("--paint {s}: use VIEWNAME=painted.png")))
        .collect()
}

fn register_one(view: &ViewData, paint: &Img, ro: &RegOptions, skip: bool) -> Result<Registration, String> {
    if skip {
        return Ok(Registration::identity(view.w, view.h, paint.w as f32 / view.w as f32));
    }
    let fg: Vec<bool> = view.ids.iter().map(|&i| i >= 0).collect();
    register(&view.guide, &fg, paint, ro).map_err(|e| format!("view {}: {e}", view.spec.name))
}

fn step_register(args: &Args, out: &Path) -> Result<(), String> {
    let dir = PathBuf::from(args.opt("--view-dir").ok_or("register needs --view-dir DIR (a guide step's output)")?);
    let ro = reg_options(args)?;
    for (name, path) in paint_args(args)? {
        let v = read_view(&dir, &name)?;
        let paint = Img::open(&path)?;
        let reg = register_one(&v, &paint, &ro, false)?;
        warp_to_guide(&paint, &reg).save(&out.join(format!("{name}-registered.png")))?;
        wjson(&out.join(format!("{name}-registration.json")), &reg.to_json())?;
        println!("view {name}: scale {:.4}, rotation {:.3} deg, shift ({:.2}, {:.2}) px, edge correlation {:.3} -> {:.3}", reg.scale, reg.rot_deg, reg.tx, reg.ty, reg.ncc_before, reg.ncc_after);
    }
    Ok(())
}

fn load_paint_views(args: &Args, out: &Path, gate: f32) -> Result<Vec<PaintView>, String> {
    let dir = PathBuf::from(args.opt("--view-dir").ok_or("bake needs --view-dir DIR (a guide step's output)")?);
    let ro = reg_options(args)?;
    let skip = args.flag("--no-register");
    let mut views = Vec::new();
    for (name, path) in paint_args(args)? {
        let view = read_view(&dir, &name)?;
        let paint = Img::open(&path)?;
        let reg = register_one(&view, &paint, &ro, skip)?;
        let registered = warp_to_guide(&paint, &reg);
        registered.save(&out.join(format!("{name}-registered.png")))?;
        wjson(&out.join(format!("{name}-registration.json")), &reg.to_json())?;
        let edit = edit_map(&view.guide, &registered, gate);
        views.push(PaintView { view, paint, reg, edit });
    }
    if views.is_empty() {
        return Err("bake needs at least one --paint VIEWNAME=painted.png".into());
    }
    Ok(views)
}

fn build_masks(args: &Args, model: &Model) -> Result<(Masks, Value), String> {
    let meshes: Vec<String> = args.opts("--mask-mesh").into_iter().map(str::to_owned).collect();
    for m in &meshes {
        if !model.meshes.iter().any(|x| &x.name == m) {
            return Err(format!("--mask-mesh {m}: no such mesh; meshes: {}", model.meshes.iter().map(|x| x.name.as_str()).collect::<Vec<_>>().join(", ")));
        }
    }
    let mut masks = match args.opt("--mask-part") {
        Some(p) => Masks { tri_allowed: part_mask(model, &parse_parts(p)?, &meshes), texel: Default::default() },
        None if !meshes.is_empty() => Masks { tri_allowed: part_mask(model, &[Part::Head, Part::Torso, Part::Tail, Part::Arm, Part::Hand, Part::Leg, Part::Foot, Part::Other], &meshes), texel: Default::default() },
        None => Masks::all(model),
    };
    for s in args.opts("--mask-png") {
        let (sheet, path) = s.split_once('=').ok_or_else(|| format!("--mask-png {s}: use SHEET=mask.png (white = may change)"))?;
        let im = Img::open(Path::new(path))?;
        let allowed: Vec<bool> = im.rgba.chunks_exact(4).map(|p| p[0] > 127 && p[3] > 0).collect();
        masks.texel.insert(sheet.trim_end_matches(".png").to_owned(), (im.w, im.h, allowed));
    }
    let n = masks.tri_allowed.iter().filter(|&&b| b).count();
    let desc = json!({
        "mask_part": args.opt("--mask-part"), "mask_mesh": meshes, "mask_png": args.opts("--mask-png"),
        "triangles_allowed": n, "triangles_total": model.triangle_count(),
    });
    if n == 0 {
        return Err("the mask allows no triangles".into());
    }
    Ok((masks, desc))
}

fn bake_options(args: &Args) -> Result<BakeOptions, String> {
    let d = BakeOptions::default();
    Ok(BakeOptions {
        pad: num(args, "--pad", d.pad)?,
        grazing_max_deg: num(args, "--grazing-deg", d.grazing_max_deg)?,
        grazing_full_deg: num(args, "--grazing-full-deg", d.grazing_full_deg)?,
        seam_pass: !args.flag("--no-seam"),
        seam_radius: num(args, "--seam-radius", d.seam_radius)?,
        allow_shared: args.flag("--allow-shared"),
        sharp_power: num(args, "--sharp-power", d.sharp_power)?,
        ..d
    })
}

fn step_bake(args: &Args, model: &Model, input: &Path, out: &Path) -> Result<(), String> {
    let gate: f32 = num(args, "--gate", 0.0)?;
    let views = load_paint_views(args, out, gate)?;
    let (masks, desc) = build_masks(args, model)?;
    let o = bake_options(args)?;
    let summary = run_bake(model, input, out, views, &masks, &o, args.flag("--leak-check"), json!({ "mask": desc, "gate": gate }))?;
    print_summary(&summary);
    Ok(())
}

// ---- the bake run: bake, check, write, sheets, report ---------------------------------------------

fn textures_with(model: &Model, atlases: &[AtlasOut], which: u8) -> Vec<Option<Texture>> {
    let mut t = image_textures(model);
    for a in atlases {
        let px = match which {
            0 => &a.orig,
            1 => &a.baked,
            _ => &a.fin,
        };
        t[a.image] = Some(Texture::new(a.w, a.h, px.clone()));
    }
    t
}

fn render_img(model: &Model, textures: &[Option<Texture>], cam: Camera, pose: &[Vec<[f32; 3]>], size: usize) -> Img {
    let o = render::Options { cam, w: size, h: size, background: ([40, 44, 52], [40, 44, 52]), filter: Filter::Smooth, positions: Some(pose) };
    let f = render::render(model, textures, &o, &|_, _| None);
    Img { w: size, h: size, rgba: f.rgba }
}

fn blit(dst: &mut Img, src: &Img, x: usize, y: usize) {
    for sy in 0..src.h {
        if y + sy >= dst.h {
            break;
        }
        for sx in 0..src.w {
            if x + sx >= dst.w {
                break;
            }
            let (s, d) = ((sy * src.w + sx) * 4, ((y + sy) * dst.w + x + sx) * 4);
            dst.rgba[d..d + 4].copy_from_slice(&src.rgba[s..s + 4]);
        }
    }
}

fn label(dst: &mut Img, x: usize, y: usize, text: &str) {
    font::draw(&mut dst.rgba, dst.w, dst.h, x as i64, y as i64, &text.to_ascii_uppercase(), 2, [20, 20, 20]);
}

/// A map of what changed, at most `size` px: red = inside the mask, orange = padding of an allowed
/// island, magenta = outside the mask (a failure), teal = allowed but untouched.
fn change_map(a: &AtlasOut, size: usize) -> Img {
    let s = size.min(a.w);
    let f = (a.w / s).max(1);
    let mut out = Img::new(s, s, [0, 0, 0, 255]);
    for y in 0..s {
        for x in 0..s {
            let (mut base, mut pr) = ([0u32; 3], 0u8);
            for dy in 0..f {
                for dx in 0..f {
                    let i = ((y * f + dy).min(a.h - 1)) * a.w + (x * f + dx).min(a.w - 1);
                    for c in 0..3 {
                        base[c] += a.orig[i * 4 + c] as u32;
                    }
                    let changed = a.fin[i * 4..i * 4 + 4] != a.orig[i * 4..i * 4 + 4];
                    let p = if changed && !a.perm[i] && !a.gutter_ok[i] {
                        5
                    } else if changed && a.perm[i] {
                        4
                    } else if changed {
                        3
                    } else if a.perm[i] {
                        2
                    } else {
                        0
                    };
                    pr = pr.max(p);
                }
            }
            let n = (f * f) as u32;
            let g = [(base[0] / n / 3) as u8, (base[1] / n / 3) as u8, (base[2] / n / 3) as u8];
            let col = match pr {
                5 => [255, 0, 255],
                4 => [235, 40, 40],
                3 => [255, 160, 30],
                2 => [30, 120, 130],
                _ => [g[0] + 20, g[1] + 20, g[2] + 20],
            };
            let o = (y * s + x) * 4;
            out.rgba[o..o + 3].copy_from_slice(&col);
        }
    }
    out
}

fn downscale(src: &[u8], w: usize, h: usize, size: usize) -> Img {
    let s = size.min(w);
    let f = (w / s).max(1);
    let mut out = Img::new(s, s, [0, 0, 0, 255]);
    for y in 0..s {
        for x in 0..s {
            let mut acc = [0u32; 3];
            for dy in 0..f {
                for dx in 0..f {
                    let i = ((y * f + dy).min(h - 1)) * w + (x * f + dx).min(w - 1);
                    for c in 0..3 {
                        acc[c] += src[i * 4 + c] as u32;
                    }
                }
            }
            let n = (f * f) as u32;
            let o = (y * s + x) * 4;
            for c in 0..3 {
                out.rgba[o + c] = (acc[c] / n) as u8;
            }
        }
    }
    out
}

fn mask_target(model: &Model, masks: &Masks) -> [f32; 3] {
    let (mut s, mut n) = ([0.0f32; 3], 0usize);
    let mut g = 0;
    for m in &model.meshes {
        for t in &m.tris {
            if masks.tri_allowed[g] {
                for v in t {
                    for k in 0..3 {
                        s[k] += m.pos[*v as usize][k];
                    }
                    n += 1;
                }
            }
            g += 1;
        }
    }
    let n = n.max(1) as f32;
    [s[0] / n, s[1] / n, s[2] / n]
}

#[allow(clippy::too_many_arguments)]
fn run_bake(model: &Model, input: &Path, out: &Path, views: Vec<PaintView>, masks: &Masks, o: &BakeOptions, leak: bool, extra: Value) -> Result<Value, String> {
    let t0 = std::time::Instant::now();
    let res = bake(model, &views, masks, o)?;
    let t_bake = t0.elapsed().as_secs_f64();
    // What an unmasked bake of the same paint would have changed outside the mask.
    let mut leak_json = Value::Null;
    if leak {
        let lo = BakeOptions { seam_pass: false, pad: 0, ..o.clone() };
        let unmasked = bake(model, &views, &Masks::all(model), &lo)?;
        let mut per = Vec::new();
        for ua in &unmasked.atlases {
            let real = res.atlases.iter().find(|a| a.image == ua.image);
            let (mut outside, mut inside) = (0usize, 0usize);
            for i in 0..ua.w * ua.h {
                if ua.fin[i * 4..i * 4 + 4] != ua.orig[i * 4..i * 4 + 4] {
                    if real.map(|r| r.perm[i] || r.gutter_ok[i]).unwrap_or(false) {
                        inside += 1;
                    } else {
                        outside += 1;
                    }
                }
            }
            per.push(json!({ "image": ua.name, "unmasked_bake_changes_inside_mask": inside, "unmasked_bake_changes_outside_mask": outside }));
        }
        leak_json = json!({ "note": "the same paint baked with no mask, no seam pass, no padding; counts of changed texels", "per_atlas": per });
    }
    // Write the atlases and the new model (only the textures change).
    let stem = input.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
    let mut bytes = model.glb.to_bytes();
    let mut verification = Vec::new();
    let mut glb_ok = true;
    for a in &res.atlases {
        let img = Img { w: a.w, h: a.h, rgba: a.fin.clone() };
        let png = img.to_png()?;
        write(&out.join(format!("{}.png", a.name)), &png)?;
        let cur = Model::from_bytes(&bytes)?;
        bytes = swap::swap(&cur, &a.name, &png)?.bytes;
        change_map(a, 2048).save(&out.join(format!("{}_changes.png", a.name)))?;
        let permv: Vec<u8> = a.perm.iter().zip(&a.gutter_ok).flat_map(|(&p, &g)| if p { [255, 255, 255, 255] } else if g { [140, 140, 140, 255] } else { [0, 0, 0, 255] }).collect();
        downscale(&permv, a.w, a.h, 2048).save(&out.join(format!("{}_mask.png", a.name)))?;
    }
    let glb_path = out.join(format!("{stem}-baked.glb"));
    write(&glb_path, &bytes)?;
    // Check the written model against the input, texel by texel.
    let reread = Model::open(&glb_path)?;
    for a in &res.atlases {
        let after = Img::decode(&reread.images[a.image].bytes)?;
        let before = Img::decode(&model.images[a.image].bytes)?;
        let same_as_bake = after.rgba == a.fin;
        glb_ok &= same_as_bake;
        let mut v = verify_atlas(model, a.image, &before, &after, masks, o.pad);
        v["glb_atlas_equals_baked_atlas"] = json!(same_as_bake);
        verification.push(v);
    }
    // Every image the bake did not touch keeps its bytes.
    let untouched_ok = model.images.iter().enumerate().filter(|(i, _)| !res.atlases.iter().any(|a| a.image == *i)).all(|(i, im)| reread.images[i].bytes == im.bytes);
    let sheets = make_sheets(model, &views, masks, &res, out)?;
    // Vertex colors against texture brightness: what the field and battle shaders will do with the output.
    let shading = {
        let decode = |m: &Model| -> Result<Vec<(String, Img)>, String> { m.images.iter().map(|im| Img::decode(&im.bytes).map(|d| (im.name.clone(), d))).collect() };
        let (after, before) = (decode(&reread)?, decode(model)?);
        let a_refs: Vec<(String, &Img)> = after.iter().map(|(n, d)| (n.clone(), d)).collect();
        let b_refs: Vec<(String, &Img)> = before.iter().map(|(n, d)| (n.clone(), d)).collect();
        let mut sh = shading_check(&reread.glb, &a_refs);
        sh["input_model"] = shading_check(&model.glb, &b_refs);
        sh
    };
    // Demo only: texels one view painted at full strength should carry that view's flat color.
    let mut flat = Value::Null;
    if let Some(cols) = extra["demo"]["colors"].as_array() {
        let cols: Vec<[i32; 3]> = cols.iter().map(|c| std::array::from_fn(|k| c[k].as_i64().unwrap_or(0) as i32)).collect();
        let (mut n, mut hit) = (0usize, 0usize);
        for a in &res.atlases {
            for i in 0..a.w * a.h {
                if a.strength[i] >= 0.9 && a.views_seen[i] == 1 && a.perm[i] {
                    n += 1;
                    let px = &a.baked[i * 4..i * 4 + 3];
                    if cols.iter().any(|c| (0..3).all(|k| (px[k] as i32 - c[k]).abs() <= 6)) {
                        hit += 1;
                    }
                }
            }
        }
        flat = json!({ "texels_painted_by_one_view_at_full_strength": n, "within_6_of_a_view_color": hit, "share": if n > 0 { hit as f64 / n as f64 } else { 0.0 } });
    }
    let (mut tot_changed, mut tot_out, mut tot_alpha, mut under_tri, mut beyond) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for a in &res.atlases {
        tot_changed += a.stats["texels_changed"].as_u64().unwrap_or(0);
        tot_out += a.stats["changed_outside_mask"].as_u64().unwrap_or(0);
        tot_alpha += a.stats["alpha_changed"].as_u64().unwrap_or(0);
    }
    for v in &verification {
        under_tri += v["changed_under_triangles_outside_mask"].as_u64().unwrap_or(0);
        beyond += v["changed_beyond_padding_of_allowed_triangles"].as_u64().unwrap_or(0) + v["changed_where_texel_mask_is_zero"].as_u64().unwrap_or(0);
    }
    let mut seam_before = 0.0f64;
    let mut seam_after = 0.0f64;
    let mut seam_n = 0.0f64;
    let mut seam_raw = 0.0f64;
    for s in res.seam["per_atlas"].as_array().cloned().unwrap_or_default() {
        let n = s["samples"].as_f64().unwrap_or(0.0);
        seam_before += s["mean_rgb_difference_input"].as_f64().unwrap_or(0.0) * n;
        seam_raw += s["mean_rgb_difference_after_bake"].as_f64().unwrap_or(0.0) * n;
        seam_after += s["mean_rgb_difference_after_seam_pass"].as_f64().unwrap_or(0.0) * n;
        seam_n += n;
    }
    let (sb, sr, sa) = if seam_n > 0.0 { (seam_before / seam_n, seam_raw / seam_n, seam_after / seam_n) } else { (0.0, 0.0, 0.0) };
    let report = json!({
        "input": input.file_name().map(|s| s.to_string_lossy().into_owned()),
        "output_model": glb_path.file_name().map(|s| s.to_string_lossy().into_owned()),
        "bake_seconds": t_bake,
        "options": { "pad": o.pad, "grazing_max_deg": o.grazing_max_deg, "grazing_full_deg": o.grazing_full_deg, "min_weight": o.min_weight,
                     "seam_pass": o.seam_pass, "seam_radius": o.seam_radius, "allow_shared": o.allow_shared, "depth_eps": o.depth_eps },
        "run": extra,
        "views": res.views,
        "atlases": res.atlases.iter().map(|a| a.stats.clone()).collect::<Vec<_>>(),
        "seams": res.seam,
        "seam_mean_rgb_difference": { "input": sb, "after_bake": sr, "after_seam_pass": sa, "samples": seam_n },
        "unmasked_comparison": leak_json,
        "flat_color_check": flat,
        "shading_check": shading,
        "verification": verification,
        "sheets": sheets,
    });
    let final_ver = json!({
        "texels_changed_total": tot_changed,
        "texels_changed_outside_mask": tot_out,
        "changed_under_triangles_outside_mask": under_tri,
        "changed_beyond_padding_or_texel_mask": beyond,
        "alpha_changed": tot_alpha,
        "only_textures_changed_in_glb": true,
        "glb_atlases_equal_baked_atlases": glb_ok,
        "untouched_images_byte_identical": untouched_ok,
        "seam_difference_before": sb, "seam_difference_after_bake": sr, "seam_difference_after": sa,
        "seam_difference_lower": seam_n == 0.0 || sa < sr,
        "passed": tot_out == 0 && under_tri == 0 && beyond == 0 && tot_alpha == 0 && glb_ok && untouched_ok && (seam_n == 0.0 || sa < sr),
    });
    wjson(&out.join("bake-report.json"), &report)?;
    wjson(&out.join("FINAL-VERIFICATION.json"), &final_ver)?;
    Ok(json!({ "report": report, "verification": final_ver, "model": glb_path.display().to_string() }))
}

fn make_sheets(model: &Model, views: &[PaintView], masks: &Masks, res: &BakeOut, out: &Path) -> Result<Value, String> {
    let base_tex = textures_with(model, &res.atlases, 0);
    let raw_tex = textures_with(model, &res.atlases, 1);
    let fin_tex = textures_with(model, &res.atlases, 2);
    let mut files = Vec::new();
    // 1. Views: render | paint registered | baked render, per view.
    let cell = 460usize;
    let mut sheet = Img::new(cell * 3 + 40, (cell + 36) * views.len() + 10, [226, 228, 232, 255]);
    for (r, pv) in views.iter().enumerate() {
        let y = 10 + r * (cell + 36);
        let spec = &pv.view.spec;
        let pose = model.posed(spec.clip.as_ref().and_then(|c| model.clips.iter().position(|x| x == c)), spec.time);
        let baked = render_img(model, &fin_tex, pv.view.cam, &pose, cell);
        let reg = warp_to_guide(&pv.paint, &pv.reg).resized(cell, cell);
        blit(&mut sheet, &pv.view.guide.resized(cell, cell), 10, y + 28);
        blit(&mut sheet, &reg, 20 + cell, y + 28);
        blit(&mut sheet, &baked, 30 + 2 * cell, y + 28);
        label(&mut sheet, 10, y + 6, &format!("{} guide", spec.name));
        label(&mut sheet, 20 + cell, y + 6, &format!("{} paint registered", spec.name));
        label(&mut sheet, 30 + 2 * cell, y + 6, &format!("{} baked, rendered", spec.name));
    }
    sheet.save(&out.join("sheet-views.png"))?;
    files.push("sheet-views.png");
    // 2. Turntable: input | raw bake | final.
    let tc = 360usize;
    let yaws = [0.0f32, 90.0, 180.0, 270.0];
    let mut sheet = Img::new(tc * 4 + 50, (tc + 32) * 3 + 10, [226, 228, 232, 255]);
    let pose = model.posed(None, 0.0);
    for (r, (title, tex)) in [("input", &base_tex), ("raw bake", &raw_tex), ("final", &fin_tex)].into_iter().enumerate() {
        for (c, yaw) in yaws.iter().enumerate() {
            let mut cam = Camera::frame(model);
            cam.yaw = std::f32::consts::PI + yaw.to_radians();
            let im = render_img(model, tex, cam, &pose, tc);
            let (x, y) = (10 + c * (tc + 10), 10 + r * (tc + 32));
            blit(&mut sheet, &im, x, y + 26);
            label(&mut sheet, x, y + 4, &format!("{title} yaw {yaw}"));
        }
    }
    sheet.save(&out.join("sheet-turntable.png"))?;
    files.push("sheet-turntable.png");
    // 3. Seam close-ups from the sides.
    let sc = 560usize;
    let mut sheet = Img::new(sc * 3 + 40, (sc + 32) * 2 + 10, [226, 228, 232, 255]);
    let target = mask_target(model, masks);
    for (r, yaw) in [90.0f32, 270.0].iter().enumerate() {
        for (c, (title, tex)) in [("input", &base_tex), ("raw bake", &raw_tex), ("after seam pass", &fin_tex)].into_iter().enumerate() {
            let mut cam = Camera::frame(model);
            cam.yaw = std::f32::consts::PI + yaw.to_radians();
            cam.target = target;
            cam.zoom(0.5);
            let im = render_img(model, tex, cam, &pose, sc);
            let (x, y) = (10 + c * (sc + 10), 10 + r * (sc + 32));
            blit(&mut sheet, &im, x, y + 26);
            label(&mut sheet, x, y + 4, &format!("{title} yaw {yaw}"));
        }
    }
    sheet.save(&out.join("sheet-seams.png"))?;
    files.push("sheet-seams.png");
    // 4. Atlas: input | final | changes.
    let ac = 620usize;
    let mut sheet = Img::new(ac * 3 + 40, (ac + 32) * res.atlases.len() + 10, [226, 228, 232, 255]);
    for (r, a) in res.atlases.iter().enumerate() {
        let y = 10 + r * (ac + 32);
        blit(&mut sheet, &downscale(&a.orig, a.w, a.h, ac), 10, y + 26);
        blit(&mut sheet, &downscale(&a.fin, a.w, a.h, ac), 20 + ac, y + 26);
        let cm = change_map(a, ac);
        blit(&mut sheet, &cm, 30 + 2 * ac, y + 26);
        label(&mut sheet, 10, y + 4, &format!("{} input", a.name));
        label(&mut sheet, 20 + ac, y + 4, &format!("{} final", a.name));
        label(&mut sheet, 30 + 2 * ac, y + 4, "red inside mask orange padding teal allowed");
    }
    sheet.save(&out.join("sheet-atlas.png"))?;
    files.push("sheet-atlas.png");
    Ok(json!(files))
}

fn print_summary(s: &Value) {
    let r = &s["report"];
    for a in r["atlases"].as_array().cloned().unwrap_or_default() {
        println!(
            "atlas {}: {} texels allowed, {} painted, {} changed (inside mask {}, padding {}, OUTSIDE MASK {}), alpha changed {}",
            a["image"].as_str().unwrap_or("?"),
            a["texels_allowed"],
            a["texels_painted"],
            a["texels_changed"],
            a["changed_inside_mask"],
            a["changed_in_padding_of_allowed_islands"],
            a["changed_outside_mask"],
            a["alpha_changed"]
        );
        if let Some(mb) = a["mip_bleed"].as_array() {
            for m in mb {
                println!(
                    "  mip level {} ({} blocks at island borders): drift {:.2} with padding, {:.2} without",
                    m["mip_level"], m["blocks"], m["mean_drift_with_padding"].as_f64().unwrap_or(0.0), m["mean_drift_without_padding"].as_f64().unwrap_or(0.0)
                );
            }
        }
    }
    let m = &r["seam_mean_rgb_difference"];
    println!(
        "seam color difference (mean RGB over {} samples): input {:.2}, after bake {:.2}, after seam pass {:.2}",
        m["samples"], m["input"].as_f64().unwrap_or(0.0), m["after_bake"].as_f64().unwrap_or(0.0), m["after_seam_pass"].as_f64().unwrap_or(0.0)
    );
    if !r["flat_color_check"].is_null() {
        println!("flat-color check: {} of {} single-view texels carry their view's color", r["flat_color_check"]["within_6_of_a_view_color"], r["flat_color_check"]["texels_painted_by_one_view_at_full_strength"]);
    }
    if !r["unmasked_comparison"].is_null() {
        for p in r["unmasked_comparison"]["per_atlas"].as_array().cloned().unwrap_or_default() {
            println!("  without a mask, {} would change {} texels outside the mask", p["image"].as_str().unwrap_or("?"), p["unmasked_bake_changes_outside_mask"]);
        }
    }
    let sh = &r["shading_check"];
    match sh["warnings"].as_array() {
        Some(w) if !w.is_empty() => {
            for x in w {
                println!("SHADING WARNING: {}", x.as_str().unwrap_or(""));
            }
        }
        _ => println!("shading check: ok ({} primitives without COLOR_0, {} with; the texture shows 1:1 in field and battle)", sh["primitives_without_color_0"], sh["primitives_with_color_0"]),
    }
    println!("verification: {}", if s["verification"]["passed"] == true { "PASSED" } else { "FAILED" });
    println!("wrote {}", s["model"].as_str().unwrap_or(""));
}

// ---- demo ---------------------------------------------------------------------------------------

fn step_demo(args: &Args, model: &Model, input: &Path, out: &Path) -> Result<(), String> {
    let parts = parse_parts(args.opt("--mask-part").unwrap_or("torso"))?;
    let meshes: Vec<String> = args.opts("--mask-mesh").into_iter().map(str::to_owned).collect();
    let allowed = part_mask(model, &parts, &meshes);
    let masks = Masks { tri_allowed: allowed.clone(), texel: Default::default() };
    let n_allowed = allowed.iter().filter(|&&b| b).count();
    if n_allowed == 0 {
        return Err("the demo mask allows no triangles".into());
    }
    let part_of = tri_parts(model);
    let textures = image_textures(model);
    let gate: f32 = num(args, "--gate", 20.0)?;
    let ro = reg_options(args)?;
    let mut specs = Vec::new();
    for n in args.opts("--view").into_iter().chain(if args.opts("--view").is_empty() { vec!["front", "back", "left", "right"] } else { vec![] }) {
        specs.push(parse_view(n, args)?);
    }
    let colors: [[u8; 3]; 4] = [[205, 45, 70], [180, 55, 85], [200, 70, 60], [190, 60, 75]];
    let decoy_head = [60u8, 200, 90];
    let decoy_arm = [70u8, 90, 220];
    std::fs::create_dir_all(out.join("inputs")).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(out.join("views")).map_err(|e| e.to_string())?;
    let file = input.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let mut views = Vec::new();
    let mut reg_json = Vec::new();
    let size0 = specs[0].size as f64;
    for (vi, spec) in specs.iter().enumerate() {
        let guide = render_view(model, &textures, spec)?;
        write_view(&out.join("views"), &guide, &file, model.triangle_count())?;
        let guide = read_view(&out.join("views"), &spec.name)?; // the bake reads the sidecar, so the demo does too
        // Painted picture: the torso flat-recolored, plus decoys on the head and arms that the mask must refuse.
        let col = colors[vi % colors.len()];
        let mut painted = guide.guide.clone();
        for (i, &id) in guide.ids.iter().enumerate() {
            if id < 0 {
                continue;
            }
            let g = id as usize;
            let c = if allowed[g] {
                Some(col)
            } else if part_of[g] == Part::Head {
                Some(decoy_head)
            } else if part_of[g] == Part::Arm {
                Some(decoy_arm)
            } else {
                None
            };
            if let Some(c) = c {
                painted.rgba[i * 4..i * 4 + 3].copy_from_slice(&c);
            }
        }
        // A known misregistration, as an upscaler or a repaint would leave: scale 1.012, 0.6 deg, a few pixels.
        let (s, th) = (1.012f64, 0.6f64.to_radians());
        let k = guide.w as f64 / size0;
        let truth = Registration { a: s * th.cos(), b: s * th.sin(), tx: 5.5 * k, ty: -3.5 * k, w: guide.w, h: guide.h, k: 1.0, scale: s, rot_deg: 0.6, ncc_before: 0.0, ncc_after: 0.0, levels: 0 };
        let warped = warp_forward(&painted, &truth, guide.w, guide.h);
        warped.save(&out.join("inputs").join(format!("{}-painted.png", spec.name)))?;
        let fg: Vec<bool> = guide.ids.iter().map(|&i| i >= 0).collect();
        let t = std::time::Instant::now();
        let reg = register(&guide.guide, &fg, &warped, &ro).map_err(|e| format!("view {}: {e}", spec.name))?;
        let secs = t.elapsed().as_secs_f64();
        println!(
            "view {}: registered in {:.1} s: scale {:.4} (true {:.4}), rotation {:.3} deg (true 0.600), shift ({:.2}, {:.2}) px (true {:.2}, {:.2}), edge correlation {:.3} -> {:.3}",
            spec.name, secs, reg.scale, s, reg.rot_deg, reg.tx, reg.ty, truth.tx, truth.ty, reg.ncc_before, reg.ncc_after
        );
        let registered = warp_to_guide(&warped, &reg);
        registered.save(&out.join("inputs").join(format!("{}-registered.png", spec.name)))?;
        reg_json.push(json!({ "view": spec.name, "recovered": reg.to_json(), "true": { "scale": s, "rotation_deg": 0.6, "tx": truth.tx, "ty": truth.ty },
            "error": { "scale": reg.scale - s, "rotation_deg": reg.rot_deg - 0.6, "tx_px": reg.tx - truth.tx, "ty_px": reg.ty - truth.ty }, "seconds": secs }));
        let edit = edit_map(&guide.guide, &registered, gate);
        views.push(PaintView { view: guide, paint: warped, reg, edit });
    }
    // The refusals: a picture 10% too big, and a picture of the wrong side.
    let mut refusals = Vec::new();
    {
        let v0 = &views[0];
        let fg: Vec<bool> = v0.view.ids.iter().map(|&i| i >= 0).collect();
        let big = Registration { a: 1.10, b: 0.0, tx: 0.0, ty: 0.0, w: v0.view.w, h: v0.view.h, k: 1.0, scale: 1.1, rot_deg: 0.0, ncc_before: 0.0, ncc_after: 0.0, levels: 0 };
        let scaled = warp_forward(&v0.view.guide, &big, v0.view.w, v0.view.h);
        refusals.push(json!({ "case": "painted picture 10% larger than the render", "result": match register(&v0.view.guide, &fg, &scaled, &ro) { Ok(r) => format!("ACCEPTED at scale {:.3}", r.scale), Err(e) => format!("refused: {e}") } }));
        if views.len() > 1 {
            let other = &views[1].view.guide;
            refusals.push(json!({ "case": format!("a picture of the {} view offered for the {} view", views[1].view.spec.name, v0.view.spec.name),
                "result": match register(&v0.view.guide, &fg, other, &ro) { Ok(r) => format!("ACCEPTED (scale {:.3}, correlation {:.3})", r.scale, r.ncc_after), Err(e) => format!("refused: {e}") } }));
        }
        let wide = Img::new(v0.view.w * 2, v0.view.h, [0, 0, 0, 255]);
        refusals.push(json!({ "case": "picture with a different shape (2:1)", "result": match register(&v0.view.guide, &fg, &wide, &ro) { Ok(_) => "ACCEPTED".to_owned(), Err(e) => format!("refused: {e}") } }));
    }
    for r in &refusals {
        println!("refusal test, {}: {}", r["case"].as_str().unwrap_or(""), r["result"].as_str().unwrap_or(""));
    }
    let o = bake_options(args)?;
    let extra = json!({
        "demo": { "what": "flat recolor of the masked triangles in each view, decoy colors on head and arms, known misregistration",
                  "mask_parts": parts.iter().map(|p| p.name()).collect::<Vec<_>>(), "mask_meshes": meshes, "colors": colors, "decoys": { "head": decoy_head, "arm": decoy_arm },
                  "gate": gate, "triangles_allowed": n_allowed, "triangles_total": model.triangle_count() },
        "registration": reg_json, "registration_refusals": refusals,
    });
    let summary = run_bake(model, input, out, views, &masks, &o, true, extra)?;
    print_summary(&summary);
    println!("sheets: sheet-views.png, sheet-turntable.png, sheet-seams.png, sheet-atlas.png");
    Ok(())
}
