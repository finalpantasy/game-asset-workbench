//! `boundary-check` and `paint-fix`: label lines and specks before editing, then edit paint with
//! those labels and the overlap order checked. The work is in `wb_core::boundary`; this file only
//! parses options, renders views and writes sheets.
use crate::Args;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use wb_core::boundary::{self as bd, Kind, LabelParams, Material};
use wb_core::render::{self, Camera, Filter};
use wb_core::shells;
use wb_core::Model;

const OWN_VALUE_OPTS: [&str; 39] = [
    "--y", "--clip", "--out", "--out-dir", "--center", "--shell", "--down", "--half", "--views", "--size", "--filters", "--distance-px", "--materials", "--max-dist", "--dip", "--reach", "--probe",
    "--op", "--uv", "--of", "--on", "--max", "--search", "--upper", "--lower", "--window", "--move", "--band", "--blur", "--min-jump", "--pad", "--protect-views", "--min-mark", "--min-crease", "--keep-uv", "--mark-uv", "--seam-uv", "--reclaim", "--reclaim-tol",
];

fn files(args: &Args) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut skip = false;
    for a in &args.0 {
        if skip {
            skip = false;
            continue;
        }
        if OWN_VALUE_OPTS.contains(&a.as_str()) || crate::VALUE_OPTS.contains(&a.as_str()) {
            skip = true;
            continue;
        }
        if !a.starts_with("--") {
            out.push(PathBuf::from(a));
        }
    }
    out
}

fn num<T: std::str::FromStr>(args: &Args, k: &str, default: T) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    match args.opt(k) {
        Some(s) => s.parse::<T>().map_err(|e| format!("{k}: {e}")),
        None => Ok(default),
    }
}

fn floats(s: &str, n: usize, name: &str) -> Result<Vec<f32>, String> {
    let v: Vec<f32> = s.split(',').map(|x| x.trim().parse::<f32>().map_err(|e| format!("{name} {x}: {e}"))).collect::<Result<_, _>>()?;
    if v.len() != n {
        return Err(format!("{name} wants {n} numbers"));
    }
    Ok(v)
}

/// A named view: yaw from the front (degrees, + turns toward the model's left) and pitch.
fn view(name: &str) -> Result<(String, f32, f32), String> {
    Ok(match name {
        "front" => ("front".into(), 0.0, 0.0),
        "back" => ("back".into(), 180.0, 0.0),
        "left" => ("left side".into(), 90.0, 0.0),
        "right" => ("right side".into(), -90.0, 0.0),
        "3q-left" => ("3/4 left".into(), 40.0, 0.0),
        "3q-right" => ("3/4 right".into(), -40.0, 0.0),
        "3q-back-left" => ("3/4 back left".into(), 140.0, 0.0),
        "3q-back-right" => ("3/4 back right".into(), -140.0, 0.0),
        "under" => ("underside".into(), 0.0, -65.0),
        other => {
            let v = floats(other, 2, "view yaw:pitch").or_else(|_| floats(&other.replace(':', ","), 2, "view yaw:pitch"))?;
            (format!("yaw {} pitch {}", v[0], v[1]), v[0], v[1])
        }
    })
}

fn cam_for(center: [f32; 3], yaw: f32, pitch: f32, half: f32) -> Camera {
    let mut c = shells::ortho_cam(center, yaw, half);
    c.pitch = pitch.to_radians();
    c
}

/// View center: `--center x,y,z`, or `--shell LABEL [--down D]`: the middle of that shell's
/// vertices at D below its top.
fn center(model: &Model, args: &Args) -> Result<[f32; 3], String> {
    if let Some(c) = args.opt("--center") {
        let v = floats(c, 3, "--center")?;
        return Ok([v[0], v[1], v[2]]);
    }
    let label = args.opt("--shell").ok_or("give --center x,y,z or --shell LABEL")?;
    let sh = shells::find(model);
    let s = sh.iter().find(|s| s.label == label).ok_or_else(|| format!("no shell {label}"))?;
    let m = &model.meshes[s.mesh];
    let down: f32 = num(args, "--down", 0.0)?;
    let ytop = s.verts.iter().map(|&v| m.pos[v][1]).fold(f32::MIN, f32::max);
    // `--y Y` takes the band at an absolute height instead (for a body shell around a belt).
    let ytop = match args.opt("--y") {
        Some(y) => y.parse::<f32>().map_err(|e| format!("--y: {e}"))? + down,
        None => ytop,
    };
    let band: Vec<[f32; 3]> = s.verts.iter().map(|&v| m.pos[v]).filter(|p| (p[1] - (ytop - down)).abs() <= 0.01).collect();
    let pts = if band.is_empty() { s.verts.iter().map(|&v| m.pos[v]).collect() } else { band };
    let (lo, hi) = wb_core::geom::bounds(pts.iter());
    Ok([0.5 * (lo[0] + hi[0]), ytop - down, 0.5 * (lo[2] + hi[2])])
}

fn materials(args: &Args) -> Result<Vec<Material>, String> {
    match args.opt("--materials") {
        Some(s) => bd::parse_materials(s),
        None => Ok(Vec::new()),
    }
}

fn uv_rect(args: &Args, w: usize, h: usize) -> Result<[usize; 4], String> {
    let b = floats(args.opt("--uv").ok_or("--uv u0,v0,u1,v1 needed")?, 4, "--uv")?;
    Ok([
        (b[0].min(b[2]) * w as f32).floor().max(0.0) as usize,
        (b[1].min(b[3]) * h as f32).floor().max(0.0) as usize,
        ((b[0].max(b[2]) * w as f32).ceil() as usize).min(w),
        ((b[1].max(b[3]) * h as f32).ceil() as usize).min(h),
    ])
}

fn mat_index(pal: &[Material], name: &str) -> Result<u8, String> {
    pal.iter().position(|m| m.name == name).map(|i| i as u8).ok_or_else(|| format!("no material {name} in --materials"))
}

struct Loaded {
    path: PathBuf,
    model: Model,
    textures: Vec<Option<wb_core::Texture>>,
    shell_of: Vec<Vec<usize>>,
    nshells: usize,
}

fn load(path: &Path) -> Result<Loaded, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let model = Model::from_bytes(&bytes)?;
    let textures = wb_core::image_textures(&model);
    let sh = shells::find(&model);
    let shell_of = shells::tri_shell(&model, &sh);
    Ok(Loaded { path: path.to_owned(), model, textures, shell_of, nshells: sh.len() })
}

/// One rendered view: picture (at `size`) and, for the texel filter, its G-buffer.
/// Per (mesh, triangle): kept unless its center lies more than `clip` in front of the view
/// target (toward the camera), so an arm or a tail does not hide the part being checked.
fn keep_table(l: &Loaded, pose: &[Vec<[f32; 3]>], cam: &Camera, clip: f32) -> Vec<Vec<bool>> {
    let (_, f, _, _) = shells::basis(cam);
    l.model
        .meshes
        .iter()
        .enumerate()
        .map(|(mi, m)| {
            m.tris
                .iter()
                .map(|t| {
                    if clip <= 0.0 {
                        return true;
                    }
                    let c: [f32; 3] = std::array::from_fn(|k| t.iter().map(|&v| pose[mi][v as usize][k]).sum::<f32>() / 3.0);
                    -wb_core::geom::dot(wb_core::geom::sub(c, cam.target), f) <= clip
                })
                .collect()
        })
        .collect()
}

fn shot(l: &Loaded, pose: &[Vec<[f32; 3]>], cam: Camera, size: usize, filter: &str, distance_px: usize, clip: f32) -> Vec<u8> {
    let (f, s) = match filter {
        "smooth" => (Filter::Smooth, size),
        "distance" => (Filter::Smooth, distance_px.max(8)),
        _ => (Filter::Texels, size),
    };
    let o = render::Options { cam, w: s, h: s, background: ([28, 30, 36], [28, 30, 36]), filter: f, positions: Some(pose) };
    let keep = keep_table(l, pose, &cam, clip);
    let fr = render::render_masked(&l.model, &l.textures, &o, &|_, _| None, Some(&|m: usize, t: usize| keep[m][t]));
    if s == size {
        return fr.rgba;
    }
    // Gameplay distance: render small with mipmaps, then show each pixel as a block.
    let mut out = vec![0u8; size * size * 4];
    for y in 0..size {
        for x in 0..size {
            let (sx, sy) = (x * s / size, y * s / size);
            let a = (sy * s + sx) * 4;
            out[(y * size + x) * 4..(y * size + x) * 4 + 4].copy_from_slice(&fr.rgba[a..a + 4]);
        }
    }
    out
}

fn gbuf(l: &Loaded, pose: &[Vec<[f32; 3]>], cam: &Camera, size: usize, clip: f32) -> bd::GBuffer {
    let so = &l.shell_of;
    let keep = keep_table(l, pose, cam, clip);
    bd::gbuffer(&l.model, Some(&l.textures), pose, &|m, t| if keep[m][t] { so[m][t] } else { shells::NONE }, cam, size, size)
}

fn crossings(l: &Loaded, pose: &[Vec<[f32; 3]>], cam: &Camera, size: usize, clip: f32) -> Vec<bd::Crossing> {
    let so = &l.shell_of;
    let keep = keep_table(l, pose, cam, clip);
    let lay = shells::layers_alpha(&l.model, &l.textures, pose, &|m, t| if keep[m][t] { so[m][t] } else { shells::NONE }, cam, size, size);
    bd::crossings(&lay, 0.03, 20)
}

fn label_params<'a>(args: &Args, pal: &'a [Material]) -> Result<LabelParams<'a>, String> {
    Ok(LabelParams { dip: num(args, "--dip", 0.12)?, reach: num(args, "--reach", 3)?, layer_gap: 0.003, seam_jump: 12.0, palette: pal, max_dist: num(args, "--max-dist", 60.0)?, min_mark_px: num(args, "--min-mark", 12)?, min_crease_px: num(args, "--min-crease", 40)? })
}

/// The main texture of the model: the image used by the most triangles.
fn main_image(model: &Model) -> usize {
    let mut n = vec![0usize; model.images.len()];
    for m in &model.meshes {
        if let Some(i) = m.image {
            n[i] += m.tris.len();
        }
    }
    n.iter().enumerate().max_by_key(|e| e.1).map(|e| e.0).unwrap_or(0)
}

fn paint_labels(rgba: &mut [u8], size: usize, labels: &[bd::Labeled]) {
    for l in labels {
        let o = (l.y * size + l.x) * 4;
        rgba[o..o + 3].copy_from_slice(&l.kind.color());
    }
}

fn put(sheet: &mut image::RgbaImage, x0: u32, y0: u32, rgba: &[u8], size: usize) {
    for y in 0..size {
        for x in 0..size {
            let o = (y * size + x) * 4;
            sheet.put_pixel(x0 + x as u32, y0 + y as u32, image::Rgba([rgba[o], rgba[o + 1], rgba[o + 2], 255]));
        }
    }
}

fn text(sheet: &mut image::RgbaImage, x: i64, y: i64, s: &str, c: [u8; 3]) {
    let (w, h) = (sheet.width() as usize, sheet.height() as usize);
    wb_core::font::draw(sheet.as_mut(), w, h, x, y, s, 2, c);
}

fn legend(sheet: &mut image::RgbaImage, x: i64, y: i64) {
    let mut cx = x;
    for k in [Kind::UvSeam, Kind::ClothingEdge, Kind::Overlap, Kind::PaintMark, Kind::Silhouette, Kind::Unknown] {
        text(sheet, cx, y, k.name(), k.color());
        cx += wb_core::font::width(k.name(), 2) as i64 + 24;
    }
}

fn kind_counts(labels: &[bd::Labeled]) -> Value {
    let mut m = serde_json::Map::new();
    for k in [Kind::UvSeam, Kind::ClothingEdge, Kind::Overlap, Kind::PaintMark, Kind::Silhouette, Kind::Unknown] {
        m.insert(k.name().into(), json!(labels.iter().filter(|l| l.kind == k).count()));
    }
    Value::Object(m)
}

fn parse_views(args: &Args, default: &str) -> Result<Vec<(String, f32, f32)>, String> {
    args.opt("--views").unwrap_or(default).split(',').map(view).collect()
}

/// `boundary-check A.glb [B.glb ...] --out-dir DIR (--center x,y,z | --shell LABEL [--down D])
/// [--half H] [--views front,back,...] [--size PX] [--filters texels,smooth,distance]
/// [--distance-px N] [--materials name=r,g,b;...] [--probe x,y]`
pub fn check_cmd(args: &Args) -> Result<(), String> {
    let paths = files(args);
    if paths.is_empty() {
        return Err("boundary-check needs one or more .glb files".into());
    }
    let out_dir = PathBuf::from(args.opt("--out-dir").ok_or("boundary-check needs --out-dir DIR")?);
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let size: usize = num(args, "--size", 320)?;
    let half: f32 = num(args, "--half", 0.12)?;
    let clip: f32 = num(args, "--clip", 0.0)?;
    let dist_px: usize = num(args, "--distance-px", 40)?;
    let filters: Vec<String> = args.opt("--filters").unwrap_or("texels").split(',').map(str::to_owned).collect();
    let views = parse_views(args, "front,3q-right,right,back,left,3q-left")?;
    let pal = materials(args)?;
    let lp = label_params(args, &pal)?;
    let models: Vec<Loaded> = paths.iter().map(|p| load(p)).collect::<Result<_, _>>()?;
    let c = center(&models[0].model, args)?;
    println!("center {:.4},{:.4},{:.4}, half {half}, {} views, {} filters", c[0], c[1], c[2], views.len(), filters.len());
    // Rows: per model, per filter a plain row, then a label row (texel filter only).
    let rows: usize = models.len() * (filters.len() + 1);
    let (lw, th) = (300u32, 28u32);
    let mut sheet = image::RgbaImage::from_pixel(lw + (views.len() * (size + 4)) as u32, th * 2 + (rows * (size + 4)) as u32, image::Rgba([18, 19, 24, 255]));
    for (vi, v) in views.iter().enumerate() {
        text(&mut sheet, lw as i64 + (vi * (size + 4)) as i64 + 4, 4, &v.0.to_uppercase(), [220, 220, 220]);
    }
    legend(&mut sheet, 8, th as i64 + 2);
    let mut report = Vec::new();
    let mut row = 0usize;
    for l in &models {
        let pose = l.model.posed(None, 0.0);
        let image = main_image(&l.model);
        let tex = l.textures[image].as_ref().ok_or("main texture did not decode")?;
        let name = l.path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let mut per_view = Vec::new();
        for (fi, f) in filters.iter().enumerate() {
            let y0 = th * 2 + ((row + fi) * (size + 4)) as u32;
            text(&mut sheet, 8, y0 as i64 + 8, &name.chars().take(22).collect::<String>(), [230, 230, 230]);
            text(&mut sheet, 8, y0 as i64 + 30, f, [160, 160, 160]);
            for (vi, v) in views.iter().enumerate() {
                let cam = cam_for(c, v.1, v.2, half);
                let pic = shot(l, &pose, cam, size, f, dist_px, clip);
                put(&mut sheet, lw + (vi * (size + 4)) as u32, y0, &pic, size);
            }
        }
        let yl = th * 2 + ((row + filters.len()) * (size + 4)) as u32;
        text(&mut sheet, 8, yl as i64 + 8, &name.chars().take(22).collect::<String>(), [230, 230, 230]);
        text(&mut sheet, 8, yl as i64 + 30, "labels", [160, 160, 160]);
        for (vi, v) in views.iter().enumerate() {
            let cam = cam_for(c, v.1, v.2, half);
            let mut pic = shot(l, &pose, cam, size, "texels", dist_px, clip);
            let g = gbuf(l, &pose, &cam, size, clip);
            let labels = bd::label_lines(&l.model, tex, image, &g, &pic, &lp);
            let cross = crossings(l, &pose, &cam, size, clip);
            if vi == 0 {
                if let Some(p) = args.opt("--probe") {
                    let q = floats(p, 2, "--probe")?;
                    probe(l, tex, &g, q[0] as i64, q[1] as i64, &pal);
                }
            }
            paint_labels(&mut pic, size, &labels);
            put(&mut sheet, lw + (vi * (size + 4)) as u32, yl, &pic, size);
            let sum: Vec<Value> = bd::summarize(&labels)
                .into_iter()
                .take(25)
                .map(|((m, t), k, n)| {
                    let ex = labels.iter().find(|l| l.hit.mesh == m && l.hit.tri == t && l.kind == k).unwrap();
                    json!({"mesh": m, "tri": t, "shell": ex.hit.shell, "kind": k.name(), "pixels": n, "why": ex.why, "texel": [ex.texel.0, ex.texel.1], "uv": ex.hit.uv, "light": ex.light})
                })
                .collect();
            println!("{name} {}: {}", v.0, kind_counts(&labels));
            for s in sum.iter().take(8) {
                println!("    {s}");
            }
            per_view.push(json!({
                "view": v.0, "counts": kind_counts(&labels), "by_triangle": sum,
                "crossings": cross.iter().map(|c| json!({"over": c.over, "under": c.under, "pixels": c.pixels})).collect::<Vec<_>>(),
            }));
        }
        row += filters.len() + 1;
        report.push(json!({"file": l.path.display().to_string(), "shells": l.nshells, "views": per_view}));
    }
    let sp = out_dir.join("boundary-sheet.png");
    sheet.save(&sp).map_err(|e| e.to_string())?;
    let jp = out_dir.join("boundary-check.json");
    std::fs::write(&jp, serde_json::to_string_pretty(&json!({"center": c, "half": half, "size": size, "files": report})).unwrap()).map_err(|e| e.to_string())?;
    println!("wrote {}\nwrote {}", sp.display(), jp.display());
    Ok(())
}

fn probe(l: &Loaded, tex: &wb_core::Texture, g: &bd::GBuffer, x: i64, y: i64, pal: &[Material]) {
    for dy in -1..=1 {
        let mut line = String::new();
        for dx in -1..=1 {
            match g.at(x + dx, y + dy) {
                Some(h) => {
                    let (tx, ty) = bd::texel_of(tex.w, tex.h, h.uv);
                    let o = (ty * tex.w + tx) * 4;
                    let px = &tex.rgba[o..o + 4];
                    let mm = bd::material_mask(&tex.rgba, tex.w, [tx, ty, tx + 1, ty + 1], pal, 60.0)[0];
                    let mn = pal.get(mm as usize).map_or("?", |m| m.name.as_str());
                    line += &format!(" [{},{}] m{} t{} s{} uv {:.4},{:.4} texel {},{} rgb {},{},{} {mn} |", x + dx, y + dy, h.mesh, h.tri, h.shell, h.uv[0], h.uv[1], tx, ty, px[0], px[1], px[2]);
                }
                None => line += &format!(" [{},{}] background |", x + dx, y + dy),
            }
        }
        println!("probe{line}");
    }
    let _ = l;
}

/// `u0,v0,u1,v1` to a texel box [x0, y0, x1, y1) covering every texel the UV box touches.
fn uv_box(s: &str, name: &str, w: usize, h: usize) -> Result<[usize; 4], String> {
    let f = floats(s, 4, name)?;
    let (x0, y0) = ((f[0].min(f[2]) * w as f32) as usize, (f[1].min(f[3]) * h as f32) as usize);
    let (x1, y1) = (((f[0].max(f[2]) * w as f32).ceil() as usize).min(w), ((f[1].max(f[3]) * h as f32).ceil() as usize).min(h));
    Ok([x0, y0, x1, y1])
}

/// Label pixels of the source in the protect views and return the texels under the given kinds.
fn protect(l: &Loaded, pose: &[Vec<[f32; 3]>], c: [f32; 3], views: &[(String, f32, f32)], half: f32, size: usize, clip: f32, lp: &LabelParams, image: usize, kinds: &[Kind]) -> (HashSet<(usize, usize)>, Vec<Vec<bd::Labeled>>) {
    let tex = l.textures[image].as_ref().unwrap();
    let mut set = HashSet::new();
    let mut all = Vec::new();
    for v in views {
        let cam = cam_for(c, v.1, v.2, half);
        let pic = shot(l, pose, cam, size, "texels", 40, clip);
        let g = gbuf(l, pose, &cam, size, clip);
        let labels = bd::label_lines(&l.model, tex, image, &g, &pic, lp);
        set.extend(bd::protected_texels(&labels, kinds, 1));
        all.push(labels);
    }
    (set, all)
}

/// `paint-fix FILE.glb --out NEW.glb --op marks|edge|seams|pad --uv u0,v0,u1,v1 --materials ...
/// (--center x,y,z | --shell LABEL [--down D]) [--protect-views front,...] [--half H] [--size PX]
/// [--keep-uv u0,v0,u1,v1;...]`
/// plus per op: marks `--on MAT (--of MAT [--max N] | --mark-uv u0,v0,u1,v1;...) [--search S]`; edge `--upper MAT --lower MAT
/// [--window N] [--move R]`; seams `[--seam-uv box;box] [--band B] [--blur K] [--min-jump T]`; pad `--pad N [--reclaim N]`.
/// Writes NEW.glb (only the texture changes), NEW.sheet.png (before, after, changed pixels in
/// red, intentional-overlap pixels in green) and NEW.json.
pub fn fix_cmd(args: &Args) -> Result<(), String> {
    let path = files(args).into_iter().next().ok_or("paint-fix needs FILE.glb")?;
    let out = PathBuf::from(args.opt("--out").ok_or("paint-fix needs --out NEW.glb")?);
    if out == path {
        return Err("--out must differ from the source".into());
    }
    let op = args.opt("--op").ok_or("paint-fix needs --op marks|edge|seams|pad")?.to_owned();
    let l = load(&path)?;
    let pal = materials(args)?;
    let lp = label_params(args, &pal)?;
    let image = main_image(&l.model);
    let tex = l.textures[image].as_ref().ok_or("texture did not decode")?;
    let (w, h) = (tex.w, tex.h);
    let rect = uv_rect(args, w, h)?;
    let size: usize = num(args, "--size", 320)?;
    let half: f32 = num(args, "--half", 0.12)?;
    let clip: f32 = num(args, "--clip", 0.0)?;
    let c = center(&l.model, args)?;
    let views = args.opt("--protect-views").unwrap_or("front,3q-right,right,back,left,3q-left").split(',').map(view).collect::<Result<Vec<_>, _>>()?;
    let pose = l.model.posed(None, 0.0);
    // Texels under intentional overlaps are never edited; marks and seams also keep clothing edges.
    let kinds: Vec<Kind> = match op.as_str() {
        "edge" => vec![Kind::Overlap],
        _ => vec![Kind::Overlap, Kind::ClothingEdge],
    };
    let (mut protected, labels_before) = protect(&l, &pose, c, &views, half, size, clip, &lp, image, &kinds);
    // Hand-marked keep boxes (`--keep-uv u0,v0,u1,v1;...`): texels an edit must not touch.
    let mut kept = 0;
    for b in args.opt("--keep-uv").unwrap_or("").split(';').filter(|s| !s.trim().is_empty()) {
        let [x0, y0, x1, y1] = uv_box(b, "--keep-uv", w, h)?;
        for y in y0..y1 {
            for x in x0..x1 {
                protected.insert((x, y));
                kept += 1;
            }
        }
    }
    if kept > 0 {
        println!("keep boxes: {kept} texels protected by hand");
    }
    let mut rgba = tex.rgba.clone();
    let mask = bd::material_mask(&rgba, w, rect, &pal, lp.max_dist);
    let detail: Value;
    let changed: usize;
    match op.as_str() {
        "marks" => {
            let on = mat_index(&pal, args.opt("--on").ok_or("--on MAT")?)?;
            // Hand-marked boxes (`--mark-uv u0,v0,u1,v1;...`, for example a selection from the
            // window UI) are filled whole; otherwise the detector finds `--of` blobs on `--on`.
            let marks = match args.opt("--mark-uv") {
                Some(spec) => {
                    let mut v = Vec::new();
                    for b in spec.split(';').filter(|s| !s.trim().is_empty()) {
                        let [bx0, by0, bx1, by1] = uv_box(b, "--mark-uv", w, h)?;
                        let texels: Vec<(usize, usize)> = (by0..by1).flat_map(|y| (bx0..bx1).map(move |x| (x, y))).collect();
                        v.push(bd::Mark { material: bd::UNKNOWN, host: on, texels, host_share: 0.0, bbox: [bx0, by0, bx1, by1] });
                    }
                    v
                }
                None => {
                    let of = mat_index(&pal, args.opt("--of").ok_or("--of MAT (or --mark-uv)")?)?;
                    bd::find_marks(&mask, rect, of, on, num(args, "--max", 40)?, 0.6)
                }
            };
            let (n, used) = bd::fill_marks(&mut rgba, w, &mask, rect, &marks, num(args, "--search", 12)?, &protected);
            for (mk, u) in marks.iter().zip(&used) {
                println!("mark: {} texels at {:?}, ring {:.0}% {}, filled from offset {:?}", mk.texels.len(), mk.bbox, 100.0 * mk.host_share, pal[on as usize].name, u);
            }
            changed = n;
            detail = json!({"marks": marks.iter().zip(&used).map(|(m, u)| json!({"texels": m.texels.len(), "bbox": m.bbox, "host_share": m.host_share, "offset": u})).collect::<Vec<_>>()});
        }
        "edge" => {
            let up = mat_index(&pal, args.opt("--upper").ok_or("--upper MAT")?)?;
            let lo = mat_index(&pal, args.opt("--lower").ok_or("--lower MAT")?)?;
            let prof = bd::edge_profile(&mask, rect, up, lo, num(args, "--window", 15)?);
            changed = bd::smooth_edge(&mut rgba, w, &mask, rect, &prof, up, lo, num(args, "--move", 6.0)?, &protected);
            let found = prof.rows.iter().filter(|r| r.is_some()).count();
            println!("edge: found in {found} of {} columns; largest column-to-column step {:.1} -> {:.1} texels", prof.rows.len(), prof.jag_before, prof.jag_after);
            detail = json!({"columns": prof.rows.len(), "found": found, "jag_before": prof.jag_before, "jag_after": prof.jag_after});
        }
        "seams" => {
            let min_jump: f32 = num(args, "--min-jump", 4.0)?;
            // Islands joined by a seam can sit far apart on the texture: `--seam-uv box;box` lists
            // where each side's triangles are (default: the --uv box).
            let boxes: Vec<[usize; 4]> = match args.opt("--seam-uv") {
                Some(spec) => spec.split(';').filter(|s| !s.trim().is_empty()).map(|b| uv_box(b, "--seam-uv", w, h)).collect::<Result<_, _>>()?,
                None => vec![rect],
            };
            let mut seams = Vec::new();
            for (mi, m) in l.model.meshes.iter().enumerate() {
                if m.image != Some(image) {
                    continue;
                }
                let inside = |t: usize| {
                    let cu = m.tris[t].iter().map(|&v| m.uv[v as usize][0]).sum::<f32>() / 3.0 * w as f32;
                    let cv = m.tris[t].iter().map(|&v| m.uv[v as usize][1]).sum::<f32>() / 3.0 * h as f32;
                    boxes.iter().any(|r| cu >= r[0] as f32 && cu < r[2] as f32 && cv >= r[1] as f32 && cv < r[3] as f32)
                };
                seams.extend(bd::uv_seams(&l.model, mi, w, h, min_jump, &inside));
            }
            println!("seams: {} edges with both sides' triangles in the box", seams.len());
            let r = bd::match_seams(&mut rgba, w, h, &seams, num(args, "--band", 10.0)?, num(args, "--blur", 2)?, 12, &protected);
            println!(
                "seams: color step across the seam {:.1} -> {:.1} (RGB); contrast {:.3}/{:.3} -> {:.3}/{:.3}; grain scale {:.1}/{:.1} texels; stroke directions {:.0} deg apart on the texture",
                r.step_before, r.step_after, r.before[0].contrast, r.before[1].contrast, r.after[0].contrast, r.after[1].contrast, r.before[0].scale, r.before[1].scale, r.direction_gap_deg
            );
            changed = r.changed;
            detail = json!({"edges": r.edges, "step_before": r.step_before, "step_after": r.step_after,
                "contrast_before": [r.before[0].contrast, r.before[1].contrast], "contrast_after": [r.after[0].contrast, r.after[1].contrast],
                "grain_scale_before": [r.before[0].scale, r.before[1].scale], "angle_before": [r.before[0].angle_deg, r.before[1].angle_deg], "direction_gap_deg": r.direction_gap_deg});
        }
        "pad" => {
            let mut cov = wb_core::strapcheck::uv_coverage(&l.model, image, w, h, rect, 0.0);
            // `--reclaim N`: first un-cover up to N edge texels that do not match the paint inside them.
            let reclaimed = match num(args, "--reclaim", 0usize)? {
                0 => 0,
                n => bd::reclaim_gutter_paint(&rgba, w, rect, &mut cov, num(args, "--reclaim-tol", 24u8)?, n),
            };
            if reclaimed > 0 {
                println!("pad: {reclaimed} sampled edge texels do not match the paint inside them; repainting them from the interior");
            }
            changed = bd::pad_by_material(&mut rgba, w, rect, &cov, &mask, num(args, "--pad", 8)?);
            detail = json!({"sampled": cov.iter().filter(|&&c| c).count(), "rect_texels": cov.len(), "reclaimed": reclaimed});
        }
        other => return Err(format!("unknown --op {other}")),
    }
    // Protected texels must be untouched whatever the op did.
    let broke = protected.iter().filter(|&&(x, y)| x < w && y < h && rgba[(y * w + x) * 4..(y * w + x) * 4 + 3] != tex.rgba[(y * w + x) * 4..(y * w + x) * 4 + 3]).count();
    println!("{}: {op}: {changed} texels changed in {:?}; {} protected texels, {broke} of them changed", path.display(), rect, protected.len());
    if broke > 0 {
        return Err("an edit touched protected texels; nothing written".into());
    }
    if changed == 0 {
        return Err("nothing changed; nothing written".into());
    }
    let mut png = std::io::Cursor::new(Vec::new());
    image::RgbaImage::from_raw(w as u32, h as u32, rgba).ok_or("bad image size")?.write_to(&mut png, image::ImageFormat::Png).map_err(|e| e.to_string())?;
    let sw = wb_core::swap::swap(&l.model, &l.model.images[image].name, &png.into_inner())?;
    std::fs::write(&out, &sw.bytes).map_err(|e| format!("{}: {e}", out.display()))?;
    println!("check: only image {} changed (other buffer views and JSON verified by swap)", sw.image);
    // After: render the same views, compare, and check overlap pixels and crossing order.
    let l2 = load(&out)?;
    let tex2 = l2.textures[image].as_ref().unwrap();
    let pose2 = l2.model.posed(None, 0.0);
    let (lw, th) = (220u32, 30u32);
    let mut sheet = image::RgbaImage::from_pixel(lw + (views.len() * (size + 4)) as u32, th + (3 * (size + 4)) as u32, image::Rgba([18, 19, 24, 255]));
    let mut per_view = Vec::new();
    for (vi, v) in views.iter().enumerate() {
        let cam = cam_for(c, v.1, v.2, half);
        let a = shot(&l, &pose, cam, size, "texels", 40, clip);
        let b = shot(&l2, &pose2, cam, size, "texels", 40, clip);
        let g2 = gbuf(&l2, &pose2, &cam, size, clip);
        let labels_after = bd::label_lines(&l2.model, tex2, image, &g2, &b, &lp);
        let overlap_px: HashSet<(usize, usize)> = labels_before[vi].iter().filter(|l| l.kind == Kind::Overlap).map(|l| (l.x, l.y)).collect();
        let mut diff = b.clone();
        let (mut nchg, mut nover) = (0, 0);
        for y in 0..size {
            for x in 0..size {
                let o = (y * size + x) * 4;
                let ch = a[o..o + 3] != b[o..o + 3];
                if ch {
                    nchg += 1;
                    diff[o..o + 3].copy_from_slice(&[255, 40, 40]);
                    if overlap_px.contains(&(x, y)) {
                        nover += 1;
                    }
                } else if overlap_px.contains(&(x, y)) {
                    diff[o..o + 3].copy_from_slice(&[80, 255, 80]);
                }
            }
        }
        let flips = bd::order_changes(&crossings(&l, &pose, &cam, size, clip), &crossings(&l2, &pose2, &cam, size, clip));
        println!("{}: {nchg} screen pixels changed, {nover} of them on intentional-overlap pixels; overlap order flips {}; labels before {} after {}", v.0, flips.len(), kind_counts(&labels_before[vi]), kind_counts(&labels_after));
        let x0 = lw + (vi * (size + 4)) as u32;
        text(&mut sheet, x0 as i64 + 4, 4, &v.0.to_uppercase(), [220, 220, 220]);
        put(&mut sheet, x0, th, &a, size);
        put(&mut sheet, x0, th + (size + 4) as u32, &b, size);
        put(&mut sheet, x0, th + 2 * (size + 4) as u32, &diff, size);
        per_view.push(json!({"view": v.0, "screen_pixels_changed": nchg, "changed_on_overlap": nover, "order_flips": flips.len(),
            "labels_before": kind_counts(&labels_before[vi]), "labels_after": kind_counts(&labels_after)}));
    }
    for (r, s) in ["before", "after", "changed (red)", ""].iter().enumerate().take(3) {
        text(&mut sheet, 8, th as i64 + (r * (size + 4)) as i64 + 8, s, [230, 230, 230]);
    }
    text(&mut sheet, 8, th as i64 + 2 * (size + 4) as i64 + 34, "overlap (green)", [80, 255, 80]);
    let base = out.with_extension("");
    let sp = PathBuf::from(format!("{}.sheet.png", base.display()));
    sheet.save(&sp).map_err(|e| e.to_string())?;
    let jp = PathBuf::from(format!("{}.json", base.display()));
    std::fs::write(&jp, serde_json::to_string_pretty(&json!({"source": path.display().to_string(), "op": op, "uv_rect": rect, "texels_changed": changed,
        "protected_texels": protected.len(), "detail": detail, "views": per_view})).unwrap()).map_err(|e| e.to_string())?;
    println!("wrote {} ({} bytes)\nwrote {}\nwrote {}", out.display(), sw.bytes.len(), sp.display(), jp.display());
    Ok(())
}
