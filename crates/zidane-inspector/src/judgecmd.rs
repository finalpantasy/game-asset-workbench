//! `gaw judge`: perceptual checks (FLIP, SSIM) on two images or two GLBs, golden
//! renders, and regression runs against them. The math is in `wb_core::perceptual`, the fixed view
//! set in `wb_core::golden`. Nothing here touches an input file or the game folder.
use crate::{Args, KIT_PARTS};
use image::RgbaImage;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use wb_core::analysis::Parts;
use wb_core::golden::{self, Kind, PartGroup, View};
use wb_core::strands;
use wb_core::perceptual::{self as pc, Rgb, Stats};
use wb_core::{image_textures, Model};

pub const JUDGE_USAGE: &str = "  gaw judge REF TEST [options]
      Perceptual check of TEST against REF: two PNG images, or two GLBs drawn from the same
      cameras (the standard view set: idle, run 0.75 s, walk 0.25 s; game size and close-ups).
      Prints mean / 95th percentile FLIP (LDR-FLIP, NVIDIA, BSD-3), SSIM, and a score per body
      part or bone; writes heat maps and review sheets with --out-dir. FLIP runs 0 (same) to 1.
      --shading-tol F          GLB pairs: warn when a texture's mean brightness moves more than F (default 0.15); vertex-color
                               differences always warn
      --out-dir DIR            write judge.json, sheet-*.png (reference | test | error) and heat maps
      --views a,b,c            only these views (GLB mode; names as in goldens manifest.json)
      --manifest FILE          use the cameras of a goldens manifest instead of fitting to REF
      --ppd N                  pixels per degree (default 67.02, the FLIP reference viewing setup)
      --heat-max X             error drawn as the brightest color (default 0.5)
      --parts DIR              kit part maps for region scores (default: the UV kit; else bones)
      --bg r,g,b               image mode: background color (default: REF's top-left pixel)
      --only a,b               regions the change is meant for (kit part or bone group names): the
                               verdict then rests on the worst region OUTSIDE them (collateral change)
      --heat-all               also write one heat map per large view and close-up (heat/)
      --warn X --fail X        verdict limits on the score (see REVIEW.md in experiments/goldens)
      --json                   print the result as JSON
  gaw judge --make-goldens MODEL.glb --name NAME --out-dir DIR [--force]
      Draws the standard view set of MODEL into DIR/NAME with a manifest.json (cameras, sizes,
      source hash). Never overwrites without --force.
  gaw judge --golden-dir DIR/NAME CANDIDATE.glb [--out-dir D2]
      Draws CANDIDATE from the stored cameras and scores it against the stored pictures.
  gaw judge --strands MODEL.glb [--ref STOCK.glb] [--out-dir DIR] [--json]
      No-reference hair check: measures the direction the painted strands run (structure tensor on
      the hair texture), maps it onto the surface, and scores how well it continues across every UV
      seam (against the same measure on edges inside a chart) and how well it follows each lock's
      long axis. Hair triangles are the head-bone triangles painted blond. --ref runs the same
      check on a reference model for comparison. --out-dir writes strands.json and a seam map per
      texture (green seams continue, red ones break).";

const VALUE_OPTS: [&str; 16] = ["--strands", "--ref", "--only", "--out-dir", "--views", "--manifest", "--ppd", "--heat-max", "--parts", "--bg", "--warn", "--fail", "--make-goldens", "--name", "--golden-dir", "--ids"];

struct A<'a>(&'a [String]);

impl A<'_> {
    fn opt(&self, k: &str) -> Option<&str> {
        self.0.iter().position(|a| a == k).and_then(|i| self.0.get(i + 1)).map(String::as_str)
    }
    fn flag(&self, k: &str) -> bool {
        self.0.iter().any(|a| a == k)
    }
    fn positionals(&self) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut skip = false;
        for a in self.0 {
            if skip {
                skip = false;
                continue;
            }
            if VALUE_OPTS.contains(&a.as_str()) {
                skip = true;
            } else if !a.starts_with("--") {
                out.push(PathBuf::from(a));
            }
        }
        out
    }
    fn num(&self, k: &str, d: f32) -> Result<f32, String> {
        match self.opt(k) {
            Some(s) => s.parse().map_err(|_| format!("{k}: cannot read {s}")),
            None => Ok(d),
        }
    }
}

/// Output goes under Modding or elsewhere outside the game; never F: and never the installed game.
fn check_out_dir(p: &Path) -> Result<(), String> {
    let s = p.to_string_lossy().replace('/', "\\").to_ascii_lowercase();
    if s.starts_with("f:") {
        return Err("refusing to write to F:".into());
    }
    if s.contains("final fantasy ix") && !s.contains("\\modding") {
        return Err("refusing to write into the installed game folder".into());
    }
    Ok(())
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

fn load_png(path: &Path) -> Result<Rgb, String> {
    let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.to_rgba8();
    Ok(Rgb::from_rgba8(img.width() as usize, img.height() as usize, img.as_raw(), golden::BACKGROUND))
}

fn save_rgb(path: &Path, im: &Rgb) -> Result<(), String> {
    RgbaImage::from_raw(im.w as u32, im.h as u32, im.to_rgba8()).ok_or("bad image size")?.save(path).map_err(|e| format!("{}: {e}", path.display()))
}

// ---------- regions ----------

/// Region name for every triangle, by global triangle id (see `render::tri_offsets`).
struct Labels {
    names: Vec<String>,
    tri: Vec<u16>,
    shape: Vec<usize>,
    source: &'static str,
}

pub(crate) fn bone_group(bone: &str) -> String {
    let b = bone.to_ascii_lowercase();
    let side = if b.contains("_l_") { "_L" } else if b.contains("_r_") { "_R" } else { "" };
    let g = if b.contains("head") || b.contains("neck") || b.contains("crest") {
        "head"
    } else if b.contains("hand") {
        "hand"
    } else if b.contains("forearm") || b.contains("upper_arm") || b.contains("shoulder") {
        "arm"
    } else if b.contains("thigh") || b.contains("shin") || b.contains("_hip") && !b.contains("hips") {
        "leg"
    } else if b.contains("foot") {
        "foot"
    } else if b.contains("tail") {
        "tail"
    } else {
        "torso"
    };
    format!("{g}{side}")
}

/// "battle_model0" is the weapon mesh of an overlay export; give it its own region.
fn weapon_name(mesh: &str) -> Option<String> {
    mesh.strip_prefix("battle_model").map(|s| format!("weapon{}", s.split('#').next().unwrap_or("")))
}

fn labels_for(model: &Model, parts: Option<&Parts>) -> Labels {
    let shape: Vec<usize> = model.meshes.iter().map(|m| m.tris.len()).collect();
    let kit_ok = parts.is_some_and(|p| (0..model.meshes.len()).all(|mi| model.meshes[mi].image.is_none() || p.map_for(&model.atlas_name(mi)).is_some()));
    let mut names: Vec<String> = Vec::new();
    let mut tri: Vec<u16> = Vec::with_capacity(shape.iter().sum());
    let id_of = |n: String, names: &mut Vec<String>| -> u16 {
        match names.iter().position(|x| *x == n) {
            Some(i) => i as u16,
            None => {
                names.push(n);
                (names.len() - 1) as u16
            }
        }
    };
    for mi in 0..model.meshes.len() {
        if kit_ok {
            let kp = parts.unwrap().triangle_parts(model, mi);
            let weapon = weapon_name(&model.meshes[mi].name);
            for &p in &kp {
                let n = weapon.clone().unwrap_or_else(|| parts.unwrap().name(p).to_owned());
                tri.push(id_of(n, &mut names));
            }
        } else {
            let weapon = weapon_name(&model.meshes[mi].name);
            for ti in 0..model.meshes[mi].tris.len() {
                let j = model.triangle_joint(mi, ti);
                let n = weapon.clone().unwrap_or_else(|| model.joint_names.get(j).map(|s| bone_group(s)).unwrap_or_else(|| "torso".into()));
                tri.push(id_of(n, &mut names));
            }
        }
    }
    Labels { names, tri, shape, source: if kit_ok { "kit parts" } else { "bone groups" } }
}

fn model_shape(m: &Model) -> Vec<usize> {
    m.meshes.iter().map(|x| x.tris.len()).collect()
}

/// Small parts that get their own close-ups: each weapon mesh, and (with kit labels) every kit part
/// covering at most 5% of the surface and at least 8 triangles, smallest six.
fn part_groups(model: &Model, labels: &Labels) -> Vec<PartGroup> {
    let offs = wb_core::render::tri_offsets(model);
    let mut groups: Vec<PartGroup> = Vec::new();
    for (mi, m) in model.meshes.iter().enumerate() {
        if let Some(n) = weapon_name(&m.name) {
            groups.push(PartGroup { name: n, tris: (0..m.tris.len()).map(|t| (mi, t)).collect() });
        }
    }
    if labels.source == "kit parts" {
        let mut by: BTreeMap<u16, (f64, Vec<(usize, usize)>)> = BTreeMap::new();
        let mut total = 0.0f64;
        for (mi, m) in model.meshes.iter().enumerate() {
            for (ti, t) in m.tris.iter().enumerate() {
                let p: [[f64; 3]; 3] = std::array::from_fn(|k| {
                    let v = m.pos[t[k] as usize];
                    [v[0] as f64, v[1] as f64, v[2] as f64]
                });
                let (u, v) = ([p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]], [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]]);
                let c = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
                let area = 0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                total += area;
                let id = labels.tri[offs[mi] + ti];
                let name = &labels.names[id as usize];
                if name == "unassigned" || name.starts_with("weapon") {
                    continue;
                }
                let e = by.entry(id).or_insert((0.0, Vec::new()));
                e.0 += area;
                e.1.push((mi, ti));
            }
        }
        let mut small: Vec<(f64, u16)> = by.iter().filter(|(_, v)| v.0 <= 0.05 * total && v.1.len() >= 8).map(|(k, v)| (v.0, *k)).collect();
        small.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, id) in small.into_iter().take(6) {
            groups.push(PartGroup { name: labels.names[id as usize].clone(), tris: by[&id].1.clone() });
        }
    }
    groups
}

fn all_views(model: &Model, labels: &Labels) -> Vec<View> {
    let mut v = golden::standard_views(model);
    v.extend(golden::part_views(model, &part_groups(model, labels)));
    v
}

// ---------- scoring ----------

struct ViewScore {
    name: String,
    kind: Kind,
    flip: Stats,
    flip_frame_mean: f32,
    ssim: f32,
    coverage: f32,
}

/// Region name -> every FLIP value of its pixels (pooled over all frames).
type RegionVals = BTreeMap<String, Vec<f32>>;

/// A region counts in a view only when it shows at least this many pixels (a sliver at a view's edge is noise).
const FRAME_MIN_PX: usize = 1500;

/// Region name -> (view, pixels, mean FLIP) for every view where the region shows FRAME_MIN_PX pixels or more.
/// The pooled mean can hide a small part that is wrong in one close-up; this does not.
type RegionFrames = BTreeMap<String, Vec<(String, usize, f32)>>;

struct Out<'a> {
    dir: Option<&'a Path>,
    heat_max: f32,
    ppd: f32,
}

fn score_frame(name: &str, kind: Kind, a: &Rgb, b: &Rgb, mask: &[bool], region: Option<&[Option<u16>]>, names: &[String], regions: &mut RegionVals, frames: &mut RegionFrames, out: &Out, sheet: &mut Vec<(Kind, RgbaImage)>) -> Result<ViewScore, String> {
    let f = pc::flip(a, b, out.ppd)?;
    let s = pc::ssim_map(a, b)?;
    let flip = pc::masked_stats(&f.err, Some(mask));
    let ssim = pc::masked_stats(&s, Some(mask)).mean;
    if let Some(reg) = region {
        let mut acc: BTreeMap<u16, (f64, usize)> = BTreeMap::new();
        for (i, r) in reg.iter().enumerate() {
            if let Some(r) = r {
                if !name.starts_with("part_") {
                    regions.entry(names[*r as usize].clone()).or_default().push(f.err[i]);
                }
                let e = acc.entry(*r).or_insert((0.0, 0));
                e.0 += f.err[i] as f64;
                e.1 += 1;
            }
        }
        for (r, (sum, n)) in acc {
            if n >= FRAME_MIN_PX {
                frames.entry(names[r as usize].clone()).or_default().push((format!("{name}__{}", kind.tag()), n, (sum / n as f64) as f32));
            }
        }
    }
    let flip_frame_mean = pc::stats(f.err.clone()).mean;
    let coverage = mask.iter().filter(|&&m| m).count() as f32 / mask.len() as f32;
    if out.dir.is_some() {
        sheet.push((kind, triptych(a, b, &f.err, mask, out.heat_max, &format!("{name} {:.4}", flip.mean))));
    }
    Ok(ViewScore { name: name.to_owned(), kind, flip, flip_frame_mean, ssim, coverage })
}

fn heat_rgb(err: &[f32], mask: &[bool], w: usize, h: usize, full: f32) -> Rgb {
    let px = (0..w * h).map(|i| if mask[i] { pc::magma(err[i] / full).map(|v| v as f32 / 255.0) } else { [0.08, 0.08, 0.09] }).collect();
    Rgb { w, h, px }
}

/// Reference | test | error heat map, side by side, with a label.
fn triptych(a: &Rgb, b: &Rgb, err: &[f32], mask: &[bool], heat_max: f32, label: &str) -> RgbaImage {
    let (w, h) = (a.w, a.h);
    let heat = heat_rgb(err, mask, w, h, heat_max);
    let mut img = RgbaImage::new((w * 3) as u32, h as u32);
    for (k, im) in [a, b, &heat].iter().enumerate() {
        let rgba = im.to_rgba8();
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 4;
                img.put_pixel((k * w + x) as u32, y as u32, image::Rgba([rgba[o], rgba[o + 1], rgba[o + 2], 255]));
            }
        }
    }
    let scale = (w / 220).max(1);
    let (iw, ih) = (img.width() as usize, img.height() as usize);
    wb_core::font::draw(img.as_mut(), iw, ih, 4, 4, &label.replace('_', "-"), scale, [0, 0, 0]);
    img
}

fn write_sheets(dir: &Path, sheets: &[(Kind, RgbaImage)]) -> Result<(), String> {
    for (kind, file, thumb_w) in [(Kind::Game, "sheet-game.png", 960usize), (Kind::Close, "sheet-close.png", 1152), (Kind::Large, "sheet-large.png", 1152)] {
        let items: Vec<&RgbaImage> = sheets.iter().filter(|s| s.0 == kind).map(|s| &s.1).collect();
        if items.is_empty() {
            continue;
        }
        let th = (items[0].height() as usize * thumb_w / items[0].width() as usize).max(1);
        let cols = 2;
        let rows = items.len().div_ceil(cols);
        let mut sheet = RgbaImage::from_pixel((cols * thumb_w) as u32, (rows * th) as u32, image::Rgba([20, 20, 24, 255]));
        for (i, im) in items.iter().enumerate() {
            let t = image::imageops::resize(*im, thumb_w as u32, th as u32, image::imageops::FilterType::Triangle);
            image::imageops::replace(&mut sheet, &t, ((i % cols) * thumb_w) as i64, ((i / cols) * th) as i64);
        }
        sheet.save(dir.join(file)).map_err(|e| format!("{file}: {e}"))?;
    }
    Ok(())
}

// ---------- verdict ----------

/// The score a verdict uses: the worse of the mean FLIP over the model's pixels in the worst view
/// and the worst body region's mean over all views; with `--only`, the worst region outside the
/// intended ones. Limits come from the calibration in experiments/goldens/REVIEW.md.
pub const WARN: f32 = 0.035;
pub const FAIL: f32 = 0.055;

fn verdict(score: f32, warn: f32, fail: f32) -> &'static str {
    if score >= fail {
        "FAIL"
    } else if score >= warn {
        "WARN"
    } else {
        "PASS"
    }
}

fn summarize(views: &[ViewScore], regions: &RegionVals, frames: &RegionFrames, only: &[String], warn: f32, fail: f32, json_out: bool, extra: Value) -> Value {
    let mut by_kind: BTreeMap<&str, Vec<&ViewScore>> = BTreeMap::new();
    for v in views {
        by_kind.entry(v.kind.tag()).or_default().push(v);
    }
    let region_rows: Vec<(String, Stats)> = {
        let mut r: Vec<(String, Stats)> = regions.iter().filter(|(_, v)| v.len() >= 60).map(|(k, v)| (k.clone(), pc::stats(v.clone()))).collect();
        r.sort_by(|a, b| b.1.mean.total_cmp(&a.1.mean));
        r
    };
    let worst_view = views.iter().max_by(|a, b| a.flip.mean.total_cmp(&b.flip.mean));
    let avg_mean = views.iter().map(|v| v.flip.mean).sum::<f32>() / views.len().max(1) as f32;
    let worst_view_mean = worst_view.map(|v| v.flip.mean).unwrap_or(0.0);
    let worst_region_mean = region_rows.first().map(|r| r.1.mean).unwrap_or(0.0);
    // With --only: the change is meant for some regions; judge what happens everywhere else.
    let inside: Vec<&(String, Stats)> = region_rows.iter().filter(|r| only.contains(&r.0)).collect();
    let outside: Vec<&(String, Stats)> = region_rows.iter().filter(|r| !only.contains(&r.0)).collect();
    let intended_px: usize = inside.iter().map(|r| r.1.n).sum();
    let intended_mean = if intended_px > 0 { inside.iter().map(|r| r.1.mean as f64 * r.1.n as f64).sum::<f64>() / intended_px as f64 } else { 0.0 } as f32;
    let collateral = outside.first().map(|r| (r.0.clone(), r.1.mean)); // rows are sorted worst first
    let ssim_min = views.iter().map(|v| v.ssim).fold(1.0f32, f32::min);
    // The worst single (region, view) pair: a small part that is wrong in one close-up shows here even
    // when its pooled mean is small. With --only, regions inside the intended set are left out.
    let region_worst = |name: &str| -> Option<(String, usize, f32)> { frames.get(name).and_then(|f| f.iter().max_by(|a, b| a.2.total_cmp(&b.2)).cloned()) };
    let mut part_rows: Vec<(String, String, usize, f32)> = frames.keys().filter(|k| !only.contains(*k)).filter_map(|k| region_worst(k).map(|w| (k.clone(), w.0, w.1, w.2))).collect();
    part_rows.sort_by(|a, b| b.3.total_cmp(&a.3));
    let worst_part = part_rows.first().cloned();
    // A part's own close-ups (views named part_<region>_a/_b): the region's mean FLIP in the view made for it.
    let mut own: Vec<(String, String, usize, f32)> = frames
        .iter()
        .filter(|(k, _)| !only.contains(*k))
        .filter_map(|(k, list)| {
            let pre = format!("part_{k}_");
            list.iter().filter(|e| e.0.starts_with(&pre)).max_by(|a, b| a.2.total_cmp(&b.2)).map(|e| (k.clone(), e.0.clone(), e.1, e.2))
        })
        .collect();
    own.sort_by(|a, b| b.3.total_cmp(&a.3));
    let own_worst = own.first().map(|r| r.3).unwrap_or(0.0);
    let score = if only.is_empty() { worst_view_mean.max(worst_region_mean) } else { collateral.as_ref().map(|c| c.1).unwrap_or(0.0) }.max(own_worst);
    let ver = verdict(score, warn, fail);
    let j = json!({
        "verdict": ver,
        "score": score,
        "score_is": if only.is_empty() { "max(worst view mean FLIP, worst region mean FLIP, worst part in its own close-up)" } else { "max(worst region mean FLIP outside the intended regions, worst part outside them in its own close-up)" },
        "intended": if only.is_empty() { Value::Null } else { json!({"regions": only, "pixels": intended_px, "flip_mean": intended_mean, "each": inside.iter().map(|r| json!({"region": r.0, "flip_mean": r.1.mean, "pixels": r.1.n})).collect::<Vec<_>>()}) },
        "collateral": collateral.as_ref().map(|c| json!({"region": c.0, "flip_mean": c.1})),
        "warn": warn,
        "fail": fail,
        "flip_mean_over_views": avg_mean,
        "flip_worst_view": worst_view.map(|v| json!({"view": format!("{}__{}", v.name, v.kind.tag()), "mean": v.flip.mean, "p95": v.flip.p95})),
        "flip_p95_worst": views.iter().map(|v| v.flip.p95).fold(0.0f32, f32::max),
        "ssim_min": ssim_min,
        "part_close_ups": own.iter().map(|w| json!({"region": w.0, "view": w.1, "pixels": w.2, "flip_mean": w.3})).collect::<Vec<_>>(),
        "worst_part_view": worst_part.as_ref().map(|w| json!({"region": w.0, "view": w.1, "pixels": w.2, "flip_mean": w.3})),
        "parts_worst_view": part_rows.iter().take(8).map(|w| json!({"region": w.0, "view": w.1, "pixels": w.2, "flip_mean": w.3})).collect::<Vec<_>>(),
        "views": views.iter().map(|v| json!({
            "view": format!("{}__{}", v.name, v.kind.tag()),
            "flip_mean": v.flip.mean, "flip_p50": v.flip.p50, "flip_p95": v.flip.p95, "flip_p99": v.flip.p99, "flip_max": v.flip.max,
            "flip_frame_mean": v.flip_frame_mean, "ssim": v.ssim, "coverage": v.coverage,
        })).collect::<Vec<_>>(),
        "regions": region_rows.iter().map(|(k, s)| json!({"region": k, "pixels": s.n, "flip_mean": s.mean, "flip_p95": s.p95, "worst_view": region_worst(k).map(|w| json!({"view": w.0, "pixels": w.1, "flip_mean": w.2}))})).collect::<Vec<_>>(),
        "info": extra,
    });
    if !json_out {
        println!("{:<26} {:>8} {:>8} {:>8} {:>8} {:>7} {:>6}", "view", "meanFLIP", "p95", "p99", "max", "SSIM", "cover");
        for v in views {
            println!("{:<26} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>7.4} {:>5.1}%", format!("{}__{}", v.name, v.kind.tag()), v.flip.mean, v.flip.p95, v.flip.p99, v.flip.max, v.ssim, v.coverage * 100.0);
        }
        println!("mean of view means {avg_mean:.4}   worst view {worst_view_mean:.4}   worst p95 {:.4}   lowest SSIM {ssim_min:.4}", j["flip_p95_worst"].as_f64().unwrap_or(0.0));
        println!("regions (worst first, at least 60 px; worst = the region's worst single view):");
        for (k, s) in region_rows.iter().take(8) {
            let w = region_worst(k).map(|w| format!("{} {:.4}", w.0, w.2)).unwrap_or_default();
            println!("  {:<22} {:>9} px   mean {:.4}   p95 {:.4}   worst {}", k, s.n, s.mean, s.p95, w);
        }
        if !own.is_empty() {
            println!("part close-ups (the region in its own view, worst first): {}", own.iter().map(|w| format!("{} {:.4}", w.0, w.3)).collect::<Vec<_>>().join(", "));
        }
        if let Some(w) = &worst_part {
            println!("worst part in one view: {} in {} {:.4} ({} px)", w.0, w.1, w.3, w.2);
        }
        if !only.is_empty() {
            println!("intended regions ({}): mean {intended_mean:.4} over {intended_px} px; worst region outside them: {}", only.join(", "), collateral.as_ref().map(|c| format!("{} {:.4}", c.0, c.1)).unwrap_or_else(|| "none".into()));
        }
        println!("score {score:.4}  (warn {warn}, fail {fail})  =>  {ver}");
    }
    j
}

// ---------- commands ----------

pub fn judge_cmd(args: &Args) -> Result<(), String> {
    let a = A(&args.0);
    if a.flag("--make-goldens") || a.opt("--make-goldens").is_some() {
        return make_goldens(&a);
    }
    if a.opt("--golden-dir").is_some() {
        return golden_run(&a);
    }
    if a.opt("--strands").is_some() {
        return strands_cmd(&a);
    }
    let pos = a.positionals();
    let [r, t] = pos.as_slice() else { return Err("judge needs two files: judge REF TEST (two PNGs or two GLBs)".into()) };
    let is_glb = |p: &Path| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("glb"));
    match (is_glb(r), is_glb(t)) {
        (true, true) => glb_pair(&a, r, t),
        (false, false) => image_pair(&a, r, t),
        _ => Err("judge needs two images or two GLBs, not one of each".into()),
    }
}

fn only_list(a: &A) -> Vec<String> {
    a.opt("--only").map(|s| s.split(',').map(|x| x.trim().to_owned()).filter(|x| !x.is_empty()).collect()).unwrap_or_default()
}

fn out_dir(a: &A) -> Result<Option<PathBuf>, String> {
    match a.opt("--out-dir") {
        Some(d) => {
            let p = PathBuf::from(d);
            check_out_dir(&p)?;
            std::fs::create_dir_all(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            Ok(Some(p))
        }
        None => Ok(None),
    }
}

fn finish(a: &A, dir: Option<&Path>, mut j: Value, sheets: &[(Kind, RgbaImage)]) -> Result<(), String> {
    if let Some(d) = dir {
        write_sheets(d, sheets)?;
        j["written"] = json!(d.display().to_string());
        std::fs::write(d.join("judge.json"), serde_json::to_vec_pretty(&j).unwrap()).map_err(|e| format!("judge.json: {e}"))?;
    }
    if a.flag("--json") {
        println!("{}", serde_json::to_string_pretty(&j).unwrap());
    }
    if let Some(d) = dir {
        if !a.flag("--json") {
            println!("wrote {}", d.display());
        }
    }
    Ok(())
}

fn image_pair(a: &A, r: &Path, t: &Path) -> Result<(), String> {
    let (ra, rb) = (load_png(r)?, load_png(t)?);
    let dir = out_dir(a)?;
    let ppd = a.num("--ppd", pc::DEFAULT_PPD)?;
    let heat_max = a.num("--heat-max", 0.5)?;
    let bg: [f32; 3] = match a.opt("--bg") {
        Some(s) => {
            let v: Vec<f32> = s.split(',').filter_map(|x| x.trim().parse::<f32>().ok()).collect();
            if v.len() != 3 {
                return Err("--bg needs r,g,b (0..255)".into());
            }
            [v[0] / 255.0, v[1] / 255.0, v[2] / 255.0]
        }
        None => ra.px[0],
    };
    let mask = pc::foreground_mask(&ra, &rb, bg, 1.5 / 255.0, 2);
    // Optional region map: an image where each distinct color is a region.
    let (mut regions, mut names, region): (RegionVals, Vec<String>, Option<Vec<Option<u16>>>) = (BTreeMap::new(), vec![], None);
    let mut frames = RegionFrames::new();
    let mut region = region;
    if let Some(ids) = a.opt("--ids") {
        let im = load_png(Path::new(ids))?;
        if (im.w, im.h) != (ra.w, ra.h) {
            return Err("--ids image must be the same size as the pictures".into());
        }
        let mut seen: Vec<[u8; 3]> = Vec::new();
        let v: Vec<Option<u16>> = im
            .px
            .iter()
            .map(|p| {
                let c = p.map(|x| (x * 255.0 + 0.5) as u8);
                if c == [128, 128, 128] || c == [0, 0, 0] {
                    return None;
                }
                let i = seen.iter().position(|s| *s == c).unwrap_or_else(|| {
                    seen.push(c);
                    seen.len() - 1
                });
                Some(i as u16)
            })
            .collect();
        names = seen.iter().map(|c| format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])).collect();
        region = Some(v);
    }
    let out = Out { dir: dir.as_deref(), heat_max, ppd };
    let mut sheets = Vec::new();
    let name = t.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "image".into());
    let s = score_frame(&name, Kind::Large, &ra, &rb, &mask, region.as_deref(), &names, &mut regions, &mut frames, &out, &mut sheets)?;
    let (warn, fail) = (a.num("--warn", WARN)?, a.num("--fail", FAIL)?);
    let j = summarize(&[s], &regions, &frames, &only_list(a), warn, fail, a.flag("--json"), json!({"mode": "images", "reference": r.display().to_string(), "test": t.display().to_string(), "ppd": ppd}));
    if let Some(d) = &dir {
        let f = pc::flip(&ra, &rb, ppd)?;
        save_rgb(&d.join("heat.png"), &heat_rgb(&f.err, &mask, ra.w, ra.h, heat_max))?;
    }
    finish(a, dir.as_deref(), j, &sheets)
}

fn pick_views(a: &A, all: Vec<View>) -> Result<Vec<View>, String> {
    match a.opt("--views") {
        Some(list) => {
            let want: Vec<&str> = list.split(',').collect();
            for w in &want {
                if !all.iter().any(|v| v.name == *w) {
                    return Err(format!("no view named {w}; views: {}", all.iter().map(|v| v.name.as_str()).collect::<Vec<_>>().join(", ")));
                }
            }
            Ok(all.into_iter().filter(|v| want.contains(&v.name.as_str())).collect())
        }
        None => Ok(all),
    }
}

fn load_parts(a: &A) -> Option<Parts> {
    let dir = a.opt("--parts").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(KIT_PARTS));
    Parts::load(&dir).ok()
}

enum Reference<'m> {
    Model { model: &'m Model, tex: &'m [Option<wb_core::Texture>] },
    Disk(PathBuf),
}

struct Run<'m> {
    reference: Reference<'m>,
    cand: &'m Model,
    cand_tex: &'m [Option<wb_core::Texture>],
    labels_ref: Option<Labels>,
    labels_cand: Labels,
}

type Scored = (Vec<ViewScore>, RegionVals, RegionFrames, Vec<Value>, Vec<(Kind, RgbaImage)>);

fn run_views(a: &A, run: &Run, views: &[View], dir: Option<&Path>) -> Result<Scored, String> {
    let ppd = a.num("--ppd", pc::DEFAULT_PPD)?;
    let heat_max = a.num("--heat-max", 0.5)?;
    let out = Out { dir, heat_max, ppd };
    let (mut scores, mut regions, mut skipped, mut sheets) = (Vec::new(), RegionVals::new(), Vec::new(), Vec::new());
    let mut frames = RegionFrames::new();
    for v in views {
        for &k in &v.kinds {
            let name = v.name.as_str();
            let Some(sb) = golden::render_view(run.cand, run.cand_tex, v, k) else {
                skipped.push(json!({"view": v.stem(k), "why": "the test model has no clip for this view"}));
                continue;
            };
            let (ra, pick_a): (Rgb, Option<Vec<i32>>) = match &run.reference {
                Reference::Model { model, tex } => match golden::render_view(model, tex, v, k) {
                    Some(s) => (s.rgb, Some(s.pick)),
                    None => {
                        skipped.push(json!({"view": v.stem(k), "why": "the reference has no clip for this view"}));
                        continue;
                    }
                },
                Reference::Disk(d) => {
                    let p = d.join(format!("{}.png", v.stem(k)));
                    match load_png(&p) {
                        Ok(im) => (im, None),
                        Err(e) => {
                            skipped.push(json!({"view": v.stem(k), "why": e}));
                            continue;
                        }
                    }
                }
            };
            if (ra.w, ra.h) != (sb.rgb.w, sb.rgb.h) {
                return Err(format!("{}: golden is {}x{}, candidate render is {}x{}", v.stem(k), ra.w, ra.h, sb.rgb.w, sb.rgb.h));
            }
            // Model pixels: covered in either picture (or, from disk, differing from the background).
            let n = ra.w * ra.h;
            let covered: Vec<bool> = (0..n).map(|i| pick_a.as_ref().is_some_and(|p| p[i] >= 0) || sb.pick[i] >= 0).collect();
            let mask = match &pick_a {
                Some(_) => pc::dilate(&covered, ra.w, ra.h, 2),
                None => pc::foreground_mask(&ra, &sb.rgb, [golden::BACKGROUND[0] as f32 / 255.0, golden::BACKGROUND[1] as f32 / 255.0, golden::BACKGROUND[2] as f32 / 255.0], 1.5 / 255.0, 2),
            };
            // Region of each pixel: the reference's triangle when it has one, else the test's.
            let (names, region): (&[String], Vec<Option<u16>>) = match (&run.labels_ref, &pick_a) {
                (Some(lr), Some(pa)) if model_shape(run.cand) == lr.shape => (&lr.names, (0..n).map(|i| if pa[i] >= 0 { Some(lr.tri[pa[i] as usize]) } else if sb.pick[i] >= 0 { Some(lr.tri[sb.pick[i] as usize]) } else { None }).collect()),
                (Some(lr), Some(pa)) => (&lr.names, (0..n).map(|i| if pa[i] >= 0 { Some(lr.tri[pa[i] as usize]) } else { None }).collect()),
                _ => (&run.labels_cand.names, sb.pick.iter().map(|&p| (p >= 0).then(|| run.labels_cand.tri[p as usize])).collect()),
            };
            // Region names differ between label sets; pool by name.
            let s = score_frame(name, k, &ra, &sb.rgb, &mask, Some(&region), names, &mut regions, &mut frames, &out, &mut sheets)?;
            if let Some(d) = dir {
                if a.flag("--heat-all") && (k == Kind::Large || k == Kind::Close) {
                    // Heat map only for the large views, next to the sheets.
                    let f = pc::flip(&ra, &sb.rgb, ppd)?;
                    let hd = d.join("heat");
                    std::fs::create_dir_all(&hd).map_err(|e| format!("{}: {e}", hd.display()))?;
                    save_rgb(&hd.join(format!("{}.png", v.stem(k))), &heat_rgb(&f.err, &mask, ra.w, ra.h, heat_max))?;
                }
            }
            scores.push(s);
        }
    }
    Ok((scores, regions, frames, skipped, sheets))
}

fn glb_pair(a: &A, r: &Path, t: &Path) -> Result<(), String> {
    let (ma, mb) = (Model::open(r)?, Model::open(t)?);
    let (ta, tb) = (image_textures(&ma), image_textures(&mb));
    let parts = load_parts(a);
    let labels_ref = labels_for(&ma, parts.as_ref());
    let views = match a.opt("--manifest") {
        Some(m) => {
            let j: Value = serde_json::from_slice(&std::fs::read(m).map_err(|e| format!("{m}: {e}"))?).map_err(|e| format!("{m}: {e}"))?;
            golden::views_from_manifest(&j)?
        }
        None => all_views(&ma, &labels_ref),
    };
    let views = pick_views(a, views)?;
    let run = Run { reference: Reference::Model { model: &ma, tex: &ta }, cand: &mb, cand_tex: &tb, labels_ref: Some(labels_ref), labels_cand: labels_for(&mb, None) };
    let dir = out_dir(a)?;
    let (scores, regions, frames, skipped, sheets) = run_views(a, &run, &views, dir.as_deref())?;
    let (warn, fail) = (a.num("--warn", WARN)?, a.num("--fail", FAIL)?);
    // Meshes the reference has and the test lacks (a missing dagger scores as a large error, but say why).
    let names = |m: &Model| -> Vec<String> { m.meshes.iter().map(|x| x.name.clone()).collect() };
    let (na, nb) = (names(&ma), names(&mb));
    let missing: Vec<&String> = na.iter().filter(|n| !nb.contains(n)).collect();
    let extra_meshes: Vec<&String> = nb.iter().filter(|n| !na.contains(n)).collect();
    // Vertex colors and texture brightness: warn when the two models would shade differently in the game.
    let shading_of = |m: &Model| -> Result<Value, String> {
        let imgs: Vec<(String, wb_core::bake::Img)> = m.images.iter().map(|im| wb_core::bake::Img::decode(&im.bytes).map(|d| (im.name.clone(), d))).collect::<Result<_, _>>()?;
        let refs: Vec<(String, &wb_core::bake::Img)> = imgs.iter().map(|(n, d)| (n.clone(), d)).collect();
        Ok(wb_core::bake::shading_check(&m.glb, &refs))
    };
    let (sh_ref, sh_test) = (shading_of(&ma)?, shading_of(&mb)?);
    let mut shading_warnings: Vec<String> = sh_test["warnings"].as_array().cloned().unwrap_or_default().iter().filter_map(|x| x.as_str().map(str::to_owned)).collect();
    shading_warnings.extend(wb_core::bake::shading_compare(&sh_ref, &sh_test, a.num("--shading-tol", 0.15)? as f64));
    let info = json!({"mode": "glb", "reference": r.display().to_string(), "test": t.display().to_string(), "region_labels": run.labels_ref.as_ref().map(|l| l.source), "skipped": skipped, "views_drawn": scores.len(), "meshes_missing_in_test": missing, "meshes_only_in_test": extra_meshes,
        "shading": { "warnings": shading_warnings, "reference": sh_ref, "test": sh_test }});
    if !a.flag("--json") {
        for w in &shading_warnings {
            println!("SHADING WARNING: {w}");
        }
        if !skipped.is_empty() {
            println!("skipped {} views", skipped.len());
        }
        if !missing.is_empty() {
            println!("test has no mesh named: {}", missing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "));
        }
    }
    let j = summarize(&scores, &regions, &frames, &only_list(a), warn, fail, a.flag("--json"), info);
    finish(a, dir.as_deref(), j, &sheets)
}

fn make_goldens(a: &A) -> Result<(), String> {
    let path = PathBuf::from(a.opt("--make-goldens").ok_or("--make-goldens needs a GLB")?);
    let name = a.opt("--name").ok_or("--make-goldens needs --name")?;
    let base = PathBuf::from(a.opt("--out-dir").ok_or("--make-goldens needs --out-dir")?);
    check_out_dir(&base)?;
    let dir = base.join(name);
    if dir.join("manifest.json").exists() && !a.flag("--force") {
        return Err(format!("{} already has goldens; pass --force to replace them", dir.display()));
    }
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let model = Model::from_bytes(&bytes)?;
    let tex = image_textures(&model);
    let labels = labels_for(&model, load_parts(a).as_ref());
    let views = all_views(&model, &labels);
    println!("part close-ups: {}", part_groups(&model, &labels).iter().map(|g| format!("{} ({} triangles)", g.name, g.tris.len())).collect::<Vec<_>>().join(", "));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut count = 0;
    let mut sheets: Vec<(Kind, RgbaImage)> = Vec::new();
    for v in &views {
        for &k in &v.kinds {
            let Some(shot) = golden::render_view(&model, &tex, v, k) else { continue };
            save_rgb(&dir.join(format!("{}.png", v.stem(k))), &shot.rgb)?;
            let mut img = RgbaImage::from_raw(shot.rgb.w as u32, shot.rgb.h as u32, shot.rgb.to_rgba8()).unwrap();
            let (iw, ih) = (img.width() as usize, img.height() as usize);
            wb_core::font::draw(img.as_mut(), iw, ih, 4, 4, &v.stem(k).replace('_', "-"), (iw / 220).max(1), [0, 0, 0]);
            sheets.push((k, img));
            count += 1;
        }
    }
    // Contact sheets (pictures only), one per size.
    for (kind, file, cols, cell) in [(Kind::Game, "contact-game.png", 6usize, 320usize), (Kind::Large, "contact-large.png", 4, 384), (Kind::Close, "contact-close.png", 4, 384)] {
        let items: Vec<&RgbaImage> = sheets.iter().filter(|s| s.0 == kind).map(|s| &s.1).collect();
        if items.is_empty() {
            continue;
        }
        let rows = items.len().div_ceil(cols);
        let mut sheet = RgbaImage::from_pixel((cols * cell) as u32, (rows * cell) as u32, image::Rgba([20, 20, 24, 255]));
        for (i, im) in items.iter().enumerate() {
            let t = image::imageops::resize(*im, cell as u32, cell as u32, image::imageops::FilterType::Triangle);
            image::imageops::replace(&mut sheet, &t, ((i % cols) * cell) as i64, ((i / cols) * cell) as i64);
        }
        sheet.save(dir.join(file)).map_err(|e| format!("{file}: {e}"))?;
    }
    let mut m = golden::manifest(name, &path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(), &views);
    m["source_bytes"] = json!(bytes.len());
    m["source_fnv64"] = json!(format!("{:016x}", fnv64(&bytes)));
    std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).map_err(|e| format!("manifest.json: {e}"))?;
    println!("wrote {count} pictures and manifest.json to {}", dir.display());
    Ok(())
}

fn golden_run(a: &A) -> Result<(), String> {
    let gdir = PathBuf::from(a.opt("--golden-dir").unwrap());
    let pos = a.positionals();
    let [cand] = pos.as_slice() else { return Err("judge --golden-dir DIR needs one candidate GLB".into()) };
    let j: Value = serde_json::from_slice(&std::fs::read(gdir.join("manifest.json")).map_err(|e| format!("{}: {e}", gdir.join("manifest.json").display()))?).map_err(|e| format!("manifest.json: {e}"))?;
    let views = pick_views(a, golden::views_from_manifest(&j)?)?;
    let model = Model::open(cand)?;
    let tex = image_textures(&model);
    let run = Run { reference: Reference::Disk(gdir.clone()), cand: &model, cand_tex: &tex, labels_ref: None, labels_cand: labels_for(&model, None) };
    let dir = out_dir(a)?;
    let (scores, regions, frames, skipped, sheets) = run_views(a, &run, &views, dir.as_deref())?;
    let (warn, fail) = (a.num("--warn", WARN)?, a.num("--fail", FAIL)?);
    let info = json!({"mode": "golden", "golden": gdir.display().to_string(), "golden_source": j["source"], "candidate": cand.display().to_string(), "region_labels": run.labels_cand.source, "skipped": skipped, "views_drawn": scores.len()});
    let out = summarize(&scores, &regions, &frames, &only_list(a), warn, fail, a.flag("--json"), info);
    finish(a, dir.as_deref(), out, &sheets)
}

// ---------- hair structure check (no reference) ----------

/// Hue (degrees), saturation, value of an 8-bit color.
fn hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (r, g, b) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
    let d = mx - mn;
    let h = if d <= 1e-6 {
        0.0
    } else if mx == r {
        60.0 * (((g - b) / d).rem_euclid(6.0))
    } else if mx == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    (h, if mx <= 1e-6 { 0.0 } else { d / mx }, mx)
}

/// Zidane's hair is blond: hue 31..68 degrees, saturated, not dark.
fn is_blond_px(px: &[u8]) -> bool {
    if px[3] < 128 {
        return false;
    }
    let (h, s, v) = hsv(px[0], px[1], px[2]);
    (31.0..=68.0).contains(&h) && s >= 0.45 && v >= 0.3
}

/// Hair triangles per mesh: on a head bone (unless --any-bone) and painted blond at 4 of 7 points.
fn hair_flags(a: &A, model: &Model, tex: &[Option<wb_core::Texture>]) -> Vec<Vec<bool>> {
    let any_bone = a.flag("--any-bone");
    model
        .meshes
        .iter()
        .enumerate()
        .map(|(mi, m)| {
            let Some(t) = tex.get(mi).and_then(|t| t.as_ref()) else { return vec![false; m.tris.len()] };
            let w3: [[f32; 3]; 7] = [[1.0 / 3.0; 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]];
            (0..m.tris.len())
                .map(|ti| {
                    if !any_bone {
                        let j = model.triangle_joint(mi, ti);
                        let g = model.joint_names.get(j).map(|s| bone_group(s)).unwrap_or_default();
                        if !g.starts_with("head") {
                            return false;
                        }
                    }
                    let tri = m.tris[ti];
                    let uv: [[f32; 2]; 3] = std::array::from_fn(|k| m.uv.get(tri[k] as usize).copied().unwrap_or([0.0, 0.0]));
                    let hits = w3
                        .iter()
                        .filter(|w| {
                            let u = uv[0][0] * w[0] + uv[1][0] * w[1] + uv[2][0] * w[2];
                            let v = uv[0][1] * w[0] + uv[1][1] * w[1] + uv[2][1] * w[2];
                            let x = ((u * t.w as f32).floor() as i64).clamp(0, t.w as i64 - 1) as usize;
                            let y = ((v * t.h as f32).floor() as i64).clamp(0, t.h as i64 - 1) as usize;
                            is_blond_px(&t.rgba[(y * t.w + x) * 4..(y * t.w + x) * 4 + 4])
                        })
                        .count();
                    hits >= 4
                })
                .collect()
        })
        .collect()
}

/// Provisional limits from ten hair models in two clusters (REVIEW.md): rough hair sits at or above
/// seam mean 23 deg, 21.8% bad seam samples and 68.7% bad root-to-tip samples; the clean versions
/// sit at or below 16.6 deg, 16.7% and 40.7%. The limits are placed in those gaps.
const STRAND_SEAM_MEAN_DEG: f64 = 20.0;
const STRAND_SEAM_BAD_FRAC: f64 = 0.19;
const STRAND_ROOT_TIP_BAD_FRAC: f64 = 0.55;

fn agg_json(g: &strands::Agg) -> Value {
    json!({"samples": g.n, "weight": g.weight, "mean_deg": g.mean_deg, "p50_deg": g.p50_deg, "p90_deg": g.p90_deg, "bad_frac": g.bad_frac})
}

fn stamp(img: &mut RgbaImage, x: i64, y: i64, r: i64, c: [u8; 4]) {
    for dy in -r..=r {
        for dx in -r..=r {
            let (px, py) = (x + dx, y + dy);
            if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                img.put_pixel(px as u32, py as u32, image::Rgba(c));
            }
        }
    }
}

fn line(img: &mut RgbaImage, p0: [f32; 2], p1: [f32; 2], r: i64, c: [u8; 4]) {
    let n = ((p1[0] - p0[0]).hypot(p1[1] - p0[1])).ceil().max(1.0) as usize;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        stamp(img, (p0[0] + (p1[0] - p0[0]) * t).round() as i64, (p0[1] + (p1[1] - p0[1]) * t).round() as i64, r, c);
    }
}

/// One model's strand report: prints it and returns it as JSON; with `dir` writes seam maps.
fn strands_one(a: &A, role: &str, path: &Path, dir: Option<&Path>) -> Result<Value, String> {
    let model = Model::open(path)?;
    let tex = wb_core::model_textures(&model);
    let flags = hair_flags(a, &model, &tex);
    let mut pieces: Vec<strands::Piece> = Vec::new();
    let mut piece_mesh: Vec<usize> = Vec::new();
    for (mi, m) in model.meshes.iter().enumerate() {
        if let (Some(t), true) = (tex.get(mi).and_then(|t| t.as_ref()), flags[mi].iter().any(|&f| f)) {
            pieces.push(strands::Piece { mesh: m, tris: &flags[mi], rgba: &t.rgba, w: t.w, h: t.h });
            piece_mesh.push(mi);
        }
    }
    let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
    if pieces.is_empty() {
        println!("{role} {name}: no hair triangles found (head-bone triangles painted blond)");
        return Ok(json!({"role": role, "model": path.display().to_string(), "hair_triangles": 0}));
    }
    let r = strands::check(&pieces, &strands::Opts::default());
    let pct = |x: f64| x * 100.0;
    println!("{role} {name}");
    println!("  hair triangles {} in {} pieces, {} charts; seam edges {}, inner edges {}", r.triangles, pieces.len(), r.charts, r.seam_edges, r.inner_edges);
    for (label, g) in [("seams (chart borders)", &r.seam), ("inner edges (baseline)", &r.inner), ("root-to-tip", &r.root_tip)] {
        println!("  {:<22} {:>6} samples   mean {:>5.1} deg   median {:>5.1}   p90 {:>5.1}   over {} deg {:>5.1}%", label, g.n, g.mean_deg, g.p50_deg, g.p90_deg, strands::Opts::default().bad_deg, pct(g.bad_frac));
    }
    println!("  seam excess (seam mean minus inner mean): {:.1} deg", r.seam_excess_deg());
    let mut flags: Vec<String> = Vec::new();
    if r.seam.n >= 30 && r.seam.mean_deg >= STRAND_SEAM_MEAN_DEG {
        flags.push(format!("seam mean {:.1} deg is at or above {STRAND_SEAM_MEAN_DEG}", r.seam.mean_deg));
    }
    if r.seam.n >= 30 && r.seam.bad_frac >= STRAND_SEAM_BAD_FRAC {
        flags.push(format!("{:.1}% of seam samples are over {} deg (limit {:.0}%)", pct(r.seam.bad_frac), strands::Opts::default().bad_deg, pct(STRAND_SEAM_BAD_FRAC)));
    }
    if r.root_tip.n >= 30 && r.root_tip.bad_frac >= STRAND_ROOT_TIP_BAD_FRAC {
        flags.push(format!("{:.1}% of root-to-tip samples are over {} deg (limit {:.0}%)", pct(r.root_tip.bad_frac), strands::Opts::default().bad_deg, pct(STRAND_ROOT_TIP_BAD_FRAC)));
    }
    if flags.is_empty() {
        println!("  strand flags: none (provisional limits)");
    } else {
        println!("  STRAND FLAGS (provisional limits): {}", flags.join("; "));
    }
    let mut worst: Vec<&strands::EdgeScore> = r.edges.iter().filter(|e| e.kind == strands::Kind::Seam).collect();
    worst.sort_by(|x, y| (y.mean_deg * y.weight).total_cmp(&(x.mean_deg * x.weight)));
    let worst_j: Vec<Value> = worst
        .iter()
        .take(10)
        .map(|e| json!({"mesh_a": model.meshes[piece_mesh[e.a.0]].name, "tri_a": e.a.1, "mesh_b": model.meshes[piece_mesh[e.b.0]].name, "tri_b": e.b.1, "mean_deg": e.mean_deg, "weight": e.weight, "uv_a": e.uv_a, "uv_b": e.uv_b}))
        .collect();
    if let Some(d) = dir {
        let sub = d.join(format!("{role}-{name}"));
        std::fs::create_dir_all(&sub).map_err(|e| format!("{}: {e}", sub.display()))?;
        for (pi, &mi) in piece_mesh.iter().enumerate() {
            let t = tex[mi].as_ref().unwrap();
            let scale = (1024 / t.w.max(t.h).max(1)).clamp(1, 4);
            let mut img = RgbaImage::from_pixel((t.w * scale) as u32, (t.h * scale) as u32, image::Rgba([30, 30, 34, 255]));
            for y in 0..t.h * scale {
                for x in 0..t.w * scale {
                    let s = &t.rgba[((y / scale) * t.w + x / scale) * 4..][..4];
                    if s[3] >= 128 {
                        img.put_pixel(x as u32, y as u32, image::Rgba([(s[0] as f32 * 0.55) as u8, (s[1] as f32 * 0.55) as u8, (s[2] as f32 * 0.55) as u8, 255]));
                    }
                }
            }
            let to_px = |uv: [f32; 2]| [uv[0] * (t.w * scale) as f32, uv[1] * (t.h * scale) as f32];
            for pass in [strands::Kind::Inner, strands::Kind::Seam] {
                for e in r.edges.iter().filter(|e| e.kind == pass) {
                    for (side, uv) in [(e.a.0, e.uv_a), (e.b.0, e.uv_b)] {
                        if side != pi {
                            continue;
                        }
                        let (c, rad) = match pass {
                            strands::Kind::Inner => ([90, 110, 150, 255], 0),
                            strands::Kind::Seam if e.mean_deg <= 12.0 => ([40, 220, 80, 255], 1),
                            strands::Kind::Seam if e.mean_deg <= 30.0 => ([240, 210, 40, 255], 1),
                            strands::Kind::Seam => ([240, 50, 50, 255], 1),
                        };
                        line(&mut img, to_px(uv[0]), to_px(uv[1]), rad, c);
                    }
                }
            }
            img.save(sub.join(format!("seams-mesh{mi}.png"))).map_err(|e| format!("seams png: {e}"))?;
        }
        println!("  wrote seam maps to {}", sub.display());
    }
    Ok(json!({
        "role": role,
        "model": path.display().to_string(),
        "hair_triangles": r.triangles,
        "pieces": pieces.len(),
        "charts": r.charts,
        "seam_edges": r.seam_edges,
        "inner_edges": r.inner_edges,
        "seam": agg_json(&r.seam),
        "inner": agg_json(&r.inner),
        "root_tip": agg_json(&r.root_tip),
        "seam_excess_deg": r.seam_excess_deg(),
        "worst_seams": worst_j,
        "flags": flags,
        "limits_are_provisional": true,
    }))
}

fn strands_cmd(a: &A) -> Result<(), String> {
    let path = PathBuf::from(a.opt("--strands").ok_or("--strands needs a GLB")?);
    let dir = out_dir(a)?;
    let mut res = vec![strands_one(a, "model", &path, dir.as_deref())?];
    if let Some(r) = a.opt("--ref") {
        res.push(strands_one(a, "reference", Path::new(r), dir.as_deref())?);
    }
    let j = json!({"check": "strands", "options": {"per_edge": strands::Opts::default().per_edge, "min_coherence": strands::Opts::default().min_coherence, "bad_deg": strands::Opts::default().bad_deg}, "results": res, "note": "no verdict: the limits come from the calibration table in experiments/goldens/REVIEW.md"});
    if let Some(d) = &dir {
        std::fs::write(d.join("strands.json"), serde_json::to_vec_pretty(&j).unwrap()).map_err(|e| format!("strands.json: {e}"))?;
    }
    if a.flag("--json") {
        println!("{}", serde_json::to_string_pretty(&j).unwrap());
    }
    Ok(())
}
