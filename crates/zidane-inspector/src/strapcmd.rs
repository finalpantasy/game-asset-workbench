//! Loose-piece and strap commands: `shells`, `strap-sweep`, `strap-rebuild`.
use crate::Args;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use wb_core::render::Camera;
use wb_core::shells::{self, Shell};
use wb_core::Model;

/// Options that take a value in these commands (the global list does not know them).
const OWN_VALUE_OPTS: [&str; 32] = ["--exit-raise", "--exit-curve", "--exit-max", "--exit-down", "--lift-len", 
    "--out", "--out-dir", "--size", "--touch", "--tol", "--reach", "--clip", "--time", "--shell", "--times", "--clips", "--on", "--json-out", "--labels",
    "--tip-thigh", "--thickness", "--ride", "--tuck-depth", "--columns", "--row-step", "--fit-frames", "--thigh-joint", "--fit-reach", "--max-thigh-share", "--clear-gap", "--tuck-len", "--exit-edge",
];

/// Arguments that are not options or option values.
fn file_args(args: &Args) -> Vec<PathBuf> {
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

/// First file argument, or the stock export.
fn file_arg(args: &Args) -> PathBuf {
    file_args(args).into_iter().next().unwrap_or_else(|| PathBuf::from(crate::STOCK))
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

fn joint_name(model: &Model, j: usize) -> String {
    model.joint_names.get(j).cloned().unwrap_or_else(|| format!("joint{j}"))
}

pub struct ShellReport {
    pub shells: Vec<Shell>,
    pub near: Vec<Vec<shells::Near>>,
    pub visible: Vec<usize>,
    pub over: Vec<Vec<usize>>,
    pub json: Value,
}

/// Status words used by the report.
fn status(s: &Shell, nearest: Option<(usize, f32)>, overlap_share: f32, mismatch: Option<f32>, touch: f32) -> &'static str {
    if s.zero_thickness() && overlap_share >= 0.10 {
        return "LOOSE CARD on a surface";
    }
    match nearest {
        None => "alone",
        Some((_, d)) if d > touch => "LOOSE (not touching)",
        Some(_) => match mismatch {
            Some(m) if m > 0.10 => "TOUCHING, skinned differently (slides)",
            _ => "attached",
        },
    }
}

pub fn shell_report(model: &Model, pose: &[Vec<[f32; 3]>], touch: f32, tol: f32, reach: f32, size: usize) -> ShellReport {
    let sh = shells::find(model);
    let ts = shells::tri_shell(model, &sh);
    let near = shells::nearest(model, &sh, pose, touch);
    let base = Camera::fit(model, 1.0, 0.9, true);
    let n = sh.len();
    let mut visible = vec![0usize; n];
    let mut over = vec![vec![0usize; n]; n];
    let mut by_view = Vec::new();
    for (name, yaw) in shells::VIEWS {
        let cam = shells::ortho_cam(base.target, yaw, base.ortho_half);
        let l = shells::layers(model, pose, &|m, t| ts[m][t], &cam, size, size);
        let (v, o) = shells::overlap(&l, n, tol);
        for a in 0..n {
            visible[a] += v[a];
            for b in 0..n {
                over[a][b] += o[a][b];
            }
        }
        by_view.push((name, v, o));
    }
    let mut rows = Vec::new();
    for s in &sh {
        let nearest = (0..n).filter(|&b| b != s.id).min_by(|&a, &b| near[s.id][a].dist.total_cmp(&near[s.id][b].dist)).map(|b| (b, near[s.id][b].dist));
        let top_over = (0..n).filter(|&b| b != s.id).max_by_key(|&b| over[s.id][b]);
        let overlap_share = top_over.map_or(0.0, |b| over[s.id][b] as f32 / visible[s.id].max(1) as f32);
        let contact = nearest.map(|(b, _)| shells::skin_contact(model, &sh, s.id, b, reach));
        let mismatch = contact.as_ref().filter(|c| c.verts > 0).map(|c| c.skin_mismatch);
        let st = status(s, nearest, overlap_share, mismatch, touch);
        rows.push(json!({
            "shell": s.label, "id": s.id, "mesh": s.mesh, "triangles": s.tris.len(), "vertices": s.verts.len(),
            "first_triangle": s.tris.first(), "open": s.open(), "boundary_edges": s.boundary_edges, "nonmanifold_edges": s.nonmanifold_edges,
            "zero_thickness": s.zero_thickness(), "doubled_faces": s.doubled, "backed_share": s.backed, "thickness": s.thickness,
            "bounds": [s.lo, s.hi],
            "bones": s.bones.iter().take(4).map(|&(j, w)| json!({"joint": j, "name": joint_name(model, j), "share": w})).collect::<Vec<_>>(),
            "nearest": nearest.map(|(b, d)| json!({"shell": sh[b].label, "distance": d, "touching_vertices": near[s.id][b].touching, "edges_through_it": near[s.id][b].crossing})),
            "skin_at_contact": contact.as_ref().map(|c| json!({"shell": sh[c.other].label, "vertices_within_reach": c.verts, "mean_mismatch": c.skin_mismatch, "max_mismatch": c.skin_mismatch_max})),
            "screen": {
                "visible_px": visible[s.id],
                "lies_on": top_over.filter(|&b| over[s.id][b] > 0).map(|b| json!({"shell": sh[b].label, "px": over[s.id][b], "share_of_visible": overlap_share})),
                "by_view": by_view.iter().map(|(name, v, o)| json!({"view": name, "visible_px": v[s.id], "on_other_px": o[s.id].iter().sum::<usize>()})).collect::<Vec<_>>(),
            },
            "status": st,
        }));
    }
    let json = json!({
        "rules": {
            "touch": touch, "screen_tolerance": tol, "skin_reach": reach, "views": shells::VIEWS.iter().map(|v| v.0).collect::<Vec<_>>(), "view_px": size,
            "zero_thickness": "open shell where under 25% of triangles have a surface of the same shell straight behind or in front, or a shell made of back-to-back faces (half or more doubled, or median thickness under 0.002)",
            "lies_on": "pixels where this shell is nearest and another shell is less than screen_tolerance behind it",
            "skin_mismatch": "half the L1 distance between a vertex's weights and the weights of the nearest shell's surface under it (0 = moves with it)",
            "status": "LOOSE CARD = zero-thickness and lies on another shell over 10% of its visible pixels; LOOSE = nearest shell farther than touch (an edge passing through another shell's surface counts as distance 0); TOUCHING, skinned differently = mean mismatch over 0.10 within reach; else attached",
        },
        "shells": rows,
    });
    ShellReport { shells: sh, near, visible, over, json }
}

pub fn pose_for(model: &Model, args: &Args) -> Result<(Vec<Vec<[f32; 3]>>, String), String> {
    match args.opt("--clip") {
        Some(name) => {
            let c = model.clips.iter().position(|c| c == name).ok_or_else(|| format!("no clip named {name}; clips: {}", model.clips.join(", ")))?;
            let t: f32 = num(args, "--time", 0.0)?;
            Ok((model.posed(Some(c), t), format!("{name} at {t} s")))
        }
        None => Ok((model.posed(None, 0.0), "rest pose".into())),
    }
}

pub fn shells_cmd(args: &Args) -> Result<(), String> {
    let path = file_arg(args);
    let model = Model::open(&path)?;
    let (pose, pose_name) = pose_for(&model, args)?;
    let touch: f32 = num(args, "--touch", 0.005)?;
    let tol: f32 = num(args, "--tol", 0.05)?;
    let reach: f32 = num(args, "--reach", 0.04)?;
    let size: usize = num(args, "--size", 512)?;
    let t0 = std::time::Instant::now();
    let r = shell_report(&model, &pose, touch, tol, reach, size);
    let mut j = r.json;
    j["file"] = json!(path.display().to_string());
    j["pose"] = json!(pose_name);
    if let Some(out) = args.opt("--json-out") {
        std::fs::write(out, serde_json::to_string_pretty(&j).unwrap()).map_err(|e| format!("{out}: {e}"))?;
    }
    if args.flag("--json") {
        println!("{}", serde_json::to_string_pretty(&j).unwrap());
        return Ok(());
    }
    println!("{}  ({})", path.display(), pose_name);
    println!("{} shells; touch {touch}, screen tolerance {tol}, skin reach {reach}", r.shells.len());
    for row in j["shells"].as_array().unwrap() {
        let bones: Vec<String> = row["bones"].as_array().unwrap().iter().map(|b| format!("{} {:.0}%", b["name"].as_str().unwrap().trim_start_matches("bone"), 100.0 * b["share"].as_f64().unwrap())).collect();
        println!(
            "  {:<10} {:>5} tris {:>5} verts  {:<6} {:<15} {}",
            row["shell"].as_str().unwrap(),
            row["triangles"],
            row["vertices"],
            if row["doubled_faces"].as_u64().unwrap() * 2 >= row["triangles"].as_u64().unwrap() { "2-face" } else if row["open"].as_bool().unwrap() { "open" } else { "closed" },
            if row["zero_thickness"].as_bool().unwrap() { "zero-thickness".to_owned() } else { format!("thick {:.4}", row["thickness"].as_f64().unwrap_or(0.0)) },
            row["status"].as_str().unwrap(),
        );
        println!("      bones: {}", bones.join(", "));
        if let Some(nr) = row["nearest"].as_object() {
            println!("      nearest {} at {:.4} ({} vertices touching, {} edges pass through it)", nr["shell"].as_str().unwrap(), nr["distance"].as_f64().unwrap(), nr["touching_vertices"], nr["edges_through_it"]);
        }
        if let Some(c) = row["skin_at_contact"].as_object() {
            println!("      skin vs {}: {} vertices within reach, mismatch mean {:.3} max {:.3}", c["shell"].as_str().unwrap(), c["vertices_within_reach"], c["mean_mismatch"].as_f64().unwrap(), c["max_mismatch"].as_f64().unwrap());
        }
        let sc = &row["screen"];
        match sc["lies_on"].as_object() {
            Some(lo) => println!("      screen: {} px visible, lies on {} at {} px ({:.0}%)", sc["visible_px"], lo["shell"].as_str().unwrap(), lo["px"], 100.0 * lo["share_of_visible"].as_f64().unwrap()),
            None => println!("      screen: {} px visible, lies on nothing", sc["visible_px"]),
        }
    }
    println!("took {} ms", t0.elapsed().as_millis());
    let _ = (&r.near, &r.visible, &r.over);
    Ok(())
}

/// The strap shell (by label, default mesh0.s0) and the shell it hangs on (by label, default the
/// largest other shell of the same mesh).
fn pick_shells(sh: &[Shell], args: &Args) -> Result<(usize, usize), String> {
    let label = args.opt("--shell").unwrap_or("mesh0.s0");
    let strap = sh
        .iter()
        .position(|s| s.label == label)
        .ok_or_else(|| format!("no shell {label}; shells: {}", sh.iter().map(|s| s.label.as_str()).collect::<Vec<_>>().join(", ")))?;
    let body = match args.opt("--on") {
        Some(l) => sh.iter().position(|s| s.label == l).ok_or_else(|| format!("no shell {l}"))?,
        None => sh.iter().filter(|s| s.mesh == sh[strap].mesh && s.id != strap).max_by_key(|s| s.tris.len()).ok_or("no other shell in the strap's mesh")?.id,
    };
    Ok((strap, body))
}

fn put(sheet: &mut image::RgbaImage, x0: u32, y0: u32, rgba: &[u8], size: usize) {
    for y in 0..size {
        for x in 0..size {
            let o = (y * size + x) * 4;
            sheet.put_pixel(x0 + x as u32, y0 + y as u32, image::Rgba([rgba[o], rgba[o + 1], rgba[o + 2], 255]));
        }
    }
}

fn frame_box(sheet: &mut image::RgbaImage, x0: u32, y0: u32, size: u32, c: [u8; 3], t: u32) {
    for k in 0..t {
        for i in 0..size {
            for (x, y) in [(x0 + i, y0 + k), (x0 + i, y0 + size - 1 - k), (x0 + k, y0 + i), (x0 + size - 1 - k, y0 + i)] {
                sheet.put_pixel(x, y, image::Rgba([c[0], c[1], c[2], 255]));
            }
        }
    }
}

fn text(sheet: &mut image::RgbaImage, x: i64, y: i64, s: &str, scale: usize, c: [u8; 3]) {
    let (w, h) = (sheet.width() as usize, sheet.height() as usize);
    wb_core::font::draw(sheet.as_mut(), w, h, x, y, s, scale, c);
}

/// A point pinned to a waist triangle at rest: (triangle, barycentric, offset).
type Track = (usize, [f32; 3], [f32; 3]);

fn track_point(model: &Model, sh: &[Shell], body: usize, p: [f32; 3]) -> Track {
    let rest: Vec<Vec<[f32; 3]>> = model.meshes.iter().map(|m| m.pos.clone()).collect();
    let (_, t, b) = shells::closest_in_shell(model, &rest, &sh[body], p);
    let m = &model.meshes[sh[body].mesh];
    let q = at_track(m.tris[t].map(|v| m.pos[v as usize]), b);
    (t, b, wb_core::geom::sub(p, q))
}

fn at_track(p3: [[f32; 3]; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|k| b[0] * p3[0][k] + b[1] * p3[1][k] + b[2] * p3[2][k])
}

fn eval_track(model: &Model, sh: &[Shell], body: usize, pose: &[Vec<[f32; 3]>], tr: Track) -> [f32; 3] {
    let mi = sh[body].mesh;
    let p3 = model.meshes[mi].tris[tr.0].map(|v| pose[mi][v as usize]);
    wb_core::geom::add(at_track(p3, tr.1), tr.2)
}

struct Loaded {
    path: PathBuf,
    label: String,
    model: Model,
    textures: Vec<Option<wb_core::Texture>>,
    sh: Vec<Shell>,
    ts: Vec<Vec<usize>>,
    setup: wb_core::sweep::Setup,
}

type Samples = Vec<(String, f32, wb_core::sweep::Sample)>;

/// Close-up camera half height for the sweep sheets.
const HALF: f32 = 0.40;

pub fn sweep_cmd(args: &Args) -> Result<(), String> {
    use wb_core::sweep::{self, Rules, CLOSE_VIEWS};
    let files = file_args(args);
    if files.is_empty() || files.len() > 2 {
        return Err("strap-sweep needs one or two files: strap-sweep before.glb [after.glb] --out-dir DIR".into());
    }
    let out_dir = PathBuf::from(args.opt("--out-dir").ok_or("strap-sweep needs --out-dir")?);
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let labels: Vec<String> = match args.opt("--labels") {
        Some(l) => l.split(',').map(str::to_owned).collect(),
        None if files.len() == 2 => vec!["before".into(), "after".into()],
        None => vec!["model".into()],
    };
    if labels.len() != files.len() {
        return Err("--labels needs one label per file".into());
    }
    let clips: Vec<String> = args.opt("--clips").unwrap_or("idle,walk,run,turn_l,turn_r").split(',').map(str::to_owned).collect();
    let times: Vec<f32> = match args.opt("--times") {
        Some(t) => t.split(',').map(|x| x.parse::<f32>().map_err(|e| format!("--times: {e}"))).collect::<Result<_, _>>()?,
        None => (0..8).map(|k| k as f32 / 8.0).collect(),
    };
    let size: usize = num(args, "--size", 300)?;
    let measure_px = size * 2;
    let rules = Rules { reach: num(args, "--reach", 0.03)?, ..Rules::default() };
    let mut models: Vec<Loaded> = Vec::new();
    for (path, label) in files.iter().zip(&labels) {
        let model = Model::open(path)?;
        let textures = wb_core::image_textures(&model);
        let sh = shells::find(&model);
        let (strap, body) = pick_shells(&sh, args)?;
        let ts = shells::tri_shell(&model, &sh);
        let setup = sweep::Setup::new(&model, &sh, strap, body, &rules)?;
        println!(
            "{label}: {}  strap {} ({} tris), waist {}; {} top-edge verts, {} tip verts",
            path.display(),
            sh[strap].label,
            sh[strap].tris.len(),
            sh[body].label,
            setup.top.len(),
            setup.tip.len()
        );
        models.push(Loaded { path: path.clone(), label: label.clone(), model, textures, sh, ts, setup });
    }
    // Cameras follow the first file's waist, so every file is framed the same.
    let first = &models[0];
    let knot_track = {
        let m = &first.model.meshes[first.sh[first.setup.strap].mesh];
        let c = first.setup.top.iter().fold([0.0f32; 3], |a, &v| wb_core::geom::add(a, m.pos[v]));
        let c = wb_core::geom::scale(c, 1.0 / first.setup.top.len().max(1) as f32);
        track_point(&first.model, &first.sh, first.setup.body, [c[0], c[1] - 0.05, c[2]])
    };
    let full_cam = Camera::fit(&first.model, 1.0, 0.92, true);
    let mut all: Vec<Samples> = vec![Vec::new(); models.len()];
    let t0 = std::time::Instant::now();
    for clip in &clips {
        let ci = first.model.clips.iter().position(|c| c == clip).ok_or_else(|| format!("no clip {clip}; clips: {}", first.model.clips.join(", ")))?;
        let dur = first.model.clip_duration(ci);
        let cols = CLOSE_VIEWS.len() + 1;
        let (label_w, head_h) = (230u32, 46u32);
        let (sw, shh) = (label_w + (cols * size) as u32, head_h + (times.len() * size) as u32);
        for (mi, ld) in models.iter().enumerate() {
            let mut sheet = image::RgbaImage::from_pixel(sw, shh, image::Rgba([24, 26, 31, 255]));
            let fname = ld.path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
            text(&mut sheet, 8, 6, &format!("{} | {} | {}", ld.label, clip, fname), 2, [235, 235, 235]);
            for (k, (vn, _)) in CLOSE_VIEWS.iter().enumerate() {
                text(&mut sheet, label_w as i64 + (k * size) as i64 + 6, 28, vn, 2, [200, 200, 210]);
            }
            text(&mut sheet, label_w as i64 + (CLOSE_VIEWS.len() * size) as i64 + 6, 28, "full", 2, [200, 200, 210]);
            for (ri, &tn) in times.iter().enumerate() {
                let ts = tn * dur;
                let pose = ld.model.posed(Some(ci), ts);
                let target = if mi == 0 { first.setup.target(&first.model, &first.sh, &pose) } else { first.setup.target(&first.model, &first.sh, &first.model.posed(Some(ci), ts)) };
                let (mut views, mut vis) = (Vec::new(), Vec::new());
                for (vn, yaw) in CLOSE_VIEWS {
                    let cam = shells::ortho_cam(target, yaw, HALF);
                    let l = shells::layers(&ld.model, &pose, &|m, t| ld.ts[m][t], &cam, measure_px, measure_px);
                    views.push((vn.to_string(), l, cam));
                    let la = shells::layers_alpha(&ld.model, &ld.textures, &pose, &|m, t| ld.ts[m][t], &cam, measure_px, measure_px);
                    vis.push((vn.to_string(), la, cam));
                }
                let mut s = sweep::measure(&ld.model, &ld.sh, &ld.setup, &rules, &pose, &views, &vis);
                s.clip = clip.clone();
                s.time_norm = tn;
                s.time_s = ts;
                let y0 = head_h + (ri * size) as u32;
                let bad = !s.flags.is_empty();
                text(&mut sheet, 8, y0 as i64 + 8, &format!("t={tn:.3} ({ts:.2}s)"), 2, if bad { [255, 120, 110] } else { [140, 230, 140] });
                text(&mut sheet, 8, y0 as i64 + 30, &format!("slide {:.4}", s.slide_max), 1, [220, 220, 220]);
                text(&mut sheet, 8, y0 as i64 + 42, &format!("top edge {:+.4}", s.ledge_max), 1, [220, 220, 220]);
                text(&mut sheet, 8, y0 as i64 + 54, &format!("cut {} tip-in {}", s.cut, s.tip_in_body), 1, [220, 220, 220]);
                for (fi, f) in s.flags.iter().take(10).enumerate() {
                    text(&mut sheet, 8, y0 as i64 + 72 + 12 * fi as i64, &f.chars().take(36).collect::<String>(), 1, [255, 130, 120]);
                }
                if !bad {
                    text(&mut sheet, 8, y0 as i64 + 72, "PASS", 2, [140, 230, 140]);
                }
                let note_y = y0 as i64 + 96 + 12 * s.flags.len().min(10) as i64;
                for (ni, n) in s.notes.iter().enumerate() {
                    text(&mut sheet, 8, note_y + 12 * ni as i64, &n.chars().take(36).collect::<String>(), 1, [240, 200, 90]);
                }
                for (k, (vn, yaw)) in CLOSE_VIEWS.iter().enumerate() {
                    let cam = shells::ortho_cam(target, *yaw, HALF);
                    let px = sweep::picture(&ld.model, &ld.textures, &pose, cam, size, None);
                    let x0 = label_w + (k * size) as u32;
                    put(&mut sheet, x0, y0, &px, size);
                    let vs = &s.views[k];
                    let _ = vn;
                    let vflag = vs.top_shown > 0;
                    frame_box(&mut sheet, x0, y0, size as u32, if vflag { [230, 60, 50] } else { [50, 54, 62] }, if vflag { 3 } else { 1 });
                }
                let px = sweep::picture(&ld.model, &ld.textures, &pose, full_cam, size, None);
                let x0 = label_w + (CLOSE_VIEWS.len() * size) as u32;
                put(&mut sheet, x0, y0, &px, size);
                frame_box(&mut sheet, x0, y0, size as u32, [50, 54, 62], 1);
                all[mi].push((clip.clone(), tn, s));
            }
            let out = out_dir.join(format!("sweep-{}-{}.png", ld.label, clip));
            sheet.save(&out).map_err(|e| format!("{}: {e}", out.display()))?;
            println!("  wrote {}", out.display());
        }
    }
    for (mi, ld) in models.iter().enumerate() {
        let samples = &all[mi];
        let mut per_clip = Vec::new();
        for clip in &clips {
            let cs: Vec<&sweep::Sample> = samples.iter().filter(|x| &x.0 == clip).map(|x| &x.2).collect();
            let fails: Vec<String> = cs.iter().filter(|s| !s.flags.is_empty()).map(|s| format!("t={:.3}: {}", s.time_norm, s.flags.join("; "))).collect();
            let geo_fails: Vec<String> = cs.iter().filter(|s| !s.geo_flags.is_empty()).map(|s| format!("t={:.3}: {}", s.time_norm, s.geo_flags.join("; "))).collect();
            let vis_fails: Vec<String> = cs.iter().filter(|s| !s.vis_flags.is_empty()).map(|s| format!("t={:.3}: {}", s.time_norm, s.vis_flags.join("; "))).collect();
            per_clip.push(json!({
                "clip": clip, "samples": cs.len(), "failing_samples": fails.len(),
                "geometry_failing_samples": geo_fails.len(), "geometry_failures": geo_fails,
                "visible_failing_samples": vis_fails.len(), "visible_failures": vis_fails,
                "worst_slide": cs.iter().map(|s| s.slide_max).fold(0.0f32, f32::max),
                "worst_top_edge_gap": cs.iter().map(|s| s.ledge_max).fold(f32::MIN, f32::max),
                "cut_vertices": cs.iter().map(|s| s.cut).sum::<usize>(),
                "tip_in_body_vertices": cs.iter().map(|s| s.tip_in_body).sum::<usize>(),
                "failures": fails,
                "thigh_clip_samples": cs.iter().filter(|s| s.thigh_clip > 0).count(),
                "thigh_clip_max_depth": cs.iter().map(|s| s.thigh_clip_depth).fold(0.0f32, f32::max),
                "notes": cs.iter().filter(|s| !s.notes.is_empty()).map(|s| format!("t={:.3}: {}", s.time_norm, s.notes.join("; "))).collect::<Vec<_>>(),
            }));
        }
        let rows: Vec<Value> = samples
            .iter()
            .map(|(clip, tn, s)| {
                json!({
                    "clip": clip, "time_norm": tn, "time_s": s.time_s, "knot_slide_max": s.slide_max, "knot_slide_limit": s.slide_limit, "slide_at_down": s.slide_at, "fabric_slide_max": s.fabric_slide, "slide_p95": s.slide_p95, "top_edge_gap_max": s.ledge_max,
                    "cut_vertices": s.cut, "cut_depth": s.cut_depth, "cut_at_down": s.cut_at, "cut_vertices_list": s.cut_verts, "cut_against_joint": s.cut_joint.map(|j| joint_name(&ld.model, j)), "slide_vertex": s.slide_v, "tip_in_body_vertices": s.tip_in_body, "flags": s.flags, "geometry_flags": s.geo_flags, "visible_flags": s.vis_flags,
                    "thigh_clip_vertices": s.thigh_clip, "thigh_clip_depth": s.thigh_clip_depth, "thigh_clip_at_down": s.thigh_clip_at, "thigh_clip_joint": s.thigh_clip_joint.map(|j| joint_name(&ld.model, j)), "notes": s.notes,
                    "views": s.views.iter().map(|v| json!({"view": v.name, "strap_px": v.strap_px, "top_edge_verts_shown": v.top_shown, "tip_px": v.tip_px, "tip_px_hidden_by_body": v.tip_hidden_by_body})).collect::<Vec<_>>(),
                })
            })
            .collect();
        let total_fail = samples.iter().filter(|s| !s.2.flags.is_empty()).count();
        let j = json!({
            "file": ld.path.display().to_string(), "label": ld.label, "strap_shell": ld.sh[ld.setup.strap].label, "waist_shell": ld.sh[ld.setup.body].label,
            "rules": {
                "slide": format!("STEP when a visible strap vertex in the top {:.0}% of the strap (where its paint meets the knot paint on the waist), within {} of the waist at rest, slides along the waist surface off its waist point (in the waist triangle's frame; movement along the surface normal is the CUT and LEDGE tests' job) by more than one texel of the stock 512 texture ({:.4} here); slide lower down, where the strap lies over plain trouser fabric, is reported as fabric_slide_max and not flagged", 100.0 * rules.knot_share, rules.reach, ld.setup.texel),
                "sections": "geometry flags (STEP, LEDGE, CUT, TIP IN BODY) use positions only; visible flags (EDGE SHOWS, TIP HIDDEN) use alpha-aware layers, where a transparent texel (alpha < 128) is not a surface",
                "top_edge": "LEDGE (geometry) when any top-edge vertex stands in front of the waist surface (gap > 0); EDGE SHOWS (visible) when a top-edge vertex is the nearest opaque surface in a close-up",
                "cut": format!("CUT when a strap vertex outside the body at rest is inside it (winding number > 0.5) and more than {} behind the surface", rules.cut_depth),
                "thigh_clip": format!("THIGH CLIP (reported, not flagged): a strap vertex below the top {:.0}% that goes inside the body where a thigh joint leads the body surface, i.e. a raised knee passing through the hanging strap; the stock strap does the same", 100.0 * rules.knot_share),
                "tip": format!("TIP IN BODY when a tip vertex (lowest {:.0}% of the strap) goes inside the body the same way; TIP HIDDEN when the body covers over {:.0}% of the tip in all three front views (front, 3/4 right, 3/4 left)", 100.0 * rules.tip_share, 100.0 * (1.0 - rules.tip_visible_min)),
                "views": CLOSE_VIEWS.iter().map(|v| json!({"name": v.0, "yaw_deg": v.1})).collect::<Vec<_>>(), "close_up_half_height": HALF, "measure_px": measure_px,
            },
            "samples_total": samples.len(), "samples_failing": total_fail, "clips": per_clip, "samples": rows,
        });
        let out = out_dir.join(format!("sweep-{}.json", ld.label));
        std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).map_err(|e| format!("{}: {e}", out.display()))?;
        println!("{}: {} of {} samples flagged", ld.label, total_fail, samples.len());
        for pc in &per_clip {
            println!(
                "  {:<7} geometry flagged {}/{}, visible flagged {}/{}  worst slide {:.4}  worst top-edge gap {:+.4}  cut {}  tip-in {}  thigh-clip samples {} (to {:.3})",
                pc["clip"].as_str().unwrap(),
                pc["geometry_failing_samples"],
                pc["samples"],
                pc["visible_failing_samples"],
                pc["samples"],
                pc["worst_slide"].as_f64().unwrap(),
                pc["worst_top_edge_gap"].as_f64().unwrap(),
                pc["cut_vertices"],
                pc["tip_in_body_vertices"],
                pc["thigh_clip_samples"],
                pc["thigh_clip_max_depth"].as_f64().unwrap()
            );
            for f in pc["geometry_failures"].as_array().unwrap().iter().take(3) {
                println!("      geometry {}", f.as_str().unwrap());
            }
            for f in pc["visible_failures"].as_array().unwrap().iter().take(3) {
                println!("      visible  {}", f.as_str().unwrap());
            }
        }
    }
    if models.len() == 2 {
        key_poses(&models, &all, &clips, &out_dir, knot_track)?;
    }
    println!("took {:.1} s", t0.elapsed().as_secs_f32());
    Ok(())
}

/// Before/after side by side at each clip's worst pose for the first file, plus run at 0.75.
fn key_poses(models: &[Loaded], all: &[Samples], clips: &[String], out_dir: &Path, knot_track: Track) -> Result<(), String> {
    let mut picks: Vec<(String, f32)> = Vec::new();
    for clip in clips {
        if let Some(w) = all[0].iter().filter(|s| &s.0 == clip).max_by(|a, b| a.2.slide_max.total_cmp(&b.2.slide_max)) {
            picks.push((clip.clone(), w.1));
        }
    }
    if clips.iter().any(|c| c == "run") && !picks.iter().any(|p| p.0 == "run" && (p.1 - 0.75).abs() < 1e-3) {
        picks.push(("run".into(), 0.75));
    }
    let first = &models[0];
    let size = 340usize;
    // (title, yaw, half height, centered on the knot)
    let cols: [(&str, f32, f32, bool); 4] = [("knot 3/4 right", -40.0, 0.15, true), ("knot front", 0.0, 0.15, true), ("waist front", 0.0, HALF, false), ("waist right", -90.0, HALF, false)];
    let (label_w, head_h) = (190u32, 50u32);
    let w = label_w + (cols.len() * 2 * size) as u32;
    let h = head_h + (picks.len() * size) as u32;
    let mut sheet = image::RgbaImage::from_pixel(w, h, image::Rgba([24, 26, 31, 255]));
    text(&mut sheet, 8, 6, &format!("key poses: {} (left of each pair) vs {} (right, blue frame)", models[0].label, models[1].label), 2, [235, 235, 235]);
    for (k, c) in cols.iter().enumerate() {
        text(&mut sheet, label_w as i64 + (k * 2 * size) as i64 + 6, 30, c.0, 2, [200, 200, 210]);
    }
    for (ri, (clip, tn)) in picks.iter().enumerate() {
        let ci = first.model.clips.iter().position(|c| c == clip).unwrap();
        let ts = tn * first.model.clip_duration(ci);
        let fpose = first.model.posed(Some(ci), ts);
        let mid = first.setup.target(&first.model, &first.sh, &fpose);
        let knot = eval_track(&first.model, &first.sh, first.setup.body, &fpose, knot_track);
        let y0 = head_h + (ri * size) as u32;
        text(&mut sheet, 8, y0 as i64 + 10, clip, 2, [235, 235, 235]);
        text(&mut sheet, 8, y0 as i64 + 32, &format!("t={tn:.3}"), 2, [235, 235, 235]);
        for (mi, ld) in models.iter().enumerate() {
            if let Some((_, _, s)) = all[mi].iter().find(|s| &s.0 == clip && (s.1 - tn).abs() < 1e-4) {
                let c = if s.flags.is_empty() { [140, 230, 140] } else { [255, 120, 110] };
                text(&mut sheet, 8, y0 as i64 + 60 + 40 * mi as i64, &format!("{}: slide {:.4}", ld.label, s.slide_max), 1, c);
                text(&mut sheet, 8, y0 as i64 + 72 + 40 * mi as i64, &format!("top {:+.4} {}", s.ledge_max, if s.flags.is_empty() { "PASS" } else { "FLAGGED" }), 1, c);
            }
            let pose = ld.model.posed(Some(ci), ts);
            for (k, c) in cols.iter().enumerate() {
                let cam = shells::ortho_cam(if c.3 { knot } else { mid }, c.1, c.2);
                let px = wb_core::sweep::picture(&ld.model, &ld.textures, &pose, cam, size, None);
                let x0 = label_w + ((k * 2 + mi) * size) as u32;
                put(&mut sheet, x0, y0, &px, size);
                frame_box(&mut sheet, x0, y0, size as u32, if mi == 0 { [70, 74, 84] } else { [90, 140, 220] }, if mi == 0 { 1 } else { 2 });
            }
        }
    }
    let out = out_dir.join("key-poses.png");
    sheet.save(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    println!("  wrote {}", out.display());
    Ok(())
}

pub fn rebuild_cmd(args: &Args) -> Result<(), String> {
    use wb_core::strap::{self, Params};
    let path = file_arg(args);
    let out = PathBuf::from(args.opt("--out").ok_or("strap-rebuild needs --out new.glb")?);
    if out.exists() {
        return Err(format!("{} exists; pick a new name", out.display()));
    }
    let model = Model::open(&path)?;
    let sh = shells::find(&model);
    let (card, body) = pick_shells(&sh, args)?;
    if !sh[card].zero_thickness() {
        return Err(format!("{} is not a zero-thickness card; nothing to rebuild", sh[card].label));
    }
    let d = Params::default();
    let p = Params {
        tip_thigh: num(args, "--tip-thigh", d.tip_thigh)?,
        thickness: num(args, "--thickness", d.thickness)?,
        ride: num(args, "--ride", d.ride)?,
        tuck_depth: num(args, "--tuck-depth", d.tuck_depth)?,
        columns: num(args, "--columns", d.columns)?,
        row_step: num(args, "--row-step", d.row_step)?,
        fit_frames: num(args, "--fit-frames", d.fit_frames)?,
        fit_reach: num(args, "--fit-reach", d.fit_reach)?,
        collide: !args.flag("--no-collide"),
        max_thigh_share: num(args, "--max-thigh-share", d.max_thigh_share)?,
        exit_by_paint: !args.flag("--no-paint-exit"),
        thigh_joint: num(args, "--thigh-joint", d.thigh_joint)?,
        clear_gap: num(args, "--clear-gap", d.clear_gap)?,
        tuck_len: num(args, "--tuck-len", d.tuck_len)?,
        lift_len: num(args, "--lift-len", d.lift_len)?,
        exit_max: num(args, "--exit-max", d.exit_max)?,
        exit_raise: num(args, "--exit-raise", d.exit_raise)?,
        exit_down: match args.opt("--exit-down") {
            Some(s) => s.split(',').map(|x| x.trim().parse::<f32>().map_err(|e| format!("--exit-down {x}: {e}"))).collect::<Result<Vec<_>, _>>()?,
            None => Vec::new(),
        },
        flip_root: args.flag("--flip-root"),
        exit_edge: match args.opt("--exit-edge") {
            Some(s) => s.split(',').map(|x| x.trim().parse::<usize>().map_err(|e| format!("--exit-edge {x}: {e}"))).collect::<Result<Vec<_>, _>>()?,
            None => Vec::new(),
        },
        exit_curve: match args.opt("--exit-curve") {
            Some(s) => s
                .split(';')
                .filter(|x| !x.trim().is_empty())
                .map(|x| {
                    let c: Vec<f32> = x.split(',').map(|y| y.trim().parse::<f32>().map_err(|e| format!("--exit-curve {x}: {e}"))).collect::<Result<_, _>>()?;
                    if c.len() != 3 {
                        return Err(format!("--exit-curve point {x}: want x,y,z"));
                    }
                    Ok([c[0], c[1], c[2]])
                })
                .collect::<Result<Vec<_>, String>>()?,
            None => Vec::new(),
        },
        ..d
    };
    println!("{}: card {} ({} tris) on {}", path.display(), sh[card].label, sh[card].tris.len(), sh[body].label);
    // Refuse before building anything if a vertex attribute cannot be carried to new vertices.
    for l in strap::preflight(&model, sh[card].mesh).map_err(|e| format!("preflight: {e}"))? {
        println!("  preflight: {l}");
    }
    println!("params: {p:?}");
    let built = strap::build(&model, &sh[card], &sh[body], &p)?;
    for l in &built.log {
        println!("  {l}");
    }
    let bytes = strap::write(&model, sh[card].mesh, &built.verts, &built.tris)?;
    let nm = Model::from_bytes(&bytes)?;
    for l in strap::verify(&model, &nm, sh[card].mesh, &sh[card].tris, &built)? {
        println!("  check: {l}");
    }
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(&out, &bytes).map_err(|e| format!("{}: {e}", out.display()))?;
    let m = &nm.meshes[sh[card].mesh];
    let mut prof: Vec<(f32, String)> = Vec::new();
    for &(v, a, face) in &built.meta {
        if face != "front" || prof.iter().any(|p| (p.0 - a).abs() < 1e-3) {
            continue;
        }
        let ws = shells::vertex_weights(m, v as usize);
        prof.push((a, ws.iter().map(|&(j, w)| format!("{} {:.2}", joint_name(&nm, j).trim_start_matches("bone"), w)).collect::<Vec<_>>().join(" + ")));
    }
    prof.sort_by(|a, b| a.0.total_cmp(&b.0));
    println!("  weights down the strap (first front point of each row):");
    for (a, d) in &prof {
        println!("    {:>6.3}  {}", a, d);
    }
    println!(
        "wrote {} ({} bytes): {} strap vertices, {} strap triangles, length {:.4}, tip at {:.4}",
        out.display(),
        bytes.len(),
        built.strap_verts,
        built.strap_tris,
        built.length,
        built.tip_a
    );
    Ok(())
}
