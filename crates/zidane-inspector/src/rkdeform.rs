//! `gaw deform`: play a stock model and a candidate through every animation clip with rigkit's
//! skin-stretch check and print the two side by side: worst edge change, worst area change,
//! triangles that turn inside out, and the worst triangles by name. Reads GLBs through wb-core,
//! hands rigkit plain arrays, writes nothing unless --render-dir names a new folder.
use crate::Args;
use rigkit::{check, Mat4, Skin, Vec3};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use wb_core::render::{self, Camera, Filter};
use wb_core::{analysis, anim, font, image_textures, Model};

pub const RKDEFORM_USAGE: &str = "  gaw deform STOCK.glb CAND.glb [--clips all|a,b] [--steps N] [--top N] [--min-edge E] [--all-meshes] [--json] [--strict] [--render-dir NEW_DIR [--size PX]]
      Skin pinch and fold check built on rigkit. Poses STOCK and CAND through every clip (CAND with no clip
      of that name, or no clips at all, is posed with STOCK's clip on the same joint order) and, against
      each model's OWN rest pose, measures per clip: the worst edge ratio (always >= 1; `pinch` = the edge
      got shorter, `pull` = longer), the worst area ratio, and triangles that turn inside out in at least
      one pose. Prints a summary row per clip (stock | candidate | WORSE, same or better) and, per clip, the
      worst triangles of each side with mesh, triangle number, dominant joint, pose time, ratios.
      `flipped` is shown as all/hard: a hard flip also changed its worst edge by 1.5x or more; a flip with
      edges barely changed is a near-degenerate sliver, not a fold.
      Verdict per clip: WORSE when CAND has more hard flips than STOCK or its worst edge ratio is more
      than 1.25x STOCK's; better when it is under 0.8x with no more hard flips; otherwise same.
      --steps N       poses per clip (default: 30 per second of the clip, 8 to 240), evenly spaced
      --top N         worst triangles listed per side per clip (default 5)
      --min-edge E    ignore edges shorter than E at rest, same number for both models (default 0.2% of
                      STOCK's height); welds and slivers have meaningless ratios
      --all-meshes    also check meshes named battle_model* (skipped by default)
      --json          the whole result as JSON on stdout instead of the table
      --strict        exit 1 when any clip is WORSE
      --render-dir D  also writes D/deform-stock.png and D/deform-cand.png: per clip the worst triangle
                      (and the worst flipped one) outlined in red, posed on the left and at rest on the
                      right, same camera. D must be new or empty. Inputs are never changed.";

/// A flipped triangle counts as a hard flip when its worst edge changed at least this much.
pub const HARD_FLIP: f32 = 1.5;

const IDENTITY: Mat4 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

/// One mesh as rigkit sees it. Every vertex has a valid joint: vertices with no weight, or a joint
/// the skeleton lacks, are tied to an extra identity bone (index `bones`) so they stay at rest.
pub struct MeshIn {
    pub mesh: usize,
    pub name: String,
    pub rest: Vec<Vec3>,
    pub indices: Vec<u32>,
    pub skin: Skin,
    pub unweighted: usize,
    pub bad_joint: usize,
}

/// Weights normalized to sum 1; unweighted vertices and joints >= `bones` go to the identity bone.
/// Returns the skin, how many vertices had no weight and how many named a missing joint.
pub fn make_skin(joints: &[[u16; 4]], weights: &[[f32; 4]], bones: usize) -> (Skin, usize, usize) {
    let (mut unweighted, mut bad) = (0, 0);
    let mut skin = Skin::default();
    for (j, w) in joints.iter().zip(weights) {
        let (mut jj, mut ww) = (*j, *w);
        let mut was_bad = false;
        for k in 0..4 {
            if ww[k] > 0.0 && jj[k] as usize >= bones {
                ww[k] = 0.0;
                was_bad = true;
            }
            if ww[k] < 0.0 || !ww[k].is_finite() {
                ww[k] = 0.0;
            }
        }
        let sum: f32 = ww.iter().sum();
        if was_bad {
            bad += 1;
        }
        if sum <= 1e-6 {
            if !was_bad {
                unweighted += 1;
            }
            jj = [bones as u16, 0, 0, 0];
            ww = [1.0, 0.0, 0.0, 0.0];
        } else {
            for k in 0..4 {
                ww[k] /= sum;
                if ww[k] == 0.0 {
                    jj[k] = 0;
                }
            }
        }
        skin.joints.push(jj);
        skin.weights.push(ww);
    }
    (skin, unweighted, bad)
}

pub fn bones_of(model: &Model) -> usize {
    model.rig.as_ref().map_or(model.joint_names.len(), |r| r.joints.len())
}

pub fn mesh_inputs(model: &Model, all_meshes: bool) -> Vec<MeshIn> {
    let bones = bones_of(model);
    model
        .meshes
        .iter()
        .enumerate()
        .filter(|(_, m)| all_meshes || !m.name.starts_with("battle_model"))
        .map(|(mi, m)| {
            let (skin, unweighted, bad_joint) = make_skin(&m.joints, &m.weights, bones);
            MeshIn { mesh: mi, name: m.name.clone(), rest: m.pos.clone(), indices: m.tris.iter().flatten().copied().collect(), skin, unweighted, bad_joint }
        })
        .collect()
}

/// The poses of one clip: one matrix per joint per step, rest to posed, from the model's own rig.
pub struct PoseSet {
    pub times: Vec<f32>,
    pub mats: Vec<Vec<Mat4>>,
    /// True when the model has no clip of this name and was posed with the stock's.
    pub borrowed: bool,
}

pub fn pose_set(model: &Model, stock: &Model, clip: &str, steps: usize) -> Result<PoseSet, String> {
    let rig = model.rig.as_ref().ok_or("this model has no skeleton, so it cannot be posed (run `gaw weights --adopt-skeleton` first)")?;
    let stock_ci = stock.clips.iter().position(|c| c == clip).ok_or_else(|| format!("stock has no clip `{clip}`"))?;
    let own_ci = model.clips.iter().position(|c| c == clip);
    let duration = match own_ci {
        Some(ci) => model.clip_duration(ci),
        None => stock.clip_duration(stock_ci),
    };
    let steps = steps.max(1);
    let times: Vec<f32> = (0..steps).map(|k| duration * k as f32 / steps as f32).collect();
    let mats = match own_ci {
        Some(ci) => times.iter().map(|&t| rig.joint_matrices(Some(ci), t)).collect(),
        None => {
            let srig = stock.rig.as_ref().ok_or("stock has no skeleton")?;
            if srig.joints.len() != rig.joints.len() {
                return Err(format!("clip `{clip}` is only in stock and the models have {} and {} joints, so it cannot be borrowed", srig.joints.len(), rig.joints.len()));
            }
            times.iter().map(|&t| rig.joint_matrices_with(&srig.joint_locals(Some(stock_ci), t))).collect()
        }
    };
    Ok(PoseSet { times, mats, borrowed: own_ci.is_none() })
}

#[derive(Clone, Debug)]
pub struct RawHit {
    pub mesh: usize,
    pub tri: usize,
    pub pose: usize,
    pub edge: f32,
    pub shrunk: bool,
    pub area: f32,
    pub flipped: bool,
}

/// One mesh's own numbers, so a hair mesh is judged apart from the body.
#[derive(Clone, Debug)]
pub struct MeshStat {
    pub name: String,
    pub worst_edge: f32,
    pub worst_shrunk: bool,
    pub worst_area: f32,
    pub flipped: usize,
    pub counted: usize,
}

#[derive(Clone, Debug, Default)]
pub struct RawSide {
    pub worst_edge: f32,
    pub worst_shrunk: bool,
    pub worst_area: f32,
    pub flipped: usize,
    /// Of those, the ones whose worst edge also changed by at least HARD_FLIP. A near-degenerate sliver can
    /// flip with its edges barely changed (1.1x); that is noise, not a fold.
    pub hard_flipped: usize,
    /// Flips rigkit counts but cannot name: it only records a triangle's flag when an edge changed, so
    /// a flip with every edge exactly unchanged (a pure mirror) is in `flipped` and not in `flipped_hits`.
    /// Such a flip is never a hard flip.
    pub flipped_unlisted: usize,
    /// Triangles long enough at rest to be measured.
    pub counted: usize,
    pub hits: Vec<RawHit>,
    /// Triangles that flipped, worst edge first (the pose is where each one flipped hardest).
    pub flipped_hits: Vec<RawHit>,
    pub per_mesh: Vec<MeshStat>,
}

fn with_identity(poses: &[Vec<Mat4>]) -> Vec<Vec<Mat4>> {
    poses.iter().map(|p| p.iter().copied().chain([IDENTITY]).collect()).collect()
}

fn counted_tris(m: &MeshIn, min_edge: f32) -> usize {
    m.indices
        .chunks_exact(3)
        .filter(|t| {
            let (a, b, c) = (m.rest[t[0] as usize], m.rest[t[1] as usize], m.rest[t[2] as usize]);
            let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
            let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() >= min_edge * min_edge
        })
        .count()
}

/// rigkit's stretch check over every mesh and every pose of one clip, merged.
pub fn run_stretch(meshes: &[MeshIn], poses: &[Vec<Mat4>], min_edge: f32, top: usize) -> RawSide {
    let poses = with_identity(poses);
    let mut side = RawSide { worst_edge: 1.0, worst_area: 1.0, ..Default::default() };
    for (k, m) in meshes.iter().enumerate() {
        let counted = counted_tris(m, min_edge);
        side.counted += counted;
        let rep = check::stretch(&m.rest, &m.indices, &m.skin, &poses, min_edge, top);
        side.per_mesh.push(MeshStat { name: m.name.clone(), worst_edge: rep.worst_edge, worst_shrunk: rep.hits.first().is_some_and(|h| h.shrunk), worst_area: rep.worst_area, flipped: rep.flipped, counted });
        if rep.worst_edge > side.worst_edge {
            side.worst_edge = rep.worst_edge;
            side.worst_shrunk = rep.hits.first().is_some_and(|h| h.shrunk);
        }
        side.worst_area = side.worst_area.max(rep.worst_area);
        side.flipped += rep.flipped;
        side.hits.extend(rep.hits.iter().map(|h| RawHit { mesh: k, tri: h.tri, pose: h.pose, edge: h.edge, shrunk: h.shrunk, area: h.area, flipped: h.flipped }));
        if rep.flipped > 0 {
            // The report only says how many flipped. Run pose by pose to learn which ones.
            let n_tris = m.indices.len() / 3;
            let mut found: Vec<RawHit> = Vec::new();
            for (p, pose) in poses.iter().enumerate() {
                let one = check::stretch(&m.rest, &m.indices, &m.skin, std::slice::from_ref(pose), min_edge, n_tris);
                for h in one.hits.iter().filter(|h| h.flipped) {
                    let hit = RawHit { mesh: k, tri: h.tri, pose: p, edge: h.edge, shrunk: h.shrunk, area: h.area, flipped: true };
                    match found.iter_mut().find(|f| f.tri == h.tri) {
                        Some(f) if f.edge >= hit.edge => {}
                        Some(f) => *f = hit,
                        None => found.push(hit),
                    }
                }
            }
            side.flipped_hits.extend(found);
        }
    }
    side.flipped_unlisted = side.flipped.saturating_sub(side.flipped_hits.len());
    side.hard_flipped = side.flipped_hits.iter().filter(|h| h.edge >= HARD_FLIP).count();
    side.hits.sort_by(|a, b| b.edge.total_cmp(&a.edge));
    side.hits.truncate(top);
    side.flipped_hits.sort_by(|a, b| b.edge.total_cmp(&a.edge));
    side
}

/// WORSE, same or better, from the two sides' numbers.
pub fn verdict(stock: &RawSide, cand: &RawSide) -> &'static str {
    if cand.hard_flipped > stock.hard_flipped || cand.worst_edge > stock.worst_edge * 1.25 {
        "WORSE"
    } else if cand.worst_edge < stock.worst_edge * 0.8 && cand.hard_flipped <= stock.hard_flipped {
        "better"
    } else {
        "same"
    }
}

fn parse(a: &[String]) -> (Vec<String>, Vec<(String, String)>, Vec<String>) {
    let valued = ["--clips", "--steps", "--top", "--min-edge", "--render-dir", "--size"];
    let (mut pos, mut opts, mut flags) = (Vec::new(), Vec::new(), Vec::new());
    let mut i = 0;
    while i < a.len() {
        if valued.contains(&a[i].as_str()) {
            if let Some(v) = a.get(i + 1) {
                opts.push((a[i].clone(), v.clone()));
            }
            i += 2;
        } else if a[i].starts_with("--") {
            flags.push(a[i].clone());
            i += 1;
        } else {
            pos.push(a[i].clone());
            i += 1;
        }
    }
    (pos, opts, flags)
}

/// A hit with the names a person reads.
fn hit_json(model: &Model, ps: &PoseSet, h: &RawHit, idx: &[MeshIn]) -> Value {
    let mi = idx[h.mesh].mesh;
    let joint = model.joint_names.get(model.triangle_joint(mi, h.tri)).cloned().unwrap_or_default();
    json!({
        "mesh": idx[h.mesh].name, "triangle": h.tri, "joint": joint, "region": analysis::region(&joint),
        "pose": h.pose, "time": ps.times.get(h.pose).copied().unwrap_or(0.0),
        "edge": h.edge, "direction": if h.shrunk { "pinch" } else { "pull" }, "area": h.area, "flipped": h.flipped,
    })
}

fn side_json(model: &Model, ps: &PoseSet, s: &RawSide, idx: &[MeshIn]) -> Value {
    json!({
        "poses": ps.mats.len(), "borrowed_clip": ps.borrowed,
        "worst_edge": s.worst_edge, "worst_edge_direction": if s.worst_shrunk { "pinch" } else { "pull" },
        "worst_area": s.worst_area, "flipped": s.flipped, "hard_flipped": s.hard_flipped, "flipped_unlisted": s.flipped_unlisted, "counted_triangles": s.counted,
        "by_mesh": s.per_mesh.iter().map(|m| json!({ "mesh": m.name, "worst_edge": m.worst_edge, "worst_edge_direction": dir_word(m.worst_shrunk), "worst_area": m.worst_area, "flipped": m.flipped, "counted_triangles": m.counted })).collect::<Vec<_>>(),
        "worst": s.hits.iter().map(|h| hit_json(model, ps, h, idx)).collect::<Vec<_>>(),
        "flipped_triangles": s.flipped_hits.iter().take(s.hits.len().max(5)).map(|h| hit_json(model, ps, h, idx)).collect::<Vec<_>>(),
    })
}

/// Everything `gaw deform` finds, with no printing. Also used by the tests.
pub struct Run {
    pub json: Value,
    pub worse: usize,
    pub rows: Vec<ClipRow>,
}

pub struct ClipRow {
    pub clip: String,
    pub stock: RawSide,
    pub cand: RawSide,
    pub stock_poses: PoseSet,
    pub cand_poses: PoseSet,
}

pub struct Settings {
    pub clips: Option<Vec<String>>,
    pub steps: Option<usize>,
    pub top: usize,
    pub min_edge: Option<f32>,
    pub all_meshes: bool,
}

pub fn run(stock_path: &Path, cand_path: &Path, set: &Settings, stock: &Model, cand: &Model) -> Result<(Run, Vec<MeshIn>, Vec<MeshIn>), String> {
    let (lo, hi) = render::bounds(stock);
    let height = hi[1] - lo[1];
    let min_edge = set.min_edge.unwrap_or(0.002 * height);
    let stock_in = mesh_inputs(stock, set.all_meshes);
    let cand_in = mesh_inputs(cand, set.all_meshes);
    if stock_in.is_empty() || cand_in.is_empty() {
        return Err("a model has no meshes to check".into());
    }
    let clips: Vec<String> = match &set.clips {
        Some(c) => c.clone(),
        None => stock.clips.clone(),
    };
    if clips.is_empty() {
        return Err("stock has no animation clips".into());
    }
    let mut rows = Vec::new();
    let mut out_clips = Vec::new();
    let mut worse = 0;
    for clip in &clips {
        let ci = stock.clips.iter().position(|c| c == clip).ok_or_else(|| format!("stock has no clip `{clip}` (clips: {})", stock.clips.join(", ")))?;
        let steps = set.steps.unwrap_or_else(|| ((stock.clip_duration(ci) * 30.0).round() as usize).clamp(8, 240));
        let sp = pose_set(stock, stock, clip, steps)?;
        let cp = pose_set(cand, stock, clip, steps)?;
        let s = run_stretch(&stock_in, &sp.mats, min_edge, set.top);
        let c = run_stretch(&cand_in, &cp.mats, min_edge, set.top);
        let v = verdict(&s, &c);
        if v == "WORSE" {
            worse += 1;
        }
        out_clips.push(json!({
            "clip": clip, "verdict": v,
            "stock": side_json(stock, &sp, &s, &stock_in), "candidate": side_json(cand, &cp, &c, &cand_in),
            "delta": { "worst_edge": c.worst_edge - s.worst_edge, "worst_area": c.worst_area - s.worst_area, "flipped": c.flipped as i64 - s.flipped as i64, "hard_flipped": c.hard_flipped as i64 - s.hard_flipped as i64 },
        }));
        rows.push(ClipRow { clip: clip.clone(), stock: s, cand: c, stock_poses: sp, cand_poses: cp });
    }
    let unweighted = |m: &[MeshIn]| -> Value { json!({ "no_weight": m.iter().map(|x| x.unweighted).sum::<usize>(), "missing_joint": m.iter().map(|x| x.bad_joint).sum::<usize>() }) };
    let json = json!({
        "stock": stock_path.display().to_string(), "candidate": cand_path.display().to_string(),
        "min_edge": min_edge, "stock_height": height, "top": set.top,
        "stock_meshes": stock_in.iter().map(|m| m.name.clone()).collect::<Vec<_>>(),
        "candidate_meshes": cand_in.iter().map(|m| m.name.clone()).collect::<Vec<_>>(),
        "vertex_problems": { "stock": unweighted(&stock_in), "candidate": unweighted(&cand_in) },
        "rule": "ratios are at least 1; pinch = worst edge got shorter, pull = longer; flipped = triangle turned inside out in at least one pose; hard_flipped = of those, worst edge changed 1.5x or more (the rest are near-degenerate slivers); verdict WORSE = more hard flips than stock or worst edge over 1.25x stock's",
        "clips": out_clips, "worse_clips": worse,
    });
    Ok((Run { json, worse, rows }, stock_in, cand_in))
}

fn unlisted(s: &RawSide) -> String {
    if s.flipped_unlisted > 0 { format!(", {} with unchanged edges not nameable", s.flipped_unlisted) } else { String::new() }
}

fn dir_word(shrunk: bool) -> &'static str {
    if shrunk { "pinch" } else { "pull" }
}

fn print_hits(label: &str, model: &Model, ps: &PoseSet, hits: &[RawHit], idx: &[MeshIn], what: &str) {
    if hits.is_empty() {
        println!("    {label}: no {what}");
        return;
    }
    println!("    {label}: {what}");
    for h in hits {
        let mi = idx[h.mesh].mesh;
        let joint = model.joint_names.get(model.triangle_joint(mi, h.tri)).cloned().unwrap_or_default();
        println!(
            "      {:<9} tri {:<6} {:<14} t={:>5.2}s  edge {:>6.2}x {:<5} area {:>7.2}x{}",
            idx[h.mesh].name, h.tri, joint, ps.times.get(h.pose).copied().unwrap_or(0.0), h.edge, dir_word(h.shrunk), h.area, if h.flipped { "  FLIPPED" } else { "" }
        );
    }
}

pub fn rkdeform_cmd(args: &Args) -> Result<(), String> {
    let (pos, opts, flags) = parse(&args.0);
    let opt = |k: &str| opts.iter().find(|o| o.0 == k).map(|o| o.1.as_str());
    let flag = |k: &str| flags.iter().any(|f| f == k);
    if pos.len() != 2 {
        return Err("usage: gaw deform STOCK.glb CAND.glb [options] (see `gaw help`)".into());
    }
    let num = |k: &str| -> Result<Option<f64>, String> { opt(k).map(|s| s.parse::<f64>().map_err(|_| format!("{k}: `{s}` is not a number"))).transpose() };
    let set = Settings {
        clips: opt("--clips").filter(|c| *c != "all").map(|c| c.split(',').map(|s| s.trim().to_owned()).collect()),
        steps: num("--steps")?.map(|v| v as usize),
        top: num("--top")?.map_or(5, |v| v as usize).max(1),
        min_edge: num("--min-edge")?.map(|v| v as f32),
        all_meshes: flag("--all-meshes"),
    };
    let render_dir = opt("--render-dir").map(PathBuf::from);
    if let Some(d) = &render_dir {
        if d.exists() && d.read_dir().map_err(|e| format!("{}: {e}", d.display()))?.next().is_some() {
            return Err(format!("--render-dir {} exists and is not empty; pass a new folder", d.display()));
        }
    }
    let (sp, cp) = (PathBuf::from(&pos[0]), PathBuf::from(&pos[1]));
    let stock = Model::open(&sp)?;
    let cand = Model::open(&cp)?;
    let (run, stock_in, cand_in) = run(&sp, &cp, &set, &stock, &cand)?;
    if let Some(d) = &render_dir {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        let size = num("--size")?.map_or(300, |v| v as usize).clamp(120, 800);
        write_sheet(&d.join("deform-stock.png"), "STOCK", &stock, &stock_in, &run.rows, true, size)?;
        write_sheet(&d.join("deform-cand.png"), "CANDIDATE", &cand, &cand_in, &run.rows, false, size)?;
    }
    if flag("--json") {
        println!("{}", serde_json::to_string_pretty(&run.json).unwrap());
    } else {
        print_table(&sp, &cp, &set, &run, &stock, &cand, &stock_in, &cand_in);
        if let Some(d) = &render_dir {
            println!("pictures: {}", d.display());
        }
    }
    if flag("--strict") && run.worse > 0 {
        std::process::exit(1);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn print_table(sp: &Path, cp: &Path, set: &Settings, run: &Run, stock: &Model, cand: &Model, stock_in: &[MeshIn], cand_in: &[MeshIn]) {
    println!("stock      {}", sp.display());
    println!("candidate  {}", cp.display());
    println!("min edge   {:.4} (edges shorter than this at rest are ignored, same for both)", run.json["min_edge"].as_f64().unwrap_or(0.0));
    for (label, m, ins) in [("stock", stock, stock_in), ("candidate", cand, cand_in)] {
        let (nw, bj) = (ins.iter().map(|x| x.unweighted).sum::<usize>(), ins.iter().map(|x| x.bad_joint).sum::<usize>());
        println!(
            "{label:<10} meshes {}  triangles {}  joints {}{}",
            ins.iter().map(|x| x.name.as_str()).collect::<Vec<_>>().join(", "),
            ins.iter().map(|x| x.indices.len() / 3).sum::<usize>(),
            bones_of(m),
            if nw + bj > 0 { format!("  (vertices with no weight: {nw}, with a missing joint: {bj}; held at rest)") } else { String::new() }
        );
    }
    println!();
    println!("{:<8} {:>5} | {:>13} {:>13} {:>9} | {:>13} {:>13} {:>9} | verdict   (flipped = all/hard)", "clip", "poses", "stock edge", "stock area", "flipped", "cand edge", "cand area", "flipped");
    for r in &run.rows {
        let e = |s: &RawSide| format!("{:.2}x {}", s.worst_edge, dir_word(s.worst_shrunk));
        println!(
            "{:<8} {:>5} | {:>13} {:>12.2}x {:>9} | {:>13} {:>12.2}x {:>9} | {}{}",
            r.clip, r.stock_poses.mats.len(), e(&r.stock), r.stock.worst_area, format!("{}/{}", r.stock.flipped, r.stock.hard_flipped), e(&r.cand), r.cand.worst_area, format!("{}/{}", r.cand.flipped, r.cand.hard_flipped),
            verdict(&r.stock, &r.cand), if r.cand_poses.borrowed { "  (cand posed with stock's clip)" } else { "" }
        );
    }
    println!();
    for r in &run.rows {
        println!("clip {}  (stock {} triangles counted, candidate {})", r.clip, r.stock.counted, r.cand.counted);
        for (label, s) in [("stock", &r.stock), ("candidate", &r.cand)] {
            let by: Vec<String> = s.per_mesh.iter().map(|m| format!("{} {:.2}x {} / {} flipped / {} tris", m.name, m.worst_edge, dir_word(m.worst_shrunk), m.flipped, m.counted)).collect();
            println!("    {label} by mesh: {}", by.join("; "));
        }
        print_hits("stock", stock, &r.stock_poses, &r.stock.hits, stock_in, &format!("worst {} by edge ratio", set.top));
        print_hits("candidate", cand, &r.cand_poses, &r.cand.hits, cand_in, &format!("worst {} by edge ratio", set.top));
        if r.cand.flipped > 0 {
            let top: Vec<RawHit> = r.cand.flipped_hits.iter().take(set.top).cloned().collect();
            print_hits("candidate", cand, &r.cand_poses, &top, cand_in, &format!("flipped triangles ({} total, {} hard{}; worst {} shown)", r.cand.flipped, r.cand.hard_flipped, unlisted(&r.cand), top.len()));
        }
        if r.stock.flipped > 0 {
            let top: Vec<RawHit> = r.stock.flipped_hits.iter().take(set.top).cloned().collect();
            print_hits("stock", stock, &r.stock_poses, &top, stock_in, &format!("flipped triangles ({} total, {} hard{}; worst {} shown)", r.stock.flipped, r.stock.hard_flipped, unlisted(&r.stock), top.len()));
        }
    }
    println!();
    println!("{} of {} clips WORSE than stock", run.worse, run.rows.len());
}

// ---------------------------------------------------------------------------------------------
// Pictures: the flagged triangle posed and at rest, same camera
// ---------------------------------------------------------------------------------------------

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn unit(v: Vec3) -> Vec3 {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l < 1e-12 { [0.0, 0.0, -1.0] } else { [v[0] / l, v[1] / l, v[2] / l] }
}

fn dist(a: Vec3, b: Vec3) -> f32 {
    let d = sub(a, b);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// One close-up of triangle (`mi`, `ti`) at `positions`, looking at it from outside the body.
fn close_up(model: &Model, textures: &[Option<wb_core::Texture>], positions: &[Vec<Vec3>], mi: usize, ti: usize, radius: f32, size: usize) -> Vec<u8> {
    let p: Vec<Vec3> = model.meshes[mi].tris[ti].iter().map(|&v| positions[mi][v as usize]).collect();
    let centre = [(p[0][0] + p[1][0] + p[2][0]) / 3.0, (p[0][1] + p[1][1] + p[2][1]) / 3.0, (p[0][2] + p[1][2] + p[2][2]) / 3.0];
    let (mut body, mut n) = ([0f32; 3], 0f32);
    for m in positions {
        for q in m {
            for k in 0..3 {
                body[k] += q[k];
            }
            n += 1.0;
        }
    }
    let body = body.map(|x| x / n.max(1.0));
    let mut nrm = unit(cross(sub(p[1], p[0]), sub(p[2], p[0])));
    let out = sub(centre, body);
    if nrm[0] * out[0] + nrm[1] * out[1] + nrm[2] * out[2] < 0.0 {
        nrm = nrm.map(|x| -x);
    }
    let mut cam = Camera::fit(model, 1.0, 0.9, true);
    cam.target = centre;
    cam.ortho = true;
    cam.ortho_half = radius;
    cam.dist = radius * 6.0;
    cam.pitch = nrm[1].clamp(-0.95, 0.95).asin();
    cam.yaw = nrm[0].atan2(nrm[2]);
    let tint = |m: usize, t: usize| -> Option<[u8; 3]> { if (m, t) == (mi, ti) { Some([255, 30, 30]) } else { None } };
    let opt = render::Options { cam, w: size, h: size, background: ([70, 76, 90], [40, 44, 54]), filter: Filter::Smooth, positions: Some(positions) };
    let mut rgba = render::render(model, textures, &opt, &tint).rgba;
    let q: Vec<[f32; 3]> = p.iter().map(|&x| cam.project(size, size, x)).collect();
    for e in 0..3 {
        crate::deformcmd::line(&mut rgba, size, size, [q[e][0], q[e][1]], [q[(e + 1) % 3][0], q[(e + 1) % 3][1]], [255, 30, 30]);
    }
    let c = cam.project(size, size, centre);
    crate::deformcmd::ring(&mut rgba, size, size, c[0], c[1], size as f32 * 0.11, [255, 255, 0]);
    rgba
}

fn write_sheet(path: &Path, title: &str, model: &Model, idx: &[MeshIn], rows: &[ClipRow], is_stock: bool, size: usize) -> Result<(), String> {
    let textures = image_textures(model);
    let (lo, hi) = render::bounds(model);
    let height = hi[1] - lo[1];
    let rest: Vec<Vec<Vec3>> = model.meshes.iter().map(|m| m.pos.clone()).collect();
    // panels: (clip row, hit, label)
    let mut panels: Vec<(usize, RawHit, &str)> = Vec::new();
    for (ri, r) in rows.iter().enumerate() {
        let s = if is_stock { &r.stock } else { &r.cand };
        if let Some(h) = s.hits.first() {
            panels.push((ri, h.clone(), "WORST EDGE"));
        }
        if let Some(h) = s.flipped_hits.first() {
            panels.push((ri, h.clone(), "FLIPPED"));
        }
    }
    let (cap, head, pair_w) = (46usize, 34usize, size * 2 + 6);
    let cols = if panels.len() > 1 { 2 } else { 1 };
    let nrows = panels.len().div_ceil(cols).max(1);
    let (w, h) = ((cols * (pair_w + 8)).max(600), head + nrows * (size + cap));
    let mut canvas = vec![0u8; w * h * 4];
    for px in canvas.chunks_exact_mut(4) {
        px.copy_from_slice(&[28, 30, 36, 255]);
    }
    font::draw(&mut canvas, w, h, 8, 8, &format!("{title}  LEFT POSED, RIGHT REST, RED = FLAGGED TRIANGLE"), 2, [230, 230, 230]);
    for (i, (ri, hit, label)) in panels.iter().enumerate() {
        let r = &rows[*ri];
        let ps = if is_stock { &r.stock_poses } else { &r.cand_poses };
        let mi = idx[hit.mesh].mesh;
        let posed: Vec<Vec<Vec3>> = model.meshes.iter().map(|m| anim::skin(m, &ps.mats[hit.pose])).collect();
        let rp: Vec<Vec3> = model.meshes[mi].tris[hit.tri].iter().map(|&v| rest[mi][v as usize]).collect();
        let longest = dist(rp[0], rp[1]).max(dist(rp[1], rp[2])).max(dist(rp[2], rp[0]));
        let radius = (3.0 * longest).clamp(0.05 * height, 0.25 * height);
        let left = close_up(model, &textures, &posed, mi, hit.tri, radius, size);
        let right = close_up(model, &textures, &rest, mi, hit.tri, radius, size);
        let (col, row) = (i % cols, i / cols);
        let (ox, oy) = (col * (pair_w + 8) + 4, head + row * (size + cap));
        for y in 0..size {
            let d = ((oy + y) * w + ox) * 4;
            canvas[d..d + size * 4].copy_from_slice(&left[y * size * 4..(y + 1) * size * 4]);
            let d2 = ((oy + y) * w + ox + size + 6) * 4;
            canvas[d2..d2 + size * 4].copy_from_slice(&right[y * size * 4..(y + 1) * size * 4]);
        }
        let joint = model.joint_names.get(model.triangle_joint(mi, hit.tri)).cloned().unwrap_or_default();
        let l1 = format!("{} {} {} TRI {} {}", r.clip, label, idx[hit.mesh].name, hit.tri, joint).to_ascii_uppercase().replace('_', " ");
        let l2 = format!("T={:.2}S EDGE {:.2}X {} AREA {:.2}X", ps.times.get(hit.pose).copied().unwrap_or(0.0), hit.edge, dir_word(hit.shrunk), hit.area).to_ascii_uppercase();
        let max_chars = (pair_w - 12) / 12;
        let cut = |s: String| -> String { s.chars().take(max_chars).collect() };
        font::draw(&mut canvas, w, h, ox as i64, (oy + size + 4) as i64, &cut(l1), 2, [230, 230, 230]);
        font::draw(&mut canvas, w, h, ox as i64, (oy + size + 24) as i64, &cut(l2), 2, [230, 230, 230]);
    }
    let mut png = Vec::new();
    image::RgbaImage::from_raw(w as u32, h as u32, canvas)
        .ok_or("bad sheet size")?
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    std::fs::write(path, png).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag(x: f32, y: f32, z: f32) -> Mat4 {
        [x, 0.0, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, 0.0, z, 0.0, 0.0, 0.0, 0.0, 1.0]
    }

    /// One triangle, vertex 2 on bone 1; bone 1 takes `m` in the single pose.
    fn tri_case(m: Mat4) -> RawSide {
        let rest = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let (skin, _, _) = make_skin(&[[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], &[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], 2);
        let mesh = MeshIn { mesh: 0, name: "m".into(), rest, indices: vec![0, 1, 2], skin, unweighted: 0, bad_joint: 0 };
        run_stretch(&[mesh], &[vec![IDENTITY, m]], 0.01, 5)
    }

    #[test]
    fn rest_pose_is_exactly_one() {
        let s = tri_case(IDENTITY);
        assert_eq!((s.worst_edge, s.worst_area, s.flipped, s.counted), (1.0, 1.0, 0, 1));
    }

    #[test]
    fn a_mirrored_and_stretched_vertex_is_a_named_hard_flip() {
        let s = tri_case(diag(1.0, -2.0, 1.0));
        assert_eq!((s.flipped, s.hard_flipped, s.flipped_unlisted), (1, 1, 0));
        assert_eq!(s.flipped_hits[0].tri, 0);
    }

    #[test]
    fn a_pure_mirror_is_counted_but_cannot_be_named_and_is_not_hard() {
        let s = tri_case(diag(1.0, -1.0, 1.0));
        assert_eq!((s.flipped, s.hard_flipped, s.flipped_unlisted), (1, 0, 1));
    }

    #[test]
    fn a_squashed_vertex_reads_as_a_pinch_of_ten() {
        let s = tri_case(diag(1.0, 0.1, 1.0));
        assert!((s.worst_edge - 10.0).abs() < 0.2, "{}", s.worst_edge);
        assert!(s.worst_shrunk);
        assert_eq!(s.flipped, 0);
        let s = tri_case(diag(1.0, 4.0, 1.0));
        assert!(!s.worst_shrunk && s.worst_edge > 3.0);
    }

    #[test]
    fn vertices_without_weight_stay_at_rest_instead_of_collapsing() {
        let (skin, unweighted, bad) = make_skin(&[[0, 0, 0, 0], [9, 0, 0, 0], [1, 1, 0, 0]], &[[0.0; 4], [1.0, 0.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0]], 2);
        assert_eq!((unweighted, bad), (1, 1));
        assert_eq!(skin.joints[0][0], 2);
        assert_eq!(skin.joints[1][0], 2);
        assert!((skin.weights[2][0] - 0.5).abs() < 1e-6 && (skin.weights[2][1] - 0.5).abs() < 1e-6);
        // a triangle with such a vertex does not read as stretched under a pose
        let rest = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let (skin, ..) = make_skin(&[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], &[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0; 4]], 1);
        let mesh = MeshIn { mesh: 0, name: "m".into(), rest, indices: vec![0, 1, 2], skin, unweighted: 1, bad_joint: 0 };
        let s = run_stretch(&[mesh], &[vec![IDENTITY]], 0.01, 5);
        assert_eq!((s.worst_edge, s.flipped), (1.0, 0));
    }

    #[test]
    fn a_barely_changed_sliver_that_flips_is_not_a_hard_flip() {
        // thin triangle (rest height 0.02 over base 1) whose tip mirrors across the base: lengths stay put
        let rest = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.02, 0.0]];
        let (skin, ..) = make_skin(&[[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], &[[1.0, 0.0, 0.0, 0.0]; 3], 2);
        let mesh = MeshIn { mesh: 0, name: "m".into(), rest, indices: vec![0, 1, 2], skin, unweighted: 0, bad_joint: 0 };
        let s = run_stretch(&[mesh], &[vec![IDENTITY, diag(1.0, -1.0, 1.0)]], 0.005, 5);
        assert_eq!((s.flipped, s.hard_flipped), (1, 0));
        assert_eq!(s.flipped_unlisted, 1, "a pure mirror leaves every edge unchanged, so rigkit cannot name it");
    }

    /// rigkit's number for the worst stock triangle in walk equals the same triangle measured with
    /// wb-core's own skinner, so the two code paths agree on a real model.
    #[test]
    fn worst_stock_triangle_matches_wb_cores_own_skinning() {
        let path = Path::new(crate::STOCK);
        if !path.exists() {
            return;
        }
        let stock = Model::open(path).unwrap();
        let set = Settings { clips: Some(vec!["walk".into()]), steps: Some(19), top: 3, min_edge: None, all_meshes: false };
        let (run, ins, _) = run(path, path, &set, &stock, &stock).unwrap();
        let row = &run.rows[0];
        let hit = &row.stock.hits[0];
        let m = &stock.meshes[ins[hit.mesh].mesh];
        let t = row.stock_poses.times[hit.pose];
        let posed = stock.posed(Some(stock.clips.iter().position(|c| c == "walk").unwrap()), t);
        let (lo, hi) = render::bounds(&stock);
        let min_edge = 0.002 * (hi[1] - lo[1]);
        let tri = m.tris[hit.tri];
        let mut worst = 1.0f32;
        for e in 0..3 {
            let (a, b) = (tri[e] as usize, tri[(e + 1) % 3] as usize);
            let (l0, l1) = (dist(m.pos[a], m.pos[b]), dist(posed[ins[hit.mesh].mesh][a], posed[ins[hit.mesh].mesh][b]));
            if l0 >= min_edge {
                worst = worst.max(if l0 > l1 { l0 / l1.max(1e-9) } else { l1 / l0 });
            }
        }
        assert!((worst - hit.edge).abs() / hit.edge < 0.01, "wb-core says {worst}, rigkit says {}", hit.edge);
    }

    #[test]
    fn verdict_needs_a_real_difference() {
        let side = |e: f32, f: usize| RawSide { worst_edge: e, flipped: f, hard_flipped: f, ..Default::default() };
        assert_eq!(verdict(&side(2.0, 3), &side(2.0, 3)), "same");
        assert_eq!(verdict(&side(2.0, 3), &side(2.2, 3)), "same");
        assert_eq!(verdict(&side(2.0, 3), &side(2.6, 3)), "WORSE");
        assert_eq!(verdict(&side(2.0, 3), &side(2.0, 4)), "WORSE");
        assert_eq!(verdict(&side(2.0, 3), &side(1.5, 3)), "better");
        assert_eq!(verdict(&side(2.0, 3), &side(1.5, 5)), "WORSE");
    }
}
