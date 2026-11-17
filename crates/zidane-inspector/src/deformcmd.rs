//! `gaw deform-check`: pose a skinned model through its clips and measure what the skin does to the
//! surface (edge stretch and squash, volume loss at joints, new intersections between parts,
//! weights that do not sum to 1 or use more than 4 bones), against limits calibrated on stock
//! Zidane or, with --ref, against the reference at the same frames. Writes deform.json and one
//! picture sheet per kind; exits 1 on FAIL.
use crate::Args;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::PathBuf;
use wb_core::deform::{self, check_weights, FrameDesc, Limits, Measured, Prep, PoseSource, Spot};
use wb_core::render::{self, Camera, Filter};
use wb_core::{font, image_textures, Model, Texture};

pub const DEFORM_USAGE: &str = "  gaw deform-check FILE.glb --out-dir NEW_DIR [--ref REF.glb] [--clips all|a,b] [--steps N] [--top N] [--size PX] [--all-meshes]
      Poses FILE through each clip at N fixed steps (default 12 per clip, clips default all) and measures,
      against the model's own rest pose: edge stretch and squash (share of the surface whose edges grew
      past 1.5x or shrank below 0.5x: candy-wrapper twists), volume loss at joints (cone volume of the
      patch around each joint), new intersections between body parts (arm through torso, hair through
      shoulders; crossings that already exist at rest are ignored), and vertices whose weights do not sum
      to 1, use more than 4 bones, or name a missing joint. FILE with no clips of its own is posed with
      --ref's clips (same skeleton). With --ref every measure is also taken on the reference at the same
      frames and the limit becomes 1.5x the reference's value (stock's own flaws do not count); without
      it the limits are the absolute ones calibrated on stock Zidane.
      Writes OUT/deform.json (verdict, per-measure value, reference value and limit, worst places as
      mesh/triangle/bone/clip/time/position, and a `summary` array shaped like defects.json's) and
      OUT/sheet-stretch|squash|volume|intersect|weights.png: the worst places as numbered markers on
      posed close-ups with a small whole-body inset. Prints PASS or FAIL per measure and exits 1 on FAIL.
      --top N       worst places per measure (default 6)    --size PX   panel size (default 360)
      --all-meshes  also check the battle_model* overlay meshes (skipped by default)";

fn parse(a: &[String]) -> (Vec<String>, Vec<(String, String)>, Vec<String>) {
    let valued = ["--out-dir", "--ref", "--clips", "--steps", "--top", "--size"];
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

/// One place that stands out, ready for the JSON and a picture panel.
struct Find {
    n: usize,
    frame: usize,
    value: f32,
    ref_value: Option<f32>,
    mesh: usize,
    tri: usize,
    tri2: Option<(usize, usize)>,
    bone: String,
    bone2: Option<String>,
    joint: Option<usize>,
    pos: [f32; 3],
    why: String,
    /// Weights: the vertex.
    vertex: Option<usize>,
}

struct Kind {
    name: &'static str,
    unit: &'static str,
    value: f32,
    ref_value: Option<f32>,
    limit: f32,
    rule: String,
    pass: bool,
    frames_over: usize,
    finds: Vec<Find>,
    detail: Value,
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Worst spots first (highest `score`), at most one per key (bone, clip) and none within `gap` of an
/// earlier one in the same frame.
fn pick<'a>(spots: &'a [Spot], top: usize, gap: f32, score: impl Fn(&Spot) -> f32, key: impl Fn(&Spot) -> (usize, String)) -> Vec<&'a Spot> {
    let mut idx: Vec<&Spot> = spots.iter().collect();
    idx.sort_by(|a, b| score(b).total_cmp(&score(a)));
    let mut out: Vec<&Spot> = Vec::new();
    let mut used: HashSet<(usize, String)> = HashSet::new();
    for s in idx {
        if out.len() >= top {
            break;
        }
        let k = key(s);
        if used.contains(&k) || out.iter().any(|o| o.frame == s.frame && dist3(o.pos, s.pos) < gap) {
            continue;
        }
        used.insert(k);
        out.push(s);
    }
    out
}

fn limit_for(has_ref: bool, abs: f32, floor: f32, max_x: f32, refv: Option<f32>) -> (f32, String) {
    match (has_ref, refv) {
        (true, Some(r)) => {
            let l = (r * max_x).max(floor);
            (l, format!("{max_x}x the reference ({r:.2}), at least {floor}"))
        }
        _ => (abs, "absolute limit calibrated on stock Zidane".to_owned()),
    }
}

/// The reference's own value at the same place and frame: the extreme (max when `high`) of its kept
/// spots within `reach` of `s`, or `default` (below its keep threshold) when it has none there.
/// None without a reference.
fn near_ref(spots: Option<&Vec<Spot>>, s: &Spot, reach: f32, high: bool, default: f32) -> Option<f32> {
    let sp = spots?;
    let mut best: Option<f32> = None;
    for r in sp.iter().filter(|r| r.frame == s.frame && dist3(r.pos, s.pos) < reach) {
        best = Some(match best {
            None => r.value,
            Some(b) => if high { b.max(r.value) } else { b.min(r.value) },
        });
    }
    Some(best.unwrap_or(default))
}

fn time_label(src: &PoseSource, f: &FrameDesc) -> String {
    format!("{} {:.2}s", f.clip, src.seconds(f))
}

pub fn deform_cmd(args: &Args) -> Result<(), String> {
    let (pos, opts, flags) = parse(&args.0);
    let get = |k: &str| opts.iter().find(|o| o.0 == k).map(|o| o.1.as_str());
    let file = PathBuf::from(pos.first().ok_or("deform-check needs FILE.glb")?);
    let out = PathBuf::from(get("--out-dir").ok_or("deform-check needs --out-dir NEW_DIR")?);
    if out.exists() && std::fs::read_dir(&out).map(|mut d| d.next().is_some()).unwrap_or(true) {
        return Err(format!("{} already has files; --out-dir must be new or empty", out.display()));
    }
    let steps: usize = get("--steps").map(|s| s.parse().map_err(|_| "--steps must be a whole number".to_string())).transpose()?.unwrap_or(12);
    let top: usize = get("--top").map(|s| s.parse().map_err(|_| "--top must be a whole number".to_string())).transpose()?.unwrap_or(6);
    let size: usize = get("--size").map(|s| s.parse().map_err(|_| "--size must be a whole number".to_string())).transpose()?.unwrap_or(360);
    if steps == 0 || steps > 200 || !(120..=1200).contains(&size) {
        return Err("--steps must be 1..200 and --size 120..1200".into());
    }
    let all_meshes = flags.iter().any(|f| f == "--all-meshes");
    let model = Model::open(&file)?;
    let ref_path = get("--ref").map(PathBuf::from);
    let refm = ref_path.as_ref().map(|p| Model::open(p)).transpose()?;
    let limits = Limits::default();

    let prep = Prep::new(&model, all_meshes).map_err(|e| format!("{}: {e}", file.display()))?;
    let ref_prep = refm.as_ref().map(|m| Prep::new(m, all_meshes)).transpose()?;
    if let (Some(_), Some(rp)) = (&refm, &ref_prep) {
        let d = wb_core::weights::skeleton_diff(&prep.skel, &rp.skel, 1e-3);
        if !d.is_empty() {
            return Err(format!("{} and the reference do not share a skeleton ({}); the comparison would not mean anything. Run `gaw weights` to see the full list.", file.display(), d[0]));
        }
    }
    let src = PoseSource { model: &model, borrowed: refm.as_ref() };
    let names = src.clip_names();
    if names.is_empty() {
        return Err("neither the file nor --ref has a clip to pose it with".into());
    }
    let wanted: Vec<String> = match get("--clips") {
        None | Some("all") => Vec::new(),
        Some(s) => s.split(',').map(|x| x.trim().to_owned()).filter(|x| !x.is_empty()).collect(),
    };
    let frames = deform::frames_for(&names, &wanted, steps)?;
    let borrowed_clips = model.rig.as_ref().map(|r| r.clips.is_empty()).unwrap_or(true);

    println!("deform-check {}: {} frames ({} clip(s) x {steps} steps){}", file.display(), frames.len(), names.len().min(if wanted.is_empty() { usize::MAX } else { wanted.len() }), if borrowed_clips { ", clips borrowed from --ref" } else { "" });
    let t0 = std::time::Instant::now();
    let m = deform::measure(&model, &prep, &src, &frames, &limits);
    let rsrc = refm.as_ref().map(|r| PoseSource { model: r, borrowed: None });
    let rm: Option<Measured> = match (&rsrc, &ref_prep) {
        (Some(s), Some(p)) => Some(deform::measure(s.model, p, s, &frames, &limits)),
        _ => None,
    };
    let wc = check_weights(&model)?;
    let rwc = refm.as_ref().map(check_weights).transpose()?;
    let has_ref = refm.is_some();
    let posed: usize = m.stats.iter().flatten().count();
    let mut notes: Vec<String> = Vec::new();
    if !m.missing.is_empty() {
        notes.push(format!("{} of {} frames could not be posed (no clip of that name to use)", m.missing.len(), frames.len()));
    }
    if let Some(rm) = &rm {
        if !rm.missing.is_empty() {
            notes.push(format!("the reference has no clip for {} of the frames; its numbers cover the others", rm.missing.len()));
        }
    }
    if prep.joint_zones.is_empty() {
        notes.push("no joint has a surface patch around it, so the volume measure has nothing to measure".into());
    }
    let gap = 0.08 * prep.height;
    let tri_loc = |i: u32| (prep.tris[i as usize].0 as usize, prep.tris[i as usize].1 as usize);
    let bone_of = |i: u32| prep.skel.names[prep.bone[i as usize] as usize].clone();

    // ---- stretch and squash: whole surface and each body region, worst scope reported
    let mut kinds: Vec<Kind> = Vec::new();
    for (name, bad_high) in [("stretch", true), ("squash", false)] {
        let (abs_all, abs_reg, floor, edge) = if bad_high { (limits.stretch_area_pct, limits.stretch_region_pct, limits.floor_stretch_area_pct, limits.stretch_edge) } else { (limits.squash_area_pct, limits.squash_region_pct, limits.floor_squash_area_pct, limits.squash_edge) };
        // scope 0 = whole surface, 1.. = regions
        let nreg = prep.regions.len();
        let mut scopes: Vec<(String, f32, Option<f32>, f32, String)> = Vec::new();
        let mut scope_region: Vec<usize> = Vec::new(); // region index of each scope, usize::MAX for the whole surface
        for sc in 0..=nreg {
            if sc > 0 && prep.region_area[sc - 1] <= 0.0 {
                continue; // no triangles in this region (for example the tail of a tailless model)
            }
            let (label, cand, refv, abs) = if sc == 0 {
                ("whole surface".to_owned(), m.max_stat(|s| if bad_high { s.stretch_area_pct } else { s.squash_area_pct }), rm.as_ref().map(|r| r.max_stat(|s| if bad_high { s.stretch_area_pct } else { s.squash_area_pct })), abs_all)
            } else {
                (format!("{} region", prep.regions[sc - 1]), m.max_region(sc - 1, bad_high), rm.as_ref().map(|r| r.max_region(sc - 1, bad_high)), abs_reg)
            };
            let (limit, rule) = limit_for(has_ref, abs, if sc == 0 { floor } else { floor.max(limits.floor_region_pct) }, limits.max_x, refv);
            scopes.push((label, cand, refv, limit, rule));
            scope_region.push(if sc == 0 { usize::MAX } else { sc - 1 });
        }
        let worst = (0..scopes.len()).max_by(|&a, &b| (scopes[a].1 / scopes[a].3.max(1e-6)).total_cmp(&(scopes[b].1 / scopes[b].3.max(1e-6)))).unwrap_or(0);
        let (wlabel, val, refv, limit, rule) = scopes[worst].clone();
        let pass = scopes.iter().all(|s| s.1 <= s.3);
        // Worst places: inside the worst scope (the whole surface when it is the worst).
        let in_scope = |s: &Spot| scope_region[worst] == usize::MAX || prep.tri_region[s.tri as usize] as usize == scope_region[worst];
        let pool: Vec<Spot> = (if bad_high { &m.stretch } else { &m.squash }).iter().filter(|s| in_scope(s)).cloned().collect();
        let ref_spots = rm.as_ref().map(|r| if bad_high { &r.stretch } else { &r.squash });
        let dflt = if bad_high { 1.0 + (edge - 1.0) * 0.8 } else { 1.0 - (1.0 - edge) * 0.8 };
        let near = |s: &Spot| near_ref(ref_spots, s, gap * 1.5, bad_high, dflt);
        let score = |s: &Spot| match near(s) {
            Some(r) => if bad_high { s.value / r.max(1e-3) } else { r / s.value.max(1e-3) },
            None => if bad_high { s.value } else { 1.0 / s.value.max(1e-3) },
        };
        let sel = pick(&pool, top, gap, score, |s| (prep.bone[s.tri as usize] as usize, frames[s.frame].clip.clone()));
        let finds: Vec<Find> = sel
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let (mesh, tri) = tri_loc(s.tri);
                Find { n: k + 1, frame: s.frame, value: s.value, ref_value: near(s), mesh, tri, tri2: None, bone: bone_of(s.tri), bone2: None, joint: None, pos: s.pos, why: format!("{} edge x{:.2} its rest length", if bad_high { "longest" } else { "shortest" }, s.value), vertex: None }
            })
            .collect();
        let over = m.stats.iter().flatten().filter(|s| {
            let all = if bad_high { s.stretch_area_pct } else { s.squash_area_pct };
            let regs = if bad_high { &s.stretch_region_pct } else { &s.squash_region_pct };
            all > scopes[0].3 || regs.iter().enumerate().any(|(r, &v)| scope_region.iter().position(|&x| x == r).is_some_and(|k| v > scopes[k].3))
        }).count();
        let worst_ratio = if bad_high { m.max_stat(|s| s.worst_stretch) } else { m.stats.iter().flatten().map(|s| s.worst_squash).fold(1.0, f32::min) };
        kinds.push(Kind {
            name,
            unit: "percent of the scope's surface area in triangles past the edge limit (worst frame; scope = whole surface or one body region)",
            value: val,
            ref_value: refv,
            limit,
            rule: format!("{wlabel}: {rule}"),
            pass,
            frames_over: over,
            finds,
            detail: json!({
                "edge_limit": edge, "worst_edge_ratio": worst_ratio,
                "reference_worst_edge_ratio": rm.as_ref().map(|r| if bad_high { r.max_stat(|s| s.worst_stretch) } else { r.stats.iter().flatten().map(|s| s.worst_squash).fold(1.0, f32::min) }),
                "worst_scope": wlabel,
                "scopes": scopes.iter().map(|s| json!({"scope": s.0, "value": s.1, "reference_value": s.2, "limit": s.3, "pass": s.1 <= s.3})).collect::<Vec<_>>(),
            }),
        });
    }

    // ---- volume
    {
        let (vl, at) = m.worst_volume_loss(&prep);
        let refv = match (&rm, &ref_prep) {
            (Some(r), Some(p)) => Some(r.worst_volume_loss(p).0),
            _ => None,
        };
        let (limit, rule) = limit_for(has_ref, limits.volume_loss_pct, limits.floor_volume_loss_pct, limits.max_x, refv);
        let spots: Vec<Spot> = m.volume.iter().filter(|s| s.value < 1.0).cloned().collect();
        let vscore = |s: &Spot| {
            let loss = 100.0 * (1.0 - s.value).max(0.0);
            match (&rm, &ref_prep) {
                (Some(r), Some(p)) => loss - r.volume_ratio_of(p, s.joint.unwrap_or(0), s.frame).map(|x| 100.0 * (1.0 - x).max(0.0)).unwrap_or(0.0),
                _ => loss,
            }
        };
        let sel = pick(&spots, top, 0.0, vscore, |s| (s.joint.unwrap_or(0), String::new()));
        let finds: Vec<Find> = sel
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let j = s.joint.unwrap_or(0);
                let rv = match (&rm, &ref_prep) {
                    (Some(r), Some(p)) => r.volume_ratio_of(p, j, s.frame).map(|x| 100.0 * (1.0 - x).max(0.0)),
                    _ => None,
                };
                let (mesh, tri) = tri_loc(s.tri);
                Find { n: k + 1, frame: s.frame, value: 100.0 * (1.0 - s.value).max(0.0), ref_value: rv, mesh, tri, tri2: None, bone: prep.skel.names[j].clone(), bone2: None, joint: Some(j), pos: s.pos, why: format!("patch around the joint keeps {:.0} percent of its rest volume", 100.0 * s.value), vertex: None }
            })
            .collect();
        let over = (0..frames.len()).filter(|&f| m.stats[f].as_ref().is_some_and(|s| s.volume_ratio.iter().any(|&r| 100.0 * (1.0 - r) > limit))).count();
        kinds.push(Kind {
            name: "volume",
            unit: "percent of the joint patch's rest volume lost (worst joint, worst frame)",
            value: vl,
            ref_value: refv,
            limit,
            rule,
            pass: vl <= limit,
            frames_over: over,
            finds,
            detail: json!({"joints_measured": prep.joint_zones.iter().map(|z| prep.skel.names[z.joint].clone()).collect::<Vec<_>>(), "worst": at.map(|(j, f)| json!({"joint": prep.skel.names[j], "clip": frames[f].clip, "time": src.seconds(&frames[f])}))}),
        });
    }

    // ---- intersections
    {
        let refv = rm.as_ref().map(|r| r.max_stat(|s| s.intersect_area_pct));
        let (limit, rule) = limit_for(has_ref, limits.intersect_area_pct, limits.floor_intersect_area_pct, limits.max_x, refv);
        let val = m.max_stat(|s| s.intersect_area_pct);
        let ref_area = ref_prep.as_ref().map(|p| p.total_area).unwrap_or(1.0).max(1e-12);
        let inear = |s: &Spot| near_ref(rm.as_ref().map(|r| &r.intersect), s, gap * 1.5, true, 0.0).map(|a| 100.0 * a / ref_area);
        let iscore = |s: &Spot| 100.0 * s.value / prep.total_area.max(1e-12) - inear(s).unwrap_or(0.0);
        let sel = pick(&m.intersect, top, gap, iscore, |s| (prep.bone[s.tri as usize] as usize * 64 + s.tri2.map(|t| prep.bone[t as usize] as usize).unwrap_or(0), frames[s.frame].clip.clone()));
        let finds: Vec<Find> = sel
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let t2 = s.tri2.unwrap_or(s.tri);
                let (mesh, tri) = tri_loc(s.tri);
                let (a, b) = (bone_of(s.tri), bone_of(t2));
                let (ra, rb) = (&prep.region_of_joint[prep.bone[s.tri as usize] as usize], &prep.region_of_joint[prep.bone[t2 as usize] as usize]);
                Find {
                    n: k + 1,
                    frame: s.frame,
                    value: 100.0 * s.value / prep.total_area.max(1e-12),
                    ref_value: inear(s),
                    mesh,
                    tri,
                    tri2: Some(tri_loc(t2)),
                    bone: a,
                    bone2: Some(b),
                    joint: None,
                    pos: s.pos,
                    why: format!("{ra} surface passes through {rb} surface (crossing patch is {:.2} percent of the surface)", 100.0 * s.value / prep.total_area.max(1e-12)),
                    vertex: None,
                }
            })
            .collect();
        let over = m.stats.iter().flatten().filter(|s| s.intersect_area_pct > limit).count();
        let clusters = m.max_stat(|s| s.intersect_clusters as f32) as usize;
        kinds.push(Kind {
            name: "intersect",
            unit: "percent of surface area in triangles that newly pass through another part",
            value: val,
            ref_value: refv,
            limit,
            rule,
            pass: val <= limit,
            frames_over: over,
            finds,
            detail: json!({"crossings_at_rest_ignored": prep.rest_pairs.len(), "most_clusters_in_one_frame": clusters, "most_pairs_in_one_frame": m.max_stat(|s| s.intersect_pairs as f32) as usize, "reference_crossings_at_rest_ignored": ref_prep.as_ref().map(|p| p.rest_pairs.len())}),
        });
    }

    // ---- weights
    {
        let refv = rwc.as_ref().map(|r| r.bad.len() as f32);
        let (limit, rule) = if has_ref { (refv.unwrap_or(0.0), "no more than the reference has".to_owned()) } else { (0.0, "none allowed".to_owned()) };
        let mut bad: Vec<&deform::BadVertex> = wc.bad.iter().collect();
        bad.sort_by(|a, b| ((b.sum - 1.0).abs() + b.influences as f32).total_cmp(&((a.sum - 1.0).abs() + a.influences as f32)));
        let mut finds: Vec<Find> = Vec::new();
        for b in bad {
            if finds.len() >= top {
                break;
            }
            let p = model.meshes[b.mesh].pos[b.vertex];
            if finds.iter().any(|f| f.mesh == b.mesh && dist3(f.pos, p) < gap) {
                continue;
            }
            let tri = model.meshes[b.mesh].tris.iter().position(|t| t.contains(&(b.vertex as u32))).unwrap_or(0);
            let j = model.meshes[b.mesh].joints[b.vertex][0] as usize;
            finds.push(Find { n: finds.len() + 1, frame: 0, value: b.influences as f32, ref_value: None, mesh: b.mesh, tri, tri2: None, bone: prep.skel.names.get(j).cloned().unwrap_or_default(), bone2: None, joint: None, pos: p, why: b.why.clone(), vertex: Some(b.vertex) });
        }
        kinds.push(Kind {
            name: "weights",
            unit: "vertices",
            value: wc.bad.len() as f32,
            ref_value: refv,
            limit,
            rule,
            pass: wc.bad.len() as f32 <= limit,
            frames_over: 0,
            finds,
            detail: json!({"vertices": wc.vertices, "weights_not_summing_to_1": wc.not_sum_one, "more_than_4_influences": wc.over_four, "missing_joint": wc.bad_joint, "reference_bad": rwc.as_ref().map(|r| r.bad.len())}),
        });
    }

    // ---- pictures
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let textures: Vec<Option<Texture>> = image_textures(&model);
    let mut sheets: Vec<(&'static str, Option<String>)> = Vec::new();
    for k in &kinds {
        if k.finds.is_empty() {
            sheets.push((k.name, None));
            continue;
        }
        let path = out.join(format!("sheet-{}.png", k.name));
        write_sheet(&path, k, &model, &prep, &src, &frames, &textures, size)?;
        sheets.push((k.name, Some(path.file_name().unwrap().to_string_lossy().into_owned())));
    }

    // ---- report
    let pass = kinds.iter().all(|k| k.pass);
    let measures: Vec<Value> = kinds
        .iter()
        .map(|k| {
            let sheet = sheets.iter().find(|s| s.0 == k.name).and_then(|s| s.1.clone());
            json!({
                "kind": k.name, "pass": k.pass, "unit": k.unit, "value": k.value, "reference_value": k.ref_value, "limit": k.limit, "limit_rule": k.rule,
                "frames_over_limit": k.frames_over, "sheet": sheet, "detail": k.detail,
                "worst": k.finds.iter().map(|f| {
                    let fr = &frames[f.frame];
                    let mut o = json!({
                        "marker": f.n, "mesh": model.meshes[f.mesh].name, "triangle": f.tri, "bone": f.bone, "clip": if k.name == "weights" { Value::Null } else { json!(fr.clip) },
                        "time": if k.name == "weights" { Value::Null } else { json!(src.seconds(fr)) }, "step": fr.step, "value": f.value, "reference_value_same_frame": f.ref_value,
                        "position": f.pos, "why": f.why,
                    });
                    if let Some((m2, t2)) = f.tri2 { o["second_mesh"] = json!(model.meshes[m2].name); o["second_triangle"] = json!(t2); o["second_bone"] = json!(f.bone2); }
                    if let Some(v) = f.vertex { o["vertex"] = json!(v); }
                    if let Some(j) = f.joint { o["joint"] = json!(prep.skel.names[j]); }
                    o
                }).collect::<Vec<_>>(),
            })
        })
        .collect();
    // Same shape as defects.json `summary`, so a gate `ratio` rule can read it.
    let summary: Vec<Value> = kinds
        .iter()
        .map(|k| {
            let load = if k.name == "weights" { k.value } else { k.value };
            json!({"kind": k.name, "skipped": false, "defects": if k.name == "weights" { k.value as usize } else { k.frames_over }, "worst_score": k.value, "load_percent": load, "limit": k.limit, "unit": k.unit,
                "reference": k.ref_value.map(|r| json!({"defects": r, "load_percent": r}))})
        })
        .collect();
    let report = json!({
        "gaw_deform_check": 1,
        "file": file.display().to_string(), "reference": ref_path.as_ref().map(|p| p.display().to_string()),
        "verdict": if pass { "PASS" } else { "FAIL" }, "pass": pass,
        "frames": {"clips": names, "steps_per_clip": steps, "count": frames.len(), "posed": posed, "borrowed_clips_from_reference": borrowed_clips, "list": frames.iter().map(|f| json!({"clip": f.clip, "step": f.step, "time": src.seconds(f)})).collect::<Vec<_>>()},
        "meshes_checked": model.meshes.iter().filter(|m| all_meshes || !m.name.starts_with("battle_model")).map(|m| m.name.clone()).collect::<Vec<_>>(),
        "triangles_checked": prep.tris.len(),
        "limits": {"stretch_edge_ratio": limits.stretch_edge, "squash_edge_ratio": limits.squash_edge, "absolute": {"stretch_area_pct": limits.stretch_area_pct, "squash_area_pct": limits.squash_area_pct, "volume_loss_pct": limits.volume_loss_pct, "intersect_area_pct": limits.intersect_area_pct, "weight_vertices": 0}, "with_reference": {"max_x": limits.max_x, "floors": {"region": limits.floor_region_pct, "stretch": limits.floor_stretch_area_pct, "squash": limits.floor_squash_area_pct, "volume": limits.floor_volume_loss_pct, "intersect": limits.floor_intersect_area_pct}}},
        "measures": measures, "summary": summary, "notes": notes,
        "seconds": t0.elapsed().as_secs_f32(),
        "gate": "summary has the shape of defects.json's summary (kind, defects, worst_score, load_percent, reference.load_percent), so a gate `ratio` rule can read it once `gaw gate` can run this check instead of `gaw defects`; see skills/gaw/references/deform.md",
    });
    std::fs::write(out.join("deform.json"), serde_json::to_vec_pretty(&report).unwrap()).map_err(|e| format!("deform.json: {e}"))?;

    for k in &kinds {
        let rv = k.ref_value.map(|r| format!(" (reference {r:.2})")).unwrap_or_default();
        println!("{} {:<9} {:.2}{rv}, limit {:.2}: {}", if k.pass { "PASS" } else { "FAIL" }, k.name, k.value, k.limit, k.rule);
        if !k.pass {
            for f in k.finds.iter().take(3) {
                let fr = &frames[f.frame];
                if k.name == "weights" {
                    println!("       #{} {} vertex {}: {}", f.n, model.meshes[f.mesh].name, f.vertex.unwrap_or(0), f.why);
                } else {
                    println!("       #{} {} {}{}: {}", f.n, time_label(&src, fr), f.bone, f.bone2.as_ref().map(|b| format!(" / {b}")).unwrap_or_default(), f.why);
                }
            }
        }
    }
    for n in &notes {
        println!("note {n}");
    }
    println!("{} ({:.1}s); wrote {}", if pass { "DEFORM PASS" } else { "DEFORM FAIL" }, t0.elapsed().as_secs_f32(), out.join("deform.json").display());
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Picture sheets
// ---------------------------------------------------------------------------------------------

fn ring(rgba: &mut [u8], w: usize, h: usize, cx: f32, cy: f32, r: f32, color: [u8; 3]) {
    let t = 2.2;
    let (x0, x1) = (((cx - r - t - 1.0).floor().max(0.0)) as usize, ((cx + r + t + 1.0).ceil().min(w as f32 - 1.0)) as usize);
    let (y0, y1) = (((cy - r - t - 1.0).floor().max(0.0)) as usize, ((cy + r + t + 1.0).ceil().min(h as f32 - 1.0)) as usize);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = (((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt() - r).abs();
            if d <= t * 0.5 {
                rgba[(y * w + x) * 4..(y * w + x) * 4 + 3].copy_from_slice(&color);
            }
        }
    }
}

fn line(rgba: &mut [u8], w: usize, h: usize, a: [f32; 2], b: [f32; 2], color: [u8; 3]) {
    let n = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()).ceil() as usize).clamp(1, 4000);
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let (x, y) = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t);
        for (dx, dy) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)] {
            let (px, py) = ((x + dx) as i64, (y + dy) as i64);
            if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                let o = (py as usize * w + px as usize) * 4;
                rgba[o..o + 3].copy_from_slice(&color);
            }
        }
    }
}

fn clean(s: &str) -> String {
    s.to_ascii_uppercase().replace('_', " ")
}

fn norm3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn write_sheet(path: &std::path::Path, k: &Kind, model: &Model, prep: &Prep, src: &PoseSource, frames: &[FrameDesc], textures: &[Option<Texture>], size: usize) -> Result<(), String> {
    let cols = 3usize.min(k.finds.len()).max(1);
    let rows = k.finds.len().div_ceil(cols);
    let cap = 62usize;
    let title = 34usize;
    let (cw, ch) = (size, size + cap);
    let (w, h) = ((cols * cw).max(760), title + rows * ch);
    let mut canvas = vec![0u8; w * h * 4];
    for px in canvas.chunks_exact_mut(4) {
        px.copy_from_slice(&[28, 30, 36, 255]);
    }
    let head = format!(
        "{} {}  VALUE {:.2}{}  LIMIT {:.2}",
        k.name.to_ascii_uppercase(),
        if k.pass { "PASS" } else { "FAIL" },
        k.value,
        k.ref_value.map(|r| format!("  REFERENCE {r:.2}")).unwrap_or_default(),
        k.limit
    );
    font::draw(&mut canvas, w, h, 8, 8, &head, 2, if k.pass { [20, 110, 30] } else { [170, 20, 20] });
    let height = prep.height;
    let mut cache: Vec<(usize, Vec<Vec<[f32; 3]>>)> = Vec::new();
    for (i, f) in k.finds.iter().enumerate() {
        let (col, row) = (i % cols, i / cols);
        let (ox, oy) = (col * cw, title + row * ch);
        let fr = &frames[f.frame];
        let posed: Vec<Vec<[f32; 3]>> = match cache.iter().find(|c| c.0 == f.frame) {
            Some(c) => c.1.clone(),
            None => {
                let p = if k.name == "weights" { model.meshes.iter().map(|m| m.pos.clone()).collect() } else { src.posed(fr).ok_or("cannot pose a frame that was measured")? };
                cache.push((f.frame, p.clone()));
                p
            }
        };
        // Body centre of the posed model, to look at the spot from outside.
        let (mut c, mut n) = ([0f32; 3], 0f32);
        for m in &posed {
            for p in m {
                for a in 0..3 {
                    c[a] += p[a];
                }
                n += 1.0;
            }
        }
        let body = [c[0] / n.max(1.0), c[1] / n.max(1.0), c[2] / n.max(1.0)];
        let tri_pts = |mi: usize, ti: usize| model.meshes[mi].tris[ti].map(|v| posed[mi][v as usize]);
        // Highlight sets: (mesh, tri) -> color.
        let mut hl: Vec<((usize, usize), [u8; 3])> = Vec::new();
        let mut view_n: [f32; 3];
        let radius;
        match k.name {
            "volume" => {
                let z = prep.joint_zones.iter().find(|z| Some(z.joint) == f.joint);
                if let Some(z) = z {
                    for &t in &z.tris {
                        hl.push(((prep.tris[t as usize].0 as usize, prep.tris[t as usize].1 as usize), [60, 120, 255]));
                    }
                }
                radius = 0.16 * height;
                view_n = norm3([f.pos[0] - body[0], 0.0, f.pos[2] - body[2]]);
                if view_n == [0.0; 3] || view_n.iter().any(|x| x.is_nan()) {
                    view_n = [0.0, 0.0, -1.0];
                }
            }
            "intersect" => {
                hl.push(((f.mesh, f.tri), [255, 40, 40]));
                if let Some(t2) = f.tri2 {
                    hl.push((t2, [255, 220, 0]));
                }
                radius = 0.12 * height;
                view_n = norm3([f.pos[0] - body[0], f.pos[1] - body[1] * 0.0 - f.pos[1], f.pos[2] - body[2]]);
                if view_n.iter().any(|x| x.is_nan()) || view_n == [0.0; 3] {
                    view_n = [0.0, 0.0, -1.0];
                }
            }
            _ => {
                let p = tri_pts(f.mesh, f.tri);
                let e = |a: [f32; 3], b: [f32; 3]| dist3(a, b);
                let longest = e(p[0], p[1]).max(e(p[1], p[2])).max(e(p[2], p[0]));
                radius = (2.5 * longest).clamp(0.06 * height, 0.25 * height);
                let nn = norm3([
                    (p[1][1] - p[0][1]) * (p[2][2] - p[0][2]) - (p[1][2] - p[0][2]) * (p[2][1] - p[0][1]),
                    (p[1][2] - p[0][2]) * (p[2][0] - p[0][0]) - (p[1][0] - p[0][0]) * (p[2][2] - p[0][2]),
                    (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]),
                ]);
                let out_dir = [f.pos[0] - body[0], f.pos[1] - body[1], f.pos[2] - body[2]];
                view_n = if nn[0] * out_dir[0] + nn[1] * out_dir[1] + nn[2] * out_dir[2] >= 0.0 { nn } else { [-nn[0], -nn[1], -nn[2]] };
                if view_n.iter().any(|x| x.is_nan()) || view_n == [0.0; 3] {
                    view_n = [0.0, 0.0, -1.0];
                }
                // The triangle itself strong, triangles sharing a corner with it softer.
                let corner: Vec<u32> = model.meshes[f.mesh].tris[f.tri].to_vec();
                for (ti, t) in model.meshes[f.mesh].tris.iter().enumerate() {
                    if ti != f.tri && t.iter().any(|v| corner.contains(v)) {
                        hl.push(((f.mesh, ti), [255, 170, 120]));
                    }
                }
                hl.push(((f.mesh, f.tri), [255, 30, 30]));
                if let Some(v) = f.vertex {
                    for (ti, t) in model.meshes[f.mesh].tris.iter().enumerate() {
                        if t.contains(&(v as u32)) {
                            hl.push(((f.mesh, ti), [255, 30, 30]));
                        }
                    }
                }
            }
        }
        let mut cam = Camera::fit(model, 1.0, 0.9, true);
        cam.target = f.pos;
        cam.ortho = true;
        cam.ortho_half = radius;
        cam.dist = radius * 6.0;
        cam.pitch = view_n[1].clamp(-0.95, 0.95).asin();
        cam.yaw = view_n[0].atan2(view_n[2]);
        // Cutaway for intersections: drop what lies in front of the crossing plane.
        let depth_of = |p: [f32; 3]| (p[0] - f.pos[0]) * view_n[0] + (p[1] - f.pos[1]) * view_n[1] + (p[2] - f.pos[2]) * view_n[2];
        let slab = 0.004 * height;
        let keep = |mi: usize, ti: usize| -> bool {
            if k.name != "intersect" {
                return true;
            }
            let p = tri_pts(mi, ti);
            p.iter().any(|&q| depth_of(q) < slab * 0.0 + 0.0) || hl.iter().any(|h| h.0 == (mi, ti))
        };
        let tint = |mi: usize, ti: usize| -> Option<[u8; 3]> { hl.iter().rev().find(|h| h.0 == (mi, ti)).map(|h| h.1) };
        let opt = render::Options { cam, w: size, h: size, background: ([70, 76, 90], [40, 44, 54]), filter: Filter::Smooth, positions: Some(&posed) };
        let frame = render::render_masked(model, textures, &opt, &tint, Some(&keep));
        let mut rgba = frame.rgba;
        // Outlines of the named triangles drawn over everything, so a hidden one still shows.
        if k.name != "volume" {
            for &((mi, ti), col) in hl.iter().filter(|h| h.1 == [255, 30, 30] || h.1 == [255, 40, 40] || h.1 == [255, 220, 0]) {
                let p = tri_pts(mi, ti).map(|q| cam.project(size, size, q));
                for e in 0..3 {
                    line(&mut rgba, size, size, [p[e][0], p[e][1]], [p[(e + 1) % 3][0], p[(e + 1) % 3][1]], col);
                }
            }
        }
        // Marker: ring around the spot and the number.
        let sp = cam.project(size, size, f.pos);
        ring(&mut rgba, size, size, sp[0], sp[1], size as f32 * 0.11, [255, 255, 0]);
        ring(&mut rgba, size, size, sp[0], sp[1], size as f32 * 0.11 + 3.0, [0, 0, 0]);
        font::draw(&mut rgba, size, size, 6, 6, &format!("{}", f.n), 4, [200, 0, 0]);
        // Inset: whole body in the same pose with a dot at the spot.
        let isz = size / 3;
        let mut icam = Camera::fit(model, 1.0, 0.92, true);
        icam.yaw = cam.yaw;
        icam.pitch = 0.0;
        let iopt = render::Options { cam: icam, w: isz, h: isz, background: ([230, 230, 236], [200, 200, 210]), filter: Filter::Smooth, positions: Some(&posed) };
        let iframe = render::render(model, textures, &iopt, &|_, _| None);
        let mut irgba = iframe.rgba;
        let ip = icam.project(isz, isz, f.pos);
        ring(&mut irgba, isz, isz, ip[0], ip[1], 6.0, [255, 0, 0]);
        ring(&mut irgba, isz, isz, ip[0], ip[1], 2.0, [255, 0, 0]);
        for y in 0..isz {
            for x in 0..isz {
                let (dx, dy) = (size - isz + x, y);
                let edge = x == 0 || y == 0 || x == isz - 1 || y == isz - 1;
                let s = (y * isz + x) * 4;
                let d = (dy * size + dx) * 4;
                if edge {
                    rgba[d..d + 3].copy_from_slice(&[0, 0, 0]);
                } else {
                    rgba[d..d + 3].copy_from_slice(&irgba[s..s + 3]);
                }
            }
        }
        // Panel into the canvas.
        for y in 0..size {
            let s = y * size * 4;
            let d = ((oy + y) * w + ox) * 4;
            canvas[d..d + size * 4].copy_from_slice(&rgba[s..s + size * 4]);
        }
        let cx = ox as i64 + 6;
        let l1 = format!("#{} {}", f.n, if k.name == "weights" { "REST POSE".to_owned() } else { clean(&time_label(src, fr)) });
        let l2 = clean(&format!("{}{}", f.bone, f.bone2.as_ref().map(|b| format!(" / {b}")).unwrap_or_default()));
        let l3 = if k.name == "weights" { f.why.to_ascii_uppercase().replace('_', " ") } else { match f.ref_value {
            Some(r) => format!("{:.2} REFERENCE {:.2}", f.value, r),
            None => format!("{:.2}", f.value),
        } };
        let max_chars = (cw - 12) / 12;
        let cut = |s: String| -> String { s.chars().take(max_chars).collect() };
        font::draw(&mut canvas, w, h, cx, (oy + size + 6) as i64, &cut(l1), 2, [20, 20, 20]);
        font::draw(&mut canvas, w, h, cx, (oy + size + 24) as i64, &cut(l2), 2, [20, 20, 20]);
        font::draw(&mut canvas, w, h, cx, (oy + size + 42) as i64, &cut(l3), 2, [20, 20, 20]);
    }
    let mut png = Vec::new();
    image::RgbaImage::from_raw(w as u32, h as u32, canvas).ok_or("bad sheet size")?.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).map_err(|e| e.to_string())?;
    std::fs::write(path, png).map_err(|e| format!("{}: {e}", path.display()))
}
