//! `gaw weights`: copy skin weights from a reference model onto a new mesh that shares its
//! skeleton and rest pose (closest point on the reference surface, barycentric blend, a few rings
//! of smoothing inside each body region). Refuses when the skeletons differ.
use crate::Args;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use wb_core::weights::{self, Skeleton, TransferOptions};
use wb_core::Model;

pub const WEIGHTS_USAGE: &str = "  gaw weights NEW.glb --from REF.glb --out OUT.glb [--bones a,b | --zero-bones a,b] [--rings N] [--smooth X]
      [--max-dist D] [--report weights.json] [--use-battle-meshes] [--adopt-skeleton [--fit-tol F]] [--force]
      [--radial R [--soften K] [--keep-cuts]] [--repair N [--repair-reach N] [--repair-anchor A]] [--polish]
      Transfers skin weights from the skinned reference onto the new mesh: each vertex takes the weights of
      the closest point on the reference surface (barycentric blend of the triangle's three corners). No
      smoothing by default: stock's weights have hard cuts, and every smoothing ring measurably worsens the
      stock, remesh and noisy-remesh tests. --rings N (strength --smooth 0.5) adds N rings over the new mesh's
      welded topology for a jagged result; smoothing only mixes vertices of the same body region (L_arm, R_arm,
      L_leg, R_leg, head, tail, torso). At most 4 influences per vertex, sum 1, bone names and order kept,
      written as JOINTS_0/WEIGHTS_0 with the file's skin (the file's other data keeps its bytes).
      Refuses, with the reason, when NEW's skeleton is not REF's: joint count, names and order, parents,
      rest pose and inverse bind matrices (tolerance 0.001).
      --bones LIST        only these bones may receive weight (names, numbers, or text a name contains);
                          weight the reference puts on others moves to their nearest allowed parent
      --zero-bones LIST   the opposite: these bones get no weight (a tailless model: --zero-bones tail)
      --radial R          a body that is not stock's (other proportions, clothes over skin): match each vertex
                          near the same bone, distance across the bone scaled by R (0.25 is a good start),
                          and move stock's cuts between bones onto the new joints (--keep-cuts: leave them);
                          --soften K (default 0.5, 0 off) averages the weights of reference triangles about as
                          close as the best one. Default R 1 = plain closest point.
      --repair N          then pose the result in every clip (the reference's when NEW has none) and, up to N
                          passes, change the weights of vertices on edges that stretch or squash past
                          deform-check's limits: candidates from the vertex's neighbors and its own main bones,
                          the one with the least stretch and squash is kept (--repair-anchor A: cost of moving
                          away from the start, default 0.5; --repair-reach N: rings around a flagged vertex that
                          may change, default 0; --repair-mode soften: blur toward the neighbors instead).
                          Scored on stretch, squash, joint volume and new crossings against 1.5x the
                          reference; the best pass is kept. 30 to 60 s on a 15000-vertex body.
      --polish            keep NEW's own weights (from Blender, say) and only run --repair on them; no transfer.
      --max-dist D        flag vertices farther than D from the reference surface (default 5% of its height)
      --report FILE       weights.json: skeleton check, distance statistics, flagged vertices, influence
                          counts, vertices per bone and region
      --adopt-skeleton    NEW is a bare mesh (no armature, what an outside modeler hands over): copy REF's
                          armature, rest pose, skin and clips into it, then transfer. Refuses when the mesh is
                          not REF's size and place (height, width, depth and bounding box center each within
                          --fit-tol, default 0.10), printing the scale to apply. Refuses a file that already has
                          a skin, and meshes moved by a node transform (apply it in Blender first).
      The reference's battle_model* overlay meshes are not used as surface unless --use-battle-meshes.
      OUT must be a new file (--force replaces an earlier output; an input is never replaced).";

fn parse(a: &[String]) -> (Vec<String>, Vec<(String, String)>, Vec<String>) {
    let valued = ["--from", "--out", "--bones", "--zero-bones", "--rings", "--smooth", "--max-dist", "--report", "--fit-tol", "--radial", "--soften", "--joint-blend", "--repair", "--repair-smooth", "--repair-reach", "--repair-sharpen", "--repair-pull", "--repair-anchor", "--repair-mode"];
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

fn num<T: std::str::FromStr>(opts: &[(String, String)], k: &str) -> Result<Option<T>, String> {
    opts.iter().find(|o| o.0 == k).map(|o| o.1.parse::<T>().map_err(|_| format!("{k}: `{}` is not a number", o.1))).transpose()
}

fn repair_opts(opts: &[(String, String)], iters: usize) -> Result<wb_core::repair::RepairOptions, String> {
    let mut ro = wb_core::repair::RepairOptions { iters, ..Default::default() };
    if let Some(l) = num::<f32>(opts, "--repair-smooth")? {
        ro.lambda = l;
    }
    if let Some(x) = num::<f32>(opts, "--repair-pull")? {
        ro.pull = x;
    }
    if let Some(x) = num::<f32>(opts, "--repair-sharpen")? {
        ro.sharpen = x;
    }
    if let Some(n) = num::<usize>(opts, "--repair-reach")? {
        ro.reach = n;
    }
    if let Some(a) = num::<f32>(opts, "--repair-anchor")? {
        ro.anchor = a;
    }
    match opts.iter().find(|o| o.0 == "--repair-mode").map(|o| o.1.as_str()) {
        None | Some("search") => {}
        Some("soften") => ro.search = false,
        Some(x) => return Err(format!("--repair-mode: `{x}` is not search or soften")),
    }
    Ok(ro)
}

fn pct(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f32 * p).round() as usize]
}

pub fn weights_cmd(args: &Args) -> Result<(), String> {
    let (pos, opts, flags) = parse(&args.0);
    let get = |k: &str| opts.iter().find(|o| o.0 == k).map(|o| o.1.as_str());
    let new_path = PathBuf::from(pos.first().ok_or("weights needs NEW.glb (the new mesh)")?);
    let adopt = flags.iter().any(|f| f == "--adopt-skeleton");
    // `--adopt-skeleton REF.glb` also reads: the flag takes no value, so REF.glb lands in the positionals.
    let ref_arg = get("--from").map(str::to_owned).or_else(|| if adopt { pos.get(1).cloned() } else { None });
    let ref_path = PathBuf::from(ref_arg.ok_or("weights needs --from REF.glb (the skinned reference)")?);
    let out_path = PathBuf::from(get("--out").ok_or("weights needs --out OUT.glb")?);
    let same = |a: &Path, b: &Path| std::fs::canonicalize(a).ok().zip(std::fs::canonicalize(b).ok()).is_some_and(|(x, y)| x == y);
    if same(&out_path, &new_path) || same(&out_path, &ref_path) {
        return Err("--out would replace an input file; choose a new file name".into());
    }
    if out_path.exists() && !flags.iter().any(|f| f == "--force") {
        return Err(format!("{} already exists; choose a new file name (--force replaces an earlier output)", out_path.display()));
    }
    let mut new = Model::open(&new_path)?;
    let reference = Model::open(&ref_path)?;
    let mut adopted: Option<Value> = None;
    if adopt {
        let tol = num::<f32>(&opts, "--fit-tol")?.unwrap_or(0.10);
        if !(0.0..=1.0).contains(&tol) {
            return Err("--fit-tol must be between 0 and 1".into());
        }
        if !new.glb.list("skins").is_empty() {
            return Err(format!("{} already has a skin; --adopt-skeleton is for meshes without an armature (drop the flag to compare its skeleton with the reference)", new_path.display()));
        }
        let moved = wb_core::adopt::moved_mesh_chain(&new.glb);
        if !moved.is_empty() {
            return Err(format!("refusing: mesh node(s) {} carry a transform (their own or a parent's). Once skinned, the transform is ignored, so the mesh would jump. Apply the transforms in Blender (Object > Apply > All Transforms) and export again.", moved.join(", ")));
        }
        let fit = wb_core::adopt::fit_check(&new, &reference, tol)?;
        if !fit.problems.is_empty() {
            let mut msg = format!("refusing: {} does not fit {} ({} problem(s); reference box {:.3}..{:.3} high {:.3}, new box high {:.3}, scale needed {:.4}):", new_path.display(), ref_path.display(), fit.problems.len(), fit.ref_lo[1], fit.ref_hi[1], fit.ref_hi[1] - fit.ref_lo[1], fit.new_hi[1] - fit.new_lo[1], fit.scale);
            for p in &fit.problems {
                msg.push_str("\n  - ");
                msg.push_str(p);
            }
            return Err(msg);
        }
        let bytes = wb_core::adopt::adopt_skeleton(&new, &reference)?;
        new = Model::from_bytes(&bytes)?;
        adopted = Some(json!({"tolerance": tol, "scale_needed": fit.scale, "reference_box": {"min": fit.ref_lo, "max": fit.ref_hi}, "new_box": {"min": fit.new_lo, "max": fit.new_hi}, "joints": new.joint_names.len(), "clips": new.clips}));
        println!("adopted    {} joints and {} clip(s) copied from {}; mesh fits (height x{:.4} of the reference's)", new.joint_names.len(), new.clips.len(), ref_path.display(), 1.0 / fit.scale);
    }

    // Same skeleton or no transfer.
    let ns = Skeleton::from_model(&new).map_err(|e| format!("{}: {e}. Weights need the new mesh to carry the reference's skeleton (same bones, order, parents and rest pose).", new_path.display()))?;
    let rs = Skeleton::from_model(&reference).map_err(|e| format!("{}: {e}", ref_path.display()))?;
    let problems = weights::skeleton_diff(&ns, &rs, 1e-3);
    if !problems.is_empty() {
        let mut msg = format!("refusing: {} does not have the skeleton of {} ({} problem(s)):", new_path.display(), ref_path.display(), problems.len());
        for p in &problems {
            msg.push_str("\n  - ");
            msg.push_str(p);
        }
        msg.push_str("\nFit the mesh to the reference rig (same bone names, order, parents and rest pose) before transferring weights.");
        return Err(msg);
    }

    // --polish: keep the file's own weights (from Blender, say) and only run the pose repair on them.
    if flags.iter().any(|f| f == "--polish") {
        if new.glb.list("skins").is_empty() || new.meshes.iter().any(|m| m.joints.is_empty()) {
            return Err(format!("{} has no weights to polish; run the transfer first", new_path.display()));
        }
        let iters = num::<usize>(&opts, "--repair")?.unwrap_or(60);
        let ro = repair_opts(&opts, iters)?;
        let t0 = std::time::Instant::now();
        let res = wb_core::repair::repair(&mut new, Some(&reference), &ro)?;
        let joints: Vec<_> = new.meshes.iter().map(|m| m.joints.clone()).collect();
        let weights: Vec<_> = new.meshes.iter().map(|m| m.weights.clone()).collect();
        let bytes = weights::write_weights(&new, &joints, &weights)?;
        if let Some(dir) = out_path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        std::fs::write(&out_path, &bytes).map_err(|e| format!("{}: {e}", out_path.display()))?;
        println!("polish     {} pass(es), best at {}; pose score {:.1} -> {:.1}; {} welded places changed ({:.1} s)", res.iterations, res.best_iteration, res.score_before, res.score_after, res.groups_touched, t0.elapsed().as_secs_f32());
        println!("wrote      {}", out_path.display());
        if let Some(rp) = get("--report") {
            let j = json!({"gaw_weights_polish": 1, "file": new_path.display().to_string(), "reference": ref_path.display().to_string(), "iterations": res.iterations, "best_iteration": res.best_iteration, "score_before": res.score_before, "score_after": res.score_after, "places_changed": res.groups_touched, "smooth": ro.lambda, "reach": ro.reach, "sharpen": ro.sharpen, "pull": ro.pull});
            std::fs::write(rp, serde_json::to_string_pretty(&j).unwrap_or_default()).map_err(|e| format!("{rp}: {e}"))?;
        }
        return Ok(());
    }

    let mut o = TransferOptions::default();
    if let Some(r) = num::<usize>(&opts, "--rings")? {
        o.rings = r;
    }
    if let Some(s) = num::<f32>(&opts, "--smooth")? {
        if !(0.0..=1.0).contains(&s) {
            return Err("--smooth must be between 0 and 1".into());
        }
        o.lambda = s;
    }
    if let Some(r) = num::<f32>(&opts, "--radial")? {
        if !(0.01..=1.0).contains(&r) {
            return Err("--radial must be between 0.01 and 1".into());
        }
        o.radial = r;
    }
    if let Some(k) = num::<f32>(&opts, "--soften")? {
        if k < 0.0 {
            return Err("--soften must be 0 or more".into());
        }
        o.soften = k;
    }
    if let Some(w) = num::<f32>(&opts, "--joint-blend")? {
        if w < 0.0 {
            return Err("--joint-blend must be 0 or more".into());
        }
        o.joint_blend = w;
    }
    o.max_dist = num::<f32>(&opts, "--max-dist")?;
    o.include_battle = flags.iter().any(|f| f == "--use-battle-meshes");
    o.align_cuts = !flags.iter().any(|f| f == "--keep-cuts");
    match (get("--bones"), get("--zero-bones")) {
        (Some(_), Some(_)) => return Err("use --bones or --zero-bones, not both".into()),
        (Some(b), None) => o.allowed = Some(weights::parse_bone_list(b, &rs.names)?),
        (None, Some(z)) => o.allowed = Some(weights::parse_bone_list(z, &rs.names)?.into_iter().map(|b| !b).collect()),
        _ => {}
    }

    let t0 = std::time::Instant::now();
    let mut r = weights::transfer(&new, &reference, &rs, &o)?;
    let mut bytes = weights::write_weights(&new, &r.joints, &r.weights)?;
    let mut repaired: Option<Value> = None;
    let repair_iters = num::<usize>(&opts, "--repair")?.unwrap_or(0);
    if repair_iters > 0 {
        let ro = repair_opts(&opts, repair_iters)?;
        let mut m = Model::from_bytes(&bytes)?;
        let res = wb_core::repair::repair(&mut m, Some(&reference), &ro)?;
        for (mi, mesh) in m.meshes.iter().enumerate() {
            r.joints[mi] = mesh.joints.clone();
            r.weights[mi] = mesh.weights.clone();
        }
        bytes = weights::write_weights(&new, &r.joints, &r.weights)?;
        println!("repair     {} pass(es), best at {}; pose score {:.1} -> {:.1}; {} welded places softened", res.iterations, res.best_iteration, res.score_before, res.score_after, res.groups_touched);
        repaired = Some(json!({"iterations": res.iterations, "best_iteration": res.best_iteration, "score_before": res.score_before, "score_after": res.score_after, "places_softened": res.groups_touched, "smooth": ro.lambda}));
    }
    if let Some(dir) = out_path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(&out_path, &bytes).map_err(|e| format!("{}: {e}", out_path.display()))?;

    // Report.
    let nj = rs.names.len();
    let mut all_d: Vec<f32> = r.dist.iter().flatten().copied().collect();
    all_d.sort_by(|a, b| a.total_cmp(b));
    let total: usize = all_d.len();
    let mut far: Vec<(f32, usize, usize)> = Vec::new();
    for (mi, ds) in r.dist.iter().enumerate() {
        for (vi, &d) in ds.iter().enumerate() {
            if d > r.max_dist {
                far.push((d, mi, vi));
            }
        }
    }
    far.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut influences = [0usize; 5];
    let mut dominant = vec![0usize; nj];
    let mut by_region = vec![0usize; r.regions.len()];
    for mi in 0..new.meshes.len() {
        for vi in 0..new.meshes[mi].pos.len() {
            influences[r.weights[mi][vi].iter().filter(|&&w| w > 0.0).count()] += 1;
            dominant[r.joints[mi][vi][0] as usize] += 1;
            by_region[r.region[mi][vi] as usize] += 1;
        }
    }
    let per_bone: Value = rs.names.iter().zip(&dominant).map(|(n, &c)| (n.clone(), json!(c))).collect::<serde_json::Map<_, _>>().into();
    let moved = weights::moved_mesh_nodes(&new.glb);
    let mut notes: Vec<String> = Vec::new();
    if !moved.is_empty() {
        notes.push(format!("mesh node(s) {} carry their own transform; skinned meshes ignore it, and so does this check, so vertex positions are taken as stored", moved.join(", ")));
    }
    if !far.is_empty() {
        notes.push(format!("{} of {total} vertices lie farther than {:.4} from the reference surface; they took the closest point's weights anyway (look at them in deform-check)", far.len(), r.max_dist));
    }
    let report = json!({
        "gaw_weights": 1,
        "new": new_path.display().to_string(), "reference": ref_path.display().to_string(), "out": out_path.display().to_string(),
        "skeleton": {"match": true, "joints": nj, "tolerance": 0.001, "adopted_from_reference": adopted},
        "params": {"radial": o.radial, "repair": repaired, "soften": o.soften, "joint_blend": o.joint_blend, "align_cuts": o.align_cuts, "rings": o.rings, "smooth": o.lambda, "max_dist": r.max_dist, "max_dist_given": o.max_dist.is_some(), "reference_meshes": r.ref_meshes, "allowed_bones": o.allowed.as_ref().map(|a| rs.names.iter().zip(a).filter(|(_, &ok)| ok).map(|(n, _)| n.clone()).collect::<Vec<_>>()), "reference_height": r.ref_height},
        "meshes": new.meshes.iter().map(|m| json!({"name": m.name, "vertices": m.pos.len(), "triangles": m.tris.len()})).collect::<Vec<_>>(),
        "welded_places": r.welded_groups,
        "distance": {"p50": pct(&all_d, 0.5), "p95": pct(&all_d, 0.95), "p99": pct(&all_d, 0.99), "max": all_d.last().copied().unwrap_or(0.0), "limit": r.max_dist, "flagged": far.len(), "vertices": total},
        "far_vertices": far.iter().take(200).map(|&(d, mi, vi)| json!({"mesh": new.meshes[mi].name, "vertex": vi, "distance": d, "position": new.meshes[mi].pos[vi], "reference_mesh": reference.meshes[r.ref_tri[mi][vi].0 as usize].name, "reference_triangle": r.ref_tri[mi][vi].1, "region": r.regions[r.region[mi][vi] as usize]})).collect::<Vec<_>>(),
        "influences": {"1": influences[1], "2": influences[2], "3": influences[3], "4": influences[4], "over_4": 0, "weight_sum": "1 (checked on read back)"},
        "vertices_by_dominant_bone": per_bone,
        "vertices_by_region": r.regions.iter().zip(&by_region).map(|(n, &c)| (n.clone(), json!(c))).collect::<serde_json::Map<_, _>>(),
        "notes": notes,
        "seconds": t0.elapsed().as_secs_f32(),
    });
    let report_path = get("--report").map(PathBuf::from);
    if let Some(p) = &report_path {
        if let Some(dir) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        std::fs::write(p, serde_json::to_vec_pretty(&report).unwrap()).map_err(|e| format!("{}: {e}", p.display()))?;
    }

    println!("skeleton   matches the reference ({nj} joints, same names, order, parents and rest pose)");
    println!("transfer   {} vertices ({} welded places) from {} reference triangles; {} smoothing ring(s) at {}", total, r.welded_groups, r.ref_meshes.join("+"), o.rings, o.lambda);
    println!("distance   median {:.4}, p95 {:.4}, max {:.4}; {} vertices beyond {:.4}", pct(&all_d, 0.5), pct(&all_d, 0.95), all_d.last().copied().unwrap_or(0.0), far.len(), r.max_dist);
    println!("influences 1: {}  2: {}  3: {}  4: {}  (none above 4; every vertex sums to 1)", influences[1], influences[2], influences[3], influences[4]);
    if let Some(a) = &o.allowed {
        println!("allowed    {} of {} bones may take weight", a.iter().filter(|&&b| b).count(), nj);
    }
    for n in &notes {
        println!("note       {n}");
    }
    println!("wrote {} ({} bytes){}", out_path.display(), bytes.len(), report_path.map(|p| format!(" and {}", p.display())).unwrap_or_default());
    Ok(())
}
