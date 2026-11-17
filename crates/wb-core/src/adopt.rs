//! `gaw weights --adopt-skeleton`: give a mesh that has no armature (what an outside modeler hands
//! over) the reference's armature and rest pose, so the weight transfer can run on it.
//!
//! The mesh keeps its bytes, materials and textures. The reference's joint nodes (and the nodes above
//! them), its skin with the inverse bind matrices, and its animation clips are copied in; every mesh
//! node points at the new skin. Before that, `fit_check` refuses a mesh that is not the reference's
//! size and place, because the rest pose only fits a mesh that stands where the reference stands.
use crate::glb::{Chunk, Glb, CHUNK_BIN, CHUNK_JSON};
use crate::weights::{skeleton_diff, Skeleton};
use crate::Model;
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};

/// How the new mesh's bounding box compares with the reference's.
pub struct Fit {
    pub ref_lo: [f32; 3],
    pub ref_hi: [f32; 3],
    pub new_lo: [f32; 3],
    pub new_hi: [f32; 3],
    /// Uniform scale that makes the new mesh as tall as the reference.
    pub scale: f32,
    /// Empty when the mesh fits within the tolerance.
    pub problems: Vec<String>,
}

fn bbox(m: &Model, skip_battle: bool) -> Option<([f32; 3], [f32; 3])> {
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    let mut any = false;
    for mesh in m.meshes.iter().filter(|x| !(skip_battle && x.name.starts_with("battle_model"))) {
        for p in &mesh.pos {
            any = true;
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    any.then_some((lo, hi))
}

/// Compares height, width, depth and position of the new mesh with the reference's (rest pose, battle
/// overlay meshes left out). `tol` is relative: 0.10 allows 10 percent on each extent and a bounding
/// box center off by 10 percent of the reference's height. Extents and center are judged after the
/// height-matching scale, so a unit mix-up gives one message, the scale to apply.
pub fn fit_check(new: &Model, reference: &Model, tol: f32) -> Result<Fit, String> {
    let (rlo, rhi) = bbox(reference, true).ok_or("the reference has no vertices")?;
    let (nlo, nhi) = bbox(new, false).ok_or("the new file has no vertices")?;
    let ext = |lo: [f32; 3], hi: [f32; 3]| -> [f32; 3] { std::array::from_fn(|k| hi[k] - lo[k]) };
    let (re, ne) = (ext(rlo, rhi), ext(nlo, nhi));
    if ne[1] <= 1e-9 || re[1] <= 1e-9 {
        return Err("a model with no height cannot be compared".into());
    }
    let scale = re[1] / ne[1];
    let mut problems = Vec::new();
    if (scale - 1.0).abs() > tol {
        let hint = if scale > 50.0 || scale < 0.02 { " (a factor near 100 usually means centimeters against meters, or FF9's 0.01 model scale)" } else { "" };
        problems.push(format!(
            "height {:.4} against the reference's {:.4} ({:+.1}%): scale the mesh by {:.4} about the origin in Blender (Apply Scale), then export again{hint}",
            ne[1],
            re[1],
            (ne[1] / re[1] - 1.0) * 100.0,
            scale
        ));
    }
    let after = if (scale - 1.0).abs() > tol { " after that scale" } else { "" };
    for (k, name) in [(0usize, "width (x)"), (2, "depth (z)")] {
        if re[k] > 1e-6 {
            let r = ne[k] * scale / re[k];
            if (r - 1.0).abs() > tol {
                problems.push(format!("{name} {:.4}{after} against the reference's {:.4} ({:+.1}%): the shape is not the reference's size (limit {:.0}%; --fit-tol widens it)", ne[k] * scale, re[k], (r - 1.0) * 100.0, tol * 100.0));
            }
        }
    }
    let off: [f32; 3] = std::array::from_fn(|k| (nlo[k] + nhi[k]) * 0.5 * scale - (rlo[k] + rhi[k]) * 0.5);
    let lim = tol * re[1];
    if off.iter().any(|d| d.abs() > lim) {
        problems.push(format!(
            "the bounding box center is off by ({:+.4}, {:+.4}, {:+.4}){after} from the reference's; limit {:.4} on each axis. Move the mesh so it stands where the reference stands (feet on the same height, centered on the same spot)",
            off[0], off[1], off[2], lim
        ));
    }
    Ok(Fit { ref_lo: rlo, ref_hi: rhi, new_lo: nlo, new_hi: nhi, scale, problems })
}

/// Mesh nodes that are moved by their own transform or by an ancestor's. A skinned mesh ignores node
/// transforms, so adopting a skeleton would silently drop them.
pub fn moved_mesh_chain(glb: &Glb) -> Vec<String> {
    let nodes = glb.list("nodes");
    let mut parent: HashMap<usize, usize> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        for c in n.get("children").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_u64) {
            parent.insert(c as usize, i);
        }
    }
    let moved = |n: &Value| -> bool {
        let f = |k: &str, d: &[f64]| n.get(k).and_then(Value::as_array).map(|a| a.iter().zip(d).any(|(x, &y)| (x.as_f64().unwrap_or(y) - y).abs() > 1e-6)).unwrap_or(false);
        f("translation", &[0.0; 3]) || f("rotation", &[0.0, 0.0, 0.0, 1.0]) || f("scale", &[1.0; 3]) || n.get("matrix").is_some_and(|m| {
            let id = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
            m.as_array().map(|a| a.iter().zip(id).any(|(x, y)| (x.as_f64().unwrap_or(y) - y).abs() > 1e-6)).unwrap_or(false)
        })
    };
    let mut out = Vec::new();
    for (i, n) in nodes.iter().enumerate() {
        if n.get("mesh").is_none() {
            continue;
        }
        let (mut at, mut bad) = (Some(i), None);
        while let Some(a) = at {
            if moved(&nodes[a]) {
                bad = Some(a);
                break;
            }
            at = parent.get(&a).copied();
        }
        if let Some(b) = bad {
            let name = |k: usize| nodes[k].get("name").and_then(Value::as_str).unwrap_or("?").to_owned();
            out.push(if b == i { name(i) } else { format!("{} (through its parent {})", name(i), name(b)) });
        }
    }
    out
}

fn add_view(json: &mut Value, bin: &mut Vec<u8>, bytes: &[u8]) -> usize {
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let off = bin.len();
    bin.extend_from_slice(bytes);
    let views = json["bufferViews"].as_array_mut().expect("bufferViews");
    views.push(json!({ "buffer": 0, "byteOffset": off, "byteLength": bytes.len() }));
    views.len() - 1
}

fn add_accessor(json: &mut Value, a: Value) -> usize {
    let list = json["accessors"].as_array_mut().expect("accessors");
    list.push(a);
    list.len() - 1
}

/// Copies one accessor of the reference (as floats) into the new file; returns its new index.
fn copy_accessor(reference: &Glb, old: usize, json: &mut Value, bin: &mut Vec<u8>, seen: &mut HashMap<usize, usize>) -> Result<usize, String> {
    if let Some(&n) = seen.get(&old) {
        return Ok(n);
    }
    let a = reference.accessor(old)?;
    let ty = match a.ncomp {
        1 => "SCALAR",
        3 => "VEC3",
        4 => "VEC4",
        n => return Err(format!("reference animation accessor {old} has {n} components")),
    };
    let bytes: Vec<u8> = a.data.iter().flat_map(|&x| (x as f32).to_le_bytes()).collect();
    let view = add_view(json, bin, &bytes);
    let mut acc = json!({"bufferView": view, "componentType": 5126, "count": a.count, "type": ty});
    if a.count > 0 {
        let (mut lo, mut hi) = (vec![f64::MAX; a.ncomp], vec![f64::MIN; a.ncomp]);
        for i in 0..a.count {
            for c in 0..a.ncomp {
                lo[c] = lo[c].min(a.get(i, c) as f32 as f64);
                hi[c] = hi[c].max(a.get(i, c) as f32 as f64);
            }
        }
        acc["min"] = json!(lo);
        acc["max"] = json!(hi);
    }
    let n = add_accessor(json, acc);
    seen.insert(old, n);
    Ok(n)
}

/// The new file's mesh data with the reference's armature, skin and clips added. The result has no
/// JOINTS_0/WEIGHTS_0 yet; run the weight transfer and `write_weights` on it.
pub fn adopt_skeleton(new: &Model, reference: &Model) -> Result<Vec<u8>, String> {
    if !new.glb.list("skins").is_empty() {
        return Err("the file already has a skin; --adopt-skeleton is for meshes without an armature (drop the flag to compare its own skeleton with the reference)".into());
    }
    let rskin = reference.glb.list("skins").first().ok_or("the reference has no skin to adopt")?;
    let rig = reference.rig.as_ref().ok_or("the reference has no skeleton to adopt")?;
    let rnodes = reference.glb.list("nodes");
    let joints: Vec<usize> = rskin.get("joints").and_then(Value::as_array).ok_or("the reference's skin has no joints")?.iter().filter_map(Value::as_u64).map(|j| j as usize).collect();
    let ibm = rig.inverse_binds();
    if ibm.len() != joints.len() {
        return Err("the reference's inverse bind matrices do not match its joint list".into());
    }
    let mut parent: HashMap<usize, usize> = HashMap::new();
    for (i, n) in rnodes.iter().enumerate() {
        for c in n.get("children").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_u64) {
            parent.insert(c as usize, i);
        }
    }
    let mut keep: BTreeSet<usize> = BTreeSet::new();
    let skeleton_root = rskin.get("skeleton").and_then(Value::as_u64).map(|s| s as usize);
    for &j in joints.iter().chain(skeleton_root.iter()) {
        let mut at = Some(j);
        while let Some(a) = at {
            if !keep.insert(a) {
                break;
            }
            at = parent.get(&a).copied();
        }
    }

    let mut json = new.glb.json.clone();
    let mut bin = new.glb.bin().to_vec();
    for k in ["nodes", "bufferViews", "accessors"] {
        if !json[k].is_array() {
            json[k] = json!([]);
        }
    }
    if json["buffers"].as_array().map(|b| b.is_empty()).unwrap_or(true) {
        json["buffers"] = json!([{"byteLength": bin.len()}]);
    }
    let new_roots_before: Vec<usize> = {
        let nn = json["nodes"].as_array().unwrap();
        let mut child = vec![false; nn.len()];
        for n in nn {
            for c in n.get("children").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_u64) {
                if (c as usize) < child.len() {
                    child[c as usize] = true;
                }
            }
        }
        (0..nn.len()).filter(|&i| !child[i]).collect()
    };
    let base = json["nodes"].as_array().unwrap().len();
    let remap: HashMap<usize, usize> = keep.iter().enumerate().map(|(k, &o)| (o, base + k)).collect();
    let mut added: Vec<Value> = Vec::new();
    for &o in &keep {
        let mut n = rnodes[o].clone();
        let obj = n.as_object_mut().ok_or("a reference node is not an object")?;
        for k in ["mesh", "skin", "camera", "weights"] {
            obj.remove(k);
        }
        let kids: Vec<Value> = rnodes[o].get("children").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_u64).filter_map(|c| remap.get(&(c as usize))).map(|&c| json!(c)).collect();
        if kids.is_empty() {
            obj.remove("children");
        } else {
            obj.insert("children".into(), Value::Array(kids));
        }
        added.push(n);
    }
    let roots: Vec<usize> = keep.iter().filter(|o| !parent.contains_key(o)).map(|o| remap[o]).collect();
    json["nodes"].as_array_mut().unwrap().extend(added);

    // Scene: the new roots go beside whatever the file already has.
    if json["scenes"].as_array().map(|s| s.is_empty()).unwrap_or(true) {
        json["scenes"] = json!([{ "nodes": new_roots_before }]);
        json["scene"] = json!(0);
    }
    let nscenes = json["scenes"].as_array().map(|s| s.len()).unwrap_or(1);
    let si = (json.get("scene").and_then(Value::as_u64).unwrap_or(0) as usize).min(nscenes - 1);
    if json["scenes"][si]["nodes"].is_array() {
        json["scenes"][si]["nodes"].as_array_mut().unwrap().extend(roots.iter().map(|&r| json!(r)));
    } else {
        json["scenes"][si]["nodes"] = json!(roots);
    }

    // Skin with the reference's inverse bind matrices.
    let ibm_bytes: Vec<u8> = ibm.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
    let v = add_view(&mut json, &mut bin, &ibm_bytes);
    let ia = add_accessor(&mut json, json!({"bufferView": v, "componentType": 5126, "count": ibm.len(), "type": "MAT4"}));
    let mut skin = json!({ "joints": joints.iter().map(|j| remap[j]).collect::<Vec<_>>(), "inverseBindMatrices": ia });
    if let Some(s) = skeleton_root.and_then(|s| remap.get(&s)) {
        skin["skeleton"] = json!(s);
    }
    json["skins"] = json!([skin]);
    if let Some(nodes) = json["nodes"].as_array_mut() {
        for n in nodes.iter_mut() {
            if n.get("mesh").is_some() {
                n["skin"] = json!(0);
            }
        }
    }

    // Clips: the same samplers and channels, pointed at the copied nodes.
    let mut seen: HashMap<usize, usize> = HashMap::new();
    let mut anims: Vec<Value> = json.get("animations").and_then(Value::as_array).cloned().unwrap_or_default();
    for (ai, a) in reference.glb.list("animations").iter().enumerate() {
        let mut samplers = Vec::new();
        for s in a.get("samplers").and_then(Value::as_array).into_iter().flatten() {
            let inp = copy_accessor(&reference.glb, s["input"].as_u64().ok_or("animation sampler without input")? as usize, &mut json, &mut bin, &mut seen)?;
            let out = copy_accessor(&reference.glb, s["output"].as_u64().ok_or("animation sampler without output")? as usize, &mut json, &mut bin, &mut seen)?;
            samplers.push(json!({"input": inp, "output": out, "interpolation": s.get("interpolation").and_then(Value::as_str).unwrap_or("LINEAR")}));
        }
        let mut channels = Vec::new();
        for c in a.get("channels").and_then(Value::as_array).into_iter().flatten() {
            let node = c["target"]["node"].as_u64().map(|n| n as usize);
            if let Some(&nn) = node.and_then(|n| remap.get(&n)) {
                channels.push(json!({"sampler": c["sampler"], "target": {"node": nn, "path": c["target"]["path"]}}));
            }
        }
        anims.push(json!({"name": a.get("name").and_then(Value::as_str).map(str::to_owned).unwrap_or_else(|| format!("clip{ai}")), "samplers": samplers, "channels": channels}));
    }
    if !anims.is_empty() {
        json["animations"] = Value::Array(anims);
    }

    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    json["buffers"][0]["byteLength"] = json!(bin.len());
    let mut jb = serde_json::to_vec(&json).map_err(|e| e.to_string())?;
    while jb.len() % 4 != 0 {
        jb.push(b' ');
    }
    let mut chunks = vec![Chunk { kind: CHUNK_JSON, data: jb }, Chunk { kind: CHUNK_BIN, data: bin }];
    chunks.extend(new.glb.chunks.iter().skip(1).filter(|c| c.kind != CHUNK_BIN).map(|c| Chunk { kind: c.kind, data: c.data.clone() }));
    let out = Glb { version: new.glb.version, chunks, json: Value::Null }.to_bytes();

    // Read back: the armature must be the reference's, bit for bit as far as the checks see it.
    let back = Model::from_bytes(&out)?;
    let problems = skeleton_diff(&Skeleton::from_model(&back)?, &Skeleton::from_model(reference)?, 1e-4);
    if !problems.is_empty() {
        return Err(format!("the copied skeleton does not match the reference ({}); this is a bug", problems.join("; ")));
    }
    if back.meshes.len() != new.meshes.len() || back.meshes.iter().zip(&new.meshes).any(|(a, b)| a.pos != b.pos || a.tris != b.tris || a.uv != b.uv) {
        return Err("adopting the skeleton changed the mesh data; this is a bug".into());
    }
    Ok(out)
}
