//! Weight transfer and skeleton matching on tiny synthetic GLBs: a two-bone reference strip with a
//! weight ramp, and denser or different new meshes on the same (or a different) skeleton.
use serde_json::json;
use wb_core::weights::{self, Skeleton, TransferOptions};
use wb_core::Model;

struct Bone(&'static str, Option<usize>, [f32; 3]);

fn pad4(v: &mut Vec<u8>) {
    while v.len() % 4 != 0 {
        v.push(0);
    }
}

/// Bones are translation-only, so the world position of a bone is the sum of translations up its chain.
fn build(bones: &[Bone], pos: &[[f32; 3]], tris: &[[u16; 3]], joints: &[[u8; 4]], weights: &[[f32; 4]]) -> Vec<u8> {
    let mut bin: Vec<u8> = Vec::new();
    let mut views = Vec::new();
    let mut accessors = Vec::new();
    let mut add = |bin: &mut Vec<u8>, data: Vec<u8>, comp: u32, count: usize, ty: &str, mm: Option<(Vec<f32>, Vec<f32>)>| {
        pad4(bin);
        views.push(json!({"buffer": 0, "byteOffset": bin.len(), "byteLength": data.len()}));
        bin.extend_from_slice(&data);
        let mut a = json!({"bufferView": views.len() - 1, "componentType": comp, "count": count, "type": ty});
        if let Some((lo, hi)) = mm {
            a["min"] = json!(lo);
            a["max"] = json!(hi);
        }
        accessors.push(a);
        accessors.len() - 1
    };
    let f32b = |v: Vec<f32>| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let a_pos = add(&mut bin, f32b(pos.iter().flatten().copied().collect()), 5126, pos.len(), "VEC3", Some((vec![-9.0; 3], vec![9.0; 3])));
    let a_j = add(&mut bin, joints.iter().flatten().copied().collect(), 5121, joints.len(), "VEC4", None);
    let a_w = add(&mut bin, f32b(weights.iter().flatten().copied().collect()), 5126, weights.len(), "VEC4", None);
    let a_i = add(&mut bin, tris.iter().flatten().flat_map(|i| i.to_le_bytes()).collect(), 5123, tris.len() * 3, "SCALAR", None);
    let mut world: Vec<[f32; 3]> = Vec::new();
    for Bone(_, p, t) in bones {
        let base = p.map(|p| world[p]).unwrap_or([0.0; 3]);
        world.push([base[0] + t[0], base[1] + t[1], base[2] + t[2]]);
    }
    let mut ibm: Vec<f32> = Vec::new();
    for w in &world {
        ibm.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -w[0], -w[1], -w[2], 1.0]);
    }
    let a_ibm = add(&mut bin, f32b(ibm), 5126, bones.len(), "MAT4", None);
    let mut nodes: Vec<serde_json::Value> = bones
        .iter()
        .enumerate()
        .map(|(i, Bone(n, _, t))| {
            let kids: Vec<usize> = bones.iter().enumerate().filter(|(_, b)| b.1 == Some(i)).map(|(k, _)| k).collect();
            let mut o = json!({"name": n, "translation": t});
            if !kids.is_empty() {
                o["children"] = json!(kids);
            }
            o
        })
        .collect();
    nodes.push(json!({"name": "mesh0", "mesh": 0, "skin": 0}));
    pad4(&mut bin);
    let doc = json!({
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, bones.len()]}], "nodes": nodes,
        "skins": [{"joints": (0..bones.len()).collect::<Vec<_>>(), "inverseBindMatrices": a_ibm, "skeleton": 0}],
        "meshes": [{"name": "mesh0", "primitives": [{"attributes": {"POSITION": a_pos, "JOINTS_0": a_j, "WEIGHTS_0": a_w}, "indices": a_i}]}],
        "bufferViews": views, "accessors": accessors, "buffers": [{"byteLength": bin.len()}],
    });
    let mut js = serde_json::to_vec(&doc).unwrap();
    while js.len() % 4 != 0 {
        js.push(b' ');
    }
    let total = 12 + 8 + js.len() + 8 + bin.len();
    let mut out = b"glTF".to_vec();
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(js.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&js);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin);
    out
}

fn rig() -> Vec<Bone> {
    vec![Bone("bone000_hips", None, [0.0, 0.0, 0.0]), Bone("bone001_L_arm", Some(0), [0.0, 1.0, 0.0]), Bone("bone002_tail", Some(0), [0.0, 0.0, -1.0])]
}

/// Strip x in {0,1}, y in {0,1,2}: y=0 all bone 0, y=1 half and half, y=2 all bone 1.
fn reference() -> Model {
    let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0]];
    let tris = [[0, 1, 3], [0, 3, 2], [2, 3, 5], [2, 5, 4]];
    let j = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]];
    let w = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
    Model::from_bytes(&build(&rig(), &pos, &tris, &j, &w)).unwrap()
}

fn new_with(bones: &[Bone]) -> Model {
    // Single triangle at y 0.5 .. 1.5, slightly in front of the strip; weights are placeholders.
    let pos = [[0.2, 0.5, 0.05], [0.8, 0.5, 0.05], [0.5, 1.5, 0.05]];
    let j = [[0, 0, 0, 0]; 3];
    let w = [[1.0, 0.0, 0.0, 0.0]; 3];
    Model::from_bytes(&build(bones, &pos, &[[0, 1, 2]], &j, &w)).unwrap()
}

fn weight_of(r: &weights::TransferResult, v: usize, joint: u16) -> f32 {
    (0..4).filter(|&k| r.joints[0][v][k] == joint).map(|k| r.weights[0][v][k]).sum()
}

#[test]
fn weights_follow_the_closest_reference_surface() {
    let refm = reference();
    let newm = new_with(&rig());
    let skel = Skeleton::from_model(&refm).unwrap();
    assert!(weights::skeleton_diff(&Skeleton::from_model(&newm).unwrap(), &skel, 1e-3).is_empty());
    let o = TransferOptions { rings: 0, ..TransferOptions::default() };
    let r = weights::transfer(&newm, &refm, &skel, &o).unwrap();
    // y=0.5 sits half way between the all-hips row and the half-and-half row: arm share 0.25.
    assert!((weight_of(&r, 0, 1) - 0.25).abs() < 1e-3, "{:?}", r.weights[0][0]);
    assert!((weight_of(&r, 0, 0) - 0.75).abs() < 1e-3);
    // y=1.5 is half way between half-and-half and all-arm: arm share 0.75.
    assert!((weight_of(&r, 2, 1) - 0.75).abs() < 1e-3, "{:?}", r.weights[0][2]);
    for v in 0..3 {
        let s: f32 = r.weights[0][v].iter().sum();
        assert!((s - 1.0).abs() < 1e-4, "vertex {v} sums to {s}");
        assert!(r.weights[0][v].iter().filter(|&&x| x > 0.0).count() <= 4);
    }
    // The distance is the 0.05 gap in front of the strip.
    assert!(r.dist[0].iter().all(|&d| (d - 0.05).abs() < 1e-3), "{:?}", r.dist);
}

#[test]
fn disallowed_bones_hand_their_weight_to_the_parent() {
    let refm = reference();
    let skel = Skeleton::from_model(&refm).unwrap();
    let newm = new_with(&rig());
    // Only the hips may take weight: the arm share must land on the hips.
    let o = TransferOptions { rings: 0, allowed: Some(vec![true, false, false]), ..TransferOptions::default() };
    let r = weights::transfer(&newm, &refm, &skel, &o).unwrap();
    for v in 0..3 {
        assert!((weight_of(&r, v, 0) - 1.0).abs() < 1e-4, "vertex {v}: {:?} {:?}", r.joints[0][v], r.weights[0][v]);
        assert_eq!(weight_of(&r, v, 1), 0.0);
    }
}

#[test]
fn written_file_reads_back_with_the_same_skeleton_and_valid_weights() {
    let refm = reference();
    let skel = Skeleton::from_model(&refm).unwrap();
    let newm = new_with(&rig());
    let r = weights::transfer(&newm, &refm, &skel, &TransferOptions::default()).unwrap();
    let bytes = weights::write_weights(&newm, &r.joints, &r.weights).unwrap();
    let back = Model::from_bytes(&bytes).unwrap();
    assert_eq!(back.joint_names, refm.joint_names, "bone names and order are kept");
    assert!(weights::skeleton_diff(&Skeleton::from_model(&back).unwrap(), &skel, 1e-3).is_empty());
    let wc = wb_core::deform::check_weights(&back).unwrap();
    assert!(wc.bad.is_empty(), "{:?}", wc.bad);
}

#[test]
fn a_different_skeleton_is_refused_with_the_reason() {
    let refm = reference();
    let skel = Skeleton::from_model(&refm).unwrap();
    let diff = |bones: Vec<Bone>| weights::skeleton_diff(&Skeleton::from_model(&new_with(&bones)).unwrap(), &skel, 1e-3);

    // A renamed bone.
    let p = diff(vec![Bone("bone000_hips", None, [0.0, 0.0, 0.0]), Bone("bone001_R_arm", Some(0), [0.0, 1.0, 0.0]), Bone("bone002_tail", Some(0), [0.0, 0.0, -1.0])]);
    assert!(!p.is_empty() && p.iter().any(|m| m.contains("bone001")), "{p:?}");
    // Same names, swapped order.
    let p = diff(vec![Bone("bone000_hips", None, [0.0, 0.0, 0.0]), Bone("bone002_tail", Some(0), [0.0, 0.0, -1.0]), Bone("bone001_L_arm", Some(0), [0.0, 1.0, 0.0])]);
    assert!(p.iter().any(|m| m.contains("different order")), "{p:?}");
    // Arm moved: the rest pose differs.
    let p = diff(vec![Bone("bone000_hips", None, [0.0, 0.0, 0.0]), Bone("bone001_L_arm", Some(0), [0.0, 1.5, 0.0]), Bone("bone002_tail", Some(0), [0.0, 0.0, -1.0])]);
    assert!(p.iter().any(|m| m.contains("rest pose differs")), "{p:?}");
    // A joint missing.
    let p = diff(vec![Bone("bone000_hips", None, [0.0, 0.0, 0.0]), Bone("bone001_L_arm", Some(0), [0.0, 1.0, 0.0])]);
    assert!(p.iter().any(|m| m.contains("2 joints")), "{p:?}");
    // Re-parented.
    let p = diff(vec![Bone("bone000_hips", None, [0.0, 0.0, 0.0]), Bone("bone001_L_arm", Some(0), [0.0, 1.0, 0.0]), Bone("bone002_tail", Some(1), [0.0, -1.0, -1.0])]);
    assert!(p.iter().any(|m| m.contains("has parent")), "{p:?}");
}

#[test]
fn regions_follow_the_skeleton_not_the_triangles() {
    // The reference strip uses hips and arm; the new single triangle only the hips. Both must list the same
    // regions in the same order, or a reference value would be read from the wrong region.
    let (a, b) = (wb_core::deform::Prep::new(&reference(), false).unwrap(), wb_core::deform::Prep::new(&new_with(&rig()), false).unwrap());
    assert_eq!(a.regions, b.regions);
    assert_eq!(a.regions, vec!["L_arm".to_string(), "tail".to_string(), "torso".to_string()]);
    assert!(b.region_area[0] == 0.0 && b.region_area[2] > 0.0, "{:?}", b.region_area);
}

// ---------------------------------------------------------------------------------------------
// --adopt-skeleton: a bare mesh gets the reference's armature
// ---------------------------------------------------------------------------------------------

use wb_core::adopt;
use wb_core::glb::{Chunk, Glb, CHUNK_BIN, CHUNK_JSON};

/// The reference strip scaled by `s` and moved by `off`, with its skin.
fn ref_bytes(s: f32, off: [f32; 3]) -> Vec<u8> {
    let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0]].map(|p| [p[0] * s + off[0], p[1] * s + off[1], p[2] * s + off[2]]);
    let tris = [[0, 1, 3], [0, 3, 2], [2, 3, 5], [2, 5, 4]];
    let j = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]];
    let w = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
    build(&rig(), &pos, &tris, &j, &w)
}

/// What an outside modeler hands over: the mesh node alone, no skin, no joint attributes.
fn strip(bytes: &[u8], node_translation: Option<[f32; 3]>) -> Vec<u8> {
    let g = Glb::parse(bytes).unwrap();
    let mut j = g.json.clone();
    j.as_object_mut().unwrap().remove("skins");
    let n = j["nodes"].as_array().unwrap().iter().position(|n| n.get("mesh").is_some()).unwrap();
    let mut node = j["nodes"][n].clone();
    node.as_object_mut().unwrap().remove("skin");
    if let Some(t) = node_translation {
        node["translation"] = json!(t);
    }
    j["nodes"] = json!([node]);
    j["scenes"] = json!([{"nodes": [0]}]);
    let at = j["meshes"][0]["primitives"][0]["attributes"].as_object_mut().unwrap();
    at.remove("JOINTS_0");
    at.remove("WEIGHTS_0");
    let mut jb = serde_json::to_vec(&j).unwrap();
    while jb.len() % 4 != 0 {
        jb.push(b' ');
    }
    let chunks = vec![Chunk { kind: CHUNK_JSON, data: jb }, Chunk { kind: CHUNK_BIN, data: g.bin().to_vec() }];
    Glb { version: g.version, chunks, json: serde_json::Value::Null }.to_bytes()
}

#[test]
fn a_bare_mesh_adopts_the_reference_armature_and_gets_its_weights_back() {
    let refm = reference();
    let bare = Model::from_bytes(&strip(&ref_bytes(1.0, [0.0; 3]), None)).unwrap();
    assert!(bare.rig.is_none() && bare.joint_names.is_empty());
    let fit = adopt::fit_check(&bare, &refm, 0.1).unwrap();
    assert!(fit.problems.is_empty(), "{:?}", fit.problems);
    let adopted = Model::from_bytes(&adopt::adopt_skeleton(&bare, &refm).unwrap()).unwrap();
    let skel = Skeleton::from_model(&refm).unwrap();
    assert_eq!(adopted.joint_names, refm.joint_names);
    assert!(weights::skeleton_diff(&Skeleton::from_model(&adopted).unwrap(), &skel, 1e-4).is_empty());
    // The transfer then reproduces the reference's own weights on the identical surface.
    let o = TransferOptions { rings: 0, ..TransferOptions::default() };
    let r = weights::transfer(&adopted, &refm, &skel, &o).unwrap();
    for v in 0..6 {
        for jt in 0..3u16 {
            let want: f32 = (0..4).filter(|&k| refm.meshes[0].joints[v][k] == jt).map(|k| refm.meshes[0].weights[v][k]).sum();
            assert!((weight_of(&r, v, jt) - want).abs() < 1e-4, "vertex {v} joint {jt}");
        }
    }
    let back = Model::from_bytes(&weights::write_weights(&adopted, &r.joints, &r.weights).unwrap()).unwrap();
    assert!(wb_core::deform::check_weights(&back).unwrap().bad.is_empty());
    assert_eq!(back.joint_names, refm.joint_names);
}

#[test]
fn a_mesh_of_the_wrong_size_or_place_is_refused_with_the_scale() {
    let refm = reference();
    let big = Model::from_bytes(&strip(&ref_bytes(2.0, [0.0; 3]), None)).unwrap();
    let fit = adopt::fit_check(&big, &refm, 0.1).unwrap();
    assert!((fit.scale - 0.5).abs() < 1e-5);
    assert!(fit.problems.iter().any(|p| p.contains("scale the mesh by 0.5000")), "{:?}", fit.problems);
    // A centimeters-for-meters mistake names the factor and the likely cause, and nothing else.
    let cm = Model::from_bytes(&strip(&ref_bytes(100.0, [0.0; 3]), None)).unwrap();
    let fit = adopt::fit_check(&cm, &refm, 0.1).unwrap();
    assert_eq!(fit.problems.len(), 1, "{:?}", fit.problems);
    assert!(fit.problems[0].contains("centimeters"), "{:?}", fit.problems);
    // Right size, standing a quarter of its height off the reference.
    let moved = Model::from_bytes(&strip(&ref_bytes(1.0, [0.0, 0.5, 0.0]), None)).unwrap();
    let fit = adopt::fit_check(&moved, &refm, 0.1).unwrap();
    assert!(fit.problems.iter().any(|p| p.contains("bounding box center")), "{:?}", fit.problems);
    // The same offset passes when the tolerance allows it.
    assert!(adopt::fit_check(&moved, &refm, 0.3).unwrap().problems.is_empty());
}

#[test]
fn adopting_refuses_a_file_with_a_skin_and_reports_moved_mesh_nodes() {
    let refm = reference();
    let err = adopt::adopt_skeleton(&refm, &refm).err().unwrap();
    assert!(err.contains("already has a skin"), "{err}");
    let moved = Glb::parse(&strip(&ref_bytes(1.0, [0.0; 3]), Some([0.0, 1.0, 0.0]))).unwrap();
    assert_eq!(adopt::moved_mesh_chain(&moved), vec!["mesh0".to_string()]);
    let still = Glb::parse(&strip(&ref_bytes(1.0, [0.0; 3]), None)).unwrap();
    assert!(adopt::moved_mesh_chain(&still).is_empty());
}

#[test]
fn bone_space_matching_keeps_the_ramp_on_a_body_that_is_off_the_surface() {
    // The new triangle sits 0.3 in front of the strip instead of 0.05: the closest-point answer is the
    // same along the bone, and the bone-space answer must agree with it and stay valid.
    let refm = reference();
    let pos = [[0.2, 0.5, 0.3], [0.8, 0.5, 0.3], [0.5, 1.5, 0.3]];
    let (j, w) = ([[0u8, 0, 0, 0]; 3], [[1.0f32, 0.0, 0.0, 0.0]; 3]);
    let newm = Model::from_bytes(&build(&rig(), &pos, &[[0, 1, 2]], &j, &w)).unwrap();
    let skel = Skeleton::from_model(&refm).unwrap();
    let base = TransferOptions { rings: 0, ..TransferOptions::default() };
    let plain = weights::transfer(&newm, &refm, &skel, &base).unwrap();
    let bone = weights::transfer(&newm, &refm, &skel, &TransferOptions { radial: 0.25, soften: 0.0, ..base }).unwrap();
    for v in 0..3 {
        let s: f32 = bone.weights[0][v].iter().sum();
        assert!((s - 1.0).abs() < 1e-4, "vertex {v} sums to {s}");
        assert!(bone.weights[0][v].iter().filter(|&&x| x > 0.0).count() <= 4);
    }
    // Arm share rises along the bone (y 0.5 < y 1.5) in both, and the distance is the same 0.3 gap.
    assert!(weight_of(&bone, 2, 1) > weight_of(&bone, 0, 1) + 0.2, "{:?} {:?}", bone.weights[0][0], bone.weights[0][2]);
    assert!(weight_of(&plain, 2, 1) > weight_of(&plain, 0, 1) + 0.2);
    assert!(bone.dist[0].iter().all(|&d| (d - 0.3).abs() < 1e-3), "{:?}", bone.dist);
}

#[test]
fn repair_says_so_when_there_is_nothing_to_pose_the_model_in() {
    // Neither the model nor the reference has a clip, so there are no frames to check.
    let mut m = new_with(&rig());
    let refm = reference();
    let err = wb_core::repair::repair(&mut m, Some(&refm), &wb_core::repair::RepairOptions::default()).err().expect("no clips, no repair");
    assert!(err.contains("no clips"), "{err}");
}
