//! FBX export read-back: the binary FBX written by `wb_core::fbx` is parsed back (by our reader, which
//! follows Memoria's rules, and by ufbx, an independent reader) and compared with the GLB: vertex count,
//! positions, UVs, winding, weights summing to 1, bone names and order, parents, rest pose, bind poses.
use serde_json::json;
use std::path::{Path, PathBuf};
use wb_core::fbx::{self, ExportOptions};
use wb_core::Model;

const S: f32 = 0.01;

/// A skinned strip in GLB form, built the way the ff9mapkit exporter lays it out (Y mirrored, scale 0.01).
/// `bones`: (name, parent index, translation, rotation xyzw) in glTF space. Vertex k uses `joints[k]`/`weights[k]`.
struct Spec {
    bones: Vec<(&'static str, Option<usize>, [f32; 3], [f32; 4])>,
    pos: Vec<[f32; 3]>,
    uv: Vec<[f32; 2]>,
    tris: Vec<[u16; 3]>,
    joints: Vec<[u8; 4]>,
    weights: Vec<[f32; 4]>,
}

fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let ([ax, ay, az, aw], [bx, by, bz, bw]) = (a, b);
    [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]
}

fn rot_apply(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let qi = [-q[0], -q[1], -q[2], q[3]];
    let r = qmul(qmul(q, [v[0], v[1], v[2], 0.0]), qi);
    [r[0], r[1], r[2]]
}

type M = [[f64; 4]; 4];

fn mat(t: [f32; 3], q: [f32; 4]) -> M {
    let q = q.map(|v| v as f64);
    let mut m = [[0.0; 4]; 4];
    for c in 0..3 {
        let mut e = [0.0; 3];
        e[c] = 1.0;
        let col = rot_apply(q, e);
        for r in 0..3 {
            m[r][c] = col[r];
        }
    }
    for r in 0..3 {
        m[r][3] = t[r] as f64;
    }
    m[3][3] = 1.0;
    m
}

fn mul(a: &M, b: &M) -> M {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..4).map(|k| a[i][k] * b[k][j]).sum()))
}

fn inv_rigid(m: &M) -> M {
    let mut o = [[0.0; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = m[j][i];
        }
        o[i][3] = -(0..3).map(|k| o[i][k] * m[k][3]).sum::<f64>();
    }
    o[3][3] = 1.0;
    o
}

fn pad4(v: &mut Vec<u8>) {
    while v.len() % 4 != 0 {
        v.push(0);
    }
}

fn build_glb(spec: &Spec, extras: bool) -> Vec<u8> {
    let mut bin: Vec<u8> = Vec::new();
    let mut views = Vec::new();
    let mut accessors = Vec::new();
    let mut add = |bin: &mut Vec<u8>, data: &[u8], comp: u32, count: usize, ty: &str, extra: Option<(Vec<f64>, Vec<f64>)>| {
        pad4(bin);
        let off = bin.len();
        bin.extend_from_slice(data);
        views.push(json!({"buffer": 0, "byteOffset": off, "byteLength": data.len()}));
        let mut a = json!({"bufferView": views.len() - 1, "componentType": comp, "count": count, "type": ty});
        if let Some((mn, mx)) = extra {
            a["min"] = json!(mn);
            a["max"] = json!(mx);
        }
        accessors.push(a);
        accessors.len() - 1
    };
    let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let pos: Vec<f32> = spec.pos.iter().flatten().copied().collect();
    let a_pos = add(&mut bin, &f32s(&pos), 5126, spec.pos.len(), "VEC3", Some((vec![-9.0; 3], vec![9.0; 3])));
    let uv: Vec<f32> = spec.uv.iter().flatten().copied().collect();
    let a_uv = add(&mut bin, &f32s(&uv), 5126, spec.uv.len(), "VEC2", None);
    let jn: Vec<u8> = spec.joints.iter().flatten().copied().collect();
    let a_j = add(&mut bin, &jn, 5121, spec.joints.len(), "VEC4", None);
    let wt: Vec<f32> = spec.weights.iter().flatten().copied().collect();
    let a_w = add(&mut bin, &f32s(&wt), 5126, spec.weights.len(), "VEC4", None);
    let idx: Vec<u8> = spec.tris.iter().flatten().flat_map(|i| i.to_le_bytes()).collect();
    let a_i = add(&mut bin, &idx, 5123, spec.tris.len() * 3, "SCALAR", None);
    // inverse bind = inverse(world at rest), column-major
    let mut world: Vec<M> = Vec::new();
    for (_, p, t, q) in &spec.bones {
        let l = mat(*t, *q);
        world.push(match p {
            Some(p) => mul(&world[*p], &l),
            None => l,
        });
    }
    let mut ibm: Vec<f32> = Vec::new();
    for w in &world {
        let i = inv_rigid(w);
        for c in 0..4 {
            for r in 0..4 {
                ibm.push(i[r][c] as f32);
            }
        }
    }
    let a_ibm = add(&mut bin, &f32s(&ibm), 5126, spec.bones.len(), "MAT4", None);
    // texture
    let mut png = Vec::new();
    let img = image::RgbaImage::from_fn(4, 4, |x, y| image::Rgba([x as u8 * 60, y as u8 * 60, 128, 255]));
    image::DynamicImage::ImageRgba8(img).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
    pad4(&mut bin);
    let off = bin.len();
    bin.extend_from_slice(&png);
    views.push(json!({"buffer": 0, "byteOffset": off, "byteLength": png.len()}));
    let png_view = views.len() - 1;
    pad4(&mut bin);

    let mut nodes = Vec::new();
    for (i, (name, _, t, q)) in spec.bones.iter().enumerate() {
        let kids: Vec<usize> = spec.bones.iter().enumerate().filter(|(_, b)| b.1 == Some(i)).map(|(k, _)| k).collect();
        let mut n = json!({"name": name, "translation": t, "rotation": q});
        if name.starts_with("bone") {
            n["extras"] = json!({"ff9_bone_num": i});
        }
        if !kids.is_empty() {
            n["children"] = json!(kids);
        }
        nodes.push(n);
    }
    nodes.push(json!({"name": "mesh0", "mesh": 0, "skin": 0}));
    let mut doc = json!({
        "asset": {"version": "2.0", "generator": "test"},
        "scene": 0,
        "scenes": [{"nodes": [0, spec.bones.len()]}],
        "nodes": nodes,
        "skins": [{"joints": (0..spec.bones.len()).collect::<Vec<_>>(), "inverseBindMatrices": a_ibm, "skeleton": 0}],
        "meshes": [{"name": "mesh0", "primitives": [{"attributes": {"POSITION": a_pos, "TEXCOORD_0": a_uv, "JOINTS_0": a_j, "WEIGHTS_0": a_w}, "indices": a_i, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "textures": [{"source": 0}],
        "images": [{"name": "7_0", "mimeType": "image/png", "bufferView": png_view}],
        "bufferViews": views,
        "accessors": accessors,
        "buffers": [{"byteLength": bin.len()}],
    });
    if extras {
        doc["asset"]["extras"] = json!({"ff9_geo": "GEO_MAIN_F0_TST", "ff9_geo_id": 7, "ff9_scale": S});
    }
    let mut js = serde_json::to_vec(&doc).unwrap();
    while js.len() % 4 != 0 {
        js.push(b' ');
    }
    let total = 12 + 8 + js.len() + 8 + bin.len();
    let mut out = Vec::new();
    out.extend_from_slice(b"glTF");
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

fn quad(bones: Vec<(&'static str, Option<usize>, [f32; 3], [f32; 4])>, joints: Vec<[u8; 4]>, weights: Vec<[f32; 4]>) -> Spec {
    Spec {
        bones,
        // glTF space: Y up, mirrored from Unity's Y down
        pos: vec![[0.10, 0.20, 0.0], [0.30, 0.20, 0.0], [0.30, 0.40, 0.05], [0.10, 0.40, 0.05]],
        uv: vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.75], [0.0, 0.75]],
        tris: vec![[0, 1, 2], [0, 2, 3]],
        joints,
        weights,
    }
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("gaw-fbx-test-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol})");
}

/// Everything we can say about an export without a second reader.
fn check_with_own_reader(model: &Model, ex: &fbx::Export) -> fbx::Scene {
    let scene = fbx::read_scene(&ex.fbx).expect("our reader parses the file the way the importer does");
    assert_eq!(scene.bones.len(), ex.bones);
    // names/order: bone000.. in order, Root then LimbNodes, parents as in the GLB
    let glb_parent: Vec<Option<usize>> = {
        let nodes = model.glb.list("nodes");
        let joints: Vec<usize> = model.glb.list("skins")[0]["joints"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        joints
            .iter()
            .map(|&j| nodes.iter().position(|n| n["children"].as_array().is_some_and(|c| c.iter().any(|x| x.as_u64() == Some(j as u64)))).and_then(|p| joints.iter().position(|&jj| jj == p)))
            .collect()
    };
    for (i, b) in scene.bones.iter().enumerate() {
        assert_eq!(b.num, i);
        assert_eq!(b.class, if i == 0 { "Root" } else { "LimbNode" });
        assert_eq!(b.parent, glb_parent[i], "parent of bone{i:03}");
    }
    scene
}

fn compare_meshes(model: &Model, ex: &fbx::Export, scene: &fbx::Scene) {
    let s = ex.scale;
    let kept: Vec<&wb_core::model::Mesh> = fbx::export_order(&model.meshes.iter().map(|m| m.name.as_str()).collect::<Vec<_>>()).into_iter().map(|i| &model.meshes[i]).filter(|m| !ex.skipped_meshes.contains(&m.name)).collect();
    assert_eq!(scene.meshes.len(), kept.len());
    for (sm, gm) in scene.meshes.iter().zip(&kept) {
        assert_eq!(sm.name, gm.name.replace(['#', ':'], "_"));
        assert_eq!(sm.verts.len(), gm.pos.len(), "{} vertex count", gm.name);
        assert_eq!(sm.uvs.len(), gm.uv.len());
        assert_eq!(sm.tris.len(), gm.tris.len());
        for (i, (v, g)) in sm.verts.iter().zip(&gm.pos).enumerate() {
            // Unity = (x, -y, z) / scale
            close(v[0], g[0] as f64 / s, 2e-3, &format!("{} v{i}.x", gm.name));
            close(v[1], -(g[1] as f64) / s, 2e-3, &format!("{} v{i}.y", gm.name));
            close(v[2], g[2] as f64 / s, 2e-3, &format!("{} v{i}.z", gm.name));
            close(sm.uvs[i][0], gm.uv[i][0] as f64, 1e-6, "u");
            close(sm.uvs[i][1], 1.0 - gm.uv[i][1] as f64, 1e-6, "v");
        }
        // winding reversed: (a, b, c) -> (a, c, b)
        for (st, gt) in sm.tris.iter().zip(&gm.tris) {
            assert_eq!(*st, [gt[0], gt[2], gt[1]]);
        }
        for (vi, w) in sm.weights.iter().enumerate() {
            let sum: f64 = w.iter().map(|e| e.1).sum();
            close(sum, 1.0, 1e-6, &format!("{} v{vi} weights sum", gm.name));
            assert!(w.len() <= 4);
            // same influence set and values as the GLB
            for c in 0..4 {
                let (j, gw) = (gm.joints[vi][c] as usize, gm.weights[vi][c] as f64);
                if gw > 1e-6 {
                    let total: f64 = (0..4).map(|k| gm.weights[vi][k] as f64).filter(|&x| x > 1e-6).sum();
                    let got = w.iter().filter(|e| e.0 == j).map(|e| e.1).sum::<f64>();
                    close(got, gw / total, 1e-6, &format!("{} v{vi} weight of joint {j}", gm.name));
                }
            }
        }
        assert_eq!(sm.shading.as_deref(), Some("Phong"));
        assert_eq!(sm.texture.as_deref(), gm.image.map(|i| format!("{}.png", model.images[i].name.trim_end_matches(".png"))).as_deref());
    }
}

/// Bind poses: world(rest) from the FBX bones, mapped to glTF space, times the GLB's inverse bind = identity.
fn compare_bind_poses(model: &Model, ex: &fbx::Export, scene: &fbx::Scene) {
    let acc = model.glb.accessor(model.glb.list("skins")[0]["inverseBindMatrices"].as_u64().unwrap() as usize).unwrap();
    let world = fbx::scene_world(scene);
    let s = ex.scale;
    // C = mirror Y and scale: glTF = C * Unity * C^-1
    let c: [[f64; 4]; 4] = [[s, 0.0, 0.0, 0.0], [0.0, -s, 0.0, 0.0], [0.0, 0.0, s, 0.0], [0.0, 0.0, 0.0, 1.0]];
    let ci = fbx::invert(&c);
    let mut worst = 0f64;
    for (i, w) in world.iter().enumerate() {
        let g = fbx::multiply(&fbx::multiply(&c, w), &ci);
        let ibm: [[f64; 4]; 4] = std::array::from_fn(|r| std::array::from_fn(|k| acc.get(i, k * 4 + r)));
        let p = fbx::multiply(&g, &ibm);
        for r in 0..4 {
            for k in 0..4 {
                worst = worst.max((p[r][k] - if r == k { 1.0 } else { 0.0 }).abs());
            }
        }
        // the cluster's TransformLink (bind world) equals the composed rest world
        if let Some(bw) = scene.bones[i].bind_world {
            for r in 0..4 {
                for k in 0..4 {
                    close(bw[r][k], w[r][k], 1e-4 * (1.0 + w[r][k].abs()), &format!("bone{i:03} TransformLink"));
                }
            }
        }
    }
    assert!(worst < 2e-3, "rest pose x inverse bind differs from identity by {worst}");
}

/// ufbx as the independent reader: counts, names, positions, UVs, weights, bind matrices.
fn check_with_ufbx(path: &Path, model: &Model, ex: &fbx::Export) {
    let root = ufbx::load_file(path.to_str().unwrap(), ufbx::LoadOpts::default()).unwrap_or_else(|e| panic!("ufbx cannot read {}: {e:?}", path.display()));
    let scene: &ufbx::Scene = &root;
    let kept: Vec<&wb_core::model::Mesh> = fbx::export_order(&model.meshes.iter().map(|m| m.name.as_str()).collect::<Vec<_>>()).into_iter().map(|i| &model.meshes[i]).filter(|m| !ex.skipped_meshes.contains(&m.name)).collect();
    assert_eq!(scene.meshes.len(), kept.len(), "mesh count");
    // bone nodes by name, in file order: bone000.. then no extras
    let mut bone_names: Vec<String> = scene.nodes.iter().filter(|n| n.element.name.starts_with("bone")).map(|n| n.element.name.to_string()).collect();
    bone_names.sort();
    assert_eq!(bone_names.len(), ex.bones);
    for (i, n) in bone_names.iter().enumerate() {
        assert_eq!(*n, format!("bone{i:03}"), "bone names");
    }
    // file order: the skeleton's Model nodes appear as bone000, bone001, ... (the original FF9 order)
    let top = fbx::parse_binary(&ex.fbx).unwrap();
    let objects = top.iter().find(|n| n.name == "Objects").unwrap();
    let in_file: Vec<String> = objects
        .children("Model")
        .filter_map(|m| match (&m.props[1], &m.props[2]) {
            (fbx::Prop::Str(n), fbx::Prop::Str(c)) if c == "Root" || c == "LimbNode" => Some(n.split('\0').next().unwrap().to_owned()),
            _ => None,
        })
        .collect();
    assert_eq!(in_file, (0..ex.bones).map(|i| format!("bone{i:03}")).collect::<Vec<_>>());
    let s = ex.scale;
    for gm in kept {
        let mesh = scene.meshes.iter().find(|m| m.element.name == gm.name.replace(['#', ':'], "_").as_str()).unwrap_or_else(|| panic!("ufbx has no mesh {}", gm.name));
        assert_eq!(mesh.num_vertices, gm.pos.len(), "{} vertex count in ufbx", gm.name);
        assert_eq!(mesh.num_triangles, gm.tris.len());
        assert!(mesh.vertex_uv.exists);
        // per corner: position, uv
        for ix in 0..mesh.num_indices {
            let v = mesh.vertex_indices[ix] as usize;
            let p = mesh.vertex_position.values[mesh.vertex_position.indices[ix] as usize];
            close(p.x, gm.pos[v][0] as f64 / s, 2e-3, "ufbx x");
            close(p.y, -(gm.pos[v][1] as f64) / s, 2e-3, "ufbx y");
            close(p.z, gm.pos[v][2] as f64 / s, 2e-3, "ufbx z");
            let uv = mesh.vertex_uv.values[mesh.vertex_uv.indices[ix] as usize];
            close(uv.x, gm.uv[v][0] as f64, 1e-6, "ufbx u");
            close(uv.y, 1.0 - gm.uv[v][1] as f64, 1e-6, "ufbx v");
        }
        // triangles: ufbx triangulates faces in order; corner triples must equal (a, c, b)
        for (ti, gt) in gm.tris.iter().enumerate() {
            let face = mesh.faces[ti];
            let ix = face.index_begin as usize;
            let tri = [mesh.vertex_indices[ix], mesh.vertex_indices[ix + 1], mesh.vertex_indices[ix + 2]];
            assert_eq!(tri, [gt[0], gt[2], gt[1]], "ufbx triangle {ti}");
        }
        // skin: weights sum to 1, same influences as the GLB
        let skin = &mesh.skin_deformers[0];
        let mut sums = vec![0f64; gm.pos.len()];
        for cl in skin.clusters.iter() {
            let bone = cl.bone_node.as_ref().expect("cluster has a bone").element.name.to_string();
            assert!(bone.starts_with("bone"));
            let num: usize = bone[4..].parse().unwrap();
            for (vi, w) in cl.vertices.iter().zip(cl.weights.iter()) {
                sums[*vi as usize] += *w as f64;
                let mut want = 0.0;
                let total: f64 = (0..4).map(|k| gm.weights[*vi as usize][k] as f64).filter(|&x| x > 1e-6).sum();
                for k in 0..4 {
                    if gm.joints[*vi as usize][k] as usize == num && gm.weights[*vi as usize][k] > 1e-6 {
                        want += gm.weights[*vi as usize][k] as f64 / total;
                    }
                }
                close(*w as f64, want, 1e-6, &format!("ufbx weight v{vi} bone{num}"));
            }
        }
        for (i, s) in sums.iter().enumerate() {
            close(*s, 1.0, 1e-6, &format!("ufbx weights of v{i}"));
        }
    }
    // bind poses: ufbx's own geometry_to_bone must invert our TransformLink; compare with the GLB inverse bind (mapped to Unity)
    let acc = model.glb.accessor(model.glb.list("skins")[0]["inverseBindMatrices"].as_u64().unwrap() as usize).unwrap();
    let c: [[f64; 4]; 4] = [[s, 0.0, 0.0, 0.0], [0.0, -s, 0.0, 0.0], [0.0, 0.0, s, 0.0], [0.0, 0.0, 0.0, 1.0]];
    let ci = fbx::invert(&c);
    let mut worst = 0f64;
    for mesh in scene.meshes.iter() {
        for cl in mesh.skin_deformers[0].clusters.iter() {
            let num: usize = cl.bone_node.as_ref().unwrap().element.name[4..].parse().unwrap();
            let g = &cl.geometry_to_bone; // Unity geometry -> bone
            let gm: [[f64; 4]; 4] = [[g.m00, g.m01, g.m02, g.m03], [g.m10, g.m11, g.m12, g.m13], [g.m20, g.m21, g.m22, g.m23], [0.0, 0.0, 0.0, 1.0]];
            // glTF inverse bind maps glTF space -> bone space (glTF); in Unity terms IBM_unity = C^-1 * IBM * C
            let ibm: [[f64; 4]; 4] = std::array::from_fn(|r| std::array::from_fn(|k| acc.get(num, k * 4 + r)));
            let want = fbx::multiply(&fbx::multiply(&ci, &ibm), &c);
            if std::env::var("FBX_DEBUG").is_ok() {
                eprintln!("bone{num} ufbx {gm:?}\n   want {want:?}");
            }
            for r in 0..3 {
                for k in 0..4 {
                    worst = worst.max((gm[r][k] - want[r][k]).abs() / (1.0 + want[r][k].abs()));
                }
            }
        }
    }
    assert!(worst < 2e-3, "ufbx geometry-to-bone differs from the GLB inverse bind by {worst} (relative)");
}

fn export_and_check(glb_bytes: &[u8], name: &str, opts: &ExportOptions) -> fbx::Export {
    let model = Model::from_bytes(glb_bytes).expect("GLB parses");
    let ex = fbx::export_fbx(&model, opts).expect("export");
    let path = tmp(&format!("{name}.fbx"));
    std::fs::write(&path, &ex.fbx).unwrap();
    let scene = check_with_own_reader(&model, &ex);
    compare_meshes(&model, &ex, &scene);
    compare_bind_poses(&model, &ex, &scene);
    check_with_ufbx(&path, &model, &ex);
    assert!(ex.euler_error < 1e-6 && ex.bind_error < 1e-3, "euler {:.2e}, bind {:.2e}", ex.euler_error, ex.bind_error);
    ex
}

#[test]
fn one_bone_quad() {
    // One root bone, 180 degrees about X like FF9's bone000, a quad skinned to it.
    let spec = quad(vec![("bone000", None, [0.0, 2.4, 0.0], [-1.0, 0.0, 0.0, 0.0007669902988709509])], vec![[0, 0, 0, 0]; 4], vec![[1.0, 0.0, 0.0, 0.0]; 4]);
    let ex = export_and_check(&build_glb(&spec, true), "onebone", &ExportOptions::default());
    assert_eq!(ex.geo_id, Some(7));
    assert_eq!(ex.type_int, Some(2));
    assert_eq!(ex.bones, 1);
    assert_eq!(ex.textures.len(), 1);
    assert_eq!(ex.textures[0].0, "7_0.png");
    // the file starts with the binary FBX magic and version 7400
    assert_eq!(&ex.fbx[..21], b"Kaydara FBX Binary  \0");
    assert_eq!(u32::from_le_bytes(ex.fbx[23..27].try_into().unwrap()), 7400);
}

#[test]
fn two_bones_blended_weights() {
    // Root with a child, vertices blended 50/50 and 70/30 so the clusters and parents are exercised.
    let spec = quad(
        vec![("bone000_root", None, [0.0, 2.4, 0.0], [-1.0, 0.0, 0.0, 0.0007669902988709509]), ("bone001_spine", Some(0), [0.0, 0.0, 0.85], [0.7481193542480469, 0.0, 0.0, 0.6635642051696777])],
        vec![[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
        vec![[0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0]],
    );
    let ex = export_and_check(&build_glb(&spec, true), "twobone", &ExportOptions::default());
    assert_eq!(ex.bones, 2);
    assert_eq!(ex.meshes[0].max_influences, 2);
}

#[test]
fn rejects_a_bad_skeleton() {
    // Not boneNNN
    let spec = quad(vec![("hips", None, [0.0; 3], [0.0, 0.0, 0.0, 1.0])], vec![[0; 4]; 4], vec![[1.0, 0.0, 0.0, 0.0]; 4]);
    let m = Model::from_bytes(&build_glb(&spec, true)).unwrap();
    assert!(fbx::export_fbx(&m, &ExportOptions::default()).unwrap_err().contains("boneNNN"));
    // Root must be bone000
    let spec = quad(vec![("bone001", None, [0.0; 3], [0.0, 0.0, 0.0, 1.0])], vec![[0; 4]; 4], vec![[1.0, 0.0, 0.0, 0.0]; 4]);
    let m = Model::from_bytes(&build_glb(&spec, true)).unwrap();
    assert!(fbx::export_fbx(&m, &ExportOptions::default()).is_err());
}

#[test]
fn writer_and_reader_agree_on_nodes() {
    let nodes = vec![fbx::Node::new("A"), fbx::Node::new("B")];
    let bytes = fbx::write_binary(&nodes);
    let back = fbx::parse_binary(&bytes).unwrap();
    assert_eq!(back, nodes);
    assert_eq!((bytes.len() - 140) % 16, 0, "the footer version field sits on a 16-byte boundary");
}

fn exp(rel: &str) -> PathBuf {
    PathBuf::from(r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding").join(rel)
}

/// Real files from the workbench; skipped when a file is not on this machine.
fn real_file(rel: &str, name: &str) {
    let p = exp(rel);
    let Ok(bytes) = std::fs::read(&p) else {
        eprintln!("skip {name}: {} not found", p.display());
        return;
    };
    let ex = export_and_check(&bytes, name, &ExportOptions::default());
    assert_eq!(ex.bones, 27);
    assert!(ex.meshes.iter().all(|m| m.max_influences <= 4));
    eprintln!("{name}: {} bones, meshes {:?}, skipped {:?}, euler {:.2e}, bind {:.2e}", ex.bones, ex.meshes.iter().map(|m| (m.name.clone(), m.vertices, m.triangles)).collect::<Vec<_>>(), ex.skipped_meshes, ex.euler_error, ex.bind_error);
}

#[test]
fn zidane_overlays_in_stock_order() {
    // Memoria wires the blink to a material by renderer index; stock Zidane's renderers are
    // battle_model0, battle_model1, mesh0, mesh1, so the overlays must come first in the FBX.
    let p = exp("exports/Zidane-idle-overlays.glb");
    let Ok(bytes) = std::fs::read(&p) else {
        eprintln!("skip: {} not found", p.display());
        return;
    };
    let ex = export_and_check(&bytes, "zidane-overlays", &ExportOptions { keep_overlays: true, ..Default::default() });
    let names: Vec<&str> = ex.meshes.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["battle_model0", "battle_model1", "mesh0", "mesh1"]);
    let scene = fbx::read_scene(&ex.fbx).unwrap();
    let in_file: Vec<&str> = scene.meshes.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(in_file, names);
}

#[test]
fn export_order_puts_overlays_first() {
    assert_eq!(fbx::export_order(&["mesh0", "mesh1", "battle_model1", "battle_model0"]), vec![3, 2, 0, 1]);
    assert_eq!(fbx::export_order(&["a", "b"]), vec![0, 1]);
}

#[test]
fn zidane_stock_mesh() {
    real_file("exports/Zidane-idle.glb", "zidane-stock");
}

#[test]
fn zidane_belt_edit() {
    real_file("experiments/zidane-belt-strap-v3/boundary/combined-2-3-4/Zidane-belt-strap-v3-v13-boundary-leftline.glb", "zidane-belt");
}

#[test]
fn zidane_hair_stock_mesh() {
    real_file("experiments/zidane-hair-tool-v3/stock-mesh/Zidane-hair-stock-mesh.glb", "zidane-hair-stock");
}
