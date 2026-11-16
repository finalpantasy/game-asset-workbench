//! Compare checks: skinning and the diff, on files that exist on this PC (skipped otherwise).
use std::path::Path;
use wb_core::{diff, image_textures, Model};

const STOCK: &str = r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding\exports\Zidane-idle.glb";
const HD: &str = r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding\experiments\stock-zidane-hd-v1\models-v5";
const V29B: &str = r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding\experiments\stock-zidane-hd-v1\finish-v29\Zidane-v29b-garment-finish.glb";
const PARTS: &str = r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding\experiments\zidane-uv-kit-claude-v1\parts";

fn open(p: &str) -> Option<Model> {
    if !Path::new(p).exists() {
        eprintln!("{p} not found, skipping");
        return None;
    }
    Some(Model::open(Path::new(p)).unwrap())
}

#[test]
fn rest_pose_skinning_keeps_positions() {
    let Some(m) = open(STOCK) else { return };
    let posed = m.posed(None, 0.0);
    let mut worst = 0f32;
    for (mesh, p) in m.meshes.iter().zip(&posed) {
        assert_eq!(mesh.pos.len(), p.len());
        for (a, b) in mesh.pos.iter().zip(p) {
            for k in 0..3 {
                worst = worst.max((a[k] - b[k]).abs());
            }
        }
    }
    // The bind pose is the rest pose, so skinning with no clip must not move anything.
    assert!(worst < 1e-3, "rest pose moved a vertex by {worst}");
}

#[test]
fn clips_move_the_model() {
    let Some(m) = open(STOCK) else { return };
    let walk = m.clips.iter().position(|c| c == "walk").expect("walk clip");
    let d = m.clip_duration(walk);
    assert!(d > 0.0);
    let rest = m.posed(None, 0.0);
    let mid = m.posed(Some(walk), d * 0.5);
    let moved = rest.iter().zip(&mid).flat_map(|(a, b)| a.iter().zip(b)).filter(|(a, b)| (0..3).any(|k| (a[k] - b[k]).abs() > 1e-3)).count();
    assert!(moved > 1000, "walk at mid-clip moved only {moved} vertices");
}

#[test]
fn stock_against_itself_is_unchanged() {
    let Some(m) = open(STOCK) else { return };
    let n = open(STOCK).unwrap();
    let (mt, nt) = (image_textures(&m), image_textures(&n));
    let d = diff::diff(&m, &mt, &n, &nt, None, 0);
    assert!(d.identical_files);
    assert_eq!(d.changed_pixels(), 0);
    assert!(d.meshes.iter().all(|x| x.status() == "same"));
    assert!(diff::changed_triangles(&m, &d).iter().flatten().all(|c| !c));
}

#[test]
fn glove_uv_repair_changes_only_uvs() {
    let (Some(paint), Some(repair)) = (open(&format!(r"{HD}\Zidane-texture-only.glb")), open(&format!(r"{HD}\Zidane-gloves-thumb-uv-repair.glb"))) else { return };
    let (pt, rt) = (image_textures(&paint), image_textures(&repair));
    let d = diff::diff(&paint, &pt, &repair, &rt, None, 0);
    assert_eq!(d.changed_pixels(), 0, "the two glove files share their textures");
    let body = d.meshes.iter().find(|m| m.name == "mesh0").unwrap();
    assert_eq!((body.moved_vertices, body.uv_changed_vertices, body.triangles_same), (0, 242, true));
    assert_eq!(d.meshes.iter().find(|m| m.name == "mesh1").unwrap().status(), "same");
    // The repaired UVs light up as changed triangles on the body only.
    let ct = diff::changed_triangles(&repair, &d);
    let n: Vec<usize> = ct.iter().map(|m| m.iter().filter(|&&c| c).count()).collect();
    assert!(n[0] > 0 && n[1] == 0, "changed triangles per mesh: {n:?}");
}

/// Kit part names left untouched on the body texture (98_1) between two files.
fn untouched_on_body(a: &Model, b: &Model) -> Vec<String> {
    let parts = wb_core::analysis::Parts::load(Path::new(PARTS)).unwrap();
    let d = diff::diff(a, &image_textures(a), b, &image_textures(b), Some(&parts), 0);
    let side = |m: &Model| (diff::changed_triangles(m, &d), (0..m.meshes.len()).map(|i| parts.triangle_parts(m, i)).collect::<Vec<_>>());
    let ((ca, pa), (cb, pb)) = (side(a), side(b));
    let u = diff::untouched_parts(&d, &parts, [diff::Side { model: a, changed: &ca, tri_parts: &pa }, diff::Side { model: b, changed: &cb, tri_parts: &pb }]);
    let ti = d.textures.iter().position(|t| t.name == "98_1").unwrap();
    u[ti].iter().map(|&id| parts.name(id).to_owned()).collect()
}

#[test]
fn untouched_parts_follow_pixels_and_uvs() {
    let (Some(g5), Some(v29b)) = (open(&format!(r"{HD}\Zidane-gloves-thumb-uv-repair.glb")), open(V29B)) else { return };
    if !Path::new(PARTS).exists() {
        return;
    }
    // v29b was painted on top of the accepted gloves: the gloves are untouched, the pants are not.
    let u = untouched_on_body(&g5, &v29b);
    assert!(u.iter().any(|p| p == "gloves"), "untouched: {u:?}");
    assert!(!u.iter().any(|p| p == "pants"), "untouched: {u:?}");
}
