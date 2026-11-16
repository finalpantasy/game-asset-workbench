//! Acceptance numbers from the workbench plan (milestone 1), checked on the stock Zidane export.
//! Skipped when the export is not on this PC.
use std::collections::BTreeMap;
use std::path::Path;
use wb_core::analysis;
use wb_core::Model;

const STOCK: &str = r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding\exports\Zidane-idle.glb";

fn stock() -> Option<(Vec<u8>, Model)> {
    let p = Path::new(STOCK);
    if !p.exists() {
        eprintln!("stock export not found, skipping");
        return None;
    }
    let bytes = std::fs::read(p).unwrap();
    let m = Model::from_bytes(&bytes).unwrap();
    Some((bytes, m))
}

#[test]
fn counts_match_the_plan() {
    let Some((_, m)) = stock() else { return };
    assert_eq!(m.triangle_count(), 3186);
    assert_eq!(m.accessors, 291);
    assert_eq!(m.joint_names.len(), 27);
    assert_eq!(m.clips, ["idle", "walk", "run", "turn_l", "turn_r"]);
}

#[test]
fn no_edit_export_is_byte_identical() {
    let Some((bytes, m)) = stock() else { return };
    assert_eq!(m.glb.to_bytes(), bytes);
}

#[test]
fn dead_triangles_by_region() {
    let Some((_, m)) = stock() else { return };
    let mut by: BTreeMap<String, usize> = BTreeMap::new();
    for mi in 0..m.meshes.len() {
        for ti in 0..m.meshes[mi].tris.len() {
            if analysis::is_dead(&m, mi, ti) {
                *by.entry(analysis::region(&m.joint_names[m.triangle_joint(mi, ti)])).or_default() += 1;
            }
        }
    }
    assert_eq!(by.values().sum::<usize>(), 116, "{by:?}");
    assert_eq!(by.get("hands"), Some(&88), "{by:?}");
    assert_eq!(by.get("shins"), Some(&20), "{by:?}");
    assert_eq!(by.get("feet"), Some(&8), "{by:?}");
}

#[test]
fn unused_texels_at_4096() {
    let Some((_, m)) = stock() else { return };
    let unused = |atlas: &str| {
        let ms: Vec<usize> = (0..m.meshes.len()).filter(|&i| m.atlas_name(i) == atlas).collect();
        4096 * 4096 - analysis::covered_texels(&m, &ms, 4096)
    };
    assert_eq!(unused("98_1"), 5_218_477);
    assert_eq!(unused("98_0"), 10_489_791);
}
