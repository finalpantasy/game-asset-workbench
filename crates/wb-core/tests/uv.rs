//! UV checks against the kit's published figures (NOTES.md, guides legend), on stock Zidane.
use std::path::Path;
use wb_core::analysis::Parts;
use wb_core::{uv, Model};

const STOCK: &str = r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding\exports\Zidane-idle.glb";
const PARTS: &str = r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding\experiments\zidane-uv-kit-claude-v1\parts";

fn stock() -> Option<Model> {
    Path::new(STOCK).exists().then(|| Model::open(Path::new(STOCK)).unwrap())
}

#[test]
fn islands_match_the_kit() {
    let Some(m) = stock() else { return };
    let t = uv::analyze(&m.meshes[0]);
    assert_eq!(t.islands, 28);
    assert_eq!(t.dup_of.iter().filter(|d| d.is_some()).count(), 11);
    assert_eq!(t.island_sizes()[7], 128, "waist island");
    if Path::new(PARTS).exists() {
        let p = Parts::load(Path::new(PARTS)).unwrap();
        for mi in 0..m.meshes.len() {
            let t = uv::analyze(&m.meshes[mi]);
            let l = p.island_labels(&m.atlas_name(mi), &t.island_sizes()).expect("labels match the stock islands");
            if mi == 0 {
                assert!(l[7].starts_with("waist"), "{}", l[7]);
            }
        }
    }
}

#[test]
fn waist_density_and_overlap_match_the_kit() {
    let Some(m) = stock() else { return };
    let mesh = &m.meshes[0];
    let t = uv::analyze(mesh);
    let d = uv::density(mesh, 4096);
    let w: Vec<f32> = (0..mesh.tris.len()).filter(|&i| t.island[i] == 7 && t.dup_of[i].is_none()).map(|i| d[i]).collect();
    let (lo, hi) = w.iter().fold((f32::MAX, 0f32), |(a, b), &x| (a.min(x), b.max(x)));
    assert_eq!((lo.round(), hi.round()), (746.0, 3287.0));
    let ov = uv::overlap(mesh, &t, 1024);
    let pairs = uv::island_pairs(mesh, &t, &ov);
    let waist = pairs.iter().find(|p| p.a == 7 && p.b == 7).expect("waist self-overlap");
    assert_eq!(waist.texels, 28_653);
    assert_eq!(waist.kind, uv::Kind::SelfOverlap);
    assert!((waist.max_dist - 1.13).abs() < 0.01, "{}", waist.max_dist);
    // The arms (islands 2 and 5) are a mirrored twin.
    // Mirror twins as the kit's PARTS.md lists them: arms, pants legs, boots, gloves.
    for (a, b) in [(2, 5), (1, 6), (8, 12), (10, 14), (23, 26)] {
        assert!(pairs.iter().any(|p| p.a == a && p.b == b && p.kind == uv::Kind::Mirrored), "{a}-{b} should be a mirrored twin");
    }
}

#[test]
fn touch_finds_both_gloves() {
    let (Some(m), true) = (stock(), Path::new(PARTS).exists()) else { return };
    let p = Parts::load(Path::new(PARTS)).unwrap();
    let g = p.id_by_name("gloves").unwrap();
    let pt = uv::part_texels(&m, 0, &p);
    let touched: Vec<usize> = (0..pt.len()).filter(|&t| pt[t].iter().any(|e| e.0 == g)).collect();
    let mesh = &m.meshes[0];
    let (l, r) = touched.iter().fold((0, 0), |(l, r), &t| if uv::center(mesh, t)[0] > 0.0 { (l, r + 1) } else { (l + 1, r) });
    assert!(l > 10 && r > 10, "glove triangles left {l}, right {r}");
}
