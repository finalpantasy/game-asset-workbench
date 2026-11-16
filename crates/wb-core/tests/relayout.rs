//! The Vivi 4K re-layout (experiments/vivi-uv-4k-v1) against the stock export.
use std::path::Path;
use wb_core::{relayout, Model};

const STOCK: &str = "../../../exports/Vivi-idle.glb";
const NEW: &str = "../../../experiments/vivi-uv-4k-v1/Vivi-uv4k-v1.glb";

fn files() -> Option<(Model, Model)> {
    if !Path::new(STOCK).exists() || !Path::new(NEW).exists() {
        eprintln!("skipped: Vivi files missing");
        return None;
    }
    Some((Model::open(Path::new(STOCK)).unwrap(), Model::open(Path::new(NEW)).unwrap()))
}

#[test]
fn vivi_relayout_is_unique_and_looks_the_same() {
    let Some((a, b)) = files() else { return };
    let c = relayout::check(&a, &b, &[0.0, 140.0, 270.0], 400);
    assert!(c.same_skeleton && c.same_clips && c.same_triangles());
    for m in &c.meshes {
        assert!(m.shared_texels[0] > 50_000, "{}: stock should share many texels, got {}", m.image, m.shared_texels[0]);
        assert!(m.shared_texels[1] <= relayout::MAX_SHARED_AFTER, "{}: {} shared texels after", m.image, m.shared_texels[1]);
        assert_eq!(m.texture_px[1], [4096, 4096]);
        assert!(m.density_median[1] > 4.0 * m.density_median[0], "{}: density {:?}", m.image, m.density_median);
    }
    for v in &c.views {
        assert!(v.mean_diff < 4.0 && v.over_24 * 100 < v.model_px, "view {}: mean {} over24 {} of {}", v.yaw_deg, v.mean_diff, v.over_24, v.model_px);
    }
    assert!(c.passed());
}

#[test]
fn a_file_against_itself_passes_but_is_not_unique_when_stock_shares() {
    let Some((a, _)) = files() else { return };
    let c = relayout::check(&a, &a, &[0.0], 200);
    assert!(c.looks_same() && c.same_triangles());
    assert!(c.views[0].mean_diff == 0.0);
    assert!(!c.unique(), "stock Vivi shares texels, so it must not count as a unique layout");
}
