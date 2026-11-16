//! Checks for a UV re-layout: the same character on a new texture layout.
//! It must keep the skeleton, clips and triangles, stop sharing texels between surfaces,
//! and look the same as the original from every side.
use crate::render::{self, Camera};
use crate::{image_textures, uv, Model};

pub struct MeshCheck {
    pub name: String,
    pub image: String,
    pub triangles: [usize; 2],
    pub texture_px: [[usize; 2]; 2],
    /// Texels that show on two surfaces, at 1024 (strict interior, back faces left out).
    pub shared_texels: [usize; 2],
    /// Of the shared texels after, those where both triangles read a kept stock box (left in place
    /// on purpose, such as the eye-blink texels both eyes read). They do not fail the layout.
    pub kept_shared: usize,
    /// Median texels per model unit at each file's own texture size.
    pub density_median: [f32; 2],
}

pub struct ViewCheck {
    pub yaw_deg: f32,
    pub model_px: usize,
    /// Mean of the largest channel difference over model pixels, 0..255.
    pub mean_diff: f32,
    /// Model pixels where some channel differs by more than 24.
    pub over_24: usize,
}

pub struct Checks {
    pub same_skeleton: bool,
    pub same_clips: bool,
    pub meshes: Vec<MeshCheck>,
    pub views: Vec<ViewCheck>,
}

/// Limits a re-layout must meet: renders within 6 of 255 on average with under 1% of model
/// pixels off by more than 24, and at most 16 shared texels per mesh.
pub const MAX_MEAN_DIFF: f32 = 6.0;
pub const MAX_SHARED_AFTER: usize = 16;

impl MeshCheck {
    /// Shared texels after that were not kept on purpose.
    pub fn unplanned_shared(&self) -> usize {
        self.shared_texels[1] - self.kept_shared
    }
}

impl Checks {
    pub fn same_triangles(&self) -> bool {
        self.meshes.iter().all(|m| m.triangles[0] == m.triangles[1])
    }
    pub fn unique(&self) -> bool {
        self.meshes.iter().all(|m| m.unplanned_shared() <= MAX_SHARED_AFTER)
    }
    pub fn looks_same(&self) -> bool {
        self.views.iter().all(|v| v.mean_diff <= MAX_MEAN_DIFF && v.over_24 * 100 <= v.model_px)
    }
    pub fn passed(&self) -> bool {
        self.same_skeleton && self.same_clips && self.same_triangles() && self.unique() && self.looks_same()
    }
}

fn median(mut v: Vec<f32>) -> f32 {
    v.retain(|x| *x > 0.0 && x.is_finite());
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Shared texels, and how many of them lie between two triangles that both read one of `boxes`
/// (0..1 UV boxes kept at their stock place).
fn shared(m: &crate::model::Mesh, boxes: &[[f64; 4]]) -> (usize, usize) {
    let topo = uv::analyze(m);
    let kept = |t: u32| {
        m.tris[t as usize].iter().any(|&v| {
            let (x, y) = (m.uv[v as usize][0] as f64, m.uv[v as usize][1] as f64);
            boxes.iter().any(|b| x > b[0] && x < b[2] && y > b[1] && y < b[3])
        })
    };
    let pairs = uv::overlap(m, &topo, uv::OVERLAP_SIZE).pairs;
    let all = pairs.iter().map(|p| p.texels as usize).sum();
    let planned = pairs.iter().filter(|p| !boxes.is_empty() && kept(p.a) && kept(p.b)).map(|p| p.texels as usize).sum();
    (all, planned)
}

/// Compares `a` (original) with `b` (re-laid-out). `yaws` in degrees, `px` render size.
pub fn check(a: &Model, b: &Model, yaws: &[f32], px: usize) -> Checks {
    check_keep(a, b, yaws, px, &[])
}

/// `check`, where `keep` names stock boxes (sheet name, 0..1 UV box) the new layout left in place on
/// purpose: texels shared inside them are counted apart and do not fail the layout.
pub fn check_keep(a: &Model, b: &Model, yaws: &[f32], px: usize, keep: &[(String, [f64; 4])]) -> Checks {
    let (ta, tb) = (image_textures(a), image_textures(b));
    let size = |m: &Model, t: &[Option<crate::Texture>], mi: usize| m.meshes[mi].image.and_then(|i| t[i].as_ref()).map(|t| [t.w, t.h]).unwrap_or([0, 0]);
    let meshes = (0..a.meshes.len().min(b.meshes.len()))
        .map(|mi| {
            let (ma, mb) = (&a.meshes[mi], &b.meshes[mi]);
            let (sa, sb) = (size(a, &ta, mi), size(b, &tb, mi));
            let image = a.atlas_name(mi);
            let boxes: Vec<[f64; 4]> = keep.iter().filter(|k| k.0 == image).map(|k| k.1).collect();
            let (after, kept_shared) = shared(mb, &boxes);
            MeshCheck {
                name: ma.name.clone(),
                image,
                triangles: [ma.tris.len(), mb.tris.len()],
                texture_px: [sa, sb],
                shared_texels: [shared(ma, &[]).0, after],
                kept_shared,
                density_median: [median(uv::density(ma, sa[0].max(sa[1]).max(1))), median(uv::density(mb, sb[0].max(sb[1]).max(1)))],
            }
        })
        .collect();
    let base = Camera::frame(a);
    let none = |_: usize, _: usize| None;
    let views = yaws
        .iter()
        .map(|&y| {
            let mut cam = base;
            cam.yaw = std::f32::consts::PI + y.to_radians();
            let o = |cam| render::Options { cam, w: px, h: px, background: ([40, 44, 52], [40, 44, 52]), filter: render::Filter::Smooth, positions: None };
            let fa = render::render(a, &ta, &o(cam), &none);
            let fb = render::render(b, &tb, &o(cam), &none);
            let (mut n, mut sum, mut over) = (0usize, 0u64, 0usize);
            for i in 0..px * px {
                if fa.pick[i] < 0 && fb.pick[i] < 0 {
                    continue;
                }
                let d = (0..3).map(|c| fa.rgba[i * 4 + c].abs_diff(fb.rgba[i * 4 + c])).max().unwrap_or(0);
                n += 1;
                sum += d as u64;
                over += (d > 24) as usize;
            }
            ViewCheck { yaw_deg: y, model_px: n, mean_diff: if n == 0 { 0.0 } else { sum as f32 / n as f32 }, over_24: over }
        })
        .collect();
    Checks { same_skeleton: a.joint_names == b.joint_names, same_clips: a.clips == b.clips, meshes, views }
}
