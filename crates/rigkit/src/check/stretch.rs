//! Skin stretch: how far each triangle's edges and area change from rest, pose by pose, and which
//! triangles turn inside out. A torn elbow or a pinched socket shows up here as a big ratio.
use crate::skin::Skin;
use crate::{m, v, Mat4, Vec3};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct StretchHit {
    pub tri: usize,
    /// Pose index where this triangle was at its worst.
    pub pose: usize,
    /// Worst edge length change, as a ratio of at least 1 (2.0 = twice or half as long).
    pub edge: f32,
    /// True when that worst edge got shorter (skin pinched or folded), false when it got longer
    /// (skin pulled).
    pub shrunk: bool,
    /// Area change at that pose, the same way.
    pub area: f32,
    pub flipped: bool,
}

#[derive(Clone, Debug, Default)]
pub struct StretchReport {
    pub worst_edge: f32,
    pub worst_area: f32,
    /// Triangles that turned inside out in at least one pose.
    pub flipped: usize,
    /// The worst triangles, worst first.
    pub hits: Vec<StretchHit>,
}

/// Runs every pose in `poses` over the mesh. Edges shorter than `min_edge` at rest are skipped (they
/// are welds and slivers whose ratios mean nothing). Keeps the `keep` worst triangles.
pub fn stretch(rest: &[Vec3], indices: &[u32], skin: &Skin, poses: &[Vec<Mat4>], min_edge: f32, keep: usize) -> StretchReport {
    let tris: Vec<[usize; 3]> = indices.chunks_exact(3).map(|t| [t[0] as usize, t[1] as usize, t[2] as usize]).collect();
    let r: Vec<glam::Vec3> = rest.iter().map(v).collect();
    let ratio = |a: f32, b: f32| if a > b { a / b } else { b / a };
    // per triangle, its worst pose
    let mut best: Vec<StretchHit> = (0..tris.len()).map(|t| StretchHit { tri: t, pose: 0, edge: 1.0, shrunk: false, area: 1.0, flipped: false }).collect();
    let mut ever_flipped = vec![false; tris.len()];
    for (k, pose) in poses.iter().enumerate() {
        let mats: Vec<glam::Mat4> = pose.iter().map(m).collect();
        let blended: Vec<glam::Mat4> = (0..rest.len()).into_par_iter().map(|i| skin.blend(&mats, i)).collect();
        let p: Vec<glam::Vec3> = (0..rest.len()).into_par_iter().map(|i| blended[i].transform_point3(r[i])).collect();
        let now: Vec<Option<(f32, bool, f32, bool)>> = tris.par_iter().map(|t| {
            let (a0, b0, c0) = (r[t[0]], r[t[1]], r[t[2]]);
            let n0 = (b0 - a0).cross(c0 - a0);
            if n0.length() < min_edge * min_edge { return None; }
            let (a1, b1, c1) = (p[t[0]], p[t[1]], p[t[2]]);
            let n1 = (b1 - a1).cross(c1 - a1);
            let (mut e, mut shrunk) = (1.0f32, false);
            for (x0, y0, x1, y1) in [(a0, b0, a1, b1), (b0, c0, b1, c1), (c0, a0, c1, a1)] {
                let (l0, l1) = (x0.distance(y0), x1.distance(y1));
                let q = ratio(l0, l1.max(1e-9));
                if l0 >= min_edge && q > e { e = q; shrunk = l1 < l0; }
            }
            let area = ratio(n0.length(), n1.length().max(1e-12));
            // inside out: the posed face points against its rest normal carried by its own bones
            let carried = blended[t[0]].transform_vector3(n0);
            Some((e, shrunk, area, n1.dot(carried) < 0.0))
        }).collect();
        for (t, x) in now.into_iter().enumerate() {
            if let Some((e, sh, a, f)) = x {
                if f { ever_flipped[t] = true; }
                if e > best[t].edge { best[t] = StretchHit { tri: t, pose: k, edge: e, shrunk: sh, area: a, flipped: f }; }
            }
        }
    }
    best.sort_by(|a, b| b.edge.total_cmp(&a.edge));
    StretchReport {
        worst_edge: best.first().map_or(1.0, |h| h.edge),
        worst_area: best.iter().map(|h| h.area).fold(1.0, f32::max),
        flipped: ever_flipped.iter().filter(|f| **f).count(),
        hits: best.into_iter().take(keep).collect(),
    }
}
