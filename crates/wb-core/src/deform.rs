//! Deformation checks: pose a skinned model through its clips at fixed time steps and measure
//! what the skin does to the surface. Everything is computed against the model's own rest pose, so
//! the numbers compare across meshes of different resolution:
//!
//! * `stretch` / `squash`: share of the rest surface area with an edge that grew past
//!   `stretch_edge` times its rest length, or shrank below `squash_edge` (candy-wrapper twists at
//!   wrists and elbows pinch the edges around the limb).
//! * `volume`: per joint, the volume of the cone from the joint to the surface patch around it
//!   (posed against rest). Loss means the limb collapsed at the joint.
//! * `intersect`: share of the surface in triangles that pass through a triangle they share no
//!   corner with, ignoring pairs that already crossed at rest (clothing shells).
//! * `weights`: vertices whose weights do not sum to 1, with more than 4 influences, or that name
//!   a joint the skin does not have.
use crate::anim::{self, Local, Mat4};
use crate::bvh::Bvh;
use crate::geom::{ray_tri, V3};
use crate::weights::{region_of_bone, Skeleton};
use crate::Model;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct Limits {
    /// An edge counts as stretched above this posed/rest length ratio.
    pub stretch_edge: f32,
    /// An edge counts as squashed below this ratio.
    pub squash_edge: f32,
    /// An edge only counts when it grew or shrank by at least this share of the model's height
    /// (a sliver whose tiny rest edge doubles is not a visible stretch).
    pub min_change_frac: f32,
    /// Absolute limits (used without a reference): percent of rest area, percent of cone volume, vertex count.
    pub stretch_area_pct: f32,
    pub squash_area_pct: f32,
    pub volume_loss_pct: f32,
    pub intersect_area_pct: f32,
    /// Stretch and squash inside one body region: percent of that region's own area.
    pub stretch_region_pct: f32,
    pub squash_region_pct: f32,
    /// With a reference: allowed multiple of the reference's value, and a floor under that limit.
    pub max_x: f32,
    /// Floor under a region's limit (regions are small: a few triangles make a few percent).
    pub floor_region_pct: f32,
    pub floor_stretch_area_pct: f32,
    pub floor_squash_area_pct: f32,
    pub floor_volume_loss_pct: f32,
    pub floor_intersect_area_pct: f32,
}

impl Default for Limits {
    fn default() -> Self {
        // Calibrated on stock Zidane (see skills/gaw/references/deform.md for the table).
        Limits {
            stretch_edge: 1.5,
            squash_edge: 0.5,
            min_change_frac: 0.006,
            stretch_area_pct: 3.0,
            squash_area_pct: 1.5,
            volume_loss_pct: 85.0,
            intersect_area_pct: 10.0,
            stretch_region_pct: 9.0,
            squash_region_pct: 4.0,
            max_x: 1.5,
            floor_region_pct: 2.5,
            floor_stretch_area_pct: 1.0,
            floor_squash_area_pct: 1.0,
            floor_volume_loss_pct: 10.0,
            floor_intersect_area_pct: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Poses
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct FrameDesc {
    pub clip: String,
    /// Position in the clip, 0..1 (the clip's duration times this is the time).
    pub frac: f32,
    pub step: usize,
}

/// Where a model's poses come from: its own clips, or (when it has none) a reference's, retargeted
/// by joint order.
pub struct PoseSource<'a> {
    pub model: &'a Model,
    pub borrowed: Option<&'a Model>,
}

impl<'a> PoseSource<'a> {
    pub fn clip_names(&self) -> Vec<String> {
        let own = self.model.rig.as_ref().map(|r| r.clips.iter().map(|c| c.name.clone()).collect::<Vec<_>>()).unwrap_or_default();
        if !own.is_empty() {
            return own;
        }
        self.borrowed.and_then(|b| b.rig.as_ref()).map(|r| r.clips.iter().map(|c| c.name.clone()).collect()).unwrap_or_default()
    }

    /// Time in seconds of a frame in the named clip of `m` (None when `m` has no such clip).
    fn clip_time(m: &Model, clip: &str, frac: f32) -> Option<(usize, f32)> {
        let rig = m.rig.as_ref()?;
        let ci = rig.clips.iter().position(|c| c.name == clip)?;
        Some((ci, frac * rig.clips[ci].duration))
    }

    /// Joint matrices of the model in a frame; None when no clip of that name exists to pose it.
    pub fn joint_matrices(&self, f: &FrameDesc) -> Option<Vec<Mat4>> {
        let rig = self.model.rig.as_ref()?;
        if let Some((ci, t)) = Self::clip_time(self.model, &f.clip, f.frac) {
            return Some(rig.joint_matrices(Some(ci), t));
        }
        let b = self.borrowed?;
        let (ci, t) = Self::clip_time(b, &f.clip, f.frac)?;
        let locals: Vec<Local> = b.rig.as_ref()?.joint_locals(Some(ci), t);
        if locals.len() != rig.joints.len() {
            return None;
        }
        Some(rig.joint_matrices_with(&locals))
    }

    pub fn seconds(&self, f: &FrameDesc) -> f32 {
        Self::clip_time(self.model, &f.clip, f.frac).or_else(|| self.borrowed.and_then(|b| Self::clip_time(b, &f.clip, f.frac))).map(|x| x.1).unwrap_or(0.0)
    }

    pub fn posed(&self, f: &FrameDesc) -> Option<Vec<Vec<V3>>> {
        let jm = self.joint_matrices(f)?;
        Some(self.model.meshes.iter().map(|m| anim::skin(m, &jm)).collect())
    }
}

/// Frames for the named clips (all when `clips` is empty) at `steps` fixed fractions each.
pub fn frames_for(names: &[String], wanted: &[String], steps: usize) -> Result<Vec<FrameDesc>, String> {
    let mut out = Vec::new();
    let chosen: Vec<&String> = if wanted.is_empty() {
        names.iter().collect()
    } else {
        let mut v = Vec::new();
        for w in wanted {
            v.push(names.iter().find(|n| *n == w).ok_or_else(|| format!("no clip named `{w}` (clips: {})", names.join(", ")))?);
        }
        v
    };
    for c in chosen {
        for s in 0..steps.max(1) {
            out.push(FrameDesc { clip: c.clone(), frac: s as f32 / steps.max(1) as f32, step: s });
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Static weight check
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct BadVertex {
    pub mesh: usize,
    pub vertex: usize,
    pub sum: f32,
    pub influences: usize,
    pub why: String,
}

pub struct WeightCheck {
    pub vertices: usize,
    pub bad: Vec<BadVertex>,
    pub not_sum_one: usize,
    pub over_four: usize,
    pub bad_joint: usize,
}

/// Reads every JOINTS_n / WEIGHTS_n pair of every primitive (the model reader keeps only set 0).
pub fn check_weights(model: &Model) -> Result<WeightCheck, String> {
    let glb = &model.glb;
    let nj = model.joint_names.len();
    let mut res = WeightCheck { vertices: 0, bad: Vec::new(), not_sum_one: 0, over_four: 0, bad_joint: 0 };
    let mut mi = 0usize;
    for mesh in glb.list("meshes") {
        for prim in mesh.get("primitives").and_then(|p| p.as_array()).into_iter().flatten() {
            let m = &model.meshes[mi];
            let attrs = prim.get("attributes");
            let mut sets: Vec<(crate::glb::Accessor, crate::glb::Accessor)> = Vec::new();
            for n in 0.. {
                let j = attrs.and_then(|a| a.get(format!("JOINTS_{n}"))).and_then(|v| v.as_u64());
                let w = attrs.and_then(|a| a.get(format!("WEIGHTS_{n}"))).and_then(|v| v.as_u64());
                match (j, w) {
                    (Some(j), Some(w)) => sets.push((glb.accessor(j as usize)?, glb.accessor(w as usize)?)),
                    _ => break,
                }
            }
            for v in 0..m.pos.len() {
                res.vertices += 1;
                let mut sum = 0f32;
                let mut used: Vec<usize> = Vec::new();
                let mut why: Vec<String> = Vec::new();
                for (ja, wa) in &sets {
                    for c in 0..4 {
                        let (j, w) = (ja.get(v, c) as usize, wa.get(v, c) as f32);
                        if !w.is_finite() || w < 0.0 {
                            why.push(format!("weight {w}"));
                            continue;
                        }
                        if w > 1e-6 {
                            sum += w;
                            if j >= nj {
                                res.bad_joint += 1;
                                why.push(format!("joint {j} does not exist"));
                            }
                            if !used.contains(&j) {
                                used.push(j);
                            }
                        }
                    }
                }
                if sets.is_empty() {
                    sum = 0.0;
                }
                let mut bad = !why.is_empty();
                if (sum - 1.0).abs() > 1e-3 {
                    res.not_sum_one += 1;
                    bad = true;
                    why.push(format!("weights sum to {sum:.4}"));
                }
                if used.len() > 4 {
                    res.over_four += 1;
                    bad = true;
                    why.push(format!("{} influences", used.len()));
                }
                if bad {
                    res.bad.push(BadVertex { mesh: mi, vertex: v, sum, influences: used.len(), why: why.join("; ") });
                }
            }
            mi += 1;
        }
    }
    Ok(res)
}

// ---------------------------------------------------------------------------------------------
// Per-model preparation
// ---------------------------------------------------------------------------------------------

pub struct Prep {
    /// Triangles that count: (mesh, triangle) of every non-overlay mesh.
    pub tris: Vec<(u32, u32)>,
    /// Rest edge lengths per counted triangle: (a-b, b-c, c-a).
    pub rest_edges: Vec<[f32; 3]>,
    pub rest_area: Vec<f32>,
    pub total_area: f32,
    /// Dominant joint per counted triangle.
    pub bone: Vec<u16>,
    /// Weld group of each corner of each counted triangle (position-welded across meshes).
    pub corner_group: Vec<[u32; 3]>,
    pub height: f32,
    /// Pairs (counted-triangle indices, a < b) that already cross at rest.
    pub rest_pairs: HashSet<(u32, u32)>,
    pub joint_zones: Vec<JointZone>,
    pub skel: Skeleton,
    pub region_of_joint: Vec<String>,
    /// Body regions present (sorted), the region of each counted triangle (index into `regions`) and each region's rest area.
    pub regions: Vec<String>,
    pub tri_region: Vec<u8>,
    pub region_area: Vec<f32>,
}

pub struct JointZone {
    pub joint: usize,
    pub head: V3,
    /// Counted-triangle indices in the zone.
    pub tris: Vec<u32>,
    pub rest_volume: f32,
}

fn sub3(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot3(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross3(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn len3(a: V3) -> f32 {
    dot3(a, a).sqrt()
}
pub(crate) fn xform(m: &Mat4, p: V3) -> V3 {
    [m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]]
}

fn tri_pts(pos: &[Vec<V3>], model: &Model, t: (u32, u32)) -> [V3; 3] {
    model.meshes[t.0 as usize].tris[t.1 as usize].map(|v| pos[t.0 as usize][v as usize])
}

/// Do two triangles cross (an edge of one passes through the other)?
pub fn tris_cross(a: [V3; 3], b: [V3; 3]) -> bool {
    let hits = |e: [V3; 3], f: [V3; 3]| {
        (0..3).any(|k| {
            let (p, q) = (e[k], e[(k + 1) % 3]);
            ray_tri(p, sub3(q, p), f[0], f[1], f[2], 1e-5).is_some_and(|t| t < 1.0 - 1e-5)
        })
    };
    hits(a, b) || hits(b, a)
}

/// Crossing pairs among the counted triangles at the given vertex positions, as counted-triangle
/// index pairs (a < b), skipping pairs that share a welded corner.
pub fn crossing_pairs(model: &Model, prep: &Prep, pos: &[Vec<V3>]) -> Vec<(u32, u32)> {
    let eps = prep.height * 1e-6;
    let pts: Vec<[V3; 3]> = prep.tris.iter().map(|&t| tri_pts(pos, model, t)).collect();
    let boxes: Vec<(V3, V3)> = pts
        .iter()
        .map(|p| (std::array::from_fn(|k| p[0][k].min(p[1][k]).min(p[2][k]) - eps), std::array::from_fn(|k| p[0][k].max(p[1][k]).max(p[2][k]) + eps)))
        .collect();
    let bvh = Bvh::build(&boxes);
    let mut out = Vec::new();
    bvh.self_pairs(|a, b| {
        let (ga, gb) = (prep.corner_group[a as usize], prep.corner_group[b as usize]);
        if ga.iter().any(|g| gb.contains(g)) {
            return;
        }
        if tris_cross(pts[a as usize], pts[b as usize]) {
            out.push((a.min(b), a.max(b)));
        }
    });
    out.sort_unstable();
    out
}

fn field_mesh(model: &Model, mi: usize, all: bool) -> bool {
    all || !model.meshes[mi].name.starts_with("battle_model")
}

impl Prep {
    pub fn new(model: &Model, all_meshes: bool) -> Result<Prep, String> {
        let skel = Skeleton::from_model(model)?;
        let mut tris = Vec::new();
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for (mi, m) in model.meshes.iter().enumerate() {
            if !field_mesh(model, mi, all_meshes) {
                continue;
            }
            for ti in 0..m.tris.len() {
                tris.push((mi as u32, ti as u32));
            }
            for p in &m.pos {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
        if tris.is_empty() {
            return Err("the model has no triangles to check".into());
        }
        let height = (hi[1] - lo[1]).max(1e-6);
        let q = (height * 1e-6).max(1e-9);
        let mut map: HashMap<[i64; 3], u32> = HashMap::new();
        let mut vgroup: Vec<Vec<u32>> = Vec::new();
        for m in &model.meshes {
            let mut v = Vec::with_capacity(m.pos.len());
            for &p in &m.pos {
                let k = [(p[0] / q).round() as i64, (p[1] / q).round() as i64, (p[2] / q).round() as i64];
                let n = map.len() as u32;
                v.push(*map.entry(k).or_insert(n));
            }
            vgroup.push(v);
        }
        let rest: Vec<Vec<V3>> = model.meshes.iter().map(|m| m.pos.clone()).collect();
        let mut rest_edges = Vec::with_capacity(tris.len());
        let mut rest_area = Vec::with_capacity(tris.len());
        let mut bone = Vec::with_capacity(tris.len());
        let mut corner_group = Vec::with_capacity(tris.len());
        for &t in &tris {
            let p = tri_pts(&rest, model, t);
            rest_edges.push([len3(sub3(p[1], p[0])), len3(sub3(p[2], p[1])), len3(sub3(p[0], p[2]))]);
            rest_area.push(0.5 * len3(cross3(sub3(p[1], p[0]), sub3(p[2], p[0]))));
            bone.push(model.triangle_joint(t.0 as usize, t.1 as usize) as u16);
            corner_group.push(model.meshes[t.0 as usize].tris[t.1 as usize].map(|v| vgroup[t.0 as usize][v as usize]));
        }
        let total_area: f32 = rest_area.iter().sum();
        let region_of_joint: Vec<String> = skel.names.iter().map(|n| region_of_bone(n)).collect();
        // Every region the skeleton has, not only those with triangles: a candidate and its reference must
        // share region indices (a tailless model has no tail triangles but the same joints).
        let mut regions: Vec<String> = region_of_joint.iter().cloned().collect::<HashSet<_>>().into_iter().collect();
        regions.sort();
        let tri_region: Vec<u8> = bone.iter().map(|&b| regions.iter().position(|r| *r == region_of_joint[b as usize]).unwrap() as u8).collect();
        let mut region_area = vec![0f32; regions.len()];
        for (i, &r) in tri_region.iter().enumerate() {
            region_area[r as usize] += rest_area[i];
        }
        let mut prep = Prep { tris, rest_edges, rest_area, total_area, bone, corner_group, height, rest_pairs: HashSet::new(), joint_zones: Vec::new(), skel, region_of_joint, regions, tri_region, region_area };
        prep.rest_pairs = crossing_pairs(model, &prep, &rest).into_iter().collect();
        prep.joint_zones = joint_zones(model, &prep, &rest);
        Ok(prep)
    }
}

/// The surface patch around each joint that has a parent and a child: counted triangles whose
/// centroid lies within 0.45 of the shorter adjacent bone of the joint's rest position and whose
/// dominant bone is the joint, its parent or one of its children.
fn joint_zones(model: &Model, prep: &Prep, rest: &[Vec<V3>]) -> Vec<JointZone> {
    let sk = &prep.skel;
    let n = sk.names.len();
    let mut kids: Vec<Vec<usize>> = vec![Vec::new(); n];
    for j in 0..n {
        if let Some(p) = sk.parent[j] {
            kids[p].push(j);
        }
    }
    let cent: Vec<V3> = prep
        .tris
        .iter()
        .map(|&t| {
            let p = tri_pts(rest, model, t);
            [(p[0][0] + p[1][0] + p[2][0]) / 3.0, (p[0][1] + p[1][1] + p[2][1]) / 3.0, (p[0][2] + p[1][2] + p[2][2]) / 3.0]
        })
        .collect();
    let mut zones = Vec::new();
    for j in 0..n {
        let Some(p) = sk.parent[j] else { continue };
        if kids[j].is_empty() {
            continue;
        }
        let head = sk.head(j);
        let seg = |k: usize| len3(sub3(sk.head(k), head));
        let shortest = std::iter::once(seg(p)).chain(kids[j].iter().map(|&k| seg(k))).filter(|&d| d > 1e-4).fold(f32::MAX, f32::min);
        if shortest == f32::MAX {
            continue;
        }
        let r = 0.45 * shortest;
        let allowed: Vec<usize> = std::iter::once(j).chain(std::iter::once(p)).chain(kids[j].iter().copied()).collect();
        let tris: Vec<u32> = (0..prep.tris.len() as u32).filter(|&i| len3(sub3(cent[i as usize], head)) <= r && allowed.contains(&(prep.bone[i as usize] as usize))).collect();
        if tris.len() < 6 {
            continue;
        }
        let v = cone_volume(model, prep, &tris, rest, head);
        let area: f32 = tris.iter().map(|&i| prep.rest_area[i as usize]).sum();
        // A patch whose cone volume is tiny against its area is not wrapped around anything.
        if v.abs() < 1e-3 * area * shortest {
            continue;
        }
        zones.push(JointZone { joint: j, head, tris, rest_volume: v });
    }
    zones
}

/// Signed volume of the cones from `apex` over the listed triangles.
pub(crate) fn cone_volume(model: &Model, prep: &Prep, tris: &[u32], pos: &[Vec<V3>], apex: V3) -> f32 {
    tris.iter()
        .map(|&i| {
            let p = tri_pts(pos, model, prep.tris[i as usize]);
            dot3(sub3(p[0], apex), cross3(sub3(p[1], apex), sub3(p[2], apex))) / 6.0
        })
        .sum()
}

// ---------------------------------------------------------------------------------------------
// Measuring a pose
// ---------------------------------------------------------------------------------------------

/// A triangle (or pair, or joint) that stands out in one frame.
#[derive(Clone, Debug)]
pub struct Spot {
    /// Index into `Measured::frames`.
    pub frame: usize,
    /// Counted-triangle index (stretch, squash, intersect), second triangle for intersect.
    pub tri: u32,
    pub tri2: Option<u32>,
    /// Joint index for volume spots.
    pub joint: Option<usize>,
    pub value: f32,
    pub pos: V3,
}

#[derive(Default, Clone, Debug)]
pub struct FrameStats {
    pub stretch_area_pct: f32,
    pub squash_area_pct: f32,
    pub worst_stretch: f32,
    pub worst_squash: f32,
    /// Per joint zone: volume ratio (posed over rest), aligned with `Prep::joint_zones`.
    pub volume_ratio: Vec<f32>,
    pub intersect_pairs: usize,
    pub intersect_tris: usize,
    pub intersect_clusters: usize,
    pub intersect_area_pct: f32,
    /// Stretch and squash again per body region (percent of the region's own rest area), aligned with `Prep::regions`.
    pub stretch_region_pct: Vec<f32>,
    pub squash_region_pct: Vec<f32>,
}

pub struct Measured {
    pub frames: Vec<FrameDesc>,
    /// Frames the model could not be posed in (no clip of that name).
    pub missing: Vec<usize>,
    pub stats: Vec<Option<FrameStats>>,
    pub stretch: Vec<Spot>,
    pub squash: Vec<Spot>,
    pub volume: Vec<Spot>,
    pub intersect: Vec<Spot>,
}

const KEEP_PER_FRAME: usize = 24;
const KEEP_PER_REGION: usize = 8;

fn keep_top(v: &mut Vec<Spot>, s: Spot, hi_is_bad: bool) {
    v.push(s);
    if v.len() > KEEP_PER_FRAME * 2 {
        v.sort_by(|a, b| if hi_is_bad { b.value.total_cmp(&a.value) } else { a.value.total_cmp(&b.value) });
        v.truncate(KEEP_PER_FRAME);
    }
}

pub fn measure(model: &Model, prep: &Prep, src: &PoseSource, frames: &[FrameDesc], limits: &Limits) -> Measured {
    let mut out = Measured { frames: frames.to_vec(), missing: Vec::new(), stats: Vec::new(), stretch: Vec::new(), squash: Vec::new(), volume: Vec::new(), intersect: Vec::new() };
    for (fi, f) in frames.iter().enumerate() {
        let Some(jm) = src.joint_matrices(f) else {
            out.missing.push(fi);
            out.stats.push(None);
            continue;
        };
        let pos: Vec<Vec<V3>> = model.meshes.iter().map(|m| anim::skin(m, &jm)).collect();
        let mut st = FrameStats { worst_squash: 1.0, worst_stretch: 1.0, ..Default::default() };
        // Edge stretch and squash.
        let (mut a_st, mut a_sq) = (0f32, 0f32);
        let (mut r_st, mut r_sq) = (vec![0f32; prep.regions.len()], vec![0f32; prep.regions.len()]);
        let min_edge = prep.height * 1e-4;
        let min_change = prep.height * limits.min_change_frac;
        let mut frame_stretch: Vec<Vec<Spot>> = vec![Vec::new(); prep.regions.len()];
        let mut frame_squash: Vec<Vec<Spot>> = vec![Vec::new(); prep.regions.len()];
        for (i, &t) in prep.tris.iter().enumerate() {
            let p = tri_pts(&pos, model, t);
            let e = [len3(sub3(p[1], p[0])), len3(sub3(p[2], p[1])), len3(sub3(p[0], p[2]))];
            let (mut mx, mut mn) = (1f32, 1f32);
            for k in 0..3 {
                let r0 = prep.rest_edges[i][k];
                if r0 < min_edge || (e[k] - r0).abs() < min_change {
                    continue;
                }
                let r = e[k] / r0;
                mx = mx.max(r);
                mn = mn.min(r);
            }
            let c = [(p[0][0] + p[1][0] + p[2][0]) / 3.0, (p[0][1] + p[1][1] + p[2][1]) / 3.0, (p[0][2] + p[1][2] + p[2][2]) / 3.0];
            if mx > limits.stretch_edge {
                a_st += prep.rest_area[i];
                r_st[prep.tri_region[i] as usize] += prep.rest_area[i];
            }
            if mn < limits.squash_edge {
                a_sq += prep.rest_area[i];
                r_sq[prep.tri_region[i] as usize] += prep.rest_area[i];
            }
            st.worst_stretch = st.worst_stretch.max(mx);
            st.worst_squash = st.worst_squash.min(mn);
            if mx > limits.stretch_edge * 0.8 {
                keep_top(&mut frame_stretch[prep.tri_region[i] as usize], Spot { frame: fi, tri: i as u32, tri2: None, joint: None, value: mx, pos: c }, true);
            }
            if mn < 1.0 - (1.0 - limits.squash_edge) * 0.8 {
                keep_top(&mut frame_squash[prep.tri_region[i] as usize], Spot { frame: fi, tri: i as u32, tri2: None, joint: None, value: mn, pos: c }, false);
            }
        }
        st.stretch_area_pct = 100.0 * a_st / prep.total_area.max(1e-12);
        st.squash_area_pct = 100.0 * a_sq / prep.total_area.max(1e-12);
        st.stretch_region_pct = r_st.iter().zip(&prep.region_area).map(|(a, t)| 100.0 * a / t.max(1e-12)).collect();
        st.squash_region_pct = r_sq.iter().zip(&prep.region_area).map(|(a, t)| 100.0 * a / t.max(1e-12)).collect();
        for mut v in frame_stretch {
            v.sort_by(|a, b| b.value.total_cmp(&a.value));
            v.truncate(KEEP_PER_REGION);
            out.stretch.extend(v);
        }
        for mut v in frame_squash {
            v.sort_by(|a, b| a.value.total_cmp(&b.value));
            v.truncate(KEEP_PER_REGION);
            out.squash.extend(v);
        }
        // Joint volumes.
        for z in &prep.joint_zones {
            let apex = xform(&jm[z.joint], z.head);
            let v = cone_volume(model, prep, &z.tris, &pos, apex);
            let ratio = if z.rest_volume.abs() > 0.0 { v / z.rest_volume } else { 1.0 };
            st.volume_ratio.push(ratio);
            out.volume.push(Spot { frame: fi, tri: z.tris[0], tri2: None, joint: Some(z.joint), value: ratio, pos: apex });
        }
        // Intersections new since rest.
        let pairs: Vec<(u32, u32)> = crossing_pairs(model, prep, &pos).into_iter().filter(|p| !prep.rest_pairs.contains(p)).collect();
        st.intersect_pairs = pairs.len();
        let mut involved: Vec<u32> = pairs.iter().flat_map(|p| [p.0, p.1]).collect();
        involved.sort_unstable();
        involved.dedup();
        st.intersect_tris = involved.len();
        st.intersect_area_pct = 100.0 * involved.iter().map(|&i| prep.rest_area[i as usize]).sum::<f32>() / prep.total_area.max(1e-12);
        // Clusters: involved triangles joined through shared welded corners.
        let mut parent: HashMap<u32, u32> = involved.iter().map(|&i| (i, i)).collect();
        fn find(p: &mut HashMap<u32, u32>, x: u32) -> u32 {
            let mut r = x;
            while p[&r] != r {
                r = p[&r];
            }
            let mut c = x;
            while p[&c] != r {
                let n = p[&c];
                p.insert(c, r);
                c = n;
            }
            r
        }
        let mut by_group: HashMap<u32, u32> = HashMap::new();
        for &i in &involved {
            for g in prep.corner_group[i as usize] {
                match by_group.get(&g) {
                    Some(&o) => {
                        let (a, b) = (find(&mut parent, i), find(&mut parent, o));
                        if a != b {
                            parent.insert(a, b);
                        }
                    }
                    None => {
                        by_group.insert(g, i);
                    }
                }
            }
        }
        let roots: HashSet<u32> = involved.iter().map(|&i| find(&mut parent, i)).collect();
        st.intersect_clusters = roots.len();
        // The worst pairs: those whose triangles have the largest rest area, one per cluster.
        let mut seen: HashSet<u32> = HashSet::new();
        let mut ranked: Vec<&(u32, u32)> = pairs.iter().collect();
        ranked.sort_by(|a, b| {
            let ar = |p: &&(u32, u32)| prep.rest_area[p.0 as usize] + prep.rest_area[p.1 as usize];
            ar(b).total_cmp(&ar(a))
        });
        for p in ranked {
            let root = find(&mut parent, p.0);
            if !seen.insert(root) {
                continue;
            }
            let (ta, tb) = (tri_pts(&pos, model, prep.tris[p.0 as usize]), tri_pts(&pos, model, prep.tris[p.1 as usize]));
            let c = [
                (ta[0][0] + ta[1][0] + ta[2][0] + tb[0][0] + tb[1][0] + tb[2][0]) / 6.0,
                (ta[0][1] + ta[1][1] + ta[2][1] + tb[0][1] + tb[1][1] + tb[2][1]) / 6.0,
                (ta[0][2] + ta[1][2] + ta[2][2] + tb[0][2] + tb[1][2] + tb[2][2]) / 6.0,
            ];
            let cluster_area: f32 = involved.iter().filter(|&&i| find(&mut parent, i) == root).map(|&i| prep.rest_area[i as usize]).sum();
            keep_top(&mut out.intersect, Spot { frame: fi, tri: p.0, tri2: Some(p.1), joint: None, value: cluster_area, pos: c }, true);
        }
        out.stats.push(Some(st));
    }
    out
}

impl Measured {
    /// Worst frame's value of one region's stretch (squash = false) or squash share.
    pub fn max_region(&self, region: usize, stretch: bool) -> f32 {
        self.stats.iter().flatten().map(|s| if stretch { s.stretch_region_pct.get(region).copied().unwrap_or(0.0) } else { s.squash_region_pct.get(region).copied().unwrap_or(0.0) }).fold(0.0, f32::max)
    }

    pub fn max_stat(&self, f: impl Fn(&FrameStats) -> f32) -> f32 {
        self.stats.iter().flatten().map(&f).fold(0.0, f32::max)
    }

    /// Worst joint volume loss in percent over all frames, and the (joint, frame) it happened at.
    pub fn worst_volume_loss(&self, prep: &Prep) -> (f32, Option<(usize, usize)>) {
        let mut best = (0f32, None);
        for (fi, s) in self.stats.iter().enumerate() {
            let Some(s) = s else { continue };
            for (zi, &r) in s.volume_ratio.iter().enumerate() {
                let loss = 100.0 * (1.0 - r).max(0.0);
                if loss > best.0 {
                    best = (loss, Some((prep.joint_zones[zi].joint, fi)));
                }
            }
        }
        best
    }

    /// Volume ratio of a joint in a frame (None when the joint has no zone or the frame was not posed).
    pub fn volume_ratio_of(&self, prep: &Prep, joint: usize, frame: usize) -> Option<f32> {
        let zi = prep.joint_zones.iter().position(|z| z.joint == joint)?;
        self.stats.get(frame)?.as_ref()?.volume_ratio.get(zi).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossing_triangles_are_found_and_touching_ones_are_not() {
        let a: [V3; 3] = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let through: [V3; 3] = [[0.5, 0.5, -1.0], [0.5, 0.5, 1.0], [1.5, 0.5, 1.0]];
        assert!(tris_cross(a, through));
        let apart: [V3; 3] = [[0.5, 0.5, 1.0], [0.5, 0.5, 2.0], [1.5, 0.5, 2.0]];
        assert!(!tris_cross(a, apart));
    }

    #[test]
    fn frames_follow_the_clip_names_and_fractions() {
        let names = vec!["walk".to_string(), "run".to_string()];
        let f = frames_for(&names, &[], 4).unwrap();
        assert_eq!(f.len(), 8);
        assert_eq!(f[1].clip, "walk");
        assert!((f[1].frac - 0.25).abs() < 1e-6);
        let f = frames_for(&names, &["run".to_string()], 3).unwrap();
        assert_eq!(f.len(), 3);
        assert!(frames_for(&names, &["jump".to_string()], 3).is_err());
    }
}
