//! Skin-weight transfer for fitting a new mesh to an existing skeleton.
//!
//! The reference model and the new mesh share one skeleton and rest pose. Each new vertex takes the
//! weights of the closest point on the reference surface (barycentric interpolation of the three
//! corner weights). Optional smoothing rings (off by default; stock's weights have hard cuts that
//! smoothing blurs) average over the new mesh's welded topology, only within a body region.
//! Output: at most 4 influences per vertex, sum 1.
use crate::anim::Mat4;
use crate::bvh::Bvh;
use crate::geom::{add, closest_on_tri, dist, dot, len, scale, sub, V3};
use crate::glb::{Chunk, Glb, CHUNK_BIN, CHUNK_JSON};
use crate::Model;
use serde_json::{json, Value};
use std::collections::HashMap;

// ---------------------------------------------------------------------------------------------
// Skeleton
// ---------------------------------------------------------------------------------------------

/// The skin's skeleton as the weights see it.
pub struct Skeleton {
    pub names: Vec<String>,
    /// Nearest ancestor that is also a joint, as a joint index.
    pub parent: Vec<Option<usize>>,
    /// Rest world matrix of each joint node.
    pub world: Vec<Mat4>,
    /// Inverse bind matrices.
    pub ibm: Vec<Mat4>,
}

impl Skeleton {
    pub fn from_model(m: &Model) -> Result<Skeleton, String> {
        let rig = m.rig.as_ref().ok_or("the file has no skin (no skeleton to match)")?;
        Ok(Skeleton { names: m.joint_names.clone(), parent: rig.joint_parents(), world: rig.rest_joint_world(), ibm: rig.inverse_binds().to_vec() })
    }

    /// Position of joint `k` at rest.
    pub fn head(&self, k: usize) -> V3 {
        [self.world[k][12], self.world[k][13], self.world[k][14]]
    }
}

fn mat_diff(a: &Mat4, b: &Mat4) -> (f32, f32) {
    let mut lin = 0f32;
    for c in 0..3 {
        for r in 0..3 {
            lin = lin.max((a[c * 4 + r] - b[c * 4 + r]).abs());
        }
    }
    let tr = dist([a[12], a[13], a[14]], [b[12], b[13], b[14]]);
    (lin, tr)
}

/// Why `new` is not the reference's skeleton: joint count, names in order, parents, rest pose and
/// inverse bind matrices. Empty when they match. `tol` is in model units (and unit-matrix entries).
pub fn skeleton_diff(new: &Skeleton, reference: &Skeleton, tol: f32) -> Vec<String> {
    let mut p: Vec<String> = Vec::new();
    if new.names.len() != reference.names.len() {
        p.push(format!("the new file has {} joints, the reference has {}", new.names.len(), reference.names.len()));
    }
    let n = new.names.len().min(reference.names.len());
    let mut name_bad = false;
    for k in 0..n {
        if new.names[k] != reference.names[k] {
            name_bad = true;
            if p.len() < 12 {
                p.push(format!("joint {k} is `{}` in the new file but `{}` in the reference", new.names[k], reference.names[k]));
            }
        }
    }
    if name_bad {
        let mut a = new.names.clone();
        let mut b = reference.names.clone();
        a.sort();
        b.sort();
        if a == b {
            p.push("same bone names but a different order: keep the reference's joint order".into());
        }
        return p; // everything below compares by index
    }
    for k in 0..n {
        let (pn, pr) = (new.parent[k].map(|i| new.names[i].as_str()), reference.parent[k].map(|i| reference.names[i].as_str()));
        if pn != pr && p.len() < 24 {
            p.push(format!("`{}` has parent {} in the new file but {} in the reference", new.names[k], pn.unwrap_or("none"), pr.unwrap_or("none")));
        }
    }
    for k in 0..n {
        let (lin, tr) = mat_diff(&new.world[k], &reference.world[k]);
        if (lin > tol || tr > tol) && p.len() < 36 {
            p.push(format!("`{}` rest pose differs: position off by {tr:.4}, rotation/scale entries by {lin:.4} (limit {tol})", new.names[k]));
        }
        let (lin, tr) = mat_diff(&new.ibm[k], &reference.ibm[k]);
        if (lin > tol || tr > tol) && p.len() < 36 {
            p.push(format!("`{}` inverse bind matrix differs: translation off by {tr:.4}, rotation/scale entries by {lin:.4} (limit {tol})", new.names[k]));
        }
    }
    p
}

/// Body region of a bone, from its name: `L_arm`, `R_arm`, `L_leg`, `R_leg`, `head`, `tail`, else `torso`.
pub fn region_of_bone(name: &str) -> String {
    let n = name.to_ascii_lowercase();
    let side = if n.contains("_l_") { Some("L") } else if n.contains("_r_") { Some("R") } else { None };
    if n.contains("tail") {
        return "tail".into();
    }
    if n.contains("head") || n.contains("neck") {
        return "head".into();
    }
    if let Some(s) = side {
        if ["arm", "hand", "shoulder"].iter().any(|k| n.contains(k)) {
            return format!("{s}_arm");
        }
        if ["hip", "thigh", "shin", "foot"].iter().any(|k| n.contains(k)) {
            return format!("{s}_leg");
        }
    }
    "torso".into()
}

/// `--bones` style list: names (exact, else every bone whose name contains the text) or joint
/// numbers, comma separated. Returns which joints are named.
pub fn parse_bone_list(spec: &str, names: &[String]) -> Result<Vec<bool>, String> {
    let mut out = vec![false; names.len()];
    for tok in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        if let Ok(i) = tok.parse::<usize>() {
            if i >= names.len() {
                return Err(format!("bone number {i} is out of range (0..{})", names.len() - 1));
            }
            out[i] = true;
        } else if let Some(i) = names.iter().position(|n| n == tok) {
            out[i] = true;
        } else {
            let hits: Vec<usize> = (0..names.len()).filter(|&i| names[i].to_ascii_lowercase().contains(&tok.to_ascii_lowercase())).collect();
            if hits.is_empty() {
                return Err(format!("no bone is named or contains `{tok}`"));
            }
            for i in hits {
                out[i] = true;
            }
        }
    }
    if !out.iter().any(|&b| b) {
        return Err("the bone list names no bones".into());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Transfer
// ---------------------------------------------------------------------------------------------

pub struct TransferOptions {
    /// Joints that may receive weight (None = all). Weight on others moves to the nearest allowed ancestor.
    pub allowed: Option<Vec<bool>>,
    /// Smoothing rings (0 = none, the default: stock's hard-cut weights are best kept as they are).
    pub rings: usize,
    /// Share of each ring's neighbor average mixed in.
    pub lambda: f32,
    /// Vertices farther than this from the reference are flagged (None = 5 percent of the reference height).
    pub max_dist: Option<f32>,
    /// Use the reference's `battle_model*` overlay meshes as surface too.
    pub include_battle: bool,
    /// Matching metric. 1.0 = plain closest point. Below 1.0, each new vertex is matched to the
    /// reference near the same bone, with distance across the bone (around its axis) scaled by this
    /// factor and distance along the bone counted in full: a body of other proportions then lands at
    /// the same place along each limb instead of wherever the surface happens to be closest.
    pub radial: f32,
    /// With `radial` below 1: also move the reference's cuts between bones onto the new body's joint
    /// planes (the reference changes bones some distance before or after the joint; a new mesh built
    /// for the same skeleton bends at the joint).
    pub align_cuts: bool,
    /// With `radial` below 1: where the reference has layers on top of each other, or the new surface
    /// sits far from it, average the weights of every reference triangle about as close as the best
    /// one. The spread is this factor times the vertex's distance from the reference surface, so a
    /// surface that coincides with the reference keeps its hard cuts and a different one gets soft ones.
    pub soften: f32,
    /// Blend zone around each joint, in model units either side of its plane (0 = none). Inside it, a
    /// vertex whose weight sits wholly on the joint's two bones is re-weighted by position alone, with a
    /// smoothstep from the parent bone to the child bone: a hard cut that lands where the new mesh has
    /// edges crossing it would stretch them, a ramp shares the bend over several edges.
    pub joint_blend: f32,
}

impl Default for TransferOptions {
    fn default() -> Self {
        TransferOptions { allowed: None, rings: 0, lambda: 0.5, max_dist: None, include_battle: false, radial: 1.0, align_cuts: true, soften: 0.5, joint_blend: 0.0 }
    }
}

pub struct TransferResult {
    /// Per new mesh, per vertex.
    pub joints: Vec<Vec<[u16; 4]>>,
    pub weights: Vec<Vec<[f32; 4]>>,
    /// Distance from each vertex to the reference surface.
    pub dist: Vec<Vec<f32>>,
    /// Closest reference triangle per vertex: (reference mesh, triangle in it).
    pub ref_tri: Vec<Vec<(u32, u32)>>,
    /// Region (index into `regions`) of each vertex after the transfer.
    pub region: Vec<Vec<u8>>,
    pub regions: Vec<String>,
    pub max_dist: f32,
    pub ref_height: f32,
    pub welded_groups: usize,
    /// Reference meshes used as the surface.
    pub ref_meshes: Vec<String>,
}

/// Position-weld key.
fn qkey(p: V3, q: f32) -> [i64; 3] {
    [(p[0] / q).round() as i64, (p[1] / q).round() as i64, (p[2] / q).round() as i64]
}

/// Keeps the 4 largest weights, drops dust, renormalizes to sum 1. Unused slots are joint 0, weight 0.
pub fn top4(dense: &[f32]) -> ([u16; 4], [f32; 4]) {
    let mut idx: Vec<usize> = (0..dense.len()).filter(|&j| dense[j] > 1e-3).collect();
    idx.sort_by(|&a, &b| dense[b].total_cmp(&dense[a]).then(a.cmp(&b)));
    idx.truncate(4);
    if idx.is_empty() {
        // Nothing above dust: keep the single largest.
        if let Some(j) = (0..dense.len()).max_by(|&a, &b| dense[a].total_cmp(&dense[b])) {
            idx.push(j);
        }
    }
    let sum: f32 = idx.iter().map(|&j| dense[j]).sum();
    let mut j4 = [0u16; 4];
    let mut w4 = [0f32; 4];
    for (s, &j) in idx.iter().enumerate() {
        j4[s] = j as u16;
        w4[s] = if sum > 0.0 { dense[j] / sum } else { 1.0 };
    }
    // Make the f32 sum exactly 1 by absorbing rounding into the largest weight.
    let s: f32 = w4.iter().sum();
    w4[0] += 1.0 - s;
    (j4, w4)
}

/// Barycentric blend of three sparse weight sets into a dense vector.
pub fn blend3(js: [[u16; 4]; 3], ws: [[f32; 4]; 3], bary: [f32; 3], nj: usize) -> Vec<f32> {
    let mut d = vec![0f32; nj];
    for c in 0..3 {
        for s in 0..4 {
            let (j, w) = (js[c][s] as usize, ws[c][s]);
            if w > 0.0 && j < nj {
                d[j] += bary[c] * w;
            }
        }
    }
    d
}

/// Moves weight on a disallowed joint to its nearest allowed ancestor.
pub fn redirect_table(skel: &Skeleton, allowed: &[bool]) -> Result<Vec<usize>, String> {
    (0..allowed.len())
        .map(|j| {
            let mut at = Some(j);
            while let Some(k) = at {
                if allowed[k] {
                    return Ok(k);
                }
                at = skel.parent[k];
            }
            Err(format!("bone `{}` is not allowed and none of its parents is: allow its parent chain (at least the root)", skel.names[j]))
        })
        .collect()
}

/// One smoothing ring over welded neighbors, mixing only within a region.
pub fn smooth_ring(cur: &[Vec<f32>], adj: &[Vec<u32>], region: &[u8], lambda: f32) -> Vec<Vec<f32>> {
    let mut next = cur.to_vec();
    for g in 0..cur.len() {
        let same: Vec<&Vec<f32>> = adj[g].iter().filter(|&&n| region[n as usize] == region[g]).map(|&n| &cur[n as usize]).collect();
        if same.is_empty() {
            continue;
        }
        let inv = 1.0 / same.len() as f32;
        for j in 0..cur[g].len() {
            let mean: f32 = same.iter().map(|v| v[j]).sum::<f32>() * inv;
            next[g][j] = (1.0 - lambda) * cur[g][j] + lambda * mean;
        }
    }
    next
}

/// Distance from `p` to segment `a`-`b`.
fn point_segment(p: V3, a: V3, b: V3) -> f32 {
    let ab = sub(b, a);
    let l2 = dot(ab, ab);
    let t = if l2 > 1e-12 { (dot(sub(p, a), ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    dist(p, add(a, scale(ab, t)))
}

/// Re-weights vertices near each joint plane by position alone (see `TransferOptions::joint_blend`).
fn joint_blend(dense: &mut [Vec<f32>], gpos: &[V3], skel: &Skeleton, width: f32) {
    let nj = skel.names.len();
    let seg: Vec<Option<(V3, V3)>> = (0..nj)
        .map(|j| {
            let p = skel.parent[j]?;
            let (a, b) = (skel.head(p), skel.head(j));
            if dist(a, b) > 1e-4 {
                Some((a, b))
            } else {
                None
            }
        })
        .collect();
    for (g, &x) in gpos.iter().enumerate() {
        let Some(near) = (0..nj).filter(|&j| seg[j].is_some()).min_by(|&p, &q| {
            let (sp, sq) = (seg[p].unwrap(), seg[q].unwrap());
            point_segment(x, sp.0, sp.1).total_cmp(&point_segment(x, sq.0, sq.1))
        }) else {
            continue;
        };
        // The joint this vertex sits near: the end of its segment, or the joint its segment starts at.
        let (a, b) = seg[near].unwrap();
        for j in [near, skel.parent[near].unwrap_or(near)] {
            let Some(p) = skel.parent[j] else { continue };
            let Some((pa, pb)) = seg[j] else { continue };
            let _ = (a, b);
            let ax = sub(pb, pa);
            let d = scale(ax, 1.0 / len(ax));
            let u = dot(sub(x, pb), d);
            if u.abs() >= width {
                continue;
            }
            let outside: f32 = (0..nj).filter(|&k| k != p && k != j).map(|k| dense[g][k]).sum();
            if outside > 0.02 {
                continue;
            }
            let t = (u + width) / (2.0 * width);
            let s = t * t * (3.0 - 2.0 * t);
            let w = &mut dense[g];
            for v in w.iter_mut() {
                *v = 0.0;
            }
            w[p] = 1.0 - s;
            w[j] = s;
            break;
        }
    }
}

/// Best single threshold along t that splits samples into w <= 0.5 (before) and w > 0.5 (after).
/// None when one side is nearly empty or the samples are too mixed to call it a cut.
fn step_cut(samples: &mut Vec<(f32, f32)>) -> Option<f32> {
    let n = samples.len();
    if n < 12 {
        return None;
    }
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let hi_total = samples.iter().filter(|s| s.1 > 0.5).count();
    if hi_total < n / 20 || n - hi_total < n / 20 {
        return None;
    }
    // errors(k) = high samples among the first k + low samples among the rest.
    let mut best = usize::MAX;
    let mut ks: Vec<usize> = Vec::new();
    let mut hi_before = 0usize;
    for k in 0..=n {
        if k > 0 && samples[k - 1].1 > 0.5 {
            hi_before += 1;
        }
        let lo_after = (n - k) - (hi_total - hi_before);
        let e = hi_before + lo_after;
        if e < best {
            best = e;
            ks.clear();
        }
        if e == best {
            ks.push(k);
        }
    }
    if best as f32 > 0.2 * n as f32 {
        return None;
    }
    let k = ks[ks.len() / 2].clamp(1, n - 1);
    Some(0.5 * (samples[k - 1].0 + samples[k].0))
}

/// Per place, the reference triangle (index into `tris`) and barycentric weights that match it best
/// in bone space: nearest bone segment of the rest skeleton, then the closest reference triangle
/// among those near that segment or its neighbors, measured with the across-bone distance scaled by
/// `radial`. None where no candidate exists.
fn bone_matched(
    gpos: &[V3],
    gdist: &[f32],
    tris: &[(u32, u32)],
    corners: &dyn Fn(u32) -> [V3; 3],
    corner_weights: &dyn Fn(u32) -> [([u16; 4], [f32; 4]); 3],
    skel: &Skeleton,
    radial: f32,
    align: bool,
    soften: f32,
) -> Vec<Option<(u32, Vec<f32>)>> {
    let nj = skel.names.len();
    // Segment j runs from its parent joint to joint j. Zero-length ones are skipped.
    let seg: Vec<Option<(V3, V3)>> = (0..nj)
        .map(|j| {
            let p = skel.parent[j]?;
            let (a, b) = (skel.head(p), skel.head(j));
            if dist(a, b) > 1e-4 {
                Some((a, b))
            } else {
                None
            }
        })
        .collect();
    let nearest_seg = |p: V3| -> Option<usize> {
        (0..nj).filter(|&j| seg[j].is_some()).min_by(|&x, &y| {
            let (sx, sy) = (seg[x].unwrap(), seg[y].unwrap());
            point_segment(p, sx.0, sx.1).total_cmp(&point_segment(p, sy.0, sy.1))
        })
    };
    let mut tri_seg: Vec<Option<usize>> = Vec::with_capacity(tris.len());
    for i in 0..tris.len() as u32 {
        let c = corners(i);
        let cen = scale(add(add(c[0], c[1]), c[2]), 1.0 / 3.0);
        tri_seg.push(nearest_seg(cen));
    }
    let mut by_seg: Vec<Vec<u32>> = vec![Vec::new(); nj];
    for (i, s) in tri_seg.iter().enumerate() {
        if let Some(s) = s {
            by_seg[*s].push(i as u32);
        }
    }
    // Places grouped by their segment.
    let mut places: Vec<Vec<usize>> = vec![Vec::new(); nj];
    for (g, &p) in gpos.iter().enumerate() {
        if let Some(s) = nearest_seg(p) {
            places[s].push(g);
        }
    }
    // Descendants (self included) of every joint.
    let is_desc = |j: usize, of: usize| -> bool {
        let mut at = Some(j);
        while let Some(k) = at {
            if k == of {
                return true;
            }
            at = skel.parent[k];
        }
        false
    };
    let mut out: Vec<Option<(u32, Vec<f32>)>> = vec![None; gpos.len()];
    for s in 0..nj {
        if places[s].is_empty() {
            continue;
        }
        let (a, b) = seg[s].unwrap();
        let ax = sub(b, a);
        let axl = len(ax);
        let d = scale(ax, 1.0 / axl);
        let to_space = |x: V3| -> V3 {
            let r = sub(x, a);
            let along = dot(r, d);
            let across = sub(r, scale(d, along));
            add(scale(d, along), scale(across, radial))
        };
        // Candidates: triangles near this segment, its parent segment and its child segments.
        let mut cand: Vec<u32> = by_seg[s].clone();
        if let Some(p) = skel.parent[s] {
            cand.extend(&by_seg[p]);
        }
        for k in 0..nj {
            if skel.parent[k] == Some(s) {
                cand.extend(&by_seg[k]);
            }
        }
        let tc: Vec<[V3; 3]> = cand.iter().map(|&i| corners(i).map(to_space)).collect();
        // Where the reference changes bones along this segment (fractions of its length).
        let (mut c_a, mut c_b) = (0.0f32, 1.0f32);
        if align {
            let a_joint = skel.parent[s].unwrap_or(s);
            let mut sa: Vec<(f32, f32)> = Vec::new();
            let mut sb: Vec<(f32, f32)> = Vec::new();
            for &i in &by_seg[s] {
                let c = corners(i);
                let cw = corner_weights(i);
                for k in 0..3 {
                    let t = dot(sub(c[k], a), d) / axl;
                    let (mut wa, mut wb) = (0.0f32, 0.0f32);
                    for q in 0..4 {
                        let (j, w) = (cw[k].0[q] as usize, cw[k].1[q]);
                        if w > 0.0 && j < nj {
                            if is_desc(j, a_joint) {
                                wa += w;
                            }
                            if is_desc(j, s) {
                                wb += w;
                            }
                        }
                    }
                    sa.push((t, wa));
                    sb.push((t, wb));
                }
            }
            if skel.parent[a_joint].is_some() {
                if let Some(c) = step_cut(&mut sa) {
                    if (-0.3..0.7).contains(&c) {
                        c_a = c;
                    }
                }
            }
            if let Some(c) = step_cut(&mut sb) {
                if (0.3..1.2).contains(&c) {
                    c_b = c;
                }
            }
            if c_b - c_a < 0.2 {
                (c_a, c_b) = (0.0, 1.0);
            }
            if std::env::var_os("GAW_WEIGHTS_DEBUG").is_some() {
                eprintln!("segment {:>22} -> {:<22} samples {:5}  cut at start {:.2}, at end {:.2}", skel.names[a_joint], skel.names[s], sa.len(), c_a, c_b);
            }
        }
        let warp = |x: V3| -> V3 {
            // Along-the-bone position remapped so this body's joint planes land on the reference's cuts.
            let r = sub(x, a);
            let along = dot(r, d);
            let across = sub(r, scale(d, along));
            let t = along / axl;
            let t_ref = c_a + t * (c_b - c_a);
            add(scale(d, t_ref * axl), scale(across, radial))
        };
        for &g in &places[s] {
            let p = warp(gpos[g]);
            let hits: Vec<(f32, [f32; 3])> = tc
                .iter()
                .map(|c| {
                    let (cp, bary) = closest_on_tri(p, c[0], c[1], c[2]);
                    (dist(p, cp), bary)
                })
                .collect();
            let Some(bk) = (0..hits.len()).min_by(|&x, &y| hits[x].0.total_cmp(&hits[y].0)) else {
                continue;
            };
            let at = |k: usize| -> Vec<f32> {
                let cw = corner_weights(cand[k]);
                blend3([cw[0].0, cw[1].0, cw[2].0], [cw[0].1, cw[1].1, cw[2].1], hits[k].1, nj)
            };
            let h = soften * gdist[g];
            let mut dense = at(bk);
            if h > 1e-5 {
                let dmin = hits[bk].0;
                let mut acc = vec![0f32; nj];
                let mut total = 0f32;
                for k in 0..hits.len() {
                    let x = (hits[k].0 - dmin) / h;
                    if x < 2.5 {
                        let w = (-x * x).exp();
                        for (a, v) in acc.iter_mut().zip(at(k)) {
                            *a += w * v;
                        }
                        total += w;
                    }
                }
                if total > 0.0 {
                    dense = acc.into_iter().map(|v| v / total).collect();
                }
            }
            out[g] = Some((cand[bk], dense));
        }
    }
    out
}

pub fn transfer(new: &Model, reference: &Model, skel: &Skeleton, opts: &TransferOptions) -> Result<TransferResult, String> {
    let nj = skel.names.len();
    let ref_meshes: Vec<usize> = (0..reference.meshes.len()).filter(|&i| opts.include_battle || !reference.meshes[i].name.starts_with("battle_model")).collect();
    if ref_meshes.is_empty() {
        return Err("the reference has no meshes to take weights from".into());
    }
    let mut tris: Vec<(u32, u32)> = Vec::new();
    for &mi in &ref_meshes {
        for ti in 0..reference.meshes[mi].tris.len() {
            tris.push((mi as u32, ti as u32));
        }
    }
    let corners = |i: u32| -> [V3; 3] {
        let (mi, ti) = tris[i as usize];
        let m = &reference.meshes[mi as usize];
        m.tris[ti as usize].map(|v| m.pos[v as usize])
    };
    let boxes: Vec<(V3, V3)> = (0..tris.len() as u32)
        .map(|i| {
            let c = corners(i);
            (std::array::from_fn(|k| c[0][k].min(c[1][k]).min(c[2][k])), std::array::from_fn(|k| c[0][k].max(c[1][k]).max(c[2][k])))
        })
        .collect();
    let bvh = Bvh::build(&boxes);
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for b in &boxes {
        for k in 0..3 {
            lo[k] = lo[k].min(b.0[k]);
            hi[k] = hi[k].max(b.1[k]);
        }
    }
    let ref_height = (hi[1] - lo[1]).max(1e-6);
    let max_dist = opts.max_dist.unwrap_or(0.05 * ref_height);

    // Weld the new mesh's vertices by position, across meshes.
    let q = (ref_height * 1e-6).max(1e-9);
    let mut map: HashMap<[i64; 3], u32> = HashMap::new();
    let mut gpos: Vec<V3> = Vec::new();
    let mut vgroup: Vec<Vec<u32>> = Vec::new();
    for m in &new.meshes {
        let mut v = Vec::with_capacity(m.pos.len());
        for &p in &m.pos {
            let g = *map.entry(qkey(p, q)).or_insert_with(|| {
                gpos.push(p);
                (gpos.len() - 1) as u32
            });
            v.push(g);
        }
        vgroup.push(v);
    }
    let ng = gpos.len();

    // Closest point on the reference for every welded place.
    let mut dense: Vec<Vec<f32>> = Vec::with_capacity(ng);
    let mut gdist = vec![0f32; ng];
    let mut gtri = vec![(0u32, 0u32); ng];
    for (g, &p) in gpos.iter().enumerate() {
        let (best, _) = bvh
            .nearest(p, |i| {
                let c = corners(i);
                let (cp, _) = closest_on_tri(p, c[0], c[1], c[2]);
                let d = sub(p, cp);
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            })
            .ok_or("the reference surface is empty")?;
        let c = corners(best);
        let (cp, bary) = closest_on_tri(p, c[0], c[1], c[2]);
        let (mi, ti) = tris[best as usize];
        let m = &reference.meshes[mi as usize];
        let t = m.tris[ti as usize];
        let js = t.map(|v| m.joints[v as usize]);
        let ws = t.map(|v| m.weights[v as usize]);
        dense.push(blend3(js, ws, bary, nj));
        gdist[g] = dist(p, cp);
        gtri[g] = (mi, ti);
    }

    // Along-the-bone matching for bodies of other proportions.
    if opts.radial < 0.999 {
        let jw = |i: u32| -> [([u16; 4], [f32; 4]); 3] {
            let (mi, ti) = tris[i as usize];
            let m = &reference.meshes[mi as usize];
            m.tris[ti as usize].map(|v| (m.joints[v as usize], m.weights[v as usize]))
        };
        let rematch = bone_matched(&gpos, &gdist, &tris, &corners, &jw, skel, opts.radial.max(0.01), opts.align_cuts, opts.soften);
        for (g, m) in rematch.into_iter().enumerate() {
            if let Some((best, d)) = m {
                dense[g] = d;
                gtri[g] = tris[best as usize];
            }
        }
    }

    if opts.joint_blend > 0.0 {
        joint_blend(&mut dense, &gpos, skel, opts.joint_blend);
    }

    // Bones the new model may not use: weight moves up to the nearest allowed ancestor.
    if let Some(allowed) = &opts.allowed {
        if allowed.len() != nj {
            return Err("the allowed-bone list does not match the skeleton".into());
        }
        let to = redirect_table(skel, allowed)?;
        for d in dense.iter_mut() {
            for j in 0..nj {
                if to[j] != j && d[j] > 0.0 {
                    let w = d[j];
                    d[j] = 0.0;
                    d[to[j]] += w;
                }
            }
        }
    }

    // Regions follow each place's dominant bone.
    let mut regions: Vec<String> = Vec::new();
    let joint_region: Vec<u8> = skel
        .names
        .iter()
        .map(|n| {
            let r = region_of_bone(n);
            match regions.iter().position(|x| *x == r) {
                Some(i) => i as u8,
                None => {
                    regions.push(r);
                    (regions.len() - 1) as u8
                }
            }
        })
        .collect();
    let gregion: Vec<u8> = dense.iter().map(|d| joint_region[(0..nj).max_by(|&a, &b| d[a].total_cmp(&d[b])).unwrap_or(0)]).collect();

    // Welded adjacency from the new triangles.
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); ng];
    for (mi, m) in new.meshes.iter().enumerate() {
        for t in &m.tris {
            let g = t.map(|v| vgroup[mi][v as usize]);
            for k in 0..3 {
                let (a, b) = (g[k], g[(k + 1) % 3]);
                if a != b {
                    adj[a as usize].push(b);
                    adj[b as usize].push(a);
                }
            }
        }
    }
    for a in adj.iter_mut() {
        a.sort_unstable();
        a.dedup();
    }
    for _ in 0..opts.rings {
        dense = smooth_ring(&dense, &adj, &gregion, opts.lambda);
    }

    let g4: Vec<([u16; 4], [f32; 4])> = dense.iter().map(|d| top4(d)).collect();
    let mut out = TransferResult {
        joints: Vec::new(),
        weights: Vec::new(),
        dist: Vec::new(),
        ref_tri: Vec::new(),
        region: Vec::new(),
        regions,
        max_dist,
        ref_height,
        welded_groups: ng,
        ref_meshes: ref_meshes.iter().map(|&i| reference.meshes[i].name.clone()).collect(),
    };
    for v in &vgroup {
        out.joints.push(v.iter().map(|&g| g4[g as usize].0).collect());
        out.weights.push(v.iter().map(|&g| g4[g as usize].1).collect());
        out.dist.push(v.iter().map(|&g| gdist[g as usize]).collect());
        out.ref_tri.push(v.iter().map(|&g| gtri[g as usize]).collect());
        out.region.push(v.iter().map(|&g| gregion[g as usize]).collect());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------------------------

fn push_view(json: &mut Value, bin: &mut Vec<u8>, bytes: &[u8]) -> usize {
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let off = bin.len();
    bin.extend_from_slice(bytes);
    let views = json["bufferViews"].as_array_mut().expect("bufferViews");
    views.push(json!({ "buffer": 0, "byteOffset": off, "byteLength": bytes.len(), "target": 34962 }));
    views.len() - 1
}

fn push_accessor(json: &mut Value, a: Value) -> usize {
    let list = json["accessors"].as_array_mut().expect("accessors");
    list.push(a);
    list.len() - 1
}

/// Writes `new` with these joints and weights (one list per mesh, as the model reads them): every
/// primitive gets JOINTS_0 (u16) and WEIGHTS_0 (f32), extra influence sets are removed, mesh nodes
/// without a skin point at skin 0. Everything else keeps its bytes. Reads the result back and checks it.
pub fn write_weights(new: &Model, joints: &[Vec<[u16; 4]>], weights: &[Vec<[f32; 4]>]) -> Result<Vec<u8>, String> {
    let glb = &new.glb;
    if glb.list("skins").is_empty() {
        return Err("the file has no skin".into());
    }
    let nj = new.joint_names.len();
    let mut json = glb.json.clone();
    let mut bin = glb.bin().to_vec();
    let mut mi = 0usize;
    let nmesh = json["meshes"].as_array().map(|a| a.len()).unwrap_or(0);
    let mut skinned_meshes: Vec<usize> = Vec::new();
    for j in 0..nmesh {
        let nprim = json["meshes"][j]["primitives"].as_array().map(|a| a.len()).unwrap_or(0);
        for p in 0..nprim {
            let (jn, wt) = (joints.get(mi).ok_or("weights are missing for a mesh")?, weights.get(mi).ok_or("weights are missing for a mesh")?);
            let m = &new.meshes[mi];
            if jn.len() != m.pos.len() || wt.len() != m.pos.len() {
                return Err(format!("mesh {}: {} joints/weights for {} vertices", m.name, jn.len(), m.pos.len()));
            }
            mi += 1;
            let jb: Vec<u8> = jn.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
            let wb: Vec<u8> = wt.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
            let jv = push_view(&mut json, &mut bin, &jb);
            let ja = push_accessor(&mut json, json!({"bufferView": jv, "componentType": 5123, "count": jn.len(), "type": "VEC4"}));
            let wv = push_view(&mut json, &mut bin, &wb);
            let wa = push_accessor(&mut json, json!({"bufferView": wv, "componentType": 5126, "count": wt.len(), "type": "VEC4"}));
            let attrs = json["meshes"][j]["primitives"][p]["attributes"].as_object_mut().ok_or("primitive without attributes")?;
            let extra: Vec<String> = attrs.keys().filter(|k| (k.starts_with("JOINTS_") || k.starts_with("WEIGHTS_")) && !k.ends_with("_0")).cloned().collect();
            for k in extra {
                attrs.remove(&k);
            }
            attrs.insert("JOINTS_0".into(), json!(ja));
            attrs.insert("WEIGHTS_0".into(), json!(wa));
        }
        skinned_meshes.push(j);
    }
    if let Some(nodes) = json["nodes"].as_array_mut() {
        for n in nodes.iter_mut() {
            if let Some(m) = n.get("mesh").and_then(Value::as_u64) {
                if skinned_meshes.contains(&(m as usize)) && n.get("skin").is_none() {
                    n["skin"] = json!(0);
                }
            }
        }
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
    chunks.extend(glb.chunks.iter().skip(1).filter(|c| c.kind != CHUNK_BIN).map(|c| Chunk { kind: c.kind, data: c.data.clone() }));
    let out = Glb { version: glb.version, chunks, json: Value::Null }.to_bytes();

    // Read back: same geometry, the weights we asked for, a valid skin.
    let back = Model::from_bytes(&out)?;
    if back.meshes.len() != new.meshes.len() || back.joint_names != new.joint_names {
        return Err("written file changed the meshes or the joint list".into());
    }
    for (mi, (a, b)) in new.meshes.iter().zip(&back.meshes).enumerate() {
        if a.pos != b.pos || a.tris != b.tris || a.uv != b.uv {
            return Err(format!("written mesh {} changed geometry", a.name));
        }
        for v in 0..b.pos.len() {
            let sum: f32 = b.weights[v].iter().sum();
            let n = b.weights[v].iter().filter(|&&w| w > 0.0).count();
            if (sum - 1.0).abs() > 1e-5 || n > 4 || b.joints[v].iter().zip(&b.weights[v]).any(|(&j, &w)| w > 0.0 && j as usize >= nj) {
                return Err(format!("written mesh {} vertex {v}: weights sum {sum}, {n} influences", b.name));
            }
            if b.joints[v] != joints[mi][v] || b.weights[v] != weights[mi][v] {
                return Err(format!("written mesh {} vertex {v} does not read back", b.name));
            }
        }
    }
    Ok(out)
}

/// Mesh nodes whose own transform is not identity (glTF ignores it for skinned meshes, the
/// workbench's readers too): names, for a warning.
pub fn moved_mesh_nodes(glb: &Glb) -> Vec<String> {
    let ident = [0.0, 0.0, 0.0, 1.0];
    glb.list("nodes")
        .iter()
        .filter(|n| n.get("mesh").is_some())
        .filter(|n| {
            let f = |k: &str, d: &[f64]| n.get(k).and_then(Value::as_array).map(|a| a.iter().zip(d).any(|(x, &y)| (x.as_f64().unwrap_or(y) - y).abs() > 1e-6)).unwrap_or(false);
            f("translation", &[0.0; 3]) || f("rotation", &ident) || f("scale", &[1.0; 3]) || n.get("matrix").is_some()
        })
        .map(|n| n.get("name").and_then(Value::as_str).unwrap_or("?").to_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top4_keeps_largest_and_sums_to_one() {
        let d = vec![0.1, 0.3, 0.05, 0.2, 0.15, 0.2, 0.0];
        let (j, w) = top4(&d);
        assert_eq!(w.iter().filter(|&&x| x > 0.0).count(), 4);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(j[0], 1);
        assert!(w[0] >= w[1] && w[1] >= w[2] && w[2] >= w[3]);
        // the two smallest (0.05 and 0.1) and the 0.0 are gone
        assert!(!j.iter().zip(&w).any(|(&jj, &ww)| ww > 0.0 && (jj == 2 || jj == 6)));
    }

    #[test]
    fn blend3_is_the_barycentric_mix() {
        let js = [[0, 1, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]];
        let ws = [[0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
        let d = blend3(js, ws, [0.5, 0.25, 0.25], 3);
        assert!((d[0] - 0.25).abs() < 1e-6 && (d[1] - (0.25 + 0.25)).abs() < 1e-6 && (d[2] - 0.25).abs() < 1e-6);
        assert!((d.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn smoothing_never_mixes_regions() {
        // Path of 5 places: 0 1 2 | 3 4, region A then B, A fully bone 0, B fully bone 1.
        let cur = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        let adj: Vec<Vec<u32>> = vec![vec![1], vec![0, 2], vec![1, 3], vec![2, 4], vec![3]];
        let region = [0u8, 0, 0, 1, 1];
        let mut v = cur.clone();
        for _ in 0..10 {
            v = smooth_ring(&v, &adj, &region, 0.5);
        }
        assert_eq!(v, cur, "places on both sides of the region seam keep their weights");
        // Inside one region a spike spreads out and the sum stays 1.
        let cur = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]];
        let adj: Vec<Vec<u32>> = vec![vec![1], vec![0, 2], vec![1]];
        let v = smooth_ring(&cur, &adj, &[0, 0, 0], 0.5);
        assert!((v[1][0] - 0.5).abs() < 1e-6 && (v[1][0] + v[1][1] - 1.0).abs() < 1e-6);
    }

    fn skel(names: &[&str], parents: &[Option<usize>]) -> Skeleton {
        let id: Mat4 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        Skeleton { names: names.iter().map(|s| s.to_string()).collect(), parent: parents.to_vec(), world: vec![id; names.len()], ibm: vec![id; names.len()] }
    }

    #[test]
    fn disallowed_bones_hand_weight_to_the_nearest_allowed_parent() {
        let s = skel(&["root", "tail_a", "tail_b"], &[None, Some(0), Some(1)]);
        let t = redirect_table(&s, &[true, false, false]).unwrap();
        assert_eq!(t, vec![0, 0, 0]);
        let t = redirect_table(&s, &[true, true, false]).unwrap();
        assert_eq!(t, vec![0, 1, 1]);
        assert!(redirect_table(&s, &[false, true, true]).is_err(), "the root is not allowed and has no parent");
    }

    #[test]
    fn skeleton_diff_names_the_problem() {
        let a = skel(&["root", "spine", "head"], &[None, Some(0), Some(1)]);
        let same = skel(&["root", "spine", "head"], &[None, Some(0), Some(1)]);
        assert!(skeleton_diff(&same, &a, 1e-3).is_empty());
        let order = skel(&["root", "head", "spine"], &[None, Some(0), Some(1)]);
        let p = skeleton_diff(&order, &a, 1e-3);
        assert!(p.iter().any(|m| m.contains("different order")), "{p:?}");
        let parent = skel(&["root", "spine", "head"], &[None, Some(0), Some(0)]);
        assert!(skeleton_diff(&parent, &a, 1e-3).iter().any(|m| m.contains("has parent")));
        let mut moved = skel(&["root", "spine", "head"], &[None, Some(0), Some(1)]);
        moved.world[2][13] = 0.5;
        assert!(skeleton_diff(&moved, &a, 1e-3).iter().any(|m| m.contains("rest pose differs")));
        let fewer = skel(&["root", "spine"], &[None, Some(0)]);
        assert!(skeleton_diff(&fewer, &a, 1e-3)[0].contains("2 joints"));
    }

    #[test]
    fn bone_lists_take_names_numbers_and_fragments() {
        let names: Vec<String> = ["bone000_root", "bone023_tail_base", "bone024_tail_01", "bone025_tail_02"].iter().map(|s| s.to_string()).collect();
        assert_eq!(parse_bone_list("bone000_root,2", &names).unwrap(), vec![true, false, true, false]);
        assert_eq!(parse_bone_list("tail", &names).unwrap(), vec![false, true, true, true]);
        assert!(parse_bone_list("nope", &names).is_err());
        assert!(parse_bone_list("9", &names).is_err());
    }

    #[test]
    fn regions_of_the_ff9_bones() {
        assert_eq!(region_of_bone("bone004_L_forearm"), "L_arm");
        assert_eq!(region_of_bone("bone009_R_shoulder"), "R_arm");
        assert_eq!(region_of_bone("bone017_L_shin"), "L_leg");
        assert_eq!(region_of_bone("bone014_hips"), "torso");
        assert_eq!(region_of_bone("bone015_L_hip"), "L_leg");
        assert_eq!(region_of_bone("bone025_tail_02"), "tail");
        assert_eq!(region_of_bone("bone007_neck"), "head");
        assert_eq!(region_of_bone("bone002_chest"), "torso");
    }
}
