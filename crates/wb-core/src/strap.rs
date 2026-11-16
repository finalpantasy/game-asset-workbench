//! Rebuilds a hanging strap card as a thin solid strap.
//!
//! The stock strap is a flat two-sided card lying over the belt: rigid on the hips, while the waist
//! under it is skinned, so the two slide apart in motion and the paint that crosses the card's edge
//! steps. The rebuild keeps the card's front surface, UVs and paint layout, and:
//! - gives it a back face and side walls (a closed strap `thickness` thick);
//! - follows the painted silhouette (alpha) at the tip, so the walls stay inside the rounded paint;
//! - tucks the top of the strap `tuck_depth` behind the waist wherever the waist paint under it is
//!   leather (the band and the painted knot fold), and brings it out in front exactly at that
//!   paint edge, so the band and knot paint occlude the strap with a real edge;
//! - skins the upper part (`ride` down from the top) with the waist's own weights under each vertex,
//!   so it rides the band instead of sliding over it, then blends to the tip weights
//!   (hips, with `tip_thigh` of the weight on the thigh joint).
//!   With `fit_frames` > 0 the ride part's weights are fitted by least squares so each vertex follows
//!   its waist triangle through sampled frames of every clip: blending the triangle's weights alone
//!   does not, because linear blend skinning of a point is not the blend of its corners' motion.
//! - with `collide`, adds the smallest left and right thigh shares per row (rising toward the tip)
//!   that keep the strap out of the thighs in every sampled frame.
//! Every other vertex and triangle of the model is copied unchanged.
use crate::geom::{self, V3};
use crate::glb::{Chunk, CHUNK_BIN, CHUNK_JSON};
use crate::model::Mesh;
use crate::shells::{self, Shell};
use crate::{Model, Texture};
use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub struct Params {
    /// Strap thickness in world units.
    pub thickness: f32,
    /// How far the top edge sits behind the waist surface.
    pub tuck_depth: f32,
    /// Smallest gap kept between the strap front and the waist below the tuck (thickness plus the
    /// back face's own clearance, so the back face stays out of the waist too).
    pub clear_gap: f32,
    /// Length above the paint exit over which the hidden strap rises from tuck_depth behind the
    /// waist to the waist surface.
    pub tuck_len: f32,
    /// Length below the paint exit over which the visible strap leaves the waist surface at a
    /// shallow angle and eases out to where the card hangs (or clear_gap, if that is further).
    pub lift_len: f32,
    /// Row spacing along the strap.
    pub row_step: f32,
    /// Length from the top that rides the waist with the waist's own weights.
    pub ride: f32,
    /// Share of the tip's weight on `thigh_joint` (the rest on `hips_joint`).
    pub tip_thigh: f32,
    pub hips_joint: usize,
    pub thigh_joint: usize,
    /// Points across the strap per row, minus one. 12 lets the exit follow a curved paint edge
    /// (at 6, the straight span between vertices left a sliver of cloth at the knot's corner).
    pub columns: usize,
    /// How far inside the edge the side walls sample the paint, as a share of the strap's width.
    pub wall_inset: f32,
    /// Frames per clip for the ride-part weight fit (0 = blend the waist triangle's weights).
    pub fit_frames: usize,
    /// Joints of waist vertices this close to a point are candidates in the fit.
    pub fit_reach: f32,
    /// Tuck the strap behind the waist down to where the waist paint stops being leather.
    pub exit_by_paint: bool,
    /// Longest tuck allowed when following the paint.
    pub exit_max: f32,
    /// Solve thigh shares per row so the strap stays out of the thighs.
    pub collide: bool,
    pub left_thigh_joint: usize,
    /// Most a row may stand further off the body when thigh shares cannot clear it.
    pub max_lift: f32,
    /// Largest thigh share the clearance solver may give a row. 0 keeps the strap on the waist's
    /// own weights below the knot (it hangs like the stock strap) and only lifts rows off the body:
    /// thigh shares make the strap twist or fold when a knee comes up (linear skinning can only
    /// average the two thighs, and they swing opposite ways).
    pub max_thigh_share: f32,
    /// Take the card's lowest end as the root instead of its highest (when the inference is wrong).
    pub flip_root: bool,
    /// Waist vertices (in order) along the edge the strap comes out from under. Empty: follow the
    /// paint behind the strap.
    pub exit_edge: Vec<usize>,
    /// Points (rest pose, model space) along the edge the strap comes out from under, for a waist
    /// whose edge is only painted (no vertices there). Every run logs its own contour in this form,
    /// so a contour found on one file can be passed to another with the same geometry.
    pub exit_curve: Vec<[f32; 3]>,
    /// The exit as distances down the strap, evenly spaced across it from the first root edge vertex
    /// to the second (the order the log prints them), for a painted edge no waist geometry follows.
    pub exit_down: Vec<f32>,
    /// Moves the exit this far up the strap (any exit source), so a small skinning miss along
    /// the waist normal in motion does not open a sliver of cloth between the painted edge and
    /// the strap.
    pub exit_raise: f32,
}

/// How much more a miss along the waist normal counts than a slide along it in the weight fit.
const FIT_NORMAL_WEIGHT: f32 = 3.0;

/// The thigh solver counts a point as a hit when it is within this of being inside the body.
const COLLIDE_MARGIN: f32 = 0.002;

impl Default for Params {
    fn default() -> Params {
        Params {
            thickness: 0.006,
            tuck_depth: 0.010,
            clear_gap: 0.009,
            tuck_len: 0.024,
            lift_len: 0.08,
            row_step: 0.04,
            ride: 0.26,
            tip_thigh: 0.0,
            hips_joint: 14,
            thigh_joint: 20,
            columns: 12,
            wall_inset: 0.33,
            fit_frames: 24,
            fit_reach: 0.10,
            exit_by_paint: true,
            exit_max: 0.15,
            collide: true,
            left_thigh_joint: 16,
            max_lift: 0.012,
            max_thigh_share: 0.0,
            flip_root: false,
            exit_edge: Vec::new(),
            exit_curve: Vec::new(),
            exit_down: Vec::new(),
            exit_raise: 0.0,
        }
    }
}

/// One output vertex of the rebuilt mesh: a copy of a source vertex or a new one.
#[derive(Clone)]
pub enum NewVert {
    Old(u32),
    /// `src`: the card triangle (source vertex indices) and barycentric weights under the vertex,
    /// for attributes the rebuild does not set itself (extra UV sets, colors).
    New { pos: V3, normal: V3, uv: [f32; 2], joints: [u16; 4], weights: [f32; 4], src: ([u32; 3], [f32; 3]) },
}

/// What the rebuild made, for the report.
pub struct Built {
    pub verts: Vec<NewVert>,
    pub tris: Vec<[u32; 3]>,
    /// Range of the strap's triangles in `tris` (they come first).
    pub strap_tris: usize,
    pub strap_verts: usize,
    pub rows: usize,
    pub length: f32,
    pub width: f32,
    pub tip_a: f32,
    pub log: Vec<String>,
    /// Per strap vertex (index into the new vertex list): distance down the strap and which face.
    pub meta: Vec<(u32, f32, &'static str)>,
}

/// Vertex normals of a mesh read from its NORMAL attribute (None if it has none).
pub fn normals(model: &Model, mesh: usize) -> Option<Vec<V3>> {
    let (mi, pi) = prim_of(model, mesh)?;
    let a = model.glb.json["meshes"][mi]["primitives"][pi]["attributes"]["NORMAL"].as_u64()? as usize;
    let acc = model.glb.accessor(a).ok()?;
    Some((0..acc.count).map(|i| [acc.get(i, 0) as f32, acc.get(i, 1) as f32, acc.get(i, 2) as f32]).collect())
}

/// (glTF mesh, primitive) of `Model::meshes[k]`.
pub fn prim_of(model: &Model, k: usize) -> Option<(usize, usize)> {
    let mut n = 0;
    for (mi, m) in model.glb.list("meshes").iter().enumerate() {
        for pi in 0..m.get("primitives").and_then(Value::as_array).map(|p| p.len()).unwrap_or(0) {
            if n == k {
                return Some((mi, pi));
            }
            n += 1;
        }
    }
    None
}

/// The card's front triangles as a map from local strap coordinates to the card surface. Local
/// (u, d): u along the root edge, d down the card's long axis, both in UV units; `atlas` turns
/// them into texture UV, so any chart rotation or mirroring works.
struct CardMap {
    tris: Vec<[usize; 3]>,
    origin: [f32; 2],
    across: [f32; 2],
    down: [f32; 2],
}

impl CardMap {
    fn atlas(&self, p: [f32; 2]) -> [f32; 2] {
        std::array::from_fn(|i| self.origin[i] + self.across[i] * p[0] + self.down[i] * p[1])
    }
    fn bary(m: &Mesh, t: [usize; 3], uv: [f32; 2]) -> [f32; 3] {
        let [a, b, c] = t.map(|v| m.uv[v]);
        let den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
        if den.abs() < 1e-14 {
            return [-1.0, -1.0, -1.0];
        }
        let l0 = ((b[1] - c[1]) * (uv[0] - c[0]) + (c[0] - b[0]) * (uv[1] - c[1])) / den;
        let l1 = ((c[1] - a[1]) * (uv[0] - c[0]) + (a[0] - c[0]) * (uv[1] - c[1])) / den;
        [l0, l1, 1.0 - l0 - l1]
    }
    /// The front triangle that holds `uv` best (largest smallest barycentric) and its weights.
    fn locate(&self, m: &Mesh, uv: [f32; 2]) -> ([usize; 3], [f32; 3], f32) {
        let uv = self.atlas(uv);
        let mut best = (self.tris[0], [1.0, 0.0, 0.0], f32::MIN);
        for &t in &self.tris {
            let b = CardMap::bary(m, t, uv);
            let lo = b[0].min(b[1]).min(b[2]);
            if lo > best.2 {
                best = (t, b, lo);
            }
        }
        best
    }
    fn inside(&self, m: &Mesh, uv: [f32; 2]) -> bool {
        self.locate(m, uv).2 >= -1e-5
    }
    fn pos(&self, m: &Mesh, uv: [f32; 2]) -> V3 {
        let (t, b, _) = self.locate(m, uv);
        let p = t.map(|v| m.pos[v]);
        std::array::from_fn(|k| b[0] * p[0][k] + b[1] * p[1][k] + b[2] * p[2][k])
    }
}

fn opaque(tex: &Texture, uv: [f32; 2]) -> bool {
    let x = ((uv[0] * tex.w as f32) as i64).clamp(0, tex.w as i64 - 1) as usize;
    let y = ((uv[1] * tex.h as f32) as i64).clamp(0, tex.h as i64 - 1) as usize;
    tex.rgba[(y * tex.w + x) * 4 + 3] >= 128
}

/// Column-major 4x4 times a point.
fn xform(m: &crate::anim::Mat4, p: V3) -> V3 {
    [
        m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
        m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
        m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    ]
}

/// Solves x*e1 + y*e2 + z*n = d.
fn local_coords(e1: V3, e2: V3, n: V3, d: V3) -> V3 {
    let det = geom::dot(e1, geom::cross(e2, n));
    if det.abs() < 1e-14 {
        return [0.0, 0.0, geom::dot(d, n)];
    }
    [geom::dot(d, geom::cross(e2, n)) / det, geom::dot(e1, geom::cross(d, n)) / det, geom::dot(e1, geom::cross(e2, d)) / det]
}

fn tri_frame(p: [V3; 3]) -> (V3, V3, V3) {
    let e1 = geom::sub(p[1], p[0]);
    let e2 = geom::sub(p[2], p[0]);
    (e1, e2, geom::norm(geom::cross(e1, e2)))
}

/// Euclidean projection onto the probability simplex.
fn project_simplex(v: &mut [f64]) {
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| b.total_cmp(a));
    let (mut css, mut theta) = (0.0, 0.0);
    for (i, &x) in u.iter().enumerate() {
        css += x;
        let t = (css - 1.0) / (i + 1) as f64;
        if x - t > 0.0 {
            theta = t;
        }
    }
    for x in v.iter_mut() {
        *x = (*x - theta).max(0.0);
    }
}

/// Least squares over the simplex: min |A w - y|^2, w >= 0, sum w = 1 (A as rows of length n).
fn simplex_ls(rows: &[Vec<f64>], y: &[f64], n: usize, start: &[f64]) -> Vec<f64> {
    let mut g = vec![0.0f64; n * n];
    let mut h = vec![0.0f64; n];
    for (r, &yy) in rows.iter().zip(y) {
        for i in 0..n {
            h[i] += r[i] * yy;
            for j in 0..n {
                g[i * n + j] += r[i] * r[j];
            }
        }
    }
    // Largest eigenvalue of G by power iteration, for the step size.
    let mut v = vec![1.0f64; n];
    let mut lam = 1.0;
    for _ in 0..50 {
        let gv: Vec<f64> = (0..n).map(|i| (0..n).map(|j| g[i * n + j] * v[j]).sum()).collect();
        lam = gv.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        v = gv.iter().map(|x| x / lam).collect();
    }
    let step = 1.0 / lam;
    let mut w = start.to_vec();
    project_simplex(&mut w);
    for _ in 0..4000 {
        let grad: Vec<f64> = (0..n).map(|i| (0..n).map(|j| g[i * n + j] * w[j]).sum::<f64>() - h[i]).collect();
        for i in 0..n {
            w[i] -= step * grad[i];
        }
        project_simplex(&mut w);
    }
    w
}

fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Sums weighted weight sets and keeps the four largest, normalized.
fn blend(parts: &[(&[(usize, f32)], f32)]) -> ([u16; 4], [f32; 4]) {
    let mut acc: Vec<(usize, f32)> = Vec::new();
    for (set, f) in parts {
        if *f <= 0.0 {
            continue;
        }
        for &(j, w) in set.iter() {
            match acc.iter_mut().find(|e| e.0 == j) {
                Some(e) => e.1 += w * f,
                None => acc.push((j, w * f)),
            }
        }
    }
    acc.sort_by(|a, b| b.1.total_cmp(&a.1));
    acc.truncate(4);
    acc.retain(|e| e.1 > 1e-4);
    let total: f32 = acc.iter().map(|e| e.1).sum::<f32>().max(1e-9);
    let mut j = [0u16; 4];
    let mut w = [0f32; 4];
    for (k, e) in acc.iter().enumerate() {
        j[k] = e.0 as u16;
        w[k] = e.1 / total;
    }
    (j, w)
}

/// Builds the new mesh for `mesh` with the card shell replaced by a solid strap.
pub fn build(model: &Model, card: &Shell, body: &Shell, p: &Params) -> Result<Built, String> {
    if card.mesh != body.mesh {
        return Err("the card and the surface it hangs on must be in the same mesh".into());
    }
    let mk = card.mesh;
    let m = &model.meshes[mk];
    let mut log = Vec::new();
    let attr_n = normals(model, mk).ok_or("mesh has no NORMAL attribute")?;
    let tex = m.image.and_then(|i| Texture::decode(&model.images[i].bytes).ok()).ok_or("the card's mesh has no decodable texture")?;
    let rest: Vec<Vec<V3>> = model.meshes.iter().map(|m| m.pos.clone()).collect();
    // Body surface point and outward normal (from the NORMAL attribute) closest to a point.
    let body_at = |q: V3| -> (V3, V3, Vec<(usize, f32)>) {
        let (_, t, b) = shells::closest_in_shell(model, &rest, body, q);
        let c = m.tris[t].map(|v| v as usize);
        let pt: V3 = std::array::from_fn(|k| b[0] * m.pos[c[0]][k] + b[1] * m.pos[c[1]][k] + b[2] * m.pos[c[2]][k]);
        let n = geom::norm(std::array::from_fn(|k| b[0] * attr_n[c[0]][k] + b[1] * attr_n[c[1]][k] + b[2] * attr_n[c[2]][k]));
        (pt, n, shells::weights_at(m, t, b))
    };
    // Front faces: the half of the card whose normals point away from the body.
    let mut front = Vec::new();
    for &t in &card.tris {
        let c = m.tris[t].map(|v| v as usize);
        let n = geom::norm(geom::add(geom::add(attr_n[c[0]], attr_n[c[1]]), attr_n[c[2]]));
        let ctr = geom::scale(geom::add(geom::add(m.pos[c[0]], m.pos[c[1]]), m.pos[c[2]]), 1.0 / 3.0);
        let (q, _, _) = body_at(ctr);
        if geom::dot(n, geom::sub(ctr, q)) > 0.0 {
            front.push(c);
        }
    }
    if front.is_empty() || front.len() * 2 != card.tris.len() {
        log.push(format!("note: {} of {} card triangles face away from the body (expected half)", front.len(), card.tris.len()));
    }
    if front.is_empty() {
        return Err("no card triangle faces away from the body".into());
    }
    // Root and long axis from geometry, not from the UV layout: the root is the end of the card that
    // sits highest at rest (the model is +Y up; flip_root takes the other end). The root edge is the
    // widest UV pair at that end, ordered by world X; the long axis is the principal axis of the
    // card's UVs, pointing away from the root.
    let fv: Vec<usize> = {
        let mut v: Vec<usize> = front.iter().flatten().copied().collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let (ymin, ymax) = fv.iter().fold((f32::MAX, f32::MIN), |acc, &v| (acc.0.min(m.pos[v][1]), acc.1.max(m.pos[v][1])));
    let end_of = |top: bool| -> Vec<usize> {
        let span = (ymax - ymin) * 0.1;
        fv.iter().copied().filter(|&v| if top { m.pos[v][1] >= ymax - span } else { m.pos[v][1] <= ymin + span }).collect()
    };
    let (root_end, other_end) = (end_of(!p.flip_root), end_of(p.flip_root));
    let uvd = |a: usize, b: usize| ((m.uv[a][0] - m.uv[b][0]).powi(2) + (m.uv[a][1] - m.uv[b][1]).powi(2)).sqrt();
    let mut pair = (root_end[0], root_end[0], 0.0f32);
    for &a in &root_end {
        for &b in &root_end {
            if uvd(a, b) > pair.2 {
                pair = (a, b, uvd(a, b));
            }
        }
    }
    if pair.2 < 1e-6 {
        return Err(format!("the card's {} end has no edge in UV (try --flip-root)", if p.flip_root { "lowest" } else { "highest" }));
    }
    let (ra, rb) = if m.pos[pair.0][0] <= m.pos[pair.1][0] { (pair.0, pair.1) } else { (pair.1, pair.0) };
    let across = [(m.uv[rb][0] - m.uv[ra][0]) / pair.2, (m.uv[rb][1] - m.uv[ra][1]) / pair.2];
    let mean_uv: [f32; 2] = std::array::from_fn(|k| fv.iter().map(|&v| m.uv[v][k]).sum::<f32>() / fv.len() as f32);
    let (mut sxx, mut sxy, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for &v in &fv {
        let (x, y) = (m.uv[v][0] - mean_uv[0], m.uv[v][1] - mean_uv[1]);
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
    }
    let ang = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let mut down = [ang.cos(), ang.sin()];
    let root_mid = [0.5 * (m.uv[ra][0] + m.uv[rb][0]), 0.5 * (m.uv[ra][1] + m.uv[rb][1])];
    if (mean_uv[0] - root_mid[0]) * down[0] + (mean_uv[1] - root_mid[1]) * down[1] < 0.0 {
        down = [-down[0], -down[1]];
    }
    if (across[0] * down[1] - across[1] * down[0]).abs() < 0.3 {
        return Err("the card's root edge runs along its long axis; cannot set up the strap frame".into());
    }
    // Mirrored chart: seen from the front (outward normal toward the viewer), an unmirrored texture
    // winds the other way in UV (v points down the image) than in world space.
    let n_out = geom::norm(front.iter().flatten().fold([0.0f32; 3], |acc, &v| geom::add(acc, attr_n[v])));
    let mirrored = front
        .iter()
        .map(|t| {
            let (p0, p1, p2) = (m.pos[t[0]], m.pos[t[1]], m.pos[t[2]]);
            let sw = geom::dot(geom::cross(geom::sub(p1, p0), geom::sub(p2, p0)), n_out);
            let (a, b, c) = (m.uv[t[0]], m.uv[t[1]], m.uv[t[2]]);
            let suv = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            (sw * suv).signum()
        })
        .sum::<f32>()
        > 0.0;
    // Sanity: the root end should be the end nearer the waist surface.
    let end_gap = |vs: &[usize]| vs.iter().map(|&v| geom::dist(m.pos[v], body_at(m.pos[v]).0)).sum::<f32>() / vs.len().max(1) as f32;
    let (root_gap, other_gap) = (end_gap(&root_end), end_gap(&other_end));
    log.push(format!(
        "root: card {} end (y {:.3}), edge vertices {}..{}, UV across ({:.3}, {:.3}), down ({:.3}, {:.3}){}; root end {:.4} from the waist, other end {:.4}",
        if p.flip_root { "lowest" } else { "highest" },
        if p.flip_root { ymin } else { ymax },
        ra,
        rb,
        across[0],
        across[1],
        down[0],
        down[1],
        if mirrored { ", texture mirrored as seen from the front" } else { "" },
        root_gap,
        other_gap
    ));
    if root_gap > other_gap + 0.01 {
        log.push("WARNING: the root end is farther from the waist than the other end; check the result or pass --flip-root".into());
    }
    let map = CardMap { tris: front.clone(), origin: m.uv[ra], across, down };
    let (ua, ub) = (0.0f32, pair.2);
    let v_top = |_u: f32| 0.0f32;
    let texel = 1.0 / tex.w.max(tex.h) as f32;
    let step = texel * 0.125;
    // Row d (UV v below the top line): opaque extent across, scanning out from `uc`.
    let extent = |d: f32, uc: f32| -> Option<(f32, f32)> {
        let at = |u: f32| [u, v_top(u) + d];
        if !(map.inside(m, at(uc)) && opaque(&tex, map.atlas(at(uc)))) {
            return None;
        }
        let mut lo = uc;
        while map.inside(m, at(lo - step)) && opaque(&tex, map.atlas(at(lo - step))) {
            lo -= step;
        }
        let mut hi = uc;
        while map.inside(m, at(hi + step)) && opaque(&tex, map.atlas(at(hi + step))) {
            hi += step;
        }
        Some((lo, hi))
    };
    // Length of the card along its middle, and the opaque tip.
    let uc0 = 0.5 * (ua + ub);
    let mut d_end = 0.0f32;
    while extent(d_end + step, uc0).is_some() {
        d_end += step;
        if d_end > 1.0 {
            return Err("card length scan ran away".into());
        }
    }
    // Middle of the strap just above the tip zone (the tip cap may not be centered on the top).
    let (l8, h8) = extent(0.8 * d_end, uc0).ok_or("no opaque paint at 80% of the strap")?;
    let uc = 0.5 * (l8 + h8);
    while extent(d_end + step, uc).is_some() {
        d_end += step;
    }
    let full_w = h8 - l8;
    // World distance down the strap for a UV offset d, along the middle.
    let samples = 400usize;
    let mut arc = vec![0.0f32; samples + 1];
    let mid = |d: f32| map.pos(m, [uc, v_top(uc) + d]);
    for i in 1..=samples {
        let (d0, d1) = (d_end * (i - 1) as f32 / samples as f32, d_end * i as f32 / samples as f32);
        arc[i] = arc[i - 1] + geom::dist(mid(d0), mid(d1));
    }
    let length = arc[samples];
    let d_of_a = |a: f32| -> f32 {
        let a = a.clamp(0.0, length);
        let i = arc.partition_point(|&x| x < a).clamp(1, samples);
        let f = (a - arc[i - 1]) / (arc[i] - arc[i - 1]).max(1e-9);
        d_end * ((i - 1) as f32 + f) / samples as f32
    };
    let a_of_d = |d: f32| -> f32 {
        let x = (d / d_end).clamp(0.0, 1.0) * samples as f32;
        let i = (x.floor() as usize).min(samples - 1);
        arc[i] + (arc[i + 1] - arc[i]) * (x - i as f32)
    };
    // Where the tip paint starts to narrow.
    let mut d_cap = d_end;
    {
        let mut d = 0.5 * d_end;
        while d < d_end {
            match extent(d, uc) {
                Some((l, h)) if h - l >= 0.97 * full_w => d_cap = d,
                _ => break,
            }
            d += step;
        }
    }
    // Row schedule: tuck rows, regular rows (with one on the ride line), then the narrowing tip.
    // Where the waist paint under the strap stops being leather, across the strap (by u).
    let body_rgb = |q: V3| -> [u8; 3] {
        let (_, t, b) = shells::closest_in_shell(model, &rest, body, q);
        let c = m.tris[t].map(|v| v as usize);
        let uv = [b[0] * m.uv[c[0]][0] + b[1] * m.uv[c[1]][0] + b[2] * m.uv[c[2]][0], b[0] * m.uv[c[0]][1] + b[1] * m.uv[c[1]][1] + b[2] * m.uv[c[2]][1]];
        let x = ((uv[0] * tex.w as f32) as i64).clamp(0, tex.w as i64 - 1) as usize;
        let y = ((uv[1] * tex.h as f32) as i64).clamp(0, tex.h as i64 - 1) as usize;
        let o = (y * tex.w + x) * 4;
        [tex.rgba[o], tex.rgba[o + 1], tex.rgba[o + 2]]
    };
    let leather = |c: [u8; 3]| c[0] as i32 > c[2] as i32 + 20 && c[0] > 60;
    // Paint a viewer in front of the strap sees behind a card point: the first waist surface hit
    // looking in along the card normal. (The nearest waist point can land on the underside of the
    // knot's rim, which is leather, below where the rim visibly ends.)
    let card_n = geom::norm(front.iter().flatten().fold([0.0f32; 3], |acc, &v| geom::add(acc, attr_n[v])));
    let seen_rgb = |q: V3| -> [u8; 3] {
        let o = geom::add(q, geom::scale(card_n, 0.04));
        let dir = geom::scale(card_n, -1.0);
        let mut best: Option<f32> = None;
        for &t in &body.tris {
            let c = m.tris[t].map(|v| rest[mk][v as usize]);
            if let Some(h) = geom::ray_tri(o, dir, c[0], c[1], c[2], 0.0) {
                if best.is_none_or(|b| h < b) {
                    best = Some(h);
                }
            }
        }
        body_rgb(best.map_or(q, |h| geom::add(o, geom::scale(dir, h))))
    };
    let exit_us: Vec<f32> = (0..=24).map(|i| ua + (ub - ua) * i as f32 / 24.0).collect();
    let card_pt = |u: f32, a: f32| map.pos(m, [u, v_top(u) + d_of_a(a)]);
    const EXIT_STEP: f32 = 0.001;
    let exits: Vec<f32> = if !p.exit_edge.is_empty() || !p.exit_curve.is_empty() {
        // Explicit edge: per column, the first point down the strap that lies below the edge, both
        // seen along the card normal (projected onto the card's plane).
        for &v in &p.exit_edge {
            if !body.verts.contains(&v) {
                return Err(format!("--exit-edge vertex {v} is not on the waist surface {}", body.label));
            }
        }
        let edge_pts: Vec<V3> = p.exit_edge.iter().map(|&v| m.pos[v]).chain(p.exit_curve.iter().copied()).collect();
        if edge_pts.len() < 2 {
            return Err("--exit-edge / --exit-curve need at least two points".into());
        }
        let um = 0.5 * (ua + ub);
        let xa = geom::norm(geom::sub(card_pt(ub, 0.0), card_pt(ua, 0.0)));
        let mut ya = geom::norm(geom::cross(card_n, xa));
        if geom::dot(ya, geom::sub(card_pt(um, 0.1), card_pt(um, 0.0))) < 0.0 {
            ya = geom::scale(ya, -1.0);
        }
        let mut e2: Vec<[f32; 2]> = edge_pts.iter().map(|&q| [geom::dot(q, xa), geom::dot(q, ya)]).collect();
        e2.sort_by(|a, b| a[0].total_cmp(&b[0]));
        let edge_y = |x: f32| -> f32 {
            let i = e2.partition_point(|q| q[0] < x).clamp(1, e2.len() - 1);
            let (q0, q1) = (e2[i - 1], e2[i]);
            q0[1] + (q1[1] - q0[1]) * (x - q0[0]) / (q1[0] - q0[0]).max(1e-9)
        };
        let (x0, x1) = (geom::dot(card_pt(ua, 0.0), xa), geom::dot(card_pt(ub, 0.0), xa));
        if e2[0][0] > x0.min(x1) + 0.01 || e2[e2.len() - 1][0] < x0.max(x1) - 0.01 {
            log.push(format!("note: the exit edge spans {:.3}..{:.3} across, the strap {:.3}..{:.3}; the ends are extended in a straight line", e2[0][0], e2[e2.len() - 1][0], x0.min(x1), x0.max(x1)));
        }
        exit_us
            .iter()
            .map(|&u| {
                let mut a = 0.0f32;
                while a < p.exit_max {
                    let q = card_pt(u, a);
                    if geom::dot(q, ya) >= edge_y(geom::dot(q, xa)) {
                        break;
                    }
                    a += 0.25 * EXIT_STEP;
                }
                a.clamp(0.5 * p.tuck_len, p.exit_max)
            })
            .collect()
    } else if p.exit_down.len() == 1 {
        vec![p.exit_down[0].clamp(0.5 * p.tuck_len, p.exit_max); 25]
    } else if p.exit_down.len() >= 2 {
        let n = p.exit_down.len() - 1;
        (0..=24)
            .map(|i| {
                let f = i as f32 / 24.0 * n as f32;
                let k = (f.floor() as usize).min(n - 1);
                (p.exit_down[k] + (p.exit_down[k + 1] - p.exit_down[k]) * (f - k as f32)).clamp(0.5 * p.tuck_len, p.exit_max)
            })
            .collect()
    } else if p.exit_by_paint {
        let raw: Vec<f32> = exit_us
            .iter()
            .map(|&u| {
                let (mut a, mut run_start, mut miss) = (0.0f32, None, 0);
                while a <= p.exit_max {
                    if leather(seen_rgb(card_pt(u, a))) {
                        miss = 0;
                        run_start = None;
                    } else {
                        miss += 1;
                        if run_start.is_none() {
                            run_start = Some(a);
                        }
                        if miss >= 4 {
                            break;
                        }
                    }
                    a += EXIT_STEP;
                }
                if std::env::var("STRAP_EXIT_DEBUG").is_ok() {
                    let line: String = (0..=((p.exit_max / 0.002) as usize))
                        .map(|i| {
                            let c = seen_rgb(card_pt(u, i as f32 * 0.002));
                            if leather(c) { 'L' } else if c[2] as i32 > c[0] as i32 { 'b' } else { '.' }
                        })
                        .collect();
                    eprintln!("exit u={u:.4}: {line}");
                    let dl: Vec<String> = (0..=((p.exit_max / 0.004) as usize))
                        .map(|i| {
                            let q = card_pt(u, i as f32 * 0.004);
                            let o = geom::add(q, geom::scale(card_n, 0.04));
                            let dir = geom::scale(card_n, -1.0);
                            let mut best: Option<f32> = None;
                            for &t in &body.tris {
                                let c = m.tris[t].map(|v| rest[mk][v as usize]);
                                if let Some(h) = geom::ray_tri(o, dir, c[0], c[1], c[2], 0.0) {
                                    if best.is_none_or(|b| h < b) {
                                        best = Some(h);
                                    }
                                }
                            }
                            best.map_or("-".to_string(), |h| format!("{:.0}", (h - 0.04) * 1000.0))
                        })
                        .collect();
                    eprintln!("depth u={u:.4}: {}", dl.join(" "));
                    let lum = |c: [u8; 3]| (0.3 * c[0] as f32 + 0.59 * c[1] as f32 + 0.11 * c[2] as f32) as i32;
                    let wl: Vec<String> = (0..=((p.exit_max / 0.004) as usize)).map(|i| format!("{}", lum(seen_rgb(card_pt(u, i as f32 * 0.004))))).collect();
                    eprintln!("waistlum u={u:.4}: {}", wl.join(" "));
                    let cl: Vec<String> = (0..=((p.exit_max / 0.004) as usize))
                        .map(|i| {
                            let q = map.atlas([u, v_top(u) + d_of_a(i as f32 * 0.004)]);
                            let x = ((q[0] * tex.w as f32) as i64).clamp(0, tex.w as i64 - 1) as usize;
                            let y = ((q[1] * tex.h as f32) as i64).clamp(0, tex.h as i64 - 1) as usize;
                            let o = (y * tex.w + x) * 4;
                            format!("{}", lum([tex.rgba[o], tex.rgba[o + 1], tex.rgba[o + 2]]))
                        })
                        .collect();
                    eprintln!("cardlum u={u:.4}: {}", cl.join(" "));
                }
                run_start.unwrap_or(p.exit_max)
            })
            .collect();
        let lost = raw.iter().filter(|&&e| e >= p.exit_max - 1e-6).count();
        if lost > 0 {
            log.push(format!("WARNING: the paint behind the strap stays leather to the longest tuck ({}) in {lost} of 25 columns; pass --exit-edge with the waist's edge vertices", p.exit_max));
        }
        // Smooth the paint noise with a 5-column moving average, but never below the raw paint
        // edge: an exit lower than the paint leaves a sliver of waist cloth showing between the
        // painted knot and the strap (a quadratic fit did that at the knot's rounded corner).
        let smooth: Vec<f32> = (0..raw.len())
            .map(|i| {
                let (lo, hi) = (i.saturating_sub(2), (i + 2).min(raw.len() - 1));
                let avg = raw[lo..=hi].iter().sum::<f32>() / (hi - lo + 1) as f32;
                avg.min(raw[i]).clamp(0.5 * p.tuck_len, p.exit_max)
            })
            .collect();
        let worst = raw.iter().zip(&smooth).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        log.push(format!("paint exit smoothed across the strap, never below the paint edge (largest change {worst:.4})"));
        smooth
    } else {
        vec![p.tuck_len; 25]
    };
    let exits: Vec<f32> = exits.into_iter().map(|e| (e - p.exit_raise).max(0.5 * p.tuck_len)).collect();
    let exit_at = |u: f32| -> f32 {
        let f = ((u - ua) / (ub - ua).max(1e-9)).clamp(0.0, 1.0) * 24.0;
        let i = (f.floor() as usize).min(23);
        exits[i] + (exits[i + 1] - exits[i]) * (f - i as f32)
    };
    let exit_hi = exits.iter().copied().fold(0.0f32, f32::max);
    let exit_lo = exits.iter().copied().fold(f32::MAX, f32::min);
    log.push(format!(
        "the strap comes out from under the waist at {:.3}..{:.3} down (by column: {})",
        exit_lo,
        exit_hi,
        exits.iter().step_by(4).map(|e| format!("{e:.3}")).collect::<Vec<_>>().join(" ")
    ));
    log.push(format!(
        "exit curve (reusable with --exit-curve): {}",
        (0..=24).step_by(3).map(|i| { let q = card_pt(exit_us[i], exits[i]); format!("{:.4},{:.4},{:.4}", q[0], q[1], q[2]) }).collect::<Vec<_>>().join(";")
    ));
    // Dense rows through the tuck and its exit, then regular rows.
    let tuck_end = exit_hi + p.lift_len;
    let mut rows_a: Vec<f32> = Vec::new();
    let mut a = 0.0f32;
    while a < tuck_end {
        rows_a.push(a);
        a += 0.008;
    }
    rows_a.push(tuck_end);
    let a_cap = a_of_d(d_cap);
    let mut a = tuck_end + p.row_step;
    while a < a_cap - 0.5 * p.row_step {
        rows_a.push(a);
        a += p.row_step;
    }
    if !rows_a.iter().any(|&x| (x - p.ride).abs() < 0.25 * p.row_step) && p.ride < a_cap {
        rows_a.push(p.ride);
    }
    rows_a.push(a_cap);
    rows_a.sort_by(f32::total_cmp);
    rows_a.dedup_by(|x, y| (*x - *y).abs() < 0.004);
    let mut rows_d: Vec<f32> = rows_a.iter().map(|&a| d_of_a(a)).collect();
    // Tip rows at falling widths, found by scanning down from the cap start.
    for frac in [0.9f32, 0.75, 0.55, 0.32] {
        let mut d = *rows_d.last().unwrap();
        while d + step < d_end {
            match extent(d + step, uc) {
                Some((l, h)) if h - l > frac * full_w => d += step,
                _ => break,
            }
        }
        if d > rows_d.last().unwrap() + 2.0 * texel {
            rows_d.push(d);
        }
    }
    let tip_d = d_end;
    let tip_a = a_of_d(tip_d);
    let c = p.columns.max(2);
    // Front grid in UV, inset a quarter texel from the paint edge.
    let mut grid: Vec<Vec<[f32; 2]>> = Vec::new();
    for &d in &rows_d {
        let (l, h) = extent(d, uc).ok_or_else(|| format!("no paint on row {d}"))?;
        let (l, h) = (l + 0.25 * texel, h - 0.25 * texel);
        grid.push((0..=c).map(|k| {
            let u = l + (h - l) * k as f32 / c as f32;
            [u, v_top(u) + d]
        }).collect());
    }
    let tip_uv = [uc, v_top(uc) + tip_d - 0.25 * texel];
    log.push(format!(
        "card {}: {} front triangles; top edge u {:.4}..{:.4}; length {:.4}; width {:.4} (UV {:.4}); tip paint narrows from {:.4} to the end at {:.4}",
        card.label, front.len(), ua, ub, length, geom::dist(map.pos(m, grid[rows_d.len() / 2][0]), map.pos(m, grid[rows_d.len() / 2][c])), full_w, a_cap, tip_a
    ));
    log.push(format!("rows at {:?}", rows_d.iter().map(|&d| (a_of_d(d) * 1000.0).round() / 1000.0).collect::<Vec<_>>()));

    // Hardware (the silver plate and tip): from where the paint down the strap's middle turns
    // silver to the tip, every point gets the same weights, so it moves rigidly; bending stays in
    // the leather above it.
    let paint = |uv: [f32; 2]| -> [u8; 3] {
        let q = map.atlas(uv);
        let x = ((q[0] * tex.w as f32) as i64).clamp(0, tex.w as i64 - 1) as usize;
        let y = ((q[1] * tex.h as f32) as i64).clamp(0, tex.h as i64 - 1) as usize;
        let o = (y * tex.w + x) * 4;
        [tex.rgba[o], tex.rgba[o + 1], tex.rgba[o + 2]]
    };
    let silver = |c: [u8; 3]| {
        let (hi, lo) = (c.iter().max().copied().unwrap_or(0), c.iter().min().copied().unwrap_or(0));
        hi > 110 && (hi - lo) < 40
    };
    let hw_a = {
        let mut a = p.ride;
        let mut found = tip_a;
        while a < tip_a {
            let run = (0..10).all(|i| silver(paint([uc, v_top(uc) + d_of_a((a + 0.002 * i as f32).min(tip_a))])));
            if run {
                found = a;
                break;
            }
            a += 0.002;
        }
        found
    };
    log.push(format!("hardware: silver paint from {hw_a:.3} down the strap to the tip ({:.3}); weights there are the tip set, so it moves rigidly", tip_a));
    // Front surface position, outward normal and weights per grid point.
    let tip_set = [(p.hips_joint, 1.0 - p.tip_thigh), (p.thigh_joint, p.tip_thigh)];
    let tip_set: Vec<(usize, f32)> = tip_set.into_iter().filter(|e| e.1 > 0.0).collect();
    struct Pt {
        pos: V3,
        n: V3,
        uv: [f32; 2],
        a: f32,
        j: [u16; 4],
        w: [f32; 4],
        /// Distance from the front face to the back face here.
        th: f32,
    }
    let surface_normal = |uv: [f32; 2]| -> V3 {
        let (t, _, _) = map.locate(m, uv);
        geom::norm(geom::add(geom::add(attr_n[t[0]], attr_n[t[1]]), attr_n[t[2]]))
    };
    // Frames for the weight fit: every clip at fit_frames evenly spaced times.
    let mut fit_poses: Vec<(Vec<crate::anim::Mat4>, Vec<V3>)> = Vec::new();
    if p.fit_frames > 0 {
        let rig = model.rig.as_ref().ok_or("model has no skeleton to fit weights against")?;
        for c in 0..rig.clips.len() {
            let dur = rig.clips[c].duration;
            for k in 0..p.fit_frames {
                let jm = rig.joint_matrices(Some(c), dur * k as f32 / p.fit_frames as f32);
                let posed = crate::anim::skin(m, &jm);
                fit_poses.push((jm, posed));
            }
        }
    }
    let (mut fit_n, mut fit_before, mut fit_after) = (0usize, 0.0f64, 0.0f64);
    // Weights that make the points `pts` (front and back of one strap point) follow the waist
    // triangle nearest each of them through the fit frames (the sweep's slide test uses the same
    // anchors); also returns the RMS miss before and after.
    let fit = |pts: [V3; 2], start: &[(usize, f32)]| -> Option<(Vec<(usize, f32)>, f64, f64)> {
        if fit_poses.is_empty() {
            return None;
        }
        struct Anchor {
            c: [usize; 3],
            b: [f32; 3],
            local: V3,
        }
        let at = |p3: [V3; 3], b: [f32; 3]| -> V3 { std::array::from_fn(|k| b[0] * p3[0][k] + b[1] * p3[1][k] + b[2] * p3[2][k]) };
        let anchors: Vec<Anchor> = pts
            .iter()
            .map(|&x| {
                let (_, t, b) = shells::closest_in_shell(model, &rest, body, x);
                let c = m.tris[t].map(|v| v as usize);
                let r3 = c.map(|v| m.pos[v]);
                let (e1, e2, nn) = tri_frame(r3);
                Anchor { c, b, local: local_coords(e1, e2, nn, geom::sub(x, at(r3, b))) }
            })
            .collect();
        let mut js: Vec<usize> = start.iter().map(|e| e.0).collect();
        // Candidate joints: every joint that moves the waist within fit_reach of the point.
        let near: Vec<usize> = body.verts.iter().copied().filter(|&v| geom::dist(m.pos[v], pts[0]) <= p.fit_reach).collect();
        for an in &anchors {
            for &v in an.c.iter().chain(near.iter()) {
                for k in 0..4 {
                    let j = m.joints[v][k] as usize;
                    if m.weights[v][k] > 0.0 && !js.contains(&j) {
                        js.push(j);
                    }
                }
            }
        }
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        for (jm, posed) in &fit_poses {
            for (pi, &x) in pts.iter().enumerate() {
                let an = &anchors[pi];
                let p3 = an.c.map(|v| posed[v]);
                let (f1, f2, fnn) = tri_frame(p3);
                let l = an.local;
                let target = geom::add(at(p3, an.b), geom::add(geom::add(geom::scale(f1, l[0]), geom::scale(f2, l[1])), geom::scale(fnn, l[2])));
                let cols: Vec<V3> = js.iter().map(|&j| xform(&jm[j], x)).collect();
                // Fit in the waist triangle's frame: a miss along the waist normal opens a gap or
                // buries the strap at the exit; a slide along the surface hardly shows on plain
                // leather. So the normal part counts FIT_NORMAL_WEIGHT times.
                let t1 = geom::norm(f1);
                let t2 = geom::cross(fnn, t1);
                for b in [t1, t2, geom::scale(fnn, FIT_NORMAL_WEIGHT)] {
                    rows.push(cols.iter().map(|cc| geom::dot(*cc, b) as f64).collect());
                    y.push(geom::dot(target, b) as f64);
                }
            }
        }
        let miss = |w: &[f64], cols: &[usize]| -> (f64, f64) {
            let (mut e, mut worst) = (0.0f64, 0.0f64);
            for (r3, y3) in rows.chunks(3).zip(y.chunks(3)) {
                let mut d2 = 0.0;
                for (r, &yy) in r3.iter().zip(y3) {
                    let v: f64 = cols.iter().zip(w).map(|(&i, &wi)| r[i] * wi).sum();
                    d2 += (v - yy) * (v - yy);
                }
                e += d2;
                worst = worst.max(d2.sqrt());
            }
            ((e / (rows.len() / 3).max(1) as f64).sqrt(), worst)
        };
        let rms = |w: &[f64], cols: &[usize]| -> f64 { miss(w, cols).0 };
        let all: Vec<usize> = (0..js.len()).collect();
        let w0: Vec<f64> = js.iter().map(|&j| start.iter().find(|e| e.0 == j).map_or(0.0, |e| e.1 as f64)).collect();
        let before = rms(&w0, &all);
        let w = simplex_ls(&rows, &y, js.len(), &w0);
        // Keep the four largest and refit on them.
        let mut order: Vec<usize> = (0..js.len()).collect();
        order.sort_by(|&a, &b| w[b].total_cmp(&w[a]));
        order.truncate(4);
        let sub_rows: Vec<Vec<f64>> = rows.iter().map(|r| order.iter().map(|&i| r[i]).collect()).collect();
        let w4 = simplex_ls(&sub_rows, &y, order.len(), &order.iter().map(|&i| w[i]).collect::<Vec<_>>());
        let after = rms(&w4, &order);
        let worst = miss(&w4, &order).1;
        if std::env::var("STRAP_FIT_DEBUG").is_ok() {
            eprintln!("fit {:?}: rms {:.4} -> {:.4}, worst {:.4}, joints {:?} of {:?}", pts[0], before, after, worst, order.iter().map(|&i| js[i]).collect::<Vec<_>>(), js);
        }
        let out: Vec<(usize, f32)> = order.iter().zip(&w4).filter(|e| *e.1 > 1e-4).map(|(&i, &wi)| (js[i], wi as f32)).collect();
        Some((out, before, after))
    };
    let (mut min_gap, mut tucked) = (f32::MAX, 0usize);
    let mut point = |uv: [f32; 2], d: f32| -> Pt {
        let a = a_of_d(d);
        let mut pos = map.pos(m, uv);
        let n = surface_normal(uv);
        let (q, bn, _) = body_at(pos);
        let gap = geom::dot(geom::sub(pos, q), bn);
        // Tuck: tuck_depth behind the waist above the paint exit, on the waist surface at the exit,
        // then out to where the card hangs. Above the exit the strap is hidden, so its slope there
        // does not show; below it the strap leaves the surface at a shallow angle (no ramp or
        // step to see from the side) and reaches its hanging gap with zero slope.
        let e = exit_at(uv[0]);
        let t1 = e + p.lift_len;
        let hang = gap.max(p.clear_gap);
        let want = if a < e - p.tuck_len {
            Some(-p.tuck_depth)
        } else if a < e {
            Some(-p.tuck_depth * (1.0 - smoothstep((a - (e - p.tuck_len)) / p.tuck_len)))
        } else if a < t1 {
            let t = (a - e) / p.lift_len;
            Some(hang * t * (2.0 - t))
        } else if gap < p.clear_gap {
            Some(p.clear_gap)
        } else {
            None
        };
        if let Some(g) = want {
            pos = geom::add(pos, geom::scale(bn, g - gap));
            if g < 0.0 {
                tucked += 1;
            }
        }
        if a >= t1 {
            min_gap = min_gap.min(gap.max(p.clear_gap));
        }
        // Weights: the waist's own at the ride part, then a blend to the tip set.
        let ride_uv = if a <= p.ride { uv } else { [uv[0], v_top(uv[0]) + d_of_a(p.ride)] };
        let (_, _, mut ride_w) = body_at(map.pos(m, ride_uv));
        let fit_at = if a <= p.ride { pos } else { map.pos(m, ride_uv) };
        if let Some((w, e0, e1)) = fit([fit_at, geom::sub(fit_at, geom::scale(n, p.thickness))], &ride_w) {
            ride_w = w;
            fit_n += 1;
            fit_before += e0;
            fit_after += e1;
        }
        // The blend starts halfway down the exit lift (or at ride, if that comes first), so the
        // change from waist motion to tip motion is spread over the exposed leather instead of
        // stretching a short span above the hardware. smoothstep keeps it near zero at the exit.
        let b0 = p.ride.min(exit_hi + 0.5 * p.lift_len);
        let f = if a <= b0 { 0.0 } else if a >= hw_a { 1.0 } else { smoothstep((a - b0) / (hw_a - b0).max(1e-6)) };
        let (j, w) = blend(&[(&ride_w, 1.0 - f), (&tip_set, f)]);
        // Leather thickness is the same everywhere; the attachment is shaped by position alone.
        Pt { pos, n, uv, a, j, w, th: p.thickness }
    };
    fn row_refs<'a>(rows: &'a [Vec<Pt>], tip: &'a Pt, r: usize) -> Vec<&'a Pt> {
        if r < rows.len() { rows[r].iter().collect() } else { vec![tip] }
    }
    let mut front_pts: Vec<Vec<Pt>> = grid.iter().zip(&rows_d).map(|(row, &d)| row.iter().map(|&uv| point(uv, d)).collect()).collect();
    let mut tip_pt = point(tip_uv, tip_d);

    // Thigh clearance: per row, top to tip, the smallest left/right thigh shares (never falling
    // toward the tip) that keep the row's front and back points out of the body in the frames.
    if p.collide && !fit_poses.is_empty() {
        let flip_b: Vec<bool> = m
            .tris
            .iter()
            .map(|t| {
                let c = t.map(|v| v as usize);
                let g = geom::tri_normal(m.pos[c[0]], m.pos[c[1]], m.pos[c[2]]);
                geom::dot(g, geom::add(geom::add(attr_n[c[0]], attr_n[c[1]]), attr_n[c[2]])) < 0.0
            })
            .collect();
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for pt in front_pts.iter().flatten() {
            for k in 0..3 {
                lo[k] = lo[k].min(pt.pos[k] - 0.35);
                hi[k] = hi[k].max(pt.pos[k] + 0.35);
            }
        }
        let near: Vec<usize> = body.tris.iter().copied().filter(|&t| m.tris[t].iter().any(|&v| (0..3).all(|k| m.pos[v as usize][k] >= lo[k] && m.pos[v as usize][k] <= hi[k]))).collect();
        let winding_at = |posed: &[V3], x: V3| -> f32 {
            let mut sum = 0.0f64;
            for &t in &body.tris {
                let c = m.tris[t].map(|v| geom::sub(posed[v as usize], x));
                let (la, lb, lc) = (geom::len(c[0]) as f64, geom::len(c[1]) as f64, geom::len(c[2]) as f64);
                let det = geom::dot(c[0], geom::cross(c[1], c[2])) as f64;
                let den = la * lb * lc + geom::dot(c[0], c[1]) as f64 * lc + geom::dot(c[1], c[2]) as f64 * la + geom::dot(c[2], c[0]) as f64 * lb;
                let om = 2.0 * det.atan2(den);
                sum += if flip_b[t] { -om } else { om };
            }
            (sum / (4.0 * std::f64::consts::PI)).abs() as f32
        };
        let inside = |posed: &[V3], x: V3| -> bool {
            let mut best = (f32::MAX, 0usize, [0.0f32; 3]);
            for &t in &near {
                let c = m.tris[t].map(|v| posed[v as usize]);
                let (q, _) = geom::closest_on_tri(x, c[0], c[1], c[2]);
                let d = geom::dist(x, q);
                if d < best.0 {
                    best = (d, t, q);
                }
            }
            let c = m.tris[best.1].map(|v| posed[v as usize]);
            let mut nrm = geom::tri_normal(c[0], c[1], c[2]);
            if flip_b[best.1] {
                nrm = geom::scale(nrm, -1.0);
            }
            geom::dot(geom::sub(x, best.2), nrm) < COLLIDE_MARGIN && winding_at(posed, geom::sub(x, geom::scale(nrm, COLLIDE_MARGIN))) > 0.5
        };
        let skin_pt = |jm: &[crate::anim::Mat4], j: [u16; 4], w: [f32; 4], x: V3| -> V3 {
            let (mut o, mut ws) = ([0.0f32; 3], 0.0f32);
            for k in 0..4 {
                if w[k] > 0.0 {
                    let q = xform(&jm[j[k] as usize], x);
                    for i in 0..3 {
                        o[i] += w[k] * q[i];
                    }
                    ws += w[k];
                }
            }
            o.map(|x| x / ws.max(1e-9))
        };
        let frames: Vec<usize> = (0..fit_poses.len()).collect();
        let back = |pt: &Pt| geom::sub(pt.pos, geom::scale(pt.n, pt.th));
        let lt = [(p.left_thigh_joint, 1.0f32)];
        let rt = [(p.thigh_joint, 1.0f32)];
        // Each thigh mostly moves the strap edge on its own side: `side` is 0 at the character's
        // right edge (column 0) and 1 at the left edge.
        let (u_right, u_left) = (front_pts[0][0].uv[0], front_pts[0][front_pts[0].len() - 1].uv[0]);
        let side = |pt: &Pt| ((pt.uv[0] - u_right) / (u_left - u_right)).clamp(0.0, 1.0);
        const SIDE_FLOOR: f32 = 0.5;
        let mixed = |pt: &Pt, cl: f32, cr: f32| -> ([u16; 4], [f32; 4]) {
            let base = shells::weights_at_vertex(&pt.j, &pt.w);
            let sd = side(pt);
            let (l, r) = (cl * (SIDE_FLOOR + (1.0 - SIDE_FLOOR) * sd), cr * (SIDE_FLOOR + (1.0 - SIDE_FLOOR) * (1.0 - sd)));
            blend(&[(&base, 1.0 - l - r), (&lt, l), (&rt, r)])
        };
        let hits = |pts: &[&Pt], cl: f32, cr: f32| -> usize {
            let mut n = 0;
            for pt in pts {
                let (j, w) = mixed(pt, cl, cr);
                for &f in &frames {
                    let (jm, posed) = &fit_poses[f];
                    for x in [pt.pos, back(pt)] {
                        if inside(posed, skin_pt(jm, j, w, x)) {
                            n += 1;
                        }
                    }
                }
            }
            n
        };
        let steps: Vec<f32> = [0.0f32, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.375, 0.45, 0.525, 0.6, 0.7, 0.8, 0.9, 1.0].into_iter().filter(|&k| k <= p.max_thigh_share + 1e-6).collect();
        let nrows = front_pts.len();
        let mut ks = vec![(0.0f32, 0.0f32); nrows + 1];
        let mut left = vec![0usize; nrows + 1];
        let (mut kl, mut kr) = (0.0f32, 0.0f32);
        for r in 0..=nrows {
            let pts: Vec<&Pt> = if r < nrows { front_pts[r].iter().collect() } else { vec![&tip_pt] };
            if pts[0].a < tuck_end {
                continue;
            }
            let mut cands: Vec<(f32, f32)> = Vec::new();
            for &cl in steps.iter().filter(|&&x| x >= kl) {
                for &cr in steps.iter().filter(|&&x| x >= kr) {
                    if (cl * SIDE_FLOOR + cr).max(cl + cr * SIDE_FLOOR) <= 1.0 {
                        cands.push((cl, cr));
                    }
                }
            }
            cands.sort_by(|a, b| (a.0 + a.1).total_cmp(&(b.0 + b.1)).then(a.0.max(a.1).total_cmp(&b.0.max(b.1))));
            let mut best = (usize::MAX, kl, kr);
            for (cl, cr) in cands {
                let n = hits(&pts, cl, cr);
                if n < best.0 {
                    best = (n, cl, cr);
                }
                if n == 0 {
                    break;
                }
            }
            (kl, kr) = (best.1, best.2);
            ks[r] = (kl, kr);
            left[r] = best.0;
        }
        // Ease in: no row's share rises faster than 0.15 per 0.04 down the strap.
        let row_a: Vec<f32> = (0..=nrows).map(|r| if r < nrows { front_pts[r][0].a } else { tip_pt.a }).collect();
        let a_of_row = |r: usize| row_a[r];
        for r in (0..nrows).rev() {
            if a_of_row(r) < tuck_end {
                break;
            }
            let da = (a_of_row(r + 1) - a_of_row(r)).max(0.0);
            let drop = 0.15 * da / 0.04;
            ks[r].0 = ks[r].0.max(ks[r + 1].0 - drop);
            ks[r].1 = ks[r].1.max(ks[r + 1].1 - drop);
        }
        for r in 0..=nrows {
            let (cl, cr) = ks[r];
            if cl == 0.0 && cr == 0.0 {
                continue;
            }
            let pts: Vec<&mut Pt> = if r < nrows { front_pts[r].iter_mut().collect() } else { vec![&mut tip_pt] };
            for pt in pts {
                let (j, w) = mixed(pt, cl, cr);
                pt.j = j;
                pt.w = w;
            }
        }
        // Rows the thigh shares cannot clear in every frame stand a little further off the body
        // (at most max_lift, easing 0.002 per row so the strap drapes rather than steps).
        let hits_lifted = |pts: &[&Pt], lift: f32| -> usize {
            let mut n = 0;
            for pt in pts {
                let front = geom::add(pt.pos, geom::scale(pt.n, lift));
                for &f in &frames {
                    let (jm, posed) = &fit_poses[f];
                    for x in [front, geom::sub(front, geom::scale(pt.n, pt.th))] {
                        if inside(posed, skin_pt(jm, pt.j, pt.w, x)) {
                            n += 1;
                        }
                    }
                }
            }
            n
        };
        let mut lifts = vec![0.0f32; nrows + 1];
        for r in 0..=nrows {
            if a_of_row(r) < tuck_end || left[r] == 0 && hits_lifted(&row_refs(&front_pts, &tip_pt, r), 0.0) == 0 {
                continue;
            }
            let pts = row_refs(&front_pts, &tip_pt, r);
            let mut best = (hits_lifted(&pts, 0.0), 0.0f32);
            let mut lift = 0.001f32;
            while best.0 > 0 && lift <= p.max_lift + 1e-6 {
                let n = hits_lifted(&pts, lift);
                if n < best.0 {
                    best = (n, lift);
                }
                lift += 0.001;
            }
            // A lift that does not clear the row in every frame only adds a gap; keep the row on the body.
            lifts[r] = if best.0 == 0 { best.1 } else { 0.0 };
            left[r] = best.0;
        }
        for _ in 0..2 {
            for r in 1..=nrows {
                lifts[r] = lifts[r].max(lifts[r - 1] - 0.002);
            }
            for r in (0..nrows).rev() {
                if a_of_row(r) >= tuck_end {
                    lifts[r] = lifts[r].max(lifts[r + 1] - 0.002);
                }
            }
        }
        for r in 0..=nrows {
            if lifts[r] > 0.0 {
                let pts: Vec<&mut Pt> = if r < nrows { front_pts[r].iter_mut().collect() } else { vec![&mut tip_pt] };
                for pt in pts {
                    pt.pos = geom::add(pt.pos, geom::scale(pt.n, lifts[r]));
                }
            }
        }
        let lifted: Vec<String> = (0..=nrows).filter(|&r| lifts[r] > 0.0).map(|r| format!("{:.2}:+{:.3}", a_of_row(r), lifts[r])).collect();
        if !lifted.is_empty() {
            log.push(format!("rows lifted off the body: {}", lifted.join(", ")));
        }
        let shown: Vec<String> = (0..=nrows).filter(|&r| ks[r] != (0.0, 0.0) || left[r] > 0).map(|r| format!("{:.2}:L{:.2}/R{:.2}{}", a_of_row(r), ks[r].0, ks[r].1, if left[r] > 0 { format!(" ({} hits left)", left[r]) } else { String::new() })).collect();
        log.push(format!("thigh clearance over {} frames: {}", frames.len(), if shown.is_empty() { "every row clear".into() } else { shown.join(", ") }));
    }
    log.push(format!("{tucked} front points tucked behind the waist; smallest gap below the tuck {:.4}", min_gap));
    if fit_n > 0 {
        log.push(format!(
            "weight fit over {} frames: {} points, mean RMS miss from the waist point (normal part counted {FIT_NORMAL_WEIGHT}x) {:.4} with blended weights, {:.4} fitted",
            fit_poses.len(),
            fit_n,
            fit_before / fit_n as f64,
            fit_after / fit_n as f64
        ));
    }

    // Shape report, kept apart from clearance: how much strap edges stretch or squash over the fit
    // frames, for the tucked part, the exposed leather and the hardware.
    if !fit_poses.is_empty() {
        let sk = |jm: &[crate::anim::Mat4], pt: &Pt| -> V3 {
            let mut o = [0.0f32; 3];
            for i in 0..4 {
                if pt.w[i] > 0.0 {
                    o = geom::add(o, geom::scale(xform(&jm[pt.j[i] as usize], pt.pos), pt.w[i]));
                }
            }
            o
        };
        let mut worst = [0.0f32; 3];
        let mut worst_at = [(0.0f32, false); 3];
        let rows_n = front_pts.len();
        // (edge ends, runs across the strap)
        let mut edges: Vec<(&Pt, &Pt, bool)> = Vec::new();
        for r in 0..rows_n {
            for k in 0..front_pts[r].len() {
                if k + 1 < front_pts[r].len() {
                    edges.push((&front_pts[r][k], &front_pts[r][k + 1], true));
                }
                if r + 1 < rows_n {
                    edges.push((&front_pts[r][k], &front_pts[r + 1][k], false));
                }
            }
        }
        for (jm, _) in &fit_poses {
            for &(x, y, across) in &edges {
                let l0 = geom::dist(x.pos, y.pos).max(1e-6);
                let l1 = geom::dist(sk(jm, x), sk(jm, y));
                let g = if x.a.min(y.a) >= hw_a { 2 } else if x.a.max(y.a) > exit_lo { 1 } else { 0 };
                let c = (l1 / l0 - 1.0).abs();
                if c > worst[g] {
                    worst[g] = c;
                    worst_at[g] = (0.5 * (x.a + y.a), across);
                }
            }
        }
        // The source card under the same frames, for comparison.
        let mut src_worst = 0.0f32;
        for (jm, _) in &fit_poses {
            let skv = |v: usize| -> V3 {
                let mut o = [0.0f32; 3];
                for i in 0..4 {
                    if m.weights[v][i] > 0.0 {
                        o = geom::add(o, geom::scale(xform(&jm[m.joints[v][i] as usize], m.pos[v]), m.weights[v][i]));
                    }
                }
                o
            };
            for t in &front {
                for k in 0..3 {
                    let (x, y) = (t[k], t[(k + 1) % 3]);
                    let l0 = geom::dist(m.pos[x], m.pos[y]);
                    if l0 > 1e-5 {
                        src_worst = src_worst.max((geom::dist(skv(x), skv(y)) / l0 - 1.0).abs());
                    }
                }
            }
        }
        log.push(format!(
            "shape over {} frames (largest edge length change): tucked {:.1}% ({} at {:.3} down), exposed leather {:.1}% ({} at {:.3} down), hardware {:.2}% (source card {:.1}%)",
            fit_poses.len(),
            100.0 * worst[0],
            if worst_at[0].1 { "across" } else { "along" },
            worst_at[0].0,
            100.0 * worst[1],
            if worst_at[1].1 { "across" } else { "along" },
            worst_at[1].0,
            100.0 * worst[2],
            100.0 * src_worst
        ));
    }
    // Assemble: every non-card vertex first (in order), then the strap.
    let card_v: std::collections::HashSet<usize> = card.verts.iter().copied().collect();
    let mut remap = vec![u32::MAX; m.pos.len()];
    let mut verts: Vec<NewVert> = Vec::new();
    for v in 0..m.pos.len() {
        if !card_v.contains(&v) {
            remap[v] = verts.len() as u32;
            verts.push(NewVert::Old(v as u32));
        }
    }
    let first_strap = verts.len();
    let mut meta = Vec::new();
    let push = |verts: &mut Vec<NewVert>, meta: &mut Vec<(u32, f32, &'static str)>, pt: &Pt, pos: V3, normal: V3, uv: [f32; 2], face: &'static str| -> u32 {
        let i = verts.len() as u32;
        let (t, b, _) = map.locate(m, uv);
        let b = b.map(|x| x.max(0.0));
        let sb = (b[0] + b[1] + b[2]).max(1e-12);
        verts.push(NewVert::New { pos, normal, uv: map.atlas(uv), joints: pt.j, weights: pt.w, src: (t.map(|v| v as u32), b.map(|x| x / sb)) });
        meta.push((i, pt.a, face));
        i
    };
    let rows = front_pts.len();
    // Front face winding: match the card's front triangles.
    let wind_ok = {
        let t = front[0];
        let g = geom::tri_normal(m.pos[t[0]], m.pos[t[1]], m.pos[t[2]]);
        let an = geom::add(geom::add(attr_n[t[0]], attr_n[t[1]]), attr_n[t[2]]);
        geom::dot(g, an) > 0.0
    };
    let mut strap_tris: Vec<[u32; 3]> = Vec::new();
    let quad = |tris: &mut Vec<[u32; 3]>, a: u32, b: u32, cc: u32, d: u32, flip: bool| {
        if flip {
            tris.push([a, cc, b]);
            tris.push([a, d, cc]);
        } else {
            tris.push([a, b, cc]);
            tris.push([a, cc, d]);
        }
    };
    let back_of = |pt: &Pt| geom::sub(pt.pos, geom::scale(pt.n, pt.th));
    let mut faces = [Vec::new(), Vec::new()];
    let mut tips = [0u32; 2];
    for (side, face) in ["front", "back"].into_iter().enumerate() {
        let mut idx = vec![vec![0u32; c + 1]; rows];
        for r in 0..rows {
            for k in 0..=c {
                let pt = &front_pts[r][k];
                let (pos, n) = if side == 0 { (pt.pos, pt.n) } else { (back_of(pt), geom::scale(pt.n, -1.0)) };
                idx[r][k] = push(&mut verts, &mut meta, pt, pos, n, pt.uv, face);
            }
        }
        let (pos, n) = if side == 0 { (tip_pt.pos, tip_pt.n) } else { (back_of(&tip_pt), geom::scale(tip_pt.n, -1.0)) };
        tips[side] = push(&mut verts, &mut meta, &tip_pt, pos, n, tip_pt.uv, face);
        faces[side] = idx;
    }
    // Decide the grid's winding once from the first front quad's geometric normal.
    let grid_flip = {
        let (a, b, d) = (&front_pts[0][0], &front_pts[0][1], &front_pts[1][0]);
        let g = geom::tri_normal(a.pos, b.pos, d.pos);
        geom::dot(g, a.n) < 0.0
    };
    let _ = wind_ok;
    for side in 0..2 {
        let flip = grid_flip ^ (side == 1);
        let idx = &faces[side];
        for r in 0..rows - 1 {
            for k in 0..c {
                // a=(r,k) b=(r,k+1) c=(r+1,k+1) d=(r+1,k): normal = (b-a)x(d-a) for the unflipped order.
                quad(&mut strap_tris, idx[r][k], idx[r][k + 1], idx[r + 1][k + 1], idx[r + 1][k], flip);
            }
        }
        for k in 0..c {
            let (a, b) = (idx[rows - 1][k], idx[rows - 1][k + 1]);
            if flip {
                strap_tris.push([a, tips[side], b]);
            } else {
                strap_tris.push([a, b, tips[side]]);
            }
        }
    }
    // Walls around the outline: top (left to right), right side down, tip, left side up.
    let mut ring: Vec<(&Pt, usize, usize)> = Vec::new(); // (point, row, col); tip = (rows, 0)
    for k in 0..=c {
        ring.push((&front_pts[0][k], 0, k));
    }
    for r in 1..rows {
        ring.push((&front_pts[r][c], r, c));
    }
    ring.push((&tip_pt, rows, 0));
    for r in (1..rows).rev() {
        ring.push((&front_pts[r][0], r, 0));
    }
    let nring = ring.len();
    // Wall UV: the point's UV moved toward the strap's middle line.
    let inset = p.wall_inset * full_w;
    let wall_uv = |pt: &Pt| -> [f32; 2] {
        let toward = [uc - pt.uv[0], 0.5 * (v_top(uc) + v_top(uc) + 0.5 * tip_d) - pt.uv[1]];
        let l = (toward[0] * toward[0] + toward[1] * toward[1]).sqrt().max(1e-9);
        let s = inset.min(l * 0.5);
        [pt.uv[0] + toward[0] / l * s, pt.uv[1] + toward[1] / l * s]
    };
    let mut wf = Vec::with_capacity(nring);
    let mut wb = Vec::with_capacity(nring);
    for i in 0..nring {
        let (pt, _, _) = ring[i];
        let prev = ring[(i + nring - 1) % nring].0.pos;
        let next = ring[(i + 1) % nring].0.pos;
        let along = geom::norm(geom::sub(next, prev));
        let mut out = geom::norm(geom::cross(along, pt.n));
        // Outward = away from the strap's middle.
        let midp = front_pts[rows / 2][c / 2].pos;
        if geom::dot(out, geom::sub(pt.pos, midp)) < 0.0 {
            out = geom::scale(out, -1.0);
        }
        let uv = wall_uv(pt);
        wf.push(push(&mut verts, &mut meta, pt, pt.pos, out, uv, "wall"));
        wb.push(push(&mut verts, &mut meta, pt, back_of(pt), out, uv, "wall"));
    }
    // Wall winding: outward normal.
    for i in 0..nring {
        let j = (i + 1) % nring;
        let (a, b) = (ring[i].0, ring[j].0);
        let g = geom::tri_normal(a.pos, b.pos, back_of(b));
        let midp = front_pts[rows / 2][c / 2].pos;
        let outward = geom::dot(g, geom::sub(a.pos, midp)) > 0.0;
        if outward {
            strap_tris.push([wf[i], wf[j], wb[j]]);
            strap_tris.push([wf[i], wb[j], wb[i]]);
        } else {
            strap_tris.push([wf[i], wb[j], wf[j]]);
            strap_tris.push([wf[i], wb[i], wb[j]]);
        }
    }
    let strap_count = strap_tris.len();
    let strap_verts = verts.len() - first_strap;
    let mut tris = strap_tris;
    for (t, tri) in m.tris.iter().enumerate() {
        if card.tris.contains(&t) {
            continue;
        }
        let r = tri.map(|v| remap[v as usize]);
        if r.contains(&u32::MAX) {
            return Err(format!("triangle {t} uses a card vertex but is not part of the card"));
        }
        tris.push(r);
    }
    log.push(format!("strap: {strap_verts} vertices, {strap_count} triangles ({} rows x {} columns, front, back, {} wall segments)", rows, c, nring));
    Ok(Built { verts, tris, strap_tris: strap_count, strap_verts, rows, length, width: full_w, tip_a, log, meta })
}

fn comp_size(ct: u64) -> Result<usize, String> {
    match ct {
        5120 | 5121 => Ok(1),
        5122 | 5123 => Ok(2),
        5125 | 5126 => Ok(4),
        _ => Err(format!("componentType {ct}")),
    }
}

fn ncomp(t: &str) -> Result<usize, String> {
    Ok(match t {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        _ => return Err(format!("accessor type {t}")),
    })
}

/// How the rebuild fills one vertex attribute on the new strap vertices.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Fill {
    /// Computed by the rebuild (position, normal, main UV, main skin set).
    Set,
    /// Extra skin sets: zero (the main set carries all the weight).
    Zero,
    /// Blended from the card triangle under the vertex (extra UV sets, colors, custom data).
    Interp,
    /// Integer data that cannot be blended: copied from the nearest card corner.
    Nearest,
}

fn fill_of(name: &str, ct: u64, nc: usize, norm: bool) -> Result<Fill, String> {
    let float = ct == 5126;
    let need = |ok: bool, what: &str| if ok { Ok(Fill::Set) } else { Err(format!("{name} must be {what} (it is componentType {ct}, {nc} components)")) };
    match name {
        "POSITION" | "NORMAL" => need(float && nc == 3, "float VEC3"),
        "TEXCOORD_0" => need((float || norm) && nc == 2, "float or normalized VEC2"),
        "WEIGHTS_0" => need((float || norm) && nc == 4, "float or normalized VEC4"),
        "JOINTS_0" => need((ct == 5121 || ct == 5123) && !norm && nc == 4, "unsigned byte or short VEC4"),
        n if n.starts_with("JOINTS_") || n.starts_with("WEIGHTS_") => Ok(Fill::Zero),
        _ if float || norm => Ok(Fill::Interp),
        _ => Ok(Fill::Nearest),
    }
}

/// One accessor component as a float (normalized integers scaled to 0..1 or -1..1).
fn read_comp(b: &[u8], ct: u64, norm: bool) -> f64 {
    match ct {
        5126 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
        5121 => b[0] as f64 / if norm { 255.0 } else { 1.0 },
        5120 => (b[0] as i8 as f64 / if norm { 127.0 } else { 1.0 }).max(if norm { -1.0 } else { f64::MIN }),
        5123 => u16::from_le_bytes([b[0], b[1]]) as f64 / if norm { 65535.0 } else { 1.0 },
        5122 => (i16::from_le_bytes([b[0], b[1]]) as f64 / if norm { 32767.0 } else { 1.0 }).max(if norm { -1.0 } else { f64::MIN }),
        5125 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
        _ => 0.0,
    }
}

fn write_comp(x: f64, ct: u64, norm: bool, out: &mut Vec<u8>) {
    match ct {
        5126 => out.extend_from_slice(&(x as f32).to_le_bytes()),
        5121 => out.push((if norm { x * 255.0 } else { x }).round().clamp(0.0, 255.0) as u8),
        5120 => out.push((if norm { x * 127.0 } else { x }).round().clamp(-128.0, 127.0) as i8 as u8),
        5123 => out.extend_from_slice(&((if norm { x * 65535.0 } else { x }).round().clamp(0.0, 65535.0) as u16).to_le_bytes()),
        5122 => out.extend_from_slice(&((if norm { x * 32767.0 } else { x }).round().clamp(-32768.0, 32767.0) as i16).to_le_bytes()),
        _ => out.extend_from_slice(&(x.round().max(0.0) as u32).to_le_bytes()),
    }
}

/// Checks, before anything is rebuilt, that every vertex attribute of mesh `k` can be carried to
/// new vertices. Returns one line per attribute saying how it will be filled.
pub fn preflight(model: &Model, k: usize) -> Result<Vec<String>, String> {
    let (mi, pi) = prim_of(model, k).ok_or("mesh not found")?;
    let prim = &model.glb.json["meshes"][mi]["primitives"][pi];
    if prim.get("targets").is_some() {
        return Err("the strap's mesh has morph targets; new vertices would need target deltas (not supported)".into());
    }
    if prim.get("mode").and_then(Value::as_u64).is_some_and(|m| m != 4) {
        return Err("the strap's mesh is not a triangle list".into());
    }
    let attrs = prim["attributes"].as_object().ok_or("primitive without attributes")?;
    for need in ["POSITION", "NORMAL", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"] {
        if !attrs.contains_key(need) {
            return Err(format!("the strap's mesh has no {need}"));
        }
    }
    let mut lines = Vec::new();
    for (name, ai) in attrs {
        let a = &model.glb.json["accessors"][ai.as_u64().ok_or("bad accessor index")? as usize];
        if a.get("sparse").is_some() {
            return Err(format!("{name}: sparse accessor (not supported)"));
        }
        if a.get("bufferView").is_none() {
            return Err(format!("{name}: accessor without a bufferView (not supported)"));
        }
        let ct = a["componentType"].as_u64().unwrap_or(0);
        let nc = ncomp(a["type"].as_str().unwrap_or(""))?;
        comp_size(ct)?;
        let norm = a["normalized"].as_bool().unwrap_or(false);
        let fill = fill_of(name, ct, nc, norm)?;
        lines.push(format!(
            "{name} ({} x{nc}{}): {}",
            match ct { 5126 => "float", 5121 => "ubyte", 5120 => "byte", 5123 => "ushort", 5122 => "short", _ => "uint" },
            if norm { ", normalized" } else { "" },
            match fill {
                Fill::Set => "computed for new vertices",
                Fill::Zero => "zero on new vertices (the main skin set holds all the weight)",
                Fill::Interp => "blended from the card triangle under each new vertex",
                Fill::Nearest => "copied from the nearest card corner",
            }
        ));
    }
    Ok(lines)
}

/// Every vertex attribute of mesh `k`, packed (name, element size, bytes).
fn attr_bytes(model: &Model, k: usize) -> Result<Vec<(String, usize, Vec<u8>)>, String> {
    let glb = &model.glb;
    let (mi, pi) = prim_of(model, k).ok_or("mesh not found")?;
    let attrs = glb.json["meshes"][mi]["primitives"][pi]["attributes"].as_object().ok_or("primitive without attributes")?;
    let mut out = Vec::new();
    for (name, ai) in attrs {
        let a = &glb.json["accessors"][ai.as_u64().unwrap_or(0) as usize];
        let ct = a["componentType"].as_u64().unwrap_or(0);
        let elem = comp_size(ct)? * ncomp(a["type"].as_str().unwrap_or(""))?;
        let view = a["bufferView"].as_u64().ok_or("accessor without bufferView")? as usize;
        let vb = glb.view_bytes(view)?;
        let stride = glb.list("bufferViews")[view].get("byteStride").and_then(Value::as_u64).map(|s| s as usize).unwrap_or(elem);
        let base = a.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let count = a["count"].as_u64().unwrap_or(0) as usize;
        let mut bytes = Vec::with_capacity(count * elem);
        for i in 0..count {
            bytes.extend_from_slice(vb.get(base + i * stride..base + i * stride + elem).ok_or(format!("{name} reads past its view"))?);
        }
        out.push((name.clone(), elem, bytes));
    }
    Ok(out)
}

fn push_view(json: &mut Value, bin: &mut Vec<u8>, bytes: &[u8], target: u64) -> usize {
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let off = bin.len();
    bin.extend_from_slice(bytes);
    let views = json["bufferViews"].as_array_mut().expect("bufferViews");
    views.push(json!({ "buffer": 0, "byteOffset": off, "byteLength": bytes.len(), "target": target }));
    views.len() - 1
}

/// Writes the model with mesh `k` replaced by `verts` and `tris`. New data is appended to the BIN
/// chunk as new views and accessors; the old ones stay (unused), so images, skins and clips keep
/// their bytes. Other meshes are not touched.
pub fn write(model: &Model, k: usize, verts: &[NewVert], tris: &[[u32; 3]]) -> Result<Vec<u8>, String> {
    let glb = &model.glb;
    let (mi, pi) = prim_of(model, k).ok_or("mesh not found")?;
    let mut json = glb.json.clone();
    let mut bin = glb.bin().to_vec();
    let attrs: Vec<(String, usize)> =
        json["meshes"][mi]["primitives"][pi]["attributes"].as_object().ok_or("primitive without attributes")?.iter().map(|(n, a)| (n.clone(), a.as_u64().unwrap_or(0) as usize)).collect();
    for (name, ai) in attrs {
        let a = json["accessors"][ai].clone();
        if a.get("sparse").is_some() {
            return Err(format!("{name}: sparse accessors are not supported"));
        }
        let ct = a["componentType"].as_u64().unwrap_or(0);
        let nc = ncomp(a["type"].as_str().unwrap_or(""))?;
        let cs = comp_size(ct)?;
        let elem = cs * nc;
        let norm = a["normalized"].as_bool().unwrap_or(false);
        let fill = fill_of(&name, ct, nc, norm)?;
        let view = a["bufferView"].as_u64().ok_or("accessor without bufferView")? as usize;
        let vb = glb.view_bytes(view)?;
        let stride = glb.list("bufferViews")[view].get("byteStride").and_then(Value::as_u64).map(|s| s as usize).unwrap_or(elem);
        let base = a.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let mut out = Vec::with_capacity(verts.len() * elem);
        for nv in verts {
            match nv {
                NewVert::Old(s) => {
                    let o = base + *s as usize * stride;
                    out.extend_from_slice(vb.get(o..o + elem).ok_or(format!("{name} reads past its view"))?);
                }
                NewVert::New { pos, normal, uv, joints, weights, src } => {
                    let corner = |j: usize, c: usize| -> Result<f64, String> {
                        let o = base + src.0[j] as usize * stride + c * cs;
                        Ok(read_comp(vb.get(o..o + cs).ok_or(format!("{name} reads past its view"))?, ct, norm))
                    };
                    let vals: Vec<f64> = match fill {
                        Fill::Set => match name.as_str() {
                            "POSITION" => pos.iter().map(|&x| x as f64).collect(),
                            "NORMAL" => normal.iter().map(|&x| x as f64).collect(),
                            "TEXCOORD_0" => uv.iter().map(|&x| x as f64).collect(),
                            "WEIGHTS_0" => weights.iter().map(|&x| x as f64).collect(),
                            _ => joints.iter().map(|&j| j as f64).collect(),
                        },
                        Fill::Zero => vec![0.0; nc],
                        Fill::Interp => (0..nc).map(|c| (0..3).map(|j| corner(j, c).map(|x| src.1[j] as f64 * x)).sum::<Result<f64, String>>()).collect::<Result<_, _>>()?,
                        Fill::Nearest => {
                            let j = (0..3).max_by(|&x, &y| src.1[x].total_cmp(&src.1[y])).unwrap_or(0);
                            (0..nc).map(|c| corner(j, c)).collect::<Result<_, _>>()?
                        }
                    };
                    if name == "JOINTS_0" && ct == 5121 && joints.iter().any(|&j| j > 255) {
                        return Err("JOINTS_0 is unsigned byte but a strap joint index is over 255".into());
                    }
                    for x in vals {
                        write_comp(x, ct, norm, &mut out);
                    }
                }
            }
        }
        let nview = push_view(&mut json, &mut bin, &out, 34962);
        let mut na = a.clone();
        let o = na.as_object_mut().unwrap();
        o.insert("bufferView".into(), json!(nview));
        o.remove("byteOffset");
        o.insert("count".into(), json!(verts.len()));
        if o.contains_key("min") || name == "POSITION" {
            // Recompute bounds for float attributes.
            if ct == 5126 {
                let (mut lo, mut hi) = (vec![f32::MAX; nc], vec![f32::MIN; nc]);
                for e in out.chunks_exact(elem) {
                    for c in 0..nc {
                        let x = f32::from_le_bytes([e[4 * c], e[4 * c + 1], e[4 * c + 2], e[4 * c + 3]]);
                        lo[c] = lo[c].min(x);
                        hi[c] = hi[c].max(x);
                    }
                }
                o.insert("min".into(), json!(lo));
                o.insert("max".into(), json!(hi));
            } else {
                o.remove("min");
                o.remove("max");
            }
        }
        let list = json["accessors"].as_array_mut().unwrap();
        list.push(na);
        let idx = list.len() - 1;
        json["meshes"][mi]["primitives"][pi]["attributes"][&name] = json!(idx);
    }
    let wide = verts.len() > 65535;
    let ib: Vec<u8> = if wide { tris.iter().flatten().flat_map(|v| v.to_le_bytes()).collect() } else { tris.iter().flatten().flat_map(|&v| (v as u16).to_le_bytes()).collect() };
    let view = push_view(&mut json, &mut bin, &ib, 34963);
    let list = json["accessors"].as_array_mut().unwrap();
    list.push(json!({ "bufferView": view, "componentType": if wide { 5125 } else { 5123 }, "count": tris.len() * 3, "type": "SCALAR" }));
    let ia = list.len() - 1;
    json["meshes"][mi]["primitives"][pi]["indices"] = json!(ia);
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
    Ok(crate::glb::Glb { version: glb.version, chunks, json }.to_bytes())
}

/// Checks a rebuilt file against its source: other meshes, images, skeleton and clips identical,
/// and every kept vertex of the rebuilt mesh unchanged. Returns lines for the report.
pub fn verify(src: &Model, out: &Model, k: usize, card_tris: &[usize], built: &Built) -> Result<Vec<String>, String> {
    let mut lines = Vec::new();
    if src.meshes.len() != out.meshes.len() {
        return Err("mesh count changed".into());
    }
    for (i, (a, b)) in src.meshes.iter().zip(&out.meshes).enumerate() {
        if i == k {
            continue;
        }
        if a.pos != b.pos || a.uv != b.uv || a.tris != b.tris || a.joints != b.joints || a.weights != b.weights {
            return Err(format!("mesh {} changed", a.name));
        }
    }
    lines.push(format!("other meshes identical ({})", out.meshes.len() - 1));
    if src.images.len() != out.images.len() || src.images.iter().zip(&out.images).any(|(a, b)| a.bytes != b.bytes) {
        return Err("images changed".into());
    }
    lines.push(format!("images identical ({})", out.images.len()));
    if src.joint_names != out.joint_names || src.clips != out.clips {
        return Err("skeleton or clips changed".into());
    }
    for c in 0..src.clips.len() {
        let (ja, jb) = (src.rig.as_ref().unwrap().joint_matrices(Some(c), 0.37), out.rig.as_ref().unwrap().joint_matrices(Some(c), 0.37));
        if ja != jb {
            return Err(format!("clip {} poses differ", src.clips[c]));
        }
    }
    lines.push(format!("skeleton and clips identical ({} joints; {})", out.joint_names.len(), out.clips.join(", ")));
    let (a, b) = (&src.meshes[k], &out.meshes[k]);
    let mut kept = 0;
    for (ni, nv) in built.verts.iter().enumerate() {
        if let NewVert::Old(s) = nv {
            let s = *s as usize;
            if a.pos[s] != b.pos[ni] || a.uv[s] != b.uv[ni] || a.joints[s] != b.joints[ni] || a.weights[s] != b.weights[ni] {
                return Err(format!("kept vertex {s} changed"));
            }
            kept += 1;
        }
    }
    // Every attribute of every kept vertex, byte for byte.
    let (sa, oa) = (attr_bytes(src, k)?, attr_bytes(out, k)?);
    let names = |v: &[(String, usize, Vec<u8>)]| v.iter().map(|e| e.0.clone()).collect::<Vec<_>>();
    if names(&sa) != names(&oa) {
        return Err(format!("attribute set changed: {:?} -> {:?}", names(&sa), names(&oa)));
    }
    for ((name, elem, sb), (_, _, ob)) in sa.iter().zip(&oa) {
        for (ni, nv) in built.verts.iter().enumerate() {
            if let NewVert::Old(s) = nv {
                let s = *s as usize;
                if sb.get(s * elem..(s + 1) * elem) != ob.get(ni * elem..(ni + 1) * elem) {
                    return Err(format!("kept vertex {s}: {name} bytes changed"));
                }
            }
        }
    }
    lines.push(format!("kept vertices: all {} attributes byte-identical ({})", sa.len(), names(&sa).join(", ")));
    // The glTF JSON outside the rebuilt primitive.
    let (sj, oj) = (&src.glb.json, &out.glb.json);
    let (mi, pi) = prim_of(src, k).ok_or("mesh not found")?;
    let skip = ["meshes", "accessors", "bufferViews", "buffers"];
    let mut same = Vec::new();
    for (key, v) in sj.as_object().ok_or("JSON root is not an object")? {
        if skip.contains(&key.as_str()) {
            continue;
        }
        if oj.get(key) != Some(v) {
            return Err(format!("JSON {key} changed"));
        }
        same.push(key.clone());
    }
    if oj.as_object().map(|o| o.len()) != sj.as_object().map(|o| o.len()) {
        return Err("JSON top-level keys changed".into());
    }
    let (sm, om) = (sj["meshes"].as_array().ok_or("no meshes")?, oj["meshes"].as_array().ok_or("no meshes")?);
    for (i, (a1, b1)) in sm.iter().zip(om).enumerate() {
        if i != mi && a1 != b1 {
            return Err(format!("JSON mesh {i} changed"));
        }
    }
    let (sp, op) = (&sm[mi]["primitives"], &om[mi]["primitives"]);
    for (i, (a1, b1)) in sp.as_array().into_iter().flatten().zip(op.as_array().into_iter().flatten()).enumerate() {
        let strip = |v: &Value| {
            let mut v = v.clone();
            if i == pi {
                if let Some(o) = v.as_object_mut() {
                    o.remove("attributes");
                    o.remove("indices");
                }
            }
            v
        };
        if strip(a1) != strip(b1) {
            return Err(format!("JSON primitive {i} of the strap mesh changed outside its attributes and indices"));
        }
    }
    for key in ["accessors", "bufferViews"] {
        let (a1, b1) = (sj[key].as_array().ok_or("missing list")?, oj[key].as_array().ok_or("missing list")?);
        if b1.len() < a1.len() || a1[..] != b1[..a1.len()] {
            return Err(format!("JSON {key}: source entries changed"));
        }
    }
    lines.push(format!("JSON identical outside the strap primitive ({}); source accessors and buffer views kept, new data appended", same.join(", ")));
    // The kept triangles follow the strap, in the source order, on the same corners.
    let src_kept: Vec<&[u32; 3]> = a.tris.iter().enumerate().filter(|(t, _)| !card_tris.contains(t)).map(|(_, t)| t).collect();
    let out_kept = &b.tris[built.strap_tris..];
    if src_kept.len() != out_kept.len() {
        return Err(format!("{} kept triangles in the source, {} in the output", src_kept.len(), out_kept.len()));
    }
    for (s, o) in src_kept.iter().zip(out_kept) {
        if s.map(|v| a.pos[v as usize]) != o.map(|v| b.pos[v as usize]) || s.map(|v| a.uv[v as usize]) != o.map(|v| b.uv[v as usize]) {
            return Err("a kept triangle changed".into());
        }
    }
    lines.push(format!("{}: {kept} kept vertices and {} kept triangles identical; {} strap triangles replace {} card triangles", b.name, out_kept.len(), built.strap_tris, card_tris.len()));
    Ok(lines)
}
