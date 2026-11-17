//! Hair texturing from parameters: game-agnostic tool for hair cards, fur and tails.
//!
//! Stages (each one is plain data, so the checks can look at any of them):
//! 1. `HairSet::build`: the selected triangles, welded, with exact duplicate faces (back faces)
//!    grouped under one primary, and locks = edge-connected welded components.
//! 2. `flow`: a root-to-tip tangent per triangle, from distance to a root anchor, refined by the
//!    strokes already painted on the stock texture and smoothed across welded edges.
//! 3. `param`: per lock, a strand parametrization (u across, v root to tip, in model units) from
//!    two least-squares solves (grad v = flow, grad u = flow turned 90 degrees in the surface).
//!    u and v are single values per welded vertex, so they are continuous across every join.
//! 4. `charts` + `pack`: each lock is cut into charts that do not overlap in (u, v); charts are
//!    laid out on the sheet as upright strips (v down the page), at one common texel density.
//! 5. `synth`: color and alpha from (lock, u, v) only: palette ramp, clumps, strand noise, root
//!    shade, highlight band, tip taper. Alpha tips come from a signed distance, color is
//!    extended under transparent texels (gutter), so no fringe shows at cutout edges.
//! 6. `checks`: flow continuity across joins, Lab difference across joins, fringe pixels,
//!    duplicate faces, and turntable sheets (in the command).
use crate::Model;
use std::collections::HashMap;

pub type V3 = [f64; 3];

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn mul(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn len(a: V3) -> f64 {
    dot(a, a).sqrt()
}
pub fn unit(a: V3) -> V3 {
    let l = len(a);
    if l > 1e-30 {
        mul(a, 1.0 / l)
    } else {
        [0.0; 3]
    }
}

/// Which triangles are hair.
pub enum Select {
    /// Every triangle of the mesh.
    All,
    /// Triangles flagged true (indexed by triangle).
    Mask(Vec<bool>),
}

/// The selected hair triangles of one mesh, welded and grouped.
pub struct HairSet {
    pub mesh: usize,
    /// Selected triangle indices into `Model::meshes[mesh].tris`.
    pub tris: Vec<usize>,
    /// Weld id per mesh vertex (same position = same id).
    pub wid: Vec<usize>,
    /// Position per weld id.
    pub wpos: Vec<V3>,
    /// Per selected triangle (index into `tris`): the welded corners.
    pub wt: Vec<[usize; 3]>,
    /// Per selected triangle: the primary of its duplicate group (itself when it is primary).
    pub primary: Vec<usize>,
    /// Per selected triangle: lock id (duplicates share their primary's lock).
    pub lock: Vec<usize>,
    pub nlocks: usize,
    /// Triangles dropped because they have no area.
    pub degenerate: usize,
    /// Duplicate faces: (same corners and same winding, same corners and opposite winding).
    pub dup_same: usize,
    pub dup_flipped: usize,
    /// Welded edges used by more than two primary triangles (cards crossing through each other).
    pub nonmanifold_edges: usize,
}

fn key(p: [f32; 3]) -> (i64, i64, i64) {
    let q = |x: f32| (x as f64 * 1e5).round() as i64;
    (q(p[0]), q(p[1]), q(p[2]))
}

fn find(p: &mut [usize], mut x: usize) -> usize {
    while p[x] != x {
        p[x] = p[p[x]];
        x = p[x];
    }
    x
}

/// Sorted welded edge key.
pub fn ekey(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

impl HairSet {
    pub fn build(model: &Model, mesh: usize, sel: &Select) -> Result<HairSet, String> {
        let m = model.meshes.get(mesh).ok_or("no such mesh")?;
        let mut map: HashMap<(i64, i64, i64), usize> = HashMap::new();
        let mut wpos: Vec<V3> = Vec::new();
        let wid: Vec<usize> = m
            .pos
            .iter()
            .map(|&p| {
                let n = map.len();
                let id = *map.entry(key(p)).or_insert(n);
                if id == wpos.len() {
                    wpos.push([p[0] as f64, p[1] as f64, p[2] as f64]);
                }
                id
            })
            .collect();
        let mut tris = Vec::new();
        let mut degenerate = 0;
        for (ti, t) in m.tris.iter().enumerate() {
            let on = match sel {
                Select::All => true,
                Select::Mask(v) => v.get(ti).copied().unwrap_or(false),
            };
            if !on {
                continue;
            }
            let w = t.map(|v| wid[v as usize]);
            let a = len(cross(sub(wpos[w[1]], wpos[w[0]]), sub(wpos[w[2]], wpos[w[0]])));
            if w[0] == w[1] || w[1] == w[2] || w[0] == w[2] || a < 1e-12 {
                degenerate += 1;
                continue;
            }
            tris.push(ti);
        }
        if tris.is_empty() {
            return Err("no hair triangles selected".into());
        }
        let wt: Vec<[usize; 3]> = tris.iter().map(|&ti| m.tris[ti].map(|v| wid[v as usize])).collect();
        // Duplicate groups by sorted corner triple.
        let mut first: HashMap<[usize; 3], usize> = HashMap::new();
        let mut primary = vec![0; tris.len()];
        let (mut dup_same, mut dup_flipped) = (0, 0);
        for (i, t) in wt.iter().enumerate() {
            let mut k = *t;
            k.sort();
            match first.get(&k) {
                Some(&p) => {
                    primary[i] = p;
                    if same_winding(wt[p], *t) {
                        dup_same += 1;
                    } else {
                        dup_flipped += 1;
                    }
                }
                None => {
                    first.insert(k, i);
                    primary[i] = i;
                }
            }
        }
        // Locks: union primaries over welded edges used by exactly two primaries.
        let mut uses: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (i, t) in wt.iter().enumerate() {
            if primary[i] != i {
                continue;
            }
            for k in 0..3 {
                uses.entry(ekey(t[k], t[(k + 1) % 3])).or_default().push(i);
            }
        }
        let mut par: Vec<usize> = (0..tris.len()).collect();
        let mut nonmanifold_edges = 0;
        for u in uses.values() {
            if u.len() == 2 {
                let (a, b) = (find(&mut par, u[0]), find(&mut par, u[1]));
                par[a] = b;
            } else if u.len() > 2 {
                nonmanifold_edges += 1;
            }
        }
        let mut lock_of_root: HashMap<usize, usize> = HashMap::new();
        let mut lock = vec![0; tris.len()];
        for i in 0..tris.len() {
            if primary[i] != i {
                continue;
            }
            let r = find(&mut par, i);
            let n = lock_of_root.len();
            lock[i] = *lock_of_root.entry(r).or_insert(n);
        }
        for i in 0..tris.len() {
            lock[i] = lock[primary[i]];
        }
        Ok(HairSet { mesh, tris, wid, wpos, wt, primary, lock, nlocks: lock_of_root.len(), degenerate, dup_same, dup_flipped, nonmanifold_edges })
    }

    /// Regroups locks so that triangles flagged in `apart` (per selected triangle) never join a
    /// lock with unflagged ones: a hairline grown into the hair part stays its own lock and does
    /// not merge the cards it touches (which would change their strand solve).
    pub fn relock(&mut self, apart: &[bool]) {
        let n = self.tris.len();
        let mut uses: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for i in 0..n {
            if self.is_primary(i) {
                let t = self.wt[i];
                for k in 0..3 {
                    uses.entry(ekey(t[k], t[(k + 1) % 3])).or_default().push(i);
                }
            }
        }
        let mut par: Vec<usize> = (0..n).collect();
        for u in uses.values() {
            if u.len() == 2 && apart[u[0]] == apart[u[1]] {
                let (a, b) = (find(&mut par, u[0]), find(&mut par, u[1]));
                par[a] = b;
            }
        }
        let mut lock_of_root: HashMap<usize, usize> = HashMap::new();
        for i in 0..n {
            if self.is_primary(i) {
                let r = find(&mut par, i);
                let k = lock_of_root.len();
                self.lock[i] = *lock_of_root.entry(r).or_insert(k);
            }
        }
        for i in 0..n {
            self.lock[i] = self.lock[self.primary[i]];
        }
        self.nlocks = lock_of_root.len();
    }

    pub fn is_primary(&self, i: usize) -> bool {
        self.primary[i] == i
    }

    /// Primary triangles of each lock.
    pub fn lock_tris(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.nlocks];
        for i in 0..self.tris.len() {
            if self.is_primary(i) {
                out[self.lock[i]].push(i);
            }
        }
        out
    }

    pub fn corners(&self, i: usize) -> [V3; 3] {
        self.wt[i].map(|w| self.wpos[w])
    }

    pub fn normal(&self, i: usize) -> V3 {
        let [a, b, c] = self.corners(i);
        unit(cross(sub(b, a), sub(c, a)))
    }

    pub fn area(&self, i: usize) -> f64 {
        let [a, b, c] = self.corners(i);
        0.5 * len(cross(sub(b, a), sub(c, a)))
    }

    pub fn centroid(&self, i: usize) -> V3 {
        let [a, b, c] = self.corners(i);
        mul(add(add(a, b), c), 1.0 / 3.0)
    }
}

/// True when two triangles over the same corners list them in the same cyclic order.
pub fn same_winding(a: [usize; 3], b: [usize; 3]) -> bool {
    (0..3).any(|r| (0..3).all(|k| a[k] == b[(k + r) % 3]))
}

// ---------------------------------------------------------------------------------------------
// Stage 2: flow field
// ---------------------------------------------------------------------------------------------

/// Where hair grows from: a segment (a point when both ends match), optionally draped onto the
/// top of the hair so it lies on the scalp (a part line). Root-to-tip runs away from it.
#[derive(Clone, Copy, Debug)]
pub struct Root {
    pub a: V3,
    pub b: V3,
    /// The segment sampled (and draped); `closest` uses this polyline.
    pub curve: [V3; ROOT_SAMPLES],
}

pub const ROOT_SAMPLES: usize = 33;

impl Root {
    pub fn line(a: V3, b: V3) -> Root {
        Root { a, b, curve: std::array::from_fn(|k| add(a, mul(sub(b, a), k as f64 / (ROOT_SAMPLES - 1) as f64))) }
    }

    /// Moves every sample straight down (or up) onto the highest hair surface under it, so the root
    /// lies on the scalp instead of cutting through it. Samples with no hair under them stay.
    pub fn draped(mut self, hs: &HairSet) -> Root {
        for q in self.curve.iter_mut() {
            let mut best: Option<f64> = None;
            for i in 0..hs.tris.len() {
                if !hs.is_primary(i) {
                    continue;
                }
                let c = hs.corners(i);
                let (x1, z1, x2, z2, x3, z3) = (c[0][0], c[0][2], c[1][0], c[1][2], c[2][0], c[2][2]);
                let den = (z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3);
                if den.abs() < 1e-14 {
                    continue;
                }
                let l1 = ((z2 - z3) * (q[0] - x3) + (x3 - x2) * (q[2] - z3)) / den;
                let l2 = ((z3 - z1) * (q[0] - x3) + (x1 - x3) * (q[2] - z3)) / den;
                let l3 = 1.0 - l1 - l2;
                if l1 < -1e-9 || l2 < -1e-9 || l3 < -1e-9 {
                    continue;
                }
                let y = l1 * c[0][1] + l2 * c[1][1] + l3 * c[2][1];
                best = Some(best.map_or(y, |b: f64| b.max(y)));
            }
            if let Some(y) = best {
                q[1] = y;
            }
        }
        self
    }

    pub fn closest(&self, p: V3) -> V3 {
        let mut best = self.curve[0];
        let mut bd = f64::MAX;
        for w in self.curve.windows(2) {
            let d = sub(w[1], w[0]);
            let l2 = dot(d, d);
            let t = if l2 < 1e-30 { 0.0 } else { (dot(sub(p, w[0]), d) / l2).clamp(0.0, 1.0) };
            let c = add(w[0], mul(d, t));
            let dd = dot(sub(p, c), sub(p, c));
            if dd < bd {
                bd = dd;
                best = c;
            }
        }
        best
    }
    pub fn dist(&self, p: V3) -> f64 {
        len(sub(p, self.closest(p)))
    }

    /// Default for a head of hair: a front-to-back line along the top middle of the selection
    /// (a center part or crown line) over the middle 60% of the depth.
    pub fn auto(hs: &HairSet) -> Root {
        let (lo, hi) = hs.bounds();
        let x = 0.5 * (lo[0] + hi[0]);
        let y = hi[1] - 0.08 * (hi[1] - lo[1]);
        let (z0, z1) = (lo[2] + 0.2 * (hi[2] - lo[2]), lo[2] + 0.8 * (hi[2] - lo[2]));
        Root::line([x, y, z0], [x, y, z1])
    }
}

impl HairSet {
    pub fn bounds(&self) -> (V3, V3) {
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for t in &self.wt {
            for &w in t {
                for k in 0..3 {
                    lo[k] = lo[k].min(self.wpos[w][k]);
                    hi[k] = hi[k].max(self.wpos[w][k]);
                }
            }
        }
        (lo, hi)
    }
}

/// A texture read as RGBA8 (row 0 at the top, UV (0,0) at the top-left like glTF).
pub struct Sheet<'a> {
    pub w: usize,
    pub h: usize,
    pub rgba: &'a [u8],
}

impl Sheet<'_> {
    pub fn px(&self, x: i64, y: i64) -> [f64; 4] {
        let x = x.clamp(0, self.w as i64 - 1) as usize;
        let y = y.clamp(0, self.h as i64 - 1) as usize;
        let o = (y * self.w + x) * 4;
        [self.rgba[o] as f64, self.rgba[o + 1] as f64, self.rgba[o + 2] as f64, self.rgba[o + 3] as f64]
    }
    /// Bilinear sample at UV.
    pub fn bilinear(&self, u: f64, v: f64) -> [f64; 4] {
        let x = u * self.w as f64 - 0.5;
        let y = v * self.h as f64 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let mut o = [0.0; 4];
        for (dx, dy, w) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
            let p = self.px(x0 + dx, y0 + dy);
            for c in 0..4 {
                o[c] += w * p[c];
            }
        }
        o
    }
}

pub struct FlowOpts {
    pub root: Root,
    /// Weight of the stock paint's stroke direction against the root direction (0 = ignore paint).
    pub paint_weight: f64,
    pub smooth_iters: usize,
    /// Neighbors whose flows differ by more than this (degrees) are cut apart (a parting).
    pub part_angle: f64,
}

pub struct Flow {
    /// Per selected triangle: unit root-to-tip tangent in the triangle's plane.
    pub dir: Vec<V3>,
    /// Per selected triangle: direction away from the root, in the plane (unit, or zero edge-on).
    pub away: Vec<V3>,
    /// Per selected triangle: stroke coherence of the stock paint (0..1).
    pub coherence: Vec<f64>,
    /// Per selected triangle: root-to-tip sign or axis is unclear.
    pub ambiguous: Vec<bool>,
    /// Per selected triangle: flow region (a lock split at partings); duplicates follow their primary.
    pub region: Vec<usize>,
    pub region_lock: Vec<usize>,
    /// Welded edges cut as partings.
    pub partings: usize,
}

/// UV of the three corners of a selected triangle in the source mesh.
pub fn tri_uv(model: &Model, hs: &HairSet, i: usize) -> [[f64; 2]; 3] {
    let m = &model.meshes[hs.mesh];
    m.tris[hs.tris[i]].map(|v| [m.uv[v as usize][0] as f64, m.uv[v as usize][1] as f64])
}

/// 3D direction on a triangle for a direction `d` in its UV space.
pub fn uv_dir_to_3d(p: [V3; 3], uv: [[f64; 2]; 3], d: [f64; 2]) -> Option<V3> {
    let (e1, e2) = (sub(p[1], p[0]), sub(p[2], p[0]));
    let (a, b, c, dd) = (uv[1][0] - uv[0][0], uv[2][0] - uv[0][0], uv[1][1] - uv[0][1], uv[2][1] - uv[0][1]);
    let det = a * dd - b * c;
    if det.abs() < 1e-14 {
        return None;
    }
    // [e1 e2] = J [[a b] [c dd]]  ->  J = [e1 e2] M^-1
    let (i00, i01, i10, i11) = (dd / det, -b / det, -c / det, a / det);
    let ju = add(mul(e1, i00), mul(e2, i10));
    let jv = add(mul(e1, i01), mul(e2, i11));
    Some(add(mul(ju, d[0]), mul(jv, d[1])))
}

pub fn project_plane(v: V3, n: V3) -> V3 {
    sub(v, mul(n, dot(v, n)))
}

/// Stroke direction of painted texels over a UV triangle: (unit direction in UV, coherence, texels).
pub fn paint_stroke(sheet: &Sheet, uv: [[f64; 2]; 3]) -> Option<([f64; 2], f64, usize)> {
    let px: [[f64; 2]; 3] = uv.map(|q| [q[0] * sheet.w as f64, q[1] * sheet.h as f64]);
    let lum = |x: i64, y: i64| {
        let p = sheet.px(x, y);
        (0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2], p[3])
    };
    let (mut jxx, mut jxy, mut jyy, mut n) = (0.0, 0.0, 0.0, 0usize);
    let cx = (px[0][0] + px[1][0] + px[2][0]) / 3.0;
    let cy = (px[0][1] + px[1][1] + px[2][1]) / 3.0;
    // Grow the triangle by 2 texels so thin triangles still see strokes.
    let grow = |q: [f64; 2]| {
        let (dx, dy) = (q[0] - cx, q[1] - cy);
        let l = (dx * dx + dy * dy).sqrt().max(1e-9);
        let k = (l + 2.0) / l;
        [cx + dx * k, cy + dy * k]
    };
    let big = px.map(grow);
    let x0 = big.iter().map(|q| q[0]).fold(f64::MAX, f64::min).min(cx - 3.5).floor() as i64;
    let x1 = big.iter().map(|q| q[0]).fold(f64::MIN, f64::max).max(cx + 3.5).ceil() as i64;
    let y0 = big.iter().map(|q| q[1]).fold(f64::MAX, f64::min).min(cy - 3.5).floor() as i64;
    let y1 = big.iter().map(|q| q[1]).fold(f64::MIN, f64::max).max(cy + 3.5).ceil() as i64;
    let inside = |x: f64, y: f64| {
        let s = |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
        let (d0, d1, d2) = (s(big[0], big[1]), s(big[1], big[2]), s(big[2], big[0]));
        (d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0) || (d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0)
    };
    let small = (x1 - x0) * (y1 - y0) < 64;
    for y in y0..=y1 {
        for x in x0..=x1 {
            if !small && !inside(x as f64 + 0.5, y as f64 + 0.5) {
                continue;
            }
            let (_, a) = lum(x, y);
            let (lr, ar) = lum(x + 1, y);
            let (ll, al) = lum(x - 1, y);
            let (ld, ad) = lum(x, y + 1);
            let (lu, au) = lum(x, y - 1);
            if a < 128.0 || ar < 128.0 || al < 128.0 || ad < 128.0 || au < 128.0 {
                continue;
            }
            let gx = 0.5 * (lr - ll);
            let gy = 0.5 * (ld - lu);
            jxx += gx * gx;
            jxy += gx * gy;
            jyy += gy * gy;
            n += 1;
        }
    }
    if n < 6 {
        return None;
    }
    let tr = jxx + jyy;
    if tr < 1e-9 {
        return None;
    }
    let disc = ((jxx - jyy).powi(2) + 4.0 * jxy * jxy).sqrt();
    let (l1, l2) = (0.5 * (tr + disc), 0.5 * (tr - disc));
    let coh = (l1 - l2) / (l1 + l2 + 1e-9);
    // Dominant gradient angle; strokes run perpendicular to it.
    let th = 0.5 * (2.0 * jxy).atan2(jxx - jyy);
    let d = [-th.sin(), th.cos()];
    Some(([d[0] / sheet.w as f64, d[1] / sheet.h as f64], coh, n))
}

/// Welded-edge neighbors between primary triangles (edges used by exactly two primaries).
pub fn neighbors(hs: &HairSet) -> Vec<Vec<(usize, (usize, usize))>> {
    let mut uses: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for i in 0..hs.tris.len() {
        if !hs.is_primary(i) {
            continue;
        }
        let t = hs.wt[i];
        for k in 0..3 {
            uses.entry(ekey(t[k], t[(k + 1) % 3])).or_default().push(i);
        }
    }
    let mut nb = vec![Vec::new(); hs.tris.len()];
    for (e, u) in &uses {
        // Same lock only: a no-op for locks built from welded edges, but it keeps a lock split off
        // by `relock` (a grown hairline) from smoothing its flow into the cards it touches.
        if u.len() == 2 && hs.lock[u[0]] == hs.lock[u[1]] {
            nb[u[0]].push((u[1], *e));
            nb[u[1]].push((u[0], *e));
        }
    }
    // HashMap order changes every run; callers that take the first match need a fixed order.
    for l in &mut nb {
        l.sort_unstable();
    }
    nb
}

/// Rotates a tangent of triangle j about the shared edge `e` into triangle i's plane (unfolding the
/// hinge; a neighbor wound the other way has its normal flipped first).
pub fn hinge(hs: &HairSet, j: usize, i: usize, e: (usize, usize), d: V3) -> V3 {
    let ax = unit(sub(hs.wpos[e.1], hs.wpos[e.0]));
    let fwd = |t: [usize; 3]| (0..3).any(|k| t[k] == e.0 && t[(k + 1) % 3] == e.1);
    let consistent = fwd(hs.wt[i]) != fwd(hs.wt[j]);
    let nj = if consistent { hs.normal(j) } else { mul(hs.normal(j), -1.0) };
    let ni = hs.normal(i);
    let (pj, pi) = (cross(nj, ax), cross(ni, ax));
    unit(add(mul(ax, dot(d, ax)), mul(pi, dot(d, pj))))
}

/// One flow solve over the whole hair surface (all locks together), on top of `FlowOpts`.
#[derive(Clone, Debug, Default)]
pub struct GlobalFlow {
    /// Weight of nearby triangles of OTHER locks (layered cards) in the smoothing; 0 = per lock.
    pub couple: f64,
    /// How far (model units) a triangle looks for other locks' triangles.
    pub couple_radius: f64,
    /// Guide strokes drawn on the surface, root to tip.
    pub comb: Vec<CombStroke>,
    /// Flip a whole lock when its flow runs away from where the stock alpha fades (its tips).
    pub tip_flip: bool,
    /// Paint weight where the stock paint runs within `paint_strong_deg` of the away-from-root
    /// direction (0 = `FlowOpts::paint_weight` everywhere). There the paint is trusted more, so
    /// strands follow the painted ones; where the two disagree more (whorls, crown sides) the
    /// plain weight keeps the strand coordinates solvable. Cards seen edge-on from the root
    /// (the away direction mostly out of plane) keep the plain weight too.
    pub paint_strong: f64,
    pub paint_strong_deg: f64,
}

/// A comb stroke: rest-pose points from root to tip, its weight (1 = override the stock paint) and
/// its reach (model units).
#[derive(Clone, Debug)]
pub struct CombStroke {
    pub points: Vec<V3>,
    pub weight: f64,
    pub radius: f64,
}

#[derive(Clone, Debug, Default)]
pub struct FlowAgree {
    /// Nearby triangle pairs from different locks with near-parallel planes.
    pub pairs: usize,
    /// Angle (degrees) between the two flows of those pairs, per lock solve and after the global solve.
    pub median_before: f64,
    pub p90_before: f64,
    pub over45_before: usize,
    pub median_after: f64,
    pub p90_after: f64,
    pub over45_after: usize,
    /// Triangles a comb stroke reached.
    pub combed: usize,
    /// Locks whose flow runs away from their faded (tip) ends, and how many were flipped.
    pub tip_against: Vec<usize>,
    pub tip_flipped: usize,
}

pub fn flow(model: &Model, hs: &HairSet, stock: &Sheet, o: &FlowOpts) -> Flow {
    flow_global(model, hs, stock, o, &GlobalFlow::default()).0
}

/// `flow`, solved over all locks at once: nearby layered cards pull toward each other (`couple`),
/// comb strokes pin the direction where they lie, and each lock's direction is checked against
/// where its stock alpha fades (the tips).
pub fn flow_global(model: &Model, hs: &HairSet, stock: &Sheet, o: &FlowOpts, g: &GlobalFlow) -> (Flow, FlowAgree) {
    let n = hs.tris.len();
    let mut dir = vec![[0.0; 3]; n];
    let mut away = vec![[0.0; 3]; n];
    let mut coherence = vec![0.0; n];
    let mut ambiguous = vec![false; n];
    let (strong, strong_cos) = (g.paint_strong, g.paint_strong_deg.to_radians().cos());
    for i in 0..n {
        if !hs.is_primary(i) {
            continue;
        }
        let nrm = hs.normal(i);
        let c = hs.centroid(i);
        let r = sub(c, o.root.closest(c));
        let g = project_plane(r, nrm);
        let g_unit = unit(g);
        away[i] = g_unit;
        let gfrac = len(g) / len(r).max(1e-12); // 1 = the away direction lies in the plane
        let mut f = g_unit;
        let mut amb = gfrac < 0.25;
        if o.paint_weight > 0.0 {
            if let Some((d, coh, _)) = paint_stroke(stock, tri_uv(model, hs, i)) {
                coherence[i] = coh;
                if let Some(d3) = uv_dir_to_3d(hs.corners(i), tri_uv(model, hs, i), d) {
                    let mut p = unit(project_plane(d3, nrm));
                    if dot(p, g_unit) < 0.0 {
                        p = mul(p, -1.0);
                    }
                    // Only where the away direction lies well in the card (not edge-on near the root
                    // line, where the crown sides turn and a strong paint seed makes flow breaks).
                    let near = strong > 0.0 && dot(p, g_unit) >= strong_cos && gfrac >= 0.6;
                    let pw = if near { strong.max(o.paint_weight) } else { o.paint_weight };
                    let w = pw * ((coh - 0.08) / 0.3).clamp(0.0, 1.0);
                    let wg = (1.0 - w) * gfrac.max(0.05);
                    f = unit(add(mul(p, w), mul(g_unit, wg)));
                    amb = gfrac < 0.25 && dot(p, g_unit).abs() < 0.3;
                }
            }
        }
        if len(f) < 0.5 {
            // Edge-on to the root and no paint: fall back to the plane's steepest descent.
            f = unit(project_plane([0.0, -1.0, 0.0], nrm));
            amb = true;
        }
        dir[i] = f;
        ambiguous[i] = amb;
    }
    let dir_plain = dir.clone();
    let prim: Vec<usize> = (0..n).filter(|&i| hs.is_primary(i)).collect();
    // Comb pins: per triangle, the strokes' direction near its centroid.
    let mut pin = vec![([0.0; 3], 0.0f64); n];
    let mut combed = 0;
    for &i in &prim {
        let c = hs.centroid(i);
        let (mut acc, mut ws) = ([0.0; 3], 0.0);
        for st in &g.comb {
            for w2 in st.points.windows(2) {
                let seg = sub(w2[1], w2[0]);
                let l2 = dot(seg, seg);
                if l2 < 1e-18 || st.radius <= 0.0 {
                    continue;
                }
                let t = (dot(sub(c, w2[0]), seg) / l2).clamp(0.0, 1.0);
                let d = len(sub(c, add(w2[0], mul(seg, t))));
                if d < st.radius {
                    let w = st.weight * (1.0 - d / st.radius).powi(2);
                    acc = add(acc, mul(unit(seg), w));
                    ws += w;
                }
            }
        }
        let p = unit(project_plane(acc, hs.normal(i)));
        if ws > 1e-6 && len(p) > 0.5 {
            let k = ws.min(1.0);
            pin[i] = (p, k);
            combed += 1;
            let d = unit(add(mul(dir[i], 1.0 - k), mul(p, k)));
            dir[i] = if len(d) > 0.5 { d } else { p };
        }
    }
    // Other locks nearby with near-parallel planes (layered cards): the coupling and the check.
    let mut near: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    if g.couple_radius > 0.0 {
        let cen: Vec<V3> = (0..n).map(|i| hs.centroid(i)).collect();
        for (a, &i) in prim.iter().enumerate() {
            for &j in &prim[a + 1..] {
                if hs.lock[i] == hs.lock[j] {
                    continue;
                }
                let d = len(sub(cen[i], cen[j]));
                if d < g.couple_radius && dot(hs.normal(i), hs.normal(j)).abs() > 0.7 {
                    let w = 1.0 - d / g.couple_radius;
                    near[i].push((j, w));
                    near[j].push((i, w));
                }
            }
        }
    }
    // Smooth across welded edges, never across a parting; optionally toward layered cards of other
    // locks and comb pins.
    let nb = neighbors(hs);
    let cos_part = o.part_angle.to_radians().cos();
    let smooth_all = |start: &[V3], couple: f64, pins: bool| -> Vec<V3> {
        let mut dir = start.to_vec();
        for _ in 0..o.smooth_iters {
            let prev = dir.clone();
            for &i in &prim {
                let nrm = hs.normal(i);
                let mut acc = mul(prev[i], 2.0);
                for &(j, e) in &nb[i] {
                    let q = hinge(hs, j, i, e, prev[j]);
                    if dot(q, prev[i]) > cos_part {
                        acc = add(acc, q);
                    }
                }
                if couple > 0.0 {
                    for &(j, w) in &near[i] {
                        let q = unit(project_plane(prev[j], nrm));
                        if dot(q, prev[i]) > 0.0 {
                            acc = add(acc, mul(q, couple * w));
                        }
                    }
                }
                if pins && pin[i].1 > 0.0 {
                    acc = add(acc, mul(pin[i].0, 6.0 * pin[i].1));
                }
                let a = unit(project_plane(acc, nrm));
                if len(a) > 0.5 {
                    dir[i] = a;
                }
            }
        }
        dir
    };
    let agree = |d: &[V3]| -> (f64, f64, usize, usize) {
        let mut v = Vec::new();
        for &i in &prim {
            for &(j, _) in &near[i] {
                if j > i {
                    let q = unit(project_plane(d[j], hs.normal(i)));
                    if len(q) > 0.5 {
                        v.push(dot(d[i], q).clamp(-1.0, 1.0).acos().to_degrees());
                    }
                }
            }
        }
        v.sort_by(f64::total_cmp);
        let at = |f: f64| v.get(((v.len().max(1) - 1) as f64 * f).round() as usize).copied().unwrap_or(0.0);
        (at(0.5), at(0.9), v.iter().filter(|&&a| a > 45.0).count(), v.len())
    };
    let before = smooth_all(&dir_plain, 0.0, false);
    let mut dir = if g.couple > 0.0 || combed > 0 { smooth_all(&dir, g.couple, true) } else { before.clone() };
    let (mb, pb, ob, npairs) = agree(&before);
    let (ma, pa, oa, _) = agree(&dir);
    // Tip check: where the stock alpha fades along the flow, the flow should point that way.
    let mut vote = vec![(0.0f64, 0.0f64); hs.nlocks];
    for &i in &prim {
        let uv = tri_uv(model, hs, i);
        let cu = [(uv[0][0] + uv[1][0] + uv[2][0]) / 3.0, (uv[0][1] + uv[1][1] + uv[2][1]) / 3.0];
        let a: [f64; 3] = std::array::from_fn(|k| {
            let q = [uv[k][0] * 0.85 + cu[0] * 0.15, uv[k][1] * 0.85 + cu[1] * 0.15];
            stock.px((q[0] * stock.w as f64).floor() as i64, (q[1] * stock.h as f64).floor() as i64)[3] / 255.0
        });
        let drop = a.iter().cloned().fold(f64::MIN, f64::max) - a.iter().cloned().fold(f64::MAX, f64::min);
        if drop < 0.4 {
            continue;
        }
        let (gr, _) = basis_grads(hs.corners(i));
        let grad = (0..3).fold([0.0; 3], |s, k| add(s, mul(gr[k], a[k])));
        let al = dot(dir[i], unit(mul(grad, -1.0)));
        if al > 0.5 {
            vote[hs.lock[i]].0 += drop;
        } else if al < -0.5 {
            vote[hs.lock[i]].1 += drop;
        }
    }
    let tip_against: Vec<usize> = (0..hs.nlocks).filter(|&l| vote[l].1 > 2.0 * vote[l].0 && vote[l].1 > 1.0).collect();
    let mut tip_flipped = 0;
    if g.tip_flip {
        for &l in &tip_against {
            for &i in &prim {
                if hs.lock[i] == l {
                    dir[i] = mul(dir[i], -1.0);
                }
            }
            tip_flipped += 1;
        }
    }
    let agreement = FlowAgree { pairs: npairs, median_before: mb, p90_before: pb, over45_before: ob, median_after: ma, p90_after: pa, over45_after: oa, combed, tip_against, tip_flipped };
    // Regions: locks cut where neighbor flows disagree.
    let mut par: Vec<usize> = (0..n).collect();
    let mut partings = 0;
    for i in 0..n {
        for &(j, e) in &nb[i] {
            if j < i {
                continue;
            }
            if dot(dir[i], hinge(hs, j, i, e, dir[j])) > cos_part {
                let (a, b) = (find(&mut par, i), find(&mut par, j));
                par[a] = b;
            } else {
                partings += 1;
            }
        }
    }
    let mut ids: HashMap<usize, usize> = HashMap::new();
    let mut region = vec![0; n];
    let mut region_lock = Vec::new();
    for i in 0..n {
        if !hs.is_primary(i) {
            continue;
        }
        let r = find(&mut par, i);
        let k = ids.len();
        let id = *ids.entry(r).or_insert(k);
        if id == region_lock.len() {
            region_lock.push(hs.lock[i]);
        }
        region[i] = id;
    }
    for i in 0..n {
        let p = hs.primary[i];
        region[i] = region[p];
        dir[i] = dir[p];
        away[i] = away[p];
        coherence[i] = coherence[p];
        ambiguous[i] = ambiguous[p];
    }
    (Flow { dir, away, coherence, ambiguous, region, region_lock, partings }, agreement)
}

// ---------------------------------------------------------------------------------------------
// Stage 3: strand parametrization
// ---------------------------------------------------------------------------------------------

/// Gradients of the three barycentric basis functions of a triangle (in its plane), and its area.
pub fn basis_grads(p: [V3; 3]) -> ([V3; 3], f64) {
    let n2 = cross(sub(p[1], p[0]), sub(p[2], p[0]));
    let a2 = len(n2);
    let n = mul(n2, 1.0 / a2.max(1e-30));
    let g = std::array::from_fn(|k| {
        let e = sub(p[(k + 2) % 3], p[(k + 1) % 3]);
        mul(cross(n, e), 1.0 / a2.max(1e-30))
    });
    (g, 0.5 * a2)
}

/// Least squares: sum over triangles of area * |grad x - target|^2 + a weak pull toward `prior`
/// (weight `lambda`, area-weighted). `tris` index `pos`. Jacobi-preconditioned conjugate gradients.
pub fn solve_gradient(pos: &[V3], tris: &[[usize; 3]], target: &[V3], prior: &[f64], lambda: f64) -> Vec<f64> {
    solve_gradient_aniso(pos, tris, target, None, prior, &vec![lambda; pos.len()])
}

/// As `solve_gradient`, with an extra penalty `w * (grad x . f)^2` per triangle (`along` = (f, w)):
/// keeps the level lines of x running along f even where |target| cannot be met everywhere.
pub fn solve_gradient_aniso(pos: &[V3], tris: &[[usize; 3]], target: &[V3], along: Option<(&[V3], f64)>, prior: &[f64], pull: &[f64]) -> Vec<f64> {
    let nv = pos.len();
    let mut rows: Vec<HashMap<usize, f64>> = vec![HashMap::new(); nv];
    let mut b = vec![0.0; nv];
    let mut mass = vec![0.0; nv];
    let mut total = 0.0f64;
    for (t, tri) in tris.iter().enumerate() {
        let (g, a) = basis_grads(tri.map(|v| pos[v]));
        if a < 1e-14 {
            continue;
        }
        total += a;
        for j in 0..3 {
            for k in 0..3 {
                let extra = along.map_or(0.0, |(f, w)| w * dot(g[j], f[t]) * dot(g[k], f[t]));
                *rows[tri[j]].entry(tri[k]).or_insert(0.0) += a * (dot(g[j], g[k]) + extra);
            }
            let extra = along.map_or(0.0, |(f, w)| w * dot(g[j], f[t]) * dot(target[t], f[t]));
            b[tri[j]] += a * (dot(g[j], target[t]) + extra);
            mass[tri[j]] += a / 3.0;
        }
    }
    // Pull scaled so it is weak against the gradient term (which has units of area / length^2).
    let ext = total.sqrt().max(1e-12);
    for v in 0..nv {
        let w = pull[v] * 1e-3 * mass[v] / (ext * ext) + 1e-12;
        *rows[v].entry(v).or_insert(0.0) += w;
        b[v] += w * prior[v];
    }
    // Sorted so the sums run in the same order every run.
    let rows: Vec<Vec<(usize, f64)>> = rows
        .into_iter()
        .map(|r| {
            let mut v: Vec<(usize, f64)> = r.into_iter().collect();
            v.sort_unstable_by_key(|e| e.0);
            v
        })
        .collect();
    let mulv = |x: &[f64], out: &mut [f64]| {
        for (i, r) in rows.iter().enumerate() {
            out[i] = r.iter().map(|&(j, a)| a * x[j]).sum();
        }
    };
    let diag: Vec<f64> = rows.iter().enumerate().map(|(i, r)| r.iter().find(|e| e.0 == i).map(|e| e.1).unwrap_or(1.0)).collect();
    let mut x = prior.to_vec();
    let mut ax = vec![0.0; nv];
    mulv(&x, &mut ax);
    let mut r: Vec<f64> = (0..nv).map(|i| b[i] - ax[i]).collect();
    let mut z: Vec<f64> = (0..nv).map(|i| r[i] / diag[i]).collect();
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let b0 = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
    let mut ap = vec![0.0; nv];
    for _ in 0..(50 * nv).max(1000) {
        if r.iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-12 * b0 {
            break;
        }
        mulv(&p, &mut ap);
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if pap.abs() < 1e-300 {
            break;
        }
        let al = rz / pap;
        for i in 0..nv {
            x[i] += al * p[i];
            r[i] -= al * ap[i];
            z[i] = r[i] / diag[i];
        }
        let rz2: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let be = rz2 / rz;
        rz = rz2;
        for i in 0..nv {
            p[i] = z[i] + be * p[i];
        }
    }
    x
}

/// A local vertex numbering for a set of selected triangles: (weld id -> local), triangles, positions.
pub fn local_mesh(hs: &HairSet, tl: &[usize]) -> (HashMap<usize, usize>, Vec<[usize; 3]>, Vec<V3>) {
    let mut local: HashMap<usize, usize> = HashMap::new();
    let mut pos = Vec::new();
    let tris = tl
        .iter()
        .map(|&i| {
            hs.wt[i].map(|w| {
                let n = local.len();
                *local.entry(w).or_insert_with(|| {
                    pos.push(hs.wpos[w]);
                    n
                })
            })
        })
        .collect();
    (local, tris, pos)
}

/// Strand coordinates per selected triangle corner: [u, v] in model units.
/// v (root to tip) is solved per lock, so it is continuous across every welded edge of a lock;
/// u (across) is solved per flow region, so it is continuous everywhere except at partings.
pub struct Param {
    pub uv: Vec<[[f64; 2]; 3]>,
    /// Per primary triangle: angle (degrees) between grad v and the flow, and between grad u and
    /// the flow turned 90 degrees in the surface.
    pub v_err: Vec<f64>,
    pub u_err: Vec<f64>,
    /// |grad u| per corner (area-weighted over the corner's region): where strands fan out it
    /// drops below 1, and the synthesizer adds clump octaves so clumps keep their width.
    pub dens: Vec<[f64; 3]>,
}

pub fn ang(a: V3, b: V3) -> f64 {
    dot(unit(a), unit(b)).clamp(-1.0, 1.0).acos().to_degrees()
}

/// A winding sign per primary triangle so that `sign * normal` is consistent across welded edges
/// within each lock (cards are often wound inconsistently; the u target needs one side).
pub fn orient(hs: &HairSet) -> Vec<f64> {
    let nb = neighbors(hs);
    let n = hs.tris.len();
    let mut s = vec![0.0f64; n];
    for start in 0..n {
        if !hs.is_primary(start) || s[start] != 0.0 {
            continue;
        }
        s[start] = 1.0;
        let mut q = std::collections::VecDeque::from([start]);
        while let Some(i) = q.pop_front() {
            for &(j, e) in &nb[i] {
                if s[j] != 0.0 || hs.lock[j] != hs.lock[i] {
                    continue;
                }
                let fwd = |t: [usize; 3]| (0..3).any(|k| t[k] == e.0 && t[(k + 1) % 3] == e.1);
                s[j] = if fwd(hs.wt[i]) != fwd(hs.wt[j]) { s[i] } else { -s[i] };
                q.push_back(j);
            }
        }
    }
    s
}

/// Weight of "u stays constant along the flow" against the weak across-flow pull.
fn along_w() -> f64 {
    300.0
}

/// Splits every flow region that lies on both sides of the vertical plane through the root line
/// (a center part). Hair parts there, so each side gets its own across-coordinate; returns how
/// many regions were split. Needs a root line, not a point.
pub fn split_at_root_plane(hs: &HairSet, fl: &mut Flow, root: &Root) -> usize {
    let seg = sub(root.b, root.a);
    let lat = cross(unit(seg), [0.0, 1.0, 0.0]);
    if len(seg) < 1e-9 || len(lat) < 0.2 {
        return 0;
    }
    let lat = unit(lat);
    let nr = fl.region_lock.len();
    let mut count = vec![(0usize, 0usize); nr];
    let side = |i: usize| dot(sub(hs.centroid(i), root.a), lat) < 0.0;
    for i in 0..hs.tris.len() {
        if hs.is_primary(i) {
            let c = &mut count[fl.region[i]];
            if side(i) { c.1 += 1 } else { c.0 += 1 }
        }
    }
    let mut new_id = vec![usize::MAX; nr];
    for r in 0..nr {
        if count[r].0 > 0 && count[r].1 > 0 {
            new_id[r] = fl.region_lock.len();
            fl.region_lock.push(fl.region_lock[r]);
        }
    }
    let old = fl.region.clone();
    for i in 0..hs.tris.len() {
        let p = hs.primary[i];
        let r = old[p];
        if new_id[r] != usize::MAX && side(p) {
            fl.region[i] = new_id[r];
        }
    }
    // Duplicates follow their primary (set above from the primary's side).
    new_id.iter().filter(|&&x| x != usize::MAX).count()
}

pub fn param(hs: &HairSet, fl: &Flow, root: &Root) -> Param {
    let n = hs.tris.len();
    let mut uv = vec![[[0.0; 2]; 3]; n];
    let sg = orient(hs);
    for tl in hs.lock_tris() {
        let (_, tris, pos) = local_mesh(hs, &tl);
        let prior: Vec<f64> = pos.iter().map(|&p| root.dist(p)).collect();
        let target: Vec<V3> = tl.iter().map(|&i| fl.dir[i]).collect();
        let v = solve_gradient(&pos, &tris, &target, &prior, 1.0);
        for (k, &i) in tl.iter().enumerate() {
            for c in 0..3 {
                uv[i][c][1] = v[tris[k][c]];
            }
        }
    }
    let mut by_region: Vec<Vec<usize>> = vec![Vec::new(); fl.region_lock.len()];
    for i in 0..n {
        if hs.is_primary(i) {
            by_region[fl.region[i]].push(i);
        }
    }
    for tl in &by_region {
        if tl.is_empty() {
            continue;
        }
        let (_, tris, pos) = local_mesh(hs, tl);
        let target: Vec<V3> = tl.iter().map(|&i| cross(mul(hs.normal(i), sg[i]), fl.dir[i])).collect();
        let mean_across = unit(target.iter().fold([0.0; 3], |a, &t| add(a, t)));
        let along: Vec<V3> = tl.iter().map(|&i| fl.dir[i]).collect();
        // v per local vertex (from the solve above).
        let mut vloc = vec![0.0; pos.len()];
        for (k, &i) in tl.iter().enumerate() {
            for c in 0..3 {
                vloc[tris[k][c]] = uv[i][c][1];
            }
        }
        let (vmin, vmax) = vloc.iter().fold((f64::MAX, f64::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
        // Roots of this region: its lowest-v vertices. u there = position along the root line
        // (its principal axis), then carried along the flow: strands keep their u from root to tip.
        let anchors: Vec<usize> = (0..pos.len()).filter(|&k| vloc[k] <= vmin + 0.12 * (vmax - vmin)).collect();
        let ac = mul(anchors.iter().fold([0.0; 3], |a, &k| add(a, pos[k])), 1.0 / anchors.len().max(1) as f64);
        let mut axis = mean_across;
        for _ in 0..20 {
            let mut nx = [0.0; 3];
            for &k in &anchors {
                let d = sub(pos[k], ac);
                nx = add(nx, mul(d, dot(d, axis)));
            }
            if len(nx) < 1e-20 {
                break;
            }
            axis = unit(nx);
        }
        if dot(axis, mean_across) < 0.0 {
            axis = mul(axis, -1.0);
        }
        // Flow per local vertex; at the ends of the root line hair fans out along the line itself
        // (a whorl), and pinning u there would turn strands sideways, so those roots stay free.
        let mut vflow = vec![[0.0; 3]; pos.len()];
        for (k, _) in tl.iter().enumerate() {
            for c in 0..3 {
                vflow[tris[k][c]] = add(vflow[tris[k][c]], along[k]);
            }
        }
        let _ = &vflow;
        // Root coordinate around the root line: distance along it, and past its ends the arc
        // around the end (hair fans out there). Same on both sides of the line, so u matches
        // across a center part.
        let seg = sub(root.b, root.a);
        let seg_len = len(seg);
        let lat = cross(unit(seg), [0.0, 1.0, 0.0]);
        // Signed by the region's side, so u stays monotone for triangles that straddle the plane.
        let side_sign = if pos.iter().map(|&p| dot(sub(p, root.a), lat)).sum::<f64>() < 0.0 { -1.0 } else { 1.0 };
        let stadium = |p: V3| -> f64 {
            let w = sub(p, root.a);
            let ax = dot(w, unit(seg));
            let l = side_sign * dot(w, unit(lat));
            if ax < 0.0 {
                let r = (ax * ax + l * l).sqrt();
                -r * (-ax).atan2(l)
            } else if ax > seg_len {
                let ex = ax - seg_len;
                let r = (ex * ex + l * l).sqrt();
                seg_len + r * ex.atan2(l)
            } else {
                ax
            }
        };
        let use_stadium = seg_len > 1e-9 && len(lat) > 0.2;
        let c0 = mul(pos.iter().fold([0.0; 3], |a, &p| add(a, p)), 1.0 / pos.len() as f64);
        let spread = |ks: &[usize], ax: V3| {
            let v: Vec<f64> = ks.iter().map(|&k| dot(pos[k], ax)).collect();
            v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min)
        };
        let all: Vec<usize> = (0..pos.len()).collect();
        let st: Vec<f64> = if use_stadium { pos.iter().map(|&p| stadium(p)).collect() } else { vec![0.0; pos.len()] };
        let st_spread = anchors.iter().map(|&k| st[k]).fold(f64::MIN, f64::max) - anchors.iter().map(|&k| st[k]).fold(f64::MAX, f64::min);
        let across = spread(&all, mean_across);
        let stadium_ok = use_stadium && anchors.len() >= 2 && st_spread >= 0.25 * across;
        let transport = stadium_ok || (anchors.len() >= 2 && spread(&anchors, axis) >= 0.25 * across);
        if std::env::var("HAIR_DEBUG").is_ok() {
            eprintln!("region tris {} anchors {} stadium {} ({:.3} vs {:.3}) transport {} side {}", tl.len(), anchors.len(), stadium_ok, st_spread, across, transport, side_sign);
        }
        let u = if transport {
            let prior: Vec<f64> = if stadium_ok { st.clone() } else { pos.iter().map(|&p| dot(sub(p, ac), axis)).collect() };
            let mut pull = vec![0.0; pos.len()];
            for &k in &anchors {
                pull[k] = 1e8;
            }
            // A weak pull toward "u grows across the flow" settles whorls, where carrying u along
            // the flow from the roots leaves it undetermined.
            // Same sense as the anchors (u can be mirrored on one side of a part).
            let agree: f64 = tl.iter().enumerate().map(|(k, _)| {
                let (g, a) = basis_grads(tris[k].map(|v| pos[v]));
                a * dot((0..3).fold([0.0; 3], |acc, c| add(acc, mul(g[c], prior[tris[k][c]]))), target[k])
            }).sum();
            let ws = if agree < 0.0 { -0.3 } else { 0.3 };
            let weak: Vec<V3> = target.iter().map(|&t| mul(t, ws)).collect();
            let u = solve_gradient_aniso(&pos, &tris, &weak, Some((&along, along_w())), &prior, &pull);
            // Keep u increasing across the flow the same way as the target.
            let s: f64 = tl.iter().enumerate().map(|(k, &_i)| {
                let (g, a) = basis_grads(tris[k].map(|v| pos[v]));
                a * dot((0..3).fold([0.0; 3], |acc, c| add(acc, mul(g[c], u[tris[k][c]]))), target[k])
            }).sum();
            if s < 0.0 { u.iter().map(|x| -x).collect() } else { u }
        } else {
            let prior: Vec<f64> = pos.iter().map(|&p| dot(sub(p, c0), mean_across)).collect();
            solve_gradient_aniso(&pos, &tris, &target, Some((&along, 40.0)), &prior, &vec![1.0; pos.len()])
        };
        for (k, &i) in tl.iter().enumerate() {
            for c in 0..3 {
                uv[i][c][0] = u[tris[k][c]];
            }
        }
    }
    finish_param(hs, fl, &sg, uv)
}

/// Errors against the flow, strand density and duplicate corners for strand coordinates `uv`
/// (per selected triangle corner; duplicates are overwritten from their primaries).
pub fn finish_param(hs: &HairSet, fl: &Flow, sg: &[f64], mut uv: Vec<[[f64; 2]; 3]>) -> Param {
    let n = hs.tris.len();
    let mut v_err = vec![0.0; n];
    let mut u_err = vec![0.0; n];
    for i in 0..n {
        let p = hs.primary[i];
        if p != i {
            continue;
        }
        let (g, _) = basis_grads(hs.corners(i));
        let gv = (0..3).fold([0.0; 3], |a, k| add(a, mul(g[k], uv[i][k][1])));
        let gu = (0..3).fold([0.0; 3], |a, k| add(a, mul(g[k], uv[i][k][0])));
        v_err[i] = ang(gv, fl.dir[i]);
        // Sign-blind: mirrored u draws the same strands (they are the lines of constant u).
        let e = ang(gu, cross(mul(hs.normal(i), sg[i]), fl.dir[i]));
        u_err[i] = e.min(180.0 - e);
    }
    let mut acc: HashMap<(usize, usize), (f64, f64)> = HashMap::new();
    for i in 0..n {
        if hs.primary[i] != i {
            continue;
        }
        let (g, a) = basis_grads(hs.corners(i));
        let gu = len((0..3).fold([0.0; 3], |s, k| add(s, mul(g[k], uv[i][k][0]))));
        for c in 0..3 {
            let e = acc.entry((fl.region[i], hs.wt[i][c])).or_insert((0.0, 0.0));
            e.0 += a * gu;
            e.1 += a;
        }
    }
    let mut dens = vec![[1.0; 3]; n];
    for i in 0..n {
        let p = hs.primary[i];
        for c in 0..3 {
            if let Some(e) = acc.get(&(fl.region[p], hs.wt[i][c])) {
                dens[i][c] = e.0 / e.1.max(1e-30);
            }
        }
    }
    for i in 0..n {
        let p = hs.primary[i];
        if p != i {
            // Duplicates take the primary's coordinates corner by corner (matched by weld id).
            let pw = hs.wt[p];
            for c in 0..3 {
                let k = pw.iter().position(|&w| w == hs.wt[i][c]).unwrap_or(c);
                uv[i][c] = uv[p][k];
            }
            v_err[i] = v_err[p];
            u_err[i] = u_err[p];
        }
    }
    Param { uv, v_err, u_err, dens }
}

// ---------------------------------------------------------------------------------------------
// Stage 4: charts and packing
// ---------------------------------------------------------------------------------------------

/// A piece of one flow region that does not overlap itself in (u, v). Laid out upright on the sheet.
pub struct Chart {
    /// Primary triangles.
    pub tris: Vec<usize>,
    pub region: usize,
    pub lock: usize,
    /// (u, v) bounds.
    pub lo: [f64; 2],
    pub hi: [f64; 2],
    /// Texel position of `lo` on the sheet (set by `pack`).
    pub at: [f64; 2],
}

fn tri_cells(p: [[f64; 2]; 3], cell: f64) -> Vec<(i64, i64)> {
    let q = p.map(|a| [a[0] / cell, a[1] / cell]);
    let den = (q[1][1] - q[2][1]) * (q[0][0] - q[2][0]) + (q[2][0] - q[1][0]) * (q[0][1] - q[2][1]);
    if den.abs() < 1e-12 {
        return vec![];
    }
    let x0 = q.iter().map(|a| a[0]).fold(f64::MAX, f64::min).floor() as i64;
    let x1 = q.iter().map(|a| a[0]).fold(f64::MIN, f64::max).ceil() as i64;
    let y0 = q.iter().map(|a| a[1]).fold(f64::MAX, f64::min).floor() as i64;
    let y1 = q.iter().map(|a| a[1]).fold(f64::MIN, f64::max).ceil() as i64;
    let mut out = Vec::new();
    // Cell centers strictly inside (a shrink of 0.15 cell keeps shared edges from counting).
    let eps = 0.15 / ((x1 - x0).max(y1 - y0).max(1) as f64);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            let l0 = ((q[1][1] - q[2][1]) * (px - q[2][0]) + (q[2][0] - q[1][0]) * (py - q[2][1])) / den;
            let l1 = ((q[2][1] - q[0][1]) * (px - q[2][0]) + (q[0][0] - q[2][0]) * (py - q[2][1])) / den;
            if l0 > eps && l1 > eps && 1.0 - l0 - l1 > eps {
                out.push((x, y));
            }
        }
    }
    out
}

/// Cuts every flow region into charts that do not overlap in (u, v) (greedy, breadth first).
/// `cell` is the overlap test's grid size in model units.
pub fn charts(hs: &HairSet, fl: &Flow, pr: &Param, cell: f64) -> (Vec<Chart>, Vec<usize>) {
    let n = hs.tris.len();
    let nb = neighbors(hs);
    let mut chart_of = vec![usize::MAX; n];
    let mut out: Vec<Chart> = Vec::new();
    let mut order: Vec<usize> = (0..n).filter(|&i| hs.is_primary(i)).collect();
    order.sort_by(|&a, &b| hs.area(b).total_cmp(&hs.area(a)));
    for &seed in &order {
        if chart_of[seed] != usize::MAX {
            continue;
        }
        let id = out.len();
        let mut cells: std::collections::HashSet<(i64, i64)> = std::collections::HashSet::new();
        let mut tris = Vec::new();
        let mut queue = std::collections::VecDeque::from([seed]);
        let mut tried = std::collections::HashSet::new();
        while let Some(i) = queue.pop_front() {
            if chart_of[i] != usize::MAX || !tried.insert(i) {
                continue;
            }
            let c = tri_cells(pr.uv[i], cell);
            if c.iter().any(|k| cells.contains(k)) {
                continue;
            }
            cells.extend(c);
            chart_of[i] = id;
            tris.push(i);
            for &(j, _) in &nb[i] {
                if chart_of[j] == usize::MAX && fl.region[j] == fl.region[i] {
                    queue.push_back(j);
                }
            }
        }
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for &i in &tris {
            for c in 0..3 {
                for k in 0..2 {
                    lo[k] = lo[k].min(pr.uv[i][c][k]);
                    hi[k] = hi[k].max(pr.uv[i][c][k]);
                }
            }
        }
        out.push(Chart { tris, region: fl.region[seed], lock: hs.lock[seed], lo, hi, at: [0.0; 2] });
    }
    for i in 0..n {
        chart_of[i] = chart_of[hs.primary[i]];
    }
    (out, chart_of)
}

/// Texels on the sheet that something else uses: every triangle of every mesh on the sheet that is
/// not hair (conservatively rasterized), plus kept boxes (UV 0..1: x0, y0, x1, y1).
pub fn occupied(model: &Model, sheet_image: usize, hs: &HairSet, size: usize, keep: &[[f64; 4]]) -> Vec<bool> {
    let mut occ = vec![false; size * size];
    let mut hair = vec![false; model.meshes[hs.mesh].tris.len()];
    for &t in &hs.tris {
        hair[t] = true;
    }
    for (mi, m) in model.meshes.iter().enumerate() {
        if m.image != Some(sheet_image) {
            continue;
        }
        for (ti, t) in m.tris.iter().enumerate() {
            if mi == hs.mesh && hair[ti] {
                continue;
            }
            let p = t.map(|v| [m.uv[v as usize][0] as f64 * size as f64, m.uv[v as usize][1] as f64 * size as f64]);
            // Conservative: every texel the triangle touches, grown by one texel.
            let x0 = (p.iter().map(|a| a[0]).fold(f64::MAX, f64::min).floor() as i64 - 1).max(0);
            let x1 = (p.iter().map(|a| a[0]).fold(f64::MIN, f64::max).ceil() as i64 + 1).min(size as i64 - 1);
            let y0 = (p.iter().map(|a| a[1]).fold(f64::MAX, f64::min).floor() as i64 - 1).max(0);
            let y1 = (p.iter().map(|a| a[1]).fold(f64::MIN, f64::max).ceil() as i64 + 1).min(size as i64 - 1);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let c = [x as f64 + 0.5, y as f64 + 0.5];
                    if point_tri_dist(c, p) <= 1.5 {
                        occ[y as usize * size + x as usize] = true;
                    }
                }
            }
        }
    }
    for b in keep {
        let (x0, y0) = ((b[0] * size as f64).floor() as usize, (b[1] * size as f64).floor() as usize);
        let (x1, y1) = (((b[2] * size as f64).ceil() as usize).min(size), ((b[3] * size as f64).ceil() as usize).min(size));
        for y in y0..y1 {
            for x in x0..x1 {
                occ[y * size + x] = true;
            }
        }
    }
    occ
}

/// Distance from a point to a 2D triangle (0 inside).
pub fn point_tri_dist(c: [f64; 2], p: [[f64; 2]; 3]) -> f64 {
    let s = |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    let (d0, d1, d2) = (s(p[0], p[1]), s(p[1], p[2]), s(p[2], p[0]));
    if (d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0) || (d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0) {
        return 0.0;
    }
    (0..3).map(|k| seg_dist(c, p[k], p[(k + 1) % 3])).fold(f64::MAX, f64::min)
}

pub fn seg_dist(c: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = dx * dx + dy * dy;
    let t = if l2 < 1e-30 { 0.0 } else { (((c[0] - a[0]) * dx + (c[1] - a[1]) * dy) / l2).clamp(0.0, 1.0) };
    let (ex, ey) = (a[0] + dx * t - c[0], a[1] + dy * t - c[1]);
    (ex * ex + ey * ey).sqrt()
}

/// Packs chart rectangles (upright, v down the page) into free cells. `scale` = texels per model
/// unit, `pad` = texels of gutter on every side. Returns false when they do not all fit.
pub fn pack(charts: &mut [Chart], occ: &[bool], size: usize, scale: f64, pad: usize) -> bool {
    const CELL: usize = 4;
    let g = size / CELL;
    let mut used = vec![false; g * g];
    for y in 0..size {
        for x in 0..size {
            if occ[y * size + x] {
                used[(y / CELL) * g + x / CELL] = true;
            }
        }
    }
    let mut order: Vec<usize> = (0..charts.len()).collect();
    let dims: Vec<(usize, usize)> = charts
        .iter()
        .map(|c| {
            let w = ((c.hi[0] - c.lo[0]) * scale).ceil() as usize + 2 * pad + 1;
            let h = ((c.hi[1] - c.lo[1]) * scale).ceil() as usize + 2 * pad + 1;
            (w.div_ceil(CELL), h.div_ceil(CELL))
        })
        .collect();
    order.sort_by(|&a, &b| (dims[b].1, dims[b].0).cmp(&(dims[a].1, dims[a].0)));
    for &k in &order {
        let (cw, ch) = dims[k];
        if cw > g || ch > g {
            return false;
        }
        // Summed-area table of used cells.
        let mut sat = vec![0u32; (g + 1) * (g + 1)];
        for y in 0..g {
            let mut row = 0u32;
            for x in 0..g {
                row += used[y * g + x] as u32;
                sat[(y + 1) * (g + 1) + x + 1] = sat[y * (g + 1) + x + 1] + row;
            }
        }
        let busy = |x: usize, y: usize| sat[(y + ch) * (g + 1) + x + cw] + sat[y * (g + 1) + x] - sat[y * (g + 1) + x + cw] - sat[(y + ch) * (g + 1) + x];
        let mut spot = None;
        'scan: for y in 0..=(g - ch) {
            for x in 0..=(g - cw) {
                if busy(x, y) == 0 {
                    spot = Some((x, y));
                    break 'scan;
                }
            }
        }
        let Some((x, y)) = spot else { return false };
        for yy in y..y + ch {
            for xx in x..x + cw {
                used[yy * g + xx] = true;
            }
        }
        charts[k].at = [(x * CELL + pad) as f64, (y * CELL + pad) as f64];
    }
    true
}

/// The largest scale (texels per model unit) at which every chart fits, by bisection.
pub fn fit_scale(charts: &mut [Chart], occ: &[bool], size: usize, pad: usize) -> Result<f64, String> {
    let area: f64 = charts.iter().map(|c| (c.hi[0] - c.lo[0]) * (c.hi[1] - c.lo[1])).sum();
    let free = occ.iter().filter(|&&o| !o).count() as f64;
    let mut hi = (free / area.max(1e-12)).sqrt();
    let mut lo = 0.0;
    for _ in 0..18 {
        let mid = 0.5 * (lo + hi);
        if pack(charts, occ, size, mid, pad) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if lo <= 0.0 || !pack(charts, occ, size, lo, pad) {
        return Err("the hair charts do not fit on the sheet".into());
    }
    Ok(lo)
}

/// UV on the sheet (0..1) for a strand coordinate in a chart.
pub fn sheet_uv(c: &Chart, scale: f64, size: usize, q: [f64; 2]) -> [f32; 2] {
    [((c.at[0] + (q[0] - c.lo[0]) * scale) / size as f64) as f32, ((c.at[1] + (q[1] - c.lo[1]) * scale) / size as f64) as f32]
}

/// New geometry for every primitive: hair corners get their chart UVs (a new vertex per chart and
/// source vertex), everything else keeps its vertex and UV.
pub fn new_meshes(model: &Model, hs: &HairSet, pr: &Param, cs: &[Chart], chart_of: &[usize], scale: f64, size: usize) -> Vec<crate::rebuild::NewMesh> {
    let mut out = Vec::new();
    for (mi, m) in model.meshes.iter().enumerate() {
        if mi != hs.mesh {
            out.push(crate::rebuild::NewMesh { src: (0..m.pos.len() as u32).collect(), uv: m.uv.clone(), tris: m.tris.clone() });
            continue;
        }
        let mut hair = vec![usize::MAX; m.tris.len()];
        for (i, &t) in hs.tris.iter().enumerate() {
            hair[t] = i;
        }
        let mut map: HashMap<(usize, u32), u32> = HashMap::new();
        let (mut src, mut uv, mut tris) = (Vec::new(), Vec::new(), Vec::new());
        for (ti, t) in m.tris.iter().enumerate() {
            let i = hair[ti];
            let nt = std::array::from_fn(|c| {
                let v = t[c];
                let key = if i == usize::MAX { (usize::MAX, v) } else { (chart_of[i], v) };
                *map.entry(key).or_insert_with(|| {
                    src.push(v);
                    uv.push(if i == usize::MAX { m.uv[v as usize] } else { sheet_uv(&cs[chart_of[i]], scale, size, pr.uv[i][c]) });
                    (src.len() - 1) as u32
                })
            });
            tris.push(nt);
        }
        out.push(crate::rebuild::NewMesh { src, uv, tris });
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Color: sRGB <-> CIE Lab (D65)
// ---------------------------------------------------------------------------------------------

fn to_lin(c: f64) -> f64 {
    let c = c / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
fn from_lin(c: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    255.0 * if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

pub fn lab(rgb: [f64; 3]) -> [f64; 3] {
    let (r, g, b) = (to_lin(rgb[0]), to_lin(rgb[1]), to_lin(rgb[2]));
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    let f = |t: f64| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

pub fn rgb(lab: [f64; 3]) -> [f64; 3] {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = fy + lab[1] / 500.0;
    let fz = fy - lab[2] / 200.0;
    let fi = |t: f64| if t * t * t > 0.008856 { t * t * t } else { (t - 16.0 / 116.0) / 7.787 };
    let (x, y, z) = (fi(fx) * 0.95047, fi(fy), fi(fz) * 1.08883);
    let r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let b = 0.0557 * x - 0.2040 * y + 1.0570 * z;
    [from_lin(r), from_lin(g), from_lin(b)]
}

pub fn delta_e(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Root-to-tip palette sampled from the stock paint: median Lab per band of v / v_max.
pub struct Palette {
    pub bands: Vec<[f64; 3]>,
    /// Interquartile range of stock lightness (sets strand and clump contrast).
    pub l_iqr: f64,
    /// Median hue angle (degrees) and chroma of the stock hair.
    pub hue: f64,
    pub chroma: f64,
}

impl Palette {
    /// `samples` = (t in 0..1, Lab).
    pub fn from_samples(samples: &[(f64, [f64; 3])], nb: usize) -> Result<Palette, String> {
        if samples.len() < 50 {
            return Err(format!("only {} stock hair texels to sample the palette from", samples.len()));
        }
        let med = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let mut bands = Vec::new();
        for b in 0..nb {
            let (lo, hi) = (b as f64 / nb as f64, (b + 1) as f64 / nb as f64);
            let in_band: Vec<&[f64; 3]> = samples.iter().filter(|s| s.0 >= lo && (s.0 < hi || (b + 1 == nb && s.0 <= 1.0))).map(|s| &s.1).collect();
            if in_band.len() < 20 {
                bands.push(None);
                continue;
            }
            bands.push(Some(std::array::from_fn(|c| med(in_band.iter().map(|l| l[c]).collect()))));
        }
        // Fill empty bands from neighbors.
        let known: Vec<(usize, [f64; 3])> = bands.iter().enumerate().filter_map(|(i, b)| b.map(|v| (i, v))).collect();
        if known.is_empty() {
            return Err("no palette band has enough stock texels".into());
        }
        let bands: Vec<[f64; 3]> = (0..nb).map(|i| bands[i].unwrap_or_else(|| known.iter().min_by_key(|k| (k.0 as i64 - i as i64).abs()).unwrap().1)).collect();
        // Smooth along the ramp (1-2-1).
        let bands: Vec<[f64; 3]> = (0..nb)
            .map(|i| {
                let a = bands[i.saturating_sub(1)];
                let c = bands[(i + 1).min(nb - 1)];
                std::array::from_fn(|k| 0.25 * a[k] + 0.5 * bands[i][k] + 0.25 * c[k])
            })
            .collect();
        let mut ls: Vec<f64> = samples.iter().map(|s| s.1[0]).collect();
        ls.sort_by(f64::total_cmp);
        let l_iqr = ls[ls.len() * 3 / 4] - ls[ls.len() / 4];
        let a = med(samples.iter().map(|s| s.1[1]).collect());
        let b = med(samples.iter().map(|s| s.1[2]).collect());
        Ok(Palette { bands, l_iqr, hue: b.atan2(a).to_degrees(), chroma: (a * a + b * b).sqrt() })
    }

    pub fn at(&self, t: f64) -> [f64; 3] {
        let n = self.bands.len();
        let x = (t.clamp(0.0, 1.0) * n as f64 - 0.5).clamp(0.0, (n - 1) as f64);
        let i = (x.floor() as usize).min(n - 1);
        let j = (i + 1).min(n - 1);
        let f = x - i as f64;
        std::array::from_fn(|k| self.bands[i][k] * (1.0 - f) + self.bands[j][k] * f)
    }

    /// True when a color does not belong to this hair: wrong hue, washed out to gray, near black,
    /// or the stock's transparent magenta.
    pub fn is_fringe(&self, rgb_: [f64; 3]) -> bool {
        self.is_fringe_by(rgb_, 35.0, 0.3, 12.0)
    }

    /// `is_fringe` with its limits as arguments (hue tolerance in degrees, minimum chroma as a
    /// fraction of the hair's, minimum L). The bake uses tighter limits than the test so that
    /// mip averages of in-palette texels stay in the palette too.
    pub fn is_fringe_by(&self, rgb_: [f64; 3], hue_tol: f64, chroma_frac: f64, min_l: f64) -> bool {
        let l = lab(rgb_);
        let c = (l[1] * l[1] + l[2] * l[2]).sqrt();
        let h = l[2].atan2(l[1]).to_degrees();
        let mut dh = (h - self.hue).abs() % 360.0;
        if dh > 180.0 {
            dh = 360.0 - dh;
        }
        let magenta = rgb_[0] > 100.0 && rgb_[2] > 90.0 && rgb_[0] > 1.35 * rgb_[1] && rgb_[2] > 1.25 * rgb_[1];
        magenta || l[0] < min_l || (c > 8.0 && dh > hue_tol) || c < chroma_frac * self.chroma
    }
}

impl Palette {
    /// Regrades the ramp: rotate hue (degrees, negative = warmer for yellow hair), scale chroma,
    /// shift lightness (Lab L). The fringe test follows the new hue and chroma.
    pub fn grade(&mut self, hue_shift: f64, chroma_scale: f64, l_shift: f64) {
        let (c, s) = (hue_shift.to_radians().cos(), hue_shift.to_radians().sin());
        for b in &mut self.bands {
            let (a0, b0) = (b[1], b[2]);
            b[0] = (b[0] + l_shift).clamp(0.0, 100.0);
            b[1] = chroma_scale * (a0 * c - b0 * s);
            b[2] = chroma_scale * (a0 * s + b0 * c);
        }
        self.hue += hue_shift;
        self.chroma *= chroma_scale;
    }
}

// ---------------------------------------------------------------------------------------------
// Stage 5: strand synthesizer
// ---------------------------------------------------------------------------------------------

/// Look parameters. Lengths are fractions of the hair's size (its largest bounding-box side).
#[derive(Clone, Debug)]
pub struct Style {
    pub seed: u64,
    /// Clump (lock bundle) width.
    pub clump: f64,
    /// Fine strand spacing.
    pub strand: f64,
    /// How far strands snake sideways, and over what length.
    pub wave_amp: f64,
    pub wave_len: f64,
    /// Lightness swings, as fractions of the stock lightness IQR.
    pub clump_contrast: f64,
    pub strand_contrast: f64,
    /// Dark line between clumps (Lab L).
    pub clump_edge_dark: f64,
    /// Root shade (L) and the length it fades over.
    pub root_dark: f64,
    pub root_len: f64,
    /// Highlight band: position (fraction of the longest v), width, strength (L).
    pub band_at: f64,
    pub band_width: f64,
    pub band_gain: f64,
    /// Tip lightening (L) and length.
    pub tip_light: f64,
    pub tip_len: f64,
    /// Depth of the notches cut between clump tips (pointed tips), and how far in from the stock
    /// cutout edge they reach.
    pub tip_depth: f64,
    /// Strength of baked ambient occlusion (0 = none).
    pub ao: f64,
    /// Rounded-bundle shading across each clump (fraction of the lightness IQR).
    pub bundle: f64,
    /// How irregular clump borders are (0 = even stripes).
    pub clump_jitter: f64,
}

impl Default for Style {
    fn default() -> Style {
        Style {
            seed: 9,
            clump: 0.06,
            strand: 0.0022,
            wave_amp: 0.0025,
            wave_len: 0.22,
            clump_contrast: 0.40,
            strand_contrast: 0.45,
            clump_edge_dark: 18.0,
            root_dark: 9.0,
            root_len: 0.12,
            band_at: 0.36,
            band_width: 0.05,
            band_gain: 9.0,
            tip_light: 6.0,
            tip_len: 0.06,
            tip_depth: 0.05,
            ao: 0.45,
            bundle: 0.35,
            clump_jitter: 0.35,
        }
    }
}

pub(crate) fn hash(a: u64) -> u64 {
    let mut x = a.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
pub(crate) fn h01(a: u64) -> f64 {
    (hash(a) >> 11) as f64 / (1u64 << 53) as f64
}
pub(crate) fn h2(seed: u64, i: i64, j: i64) -> f64 {
    h01(seed ^ (i as u64).wrapping_mul(0x51_7CC1_B727_220A) ^ (j as u64).wrapping_mul(0x2545_F491_4F6C_DD1D))
}
pub(crate) fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
/// 2D value noise in 0..1.
pub(crate) fn vnoise(seed: u64, x: f64, y: f64) -> f64 {
    let (xi, yi) = (x.floor() as i64, y.floor() as i64);
    let (fx, fy) = (smooth(x - xi as f64), smooth(y - yi as f64));
    let a = h2(seed, xi, yi);
    let b = h2(seed, xi + 1, yi);
    let c = h2(seed, xi, yi + 1);
    let d = h2(seed, xi + 1, yi + 1);
    (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy
}

/// Everything the synthesizer needs that is fixed for the whole hair.
pub struct SynthCtx<'a> {
    pub style: &'a Style,
    pub pal: &'a Palette,
    /// Hair size (model units) and the longest root-to-tip coordinate.
    pub size: f64,
    pub vmax: f64,
}

/// Strand pattern at (region, u, v): returns (Lab color before shading, tip-notch profile 0..1).
/// `group` picks the pattern seed (the lock: regions split from one lock share it, so strands
/// match where their u values meet).
pub fn strand(ctx: &SynthCtx, region: usize, u: f64, v: f64, dens: f64) -> ([f64; 3], f64) {
    // Octave level: one more halving of the clump width (in u) per halving of |grad u|.
    let l = (1.0 / dens.max(1e-3)).log2().clamp(-2.0, 6.0);
    let k0 = l.floor();
    let f = smooth(l - k0);
    let (c0, p0) = strand_octave(ctx, region, u, v, k0 as i64);
    if f < 1e-3 {
        return (c0, p0);
    }
    let (c1, p1) = strand_octave(ctx, region, u, v, k0 as i64 + 1);
    (std::array::from_fn(|i| c0[i] * (1.0 - f) + c1[i] * f), p0 * (1.0 - f) + p1 * f)
}

fn strand_octave(ctx: &SynthCtx, region: usize, u: f64, v: f64, k: i64) -> ([f64; 3], f64) {
    let u = u * 2f64.powi(k as i32);
    let region = region * 16 + (k + 4) as usize;
    let s = ctx.style;
    let e = ctx.size;
    let seed = hash(s.seed ^ (region as u64 + 1).wrapping_mul(0x1234_5678_9ABC));
    // Sideways snake, phase per region.
    let phase = h01(seed) + 1.5 * vnoise(seed ^ 3, u / (2.5 * s.clump * e), 0.37);
    let uw = u + s.wave_amp * e * ((v / (s.wave_len * e) + phase) * std::f64::consts::TAU).sin()
        + 0.5 * s.wave_amp * e * (vnoise(seed ^ 7, u / (s.clump * e), v / (s.wave_len * e * 0.5)) - 0.5);
    // Clumps with jittered borders.
    let cw = s.clump * e;
    let cj = uw / cw + s.clump_jitter * (vnoise(seed ^ 11, uw / cw * 0.5, v / (0.3 * e)) - 0.5);
    let k = cj.floor();
    let x = cj - k;
    let rk = h2(seed ^ 13, k as i64, 0);
    let edge = (2.0 * x - 1.0).abs();
    // Fine strands: streaks stretched along v.
    let sw = s.strand * e;
    let n1 = vnoise(seed ^ 17, uw / sw, v / (0.08 * e));
    let n2 = vnoise(seed ^ 19, uw / (sw * 2.7), v / (0.2 * e));
    let n3 = vnoise(seed ^ 23, uw / (sw * 0.55), v / (0.05 * e));
    let fine = 0.5 * n1 + 0.3 * n2 + 0.2 * n3 - 0.5;
    let t = (v / ctx.vmax).clamp(0.0, 1.0);
    let mut c = ctx.pal.at(t);
    let iqr = ctx.pal.l_iqr.max(4.0);
    c[0] += s.clump_contrast * iqr * (rk - 0.5) * 2.0;
    c[0] += s.strand_contrast * iqr * fine * 2.0;
    // Each clump shaded like a rounded bundle: lit middle, dark groove at its borders.
    c[0] += s.bundle * iqr * (1.0 - edge * edge - 2.0 / 3.0);
    c[0] -= s.clump_edge_dark * edge.powi(6);
    c[0] -= s.root_dark * (1.0 - smooth(v / (s.root_len * e)));
    // Highlight band, wavy, broken up by the strands.
    let vb = s.band_at * ctx.vmax + 0.03 * e * (vnoise(seed ^ 29, u / (0.08 * e), 0.5) - 0.5) * 2.0;
    let band = (-((v - vb) / (s.band_width * e)).powi(2)).exp();
    c[0] += s.band_gain * band * (0.55 + 0.9 * n1);
    // Tip notch profile: 0 at a clump's middle, 1 at its border; finer notches inside.
    let x2 = (cj * 3.0).fract();
    let prof = (0.75 * edge.powf(1.3) + 0.3 * rk + 0.2 * (2.0 * x2 - 1.0).abs()).min(1.0);
    (c, prof)
}

/// Crisp strand layer: one fiber per cell across the strands (jittered center, width and
/// brightness, broken into segments along them), dark gaps between, a second finer layer on top.
/// Widths are in texels so the strands stay a set size on the sheet; values are zero-mean so the
/// low-pass form (lock shading) is kept at any mip.
#[derive(Clone, Debug)]
pub struct Fibers {
    /// Lightness swing (Lab L) of the main layer.
    pub gain: f64,
    /// Cell width of the main and fine layers, in texels.
    pub width: f64,
    pub fine_width: f64,
    /// Fine layer strength relative to the main one.
    pub fine: f64,
    pub seed: u64,
    mean: f64,
}

impl Fibers {
    pub fn new(gain: f64, width: f64, fine_width: f64, fine: f64, seed: u64) -> Fibers {
        let mut f = Fibers { gain, width, fine_width, fine, seed, mean: 0.0 };
        let (mut acc, n) = (0.0, 40000);
        for i in 0..n {
            let u = h01(seed ^ 0xabc ^ (i as u64).wrapping_mul(7919)) * 400.0;
            let v = h01(seed ^ 0xdef ^ (i as u64).wrapping_mul(104729)) * 4000.0;
            acc += f.raw(i % 7, u, v, 1.0);
        }
        f.mean = acc / n as f64;
        f
    }

    fn raw(&self, lock: usize, u: f64, v: f64, scale: f64) -> f64 {
        let sd = hash(self.seed ^ (lock as u64 + 1).wrapping_mul(0x2F1F_3A57));
        let a = fiber_layer(sd, u, v, self.width / scale);
        if self.fine > 0.0 {
            a + self.fine * fiber_layer(sd ^ 0xF1, u, v, self.fine_width / scale)
        } else {
            a
        }
    }

    /// Offset in L (zero-mean) at strand coordinates (u, v), `scale` texels per unit.
    pub fn at(&self, lock: usize, u: f64, v: f64, scale: f64) -> f64 {
        self.gain * (self.raw(lock, u, v, scale) - self.mean)
    }
}

/// Edge ramp of a fiber, as a share of its half-width. A softer ramp (0.5) gave more one-pixel
/// specks in `gaw defects`, not fewer, and weaker layer edges.
pub(crate) const FIBER_EDGE: f64 = 0.25;

fn fiber_layer(seed: u64, u: f64, v: f64, wu: f64) -> f64 {
    let wob = 0.6 * (vnoise(seed ^ 5, u / (wu * 3.0), v / (wu * 25.0)) - 0.5);
    let x = u / wu + wob;
    let i = x.floor() as i64;
    let mut val = -1.0f64;
    // Nearest fiber edge on each side of x, with the fiber's level there, for gap closing.
    let (mut left, mut right) = ((f64::NEG_INFINITY, -1.0f64), (f64::INFINITY, -1.0f64));
    for j in i - 2..=i + 2 {
        let c = j as f64 + 0.5 + 0.5 * (h2(seed, j, 1) - 0.5);
        let hw = 0.30 + 0.25 * h2(seed, j, 2);
        let sj = seed ^ (j as u64).wrapping_mul(0x9E37_79B9);
        // Fibers thin out along their length but never vanish: several neighbors gone at once
        // left wide one-tone dark bands (flat patches).
        let pres = 0.35 + 0.65 * smooth((vnoise(sj ^ 9, v / (wu * 30.0), 0.5) - 0.25) / 0.2);
        let bright = 0.3 + 0.7 * h2(seed, j, 3);
        let at = |d: f64| -1.0 + smooth((1.0 - d) / FIBER_EDGE) * pres * (1.0 + 1.2 * bright * (1.0 - d * d).max(0.0).sqrt());
        let d = (x - c).abs() / hw;
        if d < 1.0 {
            val = val.max(at(d));
        }
        // Edge level just inside the fiber (d = 0.75), the brightness a closed gap takes.
        if c + hw <= x && c + hw > left.0 {
            left = (c + hw, at(0.75));
        }
        if c - hw >= x && c - hw < right.0 {
            right = (c - hw, at(0.75));
        }
    }
    // A dark gap narrower than 0.4 of a cell renders as a one-pixel line at the close-up mips
    // (a dot to the eye and to `gaw defects`): fill it between the two fiber edges.
    let gap = right.0 - left.0;
    if gap < 0.4 && gap > 0.0 {
        let t = (x - left.0) / gap;
        val = val.max(left.1 + (right.1 - left.1) * t);
    }
    val
}

/// Lightness with a soft knee (12 L wide) under the brightest in-gamut L at this a, b.
fn soft_top(c: [f64; 3]) -> f64 {
    let inside = |l: f64| rgb([l, c[1], c[2]]).iter().all(|&x| x <= 255.0);
    let knee = 12.0;
    if c[0] < 50.0 || !inside(0.0) {
        return c[0];
    }
    let (mut lo, mut hi) = (0.0f64, 100.0f64);
    for _ in 0..14 {
        let m = 0.5 * (lo + hi);
        if inside(m) {
            lo = m;
        } else {
            hi = m;
        }
    }
    let k0 = lo - knee;
    if c[0] > k0 {
        k0 + knee * ((c[0] - k0) / knee).tanh()
    } else {
        c[0]
    }
}

fn to_rgb8(c: [f64; 3]) -> [u8; 3] {
    let r = rgb(c);
    [r[0].round().clamp(0.0, 255.0) as u8, r[1].round().clamp(0.0, 255.0) as u8, r[2].round().clamp(0.0, 255.0) as u8]
}

/// Per-texel inputs inside one chart's grid (gutter included).
struct TexelIn {
    tri: usize,
    bary: [f64; 3],
}

/// Barycentric coordinates of a 2D point (can be outside the triangle).
pub fn bary2(p: [[f64; 2]; 3], c: [f64; 2]) -> Option<[f64; 3]> {
    let den = (p[1][1] - p[2][1]) * (p[0][0] - p[2][0]) + (p[2][0] - p[1][0]) * (p[0][1] - p[2][1]);
    if den.abs() < 1e-18 {
        return None;
    }
    let l0 = ((p[1][1] - p[2][1]) * (c[0] - p[2][0]) + (p[2][0] - p[1][0]) * (c[1] - p[2][1])) / den;
    let l1 = ((p[2][1] - p[0][1]) * (c[0] - p[2][0]) + (p[0][0] - p[2][0]) * (c[1] - p[2][1])) / den;
    Some([l0, l1, 1.0 - l0 - l1])
}

/// Distance from `q` to the segment `a`-`b`.
pub fn seg_dist3(q: V3, a: V3, b: V3) -> f64 {
    let ab = sub(b, a);
    let l2 = dot(ab, ab);
    let t = if l2 > 0.0 { (dot(sub(q, a), ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    len(sub(q, add(a, mul(ab, t))))
}

/// Barycentric coordinates of `q` (taken to lie in the plane) in the 3D triangle `p`.
pub fn bary3(p: [V3; 3], q: V3) -> Option<[f64; 3]> {
    let (e0, e1, e2) = (sub(p[1], p[0]), sub(p[2], p[0]), sub(q, p[0]));
    let (d00, d01, d11, d20, d21) = (dot(e0, e0), dot(e0, e1), dot(e1, e1), dot(e2, e0), dot(e2, e1));
    let den = d00 * d11 - d01 * d01;
    if den.abs() < 1e-18 {
        return None;
    }
    let v = (d11 * d20 - d01 * d21) / den;
    let w = (d00 * d21 - d01 * d20) / den;
    Some([1.0 - v - w, v, w])
}

/// Squared-distance transform along one line (Felzenszwalb and Huttenlocher).
fn dt1d(f: &[f64], d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        loop {
            let p = v[k];
            let sx = ((f[q] + (q * q) as f64) - (f[p] + (p * p) as f64)) / (2.0 * q as f64 - 2.0 * p as f64);
            if sx <= z[k] && k > 0 {
                k -= 1;
                continue;
            }
            if sx <= z[k] {
                // k == 0: replace the first parabola
                v[0] = q;
                z[0] = f64::NEG_INFINITY;
                z[1] = f64::INFINITY;
                break;
            }
            k += 1;
            v[k] = q;
            z[k] = sx;
            z[k + 1] = f64::INFINITY;
            break;
        }
    }
    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let p = v[k];
        d[q] = (q as f64 - p as f64).powi(2) + f[p];
    }
}

/// Euclidean distance (texels) from every cell to the nearest cell where `target` is true
/// (a large value when there is none).
pub fn edt_to(target: &[bool], w: usize, h: usize) -> Vec<f64> {
    const INF: f64 = 1e12;
    let mut g: Vec<f64> = target.iter().map(|&t| if t { 0.0 } else { INF }).collect();
    let n = w.max(h);
    let (mut f, mut d, mut v, mut z) = (vec![0.0; n], vec![0.0; n], vec![0usize; n], vec![0.0; n + 1]);
    for x in 0..w {
        for y in 0..h {
            f[y] = g[y * w + x];
        }
        dt1d(&f[..h], &mut d[..h], &mut v[..h], &mut z[..h + 1]);
        for y in 0..h {
            g[y * w + x] = d[y];
        }
    }
    for y in 0..h {
        f[..w].copy_from_slice(&g[y * w..y * w + w]);
        dt1d(&f[..w], &mut d[..w], &mut v[..w], &mut z[..w + 1]);
        for x in 0..w {
            g[y * w + x] = d[x].sqrt();
        }
    }
    g
}

/// Cutout cleanup on a grid: pieces smaller than `min_area` cells or thinner than `min_width`
/// (twice their deepest inside distance) go; enclosed holes smaller than `min_area` are filled.
pub fn clean_mask(m: &mut [bool], w: usize, h: usize, min_area: usize, min_width: f64) {
    let din = edt_to(&m.iter().map(|&x| !x).collect::<Vec<_>>(), w, h);
    let mm: Vec<bool> = m.to_vec();
    let (lab, sz, _, _) = crate::hairlook::components(w, h, &|i| mm[i], &|_, _| true);
    let mut deep = vec![0.0f64; sz.len()];
    for k in 0..w * h {
        if lab[k] != usize::MAX {
            deep[lab[k]] = deep[lab[k]].max(din[k]);
        }
    }
    for k in 0..w * h {
        if lab[k] != usize::MAX && (sz[lab[k]] < min_area || 2.0 * deep[lab[k]] < min_width) {
            m[k] = false;
        }
    }
    let mm: Vec<bool> = m.to_vec();
    let (lab, sz, bd, _) = crate::hairlook::components(w, h, &|i| !mm[i], &|_, _| true);
    for k in 0..w * h {
        if lab[k] != usize::MAX && !bd[lab[k]] && sz[lab[k]] < min_area {
            m[k] = true;
        }
    }
}

pub struct BakeIn<'a> {
    pub model: &'a Model,
    pub hs: &'a HairSet,
    pub fl: &'a Flow,
    pub pr: &'a Param,
    pub charts: &'a [Chart],
    pub scale: f64,
    pub size: usize,
    pub pad: usize,
    pub stock: Sheet<'a>,
    /// Stock alpha after a light blur (same size as `stock`), 0..1.
    pub stock_alpha: Vec<f64>,
    /// Ambient occlusion per weld id (1 = open).
    pub ao: Vec<f64>,
    pub ctx: SynthCtx<'a>,
    /// Form layers (guide low-pass, node AO, highlight); `None` = synthesizer only.
    pub look: Option<&'a Look>,
    /// Alpha ramp half-width in texels around the cutout contour (0.6 = a hard edge).
    pub sdf_spread: f64,
    /// Per selected triangle: grown into the part (no guide paint; full style instead).
    pub grown: Vec<bool>,
    /// Dark line (Lab L) at the root of grown triangles, where they tuck under the hair.
    pub root_line: f64,
    /// Grown triangles (a widow's peak) sit in the bangs' shadow: darken them by this (Lab L),
    /// half at their lowest point and all of it at their top, and cut a dark crevice of
    /// `crevice` (Lab L) down their middle (the mean x of the grown corners).
    pub grown_shade: f64,
    pub crevice: f64,
    /// Cutout cleanup in texels: drop pieces smaller than `alpha_min_area` or thinner than
    /// `alpha_min_width`, fill enclosed holes smaller than `alpha_min_area`. 0 = the plain
    /// gradient distance (v3).
    pub alpha_min_area: usize,
    pub alpha_min_width: f64,
}

/// Chart texels written by `bake`: per sheet texel, the chart (usize::MAX = none) and the selected
/// triangle that owns it (inside its triangle, not a gutter texel).
pub struct BakeMap {
    pub chart: Vec<usize>,
    pub inside: Vec<bool>,
    /// Per stock-sheet texel: (sum of output L, count) over the inside texels whose stock UV falls
    /// in it, for the form check (output low-pass against stock).
    pub form: Vec<(f64, u32)>,
    /// Gutter texels in all, and how many of them took the surface across a welded edge.
    pub gutter: usize,
    pub gutter_surface: usize,
}

/// Paints every chart into `rgba` (size x size). Returns which texels belong to which chart.
pub fn bake(b: &BakeIn, rgba: &mut [u8]) -> BakeMap {
    let size = b.size;
    let mut map = BakeMap { chart: vec![usize::MAX; size * size], inside: vec![false; size * size], form: vec![(0.0, 0); b.stock.w * b.stock.h], gutter: 0, gutter_surface: 0 };
    let dctx = b.look.map(|lk| SynthCtx { style: &lk.detail, pal: b.ctx.pal, size: b.ctx.size, vmax: b.ctx.vmax });
    // Extent of the grown triangles: (center x, half width, lowest y, highest y).
    let gb = {
        let pts: Vec<V3> = (0..b.hs.tris.len()).filter(|&i| b.grown.get(i).copied().unwrap_or(false)).flat_map(|i| b.hs.corners(i)).collect();
        let cx = pts.iter().map(|p| p[0]).sum::<f64>() / pts.len().max(1) as f64;
        let half = pts.iter().map(|p| (p[0] - cx).abs()).fold(0.0, f64::max);
        let y0 = pts.iter().map(|p| p[1]).fold(f64::MAX, f64::min);
        let y1 = pts.iter().map(|p| p[1]).fold(f64::MIN, f64::max);
        (cx, half, y0, y1)
    };
    // Strand tangent per (weld id, region), area-weighted, for a smooth highlight.
    let mut tan: HashMap<(usize, usize), V3> = HashMap::new();
    if b.look.is_some_and(|lk| lk.light.is_some()) {
        for i in 0..b.hs.tris.len() {
            if !b.hs.is_primary(i) {
                continue;
            }
            if let Some(d) = uv_dir_to_3d(b.hs.corners(i), b.pr.uv[i], [0.0, 1.0]) {
                let d = mul(unit(d), b.hs.area(i));
                for &w in &b.hs.wt[i] {
                    let e = tan.entry((w, b.fl.region[i])).or_insert([0.0; 3]);
                    *e = add(*e, d);
                }
            }
        }
    }
    // Smooth normals per welded corner (oriented per lock), so the highlight does not facet.
    let osign = orient(b.hs);
    let mut wn: HashMap<(usize, usize), V3> = HashMap::new();
    for i in 0..b.hs.tris.len() {
        if !b.hs.is_primary(i) {
            continue;
        }
        let n = mul(b.hs.normal(i), osign[i] * b.hs.area(i));
        for &w in &b.hs.wt[i] {
            let e = wn.entry((w, b.hs.lock[i])).or_insert([0.0; 3]);
            *e = add(*e, n);
        }
    }
    // Hair center, to orient normals outward for the highlight.
    let hc = {
        let ps: Vec<usize> = (0..b.hs.tris.len()).filter(|&i| b.hs.is_primary(i)).collect();
        mul(ps.iter().fold([0.0; 3], |a, &i| add(a, b.hs.centroid(i))), 1.0 / ps.len().max(1) as f64)
    };
    let stock_uv = |i: usize| tri_uv(b.model, b.hs, b.hs.primary[i]);
    let nb = neighbors(b.hs);
    let mut gutter_surface = 0usize;
    let sa = |u: f64, v: f64| {
        let x = u * b.stock.w as f64 - 0.5;
        let y = v * b.stock.h as f64 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let g = |xx: i64, yy: i64| b.stock_alpha[(yy.clamp(0, b.stock.h as i64 - 1) as usize) * b.stock.w + xx.clamp(0, b.stock.w as i64 - 1) as usize];
        let (x0, y0) = (x0 as i64, y0 as i64);
        (g(x0, y0) * (1.0 - fx) + g(x0 + 1, y0) * fx) * (1.0 - fy) + (g(x0, y0 + 1) * (1.0 - fx) + g(x0 + 1, y0 + 1) * fx) * fy
    };
    for (ci, c) in b.charts.iter().enumerate() {
        let w = ((c.hi[0] - c.lo[0]) * b.scale).ceil() as usize + 2 * b.pad + 1;
        let h = ((c.hi[1] - c.lo[1]) * b.scale).ceil() as usize + 2 * b.pad + 1;
        let (gx, gy) = (c.at[0] as i64 - b.pad as i64, c.at[1] as i64 - b.pad as i64);
        // Texel grid of this chart: owner triangle and barycentrics.
        let mut cell: Vec<Option<TexelIn>> = (0..w * h).map(|_| None).collect();
        let mut dist = vec![f64::MAX; w * h];
        for &i in &c.tris {
            let p: [[f64; 2]; 3] = std::array::from_fn(|k| {
                let q = b.pr.uv[i][k];
                [(q[0] - c.lo[0]) * b.scale + b.pad as f64, (q[1] - c.lo[1]) * b.scale + b.pad as f64]
            });
            let x0 = (p.iter().map(|a| a[0]).fold(f64::MAX, f64::min) - b.pad as f64).floor().max(0.0) as usize;
            let x1 = ((p.iter().map(|a| a[0]).fold(f64::MIN, f64::max) + b.pad as f64).ceil() as usize).min(w - 1);
            let y0 = (p.iter().map(|a| a[1]).fold(f64::MAX, f64::min) - b.pad as f64).floor().max(0.0) as usize;
            let y1 = ((p.iter().map(|a| a[1]).fold(f64::MIN, f64::max) + b.pad as f64).ceil() as usize).min(h - 1);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let q = [x as f64 + 0.5, y as f64 + 0.5];
                    let d = point_tri_dist(q, p);
                    if d > b.pad as f64 || d >= dist[y * w + x] {
                        continue;
                    }
                    if let Some(bc) = bary2(p, q) {
                        dist[y * w + x] = d;
                        cell[y * w + x] = Some(TexelIn { tri: i, bary: bc });
                    }
                }
            }
        }
        // Surface-space gutters: a gutter texel past a welded edge is the surface on the other side,
        // so it takes the neighbor triangle (up to two rings away, same lock) whose strand
        // coordinates contain it. Both sides of a seam then sample the same surface points, and
        // bilinear filtering across the seam mixes like values.
        for y in 0..h {
            for x in 0..w {
                let k = y * w + x;
                if dist[k] == 0.0 || dist[k] == f64::MAX {
                    continue;
                }
                let Some(t) = &cell[k] else { continue };
                let q = [c.lo[0] + (x as f64 + 0.5 - b.pad as f64) / b.scale, c.lo[1] + (y as f64 + 0.5 - b.pad as f64) / b.scale];
                let mut ring: Vec<usize> = vec![t.tri];
                let mut seen: Vec<usize> = vec![t.tri];
                let mut found = None;
                'rings: for _ in 0..2 {
                    let mut next = Vec::new();
                    for &a in &ring {
                        for &(n2, _) in &nb[a] {
                            if seen.contains(&n2) {
                                continue;
                            }
                            seen.push(n2);
                            next.push(n2);
                            if let Some(bc) = bary2(b.pr.uv[n2], q) {
                                if bc.iter().all(|&l| l >= -1e-6) {
                                    found = Some((n2, bc));
                                    break 'rings;
                                }
                            }
                        }
                    }
                    ring = next;
                }
                if found.is_none() {
                    // The strand coordinates jump at this seam: unfold the neighbor across the
                    // edge the texel lies past into this triangle's strand frame, keeping its 3D
                    // shape, and read the texel's barycentrics there.
                    let a = t.tri;
                    let bt = t.bary;
                    let k0 = (0..3).min_by(|&i, &j| bt[i].total_cmp(&bt[j])).unwrap_or(0);
                    if bt[k0] < 0.0 {
                        let (k1, k2) = ((k0 + 1) % 3, (k0 + 2) % 3);
                        let (e1, e2) = (b.hs.wt[a][k1], b.hs.wt[a][k2]);
                        if let Some(&(n2, _)) = nb[a].iter().find(|(_, e)| *e == ekey(e1, e2)) {
                            let wn = b.hs.wt[n2];
                            let m1 = wn.iter().position(|&x| x == e1);
                            let m2 = wn.iter().position(|&x| x == e2);
                            if let (Some(m1), Some(m2)) = (m1, m2) {
                                let m3 = 3 - m1 - m2;
                                let (pa, pb, pc) = (b.hs.wpos[e1], b.hs.wpos[e2], b.hs.wpos[wn[m3]]);
                                let ab = sub(pb, pa);
                                let l2 = dot(ab, ab).max(1e-18);
                                let along = dot(sub(pc, pa), ab) / l2;
                                let hgt = len(sub(sub(pc, pa), mul(ab, along)));
                                let (ua, ub, uo) = (b.pr.uv[a][k1], b.pr.uv[a][k2], b.pr.uv[a][k0]);
                                let d2 = [ub[0] - ua[0], ub[1] - ua[1]];
                                let l2d = (d2[0] * d2[0] + d2[1] * d2[1]).sqrt();
                                if l2d > 1e-12 {
                                    let sc = l2d / l2.sqrt();
                                    let mut pp = [-d2[1] / l2d, d2[0] / l2d];
                                    if pp[0] * (uo[0] - ua[0]) + pp[1] * (uo[1] - ua[1]) > 0.0 {
                                        pp = [-pp[0], -pp[1]];
                                    }
                                    let uc = [ua[0] + d2[0] * along + pp[0] * hgt * sc, ua[1] + d2[1] * along + pp[1] * hgt * sc];
                                    if let Some(bc) = bary2([ua, ub, uc], q) {
                                        if bc.iter().all(|&l| l >= -0.5) {
                                            let mut bn = [0.0; 3];
                                            bn[m1] = bc[0];
                                            bn[m2] = bc[1];
                                            bn[m3] = bc[2];
                                            found = Some((n2, bn));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some((n2, bc)) = found {
                    cell[k] = Some(TexelIn { tri: n2, bary: bc });
                    gutter_surface += 1;
                }
            }
        }
        // Pass 1: stock alpha per texel (for the cutout distance field).
        let mut a_s = vec![1.0; w * h];
        for k in 0..w * h {
            if let Some(t) = &cell[k] {
                let su = stock_uv(t.tri);
                let (u, v) = (
                    t.bary[0] * su[0][0] + t.bary[1] * su[1][0] + t.bary[2] * su[2][0],
                    t.bary[0] * su[0][1] + t.bary[1] * su[1][1] + t.bary[2] * su[2][1],
                );
                a_s[k] = sa(u, v);
            }
        }
        // Pass 1b: the cutout as a clean shape: threshold, drop specks and slivers, fill small
        // holes, then an exact signed distance (texels, + inside), lightly smoothed.
        let clean = b.alpha_min_area > 0;
        let sdist: Vec<f64> = if clean {
            let mut m: Vec<bool> = a_s.iter().map(|&a| a >= 0.5).collect();
            clean_mask(&mut m, w, h, b.alpha_min_area, b.alpha_min_width);
            let din = edt_to(&m.iter().map(|&x| !x).collect::<Vec<_>>(), w, h);
            let dout = edt_to(&m, w, h);
            let raw: Vec<f64> = (0..w * h).map(|k| if m[k] { din[k] - 0.5 } else { 0.5 - dout[k] }).collect();
            let mut sm = raw.clone();
            for y in 0..h {
                for x in 0..w {
                    let (mut acc, mut n) = (0.0, 0.0);
                    for dy in -1i64..=1 {
                        for dx in -1i64..=1 {
                            let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                            if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                                let wt = if dx == 0 && dy == 0 { 4.0 } else if dx == 0 || dy == 0 { 2.0 } else { 1.0 };
                                acc += wt * raw[yy as usize * w + xx as usize];
                                n += wt;
                            }
                        }
                    }
                    sm[y * w + x] = acc / n;
                }
            }
            sm
        } else {
            Vec::new()
        };
        let mut alpha_k = vec![f64::NAN; w * h];
        let mut si_k = vec![usize::MAX; w * h];
        // Surface strands per texel, computed on all cores (the trace lookups are the slow part).
        let f3: Vec<f64> = match b.look.and_then(|lk| lk.fibers3.as_ref()) {
            None => Vec::new(),
            Some(f) => {
                let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 8);
                let chunk = (w * h).div_ceil(threads).max(1);
                let parts: Vec<Vec<f64>> = std::thread::scope(|sc| {
                    let hands: Vec<_> = cell.chunks(chunk).map(|part| sc.spawn(move || part.iter().map(|c| c.as_ref().map_or(0.0, |t| f.raw(b.hs, t.tri, t.bary))).collect::<Vec<f64>>())).collect();
                    hands.into_iter().map(|hd| hd.join().unwrap()).collect()
                });
                parts.concat()
            }
        };
        // Pass 2: color and alpha.
        let e = b.ctx.size;
        for y in 0..h {
            for x in 0..w {
                let k = y * w + x;
                let Some(t) = &cell[k] else { continue };
                let (sx, sy) = (gx + x as i64, gy + y as i64);
                if sx < 0 || sy < 0 || sx >= size as i64 || sy >= size as i64 {
                    continue;
                }
                let si = sy as usize * size + sx as usize;
                let u = c.lo[0] + (x as f64 + 0.5 - b.pad as f64) / b.scale;
                let v = c.lo[1] + (y as f64 + 0.5 - b.pad as f64) / b.scale;
                let dens: f64 = (0..3).map(|q| t.bary[q] * b.pr.dens[t.tri][q]).sum();
                let (mut col, prof) = strand(&b.ctx, c.lock, u, v, dens);
                let grown = b.grown.get(t.tri).copied().unwrap_or(false);
                if grown {
                    // No guide paint here: the full style (bundles, clump grooves) and a dark line
                    // where the grown part tucks under the hair at its root. With surface strands
                    // the chart style is left out (it restarts on each grown triangle and showed
                    // their outlines): a plain palette tone, and the strands come from the surface.
                    if let Some(lk) = b.look.filter(|lk| lk.fibers3.is_some()) {
                        // The guide low-pass, which with surface strands covers the grown part
                        // too (its own paint, averaged with the cards it meets).
                        col = b.ctx.pal.at((v / b.ctx.vmax).clamp(0.0, 1.0));
                        if let Some((g, low)) = &lk.color {
                            if let Some(w) = g.weights(t.tri, t.bary) {
                                col = std::array::from_fn(|q| w.iter().map(|&(ni, wt)| wt * low[ni][q]).sum());
                            }
                        }
                    }
                    col[0] -= b.root_line * (1.0 - smooth(v / (0.04 * e)));
                    let pc = b.hs.corners(t.tri);
                    let p: V3 = std::array::from_fn(|q| (0..3).map(|k| t.bary[k] * pc[k][q]).sum());
                    let ty = ((p[1] - gb.2) / (gb.3 - gb.2).max(1e-9)).clamp(0.0, 1.0);
                    col[0] -= b.grown_shade * (0.5 + 0.5 * ty);
                    col[0] -= b.crevice * (1.0 - smooth((p[0] - gb.0).abs() / (0.15 * gb.1).max(1e-9)));
                } else if let (Some(lk), Some(dctx)) = (b.look, &dctx) {
                    if let Some((g, low)) = &lk.color {
                        if let Some(w) = g.weights(t.tri, t.bary) {
                            let lowc: [f64; 3] = std::array::from_fn(|q| w.iter().map(|&(ni, wt)| wt * low[ni][q]).sum());
                            let (dc, _) = strand(dctx, c.lock, u, v, dens);
                            let base = b.ctx.pal.at((v / b.ctx.vmax).clamp(0.0, 1.0));
                            col = [lowc[0] + dc[0] - base[0], lowc[1], lowc[2]];
                        }
                    }
                }
                if let Some(f) = b.look.and_then(|lk| lk.fibers3.as_ref()) {
                    col[0] += f.gain * (f3[k] - f.mean) * (col[0] / 60.0).clamp(0.4, 1.3);
                } else if let Some(fb) = b.look.and_then(|lk| lk.fibers.as_ref()) {
                    col[0] += fb.at(c.lock, u, v, b.scale) * (col[0] / 60.0).clamp(0.4, 1.3);
                }
                // Cutout distance (model units, + inside) from the stock alpha, then notches at tips.
                let at = |xx: usize, yy: usize| a_s[yy.min(h - 1) * w + xx.min(w - 1)];
                let gxa = 0.5 * (at(x + 1, y) - at(x.saturating_sub(1), y)) * b.scale;
                let gya = 0.5 * (at(x, y + 1) - at(x, y.saturating_sub(1))) * b.scale;
                let gl = (gxa * gxa + gya * gya).sqrt();
                let a0 = a_s[k];
                let d_s = if clean {
                    sdist[k] / b.scale
                } else if gl > 1e-9 {
                    (a0 - 0.5) / gl
                } else if a0 >= 0.5 {
                    1e9
                } else {
                    -1e9
                };
                let d_s = d_s.clamp(-1.0 * e, 1.0 * e);
                // Tip edges are where the alpha falls toward +v (the tip end).
                let tipward = if gl > 1e-9 { (-gya / gl).clamp(0.0, 1.0) } else { 0.0 };
                let depth = b.ctx.style.tip_depth * e * tipward;
                let d = d_s - depth * prof;
                // Signed-distance alpha: 0.5 on the contour, ramping over `sdf_spread` texels to each
                // side, so the game's 0.5 cutout follows a smooth line under magnification and mips.
                let alpha = (0.5 + d * b.scale / (2.0 * b.sdf_spread.max(0.1))).clamp(0.0, 1.0);
                // Tip lightening near tip edges.
                if tipward > 0.0 && d_s < b.ctx.style.tip_len * e {
                    col[0] += b.ctx.style.tip_light * tipward * (1.0 - smooth(d_s / (b.ctx.style.tip_len * e)));
                }
                // Ambient occlusion from the welded corners.
                let wt = b.hs.wt[t.tri];
                let bc = t.bary.map(|q| q.clamp(-0.5, 1.5));
                let mut ao = (bc[0] * b.ao[wt[0]] + bc[1] * b.ao[wt[1]] + bc[2] * b.ao[wt[2]]).clamp(0.0, 1.0);
                if let Some((g, nodes)) = b.look.and_then(|lk| lk.ao.as_ref()) {
                    if let Some(w) = g.weights(t.tri, t.bary) {
                        ao = w.iter().map(|&(ni, wt)| wt * nodes[ni]).sum::<f64>().clamp(0.0, 1.0);
                    }
                }
                col[0] *= 1.0 - b.ctx.style.ao * (1.0 - ao);
                // Strand highlight, dimmed where occluded.
                if let Some(li) = b.look.and_then(|lk| lk.light.as_ref()) {
                    let r = b.fl.region[t.tri];
                    let tg = (0..3).fold([0.0; 3], |a, q| add(a, mul(tan.get(&(wt[q], r)).copied().unwrap_or([0.0; 3]), bc[q].max(0.0))));
                    if len(tg) > 1e-12 {
                        let pi = b.hs.primary[t.tri];
                        let n = {
                            let lk = b.hs.lock[pi];
                            let s = (0..3).fold([0.0; 3], |a, q| add(a, mul(wn.get(&(wt[q], lk)).copied().unwrap_or([0.0; 3]), bc[q].max(0.0))));
                            let fnrm = mul(b.hs.normal(pi), osign[pi]);
                            if len(s) > 1e-12 && dot(unit(s), fnrm) > 0.2 { unit(s) } else { fnrm }
                        };
                        let sw = b.ctx.style.strand * e;
                        let jit = match b.look.and_then(|lk| lk.fibers3.as_ref()) {
                            Some(f) => (0.3 * (f3[k] - f.mean)).clamp(-0.75, 0.75),
                            None => vnoise(b.ctx.style.seed ^ 0x51ab, u / (2.0 * sw), v / (0.15 * e)) + 0.5 * vnoise(b.ctx.style.seed ^ 0x77, u / (7.0 * sw), v / (0.4 * e)) - 0.75,
                        };
                        let tg = unit(add(unit(tg), mul(n, li.shift * jit * 2.0)));
                        let nout = if dot(n, sub(b.hs.centroid(t.tri), hc)) < 0.0 { mul(n, -1.0) } else { n };
                        let to_l = unit(li.to_light);
                        let vw = if len(li.to_view) < 1e-9 { nout } else { unit(li.to_view) };
                        let hv = unit(add(to_l, vw));
                        let th = dot(tg, hv);
                        let spec = (1.0 - th * th).max(0.0).sqrt().powf(li.power);
                        let lit = ((dot(nout, to_l) + 0.3) / 1.3).clamp(0.0, 1.0);
                        let k = spec * (0.35 + 0.65 * ao) * lit;
                        col[0] += li.gain * k;
                        col[1] *= 1.0 - 0.25 * k;
                        col[2] *= 1.0 - 0.25 * k;
                    }
                }
                // Stay inside the hair's own colors: dark grooves keep hue and some chroma
                // (out-of-gamut darks clip to grey or green otherwise). Pulls toward the palette.
                col[0] = col[0].max(18.0);
                // Bright strands near the top of the sRGB gamut: compress lightness with a soft knee
                // below the brightest in-gamut L for this hue and chroma, so highlights keep their
                // strand variation instead of all clipping to one cream tone (a flat patch).
                col[0] = soft_top(col);
                let base = b.ctx.pal.at((v / b.ctx.vmax).clamp(0.0, 1.0));
                for _ in 0..30 {
                    let r = rgb(col);
                    let r8 = to_rgb8(col).map(|x| x as f64);
                    if r.iter().all(|&c| (0.0..=255.0).contains(&c)) && !b.ctx.pal.is_fringe_by(r, 28.0, 0.36, 14.0) && !b.ctx.pal.is_fringe_by(r8, 28.0, 0.36, 14.0) {
                        break;
                    }
                    col = [col[0] * 0.85 + base[0] * 0.15, col[1] * 0.8 + base[1] * 0.2, col[2] * 0.8 + base[2] * 0.2];
                }
                let px = to_rgb8(col);
                if dist[k] == 0.0 {
                    let su = stock_uv(t.tri);
                    let bs = t.bary;
                    let (fu, fv) = ((0..3).map(|q| bs[q] * su[q][0]).sum::<f64>(), (0..3).map(|q| bs[q] * su[q][1]).sum::<f64>());
                    let (fx, fy) = ((fu * b.stock.w as f64).floor(), (fv * b.stock.h as f64).floor());
                    if fx >= 0.0 && fy >= 0.0 && (fx as usize) < b.stock.w && (fy as usize) < b.stock.h {
                        let f = &mut map.form[fy as usize * b.stock.w + fx as usize];
                        f.0 += lab([px[0] as f64, px[1] as f64, px[2] as f64])[0];
                        f.1 += 1;
                    }
                }
                rgba[si * 4..si * 4 + 3].copy_from_slice(&px);
                rgba[si * 4 + 3] = (alpha * 255.0).round() as u8;
                alpha_k[k] = alpha;
                si_k[k] = si;
                map.chart[si] = ci;
                map.inside[si] = dist[k] == 0.0;
                if dist[k] > 0.0 {
                    map.gutter += 1;
                }
            }
        }
        // After the tip notches: drop new specks and slivers, fill new holes.
        if clean {
            // Only the chart's inside counts (gutter and other charts' texels are off), the same as
            // the islands check sees it: a sliver joined to its lock only through the gutter goes.
            let mut m: Vec<bool> = (0..w * h).map(|k| !alpha_k[k].is_nan() && dist[k] <= 0.0 && alpha_k[k] >= 0.5).collect();
            let before = m.clone();
            clean_mask(&mut m, w, h, b.alpha_min_area, b.alpha_min_width);
            for k in 0..w * h {
                if m[k] != before[k] && si_k[k] != usize::MAX {
                    let a = if m[k] { 0.6 } else { 0.3 };
                    rgba[si_k[k] * 4 + 3] = (a * 255.0f64).round() as u8;
                }
            }
        }
    }
    map.gutter_surface = gutter_surface;
    map
}

/// Ambient occlusion per weld id over both sides of the surface: fraction of `rays` cosine-weighted
/// directions that leave without hitting `occluders` (triangles as 3D corners) within `reach`.
pub fn ambient_occlusion(hs: &HairSet, occluders: &[[V3; 3]], rays: usize, reach: f64) -> Vec<f64> {
    let nw = hs.wpos.len();
    let mut nsum = vec![[0.0; 3]; nw];
    let mut used = vec![false; nw];
    for i in 0..hs.tris.len() {
        if !hs.is_primary(i) {
            continue;
        }
        let n = mul(hs.normal(i), hs.area(i));
        for &w in &hs.wt[i] {
            nsum[w] = add(nsum[w], n);
            used[w] = true;
        }
    }
    // Occluder bounding spheres, so each vertex tests only what its rays can reach.
    let spheres: Vec<(V3, f64)> = occluders
        .iter()
        .map(|t| {
            let c = mul(add(add(t[0], t[1]), t[2]), 1.0 / 3.0);
            (c, (0..3).map(|k| len(sub(t[k], c))).fold(0.0, f64::max))
        })
        .collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 8);
    let chunk = nw.div_ceil(threads).max(1);
    let mut out = vec![1.0; nw];
    std::thread::scope(|sc| {
        for (ci, part) in out.chunks_mut(chunk).enumerate() {
            let (spheres, nsum, used) = (&spheres, &nsum, &used);
            sc.spawn(move || {
                for (k, slot) in part.iter_mut().enumerate() {
                    let w = ci * chunk + k;
                    if !used[w] {
                        continue;
                    }
                    let p = hs.wpos[w];
                    let near: Vec<&[V3; 3]> = occluders.iter().zip(spheres).filter(|(_, s)| len(sub(s.0, p)) <= reach + s.1).map(|(t, _)| t).collect();
                    *slot = ao_vertex(p, nsum[w], &near, rays, reach);
                }
            });
        }
    });
    out
}

fn ao_vertex(p: V3, nsum: V3, occluders: &[&[V3; 3]], rays: usize, reach: f64) -> f64 {
    {
        let n = unit(nsum);
        if len(n) < 0.5 {
            return 1.0;
        }
        let t1 = unit(if n[0].abs() < 0.9 { cross(n, [1.0, 0.0, 0.0]) } else { cross(n, [0.0, 1.0, 0.0]) });
        let t2 = cross(n, t1);
        let mut open = 0.0f64;
        let mut total = 0.0f64;
        for side in [1.0, -1.0] {
            for r in 0..rays {
                // Cosine-weighted hemisphere, fixed (deterministic) sequence.
                let a = (r as f64 + 0.5) / rays as f64;
                let phi = r as f64 * 2.399_963;
                let (st, ct) = (a.sqrt(), (1.0 - a).sqrt());
                let d = add(add(mul(t1, st * phi.cos()), mul(t2, st * phi.sin())), mul(n, ct * side));
                let o = add(p, mul(n, 1e-4 * reach * side));
                let hit = occluders.iter().any(|t| {
                    crate::geom::ray_tri(
                        [o[0] as f32, o[1] as f32, o[2] as f32],
                        [d[0] as f32, d[1] as f32, d[2] as f32],
                        [t[0][0] as f32, t[0][1] as f32, t[0][2] as f32],
                        [t[1][0] as f32, t[1][1] as f32, t[1][2] as f32],
                        [t[2][0] as f32, t[2][1] as f32, t[2][2] as f32],
                        (2e-3 * reach) as f32,
                    )
                    .is_some_and(|tt| (tt as f64) < reach)
                });
                total += 1.0;
                if !hit {
                    open += 1.0;
                }
            }
        }
        // Two-sided: a card's texels serve both of its faces; keep the more open side's light.
        (open / total * 2.0).min(1.0)
    }
}

/// Fills texels where `valid` is false with colors pulled from valid neighbors (a pull-push
/// pyramid), with alpha 0. Only texels flagged in `fill` are written.
pub fn push_pull(rgba: &mut [u8], size: usize, valid: &[bool], fill: &[bool]) {
    // Level 1 straight from the texels; level 0 is never stored.
    let mut levels: Vec<(usize, Vec<[f32; 4]>)> = Vec::new();
    let n = size / 2;
    let mut l1 = vec![[0f32; 4]; n * n];
    for y in 0..n {
        for x in 0..n {
            let mut acc = [0f32; 4];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let i = (2 * y + dy) * size + 2 * x + dx;
                if valid[i] {
                    for c in 0..3 {
                        acc[c] += rgba[i * 4 + c] as f32;
                    }
                    acc[3] += 1.0;
                }
            }
            l1[y * n + x] = acc;
        }
    }
    levels.push((n, l1));
    while levels.last().unwrap().0 > 1 {
        let (s, prev) = levels.last().unwrap();
        let n = s / 2;
        let mut out = vec![[0f32; 4]; n * n];
        for y in 0..n {
            for x in 0..n {
                let mut acc = [0f32; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let p = prev[(2 * y + dy) * s + 2 * x + dx];
                    for c in 0..4 {
                        acc[c] += p[c];
                    }
                }
                out[y * n + x] = acc;
            }
        }
        levels.push((n, out));
    }
    // Normalize the top, then pull down: an empty texel takes its parent's color.
    let top = levels.len() - 1;
    for p in levels[top].1.iter_mut() {
        if p[3] > 0.0 {
            *p = [p[0] / p[3], p[1] / p[3], p[2] / p[3], 1.0];
        }
    }
    for l in (0..top).rev() {
        let (s, _) = levels[l];
        let (ps, parent) = (levels[l + 1].0, levels[l + 1].1.clone());
        for y in 0..s {
            for x in 0..s {
                let p = &mut levels[l].1[y * s + x];
                if p[3] > 0.0 {
                    *p = [p[0] / p[3], p[1] / p[3], p[2] / p[3], 1.0];
                } else {
                    *p = parent[(y / 2).min(ps - 1) * ps + (x / 2).min(ps - 1)];
                }
            }
        }
    }
    let l1 = &levels[0].1;
    for y in 0..size {
        for x in 0..size {
            let i = y * size + x;
            if fill[i] && !valid[i] {
                let q = l1[(y / 2).min(n - 1) * n + (x / 2).min(n - 1)];
                for c in 0..3 {
                    rgba[i * 4 + c] = q[c].round().clamp(0.0, 255.0) as u8;
                }
                rgba[i * 4 + 3] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Stage 5b: form layers. The low frequencies (lock layout, crevices, root shade, hue) come from a
// guide painted in the stock UV layout (the stock sheet by default), smoothed on the surface; the
// synthesizer adds only fine strands. Ambient occlusion and a strand highlight are baked per
// surface point, because the game's hair shader is unlit.
// ---------------------------------------------------------------------------------------------

/// The palette grade applied to one Lab color (hue rotation in degrees, chroma scale, L shift).
pub fn grade_lab(c: [f64; 3], hue_shift: f64, chroma_scale: f64, l_shift: f64) -> [f64; 3] {
    let (co, s) = (hue_shift.to_radians().cos(), hue_shift.to_radians().sin());
    [(c[0] + l_shift).clamp(0.0, 100.0), chroma_scale * (c[1] * co - c[2] * s), chroma_scale * (c[1] * s + c[2] * co)]
}

/// Sample nodes on every selected triangle: a barycentric grid with about `spacing` between nodes
/// (corner nodes are shared by position only, not by index).
pub struct TriGrid {
    pub n: Vec<usize>,
    pub start: Vec<usize>,
    pub len: usize,
}

impl TriGrid {
    pub fn new(hs: &HairSet, spacing: f64, on: impl Fn(usize) -> bool) -> TriGrid {
        let mut n = vec![0; hs.tris.len()];
        let mut start = vec![0; hs.tris.len()];
        let mut total = 0;
        for i in 0..hs.tris.len() {
            start[i] = total;
            if !on(i) {
                continue;
            }
            let c = hs.corners(i);
            let e = (0..3).map(|k| len(sub(c[k], c[(k + 1) % 3]))).fold(0.0, f64::max);
            let k = ((e / spacing.max(1e-9)).ceil() as usize).clamp(1, 96);
            n[i] = k;
            total += (k + 1) * (k + 2) / 2;
        }
        TriGrid { n, start, len: total }
    }
    fn idx(&self, t: usize, i: usize, j: usize) -> usize {
        let n = self.n[t];
        self.start[t] + j * (n + 1) - j * j.saturating_sub(1) / 2 + i
    }
    /// Nodes of triangle `t`: (node index, barycentric coordinates).
    pub fn nodes(&self, t: usize) -> Vec<(usize, [f64; 3])> {
        let n = self.n[t];
        if n == 0 {
            return Vec::new();
        }
        let nf = n as f64;
        let mut v = Vec::with_capacity((n + 1) * (n + 2) / 2);
        for j in 0..=n {
            for i in 0..=(n - j) {
                v.push((self.idx(t, i, j), [1.0 - (i + j) as f64 / nf, i as f64 / nf, j as f64 / nf]));
            }
        }
        v
    }
    /// Linear interpolation weights at barycentric `bc` (gutter texels outside the triangle clamp
    /// to its nearest cell). `None` when the triangle has no nodes.
    pub fn weights(&self, t: usize, bc: [f64; 3]) -> Option<[(usize, f64); 3]> {
        let n = self.n[t];
        if n == 0 {
            return None;
        }
        let nf = n as f64;
        let fi = (bc[1] * nf).clamp(0.0, nf);
        let fj = (bc[2] * nf).clamp(0.0, nf);
        let mut i0 = (fi.floor() as usize).min(n - 1);
        let mut j0 = (fj.floor() as usize).min(n - 1);
        while i0 + j0 > n - 1 {
            if i0 > j0 {
                i0 -= 1;
            } else {
                j0 -= 1;
            }
        }
        let (fx, fy) = (fi - i0 as f64, fj - j0 as f64);
        Some(if fx + fy <= 1.0 {
            [(self.idx(t, i0, j0), 1.0 - fx - fy), (self.idx(t, i0 + 1, j0), fx), (self.idx(t, i0, j0 + 1), fy)]
        } else if i0 + j0 + 2 <= n {
            [(self.idx(t, i0 + 1, j0 + 1), fx + fy - 1.0), (self.idx(t, i0, j0 + 1), 1.0 - fx), (self.idx(t, i0 + 1, j0), 1.0 - fy)]
        } else {
            let s = (fx + fy).max(1e-9);
            [(self.idx(t, i0, j0), 0.0), (self.idx(t, i0 + 1, j0), fx / s), (self.idx(t, i0, j0 + 1), fy / s)]
        })
    }
    pub fn pos(hs: &HairSet, t: usize, bc: [f64; 3]) -> V3 {
        let c = hs.corners(t);
        add(add(mul(c[0], bc[0]), mul(c[1], bc[1])), mul(c[2], bc[2]))
    }
}

fn cell_of(p: V3, cs: f64) -> (i64, i64, i64) {
    ((p[0] / cs).floor() as i64, (p[1] / cs).floor() as i64, (p[2] / cs).floor() as i64)
}

/// Separable Gaussian blur of `ch` channels (row-major w x h), sigma in pixels.
pub fn blur(data: &mut [f32], w: usize, h: usize, ch: usize, sigma: f64) {
    if sigma <= 0.05 {
        return;
    }
    let r = (3.0 * sigma).ceil() as i64;
    let k: Vec<f32> = (-r..=r).map(|x| (-(x * x) as f64 / (2.0 * sigma * sigma)).exp() as f32).collect();
    let mut tmp = vec![0f32; data.len()];
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let mut a = 0f32;
                for (o, kv) in (-r..=r).zip(&k) {
                    let xx = x as i64 + o;
                    if xx >= 0 && xx < w as i64 {
                        a += kv * data[(y * w + xx as usize) * ch + c];
                    }
                }
                tmp[(y * w + x) * ch + c] = a;
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let mut a = 0f32;
                for (o, kv) in (-r..=r).zip(&k) {
                    let yy = y as i64 + o;
                    if yy >= 0 && yy < h as i64 {
                        a += kv * tmp[(yy as usize * w + x) * ch + c];
                    }
                }
                data[(y * w + x) * ch + c] = a;
            }
        }
    }
}

/// Low-pass Lab of a guide image (painted in the stock UV layout) on the hair surface, per node of
/// `grid`. The guide is blurred in its own space over hair texels only (`sigma_tex`, guide pixels;
/// gutters and other parts never bleed in), sampled densely on the surface, then averaged in 3D
/// within `radius` over samples of the same lock facing the same way. The 3D average is
/// continuous across the stock UV seams, so the new charts meet without the stock's seam jumps.
/// `use_tri`: triangles whose guide paint counts (false for triangles grown into the hair part).
/// Nodes with no guide nearby are `None`.
pub fn guide_lowpass(model: &Model, hs: &HairSet, guide: &Sheet, use_tri: &[bool], sigma_tex: f64, radius: f64, grid: &TriGrid) -> Vec<Option<[f64; 3]>> {
    guide_lowpass_grouped(model, hs, guide, use_tri, &hs.lock, sigma_tex, radius, grid)
}

/// Locks joined across every welded edge: a lock grown into the part (its own lock, see `relock`)
/// joins the cards it touches. `guide_lowpass_grouped` averages within these groups, so the guide
/// runs on across the hairline instead of stepping where the grown triangles meet the cards.
pub fn guide_groups(hs: &HairSet) -> Vec<usize> {
    let n = hs.lock.iter().copied().max().map_or(0, |m| m + 1);
    let mut par: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let mut first: HashMap<(usize, usize), usize> = HashMap::new();
    for i in 0..hs.tris.len() {
        if !hs.is_primary(i) {
            continue;
        }
        let w = hs.wt[i];
        for k in 0..3 {
            let e = (w[k].min(w[(k + 1) % 3]), w[k].max(w[(k + 1) % 3]));
            let l = hs.lock[i];
            let o = *first.entry(e).or_insert(l);
            let (a, b) = (find(&mut par, o), find(&mut par, l));
            if a != b {
                par[a] = b;
            }
        }
    }
    hs.lock.iter().map(|&l| find(&mut par, l)).collect()
}

/// `guide_lowpass` averaging over samples of the same `group` (per triangle) instead of the same lock.
#[allow(clippy::too_many_arguments)]
pub fn guide_lowpass_grouped(model: &Model, hs: &HairSet, guide: &Sheet, use_tri: &[bool], group: &[usize], sigma_tex: f64, radius: f64, grid: &TriGrid) -> Vec<Option<[f64; 3]>> {
    let (gw, gh) = (guide.w, guide.h);
    let sign = orient(hs);
    // Hair texels of the guide.
    let mut mask = vec![false; gw * gh];
    for i in 0..hs.tris.len() {
        if !hs.is_primary(i) || !use_tri[i] {
            continue;
        }
        let p = tri_uv(model, hs, i).map(|q| [q[0] * gw as f64, q[1] * gh as f64]);
        crate::uv::raster(p, gw, gh, 0.0, |x, y| mask[y * gw + x] = true);
    }
    let mut acc = vec![0f32; gw * gh * 4];
    for y in 0..gh {
        for x in 0..gw {
            let px = guide.px(x as i64, y as i64);
            if !mask[y * gw + x] || px[3] < 128.0 {
                continue;
            }
            let l = lab([px[0], px[1], px[2]]);
            let k = (y * gw + x) * 4;
            acc[k] = l[0] as f32;
            acc[k + 1] = l[1] as f32;
            acc[k + 2] = l[2] as f32;
            acc[k + 3] = 1.0;
        }
    }
    blur(&mut acc, gw, gh, 4, sigma_tex);
    let sample = |u: f64, v: f64| -> Option<([f64; 3], f64)> {
        let x = u * gw as f64 - 0.5;
        let y = v * gh as f64 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let mut s = [0f64; 4];
        for (dx, dy, w) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
            let xx = (x0 as i64 + dx).clamp(0, gw as i64 - 1) as usize;
            let yy = (y0 as i64 + dy).clamp(0, gh as i64 - 1) as usize;
            for c in 0..4 {
                s[c] += w * acc[(yy * gw + xx) * 4 + c] as f64;
            }
        }
        (s[3] > 0.05).then(|| ([s[0] / s[3], s[1] / s[3], s[2] / s[3]], s[3]))
    };
    // Dense samples on the surface.
    let cloud_grid = TriGrid::new(hs, radius / 3.0, |i| hs.is_primary(i) && use_tri[i]);
    let mut cloud: Vec<(V3, usize, V3, [f64; 3], f64)> = Vec::new();
    for i in 0..hs.tris.len() {
        let su = tri_uv(model, hs, i);
        let n = mul(hs.normal(i), sign[i]);
        for (_, bc) in cloud_grid.nodes(i) {
            let (u, v) = ((0..3).map(|k| bc[k] * su[k][0]).sum::<f64>(), (0..3).map(|k| bc[k] * su[k][1]).sum::<f64>());
            if let Some((l, w)) = sample(u, v) {
                cloud.push((TriGrid::pos(hs, i, bc), group[i], n, l, w));
            }
        }
    }
    let mut cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (k, c) in cloud.iter().enumerate() {
        cells.entry(cell_of(c.0, radius)).or_default().push(k);
    }
    // Per node: 3D Gaussian average.
    let mut jobs: Vec<(usize, usize, [f64; 3])> = Vec::new();
    for i in 0..hs.tris.len() {
        for (ni, bc) in grid.nodes(i) {
            jobs.push((ni, i, bc));
        }
    }
    let mut out: Vec<Option<[f64; 3]>> = vec![None; grid.len];
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 8);
    let chunk = jobs.len().div_ceil(threads).max(1);
    let res: Vec<Vec<(usize, Option<[f64; 3]>)>> = std::thread::scope(|sc| {
        let hs_ = &hs;
        let (cloud, cells, sign) = (&cloud, &cells, &sign);
        let hands: Vec<_> = jobs
            .chunks(chunk)
            .map(|part| {
                sc.spawn(move || {
                    let s2 = (0.5 * radius) * (0.5 * radius);
                    part.iter()
                        .map(|&(ni, i, bc)| {
                            let p = TriGrid::pos(hs_, i, bc);
                            let n = mul(hs_.normal(i), sign[i]);
                            let (cx, cy, cz) = cell_of(p, radius);
                            let mut acc = [0f64; 4];
                            for dx in -1..=1 {
                                for dy in -1..=1 {
                                    for dz in -1..=1 {
                                        let Some(list) = cells.get(&(cx + dx, cy + dy, cz + dz)) else { continue };
                                        for &k in list {
                                            let c = &cloud[k];
                                            if c.1 != group[i] || dot(c.2, n) < 0.3 {
                                                continue;
                                            }
                                            let d = sub(c.0, p);
                                            let d2 = dot(d, d);
                                            if d2 > radius * radius {
                                                continue;
                                            }
                                            let w = (-d2 / (2.0 * s2)).exp() * c.4;
                                            for q in 0..3 {
                                                acc[q] += w * c.3[q];
                                            }
                                            acc[3] += w;
                                        }
                                    }
                                }
                            }
                            (ni, (acc[3] > 1e-6).then(|| [acc[0] / acc[3], acc[1] / acc[3], acc[2] / acc[3]]))
                        })
                        .collect()
                })
            })
            .collect();
        hands.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for part in res {
        for (ni, v) in part {
            out[ni] = v;
        }
    }
    out
}

/// Ambient occlusion at every node of `grid`, both sides of the card (the more open side counts,
/// as in `ambient_occlusion`), against `occluders` within `reach`. A uniform cell grid keeps each
/// ray to the triangles along it.
pub fn surface_ao(hs: &HairSet, grid: &TriGrid, occluders: &[[V3; 3]], rays: usize, reach: f64) -> Vec<f64> {
    let cs = reach / 3.0;
    let mut cells: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    for (k, t) in occluders.iter().enumerate() {
        let lo = cell_of([t[0][0].min(t[1][0]).min(t[2][0]), t[0][1].min(t[1][1]).min(t[2][1]), t[0][2].min(t[1][2]).min(t[2][2])], cs);
        let hi = cell_of([t[0][0].max(t[1][0]).max(t[2][0]), t[0][1].max(t[1][1]).max(t[2][1]), t[0][2].max(t[1][2]).max(t[2][2])], cs);
        for x in lo.0..=hi.0 {
            for y in lo.1..=hi.1 {
                for z in lo.2..=hi.2 {
                    cells.entry((x, y, z)).or_default().push(k as u32);
                }
            }
        }
    }
    let mut jobs: Vec<(usize, usize, [f64; 3])> = Vec::new();
    for i in 0..hs.tris.len() {
        for (ni, bc) in grid.nodes(i) {
            jobs.push((ni, i, bc));
        }
    }
    let mut out = vec![1.0; grid.len];
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 8);
    let chunk = jobs.len().div_ceil(threads).max(1);
    let res: Vec<Vec<(usize, f64)>> = std::thread::scope(|sc| {
        let cells = &cells;
        let hands: Vec<_> = jobs
            .chunks(chunk)
            .map(|part| {
                sc.spawn(move || {
                    let mut stamp = vec![0u32; occluders.len()];
                    let mut ray_id = 0u32;
                    part.iter()
                        .map(|&(ni, i, bc)| {
                            let p = TriGrid::pos(hs, i, bc);
                            let n = hs.normal(i);
                            let t1 = unit(if n[0].abs() < 0.9 { cross(n, [1.0, 0.0, 0.0]) } else { cross(n, [0.0, 1.0, 0.0]) });
                            let t2 = cross(n, t1);
                            let mut best = 0.0f64;
                            for side in [1.0, -1.0] {
                                let mut open = 0usize;
                                for r in 0..rays {
                                    let a = (r as f64 + 0.5) / rays as f64;
                                    let phi = r as f64 * 2.399_963;
                                    let (st, ct) = (a.sqrt(), (1.0 - a).sqrt());
                                    let d = add(add(mul(t1, st * phi.cos()), mul(t2, st * phi.sin())), mul(n, ct * side));
                                    let o = add(p, mul(n, 1e-4 * reach * side));
                                    ray_id = ray_id.wrapping_add(1);
                                    let mut hit = false;
                                    let steps = 10;
                                    'march: for s in 0..=steps {
                                        let q = add(o, mul(d, reach * s as f64 / steps as f64));
                                        let Some(list) = cells.get(&cell_of(q, cs)) else { continue };
                                        for &k in list {
                                            if stamp[k as usize] == ray_id {
                                                continue;
                                            }
                                            stamp[k as usize] = ray_id;
                                            let t = &occluders[k as usize];
                                            let f = |v: V3| [v[0] as f32, v[1] as f32, v[2] as f32];
                                            if crate::geom::ray_tri(f(o), f(d), f(t[0]), f(t[1]), f(t[2]), (2e-3 * reach) as f32).is_some_and(|tt| (tt as f64) < reach) {
                                                hit = true;
                                                break 'march;
                                            }
                                        }
                                    }
                                    if !hit {
                                        open += 1;
                                    }
                                }
                                best = best.max(open as f64 / rays as f64);
                            }
                            (ni, best)
                        })
                        .collect()
                })
            })
            .collect();
        hands.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for part in res {
        for (ni, v) in part {
            out[ni] = v;
        }
    }
    out
}

/// Baked strand highlight (Kajiya-Kay): bright where the strand direction is perpendicular to the
/// half vector of a fixed light and a fixed viewer. Strands are shifted along the normal by noise,
/// so the band breaks into strands instead of a flat stripe.
#[derive(Clone, Debug)]
pub struct Light {
    /// Direction to the light and to the viewer (model space). A zero `to_view` views each
    /// surface along its outward normal, so the band sits on every side's own crown (a baked
    /// highlight is seen from all sides); the band fades on the side facing away from the light.
    pub to_light: V3,
    pub to_view: V3,
    /// Sharpness (exponent) and strength (Lab L at the peak).
    pub power: f64,
    pub gain: f64,
    /// Per-strand tangent shift along the normal (0..1).
    pub shift: f64,
}

/// Form layers for `bake`.
pub struct Look {
    /// Low-pass Lab per node (already graded), from `guide_lowpass`; the synthesizer then adds
    /// only the strands of `detail`.
    pub color: Option<(TriGrid, Vec<[f64; 3]>)>,
    pub detail: Style,
    /// Ambient occlusion per node (1 = open); replaces the per-vertex occlusion.
    pub ao: Option<(TriGrid, Vec<f64>)>,
    pub light: Option<Light>,
    /// Crisp strand layer on top of everything (see `Fibers`).
    pub fibers: Option<Fibers>,
    /// Strands traced on the 3D surface (`hairfiber`); used instead of `fibers` when set, and it
    /// also drives the highlight's strand jitter, so neither restarts at chart seams.
    pub fibers3: Option<crate::hairfiber::Fibers3>,
}

impl Look {
    /// The detail style: fine strands and a little clump variation, no shading terms (those come
    /// from the guide). `strands` and `clumps` scale the style's own strand and clump contrast.
    pub fn detail_style(s: &Style, strands: f64, clumps: f64) -> Style {
        Style { strand_contrast: s.strand_contrast * strands, clump_contrast: s.clump_contrast * clumps, bundle: 0.0, clump_edge_dark: 0.0, root_dark: 0.0, band_gain: 0.0, tip_light: 0.0, ..s.clone() }
    }
}

// ---------------------------------------------------------------------------------------------
// Stage 6: checks that match the eye
// ---------------------------------------------------------------------------------------------

/// How the strand direction behaves across one welded edge between two charts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum JoinKind {
    /// Texture strands continue across the edge.
    Continuous,
    /// Both sides run away from the edge: a parting (expected where locks split).
    Parting,
    /// Both sides run into the edge where the surface folds back: the shared tip of a folded card.
    FoldTip,
    /// Both sides run into the edge at an angle (not head on) without a fold: strands converging
    /// into a point, like the two sides of a V-shaped tip.
    Merge,
    /// Strands turn sharply or meet head on (a "V"): a flow break.
    Break,
}

impl JoinKind {
    pub fn label(self) -> &'static str {
        match self {
            JoinKind::Continuous => "continuous",
            JoinKind::Parting => "parting",
            JoinKind::FoldTip => "fold-tip",
            JoinKind::Merge => "merge",
            JoinKind::Break => "BREAK",
        }
    }
}

pub struct Join {
    pub a: usize,
    pub b: usize,
    pub chart_a: usize,
    pub chart_b: usize,
    pub kind: JoinKind,
    /// Angle (degrees) between the two sides' texture strand directions, unfolded into one plane.
    pub turn: f64,
    /// Worst angle between a side's texture strand direction and its flow.
    pub tex_vs_flow: f64,
    /// Mean and max Lab difference across the edge (sheet level 0 and a mip level).
    pub de0: f64,
    pub de0_max: f64,
    pub de_mip: f64,
    pub samples: usize,
}

/// 3D direction of +v on the sheet for a triangle with sheet UVs `suv`.
pub fn tex_v_dir(p: [V3; 3], suv: [[f64; 2]; 3]) -> Option<V3> {
    uv_dir_to_3d(p, suv, [0.0, 1.0]).map(unit)
}

/// Bilinear sample of an RGBA8 level.
pub fn sample_level(w: usize, h: usize, px: &[u8], u: f64, v: f64) -> [f64; 4] {
    Sheet { w, h, rgba: px }.bilinear(u, v)
}

pub struct JoinIn<'a> {
    pub hs: &'a HairSet,
    pub fl: &'a Flow,
    /// Sheet UV per selected triangle corner (the output layout, or the stock one).
    pub suv: Vec<[[f64; 2]; 3]>,
    /// Chart per selected triangle (or stock UV island).
    pub chart_of: Vec<usize>,
    /// The sheet and one of its mip levels: (w, h, rgba).
    pub level0: (usize, usize, &'a [u8]),
    pub mip: (usize, usize, &'a [u8]),
}

/// Every welded edge between two primary triangles that sit in different charts.
pub fn joins(j: &JoinIn) -> Vec<Join> {
    let hs = j.hs;
    let nb = neighbors(hs);
    let mut out = Vec::new();
    for a in 0..hs.tris.len() {
        for &(b, e) in &nb[a] {
            if b <= a || j.chart_of[a] == j.chart_of[b] {
                continue;
            }
            let pa = hs.corners(a);
            let pb = hs.corners(b);
            let (Some(ta), Some(tb)) = (tex_v_dir(pa, j.suv[a]), tex_v_dir(pb, j.suv[b])) else { continue };
            let tb_in_a = hinge(hs, b, a, e, tb);
            let turn = ang(ta, tb_in_a);
            let tex_vs_flow = ang(ta, j.fl.dir[a]).max(ang(tb, j.fl.dir[b]));
            // Edge frame in a: does each side's strand run away from the edge or into it?
            let ax = unit(sub(hs.wpos[e.1], hs.wpos[e.0]));
            let mid = mul(add(hs.wpos[e.0], hs.wpos[e.1]), 0.5);
            let into_a = unit(project_plane(sub(hs.centroid(a), mid), ax));
            let into_b = unit(project_plane(sub(hs.centroid(b), mid), ax));
            let (sa, sb) = (dot(ta, into_a), dot(tb, into_b));
            let fwd = |t: [usize; 3]| (0..3).any(|k| t[k] == e.0 && t[(k + 1) % 3] == e.1);
            let nb_ = if fwd(hs.wt[a]) != fwd(hs.wt[b]) { hs.normal(b) } else { mul(hs.normal(b), -1.0) };
            let fold = dot(hs.normal(a), nb_) < -0.5;
            let kind = if turn <= 45.0 {
                JoinKind::Continuous
            } else if sa > 0.3 && sb > 0.3 {
                JoinKind::Parting
            } else if sa < -0.3 && sb < -0.3 && fold {
                JoinKind::FoldTip
            } else if sa < -0.3 && sb < -0.3 && turn < 150.0 {
                JoinKind::Merge
            } else {
                JoinKind::Break
            };
            let kind = if tex_vs_flow > 45.0 { JoinKind::Break } else { kind };
            // Lab across the edge: points along it, nudged 1.5 texels into each side.
            let (mut s0, mut m0, mut sm, mut n) = (0.0, 0.0f64, 0.0, 0usize);
            for f in [0.2, 0.35, 0.5, 0.65, 0.8] {
                let side = |t: usize, (w, h, px): (usize, usize, &[u8])| -> Option<[f64; 4]> {
                    let c = hs.wt[t];
                    let k0 = c.iter().position(|&x| x == e.0)?;
                    let k1 = c.iter().position(|&x| x == e.1)?;
                    let k2 = 3 - k0 - k1;
                    let q0 = j.suv[t][k0];
                    let q1 = j.suv[t][k1];
                    let q2 = j.suv[t][k2];
                    let pnt = [q0[0] + (q1[0] - q0[0]) * f, q0[1] + (q1[1] - q0[1]) * f];
                    let cen = [(q0[0] + q1[0] + q2[0]) / 3.0, (q0[1] + q1[1] + q2[1]) / 3.0];
                    let (dx, dy) = ((cen[0] - pnt[0]) * w as f64, (cen[1] - pnt[1]) * h as f64);
                    let l = (dx * dx + dy * dy).sqrt().max(1e-9);
                    let step = f64::min(1.5, 0.5 * l);
                    Some(sample_level(w, h, px, pnt[0] + dx / l * step / w as f64, pnt[1] + dy / l * step / h as f64))
                };
                let (Some(ca), Some(cb)) = (side(a, j.level0), side(b, j.level0)) else { continue };
                if ca[3] < 128.0 || cb[3] < 128.0 {
                    continue;
                }
                let d = delta_e(lab([ca[0], ca[1], ca[2]]), lab([cb[0], cb[1], cb[2]]));
                s0 += d;
                m0 = m0.max(d);
                if let (Some(ma), Some(mb)) = (side(a, j.mip), side(b, j.mip)) {
                    sm += delta_e(lab([ma[0], ma[1], ma[2]]), lab([mb[0], mb[1], mb[2]]));
                }
                n += 1;
            }
            let k = n.max(1) as f64;
            out.push(Join { a, b, chart_a: j.chart_of[a], chart_b: j.chart_of[b], kind, turn, tex_vs_flow, de0: s0 / k, de0_max: m0, de_mip: sm / k, samples: n });
        }
    }
    out
}

/// Fringe texels: hair texels that show (alpha above `min_alpha`) in a color that is not hair.
/// `hair` flags hair texels at level 0; a texel of a smaller level counts as hair when at least
/// half of its level-0 footprint is hair (otherwise it mostly shows whatever else is there).
pub fn fringe(tex: &crate::Texture, hair: &[bool], pal: &Palette, levels: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    let mut frac: Vec<f32> = hair.iter().map(|&h| if h { 1.0 } else { 0.0 }).collect();
    let (mut mw, mut mh) = (tex.w, tex.h);
    for l in 0..levels.min(tex.levels()) {
        let (w, h, px) = tex.level(l);
        if l > 0 {
            let mut f2 = vec![0.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let at = |xx: usize, yy: usize| frac[yy.min(mh - 1) * mw + xx.min(mw - 1)];
                    f2[y * w + x] = 0.25 * (at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1));
                }
            }
            frac = f2;
            (mw, mh) = (w, h);
        }
        let mask: Vec<bool> = frac.iter().map(|&f| f >= 0.5).collect();
        let min_alpha = if l == 0 { 1 } else { 128 };
        let (mut count, mut shown) = (0, 0);
        for i in 0..w * h {
            if !mask[i] || px[i * 4 + 3] < min_alpha {
                continue;
            }
            shown += 1;
            if pal.is_fringe([px[i * 4] as f64, px[i * 4 + 1] as f64, px[i * 4 + 2] as f64]) {
                count += 1;
                // GAW_FRINGE_AT=1 prints where the first few fringe texels are (for debugging).
                if count <= 8 && std::env::var_os("GAW_FRINGE_AT").is_some() {
                    eprintln!("fringe texel: level {l} at ({}, {}) rgba {:?} hair fraction {:.2}", i % w, i / w, &px[i * 4..i * 4 + 4], frac[i]);
                }
            }
        }
        out.push((l, count, shown));
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Seam stitch (safety net) and alpha coverage per mip
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct SeamReport {
    /// Points sampled on chart seams inside one flow region (strands should continue).
    pub points: usize,
    /// Lab difference between the two sides' bilinear samples at the same seam point.
    pub mean_before: f64,
    pub max_before: f64,
    pub over_before: usize,
    pub mean_after: f64,
    pub max_after: f64,
    pub over_after: usize,
    /// Gutter texels the stitch changed, and the mean change (RGB, 0..255).
    pub texels_changed: usize,
    pub mean_change: f64,
}

type SeamSample = (f64, [f64; 4], [f64; 4], [f64; 2], [f64; 2]);

/// Samples every welded edge between two charts of the same flow region at the edge itself, from
/// both sides, and compares the bilinear colors (they match when the texture is authored on the
/// surface). With `tol` > 0, nudges GUTTER texels (never inside texels) so both sides meet at their
/// mean where they differ by more than `tol` (Lab), then measures again.
#[allow(clippy::too_many_arguments)]
pub fn seam_stitch(hs: &HairSet, fl: &Flow, suv: &[[[f64; 2]; 3]], chart_of: &[usize], inside: &[bool], rgba: &mut [u8], size: usize, tol: f64) -> SeamReport {
    let nb = neighbors(hs);
    let mut pairs = Vec::new();
    for a in 0..hs.tris.len() {
        for &(bt, e) in &nb[a] {
            if bt > a && chart_of[a] != chart_of[bt] && fl.region[a] == fl.region[bt] {
                pairs.push((a, bt, e));
            }
        }
    }
    let pt = |t: usize, e: (usize, usize), f: f64| -> Option<[f64; 2]> {
        let c = hs.wt[t];
        let k0 = c.iter().position(|&x| x == e.0)?;
        let k1 = c.iter().position(|&x| x == e.1)?;
        let (q0, q1) = (suv[t][k0], suv[t][k1]);
        Some([q0[0] + (q1[0] - q0[0]) * f, q0[1] + (q1[1] - q0[1]) * f])
    };
    let measure = |rgba: &[u8]| -> Vec<SeamSample> {
        let mut v = Vec::new();
        for &(a, bt, e) in &pairs {
            let (Some(a0), Some(a1)) = (pt(a, e, 0.0), pt(a, e, 1.0)) else { continue };
            let n = (((a1[0] - a0[0]) * size as f64).hypot((a1[1] - a0[1]) * size as f64)).ceil().max(1.0) as usize;
            for s in 0..n {
                let f = (s as f64 + 0.5) / n as f64;
                let (Some(pa), Some(pb)) = (pt(a, e, f), pt(bt, e, f)) else { continue };
                let ca = sample_level(size, size, rgba, pa[0], pa[1]);
                let cb = sample_level(size, size, rgba, pb[0], pb[1]);
                if ca[3] < 128.0 || cb[3] < 128.0 {
                    continue;
                }
                let d = delta_e(lab([ca[0], ca[1], ca[2]]), lab([cb[0], cb[1], cb[2]]));
                v.push((d, ca, cb, pa, pb));
            }
        }
        v
    };
    let lim = tol.max(2.0);
    let stats = |v: &[SeamSample]| {
        let n = v.len().max(1) as f64;
        (v.iter().map(|x| x.0).sum::<f64>() / n, v.iter().map(|x| x.0).fold(0.0, f64::max), v.iter().filter(|x| x.0 > lim).count())
    };
    let before = measure(rgba);
    let (mb, xb, ob) = stats(&before);
    let mut rep = SeamReport { points: before.len(), mean_before: mb, max_before: xb, over_before: ob, mean_after: mb, max_after: xb, over_after: ob, ..Default::default() };
    if tol <= 0.0 {
        return rep;
    }
    // Desired change per gutter texel, averaged over the seam points that touch it.
    let mut want: HashMap<usize, ([f64; 3], f64)> = HashMap::new();
    for &(d, ca, cb, pa, pb) in &before {
        if d <= tol {
            continue;
        }
        let target = [(ca[0] + cb[0]) * 0.5, (ca[1] + cb[1]) * 0.5, (ca[2] + cb[2]) * 0.5];
        for (p, c) in [(pa, ca), (pb, cb)] {
            let x = p[0] * size as f64 - 0.5;
            let y = p[1] * size as f64 - 0.5;
            let (x0, y0) = (x.floor() as i64, y.floor() as i64);
            let (fx, fy) = (x - x.floor(), y - y.floor());
            let taps = [(0i64, 0i64, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)];
            let idx = |t: &(i64, i64, f64)| -> Option<usize> {
                let (xx, yy) = (x0 + t.0, y0 + t.1);
                (xx >= 0 && yy >= 0 && (xx as usize) < size && (yy as usize) < size).then(|| yy as usize * size + xx as usize)
            };
            let gw: f64 = taps.iter().filter(|t| idx(t).is_some_and(|i| !inside[i])).map(|t| t.2).sum();
            if gw < 0.25 {
                continue;
            }
            for t in &taps {
                let Some(i) = idx(t) else { continue };
                if inside[i] || t.2 <= 0.0 {
                    continue;
                }
                let e = want.entry(i).or_insert(([0.0; 3], 0.0));
                for k in 0..3 {
                    e.0[k] += (target[k] - c[k]) / gw * t.2;
                }
                e.1 += t.2;
            }
        }
    }
    let mut total = 0.0;
    for (&i, &(d, w)) in &want {
        let mut ch = 0.0;
        for k in 0..3 {
            let old = rgba[i * 4 + k] as f64;
            let new = (old + d[k] / w.max(1e-9)).round().clamp(0.0, 255.0);
            ch += (new - old).abs() / 3.0;
            rgba[i * 4 + k] = new as u8;
        }
        total += ch;
    }
    rep.texels_changed = want.len();
    rep.mean_change = total / want.len().max(1) as f64;
    let after = measure(rgba);
    let (ma, xa, oa) = stats(&after);
    rep.mean_after = ma;
    rep.max_after = xa;
    rep.over_after = oa;
    rep
}

/// Share of hair texels at alpha >= 0.5 per mip level (box mips). `mask` flags level-0 hair texels;
/// a smaller level's texel counts by the share of its footprint in the mask. A cutout that keeps
/// its coverage at every mip keeps its outline at distance.
pub fn alpha_coverage(tex: &crate::Texture, mask: &[bool], levels: usize) -> Vec<f64> {
    let mut out = Vec::new();
    let mut frac: Vec<f32> = mask.iter().map(|&h| if h { 1.0 } else { 0.0 }).collect();
    let (mut mw, mut mh) = (tex.w, tex.h);
    for l in 0..levels.min(tex.levels()) {
        let (w, h, px) = tex.level(l);
        if l > 0 {
            let mut f2 = vec![0.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let at = |xx: usize, yy: usize| frac[yy.min(mh - 1) * mw + xx.min(mw - 1)];
                    f2[y * w + x] = 0.25 * (at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1));
                }
            }
            frac = f2;
            (mw, mh) = (w, h);
        }
        let (mut cov, mut tot) = (0.0f64, 0.0f64);
        for i in 0..w * h {
            if frac[i] <= 0.0 {
                continue;
            }
            tot += frac[i] as f64;
            if px[i * 4 + 3] >= 128 {
                cov += frac[i] as f64;
            }
        }
        out.push(if tot > 0.0 { cov / tot } else { 0.0 });
    }
    out
}

/// What `to_stock_layout` wrote.
#[derive(Debug, Default, Clone)]
pub struct LayoutOut {
    /// Texels inside a hair triangle of the input layout.
    pub inside: usize,
    /// Texels within `pad` of one (gutter), extrapolated from the nearest triangle.
    pub gutter: usize,
    /// Inside texels that two places use, one on each side of x = 0 (a mirrored layout). The +x
    /// side is painted; the other side shows its mirror image.
    pub mirrored: usize,
    /// Hair texels left alone because another part uses them too.
    pub shared_other: usize,
    /// Inside texels blended with a card that passes through theirs (`cross_r`).
    pub crossed: usize,
    /// Layout edges shared by two hair triangles, and texels blended across them (`seam_w`).
    pub seams: usize,
    pub seam_texels: usize,
}

/// Moves a hair bake into the input's own UV layout, so the hair can ship as a texture alone (stock
/// mesh, stock UVs, a larger sheet). `stock_uv` and `tool_uv` are per hair-set triangle in 0..1;
/// `tool` is the baked sheet (`tool_size` square) whose texels `tool_uv` addresses. Every texel of
/// `out` (`size` square, already holding the stock sheet scaled up) that lies inside or within
/// `pad` texels of a hair triangle, and is not in `occ`, takes the bake sampled at the same surface
/// point. Duplicate faces sample through their primary. Where the layout reuses texels for two
/// places, the place with the larger x wins, so the layout's own mirror decides the other side.
///
/// `cross_r` > 0 (model units) blends cards that pass through each other: a texel whose surface
/// point lies within `cross_r` of another card's plane, over that card, takes a share of that
/// card's paint at the same point, fading to none at `cross_r` from the plane. The share is all of
/// it at this card's own outline, none at the other card's outline and half well inside both, so
/// two cards that cross or lie flush show the same colour where they meet and neither outline
/// reads as a cut. Alpha stays the texel's own.
#[allow(clippy::too_many_arguments)]
pub fn to_stock_layout(hs: &HairSet, stock_uv: &[[[f64; 2]; 3]], tool_uv: &[[[f64; 2]; 3]], tool: &[u8], tool_size: usize, occ: &[bool], out: &mut [u8], size: usize, pad: f64, cross_r: f64, seam_w: f64) -> LayoutOut {
    let n = hs.tris.len();
    // Cards that pass through each other: per primary, the other primaries (no shared edge; one
    // shared corner is allowed, as cards that fan out from one point overlap near it)
    // whose box comes within `cross_r` of its own.
    let partners: Vec<Vec<usize>> = if cross_r > 0.0 {
        let bb: Vec<(V3, V3)> = (0..n)
            .map(|i| {
                let c = hs.corners(i);
                (std::array::from_fn(|k| c.iter().map(|p| p[k]).fold(f64::MAX, f64::min) - cross_r), std::array::from_fn(|k| c.iter().map(|p| p[k]).fold(f64::MIN, f64::max) + cross_r))
            })
            .collect();
        (0..n)
            .map(|i| {
                if !hs.is_primary(i) {
                    return Vec::new();
                }
                (0..n)
                    .filter(|&j| j != i && hs.is_primary(j) && hs.wt[j].iter().filter(|w| hs.wt[i].contains(w)).count() < 2 && (0..3).all(|k| bb[i].0[k] <= bb[j].1[k] && bb[j].0[k] <= bb[i].1[k]))
                    .collect()
            })
            .collect()
    } else {
        vec![Vec::new(); n]
    };
    // Open edges (used by one primary triangle only): the card outlines.
    let open: Vec<Vec<(V3, V3)>> = if cross_r > 0.0 {
        let mut uses: HashMap<(usize, usize), usize> = HashMap::new();
        for i in (0..n).filter(|&i| hs.is_primary(i)) {
            for c in 0..3 {
                let (a, b) = (hs.wt[i][c], hs.wt[i][(c + 1) % 3]);
                *uses.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        (0..n)
            .map(|i| {
                (0..3)
                    .filter(|&c| {
                        let (a, b) = (hs.wt[i][c], hs.wt[i][(c + 1) % 3]);
                        uses.get(&(a.min(b), a.max(b))).copied().unwrap_or(0) <= 1
                    })
                    .map(|c| (hs.wpos[hs.wt[i][c]], hs.wpos[hs.wt[i][(c + 1) % 3]]))
                    .collect()
            })
            .collect()
    } else {
        vec![Vec::new(); n]
    };
    // How far inside its card a point is, as a 0..1 fade over `cross_r` from the outline.
    let inside_card = |t: usize, q: V3| -> f64 {
        let d = open[t].iter().map(|&(a, b)| seg_dist3(q, a, b)).fold(f64::MAX, f64::min);
        (d / cross_r).min(1.0)
    };
    // Per texel: (distance, priority, triangle, other side seen).
    let mut best: Vec<(f64, f64, usize, bool)> = vec![(f64::MAX, f64::MIN, usize::MAX, false); size * size];
    let sz = size as f64;
    for i in 0..n {
        let p: [[f64; 2]; 3] = std::array::from_fn(|k| [stock_uv[i][k][0] * sz, stock_uv[i][k][1] * sz]);
        let cx = hs.centroid(i)[0];
        let pri = if hs.is_primary(i) { 1e3 } else { 0.0 } + cx;
        let x0 = (p.iter().map(|a| a[0]).fold(f64::MAX, f64::min) - pad).floor().max(0.0) as usize;
        let x1 = ((p.iter().map(|a| a[0]).fold(f64::MIN, f64::max) + pad).ceil().max(0.0) as usize).min(size - 1);
        let y0 = (p.iter().map(|a| a[1]).fold(f64::MAX, f64::min) - pad).floor().max(0.0) as usize;
        let y1 = ((p.iter().map(|a| a[1]).fold(f64::MIN, f64::max) + pad).ceil().max(0.0) as usize).min(size - 1);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let d = point_tri_dist([x as f64 + 0.5, y as f64 + 0.5], p);
                if d > pad {
                    continue;
                }
                let b = &mut best[y * size + x];
                if d == 0.0 && b.0 == 0.0 && b.2 != usize::MAX && (hs.centroid(b.2)[0] > 0.02) != (cx > 0.02) {
                    b.3 = true;
                }
                if d < b.0 - 1e-9 || (d <= b.0 + 1e-9 && pri > b.1) {
                    let seen = b.3;
                    *b = (d, pri, i, seen);
                }
            }
        }
    }
    let ts = tool_size as f64;
    let sample = |u: f64, v: f64| -> [f64; 4] {
        let x = (u * ts - 0.5).clamp(0.0, ts - 1.0);
        let y = (v * ts - 0.5).clamp(0.0, ts - 1.0);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(tool_size - 1), (y0 + 1).min(tool_size - 1));
        let (fx, fy) = (x - x0 as f64, y - y0 as f64);
        let px = |xx: usize, yy: usize, c: usize| tool[(yy * tool_size + xx) * 4 + c] as f64;
        std::array::from_fn(|c| (px(x0, y0, c) * (1.0 - fx) + px(x1, y0, c) * fx) * (1.0 - fy) + (px(x0, y1, c) * (1.0 - fx) + px(x1, y1, c) * fx) * fy)
    };
    let mut r = LayoutOut::default();
    for y in 0..size {
        for x in 0..size {
            let k = y * size + x;
            let (d, _, i, seen) = best[k];
            if i == usize::MAX {
                continue;
            }
            if occ[k] {
                if d == 0.0 {
                    r.shared_other += 1;
                }
                continue;
            }
            let p: [[f64; 2]; 3] = std::array::from_fn(|c| [stock_uv[i][c][0] * sz, stock_uv[i][c][1] * sz]);
            let Some(bc) = bary2(p, [x as f64 + 0.5, y as f64 + 0.5]) else { continue };
            // The primary's tool UVs, in this triangle's corner order.
            let pi = hs.primary[i];
            let tuv: [[f64; 2]; 3] = std::array::from_fn(|c| {
                let kk = hs.wt[pi].iter().position(|&w| w == hs.wt[i][c]).unwrap_or(c);
                tool_uv[pi][kk]
            });
            let u = (0..3).map(|c| bc[c] * tuv[c][0]).sum::<f64>();
            let v = (0..3).map(|c| bc[c] * tuv[c][1]).sum::<f64>();
            let mut s = sample(u, v);
            if d == 0.0 && !partners[pi].is_empty() {
                let cr = hs.corners(i);
                let q: V3 = std::array::from_fn(|k| (0..3).map(|c| bc[c] * cr[c][k]).sum::<f64>());
                let (mut acc, mut wsum) = ([0.0f64; 3], 0.0f64);
                for &j in &partners[pi] {
                    let cj = hs.corners(j);
                    let nj = hs.normal(j);
                    let dist = dot(sub(q, cj[0]), nj);
                    if dist.abs() >= cross_r {
                        continue;
                    }
                    let Some(bj) = bary3(cj, sub(q, mul(nj, dist))) else { continue };
                    if bj.iter().any(|&b| b < -0.02) {
                        continue;
                    }
                    let t = 1.0 - dist.abs() / cross_r;
                    // At this card's own outline take the other card fully, at the other card's
                    // outline not at all, half in between: both cards then agree over the overlap
                    // and neither outline shows.
                    let qj = sub(q, mul(nj, dist));
                    let (ei, ej) = (inside_card(pi, q), inside_card(j, qj));
                    let share = if ei + ej > 0.0 { ej / (ei + ej) } else { 0.5 };
                    let w = share * t * t * (3.0 - 2.0 * t);
                    let sj = sample((0..3).map(|c| bj[c] * tool_uv[j][c][0]).sum::<f64>(), (0..3).map(|c| bj[c] * tool_uv[j][c][1]).sum::<f64>());
                    if sj[3] < 128.0 {
                        continue;
                    }
                    for c in 0..3 {
                        acc[c] += w * sj[c];
                    }
                    wsum += w;
                }
                if wsum > 0.0 {
                    // Several crossings at one point: the others' paint averaged, at most all of it.
                    let take = wsum.min(1.0);
                    for c in 0..3 {
                        s[c] = s[c] * (1.0 - take) + acc[c] / wsum * take;
                    }
                    r.crossed += 1;
                }
            }
            for c in 0..4 {
                out[k * 4 + c] = s[c].round().clamp(0.0, 255.0) as u8;
            }
            if d == 0.0 {
                r.inside += 1;
                if seen {
                    r.mirrored += 1;
                }
            } else {
                r.gutter += 1;
            }
        }
    }
    if seam_w > 0.0 {
        seam_blend(hs, stock_uv, &best.iter().map(|b| b.2 != usize::MAX).collect::<Vec<_>>(), occ, out, size, seam_w, &mut r);
    }
    r
}

/// Blends across every edge that two hair triangles share in the stock UV layout, over `w` texels
/// either side: a texel takes part of the texel mirrored across the edge (half at the edge, none
/// at `w`), so the paint runs on across the edge as the stock paint does. The bake paints each
/// triangle from its own surface point, so where the layout joins two triangles that the model
/// folds apart (a card folded back over itself), the two sides can differ and the texture filter
/// then draws a line along the fold.
#[allow(clippy::too_many_arguments)]
fn seam_blend(hs: &HairSet, stock_uv: &[[[f64; 2]; 3]], hair: &[bool], occ: &[bool], out: &mut [u8], size: usize, w: f64, r: &mut LayoutOut) {
    let sz = size as f64;
    let q = |p: [f64; 2]| ((p[0] * sz * 4.0).round() as i64, (p[1] * sz * 4.0).round() as i64);
    // UV edge -> (triangle, the corner not on the edge).
    let mut edges: HashMap<((i64, i64), (i64, i64)), Vec<(usize, usize)>> = HashMap::new();
    for i in (0..hs.tris.len()).filter(|&i| hs.is_primary(i)) {
        for c in 0..3 {
            let (a, b) = (q(stock_uv[i][c]), q(stock_uv[i][(c + 1) % 3]));
            if a != b {
                edges.entry((a.min(b), a.max(b))).or_default().push((i, (c + 2) % 3));
            }
        }
    }
    let src = out.to_vec();
    let px = |x: f64, y: f64, c: usize| -> f64 {
        let x = (x - 0.5).clamp(0.0, sz - 1.0);
        let y = (y - 0.5).clamp(0.0, sz - 1.0);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(size - 1), (y0 + 1).min(size - 1));
        let (fx, fy) = (x - x0 as f64, y - y0 as f64);
        let g = |xx: usize, yy: usize| src[(yy * size + xx) * 4 + c] as f64;
        (g(x0, y0) * (1.0 - fx) + g(x1, y0) * fx) * (1.0 - fy) + (g(x0, y1) * (1.0 - fx) + g(x1, y1) * fx) * fy
    };
    let mut done = vec![false; size * size];
    // Sorted, so the first edge to claim a texel is the same every run.
    let mut edges: Vec<_> = edges.into_iter().collect();
    edges.sort_unstable();
    for &((a, b), ref tris) in &edges {
        // Two triangles on opposite sides of the edge (mirror copies and stacked charts share all
        // three corners or lie on one side; they are skipped).
        let p0 = [a.0 as f64 / 4.0, a.1 as f64 / 4.0];
        let p1 = [b.0 as f64 / 4.0, b.1 as f64 / 4.0];
        let d = [p1[0] - p0[0], p1[1] - p0[1]];
        let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
        let n = [-d[1] / l, d[0] / l];
        let side = |&(t, c): &(usize, usize)| {
            let o = [stock_uv[t][c][0] * sz - p0[0], stock_uv[t][c][1] * sz - p0[1]];
            o[0] * n[0] + o[1] * n[1]
        };
        let pos = tris.iter().any(|t| side(t) > 0.5);
        let neg = tris.iter().any(|t| side(t) < -0.5);
        if !(pos && neg) {
            continue;
        }
        r.seams += 1;
        let x0 = (p0[0].min(p1[0]) - w).floor().max(0.0) as usize;
        let x1 = ((p0[0].max(p1[0]) + w).ceil() as usize).min(size - 1);
        let y0 = (p0[1].min(p1[1]) - w).floor().max(0.0) as usize;
        let y1 = ((p0[1].max(p1[1]) + w).ceil() as usize).min(size - 1);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let k = y * size + x;
                if !hair[k] || occ[k] || done[k] {
                    continue;
                }
                let o = [x as f64 + 0.5 - p0[0], y as f64 + 0.5 - p0[1]];
                let t = (o[0] * d[0] + o[1] * d[1]) / (l * l);
                let s = o[0] * n[0] + o[1] * n[1];
                if !(0.0..=1.0).contains(&t) || s.abs() >= w {
                    continue;
                }
                let (mx, my) = (x as f64 + 0.5 - 2.0 * s * n[0], y as f64 + 0.5 - 2.0 * s * n[1]);
                let mk = (my.floor().clamp(0.0, sz - 1.0) as usize) * size + mx.floor().clamp(0.0, sz - 1.0) as usize;
                if !hair[mk] || occ[mk] {
                    continue;
                }
                let f = 0.5 * (1.0 - s.abs() / w);
                for c in 0..3 {
                    out[k * 4 + c] = (src[k * 4 + c] as f64 * (1.0 - f) + px(mx, my, c) * f).round().clamp(0.0, 255.0) as u8;
                }
                done[k] = true;
                r.seam_texels += 1;
            }
        }
    }
}
