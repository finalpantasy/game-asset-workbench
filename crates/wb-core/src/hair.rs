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
        if u.len() == 2 {
            nb[u[0]].push((u[1], *e));
            nb[u[1]].push((u[0], *e));
        }
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

pub fn flow(model: &Model, hs: &HairSet, stock: &Sheet, o: &FlowOpts) -> Flow {
    let n = hs.tris.len();
    let mut dir = vec![[0.0; 3]; n];
    let mut away = vec![[0.0; 3]; n];
    let mut coherence = vec![0.0; n];
    let mut ambiguous = vec![false; n];
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
                    let w = o.paint_weight * ((coh - 0.08) / 0.3).clamp(0.0, 1.0);
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
    // Smooth across welded edges, never across a parting.
    let nb = neighbors(hs);
    let cos_part = o.part_angle.to_radians().cos();
    for _ in 0..o.smooth_iters {
        let prev = dir.clone();
        for i in 0..n {
            if !hs.is_primary(i) {
                continue;
            }
            let nrm = hs.normal(i);
            let mut acc = mul(prev[i], 2.0);
            for &(j, e) in &nb[i] {
                let q = hinge(hs, j, i, e, prev[j]);
                if dot(q, prev[i]) > cos_part {
                    acc = add(acc, q);
                }
            }
            let a = unit(project_plane(acc, nrm));
            if len(a) > 0.5 {
                dir[i] = a;
            }
        }
    }
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
    Flow { dir, away, coherence, ambiguous, region, region_lock, partings }
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
    let rows: Vec<Vec<(usize, f64)>> = rows.into_iter().map(|r| r.into_iter().collect()).collect();
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
            let s: f64 = tl.iter().enumerate().map(|(k, &i)| {
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
        let l = lab(rgb_);
        let c = (l[1] * l[1] + l[2] * l[2]).sqrt();
        let h = l[2].atan2(l[1]).to_degrees();
        let mut dh = (h - self.hue).abs() % 360.0;
        if dh > 180.0 {
            dh = 360.0 - dh;
        }
        let magenta = rgb_[0] > 100.0 && rgb_[2] > 90.0 && rgb_[0] > 1.35 * rgb_[1] && rgb_[2] > 1.25 * rgb_[1];
        magenta || l[0] < 12.0 || (c > 8.0 && dh > 35.0) || c < 0.3 * self.chroma
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

fn hash(a: u64) -> u64 {
    let mut x = a.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
fn h01(a: u64) -> f64 {
    (hash(a) >> 11) as f64 / (1u64 << 53) as f64
}
fn h2(seed: u64, i: i64, j: i64) -> f64 {
    h01(seed ^ (i as u64).wrapping_mul(0x51_7CC1_B727_220A) ^ (j as u64).wrapping_mul(0x2545_F491_4F6C_DD1D))
}
fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
/// 2D value noise in 0..1.
fn vnoise(seed: u64, x: f64, y: f64) -> f64 {
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
}

/// Chart texels written by `bake`: per sheet texel, the chart (usize::MAX = none) and the selected
/// triangle that owns it (inside its triangle, not a gutter texel).
pub struct BakeMap {
    pub chart: Vec<usize>,
    pub inside: Vec<bool>,
}

/// Paints every chart into `rgba` (size x size). Returns which texels belong to which chart.
pub fn bake(b: &BakeIn, rgba: &mut [u8]) -> BakeMap {
    let size = b.size;
    let mut map = BakeMap { chart: vec![usize::MAX; size * size], inside: vec![false; size * size] };
    let stock_uv = |i: usize| tri_uv(b.model, b.hs, b.hs.primary[i]);
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
                // Cutout distance (model units, + inside) from the stock alpha, then notches at tips.
                let at = |xx: usize, yy: usize| a_s[yy.min(h - 1) * w + xx.min(w - 1)];
                let gxa = 0.5 * (at(x + 1, y) - at(x.saturating_sub(1), y)) * b.scale;
                let gya = 0.5 * (at(x, y + 1) - at(x, y.saturating_sub(1))) * b.scale;
                let gl = (gxa * gxa + gya * gya).sqrt();
                let a0 = a_s[k];
                let d_s = if gl > 1e-9 { (a0 - 0.5) / gl } else if a0 >= 0.5 { 1e9 } else { -1e9 };
                let d_s = d_s.clamp(-1.0 * e, 1.0 * e);
                // Tip edges are where the alpha falls toward +v (the tip end).
                let tipward = if gl > 1e-9 { (-gya / gl).clamp(0.0, 1.0) } else { 0.0 };
                let depth = b.ctx.style.tip_depth * e * tipward;
                let d = d_s - depth * prof;
                let alpha = (0.5 + d * b.scale / 1.2).clamp(0.0, 1.0);
                // Tip lightening near tip edges.
                if tipward > 0.0 && d_s < b.ctx.style.tip_len * e {
                    col[0] += b.ctx.style.tip_light * tipward * (1.0 - smooth(d_s / (b.ctx.style.tip_len * e)));
                }
                // Ambient occlusion from the welded corners.
                let wt = b.hs.wt[t.tri];
                let bc = t.bary.map(|q| q.clamp(-0.5, 1.5));
                let ao = (bc[0] * b.ao[wt[0]] + bc[1] * b.ao[wt[1]] + bc[2] * b.ao[wt[2]]).clamp(0.0, 1.0);
                col[0] *= 1.0 - b.ctx.style.ao * (1.0 - ao);
                // Stay inside the hair's own colors: dark grooves keep hue and some chroma
                // (out-of-gamut darks clip to grey or green otherwise). Pulls toward the palette.
                col[0] = col[0].max(18.0);
                let base = b.ctx.pal.at((v / b.ctx.vmax).clamp(0.0, 1.0));
                for _ in 0..12 {
                    let r = rgb(col);
                    if r.iter().all(|&c| (0.0..=255.0).contains(&c)) && !b.ctx.pal.is_fringe(r) {
                        break;
                    }
                    col = [col[0] * 0.85 + base[0] * 0.15, col[1] * 0.8 + base[1] * 0.2, col[2] * 0.8 + base[2] * 0.2];
                }
                let px = to_rgb8(col);
                rgba[si * 4..si * 4 + 3].copy_from_slice(&px);
                rgba[si * 4 + 3] = (alpha * 255.0).round() as u8;
                map.chart[si] = ci;
                map.inside[si] = dist[k] == 0.0;
            }
        }
    }
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
            }
        }
        out.push((l, count, shown));
    }
    out
}
