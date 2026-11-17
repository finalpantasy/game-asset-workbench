//! Reference-free strand checks for hair textures.
//!
//! A repaint is meant to differ from stock, so "how far from stock" cannot find a strand break.
//! This module looks only at the candidate: it measures the direction the painted strands run
//! (the structure tensor of the texture, per chart), maps that direction onto the 3D surface, and
//! scores
//!   * seam continuity: across every UV seam, whether a strand that leaves one chart continues in
//!     the same direction in the next one (angle between the two strand axes after unfolding the
//!     two triangles into one plane), against the same measure on UV-continuous edges (`inner`),
//!     which says how much the texture turns on its own;
//!   * root-to-tip agreement: whether the strands inside a lock run along the lock's long axis
//!     (or along an axis the caller supplies).
//!
//! Small API: build `Piece`s (a mesh, a per-triangle hair mask, its base color texture) and call
//! `check` (default axes) or `check_with` (the caller's root-to-tip axis per triangle). All angles
//! are between undirected axes, 0 to 90 degrees. Weights are the structure coherence at the
//! sample, so flat or noisy areas count for little. Read-only; nothing here touches a file.
use crate::model::Mesh;
use std::collections::HashMap;

type V3 = [f64; 3];
type Key = (i64, i64, i64);

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn mul(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn len(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: V3) -> V3 {
    let l = len(a);
    if l < 1e-300 {
        [0.0; 3]
    } else {
        mul(a, 1.0 / l)
    }
}

/// Eigenvalues and eigenvectors (column i of the matrix is the vector of value i) of a symmetric
/// 3x3 matrix, by Jacobi rotations.
pub fn eigen_sym3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..60 {
        let (mut p, mut q, mut m) = (0usize, 1usize, a[0][1].abs());
        if a[0][2].abs() > m {
            (p, q, m) = (0, 2, a[0][2].abs());
        }
        if a[1][2].abs() > m {
            (p, q, m) = (1, 2, a[1][2].abs());
        }
        if m < 1e-18 {
            break;
        }
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        for k in 0..3 {
            let (x, y) = (a[k][p], a[k][q]);
            a[k][p] = c * x - s * y;
            a[k][q] = s * x + c * y;
        }
        for k in 0..3 {
            let (x, y) = (a[p][k], a[q][k]);
            a[p][k] = c * x - s * y;
            a[q][k] = s * x + c * y;
        }
        for k in 0..3 {
            let (x, y) = (v[k][p], v[k][q]);
            v[k][p] = c * x - s * y;
            v[k][q] = s * x + c * y;
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

/// One mesh of the model with the hair triangles to check and the texture they are painted on.
pub struct Piece<'a> {
    pub mesh: &'a Mesh,
    /// One flag per triangle of `mesh`: true for hair.
    pub tris: &'a [bool],
    /// Level 0 of the base color texture, RGBA8.
    pub rgba: &'a [u8],
    pub w: usize,
    pub h: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Opts {
    /// Sample points along each shared edge.
    pub per_edge: usize,
    /// Samples with less structure coherence than this are skipped (0 to 1).
    pub min_coherence: f64,
    /// An angle above this counts as bad (degrees).
    pub bad_deg: f64,
    /// Double-sided cards (a front triangle and a reversed back copy on the same three corners):
    /// never pair a triangle with its own back copy (the unfold mirrors the strand axis and reads
    /// as twice the real angle), and on an edge shared by two double-sided cards pair front with
    /// front and back with back. Off by default so older numbers stay comparable.
    pub twin_aware: bool,
}

impl Default for Opts {
    fn default() -> Opts {
        Opts { per_edge: 6, min_coherence: 0.25, bad_deg: 30.0, twin_aware: false }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The two triangles meet in 3D but their UVs do not (a chart border).
    Seam,
    /// The two triangles share UV along the edge (inside a chart): the baseline.
    Inner,
}

/// Summary of angles (degrees) weighted by coherence.
#[derive(Clone, Copy, Debug, Default)]
pub struct Agg {
    pub n: usize,
    pub weight: f64,
    pub mean_deg: f64,
    pub p50_deg: f64,
    pub p90_deg: f64,
    /// Weighted share of samples above `Opts::bad_deg`.
    pub bad_frac: f64,
}

/// An edge between two hair triangles with its mean angle.
#[derive(Clone, Debug)]
pub struct EdgeScore {
    pub kind: Kind,
    /// (piece, triangle) of both sides.
    pub a: (usize, usize),
    pub b: (usize, usize),
    /// UV (0..1) of the two edge ends on each side.
    pub uv_a: [[f32; 2]; 2],
    pub uv_b: [[f32; 2]; 2],
    pub mean_deg: f32,
    pub weight: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub triangles: usize,
    pub charts: usize,
    pub seam_edges: usize,
    pub inner_edges: usize,
    pub seam: Agg,
    pub inner: Agg,
    pub root_tip: Agg,
    pub edges: Vec<EdgeScore>,
}

impl Report {
    /// Seam angle minus the angle the texture turns by on its own, degrees.
    pub fn seam_excess_deg(&self) -> f64 {
        self.seam.mean_deg - self.inner.mean_deg
    }
}

struct T {
    piece: usize,
    ti: usize,
    p: [V3; 3],
    uv: [[f64; 2]; 3],
    key: [Key; 3],
    chart: usize,
}

struct Sample {
    theta: f64,
    coh: f64,
}

struct Ctx<'a> {
    piece: &'a Piece<'a>,
    chart: Vec<u32>,
    gk: Vec<f64>,
    dk: Vec<f64>,
    rg: i32,
    ri: i32,
    si: f64,
    win_w: f64,
}

impl<'a> Ctx<'a> {
    fn new(piece: &'a Piece<'a>, chart: Vec<u32>) -> Ctx<'a> {
        let sg = (piece.w as f64 / 3000.0).max(0.8);
        let si = (piece.w as f64 / 1200.0).max(1.5);
        let rg = (2.5 * sg).ceil() as i32;
        let ri = (2.5 * si).ceil() as i32;
        let g: Vec<f64> = (-rg..=rg).map(|i| (-((i * i) as f64) / (2.0 * sg * sg)).exp()).collect();
        let gs: f64 = g.iter().sum();
        let gk: Vec<f64> = g.iter().map(|v| v / gs).collect();
        let dk: Vec<f64> = (-rg..=rg).zip(gk.iter()).map(|(i, g)| -(i as f64) / (sg * sg) * g).collect();
        let mut win_w = 0.0;
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                win_w += (-((dx * dx + dy * dy) as f64) / (2.0 * si * si)).exp();
            }
        }
        Ctx { piece, chart, gk, dk, rg, ri, si, win_w }
    }

    fn luma(&self, x: i32, y: i32) -> f64 {
        let (w, h) = (self.piece.w as i32, self.piece.h as i32);
        let o = ((y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) * 4) as usize;
        let p = self.piece.rgba;
        0.299 * p[o] as f64 + 0.587 * p[o + 1] as f64 + 0.114 * p[o + 2] as f64
    }

    /// The pixel belongs to chart `c`, is opaque, and so are its axial neighbors at the gradient radius.
    fn valid(&self, x: i32, y: i32, c: u32) -> bool {
        let (w, h) = (self.piece.w as i32, self.piece.h as i32);
        let r = self.rg;
        for (dx, dy) in [(0, 0), (r, 0), (-r, 0), (0, r), (0, -r)] {
            let (xx, yy) = (x + dx, y + dy);
            if xx < 0 || yy < 0 || xx >= w || yy >= h {
                return false;
            }
            let i = (yy * w + xx) as usize;
            if self.chart[i] != c || self.piece.rgba[i * 4 + 3] < 128 {
                return false;
            }
        }
        true
    }

    fn grad(&self, x: i32, y: i32) -> (f64, f64) {
        let r = self.rg;
        let (mut gx, mut gy) = (0.0, 0.0);
        for j in -r..=r {
            let (gj, dj) = (self.gk[(j + r) as usize], self.dk[(j + r) as usize]);
            for i in -r..=r {
                let l = self.luma(x + i, y + j);
                gx += self.dk[(i + r) as usize] * gj * l;
                gy += self.gk[(i + r) as usize] * dj * l;
            }
        }
        (gx, gy)
    }

    /// Strand axis (texel space, radians) and coherence around a texel position inside chart `c`.
    fn axis(&self, cx: f64, cy: f64, c: u32, min_coh: f64) -> Option<Sample> {
        let (x0, y0) = (cx.round() as i32, cy.round() as i32);
        let (mut jxx, mut jxy, mut jyy, mut vw) = (0.0, 0.0, 0.0, 0.0);
        for dy in -self.ri..=self.ri {
            for dx in -self.ri..=self.ri {
                let (x, y) = (x0 + dx, y0 + dy);
                if !self.valid(x, y, c) {
                    continue;
                }
                let w = (-((dx * dx + dy * dy) as f64) / (2.0 * self.si * self.si)).exp();
                let (gx, gy) = self.grad(x, y);
                jxx += w * gx * gx;
                jxy += w * gx * gy;
                jyy += w * gy * gy;
                vw += w;
            }
        }
        if vw < 0.3 * self.win_w {
            return None;
        }
        let tr = jxx + jyy;
        if tr / vw < 1.0 {
            return None; // flatter than one luma step per texel: no structure to read
        }
        let coh = ((jxx - jyy).powi(2) + 4.0 * jxy * jxy).sqrt() / tr;
        if coh < min_coh {
            return None;
        }
        let grad_dir = 0.5 * (2.0 * jxy).atan2(jxx - jyy);
        Some(Sample { theta: grad_dir + std::f64::consts::FRAC_PI_2, coh })
    }
}

fn find(p: &mut [usize], mut x: usize) -> usize {
    while p[x] != x {
        p[x] = p[p[x]];
        x = p[x];
    }
    x
}

/// Surface vectors for one unit step of u and of v: the map from UV to the triangle plane.
fn uv_basis(p: [V3; 3], uv: [[f64; 2]; 3]) -> Option<(V3, V3)> {
    let (e1, e2) = (sub(p[1], p[0]), sub(p[2], p[0]));
    let (du1, dv1, du2, dv2) = (uv[1][0] - uv[0][0], uv[1][1] - uv[0][1], uv[2][0] - uv[0][0], uv[2][1] - uv[0][1]);
    let det = du1 * dv2 - du2 * dv1;
    if det.abs() < 1e-14 {
        return None;
    }
    let pu = mul(sub(mul(e1, dv2), mul(e2, dv1)), 1.0 / det);
    let pv = mul(sub(mul(e2, du1), mul(e1, du2)), 1.0 / det);
    Some((pu, pv))
}

fn agg(list: &mut [(f64, f64)], bad: f64) -> Agg {
    if list.is_empty() {
        return Agg::default();
    }
    let sw: f64 = list.iter().map(|x| x.1).sum();
    if sw <= 0.0 {
        return Agg::default();
    }
    let mean = list.iter().map(|x| x.0 * x.1).sum::<f64>() / sw;
    let bad_frac = list.iter().filter(|x| x.0 > bad).map(|x| x.1).sum::<f64>() / sw;
    list.sort_by(|a, b| a.0.total_cmp(&b.0));
    let q = |f: f64| -> f64 {
        let mut acc = 0.0;
        for x in list.iter() {
            acc += x.1;
            if acc >= f * sw {
                return x.0;
            }
        }
        list.last().unwrap().0
    };
    Agg { n: list.len(), weight: sw, mean_deg: mean, p50_deg: q(0.5), p90_deg: q(0.9), bad_frac }
}

fn axis_angle_deg(a: V3, b: V3) -> f64 {
    let (la, lb) = (len(a), len(b));
    if la < 1e-12 || lb < 1e-12 {
        return 0.0;
    }
    (dot(a, b).abs() / (la * lb)).min(1.0).acos().to_degrees()
}

/// Checks with the default root-to-tip axis (the long axis of each lock).
pub fn check(pieces: &[Piece], o: &Opts) -> Report {
    check_with(pieces, o, &|_, _| None)
}

/// `expected(piece, triangle)` may return the root-to-tip axis for a triangle (any length, either
/// sign); where it returns None the long axis of the triangle's lock is used.
pub fn check_with(pieces: &[Piece], o: &Opts, expected: &dyn Fn(usize, usize) -> Option<[f64; 3]>) -> Report {
    // Flatten the hair triangles.
    let mut tris: Vec<T> = Vec::new();
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for (pi, pc) in pieces.iter().enumerate() {
        for (ti, t) in pc.mesh.tris.iter().enumerate() {
            if !pc.tris.get(ti).copied().unwrap_or(false) {
                continue;
            }
            let p: [V3; 3] = std::array::from_fn(|k| {
                let v = pc.mesh.pos[t[k] as usize];
                [v[0] as f64, v[1] as f64, v[2] as f64]
            });
            for q in &p {
                for k in 0..3 {
                    lo[k] = lo[k].min(q[k]);
                    hi[k] = hi[k].max(q[k]);
                }
            }
            let uv: [[f64; 2]; 3] = std::array::from_fn(|k| {
                let u = pc.mesh.uv[t[k] as usize];
                [u[0] as f64, u[1] as f64]
            });
            tris.push(T { piece: pi, ti, p, uv, key: [(0, 0, 0); 3], chart: 0 });
        }
    }
    if tris.is_empty() {
        return Report::default();
    }
    let diag = len(sub(hi, lo)).max(1e-9);
    let quantum = diag * 1e-5;
    for t in tris.iter_mut() {
        t.key = std::array::from_fn(|k| ((t.p[k][0] / quantum).round() as i64, (t.p[k][1] / quantum).round() as i64, (t.p[k][2] / quantum).round() as i64));
    }
    // Edges shared by two triangles in 3D.
    let mut edges: HashMap<(Key, Key), Vec<(usize, usize)>> = HashMap::new();
    for (i, t) in tris.iter().enumerate() {
        for e in 0..3 {
            let (a, b) = (t.key[e], t.key[(e + 1) % 3]);
            if a == b {
                continue;
            }
            edges.entry(if a < b { (a, b) } else { (b, a) }).or_default().push((i, e));
        }
    }
    struct Pair {
        a: usize,
        ai: usize,
        b: usize,
        seam: bool,
    }
    let mut pairs: Vec<Pair> = Vec::new();
    let mut keys: Vec<&(Key, Key)> = edges.keys().collect();
    keys.sort();
    for k in keys {
        let v = &edges[k];
        let twin = |x: usize, y: usize| {
            let (mut kx, mut ky) = (tris[x].key, tris[y].key);
            kx.sort();
            ky.sort();
            kx == ky
        };
        // The pairs of (triangle, edge slot) to compare across this edge.
        let mut todo: Vec<((usize, usize), (usize, usize))> = Vec::new();
        if v.len() == 2 {
            if v[0].0 != v[1].0 && !(o.twin_aware && twin(v[0].0, v[1].0)) {
                todo.push((v[0], v[1]));
            }
        } else if o.twin_aware && v.len() == 4 {
            // Two double-sided cards on one edge: walk the edge in opposite directions on the
            // neighbours that face the same way, so match each triangle with such a one.
            let fwd = |&(t, e): &(usize, usize)| tris[t].key[e] == k.0;
            let mut used = [false; 4];
            for i in 0..4 {
                if used[i] {
                    continue;
                }
                if let Some(j) = (i + 1..4).find(|&j| !used[j] && v[j].0 != v[i].0 && !twin(v[i].0, v[j].0) && fwd(&v[i]) != fwd(&v[j])) {
                    used[i] = true;
                    used[j] = true;
                    todo.push((v[i], v[j]));
                }
            }
        }
        for ((a, ai), (b, _)) in todo {
        let (ta, tb) = (&tris[a], &tris[b]);
        let same_tex = std::ptr::eq(pieces[ta.piece].rgba.as_ptr(), pieces[tb.piece].rgba.as_ptr());
        let (w, h) = (pieces[ta.piece].w as f64, pieces[ta.piece].h as f64);
        let mut seam = !same_tex;
        if same_tex {
            for kk in [k.0, k.1] {
                let ia = (0..3).find(|&i| ta.key[i] == kk);
                let ib = (0..3).find(|&i| tb.key[i] == kk);
                let (Some(ia), Some(ib)) = (ia, ib) else {
                    seam = true;
                    continue;
                };
                let d = ((ta.uv[ia][0] - tb.uv[ib][0]) * w).hypot((ta.uv[ia][1] - tb.uv[ib][1]) * h);
                if d > 0.75 {
                    seam = true;
                }
            }
        }
        pairs.push(Pair { a, ai, b, seam });
        }
    }
    // Charts: triangles joined by UV-continuous edges.
    let mut uf: Vec<usize> = (0..tris.len()).collect();
    for p in pairs.iter().filter(|p| !p.seam) {
        let (ra, rb) = (find(&mut uf, p.a), find(&mut uf, p.b));
        if ra != rb {
            uf[ra] = rb;
        }
    }
    let mut chart_of_root: HashMap<usize, usize> = HashMap::new();
    for i in 0..tris.len() {
        let r = find(&mut uf, i);
        let n = chart_of_root.len();
        tris[i].chart = *chart_of_root.entry(r).or_insert(n);
    }
    let charts = chart_of_root.len();
    // Chart id per texel, per piece.
    let mut maps: Vec<Vec<u32>> = pieces.iter().map(|p| vec![0u32; p.w * p.h]).collect();
    for t in &tris {
        let pc = &pieces[t.piece];
        let tx: [[f64; 2]; 3] = std::array::from_fn(|k| [t.uv[k][0] * pc.w as f64, t.uv[k][1] * pc.h as f64]);
        let (x0, x1) = (tx.iter().map(|q| q[0]).fold(f64::MAX, f64::min), tx.iter().map(|q| q[0]).fold(f64::MIN, f64::max));
        let (y0, y1) = (tx.iter().map(|q| q[1]).fold(f64::MAX, f64::min), tx.iter().map(|q| q[1]).fold(f64::MIN, f64::max));
        let det = (tx[1][1] - tx[2][1]) * (tx[0][0] - tx[2][0]) + (tx[2][0] - tx[1][0]) * (tx[0][1] - tx[2][1]);
        if det.abs() < 1e-12 {
            continue;
        }
        for y in (y0.floor().max(0.0) as usize)..=((y1.ceil() as usize).min(pc.h - 1)) {
            for x in (x0.floor().max(0.0) as usize)..=((x1.ceil() as usize).min(pc.w - 1)) {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let l0 = ((tx[1][1] - tx[2][1]) * (px - tx[2][0]) + (tx[2][0] - tx[1][0]) * (py - tx[2][1])) / det;
                let l1 = ((tx[2][1] - tx[0][1]) * (px - tx[2][0]) + (tx[0][0] - tx[2][0]) * (py - tx[2][1])) / det;
                let l2 = 1.0 - l0 - l1;
                if l0 >= -1e-6 && l1 >= -1e-6 && l2 >= -1e-6 {
                    maps[t.piece][y * pc.w + x] = t.chart as u32 + 1;
                }
            }
        }
    }
    let ctxs: Vec<Ctx> = pieces.iter().zip(maps).map(|(p, m)| Ctx::new(p, m)).collect();

    // Strand axis at a texel position of triangle `t`, as a 3D vector in the triangle's plane.
    let strand3 = |t: &T, tex: [f64; 2]| -> Option<(V3, f64)> {
        let ctx = &ctxs[t.piece];
        let s = ctx.axis(tex[0], tex[1], t.chart as u32 + 1, o.min_coherence)?;
        let pc = ctx.piece;
        let (pu, pv) = uv_basis(t.p, t.uv)?;
        let (a, b) = (s.theta.cos() / pc.w as f64, s.theta.sin() / pc.h as f64);
        let v = add(mul(pu, a), mul(pv, b));
        (len(v) > 1e-12).then_some((v, s.coh))
    };

    let mut seam_list: Vec<(f64, f64)> = Vec::new();
    let mut inner_list: Vec<(f64, f64)> = Vec::new();
    let mut scores: Vec<EdgeScore> = Vec::new();
    for pr in &pairs {
        let (ta, tb) = (&tris[pr.a], &tris[pr.b]);
        let (ia0, ia1) = (pr.ai, (pr.ai + 1) % 3);
        let (Some(ib0), Some(ib1)) = ((0..3).find(|&i| tb.key[i] == ta.key[ia0]), (0..3).find(|&i| tb.key[i] == ta.key[ia1])) else { continue };
        let edge = sub(ta.p[ia1], ta.p[ia0]);
        if len(edge) < 1e-12 {
            continue;
        }
        let e_hat = unit(edge);
        let perp = |t: &T, i0: usize, i1: usize| -> V3 {
            let io = 3 - i0 - i1;
            let w = sub(t.p[io], t.p[i0]);
            unit(sub(w, mul(e_hat, dot(w, e_hat))))
        };
        let (pa, pb) = (perp(ta, ia0, ia1), perp(tb, ib0, ib1));
        let mut sum = (0.0f64, 0.0f64);
        for k in 1..=o.per_edge {
            let f = k as f64 / (o.per_edge + 1) as f64;
            let mut side: Vec<(V3, f64)> = Vec::new();
            for (t, i0, i1) in [(ta, ia0, ia1), (tb, ib0, ib1)] {
                let pc = &pieces[t.piece];
                let tex = |i: usize| [t.uv[i][0] * pc.w as f64, t.uv[i][1] * pc.h as f64];
                let (t0, t1, to) = (tex(i0), tex(i1), tex(3 - i0 - i1));
                let q = [t0[0] + (t1[0] - t0[0]) * f, t0[1] + (t1[1] - t0[1]) * f];
                let e = [t1[0] - t0[0], t1[1] - t0[1]];
                let el = e[0].hypot(e[1]);
                if el < 1e-9 {
                    break;
                }
                let mut n = [-e[1] / el, e[0] / el];
                let alt = n[0] * (to[0] - t0[0]) + n[1] * (to[1] - t0[1]);
                if alt < 0.0 {
                    n = [-n[0], -n[1]];
                }
                let alt = alt.abs();
                let d = (1.5 * ctxs[t.piece].si).min(0.5 * alt);
                if d < 2.0 {
                    break;
                }
                let pt = [q[0] + n[0] * d, q[1] + n[1] * d];
                // Inside the triangle?
                let det = (t1[1] - to[1]) * (t0[0] - to[0]) + (to[0] - t1[0]) * (t0[1] - to[1]);
                if det.abs() < 1e-12 {
                    break;
                }
                let l0 = ((t1[1] - to[1]) * (pt[0] - to[0]) + (to[0] - t1[0]) * (pt[1] - to[1])) / det;
                let l1 = ((to[1] - t0[1]) * (pt[0] - to[0]) + (t0[0] - to[0]) * (pt[1] - to[1])) / det;
                if l0 < -0.02 || l1 < -0.02 || 1.0 - l0 - l1 < -0.02 {
                    break;
                }
                match strand3(t, pt) {
                    Some(s) => side.push(s),
                    None => break,
                }
            }
            if side.len() != 2 {
                continue;
            }
            // Unfold both triangles into one plane: x along the edge, y across it from A to B.
            let ((va, ca), (vb, cb)) = (side[0], side[1]);
            let (a_x, a_y) = (dot(va, e_hat), -dot(va, pa));
            let (b_x, b_y) = (dot(vb, e_hat), dot(vb, pb));
            let la = a_x.hypot(a_y);
            let lb = b_x.hypot(b_y);
            if la < 1e-12 || lb < 1e-12 {
                continue;
            }
            let ang = ((a_x * b_x + a_y * b_y).abs() / (la * lb)).min(1.0).acos().to_degrees();
            let w = ca.min(cb);
            (if pr.seam { &mut seam_list } else { &mut inner_list }).push((ang, w));
            sum.0 += ang * w;
            sum.1 += w;
        }
        if sum.1 > 0.0 {
            let uv = |t: &T, i0: usize, i1: usize| [[t.uv[i0][0] as f32, t.uv[i0][1] as f32], [t.uv[i1][0] as f32, t.uv[i1][1] as f32]];
            scores.push(EdgeScore {
                kind: if pr.seam { Kind::Seam } else { Kind::Inner },
                a: (ta.piece, ta.ti),
                b: (tb.piece, tb.ti),
                uv_a: uv(ta, ia0, ia1),
                uv_b: uv(tb, ib0, ib1),
                mean_deg: (sum.0 / sum.1) as f32,
                weight: sum.1 as f32,
            });
        }
    }

    // Root to tip: the strand direction against the long axis of the lock the triangle belongs to.
    // A lock is one UV chart (a strip card): its long axis is the root-to-tip direction. Charts are
    // used instead of position-welded pieces because welded hair is one blob with no long axis.
    let mut lock_uf: Vec<usize> = uf.clone();
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..tris.len() {
        let r = find(&mut lock_uf, i);
        members.entry(r).or_default().push(i);
    }
    let mut lock_axis: HashMap<usize, V3> = HashMap::new();
    for (r, m) in &members {
        if m.len() < 6 {
            continue;
        }
        let mut pts: Vec<V3> = Vec::new();
        let mut seen: std::collections::HashSet<Key> = std::collections::HashSet::new();
        for &i in m {
            for k in 0..3 {
                if seen.insert(tris[i].key[k]) {
                    pts.push(tris[i].p[k]);
                }
            }
        }
        let c = mul(pts.iter().fold([0.0; 3], |a, &p| add(a, p)), 1.0 / pts.len() as f64);
        let mut cov = [[0.0f64; 3]; 3];
        for p in &pts {
            let d = sub(*p, c);
            for i in 0..3 {
                for j in 0..3 {
                    cov[i][j] += d[i] * d[j];
                }
            }
        }
        let (val, vec) = eigen_sym3(cov);
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| val[b].total_cmp(&val[a]));
        if val[order[0]] > 2.5 * val[order[1]].max(1e-18) {
            lock_axis.insert(*r, [vec[0][order[0]], vec[1][order[0]], vec[2][order[0]]]);
        }
    }
    let mut rt_list: Vec<(f64, f64)> = Vec::new();
    for i in 0..tris.len() {
        let t = &tris[i];
        let root = find(&mut lock_uf, i);
        let Some(axis) = expected(t.piece, t.ti).or_else(|| lock_axis.get(&root).copied()) else { continue };
        let n = unit(cross(sub(t.p[1], t.p[0]), sub(t.p[2], t.p[0])));
        let ax = unit(axis);
        let in_plane = sub(ax, mul(n, dot(ax, n)));
        if len(in_plane) < 0.35 {
            continue;
        }
        let pc = &pieces[t.piece];
        let c = [(t.uv[0][0] + t.uv[1][0] + t.uv[2][0]) / 3.0 * pc.w as f64, (t.uv[0][1] + t.uv[1][1] + t.uv[2][1]) / 3.0 * pc.h as f64];
        let Some((v, coh)) = strand3(t, c) else { continue };
        rt_list.push((axis_angle_deg(v, in_plane), coh));
    }

    Report {
        triangles: tris.len(),
        charts,
        seam_edges: pairs.iter().filter(|p| p.seam).count(),
        inner_edges: pairs.iter().filter(|p| !p.seam).count(),
        seam: agg(&mut seam_list, o.bad_deg),
        inner: agg(&mut inner_list, o.bad_deg),
        root_tip: agg(&mut rt_list, o.bad_deg),
        edges: scores,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two unit quads side by side in the z = 0 plane, each with its own UV chart, so the shared
    /// edge is a seam. Left texture region is striped by `left`, right by `right`.
    fn two_charts(left: &dyn Fn(f64, f64) -> f64, right: &dyn Fn(f64, f64) -> f64) -> (Mesh, Vec<u8>, usize, usize) {
        let (w, h) = (256usize, 128usize);
        let mut rgba = vec![255u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let l = if x < w / 2 { left(x as f64, y as f64) } else { right(x as f64, y as f64) };
                let v = l.clamp(0.0, 255.0) as u8;
                let o = (y * w + x) * 4;
                rgba[o] = v;
                rgba[o + 1] = v;
                rgba[o + 2] = v;
            }
        }
        // 3D: left quad x in 0..1, right quad x in 1..2, y in 0..1. UV: left u 0.05..0.45, right u 0.55..0.95; v 0.1..0.9, y up = v down.
        let mut pos = Vec::new();
        let mut uv = Vec::new();
        for (x0, u0) in [(0.0f32, 0.05f32), (1.0, 0.55)] {
            for (dx, du) in [(0.0f32, 0.0f32), (1.0, 0.4)] {
                for (y, v) in [(0.0f32, 0.9f32), (1.0, 0.1)] {
                    pos.push([x0 + dx, y, 0.0]);
                    uv.push([u0 + du, v]);
                }
            }
        }
        // vertices per quad: (x0,y0) (x0,y1) (x1,y0) (x1,y1) -> 0 1 2 3
        let tris = vec![[0, 2, 1], [1, 2, 3], [4, 6, 5], [5, 6, 7]];
        let n = pos.len();
        let mesh = Mesh { name: "t".into(), pos, uv, joints: vec![[0; 4]; n], weights: vec![[1.0, 0.0, 0.0, 0.0]; n], tris, image: None };
        (mesh, rgba, w, h)
    }

    fn horizontal(_x: f64, y: f64) -> f64 {
        128.0 + 100.0 * (std::f64::consts::TAU * y / 7.0).sin()
    }
    fn vertical(x: f64, _y: f64) -> f64 {
        128.0 + 100.0 * (std::f64::consts::TAU * x / 7.0).sin()
    }

    #[test]
    fn matching_stripes_cross_the_seam_cleanly() {
        let (mesh, rgba, w, h) = two_charts(&horizontal, &horizontal);
        let mask = vec![true; mesh.tris.len()];
        let r = check(&[Piece { mesh: &mesh, tris: &mask, rgba: &rgba, w, h }], &Opts::default());
        assert!(r.seam_edges >= 1, "no seam found: {r:?}");
        assert!(r.seam.n > 0, "no seam samples: {r:?}");
        assert!(r.seam.mean_deg < 8.0, "{r:?}");
    }

    #[test]
    fn crossed_stripes_break_at_the_seam() {
        let (mesh, rgba, w, h) = two_charts(&horizontal, &vertical);
        let mask = vec![true; mesh.tris.len()];
        let r = check(&[Piece { mesh: &mesh, tris: &mask, rgba: &rgba, w, h }], &Opts::default());
        assert!(r.seam.n > 0, "no seam samples: {r:?}");
        assert!(r.seam.mean_deg > 70.0, "{r:?}");
        assert!(r.seam.bad_frac > 0.9);
    }

    #[test]
    fn eigen_finds_the_long_axis() {
        let (val, vec) = eigen_sym3([[4.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.5]]);
        let top = (0..3).max_by(|&a, &b| val[a].total_cmp(&val[b])).unwrap();
        let v = [vec[0][top], vec[1][top], vec[2][top]];
        assert!(v[2].abs() < 1e-9);
        assert!((val[top] - (3.0 + 2f64.sqrt())).abs() < 1e-9);
    }
}
