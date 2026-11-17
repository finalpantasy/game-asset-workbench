//! Strands traced on the 3D hair surface.
//!
//! The chart fiber layer (`hair::Fibers`) draws strands in each chart's own (u, v), with a seed per
//! lock, so its strands restart wherever two charts, a fold row or a parting meet: a straight cut
//! across the hair. Here strands are streamlines of the flow traced over the welded surface of each
//! lock (evenly spaced, after Jobard and Lefer), and a texel's fiber value comes from its 3D distance
//! to the nearest strands. Every chart that covers a point of the surface bakes the same value, so
//! strands run on across chart seams.

use crate::hair::{add, cross, dot, ekey, h01, hash, hinge, len, mul, orient, smooth, sub, unit, vnoise, Flow, HairSet, V3};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy)]
struct At {
    tri: usize,
    p: V3,
}

#[derive(Clone, Copy)]
struct Sample {
    p: V3,
    t: V3,
    n: V3,
    line: u32,
    lock: u32,
    arc: f32,
}

/// Evenly spaced strands over the hair surface, `sep` model units apart.
pub struct StrandField {
    pub sep: f64,
    pub lines: usize,
    pub samples: usize,
    seed: u64,
    cell: f64,
    pts: Vec<Sample>,
    grid: HashMap<(i64, i64, i64), Vec<u32>>,
    sign: Vec<f64>,
}

/// Planar barycentric coordinates of `p` in triangle `c`.
fn bary3(c: [V3; 3], p: V3) -> [f64; 3] {
    let n = cross(sub(c[1], c[0]), sub(c[2], c[0]));
    let nn = dot(n, n).max(1e-30);
    let b0 = dot(n, cross(sub(c[1], p), sub(c[2], p))) / nn;
    let b1 = dot(n, cross(sub(c[2], p), sub(c[0], p))) / nn;
    [b0, b1, 1.0 - b0 - b1]
}

struct Surf<'a> {
    hs: &'a HairSet,
    fl: &'a Flow,
    sign: &'a [f64],
    across: HashMap<(usize, usize), Vec<usize>>,
    ctan: Vec<[V3; 3]>,
}

impl<'a> Surf<'a> {
    fn new(hs: &'a HairSet, fl: &'a Flow, sign: &'a [f64], smooth: usize) -> Surf<'a> {
        let mut across: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for i in 0..hs.tris.len() {
            if !hs.is_primary(i) {
                continue;
            }
            let t = hs.wt[i];
            for k in 0..3 {
                across.entry(ekey(t[k], t[(k + 1) % 3])).or_default().push(i);
            }
        }
        // Corner tangents: the flow averaged over the triangles of the same lock at a welded corner
        // (those pointing the other way, across a parting, left out). Flow regions are not kept
        // apart: a strand bends smoothly where one region meets the next instead of at the edge.
        let mut acc: HashMap<(usize, usize), Vec<(usize, V3)>> = HashMap::new();
        for i in 0..hs.tris.len() {
            if !hs.is_primary(i) {
                continue;
            }
            for &w in &hs.wt[i] {
                acc.entry((w, hs.lock[i])).or_default().push((i, mul(fl.dir[i], hs.area(i))));
            }
        }
        // The rebuilt strips can turn 20 to 55 degrees from one row to the next. Diffuse the flow
        // over `smooth` rings of welded neighbors in the lock (parting sides kept apart), so a
        // strand bends over several rows instead of kinking along one straight edge row.
        let mut dir: Vec<V3> = fl.dir.clone();
        for _ in 0..smooth {
            let next: Vec<V3> = (0..hs.tris.len())
                .map(|i| {
                    if !hs.is_primary(i) {
                        return dir[i];
                    }
                    let mut s = [0.0; 3];
                    for &w in &hs.wt[i] {
                        for &(j, _) in &acc[&(w, hs.lock[i])] {
                            if dot(dir[j], dir[i]) > 0.3 {
                                s = add(s, mul(dir[j], hs.area(j)));
                            }
                        }
                    }
                    if len(s) > 1e-12 { unit(s) } else { dir[i] }
                })
                .collect();
            dir = next;
        }
        for list in acc.values_mut() {
            for e in list.iter_mut() {
                e.1 = mul(dir[e.0], hs.area(e.0));
            }
        }
        let mut ctan = vec![[[0.0; 3]; 3]; hs.tris.len()];
        for i in 0..hs.tris.len() {
            if !hs.is_primary(i) {
                continue;
            }
            for k in 0..3 {
                let list = &acc[&(hs.wt[i][k], hs.lock[i])];
                let mut s = [0.0; 3];
                for &(j, d) in list {
                    if dot(dir[j], dir[i]) > 0.3 {
                        s = add(s, d);
                    }
                }
                ctan[i][k] = if len(s) > 1e-12 { unit(s) } else { dir[i] };
            }
        }
        Surf { hs, fl, sign, across, ctan }
    }

    fn normal(&self, tri: usize) -> V3 {
        let i = self.hs.primary[tri];
        mul(self.hs.normal(i), self.sign[i])
    }

    /// Unit flow tangent in the triangle's plane at `a`.
    fn tangent(&self, a: At) -> V3 {
        let i = a.tri;
        let b = bary3(self.hs.corners(i), a.p).map(|x| x.clamp(0.0, 1.0));
        let mut t = [0.0; 3];
        for k in 0..3 {
            t = add(t, mul(self.ctan[i][k], b[k]));
        }
        let n = self.hs.normal(i);
        let t = sub(t, mul(n, dot(t, n)));
        if len(t) > 1e-9 {
            unit(t)
        } else {
            self.fl.dir[i]
        }
    }

    /// Moves `dist` along direction `d` (in the plane of `a.tri`) over the welded surface of the
    /// lock. Returns the end point and the direction there, or None at the lock's outline.
    fn walk(&self, mut a: At, mut d: V3, mut dist: f64) -> Option<(At, V3)> {
        for _ in 0..256 {
            let c = self.hs.corners(a.tri);
            let b0 = bary3(c, a.p);
            let b1 = bary3(c, add(a.p, d));
            let db = [b1[0] - b0[0], b1[1] - b0[1], b1[2] - b0[2]];
            let (mut s_exit, mut ke) = (f64::INFINITY, 3);
            for k in 0..3 {
                if db[k] < -1e-12 {
                    let s = b0[k].max(0.0) / -db[k];
                    if s < s_exit {
                        s_exit = s;
                        ke = k;
                    }
                }
            }
            if s_exit >= dist || ke == 3 {
                return Some((At { tri: a.tri, p: add(a.p, mul(d, dist)) }, d));
            }
            let q = add(a.p, mul(d, s_exit));
            dist -= s_exit;
            let w = self.hs.wt[a.tri];
            let e = ekey(w[(ke + 1) % 3], w[(ke + 2) % 3]);
            let list = self.across.get(&e)?;
            if list.len() != 2 {
                return None;
            }
            let next = if list[0] == a.tri { list[1] } else { list[0] };
            if self.hs.lock[next] != self.hs.lock[a.tri] {
                return None;
            }
            d = unit(hinge(self.hs, a.tri, next, e, d));
            a = At { tri: next, p: q };
        }
        None
    }
}

impl StrandField {
    /// Traces strands `sep` apart (model units) over every lock of `hs`.
    /// `smooth`: rings of flow diffusion before tracing (see `Surf::new`).
    pub fn build(hs: &HairSet, fl: &Flow, sep: f64, seed: u64, smooth: usize) -> StrandField {
        let sign = orient(hs);
        let surf = Surf::new(hs, fl, &sign, smooth);
        let cell = 2.0 * sep;
        let step = 0.3 * sep;
        let d_test = 0.5 * sep;
        let mut f = StrandField { sep, lines: 0, samples: 0, seed, cell, pts: Vec::new(), grid: HashMap::new(), sign: sign.clone() };
        let key = |p: V3| ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64, (p[2] / cell).floor() as i64);
        // Is there a strand of another line (same lock, same facing) within `r` of `p`?
        let near = |f: &StrandField, p: V3, n: V3, lock: usize, r: f64| -> bool {
            let (cx, cy, cz) = key(p);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(list) = f.grid.get(&(cx + dx, cy + dy, cz + dz)) else { continue };
                        for &k in list {
                            let s = &f.pts[k as usize];
                            if s.lock as usize != lock || dot(s.n, n) < 0.3 {
                                continue;
                            }
                            let d = sub(s.p, p);
                            if dot(d, d) < r * r {
                                return true;
                            }
                        }
                    }
                }
            }
            false
        };
        // Seeds: centroids of the primary triangles, largest first within each lock.
        let mut cents: Vec<At> = (0..hs.tris.len()).filter(|&i| hs.is_primary(i)).map(|i| At { tri: i, p: hs.centroid(i) }).collect();
        cents.sort_by(|a, b| hs.lock[a.tri].cmp(&hs.lock[b.tri]).then(hs.area(b.tri).total_cmp(&hs.area(a.tri))));
        let mut queue: VecDeque<At> = VecDeque::new();
        let mut ci = 0usize;
        let max_steps = 4000usize;
        loop {
            let s0 = if let Some(s) = queue.pop_front() {
                s
            } else if ci < cents.len() {
                ci += 1;
                cents[ci - 1]
            } else {
                break;
            };
            let lock = hs.lock[s0.tri];
            if near(&f, s0.p, surf.normal(s0.tri), lock, 0.9 * sep) {
                continue;
            }
            // Trace both ways from the seed.
            let mut halves: [Vec<(At, V3)>; 2] = [Vec::new(), Vec::new()];
            for (h, dir) in [(0usize, 1.0f64), (1, -1.0)] {
                let mut a = s0;
                let mut d = mul(surf.tangent(a), dir);
                for _ in 0..max_steps {
                    let Some((b, dd)) = surf.walk(a, d, step) else { break };
                    let t = mul(surf.tangent(b), dir);
                    // A parting or a sharp turn in the flow ends the strand.
                    if dot(t, dd) < 0.64 {
                        break;
                    }
                    if near(&f, b.p, surf.normal(b.tri), lock, d_test) {
                        break;
                    }
                    halves[h].push((b, t));
                    a = b;
                    d = t;
                }
            }
            let mut line: Vec<(At, V3)> = halves[1].iter().rev().map(|&(a, t)| (a, mul(t, -1.0))).collect();
            line.push((s0, surf.tangent(s0)));
            line.extend(halves[0].iter().copied());
            if line.len() < 3 {
                continue;
            }
            let id = f.lines as u32;
            f.lines += 1;
            let mut arc = 0.0f64;
            let start = f.pts.len();
            for (k, &(a, t)) in line.iter().enumerate() {
                if k > 0 {
                    arc += len(sub(a.p, line[k - 1].0.p));
                }
                let n = surf.normal(a.tri);
                let idx = f.pts.len() as u32;
                f.pts.push(Sample { p: a.p, t, n, line: id, lock: lock as u32, arc: arc as f32 });
                f.grid.entry(key(a.p)).or_default().push(idx);
            }
            // New seeds one spacing to each side, every other sample.
            for k in (start..f.pts.len()).step_by(2) {
                let (a, t) = line[k - start];
                let n = hs.normal(a.tri);
                let b = unit(cross(n, t));
                for side in [1.0, -1.0] {
                    if let Some((s, _)) = surf.walk(a, mul(b, side), sep) {
                        queue.push_back(s);
                    }
                }
            }
        }
        f.samples = f.pts.len();
        f
    }

    /// Fiber value (about -1 in a gap, up to about 2.2 on a bright strand) at the point `bary` of
    /// triangle `tri`, or -1 where no strand is near.
    pub fn raw(&self, hs: &HairSet, tri: usize, bary: [f64; 3]) -> f64 {
        let near = self.near(hs, tri, bary);
        let sep = self.sep;
        let mut val = -1.0f64;
        let (mut left, mut right) = ((f64::NEG_INFINITY, -1.0f64), (f64::INFINITY, -1.0f64));
        for &(line, _, across, arc) in &near {
            let sd = hash(self.seed ^ (line as u64 + 1).wrapping_mul(0x9E37_79B9));
            let h = |k: u64| h01(sd ^ k.wrapping_mul(0x51_7CC1_B727_220A));
            // The strand's center, in spacings from this point (x = 0), with a slow sideways wobble.
            let wob = 0.6 * (vnoise(sd ^ 5, arc / (sep * 25.0), 0.5) - 0.5);
            let c = -across / sep + wob;
            let hw = 0.30 + 0.25 * h(2);
            let pres = 0.35 + 0.65 * smooth((vnoise(sd ^ 9, arc / (sep * 30.0), 0.5) - 0.25) / 0.2);
            let bright = 0.3 + 0.7 * h(3);
            let at = |d: f64| -1.0 + smooth((1.0 - d) / crate::hair::FIBER_EDGE) * pres * (1.0 + 1.2 * bright * (1.0 - d * d).max(0.0).sqrt());
            let d = c.abs() / hw;
            if d < 1.0 {
                val = val.max(at(d));
            }
            if c + hw <= 0.0 && c + hw > left.0 {
                left = (c + hw, at(0.75));
            }
            if c - hw >= 0.0 && c - hw < right.0 {
                right = (c - hw, at(0.75));
            }
        }
        // A dark gap narrower than 0.4 of a spacing renders as a one-pixel line close up: fill it.
        let gap = right.0 - left.0;
        if gap < 0.4 && gap > 0.0 {
            let t = -left.0 / gap;
            val = val.max(left.1 + (right.1 - left.1) * t);
        }
        val
    }

    /// Distance (model units, in the surface plane) to the nearest strand, if any within reach.
    pub fn nearest(&self, hs: &HairSet, tri: usize, bary: [f64; 3]) -> Option<f64> {
        self.near(hs, tri, bary).iter().map(|e| e.1).min_by(f64::total_cmp)
    }

    /// Nearest point of each strand line near the point: (line, distance in the plane, signed
    /// offset across the strand, arc length along it).
    fn near(&self, hs: &HairSet, tri: usize, bary: [f64; 3]) -> Vec<(u32, f64, f64, f64)> {
        let c = hs.corners(tri);
        let p: V3 = std::array::from_fn(|q| (0..3).map(|k| bary[k] * c[k][q]).sum());
        let i = hs.primary[tri];
        let n = mul(hs.normal(i), self.sign[i]);
        let lock = hs.lock[i] as u32;
        let sep = self.sep;
        let cell = self.cell;
        let (cx, cy, cz) = ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64, (p[2] / cell).floor() as i64);
        // Nearest point of each line: (line, distance in the plane, signed across offset, arc).
        let mut near: Vec<(u32, f64, f64, f64)> = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(list) = self.grid.get(&(cx + dx, cy + dy, cz + dz)) else { continue };
                    for &k in list {
                        let s = &self.pts[k as usize];
                        if s.lock != lock || dot(s.n, n) < 0.3 {
                            continue;
                        }
                        // Closest point on the segment to the next sample of the same line.
                        let (q, arc) = match self.pts.get(k as usize + 1) {
                            Some(s2) if s2.line == s.line => {
                                let seg = sub(s2.p, s.p);
                                let l2 = dot(seg, seg).max(1e-30);
                                let u = (dot(sub(p, s.p), seg) / l2).clamp(0.0, 1.0);
                                (add(s.p, mul(seg, u)), s.arc as f64 + u * (s2.arc - s.arc) as f64)
                            }
                            _ => (s.p, s.arc as f64),
                        };
                        let r = sub(p, q);
                        let rn = dot(r, s.n);
                        if rn.abs() > 0.75 * sep {
                            continue;
                        }
                        let rt = sub(r, mul(s.n, rn));
                        let dl = len(rt);
                        if dl > 2.2 * sep {
                            continue;
                        }
                        let across = dot(rt, unit(cross(s.n, s.t)));
                        match near.iter_mut().find(|e| e.0 == s.line) {
                            Some(e) => {
                                if dl < e.1 {
                                    *e = (s.line, dl, across, arc);
                                }
                            }
                            None => near.push((s.line, dl, across, arc)),
                        }
                    }
                }
            }
        }
        near
    }
}

/// Surface strand layer: a main set of strands and an optional finer set, as a zero-mean L offset.
pub struct Fibers3 {
    pub gain: f64,
    pub fine: f64,
    pub main: StrandField,
    pub fine_field: Option<StrandField>,
    pub mean: f64,
}

impl Fibers3 {
    /// `width` and `fine_width` in texels, `scale` texels per model unit.
    #[allow(clippy::too_many_arguments)]
    pub fn build(hs: &HairSet, fl: &Flow, gain: f64, width: f64, fine_width: f64, fine: f64, scale: f64, seed: u64, smooth: usize) -> Fibers3 {
        let main = StrandField::build(hs, fl, width / scale, seed, smooth);
        let fine_field = (fine > 0.0).then(|| StrandField::build(hs, fl, fine_width / scale, seed ^ 0xF1, smooth));
        let mut f = Fibers3 { gain, fine, main, fine_field, mean: 0.0 };
        // Mean over area-weighted random points of the primary triangles.
        let prim: Vec<usize> = (0..hs.tris.len()).filter(|&i| hs.is_primary(i)).collect();
        let total: f64 = prim.iter().map(|&i| hs.area(i)).sum();
        let (mut acc, mut n) = (0.0, 0usize);
        for &i in &prim {
            let k = ((hs.area(i) / total.max(1e-30)) * 30000.0).round() as usize;
            for j in 0..k.max(1) {
                let (a, b) = (h01(seed ^ (i as u64) << 20 ^ j as u64 ^ 0xA1), h01(seed ^ (i as u64) << 20 ^ j as u64 ^ 0xB2));
                let (a, b) = if a + b > 1.0 { (1.0 - a, 1.0 - b) } else { (a, b) };
                acc += f.raw(hs, i, [1.0 - a - b, a, b]);
                n += 1;
            }
        }
        f.mean = acc / n.max(1) as f64;
        f
    }

    pub fn raw(&self, hs: &HairSet, tri: usize, bary: [f64; 3]) -> f64 {
        let a = self.main.raw(hs, tri, bary);
        match &self.fine_field {
            Some(ff) => a + self.fine * ff.raw(hs, tri, bary),
            None => a,
        }
    }

    /// Zero-mean lightness offset (Lab L).
    pub fn at(&self, hs: &HairSet, tri: usize, bary: [f64; 3]) -> f64 {
        self.gain * (self.raw(hs, tri, bary) - self.mean)
    }

    /// Share of surface points with no main strand within one spacing (holes in the strand cover).
    pub fn bare_share(&self, hs: &HairSet) -> f64 {
        let prim: Vec<usize> = (0..hs.tris.len()).filter(|&i| hs.is_primary(i)).collect();
        let (mut bare, mut n) = (0usize, 0usize);
        for &i in &prim {
            for bc in [[1.0 / 3.0; 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]] {
                n += 1;
                if self.main.nearest(hs, i, bc).is_none_or(|d| d > self.main.sep) {
                    bare += 1;
                }
            }
        }
        bare as f64 / n.max(1) as f64
    }
}
