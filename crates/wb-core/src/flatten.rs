//! Flattening a 3D surface piece onto the plane with least-squares conformal maps (LSCM, Lévy et
//! al. 2002): angles are kept as well as possible, so painted shapes look like the model.
//! Solved with Jacobi-preconditioned conjugate gradients on the normal equations; no outside crates.

/// Flattens one connected piece. `pos` per vertex, `tris` index into `pos`.
/// Returns one UV per vertex in model units (scale follows the 3D size), with the piece's
/// triangles wound counter-clockwise, plus the number of triangles still flipped (folded).
pub fn lscm(pos: &[[f64; 3]], tris: &[[u32; 3]]) -> Result<(Vec<[f64; 2]>, usize), String> {
    let n = pos.len();
    if n < 3 || tris.is_empty() {
        return Err("piece too small to flatten".into());
    }
    // Pins: the two vertices farthest apart (two sweeps).
    let far = |from: usize| (0..n).max_by(|&a, &b| dist(pos[from], pos[a]).total_cmp(&dist(pos[from], pos[b]))).unwrap();
    let p0 = far(0);
    let p1 = far(p0);
    let d01 = dist(pos[p0], pos[p1]);
    if d01 <= 0.0 {
        return Err("piece has no extent".into());
    }
    // Per triangle: complex coefficients (a_j, b_j) for its three corners, already divided by sqrt(2A).
    let mut rows: Vec<([u32; 3], [f64; 3], [f64; 3])> = Vec::with_capacity(tris.len());
    for t in tris {
        let [a, b, c] = t.map(|v| pos[v as usize]);
        let e1 = sub(b, a);
        let e2 = sub(c, a);
        let l1 = norm(e1);
        if l1 <= 1e-12 {
            continue;
        }
        let x = scale(e1, 1.0 / l1);
        let nrm = cross(e1, e2);
        let area2 = norm(nrm);
        if area2 <= 1e-14 {
            continue;
        }
        let y = scale(cross(nrm, e1), 1.0 / (area2 * l1));
        let q = [[0.0, 0.0], [l1, 0.0], [dot(e2, x), dot(e2, y)]];
        let s = 1.0 / area2.sqrt();
        let mut ar = [0.0; 3];
        let mut br = [0.0; 3];
        for j in 0..3 {
            let (k1, k2) = ((j + 1) % 3, (j + 2) % 3);
            ar[j] = (q[k2][0] - q[k1][0]) * s;
            br[j] = (q[k2][1] - q[k1][1]) * s;
        }
        rows.push((*t, ar, br));
    }
    if rows.is_empty() {
        return Err("piece has no area".into());
    }
    // Unknowns: u then v for every vertex; pinned ones are fixed.
    let mut fixed = vec![None; 2 * n];
    fixed[p0] = Some(0.0);
    fixed[n + p0] = Some(0.0);
    fixed[p1] = Some(d01);
    fixed[n + p1] = Some(0.0);
    // A x for full vector x (pinned included): two rows per triangle.
    let apply = |x: &[f64], out: &mut Vec<f64>| {
        out.clear();
        for (t, a, b) in &rows {
            let (mut re, mut im) = (0.0, 0.0);
            for j in 0..3 {
                let v = t[j] as usize;
                re += a[j] * x[v] - b[j] * x[n + v];
                im += b[j] * x[v] + a[j] * x[n + v];
            }
            out.push(re);
            out.push(im);
        }
    };
    let apply_t = |y: &[f64], out: &mut [f64]| {
        out.iter_mut().for_each(|o| *o = 0.0);
        for (r, (t, a, b)) in rows.iter().enumerate() {
            let (re, im) = (y[2 * r], y[2 * r + 1]);
            for j in 0..3 {
                let v = t[j] as usize;
                out[v] += a[j] * re + b[j] * im;
                out[n + v] += -b[j] * re + a[j] * im;
            }
        }
    };
    // Right side: -A^T A x_fixed, restricted to free unknowns.
    let mut xf = vec![0.0; 2 * n];
    for (i, f) in fixed.iter().enumerate() {
        if let Some(v) = f {
            xf[i] = *v;
        }
    }
    let mut tmp = Vec::new();
    apply(&xf, &mut tmp);
    let mut rhs = vec![0.0; 2 * n];
    apply_t(&tmp, &mut rhs);
    for (i, r) in rhs.iter_mut().enumerate() {
        *r = if fixed[i].is_some() { 0.0 } else { -*r };
    }
    // Jacobi preconditioner: diagonal of A^T A.
    let mut diag = vec![0.0; 2 * n];
    for (t, a, b) in &rows {
        for j in 0..3 {
            let v = t[j] as usize;
            let w = a[j] * a[j] + b[j] * b[j];
            diag[v] += w;
            diag[n + v] += w;
        }
    }
    let mul = |x: &[f64], out: &mut [f64], tmp: &mut Vec<f64>| {
        apply(x, tmp);
        apply_t(tmp, out);
        for (i, o) in out.iter_mut().enumerate() {
            if fixed[i].is_some() {
                *o = 0.0;
            }
        }
    };
    let pre = |r: &[f64], z: &mut [f64]| {
        for i in 0..r.len() {
            z[i] = if fixed[i].is_some() || diag[i] <= 0.0 { 0.0 } else { r[i] / diag[i] };
        }
    };
    let mut x = vec![0.0; 2 * n];
    // Start from a planar projection along the two pins, which CG then refines.
    let axis = scale(sub(pos[p1], pos[p0]), 1.0 / d01);
    for v in 0..n {
        if fixed[v].is_none() {
            x[v] = dot(sub(pos[v], pos[p0]), axis);
        }
    }
    let mut ax = vec![0.0; 2 * n];
    mul(&x, &mut ax, &mut tmp);
    let mut r: Vec<f64> = (0..2 * n).map(|i| rhs[i] - ax[i]).collect();
    let mut z = vec![0.0; 2 * n];
    pre(&r, &mut z);
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let r0 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
    let mut ap = vec![0.0; 2 * n];
    for _ in 0..(40 * n).max(2000) {
        let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rn <= 1e-10 * r0 {
            break;
        }
        mul(&p, &mut ap, &mut tmp);
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if pap.abs() < 1e-300 {
            break;
        }
        let alpha = rz / pap;
        for i in 0..2 * n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        pre(&r, &mut z);
        let rz2: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz2 / rz;
        rz = rz2;
        for i in 0..2 * n {
            p[i] = z[i] + beta * p[i];
        }
    }
    let mut uv: Vec<[f64; 2]> = (0..n).map(|v| [fixed[v].unwrap_or(x[v]), fixed[n + v].unwrap_or(x[n + v])]).collect();
    if uv.iter().any(|q| !q[0].is_finite() || !q[1].is_finite()) {
        return Err("flattening did not converge".into());
    }
    // Wind counter-clockwise: mirror if most of the area came out clockwise.
    let signed: f64 = tris.iter().map(|t| area(t.map(|v| uv[v as usize]))).sum();
    if signed < 0.0 {
        uv.iter_mut().for_each(|q| q[0] = -q[0]);
    }
    // The pins sit at the 3D straight-line distance, which is shorter than the distance along a
    // curved surface; rescale so the flat area equals the 3D area.
    let a3: f64 = tris.iter().map(|t| area3(t.map(|v| pos[v as usize]))).sum();
    let a2: f64 = tris.iter().map(|t| area(t.map(|v| uv[v as usize])).abs()).sum();
    if a2 > 0.0 && a3 > 0.0 {
        let k = (a3 / a2).sqrt();
        uv.iter_mut().for_each(|q| *q = [q[0] * k, q[1] * k]);
    }
    let flips = tris.iter().filter(|t| area(t.map(|v| uv[v as usize])) < 0.0).count();
    Ok((uv, flips))
}

pub fn area(p: [[f64; 2]; 3]) -> f64 {
    0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]))
}

pub fn area3(p: [[f64; 3]; 3]) -> f64 {
    0.5 * norm(cross(sub(p[1], p[0]), sub(p[2], p[0])))
}

pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
pub fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

/// One triangle for `relax_sd`: its corners, its 3D area and the inverse of its edge matrix in the
/// triangle's own plane.
struct SdTri {
    v: [usize; 3],
    area: f64,
    ri: [[f64; 2]; 2],
}

/// Symmetric Dirichlet energy (area weighted, 4 per unit area when the map is an isometry) of the
/// flat layout `u` and, when `g` is given, its gradient. Infinite when a triangle flips or collapses.
fn sd_energy(ts: &[SdTri], u: &[[f64; 2]], mut g: Option<&mut [[f64; 2]]>) -> f64 {
    if let Some(g) = g.as_deref_mut() {
        g.iter_mut().for_each(|x| *x = [0.0, 0.0]);
    }
    let mut e = 0.0;
    for t in ts {
        let [u0, u1, u2] = t.v.map(|i| u[i]);
        let dm = [[u1[0] - u0[0], u2[0] - u0[0]], [u1[1] - u0[1], u2[1] - u0[1]]];
        let r = &t.ri;
        let j = [
            [dm[0][0] * r[0][0] + dm[0][1] * r[1][0], dm[0][0] * r[0][1] + dm[0][1] * r[1][1]],
            [dm[1][0] * r[0][0] + dm[1][1] * r[1][0], dm[1][0] * r[0][1] + dm[1][1] * r[1][1]],
        ];
        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        if det <= 1e-12 {
            return f64::INFINITY;
        }
        let f2 = j[0][0] * j[0][0] + j[0][1] * j[0][1] + j[1][0] * j[1][0] + j[1][1] * j[1][1];
        e += t.area * (f2 + f2 / (det * det));
        if let Some(g) = g.as_deref_mut() {
            let (a, b) = (2.0 * (1.0 + 1.0 / (det * det)), 2.0 * f2 / (det * det * det));
            let cof = [[j[1][1], -j[1][0]], [-j[0][1], j[0][0]]];
            let gj = [
                [t.area * (a * j[0][0] - b * cof[0][0]), t.area * (a * j[0][1] - b * cof[0][1])],
                [t.area * (a * j[1][0] - b * cof[1][0]), t.area * (a * j[1][1] - b * cof[1][1])],
            ];
            // dE/dDm = gj * ri^T
            let h = [
                [gj[0][0] * r[0][0] + gj[0][1] * r[0][1], gj[0][0] * r[1][0] + gj[0][1] * r[1][1]],
                [gj[1][0] * r[0][0] + gj[1][1] * r[0][1], gj[1][0] * r[1][0] + gj[1][1] * r[1][1]],
            ];
            let (c1, c2) = ([h[0][0], h[1][0]], [h[0][1], h[1][1]]);
            for k in 0..2 {
                g[t.v[1]][k] += c1[k];
                g[t.v[2]][k] += c2[k];
                g[t.v[0]][k] -= c1[k] + c2[k];
            }
        }
    }
    e
}

/// The largest step along `d` that keeps every triangle's orientation (the first zero of its area).
fn sd_max_step(ts: &[SdTri], u: &[[f64; 2]], d: &[[f64; 2]]) -> f64 {
    let mut best = f64::INFINITY;
    for t in ts {
        let m = |p: &[[f64; 2]]| [[p[t.v[1]][0] - p[t.v[0]][0], p[t.v[2]][0] - p[t.v[0]][0]], [p[t.v[1]][1] - p[t.v[0]][1], p[t.v[2]][1] - p[t.v[0]][1]]];
        let (a, n) = (m(u), m(d));
        let c0 = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        let b = a[0][0] * n[1][1] + n[0][0] * a[1][1] - a[0][1] * n[1][0] - n[0][1] * a[1][0];
        let c = n[0][0] * n[1][1] - n[0][1] * n[1][0];
        let root = if c.abs() < 1e-18 {
            if b < 0.0 { -c0 / b } else { f64::INFINITY }
        } else {
            let disc = b * b - 4.0 * c * c0;
            if disc < 0.0 {
                f64::INFINITY
            } else {
                let s = disc.sqrt();
                let (r1, r2) = ((-b - s) / (2.0 * c), (-b + s) / (2.0 * c));
                [r1, r2].into_iter().filter(|&r| r > 0.0).fold(f64::INFINITY, f64::min)
            }
        };
        best = best.min(root);
    }
    best
}

/// Relaxes a flat piece toward the 3D surface by minimizing the symmetric Dirichlet energy, which
/// punishes any stretch or squeeze (SLIM's energy, Rabinovich et al. 2017), with L-BFGS steps that
/// stop short of flipping a triangle. `uv` must start without flipped triangles and with the area
/// of the 3D piece. Returns the iterations run and the energy before and after, per unit of area
/// (4 is an isometry).
pub fn relax_sd(pos: &[[f64; 3]], tris: &[[u32; 3]], uv: &mut Vec<[f64; 2]>, max_iter: usize) -> (usize, f64, f64) {
    let mut ts: Vec<SdTri> = Vec::new();
    for t in tris {
        let [a, b, c] = t.map(|v| pos[v as usize]);
        let (e1, e2) = (sub(b, a), sub(c, a));
        let l1 = norm(e1);
        if l1 < 1e-12 {
            continue;
        }
        let x = scale(e1, 1.0 / l1);
        let q2x = dot(e2, x);
        let q2y = norm(sub(e2, scale(x, q2x)));
        let d = l1 * q2y;
        if d < 1e-14 {
            continue;
        }
        ts.push(SdTri { v: t.map(|v| v as usize), area: 0.5 * d, ri: [[q2y / d, -q2x / d], [0.0, l1 / d]] });
    }
    let total: f64 = ts.iter().map(|t| t.area).sum::<f64>().max(1e-30);
    let n = uv.len();
    let e0 = sd_energy(&ts, uv, None);
    if !e0.is_finite() || ts.is_empty() {
        return (0, e0 / total, e0 / total);
    }
    let mean_edge = (total / ts.len() as f64).sqrt();
    let flat = |v: &[[f64; 2]]| -> Vec<f64> { v.iter().flat_map(|x| [x[0], x[1]]).collect() };
    let unflat = |v: &[f64]| -> Vec<[f64; 2]> { v.chunks(2).map(|c| [c[0], c[1]]).collect() };
    let dotv = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let mut g = vec![[0.0; 2]; n];
    let mut e = sd_energy(&ts, uv, Some(&mut g));
    let mut hist: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut iters = 0;
    let mut resets = 0;
    let mut quiet = 0;
    while iters < max_iter {
        iters += 1;
        let gf = flat(&g);
        let gn = dotv(&gf, &gf).sqrt();
        if gn < 1e-12 * total.max(1.0) {
            break;
        }
        // Two-loop recursion for the L-BFGS direction.
        let mut dir: Vec<f64> = gf.iter().map(|v| -v).collect();
        if hist.is_empty() {
            let k = 0.05 * mean_edge / gn;
            dir.iter_mut().for_each(|v| *v *= k);
        } else {
            let mut al = vec![0.0; hist.len()];
            for i in (0..hist.len()).rev() {
                let rho = 1.0 / dotv(&hist[i].1, &hist[i].0).max(1e-300);
                al[i] = rho * dotv(&hist[i].0, &dir);
                for (d, y) in dir.iter_mut().zip(&hist[i].1) {
                    *d -= al[i] * y;
                }
            }
            let (s, y) = hist.last().unwrap();
            let gamma = dotv(s, y) / dotv(y, y).max(1e-300);
            dir.iter_mut().for_each(|v| *v *= gamma);
            for i in 0..hist.len() {
                let rho = 1.0 / dotv(&hist[i].1, &hist[i].0).max(1e-300);
                let be = rho * dotv(&hist[i].1, &dir);
                for (d, s) in dir.iter_mut().zip(&hist[i].0) {
                    *d += s * (al[i] - be);
                }
            }
        }
        if dotv(&dir, &gf) >= 0.0 {
            // Not a descent direction: fall back to steepest descent.
            hist.clear();
            let k = 0.05 * mean_edge / gn;
            dir = gf.iter().map(|v| -v * k).collect();
        }
        let d2 = unflat(&dir);
        let amax = sd_max_step(&ts, uv, &d2);
        let mut a = 1.0f64.min(0.9 * amax);
        let slope = dotv(&gf, &dir);
        let mut accepted = false;
        let mut cand = uv.clone();
        for _ in 0..40 {
            for i in 0..n {
                cand[i] = [uv[i][0] + a * d2[i][0], uv[i][1] + a * d2[i][1]];
            }
            let en = sd_energy(&ts, &cand, None);
            if en.is_finite() && en <= e + 1e-4 * a * slope {
                accepted = true;
                break;
            }
            a *= 0.5;
        }
        if !accepted {
            if resets >= 2 {
                break;
            }
            resets += 1;
            hist.clear();
            continue;
        }
        let mut gn2 = vec![[0.0; 2]; n];
        let e2 = sd_energy(&ts, &cand, Some(&mut gn2));
        let s: Vec<f64> = flat(&cand).iter().zip(flat(uv).iter()).map(|(x, y)| x - y).collect();
        let y: Vec<f64> = flat(&gn2).iter().zip(&gf).map(|(x, y)| x - y).collect();
        if dotv(&s, &y) > 1e-14 * dotv(&s, &s).max(1e-300) {
            hist.push((s, y));
            if hist.len() > 12 {
                hist.remove(0);
            }
        }
        let drop = (e - e2) / e.max(1e-300);
        *uv = cand;
        g = gn2;
        e = e2;
        quiet = if drop < 1e-7 { quiet + 1 } else { 0 };
        if quiet >= 5 {
            break;
        }
    }
    (iters, e0 / total, e / total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bent strip (two quads folded 90°) flattens without stretch: edge lengths survive.
    #[test]
    fn folded_strip_unfolds() {
        let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
        let tris = [[0, 1, 3], [0, 3, 2], [2, 3, 5], [2, 5, 4]];
        let (uv, flips) = lscm(&pos, &tris).unwrap();
        assert_eq!(flips, 0);
        let d = |a: usize, b: usize| ((uv[a][0] - uv[b][0]).powi(2) + (uv[a][1] - uv[b][1]).powi(2)).sqrt();
        for (a, b) in [(0, 1), (2, 3), (4, 5), (0, 2), (2, 4), (1, 3), (3, 5)] {
            assert!((d(a, b) - 1.0).abs() < 1e-6, "edge {a}-{b} is {}", d(a, b));
        }
        assert!((d(0, 4) - 2.0).abs() < 1e-6);
    }
}
