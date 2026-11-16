//! Rebuilds card-based hair geometry from the strand coordinates of `hair` (flow-aligned u, v).
//!
//! Generic: any hair made of flat cards. Each flow region is remeshed in its own strand space
//! (u across, v root to tip) with a constrained Delaunay triangulation:
//! - a regular grid of strand-space points gives clean strips with even edge flow root to tip;
//! - card outlines, seams between regions and folds (interior edges bending more than `fold_deg`)
//!   are constraints, subdivided the same way on both sides, so the surface keeps its shape and
//!   stays welded;
//! - tips are trimmed to the stock alpha outline (the 0.5 iso line), so geometry carries the
//!   outline where a coarse alpha cutout stair-steps;
//! - small gaps between cards of different locks close by moving free boundary vertices onto the
//!   neighbor card;
//! - duplicate back faces are dropped, kept where the stock had them, or added everywhere.
//! Every new vertex lies on the stock surface (or on a fold line of it), so silhouette and volume
//! stay those of the stock hair.
use crate::glb::{Chunk, Glb, CHUNK_BIN, CHUNK_JSON};
use crate::hair::{self, add, cross, dot, ekey, len, mul, sub, Flow, HairSet, Param, Sheet, V3};
use crate::Model;
use serde_json::{json, Value};
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backfaces {
    Drop,
    Stock,
    All,
}

pub struct ModelOpts {
    /// Target count of front triangles before trimming (sets the grid spacing).
    pub tris: usize,
    /// Rows are this many times longer than columns are wide (strips along the flow).
    pub row_stretch: f64,
    /// Interior edges folding more than this (degrees) keep a vertex row.
    pub fold_deg: f64,
    /// Trim to the stock alpha at this level (0..1; 0 = off).
    pub alpha_cut: f64,
    /// Close gaps between cards of different locks up to this many grid spacings (0 = off).
    pub gap: f64,
    pub backfaces: Backfaces,
}

pub struct HairTri {
    pub v: [usize; 3],
    /// Stock UV per corner (from the parent triangle's stock UVs).
    pub suv: [[f64; 2]; 3],
    /// Strand coordinates (u, v) per corner, interpolated from the stock solve on the parent.
    pub puv: [[f64; 2]; 3],
    /// Parent: the stock selected triangle (index into `HairSet::tris`) this one lies on.
    pub parent: usize,
    pub region: usize,
    pub back: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Report {
    pub spacing: f64,
    pub stock_front: usize,
    pub stock_back: usize,
    pub front: usize,
    pub back: usize,
    pub vertices: usize,
    pub regions_remeshed: usize,
    pub regions_per_triangle: usize,
    pub flipped_tris: usize,
    pub fold_edges: usize,
    pub constraints_skipped: usize,
    pub merged_points: usize,
    pub area_stock: f64,
    pub area_before_trim: f64,
    pub area_after_trim: f64,
    pub gap_vertices: usize,
    pub gap_max_move: f64,
    pub gap_hist: [usize; 6],
}

pub struct HairModel {
    pub pos: Vec<V3>,
    /// Per vertex: the stock mesh vertex whose other attributes (skin, color) it copies.
    pub src: Vec<u32>,
    pub tris: Vec<HairTri>,
    pub report: Report,
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
enum Key {
    Orig(usize),
    /// Point i of k on welded edge (lo, hi).
    Edge(usize, usize, usize, usize),
    Grid(usize, usize),
    Cut(usize, usize),
}

struct Verts {
    pos: Vec<V3>,
    src: Vec<u32>,
    protect: Vec<bool>,
    map: HashMap<Key, usize>,
}

impl Verts {
    fn get(&mut self, k: Key, p: V3, src: u32, protect: bool) -> usize {
        if let Some(&i) = self.map.get(&k) {
            return i;
        }
        let i = self.pos.len();
        self.pos.push(p);
        self.src.push(src);
        self.protect.push(protect);
        self.map.insert(k, i);
        i
    }
}

fn bary(p: [[f64; 2]; 3], q: [f64; 2]) -> [f64; 3] {
    let d = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
    if d.abs() < 1e-300 {
        return [1.0 / 3.0; 3];
    }
    let b1 = ((q[0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (q[1] - p[0][1])) / d;
    let b2 = ((p[1][0] - p[0][0]) * (q[1] - p[0][1]) - (q[0] - p[0][0]) * (p[1][1] - p[0][1])) / d;
    [1.0 - b1 - b2, b1, b2]
}

fn area2(p: [[f64; 2]; 3]) -> f64 {
    (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1])
}

fn lerp3(a: V3, b: V3, t: f64) -> V3 {
    add(a, mul(sub(b, a), t))
}

fn lerp2(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

/// Closest point on a 3D triangle.
pub fn closest_on_tri(p: V3, a: V3, b: V3, c: V3) -> V3 {
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return add(a, mul(ab, d1 / (d1 - d3)));
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return add(a, mul(ac, d2 / (d2 - d6)));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return add(b, mul(sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    }
    let den = 1.0 / (va + vb + vc);
    add(a, add(mul(ab, vb * den), mul(ac, vc * den)))
}

fn seg_dist2(q: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l = dx * dx + dy * dy;
    let t = if l > 0.0 { (((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / l).clamp(0.0, 1.0) } else { 0.0 };
    let (x, y) = (a[0] + dx * t - q[0], a[1] + dy * t - q[1]);
    (x * x + y * y).sqrt()
}

/// Remeshes the selected hair (see the module notes).
pub fn remesh(model: &Model, hs: &HairSet, fl: &Flow, pr: &Param, stock: &Sheet, o: &ModelOpts) -> Result<HairModel, String> {
    let m = &model.meshes[hs.mesh];
    let prim: Vec<usize> = (0..hs.tris.len()).filter(|&i| hs.is_primary(i)).collect();
    let area: f64 = prim.iter().map(|&i| hs.area(i)).sum();
    let h = (2.0 * area / o.tris.max(1) as f64 / o.row_stretch.max(0.1)).sqrt();
    let hr = h * o.row_stretch.max(0.1);
    let sign = hair::orient(hs);
    let onorm = |i: usize| mul(hs.normal(i), sign[i]);
    let mut rep = Report { spacing: h, stock_front: prim.len(), stock_back: hs.tris.len() - prim.len(), area_stock: area, ..Default::default() };

    let mut has_back = vec![false; hs.tris.len()];
    for i in 0..hs.tris.len() {
        if !hs.is_primary(i) && !hair::same_winding(hs.wt[i], hs.wt[hs.primary[i]]) {
            has_back[hs.primary[i]] = true;
        }
    }
    // Mesh vertex per weld id, primaries per welded edge.
    let mut wv: HashMap<usize, u32> = HashMap::new();
    for &i in &prim {
        for c in 0..3 {
            let mv = m.tris[hs.tris[i]][c];
            wv.entry(hs.wt[i][c]).or_insert(mv);
        }
    }
    let mut uses: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut w_regions: HashMap<usize, Vec<usize>> = HashMap::new();
    for &i in &prim {
        let t = hs.wt[i];
        for k in 0..3 {
            uses.entry(ekey(t[k], t[(k + 1) % 3])).or_default().push(i);
            let r = w_regions.entry(t[k]).or_default();
            if !r.contains(&fl.region[i]) {
                r.push(fl.region[i]);
            }
        }
    }
    // Seams: welded edges shared by primaries of different regions, or by more than two.
    let seam = |e: (usize, usize)| -> bool {
        uses.get(&e).map(|u| u.len() > 2 || (u.len() == 2 && fl.region[u[0]] != fl.region[u[1]])).unwrap_or(false)
    };
    let w_protect = |w: usize| -> bool { w_regions.get(&w).map(|r| r.len() > 1).unwrap_or(false) || uses.iter().any(|(e, u)| (e.0 == w || e.1 == w) && u.len() > 2) };

    let nreg = fl.region_lock.len();
    let mut reg_tris = vec![Vec::new(); nreg];
    for &i in &prim {
        reg_tris[fl.region[i]].push(i);
    }
    let mut vs = Verts { pos: Vec::new(), src: Vec::new(), protect: Vec::new(), map: HashMap::new() };
    let mut out: Vec<HairTri> = Vec::new();
    let edge_k = |a: usize, b: usize| ((len(sub(hs.wpos[a], hs.wpos[b])) / h).ceil() as usize).max(1);

    for r in 0..nreg {
        let tl = &reg_tris[r];
        if tl.is_empty() {
            continue;
        }
        // Strand space: u scaled so the region is about isometric.
        let mut gu: Vec<f64> = tl
            .iter()
            .map(|&i| {
                let (g, _) = hair::basis_grads(hs.corners(i));
                len((0..3).fold([0.0; 3], |a, k| add(a, mul(g[k], pr.uv[i][k][0]))))
            })
            .collect();
        gu.sort_by(f64::total_cmp);
        let ku = 1.0 / gu[gu.len() / 2].max(1e-3);
        let st: Vec<[[f64; 2]; 3]> = tl.iter().map(|&i| pr.uv[i].map(|q| [q[0] * ku, q[1]])).collect();
        let sgn: Vec<f64> = (0..tl.len()).map(|k| area2(st[k]) * sign[tl[k]]).collect();
        let pos_n = sgn.iter().filter(|&&a| a > 0.0).count();
        let flipped = tl.len() - pos_n.max(tl.len() - pos_n);
        rep.flipped_tris += flipped;
        // Domains: the whole region, or each triangle alone where strand space folds over.
        let domains: Vec<Vec<usize>> = if flipped == 0 {
            rep.regions_remeshed += 1;
            vec![(0..tl.len()).collect()]
        } else {
            rep.regions_per_triangle += 1;
            (0..tl.len()).map(|k| vec![k]).collect()
        };
        let whole = flipped == 0;
        for dom in domains {
            // Strand-space position per weld id (u, v are continuous within a region).
            let mut s_of: HashMap<usize, [f64; 2]> = HashMap::new();
            for &k in &dom {
                for c in 0..3 {
                    s_of.entry(hs.wt[tl[k]][c]).or_insert(st[k][c]);
                }
            }
            // Edges of the domain: outline (one use) and folds (two uses, bent) are constraints.
            let mut e_use: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
            for &k in &dom {
                let t = hs.wt[tl[k]];
                for c in 0..3 {
                    e_use.entry(ekey(t[c], t[(c + 1) % 3])).or_default().push(k);
                }
            }
            let mut cons: Vec<(usize, usize)> = Vec::new();
            let mut keys: Vec<_> = e_use.keys().copied().collect();
            keys.sort();
            for e in keys {
                let u = &e_use[&e];
                let fold = u.len() == 2 && hair::ang(onorm(tl[u[0]]), onorm(tl[u[1]])) > o.fold_deg;
                if u.len() != 2 || fold {
                    if fold {
                        rep.fold_edges += 1;
                    }
                    cons.push(e);
                }
            }
            let mut cdt: ConstrainedDelaunayTriangulation<Point2<f64>> = ConstrainedDelaunayTriangulation::new();
            let mut gid: Vec<usize> = Vec::new(); // per cdt vertex index: global vertex
            let mut s_at: Vec<[f64; 2]> = Vec::new();
            let mut insert = |cdt: &mut ConstrainedDelaunayTriangulation<Point2<f64>>, s: [f64; 2], g: usize, gid: &mut Vec<usize>, s_at: &mut Vec<[f64; 2]>, merged: &mut usize| -> Option<spade::handles::FixedVertexHandle> {
                let hdl = cdt.insert(Point2::new(s[0], s[1])).ok()?;
                let ix = hdl.index();
                if ix == gid.len() {
                    gid.push(g);
                    s_at.push(s);
                } else if gid[ix] != g {
                    *merged += 1;
                }
                Some(hdl)
            };
            let mut segs: Vec<([f64; 2], [f64; 2])> = Vec::new();
            let mut chains: Vec<Vec<spade::handles::FixedVertexHandle>> = Vec::new();
            for &(a, b) in &cons {
                let k = edge_k(a, b);
                let mut chain = Vec::new();
                for i in 0..=k {
                    let t = i as f64 / k as f64;
                    let s = lerp2(s_of[&a], s_of[&b], t);
                    let (p, key, src, prot) = if i == 0 {
                        (hs.wpos[a], Key::Orig(a), wv[&a], w_protect(a))
                    } else if i == k {
                        (hs.wpos[b], Key::Orig(b), wv[&b], w_protect(b))
                    } else {
                        (lerp3(hs.wpos[a], hs.wpos[b], t), Key::Edge(a, b, i, k), if t < 0.5 { wv[&a] } else { wv[&b] }, seam((a, b)))
                    };
                    let g = vs.get(key, p, src, prot);
                    if let Some(hd) = insert(&mut cdt, s, g, &mut gid, &mut s_at, &mut rep.merged_points) {
                        chain.push(hd);
                    }
                }
                segs.push((s_of[&a], s_of[&b]));
                chains.push(chain);
            }
            // Grid points inside the domain, clear of the constraints.
            if whole {
                let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
                for &k in &dom {
                    for c in 0..3 {
                        for d in 0..2 {
                            lo[d] = lo[d].min(st[k][c][d]);
                            hi[d] = hi[d].max(st[k][c][d]);
                        }
                    }
                }
                let mut n = 0usize;
                let mut y = (lo[1] / hr).floor() * hr;
                while y <= hi[1] {
                    let mut x = (lo[0] / h).floor() * h;
                    while x <= hi[0] {
                        let q = [x, y];
                        x += h;
                        let Some(&k) = dom.iter().find(|&&k| bary(st[k], q).iter().all(|&b| b >= -1e-9)) else { continue };
                        if segs.iter().any(|s| seg_dist2(q, s.0, s.1) < 0.45 * h.min(hr)) {
                            continue;
                        }
                        let bc = bary(st[k], q);
                        let c3 = hs.corners(tl[k]);
                        let p = add(add(mul(c3[0], bc[0]), mul(c3[1], bc[1])), mul(c3[2], bc[2]));
                        let near = (0..3).max_by(|&a, &b| bc[a].total_cmp(&bc[b])).unwrap();
                        let g = vs.get(Key::Grid(r, n + 1_000_000 * dom[0]), p, wv[&hs.wt[tl[k]][near]], false);
                        n += 1;
                        insert(&mut cdt, q, g, &mut gid, &mut s_at, &mut rep.merged_points);
                    }
                    y += hr;
                }
            }
            for ch in &chains {
                for w in ch.windows(2) {
                    if w[0] == w[1] {
                        continue;
                    }
                    if cdt.can_add_constraint(w[0], w[1]) {
                        cdt.add_constraint(w[0], w[1]);
                    } else {
                        rep.constraints_skipped += 1;
                    }
                }
            }
            for f in cdt.inner_faces() {
                let vh = f.vertices();
                let ix = vh.map(|v| v.fix().index());
                let s3 = ix.map(|i| s_at[i]);
                let cen = [(s3[0][0] + s3[1][0] + s3[2][0]) / 3.0, (s3[0][1] + s3[1][1] + s3[2][1]) / 3.0];
                let Some(&k) = dom.iter().find(|&&k| bary(st[k], cen).iter().all(|&b| b >= -1e-9)) else { continue };
                let parent = tl[k];
                let mut v = ix.map(|i| gid[i]);
                if v[0] == v[1] || v[1] == v[2] || v[0] == v[2] {
                    continue;
                }
                let su = hair::tri_uv(model, hs, parent);
                let mut suv = s3.map(|q| {
                    let b = bary(st[k], q);
                    [b[0] * su[0][0] + b[1] * su[1][0] + b[2] * su[2][0], b[0] * su[0][1] + b[1] * su[1][1] + b[2] * su[2][1]]
                });
                let pu = pr.uv[parent];
                let mut puv = s3.map(|q| {
                    let b = bary(st[k], q);
                    [b[0] * pu[0][0] + b[1] * pu[1][0] + b[2] * pu[2][0], b[0] * pu[0][1] + b[1] * pu[1][1] + b[2] * pu[2][1]]
                });
                let n = cross(sub(vs.pos[v[1]], vs.pos[v[0]]), sub(vs.pos[v[2]], vs.pos[v[0]]));
                if dot(n, hs.normal(parent)) < 0.0 {
                    v.swap(1, 2);
                    suv.swap(1, 2);
                    puv.swap(1, 2);
                }
                out.push(HairTri { v, suv, puv, parent, region: r, back: false });
            }
        }
    }
    let tri_area = |vs: &Verts, t: &HairTri| 0.5 * len(cross(sub(vs.pos[t.v[1]], vs.pos[t.v[0]]), sub(vs.pos[t.v[2]], vs.pos[t.v[0]])));
    rep.area_before_trim = out.iter().map(|t| tri_area(&vs, t)).sum();

    // Trim to the stock alpha outline.
    if o.alpha_cut > 0.0 {
        let mut alpha = vec![f64::NAN; vs.pos.len()];
        for t in &out {
            for c in 0..3 {
                if alpha[t.v[c]].is_nan() {
                    alpha[t.v[c]] = if vs.protect[t.v[c]] { 1.0 } else { stock.bilinear(t.suv[c][0], t.suv[c][1])[3] / 255.0 };
                }
            }
        }
        let f = |g: usize, alpha: &[f64]| {
            let x = alpha[g] - o.alpha_cut;
            if x.abs() < 1e-3 {
                0.0
            } else {
                x
            }
        };
        let mut trimmed = Vec::with_capacity(out.len());
        for t in out {
            let fs = t.v.map(|g| f(g, &alpha));
            if fs.iter().all(|&x| x >= 0.0) {
                trimmed.push(t);
                continue;
            }
            if fs.iter().all(|&x| x <= 0.0) {
                continue;
            }
            let mut poly: Vec<(usize, [f64; 2], [f64; 2])> = Vec::new();
            for c in 0..3 {
                let d = (c + 1) % 3;
                if fs[c] >= 0.0 {
                    poly.push((t.v[c], t.suv[c], t.puv[c]));
                }
                if (fs[c] > 0.0 && fs[d] < 0.0) || (fs[c] < 0.0 && fs[d] > 0.0) {
                    let (ga, gb) = (t.v[c].min(t.v[d]), t.v[c].max(t.v[d]));
                    let tt = f(ga, &alpha) / (f(ga, &alpha) - f(gb, &alpha));
                    let p = lerp3(vs.pos[ga], vs.pos[gb], tt);
                    let src = if tt < 0.5 { vs.src[ga] } else { vs.src[gb] };
                    let g = vs.get(Key::Cut(ga, gb), p, src, false);
                    let tl_ = if t.v[c] == ga { tt } else { 1.0 - tt };
                    poly.push((g, lerp2(t.suv[c], t.suv[d], tl_), lerp2(t.puv[c], t.puv[d], tl_)));
                }
            }
            for k in 1..poly.len().saturating_sub(1) {
                trimmed.push(HairTri { v: [poly[0].0, poly[k].0, poly[k + 1].0], suv: [poly[0].1, poly[k].1, poly[k + 1].1], puv: [poly[0].2, poly[k].2, poly[k + 1].2], parent: t.parent, region: t.region, back: false });
            }
        }
        out = trimmed;
    }
    out.retain(|t| tri_area(&vs, t) > 1e-10);
    rep.area_after_trim = out.iter().map(|t| tri_area(&vs, t)).sum();

    // Close small gaps: free boundary vertices onto the nearest card of another lock.
    {
        let mut e_n: HashMap<(usize, usize), usize> = HashMap::new();
        for t in &out {
            for c in 0..3 {
                *e_n.entry(ekey(t.v[c], t.v[(c + 1) % 3])).or_default() += 1;
            }
        }
        let mut locks_of: Vec<Vec<usize>> = vec![Vec::new(); vs.pos.len()];
        for t in &out {
            for &g in &t.v {
                let l = hs.lock[t.parent];
                if !locks_of[g].contains(&l) {
                    locks_of[g].push(l);
                }
            }
        }
        let mut bnd = vec![false; vs.pos.len()];
        for (e, &n) in &e_n {
            if n == 1 {
                bnd[e.0] = true;
                bnd[e.1] = true;
            }
        }
        let mut moves: Vec<(usize, V3)> = Vec::new();
        let edges = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0];
        for g in 0..vs.pos.len() {
            if !bnd[g] || vs.protect[g] || locks_of[g].is_empty() {
                continue;
            }
            let p = vs.pos[g];
            let mut best = (f64::MAX, p);
            for t in &out {
                if locks_of[g].contains(&hs.lock[t.parent]) {
                    continue;
                }
                let q = closest_on_tri(p, vs.pos[t.v[0]], vs.pos[t.v[1]], vs.pos[t.v[2]]);
                let d = len(sub(q, p));
                if d < best.0 {
                    best = (d, q);
                }
            }
            if best.0 < f64::MAX {
                let rel = best.0 / h;
                let bin = edges.iter().rposition(|&e| rel >= e).unwrap_or(0);
                rep.gap_hist[bin] += 1;
            }
            if best.0 > 1e-7 && best.0 <= o.gap * h {
                moves.push((g, best.1));
            }
        }
        for (g, q) in moves {
            rep.gap_max_move = rep.gap_max_move.max(len(sub(q, vs.pos[g])));
            vs.pos[g] = q;
            rep.gap_vertices += 1;
        }
    }

    // Back faces.
    rep.front = out.len();
    let mut backs = Vec::new();
    for t in &out {
        if o.backfaces == Backfaces::All || (o.backfaces == Backfaces::Stock && has_back[t.parent]) {
            backs.push(HairTri { v: [t.v[0], t.v[2], t.v[1]], suv: [t.suv[0], t.suv[2], t.suv[1]], puv: [t.puv[0], t.puv[2], t.puv[1]], parent: t.parent, region: t.region, back: true });
        }
    }
    rep.back = backs.len();
    out.extend(backs);
    // Drop unused vertices.
    let mut used = vec![usize::MAX; vs.pos.len()];
    let (mut pos, mut src) = (Vec::new(), Vec::new());
    for t in out.iter_mut() {
        for g in t.v.iter_mut() {
            if used[*g] == usize::MAX {
                used[*g] = pos.len();
                pos.push(vs.pos[*g]);
                src.push(vs.src[*g]);
            }
            *g = used[*g];
        }
    }
    rep.vertices = pos.len();
    Ok(HairModel { pos, src, tris: out, report: rep })
}

/// Strand coordinates on the rebuilt hair, carried over from the stock solve (each new corner
/// interpolates its parent's (u, v)), so the rebuilt mesh keeps the stock joins instead of
/// re-solving u on a finer mesh. Corners follow the written mesh triangle order.
pub fn transfer_param(hs_new: &HairSet, hm: &HairModel, hm_of: &[usize], fl_new: &Flow) -> hair::Param {
    let uv: Vec<[[f64; 2]; 3]> = hs_new.tris.iter().map(|&t| hm.tris[hm_of[t]].puv).collect();
    let sg = hair::orient(hs_new);
    hair::finish_param(hs_new, fl_new, &sg, uv)
}

/// Flow on the rebuilt hair: each new triangle takes its parent's flow, laid into its own plane.
/// `hm_of[t]` is the `HairModel` triangle of mesh triangle `t` (from `write`). Regions follow the
/// parents, split again where trimming cut a region into separate locks.
pub fn transfer_flow(hs_new: &HairSet, hm: &HairModel, hm_of: &[usize], fl: &Flow, smooth: usize) -> Flow {
    let n = hs_new.tris.len();
    let mut dir = vec![[0.0; 3]; n];
    let mut away = vec![[0.0; 3]; n];
    let mut coherence = vec![0.0; n];
    let mut ambiguous = vec![false; n];
    let mut region = vec![0usize; n];
    let mut key_of: HashMap<(usize, usize), usize> = HashMap::new();
    let mut region_lock = Vec::new();
    for i in 0..n {
        let p = hm.tris[hm_of[hs_new.tris[i]]].parent;
        let nn = hs_new.normal(i);
        let lay = |d: V3| {
            let q = hair::project_plane(d, nn);
            if len(q) > 1e-9 {
                hair::unit(q)
            } else {
                [0.0; 3]
            }
        };
        dir[i] = lay(fl.dir[p]);
        away[i] = lay(fl.away[p]);
        coherence[i] = fl.coherence[p];
        ambiguous[i] = fl.ambiguous[p];
        let k = (fl.region[p], hs_new.lock[i]);
        let nr = key_of.len();
        region[i] = *key_of.entry(k).or_insert_with(|| {
            region_lock.push(hs_new.lock[i]);
            nr
        });
    }
    // Smooth within regions: parents' flows are constant per parent and jump at parent edges.
    let nb = hair::neighbors(hs_new);
    for _ in 0..smooth {
        let prev = dir.clone();
        for i in 0..n {
            if !hs_new.is_primary(i) || len(prev[i]) < 0.5 {
                continue;
            }
            let mut acc = mul(prev[i], hs_new.area(i));
            for &(j, e) in &nb[i] {
                if region[j] == region[i] && len(prev[j]) > 0.5 {
                    acc = add(acc, mul(hair::hinge(hs_new, j, i, e, prev[j]), hs_new.area(j)));
                }
            }
            let q = hair::project_plane(acc, hs_new.normal(i));
            if len(q) > 1e-12 {
                dir[i] = hair::unit(q);
            }
        }
    }
    for i in 0..n {
        if !hs_new.is_primary(i) {
            dir[i] = dir[hs_new.primary[i]];
        }
    }
    // Parting edges: welded edges between primaries of different regions.
    let mut uses: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for i in 0..n {
        if hs_new.is_primary(i) {
            let t = hs_new.wt[i];
            for c in 0..3 {
                uses.entry(ekey(t[c], t[(c + 1) % 3])).or_default().push(i);
            }
        }
    }
    let partings = uses.values().filter(|u| u.len() == 2 && region[u[0]] != region[u[1]]).count();
    Flow { dir, away, coherence, ambiguous, region, region_lock, partings }
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

fn push_accessor(json: &mut Value, a: Value) -> usize {
    let list = json["accessors"].as_array_mut().expect("accessors");
    list.push(a);
    list.len() - 1
}

/// Writes `model` with the selected hair triangles of mesh `hs.mesh` replaced by `hm`. Other
/// triangles and meshes keep their vertices exactly. Every hair triangle gets its own three
/// vertices (position, flat normal, stock UV); skin, color and other attributes copy from the
/// stock vertex in `hm.src`. Returns the GLB and, for the hair mesh, a mask of the new hair
/// triangles plus the `HairModel` triangle of each mesh triangle (`usize::MAX` = not hair).
pub fn write(model: &Model, hs: &HairSet, hm: &HairModel) -> Result<(Vec<u8>, Vec<bool>, Vec<usize>), String> {
    let glb = &model.glb;
    let mut json = glb.json.clone();
    let mut bin = glb.bin().to_vec();
    let mut prims = Vec::new();
    for (mi, me) in glb.list("meshes").iter().enumerate() {
        for pi in 0..me.get("primitives").and_then(Value::as_array).map(|p| p.len()).unwrap_or(0) {
            prims.push((mi, pi));
        }
    }
    if prims.len() != model.meshes.len() {
        return Err(format!("{} primitives, {} meshes", prims.len(), model.meshes.len()));
    }
    let (mi, pi) = prims[hs.mesh];
    let m = &model.meshes[hs.mesh];
    let mut is_hair = vec![false; m.tris.len()];
    for &t in &hs.tris {
        is_hair[t] = true;
    }
    let n0 = m.pos.len();
    let mut src: Vec<u32> = (0..n0 as u32).collect();
    let mut pos: Vec<[f32; 3]> = m.pos.clone();
    let mut uv: Vec<[f32; 2]> = m.uv.clone();
    let mut nrm: Vec<Option<[f32; 3]>> = vec![None; n0];
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut mask = Vec::new();
    let mut hm_of = Vec::new();
    for (ti, t) in m.tris.iter().enumerate() {
        if !is_hair[ti] {
            tris.push(*t);
            mask.push(false);
            hm_of.push(usize::MAX);
        }
    }
    for (k, t) in hm.tris.iter().enumerate() {
        let p = t.v.map(|g| hm.pos[g]);
        let n = hair::unit(cross(sub(p[1], p[0]), sub(p[2], p[0])));
        let base = src.len() as u32;
        for c in 0..3 {
            src.push(hm.src[t.v[c]]);
            pos.push([p[c][0] as f32, p[c][1] as f32, p[c][2] as f32]);
            uv.push([t.suv[c][0] as f32, t.suv[c][1] as f32]);
            nrm.push(Some([n[0] as f32, n[1] as f32, n[2] as f32]));
        }
        tris.push([base, base + 1, base + 2]);
        mask.push(true);
        hm_of.push(k);
    }
    let attrs: Vec<(String, usize)> = json["meshes"][mi]["primitives"][pi]["attributes"]
        .as_object()
        .ok_or("primitive without attributes")?
        .iter()
        .map(|(n, a)| (n.clone(), a.as_u64().unwrap_or(0) as usize))
        .collect();
    let f32s = |v: &[[f32; 3]]| -> Vec<u8> { v.iter().flat_map(|q| q.iter().flat_map(|x| x.to_le_bytes())).collect() };
    for (name, ai) in attrs {
        let a = json["accessors"][ai].clone();
        if a.get("sparse").is_some() {
            return Err(format!("{name}: sparse accessors are not supported"));
        }
        let ct = a["componentType"].as_u64().unwrap_or(0);
        let nc = match a["type"].as_str().unwrap_or("") {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            t => return Err(format!("accessor type {t}")),
        };
        let cs = match ct {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            5125 | 5126 => 4,
            _ => return Err(format!("componentType {ct}")),
        };
        let elem = cs * nc;
        let view = a["bufferView"].as_u64().ok_or("accessor without bufferView")? as usize;
        let vb = glb.view_bytes(view)?;
        let stride = glb.list("bufferViews")[view].get("byteStride").and_then(Value::as_u64).map(|s| s as usize).unwrap_or(elem);
        let off = a.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let copy = |s: u32| -> Result<Vec<u8>, String> {
            let o = off + s as usize * stride;
            Ok(vb.get(o..o + elem).ok_or(format!("{name} reads past its view"))?.to_vec())
        };
        let mut minmax: Option<([f32; 3], [f32; 3])> = None;
        let bytes: Vec<u8> = match name.as_str() {
            "POSITION" if ct == 5126 && nc == 3 => {
                let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
                for p in &pos {
                    for c in 0..3 {
                        lo[c] = lo[c].min(p[c]);
                        hi[c] = hi[c].max(p[c]);
                    }
                }
                minmax = Some((lo, hi));
                f32s(&pos)
            }
            "TEXCOORD_0" if ct == 5126 && nc == 2 => uv.iter().flat_map(|u| u[0].to_le_bytes().into_iter().chain(u[1].to_le_bytes())).collect(),
            "NORMAL" if ct == 5126 && nc == 3 => {
                let mut out = Vec::with_capacity(src.len() * 12);
                for (k, &s) in src.iter().enumerate() {
                    match nrm[k] {
                        Some(n) => out.extend(n.iter().flat_map(|x| x.to_le_bytes())),
                        None => out.extend(copy(s)?),
                    }
                }
                out
            }
            "POSITION" | "TEXCOORD_0" | "NORMAL" => return Err(format!("{name} must be float")),
            _ => {
                let mut out = Vec::with_capacity(src.len() * elem);
                for &s in &src {
                    out.extend(copy(s)?);
                }
                out
            }
        };
        let nv = push_view(&mut json, &mut bin, &bytes, 34962);
        let mut na = a.clone();
        let ob = na.as_object_mut().unwrap();
        ob.insert("bufferView".into(), json!(nv));
        ob.remove("byteOffset");
        ob.insert("count".into(), json!(src.len()));
        if let Some((lo, hi)) = minmax {
            ob.insert("min".into(), json!(lo));
            ob.insert("max".into(), json!(hi));
        } else if name == "TEXCOORD_0" {
            let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
            for u in &uv {
                for c in 0..2 {
                    lo[c] = lo[c].min(u[c]);
                    hi[c] = hi[c].max(u[c]);
                }
            }
            ob.insert("min".into(), json!(lo));
            ob.insert("max".into(), json!(hi));
        } else {
            ob.remove("min");
            ob.remove("max");
        }
        let na_i = push_accessor(&mut json, na);
        json["meshes"][mi]["primitives"][pi]["attributes"][&name] = json!(na_i);
    }
    let wide = src.len() > 65535;
    let ib: Vec<u8> = if wide { tris.iter().flatten().flat_map(|v| v.to_le_bytes()).collect() } else { tris.iter().flatten().flat_map(|&v| (v as u16).to_le_bytes()).collect() };
    let view = push_view(&mut json, &mut bin, &ib, 34963);
    let ia = push_accessor(&mut json, json!({ "bufferView": view, "componentType": if wide { 5125 } else { 5123 }, "count": tris.len() * 3, "type": "SCALAR" }));
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
    Ok((Glb { version: glb.version, chunks, json }.to_bytes(), mask, hm_of))
}
