//! UV facts ported from the UV kit (topo.py, uv_overlap.py, guide_kit.py): welding, back-face
//! duplicates, UV islands, texel density, UV overlap, and "what will this touch".
use crate::analysis::Parts;
use crate::model::{Mesh, Model};
use std::collections::HashMap;

/// Per-mesh topology, numbered exactly like the kit's `topo.analyze`.
pub struct Topo {
    /// Per triangle: the earlier triangle with the same welded corners (a back face), if any.
    pub dup_of: Vec<Option<u32>>,
    /// Per triangle: UV island id.
    pub island: Vec<u32>,
    pub islands: usize,
    /// Per island: its mirror twin, an island with the same triangle count and the same UV
    /// points (the kit's `shares_uv_with`). One paint serves both.
    pub twin: Vec<Option<u32>>,
}

impl Topo {
    /// Triangles per island, back-face duplicates left out (the kit's `triangles` figure).
    pub fn island_sizes(&self) -> Vec<usize> {
        let mut n = vec![0; self.islands];
        for (t, &i) in self.island.iter().enumerate() {
            if self.dup_of[t].is_none() {
                n[i as usize] += 1;
            }
        }
        n
    }
}

/// numpy's `np.round(a / q)`: round half to even.
fn qk(x: f32, q: f64) -> i64 {
    (x as f64 / q).round_ties_even() as i64
}

pub fn analyze(m: &Mesh) -> Topo {
    let (pq, uq) = (1e-4, 1e-5);
    let mut gmap: HashMap<[i64; 3], u32> = HashMap::new();
    let mut umap: HashMap<[i64; 5], u32> = HashMap::new();
    let mut gid = Vec::with_capacity(m.pos.len());
    let mut uid = Vec::with_capacity(m.pos.len());
    for (p, u) in m.pos.iter().zip(&m.uv) {
        let g = [qk(p[0], pq), qk(p[1], pq), qk(p[2], pq)];
        let n = gmap.len() as u32;
        gid.push(*gmap.entry(g).or_insert(n));
        let k = [g[0], g[1], g[2], qk(u[0], uq), qk(u[1], uq)];
        let n = umap.len() as u32;
        uid.push(*umap.entry(k).or_insert(n));
    }
    let t_n = m.tris.len();
    let mut first: HashMap<[u32; 3], u32> = HashMap::new();
    let mut dup_of = vec![None; t_n];
    for (i, t) in m.tris.iter().enumerate() {
        let mut k = t.map(|v| gid[v as usize]);
        k.sort();
        match first.get(&k) {
            Some(&f) => dup_of[i] = Some(f),
            None => {
                first.insert(k, i as u32);
            }
        }
    }
    // Geometric edges in first-seen order, as the kit's defaultdict iterates them.
    let mut eidx: HashMap<(u32, u32), usize> = HashMap::new();
    let mut edges: Vec<Vec<(usize, u32, u32)>> = Vec::new();
    for (ti, t) in m.tris.iter().enumerate() {
        for c in 0..3 {
            let (a, b) = (t[c], t[(c + 1) % 3]);
            let (ga, gb) = (gid[a as usize], gid[b as usize]);
            let key = (ga.min(gb), ga.max(gb));
            let e = *eidx.entry(key).or_insert_with(|| {
                edges.push(Vec::new());
                edges.len() - 1
            });
            edges[e].push((ti, a, b));
        }
    }
    let mut par: Vec<usize> = (0..t_n).collect();
    fn find(par: &mut [usize], mut x: usize) -> usize {
        while par[x] != x {
            par[x] = par[par[x]];
            x = par[x];
        }
        x
    }
    let d = |t: usize| dup_of[t].map(|x| x as usize);
    for lst in &edges {
        for i in 0..lst.len() {
            for k in i + 1..lst.len() {
                let ((t1, a1, b1), (t2, a2, b2)) = (lst[i], lst[k]);
                if d(t1) == Some(t2) || d(t2) == Some(t1) || (d(t1).is_some() && d(t1) == d(t2)) {
                    continue;
                }
                let s1 = (uid[a1 as usize].min(uid[b1 as usize]), uid[a1 as usize].max(uid[b1 as usize]));
                let s2 = (uid[a2 as usize].min(uid[b2 as usize]), uid[a2 as usize].max(uid[b2 as usize]));
                if s1 == s2 {
                    let r2 = find(&mut par, t2);
                    let r1 = find(&mut par, t1);
                    par[r1] = r2;
                }
            }
        }
    }
    let roots: Vec<usize> = (0..t_n).map(|i| find(&mut par, i)).collect();
    let mut uniq = roots.clone();
    uniq.sort();
    uniq.dedup();
    let island: Vec<u32> = roots.iter().map(|r| uniq.binary_search(r).unwrap() as u32).collect();
    let n = uniq.len();
    // Mirror twins: same triangle count and the same UV box (back faces left out). The kit's
    // exact-point test misses the arms, whose UVs differ slightly; the box test finds them.
    let mut bbox = vec![[f32::MAX, f32::MAX, f32::MIN, f32::MIN]; n];
    let mut tris = vec![0usize; n];
    for (t, tri) in m.tris.iter().enumerate() {
        if dup_of[t].is_none() {
            let i = island[t] as usize;
            tris[i] += 1;
            for &v in tri {
                let u = m.uv[v as usize];
                bbox[i] = [bbox[i][0].min(u[0]), bbox[i][1].min(u[1]), bbox[i][2].max(u[0]), bbox[i][3].max(u[1])];
            }
        }
    }
    let same = |a: usize, b: usize| (0..4).all(|k| (bbox[a][k] - bbox[b][k]).abs() < 1e-3);
    let twin = (0..n).map(|i| (0..n).find(|&k| k != i && tris[k] == tris[i] && same(i, k)).map(|k| k as u32)).collect();
    Topo { dup_of, island, islands: n, twin }
}

fn tri_pos(m: &Mesh, t: usize) -> [[f64; 3]; 3] {
    m.tris[t].map(|v| m.pos[v as usize].map(|x| x as f64))
}

fn tri_uv(m: &Mesh, t: usize, size: f64) -> [[f64; 2]; 3] {
    m.tris[t].map(|v| [m.uv[v as usize][0] as f64 * size, m.uv[v as usize][1] as f64 * size])
}

fn world_area(p: [[f64; 3]; 3]) -> f64 {
    let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
    let c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
}

fn signed_uv_area(u: [[f64; 2]; 3]) -> f64 {
    let (d1, d2) = ([u[1][0] - u[0][0], u[1][1] - u[0][1]], [u[2][0] - u[0][0], u[2][1] - u[0][1]]);
    0.5 * (d1[0] * d2[1] - d1[1] * d2[0])
}

/// Texels per world unit for each triangle at `size` x `size` (the kit's texel-density map).
pub fn density(m: &Mesh, size: usize) -> Vec<f32> {
    (0..m.tris.len()).map(|t| (signed_uv_area(tri_uv(m, t, size as f64)).abs() / world_area(tri_pos(m, t)).max(1e-12)).sqrt() as f32).collect()
}

/// UV winding sign per triangle. Mirrored copies of a surface that reuse the same texels have the
/// opposite winding.
pub fn winding(m: &Mesh) -> Vec<i8> {
    (0..m.tris.len()).map(|t| signed_uv_area(tri_uv(m, t, 1.0)).signum() as i8).collect()
}

/// Calls `f(x, y)` for every pixel whose center is inside the triangle (pixel centers at +0.5;
/// inside when every barycentric is >= -eps; a negative eps keeps only the strict interior).
pub fn raster(p: [[f64; 2]; 3], w: usize, h: usize, eps: f64, mut f: impl FnMut(usize, usize)) {
    let [(ax, ay), (bx, by), (cx, cy)] = p.map(|q| (q[0], q[1]));
    let den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
    if den.abs() < 1e-12 {
        return;
    }
    let x0 = ((ax.min(bx).min(cx) - 0.5).floor() as i64).max(0) as usize;
    let x1 = ((ax.max(bx).max(cx) + 0.5).ceil() as i64).clamp(0, w as i64) as usize;
    let y0 = ((ay.min(by).min(cy) - 0.5).floor() as i64).max(0) as usize;
    let y1 = ((ay.max(by).max(cy) + 0.5).ceil() as i64).clamp(0, h as i64) as usize;
    for y in y0..y1 {
        let py = y as f64 + 0.5;
        for x in x0..x1 {
            let px = x as f64 + 0.5;
            let l0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den;
            let l1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den;
            if l0 >= -eps && l1 >= -eps && 1.0 - l0 - l1 >= -eps {
                f(x, y);
            }
        }
    }
}

/// How two triangles that share texels relate.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Kind {
    /// The left/right mirror copy: one paint serves both sides (expected).
    Mirrored = 1,
    /// Two different islands, not mirror copies, share paint.
    Shared = 2,
    /// One island lies on top of itself (a fold, or a cut like the waist's): paint shows twice on it.
    SelfOverlap = 3,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Mirrored => "mirrored twin",
            Kind::Shared => "shared",
            Kind::SelfOverlap => "self-overlap",
        }
    }
}

pub struct Pair {
    pub a: u32,
    pub b: u32,
    pub texels: u32,
    pub kind: Kind,
}

/// UV overlap within one mesh at `size`, as the kit's uv_overlap.py measures it: back-face
/// duplicates left out, strict interior (eps -0.02), each texel counted once per later triangle
/// that lands on an earlier one.
pub struct Overlap {
    pub size: usize,
    pub pairs: Vec<Pair>,
    /// Per texel: 0 = used once or not at all, else the worst `Kind` there.
    pub kind: Vec<u8>,
}

pub const OVERLAP_SIZE: usize = 1024;

pub fn overlap(m: &Mesh, topo: &Topo, size: usize) -> Overlap {
    let mut own = vec![u32::MAX; size * size];
    let mut kind = vec![0u8; size * size];
    let mut pairs: Vec<Pair> = Vec::new();
    let mut hits: HashMap<u32, u32> = HashMap::new();
    for t in 0..m.tris.len() {
        if topo.dup_of[t].is_some() {
            continue;
        }
        hits.clear();
        raster(tri_uv(m, t, size as f64), size, size, -0.02, |x, y| {
            let i = y * size + x;
            let o = own[i];
            if o != u32::MAX {
                *hits.entry(o).or_insert(0) += 1;
                let k = pair_kind(topo, o as usize, t);
                kind[i] = kind[i].max(k as u8);
            }
            own[i] = t as u32;
        });
        let mut hs: Vec<(u32, u32)> = hits.iter().map(|(&o, &n)| (o, n)).collect();
        hs.sort();
        pairs.extend(hs.into_iter().map(|(o, n)| Pair { a: o, b: t as u32, texels: n, kind: pair_kind(topo, o as usize, t) }));
    }
    Overlap { size, pairs, kind }
}

fn pair_kind(topo: &Topo, a: usize, b: usize) -> Kind {
    let (ia, ib) = (topo.island[a], topo.island[b]);
    if topo.twin[ia as usize] == Some(ib) {
        Kind::Mirrored
    } else if ia == ib {
        Kind::SelfOverlap
    } else {
        Kind::Shared
    }
}

/// Overlap totals per island pair (lower id first).
pub struct IslandPair {
    pub a: u32,
    pub b: u32,
    pub texels: usize,
    pub tri_pairs: usize,
    /// Largest distance between the two triangles' rest-pose centers, in world units.
    pub max_dist: f32,
    pub kind: Kind,
}

pub fn island_pairs(m: &Mesh, topo: &Topo, ov: &Overlap) -> Vec<IslandPair> {
    let cen = |t: u32| {
        let p = tri_pos(m, t as usize);
        [0, 1, 2].map(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0)
    };
    let mut out: Vec<IslandPair> = Vec::new();
    for p in &ov.pairs {
        let (ia, ib) = (topo.island[p.a as usize], topo.island[p.b as usize]);
        let (a, b) = (ia.min(ib), ia.max(ib));
        let (ca, cb) = (cen(p.a), cen(p.b));
        let dist = ((ca[0] - cb[0]).powi(2) + (ca[1] - cb[1]).powi(2) + (ca[2] - cb[2]).powi(2)).sqrt() as f32;
        match out.iter_mut().find(|e| e.a == a && e.b == b) {
            Some(e) => {
                e.texels += p.texels as usize;
                e.tri_pairs += 1;
                e.max_dist = e.max_dist.max(dist);
                e.kind = e.kind.max(p.kind);
            }
            None => out.push(IslandPair { a, b, texels: p.texels as usize, tri_pairs: 1, max_dist: dist, kind: p.kind }),
        }
    }
    out.sort_by(|x, y| y.texels.cmp(&x.texels));
    out
}

/// Per triangle: the kit parts whose texels lie inside it (strict interior), with texel counts at
/// the part map's size. A triangle that crosses a part edge lists both parts.
pub fn part_texels(model: &Model, mesh: usize, parts: &Parts) -> Vec<Vec<(u8, u32)>> {
    let m = &model.meshes[mesh];
    let Some((_, s, ids)) = parts.map_for(&model.atlas_name(mesh)) else { return vec![vec![]; m.tris.len()] };
    let s = *s;
    (0..m.tris.len())
        .map(|t| {
            let mut c = [0u32; 256];
            raster(tri_uv(m, t, s as f64), s, s, -0.02, |x, y| c[ids[y * s + x] as usize] += 1);
            (1..256).filter(|&i| c[i] > 0).map(|i| (i as u8, c[i])).collect()
        })
        .collect()
}

/// Whether a triangle's UV footprint overlaps a UV rectangle (u0 <= u1, v0 <= v1), by separating axes.
pub fn tri_hits_rect(m: &Mesh, t: usize, r: [f32; 4]) -> bool {
    let p = tri_uv(m, t, 1.0);
    let [u0, v0, u1, v1] = r.map(|x| x as f64);
    let rc = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]];
    let proj = |pts: &[[f64; 2]], ax: [f64; 2]| pts.iter().map(|q| q[0] * ax[0] + q[1] * ax[1]).fold((f64::MAX, f64::MIN), |(lo, hi), x| (lo.min(x), hi.max(x)));
    let mut axes = vec![[1.0, 0.0], [0.0, 1.0]];
    for k in 0..3 {
        let (a, b) = (p[k], p[(k + 1) % 3]);
        let n = [b[1] - a[1], a[0] - b[0]];
        if n[0] != 0.0 || n[1] != 0.0 {
            axes.push(n);
        }
    }
    axes.into_iter().all(|ax| {
        let (a, b) = (proj(&p, ax), proj(&rc, ax));
        a.0 <= b.1 && b.0 <= a.1
    })
}

/// Rest-pose center of a triangle (glTF axes: +x is the character's right).
pub fn center(m: &Mesh, t: usize) -> [f32; 3] {
    let p = tri_pos(m, t);
    [0, 1, 2].map(|k| ((p[0][k] + p[1][k] + p[2][k]) / 3.0) as f32)
}

/// One UV island with the kit's label when the islands match the stock layout.
pub struct Island {
    pub label: String,
    pub tris: usize,
    /// "right", "left" or "center": the character's side, from the island's rest-pose center.
    pub side: &'static str,
    pub twin: Option<u32>,
}

/// Everything Inspect shows about UVs, built once per file.
pub struct Facts {
    pub topo: Vec<Topo>,
    pub islands: Vec<Vec<Island>>,
    /// Texels per world unit per triangle, at `density_size[mesh]`.
    pub density: Vec<Vec<f32>>,
    pub density_size: Vec<usize>,
    pub overlap: Vec<Overlap>,
    pub pairs: Vec<Vec<IslandPair>>,
    /// Per mesh, per triangle: kit parts whose texels it shows (empty without a part map).
    pub touch: Vec<Vec<Vec<(u8, u32)>>>,
}

/// UV figures for a set of triangles.
#[derive(Default)]
pub struct Summary {
    /// (mesh, island, triangles of the set in it), largest first.
    pub islands: Vec<(usize, u32, usize)>,
    /// Lowest, median and highest texels per world unit, dead triangles left out, and the size.
    pub density: Option<(f32, f32, f32, usize)>,
    /// Texels at `OVERLAP_SIZE` that show on two surfaces, by kind (mirrored, shared, self-overlap).
    pub reuse: [usize; 3],
}

impl Facts {
    /// `sizes`: per mesh, the texture size to measure density at.
    pub fn build(model: &Model, parts: Option<&Parts>, sizes: &[usize]) -> Facts {
        let topo: Vec<Topo> = model.meshes.iter().map(analyze).collect();
        let mut islands = Vec::new();
        let mut overlap = Vec::new();
        let mut pairs = Vec::new();
        for (mi, m) in model.meshes.iter().enumerate() {
            let t = &topo[mi];
            let sizes_i = t.island_sizes();
            let labels = parts.and_then(|p| p.island_labels(&model.atlas_name(mi), &sizes_i));
            let mut xs = vec![(0f64, 0usize); t.islands];
            for ti in 0..m.tris.len() {
                if t.dup_of[ti].is_none() {
                    let e = &mut xs[t.island[ti] as usize];
                    e.0 += center(m, ti)[0] as f64;
                    e.1 += 1;
                }
            }
            islands.push(
                (0..t.islands)
                    .map(|i| {
                        let x = xs[i].0 / xs[i].1.max(1) as f64;
                        Island {
                            label: labels.as_ref().map(|l| l[i].clone()).filter(|s| !s.is_empty()).unwrap_or_else(|| format!("island {i}")),
                            tris: sizes_i[i],
                            side: if x > 0.02 { "right" } else if x < -0.02 { "left" } else { "center" },
                            twin: t.twin[i],
                        }
                    })
                    .collect(),
            );
            let ov = self::overlap(m, t, OVERLAP_SIZE);
            pairs.push(island_pairs(m, t, &ov));
            overlap.push(ov);
        }
        let density = model.meshes.iter().zip(sizes).map(|(m, &s)| density(m, s)).collect();
        let touch = (0..model.meshes.len()).map(|mi| parts.map(|p| part_texels(model, mi, p)).unwrap_or_else(|| vec![vec![]; model.meshes[mi].tris.len()])).collect();
        Facts { topo, islands, density, density_size: sizes.to_vec(), overlap, pairs, touch }
    }

    pub fn summary(&self, model: &Model, set: &dyn Fn(usize, usize) -> bool) -> Summary {
        let mut s = Summary::default();
        let mut dens: Vec<f32> = Vec::new();
        let mut size = 0;
        for (mi, m) in model.meshes.iter().enumerate() {
            let t = &self.topo[mi];
            let mut per = vec![0usize; t.islands];
            for ti in 0..m.tris.len() {
                if set(mi, ti) && t.dup_of[ti].is_none() {
                    per[t.island[ti] as usize] += 1;
                    if !crate::analysis::is_dead(model, mi, ti) {
                        dens.push(self.density[mi][ti]);
                        size = size.max(self.density_size[mi]);
                    }
                }
            }
            s.islands.extend(per.iter().enumerate().filter(|e| *e.1 > 0).map(|(i, &n)| (mi, i as u32, n)));
            for p in &self.overlap[mi].pairs {
                if set(mi, p.a as usize) || set(mi, p.b as usize) {
                    s.reuse[p.kind as usize - 1] += p.texels as usize;
                }
            }
        }
        s.islands.sort_by(|a, b| b.2.cmp(&a.2));
        if !dens.is_empty() {
            dens.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s.density = Some((dens[0], dens[dens.len() / 2], dens[dens.len() - 1], size));
        }
        s
    }

    /// "What will this touch" for a kit part: every triangle that is in the part or shows any of
    /// its texels.
    pub fn touched_by_part(&self, tri_parts: &[Vec<u8>], part: u8) -> Vec<Vec<bool>> {
        tri_parts.iter().zip(&self.touch).map(|(tp, tt)| tp.iter().zip(tt).map(|(&p, t)| p == part || t.iter().any(|e| e.0 == part)).collect()).collect()
    }

    /// "What will this touch" for a UV rectangle on one image.
    pub fn touched_by_rect(model: &Model, image: usize, r: [f32; 4]) -> Vec<Vec<bool>> {
        model.meshes.iter().map(|m| (0..m.tris.len()).map(|t| m.image == Some(image) && tri_hits_rect(m, t, r)).collect()).collect()
    }
}
