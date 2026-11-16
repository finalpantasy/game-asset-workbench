//! Rough edits for people who are not 3D artists: soft triangle selections, welded topology,
//! edge loops, sculpt brushes that move bind-pose vertices, texel helpers for brushes on the model,
//! and writing a GLB whose vertices moved. Nothing here is specific to one game.
//!
//! Vertices are welded by position across every mesh, so a UV seam (two vertices at one place) or
//! a card made of back-to-back faces moves as one surface and never cracks. Unmoved vertices keep
//! their exact bits: a group stores its offset from rest, and each vertex adds that offset to its own
//! position.
use crate::geom::{add, closest_on_tri, cross, dist, dot, len, norm, ray_tri, scale, sub, V3};
use crate::glb::{Chunk, Glb, CHUNK_BIN, CHUNK_JSON};
use crate::{render, Model};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

pub const NONE: u32 = u32::MAX;

fn ekey(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

/// Welded topology over every mesh of a model.
pub struct Topo {
    /// Start of each mesh's triangles in the global numbering (as `render::tri_offsets`).
    pub offsets: Vec<usize>,
    /// Global triangle -> (mesh, triangle).
    pub tri_loc: Vec<(usize, usize)>,
    /// Per mesh, per vertex: its weld group.
    pub vgroup: Vec<Vec<u32>>,
    /// Per group: the (mesh, vertex) pairs at that place.
    pub members: Vec<Vec<(u32, u32)>>,
    /// Rest position of each group (its first member).
    pub rest: Vec<V3>,
    /// Groups joined to each group by a triangle edge.
    pub gnbr: Vec<Vec<u32>>,
    /// Welded corners of each global triangle.
    pub tri_groups: Vec<[u32; 3]>,
    /// Triangles sharing a welded edge with each triangle.
    pub tri_nbr: Vec<Vec<u32>>,
    /// Triangles using each group.
    pub group_tris: Vec<Vec<u32>>,
    /// Distinct faces (back-to-back duplicates merged), in their first triangle's winding.
    pub faces: Vec<[u32; 3]>,
    /// Per welded edge: how many distinct faces use it. 1 = a border of the surface.
    pub edge_faces: HashMap<(u32, u32), u32>,
    /// Per group / per triangle: its mirror image across x = `mirror_x`, or `NONE`.
    pub mirror_group: Vec<u32>,
    pub mirror_tri: Vec<u32>,
    pub mirror_x: f32,
    /// Model height (rest pose), the scale for tolerances and brush sizes.
    pub height: f32,
}

impl Topo {
    pub fn build(model: &Model) -> Topo {
        let (lo, hi) = render::bounds(model);
        let diag = len(sub(hi, lo)).max(1e-6);
        let height = (hi[1] - lo[1]).max(1e-6);
        let q = diag * 1e-5;
        let offsets = render::tri_offsets(model);
        let mut tri_loc = Vec::new();
        let mut map: HashMap<[i64; 3], u32> = HashMap::new();
        let mut vgroup = Vec::new();
        let mut members: Vec<Vec<(u32, u32)>> = Vec::new();
        let mut rest = Vec::new();
        for (mi, m) in model.meshes.iter().enumerate() {
            let mut vg = Vec::with_capacity(m.pos.len());
            for (vi, p) in m.pos.iter().enumerate() {
                let k = [(p[0] / q).round() as i64, (p[1] / q).round() as i64, (p[2] / q).round() as i64];
                let g = *map.entry(k).or_insert_with(|| {
                    members.push(Vec::new());
                    rest.push(*p);
                    (members.len() - 1) as u32
                });
                members[g as usize].push((mi as u32, vi as u32));
                vg.push(g);
            }
            vgroup.push(vg);
            for ti in 0..m.tris.len() {
                tri_loc.push((mi, ti));
            }
        }
        let ng = members.len();
        let nt = tri_loc.len();
        let mut tri_groups = Vec::with_capacity(nt);
        for &(mi, ti) in &tri_loc {
            let t = model.meshes[mi].tris[ti];
            tri_groups.push(t.map(|v| vgroup[mi][v as usize]));
        }
        let mut group_tris = vec![Vec::new(); ng];
        let mut edge_tris: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        let mut face_of: HashMap<[u32; 3], usize> = HashMap::new();
        let mut faces = Vec::new();
        let mut edge_faces: HashMap<(u32, u32), u32> = HashMap::new();
        for (t, g) in tri_groups.iter().enumerate() {
            for &x in g {
                if group_tris[x as usize].last() != Some(&(t as u32)) {
                    group_tris[x as usize].push(t as u32);
                }
            }
            for k in 0..3 {
                let (a, b) = (g[k], g[(k + 1) % 3]);
                if a != b {
                    edge_tris.entry(ekey(a, b)).or_default().push(t as u32);
                }
            }
            let mut s = *g;
            s.sort();
            if s[0] == s[1] || s[1] == s[2] {
                continue;
            }
            if let std::collections::hash_map::Entry::Vacant(e) = face_of.entry(s) {
                e.insert(faces.len());
                faces.push(*g);
                for k in 0..3 {
                    *edge_faces.entry(ekey(s[k], s[(k + 1) % 3])).or_insert(0) += 1;
                }
            }
        }
        let mut gnbr = vec![Vec::new(); ng];
        let mut tri_nbr = vec![Vec::new(); nt];
        for (&(a, b), ts) in &edge_tris {
            gnbr[a as usize].push(b);
            gnbr[b as usize].push(a);
            for &x in ts {
                for &y in ts {
                    if x != y {
                        tri_nbr[x as usize].push(y);
                    }
                }
            }
        }
        for v in gnbr.iter_mut().chain(tri_nbr.iter_mut()) {
            v.sort();
            v.dedup();
        }
        let mirror_x = 0.5 * (lo[0] + hi[0]);
        let tol = height * 0.003;
        let mirror_group = mirror_points(&rest, mirror_x, tol, height);
        // Triangles: the triangle whose corners are the mirrored corners, else the nearest centroid.
        let mut by_groups: HashMap<[u32; 3], Vec<u32>> = HashMap::new();
        for (t, g) in tri_groups.iter().enumerate() {
            let mut s = *g;
            s.sort();
            by_groups.entry(s).or_default().push(t as u32);
        }
        let cent: Vec<V3> = tri_groups.iter().map(|g| scale(add(add(rest[g[0] as usize], rest[g[1] as usize]), rest[g[2] as usize]), 1.0 / 3.0)).collect();
        let by_cent = mirror_points(&cent, mirror_x, tol * 2.0, height);
        let mirror_tri = (0..nt)
            .map(|t| {
                let g = tri_groups[t];
                if g.iter().all(|&x| mirror_group[x as usize] != NONE) {
                    let mut s = g.map(|x| mirror_group[x as usize]);
                    s.sort();
                    if let Some(c) = by_groups.get(&s) {
                        let mi = tri_loc[t].0;
                        return *c.iter().find(|&&o| tri_loc[o as usize].0 == mi).unwrap_or(&c[0]);
                    }
                }
                by_cent[t]
            })
            .collect();
        Topo { offsets, tri_loc, vgroup, members, rest, gnbr, tri_groups, tri_nbr, group_tris, faces, edge_faces, mirror_group, mirror_tri, mirror_x, height }
    }

    pub fn groups(&self) -> usize {
        self.rest.len()
    }
    pub fn tris(&self) -> usize {
        self.tri_loc.len()
    }
    pub fn gid(&self, mesh: usize, tri: usize) -> usize {
        self.offsets[mesh] + tri
    }
    pub fn is_border(&self, a: u32, b: u32) -> bool {
        self.edge_faces.get(&ekey(a, b)).copied().unwrap_or(0) == 1
    }
    pub fn mirror_point(&self, p: V3) -> V3 {
        [2.0 * self.mirror_x - p[0], p[1], p[2]]
    }

    /// Per-mesh vertex positions for group positions `gpos`: each vertex keeps its own rest bits
    /// plus its group's offset, so untouched vertices are exactly as stored.
    pub fn mesh_positions(&self, model: &Model, gpos: &[V3]) -> Vec<Vec<V3>> {
        model
            .meshes
            .iter()
            .enumerate()
            .map(|(mi, m)| {
                m.pos
                    .iter()
                    .enumerate()
                    .map(|(vi, &p)| {
                        let g = self.vgroup[mi][vi] as usize;
                        let d = sub(gpos[g], self.rest[g]);
                        if d == [0.0; 3] {
                            p
                        } else {
                            add(p, d)
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Area-weighted normal per group from the distinct faces (back-to-back cards count once).
    pub fn normals(&self, gpos: &[V3]) -> Vec<V3> {
        let mut n = vec![[0.0f32; 3]; self.groups()];
        for f in &self.faces {
            let [a, b, c] = f.map(|g| gpos[g as usize]);
            let fnrm = cross(sub(b, a), sub(c, a));
            for &g in f {
                n[g as usize] = add(n[g as usize], fnrm);
            }
        }
        n.into_iter().map(|v| if len(v) < 1e-12 { [0.0, 0.0, 0.0] } else { norm(v) }).collect()
    }
}

/// For each point, the index of the point nearest its mirror image (x -> 2*mx - x), within `tol`.
fn mirror_points(pts: &[V3], mx: f32, tol: f32, height: f32) -> Vec<u32> {
    let cell = (height * 0.01).max(tol * 2.0);
    let key = |p: V3| [(p[0] / cell).floor() as i32, (p[1] / cell).floor() as i32, (p[2] / cell).floor() as i32];
    let mut grid: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
    for (i, &p) in pts.iter().enumerate() {
        grid.entry(key(p)).or_default().push(i as u32);
    }
    pts.iter()
        .map(|&p| {
            let m = [2.0 * mx - p[0], p[1], p[2]];
            let k = key(m);
            let mut best = (tol, NONE);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(v) = grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                            for &j in v {
                                let d = dist(pts[j as usize], m);
                                if d <= best.0 {
                                    best = (d, j);
                                }
                            }
                        }
                    }
                }
            }
            best.1
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Selections: a weight per global triangle, 0 = not selected, 1 = fully selected.
// ---------------------------------------------------------------------------------------------

/// Adds one ring of neighbors (each takes the largest weight next to it).
pub fn grow(topo: &Topo, w: &mut [f32]) {
    let old = w.to_vec();
    for t in 0..w.len() {
        if old[t] <= 0.0 {
            let m = topo.tri_nbr[t].iter().map(|&n| old[n as usize]).fold(0.0f32, f32::max);
            if m > 0.0 {
                w[t] = m;
            }
        }
    }
}

/// Removes the outer ring (selected triangles next to an unselected one or on a border).
pub fn shrink(topo: &Topo, w: &mut [f32]) {
    let old = w.to_vec();
    for t in 0..w.len() {
        if old[t] > 0.0 && topo.tri_nbr[t].iter().any(|&n| old[n as usize] <= 0.0) {
            w[t] = 0.0;
        }
    }
}

/// Copies the selection onto the mirrored triangles.
pub fn mirror_selection(topo: &Topo, w: &mut [f32]) {
    for t in 0..w.len() {
        let m = topo.mirror_tri[t];
        if w[t] > 0.0 && m != NONE {
            w[m as usize] = w[m as usize].max(w[t]);
        }
    }
}

/// Per group: the largest weight of its triangles.
pub fn group_weights(topo: &Topo, w: &[f32]) -> Vec<f32> {
    topo.group_tris.iter().map(|ts| ts.iter().map(|&t| w[t as usize]).fold(0.0f32, f32::max)).collect()
}

pub fn falloff(d: f32, r: f32) -> f32 {
    if d >= r || r <= 0.0 {
        return 0.0;
    }
    let x = 1.0 - d / r;
    x * x * (3.0 - 2.0 * x)
}

// ---------------------------------------------------------------------------------------------
// Edge loops
// ---------------------------------------------------------------------------------------------

/// A chain of welded vertices joined by mesh edges.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeLoop {
    pub groups: Vec<u32>,
    pub closed: bool,
    /// The edges lie on a border of the surface (an open edge of a card or a hole).
    pub border: bool,
}

/// Follows the edge a-b both ways. On a border it follows the border and stops at corners sharper
/// than 65 degrees; inside a surface it takes the straightest next edge and stops at turns over 40.
pub fn follow_loop(topo: &Topo, pos: &[V3], a: u32, b: u32) -> EdgeLoop {
    let border = topo.is_border(a, b);
    let max_turn = if border { 65f32 } else { 40f32 }.to_radians();
    let extend = |path: &mut Vec<u32>| -> bool {
        loop {
            let n = path.len();
            let (prev, cur) = (path[n - 2], path[n - 1]);
            let d0 = norm(sub(pos[cur as usize], pos[prev as usize]));
            let mut best: Option<(f32, u32)> = None;
            for &c in &topo.gnbr[cur as usize] {
                if c == prev || topo.is_border(cur, c) != border {
                    continue;
                }
                let d1 = norm(sub(pos[c as usize], pos[cur as usize]));
                let ang = dot(d0, d1).clamp(-1.0, 1.0).acos();
                if best.is_none_or(|b| ang < b.0) {
                    best = Some((ang, c));
                }
            }
            let Some((ang, c)) = best else { return false };
            if ang > max_turn {
                return false;
            }
            if c == path[0] {
                return true;
            }
            if path.contains(&c) || path.len() > 4000 {
                return false;
            }
            path.push(c);
        }
    };
    let mut path = vec![a, b];
    let closed = extend(&mut path);
    if !closed {
        path.reverse();
        extend(&mut path);
        path.reverse();
    }
    EdgeLoop { groups: path, closed, border }
}

#[derive(PartialEq)]
struct Q(f32, u32);
impl Eq for Q {}
impl PartialOrd for Q {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Q {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0)
    }
}

/// Shortest chain of edges from `from` to `to` (Dijkstra on edge lengths).
pub fn shortest_path(topo: &Topo, pos: &[V3], from: u32, to: u32) -> Option<Vec<u32>> {
    let n = topo.groups();
    let mut d = vec![f32::INFINITY; n];
    let mut prev = vec![NONE; n];
    let mut h = BinaryHeap::new();
    d[from as usize] = 0.0;
    h.push(Q(0.0, from));
    while let Some(Q(c, u)) = h.pop() {
        if u == to {
            break;
        }
        if c > d[u as usize] {
            continue;
        }
        for &v in &topo.gnbr[u as usize] {
            let nd = c + dist(pos[u as usize], pos[v as usize]);
            if nd < d[v as usize] {
                d[v as usize] = nd;
                prev[v as usize] = u;
                h.push(Q(nd, v));
            }
        }
    }
    if !d[to as usize].is_finite() {
        return None;
    }
    let mut out = vec![to];
    let mut x = to;
    while x != from {
        x = prev[x as usize];
        out.push(x);
    }
    out.reverse();
    Some(out)
}

// ---------------------------------------------------------------------------------------------
// Sculpt brushes on group positions (bind pose)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sculpt {
    Grab,
    Smooth,
    Inflate,
    Deflate,
    Flatten,
}

/// Groups within `r` of `c` with their brush weight (falloff times `allow`).
pub fn in_brush(pos: &[V3], c: V3, r: f32, allow: &[f32]) -> Vec<(u32, f32)> {
    pos.iter()
        .enumerate()
        .filter_map(|(g, &p)| {
            let w = falloff(dist(p, c), r) * allow.get(g).copied().unwrap_or(1.0);
            (w > 1e-4).then_some((g as u32, w))
        })
        .collect()
}

/// One dab of a brush at `c` (radius `r`, strength 0..1). `allow` holds a weight per group
/// (1 everywhere when nothing is selected). With `sym` the dab repeats at the mirror point.
pub fn dab(topo: &Topo, pos: &mut [V3], kind: Sculpt, c: V3, r: f32, strength: f32, allow: &[f32], sym: bool) {
    let mut centers = vec![c];
    if sym {
        let m = topo.mirror_point(c);
        if dist(m, c) > r * 0.25 {
            centers.push(m);
        }
    }
    for c in centers {
        let set = in_brush(pos, c, r, allow);
        if set.is_empty() {
            continue;
        }
        let snap: Vec<V3> = pos.to_vec();
        match kind {
            Sculpt::Grab => {}
            Sculpt::Smooth => {
                for &(g, w) in &set {
                    let nb = &topo.gnbr[g as usize];
                    if nb.is_empty() {
                        continue;
                    }
                    let mut avg = [0.0f32; 3];
                    for &n in nb {
                        avg = add(avg, snap[n as usize]);
                    }
                    avg = scale(avg, 1.0 / nb.len() as f32);
                    let p = snap[g as usize];
                    pos[g as usize] = add(p, scale(sub(avg, p), (strength * w).min(1.0)));
                }
            }
            Sculpt::Inflate | Sculpt::Deflate => {
                let nrm = topo.normals(&snap);
                let sgn = if kind == Sculpt::Inflate { 1.0 } else { -1.0 };
                for &(g, w) in &set {
                    pos[g as usize] = add(snap[g as usize], scale(nrm[g as usize], sgn * strength * w * r * 0.08));
                }
            }
            Sculpt::Flatten => {
                let nrm = topo.normals(&snap);
                let (mut pc, mut nc, mut ws) = ([0.0f32; 3], [0.0f32; 3], 0.0f32);
                for &(g, w) in &set {
                    pc = add(pc, scale(snap[g as usize], w));
                    nc = add(nc, scale(nrm[g as usize], w));
                    ws += w;
                }
                if ws <= 0.0 || len(nc) < 1e-9 {
                    continue;
                }
                let (pc, nc) = (scale(pc, 1.0 / ws), norm(nc));
                for &(g, w) in &set {
                    let p = snap[g as usize];
                    let off = dot(sub(p, pc), nc);
                    pos[g as usize] = sub(p, scale(nc, off * (strength * w).min(1.0)));
                }
            }
        }
    }
}

/// A grab in progress: the groups it holds and their start positions.
#[derive(Clone)]
pub struct Grab {
    pub start: Vec<V3>,
    /// (group, weight at the pointer, weight at the mirror point).
    pub held: Vec<(u32, f32, f32)>,
}

pub fn grab_begin(topo: &Topo, pos: &[V3], c: V3, r: f32, allow: &[f32], sym: bool) -> Grab {
    let mut held: HashMap<u32, (f32, f32)> = HashMap::new();
    for (g, w) in in_brush(pos, c, r, allow) {
        held.entry(g).or_default().0 = w;
    }
    if sym {
        for (g, w) in in_brush(pos, topo.mirror_point(c), r, allow) {
            held.entry(g).or_default().1 = w;
        }
    }
    let mut held: Vec<(u32, f32, f32)> = held.into_iter().map(|(g, (a, b))| (g, a, b)).collect();
    held.sort_by_key(|h| h.0);
    Grab { start: pos.to_vec(), held }
}

/// Moves the held groups by `delta` (world units) from where the grab started; mirrored groups
/// move by the mirrored delta, and a group on the mirror plane keeps its x.
pub fn grab_apply(g: &Grab, pos: &mut [V3], delta: V3) {
    for &(i, wa, wb) in &g.held {
        let s = g.start[i as usize];
        let off = [delta[0] * (wa - wb), delta[1] * wa.max(wb), delta[2] * wa.max(wb)];
        pos[i as usize] = add(s, off);
    }
}

/// How far each group moved from rest: (moved groups, largest move).
pub fn moved(topo: &Topo, pos: &[V3]) -> (usize, f32) {
    let mut n = 0;
    let mut mx = 0.0f32;
    for (g, &p) in pos.iter().enumerate() {
        let d = dist(p, topo.rest[g]);
        if d > 0.0 {
            n += 1;
            mx = mx.max(d);
        }
    }
    (n, mx)
}

// ---------------------------------------------------------------------------------------------
// Texels under brushes
// ---------------------------------------------------------------------------------------------

/// Calls `f(x, y, bary)` for every pixel whose center lies inside triangle `p` (pixel coordinates)
/// or within `pad` pixels of it, so painting covers the island edge and mip levels do not bleed.
pub fn raster_tri_pad<F: FnMut(usize, usize, [f32; 3])>(p: [[f32; 2]; 3], w: usize, h: usize, pad: f32, mut f: F) {
    let minx = p[0][0].min(p[1][0]).min(p[2][0]) - pad;
    let maxx = p[0][0].max(p[1][0]).max(p[2][0]) + pad;
    let miny = p[0][1].min(p[1][1]).min(p[2][1]) - pad;
    let maxy = p[0][1].max(p[1][1]).max(p[2][1]) + pad;
    if !(minx.is_finite() && maxx.is_finite() && miny.is_finite() && maxy.is_finite()) || w == 0 || h == 0 {
        return;
    }
    let area = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]);
    if area.abs() < 1e-12 {
        return;
    }
    let x0 = (minx - 0.5).floor().max(0.0) as usize;
    let y0 = (miny - 0.5).floor().max(0.0) as usize;
    let x1 = ((maxx - 0.5).ceil().max(0.0) as usize).min(w - 1);
    let y1 = ((maxy - 0.5).ceil().max(0.0) as usize).min(h - 1);
    // Signed distance to each edge, positive inside.
    let el: [f32; 3] = std::array::from_fn(|k| {
        let (a, b) = (p[(k + 1) % 3], p[(k + 2) % 3]);
        ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt().max(1e-9)
    });
    for y in y0..=y1 {
        let py = y as f32 + 0.5;
        for x in x0..=x1 {
            let px = x as f32 + 0.5;
            let b1 = ((px - p[0][0]) * (p[2][1] - p[0][1]) - (py - p[0][1]) * (p[2][0] - p[0][0])) / area;
            let b2 = ((p[1][0] - p[0][0]) * (py - p[0][1]) - (p[1][1] - p[0][1]) * (px - p[0][0])) / area;
            let b = [1.0 - b1 - b2, b1, b2];
            // bary k times (2 * area / edge length k) is the distance to edge k.
            let inside = (0..3).all(|k| b[k] * area.abs() / el[k] >= -pad);
            if inside {
                let mut c = b.map(|v| v.max(0.0));
                let s = c[0] + c[1] + c[2];
                c = c.map(|v| v / s.max(1e-9));
                f(x, y, c);
            }
        }
    }
}

/// Texels of `image` (at `tw` x `th`) that lie within `r` of `c` on the surface, through the
/// triangles that sample them: `f(global_tri, x, y, point, weight)`. Painting through 3D this way
/// reaches both sides of a UV seam.
pub fn model_texels<F: FnMut(usize, usize, usize, V3, f32)>(model: &Model, topo: &Topo, mesh_pos: &[Vec<V3>], image: usize, tw: usize, th: usize, c: V3, r: f32, mut f: F) {
    for (mi, m) in model.meshes.iter().enumerate() {
        if m.image != Some(image) {
            continue;
        }
        let pos = &mesh_pos[mi];
        for (ti, t) in m.tris.iter().enumerate() {
            let [a, b, cc] = t.map(|v| pos[v as usize]);
            let (near, _) = closest_on_tri(c, a, b, cc);
            if dist(near, c) >= r {
                continue;
            }
            let uv = t.map(|v| m.uv[v as usize]);
            let px = uv.map(|u| [u[0] * tw as f32, u[1] * th as f32]);
            let gid = topo.gid(mi, ti);
            raster_tri_pad(px, tw, th, 0.75, |x, y, bc| {
                let p = add(add(scale(a, bc[0]), scale(b, bc[1])), scale(cc, bc[2]));
                let w = falloff(dist(p, c), r);
                if w > 0.0 {
                    f(gid, x, y, p, w);
                }
            });
        }
    }
}

/// Per texel weight of the selected triangles of `image` at `tw` x `th` (largest weight wins).
pub fn selection_mask(model: &Model, topo: &Topo, w: &[f32], image: usize, tw: usize, th: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; tw * th];
    for (mi, m) in model.meshes.iter().enumerate() {
        if m.image != Some(image) {
            continue;
        }
        for (ti, t) in m.tris.iter().enumerate() {
            let wt = w[topo.gid(mi, ti)];
            if wt <= 0.0 {
                continue;
            }
            let px = t.map(|v| [m.uv[v as usize][0] * tw as f32, m.uv[v as usize][1] * th as f32]);
            raster_tri_pad(px, tw, th, 0.5, |x, y, _| {
                let o = y * tw + x;
                mask[o] = mask[o].max(wt);
            });
        }
    }
    mask
}

// ---------------------------------------------------------------------------------------------
// Writing moved vertices
// ---------------------------------------------------------------------------------------------

/// Writes the model with new vertex positions (bind pose). Primitives whose positions changed get
/// a new POSITION accessor (and a new NORMAL one, recomputed where vertices moved) appended to the
/// BIN chunk; skin joints, weights, UVs, indices, images, skins and clips keep their bytes.
pub fn write_positions(model: &Model, topo: &Topo, gpos: &[V3]) -> Result<Vec<u8>, String> {
    let glb = &model.glb;
    let mut json = glb.json.clone();
    let mut bin = glb.bin().to_vec();
    let mesh_pos = topo.mesh_positions(model, gpos);
    let nrm = topo.normals(gpos);
    let moved_g: Vec<bool> = gpos.iter().zip(&topo.rest).map(|(a, b)| a != b).collect();
    let mut k = 0usize;
    let nmesh = json["meshes"].as_array().map(|a| a.len()).unwrap_or(0);
    for j in 0..nmesh {
        let nprim = json["meshes"][j]["primitives"].as_array().map(|a| a.len()).unwrap_or(0);
        for p in 0..nprim {
            let mi = k;
            k += 1;
            let m = model.meshes.get(mi).ok_or("mesh count changed")?;
            let new = &mesh_pos[mi];
            if new == &m.pos {
                continue;
            }
            // Positions.
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            let mut bytes = Vec::with_capacity(new.len() * 12);
            for v in new {
                for c in 0..3 {
                    lo[c] = lo[c].min(v[c]);
                    hi[c] = hi[c].max(v[c]);
                    bytes.extend_from_slice(&v[c].to_le_bytes());
                }
            }
            let view = push_view(&mut json, &mut bin, &bytes);
            let acc = push_accessor(&mut json, json!({"bufferView": view, "componentType": 5126, "count": new.len(), "type": "VEC3", "min": lo, "max": hi}));
            json["meshes"][j]["primitives"][p]["attributes"]["POSITION"] = json!(acc);
            // Normals, when the primitive has float ones.
            let na = json["meshes"][j]["primitives"][p]["attributes"]["NORMAL"].as_u64();
            if let Some(na) = na {
                let a = &glb.json["accessors"][na as usize];
                if a["componentType"].as_u64() == Some(5126) && a["type"].as_str() == Some("VEC3") {
                    let old = glb.accessor(na as usize)?;
                    let mut nb = Vec::with_capacity(new.len() * 12);
                    for vi in 0..new.len() {
                        let o = [old.get(vi, 0) as f32, old.get(vi, 1) as f32, old.get(vi, 2) as f32];
                        let g = topo.vgroup[mi][vi] as usize;
                        let touched = moved_g[g] || topo.gnbr[g].iter().any(|&n| moved_g[n as usize]);
                        let mut n = if touched && len(nrm[g]) > 0.5 { nrm[g] } else { o };
                        if touched && dot(n, o) < 0.0 {
                            n = scale(n, -1.0); // the back face of a card keeps facing back
                        }
                        for c in n {
                            nb.extend_from_slice(&c.to_le_bytes());
                        }
                    }
                    let view = push_view(&mut json, &mut bin, &nb);
                    let acc = push_accessor(&mut json, json!({"bufferView": view, "componentType": 5126, "count": new.len(), "type": "VEC3"}));
                    json["meshes"][j]["primitives"][p]["attributes"]["NORMAL"] = json!(acc);
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
    // Check: same triangles, UVs, joints and weights; positions exactly as asked.
    let back = Model::from_bytes(&out)?;
    if back.meshes.len() != model.meshes.len() {
        return Err("written file has a different mesh count".into());
    }
    for (mi, (a, b)) in model.meshes.iter().zip(&back.meshes).enumerate() {
        if a.tris != b.tris || a.uv != b.uv || a.joints != b.joints || a.weights != b.weights {
            return Err(format!("written mesh {} changed more than positions", a.name));
        }
        if b.pos != mesh_pos[mi] {
            return Err(format!("written mesh {} positions do not read back", a.name));
        }
    }
    Ok(out)
}

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

// ---------------------------------------------------------------------------------------------
// Inspection helpers: part walls, open edges and cracks, intersections, ray probes, locks.
// ---------------------------------------------------------------------------------------------

/// Which way a triangle of a part faces, relative to the part's main facing axis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Wall {
    Front,
    Back,
    Side,
}

/// Splits triangles `tris` into front, back and side walls. The main axis is the dominant normal
/// direction of the set (area-weighted, sign-free, so a two-sided card works); `toward` (usually
/// toward the camera) picks which side is the front. Returns (axis, wall per input triangle).
pub fn walls(topo: &Topo, pos: &[V3], tris: &[u32], toward: V3) -> (V3, Vec<Wall>) {
    let ns: Vec<(V3, f32)> = tris
        .iter()
        .map(|&t| {
            let [a, b, c] = topo.tri_groups[t as usize].map(|g| pos[g as usize]);
            let n = cross(sub(b, a), sub(c, a));
            let l = len(n);
            (if l > 1e-12 { scale(n, 1.0 / l) } else { [0.0; 3] }, l * 0.5)
        })
        .collect();
    // Power iteration on sum(area * n n^T), started from `toward`.
    let mut e = norm(toward);
    for _ in 0..32 {
        let mut m = [0.0f32; 3];
        for &(n, a) in &ns {
            m = add(m, scale(n, a * dot(n, e)));
        }
        if len(m) < 1e-12 {
            break;
        }
        e = norm(m);
    }
    if dot(e, toward) < 0.0 {
        e = scale(e, -1.0);
    }
    let w = ns.iter().map(|&(n, _)| {
        let d = dot(n, e);
        if d > 0.5 {
            Wall::Front
        } else if d < -0.5 {
            Wall::Back
        } else {
            Wall::Side
        }
    });
    (e, w.collect())
}

/// Welded edges used by only one face: the borders of each surface (open edges).
pub fn open_edges(topo: &Topo) -> Vec<(u32, u32)> {
    let mut v: Vec<(u32, u32)> = topo.edge_faces.iter().filter(|(_, &n)| n == 1).map(|(&k, _)| k).collect();
    v.sort_unstable();
    v
}

fn seg_dist(p: V3, a: V3, b: V3) -> f32 {
    let ab = sub(b, a);
    let t = (dot(sub(p, a), ab) / dot(ab, ab).max(1e-20)).clamp(0.0, 1.0);
    dist(p, add(a, scale(ab, t)))
}

/// Open edges that run along another open edge within `tol` without being welded to it: a crack
/// or gap between two surfaces that should meet (light or background shows through). Each pair is
/// (edge, other edge) with edges as welded group pairs.
pub fn cracks(topo: &Topo, pos: &[V3], tol: f32) -> Vec<((u32, u32), (u32, u32))> {
    let open = open_edges(topo);
    let mut out = Vec::new();
    for (i, &(a, b)) in open.iter().enumerate() {
        let (pa, pb) = (pos[a as usize], pos[b as usize]);
        let la = dist(pa, pb);
        if la < 1e-9 {
            continue;
        }
        for &(c, d) in &open[i + 1..] {
            if c == a || c == b || d == a || d == b {
                continue;
            }
            let (pc, pd) = (pos[c as usize], pos[d as usize]);
            // Both ends of the shorter edge lie near the longer one, so the two run together.
            let (s0, s1, l0, l1) = if la <= dist(pc, pd) { (pa, pb, pc, pd) } else { (pc, pd, pa, pb) };
            if seg_dist(s0, l0, l1) <= tol && seg_dist(s1, l0, l1) <= tol && seg_dist(scale(add(s0, s1), 0.5), l0, l1) <= tol {
                out.push(((a, b), (c, d)));
            }
        }
    }
    out
}

/// Triangles that pass through another triangle they share no corner with (self-intersections
/// after an edit, or a surface poking through clothing). Returns the pairs, global ids.
pub fn intersections(topo: &Topo, pos: &[V3]) -> Vec<(u32, u32)> {
    let n = topo.tri_groups.len();
    let tri = |t: usize| topo.tri_groups[t].map(|g| pos[g as usize]);
    let boxes: Vec<(V3, V3)> = (0..n)
        .map(|t| {
            let p = tri(t);
            let lo = [0, 1, 2].map(|k| p[0][k].min(p[1][k]).min(p[2][k]));
            let hi = [0, 1, 2].map(|k| p[0][k].max(p[1][k]).max(p[2][k]));
            (lo, hi)
        })
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| boxes[a].0[0].partial_cmp(&boxes[b].0[0]).unwrap_or(Ordering::Equal));
    // An edge of one triangle passing through the other (tested both ways).
    let edge_hits = |e: [V3; 3], f: [V3; 3]| {
        (0..3).any(|k| {
            let (p, q) = (e[k], e[(k + 1) % 3]);
            ray_tri(p, sub(q, p), f[0], f[1], f[2], 1e-5).is_some_and(|t| t < 1.0 - 1e-5)
        })
    };
    let mut out = Vec::new();
    for (i, &a) in order.iter().enumerate() {
        let (alo, ahi) = boxes[a];
        for &b in &order[i + 1..] {
            let (blo, bhi) = boxes[b];
            if blo[0] > ahi[0] {
                break;
            }
            if blo[1] > ahi[1] || bhi[1] < alo[1] || blo[2] > ahi[2] || bhi[2] < alo[2] {
                continue;
            }
            let (ga, gb) = (topo.tri_groups[a], topo.tri_groups[b]);
            if ga.iter().any(|g| gb.contains(g)) {
                continue;
            }
            let (ta, tb) = (tri(a), tri(b));
            if edge_hits(ta, tb) || edge_hits(tb, ta) {
                out.push((a.min(b) as u32, a.max(b) as u32));
            }
        }
    }
    out.sort_unstable();
    out
}

/// Every triangle a ray crosses, nearest first: (global triangle, distance along the ray). Both
/// faces count, so this finds what hides behind a seam, a crack or another surface.
pub fn ray_hits(topo: &Topo, pos: &[V3], origin: V3, dir: V3) -> Vec<(u32, f32)> {
    let d = norm(dir);
    let mut v: Vec<(u32, f32)> = (0..topo.tri_groups.len())
        .filter_map(|t| {
            let [a, b, c] = topo.tri_groups[t].map(|g| pos[g as usize]);
            ray_tri(origin, d, a, b, c, 0.0).map(|s| (t as u32, s))
        })
        .collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    v
}

/// Locked places that moved between `before` and `after` (group positions) by more than `tol`:
/// (group, distance), worst first.
pub fn lock_violations(locked: &[bool], before: &[V3], after: &[V3], tol: f32) -> Vec<(u32, f32)> {
    let mut v: Vec<(u32, f32)> = (0..locked.len().min(before.len()).min(after.len()))
        .filter(|&g| locked[g])
        .filter_map(|g| {
            let d = dist(before[g], after[g]);
            (d > tol).then_some((g as u32, d))
        })
        .collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    v
}

/// Group positions of another file with the same vertices (an AI fix of this model, say): each
/// group takes its first member's position there. None when the meshes do not line up.
pub fn group_positions_of(topo: &Topo, other: &Model) -> Option<Vec<V3>> {
    let mut out = Vec::with_capacity(topo.members.len());
    for m in &topo.members {
        let (mi, vi) = m[0];
        out.push(*other.meshes.get(mi as usize)?.pos.get(vi as usize)?);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stock() -> Option<Model> {
        Model::open(std::path::Path::new(r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding\exports\Zidane-idle.glb")).ok()
    }

    #[test]
    fn weld_mirror_and_write() {
        let Some(m) = stock() else { return };
        let t = Topo::build(&m);
        assert!(t.groups() > 1000 && t.groups() < m.meshes.iter().map(|x| x.pos.len()).sum::<usize>());
        let mirrored = t.mirror_tri.iter().filter(|&&x| x != NONE).count();
        assert!(mirrored * 10 > t.tris() * 7, "only {mirrored} of {} triangles have a mirror", t.tris());
        // No edit writes the same positions.
        let mut g = t.rest.clone();
        let out = write_positions(&m, &t, &g).unwrap();
        let back = Model::from_bytes(&out).unwrap();
        assert!(back.meshes.iter().zip(&m.meshes).all(|(a, b)| a.pos == b.pos));
        // A smooth dab moves some vertices, and writing keeps weights.
        let c = t.rest[t.faces[100][0] as usize];
        dab(&t, &mut g, Sculpt::Inflate, c, t.height * 0.05, 0.8, &vec![1.0; t.groups()], true);
        let (n, mx) = moved(&t, &g);
        assert!(n > 0 && mx > 0.0);
        let out = write_positions(&m, &t, &g).unwrap();
        let back = Model::from_bytes(&out).unwrap();
        assert_eq!(back.triangle_count(), m.triangle_count());
        // Grow then shrink returns to one triangle.
        let mut w = vec![0.0; t.tris()];
        w[500] = 1.0;
        grow(&t, &mut w);
        assert!(w.iter().filter(|&&x| x > 0.0).count() > 1);
        shrink(&t, &mut w);
        assert!(w.iter().filter(|&&x| x > 0.0).count() <= 1);
    }
}
