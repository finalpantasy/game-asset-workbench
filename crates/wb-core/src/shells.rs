//! Loose pieces: the edge-connected shells of each mesh, whether a shell is an open
//! zero-thickness card, how close it comes to the other shells, which bones move it, and where
//! it lies on top of another surface on screen (two layers a hair apart, like a card on a belt).
use crate::geom::{self, V3};
use crate::render::Camera;
use crate::Model;
use std::collections::HashMap;

pub struct Shell {
    /// Global shell number (all meshes).
    pub id: usize,
    pub mesh: usize,
    /// "mesh0.s3": mesh name and the shell's place in that mesh (ordered by first triangle).
    pub label: String,
    pub tris: Vec<usize>,
    pub verts: Vec<usize>,
    /// Edges (welded by position) used by one triangle only.
    pub boundary_edges: usize,
    /// Edges used by three or more triangles.
    pub nonmanifold_edges: usize,
    /// Share of triangles that have a surface of the same shell behind or in front of them.
    pub backed: f32,
    /// Median distance to that opposite surface.
    pub thickness: Option<f32>,
    /// Triangles that have a twin on the same three (welded) corners: back-to-back faces.
    pub doubled: usize,
    /// Joint and its share of the shell's total skin weight, largest first.
    pub bones: Vec<(usize, f32)>,
    pub lo: V3,
    pub hi: V3,
}

impl Shell {
    pub fn open(&self) -> bool {
        self.boundary_edges > 0
    }
    /// A card: an open piece with nothing behind most of its faces, or a piece whose opposite
    /// faces sit on top of each other (back-to-back copies, under 0.002 apart).
    pub fn zero_thickness(&self) -> bool {
        (self.open() && self.backed < 0.25) || self.doubled * 2 >= self.tris.len() || self.thickness.is_some_and(|t| t < 0.002)
    }
}

fn key(p: V3) -> (i64, i64, i64) {
    let q = |x: f32| (x as f64 * 1e5).round() as i64;
    (q(p[0]), q(p[1]), q(p[2]))
}

/// Weld ids per vertex (vertices at the same position share one id).
pub fn weld(pos: &[V3]) -> Vec<usize> {
    let mut map: HashMap<(i64, i64, i64), usize> = HashMap::new();
    pos.iter()
        .map(|&p| {
            let n = map.len();
            *map.entry(key(p)).or_insert(n)
        })
        .collect()
}

fn find_root(p: &mut [usize], mut x: usize) -> usize {
    while p[x] != x {
        p[x] = p[p[x]];
        x = p[x];
    }
    x
}

/// Edge use counts keyed by welded vertex pair, for a set of triangles.
pub fn edge_uses(tris: &[[u32; 3]], wid: &[usize], subset: &[usize]) -> HashMap<(usize, usize), usize> {
    let mut uses: HashMap<(usize, usize), usize> = HashMap::new();
    for &t in subset {
        let w = tris[t].map(|v| wid[v as usize]);
        for k in 0..3 {
            let (a, b) = (w[k], w[(k + 1) % 3]);
            if a == b {
                continue;
            }
            *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    uses
}

/// Splits every mesh into edge-connected shells (rest pose).
pub fn find(model: &Model) -> Vec<Shell> {
    let mut out = Vec::new();
    for (mi, m) in model.meshes.iter().enumerate() {
        let wid = weld(&m.pos);
        let nt = m.tris.len();
        let mut parent: Vec<usize> = (0..nt).collect();
        let mut first: HashMap<(usize, usize), usize> = HashMap::new();
        for (ti, t) in m.tris.iter().enumerate() {
            let w = t.map(|v| wid[v as usize]);
            for k in 0..3 {
                let (a, b) = (w[k], w[(k + 1) % 3]);
                if a == b {
                    continue;
                }
                let e = (a.min(b), a.max(b));
                match first.get(&e) {
                    Some(&o) => {
                        let (ra, rb) = (find_root(&mut parent, ti), find_root(&mut parent, o));
                        if ra != rb {
                            parent[ra.max(rb)] = ra.min(rb);
                        }
                    }
                    None => {
                        first.insert(e, ti);
                    }
                }
            }
        }
        let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut slot: HashMap<usize, usize> = HashMap::new();
        for ti in 0..nt {
            let r = find_root(&mut parent, ti);
            let s = *slot.entry(r).or_insert_with(|| {
                groups.push((r, Vec::new()));
                groups.len() - 1
            });
            groups[s].1.push(ti);
        }
        for (k, (_, tris)) in groups.into_iter().enumerate() {
            let mut verts: Vec<usize> = tris.iter().flat_map(|&t| m.tris[t].iter().map(|&v| v as usize)).collect();
            verts.sort_unstable();
            verts.dedup();
            let uses = edge_uses(&m.tris, &wid, &tris);
            let boundary_edges = uses.values().filter(|&&u| u == 1).count();
            let nonmanifold_edges = uses.values().filter(|&&u| u > 2).count();
            let (backed, thickness) = opposite_surface(&m.pos, &m.tris, &wid, &tris);
            let mut sets: HashMap<[usize; 3], usize> = HashMap::new();
            for &t in &tris {
                let mut w = m.tris[t].map(|v| wid[v as usize]);
                w.sort_unstable();
                *sets.entry(w).or_insert(0) += 1;
            }
            let doubled = sets.values().filter(|&&c| c > 1).sum::<usize>();
            let mut acc: HashMap<usize, f32> = HashMap::new();
            for &v in &verts {
                for c in 0..4 {
                    if m.weights[v][c] > 0.0 {
                        *acc.entry(m.joints[v][c] as usize).or_insert(0.0) += m.weights[v][c];
                    }
                }
            }
            let total: f32 = acc.values().sum::<f32>().max(1e-9);
            let mut bones: Vec<(usize, f32)> = acc.into_iter().map(|(j, w)| (j, w / total)).collect();
            bones.sort_by(|a, b| b.1.total_cmp(&a.1));
            let (lo, hi) = geom::bounds(verts.iter().map(|&v| &m.pos[v]));
            out.push(Shell { id: out.len(), mesh: mi, label: format!("{}.s{k}", m.name), tris, verts, boundary_edges, nonmanifold_edges, backed, thickness, doubled, bones, lo, hi });
        }
    }
    out
}

/// For each triangle, casts a ray both ways along its normal from its center and looks for
/// another triangle of the same shell that shares no corner with it.
fn opposite_surface(pos: &[V3], tris: &[[u32; 3]], wid: &[usize], subset: &[usize]) -> (f32, Option<f32>) {
    let mut hits = Vec::new();
    let mut considered = 0usize;
    for &t in subset {
        let [a, b, c] = tris[t].map(|v| pos[v as usize]);
        let n = geom::tri_normal(a, b, c);
        if geom::len(n) == 0.0 {
            continue;
        }
        considered += 1;
        let o = geom::scale(geom::add(geom::add(a, b), c), 1.0 / 3.0);
        let wt = tris[t].map(|v| wid[v as usize]);
        let mut best = f32::MAX;
        for &u in subset {
            if u == t {
                continue;
            }
            let wu = tris[u].map(|v| wid[v as usize]);
            if wu.iter().any(|x| wt.contains(x)) {
                continue;
            }
            let [p, q, r] = tris[u].map(|v| pos[v as usize]);
            for dir in [n, geom::scale(n, -1.0)] {
                if let Some(d) = geom::ray_tri(o, dir, p, q, r, 1e-6) {
                    best = best.min(d);
                }
            }
        }
        if best < f32::MAX {
            hits.push(best);
        }
    }
    let backed = if considered == 0 { 0.0 } else { hits.len() as f32 / considered as f32 };
    if hits.is_empty() {
        return (backed, None);
    }
    hits.sort_by(f32::total_cmp);
    (backed, Some(hits[hits.len() / 2]))
}

/// Shell id of every triangle, per mesh.
pub fn tri_shell(model: &Model, shells: &[Shell]) -> Vec<Vec<usize>> {
    let mut ts: Vec<Vec<usize>> = model.meshes.iter().map(|m| vec![usize::MAX; m.tris.len()]).collect();
    for s in shells {
        for &t in &s.tris {
            ts[s.mesh][t] = s.id;
        }
    }
    ts
}

#[derive(Clone, Copy)]
pub struct Near {
    /// Closest approach between the two shells' surfaces (vertex to triangle, both ways; 0 when
    /// an edge of either passes through the other's surface).
    pub dist: f32,
    /// Vertices of this shell within the touch distance of the other.
    pub touching: usize,
    /// Edges of this shell that pass through the other's surface (a piece tucked under another).
    pub crossing: usize,
}

/// Distances between every pair of shells for one pose. `near[a][b]` is from a's side.
pub fn nearest(model: &Model, shells: &[Shell], pose: &[Vec<V3>], touch: f32) -> Vec<Vec<Near>> {
    let n = shells.len();
    let boxes: Vec<(V3, V3)> = shells.iter().map(|s| geom::bounds(s.verts.iter().map(|&v| &pose[s.mesh][v]))).collect();
    // Per shell, the closest distance from each of its vertices to each other shell.
    let mut vert_best: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n);
    for a in shells {
        let mut per_b = vec![vec![f32::MAX; a.verts.len()]; n];
        for (vi, &v) in a.verts.iter().enumerate() {
            let p = pose[a.mesh][v];
            for b in shells {
                if b.id == a.id {
                    continue;
                }
                let best = &mut per_b[b.id][vi];
                if geom::box_dist(p, boxes[b.id].0, boxes[b.id].1) >= *best {
                    continue;
                }
                let mb = &model.meshes[b.mesh];
                for &t in &b.tris {
                    let [x, y, z] = mb.tris[t].map(|w| pose[b.mesh][w as usize]);
                    let (q, _) = geom::closest_on_tri(p, x, y, z);
                    let d = geom::dist(p, q);
                    if d < *best {
                        *best = d;
                    }
                }
            }
        }
        vert_best.push(per_b);
    }
    // Edges of each shell that pass through another shell's triangles.
    let overlap = |a: (V3, V3), b: (V3, V3)| (0..3).all(|k| a.0[k] <= b.1[k] && b.0[k] <= a.1[k]);
    let mut crossing = vec![vec![0usize; n]; n];
    for a in shells {
        let ma = &model.meshes[a.mesh];
        let mut edges: Vec<(u32, u32)> = a.tris.iter().flat_map(|&t| {
            let c = ma.tris[t];
            [(c[0].min(c[1]), c[0].max(c[1])), (c[1].min(c[2]), c[1].max(c[2])), (c[2].min(c[0]), c[2].max(c[0]))]
        }).collect();
        edges.sort_unstable();
        edges.dedup();
        for b in shells {
            if b.id == a.id || !overlap(boxes[a.id], boxes[b.id]) {
                continue;
            }
            let mb = &model.meshes[b.mesh];
            let near_tris: Vec<[V3; 3]> = b
                .tris
                .iter()
                .map(|&t| mb.tris[t].map(|w| pose[b.mesh][w as usize]))
                .filter(|p3| overlap(geom::bounds(p3.iter()), boxes[a.id]))
                .collect();
            for &(e0, e1) in &edges {
                let (p, q) = (pose[a.mesh][e0 as usize], pose[a.mesh][e1 as usize]);
                let eb = geom::bounds([p, q].iter());
                let d = geom::sub(q, p);
                if near_tris.iter().any(|p3| overlap(geom::bounds(p3.iter()), eb) && geom::ray_tri(p, d, p3[0], p3[1], p3[2], 0.0).is_some_and(|t| t < 1.0)) {
                    crossing[a.id][b.id] += 1;
                }
            }
        }
    }
    let mut out = vec![vec![Near { dist: f32::MAX, touching: 0, crossing: 0 }; n]; n];
    for a in 0..n {
        for b in 0..n {
            if a == b {
                continue;
            }
            let da = vert_best[a][b].iter().copied().fold(f32::MAX, f32::min);
            let db = vert_best[b][a].iter().copied().fold(f32::MAX, f32::min);
            let dist = if crossing[a][b] + crossing[b][a] > 0 { 0.0 } else { da.min(db) };
            out[a][b] = Near { dist, touching: vert_best[a][b].iter().filter(|&&d| d <= touch).count(), crossing: crossing[a][b] };
        }
    }
    out
}

/// Camera basis exactly as `render` builds it: eye, forward, right, up.
pub fn basis(cam: &Camera) -> (V3, V3, V3, V3) {
    let (cp, sp, cy, sy) = (cam.pitch.cos(), cam.pitch.sin(), cam.yaw.cos(), cam.yaw.sin());
    let eye = [cam.target[0] + cam.dist * cp * sy, cam.target[1] + cam.dist * sp, cam.target[2] + cam.dist * cp * cy];
    let f = geom::norm(geom::sub(cam.target, eye));
    let r = geom::norm(geom::cross(f, [0.0, 1.0, 0.0]));
    let u = geom::cross(r, f);
    (eye, f, r, u)
}

/// Projects a point the way `render` does: screen x, y and a depth where larger is nearer.
pub fn project(cam: &Camera, w: usize, h: usize, p: V3) -> [f32; 3] {
    let (eye, f, r, u) = basis(cam);
    let d = geom::sub(p, eye);
    let z = geom::dot(d, f);
    let (cx, cy) = (0.5 * w as f32, 0.5 * h as f32);
    if cam.ortho {
        let s = 0.5 * h as f32 / cam.ortho_half.max(1e-6);
        [cx + s * geom::dot(d, r), cy - s * geom::dot(d, u), -z]
    } else {
        let focal = 0.5 * h as f32 / (cam.fov_y * 0.5).tan();
        [cx + focal * geom::dot(d, r) / z, cy - focal * geom::dot(d, u) / z, -z]
    }
}

/// The two nearest surfaces of different shells at each pixel.
pub struct Layers {
    pub w: usize,
    pub h: usize,
    /// Nearest surface: world depth (larger is nearer; distance along the view is -depth) and shell.
    pub d1: Vec<f32>,
    pub s1: Vec<usize>,
    /// Nearest surface of any other shell behind it.
    pub d2: Vec<f32>,
    pub s2: Vec<usize>,
}

pub const NONE: usize = usize::MAX;

/// Rasterizes depth only, keeping the nearest layer and the nearest layer of a different shell.
/// `shell_of(mesh, tri)` gives the shell id (or `NONE` to skip the triangle).
pub fn layers(model: &Model, pose: &[Vec<V3>], shell_of: &dyn Fn(usize, usize) -> usize, cam: &Camera, w: usize, h: usize) -> Layers {
    layers_with(model, None, pose, shell_of, cam, w, h)
}

/// Like `layers`, but a fragment whose base color texel is transparent (alpha under 128) is not a
/// surface: what a viewer sees with the game's alpha cutout, not where the triangles are.
/// `textures` is indexed by image, like `render::render`.
pub fn layers_alpha(model: &Model, textures: &[Option<crate::Texture>], pose: &[Vec<V3>], shell_of: &dyn Fn(usize, usize) -> usize, cam: &Camera, w: usize, h: usize) -> Layers {
    layers_with(model, Some(textures), pose, shell_of, cam, w, h)
}

fn layers_with(model: &Model, textures: Option<&[Option<crate::Texture>]>, pose: &[Vec<V3>], shell_of: &dyn Fn(usize, usize) -> usize, cam: &Camera, w: usize, h: usize) -> Layers {
    let n = w * h;
    let mut l = Layers { w, h, d1: vec![f32::MIN; n], s1: vec![NONE; n], d2: vec![f32::MIN; n], s2: vec![NONE; n] };
    let (eye, f, _, _) = basis(cam);
    for (mi, m) in model.meshes.iter().enumerate() {
        let proj: Vec<[f32; 3]> = pose[mi].iter().map(|&p| project(cam, w, h, p)).collect();
        // Perspective depth: interpolate 1/z, report -z. Orthographic: -z is linear on screen.
        let viewz: Vec<f32> = pose[mi].iter().map(|&p| geom::dot(geom::sub(p, eye), f)).collect();
        let tex = textures.and_then(|ts| m.image.and_then(|i| ts.get(i)).and_then(|t| t.as_ref()));
        for (ti, t) in m.tris.iter().enumerate() {
            let s = shell_of(mi, ti);
            if s == NONE {
                continue;
            }
            let [a, b, c] = t.map(|v| proj[v as usize]);
            let z3 = t.map(|v| viewz[v as usize]);
            if !cam.ortho && z3.iter().any(|&z| z < 1e-3) {
                continue;
            }
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            if area.abs() < 1e-9 {
                continue;
            }
            let x0 = a[0].min(b[0]).min(c[0]).floor().max(0.0) as usize;
            let x1 = (a[0].max(b[0]).max(c[0]).ceil().min(w as f32 - 1.0)).max(0.0) as usize;
            let y0 = a[1].min(b[1]).min(c[1]).floor().max(0.0) as usize;
            let y1 = (a[1].max(b[1]).max(c[1]).ceil().min(h as f32 - 1.0)).max(0.0) as usize;
            if x0 > x1 || y0 > y1 {
                continue;
            }
            for y in y0..=y1 {
                let py = y as f32 + 0.5;
                for x in x0..=x1 {
                    let px = x as f32 + 0.5;
                    let w0 = ((b[0] - px) * (c[1] - py) - (b[1] - py) * (c[0] - px)) / area;
                    let w1 = ((c[0] - px) * (a[1] - py) - (c[1] - py) * (a[0] - px)) / area;
                    let w2 = 1.0 - w0 - w1;
                    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                        continue;
                    }
                    let d = if cam.ortho {
                        -(w0 * z3[0] + w1 * z3[1] + w2 * z3[2])
                    } else {
                        -1.0 / (w0 / z3[0] + w1 / z3[1] + w2 / z3[2])
                    };
                    if let Some(tx) = tex {
                        // Perspective-correct barycentrics for the texture lookup.
                        let b = if cam.ortho {
                            [w0, w1, w2]
                        } else {
                            let q = [w0 / z3[0], w1 / z3[1], w2 / z3[2]];
                            let s = q[0] + q[1] + q[2];
                            [q[0] / s, q[1] / s, q[2] / s]
                        };
                        let uv = t.map(|v| m.uv[v as usize]);
                        let (u, v) = (b[0] * uv[0][0] + b[1] * uv[1][0] + b[2] * uv[2][0], b[0] * uv[0][1] + b[1] * uv[1][1] + b[2] * uv[2][1]);
                        let tx_x = ((u * tx.w as f32) as i64).rem_euclid(tx.w as i64) as usize;
                        let tx_y = ((v * tx.h as f32) as i64).rem_euclid(tx.h as i64) as usize;
                        if tx.rgba[(tx_y * tx.w + tx_x) * 4 + 3] < 128 {
                            continue;
                        }
                    }
                    let i = y * w + x;
                    if s == l.s1[i] {
                        if d > l.d1[i] {
                            l.d1[i] = d;
                        }
                    } else if d > l.d1[i] {
                        l.d2[i] = l.d1[i];
                        l.s2[i] = l.s1[i];
                        l.d1[i] = d;
                        l.s1[i] = s;
                    } else if s == l.s2[i] {
                        if d > l.d2[i] {
                            l.d2[i] = d;
                        }
                    } else if d > l.d2[i] {
                        l.d2[i] = d;
                        l.s2[i] = s;
                    }
                }
            }
        }
    }
    l
}

/// Per shell: pixels where it is the nearest surface, and per other shell, pixels where it lies
/// on top of that shell with less than `tol` world units between them.
pub fn overlap(l: &Layers, nshells: usize, tol: f32) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut visible = vec![0usize; nshells];
    let mut over = vec![vec![0usize; nshells]; nshells];
    for i in 0..l.w * l.h {
        let a = l.s1[i];
        if a == NONE {
            continue;
        }
        visible[a] += 1;
        let b = l.s2[i];
        if b != NONE && l.d1[i] - l.d2[i] < tol {
            over[a][b] += 1;
        }
    }
    (visible, over)
}

/// The four orthographic views the shell report uses: front, right side, back, left side
/// (yaw in degrees from the front, the same sense as `render --yaw`).
pub const VIEWS: [(&str, f32); 4] = [("front", 0.0), ("right", -90.0), ("back", 180.0), ("left", 90.0)];

/// An orthographic camera looking at `target` from `yaw_deg` (0 = front), `half` world units high.
pub fn ortho_cam(target: V3, yaw_deg: f32, half: f32) -> Camera {
    Camera { yaw: std::f32::consts::PI + yaw_deg.to_radians(), pitch: 0.0, dist: 20.0, target, fov_y: 30f32.to_radians(), ortho: true, ortho_half: half }
}

/// Skin weights of a mesh at a point inside a triangle (barycentric blend), as (joint, weight).
pub fn weights_at(m: &crate::model::Mesh, tri: usize, bary: [f32; 3]) -> Vec<(usize, f32)> {
    let mut acc: Vec<(usize, f32)> = Vec::with_capacity(8);
    for (k, &v) in m.tris[tri].iter().enumerate() {
        for c in 0..4 {
            let w = m.weights[v as usize][c] * bary[k];
            if w <= 0.0 {
                continue;
            }
            let j = m.joints[v as usize][c] as usize;
            match acc.iter_mut().find(|e| e.0 == j) {
                Some(e) => e.1 += w,
                None => acc.push((j, w)),
            }
        }
    }
    let total: f32 = acc.iter().map(|e| e.1).sum::<f32>().max(1e-9);
    acc.iter_mut().for_each(|e| e.1 /= total);
    acc
}

/// Skin weights of one vertex, normalized, as (joint, weight).
pub fn vertex_weights(m: &crate::model::Mesh, v: usize) -> Vec<(usize, f32)> {
    weights_at_vertex(&m.joints[v], &m.weights[v])
}

pub fn weights_at_vertex(j: &[u16; 4], w: &[f32; 4]) -> Vec<(usize, f32)> {
    let mut acc: Vec<(usize, f32)> = Vec::with_capacity(4);
    for c in 0..4 {
        if w[c] <= 0.0 {
            continue;
        }
        match acc.iter_mut().find(|e| e.0 == j[c] as usize) {
            Some(e) => e.1 += w[c],
            None => acc.push((j[c] as usize, w[c])),
        }
    }
    let total: f32 = acc.iter().map(|e| e.1).sum::<f32>().max(1e-9);
    acc.iter_mut().for_each(|e| e.1 /= total);
    acc
}

/// Half the L1 distance between two weight sets: 0 = they move the same, 1 = no bone in common.
pub fn weight_mismatch(a: &[(usize, f32)], b: &[(usize, f32)]) -> f32 {
    let mut s = 0.0;
    for &(j, w) in a {
        s += (w - b.iter().find(|e| e.0 == j).map_or(0.0, |e| e.1)).abs();
    }
    for &(j, w) in b {
        if !a.iter().any(|e| e.0 == j) {
            s += w;
        }
    }
    0.5 * s
}

/// Closest point of shell `b` to `p`: (distance, triangle, barycentric).
pub fn closest_in_shell(model: &Model, pose: &[Vec<V3>], b: &Shell, p: V3) -> (f32, usize, [f32; 3]) {
    let mb = &model.meshes[b.mesh];
    let mut best = (f32::MAX, 0usize, [1.0, 0.0, 0.0]);
    for &t in &b.tris {
        let [x, y, z] = mb.tris[t].map(|w| pose[b.mesh][w as usize]);
        let (q, bary) = geom::closest_on_tri(p, x, y, z);
        let d = geom::dist(p, q);
        if d < best.0 {
            best = (d, t, bary);
        }
    }
    best
}

pub struct Contact {
    pub other: usize,
    /// Vertices of the shell within reach of the other shell.
    pub verts: usize,
    /// Mean weight mismatch between those vertices and the other shell's surface under them.
    pub skin_mismatch: f32,
    /// Largest weight mismatch among them.
    pub skin_mismatch_max: f32,
}

/// How the vertices of shell `a` that lie within `reach` of shell `b` are skinned compared with b's
/// surface right under them. A piece that rides on a surface without sliding has a mismatch near 0.
pub fn skin_contact(model: &Model, shells: &[Shell], a: usize, b: usize, reach: f32) -> Contact {
    let (sa, sb) = (&shells[a], &shells[b]);
    let ma = &model.meshes[sa.mesh];
    let rest: Vec<Vec<V3>> = model.meshes.iter().map(|m| m.pos.clone()).collect();
    let mut n = 0usize;
    let (mut sum, mut max) = (0.0f32, 0.0f32);
    for &v in &sa.verts {
        let (d, t, bary) = closest_in_shell(model, &rest, sb, ma.pos[v]);
        if d > reach {
            continue;
        }
        let wm = weight_mismatch(&vertex_weights(ma, v), &weights_at(&model.meshes[sb.mesh], t, bary));
        n += 1;
        sum += wm;
        max = max.max(wm);
    }
    Contact { other: b, verts: n, skin_mismatch: if n > 0 { sum / n as f32 } else { 0.0 }, skin_mismatch_max: max }
}
