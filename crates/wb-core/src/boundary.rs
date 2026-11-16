//! Boundaries on a textured model: what a visible line, edge or speck is before anyone edits it.
//!
//! A line on screen can be a UV seam (two texture islands meet), a clothing edge (two materials
//! meet in the paint), an intentional overlap (one piece passes over another, with its edge and
//! contact shadow), or an accidental paint mark (a stroke or speck inside one material). Only UV
//! seams and accidental marks should be blended away. The functions here label them, record which
//! piece lies over which, build material masks, measure texel density, and do the paint edits
//! (mark fill, edge smoothing, seam matching, material-aware padding) with those checks built in.
//! Nothing here is specific to one model or one part.
use crate::geom::{self, V3};
use crate::render::Camera;
use crate::shells;
use crate::{Model, Texture};
use std::collections::{HashMap, HashSet};

// ------------------------------------------------------------------------------------------------
// Screen G-buffer: which triangle, shell, depth and UV own each pixel.

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub mesh: u32,
    pub tri: u32,
    pub shell: u32,
    /// Larger is nearer (same convention as `shells::project`).
    pub depth: f32,
    pub uv: [f32; 2],
}

pub struct GBuffer {
    pub w: usize,
    pub h: usize,
    pub hits: Vec<Option<Hit>>,
}

impl GBuffer {
    pub fn at(&self, x: i64, y: i64) -> Option<Hit> {
        if x < 0 || y < 0 || x >= self.w as i64 || y >= self.h as i64 {
            return None;
        }
        self.hits[y as usize * self.w + x as usize]
    }
}

/// Rasterizes the nearest surface per pixel with perspective-correct UVs. `shell_of(mesh, tri)`
/// gives a shell id (`shells::NONE` skips the triangle). With `textures`, texels under alpha 128
/// are holes, as with the game's alpha cutout.
pub fn gbuffer(model: &Model, textures: Option<&[Option<Texture>]>, pose: &[Vec<V3>], shell_of: &dyn Fn(usize, usize) -> usize, cam: &Camera, w: usize, h: usize) -> GBuffer {
    let mut g = GBuffer { w, h, hits: vec![None; w * h] };
    let (eye, f, _, _) = shells::basis(cam);
    for (mi, m) in model.meshes.iter().enumerate() {
        let proj: Vec<[f32; 3]> = pose[mi].iter().map(|&p| shells::project(cam, w, h, p)).collect();
        let viewz: Vec<f32> = pose[mi].iter().map(|&p| geom::dot(geom::sub(p, eye), f)).collect();
        let tex = textures.and_then(|ts| m.image.and_then(|i| ts.get(i)).and_then(|t| t.as_ref()));
        for (ti, t) in m.tris.iter().enumerate() {
            let s = shell_of(mi, ti);
            if s == shells::NONE {
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
            let uv3 = t.map(|v| m.uv[v as usize]);
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
                    let d = if cam.ortho { -(w0 * z3[0] + w1 * z3[1] + w2 * z3[2]) } else { -1.0 / (w0 / z3[0] + w1 / z3[1] + w2 / z3[2]) };
                    let i = y * w + x;
                    if g.hits[i].is_some_and(|hh| hh.depth >= d) {
                        continue;
                    }
                    let bw = if cam.ortho {
                        [w0, w1, w2]
                    } else {
                        let q = [w0 / z3[0], w1 / z3[1], w2 / z3[2]];
                        let s = q[0] + q[1] + q[2];
                        [q[0] / s, q[1] / s, q[2] / s]
                    };
                    let uv = [bw[0] * uv3[0][0] + bw[1] * uv3[1][0] + bw[2] * uv3[2][0], bw[0] * uv3[0][1] + bw[1] * uv3[1][1] + bw[2] * uv3[2][1]];
                    if let Some(tx) = tex {
                        let (tx_x, tx_y) = texel_of(tx.w, tx.h, uv);
                        if tx.rgba[(tx_y * tx.w + tx_x) * 4 + 3] < 128 {
                            continue;
                        }
                    }
                    g.hits[i] = Some(Hit { mesh: mi as u32, tri: ti as u32, shell: s as u32, depth: d, uv });
                }
            }
        }
    }
    g
}

/// The texel a UV lands on (nearest lookup, wrapping like the renderer's clamp-free path).
pub fn texel_of(w: usize, h: usize, uv: [f32; 2]) -> (usize, usize) {
    (((uv[0] * w as f32) as i64).rem_euclid(w as i64) as usize, ((uv[1] * h as f32) as i64).rem_euclid(h as i64) as usize)
}

pub fn luma(p: &[u8]) -> f32 {
    (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0
}

/// Shell id per (mesh, triangle), for `gbuffer` and `shells::layers`.
pub fn shell_table(model: &Model, sh: &[shells::Shell]) -> Vec<Vec<usize>> {
    shells::tri_shell(model, sh)
}

// ------------------------------------------------------------------------------------------------
// Material masks.

#[derive(Clone, Debug)]
pub struct Material {
    pub name: String,
    pub rgb: [u8; 3],
}

/// Parses `name=r,g,b;name=r,g,b`.
pub fn parse_materials(s: &str) -> Result<Vec<Material>, String> {
    s.split(';')
        .filter(|x| !x.trim().is_empty())
        .map(|kv| {
            let (n, c) = kv.split_once('=').ok_or_else(|| format!("material {kv}: want name=r,g,b"))?;
            let v: Vec<u8> = c.split(',').map(|x| x.trim().parse::<u8>().map_err(|e| format!("{kv}: {e}"))).collect::<Result<_, _>>()?;
            if v.len() != 3 {
                return Err(format!("{kv}: want three numbers"));
            }
            Ok(Material { name: n.trim().to_owned(), rgb: [v[0], v[1], v[2]] })
        })
        .collect()
}

pub const UNKNOWN: u8 = 255;

/// Color distance used for material matching: luma counts half as much as chroma, so a shaded
/// fold of leather stays leather while white shirt paint never does.
fn mat_dist(p: &[u8], c: [u8; 3]) -> f32 {
    let (r, g, b) = (p[0] as f32 - c[0] as f32, p[1] as f32 - c[1] as f32, p[2] as f32 - c[2] as f32);
    let l = 0.299 * r + 0.587 * g + 0.114 * b;
    let (cr, cg, cb) = (r - l, g - l, b - l);
    (0.25 * l * l + cr * cr + cg * cg + cb * cb).sqrt()
}

/// Material index per texel of `rect` = [x0, y0, x1, y1) (row-major), the nearest palette color
/// within `max_dist`, else `UNKNOWN`.
pub fn material_mask(rgba: &[u8], w: usize, rect: [usize; 4], palette: &[Material], max_dist: f32) -> Vec<u8> {
    let [x0, y0, x1, y1] = rect;
    let mut out = Vec::with_capacity((x1 - x0) * (y1 - y0));
    for y in y0..y1 {
        for x in x0..x1 {
            let o = (y * w + x) * 4;
            let p = &rgba[o..o + 3];
            let best = palette.iter().enumerate().map(|(i, m)| (i, mat_dist(p, m.rgb))).min_by(|a, b| a.1.total_cmp(&b.1));
            out.push(match best {
                Some((i, d)) if d <= max_dist => i as u8,
                _ => UNKNOWN,
            });
        }
    }
    out
}

/// Majority material of the texels each triangle samples (center inside its UV footprint), for
/// triangles of `mesh` whose texture is `image`. Unknown texels do not vote.
pub fn triangle_materials(model: &Model, mesh: usize, tris: &[usize], rgba: &[u8], w: usize, h: usize, palette: &[Material], max_dist: f32) -> HashMap<usize, (u8, f32)> {
    let m = &model.meshes[mesh];
    let mut out = HashMap::new();
    for &t in tris {
        let p: Vec<[f32; 2]> = m.tris[t].iter().map(|&i| [m.uv[i as usize][0] * w as f32, m.uv[i as usize][1] * h as f32]).collect();
        let mut votes = vec![0usize; palette.len()];
        let mut total = 0;
        crate::edit::raster_tri_pad([p[0], p[1], p[2]], w, h, 0.0, |x, y, _| {
            let o = (y * w + x) * 4;
            let px = &rgba[o..o + 3];
            if let Some((i, d)) = palette.iter().enumerate().map(|(i, mm)| (i, mat_dist(px, mm.rgb))).min_by(|a, b| a.1.total_cmp(&b.1)) {
                if d <= max_dist {
                    votes[i] += 1;
                    total += 1;
                }
            }
        });
        if let Some((i, &n)) = votes.iter().enumerate().max_by_key(|e| e.1) {
            if n > 0 {
                out.insert(t, (i as u8, n as f32 / total as f32));
            }
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------
// Accidental paint marks: small islands of one material inside another.

#[derive(Clone, Debug)]
pub struct Mark {
    /// Material of the mark and of the paint around it.
    pub material: u8,
    pub host: u8,
    pub texels: Vec<(usize, usize)>,
    /// Share of the one-texel ring around the mark that is host material.
    pub host_share: f32,
    pub bbox: [usize; 4],
}

/// Connected groups (8-neighbor) of material `of` inside `rect`, no larger than `max_texels`,
/// whose surrounding ring is at least `min_host` host material `on`. `mask` comes from
/// `material_mask` over the same rect.
pub fn find_marks(mask: &[u8], rect: [usize; 4], of: u8, on: u8, max_texels: usize, min_host: f32) -> Vec<Mark> {
    let [x0, y0, x1, y1] = rect;
    let (rw, rh) = (x1 - x0, y1 - y0);
    let mut seen = vec![false; rw * rh];
    let mut out = Vec::new();
    for start in 0..rw * rh {
        if seen[start] || mask[start] != of {
            continue;
        }
        let mut comp = vec![start];
        seen[start] = true;
        let mut k = 0;
        while k < comp.len() {
            let (cx, cy) = ((comp[k] % rw) as i64, (comp[k] / rw) as i64);
            k += 1;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (cx + dx, cy + dy);
                    if nx < 0 || ny < 0 || nx >= rw as i64 || ny >= rh as i64 {
                        continue;
                    }
                    let ni = ny as usize * rw + nx as usize;
                    if !seen[ni] && mask[ni] == of {
                        seen[ni] = true;
                        comp.push(ni);
                    }
                }
            }
            if comp.len() > max_texels * 4 {
                break;
            }
        }
        if comp.len() > max_texels {
            continue;
        }
        let set: HashSet<usize> = comp.iter().copied().collect();
        let (mut ring, mut host) = (0usize, 0usize);
        let mut touches_rect_edge = false;
        let mut ringset = HashSet::new();
        for &c in &comp {
            let (cx, cy) = ((c % rw) as i64, (c / rw) as i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (cx + dx, cy + dy);
                    if nx < 0 || ny < 0 || nx >= rw as i64 || ny >= rh as i64 {
                        touches_rect_edge = true;
                        continue;
                    }
                    let ni = ny as usize * rw + nx as usize;
                    if !set.contains(&ni) && ringset.insert(ni) {
                        ring += 1;
                        if mask[ni] == on {
                            host += 1;
                        }
                    }
                }
            }
        }
        let share = if ring > 0 { host as f32 / ring as f32 } else { 0.0 };
        if touches_rect_edge || share < min_host {
            continue;
        }
        let texels: Vec<(usize, usize)> = comp.iter().map(|&c| (x0 + c % rw, y0 + c / rw)).collect();
        let bbox = [
            texels.iter().map(|t| t.0).min().unwrap(),
            texels.iter().map(|t| t.1).min().unwrap(),
            texels.iter().map(|t| t.0).max().unwrap() + 1,
            texels.iter().map(|t| t.1).max().unwrap() + 1,
        ];
        out.push(Mark { material: of, host: on, texels, host_share: share, bbox });
    }
    out
}

/// Fills each mark by cloning host paint from the nearby offset (within `search` texels) whose
/// patch is all host material and best matches the ring around the mark, so the grain of the
/// host paint continues through the fill. Texels in `protected` are never written. Returns the
/// texels changed and, per mark, the offset used (None when no clean patch was found).
pub fn fill_marks(rgba: &mut [u8], w: usize, mask: &[u8], rect: [usize; 4], marks: &[Mark], search: i64, protected: &HashSet<(usize, usize)>) -> (usize, Vec<Option<(i64, i64)>>) {
    let [x0, y0, x1, y1] = rect;
    let rw = x1 - x0;
    let mat_at = |x: i64, y: i64| -> Option<u8> {
        if x < x0 as i64 || y < y0 as i64 || x >= x1 as i64 || y >= y1 as i64 {
            return None;
        }
        Some(mask[(y as usize - y0) * rw + (x as usize - x0)])
    };
    let mut changed = 0;
    let mut used = Vec::new();
    for mk in marks {
        let set: HashSet<(usize, usize)> = mk.texels.iter().copied().collect();
        let mut ring: Vec<(usize, usize)> = Vec::new();
        for &(x, y) in &mk.texels {
            for dy in -2i64..=2 {
                for dx in -2i64..=2 {
                    let p = ((x as i64 + dx) as usize, (y as i64 + dy) as usize);
                    if !set.contains(&p) && !ring.contains(&p) {
                        ring.push(p);
                    }
                }
            }
        }
        let mut best: Option<((i64, i64), f32)> = None;
        for oy in -search..=search {
            for ox in -search..=search {
                if ox.abs() + oy.abs() < 3 {
                    continue;
                }
                let clean = mk.texels.iter().chain(ring.iter()).all(|&(x, y)| {
                    let (sx, sy) = (x as i64 + ox, y as i64 + oy);
                    mat_at(sx, sy) == Some(mk.host) && !set.contains(&(sx as usize, sy as usize))
                });
                if !clean {
                    continue;
                }
                let mut err = 0.0;
                for &(x, y) in &ring {
                    let a = (y * w + x) * 4;
                    let b = (((y as i64 + oy) as usize) * w + (x as i64 + ox) as usize) * 4;
                    for c in 0..3 {
                        err += (rgba[a + c] as f32 - rgba[b + c] as f32).abs();
                    }
                }
                err += 0.5 * ((ox * ox + oy * oy) as f32).sqrt();
                if best.is_none_or(|b| err < b.1) {
                    best = Some(((ox, oy), err));
                }
            }
        }
        match best {
            Some(((ox, oy), _)) => {
                for &(x, y) in &mk.texels {
                    if protected.contains(&(x, y)) {
                        continue;
                    }
                    let a = (y * w + x) * 4;
                    let b = (((y as i64 + oy) as usize) * w + (x as i64 + ox) as usize) * 4;
                    let src = [rgba[b], rgba[b + 1], rgba[b + 2]];
                    if rgba[a..a + 3] != src {
                        changed += 1;
                    }
                    rgba[a..a + 3].copy_from_slice(&src);
                }
                used.push(Some((ox, oy)));
            }
            None => used.push(None),
        }
    }
    (changed, used)
}

// ------------------------------------------------------------------------------------------------
// Clothing edges in the paint: smoothing a ragged boundary between two materials.

#[derive(Clone, Debug)]
pub struct EdgeProfile {
    /// Per column of the rect: the first row (texel y) of `lower` material below `upper`, if any.
    pub rows: Vec<Option<f32>>,
    pub smoothed: Vec<Option<f32>>,
    /// Largest column-to-column jump before and after smoothing, in texels.
    pub jag_before: f32,
    pub jag_after: f32,
}

/// Finds the `upper`-to-`lower` material boundary in each column of `rect` (v down: upper
/// material above) and smooths its height with a running median then mean over `window` columns.
pub fn edge_profile(mask: &[u8], rect: [usize; 4], upper: u8, lower: u8, window: usize) -> EdgeProfile {
    let [x0, y0, x1, y1] = rect;
    let (rw, rh) = (x1 - x0, y1 - y0);
    let mut rows = Vec::with_capacity(rw);
    for cx in 0..rw {
        // The boundary is the last upper texel followed (within 3 texels) by lower texels.
        let mut found = None;
        for cy in 0..rh.saturating_sub(1) {
            if mask[cy * rw + cx] == upper {
                let below = (1..=3).filter(|d| cy + d < rh).any(|d| mask[(cy + d) * rw + cx] == lower);
                if below {
                    found = Some((y0 + cy + 1) as f32);
                }
            }
        }
        rows.push(found);
    }
    let half = window / 2;
    let med: Vec<Option<f32>> = (0..rw)
        .map(|i| {
            let mut v: Vec<f32> = (i.saturating_sub(half)..(i + half + 1).min(rw)).filter_map(|j| rows[j]).collect();
            if v.len() < 2 || rows[i].is_none() {
                return rows[i];
            }
            v.sort_by(f32::total_cmp);
            Some(v[v.len() / 2])
        })
        .collect();
    let smoothed: Vec<Option<f32>> = (0..rw)
        .map(|i| {
            rows[i]?;
            let v: Vec<f32> = (i.saturating_sub(half / 2)..(i + half / 2 + 1).min(rw)).filter_map(|j| med[j]).collect();
            Some(v.iter().sum::<f32>() / v.len().max(1) as f32)
        })
        .collect();
    let jag = |r: &[Option<f32>]| r.windows(2).filter_map(|p| Some((p[0]? - p[1]?).abs())).fold(0.0, f32::max);
    EdgeProfile { jag_before: jag(&rows), jag_after: jag(&smoothed), rows, smoothed }
}

/// Repaints the boundary to follow `prof.smoothed`: in each column, texels that end up on the
/// wrong side are filled from the same material further along the column (upper material is
/// copied down from above the old boundary, lower material up from below it), so each side keeps
/// its own paint and grain. `reach` limits how far a column may move. Protected texels are kept.
pub fn smooth_edge(rgba: &mut [u8], w: usize, mask: &[u8], rect: [usize; 4], prof: &EdgeProfile, upper: u8, lower: u8, reach: f32, protected: &HashSet<(usize, usize)>) -> usize {
    let [x0, y0, _, y1] = rect;
    let rw = rect[2] - x0;
    let src = rgba.to_vec();
    let mut changed = 0;
    for cx in 0..rw {
        let (Some(old), Some(new)) = (prof.rows[cx], prof.smoothed[cx]) else { continue };
        let (old, new) = (old.round() as i64, new.round() as i64);
        if (old - new).abs() as f32 > reach || old == new {
            continue;
        }
        let x = x0 + cx;
        let put = |rgba: &mut [u8], y: i64, from: i64| -> bool {
            if y < y0 as i64 || y >= y1 as i64 || from < y0 as i64 || from >= y1 as i64 || protected.contains(&(x, y as usize)) {
                return false;
            }
            let a = (y as usize * w + x) * 4;
            let b = (from as usize * w + x) * 4;
            let c = [src[b], src[b + 1], src[b + 2]];
            let ch = rgba[a..a + 3] != c;
            rgba[a..a + 3].copy_from_slice(&c);
            ch
        };
        if new > old {
            // Upper material grows down: rows old..new get upper paint mirrored from above.
            for y in old..new {
                let from = old - 1 - (y - old);
                let ok = from >= y0 as i64 && mask[(from as usize - y0) * rw + cx] == upper;
                if ok && put(rgba, y, from) {
                    changed += 1;
                }
            }
        } else {
            // Lower material grows up: rows new..old get lower paint mirrored from below.
            for y in new..old {
                let from = old + (old - 1 - y);
                let ok = from < y1 as i64 && mask[(from as usize - y0) * rw + cx] == lower;
                if ok && put(rgba, y, from) {
                    changed += 1;
                }
            }
        }
    }
    changed
}

// ------------------------------------------------------------------------------------------------
// UV seams and seam-matched painting.

#[derive(Clone, Debug)]
pub struct SeamSide {
    pub tri: usize,
    /// UVs of the shared edge's two ends and of the triangle's third corner.
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub opp: [f32; 2],
}

#[derive(Clone, Debug)]
pub struct SeamEdge {
    pub mesh: usize,
    /// World positions of the edge's two ends (rest pose).
    pub p: [V3; 2],
    pub s0: SeamSide,
    pub s1: SeamSide,
}

/// Edges shared in space (welded positions) by two triangles of `mesh` whose UVs differ there by
/// more than `min_texels`: where two texture islands meet. `keep(tri)` limits the triangles.
pub fn uv_seams(model: &Model, mesh: usize, w: usize, h: usize, min_texels: f32, keep: &dyn Fn(usize) -> bool) -> Vec<SeamEdge> {
    let m = &model.meshes[mesh];
    let wid = shells::weld(&m.pos);
    let mut by_edge: HashMap<(usize, usize), Vec<(usize, u32, u32)>> = HashMap::new();
    for (ti, t) in m.tris.iter().enumerate() {
        if !keep(ti) {
            continue;
        }
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let (wa, wb) = (wid[a as usize], wid[b as usize]);
            let key = if wa < wb { (wa, wb) } else { (wb, wa) };
            by_edge.entry(key).or_default().push((ti, a, b));
        }
    }
    let d = |p: [f32; 2], q: [f32; 2]| (((p[0] - q[0]) * w as f32).powi(2) + ((p[1] - q[1]) * h as f32).powi(2)).sqrt();
    let mut out = Vec::new();
    for ((wa, _), uses) in by_edge {
        if uses.len() != 2 {
            continue;
        }
        let side = |(ti, a, b): (usize, u32, u32)| {
            // Orient so `a` is the welded vertex `wa`.
            let (a, b) = if wid[a as usize] == wa { (a, b) } else { (b, a) };
            let opp = m.tris[ti].iter().copied().find(|&v| v != a && v != b).unwrap_or(a);
            (SeamSide { tri: ti, a: m.uv[a as usize], b: m.uv[b as usize], opp: m.uv[opp as usize] }, a, b)
        };
        let (s0, a0, b0) = side(uses[0]);
        let (s1, _, _) = side(uses[1]);
        if d(s0.a, s1.a).max(d(s0.b, s1.b)) <= min_texels {
            continue;
        }
        out.push(SeamEdge { mesh, p: [m.pos[a0 as usize], m.pos[b0 as usize]], s0, s1 });
    }
    out
}

/// Grain of a texture patch: mean color, contrast (luma standard deviation after removing a
/// `blur`-texel box mean), dominant stroke direction in UV (degrees, 0 = along u) with its
/// coherence (0 = no direction, 1 = all strokes parallel), and stroke scale (texels between
/// light and dark, from the mean gradient).
#[derive(Clone, Copy, Debug, Default)]
pub struct Grain {
    pub mean: [f32; 3],
    pub contrast: f32,
    pub angle_deg: f32,
    pub coherence: f32,
    pub scale: f32,
    pub texels: usize,
}

pub fn grain(rgba: &[u8], w: usize, h: usize, texels: &[(usize, usize)], blur: usize) -> Grain {
    if texels.is_empty() {
        return Grain::default();
    }
    let l = |x: usize, y: usize| luma(&rgba[(y.min(h - 1) * w + x.min(w - 1)) * 4..]);
    let mut mean = [0.0f32; 3];
    let (mut sxx, mut syy, mut sxy, mut gsum, mut var) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for &(x, y) in texels {
        let o = (y * w + x) * 4;
        for c in 0..3 {
            mean[c] += rgba[o + c] as f32;
        }
        let (xm, xp, ym, yp) = (x.saturating_sub(1), x + 1, y.saturating_sub(1), y + 1);
        let gx = 0.5 * (l(xp, y) - l(xm, y));
        let gy = 0.5 * (l(x, yp) - l(x, ym));
        sxx += gx * gx;
        syy += gy * gy;
        sxy += gx * gy;
        gsum += (gx * gx + gy * gy).sqrt();
        let mut s = 0.0;
        let mut n = 0.0;
        for dy in 0..=2 * blur {
            for dx in 0..=2 * blur {
                let (qx, qy) = ((x + dx).saturating_sub(blur), (y + dy).saturating_sub(blur));
                s += l(qx, qy);
                n += 1.0;
            }
        }
        let hp = l(x, y) - s / n;
        var += hp * hp;
    }
    let n = texels.len() as f32;
    for c in &mut mean {
        *c /= n;
    }
    // Structure tensor: gradients run across strokes, so the stroke direction is perpendicular.
    let ang = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let tr = sxx + syy;
    let det = ((sxx - syy).powi(2) + 4.0 * sxy * sxy).sqrt();
    let contrast = (var / n).sqrt();
    Grain {
        mean,
        contrast,
        angle_deg: (ang.to_degrees() + 90.0).rem_euclid(180.0),
        coherence: if tr > 0.0 { det / tr } else { 0.0 },
        scale: if gsum > 0.0 { 2.0 * contrast * n / gsum } else { 0.0 },
        texels: texels.len(),
    }
}

/// Texels of one seam side: inside the side's triangle (center test) and within `band` texels of
/// the seam edge, with their distance to the edge and position along it (0..1).
fn side_band(s: &SeamSide, w: usize, h: usize, band: f32) -> Vec<(usize, usize, f32, f32)> {
    let p = [s.a, s.b, s.opp].map(|q| [q[0] * w as f32, q[1] * h as f32]);
    let e = [p[1][0] - p[0][0], p[1][1] - p[0][1]];
    let el = (e[0] * e[0] + e[1] * e[1]).sqrt().max(1e-6);
    let mut out = Vec::new();
    crate::edit::raster_tri_pad(p, w, h, 0.0, |x, y, _| {
        let c = [x as f32 + 0.5 - p[0][0], y as f32 + 0.5 - p[0][1]];
        let t = (c[0] * e[0] + c[1] * e[1]) / (el * el);
        let d = (c[0] * e[1] - c[1] * e[0]).abs() / el;
        if d <= band {
            out.push((x, y, d, t.clamp(0.0, 1.0)));
        }
    });
    out
}

#[derive(Clone, Debug, Default)]
pub struct SeamMatch {
    pub edges: usize,
    pub changed: usize,
    /// Grain of the band on each side, before and after.
    pub before: [Grain; 2],
    pub after: [Grain; 2],
    /// Mean color step across the seam (RGB distance of the edge rows), before and after.
    pub step_before: f32,
    pub step_after: f32,
    /// Stroke directions compared in space (degrees between the two sides' grain on the model).
    pub direction_gap_deg: f32,
}

/// Seam-matched painting. For every seam edge, the low-frequency color of each side along the
/// edge (a `blur`-texel mean, sampled at `samples` points) is pulled toward the average of the
/// two sides, fading to nothing `band` texels into each side; each side's fine grain is scaled
/// toward the shared contrast the same way. So both islands meet with the same color, shading
/// and grain strength, and the paint further in is untouched. Protected texels are kept.
pub fn match_seams(rgba: &mut [u8], w: usize, h: usize, seams: &[SeamEdge], band: f32, blur: usize, samples: usize, protected: &HashSet<(usize, usize)>) -> SeamMatch {
    let src = rgba.to_vec();
    let low = |x: f32, y: f32| -> [f32; 3] {
        let (cx, cy) = (x as i64, y as i64);
        let mut s = [0.0f32; 3];
        let mut n = 0.0;
        for dy in -(blur as i64)..=blur as i64 {
            for dx in -(blur as i64)..=blur as i64 {
                let (qx, qy) = ((cx + dx).clamp(0, w as i64 - 1) as usize, (cy + dy).clamp(0, h as i64 - 1) as usize);
                let o = (qy * w + qx) * 4;
                for c in 0..3 {
                    s[c] += src[o + c] as f32;
                }
                n += 1.0;
            }
        }
        s.map(|v| v / n)
    };
    // A point just inside a side, at fraction t along the edge.
    let inside = |s: &SeamSide, t: f32, by: f32| -> [f32; 2] {
        let a = [s.a[0] * w as f32, s.a[1] * h as f32];
        let b = [s.b[0] * w as f32, s.b[1] * h as f32];
        let o = [s.opp[0] * w as f32, s.opp[1] * h as f32];
        let p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
        let toward = [o[0] - p[0], o[1] - p[1]];
        let l = (toward[0].powi(2) + toward[1].powi(2)).sqrt().max(1e-6);
        [p[0] + by * toward[0] / l, p[1] + by * toward[1] / l]
    };
    let mut res = SeamMatch { edges: seams.len(), ..Default::default() };
    let mut bands: [Vec<(usize, usize)>; 2] = [Vec::new(), Vec::new()];
    let mut writes: HashMap<(usize, usize), ([f32; 3], f32)> = HashMap::new();
    let (mut step_n, mut step_b) = (0.0f32, 0.0f32);
    let mut dirs: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    for e in seams {
        let sides = [&e.s0, &e.s1];
        let n = samples.max(2);
        let cols: Vec<[[f32; 3]; 2]> = (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                let q0 = inside(sides[0], t, blur as f32 + 1.0);
                let q1 = inside(sides[1], t, blur as f32 + 1.0);
                [low(q0[0], q0[1]), low(q1[0], q1[1])]
            })
            .collect();
        for c in &cols {
            step_b += ((c[0][0] - c[1][0]).powi(2) + (c[0][1] - c[1][1]).powi(2) + (c[0][2] - c[1][2]).powi(2)).sqrt();
            step_n += 1.0;
        }
        for (k, s) in sides.iter().enumerate() {
            let tex = side_band(s, w, h, band);
            let pts: Vec<(usize, usize)> = tex.iter().map(|t| (t.0, t.1)).collect();
            let g = grain(&src, w, h, &pts, blur);
            dirs[k].push(g.angle_deg);
            bands[k].extend(pts);
            for (x, y, d, t) in tex {
                let i = ((t * (n - 1) as f32).round() as usize).min(n - 1);
                let target: [f32; 3] = std::array::from_fn(|c| 0.5 * (cols[i][0][c] + cols[i][1][c]));
                let fade = 1.0 - (d / band).clamp(0.0, 1.0);
                let fade = fade * fade * (3.0 - 2.0 * fade);
                let delta: [f32; 3] = std::array::from_fn(|c| (target[c] - cols[i][k][c]) * fade);
                let ent = writes.entry((x, y)).or_insert(([0.0; 3], 0.0));
                for c in 0..3 {
                    ent.0[c] += delta[c];
                }
                ent.1 += 1.0;
            }
        }
    }
    res.step_before = if step_n > 0.0 { step_b / step_n } else { 0.0 };
    res.before = [grain(&src, w, h, &bands[0], blur), grain(&src, w, h, &bands[1], blur)];
    // Contrast: scale each side's fine detail toward the shared contrast, faded like the color.
    let target_c = 0.5 * (res.before[0].contrast + res.before[1].contrast);
    let gain: [f32; 2] = std::array::from_fn(|k| if res.before[k].contrast > 1e-4 { (target_c / res.before[k].contrast).clamp(0.5, 2.0) } else { 1.0 });
    let side_of: HashMap<(usize, usize), usize> = bands[0].iter().map(|&p| (p, 0)).chain(bands[1].iter().map(|&p| (p, 1))).collect();
    for ((x, y), (delta, n)) in &writes {
        if protected.contains(&(*x, *y)) {
            continue;
        }
        let o = (y * w + x) * 4;
        let lo = low(*x as f32, *y as f32);
        let k = side_of.get(&(*x, *y)).copied().unwrap_or(0);
        // Fade the contrast change with the same weight as the color change (its share of the
        // full delta is unknown here, so use the color fade proxy: 1 at the edge, 0 at `band`).
        let fade = (delta.iter().map(|d| d.abs()).sum::<f32>() > 0.0) as u8 as f32;
        let g = 1.0 + (gain[k] - 1.0) * fade;
        let mut out = [0u8; 3];
        for c in 0..3 {
            let v = src[o + c] as f32;
            let hp = v - lo[c];
            out[c] = (lo[c] + hp * g + delta[c] / n).round().clamp(0.0, 255.0) as u8;
        }
        if rgba[o..o + 3] != out {
            res.changed += 1;
        }
        rgba[o..o + 3].copy_from_slice(&out);
    }
    res.after = [grain(rgba, w, h, &bands[0], blur), grain(rgba, w, h, &bands[1], blur)];
    let mean = |v: &[f32]| if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 };
    res.direction_gap_deg = {
        let d = (mean(&dirs[0]) - mean(&dirs[1])).abs() % 180.0;
        d.min(180.0 - d)
    };
    // Step after: same samples on the edited texture.
    let (mut sa, mut na) = (0.0f32, 0.0f32);
    for e in seams {
        for i in 0..samples.max(2) {
            let t = i as f32 / (samples.max(2) - 1) as f32;
            let q0 = inside(&e.s0, t, blur as f32 + 1.0);
            let q1 = inside(&e.s1, t, blur as f32 + 1.0);
            let m = |q: [f32; 2]| -> [f32; 3] {
                let (cx, cy) = (q[0] as i64, q[1] as i64);
                let mut s = [0.0f32; 3];
                let mut n = 0.0;
                for dy in -(blur as i64)..=blur as i64 {
                    for dx in -(blur as i64)..=blur as i64 {
                        let o = (((cy + dy).clamp(0, h as i64 - 1) as usize) * w + (cx + dx).clamp(0, w as i64 - 1) as usize) * 4;
                        for c in 0..3 {
                            s[c] += rgba[o + c] as f32;
                        }
                        n += 1.0;
                    }
                }
                s.map(|v| v / n)
            };
            let (a, b) = (m(q0), m(q1));
            sa += ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            na += 1.0;
        }
    }
    res.step_after = if na > 0.0 { sa / na } else { 0.0 };
    res
}

// ------------------------------------------------------------------------------------------------
// Material-aware island padding.

/// Like `strapcheck::pad_uv_islands`, but a new texel takes only the material that holds the most
/// of its filled neighbors, so shirt paint never bleeds into leather padding (or leather into
/// trousers). `mask` is the material mask of `rect`; unknown-material texels still pad but never
/// mix with known ones. Returns texels changed.
pub fn pad_by_material(rgba: &mut [u8], w: usize, rect: [usize; 4], covered: &[bool], mask: &[u8], passes: usize) -> usize {
    let [x0, y0, x1, y1] = rect;
    let (rw, rh) = (x1 - x0, y1 - y0);
    let mut filled = covered.to_vec();
    let mut mat: Vec<u8> = mask.to_vec();
    let mut changed = 0;
    for _ in 0..passes {
        let mut next = Vec::new();
        for ry in 0..rh {
            for rx in 0..rw {
                if filled[ry * rw + rx] {
                    continue;
                }
                let mut by: HashMap<u8, ([u32; 3], u32)> = HashMap::new();
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (rx as i32 + dx, ry as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= rw as i32 || ny >= rh as i32 {
                        continue;
                    }
                    let ni = ny as usize * rw + nx as usize;
                    if !filled[ni] {
                        continue;
                    }
                    let o = ((y0 + ny as usize) * w + x0 + nx as usize) * 4;
                    let e = by.entry(mat[ni]).or_insert(([0; 3], 0));
                    for c in 0..3 {
                        e.0[c] += rgba[o + c] as u32;
                    }
                    e.1 += 1;
                }
                // Prefer a known material over unknown, then the most neighbors.
                if let Some((&m, &(s, n))) = by.iter().max_by_key(|(m, (_, n))| (**m != UNKNOWN, *n)) {
                    next.push((rx, ry, m, [(s[0] / n) as u8, (s[1] / n) as u8, (s[2] / n) as u8]));
                }
            }
        }
        if next.is_empty() {
            break;
        }
        for (rx, ry, m, c) in next {
            let o = ((y0 + ry) * w + x0 + rx) * 4;
            if rgba[o..o + 3] != c {
                changed += 1;
            }
            rgba[o..o + 3].copy_from_slice(&c);
            filled[ry * rw + rx] = true;
            mat[ry * rw + rx] = m;
        }
    }
    changed
}

/// Island-edge texels that are sampled (`covered`) but do not carry the island's paint: the
/// painter's mask stopped short of the triangle edge, so the edge rows show gutter or stray pad
/// color as a thin line along the UV seam. From each covered texel next to an uncovered one, the
/// reference is the mean of the three covered texels `depth + 1 ..= depth + 3` steps inward (away
/// from the gutter). Walking inward from the edge, texels whose largest channel step from that
/// reference exceeds `tol` are marked uncovered, up to `depth` of them, stopping at the first
/// texel that matches; `pad_by_material` then repaints them from the interior. Returns texels
/// reclaimed.
pub fn reclaim_gutter_paint(rgba: &[u8], w: usize, rect: [usize; 4], covered: &mut [bool], tol: u8, depth: usize) -> usize {
    let [x0, y0, x1, y1] = rect;
    let (rw, rh) = (x1 - x0, y1 - y0);
    let px = |rx: i64, ry: i64| -> [u8; 3] {
        let o = ((y0 + ry as usize) * w + x0 + rx as usize) * 4;
        [rgba[o], rgba[o + 1], rgba[o + 2]]
    };
    let inside = |x: i64, y: i64| x >= 0 && y >= 0 && x < rw as i64 && y < rh as i64;
    let orig = covered.to_vec();
    let cov = |x: i64, y: i64| inside(x, y) && orig[y as usize * rw + x as usize];
    let mut take = HashSet::new();
    for ry in 0..rh as i64 {
        for rx in 0..rw as i64 {
            if !cov(rx, ry) {
                continue;
            }
            for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                if !inside(rx + dx, ry + dy) || cov(rx + dx, ry + dy) {
                    continue;
                }
                // Reference paint a little way inside the island.
                let mut sum = [0u32; 3];
                let mut ok = true;
                for k in depth as i64 + 1..=depth as i64 + 3 {
                    let (x, y) = (rx - dx * k, ry - dy * k);
                    if !cov(x, y) {
                        ok = false;
                        break;
                    }
                    let p = px(x, y);
                    for c in 0..3 {
                        sum[c] += p[c] as u32;
                    }
                }
                if !ok {
                    continue;
                }
                let r = sum.map(|s| (s / 3) as u8);
                for k in 0..depth as i64 {
                    let (x, y) = (rx - dx * k, ry - dy * k);
                    let p = px(x, y);
                    if (0..3).all(|c| p[c].abs_diff(r[c]) <= tol) {
                        break;
                    }
                    take.insert(y as usize * rw + x as usize);
                }
            }
        }
    }
    for &i in &take {
        covered[i] = false;
    }
    take.len()
}

// ------------------------------------------------------------------------------------------------
// Overlap order.

#[derive(Clone, Debug, PartialEq)]
pub struct Crossing {
    pub over: usize,
    pub under: usize,
    /// Pixels where `over` is the nearest surface and `under` lies within `gap` behind it.
    pub pixels: usize,
}

/// Which shell lies over which wherever two shells stack within `gap` world units on screen
/// (a strap on a belt, a loop over a band). Pairs with fewer than `min_pixels` are dropped.
pub fn crossings(l: &shells::Layers, gap: f32, min_pixels: usize) -> Vec<Crossing> {
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    for i in 0..l.w * l.h {
        let (a, b) = (l.s1[i], l.s2[i]);
        if a == shells::NONE || b == shells::NONE || l.d1[i] - l.d2[i] > gap {
            continue;
        }
        *count.entry((a, b)).or_default() += 1;
    }
    let mut out: Vec<Crossing> = count.into_iter().filter(|e| e.1 >= min_pixels).map(|((over, under), pixels)| Crossing { over, under, pixels }).collect();
    out.sort_by(|a, b| b.pixels.cmp(&a.pixels));
    out
}

/// Crossings whose order flipped (or vanished) between two renders of the same view: before, A
/// lay over B on most of their shared pixels; after, B lies over A on most of them, or they no
/// longer stack. Returns (over before, under before, pixels before, pixels after same order,
/// pixels after reversed).
pub fn order_changes(before: &[Crossing], after: &[Crossing]) -> Vec<(usize, usize, usize, usize, usize)> {
    let get = |v: &[Crossing], a: usize, b: usize| v.iter().find(|c| c.over == a && c.under == b).map_or(0, |c| c.pixels);
    let mut out = Vec::new();
    for c in before {
        let rev_before = get(before, c.under, c.over);
        if c.pixels <= rev_before {
            continue;
        }
        let (same, rev) = (get(after, c.over, c.under), get(after, c.under, c.over));
        if rev >= same || same * 4 < c.pixels {
            out.push((c.over, c.under, c.pixels, same, rev));
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------
// Labeling lines and specks on screen.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    /// Two texture islands meet, or a pixel samples past its island's painted edge.
    UvSeam,
    /// Two materials meet in the paint (shirt hem on leather, belt edge on trousers).
    ClothingEdge,
    /// One piece passes over another: its edge and contact shadow.
    Overlap,
    /// A stroke or speck of one material inside another.
    PaintMark,
    /// The outline of the model against the background.
    Silhouette,
    /// None of the above: shading or filtering on one surface.
    Unknown,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::UvSeam => "uv seam",
            Kind::ClothingEdge => "clothing edge",
            Kind::Overlap => "intentional overlap",
            Kind::PaintMark => "accidental paint mark",
            Kind::Silhouette => "silhouette",
            Kind::Unknown => "unknown",
        }
    }
    /// Only UV seams and accidental paint marks may be blended away.
    pub fn blendable(self) -> bool {
        matches!(self, Kind::UvSeam | Kind::PaintMark)
    }
    pub fn color(self) -> [u8; 3] {
        match self {
            Kind::UvSeam => [255, 235, 40],
            Kind::ClothingEdge => [60, 200, 255],
            Kind::Overlap => [80, 255, 80],
            Kind::PaintMark => [255, 40, 220],
            Kind::Silhouette => [150, 150, 150],
            Kind::Unknown => [255, 255, 255],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Labeled {
    pub x: usize,
    pub y: usize,
    /// True for a light mark on darker paint, false for a dark line.
    pub light: bool,
    pub kind: Kind,
    pub hit: Hit,
    pub texel: (usize, usize),
    pub why: &'static str,
}

pub struct LabelParams<'a> {
    /// Brightness step (0..1) a pixel needs against both pixels `reach` away to count.
    pub dip: f32,
    pub reach: i64,
    /// Depth step (world units) that makes a neighbor a different surface layer.
    pub layer_gap: f32,
    /// Texels per screen pixel step above which two pixels of one shell sit on different islands.
    pub seam_jump: f32,
    pub palette: &'a [Material],
    pub max_dist: f32,
    /// Same-material painted strokes smaller than this many connected screen pixels are texture
    /// grain, not marks, and become Unknown. A different material painted in (a spill) always
    /// stays a mark, however small.
    pub min_mark_px: usize,
    /// Dark same-material strokes at least this many connected pixels long are painted creases or
    /// contact shadows (an overlap drawn into the paint, such as a strap passing under a loop):
    /// they become Overlap so edits leave them alone. 0 turns this off.
    pub min_crease_px: usize,
}

const SPILL: &str = "a different material painted inside one material";
const GRAIN: &str = "grain-size stroke inside one material";
const CREASE: &str = "long painted crease or contact shadow (painted overlap)";

/// Sorts same-material PaintMark components (8-connected) by size: smaller than min_px is grain
/// (Unknown); dark and at least crease_px is a painted crease (Overlap). Spills stay marks.
fn classify_strokes(out: &mut [Labeled], w: usize, h: usize, min_px: usize, crease_px: usize) {
    let mut at = vec![usize::MAX; w * h];
    for (i, l) in out.iter().enumerate() {
        if l.kind == Kind::PaintMark {
            at[l.y * w + l.x] = i;
        }
    }
    let mut seen = vec![false; out.len()];
    for s in 0..out.len() {
        if out[s].kind != Kind::PaintMark || seen[s] {
            continue;
        }
        let (mut comp, mut stack) = (vec![s], vec![s]);
        seen[s] = true;
        while let Some(i) = stack.pop() {
            let (x, y) = (out[i].x as i64, out[i].y as i64);
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = at[ny as usize * w + nx as usize];
                    if j != usize::MAX && !seen[j] {
                        seen[j] = true;
                        comp.push(j);
                        stack.push(j);
                    }
                }
            }
        }
        let spill = comp.iter().any(|&i| out[i].why == SPILL);
        let dark = comp.iter().filter(|&&i| !out[i].light).count() * 2 > comp.len();
        if spill {
            continue;
        }
        if comp.len() < min_px {
            for i in comp {
                out[i].kind = Kind::Unknown;
                out[i].why = GRAIN;
            }
        } else if crease_px > 0 && dark && comp.len() >= crease_px {
            for i in comp {
                out[i].kind = Kind::Overlap;
                out[i].why = CREASE;
            }
        }
    }
}

/// Finds thin dark lines and light specks in a rendered view (`rgba`, same camera and size as
/// `g`) and labels each pixel: silhouette, overlap (a different shell nearer or farther within
/// two pixels), UV seam (a UV jump between neighbors of one shell, or the texel center outside the
/// island's paint), clothing edge (the texture changes material across the line), accidental mark
/// (the texture itself is dark or light there, inside one material), or unknown.
pub fn label_lines(model: &Model, tex: &Texture, image: usize, g: &GBuffer, rgba: &[u8], p: &LabelParams) -> Vec<Labeled> {
    let (w, h) = (g.w, g.h);
    let lum = |x: i64, y: i64| -> Option<f32> {
        if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
            return None;
        }
        Some(luma(&rgba[(y as usize * w + x as usize) * 4..]))
    };
    let tl = |uv: [f32; 2]| {
        let (x, y) = texel_of(tex.w, tex.h, uv);
        luma(&tex.rgba[(y * tex.w + x) * 4..])
    };
    let tmat = |uv: [f32; 2]| -> u8 {
        let (x, y) = texel_of(tex.w, tex.h, uv);
        let o = (y * tex.w + x) * 4;
        let px = &tex.rgba[o..o + 3];
        match p.palette.iter().enumerate().map(|(i, m)| (i, mat_dist(px, m.rgb))).min_by(|a, b| a.1.total_cmp(&b.1)) {
            Some((i, d)) if d <= p.max_dist => i as u8,
            _ => UNKNOWN,
        }
    };
    let mut out = Vec::new();
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let Some(hit) = g.at(x, y) else { continue };
            if model.meshes[hit.mesh as usize].image != Some(image) {
                continue;
            }
            let c = lum(x, y).unwrap();
            // Thin feature across one of the two axes.
            let mut feat: Option<(bool, (i64, i64))> = None;
            for (dx, dy) in [(0i64, 1i64), (1, 0)] {
                let (Some(a), Some(b)) = (lum(x - dx * p.reach, y - dy * p.reach), lum(x + dx * p.reach, y + dy * p.reach)) else { continue };
                if c + p.dip <= a.min(b) {
                    feat = Some((false, (dx, dy)));
                } else if c >= a.max(b) + p.dip {
                    feat = Some((true, (dx, dy)));
                }
                if feat.is_some() {
                    break;
                }
            }
            let Some((light, (dx, dy))) = feat else { continue };
            let mut kind = None;
            let mut why = "";
            // Silhouette and overlap: neighbors within two pixels.
            'n: for ny in -2..=2i64 {
                for nx in -2..=2i64 {
                    match g.at(x + nx, y + ny) {
                        None => {
                            kind = Some(Kind::Silhouette);
                            why = "background within 2 px";
                            break 'n;
                        }
                        Some(q) if q.shell != hit.shell && (q.depth - hit.depth).abs() > p.layer_gap => {
                            kind = Some(Kind::Overlap);
                            why = "another piece within 2 px, at a different depth";
                            break 'n;
                        }
                        _ => {}
                    }
                }
            }
            if kind.is_none() {
                // UV jump between same-shell neighbors.
                let step = |q: Hit| (((q.uv[0] - hit.uv[0]) * tex.w as f32).powi(2) + ((q.uv[1] - hit.uv[1]) * tex.h as f32).powi(2)).sqrt();
                let mut jumps = Vec::new();
                for (nx, ny) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    if let Some(q) = g.at(x + nx, y + ny) {
                        if q.shell == hit.shell {
                            jumps.push(step(q));
                        }
                    }
                }
                jumps.sort_by(f32::total_cmp);
                if jumps.len() >= 2 && jumps[jumps.len() - 1] > p.seam_jump.max(6.0 * jumps[0]) {
                    kind = Some(Kind::UvSeam);
                    why = "UV jumps between neighbors on one piece";
                }
            }
            if kind.is_none() {
                // Texel center outside the pixel's own triangle by more than half a texel: the
                // lookup lands past the island's painted area.
                let m = &model.meshes[hit.mesh as usize];
                let t = m.tris[hit.tri as usize];
                let (tx, ty) = texel_of(tex.w, tex.h, hit.uv);
                let pc = [tx as f32 + 0.5, ty as f32 + 0.5];
                let q = t.map(|v| [m.uv[v as usize][0] * tex.w as f32, m.uv[v as usize][1] * tex.h as f32]);
                let d = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[1][1] - q[0][1]) * (q[2][0] - q[0][0]);
                let side = |a: [f32; 2], b: [f32; 2]| ((b[0] - a[0]) * (pc[1] - a[1]) - (b[1] - a[1]) * (pc[0] - a[0])) * d.signum();
                let inside = side(q[0], q[1]) >= 0.0 && side(q[1], q[2]) >= 0.0 && side(q[2], q[0]) >= 0.0;
                if !inside {
                    let near_other = [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)].iter().any(|&(nx, ny)| g.at(x + nx, y + ny).is_some_and(|q| q.tri != hit.tri));
                    if near_other {
                        // On a triangle edge; whether this texel is painted is decided below.
                    } else {
                        kind = Some(Kind::UvSeam);
                        why = "texel center outside the triangle's UV footprint";
                    }
                }
            }
            if kind.is_none() {
                let a = g.at(x - dx * p.reach, y - dy * p.reach);
                let b = g.at(x + dx * p.reach, y + dy * p.reach);
                let here = tl(hit.uv);
                let sides: Vec<f32> = [a, b].iter().flatten().map(|q| tl(q.uv)).collect();
                let painted = !sides.is_empty() && if light { here >= sides.iter().cloned().fold(f32::MIN, f32::max) + 0.5 * p.dip } else { here + 0.5 * p.dip <= sides.iter().cloned().fold(f32::MAX, f32::min) };
                let mats: Vec<u8> = [a, b].iter().flatten().map(|q| tmat(q.uv)).collect();
                let two_materials = mats.len() == 2 && mats[0] != mats[1] && mats[0] != UNKNOWN && mats[1] != UNKNOWN;
                let here_mat = tmat(hit.uv);
                if two_materials {
                    kind = Some(Kind::ClothingEdge);
                    why = "the paint changes material across the line";
                } else if painted {
                    kind = Some(Kind::PaintMark);
                    why = if mats.first().is_some_and(|&m| m != here_mat && here_mat != UNKNOWN) { SPILL } else { "a stroke painted inside one material" };
                } else {
                    kind = Some(Kind::Unknown);
                    why = "not in the paint at this texel (shading or filtering)";
                }
            }
            let (tx, ty) = texel_of(tex.w, tex.h, hit.uv);
            out.push(Labeled { x: x as usize, y: y as usize, light, kind: kind.unwrap(), hit, texel: (tx, ty), why });
        }
    }
    classify_strokes(&mut out, w, h, p.min_mark_px, p.min_crease_px);
    out
}

/// Texels that labeled pixels of `kinds` sample, grown by `grow` texels: what paint edits must
/// leave alone (for example every texel under an intentional overlap's edge and shadow).
pub fn protected_texels(labels: &[Labeled], kinds: &[Kind], grow: i64) -> HashSet<(usize, usize)> {
    let mut out = HashSet::new();
    for l in labels.iter().filter(|l| kinds.contains(&l.kind)) {
        for dy in -grow..=grow {
            for dx in -grow..=grow {
                out.insert(((l.texel.0 as i64 + dx).max(0) as usize, (l.texel.1 as i64 + dy).max(0) as usize));
            }
        }
    }
    out
}

/// Per triangle and kind: pixel counts, for reports.
pub fn summarize(labels: &[Labeled]) -> Vec<((u32, u32), Kind, usize)> {
    let mut m: HashMap<((u32, u32), Kind), usize> = HashMap::new();
    for l in labels {
        *m.entry(((l.hit.mesh, l.hit.tri), l.kind)).or_default() += 1;
    }
    let mut v: Vec<_> = m.into_iter().map(|((t, k), n)| (t, k, n)).collect();
    v.sort_by(|a, b| b.2.cmp(&a.2));
    v
}

// ------------------------------------------------------------------------------------------------
// Texel density.

#[derive(Clone, Debug, Default)]
pub struct Density {
    pub tris: usize,
    /// Surface area in square meters (rest pose) and texture area in texels.
    pub area_m2: f32,
    pub texels: f32,
    /// Texels per meter (square root of texels per square meter): median, lowest, highest.
    pub per_m_median: f32,
    pub per_m_min: f32,
    pub per_m_max: f32,
    /// Texels across the part's widest side, for "how many real pixels does this part get".
    pub across: f32,
}

/// Real texture resolution of a set of triangles of one mesh, whatever the atlas size.
pub fn texel_density(model: &Model, mesh: usize, tris: &[usize], tw: usize, th: usize) -> Density {
    let m = &model.meshes[mesh];
    let mut per = Vec::new();
    let (mut area, mut tex) = (0.0f32, 0.0f32);
    let mut pts = Vec::new();
    for &t in tris {
        let [a, b, c] = m.tris[t].map(|v| v as usize);
        let wa = 0.5 * geom::len(geom::cross(geom::sub(m.pos[b], m.pos[a]), geom::sub(m.pos[c], m.pos[a])));
        let (ua, ub, uc) = (m.uv[a], m.uv[b], m.uv[c]);
        let ta = 0.5 * (((ub[0] - ua[0]) * (uc[1] - ua[1]) - (ub[1] - ua[1]) * (uc[0] - ua[0])) * (tw * th) as f32).abs();
        area += wa;
        tex += ta;
        if wa > 1e-10 {
            per.push((ta / wa).sqrt());
        }
        pts.extend([m.pos[a], m.pos[b], m.pos[c]]);
    }
    per.sort_by(f32::total_cmp);
    let (lo, hi) = geom::bounds(pts.iter());
    let span = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2]);
    let med = per.get(per.len() / 2).copied().unwrap_or(0.0);
    Density {
        tris: tris.len(),
        area_m2: area,
        texels: tex,
        per_m_median: med,
        per_m_min: per.first().copied().unwrap_or(0.0),
        per_m_max: per.last().copied().unwrap_or(0.0),
        across: med * span,
    }
}
