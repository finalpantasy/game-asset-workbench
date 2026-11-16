//! What changed between two exports: texture pixels, meshes, skeleton, clips.
use crate::analysis::Parts;
use crate::{Model, Texture};

pub struct TexDiff {
    pub name: String,
    pub a_size: Option<[usize; 2]>,
    pub b_size: Option<[usize; 2]>,
    /// Size the comparison ran at (the larger of the two).
    pub size: [usize; 2],
    /// True when one side was scaled up to compare (then nearly every pixel differs).
    pub resized: bool,
    pub changed: usize,
    pub total: usize,
    /// 1 where a pixel changed, at `size`.
    pub mask: Vec<u8>,
    /// Changed area as [x0, y0, x1, y1] (exclusive), at `size`.
    pub bbox: Option<[usize; 4]>,
    /// Changed pixels per kit part id, largest first (only for atlases the kit maps).
    pub by_part: Vec<(u8, usize)>,
}

impl TexDiff {
    pub fn status(&self) -> &'static str {
        match (self.a_size, self.b_size) {
            (None, _) => "only in B",
            (_, None) => "only in A",
            _ if self.changed == 0 => "same",
            _ => "changed",
        }
    }
    /// Is the pixel at (u, v) changed?
    pub fn changed_at(&self, u: f32, v: f32) -> bool {
        let [w, h] = self.size;
        let x = ((u * w as f32) as i64).clamp(0, w as i64 - 1) as usize;
        let y = ((v * h as f32) as i64).clamp(0, h as i64 - 1) as usize;
        self.mask.get(y * w + x).copied().unwrap_or(0) != 0
    }
}

pub struct MeshDiff {
    pub name: String,
    pub a: Option<(usize, usize)>, // (vertices, triangles)
    pub b: Option<(usize, usize)>,
    pub moved_vertices: usize,
    pub uv_changed_vertices: usize,
    pub triangles_same: bool,
    /// Per vertex: moved or new UV (empty when the vertex counts differ).
    pub changed_verts: Vec<bool>,
}

impl MeshDiff {
    pub fn status(&self) -> &'static str {
        match (self.a, self.b) {
            (None, _) => "only in B",
            (_, None) => "only in A",
            (Some(a), Some(b)) if a != b => "different shape",
            _ if self.moved_vertices == 0 && self.uv_changed_vertices == 0 && self.triangles_same => "same",
            _ => "changed",
        }
    }
}

pub struct ModelDiff {
    pub identical_files: bool,
    pub bytes: [usize; 2],
    pub textures: Vec<TexDiff>,
    pub meshes: Vec<MeshDiff>,
    pub joints_same: bool,
    pub clips: [Vec<String>; 2],
}

impl ModelDiff {
    pub fn changed_pixels(&self) -> usize {
        self.textures.iter().map(|t| t.changed).sum()
    }
}

fn scaled(t: &Texture, w: usize, h: usize, x: usize, y: usize) -> &[u8] {
    let sx = x * t.w / w;
    let sy = y * t.h / h;
    let o = (sy * t.w + sx) * 4;
    &t.rgba[o..o + 4]
}

/// Pixel diff of two textures; `tol` is the largest per-channel change still counted as the same.
pub fn texture_diff(name: &str, a: Option<&Texture>, b: Option<&Texture>, parts: Option<&Parts>, tol: u8) -> TexDiff {
    let a_size = a.map(|t| [t.w, t.h]);
    let b_size = b.map(|t| [t.w, t.h]);
    let size = [a_size.map_or(0, |s| s[0]).max(b_size.map_or(0, |s| s[0])), a_size.map_or(0, |s| s[1]).max(b_size.map_or(0, |s| s[1]))];
    let [w, h] = size;
    let mut d = TexDiff { name: name.into(), a_size, b_size, size, resized: a_size.is_some() && b_size.is_some() && a_size != b_size, changed: 0, total: w * h, mask: vec![0; w * h], bbox: None, by_part: vec![], };
    let (Some(a), Some(b)) = (a, b) else {
        d.mask.fill(1);
        d.changed = w * h;
        d.bbox = Some([0, 0, w, h]);
        return d;
    };
    let mut bb = [usize::MAX, usize::MAX, 0, 0];
    let mut counts = [0usize; 256];
    let part_map = parts.and_then(|p| p.map_for(name));
    for y in 0..h {
        for x in 0..w {
            let (pa, pb) = (scaled(a, w, h, x, y), scaled(b, w, h, x, y));
            if (0..4).any(|c| pa[c].abs_diff(pb[c]) > tol) {
                d.mask[y * w + x] = 1;
                d.changed += 1;
                bb = [bb[0].min(x), bb[1].min(y), bb[2].max(x + 1), bb[3].max(y + 1)];
                if let Some((_, s, ids)) = part_map {
                    counts[ids[(y * s / h) * s + x * s / w] as usize] += 1;
                }
            }
        }
    }
    if d.changed > 0 {
        d.bbox = Some(bb);
    }
    d.by_part = counts.iter().enumerate().filter(|(_, &n)| n > 0).map(|(i, &n)| (i as u8, n)).collect();
    d.by_part.sort_by(|x, y| y.1.cmp(&x.1));
    d
}

/// Compares two opened exports. Textures are matched by image name, meshes by name.
pub fn diff(a: &Model, at: &[Option<Texture>], b: &Model, bt: &[Option<Texture>], parts: Option<&Parts>, tol: u8) -> ModelDiff {
    let mut names: Vec<String> = a.images.iter().map(|i| i.name.clone()).collect();
    for i in &b.images {
        if !names.contains(&i.name) {
            names.push(i.name.clone());
        }
    }
    fn find<'t>(m: &Model, ts: &'t [Option<Texture>], n: &str) -> Option<&'t Texture> {
        m.images.iter().position(|i| i.name == n).and_then(|i| ts.get(i)).and_then(|t| t.as_ref())
    }
    let mut textures = Vec::new();
    for n in &names {
        let (ta, tb) = (find(a, at, n), find(b, bt, n));
        let same_bytes = matches!((a.images.iter().find(|i| &i.name == n), b.images.iter().find(|i| &i.name == n)), (Some(x), Some(y)) if x.bytes == y.bytes);
        if same_bytes {
            let s = ta.map(|t| [t.w, t.h]).unwrap_or([0, 0]);
            let name = n.trim_end_matches(".png").to_owned();
            textures.push(TexDiff { name, a_size: Some(s), b_size: Some(s), size: s, resized: false, changed: 0, total: s[0] * s[1], mask: vec![0; s[0] * s[1]], bbox: None, by_part: vec![] });
        } else {
            textures.push(texture_diff(n.trim_end_matches(".png"), ta, tb, parts, tol));
        }
    }
    let mut mnames: Vec<String> = a.meshes.iter().map(|m| m.name.clone()).collect();
    for m in &b.meshes {
        if !mnames.contains(&m.name) {
            mnames.push(m.name.clone());
        }
    }
    let meshes = mnames
        .iter()
        .map(|n| {
            let ma = a.meshes.iter().find(|m| &m.name == n);
            let mb = b.meshes.iter().find(|m| &m.name == n);
            let mut d = MeshDiff {
                name: n.clone(),
                a: ma.map(|m| (m.pos.len(), m.tris.len())),
                b: mb.map(|m| (m.pos.len(), m.tris.len())),
                moved_vertices: 0,
                uv_changed_vertices: 0,
                triangles_same: false,
                changed_verts: vec![],
            };
            if let (Some(x), Some(y)) = (ma, mb) {
                if x.pos.len() == y.pos.len() && x.uv.len() == y.uv.len() {
                    let moved: Vec<bool> = x.pos.iter().zip(&y.pos).map(|(p, q)| (0..3).any(|k| (p[k] - q[k]).abs() > 1e-6)).collect();
                    let uv: Vec<bool> = x.uv.iter().zip(&y.uv).map(|(p, q)| (0..2).any(|k| (p[k] - q[k]).abs() > 1e-7)).collect();
                    d.moved_vertices = moved.iter().filter(|&&c| c).count();
                    d.uv_changed_vertices = uv.iter().filter(|&&c| c).count();
                    d.changed_verts = moved.iter().zip(&uv).map(|(a, b)| *a || *b).collect();
                }
                d.triangles_same = x.tris == y.tris;
            }
            d
        })
        .collect();
    ModelDiff {
        identical_files: a.glb.to_bytes() == b.glb.to_bytes(),
        bytes: [a.file_len, b.file_len],
        textures,
        meshes,
        joints_same: a.joint_names == b.joint_names,
        clips: [a.clips.clone(), b.clips.clone()],
    }
}

/// One side of a comparison, for `untouched_parts`: the model, its changed triangles and each triangle's part.
pub struct Side<'a> {
    pub model: &'a Model,
    pub changed: &'a [Vec<bool>],
    pub tri_parts: &'a [Vec<u8>],
}

/// Per texture in `d`: the kit parts on that texture that nothing touched. No changed pixel lies in the
/// part's area and no triangle of the part (on either side) moved, got a new UV, or covers a changed pixel.
/// Empty for textures that exist on one side only or that the kit does not map.
pub fn untouched_parts(d: &ModelDiff, parts: &Parts, sides: [Side; 2]) -> Vec<Vec<u8>> {
    d.textures
        .iter()
        .map(|t| {
            if t.a_size.is_none() || t.b_size.is_none() {
                return vec![];
            }
            let mut hit = [false; 256];
            for &(id, n) in &t.by_part {
                hit[id as usize] |= n > 0;
            }
            for s in &sides {
                for (mi, m) in s.model.meshes.iter().enumerate() {
                    let Some(img) = m.image else { continue };
                    if s.model.images[img].name.trim_end_matches(".png") != t.name {
                        continue;
                    }
                    for (&c, &p) in s.changed[mi].iter().zip(&s.tri_parts[mi]) {
                        hit[p as usize] |= c;
                    }
                }
            }
            parts.ids_in(&t.name).into_iter().filter(|&id| !hit[id as usize]).collect()
        })
        .collect()
}

/// Triangles of `model` that changed: a moved vertex or new UV, or a changed pixel in its texture area (by mesh).
pub fn changed_triangles(model: &Model, d: &ModelDiff) -> Vec<Vec<bool>> {
    let by_paint = paint_changed_triangles(model, d);
    model
        .meshes
        .iter()
        .zip(by_paint)
        .map(|(m, paint)| {
            let verts = d.meshes.iter().find(|x| x.name == m.name).map(|x| &x.changed_verts).filter(|v| v.len() == m.pos.len());
            m.tris.iter().zip(paint).map(|(t, p)| p || verts.is_some_and(|v| t.iter().any(|&i| v[i as usize]))).collect()
        })
        .collect()
}

fn paint_changed_triangles(model: &Model, d: &ModelDiff) -> Vec<Vec<bool>> {
    model
        .meshes
        .iter()
        .map(|m| {
            let Some(td) = m.image.and_then(|i| d.textures.iter().find(|t| t.name == model.images[i].name.trim_end_matches(".png"))) else {
                return vec![false; m.tris.len()];
            };
            if td.changed == 0 {
                return vec![false; m.tris.len()];
            }
            let [w, h] = td.size;
            m.tris
                .iter()
                .map(|t| {
                    let p = t.map(|v| [m.uv[v as usize][0] * w as f32, m.uv[v as usize][1] * h as f32]);
                    let x0 = p.iter().map(|q| q[0]).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
                    let x1 = (p.iter().map(|q| q[0]).fold(f32::MIN, f32::max).ceil() as usize).min(w);
                    let y0 = p.iter().map(|q| q[1]).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
                    let y1 = (p.iter().map(|q| q[1]).fold(f32::MIN, f32::max).ceil() as usize).min(h);
                    let [a, b, c] = p;
                    let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
                    if area.abs() < 1e-9 {
                        return td.changed_at((a[0] + b[0] + c[0]) / (3.0 * w as f32), (a[1] + b[1] + c[1]) / (3.0 * h as f32));
                    }
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                            let w0 = ((b[0] - px) * (c[1] - py) - (b[1] - py) * (c[0] - px)) / area;
                            let w1 = ((c[0] - px) * (a[1] - py) - (c[1] - py) * (a[0] - px)) / area;
                            if w0 >= 0.0 && w1 >= 0.0 && w0 + w1 <= 1.0 && td.mask[y * w + x] != 0 {
                                return true;
                            }
                        }
                    }
                    false
                })
                .collect()
        })
        .collect()
}
