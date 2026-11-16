//! Small CPU renderer: unlit textured triangles (the way the game draws characters),
//! a depth buffer, and a pick buffer that records which triangle owns each pixel.
use crate::{Model, Texture};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
    pub target: [f32; 3],
    pub fov_y: f32,
    /// Orthographic (the game's field camera) instead of perspective.
    pub ortho: bool,
    /// Half the view height in world units, for orthographic views.
    pub ortho_half: f32,
}

/// Bounds of the model's rest pose: (low corner, high corner).
pub fn bounds(model: &Model) -> ([f32; 3], [f32; 3]) {
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for m in &model.meshes {
        for p in &m.pos {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    (lo, hi)
}

impl Camera {
    /// Front view of the whole model in a square view. glTF front of the character is -z.
    pub fn frame(model: &Model) -> Camera {
        Camera::fit(model, 1.0, 0.8, false)
    }

    /// Front view where the model's height fills `fill` of the view height and its width fits too.
    pub fn fit(model: &Model, aspect: f32, fill: f32, ortho: bool) -> Camera {
        let (lo, hi) = bounds(model);
        let target = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
        let half_h = 0.5 * (hi[1] - lo[1]);
        // Width allows for turning the model: the larger of its x and z extents.
        let half_w = 0.5 * (hi[0] - lo[0]).max(hi[2] - lo[2]);
        let fov_y = 30f32.to_radians();
        let need = (half_h / fill).max(half_w / (aspect.max(0.1) * 0.95)).max(1e-3);
        let t = (fov_y * 0.5).tan();
        Camera { yaw: std::f32::consts::PI, pitch: 0.0, dist: need / t + half_w, target, fov_y, ortho, ortho_half: need }
    }

    /// Zoom by a factor (< 1 moves closer).
    pub fn zoom(&mut self, f: f32) {
        self.dist *= f;
        self.ortho_half *= f;
    }

    fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) {
        let (cp, sp, cy, sy) = (self.pitch.cos(), self.pitch.sin(), self.yaw.cos(), self.yaw.sin());
        let eye = [self.target[0] + self.dist * cp * sy, self.target[1] + self.dist * sp, self.target[2] + self.dist * cp * cy];
        let f = norm(sub(self.target, eye));
        let r = norm(cross(f, [0.0, 1.0, 0.0]));
        let u = cross(r, f);
        (eye, f, r, u)
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt().max(1e-20);
    [a[0] / l, a[1] / l, a[2] / l]
}

pub struct Frame {
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,
    /// Global triangle id per pixel (see `tri_offsets`), -1 for background.
    pub pick: Vec<i32>,
}

/// Start of each mesh's triangles in the global triangle numbering.
pub fn tri_offsets(model: &Model) -> Vec<usize> {
    let mut o = Vec::with_capacity(model.meshes.len());
    let mut acc = 0;
    for m in &model.meshes {
        o.push(acc);
        acc += m.tris.len();
    }
    o
}

/// Splits a global triangle id into (mesh, triangle).
pub fn split_id(model: &Model, id: usize) -> (usize, usize) {
    let mut rest = id;
    for (mi, m) in model.meshes.iter().enumerate() {
        if rest < m.tris.len() {
            return (mi, rest);
        }
        rest -= m.tris.len();
    }
    (model.meshes.len(), 0)
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Filter {
    /// Mipmaps with trilinear filtering: what the game shows with the HD Characters plugin.
    Smooth,
    /// Nearest texel from the full-size texture: shows the painted pixels exactly.
    Texels,
}

pub struct Options<'a> {
    pub cam: Camera,
    pub w: usize,
    pub h: usize,
    /// Background gradient, top and bottom.
    pub background: ([u8; 3], [u8; 3]),
    pub filter: Filter,
    /// Posed vertex positions per mesh (from `Model::posed`); None = rest pose.
    pub positions: Option<&'a [Vec<[f32; 3]>]>,
}

fn fetch(l: (usize, usize, &[u8]), u: f32, v: f32) -> [f32; 4] {
    let (w, h, px) = l;
    let x = u * w as f32 - 0.5;
    let y = v * h as f32 - 0.5;
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let cx = |x: f32| (x as i64).clamp(0, w as i64 - 1) as usize;
    let cy = |y: f32| (y as i64).clamp(0, h as i64 - 1) as usize;
    let mut out = [0.0; 4];
    for (dx, dy, wt) in [(0.0, 0.0, (1.0 - fx) * (1.0 - fy)), (1.0, 0.0, fx * (1.0 - fy)), (0.0, 1.0, (1.0 - fx) * fy), (1.0, 1.0, fx * fy)] {
        let o = (cy(y0 + dy) * w + cx(x0 + dx)) * 4;
        for c in 0..4 {
            out[c] += wt * px[o + c] as f32;
        }
    }
    out
}

fn sample(t: &Texture, u: f32, v: f32, lod: f32, filter: Filter) -> [f32; 4] {
    match filter {
        Filter::Texels => {
            let sx = ((u * t.w as f32) as i64).clamp(0, t.w as i64 - 1) as usize;
            let sy = ((v * t.h as f32) as i64).clamp(0, t.h as i64 - 1) as usize;
            let o = (sy * t.w + sx) * 4;
            [t.rgba[o] as f32, t.rgba[o + 1] as f32, t.rgba[o + 2] as f32, t.rgba[o + 3] as f32]
        }
        Filter::Smooth => {
            let lod = lod.clamp(0.0, (t.levels() - 1) as f32);
            let l0 = lod.floor() as usize;
            let f = lod - l0 as f32;
            let a = fetch(t.level(l0), u, v);
            if f < 0.01 || l0 + 1 >= t.levels() {
                return a;
            }
            let b = fetch(t.level(l0 + 1), u, v);
            std::array::from_fn(|c| a[c] * (1.0 - f) + b[c] * f)
        }
    }
}

/// A tint value that greys a triangle out (darkened, colorless) instead of mixing a color in,
/// so a selection stands out whatever its own color.
pub const DIM: [u8; 3] = [0, 0, 1];

/// `textures` is indexed by image (see `crate::image_textures`).
/// `tint(mesh, tri)` returns a highlight color mixed over the texture, `DIM`, or None.
pub fn render(model: &Model, textures: &[Option<Texture>], o: &Options, tint: &dyn Fn(usize, usize) -> Option<[u8; 3]>) -> Frame {
    render_masked(model, textures, o, tint, None)
}

/// As `render`, drawing only the triangles `keep(mesh, tri)` accepts (isolating a part, or
/// hiding one to see what is behind it). Hidden triangles never reach the pick buffer.
pub fn render_masked(model: &Model, textures: &[Option<Texture>], o: &Options, tint: &dyn Fn(usize, usize) -> Option<[u8; 3]>, keep: Option<&dyn Fn(usize, usize) -> bool>) -> Frame {
    let (w, h) = (o.w.max(1), o.h.max(1));
    let mut rgba = vec![0u8; w * h * 4];
    let (top, bot) = o.background;
    for y in 0..h {
        let f = y as f32 / h as f32;
        let c: [u8; 3] = std::array::from_fn(|k| (top[k] as f32 * (1.0 - f) + bot[k] as f32 * f) as u8);
        for x in 0..w {
            rgba[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    let mut pick = vec![-1i32; w * h];
    let mut izbuf = vec![f32::MIN; w * h];
    let cam = &o.cam;
    let (eye, f, r, u) = cam.basis();
    let focal = 0.5 * h as f32 / (cam.fov_y * 0.5).tan();
    let oscale = 0.5 * h as f32 / cam.ortho_half.max(1e-6);
    let (cx, cy) = (0.5 * w as f32, 0.5 * h as f32);
    let offsets = tri_offsets(model);
    for (mi, m) in model.meshes.iter().enumerate() {
        let tex = m.image.and_then(|i| textures.get(i)).and_then(|t| t.as_ref());
        let pos = o.positions.and_then(|p| p.get(mi)).map(|v| v.as_slice()).unwrap_or(&m.pos);
        // Project vertices once: screen x, y and camera depth z.
        let proj: Vec<[f32; 3]> = pos
            .iter()
            .map(|&p| {
                let d = sub(p, eye);
                let z = dot(d, f);
                if cam.ortho {
                    [cx + oscale * dot(d, r), cy - oscale * dot(d, u), z]
                } else {
                    [cx + focal * dot(d, r) / z, cy - focal * dot(d, u) / z, z]
                }
            })
            .collect();
        for (ti, t) in m.tris.iter().enumerate() {
            if keep.is_some_and(|k| !k(mi, ti)) {
                continue;
            }
            let [a, b, c] = t.map(|v| proj[v as usize]);
            if !cam.ortho && (a[2] < 1e-3 || b[2] < 1e-3 || c[2] < 1e-3) {
                continue;
            }
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            if area.abs() < 1e-9 {
                continue;
            }
            let x0 = (a[0].min(b[0]).min(c[0]).floor().max(0.0)) as usize;
            let x1 = (a[0].max(b[0]).max(c[0]).ceil().min(w as f32 - 1.0)).max(0.0) as usize;
            let y0 = (a[1].min(b[1]).min(c[1]).floor().max(0.0)) as usize;
            let y1 = (a[1].max(b[1]).max(c[1]).ceil().min(h as f32 - 1.0)).max(0.0) as usize;
            if x0 > x1 || y0 > y1 {
                continue;
            }
            let uv = t.map(|v| m.uv[v as usize]);
            // Depth as 1/z (perspective) or -z (orthographic): larger is nearer either way.
            let iz = if cam.ortho { [-a[2], -b[2], -c[2]] } else { [1.0 / a[2], 1.0 / b[2], 1.0 / c[2]] };
            // Mip level: texels per screen pixel over this triangle.
            let lod = tex.map_or(0.0, |tx| {
                let du1 = [(uv[1][0] - uv[0][0]) * tx.w as f32, (uv[1][1] - uv[0][1]) * tx.h as f32];
                let du2 = [(uv[2][0] - uv[0][0]) * tx.w as f32, (uv[2][1] - uv[0][1]) * tx.h as f32];
                let tex_area = (du1[0] * du2[1] - du1[1] * du2[0]).abs();
                (tex_area / area.abs()).max(1e-9).sqrt().log2().max(0.0)
            });
            let hl = tint(mi, ti);
            let gid = (offsets[mi] + ti) as i32;
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
                    let i = y * w + x;
                    let z = w0 * iz[0] + w1 * iz[1] + w2 * iz[2];
                    if z <= izbuf[i] {
                        continue;
                    }
                    let mut col = [200.0f32, 200.0, 200.0, 255.0];
                    if let Some(tx) = tex {
                        let (tu, tv) = if cam.ortho {
                            (w0 * uv[0][0] + w1 * uv[1][0] + w2 * uv[2][0], w0 * uv[0][1] + w1 * uv[1][1] + w2 * uv[2][1])
                        } else {
                            (
                                (w0 * uv[0][0] * iz[0] + w1 * uv[1][0] * iz[1] + w2 * uv[2][0] * iz[2]) / z,
                                (w0 * uv[0][1] * iz[0] + w1 * uv[1][1] * iz[1] + w2 * uv[2][1] * iz[2]) / z,
                            )
                        };
                        col = sample(tx, tu, tv, lod, o.filter);
                        if col[3] < 128.0 {
                            continue; // alpha cutout
                        }
                    }
                    if hl == Some(DIM) {
                        let g = (col[0] * 0.3 + col[1] * 0.59 + col[2] * 0.11) * 0.3 + 14.0;
                        col[0] = g;
                        col[1] = g;
                        col[2] = g + 4.0;
                    } else if let Some(hc) = hl {
                        for k in 0..3 {
                            col[k] = col[k] * 0.4 + hc[k] as f32 * 0.6;
                        }
                    }
                    izbuf[i] = z;
                    pick[i] = gid;
                    for k in 0..3 {
                        rgba[i * 4 + k] = col[k].clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }
    Frame { w, h, rgba, pick }
}
