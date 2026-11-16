//! Paint-projection bake: render a camera view of a model, take a painted or upscaled picture of
//! that view, register it back to the render, and bake the paint onto the model's texture atlas
//! through the triangles that show it, touching only the texels a mask allows.
//!
//! Method (ported from the Python/OpenCV experiments, not their code):
//! 1. `render_view` draws the view with `render::render` and keeps a per-pixel triangle id and a
//!    camera-space depth (the "sidecar"), so a later step can ask which surface a pixel shows.
//! 2. `register` aligns the painted picture to the render with a similarity transform (scale,
//!    rotation, shift) found by direct alignment of edge maps, coarse to fine. It refuses a bad scale.
//! 3. `bake` goes texel by texel: each allowed texel is placed in 3D through its triangle, projected
//!    into every view, kept only when the render shows that triangle (depth and id agree) and the
//!    face is not edge-on, and takes the painted color there. Views blend by view angle.
//! 4. A seam pass pairs the true mesh seams (edges whose two triangles sit apart in UV space) and
//!    pulls the two sides to the same color within a few texels. A gutter pass then pads each
//!    island outward so mip levels do not bleed.
//!
//! Nothing outside the mask changes: texels shared with a triangle outside the mask are excluded,
//! the seam pass only moves texels of the same island, and the gutter pass only writes texels whose
//! nearest island is allowed. `verify_atlas` re-checks that with its own raster.
use crate::pattern::{classify, Part};
use crate::render::{self, split_id, tri_offsets, Camera, Filter};
use crate::Model;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;

pub type V3 = [f32; 3];

fn sub3(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot3(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross3(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm3(a: V3) -> V3 {
    let l = dot3(a, a).sqrt().max(1e-20);
    [a[0] / l, a[1] / l, a[2] / l]
}
fn smooth01(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

// ---------------------------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------------------------

/// An RGBA8 image.
#[derive(Clone)]
pub struct Img {
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,
}

impl Img {
    pub fn new(w: usize, h: usize, fill: [u8; 4]) -> Img {
        let mut rgba = vec![0u8; w * h * 4];
        for px in rgba.chunks_exact_mut(4) {
            px.copy_from_slice(&fill);
        }
        Img { w, h, rgba }
    }
    pub fn decode(bytes: &[u8]) -> Result<Img, String> {
        let im = image::load_from_memory(bytes).map_err(|e| e.to_string())?.to_rgba8();
        Ok(Img { w: im.width() as usize, h: im.height() as usize, rgba: im.into_raw() })
    }
    pub fn open(path: &Path) -> Result<Img, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Img::decode(&b).map_err(|e| format!("{}: {e}", path.display()))
    }
    pub fn to_png(&self) -> Result<Vec<u8>, String> {
        let mut out = std::io::Cursor::new(Vec::new());
        image::RgbaImage::from_raw(self.w as u32, self.h as u32, self.rgba.clone())
            .ok_or("bad image size")?
            .write_to(&mut out, image::ImageFormat::Png)
            .map_err(|e| e.to_string())?;
        Ok(out.into_inner())
    }
    pub fn save(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.to_png()?).map_err(|e| format!("{}: {e}", path.display()))
    }
    /// Bilinear sample at a continuous coordinate (pixel i covers [i, i+1)), edges clamped.
    pub fn bilinear(&self, x: f32, y: f32) -> [f32; 4] {
        bilinear_rgba(&self.rgba, self.w, self.h, x, y)
    }
    /// Enlarged copy by a whole factor with a Lanczos3 filter (a plain resampler, no model).
    pub fn upscaled(&self, k: usize) -> Img {
        let src = image::RgbaImage::from_raw(self.w as u32, self.h as u32, self.rgba.clone()).unwrap();
        let out = image::imageops::resize(&src, (self.w * k) as u32, (self.h * k) as u32, image::imageops::FilterType::Lanczos3);
        Img { w: self.w * k, h: self.h * k, rgba: out.into_raw() }
    }
    /// Resized copy (triangle filter; a smaller target averages).
    pub fn resized(&self, w: usize, h: usize) -> Img {
        let src = image::RgbaImage::from_raw(self.w as u32, self.h as u32, self.rgba.clone()).unwrap();
        let out = image::imageops::resize(&src, w as u32, h as u32, image::imageops::FilterType::Triangle);
        Img { w, h, rgba: out.into_raw() }
    }
}

pub fn bilinear_rgba(px: &[u8], w: usize, h: usize, x: f32, y: f32) -> [f32; 4] {
    let (fx, fy) = (x - 0.5, y - 0.5);
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0, fy - y0);
    let cx = |v: f32| (v as i64).clamp(0, w as i64 - 1) as usize;
    let cy = |v: f32| (v as i64).clamp(0, h as i64 - 1) as usize;
    let mut out = [0.0f32; 4];
    for (dx, dy, wt) in [(0.0, 0.0, (1.0 - tx) * (1.0 - ty)), (1.0, 0.0, tx * (1.0 - ty)), (0.0, 1.0, (1.0 - tx) * ty), (1.0, 1.0, tx * ty)] {
        let o = (cy(y0 + dy) * w + cx(x0 + dx)) * 4;
        for c in 0..4 {
            out[c] += wt * px[o + c] as f32;
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Camera math (the same formulas as render.rs, which keeps its basis private)
// ---------------------------------------------------------------------------------------------

pub struct Rig {
    pub eye: V3,
    pub f: V3,
    pub r: V3,
    pub u: V3,
    pub focal: f32,
    pub oscale: f32,
    pub cx: f32,
    pub cy: f32,
    pub ortho: bool,
}

impl Rig {
    pub fn new(c: &Camera, w: usize, h: usize) -> Rig {
        let (cp, sp, cy, sy) = (c.pitch.cos(), c.pitch.sin(), c.yaw.cos(), c.yaw.sin());
        let eye = [c.target[0] + c.dist * cp * sy, c.target[1] + c.dist * sp, c.target[2] + c.dist * cp * cy];
        let f = norm3(sub3(c.target, eye));
        let r = norm3(cross3(f, [0.0, 1.0, 0.0]));
        let u = cross3(r, f);
        Rig {
            eye,
            f,
            r,
            u,
            focal: 0.5 * h as f32 / (c.fov_y * 0.5).tan(),
            oscale: 0.5 * h as f32 / c.ortho_half.max(1e-6),
            cx: 0.5 * w as f32,
            cy: 0.5 * h as f32,
            ortho: c.ortho,
        }
    }
    /// Screen x, y (continuous pixel coordinates) and camera depth z.
    pub fn project(&self, p: V3) -> [f32; 3] {
        let d = sub3(p, self.eye);
        let z = dot3(d, self.f);
        if self.ortho {
            [self.cx + self.oscale * dot3(d, self.r), self.cy - self.oscale * dot3(d, self.u), z]
        } else {
            [self.cx + self.focal * dot3(d, self.r) / z, self.cy - self.focal * dot3(d, self.u) / z, z]
        }
    }
    /// Unit vector from a surface point to the camera.
    pub fn view_dir(&self, p: V3) -> V3 {
        if self.ortho {
            [-self.f[0], -self.f[1], -self.f[2]]
        } else {
            norm3(sub3(self.eye, p))
        }
    }
    /// World size of one pixel at depth z.
    pub fn pixel_size(&self, z: f32) -> f32 {
        if self.ortho {
            1.0 / self.oscale
        } else {
            z / self.focal
        }
    }
    /// Camera depth where the ray through screen point (px, py) meets the plane through `a` with normal `n`.
    pub fn depth_on_plane(&self, px: f32, py: f32, a: V3, n: V3) -> Option<f32> {
        if self.ortho {
            let (aa, bb) = ((px - self.cx) / self.oscale, -(py - self.cy) / self.oscale);
            let den = dot3(n, self.f);
            if den.abs() < 1e-12 {
                return None;
            }
            let q = sub3(sub3(sub3(a, self.eye), [self.r[0] * aa, self.r[1] * aa, self.r[2] * aa]), [self.u[0] * bb, self.u[1] * bb, self.u[2] * bb]);
            Some(dot3(n, q) / den)
        } else {
            let (xr, yu) = ((px - self.cx) / self.focal, -(py - self.cy) / self.focal);
            let dir = [self.f[0] + xr * self.r[0] + yu * self.u[0], self.f[1] + xr * self.r[1] + yu * self.u[1], self.f[2] + xr * self.r[2] + yu * self.u[2]];
            let den = dot3(n, dir);
            if den.abs() < 1e-12 {
                return None;
            }
            Some(dot3(n, sub3(a, self.eye)) / den)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Views: the guide render and its sidecar
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ViewSpec {
    pub name: String,
    /// Degrees turned from the front view (positive turns the camera around the character).
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub ortho: bool,
    pub size: usize,
    /// How much of the view height the model's height fills.
    pub fill: f32,
    pub clip: Option<String>,
    pub time: f32,
}

/// Preset view names: (yaw, pitch) in degrees from the front view.
pub fn preset(name: &str) -> Option<(f32, f32)> {
    Some(match name {
        "front" => (0.0, 0.0),
        "back" => (180.0, 0.0),
        "left" => (90.0, 0.0),
        "right" => (-90.0, 0.0),
        "front-left" => (45.0, 0.0),
        "front-right" => (-45.0, 0.0),
        "back-left" => (135.0, 0.0),
        "back-right" => (-135.0, 0.0),
        "above" => (0.0, 35.0),
        "below" => (0.0, -35.0),
        _ => return None,
    })
}

pub fn make_camera(model: &Model, s: &ViewSpec) -> Camera {
    let mut c = Camera::fit(model, 1.0, s.fill, s.ortho);
    c.yaw = std::f32::consts::PI + s.yaw_deg.to_radians();
    c.pitch = s.pitch_deg.to_radians();
    c
}

pub struct ViewData {
    pub spec: ViewSpec,
    pub cam: Camera,
    pub w: usize,
    pub h: usize,
    pub guide: Img,
    /// Global triangle id per pixel (see `render::tri_offsets`), -1 for background.
    pub ids: Vec<i32>,
    /// Camera depth per pixel, NaN for background.
    pub depth: Vec<f32>,
    pub zmin: f32,
    pub zmax: f32,
}

fn clip_index(model: &Model, clip: &Option<String>) -> Result<Option<usize>, String> {
    match clip {
        Some(n) => model.clips.iter().position(|c| c == n).map(Some).ok_or_else(|| format!("no clip named {n}; clips: {}", model.clips.join(", "))),
        None => Ok(None),
    }
}

/// Renders a view and builds its id and depth buffers.
pub fn render_view(model: &Model, textures: &[Option<crate::Texture>], spec: &ViewSpec) -> Result<ViewData, String> {
    let cam = make_camera(model, spec);
    let pose = model.posed(clip_index(model, &spec.clip)?, spec.time);
    let size = spec.size;
    let o = render::Options { cam, w: size, h: size, background: ([40, 44, 52], [40, 44, 52]), filter: Filter::Smooth, positions: Some(&pose) };
    let frame = render::render(model, textures, &o, &|_, _| None);
    let rig = Rig::new(&cam, size, size);
    let mut depth = vec![f32::NAN; size * size];
    let (mut zmin, mut zmax) = (f32::MAX, f32::MIN);
    for y in 0..size {
        for x in 0..size {
            let id = frame.pick[y * size + x];
            if id < 0 {
                continue;
            }
            let (mi, ti) = split_id(model, id as usize);
            let t = model.meshes[mi].tris[ti];
            let p = &pose[mi];
            let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
            let n = cross3(sub3(b, a), sub3(c, a));
            if let Some(z) = rig.depth_on_plane(x as f32 + 0.5, y as f32 + 0.5, a, n) {
                depth[y * size + x] = z;
                zmin = zmin.min(z);
                zmax = zmax.max(z);
            }
        }
    }
    if zmin > zmax {
        (zmin, zmax) = (0.0, 1.0);
    }
    Ok(ViewData { spec: spec.clone(), cam, w: size, h: size, guide: Img { w: size, h: size, rgba: frame.rgba }, ids: frame.pick, depth, zmin, zmax })
}

/// Writes `NAME-guide.png`, `NAME-ids.png` (RGB = triangle id + 1, 0 = background),
/// `NAME-depth.png` (16-bit, 0 = background) and `NAME-view.json` into `dir`.
pub fn write_view(dir: &Path, v: &ViewData, model_file: &str, model_tris: usize) -> Result<(), String> {
    let n = &v.spec.name;
    let wr = |p: String, b: Vec<u8>| std::fs::write(dir.join(&p), b).map_err(|e| format!("{p}: {e}"));
    wr(format!("{n}-guide.png"), v.guide.to_png()?)?;
    let mut idrgb = image::RgbImage::new(v.w as u32, v.h as u32);
    for (i, &id) in v.ids.iter().enumerate() {
        let k = (id + 1) as u32;
        idrgb.put_pixel((i % v.w) as u32, (i / v.w) as u32, image::Rgb([(k >> 16) as u8, (k >> 8) as u8, k as u8]));
    }
    let mut cur = std::io::Cursor::new(Vec::new());
    idrgb.write_to(&mut cur, image::ImageFormat::Png).map_err(|e| e.to_string())?;
    wr(format!("{n}-ids.png"), cur.into_inner())?;
    let mut d16 = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::new(v.w as u32, v.h as u32);
    let span = (v.zmax - v.zmin).max(1e-9);
    for (i, &z) in v.depth.iter().enumerate() {
        let q = if z.is_nan() { 0 } else { 1 + (((z - v.zmin) / span).clamp(0.0, 1.0) * 65534.0).round() as u32 };
        d16.put_pixel((i % v.w) as u32, (i / v.w) as u32, image::Luma([q.min(65535) as u16]));
    }
    let mut cur = std::io::Cursor::new(Vec::new());
    d16.write_to(&mut cur, image::ImageFormat::Png).map_err(|e| e.to_string())?;
    wr(format!("{n}-depth.png"), cur.into_inner())?;
    let c = &v.cam;
    let meta = json!({
        "name": n, "model": model_file, "triangles": model_tris, "width": v.w, "height": v.h,
        "yaw_deg": v.spec.yaw_deg, "pitch_deg": v.spec.pitch_deg, "ortho": v.spec.ortho, "fill": v.spec.fill,
        "clip": v.spec.clip, "time": v.spec.time,
        "camera": { "yaw": c.yaw, "pitch": c.pitch, "dist": c.dist, "target": c.target, "fov_y": c.fov_y, "ortho": c.ortho, "ortho_half": c.ortho_half },
        "depth": { "zmin": v.zmin, "zmax": v.zmax, "encoding": "16-bit gray, 0 = background, else zmin + (v-1)/65534*(zmax-zmin)" },
        "ids": "RGB bytes of (global triangle id + 1) as a 24-bit number, 0 = background; ids count triangles mesh by mesh",
    });
    wr(format!("{n}-view.json"), serde_json::to_string_pretty(&meta).unwrap().into_bytes())
}

fn jf(v: &Value, k: &str) -> Result<f32, String> {
    v.get(k).and_then(Value::as_f64).map(|x| x as f32).ok_or_else(|| format!("view.json: missing {k}"))
}

/// Reads a view back from its sidecar files.
pub fn read_view(dir: &Path, name: &str) -> Result<ViewData, String> {
    let rd = |p: String| std::fs::read(dir.join(&p)).map_err(|e| format!("{p}: {e}"));
    let meta: Value = serde_json::from_slice(&rd(format!("{name}-view.json"))?).map_err(|e| e.to_string())?;
    let c = meta.get("camera").ok_or("view.json: no camera")?;
    let t = c.get("target").and_then(Value::as_array).ok_or("view.json: no target")?;
    let tv = |i: usize| t.get(i).and_then(Value::as_f64).unwrap_or(0.0) as f32;
    let cam = Camera {
        yaw: jf(c, "yaw")?,
        pitch: jf(c, "pitch")?,
        dist: jf(c, "dist")?,
        target: [tv(0), tv(1), tv(2)],
        fov_y: jf(c, "fov_y")?,
        ortho: c.get("ortho").and_then(Value::as_bool).unwrap_or(false),
        ortho_half: jf(c, "ortho_half")?,
    };
    let d = meta.get("depth").ok_or("view.json: no depth")?;
    let (zmin, zmax) = (jf(d, "zmin")?, jf(d, "zmax")?);
    let guide = Img::decode(&rd(format!("{name}-guide.png"))?)?;
    let idimg = image::load_from_memory(&rd(format!("{name}-ids.png"))?).map_err(|e| e.to_string())?.to_rgb8();
    let dimg = image::load_from_memory(&rd(format!("{name}-depth.png"))?).map_err(|e| e.to_string())?.to_luma16();
    let (w, h) = (idimg.width() as usize, idimg.height() as usize);
    if (guide.w, guide.h) != (w, h) || (dimg.width() as usize, dimg.height() as usize) != (w, h) {
        return Err("view sidecar images differ in size".into());
    }
    let ids: Vec<i32> = idimg.pixels().map(|p| (((p[0] as u32) << 16) | ((p[1] as u32) << 8) | p[2] as u32) as i32 - 1).collect();
    let span = zmax - zmin;
    let depth: Vec<f32> = dimg.pixels().map(|p| if p[0] == 0 { f32::NAN } else { zmin + (p[0] as f32 - 1.0) / 65534.0 * span }).collect();
    let spec = ViewSpec {
        name: name.to_owned(),
        yaw_deg: jf(&meta, "yaw_deg")?,
        pitch_deg: jf(&meta, "pitch_deg")?,
        ortho: cam.ortho,
        size: w,
        fill: jf(&meta, "fill").unwrap_or(0.8),
        clip: meta.get("clip").and_then(Value::as_str).map(str::to_owned),
        time: jf(&meta, "time").unwrap_or(0.0),
    };
    Ok(ViewData { spec, cam, w, h, guide, ids, depth, zmin, zmax })
}

// ---------------------------------------------------------------------------------------------
// Registration: align a painted picture of a view to its render
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RegOptions {
    /// The scale must stay within 1 +- this.
    pub scale_tol: f64,
    pub rot_tol_deg: f64,
    /// Smallest edge-map correlation after alignment.
    pub min_ncc: f64,
    /// Largest shift as a share of the image size.
    pub max_shift: f64,
    /// The painted picture's aspect ratio may differ from the render's by this share.
    pub aspect_tol: f64,
}

impl Default for RegOptions {
    fn default() -> Self {
        RegOptions { scale_tol: 0.03, rot_tol_deg: 6.0, min_ncc: 0.35, max_shift: 0.2, aspect_tol: 0.01 }
    }
}

/// Similarity transform from guide pixels to painted pixels (in the painted picture's own
/// pixel size divided by `k`): p = R(x - c) + c + t, with R = [[a, -b], [b, a]].
#[derive(Clone, Debug)]
pub struct Registration {
    pub a: f64,
    pub b: f64,
    pub tx: f64,
    pub ty: f64,
    pub w: usize,
    pub h: usize,
    /// Painted pixels per guide pixel.
    pub k: f32,
    pub scale: f64,
    pub rot_deg: f64,
    pub ncc_before: f64,
    pub ncc_after: f64,
    pub levels: usize,
}

impl Registration {
    pub fn identity(w: usize, h: usize, k: f32) -> Registration {
        Registration { a: 1.0, b: 0.0, tx: 0.0, ty: 0.0, w, h, k, scale: 1.0, rot_deg: 0.0, ncc_before: f64::NAN, ncc_after: f64::NAN, levels: 0 }
    }
    /// Where guide point (x, y) lies in the painted picture, in guide-sized units.
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let (cx, cy) = (self.w as f64 * 0.5, self.h as f64 * 0.5);
        let (xc, yc) = (x as f64 - cx, y as f64 - cy);
        ((self.a * xc - self.b * yc + self.tx + cx) as f32, (self.b * xc + self.a * yc + self.ty + cy) as f32)
    }
    pub fn to_json(&self) -> Value {
        json!({ "a": self.a, "b": self.b, "tx": self.tx, "ty": self.ty, "scale": self.scale, "rotation_deg": self.rot_deg,
                "paint_pixels_per_guide_pixel": self.k, "edge_ncc_before": self.ncc_before, "edge_ncc_after": self.ncc_after, "levels": self.levels,
                "guide_size": [self.w, self.h], "maps": "guide pixel x -> painted pixel (R(x - c) + c + t) * k, c = guide center" })
    }
}

fn gray_of(img: &Img) -> Vec<f32> {
    img.rgba.chunks_exact(4).map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).collect()
}

fn blur(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let r = (sigma * 3.0).ceil() as i64;
    let k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    let k: Vec<f32> = k.iter().map(|v| v / s).collect();
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut a = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let xx = (x as i64 + i as i64 - r).clamp(0, w as i64 - 1) as usize;
                a += kv * src[y * w + xx];
            }
            tmp[y * w + x] = a;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut a = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let yy = (y as i64 + i as i64 - r).clamp(0, h as i64 - 1) as usize;
                a += kv * tmp[yy * w + x];
            }
            out[y * w + x] = a;
        }
    }
    out
}

fn halve(src: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
    let mut out = vec![0.0f32; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let mut s = 0.0;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                s += src[((2 * y + dy).min(h - 1)) * w + (2 * x + dx).min(w - 1)];
            }
            out[y * nw + x] = s * 0.25;
        }
    }
    (out, nw, nh)
}

fn grad_mag(g: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let gx = g[y * w + (x + 1).min(w - 1)] - g[y * w + x.saturating_sub(1)];
            let gy = g[(y + 1).min(h - 1) * w + x] - g[y.saturating_sub(1) * w + x];
            m[y * w + x] = 0.5 * (gx * gx + gy * gy).sqrt();
        }
    }
    m
}

struct Level {
    w: usize,
    h: usize,
    eg: Vec<f32>,
    ep: Vec<f32>,
    gx: Vec<f32>,
    gy: Vec<f32>,
    mask: Vec<bool>,
}

fn bil(a: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let (fx, fy) = (x - 0.5, y - 0.5);
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0, fy - y0);
    let cx = |v: f32| (v as i64).clamp(0, w as i64 - 1) as usize;
    let cy = |v: f32| (v as i64).clamp(0, h as i64 - 1) as usize;
    let (xa, xb, ya, yb) = (cx(x0), cx(x0 + 1.0), cy(y0), cy(y0 + 1.0));
    a[ya * w + xa] * (1.0 - tx) * (1.0 - ty) + a[ya * w + xb] * tx * (1.0 - ty) + a[yb * w + xa] * (1.0 - tx) * ty + a[yb * w + xb] * tx * ty
}

fn ncc_level(l: &Level, p: &[f64; 4]) -> f64 {
    let (w, h) = (l.w as f64, l.h as f64);
    let (mut n, mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
    for y in 0..l.h {
        for x in 0..l.w {
            let i = y * l.w + x;
            if !l.mask[i] {
                continue;
            }
            let (xc, yc) = (x as f64 + 0.5 - w * 0.5, y as f64 + 0.5 - h * 0.5);
            let px = p[0] * xc - p[1] * yc + p[2] + w * 0.5;
            let py = p[1] * xc + p[0] * yc + p[3] + h * 0.5;
            let a = l.eg[i] as f64;
            let b = bil(&l.ep, l.w, l.h, px as f32, py as f32) as f64;
            n += 1.0;
            sa += a;
            sb += b;
            saa += a * a;
            sbb += b * b;
            sab += a * b;
        }
    }
    if n < 16.0 {
        return 0.0;
    }
    let (ma, mb) = (sa / n, sb / n);
    let (va, vb, cv) = (saa / n - ma * ma, sbb / n - mb * mb, sab / n - ma * mb);
    if va < 1e-9 || vb < 1e-9 {
        return 0.0;
    }
    cv / (va * vb).sqrt()
}

fn solve4(mut m: [[f64; 5]; 4]) -> Option<[f64; 4]> {
    for c in 0..4 {
        let piv = (c..4).max_by(|&a, &b| m[a][c].abs().total_cmp(&m[b][c].abs()))?;
        if m[piv][c].abs() < 1e-12 {
            return None;
        }
        m.swap(c, piv);
        for r in 0..4 {
            if r != c {
                let f = m[r][c] / m[c][c];
                for k in c..5 {
                    m[r][k] -= f * m[c][k];
                }
            }
        }
    }
    Some([m[0][4] / m[0][0], m[1][4] / m[1][1], m[2][4] / m[2][2], m[3][4] / m[3][3]])
}

fn lk_level(l: &Level, p: &mut [f64; 4], iters: usize) {
    let (w, h) = (l.w as f64, l.h as f64);
    let half = (w * w + h * h).sqrt() * 0.5;
    for _ in 0..iters {
        let mut res: Vec<(usize, f32, f32, f32, f32, f32)> = Vec::new(); // index, r, gx, gy, xc, yc
        for y in 0..l.h {
            for x in 0..l.w {
                let i = y * l.w + x;
                if !l.mask[i] {
                    continue;
                }
                let (xc, yc) = (x as f64 + 0.5 - w * 0.5, y as f64 + 0.5 - h * 0.5);
                let px = (p[0] * xc - p[1] * yc + p[2] + w * 0.5) as f32;
                let py = (p[1] * xc + p[0] * yc + p[3] + h * 0.5) as f32;
                if px < 1.5 || py < 1.5 || px > l.w as f32 - 1.5 || py > l.h as f32 - 1.5 {
                    continue;
                }
                let r = bil(&l.ep, l.w, l.h, px, py) - l.eg[i];
                res.push((i, r, bil(&l.gx, l.w, l.h, px, py), bil(&l.gy, l.w, l.h, px, py), xc as f32, yc as f32));
            }
        }
        if res.len() < 32 {
            return;
        }
        let mut absr: Vec<f32> = res.iter().map(|e| e.1.abs()).collect();
        let mid = absr.len() / 2;
        let (_, med, _) = absr.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        let c = (1.5 * *med).max(0.5);
        let mut nm = [[0.0f64; 5]; 4];
        for &(_, r, gx, gy, xc, yc) in &res {
            let wt = 1.0 / (1.0 + (r / c) * (r / c)) as f64;
            let j = [(gx * xc + gy * yc) as f64, (-gx * yc + gy * xc) as f64, gx as f64, gy as f64];
            for a in 0..4 {
                for b in 0..4 {
                    nm[a][b] += wt * j[a] * j[b];
                }
                nm[a][4] -= wt * j[a] * r as f64;
            }
        }
        let tr = nm[0][0] + nm[1][1] + nm[2][2] + nm[3][3];
        for a in 0..4 {
            nm[a][a] += 1e-6 * tr + 1e-9;
        }
        let Some(d) = solve4(nm) else { return };
        for k in 0..4 {
            p[k] += d[k];
        }
        let step = (d[0].abs().max(d[1].abs()) * half).max(d[2].abs()).max(d[3].abs());
        if step < 0.01 {
            return;
        }
    }
}

/// Aligns `painted` (any size with the render's aspect ratio) to `guide`. `fg` marks the pixels the
/// render covers; the alignment looks only a little beyond them.
pub fn register(guide: &Img, fg: &[bool], painted: &Img, o: &RegOptions) -> Result<Registration, String> {
    let (gw, gh) = (guide.w, guide.h);
    let (kx, ky) = (painted.w as f64 / gw as f64, painted.h as f64 / gh as f64);
    if (kx / ky - 1.0).abs() > o.aspect_tol {
        return Err(format!("registration refused: the painted picture is {} x {} but the render is {} x {}; the shapes differ", painted.w, painted.h, gw, gh));
    }
    let pg = if (painted.w, painted.h) == (gw, gh) { painted.clone() } else { painted.resized(gw, gh) };
    let g0 = gray_of(guide);
    let p0 = gray_of(&pg);
    // Foreground mask grown by 10 px.
    let mut m0: Vec<bool> = fg.to_vec();
    for _ in 0..2 {
        let mut grown = m0.clone();
        for y in 0..gh {
            for x in 0..gw {
                if m0[y * gw + x] {
                    continue;
                }
                'o: for dy in [-5i64, 0, 5] {
                    for dx in [-5i64, 0, 5] {
                        let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                        if xx >= 0 && yy >= 0 && (xx as usize) < gw && (yy as usize) < gh && m0[yy as usize * gw + xx as usize] {
                            grown[y * gw + x] = true;
                            break 'o;
                        }
                    }
                }
            }
        }
        m0 = grown;
    }
    // Pyramid down to about 32 px.
    let mut levels: Vec<Level> = Vec::new();
    let (mut gg, mut pp, mut mm, mut w, mut h) = (g0, p0, m0.iter().map(|&b| if b { 1.0f32 } else { 0.0 }).collect::<Vec<f32>>(), gw, gh);
    loop {
        let eg = grad_mag(&blur(&gg, w, h, 1.0), w, h);
        let ep = grad_mag(&blur(&pp, w, h, 1.0), w, h);
        let mut gx = vec![0.0f32; w * h];
        let mut gy = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                gx[y * w + x] = 0.5 * (ep[y * w + (x + 1).min(w - 1)] - ep[y * w + x.saturating_sub(1)]);
                gy[y * w + x] = 0.5 * (ep[(y + 1).min(h - 1) * w + x] - ep[y.saturating_sub(1) * w + x]);
            }
        }
        levels.push(Level { w, h, eg, ep, gx, gy, mask: mm.iter().map(|&v| v > 0.0).collect() });
        if w.min(h) < 48 || levels.len() >= 6 {
            break;
        }
        let (g2, nw, nh) = halve(&gg, w, h);
        let (p2, _, _) = halve(&pp, w, h);
        let (m2, _, _) = halve(&mm, w, h);
        (gg, pp, mm, w, h) = (g2, p2, m2, nw, nh);
    }
    let nl = levels.len();
    let mut p = [1.0f64, 0.0, 0.0, 0.0];
    // Exhaustive shifts at the coarsest level.
    {
        let l = &levels[nl - 1];
        let r = (l.w.min(l.h) as i64 / 4).max(4);
        let mut best = (f64::MIN, 0i64, 0i64);
        for dy in -r..=r {
            for dx in -r..=r {
                let s = ncc_level(l, &[1.0, 0.0, dx as f64, dy as f64]);
                if s > best.0 {
                    best = (s, dx, dy);
                }
            }
        }
        p[2] = best.1 as f64;
        p[3] = best.2 as f64;
    }
    for li in (0..nl).rev() {
        let iters = if li == 0 { 12 } else { 30 };
        lk_level(&levels[li], &mut p, iters);
        if li > 0 {
            p[2] *= 2.0;
            p[3] *= 2.0;
        }
    }
    let ncc_before = ncc_level(&levels[0], &[1.0, 0.0, 0.0, 0.0]);
    let ncc_after = ncc_level(&levels[0], &p);
    let scale = (p[0] * p[0] + p[1] * p[1]).sqrt();
    let rot = p[1].atan2(p[0]).to_degrees();
    let reg = Registration { a: p[0], b: p[1], tx: p[2], ty: p[3], w: gw, h: gh, k: kx as f32, scale, rot_deg: rot, ncc_before, ncc_after, levels: nl };
    if (scale - 1.0).abs() > o.scale_tol {
        return Err(format!("registration refused: scale {scale:.4} is outside 1 +- {} (the painted picture does not match this view)", o.scale_tol));
    }
    if rot.abs() > o.rot_tol_deg {
        return Err(format!("registration refused: rotation {rot:.2} deg is beyond {} deg", o.rot_tol_deg));
    }
    if p[2].abs() > o.max_shift * gw as f64 || p[3].abs() > o.max_shift * gh as f64 {
        return Err(format!("registration refused: shift ({:.1}, {:.1}) px is more than {:.0}% of the view", p[2], p[3], o.max_shift * 100.0));
    }
    if ncc_after < o.min_ncc {
        return Err(format!("registration refused: edge correlation {ncc_after:.3} is below {} (the painted picture does not look like this view)", o.min_ncc));
    }
    Ok(reg)
}

/// The painted picture brought into the render's frame, at the render's size.
pub fn warp_to_guide(painted: &Img, reg: &Registration) -> Img {
    let mut out = Img::new(reg.w, reg.h, [0, 0, 0, 255]);
    for y in 0..reg.h {
        for x in 0..reg.w {
            let (px, py) = reg.apply(x as f32 + 0.5, y as f32 + 0.5);
            let c = painted.bilinear(px * reg.k, py * reg.k);
            let o = (y * reg.w + x) * 4;
            for k in 0..4 {
                out.rgba[o + k] = c[k].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Resamples `src` through the inverse of `m` so that dst(m x) = src(x): used to make a test picture
/// from a render with a known misregistration.
pub fn warp_forward(src: &Img, m: &Registration, out_w: usize, out_h: usize) -> Img {
    // dst(p) = src(M^-1 p), M = R x + t about the center.
    let det = m.a * m.a + m.b * m.b;
    let (cx, cy) = (src.w as f64 * 0.5, src.h as f64 * 0.5);
    let mut out = Img::new(out_w, out_h, [0, 0, 0, 255]);
    for y in 0..out_h {
        for x in 0..out_w {
            let (px, py) = (x as f64 + 0.5 - cx - m.tx, y as f64 + 0.5 - cy - m.ty);
            let sx = (m.a * px + m.b * py) / det + cx;
            let sy = (-m.b * px + m.a * py) / det + cy;
            let c = src.bilinear(sx as f32, sy as f32);
            let o = (y * out_w + x) * 4;
            for k in 0..4 {
                out.rgba[o + k] = c[k].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Masks
// ---------------------------------------------------------------------------------------------

/// What may change. A triangle outside `tri_allowed`, or a texel where a texel mask says 0, never changes.
pub struct Masks {
    /// Per global triangle.
    pub tri_allowed: Vec<bool>,
    /// Texel masks per image name (without .png): (w, h, allowed). Missing = everything allowed.
    pub texel: HashMap<String, (usize, usize, Vec<bool>)>,
}

impl Masks {
    pub fn all(model: &Model) -> Masks {
        Masks { tri_allowed: vec![true; model.triangle_count()], texel: HashMap::new() }
    }
}

/// Triangles whose main bone belongs to one of `parts`, limited to the named meshes when given.
pub fn part_mask(model: &Model, parts: &[Part], meshes: &[String]) -> Vec<bool> {
    let mut v = Vec::with_capacity(model.triangle_count());
    for (mi, m) in model.meshes.iter().enumerate() {
        let mesh_ok = meshes.is_empty() || meshes.iter().any(|n| n == &m.name);
        for ti in 0..m.tris.len() {
            let ok = mesh_ok && {
                let j = model.triangle_joint(mi, ti);
                let name = model.joint_names.get(j).map(String::as_str).unwrap_or("");
                parts.contains(&classify(name).0)
            };
            v.push(ok);
        }
    }
    v
}

// ---------------------------------------------------------------------------------------------
// Topology: welded vertices, seams, islands
// ---------------------------------------------------------------------------------------------

/// One mesh edge whose two triangles sit apart in UV space. UVs are 0..1; `a` and `b` list the
/// edge's two ends in the same 3D order.
#[derive(Clone, Debug)]
pub struct SeamEdge {
    pub image: usize,
    pub ta: usize,
    pub tb: usize,
    pub a: [[f32; 2]; 2],
    pub b: [[f32; 2]; 2],
}

pub struct Topology {
    pub seams: Vec<SeamEdge>,
    /// UV island per global triangle (triangles joined by edges whose UVs agree).
    pub island: Vec<u32>,
    /// Welded vertex ids per global triangle.
    pub weld: Vec<[u32; 3]>,
    pub edges_continuous: usize,
    pub edges_open: usize,
    pub edges_ambiguous: usize,
}

fn find(p: &mut Vec<u32>, mut x: u32) -> u32 {
    while p[x as usize] != x {
        p[x as usize] = p[p[x as usize] as usize];
        x = p[x as usize];
    }
    x
}

/// Welds vertices by position inside each mesh and pairs triangle edges. `eps_uv` is the UV distance
/// under which two ends count as the same place.
pub fn topology(model: &Model, eps_texels: &dyn Fn(usize) -> f32) -> Topology {
    let (lo, hi) = render::bounds(model);
    let diag = sub3(hi, lo);
    let q = (dot3(diag, diag).sqrt() * 1e-5).max(1e-9);
    let offsets = tri_offsets(model);
    let ntri = model.triangle_count();
    let mut weld = vec![[0u32; 3]; ntri];
    let mut ids: HashMap<(usize, [i64; 3]), u32> = HashMap::new();
    for (mi, m) in model.meshes.iter().enumerate() {
        for (ti, t) in m.tris.iter().enumerate() {
            for k in 0..3 {
                let p = m.pos[t[k] as usize];
                let key = (mi, [(p[0] / q).round() as i64, (p[1] / q).round() as i64, (p[2] / q).round() as i64]);
                let n = ids.len() as u32;
                weld[offsets[mi] + ti][k] = *ids.entry(key).or_insert(n);
            }
        }
    }
    let mut parent: Vec<u32> = (0..ntri as u32).collect();
    // Edge key -> list of (global tri, corner a, corner b).
    let mut edges: HashMap<(u32, u32), Vec<(usize, usize, usize)>> = HashMap::new();
    let mut seen_tri: HashMap<[u32; 3], usize> = HashMap::new();
    let mut dup = vec![false; ntri];
    for g in 0..ntri {
        let mut s = weld[g];
        s.sort_unstable();
        if s[0] == s[1] || s[1] == s[2] {
            dup[g] = true; // degenerate in 3D
            continue;
        }
        if seen_tri.contains_key(&s) {
            dup[g] = true; // a back face or twin layer: it reads its own texels but pairs with no edge
            continue;
        }
        seen_tri.insert(s, g);
        for k in 0..3 {
            let (ca, cb) = (k, (k + 1) % 3);
            let (wa, wb) = (weld[g][ca], weld[g][cb]);
            edges.entry((wa.min(wb), wa.max(wb))).or_default().push((g, ca, cb));
        }
    }
    let mut seams = Vec::new();
    let (mut cont, mut open, mut amb) = (0, 0, 0);
    let loc = |g: usize| split_id(model, g);
    for (key, list) in &edges {
        if list.len() == 1 {
            open += 1;
            continue;
        }
        if list.len() != 2 {
            amb += 1;
            continue;
        }
        let (t1, t2) = (list[0], list[1]);
        let uvs = |t: (usize, usize, usize)| -> ([f32; 2], [f32; 2], Option<usize>) {
            let (mi, ti) = loc(t.0);
            let m = &model.meshes[mi];
            let tri = m.tris[ti];
            let (ua, ub) = (m.uv[tri[t.1] as usize], m.uv[tri[t.2] as usize]);
            // Order the ends by the weld key so both triangles list them the same way.
            if weld[t.0][t.1] == key.0 {
                (ua, ub, m.image)
            } else {
                (ub, ua, m.image)
            }
        };
        let (a0, a1, im1) = uvs(t1);
        let (b0, b1, im2) = uvs(t2);
        let (Some(im1), Some(im2)) = (im1, im2) else { continue };
        if im1 != im2 {
            continue;
        }
        let e = eps_texels(im1);
        if e <= 0.0 {
            continue;
        }
        let near = |p: [f32; 2], q: [f32; 2]| (p[0] - q[0]).abs() < e && (p[1] - q[1]).abs() < e;
        if near(a0, b0) && near(a1, b1) {
            cont += 1;
            let (ra, rb) = (find(&mut parent, t1.0 as u32), find(&mut parent, t2.0 as u32));
            if ra != rb {
                parent[ra as usize] = rb;
            }
        } else {
            seams.push(SeamEdge { image: im1, ta: t1.0, tb: t2.0, a: [a0, a1], b: [b0, b1] });
        }
    }
    let island: Vec<u32> = (0..ntri as u32).map(|g| find(&mut parent, g)).collect();
    // Back faces and twin layers join the island of the triangle they repeat.
    let mut island = island;
    for g in 0..ntri {
        if dup[g] {
            let mut s = weld[g];
            s.sort_unstable();
            if let Some(&first) = seen_tri.get(&s) {
                island[g] = island[first];
            }
        }
    }
    // Deterministic order for reports.
    seams.sort_by_key(|s| (s.ta, s.tb));
    Topology { seams, island, weld, edges_continuous: cont, edges_open: open, edges_ambiguous: amb }
}

// ---------------------------------------------------------------------------------------------
// The bake
// ---------------------------------------------------------------------------------------------

pub struct PaintView {
    pub view: ViewData,
    /// The painted picture at its own size.
    pub paint: Img,
    pub reg: Registration,
    /// Per guide pixel, how much the paint differs from the render (0 = same, 1 = edited).
    pub edit: Vec<f32>,
}

/// How much each pixel of the registered paint differs from the render. With `gate` 0 everything counts.
pub fn edit_map(guide: &Img, registered: &Img, gate: f32) -> Vec<f32> {
    let n = guide.w * guide.h;
    if gate <= 0.0 {
        return vec![1.0; n];
    }
    let mut d = vec![0.0f32; n];
    for i in 0..n {
        let mut m = 0.0f32;
        for c in 0..3 {
            m = m.max((guide.rgba[i * 4 + c] as f32 - registered.rgba[i * 4 + c] as f32).abs());
        }
        d[i] = m;
    }
    let (w, h) = (guide.w, guide.h);
    let mut out = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (xx, yy) = ((x as i64 + dx).clamp(0, w as i64 - 1) as usize, (y as i64 + dy).clamp(0, h as i64 - 1) as usize);
                    s += d[yy * w + xx];
                }
            }
            out[y * w + x] = smooth01((s / 9.0 - gate) / gate);
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct BakeOptions {
    /// Gutter width in texels for the padding pass (0 = none).
    pub pad: usize,
    /// Faces turned further than this from the camera are never baked.
    pub grazing_max_deg: f32,
    /// Faces turned less than this get full weight.
    pub grazing_full_deg: f32,
    /// A texel changes only when its best weight reaches this.
    pub min_weight: f32,
    pub seam_pass: bool,
    pub seam_radius: f32,
    /// Largest color move per seam sample in each side, in 0..255.
    pub seam_max_delta: f32,
    /// Texels covered by a triangle outside the mask are never changed unless this is set.
    pub allow_shared: bool,
    /// Depth tolerance as a share of the model's size.
    pub depth_eps: f32,
    /// Fill texels no view reached (and blend low-confidence texels toward their surroundings) with a
    /// push-pull fill that stays inside each island, before the seam and gutter passes.
    pub fill: bool,
    /// Views blend with weight cos(angle)^p on top of the angle ramp: 0 = every straight-on view counts the same,
    /// larger values let the straightest view dominate (sharper, less averaging of slightly misaligned views).
    pub sharp_power: f32,
}

impl Default for BakeOptions {
    fn default() -> Self {
        BakeOptions { pad: 16, grazing_max_deg: 78.0, grazing_full_deg: 50.0, min_weight: 0.02, seam_pass: true, seam_radius: 8.0, seam_max_delta: 40.0, allow_shared: false, depth_eps: 0.0015, fill: false, sharp_power: 0.0 }
    }
}

pub struct AtlasOut {
    pub image: usize,
    pub name: String,
    pub w: usize,
    pub h: usize,
    /// The input atlas.
    pub orig: Vec<u8>,
    /// After baking, before the seam and gutter passes.
    pub baked: Vec<u8>,
    /// Final atlas.
    pub fin: Vec<u8>,
    /// Texels covered only by allowed triangles (and the texel mask).
    pub perm: Vec<bool>,
    /// Gutter texels the padding pass was allowed to write.
    pub gutter_ok: Vec<bool>,
    /// Bake strength per texel (0 = untouched by paint).
    pub strength: Vec<f32>,
    pub views_seen: Vec<u8>,
    pub stats: Value,
    /// Triangle that owns each texel (global id + 1, 0 = none).
    pub owner: Vec<u32>,
    /// Pure weighted paint average (RGB); alpha = bake strength * 255. Unpainted texels keep the input color.
    pub paint_avg: Vec<u8>,
    /// Cosine (x 255) between the surface normal and the view direction of the straightest view that painted the texel.
    pub cos_best: Vec<u8>,
    /// Texels the fill step wrote because no view reached them.
    pub filled: Vec<bool>,
    /// Pixel positions of the valid seam samples (the two sides of each pair).
    pub seam_pairs: Vec<([f32; 2], [f32; 2])>,
}

pub struct BakeOut {
    pub atlases: Vec<AtlasOut>,
    pub seam: Value,
    pub views: Value,
}

pub fn raster_tri<F: FnMut(usize, usize, [f32; 3])>(p: [[f32; 2]; 3], w: usize, h: usize, mut f: F) {
    let minx = p[0][0].min(p[1][0]).min(p[2][0]);
    let maxx = p[0][0].max(p[1][0]).max(p[2][0]);
    let miny = p[0][1].min(p[1][1]).min(p[2][1]);
    let maxy = p[0][1].max(p[1][1]).max(p[2][1]);
    if !(minx.is_finite() && maxx.is_finite() && miny.is_finite() && maxy.is_finite()) {
        return;
    }
    let x0 = (minx - 0.5).floor().max(0.0) as usize;
    let y0 = (miny - 0.5).floor().max(0.0) as usize;
    let x1 = ((maxx - 0.5).ceil().max(0.0) as usize).min(w.saturating_sub(1));
    let y1 = ((maxy - 0.5).ceil().max(0.0) as usize).min(h.saturating_sub(1));
    let den = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]);
    if den.abs() < 1e-9 {
        return;
    }
    for y in y0..=y1 {
        let py = y as f32 + 0.5;
        for x in x0..=x1 {
            let px = x as f32 + 0.5;
            let b1 = ((px - p[0][0]) * (p[2][1] - p[0][1]) - (py - p[0][1]) * (p[2][0] - p[0][0])) / den;
            let b2 = ((p[1][0] - p[0][0]) * (py - p[0][1]) - (p[1][1] - p[0][1]) * (px - p[0][0])) / den;
            let b0 = 1.0 - b1 - b2;
            if b0 >= -1e-5 && b1 >= -1e-5 && b2 >= -1e-5 {
                f(x, y, [b0, b1, b2]);
            }
        }
    }
}

fn tri_locs(model: &Model) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(model.triangle_count());
    for (mi, m) in model.meshes.iter().enumerate() {
        for ti in 0..m.tris.len() {
            v.push((mi, ti));
        }
    }
    v
}

fn atlas_name(model: &Model, image: usize) -> String {
    model.images[image].name.trim_end_matches(".png").to_owned()
}

/// Bakes the painted views onto every atlas the allowed triangles use.
pub fn bake(model: &Model, views: &[PaintView], masks: &Masks, o: &BakeOptions) -> Result<BakeOut, String> {
    let ntri = model.triangle_count();
    if masks.tri_allowed.len() != ntri {
        return Err("triangle mask does not match the model".into());
    }
    let loc = tri_locs(model);
    let (lo, hi) = render::bounds(model);
    let ext = sub3(hi, lo);
    let diag = dot3(ext, ext).sqrt();
    // Atlases used by meshes with an allowed triangle.
    let mut image_ids: Vec<usize> = Vec::new();
    for (g, &(mi, _)) in loc.iter().enumerate() {
        if masks.tri_allowed[g] {
            if let Some(im) = model.meshes[mi].image {
                if !image_ids.contains(&im) {
                    image_ids.push(im);
                }
            }
        }
    }
    image_ids.sort_unstable();
    let mut imgs: HashMap<usize, Img> = HashMap::new();
    for &im in &image_ids {
        imgs.insert(im, Img::decode(&model.images[im].bytes).map_err(|e| format!("atlas {}: {e}", model.images[im].name))?);
    }
    // Every atlas the model has, for the topology's tolerance.
    let mut all_sizes: HashMap<usize, (usize, usize)> = HashMap::new();
    for m in &model.meshes {
        if let Some(im) = m.image {
            if !all_sizes.contains_key(&im) {
                if let Ok(t) = image::load_from_memory(&model.images[im].bytes) {
                    all_sizes.insert(im, (t.width() as usize, t.height() as usize));
                }
            }
        }
    }
    let topo = topology(model, &|im| all_sizes.get(&im).map(|&(w, _)| 0.05 / w as f32).unwrap_or(0.0));
    let poses: Vec<Vec<Vec<V3>>> = views.iter().map(|v| clip_index(model, &v.view.spec.clip).map(|c| model.posed(c, v.view.spec.time))).collect::<Result<_, _>>()?;
    let rigs: Vec<Rig> = views.iter().map(|v| Rig::new(&v.view.cam, v.view.w, v.view.h)).collect();
    let cmin = o.grazing_max_deg.to_radians().cos();
    let cfull = o.grazing_full_deg.to_radians().cos();
    let depth_eps = o.depth_eps * diag;
    let share = |a: usize, b: usize| topo.weld[a].iter().any(|w| topo.weld[b].contains(w));

    let mut atlases = Vec::new();
    let mut view_stats: Vec<(usize, usize)> = vec![(0, 0); views.len()]; // visible, contributing
    let mut seam_json: Vec<Value> = Vec::new();
    for &im in &image_ids {
        let orig = &imgs[&im];
        let (w, h) = (orig.w, orig.h);
        let name = atlas_name(model, im);
        let tris: Vec<usize> = (0..ntri).filter(|&g| model.meshes[loc[g].0].image == Some(im)).collect();
        // Coverage.
        let mut owner = vec![0u32; w * h];
        let mut cov_a = vec![0u8; w * h];
        let mut cov_o = vec![0u8; w * h];
        for &g in &tris {
            let (mi, ti) = loc[g];
            let m = &model.meshes[mi];
            let t = m.tris[ti];
            let p = t.map(|v| [m.uv[v as usize][0] * w as f32, m.uv[v as usize][1] * h as f32]);
            let allowed = masks.tri_allowed[g];
            raster_tri(p, w, h, |x, y, _| {
                let i = y * w + x;
                owner[i] = g as u32 + 1;
                if allowed {
                    cov_a[i] = cov_a[i].saturating_add(1);
                } else {
                    cov_o[i] = cov_o[i].saturating_add(1);
                }
            });
        }
        let tmask = masks.texel.get(&name);
        if let Some((mw, mh, _)) = tmask {
            if (*mw, *mh) != (w, h) {
                return Err(format!("texel mask for {name} is {mw} x {mh} but the atlas is {w} x {h}"));
            }
        }
        let tex_ok = |i: usize| tmask.map(|m| m.2[i]).unwrap_or(true);
        let mut perm = vec![false; w * h];
        let (mut n_perm, mut n_shared_excl, mut n_png_excl) = (0usize, 0usize, 0usize);
        for i in 0..w * h {
            if cov_a[i] > 0 {
                if cov_o[i] > 0 && !o.allow_shared {
                    n_shared_excl += 1;
                } else if !tex_ok(i) {
                    n_png_excl += 1;
                } else {
                    perm[i] = true;
                    n_perm += 1;
                }
            }
        }
        // Accumulate paint.
        let mut acc = vec![0.0f32; w * h * 4];
        let mut wmax = vec![0.0f32; w * h];
        let mut nview = vec![0u8; w * h];
        let mut cmax = vec![0u8; w * h];
        for (vi, pv) in views.iter().enumerate() {
            let (vw, vh) = (pv.view.w, pv.view.h);
            let rig = &rigs[vi];
            let pose = &poses[vi];
            for &g in &tris {
                if !masks.tri_allowed[g] {
                    continue;
                }
                let (mi, ti) = loc[g];
                let m = &model.meshes[mi];
                let t = m.tris[ti];
                let pos = &pose[mi];
                let (pa, pb, pc) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
                let n = norm3(cross3(sub3(pb, pa), sub3(pc, pa)));
                if !(n[0].is_finite() && n[1].is_finite() && n[2].is_finite()) || dot3(n, n) < 0.5 {
                    continue;
                }
                let uvp = t.map(|v| [m.uv[v as usize][0] * w as f32, m.uv[v as usize][1] * h as f32]);
                raster_tri(uvp, w, h, |x, y, bc| {
                    let ti_ = y * w + x;
                    if owner[ti_] != g as u32 + 1 || !perm[ti_] {
                        return;
                    }
                    let p = [
                        bc[0] * pa[0] + bc[1] * pb[0] + bc[2] * pc[0],
                        bc[0] * pa[1] + bc[1] * pb[1] + bc[2] * pc[1],
                        bc[0] * pa[2] + bc[1] * pb[2] + bc[2] * pc[2],
                    ];
                    let cosv = dot3(n, rig.view_dir(p)).abs();
                    if cosv < cmin.max(0.02) {
                        return;
                    }
                    let sp = rig.project(p);
                    if !(sp[0].is_finite() && sp[1].is_finite()) {
                        return;
                    }
                    let (ix, iy) = (sp[0].floor() as i64, sp[1].floor() as i64);
                    if ix < 1 || iy < 1 || ix >= vw as i64 - 1 || iy >= vh as i64 - 1 {
                        return;
                    }
                    let i0 = iy as usize * vw + ix as usize;
                    let id = pv.view.ids[i0];
                    if id < 0 {
                        return;
                    }
                    let pw = rig.pixel_size(sp[2]);
                    let tan = ((1.0 - cosv * cosv).max(0.0).sqrt() / cosv).min(4.0);
                    let tol = depth_eps + pw * (0.75 + 1.5 * tan);
                    if (sp[2] - pv.view.depth[i0]).abs() > tol {
                        return;
                    }
                    if id as usize != g && !share(g, id as usize) {
                        return;
                    }
                    // The four pixels a bilinear sample reads must show the same surface (no silhouette bleed).
                    let (fx, fy) = (sp[0] - 0.5, sp[1] - 0.5);
                    let (bx, by) = (fx.floor() as i64, fy.floor() as i64);
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (xx, yy) = ((bx + dx).clamp(0, vw as i64 - 1) as usize, (by + dy).clamp(0, vh as i64 - 1) as usize);
                        let j = yy * vw + xx;
                        if pv.view.ids[j] < 0 || (pv.view.depth[j] - sp[2]).abs() > 2.0 * tol {
                            return;
                        }
                    }
                    view_stats[vi].0 += 1;
                    let ang = smooth01((cosv - cmin) / (cfull - cmin).max(1e-3));
                    let edit = bil(&pv.edit, vw, vh, sp[0], sp[1]);
                    let (qx, qy) = pv.reg.apply(sp[0], sp[1]);
                    let (sx, sy) = (qx * pv.reg.k, qy * pv.reg.k);
                    if sx < 0.0 || sy < 0.0 || sx > pv.paint.w as f32 || sy > pv.paint.h as f32 {
                        return;
                    }
                    let c = pv.paint.bilinear(sx, sy);
                    let wt = ang * edit * (c[3] / 255.0);
                    if wt <= 0.0 {
                        return;
                    }
                    view_stats[vi].1 += 1;
                    let wb = if o.sharp_power > 0.0 { wt * cosv.powf(o.sharp_power) } else { wt };
                    acc[ti_ * 4] += wb * c[0];
                    acc[ti_ * 4 + 1] += wb * c[1];
                    acc[ti_ * 4 + 2] += wb * c[2];
                    acc[ti_ * 4 + 3] += wb;
                    if wt > wmax[ti_] {
                        wmax[ti_] = wt;
                    }
                    nview[ti_] = nview[ti_].saturating_add(1);
                    let cq = (cosv.clamp(0.0, 1.0) * 255.0) as u8;
                    if cq > cmax[ti_] {
                        cmax[ti_] = cq;
                    }
                });
            }
        }
        // Blend into the atlas.
        let mut baked = orig.rgba.clone();
        let mut strength = vec![0.0f32; w * h];
        for i in 0..w * h {
            let ws = acc[i * 4 + 3];
            if ws <= 0.0 || wmax[i] < o.min_weight {
                continue;
            }
            let a = wmax[i].min(1.0);
            for c in 0..3 {
                let avg = acc[i * 4 + c] / ws;
                let old = orig.rgba[i * 4 + c] as f32;
                baked[i * 4 + c] = (old * (1.0 - a) + avg * a).round().clamp(0.0, 255.0) as u8;
            }
            strength[i] = a;
        }
        // Pure paint average, for the fill step and for callers that want the paint without the old color.
        let mut paint_avg = vec![0u8; w * h * 4];
        for i in 0..w * h {
            let ws = acc[i * 4 + 3];
            for c in 0..3 {
                paint_avg[i * 4 + c] = if ws > 0.0 && strength[i] > 0.0 { (acc[i * 4 + c] / ws).round().clamp(0.0, 255.0) as u8 } else { orig.rgba[i * 4 + c] };
            }
            paint_avg[i * 4 + 3] = (strength[i] * 255.0).round() as u8;
        }
        // Fill pass: push-pull inside each island. Known texels weigh as much as their bake strength.
        let mut filled = vec![false; w * h];
        let mut fill_json = Value::Null;
        if o.fill {
            let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
            for i in 0..w * h {
                if perm[i] && owner[i] > 0 {
                    groups.entry(topo.island[owner[i] as usize - 1]).or_default().push(i);
                }
            }
            let (mut n_holes, mut n_filled, mut n_blended, mut n_islands_no_source, mut n_unfilled) = (0usize, 0usize, 0usize, 0usize, 0usize);
            for (_, texels) in groups.iter() {
                let any_src = texels.iter().any(|&i| strength[i] > 0.0);
                let holes = texels.iter().filter(|&&i| strength[i] <= 0.0).count();
                n_holes += holes;
                if !any_src {
                    n_islands_no_source += 1;
                    n_unfilled += holes;
                    continue;
                }
                let col = |i: usize| -> [f32; 3] {
                    let d = acc[i * 4 + 3].max(1e-9);
                    [acc[i * 4] / d, acc[i * 4 + 1] / d, acc[i * 4 + 2] / d]
                };
                let conf = |i: usize| strength[i];
                if let Some(f) = pushpull_island(w, texels, &col, &conf) {
                    for (k, &i) in texels.iter().enumerate() {
                        for c in 0..3 {
                            baked[i * 4 + c] = f[k][c].round().clamp(0.0, 255.0) as u8;
                        }
                        if strength[i] <= 0.0 {
                            filled[i] = true;
                            n_filled += 1;
                        } else if strength[i] < 1.0 {
                            n_blended += 1;
                        }
                    }
                }
            }
            fill_json = json!({ "holes": n_holes, "holes_filled": n_filled, "low_confidence_blended": n_blended, "islands_without_any_paint": n_islands_no_source, "holes_left_unfilled": n_unfilled });
        }
        drop(acc);
        // Seam pass.
        let mut fin = baked.clone();
        let island_of = |i: usize| -> Option<u32> { if owner[i] > 0 { Some(topo.island[owner[i] as usize - 1]) } else { None } };
        let atlas_seams: Vec<&SeamEdge> = topo.seams.iter().filter(|s| s.image == im && (masks.tri_allowed[s.ta] || masks.tri_allowed[s.tb])).collect();
        struct Prep<'a> {
            e: &'a SeamEdge,
            sa: [[f32; 2]; 2],
            sb: [[f32; 2]; 2],
            pa: Vec<[f32; 2]>,
            pb: Vec<[f32; 2]>,
            valid: Vec<bool>,
        }
        let to_px = |p: [f32; 2]| [p[0] * w as f32, p[1] * h as f32];
        let tri_center = |g: usize| -> [f32; 2] {
            let (mi, ti) = loc[g];
            let m = &model.meshes[mi];
            let t = m.tris[ti];
            let mut c = [0.0f32; 2];
            for v in t {
                let p = to_px(m.uv[v as usize]);
                c[0] += p[0] / 3.0;
                c[1] += p[1] / 3.0;
            }
            c
        };
        let mut preps: Vec<Prep> = Vec::new();
        for e in &atlas_seams {
            let (a0, a1, b0, b1) = (to_px(e.a[0]), to_px(e.a[1]), to_px(e.b[0]), to_px(e.b[1]));
            let len = ((a1[0] - a0[0]).powi(2) + (a1[1] - a0[1]).powi(2)).sqrt().max(((b1[0] - b0[0]).powi(2) + (b1[1] - b0[1]).powi(2)).sqrt());
            if len < 1.0 {
                continue;
            }
            let ns = (len.ceil() as usize).max(2);
            let side_dir = |s0: [f32; 2], s1: [f32; 2], center: [f32; 2]| -> [f32; 2] {
                let ev = [s1[0] - s0[0], s1[1] - s0[1]];
                let l = (ev[0] * ev[0] + ev[1] * ev[1]).sqrt().max(1e-9);
                let mut p = [-ev[1] / l, ev[0] / l];
                let mid = [(s0[0] + s1[0]) * 0.5, (s0[1] + s1[1]) * 0.5];
                if p[0] * (center[0] - mid[0]) + p[1] * (center[1] - mid[1]) < 0.0 {
                    p = [-p[0], -p[1]];
                }
                p
            };
            let da = side_dir(a0, a1, tri_center(e.ta));
            let db = side_dir(b0, b1, tri_center(e.tb));
            let (mut pa, mut pb, mut valid) = (Vec::with_capacity(ns), Vec::with_capacity(ns), Vec::with_capacity(ns));
            for j in 0..ns {
                let t = (j as f32 + 0.5) / ns as f32;
                let qa = [a0[0] + (a1[0] - a0[0]) * t + da[0] * 0.75, a0[1] + (a1[1] - a0[1]) * t + da[1] * 0.75];
                let qb = [b0[0] + (b1[0] - b0[0]) * t + db[0] * 0.75, b0[1] + (b1[1] - b0[1]) * t + db[1] * 0.75];
                let (ia, ib) = ((qa[1].floor() as i64) * w as i64 + qa[0].floor() as i64, (qb[1].floor() as i64) * w as i64 + qb[0].floor() as i64);
                let inb = |q: [f32; 2]| q[0] >= 0.0 && q[1] >= 0.0 && q[0] < w as f32 && q[1] < h as f32;
                let ok = inb(qa) && inb(qb) && {
                    let (ia, ib) = (ia as usize, ib as usize);
                    perm[ia] && perm[ib] && island_of(ia) == Some(topo.island[e.ta]) && island_of(ib) == Some(topo.island[e.tb]) && (strength[ia] > 0.0 || strength[ib] > 0.0 || filled[ia] || filled[ib])
                };
                pa.push(qa);
                pb.push(qb);
                valid.push(ok);
            }
            preps.push(Prep { e, sa: [a0, a1], sb: [b0, b1], pa, pb, valid });
        }
        let seam_err = |rgba: &[u8]| -> (f64, usize) {
            let (mut s, mut n) = (0.0f64, 0usize);
            for p in &preps {
                for j in 0..p.valid.len() {
                    if !p.valid[j] {
                        continue;
                    }
                    let ca = bilinear_rgba(rgba, w, h, p.pa[j][0], p.pa[j][1]);
                    let cb = bilinear_rgba(rgba, w, h, p.pb[j][0], p.pb[j][1]);
                    s += ((ca[0] - cb[0]).abs() + (ca[1] - cb[1]).abs() + (ca[2] - cb[2]).abs()) as f64 / 3.0;
                    n += 1;
                }
            }
            (if n > 0 { s / n as f64 } else { 0.0 }, n)
        };
        let (err_input, n_samples) = seam_err(&orig.rgba);
        let (err_baked, _) = seam_err(&baked);
        let mut seam_texels_moved = 0usize;
        if o.seam_pass && !preps.is_empty() {
            let mut dsum = vec![0.0f32; w * h * 3];
            let mut wsum = vec![0.0f32; w * h];
            let r = o.seam_radius.max(1.0);
            for p in &preps {
                let ns = p.valid.len();
                let mut delta: Vec<[f32; 3]> = Vec::with_capacity(ns);
                for j in 0..ns {
                    let ca = bilinear_rgba(&baked, w, h, p.pa[j][0], p.pa[j][1]);
                    let cb = bilinear_rgba(&baked, w, h, p.pb[j][0], p.pb[j][1]);
                    delta.push(std::array::from_fn(|c| ((cb[c] - ca[c]) * 0.5).clamp(-o.seam_max_delta, o.seam_max_delta)));
                }
                for (seg, tri, sign) in [(p.sa, p.e.ta, 1.0f32), (p.sb, p.e.tb, -1.0f32)] {
                    let isl = topo.island[tri];
                    let ev = [seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]];
                    let ee = (ev[0] * ev[0] + ev[1] * ev[1]).max(1e-9);
                    let (minx, maxx) = (seg[0][0].min(seg[1][0]) - r, seg[0][0].max(seg[1][0]) + r);
                    let (miny, maxy) = (seg[0][1].min(seg[1][1]) - r, seg[0][1].max(seg[1][1]) + r);
                    let (x0, x1) = (minx.floor().max(0.0) as usize, (maxx.ceil() as usize).min(w - 1));
                    let (y0, y1) = (miny.floor().max(0.0) as usize, (maxy.ceil() as usize).min(h - 1));
                    // Which side of the edge the triangle lies on.
                    let cen = tri_center(tri);
                    let nrm = [-ev[1], ev[0]];
                    let side_sign = if nrm[0] * (cen[0] - seg[0][0]) + nrm[1] * (cen[1] - seg[0][1]) >= 0.0 { 1.0f32 } else { -1.0 };
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            let i = y * w + x;
                            if !perm[i] || island_of(i) != Some(isl) {
                                continue;
                            }
                            let q = [x as f32 + 0.5, y as f32 + 0.5];
                            let t = (((q[0] - seg[0][0]) * ev[0] + (q[1] - seg[0][1]) * ev[1]) / ee).clamp(0.0, 1.0);
                            let cp = [seg[0][0] + ev[0] * t, seg[0][1] + ev[1] * t];
                            let dist = ((q[0] - cp[0]).powi(2) + (q[1] - cp[1]).powi(2)).sqrt();
                            if dist >= r {
                                continue;
                            }
                            // Stay on the triangle's own side of the edge.
                            if side_sign * (nrm[0] * (q[0] - seg[0][0]) + nrm[1] * (q[1] - seg[0][1])) / ee.sqrt() < -0.5 {
                                continue;
                            }
                            let j = ((t * ns as f32) as usize).min(ns - 1);
                            if !p.valid[j] {
                                continue;
                            }
                            let wt = smooth01(1.0 - dist / r);
                            for c in 0..3 {
                                dsum[i * 3 + c] += sign * delta[j][c] * wt;
                            }
                            wsum[i] += wt;
                        }
                    }
                }
            }
            for i in 0..w * h {
                if wsum[i] <= 0.0 {
                    continue;
                }
                let div = wsum[i].max(1.0);
                let mut moved = false;
                for c in 0..3 {
                    let v = (fin[i * 4 + c] as f32 + dsum[i * 3 + c] / div).round().clamp(0.0, 255.0) as u8;
                    if v != fin[i * 4 + c] {
                        moved = true;
                    }
                    fin[i * 4 + c] = v;
                }
                if moved {
                    seam_texels_moved += 1;
                }
            }
        }
        let (err_seam, _) = seam_err(&fin);
        // Gutter pass: pad each island outward from its changed texels.
        let mut gutter_ok = vec![false; w * h];
        let mut gutter_written = 0usize;
        if o.pad > 0 {
            let mut src = vec![u32::MAX; w * h];
            let mut frontier: Vec<u32> = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    if owner[i] == 0 {
                        continue;
                    }
                    src[i] = i as u32;
                    let mut edge = false;
                    for dy in -1i64..=1 {
                        for dx in -1i64..=1 {
                            let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                            if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 || owner[yy as usize * w + xx as usize] == 0 {
                                edge = true;
                            }
                        }
                    }
                    if edge {
                        frontier.push(i as u32);
                    }
                }
            }
            let changed_at = |buf: &[u8], i: usize| buf[i * 4..i * 4 + 3] != orig.rgba[i * 4..i * 4 + 3];
            for _ in 0..o.pad {
                let mut next: Vec<u32> = Vec::new();
                for &f in &frontier {
                    let (fx, fy) = ((f as usize % w) as i64, (f as usize / w) as i64);
                    for dy in -1i64..=1 {
                        for dx in -1i64..=1 {
                            let (xx, yy) = (fx + dx, fy + dy);
                            if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                                continue;
                            }
                            let n = yy as usize * w + xx as usize;
                            if src[n] == u32::MAX {
                                src[n] = src[f as usize];
                                next.push(n as u32);
                            }
                        }
                    }
                }
                frontier = next;
            }
            for i in 0..w * h {
                if owner[i] != 0 || src[i] == u32::MAX {
                    continue;
                }
                let s = src[i] as usize;
                if perm[s] && changed_at(&fin, s) && tex_ok(i) {
                    gutter_ok[i] = true;
                    for c in 0..3 {
                        fin[i * 4 + c] = fin[s * 4 + c];
                    }
                    gutter_written += 1;
                }
            }
        }
        // Statistics.
        let (mut chg, mut chg_perm, mut chg_gut, mut chg_out, mut baked_n, mut alpha_chg) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        for i in 0..w * h {
            if strength[i] > 0.0 {
                baked_n += 1;
            }
            if fin[i * 4 + 3] != orig.rgba[i * 4 + 3] {
                alpha_chg += 1;
            }
            if fin[i * 4..i * 4 + 4] != orig.rgba[i * 4..i * 4 + 4] {
                chg += 1;
                if perm[i] {
                    chg_perm += 1;
                } else if gutter_ok[i] {
                    chg_gut += 1;
                } else {
                    chg_out += 1;
                }
            }
        }
        seam_json.push(json!({
            "image": name,
            "seam_edges_in_mask": atlas_seams.len(),
            "seam_edges_used": preps.iter().filter(|p| p.valid.iter().any(|&v| v)).count(),
            "samples": n_samples,
            "mean_rgb_difference_input": err_input,
            "mean_rgb_difference_after_bake": err_baked,
            "mean_rgb_difference_after_seam_pass": err_seam,
            "texels_moved_by_seam_pass": seam_texels_moved,
        }));
        let stats = json!({
            "image": name, "size": [w, h],
            "texels_allowed": n_perm,
            "texels_excluded_shared_with_triangles_outside_mask": n_shared_excl,
            "texels_excluded_by_texel_mask": n_png_excl,
            "texels_painted": baked_n,
            "texels_changed": chg,
            "changed_inside_mask": chg_perm,
            "changed_in_padding_of_allowed_islands": chg_gut,
            "changed_outside_mask": chg_out,
            "padding_texels_written": gutter_written,
            "mip_bleed": [mip_bleed(&orig.rgba, &fin, &perm, &gutter_ok, w, h, 8), mip_bleed(&orig.rgba, &fin, &perm, &gutter_ok, w, h, 16)],
            "alpha_changed": alpha_chg,
        });
        let seam_pairs: Vec<([f32; 2], [f32; 2])> = preps.iter().flat_map(|p| (0..p.valid.len()).filter(|&j| p.valid[j]).map(|j| (p.pa[j], p.pb[j])).collect::<Vec<_>>()).collect();
        let mut stats = stats;
        stats["fill"] = fill_json;
        atlases.push(AtlasOut { image: im, name, w, h, orig: orig.rgba.clone(), baked, fin, perm, gutter_ok, strength, views_seen: nview, stats, owner, paint_avg, cos_best: cmax, filled, seam_pairs });
    }
    let views_json: Vec<Value> = views
        .iter()
        .enumerate()
        .map(|(i, v)| {
            json!({ "view": v.view.spec.name, "texels_visible": view_stats[i].0, "texels_with_paint": view_stats[i].1,
                    "registration": v.reg.to_json() })
        })
        .collect();
    let seam_total = json!({
        "mesh_edges_continuous": topo.edges_continuous, "mesh_edges_open": topo.edges_open, "mesh_edges_ambiguous_skipped": topo.edges_ambiguous,
        "seam_edges_total": topo.seams.len(), "per_atlas": seam_json,
    });
    Ok(BakeOut { atlases, seam: seam_total, views: Value::Array(views_json) })
}


/// Push-pull fill over the texels of one island. `texels` are indices into a `w`-wide atlas; `col` gives
/// a known texel's color and `conf` its weight (0 = unknown, 1 = fully known). Returns the filled color
/// of every listed texel (a known texel keeps `conf` of its own color and takes the rest from the
/// coarser levels), or None when nothing in the island is known.
pub fn pushpull_island(w: usize, texels: &[usize], col: &dyn Fn(usize) -> [f32; 3], conf: &dyn Fn(usize) -> f32) -> Option<Vec<[f32; 3]>> {
    if texels.is_empty() {
        return None;
    }
    let (mut x0, mut x1, mut y0, mut y1) = (usize::MAX, 0usize, usize::MAX, 0usize);
    for &i in texels {
        let (x, y) = (i % w, i / w);
        x0 = x0.min(x);
        x1 = x1.max(x);
        y0 = y0.min(y);
        y1 = y1.max(y);
    }
    let (lw, lh) = (x1 - x0 + 1, y1 - y0 + 1);
    struct Lv {
        w: usize,
        h: usize,
        c: Vec<[f32; 3]>,
        a: Vec<f32>,
    }
    let mut c0 = vec![[0.0f32; 3]; lw * lh];
    let mut a0 = vec![0.0f32; lw * lh];
    let mut any = false;
    for &i in texels {
        let a = conf(i).clamp(0.0, 1.0);
        if a > 0.0 {
            any = true;
            let k = (i / w - y0) * lw + (i % w - x0);
            c0[k] = col(i);
            a0[k] = a;
        }
    }
    if !any {
        return None;
    }
    let mut lv = vec![Lv { w: lw, h: lh, c: c0, a: a0 }];
    while lv.last().unwrap().w > 1 || lv.last().unwrap().h > 1 {
        let p = lv.last().unwrap();
        let (nw, nh) = ((p.w + 1) / 2, (p.h + 1) / 2);
        let mut c = vec![[0.0f32; 3]; nw * nh];
        let mut a = vec![0.0f32; nw * nh];
        for y in 0..nh {
            for x in 0..nw {
                let (mut s, mut sw) = ([0.0f32; 3], 0.0f32);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (xx, yy) = (2 * x + dx, 2 * y + dy);
                    if xx < p.w && yy < p.h {
                        let q = yy * p.w + xx;
                        for k in 0..3 {
                            s[k] += p.c[q][k] * p.a[q];
                        }
                        sw += p.a[q];
                    }
                }
                if sw > 0.0 {
                    c[y * nw + x] = [s[0] / sw, s[1] / sw, s[2] / sw];
                    a[y * nw + x] = sw.min(1.0);
                }
            }
        }
        lv.push(Lv { w: nw, h: nh, c, a });
    }
    // Push, coarse to fine. `fin` holds the completed colors of the level above.
    let top = lv.len() - 1;
    let mut fin: Vec<[f32; 3]> = lv[top].c.clone();
    for l in (0..top).rev() {
        let (cw, ch) = (lv[l + 1].w, lv[l + 1].h);
        let cur = &lv[l];
        let mut next = vec![[0.0f32; 3]; cur.w * cur.h];
        for y in 0..cur.h {
            for x in 0..cur.w {
                let (fx, fy) = ((x as f32 + 0.5) * 0.5 - 0.5, (y as f32 + 0.5) * 0.5 - 0.5);
                let (bx, by) = (fx.floor(), fy.floor());
                let (tx, ty) = (fx - bx, fy - by);
                let at = |xx: f32, yy: f32| -> [f32; 3] { fin[(yy as i64).clamp(0, ch as i64 - 1) as usize * cw + (xx as i64).clamp(0, cw as i64 - 1) as usize] };
                let (p00, p10, p01, p11) = (at(bx, by), at(bx + 1.0, by), at(bx, by + 1.0), at(bx + 1.0, by + 1.0));
                let q = y * cur.w + x;
                let a = cur.a[q];
                let mut out = [0.0f32; 3];
                for k in 0..3 {
                    let up = p00[k] * (1.0 - tx) * (1.0 - ty) + p10[k] * tx * (1.0 - ty) + p01[k] * (1.0 - tx) * ty + p11[k] * tx * ty;
                    out[k] = cur.c[q][k] * a + up * (1.0 - a);
                }
                next[q] = out;
            }
        }
        fin = next;
    }
    Some(texels.iter().map(|&i| fin[(i / w - y0) * lw + (i % w - x0)]).collect())
}

/// How far the mean color of each `block` x `block` texel block (what a mip level shows) drifts from
/// the mean of the painted island texels inside it, with the padding pass and without it. Only blocks
/// that hold painted texels and padding are counted.
fn mip_bleed(orig: &[u8], fin: &[u8], perm: &[bool], gutter_ok: &[bool], w: usize, h: usize, block: usize) -> Value {
    let (mut blocks, mut err_pad, mut err_none) = (0usize, 0.0f64, 0.0f64);
    for by in 0..h / block {
        for bx in 0..w / block {
            let (mut painted, mut np, mut ng) = ([0.0f64; 3], 0usize, 0usize);
            let (mut sum_pad, mut sum_none) = ([0.0f64; 3], [0.0f64; 3]);
            for y in by * block..(by + 1) * block {
                for x in bx * block..(bx + 1) * block {
                    let i = y * w + x;
                    let changed = fin[i * 4..i * 4 + 3] != orig[i * 4..i * 4 + 3];
                    if perm[i] && changed {
                        np += 1;
                        for c in 0..3 {
                            painted[c] += fin[i * 4 + c] as f64;
                        }
                    }
                    if gutter_ok[i] {
                        ng += 1;
                    }
                    for c in 0..3 {
                        sum_pad[c] += fin[i * 4 + c] as f64;
                        sum_none[c] += if gutter_ok[i] { orig[i * 4 + c] } else { fin[i * 4 + c] } as f64;
                    }
                }
            }
            if np == 0 || ng == 0 {
                continue;
            }
            let n = (block * block) as f64;
            let (mut ep, mut en) = (0.0, 0.0);
            for c in 0..3 {
                let r = painted[c] / np as f64;
                ep += (sum_pad[c] / n - r).abs() / 3.0;
                en += (sum_none[c] / n - r).abs() / 3.0;
            }
            blocks += 1;
            err_pad += ep;
            err_none += en;
        }
    }
    let d = blocks.max(1) as f64;
    json!({ "block": block, "mip_level": (block as f64).log2() as usize, "blocks": blocks, "mean_drift_with_padding": err_pad / d, "mean_drift_without_padding": err_none / d })
}

/// Independent check of a baked atlas against the input: rasterizes every triangle outside the mask
/// and counts changed texels under it, counts changes beyond the padding of the allowed triangles,
/// changes under a texel mask, and alpha changes.
pub fn verify_atlas(model: &Model, image: usize, before: &Img, after: &Img, masks: &Masks, pad: usize) -> Value {
    let (w, h) = (before.w, before.h);
    let mut under_blocked = vec![false; w * h];
    let mut allowed_cov = vec![false; w * h];
    for (g, &(mi, ti)) in tri_locs(model).iter().enumerate() {
        let m = &model.meshes[mi];
        if m.image != Some(image) {
            continue;
        }
        let t = m.tris[ti];
        let p = t.map(|v| [m.uv[v as usize][0] * w as f32, m.uv[v as usize][1] * h as f32]);
        let allowed = masks.tri_allowed[g];
        raster_tri(p, w, h, |x, y, _| {
            if allowed {
                allowed_cov[y * w + x] = true;
            } else {
                under_blocked[y * w + x] = true;
            }
        });
    }
    // Distance (Chebyshev, capped) from allowed coverage.
    let mut near = allowed_cov.clone();
    for _ in 0..pad {
        let prev = near.clone();
        for y in 0..h {
            for x in 0..w {
                if prev[y * w + x] {
                    continue;
                }
                'n: for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                        if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h && prev[yy as usize * w + xx as usize] {
                            near[y * w + x] = true;
                            break 'n;
                        }
                    }
                }
            }
        }
    }
    let name = atlas_name(model, image);
    let tmask = masks.texel.get(&name);
    let (mut changed, mut under_outside_tri, mut beyond_pad, mut under_texel_mask, mut alpha) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for i in 0..w * h {
        let a = &before.rgba[i * 4..i * 4 + 4];
        let b = &after.rgba[i * 4..i * 4 + 4];
        if a[3] != b[3] {
            alpha += 1;
        }
        if a == b {
            continue;
        }
        changed += 1;
        if under_blocked[i] {
            under_outside_tri += 1;
        }
        if !near[i] {
            beyond_pad += 1;
        }
        if let Some((_, _, m)) = tmask {
            if !m[i] {
                under_texel_mask += 1;
            }
        }
    }
    json!({
        "image": name,
        "texels_changed": changed,
        "changed_under_triangles_outside_mask": under_outside_tri,
        "changed_beyond_padding_of_allowed_triangles": beyond_pad,
        "changed_where_texel_mask_is_zero": under_texel_mask,
        "alpha_changed": alpha,
    })
}

// ---- shading check: vertex colors against texture brightness ---------------------------------------

/// What the game's character shaders do with vertex color, as text for reports.
pub const SHADING_POLICY: &str = "Field characters (PSX/FieldMapActor) draw texture * vertexColor * 2, so a vertex color of 0.5 \
means 1.0 and its alpha also multiplies the texture alpha before the 0.5 cutout. Battle characters (PSX/BattleMap_StatusEffect) ignore vertex color. \
Stock Zidane and Vivi have no vertex color channel, our glTF export writes no COLOR_0, and the ff9mapkit FBX import writes no color layer, so the \
texture shows 1:1 in field and battle. Keep it that way: no COLOR_0 on baked or pattern output, and paint any AO or shading into the texture only. \
If a mesh must carry COLOR_0, write 0.5 grey with alpha 1.0.";

fn luma_stats(img: &Img, mult: f64) -> (f64, f64) {
    let (mut sum, mut n, mut clip) = (0.0f64, 0usize, 0usize);
    for p in img.rgba.chunks_exact(4) {
        if p[3] < 128 {
            continue;
        }
        n += 1;
        sum += 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64;
        if mult > 1.0 && (0..3).any(|c| p[c] as f64 * mult > 255.0) {
            clip += 1;
        }
    }
    if n == 0 {
        (0.0, 0.0)
    } else {
        (sum / n as f64, clip as f64 / n as f64)
    }
}

/// Reads COLOR_0 on every primitive of `glb` and the brightness of each image, and says what the field
/// and battle shaders will do with them. `warnings` is empty when the mesh matches the stock assumption
/// (no vertex color, or a constant 0.5 grey with alpha 1).
pub fn shading_check(glb: &crate::glb::Glb, images: &[(String, &Img)]) -> Value {
    let mut prims = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let (mut with_color, mut without) = (0usize, 0usize);
    let mut worst_mult = 1.0f64;
    for (mi, m) in glb.list("meshes").iter().enumerate() {
        let name = m.get("name").and_then(Value::as_str).map(str::to_owned).unwrap_or(format!("mesh{mi}"));
        for (pi, p) in m.get("primitives").and_then(Value::as_array).map(|v| v.as_slice()).unwrap_or(&[]).iter().enumerate() {
            let Some(ai) = p["attributes"].get("COLOR_0").and_then(Value::as_u64) else {
                without += 1;
                prims.push(json!({ "mesh": name, "primitive": pi, "color_0": Value::Null }));
                continue;
            };
            let acc = match glb.accessor(ai as usize) {
                Ok(a) if a.count > 0 && a.ncomp >= 3 => a,
                _ => {
                    warnings.push(format!("{name}: COLOR_0 could not be read; check it by hand"));
                    continue;
                }
            };
            with_color += 1;
            let mut mean = [0.0f64; 4];
            let mut mn = [f64::MAX; 4];
            let mut mx = [f64::MIN; 4];
            for i in 0..acc.count {
                for c in 0..4 {
                    let v = if c < acc.ncomp { acc.get(i, c) } else { 1.0 };
                    mean[c] += v;
                    mn[c] = mn[c].min(v);
                    mx[c] = mx[c].max(v);
                }
            }
            for c in 0..4 {
                mean[c] /= acc.count as f64;
            }
            let mult = [mean[0] * 2.0, mean[1] * 2.0, mean[2] * 2.0];
            let mid = (mult[0] + mult[1] + mult[2]) / 3.0;
            if (mid - 1.0).abs() > (worst_mult - 1.0).abs() {
                worst_mult = mid;
            }
            let varies = (0..3).any(|c| mx[c] - mn[c] > 0.1);
            if (mid - 1.0).abs() > 0.1 {
                warnings.push(format!(
                    "{name}: COLOR_0 averages {:.2}, so the field draws this mesh at {:.2}x its texture (battle ignores it and draws 1.0x). Remove COLOR_0 or write 0.5 grey.",
                    (mean[0] + mean[1] + mean[2]) / 3.0,
                    mid
                ));
            }
            if varies {
                warnings.push(format!(
                    "{name}: COLOR_0 varies per vertex (range {:.2} to {:.2}). It shades the field only and would stack on any shading already painted into the texture; battle will not show it.",
                    mn[..3].iter().cloned().fold(f64::MAX, f64::min),
                    mx[..3].iter().cloned().fold(f64::MIN, f64::max)
                ));
            }
            if acc.ncomp == 4 && mean[3] < 0.99 {
                warnings.push(format!(
                    "{name}: COLOR_0 alpha averages {:.2}. The field multiplies it into the texture alpha before the 0.5 cutout, so cards and edges disappear. Write alpha 1.0.",
                    mean[3]
                ));
            }
            prims.push(json!({ "mesh": name, "primitive": pi, "color_0": { "vertices": acc.count, "mean": mean, "min": mn, "max": mx, "field_multiplier": mult } }));
        }
    }
    if with_color > 0 && without > 0 {
        warnings.push(format!("{with_color} primitives carry COLOR_0 and {without} do not; the field will draw them at different brightness"));
    }
    let tex: Vec<Value> = images
        .iter()
        .map(|(n, im)| {
            let (luma, clip) = luma_stats(im, worst_mult);
            if clip > 0.02 {
                warnings.push(format!("{n}: {:.1}% of opaque texels would clip to white in the field at {:.2}x", clip * 100.0, worst_mult));
            }
            json!({ "image": n, "mean_luma_opaque": (luma * 100.0).round() / 100.0, "share_clipping_in_field": (clip * 1e4).round() / 1e4 })
        })
        .collect();
    json!({
        "policy": SHADING_POLICY,
        "primitives_with_color_0": with_color,
        "primitives_without_color_0": without,
        "field_multiplier_worst": worst_mult,
        "battle_multiplier": 1.0,
        "primitives": prims,
        "textures": tex,
        "warnings": warnings,
        "ok": warnings.is_empty(),
    })
}

/// Warnings for a reference and a test model: a different field multiplier from vertex color, or a texture whose mean brightness moved by more than `tol` (0.15 = 15%).
pub fn shading_compare(reference: &Value, test: &Value, tol: f64) -> Vec<String> {
    let mut w = Vec::new();
    let (rm, tm) = (reference["field_multiplier_worst"].as_f64().unwrap_or(1.0), test["field_multiplier_worst"].as_f64().unwrap_or(1.0));
    if (rm - tm).abs() > 0.1 {
        w.push(format!("field multiplier from vertex color: reference {rm:.2}x, test {tm:.2}x"));
    }
    for (i, rt) in reference["textures"].as_array().cloned().unwrap_or_default().iter().enumerate() {
        let Some(tt) = test["textures"].as_array().and_then(|t| t.get(i)) else { continue };
        let (a, b) = (rt["mean_luma_opaque"].as_f64().unwrap_or(0.0), tt["mean_luma_opaque"].as_f64().unwrap_or(0.0));
        if a > 0.0 && ((b - a) / a).abs() > tol {
            w.push(format!(
                "image {}: mean brightness {:.1} in the test against {:.1} in the reference ({:+.0}%); with the same vertex colors the game shows that shift in field and battle",
                rt["image"].as_str().unwrap_or("?"),
                b,
                a,
                (b - a) / a * 100.0
            ));
        }
    }
    w
}
