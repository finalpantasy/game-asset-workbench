//! Generic defect finder: turns "it looks messy" into a list of specific places to fix.
//!
//! Five detectors, each with a score (1.0 = at the limit) and a place that maps back to triangles,
//! a UV island and a texel box, plus a screen box in one of the drawn views:
//!
//! * jagged edge: stair steps along the contour of the part's pixels (alpha cutout and silhouette),
//!   measured on close-up renders by the deviation from a smoothed contour;
//! * discoloration: pixels far from the part's own palette (k-means in Lab) that also differ from
//!   the ring around them, found as blobs in close-up and game-distance renders;
//! * seam stripe: a color step across an edge that is continuous in 3D but split in UV, measured
//!   on the texture on both sides of the edge and compared with the same measure inside islands;
//! * direction break: strand orientation (structure tensor, `strands`) turning across a seam or
//!   inside a chart;
//! * dark gutter: texels just outside an island that differ strongly from the island's border
//!   texels, which bilinear filtering and mip levels pull into the surface.
//!
//! Limits are provisional: they come from a handful of models (see experiments/goldens/REVIEW.md).
use crate::golden::Shot;
use crate::render::{self, Camera};
use crate::strands;
use crate::{Model, Texture};
use std::collections::{BTreeMap, HashMap};

/// Per mesh, per triangle: true for the triangles under inspection.
pub type Keep = Vec<Vec<bool>>;

pub const JAG_RMS_PX: f64 = 0.9;
pub const JAG_STAIRS: usize = 4;
pub const DISC_DE: f32 = 22.0;
pub const DISC_RING_DE: f32 = 12.0;
pub const STRIPE_DE: f64 = 14.0;
pub const DIR_DEG: f64 = 45.0;
pub const DIR_INNER_DEG: f64 = 60.0;
pub const GUTTER_DE: f32 = 28.0;
/// Across-strand contrast at which a patch counts as smeared: this many times lower than the reference's median.
pub const BLUR_RATIO: f64 = 2.2;
/// Colour variance this many times lower than the reference's median: a flat patch.
pub const FLAT_RATIO: f64 = 8.0;
/// Colour step (Lab) two pixels either side of a boundary between two triangles of the part: a hard cut.
pub const LAYER_DE: f64 = 14.0;
/// A hard cut must run this many connected pixels.
pub const LAYER_RUN_PX: usize = 24;
/// A one- or two-pixel dot must differ from its surroundings by this much (Lab).
pub const DOT_DE: f32 = 16.0;
/// Dots a 16 x 16 render tile needs before it counts.
pub const DOT_TILE_MIN: usize = 3;
/// Alpha holes or scraps up to this many texels count as stipple.
pub const STIPPLE_COMP: usize = 8;
/// A fragment of the part this small (render pixels, close view) that is cut off from the rest is a speck.
pub const SPECK_PX: usize = 60;
/// Colour difference to every one of the four neighbours that marks a dithered pixel.
pub const DITHER_DE: f32 = 14.0;
/// A flat triangle group needs this many pixels (close view) to count as a flat card.
pub const FLAT_CARD_PX: usize = 120;
/// Fragments below this area are antialiasing leftovers at a tip, not specks.
pub const SPECK_MIN_PX: usize = 5;
/// Share of covered texels in a 16 x 16 window that sit in tiny alpha holes or scraps: stipple.
pub const STIPPLE_FRAC: f64 = 0.05;

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
pub enum Kind {
    Jagged,
    Discolor,
    Stripe,
    Direction,
    Gutter,
    Blur,
    Flat,
    Speck,
    Stipple,
    LayerCut,
    Grain,
}

impl Kind {
    pub const ALL: [Kind; 11] = [Kind::Jagged, Kind::Discolor, Kind::Stripe, Kind::Direction, Kind::Gutter, Kind::Blur, Kind::Flat, Kind::Speck, Kind::Stipple, Kind::LayerCut, Kind::Grain];
    pub fn tag(self) -> &'static str {
        match self {
            Kind::Jagged => "jagged-edge",
            Kind::Discolor => "discoloration",
            Kind::Stripe => "seam-stripe",
            Kind::Direction => "direction-break",
            Kind::Gutter => "dark-gutter",
            Kind::Blur => "smear-blur",
            Kind::Flat => "flat-patch",
            Kind::Speck => "speck",
            Kind::Stipple => "alpha-stipple",
            Kind::LayerCut => "layer-direction-cut",
            Kind::Grain => "lock-scale",
        }
    }
    /// What `Defect::raw` measures.
    pub fn unit(self) -> &'static str {
        match self {
            Kind::Jagged => "px rms off the smoothed contour",
            Kind::Discolor => "Lab delta-E from the palette",
            Kind::Stripe => "Lab delta-E across the seam",
            Kind::Direction => "deg between strand axes",
            Kind::Gutter => "Lab delta-E, gutter vs island border",
            Kind::Blur => "times lower across-strand contrast than the reference",
            Kind::Flat => "times lower colour variance than the reference",
            Kind::Speck => "isolation: sqrt(limit area / fragment area), or 1 + dots / 12 for dot clusters",
            Kind::Stipple => "share of covered texels in tiny alpha holes or scraps",
            Kind::LayerCut => "Lab step across a hard cut between two triangles",
            Kind::Grain => "times more fibre-scale energy share than the reference",
        }
    }
    pub fn limit(self) -> f64 {
        match self {
            Kind::Jagged => JAG_RMS_PX,
            Kind::Discolor => DISC_DE as f64,
            Kind::Stripe => STRIPE_DE,
            Kind::Direction => DIR_DEG,
            Kind::Gutter => GUTTER_DE as f64,
            Kind::Blur => BLUR_RATIO,
            Kind::Flat => FLAT_RATIO,
            Kind::Speck => 1.0,
            Kind::Stipple => STIPPLE_FRAC,
            Kind::LayerCut => LAYER_DE,
            Kind::Grain => LOCK_RATIO,
        }
    }
}

/// A box of texels (x1, y1 exclusive) on one image of the model.
#[derive(Clone, Debug)]
pub struct TexelBox {
    pub image: usize,
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

#[derive(Clone, Debug)]
pub struct Screen {
    pub view: String,
    pub px: usize,
    /// x0, y0, x1, y1 in pixels of that view's picture.
    pub bbox: [i32; 4],
}

#[derive(Clone, Debug)]
pub struct Defect {
    pub kind: Kind,
    pub raw: f64,
    /// `raw` over the kind's limit: 1.0 sits on the limit.
    pub score: f64,
    pub note: String,
    /// (mesh, triangle) pairs the defect covers.
    pub tris: Vec<(usize, usize)>,
    /// Vertices (mesh, index) to project into views for the screen box.
    pub anchors: Vec<(usize, u32)>,
    pub boxes: Vec<TexelBox>,
    /// (mesh, island) of the first triangle.
    pub island: Option<(usize, u32)>,
    pub screen: Option<Screen>,
}

/// What one detector measured over the whole part, apart from the individual defects.
#[derive(Clone, Debug, Default)]
pub struct Load {
    /// Share of the measured thing that is over the limit, in percent.
    pub percent: f64,
    pub measured: f64,
    pub flagged: f64,
    pub extra: Vec<(String, f64)>,
}

// ---------- small helpers ----------

type V3 = [f64; 3];

fn v3(p: [f32; 3]) -> V3 {
    [p[0] as f64, p[1] as f64, p[2] as f64]
}
fn sub3(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross3(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot3(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn unit3(a: V3) -> V3 {
    let l = dot3(a, a).sqrt().max(1e-30);
    [a[0] / l, a[1] / l, a[2] / l]
}

fn lin(c: f32) -> f32 {
    let c = c / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB (0..255 per channel) to CIE Lab (D65).
pub fn lab(rgb: [f32; 3]) -> [f32; 3] {
    let (r, g, b) = (lin(rgb[0]), lin(rgb[1]), lin(rgb[2]));
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    let f = |t: f32| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

pub fn de(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Bilinear sample of level 0 at (u, v), RGBA 0..255.
pub fn bilinear(t: &Texture, u: f32, v: f32) -> [f32; 4] {
    let x = u * t.w as f32 - 0.5;
    let y = v * t.h as f32 - 0.5;
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let cx = |x: f32| (x as i64).clamp(0, t.w as i64 - 1) as usize;
    let cy = |y: f32| (y as i64).clamp(0, t.h as i64 - 1) as usize;
    let mut out = [0.0f32; 4];
    for (dx, dy, wt) in [(0.0, 0.0, (1.0 - fx) * (1.0 - fy)), (1.0, 0.0, fx * (1.0 - fy)), (0.0, 1.0, (1.0 - fx) * fy), (1.0, 1.0, fx * fy)] {
        let o = (cy(y0 + dy) * t.w + cx(x0 + dx)) * 4;
        for c in 0..4 {
            out[c] += wt * t.rgba[o + c] as f32;
        }
    }
    out
}

fn find(p: &mut [usize], mut x: usize) -> usize {
    while p[x] != x {
        p[x] = p[p[x]];
        x = p[x];
    }
    x
}

fn union(p: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(p, a), find(p, b));
    if ra != rb {
        p[ra.max(rb)] = ra.min(rb);
    }
}

fn mesh_tex<'a>(model: &Model, tex: &'a [Option<Texture>], mi: usize) -> Option<&'a Texture> {
    model.meshes[mi].image.and_then(|i| tex.get(i)).and_then(|t| t.as_ref())
}

/// `keep` as one flag per global triangle id.
pub fn keep_global(model: &Model, keep: &Keep) -> Vec<bool> {
    let mut out = Vec::with_capacity(model.triangle_count());
    for (mi, m) in model.meshes.iter().enumerate() {
        for ti in 0..m.tris.len() {
            out.push(keep.get(mi).and_then(|k| k.get(ti)).copied().unwrap_or(false));
        }
    }
    out
}

fn push_box(boxes: &mut Vec<TexelBox>, image: usize, x0: i64, y0: i64, x1: i64, y1: i64, pad: i64, w: usize, h: usize) {
    let b = TexelBox {
        image,
        x0: (x0 - pad).clamp(0, w as i64) as usize,
        y0: (y0 - pad).clamp(0, h as i64) as usize,
        x1: (x1 + pad + 1).clamp(0, w as i64) as usize,
        y1: (y1 + pad + 1).clamp(0, h as i64) as usize,
    };
    // Merge only boxes that touch (within a small gap): the two sides of a seam can sit far apart
    // in UV, and one box over both would cover most of the atlas.
    const GAP: usize = 12;
    let mut b = b;
    loop {
        let hit = boxes.iter().position(|e| e.image == image && e.x0 <= b.x1 + GAP && b.x0 <= e.x1 + GAP && e.y0 <= b.y1 + GAP && b.y0 <= e.y1 + GAP);
        match hit {
            Some(i) => {
                let e = boxes.remove(i);
                b = TexelBox { image, x0: e.x0.min(b.x0), y0: e.y0.min(b.y0), x1: e.x1.max(b.x1), y1: e.y1.max(b.y1) };
            }
            None => break,
        }
    }
    boxes.push(b);
    // The biggest first: that is the box the crop sheet shows.
    boxes.sort_by_key(|e| std::cmp::Reverse((e.x1 - e.x0) * (e.y1 - e.y0)));
}

// ---------- topology: edge pairs and islands ----------

#[derive(Clone, Debug)]
pub struct Side {
    pub mesh: usize,
    pub tri: usize,
    pub va: u32,
    pub vb: u32,
    pub wa: u32,
    pub wb: u32,
    pub uva: [f32; 2],
    pub uvb: [f32; 2],
    pub uvc: [f32; 2],
    pub pa: [f32; 3],
    pub pb: [f32; 3],
    pub n: V3,
}

#[derive(Clone, Debug)]
pub struct Pair {
    pub a: Side,
    pub b: Side,
    /// Same mesh and the same UV at both ends: the edge lies inside an island.
    pub continuous: bool,
}

pub struct Topo {
    pub pairs: Vec<Pair>,
    /// Island id per mesh, per triangle (ids are unique over the model).
    pub island: Vec<Vec<u32>>,
}

/// Edges shared by exactly two triangles (welded by position), and the UV islands.
pub fn topo(model: &Model) -> Topo {
    let mut weld: HashMap<[i64; 3], u32> = HashMap::new();
    let mut edges: HashMap<(u32, u32), Vec<Side>> = HashMap::new();
    let mut offs = Vec::new();
    let mut total = 0usize;
    for (mi, m) in model.meshes.iter().enumerate() {
        offs.push(total);
        total += m.tris.len();
        let wid: Vec<u32> = m
            .pos
            .iter()
            .map(|p| {
                let k = [(p[0] * 1e5).round() as i64, (p[1] * 1e5).round() as i64, (p[2] * 1e5).round() as i64];
                let n = weld.len() as u32;
                *weld.entry(k).or_insert(n)
            })
            .collect();
        for (ti, t) in m.tris.iter().enumerate() {
            let p: [V3; 3] = std::array::from_fn(|k| v3(m.pos[t[k] as usize]));
            let n = unit3(cross3(sub3(p[1], p[0]), sub3(p[2], p[0])));
            for k in 0..3 {
                let (i, j, c) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
                let (wi, wj) = (wid[i as usize], wid[j as usize]);
                if wi == wj {
                    continue;
                }
                let (a, b) = if wi < wj { (i, j) } else { (j, i) };
                let key = (wid[a as usize], wid[b as usize]);
                let v = edges.entry(key).or_default();
                if v.len() < 4 {
                    v.push(Side {
                        mesh: mi,
                        tri: ti,
                        va: a,
                        vb: b,
                        wa: key.0,
                        wb: key.1,
                        uva: m.uv[a as usize],
                        uvb: m.uv[b as usize],
                        uvc: m.uv[c as usize],
                        pa: m.pos[a as usize],
                        pb: m.pos[b as usize],
                        n,
                    });
                }
            }
        }
    }
    let mut keys: Vec<(u32, u32)> = edges.keys().copied().collect();
    keys.sort();
    let mut uf: Vec<usize> = (0..total).collect();
    let mut pairs = Vec::new();
    for k in keys {
        let v = &edges[&k];
        if v.len() != 2 {
            continue;
        }
        let (a, b) = (v[0].clone(), v[1].clone());
        let close = |p: [f32; 2], q: [f32; 2]| (p[0] - q[0]).abs() < 1e-4 && (p[1] - q[1]).abs() < 1e-4;
        let continuous = a.mesh == b.mesh && close(a.uva, b.uva) && close(a.uvb, b.uvb);
        if continuous {
            union(&mut uf, offs[a.mesh] + a.tri, offs[b.mesh] + b.tri);
        }
        pairs.push(Pair { a, b, continuous });
    }
    let mut dense: HashMap<usize, u32> = HashMap::new();
    let mut island: Vec<Vec<u32>> = Vec::new();
    for (mi, m) in model.meshes.iter().enumerate() {
        let mut v = Vec::with_capacity(m.tris.len());
        for ti in 0..m.tris.len() {
            let r = find(&mut uf, offs[mi] + ti);
            let n = dense.len() as u32;
            v.push(*dense.entry(r).or_insert(n));
        }
        island.push(v);
    }
    Topo { pairs, island }
}

// ---------- seam stripes ----------

#[derive(Clone, Debug)]
pub struct StripeEdge {
    pub pair: usize,
    pub de: f64,
    pub dl: f64,
    pub len: f64,
}

#[derive(Default)]
pub struct StripeOut {
    pub defects: Vec<Defect>,
    pub load: Load,
    pub edges: Vec<StripeEdge>,
    pub inner_mean: f64,
    pub inner_p90: f64,
    pub seam_mean: f64,
    pub seam_p90: f64,
}

const SAMPLES: usize = 7;
const DELTA_TEXELS: f32 = 2.0;

fn side_lab(t: &Texture, s: &Side, f: f32) -> Option<[f32; 3]> {
    let (w, h) = (t.w as f32, t.h as f32);
    let a = [s.uva[0] * w, s.uva[1] * h];
    let b = [s.uvb[0] * w, s.uvb[1] * h];
    let c = [s.uvc[0] * w, s.uvc[1] * h];
    let e = [b[0] - a[0], b[1] - a[1]];
    let l = e[0].hypot(e[1]);
    if l < 0.05 {
        return None;
    }
    let mut perp = [-e[1] / l, e[0] / l];
    if perp[0] * (c[0] - a[0]) + perp[1] * (c[1] - a[1]) < 0.0 {
        perp = [-perp[0], -perp[1]];
    }
    let x = a[0] + e[0] * f + perp[0] * DELTA_TEXELS;
    let y = a[1] + e[1] * f + perp[1] * DELTA_TEXELS;
    let px = bilinear(t, x / w, y / h);
    if px[3] < 128.0 {
        return None;
    }
    Some(lab([px[0], px[1], px[2]]))
}

fn pair_step(model: &Model, tex: &[Option<Texture>], p: &Pair) -> Option<(f64, f64)> {
    let (ta, tb) = (mesh_tex(model, tex, p.a.mesh)?, mesh_tex(model, tex, p.b.mesh)?);
    let (mut sde, mut sdl, mut n) = (0.0f64, 0.0f64, 0usize);
    // The second side's ends are in the same key order, so the same fraction lands on the same 3D point.
    for k in 0..SAMPLES {
        let f = (k as f32 + 0.5) / SAMPLES as f32;
        let (Some(ca), Some(cb)) = (side_lab(ta, &p.a, f), side_lab(tb, &p.b, f)) else { continue };
        sde += de(ca, cb) as f64;
        sdl += (ca[0] - cb[0]).abs() as f64;
        n += 1;
    }
    (n >= 3).then(|| (sde / n as f64, sdl / n as f64))
}

fn percentile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * q).round() as usize]
}

pub fn stripes(model: &Model, topo: &Topo, tex: &[Option<Texture>], keep: &Keep) -> StripeOut {
    let kept = |s: &Side| keep.get(s.mesh).and_then(|k| k.get(s.tri)).copied().unwrap_or(false);
    let mut out = StripeOut::default();
    let mut inner: Vec<f64> = Vec::new();
    let mut seam: Vec<f64> = Vec::new();
    let (mut len_seam, mut len_bad) = (0.0f64, 0.0f64);
    let mut flagged: Vec<usize> = Vec::new();
    for (pi, p) in topo.pairs.iter().enumerate() {
        if !(kept(&p.a) && kept(&p.b)) {
            continue;
        }
        if dot3(p.a.n, p.b.n) < 0.5 {
            continue; // a crease or a fold: a color step is expected there
        }
        let len = dot3(sub3(v3(p.a.pa), v3(p.a.pb)), sub3(v3(p.a.pa), v3(p.a.pb))).sqrt();
        if len < 1e-5 {
            continue;
        }
        let Some((d, dl)) = pair_step(model, tex, p) else { continue };
        if p.continuous {
            inner.push(d);
        } else {
            seam.push(d);
            len_seam += len;
            out.edges.push(StripeEdge { pair: pi, de: d, dl, len });
            if d >= STRIPE_DE {
                len_bad += len;
                flagged.push(out.edges.len() - 1);
            }
        }
    }
    out.inner_mean = if inner.is_empty() { 0.0 } else { inner.iter().sum::<f64>() / inner.len() as f64 };
    out.inner_p90 = percentile(&mut inner, 0.9);
    out.seam_mean = if seam.is_empty() { 0.0 } else { seam.iter().sum::<f64>() / seam.len() as f64 };
    out.seam_p90 = percentile(&mut seam, 0.9);
    out.load = Load {
        percent: if len_seam > 0.0 { 100.0 * len_bad / len_seam } else { 0.0 },
        measured: len_seam,
        flagged: len_bad,
        extra: vec![("seam_edges".into(), seam.len() as f64), ("inner_edges".into(), inner.len() as f64), ("seam_mean_de".into(), out.seam_mean), ("seam_p90_de".into(), out.seam_p90), ("inner_mean_de".into(), out.inner_mean)],
    };
    // Runs: flagged edges that touch at a welded vertex.
    let mut ids: HashMap<u32, usize> = HashMap::new();
    let mut uf: Vec<usize> = (0..flagged.len()).collect();
    for (i, &ei) in flagged.iter().enumerate() {
        let p = &topo.pairs[out.edges[ei].pair];
        for w in [p.a.wa, p.a.wb] {
            if let Some(&j) = ids.get(&w) {
                union(&mut uf, i, j);
            } else {
                ids.insert(w, i);
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..flagged.len() {
        let r = find(&mut uf, i);
        groups.entry(r).or_default().push(flagged[i]);
    }
    for (_, g) in groups {
        let tl: f64 = g.iter().map(|&e| out.edges[e].len).sum();
        let raw = g.iter().map(|&e| out.edges[e].de * out.edges[e].len).sum::<f64>() / tl.max(1e-12);
        let worst = g.iter().map(|&e| out.edges[e].de).fold(0.0, f64::max);
        let mut d = Defect { kind: Kind::Stripe, raw, score: raw / STRIPE_DE, note: String::new(), tris: Vec::new(), anchors: Vec::new(), boxes: Vec::new(), island: None, screen: None };
        let mut isl: Vec<u32> = Vec::new();
        for &e in &g {
            let p = &topo.pairs[out.edges[e].pair];
            for s in [&p.a, &p.b] {
                if !d.tris.contains(&(s.mesh, s.tri)) {
                    d.tris.push((s.mesh, s.tri));
                }
                let il = topo.island[s.mesh][s.tri];
                if !isl.contains(&il) {
                    isl.push(il);
                }
                if let Some(t) = mesh_tex(model, tex, s.mesh) {
                    let (w, h) = (t.w as f32, t.h as f32);
                    let img = model.meshes[s.mesh].image.unwrap_or(0);
                    let xs = [s.uva[0] * w, s.uvb[0] * w, s.uvc[0] * w];
                    let ys = [s.uva[1] * h, s.uvb[1] * h, s.uvc[1] * h];
                    // The edge and a few texels to each side of it, not the whole triangle.
                    let (ex0, ex1) = (xs[0].min(xs[1]), xs[0].max(xs[1]));
                    let (ey0, ey1) = (ys[0].min(ys[1]), ys[0].max(ys[1]));
                    push_box(&mut d.boxes, img, ex0 as i64, ey0 as i64, ex1 as i64, ey1 as i64, 4, t.w, t.h);
                }
                d.anchors.push((s.mesh, s.va));
                d.anchors.push((s.mesh, s.vb));
            }
            if d.island.is_none() {
                d.island = Some((p.a.mesh, topo.island[p.a.mesh][p.a.tri]));
            }
        }
        d.tris.truncate(64);
        d.note = format!("color step across a UV seam that is continuous in 3D: mean delta-E {:.1}, worst edge {:.1}; {} edge(s), {:.3} units long, islands {}; inside islands the same measure averages {:.1}", raw, worst, g.len(), tl, isl.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("|"), out.inner_mean);
        out.defects.push(d);
    }
    out.defects.sort_by(|a, b| b.score.total_cmp(&a.score));
    out
}

// ---------- direction breaks ----------

#[derive(Default)]
pub struct DirOut {
    pub defects: Vec<Defect>,
    pub load: Load,
    pub report: strands::Report,
}

/// Strand-direction check on the kept triangles (meant for hair). Reads each mesh's own texture.
pub fn directions(model: &Model, tex: &[Option<Texture>], keep: &Keep, topo: &Topo) -> DirOut {
    let mut pieces: Vec<strands::Piece> = Vec::new();
    let mut piece_mesh: Vec<usize> = Vec::new();
    for (mi, m) in model.meshes.iter().enumerate() {
        if let (Some(t), true) = (mesh_tex(model, tex, mi), keep[mi].iter().any(|&f| f)) {
            pieces.push(strands::Piece { mesh: m, tris: &keep[mi], rgba: &t.rgba, w: t.w, h: t.h });
            piece_mesh.push(mi);
        }
    }
    let mut out = DirOut::default();
    if pieces.is_empty() {
        return out;
    }
    let r = strands::check(&pieces, &strands::Opts::default());
    let seam_w: f64 = r.edges.iter().filter(|e| e.kind == strands::Kind::Seam).map(|e| e.weight as f64).sum();
    let mut flagged: Vec<&strands::EdgeScore> = Vec::new();
    let mut bad_w = 0.0f64;
    for e in &r.edges {
        let (limit, w_min) = match e.kind {
            strands::Kind::Seam => (DIR_DEG, 1.0),
            strands::Kind::Inner => (DIR_INNER_DEG, 1.5),
        };
        if e.mean_deg as f64 >= limit && e.weight as f64 >= w_min {
            flagged.push(e);
            if e.kind == strands::Kind::Seam {
                bad_w += e.weight as f64;
            }
        }
    }
    out.load = Load {
        percent: if seam_w > 0.0 { 100.0 * bad_w / seam_w } else { 0.0 },
        measured: seam_w,
        flagged: bad_w,
        extra: vec![
            ("seam_mean_deg".into(), r.seam.mean_deg),
            ("seam_over_30_percent".into(), 100.0 * r.seam.bad_frac),
            ("inner_mean_deg".into(), r.inner.mean_deg),
            ("root_tip_mean_deg".into(), r.root_tip.mean_deg),
            ("root_tip_over_30_percent".into(), 100.0 * r.root_tip.bad_frac),
            ("seam_edges".into(), r.seam_edges as f64),
        ],
    };
    // Cluster flagged edges that share a triangle.
    let key = |p: usize, t: usize| (p, t);
    let mut ids: HashMap<(usize, usize), usize> = HashMap::new();
    let mut uf: Vec<usize> = (0..flagged.len()).collect();
    for (i, e) in flagged.iter().enumerate() {
        for k in [key(e.a.0, e.a.1), key(e.b.0, e.b.1)] {
            if let Some(&j) = ids.get(&k) {
                union(&mut uf, i, j);
            } else {
                ids.insert(k, i);
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..flagged.len() {
        let r = find(&mut uf, i);
        groups.entry(r).or_default().push(i);
    }
    for (_, g) in groups {
        let tw: f64 = g.iter().map(|&i| flagged[i].weight as f64).sum();
        let raw = g.iter().map(|&i| flagged[i].mean_deg as f64 * flagged[i].weight as f64).sum::<f64>() / tw.max(1e-12);
        let inner = g.iter().all(|&i| flagged[i].kind == strands::Kind::Inner);
        let limit = if inner { DIR_INNER_DEG } else { DIR_DEG };
        let mut d = Defect { kind: Kind::Direction, raw, score: raw / limit, note: String::new(), tris: Vec::new(), anchors: Vec::new(), boxes: Vec::new(), island: None, screen: None };
        for &i in &g {
            let e = flagged[i];
            for (side, uv) in [(e.a, e.uv_a), (e.b, e.uv_b)] {
                let mi = piece_mesh[side.0];
                if !d.tris.contains(&(mi, side.1)) {
                    d.tris.push((mi, side.1));
                    for &v in &model.meshes[mi].tris[side.1] {
                        d.anchors.push((mi, v));
                    }
                }
                if let Some(t) = mesh_tex(model, tex, mi) {
                    let img = model.meshes[mi].image.unwrap_or(0);
                    let (x0, x1) = ((uv[0][0].min(uv[1][0]) * t.w as f32) as i64, (uv[0][0].max(uv[1][0]) * t.w as f32) as i64);
                    let (y0, y1) = ((uv[0][1].min(uv[1][1]) * t.h as f32) as i64, (uv[0][1].max(uv[1][1]) * t.h as f32) as i64);
                    push_box(&mut d.boxes, img, x0, y0, x1, y1, 8, t.w, t.h);
                }
            }
        }
        if let Some(&(mi, ti)) = d.tris.first() {
            d.island = Some((mi, topo.island[mi][ti]));
        }
        d.tris.truncate(64);
        d.note = if inner {
            format!("strands turn {:.0} deg between neighbouring triangles inside one island ({} edge(s)); the painted direction changes abruptly", raw, g.len())
        } else {
            format!("strands meet at {:.0} deg across a UV seam ({} edge(s)); inside islands strands turn {:.0} deg on average", raw, g.len(), r.inner.mean_deg)
        };
        out.defects.push(d);
    }
    out.defects.sort_by(|a, b| b.score.total_cmp(&a.score));
    out.report = r;
    out
}

// ---------- dark gutters ----------

#[derive(Default)]
pub struct GutterOut {
    pub defects: Vec<Defect>,
    pub load: Load,
}

fn raster_tri(tri_at: &mut [u32], w: usize, h: usize, uv: [[f32; 2]; 3], id: u32) {
    let p: [[f32; 2]; 3] = std::array::from_fn(|k| [uv[k][0] * w as f32, uv[k][1] * h as f32]);
    let area = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]);
    if area.abs() < 1e-6 {
        return;
    }
    let x0 = p.iter().map(|q| q[0]).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
    let x1 = (p.iter().map(|q| q[0]).fold(f32::MIN, f32::max).ceil().max(0.0) as usize).min(w.saturating_sub(1));
    let y0 = p.iter().map(|q| q[1]).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
    let y1 = (p.iter().map(|q| q[1]).fold(f32::MIN, f32::max).ceil().max(0.0) as usize).min(h.saturating_sub(1));
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (sx, sy) = (x as f32 + 0.5, y as f32 + 0.5);
            let w0 = ((p[1][0] - sx) * (p[2][1] - sy) - (p[1][1] - sy) * (p[2][0] - sx)) / area;
            let w1 = ((p[2][0] - sx) * (p[0][1] - sy) - (p[2][1] - sy) * (p[0][0] - sx)) / area;
            let w2 = 1.0 - w0 - w1;
            let e = -0.002;
            if w0 >= e && w1 >= e && w2 >= e {
                tri_at[y * w + x] = id + 1;
            }
        }
    }
}

pub fn gutters(model: &Model, topo: &Topo, tex: &[Option<Texture>], keep: &Keep) -> GutterOut {
    let offs = render::tri_offsets(model);
    let mut out = GutterOut::default();
    let (mut boundary_total, mut bad_total) = (0usize, 0usize);
    let mut per_image: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (mi, m) in model.meshes.iter().enumerate() {
        if let Some(im) = m.image {
            if tex.get(im).is_some_and(|t| t.is_some()) {
                per_image.entry(im).or_default().push(mi);
            }
        }
    }
    // Seam edges as UV segments, per image, to say whether a gutter sits on a seam.
    for (im, meshes) in per_image {
        let t = tex[im].as_ref().unwrap();
        let (w, h) = (t.w, t.h);
        let mut tri_at = vec![0u32; w * h];
        for &mi in &meshes {
            let m = &model.meshes[mi];
            for (ti, tr) in m.tris.iter().enumerate() {
                raster_tri(&mut tri_at, w, h, std::array::from_fn(|k| m.uv[tr[k] as usize]), (offs[mi] + ti) as u32);
            }
        }
        let mut seam = vec![false; w * h];
        for p in &topo.pairs {
            if p.continuous {
                continue;
            }
            for s in [&p.a, &p.b] {
                if model.meshes[s.mesh].image != Some(im) {
                    continue;
                }
                let (a, b) = ([s.uva[0] * w as f32, s.uva[1] * h as f32], [s.uvb[0] * w as f32, s.uvb[1] * h as f32]);
                let n = ((b[0] - a[0]).hypot(b[1] - a[1]) * 2.0).ceil().max(1.0) as usize;
                for i in 0..=n {
                    let f = i as f32 / n as f32;
                    let (x, y) = ((a[0] + (b[0] - a[0]) * f) as i64, (a[1] + (b[1] - a[1]) * f) as i64);
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let (xx, yy) = (x + dx, y + dy);
                            if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                                seam[yy as usize * w + xx as usize] = true;
                            }
                        }
                    }
                }
            }
        }
        let kept_at = |i: usize| -> Option<(usize, usize)> {
            let id = tri_at[i];
            if id == 0 {
                return None;
            }
            let (mi, ti) = render::split_id(model, id as usize - 1);
            keep.get(mi).and_then(|k| k.get(ti)).copied().unwrap_or(false).then_some((mi, ti))
        };
        let color = |i: usize| -> [f32; 3] { lab([t.rgba[i * 4] as f32, t.rgba[i * 4 + 1] as f32, t.rgba[i * 4 + 2] as f32]) };
        let mut bad = vec![0.0f32; w * h]; // delta-E of the worst uncovered neighbour, 0 if fine
        let mut is_edge = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if kept_at(i).is_none() {
                    continue;
                }
                // Only opaque border texels matter: a transparent one shows nothing of its own colour.
                if t.rgba[i * 4 + 3] < 128 {
                    continue;
                }
                let mut worst = 0.0f32;
                let mut edge = false;
                for (dx, dy) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                    let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                    if xx < 0 || yy < 0 || xx as usize >= w || yy as usize >= h {
                        continue;
                    }
                    let j = yy as usize * w + xx as usize;
                    if tri_at[j] != 0 {
                        continue;
                    }
                    edge = true;
                    worst = worst.max(de(color(i), color(j)));
                }
                if edge {
                    is_edge[i] = true;
                    bad[i] = worst;
                }
            }
        }
        let edge_n = is_edge.iter().filter(|&&e| e).count();
        let bad_mask: Vec<bool> = (0..w * h).map(|i| is_edge[i] && bad[i] >= GUTTER_DE).collect();
        boundary_total += edge_n;
        bad_total += bad_mask.iter().filter(|&&b| b).count();
        // Components of bad border texels, joined across gaps up to 3 texels.
        let mut comp = vec![u32::MAX; w * h];
        let mut ncomp = 0u32;
        let list: Vec<usize> = (0..w * h).filter(|&i| bad_mask[i]).collect();
        let mut members: Vec<Vec<usize>> = Vec::new();
        for &s in &list {
            if comp[s] != u32::MAX {
                continue;
            }
            let mut stack = vec![s];
            comp[s] = ncomp;
            let mut mem = Vec::new();
            while let Some(i) = stack.pop() {
                mem.push(i);
                let (x, y) = ((i % w) as i64, (i / w) as i64);
                for dy in -3..=3i64 {
                    for dx in -3..=3i64 {
                        let (xx, yy) = (x + dx, y + dy);
                        if xx < 0 || yy < 0 || xx as usize >= w || yy as usize >= h {
                            continue;
                        }
                        let j = yy as usize * w + xx as usize;
                        if bad_mask[j] && comp[j] == u32::MAX {
                            comp[j] = ncomp;
                            stack.push(j);
                        }
                    }
                }
            }
            members.push(mem);
            ncomp += 1;
        }
        for mem in members {
            if mem.len() < 8 {
                continue;
            }
            let raw = mem.iter().map(|&i| bad[i] as f64).sum::<f64>() / mem.len() as f64;
            let on_seam = mem.iter().filter(|&&i| seam[i]).count() as f64 / mem.len() as f64;
            let dark = mem.iter().filter(|&&i| {
                let (x, y) = (i % w, i / w);
                [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dy)| {
                    let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                    xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h && {
                        let j = yy as usize * w + xx as usize;
                        tri_at[j] == 0 && color(j)[0] < color(i)[0] - 10.0
                    }
                })
            }).count() as f64 / mem.len() as f64;
            let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
            let mut d = Defect { kind: Kind::Gutter, raw, score: raw / GUTTER_DE as f64, note: String::new(), tris: Vec::new(), anchors: Vec::new(), boxes: Vec::new(), island: None, screen: None };
            for &i in &mem {
                let (x, y) = (i % w, i / w);
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
                if let Some(k) = kept_at(i) {
                    if !d.tris.contains(&k) && d.tris.len() < 48 {
                        d.tris.push(k);
                    }
                }
            }
            push_box(&mut d.boxes, im, x0 as i64, y0 as i64, x1 as i64, y1 as i64, 4, w, h);
            for &(mi, ti) in d.tris.iter().take(6) {
                for &v in &model.meshes[mi].tris[ti] {
                    d.anchors.push((mi, v));
                }
            }
            if let Some(&(mi, ti)) = d.tris.first() {
                d.island = Some((mi, topo.island[mi][ti]));
            }
            d.note = format!(
                "{} border texels whose neighbours just outside the island differ by delta-E {:.0} ({:.0}% darker): filtering and mips pull that colour into the surface; {}",
                mem.len(),
                raw,
                100.0 * dark,
                if on_seam > 0.5 { "the border lies on a UV seam that is continuous in 3D, so it shows as a line" } else { "an island border" }
            );
            out.defects.push(d);
        }
    }
    out.load = Load {
        percent: if boundary_total > 0 { 100.0 * bad_total as f64 / boundary_total as f64 } else { 0.0 },
        measured: boundary_total as f64,
        flagged: bad_total as f64,
        extra: vec![("border_texels".into(), boundary_total as f64), ("bad_border_texels".into(), bad_total as f64)],
    };
    out.defects.sort_by(|a, b| b.score.total_cmp(&a.score));
    out
}

// ---------- screen detectors ----------

/// One drawn view: the picture, the triangle id under each pixel, and the camera and pose it was
/// drawn with (needed to turn a pixel back into a UV).
pub struct ShotCtx<'a> {
    pub name: String,
    pub px: usize,
    pub cam: Camera,
    pub pose: &'a [Vec<[f32; 3]>],
    pub shot: &'a Shot,
}

impl ShotCtx<'_> {
    /// (mesh, triangle, UV) under a pixel.
    pub fn hit(&self, model: &Model, x: usize, y: usize) -> Option<(usize, usize, [f32; 2])> {
        let gid = *self.shot.pick.get(y * self.px + x)?;
        if gid < 0 {
            return None;
        }
        let (mi, ti) = render::split_id(model, gid as usize);
        let m = model.meshes.get(mi)?;
        let t = m.tris.get(ti)?;
        let pr: [[f32; 3]; 3] = std::array::from_fn(|k| self.cam.project(self.px, self.px, self.pose[mi][t[k] as usize]));
        let uv: [[f32; 2]; 3] = std::array::from_fn(|k| m.uv[t[k] as usize]);
        let (a, b, c) = (pr[0], pr[1], pr[2]);
        let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        if area.abs() < 1e-9 {
            return Some((mi, ti, uv[0]));
        }
        let (sx, sy) = (x as f32 + 0.5, y as f32 + 0.5);
        let w0 = ((b[0] - sx) * (c[1] - sy) - (b[1] - sy) * (c[0] - sx)) / area;
        let w1 = ((c[0] - sx) * (a[1] - sy) - (c[1] - sy) * (a[0] - sx)) / area;
        let mut w = [w0.max(0.0), w1.max(0.0), (1.0 - w0 - w1).max(0.0)];
        let s: f32 = w.iter().sum::<f32>().max(1e-9);
        for v in &mut w {
            *v /= s;
        }
        let (mut u, mut v, mut den) = (0.0, 0.0, 0.0);
        for k in 0..3 {
            let g = if self.cam.ortho { 1.0 } else { 1.0 / pr[k][2].max(1e-6) };
            u += w[k] * g * uv[k][0];
            v += w[k] * g * uv[k][1];
            den += w[k] * g;
        }
        Some((mi, ti, [u / den, v / den]))
    }

    fn mask(&self, kg: &[bool]) -> Vec<bool> {
        self.shot.pick.iter().map(|&g| g >= 0 && kg.get(g as usize).copied().unwrap_or(false)).collect()
    }
}

fn erode(mask: &[bool], w: usize, h: usize, times: usize) -> Vec<bool> {
    let mut cur = mask.to_vec();
    for _ in 0..times {
        let mut next = vec![false; w * h];
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let i = y * w + x;
                next[i] = cur[i] && cur[i - 1] && cur[i + 1] && cur[i - w] && cur[i + w] && cur[i - w - 1] && cur[i - w + 1] && cur[i + w - 1] && cur[i + w + 1];
            }
        }
        cur = next;
    }
    cur
}

const DIRS4_I64: [(i64, i64); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const DIRS8_I64: [(i64, i64); 8] = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)];
const DIRS8: [(i32, i32); 8] = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)];

/// Ordered 8-connected boundary pixels of every blob of `mask` (Moore tracing).
fn contours(mask: &[bool], w: usize, h: usize) -> Vec<Vec<[i32; 2]>> {
    let is = |x: i32, y: i32| x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && mask[y as usize * w + x as usize];
    let boundary = |x: i32, y: i32| is(x, y) && !(is(x - 1, y) && is(x + 1, y) && is(x, y - 1) && is(x, y + 1));
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            if seen[y as usize * w + x as usize] || !boundary(x, y) {
                continue;
            }
            let start = [x, y];
            let mut cur = start;
            let mut back = 0usize; // direction from cur to the pixel we came from: west
            let mut pts = vec![start];
            seen[y as usize * w + x as usize] = true;
            for _ in 0..(w * h).min(400_000) {
                let mut found = None;
                for k in 1..=8 {
                    let d = (back + k) % 8;
                    let n = [cur[0] + DIRS8[d].0, cur[1] + DIRS8[d].1];
                    if is(n[0], n[1]) {
                        found = Some((d, n));
                        break;
                    }
                }
                let Some((d, n)) = found else { break };
                let prev = [cur[0] + DIRS8[(d + 7) % 8].0, cur[1] + DIRS8[(d + 7) % 8].1];
                let dv = (prev[0] - n[0], prev[1] - n[1]);
                back = DIRS8.iter().position(|&q| q == dv).unwrap_or(0);
                cur = n;
                if cur == start {
                    break;
                }
                if boundary(cur[0], cur[1]) {
                    pts.push(cur);
                    seen[cur[1] as usize * w + cur[0] as usize] = true;
                }
            }
            out.push(pts);
        }
    }
    out
}

#[derive(Default)]
pub struct JagOut {
    pub defects: Vec<Defect>,
    pub contour_px: usize,
    pub flagged_px: usize,
    pub worst_rms: f64,
}

/// Stair steps along the part's contour in one view. `kg` is `keep_global`.
pub fn jagged(model: &Model, tex: &[Option<Texture>], sc: &ShotCtx, kg: &[bool]) -> JagOut {
    let (w, h) = (sc.px, sc.px);
    let mask = sc.mask(kg);
    let mut out = JagOut::default();
    for pts in contours(&mask, w, h) {
        let n = pts.len();
        if n < 48 {
            continue;
        }
        let at = |i: isize| pts[i.rem_euclid(n as isize) as usize];
        // Smooth the closed contour: two box passes of 11 points.
        let mut s: Vec<[f64; 2]> = pts.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
        for _ in 0..2 {
            let prev = s.clone();
            for i in 0..n {
                let (mut sx, mut sy) = (0.0, 0.0);
                for k in -5..=5isize {
                    let q = prev[(i as isize + k).rem_euclid(n as isize) as usize];
                    sx += q[0];
                    sy += q[1];
                }
                s[i] = [sx / 11.0, sy / 11.0];
            }
        }
        // Signed offset of each point from the smooth curve, along its normal.
        let mut dev = vec![0.0f64; n];
        for i in 0..n {
            let (a, b) = (s[(i + n - 3) % n], s[(i + 3) % n]);
            let t = [b[0] - a[0], b[1] - a[1]];
            let l = t[0].hypot(t[1]).max(1e-9);
            let e = [pts[i][0] as f64 - s[i][0], pts[i][1] as f64 - s[i][1]];
            dev[i] = (t[0] * e[1] - t[1] * e[0]) / l;
        }
        // Zig-zags: sign changes of the offset that clear 0.6 px each way.
        let mut event = vec![0usize; n];
        let mut state = 0i32;
        for i in 0..n {
            if dev[i] > 0.6 && state <= 0 {
                state = 1;
                event[i] = 1;
            } else if dev[i] < -0.6 && state >= 0 {
                state = -1;
                event[i] = 1;
            }
        }
        let half = 15isize;
        let mut flag = vec![false; n];
        let mut rms_at = vec![0.0f64; n];
        for i in 0..n as isize {
            let (mut ss, mut ev) = (0.0f64, 0usize);
            for k in -half..=half {
                let j = (i + k).rem_euclid(n as isize) as usize;
                ss += dev[j] * dev[j];
                ev += event[j];
            }
            let rms = (ss / (2 * half + 1) as f64).sqrt();
            rms_at[i as usize] = rms;
            let q = at(i);
            let on_border = q[0] <= 1 || q[1] <= 1 || q[0] >= w as i32 - 2 || q[1] >= h as i32 - 2;
            flag[i as usize] = !on_border && rms >= JAG_RMS_PX && ev >= JAG_STAIRS;
        }
        out.contour_px += n;
        out.flagged_px += flag.iter().filter(|&&f| f).count();
        out.worst_rms = out.worst_rms.max(rms_at.iter().cloned().fold(0.0, f64::max));
        // Runs of flagged points, merging gaps up to 8.
        let mut i = 0;
        while i < n {
            if !flag[i] {
                i += 1;
                continue;
            }
            let start = i;
            let mut end = i;
            let mut j = i;
            while j < n && j <= end + 8 {
                if flag[j] {
                    end = j;
                }
                j += 1;
            }
            i = end + 1;
            if end - start + 1 < 12 {
                continue;
            }
            let run: Vec<[i32; 2]> = (start..=end).map(|k| pts[k]).collect();
            let peak = (start..=end).map(|k| rms_at[k]).fold(0.0, f64::max);
            let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
            for p in &run {
                x0 = x0.min(p[0]);
                y0 = y0.min(p[1]);
                x1 = x1.max(p[0]);
                y1 = y1.max(p[1]);
            }
            let mut d = Defect { kind: Kind::Jagged, raw: peak, score: peak / JAG_RMS_PX, note: String::new(), tris: Vec::new(), anchors: Vec::new(), boxes: Vec::new(), island: None, screen: Some(Screen { view: sc.name.clone(), px: sc.px, bbox: [x0 - 6, y0 - 6, x1 + 6, y1 + 6] }) };
            let (mut alpha_edge, mut samples) = (0usize, 0usize);
            for p in run.iter().step_by(2) {
                let Some((mi, ti, uv)) = sc.hit(model, p[0] as usize, p[1] as usize) else { continue };
                if !d.tris.contains(&(mi, ti)) && d.tris.len() < 48 {
                    d.tris.push((mi, ti));
                }
                if let (Some(t), Some(img)) = (mesh_tex(model, tex, mi), model.meshes[mi].image) {
                    let (tx, ty) = ((uv[0] * t.w as f32) as i64, (uv[1] * t.h as f32) as i64);
                    push_box(&mut d.boxes, img, tx, ty, tx, ty, 2, t.w, t.h);
                    samples += 1;
                    let mut near_clear = false;
                    for dy in -1..=1i64 {
                        for dx in -1..=1i64 {
                            let (xx, yy) = ((tx + dx).clamp(0, t.w as i64 - 1) as usize, (ty + dy).clamp(0, t.h as i64 - 1) as usize);
                            near_clear |= t.rgba[(yy * t.w + xx) * 4 + 3] < 128;
                        }
                    }
                    alpha_edge += near_clear as usize;
                }
            }
            if let Some(&(mi, ti)) = d.tris.first() {
                d.island = None;
                let _ = (mi, ti);
            }
            let origin = if samples > 0 && alpha_edge * 2 >= samples { "painted alpha edge (transparent texels beside it)" } else { "geometry edge (no transparent texels beside it)" };
            d.note = format!("stair-stepped contour: {:.1} px rms off the smoothed curve over {} px of edge, {}", peak, end - start + 1, origin);
            out.defects.push(d);
        }
    }
    out
}

/// Colours of the part's pixels away from its edge (every third pixel), for building a palette.
pub fn collect_labs(sc: &ShotCtx, kg: &[bool], out: &mut Vec<[f32; 3]>) {
    let mask = erode(&sc.mask(kg), sc.px, sc.px, 2);
    for (i, &m) in mask.iter().enumerate() {
        if m && i % 3 == 0 {
            let c = sc.shot.rgb.px[i];
            out.push(lab([c[0] * 255.0, c[1] * 255.0, c[2] * 255.0]));
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Palette {
    pub centers: Vec<[f32; 3]>,
    pub weights: Vec<f32>,
}

/// K-means (k = 6, farthest-point start) in Lab; clusters under 3% of the pixels are not part of the palette.
pub fn palette(labs: &[[f32; 3]]) -> Palette {
    if labs.len() < 50 {
        return Palette::default();
    }
    let k = 6usize;
    let mut c: Vec<[f32; 3]> = vec![labs[labs.len() / 2]];
    while c.len() < k {
        let far = labs.iter().step_by(7).max_by(|a, b| c.iter().map(|q| de(**a, *q)).fold(f32::MAX, f32::min).total_cmp(&c.iter().map(|q| de(**b, *q)).fold(f32::MAX, f32::min))).copied().unwrap();
        c.push(far);
    }
    let mut cnt = vec![0usize; k];
    for _ in 0..12 {
        let mut sum = vec![[0.0f64; 3]; k];
        cnt = vec![0; k];
        for p in labs.iter().step_by(3) {
            let j = (0..k).min_by(|&a, &b| de(*p, c[a]).total_cmp(&de(*p, c[b]))).unwrap();
            for d in 0..3 {
                sum[j][d] += p[d] as f64;
            }
            cnt[j] += 1;
        }
        for j in 0..k {
            if cnt[j] > 0 {
                c[j] = std::array::from_fn(|d| (sum[j][d] / cnt[j] as f64) as f32);
            }
        }
    }
    let total: usize = cnt.iter().sum::<usize>().max(1);
    let mut pal = Palette::default();
    for j in 0..k {
        if cnt[j] as f32 >= 0.03 * total as f32 {
            pal.centers.push(c[j]);
            pal.weights.push(cnt[j] as f32 / total as f32);
        }
    }
    pal
}

#[derive(Default)]
pub struct DiscOut {
    pub defects: Vec<Defect>,
    pub part_px: usize,
    pub outlier_px: usize,
}

/// Blobs of part pixels that sit far from the palette and from the ring around them.
pub fn discolor(model: &Model, tex: &[Option<Texture>], sc: &ShotCtx, kg: &[bool], pal: &Palette, min_area: usize) -> DiscOut {
    let (w, h) = (sc.px, sc.px);
    let full = sc.mask(kg);
    let mask = erode(&full, w, h, 2);
    let labs: Vec<[f32; 3]> = (0..w * h)
        .map(|i| {
            let c = sc.shot.rgb.px[i];
            lab([c[0] * 255.0, c[1] * 255.0, c[2] * 255.0])
        })
        .collect();
    let mut out = DiscOut::default();
    out.part_px = mask.iter().filter(|&&m| m).count();
    if pal.centers.is_empty() {
        return out;
    }
    let dpal: Vec<f32> = (0..w * h).map(|i| if mask[i] { pal.centers.iter().map(|c| de(labs[i], *c)).fold(f32::MAX, f32::min) } else { 0.0 }).collect();
    let outlier: Vec<bool> = (0..w * h).map(|i| mask[i] && dpal[i] >= DISC_DE).collect();
    let mut seen = vec![false; w * h];
    for s in 0..w * h {
        if !outlier[s] || seen[s] {
            continue;
        }
        let mut stack = vec![s];
        seen[s] = true;
        let mut mem = Vec::new();
        while let Some(i) = stack.pop() {
            mem.push(i);
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            for (dx, dy) in DIRS8 {
                let (xx, yy) = (x + dx as i64, y + dy as i64);
                if xx < 0 || yy < 0 || xx as usize >= w || yy as usize >= h {
                    continue;
                }
                let j = yy as usize * w + xx as usize;
                if outlier[j] && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        if mem.len() < min_area {
            continue;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        let mut sum = [0.0f64; 3];
        for &i in &mem {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            for d in 0..3 {
                sum[d] += labs[i][d] as f64;
            }
        }
        let mean: [f32; 3] = std::array::from_fn(|d| (sum[d] / mem.len() as f64) as f32);
        // The ring: part pixels within 5 px of the blob's box that are not in the blob.
        let in_blob: std::collections::HashSet<usize> = mem.iter().copied().collect();
        let (mut rs, mut rn) = ([0.0f64; 3], 0usize);
        for y in y0.saturating_sub(5)..(y1 + 6).min(h) {
            for x in x0.saturating_sub(5)..(x1 + 6).min(w) {
                let i = y * w + x;
                if mask[i] && !in_blob.contains(&i) && !outlier[i] {
                    for d in 0..3 {
                        rs[d] += labs[i][d] as f64;
                    }
                    rn += 1;
                }
            }
        }
        if rn < 10 {
            continue;
        }
        let ring: [f32; 3] = std::array::from_fn(|d| (rs[d] / rn as f64) as f32);
        let ring_de = de(mean, ring);
        if ring_de < DISC_RING_DE {
            continue;
        }
        let raw = mem.iter().map(|&i| dpal[i] as f64).sum::<f64>() / mem.len() as f64;
        out.outlier_px += mem.len();
        let mut d = Defect { kind: Kind::Discolor, raw, score: raw / DISC_DE as f64, note: String::new(), tris: Vec::new(), anchors: Vec::new(), boxes: Vec::new(), island: None, screen: Some(Screen { view: sc.name.clone(), px: sc.px, bbox: [x0 as i32 - 8, y0 as i32 - 8, x1 as i32 + 8, y1 as i32 + 8] }) };
        for &i in mem.iter().step_by((mem.len() / 40).max(1)) {
            let Some((mi, ti, uv)) = sc.hit(model, i % w, i / w) else { continue };
            if !d.tris.contains(&(mi, ti)) && d.tris.len() < 48 {
                d.tris.push((mi, ti));
            }
            if let (Some(t), Some(img)) = (mesh_tex(model, tex, mi), model.meshes[mi].image) {
                let (tx, ty) = ((uv[0] * t.w as f32) as i64, (uv[1] * t.h as f32) as i64);
                push_box(&mut d.boxes, img, tx, ty, tx, ty, 3, t.w, t.h);
            }
        }
        let dl = mean[0] - ring[0];
        d.note = format!(
            "{} px patch, delta-E {:.0} from the part's palette and {:.0} from its surroundings ({}; L {:+.0}, a {:+.0}, b {:+.0} against the ring)",
            mem.len(),
            raw,
            ring_de,
            if dl <= -10.0 { "darker" } else if dl >= 10.0 { "lighter" } else { "different hue" },
            dl,
            mean[1] - ring[1],
            mean[2] - ring[2]
        );
        out.defects.push(d);
    }
    out
}

/// Places a mesh-based defect in the best of the drawn views: the one where most of its anchor
/// vertices are visible (the triangle under the projected pixel belongs to the defect). Returns the
/// view and the pixel box.
pub fn place(model: &Model, d: &Defect, views: &[ShotCtx]) -> Option<Screen> {
    let mut best: Option<(usize, f32, Screen)> = None;
    for sc in views {
        let (mut seen, mut n) = (0usize, 0usize);
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &(mi, v) in &d.anchors {
            n += 1;
            let p = sc.cam.project(sc.px, sc.px, sc.pose[mi][v as usize]);
            let (x, y) = (p[0].round() as i64, p[1].round() as i64);
            if x < 0 || y < 0 || x >= sc.px as i64 || y >= sc.px as i64 {
                continue;
            }
            let mut ok = false;
            'o: for dy in -2..=2i64 {
                for dx in -2..=2i64 {
                    let (xx, yy) = (x + dx, y + dy);
                    if xx < 0 || yy < 0 || xx >= sc.px as i64 || yy >= sc.px as i64 {
                        continue;
                    }
                    let g = sc.shot.pick[yy as usize * sc.px + xx as usize];
                    if g >= 0 {
                        let (gm, gt) = render::split_id(model, g as usize);
                        if d.tris.contains(&(gm, gt)) {
                            ok = true;
                            break 'o;
                        }
                    }
                }
            }
            if ok {
                seen += 1;
                x0 = x0.min(p[0]);
                y0 = y0.min(p[1]);
                x1 = x1.max(p[0]);
                y1 = y1.max(p[1]);
            }
        }
        if seen == 0 || n == 0 {
            continue;
        }
        let frac = seen as f32 / n as f32;
        // Prefer views that show most anchors, then the more zoomed-in one.
        let rank = frac * 1000.0 + (1.0 / sc.cam.ortho_half.max(1e-3)).min(900.0) * 0.01;
        let pad = (0.04 * sc.px as f32).max(10.0);
        let (cx, cy) = (0.5 * (x0 + x1), 0.5 * (y0 + y1));
        let half = (0.5 * (x1 - x0).max(y1 - y0) + pad).max(0.06 * sc.px as f32);
        let scr = Screen { view: sc.name.clone(), px: sc.px, bbox: [(cx - half) as i32, (cy - half) as i32, (cx + half) as i32, (cy + half) as i32] };
        if best.as_ref().map_or(true, |b| rank > b.1) {
            best = Some((0, rank, scr));
        }
    }
    best.map(|b| b.2)
}

// ---------- sharpness, flat patches and direction cuts (screen tiles) ----------

/// Statistics of one 16 x 16 tile of part pixels.
#[derive(Clone, Debug)]
pub struct Tile {
    /// RMS luminance gradient across the strands (the larger structure-tensor axis).
    pub across: f64,
    /// Variance of L, a and b summed.
    pub var: f64,
    /// Strand axis in radians, mod pi (screen space, y down).
    pub axis: f64,
    /// 0 for no preferred direction, 1 for one clean direction.
    pub coherence: f64,
    pub tri: Option<(usize, usize)>,
}

pub struct TileGrid {
    pub tile: usize,
    pub gw: usize,
    pub gh: usize,
    pub tiles: Vec<Option<Tile>>,
}

fn lab_image(sc: &ShotCtx) -> Vec<[f32; 3]> {
    sc.shot.rgb.px.iter().map(|c| lab([c[0] * 255.0, c[1] * 255.0, c[2] * 255.0])).collect()
}

/// Per-tile contrast, variance and strand axis over the part's pixels in one view.
pub fn tiles(model: &Model, sc: &ShotCtx, kg: &[bool], tile: usize) -> TileGrid {
    let tl = tile;
    let (w, h) = (sc.px, sc.px);
    let inner = erode(&sc.mask(kg), w, h, 3);
    let labs = lab_image(sc);
    let (gw, gh) = (w / tl, h / tl);
    let mut grid = TileGrid { tile: tl, gw, gh, tiles: vec![None; gw * gh] };
    for gy in 0..gh {
        for gx in 0..gw {
            let (x0, y0) = (gx * tl, gy * tl);
            let n_in = (y0..y0 + tl).map(|y| (x0..x0 + tl).filter(|&x| inner[y * w + x]).count()).sum::<usize>();
            if (n_in as f64) < 0.9 * (tl * tl) as f64 {
                continue;
            }
            let (mut jxx, mut jyy, mut jxy) = (0.0f64, 0.0f64, 0.0f64);
            let mut sum = [0.0f64; 3];
            let mut sq = [0.0f64; 3];
            for y in y0..y0 + tl {
                for x in x0..x0 + tl {
                    let i = y * w + x;
                    let l = labs[i];
                    for d in 0..3 {
                        sum[d] += l[d] as f64;
                        sq[d] += (l[d] as f64) * (l[d] as f64);
                    }
                    let gx_ = 0.5 * (labs[i + 1][0] - labs[i - 1][0]) as f64;
                    let gy_ = 0.5 * (labs[i + w][0] - labs[i - w][0]) as f64;
                    jxx += gx_ * gx_;
                    jyy += gy_ * gy_;
                    jxy += gx_ * gy_;
                }
            }
            let n = (tl * tl) as f64;
            let var: f64 = (0..3).map(|d| sq[d] / n - (sum[d] / n) * (sum[d] / n)).sum();
            let tr = jxx + jyy;
            let disc = ((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy).sqrt();
            let (l1, l2) = (0.5 * (tr + disc), 0.5 * (tr - disc));
            let grad_dir = 0.5 * (2.0 * jxy).atan2(jxx - jyy);
            let axis = (grad_dir + std::f64::consts::FRAC_PI_2).rem_euclid(std::f64::consts::PI);
            let coh = if tr > 1e-9 { (l1 - l2) / tr } else { 0.0 };
            let tri = {
                let g = sc.shot.pick[(y0 + tl / 2) * w + x0 + tl / 2];
                (g >= 0).then(|| render::split_id(model, g as usize))
            };
            grid.tiles[gy * gw + gx] = Some(Tile { across: (l1.max(0.0) / n).sqrt(), var: var.max(0.0), axis, coherence: coh, tri });
        }
    }
    grid
}

/// Medians of the reference's tiles: what "sharp enough" and "varied enough" mean for this part.
#[derive(Clone, Debug, Default)]
pub struct Baseline {
    pub per_view: HashMap<String, (f64, f64)>,
    pub global: (f64, f64),
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

impl Baseline {
    pub fn from_grids(grids: &[(String, &TileGrid)]) -> Baseline {
        let mut b = Baseline::default();
        let (mut ga, mut gv) = (Vec::new(), Vec::new());
        for (name, g) in grids {
            let (mut a, mut v): (Vec<f64>, Vec<f64>) = g.tiles.iter().flatten().map(|t| (t.across, t.var)).unzip();
            if a.len() >= 6 {
                b.per_view.insert(name.clone(), (median(&mut a.clone()), median(&mut v.clone())));
            }
            ga.append(&mut a);
            gv.append(&mut v);
        }
        b.global = (median(&mut ga), median(&mut gv));
        b
    }
    /// (across-strand contrast, colour variance) medians that apply to `view`.
    pub fn view_baseline(&self, view: &str) -> (f64, f64) {
        self.get(view)
    }
    fn get(&self, view: &str) -> (f64, f64) {
        self.per_view.get(view).copied().unwrap_or(self.global)
    }
}

fn attach_pixels(d: &mut Defect, model: &Model, tex: &[Option<Texture>], sc: &ShotCtx, pix: &[usize]) {
    let step = (pix.len() / 80).max(1);
    for &i in pix.iter().step_by(step) {
        let Some((mi, ti, uv)) = sc.hit(model, i % sc.px, i / sc.px) else { continue };
        if !d.tris.contains(&(mi, ti)) && d.tris.len() < 48 {
            d.tris.push((mi, ti));
        }
        if let (Some(t), Some(img)) = (mesh_tex(model, tex, mi), model.meshes[mi].image) {
            let (tx, ty) = ((uv[0] * t.w as f32) as i64, (uv[1] * t.h as f32) as i64);
            push_box(&mut d.boxes, img, tx, ty, tx, ty, 3, t.w, t.h);
        }
    }
}

fn tile_pixels(sc: &ShotCtx, tile: usize, gx: usize, gy: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for dy in [tile / 8, tile / 2, tile - 1 - tile / 8] {
        for dx in [tile / 8, tile / 2, tile - 1 - tile / 8] {
            out.push((gy * tile + dy) * sc.px + gx * tile + dx);
        }
    }
    out
}

/// Groups cells (by index, on a gw x gh grid) that touch along an edge.
fn grid_groups(gw: usize, gh: usize, member: &[bool], diag: bool) -> Vec<Vec<usize>> {
    let mut seen = vec![false; gw * gh];
    let mut out = Vec::new();
    for s in 0..gw * gh {
        if !member[s] || seen[s] {
            continue;
        }
        let mut stack = vec![s];
        seen[s] = true;
        let mut g = Vec::new();
        while let Some(i) = stack.pop() {
            g.push(i);
            let (x, y) = ((i % gw) as i64, (i / gw) as i64);
            for &(dx, dy) in if diag { &DIRS8_I64[..] } else { &DIRS4_I64[..] } {
                let (xx, yy) = (x + dx, y + dy);
                if xx < 0 || yy < 0 || xx as usize >= gw || yy as usize >= gh {
                    continue;
                }
                let j = yy as usize * gw + xx as usize;
                if member[j] && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        out.push(g);
    }
    out
}

#[derive(Default)]
pub struct SharpOut {
    pub defects: Vec<Defect>,
    pub measured: usize,
    pub flagged_tiles: usize,
}

/// Connected tiles a patch needs before it counts (a tile is 8 px for flat patches, 16 for smear).
fn min_tiles(kind: Kind) -> usize {
    if kind == Kind::Flat {
        6
    } else {
        4
    }
}

/// Smeared (low contrast across the strands) or flat (almost no colour variance) patches, against
/// the reference's medians for the same view and tile size. `kind` is `Kind::Blur` or `Kind::Flat`.
pub fn sharpness(model: &Model, tex: &[Option<Texture>], sc: &ShotCtx, grid: &TileGrid, base: &Baseline, kind: Kind) -> SharpOut {
    let mut out = SharpOut::default();
    let (ba, bv) = base.get(&sc.name);
    if ba <= 1e-6 || bv <= 1e-6 {
        return out;
    }
    let n = grid.gw * grid.gh;
    let mut member = vec![false; n];
    for (i, t) in grid.tiles.iter().enumerate() {
        let Some(t) = t else { continue };
        out.measured += 1;
        member[i] = if kind == Kind::Flat { t.var * FLAT_RATIO <= bv } else { t.across * BLUR_RATIO <= ba };
    }
    let tl = grid.tile;
    for g in grid_groups(grid.gw, grid.gh, &member, false) {
        if g.len() < min_tiles(kind) {
            continue;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        let mut pix = Vec::new();
        let mut raw = 0.0f64;
        for &i in &g {
            let (gx, gy) = (i % grid.gw, i / grid.gw);
            x0 = x0.min(gx * tl);
            y0 = y0.min(gy * tl);
            x1 = x1.max(gx * tl + tl);
            y1 = y1.max(gy * tl + tl);
            pix.extend(tile_pixels(sc, tl, gx, gy));
            let t = grid.tiles[i].as_ref().unwrap();
            raw += if kind == Kind::Flat { (bv / t.var.max(bv / 60.0)).min(60.0) } else { (ba / t.across.max(ba / 20.0)).min(20.0) };
        }
        raw /= g.len() as f64;
        out.flagged_tiles += g.len();
        let mut d = Defect { kind, raw, score: raw / kind.limit(), note: String::new(), tris: Vec::new(), anchors: Vec::new(), boxes: Vec::new(), island: None, screen: Some(Screen { view: sc.name.clone(), px: sc.px, bbox: [x0 as i32, y0 as i32, x1 as i32, y1 as i32] }) };
        attach_pixels(&mut d, model, tex, sc, &pix);
        d.note = if kind == Kind::Flat {
            format!("flat patch: {} px across with {:.0} times less colour variance than the reference's median; reads as a plastic area with no strands", x1 - x0, raw)
        } else {
            format!("smeared patch: {} px across with {:.1} times less contrast across the strands than the reference's median; strands are blurred together", x1 - x0, raw)
        };
        out.defects.push(d);
    }
    out
}

#[derive(Default)]
pub struct LayerOut {
    pub defects: Vec<Defect>,
    pub boundary_px: usize,
    pub flagged_px: usize,
}

/// Hard cuts between two triangles of the part: a run of pixels where the triangle changes and the
/// colour two pixels either side differs by more than `LAYER_DE`. That is where one layer ends and
/// the next starts with different strands, tone or direction.
pub fn layer_edges(model: &Model, tex: &[Option<Texture>], sc: &ShotCtx, kg: &[bool]) -> LayerOut {
    let (w, h) = (sc.px, sc.px);
    let pick = &sc.shot.pick;
    let labs = lab_image(sc);
    let inpart = |g: i32| g >= 0 && kg.get(g as usize).copied().unwrap_or(false);
    let mut out = LayerOut::default();
    let mut step = vec![0.0f32; w * h];
    let mut hard = vec![false; w * h];
    for y in 3..h - 3 {
        for x in 3..w - 3 {
            let i = y * w + x;
            let a = pick[i];
            if !inpart(a) {
                continue;
            }
            for (dx, dy) in [(1usize, 0usize), (0, 1)] {
                let s = dy * w + dx;
                let j = i + s;
                let b = pick[j];
                if b == a || !inpart(b) {
                    continue;
                }
                // Clean two-pixel runs on both sides: skip slivers and anti-aliased corners.
                if pick[i - s] != a || pick[i - 2 * s] != a || pick[j + s] != b || pick[j + 2 * s] != b {
                    continue;
                }
                out.boundary_px += 1;
                let avg = |p: usize, q: usize| [0.5 * (labs[p][0] + labs[q][0]), 0.5 * (labs[p][1] + labs[q][1]), 0.5 * (labs[p][2] + labs[q][2])];
                let d = de(avg(i - s, i - 2 * s), avg(j + s, j + 2 * s));
                if d as f64 >= LAYER_DE {
                    out.flagged_px += 1;
                    for k in [i, j] {
                        hard[k] = true;
                        step[k] = step[k].max(d);
                    }
                }
            }
        }
    }
    let mut seen = vec![false; w * h];
    for s0 in 0..w * h {
        if !hard[s0] || seen[s0] {
            continue;
        }
        let mut stack = vec![s0];
        seen[s0] = true;
        let mut mem = Vec::new();
        while let Some(i) = stack.pop() {
            mem.push(i);
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            for (dx, dy) in DIRS8 {
                let (xx, yy) = (x + dx as i64, y + dy as i64);
                if xx < 0 || yy < 0 || xx as usize >= w || yy as usize >= h {
                    continue;
                }
                let j = yy as usize * w + xx as usize;
                if hard[j] && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        if mem.len() < LAYER_RUN_PX {
            continue;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        for &p in &mem {
            x0 = x0.min(p % w);
            y0 = y0.min(p / w);
            x1 = x1.max(p % w);
            y1 = y1.max(p / w);
        }
        let mean = mem.iter().map(|&p| step[p] as f64).sum::<f64>() / mem.len() as f64;
        let mut d = Defect { kind: Kind::LayerCut, raw: mean, score: mean / LAYER_DE, note: String::new(), tris: Vec::new(), anchors: Vec::new(), boxes: Vec::new(), island: None, screen: Some(Screen { view: sc.name.clone(), px: sc.px, bbox: [x0 as i32 - 8, y0 as i32 - 8, x1 as i32 + 8, y1 as i32 + 8] }) };
        attach_pixels(&mut d, model, tex, sc, &mem);
        d.note = format!("hard cut {} px long between two triangles of the part: colour steps by {:.0} (Lab) across it, so one layer ends and the next starts with different tone or strands", mem.len(), mean);
        out.defects.push(d);
    }
    out
}

#[derive(Default)]
pub struct SpeckOut {
    pub defects: Vec<Defect>,
    pub fragments: usize,
    pub dots: usize,
    pub px: usize,
}

/// Specks in a render: small cut-off fragments of the part (their own connected pieces) and
/// clusters of one- or two-pixel dots whose colour differs from both sides in most directions
/// (pin holes through a card, orange or dark dots on a flat area).
pub fn specks(model: &Model, tex: &[Option<Texture>], sc: &ShotCtx, kg: &[bool]) -> SpeckOut {
    let (w, h) = (sc.px, sc.px);
    let mask = sc.mask(kg);
    let mut out = SpeckOut::default();
    let mut seen = vec![false; w * h];
    struct Frag {
        pix: Vec<usize>,
        cx: f64,
        cy: f64,
    }
    let mut frags: Vec<Frag> = Vec::new();
    for s in 0..w * h {
        if !mask[s] || seen[s] {
            continue;
        }
        let mut stack = vec![s];
        seen[s] = true;
        let mut mem = Vec::new();
        let mut border = false;
        while let Some(i) = stack.pop() {
            mem.push(i);
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            if x == 0 || y == 0 || x as usize == w - 1 || y as usize == h - 1 {
                border = true;
            }
            for (dx, dy) in DIRS8 {
                let (xx, yy) = (x + dx as i64, y + dy as i64);
                if xx < 0 || yy < 0 || xx as usize >= w || yy as usize >= h {
                    continue;
                }
                let j = yy as usize * w + xx as usize;
                if mask[j] && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        if border || mem.len() > SPECK_PX || mem.len() < SPECK_MIN_PX {
            continue;
        }
        let cx = mem.iter().map(|&i| (i % w) as f64).sum::<f64>() / mem.len() as f64;
        let cy = mem.iter().map(|&i| (i / w) as f64).sum::<f64>() / mem.len() as f64;
        out.px += mem.len();
        frags.push(Frag { pix: mem, cx, cy });
    }
    out.fragments = frags.len();
    // Fragments within 24 px of each other are one defect (a spray of specks).
    let mut uf: Vec<usize> = (0..frags.len()).collect();
    for a in 0..frags.len() {
        for b in a + 1..frags.len() {
            if (frags[a].cx - frags[b].cx).hypot(frags[a].cy - frags[b].cy) <= 24.0 {
                union(&mut uf, a, b);
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..frags.len() {
        let r = find(&mut uf, i);
        groups.entry(r).or_default().push(i);
    }
    for (_, g) in groups {
        let mean_area = g.iter().map(|&i| frags[i].pix.len() as f64).sum::<f64>() / g.len() as f64;
        let raw = (SPECK_PX as f64 / mean_area).sqrt().min(2.0);
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        let mut pix = Vec::new();
        for &i in &g {
            for &p in &frags[i].pix {
                x0 = x0.min(p % w);
                y0 = y0.min(p / w);
                x1 = x1.max(p % w);
                y1 = y1.max(p / w);
            }
            pix.extend(frags[i].pix.iter().copied());
        }
        let mut d = Defect { kind: Kind::Speck, raw, score: raw / Kind::Speck.limit(), note: String::new(), tris: Vec::new(), anchors: Vec::new(), boxes: Vec::new(), island: None, screen: Some(Screen { view: sc.name.clone(), px: sc.px, bbox: [x0 as i32 - 10, y0 as i32 - 10, x1 as i32 + 10, y1 as i32 + 10] }) };
        attach_pixels(&mut d, model, tex, sc, &pix);
        d.note = format!("{} cut-off fragment(s) of the part, mean {:.0} px each (limit {} px): specks or shards floating apart from the main shape", g.len(), mean_area, SPECK_PX);
        out.defects.push(d);
    }

    // Dots: a pixel whose colour differs from the 3 x 3 median of its neighbourhood (taken per
    // channel) by `DOT_DE`. The median drops one- and two-pixel dots even when they sit close
    // together, and keeps strands (2 px wide and more) and straight edges.
    let labs = lab_image(sc);
    let inner = erode(&mask, w, h, 3);
    let mut dot = vec![false; w * h];
    for y in 3..h - 3 {
        for x in 3..w - 3 {
            let i = y * w + x;
            if !inner[i] {
                continue;
            }
            let mut med = [0.0f32; 3];
            for (c, m) in med.iter_mut().enumerate() {
                let mut v = [0.0f32; 9];
                let mut k = 0;
                for dy in 0..3 {
                    for dx in 0..3 {
                        v[k] = labs[(y + dy - 1) * w + x + dx - 1][c];
                        k += 1;
                    }
                }
                v.sort_by(|a, b| a.total_cmp(b));
                *m = v[4];
            }
            // Dither: a pixel that differs from all four neighbours is part of a checker of two tones,
            // which no strand edge or card border produces (those differ on one or two sides).
            let four = [i - 1, i + 1, i - w, i + w].iter().all(|&j| de(labs[i], labs[j]) >= DITHER_DE);
            dot[i] = four || de(labs[i], med) >= DOT_DE;
        }
    }
    let tl = 16usize;
    let (gw, gh) = (w / tl, h / tl);
    let mut count = vec![0usize; gw * gh];
    for (i, &d) in dot.iter().enumerate() {
        if d && (i % w) / tl < gw && (i / w) / tl < gh {
            count[(i / w) / tl * gw + (i % w) / tl] += 1;
        }
    }
    out.dots = count.iter().sum();
    out.px += out.dots;
    let member: Vec<bool> = count.iter().map(|&c| c >= DOT_TILE_MIN).collect();
    for g in grid_groups(gw, gh, &member, true) {
        let n: usize = g.iter().map(|&i| count[i]).sum();
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        let mut pix = Vec::new();
        for &i in &g {
            let (gx, gy) = (i % gw, i / gw);
            x0 = x0.min(gx * tl);
            y0 = y0.min(gy * tl);
            x1 = x1.max(gx * tl + tl);
            y1 = y1.max(gy * tl + tl);
            for y in gy * tl..gy * tl + tl {
                for x in gx * tl..gx * tl + tl {
                    if dot[y * w + x] {
                        pix.push(y * w + x);
                    }
                }
            }
        }
        let raw = (1.0 + n as f64 / 12.0).min(4.0);
        let mut d = Defect { kind: Kind::Speck, raw, score: raw / Kind::Speck.limit(), note: String::new(), tris: Vec::new(), anchors: Vec::new(), boxes: Vec::new(), island: None, screen: Some(Screen { view: sc.name.clone(), px: sc.px, bbox: [x0 as i32, y0 as i32, x1 as i32, y1 as i32] }) };
        attach_pixels(&mut d, model, tex, sc, &pix);
        d.note = format!("{} dots in {} px: one- or two-pixel specks whose colour differs from their surroundings (pin holes through a card, or stray dark or orange texels); the render shows them as noise", n, x1 - x0);
        out.defects.push(d);
    }
    out
}

#[derive(Default)]
pub struct TexNoiseOut {
    pub stipple_defects: Vec<Defect>,
    pub stipple_load: Load,
}

/// Alpha noise under the part: tiny closed holes and tiny opaque scraps inside the covered area
/// (they turn into stipple and pin holes at the cutout), found per image in the texels the part's
/// UVs cover. A soft alpha edge is not noise and is not counted.
pub fn texture_noise(model: &Model, topo: &Topo, tex: &[Option<Texture>], keep: &Keep) -> TexNoiseOut {
    let offs = render::tri_offsets(model);
    let mut out = TexNoiseOut::default();
    let (mut win_total, mut win_bad, mut cover_total, mut tiny_total) = (0usize, 0usize, 0usize, 0usize);
    let mut per_image: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (mi, m) in model.meshes.iter().enumerate() {
        if let Some(im) = m.image {
            if tex.get(im).is_some_and(|t| t.is_some()) {
                per_image.entry(im).or_default().push(mi);
            }
        }
    }
    for (im, meshes) in per_image {
        let t = tex[im].as_ref().unwrap();
        let (w, h) = (t.w, t.h);
        let mut tri_at = vec![0u32; w * h];
        for &mi in &meshes {
            let m = &model.meshes[mi];
            for (ti, tr) in m.tris.iter().enumerate() {
                if keep.get(mi).and_then(|k| k.get(ti)).copied().unwrap_or(false) {
                    raster_tri(&mut tri_at, w, h, std::array::from_fn(|k| m.uv[tr[k] as usize]), (offs[mi] + ti) as u32);
                }
            }
        }
        let covered = |i: usize| tri_at[i] != 0;
        let opaque = |i: usize| t.rgba[i * 4 + 3] >= 128;
        cover_total += tri_at.iter().filter(|&&v| v != 0).count();
        // Tiny components of one alpha class, closed inside the covered area.
        let mut flag = vec![false; w * h];
        let mut visited = vec![false; w * h];
        for s in 0..w * h {
            if !covered(s) || visited[s] {
                continue;
            }
            let cls = opaque(s);
            let dirs: &[(i64, i64)] = if cls { &DIRS8_I64 } else { &DIRS4_I64 };
            let mut stack = vec![s];
            visited[s] = true;
            let mut mem = Vec::new();
            let mut open = false;
            while let Some(i) = stack.pop() {
                if mem.len() <= STIPPLE_COMP {
                    mem.push(i);
                } else {
                    open = true;
                }
                let (x, y) = ((i % w) as i64, (i / w) as i64);
                for &(dx, dy) in dirs {
                    let (xx, yy) = (x + dx, y + dy);
                    if xx < 0 || yy < 0 || xx as usize >= w || yy as usize >= h {
                        open = true;
                        continue;
                    }
                    let j = yy as usize * w + xx as usize;
                    if !covered(j) {
                        open = true;
                        continue;
                    }
                    if opaque(j) == cls && !visited[j] {
                        visited[j] = true;
                        stack.push(j);
                    }
                }
            }
            if !open && mem.len() <= STIPPLE_COMP {
                tiny_total += 1;
                for &i in &mem {
                    flag[i] = true;
                }
            }
        }
        // Windows of 16 x 16, stride 8.
        let (ww, wh) = (w / 8, h / 8);
        let mut bad_win = vec![false; ww * wh];
        let mut frac_win = vec![0.0f64; ww * wh];
        for wy in 0..wh.saturating_sub(1) {
            for wx in 0..ww.saturating_sub(1) {
                let (x0, y0) = (wx * 8, wy * 8);
                let (mut cov, mut fl) = (0usize, 0usize);
                for y in y0..y0 + 16 {
                    let row = y * w;
                    for x in x0..x0 + 16 {
                        if covered(row + x) {
                            cov += 1;
                            if flag[row + x] {
                                fl += 1;
                            }
                        }
                    }
                }
                if cov >= 64 {
                    win_total += 1;
                    let f = fl as f64 / cov as f64;
                    frac_win[wy * ww + wx] = f;
                    if f >= STIPPLE_FRAC {
                        bad_win[wy * ww + wx] = true;
                        win_bad += 1;
                    }
                }
            }
        }
        for g in grid_groups(ww, wh, &bad_win, true) {
            let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
            let mut fsum = 0.0;
            for &i in &g {
                let (wx, wy) = (i % ww, i / ww);
                x0 = x0.min(wx * 8);
                y0 = y0.min(wy * 8);
                x1 = x1.max(wx * 8 + 16);
                y1 = y1.max(wy * 8 + 16);
                fsum += frac_win[i];
            }
            let raw = fsum / g.len() as f64;
            let mut d = Defect { kind: Kind::Stipple, raw, score: raw / STIPPLE_FRAC, note: String::new(), tris: Vec::new(), anchors: Vec::new(), boxes: Vec::new(), island: None, screen: None };
            let mut nflag = 0usize;
            for y in y0..y1.min(h) {
                for x in x0..x1.min(w) {
                    let i = y * w + x;
                    if flag[i] {
                        nflag += 1;
                        if covered(i) {
                            let (mi, ti) = render::split_id(model, tri_at[i] as usize - 1);
                            if !d.tris.contains(&(mi, ti)) && d.tris.len() < 48 {
                                d.tris.push((mi, ti));
                            }
                        }
                    }
                }
            }
            push_box(&mut d.boxes, im, x0 as i64, y0 as i64, x1 as i64 - 1, y1 as i64 - 1, 0, w, h);
            for &(mi, ti) in d.tris.iter().take(6) {
                for &v in &model.meshes[mi].tris[ti] {
                    d.anchors.push((mi, v));
                }
            }
            if let Some(&(mi, ti)) = d.tris.first() {
                d.island = Some((mi, topo.island[mi][ti]));
            }
            d.note = format!("{} texels in {}x{} sit in tiny alpha holes or scraps ({:.0}% of the covered texels): the cutout at alpha 0.5 turns them into stipple and pin holes", nflag, x1 - x0, y1 - y0, 100.0 * raw);
            out.stipple_defects.push(d);
        }
    }
    out.stipple_load = Load { percent: if win_total > 0 { 100.0 * win_bad as f64 / win_total as f64 } else { 0.0 }, measured: win_total as f64, flagged: win_bad as f64, extra: vec![("covered_texels".into(), cover_total as f64), ("tiny_components".into(), tiny_total as f64)] };
    out.stipple_defects.sort_by(|a, b| b.score.total_cmp(&a.score));
    out
}

/// Flat cards: a triangle whose visible pixels carry almost no colour variance (a plastic wedge or
/// a card filled with one tone), whatever sits next to it. Tiles miss these when the card is
/// narrower than a tile and its neighbours add contrast, so this reads the triangles themselves.
/// Triangles that touch are grouped; a group needs `FLAT_CARD_PX` pixels.
pub fn flat_cards(model: &Model, tex: &[Option<Texture>], sc: &ShotCtx, kg: &[bool], base: &Baseline) -> SharpOut {
    let mut out = SharpOut::default();
    let (_, bv) = base.get(&sc.name);
    if bv <= 1e-6 {
        return out;
    }
    let (w, h) = (sc.px, sc.px);
    let pick = &sc.shot.pick;
    let labs = lab_image(sc);
    let inpart = |g: i32| g >= 0 && kg.get(g as usize).copied().unwrap_or(false);
    // Per triangle: pixel count and Lab sums, over pixels whose four neighbours are the same triangle.
    let mut acc: HashMap<i32, (usize, [f64; 3], [f64; 3])> = HashMap::new();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let a = pick[i];
            if !inpart(a) || pick[i - 1] != a || pick[i + 1] != a || pick[i - w] != a || pick[i + w] != a {
                continue;
            }
            let e = acc.entry(a).or_insert((0, [0.0; 3], [0.0; 3]));
            e.0 += 1;
            for c in 0..3 {
                let v = labs[i][c] as f64;
                e.1[c] += v;
                e.2[c] += v * v;
            }
        }
    }
    let mut flat: HashMap<i32, f64> = HashMap::new();
    for (&t, &(n, s, q)) in &acc {
        if n < 24 {
            continue;
        }
        out.measured += 1;
        let nf = n as f64;
        let var: f64 = (0..3).map(|c| (q[c] / nf - (s[c] / nf).powi(2)).max(0.0)).sum();
        if var * FLAT_RATIO <= bv {
            flat.insert(t, bv / var.max(bv / 60.0));
        }
    }
    if flat.is_empty() {
        return out;
    }
    // Pixels of flat triangles, grouped by 8-connectivity.
    let member: Vec<bool> = pick.iter().map(|g| flat.contains_key(g)).collect();
    let groups = grid_groups(w, h, &member, true);
    for g in groups {
        if g.len() < FLAT_CARD_PX {
            continue;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        let mut raw = 0.0f64;
        for &i in &g {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
            raw += flat[&pick[i]];
        }
        raw /= g.len() as f64;
        out.flagged_tiles += g.len();
        let mut d = Defect { kind: Kind::Flat, raw, score: raw / Kind::Flat.limit(), note: String::new(), tris: Vec::new(), anchors: Vec::new(), boxes: Vec::new(), island: None, screen: Some(Screen { view: sc.name.clone(), px: sc.px, bbox: [x0 as i32, y0 as i32, x1 as i32, y1 as i32] }) };
        let pix: Vec<usize> = g.clone();
        attach_pixels(&mut d, model, tex, sc, &pix);
        d.note = format!("flat card: {} px of triangles filled with one tone ({:.0} times less colour variance than the reference's median); reads as plastic", g.len(), raw);
        out.defects.push(d);
    }
    out
}

// ---------- lock scale ----------

/// Tile size (render px) for the lock-scale measure.
pub const LOCK_TILE: usize = 48;
/// Box half-widths (px) whose differences give four scale bands: fibre, strand, lock, broad shading.
pub const LOCK_RADII: [usize; 5] = [1, 3, 8, 20, 48];
/// A tile counts as wood grain when its fibre-scale energy share is this many times above the reference's median.
pub const LOCK_RATIO: f64 = 2.0;
/// Connected tiles a lock-scale patch needs.
pub const LOCK_MIN_TILES: usize = 3;

#[derive(Clone, Debug, Default)]
pub struct ScaleGrid {
    pub tile: usize,
    pub gw: usize,
    pub gh: usize,
    /// Per tile: mean squared Lab-L band energy at fibre, strand, lock and broad scale.
    pub bands: Vec<Option<[f32; 4]>>,
}

/// Energy of the part's luminance at four scales, per tile. Box means are normalised by the part
/// mask, so the background does not leak in.
pub fn scale_grid(sc: &ShotCtx, kg: &[bool]) -> ScaleGrid {
    let (w, h) = (sc.px, sc.px);
    let mask = sc.mask(kg);
    let inner = erode(&mask, w, h, 4);
    let labs = lab_image(sc);
    // Integral images of L * mask and mask, (w+1) x (h+1).
    let iw = w + 1;
    let mut s_l = vec![0.0f64; iw * (h + 1)];
    let mut s_m = vec![0.0f64; iw * (h + 1)];
    for y in 0..h {
        let (mut rl, mut rm) = (0.0f64, 0.0f64);
        for x in 0..w {
            if mask[y * w + x] {
                rl += labs[y * w + x][0] as f64;
                rm += 1.0;
            }
            s_l[(y + 1) * iw + x + 1] = s_l[y * iw + x + 1] + rl;
            s_m[(y + 1) * iw + x + 1] = s_m[y * iw + x + 1] + rm;
        }
    }
    let boxsum = |s: &[f64], x: usize, y: usize, r: usize| -> f64 {
        let (x0, y0) = (x.saturating_sub(r), y.saturating_sub(r));
        let (x1, y1) = ((x + r + 1).min(w), (y + r + 1).min(h));
        s[y1 * iw + x1] - s[y0 * iw + x1] - s[y1 * iw + x0] + s[y0 * iw + x0]
    };
    let tl = LOCK_TILE;
    let (gw, gh) = (w / tl, h / tl);
    let mut grid = ScaleGrid { tile: tl, gw, gh, bands: vec![None; gw * gh] };
    for gy in 0..gh {
        for gx in 0..gw {
            let (x0, y0) = (gx * tl, gy * tl);
            let n_in = (y0..y0 + tl).map(|y| (x0..x0 + tl).filter(|&x| inner[y * w + x]).count()).sum::<usize>();
            if (n_in as f64) < 0.85 * (tl * tl) as f64 {
                continue;
            }
            let mut e = [0.0f64; 4];
            let mut n = 0usize;
            for y in (y0..y0 + tl).step_by(2) {
                for x in (x0..x0 + tl).step_by(2) {
                    if !inner[y * w + x] {
                        continue;
                    }
                    let mut m = [0.0f64; 5];
                    for (k, &r) in LOCK_RADII.iter().enumerate() {
                        let c = boxsum(&s_m, x, y, r);
                        m[k] = if c > 0.0 { boxsum(&s_l, x, y, r) / c } else { 0.0 };
                    }
                    for b in 0..4 {
                        let d = m[b] - m[b + 1];
                        e[b] += d * d;
                    }
                    n += 1;
                }
            }
            if n > 0 {
                grid.bands[gy * gw + gx] = Some([(e[0] / n as f64) as f32, (e[1] / n as f64) as f32, (e[2] / n as f64) as f32, (e[3] / n as f64) as f32]);
            }
        }
    }
    grid
}

/// Share of a tile's luminance energy that sits at fibre scale (boxes of 1 to 3 px). Broad locks
/// put their energy at the larger scales and keep this share low; a surface of thin stripes with no
/// locks under them (wood grain) pushes it up.
pub fn lock_ratio(b: &[f32; 4]) -> f64 {
    b[0] as f64 / (b.iter().map(|&v| v as f64).sum::<f64>() + 1e-3)
}

/// Median fibre-scale energy share per view and over all views of a reference.
#[derive(Clone, Debug, Default)]
pub struct ScaleBase {
    pub per_view: HashMap<String, f64>,
    pub global: f64,
}

impl ScaleBase {
    pub fn from_grids(grids: &[(String, ScaleGrid)]) -> ScaleBase {
        let mut b = ScaleBase::default();
        let mut all = Vec::new();
        for (n, g) in grids {
            let mut v: Vec<f64> = g.bands.iter().flatten().map(lock_ratio).collect();
            if v.len() >= 6 {
                b.per_view.insert(n.clone(), median(&mut v.clone()));
            }
            all.append(&mut v);
        }
        b.global = median(&mut all);
        b
    }
    pub fn get(&self, view: &str) -> f64 {
        self.per_view.get(view).copied().unwrap_or(self.global)
    }
}

/// Patches where the fibre-scale share of the energy is `LOCK_RATIO` times above the reference's
/// median for the view: fine stripes with no broad lock under them (wood grain).
pub fn lock_scale(model: &Model, tex: &[Option<Texture>], sc: &ShotCtx, grid: &ScaleGrid, base: &ScaleBase) -> SharpOut {
    let mut out = SharpOut::default();
    let b = base.get(&sc.name);
    if b <= 1e-9 {
        return out;
    }
    let n = grid.gw * grid.gh;
    let mut member = vec![false; n];
    for (i, t) in grid.bands.iter().enumerate() {
        let Some(t) = t else { continue };
        out.measured += 1;
        member[i] = lock_ratio(t) >= LOCK_RATIO * b;
    }
    let tl = grid.tile;
    for g in grid_groups(grid.gw, grid.gh, &member, true) {
        if g.len() < LOCK_MIN_TILES {
            continue;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        let mut pix = Vec::new();
        let mut raw = 0.0f64;
        for &i in &g {
            let (gx, gy) = (i % grid.gw, i / grid.gw);
            x0 = x0.min(gx * tl);
            y0 = y0.min(gy * tl);
            x1 = x1.max(gx * tl + tl);
            y1 = y1.max(gy * tl + tl);
            for dy in [tl / 4, tl / 2, 3 * tl / 4] {
                for dx in [tl / 4, tl / 2, 3 * tl / 4] {
                    pix.push((gy * tl + dy) * sc.px + gx * tl + dx);
                }
            }
            raw += (lock_ratio(grid.bands[i].as_ref().unwrap()) / b).min(20.0);
        }
        raw /= g.len() as f64;
        out.flagged_tiles += g.len();
        let mut d = Defect { kind: Kind::Grain, raw, score: raw / Kind::Grain.limit(), note: String::new(), tris: Vec::new(), anchors: Vec::new(), boxes: Vec::new(), island: None, screen: Some(Screen { view: sc.name.clone(), px: sc.px, bbox: [x0 as i32, y0 as i32, x1 as i32, y1 as i32] }) };
        attach_pixels(&mut d, model, tex, sc, &pix);
        d.note = format!("lock-scale patch: {} px across with {:.1} times the reference's share of fibre-scale energy; reads as fine wood-grain stripes with no broad locks", x1 - x0, raw);
        out.defects.push(d);
    }
    out
}
