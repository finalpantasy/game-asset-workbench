//! Unity raw textures in the form Memoria's loader accepts at a `.png` path: a 60-byte header
//! (width, height, dataLength, TextureFormat, mipCount, 40 zero bytes) followed by the raw mip
//! chain, largest level first. Format numbers are Unity's `TextureFormat` (RGBA32 = 4, DXT1 = 10,
//! DXT5 = 12). Rows are stored bottom-up, so every level is flipped before it is written.
//!
//! Everything here is plain Rust: box-filter mips (optionally alpha-weighted and in linear light),
//! alpha-coverage-preserving scaling for cutout sheets, and a BC1 / BC3 encoder (principal-axis
//! start, least-squares endpoint refinement). Nothing is specific to FF9 or to a game engine
//! version beyond the header layout. `decode_raw` reads such a file back for checks.
use crate::Level;

pub const HEADER_LEN: usize = 60;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Rgba32,
    Dxt1,
    Dxt5,
}

impl Format {
    pub fn unity_value(self) -> u32 {
        match self {
            Format::Rgba32 => 4,
            Format::Dxt1 => 10,
            Format::Dxt5 => 12,
        }
    }
    pub fn from_unity(v: u32) -> Option<Format> {
        match v {
            4 => Some(Format::Rgba32),
            10 => Some(Format::Dxt1),
            12 => Some(Format::Dxt5),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Format::Rgba32 => "RGBA32",
            Format::Dxt1 => "DXT1",
            Format::Dxt5 => "DXT5",
        }
    }
    pub fn parse(s: &str) -> Option<Format> {
        match s.to_ascii_lowercase().as_str() {
            "rgba32" | "rgba" => Some(Format::Rgba32),
            "dxt1" | "bc1" => Some(Format::Dxt1),
            "dxt5" | "bc3" => Some(Format::Dxt5),
            _ => None,
        }
    }
    /// Bytes one mip level of this size occupies (DXT levels under 4x4 still take one block).
    pub fn level_bytes(self, w: usize, h: usize) -> usize {
        match self {
            Format::Rgba32 => w * h * 4,
            Format::Dxt1 => blocks(w, h) * 8,
            Format::Dxt5 => blocks(w, h) * 16,
        }
    }
}

fn blocks(w: usize, h: usize) -> usize {
    w.div_ceil(4).max(1) * h.div_ceil(4).max(1)
}

/// Level sizes of a full chain down to 1x1.
pub fn mip_dims(w: usize, h: usize) -> Vec<(usize, usize)> {
    let mut v = vec![(w, h)];
    let (mut a, mut b) = (w, h);
    while a > 1 || b > 1 {
        a = (a / 2).max(1);
        b = (b / 2).max(1);
        v.push((a, b));
    }
    v
}

pub fn mip_count(w: usize, h: usize) -> usize {
    mip_dims(w, h).len()
}

/// Byte length of `mips` levels (the header's dataLength).
pub fn data_len(fmt: Format, w: usize, h: usize, mips: usize) -> usize {
    mip_dims(w, h).iter().take(mips).map(|&(a, b)| fmt.level_bytes(a, b)).sum()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub width: u32,
    pub height: u32,
    pub data_len: u32,
    pub format: u32,
    pub mips: u32,
}

impl Header {
    pub fn to_bytes(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        for (i, v) in [self.width, self.height, self.data_len, self.format, self.mips].iter().enumerate() {
            b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        b
    }
    pub fn kind(&self) -> Option<Format> {
        Format::from_unity(self.format)
    }
}

/// Unity formats Memoria's loader accepts (AssetManager.LoadTextureGeneric).
const LOADER_FORMATS: [u32; 7] = [1, 3, 4, 5, 7, 10, 12];

/// Parses a raw texture the way the loader tests it: file length must be 60 + dataLength and the
/// format one of the accepted list. Anything else would fall through to `LoadImage` in the game.
pub fn parse_header(bytes: &[u8]) -> Result<Header, String> {
    if bytes.len() < HEADER_LEN {
        return Err(format!("{} bytes is shorter than the {HEADER_LEN}-byte header", bytes.len()));
    }
    let u = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    let h = Header { width: u(0), height: u(1), data_len: u(2), format: u(3), mips: u(4) };
    if bytes.len() != HEADER_LEN + h.data_len as usize {
        return Err(format!("file is {} bytes but the header says {} + {}", bytes.len(), HEADER_LEN, h.data_len));
    }
    if !LOADER_FORMATS.contains(&h.format) {
        return Err(format!("TextureFormat {} is not accepted by the loader", h.format));
    }
    Ok(h)
}

// ---------------------------------------------------------------- mips

#[derive(Clone, Debug)]
pub struct MipOpts {
    /// Rescale each level's alpha so the share of texels that pass the cutout test stays the same.
    pub coverage: bool,
    /// Cutout threshold in 0..=255: a texel passes when alpha >= cutoff (128 = the shaders' 0.5).
    pub cutoff: u8,
    /// Average colors in linear light instead of on the stored (gamma) values.
    pub linear: bool,
}

impl Default for MipOpts {
    fn default() -> Self {
        MipOpts { coverage: true, cutoff: 128, linear: false }
    }
}

fn srgb_to_linear(v: u8) -> f32 {
    let c = v as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(l: f32) -> u8 {
    let l = l.clamp(0.0, 1.0);
    let c = if l <= 0.0031308 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
    (c * 255.0 + 0.5) as u8
}

/// One 2x2 box step. Colors are weighted by alpha, so transparent texels' hidden colors do not
/// bleed into the level (a texel that is fully transparent everywhere in the block averages plainly).
fn downsample(src: &Level, linear: bool, lut: &[f32; 256]) -> Level {
    let (nw, nh) = ((src.w / 2).max(1), (src.h / 2).max(1));
    let mut out = vec![0u8; nw * nh * 4];
    for y in 0..nh {
        for x in 0..nw {
            let mut px = [[0u8; 4]; 4];
            let mut k = 0;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let sx = (2 * x + dx).min(src.w - 1);
                let sy = (2 * y + dy).min(src.h - 1);
                let i = (sy * src.w + sx) * 4;
                px[k] = [src.rgba[i], src.rgba[i + 1], src.rgba[i + 2], src.rgba[i + 3]];
                k += 1;
            }
            let asum: u32 = px.iter().map(|p| p[3] as u32).sum();
            let o = (y * nw + x) * 4;
            for c in 0..3 {
                let mut acc = 0f32;
                let mut wsum = 0f32;
                for p in &px {
                    let wgt = if asum > 0 { p[3] as f32 } else { 1.0 };
                    let v = if linear { lut[p[c] as usize] } else { p[c] as f32 };
                    acc += v * wgt;
                    wsum += wgt;
                }
                let m = acc / wsum;
                out[o + c] = if linear { linear_to_srgb(m) } else { (m + 0.5) as u8 };
            }
            out[o + 3] = ((asum + 2) / 4) as u8;
        }
    }
    Level { w: nw, h: nh, rgba: out }
}

/// Share of texels with alpha >= cutoff.
pub fn coverage(rgba: &[u8], cutoff: u8) -> f64 {
    let n = rgba.len() / 4;
    if n == 0 {
        return 0.0;
    }
    rgba.chunks_exact(4).filter(|p| p[3] >= cutoff).count() as f64 / n as f64
}

fn scaled(b: u8, s: f64) -> u8 {
    (b as f64 * s + 0.5).min(255.0) as u8
}

/// Alpha scale whose passing share is nearest `target` (histogram based). Coverage moves in steps,
/// so the nearer of the two scales on either side of the target step is taken.
fn find_scale(hist: &[u64; 256], total: u64, target: f64, cutoff: u8) -> f64 {
    let cov = |s: f64| -> f64 {
        let n: u64 = (0..256usize).filter(|&b| scaled(b as u8, s) >= cutoff).map(|b| hist[b]).sum();
        n as f64 / total as f64
    };
    let (mut lo, mut hi) = (0.0f64, 256.0f64);
    if cov(hi) < target {
        return hi;
    }
    for _ in 0..50 {
        let mid = 0.5 * (lo + hi);
        if cov(mid) >= target {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    if lo > 0.0 && (cov(lo) - target).abs() < (cov(hi) - target).abs() {
        lo
    } else {
        hi
    }
}

fn dup(l: &Level) -> Level {
    Level { w: l.w, h: l.h, rgba: l.rgba.clone() }
}

/// Full chain down to 1x1, level 0 first (it is `level0` itself, untouched).
pub fn build_chain(level0: Level, o: &MipOpts) -> Vec<Level> {
    let mut lut = [0f32; 256];
    for (i, l) in lut.iter_mut().enumerate() {
        *l = srgb_to_linear(i as u8);
    }
    let target = coverage(&level0.rgba, o.cutoff);
    let do_cov = o.coverage && target > 0.0 && target < 1.0;
    let mut cur = dup(&level0);
    let mut out = vec![level0];
    while cur.w > 1 || cur.h > 1 {
        let next = downsample(&cur, o.linear, &lut);
        let mut emit = dup(&next);
        if do_cov {
            let mut hist = [0u64; 256];
            for p in next.rgba.chunks_exact(4) {
                hist[p[3] as usize] += 1;
            }
            let s = find_scale(&hist, (next.w * next.h) as u64, target, o.cutoff);
            for p in emit.rgba.chunks_exact_mut(4) {
                p[3] = scaled(p[3], s);
            }
        }
        out.push(emit);
        cur = next;
    }
    out
}

// ---------------------------------------------------------------- BC1 / BC3 encode

fn dec565(v: u16) -> [f32; 3] {
    let (r, g, b) = ((v >> 11) & 31, (v >> 5) & 63, v & 31);
    [((r << 3) | (r >> 2)) as f32, ((g << 2) | (g >> 4)) as f32, ((b << 3) | (b >> 2)) as f32]
}

fn q565(c: [f32; 3]) -> u16 {
    let r = ((c[0] / 255.0 * 31.0 + 0.5).clamp(0.0, 31.0)) as u16;
    let g = ((c[1] / 255.0 * 63.0 + 0.5).clamp(0.0, 63.0)) as u16;
    let b = ((c[2] / 255.0 * 31.0 + 0.5).clamp(0.0, 31.0)) as u16;
    (r << 11) | (g << 5) | b
}

fn d2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (x, y, z) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    x * x + y * y + z * z
}

struct Bc1 {
    c0: u16,
    c1: u16,
    idx: [u8; 16],
    err: f32,
}

/// Indices and error for an endpoint pair; endpoints are put in 4-colour order (c0 > c1).
fn bc1_eval(qa: u16, qb: u16, px: &[[f32; 3]; 16], act: &[bool; 16]) -> Bc1 {
    let (a, b) = if qa >= qb { (qa, qb) } else { (qb, qa) };
    let (pa, pb) = (dec565(a), dec565(b));
    if a == b {
        let err = (0..16).filter(|&i| act[i]).map(|i| d2(px[i], pa)).sum();
        return Bc1 { c0: a, c1: b, idx: [0; 16], err };
    }
    let mix = |t: f32| [pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t, pa[2] + (pb[2] - pa[2]) * t];
    let pal = [pa, pb, mix(1.0 / 3.0), mix(2.0 / 3.0)];
    let mut idx = [0u8; 16];
    let mut err = 0.0;
    for i in 0..16 {
        let (mut bi, mut be) = (0u8, f32::MAX);
        for (k, p) in pal.iter().enumerate() {
            let e = d2(px[i], *p);
            if e < be {
                be = e;
                bi = k as u8;
            }
        }
        idx[i] = bi;
        if act[i] {
            err += be;
        }
    }
    Bc1 { c0: a, c1: b, idx, err }
}

/// Least-squares endpoints for fixed indices.
fn bc1_refine(r: &Bc1, px: &[[f32; 3]; 16], act: &[bool; 16]) -> Option<([f32; 3], [f32; 3])> {
    let (mut a11, mut a12, mut a22) = (0f32, 0f32, 0f32);
    let (mut b1, mut b2) = ([0f32; 3], [0f32; 3]);
    for i in 0..16 {
        if !act[i] {
            continue;
        }
        let t = match r.idx[i] {
            0 => 0.0,
            1 => 1.0,
            2 => 1.0 / 3.0,
            _ => 2.0 / 3.0,
        };
        let s = 1.0 - t;
        a11 += s * s;
        a12 += s * t;
        a22 += t * t;
        for c in 0..3 {
            b1[c] += s * px[i][c];
            b2[c] += t * px[i][c];
        }
    }
    let det = a11 * a22 - a12 * a12;
    if det.abs() < 1e-4 {
        return None;
    }
    let (mut e0, mut e1) = ([0f32; 3], [0f32; 3]);
    for c in 0..3 {
        e0[c] = ((a22 * b1[c] - a12 * b2[c]) / det).clamp(0.0, 255.0);
        e1[c] = ((a11 * b2[c] - a12 * b1[c]) / det).clamp(0.0, 255.0);
    }
    Some((e0, e1))
}

/// Encode one 4x4 block of RGB (alpha is the BC3 side's business). `act` marks texels that count
/// toward the fit (BC3 leaves out fully transparent ones).
fn bc1_block(px: &[[f32; 3]; 16], act: &[bool; 16]) -> [u8; 8] {
    let mut act = *act;
    if !act.iter().any(|&a| a) {
        act = [true; 16];
    }
    let n = act.iter().filter(|&&a| a).count() as f32;
    let mut mean = [0f32; 3];
    for i in 0..16 {
        if act[i] {
            for c in 0..3 {
                mean[c] += px[i][c] / n;
            }
        }
    }
    let mut cov = [[0f32; 3]; 3];
    let (mut lo, mut hi) = ([255f32; 3], [0f32; 3]);
    for i in 0..16 {
        if !act[i] {
            continue;
        }
        let d = [px[i][0] - mean[0], px[i][1] - mean[1], px[i][2] - mean[2]];
        for a in 0..3 {
            for b in 0..3 {
                cov[a][b] += d[a] * d[b];
            }
            lo[a] = lo[a].min(px[i][a]);
            hi[a] = hi[a].max(px[i][a]);
        }
    }
    // Start from the covariance column with the most spread (a start vector orthogonal to the
    // principal axis, such as the bounding-box diagonal for a red/blue block, would never leave it).
    let col = (0..3).max_by(|&a, &b| cov[a][a].total_cmp(&cov[b][b])).unwrap();
    let mut axis = [cov[0][col], cov[1][col], cov[2][col]];
    for _ in 0..8 {
        let v = [
            cov[0][0] * axis[0] + cov[0][1] * axis[1] + cov[0][2] * axis[2],
            cov[1][0] * axis[0] + cov[1][1] * axis[1] + cov[1][2] * axis[2],
            cov[2][0] * axis[0] + cov[2][1] * axis[1] + cov[2][2] * axis[2],
        ];
        let m = v[0].abs().max(v[1].abs()).max(v[2].abs());
        if m < 1e-6 {
            break;
        }
        axis = [v[0] / m, v[1] / m, v[2] / m];
    }
    let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if len > 1e-6 {
        axis = [axis[0] / len, axis[1] / len, axis[2] / len];
    } else {
        axis = [0.0; 3];
    }
    let (mut tmin, mut tmax) = (f32::MAX, f32::MIN);
    for i in 0..16 {
        if act[i] {
            let t = (px[i][0] - mean[0]) * axis[0] + (px[i][1] - mean[1]) * axis[1] + (px[i][2] - mean[2]) * axis[2];
            tmin = tmin.min(t);
            tmax = tmax.max(t);
        }
    }
    let at = |t: f32| [mean[0] + axis[0] * t, mean[1] + axis[1] * t, mean[2] + axis[2] * t];
    let inset = (tmax - tmin) / 16.0;
    let cands = [(at(tmin), at(tmax)), (at(tmin + inset), at(tmax - inset)), (lo, hi)];
    let mut best: Option<Bc1> = None;
    for (e0, e1) in cands {
        let mut r = bc1_eval(q565(e0), q565(e1), px, &act);
        for _ in 0..4 {
            let Some((f0, f1)) = bc1_refine(&r, px, &act) else { break };
            let r2 = bc1_eval(q565(f0), q565(f1), px, &act);
            if r2.err < r.err {
                r = r2;
            } else {
                break;
            }
        }
        if best.as_ref().map_or(true, |b| r.err < b.err) {
            best = Some(r);
        }
    }
    let b = best.unwrap();
    let mut out = [0u8; 8];
    out[0..2].copy_from_slice(&b.c0.to_le_bytes());
    out[2..4].copy_from_slice(&b.c1.to_le_bytes());
    let mut bits = 0u32;
    for i in 0..16 {
        bits |= (b.idx[i] as u32) << (2 * i);
    }
    out[4..8].copy_from_slice(&bits.to_le_bytes());
    out
}

const CUT_PENALTY: f32 = 4000.0;

/// Palette of a BC4-style alpha block as the format defines it (exact fractions).
fn alpha_palette(a0: u8, a1: u8) -> [f32; 8] {
    let (x, y) = (a0 as f32, a1 as f32);
    let mut p = [0f32; 8];
    p[0] = x;
    p[1] = y;
    if a0 > a1 {
        for k in 1..=6 {
            p[1 + k] = ((7 - k) as f32 * x + k as f32 * y) / 7.0;
        }
    } else {
        for k in 1..=4 {
            p[1 + k] = ((5 - k) as f32 * x + k as f32 * y) / 5.0;
        }
        p[6] = 0.0;
        p[7] = 255.0;
    }
    p
}

struct Bc4 {
    a0: u8,
    a1: u8,
    idx: [u8; 16],
    err: f32,
}

fn bc4_eval(a0: u8, a1: u8, a: &[u8; 16], cut: bool) -> Bc4 {
    let pal = alpha_palette(a0, a1);
    let mut idx = [0u8; 16];
    let mut err = 0.0;
    for i in 0..16 {
        let (mut bi, mut be) = (0u8, f32::MAX);
        let src_pass = a[i] as f32 >= 127.5;
        for (k, &p) in pal.iter().enumerate() {
            let d = a[i] as f32 - p;
            let mut e = d * d;
            if cut && (p >= 127.5) != src_pass {
                e += CUT_PENALTY;
            }
            if e < be {
                be = e;
                bi = k as u8;
            }
        }
        idx[i] = bi;
        err += be;
    }
    Bc4 { a0, a1, idx, err }
}

fn bc4_refine(r: &Bc4, a: &[u8; 16]) -> Option<(u8, u8)> {
    let eight = r.a0 > r.a1;
    let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0f32, 0f32, 0f32, 0f32, 0f32);
    for i in 0..16 {
        let k = r.idx[i] as usize;
        let t = if eight {
            match k {
                0 => 0.0,
                1 => 1.0,
                _ => (k - 1) as f32 / 7.0,
            }
        } else {
            match k {
                0 => 0.0,
                1 => 1.0,
                2..=5 => (k - 1) as f32 / 5.0,
                _ => continue,
            }
        };
        let s = 1.0 - t;
        a11 += s * s;
        a12 += s * t;
        a22 += t * t;
        b1 += s * a[i] as f32;
        b2 += t * a[i] as f32;
    }
    let det = a11 * a22 - a12 * a12;
    if det.abs() < 1e-4 {
        return None;
    }
    let e0 = ((a22 * b1 - a12 * b2) / det).round().clamp(0.0, 255.0) as u8;
    let e1 = ((a11 * b2 - a12 * b1) / det).round().clamp(0.0, 255.0) as u8;
    if eight && e0 <= e1 || !eight && e0 > e1 {
        return None;
    }
    Some((e0, e1))
}

fn bc4_block(a: &[u8; 16], cut: bool) -> [u8; 8] {
    let (lo, hi) = (*a.iter().min().unwrap(), *a.iter().max().unwrap());
    let mut cands: Vec<(u8, u8)> = Vec::new();
    if hi > lo {
        cands.push((hi, lo)); // 8-value mode
    }
    let inner: Vec<u8> = a.iter().copied().filter(|&v| v != 0 && v != 255).collect();
    if inner.is_empty() {
        cands.push((0, 0)); // 6-value mode, only the exact 0 and 255 entries are used
    } else {
        let (il, ih) = (*inner.iter().min().unwrap(), *inner.iter().max().unwrap());
        cands.push((il, ih)); // 6-value mode with 0 and 255 kept apart
    }
    let mut best: Option<Bc4> = None;
    for (e0, e1) in cands {
        let mut r = bc4_eval(e0, e1, a, cut);
        for _ in 0..3 {
            let Some((f0, f1)) = bc4_refine(&r, a) else { break };
            let r2 = bc4_eval(f0, f1, a, cut);
            if r2.err < r.err {
                r = r2;
            } else {
                break;
            }
        }
        if best.as_ref().map_or(true, |b| r.err < b.err) {
            best = Some(r);
        }
    }
    let b = best.unwrap();
    let mut out = [0u8; 8];
    out[0] = b.a0;
    out[1] = b.a1;
    let mut bits = 0u64;
    for i in 0..16 {
        bits |= (b.idx[i] as u64) << (3 * i);
    }
    out[2..8].copy_from_slice(&bits.to_le_bytes()[0..6]);
    out
}

/// Encode one level. Rows are read bottom-up (Unity's origin). `cut` adds the cutout-edge penalty
/// to the alpha fit (DXT5 only).
fn encode_level(l: &Level, fmt: Format, cut: bool, threads: usize) -> Vec<u8> {
    if fmt == Format::Rgba32 {
        let mut out = Vec::with_capacity(l.w * l.h * 4);
        for y in (0..l.h).rev() {
            out.extend_from_slice(&l.rgba[y * l.w * 4..(y + 1) * l.w * 4]);
        }
        return out;
    }
    let (bw, bh) = (l.w.div_ceil(4), l.h.div_ceil(4));
    let bsize = if fmt == Format::Dxt1 { 8 } else { 16 };
    let mut out = vec![0u8; bw * bh * bsize];
    let work = |row0: usize, chunk: &mut [u8]| {
        for (r, rowbuf) in chunk.chunks_mut(bw * bsize).enumerate() {
            let by = row0 + r;
            for bx in 0..bw {
                let mut px = [[0f32; 3]; 16];
                let mut al = [255u8; 16];
                let mut act = [true; 16];
                for j in 0..4 {
                    let fy = (by * 4 + j).min(l.h - 1);
                    let sy = l.h - 1 - fy;
                    for i in 0..4 {
                        let sx = (bx * 4 + i).min(l.w - 1);
                        let o = (sy * l.w + sx) * 4;
                        let t = j * 4 + i;
                        px[t] = [l.rgba[o] as f32, l.rgba[o + 1] as f32, l.rgba[o + 2] as f32];
                        al[t] = l.rgba[o + 3];
                        if fmt == Format::Dxt5 {
                            act[t] = al[t] > 0;
                        }
                    }
                }
                let dst = &mut rowbuf[bx * bsize..(bx + 1) * bsize];
                if fmt == Format::Dxt5 {
                    dst[0..8].copy_from_slice(&bc4_block(&al, cut));
                    dst[8..16].copy_from_slice(&bc1_block(&px, &act));
                } else {
                    dst.copy_from_slice(&bc1_block(&px, &act));
                }
            }
        }
    };
    let threads = threads.clamp(1, 32).min(bh);
    if threads <= 1 {
        work(0, &mut out);
    } else {
        let rows_per = bh.div_ceil(threads);
        std::thread::scope(|s| {
            for (ci, chunk) in out.chunks_mut(rows_per * bw * bsize).enumerate() {
                let work = &work;
                s.spawn(move || work(ci * rows_per, chunk));
            }
        });
    }
    out
}

// ---------------------------------------------------------------- shipping

#[derive(Clone, Debug)]
pub struct ShipOpts {
    /// None picks DXT1 when every texel is opaque, else DXT5.
    pub format: Option<Format>,
    pub mips: MipOpts,
    pub threads: usize,
}

impl Default for ShipOpts {
    fn default() -> Self {
        ShipOpts { format: None, mips: MipOpts::default(), threads: 4 }
    }
}

pub struct Shipped {
    pub bytes: Vec<u8>,
    pub format: Format,
    pub levels: usize,
    pub has_alpha: bool,
    /// Share of texels passing the cutout test, per level, of the chain that was encoded.
    pub coverage: Vec<f64>,
    pub notes: Vec<String>,
}

pub fn has_alpha(rgba: &[u8]) -> bool {
    rgba.chunks_exact(4).any(|p| p[3] < 255)
}

/// Whole file: header plus the encoded chain.
pub fn ship(rgba: Vec<u8>, w: usize, h: usize, o: &ShipOpts) -> Result<Shipped, String> {
    if w == 0 || h == 0 || rgba.len() != w * h * 4 {
        return Err(format!("{w}x{h} does not match {} bytes of RGBA", rgba.len()));
    }
    let alpha = has_alpha(&rgba);
    let mut notes = Vec::new();
    let fmt = match o.format {
        Some(f) => f,
        None if alpha => Format::Dxt5,
        None => Format::Dxt1,
    };
    if fmt == Format::Dxt1 && alpha {
        notes.push("DXT1 chosen but the sheet has alpha: alpha is dropped".to_owned());
    }
    let chain = build_chain(Level { w, h, rgba }, &o.mips);
    let cov: Vec<f64> = chain.iter().map(|l| coverage(&l.rgba, o.mips.cutoff)).collect();
    let cut = fmt == Format::Dxt5 && o.mips.cutoff == 128;
    let total = data_len(fmt, w, h, chain.len());
    let hdr = Header { width: w as u32, height: h as u32, data_len: total as u32, format: fmt.unity_value(), mips: chain.len() as u32 };
    let mut bytes = Vec::with_capacity(HEADER_LEN + total);
    bytes.extend_from_slice(&hdr.to_bytes());
    for l in &chain {
        bytes.extend_from_slice(&encode_level(l, fmt, cut, o.threads));
    }
    debug_assert_eq!(bytes.len(), HEADER_LEN + total);
    Ok(Shipped { bytes, format: fmt, levels: chain.len(), has_alpha: alpha, coverage: cov, notes })
}

// ---------------------------------------------------------------- decode (checks)

fn bc1_colors(b: &[u8]) -> [[u8; 3]; 4] {
    let c0 = u16::from_le_bytes([b[0], b[1]]);
    let c1 = u16::from_le_bytes([b[2], b[3]]);
    let (p0, p1) = (dec565(c0), dec565(c1));
    let to = |v: [f32; 3]| [(v[0] + 0.5) as u8, (v[1] + 0.5) as u8, (v[2] + 0.5) as u8];
    let m = |t: f32| to([p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t, p0[2] + (p1[2] - p0[2]) * t]);
    if c0 > c1 {
        [to(p0), to(p1), m(1.0 / 3.0), m(2.0 / 3.0)]
    } else {
        [to(p0), to(p1), m(0.5), [0, 0, 0]]
    }
}

/// Decodes a DXT level back to RGBA in Unity's bottom-up row order.
pub fn decode_dxt(data: &[u8], w: usize, h: usize, fmt: Format) -> Level {
    let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
    let bsize = if fmt == Format::Dxt1 { 8 } else { 16 };
    let mut rgba = vec![255u8; w * h * 4];
    for by in 0..bh {
        for bx in 0..bw {
            let blk = &data[(by * bw + bx) * bsize..(by * bw + bx + 1) * bsize];
            let (ablk, cblk) = if fmt == Format::Dxt5 { (Some(&blk[0..8]), &blk[8..16]) } else { (None, blk) };
            let pal = bc1_colors(cblk);
            let bits = u32::from_le_bytes(cblk[4..8].try_into().unwrap());
            let apal = ablk.map(|a| {
                let p = alpha_palette(a[0], a[1]);
                let mut bits = [0u8; 8];
                bits[..6].copy_from_slice(&a[2..8]);
                (p, u64::from_le_bytes(bits))
            });
            for j in 0..4 {
                for i in 0..4 {
                    let (x, y) = (bx * 4 + i, by * 4 + j);
                    if x >= w || y >= h {
                        continue;
                    }
                    let t = j * 4 + i;
                    let c = pal[((bits >> (2 * t)) & 3) as usize];
                    let o = (y * w + x) * 4;
                    rgba[o..o + 3].copy_from_slice(&c);
                    if let Some((p, ab)) = &apal {
                        rgba[o + 3] = (p[((ab >> (3 * t)) & 7) as usize] + 0.5) as u8;
                    }
                }
            }
        }
    }
    Level { w, h, rgba }
}

/// Reads a raw texture (RGBA32, DXT1 or DXT5) back into levels in PNG row order (top first).
pub fn decode_raw(bytes: &[u8]) -> Result<(Header, Vec<Level>), String> {
    let h = parse_header(bytes)?;
    let fmt = h.kind().ok_or_else(|| format!("format {} cannot be decoded here", h.format))?;
    let (w0, h0) = (h.width as usize, h.height as usize);
    let dims: Vec<_> = mip_dims(w0, h0).into_iter().take(h.mips as usize).collect();
    if data_len(fmt, w0, h0, dims.len()) != h.data_len as usize {
        return Err("mip count does not add up to dataLength".to_owned());
    }
    let mut pos = HEADER_LEN;
    let mut out = Vec::new();
    for (w, hh) in dims {
        let n = fmt.level_bytes(w, hh);
        let d = &bytes[pos..pos + n];
        pos += n;
        let mut l = match fmt {
            Format::Rgba32 => Level { w, h: hh, rgba: d.to_vec() },
            _ => decode_dxt(d, w, hh, fmt),
        };
        // bottom-up storage -> top-down
        let row = w * 4;
        for y in 0..hh / 2 {
            let (a, b) = l.rgba.split_at_mut((hh - 1 - y) * row);
            a[y * row..(y + 1) * row].swap_with_slice(&mut b[..row]);
        }
        out.push(l);
    }
    Ok((h, out))
}

/// PSNR over RGB for texels whose source alpha is above zero (hidden colors do not count).
pub fn psnr_rgb(src: &[u8], dec: &[u8]) -> f64 {
    let (mut se, mut n) = (0f64, 0f64);
    for (s, d) in src.chunks_exact(4).zip(dec.chunks_exact(4)) {
        if s[3] == 0 {
            continue;
        }
        for c in 0..3 {
            let e = s[c] as f64 - d[c] as f64;
            se += e * e;
        }
        n += 3.0;
    }
    if n == 0.0 || se == 0.0 {
        return 99.0;
    }
    10.0 * (255.0 * 255.0 / (se / n)).log10()
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 16) as u32
        }
    }

    fn solid(w: usize, h: usize, c: [u8; 4]) -> Vec<u8> {
        c.iter().copied().cycle().take(w * h * 4).collect()
    }

    /// Smooth color field with a little noise (a stand-in for paint).
    fn paint(w: usize, h: usize) -> Vec<u8> {
        let mut r = Rng(7);
        let mut v = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let n = (r.next() % 9) as i32 - 4;
                let c = |base: f32| ((base + n as f32).clamp(0.0, 255.0)) as u8;
                v.extend_from_slice(&[c(x as f32 * 255.0 / w as f32), c(y as f32 * 255.0 / h as f32), c(128.0 + 90.0 * ((x + y) as f32 / 9.0).sin()), 255]);
            }
        }
        v
    }

    #[test]
    fn sizes_match_the_research_table_at_4096() {
        assert_eq!(mip_count(4096, 4096), 13);
        assert_eq!(data_len(Format::Rgba32, 4096, 4096, 13), 89_478_484);
        assert_eq!(data_len(Format::Dxt5, 4096, 4096, 13), 22_369_648);
        assert_eq!(data_len(Format::Dxt1, 4096, 4096, 13), 11_184_824);
        // without mips
        assert_eq!(data_len(Format::Dxt5, 4096, 4096, 1), 16 * 1024 * 1024);
        assert_eq!(data_len(Format::Dxt1, 4096, 4096, 1), 8 * 1024 * 1024);
        assert_eq!(data_len(Format::Rgba32, 4096, 4096, 1), 64 * 1024 * 1024);
    }

    #[test]
    fn header_bytes_match_the_doc() {
        let hex = |h: Header| h.to_bytes()[..20].iter().map(|b| format!("{b:02X}")).collect::<String>();
        let mk = |f: Format| Header { width: 4096, height: 4096, data_len: data_len(f, 4096, 4096, 13) as u32, format: f.unity_value(), mips: 13 };
        let doc = |s: &str| s.replace(' ', "");
        assert_eq!(hex(mk(Format::Rgba32)), doc("00100000 00100000 54555505 04000000 0D000000"));
        assert_eq!(hex(mk(Format::Dxt5)), doc("00100000 00100000 70555501 0C000000 0D000000"));
        assert_eq!(hex(mk(Format::Dxt1)), doc("00100000 00100000 B8AAAA00 0A000000 0D000000"));
        assert!(mk(Format::Dxt5).to_bytes()[20..].iter().all(|&b| b == 0));
    }

    #[test]
    fn header_round_trip_and_loader_test() {
        let hd = Header { width: 8, height: 4, data_len: data_len(Format::Dxt5, 8, 4, 4) as u32, format: 12, mips: 4 };
        let mut file = hd.to_bytes().to_vec();
        file.extend(vec![0u8; hd.data_len as usize]);
        assert_eq!(parse_header(&file).unwrap(), hd);
        // a wrong length or a format the loader does not list is refused
        file.push(0);
        assert!(parse_header(&file).is_err());
        let mut bad = Header { format: 13, ..hd }.to_bytes().to_vec();
        bad.extend(vec![0u8; hd.data_len as usize]);
        assert!(parse_header(&bad).is_err());
        assert!(parse_header(&[0u8; 10]).is_err());
    }

    #[test]
    fn built_files_have_the_exact_length() {
        for (w, h) in [(64usize, 64usize), (32, 8), (5, 3), (1, 1), (4, 4)] {
            for f in [Format::Rgba32, Format::Dxt1, Format::Dxt5] {
                let s = ship(paint(w, h), w, h, &ShipOpts { format: Some(f), ..Default::default() }).unwrap();
                let hd = parse_header(&s.bytes).unwrap();
                assert_eq!(hd.mips as usize, mip_count(w, h), "{w}x{h} {f:?}");
                assert_eq!(hd.data_len as usize, data_len(f, w, h, hd.mips as usize));
                assert_eq!(hd.format, f.unity_value());
                let (_, levels) = decode_raw(&s.bytes).unwrap();
                assert_eq!(levels.len(), mip_count(w, h));
                assert_eq!((levels.last().unwrap().w, levels.last().unwrap().h), (1, 1));
            }
        }
    }

    #[test]
    fn rows_are_stored_bottom_up() {
        // 4x4: top row red, the other rows blue (two colors, so DXT1 can hold it)
        let mut px = solid(4, 4, [0, 0, 255, 255]);
        for x in 0..4 {
            px[x * 4..x * 4 + 4].copy_from_slice(&[255, 0, 0, 255]);
        }
        let s = ship(px.clone(), 4, 4, &ShipOpts { format: Some(Format::Rgba32), ..Default::default() }).unwrap();
        // first data row is the picture's bottom row (blue)
        assert_eq!(&s.bytes[HEADER_LEN..HEADER_LEN + 4], &[0, 0, 255, 255]);
        assert_eq!(&s.bytes[HEADER_LEN + 12 * 4..HEADER_LEN + 12 * 4 + 4], &[255, 0, 0, 255]);
        // and decoding undoes the flip, DXT included
        for f in [Format::Dxt1, Format::Dxt5] {
            let s = ship(px.clone(), 4, 4, &ShipOpts { format: Some(f), ..Default::default() }).unwrap();
            let (_, lv) = decode_raw(&s.bytes).unwrap();
            for (a, b) in lv[0].rgba.chunks_exact(4).zip(px.chunks_exact(4)) {
                for c in 0..3 {
                    assert!((a[c] as i32 - b[c] as i32).abs() <= 8, "{f:?} {a:?} {b:?}");
                }
            }
        }
    }

    #[test]
    fn dxt_round_trip_quality() {
        let (w, h) = (128, 128);
        let px = paint(w, h);
        for (f, min_db) in [(Format::Dxt1, 33.0), (Format::Dxt5, 33.0)] {
            let s = ship(px.clone(), w, h, &ShipOpts { format: Some(f), ..Default::default() }).unwrap();
            let (_, lv) = decode_raw(&s.bytes).unwrap();
            let db = psnr_rgb(&px, &lv[0].rgba);
            assert!(db > min_db, "{f:?} PSNR {db}");
        }
    }

    #[test]
    fn two_color_blocks_come_out_close() {
        // hard edges between two colors inside blocks must stay near-exact
        let (w, h) = (64, 64);
        let mut px = solid(w, h, [200, 40, 40, 255]);
        for y in 0..h {
            for x in 0..w {
                if (x + y / 2) % 7 < 3 {
                    px[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&[30, 160, 220, 255]);
                }
            }
        }
        let s = ship(px.clone(), w, h, &ShipOpts { format: Some(Format::Dxt1), ..Default::default() }).unwrap();
        let (_, lv) = decode_raw(&s.bytes).unwrap();
        assert!(psnr_rgb(&px, &lv[0].rgba) > 40.0);
    }

    /// Hair-like cutout: thin opaque strands inside a few blobs on a transparent field, thin enough
    /// that plain mips thin out.
    fn strands(w: usize, h: usize) -> Vec<u8> {
        let mut r = Rng(99);
        let blobs: Vec<(i64, i64, i64)> = (0..14).map(|_| ((r.next() as usize % w) as i64, (r.next() as usize % h) as i64, (w as i64 / 10) + (r.next() % (w as u32 / 8)) as i64)).collect();
        let mut v = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let inside = blobs.iter().any(|&(bx, by, br)| (x as i64 - bx).pow(2) + (y as i64 - by).pow(2) < br * br);
                let on = inside && (x * 3 + y) % 44 < 8;
                let o = (y * w + x) * 4;
                if on {
                    v[o..o + 4].copy_from_slice(&[220, 180, 60, 255]);
                } else {
                    v[o..o + 4].copy_from_slice(&[10, 200, 10, 0]); // loud hidden color
                }
            }
        }
        v
    }

    #[test]
    fn coverage_is_kept_down_the_chain() {
        let (w, h) = (256, 256);
        let src = strands(w, h);
        let target = coverage(&src, 128);
        let plain = build_chain(Level { w, h, rgba: src.clone() }, &MipOpts { coverage: false, ..Default::default() });
        let kept = build_chain(Level { w, h, rgba: src.clone() }, &MipOpts::default());
        let (mut worst_plain, mut worst_kept) = (0f64, 0f64);
        for (p, k) in plain.iter().zip(&kept).skip(1).filter(|(p, _)| p.w >= 16) {
            let (dp, dk) = ((coverage(&p.rgba, 128) - target).abs(), (coverage(&k.rgba, 128) - target).abs());
            // never worse than the plain box filter at any level
            assert!(dk <= dp + 1e-9, "level {}: kept {dk} plain {dp}", p.w);
            worst_plain = worst_plain.max(dp);
            worst_kept = worst_kept.max(dk);
        }
        assert!(worst_plain > 0.08, "plain mips should thin out ({worst_plain})");
        assert!(worst_kept < 0.04, "coverage drifted by {worst_kept}");
    }

    #[test]
    fn hidden_colors_do_not_bleed_into_mips() {
        let (w, h) = (16, 16);
        let src = strands(w, h);
        let chain = build_chain(Level { w, h, rgba: src }, &MipOpts { coverage: false, ..Default::default() });
        // wherever a level has alpha, its color is the strand color, not the green of the hidden texels
        for l in chain.iter().skip(1) {
            for p in l.rgba.chunks_exact(4).filter(|p| p[3] > 0) {
                assert!(p[1] < 200 && p[0] > 150, "bled color {p:?}");
            }
        }
    }

    #[test]
    fn dxt5_keeps_the_cutout_edge() {
        let (w, h) = (128, 128);
        let src = strands(w, h);
        let s = ship(src.clone(), w, h, &ShipOpts { format: Some(Format::Dxt5), ..Default::default() }).unwrap();
        assert_eq!(s.format, Format::Dxt5);
        let (_, lv) = decode_raw(&s.bytes).unwrap();
        let wrong = src.chunks_exact(4).zip(lv[0].rgba.chunks_exact(4)).filter(|(a, b)| (a[3] >= 128) != (b[3] >= 128)).count();
        assert_eq!(wrong, 0, "texels on the wrong side of the cutout");
        // coverage of every decoded level stays near the source
        let target = coverage(&src, 128);
        for l in lv.iter().filter(|l| l.w >= 16) {
            assert!((coverage(&l.rgba, 128) - target).abs() < 0.05, "level {}", l.w);
        }
    }

    #[test]
    fn auto_format_follows_alpha() {
        let o = ShipOpts::default();
        assert_eq!(ship(paint(16, 16), 16, 16, &o).unwrap().format, Format::Dxt1);
        assert_eq!(ship(strands(16, 16), 16, 16, &o).unwrap().format, Format::Dxt5);
    }

    #[test]
    fn linear_mips_differ_from_gamma_mips() {
        let mut px = solid(2, 2, [0, 0, 0, 255]);
        px[0..4].copy_from_slice(&[255, 255, 255, 255]);
        let g = build_chain(Level { w: 2, h: 2, rgba: px.clone() }, &MipOpts::default());
        let l = build_chain(Level { w: 2, h: 2, rgba: px }, &MipOpts { linear: true, ..Default::default() });
        assert_eq!(g[1].rgba[0], 64);
        assert!(l[1].rgba[0] > 120, "{}", l[1].rgba[0]);
    }

    #[test]
    fn full_4096_chain_has_the_documented_size() {
        // a real 4096x4096 build (flat texture so it stays quick): sizes, header and last level
        let s = ship(solid(4096, 4096, [90, 120, 150, 255]), 4096, 4096, &ShipOpts { threads: 4, ..Default::default() }).unwrap();
        assert_eq!(s.format, Format::Dxt1);
        assert_eq!(s.bytes.len(), 60 + 11_184_824);
        let s5 = ship(solid(4096, 4096, [90, 120, 150, 254]), 4096, 4096, &ShipOpts { threads: 4, ..Default::default() }).unwrap();
        assert_eq!(s5.format, Format::Dxt5);
        assert_eq!(s5.bytes.len(), 60 + 22_369_648);
        assert_eq!(&s5.bytes[..20], &[0x00, 0x10, 0, 0, 0x00, 0x10, 0, 0, 0x70, 0x55, 0x55, 0x01, 0x0C, 0, 0, 0, 0x0D, 0, 0, 0]);
    }
}
