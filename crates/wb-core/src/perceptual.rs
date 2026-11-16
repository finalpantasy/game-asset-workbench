//! Perceptual image checks: NVIDIA's FLIP (LDR) and SSIM, in plain Rust.
//!
//! FLIP is ported from NVIDIA's reference implementation (github.com/NVlabs/flip, BSD-3-Clause,
//! Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES; Andersson et al., "FLIP: A Difference
//! Evaluator for Alternating Images", HPG 2020). The constants, the opponent-space transform, the
//! contrast-sensitivity filters, the HyAB color difference with error redistribution, and the
//! edge/point feature difference follow `src/pytorch/flip_loss.py` (`compute_ldrflip`). Differences:
//! the filters are applied as separable passes (the same kernels, evaluated faster) and the image
//! borders replicate their edge pixels, as the reference does.
//!
//! SSIM is the standard one: Gaussian window (11 taps, sigma 1.5) on luma, K1 = 0.01, K2 = 0.03.
//!
//! Everything works on `Rgb` (sRGB values 0..1) and returns per-pixel error maps, so callers can
//! pool them over any mask (the model's pixels, a body part) and draw heat maps.
use std::f32::consts::PI;

/// Pixels per degree of visual angle used by the FLIP reference tool: a 0.7 m wide, 3840 px wide
/// monitor seen from 0.7 m. Larger values make small, fine differences matter less.
pub const DEFAULT_PPD: f32 = 3840.0 * PI / 180.0;

#[derive(Clone)]
pub struct Rgb {
    pub w: usize,
    pub h: usize,
    /// sRGB, 0..1 per channel.
    pub px: Vec<[f32; 3]>,
}

impl Rgb {
    /// From RGBA8 bytes; transparent pixels are blended over `bg`.
    pub fn from_rgba8(w: usize, h: usize, rgba: &[u8], bg: [u8; 3]) -> Rgb {
        let px = (0..w * h)
            .map(|i| {
                let a = rgba[i * 4 + 3] as f32 / 255.0;
                std::array::from_fn(|c| (rgba[i * 4 + c] as f32 * a + bg[c] as f32 * (1.0 - a)) / 255.0)
            })
            .collect();
        Rgb { w, h, px }
    }

    pub fn to_rgba8(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.w * self.h * 4);
        for p in &self.px {
            out.extend(p.iter().map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8));
            out.push(255);
        }
        out
    }

    /// Box-filter downsample by an integer factor (what a mip level does).
    pub fn downsample(&self, f: usize) -> Rgb {
        let (nw, nh) = (self.w / f, self.h / f);
        let mut px = Vec::with_capacity(nw * nh);
        for y in 0..nh {
            for x in 0..nw {
                let mut s = [0.0f32; 3];
                for dy in 0..f {
                    for dx in 0..f {
                        let p = self.px[(y * f + dy) * self.w + x * f + dx];
                        for c in 0..3 {
                            s[c] += p[c];
                        }
                    }
                }
                px.push(s.map(|v| v / (f * f) as f32));
            }
        }
        Rgb { w: nw, h: nh, px }
    }
}

// ---------- separable convolution with replicated borders ----------

fn conv_rows(src: &[f32], w: usize, h: usize, k: &[f32]) -> Vec<f32> {
    let r = k.len() / 2;
    let mut out = vec![0.0f32; w * h];
    let mut pad = vec![0.0f32; w + 2 * r];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        pad[..r].fill(row[0]);
        pad[r..r + w].copy_from_slice(row);
        pad[r + w..].fill(row[w - 1]);
        let o = &mut out[y * w..(y + 1) * w];
        for (t, &kt) in k.iter().enumerate() {
            if kt == 0.0 {
                continue;
            }
            for (x, ov) in o.iter_mut().enumerate() {
                *ov += kt * pad[x + t];
            }
        }
    }
    out
}

fn conv_cols(src: &[f32], w: usize, h: usize, k: &[f32]) -> Vec<f32> {
    let r = (k.len() / 2) as i64;
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        let o = &mut out[y * w..(y + 1) * w];
        for (t, &kt) in k.iter().enumerate() {
            if kt == 0.0 {
                continue;
            }
            let sy = (y as i64 + t as i64 - r).clamp(0, h as i64 - 1) as usize;
            let s = &src[sy * w..(sy + 1) * w];
            for (ov, sv) in o.iter_mut().zip(s) {
                *ov += kt * sv;
            }
        }
    }
    out
}

fn conv_sep(src: &[f32], w: usize, h: usize, kx: &[f32], ky: &[f32]) -> Vec<f32> {
    conv_cols(&conv_rows(src, w, h, kx), w, h, ky)
}

// ---------- color spaces (D65), as in the reference ----------

const WHITE: [f32; 3] = [0.950_428_5, 1.0, 1.088_900_4];

fn srgb_to_lin(c: f32) -> f32 {
    if c > 0.04045 {
        ((c + 0.055) / 1.055).powf(2.4)
    } else {
        c / 12.92
    }
}

fn lin_to_xyz(c: [f32; 3]) -> [f32; 3] {
    [
        0.412_390_8 * c[0] + 0.357_584_3 * c[1] + 0.180_480_8 * c[2],
        0.212_639 * c[0] + 0.715_168_7 * c[1] + 0.072_192_3 * c[2],
        0.019_330_8 * c[0] + 0.119_194_8 * c[1] + 0.950_532_2 * c[2],
    ]
}

fn xyz_to_lin(c: [f32; 3]) -> [f32; 3] {
    [
        3.241_003_3 * c[0] - 1.537_398_9 * c[1] - 0.498_615_9 * c[2],
        -0.969_224_3 * c[0] + 1.875_930_1 * c[1] + 0.041_554_2 * c[2],
        0.055_639_4 * c[0] - 0.204_011_2 * c[1] + 1.057_148_9 * c[2],
    ]
}

fn xyz_to_ycxcz(c: [f32; 3]) -> [f32; 3] {
    let (x, y, z) = (c[0] / WHITE[0], c[1] / WHITE[1], c[2] / WHITE[2]);
    [116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)]
}

fn ycxcz_to_xyz(c: [f32; 3]) -> [f32; 3] {
    let y = (c[0] + 16.0) / 116.0;
    [(y + c[1] / 500.0) * WHITE[0], y * WHITE[1], (y - c[2] / 200.0) * WHITE[2]]
}

fn xyz_to_lab(c: [f32; 3]) -> [f32; 3] {
    let delta = 6.0f32 / 29.0;
    let cube = delta * delta * delta;
    let f = |t: f32| if t > cube { t.cbrt() } else { t / (3.0 * delta * delta) + 4.0 / 29.0 };
    let (fx, fy, fz) = (f(c[0] / WHITE[0]), f(c[1] / WHITE[1]), f(c[2] / WHITE[2]));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

fn hunt(lab: [f32; 3]) -> [f32; 3] {
    [lab[0], 0.01 * lab[0] * lab[1], 0.01 * lab[0] * lab[2]]
}

fn hyab(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).abs() + ((a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

// ---------- FLIP ----------

const QC: f32 = 0.7;
const QF: f32 = 0.5;
const PC: f32 = 0.4;
const PT: f32 = 0.95;

/// 1D Gaussian terms of a channel's contrast sensitivity filter: (kernel, weight).
fn csf_terms(ppd: f32, channel: usize) -> (usize, Vec<(Vec<f32>, f32)>) {
    // (a1, b1, a2, b2) for A, RG, BY.
    let p: [(f32, f32, f32, f32); 3] = [(1.0, 0.0047, 0.0, 1e-5), (1.0, 0.0053, 0.0, 1e-5), (34.1, 0.04, 13.5, 0.025)];
    let max_b = 0.04f32;
    let r = (3.0 * (max_b / (2.0 * PI * PI)).sqrt() * ppd).ceil() as i64;
    let (a1, b1, a2, b2) = p[channel];
    let dx = 1.0 / ppd;
    let kern = |b: f32| -> Vec<f32> { (-r..=r).map(|x| (-(PI * PI) * (x as f32 * dx).powi(2) / b).exp()).collect() };
    let mut terms = Vec::new();
    let mut total = 0.0f32;
    for (a, b) in [(a1, b1), (a2, b2)] {
        if a == 0.0 {
            continue;
        }
        let k = kern(b);
        let s: f32 = k.iter().sum();
        let c = a * (PI / b).sqrt();
        total += c * s * s;
        terms.push((k, c));
    }
    for t in &mut terms {
        t.1 /= total;
    }
    (r as usize, terms)
}

/// 1D pieces of the edge or point detector: (derivative kernel, gaussian kernel), both with the
/// reference's normalization (positive weights sum to 1, negative to -1 over the 2D filter).
fn feature_kernels(ppd: f32, edge: bool) -> (Vec<f32>, Vec<f32>) {
    let sd = 0.5 * 0.082 * ppd;
    let r = (3.0 * sd).ceil() as i64;
    let g: Vec<f32> = (-r..=r).map(|x| (-(x * x) as f32 / (2.0 * sd * sd)).exp()).collect();
    let mut u: Vec<f32> = (-r..=r)
        .zip(&g)
        .map(|(x, &gx)| if edge { -(x as f32) * gx } else { ((x * x) as f32 / (sd * sd) - 1.0) * gx })
        .collect();
    let pos: f32 = u.iter().filter(|&&v| v > 0.0).sum();
    let neg: f32 = -u.iter().filter(|&&v| v < 0.0).sum::<f32>();
    for v in &mut u {
        *v = if *v < 0.0 { *v / neg } else { *v / pos };
    }
    let gs: f32 = g.iter().sum();
    (u, g.iter().map(|v| v / gs).collect())
}

/// Maximum HyAB distance (after the power) between two RGB colors, green to blue, as in the reference.
fn cmax() -> f32 {
    let lab = |c: [f32; 3]| hunt(xyz_to_lab(lin_to_xyz(c)));
    hyab(lab([0.0, 1.0, 0.0]), lab([0.0, 0.0, 1.0])).powf(QC)
}

pub struct FlipMap {
    pub w: usize,
    pub h: usize,
    /// FLIP error per pixel, 0 (same) to 1.
    pub err: Vec<f32>,
}

/// LDR-FLIP error map of `test` against `reference`. Both must be the same size.
pub fn flip(reference: &Rgb, test: &Rgb, ppd: f32) -> Result<FlipMap, String> {
    if (reference.w, reference.h) != (test.w, test.h) {
        return Err(format!("image sizes differ: {}x{} against {}x{}", reference.w, reference.h, test.w, test.h));
    }
    let (w, h) = (reference.w, reference.h);
    let n = w * h;
    let planes = |im: &Rgb| -> [Vec<f32>; 3] {
        let mut p = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
        for (i, c) in im.px.iter().enumerate() {
            let y = xyz_to_ycxcz(lin_to_xyz(c.map(srgb_to_lin)));
            for k in 0..3 {
                p[k][i] = y[k];
            }
        }
        p
    };
    let (pr, pt) = (planes(reference), planes(test));

    // Color pipeline: contrast-sensitivity filtering in opponent space, then Hunt-adjusted L*a*b*.
    let filt = |p: &[Vec<f32>; 3]| -> Vec<[f32; 3]> {
        let f: Vec<Vec<f32>> = (0..3)
            .map(|ch| {
                let (_, terms) = csf_terms(ppd, ch);
                let mut acc = vec![0.0f32; n];
                for (k, wt) in &terms {
                    let c = conv_sep(&p[ch], w, h, k, k);
                    for (a, v) in acc.iter_mut().zip(&c) {
                        *a += wt * v;
                    }
                }
                acc
            })
            .collect();
        (0..n)
            .map(|i| {
                let lin = xyz_to_lin(ycxcz_to_xyz([f[0][i], f[1][i], f[2][i]])).map(|v| v.clamp(0.0, 1.0));
                hunt(xyz_to_lab(lin_to_xyz(lin)))
            })
            .collect()
    };
    let (lr, lt) = (filt(&pr), filt(&pt));
    let cm = cmax();
    let pccmax = PC * cm;
    let dc: Vec<f32> = (0..n)
        .map(|i| {
            let d = hyab(lr[i], lt[i]).powf(QC);
            if d < pccmax {
                PT / pccmax * d
            } else {
                PT + (d - pccmax) / (cm - pccmax) * (1.0 - PT)
            }
        })
        .collect();

    // Feature pipeline: edges and points in the (unfiltered) linear luminance.
    let ylum = |p: &[Vec<f32>; 3]| -> Vec<f32> { p[0].iter().map(|v| (v + 16.0) / 116.0).collect() };
    let (yr, yt) = (ylum(&pr), ylum(&pt));
    let (eu, eg) = feature_kernels(ppd, true);
    let (qu, qg) = feature_kernels(ppd, false);
    let norm2 = |a: Vec<f32>, b: Vec<f32>| -> Vec<f32> { a.iter().zip(&b).map(|(x, y)| (x * x + y * y).sqrt()).collect() };
    let feats = |y: &[f32]| -> (Vec<f32>, Vec<f32>) {
        let e = norm2(conv_sep(y, w, h, &eu, &eg), conv_sep(y, w, h, &eg, &eu));
        let p = norm2(conv_sep(y, w, h, &qu, &qg), conv_sep(y, w, h, &qg, &qu));
        (e, p)
    };
    let ((er, por), (et, pot)) = (feats(&yr), feats(&yt));
    let err = (0..n)
        .map(|i| {
            let df = (er[i] - et[i]).abs().max((pot[i] - por[i]).abs()).max(1e-15);
            let df = ((1.0 / 2f32.sqrt()) * df).powf(QF);
            if dc[i] <= 0.0 {
                0.0
            } else {
                dc[i].powf(1.0 - df.min(0.999_999))
            }
        })
        .collect();
    Ok(FlipMap { w, h, err })
}

// ---------- SSIM ----------

/// Per-pixel SSIM on luma (sRGB-coded, 0..1), Gaussian window of 11 taps with sigma 1.5.
pub fn ssim_map(a: &Rgb, b: &Rgb) -> Result<Vec<f32>, String> {
    if (a.w, a.h) != (b.w, b.h) {
        return Err("image sizes differ".into());
    }
    let (w, h) = (a.w, a.h);
    let luma = |im: &Rgb| -> Vec<f32> { im.px.iter().map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]).collect() };
    let (la, lb) = (luma(a), luma(b));
    let mut k: Vec<f32> = (-5i32..=5).map(|x| (-(x * x) as f32 / (2.0 * 1.5 * 1.5)).exp()).collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    let blur = |v: &[f32]| conv_sep(v, w, h, &k, &k);
    let (ma, mb) = (blur(&la), blur(&lb));
    let aa: Vec<f32> = la.iter().map(|v| v * v).collect();
    let bb: Vec<f32> = lb.iter().map(|v| v * v).collect();
    let ab: Vec<f32> = la.iter().zip(&lb).map(|(x, y)| x * y).collect();
    let (saa, sbb, sab) = (blur(&aa), blur(&bb), blur(&ab));
    let (c1, c2) = (0.01f32 * 0.01, 0.03f32 * 0.03);
    Ok((0..w * h)
        .map(|i| {
            let (va, vb, cov) = (saa[i] - ma[i] * ma[i], sbb[i] - mb[i] * mb[i], sab[i] - ma[i] * mb[i]);
            ((2.0 * ma[i] * mb[i] + c1) * (2.0 * cov + c2)) / ((ma[i] * ma[i] + mb[i] * mb[i] + c1) * (va + vb + c2))
        })
        .collect())
}

// ---------- pooling, masks, heat maps ----------

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub n: usize,
    pub mean: f32,
    pub p50: f32,
    pub p95: f32,
    pub p99: f32,
    pub max: f32,
}

pub fn stats(mut v: Vec<f32>) -> Stats {
    if v.is_empty() {
        return Stats::default();
    }
    let n = v.len();
    let mean = (v.iter().map(|&x| x as f64).sum::<f64>() / n as f64) as f32;
    v.sort_by(|a, b| a.total_cmp(b));
    let at = |q: f32| v[(((n - 1) as f32) * q).round() as usize];
    Stats { n, mean, p50: at(0.5), p95: at(0.95), p99: at(0.99), max: v[n - 1] }
}

/// Pools `vals` over the pixels where `mask` is set (all pixels when None).
pub fn masked_stats(vals: &[f32], mask: Option<&[bool]>) -> Stats {
    match mask {
        Some(m) => stats(vals.iter().zip(m).filter(|(_, &k)| k).map(|(&v, _)| v).collect()),
        None => stats(vals.to_vec()),
    }
}

/// Grows a mask by `r` pixels in every direction (the eye's blur reaches past the model's edge).
pub fn dilate(mask: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    let mut a = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] {
                for xx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                    a[y * w + xx] = true;
                }
            }
        }
    }
    let mut b = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if a[y * w + x] {
                for yy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                    b[yy * w + x] = true;
                }
            }
        }
    }
    b
}

/// Mask of pixels that differ from `bg` (within `tol` per channel) in either image, grown by `grow`.
pub fn foreground_mask(a: &Rgb, b: &Rgb, bg: [f32; 3], tol: f32, grow: usize) -> Vec<bool> {
    let off = |p: &[f32; 3]| (0..3).any(|c| (p[c] - bg[c]).abs() > tol);
    let m: Vec<bool> = a.px.iter().zip(&b.px).map(|(p, q)| off(p) || off(q)).collect();
    dilate(&m, a.w, a.h, grow)
}

/// Magma-like colormap, `v` in 0..1.
pub fn magma(v: f32) -> [u8; 3] {
    const C: [(f32, [f32; 3]); 9] = [
        (0.0, [0.0, 0.0, 4.0]),
        (0.13, [28.0, 16.0, 68.0]),
        (0.25, [79.0, 18.0, 123.0]),
        (0.38, [129.0, 37.0, 129.0]),
        (0.5, [181.0, 54.0, 122.0]),
        (0.63, [229.0, 80.0, 100.0]),
        (0.75, [251.0, 135.0, 97.0]),
        (0.88, [254.0, 194.0, 135.0]),
        (1.0, [252.0, 253.0, 191.0]),
    ];
    let v = v.clamp(0.0, 1.0);
    for k in 1..C.len() {
        if v <= C[k].0 {
            let t = (v - C[k - 1].0) / (C[k].0 - C[k - 1].0);
            return std::array::from_fn(|c| (C[k - 1].1[c] * (1.0 - t) + C[k].1[c] * t) as u8);
        }
    }
    [252, 253, 191]
}

/// RGBA8 heat map of an error map; `full` is the error shown as the brightest color.
pub fn heat_rgba(err: &[f32], full: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(err.len() * 4);
    for &e in err {
        let c = magma(e / full);
        out.extend_from_slice(&[c[0], c[1], c[2], 255]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(w: usize, h: usize, c: [f32; 3]) -> Rgb {
        Rgb { w, h, px: vec![c; w * h] }
    }

    fn disc(w: usize, h: usize, r: f32, fg: [f32; 3], bg: [f32; 3], shift: f32) -> Rgb {
        let mut im = flat(w, h, bg);
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f32 - w as f32 / 2.0 - shift, y as f32 - h as f32 / 2.0);
                if dx * dx + dy * dy < r * r {
                    im.px[y * w + x] = fg;
                }
            }
        }
        im
    }

    #[test]
    fn same_image_scores_zero_and_one() {
        let a = disc(96, 96, 30.0, [0.8, 0.3, 0.2], [0.2, 0.2, 0.25], 0.0);
        let f = flip(&a, &a, DEFAULT_PPD).unwrap();
        assert!(f.err.iter().all(|&e| e == 0.0));
        let s = ssim_map(&a, &a).unwrap();
        assert!(s.iter().all(|&v| (v - 1.0).abs() < 1e-4));
    }

    #[test]
    fn black_against_white_is_near_the_top() {
        let f = flip(&flat(64, 64, [0.0; 3]), &flat(64, 64, [1.0; 3]), DEFAULT_PPD).unwrap();
        let m = stats(f.err).mean;
        assert!(m > 0.9, "black vs white gave {m}");
    }

    #[test]
    fn bigger_changes_score_higher() {
        let a = disc(96, 96, 30.0, [0.8, 0.3, 0.2], [0.2, 0.2, 0.25], 0.0);
        let small = flip(&a, &disc(96, 96, 30.0, [0.8, 0.3, 0.2], [0.2, 0.2, 0.25], 1.0), DEFAULT_PPD).unwrap();
        let big = flip(&a, &disc(96, 96, 30.0, [0.8, 0.3, 0.2], [0.2, 0.2, 0.25], 6.0), DEFAULT_PPD).unwrap();
        let (s, b) = (stats(small.err).mean, stats(big.err).mean);
        assert!(s > 0.0 && b > s, "shift 1 gave {s}, shift 6 gave {b}");
    }

    #[test]
    fn a_hue_change_shows_up() {
        let a = disc(96, 96, 30.0, [0.8, 0.3, 0.2], [0.2, 0.2, 0.25], 0.0);
        let b = disc(96, 96, 30.0, [0.2, 0.3, 0.8], [0.2, 0.2, 0.25], 0.0);
        assert!(stats(flip(&a, &b, DEFAULT_PPD).unwrap().err).p99 > 0.4);
    }

    #[test]
    fn cmax_matches_the_reference_range() {
        // Green to blue in Hunt-adjusted L*a*b*, to the power 0.7: about 40.
        let c = cmax();
        assert!(c > 30.0 && c < 50.0, "cmax {c}");
    }
}
