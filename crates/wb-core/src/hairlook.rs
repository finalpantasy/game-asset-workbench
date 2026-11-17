//! Close-look detectors for strand textures (hair, fur, grass cards): the things a person rejects
//! at close range that color and flow checks pass.
//!
//! Render space (one rendered view, a part mask and a layer id per pixel, such as the lock):
//! * specks: connected bits of one layer smaller than `small` pixels, inside another layer or
//!   floating (shards, dots, the lit half of a stipple);
//! * holes: small enclosed bits of a non-part surface or the background showing through the part
//!   (see-through dots);
//! * sharpness: the mean second difference of lightness across the strands (strand direction
//!   from the structure tensor), i.e. how crisp thin strands are at this distance;
//! * flat patches: connected areas whose local lightness spread is under `flat_std` (plastic);
//! * layer jumps: pixels where two layers meet and their strand directions differ by more than
//!   `jump_deg` (a hard direction cut).
//!
//! Texture space: `alpha_islands` counts cutout islands and enclosed holes under `small` texels.
//!
//! Generic: nothing here knows about a game. Compare a candidate with its source on the same views.

use crate::hair::lab;

#[derive(Clone, Copy, Debug)]
pub struct LookOpts {
    pub small: usize,
    pub flat_std: f64,
    pub flat_min: usize,
    pub jump_deg: f64,
}

impl Default for LookOpts {
    fn default() -> Self {
        LookOpts { small: 30, flat_std: 1.6, flat_min: 60, jump_deg: 40.0 }
    }
}

/// One rendered view: RGBA pixels, the part mask, and a layer id per pixel: >= 0 for the part's
/// layers (locks), any negative id for the rest (e.g. -1 other surfaces, -2 background).
pub struct View<'a> {
    pub w: usize,
    pub h: usize,
    pub rgba: &'a [u8],
    pub part: &'a [bool],
    pub layer: &'a [i64],
}

#[derive(Clone, Debug, Default)]
pub struct LookStats {
    pub part_px: usize,
    pub specks: usize,
    pub speck_px: usize,
    pub holes: usize,
    pub hole_px: usize,
    /// Specks and holes of 1-3 pixels: the stipple signature (a source texel is bigger than that
    /// when magnified, so these come from noise near the cutout, not from the design).
    pub tiny: usize,
    /// Mean |second difference| of L across strands, over coherent interior pixels.
    pub sharpness: f64,
    pub sharp_px: usize,
    /// Pixels in flat patches (at least `flat_min` connected pixels under `flat_std`), and the
    /// largest such patch.
    pub flat_px: usize,
    pub flat_max: usize,
    pub layer_edge_px: usize,
    pub jump_px: usize,
    /// Where things were found: (kind, x, y, size) of the largest few per kind, for crops.
    pub places: Vec<(&'static str, usize, usize, usize)>,
}

impl LookStats {
    pub fn add(&mut self, o: &LookStats) {
        let sp = self.sharp_px + o.sharp_px;
        self.sharpness = if sp > 0 { (self.sharpness * self.sharp_px as f64 + o.sharpness * o.sharp_px as f64) / sp as f64 } else { 0.0 };
        self.sharp_px = sp;
        self.part_px += o.part_px;
        self.specks += o.specks;
        self.speck_px += o.speck_px;
        self.holes += o.holes;
        self.hole_px += o.hole_px;
        self.tiny += o.tiny;
        self.flat_px += o.flat_px;
        self.flat_max = self.flat_max.max(o.flat_max);
        self.layer_edge_px += o.layer_edge_px;
        self.jump_px += o.jump_px;
    }
}

/// 4-connected components of `on` pixels: (label per pixel or usize::MAX, sizes, touches border).
pub fn components(w: usize, h: usize, on: &dyn Fn(usize) -> bool, same: &dyn Fn(usize, usize) -> bool) -> (Vec<usize>, Vec<usize>, Vec<bool>, Vec<(usize, usize)>) {
    let mut lab = vec![usize::MAX; w * h];
    let (mut sizes, mut border, mut seed) = (Vec::new(), Vec::new(), Vec::new());
    let mut stack = Vec::new();
    for s in 0..w * h {
        if lab[s] != usize::MAX || !on(s) {
            continue;
        }
        let id = sizes.len();
        let (mut n, mut b) = (0usize, false);
        lab[s] = id;
        stack.push(s);
        while let Some(p) = stack.pop() {
            n += 1;
            let (x, y) = (p % w, p / w);
            if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                b = true;
            }
            let mut nb = |q: usize| {
                if lab[q] == usize::MAX && on(q) && same(p, q) {
                    lab[q] = id;
                    stack.push(q);
                }
            };
            if x > 0 {
                nb(p - 1);
            }
            if x + 1 < w {
                nb(p + 1);
            }
            if y > 0 {
                nb(p - w);
            }
            if y + 1 < h {
                nb(p + w);
            }
        }
        sizes.push(n);
        border.push(b);
        seed.push((s % w, s / w));
    }
    (lab, sizes, border, seed)
}

pub fn measure(v: &View, o: &LookOpts) -> LookStats {
    let (w, h) = (v.w, v.h);
    let mut st = LookStats { part_px: v.part.iter().filter(|&&p| p).count(), ..Default::default() };
    let mut places = Vec::new();
    // Specks and holes: small components of one layer id (part layers, other surfaces and the
    // background each count as their own), next to the part.
    let (lab_c, sz, bd, seed) = components(w, h, &|_| true, &|a, b| v.layer[a] == v.layer[b]);
    let mut near_part = vec![false; sz.len()];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if v.part[i] {
                near_part[lab_c[i]] = true;
                continue;
            }
            let c = lab_c[i];
            if (x > 0 && v.part[i - 1]) || (x + 1 < w && v.part[i + 1]) || (y > 0 && v.part[i - w]) || (y + 1 < h && v.part[i + w]) {
                near_part[c] = true;
            }
        }
    }
    let (mut ns, mut nh) = (0, 0);
    for (k, &n) in sz.iter().enumerate() {
        if n >= o.small || !near_part[k] {
            continue;
        }
        let (sx, sy) = seed[k];
        if n <= 3 && (v.part[sy * w + sx] || !bd[k]) {
            st.tiny += 1;
        }
        if v.part[sy * w + sx] {
            st.specks += 1;
            st.speck_px += n;
            if ns < 8 {
                places.push(("speck", sx, sy, n));
            }
            ns += 1;
        } else if !bd[k] {
            st.holes += 1;
            st.hole_px += n;
            if nh < 8 {
                places.push(("hole", sx, sy, n));
            }
            nh += 1;
        }
    }
    // Lightness.
    let l: Vec<f64> = (0..w * h).map(|i| lab([v.rgba[i * 4] as f64, v.rgba[i * 4 + 1] as f64, v.rgba[i * 4 + 2] as f64])[0]).collect();
    let same = |a: usize, b: usize| v.part[a] && v.part[b] && v.layer[a] == v.layer[b];
    // Structure tensor (gradients inside one layer), box-smoothed within the layer.
    let mut g = vec![[0.0f64; 3]; w * h];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w - 1 {
            let i = y * w + x;
            if !same(i, i - 1) || !same(i, i + 1) || !same(i, i - w) || !same(i, i + w) {
                continue;
            }
            let gx = 0.5 * (l[i + 1] - l[i - 1]);
            let gy = 0.5 * (l[i + w] - l[i - w]);
            g[i] = [gx * gx, gx * gy, gy * gy];
        }
    }
    let r = 3i64;
    let mut ori = vec![(0.0f64, 0.0f64); w * h]; // (strand angle, coherence)
    let mut sum_l = vec![(0.0f64, 0.0f64, 0usize); w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !v.part[i] {
                continue;
            }
            let (mut t, mut s1, mut s2, mut n) = ([0.0; 3], 0.0, 0.0, 0usize);
            for dy in -r..=r {
                for dx in -r..=r {
                    let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                    if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                        continue;
                    }
                    let j = yy as usize * w + xx as usize;
                    if !same(i, j) {
                        continue;
                    }
                    for c in 0..3 {
                        t[c] += g[j][c];
                    }
                    s1 += l[j];
                    s2 += l[j] * l[j];
                    n += 1;
                }
            }
            sum_l[i] = (s1, s2, n);
            let tr = t[0] + t[2];
            if tr > 1e-9 {
                let disc = ((t[0] - t[2]).powi(2) + 4.0 * t[1] * t[1]).sqrt();
                let coh = disc / tr;
                let th = 0.5 * (2.0 * t[1]).atan2(t[0] - t[2]); // gradient angle
                ori[i] = (th + std::f64::consts::FRAC_PI_2, coh);
            }
        }
    }
    // Sharpness across strands.
    let samp = |x: f64, y: f64, i: usize| -> Option<f64> {
        let (x0, y0) = (x.floor(), y.floor());
        if x0 < 0.0 || y0 < 0.0 || x0 as usize + 1 >= w || y0 as usize + 1 >= h {
            return None;
        }
        let (fx, fy) = (x - x0, y - y0);
        let (a, b) = (x0 as usize, y0 as usize);
        let ids = [b * w + a, b * w + a + 1, (b + 1) * w + a, (b + 1) * w + a + 1];
        if ids.iter().any(|&j| !same(i, j)) {
            return None;
        }
        Some((l[ids[0]] * (1.0 - fx) + l[ids[1]] * fx) * (1.0 - fy) + (l[ids[2]] * (1.0 - fx) + l[ids[3]] * fx) * fy)
    };
    let (mut ss, mut sn) = (0.0, 0usize);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (th, coh) = ori[i];
            if !v.part[i] || coh < 0.2 {
                continue;
            }
            let (nx, ny) = (-th.sin(), th.cos()); // across the strands
            let (Some(a), Some(b)) = (samp(x as f64 + nx, y as f64 + ny, i), samp(x as f64 - nx, y as f64 - ny, i)) else { continue };
            ss += (l[i] - 0.5 * (a + b)).abs();
            sn += 1;
        }
    }
    st.sharpness = if sn > 0 { ss / sn as f64 } else { 0.0 };
    st.sharp_px = sn;
    // Flat patches.
    let flat: Vec<bool> = (0..w * h)
        .map(|i| {
            let (s1, s2, n) = sum_l[i];
            v.part[i] && n >= 25 && {
                let m = s1 / n as f64;
                (s2 / n as f64 - m * m).max(0.0).sqrt() < o.flat_std
            }
        })
        .collect();
    let (_, sz, _, seed) = components(w, h, &|i| flat[i], &|a, b| v.layer[a] == v.layer[b]);
    let mut flats = Vec::new();
    for (k, &n) in sz.iter().enumerate() {
        if n >= o.flat_min {
            st.flat_px += n;
            st.flat_max = st.flat_max.max(n);
            flats.push(("flat", seed[k].0, seed[k].1, n));
        }
    }
    flats.sort_by(|a, b| b.3.cmp(&a.3));
    places.extend(flats.into_iter().take(6));
    // Layer jumps: two layers meet; compare strand directions two pixels into each side.
    let cosj = o.jump_deg.to_radians().cos();
    for y in 2..h.saturating_sub(2) {
        for x in 2..w - 2 {
            let i = y * w + x;
            if !v.part[i] {
                continue;
            }
            for (dx, dy) in [(1i64, 0i64), (0, 1)] {
                let j = (y as i64 + dy) as usize * w + (x as i64 + dx) as usize;
                if !v.part[j] || v.layer[j] == v.layer[i] {
                    continue;
                }
                st.layer_edge_px += 1;
                let a = (y as i64 - dy) as usize * w + (x as i64 - dx) as usize;
                let yb = y as i64 + 2 * dy;
                let xb = x as i64 + 2 * dx;
                if yb >= h as i64 || xb >= w as i64 {
                    continue;
                }
                let b = yb as usize * w + xb as usize;
                if v.layer[a] != v.layer[i] || v.layer[b] != v.layer[j] {
                    continue;
                }
                let (ta, ca) = ori[a];
                let (tb, cb) = ori[b];
                if ca < 0.3 || cb < 0.3 {
                    continue;
                }
                if (ta - tb).cos().abs() < cosj {
                    st.jump_px += 1;
                    if st.jump_px % 25 == 1 && places.iter().filter(|p| p.0 == "jump").count() < 8 {
                        places.push(("jump", x, y, 1));
                    }
                }
            }
        }
    }
    st.places = places;
    st
}

/// Texture-space cutout islands: (islands, island texels, holes, hole texels). `chart[i]` names the
/// chart of each texel (usize::MAX = none). An island is a component of alpha >= 128 inside one
/// chart under `small` texels; a hole is a component of alpha < 128 inside one chart under `small`
/// texels that touches no texel of another chart or outside all charts.
pub fn alpha_islands(rgba: &[u8], w: usize, h: usize, chart: &[usize], small: usize) -> (usize, usize, usize, usize) {
    let same = |a: usize, b: usize| chart[a] == chart[b];
    let on = |i: usize| chart[i] != usize::MAX && rgba[i * 4 + 3] >= 128;
    let (_, sz, _, _) = components(w, h, &on, &same);
    let (mut a, mut an) = (0, 0);
    for &n in &sz {
        if n < small {
            a += 1;
            an += n;
        }
    }
    let off = |i: usize| chart[i] != usize::MAX && rgba[i * 4 + 3] < 128;
    let (lab, sz, bd, _) = components(w, h, &off, &same);
    let mut open = bd.clone();
    for i in 0..w * h {
        if lab[i] == usize::MAX {
            continue;
        }
        let (x, y) = (i % w, i / w);
        let out = |j: usize| chart[j] != chart[i];
        if (x > 0 && out(i - 1)) || (x + 1 < w && out(i + 1)) || (y > 0 && out(i - w)) || (y + 1 < h && out(i + w)) {
            open[lab[i]] = true;
        }
    }
    let (mut ho, mut hn) = (0, 0);
    for (k, &n) in sz.iter().enumerate() {
        if !open[k] && n < small {
            ho += 1;
            hn += n;
        }
    }
    (a, an, ho, hn)
}
