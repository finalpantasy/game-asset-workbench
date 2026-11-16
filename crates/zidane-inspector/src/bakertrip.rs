//! `gaw bake STOCK.glb --step roundtrip --pattern PATTERN.glb --out-dir NEW`
//!
//! A round trip with a known right answer. Stock renders of a model (stock UVs and textures) are
//! upscaled with a plain resampler and baked onto a re-laid-out copy of the same model (pattern
//! atlas) with no knowledge of the stock textures. The result is compared with the truth two ways:
//! texel by texel (each pattern texel's stock color, found through the shared triangles) and by
//! rendering both models from the same views, including views the bake never saw.
//! The bake runs twice, without and with the push-pull fill step.
use super::{blit, downscale, label, num, tri_parts, wjson, write};
use crate::Args;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use wb_core::bake::*;
use wb_core::pattern::Part;
use wb_core::{image_textures, swap, Model, Texture};

const PARTS: [Part; 8] = [Part::Head, Part::Torso, Part::Tail, Part::Arm, Part::Hand, Part::Leg, Part::Foot, Part::Other];

fn tri_locs(m: &Model) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(m.triangle_count());
    for (mi, me) in m.meshes.iter().enumerate() {
        for ti in 0..me.tris.len() {
            v.push((mi, ti));
        }
    }
    v
}

#[derive(Default, Clone)]
struct Acc {
    n: u64,
    abs: f64,
    sq: f64,
    over: u64,
}

impl Acc {
    fn add(&mut self, a: [u8; 3], b: [u8; 3]) {
        let mut s = 0.0f64;
        let mut q = 0.0f64;
        for k in 0..3 {
            let d = (a[k] as f64 - b[k] as f64).abs();
            s += d;
            q += d * d;
        }
        let m = s / 3.0;
        self.n += 1;
        self.abs += m;
        self.sq += q / 3.0;
        if m > 24.0 {
            self.over += 1;
        }
    }
    fn json(&self) -> Value {
        if self.n == 0 {
            return json!({ "n": 0 });
        }
        let n = self.n as f64;
        let rmse = (self.sq / n).sqrt();
        json!({ "n": self.n, "mae": r2(self.abs / n), "rmse": r2(rmse), "psnr_db": if rmse > 0.0 { json!(r2(20.0 * (255.0 / rmse).log10())) } else { Value::Null }, "share_over_24": r4(self.over as f64 / n) })
    }
    fn mae(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.abs / self.n as f64
        }
    }
}

fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}
fn r4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

fn px3(buf: &[u8], i: usize) -> [u8; 3] {
    [buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2]]
}

fn heat(e: f32) -> [u8; 3] {
    let t = (e / 40.0).clamp(0.0, 1.0);
    [(255.0 * (t * 2.0).min(1.0)) as u8, (255.0 * ((t - 0.35) * 2.2).clamp(0.0, 1.0)) as u8, (255.0 * ((t - 0.8) * 5.0).clamp(0.0, 1.0)) as u8 + (50.0 * t) as u8]
}

struct VDef {
    name: String,
    yaw: f32,
    pitch: f32,
}

fn vd(yaw: f32, pitch: f32) -> VDef {
    VDef { name: format!("y{}_p{}", yaw, pitch), yaw, pitch }
}

fn spec(v: &VDef, size: usize, fill: f32) -> ViewSpec {
    ViewSpec { name: v.name.clone(), yaw_deg: v.yaw, pitch_deg: v.pitch, ortho: false, size, fill, clip: None, time: 0.0 }
}

fn tex_with(pat: &Model, atlases: &[AtlasOut]) -> Vec<Option<Texture>> {
    let mut t = image_textures(pat);
    for a in atlases {
        t[a.image] = Some(Texture::new(a.w, a.h, a.fin.clone()));
    }
    t
}

fn write_glb(pat: &Model, atlases: &[AtlasOut], path: &Path) -> Result<(), String> {
    let mut bytes = pat.glb.to_bytes();
    for a in atlases {
        let png = Img { w: a.w, h: a.h, rgba: a.fin.clone() }.to_png()?;
        let cur = Model::from_bytes(&bytes)?;
        bytes = swap::swap(&cur, &a.name, &png)?.bytes;
    }
    write(path, &bytes)
}

/// The stock color each pattern texel should hold, found through the triangles both models share.
fn truth_atlas(stock: &Model, pat: &Model, a: &AtlasOut, simgs: &mut HashMap<usize, Img>) -> Result<Vec<u8>, String> {
    let (w, h) = (a.w, a.h);
    let mut out = vec![0u8; w * h * 4];
    let (sl, pl) = (tri_locs(stock), tri_locs(pat));
    for g in 0..pl.len() {
        let (mi, ti) = pl[g];
        let m = &pat.meshes[mi];
        if m.image != Some(a.image) {
            continue;
        }
        let (smi, sti) = sl[g];
        let sm = &stock.meshes[smi];
        let simg = match sm.image {
            Some(i) => {
                if !simgs.contains_key(&i) {
                    simgs.insert(i, Img::decode(&stock.images[i].bytes)?);
                }
                &simgs[&i]
            }
            None => continue,
        };
        let t = m.tris[ti];
        let st = sm.tris[sti];
        let p = t.map(|v| [m.uv[v as usize][0] * w as f32, m.uv[v as usize][1] * h as f32]);
        let su = st.map(|v| sm.uv[v as usize]);
        raster_tri(p, w, h, |x, y, bc| {
            let i = y * w + x;
            if a.owner[i] != g as u32 + 1 {
                return;
            }
            let u = bc[0] * su[0][0] + bc[1] * su[1][0] + bc[2] * su[2][0];
            let v = bc[0] * su[0][1] + bc[1] * su[1][1] + bc[2] * su[2][1];
            let c = simg.bilinear(u * simg.w as f32, v * simg.h as f32);
            for k in 0..3 {
                out[i * 4 + k] = c[k].round().clamp(0.0, 255.0) as u8;
            }
            out[i * 4 + 3] = 255;
        });
    }
    Ok(out)
}

fn seam_diff(rgba: &[u8], w: usize, h: usize, pairs: &[([f32; 2], [f32; 2])]) -> f64 {
    let (mut s, mut n) = (0.0f64, 0usize);
    for (pa, pb) in pairs {
        let ca = bilinear_rgba(rgba, w, h, pa[0], pa[1]);
        let cb = bilinear_rgba(rgba, w, h, pb[0], pb[1]);
        s += ((ca[0] - cb[0]).abs() + (ca[1] - cb[1]).abs() + (ca[2] - cb[2]).abs()) as f64 / 3.0;
        n += 1;
    }
    if n > 0 {
        s / n as f64
    } else {
        0.0
    }
}

/// Mean RGB gradient (right and down neighbor, same owner triangle) of the texels in `sel`.
fn grad_at(buf: &[u8], owner: &[u32], w: usize, h: usize, i: usize) -> Option<f32> {
    let (x, y) = (i % w, i / w);
    let mut s = 0.0f32;
    let mut n = 0;
    for (dx, dy) in [(1usize, 0usize), (0, 1)] {
        let (xx, yy) = (x + dx, y + dy);
        if xx < w && yy < h {
            let j = yy * w + xx;
            if owner[j] == owner[i] {
                s += ((buf[i * 4] as f32 - buf[j * 4] as f32).abs() + (buf[i * 4 + 1] as f32 - buf[j * 4 + 1] as f32).abs() + (buf[i * 4 + 2] as f32 - buf[j * 4 + 2] as f32).abs()) / 3.0;
                n += 1;
            }
        }
    }
    if n > 0 {
        Some(s / n as f32)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn cmp_view(stock: &Img, sids: &[i32], base: &Img, bids: &[i32], size: usize, region: &[usize], acc: &mut [Acc; 8], total: &mut Acc, err_img: Option<&mut Img>) {
    let mut err_img = err_img;
    for y in 1..size - 1 {
        for x in 1..size - 1 {
            let i = y * size + x;
            if bids[i] < 0 || sids[i] < 0 {
                continue;
            }
            let mut ok = true;
            'n: for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let j = ((y as i64 + dy) as usize) * size + (x as i64 + dx) as usize;
                    if bids[j] < 0 || sids[j] < 0 {
                        ok = false;
                        break 'n;
                    }
                }
            }
            if !ok {
                continue;
            }
            let (a, b) = (px3(&stock.rgba, i), px3(&base.rgba, i));
            let r = region[bids[i] as usize];
            acc[r].add(a, b);
            total.add(a, b);
            if let Some(e) = err_img.as_mut() {
                let m = ((a[0] as f32 - b[0] as f32).abs() + (a[1] as f32 - b[1] as f32).abs() + (a[2] as f32 - b[2] as f32).abs()) / 3.0;
                let c = heat(m * 2.0);
                e.rgba[i * 4..i * 4 + 3].copy_from_slice(&c);
            }
        }
    }
}

pub fn step_roundtrip(args: &Args, stock: &Model, input: &Path, out: &Path) -> Result<(), String> {
    let t_all = std::time::Instant::now();
    let pat_path = PathBuf::from(args.opt("--pattern").ok_or("roundtrip needs --pattern PATTERN.glb")?);
    // The pattern model's atlases already hold the stock colors in the new layout. Replace the color of every
    // atlas with flat gray so the bake starts with no knowledge of the stock textures.
    let pat = {
        let m = Model::open(&pat_path)?;
        let mut bytes = m.glb.to_bytes();
        let mut done: Vec<usize> = Vec::new();
        for me in &m.meshes {
            if let Some(i) = me.image {
                if !done.contains(&i) {
                    done.push(i);
                    let d = Img::decode(&m.images[i].bytes)?;
                    // Flat gray color; the stock cut-out (alpha) stays, because it is a shape, not a color.
                    let mut g = Img::new(d.w, d.h, [128, 128, 128, 255]);
                    for k in 0..d.w * d.h {
                        g.rgba[k * 4 + 3] = d.rgba[k * 4 + 3];
                    }
                    let png = g.to_png()?;
                    let cur = Model::from_bytes(&bytes)?;
                    bytes = swap::swap(&cur, m.images[i].name.trim_end_matches(".png"), &png)?.bytes;
                }
            }
        }
        Model::from_bytes(&bytes)?
    };
    let size: usize = num(args, "--size", 1024usize)?;
    let up: usize = num(args, "--up", 4usize)?;
    let fillf: f32 = num(args, "--fill", 0.8f32)?;
    for d in ["raw", "filled", "truth"] {
        std::fs::create_dir_all(out.join(d)).map_err(|e| e.to_string())?;
    }
    // 1. The two models must share triangles (same count, same corner positions in the same order).
    let (sl, pl) = (tri_locs(stock), tri_locs(&pat));
    if sl.len() != pl.len() {
        return Err(format!("the models have {} and {} triangles; a round trip needs the same triangles", sl.len(), pl.len()));
    }
    let cen = |m: &Model, (mi, ti): (usize, usize)| -> [f32; 3] {
        let me = &m.meshes[mi];
        let t = me.tris[ti];
        let mut c = [0.0f32; 3];
        for v in t {
            for k in 0..3 {
                c[k] += me.pos[v as usize][k] / 3.0;
            }
        }
        c
    };
    let mut max_dev = 0.0f32;
    for g in 0..sl.len() {
        let (a, b) = (cen(stock, sl[g]), cen(&pat, pl[g]));
        max_dev = max_dev.max(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt());
    }
    let (lo, hi) = wb_core::render::bounds(stock);
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    if max_dev > 1e-4 * diag {
        return Err(format!("triangle {} centers differ by up to {max_dev}; the models do not share their triangle order", sl.len()));
    }
    println!("models share {} triangles (largest center difference {:.2e})", sl.len(), max_dev);
    let parts = tri_parts(&pat);
    let region: Vec<usize> = parts.iter().map(|&p| p as usize).collect();
    let names: Vec<&str> = PARTS.iter().map(|p| p.name()).collect();

    // 2. Views.
    let mut train: Vec<VDef> = Vec::new();
    for yaw in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0] {
        train.push(vd(yaw, 0.0));
    }
    for pitch in [40.0, -40.0] {
        for yaw in [0.0, 90.0, 180.0, 270.0] {
            train.push(vd(yaw, pitch));
        }
    }
    train.push(vd(0.0, 85.0));
    train.push(vd(0.0, -85.0));
    let mut held: Vec<VDef> = Vec::new();
    for yaw in [22.5, 112.5, 202.5, 292.5] {
        held.push(vd(yaw, 20.0));
    }
    for yaw in [67.5, 157.5, 247.5, 337.5] {
        held.push(vd(yaw, -20.0));
    }
    let stock_tex = image_textures(stock);
    let pat_tex = image_textures(&pat);

    // 3. Stock renders, upscaled with a plain resampler, as paint over the pattern model's guide views.
    let t0 = std::time::Instant::now();
    let mut views: Vec<PaintView> = Vec::new();
    let mut stock_train: Vec<(Img, Vec<i32>)> = Vec::new();
    let (mut id_same, mut id_fg) = (0u64, 0u64);
    let mut reg_json: Vec<Value> = Vec::new();
    for (k, d) in train.iter().enumerate() {
        let sp = spec(d, size, fillf);
        let sv = render_view(stock, &stock_tex, &sp)?;
        let pv = render_view(&pat, &pat_tex, &sp)?;
        for i in 0..sv.ids.len() {
            if sv.ids[i] >= 0 || pv.ids[i] >= 0 {
                id_fg += 1;
                if sv.ids[i] == pv.ids[i] {
                    id_same += 1;
                }
            }
        }
        let paint = sv.guide.upscaled(up);
        if k % 5 == 0 {
            // Does registration find the identity when the guide has different colors than the paint?
            let fg: Vec<bool> = pv.ids.iter().map(|&i| i >= 0).collect();
            let r = match register(&pv.guide, &fg, &paint, &RegOptions::default()) {
                Ok(r) => json!({ "view": d.name, "scale": r.scale, "rotation_deg": r.rot_deg, "shift_px": [r.tx, r.ty], "edge_ncc_after": r.ncc_after }),
                Err(e) => json!({ "view": d.name, "refused": e }),
            };
            reg_json.push(r);
        }
        let reg = Registration::identity(size, size, up as f32);
        let edit = edit_map(&pv.guide, &pv.guide, 0.0);
        stock_train.push((sv.guide.clone(), sv.ids.clone()));
        views.push(PaintView { view: pv, paint, reg, edit });
    }
    println!("{} views rendered and upscaled {}x in {:.1}s; triangle ids agree on {:.4} of the pixels either model covers", views.len(), up, t0.elapsed().as_secs_f64(), id_same as f64 / id_fg.max(1) as f64);

    // 4. Bake without and with the fill step.
    let masks = Masks::all(&pat);
    let sharp: f32 = num(args, "--sharp-power", 0.0f32)?;
    let o_raw = BakeOptions { fill: false, sharp_power: sharp, ..Default::default() };
    let o_fill = BakeOptions { fill: true, sharp_power: sharp, ..Default::default() };
    let t1 = std::time::Instant::now();
    let raw = bake(&pat, &views, &masks, &o_raw)?;
    let t_raw = t1.elapsed().as_secs_f64();
    let t2 = std::time::Instant::now();
    let fil = bake(&pat, &views, &masks, &o_fill)?;
    let t_fill = t2.elapsed().as_secs_f64();
    println!("bake {:.1}s, bake with fill {:.1}s", t_raw, t_fill);
    let views_json = raw.views.clone();
    drop(views);

    // 5. Truth atlases and the texel-level comparison.
    let mut simgs: HashMap<usize, Img> = HashMap::new();
    let mut truths: Vec<Vec<u8>> = Vec::new();
    let nreg = PARTS.len();
    // [variant][region][class]; variants: blank atlas, bake before the seam pass, bake, bake + fill before the seam pass, bake + fill; classes all, full, low, hole.
    let mut tacc: Vec<Vec<Vec<Acc>>> = vec![vec![vec![Acc::default(); 4]; nreg]; 5];
    let edges = [0.02f32, 0.2, 0.4, 0.64, 0.85, 1.01];
    let nb = edges.len() - 1;
    // [bin]: n, err raw, err fill, grad truth, grad raw, grad fill, grad n
    let mut bins: Vec<[f64; 7]> = vec![[0.0; 7]; nb];
    let seen_edges = [1u8, 2, 3, 5, 255];
    let mut by_seen: Vec<[f64; 3]> = vec![[0.0; 3]; seen_edges.len()];
    let mut seam_rows: Vec<Value> = Vec::new();
    let (mut sm_truth, mut sm_raw_b, mut sm_raw_f, mut sm_fill_b, mut sm_fill_f, mut sm_n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut cover_rows: Vec<Value> = Vec::new();
    let mut atlas_sheet_rows: Vec<Vec<Img>> = Vec::new();
    // (atlas index, block x, block y, mean error, texels) per 256 x 256 block, for the worst-block sheet.
    let mut blocks: Vec<(usize, usize, usize, f64, u32)> = Vec::new();
    for (ai, ra) in raw.atlases.iter().enumerate() {
        let fa = &fil.atlases[ai];
        let (w, h) = (ra.w, ra.h);
        let truth = truth_atlas(stock, &pat, ra, &mut simgs)?;
        write(&out.join("truth").join(format!("{}.png", ra.name)), &Img { w, h, rgba: truth.clone() }.to_png()?)?;
        write(&out.join("raw").join(format!("{}.png", ra.name)), &Img { w, h, rgba: ra.fin.clone() }.to_png()?)?;
        write(&out.join("filled").join(format!("{}.png", fa.name)), &Img { w, h, rgba: fa.fin.clone() }.to_png()?)?;
        let (mut n_perm, mut n_full, mut n_low, mut n_hole, mut n_transp) = (0u64, 0u64, 0u64, 0u64, 0u64);
        let mut cov_map = vec![0u8; w * h * 4];
        let mut err_map = vec![0u8; w * h * 4];
        let mut cos_map = vec![0u8; w * h * 4];
        let (bw, bh) = (w / 256, h / 256);
        let mut blk = vec![(0.0f64, 0u32); bw * bh];
        for i in 0..w * h {
            cov_map[i * 4 + 3] = 255;
            err_map[i * 4 + 3] = 255;
            cos_map[i * 4 + 3] = 255;
            if !ra.perm[i] || ra.owner[i] == 0 {
                let g = if ra.gutter_ok[i] { 70 } else { 24 };
                cov_map[i * 4..i * 4 + 3].copy_from_slice(&[g, g, g]);
                err_map[i * 4..i * 4 + 3].copy_from_slice(&[g / 2, g / 2, g / 2]);
                cos_map[i * 4..i * 4 + 3].copy_from_slice(&[g / 2, g / 2, g / 2]);
                continue;
            }
            if ra.orig[i * 4 + 3] < 128 {
                n_transp += 1;
                cov_map[i * 4..i * 4 + 3].copy_from_slice(&[60, 60, 90]);
                err_map[i * 4..i * 4 + 3].copy_from_slice(&[30, 30, 45]);
                cos_map[i * 4..i * 4 + 3].copy_from_slice(&[30, 30, 45]);
                continue;
            }
            n_perm += 1;
            let s = ra.strength[i];
            let class = if s <= 0.0 {
                3
            } else if s < 1.0 {
                2
            } else {
                1
            };
            match class {
                1 => n_full += 1,
                2 => n_low += 1,
                _ => n_hole += 1,
            }
            cov_map[i * 4..i * 4 + 3].copy_from_slice(&match class {
                1 => [40, 150, 70],
                2 => [235, 200, 40],
                _ => [225, 40, 40],
            });
            let t = px3(&truth, i);
            let r = region[ra.owner[i] as usize - 1];
            for (v, buf) in [(0usize, &ra.orig), (1, &ra.baked), (2, &ra.fin), (3, &fa.baked), (4, &fa.fin)] {
                let p = px3(buf, i);
                tacc[v][r][0].add(p, t);
                tacc[v][r][class].add(p, t);
            }
            let e = {
                let p = px3(&fa.fin, i);
                ((p[0] as f32 - t[0] as f32).abs() + (p[1] as f32 - t[1] as f32).abs() + (p[2] as f32 - t[2] as f32).abs()) / 3.0
            };
            err_map[i * 4..i * 4 + 3].copy_from_slice(&heat(e * 2.0));
            {
                let (bx, by) = ((i % w) / 256, (i / w) / 256);
                if bx < bw && by < bh {
                    blk[by * bw + bx].0 += e as f64;
                    blk[by * bw + bx].1 += 1;
                }
            }
            if s > 0.0 {
                let c = ra.cos_best[i] as f32 / 255.0;
                let ramp = [(40.0 + 215.0 * c) as u8, (60.0 + 190.0 * c * c) as u8, (200.0 * (1.0 - c)) as u8];
                cos_map[i * 4..i * 4 + 3].copy_from_slice(&ramp);
                let b = (0..nb).find(|&b| c >= edges[b] && c < edges[b + 1]);
                let (er, ef) = {
                    let (pr, pf) = (px3(&ra.fin, i), px3(&fa.fin, i));
                    let f = |p: [u8; 3]| (((p[0] as f32 - t[0] as f32).abs() + (p[1] as f32 - t[1] as f32).abs() + (p[2] as f32 - t[2] as f32).abs()) / 3.0) as f64;
                    (f(pr), f(pf))
                };
                let (gt, gr, gf) = (grad_at(&truth, &ra.owner, w, h, i), grad_at(&ra.fin, &ra.owner, w, h, i), grad_at(&fa.fin, &ra.owner, w, h, i));
                if let Some(b) = b {
                    bins[b][0] += 1.0;
                    bins[b][1] += er;
                    bins[b][2] += ef;
                    if let (Some(gt), Some(gr), Some(gf)) = (gt, gr, gf) {
                        bins[b][3] += gt as f64;
                        bins[b][4] += gr as f64;
                        bins[b][5] += gf as f64;
                        bins[b][6] += 1.0;
                    }
                }
                let vs = ra.views_seen[i];
                if let Some(k) = (0..seen_edges.len()).find(|&k| vs <= seen_edges[k]) {
                    by_seen[k][0] += 1.0;
                    by_seen[k][1] += er;
                    by_seen[k][2] += ef;
                }
            }
        }
        for (k, &(sum, n)) in blk.iter().enumerate() {
            if n >= 20000 {
                blocks.push((ai, k % bw, k / bw, sum / n as f64, n));
            }
        }
        // Seam difference against the truth on the same sample pairs.
        let tr_r = seam_diff(&truth, w, h, &ra.seam_pairs);
        let rb = seam_diff(&ra.baked, w, h, &ra.seam_pairs);
        let rf = seam_diff(&ra.fin, w, h, &ra.seam_pairs);
        let tr_f = seam_diff(&truth, w, h, &fa.seam_pairs);
        let fb = seam_diff(&fa.baked, w, h, &fa.seam_pairs);
        let ff = seam_diff(&fa.fin, w, h, &fa.seam_pairs);
        seam_rows.push(json!({ "image": ra.name,
            "bake": { "samples": ra.seam_pairs.len(), "truth": r2(tr_r), "before_seam_pass": r2(rb), "after_seam_pass": r2(rf) },
            "bake_with_fill": { "samples": fa.seam_pairs.len(), "truth": r2(tr_f), "before_seam_pass": r2(fb), "after_seam_pass": r2(ff) } }));
        let n = fa.seam_pairs.len() as f64;
        sm_truth += tr_f * n;
        sm_fill_b += fb * n;
        sm_fill_f += ff * n;
        sm_n += n;
        sm_raw_b += rb * ra.seam_pairs.len() as f64;
        sm_raw_f += rf * ra.seam_pairs.len() as f64;
        cover_rows.push(json!({ "image": ra.name, "texels_in_islands_opaque": n_perm, "texels_transparent_skipped": n_transp, "reached_full": n_full, "reached_low_confidence": n_low, "holes_no_view_reached": n_hole,
            "hole_share": r4(n_hole as f64 / n_perm.max(1) as f64), "fill": fa.stats["fill"].clone(), "gutter_texels_written_raw": ra.stats["padding_texels_written"], "gutter_texels_written_fill": fa.stats["padding_texels_written"] }));
        let cell = 512usize;
        atlas_sheet_rows.push(vec![
            downscale(&truth, w, h, cell),
            downscale(&ra.fin, w, h, cell),
            downscale(&fa.fin, w, h, cell),
            downscale(&cov_map, w, h, cell),
            downscale(&err_map, w, h, cell),
            downscale(&cos_map, w, h, cell),
        ]);
        truths.push(truth);
    }
    println!("texel comparison done ({:.1}s since start)", t_all.elapsed().as_secs_f64());

    // 6. Render comparison, training views and held-out views.
    let tex_raw = tex_with(&pat, &raw.atlases);
    let tex_fill = tex_with(&pat, &fil.atlases);
    // Control: the truth inside the islands (padding stays the bake's). Whatever error this shows is the floor
    // that any atlas in this layout has against the stock render (resampling, filtering), not a bake error.
    let tex_truth = {
        let mut t = image_textures(&pat);
        for (ai, fa) in fil.atlases.iter().enumerate() {
            let mut buf = fa.fin.clone();
            for i in 0..fa.w * fa.h {
                if fa.perm[i] && fa.owner[i] > 0 && fa.orig[i * 4 + 3] >= 128 {
                    buf[i * 4..i * 4 + 3].copy_from_slice(&truths[ai][i * 4..i * 4 + 3]);
                }
            }
            t[fa.image] = Some(Texture::new(fa.w, fa.h, buf));
        }
        t
    };
    write_glb(&pat, &raw.atlases, &out.join("raw").join("Zidane-roundtrip-raw.glb"))?;
    write_glb(&pat, &fil.atlases, &out.join("filled").join("Zidane-roundtrip-filled.glb"))?;
    // [variant bake / bake + fill / truth control][set train/held][region]
    let mut racc: Vec<Vec<[Acc; 8]>> = vec![vec![Default::default(), Default::default()]; 3];
    let mut rtot: Vec<Vec<Acc>> = vec![vec![Acc::default(); 2]; 3];
    let mut per_view: Vec<Value> = Vec::new();
    let mut sheet_cells: Vec<(String, Img, Img, Img, Img)> = Vec::new();
    let all: Vec<(&VDef, usize)> = train.iter().map(|d| (d, 0usize)).chain(held.iter().map(|d| (d, 1usize))).collect();
    let show: Vec<&str> = vec!["y0_p0", "y180_p0", "y90_p0", "y0_p40", "y22.5_p20", "y112.5_p20", "y202.5_p20", "y157.5_p-20", "y0_p-40", "y0_p85"];
    for (k, &(d, set)) in all.iter().enumerate() {
        let sp = spec(d, size, fillf);
        let (simg, sids) = if set == 0 {
            let (a, b) = &stock_train[k];
            (a.clone(), b.clone())
        } else {
            let sv = render_view(stock, &stock_tex, &sp)?;
            (sv.guide, sv.ids)
        };
        let vr = render_view(&pat, &tex_raw, &sp)?;
        let vf = render_view(&pat, &tex_fill, &sp)?;
        let vt = render_view(&pat, &tex_truth, &sp)?;
        let keep = show.contains(&d.name.as_str());
        let (mut e_img_f, mut e_img_r) = (Img::new(size, size, [18, 18, 22, 255]), Img::new(size, size, [18, 18, 22, 255]));
        let (mut a_r, mut a_f, mut a_t) = (Acc::default(), Acc::default(), Acc::default());
        let (mut tmp_r, mut tmp_f, mut tmp_t) = (<[Acc; 8]>::default(), <[Acc; 8]>::default(), <[Acc; 8]>::default());
        cmp_view(&simg, &sids, &vt.guide, &vt.ids, size, &region, &mut tmp_t, &mut a_t, None);
        cmp_view(&simg, &sids, &vr.guide, &vr.ids, size, &region, &mut tmp_r, &mut a_r, if keep { Some(&mut e_img_r) } else { None });
        cmp_view(&simg, &sids, &vf.guide, &vf.ids, size, &region, &mut tmp_f, &mut a_f, if keep { Some(&mut e_img_f) } else { None });
        for r in 0..8 {
            for (v, t) in [(0usize, &tmp_r[r]), (1usize, &tmp_f[r]), (2usize, &tmp_t[r])] {
                let acc = &mut racc[v][set][r];
                acc.n += t.n;
                acc.abs += t.abs;
                acc.sq += t.sq;
                acc.over += t.over;
            }
        }
        for (v, a) in [(0usize, &a_r), (1, &a_f), (2, &a_t)] {
            rtot[v][set].n += a.n;
            rtot[v][set].abs += a.abs;
            rtot[v][set].sq += a.sq;
            rtot[v][set].over += a.over;
        }
        per_view.push(json!({ "view": d.name, "yaw": d.yaw, "pitch": d.pitch, "set": if set == 0 { "trained" } else { "held_out" }, "pixels": a_r.n, "mae_bake": r2(a_r.mae()), "mae_bake_with_fill": r2(a_f.mae()), "mae_truth_atlas_control": r2(a_t.mae()) }));
        if keep {
            sheet_cells.push((format!("{} {}", if set == 0 { "trained" } else { "held out" }, d.name), simg, vr.guide, vf.guide, e_img_f));
        }
    }
    println!("render comparison done ({:.1}s since start)", t_all.elapsed().as_secs_f64());

    // 7. Detail close-ups of the head at 2048 px.
    let mut detail: Vec<Img> = Vec::new();
    for d in [vd(0.0, 0.0), vd(22.5, 20.0)] {
        let sp = spec(&d, 2048, fillf);
        let sv = render_view(stock, &stock_tex, &sp)?;
        let vr = render_view(&pat, &tex_raw, &sp)?;
        let vf = render_view(&pat, &tex_fill, &sp)?;
        let (mut x0, mut x1, mut y0, mut y1) = (usize::MAX, 0usize, usize::MAX, 0usize);
        for y in 0..2048 {
            for x in 0..2048 {
                let id = vr.ids[y * 2048 + x];
                if id >= 0 && parts[id as usize] == Part::Head {
                    x0 = x0.min(x);
                    x1 = x1.max(x);
                    y0 = y0.min(y);
                    y1 = y1.max(y);
                }
            }
        }
        if x0 == usize::MAX {
            continue;
        }
        let side = ((x1 - x0).max(y1 - y0) as f32 * 1.1) as usize;
        let (cx, cy) = ((x0 + x1) / 2, (y0 + y1) / 2);
        let (sx, sy) = (cx.saturating_sub(side / 2).min(2048 - side.min(2048)), cy.saturating_sub(side / 2).min(2048 - side.min(2048)));
        for im in [&sv.guide, &vr.guide, &vf.guide] {
            let mut c = Img::new(side, side, [0, 0, 0, 255]);
            for y in 0..side {
                for x in 0..side {
                    let (a, b) = ((sy + y) * 2048 + sx + x, y * side + x);
                    c.rgba[b * 4..b * 4 + 4].copy_from_slice(&im.rgba[a * 4..a * 4 + 4]);
                }
            }
            detail.push(c.resized(512, 512));
        }
    }

    // 8. Sheets.
    let cell = 360usize;
    // Views: stock | bake | bake + fill | error of bake + fill.
    let mut sv = Img::new(cell * 4 + 50, (cell + 34) * sheet_cells.len() + 10, [226, 228, 232, 255]);
    for (r, (title, s, a, b, e)) in sheet_cells.iter().enumerate() {
        let y = 10 + r * (cell + 34);
        label(&mut sv, 10, y, &format!("{title}   stock | bake | bake + fill | error of bake + fill"));
        for (c, im) in [s, a, b, e].iter().enumerate() {
            blit(&mut sv, &im.resized(cell, cell), 10 + c * (cell + 10), y + 28);
        }
    }
    sv.save(&out.join("sheet-views.png"))?;
    // Atlases: truth | bake | bake + fill | coverage | error | best view angle.
    let ac = 512usize;
    let mut sa = Img::new(ac * 6 + 70, (ac + 34) * atlas_sheet_rows.len() + 10, [226, 228, 232, 255]);
    for (r, row) in atlas_sheet_rows.iter().enumerate() {
        let y = 10 + r * (ac + 34);
        label(&mut sa, 10, y, &format!("{}: truth | bake | bake + fill | coverage (green full, yellow weak, red hole) | error of fill | cos of best view (blue grazing)", raw.atlases[r].name));
        for (c, im) in row.iter().enumerate() {
            blit(&mut sa, im, 10 + c * (ac + 10), y + 28);
        }
    }
    sa.save(&out.join("sheet-atlas.png"))?;
    if detail.len() >= 3 {
        let rows = detail.len() / 3;
        let mut sd = Img::new(512 * 3 + 40, (512 + 34) * rows + 10, [226, 228, 232, 255]);
        for r in 0..rows {
            let y = 10 + r * (512 + 34);
            label(&mut sd, 10, y, if r == 0 { "head, front, 2048 px: stock | bake | bake + fill" } else { "head, held-out view, 2048 px: stock | bake | bake + fill" });
            for c in 0..3 {
                blit(&mut sd, &detail[r * 3 + c], 10 + c * (512 + 10), y + 28);
            }
        }
        sd.save(&out.join("sheet-detail.png"))?;
    }

    // Worst blocks: where the bake with fill is furthest from the truth, truth | bake | bake + fill at 2x.
    blocks.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    let mut worst_json: Vec<Value> = Vec::new();
    let mut picked: Vec<(usize, usize, usize, f64, u32)> = Vec::new();
    for b in &blocks {
        if picked.len() >= 4 {
            break;
        }
        if picked.iter().any(|p| p.0 == b.0 && (p.1 as i64 - b.1 as i64).abs() <= 1 && (p.2 as i64 - b.2 as i64).abs() <= 1) {
            continue;
        }
        picked.push(*b);
    }
    if !picked.is_empty() {
        let mut sw = Img::new(512 * 3 + 40, (512 + 34) * picked.len() + 10, [226, 228, 232, 255]);
        for (r, &(ai, bx, by, err, _)) in picked.iter().enumerate() {
            let (ra, fa) = (&raw.atlases[ai], &fil.atlases[ai]);
            let y = 10 + r * (512 + 34);
            label(&mut sw, 10, y, &format!("{} block at {},{} mean error {:.1}: truth | bake | bake + fill", ra.name, bx * 256, by * 256, err));
            worst_json.push(json!({ "image": ra.name, "x": bx * 256, "y": by * 256, "size": 256, "mean_abs_error_with_fill": r2(err) }));
            for (c, buf) in [&truths[ai], &ra.fin, &fa.fin].iter().enumerate() {
                let mut crop = Img::new(256, 256, [0, 0, 0, 255]);
                for yy in 0..256 {
                    for xx in 0..256 {
                        let (a, b) = ((by * 256 + yy) * ra.w + bx * 256 + xx, yy * 256 + xx);
                        crop.rgba[b * 4..b * 4 + 3].copy_from_slice(&buf[a * 4..a * 4 + 3]);
                    }
                }
                blit(&mut sw, &crop.resized(512, 512), 10 + c * (512 + 10), y + 28);
            }
        }
        sw.save(&out.join("sheet-worst-blocks.png"))?;
    }

    // 9. Report.
    let mut tex_json = Vec::new();
    for (vi, vname) in ["blank_gray_no_paint", "bake_no_seam_pass", "bake", "bake_with_fill_no_seam_pass", "bake_with_fill"].iter().enumerate() {
        let mut per_region = serde_json::Map::new();
        for r in 0..nreg {
            per_region.insert(names[r].to_owned(), json!({ "all": tacc[vi][r][0].json(), "full": tacc[vi][r][1].json(), "weak": tacc[vi][r][2].json(), "hole": tacc[vi][r][3].json() }));
        }
        let mut all_acc = [Acc::default(), Acc::default(), Acc::default(), Acc::default()];
        for r in 0..nreg {
            for c in 0..4 {
                let t = &tacc[vi][r][c];
                all_acc[c].n += t.n;
                all_acc[c].abs += t.abs;
                all_acc[c].sq += t.sq;
                all_acc[c].over += t.over;
            }
        }
        tex_json.push(json!({ "variant": vname, "all_regions": { "all": all_acc[0].json(), "full": all_acc[1].json(), "weak": all_acc[2].json(), "hole": all_acc[3].json() }, "per_region": per_region }));
    }
    let bins_json: Vec<Value> = (0..nb)
        .map(|b| {
            let n = bins[b][0].max(1.0);
            let gn = bins[b][6].max(1.0);
            json!({ "cos_best_view": [edges[b], edges[b + 1].min(1.0)], "angle_deg": [(edges[b + 1].min(1.0) as f64).acos().to_degrees().round(), (edges[b] as f64).acos().to_degrees().round()],
                "texels": bins[b][0], "mae_bake": r2(bins[b][1] / n), "mae_bake_with_fill": r2(bins[b][2] / n),
                "gradient_truth": r2(bins[b][3] / gn), "gradient_bake": r2(bins[b][4] / gn), "gradient_bake_with_fill": r2(bins[b][5] / gn),
                "sharpness_ratio_bake": r2(if bins[b][3] > 0.0 { bins[b][4] / bins[b][3] } else { 0.0 }) })
        })
        .collect();
    let seen_json: Vec<Value> = (0..seen_edges.len())
        .map(|k| {
            let n = by_seen[k][0].max(1.0);
            json!({ "views_that_painted_the_texel_at_most": seen_edges[k], "texels": by_seen[k][0], "mae_bake": r2(by_seen[k][1] / n), "mae_bake_with_fill": r2(by_seen[k][2] / n) })
        })
        .collect();
    let mut render_json = Vec::new();
    for (vi, vname) in ["bake", "bake_with_fill", "truth_atlas_control"].iter().enumerate() {
        let mut sets = serde_json::Map::new();
        for (si, sname) in ["trained_views", "held_out_views"].iter().enumerate() {
            let mut per_region = serde_json::Map::new();
            for r in 0..nreg {
                per_region.insert(names[r].to_owned(), racc[vi][si][r].json());
            }
            sets.insert(sname.to_string(), json!({ "all": rtot[vi][si].json(), "per_region": per_region }));
        }
        render_json.push(json!({ "variant": vname, "sets": sets }));
    }
    let report = json!({
        "stock": input.file_name().map(|s| s.to_string_lossy().into_owned()),
        "pattern": pat_path.file_name().map(|s| s.to_string_lossy().into_owned()),
        "method": { "view_size": size, "upscale": format!("{up}x Lanczos3 (plain resampler, no AI)"), "paint_size": size * up, "trained_views": train.iter().map(|d| d.name.clone()).collect::<Vec<_>>(),
                    "held_out_views": held.iter().map(|d| d.name.clone()).collect::<Vec<_>>(), "edit_gate": "off (every pixel counts)", "mask": "all triangles",
                    "bake_options": { "sharp_power": sharp, "pad": o_raw.pad, "grazing_max_deg": o_raw.grazing_max_deg, "grazing_full_deg": o_raw.grazing_full_deg, "min_weight": o_raw.min_weight, "seam_pass": true, "seam_radius": o_raw.seam_radius } },
        "triangle_ids_agree_between_stock_and_pattern_renders": r4(id_same as f64 / id_fg.max(1) as f64),
        "registration_check_identity_expected": reg_json,
        "seconds": { "bake": r2(t_raw), "bake_with_fill": r2(t_fill), "total": r2(t_all.elapsed().as_secs_f64()) },
        "coverage": cover_rows,
        "texel_error_vs_truth": tex_json,
        "error_by_best_view_angle": bins_json,
        "error_by_number_of_painting_views": seen_json,
        "seam": { "per_atlas": seam_rows, "mean_rgb_difference": {
            "truth": r2(sm_truth / sm_n.max(1.0)), "bake_before_seam_pass": r2(sm_raw_b / raw.atlases.iter().map(|a| a.seam_pairs.len() as f64).sum::<f64>().max(1.0)),
            "bake_after_seam_pass": r2(sm_raw_f / raw.atlases.iter().map(|a| a.seam_pairs.len() as f64).sum::<f64>().max(1.0)),
            "fill_before_seam_pass": r2(sm_fill_b / sm_n.max(1.0)), "fill_after_seam_pass": r2(sm_fill_f / sm_n.max(1.0)), "samples_fill_run": sm_n } },
        "render_error_vs_stock": render_json,
        "worst_blocks_256px": worst_json,
        "per_view": per_view,
        "bake_views": views_json,
    });
    wjson(&out.join("roundtrip-report.json"), &report)?;
    print_rt(&report);
    Ok(())
}

fn print_rt(r: &Value) {
    println!("--- round trip ---");
    for c in r["coverage"].as_array().cloned().unwrap_or_default() {
        println!("{}: {} island texels, {} reached at full strength, {} weak, {} holes ({:.2}%)", c["image"], c["texels_in_islands_opaque"], c["reached_full"], c["reached_low_confidence"], c["holes_no_view_reached"], c["hole_share"].as_f64().unwrap_or(0.0) * 100.0);
    }
    for t in r["texel_error_vs_truth"].as_array().cloned().unwrap_or_default() {
        println!("texel error vs truth, {}: mae {} (n {})", t["variant"], t["all_regions"]["all"]["mae"], t["all_regions"]["all"]["n"]);
    }
    for t in r["render_error_vs_stock"].as_array().cloned().unwrap_or_default() {
        println!("render error vs stock, {}: trained mae {}, held out mae {}", t["variant"], t["sets"]["trained_views"]["all"]["mae"], t["sets"]["held_out_views"]["all"]["mae"]);
    }
    println!("seam mean RGB difference: {}", r["seam"]["mean_rgb_difference"]);
}
