//! Extra numbers and maps for a sewing-pattern layout (`pattern`): per-piece texel-density error and
//! anisotropy, how well each piece's dominant bone runs along a texture axis, how exactly left/right
//! pairs mirror each other, and chart / triangle / bone ID maps for painting tools and AI.
use crate::flatten::{cross, dot, norm, sub};
use crate::pattern::{p64, raster_bary, seam_color, Atlas, Layout};
use crate::Model;
use serde_json::{json, Value};

/// A piece's measured texel density may differ from its nominal density by at most this share.
pub const MAX_DENSITY_ERR: f64 = 0.005;
/// A locked mirror pair's corners may differ from an exact reflection by at most this many texels.
pub const MAX_MIRROR_ERR_PX: f64 = 0.05;

pub struct PieceMetric {
    pub piece: usize,
    pub label: String,
    pub triangles: usize,
    pub pinned: bool,
    pub density_factor: f64,
    /// Texels per model unit as laid out, and as the layout's scale and density factor promise.
    pub px_per_unit: f64,
    pub nominal_px_per_unit: f64,
    /// (actual - nominal) / nominal; None for pieces held at their stock UVs.
    pub density_err: Option<f64>,
    /// Area-weighted singular-value ratio of the 3D-to-UV map: 1 is a perfect (conformal) fit.
    pub aniso_p50: f64,
    pub aniso_p95: f64,
    pub aniso_max: f64,
    pub bone: Option<String>,
    /// Angle (degrees) between the bone's direction on the page and the nearest texture axis: of
    /// the piece's mean direction, and the area-weighted 95th percentile over its triangles.
    pub bone_dev_deg: Option<f64>,
    pub bone_dev_p95_deg: Option<f64>,
    pub mirror_of: Option<usize>,
}

pub struct MirrorMetric {
    pub a: usize,
    pub b: usize,
    pub a_label: String,
    pub b_label: String,
    pub corners: [usize; 2],
    pub matched: [usize; 2],
    pub exact: bool,
    pub locked: bool,
    pub max_dist: f64,
    /// Largest gap (texels) between b's corner and the reflection of its partner in a, after the best
    /// shift; None when not locked.
    pub max_err_px: Option<f64>,
    /// Flat area of b over that of a.
    pub area_ratio: f64,
    pub note: &'static str,
}

pub struct PatternMetrics {
    pub size: usize,
    pub pieces: Vec<PieceMetric>,
    pub mirrors: Vec<MirrorMetric>,
    /// Left and right pieces with no partner.
    pub unpaired: Vec<usize>,
    /// Limit for every piece's anisotropy p95 (set when the low-stretch pass is on).
    pub aniso_limit: Option<f64>,
}

pub(crate) fn wpercentile(mut v: Vec<(f64, f64)>, q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = v.iter().map(|x| x.1).sum();
    let mut acc = 0.0;
    for &(x, w) in &v {
        acc += w;
        if acc >= q * total {
            return x;
        }
    }
    v.last().unwrap().0
}

/// Ratio of the larger to the smaller stretch of the map from a 3D triangle to its UV triangle.
pub(crate) fn anisotropy(p: [[f64; 3]; 3], uv: [[f64; 2]; 3]) -> f64 {
    let (e1, e2) = (sub(p[1], p[0]), sub(p[2], p[0]));
    let l1 = norm(e1);
    if l1 < 1e-12 {
        return 1.0;
    }
    let x = [e1[0] / l1, e1[1] / l1, e1[2] / l1];
    let l2x = dot(e2, x);
    let perp = sub(e2, [x[0] * l2x, x[1] * l2x, x[2] * l2x]);
    let l2y = norm(perp);
    let d = l1 * l2y;
    if d < 1e-18 {
        return 1.0;
    }
    let (u1, u2) = ([uv[1][0] - uv[0][0], uv[1][1] - uv[0][1]], [uv[2][0] - uv[0][0], uv[2][1] - uv[0][1]]);
    let j00 = u1[0] * l2y / d;
    let j01 = (-u1[0] * l2x + u2[0] * l1) / d;
    let j10 = u1[1] * l2y / d;
    let j11 = (-u1[1] * l2x + u2[1] * l1) / d;
    let (a, b, c) = (j00 * j00 + j10 * j10, j00 * j01 + j10 * j11, j01 * j01 + j11 * j11);
    let mid = 0.5 * (a + c);
    let r = (0.25 * (a - c) * (a - c) + b * b).sqrt();
    let (hi, lo) = ((mid + r).max(0.0).sqrt(), (mid - r).max(0.0).sqrt());
    if lo < 1e-12 {
        1e6
    } else {
        hi / lo
    }
}

fn axis_dev(g: [f64; 2]) -> f64 {
    let m = g[1].atan2(g[0]).to_degrees().rem_euclid(90.0);
    m.min(90.0 - m)
}

pub fn metrics(model: &Model, lay: &Layout) -> PatternMetrics {
    let s = lay.size as f64;
    let mut pieces = Vec::with_capacity(lay.pieces.len());
    for (pi, p) in lay.pieces.iter().enumerate() {
        let (m, nm) = (&model.meshes[p.mesh], &lay.meshes[p.mesh]);
        let (mut a3, mut auv) = (0.0, 0.0);
        let mut aniso: Vec<(f64, f64)> = Vec::new();
        let mut grads: Vec<([f64; 2], f64)> = Vec::new();
        let axis = if p.bone.is_some() { p.bone_axis } else { [0.0; 3] };
        for &t in &p.tris {
            let pos = m.tris[t as usize].map(|v| p64(m.pos[v as usize]));
            let uv = nm.tris[t as usize].map(|v| [nm.uv[v as usize][0] as f64 * s, nm.uv[v as usize][1] as f64 * s]);
            let w = 0.5 * norm(cross(sub(pos[1], pos[0]), sub(pos[2], pos[0])));
            a3 += w;
            auv += 0.5 * ((uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1]) - (uv[1][1] - uv[0][1]) * (uv[2][0] - uv[0][0])).abs();
            if w > 1e-14 {
                aniso.push((anisotropy(pos, uv), w));
            }
            if p.bone.is_some() && w > 1e-14 {
                let (e1, e2) = ([uv[1][0] - uv[0][0], uv[1][1] - uv[0][1]], [uv[2][0] - uv[0][0], uv[2][1] - uv[0][1]]);
                let det = e1[0] * e2[1] - e1[1] * e2[0];
                if det.abs() > 1e-12 {
                    let (d1, d2) = (dot(sub(pos[1], pos[0]), axis), dot(sub(pos[2], pos[0]), axis));
                    grads.push(([(d1 * e2[1] - d2 * e1[1]) / det, (e1[0] * d2 - e2[0] * d1) / det], w));
                }
            }
        }
        let px = (auv / a3.max(1e-18)).sqrt();
        let nominal = lay.texels_per_unit * p.density_factor;
        let (mut dev, mut dev95) = (None, None);
        if p.bone.is_some() {
            // Triangles where the bone lies along the surface (the gradient of the bone coordinate is
            // at least 0.2 of its largest possible length) say where the bone runs on the page.
            let strong: Vec<([f64; 2], f64)> = grads.iter().filter(|(g, _)| (g[0] * g[0] + g[1] * g[1]).sqrt() * px >= 0.2).cloned().collect();
            if !strong.is_empty() {
                // The same rule the layout turns pieces by: gradients summed, weighted by area.
                let sum = strong.iter().fold([0.0; 2], |a, (g, w)| [a[0] + g[0] * w, a[1] + g[1] * w]);
                dev = Some(axis_dev(sum));
                dev95 = Some(wpercentile(strong.iter().map(|(g, w)| (axis_dev(*g), *w)).collect(), 0.95));
            }
        }
        pieces.push(PieceMetric {
            piece: pi,
            label: p.label.clone(),
            triangles: p.tris.len(),
            pinned: p.pinned,
            density_factor: p.density_factor,
            px_per_unit: px,
            nominal_px_per_unit: nominal,
            density_err: if p.pinned { None } else { Some((px - nominal) / nominal.max(1e-18)) },
            aniso_p50: wpercentile(aniso.clone(), 0.5),
            aniso_p95: wpercentile(aniso.clone(), 0.95),
            aniso_max: aniso.iter().map(|x| x.0).fold(1.0, f64::max),
            bone: p.bone.and_then(|b| model.joint_names.get(b).cloned()),
            bone_dev_deg: dev,
            bone_dev_p95_deg: dev95,
            mirror_of: p.mirror_of,
        });
    }
    let mut mirrors = Vec::new();
    for mp in &lay.mirrors {
        let (pa, pb) = (&lay.pieces[mp.a], &lay.pieces[mp.b]);
        let flat_area = |p: &crate::pattern::Piece| -> f64 {
            let nm = &lay.meshes[p.mesh];
            p.tris
                .iter()
                .map(|&t| {
                    let uv = nm.tris[t as usize].map(|v| [nm.uv[v as usize][0] as f64, nm.uv[v as usize][1] as f64]);
                    0.5 * ((uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1]) - (uv[1][1] - uv[0][1]) * (uv[2][0] - uv[0][0])).abs()
                })
                .sum::<f64>()
        };
        let err = if mp.locked {
            let pairs: Vec<([f64; 2], [f64; 2])> = mp.map.iter().enumerate().map(|(j, &i)| (pb.uv[j], pa.uv[i as usize])).collect();
            let n = pairs.len().max(1) as f64;
            let mx = pairs.iter().map(|(b, a)| b[0] + a[0]).sum::<f64>() / n;
            let my = pairs.iter().map(|(b, a)| b[1] - a[1]).sum::<f64>() / n;
            Some(pairs.iter().map(|(b, a)| ((b[0] + a[0] - mx).powi(2) + (b[1] - a[1] - my).powi(2)).sqrt() * s).fold(0.0, f64::max))
        } else {
            None
        };
        mirrors.push(MirrorMetric {
            a: mp.a,
            b: mp.b,
            a_label: pa.label.clone(),
            b_label: pb.label.clone(),
            corners: mp.corners,
            matched: mp.matched,
            exact: mp.exact,
            locked: mp.locked,
            max_dist: mp.max_dist,
            max_err_px: err,
            area_ratio: flat_area(pb) / flat_area(pa).max(1e-18),
            note: mp.note,
        });
    }
    let unpaired = lay.pieces.iter().enumerate().filter(|(_, p)| !p.pinned && p.side != crate::pattern::Side::C && p.twin.is_none()).map(|(i, _)| i).collect();
    PatternMetrics { size: lay.size, pieces, mirrors, unpaired, aniso_limit: None }
}

fn pct(v: &[f64], q: f64) -> f64 {
    wpercentile(v.iter().map(|&x| (x, 1.0)).collect(), q)
}

impl PatternMetrics {
    pub fn density_err_max(&self) -> f64 {
        self.pieces.iter().filter_map(|p| p.density_err).map(f64::abs).fold(0.0, f64::max)
    }
    pub fn mirror_err_max_px(&self) -> f64 {
        self.mirrors.iter().filter_map(|m| m.max_err_px).fold(0.0, f64::max)
    }
    pub fn locked(&self) -> usize {
        self.mirrors.iter().filter(|m| m.locked).count()
    }
    pub fn aniso_p95_worst(&self) -> f64 {
        self.pieces.iter().filter(|p| !p.pinned).map(|p| p.aniso_p95).fold(1.0, f64::max)
    }
    /// Texel density within `MAX_DENSITY_ERR` of its nominal value on every piece, and every locked
    /// mirror pair within `MAX_MIRROR_ERR_PX` of an exact reflection.
    pub fn passed(&self) -> bool {
        self.density_err_max() <= MAX_DENSITY_ERR && self.mirror_err_max_px() <= MAX_MIRROR_ERR_PX && self.aniso_limit.map_or(true, |l| self.aniso_p95_worst() <= l)
    }

    pub fn to_json(&self) -> Value {
        let r = |x: f64, d: f64| (x * d).round() / d;
        let free: Vec<&PieceMetric> = self.pieces.iter().filter(|p| !p.pinned).collect();
        let p95s: Vec<f64> = free.iter().map(|p| p.aniso_p95).collect();
        let devs: Vec<f64> = free.iter().filter_map(|p| p.bone_dev_deg).collect();
        json!({
            "passed": self.passed(),
            "limits": { "max_density_error": MAX_DENSITY_ERR, "max_mirror_error_px": MAX_MIRROR_ERR_PX, "max_anisotropy_p95": self.aniso_limit },
            "summary": {
                "pieces": self.pieces.len(),
                "density_error_max": self.density_err_max(),
                "anisotropy_p95_worst": r(self.aniso_p95_worst(), 1000.0),
                "anisotropy_p95_median_of_pieces": r(pct(&p95s, 0.5), 1000.0),
                "bone_axis_deviation_deg_median": r(pct(&devs, 0.5), 100.0),
                "bone_axis_deviation_deg_max": r(devs.iter().cloned().fold(0.0, f64::max), 100.0),
                "pieces_with_bone_direction": devs.len(),
                "mirror_pairs_found": self.mirrors.len(),
                "mirror_pairs_corner_exact": self.mirrors.iter().filter(|m| m.exact).count(),
                "mirror_pairs_locked": self.locked(),
                "mirror_error_max_px": self.mirror_err_max_px(),
                "unpaired_left_right_pieces": self.unpaired.len(),
            },
            "mirror_pairs": self.mirrors.iter().map(|m| json!({
                "a": m.a, "b": m.b, "a_label": m.a_label, "b_label": m.b_label,
                "corners": m.corners, "matched": m.matched, "corner_exact": m.exact, "locked": m.locked,
                "max_mirror_distance_units": r(m.max_dist, 1e6), "max_reflection_error_px": m.max_err_px, "area_ratio_b_over_a": r(m.area_ratio, 1e6), "note": m.note,
            })).collect::<Vec<_>>(),
            "unpaired": self.unpaired.iter().map(|&i| json!({"piece": i, "label": self.pieces[i].label})).collect::<Vec<_>>(),
            "pieces": self.pieces.iter().map(|p| json!({
                "id": p.piece, "chart_id": p.piece + 1, "label": p.label, "triangles": p.triangles, "pinned_at_stock": p.pinned,
                "density_factor": p.density_factor, "px_per_unit": r(p.px_per_unit, 100.0), "nominal_px_per_unit": r(p.nominal_px_per_unit, 100.0),
                "density_error": p.density_err, "anisotropy_p50": r(p.aniso_p50, 1000.0), "anisotropy_p95": r(p.aniso_p95, 1000.0), "anisotropy_max": r(p.aniso_max, 100.0),
                "bone": p.bone, "bone_axis_deviation_deg": p.bone_dev_deg.map(|x| r(x, 100.0)), "bone_axis_deviation_p95_deg": p.bone_dev_p95_deg.map(|x| r(x, 100.0)),
                "mirror_of": p.mirror_of,
            })).collect::<Vec<_>>(),
        })
    }

    pub fn print(&self) {
        let r = |x: f64, d: f64| (x * d).round() / d;
        let free: Vec<&PieceMetric> = self.pieces.iter().filter(|p| !p.pinned).collect();
        let devs: Vec<f64> = free.iter().filter_map(|p| p.bone_dev_deg).collect();
        println!(
            "pattern checks: density error max {:.4}% (limit {:.1}%)   anisotropy p95 worst {:.3}{}   bone-axis deviation median {:.2} deg, max {:.2} deg over {} pieces",
            self.density_err_max() * 100.0,
            MAX_DENSITY_ERR * 100.0,
            self.aniso_p95_worst(),
            self.aniso_limit.map(|l| format!(" (limit {l})")).unwrap_or_default(),
            pct(&devs, 0.5),
            devs.iter().cloned().fold(0.0, f64::max),
            devs.len()
        );
        println!(
            "  mirror pairs found {}   corner-exact {}   locked as reflections {}   reflection error max {} px   unpaired L/R pieces {}",
            self.mirrors.len(),
            self.mirrors.iter().filter(|m| m.exact).count(),
            self.locked(),
            r(self.mirror_err_max_px(), 1e6),
            self.unpaired.len()
        );
        println!("  {}", if self.passed() { "pattern checks PASSED" } else { "pattern checks FAILED" });
    }
}

/// Chart, triangle and bone ID maps at `gs` x `gs`: 24-bit ids packed as R + 256 G + 65536 B, alpha
/// 255 where a triangle covers the texel and 0 elsewhere (id 0 = nothing).
pub struct IdMaps {
    pub size: usize,
    pub chart: Vec<u8>,
    pub triangle: Vec<u8>,
    pub bone: Vec<u8>,
    /// A readable color per chart id (for people; the ID map is for tools).
    pub chart_preview: Vec<u8>,
    pub covered: usize,
}

/// First global triangle id (minus one) of every mesh: triangle id = offset + triangle index + 1.
pub fn triangle_offsets(model: &Model) -> Vec<usize> {
    let mut o = Vec::with_capacity(model.meshes.len());
    let mut n = 0;
    for m in &model.meshes {
        o.push(n);
        n += m.tris.len();
    }
    o
}

fn put(img: &mut [u8], i: usize, id: u32) {
    img[i * 4] = (id & 255) as u8;
    img[i * 4 + 1] = ((id >> 8) & 255) as u8;
    img[i * 4 + 2] = ((id >> 16) & 255) as u8;
    img[i * 4 + 3] = 255;
}

/// Chart id = piece index + 1 (as in `legend`), triangle id = global triangle index + 1 (meshes in
/// order, see `triangle_offsets`), bone id = joint index + 1 of the bone with the most skin weight
/// at the texel (weights blended across the triangle). Only front triangles are drawn.
pub fn id_maps(model: &Model, lay: &Layout, atlas: &Atlas, gs: usize) -> IdMaps {
    let n = gs * gs * 4;
    let (mut chart, mut triangle, mut bone, mut preview) = (vec![0u8; n], vec![0u8; n], vec![0u8; n], vec![0u8; n]);
    let offs = triangle_offsets(model);
    let s = gs as f64;
    let mut covered = 0;
    let mut is_front: Vec<Vec<bool>> = model.meshes.iter().map(|m| vec![false; m.tris.len()]).collect();
    for p in &lay.pieces {
        for &t in &p.tris {
            is_front[p.mesh][t as usize] = true;
        }
    }
    for &mi in &atlas.meshes {
        let m = &model.meshes[mi];
        let nm = &lay.meshes[mi];
        for (t, tri) in m.tris.iter().enumerate() {
            if !is_front[mi][t] {
                continue;
            }
            let new = nm.tris[t].map(|v| [nm.uv[v as usize][0] as f64 * s, nm.uv[v as usize][1] as f64 * s]);
            let piece = lay.piece_of[mi][t] as u32 + 1;
            let col = seam_color(piece as usize);
            let tid = (offs[mi] + t + 1) as u32;
            raster_bary(new, gs, gs, 0.0, |x, y, l| {
                let i = y * gs + x;
                if chart[i * 4 + 3] == 0 {
                    covered += 1;
                }
                put(&mut chart, i, piece);
                put(&mut triangle, i, tid);
                preview[i * 4..i * 4 + 3].copy_from_slice(&col);
                preview[i * 4 + 3] = 255;
                let mut acc: [(u16, f64); 12] = [(0, 0.0); 12];
                let mut k = 0;
                for c in 0..3 {
                    let v = tri[c] as usize;
                    for j in 0..4 {
                        let w = m.weights[v][j] as f64 * l[c];
                        if w <= 0.0 {
                            continue;
                        }
                        match acc[..k].iter_mut().find(|e| e.0 == m.joints[v][j]) {
                            Some(e) => e.1 += w,
                            None => {
                                acc[k] = (m.joints[v][j], w);
                                k += 1;
                            }
                        }
                    }
                }
                let best = acc[..k].iter().fold((0u16, -1.0f64), |b, e| if e.1 > b.1 || (e.1 == b.1 && e.0 < b.0) { (e.0, e.1) } else { b });
                put(&mut bone, i, best.0 as u32 + 1);
            });
        }
    }
    IdMaps { size: gs, chart, triangle, bone, chart_preview: preview, covered }
}
