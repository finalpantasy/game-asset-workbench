//! Animation sweep for a strap that hangs on the waist: poses every clip at several normalized
//! times, measures how the strap sits on the waist, and renders close-ups from all sides.
//!
//! Measures per pose (world units; the strap is about 0.67 long and 0.12 wide):
//! - slide: how far each strap vertex that lies on the waist at rest (within `reach`) moves off
//!   the waist point it sat over, in the waist triangle's own frame. Paint that crosses from the
//!   waist onto the strap steps by this much.
//! - ledge: signed distance of the strap's top edge from the waist surface (> 0 = the edge stands in
//!   front of the band and shows as a ledge; < 0 = tucked behind the band).
//! - cut: strap vertices outside the body at rest that are inside it in the pose (winding number
//!   of the body surface > 0.5), so the body shows through the strap.
//! - tip in body: the same for vertices in the tip zone.
//! - on screen, per view: strap pixels, top-edge vertices that are the nearest surface, and the
//!   tip pixels the body covers. A tip hidden by a raised knee in one view is normal; the tip
//!   counts as gone only when the body covers it in all three front views.
//! Slide counts only strap vertices that are visible (nearest surface) in at least one close-up.
use crate::geom::{self, V3};
use crate::render::{self, Camera};
use crate::shells::{self, Layers, Shell};
use crate::{Model, Texture};

/// Length of `d` with its component along `n` removed: how far a point slid along the surface.
fn along_surface(d: V3, n: V3) -> f32 {
    let nn = geom::dot(n, n).max(1e-12);
    geom::len(geom::sub(d, geom::scale(n, geom::dot(d, n) / nn)))
}

#[derive(Clone, Debug)]
pub struct Rules {
    pub reach: f32,
    /// Largest slide allowed in the knot zone; 0 = one texel of the stock 512 texture.
    pub slide_max: f32,
    /// Top share of the strap (by UV) where its paint meets the waist's knot paint.
    pub knot_share: f32,
    pub ledge_max: f32,
    pub cut_depth: f32,
    pub tip_share: f32,
    pub tip_visible_min: f32,
}

impl Default for Rules {
    fn default() -> Rules {
        Rules { reach: 0.03, slide_max: 0.0, knot_share: 0.15, ledge_max: 0.0, cut_depth: 0.0005, tip_share: 0.12, tip_visible_min: 0.5 }
    }
}

/// A strap vertex pinned to a waist triangle at rest.
struct Anchor {
    v: usize,
    tri: usize,
    bary: [f32; 3],
    /// Offset from the waist point in the triangle's frame (edge 1, edge 2, normal).
    local: V3,
    gap: f32,
}

pub struct Setup {
    pub strap: usize,
    pub body: usize,
    anchors: Vec<Anchor>,
    /// Strap vertices on the top edge (root end of the strap's UV long axis).
    pub top: Vec<usize>,
    /// Strap vertices in the lowest `tip_share` of the strap (along its UV long axis).
    pub tip: Vec<usize>,
    /// Triangles of the strap shell in the tip zone.
    pub tip_tris: Vec<usize>,
    /// Body triangles whose winding disagrees with their NORMAL attribute (flip their normal).
    flip: Vec<bool>,
    /// Strap vertices in front of the body at rest (gap > 0.001), with rest gap.
    outside: Vec<(usize, f32)>,
    /// A point on the body near the strap's middle, tracked through poses for the close-up camera.
    pub track: (usize, [f32; 3], V3),
    /// Range of the UV long-axis coordinate over the strap (top, tip), for 'how far down'.
    pub vrange: (f32, f32),
    /// The strap's UV long axis, pointing from the root to the tip (any chart rotation or mirroring).
    pub axis: [f32; 2],
    /// World size of one texel of the stock 512 texture on the strap.
    pub texel: f32,
}

/// World size of one texel of the stock 512 texture on the stock strap (fixed, so limits do not
/// change when a repaint moves the strap to a denser atlas).
pub const STOCK_TEXEL: f32 = 0.00585;

fn tri_frame(p: [V3; 3]) -> (V3, V3, V3) {
    let e1 = geom::sub(p[1], p[0]);
    let e2 = geom::sub(p[2], p[0]);
    (e1, e2, geom::norm(geom::cross(e1, e2)))
}

fn at_bary(p: [V3; 3], b: [f32; 3]) -> V3 {
    std::array::from_fn(|k| b[0] * p[0][k] + b[1] * p[1][k] + b[2] * p[2][k])
}

/// Solves x*e1 + y*e2 + z*n = d.
fn local_coords(e1: V3, e2: V3, n: V3, d: V3) -> V3 {
    let det = geom::dot(e1, geom::cross(e2, n));
    if det.abs() < 1e-14 {
        return [0.0, 0.0, geom::dot(d, n)];
    }
    [geom::dot(d, geom::cross(e2, n)) / det, geom::dot(e1, geom::cross(d, n)) / det, geom::dot(e1, geom::cross(e2, d)) / det]
}

impl Setup {
    pub fn new(model: &Model, sh: &[Shell], strap: usize, body: usize, rules: &Rules) -> Result<Setup, String> {
        let (s, b) = (&sh[strap], &sh[body]);
        if s.mesh != b.mesh {
            return Err("strap and waist must be in the same mesh".into());
        }
        let m = &model.meshes[s.mesh];
        let rest: Vec<Vec<V3>> = model.meshes.iter().map(|m| m.pos.clone()).collect();
        let attr_n = crate::strap::normals(model, s.mesh).ok_or("mesh has no NORMAL attribute")?;
        let flip: Vec<bool> = m
            .tris
            .iter()
            .map(|t| {
                let c = t.map(|v| v as usize);
                let g = geom::tri_normal(m.pos[c[0]], m.pos[c[1]], m.pos[c[2]]);
                let a = geom::add(geom::add(attr_n[c[0]], attr_n[c[1]]), attr_n[c[2]]);
                geom::dot(g, a) < 0.0
            })
            .collect();
        let corners = |t: usize, pose: &[V3]| m.tris[t].map(|v| pose[v as usize]);
        let mut anchors = Vec::new();
        let mut outside = Vec::new();
        for &v in &s.verts {
            let (d, t, bary) = shells::closest_in_shell(model, &rest, b, m.pos[v]);
            let p3 = corners(t, &m.pos);
            let (e1, e2, mut n) = tri_frame(p3);
            if flip[t] {
                n = geom::scale(n, -1.0);
            }
            let q = at_bary(p3, bary);
            let gap = geom::dot(geom::sub(m.pos[v], q), n);
            if d <= rules.reach {
                anchors.push(Anchor { v, tri: t, bary, local: local_coords(e1, e2, n, geom::sub(m.pos[v], q)), gap });
            }
            let _ = gap;
            if winding(model, b, &flip, &rest, m.pos[v]) < 0.5 {
                outside.push((v, gap));
            }
        }
        // The strap's UV long axis (principal axis of its UVs), pointing the way the strap hangs
        // (down in the rest pose, the model being +Y up).
        let n = s.verts.len().max(1) as f32;
        let mu: [f32; 2] = std::array::from_fn(|k| s.verts.iter().map(|&v| m.uv[v][k]).sum::<f32>() / n);
        let (mut sxx, mut sxy, mut syy, mut sxh, mut syh) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let my = s.verts.iter().map(|&v| m.pos[v][1]).sum::<f32>() / n;
        for &v in &s.verts {
            let (x, y, h) = (m.uv[v][0] - mu[0], m.uv[v][1] - mu[1], my - m.pos[v][1]);
            sxx += x * x;
            sxy += x * y;
            syy += y * y;
            sxh += x * h;
            syh += y * h;
        }
        let ang = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        let mut axis = [ang.cos(), ang.sin()];
        if axis[0] * sxh + axis[1] * syh < 0.0 {
            axis = [-axis[0], -axis[1]];
        }
        let along = |v: usize| m.uv[v][0] * axis[0] + m.uv[v][1] * axis[1];
        // World size of a stock texel: median of sqrt(world area / UV area) over the strap.
        let mut ratios: Vec<f32> = s
            .tris
            .iter()
            .filter_map(|&t| {
                let c = m.tris[t].map(|v| v as usize);
                let wa = geom::len(geom::cross(geom::sub(m.pos[c[1]], m.pos[c[0]]), geom::sub(m.pos[c[2]], m.pos[c[0]])));
                let (a, b, cc) = (m.uv[c[0]], m.uv[c[1]], m.uv[c[2]]);
                let ua = ((b[0] - a[0]) * (cc[1] - a[1]) - (b[1] - a[1]) * (cc[0] - a[0])).abs();
                (ua > 1e-10 && wa > 1e-10).then(|| (wa / ua).sqrt())
            })
            .collect();
        ratios.sort_by(f32::total_cmp);
        let ratio = ratios.get(ratios.len() / 2).copied().unwrap_or(1.0).max(1e-6);
        let texel = STOCK_TEXEL;
        let uv_texel = texel / ratio;
        let vmin = s.verts.iter().map(|&v| along(v)).fold(f32::MAX, f32::min);
        let vmax = s.verts.iter().map(|&v| along(v)).fold(f32::MIN, f32::max);
        let top: Vec<usize> = s.verts.iter().copied().filter(|&v| along(v) <= vmin + 0.75 * uv_texel).collect();
        let tip_line = vmax - rules.tip_share * (vmax - vmin);
        let tip: Vec<usize> = s.verts.iter().copied().filter(|&v| along(v) >= tip_line).collect();
        let mut tip_tris: Vec<usize> = s.tris.iter().copied().filter(|&t| m.tris[t].iter().all(|&v| along(v as usize) >= tip_line - uv_texel)).collect();
        if tip_tris.is_empty() {
            // Coarse strap (the stock card): the triangles that touch the tip zone.
            tip_tris = s.tris.iter().copied().filter(|&t| m.tris[t].iter().any(|&v| along(v as usize) >= tip_line)).collect();
        }
        // Track the waist point under the strap's middle.
        let ctr = geom::scale(s.verts.iter().fold([0.0; 3], |acc, &v| geom::add(acc, m.pos[v])), 1.0 / s.verts.len() as f32);
        let (_, t, bary) = shells::closest_in_shell(model, &rest, b, ctr);
        let q = at_bary(corners(t, &m.pos), bary);
        Ok(Setup { strap, body, anchors, top, tip, tip_tris, flip, outside, track: (t, bary, geom::sub(ctr, q)), vrange: (vmin, vmax), axis, texel })
    }

    /// The tracked point in a pose (camera target).
    pub fn target(&self, model: &Model, sh: &[Shell], pose: &[Vec<V3>]) -> V3 {
        let m = &model.meshes[sh[self.body].mesh];
        let (t, bary, off) = self.track;
        geom::add(at_bary(m.tris[t].map(|v| pose[sh[self.body].mesh][v as usize]), bary), off)
    }

    /// How far down the strap a vertex is (0 = top edge, 1 = tip), along the UV long axis.
    pub fn down(&self, model: &Model, sh: &[Shell], v: usize) -> f32 {
        let t = model.meshes[sh[self.strap].mesh].uv[v];
        let uv = t[0] * self.axis[0] + t[1] * self.axis[1];
        (uv - self.vrange.0) / (self.vrange.1 - self.vrange.0).max(1e-9)
    }
}

#[derive(Clone, Debug, Default)]
pub struct ViewStat {
    pub name: String,
    pub strap_px: usize,
    pub top_shown: usize,
    pub tip_px: usize,
    pub tip_hidden_by_body: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Sample {
    pub clip: String,
    pub time_norm: f32,
    pub time_s: f32,
    pub slide_max: f32,
    pub slide_p95: f32,
    pub ledge_max: f32,
    pub cut: usize,
    pub cut_depth: f32,
    pub tip_in_body: usize,
    /// How far down the strap (0 top, 1 tip) the worst slide and the deepest cut are.
    pub slide_at: f32,
    pub slide_v: usize,
    /// Largest slide of visible strap points below the knot zone (strap over fabric; reported only).
    pub fabric_slide: f32,
    pub slide_limit: f32,
    /// Strap vertices inside the body, and the joint that leads the body surface at the deepest one.
    pub cut_verts: Vec<usize>,
    pub cut_joint: Option<usize>,
    pub cut_at: f32,
    /// Strap vertices below the knot zone that a thigh passes into (raised knee); reported, not flagged.
    pub thigh_clip: usize,
    pub thigh_clip_depth: f32,
    pub thigh_clip_at: f32,
    pub thigh_clip_joint: Option<usize>,
    /// Reported findings that are not attachment failures.
    pub notes: Vec<String>,
    /// Per close-up, from alpha-aware layers (transparent texels are not surfaces).
    pub views: Vec<ViewStat>,
    /// Geometry tests (positions only): STEP, LEDGE, CUT, TIP IN BODY.
    pub geo_flags: Vec<String>,
    /// Visible-surface tests on alpha-aware layers: EDGE SHOWS, TIP HIDDEN.
    pub vis_flags: Vec<String>,
    /// Both lists, geometry first.
    pub flags: Vec<String>,
}

/// Generalized winding number of the body surface at p (about 1 inside, 0 outside), with
/// triangles oriented by their NORMAL attribute.
fn winding(model: &Model, b: &Shell, flip: &[bool], pose: &[Vec<V3>], p: V3) -> f32 {
    let m = &model.meshes[b.mesh];
    let mut sum = 0.0f64;
    for &t in &b.tris {
        let c = m.tris[t].map(|v| geom::sub(pose[b.mesh][v as usize], p));
        let (la, lb, lc) = (geom::len(c[0]) as f64, geom::len(c[1]) as f64, geom::len(c[2]) as f64);
        let det = geom::dot(c[0], geom::cross(c[1], c[2])) as f64;
        let den = la * lb * lc + geom::dot(c[0], c[1]) as f64 * lc + geom::dot(c[1], c[2]) as f64 * la + geom::dot(c[2], c[0]) as f64 * lb;
        let om = 2.0 * det.atan2(den);
        sum += if flip[t] { -om } else { om };
    }
    (sum / (4.0 * std::f64::consts::PI)).abs() as f32
}

/// Signed distance of p from the body (positive = in front of / outside its surface).
fn signed_gap(model: &Model, sh: &[Shell], setup: &Setup, pose: &[Vec<V3>], p: V3) -> f32 {
    let b = &sh[setup.body];
    let (_, t, bary) = shells::closest_in_shell(model, pose, b, p);
    let m = &model.meshes[b.mesh];
    let p3 = m.tris[t].map(|v| pose[b.mesh][v as usize]);
    let mut n = geom::tri_normal(p3[0], p3[1], p3[2]);
    if setup.flip[t] {
        n = geom::scale(n, -1.0);
    }
    geom::dot(geom::sub(p, at_bary(p3, bary)), n)
}

/// Close-up views around the waist: name and yaw in degrees (0 = front, -90 = character's right).
pub const CLOSE_VIEWS: [(&str, f32); 6] = [("front", 0.0), ("3/4 right", -40.0), ("right", -90.0), ("back", 180.0), ("left", 90.0), ("3/4 left", 40.0)];

/// Measures one pose. `layers_for(view)` gives the shell layers for that close-up.
/// `views`: depth-only layers (geometry); `vis`: the same cameras with alpha-aware layers.
pub fn measure(model: &Model, sh: &[Shell], setup: &Setup, rules: &Rules, pose: &[Vec<V3>], views: &[(String, Layers, Camera)], vis: &[(String, Layers, Camera)]) -> Sample {
    let ms = sh[setup.strap].mesh;
    let m = &model.meshes[ms];
    let mut s = Sample::default();
    // Strap vertices that are the nearest surface in at least one close-up.
    let mut visible = vec![false; m.pos.len()];
    for (_, l, cam) in views {
        for &v in &sh[setup.strap].verts {
            let pr = shells::project(cam, l.w, l.h, pose[ms][v]);
            let (x, y) = (pr[0].floor() as i64, pr[1].floor() as i64);
            if x < 0 || y < 0 || x >= l.w as i64 || y >= l.h as i64 {
                continue;
            }
            let i = y as usize * l.w + x as usize;
            if l.s1[i] == setup.strap && pr[2] >= l.d1[i] - 0.002 {
                visible[v] = true;
            }
        }
    }
    // Slide of anchored vertices that show: knot zone (flagged) and over fabric (reported).
    let mut fabric: Vec<f32> = Vec::new();
    let mut slides: Vec<(f32, usize)> = setup
        .anchors
        .iter()
        .filter(|a| visible[a.v])
        .filter(|a| {
            let knot = setup.down(model, sh, a.v) <= rules.knot_share;
            knot || {
                let p3 = m.tris[a.tri].map(|v| pose[ms][v as usize]);
                let (e1, e2, mut n) = tri_frame(p3);
                if setup.flip[a.tri] {
                    n = geom::scale(n, -1.0);
                }
                let want = geom::add(at_bary(p3, a.bary), geom::add(geom::add(geom::scale(e1, a.local[0]), geom::scale(e2, a.local[1])), geom::scale(n, a.local[2])));
                fabric.push(along_surface(geom::sub(pose[ms][a.v], want), n));
                false
            }
        })
        .map(|a| {
            let p3 = m.tris[a.tri].map(|v| pose[ms][v as usize]);
            let (e1, e2, mut n) = tri_frame(p3);
            if setup.flip[a.tri] {
                n = geom::scale(n, -1.0);
            }
            let want = geom::add(at_bary(p3, a.bary), geom::add(geom::add(geom::scale(e1, a.local[0]), geom::scale(e2, a.local[1])), geom::scale(n, a.local[2])));
            let _ = a.gap;
            (along_surface(geom::sub(pose[ms][a.v], want), n), a.v)
        })
        .collect();
    slides.sort_by(|a, b| a.0.total_cmp(&b.0));
    if let Some(&(d, v)) = slides.last() {
        s.slide_max = d;
        s.slide_at = setup.down(model, sh, v);
        s.slide_v = v;
    }
    s.slide_p95 = slides.get(((slides.len() as f32 * 0.95) as usize).min(slides.len().saturating_sub(1))).map_or(0.0, |x| x.0);
    s.ledge_max = setup.top.iter().map(|&v| signed_gap(model, sh, setup, pose, pose[ms][v])).fold(f32::MIN, f32::max);
    for &(v, _) in &setup.outside {
        if winding(model, &sh[setup.body], &setup.flip, pose, pose[ms][v]) < 0.5 {
            continue;
        }
        let g = signed_gap(model, sh, setup, pose, pose[ms][v]).min(0.0);
        if g <= -rules.cut_depth {
            // Below the knot zone and into a thigh: thigh clipping, not the waist attachment.
            let (_, t, bary) = shells::closest_in_shell(model, pose, &sh[setup.body], pose[ms][v]);
            let lead = shells::weights_at(m, t, bary).into_iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|e| e.0);
            let name = lead.and_then(|j| model.joint_names.get(j)).map(|n| n.to_lowercase()).unwrap_or_default();
            let thigh = name.contains("thigh") || (name.contains("hip") && !name.contains("hips"));
            let down = setup.down(model, sh, v);
            if thigh && down > rules.knot_share {
                s.thigh_clip += 1;
                if -g > s.thigh_clip_depth {
                    s.thigh_clip_depth = -g;
                    s.thigh_clip_at = down;
                    s.thigh_clip_joint = lead;
                }
                continue;
            }
            s.cut_verts.push(v);
            if s.cut_joint.is_none() || (!setup.tip.contains(&v) && -g > s.cut_depth) {
                let (_, t, bary) = shells::closest_in_shell(model, pose, &sh[setup.body], pose[ms][v]);
                s.cut_joint = shells::weights_at(m, t, bary).into_iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|e| e.0);
            }
            if setup.tip.contains(&v) {
                s.tip_in_body += 1;
            } else {
                s.cut += 1;
                if -g > s.cut_depth {
                    s.cut_depth = -g;
                    s.cut_at = setup.down(model, sh, v);
                }
            }
        }
    }
    for (name, l, cam) in vis {
        let mut vs = ViewStat { name: name.clone(), ..Default::default() };
        vs.strap_px = l.s1.iter().filter(|&&x| x == setup.strap).count();
        for &v in &setup.top {
            let pr = shells::project(cam, l.w, l.h, pose[ms][v]);
            let (x, y) = (pr[0].floor() as i64, pr[1].floor() as i64);
            if x < 0 || y < 0 || x >= l.w as i64 || y >= l.h as i64 {
                continue;
            }
            let i = y as usize * l.w + x as usize;
            if l.s1[i] == setup.strap && pr[2] >= l.d1[i] - 0.002 {
                vs.top_shown += 1;
            }
        }
        // Tip pixels: rasterize the tip triangles alone and see who is nearest there.
        let mut mask = vec![false; l.w * l.h];
        for &t in &setup.tip_tris {
            let p = m.tris[t].map(|v| shells::project(cam, l.w, l.h, pose[ms][v as usize]));
            fill(&mut mask, l.w, l.h, [[p[0][0], p[0][1]], [p[1][0], p[1][1]], [p[2][0], p[2][1]]]);
        }
        for (i, &on) in mask.iter().enumerate() {
            if !on {
                continue;
            }
            vs.tip_px += 1;
            if l.s1[i] == setup.body {
                vs.tip_hidden_by_body += 1;
            }
        }
        s.views.push(vs);
    }
    s.fabric_slide = fabric.iter().copied().fold(0.0, f32::max);
    s.slide_limit = if rules.slide_max > 0.0 { rules.slide_max } else { setup.texel };
    if s.slide_max > s.slide_limit {
        s.geo_flags.push(format!("STEP slide {:.3} at {:.0}% down (v{})", s.slide_max, 100.0 * s.slide_at, s.slide_v));
    }
    if s.ledge_max > rules.ledge_max {
        s.geo_flags.push(format!("LEDGE top edge {:+.3} in front of band", s.ledge_max));
    }
    if s.cut > 0 {
        s.geo_flags.push(format!("CUT {} verts behind waist (to {:.3} at {:.0}% down)", s.cut, s.cut_depth, 100.0 * s.cut_at));
    }
    if s.tip_in_body > 0 {
        s.geo_flags.push(format!("TIP IN BODY {} verts", s.tip_in_body));
    }
    if s.thigh_clip > 0 {
        let jn = s.thigh_clip_joint.and_then(|j| model.joint_names.get(j)).cloned().unwrap_or_default();
        s.notes.push(format!("THIGH CLIP {} verts (to {:.3} at {:.0}% down, {})", s.thigh_clip, s.thigh_clip_depth, 100.0 * s.thigh_clip_at, jn));
    }
    let fronts: Vec<&ViewStat> = s.views.iter().filter(|v| v.name.starts_with("front") || v.name.starts_with("3/4")).collect();
    let hidden = |vs: &ViewStat| vs.tip_px == 0 || (vs.tip_hidden_by_body as f32) > (1.0 - rules.tip_visible_min) * vs.tip_px as f32;
    if !fronts.is_empty() && fronts.iter().all(|v| hidden(v)) {
        s.vis_flags.push(format!(
            "TIP HIDDEN by body in every front view ({})",
            fronts.iter().map(|v| format!("{} {}/{}", v.name, v.tip_hidden_by_body, v.tip_px)).collect::<Vec<_>>().join(", ")
        ));
    }
    for vs in &s.views {
        if vs.top_shown > 0 {
            s.vis_flags.push(format!("EDGE SHOWS in {} ({} top verts)", vs.name, vs.top_shown));
        }
    }
    s.flags = s.geo_flags.iter().chain(&s.vis_flags).cloned().collect();
    s
}

pub fn fill(mask: &mut [bool], w: usize, h: usize, p: [[f32; 2]; 3]) {
    let [a, b, c] = p;
    let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if area.abs() < 1e-9 {
        return;
    }
    let x0 = a[0].min(b[0]).min(c[0]).floor().max(0.0) as usize;
    let x1 = (a[0].max(b[0]).max(c[0]).ceil().max(0.0) as usize).min(w);
    let y0 = a[1].min(b[1]).min(c[1]).floor().max(0.0) as usize;
    let y1 = (a[1].max(b[1]).max(c[1]).ceil().max(0.0) as usize).min(h);
    for y in y0..y1 {
        for x in x0..x1 {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let w0 = ((b[0] - px) * (c[1] - py) - (b[1] - py) * (c[0] - px)) / area;
            let w1 = ((c[0] - px) * (a[1] - py) - (c[1] - py) * (a[0] - px)) / area;
            if w0 >= 0.0 && w1 >= 0.0 && w0 + w1 <= 1.0 {
                mask[y * w + x] = true;
            }
        }
    }
}

/// Textured close-up of a pose (unlit, like the game), RGBA.
pub fn picture(model: &Model, textures: &[Option<Texture>], pose: &[Vec<V3>], cam: Camera, size: usize, highlight: Option<(usize, &[usize])>) -> Vec<u8> {
    let o = render::Options { cam, w: size, h: size, background: ([58, 62, 72], [34, 37, 44]), filter: render::Filter::Smooth, positions: Some(pose) };
    let tint = |mi: usize, ti: usize| -> Option<[u8; 3]> {
        match highlight {
            Some((hm, tris)) if hm == mi && tris.binary_search(&ti).is_ok() => Some([255, 60, 200]),
            _ => None,
        }
    };
    render::render(model, textures, &o, &tint).rgba
}
