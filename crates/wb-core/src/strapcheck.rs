//! Strap attachment checks, written as plain functions so the CLI (`strap-check`) and the Mark &
//! fix window can both call them:
//! - `ATTACH_VIEWS` / `attach_cam`: close-ups of the strap root from the front, both sides and
//!   below, centered on a point (the knot).
//! - `seam_report`: open edges, cracks, intersections and texture (UV) seams near the strap in a
//!   pose, each tied to the triangles that cause it, so a thin break can be traced before anything
//!   is changed.
//! - `overlay_picture`: a close-up with wireframe, depth shading and the flagged triangles tinted.
//! - `protected_violations`: points of a reference strap (the free-hanging part, the hardware)
//!   that a new strap no longer covers, and stand-off from the body lost, per pose.
use crate::edit::{self, Topo};
use crate::render::{self, Camera};
use crate::shells::{self, Shell};
use crate::{geom, Model, Texture};

type V3 = [f32; 3];

/// One close-up direction: yaw from the front (degrees, the `shells::ortho_cam` sense) and pitch
/// (degrees; negative looks up from below).
#[derive(Clone, Copy, Debug)]
pub struct AttachView {
    pub name: &'static str,
    pub yaw_deg: f32,
    pub pitch_deg: f32,
}

/// The views the attachment is judged from: front, both sides, both 3/4 views and the underside.
pub const ATTACH_VIEWS: [AttachView; 6] = [
    AttachView { name: "front", yaw_deg: 0.0, pitch_deg: 0.0 },
    AttachView { name: "3/4 right", yaw_deg: -40.0, pitch_deg: 0.0 },
    AttachView { name: "right side", yaw_deg: -90.0, pitch_deg: 0.0 },
    AttachView { name: "left side", yaw_deg: 90.0, pitch_deg: 0.0 },
    AttachView { name: "3/4 left", yaw_deg: 40.0, pitch_deg: 0.0 },
    AttachView { name: "underside", yaw_deg: 0.0, pitch_deg: -65.0 },
];

/// An orthographic close-up of `center`, `half` world units high.
pub fn attach_cam(center: V3, v: &AttachView, half: f32) -> Camera {
    let mut c = shells::ortho_cam(center, v.yaw_deg, half);
    c.pitch = v.pitch_deg.to_radians();
    c
}

/// Welded group positions in a pose (each group takes its first member's posed position).
pub fn group_pose(topo: &Topo, pose: &[Vec<V3>]) -> Vec<V3> {
    topo.members.iter().map(|m| pose[m[0].0 as usize][m[0].1 as usize]).collect()
}

/// A texture seam: a welded edge whose faces use different UVs, with how much the painted color
/// jumps across it (0..1, mean of the channel differences at the edge's middle).
#[derive(Clone, Debug)]
pub struct UvSeam {
    pub edge: (u32, u32),
    pub tris: Vec<u32>,
    pub color_jump: f32,
}

#[derive(Clone, Debug, Default)]
pub struct SeamReport {
    pub lo: V3,
    pub hi: V3,
    /// Triangles (global ids) with a corner inside the region.
    pub region_tris: Vec<u32>,
    /// Surface borders inside the region (welded group pairs).
    pub open_edges: Vec<(u32, u32)>,
    /// Open edges running along another open edge without being joined to it.
    pub cracks: Vec<((u32, u32), (u32, u32))>,
    /// Triangle pairs passing through each other, at least one in the region.
    pub intersections: Vec<(u32, u32)>,
    pub uv_seams: Vec<UvSeam>,
    /// Every triangle tied to one of the above, sorted.
    pub flagged_tris: Vec<u32>,
}

fn inside(p: V3, lo: V3, hi: V3) -> bool {
    (0..3).all(|k| p[k] >= lo[k] && p[k] <= hi[k])
}

fn texel(t: &Texture, uv: [f32; 2]) -> [f32; 3] {
    let x = ((uv[0] * t.w as f32) as i64).clamp(0, t.w as i64 - 1) as usize;
    let y = ((uv[1] * t.h as f32) as i64).clamp(0, t.h as i64 - 1) as usize;
    let o = (y * t.w + x) * 4;
    [t.rgba[o] as f32 / 255.0, t.rgba[o + 1] as f32 / 255.0, t.rgba[o + 2] as f32 / 255.0]
}

/// Everything that can show as a thin line or break near the strap in one pose, inside the box
/// `lo..hi` (posed positions). `crack_tol` is how close two open edges must run to count as a crack.
pub fn seam_report(model: &Model, textures: &[Option<Texture>], topo: &Topo, pose: &[Vec<V3>], lo: V3, hi: V3, crack_tol: f32) -> SeamReport {
    let gp = group_pose(topo, pose);
    let in_g = |g: u32| inside(gp[g as usize], lo, hi);
    let region_tris: Vec<u32> = (0..topo.tri_groups.len() as u32).filter(|&t| topo.tri_groups[t as usize].iter().any(|&g| in_g(g))).collect();
    let open_edges: Vec<(u32, u32)> = edit::open_edges(topo).into_iter().filter(|&(a, b)| in_g(a) || in_g(b)).collect();
    let cracks: Vec<((u32, u32), (u32, u32))> = edit::cracks(topo, &gp, crack_tol).into_iter().filter(|(e, f)| in_g(e.0) || in_g(e.1) || in_g(f.0) || in_g(f.1)).collect();
    let in_t: std::collections::HashSet<u32> = region_tris.iter().copied().collect();
    let intersections: Vec<(u32, u32)> = edit::intersections(topo, &gp).into_iter().filter(|(a, b)| in_t.contains(a) || in_t.contains(b)).collect();

    // Texture seams: for each welded edge in the region, the UVs each triangle gives its two ends.
    let corner_uv = |t: u32, g: u32| -> Option<(usize, [f32; 2])> {
        let (mi, ti) = topo.tri_loc[t as usize];
        let m = &model.meshes[mi];
        let k = topo.tri_groups[t as usize].iter().position(|&x| x == g)?;
        Some((mi, m.uv[m.tris[ti][k] as usize]))
    };
    let mut edges: std::collections::BTreeMap<(u32, u32), Vec<u32>> = std::collections::BTreeMap::new();
    for &t in &region_tris {
        let g = topo.tri_groups[t as usize];
        for k in 0..3 {
            let (a, b) = (g[k], g[(k + 1) % 3]);
            edges.entry((a.min(b), a.max(b))).or_default().push(t);
        }
    }
    let mut uv_seams = Vec::new();
    for (&(a, b), ts) in &edges {
        if ts.len() < 2 {
            continue;
        }
        let uvs: Vec<(u32, usize, [f32; 2], [f32; 2])> = ts.iter().filter_map(|&t| Some((t, corner_uv(t, a)?.0, corner_uv(t, a)?.1, corner_uv(t, b)?.1))).collect();
        let far = |p: [f32; 2], q: [f32; 2]| (p[0] - q[0]).abs() + (p[1] - q[1]).abs() > 1e-4;
        let split = uvs.iter().any(|x| uvs.iter().any(|y| x.1 != y.1 || far(x.2, y.2) || far(x.3, y.3)));
        if !split {
            continue;
        }
        // Color on each side at the edge's middle (nudged into each triangle a little).
        let colors: Vec<[f32; 3]> = uvs
            .iter()
            .filter_map(|&(t, mi, ua, ub)| {
                let tex = model.meshes[mi].image.and_then(|i| textures.get(i)).and_then(|x| x.as_ref())?;
                let (_, ti) = topo.tri_loc[t as usize];
                let m = &model.meshes[mi];
                let c = m.tris[ti].map(|v| m.uv[v as usize]);
                let cen = [(c[0][0] + c[1][0] + c[2][0]) / 3.0, (c[0][1] + c[1][1] + c[2][1]) / 3.0];
                let mid = [0.5 * (ua[0] + ub[0]), 0.5 * (ua[1] + ub[1])];
                Some(texel(tex, [mid[0] + 0.1 * (cen[0] - mid[0]), mid[1] + 0.1 * (cen[1] - mid[1])]))
            })
            .collect();
        let mut jump = 0.0f32;
        for x in &colors {
            for y in &colors {
                jump = jump.max((0..3).map(|k| (x[k] - y[k]).abs()).sum::<f32>() / 3.0);
            }
        }
        uv_seams.push(UvSeam { edge: (a, b), tris: ts.clone(), color_jump: jump });
    }

    let mut flagged: Vec<u32> = Vec::new();
    let tris_on = |e: (u32, u32)| -> Vec<u32> { topo.group_tris[e.0 as usize].iter().copied().filter(|t| topo.tri_groups[*t as usize].contains(&e.1)).collect() };
    for &e in &open_edges {
        flagged.extend(tris_on(e));
    }
    for &(e, f) in &cracks {
        flagged.extend(tris_on(e));
        flagged.extend(tris_on(f));
    }
    for &(a, b) in &intersections {
        flagged.push(a);
        flagged.push(b);
    }
    for s in &uv_seams {
        flagged.extend(&s.tris);
    }
    flagged.sort_unstable();
    flagged.dedup();
    SeamReport { lo, hi, region_tris, open_edges, cracks, intersections, uv_seams, flagged_tris: flagged }
}

/// What `overlay_picture` draws on top of the textured close-up.
#[derive(Clone, Copy, Debug)]
pub struct Overlay {
    /// Triangle edges, depth-tested (hidden edges are not drawn).
    pub wire: bool,
    /// Shade by depth: near light, far dark, so a gap or a step shows as a jump in shade.
    pub depth: bool,
    /// Tint the report's triangles: intersections red, cracks magenta, open edges orange, texture
    /// seams with a color jump over `seam_jump` yellow.
    pub flagged: bool,
    pub seam_jump: f32,
    /// Drop triangles lying wholly more than this far in front of the camera target, toward the
    /// camera (0 = keep all), so an arm or a hand does not hide the strap in a side view.
    pub clip: f32,
}

pub const RED: [u8; 3] = [255, 40, 40];
pub const MAGENTA: [u8; 3] = [255, 40, 220];
pub const ORANGE: [u8; 3] = [255, 150, 30];
pub const YELLOW: [u8; 3] = [255, 235, 40];

fn line<F: FnMut(i64, i64, f32)>(a: [f32; 3], b: [f32; 3], mut f: F) {
    let n = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()).ceil() as i64).max(1);
    for i in 0..=n {
        let t = i as f32 / n as f32;
        f((a[0] + (b[0] - a[0]) * t).round() as i64, (a[1] + (b[1] - a[1]) * t).round() as i64, a[2] + (b[2] - a[2]) * t);
    }
}

/// A `size` x `size` close-up (RGBA) with the overlays. `report` supplies the flagged triangles
/// and seam edges.
#[allow(clippy::too_many_arguments)]
pub fn overlay_picture(model: &Model, textures: &[Option<Texture>], topo: &Topo, pose: &[Vec<V3>], cam: Camera, size: usize, report: Option<&SeamReport>, ov: Overlay) -> Vec<u8> {
    let cat = |t: u32| -> Option<[u8; 3]> {
        let r = report?;
        if !ov.flagged || r.flagged_tris.binary_search(&t).is_err() {
            return None;
        }
        if r.intersections.iter().any(|&(a, b)| a == t || b == t) {
            return Some(RED);
        }
        let on = |e: (u32, u32)| topo.tri_groups[t as usize].contains(&e.0) && topo.tri_groups[t as usize].contains(&e.1);
        if r.cracks.iter().any(|&(e, f)| on(e) || on(f)) {
            return Some(MAGENTA);
        }
        if r.uv_seams.iter().any(|s| s.color_jump > ov.seam_jump && s.tris.contains(&t)) {
            return Some(YELLOW);
        }
        if r.open_edges.iter().any(|&e| on(e)) {
            return Some(ORANGE);
        }
        None
    };
    let offs = render::tri_offsets(model);
    let tint = |mi: usize, ti: usize| cat((offs[mi] + ti) as u32);
    let z0 = shells::project(&cam, size, size, cam.target)[2];
    let keep = |mi: usize, ti: usize| -> bool {
        ov.clip <= 0.0 || model.meshes[mi].tris[ti].iter().any(|&v| shells::project(&cam, size, size, pose[mi][v as usize])[2] - z0 <= ov.clip)
    };
    let o = render::Options { cam, w: size, h: size, background: ([58, 62, 72], [34, 37, 44]), filter: render::Filter::Smooth, positions: Some(pose) };
    let mut rgba = render::render_masked(model, textures, &o, &tint, Some(&keep)).rgba;
    if !(ov.wire || ov.depth) {
        return rgba;
    }
    let lay = shells::layers(model, pose, &|mi, ti| if keep(mi, ti) { 0 } else { shells::NONE }, &cam, size, size);
    if ov.depth {
        let ds: Vec<f32> = lay.d1.iter().copied().filter(|d| d.is_finite() && *d > f32::MIN / 2.0).collect();
        let (dlo, dhi) = ds.iter().fold((f32::MAX, f32::MIN), |(a, b), &d| (a.min(d), b.max(d)));
        for (i, &d) in lay.d1.iter().enumerate() {
            if lay.s1[i] == shells::NONE {
                continue;
            }
            let s = ((d - dlo) / (dhi - dlo).max(1e-6)).clamp(0.0, 1.0);
            for c in 0..3 {
                rgba[i * 4 + c] = (0.45 * rgba[i * 4 + c] as f32 + 0.55 * (40.0 + 215.0 * s)) as u8;
            }
        }
    }
    if ov.wire {
        let gp = group_pose(topo, pose);
        let px = |p: V3| shells::project(&cam, size, size, p);
        let eps = 0.004 * cam.ortho_half.max(1e-3);
        let plot = |x: i64, y: i64, z: f32, c: [u8; 3], rgba: &mut Vec<u8>| {
            if x < 0 || y < 0 || x >= size as i64 || y >= size as i64 {
                return;
            }
            let i = y as usize * size + x as usize;
            if lay.s1[i] != shells::NONE && z + eps >= lay.d1[i] {
                rgba[i * 4..i * 4 + 3].copy_from_slice(&c);
            }
        };
        for (t, g) in topo.tri_groups.iter().enumerate() {
            let (mi, ti) = topo.tri_loc[t];
            if !keep(mi, ti) {
                continue;
            }
            let p = g.map(|x| px(gp[x as usize]));
            for k in 0..3 {
                line(p[k], p[(k + 1) % 3], |x, y, z| plot(x, y, z, [20, 20, 24], &mut rgba));
            }
        }
        if let Some(r) = report {
            let mut edge = |e: (u32, u32), c: [u8; 3]| line(px(gp[e.0 as usize]), px(gp[e.1 as usize]), |x, y, z| {
                plot(x, y, z, c, &mut rgba);
                plot(x + 1, y, z, c, &mut rgba);
            });
            for &e in &r.open_edges {
                edge(e, ORANGE);
            }
            for s in r.uv_seams.iter().filter(|s| s.color_jump > ov.seam_jump) {
                edge(s.edge, YELLOW);
            }
            for &(e, f) in &r.cracks {
                edge(e, MAGENTA);
                edge(f, MAGENTA);
            }
        }
    }
    rgba
}

/// A protected place the new strap moved: which reference point, in which pose, and by how much.
#[derive(Clone, Debug)]
pub struct Violation {
    /// "off surface": the reference point is farther than the tolerance from the new strap;
    /// "stand-off": the new strap comes closer to the body than the reference did there.
    pub kind: &'static str,
    pub point: usize,
    pub pose: String,
    pub amount: f32,
}

/// Checks that a new strap keeps the protected part of a reference strap: every protected
/// reference vertex (`protect`, vertex indices of `ref_shell`) must lie within `tol` of the new
/// strap's surface (`new_shell`) in every pose, and the new strap there must stand off the body
/// (`body_shell` of `new_model`) at least as far as the reference did, less `tol`. `poses` pairs a
/// name with the reference and new posed positions. Returns violations, worst first, and the
/// largest of each kind.
#[allow(clippy::too_many_arguments)]
pub fn protected_violations(
    ref_model: &Model,
    ref_shell: &Shell,
    ref_body: &Shell,
    protect: &[usize],
    new_model: &Model,
    new_shell: &Shell,
    new_body: &Shell,
    poses: &[(String, Vec<Vec<V3>>, Vec<Vec<V3>>)],
    tol: f32,
) -> (Vec<Violation>, f32, f32) {
    let mut out = Vec::new();
    let (mut worst_surf, mut worst_off) = (0.0f32, 0.0f32);
    for (name, rp, np) in poses {
        for &v in protect {
            let p = rp[ref_shell.mesh][v];
            let (d, _, _) = shells::closest_in_shell(new_model, np, new_shell, p);
            worst_surf = worst_surf.max(d);
            if d > tol {
                out.push(Violation { kind: "off surface", point: v, pose: name.clone(), amount: d });
            }
            // Stand-off: the reference point's distance to its body against the new strap's point
            // nearest it.
            let (_, t, bary) = shells::closest_in_shell(new_model, np, new_shell, p);
            let nm = &new_model.meshes[new_shell.mesh];
            let q3 = nm.tris[t].map(|w| np[new_shell.mesh][w as usize]);
            let q = geom::add(geom::add(geom::scale(q3[0], bary[0]), geom::scale(q3[1], bary[1])), geom::scale(q3[2], bary[2]));
            let (gr, _, _) = shells::closest_in_shell(ref_model, rp, ref_body, p);
            let (gn, _, _) = shells::closest_in_shell(new_model, np, new_body, q);
            let lost = gr - gn;
            worst_off = worst_off.max(lost);
            if lost > tol {
                out.push(Violation { kind: "stand-off", point: v, pose: name.clone(), amount: lost });
            }
        }
    }
    out.sort_by(|a, b| b.amount.total_cmp(&a.amount));
    (out, worst_surf, worst_off)
}

/// A triangle under a thin dark line in a close-up: how many line pixels it holds, the UV box of
/// those pixels, and the texture's own brightness there next to the rendered brightness. A line
/// that is dark in the texture is paint; one that is bright in the texture but dark on screen
/// comes from filtering (a neighboring dark texel or mip level bleeding in) or from geometry.
#[derive(Clone, Debug)]
pub struct LineHit {
    pub tri: u32,
    pub pixels: usize,
    pub uv_lo: [f32; 2],
    pub uv_hi: [f32; 2],
    /// Mean luminance of the texel under the line pixels (nearest texel, full-size texture).
    pub texel_luma: f32,
    /// Mean luminance of the line pixels in the close-up.
    pub screen_luma: f32,
    /// Mean luminance of the pixels `reach` above and below them.
    pub around_luma: f32,
}

fn luma(c: &[u8]) -> f32 {
    (0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32) / 255.0
}

/// Finds thin dark lines in a close-up: pixels at least `dip` darker (luminance, 0..1) than the
/// pixels `reach` above and below them (or left and right), on a surface. Returns the hits per
/// triangle, most pixels first, and a mask of the line pixels (`size` x `size`).
#[allow(clippy::too_many_arguments)]
pub fn thin_lines(model: &Model, textures: &[Option<Texture>], pose: &[Vec<V3>], cam: Camera, size: usize, clip: f32, filter: render::Filter, dip: f32, reach: usize) -> (Vec<LineHit>, Vec<bool>) {
    let z0 = shells::project(&cam, size, size, cam.target)[2];
    let keep = |mi: usize, ti: usize| -> bool { clip <= 0.0 || model.meshes[mi].tris[ti].iter().any(|&v| shells::project(&cam, size, size, pose[mi][v as usize])[2] - z0 <= clip) };
    let o = render::Options { cam, w: size, h: size, background: ([0, 0, 0], [0, 0, 0]), filter, positions: Some(pose) };
    let fr = render::render_masked(model, textures, &o, &|_, _| None, Some(&keep));
    let l: Vec<f32> = fr.rgba.chunks(4).map(luma).collect();
    let mut mask = vec![false; size * size];
    let mut hits: std::collections::BTreeMap<u32, (usize, [f32; 2], [f32; 2], f32, f32, f32)> = std::collections::BTreeMap::new();
    let r = reach as i64;
    for y in r..size as i64 - r {
        for x in r..size as i64 - r {
            let i = y as usize * size + x as usize;
            let id = fr.pick[i];
            if id < 0 {
                continue;
            }
            let at = |dx: i64, dy: i64| -> Option<f32> {
                let j = (y + dy) as usize * size + (x + dx) as usize;
                (fr.pick[j] >= 0).then_some(l[j])
            };
            let mut around = None;
            for (a, b) in [(at(0, -r), at(0, r)), (at(-r, 0), at(r, 0))] {
                if let (Some(a), Some(b)) = (a, b) {
                    if l[i] + dip <= a.min(b) {
                        around = Some(0.5 * (a + b));
                    }
                }
            }
            let Some(around) = around else { continue };
            mask[i] = true;
            // UV under the pixel: barycentric of the pixel center in the projected triangle.
            let (mi, ti) = render::split_id(model, id as usize);
            let m = &model.meshes[mi];
            let p = m.tris[ti].map(|v| shells::project(&cam, size, size, pose[mi][v as usize]));
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let d = (p[1][1] - p[2][1]) * (p[0][0] - p[2][0]) + (p[2][0] - p[1][0]) * (p[0][1] - p[2][1]);
            if d.abs() < 1e-9 {
                continue;
            }
            let b0 = ((p[1][1] - p[2][1]) * (px - p[2][0]) + (p[2][0] - p[1][0]) * (py - p[2][1])) / d;
            let b1 = ((p[2][1] - p[0][1]) * (px - p[2][0]) + (p[0][0] - p[2][0]) * (py - p[2][1])) / d;
            let b = [b0, b1, 1.0 - b0 - b1];
            let uvs = m.tris[ti].map(|v| m.uv[v as usize]);
            let uv = [0, 1].map(|k| b[0] * uvs[0][k] + b[1] * uvs[1][k] + b[2] * uvs[2][k]);
            let tl = m.image.and_then(|ii| textures.get(ii)).and_then(|t| t.as_ref()).map(|t| {
                let c = texel(t, uv);
                0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            });
            let e = hits.entry(id as u32).or_insert((0, [f32::MAX; 2], [f32::MIN; 2], 0.0, 0.0, 0.0));
            e.0 += 1;
            for k in 0..2 {
                e.1[k] = e.1[k].min(uv[k]);
                e.2[k] = e.2[k].max(uv[k]);
            }
            e.3 += tl.unwrap_or(f32::NAN);
            e.4 += l[i];
            e.5 += around;
        }
    }
    let mut out: Vec<LineHit> = hits
        .into_iter()
        .map(|(tri, (n, lo, hi, tl, sl, al))| LineHit { tri, pixels: n, uv_lo: lo, uv_hi: hi, texel_luma: tl / n as f32, screen_luma: sl / n as f32, around_luma: al / n as f32 })
        .collect();
    out.sort_by(|a, b| b.pixels.cmp(&a.pixels));
    (out, mask)
}

/// Repaints thin dark horizontal lines inside `rect` (texel box x0, y0, x1, y1, end exclusive) of
/// an RGBA image: a texel at least `dip` (luminance, 0..1) darker than the texels `reach` above and
/// below it is part of a line, and each run of line texels down a column is refilled by blending
/// the texels just above and just below the run. Alpha is kept. Returns the texels changed and,
/// per image row, how many changed there.
pub fn repaint_thin_lines(rgba: &mut [u8], w: usize, h: usize, rect: [usize; 4], dip: f32, reach: usize) -> (usize, Vec<(usize, usize)>) {
    let [x0, y0, x1, y1] = [rect[0].min(w), rect[1].min(h), rect[2].min(w), rect[3].min(h)];
    let l = |x: usize, y: usize, px: &[u8]| luma(&px[(y * w + x) * 4..(y * w + x) * 4 + 3]);
    let src = rgba.to_vec();
    let mut line = vec![false; w * h];
    for y in y0.max(reach)..y1.min(h.saturating_sub(reach)) {
        for x in x0..x1 {
            if l(x, y, &src) + dip <= l(x, y - reach, &src).min(l(x, y + reach, &src)) {
                line[y * w + x] = true;
            }
        }
    }
    let mut changed = 0usize;
    let mut per_row: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for x in x0..x1 {
        let mut y = y0;
        while y < y1 {
            if !line[y * w + x] {
                y += 1;
                continue;
            }
            let start = y;
            while y < y1 && line[y * w + x] {
                y += 1;
            }
            let (above, below) = (start.saturating_sub(1), y.min(h - 1));
            let (ca, cb) = ((above * w + x) * 4, (below * w + x) * 4);
            let n = (y - start + 1) as f32;
            for (i, yy) in (start..y).enumerate() {
                let t = (i + 1) as f32 / n;
                let o = (yy * w + x) * 4;
                for c in 0..3 {
                    rgba[o + c] = ((1.0 - t) * src[ca + c] as f32 + t * src[cb + c] as f32).round() as u8;
                }
                changed += 1;
                *per_row.entry(yy).or_default() += 1;
            }
        }
    }
    (changed, per_row.into_iter().collect())
}

/// Which texels inside `rect` = [x0, y0, x1, y1) (texels, v down) some triangle of a mesh that
/// uses `image` samples: the texel center lies inside the triangle's UV footprint or within
/// `slack` texels of it. Row-major over the rect.
pub fn uv_coverage(model: &Model, image: usize, w: usize, h: usize, rect: [usize; 4], slack: f32) -> Vec<bool> {
    let [x0, y0, x1, y1] = rect;
    let (rw, rh) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
    let mut cov = vec![false; rw * rh];
    let seg = |p: [f32; 2], a: [f32; 2], b: [f32; 2]| {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l = dx * dx + dy * dy;
        let t = if l > 0.0 { (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l).clamp(0.0, 1.0) } else { 0.0 };
        let (qx, qy) = (a[0] + t * dx - p[0], a[1] + t * dy - p[1]);
        (qx * qx + qy * qy).sqrt()
    };
    for m in model.meshes.iter().filter(|m| m.image == Some(image)) {
        for t in &m.tris {
            let p: Vec<[f32; 2]> = t.iter().map(|&i| { let u = m.uv[i as usize]; [u[0] * w as f32, u[1] * h as f32] }).collect();
            let lo = |k: usize| p.iter().map(|q| q[k]).fold(f32::MAX, f32::min) - slack;
            let hi = |k: usize| p.iter().map(|q| q[k]).fold(f32::MIN, f32::max) + slack;
            let (bx0, bx1) = ((lo(0).floor().max(x0 as f32)) as usize, (hi(0).ceil().min(x1 as f32)) as usize);
            let (by0, by1) = ((lo(1).floor().max(y0 as f32)) as usize, (hi(1).ceil().min(y1 as f32)) as usize);
            if bx0 >= bx1 || by0 >= by1 {
                continue;
            }
            let d = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]);
            for y in by0..by1 {
                for x in bx0..bx1 {
                    let c = [x as f32 + 0.5, y as f32 + 0.5];
                    let side = |a: [f32; 2], b: [f32; 2]| ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * d.signum();
                    let inside = d != 0.0 && side(p[0], p[1]) >= 0.0 && side(p[1], p[2]) >= 0.0 && side(p[2], p[0]) >= 0.0;
                    if inside || seg(c, p[0], p[1]).min(seg(c, p[1], p[2])).min(seg(c, p[2], p[0])) <= slack {
                        cov[(y - y0) * rw + (x - x0)] = true;
                    }
                }
            }
        }
    }
    cov
}

/// Edge padding: grows the colors of covered texels outward into uncovered ones inside `rect`,
/// one texel ring per pass (`passes` rings). Each new texel takes the mean RGB of its already
/// filled 4-neighbors; alpha is kept. Covered texels are never changed. Returns the number of
/// texels changed.
pub fn pad_uv_islands(rgba: &mut [u8], w: usize, rect: [usize; 4], covered: &[bool], passes: usize) -> usize {
    let [x0, y0, x1, y1] = rect;
    let (rw, rh) = (x1 - x0, y1 - y0);
    let mut filled = covered.to_vec();
    let mut changed = 0;
    for _ in 0..passes {
        let mut next = Vec::new();
        for ry in 0..rh {
            for rx in 0..rw {
                if filled[ry * rw + rx] {
                    continue;
                }
                let mut sum = [0u32; 3];
                let mut n = 0;
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (rx as i32 + dx, ry as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= rw as i32 || ny >= rh as i32 || !filled[ny as usize * rw + nx as usize] {
                        continue;
                    }
                    let o = ((y0 + ny as usize) * w + x0 + nx as usize) * 4;
                    for c in 0..3 {
                        sum[c] += rgba[o + c] as u32;
                    }
                    n += 1;
                }
                if n > 0 {
                    next.push((rx, ry, [(sum[0] / n) as u8, (sum[1] / n) as u8, (sum[2] / n) as u8]));
                }
            }
        }
        if next.is_empty() {
            break;
        }
        for (rx, ry, c) in next {
            let o = ((y0 + ry) * w + x0 + rx) * 4;
            if rgba[o..o + 3] != c {
                changed += 1;
            }
            rgba[o..o + 3].copy_from_slice(&c);
            filled[ry * rw + rx] = true;
        }
    }
    changed
}
