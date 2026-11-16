//! Workbench requests: what a person marked on a model or texture, and what they want, saved as a
//! folder any AI session can read without the window (`gaw requests show latest`).
//!
//! A request folder holds `request.json` (source file, note, marked triangles per mesh, a texel mask
//! PNG per sheet, edge loops as vertex ids and positions, camera, clip and time), a screenshot with
//! the marked area outlined, a clean screenshot, and texture crops. A "fix-my-edit" request also holds
//! `before.glb` and `after.glb`: the person's rough edit, for an AI to tidy.
use crate::doc::Doc;
use crate::Args;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use wb_core::analysis::Parts;
use wb_core::edit::{self, EdgeLoop, Topo};
use wb_core::geom::V3;
use wb_core::render::{self, Camera, Filter};
use wb_core::Texture;

pub const REQUESTS_DIR: &str = r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding\workbench\requests";
pub const MODDING: &str = r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding";

pub const REQUESTS_USAGE: &str = "  gaw requests [list] [--dir DIR] [--json]
      Lists the saved workbench requests (newest first): kind, note, what was marked.
  gaw requests show NAME|latest|PATH [--json]
      Prints one request: the note, the source file, the marked triangles by part, edge loops,
      texel masks and every file in the folder. --json prints request.json itself.
      Requests are saved from the window's Mark & fix page (\"Save request\", \"Fix my edit\").
  gaw requests locks NAME|latest|PATH FILE.glb
      Checks FILE.glb (an edit or an AI fix) against the places the request locked. Exits 1 and
      lists them when any moved.";

// ---------------------------------------------------------------------------------------------
// Local time (Windows), without a date crate
// ---------------------------------------------------------------------------------------------

#[repr(C)]
#[derive(Default)]
struct SystemTime16 {
    year: u16,
    month: u16,
    dow: u16,
    day: u16,
    hour: u16,
    minute: u16,
    second: u16,
    ms: u16,
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn GetLocalTime(st: *mut SystemTime16);
}

/// ("20261116-140322", "2026-11-16 14:03:22") in local time.
pub fn local_stamp() -> (String, String) {
    let mut t = SystemTime16::default();
    #[cfg(windows)]
    unsafe {
        GetLocalTime(&mut t);
    }
    (
        format!("{:04}{:02}{:02}-{:02}{:02}{:02}", t.year, t.month, t.day, t.hour, t.minute, t.second),
        format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", t.year, t.month, t.day, t.hour, t.minute, t.second),
    )
}

pub fn slug(note: &str, fallback: &str) -> String {
    let words: Vec<String> = note
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .take(6)
        .map(|w| w.to_ascii_lowercase())
        .collect();
    let mut s = words.join("-");
    s.truncate(40);
    let s = s.trim_end_matches('-').to_owned();
    if s.is_empty() {
        fallback.to_owned()
    } else {
        s
    }
}

// ---------------------------------------------------------------------------------------------
// Saving
// ---------------------------------------------------------------------------------------------

/// Everything a request needs, owned, so it can be written on a worker thread.
pub struct ReqInput {
    pub kind: &'static str,
    pub root: PathBuf,
    pub doc: Arc<Doc>,
    pub topo: Arc<Topo>,
    pub parts: Option<Arc<Parts>>,
    pub note: String,
    pub sel: Vec<f32>,
    pub loops: Vec<EdgeLoop>,
    /// Per image: texel marks made on the texture view, (width, height, 0..255 per texel).
    pub extra: Vec<Option<(usize, usize, Vec<u8>)>>,
    /// Bind-pose group positions (edited or rest).
    pub gpos: Vec<V3>,
    /// Edited textures per image (None = unchanged).
    pub textures: Vec<Option<Texture>>,
    pub cam: Camera,
    pub aspect: f32,
    pub clip: Option<String>,
    pub time: f32,
    pub symmetry: bool,
    /// For fix-my-edit: the edited model, already written.
    pub after_glb: Option<Vec<u8>>,
    /// Per weld group: locked on the Mark & fix page (empty = none).
    pub locked: Vec<bool>,
}

pub struct Saved {
    pub dir: PathBuf,
    pub prompt: String,
    pub summary: String,
}

fn tex_of<'a>(r: &'a ReqInput, image: usize) -> Option<&'a Texture> {
    r.textures.get(image).and_then(|t| t.as_ref()).or_else(|| r.doc.textures.get(image).and_then(|t| t.as_ref()))
}

fn save_png(path: &Path, w: usize, h: usize, rgba: Vec<u8>) -> Result<(), String> {
    image::RgbaImage::from_raw(w as u32, h as u32, rgba).ok_or("bad image size")?.save(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn save_gray(path: &Path, w: usize, h: usize, v: Vec<u8>) -> Result<(), String> {
    image::GrayImage::from_raw(w as u32, h as u32, v).ok_or("bad image size")?.save(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Full-size mask of one sheet: marked triangles (their weight) and texture-view marks.
pub fn sheet_mask(r: &ReqInput, image: usize, tw: usize, th: usize) -> Vec<f32> {
    let mut m = edit::selection_mask(&r.doc.model, &r.topo, &r.sel, image, tw, th);
    if let Some(Some((ew, eh, ev))) = r.extra.get(image) {
        for y in 0..th {
            let sy = y * eh / th;
            for x in 0..tw {
                let v = ev[sy * ew + x * ew / tw] as f32 / 255.0;
                if v > m[y * tw + x] {
                    m[y * tw + x] = v;
                }
            }
        }
    }
    m
}

fn draw_disc(rgba: &mut [u8], w: usize, h: usize, x: f32, y: f32, rad: f32, c: [u8; 3]) {
    let (x0, x1) = ((x - rad).floor().max(0.0) as usize, ((x + rad).ceil() as usize).min(w.saturating_sub(1)));
    let (y0, y1) = ((y - rad).floor().max(0.0) as usize, ((y + rad).ceil() as usize).min(h.saturating_sub(1)));
    for yy in y0..=y1 {
        for xx in x0..=x1 {
            let (dx, dy) = (xx as f32 + 0.5 - x, yy as f32 + 0.5 - y);
            if dx * dx + dy * dy <= rad * rad {
                rgba[(yy * w + xx) * 4..(yy * w + xx) * 4 + 3].copy_from_slice(&c);
            }
        }
    }
}

pub fn draw_line(rgba: &mut [u8], w: usize, h: usize, a: [f32; 2], b: [f32; 2], rad: f32, c: [u8; 3]) {
    let n = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()) / (rad * 0.5).max(0.5)).ceil().max(1.0) as usize;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        draw_disc(rgba, w, h, a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, rad, c);
    }
}

/// Outlines pixels where `inside` changes, `thick` pixels wide.
fn outline(rgba: &mut [u8], w: usize, h: usize, inside: &[bool], thick: usize, c: [u8; 3]) {
    let mut edge = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !inside[i] {
                continue;
            }
            let out = (x == 0 || !inside[i - 1]) || (x + 1 == w || !inside[i + 1]) || (y == 0 || !inside[i - w]) || (y + 1 == h || !inside[i + w]);
            if out {
                edge[i] = true;
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            if !edge[y * w + x] {
                continue;
            }
            for yy in y.saturating_sub(thick / 2)..=(y + thick / 2).min(h - 1) {
                for xx in x.saturating_sub(thick / 2)..=(x + thick / 2).min(w - 1) {
                    rgba[(yy * w + xx) * 4..(yy * w + xx) * 4 + 3].copy_from_slice(&c);
                }
            }
        }
    }
}

/// The view with the marks: selected triangles tinted orange and outlined white, edge loops cyan.
pub fn marked_shot(r: &ReqInput, w: usize, marks: bool) -> (usize, usize, Vec<u8>) {
    let h = ((w as f32 / r.aspect.max(0.2)).round() as usize).clamp(200, 2400);
    let model = &r.doc.model;
    let mesh_pos = r.topo.mesh_positions(model, &r.gpos);
    let pose = posed(model, &mesh_pos, r.clip.as_deref(), r.time);
    let mut textures: Vec<Option<Texture>> = Vec::new();
    for i in 0..model.images.len() {
        textures.push(tex_of(r, i).map(copy_tex));
    }
    let sel = &r.sel;
    let topo = &r.topo;
    let tint = |mi: usize, ti: usize| -> Option<[u8; 3]> {
        let w = sel[topo.gid(mi, ti)];
        (marks && w > 0.0).then(|| sel_color(w))
    };
    let o = render::Options { cam: r.cam, w, h, background: crate::theme::VIEW_BG, filter: Filter::Smooth, positions: Some(&pose) };
    let mut f = render::render(model, &textures, &o, &tint);
    if marks {
        let inside: Vec<bool> = f.pick.iter().map(|&id| id >= 0 && sel[id as usize] > 0.0).collect();
        outline(&mut f.rgba, w, h, &inside, 3, [255, 255, 255]);
        let rig = wb_core::bake::Rig::new(&r.cam, w, h);
        let gp = posed_groups(r, &pose);
        for l in &r.loops {
            let pts: Vec<[f32; 2]> = l.groups.iter().map(|&g| {
                let p = rig.project(gp[g as usize]);
                [p[0], p[1]]
            }).collect();
            let n = pts.len();
            let segs = if l.closed { n } else { n.saturating_sub(1) };
            for k in 0..segs {
                draw_line(&mut f.rgba, w, h, pts[k], pts[(k + 1) % n], 3.5, [10, 10, 10]);
            }
            for k in 0..segs {
                draw_line(&mut f.rgba, w, h, pts[k], pts[(k + 1) % n], 2.0, [80, 240, 255]);
            }
            for p in &pts {
                draw_disc(&mut f.rgba, w, h, p[0], p[1], 3.5, [255, 255, 255]);
            }
        }
    }
    (w, h, f.rgba)
}

pub fn sel_color(w: f32) -> [u8; 3] {
    let w = w.clamp(0.0, 1.0);
    [255, (150.0 - 70.0 * w) as u8, (60.0 - 50.0 * w) as u8]
}

pub fn copy_tex(t: &Texture) -> Texture {
    Texture { w: t.w, h: t.h, rgba: t.rgba.clone(), mips: t.mips.iter().map(|l| wb_core::Level { w: l.w, h: l.h, rgba: l.rgba.clone() }).collect() }
}

/// Skinned positions of edited bind-pose vertices for a clip at `t` (rest pose when `clip` is None).
pub fn posed(model: &wb_core::Model, mesh_pos: &[Vec<V3>], clip: Option<&str>, t: f32) -> Vec<Vec<V3>> {
    let ci = clip.and_then(|c| model.clips.iter().position(|x| x == c));
    match (&model.rig, ci) {
        (Some(rig), Some(ci)) => {
            let jm = rig.joint_matrices(Some(ci), t);
            model
                .meshes
                .iter()
                .zip(mesh_pos)
                .map(|(m, p)| {
                    let tmp = wb_core::model::Mesh { name: String::new(), pos: p.clone(), uv: Vec::new(), joints: m.joints.clone(), weights: m.weights.clone(), tris: Vec::new(), image: None };
                    wb_core::anim::skin(&tmp, &jm)
                })
                .collect()
        }
        _ => mesh_pos.to_vec(),
    }
}

/// Posed position of each group (its first member's posed position).
pub fn posed_groups(r: &ReqInput, pose: &[Vec<V3>]) -> Vec<V3> {
    r.topo.members.iter().map(|m| pose[m[0].0 as usize][m[0].1 as usize]).collect()
}

pub fn save(r: ReqInput) -> Result<Saved, String> {
    let (stamp, when) = local_stamp();
    let model = &r.doc.model;
    std::fs::create_dir_all(&r.root).map_err(|e| format!("{}: {e}", r.root.display()))?;
    let base = format!("{stamp}-{}", slug(&r.note, if r.kind == "fix-my-edit" { "fix-my-edit" } else { "mark" }));
    let mut dir = r.root.join(&base);
    let mut k = 2;
    while dir.exists() {
        dir = r.root.join(format!("{base}-{k}"));
        k += 1;
    }
    std::fs::create_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut files: Vec<String> = Vec::new();

    // Screenshots.
    let (w, h, px) = marked_shot(&r, 1400, true);
    save_png(&dir.join("screenshot.png"), w, h, px)?;
    let (w, h, px) = marked_shot(&r, 1400, false);
    save_png(&dir.join("screenshot-clean.png"), w, h, px)?;
    files.push("screenshot.png".into());
    files.push("screenshot-clean.png".into());

    // Selection per mesh.
    let topo = &r.topo;
    let tri_parts = &r.doc.tri_parts;
    let part_name = |id: u8| -> String { r.parts.as_ref().map(|p| p.name(id).to_owned()).unwrap_or_else(|| format!("part {id}")) };
    let mut meshes = Vec::new();
    let mut total = 0usize;
    let mut all_parts: Vec<(String, usize)> = Vec::new();
    for (mi, m) in model.meshes.iter().enumerate() {
        let ids: Vec<usize> = (0..m.tris.len()).filter(|&t| r.sel[topo.gid(mi, t)] > 0.0).collect();
        if ids.is_empty() {
            continue;
        }
        total += ids.len();
        let soft = ids.iter().any(|&t| r.sel[topo.gid(mi, t)] < 0.999);
        let mut parts: Vec<(String, usize)> = Vec::new();
        for &t in &ids {
            let p = tri_parts.get(mi).and_then(|v| v.get(t)).copied().unwrap_or(0);
            let name = if p == 0 { "no part".to_owned() } else { part_name(p) };
            match parts.iter_mut().find(|e| e.0 == name) {
                Some(e) => e.1 += 1,
                None => parts.push((name, 1)),
            }
        }
        parts.sort_by(|a, b| b.1.cmp(&a.1));
        for (n, c) in &parts {
            match all_parts.iter_mut().find(|e| &e.0 == n) {
                Some(e) => e.1 += c,
                None => all_parts.push((n.clone(), *c)),
            }
        }
        meshes.push(json!({
            "mesh": mi, "name": m.name, "image": m.image.map(|i| model.images[i].name.trim_end_matches(".png").to_owned()),
            "triangle_count": ids.len(),
            "triangles": ids,
            "weights": if soft { json!(ids.iter().map(|&t| (r.sel[topo.gid(mi, t)] * 1000.0).round() / 1000.0).collect::<Vec<_>>()) } else { Value::Null },
            "parts": parts.iter().map(|(n, c)| json!({"part": n, "triangles": c})).collect::<Vec<_>>(),
        }));
    }
    all_parts.sort_by(|a, b| b.1.cmp(&a.1));

    // Texel masks and crops per sheet.
    let mut masks = Vec::new();
    for (ii, im) in model.images.iter().enumerate() {
        let Some(t) = tex_of(&r, ii) else { continue };
        let sheet = im.name.trim_end_matches(".png").to_owned();
        let mask = sheet_mask(&r, ii, t.w, t.h);
        let n = mask.iter().filter(|&&v| v > 0.0).count();
        if n == 0 {
            continue;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (t.w, t.h, 0usize, 0usize);
        for y in 0..t.h {
            for x in 0..t.w {
                if mask[y * t.w + x] > 0.0 {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
        }
        let mname = format!("mask-{sheet}.png");
        save_gray(&dir.join(&mname), t.w, t.h, mask.iter().map(|&v| (v * 255.0).round() as u8).collect())?;
        // Crop with a margin, at most 2048 px on the long side.
        let pad = (t.w.max(t.h) / 64).max(8);
        let (cx0, cy0) = (x0.saturating_sub(pad), y0.saturating_sub(pad));
        let (cx1, cy1) = ((x1 + pad).min(t.w - 1), (y1 + pad).min(t.h - 1));
        let (cw, ch) = (cx1 - cx0 + 1, cy1 - cy0 + 1);
        let mut raw = vec![0u8; cw * ch * 4];
        let mut marked = vec![0u8; cw * ch * 4];
        let mut inside = vec![false; cw * ch];
        for y in 0..ch {
            for x in 0..cw {
                let s = ((cy0 + y) * t.w + cx0 + x) * 4;
                let o = (y * cw + x) * 4;
                raw[o..o + 4].copy_from_slice(&t.rgba[s..s + 4]);
                raw[o + 3] = 255;
                let m = mask[(cy0 + y) * t.w + cx0 + x];
                inside[y * cw + x] = m > 0.0;
                for c in 0..3 {
                    let base = t.rgba[s + c] as f32 * if m > 0.0 { 1.0 } else { 0.45 };
                    let tint = [255.0, 140.0, 20.0][c];
                    marked[o + c] = (base * (1.0 - 0.45 * m) + tint * 0.45 * m) as u8;
                }
                marked[o + 3] = 255;
            }
        }
        outline(&mut marked, cw, ch, &inside, if cw.max(ch) > 1024 { 3 } else { 1 }, [255, 255, 255]);
        let cname = format!("texture-{sheet}-crop.png");
        let kname = format!("texture-{sheet}-crop-marked.png");
        let shrink = |v: Vec<u8>| -> (usize, usize, Vec<u8>) {
            let s = cw.max(ch);
            if s <= 2048 {
                return (cw, ch, v);
            }
            let (nw, nh) = ((cw * 2048 / s).max(1), (ch * 2048 / s).max(1));
            let img = image::RgbaImage::from_raw(cw as u32, ch as u32, v).unwrap();
            let o = image::imageops::resize(&img, nw as u32, nh as u32, image::imageops::FilterType::Triangle);
            (nw, nh, o.into_raw())
        };
        let (a, b, v) = shrink(raw);
        save_png(&dir.join(&cname), a, b, v)?;
        let (a, b, v) = shrink(marked);
        save_png(&dir.join(&kname), a, b, v)?;
        files.extend([mname.clone(), cname.clone(), kname.clone()]);
        masks.push(json!({
            "sheet": sheet, "image_index": ii, "file": mname, "size": [t.w, t.h], "texels": n,
            "bbox": [x0, y0, x1 + 1, y1 + 1], "bbox_uv": [x0 as f32 / t.w as f32, y0 as f32 / t.h as f32, (x1 + 1) as f32 / t.w as f32, (y1 + 1) as f32 / t.h as f32],
            "crop": cname, "crop_marked": kname, "crop_box": [cx0, cy0, cx1 + 1, cy1 + 1],
        }));
    }

    // Edge loops.
    let loops: Vec<Value> = r
        .loops
        .iter()
        .map(|l| {
            let pts: Vec<Value> = l
                .groups
                .iter()
                .map(|&g| {
                    let mem = &topo.members[g as usize];
                    json!({
                        "position": r.gpos[g as usize].map(|x| (x * 1e5).round() / 1e5),
                        "rest_position": topo.rest[g as usize].map(|x| (x * 1e5).round() / 1e5),
                        "vertices": mem.iter().map(|&(mi, vi)| json!({"mesh": mi, "mesh_name": model.meshes[mi as usize].name, "vertex": vi, "uv": model.meshes[mi as usize].uv[vi as usize]})).collect::<Vec<_>>(),
                    })
                })
                .collect();
            let n = l.groups.len();
            let segs = if l.closed { n } else { n.saturating_sub(1) };
            let length: f32 = (0..segs).map(|k| wb_core::geom::dist(r.gpos[l.groups[k] as usize], r.gpos[l.groups[(k + 1) % n] as usize])).sum();
            // Kit parts of the triangles along the loop.
            let mut along: Vec<String> = Vec::new();
            for k in 0..segs {
                let (a, b) = (l.groups[k], l.groups[(k + 1) % n]);
                for &t in &topo.group_tris[a as usize] {
                    if topo.tri_groups[t as usize].contains(&b) {
                        let (mi, ti) = topo.tri_loc[t as usize];
                        let p = tri_parts.get(mi).and_then(|v| v.get(ti)).copied().unwrap_or(0);
                        let name = if p == 0 { "no part".to_owned() } else { part_name(p) };
                        if !along.contains(&name) {
                            along.push(name);
                        }
                    }
                }
            }
            json!({"points": n, "closed": l.closed, "on_border": l.border, "length": (length * 1e4).round() / 1e4, "parts_along": along, "path": pts})
        })
        .collect();

    // Edits (fix my edit).
    let mut edits = Value::Null;
    if let Some(after) = &r.after_glb {
        let src = std::fs::read(&r.doc.path).map_err(|e| format!("{}: {e}", r.doc.path.display()))?;
        std::fs::write(dir.join("before.glb"), &src).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("after.glb"), after).map_err(|e| e.to_string())?;
        files.extend(["before.glb".into(), "after.glb".into()]);
        let (nm, mx) = edit::moved(topo, &r.gpos);
        let mut painted = serde_json::Map::new();
        for (ii, t) in r.textures.iter().enumerate() {
            if let (Some(t), Some(Some(o))) = (t, r.doc.textures.get(ii)) {
                let n = t.rgba.chunks_exact(4).zip(o.rgba.chunks_exact(4)).filter(|(a, b)| a != b).count();
                if n > 0 {
                    painted.insert(model.images[ii].name.trim_end_matches(".png").to_owned(), json!(n));
                }
            }
        }
        let moved_by_part = {
            let mut v: Vec<(String, usize)> = Vec::new();
            for (g, p) in r.gpos.iter().enumerate() {
                if *p == topo.rest[g] {
                    continue;
                }
                let t = topo.group_tris[g].first().copied();
                let name = t
                    .map(|t| {
                        let (mi, ti) = topo.tri_loc[t as usize];
                        let id = tri_parts.get(mi).and_then(|v| v.get(ti)).copied().unwrap_or(0);
                        if id == 0 { "no part".to_owned() } else { part_name(id) }
                    })
                    .unwrap_or_default();
                match v.iter_mut().find(|e| e.0 == name) {
                    Some(e) => e.1 += 1,
                    None => v.push((name, 1)),
                }
            }
            v.sort_by(|a, b| b.1.cmp(&a.1));
            v
        };
        edits = json!({
            "before": "before.glb", "after": "after.glb",
            "moved_places": nm, "max_move": (mx as f64 * 1e5).round() / 1e5, "model_height": (topo.height as f64 * 1e5).round() / 1e5,
            "moved_by_part": moved_by_part.iter().map(|(n, c)| json!({"part": n, "places": c})).collect::<Vec<_>>(),
            "painted_texels": painted,
            "what_to_do": "The person made this rough edit by hand. Keep what they meant and make it clean: smooth lumps and creases, keep the silhouette they pulled, re-weight moved vertices if they now bend badly, and repaint seams so both sides of a UV seam match. Compare before.glb and after.glb.",
        });
        let (w, h, px) = {
            let mut b = ReqInput { gpos: topo.rest.clone(), textures: vec![], after_glb: None, ..clone_input(&r) };
            b.sel = r.sel.clone();
            marked_shot(&b, 1400, false)
        };
        save_png(&dir.join("screenshot-before.png"), w, h, px)?;
        files.push("screenshot-before.png".into());
    }

    let prompt = format!("Look at workbench request {}", dir.display());
    let j = json!({
        "workbench_request": 1,
        "kind": r.kind,
        "created": when,
        "folder": dir.display().to_string(),
        "prompt": prompt,
        "note": r.note,
        "source": {
            "file": r.doc.path.display().to_string(), "label": r.doc.label, "triangles": model.triangle_count(),
            "meshes": model.meshes.iter().map(|m| json!({"name": m.name, "vertices": m.pos.len(), "triangles": m.tris.len(), "image": m.image.map(|i| model.images[i].name.trim_end_matches(".png").to_owned())})).collect::<Vec<_>>(),
            "images": model.images.iter().enumerate().map(|(i, im)| json!({"name": im.name.trim_end_matches(".png"), "size": r.doc.textures.get(i).and_then(|t| t.as_ref()).map(|t| [t.w, t.h])})).collect::<Vec<_>>(),
        },
        "view": {
            "camera": {"yaw_deg": r.cam.yaw.to_degrees(), "pitch_deg": r.cam.pitch.to_degrees(), "distance": r.cam.dist, "target": r.cam.target, "fov_y_deg": r.cam.fov_y.to_degrees(), "ortho": r.cam.ortho, "ortho_half": r.cam.ortho_half, "aspect": r.aspect},
            "clip": r.clip, "time": r.time, "pose": if r.clip.is_some() { "clip" } else { "rest (bind pose)" },
        },
        "symmetry_x": r.symmetry,
        "marked": {
            "triangles": total,
            "parts": all_parts.iter().map(|(n, c)| json!({"part": n, "triangles": c})).collect::<Vec<_>>(),
            "meshes": meshes,
        },
        "texel_masks": masks,
        "edge_loops": loops,
        "edits": edits,
        "locked": locked_json(&r),
        "saved_views": {"file": crate::viewscmd::views_file().display().to_string(), "names": crate::viewscmd::load_views(&crate::viewscmd::views_file()).iter().map(|v| v.name.clone()).collect::<Vec<_>>(), "render": "gaw views render FILE.glb --out-dir DIR"},
        "files": files,
        "how_to_read": {
            "triangles": "Indices into each mesh's triangle list (indices accessor / 3), meshes in file order (glTF mesh, then primitive). Same numbering as gaw and wb_core::Model.",
            "weights": "Soft-brush strength per listed triangle, 0..1 (absent when all are 1).",
            "texel_masks": "Grayscale PNG at the sheet's full size: 255 = fully marked, 0 = not. Covers the marked triangles' texels plus marks drawn on the texture view.",
            "edge_loops": "Each point is one place on the surface: its bind-pose position and every mesh vertex there (UV seams and back faces give several). Consecutive points share a mesh edge.",
            "screenshot": "screenshot.png: marked triangles orange with a white outline, edge loops cyan. screenshot-clean.png: the same view without marks.",
            "cli": "gaw requests show <folder> prints this request.",
        },
    });
    let mut j = j;
    tidy_numbers(&mut j);
    std::fs::write(dir.join("request.json"), serde_json::to_string_pretty(&j).unwrap()).map_err(|e| e.to_string())?;
    let summary = format!(
        "{} triangles marked{}{}",
        total,
        if r.loops.is_empty() { String::new() } else { format!(", {} edge loop{}", r.loops.len(), if r.loops.len() == 1 { "" } else { "s" }) },
        if r.after_glb.is_some() { ", before and after GLBs" } else { "" }
    );
    Ok(Saved { dir, prompt, summary })
}

/// Locked places: each with its bind-pose position and every mesh vertex there.
fn locked_json(r: &ReqInput) -> Value {
    let places: Vec<Value> = r
        .locked
        .iter()
        .enumerate()
        .filter(|(_, &l)| l)
        .map(|(g, _)| json!({"position": r.topo.rest[g], "vertices": r.topo.members[g].iter().map(|&(mi, vi)| json!({"mesh": mi, "vertex": vi})).collect::<Vec<_>>()}))
        .collect();
    if places.is_empty() {
        return Value::Null;
    }
    json!({
        "count": places.len(),
        "tolerance": r.topo.height * 1e-4,
        "about": "Places that must not move. Check a candidate with: gaw requests locks <this folder> FILE.glb",
        "places": places,
    })
}

fn clone_input(r: &ReqInput) -> ReqInput {
    ReqInput {
        kind: r.kind,
        root: r.root.clone(),
        doc: r.doc.clone(),
        topo: r.topo.clone(),
        parts: r.parts.clone(),
        note: r.note.clone(),
        sel: r.sel.clone(),
        loops: r.loops.clone(),
        extra: r.extra.clone(),
        gpos: r.gpos.clone(),
        textures: r.textures.iter().map(|t| t.as_ref().map(copy_tex)).collect(),
        cam: r.cam,
        aspect: r.aspect,
        clip: r.clip.clone(),
        time: r.time,
        symmetry: r.symmetry,
        after_glb: None,
        locked: r.locked.clone(),
    }
}

// ---------------------------------------------------------------------------------------------
// Writing an edited model
// ---------------------------------------------------------------------------------------------

/// The model with moved vertices and repainted sheets, as GLB bytes.
pub fn edited_glb(doc: &Doc, topo: &Topo, gpos: &[V3], textures: &[Option<Texture>]) -> Result<Vec<u8>, String> {
    let mut bytes = edit::write_positions(&doc.model, topo, gpos)?;
    for (ii, t) in textures.iter().enumerate() {
        let Some(t) = t else { continue };
        let same = doc.textures.get(ii).and_then(|o| o.as_ref()).is_some_and(|o| o.rgba == t.rgba);
        if same {
            continue;
        }
        let m = wb_core::Model::from_bytes(&bytes)?;
        let mut png = std::io::Cursor::new(Vec::new());
        image::RgbaImage::from_raw(t.w as u32, t.h as u32, t.rgba.clone()).ok_or("bad texture size")?.write_to(&mut png, image::ImageFormat::Png).map_err(|e| e.to_string())?;
        let name = m.images[ii].name.clone();
        bytes = wb_core::swap::swap(&m, &name, &png.into_inner())?.bytes;
    }
    Ok(bytes)
}

/// Refuses paths outside the Modding folder and existing files.
pub fn check_new_file(p: &Path) -> Result<(), String> {
    if p.exists() {
        return Err(format!("{} already exists; Save as always writes a new file", p.display()));
    }
    let full = p.parent().and_then(|d| d.canonicalize().ok()).map(|d| d.join(p.file_name().unwrap_or_default())).unwrap_or_else(|| p.to_path_buf());
    let s = full.to_string_lossy().to_ascii_lowercase().replace("\\\\?\\", "");
    if !s.starts_with(&MODDING.to_ascii_lowercase()) {
        return Err(format!("{} is outside the Modding folder; the installed game stays stock", p.display()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Checks after Save as
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CheckLine {
    pub ok: bool,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct CheckResult {
    pub file: PathBuf,
    pub lines: Vec<CheckLine>,
}

impl CheckResult {
    pub fn passed(&self) -> bool {
        self.lines.iter().all(|l| l.ok)
    }
}

fn run_self(args: &[String], limit_s: u64) -> Result<String, String> {
    use std::process::{Command, Stdio};
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut cmd = Command::new(exe);
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0000_4000 | 0x0800_0000); // below normal priority, no console window
    }
    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    let t0 = std::time::Instant::now();
    loop {
        if let Some(st) = child.try_wait().map_err(|e| e.to_string())? {
            let out = child.wait_with_output().map_err(|e| e.to_string())?;
            let so = String::from_utf8_lossy(&out.stdout).into_owned();
            if !st.success() && so.trim().is_empty() {
                return Err(String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or("failed").to_owned());
            }
            return Ok(so);
        }
        if t0.elapsed().as_secs() > limit_s {
            let _ = child.kill();
            return Err(format!("stopped after {limit_s} s"));
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

fn loose_count(j: &Value) -> usize {
    j["shells"].as_array().map(|a| a.iter().filter(|s| s["status"].as_str().is_some_and(|x| x.starts_with("LOOSE"))).count()).unwrap_or(0)
}

/// Runs the checks on a saved edit: structure (in process), the pattern checks' skeleton, clips,
/// triangles and shared texels, `shells` (loose pieces) and `judge` against the goldens with the
/// edited parts excluded (collateral change only).
pub fn run_checks(source: &Path, file: &Path, edited_parts: &[String], progress: &dyn Fn(String)) -> CheckResult {
    let mut lines = Vec::new();
    let mut push = |ok: bool, text: String| lines.push(CheckLine { ok, text });
    progress("Checking the file".into());
    match (wb_core::Model::open(source), wb_core::Model::open(file)) {
        (Ok(a), Ok(b)) => {
            let same_skin = a.meshes.len() == b.meshes.len() && a.meshes.iter().zip(&b.meshes).all(|(x, y)| x.joints == y.joints && x.weights == y.weights);
            let same_tris = a.meshes.iter().zip(&b.meshes).all(|(x, y)| x.tris == y.tris && x.uv == y.uv);
            push(a.joint_names == b.joint_names && a.clips == b.clips, format!("Skeleton and clips the same ({} joints, {} clips)", b.joint_names.len(), b.clips.len()));
            push(same_skin, if same_skin { "Skin weights kept".into() } else { "Skin weights changed".into() });
            push(same_tris, if same_tris { format!("Same {} triangles and UVs", b.triangle_count()) } else { "Triangles or UVs changed".into() });
            progress("Pattern checks".into());
            let c = wb_core::relayout::check_keep(&a, &b, &[0.0, 90.0, 180.0, 270.0], 400, &[]);
            let shared: Vec<String> = c.meshes.iter().map(|m| format!("{} {} -> {}", m.image, m.shared_texels[0], m.shared_texels[1])).collect();
            let no_new_shared = c.meshes.iter().all(|m| m.shared_texels[1] <= m.shared_texels[0]);
            push(c.same_skeleton && c.same_clips && c.same_triangles() && no_new_shared, format!("Pattern checks: skeleton, clips, triangles; texels on two surfaces {}", shared.join(", ")));
            let worst = c.views.iter().map(|v| v.mean_diff).fold(0.0f32, f32::max);
            lines.push(CheckLine { ok: true, text: format!("Render difference from the source: up to {worst:.2} of 255 (expected for an edit)") });
        }
        (Err(e), _) | (_, Err(e)) => push(false, format!("Could not reopen: {e}")),
    }
    progress("Looking for loose pieces (shells)".into());
    let shells = |p: &Path| -> Result<Value, String> {
        let s = run_self(&["shells".into(), p.display().to_string(), "--json".into()], 300)?;
        serde_json::from_str(&s).map_err(|e| format!("shells output: {e}"))
    };
    match (shells(source), shells(file)) {
        (Ok(a), Ok(b)) => {
            let (la, lb) = (loose_count(&a), loose_count(&b));
            lines.push(CheckLine { ok: lb <= la, text: format!("Loose pieces (shells): {la} before, {lb} after") });
        }
        (Err(e), _) | (_, Err(e)) => lines.push(CheckLine { ok: false, text: format!("shells did not run: {e}") }),
    }
    let goldens = Path::new(MODDING).join(r"experiments\goldens\zidane");
    if goldens.join("manifest.json").is_file() {
        progress("Judging against the goldens (a minute or two)".into());
        let mut a = vec!["judge".into(), "--golden-dir".into(), goldens.display().to_string(), file.display().to_string(), "--json".into()];
        if !edited_parts.is_empty() {
            a.push("--only".into());
            a.push(edited_parts.join(","));
        }
        match run_self(&a, 540).and_then(|s| serde_json::from_str::<Value>(&s).map_err(|e| format!("judge output: {e}"))) {
            Ok(j) => {
                let v = j["verdict"].as_str().unwrap_or("?").to_owned();
                let score = j["score"].as_f64().unwrap_or(0.0);
                let what = if edited_parts.is_empty() { "whole model".to_owned() } else { format!("outside {}", edited_parts.join(", ")) };
                let col = j["collateral"]["region"].as_str().map(|r| format!(", worst {r}")).unwrap_or_default();
                lines.push(CheckLine { ok: v != "FAIL", text: format!("Judge vs stock goldens ({what}): {v} {score:.3}{col}") });
            }
            Err(e) => lines.push(CheckLine { ok: false, text: format!("judge did not run: {e}") }),
        }
    }
    CheckResult { file: file.to_path_buf(), lines }
}

// ---------------------------------------------------------------------------------------------
// The `requests` command
// ---------------------------------------------------------------------------------------------

fn list_dirs(root: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(root).map(|r| r.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.join("request.json").is_file()).collect()).unwrap_or_default();
    v.sort();
    v.reverse();
    v
}

fn read_req(dir: &Path) -> Result<Value, String> {
    serde_json::from_slice(&std::fs::read(dir.join("request.json")).map_err(|e| format!("{}: {e}", dir.display()))?).map_err(|e| format!("request.json: {e}"))
}

/// f32 values widened to f64 print as 0.027869999408721924; keep 6 decimals (well under a texel or
/// a thousandth of the model's height) so request.json stays readable.
fn tidy_numbers(v: &mut Value) {
    match v {
        Value::Number(n) if n.is_f64() => {
            let f = n.as_f64().unwrap_or(0.0);
            if let Some(m) = serde_json::Number::from_f64((f * 1e6).round() / 1e6) {
                *n = m;
            }
        }
        Value::Array(a) => a.iter_mut().for_each(tidy_numbers),
        Value::Object(o) => o.values_mut().for_each(tidy_numbers),
        _ => {}
    }
}

pub fn requests_cmd(args: &Args) -> Result<(), String> {
    let root = PathBuf::from(args.opt("--dir").unwrap_or(REQUESTS_DIR));
    let pos: Vec<String> = args.positionals().iter().map(|p| p.display().to_string()).collect();
    let sub = pos.first().map(String::as_str).unwrap_or("list");
    match sub {
        "list" => {
            let dirs = list_dirs(&root);
            if args.flag("--json") {
                let v: Vec<Value> = dirs.iter().filter_map(|d| read_req(d).ok().map(|j| json!({"folder": d.display().to_string(), "kind": j["kind"], "created": j["created"], "note": j["note"], "triangles": j["marked"]["triangles"], "edge_loops": j["edge_loops"].as_array().map(|a| a.len())}))).collect();
                println!("{}", serde_json::to_string_pretty(&v).unwrap());
                return Ok(());
            }
            if dirs.is_empty() {
                println!("No requests in {}", root.display());
                return Ok(());
            }
            println!("{} request(s) in {}", dirs.len(), root.display());
            for d in dirs {
                let name = d.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
                match read_req(&d) {
                    Ok(j) => {
                        let note = j["note"].as_str().unwrap_or("").lines().next().unwrap_or("").to_owned();
                        println!(
                            "  {name}\n      {}  {} triangles, {} edge loop(s)  \"{}\"",
                            j["kind"].as_str().unwrap_or("?"),
                            j["marked"]["triangles"],
                            j["edge_loops"].as_array().map(|a| a.len()).unwrap_or(0),
                            note
                        );
                    }
                    Err(e) => println!("  {name}  (unreadable: {e})"),
                }
            }
            Ok(())
        }
        "locks" => {
            let which = pos.get(1).map(String::as_str).ok_or("requests locks needs NAME|latest|PATH and FILE.glb")?;
            let glb = pos.get(2).ok_or("requests locks needs FILE.glb after the request name")?;
            let dir = if which == "latest" {
                list_dirs(&root).into_iter().next().ok_or_else(|| format!("no requests in {}", root.display()))?
            } else if Path::new(which).join("request.json").is_file() {
                PathBuf::from(which)
            } else if root.join(which).join("request.json").is_file() {
                root.join(which)
            } else {
                list_dirs(&root).into_iter().find(|d| d.file_name().is_some_and(|f| f.to_string_lossy().starts_with(which))).ok_or_else(|| format!("no request named {which} in {}", root.display()))?
            };
            crate::viewscmd::locks_check(&read_req(&dir)?, Path::new(glb))
        }
        "show" => {
            let which = pos.get(1).map(String::as_str).unwrap_or("latest");
            let dir = if which == "latest" {
                list_dirs(&root).into_iter().next().ok_or_else(|| format!("no requests in {}", root.display()))?
            } else if Path::new(which).join("request.json").is_file() {
                PathBuf::from(which)
            } else {
                let d = root.join(which);
                if d.join("request.json").is_file() {
                    d
                } else {
                    list_dirs(&root).into_iter().find(|d| d.file_name().is_some_and(|f| f.to_string_lossy().starts_with(which))).ok_or_else(|| format!("no request named {which} in {}", root.display()))?
                }
            };
            let j = read_req(&dir)?;
            if args.flag("--json") {
                println!("{}", serde_json::to_string_pretty(&j).unwrap());
                return Ok(());
            }
            print_request(&dir, &j);
            Ok(())
        }
        other => Err(format!("requests {other}: use list or show")),
    }
}

fn print_request(dir: &Path, j: &Value) {
    println!("Workbench request {}", dir.display());
    println!("  kind     {}", j["kind"].as_str().unwrap_or("?"));
    println!("  created  {}", j["created"].as_str().unwrap_or("?"));
    println!("  source   {}", j["source"]["file"].as_str().unwrap_or("?"));
    println!("  note");
    for l in j["note"].as_str().unwrap_or("").lines() {
        println!("    | {l}");
    }
    let v = &j["view"];
    println!(
        "  view     yaw {:.0} deg, pitch {:.0} deg, {}{}",
        v["camera"]["yaw_deg"].as_f64().unwrap_or(0.0),
        v["camera"]["pitch_deg"].as_f64().unwrap_or(0.0),
        v["pose"].as_str().unwrap_or(""),
        v["clip"].as_str().map(|c| format!(" {c} at {:.2} s", v["time"].as_f64().unwrap_or(0.0))).unwrap_or_default()
    );
    println!("  mirror   {}", if j["symmetry_x"].as_bool().unwrap_or(false) { "on (marks copied left/right)" } else { "off" });
    println!("  marked   {} triangles", j["marked"]["triangles"]);
    for p in j["marked"]["parts"].as_array().into_iter().flatten() {
        println!("             {:>5}  {}", p["triangles"], p["part"].as_str().unwrap_or(""));
    }
    for m in j["marked"]["meshes"].as_array().into_iter().flatten() {
        let ids: Vec<String> = m["triangles"].as_array().into_iter().flatten().take(24).map(|x| x.to_string()).collect();
        let n = m["triangle_count"].as_u64().unwrap_or(0);
        println!("    mesh {} ({}): {}{}", m["mesh"], m["name"].as_str().unwrap_or(""), ids.join(" "), if n > 24 { format!(" ... ({n})") } else { String::new() });
    }
    for (i, l) in j["edge_loops"].as_array().into_iter().flatten().enumerate() {
        println!(
            "  edge loop {}: {} points, {}{}, length {:.4}, along {}",
            i + 1,
            l["points"],
            if l["closed"].as_bool().unwrap_or(false) { "closed" } else { "open" },
            if l["on_border"].as_bool().unwrap_or(false) { " on the surface border" } else { "" },
            l["length"].as_f64().unwrap_or(0.0),
            l["parts_along"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default()
        );
        for p in l["path"].as_array().into_iter().flatten() {
            let vs: Vec<String> = p["vertices"].as_array().into_iter().flatten().map(|v| format!("{}:{}", v["mesh"], v["vertex"])).collect();
            let pos = p["position"].as_array().map(|a| a.iter().map(|x| format!("{:.4}", x.as_f64().unwrap_or(0.0))).collect::<Vec<_>>().join(", ")).unwrap_or_default();
            println!("      ({pos})  vertices {}", vs.join(" "));
        }
    }
    for m in j["texel_masks"].as_array().into_iter().flatten() {
        println!("  mask     {} {} texels, box {} -> {}", m["sheet"].as_str().unwrap_or(""), m["texels"], m["bbox"], m["file"].as_str().unwrap_or(""));
    }
    if !j["edits"].is_null() {
        let e = &j["edits"];
        let by_part: Vec<String> = e["moved_by_part"].as_array().into_iter().flatten().map(|m| format!("{} {}", m["part"].as_str().unwrap_or("?"), m["places"])).collect();
        let painted: Vec<String> = e["painted_texels"].as_object().into_iter().flatten().map(|(k, v)| format!("{v} texels on {k}")).collect();
        println!("  edits    {} places moved, up to {} units ({})", e["moved_places"], e["max_move"], if by_part.is_empty() { "none".into() } else { by_part.join(", ") });
        println!("           painted: {}", if painted.is_empty() { "nothing".into() } else { painted.join(", ") });
        println!("           before.glb, after.glb: {}", e["what_to_do"].as_str().unwrap_or(""));
    }
    println!("  files");
    for f in j["files"].as_array().into_iter().flatten() {
        println!("    {}", dir.join(f.as_str().unwrap_or("")).display());
    }
    println!("    {}", dir.join("request.json").display());
}
