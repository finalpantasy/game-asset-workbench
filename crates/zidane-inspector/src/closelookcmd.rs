//! `gaw closelook`: the same close-up crop for every mark, for a reference and one or more
//! candidates, side by side and labeled, so the eye check can be repeated on any version. Two
//! camera sets: the orbit views of `gaw defects` (crops line up with the screen boxes in
//! defects.json) and the `gaw hair` close-up turntable cameras (crops line up with the pictures
//! the hair session reviews). Optionally draws the boxes of defect runs on the crops. Reads
//! models only; writes a new folder.
use crate::defectscmd::{make_views, part_keep};
use crate::{Args, KIT_PARTS};
use image::RgbaImage;
use serde_json::Value;
use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use wb_core::analysis::Parts;
use wb_core::font;
use wb_core::golden::{self, Kind as Size};
use wb_core::render::{self, Camera, Filter};
use wb_core::{image_textures, Model};

pub const CLOSELOOK_USAGE: &str = "  gaw closelook --marks MARKS.json --reference REF.glb --candidates A.glb [B.glb ..] --out-dir NEW_DIR
      Draws the same crop of the same view for every mark in MARKS.json, for the reference and each
      candidate, side by side with labels, as closelook.png (and closelook-items-A-B.png parts).
      Two camera sets. `orbit` (default): the orbit views of `gaw defects` fitted to the reference's
      part, drawn at 768 px, so a pixel in a crop is a pixel of the view the screen boxes in
      defects.json refer to. `turntable`: the close-up cameras of `gaw hair` (front, left, back,
      right; orthographic, 480 px, the same crops as the hair session's turntable-close.png).
      A marks file: {\"camera\":\"orbit\",\"marks\":[{\"id\":1,\"name\":\"..\",\"view\":\"orbit_y000\",
      \"x\":382,\"y\":233,\"r\":83}]}; a mark may add \"box\":[x0,y0,x1,y1] to fix the crop (view pixels).
      Examples: skills/gaw/references/zidane-hair-marks.json (orbit) and
      zidane-hair-marks-turntable.json (turntable, views front/left/back/right).
      --marks MARKS.json       the marks (view, centre x y and radius r in view pixels)
      --reference REF.glb      first column, and the model the cameras are fitted to
      --candidates A.glb ..    one column each
      --camera orbit|turntable camera set (default: the marks file's `camera`, else orbit)
      --boxes DIR,DIR,..       one `gaw defects`/`gaw gate` run folder (holding defects.json) per column,
                               reference first, `-` for none: draws each run's boxes on its column
                               (orbit: the screen boxes; turntable: the marked triangles projected)
      --box-kinds a,b          kinds to draw (default direction-break, layer-direction-cut, smear-blur,
                               flat-patch, speck, lock-scale, card-edge)
      --labels a,b,c           column labels (default: reference, then each candidate's folder name)
      --part NAME              part the cameras fit to (default hair)
      --cell N                 pixels per crop in the sheet (default 300)
      --rows N                 marks per split sheet (default 5)
      --out-dir NEW_DIR        new or empty";

fn opt<'a>(a: &'a [String], k: &str) -> Option<&'a str> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).map(String::as_str)
}

fn label_of(p: &Path) -> String {
    let parent = p.parent().and_then(|d| d.file_name()).and_then(|s| s.to_str()).unwrap_or("");
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let l = parent.trim_start_matches("zidane-hair-tool-").trim_start_matches("zidane-hair-reference-");
    if l.is_empty() || l == "exports" { stem.to_owned() } else { l.to_owned() }
}

/// The `gaw hair` turntable views: name, yaw and pitch in degrees from the front.
const TURN_VIEWS: [(&str, f32, f32); 4] = [("front", 0.0, 0.0), ("left", 90.0, 0.0), ("back", 180.0, 0.0), ("right", -90.0, 0.0)];
/// Panel size of the hair turntable's close-up sheet.
const TURN_PX: usize = 480;

fn kind_color(k: &str) -> [u8; 3] {
    match k {
        "jagged-edge" => [255, 230, 0],
        "discoloration" => [255, 0, 255],
        "seam-stripe" => [0, 230, 230],
        "direction-break" => [255, 140, 0],
        "dark-gutter" => [60, 100, 255],
        "smear-blur" => [255, 40, 40],
        "flat-patch" => [40, 230, 60],
        "speck" => [255, 255, 255],
        "layer-direction-cut" => [255, 60, 160],
        "lock-scale" => [128, 255, 0],
        "card-edge" => [0, 200, 255],
        _ => [200, 200, 200],
    }
}

/// One drawn view of one model.
struct Pic {
    px: usize,
    rgba: Vec<u8>,
}

fn num4(v: &Value) -> Option<[f64; 4]> {
    let a = v.as_array()?;
    Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?, a.get(2)?.as_f64()?, a.get(3)?.as_f64()?])
}

/// Boxes (kind, [x0,y0,x1,y1] in view pixels) of one defects.json for one view.
fn boxes_for(dj: &Value, kinds: &[String], view: &str, turn: Option<(&Camera, &Model, &str)>) -> Vec<(String, [f64; 4])> {
    let mut out = Vec::new();
    for d in dj.get("defects").and_then(Value::as_array).into_iter().flatten() {
        let kind = d.get("kind").and_then(Value::as_str).unwrap_or("");
        if !kinds.iter().any(|k| k == kind) {
            continue;
        }
        match turn {
            None => {
                if d.get("view").and_then(Value::as_str) == Some(view) {
                    if let Some(b) = d.get("screen_box").and_then(num4) {
                        out.push((kind.to_owned(), b));
                    }
                }
                for s in d.get("seen_boxes").and_then(Value::as_array).into_iter().flatten() {
                    if s.get("view").and_then(Value::as_str) == Some(view) {
                        if let Some(b) = s.get("screen_box").and_then(num4) {
                            out.push((kind.to_owned(), b));
                        }
                    }
                }
            }
            Some((cam, model, orbit_name)) => {
                let seen = d.get("view").and_then(Value::as_str) == Some(orbit_name) || d.get("seen_in").and_then(Value::as_array).map(|a| a.iter().any(|v| v.as_str() == Some(orbit_name))).unwrap_or(false);
                if !seen {
                    continue;
                }
                let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
                for e in d.get("marked").and_then(|m| m.get("meshes")).and_then(Value::as_array).into_iter().flatten() {
                    let Some(mi) = e.get("mesh").and_then(Value::as_u64).map(|x| x as usize) else { continue };
                    let Some(m) = model.meshes.get(mi) else { continue };
                    for t in e.get("triangles").and_then(Value::as_array).into_iter().flatten() {
                        let Some(tri) = t.as_u64().and_then(|ti| m.tris.get(ti as usize)) else { continue };
                        for &v in tri {
                            let p = cam.project(TURN_PX, TURN_PX, m.pos[v as usize]);
                            x0 = x0.min(p[0] as f64);
                            y0 = y0.min(p[1] as f64);
                            x1 = x1.max(p[0] as f64);
                            y1 = y1.max(p[1] as f64);
                        }
                    }
                }
                if x0 <= x1 {
                    out.push((kind.to_owned(), [x0 - 3.0, y0 - 3.0, x1 + 3.0, y1 + 3.0]));
                }
            }
        }
    }
    out
}

pub fn closelook_cmd(args: &Args) -> Result<(), String> {
    let a = &args.0;
    let marks_path = opt(a, "--marks").ok_or("closelook needs --marks MARKS.json")?;
    let ref_path = opt(a, "--reference").ok_or("closelook needs --reference REF.glb")?;
    let out = PathBuf::from(opt(a, "--out-dir").ok_or("closelook needs --out-dir NEW_DIR")?);
    let cands: Vec<String> = match a.iter().position(|x| x == "--candidates") {
        Some(i) => a[i + 1..].iter().take_while(|x| !x.starts_with("--")).cloned().collect(),
        None => Vec::new(),
    };
    if cands.is_empty() {
        return Err("closelook needs --candidates A.glb [B.glb ..]".into());
    }
    if out.exists() && std::fs::read_dir(&out).map(|mut d| d.next().is_some()).unwrap_or(true) {
        return Err(format!("{} already has files; --out-dir must be new or empty", out.display()));
    }
    let part = opt(a, "--part").unwrap_or("hair").to_owned();
    let cell: usize = opt(a, "--cell").map(|s| s.parse().map_err(|_| "--cell: not a number")).transpose()?.unwrap_or(300);
    let rows: usize = opt(a, "--rows").map(|s| s.parse().map_err(|_| "--rows: not a number")).transpose()?.unwrap_or(5).max(1);
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(marks_path).map_err(|e| format!("{marks_path}: {e}"))?).map_err(|e| format!("{marks_path}: {e}"))?;
    let marks = doc.get("marks").and_then(Value::as_array).cloned().or_else(|| doc.as_array().cloned()).ok_or("marks file: no marks list")?;
    let camera = opt(a, "--camera").or_else(|| doc.get("camera").and_then(Value::as_str)).unwrap_or("orbit").to_owned();
    if camera != "orbit" && camera != "turntable" {
        return Err(format!("--camera: `{camera}` is not orbit or turntable"));
    }
    let turntable = camera == "turntable";

    let mut paths: Vec<String> = vec![ref_path.to_owned()];
    paths.extend(cands);
    let labels: Vec<String> = match opt(a, "--labels") {
        Some(s) => s.split(',').map(|x| x.trim().to_owned()).collect(),
        None => paths.iter().enumerate().map(|(i, p)| if i == 0 { "reference".to_owned() } else { label_of(Path::new(p)) }).collect(),
    };
    if labels.len() != paths.len() {
        return Err(format!("--labels needs {} names (reference first), got {}", paths.len(), labels.len()));
    }
    let kinds: Vec<String> = match opt(a, "--box-kinds") {
        Some(s) => s.split(',').map(|x| x.trim().to_owned()).collect(),
        None => ["direction-break", "layer-direction-cut", "smear-blur", "flat-patch", "speck", "lock-scale", "card-edge"].iter().map(|s| s.to_string()).collect(),
    };
    let runs: Vec<Option<Value>> = match opt(a, "--boxes") {
        None => vec![None; paths.len()],
        Some(s) => {
            let parts: Vec<&str> = s.split(',').map(str::trim).collect();
            if parts.len() != paths.len() {
                return Err(format!("--boxes needs {} entries (reference first, `-` for none), got {}", paths.len(), parts.len()));
            }
            parts
                .iter()
                .map(|p| {
                    if *p == "-" {
                        return Ok(None);
                    }
                    let f = Path::new(p).join("defects.json");
                    let t = std::fs::read_to_string(&f).map_err(|e| format!("{}: {e}", f.display()))?;
                    serde_json::from_str(&t).map(Some).map_err(|e| format!("{}: {e}", f.display()))
                })
                .collect::<Result<_, String>>()?
        }
    };

    let models: Vec<Model> = paths.iter().map(|p| Model::open(Path::new(p))).collect::<Result<_, _>>()?;
    let parts = Parts::load(Path::new(KIT_PARTS)).ok();
    let names = vec![part.clone()];
    let keep = part_keep(&models[0], &names, parts.as_ref(), false)?;

    // The views the marks use, and the camera of each.
    let mut need: Vec<String> = Vec::new();
    for m in &marks {
        let v = m.get("view").and_then(Value::as_str).ok_or("a mark needs `view`")?.to_owned();
        if !need.contains(&v) {
            need.push(v);
        }
    }
    let orbit_views = if turntable { Vec::new() } else { make_views(&models[0], &keep, true) };
    // Turntable: the camera is fitted to the reference part's bounds, as `gaw hair` does.
    let turn_cams: Vec<(String, f32, Camera)> = if turntable {
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for (mi, m) in models[0].meshes.iter().enumerate() {
            for (ti, t) in m.tris.iter().enumerate() {
                if !keep[mi].get(ti).copied().unwrap_or(false) {
                    continue;
                }
                for &v in t {
                    for k in 0..3 {
                        lo[k] = lo[k].min(m.pos[v as usize][k]);
                        hi[k] = hi[k].max(m.pos[v as usize][k]);
                    }
                }
            }
        }
        if lo[0] > hi[0] {
            return Err(format!("no triangles of part `{part}` in the reference"));
        }
        let target: [f32; 3] = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
        let ext = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f32::max);
        TURN_VIEWS
            .iter()
            .map(|&(n, yaw, pitch)| (n.to_owned(), yaw, Camera { yaw: PI + yaw.to_radians(), pitch: pitch.to_radians(), dist: ext * 8.0, target, fov_y: 30f32.to_radians(), ortho: true, ortho_half: ext * 0.62 }))
            .collect()
    } else {
        Vec::new()
    };

    let mut shots: Vec<Vec<Pic>> = Vec::new();
    for model in &models {
        let tex = image_textures(model);
        let mut row = Vec::new();
        for v in &need {
            if turntable {
                let (_, _, cam) = turn_cams.iter().find(|(n, _, _)| n == v).ok_or_else(|| format!("no turntable view {v}; the marks use front, left, back or right"))?;
                let o = render::Options { cam: *cam, w: TURN_PX, h: TURN_PX, background: ([58, 64, 76], [32, 34, 40]), filter: Filter::Smooth, positions: None };
                let f = render::render(model, &tex, &o, &|_, _| None);
                row.push(Pic { px: TURN_PX, rgba: f.rgba });
            } else {
                let (view, _) = orbit_views.iter().find(|(x, _)| &x.name == v).ok_or_else(|| format!("no view {v}; the marks use orbit_y000 .. orbit_y315"))?;
                let shot = golden::render_view(model, &tex, view, Size::Close).ok_or("could not draw a view")?;
                row.push(Pic { px: golden::LARGE_SIZE, rgba: shot.rgb.to_rgba8() });
            }
        }
        shots.push(row);
    }

    const BG: [u8; 4] = [20, 20, 28, 255];
    let (label_w, head_h, gap) = (150usize, 22usize, 6usize);
    let draw_sheet = |ids: &[usize]| -> RgbaImage {
        let w = label_w + paths.len() * (cell + gap) + gap;
        let h = head_h + ids.len() * (cell + gap) + gap;
        let mut img = RgbaImage::from_pixel(w as u32, h as u32, image::Rgba(BG));
        for (c, l) in labels.iter().enumerate() {
            font::draw(img.as_mut(), w, h, (label_w + gap + c * (cell + gap)) as i64, 6, l, 1, [20, 20, 28]);
        }
        for (r, &mi) in ids.iter().enumerate() {
            let m = &marks[mi];
            let f = |k: &str| m.get(k).and_then(Value::as_f64).unwrap_or(0.0);
            let (id, name) = (m.get("id").and_then(Value::as_i64).unwrap_or(mi as i64 + 1), m.get("name").and_then(Value::as_str).unwrap_or(""));
            let vname = m.get("view").and_then(Value::as_str).unwrap_or("");
            let y0 = head_h + gap + r * (cell + gap);
            font::draw(img.as_mut(), w, h, 4, (y0 + cell / 2 - 4) as i64, &format!("{id} {name}"), 1, [20, 20, 28]);
            let vi = need.iter().position(|v| v == vname).unwrap_or(0);
            let px = shots[0][vi].px;
            // Crop: a given box, else a square around the mark.
            let (cx, cy, side) = if let Some(b) = m.get("box").and_then(num4) {
                let s = (b[2] - b[0]).max(b[3] - b[1]).round().clamp(32.0, px as f64) as usize;
                ((b[0].round().max(0.0) as usize).min(px - s), (b[1].round().max(0.0) as usize).min(px - s), s)
            } else {
                let s = (f("r") * 2.3).round().clamp(96.0, 384.0).min(px as f64) as usize;
                let half = s as f64 / 2.0;
                (((f("x") - half).clamp(0.0, (px - s) as f64)) as usize, ((f("y") - half).clamp(0.0, (px - s) as f64)) as usize, s)
            };
            for (c, model_shots) in shots.iter().enumerate() {
                let pic = &model_shots[vi];
                let x0 = label_w + gap + c * (cell + gap);
                for yy in 0..cell {
                    for xx in 0..cell {
                        let (sx, sy) = (cx + xx * side / cell, cy + yy * side / cell);
                        let i = (sy * pic.px + sx) * 4;
                        img.put_pixel((x0 + xx) as u32, (y0 + yy) as u32, image::Rgba([pic.rgba[i], pic.rgba[i + 1], pic.rgba[i + 2], 255]));
                    }
                }
                // Defect boxes of this column's run.
                if let Some(dj) = &runs[c] {
                    let turn = if turntable {
                        let (_, yaw, cam) = turn_cams.iter().find(|(n, _, _)| n == vname).unwrap();
                        Some((cam, &models[c], format!("orbit_y{:03}", (*yaw as i32).rem_euclid(360))))
                    } else {
                        None
                    };
                    let bx = boxes_for(dj, &kinds, vname, turn.as_ref().map(|(cam, m, n)| (*cam, *m, n.as_str())));
                    let k = cell as f64 / side as f64;
                    for (kind, b) in bx {
                        let col = kind_color(&kind);
                        let (bx0, by0, bx1, by1) = ((b[0] - cx as f64) * k, (b[1] - cy as f64) * k, (b[2] - cx as f64) * k, (b[3] - cy as f64) * k);
                        if bx1 < 0.0 || by1 < 0.0 || bx0 >= cell as f64 || by0 >= cell as f64 {
                            continue;
                        }
                        let mut put = |x: f64, y: f64| {
                            if x >= 0.0 && y >= 0.0 && x < cell as f64 && y < cell as f64 {
                                img.put_pixel((x0 + x as usize) as u32, (y0 + y as usize) as u32, image::Rgba([col[0], col[1], col[2], 255]));
                            }
                        };
                        let (xa, xb, ya, yb) = (bx0.round(), bx1.round(), by0.round(), by1.round());
                        let mut x = xa;
                        while x <= xb {
                            put(x, ya);
                            put(x, yb);
                            x += 1.0;
                        }
                        let mut y = ya;
                        while y <= yb {
                            put(xa, y);
                            put(xb, y);
                            y += 1.0;
                        }
                    }
                }
            }
        }
        img
    };
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let all: Vec<usize> = (0..marks.len()).collect();
    draw_sheet(&all).save(out.join("closelook.png")).map_err(|e| format!("closelook.png: {e}"))?;
    println!("wrote {}", out.join("closelook.png").display());
    if marks.len() > rows {
        for chunk in all.chunks(rows) {
            let id = |i: usize| marks[i].get("id").and_then(Value::as_i64).unwrap_or(i as i64 + 1);
            let name = format!("closelook-items-{}-{}.png", id(chunk[0]), id(*chunk.last().unwrap()));
            draw_sheet(chunk).save(out.join(&name)).map_err(|e| format!("{name}: {e}"))?;
            println!("wrote {}", out.join(&name).display());
        }
    }
    Ok(())
}
