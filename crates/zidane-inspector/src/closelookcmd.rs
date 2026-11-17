//! `gaw closelook`: the same close-up crop for every mark, for a reference and one or more
//! candidates, side by side and labeled, so the eye check can be repeated on any version. The
//! cameras are `gaw defects`' orbit views fitted to the reference's part, so crops line up between
//! models and with the screen boxes in defects.json. Reads models only; writes a new folder.
use crate::defectscmd::{make_views, part_keep};
use crate::{Args, KIT_PARTS};
use image::RgbaImage;
use serde_json::Value;
use std::path::{Path, PathBuf};
use wb_core::analysis::Parts;
use wb_core::font;
use wb_core::golden::{self, Kind as Size};
use wb_core::{image_textures, Model};

pub const CLOSELOOK_USAGE: &str = "  gaw closelook --marks MARKS.json --reference REF.glb --candidates A.glb [B.glb ..] --out-dir NEW_DIR
      Draws the same crop of the same view for every mark in MARKS.json, for the reference and each
      candidate, side by side with labels, as closelook.png (and closelook-items-A-B.png parts).
      The views are the orbit views of `gaw defects` fitted to the reference's part, drawn at 768 px
      and enlarged with nearest neighbour, so a pixel in a crop is a pixel of the view the screen
      boxes in defects.json refer to. A marks file: {\"marks\":[{\"id\":1,\"name\":\"..\",\"view\":\"orbit_y000\",
      \"x\":382,\"y\":233,\"r\":83}]} (see skills/gaw/references/zidane-hair-marks.json).
      --marks MARKS.json       the marks (view, centre x y and radius r in view pixels)
      --reference REF.glb      first column, and the model the cameras are fitted to
      --candidates A.glb ..    one column each
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

    let mut paths: Vec<String> = vec![ref_path.to_owned()];
    paths.extend(cands);
    let labels: Vec<String> = match opt(a, "--labels") {
        Some(s) => s.split(',').map(|x| x.trim().to_owned()).collect(),
        None => paths.iter().enumerate().map(|(i, p)| if i == 0 { "reference".to_owned() } else { label_of(Path::new(p)) }).collect(),
    };
    if labels.len() != paths.len() {
        return Err(format!("--labels needs {} names (reference first), got {}", paths.len(), labels.len()));
    }

    let models: Vec<Model> = paths.iter().map(|p| Model::open(Path::new(p))).collect::<Result<_, _>>()?;
    let parts = Parts::load(Path::new(KIT_PARTS)).ok();
    let names = vec![part.clone()];
    let keep = part_keep(&models[0], &names, parts.as_ref(), false)?;
    let views = make_views(&models[0], &keep, true);

    // Draw each needed view once per model.
    let mut need: Vec<String> = Vec::new();
    for m in &marks {
        let v = m.get("view").and_then(Value::as_str).ok_or("a mark needs `view`")?.to_owned();
        if !need.contains(&v) {
            need.push(v);
        }
    }
    let mut shots: Vec<Vec<(String, golden::Shot)>> = Vec::new();
    for model in &models {
        let tex = image_textures(model);
        let mut row = Vec::new();
        for v in &need {
            let (view, _) = views.iter().find(|(x, _)| &x.name == v).ok_or_else(|| format!("no view {v}; the marks use orbit_y000 .. orbit_y315"))?;
            let shot = golden::render_view(model, &tex, view, Size::Close).ok_or("could not draw a view")?;
            row.push((v.clone(), shot));
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
            let side = (f("r") * 2.3).round().clamp(96.0, 384.0) as usize;
            let half = side as f64 / 2.0;
            let (cx, cy) = ((f("x") - half).clamp(0.0, (golden::LARGE_SIZE - side) as f64) as usize, (f("y") - half).clamp(0.0, (golden::LARGE_SIZE - side) as f64) as usize);
            let vi = need.iter().position(|v| v == vname).unwrap_or(0);
            for (c, model_shots) in shots.iter().enumerate() {
                let px = golden::LARGE_SIZE;
                let rgba = model_shots[vi].1.rgb.to_rgba8();
                let x0 = label_w + gap + c * (cell + gap);
                for yy in 0..cell {
                    for xx in 0..cell {
                        let (sx, sy) = (cx + xx * side / cell, cy + yy * side / cell);
                        let i = (sy * px + sx) * 4;
                        img.put_pixel((x0 + xx) as u32, (y0 + yy) as u32, image::Rgba([rgba[i], rgba[i + 1], rgba[i + 2], 255]));
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
