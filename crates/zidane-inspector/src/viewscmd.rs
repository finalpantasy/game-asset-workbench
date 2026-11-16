//! Saved inspection views (camera, pose, time) shared by the Mark & fix page and the command line,
//! so a side close-up or an underside view picked once by hand can be rendered for any candidate
//! file as an acceptance picture. Also the locked-places check for saved requests.
use crate::requestcmd;
use crate::Args;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use wb_core::geom::dist;
use wb_core::render::{self, Camera, Filter};
use wb_core::Model;

pub const VIEWS_USAGE: &str = "  zidane-inspector views [list] [--json]
      Lists the saved inspection views (Mark & fix page, Views card): camera, pose and time.
  zidane-inspector views render FILE.glb --out-dir DIR [--names a,b] [--size W,H]
      Renders FILE.glb from every saved view (or the named ones) to DIR/<view>.png, posed like
      the view. Render the stock file and a candidate the same way to compare them view by view.";

#[derive(Clone)]
pub struct SavedView {
    pub name: String,
    pub cam: Camera,
    pub clip: Option<String>,
    pub time: f32,
}

/// One file for the workbench, next to the requests folder.
pub fn views_file() -> PathBuf {
    PathBuf::from(requestcmd::REQUESTS_DIR).parent().map(|p| p.join("views").join("saved-views.json")).unwrap_or_else(|| PathBuf::from("saved-views.json"))
}

pub fn view_to_json(v: &SavedView) -> Value {
    let c = &v.cam;
    json!({
        "name": v.name,
        "yaw_deg": c.yaw.to_degrees(), "pitch_deg": c.pitch.to_degrees(), "dist": c.dist,
        "target": c.target, "fov_y_deg": c.fov_y.to_degrees(), "ortho": c.ortho, "ortho_half": c.ortho_half,
        "clip": v.clip, "time": v.time,
    })
}

pub fn view_from_json(j: &Value) -> Option<SavedView> {
    let f = |k: &str| j[k].as_f64().map(|x| x as f32);
    let t = j["target"].as_array()?;
    let target = [t.first()?.as_f64()? as f32, t.get(1)?.as_f64()? as f32, t.get(2)?.as_f64()? as f32];
    Some(SavedView {
        name: j["name"].as_str()?.to_owned(),
        cam: Camera { yaw: f("yaw_deg")?.to_radians(), pitch: f("pitch_deg")?.to_radians(), dist: f("dist")?, target, fov_y: f("fov_y_deg")?.to_radians(), ortho: j["ortho"].as_bool().unwrap_or(false), ortho_half: f("ortho_half").unwrap_or(1.0) },
        clip: j["clip"].as_str().map(str::to_owned),
        time: f("time").unwrap_or(0.0),
    })
}

pub fn load_views(path: &Path) -> Vec<SavedView> {
    let Ok(b) = std::fs::read(path) else { return vec![] };
    let Ok(j) = serde_json::from_slice::<Value>(&b) else { return vec![] };
    j["views"].as_array().into_iter().flatten().filter_map(view_from_json).collect()
}

pub fn store_views(path: &Path, views: &[SavedView]) -> Result<(), String> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    let j = json!({
        "workbench_views": 1,
        "about": "Saved inspection views from Mark & fix. Render them for any GLB with: zidane-inspector views render FILE.glb --out-dir DIR",
        "views": views.iter().map(view_to_json).collect::<Vec<_>>(),
    });
    std::fs::write(path, serde_json::to_string_pretty(&j).unwrap()).map_err(|e| e.to_string())
}

pub fn views_cmd(args: &Args) -> Result<(), String> {
    let pos: Vec<String> = args.positionals().iter().map(|p| p.display().to_string()).collect();
    let file = views_file();
    let views = load_views(&file);
    match pos.first().map(String::as_str).unwrap_or("list") {
        "list" => {
            if args.flag("--json") {
                println!("{}", serde_json::to_string_pretty(&json!({"file": file.display().to_string(), "views": views.iter().map(view_to_json).collect::<Vec<_>>()})).unwrap());
                return Ok(());
            }
            println!("{} saved view(s) in {}", views.len(), file.display());
            for v in &views {
                println!("  {}  yaw {:.0}°, pitch {:.0}°, distance {:.2}, {}", v.name, v.cam.yaw.to_degrees(), v.cam.pitch.to_degrees(), v.cam.dist, match &v.clip {
                    Some(c) => format!("clip {c} at {:.2}s", v.time),
                    None => "rest pose".into(),
                });
            }
            Ok(())
        }
        "render" => {
            let glb = pos.get(1).ok_or("views render needs FILE.glb")?;
            let out = PathBuf::from(args.opt("--out-dir").ok_or("views render needs --out-dir DIR")?);
            let names: Option<Vec<String>> = args.opt("--names").map(|s| s.split(',').map(|x| x.trim().to_owned()).collect());
            let (w, h) = match args.opt("--size").map(|s| s.split(',').filter_map(|x| x.trim().parse::<usize>().ok()).collect::<Vec<_>>()) {
                Some(v) if v.len() == 2 => (v[0].clamp(32, 4096), v[1].clamp(32, 4096)),
                _ => (800, 1000),
            };
            let pick: Vec<&SavedView> = views.iter().filter(|v| names.as_ref().is_none_or(|n| n.contains(&v.name))).collect();
            if pick.is_empty() {
                return Err(format!("no saved views to render (file {})", file.display()));
            }
            let model = Model::open(Path::new(glb))?;
            let textures = wb_core::model_textures(&model);
            let rest: Vec<Vec<_>> = model.meshes.iter().map(|m| m.pos.clone()).collect();
            std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
            for v in pick {
                let pose = requestcmd::posed(&model, &rest, v.clip.as_deref(), v.time);
                let o = render::Options { cam: v.cam, w, h, background: crate::theme::VIEW_BG, filter: Filter::Smooth, positions: Some(&pose) };
                let f = render::render(&model, &textures, &o, &|_, _| None);
                let safe: String = v.name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '-' }).collect();
                let p = out.join(format!("{safe}.png"));
                image::save_buffer(&p, &f.rgba, w as u32, h as u32, image::ExtendedColorType::Rgba8).map_err(|e| format!("{}: {e}", p.display()))?;
                println!("{}  {}", v.name, p.display());
            }
            Ok(())
        }
        other => Err(format!("unknown views command {other} (list, render)")),
    }
}

/// `requests locks NAME FILE.glb`: did FILE.glb move any place the request locked?
pub fn locks_check(req: &Value, glb: &Path) -> Result<(), String> {
    let list = req["locked"]["places"].as_array().ok_or("this request has no locked places")?;
    let tol = req["locked"]["tolerance"].as_f64().unwrap_or(1e-4) as f32;
    let model = Model::open(glb)?;
    let mut bad: Vec<(f32, String)> = vec![];
    for p in list {
        let want = p["position"].as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect::<Vec<_>>()).unwrap_or_default();
        if want.len() != 3 {
            continue;
        }
        let want = [want[0], want[1], want[2]];
        let mut worst = 0.0f32;
        let mut missing = false;
        for v in p["vertices"].as_array().into_iter().flatten() {
            let (mi, vi) = (v["mesh"].as_u64().unwrap_or(0) as usize, v["vertex"].as_u64().unwrap_or(0) as usize);
            match model.meshes.get(mi).and_then(|m| m.pos.get(vi)) {
                Some(&q) => worst = worst.max(dist(q, want)),
                None => missing = true,
            }
        }
        if missing {
            bad.push((f32::INFINITY, format!("{} is missing (the meshes do not match)", p["vertices"])));
        } else if worst > tol {
            bad.push((worst, format!("({:.4}, {:.4}, {:.4}) moved {worst:.5}", want[0], want[1], want[2])));
        }
    }
    bad.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    if bad.is_empty() {
        println!("Locks held: all {} locked places in {} stayed put (tolerance {tol:.6})", list.len(), glb.display());
        return Ok(());
    }
    println!("Locks moved: {} of {} locked places in {} moved (tolerance {tol:.6})", bad.len(), list.len(), glb.display());
    for (_, t) in bad.iter().take(20) {
        println!("  {t}");
    }
    Err(format!("{} locked places moved", bad.len()))
}
