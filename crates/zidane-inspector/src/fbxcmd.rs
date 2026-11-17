//! `gaw fbx`: write a GLB as the FBX that Memoria loads for a custom model, and check the FBX by
//! reading it back and rendering it next to the GLB.
use crate::Args;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use wb_core::fbx::{self, ExportOptions};
use wb_core::render::{self, Camera, Filter};
use wb_core::{image_textures, Model, Texture};

pub const FBX_USAGE: &str = "  gaw fbx FILE.glb --out DIR [--geo-id N] [--scale S] [--mod-layout] [--keep-overlays] [--force] [--readback]
      Writes FILE.glb as the binary FBX 7.4 that Memoria's ModelImporter loads (<geo id>.fbx plus the
      texture PNGs it names), with the bones as bone000.. LimbNodes in the original FF9 order and
      one skinned mesh and material per GLB mesh. --mod-layout writes DIR/Models/<type>/<id>/ like
      a mod folder. --readback reads the FBX back, checks it against the GLB and renders both from
      the same cameras (rest and walk pose) to DIR/readback/ with a pixel diff and readback.json.";

fn unity_name(s: &str) -> String {
    s.replace(['#', ':'], "_")
}

pub fn fbx_cmd(args: &Args) -> Result<(), String> {
    let file = args.file();
    let out = PathBuf::from(args.opt("--out").ok_or("fbx needs --out DIR")?);
    let mut opts = ExportOptions::default();
    if let Some(s) = args.opt("--geo-id") {
        opts.geo_id = Some(s.parse::<u32>().map_err(|e| format!("--geo-id: {e}"))?);
    }
    if let Some(s) = args.opt("--scale") {
        let v = s.parse::<f64>().map_err(|e| format!("--scale: {e}"))?;
        if !(v.is_finite() && v > 0.0) {
            return Err("--scale must be a positive number".into());
        }
        opts.scale = Some(v);
    }
    opts.keep_overlays = args.flag("--keep-overlays");
    opts.force = args.flag("--force");

    let model = Model::open(&file)?;
    let ex = fbx::export_fbx(&model, &opts)?;

    let mut dir = out.clone();
    if args.flag("--mod-layout") {
        let t = ex.type_int.ok_or("--mod-layout needs a GLB with an ff9_geo name (GEO_MAIN_..., GEO_NPC_...) to pick the model type; write without it")?;
        let id = ex.geo_id.ok_or("--mod-layout needs a geo id: pass --geo-id N")?;
        dir = out.join("Models").join(t.to_string()).join(id.to_string());
    }
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let stem = match ex.geo_id {
        Some(id) => id.to_string(),
        None => file.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into()),
    };
    let fbx_path = dir.join(format!("{stem}.fbx"));
    std::fs::write(&fbx_path, &ex.fbx).map_err(|e| format!("{}: {e}", fbx_path.display()))?;
    for (name, bytes) in &ex.textures {
        let p = dir.join(name);
        std::fs::write(&p, bytes).map_err(|e| format!("{}: {e}", p.display()))?;
    }

    println!("{}  ({} bytes, {} bones, scale {})", fbx_path.display(), ex.fbx.len(), ex.bones, ex.scale);
    if let Some(g) = &ex.geo {
        println!("  model {g}, geo id {}, type {}", ex.geo_id.map(|i| i.to_string()).unwrap_or_else(|| "?".into()), ex.type_int.map(|t| t.to_string()).unwrap_or_else(|| "?".into()));
    }
    for m in &ex.meshes {
        println!("  mesh {}: {} vertices, {} triangles, texture {}, up to {} bone weights", m.name, m.vertices, m.triangles, m.texture.as_deref().unwrap_or("none"), m.max_influences);
    }
    for t in &ex.textures {
        println!("  texture {} ({} bytes)", t.0, t.1.len());
    }
    if !ex.skipped_meshes.is_empty() {
        println!("  skipped (battle overlay meshes; --keep-overlays keeps them): {}", ex.skipped_meshes.join(", "));
    }
    println!("  rotation round trip error {:.1e}, bind pose error {:.1e}", ex.euler_error, ex.bind_error);
    for n in &ex.notes {
        println!("  note: {n}");
    }

    if args.flag("--readback") {
        readback(&file, &out, &ex)?;
    }
    Ok(())
}

/// Reads the FBX back with Memoria's rules, rebuilds the GLB's meshes from it and renders both.
fn readback(file: &Path, out: &Path, ex: &fbx::Export) -> Result<(), String> {
    let scene = fbx::read_scene(&ex.fbx)?;
    let orig = Model::open(file)?;
    let mut back = Model::open(file)?;

    let kept: Vec<usize> = (0..orig.meshes.len()).filter(|&i| !ex.skipped_meshes.contains(&orig.meshes[i].name)).collect();
    if scene.meshes.len() != kept.len() {
        return Err(format!("read-back: the FBX has {} meshes, the GLB {} (minus skipped)", scene.meshes.len(), kept.len()));
    }
    let mut bone_joint = vec![None; scene.bones.len()];
    for (j, n) in orig.joint_names.iter().enumerate() {
        if let Some(b) = fbx::bone_number(n) {
            if b < bone_joint.len() {
                bone_joint[b] = Some(j as u16);
            }
        }
    }
    let s = ex.scale;
    // The renderer indexes textures by image, so the read-back ones go in the GLB's image slots.
    let mut textures: Vec<Option<Texture>> = (0..back.images.len()).map(|_| None).collect();
    let mut summary = Vec::new();
    for (sm, &mi) in scene.meshes.iter().zip(&kept) {
        let gm = &orig.meshes[mi];
        if sm.name != unity_name(&gm.name) {
            return Err(format!("read-back: mesh order differs ({} vs {})", sm.name, gm.name));
        }
        if sm.verts.len() != gm.pos.len() {
            return Err(format!("read-back: {} has {} vertices in the FBX, {} in the GLB", gm.name, sm.verts.len(), gm.pos.len()));
        }
        let bm = &mut back.meshes[mi];
        bm.pos = sm.verts.iter().map(|v| [(v[0] * s) as f32, (-v[1] * s) as f32, (v[2] * s) as f32]).collect();
        bm.uv = sm.uvs.iter().map(|t| [t[0] as f32, (1.0 - t[1]) as f32]).collect();
        bm.tris = sm.tris.iter().map(|t| [t[0], t[2], t[1]]).collect();
        let mut worst_sum = 0f64;
        for (vi, w) in sm.weights.iter().enumerate() {
            let mut es: Vec<(usize, f64)> = w.clone();
            es.sort_by(|a, b| b.1.total_cmp(&a.1));
            let mut j = [0u16; 4];
            let mut wt = [0f32; 4];
            for (k, e) in es.iter().take(4).enumerate() {
                j[k] = bone_joint.get(e.0).copied().flatten().ok_or_else(|| format!("read-back: bone {} has no joint in the GLB", e.0))?;
                wt[k] = e.1 as f32;
            }
            worst_sum = worst_sum.max((es.iter().map(|e| e.1).sum::<f64>() - 1.0).abs());
            bm.joints[vi] = j;
            bm.weights[vi] = wt;
        }
        if let Some(t) = &sm.texture {
            let bytes = ex.textures.iter().find(|x| &x.0 == t).map(|x| &x.1).ok_or_else(|| format!("read-back: texture {t} was not written"))?;
            let slot = gm.image.ok_or_else(|| format!("read-back: {} has a texture in the FBX but none in the GLB", gm.name))?;
            textures[slot] = Some(Texture::decode(bytes)?);
        }
        let dv = sm.verts.iter().zip(&gm.pos).map(|(v, g)| ((v[0] * s - g[0] as f64).abs()).max((-v[1] * s - g[1] as f64).abs()).max((v[2] * s - g[2] as f64).abs())).fold(0f64, f64::max);
        summary.push(json!({"mesh": gm.name, "vertices": gm.pos.len(), "triangles": gm.tris.len(), "worst_position_gap": dv, "worst_weight_sum_error": worst_sum}));
    }
    // Skipped meshes are not in the FBX: blank them in both pictures.
    let mut orig = orig;
    for i in 0..orig.meshes.len() {
        if !kept.contains(&i) {
            orig.meshes[i].tris.clear();
            back.meshes[i].tris.clear();
        }
    }
    let orig_tex = image_textures(&orig);

    let dir = out.join("readback");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let px = 600usize;
    let base = Camera::frame(&orig);
    let clip = orig.clips.iter().position(|c| c.to_lowercase().contains("walk")).or(if orig.clips.is_empty() { None } else { Some(0) });
    let t = clip.map(|c| orig.clip_duration(c) * 0.35).unwrap_or(0.0);
    let mut views: Vec<(String, f32, Option<usize>)> = vec![("front".into(), 0.0, None), ("side".into(), 90.0, None), ("back".into(), 180.0, None)];
    if let Some(c) = clip {
        views.push((format!("pose-front"), 0.0, Some(c)));
        views.push((format!("pose-side"), 90.0, Some(c)));
    }
    let mut rows: Vec<Value> = Vec::new();
    let mut worst_frac = 0f64;
    for (name, yaw, c) in &views {
        let cam = Camera { yaw: base.yaw + yaw.to_radians(), ..base };
        let (po, pb) = (orig.posed(*c, t), back.posed(*c, t));
        let a = render::render(&orig, &orig_tex, &render::Options { cam, w: px, h: px, background: crate::theme::VIEW_BG, filter: Filter::Smooth, positions: Some(&po) }, &|_, _| None).rgba;
        let b = render::render(&back, &textures, &render::Options { cam, w: px, h: px, background: crate::theme::VIEW_BG, filter: Filter::Smooth, positions: Some(&pb) }, &|_, _| None).rgba;
        let (mut bad, mut maxd) = (0usize, 0u8);
        let mut d = vec![0u8; a.len()];
        for i in 0..px * px {
            let m = (0..3).map(|k| a[i * 4 + k].abs_diff(b[i * 4 + k])).max().unwrap_or(0);
            maxd = maxd.max(m);
            if m > 24 {
                bad += 1;
                d[i * 4] = 255;
            }
            d[i * 4 + 1] = a[i * 4 + 1] / 3;
            d[i * 4 + 3] = 255;
        }
        let frac = bad as f64 / (px * px) as f64;
        worst_frac = worst_frac.max(frac);
        for (suffix, img) in [("glb", &a), ("fbx", &b), ("diff", &d)] {
            let p = dir.join(format!("{name}-{suffix}.png"));
            image::save_buffer(&p, img, px as u32, px as u32, image::ExtendedColorType::Rgba8).map_err(|e| format!("{}: {e}", p.display()))?;
        }
        println!("  view {name}: {bad} px differ by more than 24 ({:.4}%), largest channel difference {maxd}", frac * 100.0);
        rows.push(json!({"view": name, "pixels_over_24": bad, "fraction": frac, "max_channel_diff": maxd, "clip": c.map(|i| orig.clips[i].clone()), "time": if c.is_some() { t } else { 0.0 }}));
    }
    let report = json!({"fbx_meshes": summary, "views": rows, "worst_fraction": worst_frac});
    let rp = dir.join("readback.json");
    std::fs::write(&rp, serde_json::to_string_pretty(&report).unwrap()).map_err(|e| format!("{}: {e}", rp.display()))?;
    println!("  read-back pictures and {} in {}", rp.file_name().unwrap().to_string_lossy(), dir.display());
    if worst_frac > 0.001 {
        return Err(format!("read-back: the FBX renders differently from the GLB ({:.3}% of pixels); see {}", worst_frac * 100.0, dir.display()));
    }
    println!("  read-back OK: the FBX renders the same as the GLB in {} views", views.len());
    Ok(())
}
