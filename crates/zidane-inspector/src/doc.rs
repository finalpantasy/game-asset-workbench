//! Opened files and comparisons, built on worker threads so the window never freezes.
use eframe::egui::{self, ColorImage};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use wb_core::analysis::{self, Parts};
use wb_core::diff::{self, ModelDiff};
use wb_core::{image_textures, uv, Model, Texture};

pub struct Doc {
    pub path: PathBuf,
    pub label: String,
    pub model: Model,
    /// Decoded textures, one per image in the file.
    pub textures: Vec<Option<Texture>>,
    /// Texture previews for the panel (at most 2048 px), one per image.
    pub previews: Vec<Option<ColorImage>>,
    pub tri_parts: Vec<Vec<u8>>,
    pub dead: Vec<Vec<bool>>,
    pub dead_regions: Vec<(String, usize)>,
    /// Islands, density, overlap and part coverage per triangle.
    pub uv: uv::Facts,
    /// Per image: texels that show on two surfaces, colored by kind (at `uv::OVERLAP_SIZE`).
    pub overlap_previews: Vec<Option<ColorImage>>,
    pub load_ms: u128,
}

pub const REUSE_COLORS: [[u8; 4]; 3] = [[80, 150, 255, 150], [255, 165, 60, 190], [235, 70, 70, 210]];

fn overlap_preview(model: &Model, facts: &uv::Facts, image: usize) -> Option<ColorImage> {
    let s = uv::OVERLAP_SIZE;
    let mut kind = vec![0u8; s * s];
    let mut any = false;
    for (mi, m) in model.meshes.iter().enumerate() {
        if m.image == Some(image) {
            any = true;
            for (k, &x) in kind.iter_mut().zip(&facts.overlap[mi].kind) {
                *k = (*k).max(x);
            }
        }
    }
    if !any {
        return None;
    }
    let mut px = vec![0u8; s * s * 4];
    for (i, &k) in kind.iter().enumerate() {
        if k > 0 {
            px[i * 4..i * 4 + 4].copy_from_slice(&REUSE_COLORS[k as usize - 1]);
        }
    }
    Some(ColorImage::from_rgba_unmultiplied([s, s], &px))
}

impl Doc {
    pub fn dead_total(&self) -> usize {
        self.dead_regions.iter().map(|r| r.1).sum()
    }
    pub fn file_name(&self) -> String {
        self.path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()
    }
    pub fn tex_size(&self) -> Option<[usize; 2]> {
        self.textures.iter().flatten().map(|t| [t.w, t.h]).max()
    }
    pub fn clip_index(&self, name: &str) -> Option<usize> {
        self.model.clips.iter().position(|c| c == name)
    }
}

pub fn preview(t: &Texture) -> ColorImage {
    let mut lvl = 0;
    while lvl + 1 < t.levels() && t.level(lvl).0.max(t.level(lvl).1) > 2048 {
        lvl += 1;
    }
    let (w, h, px) = t.level(lvl);
    ColorImage::from_rgba_unmultiplied([w, h], px)
}

pub fn load(path: &Path, label: &str, parts: Option<&Parts>) -> Result<Doc, String> {
    let t0 = std::time::Instant::now();
    from_model(Model::open(path)?, path, label, parts, t0)
}

fn from_model(model: Model, path: &Path, label: &str, parts: Option<&Parts>, t0: std::time::Instant) -> Result<Doc, String> {
    let textures = image_textures(&model);
    let previews = textures.iter().map(|t| t.as_ref().map(preview)).collect();
    let tri_parts = (0..model.meshes.len())
        .map(|mi| match parts {
            Some(p) => p.triangle_parts(&model, mi),
            None => vec![0; model.meshes[mi].tris.len()],
        })
        .collect();
    let dead: Vec<Vec<bool>> = (0..model.meshes.len()).map(|mi| (0..model.meshes[mi].tris.len()).map(|ti| analysis::is_dead(&model, mi, ti)).collect()).collect();
    let mut regions: Vec<(String, usize)> = Vec::new();
    for (mi, d) in dead.iter().enumerate() {
        for (ti, &x) in d.iter().enumerate() {
            if x {
                let bone = model.joint_names.get(model.triangle_joint(mi, ti)).cloned().unwrap_or_default();
                let r = analysis::region(&bone);
                match regions.iter_mut().find(|e| e.0 == r) {
                    Some(e) => e.1 += 1,
                    None => regions.push((r, 1)),
                }
            }
        }
    }
    regions.sort_by(|a, b| b.1.cmp(&a.1));
    let sizes: Vec<usize> = model.meshes.iter().map(|m| m.image.and_then(|i| textures[i].as_ref()).map(|t: &Texture| t.w.max(t.h)).unwrap_or(512)).collect();
    let facts = uv::Facts::build(&model, parts, &sizes);
    let overlap_previews = (0..model.images.len()).map(|i| overlap_preview(&model, &facts, i)).collect();
    Ok(Doc { path: path.to_path_buf(), label: label.to_owned(), model, textures, previews, tri_parts, dead, dead_regions: regions, uv: facts, overlap_previews, load_ms: t0.elapsed().as_millis() })
}

/// A comparison of two docs, with everything the window draws from it.
pub struct DiffView {
    pub a: Arc<Doc>,
    pub b: Arc<Doc>,
    pub tol: u8,
    pub diff: ModelDiff,
    pub changed_a: Vec<Vec<bool>>,
    pub changed_b: Vec<Vec<bool>>,
    /// Per texture diff: B's texture dimmed, changed pixels in magenta (at most 2048 px).
    pub overlays: Vec<Option<ColorImage>>,
    /// Per texture: kit parts with no changed pixel, vertex or UV.
    pub untouched: Vec<Vec<u8>>,
    pub ms: u128,
}

impl DiffView {
    pub fn matches(&self, a: &Arc<Doc>, b: &Arc<Doc>, tol: u8) -> bool {
        Arc::ptr_eq(&self.a, a) && Arc::ptr_eq(&self.b, b) && self.tol == tol
    }
}

/// `base` dimmed to grey, changed pixels in magenta, at most 2048 px.
pub fn change_overlay(td: &diff::TexDiff, base: &Texture) -> Option<ColorImage> {
    let [w, h] = td.size;
    if w == 0 || h == 0 {
        return None;
    }
    let step = w.max(h).div_ceil(2048).max(1);
    let (ow, oh) = (w.div_ceil(step), h.div_ceil(step));
    let mut px = vec![0u8; ow * oh * 4];
    for y in 0..oh {
        for x in 0..ow {
            let mut hit = false;
            'blk: for yy in y * step..((y + 1) * step).min(h) {
                for xx in x * step..((x + 1) * step).min(w) {
                    if td.mask[yy * w + xx] != 0 {
                        hit = true;
                        break 'blk;
                    }
                }
            }
            let sx = (x * step) * base.w / w;
            let sy = (y * step) * base.h / h;
            let s = &base.rgba[(sy * base.w + sx) * 4..(sy * base.w + sx) * 4 + 4];
            let o = (y * ow + x) * 4;
            if hit {
                px[o..o + 4].copy_from_slice(&[255, 79, 216, 255]);
            } else {
                let g = ((s[0] as u32 * 3 + s[1] as u32 * 5 + s[2] as u32 * 2) / 10) as u8;
                let v = (g as f32 * 0.35) as u8 + 12;
                px[o..o + 4].copy_from_slice(&[v, v, v + 2, 255]);
            }
        }
    }
    Some(ColorImage::from_rgba_unmultiplied([ow, oh], &px))
}

/// A finished safe swap, ready to show and save.
pub struct SwapResult {
    pub base: Arc<Doc>,
    pub png: PathBuf,
    pub outcome: crate::swapjob::Outcome,
    /// The new file, opened.
    pub doc: Arc<Doc>,
    /// The swapped image's index in the file.
    pub image: usize,
    pub overlay: Option<ColorImage>,
}

pub fn spawn_swap(tx: Sender<Msg>, ctx: egui::Context, base: Arc<Doc>, image: String, png: PathBuf, parts: Option<Arc<Parts>>) {
    std::thread::spawn(move || {
        let t0 = std::time::Instant::now();
        let r = (|| -> Result<Arc<SwapResult>, String> {
            let bytes = std::fs::read(&png).map_err(|e| format!("{}: {e}", png.display()))?;
            let (outcome, model) = crate::swapjob::run(&base.model, &image, &bytes, parts.as_deref())?;
            let label = format!("{} + {}", base.label, png.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default());
            let doc = Arc::new(from_model(model, &base.path, &label, parts.as_deref(), t0)?);
            let ii = doc.model.images.iter().position(|i| i.name == outcome.image).unwrap_or(0);
            let overlay = outcome.tex.as_ref().zip(doc.textures[ii].as_ref()).and_then(|(td, t)| change_overlay(td, t));
            Ok(Arc::new(SwapResult { base, png: png.clone(), outcome, doc, image: ii, overlay }))
        })();
        let _ = tx.send(Msg::Swapped(png, r));
        ctx.request_repaint();
    });
}

pub fn compare(a: Arc<Doc>, b: Arc<Doc>, parts: Option<&Parts>, tol: u8) -> DiffView {
    let t0 = std::time::Instant::now();
    let d = diff::diff(&a.model, &a.textures, &b.model, &b.textures, parts, tol);
    let changed_a = diff::changed_triangles(&a.model, &d);
    let changed_b = diff::changed_triangles(&b.model, &d);
    let overlays = d
        .textures
        .iter()
        .map(|td| {
            let base = b.model.images.iter().position(|i| i.name.trim_end_matches(".png") == td.name).and_then(|i| b.textures[i].as_ref())
                .or_else(|| a.model.images.iter().position(|i| i.name.trim_end_matches(".png") == td.name).and_then(|i| a.textures[i].as_ref()))?;
            change_overlay(td, base)
        })
        .collect();
    let untouched = match parts {
        Some(p) => diff::untouched_parts(
            &d,
            p,
            [
                diff::Side { model: &a.model, changed: &changed_a, tri_parts: &a.tri_parts },
                diff::Side { model: &b.model, changed: &changed_b, tri_parts: &b.tri_parts },
            ],
        ),
        None => vec![vec![]; d.textures.len()],
    };
    DiffView { a, b, tol, diff: d, changed_a, changed_b, overlays, untouched, ms: t0.elapsed().as_millis() }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Slot {
    Inspect,
    A,
    B,
}

pub enum Msg {
    Loaded(Slot, PathBuf, Result<Arc<Doc>, String>),
    Compared(Arc<DiffView>),
    Swapped(PathBuf, Result<Arc<SwapResult>, String>),
    RelayoutProgress(String),
    Relayouted(Result<Arc<RelayoutResult>, String>),
}

/// A finished UV re-layout and its checks.
pub struct RelayoutResult {
    pub input: PathBuf,
    pub dir: PathBuf,
    pub glb: PathBuf,
    pub checks: wb_core::relayout::Checks,
    /// Sewing-pattern layouts: pieces, seams and the pattern sheets written beside the model.
    pub pattern: Option<(usize, usize, Vec<PathBuf>)>,
}

fn pattern_info(dir: &std::path::Path) -> Option<(usize, usize, Vec<PathBuf>)> {
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("legend.json")).ok()?).ok()?;
    let n = |k: &str| j[k].as_array().map(|a| a.len()).unwrap_or(0);
    let mut sheets: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.to_string_lossy().ends_with("_pattern.png")).collect();
    sheets.sort();
    Some((n("pieces"), n("seams"), sheets))
}

/// Runs a re-layout into `dir` (an empty folder), or, when `dir` already holds a finished one,
/// only re-checks it. `blender` picks the Blender layout; otherwise the built-in sewing pattern.
pub fn spawn_relayout(tx: Sender<Msg>, ctx: egui::Context, input: PathBuf, dir: PathBuf, size: usize, blender: bool) {
    std::thread::spawn(move || {
        let r = (|| -> Result<Arc<RelayoutResult>, String> {
            let (glb, checks) = if dir.join("report.json").is_file() {
                let _ = tx.send(Msg::RelayoutProgress("Checking the finished layout".into()));
                let glb = crate::relayoutjob::output_of(&dir)?;
                let c = crate::relayoutjob::check_files(&input, &glb)?;
                (glb, c)
            } else {
                let (ptx, pctx) = (tx.clone(), ctx.clone());
                let progress = Box::new(move |l: String| {
                    let _ = ptx.send(Msg::RelayoutProgress(l));
                    pctx.request_repaint();
                });
                if blender {
                    crate::relayoutjob::run(&input, &dir, size, None, progress)?
                } else {
                    crate::patternjob::run(&input, &dir, size, Vec::new(), progress)?
                }
            };
            let pattern = pattern_info(&dir);
            Ok(Arc::new(RelayoutResult { input, dir, glb, checks, pattern }))
        })();
        let _ = tx.send(Msg::Relayouted(r));
        ctx.request_repaint();
    });
}

pub fn spawn_load(tx: Sender<Msg>, ctx: egui::Context, slot: Slot, path: PathBuf, label: String, parts: Option<Arc<Parts>>) {
    std::thread::spawn(move || {
        let r = load(&path, &label, parts.as_deref()).map(Arc::new);
        let _ = tx.send(Msg::Loaded(slot, path, r));
        ctx.request_repaint();
    });
}

pub fn spawn_compare(tx: Sender<Msg>, ctx: egui::Context, a: Arc<Doc>, b: Arc<Doc>, parts: Option<Arc<Parts>>, tol: u8) {
    std::thread::spawn(move || {
        let v = compare(a, b, parts.as_deref(), tol);
        let _ = tx.send(Msg::Compared(Arc::new(v)));
        ctx.request_repaint();
    });
}

/// Files the window offers by name.
pub struct Known {
    pub label: &'static str,
    pub about: &'static str,
    pub path: PathBuf,
}

pub fn known_files() -> Vec<Known> {
    let root = PathBuf::from(r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding");
    let x = |p: &str| root.join(p);
    [
        ("Stock Zidane", "The game's own model and 512 textures", x(r"exports\Zidane-idle.glb")),
        ("Fabric v26 C", "Chosen fabric weave, 4K textures", x(r"experiments\stock-zidane-hd-v1\fabric-v26\c\Zidane-v26-c.glb")),
        ("Finish v29", "Garment finish baseline, 4K", x(r"experiments\stock-zidane-hd-v1\finish-v29\Zidane-v29-garment-finish.glb")),
        ("Finish v29b", "v29 with the waist linen; later passes build on this", x(r"experiments\stock-zidane-hd-v1\finish-v29\Zidane-v29b-garment-finish.glb")),
        ("Gloves v5 (accepted)", "Accepted glove paint with the thumb UV repair", x(r"experiments\stock-zidane-hd-v1\models-v5\Zidane-gloves-thumb-uv-repair.glb")),
        ("Gloves v5, paint only", "Same glove paint on the stock UVs", x(r"experiments\stock-zidane-hd-v1\models-v5\Zidane-texture-only.glb")),
        ("Belt v53d", "Belt repaint with the native mask", x(r"experiments\stock-zidane-hd-v1\belt-v53-native-mask\d\Zidane-v53d-native-belt.glb")),
        ("Hair v6j", "Hair seam cleanup; splits the head into 3 pieces", x(r"experiments\zidane-hair-reference-v6\models\Zidane-hair-seam-cleanup-v6j.glb")),
    ]
    .into_iter()
    .filter(|k| k.2.exists())
    .map(|(label, about, path)| Known { label, about, path })
    .collect()
}

pub fn label_for(path: &Path, known: &[Known]) -> String {
    known.iter().find(|k| k.path == path).map(|k| k.label.to_owned()).unwrap_or_else(|| path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default())
}
