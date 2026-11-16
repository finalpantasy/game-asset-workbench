//! Safe swap as the window and the command line use it: swap, check, and describe what changed.
use std::path::{Path, PathBuf};
use wb_core::analysis::Parts;
use wb_core::{diff, image_textures, swap, Model};

pub const GAME_ROOT: &str = r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX";

pub struct Outcome {
    pub image: String,
    pub old_size: [usize; 2],
    pub new_size: [usize; 2],
    pub old_len: usize,
    pub new_len: usize,
    pub alpha_flips: usize,
    pub changed_pixels: usize,
    pub total_pixels: usize,
    /// The new PNG is a different size, so the pixel comparison ran on a scaled copy.
    pub resized: bool,
    /// Changed pixels per kit part, largest first.
    pub by_part: Vec<(String, usize)>,
    /// Triangles that show changed pixels, per kit part, largest first.
    pub shows_on: Vec<(String, usize)>,
    pub bytes: Vec<u8>,
    /// The swapped texture's pixel diff.
    pub tex: Option<diff::TexDiff>,
    /// Per mesh, per triangle of the new file: shows changed pixels.
    pub changed: Vec<Vec<bool>>,
}

/// Where the new file may go: not over the input, not over an existing file,
/// and not into the installed game (only under `Modding`).
pub fn check_out(input: &Path, out: &Path) -> Result<PathBuf, String> {
    if out.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() != Some("glb") {
        return Err("the new file must end in .glb".into());
    }
    if out.exists() {
        return Err(format!("{} already exists; pick a new name (files are never overwritten)", out.display()));
    }
    let dir = out.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let dir = dir.canonicalize().map_err(|e| format!("{}: {e}", dir.display()))?;
    let dir = PathBuf::from(dir.to_string_lossy().trim_start_matches(r"\\?\"));
    let full = dir.join(out.file_name().ok_or("no file name")?);
    if input.canonicalize().map(|i| i == full).unwrap_or(false) {
        return Err("the new file cannot replace the input".into());
    }
    let low = |p: &Path| p.to_string_lossy().trim_start_matches(r"\\?\").to_ascii_lowercase();
    let (f, root) = (low(&full), GAME_ROOT.to_ascii_lowercase());
    if f.starts_with(&root) && !f.starts_with(&format!("{root}\\modding\\")) {
        return Err("the new file would land in the installed game; save it under Modding instead".into());
    }
    Ok(full)
}

/// Swaps, checks, and describes the change. Also returns the new file opened as a model.
pub fn run(model: &Model, image: &str, png: &[u8], parts: Option<&Parts>) -> Result<(Outcome, Model), String> {
    let s = swap::swap(model, image, png)?;
    let b = Model::from_bytes(&s.bytes)?;
    let mut d = diff::diff(model, &image_textures(model), &b, &image_textures(&b), parts, 0);
    let name = |id: u8| parts.map(|p| p.name(id).to_owned()).unwrap_or_else(|| format!("part {id}"));
    let changed = diff::changed_triangles(&b, &d);
    let ti = d.textures.iter().position(|t| t.name.trim_end_matches(".png") == s.image.trim_end_matches(".png"));
    let tex = ti.map(|i| d.textures.swap_remove(i));
    let t = tex.as_ref();
    let mut shows: Vec<(String, usize)> = Vec::new();
    if let Some(p) = parts {
        for (mi, ch) in changed.iter().enumerate() {
            let tp = p.triangle_parts(&b, mi);
            for (ti, &c) in ch.iter().enumerate() {
                if c {
                    let n = if tp[ti] == 0 { "outside any part".to_owned() } else { name(tp[ti]) };
                    match shows.iter_mut().find(|e| e.0 == n) {
                        Some(e) => e.1 += 1,
                        None => shows.push((n, 1)),
                    }
                }
            }
        }
        shows.sort_by(|a, b| b.1.cmp(&a.1));
    }
    let by_part = t.map(|t| t.by_part.iter().map(|(id, n)| (name(*id), *n)).collect()).unwrap_or_default();
    let (changed_pixels, total_pixels, resized) = (t.map_or(0, |t| t.changed), t.map_or(0, |t| t.total), t.is_some_and(|t| t.resized));
    let o = Outcome {
        image: s.image.clone(),
        old_size: s.old_size,
        new_size: s.new_size,
        old_len: s.old_len,
        new_len: s.new_len,
        alpha_flips: s.alpha_flips,
        changed_pixels,
        total_pixels,
        resized,
        by_part,
        shows_on: shows,
        bytes: s.bytes,
        tex,
        changed,
    };
    Ok((o, b))
}
