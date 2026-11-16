//! Safe swap: put a painted PNG in place of one embedded texture and change nothing else.
//!
//! The new PNG's bytes go where the old image's bytes were. Buffer views after it move by a
//! multiple of 4 so every accessor keeps its alignment. In the JSON chunk only the `bufferViews`
//! and `buffers` arrays are rewritten (their offsets and lengths); the rest of the text is kept as is.
use crate::glb::{Glb, CHUNK_BIN, CHUNK_JSON};
use crate::{Model, Texture};
use serde_json::Value;
use std::ops::Range;

pub struct Swapped {
    pub bytes: Vec<u8>,
    pub image: String,
    pub old_size: [usize; 2],
    pub new_size: [usize; 2],
    pub old_len: usize,
    pub new_len: usize,
    /// How far the buffer views after the image moved.
    pub shift: i64,
    pub views_moved: usize,
    /// Pixels whose alpha cutout (alpha < 128) flips, counted at the old texture's size.
    pub alpha_flips: usize,
}

/// The byte range of the value stored under a top-level `key` in compact or spaced JSON text.
fn json_span(text: &[u8], key: &str) -> Option<Range<usize>> {
    let pat = format!("\"{key}\"");
    let mut depth = 0i32;
    let mut i = 0;
    let mut in_str = false;
    while i < text.len() {
        let c = text[i];
        if in_str {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' => {
                if depth == 1 && text[i..].starts_with(pat.as_bytes()) {
                    let mut j = i + pat.len();
                    while j < text.len() && (text[j] == b' ' || text[j] == b'\n' || text[j] == b'\r' || text[j] == b'\t' || text[j] == b':') {
                        j += 1;
                    }
                    let start = j;
                    let (mut d, mut s) = (0i32, false);
                    while j < text.len() {
                        let c = text[j];
                        if s {
                            if c == b'\\' {
                                j += 1;
                            } else if c == b'"' {
                                s = false;
                            }
                        } else {
                            match c {
                                b'"' => s = true,
                                b'[' | b'{' => d += 1,
                                b']' | b'}' => {
                                    d -= 1;
                                    if d == 0 {
                                        return Some(start..j + 1);
                                    }
                                }
                                _ => {}
                            }
                        }
                        j += 1;
                    }
                    return None;
                }
                in_str = true;
            }
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth -= 1,
            _ => {}
        }
        i += 1;
    }
    None
}

fn set_u64(v: &mut Value, k: &str, x: u64) {
    if let Some(o) = v.as_object_mut() {
        o.insert(k.into(), Value::from(x));
    }
}

/// Replaces image `image` (by name, `.png` optional) of `model` with `png` and returns the new file.
pub fn swap(model: &Model, image: &str, png: &[u8]) -> Result<Swapped, String> {
    let want = image.trim_end_matches(".png");
    let ii = model.images.iter().position(|i| i.name.trim_end_matches(".png") == want).ok_or_else(|| {
        format!("no image named {image}; images: {}", model.images.iter().map(|i| i.name.as_str()).collect::<Vec<_>>().join(", "))
    })?;
    let im = &model.images[ii];
    if !im.mime.is_empty() && im.mime != "image/png" {
        return Err(format!("image {} is {}, only PNG images can be swapped", im.name, im.mime));
    }
    if !png.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err("the new file is not a PNG".into());
    }
    let new_tex = Texture::decode(png).map_err(|e| format!("the new PNG does not decode: {e}"))?;
    let old_tex = Texture::decode(&im.bytes).map_err(|e| format!("the old image does not decode: {e}"))?;
    let (ow, oh, nw, nh) = (old_tex.w, old_tex.h, new_tex.w, new_tex.h);
    if ow * nh != oh * nw {
        return Err(format!("the new PNG is {nw}x{nh} but {} is {ow}x{oh}: a different shape would stretch the paint on the model", im.name));
    }
    let mut alpha_flips = 0;
    for y in 0..oh {
        for x in 0..ow {
            let a = old_tex.rgba[(y * ow + x) * 4 + 3] < 128;
            let (sx, sy) = (x * nw / ow, y * nh / oh);
            let b = new_tex.rgba[(sy * nw + sx) * 4 + 3] < 128;
            alpha_flips += (a != b) as usize;
        }
    }

    let glb = &model.glb;
    let views = glb.list("bufferViews");
    let v = &views[im.view];
    let off = v.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
    let old_len = v.get("byteLength").and_then(Value::as_u64).unwrap_or(0) as usize;
    let end = off + old_len;
    for (k, w) in views.iter().enumerate() {
        if k == im.view {
            continue;
        }
        let o = w.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let l = w.get("byteLength").and_then(Value::as_u64).unwrap_or(0) as usize;
        if o < end && off < o + l {
            return Err(format!("buffer view {k} overlaps the image's bytes; refusing to swap"));
        }
    }
    let users = glb.list("images").iter().filter(|x| x.get("bufferView").and_then(Value::as_u64) == Some(im.view as u64)).count()
        + glb.list("accessors").iter().filter(|x| x.get("bufferView").and_then(Value::as_u64) == Some(im.view as u64)).count();
    if users != 1 {
        return Err(format!("the image's buffer view is used {users} times; refusing to swap"));
    }

    let new_len = png.len();
    let pad = (4 - (new_len as i64 - old_len as i64).rem_euclid(4)) % 4;
    let shift = new_len as i64 + pad - old_len as i64;
    let bin = glb.bin();
    let mut new_bin = Vec::with_capacity((bin.len() as i64 + shift) as usize);
    new_bin.extend_from_slice(&bin[..off]);
    new_bin.extend_from_slice(png);
    new_bin.extend(std::iter::repeat_n(0u8, pad as usize));
    new_bin.extend_from_slice(&bin[end..]);

    let mut new_views: Vec<Value> = views.to_vec();
    let mut moved = 0;
    for (k, w) in new_views.iter_mut().enumerate() {
        if k == im.view {
            set_u64(w, "byteLength", new_len as u64);
            continue;
        }
        let o = w.get("byteOffset").and_then(Value::as_u64).unwrap_or(0);
        if o as usize >= end {
            set_u64(w, "byteOffset", (o as i64 + shift) as u64);
            moved += 1;
        }
    }
    let mut buffers: Vec<Value> = glb.list("buffers").to_vec();
    if let Some(b0) = buffers.first_mut() {
        let l = b0.get("byteLength").and_then(Value::as_u64).unwrap_or(bin.len() as u64);
        set_u64(b0, "byteLength", (l as i64 + shift) as u64);
    }

    let jc = glb.chunks.first().filter(|c| c.kind == CHUNK_JSON).ok_or("first chunk is not JSON")?;
    let mut text = jc.data.clone();
    while text.last().is_some_and(|&c| c == b' ' || c == 0) {
        text.pop();
    }
    let mut edits: Vec<(Range<usize>, Vec<u8>)> = vec![
        (json_span(&text, "bufferViews").ok_or("JSON has no bufferViews")?, serde_json::to_vec(&new_views).map_err(|e| e.to_string())?),
        (json_span(&text, "buffers").ok_or("JSON has no buffers")?, serde_json::to_vec(&buffers).map_err(|e| e.to_string())?),
    ];
    edits.sort_by(|a, b| b.0.start.cmp(&a.0.start));
    for (r, with) in edits {
        text.splice(r, with);
    }
    while text.len() % 4 != 0 {
        text.push(b' ');
    }

    let mut out = Glb { version: glb.version, chunks: Vec::new(), json: Value::Null };
    for c in &glb.chunks {
        let data = match c.kind {
            CHUNK_JSON => text.clone(),
            CHUNK_BIN => new_bin.clone(),
            _ => c.data.clone(),
        };
        out.chunks.push(crate::glb::Chunk { kind: c.kind, data });
    }
    let bytes = out.to_bytes();
    let sw = Swapped { bytes, image: im.name.clone(), old_size: [ow, oh], new_size: [nw, nh], old_len, new_len, shift, views_moved: moved, alpha_flips };
    verify(model, &sw.bytes, ii, png)?;
    Ok(sw)
}

/// Checks a swapped file against its source: the image holds exactly `png`, every other buffer
/// view holds the same bytes, and the JSON is the same apart from view offsets and lengths.
pub fn verify(src: &Model, out: &[u8], image: usize, png: &[u8]) -> Result<(), String> {
    let g = Glb::parse(out)?;
    let s = &src.glb;
    if g.chunks.len() != s.chunks.len() {
        return Err("chunk count changed".into());
    }
    let iv = src.images[image].view;
    let n = s.list("bufferViews").len();
    if g.list("bufferViews").len() != n {
        return Err("buffer view count changed".into());
    }
    for k in 0..n {
        let b = g.view_bytes(k)?;
        if k == iv {
            if b != png {
                return Err("the image's bytes are not the new PNG".into());
            }
        } else if b != s.view_bytes(k)? {
            return Err(format!("buffer view {k} changed"));
        }
    }
    let strip = |j: &Value| {
        let mut j = j.clone();
        if let Some(o) = j.as_object_mut() {
            o.remove("bufferViews");
            o.remove("buffers");
        }
        j
    };
    if strip(&g.json) != strip(&s.json) {
        return Err("JSON changed outside bufferViews and buffers".into());
    }
    for (k, (a, b)) in s.list("bufferViews").iter().zip(g.list("bufferViews")).enumerate() {
        let (mut a, mut b) = (a.clone(), b.clone());
        for x in [&mut a, &mut b] {
            if let Some(o) = x.as_object_mut() {
                o.remove("byteOffset");
                o.remove("byteLength");
            }
        }
        if a != b {
            return Err(format!("buffer view {k} changed more than its offset or length"));
        }
    }
    let m = Model::from_bytes(out)?;
    if m.meshes.len() != src.meshes.len() || m.meshes.iter().zip(&src.meshes).any(|(a, b)| a.pos != b.pos || a.uv != b.uv || a.tris != b.tris || a.joints != b.joints) {
        return Err("a mesh changed".into());
    }
    Ok(())
}
