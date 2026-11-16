//! Sewing-pattern UV layout, all in Rust (wb-core `pattern`): cut by body part, flatten, straighten,
//! pair left/right, pack, copy the paint, then write the model, the pattern sheet and AI maps, and
//! check the result against the input like any re-layout.
use crate::relayoutjob::{check_dir, check_files_keep, to_json};
use serde_json::json;
use std::path::{Path, PathBuf};
use wb_core::pattern::{self, Keep, MirrorMode, Options};
use wb_core::patternx;
use wb_core::relayout::Checks;
use wb_core::{image_textures, rebuild, swap, Model};

/// Guide images are at most this size (the sheet is for reading, not for the game).
const GUIDE_MAX: usize = 2048;

fn png(w: usize, h: usize, rgba: Vec<u8>) -> Result<Vec<u8>, String> {
    let mut out = std::io::Cursor::new(Vec::new());
    image::RgbaImage::from_raw(w as u32, h as u32, rgba).ok_or("bad image size")?.write_to(&mut out, image::ImageFormat::Png).map_err(|e| e.to_string())?;
    Ok(out.into_inner())
}

fn write(path: PathBuf, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Switches beyond the default layout. All off by default: the default output does not change.
#[derive(Clone, Default)]
pub struct Extra {
    /// Turn each piece so its dominant bone runs along the page's vertical axis.
    pub orient_bones: bool,
    /// Left/right pairs: find only (`Report`) or also lay out as exact reflections (`Lock`).
    pub mirror: Option<MirrorMode>,
    /// Per-piece density error, anisotropy and bone-axis numbers added to the checks.
    pub metrics: bool,
    /// Chart, triangle and bone ID maps and a paint-kit manifest.
    pub id_maps: bool,
    /// Density factor for the head and hands (replaces 1.5 and 1.75).
    pub face_hand_density: Option<f64>,
    /// Size of the guide and ID maps (default: the sheet size, at most 2048). The layout does not change.
    pub guide_size: Option<usize>,
    /// Target for every piece's anisotropy p95: relax pieces above it, cut those that stay above.
    pub low_stretch: Option<f64>,
}

impl Extra {
    /// `--v8` switches on all four additions; the others are `--orient-bones`, `--mirror-pairs`,
    /// `--mirror-report`, `--metrics`, `--id-maps` and `--face-hand-density=X`.
    pub fn from_args(args: &crate::Args) -> Result<Extra, String> {
        let v9 = args.flag("--v9");
        let v8 = args.flag("--v8") || v9;
        let mut x = Extra { orient_bones: v8 || args.flag("--orient-bones"), metrics: v8 || args.flag("--metrics"), id_maps: v8 || args.flag("--id-maps"), ..Extra::default() };
        if v8 || args.flag("--mirror-pairs") {
            x.mirror = Some(MirrorMode::Lock);
        } else if args.flag("--mirror-report") {
            x.mirror = Some(MirrorMode::Report);
        }
        if let Some(v) = args.0.iter().find_map(|a| a.strip_prefix("--face-hand-density=")) {
            let d: f64 = v.parse().map_err(|e| format!("--face-hand-density={v}: {e}"))?;
            if !(0.25..=4.0).contains(&d) {
                return Err(format!("--face-hand-density={v}: pick a factor between 0.25 and 4"));
            }
            x.face_hand_density = Some(d);
        }
        if v9 || args.flag("--low-stretch") {
            x.low_stretch = Some(2.0);
        }
        if let Some(v) = args.0.iter().find_map(|a| a.strip_prefix("--low-stretch=")) {
            let t: f64 = v.parse().map_err(|e| format!("--low-stretch={v}: {e}"))?;
            if !(1.1..=10.0).contains(&t) {
                return Err(format!("--low-stretch={v}: pick a target between 1.1 and 10"));
            }
            x.low_stretch = Some(t);
        }
        if let Some(v) = args.0.iter().find_map(|a| a.strip_prefix("--guide-size=")) {
            let g: usize = v.parse().map_err(|e| format!("--guide-size={v}: {e}"))?;
            if !g.is_power_of_two() || !(256..=8192).contains(&g) {
                return Err(format!("--guide-size={v}: pick a power of two from 256 to 8192"));
            }
            x.guide_size = Some(g);
        }
        Ok(x)
    }
}

/// `keep`: stock boxes to leave in place (see `pattern::Keep`), for example a blink's source and target.
pub fn run(input: &Path, out: &Path, size: usize, keep: Vec<Keep>, progress: Box<dyn FnMut(String) + Send>) -> Result<(PathBuf, Checks), String> {
    run_with(input, out, size, keep, &Extra::default(), progress)
}

fn sha256(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[i * 4], chunk[i * 4 + 1], chunk[i * 4 + 2], chunk[i * 4 + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

/// `run` with the additions in `x`: bone-axis orientation, mirrored pairs, metrics, ID maps.
pub fn run_with(input: &Path, out: &Path, size: usize, keep: Vec<Keep>, x: &Extra, mut progress: Box<dyn FnMut(String) + Send>) -> Result<(PathBuf, Checks), String> {
    let out = check_dir(out)?;
    let model = Model::open(input)?;
    // The rebuild carries COLOR_0 through, and the field shader multiplies it into the texture (see bake::SHADING_POLICY).
    for w in wb_core::bake::shading_check(&model.glb, &[])["warnings"].as_array().cloned().unwrap_or_default() {
        progress(format!("SHADING WARNING: {}", w.as_str().unwrap_or("")));
    }
    progress("[1/5] Cutting by body part and flattening".into());
    let mut o = Options { size, guide: size.min(GUIDE_MAX), keep, ..Options::default() };
    o.orient_bones = x.orient_bones;
    o.low_stretch = x.low_stretch;
    // The metrics list the pairs found, so they need the corner match even when nothing is locked.
    o.mirror = x.mirror.unwrap_or(if x.metrics { MirrorMode::Report } else { MirrorMode::Off });
    if let Some(d) = x.face_hand_density {
        o.head_density = d;
        o.hand_density = d;
    }
    let mut lay = pattern::layout(&model, &o)?;
    progress(format!("[2/5] {} pieces, {} seams; writing the model", lay.pieces.len(), lay.seams.len()));
    let mut bytes = rebuild::rebuild(&model, &lay.meshes)?;
    let textures = image_textures(&model);
    let stem = input.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
    let gs = x.guide_size.unwrap_or(size.min(GUIDE_MAX));
    for ai in 0..lay.atlases.len() {
        let (image, name) = (lay.atlases[ai].image, lay.atlases[ai].name.clone());
        let Some(Some(src)) = textures.get(image) else { continue };
        progress(format!("[3/5] Copying the paint of {name} to {size} x {size}"));
        let (mut rgba, covered) = pattern::transfer(&model, &lay, &lay.atlases[ai], src, size, 16);
        lay.atlases[ai].coverage = covered;
        let kept = pattern::copy_keep(&o, &lay.atlases[ai], src, &mut rgba, size);
        if kept > 0 {
            progress(format!("      kept {kept} texels of stock paint in place on {name}"));
        }
        // The guide sheet shows the paint at guide size (nearest texel is enough to read it).
        let small: Vec<u8> = if gs == size {
            rgba.clone()
        } else {
            let mut s = vec![0u8; gs * gs * 4];
            for y in 0..gs {
                for x in 0..gs {
                    let j = ((y * size / gs) * size + x * size / gs) * 4;
                    s[(y * gs + x) * 4..(y * gs + x) * 4 + 4].copy_from_slice(&rgba[j..j + 4]);
                }
            }
            s
        };
        let p = png(size, size, rgba)?;
        write(out.join(format!("{name}.png")), &p)?;
        let swapped = swap::swap(&Model::from_bytes(&bytes)?, &name, &p)?;
        bytes = swapped.bytes;
        progress(format!("[4/5] Drawing the pattern sheet for {name}"));
        let g = pattern::guides(&model, &lay, &lay.atlases[ai], Some((&small, gs)), gs);
        write(out.join(format!("{name}_pattern.png")), &png(gs, gs, g.sheet)?)?;
        write(out.join(format!("{name}_parts.png")), &png(gs, gs, g.parts)?)?;
        write(out.join(format!("{name}_position.png")), &png(gs, gs, g.position)?)?;
        write(out.join(format!("{name}_normal.png")), &png(gs, gs, g.normal)?)?;
        if x.id_maps {
            progress(format!("      writing chart, triangle and bone ID maps for {name}"));
            let ids = patternx::id_maps(&model, &lay, &lay.atlases[ai], gs);
            for (suffix, img) in [("chart_id", ids.chart), ("triangle_id", ids.triangle), ("bone_id", ids.bone), ("chart_preview", ids.chart_preview)] {
                let file = format!("{name}_{suffix}.png");
                write(out.join(&file), &png(gs, gs, img)?)?;
            }
        }
    }
    let glb = out.join(format!("{stem}-pattern.glb"));
    write(glb.clone(), &bytes)?;
    write(out.join("legend.json"), serde_json::to_string_pretty(&pattern::legend(&lay)).unwrap().as_bytes())?;
    progress("[5/5] Checking against the input".into());
    let keep: Vec<(String, [f64; 4])> = o.keep.iter().map(|k| (k.sheet.clone(), k.uv)).collect();
    let c = check_files_keep(input, &glb, &keep)?;
    let mut checks_json = to_json(&c);
    let mut metrics = if x.metrics || x.id_maps || x.mirror.is_some() { Some(patternx::metrics(&model, &lay)) } else { None };
    if let Some(pm) = metrics.as_mut() {
        pm.aniso_limit = x.low_stretch;
    }
    if let (Some(pm), true) = (&metrics, x.metrics) {
        // The new numbers sit beside the existing checks; "passed" now needs both.
        let pj = pm.to_json();
        checks_json["pattern"] = pj["summary"].clone();
        checks_json["pattern"]["passed"] = pj["passed"].clone();
        checks_json["pattern"]["limits"] = pj["limits"].clone();
        checks_json["passed"] = json!(c.passed() && pm.passed());
        write(out.join("pattern-metrics.json"), serde_json::to_string_pretty(&pj).unwrap().as_bytes())?;
    }
    write(out.join("checks.json"), serde_json::to_string_pretty(&checks_json).unwrap().as_bytes())?;
    let file = glb.file_name().unwrap().to_string_lossy().into_owned();
    write(out.join("report.json"), serde_json::to_string_pretty(&json!({ "output": file, "engine": "rust-pattern", "size": size })).unwrap().as_bytes())?;
    if x.id_maps {
        let m = manifest(&model, &lay, &o, input, &out, &file, gs, x)?;
        write(out.join("PAINT-KIT-MANIFEST.json"), serde_json::to_string_pretty(&m).unwrap().as_bytes())?;
    }
    if let (Some(pm), true) = (&metrics, x.metrics) {
        if !pm.passed() {
            crate::relayoutjob::print(&c);
            pm.print();
            return Err("the pattern checks did not pass (texel density, mirror error or stretch over its limit)".into());
        }
        pm.print();
    }
    Ok((glb, c))
}

/// The paint-kit manifest: every file with its size and SHA-256, how the ID maps are encoded, and
/// what each chart, triangle range and bone id means.
#[allow(clippy::too_many_arguments)]
fn manifest(model: &Model, lay: &pattern::Layout, o: &Options, input: &Path, out: &Path, glb: &str, gs: usize, x: &Extra) -> Result<serde_json::Value, String> {
    let mut files = serde_json::Map::new();
    let mut names: Vec<PathBuf> = std::fs::read_dir(out).map_err(|e| e.to_string())?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_file()).collect();
    names.sort();
    for p in names {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if name == "PAINT-KIT-MANIFEST.json" {
            continue;
        }
        let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        files.insert(name, json!({ "bytes": b.len(), "sha256": sha256(&b) }));
    }
    let src = std::fs::read(input).map_err(|e| format!("{}: {e}", input.display()))?;
    let (lo, hi) = wb_core::render::bounds(model);
    let offs = patternx::triangle_offsets(model);
    let id = "id = R + 256 G + 65536 B; alpha 255 where a triangle covers the texel, 0 elsewhere (id 0 = no surface)";
    Ok(json!({
        "revision": 8,
        "tool": "gaw pattern (Rust, no Blender)",
        "scope": "UV sewing-pattern layout of the input model; not installed in the game",
        "source": { "file": input.file_name().map(|n| n.to_string_lossy().into_owned()), "sha256": sha256(&src) },
        "output": glb,
        "atlas_size": o.size, "map_size": gs,
        "texels_per_unit_at_density_1": (lay.texels_per_unit * 100.0).round() / 100.0,
        "options": {
            "orient_bones": x.orient_bones, "mirror": format!("{:?}", o.mirror), "mirror_tolerance_of_body_height": o.mirror_tol,
            "head_density": o.head_density, "hand_density": o.hand_density, "kept_stock_boxes": o.keep.len(),
        },
        "files": files,
        "encodings": {
            "chart_id": format!("{id}; chart id = piece id + 1 (see charts below and legend.json, where the piece id is 'id')"),
            "triangle_id": format!("{id}; triangle id = triangle offset of its mesh + triangle index + 1 (offsets below); back faces that repeat a front face are not drawn"),
            "bone_id": format!("{id}; bone id = skin joint index + 1, the joint with most weight at the texel (weights blended across the triangle)"),
            "chart_preview": "readable color per chart, for people",
            "position": { "minimum": lo, "maximum": hi, "formula": "(RGB / 255) * (maximum - minimum) + minimum, object space, rest pose" },
            "normal": "object-space normal, (RGB / 255) * 2 - 1; not a tangent-space map",
            "parts": "flat color per body part, see legend.json 'parts'",
            "uv": "0..1, v down (PNG row order)",
        },
        "meshes": model.meshes.iter().enumerate().map(|(i, m)| json!({ "mesh": i, "name": m.name, "atlas": model.atlas_name(i), "triangle_offset": offs[i], "triangles": m.tris.len() })).collect::<Vec<_>>(),
        "charts": lay.pieces.iter().enumerate().map(|(i, p)| json!({
            "chart_id": i + 1, "label": p.label, "mesh": p.mesh, "part": p.part.name(), "side": p.side.name(), "triangles": p.tris.len(),
            "dominant_bone_id": p.bone.map(|b| b + 1), "density_factor": p.density_factor,
            "twin_chart_id": p.twin.map(|t| t + 1), "mirror_of_chart_id": p.mirror_of.map(|t| t + 1), "pinned_at_stock": p.pinned,
        })).collect::<Vec<_>>(),
        "bones": model.joint_names.iter().enumerate().map(|(i, n)| json!({ "bone_id": i + 1, "name": n })).collect::<Vec<_>>(),
    }))
}
