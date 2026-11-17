//! `gaw defects`: finds specific places to fix in a candidate model (jagged edges, discoloration,
//! seam stripes, direction breaks, dark gutters) and writes defects.json plus one crop sheet per
//! kind. The math is in `wb_core::defects`; the drawing, part filter and output are here. Reads
//! models only; writes a new folder.
use crate::judgecmd;
use crate::requestcmd;
use crate::{Args, KIT_PARTS};
use image::RgbaImage;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use wb_core::analysis::Parts;
use wb_core::defects::{self as dx, Defect, Keep, Kind, Load, Palette, ShotCtx};
use wb_core::golden::{self, Kind as Size, View};
use wb_core::render::Camera;
use wb_core::{image_textures, model_textures, Model, Texture};

pub const DEFECTS_USAGE: &str = "  gaw defects CANDIDATE.glb --out-dir NEW_DIR [--ref REF.glb] [--part NAME[,NAME]] [options]
      Finds specific places to fix and says where: jagged edges (stair steps along alpha-cutout and
      silhouette contours), discoloration (patches far from the part's palette), seam stripes (a colour
      step across a UV seam that is continuous in 3D), direction breaks (hair strands turning across a
      seam or inside a card) and dark gutters (texels just outside an island that differ from its
      border), smeared patches (strands blurred together: low contrast across the strands against the
      reference), flat patches (almost no colour variance: plastic), specks (tiny cut-off fragments in
      the render or opaque scraps in the texture), alpha stipple (alpha near one half and pin holes
      that the cutout turns into noise) and layer direction cuts (strand direction jumping where
      one layer ends and the next begins) and lock-scale patches (fine wood-grain stripes with no broad
      locks under them, against the reference). Draws close-ups and game-distance views, maps every finding back to triangles, UV
      island and texel box, and writes defects.json (each defect: kind, score, view, screen box,
      triangles, island, texel box; score 1.0 = at the calibrated limit), one crop sheet per kind with
      numbered markers (sheet-<kind>.png) and the views behind them. Prints one line per kind.
      --part NAME[,NAME]       only these triangles: hair, a kit part, a bone group (head, arm_L, ..) or a mesh name
      --ref REF.glb            fit the cameras to REF's part and run the same checks on it for comparison
      --palette self|ref       discoloration palette from the candidate (default) or from --ref
      --kinds a,b              only these detectors: jagged-edge, discoloration, seam-stripe, direction-break, dark-gutter,
                               smear-blur, flat-patch, speck, alpha-stipple, layer-direction-cut, lock-scale
      --probe VIEW:X:Y         print the mesh, triangle, island, UV, texel and alpha neighbourhood under one pixel of a view
      --all-views              also write every drawn view as a picture (views/), not only those with a defect
      --top N                  defects drawn per crop sheet (default 12)
      --parts DIR              kit part maps for --part names (default: the UV kit)
      --any-bone               --part hair: do not require the head bone
      --json                   print defects.json";

struct A<'a>(&'a [String]);

const VALUE_OPTS: [&str; 8] = ["--ref", "--part", "--parts", "--out-dir", "--palette", "--kinds", "--top", "--probe"];

impl A<'_> {
    fn opt(&self, k: &str) -> Option<&str> {
        self.0.iter().position(|a| a == k).and_then(|i| self.0.get(i + 1)).map(String::as_str)
    }
    fn flag(&self, k: &str) -> bool {
        self.0.iter().any(|a| a == k)
    }
    fn positionals(&self) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut skip = false;
        for a in self.0 {
            if skip {
                skip = false;
                continue;
            }
            if VALUE_OPTS.contains(&a.as_str()) {
                skip = true;
            } else if !a.starts_with("--") {
                out.push(PathBuf::from(a));
            }
        }
        out
    }
}

fn check_out_dir(p: &Path) -> Result<(), String> {
    let s = p.to_string_lossy().replace('/', "\\").to_ascii_lowercase();
    if s.starts_with("f:") {
        return Err("refusing to write to F:".into());
    }
    if s.contains("final fantasy ix") && !s.contains("\\modding") {
        return Err("refusing to write into the installed game folder".into());
    }
    if p.is_dir() && std::fs::read_dir(p).map(|mut r| r.next().is_some()).unwrap_or(false) {
        return Err(format!("{} is not empty: --out-dir must be new or empty", p.display()));
    }
    Ok(())
}

// ---------- the part under inspection ----------

pub(crate) fn part_keep(model: &Model, names: &[String], parts: Option<&Parts>, any_bone: bool) -> Result<Keep, String> {
    if names.is_empty() {
        return Ok(model.meshes.iter().map(|m| vec![true; m.tris.len()]).collect());
    }
    let tex = model_textures(model);
    let labels = judgecmd::labels_for(model, parts);
    let offs = wb_core::render::tri_offsets(model);
    let mut keep: Keep = model.meshes.iter().map(|m| vec![false; m.tris.len()]).collect();
    for n in names {
        let low = n.to_ascii_lowercase();
        if low == "hair" {
            let f = judgecmd::hair_flags_for(any_bone, model, &tex);
            for (mi, v) in f.iter().enumerate() {
                for (ti, &b) in v.iter().enumerate() {
                    keep[mi][ti] |= b;
                }
            }
            continue;
        }
        for (mi, m) in model.meshes.iter().enumerate() {
            let by_mesh = m.name.to_ascii_lowercase() == low;
            for ti in 0..m.tris.len() {
                let lab = &labels.names[labels.tri[offs[mi] + ti] as usize];
                // A kit part name, a bone group (head, torso, arm_L ..) or a mesh name all select.
                let group = model.joint_names.get(model.triangle_joint(mi, ti)).map(|b| judgecmd::bone_group(b)).unwrap_or_default();
                if by_mesh || lab.to_ascii_lowercase() == low || group.to_ascii_lowercase() == low {
                    keep[mi][ti] = true;
                }
            }
        }
    }
    if keep.iter().all(|k| k.iter().all(|&b| !b)) {
        let mut avail: Vec<&String> = labels.names.iter().collect();
        avail.sort();
        return Err(format!("--part {}: no triangles matched. Names here: hair, {}", names.join(","), avail.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
    }
    Ok(keep)
}

// ---------- views ----------

fn turned(base: Camera, yaw_deg: f32, pitch_deg: f32) -> Camera {
    let mut c = base;
    c.yaw = std::f32::consts::PI + yaw_deg.to_radians();
    c.pitch = pitch_deg.to_radians();
    c
}

const GAME_VIEWS: [&str; 5] = ["idle0_front", "idle0_back", "idle0_side90", "idle0_side270", "idle0_34"];

pub(crate) fn make_views(fit: &Model, keep: &Keep, has_part: bool) -> Vec<(View, Size)> {
    let mut out: Vec<(View, Size)> = Vec::new();
    let sv = golden::standard_views(fit);
    if has_part {
        let tris: Vec<(usize, usize)> = keep.iter().enumerate().flat_map(|(mi, k)| k.iter().enumerate().filter(|(_, &b)| b).map(move |(ti, _)| (mi, ti))).collect();
        let group = golden::PartGroup { name: "part".into(), tris };
        let ab = golden::part_views(fit, &[group]);
        if let Some(first) = ab.first() {
            let (center, half) = (first.cam.target, first.cam.ortho_half);
            let base = Camera::fit(fit, 1.0, golden::GAME_FILL, true);
            let mk = |name: String, yaw: f32, pitch: f32| {
                let mut cam = turned(base, yaw, pitch);
                cam.target = center;
                cam.ortho_half = half * 1.15;
                View { name, cam, clip: first.clip.clone(), time: 0.0, kinds: vec![Size::Close] }
            };
            for yaw in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0] {
                out.push((mk(format!("orbit_y{:03}", yaw as i32), yaw, 10.0), Size::Close));
            }
            out.push((mk("orbit_top_y000".into(), 0.0, 55.0), Size::Close));
            out.push((mk("orbit_top_y180".into(), 180.0, 55.0), Size::Close));
            for mut v in ab {
                v.name = v.name.replacen("part_part_", "part_", 1);
                out.push((v, Size::Close));
            }
        }
    }
    for v in &sv {
        if v.kinds.contains(&Size::Close) {
            if !has_part {
                out.push((v.clone(), Size::Close));
            }
        } else if GAME_VIEWS.contains(&v.name.as_str()) {
            out.push((v.clone(), Size::Game));
        }
    }
    out
}

struct Drawn {
    name: String,
    size: Size,
    px: usize,
    cam: Camera,
    shot: golden::Shot,
}

fn draw_all(model: &Model, tex: &[Option<Texture>], views: &[(View, Size)]) -> Vec<Drawn> {
    views
        .iter()
        .filter_map(|(v, k)| {
            let shot = golden::render_view(model, tex, v, *k)?;
            Some(Drawn { name: v.name.clone(), size: *k, px: if *k == Size::Game { golden::GAME_SIZE } else { golden::LARGE_SIZE }, cam: v.cam, shot })
        })
        .collect()
}

// ---------- analysis ----------

struct Found {
    d: Defect,
    seen: Vec<String>,
    /// Where the same place was found in the other views: (view, screen box).
    also: Vec<(String, [i32; 4])>,
}

struct Analysis {
    found: Vec<Found>,
    loads: BTreeMap<Kind, Load>,
    notes: Vec<String>,
}

fn merge_found(list: Vec<Defect>, seen_from: &dyn Fn(&Defect) -> String) -> Vec<Found> {
    let mut sorted = list;
    sorted.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut out: Vec<Found> = Vec::new();
    for d in sorted {
        let view = seen_from(&d);
        // The same place seen from another view: the texel boxes meet (a big card shares triangles with
        // many unrelated spots, so triangles alone would merge them). No boxes: fall back to triangles.
        let same_place = |a: &Defect, b: &Defect| -> bool {
            if a.boxes.is_empty() || b.boxes.is_empty() {
                return a.tris.iter().any(|t| b.tris.contains(t));
            }
            const PAD: usize = 6;
            a.boxes.iter().any(|p| b.boxes.iter().any(|q| p.image == q.image && p.x0 <= q.x1 + PAD && q.x0 <= p.x1 + PAD && p.y0 <= q.y1 + PAD && q.y0 <= p.y1 + PAD))
        };
        if let Some(f) = out.iter_mut().find(|f| f.d.kind == d.kind && same_place(&f.d, &d)) {
            if let Some(sc) = &d.screen {
                if f.also.len() < 12 && !f.also.iter().any(|(v, _)| v == &sc.view) && f.d.screen.as_ref().map(|p| p.view != sc.view).unwrap_or(true) {
                    f.also.push((sc.view.clone(), sc.bbox));
                }
            }
            if !f.seen.contains(&view) && !view.is_empty() {
                f.seen.push(view);
            }
            continue;
        }
        out.push(Found { seen: if view.is_empty() { vec![] } else { vec![view] }, also: Vec::new(), d });
    }
    out
}

/// Tile statistics of the part in every close view: 16 px tiles (smear) and 8 px tiles (flat patches).
struct Grids {
    t16: Vec<(String, dx::TileGrid)>,
    t8: Vec<(String, dx::TileGrid)>,
    scale: Vec<(String, dx::ScaleGrid)>,
}

fn close_grids(model: &Model, keep: &Keep, drawn: &[Drawn], pose: &[Vec<[f32; 3]>]) -> Grids {
    let kg = dx::keep_global(model, keep);
    let mk = |tile: usize| -> Vec<(String, dx::TileGrid)> {
        drawn
            .iter()
            .filter(|d| d.size == Size::Close)
            .map(|d| (d.name.clone(), dx::tiles(model, &ShotCtx { name: d.name.clone(), px: d.px, cam: d.cam, pose, shot: &d.shot }, &kg, tile)))
            .collect()
    };
    let scale = drawn.iter().filter(|d| d.size == Size::Close).map(|d| (d.name.clone(), dx::scale_grid(&ShotCtx { name: d.name.clone(), px: d.px, cam: d.cam, pose, shot: &d.shot }, &kg))).collect();
    Grids { t16: mk(16), t8: mk(8), scale }
}

struct Bases {
    b16: dx::Baseline,
    b8: dx::Baseline,
    sb: dx::ScaleBase,
}

fn bases_from(g: &Grids) -> Bases {
    let f = |v: &Vec<(String, dx::TileGrid)>| dx::Baseline::from_grids(&v.iter().map(|(n, g)| (n.clone(), g)).collect::<Vec<_>>());
    Bases { b16: f(&g.t16), b8: f(&g.t8), sb: dx::ScaleBase::from_grids(&g.scale) }
}

fn analyze(model: &Model, tex: &[Option<Texture>], keep: &Keep, drawn: &[Drawn], pose: &[Vec<[f32; 3]>], pal: &Palette, kinds: &[Kind], want_dir: bool, grids: &Grids, bases: &Bases) -> Analysis {
    let kg = dx::keep_global(model, keep);
    let ctxs: Vec<ShotCtx> = drawn.iter().map(|d| ShotCtx { name: d.name.clone(), px: d.px, cam: d.cam, pose, shot: &d.shot }).collect();
    let mut loads: BTreeMap<Kind, Load> = BTreeMap::new();
    let mut notes: Vec<String> = Vec::new();
    let mut screen: Vec<Defect> = Vec::new();
    let mut mesh_based: Vec<Defect> = Vec::new();
    let (mut contour, mut flagged_px, mut worst_rms) = (0usize, 0usize, 0.0f64);
    let (mut part_px, mut outlier_px) = (0usize, 0usize);
    let (mut blur_measured, mut blur_tiles, mut flat_measured, mut flat_tiles) = (0usize, 0usize, 0usize, 0usize);
    let (mut edge_px, mut edge_flagged) = (0usize, 0usize);
    let (mut grain_measured, mut grain_tiles) = (0usize, 0usize);
    let (mut speck_px, mut speck_frag, mut speck_dots, mut close_part_px) = (0usize, 0usize, 0usize, 0usize);
    let want = |k: Kind| kinds.contains(&k);
    for (dr, sc) in drawn.iter().zip(&ctxs) {
        let close = dr.size == Size::Close;
        if close {
            close_part_px += dr.shot.pick.iter().filter(|&&g| g >= 0 && kg.get(g as usize).copied().unwrap_or(false)).count();
            if want(Kind::Blur) {
                if let Some((_, g)) = grids.t16.iter().find(|(n, _)| n == &dr.name) {
                    let s = dx::sharpness(model, tex, sc, g, &bases.b16, Kind::Blur);
                    blur_measured += s.measured;
                    blur_tiles += s.flagged_tiles;
                    screen.extend(s.defects);
                }
            }
            if want(Kind::Flat) {
                if let Some((_, g)) = grids.t8.iter().find(|(n, _)| n == &dr.name) {
                    let s = dx::sharpness(model, tex, sc, g, &bases.b8, Kind::Flat);
                    flat_measured += s.measured;
                    flat_tiles += s.flagged_tiles;
                    screen.extend(s.defects);
                }
                screen.extend(dx::flat_cards(model, tex, sc, &kg, &bases.b8).defects);
            }
            if want(Kind::Grain) {
                if let Some((_, g)) = grids.scale.iter().find(|(n, _)| n == &dr.name) {
                    let s = dx::lock_scale(model, tex, sc, g, &bases.sb);
                    grain_measured += s.measured;
                    grain_tiles += s.flagged_tiles;
                    screen.extend(s.defects);
                }
            }
            if want(Kind::LayerCut) {
                let s = dx::layer_edges(model, tex, sc, &kg);
                edge_px += s.boundary_px;
                edge_flagged += s.flagged_px;
                screen.extend(s.defects);
            }
            if want(Kind::Speck) {
                let s = dx::specks(model, tex, sc, &kg);
                speck_px += s.px;
                speck_frag += s.fragments;
                speck_dots += s.dots;
                screen.extend(s.defects);
            }
        }
        if close && kinds.contains(&Kind::Jagged) {
            let j = dx::jagged(model, tex, sc, &kg);
            contour += j.contour_px;
            flagged_px += j.flagged_px;
            worst_rms = worst_rms.max(j.worst_rms);
            screen.extend(j.defects);
        }
        if kinds.contains(&Kind::Discolor) {
            let o = dx::discolor(model, tex, sc, &kg, pal, if close { 40 } else { 5 });
            if close {
                part_px += o.part_px;
                outlier_px += o.outlier_px;
            }
            screen.extend(o.defects);
        }
    }
    if kinds.contains(&Kind::Jagged) {
        loads.insert(Kind::Jagged, Load { percent: if contour > 0 { 100.0 * flagged_px as f64 / contour as f64 } else { 0.0 }, measured: contour as f64, flagged: flagged_px as f64, extra: vec![("worst_rms_px".into(), worst_rms)] });
    }
    if kinds.contains(&Kind::Discolor) {
        loads.insert(Kind::Discolor, Load { percent: if part_px > 0 { 100.0 * outlier_px as f64 / part_px as f64 } else { 0.0 }, measured: part_px as f64, flagged: outlier_px as f64, extra: vec![("palette_colors".into(), pal.centers.len() as f64)] });
    }
    let pct = |a: usize, b: usize| if b > 0 { 100.0 * a as f64 / b as f64 } else { 0.0 };
    if want(Kind::Blur) {
        loads.insert(Kind::Blur, Load { percent: pct(blur_tiles, blur_measured), measured: blur_measured as f64, flagged: blur_tiles as f64, extra: vec![("baseline_across".into(), bases.b16.global.0)] });
    }
    if want(Kind::Flat) {
        loads.insert(Kind::Flat, Load { percent: pct(flat_tiles, flat_measured), measured: flat_measured as f64, flagged: flat_tiles as f64, extra: vec![("baseline_variance_8px".into(), bases.b8.global.1)] });
    }
    if want(Kind::Grain) {
        loads.insert(Kind::Grain, Load { percent: pct(grain_tiles, grain_measured), measured: grain_measured as f64, flagged: grain_tiles as f64, extra: vec![("baseline_fibre_share".into(), bases.sb.global)] });
    }
    if want(Kind::LayerCut) {
        loads.insert(Kind::LayerCut, Load { percent: pct(edge_flagged, edge_px), measured: edge_px as f64, flagged: edge_flagged as f64, extra: vec![] });
    }
    let topo = dx::topo(model);
    if want(Kind::Stipple) {
        let n = dx::texture_noise(model, &topo, tex, keep);
        loads.insert(Kind::Stipple, n.stipple_load);
        mesh_based.extend(n.stipple_defects);
    }
    if want(Kind::Speck) {
        loads.insert(Kind::Speck, Load { percent: pct(speck_px, close_part_px), measured: close_part_px as f64, flagged: speck_px as f64, extra: vec![("render_fragments".into(), speck_frag as f64), ("render_dots".into(), speck_dots as f64)] });
    }
    if kinds.contains(&Kind::Stripe) {
        let s = dx::stripes(model, &topo, tex, keep);
        loads.insert(Kind::Stripe, s.load);
        mesh_based.extend(s.defects);
    }
    if kinds.contains(&Kind::Direction) {
        if want_dir {
            let s = dx::directions(model, tex, keep, &topo);
            loads.insert(Kind::Direction, s.load);
            mesh_based.extend(s.defects);
        } else {
            notes.push("direction-break: skipped (it reads painted strand structure, so it runs for --part hair only)".into());
        }
    }
    if kinds.contains(&Kind::Gutter) {
        let s = dx::gutters(model, &topo, tex, keep);
        loads.insert(Kind::Gutter, s.load);
        mesh_based.extend(s.defects);
    }
    for d in &mut mesh_based {
        d.screen = dx::place(model, d, &ctxs);
    }
    let mut found = merge_found(screen, &|d| d.screen.as_ref().map(|s| s.view.clone()).unwrap_or_default());
    found.extend(mesh_based.into_iter().map(|d| {
        let v = d.screen.as_ref().map(|s| s.view.clone()).unwrap_or_default();
        Found { seen: if v.is_empty() { vec![] } else { vec![v] }, also: Vec::new(), d }
    }));
    found.sort_by(|a, b| a.d.kind.cmp(&b.d.kind).then(b.d.score.total_cmp(&a.d.score)));
    Analysis { found, loads, notes }
}

// ---------- output ----------

fn severity(score: f64) -> &'static str {
    if score >= 2.0 {
        "high"
    } else if score >= 1.3 {
        "medium"
    } else {
        "low"
    }
}

fn cam_json(c: &Camera) -> Value {
    json!({"yaw_deg": c.yaw.to_degrees(), "pitch_deg": c.pitch.to_degrees(), "distance": c.dist, "target": c.target, "fov_y_deg": c.fov_y.to_degrees(), "ortho": c.ortho, "ortho_half": c.ortho_half, "aspect": 1.0})
}

fn sheet_name(model: &Model, image: usize) -> String {
    model.images.get(image).map(|i| i.name.trim_end_matches(".png").to_owned()).unwrap_or_else(|| format!("image{image}"))
}

fn defect_json(model: &Model, f: &Found, id: &str, drawn: &[Drawn]) -> Value {
    let d = &f.d;
    let mut by_mesh: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &(mi, ti) in &d.tris {
        by_mesh.entry(mi).or_default().push(ti);
    }
    let meshes: Vec<Value> = by_mesh
        .iter()
        .map(|(&mi, tris)| {
            let mut t = tris.clone();
            t.sort();
            json!({"mesh": mi, "name": model.meshes[mi].name, "image": model.atlas_name(mi), "triangle_count": t.len(), "triangles": t, "weights": Value::Null})
        })
        .collect();
    let boxes: Vec<Value> = d.boxes.iter().map(|b| json!({"sheet": sheet_name(model, b.image), "image_index": b.image, "box": [b.x0, b.y0, b.x1, b.y1]})).collect();
    let mut j = json!({
        "id": id,
        "kind": d.kind.tag(),
        "score": (d.score * 100.0).round() / 100.0,
        "severity": severity(d.score),
        "measure": {"value": (d.raw * 100.0).round() / 100.0, "unit": d.kind.unit(), "limit": d.kind.limit()},
        "note": d.note,
        "marked": {"triangles": d.tris.len(), "meshes": meshes},
        "texel_box": boxes.first().cloned().unwrap_or(Value::Null),
        "texel_boxes": boxes,
        "seen_in": f.seen,
        "seen_boxes": f.also.iter().map(|(v, b)| json!({"view": v, "screen_box": b})).collect::<Vec<_>>(),
    });
    if let Some((mi, il)) = d.island {
        let n = by_mesh.get(&mi).map(|v| v.len()).unwrap_or(0);
        j["island"] = json!({"mesh": mi, "id": il, "triangles_in_defect": n});
    } else {
        j["island"] = Value::Null;
    }
    match &d.screen {
        Some(s) => {
            j["view"] = json!(s.view);
            j["picture"] = json!(format!("views/{}.png", s.view));
            j["picture_px"] = json!(s.px);
            j["screen_box"] = json!(s.bbox);
            j["camera"] = drawn.iter().find(|x| x.name == s.view).map(|x| cam_json(&x.cam)).unwrap_or(Value::Null);
        }
        None => {
            j["view"] = Value::Null;
            j["screen_box"] = Value::Null;
            j["camera"] = Value::Null;
            j["note_view"] = json!("not visible in any of the drawn views; use the texel box");
        }
    }
    j
}

// ---------- crop sheets ----------

const CELL: usize = 240;

fn fill(img: &mut RgbaImage, x0: i64, y0: i64, w: i64, h: i64, c: [u8; 4]) {
    for y in y0.max(0)..(y0 + h).min(img.height() as i64) {
        for x in x0.max(0)..(x0 + w).min(img.width() as i64) {
            img.put_pixel(x as u32, y as u32, image::Rgba(c));
        }
    }
}

fn rect(img: &mut RgbaImage, x0: i64, y0: i64, x1: i64, y1: i64, t: i64, c: [u8; 4]) {
    fill(img, x0, y0, x1 - x0, t, c);
    fill(img, x0, y1 - t, x1 - x0, t, c);
    fill(img, x0, y0, t, y1 - y0, c);
    fill(img, x1 - t, y0, t, y1 - y0, c);
}

/// A square crop around `bbox` of an RGBA picture, `CELL` px wide, with the box drawn on it.
fn crop_picture(rgba: &[u8], px: usize, bbox: [i32; 4], mark: [u8; 4]) -> RgbaImage {
    let (cx, cy) = (0.5 * (bbox[0] + bbox[2]) as f32, 0.5 * (bbox[1] + bbox[3]) as f32);
    let half = (0.5 * ((bbox[2] - bbox[0]).max(bbox[3] - bbox[1])) as f32 * 1.7).max(28.0);
    let scale = CELL as f32 / (2.0 * half);
    let mut img = RgbaImage::from_pixel(CELL as u32, CELL as u32, image::Rgba([40, 40, 44, 255]));
    for y in 0..CELL {
        for x in 0..CELL {
            let (sx, sy) = (cx - half + (x as f32 + 0.5) / scale, cy - half + (y as f32 + 0.5) / scale);
            if sx < 0.0 || sy < 0.0 || sx >= px as f32 || sy >= px as f32 {
                continue;
            }
            let i = (sy as usize * px + sx as usize) * 4;
            img.put_pixel(x as u32, y as u32, image::Rgba([rgba[i], rgba[i + 1], rgba[i + 2], 255]));
        }
    }
    let m = |v: i32, c: f32| ((v as f32 - (c - half)) * scale).round() as i64;
    rect(&mut img, m(bbox[0], cx), m(bbox[1], cy), m(bbox[2], cx), m(bbox[3], cy), 2, mark);
    img
}

fn crop_texture(t: &Texture, b: &dx::TexelBox, mark: [u8; 4]) -> RgbaImage {
    let (cx, cy) = (0.5 * (b.x0 + b.x1) as f32, 0.5 * (b.y0 + b.y1) as f32);
    let half = (0.5 * ((b.x1 - b.x0).max(b.y1 - b.y0)) as f32 + 12.0).max(24.0);
    let scale = CELL as f32 / (2.0 * half);
    let mut img = RgbaImage::from_pixel(CELL as u32, CELL as u32, image::Rgba([40, 40, 44, 255]));
    for y in 0..CELL {
        for x in 0..CELL {
            let (sx, sy) = (cx - half + (x as f32 + 0.5) / scale, cy - half + (y as f32 + 0.5) / scale);
            if sx < 0.0 || sy < 0.0 || sx >= t.w as f32 || sy >= t.h as f32 {
                continue;
            }
            let i = (sy as usize * t.w + sx as usize) * 4;
            let a = t.rgba[i + 3] as f32 / 255.0;
            // Transparent texels over a checker, so clear texels read as clear.
            let chk = if ((x / 8) + (y / 8)) % 2 == 0 { 70.0 } else { 100.0 };
            let c: [u8; 3] = std::array::from_fn(|k| (t.rgba[i + k] as f32 * a + chk * (1.0 - a)) as u8);
            img.put_pixel(x as u32, y as u32, image::Rgba([c[0], c[1], c[2], 255]));
        }
    }
    let m = |v: usize, c: f32| ((v as f32 - (c - half)) * scale).round() as i64;
    rect(&mut img, m(b.x0, cx), m(b.y0, cy), m(b.x1, cx), m(b.y1, cy), 2, mark);
    img
}

fn write_sheet(path: &Path, items: &[(&str, &Found, usize)], model: &Model, drawn: &[Drawn], tex: &[Option<Texture>]) -> Result<(), String> {
    let cols = 2usize;
    let head = 22usize;
    let gap = 6usize;
    let cell_w = 2 * CELL + gap;
    let rows = items.len().div_ceil(cols);
    let (w, h) = (cols * (cell_w + gap) + gap, rows * (CELL + head + gap) + gap);
    let mut img = RgbaImage::from_pixel(w as u32, h as u32, image::Rgba([24, 24, 28, 255]));
    for (n, (id, f, number)) in items.iter().enumerate() {
        let (cx, cy) = ((n % cols) * (cell_w + gap) + gap, (n / cols) * (CELL + head + gap) + gap);
        let label = format!("#{} {} {:.1}x {}", number, id.rsplit('-').next().unwrap_or(""), f.d.score, f.d.screen.as_ref().map(|s| s.view.as_str()).unwrap_or("no view"));
        wb_core::font::draw(img.as_mut(), w, h, cx as i64 + 2, cy as i64 + 3, &label.to_ascii_uppercase(), 2, [20, 20, 20]);
        let left = match (&f.d.screen, drawn.iter().find(|d| Some(&d.name) == f.d.screen.as_ref().map(|s| &s.view))) {
            (Some(s), Some(dr)) => crop_picture(&dr.shot.rgb.to_rgba8(), dr.px, s.bbox, [255, 60, 60, 255]),
            _ => {
                let mut blank = RgbaImage::from_pixel(CELL as u32, CELL as u32, image::Rgba([48, 48, 54, 255]));
                wb_core::font::draw(blank.as_mut(), CELL, CELL, 10, 110, "NOT VISIBLE IN A VIEW", 2, [20, 20, 20]);
                blank
            }
        };
        for y in 0..CELL {
            for x in 0..CELL {
                img.put_pixel((cx + x) as u32, (cy + head + y) as u32, *left.get_pixel(x as u32, y as u32));
            }
        }
        if let Some(b) = f.d.boxes.first() {
            if let Some(Some(t)) = tex.get(b.image) {
                let tc = crop_texture(t, b, [255, 0, 255, 255]);
                for y in 0..CELL {
                    for x in 0..CELL {
                        img.put_pixel((cx + CELL + gap + x) as u32, (cy + head + y) as u32, *tc.get_pixel(x as u32, y as u32));
                    }
                }
                let sn = format!("TEXTURE {}", sheet_name(model, b.image)).to_ascii_uppercase();
                wb_core::font::draw(img.as_mut(), w, h, (cx + CELL + gap + 6) as i64, (cy + head + 6) as i64, &sn, 2, [20, 20, 20]);
            }
        }
        // The number, large, on the picture.
        wb_core::font::draw(img.as_mut(), w, h, cx as i64 + 8, (cy + head + CELL - 30) as i64, &format!("{}", number), 4, [200, 0, 0]);
    }
    img.save(path).map_err(|e| format!("{}: {e}", path.display()))
}

// ---------- the command ----------

fn pose_of(model: &Model) -> Vec<Vec<[f32; 3]>> {
    model.posed(model.clips.iter().position(|c| c == "idle"), 0.0)
}

fn parse_kinds(a: &A) -> Result<Vec<Kind>, String> {
    match a.opt("--kinds") {
        None => Ok(Kind::ALL.to_vec()),
        Some(s) => s
            .split(',')
            .map(|w| Kind::ALL.iter().copied().find(|k| k.tag() == w.trim()).ok_or_else(|| format!("--kinds: unknown detector {w}; use {}", Kind::ALL.iter().map(|k| k.tag()).collect::<Vec<_>>().join(", "))))
            .collect(),
    }
}

fn summary_line(k: Kind, found: &[&Found], load: Option<&Load>, note: &str) -> String {
    let worst = found.iter().map(|f| f.d.score).fold(0.0, f64::max);
    let place = found
        .iter()
        .max_by(|a, b| a.d.score.total_cmp(&b.d.score))
        .map(|f| match &f.d.screen {
            Some(s) => format!("worst in {} at {:?}", s.view, s.bbox),
            None => "worst not visible in a view".into(),
        })
        .unwrap_or_default();
    match load {
        Some(l) => format!("{:<16} {:>3} defect(s)   worst {:>4.1}x limit   load {:>5.1}%   {}{}", k.tag(), found.len(), worst, l.percent, place, note),
        None => format!("{:<16} skipped{}", k.tag(), note),
    }
}

pub fn defects_cmd(args: &Args) -> Result<(), String> {
    let a = A(&args.0);
    let pos = a.positionals();
    let cand_path = pos.first().cloned().ok_or("defects needs a candidate GLB: gaw defects CANDIDATE.glb --out-dir NEW_DIR")?;
    let out = PathBuf::from(a.opt("--out-dir").ok_or("defects needs --out-dir NEW_DIR")?);
    check_out_dir(&out)?;
    let kinds = parse_kinds(&a)?;
    let top: usize = a.opt("--top").map(|s| s.parse().map_err(|_| format!("--top: cannot read {s}"))).transpose()?.unwrap_or(12);
    let names: Vec<String> = a.opt("--part").map(|s| s.split(',').map(|x| x.trim().to_owned()).filter(|x| !x.is_empty()).collect()).unwrap_or_default();
    let parts = Parts::load(&a.opt("--parts").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(KIT_PARTS))).ok();
    let any_bone = a.flag("--any-bone");

    let cand = Model::open(&cand_path)?;
    let ctex = image_textures(&cand);
    let ckeep = part_keep(&cand, &names, parts.as_ref(), any_bone)?;
    let has_part = !names.is_empty();
    let reference = match a.opt("--ref") {
        Some(r) => Some((PathBuf::from(r), Model::open(Path::new(r))?)),
        None => None,
    };
    let rkeep = match &reference {
        Some((_, m)) => Some(part_keep(m, &names, parts.as_ref(), any_bone).unwrap_or_else(|_| m.meshes.iter().map(|x| vec![false; x.tris.len()]).collect())),
        None => None,
    };
    let (fit, fit_keep) = match (&reference, &rkeep) {
        (Some((_, m)), Some(k)) if k.iter().any(|v| v.iter().any(|&b| b)) => (m, k),
        _ => (&cand, &ckeep),
    };
    let views = make_views(fit, fit_keep, has_part);
    let drawn = draw_all(&cand, &ctex, &views);
    let pose = pose_of(&cand);

    // The palette: the candidate's own colours, or the reference's.
    let kg = dx::keep_global(&cand, &ckeep);
    let ref_drawn = match (&reference, &rkeep) {
        (Some((_, m)), Some(_)) => Some((m, image_textures(m), draw_all(m, &image_textures(m), &views))),
        _ => None,
    };
    let palette_from_ref = a.opt("--palette") == Some("ref");
    let mut labs: Vec<[f32; 3]> = Vec::new();
    if palette_from_ref {
        let (m, _, rd) = ref_drawn.as_ref().ok_or("--palette ref needs --ref")?;
        let rp = pose_of(m);
        let rkg = dx::keep_global(m, rkeep.as_ref().unwrap());
        for dr in rd.iter().filter(|d| d.size == Size::Close) {
            dx::collect_labs(&ShotCtx { name: dr.name.clone(), px: dr.px, cam: dr.cam, pose: &rp, shot: &dr.shot }, &rkg, &mut labs);
        }
    } else {
        for dr in drawn.iter().filter(|d| d.size == Size::Close) {
            dx::collect_labs(&ShotCtx { name: dr.name.clone(), px: dr.px, cam: dr.cam, pose: &pose, shot: &dr.shot }, &kg, &mut labs);
        }
        if labs.len() < 50 {
            // No close views (whole-body run): use the game views.
            for dr in drawn.iter() {
                dx::collect_labs(&ShotCtx { name: dr.name.clone(), px: dr.px, cam: dr.cam, pose: &pose, shot: &dr.shot }, &kg, &mut labs);
            }
        }
    }
    let pal = dx::palette(&labs);

    if let Some(spec) = a.opt("--probe") {
        // view:x:y -> what is under that pixel: mesh, triangle, island, UV, texel and the alpha around it.
        let p: Vec<&str> = spec.split(':').collect();
        if let [view, x, y] = p[..] {
            let (x, y): (usize, usize) = (x.parse().map_err(|_| "--probe: x")?, y.parse().map_err(|_| "--probe: y")?);
            let dr = drawn.iter().find(|d| d.name == view).ok_or("--probe: no such view")?;
            let sc = ShotCtx { name: dr.name.clone(), px: dr.px, cam: dr.cam, pose: &pose, shot: &dr.shot };
            match sc.hit(&cand, x, y) {
                Some((mi, ti, uv)) => {
                    let m = &cand.meshes[mi];
                    let topo = dx::topo(&cand);
                    println!("probe {view} ({x},{y}): mesh {mi} '{}' triangle {ti} island {} uv ({:.4},{:.4}) in the part: {}", m.name, topo.island[mi][ti], uv[0], uv[1], ckeep[mi][ti]);
                    if let Some(t) = m.image.and_then(|i| ctex.get(i)).and_then(|t| t.as_ref()) {
                        let (tx, ty) = ((uv[0] * t.w as f32) as i64, (uv[1] * t.h as f32) as i64);
                        println!("  texel ({tx},{ty}) on {}x{}; alpha (rows y-6..y+6, cols x-10..x+10):", t.w, t.h);
                        for yy in ty - 6..=ty + 6 {
                            let row: Vec<String> = (tx - 10..=tx + 10).map(|xx| if xx < 0 || yy < 0 || xx >= t.w as i64 || yy >= t.h as i64 { "  .".into() } else { format!("{:>3}", t.rgba[(yy as usize * t.w + xx as usize) * 4 + 3]) }).collect();
                            println!("  {}", row.join(" "));
                        }
                    }
                }
                None => println!("probe {view} ({x},{y}): background"),
            }
        }
    }
    let want_dir = names.iter().any(|n| n.eq_ignore_ascii_case("hair"));
    // Sharpness and flatness are measured against the reference's tiles when there is one, else
    // against the candidate's own medians (which hides a candidate that is flat everywhere).
    let cgrids = close_grids(&cand, &ckeep, &drawn, &pose);
    let rgrids = ref_drawn.as_ref().map(|(m, _, rd)| close_grids(m, rkeep.as_ref().unwrap(), rd, &pose_of(m)));
    let bases = bases_from(rgrids.as_ref().unwrap_or(&cgrids));
    if a.flag("--dump-tiles") {
        // Per close view and tile: how many times less contrast across the strands ("blur") and less
        // colour variance ("flat") than the baseline, the strand axis in degrees and its coherence.
        let dump = |gr: &[(String, dx::TileGrid)], base: &dx::Baseline| -> Value {
            let mut m = serde_json::Map::new();
            for (n, g) in gr {
                let (ba, bv) = base.view_baseline(n);
                let tiles: Vec<Value> = g.tiles.iter().map(|t| match t {
                    Some(t) => json!([(ba / t.across.max(1e-9) * 100.0).round() / 100.0, (bv / t.var.max(1e-9) * 100.0).round() / 100.0, t.axis.to_degrees().round(), (t.coherence * 100.0).round() / 100.0]),
                    None => Value::Null,
                }).collect();
                m.insert(format!("{n}@{}", g.tile), json!({"gw": g.gw, "gh": g.gh, "tile_px": g.tile, "base_across": ba, "base_var": bv, "tiles": tiles}));
            }
            Value::Object(m)
        };
        let both = |g: &Grids| -> Value {
            let mut m = match dump(&g.t16, &bases.b16) { Value::Object(m) => m, _ => unreachable!() };
            if let Value::Object(m8) = dump(&g.t8, &bases.b8) {
                m.extend(m8);
            }
            Value::Object(m)
        };
        let both = |g: &Grids| -> Value {
            let mut m = match both(g) { Value::Object(m) => m, _ => unreachable!() };
            for (n, sg) in &g.scale {
                let tiles: Vec<Value> = sg.bands.iter().map(|b| match b { Some(b) => json!([b[0], b[1], b[2], b[3]]), None => Value::Null }).collect();
                m.insert(format!("{n}@scale"), json!({"gw": sg.gw, "gh": sg.gh, "tile_px": sg.tile, "base_ratio": bases.sb.get(n), "tiles": tiles}));
            }
            Value::Object(m)
        };
        let j = json!({"candidate": both(&cgrids), "reference": rgrids.as_ref().map(|g| both(g))});
        std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
        std::fs::write(out.join("tiles.json"), serde_json::to_vec(&j).unwrap()).map_err(|e| format!("tiles.json: {e}"))?;
    }
    let mut an = analyze(&cand, &ctex, &ckeep, &drawn, &pose, &pal, &kinds, want_dir, &cgrids, &bases);
    if rgrids.is_none() {
        an.notes.push("smear-blur and flat-patch compare with this part's own median (no --ref): a part that is smeared or flat everywhere will not show".into());
    }
    let ref_an = match (&ref_drawn, &rgrids) {
        (Some((m, rt, rd)), Some(rg)) => {
            let rp = pose_of(m);
            Some(analyze(m, rt, rkeep.as_ref().unwrap(), rd, &rp, &pal, &kinds, want_dir, rg, &bases))
        }
        _ => None,
    };

    // Write everything.
    std::fs::create_dir_all(out.join("views")).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut ids: Vec<(String, usize)> = Vec::new(); // per defect, in `an.found` order
    let mut counters: BTreeMap<Kind, usize> = BTreeMap::new();
    for f in &an.found {
        let n = counters.entry(f.d.kind).or_insert(0);
        *n += 1;
        ids.push((format!("{}-{:03}", f.d.kind.tag(), n), *n));
    }
    let mut used_views: Vec<String> = Vec::new();
    for f in &an.found {
        if let Some(s) = &f.d.screen {
            if !used_views.contains(&s.view) {
                used_views.push(s.view.clone());
            }
        }
    }
    if a.flag("--all-views") {
        for d in &drawn {
            if !used_views.contains(&d.name) {
                used_views.push(d.name.clone());
            }
        }
    }
    for v in &used_views {
        if let Some(dr) = drawn.iter().find(|d| &d.name == v) {
            RgbaImage::from_raw(dr.px as u32, dr.px as u32, dr.shot.rgb.to_rgba8()).ok_or("bad picture size")?.save(out.join("views").join(format!("{v}.png"))).map_err(|e| format!("view picture: {e}"))?;
        }
    }
    if a.flag("--all-views") {
        // One picture per view with every screen box of every kind drawn on it, coloured by kind.
        let tint = |k: Kind| -> [u8; 4] {
            match k {
                Kind::Jagged => [255, 255, 0, 255],
                Kind::Discolor => [255, 0, 255, 255],
                Kind::Stripe => [0, 255, 255, 255],
                Kind::Direction => [255, 128, 0, 255],
                Kind::Gutter => [128, 128, 255, 255],
                Kind::Blur => [255, 0, 0, 255],
                Kind::Flat => [0, 255, 0, 255],
                Kind::Speck => [255, 255, 255, 255],
                Kind::Stipple => [0, 128, 255, 255],
                Kind::LayerCut => [255, 0, 128, 255],
                Kind::Grain => [128, 255, 0, 255],
            }
        };
        for dr in &drawn {
            let mut img = RgbaImage::from_raw(dr.px as u32, dr.px as u32, dr.shot.rgb.to_rgba8()).ok_or("bad picture size")?;
            for f in &an.found {
                if let Some(s) = f.d.screen.as_ref().filter(|s| s.view == dr.name) {
                    rect(&mut img, s.bbox[0] as i64, s.bbox[1] as i64, s.bbox[2] as i64, s.bbox[3] as i64, 2, tint(f.d.kind));
                }
                for (v, b) in &f.also {
                    if v == &dr.name {
                        rect(&mut img, b[0] as i64, b[1] as i64, b[2] as i64, b[3] as i64, 2, tint(f.d.kind));
                    }
                }
            }
            img.save(out.join("views").join(format!("overview-{}.png", dr.name))).map_err(|e| format!("overview picture: {e}"))?;
        }
    }
    let mut summary: Vec<Value> = Vec::new();
    let mut lines: Vec<String> = Vec::new();
    for &k in &kinds {
        let fs: Vec<&Found> = an.found.iter().filter(|f| f.d.kind == k).collect();
        let load = an.loads.get(&k);
        let rload = ref_an.as_ref().and_then(|r| r.loads.get(&k));
        let rcount = ref_an.as_ref().map(|r| r.found.iter().filter(|f| f.d.kind == k).count());
        let skipped = load.is_none();
        let mut line = summary_line(k, &fs, load, "");
        if let (Some(rl), Some(rc)) = (rload, rcount) {
            line.push_str(&format!("   | reference: {} defect(s), load {:.1}%", rc, rl.percent));
        }
        lines.push(line);
        summary.push(json!({
            "kind": k.tag(),
            "skipped": skipped,
            "defects": fs.len(),
            "worst_score": fs.iter().map(|f| f.d.score).fold(0.0, f64::max),
            "load_percent": load.map(|l| l.percent),
            "load_measured": load.map(|l| l.measured),
            "load_flagged": load.map(|l| l.flagged),
            "detail": load.map(|l| l.extra.iter().map(|(n, v)| (n.clone(), json!(v))).collect::<serde_json::Map<String, Value>>()),
            "limit": k.limit(),
            "unit": k.unit(),
            "reference": rload.map(|l| json!({"defects": rcount, "load_percent": l.percent, "detail": l.extra.iter().map(|(n, v)| (n.clone(), json!(v))).collect::<serde_json::Map<String, Value>>()})),
        }));
        if !fs.is_empty() {
            let mut items: Vec<(&str, &Found, usize)> = Vec::new();
            for (i, f) in an.found.iter().enumerate() {
                if f.d.kind == k && items.len() < top {
                    items.push((ids[i].0.as_str(), f, ids[i].1));
                }
            }
            write_sheet(&out.join(format!("sheet-{}.png", k.tag())), &items, &cand, &drawn, &ctex)?;
        }
    }
    let (_, when) = requestcmd::local_stamp();
    let defect_list: Vec<Value> = an.found.iter().zip(&ids).map(|(f, (id, _))| defect_json(&cand, f, id, &drawn)).collect();
    let j = json!({
        "workbench_defects": 1,
        "created": when,
        "folder": out.display().to_string(),
        "prompt": format!("Look at workbench defects {}", out.display()),
        "candidate": {"file": cand_path.display().to_string(), "triangles": cand.triangle_count(), "meshes": cand.meshes.iter().map(|m| json!({"name": m.name, "vertices": m.pos.len(), "triangles": m.tris.len(), "image": m.image.map(|i| sheet_name(&cand, i))})).collect::<Vec<_>>()},
        "reference": reference.as_ref().map(|(p, m)| json!({"file": p.display().to_string(), "triangles": m.triangle_count()})),
        "part": names.join(","),
        "part_triangles": ckeep.iter().map(|k| k.iter().filter(|&&b| b).count()).sum::<usize>(),
        "cameras_fitted_to": if std::ptr::eq(fit, &cand) { "candidate" } else { "reference" },
        "views": drawn.iter().map(|d| json!({"name": d.name, "size": match d.size { Size::Game => "game", _ => "close" }, "px": d.px, "camera": cam_json(&d.cam), "clip": "idle", "time": 0.0})).collect::<Vec<_>>(),
        "summary": summary,
        "notes": an.notes,
        "limits_are_provisional": true,
        "defects": defect_list,
        "schema": {
            "marked": "same shape as request.json marked: per-mesh triangle indices (mesh, then triangle in that mesh)",
            "camera": "same shape as request.json view.camera; fly the window here to see the defect",
            "screen_box": "x0, y0, x1, y1 in pixels of picture (views/<view>.png, picture_px square)",
            "texel_box": "box [x0, y0, x1, y1] (x1, y1 exclusive) on sheet (the image name, e.g. 98_1); texel_boxes lists every sheet touched",
            "score": "measure over the kind's limit: 1.0 sits on the limit, higher is worse",
        },
    });
    std::fs::write(out.join("defects.json"), serde_json::to_vec_pretty(&j).unwrap()).map_err(|e| format!("defects.json: {e}"))?;
    println!("gaw defects {}{}", cand_path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(), if names.is_empty() { String::new() } else { format!("  part {}", names.join(",")) });
    println!("  {} triangles checked, {} views drawn{}", ckeep.iter().map(|k| k.iter().filter(|&&b| b).count()).sum::<usize>(), drawn.len(), if reference.is_some() { ", reference run alongside" } else { "" });
    for l in &lines {
        println!("  {l}");
    }
    for n in &an.notes {
        println!("  note: {n}");
    }
    println!("  wrote {} (defects.json, {} crop sheet(s), {} view picture(s))", out.display(), kinds.iter().filter(|k| an.found.iter().any(|f| f.d.kind == **k)).count(), used_views.len());
    if a.flag("--json") {
        println!("{}", serde_json::to_string_pretty(&j).unwrap());
    }
    Ok(())
}
