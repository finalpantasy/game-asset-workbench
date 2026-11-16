//! zidane-inspector: command line and window. Inputs are opened read-only.
mod app;
mod bakejob;
mod doc;
mod patternjob;
mod relayoutjob;
mod swapjob;
mod theme;
mod viewport;
mod strapcmd;
mod haircmd;
mod judgecmd;
mod shipcmd;
mod requestcmd;
mod viewscmd;

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use wb_core::analysis::{self, Parts};
use wb_core::render::{self, Camera};
use wb_core::{image_textures, model_textures, Model};

pub const STOCK: &str = r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding\exports\Zidane-idle.glb";
pub const KIT_PARTS: &str = r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding\experiments\zidane-uv-kit-claude-v1\parts";

const USAGE: &str = "zidane-inspector: open an FF9 character export (GLB) read-only

  zidane-inspector [view] [file.glb] [--parts DIR] [--part NAME] [--screenshot out.png]
      Opens the window. With a file it opens Inspect, without one it opens Compare.
      Window options: --page inspect|compare|swap|mark, --a FILE, --b FILE, --clip NAME, --time SEC,
      --size field|battle|close, --advanced, --changes, --dead, --tol N.
      Checks: --click-model X,Y clicks the model view once (fractions of the view);
      --screenshot saves the window when loading is done and closes it.
      Mark & fix page: --page mark [--mark-script FILE] runs scripted marks, strokes and saves
      (one command per line, see workbench/README.md) and --requests-dir DIR saves requests there.
  zidane-inspector compare a.glb b.glb [--tol N] [--json]
      What changed from A to B: pixels per texture (and per kit part), meshes, skeleton, clips.
  zidane-inspector report [file.glb] [--json] [--coverage SIZE]
      Counts: triangles, data arrays, joints, clips, dead triangles by region,
      and with --coverage, unused texels per texture at SIZE (the kit uses 4096).
  zidane-inspector uv [file.glb] [--part NAME] [--json]
      UV islands (kit label, triangles, side, mirror twin, pixel density at the texture size),
      texels that show on two surfaces (at 1024), and with --part, what its pixels touch.
  zidane-inspector swap [file.glb] --image 98_1 --png painted.png --out new.glb [--json]
      Safe swap: writes a new file where only that texture changed, checks that every other
      byte block is identical, and says which parts the new paint shows on. Never overwrites.
  zidane-inspector relayout [file.glb] --out-dir NEW_FOLDER [--size 4096] [--blender PATH] [--json]
      New unique UV layout (Blender, in the background) with the old paint copied across, then
      checks: same skeleton, clips and triangles, no texel on two surfaces, and renders from four
      sides that match the input. Writes checks.json. --check done.glb only runs the checks.
  zidane-inspector pattern [file.glb] --out-dir NEW_FOLDER [--size 4096] [--json]
      Sewing-pattern UV layout, all built in (no Blender): pieces cut by body part, flattened,
      turned upright, left/right pairs side by side, face and hands at higher density, paint
      copied across. Also writes a pattern sheet with numbered seams, part/position/normal maps
      and legend.json, then runs the same checks as relayout.
      --keep-stock SHEET:x0,y0,x1,y1  leave that box (stock-sheet pixels) where it is: the
                                      triangles that sample it keep stock UVs, its paint is kept,
                                      nothing is packed over it. Repeatable.
      --keep-blink                    the same for field Zidane's two eye-blink boxes on 98_0.
      --v8                            all four additions below at once.
      --orient-bones                  turn each piece so its dominant bind-pose bone runs up the page.
      --mirror-pairs                  lay each left/right pair out as an exact reflection (same scale);
                                      --mirror-report only matches and reports the pairs.
      --metrics                       per-piece texel-density error, anisotropy p50/p95, bone-axis
                                      deviation and mirror numbers in checks.json (+ pattern-metrics.json).
      --id-maps                       chart, triangle and bone ID PNGs and PAINT-KIT-MANIFEST.json.
      --face-hand-density=X           density factor for the head and hands (default 1.5 and 1.75).
      --guide-size=N                  size of the guide and ID maps, a power of two from 256 to 8192
                                      (default: the sheet size, at most 2048). The layout stays the same.
      --low-stretch[=X]               relax pieces whose anisotropy p95 is above X (default 2.0) and cut those
                                      that stay above; the target joins the pattern checks. --v9 = --v8 + this.
  zidane-inspector bake [file.glb] --step guide|register|bake|demo --out-dir NEW_FOLDER
      Paint-projection bake onto the texture atlas. guide: --view front|back|left|right|NAME=yaw,pitch
      (repeatable) [--size 1024 --ortho --clip C --time T] writes NAME-guide.png, -ids.png, -depth.png,
      -view.json. register: --view-dir DIR --paint NAME=painted.png aligns the picture to the guide
      and refuses a bad scale. bake: the same plus --mask-part torso,arm --mask-mesh NAME
      --mask-png SHEET=mask.png --gate N --pad N --no-seam --leak-check writes the baked model,
      atlases, change maps, sheets, bake-report.json and FINAL-VERIFICATION.json. demo: self-test.
  zidane-inspector roundtrip [file.glb] --out copy.glb
      Writes the file back from its parsed parts and checks it equals the input byte for byte.
  zidane-inspector render [file.glb] --out view.png [--part NAME] [--dead] [--yaw DEG] [--size PX] [--clip NAME --time SEC]
      Saves the model view and the texture side by side, with the part highlighted.
  zidane-inspector shells [file.glb] [--json-out report.json]
      Lists each connected piece of every mesh: closed or open, thickness, bones, nearest piece
      (edges passing through it count as contact), skin mismatch and LOOSE / LOOSE CARD status.
  zidane-inspector strap-rebuild [file.glb] --out new.glb [--no-collide] [--no-paint-exit]
      Replaces the flat belt-strap card with a solid strap tucked under the waistband.
      Exit: from the waist paint by default, or --exit-edge V,V.. / --exit-curve x,y,z;.. /
      --exit-down D,D.. (across the strap); --exit-raise M lifts it; --flip-root picks the other end.
  zidane-inspector strap-sweep before.glb after.glb --out-dir DIR [--labels a,b] [--clips ..] [--times ..] [--size PX]
      Close-up sheets of the strap through each clip, with attachment flags and notes.

file.glb defaults to the stock export; --parts defaults to the UV kit's parts folder.";

const VALUE_OPTS: [&str; 28] = [
    "--out", "--part", "--parts", "--size", "--yaw", "--screenshot", "--coverage", "--pitch", "--click-model", "--page", "--a", "--b", "--clip", "--time", "--tol", "--region",
    "--texture", "--image", "--png", "--out-dir", "--blender", "--check", "--relayout", "--keep-stock",
    "--mark-script", "--requests-dir", "--dir", "--names",
];

pub struct Args(Vec<String>);

impl Args {
    pub fn opt(&self, k: &str) -> Option<&str> {
        self.0.iter().position(|a| a == k).and_then(|i| self.0.get(i + 1)).map(String::as_str)
    }
    /// Every value given for a repeatable option.
    pub fn opts(&self, k: &str) -> Vec<&str> {
        self.0.windows(2).filter(|w| w[0] == k).map(|w| w[1].as_str()).collect()
    }
    pub fn flag(&self, k: &str) -> bool {
        self.0.iter().any(|a| a == k)
    }
    /// Arguments that are not options or option values.
    pub fn positionals(&self) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut skip = false;
        for a in &self.0 {
            if skip {
                skip = false;
                continue;
            }
            if VALUE_OPTS.contains(&a.as_str()) {
                skip = true;
                continue;
            }
            if !a.starts_with("--") {
                out.push(PathBuf::from(a));
            }
        }
        out
    }
    pub fn file(&self) -> PathBuf {
        self.positionals().into_iter().next().unwrap_or_else(|| PathBuf::from(STOCK))
    }
    pub fn tol(&self) -> Result<u8, String> {
        self.opt("--tol").map(|s| s.parse::<u8>().map_err(|e| format!("--tol: {e}"))).transpose().map(|t| t.unwrap_or(0))
    }
    pub fn parts_dir(&self) -> PathBuf {
        PathBuf::from(self.opt("--parts").unwrap_or(KIT_PARTS))
    }
}

fn main() {
    let all: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, rest) = match all.first().map(String::as_str) {
        Some(c @ ("report" | "roundtrip" | "render" | "compare" | "uv" | "swap" | "relayout" | "pattern" | "view" | "help" | "-h" | "--help")) => (c.to_owned(), all[1..].to_vec()),
        Some(c @ ("shells" | "strap-sweep" | "strap-rebuild")) => (c.to_owned(), all[1..].to_vec()),
        Some(c @ "bake") => (c.to_owned(), all[1..].to_vec()),
        Some(c @ "hair") => (c.to_owned(), all[1..].to_vec()),
        Some(c @ "judge") => (c.to_owned(), all[1..].to_vec()),
        Some(c @ "ship") => (c.to_owned(), all[1..].to_vec()),
        Some(c @ "requests") => (c.to_owned(), all[1..].to_vec()),
        Some(c @ "views") => (c.to_owned(), all[1..].to_vec()),
        // A word that is not a file is a mistyped command: say so instead of opening the window.
        Some(c) if !c.starts_with("--") && !Path::new(c).exists() => ("help".to_owned(), vec![]),
        _ => ("view".to_owned(), all.clone()),
    };
    if cmd == "help" && all.first().is_some_and(|c| !matches!(c.as_str(), "help" | "-h" | "--help")) {
        eprintln!("error: unknown command {}", all[0]);
        println!("{USAGE}");
        println!("{}", haircmd::HAIR_USAGE);
        println!("{}", judgecmd::JUDGE_USAGE);
        println!("{}", shipcmd::SHIP_USAGE);
        println!("{}", requestcmd::REQUESTS_USAGE);
        println!("{}", viewscmd::VIEWS_USAGE);
        std::process::exit(2);
    }
    let args = Args(rest);
    let result = match cmd.as_str() {
        "report" => report(&args),
        "compare" => compare_cmd(&args),
        "uv" => uv_cmd(&args),
        "swap" => swap_cmd(&args),
        "relayout" => relayout_cmd(&args),
        "pattern" => pattern_cmd(&args),
        "bake" => bakejob::bake_cmd(&args),
        "hair" => haircmd::hair_cmd(&args),
        "judge" => judgecmd::judge_cmd(&args),
        "ship" => shipcmd::ship_cmd(&args),
        "requests" => requestcmd::requests_cmd(&args),
        "views" => viewscmd::views_cmd(&args),
        "roundtrip" => roundtrip(&args),
        "render" => render_cmd(&args),
        "shells" => strapcmd::shells_cmd(&args),
        "strap-sweep" => strapcmd::sweep_cmd(&args),
        "strap-rebuild" => strapcmd::rebuild_cmd(&args),
        "view" => app::run(&args),
        _ => {
            println!("{USAGE}");
            println!("{}", haircmd::HAIR_USAGE);
            println!("{}", judgecmd::JUDGE_USAGE);
        println!("{}", shipcmd::SHIP_USAGE);
            println!("{}", requestcmd::REQUESTS_USAGE);
            println!("{}", viewscmd::VIEWS_USAGE);
            Ok(())
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Everything the report prints, as JSON.
pub fn report_json(path: &Path, model: &Model, coverage: Option<usize>) -> Value {
    let mut dead_regions: BTreeMap<String, usize> = BTreeMap::new();
    let mut meshes = Vec::new();
    let textures = model_textures(model);
    for (mi, m) in model.meshes.iter().enumerate() {
        let mut dead = 0;
        for ti in 0..m.tris.len() {
            if analysis::is_dead(model, mi, ti) {
                dead += 1;
                let bone = &model.joint_names.get(model.triangle_joint(mi, ti)).cloned().unwrap_or_default();
                *dead_regions.entry(analysis::region(bone)).or_default() += 1;
            }
        }
        let tex = textures[mi].as_ref();
        meshes.push(json!({
            "name": m.name, "vertices": m.pos.len(), "triangles": m.tris.len(), "atlas": model.atlas_name(mi),
            "texture_px": tex.map(|t| [t.w, t.h]),
            "texels_alpha_below_128": tex.map(|t| t.rgba.chunks_exact(4).filter(|p| p[3] < 128).count()),
            "dead_triangles": dead,
        }));
    }
    let mut out = json!({
        "file": path.display().to_string(), "bytes": model.file_len, "glb_version": model.glb.version,
        "data_arrays": model.accessors, "nodes": model.nodes, "triangles": model.triangle_count(),
        "joints": model.joint_names.len(), "joint_names": model.joint_names, "clips": model.clips,
        "images": model.images.iter().map(|i| json!({"name": i.name, "mime": i.mime, "bytes": i.bytes.len()})).collect::<Vec<_>>(),
        "meshes": meshes,
        "dead_triangles": { "rule": "UV area under 0.01 texel² at 512", "total": dead_regions.values().sum::<usize>(), "by_region": dead_regions },
    });
    if let Some(size) = coverage {
        let mut by_atlas: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for mi in 0..model.meshes.len() {
            by_atlas.entry(model.atlas_name(mi)).or_default().push(mi);
        }
        let unused: BTreeMap<String, usize> =
            by_atlas.iter().map(|(a, ms)| (a.clone(), size * size - analysis::covered_texels(model, ms, size))).collect();
        out["unused_texels"] = json!({ "size_px": size, "by_atlas": unused });
    }
    out
}

fn report(args: &Args) -> Result<(), String> {
    let path = args.file();
    let model = Model::open(&path)?;
    let coverage = args.opt("--coverage").map(|s| s.parse::<usize>().map_err(|e| format!("--coverage: {e}"))).transpose()?;
    let r = report_json(&path, &model, coverage);
    if args.flag("--json") {
        println!("{}", serde_json::to_string_pretty(&r).unwrap());
        return Ok(());
    }
    println!("{}  ({} bytes, GLB v{})", r["file"].as_str().unwrap(), r["bytes"], r["glb_version"]);
    println!("triangles    {}", r["triangles"]);
    for m in r["meshes"].as_array().unwrap() {
        println!(
            "  {:<8} {:>5} triangles  {:>5} vertices  texture {} {}  dead {}",
            m["name"].as_str().unwrap(), m["triangles"], m["vertices"], m["atlas"].as_str().unwrap(), m["texture_px"], m["dead_triangles"]
        );
    }
    println!("data arrays  {}", r["data_arrays"]);
    println!("joints       {}", r["joints"]);
    println!("clips        {}", r["clips"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect::<Vec<_>>().join(", "));
    let d = &r["dead_triangles"];
    let regions: Vec<String> = d["by_region"].as_object().unwrap().iter().map(|(k, v)| format!("{v} {k}")).collect();
    println!("dead         {} ({})  rule: {}", d["total"], regions.join(", "), d["rule"].as_str().unwrap());
    if let Some(u) = r.get("unused_texels") {
        for (a, n) in u["by_atlas"].as_object().unwrap() {
            println!("unused       {a}: {n} texels at {}", u["size_px"]);
        }
    }
    Ok(())
}

fn uv_cmd(args: &Args) -> Result<(), String> {
    use wb_core::uv;
    let path = args.file();
    let model = Model::open(&path)?;
    let parts = Parts::load(&args.parts_dir()).ok();
    let tex = image_textures(&model);
    let sizes: Vec<usize> = model.meshes.iter().map(|m| m.image.and_then(|i| tex[i].as_ref()).map(|t| t.w.max(t.h)).unwrap_or(512)).collect();
    let f = uv::Facts::build(&model, parts.as_ref(), &sizes);
    let mut meshes = Vec::new();
    for (mi, m) in model.meshes.iter().enumerate() {
        let t = &f.topo[mi];
        let islands: Vec<Value> = f.islands[mi]
            .iter()
            .enumerate()
            .map(|(i, isl)| {
                let d: Vec<f32> = (0..m.tris.len()).filter(|&ti| t.island[ti] as usize == i && t.dup_of[ti].is_none() && !analysis::is_dead(&model, mi, ti)).map(|ti| f.density[mi][ti]).collect();
                let (lo, hi) = d.iter().fold((f32::MAX, 0f32), |(a, b), &x| (a.min(x), b.max(x)));
                json!({ "id": i, "label": isl.label, "triangles": isl.tris, "side": isl.side, "mirror_twin": isl.twin,
                        "density": if d.is_empty() { Value::Null } else { json!([lo.round(), hi.round()]) } })
            })
            .collect();
        let pairs: Vec<Value> = f.pairs[mi].iter().map(|p| json!({ "islands": [p.a, p.b], "texels": p.texels, "kind": p.kind.label(), "triangle_pairs": p.tri_pairs, "max_distance": (p.max_dist as f64 * 100.0).round() / 100.0 })).collect();
        meshes.push(json!({ "name": m.name, "atlas": model.atlas_name(mi), "density_size": sizes[mi], "islands": islands, "back_faces": t.dup_of.iter().filter(|x| x.is_some()).count(), "reuse": pairs }));
    }
    let mut out = json!({ "file": path.display().to_string(), "reuse_size": uv::OVERLAP_SIZE, "meshes": meshes });
    if let Some(name) = args.opt("--part") {
        let p = parts.as_ref().ok_or("--part needs the kit's parts folder")?;
        let id = p.id_by_name(name).ok_or_else(|| format!("no part named {name}"))?;
        let tp: Vec<Vec<u8>> = (0..model.meshes.len()).map(|mi| p.triangle_parts(&model, mi)).collect();
        let touched = f.touched_by_part(&tp, id);
        let mut also: BTreeMap<String, usize> = BTreeMap::new();
        let mut total = 0;
        for mi in 0..model.meshes.len() {
            for ti in 0..model.meshes[mi].tris.len() {
                if touched[mi][ti] && f.topo[mi].dup_of[ti].is_none() {
                    total += 1;
                    if tp[mi][ti] != id {
                        *also.entry(if tp[mi][ti] == 0 { "no part".into() } else { p.name(tp[mi][ti]).to_owned() }).or_default() += 1;
                    }
                }
            }
        }
        let s = f.summary(&model, &|mi, ti| tp[mi][ti] == id);
        out["part"] = json!({ "name": name, "touched_triangles": total, "outside_part": also,
            "reuse_texels": { "mirrored": s.reuse[0], "shared": s.reuse[1], "self_overlap": s.reuse[2] },
            "density": s.density.map(|d| json!({ "min": d.0.round(), "median": d.1.round(), "max": d.2.round(), "size": d.3 })) });
    }
    if args.flag("--json") {
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
        return Ok(());
    }
    println!("{}", path.display());
    for m in out["meshes"].as_array().unwrap() {
        println!("{} ({}): {} islands, {} back faces, density at {} px", m["name"].as_str().unwrap(), m["atlas"].as_str().unwrap(), m["islands"].as_array().unwrap().len(), m["back_faces"], m["density_size"]);
        for i in m["islands"].as_array().unwrap() {
            let twin = i["mirror_twin"].as_u64().map(|t| format!("  twin {t}")).unwrap_or_default();
            println!("  {:>2} {:<60} {:>4} tris  {:<6} density {}{}", i["id"], i["label"].as_str().unwrap(), i["triangles"], i["side"].as_str().unwrap(), i["density"], twin);
        }
        println!("  texels on two surfaces at {}:", uv::OVERLAP_SIZE);
        for p in m["reuse"].as_array().unwrap() {
            println!("    islands {:<8} {:>7} texels  {:<14} max {} apart", format!("{}-{}", p["islands"][0], p["islands"][1]), p["texels"], p["kind"].as_str().unwrap(), p["max_distance"]);
        }
    }
    if let Some(p) = out.get("part") {
        println!("part {}: its pixels show on {} triangles; outside the part: {}", p["name"].as_str().unwrap(), p["touched_triangles"], p["outside_part"]);
        println!("  reused texels: {}  density: {}", p["reuse_texels"], p["density"]);
    }
    Ok(())
}

fn roundtrip(args: &Args) -> Result<(), String> {
    let path = args.file();
    let out = PathBuf::from(args.opt("--out").ok_or("roundtrip needs --out")?);
    if out.canonicalize().ok().is_some() && out.canonicalize().ok() == path.canonicalize().ok() {
        return Err("--out is the input file; inputs are never written".into());
    }
    let input = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let written = wb_core::glb::Glb::parse(&input)?.to_bytes();
    std::fs::write(&out, &written).map_err(|e| format!("{}: {e}", out.display()))?;
    let back = std::fs::read(&out).map_err(|e| e.to_string())?;
    if back == input {
        println!("identical: {} bytes, {} written to {}", input.len(), back.len(), out.display());
        Ok(())
    } else {
        let first = back.iter().zip(&input).position(|(a, b)| a != b).unwrap_or(back.len().min(input.len()));
        Err(format!("differs: input {} bytes, output {} bytes, first difference at byte {first}", input.len(), back.len()))
    }
}

/// Which triangles a selection covers.
pub fn part_triangles(model: &Model, tri_parts: &[Vec<u8>], part: u8) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for (mi, tp) in tri_parts.iter().enumerate() {
        for (ti, &p) in tp.iter().enumerate() {
            if p == part {
                v.push((mi, ti));
            }
        }
    }
    let _ = model;
    v
}

fn relayout_cmd(args: &Args) -> Result<(), String> {
    let input = args.file();
    // --check OUT.glb: only run the checks on an existing re-layout.
    if let Some(done) = args.opt("--check") {
        let c = relayoutjob::check_files(&input, Path::new(done))?;
        return finish_relayout(args, &c, None);
    }
    let out = PathBuf::from(args.opt("--out-dir").ok_or("relayout needs --out-dir NEW_FOLDER")?);
    let size = layout_size(args)?;
    Model::open(&input)?;
    println!("{} -> {} at {size} px", input.display(), out.display());
    let (glb, c) = relayoutjob::run(&input, &out, size, args.opt("--blender"), Box::new(|l| println!("  blender {l}")))?;
    println!("checked {}", glb.display());
    finish_relayout(args, &c, None)
}

fn layout_size(args: &Args) -> Result<usize, String> {
    let size: usize = args.opt("--size").unwrap_or("4096").parse().map_err(|e| format!("--size: {e}"))?;
    if !size.is_power_of_two() || !(256..=8192).contains(&size) {
        return Err("--size must be a power of two from 256 to 8192".into());
    }
    Ok(size)
}

/// Field Zidane's eye blink (measured in the game, 512-px PNG coordinates on the head sheet):
/// the closed-eye art at the source is copied over the eyes at the target.
const ZIDANE_BLINK: [(&str, [f64; 4]); 2] = [("98_0", [224.0, 216.0, 272.0, 248.0]), ("98_0", [32.0, 116.0, 80.0, 148.0])];

/// `--keep-stock SHEET:x0,y0,x1,y1` (repeatable, pixels of the stock sheet) and `--keep-blink`.
fn keep_boxes(args: &Args, input: &Path) -> Result<Vec<wb_core::pattern::Keep>, String> {
    let mut spec: Vec<(String, [f64; 4], f64)> = Vec::new();
    for v in args.opts("--keep-stock") {
        let (sheet, nums) = v.split_once(':').ok_or("--keep-stock wants SHEET:x0,y0,x1,y1")?;
        let n: Vec<f64> = nums.split(',').map(|s| s.trim().parse::<f64>().map_err(|e| format!("--keep-stock {v}: {e}"))).collect::<Result<_, _>>()?;
        let [x0, y0, x1, y1] = n[..] else { return Err(format!("--keep-stock {v}: needs four numbers")) };
        if x1 <= x0 || y1 <= y0 {
            return Err(format!("--keep-stock {v}: empty box"));
        }
        spec.push((sheet.to_string(), [x0, y0, x1, y1], 0.0));
    }
    if args.flag("--keep-blink") {
        // Measured at 512; scaled to the sheet's own size below.
        spec.extend(ZIDANE_BLINK.iter().map(|(s, b)| (s.to_string(), *b, 512.0)));
    }
    if spec.is_empty() {
        return Ok(Vec::new());
    }
    let model = Model::open(input)?;
    let tex = image_textures(&model);
    let mut out = Vec::new();
    for (sheet, b, at) in spec {
        let mi = (0..model.meshes.len()).find(|&mi| model.meshes[mi].image.is_some() && model.atlas_name(mi) == sheet).ok_or(format!("no sheet named {sheet} in {}", input.display()))?;
        let (tw, th) = model.meshes[mi].image.and_then(|i| tex.get(i)).and_then(|o| o.as_ref()).map(|t| (t.w as f64, t.h as f64)).ok_or(format!("sheet {sheet} has no image"))?;
        // Pixel boxes are in the sheet's own pixels, or in the size they were measured at.
        let (w, h) = if at > 0.0 { (at, at) } else { (tw, th) };
        out.push(wb_core::pattern::Keep { sheet, uv: [b[0] / w, b[1] / h, b[2] / w, b[3] / h] });
    }
    Ok(out)
}

/// Sewing-pattern layout, all in Rust: no Blender or other outside tools.
fn pattern_cmd(args: &Args) -> Result<(), String> {
    let input = args.file();
    let out = PathBuf::from(args.opt("--out-dir").ok_or("pattern needs --out-dir NEW_FOLDER")?);
    let size = layout_size(args)?;
    let keep = keep_boxes(args, &input)?;
    println!("{} -> {} at {size} px", input.display(), out.display());
    for k in &keep {
        println!("  keeping {} box {:?} (0..1) at its stock place", k.sheet, k.uv);
    }
    let t = std::time::Instant::now();
    let extra = patternjob::Extra::from_args(args)?;
    let (glb, c) = patternjob::run_with(&input, &out, size, keep, &extra, Box::new(|l| println!("  {l}")))?;
    println!("wrote {} in {:.1} s", glb.display(), t.elapsed().as_secs_f64());
    finish_relayout(args, &c, None)
}

fn finish_relayout(args: &Args, c: &wb_core::relayout::Checks, out: Option<&Path>) -> Result<(), String> {
    let j = relayoutjob::to_json(c);
    if let Some(o) = out {
        std::fs::write(o.join("checks.json"), serde_json::to_string_pretty(&j).unwrap()).map_err(|e| e.to_string())?;
    }
    if args.flag("--json") {
        println!("{}", serde_json::to_string_pretty(&j).unwrap());
    } else {
        relayoutjob::print(c);
    }
    if c.passed() {
        Ok(())
    } else {
        Err("the re-layout did not pass its checks".into())
    }
}

fn swap_cmd(args: &Args) -> Result<(), String> {
    let input = args.file();
    let image = args.opt("--image").ok_or("swap needs --image NAME (for example 98_1)")?;
    let png_path = PathBuf::from(args.opt("--png").ok_or("swap needs --png painted.png")?);
    let out = swapjob::check_out(&input, Path::new(args.opt("--out").ok_or("swap needs --out new.glb")?))?;
    let model = Model::open(&input)?;
    let png = std::fs::read(&png_path).map_err(|e| format!("{}: {e}", png_path.display()))?;
    let parts = Parts::load(&args.parts_dir()).ok();
    let (o, _) = swapjob::run(&model, image, &png, parts.as_ref())?;
    std::fs::write(&out, &o.bytes).map_err(|e| format!("{}: {e}", out.display()))?;
    if args.flag("--json") {
        let r = json!({
            "input": input.display().to_string(), "png": png_path.display().to_string(), "out": out.display().to_string(),
            "image": o.image, "old_px": o.old_size, "new_px": o.new_size, "old_bytes": o.old_len, "new_bytes": o.new_len,
            "file_bytes": o.bytes.len(), "verified": true, "alpha_flips": o.alpha_flips, "changed_pixels": o.changed_pixels,
            "total_pixels": o.total_pixels, "resized": o.resized,
            "by_part": o.by_part.iter().map(|(p, n)| json!({"part": p, "changed": n})).collect::<Vec<_>>(),
            "shows_on": o.shows_on.iter().map(|(p, n)| json!({"part": p, "triangles": n})).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&r).unwrap());
        return Ok(());
    }
    println!("wrote {} ({} bytes)", out.display(), o.bytes.len());
    println!("  {}: {}x{} -> {}x{}, {} -> {} bytes", o.image, o.old_size[0], o.old_size[1], o.new_size[0], o.new_size[1], o.old_len, o.new_len);
    println!("  checked: every other byte block, the JSON and every mesh match the input");
    if o.resized {
        println!("  the new texture is a different size, so pixel counts below compare a scaled copy");
    }
    println!("  changed pixels: {} of {}", o.changed_pixels, o.total_pixels);
    for (p, n) in o.by_part.iter().take(8) {
        println!("    {n:>10}  {p}");
    }
    if !o.shows_on.is_empty() {
        println!("  new paint shows on (triangles):");
        for (p, n) in &o.shows_on {
            println!("    {n:>6}  {p}");
        }
    }
    if o.alpha_flips > 0 {
        println!("  warning: {} pixels switch between see-through and solid (alpha 128 cutoff)", o.alpha_flips);
    }
    Ok(())
}

fn compare_cmd(args: &Args) -> Result<(), String> {
    let files = args.positionals();
    let [a_path, b_path] = match files.as_slice() {
        [a, b] => [a.clone(), b.clone()],
        _ => return Err("compare needs two files: compare a.glb b.glb".into()),
    };
    let tol = args.tol()?;
    let parts = Parts::load(&args.parts_dir()).ok();
    let t0 = std::time::Instant::now();
    let (a, b) = (Model::open(&a_path)?, Model::open(&b_path)?);
    let (at, bt) = (image_textures(&a), image_textures(&b));
    let d = wb_core::diff::diff(&a, &at, &b, &bt, parts.as_ref(), tol);
    let part_name = |id: u8| parts.as_ref().map(|p| p.name(id).to_owned()).unwrap_or_else(|| format!("part {id}"));
    let untouched: Vec<Vec<String>> = match &parts {
        Some(p) => {
            let side = |m: &Model| (wb_core::diff::changed_triangles(m, &d), (0..m.meshes.len()).map(|mi| p.triangle_parts(m, mi)).collect::<Vec<_>>());
            let ((ca, pa), (cb, pb)) = (side(&a), side(&b));
            wb_core::diff::untouched_parts(
                &d,
                p,
                [
                    wb_core::diff::Side { model: &a, changed: &ca, tri_parts: &pa },
                    wb_core::diff::Side { model: &b, changed: &cb, tri_parts: &pb },
                ],
            )
            .into_iter()
            .map(|ids| ids.into_iter().map(part_name).collect())
            .collect()
        }
        None => vec![vec![]; d.textures.len()],
    };
    let r = json!({
        "a": a_path.display().to_string(), "b": b_path.display().to_string(), "tolerance": tol,
        "identical_files": d.identical_files, "bytes": d.bytes, "changed_pixels": d.changed_pixels(),
        "textures": d.textures.iter().zip(&untouched).map(|(t, u)| json!({
            "untouched_parts": u,
            "name": t.name, "status": t.status(), "a_px": t.a_size, "b_px": t.b_size, "compared_at": t.size,
            "resized": t.resized, "changed": t.changed, "total": t.total, "bbox": t.bbox,
            "by_part": t.by_part.iter().map(|(id, n)| json!({"part": part_name(*id), "id": id, "changed": n})).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "meshes": d.meshes.iter().map(|m| json!({
            "name": m.name, "status": m.status(), "a": m.a, "b": m.b, "moved_vertices": m.moved_vertices,
            "uv_changed_vertices": m.uv_changed_vertices, "triangles_same": m.triangles_same,
        })).collect::<Vec<_>>(),
        "joints_same": d.joints_same, "clips": d.clips, "ms": t0.elapsed().as_millis() as u64,
    });
    if args.flag("--json") {
        println!("{}", serde_json::to_string_pretty(&r).unwrap());
        return Ok(());
    }
    println!("A  {}  ({} bytes)", a_path.display(), d.bytes[0]);
    println!("B  {}  ({} bytes)", b_path.display(), d.bytes[1]);
    if d.identical_files {
        println!("identical files");
    }
    println!("textures (tolerance {tol})");
    for (t, u) in d.textures.iter().zip(&untouched) {
        let size = |s: Option<[usize; 2]>| s.map(|s| format!("{}x{}", s[0], s[1])).unwrap_or("-".into());
        println!(
            "  {:<28} {:<10} A {:<10} B {:<10} changed {:>10} of {:>10} ({:.3}%){}",
            t.name, t.status(), size(t.a_size), size(t.b_size), t.changed, t.total,
            if t.total > 0 { 100.0 * t.changed as f64 / t.total as f64 } else { 0.0 },
            if t.resized { "  sizes differ: compared at the larger size" } else { "" }
        );
        if let Some(bb) = t.bbox {
            println!("    area x {}..{}, y {}..{}", bb[0], bb[2], bb[1], bb[3]);
        }
        for (id, n) in t.by_part.iter().take(6) {
            println!("    {:>10}  {}", n, part_name(*id));
        }
        let mesh_changed = d.meshes.iter().any(|m| m.status() != "same");
        if (t.status() == "changed" || (t.status() == "same" && mesh_changed)) && !u.is_empty() {
            println!("    untouched: {}", u.join(", "));
        }
    }
    println!("meshes");
    for m in &d.meshes {
        let vt = |s: Option<(usize, usize)>| s.map(|s| format!("{} verts {} tris", s.0, s.1)).unwrap_or("-".into());
        println!("  {:<10} {:<16} A {:<22} B {:<22} moved {} uv {}", m.name, m.status(), vt(m.a), vt(m.b), m.moved_vertices, m.uv_changed_vertices);
    }
    println!("skeleton   {}", if d.joints_same { "same joints" } else { "joints differ" });
    println!("clips      A: {}  B: {}", d.clips[0].join(", "), d.clips[1].join(", "));
    println!("took {} ms", t0.elapsed().as_millis());
    Ok(())
}

fn render_cmd(args: &Args) -> Result<(), String> {
    let path = args.file();
    let model = Model::open(&path)?;
    let textures = image_textures(&model);
    let out = PathBuf::from(args.opt("--out").ok_or("render needs --out")?);
    let size: usize = args.opt("--size").unwrap_or("800").parse().map_err(|e| format!("--size: {e}"))?;
    let mut cam = Camera::frame(&model);
    if let Some(y) = args.opt("--yaw") {
        cam.yaw = std::f32::consts::PI + y.parse::<f32>().map_err(|e| format!("--yaw: {e}"))?.to_radians();
    }
    let show_dead = args.flag("--dead");
    let (part, tri_parts) = match args.opt("--part") {
        Some(name) => {
            let parts = Parts::load(&args.parts_dir())?;
            let id = parts.id_by_name(name).ok_or_else(|| format!("no part named {name}; parts: {}", parts.names.iter().map(|n| n.1.as_str()).collect::<Vec<_>>().join(", ")))?;
            let tp: Vec<Vec<u8>> = (0..model.meshes.len()).map(|mi| parts.triangle_parts(&model, mi)).collect();
            (Some(id), tp)
        }
        None => (None, vec![]),
    };
    let selected = |mi: usize, ti: usize| part.is_some() && tri_parts[mi][ti] == part.unwrap();
    let dead = |mi: usize, ti: usize| show_dead && analysis::is_dead(&model, mi, ti);
    let tint = |mi: usize, ti: usize| {
        if dead(mi, ti) {
            Some([255, 30, 30])
        } else if selected(mi, ti) {
            Some([255, 220, 0])
        } else {
            None
        }
    };
    let clip = match args.opt("--clip") {
        Some(name) => Some(model.clips.iter().position(|c| c == name).ok_or_else(|| format!("no clip named {name}; clips: {}", model.clips.join(", ")))?),
        None => None,
    };
    let time: f32 = args.opt("--time").unwrap_or("0").parse().map_err(|e| format!("--time: {e}"))?;
    let pose = model.posed(clip, time);
    let o = render::Options { cam, w: size, h: size, background: ([40, 44, 52], [40, 44, 52]), filter: render::Filter::Smooth, positions: Some(&pose) };
    let frame = render::render(&model, &textures, &o, &tint);
    // Texture panel: the atlas of the mesh holding most of the part (or mesh 0), selected triangles filled.
    let atlas_mesh = match part {
        Some(p) => (0..model.meshes.len()).max_by_key(|&mi| tri_parts[mi].iter().filter(|&&x| x == p).count()).unwrap_or(0),
        None => 0,
    };
    let textures: Vec<Option<wb_core::Texture>> = {
        let mut per_mesh = model_textures(&model);
        per_mesh.truncate(model.meshes.len());
        per_mesh
    };
    let mut sheet = image::RgbaImage::from_pixel((size * 2) as u32, size as u32, image::Rgba([40, 44, 52, 255]));
    for y in 0..size {
        for x in 0..size {
            let o = (y * size + x) * 4;
            sheet.put_pixel(x as u32, y as u32, image::Rgba([frame.rgba[o], frame.rgba[o + 1], frame.rgba[o + 2], 255]));
        }
    }
    if let Some(tx) = &textures[atlas_mesh] {
        let m = &model.meshes[atlas_mesh];
        let mut mask = vec![false; size * size];
        for (ti, t) in m.tris.iter().enumerate() {
            if selected(atlas_mesh, ti) {
                let p = t.map(|v| [m.uv[v as usize][0] * size as f32, m.uv[v as usize][1] * size as f32]);
                fill_tri(&mut mask, size, p);
            }
        }
        for y in 0..size {
            for x in 0..size {
                let o = ((y * tx.h / size) * tx.w + x * tx.w / size) * 4;
                let mut c = [tx.rgba[o], tx.rgba[o + 1], tx.rgba[o + 2]];
                if mask[y * size + x] {
                    c = [((c[0] as u16 + 255) / 2) as u8, ((c[1] as u16 + 220) / 2) as u8, (c[2] as u16 / 2) as u8];
                }
                sheet.put_pixel((size + x) as u32, y as u32, image::Rgba([c[0], c[1], c[2], 255]));
            }
        }
    }
    sheet.save(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let n = part.map(|p| part_triangles(&model, &tri_parts, p).len()).unwrap_or(0);
    println!("wrote {} ({}x{}), atlas {}, {} part triangles", out.display(), size * 2, size, model.atlas_name(atlas_mesh), n);
    Ok(())
}

/// Fills a 2D triangle (pixel coordinates) into a square mask, pixel centers inside.
pub fn fill_tri(mask: &mut [bool], size: usize, p: [[f32; 2]; 3]) {
    let [a, b, c] = p;
    let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if area.abs() < 1e-9 {
        return;
    }
    let x0 = a[0].min(b[0]).min(c[0]).floor().max(0.0) as usize;
    let x1 = (a[0].max(b[0]).max(c[0]).ceil() as usize).min(size);
    let y0 = a[1].min(b[1]).min(c[1]).floor().max(0.0) as usize;
    let y1 = (a[1].max(b[1]).max(c[1]).ceil() as usize).min(size);
    for y in y0..y1 {
        for x in x0..x1 {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let w0 = ((b[0] - px) * (c[1] - py) - (b[1] - py) * (c[0] - px)) / area;
            let w1 = ((c[0] - px) * (a[1] - py) - (c[1] - py) * (a[0] - px)) / area;
            if w0 >= 0.0 && w1 >= 0.0 && w0 + w1 <= 1.0 {
                mask[y * size + x] = true;
            }
        }
    }
}
