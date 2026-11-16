//! `gaw ship`: PNG sheets to Memoria raw-header textures (DXT1 / DXT5 / RGBA32 with a
//! full mip chain), written at the same `.png` file name. The encoder and header code are in
//! `wb_core::unitytex`. Inputs are only read; outputs go to a new folder and never replace a file
//! unless `--force` is given.
use crate::Args;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use wb_core::unitytex::{self as ut, Format, MipOpts, ShipOpts};

pub const SHIP_USAGE: &str = "  gaw ship SHEET.png [SHEET2.png ...] --out-dir NEW_FOLDER [options]
      Writes each sheet as a Memoria raw texture at the same .png name: the 60-byte header (width,
      height, dataLength, Unity TextureFormat, mip count) and a full mip chain, rows flipped to
      Unity's bottom-left origin. The game loads it with no PNG decode and no run-time mip build.
      --format auto|dxt1|dxt5|rgba32  auto = DXT1 when every texel is opaque, else DXT5 (default)
      --png-only STEM[,STEM]          copy these sheets unchanged (they stay PNG), repeatable
      --skip-blink                    leave the blinking sheets as PNG: the GeoTexAnim main and sub
                                      sheets. Their names come from --blink-sheet (default 98_0, field
                                      Zidane: the head render texture is rebuilt by the HD plugin)
      --blink-sheet STEM[,STEM]       sheets that blink (with --skip-blink), repeatable
      --no-coverage                   plain mips; the default rescales each level's alpha so the share
                                      of texels at or above --cutoff stays that of the full-size sheet
      --cutoff N                      cutout threshold, 0-255 (default 128, the shaders' 0.5)
      --linear-mips                   average colors in linear light (default: stored values, which is
                                      what the game's own mip build does)
      --threads N                     encoder threads (default 4)
      --no-verify                     skip decoding the result back (PSNR, cutout mismatches, coverage)
      --force                         replace files that already exist in the output folder
      --json                          print the report as JSON (ship-report.json is always written)";

const VALUE_OPTS: [&str; 6] = ["--out-dir", "--format", "--png-only", "--blink-sheet", "--cutoff", "--threads"];

struct A<'a>(&'a [String]);

impl A<'_> {
    fn opt(&self, k: &str) -> Option<&str> {
        self.0.iter().position(|a| a == k).and_then(|i| self.0.get(i + 1)).map(String::as_str)
    }
    /// Comma lists from every occurrence of a repeatable option.
    fn list(&self, k: &str) -> Vec<String> {
        self.0.windows(2).filter(|w| w[0] == k).flat_map(|w| w[1].split(',')).map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()).collect()
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
            } else if VALUE_OPTS.contains(&a.as_str()) {
                skip = true;
            } else if !a.starts_with("--") {
                out.push(PathBuf::from(a));
            }
        }
        out
    }
}

fn mib(n: usize) -> f64 {
    n as f64 / 1_048_576.0
}

pub fn ship_cmd(args: &Args) -> Result<(), String> {
    let a = A(&args.0);
    let inputs = a.positionals();
    if inputs.is_empty() {
        return Err("give at least one sheet PNG".into());
    }
    let out_dir = PathBuf::from(a.opt("--out-dir").ok_or("--out-dir NEW_FOLDER is required")?);
    if out_dir.to_string_lossy().to_ascii_uppercase().starts_with("F:") {
        return Err("refusing to write to F:".into());
    }
    let format = match a.opt("--format") {
        None | Some("auto") => None,
        Some(s) => Some(Format::parse(s).ok_or_else(|| format!("--format {s}: use auto, dxt1, dxt5 or rgba32"))?),
    };
    let cutoff: u8 = a.opt("--cutoff").map(|s| s.parse().map_err(|e| format!("--cutoff: {e}"))).transpose()?.unwrap_or(128);
    let threads: usize = a.opt("--threads").map(|s| s.parse().map_err(|e| format!("--threads: {e}"))).transpose()?.unwrap_or(4);
    let opts = ShipOpts { format, mips: MipOpts { coverage: !a.flag("--no-coverage"), cutoff, linear: a.flag("--linear-mips") }, threads };
    let mut png_only = a.list("--png-only");
    if a.flag("--skip-blink") {
        let mut b = a.list("--blink-sheet");
        if b.is_empty() {
            b.push("98_0".to_owned());
        }
        png_only.extend(b);
    }
    let verify = !a.flag("--no-verify");
    let force = a.flag("--force");
    let json_out = a.flag("--json");

    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let mut rows: Vec<Value> = Vec::new();
    let (mut total_out, mut total_rgba) = (0usize, 0usize);
    for input in &inputs {
        let stem = input.file_stem().map(|s| s.to_string_lossy().into_owned()).ok_or("input has no file name")?;
        let name = input.file_name().unwrap().to_string_lossy().into_owned();
        let dst = out_dir.join(&name);
        if let (Ok(x), Ok(y)) = (input.canonicalize(), dst.canonicalize()) {
            if x == y {
                return Err(format!("{} would overwrite its own input", dst.display()));
            }
        }
        if dst.exists() && !force {
            return Err(format!("{} exists (use --force to replace it)", dst.display()));
        }
        let bytes = std::fs::read(input).map_err(|e| format!("{}: {e}", input.display()))?;
        if ut::parse_header(&bytes).is_ok() {
            return Err(format!("{} is already a raw texture", input.display()));
        }
        if png_only.iter().any(|s| s.eq_ignore_ascii_case(&stem)) {
            std::fs::write(&dst, &bytes).map_err(|e| format!("{}: {e}", dst.display()))?;
            rows.push(json!({"sheet": stem, "mode": "png-kept", "bytes": bytes.len()}));
            if !json_out {
                println!("{stem}: kept as PNG ({:.2} MiB, copied unchanged)", mib(bytes.len()));
            }
            continue;
        }
        let img = image::load_from_memory(&bytes).map_err(|e| format!("{}: {e}", input.display()))?.to_rgba8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgba = img.into_raw();
        let src_coverage = ut::coverage(&rgba, cutoff);
        let src_for_verify = if verify { Some(rgba.clone()) } else { None };
        let t0 = std::time::Instant::now();
        let shipped = ut::ship(rgba, w, h, &opts)?;
        let secs = t0.elapsed().as_secs_f64();
        std::fs::write(&dst, &shipped.bytes).map_err(|e| format!("{}: {e}", dst.display()))?;
        let rgba32 = ut::data_len(Format::Rgba32, w, h, shipped.levels);
        let data = shipped.bytes.len() - ut::HEADER_LEN;
        total_out += data;
        total_rgba += rgba32;
        let mut row = json!({
            "sheet": stem, "mode": "raw", "format": shipped.format.name(), "size": [w, h], "mips": shipped.levels,
            "has_alpha": shipped.has_alpha, "file_bytes": shipped.bytes.len(), "data_bytes": data,
            "vram_mib_estimate": mib(data), "rgba32_with_mips_mib": mib(rgba32), "png_bytes": bytes.len(),
            "cutout_coverage_level0": src_coverage, "encode_seconds": secs, "notes": shipped.notes,
        });
        let mut line = format!(
            "{stem}: {} {w}x{h}, {} mips, {} bytes ({:.2} MiB; RGBA32 with mips would be {:.1} MiB), encoded in {secs:.1} s",
            shipped.format.name(), shipped.levels, shipped.bytes.len(), mib(shipped.bytes.len()), mib(rgba32)
        );
        for n in &shipped.notes {
            line.push_str(&format!("\n  note: {n}"));
        }
        if let Some(src) = src_for_verify {
            let (hd, levels) = ut::decode_raw(&shipped.bytes)?;
            let psnr = ut::psnr_rgb(&src, &levels[0].rgba);
            let wrong = src.chunks_exact(4).zip(levels[0].rgba.chunks_exact(4)).filter(|(s, d)| (s[3] >= cutoff) != (d[3] >= cutoff)).count();
            let drift = levels.iter().skip(1).filter(|l| l.w >= 8 && l.h >= 8).map(|l| (ut::coverage(&l.rgba, cutoff) - src_coverage).abs()).fold(0f64, f64::max);
            row["verify"] = json!({
                "header_ok": hd.mips as usize == shipped.levels, "psnr_rgb_db": psnr,
                "cutout_mismatch_texels_level0": wrong, "max_coverage_drift_levels_8px_and_up": drift,
                "coverage_per_level": levels.iter().map(|l| ut::coverage(&l.rgba, cutoff)).collect::<Vec<_>>(),
            });
            line.push_str(&format!("\n  check: PSNR {psnr:.2} dB, {wrong} texels on the wrong side of the cutout at level 0, coverage drift {drift:.4} (levels 8 px and up)"));
        }
        if !json_out {
            println!("{line}");
        }
        rows.push(row);
    }
    let report = json!({
        "out_dir": out_dir.display().to_string(),
        "options": { "format": opts.format.map(|f| f.name()).unwrap_or("auto"), "coverage": opts.mips.coverage, "cutoff": cutoff, "linear_mips": opts.mips.linear, "png_only": png_only },
        "sheets": rows,
        "total_data_bytes_raw": total_out, "total_rgba32_with_mips_bytes": total_rgba,
        "note": "VRAM estimate = dataLength (the GPU copy). Unity also keeps a CPU copy of a loose texture unless it is made unreadable.",
    });
    std::fs::write(out_dir.join("ship-report.json"), serde_json::to_string_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
    if json_out {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        println!("total {:.2} MiB raw data vs {:.1} MiB as RGBA32 with mips; report: {}", mib(total_out), mib(total_rgba), Path::new(&out_dir).join("ship-report.json").display());
    }
    Ok(())
}
