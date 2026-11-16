//! UV re-layout as the command line runs it: Blender does the layout and the paint copy
//! (tools/uv_relayout.py), then wb-core checks the result against the input.
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use wb_core::relayout::{self, Checks};
use wb_core::Model;

const LIMIT: Duration = Duration::from_secs(600);

pub fn script() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("tools").join("uv_relayout.py")
}

/// `--blender PATH`, else the newest Blender under Program Files.
pub fn find_blender(given: Option<&str>) -> Result<PathBuf, String> {
    if let Some(g) = given {
        let p = PathBuf::from(g);
        return if p.is_file() { Ok(p) } else { Err(format!("{} not found", p.display())) };
    }
    let root = Path::new(r"C:\Program Files\Blender Foundation");
    let mut found: Vec<PathBuf> = std::fs::read_dir(root)
        .map_err(|_| "Blender not found; pass --blender PATH".to_string())?
        .filter_map(|e| e.ok().map(|e| e.path().join("blender.exe")))
        .filter(|p| p.is_file())
        .collect();
    found.sort_by_key(|p| {
        let v = p.parent().and_then(|d| d.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        v.trim_start_matches("Blender ").split('.').map(|x| x.parse::<u32>().unwrap_or(0)).collect::<Vec<_>>()
    });
    found.pop().ok_or_else(|| "Blender not found; pass --blender PATH".into())
}

/// The output folder must be new (or empty) and outside the installed game (Modding is fine).
pub fn check_dir(out: &Path) -> Result<PathBuf, String> {
    if out.exists() && std::fs::read_dir(out).map(|mut d| d.next().is_some()).unwrap_or(true) {
        return Err(format!("{} is not empty; pick a new folder (nothing is overwritten)", out.display()));
    }
    let low = std::path::absolute(out).map_err(|e| e.to_string())?.to_string_lossy().replace('/', "\\").to_ascii_lowercase();
    let root = crate::swapjob::GAME_ROOT.to_ascii_lowercase();
    if low.starts_with(&root) && !low.starts_with(&format!("{root}\\modding\\")) {
        return Err("the output would land in the installed game; use a folder under Modding".into());
    }
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(out.to_path_buf())
}

/// Runs Blender in the background at below-normal priority, with no window, passing its
/// progress lines to `progress`. Kills it after ten minutes.
pub fn run_blender(blender: &Path, input: &Path, out: &Path, size: usize, mut progress: Box<dyn FnMut(String) + Send>) -> Result<(), String> {
    let abs = |p: &Path| std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    let (input, out) = (abs(input), abs(out));
    let mut cmd = Command::new(blender);
    cmd.args(["-b", "--factory-startup", "--python-exit-code", "1", "--python"])
        .arg(script())
        .arg("--")
        .arg(&input)
        .arg(&out)
        .arg(size.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW);
    }
    let mut child = cmd.spawn().map_err(|e| format!("{}: {e}", blender.display()))?;
    let out_pipe = child.stdout.take().unwrap();
    let err_pipe = child.stderr.take().unwrap();
    let t_out = std::thread::spawn(move || {
        for line in BufReader::new(out_pipe).lines().map_while(Result::ok) {
            if line.starts_with('[') {
                progress(line);
            }
        }
    });
    let t_err = std::thread::spawn(move || BufReader::new(err_pipe).lines().map_while(Result::ok).collect::<Vec<_>>());
    let start = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
            break s;
        }
        if start.elapsed() > LIMIT {
            let _ = child.kill();
            let _ = child.wait();
            return Err("Blender ran over ten minutes and was stopped".into());
        }
        std::thread::sleep(Duration::from_millis(200));
    };
    let _ = t_out.join();
    let err = t_err.join().unwrap_or_default();
    if !status.success() {
        let tail: Vec<&String> = err.iter().rev().take(8).collect::<Vec<_>>().into_iter().rev().collect();
        return Err(format!("Blender failed ({status}):\n{}", tail.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n")));
    }
    Ok(())
}

pub fn to_json(c: &Checks) -> Value {
    json!({
        "passed": c.passed(),
        "same_skeleton": c.same_skeleton, "same_clips": c.same_clips, "same_triangles": c.same_triangles(),
        "unique": c.unique(), "looks_same": c.looks_same(),
        "limits": { "max_mean_diff": relayout::MAX_MEAN_DIFF, "max_over_24_share": 0.01, "max_shared_texels": relayout::MAX_SHARED_AFTER },
        "meshes": c.meshes.iter().map(|m| json!({
            "name": m.name, "image": m.image, "triangles": m.triangles, "texture_px": m.texture_px,
            "shared_texels_at_1024": m.shared_texels, "kept_stock_shared_texels": m.kept_shared, "density_median": m.density_median.map(|d| d.round()),
        })).collect::<Vec<_>>(),
        "views": c.views.iter().map(|v| json!({
            "yaw": v.yaw_deg, "model_px": v.model_px, "mean_diff": (v.mean_diff * 100.0).round() / 100.0, "over_24": v.over_24,
        })).collect::<Vec<_>>(),
    })
}

pub fn print(c: &Checks) {
    let ok = |b: bool| if b { "ok" } else { "FAILED" };
    println!("skeleton {}   clips {}   triangles {}", ok(c.same_skeleton), ok(c.same_clips), ok(c.same_triangles()));
    for m in &c.meshes {
        println!(
            "  {} ({}): {} x {} -> {} x {} px   texels on two surfaces {} -> {}   density median {:.0} -> {:.0}",
            m.name, m.image, m.texture_px[0][0], m.texture_px[0][1], m.texture_px[1][0], m.texture_px[1][1], m.shared_texels[0], m.shared_texels[1], m.density_median[0], m.density_median[1]
        );
        if m.kept_shared > 0 {
            println!("      of those, {} sit in boxes kept at their stock place on purpose; {} others", m.kept_shared, m.unplanned_shared());
        }
    }
    for v in &c.views {
        println!("  view {:>3}°: {} model px, mean difference {:.2} of 255, {} px over 24", v.yaw_deg, v.model_px, v.mean_diff, v.over_24);
    }
    println!("unique layout {}   looks the same {}", ok(c.unique()), ok(c.looks_same()));
    println!("{}", if c.passed() { "PASSED" } else { "FAILED" });
}

/// The GLB a finished re-layout folder holds (named in its report.json).
pub fn output_of(dir: &Path) -> Result<PathBuf, String> {
    let report: Value = serde_json::from_slice(&std::fs::read(dir.join("report.json")).map_err(|e| format!("report.json: {e}"))?).map_err(|e| e.to_string())?;
    Ok(dir.join(report["output"].as_str().ok_or("report.json names no output")?))
}

/// Re-layout plus checks, as one job (the window runs this on a thread).
pub fn run(input: &Path, out: &Path, size: usize, blender: Option<&str>, progress: Box<dyn FnMut(String) + Send>) -> Result<(PathBuf, Checks), String> {
    let blender = find_blender(blender)?;
    let out = check_dir(out)?;
    run_blender(&blender, input, &out, size, progress)?;
    let glb = output_of(&out)?;
    let c = check_files(input, &glb)?;
    std::fs::write(out.join("checks.json"), serde_json::to_string_pretty(&to_json(&c)).unwrap()).map_err(|e| e.to_string())?;
    Ok((glb, c))
}

pub fn check_files(input: &Path, output: &Path) -> Result<Checks, String> {
    check_files_keep(input, output, &[])
}

/// `check_files` for a layout that left `keep` boxes (sheet, 0..1 UV box) at their stock place.
pub fn check_files_keep(input: &Path, output: &Path, keep: &[(String, [f64; 4])]) -> Result<Checks, String> {
    let a = Model::open(input)?;
    let b = Model::open(output)?;
    Ok(relayout::check_keep(&a, &b, &[0.0, 90.0, 180.0, 270.0], 600, keep))
}
