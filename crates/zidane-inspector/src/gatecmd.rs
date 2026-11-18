//! `gaw gate`: runs `gaw defects` on a reference and a candidate and applies the rules of a spec
//! file: per-kind count and load limits against the reference, and "no more boxes of these kinds
//! than the reference had at these marks". Prints PASS or FAIL per rule, writes gate-report.json and
//! exits 1 when any rule fails. Reads models only; writes a new folder.
use crate::Args;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

pub const GATE_USAGE: &str = "  gaw gate --spec GATE.json --out-dir NEW_DIR [CANDIDATE.glb] [--ref-run DIR] [--cand-run DIR]
      Runs `gaw defects` for the reference and the candidate named in the spec (the candidate can also
      come on the command line) and applies its rules. Rule types: `ratio` (count, load or worst score
      of some kinds at most k times the reference's) and `marks` (no more boxes of some kinds inside
      the marked circles than the reference has there; `allow: none` demands zero; `slack: N` widens the
      circles by N view pixels, for marks drawn beside the cut) and `stock-edge` (each box of some
      kinds compared with the reference's picture at the same pixels: it counts as new only when
      the reference shows no edge there, under `min_share` of the cut, or the candidate's Lab step
      is worse by over `margin`; at most `max_new` new boxes). A ratio rule with
      `\"check\": \"deform\"` reads `gaw deform-check`'s summary (kinds stretch, squash, volume,
      intersect, weights; run once into OUT_DIR/deform) instead of the defects run. Prints PASS or
      FAIL per rule, writes gate-report.json with the numbers and the two defects.json runs, and
      exits 1 on any FAIL. A candidate that is the reference passes unless a rule says `allow: none`.
      A spec can also set `\"baseline\": \"self\"` (smear-blur, flat-patch and lock-scale are measured
      against each model's own medians) and `\"cut_regions\": \"skip\"` (layer cuts skip boundaries
      between UV islands of different material), both passed on to `gaw defects`; for a new painted
      model judged against stock's noisier textures. Hair specs leave both off.
      --spec GATE.json         rules: {part, reference, candidate?, marks | marks_file, rules:[..]}; see
                               skills/gaw/references/zidane-hair-gate.json
      --out-dir NEW_DIR        new or empty; gets candidate/ and reference/ (defects runs) and gate-report.json
      --ref-run DIR            reuse an earlier reference run (its defects.json) instead of running it again
      --cand-run DIR           rescore an earlier candidate run (its defects.json and views/) without running it";

fn opt<'a>(a: &'a [String], k: &str) -> Option<&'a str> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).map(String::as_str)
}

fn run_defects(cand: &Path, part: &str, reference: &Path, out: &Path, extra: &[String]) -> Result<Value, String> {
    let exe = std::env::current_exe().map_err(|e| format!("gate: cannot find gaw itself: {e}"))?;
    let mut cmd = Command::new(exe);
    cmd.arg("defects").arg(cand).arg("--part").arg(part).arg("--ref").arg(reference).arg("--out-dir").arg(out).args(extra);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0000_4000); // BELOW_NORMAL_PRIORITY_CLASS
    }
    let o = cmd.output().map_err(|e| format!("gate: cannot run defects: {e}"))?;
    if !o.status.success() {
        return Err(format!("gate: defects on {} failed: {}", cand.display(), String::from_utf8_lossy(&o.stderr).trim()));
    }
    read_json(&out.join("defects.json"))
}

/// `gaw deform-check CAND --ref REF --out-dir D`, then `D/deform.json`. deform-check exits 1 when a
/// measure fails; the gate still reads its numbers, so only a missing deform.json is an error.
fn run_deform(cand: &Path, reference: &Path, out: &Path) -> Result<Value, String> {
    let exe = std::env::current_exe().map_err(|e| format!("gate: cannot find gaw itself: {e}"))?;
    let mut cmd = Command::new(exe);
    cmd.arg("deform-check").arg(cand).arg("--ref").arg(reference).arg("--out-dir").arg(out);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0000_4000); // BELOW_NORMAL_PRIORITY_CLASS
    }
    let o = cmd.output().map_err(|e| format!("gate: cannot run deform-check: {e}"))?;
    if !out.join("deform.json").exists() {
        return Err(format!("gate: deform-check on {} wrote no deform.json: {}", cand.display(), String::from_utf8_lossy(&o.stderr).trim()));
    }
    read_json(&out.join("deform.json"))
}

fn read_json(p: &Path) -> Result<Value, String> {
    let t = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_str(&t).map_err(|e| format!("{}: {e}", p.display()))
}

struct Mark {
    id: i64,
    name: String,
    view: String,
    x: f64,
    y: f64,
    r: f64,
}

fn marks_of(spec: &Value, spec_dir: &Path) -> Result<Vec<Mark>, String> {
    let list: Vec<Value> = if let Some(a) = spec.get("marks").and_then(Value::as_array) {
        a.clone()
    } else if let Some(f) = spec.get("marks_file").and_then(Value::as_str) {
        let p = if Path::new(f).is_absolute() { PathBuf::from(f) } else { spec_dir.join(f) };
        let v = read_json(&p)?;
        v.get("marks").and_then(Value::as_array).cloned().or_else(|| v.as_array().cloned()).ok_or_else(|| format!("{}: no marks list", p.display()))?
    } else {
        Vec::new()
    };
    list.iter()
        .map(|m| {
            let f = |k: &str| m.get(k).and_then(Value::as_f64).ok_or_else(|| format!("a mark needs a number `{k}`"));
            Ok(Mark { id: f("id")? as i64, name: m.get("name").and_then(Value::as_str).unwrap_or("").to_owned(), view: m.get("view").and_then(Value::as_str).ok_or("a mark needs `view`")?.to_owned(), x: f("x")?, y: f("y")?, r: f("r")? })
        })
        .collect()
}

fn boxes_of(d: &Value) -> Vec<(String, [f64; 4])> {
    let mut out = Vec::new();
    let arr = |v: &Value| -> Option<[f64; 4]> {
        let a = v.as_array()?;
        Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?, a.get(2)?.as_f64()?, a.get(3)?.as_f64()?])
    };
    if let (Some(v), Some(b)) = (d.get("view").and_then(Value::as_str), d.get("screen_box").and_then(arr)) {
        out.push((v.to_owned(), b));
    }
    for s in d.get("seen_boxes").and_then(Value::as_array).into_iter().flatten() {
        if let (Some(v), Some(b)) = (s.get("view").and_then(Value::as_str), s.get("screen_box").and_then(arr)) {
            out.push((v.to_owned(), b));
        }
    }
    out
}

/// Defects of the given kinds that have a screen box in the mark's view touching its circle.
/// `slack` (view pixels) widens the circle: the user's circles are hand-drawn and can sit beside the cut.
fn at_mark(run: &Value, kinds: &[String], m: &Mark, slack: f64) -> Vec<String> {
    let reach = m.r + slack;
    let mut hits: Vec<String> = Vec::new();
    for d in run.get("defects").and_then(Value::as_array).into_iter().flatten() {
        let k = d.get("kind").and_then(Value::as_str).unwrap_or("");
        if !kinds.iter().any(|x| x == k) {
            continue;
        }
        let touch = boxes_of(d).iter().any(|(v, b)| {
            if v != &m.view {
                return false;
            }
            let (nx, ny) = (m.x.clamp(b[0], b[2]), m.y.clamp(b[1], b[3]));
            (nx - m.x).powi(2) + (ny - m.y).powi(2) <= reach * reach
        });
        if touch {
            hits.push(k.to_owned());
        }
    }
    hits
}

fn summary_of(run: &Value, kinds: &[String], measure: &str, reference: bool) -> f64 {
    let mut total = 0.0;
    for s in run.get("summary").and_then(Value::as_array).into_iter().flatten() {
        let k = s.get("kind").and_then(Value::as_str).unwrap_or("");
        if !kinds.iter().any(|x| x == k) {
            continue;
        }
        let src = if reference { s.get("reference").unwrap_or(&Value::Null) } else { s };
        let v = match measure {
            "load" => src.get("load_percent").and_then(Value::as_f64),
            "count" => src.get("defects").and_then(Value::as_f64),
            _ => src.get("worst_score").and_then(Value::as_f64),
        };
        total += v.unwrap_or(0.0);
    }
    total
}

pub fn gate_cmd(args: &Args) -> Result<(), String> {
    let a = &args.0;
    let spec_path = PathBuf::from(opt(a, "--spec").ok_or("gate needs --spec GATE.json")?);
    let out = PathBuf::from(opt(a, "--out-dir").ok_or("gate needs --out-dir NEW_DIR")?);
    if out.exists() && std::fs::read_dir(&out).map(|mut d| d.next().is_some()).unwrap_or(true) {
        return Err(format!("{} already has files; --out-dir must be new or empty", out.display()));
    }
    let spec = read_json(&spec_path)?;
    let spec_dir = spec_path.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut skip = false;
    let mut pos: Vec<&String> = Vec::new();
    for x in a {
        if skip {
            skip = false;
        } else if ["--spec", "--out-dir", "--ref-run"].contains(&x.as_str()) {
            skip = true;
        } else if !x.starts_with("--") {
            pos.push(x);
        }
    }
    let cand = pos.first().map(|s| s.to_string()).or_else(|| spec.get("candidate").and_then(Value::as_str).map(String::from)).ok_or("gate needs a candidate GLB (on the command line or in the spec)")?;
    let reference = spec.get("reference").and_then(Value::as_str).ok_or("the spec needs `reference`")?.to_owned();
    let part = spec.get("part").and_then(Value::as_str).unwrap_or("hair").to_owned();
    let marks = marks_of(&spec, &spec_dir)?;
    let rules = spec.get("rules").and_then(Value::as_array).ok_or("the spec needs a `rules` list")?;
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;

    println!("gate {}: candidate {cand}, reference {reference}, part {part}", spec.get("name").and_then(Value::as_str).unwrap_or(""));
    // A spec with `"defects": false` runs no defects at all (deform rules only).
    let want_defects = spec.get("defects").and_then(Value::as_bool).unwrap_or(true);
    // Spec fields that choose how smear-blur, flat-patch and layer cuts are measured, passed on to
    // `gaw defects`: `"baseline": "self"` and `"cut_regions": "skip"` (or a Lab number).
    let mut extra: Vec<String> = Vec::new();
    if let Some(b) = spec.get("baseline").and_then(Value::as_str) {
        extra.extend(["--baseline".to_owned(), b.to_owned()]);
    }
    match spec.get("cut_regions") {
        Some(Value::String(s)) => extra.extend(["--cut-regions".to_owned(), s.clone()]),
        Some(Value::Number(n)) => extra.extend(["--cut-regions".to_owned(), n.to_string()]),
        _ => {}
    }
    // `--cand-run DIR` rescores an earlier candidate run (its defects.json and views) without running it again.
    let cand_run_dir = opt(a, "--cand-run").map(|d| Path::new(d).to_path_buf());
    let crun = if !want_defects {
        Value::Null
    } else if let Some(d) = &cand_run_dir {
        read_json(&d.join("defects.json"))?
    } else {
        run_defects(Path::new(&cand), &part, Path::new(&reference), &out.join("candidate"), &extra)?
    };
    let rrun = if !want_defects {
        Value::Null
    } else if let Some(d) = opt(a, "--ref-run") {
        read_json(&Path::new(d).join("defects.json"))?
    } else if Path::new(&cand) == Path::new(&reference) {
        crun.clone()
    } else {
        run_defects(Path::new(&reference), &part, Path::new(&reference), &out.join("reference"), &extra)?
    };

    // `"check": "deform"` on a ratio rule reads gaw deform-check's summary[] (kinds stretch, squash,
    // volume, intersect, weights) instead of defects.json's; deform-check runs once, on first use.
    let uses_deform = rules.iter().any(|r| r.get("check").and_then(Value::as_str) == Some("deform"));
    let drun: Option<Value> = if uses_deform {
        println!("deform-check {cand} against {reference}");
        Some(run_deform(Path::new(&cand), Path::new(&reference), &out.join("deform"))?)
    } else {
        None
    };
    let mut results: Vec<Value> = Vec::new();
    let mut all_pass = true;
    for r in rules {
        let is_deform = r.get("check").and_then(Value::as_str) == Some("deform");
        let crun_rule: &Value = if is_deform { drun.as_ref().unwrap() } else { &crun };
        let id = r.get("id").and_then(Value::as_str).unwrap_or("rule").to_owned();
        let kinds: Vec<String> = r.get("kinds").and_then(Value::as_array).map(|k| k.iter().filter_map(Value::as_str).map(String::from).collect()).unwrap_or_default();
        let ty = r.get("type").and_then(Value::as_str).unwrap_or(if r.get("marks").is_some() { "marks" } else { "ratio" });
        let mut lines: Vec<String> = Vec::new();
        let mut pass = true;
        let mut detail = json!({});
        match ty {
            "ratio" => {
                let measure = r.get("measure").and_then(Value::as_str).unwrap_or("load");
                let max_x = r.get("max_x").and_then(Value::as_f64).unwrap_or(1.0);
                let floor = r.get("floor").and_then(Value::as_f64).unwrap_or(0.0);
                let (c, rf) = (summary_of(crun_rule, &kinds, measure, false), summary_of(crun_rule, &kinds, measure, true));
                let limit = max_x * rf.max(floor);
                pass = c <= limit + 1e-9;
                let unit = if measure == "load" { "%" } else { "" };
                lines.push(format!("{} {measure} {:.1}{unit} against the reference's {:.1}{unit}; limit {:.1}x = {:.1}{unit}", kinds.join("+"), c, rf, max_x, limit));
                detail = json!({"measure": measure, "candidate": c, "reference": rf, "max_x": max_x, "floor": floor, "limit": limit});
            }
            "marks" => {
                if is_deform {
                    return Err(format!("rule {id}: `check: deform` works with ratio rules only"));
                }
                let ids: Vec<i64> = r.get("marks").and_then(Value::as_array).map(|k| k.iter().filter_map(Value::as_i64).collect()).unwrap_or_default();
                let none = r.get("allow").and_then(Value::as_str) == Some("none");
                let slack = r.get("slack").and_then(Value::as_f64).unwrap_or(0.0).max(0.0);
                let mut per = Vec::new();
                for id_m in ids {
                    let Some(m) = marks.iter().find(|m| m.id == id_m) else {
                        pass = false;
                        lines.push(format!("mark {id_m}: not in the marks list"));
                        continue;
                    };
                    let (ch, rh) = (at_mark(&crun, &kinds, m, slack), at_mark(&rrun, &kinds, m, slack));
                    let allowed = if none { 0 } else { rh.len() };
                    let ok = ch.len() <= allowed;
                    pass &= ok;
                    let tally = |h: &Vec<String>| -> String {
                        let mut k: Vec<(String, usize)> = Vec::new();
                        for x in h {
                            match k.iter_mut().find(|(n, _)| n == x) {
                                Some(e) => e.1 += 1,
                                None => k.push((x.clone(), 1)),
                            }
                        }
                        if k.is_empty() { "none".into() } else { k.iter().map(|(n, c)| format!("{c} {n}")).collect::<Vec<_>>().join(", ") }
                    };
                    lines.push(format!("{} mark {} {}: candidate {}; reference {}", if ok { "ok  " } else { "HIT " }, m.id, m.name, tally(&ch), tally(&rh)));
                    per.push(json!({"mark": m.id, "name": m.name, "view": m.view, "candidate": ch.len(), "reference": rh.len(), "ok": ok}));
                }
                detail = json!({"allow": if none { "none" } else { "reference" }, "slack": slack, "marks": per});
            }
            "stock-edge" => {
                if is_deform || !want_defects {
                    return Err(format!("rule {id}: `stock-edge` needs the defects runs"));
                }
                let margin = r.get("margin").and_then(Value::as_f64).unwrap_or(12.0) as f32;
                let min_share = r.get("min_share").and_then(Value::as_f64).unwrap_or(0.5) as f32;
                let max_new = r.get("max_new").and_then(Value::as_u64).unwrap_or(0) as usize;
                let cdir = cand_run_dir.clone().unwrap_or_else(|| out.join("candidate"));
                let rdir = match opt(a, "--ref-run") {
                    Some(d) => Path::new(d).to_path_buf(),
                    None if Path::new(&cand) == Path::new(&reference) => cdir.clone(),
                    None => out.join("reference"),
                };
                let (new, rows) = crate::cutpair::score_run(&cdir, &rdir, &crun, &kinds, margin, min_share)?;
                pass = new <= max_new;
                let all = rows.len();
                lines.push(format!("{} boxes: {new} new against stock (step worse than stock's by over {margin} Lab, or stock edge on under {:.0}% of the cut), {} stock shows too; limit {max_new}", kinds.join("+"), min_share * 100.0, all - new));
                for row in rows.iter().filter(|r| r.get("new").and_then(Value::as_bool) == Some(true)) {
                    lines.push(format!("NEW  {} {}: step {:.1} against stock's {:.1}, stock edge on {:.0}%, another object in stock's picture on {:.0}%", row["id"].as_str().unwrap_or(""), row["view"].as_str().unwrap_or(""), row["candidate_step"].as_f64().unwrap_or(0.0), row["stock_step"].as_f64().unwrap_or(0.0), row["stock_edge_share"].as_f64().unwrap_or(0.0) * 100.0, row["other_object_share"].as_f64().unwrap_or(0.0) * 100.0));
                }
                detail = json!({"margin": margin, "min_share": min_share, "max_new": max_new, "new": new, "boxes": all, "rows": rows});
            }
            other => return Err(format!("rule {id}: unknown type {other} (use ratio, marks or stock-edge)")),
        }
        all_pass &= pass;
        println!("{} {id}", if pass { "PASS" } else { "FAIL" });
        for l in &lines {
            println!("       {l}");
        }
        results.push(json!({"id": id, "type": ty, "kinds": kinds, "pass": pass, "lines": lines, "detail": detail, "why": r.get("why")}));
    }
    println!("{}", if all_pass { "GATE PASS" } else { "GATE FAIL" });
    let report = json!({"gaw_gate": 1, "name": spec.get("name"), "pass": all_pass, "candidate": cand, "reference": reference, "part": part, "spec": spec_path.display().to_string(), "rules": results, "runs": {"candidate": cand_run_dir.clone().unwrap_or_else(|| out.join("candidate")).display().to_string(), "reference": if Path::new(&cand) == Path::new(&reference) { Value::Null } else { json!(out.join("reference").display().to_string()) }}});
    std::fs::write(out.join("gate-report.json"), serde_json::to_vec_pretty(&report).unwrap()).map_err(|e| format!("gate-report.json: {e}"))?;
    println!("wrote {}", out.join("gate-report.json").display());
    if !all_pass {
        std::process::exit(1);
    }
    Ok(())
}
