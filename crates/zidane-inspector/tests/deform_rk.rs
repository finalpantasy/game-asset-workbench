//! `gaw deform` end to end: runs the built program on the stock export (offline, no window).

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

fn stock() -> Option<PathBuf> {
    let p = PathBuf::from(r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding\exports\Zidane-idle.glb");
    p.exists().then_some(p)
}

fn gaw(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_gaw")).args(args).output().expect("run gaw")
}

#[test]
fn stock_against_itself_has_no_difference() {
    let Some(s) = stock() else { return };
    let s = s.to_string_lossy().to_string();
    let out = gaw(&["deform", &s, &s, "--json", "--top", "2"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let v: Value = serde_json::from_slice(&out.stdout).expect("json");
    let clips = v["clips"].as_array().unwrap();
    assert_eq!(clips.len(), 5, "idle, walk, run, turn_l, turn_r");
    for c in clips {
        assert_eq!(c["verdict"], "same", "{}", c["clip"]);
        assert_eq!(c["delta"]["worst_edge"], 0.0);
        assert_eq!(c["delta"]["worst_area"], 0.0);
        assert_eq!(c["delta"]["flipped"], 0);
        assert_eq!(c["delta"]["hard_flipped"], 0);
        assert!(c["stock"]["worst_edge"].as_f64().unwrap() >= 1.0);
    }
    assert_eq!(v["worse_clips"], 0);
    // stock's own flaws are what the comparison is measured against: run flips triangles in stock
    let run = clips.iter().find(|c| c["clip"] == "run").unwrap();
    assert!(run["stock"]["flipped"].as_u64().unwrap() > 0);
    assert_eq!(run["stock"]["worst"][0]["mesh"], run["candidate"]["worst"][0]["mesh"]);
}

#[test]
fn table_lists_the_asked_clip_and_help_lists_the_command() {
    let Some(s) = stock() else { return };
    let s = s.to_string_lossy().to_string();
    let out = gaw(&["deform", &s, &s, "--clips", "walk", "--steps", "8"]);
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success());
    assert!(text.contains("walk") && !text.contains("turn_l"), "{text}");
    assert!(text.contains("0 of 1 clips WORSE"), "{text}");
    let help = gaw(&["help"]);
    let h = String::from_utf8_lossy(&help.stdout).to_string() + &String::from_utf8_lossy(&help.stderr);
    assert!(h.contains("gaw deform STOCK.glb CAND.glb"), "help is missing gaw deform");
}

#[test]
fn bad_input_is_refused_and_render_dir_is_never_reused() {
    let Some(s) = stock() else { return };
    let s = s.to_string_lossy().to_string();
    let out = gaw(&["deform", &s]);
    assert!(!out.status.success());
    let out = gaw(&["deform", &s, &s, "--clips", "no_such_clip"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("no_such_clip"));
    // a folder that already has files is refused, and nothing in it changes
    let dir = std::env::temp_dir().join(format!("gaw-deform-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let keep = dir.join("keep.txt");
    std::fs::write(&keep, "mine").unwrap();
    let out = gaw(&["deform", &s, &s, "--clips", "idle", "--steps", "4", "--render-dir", dir.to_str().unwrap()]);
    assert!(!out.status.success());
    assert_eq!(std::fs::read_to_string(&keep).unwrap(), "mine");
    let fresh = dir.join("new");
    let out = gaw(&["deform", &s, &s, "--clips", "idle", "--steps", "4", "--render-dir", fresh.to_str().unwrap()]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(Path::new(&fresh.join("deform-stock.png")).exists() && fresh.join("deform-cand.png").exists());
    std::fs::remove_dir_all(&dir).ok();
}
