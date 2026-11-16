//! Every command in `gaw help` must appear in the gaw skill (`skills/gaw`: SKILL.md and references).
//!
//! Changing a command's behavior, flags or output means updating the matching skill reference in
//! the same change. A new command needs a router line in SKILL.md and a reference entry; this test
//! enforces the minimum. It runs the built program's `help` (offline, no window) and reads the
//! skill files.

use std::path::{Path, PathBuf};
use std::process::Command;

fn skill_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../skills/gaw")
}

fn skill_text() -> String {
    let dir = skill_dir();
    let mut text = std::fs::read_to_string(dir.join("SKILL.md"))
        .unwrap_or_else(|e| panic!("skills/gaw/SKILL.md missing ({e}); the gaw skill must ship with the tool"));
    let refs = dir.join("references");
    let mut files: Vec<_> = std::fs::read_dir(&refs)
        .unwrap_or_else(|e| panic!("skills/gaw/references missing ({e})"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    files.sort();
    for f in files {
        text.push('\n');
        text.push_str(&std::fs::read_to_string(f).unwrap());
    }
    text
}

/// First word after `gaw` on every usage line of `gaw help` (two-space indent).
fn help_commands() -> Vec<String> {
    let out = Command::new(env!("CARGO_BIN_EXE_gaw")).arg("help").output().expect("run gaw help");
    let help = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    let mut cmds: Vec<String> = Vec::new();
    for line in help.lines() {
        let Some(rest) = line.strip_prefix("  gaw ") else { continue };
        let word = rest.split_whitespace().next().unwrap_or("");
        let word = word.trim_matches(|c| c == '[' || c == ']');
        if word.is_empty() || word.starts_with('-') || word.starts_with("file") {
            continue;
        }
        if !cmds.iter().any(|c| c == word) {
            cmds.push(word.to_owned());
        }
    }
    cmds
}

#[test]
fn every_command_is_in_the_skill() {
    let cmds = help_commands();
    assert!(cmds.len() >= 10, "found only {} commands in `gaw help`: {cmds:?}", cmds.len());
    let text = skill_text();
    let missing: Vec<&String> = cmds
        .iter()
        .filter(|c| {
            let ticked = format!("`{c}");
            let spoken = format!("gaw {c}");
            !(text.contains(&ticked) || text.contains(&spoken))
        })
        .collect();
    assert!(
        missing.is_empty(),
        "commands in `gaw help` that skills/gaw does not mention: {missing:?}. Add a router line to SKILL.md and an entry in the matching references/*.md (see skills/gaw/MAINTAINING.md)."
    );
}
