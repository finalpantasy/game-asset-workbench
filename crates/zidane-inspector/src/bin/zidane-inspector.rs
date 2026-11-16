//! Old name kept as an alias: runs `gaw` from the same folder with the same arguments and exit code.
use std::process::{exit, Command};

fn main() {
    let me = std::env::current_exe().expect("current exe");
    let gaw = me.with_file_name(if cfg!(windows) { "gaw.exe" } else { "gaw" });
    match Command::new(&gaw).args(std::env::args_os().skip(1)).status() {
        Ok(s) => exit(s.code().unwrap_or(1)),
        Err(e) => {
            eprintln!("zidane-inspector is now gaw; could not run {}: {e}", gaw.display());
            exit(1);
        }
    }
}
