//! Mark & fix, the "Look" tools: isolate a part (and its front, back or side walls), lock places
//! an edit or an AI fix must not move, overlays that find what sits behind a seam or a break
//! (wireframe, depth, open edges and cracks, intersections, a probe that lists every surface under
//! the pointer), and saved inspection views that double as acceptance views. Nothing here knows
//! about one game: it works on welded triangles, groups and cameras.
use super::*;
use crate::viewscmd::{load_views, store_views, views_file, SavedView};
use wb_core::edit::Wall;

pub struct LockReport {
    pub file: String,
    pub checked: usize,
    /// (group, distance moved), worst first.
    pub bad: Vec<(u32, f32)>,
}

#[derive(Default)]
pub struct Look {
    pub ver: u64,
    /// Per global triangle: hidden from the model view (and from picking).
    pub hidden: Vec<bool>,
    /// Per weld group: locked (shape brushes skip it; checks report it if it moves).
    pub locked: Vec<bool>,
    pub lock_report: Option<LockReport>,
    pub wire: bool,
    pub depth: bool,
    pub open: bool,
    pub cross: bool,
    open_edges: Option<Vec<(u32, u32)>>,
    cracks: Option<(u64, Vec<((u32, u32), (u32, u32))>)>,
    inter: Option<(u64, Vec<(u32, u32)>, Vec<bool>)>,
    /// Last probe: every triangle under the pointer, nearest first, and where it was (buffer px).
    pub probe: Vec<(u32, f32)>,
    probe_px: Option<[f32; 2]>,
    pub views: Vec<SavedView>,
    views_loaded: bool,
    view_name: String,
}

const LOCK_BLUE: [u8; 3] = [70, 140, 255];

impl Mark {
    /// Size the per-triangle and per-group lists once the welded topology exists.
    pub(super) fn ensure_look(&mut self, topo: &Topo) {
        if self.look.hidden.len() != topo.tris() {
            self.look.hidden = vec![false; topo.tris()];
            self.look.probe.clear();
            self.look.ver += 1;
        }
        if self.look.locked.len() != topo.groups() {
            self.look.locked = vec![false; topo.groups()];
            self.look.lock_report = None;
            self.look.open_edges = None;
            self.look.ver += 1;
        }
        if !self.look.views_loaded {
            self.look.views = load_views(&views_file());
            self.look.views_loaded = true;
        }
    }

    pub(super) fn any_hidden(&self) -> bool {
        self.look.hidden.iter().any(|&h| h)
    }

    /// Shape brushes skip locked places and places only hidden triangles use.
    pub(super) fn look_allow(&self, topo: &Topo, allow: &mut [f32]) {
        let hidden = self.any_hidden();
        for (g, a) in allow.iter_mut().enumerate() {
            if self.look.locked.get(g).copied().unwrap_or(false) {
                *a = 0.0;
            } else if hidden && topo.group_tris[g].iter().all(|&t| self.look.hidden[t as usize]) {
                *a = 0.0;
            }
        }
    }

    fn pose_stamp(&self) -> u64 {
        let (v, clip, t) = self.pose_key.clone().unwrap_or((0, None, 0));
        let h = clip.map(|c| c.bytes().fold(7u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64))).unwrap_or(0);
        v.wrapping_mul(1_000_003).wrapping_add(h).wrapping_add(t as u64 * 17)
    }

    /// Per-triangle overlay colors for the model view (depth, intersections, locked areas).
    pub(super) fn look_tints(&mut self, topo: &Topo, cam: &Camera, w: usize, h: usize) -> Option<Vec<Option<[u8; 3]>>> {
        let locked_any = self.look.locked.iter().any(|&l| l);
        if !self.look.depth && !self.look.cross && !locked_any {
            return None;
        }
        let n = topo.tris();
        let mut out: Vec<Option<[u8; 3]>> = vec![None; n];
        if locked_any {
            for t in 0..n {
                if topo.tri_groups[t].iter().all(|&g| self.look.locked[g as usize]) {
                    out[t] = Some(LOCK_BLUE);
                }
            }
        }
        if self.look.depth {
            let rig = Rig::new(cam, w, h);
            let z: Vec<f32> = (0..n)
                .map(|t| {
                    let [a, b, c] = topo.tri_groups[t].map(|g| self.gpose[g as usize]);
                    rig.project(scale(add(add(a, b), c), 1.0 / 3.0))[2]
                })
                .collect();
            let vis = |t: usize| !self.look.hidden[t] && z[t] > 0.0;
            let lo = (0..n).filter(|&t| vis(t)).map(|t| z[t]).fold(f32::MAX, f32::min);
            let hi = (0..n).filter(|&t| vis(t)).map(|t| z[t]).fold(f32::MIN, f32::max);
            for t in 0..n {
                let f = ((z[t] - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);
                // Near: warm yellow. Far: deep blue.
                let near = [255.0, 214.0, 70.0];
                let far = [40.0, 70.0, 220.0];
                out[t] = Some(std::array::from_fn(|k| (near[k] * (1.0 - f) + far[k] * f) as u8));
            }
        }
        if self.look.cross {
            let stamp = self.pose_stamp();
            if self.look.inter.as_ref().map(|c| c.0) != Some(stamp) {
                let pairs = edit::intersections(topo, &self.gpose);
                let mut hit = vec![false; n];
                for &(a, b) in &pairs {
                    hit[a as usize] = true;
                    hit[b as usize] = true;
                }
                self.look.inter = Some((stamp, pairs, hit));
            }
            if let Some((_, _, hit)) = &self.look.inter {
                for t in 0..n {
                    if hit[t] {
                        out[t] = Some([255, 40, 60]);
                    }
                }
            }
        }
        Some(out)
    }

    pub(super) fn look_key(&self) -> u64 {
        self.look.ver.wrapping_mul(2654435761) ^ (self.look.depth as u64) << 1 ^ (self.look.cross as u64) << 2
    }

    // ---------------------------------------------------------------------------------------------
    // Actions
    // ---------------------------------------------------------------------------------------------

    pub(super) fn isolate(&mut self, op: &str) {
        let marked: Vec<bool> = self.sel.iter().map(|&w| w > 0.0).collect();
        match op {
            "only" => {
                if !marked.iter().any(|&m| m) {
                    self.say("Mark something first, then show only it".into(), false);
                    return;
                }
                self.look.hidden = marked.iter().map(|&m| !m).collect();
            }
            "hide" => {
                for (h, &m) in self.look.hidden.iter_mut().zip(&marked) {
                    *h |= m;
                }
            }
            "nearest" => {
                let Some(&(t, _)) = self.look.probe.iter().find(|(t, _)| !self.look.hidden[*t as usize]) else {
                    self.say("Probe a spot first (the Look tool), then hide the nearest surface".into(), false);
                    return;
                };
                self.look.hidden[t as usize] = true;
            }
            _ => self.look.hidden.iter_mut().for_each(|h| *h = false),
        }
        self.look.ver += 1;
        let shown = self.look.hidden.iter().filter(|&&h| !h).count();
        self.say(format!("Showing {} of {} triangles", theme::num(shown), theme::num(self.look.hidden.len())), true);
    }

    /// Keep only the front, back or side walls of the marked triangles. Front faces the camera.
    pub(super) fn keep_walls(&mut self, which: Wall, cam: &Camera) {
        let Some(topo) = self.topo.clone() else { return };
        let tris: Vec<u32> = (0..self.sel.len()).filter(|&t| self.sel[t] > 0.0).map(|t| t as u32).collect();
        if tris.is_empty() {
            self.say("Mark a part first (Part tool), then pick its walls".into(), false);
            return;
        }
        let rig = Rig::new(cam, 64, 64);
        let (_, walls) = edit::walls(&topo, &self.gpose, &tris, scale(rig.f, -1.0));
        let count = |w: Wall| walls.iter().filter(|&&x| x == w).count();
        let (f, b, s) = (count(Wall::Front), count(Wall::Back), count(Wall::Side));
        let before = self.sel_state();
        for (&t, &w) in tris.iter().zip(&walls) {
            if w != which {
                self.sel[t as usize] = 0.0;
            }
        }
        self.push_undo(Snap { what: format!("{which:?} walls").to_lowercase(), sel: Some(before), gpos: None, tex: None });
        self.sel_ver += 1;
        self.say(format!("Kept the {} walls: {} triangles (front {f}, back {b}, sides {s})", format!("{which:?}").to_lowercase(), self.marked_count()), true);
    }

    pub(super) fn lock_op(&mut self, op: &str) {
        let Some(topo) = self.topo.clone() else { return };
        match op {
            "lock" | "unlock" => {
                let on = op == "lock";
                let mut n = 0;
                for (t, &w) in self.sel.iter().enumerate() {
                    if w > 0.0 {
                        for g in topo.tri_groups[t] {
                            self.look.locked[g as usize] = on;
                            n += 1;
                        }
                    }
                }
                if n == 0 {
                    self.say("Mark the places to lock first".into(), false);
                    return;
                }
            }
            _ => self.look.locked.iter_mut().for_each(|l| *l = false),
        }
        self.look.lock_report = None;
        self.look.ver += 1;
        let c = self.look.locked.iter().filter(|&&l| l).count();
        self.say(format!("{} places locked: shape brushes leave them alone, and checks report them if they move", theme::num(c)), true);
    }

    /// Compare the locked places with another file (an AI fix, an export): which moved?
    pub(super) fn check_locks(&mut self, path: &std::path::Path) {
        let Some(topo) = self.topo.clone() else { return };
        let other = match wb_core::Model::open(path) {
            Ok(m) => m,
            Err(e) => {
                self.say(format!("Could not open {}: {e}", path.display()), false);
                return;
            }
        };
        let Some(after) = edit::group_positions_of(&topo, &other) else {
            self.say(format!("{} has different meshes or vertex counts; it cannot be compared place by place", path.display()), false);
            return;
        };
        let tol = topo.height * 1e-4;
        let bad = edit::lock_violations(&self.look.locked, &topo.rest, &after, tol);
        let checked = self.look.locked.iter().filter(|&&l| l).count();
        let file = path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
        self.say(if bad.is_empty() { format!("{file}: all {checked} locked places stayed put") } else { format!("{file}: {} of {checked} locked places moved", bad.len()) }, bad.is_empty());
        if self.scripted {
            println!("locks {}: {} checked, {} moved{}", path.display(), checked, bad.len(), bad.first().map(|b| format!(", worst {:.5}", b.1)).unwrap_or_default());
        }
        self.look.lock_report = Some(LockReport { file, checked, bad });
        self.look.ver += 1;
    }

    /// The edit itself: locked places that moved (brushes skip them, but a Grab started before a
    /// lock, or an undo across one, can still move them).
    pub(super) fn own_lock_line(&self, topo: &Topo) -> Option<(bool, String)> {
        let n = self.look.locked.iter().filter(|&&l| l).count();
        if n == 0 {
            return None;
        }
        let bad = edit::lock_violations(&self.look.locked, &topo.rest, &self.gpos, topo.height * 1e-4);
        Some((bad.is_empty(), if bad.is_empty() { format!("Locked places unchanged ({n})") } else { format!("{} of {n} locked places moved", bad.len()) }))
    }

    pub(super) fn probe_at(&mut self, b: [f32; 2], cam: &Camera, topo: &Topo) {
        let rig = Rig::new(cam, self.vb.w, self.vb.h);
        let o = unproject(&rig, b[0], b[1], 0.0);
        let dir = sub(unproject(&rig, b[0], b[1], 1.0), o);
        self.look.probe = edit::ray_hits(topo, &self.gpose, o, dir);
        self.look.probe_px = Some(b);
        self.look.ver += 1;
        let n = self.look.probe.len();
        self.say(if n == 0 { "Nothing under that spot".into() } else { format!("{n} surfaces under that spot, nearest first (see Look)") }, n > 0);
    }

    fn part_label(&self, d: &Doc, topo: &Topo, t: u32) -> String {
        let (mi, ti) = topo.tri_loc[t as usize];
        let p = d.tri_parts.get(mi).and_then(|v| v.get(ti)).copied().unwrap_or(0);
        let part = if p == 0 { "no part".to_owned() } else { self.parts.as_ref().map(|x| super::super::title_case(x.name(p))).unwrap_or_else(|| format!("part {p}")) };
        format!("{part} (mesh {mi}, triangle {ti})")
    }

    pub(super) fn goto_view(&mut self, v: &SavedView, cam: &mut Camera) {
        *cam = v.cam;
        self.clip = v.clip.clone();
        self.time = v.time;
        self.say(format!("View: {}", v.name), true);
    }

    pub(super) fn save_view(&mut self, name: &str, cam: &Camera) {
        let name = name.trim();
        if name.is_empty() {
            self.say("Give the view a name first".into(), false);
            return;
        }
        let v = SavedView { name: name.to_owned(), cam: *cam, clip: self.posed_clip(), time: self.time };
        match self.look.views.iter_mut().find(|x| x.name == name) {
            Some(x) => *x = v,
            None => self.look.views.push(v),
        }
        match store_views(&views_file(), &self.look.views) {
            Ok(()) => self.say(format!("Saved view \"{name}\" ({} views)", self.look.views.len()), true),
            Err(e) => self.say(format!("Could not save the views: {e}"), false),
        }
    }

    fn delete_view(&mut self, i: usize) {
        if i < self.look.views.len() {
            let v = self.look.views.remove(i);
            let _ = store_views(&views_file(), &self.look.views);
            self.say(format!("Deleted view \"{}\"", v.name), true);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Panel
    // ---------------------------------------------------------------------------------------------

    pub(super) fn look_card(&mut self, ui: &mut egui::Ui, d: &Doc, cam: &mut Camera) {
        let Some(topo) = self.topo.clone() else { return };
        theme::card(ui, "Look", |ui| {
            // Isolate.
            let marked = self.marked_count() > 0;
            let hidden = self.look.hidden.iter().filter(|&&h| h).count();
            ui.label(RichText::new(if hidden == 0 { "Showing everything".to_owned() } else { format!("{} triangles hidden", theme::num(hidden)) }).size(12.5).color(theme::WEAK));
            ui.horizontal_wrapped(|ui| {
                if ui.add_enabled(marked, egui::Button::new("Show only marked")).on_hover_text("Hide everything that is not marked, to see a part on its own").clicked() {
                    self.isolate("only");
                }
                if ui.add_enabled(marked, egui::Button::new("Hide marked")).on_hover_text("Hide the marked triangles to see what is behind them").clicked() {
                    self.isolate("hide");
                }
                if ui.add_enabled(hidden > 0, egui::Button::new("Show all")).clicked() {
                    self.isolate("all");
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Keep walls:").size(12.5).color(theme::WEAK));
                for (w, name, tip) in [(Wall::Front, "Front", "Keep the marked triangles that face you"), (Wall::Back, "Back", "Keep the marked triangles that face away"), (Wall::Side, "Sides", "Keep the side walls (edges and thickness) of the marked part")] {
                    if ui.add_enabled(marked, egui::Button::new(name)).on_hover_text(tip).clicked() {
                        self.keep_walls(w, cam);
                    }
                }
            });
            ui.add_space(4.0);
            // Overlays.
            ui.horizontal_wrapped(|ui| {
                let mut changed = false;
                for (flag, name, tip) in [
                    (&mut self.look.wire, "Wireframe", "Draw every triangle edge you can see"),
                    (&mut self.look.depth, "Depth", "Color by distance: near is yellow, far is blue. Shows layers that sit at the same place"),
                    (&mut self.look.open, "Open edges", "Yellow: borders of a surface. Magenta: two borders that run side by side without joining (a crack or break)"),
                    (&mut self.look.cross, "Intersections", "Red: triangles that pass through another surface"),
                ] {
                    changed |= ui.selectable_label(*flag, name).on_hover_text(tip).clicked().then(|| *flag = !*flag).is_some();
                }
                if changed {
                    self.look.ver += 1;
                }
            });
            if self.look.open {
                let (o, c) = (self.look.open_edges.as_ref().map(|v| v.len()).unwrap_or(0), self.look.cracks.as_ref().map(|v| v.1.len()).unwrap_or(0));
                ui.label(RichText::new(format!("{o} open edges, {c} crack pairs")).size(12.0).color(if c > 0 { theme::GOLD } else { theme::WEAK }));
            }
            if self.look.cross {
                let n = self.look.inter.as_ref().map(|v| v.1.len()).unwrap_or(0);
                ui.label(RichText::new(format!("{n} intersecting triangle pairs")).size(12.0).color(if n > 0 { theme::RED } else { theme::WEAK }));
            }
            // Probe results.
            if !self.look.probe.is_empty() {
                ui.add_space(4.0);
                ui.label(RichText::new("Under the probe, nearest first:").size(12.5));
                let mut mark = None;
                for (i, &(t, dd)) in self.look.probe.iter().take(8).enumerate() {
                    let hid = self.look.hidden[t as usize];
                    let txt = format!("{}. {} at {:.3}{}", i + 1, self.part_label(d, &topo, t), dd, if hid { " (hidden)" } else { "" });
                    if ui.link(RichText::new(txt).size(12.0).color(if hid { theme::FAINT } else { theme::TEXT })).on_hover_text("Click to mark this triangle").clicked() {
                        mark = Some(t);
                    }
                }
                if let Some(t) = mark {
                    let before = self.sel_state();
                    self.sel[t as usize] = 1.0;
                    self.push_undo(Snap { what: "mark probed".into(), sel: Some(before), gpos: None, tex: None });
                    self.sel_ver += 1;
                }
                ui.horizontal(|ui| {
                    if ui.button("Hide nearest").on_hover_text("Hide the nearest surface under the probe to see the next one").clicked() {
                        self.isolate("nearest");
                    }
                    if ui.button("Mark all").on_hover_text("Mark every triangle under the probe").clicked() {
                        let before = self.sel_state();
                        for &(t, _) in &self.look.probe {
                            self.sel[t as usize] = 1.0;
                        }
                        self.push_undo(Snap { what: "mark probed".into(), sel: Some(before), gpos: None, tex: None });
                        self.sel_ver += 1;
                    }
                });
            }
        });
        ui.add_space(8.0);
        theme::card(ui, "Locked places", |ui| {
            let n = self.look.locked.iter().filter(|&&l| l).count();
            ui.label(RichText::new(if n == 0 { "Nothing locked".to_owned() } else { format!("{} places locked (blue)", theme::num(n)) }).size(12.5).color(theme::WEAK));
            ui.horizontal_wrapped(|ui| {
                let marked = self.marked_count() > 0;
                if ui.add_enabled(marked, egui::Button::new("Lock marked")).on_hover_text("Shape brushes will not move these places, and an edit or AI fix that moves them is reported").clicked() {
                    self.lock_op("lock");
                }
                if ui.add_enabled(marked && n > 0, egui::Button::new("Unlock marked")).clicked() {
                    self.lock_op("unlock");
                }
                if ui.add_enabled(n > 0, egui::Button::new("Unlock all")).clicked() {
                    self.lock_op("unlock-all");
                }
                if ui.add_enabled(n > 0, egui::Button::new("Check a file...")).on_hover_text("Open an edited or AI-fixed GLB of this model and show which locked places moved (red)").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("GLB", &["glb"]).set_directory(requestcmd::MODDING).pick_file() {
                        self.check_locks(&p);
                    }
                }
            });
            if let Some(r) = &self.look.lock_report {
                let ok = r.bad.is_empty();
                pill(ui, if ok { "Locks held" } else { "Locks moved" }, if ok { theme::GREEN } else { theme::RED });
                ui.label(RichText::new(if ok { format!("{}: all {} locked places stayed put", r.file, r.checked) } else { format!("{}: {} of {} moved, worst {:.4} ({:.2}% of the height)", r.file, r.bad.len(), r.checked, r.bad[0].1, r.bad[0].1 / topo.height * 100.0) }).size(12.0).color(if ok { theme::WEAK } else { theme::RED }));
                if !ok && ui.button("Mark the moved places").clicked() {
                    let before = self.sel_state();
                    let bad: Vec<u32> = r.bad.iter().map(|b| b.0).collect();
                    for g in bad {
                        for &t in &topo.group_tris[g as usize] {
                            self.sel[t as usize] = 1.0;
                        }
                    }
                    self.push_undo(Snap { what: "mark moved locks".into(), sel: Some(before), gpos: None, tex: None });
                    self.sel_ver += 1;
                }
            }
        });
        ui.add_space(8.0);
        theme::card(ui, "Views", |ui| {
            ui.horizontal_wrapped(|ui| {
                for (name, yaw, pitch) in [("Under", None, -1.2f32), ("Top", None, 1.2), ("Level", None, 0.0), ("Front-left", Some(FRONT - 0.7), 0.15), ("Front-right", Some(FRONT + 0.7), 0.15)] {
                    if ui.button(name).on_hover_text("Turn the camera; the zoom and center stay").clicked() {
                        if let Some(y) = yaw {
                            cam.yaw = y;
                        }
                        cam.pitch = pitch;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.look.view_name).desired_width(150.0).hint_text("strap side close-up"));
                if ui.button("Save view").on_hover_text("Saves the camera, the pose and the time. Any AI session can render saved views of any file as acceptance pictures").clicked() {
                    let n = self.look.view_name.clone();
                    self.save_view(&n, cam);
                }
            });
            let mut go = None;
            let mut del = None;
            for (i, v) in self.look.views.iter().enumerate() {
                ui.horizontal(|ui| {
                    if ui.link(&v.name).on_hover_text(format!("{} · yaw {:.0}°, pitch {:.0}°", v.clip.clone().unwrap_or_else(|| "rest pose".into()), v.cam.yaw.to_degrees(), v.cam.pitch.to_degrees())).clicked() {
                        go = Some(i);
                    }
                    if ui.small_button("✖").on_hover_text("Delete this view").clicked() {
                        del = Some(i);
                    }
                });
            }
            if let Some(i) = go {
                let v = self.look.views[i].clone();
                self.goto_view(&v, cam);
            }
            if let Some(i) = del {
                self.delete_view(i);
            }
        });
    }

    // ---------------------------------------------------------------------------------------------
    // Drawing on the model view
    // ---------------------------------------------------------------------------------------------

    pub(super) fn draw_look(&mut self, painter: &Painter, topo: &Topo, cam: &Camera) {
        if self.vb.w == 0 || self.gpose.len() != topo.groups() {
            return;
        }
        if self.look.open {
            if self.look.open_edges.is_none() {
                self.look.open_edges = Some(edit::open_edges(topo));
            }
            let stamp = self.pose_stamp();
            if self.look.cracks.as_ref().map(|c| c.0) != Some(stamp) {
                self.look.cracks = Some((stamp, edit::cracks(topo, &self.gpose, topo.height * 0.004)));
            }
        }
        let rig = Rig::new(cam, self.vb.w, self.vb.h);
        let scr = |p: V3| -> Option<Pos2> {
            let q = rig.project(p);
            (q[2] > 0.0 || rig.ortho).then(|| self.to_screen([q[0], q[1]]))
        };
        let mut seen = vec![false; topo.tris()];
        if self.look.wire || self.look.open {
            for &id in self.vb.pick.iter().step_by(1) {
                if id >= 0 {
                    seen[id as usize] = true;
                }
            }
        }
        if self.look.wire {
            let st = Stroke::new(0.8, Color32::from_rgba_unmultiplied(235, 240, 255, 110));
            let mut done = std::collections::HashSet::new();
            for t in 0..topo.tris() {
                if !seen[t] {
                    continue;
                }
                let g = topo.tri_groups[t];
                for k in 0..3 {
                    let (a, b) = (g[k].min(g[(k + 1) % 3]), g[k].max(g[(k + 1) % 3]));
                    if done.insert((a, b)) {
                        if let (Some(p), Some(q)) = (scr(self.gpose[a as usize]), scr(self.gpose[b as usize])) {
                            painter.line_segment([p, q], st);
                        }
                    }
                }
            }
        }
        if self.look.open {
            let vis_edge = |a: u32, b: u32| topo.group_tris[a as usize].iter().any(|t| seen[*t as usize] && topo.tri_groups[*t as usize].contains(&b));
            for &(a, b) in self.look.open_edges.as_deref().unwrap_or(&[]) {
                let c = if vis_edge(a, b) { Color32::from_rgb(255, 220, 60) } else { Color32::from_rgba_unmultiplied(255, 220, 60, 70) };
                if let (Some(p), Some(q)) = (scr(self.gpose[a as usize]), scr(self.gpose[b as usize])) {
                    painter.line_segment([p, q], Stroke::new(1.8, c));
                }
            }
            for &(e1, e2) in self.look.cracks.as_ref().map(|c| c.1.as_slice()).unwrap_or(&[]) {
                for (a, b) in [e1, e2] {
                    if let (Some(p), Some(q)) = (scr(self.gpose[a as usize]), scr(self.gpose[b as usize])) {
                        painter.line_segment([p, q], Stroke::new(3.0, Color32::from_rgb(255, 60, 220)));
                    }
                }
            }
        }
        // Locked places: small blue dots; ones a checked file moved: red rings.
        if self.look.locked.iter().any(|&l| l) {
            for (g, &l) in self.look.locked.iter().enumerate() {
                if l && topo.group_tris[g].iter().any(|&t| !self.look.hidden[t as usize]) {
                    if let Some(p) = scr(self.gpose[g]) {
                        painter.circle_filled(p, 2.2, Color32::from_rgb(LOCK_BLUE[0], LOCK_BLUE[1], LOCK_BLUE[2]));
                    }
                }
            }
        }
        if let Some(r) = &self.look.lock_report {
            for &(g, _) in &r.bad {
                if let Some(p) = scr(self.gpose[g as usize]) {
                    painter.circle_stroke(p, 6.0, Stroke::new(2.0, theme::RED));
                }
            }
        }
        if let Some(b) = self.look.probe_px {
            if !self.look.probe.is_empty() {
                let p = self.to_screen(b);
                painter.circle_stroke(p, 7.0, Stroke::new(2.0, theme::TEAL));
                painter.line_segment([p - Vec2::new(11.0, 0.0), p + Vec2::new(11.0, 0.0)], Stroke::new(1.2, theme::TEAL));
                painter.line_segment([p - Vec2::new(0.0, 11.0), p + Vec2::new(0.0, 11.0)], Stroke::new(1.2, theme::TEAL));
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Script commands (tests on the hidden desktop)
    // ---------------------------------------------------------------------------------------------

    /// Returns false when `cmd` is not a Look command.
    pub(super) fn look_script(&mut self, cmd: &str, rest: &str, cam: &mut Camera, d: &Doc, topo: &Topo) -> bool {
        match cmd {
            "isolate" => self.isolate("only"),
            "hide-marked" => self.isolate("hide"),
            "hide-nearest" => self.isolate("nearest"),
            "show-all" => self.isolate("all"),
            "walls" => match rest {
                "front" => self.keep_walls(Wall::Front, cam),
                "back" => self.keep_walls(Wall::Back, cam),
                _ => self.keep_walls(Wall::Side, cam),
            },
            "lock" => self.lock_op("lock"),
            "unlock" => self.lock_op("unlock"),
            "unlock-all" => self.lock_op("unlock-all"),
            "check-locks" => self.check_locks(std::path::Path::new(rest)),
            "overlay" => {
                // overlay wire|depth|open|cross [on|off]
                let mut it = rest.split_whitespace();
                let which = it.next().unwrap_or("");
                let on = it.next().map(|s| s != "off").unwrap_or(true);
                match which {
                    "wire" | "wireframe" => self.look.wire = on,
                    "depth" => self.look.depth = on,
                    "open" | "cracks" => self.look.open = on,
                    "cross" | "intersections" => self.look.cross = on,
                    _ => println!("mark script: overlay {which}?"),
                }
                self.look.ver += 1;
            }
            "probe" => {
                let v: Vec<f32> = rest.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if v.len() == 2 {
                    let r = self.rect_v;
                    let p = Pos2::new(r.min.x + v[0] * r.width(), r.min.y + v[1] * r.height());
                    let b = self.to_buf(p);
                    self.probe_at(b, cam, topo);
                    for (i, &(t, dd)) in self.look.probe.iter().enumerate() {
                        println!("  probe {}: {} at {dd:.4}", i + 1, self.part_label(d, topo, t));
                    }
                }
            }
            "clip" => {
                if rest == "none" || rest.is_empty() {
                    self.clip = None;
                } else if let Some(c) = d.model.clips.iter().find(|c| c.as_str() == rest || c.contains(rest)) {
                    self.clip = Some(c.clone());
                } else {
                    println!("mark script: no clip {rest}; clips: {}", d.model.clips.join(", "));
                }
            }
            "time" => self.time = rest.parse().unwrap_or(self.time),
            "save-view" => self.save_view(rest, cam),
            "goto-view" => match self.look.views.iter().find(|v| v.name == rest).cloned() {
                Some(v) => self.goto_view(&v, cam),
                None => println!("mark script: no saved view {rest}"),
            },
            "look-report" => {
                let o = self.look.open_edges.as_ref().map(|v| v.len());
                let c = self.look.cracks.as_ref().map(|v| v.1.len());
                let x = self.look.inter.as_ref().map(|v| v.1.len());
                println!(
                    "look: {} hidden, {} locked, open edges {:?}, crack pairs {:?}, intersecting pairs {:?}, probe {}, views {}",
                    self.look.hidden.iter().filter(|&&h| h).count(),
                    self.look.locked.iter().filter(|&&l| l).count(),
                    o,
                    c,
                    x,
                    self.look.probe.len(),
                    self.look.views.len()
                );
                if let Some((_, cr)) = &self.look.cracks {
                    for &((a, b), (e, f)) in cr.iter().take(12) {
                        let p = |g: u32| self.gpose[g as usize];
                        let t1 = topo.group_tris[a as usize].iter().find(|t| topo.tri_groups[**t as usize].contains(&b)).copied();
                        let t2 = topo.group_tris[e as usize].iter().find(|t| topo.tri_groups[**t as usize].contains(&f)).copied();
                        println!(
                            "  crack ({:.3},{:.3},{:.3})-({:.3},{:.3},{:.3}) {} | {}",
                            p(a)[0], p(a)[1], p(a)[2], p(b)[0], p(b)[1], p(b)[2],
                            t1.map(|t| self.part_label(d, topo, t)).unwrap_or_default(),
                            t2.map(|t| self.part_label(d, topo, t)).unwrap_or_default()
                        );
                    }
                }
            }
            _ => return false,
        }
        true
    }
}
