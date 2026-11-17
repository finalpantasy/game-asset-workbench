//! The Mark & fix page: point at part of the model or texture and say what is wrong (a request any
//! AI session can read), or make a rough edit by hand and save it as a new file.
//!
//! Marking: a soft brush, a lasso, kit parts and edge loops, on the model and on the texture, with
//! X mirror on by default. The marks show in both views at once.
//! Mesh brushes (Grab, Smooth, Inflate, Deflate, Flatten) move bind-pose vertices and keep skin
//! weights; texture brushes (Paint, Clone, Smudge) paint the sheet directly or through the model.
//! Everything has undo (Ctrl+Z) and redo (Ctrl+Y); Esc cancels a stroke.
use super::{App, BACK, FRONT, SIDE};
use crate::doc::Doc;
use crate::requestcmd::{self, CheckResult, ReqInput, Saved};
use crate::theme::{self, pill};
use eframe::egui::{self, Color32, CornerRadius, FontFamily, Key, Margin, Painter, PointerButton, Pos2, Rect, RichText, Sense, Shape, Stroke, Vec2};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use wb_core::analysis::Parts;
use wb_core::bake::Rig;
use wb_core::edit::{self, EdgeLoop, Grab, Sculpt, Topo, NONE};
use wb_core::geom::{add, cross, dist, norm, scale, sub, V3};
use wb_core::render::{self, Camera, Filter};
use wb_core::Texture;

#[path = "markinspect.rs"]
mod markinspect;
#[path = "markhair.rs"]
mod markhair;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tool {
    Turn,
    Part,
    Brush,
    Lasso,
    Edge,
    Probe,
    Grab,
    Smooth,
    Inflate,
    Deflate,
    Flatten,
    Paint,
    Clone,
    Smudge,
    Comb,
}

const TOOLS: [Tool; 15] = [Tool::Turn, Tool::Part, Tool::Brush, Tool::Lasso, Tool::Edge, Tool::Probe, Tool::Grab, Tool::Smooth, Tool::Inflate, Tool::Deflate, Tool::Flatten, Tool::Paint, Tool::Clone, Tool::Smudge, Tool::Comb];

impl Tool {
    fn label(self) -> &'static str {
        match self {
            Tool::Turn => "Turn",
            Tool::Part => "Part",
            Tool::Brush => "Mark",
            Tool::Lasso => "Lasso",
            Tool::Edge => "Edge",
            Tool::Probe => "Look",
            Tool::Grab => "Grab",
            Tool::Smooth => "Smooth",
            Tool::Inflate => "Inflate",
            Tool::Deflate => "Deflate",
            Tool::Flatten => "Flatten",
            Tool::Paint => "Paint",
            Tool::Clone => "Clone",
            Tool::Smudge => "Smudge",
            Tool::Comb => "Comb",
        }
    }
    fn tip(self) -> &'static str {
        match self {
            Tool::Turn => "Drag to turn the model or move the texture. Right-drag turns with any tool; scroll zooms; middle-drag moves.",
            Tool::Part => "Click a part (glove, belt, hair...) to mark all of it. Ctrl+click removes it.",
            Tool::Brush => "Paint a soft mark on the model or the texture. Ctrl+drag rubs marks out.",
            Tool::Lasso => "Draw around an area to mark everything inside that you can see. Ctrl+lasso removes.",
            Tool::Edge => "Click an edge to pick the edge loop it belongs to. Shift+click adds a path from the last point to here. Ctrl+click removes a loop.",
            Tool::Probe => "Click a spot to list every surface under it, front to back, even hidden ones. Use it to find what sits behind a seam or a break.",
            Tool::Grab => "Drag to pull the surface. Works inside the marked area when something is marked.",
            Tool::Smooth => "Brush over lumps and creases to even them out.",
            Tool::Inflate => "Brush to push the surface outward.",
            Tool::Deflate => "Brush to pull the surface inward.",
            Tool::Flatten => "Brush to press the surface flat.",
            Tool::Paint => "Paint color on the model or the texture. On the model it reaches across texture seams.",
            Tool::Clone => "Alt+click where to copy from, then paint to copy that area here.",
            Tool::Smudge => "Drag to smear the colors under the brush.",
            Tool::Comb => "Drag along the hair the way it should flow, root to tip. The strokes go into the request (strokes.json) for the hair tool. Ctrl+click a stroke removes it.",
        }
    }
    fn by_name(s: &str) -> Option<Tool> {
        TOOLS.iter().copied().find(|t| t.label().eq_ignore_ascii_case(s) || format!("{t:?}").eq_ignore_ascii_case(s))
    }
    fn sculpt(self) -> Option<Sculpt> {
        match self {
            Tool::Grab => Some(Sculpt::Grab),
            Tool::Smooth => Some(Sculpt::Smooth),
            Tool::Inflate => Some(Sculpt::Inflate),
            Tool::Deflate => Some(Sculpt::Deflate),
            Tool::Flatten => Some(Sculpt::Flatten),
            _ => None,
        }
    }
    fn paints(self) -> bool {
        matches!(self, Tool::Paint | Tool::Clone | Tool::Smudge)
    }
    fn round_brush(self) -> bool {
        matches!(self, Tool::Brush | Tool::Grab | Tool::Smooth | Tool::Inflate | Tool::Deflate | Tool::Flatten | Tool::Paint | Tool::Clone | Tool::Smudge)
    }
}

#[derive(Clone)]
struct SelState {
    sel: Vec<f32>,
    extra: Vec<Option<(usize, usize, Vec<u8>)>>,
    loops: Vec<EdgeLoop>,
}

/// One undo step: whatever it holds is swapped with the current state.
struct Snap {
    what: String,
    sel: Option<SelState>,
    gpos: Option<Vec<V3>>,
    /// (image, x0, y0, w, h, rgba of that box)
    tex: Option<(usize, [usize; 4], Vec<u8>)>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Where {
    Model,
    Texture,
}

#[derive(Clone, Copy, Debug, Default)]
struct Mods {
    ctrl: bool,
    shift: bool,
    alt: bool,
}

#[derive(Clone, Copy, Debug)]
enum Ev {
    Press(Where, Pos2, Mods),
    Move(Pos2),
    Release,
}

enum Drag {
    Select { before: SelState, erase: bool, place: Where, last: Option<Pos2> },
    Lasso { before: SelState, erase: bool, place: Where, pts: Vec<Pos2> },
    Sculpt { kind: Sculpt, before: Vec<V3>, grab: Option<(Grab, f32, Pos2)>, last: Option<Pos2> },
    Paint { place: Where, image: usize, before: Vec<u8>, bbox: [usize; 4], stroke_w: HashMap<usize, f32>, last: Option<Pos2>, clone_off: Option<Vec2>, clean: Option<(usize, usize, Vec<u8>)> },
    Comb { pts: Vec<markhair::CombPt>, last: Pos2 },
}

enum Work {
    Request(Result<Saved, String>),
    Glb(Result<PathBuf, String>),
    Progress(String),
    Checks(CheckResult),
}

struct ViewBuf {
    tex: Option<egui::TextureHandle>,
    w: usize,
    h: usize,
    pick: Vec<i32>,
    key: Option<(Camera, usize, usize, u64, bool, Option<String>, u32, bool)>,
}

pub struct Mark {
    doc_key: usize,
    topo: Option<Arc<Topo>>,
    topo_rx: Option<Receiver<Topo>>,
    sel: Vec<f32>,
    extra: Vec<Option<(usize, usize, Vec<u8>)>>,
    loops: Vec<EdgeLoop>,
    gpos: Vec<V3>,
    /// Painted copies of the sheets (all of them once anything is painted).
    textures: Vec<Option<Texture>>,
    painted: bool,
    parts: Option<Arc<Parts>>,
    cam_on_model: bool,
    pub tool: Tool,
    radius: f32,
    strength: f32,
    sym: bool,
    color: Color32,
    note: String,
    clip: Option<String>,
    time: f32,
    // versions, for redraws
    sel_ver: u64,
    pos_ver: u64,
    tex_ver: u64,
    /// (pos_ver, tex_ver) when the edit was last saved to a file.
    saved_ver: (u64, u64),
    // cached pose
    pose_key: Option<(u64, Option<String>, u32)>,
    mesh_pos: Vec<Vec<V3>>,
    pose: Vec<Vec<V3>>,
    gpose: Vec<V3>,
    // model view
    vb: ViewBuf,
    rect_v: Rect,
    look: markinspect::Look,
    hair: markhair::HairTools,
    // texture view
    image: usize,
    tex_zoom: f32,
    tex_pan: Vec2,
    rect_t: Rect,
    sheet_tex: Vec<Option<egui::TextureHandle>>,
    sheet_ver: Vec<u64>,
    overlay: Option<(usize, u64, egui::TextureHandle)>,
    // input
    drag: Option<Drag>,
    cam_drag: Option<PointerButton>,
    clone_src: Option<(Where, Pos2)>,
    undo: Vec<Snap>,
    redo: Vec<Snap>,
    // work in the background
    work_rx: Option<Receiver<Work>>,
    busy: Option<String>,
    checks: Option<CheckResult>,
    checks_running: bool,
    saved_glb: Option<PathBuf>,
    saved_req: Option<Saved>,
    message: Option<(String, bool)>,
    // scripted runs
    script: VecDeque<String>,
    script_events: VecDeque<(Ev, Where)>,
    script_wait: u32,
    scripted: bool,
    requests_dir: PathBuf,
}

impl Mark {
    pub fn from_args(args: &crate::Args) -> Result<Mark, String> {
        let script: VecDeque<String> = match args.opt("--mark-script") {
            Some(p) => std::fs::read_to_string(p)
                .map_err(|e| format!("--mark-script {p}: {e}"))?
                .lines()
                .map(|l| l.trim().to_owned())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .collect(),
            None => VecDeque::new(),
        };
        Ok(Mark {
            doc_key: 0,
            topo: None,
            topo_rx: None,
            sel: vec![],
            extra: vec![],
            loops: vec![],
            gpos: vec![],
            textures: vec![],
            painted: false,
            parts: None,
            cam_on_model: true,
            tool: Tool::Brush,
            radius: 28.0,
            strength: 0.6,
            sym: true,
            color: Color32::from_rgb(0xd8, 0x3a, 0x2e),
            note: String::new(),
            clip: None,
            time: 0.0,
            sel_ver: 1,
            pos_ver: 1,
            tex_ver: 1,
            saved_ver: (0, 0),
            pose_key: None,
            mesh_pos: vec![],
            pose: vec![],
            gpose: vec![],
            vb: ViewBuf { tex: None, w: 0, h: 0, pick: vec![], key: None },
            rect_v: Rect::NOTHING,
            look: markinspect::Look::default(),
            hair: markhair::HairTools::default(),
            image: 0,
            tex_zoom: 1.0,
            tex_pan: Vec2::ZERO,
            rect_t: Rect::NOTHING,
            sheet_tex: vec![],
            sheet_ver: vec![],
            overlay: None,
            drag: None,
            cam_drag: None,
            clone_src: None,
            undo: vec![],
            redo: vec![],
            work_rx: None,
            busy: None,
            checks: None,
            checks_running: false,
            saved_glb: None,
            saved_req: None,
            message: None,
            scripted: !script.is_empty(),
            script,
            script_events: VecDeque::new(),
            script_wait: 0,
            requests_dir: args.opt("--requests-dir").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(requestcmd::REQUESTS_DIR)),
        })
    }

    /// Something is still running or the script is not finished (screenshots wait for it).
    pub fn busy(&self) -> bool {
        self.busy.is_some() || self.checks_running || self.topo_rx.is_some() || !self.script.is_empty() || !self.script_events.is_empty() || self.script_wait > 0 || (self.scripted && self.topo.is_none())
    }

    fn reset(&mut self, d: &Arc<Doc>) {
        self.doc_key = Arc::as_ptr(d) as usize;
        self.topo = None;
        let (tx, rx) = channel();
        let dd = d.clone();
        std::thread::spawn(move || {
            let _ = tx.send(Topo::build(&dd.model));
        });
        self.topo_rx = Some(rx);
        self.textures.clear();
        self.painted = false;
        self.undo.clear();
        self.redo.clear();
        self.loops.clear();
        self.drag = None;
        self.pose_key = None;
        self.vb.key = None;
        self.sheet_tex.clear();
        self.sheet_ver.clear();
        self.overlay = None;
        self.image = d.model.meshes.iter().position(|m| m.image.is_some()).and_then(|i| d.model.meshes[i].image).unwrap_or(0);
        self.bump_all();
    }

    fn bump_all(&mut self) {
        self.sel_ver += 1;
        self.pos_ver += 1;
        self.tex_ver += 1;
    }

    fn sel_state(&self) -> SelState {
        SelState { sel: self.sel.clone(), extra: self.extra.clone(), loops: self.loops.clone() }
    }

    fn textures<'a>(&'a self, d: &'a Doc) -> &'a [Option<Texture>] {
        if self.painted {
            &self.textures
        } else {
            &d.textures
        }
    }

    fn marked_count(&self) -> usize {
        self.sel.iter().filter(|&&w| w > 0.0).count()
    }

    fn has_edits(&self, topo: &Topo) -> bool {
        self.gpos.iter().zip(&topo.rest).any(|(a, b)| a != b) || self.painted
    }

    fn edited_textures(&self, d: &Doc) -> Vec<Option<Texture>> {
        if !self.painted {
            return vec![];
        }
        self.textures
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let same = match (t, d.textures.get(i).and_then(|o| o.as_ref())) {
                    (Some(a), Some(b)) => a.rgba == b.rgba,
                    _ => true,
                };
                if same {
                    None
                } else {
                    t.as_ref().map(|t| Texture::new(t.w, t.h, t.rgba.clone()))
                }
            })
            .collect()
    }

    fn push_undo(&mut self, s: Snap) {
        self.undo.push(s);
        if self.undo.len() > 80 {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    fn swap_snap(&mut self, mut s: Snap) -> Snap {
        if let Some(st) = s.sel.take() {
            let cur = self.sel_state();
            self.sel = st.sel;
            self.extra = st.extra;
            self.loops = st.loops;
            s.sel = Some(cur);
            self.sel_ver += 1;
        }
        if let Some(g) = s.gpos.take() {
            s.gpos = Some(std::mem::replace(&mut self.gpos, g));
            self.pos_ver += 1;
        }
        if let Some((img, b, px)) = s.tex.take() {
            if let Some(Some(t)) = self.textures.get_mut(img) {
                let cur = crop(&t.rgba, t.w, b);
                paste(&mut t.rgba, t.w, b, &px);
                let nt = Texture::new(t.w, t.h, std::mem::take(&mut t.rgba));
                *t = nt;
                s.tex = Some((img, b, cur));
                self.tex_ver += 1;
            }
        }
        s
    }

    fn undo(&mut self) {
        if let Some(s) = self.undo.pop() {
            let what = s.what.clone();
            let r = self.swap_snap(s);
            self.redo.push(r);
            self.say(format!("Undid: {what}"), true);
        }
    }

    fn redo(&mut self) {
        if let Some(s) = self.redo.pop() {
            let what = s.what.clone();
            let r = self.swap_snap(s);
            self.undo.push(r);
            self.say(format!("Redid: {what}"), true);
        }
    }

    fn say(&mut self, s: String, ok: bool) {
        if self.scripted {
            println!("mark: {s}");
        }
        self.message = Some((s, ok));
    }

    /// Pose of the edited model (rest pose while a mesh or texture brush is chosen).
    fn update_pose(&mut self, d: &Doc, topo: &Topo) {
        let clip = if self.tool.sculpt().is_some() || self.tool.paints() { None } else { self.clip.clone() };
        let key = (self.pos_ver, clip.clone(), (self.time * 1000.0).round() as u32);
        if self.pose_key.as_ref() == Some(&key) {
            return;
        }
        self.mesh_pos = topo.mesh_positions(&d.model, &self.gpos);
        self.pose = requestcmd::posed(&d.model, &self.mesh_pos, clip.as_deref(), self.time);
        self.gpose = topo.members.iter().map(|m| self.pose[m[0].0 as usize][m[0].1 as usize]).collect();
        self.pose_key = Some(key);
    }

    fn posed_clip(&self) -> Option<String> {
        if self.tool.sculpt().is_some() || self.tool.paints() {
            None
        } else {
            self.clip.clone()
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Small geometry helpers
// -------------------------------------------------------------------------------------------------

fn crop(rgba: &[u8], w: usize, b: [usize; 4]) -> Vec<u8> {
    let [x0, y0, bw, bh] = b;
    let mut out = Vec::with_capacity(bw * bh * 4);
    for y in y0..y0 + bh {
        out.extend_from_slice(&rgba[(y * w + x0) * 4..(y * w + x0 + bw) * 4]);
    }
    out
}

fn paste(rgba: &mut [u8], w: usize, b: [usize; 4], px: &[u8]) {
    let [x0, y0, bw, bh] = b;
    for (k, y) in (y0..y0 + bh).enumerate() {
        rgba[(y * w + x0) * 4..(y * w + x0 + bw) * 4].copy_from_slice(&px[k * bw * 4..(k + 1) * bw * 4]);
    }
}

fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_sq().max(1e-9)).clamp(0.0, 1.0);
    (a + ab * t).distance(p)
}

/// Distance from a point to a 2D triangle (0 inside).
fn tri_dist2(p: [f32; 2], t: [[f32; 2]; 3]) -> f32 {
    let s = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    let (d1, d2, d3) = (s(t[0], t[1], p), s(t[1], t[2], p), s(t[2], t[0], p));
    let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    if !(neg && pos) {
        return 0.0;
    }
    let q = Pos2::new(p[0], p[1]);
    let v = t.map(|a| Pos2::new(a[0], a[1]));
    seg_dist(q, v[0], v[1]).min(seg_dist(q, v[1], v[2])).min(seg_dist(q, v[2], v[0]))
}

fn point_in_poly(p: Pos2, poly: &[Pos2]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Calls `f(x, y)` for every pixel center inside the polygon (even-odd), scanline by scanline.
fn fill_poly(poly: &[Pos2], w: usize, h: usize, mut f: impl FnMut(usize, usize)) {
    if poly.len() < 3 {
        return;
    }
    let (y0, y1) = poly.iter().fold((f32::MAX, f32::MIN), |a, p| (a.0.min(p.y), a.1.max(p.y)));
    let ys = (y0.floor().max(0.0)) as usize;
    let ye = (y1.ceil().max(0.0) as usize).min(h);
    let mut xs: Vec<f32> = Vec::new();
    for y in ys..ye {
        let cy = y as f32 + 0.5;
        xs.clear();
        let n = poly.len();
        for i in 0..n {
            let (a, b) = (poly[i], poly[(i + 1) % n]);
            if (a.y > cy) != (b.y > cy) {
                xs.push(a.x + (cy - a.y) * (b.x - a.x) / (b.y - a.y));
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for pair in xs.chunks_exact(2) {
            let xa = (pair[0] - 0.5).ceil().max(0.0) as usize;
            let xb = ((pair[1] - 0.5).floor()).min(w as f32 - 1.0);
            if xb < 0.0 {
                continue;
            }
            for x in xa..=(xb as usize) {
                f(x, y);
            }
        }
    }
}

fn unproject(rig: &Rig, px: f32, py: f32, z: f32) -> V3 {
    if rig.ortho {
        let (a, b) = ((px - rig.cx) / rig.oscale, -(py - rig.cy) / rig.oscale);
        add(add(add(rig.eye, scale(rig.r, a)), scale(rig.u, b)), scale(rig.f, z))
    } else {
        let (xr, yu) = ((px - rig.cx) / rig.focal, -(py - rig.cy) / rig.focal);
        add(rig.eye, scale(add(add(rig.f, scale(rig.r, xr)), scale(rig.u, yu)), z))
    }
}

fn lerp_px(a: [u8; 4], b: [f32; 3], t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    let m = |c: usize| (a[c] as f32 + (b[c] - a[c] as f32) * t).round().clamp(0.0, 255.0) as u8;
    [m(0), m(1), m(2), a[3]]
}

fn sample_rgb(rgba: &[u8], w: usize, h: usize, x: f32, y: f32) -> Option<[f32; 3]> {
    if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 {
        return None;
    }
    let o = (y as usize * w + x as usize) * 4;
    Some([rgba[o] as f32, rgba[o + 1] as f32, rgba[o + 2] as f32])
}

// -------------------------------------------------------------------------------------------------
// The page
// -------------------------------------------------------------------------------------------------

impl App {
    pub(super) fn mark_page(&mut self, ctx: &egui::Context) {
        let Some(d) = self.inspect.clone() else {
            egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG).inner_margin(Margin::same(14))).show(ctx, |ui| {
                ui.label(RichText::new("Mark & fix").family(FontFamily::Name("strong".into())).size(22.0));
                ui.label(RichText::new("Opening the model...").color(theme::WEAK));
            });
            return;
        };
        if self.mark.doc_key != Arc::as_ptr(&d) as usize {
            self.mark.reset(&d);
        }
        if let Some(rx) = &self.mark.topo_rx {
            if let Ok(t) = rx.try_recv() {
                self.mark.sel = vec![0.0; t.tris()];
                self.mark.gpos = t.rest.clone();
                self.mark.extra = vec![None; d.model.images.len()];
                self.mark.topo = Some(Arc::new(t));
                self.mark.topo_rx = None;
                self.mark.bump_all();
            } else {
                ctx.request_repaint();
            }
        }
        self.mark.receive(ctx);
        self.mark.parts = self.parts.clone();
        let parts = self.parts.clone();

        egui::SidePanel::right("mark-info").default_width(330.0).min_width(300.0).resizable(true).frame(egui::Frame::new().fill(theme::PANEL).inner_margin(Margin::same(12))).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                self.mark_panel(ui, ctx, &d);
            });
        });
        egui::SidePanel::left("mark-tools").exact_width(84.0).resizable(false).frame(egui::Frame::new().fill(theme::BG).inner_margin(Margin::symmetric(6, 12))).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.mark.tool_column(ui));
        });
        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG).inner_margin(Margin::same(14))).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Mark & fix").family(FontFamily::Name("strong".into())).size(22.0));
                ui.add_space(12.0);
                self.file_picker(ui, ctx, crate::doc::Slot::Inspect, Some(&d));
                ui.add_space(8.0);
                for (yaw, name, key) in [(FRONT, "Front", "F"), (BACK, "Back", "B"), (SIDE, "Side", "S")] {
                    if ui.button(name).on_hover_text(format!("Turn to the {} ({key})", name.to_lowercase())).clicked() {
                        self.reframe = Some(yaw);
                    }
                }
                if ui.add_enabled(self.mark.marked_count() > 0, egui::Button::new("Frame marks")).on_hover_text("Zoom in on the marked area").clicked() {
                    self.mark.frame_marks(&mut self.cam);
                }
            });
            ui.add_space(6.0);
            let Some(topo) = self.mark.topo.clone() else {
                ui.add(egui::Spinner::new().size(18.0).color(theme::GOLD));
                ui.label(RichText::new("Finding the surfaces (welding seams, mirror pairs)...").color(theme::WEAK));
                return;
            };
            let (area, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
            let gap = 12.0;
            let tex_w = (area.width() * 0.44).min(area.height() + 40.0);
            let rect_v = Rect::from_min_size(area.min, Vec2::new(area.width() - tex_w - gap, area.height()));
            let rect_t = Rect::from_min_size(Pos2::new(rect_v.max.x + gap, area.min.y), Vec2::new(tex_w, area.height()));
            if let Some(yaw) = self.reframe.take() {
                let mut c = Camera::fit(&d.model, rect_v.width() / rect_v.height().max(1.0), 0.8, false);
                c.yaw = yaw;
                self.cam = c;
            }
            self.mark.rect_v = rect_v;
            self.mark.rect_t = rect_t;
            self.mark.update_pose(&d, &topo);
            self.mark.ensure_look(&topo);
            self.mark.render_view(ui, &d, &topo, &self.cam);
            // Script first (one step per frame), then real input.
            self.mark.run_script(ctx, &d, &topo, &mut self.cam, parts.as_deref());
            self.mark.input(ui, &d, &topo, &mut self.cam, parts.as_deref());
            self.mark.draw_view(ui, &d, &topo, &self.cam, parts.as_deref());
            self.mark.texture_view(ui, ctx, &d, &topo);
        });
        let t = self.mark.tool;
        self.hint = format!("{}: {}   ·   Ctrl+Z undo, Ctrl+Y redo, Esc cancels, [ and ] change the brush size", t.label(), t.tip());
        if self.mark.drag.is_some() || self.mark.busy.is_some() || self.mark.checks_running {
            ctx.request_repaint();
        }
    }

    fn mark_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, d: &Arc<Doc>) {
        let m = &mut self.mark;
        let topo = m.topo.clone();
        theme::card(ui, "Brush", |ui| {
            ui.add(egui::Slider::new(&mut m.radius, 3.0..=220.0).text("size (px)").logarithmic(true)).on_hover_text("Brush size on screen. [ and ] change it.");
            ui.add(egui::Slider::new(&mut m.strength, 0.02..=1.0).text("strength")).on_hover_text("How much one stroke does");
            ui.horizontal(|ui| {
                super::toggle(ui, &mut m.sym).on_hover_text("Marks and brushes work on both sides (left and right) at once");
                ui.label("Mirror left and right");
            });
            ui.horizontal(|ui| {
                ui.label("Paint color");
                egui::color_picker::color_edit_button_srgba(ui, &mut m.color, egui::color_picker::Alpha::Opaque);
            });
        });
        ui.add_space(8.0);
        theme::card(ui, "Marked", |ui| {
            let n = m.marked_count();
            ui.label(RichText::new(if n == 0 { "Nothing marked yet".to_owned() } else { format!("{} triangles marked", theme::num(n)) }).color(if n == 0 { theme::WEAK } else { theme::TEXT }));
            if let (Some(t), true) = (&topo, n > 0) {
                let mut by: Vec<(String, usize)> = Vec::new();
                for (g, &w) in m.sel.iter().enumerate() {
                    if w <= 0.0 {
                        continue;
                    }
                    let (mi, ti) = t.tri_loc[g];
                    let p = d.tri_parts.get(mi).and_then(|v| v.get(ti)).copied().unwrap_or(0);
                    let name = if p == 0 { "no part".to_owned() } else { self.parts.as_ref().map(|x| super::title_case(x.name(p))).unwrap_or_else(|| format!("part {p}")) };
                    match by.iter_mut().find(|e| e.0 == name) {
                        Some(e) => e.1 += 1,
                        None => by.push((name, 1)),
                    }
                }
                by.sort_by(|a, b| b.1.cmp(&a.1));
                for (name, c) in by.iter().take(5) {
                    ui.label(RichText::new(format!("{name}: {c}")).color(theme::WEAK).size(12.5));
                }
            }
            if !m.loops.is_empty() {
                let mut remove = None;
                for (i, l) in m.loops.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("Edge {}: {} points{}{}", i + 1, l.groups.len(), if l.closed { ", closed" } else { "" }, if l.border { ", open edge" } else { "" })).color(theme::TEAL));
                        if ui.small_button("Remove").on_hover_text("Remove this edge").clicked() {
                            remove = Some(i);
                        }
                    });
                }
                if let Some(i) = remove {
                    let before = m.sel_state();
                    m.loops.remove(i);
                    m.push_undo(Snap { what: "remove edge".into(), sel: Some(before), gpos: None, tex: None });
                    m.sel_ver += 1;
                }
            }
            ui.horizontal_wrapped(|ui| {
                let any = n > 0;
                if ui.add_enabled(any, egui::Button::new("Grow")).on_hover_text("Mark one more ring of triangles").clicked() {
                    m.sel_op("grow");
                }
                if ui.add_enabled(any, egui::Button::new("Shrink")).on_hover_text("Unmark the outer ring").clicked() {
                    m.sel_op("shrink");
                }
                if ui.add_enabled(any, egui::Button::new("Mirror")).on_hover_text("Copy the marks to the other side").clicked() {
                    m.sel_op("mirror");
                }
                if ui.add_enabled(any || !m.loops.is_empty() || m.extra.iter().any(|e| e.is_some()), egui::Button::new("Clear")).on_hover_text("Unmark everything").clicked() {
                    m.sel_op("clear");
                }
            });
        });
        ui.add_space(8.0);
        m.look_card(ui, d, &mut self.cam);
        ui.add_space(8.0);
        m.hair_card(ui, ctx, d, &mut self.cam);
        ui.add_space(8.0);
        theme::card(ui, "Tell the AI", |ui| {
            ui.add(egui::TextEdit::multiline(&mut m.note).desired_rows(5).desired_width(f32::INFINITY).hint_text("What is wrong here, or what you want. For example: \"the strap should leave the belt along this edge, not float in front of it\"."));
            ui.add_space(4.0);
            let can = m.busy.is_none() && topo.is_some() && (!m.note.trim().is_empty() || m.marked_count() > 0 || !m.loops.is_empty());
            let b = egui::Button::new(RichText::new("Save request").size(16.0).strong().color(Color32::from_rgb(0x14, 0x16, 0x1b))).fill(theme::GOLD).min_size(Vec2::new(ui.available_width(), 36.0));
            if ui.add_enabled(can, b).on_hover_text("Saves the marks, your note and pictures to a new request folder and copies a one-line prompt for the AI").clicked() {
                m.save_request(ctx, d, self.parts.clone(), &self.cam, false);
            }
            if let Some(s) = &m.saved_req {
                ui.label(RichText::new(format!("Saved: {}", s.summary)).color(theme::GREEN).size(12.5));
                ui.label(RichText::new(s.dir.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()).monospace().size(11.5).color(theme::WEAK));
                ui.horizontal(|ui| {
                    if ui.button("Copy prompt").on_hover_text(&s.prompt).clicked() {
                        ctx.copy_text(s.prompt.clone());
                    }
                    if ui.button("Open folder").clicked() {
                        let _ = std::process::Command::new("explorer").arg(&s.dir).spawn();
                    }
                });
            }
        });
        ui.add_space(8.0);
        theme::card(ui, "Your edit", |ui| {
            let (moved, mx) = topo.as_ref().map(|t| edit::moved(t, &m.gpos)).unwrap_or((0, 0.0));
            let h = topo.as_ref().map(|t| t.height).unwrap_or(1.0);
            ui.label(RichText::new(if moved == 0 { "Shape unchanged".to_owned() } else { format!("{} places moved, up to {:.1}% of his height", theme::num(moved), mx / h * 100.0) }).color(theme::WEAK).size(12.5));
            ui.label(RichText::new(if m.painted { "Texture painted" } else { "Texture unchanged" }).color(theme::WEAK).size(12.5));
            ui.horizontal(|ui| {
                if ui.add_enabled(!m.undo.is_empty(), egui::Button::new("Undo")).on_hover_text(m.undo.last().map(|s| format!("Undo {} (Ctrl+Z)", s.what)).unwrap_or_default()).clicked() {
                    m.undo();
                }
                if ui.add_enabled(!m.redo.is_empty(), egui::Button::new("Redo")).on_hover_text("Ctrl+Y").clicked() {
                    m.redo();
                }
            });
            let edits = topo.as_ref().is_some_and(|t| m.has_edits(t));
            ui.horizontal(|ui| {
                if ui.add_enabled(edits && m.busy.is_none(), egui::Button::new("Save as...")).on_hover_text("Writes the edited model to a NEW file in the Modding folder, then runs the checks").clicked() {
                    m.save_as_dialog(ctx, d);
                }
                if ui.add_enabled(edits && m.busy.is_none(), egui::Button::new("Fix my edit")).on_hover_text("Saves a request with the model before and after your edit, so the AI can clean it up (smooth, re-weight, repaint seams). Uses the note above.").clicked() {
                    m.save_request(ctx, d, self.parts.clone(), &self.cam, true);
                }
            });
            if let Some(p) = &m.saved_glb {
                ui.label(RichText::new(format!("Saved {}", p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default())).size(12.5).color(theme::WEAK));
            }
            if m.checks_running {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new().size(14.0).color(theme::GOLD));
                    ui.label(RichText::new(m.busy.clone().unwrap_or_else(|| "Checking".into())).size(12.5));
                });
            }
            if let Some(c) = &m.checks {
                let ok = c.passed();
                pill(ui, if ok { "Checks passed" } else { "Checks failed" }, if ok { theme::GREEN } else { theme::RED });
                for l in &c.lines {
                    ui.label(RichText::new(format!("{} {}", if l.ok { "✔" } else { "✖" }, l.text)).size(12.0).color(if l.ok { theme::WEAK } else { theme::RED }));
                }
            }
        });
        if let Some((msg, ok)) = &m.message {
            ui.add_space(6.0);
            ui.label(RichText::new(msg).size(12.5).color(if *ok { theme::WEAK } else { theme::RED }));
        }
        ui.add_space(8.0);
        theme::card(ui, "Pose", |ui| {
            let mut sel = m.clip.clone();
            egui::ComboBox::from_id_salt("mark-clip").selected_text(sel.clone().unwrap_or_else(|| "Rest pose".into())).width(200.0).show_ui(ui, |ui| {
                ui.selectable_value(&mut sel, None, "Rest pose");
                for c in &d.model.clips {
                    ui.selectable_value(&mut sel, Some(c.clone()), c);
                }
            });
            if sel != m.clip {
                m.clip = sel;
                m.time = 0.0;
            }
            if let Some(ci) = m.clip.as_deref().and_then(|c| d.model.clips.iter().position(|x| x == c)) {
                let dur = d.model.clip_duration(ci).max(0.01);
                ui.add(egui::Slider::new(&mut m.time, 0.0..=dur).text("seconds"));
            }
            if m.tool.sculpt().is_some() || m.tool.paints() {
                ui.label(RichText::new("Brushes that change the model work in the rest pose.").size(12.0).color(theme::FAINT));
            }
        });
    }
}

impl Mark {
    fn tool_column(&mut self, ui: &mut egui::Ui) {
        let groups: [(&str, &[Tool]); 4] = [("MARK", &[Tool::Turn, Tool::Part, Tool::Brush, Tool::Lasso, Tool::Edge, Tool::Probe]), ("SHAPE", &[Tool::Grab, Tool::Smooth, Tool::Inflate, Tool::Deflate, Tool::Flatten]), ("PAINT", &[Tool::Paint, Tool::Clone, Tool::Smudge]), ("HAIR", &[Tool::Comb])];
        for (title, tools) in groups {
            ui.vertical_centered(|ui| ui.label(RichText::new(title).size(10.5).color(theme::FAINT).strong()));
            for &t in tools {
                let sel = self.tool == t;
                let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 58.0), Sense::click());
                let p = ui.painter();
                if sel {
                    p.rect_filled(rect, CornerRadius::same(9), theme::CARD_HI);
                    p.rect_stroke(rect, CornerRadius::same(9), Stroke::new(1.5, theme::GOLD), egui::StrokeKind::Inside);
                } else if resp.hovered() {
                    p.rect_filled(rect, CornerRadius::same(9), theme::CARD);
                }
                let fg = if sel { theme::GOLD } else { theme::TEXT };
                let icon = Rect::from_center_size(rect.center() - Vec2::new(0.0, 8.0), Vec2::splat(28.0));
                draw_icon(p, icon, t, fg);
                p.text(rect.center() + Vec2::new(0.0, 18.0), egui::Align2::CENTER_CENTER, t.label(), egui::FontId::proportional(11.5), fg);
                if resp.on_hover_text(t.tip()).clicked() {
                    self.set_tool(t);
                }
            }
            ui.add_space(6.0);
        }
    }

    fn set_tool(&mut self, t: Tool) {
        self.drag = None;
        if t == Tool::Comb && self.tool != Tool::Comb {
            self.hair.show_cards = true;
        }
        self.tool = t;
        self.vb.key = None;
    }

    fn sel_op(&mut self, op: &str) {
        let Some(topo) = self.topo.clone() else { return };
        let before = self.sel_state();
        match op {
            "grow" => edit::grow(&topo, &mut self.sel),
            "shrink" => edit::shrink(&topo, &mut self.sel),
            "mirror" => {
                edit::mirror_selection(&topo, &mut self.sel);
                let ml: Vec<EdgeLoop> = self.loops.iter().filter_map(|l| mirror_loop(&topo, l)).collect();
                for l in ml {
                    if !self.loops.contains(&l) {
                        self.loops.push(l);
                    }
                }
            }
            "clear" => {
                self.sel.iter_mut().for_each(|w| *w = 0.0);
                self.extra.iter_mut().for_each(|e| *e = None);
                self.loops.clear();
            }
            "all" => self.sel.iter_mut().for_each(|w| *w = 1.0),
            _ => return,
        }
        self.push_undo(Snap { what: op.to_owned(), sel: Some(before), gpos: None, tex: None });
        self.sel_ver += 1;
        self.say(format!("{op}: {} triangles marked", self.marked_count()), true);
    }

    fn frame_marks(&mut self, cam: &mut Camera) {
        let Some(topo) = self.topo.clone() else { return };
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        let mut any = false;
        for (t, &w) in self.sel.iter().enumerate() {
            if w <= 0.0 {
                continue;
            }
            for &g in &topo.tri_groups[t] {
                let p = self.gpose.get(g as usize).copied().unwrap_or(topo.rest[g as usize]);
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
                any = true;
            }
        }
        for l in &self.loops {
            for &g in &l.groups {
                let p = self.gpose.get(g as usize).copied().unwrap_or(topo.rest[g as usize]);
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
                any = true;
            }
        }
        if !any {
            return;
        }
        let c = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
        let size = dist(lo, hi).max(topo.height * 0.05);
        cam.target = c;
        cam.dist = size * 0.6 / (cam.fov_y * 0.5).tan() + size * 0.5;
        cam.ortho_half = size * 0.6;
        self.vb.key = None;
    }

    // ---------------------------------------------------------------------------------------------
    // Model view: render, input, overlays
    // ---------------------------------------------------------------------------------------------

    fn render_view(&mut self, ui: &egui::Ui, d: &Doc, topo: &Topo, cam: &Camera) {
        let rect = self.rect_v;
        let ppp = ui.ctx().pixels_per_point();
        let (mut w, mut h) = ((rect.width() * ppp) as usize, (rect.height() * ppp) as usize);
        let cap = 1_300_000f32;
        let s = ((cap / (w * h).max(1) as f32).sqrt()).min(1.0);
        w = ((w as f32 * s) as usize).max(32);
        h = ((h as f32 * s) as usize).max(32);
        let outline_only = self.tool.paints() || self.tool.sculpt().is_some();
        let painting = matches!(self.drag, Some(Drag::Paint { .. }));
        let clip = self.posed_clip();
        let key = (*cam, w, h, (self.sel_ver * 1_000_003 + self.pos_ver * 1009 + self.tex_ver) ^ self.look_key(), outline_only, clip, (self.time * 1000.0) as u32, painting);
        if self.vb.key.as_ref() == Some(&key) && self.vb.tex.is_some() {
            return;
        }
        self.ensure_look(topo);
        let looks = self.look_tints(topo, cam, w, h);
        let sel = &self.sel;
        let tint = |mi: usize, ti: usize| -> Option<[u8; 3]> {
            let g = topo.gid(mi, ti);
            let wt = sel[g];
            if !outline_only && wt > 0.0 {
                return Some(requestcmd::sel_color(wt));
            }
            looks.as_ref().and_then(|l| l[g])
        };
        let hidden = &self.look.hidden;
        let any_hidden = hidden.iter().any(|&h| h);
        let keep = |mi: usize, ti: usize| !hidden[topo.gid(mi, ti)];
        let o = render::Options { cam: *cam, w, h, background: theme::VIEW_BG, filter: if painting { Filter::Texels } else { Filter::Smooth }, positions: Some(&self.pose) };
        let mut f = render::render_masked(&d.model, self.textures(d), &o, &tint, if any_hidden { Some(&keep) } else { None });
        if self.marked_count() > 0 {
            let inside: Vec<bool> = f.pick.iter().map(|&id| id >= 0 && sel[id as usize] > 0.0).collect();
            outline_px(&mut f.rgba, w, h, &inside, if outline_only { [255, 160, 40] } else { [255, 255, 255] });
        }
        let img = egui::ColorImage::from_rgba_unmultiplied([w, h], &f.rgba);
        let topt = if s < 1.0 { egui::TextureOptions::LINEAR } else { egui::TextureOptions::NEAREST };
        match &mut self.vb.tex {
            Some(t) => t.set(img, topt),
            None => self.vb.tex = Some(ui.ctx().load_texture("mark-view", img, topt)),
        }
        self.vb.pick = f.pick;
        self.vb.w = w;
        self.vb.h = h;
        self.vb.key = Some(key);
    }

    /// Screen point -> buffer pixel.
    fn to_buf(&self, p: Pos2) -> [f32; 2] {
        let r = self.rect_v;
        [(p.x - r.min.x) / r.width() * self.vb.w as f32, (p.y - r.min.y) / r.height() * self.vb.h as f32]
    }

    fn to_screen(&self, b: [f32; 2]) -> Pos2 {
        let r = self.rect_v;
        Pos2::new(r.min.x + b[0] / self.vb.w as f32 * r.width(), r.min.y + b[1] / self.vb.h as f32 * r.height())
    }

    fn buf_scale(&self) -> f32 {
        self.vb.w as f32 / self.rect_v.width().max(1.0)
    }

    fn pick_at(&self, b: [f32; 2]) -> Option<usize> {
        let (x, y) = (b[0].floor(), b[1].floor());
        if x < 0.0 || y < 0.0 || x >= self.vb.w as f32 || y >= self.vb.h as f32 {
            return None;
        }
        let id = self.vb.pick[y as usize * self.vb.w + x as usize];
        (id >= 0).then_some(id as usize)
    }

    /// The surface point under a buffer pixel: (triangle, posed point, camera depth).
    fn surface_at(&self, d: &Doc, topo: &Topo, cam: &Camera, b: [f32; 2]) -> Option<(usize, V3, f32)> {
        let t = self.pick_at(b)?;
        let (mi, ti) = topo.tri_loc[t];
        let tri = d.model.meshes[mi].tris[ti];
        let [a, bb, c] = tri.map(|v| self.pose[mi][v as usize]);
        let n = norm(cross(sub(bb, a), sub(c, a)));
        let rig = Rig::new(cam, self.vb.w, self.vb.h);
        let z = rig.depth_on_plane(b[0], b[1], a, n)?;
        Some((t, unproject(&rig, b[0], b[1], z), z))
    }

    fn input(&mut self, ui: &mut egui::Ui, d: &Doc, topo: &Topo, cam: &mut Camera, parts: Option<&Parts>) {
        let ctx = ui.ctx().clone();
        // Keys (not while typing a note).
        if !ctx.wants_keyboard_input() {
            let (esc, z, y, shz, lb, rb) = ctx.input(|i| {
                (
                    i.key_pressed(Key::Escape),
                    i.modifiers.command && !i.modifiers.shift && i.key_pressed(Key::Z),
                    i.modifiers.command && i.key_pressed(Key::Y),
                    i.modifiers.command && i.modifiers.shift && i.key_pressed(Key::Z),
                    i.key_pressed(Key::OpenBracket),
                    i.key_pressed(Key::CloseBracket),
                )
            });
            if esc {
                if self.drag.is_some() {
                    self.cancel_drag();
                } else {
                    self.set_tool(Tool::Turn);
                }
            }
            if z && self.drag.is_none() {
                self.undo();
            }
            if (y || shz) && self.drag.is_none() {
                self.redo();
            }
            if lb {
                self.radius = (self.radius / 1.2).max(3.0);
            }
            if rb {
                self.radius = (self.radius * 1.2).min(220.0);
            }
        }
        let rv = ui.interact(self.rect_v, egui::Id::new("mark-view"), Sense::click_and_drag());
        let tex_area = Rect::from_min_max(self.rect_t.min + Vec2::new(0.0, 40.0), self.rect_t.max);
        let rt = ui.interact(tex_area, egui::Id::new("mark-tex"), Sense::click_and_drag());
        let (pos, pressed, released, down, mods, delta, scroll) = ctx.input(|i| {
            (
                i.pointer.hover_pos(),
                [i.pointer.button_pressed(PointerButton::Primary), i.pointer.button_pressed(PointerButton::Secondary), i.pointer.button_pressed(PointerButton::Middle)],
                [i.pointer.button_released(PointerButton::Primary), i.pointer.button_released(PointerButton::Secondary), i.pointer.button_released(PointerButton::Middle)],
                [i.pointer.button_down(PointerButton::Primary), i.pointer.button_down(PointerButton::Secondary), i.pointer.button_down(PointerButton::Middle)],
                Mods { ctrl: i.modifiers.command, shift: i.modifiers.shift, alt: i.modifiers.alt },
                i.pointer.delta(),
                i.smooth_scroll_delta.y,
            )
        });
        let place = pos.and_then(|p| {
            if self.rect_v.contains(p) && rv.hovered() {
                Some(Where::Model)
            } else if tex_area.contains(p) && rt.hovered() {
                Some(Where::Texture)
            } else {
                None
            }
        });
        // Camera and texture navigation.
        if scroll != 0.0 {
            match place {
                Some(Where::Model) => cam.zoom((1.0 - scroll * 0.002).clamp(0.5, 2.0)),
                Some(Where::Texture) => {
                    let p = pos.unwrap();
                    let before = self.screen_to_uv(p);
                    self.tex_zoom = (self.tex_zoom * (1.0 + scroll * 0.002)).clamp(1.0, 40.0);
                    let after = self.uv_to_screen(before);
                    self.tex_pan += p - after;
                }
                None => {}
            }
        }
        let turn_tool = self.tool == Tool::Turn;
        if self.cam_drag.is_none() && self.drag.is_none() {
            if let Some(pl) = place {
                if pressed[1] {
                    self.cam_drag = Some(PointerButton::Secondary);
                } else if pressed[2] {
                    self.cam_drag = Some(PointerButton::Middle);
                } else if pressed[0] && (turn_tool || (pl == Where::Texture && !self.tool_works_on_texture())) {
                    self.cam_drag = Some(PointerButton::Primary);
                }
                if self.cam_drag.is_some() {
                    self.cam_on_model = pl == Where::Model;
                }
            }
        }
        if let Some(b) = self.cam_drag {
            let on_model = self.cam_on_model;
            let idx = match b {
                PointerButton::Primary => 0,
                PointerButton::Secondary => 1,
                _ => 2,
            };
            if down[idx] {
                if on_model {
                    if b == PointerButton::Middle {
                        let rig = Rig::new(cam, self.vb.w, self.vb.h);
                        let pix = rig.pixel_size(cam.dist) * self.buf_scale();
                        cam.target = add(cam.target, add(scale(rig.r, -delta.x * pix), scale(rig.u, delta.y * pix)));
                    } else {
                        cam.yaw -= delta.x * 0.01;
                        cam.pitch = (cam.pitch + delta.y * 0.01).clamp(-1.4, 1.4);
                    }
                } else {
                    self.tex_pan += delta;
                }
            }
            if released[idx] || !down[idx] {
                self.cam_drag = None;
            }
            return;
        }
        // Tool strokes.
        if pressed[0] {
            if let (Some(pl), Some(p)) = (place, pos) {
                self.event(Ev::Press(pl, p, mods), d, topo, cam, parts);
            }
        } else if self.drag.is_some() && !self.scripted {
            // A scripted stroke has no real button down; only live drags end here.
            if let Some(p) = pos {
                if delta != Vec2::ZERO {
                    self.event(Ev::Move(p), d, topo, cam, parts);
                }
            }
            if released[0] || !down[0] {
                self.event(Ev::Release, d, topo, cam, parts);
            }
        }
    }

    fn tool_works_on_texture(&self) -> bool {
        matches!(self.tool, Tool::Part | Tool::Brush | Tool::Lasso | Tool::Paint | Tool::Clone | Tool::Smudge)
    }

    fn cancel_drag(&mut self) {
        match self.drag.take() {
            Some(Drag::Select { before, .. }) | Some(Drag::Lasso { before, .. }) => {
                self.sel = before.sel;
                self.extra = before.extra;
                self.loops = before.loops;
                self.sel_ver += 1;
            }
            Some(Drag::Sculpt { before, .. }) => {
                self.gpos = before;
                self.pos_ver += 1;
            }
            Some(Drag::Paint { image, before, .. }) => {
                if let Some(Some(t)) = self.textures.get_mut(image) {
                    t.rgba = before;
                    self.tex_ver += 1;
                }
            }
            Some(Drag::Comb { .. }) => {}
            None => {}
        }
        self.say("Stroke cancelled".into(), true);
    }

    fn event(&mut self, ev: Ev, d: &Doc, topo: &Topo, cam: &mut Camera, parts: Option<&Parts>) {
        match ev {
            Ev::Press(place, p, mods) => self.press(place, p, mods, d, topo, cam, parts),
            Ev::Move(p) => self.stroke_to(p, d, topo, cam),
            Ev::Release => self.release(d, topo),
        }
    }

    fn press(&mut self, place: Where, p: Pos2, mods: Mods, d: &Doc, topo: &Topo, cam: &mut Camera, parts: Option<&Parts>) {
        let _ = parts;
        if mods.alt && self.tool == Tool::Clone {
            self.clone_src = Some((place, p));
            self.say("Clone source set".into(), true);
            return;
        }
        match self.tool {
            Tool::Turn => {}
            Tool::Part => self.click_part(place, p, mods, d, topo),
            Tool::Edge => {
                if place == Where::Model {
                    self.click_edge(p, mods, topo, cam);
                } else {
                    self.say("Edges are picked on the model".into(), false);
                }
            }
            Tool::Probe => {
                if place == Where::Model {
                    let b = self.to_buf(p);
                    self.probe_at(b, cam, topo);
                } else {
                    self.say("Look through works on the model".into(), false);
                }
            }
            Tool::Comb => self.comb_press(place, p, mods, d, topo, cam),
            Tool::Brush => {
                self.drag = Some(Drag::Select { before: self.sel_state(), erase: mods.ctrl, place, last: None });
                self.stroke_to(p, d, topo, cam);
            }
            Tool::Lasso => self.drag = Some(Drag::Lasso { before: self.sel_state(), erase: mods.ctrl, place, pts: vec![p] }),
            t if t.sculpt().is_some() => {
                if place != Where::Model {
                    self.say("Shape brushes work on the model".into(), false);
                    return;
                }
                let kind = t.sculpt().unwrap();
                let before = self.gpos.clone();
                let mut grab = None;
                if kind == Sculpt::Grab {
                    let b = self.to_buf(p);
                    if let Some((_, c, z)) = self.surface_at(d, topo, cam, b) {
                        let rig = Rig::new(cam, self.vb.w, self.vb.h);
                        let r = self.radius * self.buf_scale() * rig.pixel_size(z);
                        let allow = self.allow(topo);
                        grab = Some((edit::grab_begin(topo, &self.gpos, c, r, &allow, self.sym), z, p));
                    } else {
                        return;
                    }
                }
                self.drag = Some(Drag::Sculpt { kind, before, grab, last: None });
                if kind != Sculpt::Grab {
                    self.stroke_to(p, d, topo, cam);
                }
            }
            _ => {
                // Paint, Clone, Smudge.
                let image = match place {
                    Where::Texture => self.image,
                    Where::Model => match self.pick_at(self.to_buf(p)).and_then(|t| d.model.meshes[topo.tri_loc[t].0].image) {
                        Some(i) => i,
                        None => return,
                    },
                };
                if self.tool == Tool::Clone && self.clone_src.is_none_or(|s| s.0 != place) {
                    self.say("Alt+click first to choose where Clone copies from (in the same view)".into(), false);
                    return;
                }
                self.ensure_textures(d);
                let Some(Some(t)) = self.textures.get(image) else { return };
                let before = t.rgba.clone();
                let clone_off = self.clone_src.map(|s| s.1 - p);
                let clean = (place == Where::Model && matches!(self.tool, Tool::Clone | Tool::Smudge)).then(|| {
                    let o = render::Options { cam: *cam, w: self.vb.w, h: self.vb.h, background: theme::VIEW_BG, filter: Filter::Texels, positions: Some(&self.pose) };
                    let f = render::render(&d.model, self.textures(d), &o, &|_, _| None);
                    (f.w, f.h, f.rgba)
                });
                self.drag = Some(Drag::Paint { place, image, before, bbox: [usize::MAX, usize::MAX, 0, 0], stroke_w: HashMap::new(), last: None, clone_off, clean });
                self.stroke_to(p, d, topo, cam);
            }
        }
    }

    fn allow(&self, topo: &Topo) -> Vec<f32> {
        let mut a = if self.marked_count() > 0 { edit::group_weights(topo, &self.sel) } else { vec![1.0; topo.groups()] };
        self.look_allow(topo, &mut a);
        a
    }

    fn ensure_textures(&mut self, d: &Doc) {
        if !self.painted {
            self.textures = d.textures.iter().map(|t| t.as_ref().map(requestcmd::copy_tex)).collect();
            self.painted = true;
        }
    }

    fn release(&mut self, d: &Doc, topo: &Topo) {
        let Some(drag) = self.drag.take() else { return };
        match drag {
            Drag::Comb { pts, .. } => self.comb_release(pts),
            Drag::Select { before, .. } => {
                self.push_undo(Snap { what: "mark".into(), sel: Some(before), gpos: None, tex: None });
                self.say(format!("{} triangles marked", self.marked_count()), true);
            }
            Drag::Lasso { before, erase, place, pts } => {
                self.apply_lasso(&pts, erase, place, d, topo);
                self.push_undo(Snap { what: "lasso".into(), sel: Some(before), gpos: None, tex: None });
                self.sel_ver += 1;
                self.say(format!("Lasso: {} triangles marked", self.marked_count()), true);
            }
            Drag::Sculpt { kind, before, .. } => {
                if before != self.gpos {
                    self.push_undo(Snap { what: format!("{kind:?}").to_lowercase(), sel: None, gpos: Some(before), tex: None });
                    let (n, mx) = edit::moved(topo, &self.gpos);
                    self.say(format!("{kind:?}: {n} places moved from the original, up to {:.2}% of his height", mx / topo.height * 100.0), true);
                } else {
                    // Low-poly models have few corners; a small brush can miss them all.
                    self.say("Nothing moved: no corner of the model was inside the brush. Make the brush bigger (] key) and go over a corner.".into(), false);
                }
            }
            Drag::Paint { image, before, bbox, .. } => {
                if bbox[0] != usize::MAX {
                    let Some(Some(t)) = self.textures.get_mut(image) else { return };
                    let b = [bbox[0], bbox[1], bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1];
                    let old = crop(&before, t.w, b);
                    // Rebuild the mipmaps so the smooth view shows the new paint.
                    let nt = Texture::new(t.w, t.h, std::mem::take(&mut t.rgba));
                    *t = nt;
                    self.push_undo(Snap { what: self.tool.label().to_lowercase(), sel: None, gpos: None, tex: Some((image, b, old)) });
                    self.tex_ver += 1;
                    self.say(format!("{}: {} x {} texels changed on {}", self.tool.label(), b[2], b[3], d.model.images[image].name.trim_end_matches(".png")), true);
                }
            }
        }
    }

    fn stroke_to(&mut self, p: Pos2, d: &Doc, topo: &Topo, cam: &Camera) {
        let spacing = (self.radius * 0.25).max(1.5);
        let last = match &self.drag {
            Some(Drag::Select { last, .. }) | Some(Drag::Sculpt { last, .. }) | Some(Drag::Paint { last, .. }) => *last,
            Some(Drag::Lasso { .. }) => {
                if let Some(Drag::Lasso { pts, .. }) = &mut self.drag {
                    if pts.last().is_none_or(|q| q.distance(p) > 2.0) {
                        pts.push(p);
                    }
                }
                return;
            }
            Some(Drag::Comb { .. }) => {
                self.comb_move(p, d, topo, cam);
                return;
            }
            None => return,
        };
        // Grab follows the pointer directly.
        if let Some(Drag::Sculpt { grab: Some((g, z, start)), .. }) = &self.drag {
            let rig = Rig::new(cam, self.vb.w, self.vb.h);
            let pix = rig.pixel_size(*z) * self.buf_scale();
            let dv = p - *start;
            let delta = add(scale(rig.r, dv.x * pix), scale(rig.u, -dv.y * pix));
            let g = g.clone();
            edit::grab_apply(&g, &mut self.gpos, delta);
            self.pos_ver += 1;
            self.set_last(p);
            return;
        }
        let pts: Vec<Pos2> = match last {
            None => vec![p],
            Some(l) => {
                let n = (l.distance(p) / spacing).floor() as usize;
                (1..=n).map(|k| l + (p - l) * (k as f32 * spacing / l.distance(p))).collect()
            }
        };
        if pts.is_empty() {
            return;
        }
        for q in &pts {
            self.dab(*q, d, topo, cam);
        }
        self.set_last(*pts.last().unwrap());
    }

    fn set_last(&mut self, p: Pos2) {
        match &mut self.drag {
            Some(Drag::Select { last, .. }) | Some(Drag::Sculpt { last, .. }) | Some(Drag::Paint { last, .. }) => *last = Some(p),
            _ => {}
        }
    }

    fn dab(&mut self, p: Pos2, d: &Doc, topo: &Topo, cam: &Camera) {
        match &self.drag {
            Some(Drag::Select { erase, place, .. }) => {
                let (erase, place) = (*erase, *place);
                match place {
                    Where::Model => self.select_dab_model(p, erase, topo),
                    Where::Texture => self.select_dab_texture(p, erase, d, topo),
                }
                self.sel_ver += 1;
            }
            Some(Drag::Sculpt { kind, .. }) => {
                let kind = *kind;
                let b = self.to_buf(p);
                let Some((_, c, z)) = self.surface_at(d, topo, cam, b) else { return };
                let rig = Rig::new(cam, self.vb.w, self.vb.h);
                let r = self.radius * self.buf_scale() * rig.pixel_size(z);
                let allow = self.allow(topo);
                edit::dab(topo, &mut self.gpos, kind, c, r, self.strength * 0.5, &allow, self.sym);
                self.pos_ver += 1;
                // The pick buffer moves with the surface; refresh positions for the next dab.
                self.pose_key = None;
                self.update_pose(d, topo);
            }
            Some(Drag::Paint { place, .. }) => {
                let place = *place;
                match place {
                    Where::Texture => self.paint_dab_texture(p),
                    Where::Model => self.paint_dab_model(p, d, topo, cam),
                }
            }
            _ => {}
        }
    }

    fn select_dab_model(&mut self, p: Pos2, erase: bool, topo: &Topo) {
        let b = self.to_buf(p);
        let r = self.radius * self.buf_scale();
        let (w, h) = (self.vb.w as i64, self.vb.h as i64);
        let mut hit: HashMap<usize, f32> = HashMap::new();
        let (x0, x1) = (((b[0] - r).floor() as i64).max(0), ((b[0] + r).ceil() as i64).min(w - 1));
        let (y0, y1) = (((b[1] - r).floor() as i64).max(0), ((b[1] + r).ceil() as i64).min(h - 1));
        for y in y0..=y1 {
            for x in x0..=x1 {
                let dd = ((x as f32 + 0.5 - b[0]).powi(2) + (y as f32 + 0.5 - b[1]).powi(2)).sqrt();
                let f = edit::falloff(dd, r);
                if f <= 0.0 {
                    continue;
                }
                let id = self.vb.pick[(y * w + x) as usize];
                if id >= 0 {
                    let e = hit.entry(id as usize).or_insert(0.0);
                    *e = e.max(f);
                }
            }
        }
        for (t, f) in hit {
            let f = (f * 1.6 * self.strength.max(0.35)).min(1.0);
            let mut apply = |t: usize| {
                if erase {
                    self.sel[t] *= 1.0 - f;
                    if self.sel[t] < 0.02 {
                        self.sel[t] = 0.0;
                    }
                } else {
                    self.sel[t] = self.sel[t].max(f);
                }
            };
            apply(t);
            if self.sym && topo.mirror_tri[t] != NONE {
                apply(topo.mirror_tri[t] as usize);
            }
        }
    }

    fn extra_mask(&mut self, d: &Doc, image: usize) -> Option<&mut (usize, usize, Vec<u8>)> {
        let t = d.textures.get(image)?.as_ref()?;
        let s = t.w.max(t.h);
        let k = (1024.0 / s as f32).min(1.0);
        let (w, h) = (((t.w as f32 * k) as usize).max(1), ((t.h as f32 * k) as usize).max(1));
        let e = self.extra.get_mut(image)?;
        if e.is_none() {
            *e = Some((w, h, vec![0u8; w * h]));
        }
        e.as_mut()
    }

    /// Triangles of meshes on `image` with their UV corners.
    fn uv_tris<'a>(d: &'a Doc, topo: &'a Topo, image: usize) -> impl Iterator<Item = (usize, [[f32; 2]; 3])> + 'a {
        d.model.meshes.iter().enumerate().filter(move |(_, m)| m.image == Some(image)).flat_map(move |(mi, m)| m.tris.iter().enumerate().map(move |(ti, t)| (topo.gid(mi, ti), t.map(|v| m.uv[v as usize]))))
    }

    fn select_dab_texture(&mut self, p: Pos2, erase: bool, d: &Doc, topo: &Topo) {
        let uv = self.screen_to_uv(p);
        let r_uv = self.radius / self.tex_rect().width().max(1.0);
        let strength = (1.6 * self.strength.max(0.35)).min(1.0);
        let image = self.image;
        if let Some((w, h, m)) = self.extra_mask(d, image) {
            let (w, h) = (*w, *h);
            let (cx, cy, r) = (uv[0] * w as f32, uv[1] * h as f32, r_uv * w as f32);
            for y in ((cy - r).floor().max(0.0) as usize)..((cy + r).ceil().max(0.0) as usize).min(h) {
                for x in ((cx - r).floor().max(0.0) as usize)..((cx + r).ceil().max(0.0) as usize).min(w) {
                    let f = edit::falloff(((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt(), r) * strength;
                    let v = &mut m[y * w + x];
                    if erase {
                        *v = (*v as f32 * (1.0 - f)) as u8;
                    } else {
                        *v = (*v).max((f.min(1.0) * 255.0) as u8);
                    }
                }
            }
        }
        let hits: Vec<(usize, f32)> = Self::uv_tris(d, topo, image).filter_map(|(g, t)| {
            let dd = tri_dist2(uv, t);
            let f = edit::falloff(dd, r_uv);
            (f > 0.0).then_some((g, (f * strength).min(1.0)))
        }).collect();
        for (t, f) in hits {
            let mut apply = |t: usize| {
                if erase {
                    self.sel[t] *= 1.0 - f;
                    if self.sel[t] < 0.02 {
                        self.sel[t] = 0.0;
                    }
                } else {
                    self.sel[t] = self.sel[t].max(f);
                }
            };
            apply(t);
            if self.sym && topo.mirror_tri[t] != NONE {
                apply(topo.mirror_tri[t] as usize);
            }
        }
    }

    fn apply_lasso(&mut self, pts: &[Pos2], erase: bool, place: Where, d: &Doc, topo: &Topo) {
        if pts.len() < 3 {
            return;
        }
        let set = |s: &mut Vec<f32>, t: usize| s[t] = if erase { 0.0 } else { 1.0 };
        match place {
            Where::Model => {
                let poly: Vec<Pos2> = pts.iter().map(|&p| {
                    let b = self.to_buf(p);
                    Pos2::new(b[0], b[1])
                }).collect();
                let mut hit = vec![false; self.sel.len()];
                let w = self.vb.w;
                fill_poly(&poly, self.vb.w, self.vb.h, |x, y| {
                    let id = self.vb.pick[y * w + x];
                    if id >= 0 {
                        hit[id as usize] = true;
                    }
                });
                for t in 0..hit.len() {
                    if hit[t] {
                        set(&mut self.sel, t);
                        if self.sym && topo.mirror_tri[t] != NONE {
                            set(&mut self.sel, topo.mirror_tri[t] as usize);
                        }
                    }
                }
            }
            Where::Texture => {
                let uvp: Vec<Pos2> = pts.iter().map(|&p| {
                    let u = self.screen_to_uv(p);
                    Pos2::new(u[0], u[1])
                }).collect();
                let image = self.image;
                let hits: Vec<usize> = Self::uv_tris(d, topo, image).filter(|(_, t)| {
                    let c = Pos2::new((t[0][0] + t[1][0] + t[2][0]) / 3.0, (t[0][1] + t[1][1] + t[2][1]) / 3.0);
                    point_in_poly(c, &uvp)
                }).map(|(g, _)| g).collect();
                for t in hits {
                    set(&mut self.sel, t);
                    if self.sym && topo.mirror_tri[t] != NONE {
                        set(&mut self.sel, topo.mirror_tri[t] as usize);
                    }
                }
                if let Some((w, h, m)) = self.extra_mask(d, image) {
                    let (w, h) = (*w, *h);
                    let poly: Vec<Pos2> = uvp.iter().map(|p| Pos2::new(p.x * w as f32, p.y * h as f32)).collect();
                    fill_poly(&poly, w, h, |x, y| m[y * w + x] = if erase { 0 } else { 255 });
                }
            }
        }
    }

    fn click_part(&mut self, place: Where, p: Pos2, mods: Mods, d: &Doc, topo: &Topo) {
        let t = match place {
            Where::Model => self.pick_at(self.to_buf(p)),
            Where::Texture => {
                let uv = self.screen_to_uv(p);
                Self::uv_tris(d, topo, self.image).find(|(_, t)| tri_dist2(uv, *t) == 0.0).map(|(g, _)| g)
            }
        };
        let Some(t) = t else { return };
        let (mi, ti) = topo.tri_loc[t];
        let part = d.tri_parts.get(mi).and_then(|v| v.get(ti)).copied().unwrap_or(0);
        let before = self.sel_state();
        let v = if mods.ctrl { 0.0 } else { 1.0 };
        let mut n = 0;
        for g in 0..self.sel.len() {
            let (m2, t2) = topo.tri_loc[g];
            let same = if part == 0 { g == t } else { d.tri_parts.get(m2).and_then(|x| x.get(t2)).copied().unwrap_or(0) == part };
            if same {
                self.sel[g] = v;
                n += 1;
                if self.sym && topo.mirror_tri[g] != NONE {
                    self.sel[topo.mirror_tri[g] as usize] = v;
                }
            }
        }
        self.push_undo(Snap { what: "part".into(), sel: Some(before), gpos: None, tex: None });
        self.sel_ver += 1;
        self.say(format!("{} part {} ({n} triangles); {} marked", if mods.ctrl { "Removed" } else { "Added" }, part, self.marked_count()), true);
    }

    fn click_edge(&mut self, p: Pos2, mods: Mods, topo: &Topo, cam: &Camera) {
        let rig = Rig::new(cam, self.vb.w, self.vb.h);
        let scr = |g: u32| -> Pos2 {
            let q = rig.project(self.gpose[g as usize]);
            self.to_screen([q[0], q[1]])
        };
        if mods.ctrl {
            let best = self.loops.iter().enumerate().map(|(i, l)| {
                let n = l.groups.len();
                let dmin = (0..n.saturating_sub(1)).map(|k| seg_dist(p, scr(l.groups[k]), scr(l.groups[k + 1]))).fold(f32::MAX, f32::min);
                (i, if n == 1 { scr(l.groups[0]).distance(p) } else { dmin })
            }).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            if let Some((i, dd)) = best {
                if dd < 24.0 {
                    let before = self.sel_state();
                    self.loops.remove(i);
                    self.push_undo(Snap { what: "remove edge".into(), sel: Some(before), gpos: None, tex: None });
                    self.sel_ver += 1;
                    self.say("Edge removed".into(), true);
                }
            }
            return;
        }
        let Some(t) = self.pick_at(self.to_buf(p)) else { return };
        let g3 = topo.tri_groups[t];
        let before = self.sel_state();
        if mods.shift {
            let g = *g3.iter().min_by(|a, b| scr(**a).distance(p).partial_cmp(&scr(**b).distance(p)).unwrap()).unwrap();
            match self.loops.last_mut() {
                Some(l) if !l.closed => {
                    let from = *l.groups.last().unwrap();
                    if from == g {
                        return;
                    }
                    match edit::shortest_path(topo, &self.gpos, from, g) {
                        Some(path) => {
                            l.groups.extend(path.into_iter().skip(1));
                            l.border = l.groups.windows(2).all(|w| topo.is_border(w[0], w[1]));
                        }
                        None => {
                            self.say("No path along the surface to that point".into(), false);
                            return;
                        }
                    }
                }
                _ => self.loops.push(EdgeLoop { groups: vec![g], closed: false, border: false }),
            }
        } else {
            // Nearest edge of the clicked triangle.
            let edges = [(g3[0], g3[1]), (g3[1], g3[2]), (g3[2], g3[0])];
            let (a, b) = *edges.iter().min_by(|x, y| seg_dist(p, scr(x.0), scr(x.1)).partial_cmp(&seg_dist(p, scr(y.0), scr(y.1))).unwrap()).unwrap();
            let l = edit::follow_loop(topo, &self.gpos, a, b);
            if self.sym {
                if let Some(m) = mirror_loop(topo, &l) {
                    if m != l && !self.loops.contains(&m) {
                        self.loops.push(m);
                    }
                }
            }
            self.loops.push(l);
        }
        self.push_undo(Snap { what: "edge".into(), sel: Some(before), gpos: None, tex: None });
        self.sel_ver += 1;
        let l = self.loops.last().unwrap();
        self.say(format!("Edge: {} points{}{}", l.groups.len(), if l.closed { ", closed loop" } else { "" }, if l.border { ", along an open edge" } else { "" }), true);
    }

    fn paint_px(&mut self, image: usize, idx: usize, w: f32, color: [f32; 3]) {
        let Some(Drag::Paint { before, bbox, stroke_w, .. }) = &mut self.drag else { return };
        let Some(Some(t)) = self.textures.get_mut(image) else { return };
        let e = stroke_w.entry(idx).or_insert(0.0);
        if w <= *e && self.tool == Tool::Paint {
            return;
        }
        *e = e.max(w);
        let o = idx * 4;
        let base = if self.tool == Tool::Paint { [before[o], before[o + 1], before[o + 2], before[o + 3]] } else { [t.rgba[o], t.rgba[o + 1], t.rgba[o + 2], t.rgba[o + 3]] };
        let a = if self.tool == Tool::Paint { *e } else { w };
        let px = lerp_px(base, color, a);
        t.rgba[o..o + 4].copy_from_slice(&px);
        let (x, y) = (idx % t.w, idx / t.w);
        bbox[0] = bbox[0].min(x);
        bbox[1] = bbox[1].min(y);
        bbox[2] = bbox[2].max(x);
        bbox[3] = bbox[3].max(y);
    }

    fn paint_dab_texture(&mut self, p: Pos2) {
        let image = self.image;
        let (off, smear) = match &self.drag {
            Some(Drag::Paint { clone_off, last, .. }) => (*clone_off, last.map(|l| p - l)),
            _ => return,
        };
        let Some(Some(t)) = self.textures.get(image) else { return };
        let (tw, th) = (t.w, t.h);
        let uv = self.screen_to_uv(p);
        let texel_per_pt = tw as f32 / self.tex_rect().width().max(1.0);
        let r = (self.radius * texel_per_pt).max(1.0);
        let (cx, cy) = (uv[0] * tw as f32, uv[1] * th as f32);
        let col = self.color.to_array();
        let col = [col[0] as f32, col[1] as f32, col[2] as f32];
        let (off, smear) = (off.map(|o| o * texel_per_pt), smear.map(|s| s * texel_per_pt));
        // Read first, then write, so Clone and Smudge read the colors from before this dab.
        let mut writes: Vec<(usize, f32, [f32; 3])> = Vec::new();
        for y in ((cy - r).floor().max(0.0) as usize)..((cy + r).ceil().max(0.0) as usize).min(th) {
            for x in ((cx - r).floor().max(0.0) as usize)..((cx + r).ceil().max(0.0) as usize).min(tw) {
                let f = edit::falloff(((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt(), r) * self.strength;
                if f <= 0.0 {
                    continue;
                }
                let c = match self.tool {
                    Tool::Paint => Some(col),
                    Tool::Clone => off.and_then(|o| sample_rgb(&t.rgba, tw, th, x as f32 + o.x, y as f32 + o.y)),
                    _ => match smear {
                        Some(s) if s.length() > 0.0 => sample_rgb(&t.rgba, tw, th, x as f32 - s.x, y as f32 - s.y),
                        _ => None,
                    },
                };
                if let Some(c) = c {
                    writes.push((y * tw + x, f, c));
                }
            }
        }
        for (i, f, c) in writes {
            self.paint_px(image, i, f, c);
        }
        self.tex_ver += 1;
    }

    fn paint_dab_model(&mut self, p: Pos2, d: &Doc, topo: &Topo, cam: &Camera) {
        let b = self.to_buf(p);
        let Some((_, c, z)) = self.surface_at(d, topo, cam, b) else { return };
        let rig = Rig::new(cam, self.vb.w, self.vb.h);
        let r = self.radius * self.buf_scale() * rig.pixel_size(z);
        let col = self.color.to_array();
        let col = [col[0] as f32, col[1] as f32, col[2] as f32];
        let bs = self.buf_scale();
        let (off, smear, clean) = match &mut self.drag {
            Some(Drag::Paint { clone_off, last, clean, .. }) => (clone_off.map(|o| o * bs), last.map(|l| (p - l) * bs), clean.take()),
            _ => return,
        };
        let (vw, vh) = (self.vb.w, self.vb.h);
        let visible = |gid: usize, q: V3, pick: &[i32]| -> Option<[f32; 2]> {
            let s = rig.project(q);
            if s[2] <= 0.0 {
                return None;
            }
            let (x, y) = (s[0].floor() as i64, s[1].floor() as i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (xx, yy) = (x + dx, y + dy);
                    if xx < 0 || yy < 0 || xx >= vw as i64 || yy >= vh as i64 {
                        continue;
                    }
                    let id = pick[(yy as usize) * vw + xx as usize];
                    if id >= 0 && (id as usize == gid || topo.tri_nbr[gid].contains(&(id as u32))) {
                        return Some([s[0], s[1]]);
                    }
                }
            }
            None
        };
        let mut writes: Vec<(usize, usize, f32, [f32; 3])> = Vec::new();
        let images: Vec<usize> = (0..d.model.images.len()).collect();
        let centers: Vec<(V3, bool)> = if self.sym { vec![(c, false), (topo.mirror_point(c), true)] } else { vec![(c, false)] };
        for &img in &images {
            let Some(Some(t)) = self.textures.get(img) else { continue };
            let (tw, th) = (t.w, t.h);
            for &(cc, mirrored) in &centers {
                edit::model_texels(&d.model, topo, &self.mesh_pos, img, tw, th, cc, r, |gid, x, y, q, wgt| {
                    // Seen from the camera: this side directly, the mirror side through its twin.
                    let screen = if mirrored {
                        let mt = topo.mirror_tri[gid];
                        if mt == NONE {
                            return;
                        }
                        match visible(mt as usize, topo.mirror_point(q), &self.vb.pick) {
                            Some(s) => s,
                            None => return,
                        }
                    } else {
                        match visible(gid, q, &self.vb.pick) {
                            Some(s) => s,
                            None => return,
                        }
                    };
                    let f = wgt * self.strength;
                    let color = match self.tool {
                        Tool::Paint => Some(col),
                        Tool::Clone => clean.as_ref().zip(off).and_then(|((w, h, px), o)| sample_rgb(px, *w, *h, screen[0] + o.x, screen[1] + o.y)),
                        _ => match (clean.as_ref(), smear) {
                            (Some((w, h, px)), Some(s)) if s.length() > 0.0 => sample_rgb(px, *w, *h, screen[0] - s.x * 2.0, screen[1] - s.y * 2.0),
                            _ => None,
                        },
                    };
                    if let Some(c) = color {
                        writes.push((img, y * tw + x, f, c));
                    }
                });
            }
        }
        let mut touched = vec![false; d.model.images.len()];
        for (img, i, f, c) in writes {
            self.paint_px(img, i, f, c);
            touched[img] = true;
        }
        let _ = touched;
        if let Some(Drag::Paint { clean: slot, .. }) = &mut self.drag {
            *slot = clean;
        }
        self.tex_ver += 1;
    }

    fn draw_view(&mut self, ui: &mut egui::Ui, d: &Doc, topo: &Topo, cam: &Camera, parts: Option<&Parts>) {
        let rect = self.rect_v;
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, CornerRadius::same(10), Color32::from_rgb(0x12, 0x14, 0x18));
        if let Some(t) = &self.vb.tex {
            painter.image(t.id(), rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        }
        painter.rect_stroke(rect, CornerRadius::same(10), Stroke::new(1.0, theme::LINE), egui::StrokeKind::Inside);
        // Edge loops.
        let rig = Rig::new(cam, self.vb.w, self.vb.h);
        for l in &self.loops {
            let pts: Vec<Pos2> = l.groups.iter().map(|&g| {
                let q = rig.project(self.gpose[g as usize]);
                self.to_screen([q[0], q[1]])
            }).collect();
            let mut line = pts.clone();
            if l.closed && !pts.is_empty() {
                line.push(pts[0]);
            }
            painter.add(Shape::line(line.clone(), Stroke::new(4.5, Color32::from_black_alpha(200))));
            painter.add(Shape::line(line, Stroke::new(2.2, Color32::from_rgb(80, 240, 255))));
            for q in &pts {
                painter.circle_filled(*q, 2.8, Color32::WHITE);
            }
        }
        // Lasso in progress.
        if let Some(Drag::Lasso { pts, place: Where::Model, erase, .. }) = &self.drag {
            let c = if *erase { theme::RED } else { theme::GOLD };
            let mut l = pts.clone();
            l.push(pts[0]);
            painter.extend(Shape::dashed_line(&l, Stroke::new(1.6, c), 6.0, 4.0));
        }
        // Brush ring and what is under the pointer.
        if let Some(hp) = ui.ctx().pointer_hover_pos().filter(|p| rect.contains(*p)) {
            if self.tool.round_brush() {
                let c = if ui.input(|i| i.modifiers.command) && self.tool == Tool::Brush { theme::RED } else { Color32::from_white_alpha(220) };
                painter.circle_stroke(hp, self.radius, Stroke::new(1.3, c));
                painter.circle_stroke(hp, self.radius * 0.5, Stroke::new(0.8, Color32::from_white_alpha(70)));
            }
            if self.drag.is_none() {
                if let Some(t) = self.pick_at(self.to_buf(hp)) {
                    let (mi, ti) = topo.tri_loc[t];
                    let pid = d.tri_parts.get(mi).and_then(|v| v.get(ti)).copied().unwrap_or(0);
                    let pname = if pid == 0 { String::new() } else { parts.map(|p| super::title_case(p.name(pid))).unwrap_or_default() };
                    let txt = if pname.is_empty() { format!("{} triangle {ti}", super::mesh_title(&d.model.meshes[mi].name)) } else { format!("{pname} · triangle {ti}") };
                    painter.text(rect.left_top() + Vec2::new(12.0, 12.0), egui::Align2::LEFT_TOP, txt, egui::FontId::proportional(12.5), theme::WEAK);
                }
            }
        }
        if let Some((Where::Model, s)) = self.clone_src {
            if self.tool == Tool::Clone {
                painter.circle_stroke(s, 7.0, Stroke::new(1.5, theme::TEAL));
                painter.line_segment([s - Vec2::new(10.0, 0.0), s + Vec2::new(10.0, 0.0)], Stroke::new(1.0, theme::TEAL));
                painter.line_segment([s - Vec2::new(0.0, 10.0), s + Vec2::new(0.0, 10.0)], Stroke::new(1.0, theme::TEAL));
            }
        }
        self.draw_look(&painter, topo, cam);
        self.draw_hair(&painter, d, topo, cam);
        let cap = format!("{} · {}{}", if self.posed_clip().is_some() { self.clip.clone().unwrap_or_default() } else { "Rest pose".into() }, if self.sym { "Mirror on" } else { "Mirror off" }, if !self.has_edits(topo) { "" } else if self.saved_ver == (self.pos_ver, self.tex_ver) { " · edited, saved" } else { " · edited (not saved)" });
        let g = painter.layout_no_wrap(cap, egui::FontId::proportional(12.5), theme::WEAK);
        let r = Rect::from_min_size(rect.left_bottom() + Vec2::new(10.0, -g.size().y - 16.0), g.size() + Vec2::new(14.0, 8.0));
        painter.rect_filled(r, CornerRadius::same(6), Color32::from_black_alpha(150));
        painter.galley(r.min + Vec2::new(7.0, 4.0), g, theme::WEAK);
    }

    // ---------------------------------------------------------------------------------------------
    // Texture view
    // ---------------------------------------------------------------------------------------------

    fn tex_rect(&self) -> Rect {
        let r = self.rect_t;
        let top = 40.0;
        let avail = Rect::from_min_max(r.min + Vec2::new(0.0, top), r.max);
        let s = avail.width().min(avail.height());
        let base = Rect::from_center_size(avail.center(), Vec2::splat(s));
        Rect::from_center_size(base.center() + self.tex_pan, base.size() * self.tex_zoom)
    }

    fn screen_to_uv(&self, p: Pos2) -> [f32; 2] {
        let r = self.tex_rect();
        [(p.x - r.min.x) / r.width(), (p.y - r.min.y) / r.height()]
    }

    fn uv_to_screen(&self, uv: [f32; 2]) -> Pos2 {
        let r = self.tex_rect();
        Pos2::new(r.min.x + uv[0] * r.width(), r.min.y + uv[1] * r.height())
    }

    fn texture_view(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, d: &Doc, topo: &Topo) {
        let rect = self.rect_t;
        // Sheet picker.
        let sheets: Vec<usize> = (0..d.model.images.len()).filter(|&i| d.model.meshes.iter().any(|m| m.image == Some(i))).collect();
        let bar = Rect::from_min_size(rect.min, Vec2::new(rect.width(), 32.0));
        ui.scope_builder(egui::UiBuilder::new().max_rect(bar), |ui| {
            ui.horizontal(|ui| {
                for &i in &sheets {
                    let name = super::texture_title(d.model.images[i].name.trim_end_matches(".png"));
                    if ui.selectable_label(self.image == i, name).clicked() {
                        self.image = i;
                    }
                }
                if ui.small_button("Fit").on_hover_text("Show the whole sheet").clicked() {
                    self.tex_zoom = 1.0;
                    self.tex_pan = Vec2::ZERO;
                }
            });
        });
        if self.sheet_tex.len() != d.model.images.len() {
            self.sheet_tex = (0..d.model.images.len()).map(|_| None).collect();
            self.sheet_ver = vec![0; d.model.images.len()];
        }
        let image = self.image;
        let texs = if self.painted { &self.textures } else { &d.textures };
        if let Some(Some(t)) = texs.get(image) {
            // Upload the sheet (again after painting; only the changed box while a stroke runs).
            let stale = self.sheet_ver[image] != self.tex_ver || self.sheet_tex[image].is_none();
            if stale {
                let partial = match (&self.drag, &self.sheet_tex[image]) {
                    (Some(Drag::Paint { bbox, image: di, .. }), Some(_)) if *di == image && bbox[0] != usize::MAX => Some(*bbox),
                    _ => None,
                };
                let opts = egui::TextureOptions { magnification: egui::TextureFilter::Nearest, minification: egui::TextureFilter::Linear, ..Default::default() };
                match (partial, &mut self.sheet_tex[image]) {
                    (Some(b), Some(h)) => {
                        let bb = [b[0], b[1], b[2] - b[0] + 1, b[3] - b[1] + 1];
                        let px = crop(&t.rgba, t.w, bb);
                        h.set_partial([bb[0], bb[1]], egui::ColorImage::from_rgba_unmultiplied([bb[2], bb[3]], &px), opts);
                    }
                    (_, slot) => {
                        let img = egui::ColorImage::from_rgba_unmultiplied([t.w, t.h], &t.rgba);
                        match slot {
                            Some(h) => h.set(img, opts),
                            None => *slot = Some(ctx.load_texture(format!("mark-sheet-{image}"), img, opts)),
                        }
                    }
                }
                self.sheet_ver[image] = self.tex_ver;
            }
        }
        // Marks overlay at up to 1024 px.
        let over_stale = self.overlay.as_ref().is_none_or(|o| o.0 != image || o.1 != self.sel_ver);
        if over_stale {
            if let Some(Some(t)) = d.textures.get(image) {
                let k = (1024.0 / t.w.max(t.h) as f32).min(1.0);
                let (w, h) = (((t.w as f32 * k) as usize).max(1), ((t.h as f32 * k) as usize).max(1));
                let mut m = edit::selection_mask(&d.model, topo, &self.sel, image, w, h);
                if let Some(Some((ew, eh, ev))) = self.extra.get(image) {
                    for y in 0..h {
                        for x in 0..w {
                            let v = ev[(y * eh / h) * ew + x * ew / w] as f32 / 255.0;
                            m[y * w + x] = m[y * w + x].max(v);
                        }
                    }
                }
                let mut rgba = vec![0u8; w * h * 4];
                for i in 0..w * h {
                    let v = m[i];
                    if v > 0.0 {
                        let edge = {
                            let (x, y) = (i % w, i / w);
                            (x == 0 || m[i - 1] <= 0.0) || (x + 1 == w || m[i + 1] <= 0.0) || (y == 0 || m[i - w] <= 0.0) || (y + 1 == h || m[i + w] <= 0.0)
                        };
                        let c = if edge { [255, 255, 255, 230] } else { [255, 140, 20, (60.0 + 110.0 * v) as u8] };
                        rgba[i * 4..i * 4 + 4].copy_from_slice(&c);
                    }
                }
                let img = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
                let opts = egui::TextureOptions::NEAREST;
                match &mut self.overlay {
                    Some(o) => {
                        o.2.set(img, opts);
                        o.0 = image;
                        o.1 = self.sel_ver;
                    }
                    None => self.overlay = Some((image, self.sel_ver, ctx.load_texture("mark-overlay", img, opts))),
                }
            }
        }
        let view = Rect::from_min_max(rect.min + Vec2::new(0.0, 40.0), rect.max);
        let painter = ui.painter_at(view);
        painter.rect_filled(view, CornerRadius::same(10), Color32::from_rgb(0x10, 0x11, 0x14));
        let tr = self.tex_rect();
        if let Some(Some(h)) = self.sheet_tex.get(image) {
            painter.image(h.id(), tr, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        }
        if let Some(o) = &self.overlay {
            painter.image(o.2.id(), tr, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        }
        // Edge loops in UV: each segment where a triangle on this sheet holds both ends.
        for l in &self.loops {
            let n = l.groups.len();
            let segs = if l.closed { n } else { n.saturating_sub(1) };
            for k in 0..segs {
                let (a, b) = (l.groups[k], l.groups[(k + 1) % n]);
                for &t in &topo.group_tris[a as usize] {
                    let tg = topo.tri_groups[t as usize];
                    if !tg.contains(&b) {
                        continue;
                    }
                    let (mi, ti) = topo.tri_loc[t as usize];
                    let m = &d.model.meshes[mi];
                    if m.image != Some(image) {
                        continue;
                    }
                    let tri = m.tris[ti];
                    let ia = tg.iter().position(|&g| g == a).unwrap();
                    let ib = tg.iter().position(|&g| g == b).unwrap();
                    let (ua, ub) = (m.uv[tri[ia] as usize], m.uv[tri[ib] as usize]);
                    let (pa, pb) = (self.uv_to_screen(ua), self.uv_to_screen(ub));
                    painter.line_segment([pa, pb], Stroke::new(3.0, Color32::from_black_alpha(200)));
                    painter.line_segment([pa, pb], Stroke::new(1.6, Color32::from_rgb(80, 240, 255)));
                }
            }
        }
        if let Some(Drag::Lasso { pts, place: Where::Texture, erase, .. }) = &self.drag {
            let c = if *erase { theme::RED } else { theme::GOLD };
            let mut l = pts.clone();
            l.push(pts[0]);
            painter.extend(Shape::dashed_line(&l, Stroke::new(1.6, c), 6.0, 4.0));
        }
        if let Some(hp) = ctx.pointer_hover_pos().filter(|p| view.contains(*p)) {
            if self.tool_works_on_texture() && self.tool.round_brush() {
                painter.circle_stroke(hp, self.radius, Stroke::new(1.3, Color32::from_white_alpha(220)));
            }
            let uv = self.screen_to_uv(hp);
            if (0.0..=1.0).contains(&uv[0]) && (0.0..=1.0).contains(&uv[1]) {
                painter.text(view.left_bottom() + Vec2::new(10.0, -10.0), egui::Align2::LEFT_BOTTOM, format!("u {:.3}  v {:.3}", uv[0], uv[1]), egui::FontId::monospace(11.5), theme::WEAK);
            }
        }
        if let Some((Where::Texture, s)) = self.clone_src {
            if self.tool == Tool::Clone {
                painter.circle_stroke(s, 7.0, Stroke::new(1.5, theme::TEAL));
            }
        }
        self.draw_hair_tex(&painter, d);
        painter.rect_stroke(view, CornerRadius::same(10), Stroke::new(1.0, theme::LINE), egui::StrokeKind::Inside);
    }

    // ---------------------------------------------------------------------------------------------
    // Saving
    // ---------------------------------------------------------------------------------------------

    fn receive(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.work_rx.take() else { return };
        let mut done = false;
        while let Ok(w) = rx.try_recv() {
            match w {
                Work::Progress(s) => self.busy = Some(s),
                Work::Request(r) => {
                    self.busy = None;
                    done = true;
                    match r {
                        Ok(s) => {
                            if self.scripted {
                                println!("request saved {}", s.dir.display());
                                println!("prompt: {}", s.prompt);
                            } else {
                                ctx.copy_text(s.prompt.clone());
                            }
                            self.say(format!("Request saved ({}). {}", s.summary, if self.scripted { "Prompt printed." } else { "The prompt is on the clipboard: paste it to any AI chat." }), true);
                            self.saved_req = Some(s);
                        }
                        Err(e) => self.say(format!("Could not save the request: {e}"), false),
                    }
                }
                Work::Glb(r) => match r {
                    Ok(p) => {
                        self.say(format!("Saved {}; checking it now", p.display()), true);
                        if self.scripted {
                            println!("saved glb {}", p.display());
                        }
                        self.saved_glb = Some(p);
                        self.saved_ver = (self.pos_ver, self.tex_ver);
                    }
                    Err(e) => {
                        self.busy = None;
                        self.checks_running = false;
                        done = true;
                        self.say(format!("Could not save: {e}"), false);
                    }
                },
                Work::Checks(mut c) => {
                    if let Some((ok, text)) = self.topo.clone().and_then(|t| self.own_lock_line(&t)) {
                        c.lines.push(requestcmd::CheckLine { ok, text });
                    }
                    self.busy = None;
                    self.checks_running = false;
                    done = true;
                    if self.scripted {
                        println!("checks {} for {}", if c.passed() { "PASSED" } else { "FAILED" }, c.file.display());
                        for l in &c.lines {
                            println!("  {} {}", if l.ok { "ok  " } else { "FAIL" }, l.text);
                        }
                    }
                    self.checks = Some(c);
                }
            }
        }
        if !(done && self.busy.is_none() && !self.checks_running) {
            self.work_rx = Some(rx);
        }
        ctx.request_repaint();
    }

    fn req_input(&self, d: &Arc<Doc>, parts: Option<Arc<Parts>>, cam: &Camera, kind: &'static str) -> Option<ReqInput> {
        let topo = self.topo.clone()?;
        Some(ReqInput {
            kind,
            root: self.requests_dir.clone(),
            doc: d.clone(),
            topo,
            parts,
            note: self.note.trim().to_owned(),
            sel: self.sel.clone(),
            loops: self.loops.clone(),
            extra: self.extra.clone(),
            gpos: self.gpos.clone(),
            textures: if self.painted { self.textures.iter().map(|t| t.as_ref().map(requestcmd::copy_tex)).collect() } else { vec![] },
            cam: *cam,
            aspect: self.rect_v.width() / self.rect_v.height().max(1.0),
            clip: self.posed_clip(),
            time: if self.posed_clip().is_some() { self.time } else { 0.0 },
            symmetry: self.sym,
            after_glb: None,
            locked: self.look.locked.clone(),
            attach: self.hair_attachments(d),
        })
    }

    fn save_request(&mut self, ctx: &egui::Context, d: &Arc<Doc>, parts: Option<Arc<Parts>>, cam: &Camera, fix: bool) {
        let Some(mut input) = self.req_input(d, parts, cam, if fix { "fix-my-edit" } else { "mark" }) else { return };
        let edited = if fix { self.edited_textures(d) } else { vec![] };
        let (tx, rx) = channel();
        self.work_rx = Some(rx);
        self.busy = Some(if fix { "Saving the before and after models".into() } else { "Saving the request".into() });
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            if fix {
                match requestcmd::edited_glb(&input.doc, &input.topo, &input.gpos, &edited) {
                    Ok(b) => input.after_glb = Some(b),
                    Err(e) => {
                        let _ = tx.send(Work::Request(Err(e)));
                        ctx.request_repaint();
                        return;
                    }
                }
            }
            let _ = tx.send(Work::Request(requestcmd::save(input)));
            ctx.request_repaint();
        });
    }

    fn save_as_dialog(&mut self, ctx: &egui::Context, d: &Arc<Doc>) {
        let (_, stamp) = requestcmd::local_stamp();
        let stem = d.path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
        let name = format!("{stem}-edit-{}.glb", stamp.replace([' ', ':'], "-").replace("--", "-"));
        let dir = Path::new(requestcmd::MODDING).join("exports");
        let pick = rfd::FileDialog::new().set_title("Save the edited model as a new file").set_directory(&dir).set_file_name(&name).add_filter("Character export (GLB)", &["glb"]).save_file();
        if let Some(p) = pick {
            self.save_glb(ctx, d, p);
        }
    }

    fn save_glb(&mut self, ctx: &egui::Context, d: &Arc<Doc>, path: PathBuf) {
        if let Err(e) = requestcmd::check_new_file(&path) {
            self.say(e, false);
            return;
        }
        let Some(topo) = self.topo.clone() else { return };
        let gpos = self.gpos.clone();
        let edited = self.edited_textures(d);
        // Parts the edit touched, so the judge looks at everything else.
        let mut parts_touched: Vec<String> = Vec::new();
        if let Some(p) = self.parts_for_judge(d, &topo) {
            parts_touched = p;
        }
        let (tx, rx) = channel();
        self.work_rx = Some(rx);
        self.busy = Some("Writing the new file".into());
        self.checks_running = true;
        self.checks = None;
        let d = d.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let r = requestcmd::edited_glb(&d, &topo, &gpos, &edited).and_then(|b| std::fs::write(&path, b).map_err(|e| format!("{}: {e}", path.display())));
            match r {
                Ok(()) => {
                    let _ = tx.send(Work::Glb(Ok(path.clone())));
                    ctx.request_repaint();
                    let tx2 = tx.clone();
                    let c2 = ctx.clone();
                    let progress = move |s: String| {
                        let _ = tx2.send(Work::Progress(s));
                        c2.request_repaint();
                    };
                    let c = requestcmd::run_checks(&d.path, &path, &parts_touched, &progress);
                    let _ = tx.send(Work::Checks(c));
                }
                Err(e) => {
                    let _ = tx.send(Work::Glb(Err(e)));
                }
            }
            ctx.request_repaint();
        });
    }

    /// Kit part names under moved vertices and painted texels.
    fn parts_for_judge(&self, d: &Doc, topo: &Topo) -> Option<Vec<String>> {
        let parts = self.parts.clone()?;
        let mut ids: Vec<u8> = Vec::new();
        for (g, p) in self.gpos.iter().enumerate() {
            if *p == topo.rest[g] {
                continue;
            }
            for &t in &topo.group_tris[g] {
                let (mi, ti) = topo.tri_loc[t as usize];
                let id = d.tri_parts.get(mi).and_then(|v| v.get(ti)).copied().unwrap_or(0);
                if id != 0 && !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        if self.painted {
            for (ii, t) in self.textures.iter().enumerate() {
                let (Some(t), Some(Some(o))) = (t, d.textures.get(ii)) else { continue };
                let atlas = d.model.images[ii].name.trim_end_matches(".png").to_owned();
                for (i, (a, b)) in t.rgba.chunks_exact(4).zip(o.rgba.chunks_exact(4)).enumerate().step_by(7) {
                    if a != b {
                        let id = parts.id_at(&atlas, (i % t.w) as f32 / t.w as f32, (i / t.w) as f32 / t.h as f32);
                        if id != 0 && !ids.contains(&id) {
                            ids.push(id);
                        }
                    }
                }
            }
        }
        let mut names: Vec<String> = ids.iter().map(|&i| parts.name(i).to_owned()).collect();
        // Judge falls back to bone-group regions (leg_L, torso...) when a sheet has no kit map,
        // so name the bone groups of the edited parts too.
        for (mi, v) in d.tri_parts.iter().enumerate() {
            for (ti, id) in v.iter().enumerate() {
                if *id != 0 && ids.contains(id) {
                    let j = d.model.triangle_joint(mi, ti);
                    if let Some(b) = d.model.joint_names.get(j) {
                        let g = crate::judgecmd::bone_group(b);
                        if !names.contains(&g) {
                            names.push(g);
                        }
                    }
                }
            }
        }
        Some(names)
    }

    // ---------------------------------------------------------------------------------------------
    // Scripted runs (tests on the hidden desktop): one command per frame.
    // ---------------------------------------------------------------------------------------------

    fn run_script(&mut self, ctx: &egui::Context, d: &Arc<Doc>, topo: &Topo, cam: &mut Camera, parts: Option<&Parts>) {
        if !self.scripted {
            return;
        }
        ctx.request_repaint();
        if self.busy.is_some() || self.checks_running {
            return;
        }
        if self.script_wait > 0 {
            self.script_wait -= 1;
            return;
        }
        if let Some((ev, _)) = self.script_events.pop_front() {
            self.event(ev, d, topo, cam, parts);
            return;
        }
        let Some(line) = self.script.pop_front() else { return };
        println!("> {line}");
        let (cmd, rest) = line.split_once(' ').map(|(a, b)| (a.to_owned(), b.trim().to_owned())).unwrap_or((line.clone(), String::new()));
        let fr = |s: &str| -> Option<[f32; 2]> {
            let v: Vec<f32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (v.len() == 2).then(|| [v[0], v[1]])
        };
        let mv = |f: [f32; 2], r: Rect| Pos2::new(r.min.x + f[0] * r.width(), r.min.y + f[1] * r.height());
        let bad = |s: &str| println!("mark script: {s}");
        match cmd.as_str() {
            "tool" => match Tool::by_name(&rest) {
                Some(t) => self.set_tool(t),
                None => bad(&format!("unknown tool {rest}")),
            },
            "radius" => self.radius = rest.parse().unwrap_or(self.radius),
            "strength" => self.strength = rest.parse().unwrap_or(self.strength),
            "sym" | "mirror-x" => self.sym = matches!(rest.as_str(), "on" | "1" | "true"),
            "color" => {
                let h = rest.trim_start_matches('#');
                if let (Ok(r), Ok(g), Ok(b)) = (u8::from_str_radix(h.get(0..2).unwrap_or(""), 16), u8::from_str_radix(h.get(2..4).unwrap_or(""), 16), u8::from_str_radix(h.get(4..6).unwrap_or(""), 16)) {
                    self.color = Color32::from_rgb(r, g, b);
                }
            }
            "note" => self.note = rest.replace("\\n", "\n"),
            "grow" | "shrink" | "mirror" | "clear" | "all" => self.sel_op(&cmd),
            "undo" => self.undo(),
            "redo" => self.redo(),
            "wait" => self.script_wait = rest.parse().unwrap_or(10),
            "view" => {
                cam.yaw = match rest.as_str() {
                    "back" => BACK,
                    "side" | "left" => SIDE,
                    "right" => -SIDE,
                    _ => FRONT,
                };
            }
            "fit" => {
                let mut c = Camera::fit(&d.model, self.rect_v.width() / self.rect_v.height().max(1.0), 0.8, false);
                c.yaw = cam.yaw;
                *cam = c;
            }
            "yaw" => cam.yaw = rest.parse::<f32>().unwrap_or(180.0).to_radians(),
            "pitch" => cam.pitch = rest.parse::<f32>().unwrap_or(0.0).to_radians(),
            "zoom" => cam.zoom(1.0 / rest.parse::<f32>().unwrap_or(1.0).max(0.01)),
            "look" => {
                let v: Vec<f32> = rest.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if v.len() == 3 {
                    cam.target = [v[0], v[1], v[2]];
                }
            }
            "frame-marks" => self.frame_marks(cam),
            "texture" => {
                if let Some(i) = d.model.images.iter().position(|im| im.name.trim_end_matches(".png") == rest) {
                    self.image = i;
                }
            }
            "texzoom" => {
                let v: Vec<f32> = rest.split([' ', ',']).filter_map(|x| x.trim().parse().ok()).collect();
                if v.len() == 3 {
                    // texzoom Z U,V: zoom Z times, centered on U,V.
                    self.tex_zoom = v[0].max(1.0);
                    self.tex_pan = Vec2::ZERO;
                    let now = self.uv_to_screen([v[1], v[2]]);
                    let view_c = Rect::from_min_max(self.rect_t.min + Vec2::new(0.0, 40.0), self.rect_t.max).center();
                    self.tex_pan = view_c - now;
                }
            }
            "select-tris" => {
                // select-tris MESH ID,ID,...
                let mut it = rest.split_whitespace();
                let mesh = it.next().and_then(|m| d.model.meshes.iter().position(|x| x.name == m).or_else(|| m.parse().ok()));
                if let Some(mi) = mesh {
                    let before = self.sel_state();
                    for id in it.flat_map(|s| s.split(',')).filter_map(|s| s.parse::<usize>().ok()) {
                        if id < d.model.meshes[mi].tris.len() {
                            self.sel[topo.gid(mi, id)] = 1.0;
                        }
                    }
                    self.push_undo(Snap { what: "select".into(), sel: Some(before), gpos: None, tex: None });
                    self.sel_ver += 1;
                    self.say(format!("{} triangles marked", self.marked_count()), true);
                }
            }
            "click" | "shift-click" | "ctrl-click" | "alt-click" | "texclick" | "tex-ctrl-click" | "tex-alt-click" => {
                let on_tex = cmd.starts_with("tex");
                let mods = Mods { ctrl: cmd.contains("ctrl"), shift: cmd.contains("shift"), alt: cmd.contains("alt") };
                match fr(&rest) {
                    Some(f) => {
                        let (pl, p) = if on_tex { (Where::Texture, self.uv_to_screen(f)) } else { (Where::Model, mv(f, self.rect_v)) };
                        self.script_events.push_back((Ev::Press(pl, p, mods), pl));
                        self.script_events.push_back((Ev::Release, pl));
                    }
                    None => bad("click X,Y"),
                }
            }
            "stroke" | "ctrl-stroke" | "texstroke" | "tex-ctrl-stroke" => {
                let on_tex = cmd.starts_with("tex");
                let mods = Mods { ctrl: cmd.contains("ctrl"), shift: false, alt: false };
                let pts: Vec<[f32; 2]> = rest.split_whitespace().filter_map(fr).collect();
                if pts.len() < 2 {
                    bad("stroke needs two or more X,Y points");
                } else {
                    let pl = if on_tex { Where::Texture } else { Where::Model };
                    let to = |f: [f32; 2]| if on_tex { self.uv_to_screen(f) } else { mv(f, self.rect_v) };
                    let mut evs = vec![(Ev::Press(pl, to(pts[0]), mods), pl)];
                    // Small steps, like a hand drag: one move per frame.
                    for w in pts.windows(2) {
                        let (a, b) = (to(w[0]), to(w[1]));
                        let n = ((a.distance(b) / 6.0).ceil() as usize).max(1);
                        for k in 1..=n {
                            evs.push((Ev::Move(a + (b - a) * (k as f32 / n as f32)), pl));
                        }
                    }
                    evs.push((Ev::Release, pl));
                    self.script_events.extend(evs);
                }
            }
            "save-request" => self.save_request(ctx, d, self.parts.clone(), cam, false),
            "fix-request" => self.save_request(ctx, d, self.parts.clone(), cam, true),
            "save-glb" => self.save_glb(ctx, d, PathBuf::from(rest)),
            "print" => {
                let (n, mx) = edit::moved(topo, &self.gpos);
                println!(
                    "mark state: {} triangles marked, {} edge loops {:?}, {} places moved (max {:.5}), painted {}, undo {}, redo {}",
                    self.marked_count(),
                    self.loops.len(),
                    self.loops.iter().map(|l| (l.groups.len(), l.closed, l.border)).collect::<Vec<_>>(),
                    n,
                    mx,
                    self.painted,
                    self.undo.len(),
                    self.redo.len()
                );
            }
            other => {
                if !self.look_script(other, &rest, cam, d, topo) && !self.hair_script(other, &rest, ctx, d, topo, cam) {
                    bad(&format!("unknown command {other}"));
                }
            }
        }
    }
}

fn mirror_loop(topo: &Topo, l: &EdgeLoop) -> Option<EdgeLoop> {
    let g: Vec<u32> = l.groups.iter().map(|&g| topo.mirror_group[g as usize]).collect();
    if g.iter().any(|&x| x == NONE) {
        return None;
    }
    Some(EdgeLoop { groups: g, closed: l.closed, border: l.border })
}

fn outline_px(rgba: &mut [u8], w: usize, h: usize, inside: &[bool], c: [u8; 3]) {
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !inside[i] {
                continue;
            }
            let edge = (x == 0 || !inside[i - 1]) || (x + 1 == w || !inside[i + 1]) || (y == 0 || !inside[i - w]) || (y + 1 == h || !inside[i + w]);
            if edge {
                rgba[i * 4..i * 4 + 3].copy_from_slice(&c);
                if x + 1 < w {
                    rgba[(i + 1) * 4..(i + 1) * 4 + 3].copy_from_slice(&c);
                }
            }
        }
    }
}

/// Simple line icons for the tool column.
fn draw_icon(p: &Painter, r: Rect, t: Tool, c: Color32) {
    let s = Stroke::new(2.0, c);
    let cc = r.center();
    let w = r.width();
    let at = |x: f32, y: f32| Pos2::new(r.min.x + x * w, r.min.y + y * w);
    match t {
        Tool::Turn => {
            let pts: Vec<Pos2> = (0..=28).map(|k| {
                let a = 0.4 + k as f32 / 28.0 * 5.0;
                cc + Vec2::new(a.cos(), a.sin()) * w * 0.38
            }).collect();
            let end = *pts.last().unwrap();
            p.add(Shape::line(pts, s));
            p.add(Shape::convex_polygon(vec![end + Vec2::new(-6.0, -3.0), end + Vec2::new(5.0, -5.0), end + Vec2::new(1.0, 6.0)], c, Stroke::NONE));
        }
        Tool::Part => {
            for (x, y, f) in [(0.12, 0.12, true), (0.54, 0.12, false), (0.12, 0.54, false), (0.54, 0.54, false)] {
                let q = Rect::from_min_size(at(x, y), Vec2::splat(w * 0.34));
                if f {
                    p.rect_filled(q, CornerRadius::same(3), c);
                } else {
                    p.rect_stroke(q, CornerRadius::same(3), Stroke::new(1.5, c), egui::StrokeKind::Inside);
                }
            }
        }
        Tool::Brush => {
            p.circle_filled(cc, w * 0.22, c);
            p.circle_stroke(cc, w * 0.38, Stroke::new(1.2, c.gamma_multiply(0.6)));
        }
        Tool::Lasso => {
            let pts: Vec<Pos2> = (0..=30).map(|k| {
                let a = k as f32 / 30.0 * std::f32::consts::TAU;
                cc + Vec2::new(a.cos() * w * 0.4, a.sin() * w * 0.26 - w * 0.08)
            }).collect();
            p.extend(Shape::dashed_line(&pts, s, 4.0, 3.0));
            p.line_segment([at(0.3, 0.6), at(0.22, 0.95)], s);
        }
        Tool::Edge => {
            let pts = vec![at(0.05, 0.75), at(0.35, 0.45), at(0.65, 0.55), at(0.95, 0.2)];
            p.add(Shape::line(pts.clone(), Stroke::new(2.4, theme::TEAL)));
            for q in pts {
                p.circle_filled(q, 3.0, c);
            }
        }
        Tool::Probe => {
            for (k, a) in [(0.0f32, 1.0f32), (1.0, 0.6), (2.0, 0.35)] {
                let y = 0.3 + k * 0.2;
                p.line_segment([at(0.15 + k * 0.08, y), at(0.85 - k * 0.08, y)], Stroke::new(1.6, c.gamma_multiply(a)));
            }
            p.circle_stroke(at(0.5, 0.5), w * 0.2, Stroke::new(1.6, theme::TEAL));
            p.line_segment([at(0.5, 0.05), at(0.5, 0.95)], Stroke::new(1.0, theme::TEAL));
        }
        Tool::Comb => {
            // A comb: a spine with teeth, and a flow line under it.
            p.line_segment([at(0.12, 0.3), at(0.88, 0.3)], Stroke::new(2.2, c));
            for k in 0..6 {
                let x = 0.18 + k as f32 * 0.13;
                p.line_segment([at(x, 0.3), at(x, 0.62)], Stroke::new(1.4, c));
            }
            p.line_segment([at(0.1, 0.85), at(0.9, 0.78)], Stroke::new(1.4, theme::TEAL));
        }
        Tool::Grab => {
            p.line_segment([at(0.5, 0.05), at(0.5, 0.95)], s);
            p.line_segment([at(0.05, 0.5), at(0.95, 0.5)], s);
            for (a, b, c2) in [((0.5, 0.0), (0.38, 0.16), (0.62, 0.16)), ((0.5, 1.0), (0.38, 0.84), (0.62, 0.84)), ((0.0, 0.5), (0.16, 0.38), (0.16, 0.62)), ((1.0, 0.5), (0.84, 0.38), (0.84, 0.62))] {
                p.add(Shape::convex_polygon(vec![at(a.0, a.1), at(b.0, b.1), at(c2.0, c2.1)], c, Stroke::NONE));
            }
        }
        Tool::Smooth => {
            let wave: Vec<Pos2> = (0..=24).map(|k| {
                let x = k as f32 / 24.0;
                at(x, 0.35 + (x * 14.0).sin() * 0.12)
            }).collect();
            p.add(Shape::line(wave, Stroke::new(1.4, c.gamma_multiply(0.5))));
            let calm: Vec<Pos2> = (0..=24).map(|k| {
                let x = k as f32 / 24.0;
                at(x, 0.7 + (x * 6.0).sin() * 0.03)
            }).collect();
            p.add(Shape::line(calm, s));
        }
        Tool::Inflate | Tool::Deflate => {
            p.circle_stroke(cc, w * 0.38, s);
            p.line_segment([cc - Vec2::new(w * 0.2, 0.0), cc + Vec2::new(w * 0.2, 0.0)], s);
            if t == Tool::Inflate {
                p.line_segment([cc - Vec2::new(0.0, w * 0.2), cc + Vec2::new(0.0, w * 0.2)], s);
            }
        }
        Tool::Flatten => {
            p.line_segment([at(0.05, 0.72), at(0.95, 0.72)], Stroke::new(2.6, c));
            for x in [0.25, 0.5, 0.75] {
                p.line_segment([at(x, 0.1), at(x, 0.5)], Stroke::new(1.5, c));
                p.add(Shape::convex_polygon(vec![at(x - 0.1, 0.45), at(x + 0.1, 0.45), at(x, 0.6)], c, Stroke::NONE));
            }
        }
        Tool::Paint => {
            p.line_segment([at(0.9, 0.1), at(0.42, 0.58)], Stroke::new(3.0, c));
            p.circle_filled(at(0.3, 0.7), w * 0.16, c);
        }
        Tool::Clone => {
            p.circle_stroke(at(0.36, 0.42), w * 0.24, s);
            p.circle_filled(at(0.64, 0.6), w * 0.24, c.gamma_multiply(0.8));
        }
        Tool::Smudge => {
            p.circle_filled(at(0.7, 0.4), w * 0.18, c);
            for (k, y) in [0.3f32, 0.42, 0.54].iter().enumerate() {
                p.line_segment([at(0.08 + k as f32 * 0.05, *y + 0.0), at(0.52, *y)], Stroke::new(1.6, c.gamma_multiply(0.6)));
            }
        }
    }
}
