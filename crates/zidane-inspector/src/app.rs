//! The window: a nav rail, the Compare and Inspect pages, a status bar with hints.
use crate::doc::{self, DiffView, Doc, Known, Msg, Slot};
use crate::theme::{self, card, kv, num, pct, pill, segmented};
use crate::viewport::{hover_tag, Draw, Size, ViewOpts, Viewport};
use crate::Args;
use eframe::egui::{self, Color32, CornerRadius, FontFamily, Key, Margin, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use wb_core::analysis::Parts;
use wb_core::render::{Camera, Filter};
use wb_core::uv;

#[path = "markpage.rs"]
mod markpage;

const FRONT: f32 = std::f32::consts::PI;
const BACK: f32 = 0.0;
const SIDE: f32 = std::f32::consts::FRAC_PI_2;
/// Zidane's height on screen in the field, measured at a 896 px tall frame.
const FIELD_PX: f32 = 237.0;
/// Battle height is an estimate until it is measured in a battle capture.
const BATTLE_PX: f32 = 160.0;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Page {
    Compare,
    Inspect,
    Swap,
    Mark,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum SwapView {
    After,
    Before,
    Changes,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Selection {
    None,
    Part(u8),
    Tri(usize, usize),
    /// A box dragged on a texture: image index and UV (u0, v0, u1, v1).
    Region(usize, [f32; 4]),
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Overlay {
    Changes,
    A,
    B,
}

#[derive(Default)]
struct PoseCache {
    key: Option<(usize, Option<usize>, u32)>,
    pose: Vec<Vec<[f32; 3]>>,
}

impl PoseCache {
    fn get(&mut self, d: &Doc, clip: Option<usize>, t: f32) -> (&[Vec<[f32; 3]>], (Option<usize>, u32)) {
        let tk = (t * 1000.0).round() as u32;
        let key = (d as *const Doc as usize, clip, tk);
        if self.key != Some(key) {
            self.pose = d.model.posed(clip, t);
            self.key = Some(key);
        }
        (&self.pose, (clip, tk))
    }
}

struct App {
    page: Page,
    known: Vec<Known>,
    parts: Option<Arc<Parts>>,
    parts_error: Option<String>,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    loading: Vec<(Slot, PathBuf)>,
    error: Option<String>,

    inspect: Option<Arc<Doc>>,
    inspect_path: PathBuf,
    a: Option<Arc<Doc>>,
    b: Option<Arc<Doc>>,
    diff: Option<Arc<DiffView>>,
    diff_running: bool,
    tol: u8,

    cam: Camera,
    reframe: Option<f32>,
    view: ViewOpts,
    clip: Option<String>,
    playing: bool,
    time: f32,
    speed: f32,
    show_changes: bool,
    show_dead: bool,
    selection: Selection,
    overlay: Overlay,
    inspect_image: usize,
    /// "What will this touch": light every surface that shows the selected pixels.
    show_touch: bool,
    /// Color texels that show on two surfaces.
    show_reuse: bool,
    touch_cache: Option<(usize, Selection, Arc<Vec<Vec<bool>>>)>,
    drag_from: Option<(f32, f32)>,
    want_texture: Option<String>,

    swap_png: Option<PathBuf>,
    swap_image: Option<String>,
    swap: Option<Arc<doc::SwapResult>>,
    swap_running: bool,
    swap_error: Option<String>,
    /// Base file, image and PNG of the last swap started, so each is run once.
    swap_started: Option<(usize, String, PathBuf)>,
    swap_saved: Option<PathBuf>,
    swap_view: SwapView,
    vp_swap: Viewport,
    pose_s: PoseCache,

    relayout_size: usize,
    /// Use Blender's dense layout instead of the built-in sewing pattern.
    relayout_blender: bool,
    relayout_running: bool,
    relayout_progress: String,
    relayout: Option<Arc<doc::RelayoutResult>>,
    relayout_error: Option<String>,
    /// `--relayout DIR`: start (or re-check) a re-layout once the Inspect file opens.
    relayout_auto: Option<PathBuf>,

    vp_inspect: Viewport,
    vp_a: Viewport,
    vp_b: Viewport,
    pose_i: PoseCache,
    pose_a: PoseCache,
    pose_b: PoseCache,
    tex_cache: HashMap<(usize, usize, u8), egui::TextureHandle>,

    advanced: bool,
    help: bool,
    hint: String,

    screenshot: Option<PathBuf>,
    settled: u32,
    frames: u32,
    sim_click: Option<(f32, f32)>,
    want_part: Option<String>,
    styled: bool,
    mark: markpage::Mark,
}

pub fn run(args: &Args) -> Result<(), String> {
    let known = doc::known_files();
    let (parts, parts_error) = match Parts::load(&args.parts_dir()) {
        Ok(p) => (Some(Arc::new(p)), None),
        Err(e) => (None, Some(e)),
    };
    let positional = args.positionals().into_iter().next();
    let page = match args.opt("--page") {
        Some("inspect") => Page::Inspect,
        Some("compare") => Page::Compare,
        Some("swap") => Page::Swap,
        Some("mark") => Page::Mark,
        Some(p) => return Err(format!("--page {p}: use inspect, compare, swap or mark")),
        None if positional.is_some() && args.opt("--b").is_none() => Page::Inspect,
        None => Page::Compare,
    };
    let size = match args.opt("--size") {
        Some("field") | None => Size::Field,
        Some("battle") => Size::Battle,
        Some("close") => Size::Close,
        Some(s) => return Err(format!("--size {s}: use field, battle or close")),
    };
    let sim_click = match args.opt("--click-model") {
        Some(s) => {
            let v: Vec<f32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            if v.len() != 2 {
                return Err("--click-model takes X,Y as fractions of the view, e.g. 0.5,0.6".into());
            }
            Some((v[0], v[1]))
        }
        None => None,
    };
    let baseline = known.iter().find(|k| k.label == "Finish v29").or_else(|| known.first()).map(|k| k.path.clone()).unwrap_or_else(|| PathBuf::from(crate::STOCK));
    let a_path = args.opt("--a").map(PathBuf::from).unwrap_or(baseline);
    let b_path = args.opt("--b").map(PathBuf::from);
    let inspect_path = positional.unwrap_or_else(|| PathBuf::from(crate::STOCK));
    let (tx, rx) = channel();
    let app = App {
        page,
        known,
        parts,
        parts_error,
        tx,
        rx,
        loading: vec![],
        error: None,
        inspect: None,
        inspect_path,
        a: None,
        b: None,
        diff: None,
        diff_running: false,
        tol: args.tol()?,
        cam: Camera { yaw: FRONT, pitch: 0.0, dist: 3.0, target: [0.0; 3], fov_y: 0.5, ortho: false, ortho_half: 1.0 },
        reframe: Some(FRONT),
        view: ViewOpts { size, field_px: FIELD_PX, battle_px: BATTLE_PX, filter: Filter::Smooth, actual: false },
        clip: args.opt("--clip").map(str::to_owned),
        playing: false,
        time: args.opt("--time").and_then(|t| t.parse().ok()).unwrap_or(0.0),
        speed: 1.0,
        show_changes: args.flag("--changes") || page == Page::Compare,
        show_dead: args.flag("--dead"),
        selection: match args.opt("--region") {
            Some(s) => {
                let v: Vec<f32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if v.len() != 4 {
                    return Err("--region takes U0,V0,U1,V1 in texture fractions, e.g. 0.6,0.05,0.75,0.3".into());
                }
                Selection::Region(0, [v[0].min(v[2]), v[1].min(v[3]), v[0].max(v[2]), v[1].max(v[3])])
            }
            None => Selection::None,
        },
        overlay: Overlay::Changes,
        inspect_image: 0,
        show_touch: args.flag("--touch"),
        show_reuse: args.flag("--reuse"),
        touch_cache: None,
        drag_from: None,
        want_texture: args.opt("--texture").map(str::to_owned),
        swap_png: args.opt("--png").map(PathBuf::from),
        swap_image: args.opt("--image").map(|s| s.trim_end_matches(".png").to_owned()),
        swap: None,
        swap_running: false,
        swap_error: None,
        swap_started: None,
        swap_saved: None,
        swap_view: SwapView::After,
        vp_swap: Viewport::default(),
        pose_s: PoseCache::default(),
        relayout_size: 4096,
        relayout_blender: args.flag("--blender-layout"),
        relayout_running: false,
        relayout_progress: String::new(),
        relayout: None,
        relayout_error: None,
        relayout_auto: args.opt("--relayout").map(PathBuf::from),
        vp_inspect: Viewport::default(),
        vp_a: Viewport::default(),
        vp_b: Viewport::default(),
        pose_i: PoseCache::default(),
        pose_a: PoseCache::default(),
        pose_b: PoseCache::default(),
        tex_cache: HashMap::new(),
        advanced: args.flag("--advanced"),
        help: false,
        hint: String::new(),
        screenshot: args.opt("--screenshot").map(PathBuf::from),
        settled: 0,
        frames: 0,
        sim_click,
        want_part: args.opt("--part").map(str::to_owned),
        styled: false,
        mark: markpage::Mark::from_args(args)?,
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1600.0, 940.0]).with_min_inner_size([1100.0, 640.0]).with_title("Zidane Workbench"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "Zidane Workbench",
        options,
        Box::new(move |cc| {
            let mut app = app;
            app.start(&cc.egui_ctx, a_path, b_path);
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| e.to_string())
}

impl App {
    fn start(&mut self, ctx: &egui::Context, a: PathBuf, b: Option<PathBuf>) {
        theme::install(ctx);
        self.styled = true;
        match self.page {
            Page::Compare => {
                self.load(ctx, Slot::A, a);
                if let Some(b) = b {
                    self.load(ctx, Slot::B, b);
                }
            }
            Page::Inspect | Page::Swap | Page::Mark => {
                let p = self.inspect_path.clone();
                self.load(ctx, Slot::Inspect, p);
            }
        }
    }

    fn load(&mut self, ctx: &egui::Context, slot: Slot, path: PathBuf) {
        self.loading.retain(|l| l.0 != slot);
        self.loading.push((slot, path.clone()));
        let label = doc::label_for(&path, &self.known);
        doc::spawn_load(self.tx.clone(), ctx.clone(), slot, path, label, self.parts.clone());
    }

    fn is_loading(&self, slot: Slot) -> bool {
        self.loading.iter().any(|l| l.0 == slot)
    }

    fn receive(&mut self, ctx: &egui::Context) {
        while let Ok(m) = self.rx.try_recv() {
            match m {
                Msg::Loaded(slot, path, r) => {
                    // A newer request for the slot wins.
                    if !self.loading.iter().any(|l| l.0 == slot && l.1 == path) {
                        continue;
                    }
                    self.loading.retain(|l| l.0 != slot);
                    match r {
                        Ok(d) => {
                            match slot {
                                Slot::Inspect => {
                                    let want = self.want_texture.take();
                                    self.inspect_image = want.and_then(|n| d.model.images.iter().position(|i| i.name.trim_end_matches(".png") == n)).unwrap_or(0);
                                    if let Selection::Region(_, r) = self.selection {
                                        self.selection = Selection::Region(self.inspect_image, r);
                                    }
                                    if let Some(dir) = self.relayout_auto.take() {
                                        self.start_relayout(ctx, d.path.clone(), dir);
                                    }
                                    self.inspect = Some(d);
                                }
                                Slot::A => {
                                    self.a = Some(d);
                                    self.reframe.get_or_insert(self.cam.yaw);
                                }
                                Slot::B => self.b = Some(d),
                            }
                            if let (Some(name), Some(p)) = (self.want_part.take(), self.parts.as_ref()) {
                                if let Some(id) = p.id_by_name(&name) {
                                    self.selection = Selection::Part(id);
                                }
                            }
                            self.error = None;
                        }
                        Err(e) => self.error = Some(format!("Could not open {}: {e}", path.display())),
                    }
                    self.tex_cache.clear();
                }
                Msg::Swapped(png, r) => {
                    if self.swap_started.as_ref().is_some_and(|s| s.2 == png) {
                        self.swap_running = false;
                        match r {
                            Ok(v) => {
                                self.swap = Some(v);
                                self.swap_error = None;
                            }
                            Err(e) => self.swap_error = Some(e),
                        }
                    }
                }
                Msg::RelayoutProgress(l) => self.relayout_progress = l,
                Msg::Relayouted(r) => {
                    self.relayout_running = false;
                    match r {
                        Ok(v) => {
                            self.relayout = Some(v);
                            self.relayout_error = None;
                        }
                        Err(e) => self.relayout_error = Some(e),
                    }
                }
                Msg::Compared(v) => {
                    self.diff_running = false;
                    if let (Some(a), Some(b)) = (&self.a, &self.b) {
                        if v.matches(a, b, self.tol) {
                            self.diff = Some(v);
                            self.tex_cache.retain(|k, _| k.2 != 1);
                        }
                    }
                }
            }
        }
        // Keep the comparison current.
        if let (Some(a), Some(b)) = (&self.a, &self.b) {
            let stale = self.diff.as_ref().map_or(true, |d| !d.matches(a, b, self.tol));
            if stale && !self.diff_running {
                self.diff_running = true;
                doc::spawn_compare(self.tx.clone(), ctx.clone(), a.clone(), b.clone(), self.parts.clone(), self.tol);
            }
        }
    }

    fn part_name(&self, id: u8) -> String {
        self.parts.as_ref().map(|p| p.name(id).to_owned()).unwrap_or_else(|| format!("part {id}"))
    }

    fn texture(&mut self, ctx: &egui::Context, owner: usize, idx: usize, kind: u8, img: Option<&egui::ColorImage>) -> Option<egui::TextureHandle> {
        if let Some(t) = self.tex_cache.get(&(owner, idx, kind)) {
            return Some(t.clone());
        }
        let img = img?;
        let t = ctx.load_texture(format!("tex-{owner}-{idx}-{kind}"), img.clone(), egui::TextureOptions::NEAREST);
        self.tex_cache.insert((owner, idx, kind), t.clone());
        Some(t)
    }

    fn clip_names(&self) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();
        for d in [&self.a, &self.b, &self.inspect].into_iter().flatten() {
            for c in &d.model.clips {
                if !v.contains(c) {
                    v.push(c.clone());
                }
            }
        }
        v
    }

    fn clip_duration(&self, d: &Doc) -> f32 {
        self.clip.as_deref().and_then(|c| d.clip_index(c)).map(|i| d.model.clip_duration(i)).unwrap_or(0.0)
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }
        let (t, u) = ctx.input(|i| (i.key_pressed(Key::T), i.key_pressed(Key::U)));
        if self.page == Page::Inspect {
            self.show_touch ^= t;
            self.show_reuse ^= u;
        }
        let (space, k1, k2, k3, f, b, s, d, c, o, f1) = ctx.input(|i| {
            (
                i.key_pressed(Key::Space),
                i.key_pressed(Key::Num1),
                i.key_pressed(Key::Num2),
                i.key_pressed(Key::Num3),
                i.key_pressed(Key::F),
                i.key_pressed(Key::B),
                i.key_pressed(Key::S),
                i.key_pressed(Key::D),
                i.key_pressed(Key::C),
                i.modifiers.command && i.key_pressed(Key::O),
                i.key_pressed(Key::F1),
            )
        });
        if space && self.clip.is_some() {
            self.playing = !self.playing;
        }
        if k1 {
            self.view.size = Size::Field;
        }
        if k2 {
            self.view.size = Size::Battle;
        }
        if k3 {
            self.view.size = Size::Close;
        }
        if f {
            self.reframe = Some(FRONT);
        }
        if b {
            self.reframe = Some(BACK);
        }
        if s {
            self.reframe = Some(SIDE);
        }
        if d {
            self.show_dead = !self.show_dead;
        }
        if c {
            self.show_changes = !self.show_changes;
        }
        if f1 {
            self.help = !self.help;
        }
        if o {
            let slot = if self.page == Page::Compare { Slot::B } else { Slot::Inspect };
            self.browse(ctx, slot);
        }
    }

    fn browse(&mut self, ctx: &egui::Context, slot: Slot) {
        let start = self.known.first().and_then(|k| k.path.parent().map(|p| p.to_path_buf()));
        let mut dlg = rfd::FileDialog::new().add_filter("Character export (GLB)", &["glb"]).set_title("Open a character export (read-only)");
        if let Some(s) = start {
            dlg = dlg.set_directory(s);
        }
        if let Some(p) = dlg.pick_file() {
            if slot == Slot::Inspect {
                self.inspect_path = p.clone();
            }
            self.load(ctx, slot, p);
        }
    }

    // ---------- shell ----------

    fn rail(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("rail").exact_width(92.0).resizable(false).frame(egui::Frame::new().fill(theme::BG).inner_margin(Margin::symmetric(8, 14))).show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.label(RichText::new("ZW").family(FontFamily::Name("strong".into())).size(22.0).color(theme::GOLD));
                ui.label(RichText::new("Workbench").size(11.0).color(theme::FAINT));
            });
            ui.add_space(18.0);
            let items: [(Option<Page>, &str, &str, &str); 5] = [
                (Some(Page::Compare), "⚖", "Compare", "Original and candidate side by side, with what changed"),
                (Some(Page::Inspect), "🔍", "Inspect", "One file: parts, textures, dead triangles"),
                (Some(Page::Swap), "🔁", "Safe swap", "Put a painted texture into a copy of the file and check nothing else changed"),
                (Some(Page::Mark), "✏", "Mark & fix", "Point at what is wrong and tell the AI, or make a rough edit yourself and save it as a new file"),
                (None, "🎞", "Review reel", "Coming next: turntables and clips for sign-off"),
            ];
            for (page, icon, label, tip) in items {
                let sel = page == Some(self.page);
                let enabled = page.is_some();
                let fg = if sel { theme::GOLD } else if enabled { theme::TEXT } else { theme::FAINT };
                let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 62.0), if enabled { Sense::click() } else { Sense::hover() });
                let p = ui.painter();
                if sel {
                    p.rect_filled(rect, CornerRadius::same(9), theme::CARD_HI);
                    p.rect_filled(Rect::from_min_size(rect.min + Vec2::new(0.0, 14.0), Vec2::new(3.0, 34.0)), CornerRadius::same(2), theme::GOLD);
                } else if resp.hovered() && enabled {
                    p.rect_filled(rect, CornerRadius::same(9), theme::CARD);
                }
                p.text(rect.center() - Vec2::new(0.0, 9.0), egui::Align2::CENTER_CENTER, icon, egui::FontId::proportional(20.0), fg);
                p.text(rect.center() + Vec2::new(0.0, 15.0), egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(12.0), fg);
                let resp = resp.on_hover_text(tip);
                if resp.clicked() {
                    if let Some(pg) = page {
                        self.page = pg;
                        self.selection = Selection::None;
                        if pg != Page::Compare && self.inspect.is_none() && !self.is_loading(Slot::Inspect) {
                            let p = self.inspect_path.clone();
                            self.load(ctx, Slot::Inspect, p);
                        }
                    }
                }
                ui.add_space(4.0);
            }
            ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                if ui.add(egui::Button::new(RichText::new("?  Help").size(13.0)).min_size(Vec2::new(76.0, 30.0))).on_hover_text("What everything means (F1)").clicked() {
                    self.help = !self.help;
                }
                ui.add_space(4.0);
                ui.label(RichText::new("Advanced").size(11.5).color(if self.advanced { theme::GOLD } else { theme::FAINT }));
                toggle(ui, &mut self.advanced).on_hover_text("Show exact numbers, tolerance, filters and paths");
            });
        });
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status").exact_height(30.0).frame(egui::Frame::new().fill(theme::BG).inner_margin(Margin::symmetric(14, 5))).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                if !self.loading.is_empty() || self.diff_running || self.swap_running || self.relayout_running {
                    ui.add(egui::Spinner::new().size(14.0).color(theme::GOLD));
                    let what: Vec<String> = self
                        .loading
                        .iter()
                        .map(|l| format!("Opening {}", doc::label_for(&l.1, &self.known)))
                        .chain(self.diff_running.then(|| "Comparing".to_owned()))
                        .chain(self.swap_running.then(|| "Swapping and checking".to_owned()))
                        .chain(self.relayout_running.then(|| "Making the unique layout".to_owned()))
                        .collect();
                    ui.label(RichText::new(what.join(" · ")).color(theme::TEXT).size(12.5));
                    ui.separator();
                }
                if let Some(e) = &self.error {
                    ui.label(RichText::new(e).color(theme::RED).size(12.5));
                } else {
                    ui.label(RichText::new(&self.hint).color(theme::WEAK).size(12.5));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.page == Page::Mark {
                        pill(ui, "Edits save to new files only", theme::GREEN);
                    } else {
                        pill(ui, "Read-only: files are never changed", theme::GREEN);
                    }
                    if let Some(e) = &self.parts_error {
                        pill(ui, "Part names unavailable", theme::RED).on_hover_text(e.as_str());
                    }
                });
            });
        });
    }

    fn help_window(&mut self, ctx: &egui::Context) {
        let mut open = self.help;
        egui::Window::new(RichText::new("Help").family(FontFamily::Name("strong".into())).size(16.0)).open(&mut open).collapsible(false).resizable(false).default_width(520.0).show(ctx, |ui| {
            ui.label(RichText::new("Words used here").strong());
            egui::Grid::new("glossary").num_columns(2).spacing([14.0, 6.0]).show(ui, |ui| {
                for (k, v) in [
                    ("A and B", "A is the original (left), B is the candidate (right). Both use the same camera."),
                    ("Changed pixels", "Texture pixels in B that differ from A. Shown in pink on the model and the texture."),
                    ("Parts", "Named areas from the UV kit (gloves, belt band, hair...). Click one to light it up."),
                    ("Dead triangles", "Triangles with almost no texture area. Paint there never shows."),
                    ("Touch", "Every surface that shows the selected pixels. A part can reach past its own edge, and mirrored pieces show the same paint."),
                    ("Reuse", "Texture pixels that show on two surfaces: a mirrored twin (left and right share paint), two pieces sharing, or a piece folded onto itself."),
                    ("Pixel density", "Texture pixels per world unit of surface. Where it varies, a mark drawn one size comes out larger or smaller on the model."),
                    ("In field", "Zidane at his real field size: 237 px tall in a 896 px frame, straight-on camera."),
                    ("In battle", "Battle size, 160 px tall. An estimate until a battle capture measures it."),
                    ("Close-up", "Full window resolution. Scroll to zoom. Good for checking paint detail."),
                    ("Rest pose", "The model as stored, before any animation."),
                ] {
                    ui.label(RichText::new(k).color(theme::GOLD));
                    ui.label(v);
                    ui.end_row();
                }
            });
            ui.add_space(8.0);
            ui.label(RichText::new("Mouse and keys").strong());
            egui::Grid::new("keys").num_columns(2).spacing([14.0, 4.0]).show(ui, |ui| {
                for (k, v) in [
                    ("Drag", "Turn the model"),
                    ("Scroll", "Zoom (Close-up only)"),
                    ("Click", "Select the part under the pointer"),
                    ("Space", "Play or pause the clip"),
                    ("1  2  3", "In field, In battle, Close-up"),
                    ("F  B  S", "Front, back, side view"),
                    ("C  D", "Show changes, show dead triangles"),
                    ("T  U", "Inspect: Touch and Reuse"),
                    ("Drag on texture", "Inspect: see where paint in that box shows"),
                    ("Ctrl+O", "Open a file as B (or in Inspect)"),
                    ("F1", "This help"),
                ] {
                    ui.label(RichText::new(k).monospace().color(theme::TEXT));
                    ui.label(RichText::new(v).color(theme::WEAK));
                    ui.end_row();
                }
            });
        });
        self.help = open;
    }

    // ---------- shared controls ----------

    fn file_picker(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, slot: Slot, current: Option<&Arc<Doc>>) {
        let text = match current {
            Some(d) => d.label.clone(),
            None if self.is_loading(slot) => "Opening...".into(),
            None => "Choose a file".into(),
        };
        let mut pick: Option<PathBuf> = None;
        let mut browse = false;
        egui::ComboBox::from_id_salt(("pick", slot as u8)).selected_text(RichText::new(text).strong()).width(250.0).height(420.0).show_ui(ui, |ui| {
            for k in &self.known {
                let sel = current.is_some_and(|d| d.path == k.path);
                let r = ui.selectable_label(sel, RichText::new(k.label).strong()).on_hover_text(k.about);
                if r.clicked() {
                    pick = Some(k.path.clone());
                }
            }
            ui.separator();
            if ui.selectable_label(false, "Browse for a file...").clicked() {
                browse = true;
            }
        });
        if let Some(p) = pick {
            if slot == Slot::Inspect {
                self.inspect_path = p.clone();
            }
            self.load(ctx, slot, p);
        }
        if browse {
            self.browse(ctx, slot);
        }
    }

    fn playback(&mut self, ui: &mut egui::Ui, dur_doc: Option<Arc<Doc>>) {
        let clips = self.clip_names();
        let mut choice: Option<String> = self.clip.clone();
        let mut opts: Vec<(Option<String>, String, String)> = vec![(None, "Rest pose".into(), "The model as stored, no animation".into())];
        for c in &clips {
            opts.push((Some(c.clone()), title_case(c), format!("Play the {c} clip")));
        }
        let opt_refs: Vec<(usize, &str, &str)> = opts.iter().enumerate().map(|(i, o)| (i, o.1.as_str(), o.2.as_str())).collect();
        let mut idx = opts.iter().position(|o| o.0 == choice).unwrap_or(0);
        if segmented(ui, &mut idx, &opt_refs) {
            choice = opts[idx].0.clone();
            self.clip = choice.clone();
            self.time = 0.0;
            self.playing = choice.is_some();
        }
        let dur = dur_doc.as_deref().map(|d| self.clip_duration(d)).unwrap_or(0.0);
        ui.add_enabled_ui(self.clip.is_some(), |ui| {
            let label = if self.playing { "⏸ Pause" } else { "▶ Play" };
            if ui.add(egui::Button::new(label).min_size(Vec2::new(78.0, 28.0))).on_hover_text("Space").clicked() {
                self.playing = !self.playing;
            }
            ui.spacing_mut().slider_width = 150.0;
            if dur > 0.0 {
                let r = ui.add(egui::Slider::new(&mut self.time, 0.0..=dur).show_value(self.advanced).suffix(" s").max_decimals(2));
                if r.dragged() {
                    self.playing = false;
                }
            } else {
                // Not loaded yet (or no clip): keep the requested time instead of clamping it.
                let mut zero = 0.0f32;
                ui.add_enabled(false, egui::Slider::new(&mut zero, 0.0..=1.0).show_value(false));
            }
            if self.advanced {
                ui.add(egui::Slider::new(&mut self.speed, 0.1..=2.0).text("speed").max_decimals(1));
            }
        });
    }

    fn size_controls(&mut self, ui: &mut egui::Ui) {
        segmented(
            ui,
            &mut self.view.size,
            &[
                (Size::Field, "In field", "Real field size: 237 px tall (key 1)"),
                (Size::Battle, "In battle", "Battle size, 160 px tall, an estimate (key 2)"),
                (Size::Close, "Close-up", "Full resolution, scroll to zoom (key 3)"),
            ],
        );
    }

    fn view_buttons(&mut self, ui: &mut egui::Ui) {
        for (label, yaw, key) in [("Front", FRONT, "F"), ("Back", BACK, "B"), ("Side", SIDE, "S")] {
            if ui.button(label).on_hover_text(format!("Turn to the {} view ({key})", label.to_lowercase())).clicked() {
                self.reframe = Some(yaw);
            }
        }
    }

    fn advance_time(&mut self, ctx: &egui::Context, dur: f32) {
        if self.playing && dur > 0.0 {
            let dt = ctx.input(|i| i.stable_dt).min(0.1);
            self.time = (self.time + dt * self.speed) % dur;
            ctx.request_repaint();
        }
        if dur > 0.0 {
            self.time = self.time.min(dur);
        }
    }

    fn tint_key(&self, extra: usize) -> u64 {
        let sel = match self.selection {
            Selection::None => 0u64,
            Selection::Part(p) => 1 + p as u64,
            Selection::Tri(m, t) => 1000 + (m as u64) * 100_000 + t as u64,
            Selection::Region(i, r) => r.iter().fold(7 + i as u64, |h, x| h.rotate_left(13) ^ x.to_bits() as u64) | (1 << 39),
        };
        sel ^ ((self.show_changes as u64) << 40) ^ ((self.show_dead as u64) << 41) ^ ((self.show_touch as u64) << 38) ^ ((extra as u64) << 42).rotate_left(7)
    }

    /// "What will this touch" for the current selection: per mesh, per triangle.
    fn touched(&mut self, d: &Arc<Doc>) -> Option<Arc<Vec<Vec<bool>>>> {
        let key = Arc::as_ptr(d) as usize;
        let want = self.selection;
        if !matches!(want, Selection::Part(_) | Selection::Region(..)) {
            return None;
        }
        if let Some((k, s, v)) = &self.touch_cache {
            if *k == key && *s == want {
                return Some(v.clone());
            }
        }
        let v = Arc::new(match want {
            Selection::Part(p) => d.uv.touched_by_part(&d.tri_parts, p),
            Selection::Region(img, r) => uv::Facts::touched_by_rect(&d.model, img, r),
            _ => return None,
        });
        self.touch_cache = Some((key, want, v.clone()));
        Some(v)
    }

    fn view_caption(&self, d: &Doc, clip_missing: bool) -> String {
        let size = match self.view.size {
            Size::Field => format!("In field · {} px tall", self.view.field_px.round()),
            Size::Battle => format!("In battle · {} px tall (estimate)", self.view.battle_px.round()),
            Size::Close => "Close-up".to_owned(),
        };
        let pose = match (&self.clip, clip_missing) {
            (Some(c), false) => format!("{} {:.2} s", title_case(c), self.time),
            (Some(c), true) => format!("no {c} clip: rest pose"),
            (None, _) => "Rest pose".into(),
        };
        let tex = d.tex_size().map(|s| format!(" · {} textures", if s[0] >= 4096 { "4K".into() } else { format!("{}", s[0]) })).unwrap_or_default();
        format!("{size} · {pose}{tex}")
    }

    // ---------- Compare ----------

    fn compare_page(&mut self, ctx: &egui::Context) {
        let has_b = self.b.is_some() || self.is_loading(Slot::B);
        if has_b {
            egui::SidePanel::right("changes").default_width(360.0).min_width(300.0).max_width(520.0).resizable(true).frame(egui::Frame::new().fill(theme::PANEL).inner_margin(Margin::same(12))).show(ctx, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.changes_panel(ui, ctx));
            });
        }
        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG).inner_margin(Margin::same(14))).show(ctx, |ui| {
            // Row 1: files.
            ui.horizontal(|ui| {
                ui.label(RichText::new("Compare").family(FontFamily::Name("strong".into())).size(22.0));
                ui.add_space(16.0);
                badge(ui, "A", theme::A_COLOR);
                let a = self.a.clone();
                self.file_picker(ui, ctx, Slot::A, a.as_ref());
                if ui.add_enabled(self.a.is_some() && self.b.is_some(), egui::Button::new("Swap")).on_hover_text("Swap A and B").clicked() {
                    std::mem::swap(&mut self.a, &mut self.b);
                    self.diff = None;
                }
                badge(ui, "B", theme::B_COLOR);
                let b = self.b.clone();
                self.file_picker(ui, ctx, Slot::B, b.as_ref());
                if self.b.is_some() {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for (label, yaw, key) in [("Side", SIDE, "S"), ("Back", BACK, "B"), ("Front", FRONT, "F")] {
                            if ui.button(label).on_hover_text(format!("Turn to the {} view ({key})", label.to_lowercase())).clicked() {
                                self.reframe = Some(yaw);
                            }
                        }
                    });
                }
            });
            ui.add_space(6.0);
            if self.b.is_none() && !self.is_loading(Slot::B) {
                self.empty_state(ui, ctx);
                self.hint = "Pick a candidate to see what it changed. A is the baseline on the left.".into();
                return;
            }
            // Row 2: animation. Row 3: how the views draw.
            ui.horizontal(|ui| {
                row_label(ui, "Animation");
                let a = self.a.clone();
                self.playback(ui, a);
            });
            ui.horizontal(|ui| {
                row_label(ui, "View");
                self.size_controls(ui);
                ui.separator();
                let cc = if self.show_changes { theme::MAGENTA } else { theme::WEAK };
                let dc = if self.show_dead { theme::RED } else { theme::WEAK };
                ui.toggle_value(&mut self.show_changes, RichText::new("● Changes").color(cc)).on_hover_text("Light up triangles whose texture changed (C)");
                ui.toggle_value(&mut self.show_dead, RichText::new("● Dead").color(dc)).on_hover_text("Light up triangles too small to show paint (D)");
                if self.advanced {
                    ui.separator();
                    segmented(ui, &mut self.view.filter, &[(Filter::Smooth, "Smooth", "Mipmapped, like the game with HD Characters"), (Filter::Texels, "Texels", "Nearest texel from the full texture")]);
                    ui.checkbox(&mut self.view.actual, "1:1").on_hover_text("Show field and battle sizes at their real pixel size instead of enlarged");
                    match self.view.size {
                        Size::Field => {
                            ui.add(egui::DragValue::new(&mut self.view.field_px).range(60.0..=900.0).suffix(" px").speed(1.0)).on_hover_text("Model height on screen");
                        }
                        Size::Battle => {
                            ui.add(egui::DragValue::new(&mut self.view.battle_px).range(60.0..=900.0).suffix(" px").speed(1.0)).on_hover_text("Model height on screen");
                        }
                        Size::Close => {}
                    }
                }
            });
            ui.add_space(8.0);
            self.compare_views(ui, ctx);
        });
    }

    fn empty_state(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.add_space(30.0);
        ui.vertical_centered(|ui| {
            ui.label(RichText::new("What did a candidate change?").family(FontFamily::Name("strong".into())).size(26.0));
            ui.label(RichText::new("Pick one. It opens on the right, next to A, with every changed pixel marked.").color(theme::WEAK).size(15.0));
        });
        ui.add_space(20.0);
        let a_path = self.a.as_ref().map(|d| d.path.clone());
        let list: Vec<(String, String, PathBuf)> = self.known.iter().filter(|k| Some(&k.path) != a_path.as_ref()).map(|k| (k.label.to_owned(), k.about.to_owned(), k.path.clone())).collect();
        let w = ui.available_width();
        let cols = ((w / 300.0).floor() as usize).clamp(1, 3);
        let cw = (w - 16.0 * (cols as f32 - 1.0)) / cols as f32;
        let mut chosen: Option<PathBuf> = None;
        egui::Grid::new("cands").num_columns(cols).spacing([16.0, 16.0]).show(ui, |ui| {
            for (i, (label, about, path)) in list.iter().enumerate() {
                let r = egui::Frame::new().fill(theme::CARD).stroke(Stroke::new(1.0, theme::LINE)).corner_radius(CornerRadius::same(12)).inner_margin(Margin::same(16)).show(ui, |ui| {
                    ui.vertical(|ui| {
                        ui.set_width(cw - 34.0);
                        ui.set_min_height(92.0);
                        ui.label(RichText::new(label).family(FontFamily::Name("strong".into())).size(17.0));
                        ui.add(egui::Label::new(RichText::new(about).color(theme::WEAK)).wrap());
                        ui.add_space(4.0);
                        ui.add(egui::Label::new(RichText::new(path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()).size(11.5).color(theme::FAINT).monospace()).truncate());
                    });
                });
                let r = ui.interact(r.response.rect, egui::Id::new(("cand", i)), Sense::click());
                if r.hovered() {
                    ui.painter().rect_stroke(r.rect, CornerRadius::same(12), Stroke::new(1.5, theme::GOLD), egui::StrokeKind::Inside);
                    ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
                }
                if r.clicked() {
                    chosen = Some(path.clone());
                }
                if (i + 1) % cols == 0 {
                    ui.end_row();
                }
            }
        });
        ui.add_space(14.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Or").color(theme::WEAK));
            if ui.button("Browse for a file...").clicked() {
                self.browse(ctx, Slot::B);
            }
        });
        if let Some(p) = chosen {
            self.load(ctx, Slot::B, p);
        }
    }

    fn compare_views(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let (area, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
        let gap = 12.0;
        let head = 30.0;
        let half = (area.width() - gap) / 2.0;
        let rect_a = Rect::from_min_size(area.min + Vec2::new(0.0, head), Vec2::new(half, area.height() - head));
        let rect_b = Rect::from_min_size(area.min + Vec2::new(half + gap, head), Vec2::new(half, area.height() - head));

        let (Some(a), b) = (self.a.clone(), self.b.clone()) else {
            ui.put(area, egui::Spinner::new().size(32.0));
            return;
        };
        if let Some(yaw) = self.reframe.take() {
            let mut c = Camera::fit(&a.model, rect_a.width() / rect_a.height().max(1.0), 0.8, false);
            c.yaw = yaw;
            self.cam = c;
        }
        let dur = self.clip_duration(&a);
        self.advance_time(ctx, dur);

        // Headers.
        for (rect, letter, color, d) in [(rect_a, "A", theme::A_COLOR, Some(&a)), (rect_b, "B", theme::B_COLOR, b.as_ref())] {
            let hr = Rect::from_min_size(Pos2::new(rect.min.x, area.min.y), Vec2::new(rect.width(), head - 6.0));
            ui.scope_builder(egui::UiBuilder::new().max_rect(hr), |ui| {
                ui.horizontal_centered(|ui| {
                    badge(ui, letter, color);
                    match d {
                        Some(d) => {
                            ui.label(RichText::new(&d.label).strong().size(15.0));
                            ui.label(RichText::new(d.file_name()).color(theme::FAINT).size(12.0));
                        }
                        None => {
                            ui.add(egui::Spinner::new().size(14.0));
                            ui.label(RichText::new("Opening...").color(theme::WEAK));
                        }
                    }
                });
            });
        }

        let diff = self.diff.clone();
        let sel = self.selection;
        let (show_dead, show_changes) = (self.show_dead, self.show_changes);
        let diff_id = diff.as_ref().map_or(0, |d| Arc::as_ptr(d) as usize);
        let tkey = self.tint_key(diff_id);
        let clip = self.clip.clone();
        let time = self.time;
        let sim = self.sim_click.take();
        let mut hovered: Option<(char, usize, usize)> = None;
        let mut clicked: Option<Option<u8>> = None;

        for side in ['A', 'B'] {
            let Some(d) = (if side == 'A' { Some(a.clone()) } else { b.clone() }) else {
                continue;
            };
            let ci = clip.as_deref().and_then(|c| d.clip_index(c));
            let missing = clip.is_some() && ci.is_none();
            let caption = self.view_caption(&d, missing);
            let changed = diff.as_ref().map(|v| if side == 'A' { &v.changed_a } else { &v.changed_b });
            let tint = |mi: usize, ti: usize| -> Option<[u8; 3]> {
                if show_dead && d.dead[mi][ti] {
                    return Some(theme::HL_DEAD);
                }
                if let Selection::Part(p) = sel {
                    if d.tri_parts[mi][ti] != p {
                        return Some(theme::HL_DIM);
                    }
                }
                if show_changes && changed.and_then(|c| c.get(mi)).and_then(|v| v.get(ti)).copied().unwrap_or(false) {
                    return Some(theme::HL_CHANGED);
                }
                None
            };
            let (pose_cache, vp, rect) = if side == 'A' { (&mut self.pose_a, &mut self.vp_a, rect_a) } else { (&mut self.pose_b, &mut self.vp_b, rect_b) };
            let (pose, pose_key) = pose_cache.get(&d, ci, time);
            let shown = vp.show(
                ui,
                rect,
                egui::Id::new(("view", side)),
                &mut self.cam,
                &self.view,
                Draw { doc: &d, frame: &a.model, pose, pose_key, tint: &tint, tint_key: tkey, caption, sim_click: if side == 'A' { sim } else { None } },
            );
            if let Some((mi, ti)) = shown.hovered {
                hovered = Some((side, mi, ti));
            }
            if let Some((mi, ti)) = shown.clicked {
                let p = d.tri_parts[mi][ti];
                clicked = Some((p != 0).then_some(p));
            } else if shown.clicked_empty {
                clicked = Some(None);
            }
        }
        if let Some(c) = clicked {
            self.selection = match c {
                Some(p) if self.selection == Selection::Part(p) => Selection::None,
                Some(p) => Selection::Part(p),
                None => Selection::None,
            };
        }
        self.hint = "Drag to turn · Click a part to light it up on both sides · Space plays the clip · F1 for help".into();
        if let Some((side, mi, ti)) = hovered {
            let d = if side == 'A' { &a } else { b.as_ref().unwrap() };
            let p = d.tri_parts[mi][ti];
            let name = if p != 0 { self.part_name(p) } else { "no part".into() };
            let ch = diff.as_ref().map(|v| if side == 'A' { &v.changed_a } else { &v.changed_b }).and_then(|c| c.get(mi)).and_then(|v| v.get(ti)).copied().unwrap_or(false);
            let mut text = title_case(&name);
            if ch {
                text.push_str("  ·  texture changed");
            }
            if d.dead[mi][ti] {
                text.push_str("  ·  dead triangle");
            }
            if self.advanced {
                text.push_str(&format!("  ·  {} tri {ti}", d.model.meshes[mi].name));
            }
            hover_tag(ui, &text, if ch { theme::MAGENTA } else { theme::LINE });
            self.hint = format!("Click to select {}", title_case(&name));
        }
    }

    fn changes_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label(RichText::new("What changed").family(FontFamily::Name("strong".into())).size(19.0));
        ui.add_space(4.0);
        let (Some(a), Some(b)) = (self.a.clone(), self.b.clone()) else {
            card(ui, "", |ui| {
                ui.label(RichText::new("Pick a B file to see what it changed.").color(theme::WEAK));
            });
            return;
        };
        let Some(v) = self.diff.clone() else {
            card(ui, "", |ui| {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new().color(theme::GOLD));
                    ui.label("Comparing every pixel...");
                });
            });
            return;
        };
        let d = &v.diff;
        // Headline.
        card(ui, "", |ui| {
            if d.identical_files {
                ui.label(RichText::new("These two files are identical.").strong().size(16.0).color(theme::GREEN));
                return;
            }
            let changed_tex = d.textures.iter().filter(|t| t.status() != "same").count();
            let changed_mesh = d.meshes.iter().filter(|m| m.status() != "same").count();
            let line = match (changed_tex, changed_mesh) {
                (0, 0) => "Same textures and meshes; only file details differ.".to_owned(),
                (t, 0) => format!("Paint only: {t} of {} textures changed, meshes untouched.", d.textures.len()),
                (0, m) => format!("Shape only: {m} mesh{} changed, textures untouched.", if m == 1 { "" } else { "es" }),
                (t, m) => format!("{t} texture{} and {m} mesh{} changed.", if t == 1 { "" } else { "s" }, if m == 1 { "" } else { "es" }),
            };
            ui.label(RichText::new(line).strong().size(15.5));
            ui.label(RichText::new(format!("{} pixels changed in all.", num(d.changed_pixels()))).color(theme::WEAK));
        });
        // Size warning.
        if d.textures.iter().any(|t| t.resized) {
            card(ui, "", |ui| {
                ui.label(RichText::new("⚠  Texture sizes differ").strong().color(theme::GOLD));
                let sizes: Vec<String> = d.textures.iter().filter(|t| t.resized).map(|t| format!("{}: {} vs {}", t.name, t.a_size.map_or(0, |s| s[0]), t.b_size.map_or(0, |s| s[0]))).collect();
                ui.label(RichText::new(format!("{}. The smaller one is scaled up to compare, so nearly every pixel counts as changed. Compare 4K against 4K to see real edits.", sizes.join(", "))).color(theme::WEAK));
                let base = self.known.iter().find(|k| k.label == "Finish v29").map(|k| k.path.clone());
                if let Some(p) = base {
                    if p != a.path && ui.button("Use Finish v29 (4K) as A").clicked() {
                        self.load(ctx, Slot::A, p);
                    }
                }
            });
        }
        // Textures.
        let diff_id = Arc::as_ptr(&v) as usize;
        ui.horizontal(|ui| {
            ui.label(RichText::new("TEXTURES").size(11.5).color(theme::FAINT).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                segmented(ui, &mut self.overlay, &[(Overlay::B, "B", "B's texture"), (Overlay::A, "A", "A's texture"), (Overlay::Changes, "Changes", "Changed pixels in pink")]);
            });
        });
        for (ti, t) in d.textures.iter().enumerate() {
            let status = t.status();
            let color = match status {
                "same" => theme::GREEN,
                "changed" => theme::MAGENTA,
                _ => theme::GOLD,
            };
            let thumb = match self.overlay {
                Overlay::Changes => self.texture(ctx, diff_id, ti, 1, v.overlays[ti].as_ref()),
                Overlay::A => a.model.images.iter().position(|i| i.name.trim_end_matches(".png") == t.name).and_then(|ii| self.texture(ctx, Arc::as_ptr(&a) as usize, ii, 0, a.previews[ii].as_ref())),
                Overlay::B => b.model.images.iter().position(|i| i.name.trim_end_matches(".png") == t.name).and_then(|ii| self.texture(ctx, Arc::as_ptr(&b) as usize, ii, 0, b.previews[ii].as_ref())),
            };
            let mut select: Option<u8> = None;
            card(ui, "", |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(texture_title(&t.name)).strong().size(15.0));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        pill(ui, status, color);
                    });
                });
                ui.label(RichText::new(&t.name).monospace().size(11.5).color(theme::FAINT));
                if let Some(h) = thumb {
                    let s = ui.available_width();
                    let r = ui.add(egui::Image::new((h.id(), Vec2::splat(s))).corner_radius(CornerRadius::same(6)).sense(Sense::hover()));
                    if t.changed > 0 {
                        r.on_hover_text("Pink: changed pixels. Gray: unchanged.");
                    }
                } else {
                    ui.label(RichText::new("Not in this file").color(theme::FAINT));
                }
                if status != "same" && t.a_size.is_some() && t.b_size.is_some() {
                    ui.label(RichText::new(format!("{} pixels changed ({})", num(t.changed), pct(t.changed, t.total))).strong());
                }
                if !t.by_part.is_empty() {
                    ui.add_space(2.0);
                    ui.label(RichText::new("Where").size(12.0).color(theme::FAINT));
                    let top = t.by_part.iter().take(if self.advanced { 14 } else { 5 });
                    for (id, n) in top {
                        let name = if *id == 0 { "outside any part".to_owned() } else { self.part_name(*id) };
                        let sel = self.selection == Selection::Part(*id);
                        let share = *n as f32 / t.changed.max(1) as f32;
                        let r = ui.horizontal(|ui| {
                            let resp = ui.selectable_label(sel, title_case(&name));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(RichText::new(if self.advanced { num(*n) } else { format!("{:.0}%", share * 100.0) }).color(theme::WEAK));
                                let (br, _) = ui.allocate_exact_size(Vec2::new(70.0, 6.0), Sense::hover());
                                ui.painter().rect_filled(br, CornerRadius::same(3), theme::BG);
                                ui.painter().rect_filled(Rect::from_min_size(br.min, Vec2::new(70.0 * share.clamp(0.02, 1.0), 6.0)), CornerRadius::same(3), theme::MAGENTA);
                            });
                            resp
                        });
                        if r.inner.clicked() && *id != 0 {
                            select = Some(*id);
                        }
                    }
                }
                let untouched = &v.untouched[ti];
                let mesh_changed = d.meshes.iter().any(|m| m.status() != "same");
                if (status == "changed" || (status == "same" && mesh_changed)) && !untouched.is_empty() {
                    ui.add_space(4.0);
                    ui.label(RichText::new("Untouched").size(12.0).color(theme::FAINT))
                        .on_hover_text("Parts where no pixel, vertex or UV changed: B keeps A's version exactly.");
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = Vec2::new(6.0, 6.0);
                        for &id in untouched {
                            let sel = self.selection == Selection::Part(id);
                            let text = RichText::new(title_case(&self.part_name(id))).size(12.5).color(if sel { theme::BG } else { theme::GREEN });
                            let b = egui::Button::new(text)
                                .fill(if sel { theme::GREEN } else { theme::GREEN.gamma_multiply(0.14) })
                                .stroke(Stroke::new(1.0, theme::GREEN.gamma_multiply(0.5)))
                                .corner_radius(CornerRadius::same(10))
                                .small();
                            if ui.add(b).on_hover_text("Click to light it up on both models").clicked() {
                                select = Some(id);
                            }
                        }
                    });
                }
                if self.advanced {
                    egui::Grid::new(("texadv", ti)).num_columns(2).show(ui, |ui| {
                        let sz = |s: Option<[usize; 2]>| s.map(|s| format!("{} x {}", s[0], s[1])).unwrap_or("-".into());
                        kv(ui, "A size", sz(t.a_size));
                        kv(ui, "B size", sz(t.b_size));
                        kv(ui, "Area", t.bbox.map(|b| format!("x {}..{}, y {}..{}", b[0], b[2], b[1], b[3])).unwrap_or("-".into()));
                    });
                }
            });
            if let Some(id) = select {
                self.selection = if self.selection == Selection::Part(id) { Selection::None } else { Selection::Part(id) };
            }
        }
        // Meshes, skeleton, clips.
        card(ui, "Shape", |ui| {
            egui::Grid::new("meshes").num_columns(2).spacing([10.0, 5.0]).show(ui, |ui| {
                for m in &d.meshes {
                    let st = m.status();
                    ui.label(RichText::new(mesh_title(&m.name)).strong());
                    ui.horizontal(|ui| {
                        pill(ui, st, if st == "same" { theme::GREEN } else { theme::GOLD });
                        let vt = |s: Option<(usize, usize)>| s.map(|s| format!("{} tris", num(s.1))).unwrap_or("-".into());
                        if st != "same" {
                            ui.label(RichText::new(format!("{} → {}", vt(m.a), vt(m.b))).color(theme::WEAK).size(12.5));
                        }
                    });
                    ui.end_row();
                    if self.advanced && st == "changed" {
                        ui.label("");
                        ui.label(RichText::new(format!("{} moved vertices, {} new UVs, triangles {}", m.moved_vertices, m.uv_changed_vertices, if m.triangles_same { "same" } else { "differ" })).size(12.0).color(theme::FAINT));
                        ui.end_row();
                    }
                }
                ui.label(RichText::new("Skeleton").strong());
                pill(ui, if d.joints_same { "same" } else { "changed" }, if d.joints_same { theme::GREEN } else { theme::GOLD });
                ui.end_row();
                ui.label(RichText::new("Clips").strong());
                let same = d.clips[0] == d.clips[1];
                pill(ui, if same { "same" } else { "changed" }, if same { theme::GREEN } else { theme::GOLD }).on_hover_text(format!("A: {}\nB: {}", d.clips[0].join(", "), d.clips[1].join(", ")));
                ui.end_row();
            });
        });
        card(ui, "Files", |ui| {
            egui::Grid::new("files").num_columns(2).spacing([10.0, 4.0]).show(ui, |ui| {
                kv(ui, "A", format!("{} · {}", a.label, mb(d.bytes[0])));
                kv(ui, "B", format!("{} · {}", b.label, mb(d.bytes[1])));
                if self.advanced {
                    kv(ui, "Opened in", format!("{} ms, {} ms", a.load_ms, b.load_ms));
                    kv(ui, "Compared in", format!("{} ms", v.ms));
                }
            });
            if self.advanced {
                for (k, p) in [("A", &a.path), ("B", &b.path)] {
                    ui.add(egui::Label::new(RichText::new(format!("{k}: {}", p.display())).size(11.0).monospace().color(theme::FAINT)).wrap());
                }
            }
        });
        if self.advanced {
            card(ui, "Tolerance", |ui| {
                ui.label(RichText::new("Channel change still counted as the same (0 = exact).").color(theme::WEAK).size(12.5));
                ui.add(egui::Slider::new(&mut self.tol, 0..=32));
            });
        }
    }

    // ---------- Inspect ----------

    fn inspect_page(&mut self, ctx: &egui::Context) {
        let docref = self.inspect.clone();
        egui::SidePanel::right("inspect-info").default_width(330.0).min_width(280.0).resizable(true).frame(egui::Frame::new().fill(theme::PANEL).inner_margin(Margin::same(12))).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                if let Some(d) = &docref {
                    self.inspect_panel(ui, d);
                }
            });
        });
        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG).inner_margin(Margin::same(14))).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Inspect").family(FontFamily::Name("strong".into())).size(22.0));
                ui.add_space(16.0);
                self.file_picker(ui, ctx, Slot::Inspect, docref.as_ref());
                if let Some(parts) = self.parts.clone() {
                    let current = match self.selection {
                        Selection::Part(p) => title_case(parts.name(p)),
                        _ => "Pick a part".into(),
                    };
                    let mut pick = None;
                    egui::ComboBox::from_id_salt("part").selected_text(current).width(190.0).height(420.0).show_ui(ui, |ui| {
                        for (id, name) in &parts.names {
                            if ui.selectable_label(self.selection == Selection::Part(*id), title_case(name)).clicked() {
                                pick = Some(*id);
                            }
                        }
                    });
                    if let Some(id) = pick {
                        self.selection = Selection::Part(id);
                        self.follow_selection();
                    }
                }
                if ui.add_enabled(self.selection != Selection::None, egui::Button::new("Clear")).clicked() {
                    self.selection = Selection::None;
                }
            });
            ui.add_space(6.0);
            let Some(d) = docref else {
                ui.centered_and_justified(|ui| {
                    ui.add(egui::Spinner::new().size(32.0).color(theme::GOLD));
                });
                return;
            };
            ui.horizontal(|ui| {
                row_label(ui, "Animation");
                self.playback(ui, Some(d.clone()));
            });
            ui.horizontal(|ui| {
                row_label(ui, "View");
                self.size_controls(ui);
                ui.separator();
                let dc = if self.show_dead { theme::RED } else { theme::WEAK };
                ui.toggle_value(&mut self.show_dead, RichText::new("● Dead").color(dc)).on_hover_text("Light up triangles too small to show paint (D)");
                let tc = if self.show_touch { theme::TEAL } else { theme::WEAK };
                ui.toggle_value(&mut self.show_touch, RichText::new("● Touch").color(tc))
                    .on_hover_text("What will this touch: light up every surface that shows the selected part's pixels. Surfaces outside the part turn teal (T)");
                let rc = if self.show_reuse { Color32::from_rgb(255, 165, 60) } else { theme::WEAK };
                ui.toggle_value(&mut self.show_reuse, RichText::new("● Reuse").color(rc))
                    .on_hover_text("Color texture pixels that show on two surfaces: blue = mirrored twin, orange = shared by two pieces, red = a piece overlaps itself (U)");
                ui.separator();
                self.view_buttons(ui);
            });
            ui.add_space(8.0);
            let (area, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
            let gap = 12.0;
            let tex_w = (area.width() * 0.46).min(area.height() + 40.0);
            let rect_v = Rect::from_min_size(area.min, Vec2::new(area.width() - tex_w - gap, area.height()));
            let rect_t = Rect::from_min_size(Pos2::new(rect_v.max.x + gap, area.min.y), Vec2::new(tex_w, area.height()));
            if let Some(yaw) = self.reframe.take() {
                let mut c = Camera::fit(&d.model, rect_v.width() / rect_v.height().max(1.0), 0.8, false);
                c.yaw = yaw;
                self.cam = c;
            }
            let dur = self.clip_duration(&d);
            self.advance_time(ctx, dur);
            let ci = self.clip.as_deref().and_then(|c| d.clip_index(c));
            let caption = self.view_caption(&d, self.clip.is_some() && ci.is_none());
            let touched = self.touched(&d);
            let (sel, show_dead, show_touch) = (self.selection, self.show_dead, self.show_touch);
            let tint = |mi: usize, ti: usize| -> Option<[u8; 3]> {
                if show_dead && d.dead[mi][ti] {
                    return Some(theme::HL_DEAD);
                }
                let hit = touched.as_ref().is_some_and(|t| t[mi][ti]);
                match sel {
                    Selection::Region(..) => (!hit).then_some(theme::HL_DIM),
                    Selection::Part(p) if show_touch => {
                        if !hit {
                            Some(theme::HL_DIM)
                        } else if d.tri_parts[mi][ti] != p {
                            Some(theme::HL_TOUCH)
                        } else {
                            None
                        }
                    }
                    Selection::Part(p) if d.tri_parts[mi][ti] != p => Some(theme::HL_DIM),
                    Selection::Tri(m, t) if m == mi && t == ti => Some(theme::HL_SELECT),
                    _ => None,
                }
            };
            let tkey = self.tint_key(0);
            let (pose, pose_key) = self.pose_i.get(&d, ci, self.time);
            let sim = self.sim_click.take();
            let shown = self.vp_inspect.show(ui, rect_v, egui::Id::new("inspect-view"), &mut self.cam, &self.view, Draw { doc: &d, frame: &d.model, pose, pose_key, tint: &tint, tint_key: tkey, caption, sim_click: sim });
            self.hint = "Drag to turn · Click the model or the texture to pick a part · Drag a box on the texture to see where paint shows · F1 for help".into();
            if let Some((mi, ti)) = shown.hovered {
                let p = d.tri_parts[mi][ti];
                let mut text = if p != 0 { title_case(&self.part_name(p)) } else { "No part".into() };
                if d.dead[mi][ti] {
                    text.push_str("  ·  dead triangle");
                }
                if self.advanced {
                    text.push_str(&format!("  ·  {} tri {ti}", d.model.meshes[mi].name));
                }
                hover_tag(ui, &text, theme::LINE);
            }
            if let Some((mi, ti)) = shown.clicked {
                let p = d.tri_parts[mi][ti];
                self.selection = if p != 0 { Selection::Part(p) } else { Selection::Tri(mi, ti) };
                self.follow_selection();
            } else if shown.clicked_empty {
                self.selection = Selection::None;
            }
            self.texture_view(ui, ctx, &d, rect_t);
        });
    }

    /// Shows the texture that holds most of the selection.
    fn follow_selection(&mut self) {
        let Some(d) = &self.inspect else { return };
        let count = |img: usize| -> usize {
            (0..d.model.meshes.len())
                .filter(|&mi| d.model.meshes[mi].image == Some(img))
                .map(|mi| match self.selection {
                    Selection::Part(p) => d.tri_parts[mi].iter().filter(|&&x| x == p).count(),
                    Selection::Tri(m, _) => (m == mi) as usize,
                    Selection::None | Selection::Region(..) => 0,
                })
                .sum()
        };
        if let Some(best) = (0..d.model.images.len()).max_by_key(|&i| count(i)) {
            if count(best) > 0 {
                self.inspect_image = best;
            }
        }
    }

    fn texture_view(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, d: &Arc<Doc>, rect: Rect) {
        let mut ui = ui.new_child(egui::UiBuilder::new().max_rect(rect));
        let ui = &mut ui;
        ui.horizontal(|ui| {
            for (i, img) in d.model.images.iter().enumerate() {
                let name = img.name.trim_end_matches(".png");
                if ui.selectable_label(self.inspect_image == i, texture_title(name)).on_hover_text(name).clicked() {
                    self.inspect_image = i;
                }
            }
        });
        let ii = self.inspect_image.min(d.model.images.len().saturating_sub(1));
        let Some(h) = self.texture(ctx, Arc::as_ptr(d) as usize, ii, 0, d.previews.get(ii).and_then(|p| p.as_ref())) else {
            ui.label("No texture");
            return;
        };
        let s = (ui.available_width().min(ui.available_height() - 22.0)).max(64.0);
        let (r, resp) = ui.allocate_exact_size(Vec2::splat(s), Sense::click_and_drag());
        let painter = ui.painter_at(r);
        painter.rect_filled(r, CornerRadius::same(10), Color32::from_rgb(0x12, 0x14, 0x18));
        painter.image(h.id(), r, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        if self.show_reuse {
            if let Some(o) = self.texture(ctx, Arc::as_ptr(d) as usize, ii, 3, d.overlap_previews.get(ii).and_then(|p| p.as_ref())) {
                painter.rect_filled(r, CornerRadius::same(10), Color32::from_black_alpha(120));
                painter.image(o.id(), r, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
            }
        }
        let touched = self.touched(d);
        let fc = Color32::from_rgba_unmultiplied(0xe9, 0xb9, 0x49, 110);
        let tc = Color32::from_rgba_unmultiplied(0x3f, 0xd0, 0xe0, 120);
        let edge = Stroke::new(1.0, theme::GOLD);
        let mut fill = egui::Mesh::default();
        for (mi, m) in d.model.meshes.iter().enumerate() {
            if m.image != Some(ii) {
                continue;
            }
            let at = |v: u32| r.min + Vec2::new(m.uv[v as usize][0], m.uv[v as usize][1]) * s;
            for (ti, t) in m.tris.iter().enumerate() {
                let hit = touched.as_ref().is_some_and(|x| x[mi][ti]);
                let (selected, color) = match self.selection {
                    Selection::Part(p) if d.tri_parts[mi][ti] == p => (true, fc),
                    Selection::Part(_) => (self.show_touch && hit, tc),
                    Selection::Region(..) => (hit, fc),
                    Selection::Tri(a, b) => (a == mi && b == ti, fc),
                    Selection::None => (false, fc),
                };
                if selected {
                    let base = fill.vertices.len() as u32;
                    for &v in t {
                        fill.colored_vertex(at(v), color);
                    }
                    fill.add_triangle(base, base + 1, base + 2);
                    if color == fc {
                        for k in 0..3 {
                            painter.line_segment([at(t[k]), at(t[(k + 1) % 3])], edge);
                        }
                    }
                }
                if self.show_dead && d.dead[mi][ti] {
                    let c = (at(t[0]).to_vec2() + at(t[1]).to_vec2() + at(t[2]).to_vec2()) / 3.0;
                    painter.circle_stroke(c.to_pos2(), 4.0, Stroke::new(1.5, theme::RED));
                }
            }
        }
        painter.add(egui::Shape::mesh(fill));
        painter.rect_stroke(r, CornerRadius::same(10), Stroke::new(1.0, theme::LINE), egui::StrokeKind::Inside);
        let uv_at = |p: Pos2| (((p.x - r.min.x) / s).clamp(0.0, 1.0), ((p.y - r.min.y) / s).clamp(0.0, 1.0));
        let atlas = d.model.images[ii].name.trim_end_matches(".png").to_owned();
        // Drag a box: "if I paint here, where does it show?"
        if resp.drag_started() {
            self.drag_from = resp.interact_pointer_pos().map(uv_at);
        }
        let box_of = |a: (f32, f32), b: (f32, f32)| [a.0.min(b.0), a.1.min(b.1), a.0.max(b.0), a.1.max(b.1)];
        if let (Some(a), Some(p)) = (self.drag_from, resp.interact_pointer_pos()) {
            let bx = box_of(a, uv_at(p));
            if resp.drag_stopped() {
                self.drag_from = None;
                if (bx[2] - bx[0]) * s > 3.0 && (bx[3] - bx[1]) * s > 3.0 {
                    self.selection = Selection::Region(ii, bx);
                }
            } else if resp.dragged() {
                let rr = Rect::from_min_max(r.min + Vec2::new(bx[0], bx[1]) * s, r.min + Vec2::new(bx[2], bx[3]) * s);
                painter.rect_stroke(rr, CornerRadius::ZERO, Stroke::new(1.5, Color32::WHITE), egui::StrokeKind::Middle);
            }
        }
        if let Selection::Region(img, bx) = self.selection {
            if img == ii {
                let rr = Rect::from_min_max(r.min + Vec2::new(bx[0], bx[1]) * s, r.min + Vec2::new(bx[2], bx[3]) * s);
                painter.rect_stroke(rr, CornerRadius::ZERO, Stroke::new(2.0, Color32::WHITE), egui::StrokeKind::Middle);
            }
        }
        if self.show_reuse {
            ui.horizontal(|ui| {
                for (c, t) in doc::REUSE_COLORS.iter().zip(["Mirrored twin", "Shared by two pieces", "Overlaps itself"]) {
                    ui.label(RichText::new("■").color(Color32::from_rgb(c[0], c[1], c[2])));
                    ui.label(RichText::new(t).small().color(theme::WEAK));
                }
            });
        } else {
            ui.label(RichText::new("Drag a box on the texture to see where paint there shows.").small().color(theme::FAINT));
        }
        if let Some(p) = resp.hover_pos() {
            let (u, v) = uv_at(p);
            let id = self.parts.as_ref().map(|ps| ps.id_at(&atlas, u, v)).unwrap_or(0);
            if id != 0 {
                hover_tag(ui, &title_case(&self.part_name(id)), theme::LINE);
            }
        }
        if resp.clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                let (u, v) = uv_at(p);
                let id = self.parts.as_ref().map(|ps| ps.id_at(&atlas, u, v)).unwrap_or(0);
                self.selection = if id != 0 { Selection::Part(id) } else { Selection::None };
            }
        }
    }

    fn inspect_panel(&mut self, ui: &mut egui::Ui, d: &Arc<Doc>) {
        ui.label(RichText::new(&d.label).family(FontFamily::Name("strong".into())).size(19.0));
        ui.label(RichText::new(d.file_name()).monospace().size(11.5).color(theme::FAINT));
        ui.add_space(4.0);
        card(ui, "Selection", |ui| match self.selection {
            Selection::None => {
                ui.label(RichText::new("Click the model or the texture to pick a part.").color(theme::WEAK));
            }
            Selection::Part(p) => {
                ui.label(RichText::new(title_case(&self.part_name(p))).strong().size(17.0).color(theme::GOLD));
                for (mi, m) in d.model.meshes.iter().enumerate() {
                    let n = d.tri_parts[mi].iter().filter(|&&x| x == p).count();
                    if n > 0 {
                        let dead = (0..m.tris.len()).filter(|&ti| d.tri_parts[mi][ti] == p && d.dead[mi][ti]).count();
                        ui.label(format!("{} triangles on the {}", num(n), mesh_title(&m.name).to_lowercase()));
                        if dead > 0 {
                            ui.label(RichText::new(format!("{dead} of them dead (too small to show paint)")).color(theme::RED));
                        }
                    }
                }
            }
            Selection::Tri(mi, ti) => {
                ui.label(RichText::new(format!("Triangle {ti}")).strong().size(17.0));
                ui.label(RichText::new(format!("On the {}, not inside a named part.", mesh_title(&d.model.meshes[mi].name).to_lowercase())).color(theme::WEAK));
            }
            Selection::Region(img, bx) => {
                ui.label(RichText::new("Box on the texture").strong().size(17.0).color(theme::GOLD));
                let name = d.model.images.get(img).map(|i| texture_title(i.name.trim_end_matches(".png"))).unwrap_or_default();
                let sz = d.textures.get(img).and_then(|t| t.as_ref()).map(|t| t.w).unwrap_or(0) as f32;
                ui.label(RichText::new(format!("{name}, {:.0} x {:.0} px", (bx[2] - bx[0]) * sz, (bx[3] - bx[1]) * sz)).color(theme::WEAK));
            }
        });
        self.uv_cards(ui, d);
        self.relayout_card(ui, d);
        card(ui, "Model", |ui| {
            egui::Grid::new("facts").num_columns(2).spacing([10.0, 4.0]).show(ui, |ui| {
                kv(ui, "Triangles", num(d.model.triangle_count()));
                kv(ui, "Joints", d.model.joint_names.len().to_string());
                kv(ui, "Clips", d.model.clips.iter().map(|c| title_case(c)).collect::<Vec<_>>().join(", "));
                kv(ui, "Dead triangles", RichText::new(d.dead_total().to_string()).color(if d.dead_total() > 0 { theme::RED } else { theme::TEXT }));
                if !d.dead_regions.is_empty() {
                    kv(ui, "", RichText::new(d.dead_regions.iter().map(|r| format!("{} {}", r.1, r.0)).collect::<Vec<_>>().join(", ")).color(theme::WEAK));
                }
                kv(ui, "File size", mb(d.model.file_len));
                if self.advanced {
                    kv(ui, "Data arrays", d.model.accessors.to_string());
                    kv(ui, "Nodes", d.model.nodes.to_string());
                    kv(ui, "Opened in", format!("{} ms", d.load_ms));
                }
            });
        });
        card(ui, "Textures", |ui| {
            egui::Grid::new("texs").num_columns(2).spacing([10.0, 4.0]).show(ui, |ui| {
                for (i, img) in d.model.images.iter().enumerate() {
                    let sz = d.textures[i].as_ref().map(|t| format!("{} x {}", t.w, t.h)).unwrap_or("unreadable".into());
                    kv(ui, &texture_title(img.name.trim_end_matches(".png")), sz);
                }
            });
        });
        if self.advanced {
            card(ui, "Meshes", |ui| {
                egui::Grid::new("ms").num_columns(2).spacing([10.0, 4.0]).show(ui, |ui| {
                    for m in &d.model.meshes {
                        kv(ui, &m.name, format!("{} verts, {} tris", num(m.pos.len()), num(m.tris.len())));
                    }
                });
            });
        }
    }

    /// "What will this touch" and the UV facts for the selection.
    fn uv_cards(&mut self, ui: &mut egui::Ui, d: &Arc<Doc>) {
        let Some(touched) = self.touched(d) else { return };
        let part = match self.selection {
            Selection::Part(p) => Some(p),
            _ => None,
        };
        let parts = self.parts.clone();
        let pname = |id: u8| if id == 0 { "no part".to_owned() } else { parts.as_ref().map(|p| p.name(id).replace('-', " ")).unwrap_or_else(|| format!("part {id}")) };
        // Surfaces that show the pixels, grouped by part; for a part, the ones outside it.
        let mut by_part: Vec<(u8, usize)> = Vec::new();
        let mut total = 0;
        let mut sides = [0usize; 2];
        for (mi, m) in d.model.meshes.iter().enumerate() {
            for ti in 0..m.tris.len() {
                if !touched[mi][ti] || d.uv.topo[mi].dup_of[ti].is_some() {
                    continue;
                }
                total += 1;
                sides[(uv::center(m, ti)[0] > 0.0) as usize] += 1;
                let p = d.tri_parts[mi][ti];
                if Some(p) != part {
                    match by_part.iter_mut().find(|e| e.0 == p) {
                        Some(e) => e.1 += 1,
                        None => by_part.push((p, 1)),
                    }
                }
            }
        }
        by_part.sort_by(|a, b| b.1.cmp(&a.1));
        card(ui, "What will this touch", |ui| {
            let what = if part.is_some() { "Paint on this part's pixels" } else { "Paint in this box" };
            ui.label(format!("{what} shows on {} triangles.", num(total)));
            if sides[0] > 0 && sides[1] > 0 {
                ui.label(RichText::new(format!("Both sides of the body: {} on the character's right, {} on the left.", num(sides[1]), num(sides[0]))).color(theme::WEAK));
            }
            if part.is_some() {
                if by_part.is_empty() {
                    ui.label(RichText::new("Nothing outside the part.").color(theme::GREEN));
                } else {
                    let list: Vec<String> = by_part.iter().map(|(p, n)| format!("{} ({n})", pname(*p))).collect();
                    ui.label(RichText::new(format!("Also on: {}", list.join(", "))).color(theme::TEAL))
                        .on_hover_text("Triangles of other parts that cross into this part's pixels. Turn on Touch (T) to see them in teal.");
                }
            } else if !by_part.is_empty() {
                let list: Vec<String> = by_part.iter().map(|(p, n)| format!("{} ({n})", pname(*p))).collect();
                ui.label(RichText::new(format!("Parts: {}", list.join(", "))).color(theme::WEAK));
            }
        });
        let set = |mi: usize, ti: usize| match part {
            Some(p) => d.tri_parts[mi][ti] == p,
            None => touched[mi][ti],
        };
        let s = d.uv.summary(&d.model, &set);
        card(ui, "UV layout", |ui| {
            if let Some((lo, med, hi, size)) = s.density {
                let spread = hi / lo.max(1e-3);
                ui.label(format!("Pixel density {}–{} per unit at {size} px (median {}).", num(lo.round() as usize), num(hi.round() as usize), num(med.round() as usize)))
                    .on_hover_text("Texture pixels per world unit of surface. Higher means sharper paint.");
                if spread >= 1.5 {
                    ui.label(RichText::new(format!("Detail varies {spread:.1}x: a mark drawn one size on the texture comes out up to {spread:.1}x larger on the model.")).color(theme::WEAK));
                }
            }
            let [mir, shared, own] = s.reuse;
            let line = |ui: &mut egui::Ui, k: usize, text: String, tip: &str| {
                let c = doc::REUSE_COLORS[k];
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("■").color(Color32::from_rgb(c[0], c[1], c[2])));
                    ui.label(text).on_hover_text(tip);
                });
            };
            if mir > 0 {
                line(ui, 0, format!("Mirrored twin: {} texels serve both sides", num(mir)), "The left and right copies share one paint. Expected.");
            }
            if shared > 0 {
                line(ui, 1, format!("Shared: {} texels show on two pieces", num(shared)), "Two different pieces on the same side use these pixels.");
            }
            if own > 0 {
                line(ui, 2, format!("Overlaps itself: {} {} twice", num(own), if own == 1 { "texel shows" } else { "texels show" }), "One piece lies on top of itself in the texture (a fold, or a cut like the waist's).");
            }
            if mir + shared + own > 0 {
                ui.label(RichText::new(format!("Counted at {0} x {0}. Turn on Reuse (U) to see them.", uv::OVERLAP_SIZE)).small().color(theme::FAINT));
            }
            if !s.islands.is_empty() {
                ui.add_space(4.0);
                ui.label(RichText::new("Islands").color(theme::FAINT));
                for &(mi, i, n) in s.islands.iter().take(8) {
                    let isl = &d.uv.islands[mi][i as usize];
                    let mut t = format!("{} · {} of {} triangles", first_upper(&isl.label), n, isl.tris);
                    if isl.side != "center" && !isl.label.contains(isl.side) {
                        t.push_str(&format!(" · character's {}", isl.side));
                    }
                    let r = ui.label(t);
                    if let Some(tw) = isl.twin {
                        r.on_hover_text(format!("Mirror twin of {}", d.uv.islands[mi][tw as usize].label));
                    }
                }
                if s.islands.len() > 8 {
                    ui.label(RichText::new(format!("and {} more", s.islands.len() - 8)).color(theme::FAINT));
                }
            }
        });
    }

    // ---------- unique layout ----------

    fn start_relayout(&mut self, ctx: &egui::Context, input: PathBuf, dir: PathBuf) {
        self.relayout = None;
        self.relayout_error = None;
        self.relayout_progress = "Starting".into();
        self.relayout_running = true;
        doc::spawn_relayout(self.tx.clone(), ctx.clone(), input, dir, self.relayout_size, self.relayout_blender);
    }

    /// Gives every surface its own pixels on a bigger sheet (the built-in sewing pattern, or
    /// Blender's dense layout), then checks the new file against this one.
    fn relayout_card(&mut self, ui: &mut egui::Ui, d: &Arc<Doc>) {
        let ctx = ui.ctx().clone();
        let mut start: Option<PathBuf> = None;
        let mut compare: Option<(PathBuf, PathBuf)> = None;
        card(ui, "Unique layout", |ui| {
            ui.label(RichText::new("Gives every surface its own pixels on a bigger texture, and copies the paint across. The original file is not changed.").color(theme::WEAK));
            if self.relayout_running {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new().color(theme::GOLD));
                    ui.label(RichText::new(&self.relayout_progress).size(12.5));
                });
                let note = if self.relayout_blender { "Blender runs hidden at low priority. This takes a few minutes." } else { "Built in; runs in the background. Under a minute for a field model." };
                ui.label(RichText::new(note).small().color(theme::FAINT));
                return;
            }
            if let Some(e) = &self.relayout_error {
                ui.label(RichText::new(e).color(theme::RED));
            }
            let mine = self.relayout.as_ref().filter(|r| r.input == d.path).cloned();
            if let Some(r) = &mine {
                let c = &r.checks;
                if c.passed() {
                    pill(ui, "Checked: unique layout, looks the same", theme::GREEN)
                        .on_hover_text("Same skeleton, clips and triangles; no texel shows on two surfaces; renders from four sides match the original.");
                } else {
                    pill(ui, "Checks failed", theme::RED);
                }
                ui.add_space(4.0);
                egui::Grid::new("relayout-kv").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                    for m in &c.meshes {
                        let name = texture_title(m.image.trim_end_matches(".png"));
                        let gain = m.density_median[1] / m.density_median[0].max(1e-3);
                        kv(ui, &name, format!("{} → {} px, {gain:.1}x sharper", m.texture_px[0][0], m.texture_px[1][0]));
                        let shared = format!("{} → {}", num(m.shared_texels[0]), num(m.shared_texels[1]));
                        kv(ui, "  Texels on two surfaces", RichText::new(shared).color(if m.unplanned_shared() <= wb_core::relayout::MAX_SHARED_AFTER { theme::GREEN } else { theme::RED }));
                        if m.kept_shared > 0 {
                            kv(ui, "  Kept at stock on purpose", num(m.kept_shared));
                        }
                    }
                    let worst = c.views.iter().map(|v| v.mean_diff).fold(0.0f32, f32::max);
                    kv(ui, "Render difference", format!("{worst:.1} of 255 at worst ({} views)", c.views.len()));
                    if let Some((pieces, seams, _)) = &r.pattern {
                        kv(ui, "Sewing pattern", format!("{pieces} pieces, {seams} numbered seams"));
                    }
                });
                ui.label(RichText::new(r.glb.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()).monospace().size(11.5).color(theme::FAINT)).on_hover_text(r.glb.display().to_string());
                ui.horizontal(|ui| {
                    if ui.button("Open in Compare").on_hover_text("This file as A, the new layout as B").clicked() {
                        compare = Some((r.input.clone(), r.glb.clone()));
                    }
                    if ui.button("Open folder").clicked() {
                        let _ = std::process::Command::new("explorer").arg(&r.dir).spawn();
                    }
                    if let Some((_, _, sheets)) = &r.pattern {
                        for s in sheets {
                            let label = s.file_stem().map(|f| f.to_string_lossy().trim_end_matches("_pattern").to_owned()).unwrap_or_default();
                            if ui.button(format!("Pattern {}", texture_title(&label))).on_hover_text("The sewing-pattern sheet: piece outlines, part names, numbered seam pairs, paint borders").clicked() {
                                let _ = std::process::Command::new("explorer").arg(s).spawn();
                            }
                        }
                    }
                });
                ui.add_space(6.0);
            }
            ui.horizontal(|ui| {
                ui.label(RichText::new("Style").color(theme::WEAK));
                ui.selectable_value(&mut self.relayout_blender, false, "Sewing pattern").on_hover_text("Built in: pieces cut by body part, upright, left/right pairs side by side, numbered seams");
                ui.selectable_value(&mut self.relayout_blender, true, "Dense (Blender)").on_hover_text("Blender's tight packing; needs Blender installed");
            });
            ui.horizontal(|ui| {
                ui.label(RichText::new("Size").color(theme::WEAK));
                for s in [2048usize, 4096, 8192] {
                    ui.selectable_value(&mut self.relayout_size, s, format!("{}K", s / 1024));
                }
            });
            let go = egui::Button::new(RichText::new("Make unique layout...").color(Color32::from_rgb(0x1a, 0x1a, 0x1a))).fill(theme::GOLD);
            if ui.add(go).on_hover_text("Pick an empty folder for the new file. Picking a finished layout folder only re-checks it.").clicked() {
                let mut dlg = rfd::FileDialog::new().set_title("Choose an empty folder for the new layout");
                if let Some(p) = d.path.parent() {
                    dlg = dlg.set_directory(p);
                }
                start = dlg.pick_folder();
            }
        });
        if let Some(dir) = start {
            self.start_relayout(&ctx, d.path.clone(), dir);
        }
        if let Some((a, b)) = compare {
            self.load(&ctx, Slot::A, a);
            self.load(&ctx, Slot::B, b);
            self.page = Page::Compare;
            self.selection = Selection::None;
        }
    }

    // ---------- safe swap ----------

    fn swap_page(&mut self, ctx: &egui::Context) {
        let base = self.inspect.clone();
        // A PNG dropped anywhere on the window becomes the painted texture.
        let dropped = ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).find(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("png"))));
        if let Some(p) = dropped {
            self.swap_png = Some(p);
        }
        if let Some(b) = &base {
            if self.swap_image.as_ref().is_none_or(|n| !b.model.images.iter().any(|i| i.name.trim_end_matches(".png") == n)) {
                let first = b.model.images.iter().find(|i| i.name.starts_with("98_1")).or(b.model.images.first());
                self.swap_image = first.map(|i| i.name.trim_end_matches(".png").to_owned());
            }
            // Start the swap once per base file, texture and PNG.
            if let (Some(png), Some(img)) = (&self.swap_png, &self.swap_image) {
                let key = (Arc::as_ptr(b) as usize, img.clone(), png.clone());
                if self.swap_started.as_ref() != Some(&key) {
                    self.swap_started = Some(key);
                    self.swap = None;
                    self.swap_error = None;
                    self.swap_saved = None;
                    self.swap_running = true;
                    doc::spawn_swap(self.tx.clone(), ctx.clone(), b.clone(), img.clone(), png.clone(), self.parts.clone());
                }
            }
        }
        let result = self.swap.clone().filter(|r| base.as_ref().is_some_and(|b| Arc::ptr_eq(&r.base, b)));

        egui::SidePanel::right("swap-info").default_width(340.0).min_width(290.0).resizable(true).frame(egui::Frame::new().fill(theme::PANEL).inner_margin(Margin::same(12))).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.swap_panel(ui, ctx, base.as_ref(), result.as_ref()));
        });
        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG).inner_margin(Margin::same(14))).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Safe swap").family(FontFamily::Name("strong".into())).size(22.0));
                ui.add_space(12.0);
                ui.label(RichText::new("Put a painted texture into a copy of the file. The original is never changed.").color(theme::WEAK));
            });
            ui.add_space(6.0);
            let Some(b) = base.clone() else {
                ui.centered_and_justified(|ui| {
                    ui.add(egui::Spinner::new().size(32.0).color(theme::GOLD));
                });
                return;
            };
            ui.horizontal(|ui| {
                row_label(ui, "Animation");
                self.playback(ui, Some(b.clone()));
            });
            ui.horizontal(|ui| {
                row_label(ui, "View");
                self.size_controls(ui);
                ui.separator();
                ui.add_enabled_ui(result.is_some(), |ui| {
                    segmented(
                        ui,
                        &mut self.swap_view,
                        &[
                            (SwapView::After, "After", "The new file"),
                            (SwapView::Before, "Before", "The file you started from"),
                            (SwapView::Changes, "Changes", "Where the new paint differs, in pink"),
                        ],
                    );
                });
                ui.separator();
                self.view_buttons(ui);
            });
            ui.add_space(8.0);
            let (area, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
            let gap = 12.0;
            let tex_w = (area.width() * 0.46).min(area.height() + 40.0);
            let rect_v = Rect::from_min_size(area.min, Vec2::new(area.width() - tex_w - gap, area.height()));
            let rect_t = Rect::from_min_size(Pos2::new(rect_v.max.x + gap, area.min.y), Vec2::new(tex_w, area.height()));
            if let Some(yaw) = self.reframe.take() {
                let mut c = Camera::fit(&b.model, rect_v.width() / rect_v.height().max(1.0), 0.8, false);
                c.yaw = yaw;
                self.cam = c;
            }
            let view = if result.is_some() { self.swap_view } else { SwapView::Before };
            let d = match (&result, view) {
                (Some(r), SwapView::After | SwapView::Changes) => r.doc.clone(),
                _ => b.clone(),
            };
            let dur = self.clip_duration(&d);
            self.advance_time(ctx, dur);
            let ci = self.clip.as_deref().and_then(|c| d.clip_index(c));
            let caption = self.view_caption(&d, self.clip.is_some() && ci.is_none());
            let changed = result.as_ref().filter(|_| view == SwapView::Changes).map(|r| &r.outcome.changed);
            let tint = |mi: usize, ti: usize| -> Option<[u8; 3]> {
                let c = changed?;
                if c.get(mi).and_then(|m| m.get(ti)).copied().unwrap_or(false) {
                    Some(theme::HL_CHANGED)
                } else {
                    Some(theme::HL_DIM)
                }
            };
            let (pose, pose_key) = self.pose_s.get(&d, ci, self.time);
            let tkey = 5000 + view as u64;
            self.vp_swap.show(ui, rect_v, egui::Id::new("swap-view"), &mut self.cam, &self.view, Draw { doc: &d, frame: &d.model, pose, pose_key, tint: &tint, tint_key: tkey, caption, sim_click: None });
            self.hint = "Drop a painted PNG on the window or choose one on the right · Drag to turn · F1 for help".into();

            // Texture panel.
            let mut tui = ui.new_child(egui::UiBuilder::new().max_rect(rect_t));
            let ii = b.model.images.iter().position(|i| Some(i.name.trim_end_matches(".png")) == self.swap_image.as_deref()).unwrap_or(0);
            let title = b.model.images.get(ii).map(|i| texture_title(i.name.trim_end_matches(".png"))).unwrap_or_default();
            tui.label(RichText::new(format!("{title} · {}", match view { SwapView::After => "after", SwapView::Before => "before", SwapView::Changes => "changes" })).color(theme::WEAK));
            let handle = match (&result, view) {
                (Some(r), SwapView::Changes) => self.texture(ctx, Arc::as_ptr(&r.doc) as usize, r.image, 5, r.overlay.as_ref()),
                (Some(r), SwapView::After) => self.texture(ctx, Arc::as_ptr(&r.doc) as usize, r.image, 0, r.doc.previews.get(r.image).and_then(|p| p.as_ref())),
                _ => self.texture(ctx, Arc::as_ptr(&b) as usize, ii, 0, b.previews.get(ii).and_then(|p| p.as_ref())),
            };
            let s = (tui.available_width().min(tui.available_height() - 22.0)).max(64.0);
            let (r, _) = tui.allocate_exact_size(Vec2::splat(s), Sense::hover());
            let painter = tui.painter_at(r);
            painter.rect_filled(r, CornerRadius::same(10), Color32::from_rgb(0x12, 0x14, 0x18));
            if let Some(h) = handle {
                painter.image(h.id(), r, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
            }
            let hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());
            if self.swap_png.is_none() || hovering {
                painter.rect_filled(r, CornerRadius::same(10), Color32::from_black_alpha(if hovering { 200 } else { 150 }));
                painter.rect_stroke(r.shrink(10.0), CornerRadius::same(10), Stroke::new(2.0, if hovering { theme::GOLD } else { theme::LINE }), egui::StrokeKind::Inside);
                painter.text(r.center(), egui::Align2::CENTER_CENTER, if hovering { "Drop to swap it in" } else { "Drop a painted PNG here" }, egui::FontId::proportional(18.0), theme::TEXT);
            }
        });
    }

    fn swap_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, base: Option<&Arc<Doc>>, result: Option<&Arc<doc::SwapResult>>) {
        card(ui, "1 · Start from", |ui| {
            self.file_picker(ui, ctx, Slot::Inspect, base);
            ui.label(RichText::new("The file you want a new version of. It is only read.").size(12.0).color(theme::FAINT));
        });
        ui.add_space(8.0);
        card(ui, "2 · Texture to replace", |ui| {
            if let Some(b) = base {
                ui.horizontal_wrapped(|ui| {
                    for im in &b.model.images {
                        let n = im.name.trim_end_matches(".png").to_owned();
                        let sel = self.swap_image.as_deref() == Some(n.as_str());
                        if ui.selectable_label(sel, texture_title(&n)).on_hover_text(&n).clicked() && !sel {
                            self.swap_image = Some(n);
                        }
                    }
                });
                if let Some(t) = b.model.images.iter().position(|i| Some(i.name.trim_end_matches(".png")) == self.swap_image.as_deref()).and_then(|i| b.textures[i].as_ref()) {
                    ui.label(RichText::new(format!("Now {} x {} px. The new PNG must be the same shape; any size works.", t.w, t.h)).size(12.0).color(theme::FAINT));
                }
            }
        });
        ui.add_space(8.0);
        card(ui, "3 · Painted PNG", |ui| {
            ui.horizontal(|ui| {
                if ui.button("Choose PNG...").clicked() {
                    let mut dlg = rfd::FileDialog::new().add_filter("PNG image", &["png"]).set_title("Choose the painted texture");
                    if let Some(d) = self.swap_png.as_ref().and_then(|p| p.parent()).or_else(|| base.and_then(|b| b.path.parent())) {
                        dlg = dlg.set_directory(d);
                    }
                    if let Some(p) = dlg.pick_file() {
                        self.swap_png = Some(p);
                    }
                }
                ui.label(RichText::new("or drop one on the window").color(theme::FAINT).size(12.0));
            });
            if let Some(p) = &self.swap_png {
                ui.label(RichText::new(p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()).strong()).on_hover_text(p.display().to_string());
            }
        });
        ui.add_space(8.0);
        card(ui, "Result", |ui| {
            if self.swap_running {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new().color(theme::GOLD));
                    ui.label("Swapping and checking...");
                });
                return;
            }
            if let Some(e) = &self.swap_error {
                ui.label(RichText::new(e).color(theme::RED));
                return;
            }
            let Some(r) = result else {
                ui.label(RichText::new("Pick a PNG to see what it would change.").color(theme::WEAK));
                return;
            };
            let o = &r.outcome;
            pill(ui, "Checked: only this texture changed", theme::GREEN).on_hover_text("Every other block of bytes, the file's description (JSON) and every mesh match the original.");
            ui.add_space(4.0);
            let mb = |n: usize| format!("{:.1} MB", n as f64 / 1_048_576.0);
            egui::Grid::new("swap-kv").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                kv(ui, "Texture", texture_title(o.image.trim_end_matches(".png")));
                kv(ui, "Size", format!("{} x {}  →  {} x {}", o.old_size[0], o.old_size[1], o.new_size[0], o.new_size[1]));
                kv(ui, "File", format!("{}  →  {}", mb(r.base.model.file_len), mb(o.bytes.len())));
                kv(ui, "Changed", format!("{} px ({})", num(o.changed_pixels), pct(o.changed_pixels, o.total_pixels)));
            });
            if o.changed_pixels == 0 {
                ui.label(RichText::new("The PNG has the same pixels as the texture it replaces.").color(theme::WEAK));
            }
            if o.resized {
                ui.label(RichText::new("The new PNG is a different size, so nearly every pixel counts as changed.").size(12.0).color(theme::WEAK));
            }
            if o.alpha_flips > 0 {
                ui.label(RichText::new(format!("{} pixels switch between see-through and solid. Check edges and the hair.", num(o.alpha_flips))).color(theme::RED));
            }
            if !o.by_part.is_empty() {
                ui.add_space(6.0);
                ui.label(RichText::new("Where the new paint is").color(theme::WEAK));
                for (p, n) in o.by_part.iter().take(6) {
                    ui.label(format!("{} · {}", title_case(p), pct(*n, o.changed_pixels)));
                }
            }
            if !o.shows_on.is_empty() {
                ui.add_space(6.0);
                ui.label(RichText::new("It shows on").color(theme::WEAK));
                let line: Vec<String> = o.shows_on.iter().map(|(p, n)| format!("{} ({n})", title_case(p).to_lowercase())).collect();
                ui.label(first_upper(&format!("{} triangles: {}", num(o.shows_on.iter().map(|x| x.1).sum()), line.join(", "))));
            }
            ui.add_space(8.0);
            if let Some(saved) = &self.swap_saved {
                ui.label(RichText::new(format!("Saved as {}", saved.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default())).color(theme::GREEN)).on_hover_text(saved.display().to_string());
                if ui.button("Open in Compare").on_hover_text("Original as A, the new file as B").clicked() {
                    let (a, b) = (r.base.path.clone(), saved.clone());
                    self.load(ctx, Slot::A, a);
                    self.load(ctx, Slot::B, b);
                    self.page = Page::Compare;
                    self.selection = Selection::None;
                }
            } else {
                let save = egui::Button::new(RichText::new("Save new file...").color(Color32::from_rgb(0x1a, 0x1a, 0x1a))).fill(theme::GOLD);
                if ui.add(save).on_hover_text("Writes a new .glb. Never overwrites a file, and never writes into the installed game.").clicked() {
                    let stem = |p: &std::path::Path| p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                    let name = format!("{}-{}.glb", stem(&r.base.path), stem(&r.png));
                    let mut dlg = rfd::FileDialog::new().add_filter("Character export (GLB)", &["glb"]).set_title("Save the new file").set_file_name(&name);
                    if let Some(d) = r.base.path.parent() {
                        dlg = dlg.set_directory(d);
                    }
                    if let Some(p) = dlg.save_file() {
                        match crate::swapjob::check_out(&r.base.path, &p).and_then(|full| std::fs::write(&full, &o.bytes).map(|_| full).map_err(|e| e.to_string())) {
                            Ok(full) => self.swap_saved = Some(full),
                            Err(e) => self.error = Some(format!("Not saved: {e}")),
                        }
                    }
                }
            }
        });
    }

    fn screenshot_step(&mut self, ctx: &egui::Context) {
        let Some(out) = self.screenshot.clone() else { return };
        self.frames += 1;
        ctx.request_repaint();
        let busy = !self.loading.is_empty() || self.diff_running || self.swap_running || self.relayout_running || self.relayout_auto.is_some() || (self.page == Page::Compare && self.a.is_some() && self.b.is_some() && self.diff.is_none()) || (self.page == Page::Mark && (self.mark.busy() || self.inspect.is_none()));
        if busy {
            self.settled = 0;
        } else {
            self.settled += 1;
        }
        if self.settled == 12 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        }
        let shot = ctx.input(|i| {
            i.raw.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(img) = shot {
            let rgba: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
            match image::RgbaImage::from_raw(img.size[0] as u32, img.size[1] as u32, rgba).map(|i| i.save(&out)) {
                Some(Ok(())) => println!("screenshot {} ({}x{})", out.display(), img.size[0], img.size[1]),
                _ => eprintln!("screenshot failed"),
            }
            if let Some(v) = &self.diff {
                println!("diff: {} changed pixels; selection {:?}", v.diff.changed_pixels(), self.selection);
            } else {
                println!("selection {:?}", self.selection);
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        } else if self.frames > 6000 && !(self.page == Page::Mark && (self.mark.busy() || (1..60).contains(&self.settled)) && self.frames < 120_000) {
            // Mark & fix may be running the checks (judge takes minutes; they stop themselves), and
            // once they finish the screenshot needs a few frames to arrive.
            eprintln!("no screenshot after {} frames", self.frames);
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.styled {
            theme::install(ctx);
            self.styled = true;
        }
        self.receive(ctx);
        self.shortcuts(ctx);
        self.rail(ctx);
        self.status_bar(ctx);
        match self.page {
            Page::Compare => self.compare_page(ctx),
            Page::Inspect => self.inspect_page(ctx),
            Page::Swap => self.swap_page(ctx),
            Page::Mark => self.mark_page(ctx),
        }
        self.help_window(ctx);
        self.screenshot_step(ctx);
    }
}

// ---------- small helpers ----------

fn row_label(ui: &mut egui::Ui, text: &str) {
    ui.add_sized([78.0, 28.0], egui::Label::new(RichText::new(text.to_uppercase()).size(11.0).color(theme::FAINT).strong()));
}

fn badge(ui: &mut egui::Ui, letter: &str, color: Color32) {
    let (r, _) = ui.allocate_exact_size(Vec2::splat(24.0), Sense::hover());
    ui.painter().rect_filled(r, CornerRadius::same(6), color);
    ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, letter, egui::FontId::new(14.0, FontFamily::Name("strong".into())), Color32::from_rgb(0x14, 0x16, 0x1b));
}

/// An on/off switch.
fn toggle(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
    let size = Vec2::new(40.0, 22.0);
    let (rect, mut resp) = ui.allocate_exact_size(size, Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    let t = ui.ctx().animate_bool_responsive(resp.id, *on);
    let bg = if *on { theme::GOLD.gamma_multiply(0.85) } else { theme::CARD_HI };
    ui.painter().rect(rect, CornerRadius::same(11), bg, Stroke::new(1.0, theme::LINE), egui::StrokeKind::Inside);
    let x = egui::lerp((rect.left() + 11.0)..=(rect.right() - 11.0), t);
    ui.painter().circle_filled(Pos2::new(x, rect.center().y), 8.0, theme::TEXT);
    resp
}

fn title_case(s: &str) -> String {
    let s = s.replace(['-', '_'], " ");
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

/// Friendly texture names: 98_1 is the body sheet, 98_0 the head sheet.
fn texture_title(name: &str) -> String {
    match name {
        "98_1" => "Body texture".into(),
        "98_0" => "Head texture".into(),
        n if n.starts_with("98_0_") => format!("Head: {}", n.trim_start_matches("98_0_").replace('_', " ")),
        n if n.starts_with("98_1_") => format!("Body: {}", n.trim_start_matches("98_1_").replace('_', " ")),
        n => n.to_owned(),
    }
}

fn mesh_title(name: &str) -> String {
    let (base, piece) = name.split_once('#').map(|(b, p)| (b, Some(p))).unwrap_or((name, None));
    let t = match base {
        "mesh0" => "Body",
        "mesh1" => "Head",
        b => b,
    };
    match piece {
        Some(p) => format!("{t} piece {}", p.parse::<usize>().map(|n| n + 1).unwrap_or(0)),
        None => t.to_owned(),
    }
}

fn first_upper(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
}

fn mb(bytes: usize) -> String {
    if bytes >= 1 << 20 {
        format!("{:.1} MB", bytes as f64 / (1u64 << 20) as f64)
    } else {
        format!("{} KB", bytes.div_ceil(1024))
    }
}
