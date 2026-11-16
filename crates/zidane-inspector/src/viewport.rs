//! One 3D view: renders a doc with the CPU renderer, handles turn/zoom/click/hover.
use crate::doc::Doc;
use crate::theme;
use eframe::egui::{self, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use wb_core::render::{self, Camera, Filter};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Size {
    Field,
    Battle,
    Close,
}

pub struct ViewOpts {
    pub size: Size,
    /// Model height on screen in game pixels, at 896 px frame height.
    pub field_px: f32,
    pub battle_px: f32,
    pub filter: Filter,
    /// Show game sizes at 1:1 instead of enlarged.
    pub actual: bool,
}

impl ViewOpts {
    pub fn game_px(&self) -> Option<f32> {
        match self.size {
            Size::Field => Some(self.field_px),
            Size::Battle => Some(self.battle_px),
            Size::Close => None,
        }
    }
}

#[derive(Clone, PartialEq)]
struct Key {
    cam: Camera,
    buf: [usize; 2],
    filter: Filter,
    pose: (Option<usize>, u32),
    tint: u64,
    doc: usize,
}

#[derive(Default)]
pub struct Viewport {
    tex: Option<egui::TextureHandle>,
    pick: Vec<i32>,
    buf: [usize; 2],
    key: Option<Key>,
}

pub struct Shown {
    pub hovered: Option<(usize, usize)>,
    pub clicked: Option<(usize, usize)>,
    /// A click that hit nothing (clears the selection).
    pub clicked_empty: bool,
}

pub struct Draw<'a> {
    pub doc: &'a Doc,
    /// The model both views are framed on (A), so A and B line up pixel for pixel.
    pub frame: &'a wb_core::Model,
    pub pose: &'a [Vec<[f32; 3]>],
    pub pose_key: (Option<usize>, u32),
    pub tint: &'a dyn Fn(usize, usize) -> Option<[u8; 3]>,
    pub tint_key: u64,
    pub caption: String,
    pub sim_click: Option<(f32, f32)>,
}

impl Viewport {
    pub fn show(&mut self, ui: &mut egui::Ui, rect: Rect, id: egui::Id, cam: &mut Camera, opts: &ViewOpts, d: Draw) -> Shown {
        let ppp = ui.ctx().pixels_per_point();
        let resp = ui.interact(rect, id, Sense::click_and_drag());
        let aspect = (rect.width() / rect.height().max(1.0)).max(0.05);

        // Interaction on the shared camera.
        if resp.dragged() {
            let dlt = resp.drag_delta();
            cam.yaw -= dlt.x * 0.01;
            cam.pitch = (cam.pitch + dlt.y * 0.01).clamp(-1.4, 1.4);
        }
        if resp.hovered() && opts.size == Size::Close {
            let s = ui.input(|i| i.smooth_scroll_delta.y);
            if s != 0.0 {
                cam.zoom((1.0 - s * 0.002).clamp(0.5, 2.0));
            }
        }

        // Render buffer: physical pixels for close-up; the game's pixel count for field and battle.
        let (buf, rcam, smooth_display) = match opts.game_px() {
            Some(px) => {
                let bh = (px / 0.8).round().max(16.0) as usize;
                let bw = ((bh as f32) * aspect).round().max(16.0) as usize;
                let mut c = Camera::fit(d.frame, bw as f32 / bh as f32, 0.8, opts.size == Size::Field);
                c.yaw = cam.yaw;
                c.pitch = cam.pitch;
                ([bw, bh], c, false)
            }
            None => {
                let (mut w, mut h) = ((rect.width() * ppp) as usize, (rect.height() * ppp) as usize);
                let cap = 1_600_000f32;
                let scale = ((cap / (w * h).max(1) as f32).sqrt()).min(1.0);
                w = ((w as f32 * scale) as usize).max(32);
                h = ((h as f32 * scale) as usize).max(32);
                ([w, h], *cam, scale < 1.0)
            }
        };
        let key = Key { cam: rcam, buf, filter: opts.filter, pose: d.pose_key, tint: d.tint_key, doc: d.doc as *const Doc as usize };
        if self.key.as_ref() != Some(&key) || self.tex.is_none() {
            let o = render::Options { cam: rcam, w: buf[0], h: buf[1], background: theme::VIEW_BG, filter: opts.filter, positions: Some(d.pose) };
            let f = render::render(&d.doc.model, &d.doc.textures, &o, d.tint);
            let img = egui::ColorImage::from_rgba_unmultiplied(buf, &f.rgba);
            let topt = if smooth_display { egui::TextureOptions::LINEAR } else { egui::TextureOptions::NEAREST };
            match &mut self.tex {
                Some(t) => t.set(img, topt),
                None => self.tex = Some(ui.ctx().load_texture(format!("view-{id:?}"), img, topt)),
            }
            self.pick = f.pick;
            self.buf = buf;
            self.key = Some(key);
        }

        // Where the buffer is drawn.
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, CornerRadius::same(10), Color32::from_rgb(0x12, 0x14, 0x18));
        let draw = if opts.actual && opts.game_px().is_some() {
            Rect::from_center_size(rect.center(), Vec2::new(buf[0] as f32, buf[1] as f32) / ppp)
        } else {
            let s = (rect.width() / buf[0] as f32).min(rect.height() / buf[1] as f32);
            Rect::from_center_size(rect.center(), Vec2::new(buf[0] as f32, buf[1] as f32) * s)
        };
        if let Some(t) = &self.tex {
            painter.image(t.id(), draw, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        }
        painter.rect_stroke(rect, CornerRadius::same(10), Stroke::new(1.0, theme::LINE), egui::StrokeKind::Inside);

        let pick_at = |p: Pos2| -> Option<(usize, usize)> {
            if !draw.contains(p) {
                return None;
            }
            let x = (((p.x - draw.min.x) / draw.width()) * buf[0] as f32) as usize;
            let y = (((p.y - draw.min.y) / draw.height()) * buf[1] as f32) as usize;
            match self.pick.get(y.min(buf[1] - 1) * buf[0] + x.min(buf[0] - 1)).copied().unwrap_or(-1) {
                id if id >= 0 => Some(render::split_id(&d.doc.model, id as usize)),
                _ => None,
            }
        };
        let hovered = resp.hover_pos().and_then(pick_at);
        let mut click_pos = if resp.clicked() { resp.interact_pointer_pos() } else { None };
        if let Some((fx, fy)) = d.sim_click {
            click_pos = Some(Pos2::new(draw.min.x + fx * draw.width(), draw.min.y + fy * draw.height()));
        }
        let clicked = click_pos.and_then(pick_at);

        if !d.caption.is_empty() {
            let g = painter.layout_no_wrap(d.caption.clone(), FontId::proportional(12.5), theme::WEAK);
            let r = Rect::from_min_size(rect.left_bottom() + Vec2::new(10.0, -g.size().y - 16.0), g.size() + Vec2::new(14.0, 8.0));
            painter.rect_filled(r, CornerRadius::same(6), Color32::from_black_alpha(150));
            painter.galley(r.min + Vec2::new(7.0, 4.0), g, theme::WEAK);
        }
        Shown { hovered, clicked, clicked_empty: click_pos.is_some() && clicked.is_none() }
    }
}

/// A name tag that follows the pointer.
pub fn hover_tag(ui: &egui::Ui, text: &str, color: Color32) {
    let Some(p) = ui.ctx().pointer_hover_pos() else { return };
    let painter = ui.ctx().layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("hover-tag")));
    let g = painter.layout_no_wrap(text.to_owned(), FontId::proportional(13.5), theme::TEXT);
    let r = Rect::from_min_size(p + Vec2::new(16.0, 14.0), g.size() + Vec2::new(16.0, 8.0));
    painter.rect_filled(r, CornerRadius::same(6), Color32::from_rgba_unmultiplied(20, 22, 27, 235));
    painter.rect_stroke(r, CornerRadius::same(6), Stroke::new(1.0, color), egui::StrokeKind::Inside);
    painter.galley(r.min + Vec2::new(8.0, 4.0), g, theme::TEXT);
}
