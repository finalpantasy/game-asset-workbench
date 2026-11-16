//! Look and feel: palette, fonts, spacing, and small widgets (cards, pills, segmented buttons).
use eframe::egui::{self, Color32, CornerRadius, FontFamily, FontId, Margin, RichText, Stroke, TextStyle};

pub const BG: Color32 = Color32::from_rgb(0x14, 0x16, 0x1b);
pub const PANEL: Color32 = Color32::from_rgb(0x1a, 0x1d, 0x23);
pub const CARD: Color32 = Color32::from_rgb(0x22, 0x26, 0x2e);
pub const CARD_HI: Color32 = Color32::from_rgb(0x2a, 0x2f, 0x39);
pub const LINE: Color32 = Color32::from_rgb(0x30, 0x35, 0x40);
pub const TEXT: Color32 = Color32::from_rgb(0xe6, 0xe8, 0xec);
pub const WEAK: Color32 = Color32::from_rgb(0x9a, 0xa1, 0xad);
pub const FAINT: Color32 = Color32::from_rgb(0x6b, 0x72, 0x7e);
pub const GOLD: Color32 = Color32::from_rgb(0xe9, 0xb9, 0x49);
pub const TEAL: Color32 = Color32::from_rgb(0x4f, 0xb3, 0xbf);
pub const MAGENTA: Color32 = Color32::from_rgb(0xff, 0x4f, 0xd8);
pub const RED: Color32 = Color32::from_rgb(0xe5, 0x53, 0x4b);
pub const GREEN: Color32 = Color32::from_rgb(0x5f, 0xc2, 0x7e);

pub const A_COLOR: Color32 = TEAL;
pub const B_COLOR: Color32 = GOLD;

/// Viewport backdrop gradient (top, bottom).
pub const VIEW_BG: ([u8; 3], [u8; 3]) = ([0x2a, 0x2f, 0x3a], [0x12, 0x14, 0x18]);
pub const HL_SELECT: [u8; 3] = [0xe9, 0xb9, 0x49];
pub const HL_CHANGED: [u8; 3] = [0xff, 0x4f, 0xd8];
pub const HL_DEAD: [u8; 3] = [0xff, 0x30, 0x30];
/// A surface outside the selected part that still shows some of its pixels.
pub const HL_TOUCH: [u8; 3] = [0x3f, 0xd0, 0xe0];
/// Everything outside the selected part is dimmed toward this, so the part stands out whatever its color.
pub const HL_DIM: [u8; 3] = wb_core::render::DIM;

pub fn install(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let windir = std::env::var("WINDIR").unwrap_or_else(|_| r"C:\Windows".into());
    let load = |f: &str| std::fs::read(std::path::Path::new(&windir).join("Fonts").join(f)).ok();
    if let Some(b) = load("segoeui.ttf") {
        fonts.font_data.insert("segoe".into(), std::sync::Arc::new(egui::FontData::from_owned(b)));
        fonts.families.get_mut(&FontFamily::Proportional).unwrap().insert(0, "segoe".into());
    }
    let mut strong = fonts.families.get(&FontFamily::Proportional).cloned().unwrap_or_default();
    if let Some(b) = load("seguisb.ttf") {
        fonts.font_data.insert("segoe-sb".into(), std::sync::Arc::new(egui::FontData::from_owned(b)));
        strong.insert(0, "segoe-sb".into());
    }
    fonts.families.insert(FontFamily::Name("strong".into()), strong);
    if let Some(b) = load("consola.ttf") {
        fonts.font_data.insert("consolas".into(), std::sync::Arc::new(egui::FontData::from_owned(b)));
        fonts.families.get_mut(&FontFamily::Monospace).unwrap().insert(0, "consolas".into());
    }
    ctx.set_fonts(fonts);

    ctx.style_mut(|s| {
        s.text_styles = [
            (TextStyle::Heading, FontId::new(21.0, FontFamily::Name("strong".into()))),
            (TextStyle::Body, FontId::new(14.5, FontFamily::Proportional)),
            (TextStyle::Button, FontId::new(14.5, FontFamily::Proportional)),
            (TextStyle::Small, FontId::new(12.0, FontFamily::Proportional)),
            (TextStyle::Monospace, FontId::new(13.5, FontFamily::Monospace)),
        ]
        .into();
        s.spacing.item_spacing = egui::vec2(8.0, 7.0);
        s.spacing.button_padding = egui::vec2(11.0, 5.0);
        s.spacing.interact_size.y = 28.0;
        s.spacing.slider_width = 160.0;
        let v = &mut s.visuals;
        *v = egui::Visuals::dark();
        v.panel_fill = PANEL;
        v.window_fill = CARD;
        v.extreme_bg_color = BG;
        v.faint_bg_color = CARD;
        v.window_stroke = Stroke::new(1.0, LINE);
        v.window_corner_radius = CornerRadius::same(10);
        v.menu_corner_radius = CornerRadius::same(8);
        v.selection.bg_fill = Color32::from_rgb(0x6b, 0x55, 0x22);
        v.selection.stroke = Stroke::new(1.0, GOLD);
        v.hyperlink_color = TEAL;
        v.override_text_color = None;
        let r = CornerRadius::same(7);
        for (w, fill, stroke, fg) in [
            (&mut v.widgets.noninteractive, PANEL, LINE, WEAK),
            (&mut v.widgets.inactive, CARD_HI, LINE, TEXT),
            (&mut v.widgets.hovered, Color32::from_rgb(0x34, 0x3a, 0x46), Color32::from_rgb(0x48, 0x50, 0x5e), TEXT),
            (&mut v.widgets.active, Color32::from_rgb(0x3d, 0x44, 0x52), GOLD, TEXT),
            (&mut v.widgets.open, CARD_HI, GOLD, TEXT),
        ] {
            w.bg_fill = fill;
            w.weak_bg_fill = fill;
            w.bg_stroke = Stroke::new(1.0, stroke);
            w.fg_stroke = Stroke::new(1.2, fg);
            w.corner_radius = r;
            w.expansion = 0.0;
        }
    });
}

/// A rounded panel with an optional title.
pub fn card<R>(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::Frame::new()
        .fill(CARD)
        .stroke(Stroke::new(1.0, LINE))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::same(12))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            if !title.is_empty() {
                ui.label(RichText::new(title.to_uppercase()).size(11.5).color(FAINT).strong());
                ui.add_space(2.0);
            }
            add(ui)
        })
        .inner
}

/// A small colored tag.
pub fn pill(ui: &mut egui::Ui, text: &str, color: Color32) -> egui::Response {
    egui::Frame::new()
        .fill(color.gamma_multiply(0.18))
        .stroke(Stroke::new(1.0, color.gamma_multiply(0.6)))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::symmetric(8, 1))
        .show(ui, |ui| ui.label(RichText::new(text).size(12.0).color(color)))
        .response
}

/// Buttons in a row where one is selected. Returns true when the value changed.
pub fn segmented<T: PartialEq + Copy>(ui: &mut egui::Ui, value: &mut T, options: &[(T, &str, &str)]) -> bool {
    let mut changed = false;
    egui::Frame::new().fill(BG).corner_radius(CornerRadius::same(8)).inner_margin(Margin::same(3)).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for (v, label, tip) in options {
                let sel = *value == *v;
                let text = RichText::new(*label).color(if sel { Color32::from_rgb(0x1a, 0x1a, 0x1a) } else { WEAK });
                let mut b = egui::Button::new(text).corner_radius(CornerRadius::same(6)).stroke(Stroke::NONE);
                b = b.fill(if sel { GOLD } else { Color32::TRANSPARENT });
                let r = ui.add(b);
                let r = if tip.is_empty() { r } else { r.on_hover_text(*tip) };
                if r.clicked() && !sel {
                    *value = *v;
                    changed = true;
                }
            }
        });
    });
    changed
}

/// A two-column row: weak label, value.
pub fn kv(ui: &mut egui::Ui, k: &str, v: impl Into<RichText>) {
    ui.label(RichText::new(k).color(WEAK));
    ui.label(v.into());
    ui.end_row();
}

/// Thousands separators: 5218477 -> "5,218,477".
pub fn num(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub fn pct(part: usize, total: usize) -> String {
    if total == 0 {
        return "0%".into();
    }
    let p = 100.0 * part as f64 / total as f64;
    if p > 0.0 && p < 0.1 {
        "<0.1%".into()
    } else {
        format!("{p:.1}%")
    }
}
