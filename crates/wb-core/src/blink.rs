//! Field Zidane's eye blink: the three boxes on sheet `98_0` that the game's texture animation reads
//! and writes (`Models/GeoTexAnim/GEO_MAIN_F0_ZDN.tab`, read by Memoria's `GEOTEXHEADER`).
//!
//! The head has three texture animations on the same target (the eye area):
//! - anim 0 copies the closed-eye box onto the target,
//! - anim 1 copies the open-eye box onto the target,
//! - anim 2 (random timing) alternates the two, which is the blink.
//!
//! So a layout must keep all three boxes where they are, and the open-eye box must still hold open-eye
//! art: whatever is painted there shows over the eyes whenever they are open.
//!
//! gaw cannot read the `.tab` (it lives in the game's archives). The boxes below are the game's own
//! values: the harness `texanim` command prints each animation's `rectuvs` and `target`, and
//! `boxes_from_rects` turns those into the boxes here (a unit test does that with the recorded data).

/// The sheet the head texture animation works on.
pub const SHEET: &str = "98_0";
/// Pixel size the boxes are measured at.
pub const SIZE: f64 = 512.0;

/// One box: what it is for and `[x0, y0, x1, y1]` in pixels at `SIZE`, y down.
pub struct BlinkBox {
    pub role: &'static str,
    pub px: [f64; 4],
}

/// Closed-eye source, open-eye source, eye target.
pub const ZIDANE_BOXES: [BlinkBox; 3] = [
    BlinkBox { role: "closed-eye source", px: [224.0, 216.0, 272.0, 248.0] },
    BlinkBox { role: "open-eye source", px: [200.0, 184.0, 248.0, 216.0] },
    BlinkBox { role: "eye target", px: [32.0, 116.0, 80.0, 148.0] },
];

/// Boxes as UV fractions `[u0, v0, u1, v1]` (v down, like the image), for `pattern::Keep`.
pub fn zidane_uv_boxes() -> Vec<[f64; 4]> {
    ZIDANE_BOXES.iter().map(|b| b.px.map(|v| v / SIZE)).collect()
}

/// Turns the game's texture-animation data into pixel boxes: Unity `Rect`s `[x, y, w, h]` in UV units with
/// y up (the `rectuvs` of an animation) into `[x0, y0, x1, y1]` pixels at `size`, y down.
pub fn boxes_from_rects(rectuvs: &[[f64; 4]], size: f64) -> Vec<[f64; 4]> {
    rectuvs.iter().map(|r| [r[0] * size, (1.0 - r[1] - r[3]) * size, (r[0] + r[2]) * size, (1.0 - r[1]) * size].map(|v| (v * 8.0).round() / 8.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_rects_give_the_three_boxes() {
        // Recorded from the harness `texanim` on a field Zidane: anim 0 closed, anim 1 open, anim 2 both.
        let closed = [0.4376221, 0.5157471, 0.09350586, 0.06225586];
        let open = [0.3907471, 0.5782471, 0.09350586, 0.06225586];
        let b = boxes_from_rects(&[closed, open], SIZE);
        let near = |a: [f64; 4], c: [f64; 4]| a.iter().zip(c).all(|(x, y)| (x - y).abs() < 0.5);
        assert!(near(b[0], ZIDANE_BOXES[0].px), "closed {:?}", b[0]);
        assert!(near(b[1], ZIDANE_BOXES[1].px), "open {:?}", b[1]);
        // The target is the game's integer box on the 128-px head sheet: 8,29 size 12x8 at 4x.
        let t = [8.0 * 4.0, 29.0 * 4.0, 20.0 * 4.0, 37.0 * 4.0];
        assert!(near(t, ZIDANE_BOXES[2].px));
    }

    #[test]
    fn three_boxes_do_not_overlap() {
        for i in 0..3 {
            for j in i + 1..3 {
                let (a, b) = (ZIDANE_BOXES[i].px, ZIDANE_BOXES[j].px);
                let ox = a[2].min(b[2]) - a[0].max(b[0]);
                let oy = a[3].min(b[3]) - a[1].max(b[1]);
                assert!(ox <= 0.0 || oy <= 0.0, "{} overlaps {}", ZIDANE_BOXES[i].role, ZIDANE_BOXES[j].role);
            }
        }
    }
}
