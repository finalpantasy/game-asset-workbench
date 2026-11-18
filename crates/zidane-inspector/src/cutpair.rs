//! Edge-by-edge comparison of a candidate's cut boxes against stock (`gaw gate` rule type
//! `stock-edge`). The gate's defects runs render both models from the same cameras (fitted to the
//! reference), so a box's pixels in the candidate's view are the same place in stock's view. For
//! each box of the rule's kinds, this takes the cut's own pixels (the strongest colour steps in the
//! box, as many as the detector's run length) and measures the Lab step across them the same way
//! the detector does (two-pixel averages either side), in the candidate's picture and in stock's.
//! A box counts as new when stock shows no edge there (fewer than `min_share` of those pixels step
//! by the detector's limit in stock) or when the candidate's step is worse than stock's by more
//! than `margin` Lab. Card-overlap edges and silhouettes that stock shows just as hard do not count.
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use wb_core::defects::lab;

/// The detector's Lab limit for a layer cut (`LAYER_DE` in wb-core defects).
pub const STEP_DE: f32 = 14.0;
/// A pixel whose colour differs from stock's by more than this shows another object there (a
/// weapon in front, a rebuilt card elsewhere), not the same surface repainted. Recolours measured
/// on v3-v7, w0 and w0c move hair pixels by 11-21 Lab (median), at most 45 in a box's top tenth.
pub const OTHER_DE: f32 = 45.0;

pub struct View {
    pub w: usize,
    pub h: usize,
    pub lab: Vec<[f32; 3]>,
    /// Background (the renderer's flat grey) or outside the picture.
    pub bg: Vec<bool>,
}

impl View {
    pub fn from_rgb(w: usize, h: usize, rgb: &[[u8; 3]]) -> View {
        // wb-core's lab() takes 0-255 channels.
        let lab = rgb.iter().map(|p| lab([p[0] as f32, p[1] as f32, p[2] as f32])).collect();
        let bg = rgb.iter().map(|p| p[0] == p[1] && p[1] == p[2] && (p[0] as i32 - 128).abs() <= 2).collect();
        View { w, h, lab, bg }
    }

    pub fn load(p: &Path) -> Result<View, String> {
        let im = image::open(p).map_err(|e| format!("{}: {e}", p.display()))?.to_rgb8();
        let (w, h) = (im.width() as usize, im.height() as usize);
        let rgb: Vec<[u8; 3]> = im.pixels().map(|p| p.0).collect();
        Ok(View::from_rgb(w, h, &rgb))
    }

    /// Lab step across the pixel pair (x, y) -> (x + dx, y + dy): mean of the two pixels on one side
    /// against the two on the other. None when any of the four is background or off the picture.
    pub fn step(&self, x: i64, y: i64, dx: i64, dy: i64) -> Option<f32> {
        let at = |o: i64| -> Option<usize> {
            let (xx, yy) = (x + o * dx, y + o * dy);
            if xx < 0 || yy < 0 || xx as usize >= self.w || yy as usize >= self.h {
                return None;
            }
            let i = yy as usize * self.w + xx as usize;
            if self.bg[i] { None } else { Some(i) }
        };
        let (a0, a1, b0, b1) = (at(-1)?, at(0)?, at(1)?, at(2)?);
        let avg = |p: usize, q: usize| -> [f32; 3] { std::array::from_fn(|k| 0.5 * (self.lab[p][k] + self.lab[q][k])) };
        let (a, b) = (avg(a0, a1), avg(b0, b1));
        Some(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt())
    }
}

#[derive(Debug, Clone)]
pub struct EdgeCmp {
    /// Pixel pairs compared (the cut's own pixels).
    pub n: usize,
    /// Mean step in the candidate and in stock at the same pixel pairs (stock 0 where its pixels are background).
    pub cand: f32,
    pub stock: f32,
    /// Share of those pairs where stock steps by `STEP_DE` or more: stock shows an edge there.
    pub stock_share: f32,
    /// Share of the cut's pairs where stock's picture shows another object (colour over `OTHER_DE`
    /// away); those pairs are left out of the means. Over half: the box cannot be compared.
    pub other_share: f32,
}

/// Compares one box. `run_px` is the detector's run length (how many of the strongest pairs are the cut).
pub fn compare_box(cand: &View, stock: &View, bbox: [i64; 4], run_px: usize) -> Option<EdgeCmp> {
    let mut pairs: Vec<(f32, f32, bool)> = Vec::new();
    for (dx, dy) in [(1i64, 0i64), (0, 1)] {
        for y in bbox[1].max(0)..=bbox[3] {
            for x in bbox[0].max(0)..=bbox[2] {
                let Some(c) = cand.step(x, y, dx, dy) else { continue };
                if c < STEP_DE {
                    continue;
                }
                let i = y as usize * cand.w + x as usize;
                let other = i < stock.lab.len() && !stock.bg[i] && {
                    let (a, b) = (cand.lab[i], stock.lab[i]);
                    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt() > OTHER_DE
                };
                pairs.push((c, stock.step(x, y, dx, dy).unwrap_or(0.0), other));
            }
        }
    }
    if pairs.is_empty() {
        return None;
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    pairs.truncate(run_px.max(1));
    let all = pairs.len();
    let other_share = pairs.iter().filter(|p| p.2).count() as f32 / all as f32;
    let same: Vec<&(f32, f32, bool)> = pairs.iter().filter(|p| !p.2).collect();
    let n = same.len();
    if n == 0 {
        return Some(EdgeCmp { n: 0, cand: 0.0, stock: 0.0, stock_share: 0.0, other_share });
    }
    let cand_m = same.iter().map(|p| p.0).sum::<f32>() / n as f32;
    let stock_m = same.iter().map(|p| p.1).sum::<f32>() / n as f32;
    let share = same.iter().filter(|p| p.1 >= STEP_DE).count() as f32 / n as f32;
    Some(EdgeCmp { n, cand: cand_m, stock: stock_m, stock_share: share, other_share })
}

/// A box is new when stock shows no edge there, or the candidate's step is worse than stock's by
/// more than `margin`. A box stock's picture cannot be compared with (another object over half
/// the cut) counts as new too, so a misaligned view can only fail a candidate, never pass it.
pub fn is_new(c: &EdgeCmp, margin: f32, min_share: f32) -> bool {
    c.other_share > 0.5 || c.stock_share < min_share || c.cand - c.stock > margin
}

fn run_len(d: &Value) -> usize {
    // "hard cut N px long ..." (layer cuts); other kinds fall back to the box's longer side.
    let note = d.get("note").and_then(Value::as_str).unwrap_or("");
    if let Some(rest) = note.strip_prefix("hard cut ") {
        if let Some(n) = rest.split_whitespace().next().and_then(|t| t.parse::<usize>().ok()) {
            return n;
        }
    }
    let b = d.get("screen_box").and_then(Value::as_array);
    let g = |i: usize| b.and_then(|a| a.get(i)).and_then(Value::as_f64).unwrap_or(0.0);
    (g(2) - g(0)).max(g(3) - g(1)).max(1.0) as usize
}

/// Scores every box of `kinds` in the candidate run against the reference run's pictures.
/// Returns (new boxes, all box results as JSON).
pub fn score_run(cand_dir: &Path, ref_dir: &Path, cand_run: &Value, kinds: &[String], margin: f32, min_share: f32) -> Result<(usize, Vec<Value>), String> {
    let mut views: HashMap<(bool, String), View> = HashMap::new();
    fn get(views: &mut HashMap<(bool, String), View>, dir: &Path, stock: bool, v: &str) -> Result<(), String> {
        let k = (stock, v.to_owned());
        if !views.contains_key(&k) {
            views.insert(k, View::load(&dir.join("views").join(format!("{v}.png")))?);
        }
        Ok(())
    }
    let mut new = 0;
    let mut rows = Vec::new();
    for d in cand_run.get("defects").and_then(Value::as_array).into_iter().flatten() {
        let kind = d.get("kind").and_then(Value::as_str).unwrap_or("");
        if !kinds.iter().any(|k| k == kind) {
            continue;
        }
        let id = d.get("id").and_then(Value::as_str).unwrap_or("").to_owned();
        let (Some(v), Some(b)) = (d.get("view").and_then(Value::as_str), d.get("screen_box").and_then(Value::as_array)) else {
            new += 1;
            rows.push(json!({"id": id, "new": true, "why": "no view box"}));
            continue;
        };
        let bb: Vec<i64> = b.iter().filter_map(Value::as_f64).map(|f| f as i64).collect();
        if bb.len() != 4 {
            continue;
        }
        get(&mut views, cand_dir, false, v)?;
        get(&mut views, ref_dir, true, v)?;
        let (c, s) = (&views[&(false, v.to_owned())], &views[&(true, v.to_owned())]);
        let cmp = compare_box(c, s, [bb[0], bb[1], bb[2], bb[3]], run_len(d));
        let (is, row) = match cmp {
            Some(e) => {
                let n = is_new(&e, margin, min_share);
                (n, json!({"id": id, "view": v, "new": n, "pairs": e.n, "candidate_step": e.cand, "stock_step": e.stock, "stock_edge_share": e.stock_share, "other_object_share": e.other_share}))
            }
            None => (false, json!({"id": id, "view": v, "new": false, "why": "no step over the limit in the box"})),
        };
        new += is as usize;
        rows.push(row);
    }
    Ok((new, rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 40x40 picture: hair-coloured left half, a second tone right of x = split, grey above y = 4.
    fn pic(left: [u8; 3], right: [u8; 3], split: usize) -> View {
        let (w, h) = (40, 40);
        let rgb: Vec<[u8; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                if y < 4 { [128, 128, 128] } else if x < split { left } else { right }
            })
            .collect();
        View::from_rgb(w, h, &rgb)
    }

    #[test]
    fn same_edge_as_stock_is_not_new() {
        let stock = pic([200, 170, 90], [120, 90, 40], 20);
        let cand = pic([210, 165, 80], [118, 85, 35], 20);
        let c = compare_box(&cand, &stock, [10, 8, 30, 30], 23).unwrap();
        assert!(c.stock_share > 0.9, "{c:?}");
        assert!(!is_new(&c, 12.0, 0.5), "{c:?}");
    }

    #[test]
    fn edge_stock_does_not_have_is_new() {
        let stock = pic([200, 170, 90], [200, 170, 90], 20);
        let cand = pic([200, 170, 90], [110, 80, 30], 20);
        let c = compare_box(&cand, &stock, [10, 8, 30, 30], 23).unwrap();
        assert!(c.stock_share < 0.1, "{c:?}");
        assert!(is_new(&c, 12.0, 0.5));
    }

    #[test]
    fn much_harder_than_stock_is_new() {
        // Stock steps a little over the limit; the candidate's step is far harder.
        let stock = pic([200, 170, 90], [175, 140, 70], 20);
        let cand = pic([230, 190, 100], [70, 45, 15], 20);
        let c = compare_box(&cand, &stock, [10, 8, 30, 30], 23).unwrap();
        assert!(c.cand - c.stock > 12.0, "{c:?}");
        assert!(is_new(&c, 12.0, 0.5));
    }

    #[test]
    fn another_object_in_stock_cannot_pass() {
        // Stock shows a different object (dark green) over the whole box: not comparable, so new.
        let stock = pic([40, 70, 50], [40, 70, 50], 20);
        let cand = pic([210, 165, 80], [118, 85, 35], 20);
        let c = compare_box(&cand, &stock, [10, 8, 30, 30], 23).unwrap();
        assert!(c.other_share > 0.5, "{c:?}");
        assert!(is_new(&c, 12.0, 0.5));
    }

    #[test]
    fn background_pixels_are_skipped() {
        let stock = pic([200, 170, 90], [120, 90, 40], 20);
        // Box covering only the grey band: no hair pixels, so nothing to compare.
        assert!(compare_box(&stock, &stock, [0, 0, 39, 2], 10).is_none());
    }
}
