//! Golden renders: a fixed, versioned set of camera views of a character, drawn with the CPU
//! renderer, so a new model can be compared with the stock one from the same cameras.
//!
//! The set (version `GOLDEN_VERSION`): whole-body views of idle frame 0 (front, back, both sides,
//! 3/4, top), run at 0.75 s (front, 3/4, side, back) and walk at 0.25 s (front, side), each at game
//! size and at a large size; close-ups of idle frame 0 (face from three angles, hands, waist from
//! three angles). Cameras come from the reference model's bounds and bones and are stored in the
//! manifest, so a candidate is always drawn from the reference's cameras, never its own.
//!
//! Game size follows the in-game field view: the figure is 237 px tall in a 320 px frame, drawn at
//! four times the size and box-filtered down, the way a mip chain reduces it.
use crate::perceptual::Rgb;
use crate::render::{self, Camera, Filter};
use crate::{Model, Texture};
use serde_json::{json, Value};

pub const GOLDEN_VERSION: &str = "goldens-v2";
pub const BACKGROUND: [u8; 3] = [128, 128, 128];
pub const GAME_SIZE: usize = 320;
pub const GAME_SUPERSAMPLE: usize = 4;
pub const LARGE_SIZE: usize = 768;
/// Figure height as a fraction of the frame height at game size (237 / 320).
pub const GAME_FILL: f32 = 237.0 / 320.0;
pub const LARGE_FILL: f32 = 0.82;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Game,
    Large,
    /// A close-up view (drawn once, at the large size).
    Close,
}

impl Kind {
    pub fn tag(self) -> &'static str {
        match self {
            Kind::Game => "game",
            Kind::Large => "large",
            Kind::Close => "close",
        }
    }
}

#[derive(Clone)]
pub struct View {
    pub name: String,
    pub cam: Camera,
    pub clip: Option<String>,
    pub time: f32,
    pub kinds: Vec<Kind>,
}

impl View {
    /// File stem of the picture for a size, such as `run075_front__game`.
    pub fn stem(&self, kind: Kind) -> String {
        format!("{}__{}", self.name, kind.tag())
    }
}

/// One rendered view: the picture and the triangle id under each pixel (-1 for background).
pub struct Shot {
    pub rgb: Rgb,
    pub pick: Vec<i32>,
}

struct Marks {
    height: f32,
    center: [f32; 3],
    head: [f32; 3],
    hand_l: [f32; 3],
    hand_r: [f32; 3],
    waist: [f32; 3],
}

fn marks(model: &Model) -> Marks {
    let (lo, hi) = render::bounds(model);
    let height = hi[1] - lo[1];
    let center: [f32; 3] = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
    // Landmarks are the centers of the vertices a bone owns (most weight), in idle frame 0.
    let pose = model.posed(model.clips.iter().position(|c| c == "idle"), 0.0);
    let centroid = |pred: &dyn Fn(&str) -> bool| -> Option<[f32; 3]> {
        let js: Vec<u16> = model.joint_names.iter().enumerate().filter(|(_, n)| pred(&n.to_ascii_lowercase())).map(|(i, _)| i as u16).collect();
        let (mut sum, mut n) = ([0.0f64; 3], 0usize);
        for (mi, m) in model.meshes.iter().enumerate() {
            for v in 0..m.pos.len() {
                let top = (0..4).max_by(|&a, &b| m.weights[v][a].total_cmp(&m.weights[v][b])).unwrap_or(0);
                if js.contains(&m.joints[v][top]) {
                    for k in 0..3 {
                        sum[k] += pose[mi][v][k] as f64;
                    }
                    n += 1;
                }
            }
        }
        (n > 0).then(|| std::array::from_fn(|k| (sum[k] / n as f64) as f32))
    };
    let head = centroid(&|n| n.ends_with("_head")).unwrap_or([center[0], lo[1] + 0.9 * height, center[2]]);
    let hand = |side: &str| -> [f32; 3] {
        let tag = format!("_{side}_hand");
        centroid(&|n| n.contains(&tag)).unwrap_or([center[0], lo[1] + 0.45 * height, center[2]])
    };
    let waist = centroid(&|n| n.ends_with("_hips") || n.ends_with("_spine")).unwrap_or([center[0], lo[1] + 0.5 * height, center[2]]);
    // The belt sits below the middle of the hips and spine.
    let waist = [waist[0], waist[1] - 0.06 * height, waist[2]];
    Marks { height, center, head, hand_l: hand("l"), hand_r: hand("r"), waist }
}

fn turned(mut cam: Camera, yaw_deg: f32, pitch_deg: f32) -> Camera {
    cam.yaw = std::f32::consts::PI + yaw_deg.to_radians();
    cam.pitch = pitch_deg.to_radians();
    cam
}

/// The standard view set, with cameras fitted to `model` (the reference).
pub fn standard_views(model: &Model) -> Vec<View> {
    let base = Camera::fit(model, 1.0, GAME_FILL, true);
    let mk = marks(model);
    let has = |c: &str| model.clips.iter().any(|x| x == c);
    let both = vec![Kind::Game, Kind::Large];
    let mut v: Vec<View> = Vec::new();
    let mut whole = |name: &str, yaw: f32, pitch: f32, clip: Option<&str>, time: f32| {
        if clip.is_some_and(|c| !has(c)) {
            return;
        }
        v.push(View { name: name.into(), cam: turned(base, yaw, pitch), clip: clip.map(str::to_owned), time, kinds: both.clone() });
    };
    whole("idle0_front", 0.0, 0.0, Some("idle"), 0.0);
    whole("idle0_back", 180.0, 0.0, Some("idle"), 0.0);
    whole("idle0_side90", 90.0, 0.0, Some("idle"), 0.0);
    whole("idle0_side270", 270.0, 0.0, Some("idle"), 0.0);
    whole("idle0_34", 35.0, 10.0, Some("idle"), 0.0);
    whole("idle0_top", 0.0, 70.0, Some("idle"), 0.0);
    whole("run075_front", 0.0, 0.0, Some("run"), 0.75);
    whole("run075_34", 35.0, 10.0, Some("run"), 0.75);
    whole("run075_side90", 90.0, 0.0, Some("run"), 0.75);
    whole("run075_back", 180.0, 0.0, Some("run"), 0.75);
    whole("walk025_front", 0.0, 0.0, Some("walk"), 0.25);
    whole("walk025_side90", 90.0, 0.0, Some("walk"), 0.25);
    let h = mk.height;
    let mut close = |name: &str, at: [f32; 3], half: f32, yaw: f32, pitch: f32| {
        let mut cam = turned(base, yaw, pitch);
        cam.target = at;
        cam.ortho_half = half * h;
        v.push(View { name: name.into(), cam, clip: has("idle").then(|| "idle".to_owned()), time: 0.0, kinds: vec![Kind::Close] });
    };
    close("face_front", mk.head, 0.10, 0.0, 0.0);
    close("face_34", mk.head, 0.10, 35.0, 8.0);
    close("face_side90", mk.head, 0.10, 90.0, 0.0);
    close("hand_l_front", mk.hand_l, 0.07, 0.0, 0.0);
    close("hand_r_front", mk.hand_r, 0.07, 0.0, 0.0);
    close("hand_l_side90", mk.hand_l, 0.07, 90.0, 0.0);
    close("waist_front", mk.waist, 0.14, 0.0, 0.0);
    close("waist_back", mk.waist, 0.14, 180.0, 0.0);
    close("waist_side90", mk.waist, 0.14, 90.0, 0.0);
    v
}

/// Draws one view at one size. Returns None when the model has no clip the view needs.
pub fn render_view(model: &Model, textures: &[Option<Texture>], view: &View, kind: Kind) -> Option<Shot> {
    let clip = match &view.clip {
        Some(name) => Some(model.clips.iter().position(|c| c == name)?),
        None => None,
    };
    let pose = model.posed(clip, view.time);
    let mut cam = view.cam;
    let (px, ss) = match kind {
        Kind::Game => (GAME_SIZE, GAME_SUPERSAMPLE),
        Kind::Large | Kind::Close => (LARGE_SIZE, 1),
    };
    if kind == Kind::Large {
        // Whole-body cameras are fitted for the game fill; zoom so the figure fills the large frame.
        cam.zoom(GAME_FILL / LARGE_FILL);
    }
    let size = px * ss;
    let o = render::Options { cam, w: size, h: size, background: (BACKGROUND, BACKGROUND), filter: Filter::Smooth, positions: Some(&pose) };
    let fr = render::render(model, textures, &o, &|_, _| None);
    let full = Rgb::from_rgba8(size, size, &fr.rgba, BACKGROUND);
    if ss == 1 {
        return Some(Shot { rgb: full, pick: fr.pick });
    }
    let mut pick = vec![-1i32; px * px];
    for y in 0..px {
        for x in 0..px {
            // The first covered sample of the block, scanning from its center outward.
            for (dy, dx) in [(1, 1), (2, 2), (1, 2), (2, 1), (0, 0), (3, 3), (0, 3), (3, 0), (1, 0), (0, 1), (2, 3), (3, 2), (1, 3), (3, 1), (2, 0), (0, 2)] {
                let p = fr.pick[(y * ss + dy) * size + x * ss + dx];
                if p >= 0 {
                    pick[y * px + x] = p;
                    break;
                }
            }
        }
    }
    // Round to 8 bits like the stored PNGs, so a model compared with its own goldens scores exactly 0.
    let small = full.downsample(ss);
    Some(Shot { rgb: Rgb::from_rgba8(px, px, &small.to_rgba8(), BACKGROUND), pick })
}

/// A small part of the model to get its own close-ups: a name and the (mesh, triangle) pairs it covers.
pub struct PartGroup {
    pub name: String,
    pub tris: Vec<(usize, usize)>,
}

/// Two close-ups per part (`part_NAME_a`, `part_NAME_b`), drawn along the part's flattest axis from
/// both sides, in idle frame 0, so a small part such as a dagger or a buckle is never reduced to a
/// few pixels of a whole-body view. The frame fits the part's own size.
pub fn part_views(model: &Model, groups: &[PartGroup]) -> Vec<View> {
    let idle = model.clips.iter().position(|c| c == "idle");
    let pose = model.posed(idle, 0.0);
    let (lo, hi) = render::bounds(model);
    let height = hi[1] - lo[1];
    let base = Camera::fit(model, 1.0, GAME_FILL, true);
    let mut out = Vec::new();
    for g in groups {
        let mut pts: Vec<[f64; 3]> = Vec::new();
        for &(mi, ti) in &g.tris {
            let Some(t) = model.meshes.get(mi).and_then(|m| m.tris.get(ti)) else { continue };
            for &v in t {
                let p = pose[mi][v as usize];
                pts.push([p[0] as f64, p[1] as f64, p[2] as f64]);
            }
        }
        if pts.len() < 3 {
            continue;
        }
        let (mut plo, mut phi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for p in &pts {
            for k in 0..3 {
                plo[k] = plo[k].min(p[k]);
                phi[k] = phi[k].max(p[k]);
            }
        }
        let center: [f64; 3] = std::array::from_fn(|k| 0.5 * (plo[k] + phi[k]));
        let mut cov = [[0.0f64; 3]; 3];
        for p in &pts {
            for i in 0..3 {
                for j in 0..3 {
                    cov[i][j] += (p[i] - center[i]) * (p[j] - center[j]);
                }
            }
        }
        let (val, vec) = crate::strands::eigen_sym3(cov);
        let flat = (0..3).min_by(|&a, &b| val[a].total_cmp(&val[b])).unwrap_or(2);
        let n = [vec[0][flat], vec[1][flat], vec[2][flat]];
        let extent = (0..3).map(|k| phi[k] - plo[k]).fold(0.0f64, f64::max);
        let half = ((0.62 * extent) as f32).max(0.035 * height);
        for (tag, s) in [("a", 1.0f64), ("b", -1.0)] {
            let d = [s * n[0], s * n[1], s * n[2]];
            let yaw = (-d[0]).atan2(-d[2]).to_degrees() as f32;
            let pitch = (d[1].clamp(-1.0, 1.0).asin().to_degrees() as f32).clamp(-75.0, 75.0);
            let mut cam = turned(base, yaw, pitch);
            cam.target = [center[0] as f32, center[1] as f32, center[2] as f32];
            cam.ortho_half = half;
            out.push(View { name: format!("part_{}_{tag}", g.name), cam, clip: idle.map(|_| "idle".to_owned()), time: 0.0, kinds: vec![Kind::Close] });
        }
    }
    out
}

fn cam_json(c: &Camera) -> Value {
    json!({"yaw": c.yaw, "pitch": c.pitch, "dist": c.dist, "target": c.target, "fov_y": c.fov_y, "ortho": c.ortho, "ortho_half": c.ortho_half})
}

fn cam_from(v: &Value) -> Option<Camera> {
    let f = |k: &str| v.get(k)?.as_f64().map(|x| x as f32);
    let t = v.get("target")?.as_array()?;
    Some(Camera {
        yaw: f("yaw")?,
        pitch: f("pitch")?,
        dist: f("dist")?,
        target: [t.first()?.as_f64()? as f32, t.get(1)?.as_f64()? as f32, t.get(2)?.as_f64()? as f32],
        fov_y: f("fov_y")?,
        ortho: v.get("ortho")?.as_bool()?,
        ortho_half: f("ortho_half")?,
    })
}

pub fn manifest(character: &str, source: &str, views: &[View]) -> Value {
    json!({
        "version": GOLDEN_VERSION,
        "character": character,
        "source": source,
        "background": BACKGROUND,
        "game": {"size": GAME_SIZE, "supersample": GAME_SUPERSAMPLE, "figure_px": (GAME_FILL * GAME_SIZE as f32).round()},
        "large": {"size": LARGE_SIZE, "fill": LARGE_FILL},
        "views": views.iter().map(|v| json!({
            "name": v.name,
            "clip": v.clip,
            "time": v.time,
            "sizes": v.kinds.iter().map(|k| k.tag()).collect::<Vec<_>>(),
            "camera": cam_json(&v.cam),
        })).collect::<Vec<_>>(),
    })
}

/// Reads the views back from a manifest.
pub fn views_from_manifest(m: &Value) -> Result<Vec<View>, String> {
    if m.get("version").and_then(Value::as_str) != Some(GOLDEN_VERSION) {
        return Err(format!("manifest version is not {GOLDEN_VERSION}"));
    }
    let list = m.get("views").and_then(Value::as_array).ok_or("manifest has no views")?;
    list.iter()
        .map(|v| {
            let kinds = v
                .get("sizes")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|s| match s.as_str() {
                            Some("game") => Some(Kind::Game),
                            Some("large") => Some(Kind::Large),
                            Some("close") => Some(Kind::Close),
                            _ => None,
                        })
                        .collect()
                })
                .unwrap_or_default();
            Ok(View {
                name: v.get("name").and_then(Value::as_str).ok_or("view without a name")?.to_owned(),
                cam: v.get("camera").and_then(cam_from).ok_or("view without a camera")?,
                clip: v.get("clip").and_then(Value::as_str).map(str::to_owned),
                time: v.get("time").and_then(Value::as_f64).unwrap_or(0.0) as f32,
                kinds,
            })
        })
        .collect()
}
