//! Sewing-pattern UV layout, all in Rust (no Blender, no paid packers).
//!
//! 1. Cut: every triangle gets a body part from the bone that carries it (head, torso, arm, hand,
//!    leg, foot, tail; left/right from the bone label). Pieces are the stock UV islands split where
//!    the part changes; slivers fold back into a neighbor.
//! 2. Flatten: each piece is re-flattened from the 3D surface (LSCM, `flatten.rs`), which also
//!    undoes stock folds and overlaps.
//! 3. Straighten: each piece turns so the body's up points up the page; arms and hands put the
//!    shoulder end up.
//! 4. Scale: one texel density for the whole model, times a factor for the face and hands.
//! 5. Pair: left/right mirror pieces sit side by side (character's right on the left, as seen).
//! 6. Pack: pieces go in part blocks, read top to bottom (head, torso, arms, hands, legs, feet).
//! 7. Paint: the old texture is copied onto the new layout through each triangle, then bled
//!    outward so mipmaps do not pull in the background.
//! 8. Guides: a pattern sheet (outlines, part names, numbered seam pairs, paint-boundary lines)
//!    plus part, position and normal maps for painting tools and AI.
use crate::flatten::{self, area, area3, cross, dot, norm, scale, sub};
use crate::rebuild::NewMesh;
use crate::{font, uv, Model, Texture};
use serde_json::{json, Value};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Part {
    Head,
    Torso,
    Tail,
    Arm,
    Hand,
    Leg,
    Foot,
    Other,
}

impl Part {
    pub fn name(self) -> &'static str {
        match self {
            Part::Head => "head",
            Part::Torso => "torso",
            Part::Tail => "tail",
            Part::Arm => "arm",
            Part::Hand => "hand",
            Part::Leg => "leg",
            Part::Foot => "foot",
            Part::Other => "other",
        }
    }
    pub fn color(self) -> [u8; 3] {
        match self {
            Part::Head => [242, 170, 120],
            Part::Torso => [120, 170, 235],
            Part::Tail => [200, 160, 220],
            Part::Arm => [130, 210, 150],
            Part::Hand => [245, 215, 100],
            Part::Leg => [110, 200, 210],
            Part::Foot => [215, 140, 170],
            Part::Other => [180, 180, 180],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Side {
    C,
    L,
    R,
}

impl Side {
    pub fn name(self) -> &'static str {
        match self {
            Side::C => "",
            Side::L => "L",
            Side::R => "R",
        }
    }
}

/// Body part and side from a bone label such as `bone016_R_hand` (labels from `model-gltf`).
pub fn classify(joint: &str) -> (Part, Side) {
    let s = joint.to_ascii_lowercase();
    let side = if s.contains("_l_") || s.ends_with("_l") || s.contains("left") {
        Side::L
    } else if s.contains("_r_") || s.ends_with("_r") || s.contains("right") {
        Side::R
    } else {
        Side::C
    };
    let has = |k: &[&str]| k.iter().any(|w| s.contains(w));
    let part = if has(&["hand", "finger", "thumb", "palm"]) {
        Part::Hand
    } else if has(&["arm", "shoulder", "elbow", "clav"]) {
        Part::Arm
    } else if has(&["foot", "toe", "ankle"]) {
        Part::Foot
    } else if has(&["thigh", "shin", "knee", "leg", "calf"]) || (has(&["hip"]) && side != Side::C) {
        Part::Leg
    } else if has(&["head", "neck", "crest", "hair", "hat", "ear", "jaw", "eye", "face", "mouth"]) {
        Part::Head
    } else if has(&["tail"]) {
        Part::Tail
    } else if has(&["spine", "chest", "hips", "pelvis", "root", "waist", "torso", "body"]) {
        Part::Torso
    } else {
        Part::Other
    };
    (part, side)
}

pub struct Options {
    /// Texture size of the output atlases.
    pub size: usize,
    /// Gap between pieces, in texels at `size`.
    pub pad_px: f64,
    /// Density factors for the face/head and the hands (1 = same as the body).
    pub head_density: f64,
    pub hand_density: f64,
    /// Pieces with fewer triangles fold into a neighbor of the same stock island.
    pub min_piece_tris: usize,
    /// Size of the pattern sheet (labels get room at this size).
    pub guide: usize,
    /// One texel density across all sheets (a sewing pattern at one scale).
    pub same_density: bool,
    /// Boxes that stay where they are on the stock sheet (for example the boxes the game's eye blink
    /// copies between). Triangles that sample them keep their stock UVs, the paint inside is copied
    /// as is, and no other piece is packed over them.
    pub keep: Vec<Keep>,
    /// Turn each piece so its dominant bone (bind pose) runs along the page's vertical axis,
    /// instead of the body-height rule. Off by default.
    pub orient_bones: bool,
    /// What to do with left/right pairs: find them only, or also lay the left piece out as the exact
    /// reflection of the right one (same scale, same corner positions). Off by default.
    pub mirror: MirrorMode,
    /// How close (share of the body height) a corner must lie to the mirror image of its partner.
    pub mirror_tol: f64,
    /// Target for a piece's anisotropy p95. A flat piece above it is relaxed (symmetric Dirichlet,
    /// no flips) and, when that is not enough, cut where the cut helps most. None: off (the default).
    pub low_stretch: Option<f64>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MirrorMode {
    /// Pairs get no corner match; they only sit side by side (the default layout).
    Off,
    /// Corner-match the pairs and report them, but leave the layout as it is.
    Report,
    /// Match the pairs and lay exact matches out as reflections.
    Lock,
}

#[derive(Clone, Debug)]
pub struct Keep {
    /// Sheet (image) name, for example `98_0`.
    pub sheet: String,
    /// u0, v0, u1, v1 in 0..1, v down (PNG pixel / size).
    pub uv: [f64; 4],
}

impl Default for Options {
    fn default() -> Self {
        Options { size: 4096, pad_px: 24.0, head_density: 1.5, hand_density: 1.75, min_piece_tris: 6, guide: 2048, same_density: true, keep: Vec::new(), orient_bones: false, mirror: MirrorMode::Off, mirror_tol: 0.004, low_stretch: None }
    }
}

/// Font scale for names and seam numbers on a pattern sheet of `gs` px.
pub fn label_scale(gs: usize) -> usize {
    (gs / 1024).max(1)
}

pub struct Piece {
    pub mesh: usize,
    pub part: Part,
    pub side: Side,
    /// "FRONT" or "BACK" when a ring was cut into panels, else empty.
    pub facet: &'static str,
    pub label: String,
    /// Front triangles (back faces follow their front).
    pub tris: Vec<u32>,
    /// Welded corner ids (position + stock UV) of the piece's vertices.
    pub corners: Vec<u32>,
    /// Final UV per corner (0..1, v down).
    pub uv: Vec<[f64; 2]>,
    pub area3d: f64,
    /// Triangles left folded by the flattening.
    pub flips: usize,
    /// True when flattening failed and the stock UV shape was kept.
    pub kept_stock: bool,
    /// True for triangles held at their stock UV position by `Options::keep`.
    pub pinned: bool,
    pub twin: Option<usize>,
    pub density_factor: f64,
    /// Dominant bone (joint index) by triangle area, and its bind-pose direction signed to point up
    /// the page (None for pinned pieces and models without a rig).
    pub bone: Option<usize>,
    pub bone_axis: [f64; 3],
    /// Set when this piece is the exact reflection of that piece (`MirrorMode::Lock`).
    pub mirror_of: Option<usize>,
}

pub struct Seam {
    pub id: usize,
    pub a: usize,
    pub b: usize,
    /// Per 3D edge: its UV endpoints in piece a and in piece b.
    pub edges: Vec<([[f64; 2]; 2], [[f64; 2]; 2])>,
}

pub struct Atlas {
    pub image: usize,
    pub name: String,
    pub meshes: Vec<usize>,
    pub coverage: f64,
}

pub struct Layout {
    pub pieces: Vec<Piece>,
    pub seams: Vec<Seam>,
    pub meshes: Vec<NewMesh>,
    pub atlases: Vec<Atlas>,
    /// Per mesh, per triangle: its piece (back faces share their front's piece).
    pub piece_of: Vec<Vec<usize>>,
    /// Texels per model unit at `size` for density factor 1.
    pub texels_per_unit: f64,
    pub size: usize,
    /// Left/right pairs as matched corner by corner (empty unless `Options::mirror` is on).
    pub mirrors: Vec<MirrorPair>,
}

/// One left/right pair of pieces, matched corner by corner across the body's mirror plane.
#[derive(Clone, Debug)]
pub struct MirrorPair {
    /// The piece that is laid out first (the character's right) and the one that mirrors it.
    pub a: usize,
    pub b: usize,
    pub corners: [usize; 2],
    /// Corners of a (and of b) that have a partner within the tolerance.
    pub matched: [usize; 2],
    /// Every corner matched, one to one.
    pub exact: bool,
    /// b was laid out as the reflection of a.
    pub locked: bool,
    /// Largest distance (model units) between a corner's mirror image and its partner.
    pub max_dist: f64,
    /// Triangles of b that fold under the reflection.
    pub flips: usize,
    pub note: &'static str,
    /// For each corner of b, the corner of a it mirrors (empty unless exact).
    pub map: Vec<u32>,
}

pub(crate) fn qk(x: f32, q: f64) -> i64 {
    (x as f64 / q).round_ties_even() as i64
}

/// Welded position ids and welded corner ids (position + UV), quantized like `uv::analyze`.
fn weld(m: &crate::model::Mesh) -> (Vec<u32>, Vec<u32>) {
    let mut gmap: HashMap<[i64; 3], u32> = HashMap::new();
    let mut umap: HashMap<[i64; 5], u32> = HashMap::new();
    let (mut gid, mut uid) = (Vec::new(), Vec::new());
    for (p, u) in m.pos.iter().zip(&m.uv) {
        let g = [qk(p[0], 1e-4), qk(p[1], 1e-4), qk(p[2], 1e-4)];
        let n = gmap.len() as u32;
        gid.push(*gmap.entry(g).or_insert(n));
        let n = umap.len() as u32;
        uid.push(*umap.entry([g[0], g[1], g[2], qk(u[0], 1e-5), qk(u[1], 1e-5)]).or_insert(n));
    }
    (gid, uid)
}

fn find(par: &mut [usize], mut x: usize) -> usize {
    while par[x] != x {
        par[x] = par[par[x]];
        x = par[x];
    }
    x
}

pub(crate) fn p64(p: [f32; 3]) -> [f64; 3] {
    p.map(|x| x as f64)
}

/// Body axes from the parts: lateral (character's left to right), up, and the center.
fn axes(model: &Model, labels: &[Vec<(Part, Side)>]) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let mut acc: HashMap<&str, ([f64; 3], f64)> = HashMap::new();
    let mut all = ([0.0; 3], 0.0);
    for (mi, m) in model.meshes.iter().enumerate() {
        for (t, tri) in m.tris.iter().enumerate() {
            let c = scale(tri.iter().fold([0.0; 3], |a, &v| [a[0] + m.pos[v as usize][0] as f64, a[1] + m.pos[v as usize][1] as f64, a[2] + m.pos[v as usize][2] as f64]), 1.0 / 3.0);
            let (part, side) = labels[mi][t];
            let mut add = |k: &'static str| {
                let e = acc.entry(k).or_insert(([0.0; 3], 0.0));
                e.0 = [e.0[0] + c[0], e.0[1] + c[1], e.0[2] + c[2]];
                e.1 += 1.0;
            };
            match side {
                Side::L => add("L"),
                Side::R => add("R"),
                Side::C => {}
            }
            match part {
                Part::Head => add("top"),
                Part::Foot | Part::Leg => add("bottom"),
                _ => {}
            }
            all = ([all.0[0] + c[0], all.0[1] + c[1], all.0[2] + c[2]], all.1 + 1.0);
        }
    }
    let mean = |k: &str| acc.get(k).filter(|e| e.1 > 0.0).map(|e| scale(e.0, 1.0 / e.1));
    let center = if all.1 > 0.0 { scale(all.0, 1.0 / all.1) } else { [0.0; 3] };
    let mut lat = match (mean("L"), mean("R")) {
        (Some(l), Some(r)) if norm(sub(r, l)) > 1e-9 => scale(sub(r, l), 1.0 / norm(sub(r, l))),
        _ => [1.0, 0.0, 0.0],
    };
    let mut up = match (mean("top"), mean("bottom")) {
        (Some(t), Some(b)) if norm(sub(t, b)) > 1e-9 => sub(t, b),
        _ => [0.0, 1.0, 0.0],
    };
    up = sub(up, scale(lat, dot(up, lat)));
    if norm(up) < 1e-9 {
        up = [0.0, 1.0, 0.0];
        lat = [1.0, 0.0, 0.0];
    }
    (lat, scale(up, 1.0 / norm(up)), center)
}

/// Lays out every mesh of `model`. Meshes that share an image share one atlas.
pub fn layout(model: &Model, o: &Options) -> Result<Layout, String> {
    let nm = model.meshes.len();
    // Per triangle: part label from its main bone.
    let mut labels: Vec<Vec<(Part, Side)>> = Vec::with_capacity(nm);
    for mi in 0..nm {
        labels.push((0..model.meshes[mi].tris.len()).map(|t| classify(model.joint_names.get(model.triangle_joint(mi, t)).map(String::as_str).unwrap_or(""))).collect());
    }
    let (lat, up, center) = axes(model, &labels);
    let mut pieces: Vec<Piece> = Vec::new();
    let mut piece_of: Vec<Vec<usize>> = Vec::with_capacity(nm);
    let mut welds = Vec::with_capacity(nm);
    let mut topos = Vec::with_capacity(nm);
    for mi in 0..nm {
        let m = &model.meshes[mi];
        let mut topo = uv::analyze(m);
        let (gid, uid) = weld(m);
        // A back face follows its front only when it reads the same texels. One with its own paint
        // (the dark inside of a hair card) is a surface of its own and gets its own place.
        for t in 0..m.tris.len() {
            if let Some(f) = topo.dup_of[t] {
                let key = |x: usize| {
                    let mut k = m.tris[x].map(|v| uid[v as usize]);
                    k.sort_unstable();
                    k
                };
                if key(t) != key(f as usize) {
                    topo.dup_of[t] = None;
                }
            }
        }
        let nt = m.tris.len();
        let front = |t: usize| topo.dup_of[t].is_none();
        // 3D edges and which front triangles use them; an edge is "joined" when both sides share UVs.
        let mut edges: HashMap<(u32, u32), Vec<(usize, (u32, u32))>> = HashMap::new();
        for (t, tri) in m.tris.iter().enumerate() {
            if !front(t) {
                continue;
            }
            for c in 0..3 {
                let (a, b) = (tri[c] as usize, tri[(c + 1) % 3] as usize);
                let key = (gid[a].min(gid[b]), gid[a].max(gid[b]));
                let su = (uid[a].min(uid[b]), uid[a].max(uid[b]));
                edges.entry(key).or_default().push((t, su));
            }
        }
        let mut joined: Vec<Vec<usize>> = vec![Vec::new(); nt];
        let mut touch: Vec<Vec<usize>> = vec![Vec::new(); nt];
        for list in edges.values() {
            for i in 0..list.len() {
                for k in i + 1..list.len() {
                    let ((t1, s1), (t2, s2)) = (list[i], list[k]);
                    if t1 == t2 {
                        continue;
                    }
                    touch[t1].push(t2);
                    touch[t2].push(t1);
                    if s1 == s2 {
                        joined[t1].push(t2);
                        joined[t2].push(t1);
                    }
                }
            }
        }
        // Smooth jagged bone-weight borders: a triangle takes the label most of its joined
        // neighbors carry, when at least two agree and it is a minority among them.
        let lab = &mut labels[mi];
        for _ in 0..3 {
            let prev = lab.clone();
            for t in 0..nt {
                if !front(t) || joined[t].len() < 2 {
                    continue;
                }
                let mut count: HashMap<(Part, Side), usize> = HashMap::new();
                for &n in &joined[t] {
                    *count.entry(prev[n]).or_insert(0) += 1;
                }
                let own = count.get(&prev[t]).copied().unwrap_or(0);
                if let Some((&best, &n)) = count.iter().max_by_key(|(k, n)| (**n, std::cmp::Reverse(**k))) {
                    if best != prev[t] && n >= 2 && n > own + 1 {
                        lab[t] = best;
                    }
                }
            }
        }
        // Pieces: joined neighbors with the same label.
        let mut par: Vec<usize> = (0..nt).collect();
        for t in 0..nt {
            for &n in &joined[t] {
                if lab[t] == lab[n] {
                    let (a, b) = (find(&mut par, t), find(&mut par, n));
                    par[a] = b;
                }
            }
        }
        // Slivers fold into the joined neighbor piece they share the most edges with.
        loop {
            let mut size: HashMap<usize, usize> = HashMap::new();
            for t in 0..nt {
                if front(t) {
                    *size.entry(find(&mut par, t)).or_insert(0) += 1;
                }
            }
            let mut merged = false;
            let small: Vec<usize> = size.iter().filter(|(_, &n)| n < o.min_piece_tris).map(|(&r, _)| r).collect();
            for r in small {
                if find(&mut par, r) != r {
                    continue;
                }
                let mut votes: HashMap<usize, usize> = HashMap::new();
                for t in 0..nt {
                    if front(t) && find(&mut par, t) == r {
                        for &n in &joined[t] {
                            let rn = find(&mut par, n);
                            if rn != r {
                                *votes.entry(rn).or_insert(0) += 1;
                            }
                        }
                    }
                }
                if let Some((&to, _)) = votes.iter().max_by_key(|(k, n)| (**n, std::cmp::Reverse(**k))) {
                    let new_label = (0..nt).find(|&t| front(t) && find(&mut par, t) == to).map(|t| lab[t]).unwrap();
                    for t in 0..nt {
                        if front(t) && find(&mut par, t) == r {
                            lab[t] = new_label;
                        }
                    }
                    par[r] = to;
                    merged = true;
                }
            }
            if !merged {
                break;
            }
        }
        let mut root_piece: HashMap<usize, usize> = HashMap::new();
        let mut po = vec![usize::MAX; nt];
        for t in 0..nt {
            if !front(t) {
                continue;
            }
            let r = find(&mut par, t);
            let p = *root_piece.entry(r).or_insert_with(|| {
                pieces.push(Piece {
                    mesh: mi,
                    part: lab[t].0,
                    side: lab[t].1,
                    facet: "",
                    label: String::new(),
                    tris: Vec::new(),
                    corners: Vec::new(),
                    uv: Vec::new(),
                    area3d: 0.0,
                    flips: 0,
                    kept_stock: false,
                    pinned: false,
                    twin: None,
                    density_factor: 1.0,
                    bone: None,
                    bone_axis: [0.0; 3],
                    mirror_of: None,
                });
                pieces.len() - 1
            });
            pieces[p].tris.push(t as u32);
            po[t] = p;
        }
        for t in 0..nt {
            if let Some(f) = topo.dup_of[t] {
                po[t] = po[f as usize];
            }
        }
        piece_of.push(po);
        welds.push((gid, uid));
        topos.push(topo);
    }
    // Kept boxes: triangles with a stock UV corner inside one leave their piece and form one pinned
    // piece per mesh that stays at its stock position.
    let keep_of = |mi: usize| -> Vec<[f64; 4]> {
        if model.meshes[mi].image.is_none() {
            return Vec::new();
        }
        let name = model.atlas_name(mi);
        o.keep.iter().filter(|k| k.sheet == name).map(|k| k.uv).collect()
    };
    for mi in 0..nm {
        let boxes = keep_of(mi);
        if boxes.is_empty() {
            continue;
        }
        let m = &model.meshes[mi];
        let inside = |t: u32| {
            m.tris[t as usize].iter().any(|&v| {
                let u = m.uv[v as usize];
                let (x, y) = (u[0] as f64, u[1] as f64);
                boxes.iter().any(|b| x > b[0] && x < b[2] && y > b[1] && y < b[3])
            })
        };
        let mut taken: Vec<u32> = Vec::new();
        for p in pieces.iter_mut().filter(|p| p.mesh == mi) {
            p.tris.retain(|&t| if inside(t) { taken.push(t); false } else { true });
        }
        if taken.is_empty() {
            continue;
        }
        let mut votes: HashMap<Part, usize> = HashMap::new();
        for &t in &taken {
            *votes.entry(labels[mi][t as usize].0).or_insert(0) += 1;
        }
        let part = votes.into_iter().max_by_key(|(k, n)| (*n, std::cmp::Reverse(*k))).map(|(k, _)| k).unwrap_or(Part::Head);
        pieces.push(Piece { mesh: mi, part, side: Side::C, facet: "STOCK", label: String::new(), tris: taken, corners: Vec::new(), uv: Vec::new(), area3d: 0.0, flips: 0, kept_stock: false, pinned: true, twin: None, density_factor: 1.0, bone: None, bone_axis: [0.0; 3], mirror_of: None });
        // Pieces that lost triangles may now be in several parts: split them into connected parts.
        let ids: Vec<usize> = (0..pieces.len()).filter(|&i| pieces[i].mesh == mi && !pieces[i].pinned && !pieces[i].tris.is_empty()).collect();
        for i in ids {
            let parts = split(&pieces[i], model, &welds[mi].1, [0.0; 3]);
            for (k, (tris, _)) in parts.into_iter().enumerate() {
                if k == 0 {
                    pieces[i].tris = tris;
                } else {
                    let p = &pieces[i];
                    let np = Piece { mesh: mi, part: p.part, side: p.side, facet: "", label: String::new(), tris, corners: Vec::new(), uv: Vec::new(), area3d: 0.0, flips: 0, kept_stock: false, pinned: false, twin: None, density_factor: 1.0, bone: None, bone_axis: [0.0; 3], mirror_of: None };
                    pieces.push(np);
                }
            }
        }
    }
    pieces.retain(|p| !p.tris.is_empty());
    // Flatten, straighten and scale each piece (model units x density factor).
    let body_h = {
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for m in &model.meshes {
            for p in &m.pos {
                let h = dot(sub(p64(*p), center), up);
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
        (hi - lo).max(1e-9)
    };
    // The character faces -z (model-gltf convention); FRONT panels are on that side.
    let fwd = {
        let f = cross(up, lat);
        if f[2] <= 0.0 { f } else { scale(f, -1.0) }
    };
    // Pieces that are closed rings (sleeves, hat, coat tail) fold flat onto themselves. Such a piece
    // is cut into FRONT and BACK panels (then left/right, then top/bottom) and flattened again.
    let bones: Vec<([f64; 3], [f64; 3])> = model.rig.as_ref().map(|r| r.bind_bones()).unwrap_or_default();
    let mut queue: Vec<(usize, u8)> = (0..pieces.len()).rev().map(|i| (i, 0)).collect();
    while let Some((pi, depth)) = queue.pop() {
        let pm = pieces[pi].mesh;
        let (pos, ltris) = flatten_piece(&mut pieces[pi], model, &welds[pm].1);
        if pieces[pi].pinned {
            // Stock UVs, unchanged: the game reads these texels at fixed places.
            let (m, uid) = (&model.meshes[pm], &welds[pm].1);
            let mut at: HashMap<u32, [f64; 2]> = HashMap::new();
            for &t in &pieces[pi].tris {
                for &v in &m.tris[t as usize] {
                    at.insert(uid[v as usize], [m.uv[v as usize][0] as f64, m.uv[v as usize][1] as f64]);
                }
            }
            let p = &mut pieces[pi];
            p.uv = p.corners.iter().map(|c| at[c]).collect();
            p.flips = 0;
            p.kept_stock = true;
            continue;
        }
        // Cut when the flat piece lies on itself: a ring that folded, or even one flipped triangle.
        // Plane cuts first (front/back, left/right, top/bottom, twice over); when those run out or
        // change nothing, a cut by facing parts layers that lie on each other.
        if depth < 9 && pieces[pi].tris.len() >= 2 && (pieces[pi].flips > 0 || overlap(&pieces[pi].uv, &ltris) > 0.0002) {
            let uid = &welds[pm].1;
            let mut halves = if depth < 6 && pieces[pi].tris.len() >= 4 { split(&pieces[pi], model, uid, [fwd, lat, up][depth as usize % 3]) } else { Vec::new() };
            let by_plane = halves.len() > 1;
            if !by_plane {
                halves = split_facing(&pieces[pi], model, uid);
            }
            if halves.len() > 1 {
                pieces[pi].tris.clear();
                for (k, (tris, positive)) in halves.into_iter().enumerate() {
                    let facet = if depth == 0 && by_plane { if positive { "FRONT" } else { "BACK" } } else { pieces[pi].facet };
                    let idx = if k == 0 {
                        pi
                    } else {
                        let p = &pieces[pi];
                        let np = Piece { mesh: p.mesh, part: p.part, side: p.side, facet, label: String::new(), tris: Vec::new(), corners: Vec::new(), uv: Vec::new(), area3d: 0.0, flips: 0, kept_stock: false, pinned: false, twin: None, density_factor: 1.0, bone: None, bone_axis: [0.0; 3], mirror_of: None };
                        pieces.push(np);
                        pieces.len() - 1
                    };
                    pieces[idx].tris = tris;
                    pieces[idx].facet = facet;
                    queue.push((idx, depth + 1));
                }
                continue;
            }
        }
        // The cuts ran out and the flat piece still lies on itself (small folded cards in hair and
        // ears): use the piece's stock shape instead when that one is flat and has no folds.
        if !pieces[pi].kept_stock && (pieces[pi].flips > 0 || overlap(&pieces[pi].uv, &ltris) > 0.0002) {
            let (m, uid) = (&model.meshes[pm], &welds[pm].1);
            let mut at: HashMap<u32, [f64; 2]> = HashMap::new();
            for &t in &pieces[pi].tris {
                for &v in &m.tris[t as usize] {
                    at.insert(uid[v as usize], [m.uv[v as usize][0] as f64, m.uv[v as usize][1] as f64]);
                }
            }
            let mut s: Vec<[f64; 2]> = pieces[pi].corners.iter().map(|c| at[c]).collect();
            if ltris.iter().map(|t| area(t.map(|v| s[v as usize]))).sum::<f64>() < 0.0 {
                s.iter_mut().for_each(|x| x[0] = -x[0]);
            }
            let folds = ltris.iter().filter(|t| area(t.map(|v| s[v as usize])) < 0.0).count();
            if folds == 0 && overlap(&s, &ltris) <= 0.0002 {
                let p = &mut pieces[pi];
                p.uv = s;
                p.flips = 0;
                p.kept_stock = true;
            }
        }
        // Low-stretch pass: a flat piece whose anisotropy p95 is above the target is relaxed; when that
        // is not enough, it is cut by the plane that helps most and the halves are checked again.
        if let Some(target) = o.low_stretch {
            if pieces[pi].flips == 0 && !pieces[pi].tris.is_empty() {
                let mut q = std::mem::take(&mut pieces[pi].uv);
                let stock = pieces[pi].kept_stock;
                if stock {
                    // The stock shape is in texture units; bring it to the size of the 3D piece first.
                    let auv: f64 = ltris.iter().map(|t| area(t.map(|v| q[v as usize])).abs()).sum::<f64>().max(1e-18);
                    let k = (pieces[pi].area3d / auv).sqrt();
                    q.iter_mut().for_each(|x| *x = [x[0] * k, x[1] * k]);
                }
                let before = stretch_p95(&pos, &ltris, &q);
                let p95 = relax_to_target(&pos, &ltris, &mut q, target);
                if stock && p95 < before {
                    // Relaxed, so no longer the stock shape.
                    pieces[pi].kept_stock = false;
                }
                pieces[pi].uv = q;
                if p95 > target && depth < 9 && pieces[pi].tris.len() >= 8 {
                    let uid = &welds[pm].1;
                    let mut best: Option<(f64, Vec<(Vec<u32>, bool)>)> = None;
                    for axis in [fwd, lat, up] {
                        let halves = split(&pieces[pi], model, uid, axis);
                        if halves.len() < 2 {
                            continue;
                        }
                        let worst = halves.iter().map(|(t, _)| piece_p95(model, uid, &pieces[pi], t, target)).fold(0.0, f64::max);
                        if best.as_ref().map_or(true, |b| worst < b.0) {
                            best = Some((worst, halves));
                        }
                    }
                    if let Some((worst, halves)) = best {
                        if worst < p95 - 0.05 {
                            pieces[pi].tris.clear();
                            for (k, (tris, _)) in halves.into_iter().enumerate() {
                                let idx = if k == 0 {
                                    pi
                                } else {
                                    let p = &pieces[pi];
                                    let np = Piece { mesh: p.mesh, part: p.part, side: p.side, facet: p.facet, label: String::new(), tris: Vec::new(), corners: Vec::new(), uv: Vec::new(), area3d: 0.0, flips: 0, kept_stock: false, pinned: false, twin: None, density_factor: 1.0, bone: None, bone_axis: [0.0; 3], mirror_of: None };
                                    pieces.push(np);
                                    pieces.len() - 1
                                };
                                pieces[idx].tris = tris;
                                queue.push((idx, depth + 1));
                            }
                            continue;
                        }
                    }
                }
            }
        }
        let (bone, baxis) = dominant_bone(model, &pieces[pi], &bones, center, up);
        let p = &mut pieces[pi];
        p.bone = bone;
        p.bone_axis = baxis;
        let mut q = std::mem::take(&mut p.uv);
        // Straighten: turn the gradient of "height" on the piece to point up the page (-v).
        let h = |x: [f64; 3]| match p.part {
            Part::Arm | Part::Hand => -dot(sub(x, center), lat).abs(),
            _ => dot(sub(x, center), up),
        };
        let hv: Vec<f64> = pos.iter().map(|&x| h(x)).collect();
        let mut g = mean_gradient(&q, &ltris, &hv).0;
        let auv: f64 = ltris.iter().map(|t| area(t.map(|v| q[v as usize])).abs()).sum::<f64>().max(1e-18);
        if o.orient_bones && bone.is_some() {
            // The bone's own direction, when it lies along the surface (a bone that points out of the
            // piece has no direction on it; that piece keeps the body-height rule).
            let hb: Vec<f64> = pos.iter().map(|&x| dot(x, baxis)).collect();
            let (gb, sum) = mean_gradient(&q, &ltris, &hb);
            let k0 = (p.area3d / auv).sqrt();
            let strength = (gb[0] * gb[0] + gb[1] * gb[1]).sqrt() / sum.max(1e-18) / k0.max(1e-18);
            if strength >= 0.15 {
                g = gb;
            }
        }
        let phi = if g[0].abs() + g[1].abs() > 1e-18 { -std::f64::consts::FRAC_PI_2 - g[1].atan2(g[0]) } else { 0.0 };
        let (s, c) = phi.sin_cos();
        p.density_factor = match p.part {
            Part::Head => o.head_density,
            Part::Hand => o.hand_density,
            _ => 1.0,
        };
        let k = (p.area3d / auv).sqrt() * p.density_factor;
        for x in q.iter_mut() {
            *x = [(x[0] * c - x[1] * s) * k, (x[0] * s + x[1] * c) * k];
        }
        p.uv = q;
    }
    // Triangles moved between pieces when rings were cut: rebuild the per-triangle piece index.
    for (mi, po) in piece_of.iter_mut().enumerate() {
        for (pi, p) in pieces.iter().enumerate() {
            if p.mesh == mi {
                for &t in &p.tris {
                    po[t as usize] = pi;
                }
            }
        }
        for t in 0..po.len() {
            if let Some(f) = topos[mi].dup_of[t] {
                po[t] = po[f as usize];
            }
        }
    }
    // Mirror pairs: an L piece and the R piece of the same part whose mirrored center is closest.
    let centroid = |p: &Piece| {
        let m = &model.meshes[p.mesh];
        let mut c = [0.0; 3];
        for &t in &p.tris {
            for &v in &m.tris[t as usize] {
                let x = p64(m.pos[v as usize]);
                c = [c[0] + x[0], c[1] + x[1], c[2] + x[2]];
            }
        }
        scale(c, 1.0 / (3.0 * p.tris.len() as f64))
    };
    let cents: Vec<[f64; 3]> = pieces.iter().map(centroid).collect();
    let mut cands = Vec::new();
    for a in 0..pieces.len() {
        for b in 0..pieces.len() {
            let (pa, pb) = (&pieces[a], &pieces[b]);
            if pa.pinned || pb.pinned || pa.side != Side::L || pb.side != Side::R || pa.part != pb.part || pa.mesh != pb.mesh || pa.facet != pb.facet {
                continue;
            }
            let (na, nb) = (pa.tris.len() as f64, pb.tris.len() as f64);
            if (na - nb).abs() > 0.25 * na.max(nb) {
                continue;
            }
            let x = sub(cents[a], center);
            let mirrored = sub(cents[a], scale(lat, 2.0 * dot(x, lat)));
            let cost = norm(sub(mirrored, cents[b]));
            if cost < 0.15 * body_h {
                cands.push((cost, a, b));
            }
        }
    }
    cands.sort_by(|x, y| x.0.total_cmp(&y.0));
    for (_, a, b) in cands {
        if pieces[a].twin.is_none() && pieces[b].twin.is_none() {
            pieces[a].twin = Some(b);
            pieces[b].twin = Some(a);
        }
    }
    let mirrors = if o.mirror == MirrorMode::Off { Vec::new() } else { mirror_pairs(model, &mut pieces, &welds, lat, body_h, o.mirror_tol, o.mirror == MirrorMode::Lock) };
    // Labels: "ARM R", numbered when a part has several pieces.
    let mut count: HashMap<(usize, Part, Side, &'static str), usize> = HashMap::new();
    for p in &pieces {
        *count.entry((p.mesh, p.part, p.side, p.facet)).or_insert(0) += 1;
    }
    let mut seen: HashMap<(usize, Part, Side, &'static str), usize> = HashMap::new();
    for p in pieces.iter_mut() {
        let k = (p.mesh, p.part, p.side, p.facet);
        let n = seen.entry(k).or_insert(0);
        *n += 1;
        let base = [p.part.name().to_ascii_uppercase().as_str(), p.facet, p.side.name()].iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join(" ");
        p.label = if count[&k] > 1 { format!("{base} {}", n) } else { base };
    }
    // Atlases: meshes grouped by image.
    let mut atlases: Vec<Atlas> = Vec::new();
    for mi in 0..nm {
        let img = model.meshes[mi].image.unwrap_or(usize::MAX);
        match atlases.iter_mut().find(|a| a.image == img) {
            Some(a) => a.meshes.push(mi),
            None => atlases.push(Atlas { image: img, name: if img == usize::MAX { format!("mesh{mi}") } else { model.atlas_name(mi) }, meshes: vec![mi], coverage: 0.0 }),
        }
    }
    // Room above each piece for its name on the pattern sheet (drawn at `o.guide` px).
    let ls = label_scale(o.guide);
    let lw: Vec<f64> = pieces.iter().map(|p| (font::width(&p.label, ls) + 2 * ls) as f64 / o.guide as f64).collect();
    let lh = (9 * ls + 2) as f64 / o.guide as f64;
    let pad = o.pad_px / o.size as f64;
    let ids_of = |atlas: &Atlas, pieces: &[Piece]| -> Vec<usize> { (0..pieces.len()).filter(|&i| atlas.meshes.contains(&pieces[i].mesh) && !pieces[i].pinned).collect() };
    // Areas the packer must leave free: kept boxes and pinned pieces (with their name strip).
    let reserved_of = |atlas: &Atlas, pieces: &[Piece]| -> Vec<[f64; 4]> {
        let mut r: Vec<[f64; 4]> = Vec::new();
        for &mi in &atlas.meshes {
            for b in keep_of(mi) {
                r.push([b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad]);
            }
        }
        for p in pieces.iter().filter(|p| p.pinned && atlas.meshes.contains(&p.mesh)) {
            let b = bbox(&p.uv);
            r.push([b[0] - pad, b[1] - pad - lh, b[2] + pad, b[3] + pad]);
        }
        r
    };
    // One texel density for every sheet: the largest scale at which every sheet fits.
    let mut fit = Vec::with_capacity(atlases.len());
    for atlas in &atlases {
        let ids = ids_of(atlas, &pieces);
        let res = reserved_of(atlas, &pieces);
        fit.push(pack(&mut pieces, &ids, pad, &lw, lh, &res, None)?);
    }
    let common = fit.iter().cloned().fold(f64::MAX, f64::min);
    let mut tpu = f64::MAX;
    for (ai, atlas) in atlases.iter().enumerate() {
        let ids = ids_of(atlas, &pieces);
        let res = reserved_of(atlas, &pieces);
        let s = if o.same_density { common } else { fit[ai] };
        pack(&mut pieces, &ids, pad, &lw, lh, &res, Some(s))?;
        tpu = tpu.min(s * o.size as f64);
    }
    // New vertices: one per (source vertex, piece, welded corner).
    let mut meshes = Vec::with_capacity(nm);
    for mi in 0..nm {
        let m = &model.meshes[mi];
        let (gid, uid) = &welds[mi];
        let topo = &topos[mi];
        let po = &piece_of[mi];
        let mut lookup: Vec<HashMap<u32, usize>> = Vec::new();
        let index_of = |p: usize, pieces: &[Piece], lookup: &mut Vec<HashMap<u32, usize>>| -> usize {
            while lookup.len() <= p {
                lookup.push(HashMap::new());
            }
            if lookup[p].is_empty() {
                for (i, &c) in pieces[p].corners.iter().enumerate() {
                    lookup[p].insert(c, i);
                }
            }
            p
        };
        let mut key: HashMap<(u32, usize, u32), u32> = HashMap::new();
        let mut nmesh = NewMesh { src: Vec::new(), uv: Vec::new(), tris: Vec::with_capacity(m.tris.len()) };
        for (t, tri) in m.tris.iter().enumerate() {
            let p = po[t];
            index_of(p, &pieces, &mut lookup);
            let front_tri = topo.dup_of[t].map(|f| m.tris[f as usize]);
            let mut out = [0u32; 3];
            for c in 0..3 {
                let v = tri[c];
                let corner = match front_tri {
                    None => uid[v as usize],
                    Some(ft) => ft.iter().find(|&&w| gid[w as usize] == gid[v as usize]).map(|&w| uid[w as usize]).unwrap_or(uid[ft[c] as usize]),
                };
                let li = *lookup[p].get(&corner).ok_or_else(|| format!("mesh {mi} triangle {t}: corner not in its piece"))?;
                let q = pieces[p].uv[li];
                out[c] = *key.entry((v, p, corner)).or_insert_with(|| {
                    nmesh.src.push(v);
                    nmesh.uv.push([q[0] as f32, q[1] as f32]);
                    (nmesh.src.len() - 1) as u32
                });
            }
            nmesh.tris.push(out);
        }
        meshes.push(nmesh);
    }
    // Seams: 3D edges whose two sides landed in different pieces.
    let mut seam_map: HashMap<(usize, usize), usize> = HashMap::new();
    let mut seams: Vec<Seam> = Vec::new();
    for mi in 0..nm {
        let m = &model.meshes[mi];
        let (gid, _) = &welds[mi];
        let topo = &topos[mi];
        let nmesh = &meshes[mi];
        let mut by_edge: HashMap<(u32, u32), Vec<(usize, [[f64; 2]; 2])>> = HashMap::new();
        for (t, tri) in m.tris.iter().enumerate() {
            if topo.dup_of[t].is_some() {
                continue;
            }
            for c in 0..3 {
                let (a, b) = (c, (c + 1) % 3);
                let (ga, gb) = (gid[tri[a] as usize], gid[tri[b] as usize]);
                let q = |k: usize| {
                    let u = nmesh.uv[nmesh.tris[t][k] as usize];
                    [u[0] as f64, u[1] as f64]
                };
                let ends = if ga < gb { [q(a), q(b)] } else { [q(b), q(a)] };
                by_edge.entry((ga.min(gb), ga.max(gb))).or_default().push((piece_of[mi][t], ends));
            }
        }
        let mut keys: Vec<&(u32, u32)> = by_edge.keys().collect();
        keys.sort();
        for k in keys {
            let list = &by_edge[k];
            if list.len() != 2 || list[0].0 == list[1].0 {
                continue;
            }
            let (x, y) = if list[0].0 < list[1].0 { (list[0], list[1]) } else { (list[1], list[0]) };
            let si = *seam_map.entry((x.0, y.0)).or_insert_with(|| {
                seams.push(Seam { id: 0, a: x.0, b: y.0, edges: Vec::new() });
                seams.len() - 1
            });
            seams[si].edges.push((x.1, y.1));
        }
    }
    // Number seams in reading order of their first piece.
    let order: Vec<usize> = {
        let mut o: Vec<usize> = (0..seams.len()).collect();
        let key = |s: &Seam| {
            let p = &pieces[s.a];
            let c = p.uv.iter().fold([f64::MAX, f64::MAX], |a, q| [a[0].min(q[0]), a[1].min(q[1])]);
            (p.mesh, (c[1] * 1000.0) as i64, (c[0] * 1000.0) as i64, s.b)
        };
        o.sort_by_key(|&i| key(&seams[i]));
        o
    };
    for (n, &i) in order.iter().enumerate() {
        seams[i].id = n + 1;
    }
    seams.sort_by_key(|s| s.id);
    Ok(Layout { pieces, seams, meshes, atlases, piece_of, texels_per_unit: tpu, size: o.size, mirrors })
}

/// Area-weighted 95th percentile of the anisotropy of a flat piece (the number `patternx` reports).
fn stretch_p95(pos: &[[f64; 3]], ltris: &[[u32; 3]], q: &[[f64; 2]]) -> f64 {
    let v: Vec<(f64, f64)> = ltris
        .iter()
        .filter_map(|t| {
            let p = t.map(|v| pos[v as usize]);
            let w = area3(p);
            (w > 1e-14).then(|| (crate::patternx::anisotropy(p, t.map(|v| q[v as usize])), w))
        })
        .collect();
    crate::patternx::wpercentile(v, 0.95)
}

/// Relaxes a flat piece (symmetric Dirichlet, no flips) when its anisotropy p95 is above `target`
/// and keeps the result when it is better. Returns the p95 of the piece as it stands.
fn relax_to_target(pos: &[[f64; 3]], ltris: &[[u32; 3]], q: &mut Vec<[f64; 2]>, target: f64) -> f64 {
    let before = stretch_p95(pos, ltris, q);
    if before <= target {
        return before;
    }
    let mut r = q.clone();
    flatten::relax_sd(pos, ltris, &mut r, 800);
    let after = stretch_p95(pos, ltris, &r);
    if after < before && overlap(&r, ltris) <= 0.0002 {
        *q = r;
        after
    } else {
        before
    }
}

/// The anisotropy p95 a group of triangles would have as a piece of its own, after the low-stretch
/// pass (a huge number when it would fold).
fn piece_p95(model: &Model, uid: &[u32], base: &Piece, tris: &[u32], target: f64) -> f64 {
    let mut t = Piece { mesh: base.mesh, part: base.part, side: base.side, facet: base.facet, label: String::new(), tris: tris.to_vec(), corners: Vec::new(), uv: Vec::new(), area3d: 0.0, flips: 0, kept_stock: false, pinned: false, twin: None, density_factor: 1.0, bone: None, bone_axis: [0.0; 3], mirror_of: None };
    let (pos, lt) = flatten_piece(&mut t, model, uid);
    if t.kept_stock || t.flips > 0 || overlap(&t.uv, &lt) > 0.0002 {
        return f64::MAX;
    }
    let mut q = std::mem::take(&mut t.uv);
    relax_to_target(&pos, &lt, &mut q, target)
}

/// The area-weighted gradient of the per-corner function `hv` over a flat piece, and the sum of the
/// weights (twice the flat area).
fn mean_gradient(q: &[[f64; 2]], ltris: &[[u32; 3]], hv: &[f64]) -> ([f64; 2], f64) {
    let mut g = [0.0, 0.0];
    let mut sum = 0.0;
    for t in ltris {
        let [a, b, c] = t.map(|v| v as usize);
        let (e1, e2) = ([q[b][0] - q[a][0], q[b][1] - q[a][1]], [q[c][0] - q[a][0], q[c][1] - q[a][1]]);
        let det = e1[0] * e2[1] - e1[1] * e2[0];
        if det.abs() < 1e-18 {
            continue;
        }
        let (d1, d2) = (hv[b] - hv[a], hv[c] - hv[a]);
        // Solve [e1; e2] g = [d1; d2]; weight by area (det/2), so sum det * g.
        let gx = (d1 * e2[1] - d2 * e1[1]) / det;
        let gy = (e1[0] * d2 - e2[0] * d1) / det;
        g = [g[0] + gx * det.abs(), g[1] + gy * det.abs()];
        sum += det.abs();
    }
    (g, sum)
}

/// The joint that carries most of a piece's surface (by triangle area), and its bind-pose bone
/// direction signed to point up the page: along the body's up when the bone is mostly vertical, else
/// toward the body's center (shoulder end up).
fn dominant_bone(model: &Model, p: &Piece, bones: &[([f64; 3], [f64; 3])], center: [f64; 3], up: [f64; 3]) -> (Option<usize>, [f64; 3]) {
    if bones.is_empty() || p.tris.is_empty() || p.pinned {
        return (None, [0.0; 3]);
    }
    let m = &model.meshes[p.mesh];
    let mut w: HashMap<usize, f64> = HashMap::new();
    for &t in &p.tris {
        let a = area3(m.tris[t as usize].map(|v| p64(m.pos[v as usize])));
        *w.entry(model.triangle_joint(p.mesh, t as usize)).or_insert(0.0) += a;
    }
    let mut best: Option<(usize, f64)> = None;
    for (&j, &a) in &w {
        if j < bones.len() && best.map_or(true, |(bj, ba)| a > ba || (a == ba && j < bj)) {
            best = Some((j, a));
        }
    }
    let Some((j, _)) = best else { return (None, [0.0; 3]) };
    let (head, ax) = bones[j];
    let s = if dot(ax, up).abs() > 0.35 {
        dot(ax, up).signum()
    } else if dot(sub(center, head), ax).abs() > 1e-9 {
        dot(sub(center, head), ax).signum()
    } else {
        1.0
    };
    (Some(j), scale(ax, s))
}

fn local_tris(model: &Model, p: &Piece, uid: &[u32]) -> Vec<[u32; 3]> {
    let at: HashMap<u32, u32> = p.corners.iter().enumerate().map(|(i, &c)| (c, i as u32)).collect();
    let m = &model.meshes[p.mesh];
    p.tris.iter().map(|&t| m.tris[t as usize].map(|v| at[&uid[v as usize]])).collect()
}

fn corner_positions(model: &Model, p: &Piece, uid: &[u32]) -> Vec<[f64; 3]> {
    let m = &model.meshes[p.mesh];
    let mut at: HashMap<u32, [f64; 3]> = HashMap::new();
    for &t in &p.tris {
        for &v in &m.tris[t as usize] {
            at.entry(uid[v as usize]).or_insert_with(|| p64(m.pos[v as usize]));
        }
    }
    p.corners.iter().map(|c| at[c]).collect()
}

/// Matches every twin pair corner by corner across the body's mirror plane. With `lock`, a pair whose
/// corners all match one to one lays piece b out as the exact reflection of piece a (the character's
/// right), at the same scale; a reflection that would fold triangles is left alone.
fn mirror_pairs(model: &Model, pieces: &mut [Piece], welds: &[(Vec<u32>, Vec<u32>)], lat: [f64; 3], body_h: f64, tol_frac: f64, lock: bool) -> Vec<MirrorPair> {
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for a in 0..pieces.len() {
        if let Some(b) = pieces[a].twin {
            if a < b {
                // The character's right goes first (it sits on the left of the sheet).
                pairs.push(if pieces[a].side == Side::R { (a, b) } else { (b, a) });
            }
        }
    }
    let pos: Vec<Vec<[f64; 3]>> = pairs.iter().flat_map(|&(a, b)| [a, b]).map(|i| corner_positions(model, &pieces[i], &welds[pieces[i].mesh].1)).collect();
    let cen = |v: &Vec<[f64; 3]>| scale(v.iter().fold([0.0; 3], |a, x| [a[0] + x[0], a[1] + x[1], a[2] + x[2]]), 1.0 / v.len().max(1) as f64);
    // The mirror plane: normal from the pairs' centers (left to right), offset from their midpoints.
    let mut n = [0.0; 3];
    for (k, &(a, _)) in pairs.iter().enumerate() {
        // pairs are (right, left); the normal points from the character's left to its right.
        let d = sub(cen(&pos[2 * k]), cen(&pos[2 * k + 1]));
        let l = norm(d);
        if l > 1e-9 {
            let w = pieces[a].tris.len() as f64 / l;
            n = [n[0] + d[0] * w, n[1] + d[1] * w, n[2] + d[2] * w];
        }
    }
    n = if norm(n) > 1e-9 && dot(n, lat) > 0.0 { scale(n, 1.0 / norm(n)) } else { lat };
    let mut offs: Vec<f64> = (0..pairs.len()).map(|k| 0.5 * (dot(cen(&pos[2 * k]), n) + dot(cen(&pos[2 * k + 1]), n))).collect();
    offs.sort_by(|a, b| a.total_cmp(b));
    let d0 = offs.get(offs.len() / 2).copied().unwrap_or(0.0);
    let reflect = |x: [f64; 3]| {
        let s = 2.0 * (dot(x, n) - d0);
        sub(x, scale(n, s))
    };
    let tol = tol_frac * body_h;
    let mut out = Vec::new();
    for (k, &(a, b)) in pairs.iter().enumerate() {
        let (pa, pb) = (&pos[2 * k], &pos[2 * k + 1]);
        let ma: Vec<[f64; 3]> = pa.iter().map(|&x| reflect(x)).collect();
        let nearest = |from: &[[f64; 3]], to: &[[f64; 3]]| -> Vec<(usize, f64)> {
            from.iter()
                .map(|&x| {
                    let mut best = (0usize, f64::MAX);
                    for (i, &y) in to.iter().enumerate() {
                        let d = (x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2);
                        if d < best.1 {
                            best = (i, d);
                        }
                    }
                    (best.0, best.1.sqrt())
                })
                .collect()
        };
        let mut b_to_a = nearest(pb, &ma);
        let a_to_b = nearest(&ma, pb);
        let matched = [a_to_b.iter().filter(|x| x.1 <= tol).count(), b_to_a.iter().filter(|x| x.1 <= tol).count()];
        // Corners that share a position (a stock UV seam inside the piece) cannot be told apart by
        // position. Triangles that mirror each other fix those: three corners, three distinct places.
        let (la, lb) = (local_tris(model, &pieces[a], &welds[pieces[a].mesh].1), local_tris(model, &pieces[b], &welds[pieces[b].mesh].1));
        let mid = |t: &[u32; 3], v: &[[f64; 3]]| scale((0..3).fold([0.0; 3], |acc, k| [acc[0] + v[t[k] as usize][0], acc[1] + v[t[k] as usize][1], acc[2] + v[t[k] as usize][2]]), 1.0 / 3.0);
        let cas: Vec<[f64; 3]> = la.iter().map(|t| mid(t, &ma)).collect();
        let mut disagree = false;
        let mut by_tri: Vec<Option<u32>> = vec![None; pb.len()];
        for tb in &lb {
            let c = mid(tb, pb);
            let Some((ti, _)) = cas.iter().enumerate().map(|(i, x)| (i, norm(sub(*x, c)))).min_by(|x, y| x.1.total_cmp(&y.1)) else { continue };
            let ta = la[ti];
            let mut used = [false; 3];
            let mut got = [0u32; 3];
            let mut ok = true;
            for k in 0..3 {
                let best = (0..3).filter(|&j| !used[j]).map(|j| (j, norm(sub(pb[tb[k] as usize], ma[ta[j] as usize])))).min_by(|x, y| x.1.total_cmp(&y.1));
                match best {
                    Some((j, d)) if d <= tol => {
                        used[j] = true;
                        got[k] = ta[j];
                    }
                    _ => ok = false,
                }
            }
            if ok {
                for k in 0..3 {
                    match by_tri[tb[k] as usize] {
                        None => by_tri[tb[k] as usize] = Some(got[k]),
                        Some(x) if x != got[k] => disagree = true,
                        _ => {}
                    }
                }
            }
        }
        for (j, e) in by_tri.iter().enumerate() {
            if let Some(i) = e {
                b_to_a[j].0 = *i as usize;
            }
        }
        let mut seen = vec![false; pa.len()];
        let one_to_one = b_to_a.iter().all(|&(i, _)| !std::mem::replace(&mut seen[i], true));
        let exact = pa.len() == pb.len() && matched[0] == pa.len() && matched[1] == pb.len() && one_to_one && !disagree;
        let max_dist = b_to_a.iter().map(|&(i, _)| i).enumerate().map(|(j, i)| norm(sub(pb[j], ma[i]))).fold(0.0, f64::max);
        let note = if exact {
            "corners match"
        } else if pa.len() != pb.len() {
            "corner counts differ"
        } else if matched[0] < pa.len() || matched[1] < pb.len() {
            "some corners have no partner"
        } else {
            "corners match by place but not one to one"
        };
        let mut mp = MirrorPair { a, b, corners: [pa.len(), pb.len()], matched, exact, locked: false, max_dist, flips: 0, note, map: Vec::new() };
        if exact {
            mp.map = b_to_a.iter().map(|x| x.0 as u32).collect();
            let uid = &welds[pieces[b].mesh].1;
            let new_uv: Vec<[f64; 2]> = mp.map.iter().map(|&i| [-pieces[a].uv[i as usize][0], pieces[a].uv[i as usize][1]]).collect();
            let lt = local_tris(model, &pieces[b], uid);
            mp.flips = lt.iter().filter(|t| area(t.map(|v| new_uv[v as usize])) < 0.0).count();
            if lock {
                if mp.flips <= pieces[a].flips {
                    let (df, ks) = (pieces[a].density_factor, pieces[a].kept_stock);
                    let fl = mp.flips;
                    let pb_ = &mut pieces[b];
                    pb_.uv = new_uv;
                    pb_.mirror_of = Some(a);
                    pb_.density_factor = df;
                    pb_.kept_stock = ks;
                    pb_.flips = fl;
                    mp.locked = true;
                    mp.note = "locked: exact reflection";
                } else {
                    mp.note = "reflection folds triangles";
                }
            }
        }
        out.push(mp);
    }
    out
}

/// Flattens a piece from its 3D surface (LSCM), or keeps its stock UV shape when that folds.
/// Fills corners, area, flips and a raw `uv`; returns the corner positions and local triangles.
fn flatten_piece(p: &mut Piece, model: &Model, uid: &[u32]) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let m = &model.meshes[p.mesh];
    p.corners.clear();
    p.kept_stock = false;
    p.flips = 0;
    let mut local: HashMap<u32, u32> = HashMap::new();
    let mut rep: Vec<usize> = Vec::new();
    let mut ltris = Vec::with_capacity(p.tris.len());
    for &t in &p.tris {
        let tri = m.tris[t as usize];
        ltris.push(tri.map(|v| {
            let u = uid[v as usize];
            *local.entry(u).or_insert_with(|| {
                rep.push(v as usize);
                p.corners.push(u);
                (rep.len() - 1) as u32
            })
        }));
    }
    let pos: Vec<[f64; 3]> = rep.iter().map(|&v| p64(m.pos[v])).collect();
    p.area3d = ltris.iter().map(|t| area3(t.map(|v| pos[v as usize]))).sum();
    p.uv = match flatten::lscm(&pos, &ltris) {
        Ok((q, flips)) if flips * 50 <= ltris.len() => {
            p.flips = flips;
            q
        }
        _ => {
            p.kept_stock = true;
            let mut s: Vec<[f64; 2]> = rep.iter().map(|&v| [m.uv[v][0] as f64, m.uv[v][1] as f64]).collect();
            if ltris.iter().map(|t| area(t.map(|v| s[v as usize]))).sum::<f64>() < 0.0 {
                s.iter_mut().for_each(|x| x[0] = -x[0]);
            }
            s
        }
    };
    (pos, ltris)
}

/// Share of a flat piece's covered area that is covered twice or more (folds).
fn overlap(q: &[[f64; 2]], tris: &[[u32; 3]]) -> f64 {
    let b = bbox(q);
    let n = 256usize;
    let sc = (n as f64 - 1.0) / (b[2] - b[0]).max(b[3] - b[1]).max(1e-12);
    let mut cnt = vec![0u8; n * n];
    for t in tris {
        let p = t.map(|v| [(q[v as usize][0] - b[0]) * sc, (q[v as usize][1] - b[1]) * sc]);
        // Strict interior (0.3 px in from every edge), so triangles that only share an edge never count.
        let den = (p[1][1] - p[2][1]) * (p[0][0] - p[2][0]) + (p[2][0] - p[1][0]) * (p[0][1] - p[2][1]);
        if den.abs() < 1e-12 {
            continue;
        }
        let lens = [0, 1, 2].map(|k| {
            let (a, b) = (p[(k + 1) % 3], p[(k + 2) % 3]);
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt().max(1e-12)
        });
        let (xs, ys) = (p.map(|v| v[0]), p.map(|v| v[1]));
        let lo = |v: [f64; 3]| (v.iter().cloned().fold(f64::MAX, f64::min).floor().max(0.0)) as usize;
        let hi = |v: [f64; 3]| (v.iter().cloned().fold(f64::MIN, f64::max).ceil().min(n as f64)) as usize;
        for y in lo(ys)..hi(ys) {
            for x in lo(xs)..hi(xs) {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let l0 = ((p[1][1] - p[2][1]) * (px - p[2][0]) + (p[2][0] - p[1][0]) * (py - p[2][1])) / den;
                let l1 = ((p[2][1] - p[0][1]) * (px - p[2][0]) + (p[0][0] - p[2][0]) * (py - p[2][1])) / den;
                let l = [l0, l1, 1.0 - l0 - l1];
                if (0..3).all(|k| l[k] * den.abs() / lens[k] >= 0.3) {
                    cnt[y * n + x] = cnt[y * n + x].saturating_add(1);
                }
            }
        }
    }
    let covered = cnt.iter().filter(|&&c| c > 0).count();
    let twice = cnt.iter().filter(|&&c| c > 1).count();
    twice as f64 / covered.max(1) as f64
}

/// Cuts a piece by the plane through its center across `axis`. Returns the connected parts
/// (triangles, and whether they lie on the +axis side), or one part when the cut changes nothing.
fn split(p: &Piece, model: &Model, uid: &[u32], axis: [f64; 3]) -> Vec<(Vec<u32>, bool)> {
    let m = &model.meshes[p.mesh];
    let cen = |t: u32| {
        let tri = m.tris[t as usize];
        scale(tri.iter().fold([0.0; 3], |a, &v| {
            let x = p64(m.pos[v as usize]);
            [a[0] + x[0], a[1] + x[1], a[2] + x[2]]
        }), 1.0 / 3.0)
    };
    let cs: Vec<[f64; 3]> = p.tris.iter().map(|&t| cen(t)).collect();
    let mid = scale(cs.iter().fold([0.0; 3], |a, c| [a[0] + c[0], a[1] + c[1], a[2] + c[2]]), 1.0 / cs.len() as f64);
    let side: Vec<bool> = cs.iter().map(|&c| dot(sub(c, mid), axis) >= 0.0).collect();
    split_sides(p, model, uid, &side, 3)
}

/// Splits a piece into the triangles facing along its mean normal and those facing away. This
/// parts two layers that lie in the same place, such as the outside of a hair card and its inside
/// joined at the rim, which no plane cut can separate.
fn split_facing(p: &Piece, model: &Model, uid: &[u32]) -> Vec<(Vec<u32>, bool)> {
    let m = &model.meshes[p.mesh];
    let nrm = |t: u32| {
        let [a, b, c] = m.tris[t as usize].map(|v| p64(m.pos[v as usize]));
        cross(sub(b, a), sub(c, a))
    };
    let ns: Vec<[f64; 3]> = p.tris.iter().map(|&t| nrm(t)).collect();
    let mut mean = ns.iter().fold([0.0; 3], |a, n| [a[0] + n[0], a[1] + n[1], a[2] + n[2]]);
    if dot(mean, mean) < 1e-24 {
        mean = *ns.iter().max_by(|a, b| dot(**a, **a).total_cmp(&dot(**b, **b))).unwrap_or(&[0.0, 0.0, 1.0]);
    }
    let side: Vec<bool> = ns.iter().map(|&n| dot(n, mean) >= 0.0).collect();
    split_sides(p, model, uid, &side, 1)
}

/// Connected parts of a piece within each side. Parts under `crumb` triangles join a neighbor.
fn split_sides(p: &Piece, model: &Model, uid: &[u32], side: &[bool], crumb: usize) -> Vec<(Vec<u32>, bool)> {
    let m = &model.meshes[p.mesh];
    // Connected parts within each side, joined through shared corners (position + UV).
    let n = p.tris.len();
    let mut par: Vec<usize> = (0..n).collect();
    let mut by_edge: HashMap<(u32, u32), usize> = HashMap::new();
    for (i, &t) in p.tris.iter().enumerate() {
        let tri = m.tris[t as usize].map(|v| uid[v as usize]);
        for c in 0..3 {
            let e = (tri[c].min(tri[(c + 1) % 3]), tri[c].max(tri[(c + 1) % 3]));
            if let Some(&j) = by_edge.get(&e) {
                if side[i] == side[j] {
                    let (a, b) = (find(&mut par, i), find(&mut par, j));
                    par[a] = b;
                }
            } else {
                by_edge.insert(e, i);
            }
        }
    }
    let mut groups: HashMap<usize, (Vec<u32>, bool)> = HashMap::new();
    for i in 0..n {
        let r = find(&mut par, i);
        groups.entry(r).or_insert_with(|| (Vec::new(), side[i])).0.push(p.tris[i]);
    }
    let mut out: Vec<(Vec<u32>, bool)> = groups.into_values().collect();
    // Crumbs (under 3 triangles) join the biggest part they share an edge with, across the cut.
    // A crumb with no such neighbor stays a piece of its own: flattening needs a piece to be
    // edge-connected, or the loose island has no pin and shrinks to a point.
    // Biggest first; equal sizes by their lowest triangle, so the pieces come out in the same order every run.
    out.sort_by_key(|g| (std::cmp::Reverse(g.0.len()), g.0.iter().min().copied()));
    let edges_of = |tris: &[u32]| -> Vec<(u32, u32)> {
        tris.iter()
            .flat_map(|&t| {
                let tri = m.tris[t as usize].map(|v| uid[v as usize]);
                (0..3).map(move |c| (tri[c].min(tri[(c + 1) % 3]), tri[c].max(tri[(c + 1) % 3])))
            })
            .collect()
    };
    let mut keep: Vec<(Vec<u32>, bool)> = Vec::new();
    let mut keep_edges: Vec<std::collections::HashSet<(u32, u32)>> = Vec::new();
    for g in out {
        let ge = edges_of(&g.0);
        if g.0.len() < crumb {
            if let Some(k) = (0..keep.len()).find(|&k| ge.iter().any(|e| keep_edges[k].contains(e))) {
                keep[k].0.extend(g.0);
                keep_edges[k].extend(ge);
                continue;
            }
        }
        keep.push(g);
        keep_edges.push(ge.into_iter().collect());
    }
    for k in keep.iter_mut() {
        k.0.sort_unstable();
    }
    keep
}

pub(crate) fn bbox(q: &[[f64; 2]]) -> [f64; 4] {
    q.iter().fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |b, p| [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])])
}

/// Shelf-packs the pieces `ids` into the unit square, part block by part block, mirror pairs side by
/// side, each piece with room for its name (`lw` wide per piece, `lh` high) above it.
/// With `fixed` = None it only finds the largest scale that fits and changes nothing; with a scale
/// it places the pieces at that scale. Nothing is placed over a `reserved` box.
/// Returns the scale (UV per model unit).
fn pack(pieces: &mut [Piece], ids: &[usize], pad: f64, lw: &[f64], lh: f64, reserved: &[[f64; 4]], fixed: Option<f64>) -> Result<f64, String> {
    if ids.is_empty() {
        return Ok(fixed.unwrap_or(f64::MAX));
    }
    // Items: one or two pieces (R then L, as seen facing the character).
    let mut items: Vec<(Part, Vec<usize>)> = Vec::new();
    let mut done = vec![false; pieces.len()];
    for &i in ids {
        if done[i] {
            continue;
        }
        done[i] = true;
        match pieces[i].twin.filter(|t| ids.contains(t)) {
            Some(t) => {
                done[t] = true;
                let (r, l) = if pieces[i].side == Side::R { (i, t) } else { (t, i) };
                items.push((pieces[i].part, vec![r, l]));
            }
            None => items.push((pieces[i].part, vec![i])),
        }
    }
    let boxes: Vec<[f64; 4]> = pieces.iter().map(|p| bbox(&p.uv)).collect();
    let pw = |i: usize, s: f64| ((boxes[i][2] - boxes[i][0]) * s).max(lw[i]);
    let dims = |it: &Vec<usize>, s: f64| -> (f64, f64) {
        let w: f64 = it.iter().map(|&i| pw(i, s)).sum::<f64>() + pad * (it.len() as f64 - 1.0);
        let h = it.iter().map(|&i| (boxes[i][3] - boxes[i][1]) * s).fold(0.0, f64::max) + lh;
        (w, h)
    };
    items.sort_by(|a, b| a.0.cmp(&b.0).then(dims(&b.1, 1.0).1.total_cmp(&dims(&a.1, 1.0).1)));
    let place = |s: f64| -> Option<Vec<(f64, f64)>> {
        let (mut x, mut y, mut row_h) = (pad, pad, 0.0f64);
        let mut cur = None;
        let mut at = Vec::with_capacity(items.len());
        for (part, it) in &items {
            let (w, h) = dims(it, s);
            let new_block = cur.is_some() && cur != Some(*part);
            if (x > pad && x + w > 1.0 - pad) || (new_block && x > pad) {
                x = pad;
                y += row_h + pad;
                row_h = 0.0;
            }
            // Step right past reserved boxes; wrap to a new row (below the box on an empty row).
            while let Some(r) = reserved.iter().find(|r| x < r[2] && x + w > r[0] && y < r[3] && y + h > r[1]) {
                x = r[2] + pad;
                if x + w > 1.0 - pad {
                    x = pad;
                    y = if row_h > 0.0 { y + row_h + pad } else { r[3] + pad };
                    row_h = 0.0;
                }
                if y + h > 1.0 - pad + 1e-12 {
                    return None;
                }
            }
            if x + w > 1.0 - pad + 1e-12 || y + h > 1.0 - pad + 1e-12 {
                return None;
            }
            at.push((x, y));
            x += w + pad;
            row_h = row_h.max(h);
            cur = Some(*part);
        }
        Some(at)
    };
    let Some(lo) = fixed else {
        let biggest = items.iter().map(|(_, it)| {
            let (w, h) = dims(it, 1.0);
            w.max(h)
        });
        let mut hi = 1.0 / biggest.fold(1e-12, f64::max);
        let mut lo = 0.0;
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if place(mid).is_some() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if lo <= 0.0 {
            return Err("piece names do not fit the sheet".into());
        }
        return Ok(lo);
    };
    let at = place(lo).ok_or("pieces do not fit the texture")?;
    for ((_, it), (x0, y0)) in items.iter().zip(at) {
        let mut x = x0;
        for &i in it {
            let b = boxes[i];
            let w = pw(i, lo);
            let cx = x + 0.5 * (w - (b[2] - b[0]) * lo);
            for q in pieces[i].uv.iter_mut() {
                *q = [cx + (q[0] - b[0]) * lo, y0 + lh + (q[1] - b[1]) * lo];
            }
            x += w + pad;
        }
    }
    Ok(lo)
}

fn sample(t: &Texture, u: f64, v: f64) -> [f32; 4] {
    let (w, h) = (t.w as f64, t.h as f64);
    let x = (u * w - 0.5).clamp(0.0, w - 1.0);
    let y = (v * h - 0.5).clamp(0.0, h - 1.0);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(t.w - 1), (y0 + 1).min(t.h - 1));
    let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
    let px = |xx: usize, yy: usize| {
        let i = (yy * t.w + xx) * 4;
        [t.rgba[i] as f32, t.rgba[i + 1] as f32, t.rgba[i + 2] as f32, t.rgba[i + 3] as f32]
    };
    let (a, b, c, d) = (px(x0, y0), px(x1, y0), px(x0, y1), px(x1, y1));
    std::array::from_fn(|k| (a[k] * (1.0 - fx) + b[k] * fx) * (1.0 - fy) + (c[k] * (1.0 - fx) + d[k] * fx) * fy)
}

/// Calls `f(x, y, l0, l1, l2)` for pixels whose center is inside (or within `grow` texels of) the
/// triangle, with barycentrics clamped to the triangle.
pub(crate) fn raster_bary(p: [[f64; 2]; 3], w: usize, h: usize, grow: f64, mut f: impl FnMut(usize, usize, [f64; 3])) {
    let den = (p[1][1] - p[2][1]) * (p[0][0] - p[2][0]) + (p[2][0] - p[1][0]) * (p[0][1] - p[2][1]);
    if den.abs() < 1e-12 {
        return;
    }
    let xs = [p[0][0], p[1][0], p[2][0]];
    let ys = [p[0][1], p[1][1], p[2][1]];
    let x0 = ((xs.iter().cloned().fold(f64::MAX, f64::min) - grow - 0.5).floor().max(0.0)) as usize;
    let x1 = ((xs.iter().cloned().fold(f64::MIN, f64::max) + grow + 0.5).ceil().min(w as f64)) as usize;
    let y0 = ((ys.iter().cloned().fold(f64::MAX, f64::min) - grow - 0.5).floor().max(0.0)) as usize;
    let y1 = ((ys.iter().cloned().fold(f64::MIN, f64::max) + grow + 0.5).ceil().min(h as f64)) as usize;
    // Edge lengths turn the barycentric slack into a distance.
    let lens = [0, 1, 2].map(|k| {
        let (a, b) = (p[(k + 1) % 3], p[(k + 2) % 3]);
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
    });
    let area2 = den.abs();
    for y in y0..y1 {
        let py = y as f64 + 0.5;
        for x in x0..x1 {
            let px = x as f64 + 0.5;
            let l0 = ((p[1][1] - p[2][1]) * (px - p[2][0]) + (p[2][0] - p[1][0]) * (py - p[2][1])) / den;
            let l1 = ((p[2][1] - p[0][1]) * (px - p[2][0]) + (p[0][0] - p[2][0]) * (py - p[2][1])) / den;
            let l = [l0, l1, 1.0 - l0 - l1];
            // Distance outside edge k is -l_k * 2A / |edge k|.
            let out = (0..3).map(|k| if l[k] < 0.0 { -l[k] * area2 / lens[k].max(1e-12) } else { 0.0 }).fold(0.0, f64::max);
            if out <= grow {
                let c = l.map(|v| v.max(0.0));
                let s = c[0] + c[1] + c[2];
                f(x, y, c.map(|v| v / s));
            }
        }
    }
}

/// Copies the stock paint inside every kept box of `atlas` (bilinear, scaled to `size`), so art the
/// game copies at run time (such as closed eyes for a blink) is where it expects.
pub fn copy_keep(o: &Options, atlas: &Atlas, src: &Texture, rgba: &mut [u8], size: usize) -> usize {
    let mut n = 0;
    for k in o.keep.iter().filter(|k| k.sheet == atlas.name) {
        let b = k.uv;
        let (x0, y0) = ((b[0] * size as f64).floor().max(0.0) as usize, (b[1] * size as f64).floor().max(0.0) as usize);
        let (x1, y1) = ((b[2] * size as f64).ceil().min(size as f64) as usize, (b[3] * size as f64).ceil().min(size as f64) as usize);
        for y in y0..y1 {
            for x in x0..x1 {
                let c = sample(src, (x as f64 + 0.5) / size as f64, (y as f64 + 0.5) / size as f64);
                let i = (y * size + x) * 4;
                for ch in 0..4 {
                    rgba[i + ch] = c[ch].round().clamp(0.0, 255.0) as u8;
                }
                n += 1;
            }
        }
    }
    n
}

/// The old texture copied onto the new layout at `size`, bled `bleed` texels past every piece.
/// Returns RGBA and the covered-texel share.
pub fn transfer(model: &Model, lay: &Layout, atlas: &Atlas, src: &Texture, size: usize, bleed: usize) -> (Vec<u8>, f64) {
    let mut acc = vec![0f32; size * size * 4];
    let mut hit = vec![0u8; size * size];
    let s = size as f64;
    for &mi in &atlas.meshes {
        let m = &model.meshes[mi];
        let nm = &lay.meshes[mi];
        for (t, tri) in m.tris.iter().enumerate() {
            let old = tri.map(|v| [m.uv[v as usize][0] as f64, m.uv[v as usize][1] as f64]);
            let new = nm.tris[t].map(|v| [nm.uv[v as usize][0] as f64 * s, nm.uv[v as usize][1] as f64 * s]);
            if area(new).abs() < 1e-9 {
                continue;
            }
            // Inside pixels first (exact), then a 1-texel rim so seams have no gaps.
            raster_bary(new, size, size, 1.0, |x, y, l| {
                let i = y * size + x;
                let inside = {
                    let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                    let d = |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]);
                    let (d0, d1, d2) = (d(new[0], new[1]), d(new[1], new[2]), d(new[2], new[0]));
                    (d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0) || (d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0)
                };
                let rank = if inside { 2 } else { 1 };
                if hit[i] > rank {
                    return;
                }
                let u = l[0] * old[0][0] + l[1] * old[1][0] + l[2] * old[2][0];
                let v = l[0] * old[0][1] + l[1] * old[1][1] + l[2] * old[2][1];
                let c = sample(src, u, v);
                acc[i * 4..i * 4 + 4].copy_from_slice(&c);
                hit[i] = rank;
            });
        }
    }
    let covered = hit.iter().filter(|&&h| h == 2).count() as f64 / (size * size) as f64;
    // Bleed: grow the painted area outward, one ring per pass, averaging painted neighbors.
    let mut front: Vec<usize> = (0..size * size).filter(|&i| hit[i] == 0 && neighbors(i, size).any(|j| hit[j] > 0)).collect();
    for _ in 0..bleed {
        if front.is_empty() {
            break;
        }
        let mut fills = Vec::with_capacity(front.len());
        for &i in &front {
            let (mut c, mut n) = ([0f32; 4], 0f32);
            for j in neighbors(i, size) {
                if hit[j] > 0 {
                    for k in 0..4 {
                        c[k] += acc[j * 4 + k];
                    }
                    n += 1.0;
                }
            }
            if n > 0.0 {
                fills.push((i, c.map(|x| x / n)));
            }
        }
        let mut next = Vec::new();
        for (i, c) in fills {
            acc[i * 4..i * 4 + 4].copy_from_slice(&c);
            hit[i] = 1;
        }
        for &i in &front {
            for j in neighbors(i, size) {
                if hit[j] == 0 {
                    next.push(j);
                }
            }
        }
        next.sort_unstable();
        next.dedup();
        front = next;
    }
    // Everything else: the mean painted color.
    let (mut mean, mut n) = ([0f64; 4], 0f64);
    for i in 0..size * size {
        if hit[i] == 2 {
            for k in 0..4 {
                mean[k] += acc[i * 4 + k] as f64;
            }
            n += 1.0;
        }
    }
    let mean = mean.map(|x| (x / n.max(1.0)) as f32);
    let mut out = vec![0u8; size * size * 4];
    for i in 0..size * size {
        let c = if hit[i] > 0 { [acc[i * 4], acc[i * 4 + 1], acc[i * 4 + 2], acc[i * 4 + 3]] } else { mean };
        for k in 0..4 {
            out[i * 4 + k] = c[k].round().clamp(0.0, 255.0) as u8;
        }
    }
    (out, covered)
}

fn neighbors(i: usize, size: usize) -> impl Iterator<Item = usize> {
    let (x, y) = (i % size, i / size);
    let mut v = [usize::MAX; 4];
    if x > 0 {
        v[0] = i - 1;
    }
    if x + 1 < size {
        v[1] = i + 1;
    }
    if y > 0 {
        v[2] = i - size;
    }
    if y + 1 < size {
        v[3] = i + size;
    }
    v.into_iter().filter(|&j| j != usize::MAX)
}

fn line(rgba: &mut [u8], w: usize, h: usize, a: [f64; 2], b: [f64; 2], r: i64, c: [u8; 3]) {
    let steps = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()).ceil() as usize).max(1);
    for s in 0..=steps {
        let t = s as f64 / steps as f64;
        let (x, y) = ((a[0] + (b[0] - a[0]) * t) as i64, (a[1] + (b[1] - a[1]) * t) as i64);
        for dy in -r..=r {
            for dx in -r..=r {
                let (px, py) = (x + dx, y + dy);
                if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                    let i = (py as usize * w + px as usize) * 4;
                    rgba[i..i + 3].copy_from_slice(&c);
                    rgba[i + 3] = 255;
                }
            }
        }
    }
}

pub(crate) fn seam_color(id: usize) -> [u8; 3] {
    let hue = (id as f64 * 0.618_033_988_75).fract() * 6.0;
    let (i, f) = (hue.floor() as i32, hue.fract());
    let (p, q, t) = (0.15, 1.0 - 0.85 * f, 0.15 + 0.85 * f);
    let (r, g, b) = match i {
        0 => (1.0, t, p),
        1 => (q, 1.0, p),
        2 => (p, 1.0, t),
        3 => (p, q, 1.0),
        4 => (t, p, 1.0),
        _ => (1.0, p, q),
    };
    [(r * 200.0) as u8, (g * 200.0) as u8, (b * 200.0) as u8]
}

/// Maps for painting tools and AI, at `gs` x `gs`: the pattern sheet, part colors, rest-pose
/// position (normalized to the model box) and object-space normals.
pub struct Guides {
    pub size: usize,
    pub sheet: Vec<u8>,
    pub parts: Vec<u8>,
    pub position: Vec<u8>,
    pub normal: Vec<u8>,
}

pub fn guides(model: &Model, lay: &Layout, atlas: &Atlas, paint: Option<(&[u8], usize)>, gs: usize) -> Guides {
    let n = gs * gs * 4;
    let (mut parts, mut position, mut normal) = (vec![0u8; n], vec![0u8; n], vec![0u8; n]);
    let mut sheet = vec![255u8; n];
    let s = gs as f64;
    let (lo, hi) = crate::render::bounds(model);
    // Smooth normals per welded position.
    let mut normals: Vec<Vec<[f64; 3]>> = Vec::new();
    for m in &model.meshes {
        let mut acc: HashMap<[i64; 3], [f64; 3]> = HashMap::new();
        let key = |p: [f32; 3]| [qk(p[0], 1e-4), qk(p[1], 1e-4), qk(p[2], 1e-4)];
        for tri in &m.tris {
            let p = tri.map(|v| p64(m.pos[v as usize]));
            let fnrm = cross(sub(p[1], p[0]), sub(p[2], p[0]));
            for &v in tri {
                let e = acc.entry(key(m.pos[v as usize])).or_insert([0.0; 3]);
                *e = [e[0] + fnrm[0], e[1] + fnrm[1], e[2] + fnrm[2]];
            }
        }
        normals.push(m.pos.iter().map(|&p| {
            let x = acc[&key(p)];
            let l = norm(x).max(1e-18);
            scale(x, 1.0 / l)
        }).collect());
    }
    let mut owner = vec![usize::MAX; gs * gs];
    for &mi in &atlas.meshes {
        let m = &model.meshes[mi];
        let nm = &lay.meshes[mi];
        for (t, tri) in m.tris.iter().enumerate() {
            let new = nm.tris[t].map(|v| [nm.uv[v as usize][0] as f64 * s, nm.uv[v as usize][1] as f64 * s]);
            let pos = tri.map(|v| p64(m.pos[v as usize]));
            let nrm = tri.map(|v| normals[mi][v as usize]);
            raster_bary(new, gs, gs, 0.0, |x, y, l| {
                let i = (y * gs + x) * 4;
                let mix = |a: [[f64; 3]; 3]| [0, 1, 2].map(|k| l[0] * a[0][k] + l[1] * a[1][k] + l[2] * a[2][k]);
                let p = mix(pos);
                let q = mix(nrm);
                let ql = norm(q).max(1e-18);
                for k in 0..3 {
                    position[i + k] = (((p[k] - lo[k] as f64) / ((hi[k] - lo[k]) as f64).max(1e-9)) * 255.0).round().clamp(0.0, 255.0) as u8;
                    normal[i + k] = ((q[k] / ql * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
                }
                position[i + 3] = 255;
                normal[i + 3] = 255;
                owner[y * gs + x] = mi << 20 | t;
            });
        }
    }
    for y in 0..gs {
        for x in 0..gs {
            let o = owner[y * gs + x];
            if o == usize::MAX {
                continue;
            }
            let pi = lay.piece_of[o >> 20][o & 0xFFFFF];
            let c = lay.pieces[pi].part.color();
            let i = (y * gs + x) * 4;
            parts[i..i + 3].copy_from_slice(&c);
            parts[i + 3] = 255;
            // Sheet: the paint at 35% over white, tinted by the part color where there is no paint.
            let base = match paint {
                Some((rgba, ps)) => {
                    let (sx, sy) = (x * ps / gs, y * ps / gs);
                    let j = (sy * ps + sx) * 4;
                    [rgba[j], rgba[j + 1], rgba[j + 2]]
                }
                None => c,
            };
            for k in 0..3 {
                sheet[i + k] = (255.0 * 0.6 + base[k] as f64 * 0.4) as u8;
            }
        }
    }
    // Paint-boundary lines: where the paint's color jumps inside a piece (e.g. shirt to belt on one
    // polygon), so a repaint keeps the same material borders.
    if let Some((rgba, ps)) = paint {
        let at = |x: usize, y: usize| {
            let (sx, sy) = (x * ps / gs, y * ps / gs);
            let j = (sy * ps + sx) * 4;
            [rgba[j] as i32, rgba[j + 1] as i32, rgba[j + 2] as i32]
        };
        for y in 1..gs - 1 {
            for x in 1..gs - 1 {
                if owner[y * gs + x] == usize::MAX || owner[y * gs + x + 1] == usize::MAX || owner[(y + 1) * gs + x] == usize::MAX {
                    continue;
                }
                let (c, r, d) = (at(x, y), at(x + 1, y), at(x, y + 1));
                let jump = (0..3).map(|k| (c[k] - r[k]).abs().max((c[k] - d[k]).abs())).max().unwrap_or(0);
                if jump > 48 && (x + y) % 3 != 0 {
                    let i = (y * gs + x) * 4;
                    sheet[i..i + 3].copy_from_slice(&[90, 90, 90]);
                }
            }
        }
    }
    // Outlines: every piece's open edges in black, seams thicker in their pair color.
    // Edges are keyed by their UV end points, so split vertices at one corner still join up.
    let qk2 = |u: [f32; 2]| [(u[0] as f64 * 1e6).round() as i64, (u[1] as f64 * 1e6).round() as i64];
    for p in lay.pieces.iter().filter(|p| atlas.meshes.contains(&p.mesh)) {
        let nm = &lay.meshes[p.mesh];
        let mut count: HashMap<([i64; 2], [i64; 2]), ([f32; 2], [f32; 2], usize)> = HashMap::new();
        for &t in &p.tris {
            let tri = nm.tris[t as usize];
            for c in 0..3 {
                let (ua, ub) = (nm.uv[tri[c] as usize], nm.uv[tri[(c + 1) % 3] as usize]);
                let (ka, kb) = (qk2(ua), qk2(ub));
                let e = count.entry(if ka < kb { (ka, kb) } else { (kb, ka) }).or_insert((ua, ub, 0));
                e.2 += 1;
            }
        }
        for (ua, ub, k) in count.into_values() {
            if k == 1 {
                line(&mut sheet, gs, gs, [ua[0] as f64 * s, ua[1] as f64 * s], [ub[0] as f64 * s, ub[1] as f64 * s], 1, [20, 20, 20]);
            }
        }
    }
    // Same text scale the packer reserved room for (see `layout`).
    // Above 2048 px the text grows in whole steps so it takes the same share of the sheet.
    let base = gs.min(2048);
    let scale_txt = label_scale(base) * (gs / base).max(1);
    for seam in &lay.seams {
        if !atlas.meshes.contains(&lay.pieces[seam.a].mesh) {
            continue;
        }
        let c = seam_color(seam.id);
        for (ea, eb) in &seam.edges {
            for e in [ea, eb] {
                line(&mut sheet, gs, gs, [e[0][0] * s, e[0][1] * s], [e[1][0] * s, e[1][1] * s], 2, c);
            }
        }
        // Number at the middle edge on both sides, pushed into the piece a little.
        let mid = &seam.edges[seam.edges.len() / 2];
        let tag = format!("{}", seam.id);
        for (e, pi) in [(mid.0, seam.a), (mid.1, seam.b)] {
            let cx = (e[0][0] + e[1][0]) * 0.5 * s;
            let cy = (e[0][1] + e[1][1]) * 0.5 * s;
            let pb = bbox(&lay.pieces[pi].uv);
            let (pcx, pcy) = ((pb[0] + pb[2]) * 0.5 * s, (pb[1] + pb[3]) * 0.5 * s);
            let d = ((pcx - cx).powi(2) + (pcy - cy).powi(2)).sqrt().max(1e-9);
            let push = (6.0 * scale_txt as f64).min(d * 0.5);
            let (tx, ty) = (cx + (pcx - cx) / d * push, cy + (pcy - cy) / d * push);
            let w = font::width(&tag, scale_txt) as f64;
            font::draw(&mut sheet, gs, gs, (tx - w * 0.5) as i64, (ty - 3.5 * scale_txt as f64) as i64, &tag, scale_txt, c.map(|v| v / 2));
        }
    }
    for p in lay.pieces.iter().filter(|p| atlas.meshes.contains(&p.mesh)) {
        let b = bbox(&p.uv);
        let w = font::width(&p.label, scale_txt) as f64;
        let x = ((b[0] + b[2]) * 0.5 * s - w * 0.5) as i64;
        // In the strip the packer left above the piece.
        let y = (b[1] * s) as i64 - (9 * scale_txt as i64 + 2) + scale_txt as i64;
        font::draw(&mut sheet, gs, gs, x, y, &p.label, scale_txt, [0, 0, 0]);
    }
    Guides { size: gs, sheet, parts, position, normal }
}

/// The legend: pieces, seams and atlases as JSON (for people and tools).
pub fn legend(lay: &Layout) -> Value {
    let all = [Part::Head, Part::Torso, Part::Tail, Part::Arm, Part::Hand, Part::Leg, Part::Foot, Part::Other];
    let parts: Vec<Value> = all.iter().map(|p| json!({"part": p.name(), "color": p.color()})).collect();
    json!({
        "size": lay.size,
        "texels_per_unit": (lay.texels_per_unit * 100.0).round() / 100.0,
        "parts": parts,
        "pieces": lay.pieces.iter().enumerate().map(|(i, p)| {
            let b = bbox(&p.uv);
            json!({
                "id": i, "label": p.label, "mesh": p.mesh, "part": p.part.name(), "side": p.side.name(),
                "triangles": p.tris.len(), "uv_box": b.map(|x| (x * 10000.0).round() / 10000.0),
                "density_factor": p.density_factor, "twin": p.twin, "folded_triangles": p.flips, "kept_stock_shape": p.kept_stock, "pinned_at_stock": p.pinned,
            })
        }).collect::<Vec<_>>(),
        "seams": lay.seams.iter().map(|s| json!({"id": s.id, "a": s.a, "b": s.b, "a_label": lay.pieces[s.a].label, "b_label": lay.pieces[s.b].label, "edges": s.edges.len()})).collect::<Vec<_>>(),
        "atlases": lay.atlases.iter().map(|a| json!({"image": a.name, "meshes": a.meshes, "coverage": (a.coverage * 1000.0).round() / 1000.0})).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bone_labels() {
        assert_eq!(classify("bone016_R_hand"), (Part::Hand, Side::R));
        assert_eq!(classify("bone025_L_upper_arm"), (Part::Arm, Side::L));
        assert_eq!(classify("bone004_R_hip"), (Part::Leg, Side::R));
        assert_eq!(classify("bone001_hips"), (Part::Torso, Side::C));
        assert_eq!(classify("bone023_crest_01"), (Part::Head, Side::C));
        assert_eq!(classify("bone007_R_foot"), (Part::Foot, Side::R));
        assert_eq!(classify("bone003_tail_01"), (Part::Tail, Side::C));
        assert_eq!(classify("bone012"), (Part::Other, Side::C));
    }
}
