//! Mark & fix, hair tools, for any card hair or fur:
//! - Comb: drag on the model the way the hair should flow (root to tip). Strokes are saved as
//!   `strokes.json` (triangle plus barycentric points) for the hair tool to follow.
//! - Flow overlay: the current flow direction as short arrows on the model and the texture, with
//!   breaks (neighbors whose flow turns sharply) in red. Read from the hair tool's `flow.json`, or
//!   a quick estimate from the paint of the marked triangles (`gaw hair` runs more steps).
//! - Defects: load `defects.json` (from `gaw defects`), click one to jump to it with its triangles
//!   marked and its texels outlined, and turn one or more into a Mark & fix request.
//! Nothing here knows about one game: it works on triangles, barycentric points and UVs.
use super::*;
use serde_json::{json, Value};
use wb_core::hair::{self, FlowOpts, HairSet, Root, Select, Sheet};

/// Flow neighbors that turn more than this many degrees count as a break.
pub const BREAK_DEG: f64 = 60.0;
/// A comb stroke that runs more than this many degrees off the flow disagrees with it.
pub const DISAGREE_DEG: f32 = 60.0;

#[derive(Clone, Debug)]
pub struct CombPt {
    pub mesh: usize,
    pub tri: usize,
    pub bary: [f32; 3],
}

#[derive(Clone, Debug, Default)]
pub struct CombStroke {
    pub pts: Vec<CombPt>,
}

pub struct FlowTri {
    pub mesh: usize,
    pub tri: usize,
    /// Direction as a*(p1-p0) + b*(p2-p0) in the triangle's own corners, so it follows the pose
    /// and maps straight to UV.
    pub coef: [f32; 2],
    pub ambiguous: bool,
    /// Rest-pose corners.
    pub corners: [V3; 3],
}

pub struct FlowField {
    pub source: String,
    pub tris: Vec<FlowTri>,
    /// (index into tris, index into tris, degrees, shared corner pair in the first triangle).
    pub breaks: Vec<(usize, usize, f32, [usize; 2])>,
    by_tri: HashMap<(usize, usize), usize>,
}

#[derive(Clone)]
pub struct Defect {
    pub id: String,
    pub kind: String,
    /// Sort key: `severity` (0..1) or the checker's `score` (1.0 = at the limit).
    pub severity: f32,
    /// What the list shows for it, e.g. "0.80" or "high 2.10".
    pub grade: String,
    /// 2 red, 1 gold, 0 gray.
    pub level: u8,
    pub summary: String,
    pub part: Option<String>,
    pub at: Option<V3>,
    pub view: Option<Camera>,
    pub clip: Option<String>,
    pub time: f32,
    pub tris: Vec<(usize, usize)>,
    /// (image, pixel box x0, y0, x1, y1)
    pub texels: Vec<(usize, [f32; 4])>,
    pub raw: Value,
}

#[derive(Default)]
pub struct HairTools {
    pub strokes: Vec<CombStroke>,
    pub flow_on: bool,
    pub flow: Option<FlowField>,
    /// Comb points checked against the flow, and the ones that run against it (stroke, point).
    pub disagree: (usize, Vec<(usize, usize)>),
    pub defects: Vec<Defect>,
    pub defects_file: Option<String>,
    pub checked: Vec<bool>,
    pub current: Option<usize>,
    /// Defects to attach to the next saved request.
    pub pending_defects: Option<Value>,
    /// Scroll the side panel to the hair cards on the next frame.
    pub show_cards: bool,
}

fn f3(v: &Value) -> Option<[f32; 3]> {
    let a = v.as_array()?;
    Some([a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32, a.get(2)?.as_f64()? as f32])
}

/// Least-squares a, b with d = a*e1 + b*e2.
fn coef_of(p: [V3; 3], d: V3) -> [f32; 2] {
    let (e1, e2) = (sub(p[1], p[0]), sub(p[2], p[0]));
    let (a11, a12, a22) = (wb_core::geom::dot(e1, e1), wb_core::geom::dot(e1, e2), wb_core::geom::dot(e2, e2));
    let (b1, b2) = (wb_core::geom::dot(d, e1), wb_core::geom::dot(d, e2));
    let det = a11 * a22 - a12 * a12;
    if det.abs() < 1e-20 {
        return [0.0, 0.0];
    }
    [(b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det]
}

fn bary_of(p: V3, t: [V3; 3]) -> [f32; 3] {
    let (_, b) = wb_core::geom::closest_on_tri(p, t[0], t[1], t[2]);
    b
}

impl FlowField {
    /// Builds the field and, when `breaks` is None, finds breaks between welded neighbors.
    fn build(model: &wb_core::Model, rest: &[Vec<V3>], source: String, mesh: usize, items: Vec<(usize, V3, bool)>, breaks: Option<Vec<(usize, usize, f32)>>) -> FlowField {
        let m = &model.meshes[mesh];
        let corners = |ti: usize| m.tris[ti].map(|v| rest[mesh][v as usize]);
        let mut tris = Vec::new();
        let mut by_tri = HashMap::new();
        let mut dirs: HashMap<usize, V3> = HashMap::new();
        for (ti, d, amb) in items {
            if ti >= m.tris.len() || by_tri.contains_key(&(mesh, ti)) {
                continue;
            }
            by_tri.insert((mesh, ti), tris.len());
            dirs.insert(ti, d);
            tris.push(FlowTri { mesh, tri: ti, coef: coef_of(corners(ti), d), ambiguous: amb, corners: corners(ti) });
        }
        let shared = |a: usize, b: usize| -> [usize; 2] {
            let (ca, cb) = (corners(a), corners(b));
            let mut s = vec![];
            for k in 0..3 {
                if cb.iter().any(|q| dist(*q, ca[k]) < 1e-5) {
                    s.push(k);
                }
            }
            [s.first().copied().unwrap_or(0), s.get(1).copied().unwrap_or(1)]
        };
        let mut out = vec![];
        match breaks {
            Some(list) => {
                for (a, b, deg) in list {
                    if let (Some(&ia), Some(&ib)) = (by_tri.get(&(mesh, a)), by_tri.get(&(mesh, b))) {
                        out.push((ia, ib, deg, shared(a, b)));
                    }
                }
            }
            None => {
                let mask: Vec<bool> = (0..m.tris.len()).map(|t| dirs.contains_key(&t)).collect();
                if let Ok(hs) = HairSet::build(model, mesh, &Select::Mask(mask)) {
                    let nb = hair::neighbors(&hs);
                    let d64 = |ti: usize| dirs.get(&ti).map(|d| [d[0] as f64, d[1] as f64, d[2] as f64]).unwrap_or([0.0; 3]);
                    for i in 0..hs.tris.len() {
                        let di = d64(hs.tris[i]);
                        if hair::len(di) < 0.5 {
                            continue;
                        }
                        for &(j, e) in &nb[i] {
                            if j <= i {
                                continue;
                            }
                            let dj = d64(hs.tris[j]);
                            if hair::len(dj) < 0.5 {
                                continue;
                            }
                            let deg = hair::ang(di, hair::hinge(&hs, j, i, e, dj));
                            if deg > BREAK_DEG {
                                let (a, b) = (hs.tris[i], hs.tris[j]);
                                out.push((by_tri[&(mesh, a)], by_tri[&(mesh, b)], deg as f32, shared(a, b)));
                            }
                        }
                    }
                }
            }
        }
        FlowField { source, tris, breaks: out, by_tri }
    }

    pub fn get(&self, mesh: usize, tri: usize) -> Option<&FlowTri> {
        self.by_tri.get(&(mesh, tri)).map(|&i| &self.tris[i])
    }
}

impl Mark {
    /// The posed or rest position of a comb point.
    fn comb_at(model: &wb_core::Model, pos: &[Vec<V3>], p: &CombPt) -> Option<V3> {
        let m = model.meshes.get(p.mesh)?;
        let t = m.tris.get(p.tri)?;
        let c = t.map(|v| pos[p.mesh][v as usize]);
        Some(add(add(scale(c[0], p.bary[0]), scale(c[1], p.bary[1])), scale(c[2], p.bary[2])))
    }

    /// A comb point under a screen position, or None off the model.
    pub(super) fn comb_point(&self, p: Pos2, d: &Doc, topo: &Topo, cam: &Camera) -> Option<CombPt> {
        let b = self.to_buf(p);
        let (g, hit, _) = self.surface_at(d, topo, cam, b)?;
        let (mi, ti) = topo.tri_loc[g];
        let c = d.model.meshes[mi].tris[ti].map(|v| self.pose[mi][v as usize]);
        Some(CombPt { mesh: mi, tri: ti, bary: bary_of(hit, c) })
    }

    pub(super) fn comb_press(&mut self, place: Where, p: Pos2, mods: Mods, d: &Doc, topo: &Topo, cam: &Camera) {
        if place != Where::Model {
            self.say("Comb strokes are drawn on the model".into(), false);
            return;
        }
        if mods.ctrl {
            // Remove the stroke nearest the pointer.
            let rig = Rig::new(cam, self.vb.w, self.vb.h);
            let mut best: Option<(usize, f32)> = None;
            for (si, s) in self.hair.strokes.iter().enumerate() {
                for q in &s.pts {
                    if let Some(w) = Self::comb_at(&d.model, &self.pose, q) {
                        let pr = rig.project(w);
                        let dd = self.to_screen([pr[0], pr[1]]).distance(p);
                        if best.is_none_or(|b| dd < b.1) {
                            best = Some((si, dd));
                        }
                    }
                }
            }
            match best {
                Some((si, dd)) if dd < 16.0 => {
                    self.hair.strokes.remove(si);
                    self.comb_changed();
                    self.say(format!("Comb stroke removed ({} left)", self.hair.strokes.len()), true);
                }
                _ => self.say("No comb stroke near the pointer".into(), false),
            }
            return;
        }
        let Some(q) = self.comb_point(p, d, topo, cam) else {
            self.say("Start the comb stroke on the model".into(), false);
            return;
        };
        self.drag = Some(Drag::Comb { pts: vec![q], last: p });
    }

    pub(super) fn comb_move(&mut self, p: Pos2, d: &Doc, topo: &Topo, cam: &Camera) {
        let far = matches!(&self.drag, Some(Drag::Comb { last, .. }) if last.distance(p) >= 6.0);
        if !far {
            return;
        }
        let q = self.comb_point(p, d, topo, cam);
        if let Some(Drag::Comb { pts, last }) = &mut self.drag {
            if let Some(q) = q {
                pts.push(q);
            }
            *last = p;
        }
    }

    pub(super) fn comb_release(&mut self, pts: Vec<CombPt>) {
        if pts.len() < 2 {
            self.say("Comb stroke too short: drag along the hair, root to tip".into(), false);
            return;
        }
        let n = pts.len();
        self.hair.strokes.push(CombStroke { pts });
        self.comb_changed();
        let dis = self.hair.disagree.1.iter().filter(|(s, _)| *s + 1 == self.hair.strokes.len()).count();
        self.say(
            format!("Comb stroke {} ({n} points){}", self.hair.strokes.len(), if self.hair.flow.is_some() && dis > 0 { format!(", {dis} points run against the current flow") } else { String::new() }),
            true,
        );
    }

    fn comb_changed(&mut self) {
        self.recheck_comb();
    }

    /// Compares comb strokes with the flow (rest pose).
    fn recheck_comb(&mut self) {
        let Some(f) = &self.hair.flow else {
            self.hair.disagree = (0, vec![]);
            return;
        };
        let mut checked = 0;
        let mut bad = vec![];
        for (si, s) in self.hair.strokes.iter().enumerate() {
            for k in 0..s.pts.len().saturating_sub(1) {
                // Look a few points ahead (6 px steps are noisy across stacked cards) and compare
                // in the plane of the triangle under the point.
                let a = &s.pts[k];
                let Some(fa) = f.get(a.mesh, a.tri) else { continue };
                let at = |t: &[V3; 3], w: [f32; 3]| add(add(scale(t[0], w[0]), scale(t[1], w[1])), scale(t[2], w[2]));
                let pa = at(&fa.corners, a.bary);
                let ahead = (k + 1..s.pts.len().min(k + 4)).rev().find_map(|j| {
                    let b = &s.pts[j];
                    f.get(b.mesh, b.tri).map(|fb| at(&fb.corners, b.bary))
                });
                let Some(pb) = ahead else { continue };
                let tri = fa.corners;
                let nrm = norm(cross(sub(tri[1], tri[0]), sub(tri[2], tri[0])));
                let raw = sub(pb, pa);
                let sd = sub(raw, scale(nrm, wb_core::geom::dot(raw, nrm)));
                let fd = add(scale(sub(tri[1], tri[0]), fa.coef[0]), scale(sub(tri[2], tri[0]), fa.coef[1]));
                if wb_core::geom::len(sd) < 1e-6 || wb_core::geom::len(fd) < 1e-9 {
                    continue;
                }
                checked += 1;
                let c = (wb_core::geom::dot(norm(sd), norm(fd))).clamp(-1.0, 1.0).acos().to_degrees();
                if c > DISAGREE_DEG {
                    bad.push((si, k));
                }
            }
        }
        self.hair.disagree = (checked, bad);
    }

    // ---------------------------------------------------------------------------------------------
    // Strokes file
    // ---------------------------------------------------------------------------------------------

    pub(super) fn strokes_json(&self, d: &Doc) -> Option<Value> {
        if self.hair.strokes.is_empty() {
            return None;
        }
        let strokes: Vec<Value> = self
            .hair
            .strokes
            .iter()
            .map(|s| {
                let mesh = s.pts[0].mesh;
                json!({
                    "mesh": mesh,
                    "mesh_name": d.model.meshes.get(mesh).map(|m| m.name.clone()),
                    "points": s.pts.iter().map(|p| json!({"mesh": p.mesh, "tri": p.tri, "bary": p.bary})).collect::<Vec<_>>(),
                    "positions": s.pts.iter().filter_map(|p| Self::comb_at(&d.model, &self.mesh_pos, p)).collect::<Vec<_>>(),
                })
            })
            .collect();
        Some(json!({
            "gaw_strokes": 1,
            "source": d.path.display().to_string(),
            "about": "Comb strokes from Mark & fix: the way the hair should flow, root to tip. Each point is a triangle (index into meshes[mesh].tris) and barycentric weights; positions are the rest-pose places, for readers without the GLB.",
            "strokes": strokes,
        }))
    }

    pub(super) fn save_strokes(&mut self, d: &Doc, path: &Path) {
        let Some(j) = self.strokes_json(d) else {
            self.say("No comb strokes to save".into(), false);
            return;
        };
        let r = path.parent().map(std::fs::create_dir_all).unwrap_or(Ok(())).map_err(|e| e.to_string()).and_then(|_| std::fs::write(path, serde_json::to_string_pretty(&j).unwrap()).map_err(|e| e.to_string()));
        match r {
            Ok(()) => self.say(format!("Saved {} comb strokes to {}", self.hair.strokes.len(), path.display()), true),
            Err(e) => self.say(format!("Could not save {}: {e}", path.display()), false),
        }
    }

    pub(super) fn load_strokes(&mut self, d: &Doc, path: &Path) {
        let j: Value = match std::fs::read(path).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string())) {
            Ok(j) => j,
            Err(e) => return self.say(format!("Could not read {}: {e}", path.display()), false),
        };
        let mut out = vec![];
        for s in j["strokes"].as_array().into_iter().flatten() {
            let mesh0 = s["mesh"].as_u64().unwrap_or(0) as usize;
            let pts: Vec<CombPt> = s["points"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|p| {
                    let mesh = p["mesh"].as_u64().map(|m| m as usize).unwrap_or(mesh0);
                    let tri = p["tri"].as_u64()? as usize;
                    let b = f3(&p["bary"])?;
                    (d.model.meshes.get(mesh)?.tris.len() > tri).then_some(CombPt { mesh, tri, bary: b })
                })
                .collect();
            if pts.len() >= 2 {
                out.push(CombStroke { pts });
            }
        }
        let n = out.len();
        self.hair.strokes = out;
        self.comb_changed();
        self.say(format!("Loaded {n} comb strokes from {}", path.display()), n > 0);
    }

    // ---------------------------------------------------------------------------------------------
    // Flow
    // ---------------------------------------------------------------------------------------------

    /// Works the flow out from the paint of the marked triangles (the mesh with most of them).
    pub(super) fn flow_from_paint(&mut self, d: &Doc, topo: &Topo) {
        let mut per = vec![0usize; d.model.meshes.len()];
        for (g, &w) in self.sel.iter().enumerate() {
            if w > 0.0 {
                per[topo.tri_loc[g].0] += 1;
            }
        }
        let Some((mesh, _)) = per.iter().enumerate().filter(|x| *x.1 > 0).max_by_key(|x| *x.1) else {
            self.say("Mark the hair first (Part tool, click the hair), then work the flow out".into(), false);
            return;
        };
        let m = &d.model.meshes[mesh];
        let mask: Vec<bool> = (0..m.tris.len()).map(|ti| self.sel[topo.gid(mesh, ti)] > 0.0).collect();
        let hs = match HairSet::build(&d.model, mesh, &Select::Mask(mask)) {
            Ok(h) => h,
            Err(e) => return self.say(format!("Flow: {e}"), false),
        };
        let Some(Some(tex)) = m.image.and_then(|i| self.textures(d).get(i)) else {
            return self.say("Flow: that mesh has no texture to read the paint from".into(), false);
        };
        let sheet = Sheet { w: tex.w, h: tex.h, rgba: &tex.rgba };
        let fo = FlowOpts { root: Root::auto(&hs), paint_weight: 0.3, smooth_iters: 100, part_angle: 70.0 };
        let fl = hair::flow(&d.model, &hs, &sheet, &fo);
        let items: Vec<(usize, V3, bool)> = (0..hs.tris.len()).filter(|&i| hs.is_primary(i)).map(|i| (hs.tris[i], fl.dir[i].map(|x| x as f32), fl.ambiguous[i])).collect();
        // Only an estimate: `gaw hair` adds root drape, part relock and comb strokes, so its flow.json is the real one.
        let src = format!("quick estimate from the paint of {} marked triangles; gaw hair's flow.json is what the hair tool uses", items.len());
        if let Some(i) = m.image {
            self.image = i;
        }
        let f = FlowField::build(&d.model, &self.mesh_pos, src, mesh, items, None);
        self.set_flow(f);
    }

    /// Reads a hair tool `flow.json` ({"gaw_flow":1, "mesh", "triangles":[{tri, dir, ambiguous}], "breaks":[{a, b, angle_deg}]}).
    pub(super) fn flow_load(&mut self, d: &Doc, path: &Path) {
        let j: Value = match std::fs::read(path).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string())) {
            Ok(j) => j,
            Err(e) => return self.say(format!("Could not read {}: {e}", path.display()), false),
        };
        let mesh = match &j["mesh"] {
            Value::String(s) => d.model.meshes.iter().position(|m| &m.name == s).unwrap_or(0),
            v => v.as_u64().unwrap_or(0) as usize,
        };
        if mesh >= d.model.meshes.len() {
            return self.say(format!("{}: mesh {mesh} is not in this model", path.display()), false);
        }
        let items: Vec<(usize, V3, bool)> = j["triangles"].as_array().into_iter().flatten().filter_map(|t| Some((t["tri"].as_u64()? as usize, f3(&t["dir"])?, t["ambiguous"].as_bool().unwrap_or(false)))).collect();
        if items.is_empty() {
            return self.say(format!("{} has no triangles with a flow direction", path.display()), false);
        }
        let breaks = j["breaks"].as_array().map(|a| a.iter().filter_map(|b| Some((b["a"].as_u64()? as usize, b["b"].as_u64()? as usize, b["angle_deg"].as_f64().unwrap_or(90.0) as f32))).collect());
        let combed = j["comb_strokes"].as_u64().unwrap_or(0);
        let src = format!("from {}{}", path.display(), if combed > 0 { format!(", {combed} comb strokes applied") } else { String::new() });
        let f = FlowField::build(&d.model, &self.mesh_pos, src, mesh, items, breaks);
        if let Some(i) = d.model.meshes[mesh].image {
            self.image = i;
        }
        self.set_flow(f);
    }

    fn set_flow(&mut self, f: FlowField) {
        let (n, b) = (f.tris.len(), f.breaks.len());
        let src = f.source.clone();
        self.hair.flow = Some(f);
        self.hair.flow_on = true;
        self.recheck_comb();
        let dis = self.hair.disagree.1.len();
        self.say(format!("Flow on {n} triangles ({src}): {b} breaks{}", if self.hair.strokes.is_empty() { String::new() } else { format!(", {dis} comb points run against it") }), true);
    }

    // ---------------------------------------------------------------------------------------------
    // Defects
    // ---------------------------------------------------------------------------------------------

    pub(super) fn defects_load(&mut self, d: &Doc, path: &Path) {
        let j: Value = match std::fs::read(path).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string())) {
            Ok(j) => j,
            Err(e) => return self.say(format!("Could not read {}: {e}", path.display()), false),
        };
        let list = j["defects"].as_array().cloned().unwrap_or_default();
        let mut out: Vec<Defect> = list
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let ok = |mi: usize, ti: usize| d.model.meshes.get(mi).is_some_and(|m| m.tris.len() > ti);
                // `gaw defects` writes marked.meshes[{mesh, triangles}]; the first draft wrote triangles[{mesh, tri}].
                let mut tris: Vec<(usize, usize)> = vec![];
                for m in x["marked"]["meshes"].as_array().into_iter().flatten() {
                    let mi = m["mesh"].as_u64().unwrap_or(0) as usize;
                    tris.extend(m["triangles"].as_array().into_iter().flatten().filter_map(|t| t.as_u64()).map(|t| (mi, t as usize)).filter(|&(a, b)| ok(a, b)));
                }
                for t in x["triangles"].as_array().into_iter().flatten() {
                    if let Some(ti) = t["tri"].as_u64() {
                        let mi = t["mesh"].as_u64().unwrap_or(0) as usize;
                        if ok(mi, ti as usize) {
                            tris.push((mi, ti as usize));
                        }
                    }
                }
                let texels = x["texel_boxes"]
                    .as_array()
                    .or(x["texels"].as_array())
                    .into_iter()
                    .flatten()
                    .filter_map(|t| {
                        let b = t["box"].as_array()?;
                        let v: Vec<f32> = b.iter().filter_map(|q| q.as_f64()).map(|q| q as f32).collect();
                        let image = t["image_index"].as_u64().or(t["image"].as_u64()).unwrap_or(0) as usize;
                        (v.len() == 4).then(|| (image, [v[0], v[1], v[2], v[3]]))
                    })
                    .collect();
                // Camera: `camera` (gaw defects, with "distance") or an object `view` (first draft).
                let cam_j = if x["camera"].is_object() { &x["camera"] } else { &x["view"] };
                let view = if cam_j.is_object() {
                    let mut vj = cam_j.clone();
                    vj["name"] = json!("defect");
                    if vj["dist"].is_null() {
                        vj["dist"] = vj["distance"].clone();
                    }
                    crate::viewscmd::view_from_json(&vj).map(|v| v.cam)
                } else {
                    None
                };
                // Pose of the view the checker drew it in, when it names one.
                let drawn = x["view"].as_str().and_then(|n| j["views"].as_array()?.iter().find(|v| v["name"].as_str() == Some(n)));
                let num = x["severity"].as_f64().or(x["score"].as_f64()).unwrap_or(0.5) as f32;
                let (grade, level) = match x["severity"].as_str() {
                    Some(w) => (format!("{w} {num:.2}"), match w { "high" => 2, "medium" => 1, _ => 0 }),
                    None => (format!("{num:.2}"), if num >= 0.66 { 2 } else if num >= 0.33 { 1 } else { 0 }),
                };
                Defect {
                    id: x["id"].as_str().map(str::to_owned).unwrap_or_else(|| format!("d{}", i + 1)),
                    kind: x["kind"].as_str().unwrap_or("defect").to_owned(),
                    severity: num,
                    grade,
                    level,
                    summary: x["summary"].as_str().or(x["note"].as_str()).unwrap_or("").to_owned(),
                    part: x["part"].as_str().or(j["part"].as_str().filter(|p| !p.is_empty())).map(str::to_owned),
                    at: f3(&x["at"]),
                    view,
                    clip: x["clip"].as_str().or(drawn.and_then(|v| v["clip"].as_str())).map(str::to_owned),
                    time: x["time"].as_f64().or(drawn.and_then(|v| v["time"].as_f64())).unwrap_or(0.0) as f32,
                    tris,
                    texels,
                    raw: x.clone(),
                }
            })
            .collect();
        out.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
        let n = out.len();
        self.hair.checked = vec![false; n];
        self.hair.defects = out;
        self.hair.current = None;
        self.hair.defects_file = Some(path.display().to_string());
        self.hair.show_cards = true;
        if let Some(src) = j["source"].as_str().or(j["candidate"]["file"].as_str()) {
            let same = Path::new(src).file_name() == d.path.file_name();
            if !same {
                self.say(format!("Loaded {n} defects, but they were found on {src}, not this file; triangles may not line up"), false);
                return;
            }
        }
        self.say(format!("Loaded {n} defects from {}", path.display()), true);
    }

    /// Jumps to a defect: camera, pose, its triangles marked, its texels outlined.
    pub(super) fn goto_defect(&mut self, i: usize, d: &Doc, topo: &Topo, cam: &mut Camera) {
        let Some(df) = self.hair.defects.get(i).cloned() else { return };
        // Mark its triangles (undoable).
        if !df.tris.is_empty() {
            let before = self.sel_state();
            self.sel.iter_mut().for_each(|w| *w = 0.0);
            for &(mi, ti) in &df.tris {
                self.sel[topo.gid(mi, ti)] = 1.0;
            }
            self.push_undo(Snap { what: format!("defect {}", df.id), sel: Some(before), gpos: None, tex: None });
            self.sel_ver += 1;
        }
        if df.clip.is_some() {
            self.clip = df.clip.clone();
            self.time = df.time;
        }
        match df.view {
            Some(v) => *cam = v,
            None => {
                // Aim at the defect from the side its triangles face.
                let rest = &self.mesh_pos;
                let mut c = [0.0f32; 3];
                let mut n = [0.0f32; 3];
                let mut k = 0.0;
                for &(mi, ti) in &df.tris {
                    let t = d.model.meshes[mi].tris[ti].map(|v| rest[mi][v as usize]);
                    c = add(c, scale(add(add(t[0], t[1]), t[2]), 1.0 / 3.0));
                    n = add(n, cross(sub(t[1], t[0]), sub(t[2], t[0])));
                    k += 1.0;
                }
                let target = if k > 0.0 { scale(c, 1.0 / k) } else { df.at.unwrap_or(cam.target) };
                cam.target = df.at.unwrap_or(target);
                if wb_core::geom::len(n) > 1e-9 {
                    let n = norm(n);
                    cam.yaw = n[0].atan2(n[2]);
                    cam.pitch = n[1].clamp(-0.95, 0.95).asin().clamp(-1.2, 1.2);
                }
                let size = topo.height * 0.18;
                cam.dist = size / (cam.fov_y * 0.5).tan();
                cam.ortho_half = size;
            }
        }
        // Texels: show that sheet, zoomed on the box.
        if let Some(&(img, b)) = df.texels.first() {
            if let Some(Some(t)) = d.textures.get(img) {
                self.image = img;
                let (u0, v0, u1, v1) = (b[0] / t.w as f32, b[1] / t.h as f32, b[2] / t.w as f32, b[3] / t.h as f32);
                let span = (u1 - u0).abs().max((v1 - v0).abs()).max(1e-3);
                self.tex_zoom = (0.35 / span).clamp(1.0, 40.0);
                self.tex_pan = Vec2::ZERO;
                let now = self.uv_to_screen([(u0 + u1) * 0.5, (v0 + v1) * 0.5]);
                let view_c = Rect::from_min_max(self.rect_t.min + Vec2::new(0.0, 40.0), self.rect_t.max).center();
                self.tex_pan = view_c - now;
            }
        }
        self.hair.current = Some(i);
        self.vb.key = None;
        self.say(format!("Defect {} ({}, severity {}): {}", df.id, df.kind, df.grade, df.summary), true);
    }

    fn picked_defects(&self) -> Vec<usize> {
        let c: Vec<usize> = self.hair.checked.iter().enumerate().filter(|x| *x.1).map(|x| x.0).collect();
        if c.is_empty() {
            self.hair.current.into_iter().collect()
        } else {
            c
        }
    }

    /// Turns the checked defects (or the one shown) into a Mark & fix request.
    pub(super) fn defects_request(&mut self, ctx: &egui::Context, d: &Arc<Doc>, topo: &Topo, cam: &Camera) {
        let pick = self.picked_defects();
        if pick.is_empty() {
            return self.say("Check one or more defects (or click one) first".into(), false);
        }
        let before = self.sel_state();
        self.sel.iter_mut().for_each(|w| *w = 0.0);
        let mut lines = vec![];
        let mut raw = vec![];
        let mut boxes: Vec<(usize, [f32; 4])> = vec![];
        for &i in &pick {
            let df = self.hair.defects[i].clone();
            for &(mi, ti) in &df.tris {
                self.sel[topo.gid(mi, ti)] = 1.0;
            }
            boxes.extend(df.texels.iter().copied());
            lines.push(format!("- {} {} (severity {}){}: {}", df.id, df.kind, df.grade, df.part.as_ref().map(|p| format!(" on {p}")).unwrap_or_default(), df.summary));
            raw.push(df.raw.clone());
        }
        // Texel boxes become texture marks too, so the mask PNG covers them.
        for (img, b) in boxes {
            let Some(Some(t)) = d.textures.get(img) else { continue };
            let (tw, th) = (t.w as f32, t.h as f32);
            if let Some((w, h, m)) = self.extra_mask(d, img) {
                let (w, h) = (*w, *h);
                let (x0, x1) = (((b[0] / tw) * w as f32).floor().max(0.0) as usize, ((b[2] / tw) * w as f32).ceil().min(w as f32) as usize);
                let (y0, y1) = (((b[1] / th) * h as f32).floor().max(0.0) as usize, ((b[3] / th) * h as f32).ceil().min(h as f32) as usize);
                for y in y0..y1 {
                    for x in x0..x1 {
                        m[y * w + x] = 255;
                    }
                }
            }
        }
        self.push_undo(Snap { what: "mark defects".into(), sel: Some(before), gpos: None, tex: None });
        self.sel_ver += 1;
        let head = if pick.len() == 1 { "Fix this defect:" } else { "Fix these defects:" };
        let body = format!("{head}\n{}", lines.join("\n"));
        self.note = if self.note.trim().is_empty() { body } else { format!("{}\n\n{body}", self.note.trim()) };
        self.hair.pending_defects = Some(json!({"workbench_defects": 1, "picked_from": self.hair.defects_file, "defects": raw}));
        self.save_request(ctx, d, self.parts.clone(), cam, false);
        self.hair.pending_defects = None;
    }

    /// Extra files for a request: strokes.json and defects.json.
    pub(super) fn hair_attachments(&self, d: &Doc) -> Vec<(String, Value)> {
        let mut v = vec![];
        if let Some(s) = self.strokes_json(d) {
            v.push(("strokes.json".to_owned(), s));
        }
        if let Some(df) = &self.hair.pending_defects {
            v.push(("defects.json".to_owned(), df.clone()));
        }
        v
    }

    // ---------------------------------------------------------------------------------------------
    // Cards
    // ---------------------------------------------------------------------------------------------

    pub(super) fn hair_card(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, d: &Arc<Doc>, cam: &mut Camera) {
        let Some(topo) = self.topo.clone() else { return };
        if std::mem::take(&mut self.hair.show_cards) {
            ui.scroll_to_cursor(Some(egui::Align::Min));
        }
        theme::card(ui, "Hair flow", |ui| {
            let n = self.hair.strokes.len();
            ui.label(RichText::new(if n == 0 { "No comb strokes. Pick Comb and drag along the hair, root to tip.".to_owned() } else { format!("{n} comb stroke{}", if n == 1 { "" } else { "s" }) }).size(12.5).color(theme::WEAK));
            ui.horizontal_wrapped(|ui| {
                if ui.add_enabled(n > 0, egui::Button::new("Remove last")).on_hover_text("Remove the last comb stroke (Ctrl+click a stroke with Comb removes that one)").clicked() {
                    self.hair.strokes.pop();
                    self.comb_changed();
                }
                if ui.add_enabled(n > 0, egui::Button::new("Clear")).on_hover_text("Remove every comb stroke").clicked() {
                    self.hair.strokes.clear();
                    self.comb_changed();
                }
                if ui.add_enabled(n > 0, egui::Button::new("Save strokes...")).on_hover_text("Write strokes.json for the hair tool. Save request also puts it in the request folder").clicked() {
                    let stem = d.path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
                    if let Some(p) = rfd::FileDialog::new().set_directory(Path::new(requestcmd::MODDING).join("exports")).set_file_name(format!("{stem}-strokes.json")).add_filter("Comb strokes", &["json"]).save_file() {
                        self.save_strokes(d, &p);
                    }
                }
                if ui.button("Load strokes...").on_hover_text("Open a strokes.json made earlier").clicked() {
                    if let Some(p) = rfd::FileDialog::new().set_directory(requestcmd::MODDING).add_filter("Comb strokes", &["json"]).pick_file() {
                        self.load_strokes(d, &p);
                    }
                }
            });
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                let has = self.hair.flow.is_some();
                if ui.add_enabled(has, egui::Button::selectable(self.hair.flow_on, "Flow arrows")).on_hover_text("Show which way the hair flows now, as short arrows on the model and the texture. Red lines are breaks: neighbors whose flow turns sharply").clicked() {
                    self.hair.flow_on = !self.hair.flow_on;
                }
                if ui.button("Load flow...").on_hover_text("Open the flow.json that gaw hair wrote in its output folder: the flow the hair tool actually used").clicked() {
                    if let Some(p) = rfd::FileDialog::new().set_directory(requestcmd::MODDING).add_filter("Hair flow", &["json"]).pick_file() {
                        self.flow_load(d, &p);
                    }
                }
                if ui.add_enabled(self.marked_count() > 0, egui::Button::new("Quick estimate")).on_hover_text("A rough flow from the painted strands of the marked triangles (mark the hair with Part first). gaw hair runs more steps (root drape, part relock, your comb strokes), so Load flow... shows the real one").clicked() {
                    self.flow_from_paint(d, &topo);
                }
            });
            if let Some(f) = &self.hair.flow {
                ui.label(RichText::new(format!("{} triangles, {} breaks ({})", theme::num(f.tris.len()), f.breaks.len(), f.source)).size(12.0).color(if f.breaks.is_empty() { theme::WEAK } else { theme::RED }));
                if n > 0 {
                    let (c, bad) = (&self.hair.disagree.0, self.hair.disagree.1.len());
                    ui.label(RichText::new(format!("Comb vs flow: {bad} of {c} stroke points run against it (orange)")).size(12.0).color(if bad == 0 { theme::WEAK } else { theme::GOLD }));
                }
            }
        });
        ui.add_space(8.0);
        theme::card(ui, "Defects", |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Load defects...").on_hover_text("Open the defects.json that gaw defects wrote").clicked() {
                    if let Some(p) = rfd::FileDialog::new().set_directory(requestcmd::MODDING).add_filter("Defects", &["json"]).pick_file() {
                        self.defects_load(d, &p);
                    }
                }
                let any = !self.picked_defects().is_empty();
                let checked = self.hair.checked.iter().filter(|&&c| c).count();
                let label = if checked > 1 { format!("Make request ({checked})") } else { "Make request".into() };
                if ui.add_enabled(any && self.busy.is_none(), egui::Button::new(label)).on_hover_text("Save a Mark & fix request for the checked defects (or the one shown): their triangles and texels marked, the list in the note").clicked() {
                    self.defects_request(ctx, d, &topo, cam);
                }
            });
            if self.hair.defects.is_empty() {
                ui.label(RichText::new("No defects loaded").size(12.5).color(theme::WEAK));
                return;
            }
            if let Some(f) = &self.hair.defects_file {
                // `gaw defects` always writes defects.json, so show its folder too.
                let pf = Path::new(f);
                let short: Vec<String> = [pf.parent().and_then(|x| x.file_name()), pf.file_name()].iter().flatten().map(|x| x.to_string_lossy().into_owned()).collect();
                ui.label(RichText::new(short.join("/")).monospace().size(11.0).color(theme::FAINT)).on_hover_text(f.as_str());
            }
            let mut go = None;
            for i in 0..self.hair.defects.len() {
                let df = &self.hair.defects[i];
                let col = match df.level { 2 => theme::RED, 1 => theme::GOLD, _ => theme::WEAK };
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.hair.checked[i], "").on_hover_text("Include in Make request");
                    // The checker's notes run long; keep the row short and put the whole note in the tip.
                    let mut short: String = df.summary.split([':', ';']).next().unwrap_or("").chars().take(48).collect();
                    if short.len() < df.summary.len() {
                        short.push_str("...");
                    }
                    let txt = RichText::new(format!("{}  {}: {}", df.grade, df.kind, short)).size(12.5).color(col);
                    let tip = format!("{}{}
{}
Click to go there and mark it", df.id, df.part.as_ref().map(|p| format!(", {p}")).unwrap_or_default(), df.summary);
                    if ui.add(egui::Button::selectable(self.hair.current == Some(i), txt).truncate()).on_hover_text(tip).clicked() {
                        go = Some(i);
                    }
                });
            }
            if let Some(i) = go {
                self.goto_defect(i, d, &topo, cam);
            }
        });
    }

    // ---------------------------------------------------------------------------------------------
    // Drawing
    // ---------------------------------------------------------------------------------------------

    fn arrow(painter: &Painter, c: Pos2, v: Vec2, col: Color32) {
        let l = v.length();
        if l < 1e-3 {
            return;
        }
        let u = v / l;
        let (a, b) = (c - u * 5.0, c + u * 5.0);
        let side = Vec2::new(-u.y, u.x);
        painter.line_segment([a, b], Stroke::new(1.4, col));
        painter.add(Shape::convex_polygon(vec![b + u * 2.0, b - u * 2.5 + side * 2.6, b - u * 2.5 - side * 2.6], col, Stroke::NONE));
    }

    pub(super) fn draw_hair(&self, painter: &Painter, d: &Doc, topo: &Topo, cam: &Camera) {
        if self.vb.w == 0 || self.pose.len() != d.model.meshes.len() {
            return;
        }
        let rig = Rig::new(cam, self.vb.w, self.vb.h);
        let scr = |p: V3| -> Option<Pos2> {
            let q = rig.project(p);
            (q[2] > 0.0 || rig.ortho).then(|| self.to_screen([q[0], q[1]]))
        };
        // Flow arrows on triangles you can see; breaks in red.
        if let (true, Some(f)) = (self.hair.flow_on, &self.hair.flow) {
            let mut seen = vec![false; topo.tris()];
            for &id in &self.vb.pick {
                if id >= 0 {
                    seen[id as usize] = true;
                }
            }
            let mut cells = std::collections::HashSet::new();
            for ft in &f.tris {
                if !seen[topo.gid(ft.mesh, ft.tri)] {
                    continue;
                }
                let c3 = d.model.meshes[ft.mesh].tris[ft.tri].map(|v| self.pose[ft.mesh][v as usize]);
                let cen = scale(add(add(c3[0], c3[1]), c3[2]), 1.0 / 3.0);
                let dir = add(scale(sub(c3[1], c3[0]), ft.coef[0]), scale(sub(c3[2], c3[0]), ft.coef[1]));
                let (Some(p), Some(q)) = (scr(cen), scr(add(cen, scale(norm(dir), topo.height * 0.01)))) else { continue };
                if !cells.insert(((p.x / 11.0) as i32, (p.y / 11.0) as i32)) {
                    continue;
                }
                Self::arrow(painter, p, q - p, if ft.ambiguous { Color32::from_rgb(170, 170, 180) } else { Color32::from_rgb(90, 230, 255) });
            }
            for &(a, _, _, e) in &f.breaks {
                let ft = &f.tris[a];
                let c3 = d.model.meshes[ft.mesh].tris[ft.tri].map(|v| self.pose[ft.mesh][v as usize]);
                if let (Some(p), Some(q)) = (scr(c3[e[0]]), scr(c3[e[1]])) {
                    painter.line_segment([p, q], Stroke::new(4.0, Color32::from_black_alpha(180)));
                    painter.line_segment([p, q], Stroke::new(2.4, theme::RED));
                }
            }
        }
        // Comb strokes (and the one being drawn).
        let draw_stroke = |pts: &[CombPt], si: Option<usize>| {
            let sp: Vec<Option<Pos2>> = pts.iter().map(|p| Self::comb_at(&d.model, &self.pose, p).and_then(scr)).collect();
            for k in 0..sp.len().saturating_sub(1) {
                if let (Some(a), Some(b)) = (sp[k], sp[k + 1]) {
                    let bad = si.is_some_and(|s| self.hair.disagree.1.contains(&(s, k)));
                    painter.line_segment([a, b], Stroke::new(5.0, Color32::from_black_alpha(170)));
                    painter.line_segment([a, b], Stroke::new(2.6, if bad { Color32::from_rgb(255, 150, 40) } else { Color32::from_rgb(140, 240, 120) }));
                }
            }
            if let Some(Some(s)) = sp.first() {
                painter.circle_filled(*s, 3.6, Color32::from_rgb(140, 240, 120));
            }
            let ends: Vec<Pos2> = sp.iter().flatten().copied().collect();
            if ends.len() >= 2 {
                let (a, b) = (ends[ends.len() - 2], ends[ends.len() - 1]);
                Self::arrow(painter, b, b - a, Color32::from_rgb(140, 240, 120));
            }
        };
        for (si, s) in self.hair.strokes.iter().enumerate() {
            draw_stroke(&s.pts, Some(si));
        }
        if let Some(Drag::Comb { pts, .. }) = &self.drag {
            draw_stroke(pts, None);
        }
        // The defect shown: a ring at its center.
        if let Some(df) = self.hair.current.and_then(|i| self.hair.defects.get(i)) {
            let mut c = [0.0f32; 3];
            let mut k = 0.0;
            for &(mi, ti) in &df.tris {
                let t = d.model.meshes[mi].tris[ti].map(|v| self.pose[mi][v as usize]);
                c = add(c, scale(add(add(t[0], t[1]), t[2]), 1.0 / 3.0));
                k += 1.0;
            }
            let at = if k > 0.0 { Some(scale(c, 1.0 / k)) } else { df.at };
            if let Some(p) = at.and_then(scr) {
                painter.circle_stroke(p, 14.0, Stroke::new(2.0, theme::GOLD));
            }
        }
    }

    pub(super) fn draw_hair_tex(&self, painter: &Painter, d: &Doc) {
        let image = self.image;
        let tr = self.tex_rect();
        let uvs = |v: [f32; 2]| self.uv_to_screen(v);
        if let (true, Some(f)) = (self.hair.flow_on, &self.hair.flow) {
            let mut cells = std::collections::HashSet::new();
            for ft in &f.tris {
                let m = &d.model.meshes[ft.mesh];
                if m.image != Some(image) {
                    continue;
                }
                let u = m.tris[ft.tri].map(|v| m.uv[v as usize]);
                let c = [(u[0][0] + u[1][0] + u[2][0]) / 3.0, (u[0][1] + u[1][1] + u[2][1]) / 3.0];
                let du = [(u[1][0] - u[0][0]) * ft.coef[0] + (u[2][0] - u[0][0]) * ft.coef[1], (u[1][1] - u[0][1]) * ft.coef[0] + (u[2][1] - u[0][1]) * ft.coef[1]];
                let p = uvs(c);
                if !tr.expand(10.0).contains(p) || !cells.insert(((p.x / 10.0) as i32, (p.y / 10.0) as i32)) {
                    continue;
                }
                let v = Vec2::new(du[0] * tr.width(), du[1] * tr.height());
                Self::arrow(painter, p, v, if ft.ambiguous { Color32::from_rgb(170, 170, 180) } else { Color32::from_rgb(90, 230, 255) });
            }
            for &(a, b, _, e) in &f.breaks {
                for (ti, edge) in [(a, Some(e)), (b, None)] {
                    let ft = &f.tris[ti];
                    let m = &d.model.meshes[ft.mesh];
                    if m.image != Some(image) {
                        continue;
                    }
                    let tri = m.tris[ft.tri];
                    let (i0, i1) = match edge {
                        Some(e) => (e[0], e[1]),
                        None => {
                            // The same two places in the neighbor.
                            let fa = &f.tris[a];
                            let ta = d.model.meshes[fa.mesh].tris[fa.tri];
                            let pa = [self.mesh_pos[fa.mesh][ta[e[0]] as usize], self.mesh_pos[fa.mesh][ta[e[1]] as usize]];
                            let f0 = (0..3).find(|&k| dist(self.mesh_pos[ft.mesh][tri[k] as usize], pa[0]) < 1e-5);
                            let f1 = (0..3).find(|&k| dist(self.mesh_pos[ft.mesh][tri[k] as usize], pa[1]) < 1e-5);
                            match (f0, f1) {
                                (Some(x), Some(y)) => (x, y),
                                _ => continue,
                            }
                        }
                    };
                    let (p, q) = (uvs(m.uv[tri[i0] as usize]), uvs(m.uv[tri[i1] as usize]));
                    painter.line_segment([p, q], Stroke::new(3.6, Color32::from_black_alpha(180)));
                    painter.line_segment([p, q], Stroke::new(2.0, theme::RED));
                }
            }
        }
        // Comb strokes in UV (segments that stay on one island).
        for s in &self.hair.strokes {
            let pts: Vec<Option<[f32; 2]>> = s
                .pts
                .iter()
                .map(|p| {
                    let m = &d.model.meshes[p.mesh];
                    (m.image == Some(image)).then(|| {
                        let u = m.tris[p.tri].map(|v| m.uv[v as usize]);
                        [u[0][0] * p.bary[0] + u[1][0] * p.bary[1] + u[2][0] * p.bary[2], u[0][1] * p.bary[0] + u[1][1] * p.bary[1] + u[2][1] * p.bary[2]]
                    })
                })
                .collect();
            for k in 0..pts.len().saturating_sub(1) {
                if let (Some(a), Some(b)) = (pts[k], pts[k + 1]) {
                    if (a[0] - b[0]).abs() + (a[1] - b[1]).abs() < 0.08 {
                        painter.line_segment([uvs(a), uvs(b)], Stroke::new(2.2, Color32::from_rgb(140, 240, 120)));
                    }
                }
            }
        }
        // Texel boxes of the defect shown and the checked ones.
        for (i, df) in self.hair.defects.iter().enumerate() {
            let on = self.hair.current == Some(i) || self.hair.checked.get(i).copied().unwrap_or(false);
            if !on {
                continue;
            }
            for &(img, b) in &df.texels {
                if img != image {
                    continue;
                }
                let Some(Some(t)) = d.textures.get(img) else { continue };
                let r = Rect::from_min_max(uvs([b[0] / t.w as f32, b[1] / t.h as f32]), uvs([b[2] / t.w as f32, b[3] / t.h as f32]));
                painter.rect_stroke(r.expand(2.0), CornerRadius::same(2), Stroke::new(2.0, if self.hair.current == Some(i) { theme::GOLD } else { theme::TEAL }), egui::StrokeKind::Outside);
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Script commands (tests on the hidden desktop)
    // ---------------------------------------------------------------------------------------------

    /// Returns false when `cmd` is not a hair command.
    pub(super) fn hair_script(&mut self, cmd: &str, rest: &str, ctx: &egui::Context, d: &Arc<Doc>, topo: &Topo, cam: &mut Camera) -> bool {
        let nums = |s: &str| -> Vec<usize> { s.split([',', ' ']).filter_map(|x| x.trim().parse::<usize>().ok()).collect() };
        match cmd {
            "comb-clear" => {
                self.hair.strokes.clear();
                self.comb_changed();
            }
            "comb-undo" => {
                self.hair.strokes.pop();
                self.comb_changed();
            }
            "save-strokes" => self.save_strokes(d, Path::new(rest)),
            "load-strokes" => self.load_strokes(d, Path::new(rest)),
            "flow" => match rest {
                "on" => self.hair.flow_on = true,
                "off" => self.hair.flow_on = false,
                "paint" | "compute" => self.flow_from_paint(d, topo),
                p => self.flow_load(d, Path::new(p)),
            },
            "defects" => self.defects_load(d, Path::new(rest)),
            "defect" => {
                if let Some(&n) = nums(rest).first() {
                    if n >= 1 {
                        self.goto_defect(n - 1, d, topo, cam);
                    }
                }
            }
            "defect-check" => {
                for n in nums(rest) {
                    if let Some(c) = self.hair.checked.get_mut(n.wrapping_sub(1)) {
                        *c = true;
                    }
                }
            }
            "defects-request" => self.defects_request(ctx, d, topo, cam),
            "hair-report" => {
                let pts: usize = self.hair.strokes.iter().map(|s| s.pts.len()).sum();
                println!("comb: {} strokes, {pts} points", self.hair.strokes.len());
                match &self.hair.flow {
                    Some(f) => {
                        println!("flow: {} triangles, {} breaks, arrows {} ({})", f.tris.len(), f.breaks.len(), if self.hair.flow_on { "on" } else { "off" }, f.source);
                        for &(a, b, deg, _) in f.breaks.iter().take(12) {
                            println!("  break: mesh {} triangles {} and {}, {deg:.0} degrees", f.tris[a].mesh, f.tris[a].tri, f.tris[b].tri);
                        }
                        println!("comb vs flow: {} of {} stroke points run against it", self.hair.disagree.1.len(), self.hair.disagree.0);
                    }
                    None => println!("flow: none"),
                }
                println!("defects: {} loaded, {} checked, shown {:?}", self.hair.defects.len(), self.hair.checked.iter().filter(|&&c| c).count(), self.hair.current.map(|i| self.hair.defects[i].id.clone()));
                for df in self.hair.defects.iter().take(12) {
                    println!("  {} {} {}: {} ({} triangles, {} texel boxes)", df.id, df.grade, df.kind, df.summary, df.tris.len(), df.texels.len());
                }
            }
            _ => return false,
        }
        true
    }
}
