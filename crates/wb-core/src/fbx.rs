//! Binary FBX 7.4 export of a skinned FF9 character GLB, for Memoria's custom model importer
//! (`ModelImporter.cs`, loaded from `<mod>/Models/<type>/<id>/<id>.fbx` by `ModelFactory.CreateModel`).
//!
//! The GLB comes from the ff9mapkit exporter (glTF is Unity mirrored on Y, scaled by `ff9_scale`
//! = 0.01, triangle winding reversed, V flipped). The importer applies no axis, winding, UV or scale
//! conversion, so this module undoes all four and writes Unity-space data:
//!
//! * positions `(x, -y, z) / s`, normals `(x, -y, z)`, bone translations `(x, -y, z) / s`,
//!   bone rotations `(-x, y, -z, w)` (written as XYZ Euler degrees, self-checked), UV `(u, 1 - v)`,
//!   triangles `(a, c, b)`;
//! * one `Geometry` + `Model` per mesh at identity, one `Skin` per geometry, one `Cluster` per (mesh, bone);
//! * the skeleton as `Root` / `LimbNode` models named `bone000`... in the GLB's hierarchy, the root under
//!   an identity `Armature` null (the importer dereferences `bone.Parent` for a root);
//! * one `Material` ("Phong", which the importer maps to its default shader) + `Texture`
//!   (`RelativeFilename` = `<image>.png`) per mesh, no vertex colors.
//!
//! The reader at the bottom follows the importer's rules (ids through `Convert.ToInt32`, string name
//! separator, connection lookup) so the read-back check fails where the game would.
use crate::model::Model;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

pub const DEFAULT_SCALE: f64 = 0.01;
const HEADER_MAGIC: &[u8] = b"Kaydara FBX Binary  \0\x1a\0";
const FOOT_ID: [u8; 16] = [0xfa, 0xbc, 0xab, 0x09, 0xd0, 0xc8, 0xd4, 0x66, 0xb1, 0x76, 0xfb, 0x83, 0x1c, 0xf7, 0x26, 0x7e];
const FOOT_MAGIC: [u8; 16] = [0xf8, 0x5a, 0x8c, 0x6a, 0xde, 0xf5, 0xd9, 0x7e, 0xec, 0xe9, 0x0c, 0xe3, 0x75, 0x8f, 0x29, 0x0b];
/// Separator between an object's name and its class inside a binary name (`bone001\0\1Model`).
const SEP: &str = "\u{0}\u{1}";

// ------------------------------------------------------------------------------------------------ node tree

#[derive(Clone, Debug, PartialEq)]
pub enum Prop {
    Int(i32),
    Long(i64),
    Double(f64),
    Str(String),
    Ints(Vec<i32>),
    Doubles(Vec<f64>),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Node {
    pub name: String,
    pub props: Vec<Prop>,
    pub kids: Vec<Node>,
}

impl Node {
    pub fn new(name: &str) -> Node {
        Node { name: name.to_owned(), props: vec![], kids: vec![] }
    }
    fn p(mut self, p: Prop) -> Node {
        self.props.push(p);
        self
    }
    fn s(self, v: &str) -> Node {
        self.p(Prop::Str(v.to_owned()))
    }
    fn k(mut self, n: Node) -> Node {
        self.kids.push(n);
        self
    }
    pub fn child(&self, name: &str) -> Option<&Node> {
        self.kids.iter().find(|k| k.name == name)
    }
    pub fn children<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Node> {
        self.kids.iter().filter(move |k| k.name == name)
    }
    fn text(&self, i: usize) -> Option<&str> {
        match self.props.get(i) {
            Some(Prop::Str(s)) => Some(s),
            _ => None,
        }
    }
    fn id(&self) -> Option<i64> {
        match self.props.first() {
            Some(Prop::Long(v)) => Some(*v),
            Some(Prop::Int(v)) => Some(*v as i64),
            _ => None,
        }
    }
    fn doubles(&self) -> Vec<f64> {
        for p in &self.props {
            match p {
                Prop::Doubles(v) => return v.clone(),
                Prop::Ints(v) => return v.iter().map(|&x| x as f64).collect(),
                _ => {}
            }
        }
        vec![]
    }
    fn ints(&self) -> Vec<i32> {
        for p in &self.props {
            if let Prop::Ints(v) = p {
                return v.clone();
            }
        }
        vec![]
    }
}

fn put_prop(out: &mut Vec<u8>, p: &Prop) {
    match p {
        Prop::Int(v) => {
            out.push(b'I');
            out.extend_from_slice(&v.to_le_bytes());
        }
        Prop::Long(v) => {
            out.push(b'L');
            out.extend_from_slice(&v.to_le_bytes());
        }
        Prop::Double(v) => {
            out.push(b'D');
            out.extend_from_slice(&v.to_le_bytes());
        }
        Prop::Str(s) => {
            out.push(b'S');
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        Prop::Ints(v) => {
            out.push(b'i');
            out.extend_from_slice(&(v.len() as u32).to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&((v.len() * 4) as u32).to_le_bytes());
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        Prop::Doubles(v) => {
            out.push(b'd');
            out.extend_from_slice(&(v.len() as u32).to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&((v.len() * 8) as u32).to_le_bytes());
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
}

fn put_node(out: &mut Vec<u8>, n: &Node) {
    let start = out.len();
    out.extend_from_slice(&[0u8; 12]); // end offset, property count, property bytes: patched below
    out.push(n.name.len() as u8);
    out.extend_from_slice(n.name.as_bytes());
    let pstart = out.len();
    for p in &n.props {
        put_prop(out, p);
    }
    let plen = out.len() - pstart;
    for k in &n.kids {
        put_node(out, k);
    }
    if !n.kids.is_empty() {
        out.extend_from_slice(&[0u8; 13]); // null record closes the child list
    }
    let end = out.len() as u32;
    out[start..start + 4].copy_from_slice(&end.to_le_bytes());
    out[start + 4..start + 8].copy_from_slice(&(n.props.len() as u32).to_le_bytes());
    out[start + 8..start + 12].copy_from_slice(&(plen as u32).to_le_bytes());
}

/// Serializes top-level nodes as a binary FBX 7.4 file (32-bit offsets, raw arrays, standard footer).
pub fn write_binary(nodes: &[Node]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(HEADER_MAGIC);
    out.extend_from_slice(&7400u32.to_le_bytes());
    for n in nodes {
        put_node(&mut out, n);
    }
    out.extend_from_slice(&[0u8; 13]);
    out.extend_from_slice(&FOOT_ID);
    out.extend_from_slice(&[0u8; 4]);
    let pad = match (16 - out.len() % 16) % 16 {
        0 => 16,
        p => p,
    };
    out.extend(std::iter::repeat(0u8).take(pad));
    out.extend_from_slice(&7400u32.to_le_bytes());
    out.extend_from_slice(&[0u8; 120]);
    out.extend_from_slice(&FOOT_MAGIC);
    out
}

// ------------------------------------------------------------------------------------------------ math

type M4 = [[f64; 4]; 4];

fn ident() -> M4 {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 }))
}

fn mmul(a: &M4, b: &M4) -> M4 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..4).map(|k| a[i][k] * b[k][j]).sum()))
}

fn minv(m: &M4) -> M4 {
    let mut a = [[0f64; 8]; 4];
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = m[i][j];
        }
        a[i][4 + i] = 1.0;
    }
    for c in 0..4 {
        let piv = (c..4).max_by(|&x, &y| a[x][c].abs().total_cmp(&a[y][c].abs())).unwrap();
        a.swap(c, piv);
        let d = if a[c][c].abs() < 1e-300 { 1e-300 } else { a[c][c] };
        for k in 0..8 {
            a[c][k] /= d;
        }
        for r in 0..4 {
            if r != c {
                let f = a[r][c];
                for k in 0..8 {
                    a[r][k] -= f * a[c][k];
                }
            }
        }
    }
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][4 + j]))
}

fn quat_matrix(q: [f64; 4]) -> [[f64; 3]; 3] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let n = if n == 0.0 { 1.0 } else { n };
    let [x, y, z, w] = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

fn trs(t: [f64; 3], q: [f64; 4], s: [f64; 3]) -> M4 {
    let r = quat_matrix(q);
    let mut m = ident();
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = r[i][j] * s[j];
        }
        m[i][3] = t[i];
    }
    m
}

fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let ([ax, ay, az, aw], [bx, by, bz, bw]) = (a, b);
    [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]
}

/// Memoria's `SetupFromEulerAngles(EULER_XYZ)`: q = qz * qy * qx, angles in degrees.
pub fn quat_from_euler_xyz(e: [f64; 3]) -> [f64; 4] {
    let h = std::f64::consts::PI / 360.0;
    let qx = [(e[0] * h).sin(), 0.0, 0.0, (e[0] * h).cos()];
    let qy = [0.0, (e[1] * h).sin(), 0.0, (e[1] * h).cos()];
    let qz = [0.0, 0.0, (e[2] * h).sin(), (e[2] * h).cos()];
    qmul(qmul(qz, qy), qx)
}

/// Inverse of `quat_from_euler_xyz` (R = Rz * Ry * Rx), degrees.
pub fn euler_xyz_from_quat(q: [f64; 4]) -> [f64; 3] {
    let m = quat_matrix(q);
    let sy = (-m[2][0]).clamp(-1.0, 1.0);
    let y = sy.asin();
    let (x, z) = if sy.abs() < 0.9999995 { (m[2][1].atan2(m[2][2]), m[1][0].atan2(m[0][0])) } else { ((-m[1][2]).atan2(m[1][1]), 0.0) };
    [x.to_degrees(), y.to_degrees(), z.to_degrees()]
}

fn same_rotation(a: [f64; 4], b: [f64; 4]) -> f64 {
    let d: f64 = (0..4).map(|i| a[i] * b[i]).sum::<f64>();
    let na: f64 = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    1.0 - (d / (na * nb)).abs()
}

fn flat_col_major(m: &M4) -> Vec<f64> {
    (0..4).flat_map(|c| (0..4).map(move |r| m[r][c])).collect()
}

fn from_col_major(v: &[f64]) -> M4 {
    std::array::from_fn(|r| std::array::from_fn(|c| v[c * 4 + r]))
}

// ------------------------------------------------------------------------------------------------ export

pub struct ExportOptions {
    /// Name of the file (`<geo_id>.fbx`); defaults to the GLB's `ff9_geo_id`.
    pub geo_id: Option<u32>,
    /// Override of the GLB's `ff9_scale` (glTF units per FF9 unit).
    pub scale: Option<f64>,
    /// Keep `battle_model*` overlay meshes (battle-only daggers); a field model has none.
    pub keep_overlays: bool,
    /// Write the FBX even when the GLB's inverse bind matrices disagree with its rest pose.
    pub force: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        ExportOptions { geo_id: None, scale: None, keep_overlays: false, force: false }
    }
}

#[derive(Debug)]
pub struct ExportedMesh {
    pub name: String,
    pub vertices: usize,
    pub triangles: usize,
    pub texture: Option<String>,
    pub max_influences: usize,
}

#[derive(Debug)]
pub struct Export {
    pub fbx: Vec<u8>,
    /// `(file name, PNG bytes)` for every texture the FBX references.
    pub textures: Vec<(String, Vec<u8>)>,
    pub geo_id: Option<u32>,
    pub geo: Option<String>,
    /// Folder type under `Models/` (1 acc, 2 main, 3 mon, 4 npc, 5 sub) from the `ff9_geo` name.
    pub type_int: Option<u8>,
    pub scale: f64,
    pub bones: usize,
    pub meshes: Vec<ExportedMesh>,
    pub skipped_meshes: Vec<String>,
    /// Largest quaternion disagreement after the quaternion -> Euler -> quaternion round trip (1 - |dot|).
    pub euler_error: f64,
    /// Largest entry of `world(rest) * inverseBind - I` over the GLB's joints (glTF space).
    pub bind_error: f64,
    pub notes: Vec<String>,
}

struct Bone {
    num: usize,
    parent: Option<usize>, // index into the bone list
    t: [f64; 3],
    q: [f64; 4],
    s: [f64; 3],
}

fn arr<const N: usize>(v: Option<&Value>, d: [f64; N]) -> [f64; N] {
    match v.and_then(Value::as_array) {
        Some(a) if a.len() == N => std::array::from_fn(|i| a[i].as_f64().unwrap_or(d[i])),
        _ => d,
    }
}

/// `bone012_R_hand`, `bone012`, `bone012.001` -> 12.
pub fn bone_number(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("bone")?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.len() < 3 {
        return None;
    }
    digits.parse().ok()
}

fn geo_type(geo: &str) -> Option<u8> {
    ["GEO_ACC_", "GEO_MAIN_", "GEO_MON_", "GEO_NPC_", "GEO_SUB_"].iter().position(|p| geo.starts_with(p)).map(|i| i as u8 + 1)
}

fn model_node(id: i64, name: &str, class: &str, t: [f64; 3], e: [f64; 3], s: [f64; 3]) -> Node {
    let lcl = |n: &str, v: [f64; 3]| Node::new("P").s(n).s(n).s("").s("A").p(Prop::Double(v[0])).p(Prop::Double(v[1])).p(Prop::Double(v[2]));
    Node::new("Model")
        .p(Prop::Long(id))
        .s(&format!("{name}{SEP}Model"))
        .s(class)
        .k(Node::new("Version").p(Prop::Int(232)))
        .k(Node::new("Properties70").k(lcl("Lcl Translation", t)).k(lcl("Lcl Rotation", e)).k(lcl("Lcl Scaling", s)))
}

fn conn(kind: &str, child: i64, parent: i64, prop: Option<&str>) -> Node {
    let n = Node::new("C").s(kind).p(Prop::Long(child)).p(Prop::Long(parent));
    match prop {
        Some(p) => n.s(p),
        None => n,
    }
}

fn vertex_normals(pos: &[[f32; 3]], tris: &[[u32; 3]]) -> Vec<[f64; 3]> {
    let mut acc = vec![[0f64; 3]; pos.len()];
    for t in tris {
        let p: Vec<[f64; 3]> = t.iter().map(|&i| pos[i as usize].map(|v| v as f64)).collect();
        let (a, b) = ([p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]], [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]]);
        let n = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        for &i in t {
            for k in 0..3 {
                acc[i as usize][k] += n[k];
            }
        }
    }
    acc.into_iter()
        .map(|n| {
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if l > 0.0 {
                [n[0] / l, n[1] / l, n[2] / l]
            } else {
                [0.0, 1.0, 0.0]
            }
        })
        .collect()
}

/// glTF normals per model mesh (same order as `Model::meshes`); None where the primitive has none.
fn gltf_normals(model: &Model) -> Vec<Option<Vec<[f64; 3]>>> {
    let glb = &model.glb;
    let mut out = Vec::new();
    for m in glb.list("meshes") {
        for p in m.get("primitives").and_then(Value::as_array).into_iter().flatten() {
            let n = p.get("attributes").and_then(|a| a.get("NORMAL")).and_then(Value::as_u64).and_then(|a| glb.accessor(a as usize).ok());
            out.push(n.map(|a| (0..a.count).map(|i| [a.get(i, 0), a.get(i, 1), a.get(i, 2)]).collect()));
        }
    }
    out
}

/// Order meshes are written in. Memoria wires a texture animation (the eye blink) to a material by its
/// index in `GetComponentsInChildren<SkinnedMeshRenderer>()` order, and stock Zidane is `battle_model0`,
/// `battle_model1`, `mesh0`, `mesh1` with the blink on index 3. So overlay meshes go first (battle_model,
/// then field_model); every other mesh keeps its GLB order.
pub fn export_order(names: &[&str]) -> Vec<usize> {
    let rank = |n: &str| if n.starts_with("battle_model") { 0 } else if n.starts_with("field_model") { 1 } else { 2 };
    let mut order: Vec<usize> = (0..names.len()).collect();
    order.sort_by_key(|&i| (rank(names[i]), if rank(names[i]) < 2 { names[i] } else { "" }, i));
    order
}

pub fn export_fbx(model: &Model, opts: &ExportOptions) -> Result<Export, String> {
    let glb = &model.glb;
    let extras = glb.json.get("asset").and_then(|a| a.get("extras"));
    let mut notes = Vec::new();
    let scale = match (opts.scale, extras.and_then(|e| e.get("ff9_scale")).and_then(Value::as_f64)) {
        (Some(s), _) => s,
        (None, Some(s)) => s,
        (None, None) => {
            notes.push(format!("no asset.extras.ff9_scale: assumed {DEFAULT_SCALE} (pass --scale to change)"));
            DEFAULT_SCALE
        }
    };
    if !(scale > 0.0) {
        return Err(format!("scale must be positive, got {scale}"));
    }
    let geo = extras.and_then(|e| e.get("ff9_geo")).and_then(Value::as_str).map(str::to_owned);
    let geo_id = opts.geo_id.or_else(|| extras.and_then(|e| e.get("ff9_geo_id")).and_then(Value::as_u64).map(|v| v as u32));

    // ---- skeleton from the first skin
    let skin = glb.list("skins").first().ok_or("the GLB has no skin: gaw fbx writes skinned character models")?;
    let joints: Vec<usize> = skin.get("joints").and_then(Value::as_array).ok_or("skin without joints")?.iter().filter_map(Value::as_u64).map(|v| v as usize).collect();
    let nodes = glb.list("nodes");
    let mut parent_of: HashMap<usize, usize> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        for c in n.get("children").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_u64) {
            parent_of.insert(c as usize, i);
        }
    }
    let joint_pos: HashMap<usize, usize> = joints.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let mut bones: Vec<Bone> = Vec::new(); // in skin joint order
    for &ni in &joints {
        let n = nodes.get(ni).ok_or("joint node missing")?;
        if n.get("matrix").is_some() {
            return Err("a joint node uses `matrix`; only translation/rotation/scale joints are supported".into());
        }
        let name = n.get("name").and_then(Value::as_str).unwrap_or("");
        let num = bone_number(name).or_else(|| n.get("extras").and_then(|e| e.get("ff9_bone_num")).and_then(Value::as_u64).map(|v| v as usize)).ok_or_else(|| format!("joint `{name}` is not named boneNNN"))?;
        let parent = parent_of.get(&ni).and_then(|p| joint_pos.get(p)).copied();
        let q = arr(n.get("rotation"), [0.0, 0.0, 0.0, 1.0]);
        bones.push(Bone { num, parent, t: arr(n.get("translation"), [0.0; 3]), q, s: arr(n.get("scale"), [1.0; 3]) });
    }
    let mut seen: BTreeMap<usize, usize> = BTreeMap::new();
    for (i, b) in bones.iter().enumerate() {
        if seen.insert(b.num, i).is_some() {
            return Err(format!("two joints are numbered bone{:03}", b.num));
        }
    }
    if seen.keys().copied().ne(0..bones.len()) {
        return Err(format!("bone numbers must run 0..{} without gaps (the importer indexes bones by number), got {:?}", bones.len() - 1, seen.keys().collect::<Vec<_>>()));
    }
    let roots: Vec<usize> = (0..bones.len()).filter(|&i| bones[i].parent.is_none()).collect();
    if roots.len() != 1 || bones[roots[0]].num != 0 {
        return Err("the skeleton must have exactly one root joint and it must be bone000".into());
    }

    // ---- rest pose in glTF space: world(rest) * inverseBind = I
    let mut world_g: Vec<Option<M4>> = vec![None; bones.len()];
    fn world(i: usize, bones: &[Bone], cache: &mut Vec<Option<M4>>) -> M4 {
        if let Some(m) = cache[i] {
            return m;
        }
        let local = trs(bones[i].t, bones[i].q, bones[i].s);
        let m = match bones[i].parent {
            Some(p) => mmul(&world(p, bones, cache), &local),
            None => local,
        };
        cache[i] = Some(m);
        m
    }
    for i in 0..bones.len() {
        world(i, &bones, &mut world_g);
    }
    let world_g: Vec<M4> = world_g.into_iter().map(|m| m.unwrap()).collect();
    let mut bind_error = 0f64;
    match skin.get("inverseBindMatrices").and_then(Value::as_u64) {
        Some(a) => {
            let acc = glb.accessor(a as usize)?;
            if acc.count != bones.len() {
                return Err(format!("inverseBindMatrices has {} entries for {} joints", acc.count, bones.len()));
            }
            for (i, w) in world_g.iter().enumerate() {
                let ibm = from_col_major(&(0..16).map(|k| acc.get(i, k)).collect::<Vec<_>>());
                let p = mmul(w, &ibm);
                for r in 0..4 {
                    for c in 0..4 {
                        bind_error = bind_error.max((p[r][c] - if r == c { 1.0 } else { 0.0 }).abs());
                    }
                }
            }
        }
        None => notes.push("the skin has no inverseBindMatrices: bind pose taken from the rest pose".into()),
    }
    if bind_error > 1e-3 && !opts.force {
        return Err(format!("the GLB was bound away from its rest pose (world * inverseBind differs from identity by {bind_error:.4}); Memoria binds at the rest pose, so the mesh would deform. Re-export from the rest pose, or pass --force"));
    }

    // ---- Unity-space skeleton (mirror Y, undo scale), Euler degrees with a self-check
    let mut euler_error = 0f64;
    let mut unity: Vec<([f64; 3], [f64; 3], [f64; 3], [f64; 4])> = Vec::new(); // t, euler, s, q
    for b in &bones {
        let t = [b.t[0] / scale, -b.t[1] / scale, b.t[2] / scale];
        let q = [-b.q[0], b.q[1], -b.q[2], b.q[3]];
        let e = euler_xyz_from_quat(q);
        euler_error = euler_error.max(same_rotation(quat_from_euler_xyz(e), q));
        unity.push((t, e, b.s, q));
    }
    if euler_error > 1e-6 {
        return Err(format!("quaternion -> Euler round trip is off by {euler_error:.2e}"));
    }
    let mut world_u: Vec<Option<M4>> = vec![None; bones.len()];
    fn world_unity(i: usize, bones: &[Bone], u: &[([f64; 3], [f64; 3], [f64; 3], [f64; 4])], cache: &mut Vec<Option<M4>>) -> M4 {
        if let Some(m) = cache[i] {
            return m;
        }
        let local = trs(u[i].0, u[i].3, u[i].2);
        let m = match bones[i].parent {
            Some(p) => mmul(&world_unity(p, bones, u, cache), &local),
            None => local,
        };
        cache[i] = Some(m);
        m
    }
    for i in 0..bones.len() {
        world_unity(i, &bones, &unity, &mut world_u);
    }
    let world_u: Vec<M4> = world_u.into_iter().map(|m| m.unwrap()).collect();

    // ---- joint index (per GLB joint slot) -> bone number
    let bone_of_joint: Vec<usize> = bones.iter().map(|b| b.num).collect();

    // ---- objects
    let mut next_id = 1_000_000i64;
    let mut id = || {
        next_id += 1;
        next_id
    };
    let mut objects = Node::new("Objects");
    let mut conns = Node::new("Connections");
    let one = [1.0, 1.0, 1.0];
    let armature = id();
    objects.kids.push(model_node(armature, "Armature", "Null", [0.0; 3], [0.0; 3], one));
    let mut bone_id = vec![0i64; bones.len()];
    // Objects in bone-number order, so the file lists boneNNN in the original FF9 order.
    for (&num, &i) in seen.iter() {
        bone_id[i] = id();
        let class = if num == 0 { "Root" } else { "LimbNode" };
        let (t, e, s, _) = unity[i];
        objects.kids.push(model_node(bone_id[i], &format!("bone{num:03}"), class, t, e, s));
    }
    for (i, b) in bones.iter().enumerate() {
        let parent = b.parent.map(|p| bone_id[p]).unwrap_or(armature);
        conns.kids.push(conn("OO", bone_id[i], parent, None));
    }

    let normals_g = gltf_normals(model);
    let mut textures: Vec<(String, Vec<u8>)> = Vec::new();
    let mut tex_node: HashMap<usize, i64> = HashMap::new(); // image index -> texture object id
    let mut exported = Vec::new();
    let mut skipped = Vec::new();
    for mi in export_order(&model.meshes.iter().map(|m| m.name.as_str()).collect::<Vec<_>>()) {
        let m = &model.meshes[mi];
        if !opts.keep_overlays && m.name.starts_with("battle_model") {
            skipped.push(m.name.clone());
            continue;
        }
        if m.pos.is_empty() || m.tris.is_empty() {
            skipped.push(m.name.clone());
            continue;
        }
        let name = m.name.replace(['#', ':'], "_");
        let (geo_oid, model_oid) = (id(), id());
        let verts: Vec<f64> = m.pos.iter().flat_map(|p| [p[0] as f64 / scale, -(p[1] as f64) / scale, p[2] as f64 / scale]).collect();
        let mut poly: Vec<i32> = Vec::with_capacity(m.tris.len() * 3);
        for t in &m.tris {
            poly.extend_from_slice(&[t[0] as i32, t[2] as i32, !(t[1] as i32)]); // winding reversed, last index negated
        }
        let nrm: Vec<f64> = match normals_g.get(mi).and_then(|n| n.as_ref()) {
            Some(n) => n.iter().flat_map(|v| [v[0], -v[1], v[2]]).collect(),
            None => vertex_normals(&m.pos, &m.tris).into_iter().flat_map(|v| [v[0], -v[1], v[2]]).collect(),
        };
        let uv: Vec<f64> = m.uv.iter().flat_map(|u| [u[0] as f64, 1.0 - u[1] as f64]).collect();
        let element = |kind: &str, label: &str, data_name: &str, data: Vec<f64>| {
            Node::new(kind)
                .p(Prop::Int(0))
                .k(Node::new("Version").p(Prop::Int(101)))
                .k(Node::new("Name").s(label))
                .k(Node::new("MappingInformationType").s("ByVertex"))
                .k(Node::new("ReferenceInformationType").s("Direct"))
                .k(Node::new(data_name).p(Prop::Doubles(data)))
        };
        let layer_entry = |ty: &str| Node::new("LayerElement").k(Node::new("Type").s(ty)).k(Node::new("TypedIndex").p(Prop::Int(0)));
        let geometry = Node::new("Geometry")
            .p(Prop::Long(geo_oid))
            .s(&format!("{name}{SEP}Geometry"))
            .s("Mesh")
            .k(Node::new("Vertices").p(Prop::Doubles(verts)))
            .k(Node::new("PolygonVertexIndex").p(Prop::Ints(poly)))
            .k(Node::new("GeometryVersion").p(Prop::Int(124)))
            .k(element("LayerElementNormal", "Normals", "Normals", nrm))
            .k(element("LayerElementUV", "UVMap", "UV", uv))
            .k(Node::new("Layer").p(Prop::Int(0)).k(Node::new("Version").p(Prop::Int(100))).k(layer_entry("LayerElementNormal")).k(layer_entry("LayerElementUV")));
        objects.kids.push(geometry);
        objects.kids.push(model_node(model_oid, &name, "Mesh", [0.0; 3], [0.0; 3], one));
        conns.kids.push(conn("OO", geo_oid, model_oid, None));

        // material + texture (one per mesh; the importer reads one)
        let mut tex_name = None;
        if let Some(ii) = m.image {
            let img = &model.images[ii];
            let stem = img.name.trim_end_matches(".png").to_owned();
            let file = format!("{stem}.png");
            let mat = id();
            objects.kids.push(
                Node::new("Material")
                    .p(Prop::Long(mat))
                    .s(&format!("{name}_mat{SEP}Material"))
                    .s("")
                    .k(Node::new("Version").p(Prop::Int(102)))
                    .k(Node::new("ShadingModel").s("Phong"))
                    .k(Node::new("Properties70")),
            );
            conns.kids.push(conn("OO", mat, model_oid, None));
            let tex = *tex_node.entry(ii).or_insert_with(|| {
                let tid = id();
                objects.kids.push(
                    Node::new("Texture")
                        .p(Prop::Long(tid))
                        .s(&format!("{stem}{SEP}Texture"))
                        .s("")
                        .k(Node::new("Type").s("TextureVideoClip"))
                        .k(Node::new("Version").p(Prop::Int(202)))
                        .k(Node::new("TextureName").s(&format!("Texture::{stem}")))
                        .k(Node::new("Properties70"))
                        .k(Node::new("FileName").s(&file))
                        .k(Node::new("RelativeFilename").s(&file)),
                );
                tid
            });
            conns.kids.push(conn("OP", tex, mat, Some("DiffuseColor")));
            if !textures.iter().any(|(n, _)| *n == file) {
                let png = if img.mime == "image/png" || img.bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
                    img.bytes.clone()
                } else {
                    let rgba = image::load_from_memory(&img.bytes).map_err(|e| format!("texture {}: {e}", img.name))?.to_rgba8();
                    let mut buf = Vec::new();
                    image::DynamicImage::ImageRgba8(rgba).write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png).map_err(|e| e.to_string())?;
                    buf
                };
                textures.push((file.clone(), png));
            }
            tex_name = Some(file);
        }

        // skin: up to four weights per vertex, normalized, grouped by bone number
        let mut by_bone: BTreeMap<usize, (Vec<i32>, Vec<f64>)> = BTreeMap::new();
        let mut max_infl = 0;
        for vi in 0..m.pos.len() {
            let mut infl: Vec<(usize, f64)> = Vec::new();
            for c in 0..4 {
                let (j, w) = (m.joints[vi][c] as usize, m.weights[vi][c] as f64);
                if w <= 1e-6 {
                    continue;
                }
                let num = *bone_of_joint.get(j).ok_or_else(|| format!("{}: vertex {vi} uses joint {j}, the skin has {}", m.name, bone_of_joint.len()))?;
                match infl.iter_mut().find(|e| e.0 == num) {
                    Some(e) => e.1 += w,
                    None => infl.push((num, w)),
                }
            }
            if infl.is_empty() {
                infl.push((0, 1.0));
            }
            let total: f64 = infl.iter().map(|e| e.1).sum();
            max_infl = max_infl.max(infl.len());
            for (num, w) in infl {
                let e = by_bone.entry(num).or_default();
                e.0.push(vi as i32);
                e.1.push(w / total);
            }
        }
        let skin_id = id();
        objects.kids.push(Node::new("Deformer").p(Prop::Long(skin_id)).s(&format!("{name}_skin{SEP}Deformer")).s("Skin").k(Node::new("Version").p(Prop::Int(100))).k(Node::new("Link_DeformAcc").p(Prop::Double(1.0))));
        conns.kids.push(conn("OO", skin_id, geo_oid, None));
        for (num, (idx, w)) in by_bone {
            let bi = seen[&num];
            let cl = id();
            objects.kids.push(
                Node::new("Deformer")
                    .p(Prop::Long(cl))
                    .s(&format!("bone{num:03}{SEP}SubDeformer"))
                    .s("Cluster")
                    .k(Node::new("Version").p(Prop::Int(100)))
                    .k(Node::new("UserData").s("").s(""))
                    .k(Node::new("Indexes").p(Prop::Ints(idx)))
                    .k(Node::new("Weights").p(Prop::Doubles(w)))
                    .k(Node::new("Transform").p(Prop::Doubles(flat_col_major(&minv(&world_u[bi])))))
                    .k(Node::new("TransformLink").p(Prop::Doubles(flat_col_major(&world_u[bi])))),
            );
            conns.kids.push(conn("OO", cl, skin_id, None));
            // The importer resolves a cluster's bone with the bone as child and the cluster as parent.
            conns.kids.push(conn("OO", bone_id[bi], cl, None));
        }
        exported.push(ExportedMesh { name: m.name.clone(), vertices: m.pos.len(), triangles: m.tris.len(), texture: tex_name, max_influences: max_infl });
    }
    if exported.is_empty() {
        return Err("no mesh left to export".into());
    }

    let header = Node::new("FBXHeaderExtension")
        .k(Node::new("FBXHeaderVersion").p(Prop::Int(1003)))
        .k(Node::new("FBXVersion").p(Prop::Int(7400)))
        .k(Node::new("Creator").s("gaw fbx (Game Asset Workbench)"));
    let fbx = write_binary(&[header, objects, conns]);
    Ok(Export { fbx, textures, geo_id, geo: geo.clone(), type_int: geo.as_deref().and_then(geo_type), scale, bones: bones.len(), meshes: exported, skipped_meshes: skipped, euler_error, bind_error, notes })
}

// ------------------------------------------------------------------------------------------------ reader

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.i..self.i + n).ok_or_else(|| format!("unexpected end of file at byte {}", self.i))?;
        self.i += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

fn read_prop(r: &mut Reader) -> Result<Prop, String> {
    let code = r.take(1)?[0];
    Ok(match code {
        b'Y' => Prop::Int(i16::from_le_bytes(r.take(2)?.try_into().unwrap()) as i32),
        b'C' => Prop::Int(r.take(1)?[0] as i32),
        b'I' => Prop::Int(r.u32()? as i32),
        b'F' => Prop::Double(f32::from_le_bytes(r.take(4)?.try_into().unwrap()) as f64),
        b'D' => Prop::Double(f64::from_le_bytes(r.take(8)?.try_into().unwrap())),
        b'L' => Prop::Long(i64::from_le_bytes(r.take(8)?.try_into().unwrap())),
        b'S' => {
            let n = r.u32()? as usize;
            Prop::Str(String::from_utf8_lossy(r.take(n)?).into_owned())
        }
        b'R' => {
            let n = r.u32()? as usize;
            r.take(n)?;
            Prop::Str(String::new())
        }
        b'f' | b'd' | b'l' | b'i' | b'b' => {
            let (count, enc, clen) = (r.u32()? as usize, r.u32()?, r.u32()? as usize);
            if enc != 0 {
                return Err(format!("compressed array (encoding {enc}) in a file gaw wrote itself: not supported by the read-back"));
            }
            let width = match code {
                b'b' => 1,
                b'f' | b'i' => 4,
                _ => 8,
            };
            if clen != count * width {
                return Err("array byte length does not match its count".into());
            }
            let raw = r.take(clen)?;
            match code {
                b'i' => Prop::Ints(raw.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()),
                b'd' => Prop::Doubles(raw.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()),
                b'f' => Prop::Doubles(raw.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect()),
                b'l' => Prop::Doubles(raw.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64).collect()),
                _ => Prop::Ints(raw.iter().map(|&v| v as i32).collect()),
            }
        }
        c => return Err(format!("invalid property data type `{}`", c as char)),
    })
}

fn read_node(r: &mut Reader) -> Result<Option<Node>, String> {
    let start = r.i;
    let end = r.u32()? as usize;
    let (nprops, plen) = (r.u32()? as usize, r.u32()? as usize);
    let nlen = r.take(1)?[0] as usize;
    if end == 0 {
        if nprops != 0 || plen != 0 || nlen != 0 {
            return Err("invalid node; expected NULL record".into());
        }
        return Ok(None);
    }
    let name = String::from_utf8_lossy(r.take(nlen)?).into_owned();
    let pstart = r.i;
    let mut props = Vec::with_capacity(nprops);
    for _ in 0..nprops {
        props.push(read_prop(r)?);
    }
    if r.i - pstart != plen {
        return Err(format!("node {name} at {start}: too many bytes in property list"));
    }
    let mut kids = Vec::new();
    if end > r.i {
        while let Some(k) = read_node(r)? {
            kids.push(k);
        }
    }
    if r.i != end {
        return Err(format!("node {name} at {start}: ends at {}, header says {end}", r.i));
    }
    Ok(Some(Node { name, props, kids }))
}

pub fn parse_binary(bytes: &[u8]) -> Result<Vec<Node>, String> {
    if !bytes.starts_with(&HEADER_MAGIC[..21]) {
        return Err("not a binary FBX file".into());
    }
    let version = u32::from_le_bytes(bytes.get(23..27).ok_or("short header")?.try_into().unwrap());
    if version >= 7500 {
        return Err(format!("FBX {version} uses 64-bit node records; Memoria's importer era files are 7.4 (7400)"));
    }
    let mut r = Reader { b: bytes, i: 27 };
    let mut out = Vec::new();
    while let Some(n) = read_node(&mut r)? {
        out.push(n);
    }
    let tail = bytes.len() - r.i;
    if tail < 16 + 4 + 4 + 120 + 16 {
        return Err("footer is too short".into());
    }
    if bytes[bytes.len() - 16..] != FOOT_MAGIC {
        return Err("footer magic is wrong".into());
    }
    Ok(out)
}

// ---- scene as the importer sees it

pub struct SceneBone {
    pub num: usize,
    pub class: String,
    pub parent: Option<usize>,
    pub t: [f64; 3],
    pub euler: [f64; 3],
    pub s: [f64; 3],
    /// TransformLink of any cluster that names this bone (Unity-space bone world at bind), if one does.
    pub bind_world: Option<M4Pub>,
}

pub type M4Pub = [[f64; 4]; 4];

pub struct SceneMesh {
    pub name: String,
    pub verts: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
    pub uvs: Vec<[f64; 2]>,
    pub tris: Vec<[u32; 3]>,
    /// Per vertex: `(bone number, weight)`.
    pub weights: Vec<Vec<(usize, f64)>>,
    pub texture: Option<String>,
    pub shading: Option<String>,
}

pub struct Scene {
    pub bones: Vec<SceneBone>,
    pub meshes: Vec<SceneMesh>,
}

fn obj_name(s: &str) -> &str {
    s.split(SEP).next().unwrap_or(s)
}

fn p70(model: &Node, key: &str, d: [f64; 3]) -> [f64; 3] {
    if let Some(p) = model.child("Properties70").and_then(|pp| pp.children("P").find(|p| p.text(0) == Some(key))) {
        let v: Vec<f64> = p.props.iter().skip(4).filter_map(|x| if let Prop::Double(v) = x { Some(*v) } else { None }).collect();
        if v.len() == 3 {
            return [v[0], v[1], v[2]];
        }
    }
    d
}

/// Reads the scene the way `FbxDocument` + `FbxSkeleton` do (ids as 32-bit ints, bone numbers from the trailing digits).
pub fn read_scene(bytes: &[u8]) -> Result<Scene, String> {
    let top = parse_binary(bytes)?;
    let objects = top.iter().find(|n| n.name == "Objects").ok_or("no Objects node")?;
    let connections = top.iter().find(|n| n.name == "Connections").ok_or("no Connections node")?;
    let to_i32 = |v: i64| i32::try_from(v).map_err(|_| format!("object id {v} does not fit in 32 bits (the importer converts ids with Convert.ToInt32)"));
    let mut conns: Vec<(String, i64, i64)> = Vec::new();
    for c in connections.children("C") {
        let (Some(kind), Some(a), Some(b)) = (c.text(0), c.props.get(1), c.props.get(2)) else { return Err("a connection has fewer than 3 properties".into()) };
        let id = |p: &Prop| match p {
            Prop::Long(v) => Ok(*v),
            Prop::Int(v) => Ok(*v as i64),
            _ => Err("connection id is not an integer".to_string()),
        };
        let (a, b) = (id(a)?, id(b)?);
        to_i32(a)?;
        to_i32(b)?;
        conns.push((kind.to_owned(), a, b));
    }
    let by_id: HashMap<i64, &Node> = objects.kids.iter().filter_map(|n| n.id().map(|i| (i, n))).collect();
    let parent_of = |child: i64, class: &str| -> Option<i64> { conns.iter().find(|c| c.1 == child && by_id.get(&c.2).is_some_and(|n| n.name == class)).map(|c| c.2) };
    let child_of = |parent: i64, class: &str| -> Vec<i64> { conns.iter().filter(|c| c.2 == parent && by_id.get(&c.1).is_some_and(|n| n.name == class)).map(|c| c.1).collect() };

    // bones: Model nodes typed Root / LimbNode
    let mut bone_ids: Vec<(i64, usize)> = Vec::new();
    for (id, n) in &by_id {
        if n.name != "Model" {
            continue;
        }
        let class = n.text(2).unwrap_or("");
        if class == "Root" || class == "LimbNode" {
            let nm = obj_name(n.text(1).unwrap_or(""));
            let digits: String = nm.chars().rev().take(3).collect::<Vec<_>>().into_iter().rev().collect();
            let num: usize = digits.parse().map_err(|_| format!("bone `{nm}` does not end in three digits"))?;
            if class == "Root" && num != 0 {
                return Err(format!("the Root model is `{nm}`; the importer forces it to bone 0"));
            }
            if class == "LimbNode" && num == 0 {
                return Err(format!("LimbNode `{nm}` ends in 000, which the importer reserves for the Root"));
            }
            bone_ids.push((*id, num));
        }
    }
    bone_ids.sort_by_key(|b| b.1);
    let num_of: HashMap<i64, usize> = bone_ids.iter().copied().collect();
    let mut bones = Vec::new();
    for &(id, num) in &bone_ids {
        let n = by_id[&id];
        // parent: the first connection whose child is this bone and whose parent is a Model
        let parent = conns.iter().find(|c| c.1 == id && by_id.get(&c.2).is_some_and(|p| p.name == "Model")).map(|c| c.2);
        let parent = match parent {
            Some(p) if num_of.contains_key(&p) => Some(num_of[&p]),
            Some(_) => None, // the Armature null
            None => None,
        };
        bones.push(SceneBone { num, class: n.text(2).unwrap_or("").to_owned(), parent, t: p70(n, "Lcl Translation", [0.0; 3]), euler: p70(n, "Lcl Rotation", [0.0; 3]), s: p70(n, "Lcl Scaling", [1.0; 3]), bind_world: None });
    }
    if bones.iter().enumerate().any(|(i, b)| b.num != i) {
        return Err("bone numbers are not 0..n-1".into());
    }

    let mut meshes = Vec::new();
    let mut geos: Vec<(&i64, &&Node)> = by_id.iter().filter(|(_, n)| n.name == "Geometry").collect();
    geos.sort_by_key(|g| *g.0);
    for (gid, g) in geos {
        let model_id = child_of(*gid, "Model").first().copied().or_else(|| parent_of(*gid, "Model")).ok_or("geometry without a mesh model")?;
        let model_node = by_id[&model_id];
        let verts: Vec<[f64; 3]> = g.child("Vertices").map(|n| n.doubles()).unwrap_or_default().chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        let poly = g.child("PolygonVertexIndex").map(|n| n.ints()).unwrap_or_default();
        let mut tris = Vec::new();
        let mut face: Vec<u32> = Vec::new();
        for p in poly {
            if p < 0 {
                face.push((!p) as u32);
                for k in 1..face.len() - 1 {
                    tris.push([face[0], face[k], face[k + 1]]);
                }
                face.clear();
            } else {
                face.push(p as u32);
            }
        }
        let layer = |name: &str, data: &str| -> Result<Vec<f64>, String> {
            let Some(l) = g.child(name) else { return Ok(vec![]) };
            let map = l.child("MappingInformationType").and_then(|n| n.text(0)).unwrap_or("");
            let rf = l.child("ReferenceInformationType").and_then(|n| n.text(0)).unwrap_or("");
            if map != "ByVertex" && map != "ByVertice" || rf != "Direct" {
                return Err(format!("{name}: only ByVertex/Direct is read here, found {map}/{rf}"));
            }
            Ok(l.child(data).map(|n| n.doubles()).unwrap_or_default())
        };
        let uvs: Vec<[f64; 2]> = layer("LayerElementUV", "UV")?.chunks(2).map(|c| [c[0], c[1]]).collect();
        let normals: Vec<[f64; 3]> = layer("LayerElementNormal", "Normals")?.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        let mut weights = vec![Vec::new(); verts.len()];
        for skin in child_of(*gid, "Deformer") {
            if by_id[&skin].text(2) != Some("Skin") {
                continue;
            }
            for cl in child_of(skin, "Deformer") {
                let cn = by_id[&cl];
                if cn.text(2) != Some("Cluster") {
                    continue;
                }
                // the bone is the child whose parent is the cluster
                let Some(bid) = conns.iter().find(|c| c.2 == cl && num_of.contains_key(&c.1)).map(|c| c.1) else { continue };
                let num = num_of[&bid];
                let idx = cn.child("Indexes").map(|n| n.ints()).unwrap_or_default();
                let w = cn.child("Weights").map(|n| n.doubles()).unwrap_or_default();
                if idx.len() != w.len() {
                    return Err("a cluster has different Indexes and Weights lengths".into());
                }
                for (i, w) in idx.iter().zip(w) {
                    weights.get_mut(*i as usize).ok_or("cluster index past the vertex count")?.push((num, w));
                }
                if let Some(tl) = cn.child("TransformLink") {
                    let v = tl.doubles();
                    if v.len() == 16 {
                        bones[num].bind_world = Some(from_col_major(&v));
                    }
                }
            }
        }
        let mat = child_of(model_id, "Material").first().copied();
        let (mut texture, mut shading) = (None, None);
        if let Some(m) = mat {
            shading = by_id[&m].child("ShadingModel").and_then(|n| n.text(0)).map(str::to_owned);
            let tex = conns.iter().find(|c| c.0 == "OP" && c.2 == m && by_id.get(&c.1).is_some_and(|n| n.name == "Texture")).map(|c| c.1);
            texture = tex.and_then(|t| by_id[&t].child("RelativeFilename").and_then(|n| n.text(0)).map(str::to_owned));
        }
        meshes.push(SceneMesh { name: obj_name(model_node.text(1).unwrap_or("")).to_owned(), verts, normals, uvs, tris, weights, texture, shading });
    }
    Ok(Scene { bones, meshes })
}

impl SceneBone {
    /// Local rotation as a quaternion (x, y, z, w), the way the importer composes Euler angles.
    pub fn quat(&self) -> [f64; 4] {
        quat_from_euler_xyz(self.euler)
    }
}

/// Unity-space bone world matrices (row-major) composed from the local transforms.
pub fn scene_world(scene: &Scene) -> Vec<[[f64; 4]; 4]> {
    let mut out: Vec<[[f64; 4]; 4]> = Vec::new();
    for b in &scene.bones {
        let local = trs(b.t, b.quat(), b.s);
        out.push(match b.parent {
            Some(p) => mmul(&out[p], &local),
            None => local,
        });
    }
    out
}

/// Inverse of a 4x4 (exposed for the checks).
pub fn invert(m: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    minv(m)
}

pub fn multiply(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    mmul(a, b)
}
