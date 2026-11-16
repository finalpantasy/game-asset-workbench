//! Skeleton, clips and skinning, read from the GLB (glTF 2.0 rules).
use crate::glb::Glb;
use crate::model::Mesh;
use serde_json::Value;

pub type Mat4 = [f32; 16]; // column-major

pub fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut r = [0.0; 16];
    for c in 0..4 {
        for row in 0..4 {
            r[c * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[c * 4 + k]).sum();
        }
    }
    r
}

pub fn from_trs(t: [f32; 3], q: [f32; 4], s: [f32; 3]) -> Mat4 {
    let [x, y, z, w] = q;
    let (xx, yy, zz, xy, xz, yz, wx, wy, wz) = (x * x, y * y, z * z, x * y, x * z, y * z, w * x, w * y, w * z);
    [
        (1.0 - 2.0 * (yy + zz)) * s[0], 2.0 * (xy + wz) * s[0], 2.0 * (xz - wy) * s[0], 0.0,
        2.0 * (xy - wz) * s[1], (1.0 - 2.0 * (xx + zz)) * s[1], 2.0 * (yz + wx) * s[1], 0.0,
        2.0 * (xz + wy) * s[2], 2.0 * (yz - wx) * s[2], (1.0 - 2.0 * (xx + yy)) * s[2], 0.0,
        t[0], t[1], t[2], 1.0,
    ]
}

fn transform(m: &Mat4, p: [f32; 3]) -> [f32; 3] {
    [
        m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
        m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
        m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    ]
}

const IDENTITY: Mat4 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

#[derive(Clone)]
struct Node {
    parent: Option<usize>,
    t: [f32; 3],
    r: [f32; 4],
    s: [f32; 3],
    matrix: Option<Mat4>,
}

#[derive(Clone, Copy, PartialEq)]
enum PathKind {
    T,
    R,
    S,
}

#[derive(Clone, Copy, PartialEq)]
enum Interp {
    Step,
    Linear,
    Cubic,
}

struct Channel {
    node: usize,
    path: PathKind,
    interp: Interp,
    times: Vec<f32>,
    values: Vec<[f32; 4]>,
}

pub struct Clip {
    pub name: String,
    pub duration: f32,
    channels: Vec<Channel>,
}

pub struct Rig {
    nodes: Vec<Node>,
    /// Nodes ordered so every parent comes before its children.
    order: Vec<usize>,
    pub joints: Vec<usize>,
    ibm: Vec<Mat4>,
    pub clips: Vec<Clip>,
}

fn arr<const N: usize>(v: Option<&Value>, default: [f32; N]) -> [f32; N] {
    match v.and_then(Value::as_array) {
        Some(a) if a.len() == N => std::array::from_fn(|i| a[i].as_f64().unwrap_or(default[i] as f64) as f32),
        _ => default,
    }
}

impl Rig {
    pub fn from_glb(glb: &Glb) -> Result<Option<Rig>, String> {
        let Some(skin) = glb.list("skins").first() else { return Ok(None) };
        let jn = glb.list("nodes");
        let mut nodes: Vec<Node> = jn
            .iter()
            .map(|n| Node {
                parent: None,
                t: arr(n.get("translation"), [0.0; 3]),
                r: arr(n.get("rotation"), [0.0, 0.0, 0.0, 1.0]),
                s: arr(n.get("scale"), [1.0; 3]),
                matrix: n.get("matrix").map(|m| arr::<16>(Some(m), IDENTITY)),
            })
            .collect();
        for (i, n) in jn.iter().enumerate() {
            for c in n.get("children").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_u64) {
                if let Some(child) = nodes.get_mut(c as usize) {
                    child.parent = Some(i);
                }
            }
        }
        let mut order = Vec::with_capacity(nodes.len());
        let mut placed = vec![false; nodes.len()];
        while order.len() < nodes.len() {
            let before = order.len();
            for i in 0..nodes.len() {
                if !placed[i] && nodes[i].parent.map_or(true, |p| placed[p]) {
                    placed[i] = true;
                    order.push(i);
                }
            }
            if order.len() == before {
                return Err("node hierarchy has a cycle".into());
            }
        }
        let joints: Vec<usize> = skin.get("joints").and_then(Value::as_array).ok_or("skin without joints")?.iter().filter_map(Value::as_u64).map(|j| j as usize).collect();
        let ibm = match skin.get("inverseBindMatrices").and_then(Value::as_u64) {
            Some(a) => {
                let acc = glb.accessor(a as usize)?;
                (0..acc.count).map(|i| std::array::from_fn(|k| acc.get(i, k) as f32)).collect()
            }
            None => vec![IDENTITY; joints.len()],
        };
        let mut clips = Vec::new();
        for (ai, a) in glb.list("animations").iter().enumerate() {
            let samplers = a.get("samplers").and_then(Value::as_array).cloned().unwrap_or_default();
            let mut channels = Vec::new();
            let mut duration = 0f32;
            for ch in a.get("channels").and_then(Value::as_array).into_iter().flatten() {
                let target = &ch["target"];
                let (Some(node), Some(path)) = (target["node"].as_u64(), target["path"].as_str()) else { continue };
                let path = match path {
                    "translation" => PathKind::T,
                    "rotation" => PathKind::R,
                    "scale" => PathKind::S,
                    _ => continue, // morph weights: not used by FF9 exports
                };
                let s = samplers.get(ch["sampler"].as_u64().unwrap_or(0) as usize).ok_or("channel points to a missing sampler")?;
                let interp = match s.get("interpolation").and_then(Value::as_str).unwrap_or("LINEAR") {
                    "STEP" => Interp::Step,
                    "CUBICSPLINE" => Interp::Cubic,
                    _ => Interp::Linear,
                };
                let input = glb.accessor(s["input"].as_u64().ok_or("sampler without input")? as usize)?;
                let output = glb.accessor(s["output"].as_u64().ok_or("sampler without output")? as usize)?;
                let times: Vec<f32> = (0..input.count).map(|i| input.get(i, 0) as f32).collect();
                let values = (0..output.count).map(|i| std::array::from_fn(|k| if k < output.ncomp { output.get(i, k) as f32 } else { 0.0 })).collect();
                duration = duration.max(times.last().copied().unwrap_or(0.0));
                channels.push(Channel { node: node as usize, path, interp, times, values });
            }
            let name = a.get("name").and_then(Value::as_str).map(str::to_owned).unwrap_or_else(|| format!("clip{ai}"));
            clips.push(Clip { name, duration, channels });
        }
        Ok(Some(Rig { nodes, order, joints, ibm, clips }))
    }

    /// Joint matrices (world * inverse bind) for a clip at time `t` seconds, or the rest pose.
    pub fn joint_matrices(&self, clip: Option<usize>, t: f32) -> Vec<Mat4> {
        let mut local: Vec<Node> = self.nodes.clone();
        if let Some(c) = clip.and_then(|c| self.clips.get(c)) {
            let t = if c.duration > 0.0 { t.rem_euclid(c.duration) } else { 0.0 };
            for ch in &c.channels {
                let v = sample(ch, t);
                if let Some(n) = local.get_mut(ch.node) {
                    n.matrix = None;
                    match ch.path {
                        PathKind::T => n.t = [v[0], v[1], v[2]],
                        PathKind::R => n.r = v,
                        PathKind::S => n.s = [v[0], v[1], v[2]],
                    }
                }
            }
        }
        let mut world = vec![IDENTITY; local.len()];
        for &i in &self.order {
            let n = &local[i];
            let m = n.matrix.unwrap_or_else(|| from_trs(n.t, n.r, n.s));
            world[i] = match n.parent {
                Some(p) => mat_mul(&world[p], &m),
                None => m,
            };
        }
        self.joints.iter().enumerate().map(|(k, &j)| mat_mul(&world[j], &self.ibm[k])).collect()
    }
}

fn sample(ch: &Channel, t: f32) -> [f32; 4] {
    let n = ch.times.len();
    // Cubic spline output holds (in-tangent, value, out-tangent) per key; use the value.
    let val = |k: usize| if ch.interp == Interp::Cubic { ch.values[k * 3 + 1] } else { ch.values[k] };
    if n == 0 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    if t <= ch.times[0] || n == 1 {
        return val(0);
    }
    if t >= ch.times[n - 1] {
        return val(n - 1);
    }
    let k = ch.times.partition_point(|&x| x <= t) - 1;
    let (a, b) = (val(k), val(k + 1));
    if ch.interp == Interp::Step {
        return a;
    }
    let f = (t - ch.times[k]) / (ch.times[k + 1] - ch.times[k]).max(1e-9);
    if ch.path == PathKind::R {
        let d: f32 = (0..4).map(|i| a[i] * b[i]).sum();
        let sgn = if d < 0.0 { -1.0 } else { 1.0 };
        let mut q: [f32; 4] = std::array::from_fn(|i| a[i] * (1.0 - f) + b[i] * f * sgn);
        let l = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        q.iter_mut().for_each(|x| *x /= l);
        q
    } else {
        std::array::from_fn(|i| a[i] * (1.0 - f) + b[i] * f)
    }
}

impl Rig {
    /// Per joint, in the bind pose (object space): the joint's head and the direction its bone runs
    /// (toward the mean head of its child joints, else away from its parent joint, else +Y).
    pub fn bind_bones(&self) -> Vec<([f64; 3], [f64; 3])> {
        let n = self.joints.len();
        let world: Vec<Option<Mat4>> = (0..n).map(|k| invert_affine(&self.ibm[k])).collect();
        let head: Vec<[f64; 3]> = world.iter().map(|w| w.map(|m| [m[12] as f64, m[13] as f64, m[14] as f64]).unwrap_or([0.0; 3])).collect();
        let joint_of_node: std::collections::HashMap<usize, usize> = self.joints.iter().enumerate().map(|(k, &j)| (j, k)).collect();
        let parent: Vec<Option<usize>> = (0..n)
            .map(|k| {
                let mut at = self.nodes[self.joints[k]].parent;
                while let Some(p) = at {
                    if let Some(&pk) = joint_of_node.get(&p) {
                        return Some(pk);
                    }
                    at = self.nodes[p].parent;
                }
                None
            })
            .collect();
        (0..n)
            .map(|k| {
                let kids: Vec<usize> = (0..n).filter(|&c| parent[c] == Some(k)).collect();
                let unit = |v: [f64; 3]| {
                    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    if l > 1e-9 { Some([v[0] / l, v[1] / l, v[2] / l]) } else { None }
                };
                let toward = if kids.is_empty() {
                    None
                } else {
                    let m = kids.iter().fold([0.0; 3], |a, &c| [a[0] + head[c][0], a[1] + head[c][1], a[2] + head[c][2]]);
                    let m = [m[0] / kids.len() as f64, m[1] / kids.len() as f64, m[2] / kids.len() as f64];
                    unit([m[0] - head[k][0], m[1] - head[k][1], m[2] - head[k][2]])
                };
                let away = parent[k].and_then(|p| unit([head[k][0] - head[p][0], head[k][1] - head[p][1], head[k][2] - head[p][2]]));
                (head[k], toward.or(away).unwrap_or([0.0, 1.0, 0.0]))
            })
            .collect()
    }
}

/// Inverse of an affine column-major matrix (None when its 3x3 part is singular).
fn invert_affine(m: &Mat4) -> Option<Mat4> {
    let a = [[m[0] as f64, m[4] as f64, m[8] as f64], [m[1] as f64, m[5] as f64, m[9] as f64], [m[2] as f64, m[6] as f64, m[10] as f64]];
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-18 {
        return None;
    }
    let i = [
        [(a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det, (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det, (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det],
        [(a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det, (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det, (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det],
        [(a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det, (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det, (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det],
    ];
    let t = [m[12] as f64, m[13] as f64, m[14] as f64];
    let it = [
        -(i[0][0] * t[0] + i[0][1] * t[1] + i[0][2] * t[2]),
        -(i[1][0] * t[0] + i[1][1] * t[1] + i[1][2] * t[2]),
        -(i[2][0] * t[0] + i[2][1] * t[1] + i[2][2] * t[2]),
    ];
    Some([
        i[0][0] as f32, i[1][0] as f32, i[2][0] as f32, 0.0,
        i[0][1] as f32, i[1][1] as f32, i[2][1] as f32, 0.0,
        i[0][2] as f32, i[1][2] as f32, i[2][2] as f32, 0.0,
        it[0] as f32, it[1] as f32, it[2] as f32, 1.0,
    ])
}

/// Linear blend skinning of one mesh's vertices.
pub fn skin(mesh: &Mesh, joints: &[Mat4]) -> Vec<[f32; 3]> {
    mesh.pos
        .iter()
        .enumerate()
        .map(|(v, &p)| {
            let mut out = [0.0f32; 3];
            let mut wsum = 0.0;
            for c in 0..4 {
                let w = mesh.weights[v][c];
                if w <= 0.0 {
                    continue;
                }
                let Some(m) = joints.get(mesh.joints[v][c] as usize) else { continue };
                let q = transform(m, p);
                for k in 0..3 {
                    out[k] += w * q[k];
                }
                wsum += w;
            }
            if wsum > 0.0 {
                out.map(|x| x / wsum)
            } else {
                p
            }
        })
        .collect()
}
