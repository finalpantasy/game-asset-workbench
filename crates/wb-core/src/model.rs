//! A read-only view of a skinned FF9 character export (GLB).
use crate::glb::Glb;
use serde_json::Value;
use std::path::Path;

pub struct Mesh {
    pub name: String,
    pub pos: Vec<[f32; 3]>,
    pub uv: Vec<[f32; 2]>,
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub tris: Vec<[u32; 3]>,
    /// Index into `Model::images` of the base color texture.
    pub image: Option<usize>,
}

pub struct Image {
    pub name: String,
    pub mime: String,
    pub view: usize,
    pub bytes: Vec<u8>,
}

pub struct Model {
    pub glb: Glb,
    pub file_len: usize,
    pub meshes: Vec<Mesh>,
    pub images: Vec<Image>,
    pub joint_names: Vec<String>,
    pub clips: Vec<String>,
    pub accessors: usize,
    pub nodes: usize,
    pub rig: Option<crate::anim::Rig>,
}

fn idx(v: &Value, k: &str) -> Option<usize> {
    v.get(k).and_then(Value::as_u64).map(|x| x as usize)
}

impl Model {
    pub fn open(path: &Path) -> Result<Model, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Model::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Model, String> {
        let glb = Glb::parse(bytes)?;
        let node_name = |i: usize| -> String {
            glb.list("nodes").get(i).and_then(|n| n.get("name")).and_then(Value::as_str).map(str::to_owned).unwrap_or_else(|| format!("node{i}"))
        };
        let mut images = Vec::new();
        for (i, im) in glb.list("images").iter().enumerate() {
            let view = idx(im, "bufferView").ok_or(format!("image {i} is not embedded"))?;
            images.push(Image {
                name: im.get("name").and_then(Value::as_str).map(str::to_owned).unwrap_or_else(|| format!("image{i}")),
                mime: im.get("mimeType").and_then(Value::as_str).unwrap_or("").to_owned(),
                view,
                bytes: glb.view_bytes(view)?.to_vec(),
            });
        }
        let material_image = |m: usize| -> Option<usize> {
            let tex = glb.list("materials").get(m)?.get("pbrMetallicRoughness")?.get("baseColorTexture")?.get("index")?.as_u64()? as usize;
            idx(glb.list("textures").get(tex)?, "source")
        };
        let mut meshes = Vec::new();
        for (mi, m) in glb.list("meshes").iter().enumerate() {
            let mname = m.get("name").and_then(Value::as_str).map(str::to_owned).unwrap_or_else(|| format!("mesh{mi}"));
            let prims = m.get("primitives").and_then(Value::as_array).cloned().unwrap_or_default();
            for (pi, p) in prims.iter().enumerate() {
                let mode = p.get("mode").and_then(Value::as_u64).unwrap_or(4);
                if mode != 4 {
                    return Err(format!("{mname} primitive {pi}: mode {mode}, only triangle lists are supported"));
                }
                let at = p.get("attributes").ok_or("primitive without attributes")?;
                let get = |k: &str| -> Result<Option<crate::glb::Accessor>, String> {
                    match idx(at, k) {
                        Some(a) => glb.accessor(a).map(Some),
                        None => Ok(None),
                    }
                };
                let pos = get("POSITION")?.ok_or("primitive without POSITION")?;
                let n = pos.count;
                let uv = get("TEXCOORD_0")?;
                let jn = get("JOINTS_0")?;
                let wt = get("WEIGHTS_0")?;
                let tris = match idx(p, "indices") {
                    Some(a) => {
                        let ia = glb.accessor(a)?;
                        (0..ia.count / 3).map(|t| [ia.get(3 * t, 0) as u32, ia.get(3 * t + 1, 0) as u32, ia.get(3 * t + 2, 0) as u32]).collect()
                    }
                    None => (0..n as u32 / 3).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect(),
                };
                meshes.push(Mesh {
                    name: if prims.len() > 1 { format!("{mname}#{pi}") } else { mname.clone() },
                    pos: (0..n).map(|i| [pos.get(i, 0) as f32, pos.get(i, 1) as f32, pos.get(i, 2) as f32]).collect(),
                    uv: match &uv {
                        Some(u) => (0..n).map(|i| [u.get(i, 0) as f32, u.get(i, 1) as f32]).collect(),
                        None => vec![[0.0; 2]; n],
                    },
                    joints: match &jn {
                        Some(j) => (0..n).map(|i| std::array::from_fn(|c| j.get(i, c) as u16)).collect(),
                        None => vec![[0; 4]; n],
                    },
                    weights: match &wt {
                        Some(w) => (0..n).map(|i| std::array::from_fn(|c| w.get(i, c) as f32)).collect(),
                        None => vec![[1.0, 0.0, 0.0, 0.0]; n],
                    },
                    tris,
                    image: idx(p, "material").and_then(material_image),
                });
            }
        }
        let joint_names = glb
            .list("skins")
            .first()
            .and_then(|s| s.get("joints"))
            .and_then(Value::as_array)
            .map(|js| js.iter().filter_map(Value::as_u64).map(|j| node_name(j as usize)).collect())
            .unwrap_or_default();
        let clips = glb
            .list("animations")
            .iter()
            .enumerate()
            .map(|(i, a)| a.get("name").and_then(Value::as_str).map(str::to_owned).unwrap_or_else(|| format!("clip{i}")))
            .collect();
        let rig = crate::anim::Rig::from_glb(&glb)?;
        Ok(Model {
            rig,
            accessors: glb.list("accessors").len(),
            nodes: glb.list("nodes").len(),
            file_len: bytes.len(),
            glb,
            meshes,
            images,
            joint_names,
            clips,
        })
    }

    /// Vertex positions of every mesh posed by a clip at `t` seconds (None = rest pose).
    pub fn posed(&self, clip: Option<usize>, t: f32) -> Vec<Vec<[f32; 3]>> {
        match &self.rig {
            Some(rig) => {
                let jm = rig.joint_matrices(clip, t);
                self.meshes.iter().map(|m| crate::anim::skin(m, &jm)).collect()
            }
            None => self.meshes.iter().map(|m| m.pos.clone()).collect(),
        }
    }

    pub fn clip_duration(&self, clip: usize) -> f32 {
        self.rig.as_ref().and_then(|r| r.clips.get(clip)).map(|c| c.duration).unwrap_or(0.0)
    }

    pub fn triangle_count(&self) -> usize {
        self.meshes.iter().map(|m| m.tris.len()).sum()
    }

    /// Atlas name for a mesh (the image name without extension, e.g. "98_1").
    pub fn atlas_name(&self, mesh: usize) -> String {
        match self.meshes[mesh].image {
            Some(i) => self.images[i].name.trim_end_matches(".png").to_owned(),
            None => "none".into(),
        }
    }

    /// The joint with the most total weight over a triangle's three corners.
    pub fn triangle_joint(&self, mesh: usize, tri: usize) -> usize {
        let m = &self.meshes[mesh];
        let mut acc: Vec<(u16, f32)> = Vec::with_capacity(12);
        for &v in &m.tris[tri] {
            for c in 0..4 {
                let (j, w) = (m.joints[v as usize][c], m.weights[v as usize][c]);
                if w <= 0.0 {
                    continue;
                }
                match acc.iter_mut().find(|e| e.0 == j) {
                    Some(e) => e.1 += w,
                    None => acc.push((j, w)),
                }
            }
        }
        acc.iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|e| e.0 as usize).unwrap_or(0)
    }
}
