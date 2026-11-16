//! Writes a model with new vertices and triangles (a new UV layout that cuts new seams).
//!
//! Every primitive gets fresh vertex data: each new vertex copies all attributes (position, normal,
//! skin joints and weights, ...) byte for byte from a source vertex, and TEXCOORD_0 takes the new UV.
//! The new data is appended to the BIN chunk as new buffer views and accessors; the old ones stay
//! (unused) so images, skins and animations keep their offsets. Nothing else in the JSON changes.
use crate::glb::{Chunk, Glb, CHUNK_BIN, CHUNK_JSON};
use crate::Model;
use serde_json::{json, Value};

/// One primitive's new geometry, in the same order as `Model::meshes`.
pub struct NewMesh {
    /// Per new vertex: the source vertex whose attributes it copies.
    pub src: Vec<u32>,
    /// Per new vertex: its UV (glTF convention, v down).
    pub uv: Vec<[f32; 2]>,
    pub tris: Vec<[u32; 3]>,
}

fn comp_size(ct: u64) -> Result<usize, String> {
    match ct {
        5120 | 5121 => Ok(1),
        5122 | 5123 => Ok(2),
        5125 | 5126 => Ok(4),
        _ => Err(format!("componentType {ct}")),
    }
}

fn ncomp(t: &str) -> Result<usize, String> {
    Ok(match t {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        "MAT4" => 16,
        _ => return Err(format!("accessor type {t}")),
    })
}

fn push_view(json: &mut Value, bin: &mut Vec<u8>, bytes: &[u8], target: u64) -> usize {
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let off = bin.len();
    bin.extend_from_slice(bytes);
    let views = json["bufferViews"].as_array_mut().expect("bufferViews");
    views.push(json!({ "buffer": 0, "byteOffset": off, "byteLength": bytes.len(), "target": target }));
    views.len() - 1
}

fn push_accessor(json: &mut Value, a: Value) -> usize {
    let list = json["accessors"].as_array_mut().expect("accessors");
    list.push(a);
    list.len() - 1
}

pub fn rebuild(model: &Model, new: &[NewMesh]) -> Result<Vec<u8>, String> {
    let glb = &model.glb;
    let mut json = glb.json.clone();
    let mut bin = glb.bin().to_vec();
    let mut prims = Vec::new();
    for (mi, m) in glb.list("meshes").iter().enumerate() {
        for pi in 0..m.get("primitives").and_then(Value::as_array).map(|p| p.len()).unwrap_or(0) {
            prims.push((mi, pi));
        }
    }
    if prims.len() != new.len() {
        return Err(format!("{} primitives, {} new meshes", prims.len(), new.len()));
    }
    for (k, &(mi, pi)) in prims.iter().enumerate() {
        let nm = &new[k];
        if nm.src.len() != nm.uv.len() {
            return Err("src and uv lengths differ".into());
        }
        let old_n = model.meshes[k].pos.len() as u32;
        if let Some(bad) = nm.src.iter().find(|&&s| s >= old_n) {
            return Err(format!("mesh {k}: source vertex {bad} out of range"));
        }
        if let Some(t) = nm.tris.iter().find(|t| t.iter().any(|&v| v as usize >= nm.src.len())) {
            return Err(format!("mesh {k}: triangle {t:?} out of range"));
        }
        let attrs: Vec<(String, usize)> = json["meshes"][mi]["primitives"][pi]["attributes"]
            .as_object()
            .ok_or("primitive without attributes")?
            .iter()
            .map(|(n, a)| (n.clone(), a.as_u64().unwrap_or(0) as usize))
            .collect();
        for (name, ai) in attrs {
            let a = json["accessors"][ai].clone();
            if a.get("sparse").is_some() {
                return Err(format!("{name}: sparse accessors are not supported"));
            }
            let ct = a["componentType"].as_u64().unwrap_or(0);
            let nc = ncomp(a["type"].as_str().unwrap_or(""))?;
            let elem = comp_size(ct)? * nc;
            let bytes: Vec<u8> = if name == "TEXCOORD_0" {
                if ct != 5126 || nc != 2 {
                    return Err("TEXCOORD_0 must be float VEC2".into());
                }
                nm.uv.iter().flat_map(|u| u[0].to_le_bytes().into_iter().chain(u[1].to_le_bytes())).collect()
            } else {
                let view = a["bufferView"].as_u64().ok_or("accessor without bufferView")? as usize;
                let vb = glb.view_bytes(view)?;
                let stride = glb.list("bufferViews")[view].get("byteStride").and_then(Value::as_u64).map(|s| s as usize).unwrap_or(elem);
                let base = a.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
                let mut out = Vec::with_capacity(nm.src.len() * elem);
                for &s in &nm.src {
                    let o = base + s as usize * stride;
                    out.extend_from_slice(vb.get(o..o + elem).ok_or(format!("{name} reads past its view"))?);
                }
                out
            };
            let view = push_view(&mut json, &mut bin, &bytes, 34962);
            let mut na = a.clone();
            let o = na.as_object_mut().unwrap();
            o.insert("bufferView".into(), json!(view));
            o.remove("byteOffset");
            o.insert("count".into(), json!(nm.src.len()));
            if name == "TEXCOORD_0" {
                let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
                for u in &nm.uv {
                    for c in 0..2 {
                        lo[c] = lo[c].min(u[c]);
                        hi[c] = hi[c].max(u[c]);
                    }
                }
                o.insert("min".into(), json!(lo));
                o.insert("max".into(), json!(hi));
            }
            let na_i = push_accessor(&mut json, na);
            json["meshes"][mi]["primitives"][pi]["attributes"][&name] = json!(na_i);
        }
        let wide = nm.src.len() > 65535;
        let ib: Vec<u8> = if wide {
            nm.tris.iter().flatten().flat_map(|v| v.to_le_bytes()).collect()
        } else {
            nm.tris.iter().flatten().flat_map(|&v| (v as u16).to_le_bytes()).collect()
        };
        let view = push_view(&mut json, &mut bin, &ib, 34963);
        let ia = push_accessor(&mut json, json!({ "bufferView": view, "componentType": if wide { 5125 } else { 5123 }, "count": nm.tris.len() * 3, "type": "SCALAR" }));
        json["meshes"][mi]["primitives"][pi]["indices"] = json!(ia);
    }
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    json["buffers"][0]["byteLength"] = json!(bin.len());
    let mut jb = serde_json::to_vec(&json).map_err(|e| e.to_string())?;
    while jb.len() % 4 != 0 {
        jb.push(b' ');
    }
    let mut chunks = vec![Chunk { kind: CHUNK_JSON, data: jb }, Chunk { kind: CHUNK_BIN, data: bin }];
    chunks.extend(glb.chunks.iter().skip(1).filter(|c| c.kind != CHUNK_BIN).map(|c| Chunk { kind: c.kind, data: c.data.clone() }));
    Ok(Glb { version: glb.version, chunks, json }.to_bytes())
}
