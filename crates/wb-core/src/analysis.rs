//! Mesh checks ported from the Python UV kit (experiments/zidane-uv-kit-claude-v1).
use crate::model::Model;
use serde_json::Value;
use std::path::Path;

/// Signed UV area of a triangle in texels at `size` x `size`.
pub fn uv_area_px(model: &Model, mesh: usize, tri: usize, size: f64) -> f64 {
    let m = &model.meshes[mesh];
    let [a, b, c] = m.tris[tri].map(|v| m.uv[v as usize]);
    let (e1x, e1y) = ((b[0] - a[0]) as f64 * size, (b[1] - a[1]) as f64 * size);
    let (e2x, e2y) = ((c[0] - a[0]) as f64 * size, (c[1] - a[1]) as f64 * size);
    0.5 * (e1x * e2y - e1y * e2x)
}

/// A triangle with almost no texture area: it shows on the model but no pixel paints it.
/// Rule: under 0.01 texel² at 512 (the stock texture size). The kit's count, 116 on stock Zidane.
pub const DEAD_AREA_AT_512: f64 = 0.01;

pub fn is_dead(model: &Model, mesh: usize, tri: usize) -> bool {
    uv_area_px(model, mesh, tri, 512.0).abs() < DEAD_AREA_AT_512
}

/// Body-region label from a bone name: hands, shins, feet, or the bone name itself.
pub fn region(bone: &str) -> String {
    let b = bone.to_ascii_lowercase();
    for (k, r) in [("hand", "hands"), ("shin", "shins"), ("foot", "feet")] {
        if b.contains(k) {
            return r.into();
        }
    }
    bone.to_owned()
}

/// Texels covered by at least one triangle at `size`, using the kit's rule
/// (pixel centers at +0.5, inside when every barycentric is >= -1e-7).
/// Pass every mesh that uses the same atlas, so shared texels count once.
pub fn covered_texels(model: &Model, meshes: &[usize], size: usize) -> usize {
    let s = size as f64;
    let mut mask = vec![false; size * size];
    let eps = 1e-7;
    for t in meshes.iter().flat_map(|&mi| model.meshes[mi].tris.iter().map(move |t| (mi, t))) {
        let (mi, t) = t;
        let m = &model.meshes[mi];
        let p = t.map(|v| [m.uv[v as usize][0] as f64 * s, m.uv[v as usize][1] as f64 * s]);
        let (ax, ay, bx, by, cx, cy) = (p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1]);
        let den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
        if den.abs() < 1e-12 {
            continue;
        }
        let x0 = ((ax.min(bx).min(cx) - 0.5).floor() as i64).max(0) as usize;
        let x1 = ((ax.max(bx).max(cx) + 0.5).ceil() as i64).clamp(0, size as i64) as usize;
        let y0 = ((ay.min(by).min(cy) - 0.5).floor() as i64).max(0) as usize;
        let y1 = ((ay.max(by).max(cy) + 0.5).ceil() as i64).clamp(0, size as i64) as usize;
        for y in y0..y1 {
            let py = y as f64 + 0.5;
            for x in x0..x1 {
                let px = x as f64 + 0.5;
                let l0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den;
                let l1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den;
                let l2 = 1.0 - l0 - l1;
                if l0 >= -eps && l1 >= -eps && l2 >= -eps {
                    mask[y * size + x] = true;
                }
            }
        }
    }
    mask.iter().filter(|&&b| b).count()
}

/// The kit's named parts: a part-id image per atlas (pixel value = part id) plus names.
pub struct Parts {
    pub names: Vec<(u8, String)>,
    /// (atlas name, size, ids)
    pub maps: Vec<(String, usize, Vec<u8>)>,
    /// Island labels and triangle counts per atlas, from the kit's `guides/<atlas>/legend.json`.
    pub islands: Vec<(String, Vec<(String, usize)>)>,
}

impl Parts {
    pub fn load(dir: &Path) -> Result<Parts, String> {
        let j: Value = serde_json::from_slice(&std::fs::read(dir.join("parts-map.json")).map_err(|e| format!("parts-map.json: {e}"))?)
            .map_err(|e| format!("parts-map.json: {e}"))?;
        let mut names: Vec<(u8, String)> = j["parts"]
            .as_object()
            .ok_or("parts-map.json has no parts")?
            .iter()
            .filter_map(|(k, v)| v["id"].as_u64().map(|id| (id as u8, k.clone())))
            .collect();
        names.sort();
        let mut maps = Vec::new();
        for atlas in j["atlases"].as_object().map(|o| o.keys().cloned().collect::<Vec<_>>()).unwrap_or_default() {
            let p = dir.join(format!("{atlas}-part-ids.png"));
            let img = image::open(&p).map_err(|e| format!("{}: {e}", p.display()))?.to_luma8();
            if img.width() != img.height() {
                return Err(format!("{} is not square", p.display()));
            }
            maps.push((atlas, img.width() as usize, img.into_raw()));
        }
        // Optional: island labels written by the kit's guide_kit.py next to the parts folder.
        let mut islands = Vec::new();
        for (atlas, _, _) in &maps {
            let p = dir.join("..").join("guides").join(atlas).join("legend.json");
            let Ok(bytes) = std::fs::read(&p) else { continue };
            let Ok(j) = serde_json::from_slice::<Value>(&bytes) else { continue };
            let Some(o) = j["islands"].as_object() else { continue };
            let mut v: Vec<(usize, String, usize)> = o
                .iter()
                .filter_map(|(k, e)| Some((k.parse().ok()?, e["label"].as_str().unwrap_or("").to_owned(), e["triangles"].as_u64()? as usize)))
                .collect();
            v.sort();
            if v.iter().enumerate().all(|(i, e)| e.0 == i) {
                islands.push((atlas.clone(), v.into_iter().map(|e| (e.1, e.2)).collect()));
            }
        }
        Ok(Parts { names, maps, islands })
    }

    /// The kit's island labels for an atlas, only when the mesh has the same islands
    /// (same count and the same triangles per island); otherwise `None`.
    pub fn island_labels(&self, atlas: &str, sizes: &[usize]) -> Option<Vec<String>> {
        let parent: String = atlas.splitn(3, '_').take(2).collect::<Vec<_>>().join("_");
        let (_, l) = self.islands.iter().find(|i| i.0 == atlas).or_else(|| self.islands.iter().find(|i| i.0 == parent))?;
        (l.len() == sizes.len() && l.iter().zip(sizes).all(|(a, &b)| a.1 == b)).then(|| l.iter().map(|e| e.0.clone()).collect())
    }

    pub fn name(&self, id: u8) -> &str {
        self.names.iter().find(|n| n.0 == id).map(|n| n.1.as_str()).unwrap_or("unassigned")
    }

    pub fn id_by_name(&self, name: &str) -> Option<u8> {
        self.names.iter().find(|n| n.1 == name).map(|n| n.0)
    }

    /// The part map for an atlas. A derived sheet such as `98_0_hair_corrected` keeps its parent's
    /// UV layout, so it falls back to the parent's map (`98_0`).
    pub fn map_for(&self, atlas: &str) -> Option<&(String, usize, Vec<u8>)> {
        self.maps.iter().find(|m| m.0 == atlas).or_else(|| {
            let parent: String = atlas.splitn(3, '_').take(2).collect::<Vec<_>>().join("_");
            (parent != atlas).then(|| self.maps.iter().find(|m| m.0 == parent)).flatten()
        })
    }

    /// Part ids (not 0) that cover at least one pixel of an atlas's map, ascending.
    pub fn ids_in(&self, atlas: &str) -> Vec<u8> {
        let Some((_, _, ids)) = self.map_for(atlas) else { return vec![] };
        let mut seen = [false; 256];
        for &i in ids {
            seen[i as usize] = true;
        }
        (1..256).filter(|&i| seen[i]).map(|i| i as u8).collect()
    }

    pub fn id_at(&self, atlas: &str, u: f32, v: f32) -> u8 {
        let Some((_, s, ids)) = self.map_for(atlas) else { return 0 };
        let x = ((u * *s as f32) as i64).clamp(0, *s as i64 - 1) as usize;
        let y = ((v * *s as f32) as i64).clamp(0, *s as i64 - 1) as usize;
        ids[y * s + x]
    }

    /// Part of each triangle: sampled at the UV centroid, then at points pulled from each corner toward it.
    pub fn triangle_parts(&self, model: &Model, mesh: usize) -> Vec<u8> {
        let atlas = model.atlas_name(mesh);
        let m = &model.meshes[mesh];
        m.tris
            .iter()
            .map(|t| {
                let c = t.map(|v| m.uv[v as usize]);
                let cu = (c[0][0] + c[1][0] + c[2][0]) / 3.0;
                let cv = (c[0][1] + c[1][1] + c[2][1]) / 3.0;
                let mut id = self.id_at(&atlas, cu, cv);
                for k in 0..3 {
                    if id != 0 {
                        break;
                    }
                    id = self.id_at(&atlas, cu * 0.5 + c[k][0] * 0.5, cv * 0.5 + c[k][1] * 0.5);
                }
                id
            })
            .collect()
    }
}
