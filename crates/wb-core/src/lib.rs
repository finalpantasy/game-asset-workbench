//! Core of the FF9 asset workbench. Everything here only reads the input files.
pub mod analysis;
pub mod anim;
pub mod bake;
pub mod diff;
pub mod flatten;
pub mod font;
pub mod glb;
pub mod model;
pub mod pattern;
pub mod patternx;
pub mod rebuild;
pub mod relayout;
pub mod render;
pub mod swap;
pub mod uv;
pub mod unitytex;
pub mod geom;
pub mod shells;
pub mod strap;
pub mod strapcheck;
pub mod boundary;
pub mod sweep;
pub mod hair;
pub mod hair_model;
pub mod hairfiber;
pub mod perceptual;
pub mod golden;
pub mod edit;
pub mod strands;
pub mod defects;
pub mod hairlook;
pub mod fbx;
pub mod bvh;
pub mod weights;
pub mod deform;
pub mod adopt;
pub mod repair;
pub mod blink;

pub use model::Model;

/// One mip level.
pub struct Level {
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,
}

/// RGBA8 texture decoded from an embedded image, with a box-filtered mip chain
/// (the HD Characters plugin gives the game the same kind of chain).
pub struct Texture {
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,
    /// Levels 1.. (half size each); level 0 is `rgba`.
    pub mips: Vec<Level>,
}

impl Texture {
    pub fn decode(bytes: &[u8]) -> Result<Texture, String> {
        let img = image::load_from_memory(bytes).map_err(|e| e.to_string())?.to_rgba8();
        Ok(Texture::new(img.width() as usize, img.height() as usize, img.into_raw()))
    }

    pub fn new(w: usize, h: usize, rgba: Vec<u8>) -> Texture {
        let mut mips: Vec<Level> = Vec::new();
        let (mut pw, mut ph) = (w, h);
        while pw > 1 || ph > 1 {
            let src = mips.last().map(|l| l.rgba.as_slice()).unwrap_or(&rgba);
            let (nw, nh) = ((pw / 2).max(1), (ph / 2).max(1));
            let mut out = vec![0u8; nw * nh * 4];
            for y in 0..nh {
                for x in 0..nw {
                    for c in 0..4 {
                        let mut s = 0u32;
                        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                            let sx = (2 * x + dx).min(pw - 1);
                            let sy = (2 * y + dy).min(ph - 1);
                            s += src[(sy * pw + sx) * 4 + c] as u32;
                        }
                        out[(y * nw + x) * 4 + c] = ((s + 2) / 4) as u8;
                    }
                }
            }
            mips.push(Level { w: nw, h: nh, rgba: out });
            (pw, ph) = (nw, nh);
        }
        Texture { w, h, rgba, mips }
    }

    pub fn level(&self, i: usize) -> (usize, usize, &[u8]) {
        if i == 0 {
            (self.w, self.h, &self.rgba)
        } else {
            let l = &self.mips[(i - 1).min(self.mips.len().saturating_sub(1))];
            (l.w, l.h, &l.rgba)
        }
    }

    pub fn levels(&self) -> usize {
        1 + self.mips.len()
    }
}

/// Textures per image of a model (None where an image fails to decode).
pub fn image_textures(model: &Model) -> Vec<Option<Texture>> {
    model.images.iter().map(|i| Texture::decode(&i.bytes).ok()).collect()
}

/// Per-mesh textures for a model (None where the mesh has no image or it fails to decode).
pub fn model_textures(model: &Model) -> Vec<Option<Texture>> {
    model.meshes.iter().map(|m| m.image.and_then(|i| Texture::decode(&model.images[i].bytes).ok())).collect()
}
