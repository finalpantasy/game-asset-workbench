//! Safe swap on stock Zidane: only the swapped texture's bytes change.
use std::path::Path;
use wb_core::analysis::Parts;
use wb_core::{diff, image_textures, swap, Model, Texture};

const STOCK: &str = r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding\exports\Zidane-idle.glb";
const PARTS: &str = r"E:\SteamLibrary\steamapps\common\FINAL FANTASY IX\Modding\experiments\zidane-uv-kit-claude-v1\parts";

fn png(w: usize, h: usize, rgba: Vec<u8>) -> Vec<u8> {
    let mut out = std::io::Cursor::new(Vec::new());
    image::RgbaImage::from_raw(w as u32, h as u32, rgba).unwrap().write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

#[test]
fn swapping_the_same_bytes_gives_the_same_file() {
    if !Path::new(STOCK).exists() {
        return;
    }
    let bytes = std::fs::read(STOCK).unwrap();
    let m = Model::from_bytes(&bytes).unwrap();
    let same = m.images.iter().find(|i| i.name.starts_with("98_1")).unwrap().bytes.clone();
    let s = swap::swap(&m, "98_1", &same).unwrap();
    assert_eq!(s.shift, 0);
    assert!(s.bytes == bytes, "no-op swap must be byte-identical");
}

#[test]
fn painting_the_belt_band_changes_only_the_belt_band() {
    if !Path::new(STOCK).exists() || !Path::new(PARTS).exists() {
        return;
    }
    let m = Model::open(Path::new(STOCK)).unwrap();
    let parts = Parts::load(Path::new(PARTS)).unwrap();
    let belt = parts.id_by_name("belt-band").unwrap();
    let ii = m.images.iter().position(|i| i.name.starts_with("98_1")).unwrap();
    let t = Texture::decode(&m.images[ii].bytes).unwrap();
    let (_, s, ids) = parts.map_for("98_1").unwrap();
    let mut rgba = t.rgba.clone();
    let mut painted = 0;
    for y in 0..t.h {
        for x in 0..t.w {
            if ids[(y * s / t.h) * s + x * s / t.w] == belt {
                rgba[(y * t.w + x) * 4..(y * t.w + x) * 4 + 3].copy_from_slice(&[200, 30, 30]);
                painted += 1;
            }
        }
    }
    assert!(painted > 100);
    let new_png = png(t.w, t.h, rgba);
    let sw = swap::swap(&m, "98_1.png", &new_png).unwrap();
    assert_eq!(sw.shift.rem_euclid(4), 0);
    assert_eq!(sw.alpha_flips, 0);
    let b = Model::from_bytes(&sw.bytes).unwrap();
    let d = diff::diff(&m, &image_textures(&m), &b, &image_textures(&b), Some(&parts), 0);
    let body = d.textures.iter().find(|x| x.name.starts_with("98_1")).unwrap();
    assert!(body.changed > 0 && body.changed <= painted, "{} changed, {painted} painted", body.changed);
    assert_eq!(body.by_part.iter().map(|p| p.0).collect::<Vec<_>>(), vec![belt]);
    for td in d.textures.iter().filter(|x| !x.name.starts_with("98_1")) {
        assert_eq!(td.changed, 0, "{} changed", td.name);
    }
    assert!(d.meshes.iter().all(|x| x.status() == "same"));
}

#[test]
fn a_different_shape_is_refused() {
    if !Path::new(STOCK).exists() {
        return;
    }
    let m = Model::open(Path::new(STOCK)).unwrap();
    let wide = png(1024, 512, vec![255; 1024 * 512 * 4]);
    let e = swap::swap(&m, "98_1", &wide).err().unwrap();
    assert!(e.contains("stretch"), "{e}");
    assert!(swap::swap(&m, "nope", &wide).is_err());
}
