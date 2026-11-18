use crate::check::{contact, lowest_point, stretch, Leg};
use crate::weights::Source;
use crate::{Clip, Mat4, Skin, Vec3};
use glam::{Mat4 as M, Vec3 as V};

/// A strip of quads along +X from 0 to 2, bone 0 owns x < 1, bone 1 owns x >= 1 (hard weights).
fn strip() -> (Vec<Vec3>, Vec<u32>, Skin) {
    let mut p = Vec::new();
    for i in 0..=8 {
        let x = i as f32 * 0.25;
        p.push([x, 0.0, 0.0]);
        p.push([x, 0.2, 0.0]);
    }
    let mut idx = Vec::new();
    for i in 0..8u32 {
        let (a, b, c, d) = (2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3);
        idx.extend([a, c, b, b, c, d]);
    }
    let joints = p.iter().map(|q| [if q[0] < 1.0 { 0 } else { 1 }, 0, 0, 0]).collect();
    let weights = p.iter().map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
    (p, idx, Skin { joints, weights })
}

fn bend(deg: f32) -> Vec<Mat4> {
    let pivot = V::new(1.0, 0.0, 0.0);
    let r = M::from_translation(pivot) * M::from_rotation_z(deg.to_radians()) * M::from_translation(-pivot);
    vec![M::IDENTITY.to_cols_array(), r.to_cols_array()]
}

#[test]
fn rigid_motion_does_not_stretch() {
    let (p, idx, skin) = strip();
    let whole = M::from_rotation_y(1.0) * M::from_translation(V::new(3.0, 1.0, 0.0));
    let rep = stretch(&p, &idx, &skin, &[vec![whole.to_cols_array(); 2]], 0.01, 3);
    assert!((rep.worst_edge - 1.0).abs() < 1e-4 && rep.flipped == 0, "{rep:?}");
}

#[test]
fn a_hard_bend_tears_the_seam() {
    let (p, idx, skin) = strip();
    let small = stretch(&p, &idx, &skin, &[bend(10.0)], 0.01, 3);
    let big = stretch(&p, &idx, &skin, &[bend(60.0)], 0.01, 3);
    // the seam quad (x 0.75..1.0) shears open; a bigger bend tears it further
    assert!(big.worst_edge > small.worst_edge && big.worst_edge > 1.2, "small {small:?} big {big:?}");
    let seam = big.hits[0].tri / 2;
    assert_eq!(seam, 3, "worst quad should be the one spanning the bone seam: {:?}", big.hits[0]);
}

#[test]
fn transfer_copies_the_nearest_bone() {
    let (p, _, skin) = strip();
    let src = Source::new(&p, skin, 2, 0.3);
    let got = src.transfer(&[[0.1, 0.1, 0.01], [1.9, 0.1, -0.01]]);
    assert_eq!(got.joints[0][0], 0);
    assert_eq!(got.joints[1][0], 1);
    for w in &got.weights { assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5); }
}

#[test]
fn smoothing_spreads_the_seam() {
    let (p, idx, skin) = strip();
    let hard = stretch(&p, &idx, &skin, &[bend(60.0)], 0.01, 3);
    let mut src = Source::new(&p, skin, 2, 0.3);
    src.smooth(&[0, 1], 0.3, 2);
    let soft = stretch(&p, &idx, src.skin(), &[bend(60.0)], 0.01, 3);
    // the hard seam folds one triangle inside out; spread weights bend it without folding
    assert!(soft.flipped < hard.flipped && soft.worst_area < hard.worst_area, "hard {hard:?} soft {soft:?}");
    // far ends stay with their own bone, weights still sum to one
    let w = |i: usize, b: u16| (0..4).filter(|&j| src.skin().joints[i][j] == b).map(|j| src.skin().weights[i][j]).sum::<f32>();
    assert!(w(0, 0) > 0.99 && w(17, 1) > 0.99);
    for x in &src.skin().weights { assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-5); }
}

#[test]
fn fade_hands_weight_to_the_parent_past_the_joint() {
    let (p, _, skin) = strip();
    let mut src = Source::new(&p, skin, 2, 0.3);
    // bone 1 must not reach past x = 1.5 (toward -X): fade it out over 0.25
    let n = src.fade_past(1, [1.5, 0.0, 0.0], [-1.0, 0.0, 0.0], 0.25, 0);
    assert!(n > 0);
    let w1 = |i: usize| (0..4).filter(|&j| src.skin().joints[i][j] == 1).map(|j| src.skin().weights[i][j]).sum::<f32>();
    assert!(w1(8) < 1e-5, "x 1.0 is fully past the band: {}", w1(8)); // vertex at x = 1.0
    assert!(w1(12) > 0.99, "x 1.5 is on the plane"); // vertex at x = 1.5
}

#[test]
fn confine_clears_stray_bones() {
    let (p, _, mut skin) = strip();
    // a stray 10% of bone 2 on a vertex owned by bone 1
    skin.joints[16] = [1, 2, 0, 0];
    skin.weights[16] = [0.9, 0.1, 0.0, 0.0];
    let mut src = Source::new(&p, skin, 3, 0.3);
    assert_eq!(src.confine(1, &[0]), 1);
    assert_eq!((src.skin().joints[16][0], src.skin().weights[16][0]), (1, 1.0));
}

#[test]
fn a_planted_foot_does_not_slide() {
    // one foot bone: planted for half the cycle (moving back at body speed), lifted and carried
    // forward for the other half; the body travels 1.0 per cycle along +X
    let n = 40;
    let frames: Vec<Vec<Mat4>> = (0..n).map(|f| {
        let t = f as f32 / n as f32;
        let (x, y) = if t < 0.5 {
            (0.25 - t, 0.0)
        } else {
            let u = (t - 0.5) / 0.5;
            let e = u * u * u * (u * (6.0 * u - 15.0) + 10.0);
            (-0.25 + e * 1.0 - 0.5 * u, 0.1 * (std::f32::consts::PI * u).sin().powi(2))
        };
        vec![M::IDENTITY.to_cols_array(), M::from_translation(V::new(x, y, 0.0)).to_cols_array()]
    }).collect();
    let clip = Clip { fps: 40.0, frames };
    let skin = Skin { joints: vec![[1, 0, 0, 0]], weights: vec![[1.0, 0.0, 0.0, 0.0]] };
    let point = lowest_point(&[[0.0, 0.0, 0.0]], &skin, 1, [0.0, 1.0, 0.0]).unwrap();
    let rep = contact(&clip, &[Leg { foot: 1, point, top: 1, top_parent: 0 }], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0, 0.003, 0.01);
    assert!(rep[0].stance_slide < 1e-3, "{rep:?}");
}
