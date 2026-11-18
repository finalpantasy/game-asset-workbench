//! Foot contact: does each foot stay put while planted and set down softly, and does the bone at the
//! top of the leg turn without kicks? Measured at the clip's own frames (a player lerps between
//! them, so anything wrong is already wrong there).
use crate::clip::Clip;
use crate::skin::Skin;
use crate::{m, v, Vec3};

/// One leg to watch.
#[derive(Clone, Debug)]
pub struct Leg {
    /// The bone the foot rides on.
    pub foot: usize,
    /// The foot's contact point at rest (often its lowest vertex: see `lowest_point`).
    pub point: Vec3,
    /// The bone at the top of the leg (hip or shoulder) and its parent. The skin where the leg meets
    /// the body follows the top bone, so a kick in its turning shows as a pop at the socket.
    pub top: usize,
    pub top_parent: usize,
}

#[derive(Clone, Debug)]
pub struct LegReport {
    pub foot: usize,
    /// Fastest the foot moves over the ground mid-stance (units/s): sliding.
    pub stance_slide: f32,
    /// Fastest it moves over the ground while within `near` of the floor before touchdown: a skid in.
    pub touchdown_skid: f32,
    /// Speed it drops at over the last frame before it touches.
    pub landing_speed: f32,
    /// Biggest acceleration of the foot from two frames before touchdown to two after (units/s²).
    pub touchdown_jolt: f32,
    /// Biggest change in the top bone's turning rate between neighboring frames, in degrees per
    /// frame per frame: a kick at the socket.
    pub socket_kick: f32,
    /// Cycle phase of that kick.
    pub socket_kick_at: f32,
}

/// The lowest rest vertex (along `up`) bound mostly (over half its weight) to `bone`.
pub fn lowest_point(rest: &[Vec3], skin: &Skin, bone: usize, up: Vec3) -> Option<Vec3> {
    let up = v(&up);
    (0..rest.len())
        .filter(|&i| (0..4).any(|k| skin.joints[i][k] as usize == bone && skin.weights[i][k] > 0.5))
        .min_by(|&a, &b| v(&rest[a]).dot(up).total_cmp(&v(&rest[b]).dot(up)))
        .map(|i| rest[i])
}

/// Plays `clip` with the body carried along `forward` by `travel` per cycle (as a game moves it) and
/// measures each leg. A foot within `planted` of its lowest height (along `up`) counts as down;
/// `near` is the height under which ground speed before touchdown counts as a skid.
pub fn contact(clip: &Clip, legs: &[Leg], forward: Vec3, up: Vec3, travel: f32, planted: f32, near: f32) -> Vec<LegReport> {
    let n = clip.frames.len();
    let dt = clip.period() / n as f32;
    let (fwd, up) = (v(&forward).normalize(), v(&up).normalize());
    legs.iter().map(|leg| {
        let p = v(&leg.point);
        // three cycles end to end so every window wraps cleanly
        let pos: Vec<glam::Vec3> = (0..3 * n + 2)
            .map(|i| m(&clip.frames[i % n][leg.foot]).transform_point3(p) + fwd * travel * i as f32 / n as f32)
            .collect();
        let h = |i: usize| pos[i].dot(up);
        let floor = (0..pos.len()).map(h).fold(f32::MAX, f32::min);
        let down = |i: usize| h(i) - floor < planted;
        let vel = |i: usize| (pos[i + 1] - pos[i]) / dt;
        let flat = |x: glam::Vec3| (x - up * x.dot(up)).length();
        // touchdown: first planted frame after a lifted one, in the middle cycle
        let td = (n..2 * n).find(|&i| down(i) && !down(i - 1)).unwrap_or(n);
        let lo = (td..td + n).find(|&i| !down(i)).unwrap_or(td + n);
        let stance_slide = (td + 2..lo.saturating_sub(2).max(td + 2)).map(|i| flat(vel(i))).fold(0.0, f32::max);
        let touchdown_skid = (td - 6..td).filter(|&i| h(i) - floor < near).map(|i| flat(vel(i))).fold(0.0, f32::max);
        let landing_speed = -vel(td - 1).dot(up);
        let touchdown_jolt = (td - 2..=td + 2).map(|i| ((vel(i) - vel(i - 1)) / dt).length()).fold(0.0, f32::max);
        // socket: the top bone's rotation against its parent, frame to frame
        let rot = |f: usize| {
            let x = m(&clip.frames[f % n][leg.top_parent]).inverse() * m(&clip.frames[f % n][leg.top]);
            x.to_scale_rotation_translation().1
        };
        let step = |f: usize| {
            let (a, b) = (rot(f), rot(f + 1));
            (b * a.inverse()).to_scaled_axis()
        };
        let (k, kick) = (0..n)
            .map(|f| (f, (step(f + 1) - step(f)).length().to_degrees()))
            .fold((0, 0.0f32), |b, x| if x.1 > b.1 { x } else { b });
        LegReport { foot: leg.foot, stance_slide, touchdown_skid, landing_speed, touchdown_jolt, socket_kick: kick, socket_kick_at: (k + 1) as f32 / n as f32 }
    }).collect()
}
