//! A baked animation cycle: one matrix per bone per frame.
use crate::{m, Mat4};

#[derive(Clone, Debug)]
pub struct Clip {
    pub fps: f32,
    /// `frames[f][bone]`, rest->posed. The cycle loops: the frame after the last is frame 0.
    pub frames: Vec<Vec<Mat4>>,
}

impl Clip {
    pub fn bones(&self) -> usize { self.frames.first().map_or(0, |f| f.len()) }

    /// Seconds per cycle.
    pub fn period(&self) -> f32 { self.frames.len() as f32 / self.fps }

    /// The pose at cycle phase `ph` (wraps), each matrix lerped between the two nearest frames.
    pub fn sample(&self, ph: f32) -> Vec<Mat4> {
        let n = self.frames.len();
        let x = ph.rem_euclid(1.0) * n as f32;
        let i = (x.floor() as usize) % n;
        let f = x - x.floor();
        let (a, b) = (&self.frames[i], &self.frames[(i + 1) % n]);
        a.iter().zip(b).map(|(a, b)| (m(a) * (1.0 - f) + m(b) * f).to_cols_array()).collect()
    }
}
