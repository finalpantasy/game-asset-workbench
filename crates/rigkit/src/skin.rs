//! Linear blend skinning.
use crate::{m, v, Mat4, Vec3};
use rayon::prelude::*;

/// Up to four bones and weights per vertex, in the mesh's vertex order.
#[derive(Clone, Debug, Default)]
pub struct Skin {
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
}

impl Skin {
    pub fn len(&self) -> usize { self.joints.len() }
    pub fn is_empty(&self) -> bool { self.joints.is_empty() }

    /// The blended matrix of one vertex.
    pub(crate) fn blend(&self, mats: &[glam::Mat4], i: usize) -> glam::Mat4 {
        let mut acc = glam::Mat4::ZERO;
        for k in 0..4 {
            let w = self.weights[i][k];
            if w > 0.0 { acc += mats[self.joints[i][k] as usize] * w; }
        }
        acc
    }

    /// Rest positions moved by one pose (one matrix per bone).
    pub fn deform(&self, rest: &[Vec3], pose: &[Mat4]) -> Vec<Vec3> {
        let mats: Vec<glam::Mat4> = pose.iter().map(m).collect();
        (0..rest.len()).into_par_iter().map(|i| self.blend(&mats, i).transform_point3(v(&rest[i])).to_array()).collect()
    }
}
