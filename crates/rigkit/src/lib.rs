//! rigkit: CPU skinning, skin-weight transfer and motion checks for skinned meshes.
//!
//! Shared by whitetail-rs and the Game Asset Workbench (gaw). Each caller keeps its own file
//! reader; rigkit sees only plain arrays:
//! - points are `[f32; 3]`,
//! - matrices are `[f32; 16]`, column-major (glTF order), one rest->posed matrix per bone,
//! - triangles are a flat index list, three per triangle.
//!
//! glam and rayon are used inside only, so callers are not tied to a glam version.

pub mod check;
pub mod clip;
pub mod skin;
pub mod weights;

/// A point or direction.
pub type Vec3 = [f32; 3];
/// A rest->posed bone matrix, column-major.
pub type Mat4 = [f32; 16];

pub use clip::Clip;
pub use skin::Skin;

pub(crate) fn m(a: &Mat4) -> glam::Mat4 { glam::Mat4::from_cols_array(a) }
#[cfg(test)]
mod tests;

pub(crate) fn v(a: &Vec3) -> glam::Vec3 { glam::Vec3::from_array(*a) }
