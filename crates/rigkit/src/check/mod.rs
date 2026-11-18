//! Checks that turn "does this deform look right?" into numbers with limits a test can enforce.
pub mod contact;
pub mod stretch;

pub use contact::{contact, lowest_point, Leg, LegReport};
pub use stretch::{stretch, StretchHit, StretchReport};
