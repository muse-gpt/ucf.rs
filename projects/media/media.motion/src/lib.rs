//! Animation evaluation: stable identity, property tracks, time windows and
//! path following.
//!
//! The core is deliberately pure. Every value is derived from the authored
//! data plus one frame index, so a preview that samples frame 137 and an export
//! that renders frame 137 reach the same result without a second code path.
//!
//! Layering:
//! - [`NodeId`] names an animatable object.
//! - [`Track`] animates one property through [`Key`]s and easing.
//! - [`TimeWindow`] maps project frames to the local frames a track uses.
//! - [`Animated`] is the authoring shape; [`Motion`] is its evaluated value.

mod anim;
mod id;
mod time;
mod track;

pub use anim::{Animated, Follow, Motion};
pub use id::NodeId;
pub use time::{LoopMode, TimeWindow};
pub use track::{Key, Lerp, Track};

/// Interpolation curve for one key segment.
///
/// The camera and the track system share one easing vocabulary so a push and a
/// property ramp cannot disagree on what `in-out-cubic` means.
pub use geometry::Ease;
