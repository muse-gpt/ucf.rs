//! Particles and beams: generative elements that do not come from the frame.
//!
//! Same rule as everywhere else in the family: a dot is a pure function of
//! `(index, seed, time)`. A particle never carries state from the previous
//! frame, which keeps renders reproducible and lets any frame be evaluated on
//! its own — out of order, skipping frames or in parallel.
//!
//! The seed here is the *user's* seed. The grain operator in `media.post`
//! derives its seed from the frame index instead; the two random contracts
//! stay deliberately separate.
//!
//! ```rust
//! use geometry_types::Rgba;
//! use media_image::Frame;
//! use media_particles::Particles;
//!
//! let mut frame = Frame::new(64, 64);
//! Particles::sparks(40, 7, Rgba([255, 180, 90, 255])).apply(&mut frame, 1.5);
//! assert!(frame.pixels.iter().any(|v| *v > 0));
//! ```

pub mod generative;

pub use generative::{Beams, Particles};
