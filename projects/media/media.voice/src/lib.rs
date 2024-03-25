//! Formant singing-voice synthesis — the vocal half of the synthesis pair.
//!
//! Voice synthesis is deliberately separate from [`media_synthesis`]'s
//! parameterized sound effects (contract 03 §3.1): a sung word is a syllable
//! sequence ([`spec::Syllable`]) with pitch, duration and onset classes, not
//! an eight-family noise design. This crate owns the minimal honest engine:
//! deterministic formant synthesis with explicit, real control (pitch,
//! duration, vowel quality, onset class, vibrato, breath) and no hidden
//! parameter names pretending to be speaker models or neural vocoders.
//!
//! Contracts owned here:
//! - vowel and onset names are kebab-case ([`Vowel::as_str`] /
//!   [`Vowel::parse`], [`Onset::as_str`] / [`Onset::parse`]) — the same
//!   authoritative resolution rule the sfx families use;
//! - the range contract lives in Rust ([`VoiceSpec::validate`]); bindings
//!   only map errors, never clamp;
//! - the breath random stream is `media.dsp::Rng` with the spec seed
//!   (default 1), the single house audio randomness contract;
//! - same spec + seed reproduces byte-identical samples.
//!
//! File IO lives elsewhere: `media.render` composes this render with
//! `media.codec` behind its `voice-export` feature.

pub mod formant;
pub mod spec;

pub use formant::render;
pub use spec::{Onset, Syllable, Vibrato, Vowel, VoiceSpec};
