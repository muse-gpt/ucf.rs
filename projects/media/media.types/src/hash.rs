//! Deterministic integer hashing.

/// Cheap 32-bit integer hash for deterministic randomness.
///
/// Frames are a pure function of time, so grain and dither must never touch a
/// global RNG: the same frame index has to produce the same pixels on every
/// render.
#[must_use]
pub fn hash(mut x: u32) -> u32 {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x.wrapping_mul(0x9e37_79b9)
}
