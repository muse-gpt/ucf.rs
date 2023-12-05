//! Color values and hex parsing.

use serde::{Deserialize, Serialize};

/// An RGBA color with 8-bit channels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rgba(pub [u8; 4]);

impl Rgba {
    /// Linear interpolation toward `other`.
    #[must_use]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round().clamp(0.0, 255.0) as u8;
        Self([mix(self.0[0], other.0[0]), mix(self.0[1], other.0[1]), mix(self.0[2], other.0[2]), mix(self.0[3], other.0[3])])
    }

    /// Same color with a replaced alpha channel.
    #[must_use]
    pub fn with_alpha(self, alpha: f32) -> Self {
        Self([self.0[0], self.0[1], self.0[2], (alpha.clamp(0.0, 1.0) * 255.0).round() as u8])
    }
}

/// Parses `#rgb`, `#rrggbb` and `#rrggbbaa` strings.
#[must_use]
pub fn parse(input: &str) -> Option<Rgba> {
    let hex = input.strip_prefix('#')?;
    let nibble = |c: u8| -> Option<u32> {
        match c {
            b'0'..=b'9' => Some((c - b'0') as u32),
            b'a'..=b'f' => Some((c - b'a') as u32 + 10),
            b'A'..=b'F' => Some((c - b'A') as u32 + 10),
            _ => None,
        }
    };
    let byte = |i: usize, j: usize, double: bool| -> Option<u8> {
        let hi = nibble(*hex.as_bytes().get(i)?)?;
        if double {
            let lo = nibble(*hex.as_bytes().get(j)?)?;
            Some((hi * 16 + lo) as u8)
        } else {
            Some((hi * 17) as u8)
        }
    };
    match hex.len() {
        3 => Some(Rgba([byte(0, 0, false)?, byte(1, 1, false)?, byte(2, 2, false)?, 255])),
        6 => Some(Rgba([byte(0, 1, true)?, byte(2, 3, true)?, byte(4, 5, true)?, 255])),
        8 => Some(Rgba([byte(0, 1, true)?, byte(2, 3, true)?, byte(4, 5, true)?, byte(6, 7, true)?])),
        _ => None,
    }
}

/// Parses with a fallback when the string is absent or malformed.
#[must_use]
pub fn parse_or(input: Option<&str>, fallback: Rgba) -> Rgba {
    input.and_then(parse).unwrap_or(fallback)
}

/// A linear-light RGB color with `f64` channels.
///
/// Channels are nominally `0..=1` for display values, but HDR pipelines may
/// carry energies above 1 (e.g. pre-tonemap radiance), so only `to_u8` clamps.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearRgb {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl LinearRgb {
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0 };
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0 };

    #[must_use]
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    #[must_use]
    pub fn scale(self, s: f64) -> Self {
        Self { r: self.r * s, g: self.g * s, b: self.b * s }
    }

    #[must_use]
    pub fn add(self, other: Self) -> Self {
        Self { r: self.r + other.r, g: self.g + other.g, b: self.b + other.b }
    }

    #[must_use]
    pub fn clamp01(self) -> Self {
        Self { r: self.r.clamp(0.0, 1.0), g: self.g.clamp(0.0, 1.0), b: self.b.clamp(0.0, 1.0) }
    }

    /// Encodes to 8-bit RGB after clamping to the display range.
    #[must_use]
    pub fn to_u8(self) -> [u8; 3] {
        let c = self.clamp01();
        [
            (c.r * 255.0).round() as u8,
            (c.g * 255.0).round() as u8,
            (c.b * 255.0).round() as u8,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_rgb_serde_round_trip_keeps_fields() {
        let value = LinearRgb::new(0.25, 1.5, 0.0);
        let json = serde_json::to_string(&value).expect("serialize");
        assert_eq!(json, r#"{"r":0.25,"g":1.5,"b":0.0}"#);
        let back: LinearRgb = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, value);
    }

    #[test]
    fn linear_rgb_to_u8_clamps_before_encoding() {
        assert_eq!(LinearRgb::new(-0.5, 2.0, 0.5).to_u8(), [0, 255, 128]);
        assert_eq!(LinearRgb::BLACK.to_u8(), [0, 0, 0]);
        assert_eq!(LinearRgb::WHITE.to_u8(), [255, 255, 255]);
    }
}
