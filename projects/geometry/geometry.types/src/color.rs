//! Color values and hex parsing.

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
